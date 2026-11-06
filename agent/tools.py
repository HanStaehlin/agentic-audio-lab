"""Tools the agent can call, with their safeguards.

Safeguards:
- only whitelisted config keys within fixed ranges can be changed, and the resulting STFT
  setup must be valid (hop at most half the window);
- a cap on parallel runs (1 locally, a few on Modal) and a hard cap on the number of launches;
- every launch needs a human "y" unless the session was started with --auto-approve.
"""
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STALE_SECONDS = 15 * 60
ACTIVE = ("starting", "running", "evaluating")

ALLOWED = {  # key: (type, min, max) or (type, choices)
    "lr": (float, 1e-5, 1e-2),
    "hidden": (int, 32, 512),
    "layers": (int, 1, 4),
    "batch_size": (int, 4, 64),
    "steps": (int, 100, 30000),
    "crop_seconds": (float, 0.5, 4.0),
    "n_fft": (int, [128, 256, 512, 1024]),  # window = algorithmic latency: 8 / 16 / 32 / 64 ms at 16 kHz
    "hop": (int, [32, 64, 128, 256]),
    "grad_clip": (float, 0.1, 100.0),
    "seed": (int, 0, 99),
    # training objective and schedule (do not change inference cost)
    "spec_weight": (float, 0.0, 1000.0),  # weight of the compressed multi-resolution spectral loss
    "feature": (str, ["log1p", "pow0.3"]),  # model input compression
    "lr_schedule": (str, ["constant", "cosine"]),
}


class Session:
    def __init__(self, max_runs, auto_approve=False, backend="local", base_config=None,
                 study="", max_parallel=1):
        self.max_runs, self.auto_approve, self.launched = max_runs, auto_approve, 0
        # local: train on this machine; modal: same files, but the run executes on a Modal GPU
        self.train_module = {"local": "audiolab.train", "modal": "audiolab.remote"}[backend]
        self.base_config = Path(base_config or ROOT / "configs" / "baseline.json")
        self.runs = ROOT / "runs" / study if study else ROOT / "runs"
        self.reports = ROOT / "reports" / study if study else ROOT / "reports"
        self.max_parallel = max_parallel
        self.procs = {}  # run_id -> Popen, for runs started in this session

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _read(path):
        return json.loads(path.read_text()) if path.exists() else None

    def _status(self, run_dir):
        status = self._read(run_dir / "status.json") or {"state": "starting"}
        if status["state"] in ACTIVE and not self._alive(run_dir):
            status = {**status, "state": "crashed", "error": "process exited without finishing"}
        return status

    def _alive(self, run_dir):
        if run_dir.name in self.procs:  # our child: poll() also reaps it, so no zombies
            return self.procs[run_dir.name].poll() is None
        pid_file = run_dir / "pid"
        if not pid_file.exists():
            # e.g. a baseline started by hand: trust status.json unless it has gone stale
            # (training rewrites it every log interval; evaluation takes a few minutes)
            status_file = run_dir / "status.json"
            return not status_file.exists() or time.time() - status_file.stat().st_mtime < STALE_SECONDS
        try:
            os.kill(int(pid_file.read_text()), 0)
            return True
        except ProcessLookupError:
            return False

    def _run_dirs(self):
        return sorted(d for d in self.runs.glob("*") if d.is_dir())

    def _active_runs(self):
        return [d.name for d in self._run_dirs() if self._status(d)["state"] in ACTIVE]

    def _config(self, run_id):
        return self._read(self.runs / run_id / "config.json") or {}

    def _validate(self, overrides):
        clean = {}
        for key, value in overrides.items():
            if key not in ALLOWED:
                raise ValueError(f"'{key}' may not be changed. Allowed: {sorted(ALLOWED)}")
            spec = ALLOWED[key]
            value = spec[0](value)
            if isinstance(spec[1], list):
                if value not in spec[1]:
                    raise ValueError(f"{key}={value} not in {spec[1]}")
            elif not spec[1] <= value <= spec[2]:
                raise ValueError(f"{key}={value} outside [{spec[1]}, {spec[2]}]")
            clean[key] = value
        cfg = {**json.loads(self.base_config.read_text()), **clean}
        if cfg["hop"] > cfg["n_fft"] // 2:
            raise ValueError(f"hop={cfg['hop']} must be at most n_fft/2={cfg['n_fft'] // 2} "
                             "(Hann overlap-add needs at least 50% overlap); set hop too")
        return clean

    def _row(self, run_id):
        m = self._read(self.runs / run_id / "metrics.json")
        cfg = self._config(run_id)
        row = {"run_id": run_id, "overrides": self._read(self.runs / run_id / "overrides.json") or {},
               "seed": cfg.get("seed")}
        if m is None:
            return {**row, "state": self._status(self.runs / run_id)["state"]}
        for k in ("pesq", "stoi", "si_sdr"):
            row[k] = round(m["enhanced"][k], 3)
        row["algorithmic_latency_ms"] = round(m["latency"]["algorithmic_latency_ms"], 1)
        row["ms_per_hop"] = round(m["latency"]["ms_per_hop"], 3)
        row["rtf"] = round(m["latency"]["rtf"], 3)
        row["params"] = m["latency"]["params"]
        return row

    # ---- tools ---------------------------------------------------------
    def list_runs(self):
        rows = []
        for d in self._run_dirs():
            m = self._read(d / "metrics.json")
            rows.append({
                "run_id": d.name,
                "state": self._status(d)["state"],
                "overrides": (self._read(d / "overrides.json") or {}),
                "pesq": m and round(m["enhanced"]["pesq"], 3),
                "latency_ms": m and round(m["latency"]["algorithmic_latency_ms"], 1),
                "rtf": m and round(m["latency"]["rtf"], 3),
            })
        return {"runs": rows, "runs_left_in_budget": self.max_runs - self.launched,
                "parallel_slots_free": self.max_parallel - len(self._active_runs()),
                "base_config": json.loads(self.base_config.read_text())}

    def launch_run(self, name, overrides, reason):
        if self.launched >= self.max_runs:
            raise RuntimeError(f"Run budget of {self.max_runs} is used up. Write the report instead.")
        active = self._active_runs()
        if len(active) >= self.max_parallel:
            raise RuntimeError(f"{len(active)} runs active ({', '.join(active)}), the limit is "
                               f"{self.max_parallel}; wait for one first.")
        overrides = self._validate(overrides)
        run_id = f"{datetime.now():%m%d-%H%M%S}-{''.join(c if c.isalnum() else '-' for c in name)[:40]}"
        if not self.auto_approve:
            print(f"\n[approval] launch {run_id}\n  overrides: {overrides}\n  reason: {reason}")
            if input("  approve? [y/N] ").strip().lower() != "y":
                return {"launched": False, "message": "The human declined this run. Ask why or propose something else."}
        run_dir = self.runs / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "overrides.json").write_text(json.dumps(overrides, indent=2))
        (run_dir / "reason.txt").write_text(reason)
        cmd = [sys.executable, "-m", self.train_module, "--config", str(self.base_config), "--run-dir", str(run_dir),
               "--set", *[f"{k}={json.dumps(v)}" for k, v in overrides.items()]]
        out = open(run_dir / "stdout.log", "w")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        (run_dir / "pid").write_text(str(proc.pid))
        self.procs[run_id] = proc
        self.launched += 1
        time.sleep(2)  # space out launches so run ids and Modal app starts don't collide
        return {"launched": True, "run_id": run_id, "runs_left_in_budget": self.max_runs - self.launched}

    def get_run_status(self, run_id, tail_lines=20):
        run_dir = self.runs / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"No run '{run_id}'")
        status = self._status(run_dir)
        logs = {}
        for name in ("train.log", "stdout.log"):
            p = run_dir / name
            if p.exists():
                logs[name] = p.read_text().splitlines()[-tail_lines:]
        return {"status": status, "logs": logs, "metrics": self._read(run_dir / "metrics.json")}

    def wait_for_run(self, run_id, timeout_minutes=30):
        deadline = time.time() + 60 * min(timeout_minutes, 60)
        while time.time() < deadline:
            if self._status(self.runs / run_id)["state"] not in ACTIVE:
                break
            time.sleep(20)
        return self.get_run_status(run_id, tail_lines=10)

    def compare_runs(self, run_ids, baseline_id):
        """Rows per run, plus groups: runs with the same config except the seed, as mean and spread."""
        base = self._read(self.runs / baseline_id / "metrics.json")
        if base is None:
            raise ValueError(f"Baseline '{baseline_id}' has no metrics yet")
        rows = [self._row(r) for r in run_ids]
        for row in rows:
            if "pesq" in row:
                for k in ("pesq", "stoi", "si_sdr"):
                    row[f"delta_{k}"] = round(row[k] - base["enhanced"][k], 3)
        groups = {}
        for row in rows:
            if "pesq" not in row:
                continue
            key = json.dumps({k: v for k, v in sorted(row["overrides"].items()) if k != "seed"})
            groups.setdefault(key, []).append(row)
        summary = []
        for key, members in groups.items():
            g = {"config": json.loads(key), "n_seeds": len(members),
                 "algorithmic_latency_ms": members[0]["algorithmic_latency_ms"],
                 "rtf": round(max(r["rtf"] for r in members), 3)}
            for k in ("pesq", "stoi", "si_sdr"):
                vals = [r[k] for r in members]
                g[f"{k}_mean"] = round(statistics.mean(vals), 3)
                g[f"{k}_std"] = round(statistics.stdev(vals), 3) if len(vals) > 1 else None
            summary.append(g)
        return {"baseline": baseline_id, "baseline_pesq": round(base["enhanced"]["pesq"], 3),
                "noisy_input": base["noisy_input"], "rows": rows, "groups": summary}

    def plot_tradeoff(self, run_ids, title="Speech quality vs. algorithmic latency"):
        """Plot PESQ (mean and std over seeds) against algorithmic latency; one point per config.

        Filled points meet the compute budget (rtf < 0.1), crosses do not. The line connects the
        best feasible config of each latency budget.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ref = next((r for r in run_ids if (self.runs / r / "metrics.json").exists()), None)
        if ref is None:
            raise ValueError("None of these runs has metrics yet")
        groups = self.compare_runs(run_ids, ref)["groups"]
        base = json.loads(self.base_config.read_text())

        def label(g):
            c = {**base, **g["config"]}
            return f"hop {c['hop'] / c['sample_rate'] * 1e3:g} ms, {c['layers']}x{c['hidden']}"

        by_latency = {}
        for g in groups:
            by_latency.setdefault(g["algorithmic_latency_ms"], []).append(g)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        best = []  # (x, group) of the best feasible config per budget
        for lat, members in sorted(by_latency.items()):
            members.sort(key=lambda g: g["pesq_mean"])
            top = max((g for g in members if g["rtf"] < 0.1), key=lambda g: g["pesq_mean"], default=None)
            for i, g in enumerate(members):
                x = lat * 2 ** ((i - (len(members) - 1) / 2) * 0.09)  # spread configs of one budget sideways
                ok = g["rtf"] < 0.1
                if g is top:
                    best.append((x, g))
                ax.errorbar(x, g["pesq_mean"], yerr=g["pesq_std"] or 0, fmt="o" if ok else "x",
                            color="tab:blue" if ok else "tab:red", ms=6, capsize=3, zorder=3)
                seeds = f", {g['n_seeds']} seeds" if g["n_seeds"] > 1 else ""
                note = "" if ok else f" (rtf {g['rtf']:.3f})"
                if g is top:  # the budget's answer: label above the error bar, in bold
                    ax.annotate(label(g) + seeds, (x, g["pesq_mean"] + (g["pesq_std"] or 0)), fontsize=7.5,
                                xytext=(0, 5), textcoords="offset points", ha="center", weight="bold")
                    continue
                left = i % 2 == 0  # alternate sides so neighbours do not overlap
                ax.annotate(label(g) + seeds + note, (x, g["pesq_mean"]), fontsize=7,
                            xytext=(-7 if left else 7, -3), textcoords="offset points",
                            ha="right" if left else "left", color="black" if ok else "tab:red")
        if best:
            ax.plot([x for x, _ in best], [g["pesq_mean"] for _, g in best],
                    "--", color="tab:orange", lw=1.5, zorder=2, label="best with rtf < 0.1")
        ax.plot([], [], "x", color="tab:red", label="over compute budget (rtf >= 0.1)")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xscale("log", base=2)  # 8, 16, 32 ms evenly spaced
        ax.set_xticks(sorted(by_latency), [f"{v:g}" for v in sorted(by_latency)])
        ax.minorticks_off()
        ax.set_xlim(min(by_latency) / 1.6, max(by_latency) * 1.6)
        ax.set_xlabel("algorithmic latency = STFT window (ms)")
        ax.set_ylabel("PESQ (wideband), mean ± std over seeds")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        self.reports.mkdir(parents=True, exist_ok=True)
        path = self.reports / "tradeoff.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return {"saved": str(path.relative_to(ROOT)), "groups_plotted": len(groups),
                "best_per_budget": [{"latency_ms": g["algorithmic_latency_ms"], "pesq_mean": g["pesq_mean"],
                                     "pesq_std": g["pesq_std"], "n_seeds": g["n_seeds"], "config": g["config"]}
                                    for _, g in best]}

    def write_report(self, title, markdown):
        self.reports.mkdir(parents=True, exist_ok=True)
        path = self.reports / f"{datetime.now():%Y%m%d-%H%M%S}.md"
        path.write_text(f"# {title}\n\n{markdown}\n")
        return {"saved": str(path.relative_to(ROOT))}


SCHEMAS = [
    {
        "name": "list_runs",
        "description": "List all runs of this study with state, overrides, PESQ, latency and rtf, plus the base config, "
                       "launches left in the budget and free parallel slots.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "launch_run",
        "description": "Start one training + evaluation run in the background with the base config plus overrides. "
                       "Several runs can be active up to the parallel limit. A human may decline the launch. "
                       "hop must be at most n_fft/2. "
                       f"Changeable keys and ranges: {json.dumps({k: (v[1] if isinstance(v[1], list) else [v[1], v[2]]) for k, v in ALLOWED.items()})}.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "short descriptive name, e.g. 'w256-h64-s1'"},
                "overrides": {"type": "object", "description": "config keys to change, e.g. {\"n_fft\": 256, \"hop\": 64}"},
                "reason": {"type": "string", "description": "the hypothesis this run tests, shown to the human"},
            },
            "required": ["name", "overrides", "reason"],
        },
    },
    {
        "name": "get_run_status",
        "description": "State, last log lines and metrics (if finished) of one run. Use it to diagnose failed or crashed runs.",
        "input_schema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}, "tail_lines": {"type": "integer"}},
            "required": ["run_id"],
        },
    },
    {
        "name": "wait_for_run",
        "description": "Block until a run finishes, fails or the timeout passes (max 60 min), then return its status. Prefer this over polling.",
        "input_schema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}, "timeout_minutes": {"type": "integer"}},
            "required": ["run_id"],
        },
    },
    {
        "name": "compare_runs",
        "description": "Compare finished runs against a baseline run: PESQ, STOI, SI-SDR with deltas, algorithmic latency, "
                       "rtf and params per run, and 'groups' that pool runs differing only in seed (mean and std).",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_ids": {"type": "array", "items": {"type": "string"}},
                "baseline_id": {"type": "string"},
            },
            "required": ["run_ids", "baseline_id"],
        },
    },
    {
        "name": "plot_tradeoff",
        "description": "Save a PNG of PESQ (mean and std over seeds) against algorithmic latency for the given runs, "
                       "with the best-so-far front among configs with rtf < 0.1. Returns the path and the front.",
        "input_schema": {
            "type": "object",
            "properties": {"run_ids": {"type": "array", "items": {"type": "string"}}, "title": {"type": "string"}},
            "required": ["run_ids"],
        },
    },
    {
        "name": "write_report",
        "description": "Save the final Markdown report for the human: question, what was tried and why, a results table, "
                       "the trade-off plot (link tradeoff.png), failures and their diagnosis, and the answer.",
        "input_schema": {
            "type": "object",
            "properties": {"title": {"type": "string"}, "markdown": {"type": "string"}},
            "required": ["title", "markdown"],
        },
    },
]
