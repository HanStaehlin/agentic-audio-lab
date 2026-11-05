"""Tools the agent can call, with their safeguards.

Safeguards:
- only whitelisted config keys within fixed ranges can be changed;
- one run at a time (Colab has one GPU) and a hard cap on the number of runs;
- every launch needs a human "y" unless the session was started with --auto-approve.
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS, REPORTS = ROOT / "runs", ROOT / "reports"
BASE_CONFIG = ROOT / "configs" / "baseline.json"

ALLOWED = {  # key: (type, min, max) or (type, choices)
    "lr": (float, 1e-5, 1e-2),
    "hidden": (int, 32, 512),
    "layers": (int, 1, 4),
    "batch_size": (int, 4, 64),
    "steps": (int, 100, 20000),
    "crop_seconds": (float, 0.5, 4.0),
    "n_fft": (int, [256, 512, 1024]),
    "hop": (int, [64, 128, 256]),
    "grad_clip": (float, 0.1, 100.0),
}


class Session:
    def __init__(self, max_runs, auto_approve=False):
        self.max_runs, self.auto_approve, self.launched = max_runs, auto_approve, 0
        self.procs = {}  # run_id -> Popen, for runs started in this session

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _read(path):
        return json.loads(path.read_text()) if path.exists() else None

    def _status(self, run_dir):
        status = self._read(run_dir / "status.json") or {"state": "starting"}
        if status["state"] in ("starting", "running", "evaluating") and not self._alive(run_dir):
            status = {**status, "state": "crashed", "error": "process exited without finishing"}
        return status

    def _alive(self, run_dir):
        if run_dir.name in self.procs:  # our child: poll() also reaps it, so no zombies
            return self.procs[run_dir.name].poll() is None
        pid_file = run_dir / "pid"
        if not pid_file.exists():
            return True  # e.g. a baseline started by hand; trust status.json
        try:
            os.kill(int(pid_file.read_text()), 0)
            return True
        except ProcessLookupError:
            return False

    def _active_run(self):
        for d in sorted(RUNS.glob("*")):
            if self._status(d)["state"] in ("starting", "running", "evaluating"):
                return d.name
        return None

    @staticmethod
    def _validate(overrides):
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
        return clean

    # ---- tools ---------------------------------------------------------
    def list_runs(self):
        rows = []
        for d in sorted(RUNS.glob("*")):
            m = self._read(d / "metrics.json")
            rows.append({
                "run_id": d.name,
                "state": self._status(d)["state"],
                "overrides": (self._read(d / "overrides.json") or {}),
                "pesq": m and round(m["enhanced"]["pesq"], 3),
                "si_sdr": m and round(m["enhanced"]["si_sdr"], 2),
            })
        return {"runs": rows, "runs_left_in_budget": self.max_runs - self.launched}

    def launch_run(self, name, overrides, reason):
        if self.launched >= self.max_runs:
            raise RuntimeError(f"Run budget of {self.max_runs} is used up. Write the report instead.")
        if active := self._active_run():
            raise RuntimeError(f"Run '{active}' is still active; wait for it first (one GPU).")
        overrides = self._validate(overrides)
        run_id = f"{datetime.now():%m%d-%H%M%S}-{''.join(c if c.isalnum() else '-' for c in name)[:40]}"
        if not self.auto_approve:
            print(f"\n[approval] launch {run_id}\n  overrides: {overrides}\n  reason: {reason}")
            if input("  approve? [y/N] ").strip().lower() != "y":
                return {"launched": False, "message": "The human declined this run. Ask why or propose something else."}
        run_dir = RUNS / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "overrides.json").write_text(json.dumps(overrides, indent=2))
        (run_dir / "reason.txt").write_text(reason)
        cmd = [sys.executable, "-m", "audiolab.train", "--config", str(BASE_CONFIG), "--run-dir", str(run_dir),
               "--set", *[f"{k}={json.dumps(v)}" for k, v in overrides.items()]]
        out = open(run_dir / "stdout.log", "w")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        (run_dir / "pid").write_text(str(proc.pid))
        self.procs[run_id] = proc
        self.launched += 1
        return {"launched": True, "run_id": run_id, "runs_left_in_budget": self.max_runs - self.launched}

    def get_run_status(self, run_id, tail_lines=20):
        run_dir = RUNS / run_id
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
            state = self._status(RUNS / run_id)["state"]
            if state not in ("starting", "running", "evaluating"):
                break
            time.sleep(20)
        return self.get_run_status(run_id, tail_lines=10)

    def compare_runs(self, run_ids, baseline_id):
        base = self._read(RUNS / baseline_id / "metrics.json")
        if base is None:
            raise ValueError(f"Baseline '{baseline_id}' has no metrics yet")
        rows = []
        for rid in run_ids:
            m = self._read(RUNS / rid / "metrics.json")
            if m is None:
                rows.append({"run_id": rid, "state": self._status(RUNS / rid)["state"]})
                continue
            row = {"run_id": rid, "overrides": self._read(RUNS / rid / "overrides.json")}
            for k in ("pesq", "stoi", "si_sdr"):
                row[k] = round(m["enhanced"][k], 3)
                row[f"delta_{k}"] = round(m["enhanced"][k] - base["enhanced"][k], 3)
            row["ms_per_hop"] = round(m["latency"]["ms_per_hop"], 3)
            row["rtf"] = round(m["latency"]["rtf"], 3)
            rows.append(row)
        return {"baseline": baseline_id, "noisy_input": base["noisy_input"], "rows": rows}

    def write_report(self, title, markdown):
        REPORTS.mkdir(exist_ok=True)
        path = REPORTS / f"{datetime.now():%Y%m%d-%H%M%S}.md"
        path.write_text(f"# {title}\n\n{markdown}\n")
        return {"saved": str(path.relative_to(ROOT))}


SCHEMAS = [
    {
        "name": "list_runs",
        "description": "List all experiment runs with state, config overrides and headline metrics, plus how many launches are left in the budget.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "launch_run",
        "description": "Start one training + evaluation run in the background with the baseline config plus overrides. "
                       "Only one run can be active. A human may decline the launch. "
                       f"Changeable keys and ranges: {json.dumps({k: (v[1] if isinstance(v[1], list) else [v[1], v[2]]) for k, v in ALLOWED.items()})}.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "short descriptive name, e.g. 'hidden384'"},
                "overrides": {"type": "object", "description": "config keys to change, e.g. {\"hidden\": 384}"},
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
        "description": "Compare finished runs against a baseline run: PESQ, STOI, SI-SDR with deltas, and streaming latency.",
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
        "name": "write_report",
        "description": "Save the final Markdown report for the human: goal, what was tried and why, a results table, failures and their diagnosis, and a recommendation.",
        "input_schema": {
            "type": "object",
            "properties": {"title": {"type": "string"}, "markdown": {"type": "string"}},
            "required": ["title", "markdown"],
        },
    },
]
