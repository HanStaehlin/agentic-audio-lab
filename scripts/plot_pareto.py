"""Pareto plots for the latency study, from results/latency/runs.csv.

    python scripts/plot_pareto.py

Left: PESQ against algorithmic latency, among configs that meet the compute budget (rtf < 0.1).
Right: PESQ against compute (rtf), all configs, coloured by latency.
A config is on the front if no other config is at least as good on both axes and better on one.
"""
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "latency" / "runs.csv"
OUT = ROOT / "results" / "latency" / "pareto.png"
RTF_BUDGET = 0.1
NOISY_PESQ = 1.97
COLORS = {8.0: "#d62728", 16.0: "#1f77b4", 32.0: "#7f7f7f"}


def load_configs():
    groups = defaultdict(list)
    for r in csv.DictReader(RUNS.open()):
        if r["state"] != "done":
            continue
        key = (float(r["latency_ms"]), int(r["hop"]), int(r["hidden"]), int(r["layers"]))
        groups[key].append(r)
    configs = []
    for (lat, hop, hidden, layers), rows in groups.items():
        pesq = [float(r["pesq"]) for r in rows]
        configs.append({
            "latency": lat, "hop_ms": hop / 16, "hidden": hidden, "layers": layers,
            "pesq": mean(pesq), "std": stdev(pesq) if len(pesq) > 1 else 0.0, "seeds": len(pesq),
            "rtf": mean(float(r["rtf"]) for r in rows),
        })
    return configs


def front(points, x, y):
    """Configs not dominated when lower x and higher y are better, sorted by x."""
    keep = [p for p in points
            if not any(q is not p and q[x] <= p[x] and q[y] >= p[y] and (q[x] < p[x] or q[y] > p[y])
                       for q in points)]
    return sorted(keep, key=lambda p: p[x])


def staircase(ax, pts, x, y, x_max, color):
    xs, ys = [], []
    for i, p in enumerate(pts):
        xs += [p[x]] if i == 0 else [p[x], p[x]]
        ys += [p[y]] if i == 0 else [pts[i - 1][y], p[y]]
    xs.append(x_max)
    ys.append(pts[-1][y])
    ax.plot(xs, ys, color=color, lw=2, zorder=2, solid_joinstyle="miter")
    ax.fill_between(xs, ys, 0, color=color, alpha=0.05, zorder=1)


def name(c):
    return f"{c['layers']}×{c['hidden']} GRU, hop {c['hop_ms']:g} ms"


def main():
    configs = load_configs()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"wspace": 0.28})
    ylo, yhi = 2.30, 2.545

    # ---- left: quality vs latency, compute budget met ----
    ok = [c for c in configs if c["rtf"] < RTF_BUDGET]
    pf = front(ok, "latency", "pesq")
    staircase(a, pf, "latency", "pesq", 40, "#ff7f0e")
    for c in configs:
        on = c in pf
        if c["rtf"] >= RTF_BUDGET:
            a.scatter(c["latency"], c["pesq"], marker="x", s=36, color="#bbbbbb", zorder=3)
            continue
        if on:
            a.errorbar(c["latency"], c["pesq"], yerr=c["std"], fmt="o", ms=9, color="#ff7f0e",
                       mec="black", capsize=4, zorder=4)
        else:
            a.scatter(c["latency"], c["pesq"], s=28, color="#9e9e9e", zorder=3)
    for c in pf:
        a.annotate(f"{c['latency']:g} ms: PESQ {c['pesq']:.2f} ± {c['std']:.2f}\n{name(c)}, {c['seeds']} seeds",
                   (c["latency"], c["pesq"]), xytext=(10, 10), textcoords="offset points", fontsize=8,
                   weight="bold")
    ref = next(c for c in ok if c["latency"] == 32 and c["hop_ms"] == 8 and c["hidden"] == 256)
    a.annotate(f"32 ms default: {ref['pesq']:.2f}\n(dominated by 16 ms)", (ref["latency"], ref["pesq"]),
               xytext=(-10, -44), textcoords="offset points", fontsize=8, ha="right", color="#555555",
               arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.8})
    a.set_xscale("log", base=2)
    a.set_xticks([8, 16, 32], ["8", "16", "32"])
    a.minorticks_off()
    a.set_xlim(6.5, 40)
    a.set_ylim(ylo, yhi)
    a.set_xlabel("algorithmic latency (ms)")
    a.set_ylabel("PESQ (wideband)")
    a.set_title("Quality vs. latency (rtf < 0.1)", fontsize=11)
    a.scatter([], [], marker="o", color="#ff7f0e", edgecolor="black", s=60, label="Pareto front (mean ± std over seeds)")
    a.scatter([], [], marker="o", color="#9e9e9e", s=25, label="dominated config")
    a.scatter([], [], marker="x", color="#bbbbbb", s=36, label="over compute budget")
    a.legend(fontsize=8, loc="lower right", frameon=False)

    # ---- right: quality vs compute, all configs ----
    b.axvspan(RTF_BUDGET, 0.16, color="#d62728", alpha=0.06, zorder=0)
    b.axvline(RTF_BUDGET, color="#d62728", ls="--", lw=1)
    b.text(RTF_BUDGET + 0.003, yhi - 0.004, "over compute budget", fontsize=8, color="#d62728", va="top")
    pf2 = front(configs, "rtf", "pesq")
    staircase(b, pf2, "rtf", "pesq", 0.16, "#444444")
    for c in configs:
        on = c in pf2
        b.scatter(c["rtf"], c["pesq"], s=70 if on else 30, color=COLORS[c["latency"]],
                  edgecolor="black" if on else "none", zorder=3)
    for c in pf2:
        b.annotate(f"{c['latency']:g} ms, {name(c)}", (c["rtf"], c["pesq"]), xytext=(6, 6),
                   textcoords="offset points", fontsize=7.5)
    for lat, col in COLORS.items():
        b.scatter([], [], color=col, s=30, label=f"{lat:g} ms window")
    b.legend(fontsize=8, loc="lower left", frameon=False)
    b.set_xlim(0.03, 0.16)
    b.set_ylim(ylo, yhi)
    b.set_xlabel("real-time factor (one CPU thread, lower is cheaper)")
    b.set_ylabel("PESQ (wideband)")
    b.set_title("Quality vs. compute (all configs)", fontsize=11)

    for ax in (a, b):
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Causal GRU speech enhancer on VoiceBank-DEMAND (824 test files; noisy input PESQ {NOISY_PESQ})",
                 fontsize=10, y=1.0, color="#333333")
    fig.savefig(OUT, dpi=170, bbox_inches="tight")
    print("saved", OUT.relative_to(ROOT))
    print("front (latency):", [(c["latency"], round(c["pesq"], 3)) for c in pf])
    print("front (compute):", [(c["latency"], round(c["rtf"], 3), round(c["pesq"], 3)) for c in pf2])


if __name__ == "__main__":
    main()
