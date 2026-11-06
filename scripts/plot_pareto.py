"""Pareto plot of the latency study (PESQ against latency), from results/latency/runs.csv.

    python scripts/plot_pareto.py

Only configs that meet the compute budget (rtf < 0.1) can be on the front. A config is on the
front if no other config has lower or equal latency and higher or equal PESQ (and is better in one).
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
    ok = [c for c in configs if c["rtf"] < RTF_BUDGET]
    pf = front(ok, "latency", "pesq")
    ref = next(c for c in ok if c["latency"] == 32 and c["hop_ms"] == 8 and c["hidden"] == 256)

    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(7, 4.4))
    staircase(ax, pf, "latency", "pesq", 40, "#ff7f0e")
    others = [c for c in ok if c not in pf and c is not ref]
    ax.scatter([c["latency"] for c in others], [c["pesq"] for c in others], s=30, color="#c4c4c4",
               zorder=3, label="other configurations")
    for c in pf:
        ax.errorbar(c["latency"], c["pesq"], yerr=c["std"], fmt="o", ms=11, color="#ff7f0e",
                    mec="black", capsize=5, zorder=4)
        ax.annotate(f"{c['pesq']:.2f}", (c["latency"], c["pesq"]), xytext=(12, -4),
                    textcoords="offset points", fontsize=12, weight="bold")
    ax.scatter(ref["latency"], ref["pesq"], s=60, color="white", edgecolor="#555555", lw=1.5, zorder=4)
    ax.annotate(f"usual 32 ms window: {ref['pesq']:.2f}\nworse and slower than 16 ms",
                (ref["latency"], ref["pesq"]), xytext=(-18, -2), textcoords="offset points", va="center",
                ha="right", fontsize=9, color="#444444",
                arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.8})
    ax.errorbar([], [], yerr=[], fmt="o", ms=9, color="#ff7f0e", mec="black", capsize=4,
                label="best trade-off (mean ± std, 3 seeds)")
    ax.legend(fontsize=9, loc="lower right", frameon=False)
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32], ["8 ms", "16 ms", "32 ms"])
    ax.minorticks_off()
    ax.set_xlim(6.5, 40)
    ax.set_ylim(2.30, 2.55)
    ax.set_xlabel("latency (STFT window)  \u2190 better")
    ax.set_ylabel("speech quality (PESQ)  better \u2192")
    ax.set_title("Quality vs. latency of a real-time speech enhancer", fontsize=12, pad=10)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, -0.02, f"VoiceBank-DEMAND, 824 test files (noisy input: PESQ {NOISY_PESQ}). "
             "Shown: configurations that run below 10% of one CPU core (rtf < 0.1).", fontsize=8, color="#666666")
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print("saved", OUT.relative_to(ROOT), "front:", [(c["latency"], round(c["pesq"], 3)) for c in pf])


if __name__ == "__main__":
    main()
