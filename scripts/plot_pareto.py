"""Pareto plot: speech quality (PESQ) against latency, before and after the training-recipe study.

    python scripts/plot_pareto.py

Reads results/latency/runs.csv (before: SI-SDR loss only) and results/objective/runs.csv
(after: SI-SDR + compressed spectral loss). Each budget keeps its architecture, so the compute
cost (rtf) of a point does not change between before and after.
"""
import csv
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "pareto.png"
NOISY_PESQ = 1.97

# the architecture chosen per latency budget in the latency study (rtf < 0.1)
ARCH = {8: dict(n_fft=128, hop=32, hidden=128, layers=1),
        16: dict(n_fft=256, hop=64, hidden=256, layers=2),
        32: dict(n_fft=512, hop=128, hidden=256, layers=2)}
# the training recipe the agent confirmed per budget in the objective study
RECIPE = {8: dict(spec_weight=400.0, steps=30000), 16: dict(spec_weight=400.0, steps=10000),
          32: dict(spec_weight=400.0, steps=10000)}


def rows(path):
    return [r for r in csv.DictReader(path.open()) if r["state"] == "done"]


def matches(r, **want):
    return all(float(r[k]) == float(v) for k, v in want.items())


def point(rs):
    pesq = [float(r["pesq"]) for r in rs]
    return mean(pesq), (stdev(pesq) if len(pesq) > 1 else 0.0), len(pesq)


def main():
    before_rows = rows(ROOT / "results" / "latency" / "runs.csv")
    after_rows = rows(ROOT / "results" / "objective" / "runs.csv")
    lats = sorted(ARCH)
    before = {l: point([r for r in before_rows if matches(r, **ARCH[l])]) for l in lats}
    after = {l: point([r for r in after_rows if matches(r, **ARCH[l], **RECIPE[l])]) for l in lats}

    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for data, color, label, z in ((before, "#9e9e9e", "SI-SDR loss only (before)", 2),
                                  (after, "#ff7f0e", "+ spectral loss (after)", 3)):
        # Pareto staircase: best quality available at or below each latency
        xs, ys, best = [], [], 0.0
        for l in lats:
            if data[l][0] > best:
                if xs:
                    xs.append(l)
                    ys.append(best)
                best = data[l][0]
                xs.append(l)
                ys.append(best)
        xs.append(44)
        ys.append(best)
        ax.plot(xs, ys, color=color, lw=2.2, zorder=z, label=label)
        for l in lats:
            m, s, n = data[l]
            ax.errorbar(l, m, yerr=s, fmt="o", ms=10, color=color, mec="black" if color != "#9e9e9e" else "none",
                        capsize=4, zorder=z + 1)
    for l in lats:
        m, s, n = after[l]
        ax.annotate(f"{m:.2f}", (l, m), xytext=(-12, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=12, weight="bold")
        gain = m - before[l][0]
        ax.annotate(f"+{gain:.2f}", (l, (m + before[l][0]) / 2), xytext=(12, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#cc6600")
    ax.set_xscale("log", base=2)
    ax.set_xticks(lats, [f"{l} ms" for l in lats])
    ax.minorticks_off()
    ax.set_xlim(6.5, 44)
    ax.set_ylim(2.35, 2.85)
    ax.set_xlabel("latency (STFT window)   ← better")
    ax.set_ylabel("speech quality (PESQ)   better →")
    ax.set_title("Quality vs. latency of a real-time speech enhancer", fontsize=12, pad=10)
    ax.legend(fontsize=9, loc="upper left", frameon=False)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, -0.08, f"VoiceBank-DEMAND, 824 test files (noisy input: PESQ {NOISY_PESQ}). Mean ± std over seeds "
             "(3; 2 at 8 ms after).\nSame model sizes before and after, chosen to run below 10% of one CPU core "
             "(rtf < 0.1). Only training changed\n(spectral loss; 30k instead of 10k steps at 8 ms).",
             fontsize=8, color="#666666")
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    for l in lats:
        print(f"{l} ms: before {before[l][0]:.3f}±{before[l][1]:.3f} (n={before[l][2]}), "
              f"after {after[l][0]:.3f}±{after[l][1]:.3f} (n={after[l][2]})")
    print("saved", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
