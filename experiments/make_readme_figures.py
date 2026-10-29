"""Render the two PNG summary figures embedded in README.md.

Reads only the committed CSVs in results/processed/ (produced by
analyze_results.py); makes no API calls.

    python experiments/make_readme_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "results" / "processed"
OUT = ROOT / "docs" / "figures"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

FAMILIES = [
    ("claude", "Claude\nSonnet 4.6"),
    ("gpt55", "GPT-5.5"),
    ("gemini", "Gemini 3\nFlash"),
    ("deepseek", "DeepSeek-\nV3.1"),
    ("qwen35", "Qwen3.5-\n397B"),
]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bars(ax, x, vals, lo, hi, color, label, w):
    ax.bar(x, vals, w, color=color, label=label, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(x, vals, yerr=[vals - lo, hi - vals], fmt="none",
                ecolor=INK2, elinewidth=1, capsize=3)


def fig_easy_hard():
    df = pd.read_csv(PROC / "m2b_easy_vs_hard.csv")
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    _style(ax)
    strata = ["easy", "hard"]
    x = np.arange(len(strata))
    w = 0.36
    for off, mode, color, label in [(-w / 2, "instant", BLUE, "Instant"),
                                    (w / 2, "light", ORANGE, "Thinking")]:
        d = df[df["mode"] == mode].set_index("difficulty").loc[strata]
        v, lo, hi = d["alpha"].values, d["ci_lo"].values, d["ci_hi"].values
        _bars(ax, x + off, v, lo, hi, color, label, w)
        for xi, vi, hii in zip(x + off, v, hi):
            ax.text(xi, max(vi, hii) + 0.03, f"{vi:.2f}", ha="center",
                    va="bottom", color=INK, fontsize=10)
    n = df.groupby("difficulty")["n_scenarios"].first()
    ax.set_xticks(x, [f"Consensus scenarios (n={n['easy']})",
                      f"Model-disputed scenarios (n={n['hard']})"], color=INK)
    ax.set_ylim(-0.1, 1.2)
    ax.set_ylabel("Krippendorff's α (cross-model)", color=INK2)
    ax.set_title("Cross-model agreement collapses on disputed cases;\n"
                 "thinking narrows it only modestly",
                 loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK)
    fig.text(0.01, 0.01, "Error bars: bootstrap 95% CI. Source: "
             "results/processed/m2b_easy_vs_hard.csv", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "agreement_easy_vs_hard.png", dpi=200)
    plt.close(fig)


def fig_thinking_effect():
    eff = pd.read_csv(PROC / "m7_thinking_effect.csv").set_index("family")
    tok = pd.read_csv(PROC / "m_reasoning_tokens.csv")
    tok = tok[tok["mode"] == "light"].set_index("family")["mean"]
    keys = [k for k, _ in FAMILIES]
    labels = [f"{name}\n({tok[k]:,.0f} tok)" for k, name in FAMILIES]
    e = eff.loc[keys]
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    _style(ax)
    x = np.arange(len(keys))
    w = 0.36
    _bars(ax, x - w / 2, e["verdict_flip_rate"].values, e["verdict_ci_lo"].values,
          e["verdict_ci_hi"].values, BLUE, "Binary verdict flipped", w)
    _bars(ax, x + w / 2, e["framework_shift_rate"].values, e["fw_ci_lo"].values,
          e["fw_ci_hi"].values, ORANGE, "Self-labeled framework shifted", w)
    ax.set_xticks(x, labels, color=INK, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Share of scenarios (instant → thinking)", color=INK2)
    ax.set_title("Thinking changes stated frameworks more often than verdicts",
                 loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    fig.text(0.01, 0.01, "Tick labels: mean reasoning tokens/call in thinking mode. "
             "Error bars: bootstrap 95% CI.\nSource: results/processed/"
             "m7_thinking_effect.csv, results/processed/m_reasoning_tokens.csv",
             color=INK2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "thinking_effect_by_model.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_easy_hard()
    fig_thinking_effect()
    print(f"Wrote figures to {OUT}")
