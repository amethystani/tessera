"""Plot the main results from the stored outputs into results/figures/.

Everything is read from files under results/ and experiments/, so the plots
change whenever those files do.

Usage: python3 scripts/make_result_plots.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(EXP))

# Okabe-Ito palette
BLUE, ORANGE, GREEN, RED, PURPLE, SKY, YELLOW, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#7f7f7f")

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
})


def load(name):
    return json.loads((EXP / name).read_text())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(ROOT))


def decomposition_pairs():
    df = pd.read_csv(ROOT / "results" / "decap_decomposition.csv")
    df["share"] = df.mechanical_fraction * 100
    df = df.sort_values("share").reset_index(drop=True)
    colors = [RED if w else BLUE for w in df.disposition_worsened]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(range(len(df)), df.share, color=colors, width=0.8)
    ax.axhline(100, color="black", lw=0.8, ls="--")
    ax.axhline(df.share.median(), color=GREY, lw=0.8, ls=":")
    ax.text(0.4, df.share.median() + 3, f"median {df.share.median():.1f}%",
            ha="left", color="#444444", fontsize=8)
    ax.set_xticks([])
    ax.set_xlabel("32 published method/model pairs, sorted")
    ax.set_ylabel("Share of the reported improvement\nthat is extra abstention (%)")
    ax.bar([0], [0], color=BLUE, label="committed-answer bias improved")
    ax.bar([0], [0], color=RED, label="committed-answer bias got worse")
    ax.legend(loc="upper left")
    ax.set_title("Published BBQ improvements split into coverage and disposition", fontsize=10)
    save(fig, "decomposition_pairs.png")


def cells():
    d = load("results_bbq11_by_category.json")
    c = pd.DataFrame(d["cells"])
    models = list(dict.fromkeys(c.model))
    pal = dict(zip(models, [BLUE, ORANGE, GREEN, RED, PURPLE, SKY]))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    for m in models:
        s = c[c.model == m]
        axes[0].scatter(s.abstention, s.s_AMB, s=14, color=pal[m], label=m, alpha=0.85)
        axes[1].scatter(s.theta, s.s_AMB, s=14, color=pal[m], alpha=0.85)
    axes[0].set_xlabel("Abstention rate (baseline)")
    axes[1].set_xlabel("Stereotype-aligned rate under forced choice")
    axes[0].set_ylabel("Reported bias score $s_{AMB}$")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("66 checkpoint x category cells (200 items each)", fontsize=10)
    save(fig, "cells_score_vs_abstention_theta.png")


def multilingual_rho():
    d, e = load("results_multilingual.json"), load("results_catalan.json")
    langs = dict(d["languages"]); langs["ca"] = e["languages"]["ca"]
    order = ["en", "es", "nl", "tr", "ko", "ca"]
    names = [langs[k]["name"] for k in order]
    rho = [langs[k]["ckpt_rho_abstention_score"] for k in order]
    fig, ax = plt.subplots(figsize=(5.2, 3))
    cols = [RED if k == "ca" else BLUE for k in order]
    ax.barh(range(len(order)), rho, color=cols, height=0.65)
    for i, v in enumerate(rho):
        ax.text(v - 0.02, i, f"{v:+.2f}", va="center", ha="right", fontsize=8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlim(-1.15, 0.1)
    ax.set_xlabel("Spearman correlation of abstention with the reported score\n(across 12 checkpoints)")
    ax.set_title("The score ranks checkpoints by abstention in five of six languages", fontsize=9.5)
    save(fig, "multilingual_rho.png")


def catalan_bias():
    from make_paper_assets import _catalan_mechanism_rows
    rows = _catalan_mechanism_rows()
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for i, (_, be, bc) in enumerate(rows):
        ax.plot([bc, be], [i, i], color="#bbbbbb", lw=2, zorder=1)
    ax.scatter([r[1] for r in rows], range(len(rows)), s=30, facecolor="white",
               edgecolor="black", zorder=2, label="English")
    ax.scatter([r[2] for r in rows], range(len(rows)), s=30, color=RED, zorder=3, label="Catalan")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel("Committed-answer bias $b$")
    ax.legend(loc="upper right")
    ax.set_title("Bias among committed answers, English vs Catalan", fontsize=10)
    save(fig, "catalan_bias_by_checkpoint.png")


def false_positive():
    d = load("draw_count_calibration.json")
    r = pd.DataFrame(d["rows"])
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.plot(r.n_draws, r.naive_false_positive_rate * 100, "o-", color=RED, label="plug-in certificate")
    ax.plot(r.n_draws, r.corrected_per_item_false_positive_rate * 100, "s-", color=BLUE,
            label="exact-binomial corrected")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Draws per condition")
    ax.set_ylabel("False-positive rate (%)")
    ax.set_title("False rejections on compatible panels (resampled null)", fontsize=10)
    ax.legend()
    save(fig, "false_positive_vs_draws.png")


def power():
    d = load("results_power_by_magnitude.json")
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    pal = [SKY, BLUE, GREEN, ORANGE, RED, PURPLE]
    for b, col in zip(d["incompatible_bins"], pal):
        xs = [x["n_draws"] for x in b["by_draw"] if x["power_corrected_per_item"] is not None]
        ys = [x["power_corrected_per_item"] * 100 for x in b["by_draw"]
              if x["power_corrected_per_item"] is not None]
        last = b["gamma_high"] >= 1.0
        lab = f"$\\gamma^*>{b['gamma_low']:.2f}$" if last else f"$\\gamma^*\\in({b['gamma_low']:.2f},{b['gamma_high']:.2f}]$"
        ax.plot(xs, ys, "o-", color=col, label=lab, ms=3.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Draws per condition")
    ax.set_ylabel("Detection power (%)")
    ax.set_title("Corrected certificate: power by size of violation", fontsize=10)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(fig, "power_by_violation_size.png")


def certificate_counts():
    d = load("results_n480_summary.json")
    rows = d["rows"]
    names = [r["model"] for r in rows]
    x = np.arange(len(rows)); w = 0.27
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar(x - w, [r["naive_incompatible"] for r in rows], w, color=RED, label="plug-in rejections")
    ax.bar(x, [r["null_mean"] for r in rows], w, color=GREY, label="expected under no violation")
    ax.bar(x + w, [r["corrected_incompatible"] for r in rows], w, color=BLUE, label="corrected rejections")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Items (of 480)")
    ax.set_title("WinoGender certificate on 138,240 generations", fontsize=10)
    ax.legend(fontsize=7.5)
    save(fig, "certificate_counts.png")


if __name__ == "__main__":
    decomposition_pairs(); cells(); multilingual_rho(); catalan_bias()
    false_positive(); power(); certificate_counts()
