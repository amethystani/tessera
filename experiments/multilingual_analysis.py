"""Check whether the BBQ results hold beyond English and the original six
checkpoints.

Input: ml_{tag}_per_item.csv files from bbq_firstparty.py --items-csv
multilingual_items.csv, one per checkpoint (MBBQ en/es/nl/tr and KoBBQ ko; the
Catalan run uses the ca_ prefix). For each language, over checkpoint x category
cells:

  variance shares   how much of abstention is checkpoint, and of theta category
  ranking           Spearman(abstention, s_AMB) across checkpoints
  score vs theta    Spearman(s_AMB, theta) across cells
  Manski            baseline-only interval contains the measured theta
  abstainer lean    theta among items the checkpoint abstained on

Checkpoints are included only if their file exists, and the output records
which were present.

Output: results_multilingual.json
Usage: python3 multilingual_analysis.py [--prefix ml] [--min-abstained 20]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bbq_by_category import cell_stats

LAB = Path(__file__).parent / "lab_results"
OUT = Path(__file__).parent / "results_multilingual.json"

MODELS = {
    # original six (fp16), then the newer families (bf16)
    "qwen": "Qwen2.5-7B", "olmo_sft": "OLMo-2-SFT", "olmo_dpo": "OLMo-2-DPO",
    "olmo_inst": "OLMo-2-Instruct", "mistral": "Mistral-7B", "phi": "Phi-3.5-mini",
    "qwen3_4b": "Qwen3-4B-2507", "gemma4_e2b": "Gemma-4-E2B", "phi4_mini": "Phi-4-mini",
    "granite4_micro": "Granite-4.0-micro", "olmo3_7b": "OLMo-3-7B",
    "ministral3_3b": "Ministral-3-3B",
}
LANG_NAMES = {"en": "English", "es": "Spanish", "nl": "Dutch", "tr": "Turkish", "ko": "Korean", "ca": "Catalan"}


def shares(cells: pd.DataFrame, col: str) -> dict:
    grid = cells.pivot(index="model", columns="category", values=col).dropna(axis=1)
    v = grid.to_numpy(dtype=float)
    gm = v.mean()
    tot = ((v - gm) ** 2).sum()
    if tot == 0:
        return {"checkpoint": float("nan"), "category": float("nan"), "residual": float("nan")}
    r = v.shape[1] * ((v.mean(axis=1) - gm) ** 2).sum()
    c = v.shape[0] * ((v.mean(axis=0) - gm) ** 2).sum()
    return {"checkpoint": float(r / tot), "category": float(c / tot),
            "residual": float(1 - (r + c) / tot)}


def abstainer_lean(df: pd.DataFrame, min_abstained: int) -> dict:
    d = df[df.pred_baseline.notna()]
    ab = d[d.pred_baseline == d.unknown_idx]
    if len(ab) < min_abstained:
        return {"n_abstained": int(len(ab)), "theta_abstainers": None}
    theta = float((ab.pred_forced_lp == ab.biased_idx).mean())
    rng = np.random.default_rng(7)
    x = (ab.pred_forced_lp == ab.biased_idx).to_numpy(float)
    boots = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_abstained": int(len(ab)), "theta_abstainers": theta,
            "ci": [float(lo), float(hi)], "above_half": bool(lo > 0.5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="ml")
    ap.add_argument("--min-abstained", type=int, default=20)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--lab", default=str(LAB), help="directory holding the per-item files")
    a = ap.parse_args()

    frames = {}
    for tag, name in MODELS.items():
        f = Path(a.lab) / f"{a.prefix}_{tag}_per_item.csv"
        if f.exists():
            df = pd.read_csv(f, keep_default_na=False, na_values=[""])
            frames[name] = df
    assert len(frames) >= 3, f"need >=3 checkpoints, found {list(frames)}"

    out: dict = {"checkpoints": list(frames), "languages": {}}
    for lang in ("en", "es", "nl", "tr", "ko", "ca"):
        cells, per_ckpt = [], []
        for name, df in frames.items():
            sub = df[df.lang == lang]
            if sub.empty:
                continue
            for cat, g in sub.groupby("category"):
                cells.append({"model": name, "category": cat, **cell_stats(g)})
            whole = cell_stats(sub)
            per_ckpt.append({"model": name, "abstention": whole["abstention"],
                             "s_AMB": whole["s_AMB"], "theta": whole["theta"],
                             "manski_lo": whole["manski_lo"], "manski_hi": whole["manski_hi"],
                             "inside": whole["inside"], "n_items": whole["n"],
                             **{f"abst_{k}": v for k, v in abstainer_lean(sub, a.min_abstained).items()}})
        if len(per_ckpt) < 3:
            continue
        cells = pd.DataFrame(cells)
        ck = pd.DataFrame(per_ckpt)
        rho_rank, p_rank = spearmanr(ck.abstention, ck.s_AMB)
        rho_cell, p_cell = spearmanr(cells.s_AMB, cells.theta)
        leaners = ck[ck.abst_theta_abstainers.notna()]
        out["languages"][lang] = {
            "name": LANG_NAMES[lang], "n_checkpoints": int(len(ck)),
            "n_cells": int(len(cells)), "n_categories": int(cells.category.nunique()),
            "variance_shares": {c: shares(cells, c) for c in ("abstention", "theta", "s_AMB")},
            "ckpt_rho_abstention_score": float(rho_rank), "ckpt_p_abstention_score": float(p_rank),
            "cell_rho_score_theta": float(rho_cell), "cell_p_score_theta": float(p_cell),
            "manski_inside_cells": int(cells.inside.sum()), "manski_total_cells": int(len(cells)),
            "manski_inside_checkpoints": int(ck.inside.sum()),
            "abstention_range": [float(ck.abstention.min()), float(ck.abstention.max())],
            "theta_range": [float(ck.theta.min()), float(ck.theta.max())],
            "abstainer_lean_checkpoints_tested": int(len(leaners)),
            "abstainer_lean_above_half": int(leaners.abst_above_half.sum()) if len(leaners) else 0,
            "abstainer_theta_range": ([float(leaners.abst_theta_abstainers.min()),
                                       float(leaners.abst_theta_abstainers.max())]
                                      if len(leaners) else None),
            "per_checkpoint": ck.to_dict(orient="records"),
        }
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")

    print(f"checkpoints: {', '.join(out['checkpoints'])}\n")
    print(f"{'lang':8s} {'ck':>3s} {'abst->ckpt':>10s} {'theta->cat':>10s} {'rho(abst,s)':>11s} "
          f"{'rho(s,th)':>9s} {'Manski':>9s} {'abst lean>.5':>12s}")
    for lang, r in out["languages"].items():
        vs = r["variance_shares"]
        print(f"{r['name']:8s} {r['n_checkpoints']:3d} {vs['abstention']['checkpoint']*100:9.0f}% "
              f"{vs['theta']['category']*100:9.0f}% {r['ckpt_rho_abstention_score']:+11.2f} "
              f"{r['cell_rho_score_theta']:+9.2f} {r['manski_inside_cells']:4d}/{r['manski_total_cells']:<4d} "
              f"{r['abstainer_lean_above_half']:5d}/{r['abstainer_lean_checkpoints_tested']:<3d}")
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
