"""Per-category BBQ analysis over the 66 (checkpoint x category) cells.

With six checkpoints the Spearman correlation between reported score and
measured disposition has a bootstrap interval of roughly [-1.00, +0.80], so it
cannot answer whether the score tracks disposition. Each checkpoint was run on
eleven BBQ categories (200 items each), which gives 66 cells to ask the same
question on.

Cells are not independent, since cells from one checkpoint share its behaviour
and cells from one category share its items. Two cluster bootstraps handle
that, one resampling checkpoints and one resampling categories. We report both
and read the wider interval.

Per cell:
  abstention  P(model picks UNKNOWN) at baseline
  s_AMB       (1 - abstention) * b_cond, BBQ's ambiguous bias score
  theta       stereotype-aligned rate under logit-forced choice

It also records whether the Manski interval built from a cell's baseline
alone contains that cell's measured theta. At 200 items some misses are
expected, so the count is reported as observed.

Usage: python3 bbq_by_category.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

LAB = Path(__file__).parent / "lab_results"
OUT = Path(__file__).parent / "results_bbq11_by_category.json"

MODELS = {
    "qwen": "Qwen2.5-7B", "olmo_sft": "OLMo-2-SFT", "olmo_dpo": "OLMo-2-DPO",
    "olmo_inst": "OLMo-2-Instruct", "mistral": "Mistral-7B", "phi": "Phi-3.5-mini",
}


def cell_stats(g: pd.DataFrame) -> dict:
    """Abstention, s_AMB, forced stereotype-aligned rate theta and the baseline-only Manski interval for one cell of items."""
    scored = g[g.pred_baseline.notna()]
    p = float((scored.pred_baseline == scored.unknown_idx).mean())
    committed = scored[scored.pred_baseline != scored.unknown_idx]
    if len(committed):
        q = float((committed.pred_baseline == committed.biased_idx).mean())
        b = 2.0 * q - 1.0
    else:
        q, b = 0.5, 0.0
    theta = float((g.pred_forced_lp == g.biased_idx).mean())
    lo = (1 - p) * q
    hi = lo + p
    return {"n": int(len(g)), "abstention": p, "s_AMB": (1 - p) * b,
            "theta": theta, "manski_lo": lo, "manski_hi": hi,
            "inside": bool(lo - 1e-9 <= theta <= hi + 1e-9)}


def cluster_boot(df, x, y, cluster, reps=5000, seed=11):
    """Spearman(x, y) over cells, resampling whole clusters."""
    rng = np.random.default_rng(seed)
    keys = df[cluster].unique()
    out = []
    for _ in range(reps):
        pick = rng.choice(keys, size=len(keys), replace=True)
        sub = pd.concat([df[df[cluster] == k] for k in pick])
        if sub[x].nunique() < 3 or sub[y].nunique() < 3:
            continue
        r, _ = spearmanr(sub[x], sub[y])
        if not np.isnan(r):
            out.append(r)
    return np.percentile(out, [2.5, 97.5])


def main() -> None:
    rows = []
    for tag, name in MODELS.items():
        df = pd.read_csv(LAB / f"bbq11_{tag}_per_item.csv")
        for cat, g in df.groupby("category"):
            rows.append({"model": name, "category": cat, **cell_stats(g)})
    cells = pd.DataFrame(rows)
    assert len(cells) == 66, len(cells)

    out: dict = {"n_cells": len(cells), "n_items_per_cell": 200}
    for x, y, label in [("s_AMB", "theta", "score_vs_theta"),
                        ("abstention", "s_AMB", "abstention_vs_score"),
                        ("abstention", "theta", "abstention_vs_theta")]:
        r, p = spearmanr(cells[x], cells[y])
        lo_m, hi_m = cluster_boot(cells, x, y, "model")
        lo_c, hi_c = cluster_boot(cells, x, y, "category")
        out[label] = {"spearman": float(r), "p_naive_independent": float(p),
                      "ci_model_cluster": [float(lo_m), float(hi_m)],
                      "ci_category_cluster": [float(lo_c), float(hi_c)]}

    # Within-checkpoint association across categories: does the score track
    # theta when the checkpoint is held fixed and only the social category varies?
    within = []
    for name, g in cells.groupby("model"):
        r, _ = spearmanr(g.s_AMB, g.theta)
        r2, _ = spearmanr(g.abstention, g.s_AMB)
        within.append({"model": name, "rho_score_theta": float(r),
                       "rho_abstention_score": float(r2)})
    out["within_checkpoint"] = within
    out["within_mean_rho_score_theta"] = float(np.mean([w["rho_score_theta"] for w in within]))
    out["within_mean_rho_abstention_score"] = float(np.mean([w["rho_abstention_score"] for w in within]))
    out["within_n_positive_score_theta"] = int(sum(w["rho_score_theta"] > 0 for w in within))
    out["within_n_negative_abstention_score"] = int(sum(w["rho_abstention_score"] < 0 for w in within))

    # Where does the variance live? Additive two-way decomposition of each
    # quantity over the 6 x 11 grid (one cell per checkpoint x category):
    # share attributable to the checkpoint, to the category, and the remainder
    # (interaction + within-cell sampling noise). If abstention lives in the
    # checkpoint and disposition lives in the category, then comparing
    # checkpoints ranks them by abstention, which is the paper's claim in one
    # number, and does so without relying on n=6.
    def shares(col):
        grid = cells.pivot(index="model", columns="category", values=col)
        v = grid.to_numpy(dtype=float)
        gm = v.mean()
        ss_tot = ((v - gm) ** 2).sum()
        ss_row = v.shape[1] * ((v.mean(axis=1) - gm) ** 2).sum()
        ss_col = v.shape[0] * ((v.mean(axis=0) - gm) ** 2).sum()
        return {"checkpoint": float(ss_row / ss_tot),
                "category": float(ss_col / ss_tot),
                "residual": float(1 - (ss_row + ss_col) / ss_tot)}
    out["variance_shares"] = {c: shares(c) for c in ("abstention", "theta", "s_AMB")}

    # Standardised OLS: s_AMB on abstention and theta across the 66 cells.
    Z = cells[["abstention", "theta"]].to_numpy(dtype=float)
    Z = (Z - Z.mean(0)) / Z.std(0)
    yv = cells.s_AMB.to_numpy(dtype=float)
    yv = (yv - yv.mean()) / yv.std()
    X = np.column_stack([np.ones(len(Z)), Z])
    beta = np.linalg.lstsq(X, yv, rcond=None)[0]
    resid = yv - X @ beta
    out["ols_standardised"] = {
        "beta_abstention": float(beta[1]), "beta_theta": float(beta[2]),
        "r_squared": float(1 - resid.var() / yv.var()),
    }
    rng = np.random.default_rng(3)
    names = cells.model.unique()
    bs = []
    for _ in range(5000):
        pick = rng.choice(names, size=len(names), replace=True)
        sub = pd.concat([cells[cells.model == k] for k in pick])
        Zs = sub[["abstention", "theta"]].to_numpy(dtype=float)
        if Zs.std(0).min() < 1e-9 or sub.s_AMB.std() < 1e-9:
            continue
        Zs = (Zs - Zs.mean(0)) / Zs.std(0)
        ys = (sub.s_AMB.to_numpy(dtype=float) - sub.s_AMB.mean()) / sub.s_AMB.std()
        Xs = np.column_stack([np.ones(len(Zs)), Zs])
        bs.append(np.linalg.lstsq(Xs, ys, rcond=None)[0][1:])
    bs = np.array(bs)
    out["ols_standardised"]["ci_abstention_model_cluster"] = [float(x) for x in np.percentile(bs[:, 0], [2.5, 97.5])]
    out["ols_standardised"]["ci_theta_model_cluster"] = [float(x) for x in np.percentile(bs[:, 1], [2.5, 97.5])]

    out["manski_misses"] = cells[~cells.inside][["model", "category", "theta", "manski_lo", "manski_hi"]].to_dict(orient="records")
    out["manski_inside"] = int(cells.inside.sum())
    out["manski_total"] = int(len(cells))
    out["theta_min"] = float(cells.theta.min())
    out["theta_max"] = float(cells.theta.max())
    out["cells_theta_above_half"] = int((cells.theta > 0.5).sum())
    out["cells"] = cells.to_dict(orient="records")
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    for k in ("score_vs_theta", "abstention_vs_score", "abstention_vs_theta"):
        v = out[k]
        print(f"{k:22s} rho={v['spearman']:+.3f}  "
              f"CI(model-cluster)=[{v['ci_model_cluster'][0]:+.2f},{v['ci_model_cluster'][1]:+.2f}]  "
              f"CI(category-cluster)=[{v['ci_category_cluster'][0]:+.2f},{v['ci_category_cluster'][1]:+.2f}]")
    print("\nwithin-checkpoint (across the 11 categories):")
    for w in within:
        print(f"  {w['model']:16s} rho(score,theta)={w['rho_score_theta']:+.2f}  "
              f"rho(abstention,score)={w['rho_abstention_score']:+.2f}")
    vs = out["variance_shares"]
    print("\nvariance shares over the 6x11 grid (checkpoint / category / residual):")
    for c in ("abstention", "theta", "s_AMB"):
        v = vs[c]
        print(f"  {c:11s} {v['checkpoint']*100:5.1f}% / {v['category']*100:5.1f}% / {v['residual']*100:5.1f}%")
    o = out["ols_standardised"]
    print(f"\nstandardised OLS of s_AMB: beta_abstention={o['beta_abstention']:+.2f} "
          f"CI{[round(x,2) for x in o['ci_abstention_model_cluster']]}, "
          f"beta_theta={o['beta_theta']:+.2f} CI{[round(x,2) for x in o['ci_theta_model_cluster']]}, "
          f"R2={o['r_squared']:.2f}")
    print(f"Manski misses: {out['manski_misses']}")
    print(f"\nManski interval contains own theta: {out['manski_inside']}/{out['manski_total']} cells")
    print(f"theta range across cells: {out['theta_min']:.3f}-{out['theta_max']:.3f}; "
          f"above 0.5 in {out['cells_theta_above_half']}/66")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
