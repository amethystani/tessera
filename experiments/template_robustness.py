"""Does the checkpoint-level structure survive rewording the prompt?

Compares the original and reworded ("para") first-party BBQ runs, same 2,200
items per checkpoint, over the 66 checkpoint x category cells:
  * agreement of per-cell abstention, theta and s_AMB across prompts (Spearman)
  * the variance decomposition (checkpoint / category / residual) for each
  * the between-checkpoint rank correlation of abstention with s_AMB
  * Manski containment under the reworded prompt

Output: results_template_robustness.json
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
from bbq_by_category import MODELS, cell_stats

LAB = Path(__file__).parent / "lab_results"
OUT = Path(__file__).parent / "results_template_robustness.json"


def cells(prefix):
    rows = []
    for tag, name in MODELS.items():
        df = pd.read_csv(LAB / f"{prefix}_{tag}_per_item.csv")
        for cat, g in df.groupby("category"):
            rows.append({"model": name, "category": cat, **cell_stats(g)})
    return pd.DataFrame(rows)


def shares(c, col):
    v = c.pivot(index="model", columns="category", values=col).to_numpy(float)
    gm = v.mean(); tot = ((v - gm) ** 2).sum()
    r = v.shape[1] * ((v.mean(1) - gm) ** 2).sum()
    k = v.shape[0] * ((v.mean(0) - gm) ** 2).sum()
    return {"checkpoint": r / tot, "category": k / tot, "residual": 1 - (r + k) / tot}


def main():
    a, b = cells("bbq11"), cells("bbq11para")
    assert len(a) == len(b) == 66
    m = a.merge(b, on=["model", "category"], suffixes=("_orig", "_para"))
    out = {"n_cells": len(m)}
    for q in ("abstention", "theta", "s_AMB"):
        r, _ = spearmanr(m[f"{q}_orig"], m[f"{q}_para"])
        out[f"cell_agreement_{q}"] = float(r)
    for name, c in (("orig", a), ("para", b)):
        out[f"variance_{name}"] = {q: shares(c, q) for q in ("abstention", "theta", "s_AMB")}
        ck = c.groupby("model")[["abstention", "s_AMB", "theta"]].mean()
        out[f"ckpt_rho_abstention_score_{name}"] = float(spearmanr(ck.abstention, ck.s_AMB)[0])
        out[f"cell_rho_score_theta_{name}"] = float(spearmanr(c.s_AMB, c.theta)[0])
        out[f"manski_inside_{name}"] = int(c.inside.sum())
    ca, cb = a.groupby("model").abstention.mean(), b.groupby("model").abstention.mean()
    out["checkpoint_abstention_orig"] = ca.to_dict()
    out["checkpoint_abstention_para"] = cb.to_dict()
    out["checkpoint_abstention_rank_rho"] = float(spearmanr(ca, cb.loc[ca.index])[0])
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=1))
    for n in ("orig", "para"):
        print(n, {q: round(v["checkpoint"], 2) for q, v in out[f"variance_{n}"].items()},
              {q: round(v["category"], 2) for q, v in out[f"variance_{n}"].items()})


if __name__ == "__main__":
    main()
