"""Disposition of the items a checkpoint abstains on.

For each checkpoint on BBQ ambiguous items we have the baseline answer (which may
be UNKNOWN) and a logit-forced choice with UNKNOWN removed. The forced choice
gives a stereotype-aligned indicator for every item, so the abstained subset can
be looked at directly. Per checkpoint:

  theta_abst   P(forced pick is stereotype-aligned | baseline abstained)
  theta_ans    the same, given the baseline committed to an answer
  delta_adj    theta_abst - theta_ans computed within category and averaged,
               so category mix cannot produce a difference by itself
  Lambda       odds(abstain | aligned) / odds(abstain | not aligned)
  consistency  P(forced pick == committed pick | committed)

CIs come from a category-stratified item bootstrap (5000 reps). delta_adj gets a
permutation test with abstain labels shuffled within category. A result is only
reported as real if it has the same sign under both prompt variants.

Output: results_abstainer_disposition.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

LAB = Path(__file__).parent / "lab_results"
OUT = Path(__file__).parent / "results_abstainer_disposition.json"
MODELS = {"qwen": "Qwen2.5-7B", "olmo_sft": "OLMo-2-SFT", "olmo_dpo": "OLMo-2-DPO",
          "olmo_inst": "OLMo-2-Instruct", "mistral": "Mistral-7B", "phi": "Phi-3.5-mini"}
B = 5000


def prep(prefix, tag):
    d = pd.read_csv(LAB / f"{prefix}_{tag}_per_item.csv")
    d = d[d.pred_baseline.notna() & d.pred_forced_lp.notna()].copy()
    d["ab"] = (d.pred_baseline == d.unknown_idx).to_numpy()
    d["al"] = (d.pred_forced_lp == d.biased_idx).to_numpy()
    d["same"] = (d.pred_forced_lp == d.pred_baseline).to_numpy()
    return d[["category", "ab", "al", "same"]].reset_index(drop=True)


def stats(cat, ab, al, same):
    """cat: int codes. Returns dict of the statistics on one (re)sample."""
    n_ab, n_an = ab.sum(), (~ab).sum()
    t_ab = al[ab].mean() if n_ab else np.nan
    t_an = al[~ab].mean() if n_an else np.nan
    num = den = 0.0
    for c in np.unique(cat):
        m = cat == c
        a, b = m & ab, m & ~ab
        if a.sum() and b.sum():
            w = 1.0 / (1.0 / a.sum() + 1.0 / b.sum())      # inverse-variance-style weight
            num += w * (al[a].mean() - al[b].mean()); den += w
    d_adj = num / den if den else np.nan
    # selection odds ratio
    def odds(x):
        return x / (1 - x) if 0 < x < 1 else np.nan
    p_al = ab[al].mean() if al.sum() else np.nan
    p_no = ab[~al].mean() if (~al).sum() else np.nan
    lam = odds(p_al) / odds(p_no) if not np.isnan(p_al) and not np.isnan(p_no) else np.nan
    cons = same[~ab].mean() if n_an else np.nan
    return {"theta_abst": t_ab, "theta_ans": t_an, "delta_adj": d_adj, "lambda": lam, "consistency": cons}


def one(prefix, tag, rng):
    d = prep(prefix, tag)
    cats = pd.factorize(d.category)[0]
    ab, al, same = d.ab.to_numpy(), d.al.to_numpy(), d.same.to_numpy()
    est = stats(cats, ab, al, same)
    est["abstention"] = float(ab.mean()); est["n_abstained"] = int(ab.sum()); est["n"] = int(len(d))
    idx_by_cat = [np.where(cats == c)[0] for c in np.unique(cats)]
    boots = {k: [] for k in ("theta_abst", "theta_ans", "delta_adj", "lambda", "consistency")}
    for _ in range(B):
        pick = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx_by_cat])
        s = stats(cats[pick], ab[pick], al[pick], same[pick])
        for k in boots:
            boots[k].append(s[k])
    ci = {k: [float(x) for x in np.nanpercentile(v, [2.5, 97.5])] for k, v in boots.items()}
    # permutation of delta_adj: shuffle abstain labels within category
    obs = est["delta_adj"]; cnt = 0; reps = 2000
    for _ in range(reps):
        ab_p = ab.copy()
        for ix in idx_by_cat:
            ab_p[ix] = rng.permutation(ab[ix])
        if abs(stats(cats, ab_p, al, same)["delta_adj"]) >= abs(obs):
            cnt += 1
    est["p_perm_delta_adj"] = (cnt + 1) / (reps + 1)
    est["ci"] = ci
    est["theta_abst_above_half"] = bool(ci["theta_abst"][0] > 0.5)
    return est


def main():
    rng = np.random.default_rng(20260926)
    out = {"bootstrap_reps": B, "perm_reps": 2000, "results": {}}
    for prefix, label in (("bbq11", "orig"), ("bbq11para", "para")):
        out["results"][label] = {}
        for tag, name in MODELS.items():
            r = one(prefix, tag, rng)
            out["results"][label][name] = r
            print(f"{label:4s} {name:16s} abst={r['abstention']:.2f} n_ab={r['n_abstained']:4d} "
                  f"theta_abst={r['theta_abst']:.3f}{[round(x,2) for x in r['ci']['theta_abst']]} "
                  f"theta_ans={r['theta_ans']:.3f} d_adj={r['delta_adj']:+.3f}"
                  f"{[round(x,2) for x in r['ci']['delta_adj']]} p={r['p_perm_delta_adj']:.3f} "
                  f"lam={r['lambda']:.2f}{[round(x,2) for x in r['ci']['lambda']]} cons={r['consistency']:.2f}")
    OUT.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
