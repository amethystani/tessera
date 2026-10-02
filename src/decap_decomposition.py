"""Coverage/disposition decomposition of published BBQ ambiguous bias scores.

Applies the identity from Parrish et al. (2022) to the numbers published in
DeCAP (NAACL 2025, arXiv:2503.19426, Table 11a). Inputs are in
data/decap_bbq_ambiguous.csv, which records their source.

On ambiguous BBQ items the gold answer is UNKNOWN, so accuracy_ambig is the
abstention rate p:

    s_AMB = (1 - accuracy_ambig) * b_cond

With c = 1 - p as the committed-answer rate, a reported change in s_AMB splits
exactly into

    s1 - s0 = b0 * (c1 - c0)      coverage: the model abstains more
            + c1 * (b1 - b0)      disposition: the model is less biased

Only the second term is debiasing. The first moves mass into the abstain
option, which shrinks the reported score and also widens the Manski identified
set from width p0 to p1.

Usage: python3 decap_decomposition.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASELINE = "Base"


def decompose(acc0: float, s0: float, acc1: float, s1: float) -> dict:
    """All inputs in percentage points, as published."""
    p0, p1 = acc0 / 100.0, acc1 / 100.0      # abstention rates
    c0, c1 = 1.0 - p0, 1.0 - p1              # committed-answer rates
    b0 = s0 / c0 if c0 > 0 else float("nan")  # bias among committed answers
    b1 = s1 / c1 if c1 > 0 else float("nan")

    delta = s1 - s0
    coverage = b0 * (c1 - c0)
    disposition = c1 * (b1 - b0)

    # Fraction of the reported movement attributable to coverage. Defined only
    # when the reported score actually moved.
    mechanical = coverage / delta if abs(delta) > 1e-12 else float("nan")
    return {
        "abstention_before": p0, "abstention_after": p1,
        "b_cond_before": b0, "b_cond_after": b1,
        "reported_delta": delta,
        "coverage_component": coverage,
        "disposition_component": disposition,
        "mechanical_fraction": mechanical,
        "disposition_worsened": bool(b1 > b0 + 1e-9),
        "identified_width_before": p0,
        "identified_width_after": p1,
        "identified_set_widened": bool(p1 > p0 + 1e-9),
        "reported_improvement": bool(delta < -1e-9),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/decap_bbq_ambiguous.csv")
    ap.add_argument("--out", default="results/decap_decomposition.csv")
    ap.add_argument("--summary", default="results/decap_decomposition_summary.json")
    a = ap.parse_args()

    df = pd.read_csv(a.data, comment="#")
    base = df[df.method == BASELINE].set_index("model")
    rows = []
    for _, r in df[df.method != BASELINE].iterrows():
        b = base.loc[r["model"]]
        d = decompose(float(b["accuracy"]), float(b["bias_score"]),
                      float(r["accuracy"]), float(r["bias_score"]))
        d.update(method=r["method"], model=r["model"],
                 acc_before=float(b["accuracy"]), acc_after=float(r["accuracy"]),
                 score_before=float(b["bias_score"]), score_after=float(r["bias_score"]))
        rows.append(d)

    out = pd.DataFrame(rows)
    # Verify the identity reconstructs the reported delta on every row.
    recon = out.coverage_component + out.disposition_component
    assert np.allclose(recon, out.reported_delta, atol=1e-9), "identity failed"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)

    improved = out[out.reported_improvement]
    mech = improved.mechanical_fraction.dropna()
    summary = {
        "source": "DeCAP (NAACL 2025, arXiv:2503.19426) Table 11a, BBQ ambiguous",
        "n_pairs": int(len(out)),
        "n_reporting_improvement": int(len(improved)),
        "median_mechanical_fraction": float(mech.median()),
        "mean_mechanical_fraction": float(mech.mean()),
        "n_at_least_90pct_mechanical": int((mech >= 0.90).sum()),
        "n_at_least_75pct_mechanical": int((mech >= 0.75).sum()),
        "n_disposition_worsened": int(out.disposition_worsened.sum()),
        "n_identified_set_widened": int(out.identified_set_widened.sum()),
        "mean_width_before": float(out.identified_width_before.mean()),
        "mean_width_after": float(out.identified_width_after.mean()),
    }
    Path(a.summary).write_text(json.dumps(summary, indent=2) + "\n")

    pd.set_option("display.width", 200)
    shown = out[["method", "model", "score_before", "score_after",
                 "b_cond_before", "b_cond_after", "mechanical_fraction",
                 "disposition_worsened"]].round(3)
    print(shown.to_string(index=False))
    print()
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {a.out} and {a.summary}")


if __name__ == "__main__":
    main()
