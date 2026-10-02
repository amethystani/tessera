"""Two robustness checks on the 32-pair published-table decomposition. Neither
needs new model runs.

(1) Symmetric decomposition. decap_decomposition.py uses the first-order split
        s1 - s0 = b0*(c1-c0) + c1*(b1-b0)
    which evaluates coverage at the before bias and disposition at the after
    coverage. The symmetric split
        s1 - s0 = [(b0+b1)/2]*(c1-c0) + [(c0+c1)/2]*(b1-b0)
    is also exact and favours neither endpoint. We recompute every pair with it
    and check whether the median coverage share or any pair's conclusion
    (disposition worsened or improved) changes.

(2) Rounding sensitivity. Published accuracy and bias score are printed to two
    decimals, so each true value lies within 0.005 of the printed one. We redraw
    all four inputs per pair uniformly in that window, recompute the original
    decomposition, and repeat 20,000 times per pair. Per pair we report how often
    disposition_worsened or identified_set_widened flips relative to the printed
    values, and the distribution of the median mechanical fraction.

Usage: python3 decomposition_robustness.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from decap_decomposition import BASELINE, decompose

DATA = "data/decap_bbq_ambiguous.csv"
OUT = "results/decap_decomposition_robustness.json"
ROUNDING_HALF_WIDTH = 0.005  # printed to 2 dp
N_DRAWS = 20_000


def symmetric_decompose(acc0, s0, acc1, s1) -> dict:
    """Split s1 - s0 into coverage and disposition using the average of the two endpoints. Accuracy is in percent."""
    p0, p1 = acc0 / 100.0, acc1 / 100.0
    c0, c1 = 1.0 - p0, 1.0 - p1
    b0 = s0 / c0 if c0 > 0 else float("nan")
    b1 = s1 / c1 if c1 > 0 else float("nan")
    delta = s1 - s0
    coverage = 0.5 * (b0 + b1) * (c1 - c0)
    disposition = 0.5 * (c0 + c1) * (b1 - b0)
    mech = coverage / delta if abs(delta) > 1e-12 else float("nan")
    return {"coverage_component_sym": coverage, "disposition_component_sym": disposition,
            "mechanical_fraction_sym": mech}


def pairs():
    """Yield every method/model pair with its baseline accuracy and bias score, as printed in the published table."""
    df = pd.read_csv(DATA, comment="#")
    base = df[df.method == BASELINE].set_index("model")
    for _, r in df[df.method != BASELINE].iterrows():
        b = base.loc[r["model"]]
        yield dict(method=r["method"], model=r["model"],
                   acc0=float(b["accuracy"]), s0=float(b["bias_score"]),
                   acc1=float(r["accuracy"]), s1=float(r["bias_score"]))


def main() -> None:
    rows = []
    for pr in pairs():
        base = decompose(pr["acc0"], pr["s0"], pr["acc1"], pr["s1"])
        sym = symmetric_decompose(pr["acc0"], pr["s0"], pr["acc1"], pr["s1"])
        rows.append({**pr, **base, **sym})
    df = pd.DataFrame(rows)

    improved = df[df.reported_improvement]
    mech_o = improved.mechanical_fraction.dropna()
    mech_s = improved.mechanical_fraction_sym.dropna()
    med_orig = float(mech_o.median())
    med_sym = float(mech_s.median())
    sign_flips = int((np.sign(df.coverage_component) != np.sign(df.coverage_component_sym)).sum())
    # disposition_worsened is sign(b1-b0); both conventions weight the
    # disposition term by a strictly positive coverage rate (c1, or (c0+c1)/2),
    # so this qualitative flag is convention-independent BY CONSTRUCTION, not
    # empirically - it is reported to make that explicit, not as a live check.
    disposition_flag_is_convention_independent = bool(
        ((df.disposition_component > 0) == (df.disposition_component_sym > 0)).all()
        and ((df.disposition_component < 0) == (df.disposition_component_sym < 0)).all())

    # --- Rounding sensitivity -----------------------------------------------
    rng = np.random.default_rng(20260927)
    all_medians = []
    plist = list(pairs())  # read once; re-reading the CSV per replication would be wasteful
    worsened0 = {}
    widened0 = {}
    mech0 = {}
    for pr in plist:
        key = (pr["method"], pr["model"])
        base = decompose(pr["acc0"], pr["s0"], pr["acc1"], pr["s1"])
        worsened0[key] = base["disposition_worsened"]
        widened0[key] = base["identified_set_widened"]
        mech0[key] = base["mechanical_fraction"]

    flip_worsened = {k: 0 for k in worsened0}
    flip_widened = {k: 0 for k in widened0}
    for _ in range(N_DRAWS):
        u = rng.uniform(-ROUNDING_HALF_WIDTH, ROUNDING_HALF_WIDTH, size=(len(plist), 4))
        mechs = []
        for (pr, du) in zip(plist, u):
            key = (pr["method"], pr["model"])
            acc0, s0, acc1, s1 = (pr["acc0"] + du[0], pr["s0"] + du[1],
                                  pr["acc1"] + du[2], pr["s1"] + du[3])
            d = decompose(acc0, s0, acc1, s1)
            if d["disposition_worsened"] != worsened0[key]:
                flip_worsened[key] += 1
            if d["identified_set_widened"] != widened0[key]:
                flip_widened[key] += 1
            if d["reported_improvement"] and not np.isnan(d["mechanical_fraction"]):
                mechs.append(d["mechanical_fraction"])
        if mechs:
            all_medians.append(float(np.median(mechs)))

    median_ci = [float(np.percentile(all_medians, 2.5)), float(np.percentile(all_medians, 97.5))]
    flip_rate_worsened = {f"{m}/{mo}": v / N_DRAWS for (m, mo), v in flip_worsened.items() if v > 0}
    flip_rate_widened = {f"{m}/{mo}": v / N_DRAWS for (m, mo), v in flip_widened.items() if v > 0}

    out = {
        "symmetric_decomposition": {
            "median_mechanical_fraction_original": med_orig,
            "median_mechanical_fraction_symmetric": med_sym,
            "mean_mechanical_fraction_original": float(mech_o.mean()),
            "mean_mechanical_fraction_symmetric": float(mech_s.mean()),
            "absolute_difference_in_median": abs(med_orig - med_sym),
            "n_pairs_coverage_sign_differs": sign_flips,
            "n_at_least_90pct_mechanical_original": int((mech_o >= 0.90).sum()),
            "n_at_least_90pct_mechanical_symmetric": int((mech_s >= 0.90).sum()),
            "n_at_least_75pct_mechanical_original": int((mech_o >= 0.75).sum()),
            "n_at_least_75pct_mechanical_symmetric": int((mech_s >= 0.75).sum()),
            "n_at_least_50pct_mechanical_original": int((mech_o >= 0.50).sum()),
            "n_at_least_50pct_mechanical_symmetric": int((mech_s >= 0.50).sum()),
            "disposition_worsened_flag_convention_independent": disposition_flag_is_convention_independent,
        },
        "rounding_sensitivity": {
            "rounding_half_width": ROUNDING_HALF_WIDTH, "n_draws": N_DRAWS,
            "median_mechanical_fraction_point_estimate": float(pd.Series(mech0).dropna().median()),
            "median_mechanical_fraction_mc_95ci": median_ci,
            "pairs_with_nonzero_disposition_worsened_flip_rate": flip_rate_worsened,
            "pairs_with_nonzero_identified_set_widened_flip_rate": flip_rate_widened,
            "max_flip_rate_disposition_worsened": max(flip_rate_worsened.values(), default=0.0),
            "max_flip_rate_identified_set_widened": max(flip_rate_widened.values(), default=0.0),
        },
    }
    Path(OUT).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
