"""Per-model parametric bootstrap: how much of the observed naive incompatibility
is chance?

The false-positive rate of the naive certificate depends on the answer laws a
model's items induce. A model with near-deterministic items produces few
spurious rejections, and one sitting near 50/50 on many items produces many.
So for each real item we take the pooled across-condition answer distribution as
one common law (the null), keep that item's observed per-condition disclosure
rates, and resample. Repeating gives the distribution of naive incompatibility
counts the model's own items would produce with no real cross-condition
disagreement, and the observed count is compared against it.

Usage: python3 model_null_calibration.py --glob 'screen_*_per_item.csv' --reps 2000
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from check_theory import compatible, simultaneous_lower_bounds

CONDITIONS = ("truth", "direct", "forced")


def item_counts_from_row(row) -> tuple[np.ndarray, int]:
    counts = np.array([
        [row.n_truth_occ, row.n_truth_part],
        [row.n_direct_occ, row.n_direct_part],
        [row.n_forced_occ, row.n_forced_part],
    ], dtype=int)
    # Every condition used the same number of draws; recover it from the panel.
    n_draws = int(max(counts.sum(axis=1).max(), 1))
    return counts, n_draws


def null_resample(rng, counts: np.ndarray, n_draws: int) -> np.ndarray:
    """Resample this item's panel under the null: one shared law, observed disclosure."""
    answered = counts.sum(axis=1)
    pooled = counts.sum(axis=0)
    if pooled.sum() == 0:
        law = np.array([0.5, 0.5])
    else:
        law = pooled / pooled.sum()
    out = np.zeros_like(counts)
    for z in range(counts.shape[0]):
        disclose = answered[z] / n_draws
        probs = np.array([law[0] * disclose, law[1] * disclose, 1 - disclose])
        probs = np.clip(probs, 0, None)
        probs = probs / probs.sum()
        out[z] = rng.multinomial(n_draws, probs)[:2]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="screen_*_per_item.csv")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default="model_null_calibration.json")
    a = ap.parse_args()

    rng = np.random.default_rng(20260917)
    results = []
    for path in sorted(glob.glob(a.glob)):
        df = pd.read_csv(path)
        panels = [item_counts_from_row(r) for r in df.itertuples()]
        n_items = len(panels)
        n_draws = panels[0][1]

        observed_naive = int((~df.naive_compatible.astype(bool)).sum())
        observed_corrected = int((~df.corrected_compatible.astype(bool)).sum())

        null_naive_counts = np.zeros(a.reps, dtype=int)
        null_corrected_counts = np.zeros(a.reps, dtype=int)
        alpha_fw = 0.05 / n_items
        for rep in range(a.reps):
            n_bad = 0
            n_bad_corr = 0
            for counts, nd in panels:
                sim = null_resample(rng, counts, nd)
                if not compatible(sim / nd):
                    n_bad += 1
                totals = np.full(sim.shape[0], nd)
                if not compatible(simultaneous_lower_bounds(sim, totals, alpha=alpha_fw)):
                    n_bad_corr += 1
            null_naive_counts[rep] = n_bad
            null_corrected_counts[rep] = n_bad_corr

        # One-sided p: how often does the null produce at least what we observed?
        p_naive = float((null_naive_counts >= observed_naive).mean())
        p_corrected = float((null_corrected_counts >= observed_corrected).mean())

        row = {
            "file": Path(path).name,
            "n_items": n_items,
            "n_draws": n_draws,
            "observed_naive_incompatible": observed_naive,
            "null_naive_mean": float(null_naive_counts.mean()),
            "null_naive_p95": float(np.percentile(null_naive_counts, 95)),
            "p_value_naive": p_naive,
            "observed_corrected_incompatible": observed_corrected,
            "null_corrected_mean": float(null_corrected_counts.mean()),
            "p_value_corrected": p_corrected,
        }
        results.append(row)
        print(f"{Path(path).name:45s} n={n_items:3d} d={n_draws:3d}  "
              f"naive obs={observed_naive:3d} null={null_naive_counts.mean():5.1f} "
              f"p={p_naive:.3f} | corrected obs={observed_corrected:2d} "
              f"null={null_corrected_counts.mean():4.2f} p={p_corrected:.3f}")

    Path(a.out).write_text(json.dumps({"reps": a.reps, "seed": 20260917,
                                       "results": results}, indent=2) + "\n")
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
