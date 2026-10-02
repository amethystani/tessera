"""False-positive rate of the compatibility certificate as a function of draw
count, under a null built from the real six-checkpoint panel.

The null resamples disclosure rates and answer-share skew from the pooled
empirical distributions of the n=480 sweep (empirical_disclosure_pool.npy with
8,640 observations, empirical_occshare_pool.npy with 8,612), extracted from
screen_*_n480_d16_per_item.csv. A parametric null with disclosure uniform on
[0.55, 0.95] and a Dirichlet(1,1) answer law fits the real panel poorly: 92.5%
of real disclosure rates are above 0.95, and 82.2% of conditional answer
splits are below 0.1 or above 0.9.

Each trial uses one shared answer law across the three conditions and an
independent disclosure rate per condition, so the population is compatible by
construction and every rejection is a false positive. The pools are pooled
across conditions, so which condition a given rate came from is not kept.

Under the parametric null the naive false-positive rate falls from 39.4% at 4
draws to 0.1% at 512. Under the empirical null it does not fall. It rises from
about 17.5% at 4 draws to a plateau around 21-23% that holds out to 4,096
draws. When disclosure is close to 100% in every condition, compatible()
asks three continuous point estimates to coincide exactly, and sampling noise
prevents that at any n. Only an explicit confidence correction fixes it.

Usage: python3 draw_count_calibration.py [--trials 8000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from check_theory import compatible, simultaneous_lower_bounds

DRAW_COUNTS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096)
N_CONDITIONS = 3


def load_pools(disclosure_path: str, occshare_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the empirical disclosure-rate and answer-share pools saved by extract_empirical_pools.py."""
    disclosure = np.load(disclosure_path)
    occshare = np.load(occshare_path)
    return disclosure, occshare


def simulate_panel(rng, n_draws: int, disclosure_pool: np.ndarray,
                   occshare_pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One null replication: ONE shared answer law (bootstrap-resampled from
    real conditional answer-splits), independent per-condition disclosure
    rates (bootstrap-resampled from real disclosure rates). Returns
    (counts [Z x 2], totals [Z])."""
    occ_share = rng.choice(occshare_pool)
    counts = np.zeros((N_CONDITIONS, 2), dtype=int)
    for z in range(N_CONDITIONS):
        disclose = rng.choice(disclosure_pool)
        probs = np.array([occ_share * disclose, (1 - occ_share) * disclose,
                          1 - disclose])
        probs = np.clip(probs, 0, None)
        probs = probs / probs.sum()
        draw = rng.multinomial(n_draws, probs)
        counts[z] = draw[:2]
    return counts, np.full(N_CONDITIONS, n_draws)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--items-per-screen", type=int, default=480,
                    help="n_items assumed for the family-wise regime, matching the real sweep")
    ap.add_argument("--disclosure-pool", default="lab_results/empirical_disclosure_pool.npy")
    ap.add_argument("--occshare-pool", default="lab_results/empirical_occshare_pool.npy")
    ap.add_argument("--out", default="draw_count_calibration.json")
    a = ap.parse_args()

    disclosure_pool, occshare_pool = load_pools(a.disclosure_pool, a.occshare_pool)
    print(f"Loaded {len(disclosure_pool)} disclosure obs, "
          f"{len(occshare_pool)} answer-share obs")

    rng = np.random.default_rng(20260917)
    rows = []
    for n_draws in DRAW_COUNTS:
        naive_fp = 0
        per_item_fp = 0
        familywise_fp = 0
        for _ in range(a.trials):
            counts, totals = simulate_panel(rng, n_draws, disclosure_pool, occshare_pool)

            naive_panel = counts / n_draws
            if not compatible(naive_panel):
                naive_fp += 1

            lo_item = simultaneous_lower_bounds(counts, totals, alpha=0.05)
            if not compatible(lo_item):
                per_item_fp += 1

            lo_fw = simultaneous_lower_bounds(
                counts, totals, alpha=0.05 / a.items_per_screen)
            if not compatible(lo_fw):
                familywise_fp += 1

        rows.append({
            "n_draws": n_draws, "trials": a.trials,
            "naive_false_positive_rate": naive_fp / a.trials,
            "corrected_per_item_false_positive_rate": per_item_fp / a.trials,
            "corrected_familywise_false_positive_rate": familywise_fp / a.trials,
        })
        print(f"n_draws={n_draws:4d}  naive FP={naive_fp/a.trials:6.1%}   "
              f"corrected(per-item) FP={per_item_fp/a.trials:6.1%}   "
              f"corrected(family-wise) FP={familywise_fp/a.trials:6.1%}")

    out = {
        "description": "Null: one shared answer law (resampled from real "
                       "conditional answer-splits), independent per-condition "
                       "disclosure (resampled from real disclosure rates), "
                       "so the population panel is compatible by construction "
                       "and every rejection below is spurious.",
        "seed": 20260917, "n_conditions": N_CONDITIONS, "n_categories": 2,
        "disclosure_pool_size": int(len(disclosure_pool)),
        "occshare_pool_size": int(len(occshare_pool)),
        "items_per_screen_for_familywise": a.items_per_screen,
        "rows": rows,
    }
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
