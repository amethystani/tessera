"""Detection power of the certificate against a controlled incompatibility, on top
of the null in draw_count_calibration.py.

The calibration results measure false positives only, since the population is
compatible by construction. This measures power, meaning whether a real but
moderate violation is detected and not only the unanimous 16-vs-16 opposition
discussed in the paper.

It reuses draw_count_calibration.simulate_panel for conditions 0 and 1 and
shifts condition 2's answer share by `mix` before disclosure is applied. At
mix=0 this reduces exactly to the existing null, so the measured "power" there
reproduces the ~21% naive and ~0.1% corrected false-positive rates. gamma* of
the median-disclosure population is reported next to each mix as a descriptive
index, computed with check_theory.necessary_dependence.

Usage: python3 power_simulation.py [--trials 8000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from check_theory import compatible, necessary_dependence, simultaneous_lower_bounds
from draw_count_calibration import N_CONDITIONS, load_pools

MIX_LEVELS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.35)
DRAW_COUNTS = (16, 32, 64, 128, 256, 1024)
ITEMS_PER_SCREEN = 480  # matches the real six-checkpoint screen, for the family-wise regime


def simulate_shifted_panel(rng, n_draws, disclosure_pool, occshare_pool, mix):
    occ_share = rng.choice(occshare_pool)
    shares = [occ_share, occ_share, float(np.clip(occ_share + mix, 0.0, 1.0))]
    counts = np.zeros((N_CONDITIONS, 2), dtype=int)
    for z in range(N_CONDITIONS):
        disclose = rng.choice(disclosure_pool)
        probs = np.array([shares[z] * disclose, (1 - shares[z]) * disclose, 1 - disclose])
        probs = np.clip(probs, 0, None)
        probs = probs / probs.sum()
        counts[z] = rng.multinomial(n_draws, probs)[:2]
    return counts, np.full(N_CONDITIONS, n_draws)


def median_population_gamma(disclosure_pool, occshare_pool, mix) -> float:
    d, o = float(np.median(disclosure_pool)), float(np.median(occshare_pool))
    shares = [o, o, float(np.clip(o + mix, 0.0, 1.0))]
    pop = np.array([[s * d, (1 - s) * d] for s in shares])
    if compatible(pop):
        return 0.0
    gamma, *_ = necessary_dependence(pop)
    return float(gamma)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8000)
    ap.add_argument("--disclosure-pool", default="lab_results/empirical_disclosure_pool.npy")
    ap.add_argument("--occshare-pool", default="lab_results/empirical_occshare_pool.npy")
    ap.add_argument("--out", default="results_power_simulation.json")
    a = ap.parse_args()
    disclosure_pool, occshare_pool = load_pools(a.disclosure_pool, a.occshare_pool)
    rng = np.random.default_rng(20260927)

    rows = []
    for mix in MIX_LEVELS:
        gamma_med = median_population_gamma(disclosure_pool, occshare_pool, mix)
        for n_draws in DRAW_COUNTS:
            naive_rej = per_item_rej = fw_rej = 0
            for _ in range(a.trials):
                counts, totals = simulate_shifted_panel(rng, n_draws, disclosure_pool,
                                                        occshare_pool, mix)
                naive_rej += not compatible(counts / n_draws)
                per_item_rej += not compatible(
                    simultaneous_lower_bounds(counts, totals, alpha=0.05))
                fw_rej += not compatible(
                    simultaneous_lower_bounds(counts, totals, alpha=0.05 / ITEMS_PER_SCREEN))
            rows.append({
                "mix": mix, "gamma_median_population": gamma_med, "n_draws": n_draws,
                "trials": a.trials,
                "power_naive": naive_rej / a.trials,
                "power_corrected_per_item": per_item_rej / a.trials,
                "power_corrected_family_wise": fw_rej / a.trials,
            })
        print(f"mix={mix:.2f} gamma*(median pop)={gamma_med:.4f}  " +
              "  ".join(f"n={r['n_draws']}:naive={r['power_naive']:.3f}/"
                        f"item={r['power_corrected_per_item']:.3f}/"
                        f"fw={r['power_corrected_family_wise']:.3f}"
                        for r in rows if r["mix"] == mix))

    out = {
        "description": "Detection power when condition 2's answer share is shifted by "
                       "`mix` from the shared value conditions 0,1 use, layered on the "
                       "exact per-trial resampling of draw_count_calibration.py "
                       "(one shared answer share plus independent, empirically "
                       "resampled per-condition disclosure). mix=0 reproduces the "
                       "existing null exactly, as a consistency check. gamma* is "
                       "reported for the representative median-disclosure, "
                       "median-answer-share population at each mix, computed exactly "
                       "via the LP; individual trials vary around it.",
        "mix_levels": list(MIX_LEVELS), "n_conditions": N_CONDITIONS,
        "items_per_screen_for_family_wise": ITEMS_PER_SCREEN, "rows": rows,
    }
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
