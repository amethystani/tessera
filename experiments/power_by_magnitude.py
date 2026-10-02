"""Detection power of the certificate grouped by each population's own gamma*.

Refines power_simulation.py. Every simulated population gets its own gamma*
(not just the median-disclosure one), and power is estimated only over
populations that are actually incompatible (gamma* > 0), grouped by magnitude.

Each trial draws one shared answer share and independent per-condition
disclosure rates from the empirical pools, then shifts condition 2's answer
share by a per-trial `mix` drawn uniformly from [0, MIX_MAX]. The population
gamma* is computed exactly from the trial's unconditional disclosed-mass
matrix. For K=2 this is max(0, sum_k max_z p[z,k] - 1), which equals the LP in
check_theory.

Compatible populations (gamma* = 0, where disclosure masks the shift) are
excluded from power. Their rejection rate is a false-positive rate and is
reported separately.

CPU only. Usage: python3 power_by_magnitude.py [--trials 60000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from check_theory import compatible, simultaneous_lower_bounds
from draw_count_calibration import N_CONDITIONS, load_pools

DRAW_COUNTS = (16, 32, 64, 128, 256, 1024)
# right-closed bin edges on population gamma*; first bin is the smallest
# detectable-scale violations, last catches everything large.
GAMMA_EDGES = (0.0, 0.02, 0.05, 0.10, 0.20, 1.0)
MIX_MAX = 0.40
ITEMS_PER_SCREEN = 480


def population_gamma(shares, disclose) -> float:
    """Exact gamma* for the K=2 unconditional disclosed-mass matrix."""
    p = np.array([[s * d, (1 - s) * d] for s, d in zip(shares, disclose)])
    return float(max(0.0, p.max(axis=0).sum() - 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60000)
    ap.add_argument("--disclosure-pool", default="lab_results/empirical_disclosure_pool.npy")
    ap.add_argument("--occshare-pool", default="lab_results/empirical_occshare_pool.npy")
    ap.add_argument("--out", default="results_power_by_magnitude.json")
    a = ap.parse_args()
    disclosure_pool, occshare_pool = load_pools(a.disclosure_pool, a.occshare_pool)
    rng = np.random.default_rng(20260930)

    n_bins = len(GAMMA_EDGES) - 1
    n_nd = len(DRAW_COUNTS)
    # per (bin, draw): counts of trials, and rejections by each test
    trials_bin = np.zeros((n_bins, n_nd), dtype=int)
    rej = {k: np.zeros((n_bins, n_nd), dtype=int)
           for k in ("naive", "item", "fw")}
    # compatible (gamma*=0) group, reported as a false-positive consistency check
    compat_trials = np.zeros(n_nd, dtype=int)
    compat_rej = {k: np.zeros(n_nd, dtype=int) for k in ("naive", "item", "fw")}
    fw_alpha = 0.05 / ITEMS_PER_SCREEN

    for _ in range(a.trials):
        occ = float(rng.choice(occshare_pool))
        mix = float(rng.uniform(0.0, MIX_MAX))
        shares = [occ, occ, float(np.clip(occ + mix, 0.0, 1.0))]
        disclose = [float(rng.choice(disclosure_pool)) for _ in range(N_CONDITIONS)]
        g = population_gamma(shares, disclose)

        # one panel across all draw counts reuses the same population
        for j, n in enumerate(DRAW_COUNTS):
            counts = np.zeros((N_CONDITIONS, 2), dtype=int)
            for z in range(N_CONDITIONS):
                probs = np.array([shares[z] * disclose[z],
                                  (1 - shares[z]) * disclose[z], 1 - disclose[z]])
                probs = np.clip(probs, 0, None)
                probs /= probs.sum()
                counts[z] = rng.multinomial(n, probs)[:2]
            totals = np.full(N_CONDITIONS, n)
            r_naive = not compatible(counts / n)
            r_item = not compatible(simultaneous_lower_bounds(counts, totals, alpha=0.05))
            r_fw = not compatible(simultaneous_lower_bounds(counts, totals, alpha=fw_alpha))
            if g <= 0.0:
                compat_trials[j] += 1
                compat_rej["naive"][j] += r_naive
                compat_rej["item"][j] += r_item
                compat_rej["fw"][j] += r_fw
            else:
                b = int(np.searchsorted(GAMMA_EDGES, g, side="left")) - 1
                b = min(max(b, 0), n_bins - 1)
                trials_bin[b, j] += 1
                rej["naive"][b, j] += r_naive
                rej["item"][b, j] += r_item
                rej["fw"][b, j] += r_fw

    def rate(num, den):
        return float(num / den) if den > 0 else None

    bins = []
    for b in range(n_bins):
        lo, hi = GAMMA_EDGES[b], GAMMA_EDGES[b + 1]
        per_draw = []
        for j, n in enumerate(DRAW_COUNTS):
            d = int(trials_bin[b, j])
            per_draw.append({
                "n_draws": n, "n_populations": d,
                "power_naive": rate(rej["naive"][b, j], d),
                "power_corrected_per_item": rate(rej["item"][b, j], d),
                "power_corrected_family_wise": rate(rej["fw"][b, j], d),
            })
        bins.append({"gamma_low": lo, "gamma_high": hi,
                     "label": f"({lo:.2f}, {hi:.2f}]", "by_draw": per_draw})

    compat = [{"n_draws": n, "n_populations": int(compat_trials[j]),
               "fp_naive": rate(compat_rej["naive"][j], compat_trials[j]),
               "fp_corrected_per_item": rate(compat_rej["item"][j], compat_trials[j]),
               "fp_corrected_family_wise": rate(compat_rej["fw"][j], compat_trials[j])}
              for j, n in enumerate(DRAW_COUNTS)]

    out = {
        "description": "Detection power grouped by per-population gamma*. mix ~ "
                       "U[0, %.2f] per trial; population gamma* computed exactly "
                       "from each trial's own disclosed-mass matrix. Power is over "
                       "incompatible populations (gamma*>0) only; the gamma*=0 group "
                       "is reported separately as a false-positive check." % MIX_MAX,
        "trials": a.trials, "n_conditions": N_CONDITIONS,
        "gamma_edges": list(GAMMA_EDGES), "mix_max": MIX_MAX,
        "alpha_per_item": 0.05, "alpha_family_wise": fw_alpha,
        "items_per_screen": ITEMS_PER_SCREEN,
        "incompatible_bins": bins, "compatible_group": compat,
    }
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")

    for b in bins:
        line = "  ".join(
            f"n={d['n_draws']}:item={d['power_corrected_per_item']:.2f}/"
            f"fw={d['power_corrected_family_wise']:.2f}(N={d['n_populations']})"
            if d['power_corrected_per_item'] is not None else f"n={d['n_draws']}:--"
            for d in b["by_draw"])
        print(f"gamma* in {b['label']}: {line}")
    print("\ncompatible group (gamma*=0) FPR, per-item: " +
          "  ".join(f"n={c['n_draws']}:{c['fp_corrected_per_item']:.4f}" for c in compat))
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
