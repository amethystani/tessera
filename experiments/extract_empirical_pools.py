"""Extract the disclosure-rate and answer-share pools used by
draw_count_calibration.py from the six-checkpoint sweep.

Every (item, condition) pair across the three conditions and six checkpoints is
pooled, so the pools describe disclosure in this kind of experiment generally
and not any one model or condition.

Usage: python3 extract_empirical_pools.py
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

CONDITION_COLUMNS = [
    ("n_truth_occ", "n_truth_part"),
    ("n_direct_occ", "n_direct_part"),
    ("n_forced_occ", "n_forced_part"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="lab_results/screen_*n480*_per_item.csv",
                    help="per-item CSVs to pool (default: the full 480-item panel)")
    a = ap.parse_args()
    disclosure_pool, occshare_pool = [], []
    files = sorted(glob.glob(a.glob))
    assert files, f"no per-item files match {a.glob}"
    for f in files:
        df = pd.read_csv(f)
        for occ_col, part_col in CONDITION_COLUMNS:
            total = df[occ_col] + df[part_col]
            disclosure_pool.extend((total / 16.0).tolist())
            mask = total > 0
            occshare_pool.extend((df[occ_col][mask] / total[mask]).tolist())

    disclosure_pool = np.array(disclosure_pool)
    occshare_pool = np.array(occshare_pool)
    print(f"{len(files)} checkpoints, {len(disclosure_pool)} disclosure obs, "
          f"{len(occshare_pool)} answer-share obs")
    print(f"disclosure: {(disclosure_pool > 0.95).mean():.1%} above 0.95, "
          f"{(disclosure_pool < 0.55).mean():.1%} below 0.55")
    print(f"answer-share: "
          f"{((occshare_pool < 0.1) | (occshare_pool > 0.9)).mean():.1%} "
          f"near-deterministic (<0.1 or >0.9)")

    np.save("lab_results/empirical_disclosure_pool.npy", disclosure_pool)
    np.save("lab_results/empirical_occshare_pool.npy", occshare_pool)
    print("Wrote lab_results/empirical_disclosure_pool.npy and "
          "empirical_occshare_pool.npy")


if __name__ == "__main__":
    main()
