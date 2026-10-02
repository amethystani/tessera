"""Bootstrap CIs and permutation p-values for the someone-vs-named difference in
recovery error.

Estimand per model:
  D = mean|direct - truth| on someone items - mean|direct - truth| on named items
D > 0 means the abstain-permitted condition recovers ground truth worse when
the participant is unnamed. Items are the resampling unit, and the bootstrap is
stratified to keep each stratum's size.

Usage: python3 mechanism_inference.py --glob 'screen_*_per_item.csv'
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["participant"] = (df.sentid.str.replace(".txt", "", regex=False)
                         .str.split(".").str[1])
    df["someone"] = df.participant == "someone"
    df["direct_err"] = (df.direct_aligned - df.truth_aligned).abs()
    df["forced_err"] = (df.forced_aligned - df.truth_aligned).abs()
    return df


def bootstrap_diff(rng, a: np.ndarray, b: np.ndarray, reps: int) -> np.ndarray:
    """Stratified bootstrap of mean(a) - mean(b), resampling items within stratum."""
    out = np.empty(reps)
    for i in range(reps):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        out[i] = sa.mean() - sb.mean()
    return out


def permutation_p(rng, a: np.ndarray, b: np.ndarray, reps: int) -> float:
    """Two-sided permutation test on the stratum label."""
    observed = abs(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    n_a = len(a)
    hits = 0
    for _ in range(reps):
        perm = rng.permutation(pool)
        if abs(perm[:n_a].mean() - perm[n_a:].mean()) >= observed - 1e-12:
            hits += 1
    return hits / reps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="screen_*_per_item.csv")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--out", default="mechanism_inference.json")
    a = ap.parse_args()

    rng = np.random.default_rng(20260917)
    results = []
    for path in sorted(glob.glob(a.glob)):
        df = prepare(path)
        someone = df[df.someone]
        named = df[~df.someone]
        if len(someone) < 2 or len(named) < 2:
            continue

        for metric in ("direct_err", "forced_err"):
            s = np.asarray(someone[metric], dtype=float)
            n = np.asarray(named[metric], dtype=float)
            diff = float(s.mean() - n.mean())
            boot = bootstrap_diff(rng, s, n, a.reps)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            p = permutation_p(rng, s, n, a.reps)
            results.append({
                "file": Path(path).name,
                "metric": metric,
                "n_someone": int(len(s)),
                "n_named": int(len(n)),
                "mean_someone": float(s.mean()),
                "mean_named": float(n.mean()),
                "difference": diff,
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "p_permutation": p,
                "significant_at_05": bool(p < 0.05),
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
            })

    df_out = pd.DataFrame(results)
    for _, r in df_out.iterrows():
        star = "*" if r.significant_at_05 else " "
        print(f"{r.file:45s} {r.metric:10s} "
              f"someone={r.mean_someone:.3f} named={r.mean_named:.3f} "
              f"D={r.difference:+.3f} [{r.ci_lo:+.3f},{r.ci_hi:+.3f}] "
              f"p={r.p_permutation:.3f}{star}")

    Path(a.out).write_text(json.dumps(
        {"reps": a.reps, "seed": 20260917, "results": results}, indent=2) + "\n")
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
