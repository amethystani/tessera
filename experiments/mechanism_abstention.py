"""Does abstention concentrate on items whose participant is the unnamed token
"someone"?

mechanism_inference.py tests the downstream effect, that the abstain-permitted
condition recovers ground truth worse on someone items, and finds it for
Qwen2.5 only. A null there is ambiguous: either the mechanism doesn't
generalise, or the checkpoint doesn't abstain enough for the effect to show up.
If the second is right, abstention should still lean toward someone items in
the other models, just at a lower rate.

Estimand per checkpoint: mean per-item non-disclosure rate in the direct
condition on someone items minus on named items, where an item's rate is
1 - (answered draws / 16). Items are the resampling unit. The overall direct
abstention rate is reported next to it since it bounds the effect size.

Usage: python3 mechanism_abstention.py --glob 'screen_*n480*_per_item.csv'
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mechanism_inference import bootstrap_diff, permutation_p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="screen_*n480*_per_item.csv")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--out", default="mechanism_abstention_n480.json")
    a = ap.parse_args()

    rng = np.random.default_rng(20260924)
    results = []
    for path in sorted(glob.glob(a.glob)):
        df = pd.read_csv(path)
        parts = df.sentid.str.replace(".txt", "", regex=False).str.split(".").str[1]
        someone_mask = np.asarray(parts == "someone")
        n_draws = int((df.n_direct_occ + df.n_direct_part).max())
        n_draws = max(n_draws, 16)
        nondisc = 1.0 - (np.asarray(df.n_direct_occ + df.n_direct_part,
                                    dtype=float) / 16.0)
        s, n = nondisc[someone_mask], nondisc[~someone_mask]
        diff = float(s.mean() - n.mean())
        boot = bootstrap_diff(rng, s, n, a.reps)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p = permutation_p(rng, s, n, a.reps)
        row = {
            "file": Path(path).name,
            "n_someone": int(len(s)), "n_named": int(len(n)),
            "overall_abstention_rate": float(nondisc.mean()),
            "abstention_someone": float(s.mean()),
            "abstention_named": float(n.mean()),
            "difference": diff, "ci_lo": float(lo), "ci_hi": float(hi),
            "p_permutation": p,
            "ratio_someone_over_named": (float(s.mean() / n.mean())
                                         if n.mean() > 0 else None),
        }
        results.append(row)
        ratio = (f"{row['ratio_someone_over_named']:.1f}x"
                 if row["ratio_someone_over_named"] else "n/a")
        print(f"{row['file']:44s} overall={row['overall_abstention_rate']*100:5.2f}%  "
              f"someone={row['abstention_someone']*100:5.2f}% "
              f"named={row['abstention_named']*100:5.2f}% ({ratio})  "
              f"D={diff*100:+5.2f}pp [{lo*100:+.2f},{hi*100:+.2f}] p={p:.4f}")

    # Holm step-down across the checkpoints that abstain at all. A checkpoint
    # with zero abstention has no defined lean (0 vs 0) and is not a test, so it
    # is excluded from the family rather than counted as a trivially
    # non-significant one.
    testable = [r for r in results if r["overall_abstention_rate"] > 0]
    order = sorted(range(len(testable)), key=lambda i: testable[i]["p_permutation"])
    m = len(testable)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * testable[i]["p_permutation"])
        running = max(running, adj)
        testable[i]["p_holm"] = running
    for r in results:
        r.setdefault("p_holm", None)
    n_sig = sum(1 for r in testable if r["p_holm"] < 0.05 and r["difference"] > 0)
    print(f"\nHolm-adjusted across {m} abstaining checkpoints: "
          f"{n_sig} significant (positive lean)")
    for r in testable:
        print(f"  {r['file'][:34]:36s} p={r['p_permutation']:.4f} "
              f"holm={r['p_holm']:.4f}")

    Path(a.out).write_text(json.dumps(
        {"reps": a.reps, "seed": 20260924, "n_testable": m,
         "n_significant_holm": n_sig, "results": results}, indent=2) + "\n")
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
