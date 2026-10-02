"""Classify certified-incompatible items by which condition is the odd one out.

The certificate says the three conditions can't share one answer law, but not
which condition is responsible, and that matters:

  instrument_failure   ground truth (refusal-ablated) is the outlier while the
                       two intact conditions agree. The ablation changed the
                       model's answer, so the ground-truth condition can't be
                       trusted on that item.
  policy_effect        direct (abstain-permitted) is the outlier while forced
                       and ablated-forced agree. The instrument is fine and the
                       response policy changed the answer.
  forced_outlier       forced is the outlier while direct and ablated agree.
                       Unexpected under either reading, so flagged.
  no_majority          all three differ, or a condition had no parseable answers.

Usage: python3 violation_taxonomy.py --glob 'screen_*_per_item.csv'
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

CONDITION_COLUMNS = {
    "truth": ("n_truth_occ", "n_truth_part"),
    "direct": ("n_direct_occ", "n_direct_part"),
    "forced": ("n_forced_occ", "n_forced_part"),
}


def dominant(row, condition: str) -> str | None:
    occ_col, part_col = CONDITION_COLUMNS[condition]
    occ, part = int(row[occ_col]), int(row[part_col])
    if occ == 0 and part == 0:
        return None          # condition produced no parseable answer at all
    if occ == part:
        return None          # exact tie: no dominant category
    return "occupation" if occ > part else "participant"


def classify(row) -> str:
    labels = {c: dominant(row, c) for c in CONDITION_COLUMNS}
    if any(v is None for v in labels.values()):
        return "no_majority"
    truth, direct, forced = labels["truth"], labels["direct"], labels["forced"]
    if truth == direct == forced:
        return "all_agree"           # incompatible on mass, not on direction
    if direct == forced and truth != direct:
        return "instrument_failure"
    if truth == forced and direct != forced:
        return "policy_effect"
    if truth == direct and forced != truth:
        return "forced_outlier"
    return "no_majority"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="screen_*_per_item.csv")
    ap.add_argument("--out", default="violation_taxonomy.json")
    a = ap.parse_args()

    summary = []
    for path in sorted(glob.glob(a.glob)):
        df = pd.read_csv(path)
        bad = df[~df.corrected_compatible.astype(bool)].copy()
        if len(bad) == 0:
            summary.append({"file": Path(path).name, "n_items": int(len(df)),
                            "n_corrected_incompatible": 0, "counts": {}})
            print(f"{Path(path).name:45s} n={len(df):3d}  no corrected violations")
            continue
        bad["pattern"] = [classify(r) for _, r in bad.iterrows()]
        counts = bad.pattern.value_counts().to_dict()
        summary.append({
            "file": Path(path).name,
            "n_items": int(len(df)),
            "n_corrected_incompatible": int(len(bad)),
            "counts": {k: int(v) for k, v in counts.items()},
            "items": [{"sentid": r.sentid, "pattern": r.pattern,
                       "gamma_star": float(r.corrected_gamma_star)}
                      for _, r in bad.iterrows()],
        })
        print(f"{Path(path).name:45s} n={len(df):3d}  "
              f"violations={len(bad):2d}  {counts}")
        for _, r in bad.iterrows():
            print(f"    {r.pattern:20s} {r.sentid}")

    Path(a.out).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
