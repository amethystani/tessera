"""Build the Catalan item set from CaBBQ (BSC-LT, CC-BY-4.0).

CaBBQ was written from scratch in Catalan, not translated. It ships as parquet
files per category with the same fields as the original BBQ release, so the
matchers in bbq_firstparty.py (unknown_index, biased_index) work on it as they
are. Output uses the same schema as multilingual_items.csv.

Usage: python3 cabbq_prepare.py [--per-category 200]
Writes cabbq_items.csv and prints per-category match rates.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import urllib.request
from pathlib import Path

import pandas as pd

from bbq_firstparty import biased_index, unknown_index

HERE = Path(__file__).parent
RAW = "https://huggingface.co/datasets/BSC-LT/CaBBQ/resolve/main/{cat}/test-00000-of-00001.parquet"
CATEGORIES = ("Age", "DisabilityStatus", "Gender", "LGBTQIA", "Nationality",
              "PhysicalAppearance", "RaceEthnicity", "Religion", "SES", "SpanishRegion")


def fetch(url: str, tries: int = 5) -> bytes:
    import time
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(4 * (k + 1))
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE / "cabbq_items.csv"))
    ap.add_argument("--report", default=str(HERE / "cabbq_items_report.json"))
    a = ap.parse_args()

    rows, report = [], {"per_category": a.per_category, "categories": {}}
    for cat in CATEGORIES:
        raw = fetch(RAW.format(cat=cat))
        df = pd.read_parquet(io.BytesIO(raw))
        amb = df[df.context_condition == "ambig"]
        usable = []
        for r in amb.itertuples():
            rec = {"answer_info": r.answer_info, "ans0": r.ans0, "ans1": r.ans1,
                  "ans2": r.ans2, "additional_metadata": {"stereotyped_groups": r.stereotyped_groups},
                  "question_polarity": r.question_polarity}
            unk = unknown_index(rec)
            if unk is None:
                continue
            bad = biased_index(rec, unk)
            if bad is None or r.label != unk:
                continue
            usable.append((r, unk, bad))
        report["categories"][cat] = {"ambiguous": int(len(amb)), "matched": len(usable)}
        rng = random.Random(f"{a.seed}-{cat}")
        rng.shuffle(usable)
        chosen = usable[:a.per_category] if a.per_category else usable
        for r, unk, bad in sorted(chosen, key=lambda x: x[0].instance_id):
            rows.append({"dataset": "cabbq", "lang": "ca", "category": cat,
                         "item_id": f"{cat}-{r.instance_id}", "context": r.context,
                         "question": r.question, "a0": r.ans0, "a1": r.ans1, "a2": r.ans2,
                         "unknown_idx": unk, "biased_idx": bad,
                         "polarity": r.question_polarity, "label": r.label,
                         "adaptation": ""})
        print(f"{cat:20s} ambiguous={len(amb):5d} matched={len(usable):5d} "
              f"kept={len(chosen):4d}")

    out = pd.DataFrame(rows)
    assert (out.label == out.unknown_idx).all()
    assert (out.unknown_idx != out.biased_idx).all()
    out.to_csv(a.out, index=False)
    report["total_items"] = int(len(out))
    Path(a.report).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n{len(out)} Catalan items written to {a.out}")


if __name__ == "__main__":
    main()
