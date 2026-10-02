"""Build the non-English item sets for the multilingual runs.

Two BBQ-format benchmarks, both with BBQ's ambiguous-context design and an
explicit UNKNOWN option:

  MBBQ   (Neplenbroek et al., 2024; CC-BY-4.0)  English, Dutch, Spanish, Turkish
  KoBBQ  (Jin et al., 2024; MIT)  Korean, culturally adapted rather than
         translated. label_annotation records whether an item was transferred
         (ST), target-modified (TM) or newly created (NC).

MBBQ's translated files keep BBQ's English group codes in answer_info for every
category except Gender_identity, where the tags were translated ("kadin",
"mujer", "vrouw"). They are also not item-parallel across languages: the same
example_id can be a different question, and answer options get reordered. Each
language file is therefore scored on its own tags with the English matcher,
after mapping the 12 translated Gender_identity tags to their English
equivalents. Items whose stereotyped group is not among the options are dropped
and counted.

Sampling: up to --per-category ambiguous MBBQ items per category and language
(seeded). KoBBQ uses every ambiguous item in the evaluation split.

Output: multilingual_items.csv and multilingual_items_report.json
Usage: python3 multilingual_prepare.py [--per-category 200]
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import random
import urllib.request
from pathlib import Path

import pandas as pd

from bbq_firstparty import biased_index, unknown_index

HERE = Path(__file__).parent
MBBQ_TREE = "https://api.github.com/repos/Veranep/MBBQ/git/trees/main?recursive=1"
MBBQ_RAW = "https://raw.githubusercontent.com/Veranep/MBBQ/main/"
KOBBQ = ("https://huggingface.co/datasets/naver-ai/kobbq/resolve/main/"
         "data/kobbq-evaluation-test.parquet")
LANGS = ("en", "es", "nl", "tr")
# Translated Gender_identity group tags -> the English words the BBQ matcher knows.
TAG_EN = {"mujer": "woman", "hombre": "man", "chica": "girl", "chico": "boy",
          "vrouw": "woman", "man": "man", "meisje": "girl", "jongen": "boy",
          "kad\u0131n": "woman", "adam": "man", "k\u0131z": "girl", "erkek \u00e7ocuk": "boy"}


def fetch(url: str, tries: int = 6) -> bytes:
    import time
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))
    raise RuntimeError("unreachable")


def mbbq(per_category: int, seed: int, report: dict) -> list[dict]:
    tree = json.loads(fetch(MBBQ_TREE))["tree"]
    files = [t["path"] for t in tree if t["path"].startswith("data/")
             and t["path"].endswith(".jsonl") and "_control_" not in t["path"]]
    cats = sorted({Path(f).stem.rsplit("_", 1)[0] for f in files})
    report["mbbq_categories"] = cats
    rows = []
    for cat in cats:
        for lang in LANGS:
            path = f"data/{cat}_{lang}.jsonl"
            if path not in files:
                continue
            recs = [r for r in map(json.loads, filter(None, fetch(MBBQ_RAW + path)
                    .decode("utf-8").splitlines())) if r["context_condition"] == "ambig"]
            usable = []
            for r in recs:
                for i in range(3):
                    tag = str(r["answer_info"][f"ans{i}"][1]).strip().lower()
                    if tag in TAG_EN:
                        r["answer_info"][f"ans{i}"][1] = TAG_EN[tag]
                unk = unknown_index(r)
                bad = biased_index(r, unk) if unk is not None else None
                if unk is not None and bad is not None and r["label"] == unk:
                    usable.append((r, unk, bad))
            report.setdefault("mbbq_matching", {}).setdefault(cat, {})[lang] = {
                "ambiguous": len(recs), "matched": len(usable)}
            rng = random.Random(f"{seed}-{cat}-{lang}")
            rng.shuffle(usable)
            for r, unk, bad in sorted(usable[:per_category] if per_category else usable,
                                      key=lambda x: x[0]["example_id"]):
                rows.append({"dataset": "mbbq", "lang": lang, "category": cat,
                             "item_id": f"{cat}-{r['example_id']}", "context": r["context"],
                             "question": r["question"], "a0": r["ans0"], "a1": r["ans1"],
                             "a2": r["ans2"], "unknown_idx": unk, "biased_idx": bad,
                             "polarity": r["question_polarity"], "label": r["label"],
                             "adaptation": ""})
    return rows


def kobbq(report: dict) -> list[dict]:
    df = pd.read_parquet(io.BytesIO(fetch(KOBBQ)))
    amb = df[df.sample_id.str.contains("-amb-")]
    rows, dropped = [], 0
    for r in amb.itertuples():
        choices = ast.literal_eval(r.choices)
        if len(choices) != 3 or r.answer not in choices or r.biased_answer not in choices:
            dropped += 1
            continue
        unk, bad = choices.index(r.answer), choices.index(r.biased_answer)
        if unk == bad:
            dropped += 1
            continue
        rows.append({"dataset": "kobbq", "lang": "ko", "category": r.bbq_category,
                     "item_id": r.sample_id, "context": r.context, "question": r.question,
                     "a0": choices[0], "a1": choices[1], "a2": choices[2],
                     "unknown_idx": unk, "biased_idx": bad, "polarity": "",
                     "label": unk, "adaptation": r.label_annotation})
    report["kobbq"] = {"ambiguous": int(len(amb)), "kept": len(rows), "dropped": dropped}
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    report: dict = {"per_category": a.per_category, "seed": a.seed}
    rows = mbbq(a.per_category, a.seed, report) + kobbq(report)
    out = pd.DataFrame(rows)
    # Gold on an ambiguous item must be UNKNOWN, and UNKNOWN is never the biased slot.
    assert (out.label == out.unknown_idx).all()
    assert (out.unknown_idx != out.biased_idx).all()
    counts = out.groupby(["dataset", "lang"]).size().to_dict()
    report["items"] = {f"{k[0]}/{k[1]}": int(v) for k, v in counts.items()}
    out.to_csv(HERE / "multilingual_items.csv", index=False)
    (HERE / "multilingual_items_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
