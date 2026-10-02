"""Re-score saved BBQ generations with the corrected response parser.

The first parser fell back to the first character of a response when no
standalone option letter appeared, so "Cannot be determined" was scored as
option C whatever C was. bbq_firstparty.parse_response now matches option text
and looks for explicit abstention cues. All raw generations were saved, so this
runs offline with no model calls. The logit-forced condition never parsed text
and is unchanged.

The forced condition's A/B order wasn't stored. It was drawn from
random.Random(4242) over items in file order, so it is regenerated here and
only used after checking it against every stored forced answer whose raw text
holds a standalone letter.

Usage: python3 rescore_bbq.py lab_results/bbq11_*_per_item.csv ...
Rewrites each per-item CSV and its summary JSON in place. The script is
deterministic, so running it twice gives the same files.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

from bbq_firstparty import parse_letter, parse_response, summarize


def rescore(path: Path) -> dict:
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    old = json.loads(path.with_name(path.name.replace("_per_item.csv", ".json")).read_text())
    changed = {}
    for cond in ("baseline", "instructed"):
        new = [parse_response("" if pd.isna(r.get(f"raw_{cond}")) else str(r[f"raw_{cond}"]),
                              [r["a0"], r["a1"], r["a2"]], int(r["unknown_idx"]))
               for _, r in df.iterrows()]
        before = df[f"pred_{cond}"].tolist()
        changed[cond] = sum(1 for x, y in zip(before, new)
                            if not ((pd.isna(x) and y is None) or (not pd.isna(x) and y == int(x))))
        df[f"pred_{cond}"] = pd.array(new, dtype="Int64")
    # Regenerate the forced A/B order and verify it before using it.
    rng = random.Random(4242)
    order = []
    for _, r in df.iterrows():
        others = [i for i in range(3) if i != int(r["unknown_idx"])]
        if rng.random() < 0.5:
            others = others[::-1]
        order.append(others)
    for i, r in df.iterrows():
        raw = "" if pd.isna(r["raw_forced"]) else str(r["raw_forced"])
        letter = parse_letter(raw)
        if letter is not None and letter < 2:
            assert int(r["pred_forced"]) == order[i][letter], f"{path}: forced order mismatch at row {i}"
    newf = []
    for i, r in df.iterrows():
        raw = "" if pd.isna(r["raw_forced"]) else str(r["raw_forced"])
        opts = [r[f"a{j}"] for j in order[i]]
        slot = parse_response(raw, opts, None)
        newf.append(None if slot is None else order[i][slot])
    changed["forced"] = sum(1 for x, y in zip(df.pred_forced.tolist(), newf)
                            if not ((pd.isna(x) and y is None) or (not pd.isna(x) and y == int(x))))
    df["pred_forced"] = pd.array(newf, dtype="Int64")
    summary = summarize(df, old["model"])
    path.with_name(path.name.replace("_per_item.csv", ".json")).write_text(
        json.dumps(summary, indent=2) + "\n")
    df.to_csv(path, index=False)
    return changed


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(Path(p).name, rescore(Path(p)))
