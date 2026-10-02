"""Summarise winogender_parser_audit_sample.csv: agreement between
validation_experiment.parse_response and a separately written classifier,
across all checkpoints and both conditions.

Usage: python3 summarize_winogender_audit.py
Writes results_winogender_parser_audit.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from winogender_parser_audit import independent_parse

HERE = Path(__file__).parent
SAMPLE = HERE / "winogender_parser_audit_sample.csv"
OUT = HERE / "results_winogender_parser_audit.json"


def main() -> None:
    df = pd.read_csv(SAMPLE, keep_default_na=False, na_values=[""])
    df["indep"] = df.apply(
        lambda r: independent_parse(r.raw, r.occupation, r.participant), axis=1)

    def agree(r):
        p = r.parser_outcome
        i = r.indep
        if p == "answered":
            return (r.parser_pred == r.occupation and i == r.occupation) or \
                   (r.parser_pred == r.participant and i == r.participant)
        if p in ("abstained", "empty_response"):
            return i in ("abstained", "empty_response")
        return i == "unparsed"  # p == "unparsed"

    df["agree"] = df.apply(agree, axis=1)
    by_outcome = df.groupby("parser_outcome").agree.agg(["mean", "count"]).to_dict(orient="index")
    by_tag = df.groupby("tag").agree.agg(["mean", "count"]).to_dict(orient="index")
    out = {
        "n_sampled": int(len(df)),
        "n_checkpoints": int(df.tag.nunique()),
        "n_items": int(df.sentid.nunique()),
        "draws_per_item_per_condition": int(len(df) / (df.tag.nunique() * df.sentid.nunique() * 2))
                                        if df.sentid.nunique() else None,
        "agreement_rate_overall": float(df.agree.mean()),
        "agreement_rate_by_outcome": by_outcome,
        "agreement_rate_by_checkpoint": by_tag,
        "n_disagreements": int((~df.agree).sum()),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    dis = df[~df.agree]
    if len(dis):
        print(f"\n{len(dis)} disagreements:")
        for _, r in dis.iterrows():
            print(f"[{r.tag}/{r.condition}] parser={r.parser_outcome}/{r.parser_pred} "
                  f"indep={r.indep}  raw={r.raw!r}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
