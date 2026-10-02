"""Audit the WinoGender parser (validation_experiment.parse_response) against
saved raw generations.

natural_model_screen.py keeps only aggregated per-item counts, so for this audit
we draw fresh completions under the DIRECT and FORCED templates, save every raw
response, and compare parse_response with a separately written classifier
(fuzzy whole-response matching against the occupation and participant strings,
plus its own abstention-cue list). The ground-truth condition scores draws with
the same parse_response call, so checking the function covers all three
conditions.

Usage: python3 winogender_parser_audit.py --model <hf id> --tag <short name>
Appends to winogender_parser_audit_sample.csv. Run it once per checkpoint, then
summarize_winogender_audit.py reports the aggregate.
"""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from validation_experiment import DIRECT, FORCED, query, parse_response  # noqa: E402

HERE = Path(__file__).parent
ITEMS = HERE.parent / "data" / "winogender" / "all_sentences.tsv"
STATS = HERE.parent / "data" / "winogender" / "occupations-stats.tsv"
OUT = HERE / "winogender_parser_audit_sample.csv"
N_ITEMS = 80
DRAWS_PER_ITEM = 4


def independent_parse(resp: str, occupation: str, participant: str) -> str | None:
    """Deliberately different algorithm from parse_response."""
    text = (resp or "").strip().lower()
    if not text:
        return "empty_response"
    cues = ("cannot", "can't", "can not", "unclear", "not clear", "ambiguous",
            "no way to", "not possible to", "insufficient", "not enough",
            "undetermined", "determine")
    if any(c in text for c in cues):
        return "abstained"
    scores = {occupation: difflib.SequenceMatcher(None, text, occupation).ratio(),
             participant: difflib.SequenceMatcher(None, text, participant).ratio()}
    for cand in (occupation, participant):
        if cand in text:
            scores[cand] = max(scores[cand], 0.9)
    best = max(scores, key=scores.get)
    other = participant if best == occupation else occupation
    if scores[best] >= 0.5 and scores[best] - scores[other] > 0.05:
        return best
    return "unparsed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-items", type=int, default=N_ITEMS)
    ap.add_argument("--draws", type=int, default=DRAWS_PER_ITEM)
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sent = pd.read_csv(ITEMS, sep="\t")
    stats = pd.read_csv(STATS, sep="\t")
    parts = sent.sentid.str.replace(".txt", "", regex=False).str.split(".")
    sent["occupation"], sent["participant"] = parts.str[0], parts.str[1]
    sent["gender"] = parts.str[3]
    sent = sent[sent.gender != "neutral"].merge(stats, on="occupation")
    sent = sent.sample(n=min(a.n_items, len(sent)), random_state=7).reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None)
    model.eval()

    rows = []
    for cond_name, template in (("direct", DIRECT), ("forced", FORCED)):
        for i, r in enumerate(sent.itertuples()):
            for d in range(a.draws):
                resp = query(model, tok, template.format(
                    sentence=r.sentence, a=r.occupation, b=r.participant),
                    temperature=0.8, seed=5000 + i * 100 + d)
                outcome, pred = parse_response(resp, r.occupation, r.participant)
                rows.append({"tag": a.tag, "condition": cond_name, "sentid": r.sentid,
                             "occupation": r.occupation, "participant": r.participant,
                             "raw": resp, "parser_outcome": outcome, "parser_pred": pred})
        print(f"[{cond_name}] {len(sent)} items x {a.draws} draws done")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
    print(f"Appended {len(df)} rows to {OUT}")


if __name__ == "__main__":
    main()
