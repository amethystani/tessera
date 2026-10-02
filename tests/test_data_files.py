import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

BBQ = {
    "Age": (3680, 1840), "Disability_status": (1556, 778), "Gender_identity": (5672, 2836),
    "Nationality": (3080, 1540), "Physical_appearance": (1576, 788),
    "Race_ethnicity": (6880, 3440), "Race_x_SES": (11160, 5580),
    "Race_x_gender": (15960, 7980), "Religion": (1200, 600), "SES": (6864, 3432),
    "Sexual_orientation": (864, 432),
}


@pytest.mark.parametrize("category,counts", BBQ.items())
def test_bbq_counts(category, counts):
    rows = [json.loads(line) for line in (ROOT / "data/bbq" / f"{category}.jsonl").open()]
    assert len(rows) == counts[0]
    assert sum(r["context_condition"] == "ambig" for r in rows) == counts[1]


def test_winogender_files():
    sent = pd.read_csv(ROOT / "data/winogender/all_sentences.tsv", sep="\t")
    assert len(sent) == 720
    assert sent.sentid.str.split(".").str[3].value_counts().to_dict() == {
        "male": 240, "female": 240, "neutral": 240}
    assert len(pd.read_csv(ROOT / "data/winogender/occupations-stats.tsv", sep="\t")) == 60


def test_published_table():
    d = pd.read_csv(ROOT / "data/decap_bbq_ambiguous.csv", comment="#")
    assert len(d) == 40
    assert d.method.nunique() == 5
    assert d.model.nunique() == 8


def test_multilingual_item_sets():
    m = pd.read_csv(ROOT / "experiments/multilingual_items.csv")
    assert m.groupby("lang").size().to_dict() == {
        "en": 1076, "es": 1076, "nl": 1076, "tr": 1076, "ko": 1140}
    c = pd.read_csv(ROOT / "experiments/cabbq_items.csv")
    assert len(c) == 1968
    assert c.category.nunique() == 10
