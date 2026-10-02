"""Re-derive each stored BBQ summary from its per-item CSV.

Needs no model or GPU. It checks that the saved per-item outputs and the saved summaries agree, so
a broken parser, scorer or summary function shows up here before a long run does.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

import bbq_firstparty as bbq

LAB = Path(__file__).resolve().parent.parent / "experiments" / "lab_results"
TAGS = sorted(p.name[: -len("_per_item.csv")] for p in LAB.glob("bbq11_*_per_item.csv")
              if not p.name.startswith("bbq11para"))


def close(a, b, path=""):
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            close(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12), path
    else:
        assert a == b, path


def test_there_are_stored_runs():
    assert len(TAGS) == 12


@pytest.mark.parametrize("tag", TAGS)
def test_summary_matches_per_item_csv(tag):
    df = pd.read_csv(LAB / f"{tag}_per_item.csv", keep_default_na=False, na_values=[""])
    stored = json.loads((LAB / f"{tag}.json").read_text())
    fresh = json.loads(json.dumps(bbq.summarize(df, stored["model"]), default=float))
    close(fresh, stored)


@pytest.mark.parametrize("tag", TAGS)
def test_per_item_file_is_complete(tag):
    df = pd.read_csv(LAB / f"{tag}_per_item.csv", keep_default_na=False, na_values=[""])
    assert len(df) == 2200
    assert df.category.nunique() == 11
    assert (df.category.value_counts() == 200).all()
