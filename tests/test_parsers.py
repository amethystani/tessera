import pytest

import bbq_firstparty as bbq
import validation_experiment as wg

OPTIONS = ["The boy", "The girl", "Cannot be determined"]


@pytest.mark.parametrize("resp,expected", [
    ("A", 0), ("b", 1), ("C.", 2), ("Answer: B", 1), ("(c)", 2),
    ("", None), (None, None), ("Because", None), ("D", None),
])
def test_parse_letter(resp, expected):
    assert bbq.parse_letter(resp) == expected


def test_cannot_be_determined_is_not_scored_as_the_first_character():
    # the old parser read the leading "C" as option C regardless of what C was
    options = ["Alice", "Bob", "Cannot be determined"]
    assert bbq.parse_response("Cannot be determined", options, unknown=2) == 2
    options = ["Cannot be determined", "Alice", "Bob"]
    assert bbq.parse_response("Cannot be determined", options, unknown=0) == 0


@pytest.mark.parametrize("resp", [
    "It cannot be determined from the text.",
    "I can't say.",
    "Not enough information",
    "unknown",
])
def test_abstention_phrases_map_to_the_unknown_option(resp):
    assert bbq.parse_response(resp, OPTIONS, unknown=2) == 2


def test_option_text_is_matched_in_other_languages():
    options = ["Kadın", "Erkek", "Bilinmeyen"]
    assert bbq.parse_response("Bilinmeyen", options, unknown=2) == 2


def test_truncated_refusal_is_left_unparsed():
    assert bbq.parse_response("I'm sorry, but", OPTIONS, unknown=2) is None
    assert bbq.parse_response("To solve this problem,", OPTIONS, unknown=2) is None
    assert bbq.parse_response("", OPTIONS, unknown=2) is None


def test_letter_beyond_the_offered_options_is_unparsed():
    # forced condition offers two options only
    assert bbq.parse_response("C", ["The boy", "The girl"], unknown=None) is None


@pytest.mark.parametrize("resp,outcome,pred", [
    ("nurse", "answered", "nurse"),
    ("the patient.", "answered", "patient"),
    ("cannot determine", "abstained", None),
    ("it is hard to determine", "abstained", None),
    ("", "empty_response", None),
    ("she", "unparsed", None),
    ("barrister", "unparsed", None),
])
def test_winogender_parser(resp, outcome, pred):
    assert wg.parse_response(resp, "nurse", "patient") == (outcome, pred)


def test_winogender_abstention_wins_over_a_named_option():
    assert wg.parse_response("cannot determine, maybe the nurse", "nurse", "patient") == (
        "abstained", None)
