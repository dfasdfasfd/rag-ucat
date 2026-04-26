"""Tests for subtype-targeted bulk generation. Runnable directly:

    ./venv/bin/python tests/test_subtype_targeting.py

Each function with a name starting `test_` is run; failures raise.
"""
from __future__ import annotations

import os
import sys

# Make the project importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ucat.models import Question


def _make_question(**overrides):
    """Build a minimal valid Question for tests."""
    base = {
        "number": 1,
        "text": "Sample question?",
        "options": [
            {"label": "A", "text": "first"},
            {"label": "B", "text": "second"},
        ],
        "answer": "A",
        "explanation": "because.",
        "difficulty": 3.0,
        "coverage": {
            "topic": "general",
            "scenario_type": "everyday",
            "contains_named_entities": False,
            "cultural_context": None,
        },
    }
    base.update(overrides)
    return Question.model_validate(base)


# ─── Schema: minigame_kind ───────────────────────────────────────────────────

def test_question_accepts_minigame_kind_string():
    q = _make_question(minigame_kind="main-idea")
    assert q.minigame_kind == "main-idea"


def test_question_minigame_kind_defaults_to_none():
    q = _make_question()
    assert q.minigame_kind is None


def test_legacy_question_without_minigame_kind_still_validates():
    """Legacy KB rows have no minigame_kind; the field is Optional so they
    must continue to deserialise cleanly."""
    raw = {
        "number": 1,
        "text": "Legacy question",
        "options": [
            {"label": "A", "text": "first"},
            {"label": "B", "text": "second"},
        ],
        "answer": "A",
        "explanation": "x",
        "difficulty": 2.0,
        "coverage": {
            "topic": "ecology", "scenario_type": "scientific",
            "contains_named_entities": False, "cultural_context": None,
        },
    }
    q = Question.model_validate(raw)
    assert q.minigame_kind is None


# ─── Config: SET_SIZES + SUBTYPES_BY_SECTION ─────────────────────────────────

from ucat.config import SET_SIZES, SUBTYPES_BY_SECTION


def test_set_sizes_match_section_models():
    """SET_SIZES must match the min/max question count baked into the Pydantic
    section models so the ceil math in the bulk worker stays accurate."""
    expected = {"VR": 4, "DM": 5, "QR": 4, "AR": 5}
    assert SET_SIZES == expected, f"expected {expected}, got {SET_SIZES}"


def test_subtypes_dm_has_five_named_subtypes():
    values = [v for v, _ in SUBTYPES_BY_SECTION["DM"]]
    assert values == ["syllogism", "logical", "venn", "probability", "argument"], \
        f"unexpected DM subtypes: {values}"


def test_subtypes_vr_has_five_minigame_kinds():
    values = [v for v, _ in SUBTYPES_BY_SECTION["VR"]]
    assert values == ["tfc", "main-idea", "paraphrase", "tone-purpose", "inference"], \
        f"unexpected VR subtypes: {values}"


def test_subtypes_qr_chart_types_match_schema_enum():
    values = [v for v, _ in SUBTYPES_BY_SECTION["QR"]]
    assert values == ["table", "bar", "line", "stacked_bar", "pie"], \
        f"unexpected QR subtypes: {values}"


def test_subtypes_ar_is_empty_list():
    """AR has no subtype targeting in this spec; the empty list signals the
    UI to disable the dropdown."""
    assert SUBTYPES_BY_SECTION["AR"] == [], \
        f"expected AR to have no subtypes, got {SUBTYPES_BY_SECTION['AR']}"


def test_subtypes_all_entries_are_value_label_tuples():
    for section, entries in SUBTYPES_BY_SECTION.items():
        for entry in entries:
            assert isinstance(entry, tuple) and len(entry) == 2, \
                f"{section}: bad entry {entry!r}"
            value, label = entry
            assert isinstance(value, str) and value, f"{section}: empty value"
            assert isinstance(label, str) and label, f"{section}: empty label"


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
            except Exception as e:
                failures += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    if failures:
        print(f"\n{failures} test(s) failed")
        sys.exit(1)
    print("\nAll tests passed.")
