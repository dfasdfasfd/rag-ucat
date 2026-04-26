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


# ─── Settings: new bulk subtype keys ─────────────────────────────────────────

import tempfile

from ucat.config import Settings


def test_settings_defaults_include_bulk_subtype():
    s = Settings(path=tempfile.mktemp(suffix=".json"))
    assert s.get("bulk_subtype") == "", \
        f"expected '', got {s.get('bulk_subtype')!r}"


def test_settings_defaults_include_per_section_subtype_memory():
    s = Settings(path=tempfile.mktemp(suffix=".json"))
    by_section = s.get("bulk_subtype_by_section")
    assert by_section == {"VR": "", "DM": "", "QR": "", "AR": ""}, \
        f"unexpected default: {by_section!r}"


def test_settings_defaults_include_quantity_unit():
    s = Settings(path=tempfile.mktemp(suffix=".json"))
    assert s.get("bulk_quantity_unit") == "sets", \
        f"expected 'sets', got {s.get('bulk_quantity_unit')!r}"


def test_settings_subtype_persists_via_save_load():
    """Confirm the new keys survive a save/load roundtrip."""
    path = tempfile.mktemp(suffix=".json")
    s = Settings(path=path)
    s.set("bulk_subtype", "venn")
    s.set("bulk_subtype_by_section",
          {"VR": "main-idea", "DM": "venn", "QR": "bar", "AR": ""})
    s.set("bulk_quantity_unit", "questions")

    s2 = Settings(path=path)
    assert s2.get("bulk_subtype") == "venn"
    assert s2.get("bulk_subtype_by_section")["DM"] == "venn"
    assert s2.get("bulk_subtype_by_section")["VR"] == "main-idea"
    assert s2.get("bulk_quantity_unit") == "questions"


# ─── _system_blocks: subtype overrides ───────────────────────────────────────

from ucat.config import Settings as _S
from ucat.rag import RAGEngine


class _StubDB:
    """Minimal DB stand-in — _system_blocks doesn't query the DB."""
    pass


def _engine() -> RAGEngine:
    """Build a RAGEngine with stubbed DB and an in-memory Settings."""
    s = _S(path=tempfile.mktemp(suffix=".json"))
    return RAGEngine(_StubDB(), s)


def _block_text(blocks) -> str:
    """Concatenate every text segment in the system_blocks list."""
    return "\n".join(b.get("text", "") for b in blocks)


def test_system_blocks_dm_subtype_locks_to_one_kind():
    eng = _engine()
    blocks = eng._system_blocks("DM", retrieved=[], target_difficulty=3.0,
                                  subtype="venn")
    text = _block_text(blocks)
    assert "All 5 questions MUST be venn" in text, \
        f"missing venn lock-in:\n{text}"
    # The "include one of each" guidance must be suppressed when subtype is set.
    assert "one of each" not in text, \
        "expected 'one of each' to be suppressed when subtype is set"


def test_system_blocks_dm_no_subtype_keeps_variety_guidance():
    """Backwards compat: with no subtype, the existing variety guidance stays."""
    eng = _engine()
    blocks = eng._system_blocks("DM", retrieved=[], target_difficulty=3.0,
                                  subtype=None)
    text = _block_text(blocks)
    assert "one of each" in text, \
        f"expected variety guidance to be present:\n{text}"


def test_system_blocks_dm_venn_includes_kind_specific_reminder():
    eng = _engine()
    blocks = eng._system_blocks("DM", retrieved=[], target_difficulty=3.0,
                                  subtype="venn")
    text = _block_text(blocks)
    assert "structured `venn` field" in text, \
        f"venn reminder missing:\n{text}"


def test_system_blocks_dm_argument_includes_kind_specific_reminder():
    eng = _engine()
    blocks = eng._system_blocks("DM", retrieved=[], target_difficulty=3.0,
                                  subtype="argument")
    text = _block_text(blocks)
    assert "argument strength" in text, \
        f"argument reminder missing:\n{text}"


def test_system_blocks_vr_tfc_locks_type_tf_and_minigame_kind():
    eng = _engine()
    blocks = eng._system_blocks("VR", retrieved=[], target_difficulty=3.0,
                                  subtype="tfc")
    text = _block_text(blocks)
    assert "minigame_kind: 'tfc'" in text, f"tfc tag missing:\n{text}"
    assert "type:'tf'" in text, f"type:tf lock missing:\n{text}"
    assert '"True", "False", "Can\'t Tell"' in text, \
        f"option labels missing:\n{text}"


def test_system_blocks_vr_main_idea_locks_mc_and_kind():
    eng = _engine()
    blocks = eng._system_blocks("VR", retrieved=[], target_difficulty=3.0,
                                  subtype="main-idea")
    text = _block_text(blocks)
    assert "minigame_kind: 'main-idea'" in text, f"tag missing:\n{text}"
    assert "type:'mc'" in text, f"mc lock missing:\n{text}"
    assert "main idea" in text, f"kind reminder missing:\n{text}"


def test_system_blocks_vr_inference_includes_kind_reminder():
    eng = _engine()
    blocks = eng._system_blocks("VR", retrieved=[], target_difficulty=3.0,
                                  subtype="inference")
    text = _block_text(blocks)
    assert "minigame_kind: 'inference'" in text
    assert "inferred" in text, f"inference reminder missing:\n{text}"


def test_system_blocks_vr_no_subtype_omits_minigame_kind_lock():
    """Mixed-mode VR runs must not force a minigame_kind."""
    eng = _engine()
    blocks = eng._system_blocks("VR", retrieved=[], target_difficulty=3.0,
                                  subtype=None)
    text = _block_text(blocks)
    assert "minigame_kind" not in text, \
        f"expected no minigame_kind lock in mixed mode:\n{text}"


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
