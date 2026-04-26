# Subtype-Targeted Bulk Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Subtype dropdown to the Bulk Generate tab so users can target a specific subtype within a section (e.g. DM venn, QR bar chart, VR main idea) and receive homogeneous sets back. Quantity flips to "questions" when a subtype is selected; full sets are always generated and rounded up.

**Architecture:** Three layers, bottom-up.
1. **Schema** — add one optional `minigame_kind` field to `Question` so the LLM can tag VR questions with one of five rich subtypes (TFC, main idea, paraphrase, tone & purpose, inference). DM uses the existing `Question.type`. QR uses `QRChart.type`.
2. **Pipeline** — `RAGEngine.generate()` and `_system_blocks()` accept a new `subtype` kwarg. When set, a subtype-specific prompt override replaces the existing "include one of each" guidance for DM, locks `minigame_kind` for VR, and forces the chart type for QR. After parsing, a small drift check flags responses where the LLM didn't comply.
3. **UI** — a new combobox sits between the Section radios and the Quantity row in the Bulk tab. Quantity unit and helper line flip based on subtype selection. A new Treeview column shows the chosen subtype per row; verdict column shows `⚠ drift` when the model strayed.

**Tech Stack:** Python 3.13, Pydantic v2, Tkinter (`ttk.Combobox`), Anthropic SDK (structured outputs). Tests use the direct-run pattern (no pytest dependency) — each test file has a `__main__` runner that calls every `test_*` function.

**Spec:** `docs/superpowers/specs/2026-04-26-subtype-bulk-generate-design.md`

---

## File Structure

**New files:**
- `tests/test_subtype_targeting.py` — single test file covering the schema field, config dicts, system-block subtype overrides, drift detection, and the set-count rounding helper. One file because the surface is small and these tests share imports.

**Modified files:**
- `ucat/models.py` — add `minigame_kind: Optional[str] = None` to `Question`.
- `ucat/config.py` — add `SET_SIZES`, `SUBTYPES_BY_SECTION`, three `Settings.DEFAULTS` keys.
- `ucat/rag.py` — `generate()` gains `subtype` kwarg; `_system_blocks()` gains `subtype` param and emits section-specific overrides; new `_detect_subtype_drift()` helper; trace fields include `subtype`; result dict carries `subtype_drift`.
- `ucat/ui.py` — Subtype combobox + helper-line label in `_tab_bulk()`; new `_bulk_refresh_subtype_choices()`; `_bulk_inputs_changed()` flips the Quantity label and renders the helper line; `_bulk_start()` applies the ceil math via a new `_compute_set_count()` helper; `_bulk_worker()` forwards subtype; new Treeview column; `_mark_row()` populates Subtype and `⚠ drift` badge; new telemetry fields.

---

## Task 1: Add `minigame_kind` field to `Question`

**Files:**
- Modify: `ucat/models.py:49-67` (Question class)
- Test: `tests/test_subtype_targeting.py` (create)

- [ ] **Step 1: Create the test file with the first failing test**

Create `tests/test_subtype_targeting.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify the first one fails**

Run: `./venv/bin/python tests/test_subtype_targeting.py`

Expected output (the failure mode is `ValidationError` because Pydantic rejects unknown extra fields by default in some configurations, OR the attribute is missing):

```
ERROR test_question_accepts_minigame_kind_string: ValidationError: ...
```

OR (if extras are silently dropped):

```
FAIL  test_question_accepts_minigame_kind_string: assert q.minigame_kind == "main-idea"
```

Either way, the test does not pass.

- [ ] **Step 3: Add the `minigame_kind` field to `Question`**

Edit `ucat/models.py`. Locate the `Question` class (around line 49). Add a new field directly after `type`:

```python
class Question(BaseModel):
    number: int
    text: str
    type: Optional[str] = None  # DM subtype: syllogism / logical / probability / argument / venn
    minigame_kind: Optional[str] = Field(
        default=None,
        description=(
            "Pocket UCAT routing tag. For VR subtype targeting: one of "
            "'tfc', 'main-idea', 'paraphrase', 'tone-purpose', 'inference'. "
            "Optional — legacy rows and mixed-mode runs may leave this null."
        ),
    )
    options: List[OptionItem] = Field(
        # ... existing definition unchanged ...
    )
    # ... rest of class unchanged ...
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/bin/python tests/test_subtype_targeting.py`

Expected:

```
PASS  test_question_accepts_minigame_kind_string
PASS  test_question_minigame_kind_defaults_to_none
PASS  test_legacy_question_without_minigame_kind_still_validates

All tests passed.
```

- [ ] **Step 5: Run the existing schema tests to confirm no regression**

Run: `./venv/bin/python tests/test_schema.py`

Expected: all existing tests still pass (no failures).

- [ ] **Step 6: Commit**

```bash
git add ucat/models.py tests/test_subtype_targeting.py
git commit -m "feat: add optional minigame_kind field to Question schema"
```

---

## Task 2: Add `SET_SIZES` and `SUBTYPES_BY_SECTION` to config

**Files:**
- Modify: `ucat/config.py:67-94` (UCAT domain section, after `SECTION_DESC`)
- Test: `tests/test_subtype_targeting.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_subtype_targeting.py`, right above the `if __name__ == "__main__":` block:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/bin/python tests/test_subtype_targeting.py`

Expected: the new tests fail with `ImportError: cannot import name 'SET_SIZES' from 'ucat.config'`.

- [ ] **Step 3: Add the constants to config**

Edit `ucat/config.py`. Append to the UCAT domain block, right after `SECTION_DESC` (around line 84):

```python
# Question count per generated set, mirroring the min/max in the Pydantic
# section models. Used by bulk generation to convert "N questions" → "M sets".
SET_SIZES = {"VR": 4, "DM": 5, "QR": 4, "AR": 5}

# Per-section subtype catalogue. Each entry is (storage_value, human_label).
# storage_value matches the field used by the schema:
#   - DM → Question.type
#   - VR → Question.minigame_kind (with type=tf when value=='tfc', type=mc otherwise)
#   - QR → QRChart.type on the stimulus
# AR is intentionally empty: subtype dropdown is disabled in the UI.
SUBTYPES_BY_SECTION = {
    "DM": [
        ("syllogism",    "Syllogism"),
        ("logical",      "Logical (logic puzzles)"),
        ("venn",         "Venn"),
        ("probability",  "Probability"),
        ("argument",     "Argument"),
    ],
    "VR": [
        ("tfc",          "True / False / Can't Tell"),
        ("main-idea",    "Main idea"),
        ("paraphrase",   "Paraphrase match"),
        ("tone-purpose", "Tone & purpose"),
        ("inference",    "Inference"),
    ],
    "QR": [
        ("table",        "Table"),
        ("bar",          "Bar chart"),
        ("line",         "Line chart"),
        ("stacked_bar",  "Stacked bar"),
        ("pie",          "Pie"),
    ],
    "AR": [],
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/bin/python tests/test_subtype_targeting.py`

Expected: all tests pass (including the 3 from Task 1 and the 6 new ones).

- [ ] **Step 5: Commit**

```bash
git add ucat/config.py tests/test_subtype_targeting.py
git commit -m "feat: add SET_SIZES and SUBTYPES_BY_SECTION config catalogue"
```

---

## Task 3: Add new `Settings.DEFAULTS` keys

**Files:**
- Modify: `ucat/config.py:106-117` (`Settings.DEFAULTS` dict)
- Test: `tests/test_subtype_targeting.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_subtype_targeting.py`:

```python
# ─── Settings: new bulk subtype keys ─────────────────────────────────────────

import json
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/bin/python tests/test_subtype_targeting.py`

Expected: the four new tests fail with assertions like `expected '', got None`.

- [ ] **Step 3: Add the new keys to `Settings.DEFAULTS`**

Edit `ucat/config.py`. In the `Settings.DEFAULTS` dict (around line 106), add three keys after `bulk_hint`:

```python
class Settings:
    """User preferences that survive across runs."""

    DEFAULTS: Dict[str, Any] = {
        "llm":              DEFAULT_LLM,
        "embed":            DEFAULT_EMBED,
        "top_k":            DEFAULT_TOP_K,
        "mmr_lambda":       DEFAULT_MMR_LAMBDA,
        "target_difficulty": DEFAULT_TARGET_DIFFICULTY,
        "verify":           DEFAULT_VERIFY,
        "multi_judge":      DEFAULT_MULTI_JUDGE,
        "bulk_section":     "VR",
        "bulk_quantity":    10,
        "bulk_hint":        "",
        # ── Subtype targeting (added 2026-04-26) ───────────────────────────
        "bulk_subtype":            "",   # current dropdown value ("" == Any/mixed)
        "bulk_subtype_by_section": {     # remember per-section choice
            "VR": "", "DM": "", "QR": "", "AR": "",
        },
        "bulk_quantity_unit":      "sets",  # "sets" or "questions" (derived from subtype)
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/bin/python tests/test_subtype_targeting.py`

Expected: all tests pass (10 from earlier tasks + 4 new ones = 13 total).

- [ ] **Step 5: Commit**

```bash
git add ucat/config.py tests/test_subtype_targeting.py
git commit -m "feat: persist bulk subtype + per-section memory + quantity unit"
```

---

## Task 4: `_system_blocks()` accepts subtype param + DM override

**Files:**
- Modify: `ucat/rag.py:109-187` (`_system_blocks` method)
- Test: `tests/test_subtype_targeting.py` (append)

- [ ] **Step 1: Append the failing test**

Append to `tests/test_subtype_targeting.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/bin/python tests/test_subtype_targeting.py`

Expected: the four new DM tests fail because `_system_blocks` does not accept a `subtype` kwarg.

- [ ] **Step 3: Update `_system_blocks` signature and DM block**

Edit `ucat/rag.py`. Modify `_system_blocks` (around line 109):

```python
def _system_blocks(self, section: str, retrieved: list,
                    target_difficulty: float,
                    subtype: Optional[str] = None) -> List[Dict[str, Any]]:
    """Build cache-friendly system blocks.

      [0] Frozen role + structural description + retrieved KB     (CACHED)
      [1] Per-request difficulty + variation guidance              (NOT CACHED)

    When ``subtype`` is set, the per-request layer adds a section-specific
    override that locks every question in the set to that subtype.
    """
    role = (
        # ... existing role text unchanged ...
    )
    # ... QR block, AR block unchanged ...

    if section == "DM":
        if subtype:
            # Subtype override: replace the variety guidance with a lock-in.
            reminders = {
                "venn":        "Every question must include a structured `venn` field with 2 or 3 sets.",
                "probability": "State all probability values clearly; answers must be mathematically verifiable.",
                "syllogism":   "Premises must be logically sound; conclusions testable.",
                "logical":     "Each question is a clue-based deduction puzzle. Conclusions must follow from the clues.",
                "argument":    "Present a clear proposition; options vary in argument strength (strongest/weakest for/against).",
            }
            role += (
                f"\nAll 5 questions MUST be {subtype} type. "
                f"Set `type: '{subtype}'` on every question.\n"
                f"{reminders.get(subtype, '')}\n"
            )
        else:
            role += (
                "\nFor venn-type DM questions, include a structured `venn` field with "
                "2 or 3 sets so the diagram can be rendered. For other DM subtypes, "
                "leave `venn` null.\n"
                "Aim for variety: include syllogism, logical, venn, probability, AND "
                "argument subtypes across the 5 questions — one of each is ideal.\n"
            )

    # ... rest of method unchanged (ex_text assembly, diff label, return) ...
```

The key change: the existing DM block (which today *always* injects the variety guidance) becomes the `else:` branch. When `subtype` is set, the new lock-in text replaces it.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/bin/python tests/test_subtype_targeting.py`

Expected: all 17 tests pass (13 from earlier + 4 new DM subtype tests).

- [ ] **Step 5: Commit**

```bash
git add ucat/rag.py tests/test_subtype_targeting.py
git commit -m "feat: DM subtype override in _system_blocks"
```

---

## Task 5: `_system_blocks()` VR override

**Files:**
- Modify: `ucat/rag.py` (existing `_system_blocks` method, after the QR block)
- Test: `tests/test_subtype_targeting.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_subtype_targeting.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/bin/python tests/test_subtype_targeting.py`

Expected: the four new VR tests fail because no VR subtype block is emitted yet.

- [ ] **Step 3: Add the VR override block**

Edit `ucat/rag.py`. In `_system_blocks`, immediately after the existing `if section == "DM":` block (the one you modified in Task 4), add a new VR block:

```python
    if section == "VR" and subtype:
        kind_reminders = {
            "tfc": (
                "Set type:'tf' on every question. Use exactly 3 options labelled "
                '"True", "False", "Can\'t Tell". Each question is a statement the '
                "student judges as supported / contradicted / not addressed by the "
                "passage."
            ),
            "main-idea": (
                "Set type:'mc' on every question. Use 4 options labelled A, B, C, D. "
                "Each question asks for the main idea, central thesis, best title, "
                "or overall conclusion of the passage. Distractors should be "
                "plausible secondary points or over-specific details."
            ),
            "paraphrase": (
                "Set type:'mc' on every question. Use 4 options labelled A, B, C, D. "
                "Each question quotes a phrase or sentence from the passage and asks "
                "which option best restates it. Distractors should be near-paraphrases "
                "that subtly distort the original meaning."
            ),
            "tone-purpose": (
                "Set type:'mc' on every question. Use 4 options labelled A, B, C, D. "
                "Each question asks for the author's tone, attitude, or rhetorical "
                "purpose (e.g. to argue / inform / caution / evaluate). Options should "
                "be precise tone words, not synonyms of each other."
            ),
            "inference": (
                "Set type:'mc' on every question. Use 4 options labelled A, B, C, D. "
                "Each question asks what can be inferred — a conclusion supported by "
                "but not stated in the passage. Distractors should be either "
                "explicitly stated (not inferences) or unsupported."
            ),
        }
        role += (
            f"\nAll 4 questions MUST set `minigame_kind: '{subtype}'`.\n"
            f"{kind_reminders.get(subtype, '')}\n"
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/bin/python tests/test_subtype_targeting.py`

Expected: all 21 tests pass.

- [ ] **Step 5: Commit**

```bash
git add ucat/rag.py tests/test_subtype_targeting.py
git commit -m "feat: VR subtype override in _system_blocks (5 minigame kinds)"
```

---

## Task 6: `_system_blocks()` QR override

**Files:**
- Modify: `ucat/rag.py` (existing `_system_blocks` method, in the QR block)
- Test: `tests/test_subtype_targeting.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_subtype_targeting.py`:

```python
def test_system_blocks_qr_bar_locks_chart_type():
    eng = _engine()
    blocks = eng._system_blocks("QR", retrieved=[], target_difficulty=3.0,
                                  subtype="bar")
    text = _block_text(blocks)
    assert "stimulus chart MUST be type: 'bar'" in text, \
        f"missing chart-type lock:\n{text}"


def test_system_blocks_qr_table_locks_chart_type():
    eng = _engine()
    blocks = eng._system_blocks("QR", retrieved=[], target_difficulty=3.0,
                                  subtype="table")
    text = _block_text(blocks)
    assert "stimulus chart MUST be type: 'table'" in text


def test_system_blocks_qr_no_subtype_omits_chart_lock():
    eng = _engine()
    blocks = eng._system_blocks("QR", retrieved=[], target_difficulty=3.0,
                                  subtype=None)
    text = _block_text(blocks)
    assert "stimulus chart MUST be" not in text, \
        f"expected no chart lock in mixed mode:\n{text}"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/bin/python tests/test_subtype_targeting.py`

Expected: three new QR tests fail.

- [ ] **Step 3: Add the QR override**

Edit `ucat/rag.py`. In `_system_blocks`, find the existing `if section == "QR":` block (around line 134). Append the subtype lock at the end of the existing QR text:

```python
    if section == "QR":
        role += (
            "\nCRITICAL — QR stimulus is a STRUCTURED chart spec, not text. "
            "Choose `type` from {bar, line, stacked_bar, pie, table}. "
            "Populate `categories` and `series[]` for bar/line/stacked_bar/pie "
            "(leave `rows` null). "
            "For `table`, populate `rows` as a LIST of {\"name\": <column header>, "
            "\"values\": [<cell values aligned with categories>]} objects "
            "(leave `series` empty). "
            "Include realistic units (e.g. £000s, %, kg). "
            "Make data internally consistent with the questions you write.\n"
        )
        if subtype:
            role += f"\nThe stimulus chart MUST be type: '{subtype}'.\n"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/bin/python tests/test_subtype_targeting.py`

Expected: all 24 tests pass.

- [ ] **Step 5: Commit**

```bash
git add ucat/rag.py tests/test_subtype_targeting.py
git commit -m "feat: QR subtype override locks stimulus chart type"
```

---

## Task 7: `generate()` accepts subtype kwarg + appends user prompt line

**Files:**
- Modify: `ucat/rag.py:190-250` (`generate` method signature, system-block call, user-prompt assembly)
- Test: deferred to manual smoke (Task 13) — `generate()` makes a network call so it isn't unit-testable cheaply.

- [ ] **Step 1: Update the `generate()` signature**

Edit `ucat/rag.py`. The current signature (around line 190):

```python
def generate(
    self,
    section: str,
    hint: str = "",
    *,
    on_progress: Optional[Callable[[str], None]] = None,
    on_delta: Optional[Callable[[str], None]] = None,
    on_verify_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
    variation_seed: Optional[str] = None,
    force_scenario: Optional[str] = None,
    avoid_topics: Optional[List[str]] = None,
) -> Dict[str, Any]:
```

becomes:

```python
def generate(
    self,
    section: str,
    hint: str = "",
    *,
    subtype: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None,
    on_delta: Optional[Callable[[str], None]] = None,
    on_verify_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
    variation_seed: Optional[str] = None,
    force_scenario: Optional[str] = None,
    avoid_topics: Optional[List[str]] = None,
) -> Dict[str, Any]:
```

- [ ] **Step 2: Pass `subtype` to `_system_blocks` and the trace**

In the same `generate()` body, find the trace-open and the `_system_blocks` call (around line 215-225):

```python
with trace("rag_generate", section=section, hint=hint[:80],
           target_difficulty=target,
           verify=self.verify_enabled,
           multi_judge=self.multi_judge,
           async_verify=async_verify) as t:
    # ...
    system_blocks = self._system_blocks(section, retrieved, target)
```

becomes:

```python
with trace("rag_generate", section=section, hint=hint[:80],
           target_difficulty=target,
           verify=self.verify_enabled,
           multi_judge=self.multi_judge,
           async_verify=async_verify,
           subtype=subtype) as t:
    # ...
    system_blocks = self._system_blocks(section, retrieved, target, subtype=subtype)
```

- [ ] **Step 3: Append the subtype line to the user prompt**

Find the user-parts assembly (around line 227-249). Add the subtype line right before the final `"Return ONLY the JSON object."` append:

```python
    user_parts.append(
        "Content, numbers, scenarios, and passages must be entirely original — "
        "do NOT reuse wording or fact-patterns from the example documents. "
        "Mirror the JSON structure of the examples precisely. "
    )
    if force_scenario:
        # ... unchanged ...
    if avoid_topics:
        # ... unchanged ...
    if variation_seed:
        user_parts.append(f"Variation seed for stylistic diversity: {variation_seed}. ")

    if subtype:
        # Look up the human label from SUBTYPES_BY_SECTION (imported below).
        label = next(
            (lbl for v, lbl in SUBTYPES_BY_SECTION.get(section, []) if v == subtype),
            subtype,
        )
        user_parts.append(f"All questions in this set must be of subtype: {label}. ")

    user_parts.append("Return ONLY the JSON object.")
    user = "".join(user_parts)
```

- [ ] **Step 4: Add the import for `SUBTYPES_BY_SECTION`**

At the top of `ucat/rag.py` (line 10), find the existing import line:

```python
from .config import (DEFAULT_VERIFY_LLM, DEFAULT_JUDGE2_LLM, IRT_BANDS,
                      Settings, SECTIONS, SECTION_DESC, difficulty_label)
```

Add `SUBTYPES_BY_SECTION` to it:

```python
from .config import (DEFAULT_VERIFY_LLM, DEFAULT_JUDGE2_LLM, IRT_BANDS,
                      Settings, SECTIONS, SECTION_DESC, difficulty_label,
                      SUBTYPES_BY_SECTION)
```

- [ ] **Step 5: Run all unit tests to confirm no regression**

Run:

```bash
./venv/bin/python tests/test_subtype_targeting.py
./venv/bin/python tests/test_schema.py
./venv/bin/python tests/test_bulk_cost.py
./venv/bin/python tests/test_coverage.py
```

Expected: all tests pass. (No new tests in this task — `generate()` itself is only smoke-tested in Task 13.)

- [ ] **Step 6: Commit**

```bash
git add ucat/rag.py
git commit -m "feat: generate() accepts subtype kwarg and threads it through"
```

---

## Task 8: Drift detection helper + wire into `generate()`

**Files:**
- Modify: `ucat/rag.py` (add helper near `_system_blocks`; call from `generate` before the result dict)
- Test: `tests/test_subtype_targeting.py` (append)

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_subtype_targeting.py`:

```python
# ─── Drift detection ─────────────────────────────────────────────────────────

from ucat.rag import _detect_subtype_drift


def test_drift_dm_all_match():
    data = {"questions": [{"type": "venn"} for _ in range(5)]}
    assert _detect_subtype_drift("DM", data, "venn") is None


def test_drift_dm_one_wrong_flags_drift():
    data = {"questions": [
        {"type": "venn"}, {"type": "venn"}, {"type": "syllogism"},
        {"type": "venn"}, {"type": "venn"},
    ]}
    drift = _detect_subtype_drift("DM", data, "venn")
    assert drift is not None and "syllogism" in drift, \
        f"expected drift mentioning syllogism, got {drift!r}"


def test_drift_vr_uses_minigame_kind_field():
    data = {"questions": [{"minigame_kind": "main-idea"} for _ in range(4)]}
    assert _detect_subtype_drift("VR", data, "main-idea") is None


def test_drift_vr_minigame_kind_mismatch():
    data = {"questions": [
        {"minigame_kind": "main-idea"},
        {"minigame_kind": "inference"},
        {"minigame_kind": "main-idea"},
        {"minigame_kind": "main-idea"},
    ]}
    drift = _detect_subtype_drift("VR", data, "main-idea")
    assert drift is not None and "inference" in drift


def test_drift_qr_chart_type_mismatch():
    data = {"stimulus": {"type": "line"}, "questions": []}
    drift = _detect_subtype_drift("QR", data, "bar")
    assert drift is not None and "line" in drift


def test_drift_qr_chart_type_match():
    data = {"stimulus": {"type": "bar"}, "questions": []}
    assert _detect_subtype_drift("QR", data, "bar") is None


def test_drift_subtype_none_returns_none():
    """No subtype requested → no drift possible, regardless of content."""
    data = {"questions": [{"type": "anything"}]}
    assert _detect_subtype_drift("DM", data, None) is None


def test_drift_vr_missing_minigame_kind_flags_drift():
    """Legacy/un-tagged questions should be flagged as drift when subtype was asked."""
    data = {"questions": [{}, {}, {}, {}]}
    drift = _detect_subtype_drift("VR", data, "main-idea")
    assert drift is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/bin/python tests/test_subtype_targeting.py`

Expected: all 8 new tests fail with `ImportError: cannot import name '_detect_subtype_drift'`.

- [ ] **Step 3: Add the helper function**

Edit `ucat/rag.py`. Add the helper near the top of the module (after imports, before the `RAGEngine` class):

```python
def _detect_subtype_drift(section: str, data: Dict[str, Any],
                            subtype: Optional[str]) -> Optional[str]:
    """Return a human-readable drift message if the parsed set doesn't match
    the requested subtype, else None.

    - DM → checks Question.type on every question
    - VR → checks Question.minigame_kind on every question
    - QR → checks QRChart.type on the stimulus
    - AR → no subtype targeting; always returns None
    - subtype is None → always returns None
    """
    if not subtype:
        return None

    if section == "QR":
        actual = (data.get("stimulus") or {}).get("type")
        if actual != subtype:
            return f"Asked {subtype}, got chart type {actual!r}"
        return None

    if section == "AR":
        return None

    field = "minigame_kind" if section == "VR" else "type"
    actuals = [q.get(field) for q in data.get("questions", [])]
    if not all(a == subtype for a in actuals):
        return f"Asked {subtype}, got {actuals}"
    return None
```

- [ ] **Step 4: Wire the helper into `generate()`**

In `generate()`, after `data["section"] = section` (around line 262, right before the verification block), call the helper and stash the result:

```python
            data = parsed.model_dump()
            data["section"] = section

            # Subtype drift check — fast, deterministic, no API call.
            subtype_drift = _detect_subtype_drift(section, data, subtype)

            # ── existing verification block continues here ──
```

Then in the trace `t.update(...)` block (around line 302), include the drift in the trace fields:

```python
            t.update({
                "input_tokens": total_usage["input_tokens"],
                # ... existing fields ...
                "retrieved_ids": [d["id"] for _, d in retrieved],
                "subtype_drift": subtype_drift,   # NEW
            })
```

Finally, add `subtype_drift` to the returned dict (around line 349):

```python
        return {
            "data":         data,
            "retrieved":    retrieved,
            "usage":        total_usage,
            "verdict":      verdict_dict,
            "coverage":     coverage_dict,
            "difficulty":   cal,
            "dup_warning":  dup_warning,
            "row_id":       row_id,
            "subtype_drift": subtype_drift,   # NEW (None when no drift)
        }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `./venv/bin/python tests/test_subtype_targeting.py`

Expected: all 32 tests pass.

- [ ] **Step 6: Commit**

```bash
git add ucat/rag.py tests/test_subtype_targeting.py
git commit -m "feat: subtype drift detection on parsed sets"
```

---

## Task 9: UI — Subtype combobox + per-section restoration

**Files:**
- Modify: `ucat/ui.py:395-477` (`_tab_bulk` method)

- [ ] **Step 1: Add the Subtype row to `_tab_bulk`**

Edit `ucat/ui.py`. In `_tab_bulk`, after the Section radios row (around line 416, after the `for code in SECTIONS:` loop) and BEFORE the Quantity row, insert the Subtype combobox:

```python
        # Subtype dropdown — populated based on selected section.
        sb = tk.Frame(p, bg=BG); sb.pack(anchor="w", pady=(0, 10))
        tk.Label(sb, text="Subtype:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self._bulk_subtype = tk.StringVar(value="")
        self._bulk_subtype_cb = ttk.Combobox(
            sb, textvariable=self._bulk_subtype,
            state="readonly", width=32, font=FM,
        )
        self._bulk_subtype_cb.pack(side="left")
        self._bulk_subtype_cb.bind(
            "<<ComboboxSelected>>", lambda _e: self._bulk_inputs_changed()
        )
        # Initial population uses the current section.
        self._bulk_refresh_subtype_choices()
```

- [ ] **Step 2: Update Section radio command to refresh subtype choices**

In the same method, find the Section radio loop (around line 410-416). Change the `command=` callback so it also refreshes the subtype dropdown:

```python
        for code in SECTIONS:
            tk.Radiobutton(sr, text=f" {code} ", variable=self._bulk_sec, value=code,
                           bg=BG, fg=TEXT, selectcolor=PANEL, activebackground=BG,
                           activeforeground=ACCENT, font=FB, indicatoron=False,
                           relief="flat", bd=1, padx=12, pady=6, cursor="hand2",
                           command=self._bulk_section_changed   # CHANGED
                           ).pack(side="left", padx=4)
```

- [ ] **Step 3: Add the `_bulk_section_changed` and `_bulk_refresh_subtype_choices` helpers**

Add these two new methods to the same class as `_tab_bulk` (immediately after `_tab_bulk`, before `_bulk_inputs_changed`):

```python
    def _bulk_section_changed(self):
        """Called when the Section radio changes. Refreshes subtype choices
        and runs the standard inputs-changed update."""
        self._bulk_refresh_subtype_choices()
        self._bulk_inputs_changed()

    def _bulk_refresh_subtype_choices(self):
        """Rebuild the Subtype combobox values for the current section,
        restoring the user's last choice for that section."""
        section = self._bulk_sec.get()
        entries = SUBTYPES_BY_SECTION.get(section, [])

        # Build the displayed list: "Any (mixed)" always first, then human labels.
        labels = ["Any (mixed)"] + [lbl for _v, lbl in entries]
        self._bulk_subtype_cb.config(values=labels)

        # Disable for sections with no subtypes (AR).
        if not entries:
            self._bulk_subtype_cb.config(state="disabled")
            self._bulk_subtype.set("Any (mixed)")
            return
        self._bulk_subtype_cb.config(state="readonly")

        # Restore last-used subtype for this section.
        by_section = self.settings.get("bulk_subtype_by_section") or {}
        stored_value = by_section.get(section, "")
        if stored_value:
            stored_label = next(
                (lbl for v, lbl in entries if v == stored_value),
                "Any (mixed)",
            )
            self._bulk_subtype.set(stored_label)
        else:
            self._bulk_subtype.set("Any (mixed)")
```

- [ ] **Step 4: Add the imports**

At the top of `ucat/ui.py` (around line 26), find the line that imports from `.config`:

```python
from .config import (APP_TITLE, SECTIONS, SECTION_COLORS, SECTION_DESC,
                      LLM_CHOICES, EMBED_CHOICES, IRT_BANDS, Settings, api_status,
                      BULK_MAX_QUANTITY, BULK_COST_CONFIRM_THRESHOLD, estimate_bulk_cost)
```

(The exact ordering of names in the existing import may differ — adapt accordingly.) Add `SUBTYPES_BY_SECTION`, `SET_SIZES`, and `compute_set_count` (the helper added in Task 10 — safe to import here even though it's used later):

```python
from .config import (APP_TITLE, SECTIONS, SECTION_COLORS, SECTION_DESC,
                      LLM_CHOICES, EMBED_CHOICES, IRT_BANDS, Settings, api_status,
                      BULK_MAX_QUANTITY, BULK_COST_CONFIRM_THRESHOLD, estimate_bulk_cost,
                      SUBTYPES_BY_SECTION, SET_SIZES, compute_set_count)
```

- [ ] **Step 5: Manually smoke-test the dropdown**

Run the app:

```bash
./venv/bin/python -m ucat
```

Open the **BULK** tab. Verify:
- A "Subtype:" row appears between the Section radios and the Quantity row.
- With section=DM selected, the dropdown shows: `Any (mixed)`, `Syllogism`, `Logical (logic puzzles)`, `Venn`, `Probability`, `Argument`.
- Switching to VR changes the dropdown to: `Any (mixed)`, `True / False / Can't Tell`, `Main idea`, `Paraphrase match`, `Tone & purpose`, `Inference`.
- Switching to QR changes the dropdown to chart types.
- Switching to AR disables the dropdown (greyed out, locked on "Any (mixed)").
- Default selection on first run is "Any (mixed)" for every section.

Close the app. (Persistence is tested in Task 10 once `_bulk_inputs_changed` writes the setting.)

- [ ] **Step 6: Commit**

```bash
git add ucat/ui.py
git commit -m "feat: bulk-tab subtype combobox with per-section choices"
```

---

## Task 10: UI — Quantity label flip + helper line + ceil math + persistence

**Files:**
- Modify: `ucat/ui.py:418-525` (`_tab_bulk` quantity row, `_bulk_inputs_changed`)
- Test: `tests/test_subtype_targeting.py` (append — for the ceil-math helper)

- [ ] **Step 1: Append the failing tests for the ceil-math helper**

Append to `tests/test_subtype_targeting.py`:

```python
# ─── Set-count math helper ───────────────────────────────────────────────────

from ucat.config import compute_set_count


def test_compute_set_count_no_subtype_returns_input():
    """Without subtype, quantity is already 'sets' — no conversion."""
    assert compute_set_count(10, "DM", subtype=None) == 10
    assert compute_set_count(1,  "VR", subtype=None) == 1


def test_compute_set_count_dm_questions_round_up():
    # DM has 5 questions/set
    assert compute_set_count(10, "DM", subtype="venn") == 2     # 10/5 = 2 exact
    assert compute_set_count(7,  "DM", subtype="venn") == 2     # ceil(7/5) = 2
    assert compute_set_count(11, "DM", subtype="venn") == 3     # ceil(11/5) = 3
    assert compute_set_count(1,  "DM", subtype="venn") == 1     # ceil(1/5) = 1


def test_compute_set_count_vr_questions_round_up():
    # VR has 4 questions/set
    assert compute_set_count(8,  "VR", subtype="main-idea") == 2  # 8/4 = 2 exact
    assert compute_set_count(13, "VR", subtype="main-idea") == 4  # ceil(13/4) = 4


def test_compute_set_count_qr_questions_round_up():
    # QR has 4 questions/set
    assert compute_set_count(5, "QR", subtype="bar") == 2          # ceil(5/4) = 2


def test_compute_set_count_zero_returns_zero():
    assert compute_set_count(0, "DM", subtype="venn") == 0


def test_compute_set_count_negative_returns_zero():
    assert compute_set_count(-3, "DM", subtype="venn") == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/bin/python tests/test_subtype_targeting.py`

Expected: 6 new tests fail with `ImportError: cannot import name 'compute_set_count'`.

- [ ] **Step 3: Add `compute_set_count` to `ucat/config.py`**

Edit `ucat/config.py`. Append after the `SUBTYPES_BY_SECTION` definition:

```python
import math


def compute_set_count(n_input: int, section: str,
                       subtype: Optional[str]) -> int:
    """Convert the user's quantity input into a number of sets to generate.

    With ``subtype=None`` (Any/mixed mode), the input is already in sets and
    is returned as-is. With a subtype selected, the input is in *questions*
    and is rounded UP to the next full set using ``SET_SIZES[section]``.

    Returns 0 for non-positive inputs.
    """
    if n_input <= 0:
        return 0
    if not subtype:
        return n_input
    per_set = SET_SIZES.get(section, 1) or 1
    return math.ceil(n_input / per_set)
```

(Note: `import math` should go at the module top alongside the other imports if not already present. Check the existing imports — if `math` is not already imported, move the import line to the top of the file.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/bin/python tests/test_subtype_targeting.py`

Expected: all 38 tests pass.

- [ ] **Step 5: Update `_tab_bulk` to capture the Quantity label and add a helper-line label**

Edit `ucat/ui.py`. In `_tab_bulk`, find the Quantity row (around line 418-424):

```python
        # Quantity + topic hint row.
        qr = tk.Frame(p, bg=BG); qr.pack(fill="x", pady=(0, 10))
        tk.Label(qr, text="Quantity:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
```

Replace with a stored handle:

```python
        # Quantity + topic hint row.
        qr = tk.Frame(p, bg=BG); qr.pack(fill="x", pady=(0, 10))
        self._bulk_qty_lbl = tk.Label(qr, text="Sets:", bg=BG, fg=MUTED, font=FM)
        self._bulk_qty_lbl.pack(side="left", padx=(0, 14))
```

Then, immediately after the Quantity row (before the cost preview banner), add a helper-line label:

```python
        # Helper line: "→ 2 sets × 5 questions = 10 questions". Hidden when
        # no subtype is selected.
        self._bulk_yield_lbl = tk.Label(
            p, text="", bg=BG, fg=MUTED, font=FS, anchor="w"
        )
        self._bulk_yield_lbl.pack(anchor="w", pady=(0, 4))
```

- [ ] **Step 6: Update `_bulk_inputs_changed` to flip the label, render the helper line, and persist subtype**

Replace the body of `_bulk_inputs_changed` (around line 479-525) with:

```python
    def _bulk_inputs_changed(self):
        """Called whenever section / subtype / quantity / hint changes.
        Validates input, updates labels, computes the cost preview, persists
        settings, and gates the Start button."""
        section = self._bulk_sec.get()
        hint    = self._bulk_hint.get()

        # Resolve subtype: combobox displays human labels but we store the
        # internal value (e.g. 'venn' not 'Venn'). 'Any (mixed)' → "" → None.
        label_to_value = {
            lbl: v for v, lbl in SUBTYPES_BY_SECTION.get(section, [])
        }
        chosen_label = self._bulk_subtype.get()
        subtype_value = label_to_value.get(chosen_label, "")
        subtype = subtype_value or None

        # Persist settings.
        self.settings.set("bulk_section", section)
        self.settings.set("bulk_hint",    hint)
        self.settings.set("bulk_subtype", subtype_value)
        by_section = dict(self.settings.get("bulk_subtype_by_section") or {})
        by_section[section] = subtype_value
        self.settings.set("bulk_subtype_by_section", by_section)
        self.settings.set("bulk_quantity_unit",
                            "questions" if subtype else "sets")

        # Flip the Quantity label.
        self._bulk_qty_lbl.config(
            text=("Questions:" if subtype else "Sets:")
        )

        # Parse quantity.
        raw = self._bulk_qty.get().strip()
        n_input: Optional[int] = None
        if raw.isdigit():
            n_input = int(raw)
            if n_input >= 1:
                self.settings.set("bulk_quantity", n_input)

        # Compute set count and yield helper line.
        if n_input is None or n_input < 1:
            self._bulk_yield_lbl.config(text="")
            self._bulk_cost_lbl.config(
                text=f"Enter a number 1 - {BULK_MAX_QUANTITY * (SET_SIZES.get(section, 5) if subtype else 1)}.",
                fg=WARN,
            )
            self._bulk_start_btn.config(state="disabled")
            return

        n_sets = compute_set_count(n_input, section, subtype)
        capped = False
        if n_sets > BULK_MAX_QUANTITY:
            n_sets = BULK_MAX_QUANTITY
            capped = True

        # Helper line — only when subtype is set.
        if subtype:
            per_set = SET_SIZES[section]
            yielded = n_sets * per_set
            extra = yielded - n_input
            extra_note = f"  ({extra} extra)" if extra > 0 else ""
            cap_note = "  (capped — split into multiple runs for more)" if capped else ""
            self._bulk_yield_lbl.config(
                text=f"→ {n_sets} sets × {per_set} questions = {yielded} questions{extra_note}{cap_note}",
                fg=MUTED,
            )
        else:
            self._bulk_yield_lbl.config(text="")

        # Cost preview (always in sets — that's what's billed).
        llm     = self.settings.get("llm")
        verify  = bool(self.settings.get("verify"))
        jury    = bool(self.settings.get("multi_judge"))
        low, high = estimate_bulk_cost(n_sets, llm, multi_judge=jury, verify=verify)

        suffix = "  (capped at the max — split into multiple runs for more)" if capped else ""
        self._bulk_cost_lbl.config(
            text=f"Estimated cost: ~${low:.2f} - ${high:.2f}   "
                 f"({n_sets} sets × {llm}{suffix})",
            fg=ACCENT,
        )

        if self._bulk_thread is None or not self._bulk_thread.is_alive():
            self._bulk_start_btn.config(state="normal")
```

- [ ] **Step 7: Manually smoke-test the label flip and helper line**

Run the app:

```bash
./venv/bin/python -m ucat
```

Open the **BULK** tab. Verify:
- With Subtype = "Any (mixed)", label reads `Sets:`, helper line is empty, cost preview reads "N sets × …".
- With Subtype = "Venn" and Quantity = 10, label reads `Questions:`, helper line reads `→ 2 sets × 5 questions = 10 questions`, cost preview reads "2 sets × …".
- With Subtype = "Venn" and Quantity = 7, helper line reads `→ 2 sets × 5 questions = 10 questions  (3 extra)`.
- Switching section to VR while Subtype is "Venn": dropdown resets and the per-section memory mechanism kicks in (Subtype shows "Any (mixed)" or whatever was last chosen for VR).
- Restart the app. The last-used subtype per section is restored.

- [ ] **Step 8: Commit**

```bash
git add ucat/ui.py ucat/config.py tests/test_subtype_targeting.py
git commit -m "feat: bulk-tab quantity unit flip + yield helper line + persistence"
```

---

## Task 11: UI — Treeview Subtype column + plumb subtype through worker + drift badge

**Files:**
- Modify: `ucat/ui.py:454-465` (Treeview definition in `_tab_bulk`), `_bulk_seed_rows` (line 629), `_bulk_set_row` (line 641), `_bulk_start` (line 527), `_bulk_worker` (line 792), `_bulk_run_finished` (line 720).

The actual UI layer uses `_bulk_set_row(idx, *, status, result, error, started, progress)` — not `_mark_row` — and computes display cells from a row dict with keys `idx`, `status`, `result`, `error`, `started`. This task threads `subtype` through that existing structure.

- [ ] **Step 1: Add the Subtype column to the Treeview**

Edit `ucat/ui.py`. Find the Treeview definition in `_tab_bulk` (around line 454-462):

```python
        # Treeview of per-set rows.
        tf = tk.Frame(p, bg=BG); tf.pack(fill="both", expand=True, pady=(8, 0))
        cols = ("#", "Started", "Status", "Verdict", "Cost", "Difficulty")
        self._bulk_tree = ttk.Treeview(tf, columns=cols, show="headings", height=10)
        for c, w in zip(cols, (44, 90, 180, 100, 70, 80)):
            self._bulk_tree.heading(c, text=c)
            self._bulk_tree.column(c, width=w, anchor="w" if c == "Status" else "center")
```

Change to add `Subtype` as the third column:

```python
        # Treeview of per-set rows.
        tf = tk.Frame(p, bg=BG); tf.pack(fill="both", expand=True, pady=(8, 0))
        cols = ("#", "Started", "Subtype", "Status", "Verdict", "Cost", "Difficulty")
        self._bulk_tree = ttk.Treeview(tf, columns=cols, show="headings", height=10)
        for c, w in zip(cols, (44, 90, 110, 180, 100, 70, 80)):
            self._bulk_tree.heading(c, text=c)
            self._bulk_tree.column(c, width=w, anchor="w" if c == "Status" else "center")
```

- [ ] **Step 2: Add `subtype` field to the row dict in `_bulk_seed_rows`**

Edit `_bulk_seed_rows` (around line 629). The current implementation:

```python
    def _bulk_seed_rows(self, n: int):
        """Initialise _bulk_rows + Treeview with N queued entries."""
        self._bulk_rows = [{"idx": i, "status": "queued", "result": None,
                             "error": None, "started": ""} for i in range(1, n + 1)]
        for iid in self._bulk_tree.get_children():
            self._bulk_tree.delete(iid)
        for r in self._bulk_rows:
            self._bulk_tree.insert(
                "", "end", iid=self._bulk_row_iid(r["idx"]),
                values=(r["idx"], "", "queued", "—", "—", "—"),
            )
```

Add a `subtype` field to each row dict, derive its label from settings, and include it in the Treeview values tuple as the new third column:

```python
    def _bulk_seed_rows(self, n: int):
        """Initialise _bulk_rows + Treeview with N queued entries."""
        section = self._bulk_sec.get()
        subtype_value = self.settings.get("bulk_subtype") or ""
        subtype_label = next(
            (lbl for v, lbl in SUBTYPES_BY_SECTION.get(section, []) if v == subtype_value),
            "—",
        )
        self._bulk_rows = [
            {"idx": i, "status": "queued", "result": None,
              "error": None, "started": "", "subtype": subtype_value or None}
            for i in range(1, n + 1)
        ]
        for iid in self._bulk_tree.get_children():
            self._bulk_tree.delete(iid)
        for r in self._bulk_rows:
            self._bulk_tree.insert(
                "", "end", iid=self._bulk_row_iid(r["idx"]),
                values=(r["idx"], "", subtype_label, "queued", "—", "—", "—"),
            )
```

- [ ] **Step 3: Update `_bulk_set_row` to write the Subtype column and a drift-aware Verdict**

Edit `_bulk_set_row` (around line 641). Two changes:

1. Compute a `subtype_cell` from the row dict and include it in the values tuple.
2. When `row["result"].get("subtype_drift")` is truthy, override the verdict_cell to `"⚠ drift"` (preserving the rest of the verdict logic for non-drift rows).

The full updated method body — replace the existing `_bulk_set_row` entirely:

```python
    def _bulk_set_row(self, idx: int, *,
                       status: Optional[str] = None,
                       result: Optional[Dict[str, Any]] = None,
                       error:  Optional[str] = None,
                       started: Optional[str] = None,
                       progress: Optional[str] = None):
        """Update an in-memory row + the Treeview cells. Main-thread only."""
        if idx < 1 or idx > len(self._bulk_rows):
            return
        row = self._bulk_rows[idx - 1]
        if started is not None: row["started"] = started
        if result  is not None: row["result"]  = result
        if error   is not None: row["error"]   = error
        if status  is not None: row["status"]  = status

        # Compute display cells.
        if status == "running" and progress:
            st_cell = f"⟳ {progress[:60]}"
        elif row["status"] == "running":
            st_cell = "⟳ running"
        elif row["status"] == "done":
            st_cell = "✓ done"
        elif row["status"] == "failed":
            st_cell = f"✗ {(row['error'] or '')[:60]}"
        elif row["status"] == "skipped":
            st_cell = "· skipped"
        else:  # queued
            st_cell = "queued"

        # Subtype cell.
        subtype_value = row.get("subtype")
        subtype_cell = next(
            (lbl for v, lbl in SUBTYPES_BY_SECTION.get(self._bulk_sec.get(), [])
             if v == subtype_value),
            "—",
        ) if subtype_value else "—"

        verdict_cell = "—"
        cost_cell    = "—"
        diff_cell    = "—"
        if row["result"]:
            v = row["result"].get("verdict") or {}
            sym_disagreed = len((v.get("symbolic_qr") or {}).get("disagreed") or [])
            drift = row["result"].get("subtype_drift")
            if drift:
                # Drift takes precedence over the normal verdict badge.
                verdict_cell = "⚠ drift"
            elif not v:
                verdict_cell = "—"
            elif v.get("pending"):
                verdict_cell = f"⟳ ⚠{sym_disagreed}" if sym_disagreed else "⟳"
            elif v.get("overall_correct", True):
                verdict_cell = "✓" if sym_disagreed == 0 else f"⚠ {sym_disagreed}"
            else:
                fq = len(v.get("flagged_questions") or [])
                verdict_cell = f"⚠ {fq + sym_disagreed}"

            u = row["result"].get("usage") or {}
            if u.get("cost_usd") is not None:
                cost_cell = f"${u['cost_usd']:.3f}"

            cal = row["result"].get("difficulty") or {}
            sd  = cal.get("set_difficulty")
            if isinstance(sd, (int, float)):
                diff_cell = f"{sd:.1f}"

        self._bulk_tree.item(
            self._bulk_row_iid(idx),
            values=(idx, row["started"], subtype_cell, st_cell,
                      verdict_cell, cost_cell, diff_cell),
        )
```

- [ ] **Step 4: Plumb subtype through `_bulk_start` (with ceil math)**

Edit `_bulk_start` (around line 527). Find the section that reads inputs:

```python
        section = self._bulk_sec.get()
        raw = self._bulk_qty.get().strip()
        if not raw.isdigit():
            return
        n = min(int(raw), BULK_MAX_QUANTITY)
        if n < 1:
            return
```

Replace with subtype resolution + ceil math:

```python
        section = self._bulk_sec.get()

        # Resolve subtype from the dropdown's human label back to the storage value.
        label_to_value = {
            lbl: v for v, lbl in SUBTYPES_BY_SECTION.get(section, [])
        }
        subtype_value = label_to_value.get(self._bulk_subtype.get(), "")
        subtype = subtype_value or None

        raw = self._bulk_qty.get().strip()
        if not raw.isdigit():
            return
        n_input = int(raw)
        if n_input < 1:
            return

        # Convert questions → sets when subtype is set; cap at the max set count.
        n = compute_set_count(n_input, section, subtype)
        n = min(n, BULK_MAX_QUANTITY)
        if n < 1:
            return
```

Then find the `threading.Thread` call (around line 575-579):

```python
        self._bulk_stop.clear()
        self._bulk_thread = threading.Thread(
            target=self._bulk_worker, args=(section, hint, n), daemon=True
        )
        self._bulk_thread.start()
```

Add `subtype` to the args tuple:

```python
        self._bulk_stop.clear()
        self._bulk_thread = threading.Thread(
            target=self._bulk_worker, args=(section, hint, n, subtype), daemon=True
        )
        self._bulk_thread.start()
```

- [ ] **Step 5: Update `_bulk_worker` signature and pass subtype to `rag.generate`**

Edit `_bulk_worker` (around line 792). The current signature is:

```python
    def _bulk_worker(self, section: str, hint: str, n: int):
        self.after(0, lambda: self._bulk_run_started(n))
```

Add the new `subtype` parameter:

```python
    def _bulk_worker(self, section: str, hint: str, n: int,
                       subtype: Optional[str]):
        self.after(0, lambda: self._bulk_run_started(n))
```

Then find the `result = self.rag.generate(` call inside the loop (around line 820):

```python
                    result = self.rag.generate(
                        section, hint,
                        on_progress=lambda m, idx=i: self.after(0, lambda msg=m, _i=idx:
                            self._bulk_set_row(_i, status="running", progress=msg)),
                        on_delta=None,
                        on_verify_complete=lambda upd, idx=i: self.after(
                            0, lambda u=upd, _i=idx: self._bulk_verify_complete(_i, u)),
                        variation_seed=str(uuid.uuid4())[:8],
                        force_scenario=diversify.get("scenario"),
                        avoid_topics=diversify.get("avoid_topics"),
                    )
```

Add the `subtype` kwarg:

```python
                    result = self.rag.generate(
                        section, hint,
                        subtype=subtype,                       # NEW
                        on_progress=lambda m, idx=i: self.after(0, lambda msg=m, _i=idx:
                            self._bulk_set_row(_i, status="running", progress=msg)),
                        on_delta=None,
                        on_verify_complete=lambda upd, idx=i: self.after(
                            0, lambda u=upd, _i=idx: self._bulk_verify_complete(_i, u)),
                        variation_seed=str(uuid.uuid4())[:8],
                        force_scenario=diversify.get("scenario"),
                        avoid_topics=diversify.get("avoid_topics"),
                    )
```

- [ ] **Step 6: Update `_bulk_run_finished` to count drift**

Edit `_bulk_run_finished` (around line 720). Current method:

```python
    def _bulk_run_finished(self, succeeded: int, failed: int, stopped: bool = False):
        n = len(self._bulk_rows)
        skipped = sum(1 for r in self._bulk_rows if r["status"] == "skipped")
        elapsed = (time.perf_counter() - (self._bulk_started_at or time.perf_counter()))
        emit("bulk_run_end",
             section=self._bulk_sec.get(),
             n=n,
             succeeded=succeeded,
             failed=failed,
             stopped=stopped,
             actual_cost_usd=round(self._bulk_run_cost, 4),
             duration_s=round(elapsed, 1))
        # ... existing button reset / status bar ...
        if stopped:
            tail = f"Stopped at {succeeded + failed} / {n}."
        else:
            tail = f"Bulk run finished: {succeeded} succeeded, {failed} failed"
            if skipped: tail += f", {skipped} skipped"
            tail += "."
```

Add a drift count and weave it into the tail message (telemetry fields are added in Task 12):

```python
    def _bulk_run_finished(self, succeeded: int, failed: int, stopped: bool = False):
        n = len(self._bulk_rows)
        skipped = sum(1 for r in self._bulk_rows if r["status"] == "skipped")
        drift_count = sum(
            1 for r in self._bulk_rows
            if (r.get("result") or {}).get("subtype_drift")
        )
        elapsed = (time.perf_counter() - (self._bulk_started_at or time.perf_counter()))
        emit("bulk_run_end",
             section=self._bulk_sec.get(),
             n=n,
             succeeded=succeeded,
             failed=failed,
             stopped=stopped,
             actual_cost_usd=round(self._bulk_run_cost, 4),
             duration_s=round(elapsed, 1))
        self._bulk_thread = None
        self._bulk_started_at = None
        self._bulk_start_btn.config(state="normal", text="⚡  START BULK RUN")
        self._bulk_stop_btn.config(state="disabled")
        if stopped:
            tail = f"Stopped at {succeeded + failed} / {n}."
        else:
            drift_note = f" ({drift_count} with subtype drift)" if drift_count else ""
            tail = f"Bulk run finished: {succeeded} succeeded{drift_note}, {failed} failed"
            if skipped: tail += f", {skipped} skipped"
            tail += "."
        self._bulk_progress_lbl.config(text=tail)
        self._status(tail)
        self._bulk_inputs_changed()  # re-evaluate Start button against new state
```

- [ ] **Step 7: Manually smoke-test end-to-end with subtype targeting**

Run the app:

```bash
./venv/bin/python -m ucat
```

Open the **BULK** tab. Configure: Section=DM, Subtype=Venn, Quantity=10. Click START.

Verify:
- Treeview shows rows with the `Subtype` column populated as `Venn`.
- Each row goes `queued → running → ✓ done`. Verdict shows `✓` (or `⚠ N` for verifier-flagged), or `⚠ drift` if the model didn't honor the subtype.
- Click a completed row — preview shows 5 venn questions with rendered Venn diagrams.
- Final status bar: `"Bulk run finished: 2 succeeded, 0 failed."` plus `(N with subtype drift)` if any rows drifted.

Then test with VR=Main idea, Quantity=4 → 1 set of 4 main-idea questions; preview shows `minigame_kind: main-idea` on each.

Then test with QR=Bar chart, Quantity=4 → 1 set with a bar chart stimulus.

- [ ] **Step 8: Commit**

```bash
git add ucat/ui.py
git commit -m "feat: bulk-tab subtype column + worker plumbing + drift badge"
```

- [ ] **Step 6: Manually smoke-test end-to-end with subtype targeting**

Run the app:

```bash
./venv/bin/python -m ucat
```

Open the **BULK** tab. Configure: Section=DM, Subtype=Venn, Quantity=10. Click START.

Verify:
- Treeview shows rows with the `Subtype` column populated as `Venn`.
- After each set completes: row goes `running → done`, Verdict shows `✓` (or `⚠ flagged` for verifier-flagged), or `⚠ drift` if the model didn't honor the subtype.
- Click a completed row — preview shows 5 venn questions with rendered Venn diagrams.
- Final status bar reads e.g. `"Bulk run finished: 2 succeeded, 0 failed."` — and includes drift count if any rows drifted.

Then test with VR=Main idea, Quantity=4 → should generate 1 set of 4 main-idea questions. Check the preview shows `minigame_kind: main-idea` on each question (or that the question text matches the main-idea pattern).

Then test with QR=Bar chart, Quantity=4 → 1 set with a bar chart stimulus.

- [ ] **Step 7: Commit**

```bash
git add ucat/ui.py
git commit -m "feat: bulk-tab subtype column + worker plumbing + drift badge"
```

---

## Task 12: Telemetry fields

**Files:**
- Modify: `ucat/ui.py:701-718` (`_bulk_run_started` emit) and `ucat/ui.py:720-731` (`_bulk_run_finished` emit)

- [ ] **Step 1: Add subtype to the `bulk_run_start` emit**

Edit `_bulk_run_started`. The current emit (around line 712-718):

```python
        emit("bulk_run_start",
             section=self._bulk_sec.get(),
             n=n,
             model=llm,
             verify=verify,
             multi_judge=jury,
             estimated_cost_high=round(est_high, 4))
```

Add a `subtype` field:

```python
        emit("bulk_run_start",
             section=self._bulk_sec.get(),
             n=n,
             model=llm,
             verify=verify,
             multi_judge=jury,
             estimated_cost_high=round(est_high, 4),
             subtype=(self.settings.get("bulk_subtype") or None))
```

- [ ] **Step 2: Add subtype + drift_count to the `bulk_run_end` emit**

In `_bulk_run_finished` (you already added `drift_count` calculation in Task 11 Step 6), find the existing emit (around line 724-731):

```python
        emit("bulk_run_end",
             section=self._bulk_sec.get(),
             n=n,
             succeeded=succeeded,
             failed=failed,
             stopped=stopped,
             actual_cost_usd=round(self._bulk_run_cost, 4),
             duration_s=round(elapsed, 1))
```

Add the new fields:

```python
        emit("bulk_run_end",
             section=self._bulk_sec.get(),
             n=n,
             succeeded=succeeded,
             failed=failed,
             stopped=stopped,
             actual_cost_usd=round(self._bulk_run_cost, 4),
             duration_s=round(elapsed, 1),
             subtype=(self.settings.get("bulk_subtype") or None),
             drift_count=drift_count)
```

(`drift_count` is the local variable computed in Task 11 Step 6 — make sure that calculation lives ABOVE this emit call.)

- [ ] **Step 3: Manually verify telemetry**

Run a small bulk run with a subtype:

```bash
./venv/bin/python -m ucat
# In the app: BULK → DM → Venn → Quantity 10 → START → wait → close.
```

Inspect the telemetry log (last 30 lines, filter for the two event types):

```bash
tail -30 ucat_telemetry.jsonl | grep -E '"event": "bulk_run_(start|end)"'
```

Expected: each matching line is one JSON event. The `bulk_run_start` line contains `"subtype": "venn"`. The `bulk_run_end` line contains both `"subtype": "venn"` and `"drift_count": <int>`.

- [ ] **Step 4: Commit**

```bash
git add ucat/ui.py
git commit -m "feat: telemetry includes subtype + drift_count for bulk runs"
```

---

## Task 13: End-to-end smoke checklist

**Files:**
- None modified. This task is a manual verification sweep.

- [ ] **Step 1: Run the full unit-test suite**

```bash
./venv/bin/python tests/test_subtype_targeting.py
./venv/bin/python tests/test_schema.py
./venv/bin/python tests/test_bulk_cost.py
./venv/bin/python tests/test_coverage.py
```

Expected: every file ends with `All tests passed.`

- [ ] **Step 2: Smoke — DM Venn**

Launch the app, BULK tab, Section=DM, Subtype=Venn, Quantity=10. Click START.

Verify:
- Helper line shows `→ 2 sets × 5 questions = 10 questions`.
- Two sets generated. Each row's Subtype column = "Venn".
- Click each row: preview shows 5 questions, each with `type: "venn"` and a Venn diagram rendered.

- [ ] **Step 3: Smoke — DM Logical with extras**

Section=DM, Subtype=Logical (logic puzzles), Quantity=7.

Verify:
- Helper line shows `→ 2 sets × 5 questions = 10 questions  (3 extra)`.
- Two sets generated; each contains 5 logic-puzzle questions.

- [ ] **Step 4: Smoke — VR Main idea**

Section=VR, Subtype=Main idea, Quantity=8.

Verify:
- Helper line shows `→ 2 sets × 4 questions = 8 questions`.
- Two sets generated; each is a passage + 4 main-idea questions (`minigame_kind: main-idea`, `type: mc`, 4 options).

- [ ] **Step 5: Smoke — VR TFC**

Section=VR, Subtype=True / False / Can't Tell, Quantity=4.

Verify:
- One set generated with 4 questions, each having 3 options (`True`, `False`, `Can't Tell`), `type: tf`, `minigame_kind: tfc`.

- [ ] **Step 6: Smoke — QR Bar chart**

Section=QR, Subtype=Bar chart, Quantity=4.

Verify:
- One set generated with stimulus `type: bar`, populated `categories` and `series[]`, 4 calculation questions.

- [ ] **Step 7: Smoke — Backwards compat (Any/mixed)**

Section=DM, Subtype=Any (mixed), Quantity=2.

Verify:
- Quantity label reads `Sets:`. Helper line is empty.
- Two sets generated. Each contains a *mix* of subtypes (one of each ideally) — i.e. behavior identical to pre-change bulk runs.

- [ ] **Step 8: Smoke — AR unchanged**

Section=AR.

Verify:
- Subtype dropdown is disabled and locked on "Any (mixed)".
- Quantity label reads `Sets:`.
- Generation works as today (no errors related to subtype).

- [ ] **Step 9: Smoke — Per-section memory**

Set DM=Venn, then switch to QR and pick Bar chart, then switch back to DM.

Verify: DM dropdown restores to "Venn". Switch to QR: restores to "Bar chart".

Restart the app. Verify the last-used subtypes per section are still restored.

- [ ] **Step 10: Smoke — Schema backwards compat**

Open the **HISTORY** tab and view an existing pre-change DM or VR set (one without `minigame_kind`). Verify it renders correctly and re-validates without errors. Ensure no Python tracebacks in the log.

- [ ] **Step 11: Smoke — Drift badge surfacing (optional, opportunistic)**

If during any of the above runs a `⚠ drift` badge appears, click the row and verify the preview still renders the questions (drift is a warning, not a failure). Confirm the final status-bar message includes the drift count.

- [ ] **Step 12: Final commit (no code change — just a marker)**

If any docs or comments need a touch-up after smoke testing, commit them. Otherwise this step is a no-op — the feature is complete.

```bash
git status
# If clean, you're done. Otherwise commit any tweaks discovered during smoke.
```

---

End of plan.
