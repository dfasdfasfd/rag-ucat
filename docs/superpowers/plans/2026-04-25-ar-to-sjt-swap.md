# UCAT Trainer: AR → SJT Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Abstract Reasoning (AR) section with Situational Judgement (SJT) throughout the active `ucat/` package, achieving full parity with VR/DM/QR for generation, retrieval, calibration, coverage, and UI.

**Architecture:** Single PR. Schema first (pydantic + label constants in `models.py`), then config changes, then verifier check, then per-section logic (db, samples, format, calibration, coverage), then rendering/UI cleanup, then generation prompts, then manual smoke. Existing AR rows in the SQLite DB stay but are filtered out of UI listings; the `section` column is unconstrained TEXT so no schema migration is needed.

**Tech Stack:** Python 3.11+, pydantic v2, sqlite3, matplotlib (renderer), tkinter (UI), Anthropic + Voyage SDKs. Tests are plain `def test_*` functions runnable directly via `./venv/bin/python tests/test_<name>.py`.

**Spec:** [docs/superpowers/specs/2026-04-25-ar-to-sjt-swap-design.md](../specs/2026-04-25-ar-to-sjt-swap-design.md)

---

## File Structure

**Modified (all in `ucat/`):**
- `models.py` — drop AR pydantic classes, add `SJTQuestion` + `SJTSet`, add `APPROPRIATENESS_LABELS` / `IMPORTANCE_LABELS` / `SJT_LABELS` constants, update `SECTION_MODELS`
- `config.py` — `SECTIONS`, `SECTION_COLORS`, `SECTION_DESC`, `Settings.load()` defensive coercion
- `verification.py` — pre-LLM syntactic check on SJT label maps
- `db.py` — replace AR branch in `embed_text_for` with SJT branch
- `samples.py` — replace AR sample with one SJT sample
- `format.py` — replace AR rendering block with SJT block
- `calibration.py` — replace AR branch in `feature_difficulty` with SJT branch
- `coverage.py` — `EXPECTED_SCENARIOS["SJT"]`, drop `["AR"]`
- `rendering.py` — delete `render_ar_panel`, `render_ar_set`, AR branch in `render_visuals_for`, `_AR_FILL` / `_AR_EDGE` constants
- `rag.py` — drop AR prompt blocks, add SJT prompt block
- `ui.py` — drop AR visuals branch, filter AR from KB browser, drop AR-specific copy in tooltips, log count of hidden AR rows on startup

**New:**
- `tests/test_sjt.py` — schema, settings coercion, verifier, calibration, coverage, db embed-text tests

**Untouched:**
- `crawler_import.py` — already handles SJT and skips AR
- `src/` — orphaned, separate spec
- `requirements.txt`, `assets/`, DB schema

---

## Task 1: SJT schema and label constants in `models.py`

**Files:**
- Modify: `ucat/models.py:53-82` (drop AR classes), `ucat/models.py:147` (`SECTION_MODELS`)
- Modify: `ucat/models.py` (add SJT classes + label constants)
- Test: `tests/test_sjt.py` (new)

- [ ] **Step 1.1: Create `tests/test_sjt.py` with the failing schema tests**

```python
"""Tests for the SJT section schema, settings coercion, verifier, calibration,
coverage, and db embed-text. Runnable directly:

    ./venv/bin/python tests/test_sjt.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import ValidationError


# ─── Schema tests ────────────────────────────────────────────────────────────

def _valid_sjt_set() -> dict:
    """A minimal SJT set that should validate."""
    from ucat.models import APPROPRIATENESS_LABELS, IMPORTANCE_LABELS
    return {
        "section": "SJT",
        "scenario": (
            "You are a first-year doctor on a busy ward. A patient asks you about "
            "their diagnosis, but the consultant has not yet briefed them. The patient "
            "is becoming distressed and insists on knowing now. Other staff are "
            "occupied with an emergency in the next bay."
        ),
        "questions": [
            {"number": 1, "text": "How appropriate is it to tell the patient the diagnosis yourself?",
             "rating_type": "appropriateness",
             "options": dict(APPROPRIATENESS_LABELS),
             "answer": "D",
             "difficulty": 3.0,
             "explanation": "An F1 should not deliver new diagnoses; this is the consultant's role.",
             "coverage": {"topic": "professionalism", "scenario_type": "medical",
                          "contains_named_entities": False, "cultural_context": "UK"}},
            {"number": 2, "text": "How appropriate is it to ask the patient to wait for the consultant?",
             "rating_type": "appropriateness",
             "options": dict(APPROPRIATENESS_LABELS),
             "answer": "B",
             "difficulty": 2.5,
             "explanation": "Reasonable but not ideal — the patient is already distressed.",
             "coverage": {"topic": "professionalism", "scenario_type": "medical",
                          "contains_named_entities": False, "cultural_context": "UK"}},
            {"number": 3, "text": "How important is it to acknowledge the patient's distress?",
             "rating_type": "importance",
             "options": dict(IMPORTANCE_LABELS),
             "answer": "A",
             "difficulty": 2.0,
             "explanation": "Empathy is core to the doctor-patient relationship.",
             "coverage": {"topic": "communication", "scenario_type": "medical",
                          "contains_named_entities": False, "cultural_context": "UK"}},
            {"number": 4, "text": "How important is it to find the consultant immediately?",
             "rating_type": "importance",
             "options": dict(IMPORTANCE_LABELS),
             "answer": "B",
             "difficulty": 2.5,
             "explanation": "Important — but the consultant is busy with the emergency, so balance is required.",
             "coverage": {"topic": "professionalism", "scenario_type": "medical",
                          "contains_named_entities": False, "cultural_context": "UK"}},
        ],
    }


def test_sjt_set_accepts_valid_payload():
    from ucat.models import SJTSet
    SJTSet.model_validate(_valid_sjt_set())  # must not raise


def test_sjt_set_rejects_short_scenario():
    from ucat.models import SJTSet
    payload = _valid_sjt_set()
    payload["scenario"] = "too short"  # < 50 chars
    try:
        SJTSet.model_validate(payload)
    except ValidationError:
        return
    raise AssertionError("expected ValidationError for short scenario")


def test_sjt_set_rejects_wrong_question_count():
    from ucat.models import SJTSet
    payload = _valid_sjt_set()
    payload["questions"] = payload["questions"][:3]  # only 3
    try:
        SJTSet.model_validate(payload)
    except ValidationError:
        return
    raise AssertionError("expected ValidationError for 3-question set")


def test_sjt_question_requires_rating_type():
    from ucat.models import SJTSet
    payload = _valid_sjt_set()
    del payload["questions"][0]["rating_type"]
    try:
        SJTSet.model_validate(payload)
    except ValidationError:
        return
    raise AssertionError("expected ValidationError for missing rating_type")


def test_sjt_labels_exposed_via_constant():
    from ucat.models import APPROPRIATENESS_LABELS, IMPORTANCE_LABELS, SJT_LABELS
    assert APPROPRIATENESS_LABELS["A"] == "Very appropriate"
    assert APPROPRIATENESS_LABELS["D"] == "Very inappropriate"
    assert IMPORTANCE_LABELS["A"] == "Very important"
    assert IMPORTANCE_LABELS["D"] == "Not important at all"
    assert SJT_LABELS["appropriateness"] is APPROPRIATENESS_LABELS
    assert SJT_LABELS["importance"] is IMPORTANCE_LABELS


def test_section_models_swap_ar_for_sjt():
    from ucat.models import SECTION_MODELS, SJTSet
    assert "AR" not in SECTION_MODELS
    assert SECTION_MODELS["SJT"] is SJTSet


# ─── Test runner ─────────────────────────────────────────────────────────────

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

- [ ] **Step 1.2: Run the test, verify schema tests fail**

Run: `./venv/bin/python tests/test_sjt.py`
Expected: Errors like `ImportError: cannot import name 'SJTSet' from 'ucat.models'` or `cannot import name 'APPROPRIATENESS_LABELS'`.

- [ ] **Step 1.3: Edit `ucat/models.py` — drop AR classes**

Delete these blocks entirely from `ucat/models.py:53-82`:

```python
# ─── Visual specs ─────────────────────────────────────────────────────────────

# AR — abstract reasoning shape sets.

ARShapeKind  = Literal["square", "circle", "triangle", "diamond", "star", "pentagon", "hexagon", "cross", "arrow"]
ARShapeColor = Literal["black", "white", "grey"]
ARShapeSize  = Literal["small", "medium", "large"]


class ARShape(BaseModel):
    kind: ARShapeKind
    color: ARShapeColor = "black"
    size: ARShapeSize = "medium"
    rotation_deg: int = Field(default=0, ge=0, le=359)
    # Position is auto-laid-out by the renderer; the model doesn't pick coords.


class ARPanel(BaseModel):
    label: str = ""           # optional human label, e.g. "Panel 1"
    shapes: List[ARShape]


class ARSet(BaseModel):
    section: Literal["AR"]
    set_a_panels: List[ARPanel] = Field(min_length=6, max_length=6)
    set_a_rule: str
    set_b_panels: List[ARPanel] = Field(min_length=6, max_length=6)
    set_b_rule: str
    test_panels: List[ARPanel] = Field(min_length=5, max_length=5,
                                         description="Test shapes shown to the student — one per question.")
    questions: List[Question] = Field(min_length=5, max_length=5)
```

Replace the `# ─── Visual specs ───` heading with `# ─── QR visual spec ───` (since QR is the only remaining structured-visual section).

- [ ] **Step 1.4: Edit `ucat/models.py` — add SJT classes**

After the `VRSet` class at line 144 (just above `SECTION_MODELS`), insert:

```python
# ─── SJT — Situational Judgement ─────────────────────────────────────────────

# Canonical option-label maps. Each SJT question's `rating_type` selects which
# map applies. These constants are also imported by:
#   • verification.py — to syntactically reject mismatched labels before LLM judge
#   • rag.py          — to interpolate the labels into the generation prompt
#   • format.py       — to annotate answers in the human-readable output
APPROPRIATENESS_LABELS = {
    "A": "Very appropriate",
    "B": "Appropriate, but not ideal",
    "C": "Inappropriate, but not awful",
    "D": "Very inappropriate",
}
IMPORTANCE_LABELS = {
    "A": "Very important",
    "B": "Important",
    "C": "Of minor importance",
    "D": "Not important at all",
}
SJT_LABELS = {
    "appropriateness": APPROPRIATENESS_LABELS,
    "importance": IMPORTANCE_LABELS,
}


class SJTQuestion(Question):
    """An SJT question rates one action or consideration on a 4-point scale."""
    rating_type: Literal["appropriateness", "importance"]


class SJTSet(BaseModel):
    section: Literal["SJT"]
    scenario: str = Field(min_length=50,
                            description="Realistic clinical/professional situation, 80-150 words.")
    questions: List[SJTQuestion] = Field(min_length=4, max_length=4)
```

- [ ] **Step 1.5: Edit `ucat/models.py:147` — update `SECTION_MODELS`**

Change:

```python
SECTION_MODELS = {"VR": VRSet, "DM": DMSet, "QR": QRSet, "AR": ARSet}
```

To:

```python
SECTION_MODELS = {"VR": VRSet, "DM": DMSet, "QR": QRSet, "SJT": SJTSet}
```

- [ ] **Step 1.6: Run the test, verify schema tests pass**

Run: `./venv/bin/python tests/test_sjt.py`
Expected: `PASS  test_sjt_set_accepts_valid_payload`, `PASS  test_sjt_set_rejects_short_scenario`, `PASS  test_sjt_set_rejects_wrong_question_count`, `PASS  test_sjt_question_requires_rating_type`, `PASS  test_sjt_labels_exposed_via_constant`, `PASS  test_section_models_swap_ar_for_sjt`. All tests pass.

- [ ] **Step 1.7: Commit**

```bash
git add ucat/models.py tests/test_sjt.py
git commit -m "models: replace ARSet with SJTSet + canonical label constants"
```

---

## Task 2: Section config and Settings coercion

**Files:**
- Modify: `ucat/config.py:69-84` (`SECTIONS`, `SECTION_COLORS`, `SECTION_DESC`)
- Modify: `ucat/config.py:124-133` (`Settings.load`)
- Test: `tests/test_sjt.py` (extend)

- [ ] **Step 2.1: Add failing tests to `tests/test_sjt.py`**

Append these test functions to `tests/test_sjt.py` *before* the `if __name__ == "__main__":` block:

```python
# ─── Config tests ────────────────────────────────────────────────────────────

def test_sections_swap_ar_for_sjt():
    from ucat.config import SECTIONS, SECTION_COLORS, SECTION_DESC
    assert "AR" not in SECTIONS
    assert SECTIONS["SJT"] == "Situational Judgement"
    assert SECTION_COLORS["SJT"] == "#A78BFA"
    assert "AR" not in SECTION_COLORS
    assert "scenario" in SECTION_DESC["SJT"].lower()
    assert "AR" not in SECTION_DESC


def test_settings_coerces_unknown_bulk_section_to_vr(tmp_path=None):
    """If ucat_settings.json holds a stale section like 'AR', Settings.load
    must coerce bulk_section back to a valid value."""
    import json
    import tempfile
    from ucat.config import Settings, SECTIONS

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "settings.json")
        with open(path, "w") as f:
            json.dump({"bulk_section": "AR"}, f)
        s = Settings(path=path)
        assert s.get("bulk_section") in SECTIONS, (
            f"expected coerced bulk_section in {set(SECTIONS)}, "
            f"got {s.get('bulk_section')!r}"
        )
        assert s.get("bulk_section") == "VR"


def test_settings_preserves_valid_bulk_section():
    import json
    import tempfile
    from ucat.config import Settings

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "settings.json")
        with open(path, "w") as f:
            json.dump({"bulk_section": "QR"}, f)
        s = Settings(path=path)
        assert s.get("bulk_section") == "QR"
```

- [ ] **Step 2.2: Run the new tests, verify they fail**

Run: `./venv/bin/python tests/test_sjt.py`
Expected: `FAIL  test_sections_swap_ar_for_sjt: ...` (since `AR` still in `SECTIONS`), and `FAIL  test_settings_coerces_unknown_bulk_section_to_vr` (no coercion yet).

- [ ] **Step 2.3: Edit `ucat/config.py:69-84` — replace section dicts**

Change:

```python
SECTIONS = {
    "VR": "Verbal Reasoning",
    "DM": "Decision Making",
    "QR": "Quantitative Reasoning",
    "AR": "Abstract Reasoning",
}
SECTION_COLORS = {
    "VR": "#4A90D9", "DM": "#E8943A",
    "QR": "#3FB950", "AR": "#A78BFA",
}
SECTION_DESC = {
    "VR": "A passage (200-300 words) followed by exactly 4 questions. Each question is either True/False/Can't Tell OR 4-option multiple choice (A-D). Questions answerable ONLY from the passage.",
    "DM": "Exactly 5 standalone questions. Each is one of: syllogism, logical (clue-based), venn (set relationships), probability, or argument (strongest argument for/against). Each has 5 options (A-E). Venn questions MUST include a structured set spec (sets[]) for visual rendering.",
    "QR": "One data stimulus (table, bar chart, or line chart) followed by exactly 4 calculation questions. Each has 5 numerical options (A-E). Step-by-step working in each explanation. Stimulus MUST be provided as a structured chart spec for visual rendering.",
    "AR": "Type 1 set. Set A (6 panels with shape sets, hidden rule). Set B (6 panels with shape sets, different rule). Then 5 test shapes answered Set A / Set B / Neither. Panels MUST be provided as structured shape specs for visual rendering.",
}
```

To:

```python
SECTIONS = {
    "VR": "Verbal Reasoning",
    "DM": "Decision Making",
    "QR": "Quantitative Reasoning",
    "SJT": "Situational Judgement",
}
SECTION_COLORS = {
    "VR": "#4A90D9", "DM": "#E8943A",
    "QR": "#3FB950", "SJT": "#A78BFA",
}
SECTION_DESC = {
    "VR": "A passage (200-300 words) followed by exactly 4 questions. Each question is either True/False/Can't Tell OR 4-option multiple choice (A-D). Questions answerable ONLY from the passage.",
    "DM": "Exactly 5 standalone questions. Each is one of: syllogism, logical (clue-based), venn (set relationships), probability, or argument (strongest argument for/against). Each has 5 options (A-E). Venn questions MUST include a structured set spec (sets[]) for visual rendering.",
    "QR": "One data stimulus (table, bar chart, or line chart) followed by exactly 4 calculation questions. Each has 5 numerical options (A-E). Step-by-step working in each explanation. Stimulus MUST be provided as a structured chart spec for visual rendering.",
    "SJT": "A scenario (80-150 words) describing a realistic professional/clinical situation, followed by exactly 4 questions. Each question rates one action or consideration on a 4-point scale (appropriateness OR importance, A-D). Options are fixed by rating type.",
}
```

- [ ] **Step 2.4: Edit `ucat/config.py:124-133` — add `Settings.load()` coercion**

Change the `load` method:

```python
    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    saved = json.load(f)
                for k in self.DEFAULTS:
                    if k in saved:
                        self.data[k] = saved[k]
            except Exception:
                pass
```

To:

```python
    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    saved = json.load(f)
                for k in self.DEFAULTS:
                    if k in saved:
                        self.data[k] = saved[k]
            except Exception:
                pass
        # Coerce stale section values (e.g. "AR" left over from before the SJT swap).
        if self.data.get("bulk_section") not in SECTIONS:
            self.data["bulk_section"] = self.DEFAULTS["bulk_section"]
```

- [ ] **Step 2.5: Run the tests, verify they pass**

Run: `./venv/bin/python tests/test_sjt.py`
Expected: All previous tests still PASS, plus `PASS  test_sections_swap_ar_for_sjt`, `PASS  test_settings_coerces_unknown_bulk_section_to_vr`, `PASS  test_settings_preserves_valid_bulk_section`.

- [ ] **Step 2.6: Commit**

```bash
git add ucat/config.py tests/test_sjt.py
git commit -m "config: swap AR for SJT in SECTIONS, coerce stale bulk_section"
```

---

## Task 3: SJT verification syntactic check

**Files:**
- Modify: `ucat/verification.py` (add helper + integrate into `llm_judge`)
- Test: `tests/test_sjt.py` (extend)

- [ ] **Step 3.1: Add failing tests to `tests/test_sjt.py`**

Append before `if __name__ == "__main__":`:

```python
# ─── Verification tests ──────────────────────────────────────────────────────

def test_sjt_label_check_passes_for_canonical_options():
    from ucat.verification import sjt_label_check
    payload = _valid_sjt_set()
    ok, note = sjt_label_check(payload)
    assert ok is True, f"expected ok=True, got ({ok!r}, {note!r})"
    assert note == "" or note is None


def test_sjt_label_check_rejects_paraphrased_options():
    from ucat.verification import sjt_label_check
    payload = _valid_sjt_set()
    payload["questions"][0]["options"]["A"] = "A very good response"  # paraphrase
    ok, note = sjt_label_check(payload)
    assert ok is False, "expected rejection of paraphrased option"
    assert "Q1" in (note or ""), f"note should name the question: {note!r}"


def test_sjt_label_check_rejects_wrong_label_set():
    from ucat.verification import sjt_label_check
    payload = _valid_sjt_set()
    # Question 3 has rating_type=importance; swap in appropriateness labels.
    from ucat.models import APPROPRIATENESS_LABELS
    payload["questions"][2]["options"] = dict(APPROPRIATENESS_LABELS)
    ok, note = sjt_label_check(payload)
    assert ok is False, "expected rejection of mismatched label set"
    assert "Q3" in (note or ""), f"note should name Q3: {note!r}"


def test_sjt_label_check_skips_non_sjt_sets():
    from ucat.verification import sjt_label_check
    ok, note = sjt_label_check({"section": "VR", "passage": "...", "questions": []})
    assert ok is True, "non-SJT sets should pass through"
```

- [ ] **Step 3.2: Run the new tests, verify they fail**

Run: `./venv/bin/python tests/test_sjt.py`
Expected: `ERROR test_sjt_label_check_passes_for_canonical_options: ImportError: cannot import name 'sjt_label_check' from 'ucat.verification'` and similar for the other three.

- [ ] **Step 3.3: Edit `ucat/verification.py` — add `sjt_label_check`**

Find the import block at the top of `ucat/verification.py`. Add to the `from .models import` line (or add a new import line just below it):

```python
from .models import SJT_LABELS
```

Then add this function near the top of the file, just below the imports and before `_judge_system_blocks`:

```python
def sjt_label_check(data: Dict[str, Any]) -> Tuple[bool, str]:
    """Pre-LLM syntactic check: SJT options must use canonical fixed labels.

    Returns (True, "") if the set is OK or not an SJT set.
    Returns (False, reason) if any SJT question has options that diverge from
    the canonical map for its rating_type. The reason names the offending
    question (e.g. "Q3 has wrong options for rating_type=importance").
    """
    if data.get("section") != "SJT":
        return True, ""
    for q in data.get("questions") or []:
        rt = q.get("rating_type")
        if rt not in SJT_LABELS:
            return False, f"Q{q.get('number','?')} has unknown rating_type={rt!r}"
        expected = SJT_LABELS[rt]
        actual = q.get("options") or {}
        if actual != expected:
            return False, (
                f"Q{q.get('number','?')} has wrong options for rating_type={rt}: "
                f"expected keys/values {sorted(expected.items())}, "
                f"got {sorted(actual.items())}"
            )
    return True, ""
```

- [ ] **Step 3.4: Edit `ucat/verification.py` — integrate the check into `llm_judge`**

Find `llm_judge` (around line 74). Change:

```python
def llm_judge(section: str, data: Dict[str, Any], model: str
              ) -> Tuple[_JudgeVerdict, Dict[str, Any]]:
    """Run a single LLM judge over one generated set."""
    client = LLMClient.anthropic()
    user = (
```

To:

```python
def llm_judge(section: str, data: Dict[str, Any], model: str
              ) -> Tuple[_JudgeVerdict, Dict[str, Any]]:
    """Run a single LLM judge over one generated set."""
    # Pre-LLM syntactic check: catch mismatched SJT option labels before paying
    # for the judge call.
    sjt_ok, sjt_note = sjt_label_check(data)
    if not sjt_ok:
        v = _JudgeVerdict(
            per_question=[],
            overall_correct=False,
            confidence="high",
            notes=[f"SJT label mismatch: {sjt_note}"],
        )
        # Mirror the dict shape that extract_usage returns (see ucat/llm.py:192-196).
        empty_usage = {
            "input_tokens": 0, "output_tokens": 0,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
            "cost_usd": 0.0, "model": model,
        }
        return v, empty_usage
    client = LLMClient.anthropic()
    user = (

- [ ] **Step 3.5: Run the tests, verify they pass**

Run: `./venv/bin/python tests/test_sjt.py`
Expected: All previous tests still PASS, plus `PASS  test_sjt_label_check_passes_for_canonical_options`, `PASS  test_sjt_label_check_rejects_paraphrased_options`, `PASS  test_sjt_label_check_rejects_wrong_label_set`, `PASS  test_sjt_label_check_skips_non_sjt_sets`.

- [ ] **Step 3.6: Commit**

```bash
git add ucat/verification.py tests/test_sjt.py
git commit -m "verification: add SJT canonical-label syntactic check"
```

---

## Task 4: Replace AR embed-text branch in `db.py`

**Files:**
- Modify: `ucat/db.py:43-50` (AR branch in `embed_text_for`)
- Test: `tests/test_sjt.py` (extend)

- [ ] **Step 4.1: Add failing test to `tests/test_sjt.py`**

Append:

```python
# ─── Embed-text tests ────────────────────────────────────────────────────────

def test_embed_text_for_sjt_includes_scenario_and_rating_types():
    from ucat.db import embed_text_for
    payload = _valid_sjt_set()
    text = embed_text_for(payload)
    assert "first-year doctor" in text, "scenario should appear in embed text"
    assert "appropriateness" in text, "rating types should appear in embed text"
    assert "importance" in text, "rating types should appear in embed text"


def test_embed_text_for_sjt_does_not_reference_ar_fields():
    from ucat.db import embed_text_for
    payload = _valid_sjt_set()
    text = embed_text_for(payload)
    assert "Set A rule" not in text
    assert "Set B rule" not in text
```

- [ ] **Step 4.2: Run, verify the new tests fail**

Run: `./venv/bin/python tests/test_sjt.py`
Expected: `FAIL  test_embed_text_for_sjt_includes_scenario_and_rating_types: ...` (scenario / rating types absent).

- [ ] **Step 4.3: Edit `ucat/db.py:43-50` — replace AR branch with SJT branch**

Find:

```python
    # AR — rule summary.
    if section == "AR":
        parts.append(f"Set A rule: {str(data.get('set_a_rule',''))[:200]}")
        parts.append(f"Set B rule: {str(data.get('set_b_rule',''))[:200]}")
        # back-compat with legacy text descriptions
        for k in ("set_a_description", "set_b_description"):
            if data.get(k):
                parts.append(str(data[k])[:300])
```

Replace with:

```python
    # SJT — scenario + rating-type mix.
    if section == "SJT":
        parts.append(str(data.get("scenario", ""))[:1200])
        rating_types = [q.get("rating_type", "?") for q in (data.get("questions") or [])]
        if rating_types:
            parts.append(f"Rating types: {', '.join(rating_types)}")
```

- [ ] **Step 4.4: Run, verify tests pass**

Run: `./venv/bin/python tests/test_sjt.py`
Expected: All previous PASS, plus `PASS  test_embed_text_for_sjt_includes_scenario_and_rating_types`, `PASS  test_embed_text_for_sjt_does_not_reference_ar_fields`.

- [ ] **Step 4.5: Commit**

```bash
git add ucat/db.py tests/test_sjt.py
git commit -m "db: replace AR embed-text branch with SJT (scenario + rating types)"
```

---

## Task 5: Replace AR sample with SJT sample in `samples.py`

**Files:**
- Modify: `ucat/samples.py:167-254` (delete the AR sample; insert SJT sample)
- Test: `tests/test_sjt.py` (extend)

- [ ] **Step 5.1: Add failing test to `tests/test_sjt.py`**

Append:

```python
# ─── Sample seed tests ───────────────────────────────────────────────────────

def test_samples_have_no_ar_section():
    from ucat.samples import SAMPLES
    sections = {s["section"] for s in SAMPLES}
    assert "AR" not in sections, f"expected AR removed, got {sections}"
    assert "SJT" in sections, f"expected SJT present, got {sections}"


def test_samples_sjt_validates_against_schema():
    from ucat.samples import SAMPLES
    from ucat.models import SJTSet
    sjt = next(s for s in SAMPLES if s["section"] == "SJT")
    SJTSet.model_validate(sjt)  # must not raise
```

- [ ] **Step 5.2: Run, verify tests fail**

Run: `./venv/bin/python tests/test_sjt.py`
Expected: `FAIL  test_samples_have_no_ar_section: expected AR removed, got {'VR', 'DM', 'QR', 'AR'}`.

- [ ] **Step 5.3: Edit `ucat/samples.py:167-254` — delete the entire AR block and insert SJT**

Delete lines 167–254 inclusive (the AR sample dict — the block starting `# ────────────────────────────── AR ──────────────────────────────` and ending with the closing `},`). Replace with:

```python
    # ────────────────────────────── SJT ─────────────────────────────
    {
        "section": "SJT",
        "scenario": (
            "You are a foundation-year-one (FY1) doctor on a busy general medical ward. A patient "
            "you have been looking after asks you about the results of an MRI scan that arrived "
            "this morning. The consultant is currently with another patient and has not yet "
            "reviewed the results, but has asked the team not to discuss new findings with patients "
            "before the consultant ward round. The patient is becoming increasingly anxious and "
            "tells you they cannot wait until the afternoon round to know what the scan showed."
        ),
        "questions": [
            {"number": 1,
             "text": "How appropriate is it for the FY1 to read out the MRI report to the patient?",
             "rating_type": "appropriateness",
             "options": {
                 "A": "Very appropriate",
                 "B": "Appropriate, but not ideal",
                 "C": "Inappropriate, but not awful",
                 "D": "Very inappropriate",
             },
             "answer": "D", "difficulty": 2.0,
             "explanation": (
                 "An FY1 should not break new diagnostic findings to a patient — that is the "
                 "consultant's role, and the team has been explicitly asked to wait. Reading out "
                 "the report directly bypasses both the team plan and the seniority required for "
                 "this kind of conversation."
             ),
             "coverage": _cov("professionalism", "medical")},
            {"number": 2,
             "text": "How appropriate is it to acknowledge the patient's anxiety and explain that the consultant will discuss the scan on the afternoon round?",
             "rating_type": "appropriateness",
             "options": {
                 "A": "Very appropriate",
                 "B": "Appropriate, but not ideal",
                 "C": "Inappropriate, but not awful",
                 "D": "Very inappropriate",
             },
             "answer": "A", "difficulty": 1.5,
             "explanation": (
                 "Empathic acknowledgement plus a clear plan respects both the patient's distress "
                 "and the team's protocol. This is the standard professional response."
             ),
             "coverage": _cov("communication", "medical")},
            {"number": 3,
             "text": "How important is it to inform the consultant that the patient is asking about the results before the round?",
             "rating_type": "importance",
             "options": {
                 "A": "Very important",
                 "B": "Important",
                 "C": "Of minor importance",
                 "D": "Not important at all",
             },
             "answer": "A", "difficulty": 2.0,
             "explanation": (
                 "The consultant needs to know so they can adjust the round order or send a "
                 "registrar earlier. Withholding this changes the consultant's planning."
             ),
             "coverage": _cov("teamwork", "medical")},
            {"number": 4,
             "text": "How important is it to consider whether the patient has the capacity to wait for further information given their anxiety?",
             "rating_type": "importance",
             "options": {
                 "A": "Very important",
                 "B": "Important",
                 "C": "Of minor importance",
                 "D": "Not important at all",
             },
             "answer": "B", "difficulty": 3.0,
             "explanation": (
                 "Patient autonomy and welfare matter, but anxiety alone rarely justifies "
                 "overriding the team plan; clinical judgement on the balance is what matters."
             ),
             "coverage": _cov("ethics", "medical")},
        ],
    },
```

- [ ] **Step 5.4: Run, verify tests pass**

Run: `./venv/bin/python tests/test_sjt.py`
Expected: All previous PASS, plus `PASS  test_samples_have_no_ar_section`, `PASS  test_samples_sjt_validates_against_schema`.

- [ ] **Step 5.5: Commit**

```bash
git add ucat/samples.py tests/test_sjt.py
git commit -m "samples: replace AR seed sample with FY1 ward SJT scenario"
```

---

## Task 6: Replace AR rendering branch in `format.py` with SJT

**Files:**
- Modify: `ucat/format.py:42-55` (AR rendering block)
- Test: `tests/test_sjt.py` (extend)

- [ ] **Step 6.1: Add failing test**

Append to `tests/test_sjt.py`:

```python
# ─── Format tests ────────────────────────────────────────────────────────────

def test_format_qset_for_sjt_shows_scenario_and_rating_types():
    from ucat.format import format_qset
    text = format_qset(_valid_sjt_set())
    assert "SCENARIO" in text or "Scenario" in text, "scenario heading expected"
    assert "first-year doctor" in text, "scenario body expected"
    assert "appropriateness" in text or "Appropriateness" in text, \
        "rating type label expected near each Q"
```

- [ ] **Step 6.2: Run, verify it fails**

Run: `./venv/bin/python tests/test_sjt.py`
Expected: `FAIL  test_format_qset_for_sjt_shows_scenario_and_rating_types: scenario heading expected`.

- [ ] **Step 6.3: Edit `ucat/format.py:42-55` — replace AR block with SJT block**

Find:

```python
    # AR sets.
    if section == "AR":
        if data.get("set_a_rule"):
            lines += ["SET A — RULE", "─" * 40,
                       textwrap.fill(str(data["set_a_rule"]), 72), ""]
        if data.get("set_b_rule"):
            lines += ["SET B — RULE", "─" * 40,
                       textwrap.fill(str(data["set_b_rule"]), 72), ""]
        if data.get("set_a_description"):    # legacy
            lines += ["SET A (legacy)", "─" * 40, str(data["set_a_description"]), ""]
        if data.get("set_b_description"):
            lines += ["SET B (legacy)", "─" * 40, str(data["set_b_description"]), ""]
        lines.append("(See panel images for shape sets)")
        lines.append("")
```

Replace with:

```python
    # SJT scenario.
    if section == "SJT" and data.get("scenario"):
        lines += ["SCENARIO", "─" * 40]
        for para in str(data["scenario"]).split("\n"):
            lines.append(textwrap.fill(para, 72) if para.strip() else "")
        lines.append("")
```

Then find the question loop further down (the block starting `for q in data.get("questions", []) or []:`) and add the rating-type annotation by changing:

```python
        if q.get("type"):
            lines.append(f"       (type: {q['type']})")
```

To:

```python
        if q.get("type"):
            lines.append(f"       (type: {q['type']})")
        if q.get("rating_type"):
            lines.append(f"       (rating: {q['rating_type']})")
```

- [ ] **Step 6.4: Run, verify the test passes**

Run: `./venv/bin/python tests/test_sjt.py`
Expected: All PASS, plus `PASS  test_format_qset_for_sjt_shows_scenario_and_rating_types`.

- [ ] **Step 6.5: Commit**

```bash
git add ucat/format.py tests/test_sjt.py
git commit -m "format: replace AR rendering block with SJT scenario + rating type"
```

---

## Task 7: Replace AR calibration branch with SJT

**Files:**
- Modify: `ucat/calibration.py:104-111` (AR branch in `feature_difficulty`)
- Test: `tests/test_sjt.py` (extend)

- [ ] **Step 7.1: Add failing tests**

Append:

```python
# ─── Calibration tests ───────────────────────────────────────────────────────

def test_feature_difficulty_returns_clamped_score_for_sjt():
    from ucat.calibration import feature_difficulty
    payload = _valid_sjt_set()
    q = payload["questions"][0]
    score = feature_difficulty(q, "SJT")
    assert 1.0 <= score <= 5.0, f"score must be clamped to [1.0, 5.0], got {score}"


def test_feature_difficulty_for_sjt_rises_with_explanation_length():
    """Longer explanations correlate with more reasoning steps → higher difficulty."""
    from ucat.calibration import feature_difficulty
    short_q = {
        "text": "How appropriate is X?",
        "explanation": "Brief.",
        "rating_type": "appropriateness",
    }
    long_q = {
        "text": "How appropriate is X?",
        "explanation": (
            "This action requires balancing patient autonomy, professional duty, the consultant's "
            "explicit instructions, and the immediate emotional state of the patient. Each factor "
            "must be weighed before deciding whether the proposed action is the most appropriate."
        ),
        "rating_type": "appropriateness",
    }
    s_short = feature_difficulty(short_q, "SJT")
    s_long  = feature_difficulty(long_q, "SJT")
    assert s_long > s_short, f"long explanation should score higher: {s_long} vs {s_short}"
```

- [ ] **Step 7.2: Run, verify they fail**

Run: `./venv/bin/python tests/test_sjt.py`
Expected: Both new tests fail (the existing function returns the fallback `3.0` for SJT, so monotonicity test fails).

- [ ] **Step 7.3: Edit `ucat/calibration.py:104-111` — replace AR branch with SJT branch**

Find:

```python
    if section == "AR":
        # Number of overlapping rules in shapes increases difficulty.
        # Without the panel structure here we approximate from explanation length.
        score = 2.5
        if len(expl) > 80:  score += 0.5
        if "test shape" in text.lower() and " not " in expl.lower():
            score += 0.4   # negative-rule reasoning
        return _clamp(score)
```

Replace with:

```python
    if section == "SJT":
        # SJT difficulty rises with explanation depth (more factors to balance)
        # and named-entity density (more parties to track in the scenario).
        score = 2.5
        if len(expl) > 120: score += 0.6
        elif len(expl) > 60: score += 0.3
        # Named-entity proxy — capitalised words in the scenario/explanation.
        named = len(re.findall(r"\b[A-Z][a-z]{2,}\b", text + " " + expl))
        if named >= 3: score += 0.4
        # Importance questions are typically harder than straight appropriateness.
        if question.get("rating_type") == "importance":
            score += 0.2
        return _clamp(score)
```

- [ ] **Step 7.4: Run, verify tests pass**

Run: `./venv/bin/python tests/test_sjt.py`
Expected: All PASS, plus `PASS  test_feature_difficulty_returns_clamped_score_for_sjt`, `PASS  test_feature_difficulty_for_sjt_rises_with_explanation_length`.

- [ ] **Step 7.5: Commit**

```bash
git add ucat/calibration.py tests/test_sjt.py
git commit -m "calibration: replace AR difficulty branch with SJT signal"
```

---

## Task 8: Update `coverage.EXPECTED_SCENARIOS`

**Files:**
- Modify: `ucat/coverage.py:41-46`
- Test: `tests/test_sjt.py` (extend)

- [ ] **Step 8.1: Add failing test**

Append:

```python
# ─── Coverage tests ──────────────────────────────────────────────────────────

def test_expected_scenarios_swap_ar_for_sjt():
    from ucat.coverage import EXPECTED_SCENARIOS
    assert "AR" not in EXPECTED_SCENARIOS
    assert EXPECTED_SCENARIOS["SJT"] == {"medical", "social", "everyday"}
```

- [ ] **Step 8.2: Run, verify it fails**

Run: `./venv/bin/python tests/test_sjt.py`
Expected: `FAIL  test_expected_scenarios_swap_ar_for_sjt`.

- [ ] **Step 8.3: Edit `ucat/coverage.py:41-46`**

Find:

```python
EXPECTED_SCENARIOS = {
    "VR": {"scientific", "humanities", "business", "social", "everyday"},
    "DM": {"abstract", "everyday", "business", "social"},
    "QR": {"business", "scientific", "everyday", "medical", "sport"},
    "AR": {"abstract"},
}
```

Replace with:

```python
EXPECTED_SCENARIOS = {
    "VR": {"scientific", "humanities", "business", "social", "everyday"},
    "DM": {"abstract", "everyday", "business", "social"},
    "QR": {"business", "scientific", "everyday", "medical", "sport"},
    "SJT": {"medical", "social", "everyday"},
}
```

- [ ] **Step 8.4: Run, verify it passes**

Run: `./venv/bin/python tests/test_sjt.py`
Expected: All PASS, plus `PASS  test_expected_scenarios_swap_ar_for_sjt`.

- [ ] **Step 8.5: Commit**

```bash
git add ucat/coverage.py tests/test_sjt.py
git commit -m "coverage: swap AR for SJT in EXPECTED_SCENARIOS"
```

---

## Task 9: Drop AR rendering helpers in `rendering.py`

**Files:**
- Modify: `ucat/rendering.py:50-51` (`_AR_FILL`, `_AR_EDGE` constants)
- Modify: `ucat/rendering.py:242-405` (`render_ar_panel`, `render_ar_set`, `_draw_shape`)
- Modify: `ucat/rendering.py:503-510` (AR branch in `render_visuals_for`)
- Modify: `ucat/rendering.py:1-11` (module docstring — drop AR mention)

**Note:** `_draw_shape` (line 336) is used only by `render_ar_panel`/`render_ar_set`. Verify with grep before deleting; if any non-AR caller exists, keep the helper.

- [ ] **Step 9.1: Verify `_draw_shape` is only used by AR helpers**

Run: `grep -n "_draw_shape" ucat/rendering.py`
Expected: Three matches — the def at line 336 and two calls inside `render_ar_panel` (line 271) and `render_ar_set` (line 325). If any other call exists, stop and reassess.

- [ ] **Step 9.2: Edit `ucat/rendering.py:1-11` — update module docstring**

Change:

```python
"""Visual rendering for QR charts, AR shape panels, and DM Venn diagrams.

All renderers return PIL Images. The Tkinter UI converts them to PhotoImage
objects for display. Headless callers can save them to disk via `.save(path)`.

Design notes:
  * matplotlib is the workhorse — same backend for charts, shapes, and venns.
  * Renderers are pure functions of their spec; no DB, no LLM. Easy to test.
  * AR panels use matplotlib.patches with a grid layout; rotation is honoured.
  * Falls back gracefully if matplotlib isn't installed (returns a placeholder).
"""
```

To:

```python
"""Visual rendering for QR charts and DM Venn diagrams.

All renderers return PIL Images. The Tkinter UI converts them to PhotoImage
objects for display. Headless callers can save them to disk via `.save(path)`.

Design notes:
  * matplotlib is the workhorse — same backend for charts and venns.
  * Renderers are pure functions of their spec; no DB, no LLM. Easy to test.
  * Falls back gracefully if matplotlib isn't installed (returns a placeholder).
"""
```

- [ ] **Step 9.3: Edit `ucat/rendering.py:50-51` — drop AR colour constants**

Delete these two lines:

```python
_AR_FILL = {"black": "#222222", "white": "#FFFFFF", "grey": "#9CA3AF"}
_AR_EDGE = "#000000"
```

- [ ] **Step 9.4: Delete the AR section block**

Delete the entire block from `# ═══ AR SHAPE PANELS ═══` (line 242) through the end of `_draw_shape` (around line 405, just before `# ═══ DM VENN ═══` or whatever heading follows). This removes:
- `render_ar_panel`
- `render_ar_set`
- `_draw_shape`

Verify the next section heading (DM venn or similar) is intact afterward.

- [ ] **Step 9.5: Edit `ucat/rendering.py:503-510` — drop AR branch in `render_visuals_for`**

Find:

```python
    elif section == "AR":
        if data.get("set_a_panels"):
            out["set_a"] = render_ar_set(data["set_a_panels"], title="Set A")
        if data.get("set_b_panels"):
            out["set_b"] = render_ar_set(data["set_b_panels"], title="Set B")
        tests = data.get("test_panels") or []
        if tests:
            out["tests"] = [render_ar_panel(p, cell_size=140) for p in tests]
```

Delete that block entirely. SJT has no visuals, so no replacement is needed — `render_visuals_for` returns `{}` for SJT via the existing fall-through.

- [ ] **Step 9.6: Verify imports — drop unused `math`, `mp_patches` if no longer used**

Run: `grep -n "math\.\|mp_patches" ucat/rendering.py`
Inspect the output. If `math.` only appeared in deleted AR helpers and no longer appears, delete `import math` from the top of the file. Same for `mp_patches`. Don't delete imports that are still in use by `render_qr_chart` or `render_dm_venn`.

- [ ] **Step 9.7: Smoke-test the module loads cleanly**

Run: `./venv/bin/python -c "from ucat.rendering import render_visuals_for, render_qr_chart, render_dm_venn; print('OK')"`
Expected: `OK`. No import errors.

- [ ] **Step 9.8: Re-run the full test suite to make sure nothing broke**

Run: `./venv/bin/python tests/test_sjt.py && ./venv/bin/python tests/test_schema.py && ./venv/bin/python tests/test_bulk_cost.py`
Expected: All three suites print `All tests passed.`

- [ ] **Step 9.9: Commit**

```bash
git add ucat/rendering.py
git commit -m "rendering: delete AR shape helpers and visuals branch"
```

---

## Task 10: Update generation prompts in `rag.py`

**Files:**
- Modify: `ucat/rag.py:97-110` (output requirements block — drop AR test-item key hint)
- Modify: `ucat/rag.py:119-126` (AR-specific block — replace with SJT block)

- [ ] **Step 10.1: Edit `ucat/rag.py:97-110` — drop the AR key hint**

Find:

```python
            "• `options` MUST be a non-empty dict mapping option labels to option text:\n"
            "    - VR / QR multiple choice → keys A, B, C, D (4 options).\n"
            "    - VR True/False/Can't Tell items → keys exactly \"True\", \"False\", \"Can't Tell\".\n"
            "    - DM → keys A, B, C, D, E (5 options).\n"
            "    - AR test items → keys exactly \"Set A\", \"Set B\", \"Neither\".\n"
            "  Never leave `options` empty. Never reference option text only inside `explanation`.\n"
```

Replace with:

```python
            "• `options` MUST be a non-empty dict mapping option labels to option text:\n"
            "    - VR / QR multiple choice → keys A, B, C, D (4 options).\n"
            "    - VR True/False/Can't Tell items → keys exactly \"True\", \"False\", \"Can't Tell\".\n"
            "    - DM → keys A, B, C, D, E (5 options).\n"
            "    - SJT → keys A, B, C, D with values exactly matching the canonical labels for the question's rating_type (see SJT block below).\n"
            "  Never leave `options` empty. Never reference option text only inside `explanation`.\n"
```

- [ ] **Step 10.2: Edit `ucat/rag.py:119-126` — replace AR block with SJT block**

Find:

```python
        if section == "AR":
            role += (
                "\nCRITICAL — AR panels are STRUCTURED shape lists. "
                "Each panel contains `shapes[]` with `kind`, `color`, `size`, and "
                "`rotation_deg`. Hidden rules can use any combination of shape kind, "
                "color, size, count, rotation, or position. Provide BOTH a structured "
                "spec for rendering AND a clear English description in the rule field.\n"
            )
```

Replace with:

```python
        if section == "SJT":
            from .models import APPROPRIATENESS_LABELS, IMPORTANCE_LABELS
            appr = ", ".join(f'{k}="{v}"' for k, v in APPROPRIATENESS_LABELS.items())
            imp  = ", ".join(f'{k}="{v}"' for k, v in IMPORTANCE_LABELS.items())
            role += (
                "\nCRITICAL — SJT scenarios are realistic professional/clinical "
                "situations (80-150 words) involving an interpersonal or ethical "
                "dilemma a healthcare student/professional might face. Generate "
                "exactly 4 questions per scenario.\n"
                "Each question MUST set `rating_type` to either \"appropriateness\" "
                "or \"importance\" and use the EXACT canonical option labels for "
                "that type — do NOT paraphrase them:\n"
                f"  appropriateness → {appr}\n"
                f"  importance      → {imp}\n"
                "Mix at least one \"importance\" and one \"appropriateness\" "
                "question per set. Avoid obviously correct/incorrect answers; the "
                "most-correct rating should require nuanced judgement.\n"
            )
```

- [ ] **Step 10.3: Sanity-check the prompt at runtime**

Run:

```bash
./venv/bin/python -c "
from ucat.rag import RAGPipeline
from ucat.db import Database
import tempfile, os
d = tempfile.mkdtemp()
db = Database(path=os.path.join(d, 'test.db'))
rp = RAGPipeline(db)
blocks = rp._system_blocks('SJT', [], 3.0)
text = blocks[0]['text']
assert 'CRITICAL — SJT' in text
assert 'Very appropriate' in text
assert 'Very important' in text
assert 'AR' not in text or 'AR' in 'rating_type'  # no stray AR refs
print('prompt OK, length=', len(text))
"
```

Expected: `prompt OK, length= <some number>`. No assertion errors.

- [ ] **Step 10.4: Re-run all tests**

Run: `./venv/bin/python tests/test_sjt.py && ./venv/bin/python tests/test_schema.py && ./venv/bin/python tests/test_bulk_cost.py`
Expected: All `All tests passed.`

- [ ] **Step 10.5: Commit**

```bash
git add ucat/rag.py
git commit -m "rag: drop AR prompt blocks, add SJT prompt with canonical labels"
```

---

## Task 11: Drop AR visuals branch and filter AR from KB browser in `ui.py`

**Files:**
- Modify: `ucat/ui.py:1144-1153` (AR visuals branch)
- Modify: `ucat/ui.py:1275` and `ucat/ui.py:1287` (KB browser queries)
- Modify: `ucat/ui.py:309-313` (tooltip mentioning AR)

- [ ] **Step 11.1: Drop the AR visuals branch at `ui.py:1144-1153`**

Find:

```python
        if section == "AR":
            if "set_a" in visuals:
                self._add_image_to_visuals(visuals["set_a"], label="Set A Panels")
                any_added = True
            if "set_b" in visuals:
                self._add_image_to_visuals(visuals["set_b"], label="Set B Panels")
                any_added = True
            for i, img in enumerate(visuals.get("tests") or []):
                self._add_image_to_visuals(img, label=f"Test Shape {i + 1}")
                any_added = True
```

Delete it entirely. SJT has no visuals — the existing fall-through path renders the "(no visuals for Situational Judgement)" placeholder.

- [ ] **Step 11.2: Filter AR rows from the KB browser table at `ui.py:1275`**

Find:

```python
        docs = self.db.get_all_docs(None if filt == "ALL" else filt, limit=2000)
```

Replace with:

```python
        # Hide retired AR rows: when "ALL" is selected, exclude AR.
        if filt == "ALL":
            docs = [d for d in self.db.get_all_docs(None, limit=2000)
                    if d.get("section") != "AR"]
        else:
            docs = self.db.get_all_docs(filt, limit=2000)
```

- [ ] **Step 11.3: Filter AR rows in the second KB browser callsite at `ui.py:1287`**

Find:

```python
        docs = self.db.get_all_docs(limit=100000)
```

Replace with:

```python
        docs = [d for d in self.db.get_all_docs(limit=100000)
                if d.get("section") != "AR"]
```

- [ ] **Step 11.4: Update the tooltip at `ui.py:309-313` to drop AR mention**

Find (the tooltip text — note the exact line may shift):

```python
                      "automatically for QR charts, AR shapes, and DM venns.",
```

Replace with:

```python
                      "automatically for QR charts and DM venns.",
```

- [ ] **Step 11.5: Smoke-test the module loads cleanly**

Run: `./venv/bin/python -c "from ucat.ui import run; print('OK')"`
Expected: `OK`. No import errors.

- [ ] **Step 11.6: Re-run all tests**

Run: `./venv/bin/python tests/test_sjt.py && ./venv/bin/python tests/test_schema.py && ./venv/bin/python tests/test_bulk_cost.py`
Expected: All `All tests passed.`

- [ ] **Step 11.7: Commit**

```bash
git add ucat/ui.py
git commit -m "ui: drop AR visuals branch, hide AR rows from KB browser"
```

---

## Task 12: Startup log line for hidden AR rows

**Files:**
- Modify: `ucat/ui.py` (in `run` or the main App `__init__` startup path)

This is a one-line observability hook so the user can see, in the terminal, how many historical AR rows have been hidden by the swap. Spec section 4.

- [ ] **Step 12.1: Find the right startup callsite**

Run: `grep -n "def run\|def __init__.*Database\|self.db = Database\|api_status\|emit\|\"started\"\|trainer started" ucat/ui.py | head -20`
Expected: Locate the App `__init__` that creates `self.db`, or the `run()` function that constructs the App. The log line goes immediately after `self.db = Database(...)` so the count query has a connection to use.

- [ ] **Step 12.2: Add the log line**

In the App `__init__` (just after `self.db = Database(...)`), add:

```python
        try:
            ar_count = self.db.count("AR")
            if ar_count:
                logger.info("AR section retired; %d historical row(s) hidden from KB views", ar_count)
        except Exception:
            pass  # observability only — never block startup
```

If `logger` is not already imported at the top of `ui.py`, add `from .telemetry import logger` near the other `from .telemetry` imports.

- [ ] **Step 12.3: Smoke-test startup**

Run: `./venv/bin/python -c "from ucat.ui import run; print('OK')"`
Expected: `OK`. No import errors.

- [ ] **Step 12.4: Commit**

```bash
git add ucat/ui.py
git commit -m "ui: log count of hidden AR rows on startup"
```

---

## Task 13: Manual smoke test

This task does not change code. It verifies the swap end-to-end against the live UI.

- [ ] **Step 13.1: Launch the trainer**

Run: `./venv/bin/python ucat_trainer.py`
Expected: The app window opens. No tracebacks in the terminal. If the DB had any AR rows, the terminal shows the `AR section retired; N historical row(s) hidden from KB views` log line from Task 12.

- [ ] **Step 13.2: Verify the section radios**

Look at the Bulk tab section radios.
Expected: Four radios labelled VR, DM, QR, SJT. No AR radio.

- [ ] **Step 13.3: Verify the single-shot generation tab**

Switch to the Generate tab. The section dropdown / radios.
Expected: VR, DM, QR, SJT only.

- [ ] **Step 13.4: Seed and index the SJT sample**

Click `⊕ Add Sample Documents`, then `⊛ Index Knowledge Base`.
Expected: Status shows samples added (one per section, including SJT). Indexing completes without error.

- [ ] **Step 13.5: Generate one SJT set**

In the Generate tab pick `SJT`, leave hint empty, click Generate.
Expected: A 4-question SJT set appears. The Format panel shows a SCENARIO heading with the model's scenario, followed by 4 questions each annotated `(rating: appropriateness)` or `(rating: importance)`. The visuals panel shows `(no visuals for Situational Judgement)`.

- [ ] **Step 13.6: Verify KB browser hides AR**

If your DB has historical AR rows, open the KB tab. Switch the section filter to `ALL`.
Expected: Only VR/DM/QR/SJT rows appear. AR rows are not listed.
If your DB has no AR rows, this step is a visual-confirmation only — note that the filter still works correctly.

- [ ] **Step 13.7: Verify settings coercion**

Open `ucat_settings.json` (in the project root) in a text editor. Set `"bulk_section": "AR"` and save. Restart the app.
Expected: On the Bulk tab the section radio is on `VR` (coerced). Re-saving the settings (any change) rewrites the file with `bulk_section: "VR"`.
Cleanup: revert any test changes to `ucat_settings.json` before continuing.

- [ ] **Step 13.8: Final commit (if anything in the smoke test surfaced issues)**

If the smoke test reveals any issue (typo in prompt, layout glitch, missed callsite), fix it inline, commit with a clear message (e.g. `ui: fix lingering AR string in tooltip`), and re-run the relevant smoke step.

If nothing surfaced, no commit is needed.

---

## Task 14: Final verification

- [ ] **Step 14.1: Final test run across all suites**

Run: `./venv/bin/python tests/test_sjt.py && ./venv/bin/python tests/test_schema.py && ./venv/bin/python tests/test_bulk_cost.py`
Expected: All three suites print `All tests passed.`

- [ ] **Step 14.2: Grep for any remaining AR references that shouldn't be there**

Run:

```bash
grep -rn "\bAR\b\|Abstract Reasoning\|set_a_panels\|set_b_panels\|render_ar_" --include="*.py" ucat/
```

Expected output should ONLY contain references in `crawler_import.py` (which intentionally still skips AR captures with a warning), the AR-filter guards in `ui.py` (Task 11), and the AR-count log line in `ui.py` (Task 12). No stray references in `models.py`, `config.py`, `rag.py`, `samples.py`, `format.py`, `calibration.py`, `coverage.py`, `db.py`, `verification.py`, or `rendering.py`.

If any unexpected match appears, address it and commit with a clear message.

- [ ] **Step 14.3: Inspect the final diff range against `main`**

Run: `git log --oneline main..HEAD`
Expected: A clean stack of commits matching the task names above (one or two per task). Commit messages should each describe a single coherent change.
