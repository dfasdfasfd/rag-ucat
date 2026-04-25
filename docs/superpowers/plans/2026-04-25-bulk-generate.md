# Bulk Question Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Bulk tab to the UCAT Trainer that generates N questions of a chosen section + subtype, persists them with a batch_id, and supports manual HTML export.

**Architecture:** A new `_t_bulk` Tk tab spawns a worker thread that runs a sequential loop in `RAGEngine.generate_bulk()`. The loop dispatches per-section: VR/QR/AR call the existing `generate()` (set-shaped) with a subtype constraint injected into `_system_blocks`; DM uses a new `generate_single_dm_question()` that returns one question per LLM call. Items are saved to the existing `generated` table with a new `batch_id` column; batch metadata lives in a new `batches` table. A new `ucat/export.py` produces standalone HTML for any saved batch.

**Tech Stack:** Python 3.13, Tkinter, SQLite, Pydantic v2, anthropic SDK, pytest (newly introduced).

**Spec:** [docs/superpowers/specs/2026-04-25-bulk-generate-design.md](../specs/2026-04-25-bulk-generate-design.md)

---

## File Map

### New files
- `tests/__init__.py` — empty marker
- `tests/conftest.py` — pytest fixtures: in-memory `Database`, fake LLM responses
- `tests/test_db_batch.py` — batches table, batch_id round-trip, list filters
- `tests/test_rag_subtype.py` — `_system_blocks` subtype injection, `generate_single_dm_question`, `generate_bulk` orchestration
- `tests/test_export_html.py` — HTML export structure
- `ucat/export.py` — `export_batch_html(db, batch_id, dest_folder) -> Path`

### Modified files
- `ucat/config.py` — add `SUBTYPES` dict
- `ucat/models.py` — add `batch_id` to `Question`; add `Batch` and `DMSingleQuestion` schemas
- `ucat/db.py` — `_init` migration; `create_batch`, `update_batch`, `list_batches`, `questions_by_batch`; `add_generated(batch_id=...)`
- `ucat/rag.py` — `_system_blocks(subtype=...)`; `generate(subtype=...)`; `generate_single_dm_question(subtype, ...)`; `generate_bulk(...)`
- `ucat/ui.py` — add `_t_bulk` frame; `_tab_bulk()`; bulk handlers; "Filter by batch" dropdown on history tab
- `requirements.txt` — add `pytest>=8.0`

---

## Task 1: Bootstrap pytest infrastructure

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Add pytest to requirements**

Edit `requirements.txt`, append:

```
pytest>=8.0
```

- [ ] **Step 2: Install pytest**

Run: `pip install pytest>=8.0`
Expected: `Successfully installed pytest-8.x.x`

- [ ] **Step 3: Create tests package marker**

Create `tests/__init__.py` with empty content.

- [ ] **Step 4: Create conftest with in-memory DB fixture**

Create `tests/conftest.py`:

```python
"""Shared pytest fixtures."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import pytest

# Force in-memory DB before any ucat module imports config defaults.
os.environ["UCAT_DB_FILE"] = ":memory:"
os.environ["UCAT_SETTINGS_FILE"] = "/tmp/ucat_settings_test.json"
os.environ["UCAT_TELEMETRY_FILE"] = "/tmp/ucat_telemetry_test.jsonl"


@pytest.fixture
def db():
    """A fresh in-memory Database per test."""
    from ucat.db import Database
    d = Database(":memory:")
    yield d
    d.close()


@pytest.fixture
def settings(tmp_path):
    """A fresh Settings backed by a temp file."""
    from ucat.config import Settings
    return Settings(path=str(tmp_path / "settings.json"))


class FakeUsage:
    """Stub matching the dict shape returned by ucat.llm.extract_usage."""
    @staticmethod
    def make() -> Dict[str, Any]:
        return {
            "input_tokens": 100, "output_tokens": 200,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
            "cost_usd": 0.001, "model": "fake",
        }


@pytest.fixture
def fake_generate_structured(monkeypatch):
    """Replace ucat.llm.generate_structured with a callable the test programs.

    Usage:
        fake_generate_structured.queue.append(my_pydantic_instance)
        # ... code under test calls generate_structured(...) and gets that instance.

    The fixture exposes `.calls` (list of kwargs each call received) and
    `.queue` (FIFO of pydantic instances to return).
    """
    class Recorder:
        def __init__(self):
            self.calls: List[Dict[str, Any]] = []
            self.queue: List[Any] = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            if not self.queue:
                raise AssertionError("fake_generate_structured queue is empty")
            return self.queue.pop(0), FakeUsage.make()

    rec = Recorder()
    # Patch in both modules that import the symbol.
    monkeypatch.setattr("ucat.llm.generate_structured", rec)
    monkeypatch.setattr("ucat.rag.generate_structured", rec)
    return rec
```

- [ ] **Step 5: Write smoke test**

Create `tests/test_smoke.py`:

```python
def test_db_can_be_created(db):
    assert db.count() == 0


def test_settings_round_trip(settings):
    settings.set("llm", "claude-haiku-4-5")
    assert settings.get("llm") == "claude-haiku-4-5"
```

- [ ] **Step 6: Run smoke tests**

Run: `pytest tests/test_smoke.py -v`
Expected: 2 passed

- [ ] **Step 7: Commit**

```bash
git add requirements.txt tests/__init__.py tests/conftest.py tests/test_smoke.py
git commit -m "test: bootstrap pytest with in-memory DB and fake-LLM fixtures"
```

---

## Task 2: Add SUBTYPES config

**Files:**
- Modify: `ucat/config.py` (after line 79, after `SECTION_DESC`)
- Test: `tests/test_config_subtypes.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_config_subtypes.py`:

```python
from ucat.config import SUBTYPES, SECTIONS


def test_subtypes_defined_for_every_section():
    assert set(SUBTYPES.keys()) == set(SECTIONS.keys())


def test_dm_subtypes_match_models_type_field():
    # DM Question.type can be: syllogism, logical, venn, probability, argument
    assert SUBTYPES["DM"] == [
        "mixed", "syllogism", "logical", "venn", "probability", "argument",
    ]


def test_qr_subtypes_match_chart_types():
    # QRChartType: table, bar, line, stacked_bar, pie
    assert SUBTYPES["QR"] == [
        "mixed", "table", "bar", "line", "stacked_bar", "pie",
    ]


def test_vr_subtypes():
    assert SUBTYPES["VR"] == ["mixed", "tf", "mcq"]


def test_ar_has_only_standard():
    assert SUBTYPES["AR"] == ["standard"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_subtypes.py -v`
Expected: FAIL — `ImportError: cannot import name 'SUBTYPES' from 'ucat.config'`

- [ ] **Step 3: Add SUBTYPES to config**

Edit `ucat/config.py`. After the `SECTION_DESC` block (around line 79), insert:

```python
# Per-section subtype taxonomy used by the bulk-generate UI.
# "mixed" preserves the existing variety-focused prompt behaviour.
SUBTYPES: Dict[str, list[str]] = {
    "VR": ["mixed", "tf", "mcq"],
    "DM": ["mixed", "syllogism", "logical", "venn", "probability", "argument"],
    "QR": ["mixed", "table", "bar", "line", "stacked_bar", "pie"],
    "AR": ["standard"],
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config_subtypes.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add ucat/config.py tests/test_config_subtypes.py
git commit -m "feat(config): add SUBTYPES taxonomy for bulk generation"
```

---

## Task 3: Add `batch_id` to Question and Batch model

**Files:**
- Modify: `ucat/models.py` (line 39 area for Question; new Batch class at end)
- Test: `tests/test_models_batch.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_models_batch.py`:

```python
from ucat.models import Question, Batch, CoverageTags


def _q(**overrides):
    base = dict(
        number=1, text="What is 2+2?",
        options={"A": "3", "B": "4", "C": "5", "D": "6"},
        answer="B", explanation="Basic arithmetic.",
        difficulty=2.0,
        coverage=CoverageTags(topic="arithmetic", scenario_type="abstract"),
    )
    base.update(overrides)
    return Question(**base)


def test_question_batch_id_defaults_to_none():
    q = _q()
    assert q.batch_id is None


def test_question_accepts_batch_id():
    q = _q(batch_id="abc-123")
    assert q.batch_id == "abc-123"


def test_batch_minimal_fields():
    b = Batch(
        id="uuid-1", label="DM-syllogism-2026-04-25",
        section="DM", subtype="syllogism",
        requested=30, started_at="2026-04-25T12:00:00",
    )
    assert b.succeeded == 0
    assert b.failed == 0
    assert b.completed_at is None
    assert b.cancelled is False


def test_batch_full_fields():
    b = Batch(
        id="uuid-2", label="VR-tf-2026-04-25",
        section="VR", subtype="tf",
        requested=40, succeeded=38, failed=2,
        started_at="2026-04-25T12:00:00",
        completed_at="2026-04-25T12:30:00",
        cancelled=False,
    )
    assert b.succeeded == 38
    assert b.failed == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models_batch.py -v`
Expected: FAIL — `ImportError: cannot import name 'Batch' from 'ucat.models'`

- [ ] **Step 3: Add `batch_id` to Question**

Edit `ucat/models.py`. Find the `Question` class (around line 39) and add the `batch_id` field after `coverage`:

```python
class Question(BaseModel):
    number: int
    text: str
    type: Optional[str] = None  # DM subtype: syllogism / logical / probability / argument / venn
    options: Dict[str, str] = Field(min_length=2,
                                       description="Label → option text. Must be populated; empty dicts are invalid.")
    answer: str
    explanation: str
    difficulty: float = Field(ge=1.0, le=5.0,
                              description="IRT logits on the 1.0-5.0 scale.")
    coverage: CoverageTags
    batch_id: Optional[str] = None  # Set when the question came from a bulk run.
```

- [ ] **Step 4: Add Batch model**

At the bottom of `ucat/models.py`, after the `JuryVerdict` class, append:

```python
# ─── Bulk generation ──────────────────────────────────────────────────────────

class Batch(BaseModel):
    """Metadata for one bulk-generation run."""
    id: str
    label: str
    section: Literal["VR", "DM", "QR", "AR"]
    subtype: Optional[str] = None  # null for "mixed"
    requested: int = Field(ge=1)
    succeeded: int = 0
    failed: int = 0
    started_at: str  # ISO 8601
    completed_at: Optional[str] = None
    cancelled: bool = False
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_models_batch.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add ucat/models.py tests/test_models_batch.py
git commit -m "feat(models): add Batch schema and batch_id field on Question"
```

---

## Task 4: DB schema migration — `batches` table + `batch_id` column

**Files:**
- Modify: `ucat/db.py` (`_init` method, around line 67-97)
- Test: `tests/test_db_batch.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_db_batch.py`:

```python
import sqlite3
from ucat.db import Database


def test_init_creates_batches_table():
    db = Database(":memory:")
    rows = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='batches'"
    ).fetchall()
    assert rows == [("batches",)]


def test_init_adds_batch_id_column_to_generated():
    db = Database(":memory:")
    cols = [r[1] for r in db.conn.execute("PRAGMA table_info(generated)").fetchall()]
    assert "batch_id" in cols


def test_init_is_idempotent_on_existing_generated_table():
    # Simulate an old DB that has `generated` but no `batch_id`.
    conn = sqlite3.connect(":memory:")
    conn.execute("""CREATE TABLE generated (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        section TEXT NOT NULL,
        data TEXT NOT NULL,
        context_ids TEXT NOT NULL
    )""")
    conn.commit()
    conn.close()
    # Running Database init on the same path must not raise.
    db = Database(":memory:")
    db._init()  # second call must also be safe
    cols = [r[1] for r in db.conn.execute("PRAGMA table_info(generated)").fetchall()]
    assert "batch_id" in cols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db_batch.py -v`
Expected: FAIL — `assert rows == [("batches",)]` fails (table not created)

- [ ] **Step 3: Extend `_init` migration**

Edit `ucat/db.py`. In the `_init` method (around line 67), after the existing `ALTER TABLE generated` migration loop and before `self.conn.commit()`, add:

```python
        # Migration: batch_id on generated.
        try:
            c.execute("ALTER TABLE generated ADD COLUMN batch_id TEXT DEFAULT NULL")
        except sqlite3.OperationalError:
            pass

        # Bulk-generation batches.
        c.execute("""CREATE TABLE IF NOT EXISTS batches (
            id           TEXT PRIMARY KEY,
            label        TEXT NOT NULL,
            section      TEXT NOT NULL,
            subtype      TEXT,
            requested    INTEGER NOT NULL,
            succeeded    INTEGER NOT NULL DEFAULT 0,
            failed       INTEGER NOT NULL DEFAULT 0,
            started_at   TEXT NOT NULL,
            completed_at TEXT,
            cancelled    INTEGER NOT NULL DEFAULT 0
        )""")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_db_batch.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ucat/db.py tests/test_db_batch.py
git commit -m "feat(db): add batches table and batch_id column with idempotent migration"
```

---

## Task 5: DB batch CRUD methods

**Files:**
- Modify: `ucat/db.py` (append methods to `Database` class; modify `add_generated` signature)
- Modify: `tests/test_db_batch.py` (append new tests)

- [ ] **Step 1: Append failing tests**

Append to `tests/test_db_batch.py`:

```python
import json
from ucat.db import Database


def test_create_batch_round_trip():
    db = Database(":memory:")
    db.create_batch(
        batch_id="uuid-1", label="DM-syllogism-2026-04-25",
        section="DM", subtype="syllogism", requested=30,
        started_at="2026-04-25T12:00:00",
    )
    batches = db.list_batches()
    assert len(batches) == 1
    b = batches[0]
    assert b["id"] == "uuid-1"
    assert b["label"] == "DM-syllogism-2026-04-25"
    assert b["section"] == "DM"
    assert b["subtype"] == "syllogism"
    assert b["requested"] == 30
    assert b["succeeded"] == 0
    assert b["failed"] == 0
    assert b["completed_at"] is None
    assert b["cancelled"] is False


def test_update_batch_counts_and_completion():
    db = Database(":memory:")
    db.create_batch(batch_id="u1", label="L", section="DM", subtype="syllogism",
                    requested=10, started_at="2026-04-25T12:00:00")
    db.update_batch("u1", succeeded=8, failed=2,
                    completed_at="2026-04-25T12:30:00", cancelled=False)
    b = db.list_batches()[0]
    assert b["succeeded"] == 8
    assert b["failed"] == 2
    assert b["completed_at"] == "2026-04-25T12:30:00"


def test_list_batches_reverse_chronological():
    db = Database(":memory:")
    db.create_batch(batch_id="old", label="L1", section="DM", subtype=None,
                    requested=5, started_at="2026-04-25T10:00:00")
    db.create_batch(batch_id="new", label="L2", section="VR", subtype="tf",
                    requested=8, started_at="2026-04-25T14:00:00")
    ids = [b["id"] for b in db.list_batches()]
    assert ids == ["new", "old"]


def test_add_generated_with_batch_id_round_trip():
    db = Database(":memory:")
    db.create_batch(batch_id="u1", label="L", section="DM", subtype="syllogism",
                    requested=2, started_at="2026-04-25T12:00:00")
    data1 = {"section": "DM", "questions": [{"number": 1, "text": "Q1"}]}
    data2 = {"section": "DM", "questions": [{"number": 1, "text": "Q2"}]}
    id1 = db.add_generated("DM", data1, [], batch_id="u1")
    id2 = db.add_generated("DM", data2, [], batch_id="u1")
    # Single-Generate items still work without batch_id.
    db.add_generated("DM", {"section": "DM", "questions": []}, [])

    items = db.questions_by_batch("u1")
    assert {it["id"] for it in items} == {id1, id2}
    assert all(it["batch_id"] == "u1" for it in items)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_db_batch.py -v -k "create_batch or update_batch or list_batches or batch_id_round_trip"`
Expected: FAIL — `AttributeError: 'Database' object has no attribute 'create_batch'`

- [ ] **Step 3: Modify `add_generated` to accept `batch_id`**

Edit `ucat/db.py`. Replace the `add_generated` method (around line 188) with:

```python
    def add_generated(
        self,
        section: str,
        data: dict,
        ctx_ids: list,
        *,
        usage: Optional[dict] = None,
        verdict: Optional[dict] = None,
        coverage: Optional[dict] = None,
        difficulty: Optional[float] = None,
        batch_id: Optional[str] = None,
    ) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO generated (section,data,context_ids,usage,verdict,coverage,difficulty,batch_id)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (
                section, json.dumps(data), json.dumps(ctx_ids),
                json.dumps(usage)    if usage    else None,
                json.dumps(verdict)  if verdict  else None,
                json.dumps(coverage) if coverage else None,
                difficulty,
                batch_id,
            ),
        )
        self.conn.commit()
        return c.lastrowid
```

- [ ] **Step 4: Add batch CRUD methods**

Edit `ucat/db.py`. Append these methods to the `Database` class, just before `def close(self):` (around line 272):

```python
    # ── Batches ──

    def create_batch(self, *, batch_id: str, label: str, section: str,
                      subtype: Optional[str], requested: int,
                      started_at: str) -> None:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO batches (id,label,section,subtype,requested,started_at)"
            " VALUES (?,?,?,?,?,?)",
            (batch_id, label, section, subtype, requested, started_at),
        )
        self.conn.commit()

    def update_batch(self, batch_id: str, *,
                      succeeded: Optional[int] = None,
                      failed: Optional[int] = None,
                      completed_at: Optional[str] = None,
                      cancelled: Optional[bool] = None) -> None:
        sets, args = [], []
        if succeeded is not None:    sets.append("succeeded=?");    args.append(succeeded)
        if failed is not None:       sets.append("failed=?");       args.append(failed)
        if completed_at is not None: sets.append("completed_at=?"); args.append(completed_at)
        if cancelled is not None:    sets.append("cancelled=?");    args.append(1 if cancelled else 0)
        if not sets:
            return
        args.append(batch_id)
        c = self.conn.cursor()
        c.execute(f"UPDATE batches SET {', '.join(sets)} WHERE id=?", args)
        self.conn.commit()

    def list_batches(self, limit: int = 200) -> List[Dict[str, Any]]:
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT id,label,section,subtype,requested,succeeded,failed,"
            "started_at,completed_at,cancelled"
            " FROM batches ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{
            "id": r[0], "label": r[1], "section": r[2], "subtype": r[3],
            "requested": r[4], "succeeded": r[5], "failed": r[6],
            "started_at": r[7], "completed_at": r[8],
            "cancelled": bool(r[9]),
        } for r in rows]

    def questions_by_batch(self, batch_id: str) -> List[Dict[str, Any]]:
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT id,section,data,context_ids,usage,verdict,coverage,difficulty,created,batch_id"
            " FROM generated WHERE batch_id=? ORDER BY id ASC",
            (batch_id,),
        ).fetchall()
        return [{
            "id": r[0], "section": r[1], "data": json.loads(r[2]),
            "context_ids": json.loads(r[3]),
            "usage":    json.loads(r[4]) if r[4] else None,
            "verdict":  json.loads(r[5]) if r[5] else None,
            "coverage": json.loads(r[6]) if r[6] else None,
            "difficulty": r[7], "created": r[8], "batch_id": r[9],
        } for r in rows]
```

- [ ] **Step 5: Run all DB tests**

Run: `pytest tests/test_db_batch.py -v`
Expected: 7 passed (3 from Task 4 + 4 new)

- [ ] **Step 6: Commit**

```bash
git add ucat/db.py tests/test_db_batch.py
git commit -m "feat(db): add batch CRUD methods and batch_id support on add_generated"
```

---

## Task 6: `_system_blocks` accepts subtype constraint

**Files:**
- Modify: `ucat/rag.py` (`_system_blocks`, around line 87-160; `generate`, around line 162-185)
- Test: `tests/test_rag_subtype.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_rag_subtype.py`:

```python
from ucat.config import Settings
from ucat.db import Database
from ucat.rag import RAGEngine


def _engine():
    return RAGEngine(Database(":memory:"), Settings(path="/tmp/_test_settings.json"))


def _block_text(blocks):
    return "\n\n".join(b["text"] for b in blocks)


def test_no_subtype_unchanged_dm_variety_clause():
    eng = _engine()
    blocks = eng._system_blocks("DM", retrieved=[], target_difficulty=3.0,
                                  subtype=None)
    text = _block_text(blocks)
    assert "include syllogism, logical, venn, probability, AND" in text


def test_dm_subtype_locks_to_single_type():
    eng = _engine()
    blocks = eng._system_blocks("DM", retrieved=[], target_difficulty=3.0,
                                  subtype="syllogism")
    text = _block_text(blocks)
    assert "ALL 5 questions must be of subtype `syllogism`" in text
    # The mixed-variety clause must NOT appear.
    assert "one of each is ideal" not in text


def test_vr_tf_subtype_constrains_question_format():
    eng = _engine()
    blocks = eng._system_blocks("VR", retrieved=[], target_difficulty=3.0,
                                  subtype="tf")
    text = _block_text(blocks)
    assert "True/False/Can't Tell" in text
    assert "ALL 4 questions" in text


def test_vr_mcq_subtype_constrains_question_format():
    eng = _engine()
    blocks = eng._system_blocks("VR", retrieved=[], target_difficulty=3.0,
                                  subtype="mcq")
    text = _block_text(blocks)
    assert "4-option multiple choice" in text
    assert "ALL 4 questions" in text


def test_qr_subtype_pins_chart_type():
    eng = _engine()
    blocks = eng._system_blocks("QR", retrieved=[], target_difficulty=3.0,
                                  subtype="bar")
    text = _block_text(blocks)
    assert "stimulus.type` MUST be `bar`" in text


def test_mixed_subtype_treated_as_no_constraint():
    eng = _engine()
    blocks = eng._system_blocks("DM", retrieved=[], target_difficulty=3.0,
                                  subtype="mixed")
    text = _block_text(blocks)
    # "mixed" is the existing variety-mode behaviour, no lock clause.
    assert "ALL 5 questions must be of subtype" not in text


def test_ar_subtype_ignored():
    eng = _engine()
    blocks = eng._system_blocks("AR", retrieved=[], target_difficulty=3.0,
                                  subtype="standard")
    text = _block_text(blocks)
    # Just verify it doesn't crash and returns the AR baseline.
    assert "AR panels are STRUCTURED shape lists" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag_subtype.py -v`
Expected: FAIL — `TypeError: _system_blocks() got an unexpected keyword argument 'subtype'`

- [ ] **Step 3: Modify `_system_blocks` signature and add subtype branch**

Edit `ucat/rag.py`. Update the `_system_blocks` method signature and body:

```python
    def _system_blocks(self, section: str, retrieved: list,
                        target_difficulty: float,
                        subtype: Optional[str] = None) -> List[Dict[str, Any]]:
        """Build cache-friendly system blocks.

          [0] Frozen role + structural description + retrieved KB     (CACHED)
          [1] Per-request difficulty + variation guidance              (NOT CACHED)

        If `subtype` is non-null and not "mixed", a constraint is added that
        locks the whole set to one subtype/format/chart-type.
        """
        role = (
            f"You are an expert UCAT {SECTIONS[section]} question writer.\n\n"
            f"You generate: {SECTION_DESC[section]}\n\n"
            "OUTPUT REQUIREMENTS:\n"
            "• Return ONLY a JSON object matching the provided schema. No prose, no markdown fences.\n"
            "• Every question's answer must be derivable from the passage / stimulus / rules.\n"
            "• Distractors must be plausible — each wrong option should reflect a realistic mistake.\n"
            "• `options` MUST be a non-empty dict mapping option labels to option text:\n"
            "    - VR / QR multiple choice → keys A, B, C, D (4 options).\n"
            "    - VR True/False/Can't Tell items → keys exactly \"True\", \"False\", \"Can't Tell\".\n"
            "    - DM → keys A, B, C, D, E (5 options).\n"
            "    - AR test items → keys exactly \"Set A\", \"Set B\", \"Neither\".\n"
            "  Never leave `options` empty. Never reference option text only inside `explanation`.\n"
            "• Explanations must show the reasoning a student needs to learn from the question.\n"
            "• Each question must have a `difficulty` field (1.0-5.0 IRT logits) and `coverage` "
            "metadata identifying its topic and scenario type.\n"
        )
        if section == "QR":
            role += (
                "\nCRITICAL — QR stimulus is a STRUCTURED chart spec, not text. "
                "Choose `type` from {bar, line, stacked_bar, pie, table}. "
                "Populate `categories` and `series[]` for bar/line/stacked_bar/pie. "
                "Use `rows` for tables. Include realistic units (e.g. £000s, %, kg). "
                "Make data internally consistent with the questions you write.\n"
            )
        if section == "AR":
            role += (
                "\nCRITICAL — AR panels are STRUCTURED shape lists. "
                "Each panel contains `shapes[]` with `kind`, `color`, `size`, and "
                "`rotation_deg`. Hidden rules can use any combination of shape kind, "
                "color, size, count, rotation, or position. Provide BOTH a structured "
                "spec for rendering AND a clear English description in the rule field.\n"
            )

        # DM variety clause is suppressed when a single-subtype lock is requested.
        active_subtype = subtype if subtype and subtype != "mixed" else None
        if section == "DM" and active_subtype is None:
            role += (
                "\nFor venn-type DM questions, include a structured `venn` field with "
                "2 or 3 sets so the diagram can be rendered. For other DM subtypes, "
                "leave `venn` null.\n"
                "Aim for variety: include syllogism, logical, venn, probability, AND "
                "argument subtypes across the 5 questions — one of each is ideal.\n"
            )
        elif section == "DM" and active_subtype is not None:
            role += (
                f"\nSUBTYPE LOCK: ALL 5 questions must be of subtype `{active_subtype}`. "
                f"Set `type` = \"{active_subtype}\" on every question. "
                "Do NOT mix subtypes in this set.\n"
            )
            if active_subtype == "venn":
                role += (
                    "Every question must include a structured `venn` field with 2 or 3 sets.\n"
                )
            else:
                role += "Leave `venn` null on every question.\n"

        if section == "VR" and active_subtype is not None:
            if active_subtype == "tf":
                role += (
                    "\nQUESTION FORMAT LOCK: ALL 4 questions must be True/False/Can't Tell items. "
                    "Each question's `options` must use exactly the keys \"True\", \"False\", "
                    "\"Can't Tell\". Do not write 4-option MCQs in this set.\n"
                )
            elif active_subtype == "mcq":
                role += (
                    "\nQUESTION FORMAT LOCK: ALL 4 questions must be 4-option multiple choice. "
                    "Each question's `options` must use exactly the keys A, B, C, D. "
                    "Do not write True/False/Can't Tell items in this set.\n"
                )

        if section == "QR" and active_subtype is not None:
            role += (
                f"\nCHART TYPE LOCK: `stimulus.type` MUST be `{active_subtype}`. "
                "Do not vary chart type.\n"
            )

        ex_text = ""
        if retrieved:
            blocks = [
                f"--- KNOWLEDGE BASE EXAMPLE {i+1} (similarity {sc:.3f}) ---\n"
                f"{json.dumps(d['data'], indent=2)}"
                for i, (sc, d) in enumerate(retrieved)
            ]
            ex_text = (
                "\n\nThe documents below are gold-standard examples from the user's "
                "knowledge base. They define authoritative format, voice, and topical "
                "range. Your output must mirror their JSON structure exactly.\n\n"
                + "\n\n".join(blocks)
            )

        diff_label = difficulty_label(target_difficulty)
        diff = (
            f"\n\nTARGET DIFFICULTY: {target_difficulty:.1f} logits — {diff_label}\n"
            "Aim for an average set difficulty within ±0.4 of the target. "
            "Include some easier and some harder items around that mean.\n"
        )

        return [
            {"type": "text", "text": role + ex_text, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": diff},
        ]
```

- [ ] **Step 4: Update `generate` to forward subtype**

In the same file, modify the `generate` method signature and call (around line 162):

```python
    def generate(
        self,
        section: str,
        hint: str = "",
        *,
        on_progress: Optional[Callable[[str], None]] = None,
        on_delta: Optional[Callable[[str], None]] = None,
        variation_seed: Optional[str] = None,
        subtype: Optional[str] = None,
        batch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
```

And inside `generate`, replace the `_system_blocks` call (around line 185) with:

```python
            system_blocks = self._system_blocks(section, retrieved, target,
                                                  subtype=subtype)
```

Also, replace the `self.db.add_generated(...)` call (around line 297) with:

```python
            self.db.add_generated(section, data, ctx_ids,
                                   usage=total_usage,
                                   verdict=verdict_dict,
                                   coverage=coverage_dict,
                                   difficulty=set_difficulty,
                                   batch_id=batch_id)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_rag_subtype.py -v`
Expected: 7 passed

- [ ] **Step 6: Run all tests so nothing regressed**

Run: `pytest tests/ -v`
Expected: all green

- [ ] **Step 7: Commit**

```bash
git add ucat/rag.py tests/test_rag_subtype.py
git commit -m "feat(rag): _system_blocks accepts subtype lock and generate forwards batch_id"
```

---

## Task 7: `generate_single_dm_question` method

**Files:**
- Modify: `ucat/models.py` (add `DMSingleQuestionResult` schema near `DMSet`)
- Modify: `ucat/rag.py` (append `generate_single_dm_question` method)
- Modify: `tests/test_rag_subtype.py` (append tests)

- [ ] **Step 1: Append failing tests**

Append to `tests/test_rag_subtype.py`:

```python
from ucat.models import DMQuestion, DMSingleQuestionResult, CoverageTags


def _dm_q(subtype="syllogism"):
    return DMQuestion(
        number=1, type=subtype,
        text="If all A are B, and all B are C, then…",
        options={"A": "All A are C", "B": "Some A are not C", "C": "No A are C",
                 "D": "Cannot tell", "E": "All C are A"},
        answer="A", explanation="Transitive syllogism.",
        difficulty=2.5,
        coverage=CoverageTags(topic="logic", scenario_type="abstract"),
    )


def test_generate_single_dm_question_returns_one_dm_question(fake_generate_structured):
    fake_generate_structured.queue.append(
        DMSingleQuestionResult(question=_dm_q("syllogism"))
    )
    eng = _engine()
    result = eng.generate_single_dm_question(subtype="syllogism")
    assert result["data"]["section"] == "DM"
    assert len(result["data"]["questions"]) == 1
    assert result["data"]["questions"][0]["type"] == "syllogism"


def test_generate_single_dm_question_uses_subtype_locked_prompt(fake_generate_structured):
    fake_generate_structured.queue.append(
        DMSingleQuestionResult(question=_dm_q("probability"))
    )
    eng = _engine()
    eng.generate_single_dm_question(subtype="probability")
    call = fake_generate_structured.calls[0]
    sys_text = "\n\n".join(b["text"] for b in call["system_blocks"])
    assert "Generate ONE single DM question of subtype `probability`" in sys_text


def test_generate_single_dm_question_persists_with_batch_id(fake_generate_structured):
    fake_generate_structured.queue.append(
        DMSingleQuestionResult(question=_dm_q("syllogism"))
    )
    eng = _engine()
    eng.db.create_batch(batch_id="b1", label="L", section="DM",
                        subtype="syllogism", requested=1,
                        started_at="2026-04-25T12:00:00")
    eng.generate_single_dm_question(subtype="syllogism", batch_id="b1")
    items = eng.db.questions_by_batch("b1")
    assert len(items) == 1
    assert items[0]["section"] == "DM"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag_subtype.py -k "single_dm" -v`
Expected: FAIL — `ImportError: cannot import name 'DMSingleQuestionResult'`

- [ ] **Step 3: Add `DMSingleQuestionResult` schema**

Edit `ucat/models.py`. After the `DMSet` class, add:

```python
class DMSingleQuestionResult(BaseModel):
    """Wrapper schema for the single-question DM bulk path.

    Distinct from DMSet (5 questions). The bulk-DM loop calls Claude with
    this schema so the model produces exactly one question per call.
    """
    section: Literal["DM"] = "DM"
    question: DMQuestion
```

- [ ] **Step 4: Append `generate_single_dm_question` to `RAGEngine`**

Edit `ucat/rag.py`. Add this import at the top with the other model imports (around line 15):

```python
from .models import SECTION_MODELS, DMSingleQuestionResult
```

Then append this method to the `RAGEngine` class, after `generate` (around line 312):

```python
    def generate_single_dm_question(
        self,
        subtype: str,
        *,
        hint: str = "",
        on_progress: Optional[Callable[[str], None]] = None,
        batch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate ONE standalone DM question of a fixed subtype.

        Used by the bulk path when the user wants exact-count DM batches.
        Returns the same dict shape as `generate(...)` for UI symmetry, with
        `data.questions` containing exactly one DMQuestion.
        """
        target = self.target_difficulty

        with trace("rag_generate_single_dm", subtype=subtype, hint=hint[:80]) as t:
            if on_progress: on_progress("Embedding retrieval query…")
            qvec, retrieved = self.retrieve("DM", hint or subtype)

            # Build a focused system block: cached prefix is the standard DM role
            # but we replace the variety clause with a single-subtype lock.
            base = self._system_blocks("DM", retrieved, target, subtype=subtype)
            extra = (
                f"\n\nSINGLE-QUESTION MODE: Generate ONE single DM question of "
                f"subtype `{subtype}`. The output schema has a `question` field "
                f"(not `questions`). Number it 1.\n"
            )
            system_blocks = [
                base[0],
                {"type": "text", "text": base[1]["text"] + extra},
            ]

            user = (
                f"Generate ONE NEW UCAT Decision Making question of subtype "
                f"`{subtype}`. Original content; do not reuse fact-patterns from "
                "the example documents. Return ONLY the JSON object."
            )

            if on_progress: on_progress(f"Generating with {self.llm}…")
            with trace("generate_single", model=self.llm) as gt:
                parsed, gen_usage = generate_structured(
                    system_blocks=system_blocks, user=user,
                    model=self.llm, output_schema=DMSingleQuestionResult,
                    max_tokens=2000,
                )
                gt.update(gen_usage)

            q_dump = parsed.question.model_dump()
            q_dump["number"] = 1
            data = {"section": "DM", "questions": [q_dump]}

            # Verification — single judge per question (jury skipped for cost).
            verdict_dict: Optional[Dict[str, Any]] = None
            verify_usages: List[Dict[str, Any]] = []
            if self.verify_enabled:
                if on_progress: on_progress("Verifying answer…")
                try:
                    v, vu = llm_judge("DM", data, DEFAULT_VERIFY_LLM)
                    verdict_dict = {"mode": "single",
                                     "judge": DEFAULT_VERIFY_LLM,
                                     **v.model_dump()}
                    verify_usages = [vu]
                except Exception as e:
                    verdict_dict = {"mode": "single", "error": str(e),
                                      "overall_correct": True, "confidence": "low"}

            cal = calibrate_set(data["questions"], "DM", judge_predictions={})
            data["calibrated_difficulty"] = cal
            set_difficulty = cal.get("set_difficulty", target)

            coverage_dict = aggregate_set(data)
            total_usage = merge_usage(gen_usage, *verify_usages)

            ctx_ids = [d["id"] for _, d in retrieved]
            self.db.add_generated("DM", data, ctx_ids,
                                   usage=total_usage,
                                   verdict=verdict_dict,
                                   coverage=coverage_dict,
                                   difficulty=set_difficulty,
                                   batch_id=batch_id)

            return {
                "data": data,
                "retrieved": retrieved,
                "usage": total_usage,
                "verdict": verdict_dict,
                "coverage": coverage_dict,
                "difficulty": cal,
                "dup_warning": None,
            }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_rag_subtype.py -v`
Expected: 10 passed (7 from Task 6 + 3 new)

- [ ] **Step 6: Commit**

```bash
git add ucat/models.py ucat/rag.py tests/test_rag_subtype.py
git commit -m "feat(rag): generate_single_dm_question for exact-count DM batches"
```

---

## Task 8: `generate_bulk` orchestration

**Files:**
- Modify: `ucat/rag.py` (append method)
- Modify: `tests/test_rag_subtype.py` (append tests)

- [ ] **Step 1: Append failing tests**

Append to `tests/test_rag_subtype.py`:

```python
import math
import threading
from ucat.models import DMSet, VRSet, ARSet, ARShape, ARPanel, QRSet, QRChart


def _dm_set(subtype="logical"):
    return DMSet(section="DM", questions=[_dm_q(subtype) for _ in range(5)])


def _vr_set():
    base_q = {
        "number": 1, "text": "Per the passage…",
        "options": {"True": "T", "False": "F", "Can't Tell": "?"},
        "answer": "True", "explanation": "Stated.",
        "difficulty": 2.5,
        "coverage": {"topic": "ecology", "scenario_type": "scientific"},
    }
    return VRSet(section="VR", passage="A passage about ecology.",
                  questions=[{**base_q, "number": i+1} for i in range(4)])


def _qr_set():
    chart = QRChart(type="bar", title="Sales", x_label="Q", y_label="£",
                    categories=["Q1","Q2","Q3","Q4"],
                    series=[{"name":"A","values":[1,2,3,4]}])
    base_q = {
        "number": 1, "text": "Compute…",
        "options": {"A":"1","B":"2","C":"3","D":"4","E":"5"},
        "answer": "B", "explanation": "1+1=2.",
        "difficulty": 3.0,
        "coverage": {"topic":"finance","scenario_type":"business"},
    }
    return QRSet(section="QR", stimulus=chart,
                  questions=[{**base_q, "number": i+1} for i in range(4)])


def test_generate_bulk_dm_calls_single_question_per_item(fake_generate_structured):
    for _ in range(3):
        fake_generate_structured.queue.append(
            DMSingleQuestionResult(question=_dm_q("syllogism"))
        )
    eng = _engine()
    progress: list[tuple] = []
    result = eng.generate_bulk(
        section="DM", subtype="syllogism", quantity=3,
        on_progress=lambda i, total, status, msg: progress.append((i, total, status)),
    )
    assert result["batch_id"]
    assert result["succeeded"] == 3
    assert result["failed"] == 0
    # 3 LLM calls total — exact-count.
    assert len(fake_generate_structured.calls) == 3


def test_generate_bulk_vr_uses_set_path_with_ceil(fake_generate_structured):
    # Quantity 10 → ceil(10/4) = 3 sets.
    for _ in range(3):
        fake_generate_structured.queue.append(_vr_set())
    eng = _engine()
    eng.settings.set("verify", False)
    result = eng.generate_bulk(section="VR", subtype="tf", quantity=10)
    assert result["succeeded"] == 3
    assert len(fake_generate_structured.calls) == 3


def test_generate_bulk_cancel_stops_loop(fake_generate_structured):
    fake_generate_structured.queue.append(
        DMSingleQuestionResult(question=_dm_q("logical"))
    )
    cancel = threading.Event()
    eng = _engine()

    def on_progress(i, total, status, msg):
        if i == 1:
            cancel.set()  # request cancel after item 1

    result = eng.generate_bulk(
        section="DM", subtype="logical", quantity=5,
        cancel_event=cancel, on_progress=on_progress,
    )
    assert result["succeeded"] == 1
    assert result["cancelled"] is True
    assert len(fake_generate_structured.calls) == 1


def test_generate_bulk_per_item_failure_is_logged_and_retried(monkeypatch):
    """First attempt raises; retry succeeds → item counts as success."""
    from tests.conftest import FakeUsage  # the helper from Task 1's conftest

    attempts = {"n": 0}
    success = DMSingleQuestionResult(question=_dm_q("logical"))

    def gs(**kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient API error")
        return success, FakeUsage.make()

    monkeypatch.setattr("ucat.rag.generate_structured", gs)
    eng = _engine()
    eng.settings.set("verify", False)  # skip llm_judge in the retry path
    result = eng.generate_bulk(section="DM", subtype="logical", quantity=1)
    assert result["succeeded"] == 1
    assert result["failed"] == 0
    assert attempts["n"] == 2  # one fail + one retry success
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag_subtype.py -k "generate_bulk" -v`
Expected: FAIL — `AttributeError: 'RAGEngine' object has no attribute 'generate_bulk'`

- [ ] **Step 3: Add `generate_bulk` to `RAGEngine`**

Edit `ucat/rag.py`. Add these imports at the top:

```python
import math
import threading
import uuid
from datetime import datetime
```

Append to the `RAGEngine` class:

```python
    def generate_bulk(
        self,
        *,
        section: str,
        subtype: Optional[str],
        quantity: int,
        label: Optional[str] = None,
        on_progress: Optional[Callable[[int, int, str, str], None]] = None,
        cancel_event: Optional["threading.Event"] = None,
    ) -> Dict[str, Any]:
        """Run a sequential bulk-generation loop.

        Returns a dict: batch_id, label, succeeded, failed, cancelled, total_items.

        Per section, one item is:
          - VR / QR: one passage/chart set (4 questions inside)
          - AR: one full set (12 panels + 5 test questions)
          - DM: one standalone DMQuestion

        For VR/QR/AR, quantity is interpreted as "number of questions" and
        rounded up (`ceil(quantity / per_set)`) to whole sets.

        `on_progress(item_idx, total_items, status, message)` is called as
        each item starts/finishes/fails. status ∈ {"started","succeeded","failed","retrying"}.
        """
        if section not in SECTIONS:
            raise ValueError(f"Unknown section: {section}")

        per_set = {"VR": 4, "QR": 4, "AR": 5, "DM": 1}[section]
        total_items = quantity if section == "DM" else math.ceil(quantity / per_set)

        batch_id = str(uuid.uuid4())
        if label is None:
            sub = subtype or "mixed"
            label = f"{section}-{sub}-{datetime.now().strftime('%Y-%m-%d')}"

        started_at = datetime.now().isoformat(timespec="seconds")
        self.db.create_batch(
            batch_id=batch_id, label=label,
            section=section, subtype=(subtype if subtype != "mixed" else None),
            requested=total_items, started_at=started_at,
        )

        succeeded = 0
        failed = 0
        cancelled = False

        for i in range(1, total_items + 1):
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break

            if on_progress:
                on_progress(i, total_items, "started", f"item {i}/{total_items}")

            attempt_count = 0
            last_error: Optional[Exception] = None
            while attempt_count < 2:  # initial + 1 retry
                attempt_count += 1
                try:
                    if section == "DM" and subtype and subtype != "mixed":
                        self.generate_single_dm_question(
                            subtype=subtype, batch_id=batch_id,
                        )
                    else:
                        self.generate(
                            section, hint="",
                            subtype=subtype, batch_id=batch_id,
                        )
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    if attempt_count < 2 and on_progress:
                        on_progress(i, total_items, "retrying", f"{type(e).__name__}: {e}")
                    continue

            if last_error is None:
                succeeded += 1
                self.db.update_batch(batch_id, succeeded=succeeded, failed=failed)
                if on_progress:
                    on_progress(i, total_items, "succeeded", f"item {i} ✓")
            else:
                failed += 1
                self.db.update_batch(batch_id, succeeded=succeeded, failed=failed)
                if on_progress:
                    on_progress(i, total_items, "failed",
                                  f"item {i} ✗ {type(last_error).__name__}: {last_error}")

        completed_at = datetime.now().isoformat(timespec="seconds")
        self.db.update_batch(batch_id,
                              succeeded=succeeded, failed=failed,
                              completed_at=completed_at, cancelled=cancelled)

        return {
            "batch_id": batch_id, "label": label,
            "succeeded": succeeded, "failed": failed,
            "cancelled": cancelled, "total_items": total_items,
        }
```

- [ ] **Step 4: Run all rag tests**

Run: `pytest tests/test_rag_subtype.py -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add ucat/rag.py tests/test_rag_subtype.py
git commit -m "feat(rag): generate_bulk sequential loop with cancel and one retry per item"
```

---

## Task 9: HTML export

**Files:**
- Create: `ucat/export.py`
- Create: `tests/test_export_html.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_export_html.py`:

```python
from pathlib import Path

from ucat.db import Database
from ucat.export import export_batch_html


def _stub_dm_data(text="Q1?"):
    return {
        "section": "DM",
        "questions": [{
            "number": 1, "type": "syllogism", "text": text,
            "options": {"A":"a","B":"b","C":"c","D":"d","E":"e"},
            "answer": "A", "explanation": "Because.",
            "difficulty": 2.5,
            "coverage": {"topic":"logic","scenario_type":"abstract"},
        }],
    }


def test_export_writes_index_and_per_item_files(tmp_path):
    db = Database(":memory:")
    db.create_batch(batch_id="b1", label="DM-syllogism-2026-04-25",
                    section="DM", subtype="syllogism", requested=2,
                    started_at="2026-04-25T12:00:00")
    db.add_generated("DM", _stub_dm_data("Q1?"), [], batch_id="b1")
    db.add_generated("DM", _stub_dm_data("Q2?"), [], batch_id="b1")

    out = export_batch_html(db, batch_id="b1", dest_folder=tmp_path)
    assert out.is_dir()
    assert (out / "index.html").exists()
    items = sorted(out.glob("q*.html"))
    assert len(items) == 2

    idx = (out / "index.html").read_text(encoding="utf-8")
    assert "DM-syllogism-2026-04-25" in idx
    assert "q1.html" in idx and "q2.html" in idx

    body1 = items[0].read_text(encoding="utf-8")
    assert "Q1?" in body1
    assert "Because." in body1


def test_export_empty_batch_creates_empty_index(tmp_path):
    db = Database(":memory:")
    db.create_batch(batch_id="b2", label="EMPTY",
                    section="DM", subtype=None, requested=0,
                    started_at="2026-04-25T12:00:00")
    out = export_batch_html(db, batch_id="b2", dest_folder=tmp_path)
    assert (out / "index.html").exists()
    assert list(out.glob("q*.html")) == []


def test_export_unknown_batch_raises(tmp_path):
    db = Database(":memory:")
    try:
        export_batch_html(db, batch_id="missing", dest_folder=tmp_path)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "missing" in str(e)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_export_html.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ucat.export'`

- [ ] **Step 3: Create export module**

Create `ucat/export.py`:

```python
"""HTML export for bulk-generated question batches.

Writes one folder per batch:
    <dest>/batch-<label>/
        index.html       — list of items with links
        q1.html, q2.html — one per saved question item

Visuals (QR charts, AR panels, DM venns) reuse `ucat.rendering` to produce
PNGs that are referenced as data: URIs so each HTML file is fully standalone.
"""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Any, Dict, List

from .db import Database


def _safe_name(label: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in label)


def _png_data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _render_visuals_to_pngs(data: Dict[str, Any]) -> List[bytes]:
    """Best-effort visual rendering. Returns empty list on failure."""
    try:
        from . import rendering
    except Exception:
        return []
    pngs: List[bytes] = []
    section = data.get("section")
    try:
        if section == "QR" and data.get("stimulus"):
            buf = io.BytesIO()
            rendering.render_qr_chart(data["stimulus"], out=buf)
            pngs.append(buf.getvalue())
        elif section == "AR":
            for kind in ("set_a_panels", "set_b_panels", "test_panels"):
                for panel in data.get(kind) or []:
                    buf = io.BytesIO()
                    rendering.render_ar_panel(panel, out=buf)
                    pngs.append(buf.getvalue())
        elif section == "DM":
            for q in data.get("questions") or []:
                if q.get("venn"):
                    buf = io.BytesIO()
                    rendering.render_venn(q["venn"], out=buf)
                    pngs.append(buf.getvalue())
    except Exception:
        # Rendering is decorative — never block export on a failure.
        pass
    return pngs


def _item_html(item: Dict[str, Any], idx: int) -> str:
    data = item["data"]
    section = data.get("section", "?")
    qs = data.get("questions") or []
    parts: List[str] = [
        "<!doctype html><html><head>",
        "<meta charset='utf-8'>",
        f"<title>Item {idx} — {section}</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:780px;margin:2rem auto;padding:0 1rem;line-height:1.55}",
        "h1{font-size:1.2rem;color:#444}h2{font-size:1rem;margin-top:1.5rem}",
        ".meta{color:#888;font-size:.9rem}.option{margin:.25rem 0}",
        ".answer{color:#2a7;font-weight:600}.explanation{background:#f6f6f6;padding:.75rem;border-radius:.4rem;margin-top:.5rem}",
        "img{max-width:100%;border:1px solid #ddd;border-radius:.3rem;margin:.5rem 0}",
        "</style></head><body>",
        f"<h1>{section} item {idx}</h1>",
        f"<p class='meta'>Generated {html.escape(item.get('created') or '')}</p>",
    ]
    if data.get("passage"):
        parts.append(f"<h2>Passage</h2><p>{html.escape(data['passage'])}</p>")
    for png in _render_visuals_to_pngs(data):
        parts.append(f"<img src='{_png_data_uri(png)}'>")
    for q in qs:
        parts.append(f"<h2>Q{q.get('number','?')}. {html.escape(q.get('text',''))}</h2>")
        for label, txt in (q.get("options") or {}).items():
            parts.append(
                f"<div class='option'><b>{html.escape(label)}.</b> {html.escape(str(txt))}</div>"
            )
        parts.append(
            f"<p class='answer'>Answer: {html.escape(q.get('answer','?'))}</p>"
            f"<div class='explanation'>{html.escape(q.get('explanation',''))}</div>"
        )
    parts.append("<p><a href='index.html'>← back to index</a></p>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _index_html(label: str, items: List[Dict[str, Any]]) -> str:
    rows = []
    for idx, item in enumerate(items, 1):
        section = item["data"].get("section", "?")
        qs = item["data"].get("questions") or []
        first_q = qs[0].get("text", "") if qs else ""
        rows.append(
            f"<li><a href='q{idx}.html'>Item {idx} ({section})</a> "
            f"<span class='meta'>— {html.escape(first_q[:90])}</span></li>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(label)}</title>"
        "<style>body{font-family:system-ui,sans-serif;max-width:780px;margin:2rem auto;padding:0 1rem}"
        "li{margin:.4rem 0}.meta{color:#888;font-size:.9rem}</style></head><body>"
        f"<h1>{html.escape(label)}</h1>"
        f"<p>{len(items)} item(s)</p>"
        "<ol>" + "\n".join(rows) + "</ol></body></html>"
    )


def export_batch_html(db: Database, *, batch_id: str, dest_folder: Path) -> Path:
    """Write index.html + qN.html files for the batch. Returns the folder path."""
    batches = [b for b in db.list_batches() if b["id"] == batch_id]
    if not batches:
        raise ValueError(f"Unknown batch_id: {batch_id}")
    batch = batches[0]
    items = db.questions_by_batch(batch_id)

    out = Path(dest_folder) / f"batch-{_safe_name(batch['label'])}"
    out.mkdir(parents=True, exist_ok=True)

    (out / "index.html").write_text(
        _index_html(batch["label"], items), encoding="utf-8",
    )
    for idx, item in enumerate(items, 1):
        (out / f"q{idx}.html").write_text(_item_html(item, idx), encoding="utf-8")

    return out
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_export_html.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add ucat/export.py tests/test_export_html.py
git commit -m "feat(export): HTML export for bulk batches with embedded visuals"
```

---

## Task 10: Bulk tab UI scaffold

**Files:**
- Modify: `ucat/ui.py` (add tab in `_ui` around line 161-172; new `_tab_bulk` method)
- Test: manual (Tk smoke test inside an off-screen root)

- [ ] **Step 1: Add Tk smoke test for the new tab**

Create `tests/test_ui_bulk_smoke.py`:

```python
"""Smoke test: import the App class and assert the Bulk tab exists.

We don't pump the event loop — we just instantiate enough to verify the
tab method exists, the SUBTYPES dropdown can be built for any section, and
the App.__init__ wires _t_bulk into the notebook.
"""
import os
import pytest

# Skip on systems without a display.
pytest.importorskip("tkinter")

import tkinter as tk


def test_app_imports_without_error():
    from ucat import ui  # noqa: F401


def test_tab_bulk_method_exists():
    from ucat.ui import App
    assert callable(getattr(App, "_tab_bulk", None))


def test_subtypes_dropdown_values_per_section():
    from ucat.config import SUBTYPES
    # Sanity: every section has at least one entry, all are strings.
    for sec, opts in SUBTYPES.items():
        assert opts and all(isinstance(o, str) for o in opts)
```

- [ ] **Step 2: Run smoke test (will fail on `_tab_bulk`)**

Run: `pytest tests/test_ui_bulk_smoke.py -v`
Expected: FAIL — `_tab_bulk` does not exist yet.

- [ ] **Step 3: Wire the new tab into `_ui`**

Edit `ucat/ui.py`. In the `_ui` method (around line 159-172), update the notebook setup:

```python
        self._nb = ttk.Notebook(body)
        self._nb.pack(side="left", fill="both", expand=True)
        self._t_gen      = tk.Frame(self._nb, bg=BG)
        self._t_bulk     = tk.Frame(self._nb, bg=BG)
        self._t_kb       = tk.Frame(self._nb, bg=BG)
        self._t_out      = tk.Frame(self._nb, bg=BG)
        self._t_insights = tk.Frame(self._nb, bg=BG)
        self._nb.add(self._t_gen,      text="  ⚡  GENERATE  ")
        self._nb.add(self._t_bulk,     text="  ☰  BULK  ")
        self._nb.add(self._t_kb,       text="  🗄  KNOWLEDGE BASE  ")
        self._nb.add(self._t_out,      text="  📋  HISTORY  ")
        self._nb.add(self._t_insights, text="  📊  INSIGHTS  ")
        self._tab_gen()
        self._tab_bulk()
        self._tab_kb()
        self._tab_out()
        self._tab_insights()
```

- [ ] **Step 4: Add bulk-tab state to `App.__init__`**

In `App.__init__` (around line 109-116), append after `self._photo_refs`:

```python
        # Bulk-generation state.
        self._bulk_thread: Optional[threading.Thread] = None
        self._bulk_cancel: threading.Event = threading.Event()
        self._bulk_last_batch: Optional[Dict[str, Any]] = None
```

Also add this import at the top of the file with the other imports:

```python
import threading
```

(Check first — it may already be imported. If so, skip.)

- [ ] **Step 5: Add `_tab_bulk` method**

Add this method after `_tab_gen` (around line 378, before `_tab_kb`):

```python
    # ── Bulk tab ──────────────────────────────────────────────────────────────

    def _tab_bulk(self):
        from .config import SUBTYPES
        p = tk.Frame(self._t_bulk, bg=BG)
        p.pack(fill="both", expand=True, padx=24, pady=20)

        tk.Label(p, text="Bulk Generate", bg=BG, fg=TEXT, font=FT).pack(anchor="w")
        tk.Label(p,
                 text="Generate many questions of one section + subtype. "
                      "Sequential. Saved to the question DB with a batch_id; "
                      "use the History tab to review or click Export to dump HTML.",
                 bg=BG, fg=MUTED, font=FS, wraplength=1100, justify="left"
                 ).pack(anchor="w", pady=(2, 14))

        # Section radios.
        sr = tk.Frame(p, bg=BG); sr.pack(anchor="w", pady=(0, 10))
        tk.Label(sr, text="Section:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self._bulk_sec = tk.StringVar(value="DM")
        for code in SECTIONS:
            tk.Radiobutton(sr, text=f" {code} ", variable=self._bulk_sec, value=code,
                           bg=BG, fg=TEXT, selectcolor=PANEL, activebackground=BG,
                           activeforeground=ACCENT, font=FB, indicatoron=False,
                           relief="flat", bd=1, padx=12, pady=6, cursor="hand2",
                           command=self._bulk_section_changed,
                           ).pack(side="left", padx=4)

        # Subtype dropdown.
        sb_row = tk.Frame(p, bg=BG); sb_row.pack(anchor="w", pady=(0, 10))
        tk.Label(sb_row, text="Subtype:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self._bulk_subtype = tk.StringVar(value=SUBTYPES["DM"][0])
        self._bulk_subtype_cb = ttk.Combobox(
            sb_row, textvariable=self._bulk_subtype, width=18, state="readonly",
            values=SUBTYPES["DM"],
        )
        self._bulk_subtype_cb.pack(side="left")

        # Quantity.
        qr = tk.Frame(p, bg=BG); qr.pack(anchor="w", pady=(0, 10))
        tk.Label(qr, text="Number of questions:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self._bulk_qty = tk.IntVar(value=10)
        qty_entry = tk.Spinbox(qr, from_=1, to=100, textvariable=self._bulk_qty,
                                bg=PANEL2, fg=TEXT, font=FM, width=6,
                                relief="flat", buttonbackground=PANEL)
        qty_entry.pack(side="left")
        self._bulk_qty_hint = tk.Label(qr, text="", bg=BG, fg=MUTED, font=FS)
        self._bulk_qty_hint.pack(side="left", padx=12)
        self._bulk_qty.trace_add("write", lambda *_: self._bulk_update_hint())

        # Label.
        lr = tk.Frame(p, bg=BG); lr.pack(anchor="w", pady=(0, 10))
        tk.Label(lr, text="Batch label (optional):", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self._bulk_label = tk.StringVar()
        tk.Entry(lr, textvariable=self._bulk_label, bg=PANEL2, fg=TEXT,
                 font=FM, insertbackground=ACCENT, relief="flat", width=46
                 ).pack(side="left")

        # Action row.
        ar = tk.Frame(p, bg=BG); ar.pack(anchor="w", pady=(8, 10))
        self._bulk_start_btn = mkbtn(ar, "▶  START BATCH", self._do_bulk_start,
                                       padx=22, pady=10, font=("Courier New", 12, "bold"))
        self._bulk_start_btn.pack(side="left", padx=(0, 12))
        self._bulk_cancel_btn = mkbtn(ar, "■  CANCEL", self._do_bulk_cancel,
                                        bg=DANGER, fg="white", padx=14, pady=10,
                                        state="disabled")
        self._bulk_cancel_btn.pack(side="left", padx=(0, 12))
        self._bulk_export_btn = mkbtn(ar, "📁  EXPORT LAST BATCH…", self._do_bulk_export,
                                        bg=PANEL, fg=ACCENT, font=FS, padx=14, pady=10,
                                        state="disabled")
        self._bulk_export_btn.pack(side="left")

        # Progress.
        pr = tk.Frame(p, bg=BG); pr.pack(fill="x", pady=(8, 6))
        self._bulk_progbar = ttk.Progressbar(pr, mode="determinate", length=420)
        self._bulk_progbar.pack(side="left", padx=(0, 12))
        self._bulk_prog_lbl = tk.Label(pr, text="—", bg=BG, fg=MUTED, font=FS)
        self._bulk_prog_lbl.pack(side="left")

        # Run log.
        tk.Label(p, text="RUN LOG", bg=BG, fg=MUTED, font=FH).pack(anchor="w", pady=(10, 4))
        lw = tk.Frame(p, bg=BORDER, bd=1); lw.pack(fill="both", expand=True)
        self._bulk_log = scrolledtext.ScrolledText(
            lw, bg=PANEL, fg=TEXT, font=FS, relief="flat",
            padx=12, pady=10, wrap=tk.WORD, height=14, state="disabled",
        )
        self._bulk_log.pack(fill="both", expand=True)

        self._bulk_section_changed()
        self._bulk_update_hint()
        self._bulk_restore_last_batch()

    def _bulk_restore_last_batch(self):
        """On app launch, surface the most recent batch so 'Export last batch…'
        is immediately usable without re-running."""
        batches = self.db.list_batches(limit=1)
        if not batches:
            return
        last = batches[0]
        self._bulk_last_batch = {
            "batch_id": last["id"], "label": last["label"],
            "succeeded": last["succeeded"], "failed": last["failed"],
            "cancelled": last["cancelled"],
            "total_items": last["requested"],
        }
        self._bulk_export_btn.config(state="normal")
        status = "incomplete" if last["completed_at"] is None else "complete"
        self._bulk_prog_lbl.config(
            text=f"Last batch: {last['label']} ({status}, {last['succeeded']} saved)"
        )

    def _bulk_section_changed(self):
        from .config import SUBTYPES
        sec = self._bulk_sec.get()
        opts = SUBTYPES[sec]
        self._bulk_subtype_cb.config(values=opts)
        self._bulk_subtype.set(opts[0])
        # AR has only "standard" — disable the dropdown.
        self._bulk_subtype_cb.config(state="disabled" if sec == "AR" else "readonly")
        self._bulk_update_hint()

    def _bulk_update_hint(self):
        try:
            qty = int(self._bulk_qty.get())
        except (tk.TclError, ValueError):
            self._bulk_qty_hint.config(text="")
            return
        sec = self._bulk_sec.get()
        per_set = {"VR": 4, "QR": 4, "AR": 5, "DM": 1}[sec]
        if sec == "DM":
            self._bulk_qty_hint.config(text=f"≈ {qty} standalone question(s)")
        else:
            import math
            sets = math.ceil(qty / per_set)
            kind = {"VR":"passage(s)", "QR":"chart(s)", "AR":"AR set(s)"}[sec]
            self._bulk_qty_hint.config(
                text=f"≈ {sets} {kind}, {sets * per_set} questions total",
            )

    def _bulk_log_append(self, line: str):
        self._bulk_log.config(state="normal")
        self._bulk_log.insert("end", line + "\n")
        self._bulk_log.see("end")
        self._bulk_log.config(state="disabled")

    # Stub handlers — implemented in Task 11/12.
    def _do_bulk_start(self):
        messagebox.showinfo("Not implemented", "Bulk start handler arrives in Task 11.")

    def _do_bulk_cancel(self):
        pass

    def _do_bulk_export(self):
        pass
```

- [ ] **Step 6: Run smoke tests**

Run: `pytest tests/test_ui_bulk_smoke.py -v`
Expected: 3 passed

- [ ] **Step 7: Manual verification (no API calls)**

Run: `python ucat_trainer.py`
Expected: app opens, the **BULK** tab is between GENERATE and KNOWLEDGE BASE, clicking it shows section radios, subtype dropdown, quantity spinbox, label entry, action row, progress bar, and an empty run log.
- Click each section radio: subtype dropdown should repopulate; AR should disable the dropdown showing "standard".
- Type in the quantity spinbox: hint should update live.
- Click START BATCH: shows the "Not implemented" dialog.

- [ ] **Step 8: Commit**

```bash
git add ucat/ui.py tests/test_ui_bulk_smoke.py
git commit -m "feat(ui): scaffold Bulk tab with section/subtype/quantity controls"
```

---

## Task 11: Bulk tab — start and cancel handlers

**Files:**
- Modify: `ucat/ui.py` (replace stub handlers from Task 10)

- [ ] **Step 1: Replace `_do_bulk_start` and `_do_bulk_cancel`**

Edit `ucat/ui.py`. Replace the three stub methods (`_do_bulk_start`, `_do_bulk_cancel`, `_do_bulk_export`) with:

```python
    def _do_bulk_start(self):
        ok, msg = api_status()
        if not ok:
            messagebox.showerror("API Not Ready",
                f"{msg}\n\nCopy .env.example → .env and fill in your keys.")
            return

        section = self._bulk_sec.get()
        subtype = self._bulk_subtype.get()
        try:
            qty = int(self._bulk_qty.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid quantity", "Quantity must be a positive integer.")
            return
        if qty < 1 or qty > 100:
            messagebox.showerror("Invalid quantity", "Quantity must be between 1 and 100.")
            return
        label = self._bulk_label.get().strip() or None

        if self.db.count(section, indexed_only=True) == 0:
            if not messagebox.askyesno("No Indexed Documents",
                f"No indexed documents for {SECTIONS[section]}.\n\n"
                "Continue without RAG context?"):
                return

        self._bulk_start_btn.config(state="disabled")
        self._bulk_cancel_btn.config(state="normal")
        self._bulk_export_btn.config(state="disabled")
        self._bulk_log.config(state="normal")
        self._bulk_log.delete("1.0", "end")
        self._bulk_log.config(state="disabled")
        self._bulk_progbar.config(value=0, maximum=100)
        self._bulk_prog_lbl.config(text="starting…")
        self._bulk_cancel.clear()

        def on_progress(i: int, total: int, status: str, message: str):
            def apply():
                pct = (i / total) * 100 if total else 0
                self._bulk_progbar.config(value=pct, maximum=100)
                self._bulk_prog_lbl.config(text=f"{i}/{total}  ·  {status}")
                ts = datetime.now().strftime("%H:%M:%S")
                self._bulk_log_append(f"[{ts}] {message}")
            self.after(0, apply)

        def worker():
            try:
                result = self.rag.generate_bulk(
                    section=section,
                    subtype=(subtype if subtype != "mixed" else None),
                    quantity=qty, label=label,
                    on_progress=on_progress,
                    cancel_event=self._bulk_cancel,
                )
                self.after(0, lambda: self._bulk_done(result))
            except Exception as e:
                logger.exception("Bulk run failed")
                self.after(0, lambda err=str(e): self._bulk_failed(err))

        self._bulk_thread = threading.Thread(target=worker, daemon=True)
        self._bulk_thread.start()

    def _do_bulk_cancel(self):
        if self._bulk_thread and self._bulk_thread.is_alive():
            self._bulk_cancel.set()
            self._bulk_log_append("⟶ cancel requested — finishing current item…")
            self._bulk_cancel_btn.config(state="disabled")

    def _bulk_done(self, result: Dict[str, Any]):
        self._bulk_last_batch = result
        self._bulk_start_btn.config(state="normal")
        self._bulk_cancel_btn.config(state="disabled")
        self._bulk_export_btn.config(state="normal")
        summary = (
            f"✓ {result['succeeded']} ok  ·  ✗ {result['failed']} failed"
            + ("  ·  cancelled" if result['cancelled'] else "")
        )
        self._bulk_prog_lbl.config(text=summary)
        self._bulk_log_append(f"── batch {result['label']} done — {summary}")

    def _bulk_failed(self, err: str):
        self._bulk_start_btn.config(state="normal")
        self._bulk_cancel_btn.config(state="disabled")
        self._bulk_prog_lbl.config(text="error")
        self._bulk_log_append(f"✗ batch failed: {err}")

    def _do_bulk_export(self):
        # Implemented in Task 12.
        messagebox.showinfo("Not implemented", "Export arrives in Task 12.")
```

- [ ] **Step 2: Manual end-to-end with a tiny batch**

Run: `python ucat_trainer.py`
- Open BULK tab.
- Section: DM, Subtype: syllogism, Quantity: 2, Label: (blank).
- Click START BATCH.
- Expected: progress bar advances; log shows `started`, `succeeded` lines for items 1 and 2; final summary "✓ 2 ok · ✗ 0 failed".
- Switch to HISTORY tab; the two new DM rows are visible.
- Click START BATCH again with quantity 5; press CANCEL after the first item completes.
- Expected: log shows `cancel requested — finishing current item…`; final summary includes `cancelled`.

- [ ] **Step 3: Commit**

```bash
git add ucat/ui.py
git commit -m "feat(ui): wire Bulk tab start/cancel handlers to RAGEngine.generate_bulk"
```

---

## Task 12: Bulk tab — export handler

**Files:**
- Modify: `ucat/ui.py` (replace `_do_bulk_export`)

- [ ] **Step 1: Implement `_do_bulk_export`**

Edit `ucat/ui.py`. Replace the `_do_bulk_export` stub with:

```python
    def _do_bulk_export(self):
        if not self._bulk_last_batch:
            messagebox.showinfo("No batch", "Run a batch first.")
            return
        from tkinter import filedialog
        from pathlib import Path
        from .export import export_batch_html

        dest = filedialog.askdirectory(title="Choose export folder")
        if not dest:
            return
        try:
            out = export_batch_html(
                self.db,
                batch_id=self._bulk_last_batch["batch_id"],
                dest_folder=Path(dest),
            )
        except Exception as e:
            logger.exception("Export failed")
            messagebox.showerror("Export failed", str(e))
            return
        self._bulk_log_append(f"📁 exported to {out}")
        messagebox.showinfo("Export complete", f"Wrote files to:\n{out}")
```

- [ ] **Step 2: Manual verification**

Run: `python ucat_trainer.py`
- Run a small DM-syllogism batch (quantity=2) from Task 11.
- Click EXPORT LAST BATCH…; pick `/tmp` (or any folder).
- Expected: success dialog; folder `/tmp/batch-DM-syllogism-…/` exists with `index.html` and `q1.html`, `q2.html`.
- Open `index.html` in a browser and click each item link.

- [ ] **Step 3: Commit**

```bash
git add ucat/ui.py
git commit -m "feat(ui): wire Bulk tab export handler to ucat.export"
```

---

## Task 13: History tab — Filter by batch

**Files:**
- Modify: `ucat/ui.py` (`_tab_out` and `_refresh_out`)

- [ ] **Step 1: Add batch filter controls to `_tab_out`**

Edit `ucat/ui.py`. Find `_tab_out` (around line 415). Replace the `top` Frame block with:

```python
    def _tab_out(self):
        top = tk.Frame(self._t_out, bg=BG); top.pack(fill="x", padx=22, pady=(16, 6))
        tk.Label(top, text="Output History", bg=BG, fg=TEXT, font=FT).pack(side="left")

        # Batch filter dropdown.
        ff = tk.Frame(top, bg=BG); ff.pack(side="left", padx=(24, 0))
        tk.Label(ff, text="Batch:", bg=BG, fg=MUTED, font=FS).pack(side="left", padx=(0, 6))
        self._out_batch_var = tk.StringVar(value="ALL")
        self._out_batch_cb = ttk.Combobox(
            ff, textvariable=self._out_batch_var, width=36, state="readonly",
            values=["ALL"],
        )
        self._out_batch_cb.pack(side="left")
        self._out_batch_cb.bind("<<ComboboxSelected>>", lambda _e: self._refresh_out())

        mkbtn(top, "↻  Refresh", self._refresh_out, bg=PANEL, fg=MUTED, font=FS, pady=5
              ).pack(side="right")
```

- [ ] **Step 2: Update `_refresh_out` to apply the batch filter**

Find `_refresh_out` (around line 863). Replace the entire method with the version below. The original 21-line body is preserved verbatim from `for iid in ...` onward; only the data-fetch and dropdown-refresh logic at the top is new:

```python
    def _refresh_out(self):
        # Refresh the batch dropdown values.
        batches = self.db.list_batches(limit=100)
        choices = ["ALL"] + [f"{b['label']}  ({b['id'][:8]})" for b in batches]
        self._out_batch_cb.config(values=choices)
        if self._out_batch_var.get() not in choices:
            self._out_batch_var.set("ALL")

        rows = self.db.get_generated(limit=500)
        sel = self._out_batch_var.get()
        if sel != "ALL":
            # Recover the batch_id prefix from "label  (xxxxxxxx)".
            wanted_id_prefix = sel.rsplit("(", 1)[-1].rstrip(")")
            rows = [r for r in rows
                    if (r.get("batch_id") or "").startswith(wanted_id_prefix)]

        for iid in self._outt.get_children(): self._outt.delete(iid)
        for r in rows:
            cost = f"${r['usage']['cost_usd']:.3f}" if r.get("usage") else "—"
            d = r.get("difficulty")
            d_str = f"{d:.1f}" if isinstance(d, (int, float)) else "—"
            v = r.get("verdict") or {}
            if not v:
                badge = "—"
            elif v.get("overall_correct", True):
                badge = "✓"
            else:
                fq = len(v.get("flagged_questions") or [])
                sym = (v.get("symbolic_qr") or {}).get("disagreed") or []
                badge = f"⚠ {fq + len(sym)}"
            self._outt.insert("", "end", iid=str(r["id"]),
                               values=(r["id"], SECTIONS.get(r["section"], r["section"]),
                                       d_str, cost, badge,
                                       (r["created"] or "")[:16]))
```

- [ ] **Step 3: Surface `batch_id` on rows from `get_generated`**

Edit `ucat/db.py`. Update `get_generated` (around line 214) to include `batch_id` in the SELECT and return dict:

```python
    def get_generated(self, limit: int = 500):
        c = self.conn.cursor()
        rows = c.execute(
            "SELECT id,section,data,context_ids,usage,verdict,coverage,difficulty,created,batch_id"
            " FROM generated ORDER BY created DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{
            "id": r[0], "section": r[1], "data": json.loads(r[2]),
            "context_ids": json.loads(r[3]),
            "usage":    json.loads(r[4]) if r[4] else None,
            "verdict":  json.loads(r[5]) if r[5] else None,
            "coverage": json.loads(r[6]) if r[6] else None,
            "difficulty": r[7],
            "created": r[8],
            "batch_id": r[9],
        } for r in rows]
```

- [ ] **Step 4: Add a regression test for `get_generated.batch_id`**

Append to `tests/test_db_batch.py`:

```python
def test_get_generated_includes_batch_id():
    db = Database(":memory:")
    db.create_batch(batch_id="b1", label="L", section="DM", subtype=None,
                    requested=1, started_at="2026-04-25T12:00:00")
    db.add_generated("DM", {"section":"DM","questions":[]}, [], batch_id="b1")
    db.add_generated("DM", {"section":"DM","questions":[]}, [])  # no batch
    rows = db.get_generated()
    assert any(r["batch_id"] == "b1" for r in rows)
    assert any(r["batch_id"] is None for r in rows)
```

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: all green.

- [ ] **Step 6: Manual verification**

Run: `python ucat_trainer.py`
- Open HISTORY tab; the Batch dropdown shows "ALL" plus any prior bulk runs.
- Pick a batch from the dropdown; the table filters to those items.
- Pick "ALL"; the table shows everything.

- [ ] **Step 7: Commit**

```bash
git add ucat/ui.py ucat/db.py tests/test_db_batch.py
git commit -m "feat(ui): History tab gains 'Filter by batch' dropdown"
```

---

## Self-Review Checklist (run before handoff)

After all 13 tasks pass, verify:

1. **Spec coverage** — for each spec section, point to a task:
   - Subtype taxonomy → Task 2
   - Generation unit per section → Task 6 (constraint), Task 7 (single-DM), Task 8 (loop math)
   - UI tab → Tasks 10, 11, 12
   - Threading model → Task 11 (worker thread + cancel event)
   - `batch_id` on questions + `batches` table → Tasks 3, 4, 5
   - Subtype prompt rewrites → Task 6
   - DM single-question schema → Task 7
   - Per-item failure / 1 retry → Task 8
   - HTML export → Task 9 (impl) + Task 12 (UI)
   - Output tab batch filter → Task 13
   - AR single-option dropdown disabled → Task 10 (`_bulk_section_changed`)
   - Live "≈ N items, M questions" hint → Task 10 (`_bulk_update_hint`)

2. **Type consistency** — `generate_bulk` returns `{batch_id, label, succeeded, failed, cancelled, total_items}`; UI `_bulk_done` reads exactly those keys. ✓

3. **Run full suite** — `pytest tests/ -v` shows all green.

4. **Manual smoke** — open the app, run a 2-question DM-syllogism batch, cancel a 5-question batch, export a completed batch, filter History by batch.
