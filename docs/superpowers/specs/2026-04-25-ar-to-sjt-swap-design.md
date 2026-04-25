# UCAT Trainer: Swap AR for SJT

**Date:** 2026-04-25
**Status:** Design — awaiting approval
**Authors:** Ian (product), Claude (design)

## Goal

The active `ucat/` package treats Abstract Reasoning (AR) as a first-class section with structured shape-panel rendering. The real UCAT no longer includes AR; it tests Situational Judgement (SJT) instead. Replace AR with full SJT support across `ucat/` so the four sections become VR / DM / QR / SJT, with parity for generation, retrieval, calibration, coverage, and UI.

## Why this shape

- **Aligns with the live UCAT format.** Generated questions are useless if their format doesn't match the test students will sit.
- **Crawler-import is already half-migrated.** `ucat/crawler_import.py` already groups SJT (4-question scenarios) and skips AR with a warning. Finishing the swap removes a half-built state.
- **Reference schema exists.** The orphaned `src/` directory has a complete SJT design (config, prompts, retrieval strategy, embedding instructions). We mirror it rather than designing from scratch.
- **No DB migration required.** The `section` column is unconstrained TEXT; existing AR rows can sit untouched and become invisible to the live app.

## Out of scope

- **Cleaning up the legacy `src/` directory.** It is orphaned (not imported by `ucat_trainer.py` or `ucat/`). A separate spec can delete it.
- **Hard-deleting historical AR rows.** They stay in the DB, hidden by query filters. The user can run `DELETE FROM kb WHERE section='AR'` manually whenever they choose.
- **Multi-action ranking SJT format.** Real UCAT used to have "rank these 4 actions from most to least appropriate" questions; modern UCAT is single-rating only. We support single-rating only (one `rating_type` per question).
- **AR rendering helpers.** `render_ar_set` / `render_ar_panel` and their imports are removed entirely. No section uses them.

## Section 1 — SJT schema

### Pydantic model (`ucat/models.py`)

Replace `ARSet` with:

```python
class SJTQuestion(Question):
    """Each SJT question rates one action/consideration on a 4-point scale."""
    rating_type: Literal["appropriateness", "importance"]


class SJTSet(BaseModel):
    section: Literal["SJT"]
    scenario: str = Field(min_length=50,
                            description="Realistic clinical/professional situation, 80-150 words.")
    questions: List[SJTQuestion] = Field(min_length=4, max_length=4)
```

`SECTION_MODELS` swaps `"AR": ARSet` → `"SJT": SJTSet`. Drop `ARShape`, `ARShapeKind`, `ARShapeColor`, `ARShapeSize`, `ARPanel`, `ARSet` and any imports.

### Fixed-label option tables

Each `SJTQuestion.options` dict must equal the canonical map for its `rating_type`:

| `rating_type` | A | B | C | D |
|---|---|---|---|---|
| `appropriateness` | Very appropriate | Appropriate, but not ideal | Inappropriate, but not awful | Very inappropriate |
| `importance` | Very important | Important | Of minor importance | Not important at all |

Enforced by `verification.py` at validation time (see Section 4).

### Section config (`ucat/config.py`)

```python
SECTIONS = {
    "VR": "Verbal Reasoning",
    "DM": "Decision Making",
    "QR": "Quantitative Reasoning",
    "SJT": "Situational Judgement",
}
SECTION_COLORS = {
    "VR": "#4A90D9", "DM": "#E8943A",
    "QR": "#3FB950", "SJT": "#A78BFA",  # SJT inherits AR's purple
}
SECTION_DESC["SJT"] = (
    "A scenario (80-150 words) describing a realistic professional/clinical "
    "situation, followed by exactly 4 questions. Each question rates one action "
    "or consideration on a 4-point scale (appropriateness OR importance, A-D). "
    "Options are fixed by rating type."
)
```

Drop `SECTIONS["AR"]`, `SECTION_COLORS["AR"]`, `SECTION_DESC["AR"]`.

### Settings safety

In `Settings.load()`, after loading, coerce: if `self.data["bulk_section"] not in SECTIONS`, reset to `"VR"`. Protects against stale `ucat_settings.json` files holding `"bulk_section": "AR"`.

## Section 2 — Generation prompts and seed corpus

### `ucat/rag.py`

- Drop the AR-specific block at `rag.py:119` (the "AR panels are STRUCTURED shape lists" block).
- Drop the AR test-item key hint at `rag.py:105` ("AR test items → keys exactly Set A, Set B, Neither").
- Add an SJT block in `_system_blocks`:

```text
SJT scenarios MUST be realistic professional/clinical situations (80-150 words)
involving an interpersonal or ethical dilemma a healthcare student/professional
might face. Generate exactly 4 questions per scenario. Each question MUST set
rating_type to either "appropriateness" or "importance" and use the EXACT fixed
option labels for that type — do not paraphrase. Mix at least one
"importance" and one "appropriateness" question per set. Avoid obviously
correct/incorrect answers; the most-correct rating should require nuanced
judgement.
```

### `ucat/samples.py`

Replace the AR sample (line 167+) with one SJT sample:

- Scenario: a 100-word clinical scenario (e.g. F1 doctor witnesses a colleague making an error).
- 4 questions: 2 `appropriateness` + 2 `importance`, each with the canonical 4-option map.
- Plausibly graded difficulties (mix 2.0 to 4.0 logits).
- Coverage tags reflect medical/social scenario types.

This single seed unblocks SJT retrieval until crawler imports flesh out the corpus.

## Section 3 — Per-section logic

### `ucat/coverage.py`

```python
EXPECTED_SCENARIOS = {
    "VR":  {"scientific", "humanities", "business", "social", "everyday"},
    "DM":  {"scientific", "humanities", "business", "everyday", "abstract"},
    "QR":  {"business", "everyday", "scientific"},
    "SJT": {"medical", "social", "everyday"},   # replaces AR's {"abstract"}
}
```

Per-question coverage logic at line 79+ already loops generically; no further branching needed.

### `ucat/calibration.py`

Replace the `AR` branch at line 104 (option-count weighting — irrelevant to SJT) with an SJT branch:

- Sentence count of the scenario (longer = harder).
- Named-entity density (more entities = harder to track).
- Rating-type mix (mixed sets harder than uniform).

Mirror the structure of the existing VR branch (line 83+); same return semantics.

### `ucat/format.py`

Replace the AR text-rendering block at line 43 (set rules + panel summaries) with an SJT block:

```text
Scenario:
  <80-150 word scenario, wrapped at 64 cols>

Questions:
  Q1 [appropriateness] — How appropriate is X?
    Answer: A (Very appropriate)
    Explanation: ...
  Q2 [importance] — ...
```

### `ucat/rendering.py`

- Drop the entire `elif section == "AR"` branch at line 503.
- Delete `render_ar_set`, `render_ar_panel`, and their helpers along with PIL imports they exclusively use.
- `render_visuals_for("SJT", data)` returns `{}` (no visuals; falls through naturally).

### `ucat/ui.py`

- Drop the AR visuals branch at line 1144 (Set A / Set B / Test Shape rendering).
- SJT relies on the existing fall-through "no visuals for this section" path — no positive code needed for SJT visuals.
- The KB browser callsites that list/render docs (the KB tab table population, any "show docs" panel) get `WHERE section != 'AR'` added at the SQL level — applied at the *callsite*, not as a global `Database` filter, so internal paths (e.g. coverage counts, future migration tooling) can still see AR rows when needed.
- The bulk-tab section dropdown reads from `SECTIONS`, so dropping `"AR"` from config removes it from the UI automatically.

### Canonical label constants (`ucat/models.py`)

The two label maps are the source of truth for the schema and are used in three places (verification, generation prompt, output formatting). Define them once in `models.py` next to `SJTQuestion`:

```python
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
SJT_LABELS = {"appropriateness": APPROPRIATENESS_LABELS, "importance": IMPORTANCE_LABELS}
```

`rag.py` imports these to interpolate into the SJT prompt block. `verification.py` imports them for the syntactic check. `format.py` imports them when annotating answers.

### `ucat/verification.py`

Add a pre-LLM-judge syntactic check in `llm_judge` (or a helper called from it) for SJT sets: for each SJT question, if `rating_type` is set, `options` must equal `SJT_LABELS[rating_type]` exactly (key set and values, case-sensitive). On mismatch, return a `Verdict` with `overall_correct=False`, `confidence="high"`, and a `notes` entry naming the offending question — same shape as existing missing-key rejection. The LLM judge is not invoked when the syntactic check fails.

### `ucat/db.py`

Replace the AR embed-text block at lines 43–50 with an SJT block:

```python
if section == "SJT":
    parts.append(str(data.get("scenario", ""))[:1200])
    rating_types = [q.get("rating_type", "?") for q in (data.get("questions") or [])]
    parts.append(f"Rating types: {', '.join(rating_types)}")
```

The generic `for q in questions[:5]` loop afterwards already pulls in question text; no further per-section branching.

## Section 4 — Hidden AR rows (the "keep but hide" strategy)

- KB browser queries (any UI list of stored docs) get `WHERE section != 'AR'`.
- Coverage report (`coverage.py:149+`) iterates `EXPECTED_SCENARIOS`, which no longer contains AR; AR rows become invisible to the report automatically.
- `SECTION_MODELS` only knows VR/DM/QR/SJT — any attempt to `validate_schema(ar_row)` will raise. Acceptable because the UI never offers AR.
- One-time startup log: `logger.info("AR section retired; %d historical rows hidden", db.count_section('AR'))` so it's visible if cleanup is ever wanted.

## Section 5 — Testing

### Unit tests (new file `tests/test_sjt.py` or section in existing test file)

- `SJTSet` model accepts a valid set (4 questions, mixed rating types, fixed labels).
- `SJTSet` rejects: scenario < 50 chars; question count != 4; question with options not matching its `rating_type` canonical map; missing `rating_type`.
- `Settings.load()` coerces `bulk_section: "AR"` → `"VR"`.
- `verification.llm_judge` syntactic check rejects SJT sets with mismatched labels before calling the LLM.

### Integration tests

- Generation smoke: seed the SJT sample, run `rag.generate("SJT", hint="")` (mocked LLM returning a fixture), assert the result validates and the verifier passes.
- Crawler-import round-trip: drop a fixture `trainer-import.json` containing one SJT bundle into a temp KB, run import, generate, confirm retrieval picks up the seeded scenario.

### Manual UI test (golden-path checklist)

- App launches without errors; tabs read VR / DM / QR / SJT.
- Bulk tab dropdown shows VR / DM / QR / SJT (no AR).
- Generate one SJT set; format panel renders scenario + per-question rating types; visuals panel shows "(no visuals for Situational Judgement)".
- KB browser shows no AR rows even if AR rows exist in the DB.
- Generated SJT set persists, can be re-loaded, no rendering errors.

## Section 6 — Rollout

One PR. The changes are tightly coupled: a half-merged state (e.g. `config.py` updated but `models.py` still defines `ARSet`) would crash on startup. The change set is mechanical (~10 files, all in `ucat/`), small enough to review as one diff, and the test plan covers both code paths and a manual smoke run.

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Stale `ucat_settings.json` on user machines holds `bulk_section: "AR"` | `Settings.load()` coerces unknown values back to `"VR"` |
| Historical AR rows in user DBs surface in some unfiltered query | `WHERE section != 'AR'` filter in KB browser; `SECTION_MODELS` excludes AR so loading raises a clear error if it slips through |
| `render_ar_*` helpers might be used by something I missed | Grep before delete; if any caller exists, keep the function and mark with deprecation note |
| LLM ignores fixed-label instruction and paraphrases options | Pre-LLM-judge syntactic check rejects mismatched labels and logs the failure; bulk worker retries the set per existing retry policy |
| Generated SJT scenarios drift away from clinical/professional realism | Seed sample anchors retrieval; coverage report tracks scenario types and flags drift toward "abstract" or off-domain |

## File touch list (estimate)

**Modified in `ucat/`:**
- `config.py` — `SECTIONS`, `SECTION_COLORS`, `SECTION_DESC`, `Settings.load()` coercion
- `models.py` — drop AR classes, add `SJTQuestion` + `SJTSet`, add `APPROPRIATENESS_LABELS` / `IMPORTANCE_LABELS` / `SJT_LABELS` constants, update `SECTION_MODELS`
- `rag.py` — drop AR prompt blocks, add SJT prompt block
- `samples.py` — drop AR sample, add one SJT sample
- `coverage.py` — replace `AR` entry in `EXPECTED_SCENARIOS`
- `calibration.py` — replace AR branch with SJT branch in `feature_difficulty`
- `format.py` — replace AR rendering branch with SJT branch
- `rendering.py` — drop AR rendering branch and helpers
- `ui.py` — drop AR visuals branch; add `WHERE section != 'AR'` to KB browser queries
- `verification.py` — add fixed-label check for SJT
- `db.py` — replace AR embed-text branch with SJT branch

**New:**
- `tests/test_sjt.py` (or add to existing test file) — unit + integration tests for the schema, settings coercion, and verifier check

**Untouched:**
- `crawler_import.py` — already handles SJT and skips AR
- `src/` — orphaned, separate spec
- DB schema — `section` column unconstrained TEXT
- `requirements.txt`, `assets/`
