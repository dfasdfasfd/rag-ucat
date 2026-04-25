# Bulk Question Generation by Subtype

**Date:** 2026-04-25
**Status:** Design — awaiting approval
**Authors:** Ian (product), Claude (design)

## Goal

Add a bulk-generation feature to the UCAT Trainer that lets the user pick a section, narrow to a subtype within that section, choose a quantity, and run a batch generation. Results land in the existing question DB tagged with a `batch_id` so they can be reviewed in the Output tab and optionally exported to a folder as HTML files.

The current Generate tab produces exactly one item per click (a passage set, a chart set, an AR set, or 5 standalone DM questions). It cannot target a single DM subtype, a single VR question format, or a single QR chart type — DM is explicitly told to mix all 5 subtypes in one set. Bulk generate solves both: scale + subtype filtering.

## Why this shape

- **Tab-based UI, not modal/inline.** A bulk run of 30 questions takes ~15 minutes at sequential pace. The user must be able to leave it running, switch to the Output or KB tab, and come back without losing progress. A dedicated tab is the right parking spot for a long-lived operation. A modal blocks the rest of the app; an inline toggle on the Generate tab crowds the existing single-item preview.
- **Reuse the existing generation pipeline.** `RAGEngine` already does retrieval, generation, verification, calibration, telemetry, and DB write for one item. Bulk = a loop that calls a slightly modified single-item path with a subtype filter. No new pipeline, no parallel implementation.
- **Sequential before parallel.** Concurrency=1 sidesteps Anthropic rate limits, simplifies cancel semantics, and keeps the Tkinter event loop sane (one worker thread, one progress callback). Parallelisation is a future optimisation.
- **DB persistence + manual export, not auto-export.** Auto-writing files on every run produces a sprawling folder. The user already has the Output tab for review; export is a one-click action when they actually want files (e.g. for a study session offline).

## Out of scope

- **Parallel/concurrent generation.** Sequential only in v1.
- **Resume after crash/restart.** If the user closes the app mid-run, the partially-completed batch survives in the DB but cannot be "continued" — they would start a new run for the missing count.
- **Cross-section batches.** One run targets one section and one subtype. "Generate 50 mixed UCAT questions" is not a goal here.
- **Difficulty scheduling per item.** The whole batch uses the current `target_difficulty` setting; per-item difficulty curves (e.g. "10 easy, 10 medium, 10 hard") are out of scope.
- **PDF export.** v1 exports HTML only. PDF is a follow-up — users can print-to-PDF from a browser if needed.
- **AR subtypes.** AR has only Type 1 implemented today. The subtype dropdown shows a single disabled "Standard" option. Adding Types 2–4 is a separate spec.

## Subtype taxonomy

For each section, the subtype dropdown offers:

| Section | Subtype options                                                       | Notes                                                 |
|---------|-----------------------------------------------------------------------|-------------------------------------------------------|
| VR      | `mixed` (default), `tf` (T/F/Can't Tell), `mcq` (4-option A–D)        | Drives both passage style and question format         |
| DM      | `mixed` (default), `syllogism`, `logical`, `venn`, `probability`, `argument` | Already exists as `Question.type`              |
| QR      | `mixed` (default), `table`, `bar`, `line`, `stacked_bar`, `pie`       | Pins `QRChart.type`                                   |
| AR      | `standard` (only option, disabled)                                    | Future-proof slot                                     |

The `mixed` option for VR/DM/QR preserves the current behaviour (the prompt mixes subtypes within a set) so bulk-mixed is still useful for variety-focused practice.

## Generation unit per section

Per the Q2 = (c) decision, the unit per LLM call depends on what makes natural sense for each section:

| Section | LLM call produces                  | "30 questions" means                 |
|---------|-----------------------------------|--------------------------------------|
| VR      | 1 passage + 4 questions            | `ceil(30/4)` = 8 passages = 32 questions  |
| DM      | 1 standalone question (subtype-locked) | 30 LLM calls = exactly 30 questions  |
| QR      | 1 chart + 4 questions              | `ceil(30/4)` = 8 charts = 32 questions    |
| AR      | 1 full set (12 panels + 5 test Qs) | `ceil(30/5)` = 6 sets = 30 questions      |

For VR/QR/AR, quantity is rounded **up** to the nearest set so the user gets *at least* what they asked for. The UI input is labelled **"Number of questions"**. As the user types, the form shows a live preview: *"≈ 8 passages, 32 questions total"* for VR, *"30 standalone questions"* for DM, etc.

For DM, generating one question at a time requires a **new LLM path**: the existing pipeline returns a `DMSet` with 5 questions. Bulk-DM calls a new `generate_single_dm_question(subtype)` that returns one `DMQuestion`. Same retrieval, same verification, lighter prompt.

## UI

A new **Bulk** tab joins the existing Generate / KB / Output / Insights tabs.

```
┌─ Bulk Generate ────────────────────────────────────────────────┐
│                                                                │
│  Section:    ( ) VR  (•) DM  ( ) QR  ( ) AR                    │
│  Subtype:    [ syllogism ▼ ]                                   │
│  Quantity:   [   30  ]   ≈ 30 standalone questions             │
│  Label:      [ optional batch label ]                          │
│                                                                │
│  [▶ Start batch ]   [■ Cancel ]   [📁 Export last batch… ]    │
│                                                                │
│  ─── Progress ───────────────────────────────────────────────  │
│  ████████░░░░░░░░░░░░░░  12 / 30  •  est. 9 min remaining     │
│                                                                │
│  ─── Run log ────────────────────────────────────────────────  │
│  [12:01:14] Q1 ✓ syllogism, difficulty 3.1                    │
│  [12:01:48] Q2 ✓ syllogism, difficulty 2.8                    │
│  [12:02:21] Q3 ✗ verifier rejected — retrying                 │
│  [12:02:55] Q3 ✓ syllogism, difficulty 3.4                    │
│  …                                                             │
│                                                                │
│  Last batch: DM-syllogism-2026-04-25 (28 ✓ / 2 ✗)             │
└────────────────────────────────────────────────────────────────┘
```

### Controls

- **Section radios** — same VR/DM/QR/AR choices as the Generate tab.
- **Subtype dropdown** — repopulates on section change. Disabled for AR.
- **Quantity** — integer input, 1–100, with a live "≈ N items, M questions total" hint.
- **Label** — optional. Defaults to `{SECTION}-{subtype}-{YYYY-MM-DD}`.
- **Start batch** — disabled while a batch is running. Spawns a worker thread.
- **Cancel** — enabled only while running. Sets a flag the worker checks between items; finishes the in-flight item, then stops.
- **Export last batch…** — enabled when a completed batch exists. Opens a folder picker, writes `batch-{label}/index.html` + `q{n}.html` per item.

### Threading model

A single `threading.Thread` runs the batch loop. Inside the loop, each completed item triggers a `self.after(0, callback, ...)` to update the progress bar, log line, and last-batch summary on the Tk main thread. Cancel sets a `threading.Event`; the worker checks it between items.

If the user clicks Start with a batch already running, the button is disabled — no race.

## Data model

### `batch_id` on questions

Add a column `batch_id TEXT` to the questions table (nullable — single-Generate items leave it NULL). The Output tab gains a "Filter by batch" dropdown that lists distinct `batch_id`s in reverse-chrono order with their item counts.

### Batch record

A new `batches` table:

```sql
CREATE TABLE batches (
    id           TEXT PRIMARY KEY,         -- uuid
    label        TEXT NOT NULL,            -- "DM-syllogism-2026-04-25"
    section      TEXT NOT NULL,            -- VR/DM/QR/AR
    subtype      TEXT,                     -- nullable for "mixed"
    requested    INTEGER NOT NULL,         -- target question count
    succeeded    INTEGER NOT NULL DEFAULT 0,
    failed       INTEGER NOT NULL DEFAULT 0,
    started_at   TEXT NOT NULL,            -- ISO timestamp
    completed_at TEXT,                     -- NULL while running
    cancelled    INTEGER NOT NULL DEFAULT 0
);
```

This lets the Bulk tab restore "last batch" status across app restarts and powers the export feature.

## Prompt changes

`RAGEngine._system_blocks` gains an optional `subtype: str | None` parameter. When non-null:

- **VR**: prepend "ALL 4 questions in this set must be {True/False/Can't Tell items} or {4-option MCQ items}, not mixed."
- **DM** (set generation, when used with mixed): unchanged. (DM bulk single-question path uses a different system block — see below.)
- **QR**: append "Set `stimulus.type` to `{table|bar|line|stacked_bar|pie}`. Do not vary chart type."
- **AR**: ignored (only one type).

For **DM single-question generation**, a new method `RAGEngine.generate_single_dm_question(subtype)` builds a much shorter system block: role + the single-subtype constraint + retrieval. Output schema is `DMQuestion` (not `DMSet`). The verification pass also runs on the single question.

## Module changes

| File              | Change                                                                          |
|-------------------|---------------------------------------------------------------------------------|
| `ucat/models.py`  | Add `batch_id: Optional[str] = None` to `Question`. Add `Batch` Pydantic model. |
| `ucat/db.py`      | New table `batches`; new column `batch_id` on questions; `create_batch`, `update_batch`, `list_batches`, `questions_by_batch`. |
| `ucat/rag.py`     | New `generate_bulk(section, subtype, quantity, on_progress, cancel_event)`. Modify `_system_blocks` to accept optional `subtype`. New `generate_single_dm_question`. |
| `ucat/ui.py`      | New `_t_bulk` frame; new `_tab_bulk()` builder; worker-thread runner; progress callbacks; export trigger. Output tab gains a "Filter by batch" dropdown. |
| `ucat/export.py`  | **New module.** `export_batch_html(batch_id, dest_folder)` writes `index.html` + per-item HTML using existing `rendering` for visuals. |
| `ucat/config.py`  | Add `SUBTYPES` mapping section → list of subtype keys for the dropdown.         |
| `ucat_trainer.py` | No change.                                                                      |

## Error handling

- **Per-item failure** — caught and logged; batch continues. The item is *not* saved to DB. `failed` count increments.
- **Retry policy** — one retry per item, regardless of failure cause (network timeout, 5xx, schema validation, verifier rejection). After the retry also fails, log and skip. This matches the run-log example showing `Q3 ✗ verifier rejected — retrying` then `Q3 ✓`.
- **Cancel mid-item** — current item finishes (so its API spend isn't wasted); loop exits; batch marked `cancelled=1`.
- **App close mid-run** — partial batch survives in DB; `completed_at` stays NULL; the Bulk tab on next launch shows it as "Last batch: incomplete (N saved)" but does not auto-resume.

## Testing

- Unit: `_system_blocks` produces the right subtype constraint for each (section, subtype) pair.
- Unit: `generate_single_dm_question` returns a valid `DMQuestion` with `type == subtype`.
- Unit: `db.create_batch` + `db.questions_by_batch` round-trip.
- Integration: a dry-run mode where the LLM is mocked, so a 5-question batch can be exercised end-to-end (UI → worker → DB → export) without API calls.
- Manual: real 5-question DM-syllogism batch hits Anthropic, lands in DB, exports to a folder, opens in browser correctly.

## Open questions

None — all decisions captured above (PDF deferred, AR single-option, sequential v1, label auto-defaulted).
