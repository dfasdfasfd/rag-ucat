# Bulk Generate (Section-level) — Design Spec

**Date:** 2026-04-25
**Status:** Approved (pending review)
**Scope:** Add a Bulk Generate feature to the UCAT Trainer that produces N question sets sequentially for a chosen section, with progress tracking, retry-on-error, cost preview, and a Stop control.

> Supersedes the earlier draft at `2026-04-25-bulk-generate-design.md`. That draft scoped subtype filtering, a `batches` DB table, single-DM-question generation, and HTML export — all explicitly cut from this design (see §2).

---

## 1. Goal

Today, generating questions is a one-shot loop: pick a section → click Generate → review → save. Producing a meaningful KB or practice batch means clicking Generate dozens of times. Bulk Generate replaces that grind with a single launch that produces N sets unattended.

## 2. Non-goals

- **Sub-section / subtype targeting.** Section level only (VR / DM / QR / AR). DM subtypes (syllogism, venn, etc.) remain mixed within a set as today.
- **Parallel generation.** Sequential only. Anthropic rate limits and UX complexity make parallelism a poor fit for v1.
- **Auto-promote to KB.** Bulk runs deposit results in History only. Promotion to KB stays a manual decision per set, mirroring single-shot.
- **Resumable runs across app restarts.** A run lives only for the duration of the app session. Closing the app mid-run is a hard stop.
- **Bulk runs as first-class DB entities.** No `batches` table. The History tab is already the canonical record of every generation.
- **HTML / PDF export.** Out of scope. The existing `Export Output` button covers JSON.
- **A new `RAGEngine.generate_bulk()` or `generate_single_dm_question()`.** The existing `RAGEngine.generate()` is reused unchanged.

## 3. User flow

1. User clicks the new `BULK` tab.
2. Picks a section, enters a quantity (1-100), optionally a topic hint.
3. The cost preview banner updates live: `"Estimated cost: ~$1.20 - $3.00 (10 sets × Claude Opus 4.7)"`.
4. Clicks `START BULK RUN`.
5. If estimated cost (high end) exceeds `$5.00`, a confirmation modal appears. Otherwise it launches immediately.
6. Per-set rows update through `queued → running → done | failed` states.
7. User can browse other tabs; History updates as each set completes.
8. User can click `STOP` to halt after the current set finishes.
9. On completion, summary shows in status bar: `"Bulk run finished: 8 succeeded, 2 failed."`

## 4. UI — Bulk tab

New tab in the notebook, between `GENERATE` and `KNOWLEDGE BASE`.

### Layout

```
┌────────────────────────────────────────────────────────────────┐
│ Bulk Generate                                                  │
│ Generate multiple question sets in sequence. Results land in   │
│ History — promote good ones to the KB from there.              │
│                                                                │
│ Section:  ( ) VR  (•) DM  ( ) QR  ( ) AR                       │
│ Quantity: [  10  ]   Topic hint: [____________________________]│
│                                                                │
│ Estimated cost: ~$1.20 - $3.00   (10 sets × Claude Opus 4.7)   │
│                                                                │
│ [ ⚡  START BULK RUN ]   [ ⏹  STOP ] (disabled until running) │
│ Progress: 3 / 10 — generating set 4…                           │
│                                                                │
│ ┌──────────────────────────────────────────────────────────┐  │
│ │ #  Started   Status     Verdict   Cost     Difficulty    │  │
│ │ 1  14:02:11  ✓ done     ✓         $0.12    3.2           │  │
│ │ 2  14:02:48  ✓ done     ⚠ 1 fl    $0.14    2.9           │  │
│ │ 3  14:03:31  ✗ failed   —         —        —             │  │
│ │ 4  14:04:05  ⟳ running  …         …        …             │  │
│ │ 5  …         queued                                       │  │
│ └──────────────────────────────────────────────────────────┘  │
│                                                                │
│ Selected set preview (read-only view of the chosen row)        │
└────────────────────────────────────────────────────────────────┘
```

### Inputs

- **Section** — radio buttons matching the Generate tab style. Default from `settings["bulk_section"]`.
- **Quantity** — integer entry, range 1-100. Default from `settings["bulk_quantity"]` (= 10 first run). Out-of-range / non-numeric input disables the Start button with an inline hint.
- **Topic hint** — free-text entry, optional. Default from `settings["bulk_hint"]`.

All three persist to `Settings` on change via the existing `set()` pattern.

### Cost preview banner

Always visible. Recomputes on any change to section, quantity, or LLM dropdown (sidebar). Shows a low-high range (cache-hit vs cold).

### Buttons

- **START BULK RUN** — primary action. Disabled while a bulk run is in progress, while a single-shot generate is in progress, or while quantity is invalid.
- **STOP** — disabled until a run is in progress. Click sets a `threading.Event`; the loop exits after the current iteration. Status bar reads "Stopping after current set…" then "Stopped at 4 / 10."

### Treeview

Columns: `#`, `Started`, `Status`, `Verdict`, `Cost`, `Difficulty`. Same visual style as the History tab.

### Preview pane

Below the Treeview. Selecting a row renders the set with `format_qset(result["data"])`. Failed rows show the error message instead.

## 5. Threading & control flow

One worker thread per bulk run.

### State on `App`

```python
self._bulk_stop: threading.Event   = threading.Event()
self._bulk_thread: Optional[threading.Thread] = None
self._bulk_rows: List[Dict[str, Any]] = []   # mirror of Treeview rows by index
```

### Worker loop (pseudocode)

```python
def _bulk_worker(self, section: str, hint: str, n: int):
    self.after(0, lambda: self._bulk_run_started(n))
    succeeded = 0
    failed    = 0
    for i in range(1, n + 1):
        if self._bulk_stop.is_set():
            self.after(0, lambda idx=i: self._mark_remaining_skipped(idx))
            break

        self.after(0, lambda idx=i: self._mark_row(idx, status="running"))

        attempts = 0
        while attempts < 2:
            try:
                result = self.rag.generate(
                    section, hint,
                    on_progress=lambda m, idx=i: self.after(0,
                        lambda: self._mark_row(idx, progress=m)),
                    on_delta=None,
                    variation_seed=str(uuid.uuid4())[:8],
                )
                self.after(0, lambda idx=i, r=result: self._mark_row(idx, status="done", result=r))
                succeeded += 1
                break
            except Exception as e:
                attempts += 1
                if attempts >= 2:
                    logger.exception(f"Bulk set {i} failed twice")
                    err = str(e)
                    self.after(0, lambda idx=i, msg=err: self._mark_row(idx, status="failed", error=msg))
                    failed += 1
                else:
                    time.sleep(1.0)

    self.after(0, lambda: self._bulk_run_finished(succeeded, failed))
```

### Cancellation semantics

- Stop is **immediate on click** (no confirmation modal).
- The currently-in-flight set finishes (or errors out and retries). We don't kill the API call — it's already paid for.
- Remaining queued rows transition to status `· skipped`.

### Retry policy

- Any exception triggers one retry after 1.0s backoff.
- Retry hits the full pipeline (retrieve → generate → verify); transient failures could be on any step.
- After two attempts, the set is recorded as `failed`, the loop continues.

### What is *not* retried

- A successful generation with `verdict.overall_correct = false` (verified-but-flagged) is **success**, not failure. The user reviews flagged sets in the preview pane / History.
- Stop pressed mid-iteration does not cancel the current API call.

### No streaming in bulk

`on_delta=None` is passed to `rag.generate()`. Streaming 50 sets to a single widget would be noise; per-row progress text from `on_progress` is sufficient.

### Variation seed

Every iteration receives a unique `variation_seed = uuid4()[:8]` so the LLM doesn't return near-identical sets within one run. This reuses the existing single-shot mechanism (it's how Regenerate works).

## 6. Cost preview

### Constants

In `ucat/config.py`:

```python
BULK_MAX_QUANTITY = 100
BULK_COST_CONFIRM_THRESHOLD = 5.00  # USD
```

### Estimator

```python
def estimate_bulk_cost(n: int, llm: str, multi_judge: bool, verify: bool) -> tuple[float, float]:
    """
    Returns (low, high) USD estimate.

    Per-set token assumptions (from telemetry observations):
      gen:    ~3000 input + ~2000 output  (input mostly cached after the first set)
      verify: ~1500 input + ~600 output   (Haiku, single-judge)
      jury:   +Sonnet (~1500 in + ~600 out) and +Opus second-pass (similar)
    """
    costs  = MODEL_COSTS[llm]
    gen_low  = (3000 * costs["cache_read"] + 2000 * costs["out"]) / 1_000_000
    gen_high = (3000 * costs["in"]         + 2000 * costs["out"]) / 1_000_000
    per_low, per_high = gen_low, gen_high

    if verify:
        haiku = MODEL_COSTS["claude-haiku-4-5"]
        verify_per = (1500 * haiku["in"] + 600 * haiku["out"]) / 1_000_000
        per_low  += verify_per
        per_high += verify_per

    if multi_judge:
        sonnet = MODEL_COSTS["claude-sonnet-4-6"]
        opus   = MODEL_COSTS["claude-opus-4-7"]
        # Jury costs are roughly cache-insensitive — add the same amount to both ends.
        jury_per = ((1500 * sonnet["in"] + 600 * sonnet["out"]) / 1_000_000
                  + (1500 * opus["in"]   + 600 * opus["out"])   / 1_000_000)
        per_low  += jury_per
        per_high += jury_per

    return n * per_low, n * per_high
```

### Confirmation modal

Triggered when `estimate_bulk_cost(n, ...)[1] > BULK_COST_CONFIRM_THRESHOLD`:

```
Confirm bulk run

Estimated cost: $1.20 - $3.50
Sets: 10
Section: Decision Making
Model: claude-opus-4-7

Continue?
```

Implemented with `messagebox.askyesno`. Below threshold = launches without modal.

## 7. Result handling & persistence

### Where each set goes

| Outcome              | Goes to                                    |
|----------------------|--------------------------------------------|
| Successful set       | `generated` table (via existing `db.add_generated`), Treeview row, in-memory `_bulk_rows` |
| Verdict-flagged set  | Same as successful — `overall_correct=false` is a real result, surfaced via the verdict badge |
| Failed set (2 attempts) | `_bulk_rows` only; no DB row, no telemetry beyond aggregate counts |
| Skipped set (Stop)   | `_bulk_rows` only; row shows `· skipped`   |

### History tab integration

Successful sets appear in the History tab automatically because `rag.generate()` already calls `db.add_generated()` at its tail. After each successful set, schedule the same refresh trio the single-shot flow uses:

```python
self.after(0, lambda: (
    self._refresh_stats(),
    self._refresh_out(),
    self._refresh_insights(),
))
```

### Row → result mapping

Each `_bulk_rows` entry holds the full `result` dict returned by `rag.generate()` (data, retrieved, usage, verdict, coverage, difficulty, dup_warning). Selecting a Treeview row looks up the matching entry by index and renders the preview via `format_qset(result["data"])`. The DB-assigned `generated.id` is *not* tracked on the row — `rag.generate()` doesn't return it today and bulk doesn't need it (the History tab queries the table independently).

### Cost meter

The header `_cost_lbl` already accumulates session totals. Bulk just adds to it via the same `_session_cost` / `_session_tokens` updates that `_gen_ok` performs, called from `_mark_row(status="done")`.

## 8. Edge cases & guardrails

### Input validation

- **Quantity** — integer 1-100. Below 1 = Start disabled. Above 100 = capped with status-bar message: "Capped at 100 to limit cost — split into multiple runs for more." Non-integer / blank = Start disabled with inline hint.

### Empty KB

Before launch, if `db.count(section, indexed_only=True) == 0`, the existing "No Indexed Documents" `askyesno` dialog appears (same wording as the single-shot path). If the user confirms, the run proceeds with empty retrieval context.

### API key check

`api_status()` is called before launch; missing keys → existing error dialog, no run.

### Concurrent runs

- Bulk-while-bulk: blocked. Start button disabled while `_bulk_thread.is_alive()`.
- Bulk-while-single-shot: blocked. The single-shot Generate button checks `_bulk_thread`; the Bulk Start button checks `_gbtn["state"]`. Conflict shows status-bar message "A run is already in progress — wait or stop it first." No modal.

### Tab navigation during a run

Allowed. Switching tabs doesn't reset bulk progress. History auto-refreshes after each set.

### App close mid-run

`on_close` calls `self._bulk_stop.set()` and joins the worker thread with a 5s timeout before destroying the window. Database closes after the join.

## 9. Settings additions

Three new keys in `Settings.DEFAULTS`:

```python
"bulk_section":  "VR",
"bulk_quantity": 10,
"bulk_hint":     "",
```

Saved via the existing `set()` pattern when the user changes the corresponding inputs.

## 10. Telemetry

Two new event types via `telemetry.emit()`:

- `bulk_run_start` — fields: `section`, `n`, `model`, `verify`, `multi_judge`, `estimated_cost_high`.
- `bulk_run_end` — fields: `section`, `n`, `succeeded`, `failed`, `stopped`, `actual_cost_usd`, `duration_s`.

Each underlying `rag.generate()` call already emits its own pipeline events; bulk doesn't double-emit them.

## 11. Files touched

- **`ucat/ui.py`** — new `_tab_bulk()`, `_bulk_*` worker methods, registered in `_ui()` notebook. New imports: `uuid`, `time`.
- **`ucat/config.py`** — `BULK_MAX_QUANTITY`, `BULK_COST_CONFIRM_THRESHOLD`, `Settings.DEFAULTS` keys, `estimate_bulk_cost()` function.
- **`ucat/rag.py`** — *no changes*. The single-shot pipeline is reused as is.
- **`ucat/telemetry.py`** — *no changes*. `emit()` already accepts arbitrary event types.

## 12. Testing notes

- Smoke: launch with quantity=2, confirm both sets land in History and the Treeview shows two `✓ done` rows.
- Stop mid-run: launch with quantity=5, click Stop after first row completes; remaining 4 should show `· skipped`, the Stopping → Stopped transition should display in the status bar.
- Retry: simulate a transient failure (e.g. temporarily revoke API key after launch, then restore) — the failing set should retry once, then either succeed or be marked `✗ failed`.
- Cost confirm: set quantity such that `estimate × 1.1 > $5`; modal must appear. Below threshold: must not appear.
- Concurrent block: start single-shot generate, then try bulk Start — must be blocked with status-bar message.
- App close mid-run: launch quantity=10, close the app after row 2 completes — process should exit cleanly within 5s.
- Empty KB: pick a section with zero indexed docs, click Start — existing askyesno appears; Yes proceeds with empty retrieval, No aborts.

---

End of spec.
