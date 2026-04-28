# Bulk Equate Mode + Cost Transparency — Design Spec

**Date:** 2026-04-28
**Status:** Approved (pending review)
**Scope:** Add an "equate" toggle to the Bulk Generate tab so a single quantity input runs the same number of sets across **VR, QR, SJT, and DM** (AR excluded). Round-robin execution. Layer in three cost-transparency improvements: a per-section cost breakdown for equate mode, a live "spent so far" label, and a post-run actual-vs-estimate summary. Replace the hardcoded `$5.00` cost-confirm threshold with a user-configurable persisted setting.

> Builds on `2026-04-25-bulk-generate-section-design.md` and `2026-04-26-subtype-bulk-generate-design.md`. Those specs covered single-section bulk runs and subtype targeting respectively. Equate mode is the next logical layer: balanced practice across the four reasoning sections in one run.

---

## 1. Goal

Today's Bulk Generate runs target one section per run. A student preparing for a full UCAT mock has to launch four separate runs (VR, then QR, then SJT, then DM) to generate a balanced batch of practice material. They have to remember to do all four, balance the counts manually, and click through four cost-confirm dialogs.

Equate mode collapses this into one run: tick a checkbox, enter `25`, click Start, and the worker round-robins through `VR₁, QR₁, SJT₁, DM₁, VR₂, QR₂, SJT₂, DM₂, …` until it has 25 sets per section (100 total). One Stop button, one cost preview, one progress bar, balanced output even if the run is interrupted.

Alongside, three cost-transparency improvements address rough edges that get more visible in equate mode (where a single run is 4× the size and cost of today's typical run):

- **Per-section cost breakdown** in the estimate banner so the user can see what each section contributes.
- **Live spent-so-far** label that updates as sets complete, so the user can pace themselves against the estimate.
- **Post-run summary** showing actual vs estimated cost — useful for budget tracking and for refining future estimates.

And one ergonomics fix: the `$5.00` cost-confirm threshold becomes a user-configurable setting, so power users running large equate batches can raise it (or lower it for tighter control) without code changes.

## 2. Non-goals

- **Custom section selection in equate mode.** Always VR/QR/SJT/DM, never user-selectable. (Earlier alternative: per-section checkboxes — deferred.)
- **Per-section subtype targeting in equate mode.** Equate forces "Any (mixed)" for all four sections. (Earlier alternative: four subtype dropdowns — deferred. Easy to layer on later.)
- **Including AR in equate mode.** AR is pattern-matching, not directly comparable to the four reasoning sections; a balanced UCAT mock typically pairs the reasoning sections together and treats AR as its own drill.
- **Custom execution orders.** Round-robin only, no user-selectable sequential or shuffled mode.
- **Telemetry-driven per-section cost multipliers.** v1 ships with educated-guess multipliers (see §6.1). Wiring `ucat_telemetry.jsonl` into the estimator to derive multipliers from real per-section actuals is a follow-up spec.
- **Auto-approve toggle for the no-RAG warning.** Only the cost-confirm dialog is configurable. The "no indexed docs for this section" warning still fires when relevant — it concerns quality, not money.
- **Changes to the single-shot Generate tab.** Equate mode lives only in the Bulk tab.

## 3. User flow

1. User opens the **Bulk** tab.
2. Above the Section radios, a new checkbox: `☐ Equate across VR/QR/SJT/DM (same qty for each section)`.
3. **Equate off (default):** identical behavior to today. Section radios + Subtype dropdown active. Quantity unit derived from subtype.
4. **Equate on:**
   - Section radios become **disabled** (greyed but retain their last selection — visual cue that they're suspended, not lost).
   - Subtype dropdown row is **hidden** (collapsed; the user's last subtype is preserved in settings and restored when equate is unticked).
   - Yield helper line is hidden (no subtype = no question/set conversion to show).
   - Quantity label is forced to `Sets:`.
   - Cost preview banner expands to show:
     ```
     Estimated cost: ~$3.42 - $4.85   (25 sets × 4 sections × claude-sonnet-4-6)
     ↳ VR ~$0.85  ·  QR ~$1.10  ·  SJT ~$0.78  ·  DM ~$1.12
     ```
5. Below the cost banner, a new `Confirm above: $[ 5.00 ]` numeric entry replaces the hardcoded threshold. Persists across sessions.
6. Click **START BULK RUN**. If `high > threshold`, the confirm dialog fires (wording adjusted to mention equate mode and the four sections).
7. Treeview always shows a **Section** column (visible in both equate and single-section modes — see §4.5).
8. Round-robin execution: row 1 = VR, row 2 = QR, row 3 = SJT, row 4 = DM, row 5 = VR, etc.
9. As each set completes, a new **live spent** label below the cost banner updates:
   ```
   Spent: $1.40 / ~$3.85 est (36%)
   ```
10. On run finish or Stop, the progress label shows a summary line:
    ```
    Done. 100/100 sets. Actual: $3.42 (est $3.85)
    Stopped at 23/100 sets. Actual: $0.78 (est $3.85)
    ```
    Stays visible until next run starts.
11. Untick equate: radios re-enable, subtype row reappears with the user's last selection restored. Cost banner reverts to single-section format.

## 4. UI changes — Bulk tab

Five changes to `_tab_bulk()` in `ucat/ui.py`. Treeview gains one column; everything else (preview pane, Stop button, threading, progress label) stays.

### 4.1 Equate checkbox

Inserted as the first row inside the bulk frame, above the Section radios.

```
☑ Equate across VR/QR/SJT/DM (same qty for each section)

Section:  ( ) VR  (•) DM  ( ) QR  ( ) AR  ( ) SJT     ← disabled when equate is on
Subtype:  [ Any (mixed)                       ▾ ]     ← hidden when equate is on
```

- Implemented as a `tk.Checkbutton` bound to `self._bulk_equate` (`tk.BooleanVar`).
- `command` callback is `self._bulk_equate_changed`, which:
  1. Enables/disables every Section `Radiobutton` (`state="disabled"` / `state="normal"`).
  2. Calls `self._bulk_subtype_frame.pack_forget()` or `self._bulk_subtype_frame.pack(...)` to hide/show the subtype row. (Requires wrapping the subtype row contents in a stored `tk.Frame` handle — current code uses an inline frame; refactor to assign `self._bulk_subtype_frame`.)
  3. Forces the quantity label to `Sets:` and hides the yield helper line.
  4. Calls `self._bulk_inputs_changed()` to refresh the cost banner.
- Persisted via `settings["bulk_equate"]` (default `False`).

### 4.2 Configurable confirm-threshold entry

Inserted on a new row immediately below the cost banner.

```
Estimated cost: ~$3.42 - $4.85   (25 sets × 4 sections × claude-sonnet-4-6)
↳ VR ~$0.85  ·  QR ~$1.10  ·  SJT ~$0.78  ·  DM ~$1.12

Confirm above: $[ 5.00 ]      ← NEW row
```

- Implemented as `tk.Entry` bound to `self._bulk_confirm_threshold` (`tk.StringVar`), width 6, parsed as float.
- `trace_add("write", ...)` validates and persists. Invalid input (non-numeric, negative) reverts to last good value on focus-out; the entry is shown with a `WARN` border briefly.
- Default `5.00` (matches today's hardcoded constant). Set high (e.g. `999`) to disable the dialog; set `0` to confirm every run.
- Persisted via `settings["bulk_cost_confirm_threshold"]`.

### 4.3 Cost banner — per-section breakdown

The existing single-line banner stays for single-section mode. In equate mode, a second line appears beneath it:

```
Estimated cost: ~$3.42 - $4.85   (25 sets × 4 sections × claude-sonnet-4-6)
↳ VR ~$0.85  ·  QR ~$1.10  ·  SJT ~$0.78  ·  DM ~$1.12
```

- Second-line text is built by `_format_section_breakdown(n_sets, llm, multi_judge, verify)` which calls `estimate_section_cost(section, n_sets, ...)` for each of the four sections and concatenates results.
- Per-section costs use `SECTION_COST_MULTIPLIERS` (see §6.1) applied to the existing single-shape estimator. Sum of per-section figures equals total within rounding (within ±1 cent).
- Hidden (`pack_forget()`) when equate is off.

### 4.4 Live spent + post-run summary

Two display elements, shown in both single-section and equate modes (since the underlying `_bulk_run_cost` accumulator already exists):

- **Live spent label** (`self._bulk_live_spent_lbl`): rendered below the cost banner during a run. Format: `Spent: $1.40 / ~$3.85 est (36%)`. Updated by `_bulk_set_row()` (or a new `_bulk_update_spent()` helper called from there) after each successful set's verdict cell is written. Hidden when no run is active.
- **Post-run summary** (the existing `self._bulk_progress_lbl`): on run completion or stop, replaces the in-progress text with the summary line per §3 step 10. The label already exists; we're just changing the final-state text. Stays visible until `_bulk_seed_rows()` clears it on the next run start.

### 4.5 Treeview Section column

The existing `cols` tuple becomes:

```python
cols = ("#", "Started", "Section", "Subtype", "Status", "Verdict", "Cost", "Difficulty")
widths = (44, 90,        70,        110,       180,      100,       70,     80)
```

- Always present (single-section mode shows the same VR/QR/SJT/DM/AR code on every row — informational, not noisy).
- Value populated from the per-row `section` field (new — see §5.4).
- Anchor `center`. Width 70px.

## 5. Worker / orchestration

### 5.1 New constants

In `ucat/config.py`:

```python
EQUATE_SECTIONS: tuple[str, ...] = ("VR", "QR", "SJT", "DM")
```

Order matters — it's the round-robin order. Chosen to interleave the heavier (QR with charts) and lighter (SJT scenarios) sections, but the exact order is cosmetic.

### 5.2 New helper — `equate_task_list`

In `ucat/config.py`, alongside `compute_set_count`:

```python
def equate_task_list(n: int) -> list[str]:
    """Round-robin section list for an equate run.

    n=0 → []
    n=1 → ["VR", "QR", "SJT", "DM"]
    n=3 → ["VR", "QR", "SJT", "DM", "VR", "QR", "SJT", "DM", "VR", "QR", "SJT", "DM"]
    """
    if n <= 0:
        return []
    return list(EQUATE_SECTIONS) * n
```

Pure function, no UI dependencies, easy to unit-test.

### 5.3 `_bulk_start` changes

Two branches:

```python
if self._bulk_equate.get():
    n_input = int(raw)                                 # interpreted as sets per section
    n_per_section = min(n_input, BULK_MAX_QUANTITY)    # cap per-section
    if n_per_section < 1:
        return
    task_list = equate_task_list(n_per_section)        # length 4 * n_per_section
    subtype = None
    # Indexed-docs check applies to every section in EQUATE_SECTIONS
    empty = [s for s in EQUATE_SECTIONS
             if self.db.count(s, indexed_only=True) == 0]
    if empty and not messagebox.askyesno(
        "No Indexed Documents",
        f"No indexed documents for: {', '.join(empty)}.\n\n"
        "Add docs and click Index, or generate without RAG context?",
    ):
        return
else:
    # Existing single-section path (unchanged).
    n = compute_set_count(n_input, section, subtype)
    n = min(n, BULK_MAX_QUANTITY)
    task_list = [section] * n
```

The cost-confirm dialog now reads `self._bulk_confirm_threshold` instead of `BULK_COST_CONFIRM_THRESHOLD`:

```python
threshold = float(self.settings.get("bulk_cost_confirm_threshold") or
                  BULK_COST_CONFIRM_THRESHOLD)
if high > threshold:
    if not messagebox.askyesno("Confirm bulk run", _confirm_dialog_text(...)):
        return
```

Both branches converge by passing `task_list` (and `subtype`) to the worker thread.

### 5.4 `_bulk_worker` changes

Worker signature changes from `(section, hint, n, subtype)` to `(task_list, hint, subtype)`:

```python
def _bulk_worker(self, task_list: list[str], hint: str,
                 subtype: Optional[str]) -> None:
    n_total = len(task_list)
    for idx, section in enumerate(task_list, start=1):
        if self._bulk_stop.is_set():
            self._bulk_mark_remaining_skipped(idx, n_total)
            break
        # … existing per-set generation, telemetry, treeview update …
        result = self.rag.generate(section, hint, subtype=subtype, …)
```

Per-row state in `_bulk_rows` gains a `section` field (set from `task_list[idx-1]` in `_bulk_seed_rows`). The Treeview Section column reads from this field. In single-section mode, all rows have the same value. In equate mode, they round-robin.

### 5.5 Cap behavior

`BULK_MAX_QUANTITY = 100` is unchanged. Interpretation in equate mode: cap is **per-section**, not total. Input `100` in equate → 400 tasks. Input `101` → caps the input at 100 and shows a cap note in the cost banner:

```
Estimated cost: ~$13.42 - $19.85   (100 sets × 4 sections × claude-sonnet-4-6)
↳ VR ~$3.40  ·  QR ~$4.40  ·  SJT ~$3.10  ·  DM ~$4.50
                          (capped at 100 per section — split into multiple runs for more)
```

### 5.6 Stop semantics

Round-robin guarantees a Stop produces a balanced sample (worst-case ±1 set across the four sections). No special bookkeeping — the existing "skip remaining" loop already iterates `task_list` and marks queued rows as `skipped`.

## 6. Cost estimator changes

### 6.1 Per-section breakdown — multiplier table

Add to `ucat/config.py`:

```python
# Per-section cost multipliers, normalised to mean 1.0 across EQUATE_SECTIONS.
# These are educated guesses based on prompt structure (QR carries chart specs;
# SJT scenarios are short; VR has the longest passage; DM is mid-weight).
# Refine via telemetry in a follow-up spec.
SECTION_COST_MULTIPLIERS: dict[str, float] = {
    "VR":  0.95,   # passage + 4 questions
    "QR":  1.20,   # chart spec adds output tokens
    "SJT": 0.85,   # shorter scenario, 4 Likert items
    "DM":  1.00,   # baseline (5 standalone items)
    "AR":  1.10,   # only used if equate ever extends to AR
}
```

Sum across `EQUATE_SECTIONS` is `0.95 + 1.20 + 0.85 + 1.00 = 4.00` — exactly 4× the baseline, so total cost is unchanged from today's flat estimate (it just gets split four ways instead of multiplied by 4 evenly).

New helper `estimate_section_cost(section, n_sets, llm, *, multi_judge, verify)` wraps `estimate_bulk_cost` and applies the multiplier:

```python
def estimate_section_cost(section, n_sets, llm, *, multi_judge, verify):
    low, high = estimate_bulk_cost(n_sets, llm,
                                   multi_judge=multi_judge, verify=verify)
    m = SECTION_COST_MULTIPLIERS.get(section, 1.0)
    return low * m, high * m
```

### 6.2 Live spent label

The `_bulk_run_cost: float` accumulator already exists and increments after each set's verdict is written. Add a new label `self._bulk_live_spent_lbl` packed below the cost banner. New helper:

```python
def _bulk_update_spent(self):
    spent = self._bulk_run_cost
    est_high = self._bulk_last_estimate_high or 0.0
    pct = int(round(100 * spent / est_high)) if est_high > 0 else 0
    self._bulk_live_spent_lbl.config(
        text=f"Spent: ${spent:.2f} / ~${est_high:.2f} est ({pct}%)",
    )
```

Called from `_bulk_set_row()` whenever a row's `cost` cell is written. Hidden (`text=""`) when `_bulk_thread` is None or finished.

`self._bulk_last_estimate_high` is set in `_bulk_start()` from the `high` value of the cost-preview computation, so the live percentage compares against the upper-bound estimate the user actually saw.

### 6.3 Post-run summary

In `_bulk_run_finished` (the existing main-thread callback when the worker thread exits), set the existing `self._bulk_progress_lbl` to:

```python
if stopped:
    text = f"Stopped at {done_count}/{n_total} sets. Actual: ${spent:.2f} (est ${est_high:.2f})"
else:
    text = f"Done. {done_count}/{n_total} sets. Actual: ${spent:.2f} (est ${est_high:.2f})"
self._bulk_progress_lbl.config(text=text)
```

Stays until the next run starts (when `_bulk_seed_rows` clears it).

The live-spent label is hidden in this same callback (since the run is over and the summary line carries the actual figure).

### 6.4 Configurable confirm threshold

Already covered in §4.2 (UI) and §5.3 (worker). Default value matches today's `BULK_COST_CONFIRM_THRESHOLD = 5.00` constant; the constant becomes the seed for the new persisted setting and is no longer read directly by `_bulk_start`.

## 7. Settings additions

Two new keys in `Settings.DEFAULTS`:

```python
"bulk_equate":                    False,    # equate mode toggle
"bulk_cost_confirm_threshold":    5.00,     # USD; replaces hardcoded constant
```

Persistence flow:

- On equate checkbox toggle: write `bulk_equate`.
- On confirm-threshold entry change (validated): write `bulk_cost_confirm_threshold`.
- Existing `bulk_subtype` and `bulk_subtype_by_section` are preserved while equate is on, untouched. When equate is unticked, the dropdown reads from `bulk_subtype_by_section[current_section]` exactly as today.

## 8. Telemetry

Existing `bulk_run_start` and `bulk_run_end` events gain new fields:

- `bulk_run_start`: add `equate_mode: bool`, `task_list_length: int`.
- `bulk_run_end`: add `equate_mode: bool`, `actual_cost: float`, `estimated_cost_high: float`.
- Per-row `rag_generate` traces continue to emit `section` per call — already happens, no change.

No new event types; `telemetry.emit()` accepts arbitrary fields.

## 9. Edge cases & guardrails

- **Equate + topic hint together.** Allowed. The same hint is forwarded to all four sections' generations. No validation; some hints will be more relevant to some sections than others, which is fine — the user is responsible for choosing a hint that fits.
- **Equate on, then user changes Section radio.** Impossible — radios are disabled. No code path needed.
- **Equate on, then user changes model / verify / jury.** Cost banner refreshes via the existing `_bulk_inputs_changed`. Per-section breakdown recomputes.
- **Empty KB for one or more equate sections.** Single batched dialog: `"No indexed documents for: VR, QR. Continue without RAG context?"` Yes proceeds; No aborts. (Today's per-section dialog would fire four times; equate batches it.)
- **Cap with equate.** Per-section cap of 100 (so 400 total tasks max). Cap note in cost banner: "(capped at 100 per section — split into multiple runs for more)".
- **Confirm-threshold misuse.** Setting threshold to a negative number reverts on focus-out. Setting to `0` confirms every run. Setting very large (e.g. `999`) effectively silences the dialog — intended use.
- **Stop mid-run with equate.** Round-robin invariant: if `idx` sets completed, the per-section counts are within 1 of each other. Worst-case imbalance: 1 set across the four sections.
- **Indexed-docs prompt declined.** User sees the batched dialog, clicks No → `_bulk_start` returns without launching the worker. No partial state.
- **App close mid-run.** Unchanged (`on_close` sets `_bulk_stop`, joins thread with 5s timeout). Live spent label and summary line are not persisted across app restarts.
- **Per-section breakdown rounding.** Per-section figures may not sum to the total figure exactly because each is rounded to 2 decimal places. Acceptable; the discrepancy is at most $0.04. Documented in code comment.
- **Subtype drift in equate mode.** Cannot occur — `subtype=None` is forced. Drift detection in `rag.generate()` skips when subtype is None (already does).
- **Cost preview when input invalid.** Same as today: banner shows `"Enter a number 1 - 100."` (in equate mode, the message reflects the per-section cap, not 4× it).

## 10. Files touched

- **`ucat/config.py`**
  - Add `EQUATE_SECTIONS` tuple.
  - Add `SECTION_COST_MULTIPLIERS` dict.
  - Add `equate_task_list(n)` function.
  - Add `estimate_section_cost(section, n_sets, llm, …)` function.
  - Add two new keys to `Settings.DEFAULTS`: `bulk_equate`, `bulk_cost_confirm_threshold`.

- **`ucat/ui.py`**
  - `_tab_bulk()` adds the equate checkbox row, the confirm-threshold entry, the live spent label. Wraps the subtype row in a stored frame handle for show/hide.
  - New `_bulk_equate_changed()` callback handles radio enable/disable, subtype frame show/hide, quantity label flip.
  - `_bulk_inputs_changed()` reads `_bulk_equate`, swaps cost-preview format (single-line vs two-line with breakdown), persists threshold setting.
  - `_bulk_start()` branches on `_bulk_equate`, builds the appropriate `task_list`, batches the no-RAG dialog when equate is on, reads threshold from settings.
  - `_bulk_worker()` signature changes to `(task_list, hint, subtype)`; loop iterates `task_list`.
  - `_bulk_seed_rows()` populates a `section` field per row; Treeview gets the new Section column always.
  - `_bulk_set_row()` populates the new Section cell and calls `_bulk_update_spent()` after writing a cost.
  - `_bulk_run_finished()` writes the summary line and hides the live-spent label.
  - Treeview `cols` and `widths` tuples updated.

- **`ucat/telemetry.py`** — *no changes.* Extra fields ride existing `emit()`.

- **`ucat/rag.py`** — *no changes.* Equate mode is a UI/orchestration feature; per-call generation is unchanged (always `subtype=None`).

- **`tests/test_equate_mode.py`** — *new file.* Unit tests for `equate_task_list`, `estimate_section_cost` multiplier math, and settings defaults.

## 11. Testing notes

- **Helper math:**
  - `equate_task_list(0)` → `[]`.
  - `equate_task_list(1)` → `["VR", "QR", "SJT", "DM"]`.
  - `equate_task_list(3)` → 12 entries, exact round-robin order, three of each section.
  - `equate_task_list(-1)` → `[]`.

- **Cap math:**
  - Input `100` in equate mode → `task_list` length 400, no truncation.
  - Input `101` in equate mode → input clamped to 100 → `task_list` length 400, banner shows cap note.
  - Input `0` or negative → no run, banner shows input-error message.

- **Cost math:**
  - `estimate_section_cost("VR", 10, "claude-sonnet-4-6", multi_judge=False, verify=False)` ≈ `0.95 ×` baseline 10-set cost.
  - Sum of per-section costs across `EQUATE_SECTIONS` for the same `n_sets` equals total (within $0.04).

- **UI state — equate toggle:**
  - Tick equate → all five Section radios become `state="disabled"`. Subtype row removed from layout. Quantity label reads `Sets:`. Yield helper hidden.
  - Untick equate → radios restored (last selection retained). Subtype row repacked. Quantity label flips to whatever `bulk_subtype_by_section[section]` implies. Yield helper restored if subtype is set.
  - Restart app with `bulk_equate=True` saved → tab loads in equate state.

- **UI state — confirm threshold:**
  - Default value loaded as `5.00` for fresh users.
  - Change to `10.00`, restart app → loads as `10.00`.
  - Change to `-1` and tab away → reverts to last good value.
  - Change to `999`, set quantity high enough to exceed historical $5 threshold → no dialog, run proceeds.
  - Change to `0`, set quantity to 1 → dialog still fires for any non-zero estimate.

- **Cost preview — equate mode:**
  - Tick equate, set quantity to 25, model `claude-sonnet-4-6`, no jury, no verify → banner shows two lines, total figure matches `4 × per-set × 25`, breakdown shows VR/QR/SJT/DM split summing to total within rounding.
  - Toggle verify on → both lines update; per-section figures shift up consistently with multipliers.
  - Toggle equate off → banner reverts to single line.

- **Worker — round-robin execution:**
  - Equate, quantity 3 → Treeview seeds 12 rows. Section column reads VR, QR, SJT, DM, VR, QR, SJT, DM, VR, QR, SJT, DM in order.
  - During run, rows complete in that order (modulo per-set timing variation).
  - Press Stop after row 6 completes → rows 7–12 marked `skipped`. Actual completed: 1×VR, 1×QR, 1×SJT, 1×DM from cycle 1, plus 1×VR, 1×QR from cycle 2 → 2×VR, 2×QR, 1×SJT, 1×DM. Imbalance ≤ 1, as designed.

- **Worker — single-section unchanged:**
  - Equate off, pick QR, subtype `bar`, quantity 8 → existing flow: 2 sets generated, all bar charts. No regression from the worker refactor.

- **Live spent label:**
  - Run starts → label appears with `Spent: $0.00 / ~$X.XX est (0%)`.
  - After first set completes → label updates with the actual cost from telemetry.
  - Run finishes → live label is hidden; progress label shows summary.

- **Post-run summary:**
  - Run completes normally → progress label reads `Done. N/N sets. Actual: $X.XX (est $Y.YY)`.
  - Run is stopped mid-way → progress label reads `Stopped at K/N sets. Actual: $X.XX (est $Y.YY)`.
  - Start a new run → summary label cleared in `_bulk_seed_rows`.

- **No-RAG batched dialog:**
  - Equate on, only DM has indexed docs → dialog reads `"No indexed documents for: VR, QR, SJT."` Yes proceeds; No aborts.
  - All four sections have indexed docs → no dialog.

- **Cost-confirm with custom threshold:**
  - Threshold `0.50`, estimate `$0.30` → no dialog.
  - Threshold `0.50`, estimate `$0.80` → dialog fires; equate-mode wording mentions four sections.

- **Telemetry:**
  - `bulk_run_start` event includes `equate_mode: true`, `task_list_length: 400` for a 100-input equate run.
  - `bulk_run_end` event includes `actual_cost` (sum of per-set actuals) and `estimated_cost_high` (the figure shown in the banner at run start).

- **Backwards compat:**
  - Existing settings file without `bulk_equate` or `bulk_cost_confirm_threshold` keys → both fall back to defaults (`False`, `5.00`).
  - Existing single-section runs unaffected.

---

End of spec.
