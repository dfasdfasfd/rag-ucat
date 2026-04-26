# Subtype-Targeted Bulk Generation — Design Spec

**Date:** 2026-04-26
**Status:** Approved (pending review)
**Scope:** Extend the existing Bulk Generate tab so users can target a specific question subtype within a section (e.g. "10 venn diagrams from DM"). Generation produces homogeneous sets in which every question in a set is of the chosen subtype. AR is excluded from subtype targeting.

> Builds on `2026-04-25-bulk-generate-section-design.md`. That spec explicitly de-scoped subtype filtering; this spec brings it back as a follow-up.

---

## 1. Goal

Today's Bulk Generate runs at the section level: pick `DM` and you get sets of 5 mixed-subtype questions (one syllogism, one venn, one logical, etc.). Students drilling a specific subtype have no targeted practice path — they generate full mixed sets and ignore four-fifths of each one.

Subtype targeting lets the user pick (for example) `DM → Venn → 10 questions` and get exactly that: 2 sets of 5 venn questions each. The same flow works for VR (True/False/Can't Tell vs Multiple choice) and QR (table, bar, line, stacked bar, pie). AR is unchanged.

## 2. Non-goals

- **Mixed-subtype quotas within a single set.** A set is either all-of-one-subtype or fully mixed (today's behavior). No "3 venns + 2 syllogisms in this set" mode.
- **Standalone single-question generation.** A new pipeline that produces flat lists of individual questions without the set wrapper. Out of scope; sets remain the atomic unit.
- **Subtype targeting for AR.** AR currently has only Type 1 in the schema. Adding Type 2/3/4 is a separate spec; for now the AR subtype dropdown is disabled.
- **Subtype targeting in the single-shot Generate tab.** The plumbing in `rag.generate()` will accept a `subtype` kwarg, but the Generate tab UI is unchanged. Adding it there is a follow-up.
- **Per-subtype KB coverage gating.** No new check on whether the KB has indexed examples of the chosen subtype. Today's "no indexed docs for this section" check is sufficient.
- **Trimming overflow questions.** If the user asks for 7 questions and a set has 5, we generate 2 full sets (10 questions), not 7 with 3 trimmed. Sets stay whole.
- **Richer VR subtypes (main idea, paraphrase, tone & purpose, inference).** The `ucat/models.py` Question schema does not have a `minigame_kind` field today; only `Question.type` (which carries `tf` vs `mc` for VR). Adding the richer 5-way VR breakdown requires a schema change to Question, prompt-side support for emitting `minigame_kind`, and updates to verification/format/db. That's a separate spec. This spec restricts VR to the two subtypes the current schema can enforce: `tf` and `mc`.

## 3. User flow

1. User opens the **Bulk** tab, picks a section.
2. The new **Subtype** dropdown populates with that section's subtypes plus "Any (mixed)" at the top. Default selection is "Any" (or the user's last choice for that section, restored from settings).
3. The **Quantity** label flips: `Sets:` when subtype is Any, `Questions:` when a specific subtype is selected.
4. A helper line appears below the quantity input when a subtype is selected: `→ 2 sets × 5 questions = 10 questions`.
5. Cost preview banner updates as before, but always expressed in sets (because that's what's billed): `Estimated cost: ~$0.24 - $0.60   (2 sets × Claude Opus 4.7)`.
6. Click `START BULK RUN`. Cost-confirmation modal still triggers if `high > $5.00`.
7. Treeview adds a new `Subtype` column showing the human-readable subtype name (or `—` for Any).
8. Per-set generation is unchanged; results land in History exactly as today.
9. If the LLM fails to honor the subtype request on a set (subtype drift), the row's `Verdict` column shows `⚠ drift` instead of the usual verdict badge. The set still counts as a success — drift is a warning, not a failure.
10. Final status bar message includes drift count: `"Bulk run finished: 8 succeeded (2 with subtype drift), 0 failed."`

## 4. Subtype catalogue

```python
# In ucat/config.py

SET_SIZES = {"VR": 4, "DM": 5, "QR": 4, "AR": 5}

# Each entry is (storage_value, human_label).
# storage_value matches the field already used by the schema:
#   - DM → Question.type
#   - VR → minigame_kind (with type=tf for "tfc", type=mc for the rest)
#   - QR → QRChart.type
SUBTYPES_BY_SECTION = {
    "DM": [
        ("syllogism",    "Syllogism"),
        ("logical",      "Logical (logic puzzles)"),
        ("venn",         "Venn"),
        ("probability",  "Probability"),
        ("argument",     "Argument"),
    ],
    "VR": [
        ("tf",           "True / False / Can't Tell"),
        ("mc",           "Multiple choice (A-D)"),
    ],
    "QR": [
        ("table",        "Table"),
        ("bar",          "Bar chart"),
        ("line",         "Line chart"),
        ("stacked_bar",  "Stacked bar"),
        ("pie",          "Pie"),
    ],
    "AR": [],   # empty list → dropdown disabled in UI
}
```

The dropdown always prepends an `("", "Any (mixed)")` option in code. "Any" is represented as empty string at the widget layer and `None` once normalized in `_bulk_start()`.

Storage values match the `Question.type` field already in the schema:
- DM uses `Question.type` directly (`syllogism`, `logical`, `venn`, `probability`, `argument`).
- VR uses `Question.type` set to `tf` or `mc`.
- QR uses `QRChart.type` on the stimulus (one chart per set, so all 4 questions naturally share the chart type).

## 5. UI changes — Bulk tab

Three changes to `_tab_bulk()` in `ucat/ui.py`. Everything else (Treeview except for one new column, preview pane, cost banner, Stop button, threading) is unchanged.

### 5.1 Subtype dropdown

Inserted between the Section radios row and the Quantity row.

```
Section:  ( ) VR  (•) DM  ( ) QR  ( ) AR

Subtype:  [ Venn                          ▾ ]   ← NEW

Questions: [  10  ]   Topic hint: [_________________________]
→ 2 sets × 5 questions = 10 questions               ← NEW helper
```

- Implemented as a `ttk.Combobox` (read-only state) bound to `self._bulk_subtype` (`tk.StringVar`).
- Values rebuild on Section change via the existing `_bulk_inputs_changed` callback, which now also calls a new `_bulk_refresh_subtype_choices()`.
- For AR, the combobox is disabled (`state="disabled"`) and shows "Any" greyed out.
- Selection persists per-section via `settings["bulk_subtype_by_section"]` (see §8).

### 5.2 Quantity label flip + helper line

- The `tk.Label(qr, text="Quantity:" ...)` becomes a stored handle so `_bulk_inputs_changed` can flip its `text` between `"Quantity:"` (legacy alias for sets), `"Sets:"`, and `"Questions:"`. Implementation uses `"Sets:"` when subtype is empty/Any, `"Questions:"` when a subtype is chosen.
- A new `self._bulk_yield_lbl` label sits directly below the quantity row. Renders only when subtype is set:
  ```
  → 2 sets × 5 questions = 10 questions
  → 2 sets × 5 questions = 10 questions  (3 extra)   # when n_input not divisible
  ```
  Hidden (empty text) when subtype is Any.

### 5.3 Treeview column

A new `Subtype` column inserted between `Started` and `Status`:

```
#  Started   Subtype    Status     Verdict   Cost     Difficulty
1  14:02:11  Venn       ✓ done     ✓         $0.12    3.2
2  14:02:48  Venn       ✓ done     ⚠ drift   $0.14    2.9
3  14:03:31  Venn       ✗ failed   —         —        —
```

Width: 110px. Anchor: center. Value is the human label from `SUBTYPES_BY_SECTION` or `"—"` for Any.

## 6. Quantity → set count math

In `_bulk_start()`, after parsing the raw input:

```python
n_input = int(raw)                                     # 10
if subtype:                                            # subtype = "venn" → questions
    per_set = SET_SIZES[section]                       # 5
    n_sets  = math.ceil(n_input / per_set)             # 2
else:                                                  # subtype = None → sets
    n_sets  = n_input                                  # 10

n_sets = min(n_sets, BULK_MAX_QUANTITY)                # cap at 100 sets
```

`BULK_MAX_QUANTITY = 100` is now applied to **set count**, regardless of input unit. So:

- DM: max questions input → 500 questions → capped at 100 sets.
- VR / QR: max questions input → 400 questions → capped at 100 sets.

Cap behavior in the helper line (when capped):
```
→ 100 sets × 5 questions = 500 questions   (capped — split into multiple runs for more)
```

The cost-preview banner always expresses cost in `n_sets`.

## 7. Generation pipeline changes

### 7.1 `RAGEngine.generate()` signature

```python
def generate(
    self,
    section: str,
    hint: str = "",
    *,
    subtype: Optional[str] = None,                     # NEW
    on_progress: Optional[Callable[[str], None]] = None,
    on_delta: Optional[Callable[[str], None]] = None,
    variation_seed: Optional[str] = None,
    force_scenario: Optional[str] = None,
    avoid_topics: Optional[List[str]] = None,
) -> Dict[str, Any]:
```

`subtype` is passed through to `_system_blocks()` and to the user-prompt builder. When `None`, behavior is identical to today.

### 7.2 `_system_blocks()` — subtype override

When `subtype` is set, the section-specific block changes:

- **DM** — replace today's `"Aim for variety: include syllogism, logical, venn, probability, AND argument subtypes across the 5 questions — one of each is ideal."` with:
  ```
  All 5 questions MUST be {subtype} type. Set `type: '{subtype}'` on every question.
  {subtype-specific reminder}
  ```
  Subtype-specific reminders:
  - `venn`: "Every question must include a structured `venn` field with 2 or 3 sets."
  - `probability`: "State all probability values clearly; answers must be mathematically verifiable."
  - `syllogism`: "Premises must be logically sound; conclusions testable."
  - `logical`: "Each question is a clue-based deduction puzzle. Conclusions must follow from the clues."
  - `argument`: "Present a clear proposition; options vary in argument strength (strongest/weakest for/against)."

- **VR** — append:
  ```
  All 4 questions MUST use `type: '{subtype}'`.
  {tf-or-mc reminder}
  ```
  - If `subtype == "tf"`: `Use exactly 3 options labelled "True", "False", "Can't Tell".`
  - If `subtype == "mc"`: `Use 4 options labelled A, B, C, D.`

- **QR** — append:
  ```
  The stimulus chart MUST be type: '{subtype}'.
  ```
  Plus the existing chart-shape guidance (already in the role block — applies to all chart types).

These overrides go into the **non-cached** block (`system_blocks[1]`) so changing subtype between calls doesn't invalidate the cache.

### 7.3 User prompt addition

In `generate()`, after the existing `user_parts` assembly:

```python
if subtype:
    label = next((lbl for v, lbl in SUBTYPES_BY_SECTION[section] if v == subtype), subtype)
    user_parts.append(f"All questions in this set must be of subtype: {label}. ")
```

### 7.4 Subtype drift detection

After parsing the response and before returning:

```python
if subtype:
    if section == "QR":
        actual_chart = data.get("stimulus", {}).get("type")
        if actual_chart != subtype:
            result["subtype_drift"] = f"Asked {subtype}, got chart type {actual_chart}"
    else:
        # DM and VR both check Question.type on each item.
        actuals = [q.get("type") for q in data.get("questions", [])]
        if not all(a == subtype for a in actuals):
            result["subtype_drift"] = f"Asked {subtype}, got {actuals}"
```

`result["subtype_drift"]` is a string (or absent). Drift is **not** retried — empirically the LLM rarely self-corrects, and a paid retry on the same prompt usually produces the same drift. The user reviews flagged sets and discards as needed.

### 7.5 No schema changes

- `Question.type` is already `Optional[str]` and accepts arbitrary subtype strings (used today as a DM tag; this spec extends that usage to VR with values `tf` / `mc`).
- `QRChart.type` already enumerates `table | bar | line | stacked_bar | pie`.
- No new Pydantic fields, no DB migration. Legacy KB rows with no `type` on VR questions still validate.

## 8. Settings additions

Three new keys in `Settings.DEFAULTS`:

```python
"bulk_subtype":            "",          # current dropdown value (empty == "Any")
"bulk_subtype_by_section": {            # remember per-section choice across switches
    "VR": "", "DM": "", "QR": "", "AR": "",
},
"bulk_quantity_unit":      "sets",      # derived from subtype but persisted for UX continuity
```

Persistence flow:

- On Subtype dropdown change: write `bulk_subtype` and `bulk_subtype_by_section[current_section]`.
- On Section change: read `bulk_subtype_by_section[new_section]` and restore it into the dropdown.
- `bulk_quantity_unit` flips automatically and is saved alongside other inputs in `_bulk_inputs_changed`.

## 9. Telemetry

Existing `bulk_run_start` and `bulk_run_end` events gain new fields:

- `bulk_run_start`: add `subtype` (string or null).
- `bulk_run_end`: add `subtype` (string or null), `drift_count` (int — number of sets with subtype_drift recorded).
- `rag_generate` trace: add `subtype` so future single-shot use is captured automatically.

No new event types; `telemetry.emit()` already accepts arbitrary fields.

## 10. Edge cases & guardrails

- **Subtype + topic hint together.** Allowed and intended (e.g. "ecology" + "venn"). No validation; the prompt simply layers both.
- **Section change resets subtype to last-used for that section.** Restored from `bulk_subtype_by_section[new_section]`. Dropdown does not silently inherit the previous section's subtype.
- **AR + subtype.** Impossible by construction — dropdown is disabled. No code path needs to handle it.
- **Empty KB for the chosen section.** Today's `db.count(section, indexed_only=True) == 0` check still applies. No subtype-level KB check.
- **Subtype drift on every set in a run.** Run still completes. Treeview shows N `⚠ drift` badges. Final status: `"Bulk run finished: N succeeded (M with subtype drift), F failed."`
- **Rounding visible to user.** Helper line ("→ 2 sets × 5 questions = 10 questions") makes the rounding explicit before the user clicks Start. No surprise.
- **Cost confirmation threshold.** Unchanged. Triggered when `high > $5.00`. Modal includes the subtype name.
- **Concurrent runs.** Unchanged (Bulk-while-Bulk and Bulk-while-Single both blocked).
- **App close mid-run.** Unchanged (`on_close` sets `_bulk_stop`, joins thread with 5s timeout).

## 11. Files touched

- **`ucat/config.py`**
  - Add `SET_SIZES` dict.
  - Add `SUBTYPES_BY_SECTION` dict.
  - Add three new keys to `Settings.DEFAULTS`: `bulk_subtype`, `bulk_subtype_by_section`, `bulk_quantity_unit`.

- **`ucat/rag.py`**
  - `generate()` gains `subtype: Optional[str] = None` kwarg.
  - `_system_blocks()` gains `subtype` param and emits the override text described in §7.2.
  - User prompt assembly appends the subtype line per §7.3.
  - Drift detection block per §7.4 added before `return result`.
  - Trace fields include `subtype`.

- **`ucat/ui.py`**
  - `_tab_bulk()` adds the Subtype combobox row and the helper-line label.
  - `_bulk_inputs_changed()` flips the Quantity label, recomputes `n_sets` for cost preview, refreshes the helper line, and persists settings.
  - New `_bulk_refresh_subtype_choices()` rebuilds combobox values on Section change and restores the per-section selection.
  - `_bulk_start()` reads subtype, applies the `ceil(n_input / per_set)` math, and threads `subtype` through to the worker.
  - `_bulk_worker()` forwards `subtype` to `rag.generate()`.
  - `_mark_row()` populates the new `Subtype` Treeview column and renders `⚠ drift` in the `Verdict` column when `result["subtype_drift"]` is present.
  - Treeview `cols` tuple updated to `("#", "Started", "Subtype", "Status", "Verdict", "Cost", "Difficulty")` and the parallel widths tuple updated to `(44, 90, 110, 180, 100, 70, 80)`.

- **`ucat/telemetry.py`** — *no changes.* Extra fields ride existing `emit()`.

## 12. Testing notes

- **Smoke (DM venn):** Pick DM → Venn → Quantity 10 → Start. Verify helper line reads `→ 2 sets × 5 questions = 10 questions`. Two sets land in History. Open each in the preview pane: every question has `type: "venn"` and a structured `venn` field renders.
- **Rounding (DM logical):** Quantity 7 → helper reads `→ 2 sets × 5 questions = 10 questions (3 extra)`. Two sets generated.
- **VR tf:** Pick VR → True/False/Can't Tell → Quantity 8. Two sets generated; every question has 3 options (`True`, `False`, `Can't Tell`) and `type: tf`.
- **VR mc:** Pick VR → Multiple choice → Quantity 8. Two sets generated; every question has 4 options (A-D) and `type: mc`.
- **QR bar:** Pick QR → Bar chart → Quantity 8. Two sets; both stimuli have `type: "bar"` with categories + series.
- **Any backwards-compat:** Subtype dropdown left on "Any (mixed)". Quantity label reads `Sets:`. Helper line hidden. Behavior identical to pre-change bulk runs.
- **AR disabled:** Switch to AR. Subtype dropdown shows "Any" greyed out. Quantity label stays `Sets:`. Run proceeds as today.
- **Per-section memory:** DM → Venn → switch to VR → switch back to DM. Dropdown should restore to "Venn".
- **Drift surfacing:** Manually edit the prompt to force the LLM to emit a wrong subtype (or run enough trials to observe one). Verify `⚠ drift` appears in Verdict column and the row is still recorded as a success in the final summary.
- **Cap:** Pick DM → Venn → Quantity 600. Helper line reads `→ 100 sets × 5 questions = 500 questions  (capped — split into multiple runs for more)`. Cost preview reflects 100 sets.
- **Cost confirm:** Pick a quantity that pushes `high > $5.00`. Modal appears, including subtype name. Below threshold: no modal.

---

End of spec.
