# Bulk Equate Mode + Cost Transparency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an "equate" toggle to the Bulk Generate tab that runs the same set count across VR/QR/SJT/DM in one round-robin run, plus three cost-transparency improvements (per-section estimate breakdown, live spent label, post-run summary) and a user-configurable cost-confirm threshold.

**Architecture:** A `tk.BooleanVar` `_bulk_equate` toggle reshapes the Bulk tab UI (radios disable, subtype frame hides, quantity forced to "Sets"). `_bulk_start` branches on the toggle and either builds `[section] * n` (single-section) or `equate_task_list(n)` (round-robin VR/QR/SJT/DM repeated n times). The worker takes a `task_list` instead of `(section, n)` so both paths share one loop. Cost transparency is layered onto the existing `_bulk_run_cost` accumulator: a per-section breakdown via a multiplier table, a live spent label, and a summary line written by `_bulk_run_finished`.

**Tech Stack:** Python 3, Tkinter (`ttk`), no test framework (homegrown runner using plain `assert` per existing convention in `tests/test_*.py`).

**Spec:** `docs/superpowers/specs/2026-04-28-bulk-equate-mode-design.md`

---

## File Structure

- **Modify** `ucat/config.py` — add `EQUATE_SECTIONS` tuple, `equate_task_list` function, `SECTION_COST_MULTIPLIERS` dict, `estimate_section_cost` function, and two new `Settings.DEFAULTS` keys (`bulk_equate`, `bulk_cost_confirm_threshold`).
- **Modify** `ucat/ui.py` — wrap the existing subtype row in a stored frame handle, add equate checkbox, threshold entry, two-line cost banner, live-spent label, Section column in Treeview, refactor `_bulk_worker` signature to `(task_list, hint, subtype)`, branch in `_bulk_start`, batched no-RAG dialog, post-run summary in `_bulk_run_finished`.
- **Create** `tests/test_equate_mode.py` — unit tests for the new pure helpers (`equate_task_list`, `estimate_section_cost`) and settings defaults.

The UI changes all live inside the existing `_tab_bulk()` method and its callbacks (`_bulk_inputs_changed`, `_bulk_start`, `_bulk_seed_rows`, `_bulk_set_row`, `_bulk_run_started`, `_bulk_run_finished`, `_bulk_worker`). No new modules are needed — the spec changes are deliberately scoped to extending existing structures.

---

## Task 1: Add `EQUATE_SECTIONS` constant and `equate_task_list` helper

**Files:**
- Modify: `ucat/config.py` (add constant + helper after `compute_set_count`, near line 158)
- Create: `tests/test_equate_mode.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_equate_mode.py` with the following content:

```python
"""Tests for bulk equate mode + cost transparency. Runnable directly:

    ./venv/bin/python tests/test_equate_mode.py

Each function with a name starting `test_` is run; failures raise.
"""
from __future__ import annotations

import os
import sys
import tempfile

# Make the project importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ucat.config import (
    EQUATE_SECTIONS,
    equate_task_list,
)


# ─── equate_task_list ────────────────────────────────────────────────────────

def test_equate_sections_excludes_ar():
    assert "AR" not in EQUATE_SECTIONS, \
        f"AR should not be in EQUATE_SECTIONS, got {EQUATE_SECTIONS}"
    assert set(EQUATE_SECTIONS) == {"VR", "QR", "SJT", "DM"}, \
        f"unexpected sections: {EQUATE_SECTIONS}"


def test_equate_task_list_zero_returns_empty():
    assert equate_task_list(0) == []


def test_equate_task_list_negative_returns_empty():
    assert equate_task_list(-3) == []


def test_equate_task_list_n1_returns_one_of_each():
    result = equate_task_list(1)
    assert len(result) == 4
    assert set(result) == {"VR", "QR", "SJT", "DM"}


def test_equate_task_list_n3_round_robin_order():
    result = equate_task_list(3)
    assert len(result) == 12
    # Round-robin: cycle 1, cycle 2, cycle 3 — each cycle is the EQUATE_SECTIONS order.
    expected_cycle = list(EQUATE_SECTIONS)
    assert result[0:4] == expected_cycle
    assert result[4:8] == expected_cycle
    assert result[8:12] == expected_cycle


def test_equate_task_list_balanced_counts():
    """Every n should produce 4*n total tasks with exactly n per section."""
    for n in (1, 5, 25, 100):
        result = equate_task_list(n)
        assert len(result) == 4 * n
        for s in EQUATE_SECTIONS:
            assert result.count(s) == n, \
                f"n={n}, section {s}: expected {n}, got {result.count(s)}"


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

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python tests/test_equate_mode.py`
Expected: ImportError on `EQUATE_SECTIONS` / `equate_task_list` (those names don't exist yet).

- [ ] **Step 3: Add constant and helper to `ucat/config.py`**

Open `ucat/config.py`. Find the existing `compute_set_count` function (around line 142). Immediately **after** that function, add:

```python
# ─── Bulk equate mode ────────────────────────────────────────────────────────

# The four sections that participate in equate mode. AR is intentionally
# excluded — it's pattern-matching, not directly comparable to the four
# reasoning sections, and a balanced UCAT mock typically pairs the reasoning
# sections together. Order is the round-robin order used by equate_task_list.
EQUATE_SECTIONS: tuple[str, ...] = ("VR", "QR", "SJT", "DM")


def equate_task_list(n: int) -> list[str]:
    """Round-robin section list for an equate run.

    n=0 → []
    n=1 → ["VR", "QR", "SJT", "DM"]
    n=3 → ["VR", "QR", "SJT", "DM", "VR", "QR", "SJT", "DM", "VR", "QR", "SJT", "DM"]

    Returns an empty list for non-positive n.
    """
    if n <= 0:
        return []
    return list(EQUATE_SECTIONS) * n
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python tests/test_equate_mode.py`
Expected: All 6 tests print `PASS`. Final line: `All tests passed.`

- [ ] **Step 5: Verify existing tests still pass**

Run: `./venv/bin/python tests/test_subtype_targeting.py && ./venv/bin/python tests/test_bulk_cost.py`
Expected: All tests pass in both files. (Sanity check that adding the new constant didn't break imports.)

- [ ] **Step 6: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/config.py tests/test_equate_mode.py
git commit -m "$(cat <<'EOF'
feat: EQUATE_SECTIONS constant + equate_task_list helper

Pure helper that builds a round-robin section task list for the
upcoming equate mode in the Bulk tab. AR is intentionally excluded.

Co-Authored-By: claude-flow <ruv@ruv.net>
EOF
)"
```

---

## Task 2: Add `SECTION_COST_MULTIPLIERS` and `estimate_section_cost` helper

**Files:**
- Modify: `ucat/config.py` (add dict + helper near `estimate_bulk_cost`, around line 311)
- Modify: `tests/test_equate_mode.py` (extend with cost tests)

- [ ] **Step 1: Add failing tests for `estimate_section_cost`**

Open `tests/test_equate_mode.py`. After the `test_equate_task_list_balanced_counts` function and before the `if __name__ == "__main__":` block, append:

```python
# ─── estimate_section_cost ───────────────────────────────────────────────────

from ucat.config import (
    SECTION_COST_MULTIPLIERS,
    estimate_section_cost,
    estimate_bulk_cost,
)


def test_section_multipliers_normalised_across_equate():
    """Sum of multipliers across EQUATE_SECTIONS must equal 4.0 so total cost
    is unchanged from a flat 4× estimate (just split, not inflated)."""
    total = sum(SECTION_COST_MULTIPLIERS[s] for s in EQUATE_SECTIONS)
    assert abs(total - 4.0) < 1e-9, \
        f"EQUATE_SECTIONS multipliers sum to {total}, expected 4.0"


def test_section_multipliers_all_positive():
    for s, m in SECTION_COST_MULTIPLIERS.items():
        assert m > 0, f"section {s} has non-positive multiplier {m}"


def test_estimate_section_cost_applies_multiplier():
    """Per-section cost is base × multiplier."""
    base_low, base_high = estimate_bulk_cost(
        10, "claude-sonnet-4-6", multi_judge=False, verify=False)
    sec_low, sec_high = estimate_section_cost(
        "VR", 10, "claude-sonnet-4-6", multi_judge=False, verify=False)
    expected_mult = SECTION_COST_MULTIPLIERS["VR"]
    assert abs(sec_low  - base_low  * expected_mult) < 1e-9
    assert abs(sec_high - base_high * expected_mult) < 1e-9


def test_estimate_section_cost_sums_to_4x_baseline_across_equate():
    """For the same n_sets and model, sum of per-section figures across
    EQUATE_SECTIONS equals 4× the baseline estimate (within rounding)."""
    base_low, base_high = estimate_bulk_cost(
        25, "claude-sonnet-4-6", multi_judge=False, verify=True)
    total_low, total_high = 0.0, 0.0
    for s in EQUATE_SECTIONS:
        l, h = estimate_section_cost(
            s, 25, "claude-sonnet-4-6", multi_judge=False, verify=True)
        total_low  += l
        total_high += h
    assert abs(total_low  - 4 * base_low)  < 1e-9
    assert abs(total_high - 4 * base_high) < 1e-9


def test_estimate_section_cost_unknown_section_uses_1():
    """Sections outside the multiplier table fall back to multiplier 1.0."""
    base_low, base_high = estimate_bulk_cost(
        5, "claude-sonnet-4-6", multi_judge=False, verify=False)
    sec_low, sec_high = estimate_section_cost(
        "UNKNOWN", 5, "claude-sonnet-4-6", multi_judge=False, verify=False)
    assert abs(sec_low  - base_low)  < 1e-9
    assert abs(sec_high - base_high) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/python tests/test_equate_mode.py`
Expected: ImportError on `SECTION_COST_MULTIPLIERS` / `estimate_section_cost`.

- [ ] **Step 3: Add the multiplier table and helper to `ucat/config.py`**

Open `ucat/config.py`. Find the existing `estimate_bulk_cost` function (around line 271). Immediately **before** the `def estimate_bulk_cost(` line, add:

```python
# Per-section cost multipliers, normalised to mean 1.0 across EQUATE_SECTIONS
# so that sum across the four equate sections is exactly 4.0 (i.e. 4× the
# flat baseline). These are educated guesses based on prompt structure:
#   - VR  carries a passage (200-300 words) + 4 short questions
#   - QR  carries a chart spec (table/bar/line/etc.) + 4 numerical questions
#         — chart spec adds output tokens, hence the bump above 1.0
#   - SJT carries a workplace scenario + 4 Likert items — short, hence below 1.0
#   - DM  carries 5 standalone items — used as the baseline (1.0)
# Refine via telemetry in a follow-up spec.
SECTION_COST_MULTIPLIERS: dict[str, float] = {
    "VR":  0.95,
    "QR":  1.20,
    "SJT": 0.85,
    "DM":  1.00,
    "AR":  1.10,   # not in EQUATE_SECTIONS today, included for forward-compat
}


```

Then, immediately **after** the existing `estimate_bulk_cost` function (after its closing line that returns `(per_low * n, per_high * n)`), add:

```python


def estimate_section_cost(
    section: str,
    n_sets: int,
    llm: str,
    *,
    multi_judge: bool,
    verify: bool,
) -> tuple[float, float]:
    """Per-section cost estimate. Wraps ``estimate_bulk_cost`` and applies the
    section-specific multiplier from ``SECTION_COST_MULTIPLIERS``.

    Sections outside the table fall back to multiplier 1.0.
    """
    low, high = estimate_bulk_cost(
        n_sets, llm, multi_judge=multi_judge, verify=verify)
    m = SECTION_COST_MULTIPLIERS.get(section, 1.0)
    return low * m, high * m
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/python tests/test_equate_mode.py`
Expected: All tests pass (6 from Task 1 + 5 new = 11 total).

- [ ] **Step 5: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/config.py tests/test_equate_mode.py
git commit -m "$(cat <<'EOF'
feat: SECTION_COST_MULTIPLIERS + estimate_section_cost helper

Per-section cost multipliers normalised so sum across EQUATE_SECTIONS
is 4.0 — total estimate matches a flat 4× baseline, just split per
section. Uses educated guesses about prompt size; telemetry-derived
refinement deferred to a follow-up spec.

Co-Authored-By: claude-flow <ruv@ruv.net>
EOF
)"
```

---

## Task 3: Add new settings keys (`bulk_equate`, `bulk_cost_confirm_threshold`)

**Files:**
- Modify: `ucat/config.py` (`Settings.DEFAULTS` dict, around line 192)
- Modify: `tests/test_equate_mode.py` (extend with settings tests)

- [ ] **Step 1: Add failing tests**

Open `tests/test_equate_mode.py`. Append before the `if __name__ == "__main__":` block:

```python
# ─── Settings defaults ───────────────────────────────────────────────────────

from ucat.config import Settings


def test_settings_default_bulk_equate_is_false():
    s = Settings(path=tempfile.mktemp(suffix=".json"))
    assert s.get("bulk_equate") is False, \
        f"expected default False, got {s.get('bulk_equate')!r}"


def test_settings_default_bulk_cost_confirm_threshold_is_5():
    s = Settings(path=tempfile.mktemp(suffix=".json"))
    threshold = s.get("bulk_cost_confirm_threshold")
    assert threshold == 5.00, \
        f"expected default 5.00, got {threshold!r}"


def test_settings_persists_bulk_equate():
    path = tempfile.mktemp(suffix=".json")
    s1 = Settings(path=path)
    s1.set("bulk_equate", True)
    s2 = Settings(path=path)
    assert s2.get("bulk_equate") is True


def test_settings_persists_bulk_cost_confirm_threshold():
    path = tempfile.mktemp(suffix=".json")
    s1 = Settings(path=path)
    s1.set("bulk_cost_confirm_threshold", 12.50)
    s2 = Settings(path=path)
    assert s2.get("bulk_cost_confirm_threshold") == 12.50


def test_settings_missing_keys_fall_back_to_defaults():
    """A settings file from before this change should load cleanly with the
    new keys taking their defaults."""
    import json
    path = tempfile.mktemp(suffix=".json")
    with open(path, "w") as f:
        json.dump({"llm": "claude-sonnet-4-6"}, f)  # no bulk_equate, no threshold
    s = Settings(path=path)
    assert s.get("bulk_equate") is False
    assert s.get("bulk_cost_confirm_threshold") == 5.00
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/python tests/test_equate_mode.py`
Expected: The 5 new tests fail because `bulk_equate` / `bulk_cost_confirm_threshold` aren't in `DEFAULTS`. Earlier tests still pass.

- [ ] **Step 3: Add the keys to `Settings.DEFAULTS`**

Open `ucat/config.py`. Find `Settings.DEFAULTS` (around line 192). Currently the relevant section reads:

```python
        "bulk_subtype":            "",   # current dropdown value ("" == Any/mixed)
        "bulk_subtype_by_section": {     # remember per-section choice
            "VR": "", "DM": "", "QR": "", "AR": "", "SJT": "",
        },
        "bulk_quantity_unit":      "sets",  # "sets" or "questions" (derived from subtype)
    }
```

Add two new keys immediately above the closing `}`. The block becomes:

```python
        "bulk_subtype":            "",   # current dropdown value ("" == Any/mixed)
        "bulk_subtype_by_section": {     # remember per-section choice
            "VR": "", "DM": "", "QR": "", "AR": "", "SJT": "",
        },
        "bulk_quantity_unit":      "sets",  # "sets" or "questions" (derived from subtype)
        # ── Equate mode + configurable confirm threshold (added 2026-04-28) ─
        "bulk_equate":                  False,   # tick to run VR/QR/SJT/DM together
        "bulk_cost_confirm_threshold":  5.00,    # USD; overrides BULK_COST_CONFIRM_THRESHOLD
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/python tests/test_equate_mode.py`
Expected: All tests pass (now 16 total).

- [ ] **Step 5: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/config.py tests/test_equate_mode.py
git commit -m "$(cat <<'EOF'
feat: add bulk_equate + bulk_cost_confirm_threshold settings

bulk_equate (default False) and bulk_cost_confirm_threshold (default
5.00, replacing the hardcoded constant). Existing settings files load
cleanly with both keys taking their defaults.

Co-Authored-By: claude-flow <ruv@ruv.net>
EOF
)"
```

---

## Task 4: Refactor `_bulk_worker` signature to accept a `task_list`

This is a **preparatory refactor** with no user-visible behavior change. After this task, single-section runs work exactly as before, but the worker is shaped so equate mode can drop in cleanly.

**Files:**
- Modify: `ucat/ui.py` (`_bulk_seed_rows`, `_bulk_set_row`, `_bulk_worker`, `_bulk_start`)

- [ ] **Step 1: Update `_bulk_seed_rows` to take a task list**

Open `ucat/ui.py`. Find `_bulk_seed_rows` (around line 740). Replace its body with:

```python
    def _bulk_seed_rows(self, task_list: list[str]):
        """Initialise _bulk_rows + Treeview with one queued entry per task.
        Each task is the section code for that row (single-section mode passes
        [section] * n; equate mode passes a round-robin list)."""
        subtype_value = self.settings.get("bulk_subtype") or ""
        # Subtype label is per-section; computed once for single-section runs
        # (all rows share the section). For equate runs subtype is forced None,
        # so the per-row label is always "—".
        self._bulk_rows = [
            {"idx": i, "status": "queued", "result": None,
              "error": None, "started": "",
              "section": section,
              "subtype": subtype_value or None}
            for i, section in enumerate(task_list, start=1)
        ]
        for iid in self._bulk_tree.get_children():
            self._bulk_tree.delete(iid)
        for r in self._bulk_rows:
            section = r["section"]
            subtype_label = next(
                (lbl for v, lbl in SUBTYPES_BY_SECTION.get(section, [])
                 if v == subtype_value),
                "—",
            ) if subtype_value else "—"
            # Note: Section column isn't added to the Treeview yet — that's
            # Task 5. For now the row values match the existing 7-column shape.
            self._bulk_tree.insert(
                "", "end", iid=self._bulk_row_iid(r["idx"]),
                values=(r["idx"], "", subtype_label, "queued", "—", "—", "—"),
            )
```

- [ ] **Step 2: Update `_bulk_set_row` to use the row's section field**

Open `ucat/ui.py`. Find `_bulk_set_row` (around line 761). Find this block:

```python
        # Subtype cell — derived from the row's stored subtype (set in seed_rows).
        subtype_value = row.get("subtype")
        subtype_cell = next(
            (lbl for v, lbl in SUBTYPES_BY_SECTION.get(self._bulk_sec.get(), [])
             if v == subtype_value),
            "—",
        ) if subtype_value else "—"
```

Replace with (changing the section source from `self._bulk_sec.get()` to the row's own section):

```python
        # Subtype cell — derived from the row's stored subtype + section.
        # Use the row's own section so equate-mode rows resolve their subtype
        # against the correct section's catalogue.
        subtype_value = row.get("subtype")
        row_section   = row.get("section") or self._bulk_sec.get()
        subtype_cell = next(
            (lbl for v, lbl in SUBTYPES_BY_SECTION.get(row_section, [])
             if v == subtype_value),
            "—",
        ) if subtype_value else "—"
```

- [ ] **Step 3: Refactor `_bulk_worker` to take a task list**

Open `ucat/ui.py`. Find `_bulk_worker` (around line 933). Replace the entire function with:

```python
    def _bulk_worker(self, task_list: list[str], hint: str,
                       subtype: Optional[str]):
        n = len(task_list)
        self.after(0, lambda: self._bulk_run_started(task_list))
        succeeded = 0
        failed    = 0
        for i, section in enumerate(task_list, start=1):
            if self._bulk_stop.is_set():
                # Mark this and every later row as skipped, then exit.
                for j in range(i, n + 1):
                    self.after(0, lambda idx=j: self._bulk_set_row(idx, status="skipped"))
                self.after(0, lambda: self._bulk_run_finished(succeeded, failed, stopped=True))
                return

            started_at = datetime.now().strftime("%H:%M:%S")
            self.after(0, lambda idx=i, t=started_at: self._bulk_set_row(
                idx, status="running", started=t))
            self.after(0, lambda idx=i, total=n: self._bulk_progress_lbl.config(
                text=f"{idx - 1} / {total} — generating set {idx}…"))

            # Pre-pick an under-represented scenario + recently-overused topics
            # to avoid, so the bulk run rotates through diverse scenarios instead
            # of clustering on whatever the LLM defaults to. Bounded prompt cost.
            stats = self.db.coverage_stats(section, last_n=200)
            diversify = pick_diversification(stats, section) or {}

            attempts = 0
            done = False
            while attempts < 2 and not done:
                try:
                    result = self.rag.generate(
                        section, hint,
                        subtype=subtype,
                        on_progress=lambda m, idx=i: self.after(0, lambda msg=m, _i=idx:
                            self._bulk_set_row(_i, status="running", progress=msg)),
                        on_delta=None,
                        on_verify_complete=lambda upd, idx=i: self.after(
                            0, lambda u=upd, _i=idx: self._bulk_verify_complete(_i, u)),
                        force_scenario=diversify.get("scenario"),
                        avoid_topics=diversify.get("avoid_topics"),
                    )
                    self.after(0, lambda idx=i, r=result: self._bulk_after_success(idx, r))
                    succeeded += 1
                    done = True
                except Exception as e:
                    attempts += 1
                    logger.exception(f"Bulk set {i} attempt {attempts} failed")
                    if attempts >= 2:
                        err = str(e)
                        self.after(0, lambda idx=i, msg=err: self._bulk_set_row(
                            idx, status="failed", error=msg))
                        failed += 1
                    else:
                        self.after(0, lambda idx=i, msg=str(e): self._bulk_set_row(
                            idx, status="running",
                            progress=f"retrying ({msg[:40]})"))
                        time.sleep(1.0)

        self.after(0, lambda: self._bulk_run_finished(succeeded, failed))
```

The key changes:
- Signature is `(task_list, hint, subtype)` instead of `(section, hint, n, subtype)`
- `n` is derived from `len(task_list)`
- Inside the loop, `section` comes from `task_list[i-1]` (via `enumerate`)

- [ ] **Step 4: Update `_bulk_run_started` to accept the task list**

Open `ucat/ui.py`. Find `_bulk_run_started` (around line 834). Replace its body with:

```python
    def _bulk_run_started(self, task_list: list[str]):
        n = len(task_list)
        self._bulk_started_at = time.perf_counter()
        self._bulk_run_cost   = 0.0
        self._bulk_start_btn.config(state="disabled", text="Generating…")
        self._bulk_stop_btn.config(state="normal")
        self._bulk_progress_lbl.config(text=f"0 / {n}")
        self._bulk_seed_rows(task_list)
        llm    = self.settings.get("llm")
        verify = bool(self.settings.get("verify"))
        jury   = bool(self.settings.get("multi_judge"))
        _, est_high = estimate_bulk_cost(n, llm, multi_judge=jury, verify=verify)
        # Section field reflects the *first* section in the task list — single-
        # section runs share one section across all rows; equate runs use the
        # task_list field on the bulk_run_end event instead.
        emit("bulk_run_start",
             section=task_list[0] if task_list else self._bulk_sec.get(),
             n=n,
             model=llm,
             verify=verify,
             multi_judge=jury,
             estimated_cost_high=round(est_high, 4),
             subtype=(self.settings.get("bulk_subtype") or None))
```

- [ ] **Step 5: Update `_bulk_start` to build a task list and pass it to the worker**

Open `ucat/ui.py`. Find `_bulk_start` (around line 624). Find this block near the end of the method:

```python
        self._bulk_stop.clear()
        self._bulk_thread = threading.Thread(
            target=self._bulk_worker, args=(section, hint, n, subtype), daemon=True
        )
        self._bulk_thread.start()
```

Replace with:

```python
        # Build the task list. Single-section mode = [section] * n; equate
        # mode (added in a later task) will branch here to call
        # equate_task_list(n) instead.
        task_list = [section] * n

        self._bulk_stop.clear()
        self._bulk_thread = threading.Thread(
            target=self._bulk_worker, args=(task_list, hint, subtype), daemon=True
        )
        self._bulk_thread.start()
```

- [ ] **Step 6: Smoke-test the refactor (no functional change)**

Run: `./venv/bin/python tests/test_equate_mode.py && ./venv/bin/python tests/test_subtype_targeting.py && ./venv/bin/python tests/test_bulk_cost.py`
Expected: All three test files pass. The refactor is type-clean.

- [ ] **Step 7: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/ui.py
git commit -m "$(cat <<'EOF'
refactor: _bulk_worker takes task_list instead of (section, n)

Preparatory refactor for equate mode: the worker now iterates a list
of section codes (one per row) instead of looping a fixed n with a
single section. Single-section mode passes [section] * n so behavior
is unchanged for users. Each bulk row dict gains a 'section' field
so per-row Treeview cells can resolve against the correct section's
subtype catalogue.

Co-Authored-By: claude-flow <ruv@ruv.net>
EOF
)"
```

---

## Task 5: Add Section column to the Treeview

**Files:**
- Modify: `ucat/ui.py` (`_tab_bulk` cols/widths tuples, `_bulk_seed_rows`, `_bulk_set_row`)

- [ ] **Step 1: Update the Treeview column tuples**

Open `ucat/ui.py`. Find the existing column setup in `_tab_bulk` (around line 479). Currently:

```python
        cols = ("#", "Started", "Subtype", "Status", "Verdict", "Cost", "Difficulty")
        self._bulk_tree = ttk.Treeview(tf, columns=cols, show="headings", height=10)
        for c, w in zip(cols, (44, 90, 110, 180, 100, 70, 80)):
            self._bulk_tree.heading(c, text=c)
            self._bulk_tree.column(c, width=w, anchor="w" if c == "Status" else "center")
```

Replace with (insert `"Section"` between `"Started"` and `"Subtype"`, and add a width of 70):

```python
        cols = ("#", "Started", "Section", "Subtype", "Status", "Verdict", "Cost", "Difficulty")
        self._bulk_tree = ttk.Treeview(tf, columns=cols, show="headings", height=10)
        for c, w in zip(cols, (44, 90, 70, 110, 180, 100, 70, 80)):
            self._bulk_tree.heading(c, text=c)
            self._bulk_tree.column(c, width=w, anchor="w" if c == "Status" else "center")
```

- [ ] **Step 2: Update `_bulk_seed_rows` to populate the Section cell**

Open `ucat/ui.py`. Find the `self._bulk_tree.insert(...)` call inside `_bulk_seed_rows` (the body you wrote in Task 4). Currently:

```python
            self._bulk_tree.insert(
                "", "end", iid=self._bulk_row_iid(r["idx"]),
                values=(r["idx"], "", subtype_label, "queued", "—", "—", "—"),
            )
```

Replace with (insert the section between the empty "Started" placeholder and the subtype label):

```python
            self._bulk_tree.insert(
                "", "end", iid=self._bulk_row_iid(r["idx"]),
                values=(r["idx"], "", section, subtype_label,
                          "queued", "—", "—", "—"),
            )
```

- [ ] **Step 3: Update `_bulk_set_row` to include the Section cell on every update**

Open `ucat/ui.py`. Find the `self._bulk_tree.item(...)` call inside `_bulk_set_row` (around line 828). Currently:

```python
        self._bulk_tree.item(
            self._bulk_row_iid(idx),
            values=(idx, row["started"], subtype_cell, st_cell,
                      verdict_cell, cost_cell, diff_cell),
        )
```

Replace with (insert `row.get("section") or "—"` between `row["started"]` and `subtype_cell`):

```python
        self._bulk_tree.item(
            self._bulk_row_iid(idx),
            values=(idx, row["started"],
                      row.get("section") or "—",
                      subtype_cell, st_cell,
                      verdict_cell, cost_cell, diff_cell),
        )
```

- [ ] **Step 4: Manual smoke test**

Run: `./venv/bin/python ucat_trainer.py`
Open the Bulk tab. Confirm:
- The Treeview has 8 columns: `#`, `Started`, `Section`, `Subtype`, `Status`, `Verdict`, `Cost`, `Difficulty`.
- Click `START BULK RUN` with default settings (VR, 1 set). The Section column on the row reads `VR`.
- Stop the app cleanly with the X button.

(If you don't have API keys set up, you can verify just the column header layout by opening the tab — no run needed.)

- [ ] **Step 5: Run all tests**

Run: `./venv/bin/python tests/test_equate_mode.py && ./venv/bin/python tests/test_subtype_targeting.py && ./venv/bin/python tests/test_bulk_cost.py`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/ui.py
git commit -m "$(cat <<'EOF'
feat: bulk-tab Section column in Treeview

Always-on column showing the section each row was generated for.
In single-section mode every row shows the same code (informational);
in equate mode (next task) rows alternate VR/QR/SJT/DM round-robin.

Co-Authored-By: claude-flow <ruv@ruv.net>
EOF
)"
```

---

## Task 6: Add the equate checkbox UI + state callback

**Files:**
- Modify: `ucat/ui.py` (`_tab_bulk` adds checkbox + wraps subtype row in stored frame, new `_bulk_equate_changed` callback)

- [ ] **Step 1: Wrap the subtype row in a stored frame handle**

Open `ucat/ui.py`. Find the subtype row in `_tab_bulk` (around line 419). Currently:

```python
        # Subtype dropdown — populated based on the selected section.
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

Replace the first line `sb = tk.Frame(p, bg=BG); sb.pack(anchor="w", pady=(0, 10))` with:

```python
        # Subtype dropdown — populated based on the selected section.
        # Stored on self so equate mode can hide/show the whole row.
        self._bulk_subtype_frame = tk.Frame(p, bg=BG)
        self._bulk_subtype_frame.pack(anchor="w", pady=(0, 10))
        sb = self._bulk_subtype_frame
```

(The rest of the block is unchanged — it still references `sb`, which now points at the stored frame.)

- [ ] **Step 2: Add the equate checkbox row above the Section radios**

Open `ucat/ui.py`. Find the Section radios block in `_tab_bulk` (around line 408):

```python
        # Section radios.
        sr = tk.Frame(p, bg=BG); sr.pack(anchor="w", pady=(0, 10))
        tk.Label(sr, text="Section:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self._bulk_sec = tk.StringVar(value=self.settings.get("bulk_section"))
        for code in SECTIONS:
            tk.Radiobutton(sr, text=f" {code} ", variable=self._bulk_sec, value=code,
                           bg=BG, fg=TEXT, selectcolor=PANEL, activebackground=BG,
                           activeforeground=ACCENT, font=FB, indicatoron=False,
                           relief="flat", bd=1, padx=12, pady=6, cursor="hand2",
                           command=self._bulk_section_changed
                           ).pack(side="left", padx=4)
```

Replace with (add the equate checkbox **before** the radios, and store the radio widgets in a list so the callback can disable them):

```python
        # Equate mode checkbox — inserted above the Section radios so it
        # visually frames the entire single/equate decision.
        er = tk.Frame(p, bg=BG); er.pack(anchor="w", pady=(0, 6))
        self._bulk_equate = tk.BooleanVar(value=bool(self.settings.get("bulk_equate")))
        tk.Checkbutton(
            er,
            text=" Equate across VR/QR/SJT/DM (same qty for each section)",
            variable=self._bulk_equate, bg=BG, fg=TEXT, selectcolor=PANEL,
            activebackground=BG, activeforeground=ACCENT, font=FM,
            command=self._bulk_equate_changed,
        ).pack(side="left")

        # Section radios. Stored so _bulk_equate_changed can toggle their state.
        sr = tk.Frame(p, bg=BG); sr.pack(anchor="w", pady=(0, 10))
        tk.Label(sr, text="Section:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self._bulk_sec = tk.StringVar(value=self.settings.get("bulk_section"))
        self._bulk_section_radios: list[tk.Radiobutton] = []
        for code in SECTIONS:
            rb = tk.Radiobutton(sr, text=f" {code} ", variable=self._bulk_sec, value=code,
                           bg=BG, fg=TEXT, selectcolor=PANEL, activebackground=BG,
                           activeforeground=ACCENT, font=FB, indicatoron=False,
                           relief="flat", bd=1, padx=12, pady=6, cursor="hand2",
                           command=self._bulk_section_changed
                           )
            rb.pack(side="left", padx=4)
            self._bulk_section_radios.append(rb)
```

- [ ] **Step 3: Add the `_bulk_equate_changed` callback method**

Open `ucat/ui.py`. Find the `_bulk_section_changed` method (around line 502). Immediately **before** it, add:

```python
    def _bulk_equate_changed(self):
        """Called when the equate checkbox is toggled. Disables the Section
        radios + hides the Subtype row when on; restores them when off.
        Persists the setting and triggers a cost-banner refresh."""
        equate = self._bulk_equate.get()
        self.settings.set("bulk_equate", equate)

        # Section radios.
        radio_state = "disabled" if equate else "normal"
        for rb in self._bulk_section_radios:
            rb.config(state=radio_state)

        # Subtype row — entirely hidden when equate is on.
        if equate:
            self._bulk_subtype_frame.pack_forget()
        else:
            # Repack at its original position (between section radios and
            # quantity row). pack() without `before=` re-appends to the end,
            # which is fine here because nothing has been packed since.
            # If the layout changes in future, switch to `before=<next-frame>`.
            self._bulk_subtype_frame.pack(anchor="w", pady=(0, 10))
            # When restoring, also force the choices to refresh in case the
            # current section's catalogue has changed.
            self._bulk_refresh_subtype_choices()

        self._bulk_inputs_changed()
```

- [ ] **Step 4: Apply persisted equate state on tab build**

Open `ucat/ui.py`. Find the end of `_tab_bulk` — the line `self._bulk_inputs_changed()` (around line 500). Replace with:

```python
        # If the user previously left equate ticked, apply the disabled/hidden
        # state now (the BooleanVar's initial value is set above; this call
        # propagates it through to the radios and subtype frame).
        if self._bulk_equate.get():
            self._bulk_equate_changed()

        # Initialise the cost banner.
        self._bulk_inputs_changed()
```

- [ ] **Step 5: Manual smoke test**

Run: `./venv/bin/python ucat_trainer.py`
Open the Bulk tab. Confirm:
- The new checkbox appears above the Section radios.
- Tick it: Section radios visibly grey out (disabled), Subtype row vanishes.
- Untick: radios re-enable, Subtype row reappears with the previously selected subtype.
- Quit and relaunch: equate state is restored from settings.

- [ ] **Step 6: Run all tests**

Run: `./venv/bin/python tests/test_equate_mode.py && ./venv/bin/python tests/test_subtype_targeting.py && ./venv/bin/python tests/test_bulk_cost.py`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/ui.py
git commit -m "$(cat <<'EOF'
feat: bulk-tab equate checkbox + UI state callback

Adds the equate toggle above Section radios. Ticking disables the
radios and hides the Subtype row entirely; unticking restores both
(with the user's previous subtype). State is persisted across
sessions via settings.bulk_equate. The worker is not yet wired —
that's the next task.

Co-Authored-By: claude-flow <ruv@ruv.net>
EOF
)"
```

---

## Task 7: Branch `_bulk_start` on equate mode + batched no-RAG dialog

**Files:**
- Modify: `ucat/ui.py` (`_bulk_start`, `_bulk_inputs_changed`)
- Modify: `ucat/ui.py` imports (add `EQUATE_SECTIONS`, `equate_task_list`)

- [ ] **Step 1: Update the imports at the top of `ucat/ui.py`**

Open `ucat/ui.py`. Find the existing `from ucat.config import …` block near the top (around line 25). Currently includes things like:

```python
from ucat.config import (… SUBTYPES_BY_SECTION, SET_SIZES, compute_set_count,
                      BULK_MAX_QUANTITY, BULK_COST_CONFIRM_THRESHOLD, estimate_bulk_cost,
                      …)
```

Add `EQUATE_SECTIONS, equate_task_list, estimate_section_cost` to the imported names. The block becomes (preserving every other name already there — only adding three):

```python
from ucat.config import (… SUBTYPES_BY_SECTION, SET_SIZES, compute_set_count,
                      BULK_MAX_QUANTITY, BULK_COST_CONFIRM_THRESHOLD, estimate_bulk_cost,
                      EQUATE_SECTIONS, equate_task_list, estimate_section_cost,
                      …)
```

(The exact existing list of imports is whatever is currently there — just append the three new names within the parentheses.)

- [ ] **Step 2: Branch `_bulk_start` on the equate flag**

Open `ucat/ui.py`. Find `_bulk_start` (around line 624). Find this block (after the `if not raw.isdigit(): return` and `n_input < 1` checks):

```python
        # Convert questions → sets when subtype is set; cap at the max set count.
        n = compute_set_count(n_input, section, subtype)
        n = min(n, BULK_MAX_QUANTITY)
        if n < 1:
            return

        if self.db.count(section, indexed_only=True) == 0:
            if not messagebox.askyesno("No Indexed Documents",
                f"No indexed documents for {SECTIONS[section]}.\n\n"
                "Add docs and click Index, or generate without RAG context?"):
                return
```

Replace with (adding the equate branch before the existing logic):

```python
        equate = self._bulk_equate.get()

        if equate:
            # Equate: input is sets-per-section, cap is per-section.
            subtype = None
            n_per_section = min(n_input, BULK_MAX_QUANTITY)
            if n_per_section < 1:
                return
            n = n_per_section * len(EQUATE_SECTIONS)  # used for cost preview math
            task_list = equate_task_list(n_per_section)

            # Batched no-RAG dialog: collect every empty section and ask once.
            empty = [s for s in EQUATE_SECTIONS
                     if self.db.count(s, indexed_only=True) == 0]
            if empty:
                if not messagebox.askyesno(
                    "No Indexed Documents",
                    f"No indexed documents for: {', '.join(empty)}.\n\n"
                    "Add docs and click Index, or generate without RAG context?"):
                    return
        else:
            # Single-section: existing logic (questions → sets when subtype set).
            n = compute_set_count(n_input, section, subtype)
            n = min(n, BULK_MAX_QUANTITY)
            if n < 1:
                return
            task_list = [section] * n

            if self.db.count(section, indexed_only=True) == 0:
                if not messagebox.askyesno("No Indexed Documents",
                    f"No indexed documents for {SECTIONS[section]}.\n\n"
                    "Add docs and click Index, or generate without RAG context?"):
                    return
```

- [ ] **Step 3: Use the user-configurable threshold for the cost-confirm dialog**

Continue in `_bulk_start`. Find this block (just below the no-RAG check):

```python
        # Cost preview + threshold check.
        llm    = self.settings.get("llm")
        verify = bool(self.settings.get("verify"))
        jury   = bool(self.settings.get("multi_judge"))
        low, high = estimate_bulk_cost(n, llm, multi_judge=jury, verify=verify)

        if high > BULK_COST_CONFIRM_THRESHOLD:
            ok = messagebox.askyesno(
                "Confirm bulk run",
                f"Estimated cost: ${low:.2f} - ${high:.2f}\n"
                f"Sets: {n}\n"
                f"Section: {SECTIONS[section]}\n"
                f"Model: {llm}\n\n"
                f"Continue?",
            )
            if not ok:
                return
```

Replace with (read the threshold from settings; tweak dialog wording for equate):

```python
        # Cost preview + threshold check.
        llm    = self.settings.get("llm")
        verify = bool(self.settings.get("verify"))
        jury   = bool(self.settings.get("multi_judge"))
        low, high = estimate_bulk_cost(n, llm, multi_judge=jury, verify=verify)

        threshold = float(self.settings.get("bulk_cost_confirm_threshold")
                            or BULK_COST_CONFIRM_THRESHOLD)
        if high > threshold:
            if equate:
                section_line = (f"Sections: {', '.join(EQUATE_SECTIONS)} "
                                f"({n // len(EQUATE_SECTIONS)} sets each)")
            else:
                section_line = f"Section: {SECTIONS[section]}"
            ok = messagebox.askyesno(
                "Confirm bulk run",
                f"Estimated cost: ${low:.2f} - ${high:.2f}\n"
                f"Sets: {n}\n"
                f"{section_line}\n"
                f"Model: {llm}\n\n"
                f"Continue?",
            )
            if not ok:
                return
```

- [ ] **Step 4: Replace the worker-launch block at the end of `_bulk_start`**

Continue in `_bulk_start`. Find the worker-launch block (the one you edited in Task 4):

```python
        # Build the task list. Single-section mode = [section] * n; equate
        # mode (added in a later task) will branch here to call
        # equate_task_list(n) instead.
        task_list = [section] * n

        self._bulk_stop.clear()
        self._bulk_thread = threading.Thread(
            target=self._bulk_worker, args=(task_list, hint, subtype), daemon=True
        )
        self._bulk_thread.start()
```

Replace with (the task list is now built earlier in the method, so just launch the worker):

```python
        # task_list was built above (either equate or single-section path).
        self._bulk_stop.clear()
        self._bulk_thread = threading.Thread(
            target=self._bulk_worker, args=(task_list, hint, subtype), daemon=True
        )
        self._bulk_thread.start()
```

- [ ] **Step 5: Compute `equate` early and override the Quantity label flip**

Open `ucat/ui.py`. Find `_bulk_inputs_changed` (around line 537). Find this block near the top:

```python
        section = self._bulk_sec.get()
        hint    = self._bulk_hint.get()

        # Resolve subtype: combobox stores the human label; we map back to the
        # internal storage value (e.g. 'venn'). 'Any (mixed)' → "" → None.
        label_to_value = {
            lbl: v for v, lbl in SUBTYPES_BY_SECTION.get(section, [])
        }
        chosen_label = self._bulk_subtype.get()
        subtype_value = label_to_value.get(chosen_label, "")
        subtype = subtype_value or None
```

Add `equate = self._bulk_equate.get()` at the top so it's available throughout the method:

```python
        section = self._bulk_sec.get()
        hint    = self._bulk_hint.get()
        equate  = self._bulk_equate.get()

        # Resolve subtype: combobox stores the human label; we map back to the
        # internal storage value (e.g. 'venn'). 'Any (mixed)' → "" → None.
        label_to_value = {
            lbl: v for v, lbl in SUBTYPES_BY_SECTION.get(section, [])
        }
        chosen_label = self._bulk_subtype.get()
        subtype_value = label_to_value.get(chosen_label, "")
        subtype = subtype_value or None
```

Find the Quantity label flip (around line 564):

```python
        # Flip the Quantity label.
        self._bulk_qty_lbl.config(text=("Questions:" if subtype else "Sets:"))
```

Replace with (force "Sets:" in equate mode regardless of any persisted subtype):

```python
        # Flip the Quantity label. In equate mode the label is always "Sets:"
        # since equate forces subtype to None even if a value is persisted.
        self._bulk_qty_lbl.config(
            text=("Sets:" if equate or not subtype else "Questions:"))
```

- [ ] **Step 6: Update the invalid-input bail block**

Continue in `_bulk_inputs_changed`. Find:

```python
        # Bail if quantity is invalid.
        if n_input is None or n_input < 1:
            self._bulk_yield_lbl.config(text="")
            max_input = (BULK_MAX_QUANTITY * SET_SIZES.get(section, 5)) if subtype \
                          else BULK_MAX_QUANTITY
            self._bulk_cost_lbl.config(
                text=f"Enter a number 1 - {max_input}.",
                fg=WARN,
            )
            self._bulk_start_btn.config(state="disabled")
            return
```

Replace with (use `BULK_MAX_QUANTITY` directly when equate is on, since the cap is per-section):

```python
        # Bail if quantity is invalid.
        if n_input is None or n_input < 1:
            self._bulk_yield_lbl.config(text="")
            if equate:
                max_input = BULK_MAX_QUANTITY  # per-section cap
            else:
                max_input = (BULK_MAX_QUANTITY * SET_SIZES.get(section, 5)) if subtype \
                              else BULK_MAX_QUANTITY
            self._bulk_cost_lbl.config(
                text=f"Enter a number 1 - {max_input}.",
                fg=WARN,
            )
            self._bulk_start_btn.config(state="disabled")
            return
```

Then find this block further down (computes `n_sets`):

```python
        # Compute set count and yield helper line.
        n_sets = compute_set_count(n_input, section, subtype)
        capped = False
        if n_sets > BULK_MAX_QUANTITY:
            n_sets = BULK_MAX_QUANTITY
            capped = True
```

Replace with (in equate mode, n_sets means per-section count × 4):

```python
        # Compute set count and yield helper line.
        capped = False
        if equate:
            # Input is per-section count; total n_sets = per_section × 4.
            per_section = min(n_input, BULK_MAX_QUANTITY)
            if per_section < n_input:
                capped = True
            n_sets = per_section * len(EQUATE_SECTIONS)
        else:
            n_sets = compute_set_count(n_input, section, subtype)
            if n_sets > BULK_MAX_QUANTITY:
                n_sets = BULK_MAX_QUANTITY
                capped = True
```

- [ ] **Step 7: Hide the yield helper line when equate is on**

Continue in `_bulk_inputs_changed`. Find this block:

```python
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
```

Replace the outer `if subtype:` condition with `if subtype and not equate:` so the yield line is also suppressed in equate mode:

```python
        if subtype and not equate:
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
```

- [ ] **Step 8: Update the cost banner suffix for equate mode**

Continue in `_bulk_inputs_changed`. Find the cost-preview block (the one that builds the banner text):

```python
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
```

Replace with (different suffix wording in equate mode; per-section breakdown is added in Task 8):

```python
        # Cost preview (always in sets — that's what's billed).
        llm     = self.settings.get("llm")
        verify  = bool(self.settings.get("verify"))
        jury    = bool(self.settings.get("multi_judge"))
        low, high = estimate_bulk_cost(n_sets, llm, multi_judge=jury, verify=verify)

        if equate:
            cap_suffix = ("  (capped at 100 per section — split into multiple "
                          "runs for more)") if capped else ""
            descriptor = (f"{n_sets // len(EQUATE_SECTIONS)} sets × "
                          f"{len(EQUATE_SECTIONS)} sections × {llm}{cap_suffix}")
        else:
            cap_suffix = "  (capped at the max — split into multiple runs for more)" \
                if capped else ""
            descriptor = f"{n_sets} sets × {llm}{cap_suffix}"

        self._bulk_cost_lbl.config(
            text=f"Estimated cost: ~${low:.2f} - ${high:.2f}   ({descriptor})",
            fg=ACCENT,
        )
```

- [ ] **Step 9: Manual smoke test**

Run: `./venv/bin/python ucat_trainer.py`
Open the Bulk tab. Confirm:
- Without equate ticked: cost banner reads `… (10 sets × claude-…)` as before.
- Tick equate, set quantity 5: banner reads `… (5 sets × 4 sections × claude-…)`.
- Set quantity 200 with equate on: banner reads `… (capped at 100 per section …)`.
- Quantity label always reads `Sets:` when equate is ticked, even if the underlying `bulk_subtype` setting was previously non-empty.
- If you have at least one section indexed and another not, ticking equate and starting a run shows a single dialog listing the empty sections.
- Click Start with quantity 1 and equate on: 4 rows seed in the Treeview, alternating VR/QR/SJT/DM. (Stop the run before it costs money if you're testing live.)

- [ ] **Step 10: Run all tests**

Run: `./venv/bin/python tests/test_equate_mode.py && ./venv/bin/python tests/test_subtype_targeting.py && ./venv/bin/python tests/test_bulk_cost.py`
Expected: All pass.

- [ ] **Step 11: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/ui.py
git commit -m "$(cat <<'EOF'
feat: equate path in _bulk_start + batched no-RAG dialog

When equate is on, _bulk_start builds a round-robin task list via
equate_task_list, applies the per-section cap, and shows a single
batched dialog listing every empty section instead of one prompt
per section. Cost banner suffix shows the "N sets × 4 sections"
form. Confirm-threshold now reads from settings.

Co-Authored-By: claude-flow <ruv@ruv.net>
EOF
)"
```

---

## Task 8: Per-section cost breakdown in the cost banner

**Files:**
- Modify: `ucat/ui.py` (add a second-line breakdown label, update `_bulk_inputs_changed`)

- [ ] **Step 1: Add the breakdown label widget in `_tab_bulk`**

Open `ucat/ui.py`. Find this block in `_tab_bulk` (around line 456):

```python
        # Cost preview banner.
        self._bulk_cost_lbl = tk.Label(
            p, text="", bg=BG, fg=ACCENT, font=FB, anchor="w"
        )
        self._bulk_cost_lbl.pack(anchor="w", pady=(2, 12))
```

Replace with (add a sibling label for the per-section breakdown):

```python
        # Cost preview banner.
        self._bulk_cost_lbl = tk.Label(
            p, text="", bg=BG, fg=ACCENT, font=FB, anchor="w"
        )
        self._bulk_cost_lbl.pack(anchor="w", pady=(2, 0))

        # Per-section breakdown (equate mode only — empty otherwise).
        self._bulk_breakdown_lbl = tk.Label(
            p, text="", bg=BG, fg=MUTED, font=FS, anchor="w"
        )
        self._bulk_breakdown_lbl.pack(anchor="w", pady=(0, 12))
```

- [ ] **Step 2: Render the breakdown when equate is on**

Open `ucat/ui.py`. Find the cost-preview block in `_bulk_inputs_changed` (the one you edited in Task 7). After the `self._bulk_cost_lbl.config(...)` call (right before the next block that re-enables the start button), add:

```python
        # Per-section breakdown (equate mode only).
        if equate and not capped:
            per_section = n_sets // len(EQUATE_SECTIONS)
            parts = []
            for s in EQUATE_SECTIONS:
                _l, h = estimate_section_cost(
                    s, per_section, llm, multi_judge=jury, verify=verify)
                parts.append(f"{s} ~${h:.2f}")
            self._bulk_breakdown_lbl.config(
                text="↳ " + "  ·  ".join(parts),
            )
        else:
            self._bulk_breakdown_lbl.config(text="")
```

(Note: when `capped` is True the per-section breakdown would mislead — the banner already shows the cap note, and the breakdown is suppressed for clarity. The user can lower the input below the cap to see the breakdown.)

Also, in the **invalid-input early-return branch** of the same method, clear the breakdown:

Find:

```python
        # Bail if quantity is invalid.
        equate = self._bulk_equate.get()
        if n_input is None or n_input < 1:
            self._bulk_yield_lbl.config(text="")
            if equate:
                max_input = BULK_MAX_QUANTITY  # per-section cap
            else:
                max_input = (BULK_MAX_QUANTITY * SET_SIZES.get(section, 5)) if subtype \
                              else BULK_MAX_QUANTITY
            self._bulk_cost_lbl.config(
                text=f"Enter a number 1 - {max_input}.",
                fg=WARN,
            )
            self._bulk_start_btn.config(state="disabled")
            return
```

Add `self._bulk_breakdown_lbl.config(text="")` immediately before `self._bulk_start_btn.config(state="disabled")`:

```python
        # Bail if quantity is invalid.
        equate = self._bulk_equate.get()
        if n_input is None or n_input < 1:
            self._bulk_yield_lbl.config(text="")
            if equate:
                max_input = BULK_MAX_QUANTITY  # per-section cap
            else:
                max_input = (BULK_MAX_QUANTITY * SET_SIZES.get(section, 5)) if subtype \
                              else BULK_MAX_QUANTITY
            self._bulk_cost_lbl.config(
                text=f"Enter a number 1 - {max_input}.",
                fg=WARN,
            )
            self._bulk_breakdown_lbl.config(text="")
            self._bulk_start_btn.config(state="disabled")
            return
```

- [ ] **Step 3: Manual smoke test**

Run: `./venv/bin/python ucat_trainer.py`
Open the Bulk tab. Confirm:
- Without equate: breakdown line is empty (no second line under the cost banner).
- Tick equate, set quantity 25: a second line appears reading `↳ VR ~$0.??  ·  QR ~$0.??  ·  SJT ~$0.??  ·  DM ~$0.??`.
- Sum of the four figures roughly matches the total in the line above (within rounding).
- Toggle verify or jury — both lines update.
- Untick equate: breakdown line disappears.

- [ ] **Step 4: Run all tests**

Run: `./venv/bin/python tests/test_equate_mode.py && ./venv/bin/python tests/test_subtype_targeting.py && ./venv/bin/python tests/test_bulk_cost.py`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/ui.py
git commit -m "$(cat <<'EOF'
feat: per-section cost breakdown in equate mode

A second-line label below the main cost banner shows
'↳ VR ~$x  ·  QR ~$y  ·  SJT ~$z  ·  DM ~$w' when equate is on.
Sums to the total within rounding (multipliers normalised to 4.0
across EQUATE_SECTIONS).

Co-Authored-By: claude-flow <ruv@ruv.net>
EOF
)"
```

---

## Task 9: Configurable cost-confirm threshold entry

**Files:**
- Modify: `ucat/ui.py` (add entry widget in `_tab_bulk`, plumbing in `_bulk_inputs_changed`)

- [ ] **Step 1: Add the threshold entry widget**

Open `ucat/ui.py`. Find the cost-preview banner block (the one you just edited in Task 8). Immediately **after** the `self._bulk_breakdown_lbl.pack(...)` line, add:

```python
        # Configurable cost-confirm threshold. Set high (e.g. 999) to silence
        # the dialog; set low (e.g. 0) to confirm every run.
        tr = tk.Frame(p, bg=BG); tr.pack(anchor="w", pady=(0, 12))
        tk.Label(tr, text="Confirm above: $", bg=BG, fg=MUTED, font=FS
                 ).pack(side="left")
        self._bulk_threshold_var = tk.StringVar(
            value=f"{float(self.settings.get('bulk_cost_confirm_threshold') or BULK_COST_CONFIRM_THRESHOLD):.2f}"
        )
        threshold_entry = tk.Entry(
            tr, textvariable=self._bulk_threshold_var,
            bg=PANEL2, fg=TEXT, font=FS,
            insertbackground=ACCENT, relief="flat", width=8,
        )
        threshold_entry.pack(side="left")
        threshold_entry.bind("<FocusOut>", lambda _e: self._bulk_threshold_finalised())
        threshold_entry.bind("<Return>",   lambda _e: self._bulk_threshold_finalised())
```

- [ ] **Step 2: Add the validation/persistence callback**

Open `ucat/ui.py`. Find the `_bulk_equate_changed` method (added in Task 6). Immediately **after** it, add:

```python
    def _bulk_threshold_finalised(self):
        """Validate the cost-confirm threshold entry on focus-out / Return.
        Reverts to the last good value on invalid input."""
        raw = self._bulk_threshold_var.get().strip()
        try:
            value = float(raw)
            if value < 0:
                raise ValueError("threshold must be non-negative")
        except ValueError:
            # Revert: read the last good value back into the entry.
            current = float(self.settings.get("bulk_cost_confirm_threshold")
                              or BULK_COST_CONFIRM_THRESHOLD)
            self._bulk_threshold_var.set(f"{current:.2f}")
            return
        self.settings.set("bulk_cost_confirm_threshold", value)
        self._bulk_threshold_var.set(f"{value:.2f}")
```

- [ ] **Step 3: Verify the threshold is honoured on Start**

This was already wired in Task 7 step 3 — `_bulk_start` reads `self.settings.get("bulk_cost_confirm_threshold")`. No change needed here; just confirm the line still reads:

```python
        threshold = float(self.settings.get("bulk_cost_confirm_threshold")
                            or BULK_COST_CONFIRM_THRESHOLD)
```

- [ ] **Step 4: Manual smoke test**

Run: `./venv/bin/python ucat_trainer.py`
Open the Bulk tab. Confirm:
- Default value is `5.00` for fresh installs.
- Type `12.5` and Tab away: persists; quit and relaunch — entry shows `12.50`.
- Type `not-a-number` and Tab away: reverts to last good value.
- Type `-1` and Tab away: reverts.
- Type `0` and Tab away: persists as `0.00`. Set quantity to 1 — clicking Start now triggers the confirm dialog even for tiny estimates.
- Type `999` and Tab away: confirm dialog never fires.

- [ ] **Step 5: Run all tests**

Run: `./venv/bin/python tests/test_equate_mode.py && ./venv/bin/python tests/test_subtype_targeting.py && ./venv/bin/python tests/test_bulk_cost.py`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/ui.py
git commit -m "$(cat <<'EOF'
feat: user-configurable cost-confirm threshold entry

Numeric entry near the cost banner, validated on focus-out / Return.
Replaces the hardcoded $5 BULK_COST_CONFIRM_THRESHOLD default with a
persisted user setting. Set to 999 to silence the dialog; set to 0
to confirm every run. Invalid input reverts to last good value.

Co-Authored-By: claude-flow <ruv@ruv.net>
EOF
)"
```

---

## Task 10: Live "spent so far" label

**Files:**
- Modify: `ucat/ui.py` (new label widget, helper, hooks in `_bulk_run_started` / `_bulk_after_success` / `_bulk_run_finished`)

- [ ] **Step 1: Add the live-spent label widget**

Open `ucat/ui.py`. Find the threshold row block from Task 9 (`tr = tk.Frame(p, bg=BG); tr.pack(...)`). Immediately **after** it, add:

```python
        # Live "spent so far" label. Empty unless a bulk run is in progress
        # (or just finished — _bulk_run_finished hides it then).
        self._bulk_live_spent_lbl = tk.Label(
            p, text="", bg=BG, fg=MUTED, font=FS, anchor="w"
        )
        self._bulk_live_spent_lbl.pack(anchor="w", pady=(0, 8))
```

- [ ] **Step 2: Track the estimate-high in a stateful field**

Open `ucat/ui.py`. Find the `__init__` method's bulk-run-state block (around line 123):

```python
        # Bulk-run state.
        self._bulk_stop: threading.Event = threading.Event()
        self._bulk_thread: Optional[threading.Thread] = None
        self._bulk_rows: List[Dict[str, Any]] = []
        self._bulk_started_at: Optional[float] = None
        self._bulk_run_cost: float = 0.0  # accumulated USD for the active run
```

Add one new field at the end of the block:

```python
        # Bulk-run state.
        self._bulk_stop: threading.Event = threading.Event()
        self._bulk_thread: Optional[threading.Thread] = None
        self._bulk_rows: List[Dict[str, Any]] = []
        self._bulk_started_at: Optional[float] = None
        self._bulk_run_cost: float = 0.0  # accumulated USD for the active run
        self._bulk_last_estimate_high: float = 0.0  # cost-banner upper bound at run start
```

- [ ] **Step 3: Capture the estimate at run start**

Open `ucat/ui.py`. Find `_bulk_run_started` (the version edited in Task 4). Find:

```python
        _, est_high = estimate_bulk_cost(n, llm, multi_judge=jury, verify=verify)
```

Immediately after that line, add:

```python
        self._bulk_last_estimate_high = est_high
        self._bulk_update_spent()  # show "Spent: $0.00 / ~$X.XX est (0%)" immediately
```

- [ ] **Step 4: Add the `_bulk_update_spent` helper**

Open `ucat/ui.py`. Find the `_bulk_after_success` method (around line 917). Immediately **before** it, add:

```python
    def _bulk_update_spent(self):
        """Refresh the live-spent label. No-op if no run is active."""
        if self._bulk_thread is None or not self._bulk_thread.is_alive():
            self._bulk_live_spent_lbl.config(text="")
            return
        spent    = self._bulk_run_cost
        est_high = self._bulk_last_estimate_high or 0.0
        pct = int(round(100 * spent / est_high)) if est_high > 0 else 0
        self._bulk_live_spent_lbl.config(
            text=f"Spent: ${spent:.2f} / ~${est_high:.2f} est ({pct}%)",
        )
```

- [ ] **Step 5: Hook the helper into cost-accumulating callbacks**

Open `ucat/ui.py`. In `_bulk_after_success` (around line 917), find the `self._cost_lbl.config(...)` call near the end. Immediately **after** it, add:

```python
        self._bulk_update_spent()
```

In `_bulk_verify_complete` (around line 887), find the `self._cost_lbl.config(...)` call near the end. Immediately **after** it, add:

```python
        self._bulk_update_spent()
```

- [ ] **Step 6: Hide the label on run finish**

Open `ucat/ui.py`. Find `_bulk_run_finished` (around line 854). At the very end of the method (after `self._bulk_inputs_changed()`), add:

```python
        self._bulk_live_spent_lbl.config(text="")
```

- [ ] **Step 7: Manual smoke test**

Run: `./venv/bin/python ucat_trainer.py`
Open the Bulk tab. With API keys configured:
- Set quantity to 2, click Start. Live-spent label appears immediately reading `Spent: $0.00 / ~$0.X est (0%)`.
- After the first set completes, label updates (e.g. `Spent: $0.07 / ~$0.18 est (39%)`).
- Run finishes — live-spent label disappears.
- Without API keys: label still appears with `$0.00` and stays until the run errors and `_bulk_run_finished` fires.

- [ ] **Step 8: Run all tests**

Run: `./venv/bin/python tests/test_equate_mode.py && ./venv/bin/python tests/test_subtype_targeting.py && ./venv/bin/python tests/test_bulk_cost.py`
Expected: All pass.

- [ ] **Step 9: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/ui.py
git commit -m "$(cat <<'EOF'
feat: live spent-so-far label during bulk runs

Shows 'Spent: $X.XX / ~$Y.YY est (Z%)' under the cost banner during
an active run; updates after each set's verdict + cost are written.
Hidden when no run is active. Compares against the upper-bound
estimate captured at run start.

Co-Authored-By: claude-flow <ruv@ruv.net>
EOF
)"
```

---

## Task 11: Post-run actual-vs-estimate summary line

**Files:**
- Modify: `ucat/ui.py` (`_bulk_run_finished` — refine summary text)

- [ ] **Step 1: Replace the summary text in `_bulk_run_finished`**

Open `ucat/ui.py`. Find `_bulk_run_finished` (around line 854). Find this block:

```python
        if stopped:
            tail = f"Stopped at {succeeded + failed} / {n}."
        else:
            drift_note = f" ({drift_count} with subtype drift)" if drift_count else ""
            tail = f"Bulk run finished: {succeeded} succeeded{drift_note}, {failed} failed"
            if skipped: tail += f", {skipped} skipped"
            tail += "."
        self._bulk_progress_lbl.config(text=tail)
        self._status(tail)
```

Replace with (adds the actual / estimate cost figures to both stop and done variants):

```python
        actual = self._bulk_run_cost
        est    = self._bulk_last_estimate_high or 0.0
        cost_tail = f"  Actual: ${actual:.2f} (est ${est:.2f})"
        if stopped:
            done_count = succeeded + failed
            tail = f"Stopped at {done_count} / {n} sets.{cost_tail}"
        else:
            drift_note = f" ({drift_count} with subtype drift)" if drift_count else ""
            tail = f"Bulk run finished: {succeeded} succeeded{drift_note}, {failed} failed"
            if skipped: tail += f", {skipped} skipped"
            tail += "."
            tail += cost_tail
        self._bulk_progress_lbl.config(text=tail)
        self._status(tail)
```

- [ ] **Step 2: Manual smoke test**

Run: `./venv/bin/python ucat_trainer.py`
Open the Bulk tab. With API keys:
- Run a 2-set bulk run to completion. Final progress label reads e.g. `Bulk run finished: 2 succeeded, 0 failed.  Actual: $0.18 (est $0.20)`.
- Start a 5-set run, click Stop after 2 sets complete. Progress label reads e.g. `Stopped at 2 / 5 sets.  Actual: $0.18 (est $0.50)`.
- Start a new run — `_bulk_seed_rows` clears the progress label to `0 / N`.

- [ ] **Step 3: Run all tests**

Run: `./venv/bin/python tests/test_equate_mode.py && ./venv/bin/python tests/test_subtype_targeting.py && ./venv/bin/python tests/test_bulk_cost.py`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/ui.py
git commit -m "$(cat <<'EOF'
feat: post-run actual-vs-estimate summary line

After a bulk run finishes (or is stopped), the progress label includes
the actual accumulated cost alongside the estimate captured at run
start: 'Actual: $X.XX (est $Y.YY)'. Helpful for budget tracking and
for seeing where the estimator under/overshoots.

Co-Authored-By: claude-flow <ruv@ruv.net>
EOF
)"
```

---

## Task 12: Telemetry fields for equate runs

**Files:**
- Modify: `ucat/ui.py` (`_bulk_run_started`, `_bulk_run_finished` — extend `emit` calls)

- [ ] **Step 1: Extend the `bulk_run_start` event**

Open `ucat/ui.py`. Find `_bulk_run_started` (the version edited in Tasks 4 + 10). Find the `emit("bulk_run_start", ...)` call:

```python
        emit("bulk_run_start",
             section=task_list[0] if task_list else self._bulk_sec.get(),
             n=n,
             model=llm,
             verify=verify,
             multi_judge=jury,
             estimated_cost_high=round(est_high, 4),
             subtype=(self.settings.get("bulk_subtype") or None))
```

Replace with (add `equate_mode` and `task_list_length`):

```python
        equate_mode = bool(self._bulk_equate.get())
        emit("bulk_run_start",
             section=task_list[0] if task_list else self._bulk_sec.get(),
             n=n,
             model=llm,
             verify=verify,
             multi_judge=jury,
             estimated_cost_high=round(est_high, 4),
             subtype=(self.settings.get("bulk_subtype") or None),
             equate_mode=equate_mode,
             task_list_length=len(task_list))
```

- [ ] **Step 2: Extend the `bulk_run_end` event**

Open `ucat/ui.py`. Find `_bulk_run_finished`. Find the `emit("bulk_run_end", ...)` call:

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

Replace with (add `equate_mode` and `estimated_cost_high`):

```python
        emit("bulk_run_end",
             section=self._bulk_sec.get(),
             n=n,
             succeeded=succeeded,
             failed=failed,
             stopped=stopped,
             actual_cost_usd=round(self._bulk_run_cost, 4),
             estimated_cost_high=round(self._bulk_last_estimate_high, 4),
             duration_s=round(elapsed, 1),
             subtype=(self.settings.get("bulk_subtype") or None),
             drift_count=drift_count,
             equate_mode=bool(self._bulk_equate.get()))
```

- [ ] **Step 3: Manual smoke test**

Run: `./venv/bin/python ucat_trainer.py`
Open the Bulk tab. With API keys, tick equate, set quantity 1, click Start. After the run finishes, inspect `ucat_telemetry.jsonl`:

```bash
tail -10 ucat_telemetry.jsonl | grep -E "bulk_run_(start|end)"
```

Expected: the `bulk_run_start` event has `"equate_mode": true, "task_list_length": 4`. The `bulk_run_end` event has `"equate_mode": true, "estimated_cost_high": 0.X`.

Repeat with equate unticked: events have `"equate_mode": false, "task_list_length": <n>`.

- [ ] **Step 4: Run all tests**

Run: `./venv/bin/python tests/test_equate_mode.py && ./venv/bin/python tests/test_subtype_targeting.py && ./venv/bin/python tests/test_bulk_cost.py`
Expected: All pass.

- [ ] **Step 5: Final smoke test (full equate run end-to-end)**

Run: `./venv/bin/python ucat_trainer.py`
With API keys + at least one section indexed:

1. Open Bulk tab.
2. Tick equate.
3. Set quantity to 2.
4. Verify cost banner reads `… (2 sets × 4 sections × claude-…)` and the breakdown line shows VR/QR/SJT/DM ~$.
5. Set the threshold entry to `999`. Click Start — no confirm dialog fires.
6. Treeview seeds 8 rows, alternating VR, QR, SJT, DM, VR, QR, SJT, DM.
7. As rows complete, Section column shows the correct code; live-spent label updates.
8. Run finishes; progress label shows `Bulk run finished: 8 succeeded, 0 failed.  Actual: $X.XX (est $Y.YY)`.
9. Untick equate — radios re-enable, subtype row reappears, breakdown line vanishes.
10. Quit and relaunch — `bulk_equate=False`, `bulk_cost_confirm_threshold=999.00` both restored.

- [ ] **Step 6: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/ui.py
git commit -m "$(cat <<'EOF'
feat: telemetry fields for equate runs

bulk_run_start now emits equate_mode + task_list_length;
bulk_run_end emits equate_mode + estimated_cost_high. Useful for
analysing the actual-vs-estimate gap and refining
SECTION_COST_MULTIPLIERS from real per-section data.

Co-Authored-By: claude-flow <ruv@ruv.net>
EOF
)"
```

---

## Final Verification

After all 12 tasks are committed:

- [ ] Run the full test suite:
  ```bash
  cd "/Users/ibutlerking/Documents/question generation app"
  for f in tests/test_*.py; do echo "=== $f ==="; ./venv/bin/python "$f" || break; done
  ```
  Expected: every file ends with `All tests passed.`

- [ ] Skim the spec (`docs/superpowers/specs/2026-04-28-bulk-equate-mode-design.md`) and confirm every section has a corresponding task / commit:
  - §4.1 Equate checkbox → Task 6
  - §4.2 Configurable threshold entry → Task 9
  - §4.3 Per-section breakdown → Task 8
  - §4.4 Live spent + post-run summary → Tasks 10, 11
  - §4.5 Treeview Section column → Task 5
  - §5.1 EQUATE_SECTIONS → Task 1
  - §5.2 equate_task_list → Task 1
  - §5.3 _bulk_start branch → Task 7
  - §5.4 _bulk_worker refactor → Task 4
  - §5.5 Per-section cap → Task 7
  - §5.6 Stop semantics → Task 4 (round-robin invariant via task_list iteration)
  - §6.1 SECTION_COST_MULTIPLIERS → Task 2
  - §6.2 Live spent label → Task 10
  - §6.3 Post-run summary → Task 11
  - §6.4 Configurable threshold → Tasks 7, 9
  - §7 Settings additions → Task 3
  - §8 Telemetry fields → Task 12

- [ ] Spot-check `git log --oneline` shows 12 distinct commits in this feature.
