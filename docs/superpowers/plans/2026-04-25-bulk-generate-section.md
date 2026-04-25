# Bulk Generate (Section-level) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `BULK` tab to the UCAT Trainer that runs N sequential question-set generations for a chosen section, with per-row progress, retry-on-error, immediate Stop, and a cost preview + confirmation modal.

**Architecture:** A thin background worker thread calls the existing `RAGEngine.generate()` in a loop. No engine changes. New UI tab + cost estimator + settings keys + telemetry events. Sequential only (no parallelism).

**Tech Stack:** Python 3.13, Tkinter (stdlib UI), threading (stdlib), existing Anthropic + Voyage SDK calls inside `RAGEngine`. Tests use stdlib-only assertions runnable via `python tests/<file>.py` (no pytest dependency).

**Spec reference:** [docs/superpowers/specs/2026-04-25-bulk-generate-section-design.md](../specs/2026-04-25-bulk-generate-section-design.md)

> Supersedes the earlier plan at `2026-04-25-bulk-generate.md` (which targeted the older subtype-based design with batch tables and HTML export — both explicitly out-of-scope here).

---

## File map

| File | Status | Responsibility |
|---|---|---|
| `ucat/config.py` | modify | Add constants `BULK_MAX_QUANTITY`, `BULK_COST_CONFIRM_THRESHOLD`; add 3 keys to `Settings.DEFAULTS`; add `estimate_bulk_cost()` pure function. |
| `ucat/ui.py` | modify | Add `_tab_bulk()` builder; add `_bulk_*` worker / row / start / stop methods; register new tab in `_ui()`; wire concurrent-run guards on existing single-shot path; extend `on_close` to join the bulk thread. |
| `tests/test_bulk_cost.py` | create | Stdlib-only assertions for `estimate_bulk_cost()`. Runnable as `python tests/test_bulk_cost.py`. |
| `ucat/rag.py` | unchanged | — |
| `ucat/telemetry.py` | unchanged | `emit()` already accepts arbitrary event names. |

---

## Task 1: Add config constants and settings defaults

**Files:**
- Modify: `ucat/config.py`

- [ ] **Step 1: Add the two bulk constants near the existing retrieval defaults**

Open `ucat/config.py`. Find the `# ─── Retrieval / generation defaults ───` block ending at the line `EMBED_BATCH_SIZE = 64`. Immediately after that line, add a new block:

```python
# ─── Bulk generation defaults ────────────────────────────────────────────────

BULK_MAX_QUANTITY           = 100      # hard cap on a single bulk run
BULK_COST_CONFIRM_THRESHOLD = 5.00     # USD; above this, ask before launching
```

- [ ] **Step 2: Add three keys to `Settings.DEFAULTS`**

In `ucat/config.py`, find the `class Settings` definition. Inside `DEFAULTS`, add three new entries after the existing `"multi_judge"` line:

```python
        "verify":           DEFAULT_VERIFY,
        "multi_judge":      DEFAULT_MULTI_JUDGE,
        "bulk_section":     "VR",
        "bulk_quantity":    10,
        "bulk_hint":        "",
    }
```

- [ ] **Step 3: Verify import still works**

Run: `cd "/Users/ibutlerking/Documents/question generation app" && ./venv/bin/python -c "from ucat.config import BULK_MAX_QUANTITY, BULK_COST_CONFIRM_THRESHOLD, Settings; s = Settings(); print(BULK_MAX_QUANTITY, BULK_COST_CONFIRM_THRESHOLD, s.get('bulk_section'), s.get('bulk_quantity'), repr(s.get('bulk_hint')))"`

Expected output: `100 5.0 VR 10 ''`

- [ ] **Step 4: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/config.py
git commit -m "config: add bulk-generate constants and settings defaults"
```

---

## Task 2: Cost estimator (TDD)

**Files:**
- Create: `tests/test_bulk_cost.py`
- Modify: `ucat/config.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_bulk_cost.py` with the following content. These tests rely only on stdlib + the project's `ucat.config` module.

```python
"""Tests for estimate_bulk_cost. Runnable directly:

    ./venv/bin/python tests/test_bulk_cost.py

Each function with a name starting `test_` is run; failures raise.
"""
from __future__ import annotations

import sys
import os

# Make the project importable when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ucat.config import estimate_bulk_cost, MODEL_COSTS


def _approx_eq(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def test_zero_count_returns_zero():
    low, high = estimate_bulk_cost(0, "claude-opus-4-7", multi_judge=False, verify=True)
    assert low == 0.0 and high == 0.0, f"expected (0,0), got ({low},{high})"


def test_scales_linearly_in_count():
    low1,  high1  = estimate_bulk_cost(1,  "claude-opus-4-7", multi_judge=False, verify=True)
    low10, high10 = estimate_bulk_cost(10, "claude-opus-4-7", multi_judge=False, verify=True)
    assert _approx_eq(low10,  10 * low1),  f"low not linear: {low10} vs {10*low1}"
    assert _approx_eq(high10, 10 * high1), f"high not linear: {high10} vs {10*high1}"


def test_high_is_at_least_low():
    low, high = estimate_bulk_cost(5, "claude-opus-4-7", multi_judge=False, verify=True)
    assert high >= low, f"high ({high}) must be >= low ({low})"


def test_verify_disabled_costs_less():
    low_v,  _ = estimate_bulk_cost(5, "claude-opus-4-7", multi_judge=False, verify=True)
    low_nv, _ = estimate_bulk_cost(5, "claude-opus-4-7", multi_judge=False, verify=False)
    assert low_nv < low_v, f"verify=False should be cheaper (got {low_nv} >= {low_v})"


def test_jury_costs_more_than_single_judge():
    low_single, _ = estimate_bulk_cost(5, "claude-opus-4-7", multi_judge=False, verify=True)
    low_jury,   _ = estimate_bulk_cost(5, "claude-opus-4-7", multi_judge=True,  verify=True)
    assert low_jury > low_single, f"jury should cost more (got {low_jury} <= {low_single})"


def test_haiku_cheaper_than_opus():
    low_h, _ = estimate_bulk_cost(5, "claude-haiku-4-5",  multi_judge=False, verify=True)
    low_o, _ = estimate_bulk_cost(5, "claude-opus-4-7",   multi_judge=False, verify=True)
    assert low_h < low_o, f"haiku should be cheaper than opus (got {low_h} >= {low_o})"


def test_jury_keeps_high_above_low():
    """Regression: an earlier draft set per_high = per_low under multi_judge.
    Ensure high remains >= low when jury is on."""
    low, high = estimate_bulk_cost(10, "claude-opus-4-7", multi_judge=True, verify=True)
    assert high >= low, f"high ({high}) must be >= low ({low}) with jury enabled"


def test_unknown_model_raises_keyerror():
    raised = False
    try:
        estimate_bulk_cost(1, "model-that-does-not-exist", multi_judge=False, verify=True)
    except KeyError:
        raised = True
    assert raised, "expected KeyError for unknown model"


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

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd "/Users/ibutlerking/Documents/question generation app" && ./venv/bin/python tests/test_bulk_cost.py`

Expected: `ImportError: cannot import name 'estimate_bulk_cost' from 'ucat.config'` (the function doesn't exist yet).

- [ ] **Step 3: Implement `estimate_bulk_cost` in `ucat/config.py`**

In `ucat/config.py`, append a new function at the bottom of the file (after `api_status`):

```python
# ─── Bulk cost estimator ──────────────────────────────────────────────────────

def estimate_bulk_cost(
    n: int,
    llm: str,
    *,
    multi_judge: bool,
    verify: bool,
) -> tuple[float, float]:
    """Estimate total USD cost for an N-set bulk run.

    Returns ``(low, high)`` where:
      • low  assumes generation input tokens hit the prompt cache (typical after
        the first iteration).
      • high assumes a cold cache for every iteration (worst case).

    Token assumptions reflect telemetry observations from single-shot runs:
      gen:    ~3000 input + ~2000 output
      verify: ~1500 input + ~600 output  (Haiku, single-judge)
      jury:   add Sonnet pass + Opus second-pass at similar token shapes
    """
    if n <= 0:
        return 0.0, 0.0

    costs = MODEL_COSTS[llm]
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
        jury_per = ((1500 * sonnet["in"] + 600 * sonnet["out"]) / 1_000_000
                  + (1500 * opus["in"]   + 600 * opus["out"])   / 1_000_000)
        per_low  += jury_per
        per_high += jury_per

    return n * per_low, n * per_high
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd "/Users/ibutlerking/Documents/question generation app" && ./venv/bin/python tests/test_bulk_cost.py`

Expected: `All tests passed.` and exit code 0.

- [ ] **Step 5: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/config.py tests/test_bulk_cost.py
git commit -m "config: add estimate_bulk_cost with tests"
```

---

## Task 3: Bulk tab — UI skeleton (layout only, no behaviour)

**Files:**
- Modify: `ucat/ui.py`

This task lands the static layout. All interactive behaviour (Start, Stop, validation, banner update, worker) comes in later tasks. Goal: when you launch the app, the Bulk tab renders correctly with placeholder values.

- [ ] **Step 1: Add new imports**

In `ucat/ui.py`, find the existing import block at the top. Update the line:

```python
import io
import json
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional
```

to also import `time` and `uuid`:

```python
import io
import json
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
```

And update the `from .config import` block to include the new symbols:

```python
from .config import (APP_TITLE, LLM_CHOICES, EMBED_CHOICES, SECTIONS, SECTION_COLORS,
                      DEFAULT_LLM, DEFAULT_EMBED, IRT_BANDS, api_status, Settings,
                      BULK_MAX_QUANTITY, BULK_COST_CONFIRM_THRESHOLD, estimate_bulk_cost)
```

- [ ] **Step 2: Initialise bulk state in `App.__init__`**

In `ucat/ui.py`, find `class App(tk.Tk):` → `__init__`. Locate the block:

```python
        self._last_data: Optional[Dict[str, Any]] = None
        self._last_section: Optional[str] = None
        self._last_retrieved: List = []
        self._sel_gen_id: Optional[int] = None
        self._session_cost = 0.0
        self._session_tokens = {"in": 0, "out": 0, "cache_r": 0, "cache_w": 0}
        # Keep refs to PhotoImages so Tk doesn't GC them.
        self._photo_refs: List[Any] = []
```

Append, immediately after `self._photo_refs`:

```python
        # Bulk-run state.
        self._bulk_stop: threading.Event = threading.Event()
        self._bulk_thread: Optional[threading.Thread] = None
        self._bulk_rows: List[Dict[str, Any]] = []
        self._bulk_started_at: Optional[float] = None
        self._bulk_run_cost: float = 0.0  # accumulated USD for the active run
```

- [ ] **Step 3: Register the new tab in `_ui()`**

In `ucat/ui.py`, find `_ui` method, locate the block creating tab frames:

```python
        self._t_gen      = tk.Frame(self._nb, bg=BG)
        self._t_kb       = tk.Frame(self._nb, bg=BG)
        self._t_out      = tk.Frame(self._nb, bg=BG)
        self._t_insights = tk.Frame(self._nb, bg=BG)
        self._nb.add(self._t_gen,      text="  ⚡  GENERATE  ")
        self._nb.add(self._t_kb,       text="  🗄  KNOWLEDGE BASE  ")
        self._nb.add(self._t_out,      text="  📋  HISTORY  ")
        self._nb.add(self._t_insights, text="  📊  INSIGHTS  ")
        self._tab_gen()
        self._tab_kb()
        self._tab_out()
        self._tab_insights()
```

Replace it with:

```python
        self._t_gen      = tk.Frame(self._nb, bg=BG)
        self._t_bulk     = tk.Frame(self._nb, bg=BG)
        self._t_kb       = tk.Frame(self._nb, bg=BG)
        self._t_out      = tk.Frame(self._nb, bg=BG)
        self._t_insights = tk.Frame(self._nb, bg=BG)
        self._nb.add(self._t_gen,      text="  ⚡  GENERATE  ")
        self._nb.add(self._t_bulk,     text="  ⚡⚡ BULK  ")
        self._nb.add(self._t_kb,       text="  🗄  KNOWLEDGE BASE  ")
        self._nb.add(self._t_out,      text="  📋  HISTORY  ")
        self._nb.add(self._t_insights, text="  📊  INSIGHTS  ")
        self._tab_gen()
        self._tab_bulk()
        self._tab_kb()
        self._tab_out()
        self._tab_insights()
```

- [ ] **Step 4: Implement `_tab_bulk` builder method**

Add this method to `class App`, immediately after the `_tab_gen` method ends (before `_tab_kb`):

```python
    # ── Bulk tab ──────────────────────────────────────────────────────────────

    def _tab_bulk(self):
        p = tk.Frame(self._t_bulk, bg=BG)
        p.pack(fill="both", expand=True, padx=24, pady=20)

        tk.Label(p, text="Bulk Generate", bg=BG, fg=TEXT, font=FT).pack(anchor="w")
        tk.Label(p,
                 text="Generate multiple question sets in sequence. Results land "
                      "in History — promote good ones to the KB from there.",
                 bg=BG, fg=MUTED, font=FS, wraplength=1100, justify="left"
                 ).pack(anchor="w", pady=(2, 14))

        # Section radios.
        sr = tk.Frame(p, bg=BG); sr.pack(anchor="w", pady=(0, 10))
        tk.Label(sr, text="Section:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self._bulk_sec = tk.StringVar(value=self.settings.get("bulk_section"))
        for code in SECTIONS:
            tk.Radiobutton(sr, text=f" {code} ", variable=self._bulk_sec, value=code,
                           bg=BG, fg=TEXT, selectcolor=PANEL, activebackground=BG,
                           activeforeground=ACCENT, font=FB, indicatoron=False,
                           relief="flat", bd=1, padx=12, pady=6, cursor="hand2",
                           command=self._bulk_inputs_changed
                           ).pack(side="left", padx=4)

        # Quantity + topic hint row.
        qr = tk.Frame(p, bg=BG); qr.pack(fill="x", pady=(0, 10))
        tk.Label(qr, text="Quantity:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self._bulk_qty = tk.StringVar(value=str(self.settings.get("bulk_quantity")))
        qty_entry = tk.Entry(qr, textvariable=self._bulk_qty, bg=PANEL2, fg=TEXT, font=FM,
                              insertbackground=ACCENT, relief="flat", width=6)
        qty_entry.pack(side="left")
        self._bulk_qty.trace_add("write", lambda *_: self._bulk_inputs_changed())

        tk.Label(qr, text="   Topic hint:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(20, 14))
        self._bulk_hint = tk.StringVar(value=self.settings.get("bulk_hint"))
        tk.Entry(qr, textvariable=self._bulk_hint, bg=PANEL2, fg=TEXT, font=FM,
                 insertbackground=ACCENT, relief="flat", width=46).pack(side="left", fill="x", expand=True)
        self._bulk_hint.trace_add("write", lambda *_: self._bulk_inputs_changed())

        # Cost preview banner.
        self._bulk_cost_lbl = tk.Label(
            p, text="", bg=BG, fg=ACCENT, font=FB, anchor="w"
        )
        self._bulk_cost_lbl.pack(anchor="w", pady=(2, 12))

        # Action row.
        ar = tk.Frame(p, bg=BG); ar.pack(anchor="w", pady=(0, 10))
        self._bulk_start_btn = mkbtn(
            ar, "⚡  START BULK RUN", self._bulk_start,
            padx=22, pady=10, font=("Courier New", 12, "bold")
        )
        self._bulk_start_btn.pack(side="left", padx=(0, 12))
        self._bulk_stop_btn = mkbtn(
            ar, "⏹  STOP", self._bulk_stop_clicked,
            bg=DANGER, padx=18, pady=10, state="disabled"
        )
        self._bulk_stop_btn.pack(side="left", padx=(0, 14))
        self._bulk_progress_lbl = tk.Label(ar, text="", bg=BG, fg=MUTED, font=FS)
        self._bulk_progress_lbl.pack(side="left")

        # Treeview of per-set rows.
        tf = tk.Frame(p, bg=BG); tf.pack(fill="both", expand=True, pady=(8, 0))
        cols = ("#", "Started", "Status", "Verdict", "Cost", "Difficulty")
        self._bulk_tree = ttk.Treeview(tf, columns=cols, show="headings", height=10)
        for c, w in zip(cols, (44, 90, 180, 100, 70, 80)):
            self._bulk_tree.heading(c, text=c)
            self._bulk_tree.column(c, width=w, anchor="w" if c == "Status" else "center")
        vsb = ttk.Scrollbar(tf, orient="vertical", command=self._bulk_tree.yview)
        self._bulk_tree.configure(yscrollcommand=vsb.set)
        self._bulk_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._bulk_tree.bind("<<TreeviewSelect>>", self._bulk_row_selected)

        # Preview pane.
        pf = tk.Frame(p, bg=BORDER); pf.pack(fill="both", expand=True, pady=(8, 0))
        self._bulk_preview = scrolledtext.ScrolledText(
            pf, bg=PANEL, fg=TEXT, font=FS, relief="flat",
            padx=12, pady=10, wrap=tk.WORD, height=8
        )
        self._bulk_preview.pack(fill="both", expand=True)
        self._bulk_preview.config(state="disabled")

        # Initialise the cost banner.
        self._bulk_inputs_changed()

    # Stub methods — bodies filled in by later tasks.
    def _bulk_inputs_changed(self):
        pass

    def _bulk_start(self):
        pass

    def _bulk_stop_clicked(self):
        pass

    def _bulk_row_selected(self, _e):
        pass
```

- [ ] **Step 5: Smoke-test that the app launches and the tab renders**

Run: `cd "/Users/ibutlerking/Documents/question generation app" && ./venv/bin/python ucat_trainer.py`

Verify:
- App launches without exception.
- A new `⚡⚡ BULK` tab is visible between Generate and Knowledge Base.
- Clicking it shows the section radios, quantity/hint inputs, two buttons, an empty Treeview, and an empty preview pane.
- Quitting the app cleanly leaves no zombie processes.

Close the app.

- [ ] **Step 6: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/ui.py
git commit -m "ui: scaffold Bulk tab layout (no behaviour)"
```

---

## Task 4: Quantity validation + cost preview banner

**Files:**
- Modify: `ucat/ui.py`

- [ ] **Step 1: Implement `_bulk_inputs_changed`**

In `ucat/ui.py`, replace the `_bulk_inputs_changed` stub method body with the following:

```python
    def _bulk_inputs_changed(self):
        """Called whenever section / quantity / hint changes. Validates input,
        updates the cost preview banner, persists settings, and gates the Start
        button."""
        section = self._bulk_sec.get()
        hint    = self._bulk_hint.get()

        # Persist settings.
        self.settings.set("bulk_section", section)
        self.settings.set("bulk_hint",    hint)

        # Parse quantity.
        raw = self._bulk_qty.get().strip()
        n: Optional[int] = None
        capped = False
        if raw.isdigit():
            n = int(raw)
            if n > BULK_MAX_QUANTITY:
                n = BULK_MAX_QUANTITY
                capped = True
            if n >= 1:
                self.settings.set("bulk_quantity", n)

        # Update cost banner + Start enable state.
        if n is None or n < 1:
            self._bulk_cost_lbl.config(
                text=f"Enter a number 1 - {BULK_MAX_QUANTITY}.",
                fg=WARN,
            )
            self._bulk_start_btn.config(state="disabled")
            return

        llm     = self.settings.get("llm")
        verify  = bool(self.settings.get("verify"))
        jury    = bool(self.settings.get("multi_judge"))
        low, high = estimate_bulk_cost(n, llm, multi_judge=jury, verify=verify)

        suffix = "  (capped at the max — split into multiple runs for more)" if capped else ""
        self._bulk_cost_lbl.config(
            text=f"Estimated cost: ~${low:.2f} - ${high:.2f}   "
                 f"({n} sets × {llm}{suffix})",
            fg=ACCENT,
        )

        # Don't override "running" state — we re-enable in _bulk_run_finished.
        if self._bulk_thread is None or not self._bulk_thread.is_alive():
            self._bulk_start_btn.config(state="normal")
```

- [ ] **Step 2: Smoke-test**

Run: `cd "/Users/ibutlerking/Documents/question generation app" && ./venv/bin/python ucat_trainer.py`

Verify:
- Open Bulk tab. Banner shows e.g. `Estimated cost: ~$0.03 - $0.18   (10 sets × claude-opus-4-7)`.
- Type non-numeric in Quantity → banner becomes `Enter a number 1 - 100.` (warn colour) and Start is disabled.
- Type `0` → still disabled.
- Type `5` → banner updates with new estimate, Start re-enables.
- Type `500` → banner shows `(capped at the max …)` and the cost reflects 100 sets.
- Switch section → settings persist (close and reopen — values stick).
- Change LLM in sidebar → banner updates with new model name and cost.

Close the app.

- [ ] **Step 3: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/ui.py
git commit -m "ui: bulk-tab quantity validation and cost banner"
```

---

## Task 5: Worker thread loop (happy path, no retry / no stop yet)

**Files:**
- Modify: `ucat/ui.py`

- [ ] **Step 1: Add row-management and worker helpers**

Add these methods to `class App`, immediately after the `_bulk_row_selected` stub (which we'll fill in Task 9):

```python
    # ── Bulk worker helpers ───────────────────────────────────────────────────

    def _bulk_row_iid(self, idx: int) -> str:
        return f"bulk-{idx}"

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

        verdict_cell = "—"
        cost_cell    = "—"
        diff_cell    = "—"
        if row["result"]:
            v = row["result"].get("verdict") or {}
            if not v:
                verdict_cell = "—"
            elif v.get("overall_correct", True):
                fq = len((v.get("symbolic_qr") or {}).get("disagreed") or [])
                verdict_cell = "✓" if fq == 0 else f"⚠ {fq}"
            else:
                fq  = len(v.get("flagged_questions") or [])
                sym = len((v.get("symbolic_qr") or {}).get("disagreed") or [])
                verdict_cell = f"⚠ {fq + sym}"

            u = row["result"].get("usage") or {}
            if u.get("cost_usd") is not None:
                cost_cell = f"${u['cost_usd']:.3f}"

            cal = row["result"].get("difficulty") or {}
            sd  = cal.get("set_difficulty")
            if isinstance(sd, (int, float)):
                diff_cell = f"{sd:.1f}"

        self._bulk_tree.item(
            self._bulk_row_iid(idx),
            values=(idx, row["started"], st_cell, verdict_cell, cost_cell, diff_cell),
        )

    def _bulk_run_started(self, n: int):
        self._bulk_started_at = time.perf_counter()
        self._bulk_run_cost   = 0.0
        self._bulk_start_btn.config(state="disabled", text="Generating…")
        self._bulk_stop_btn.config(state="normal")
        self._bulk_progress_lbl.config(text=f"0 / {n}")
        self._bulk_seed_rows(n)
        llm    = self.settings.get("llm")
        verify = bool(self.settings.get("verify"))
        jury   = bool(self.settings.get("multi_judge"))
        _, est_high = estimate_bulk_cost(n, llm, multi_judge=jury, verify=verify)
        emit("bulk_run_start",
             section=self._bulk_sec.get(),
             n=n,
             model=llm,
             verify=verify,
             multi_judge=jury,
             estimated_cost_high=round(est_high, 4))

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
        self._bulk_thread = None
        self._bulk_started_at = None
        self._bulk_start_btn.config(state="normal", text="⚡  START BULK RUN")
        self._bulk_stop_btn.config(state="disabled")
        if stopped:
            tail = f"Stopped at {succeeded + failed} / {n}."
        else:
            tail = f"Bulk run finished: {succeeded} succeeded, {failed} failed"
            if skipped: tail += f", {skipped} skipped"
            tail += "."
        self._bulk_progress_lbl.config(text=tail)
        self._status(tail)
        self._bulk_inputs_changed()  # re-evaluate Start button against new state

    def _bulk_after_success(self, idx: int, result: Dict[str, Any]):
        """Main-thread; updates row + global session counters + History."""
        self._bulk_set_row(idx, status="done", result=result)
        usage = result.get("usage") or {}
        self._bulk_run_cost += usage.get("cost_usd", 0.0) or 0.0
        self._session_cost  += usage.get("cost_usd", 0.0) or 0.0
        self._session_tokens["in"]      += usage.get("input_tokens", 0) or 0
        self._session_tokens["out"]     += usage.get("output_tokens", 0) or 0
        self._session_tokens["cache_r"] += usage.get("cache_read_input_tokens", 0) or 0
        self._session_tokens["cache_w"] += usage.get("cache_creation_input_tokens", 0) or 0
        tot = sum(self._session_tokens.values())
        self._cost_lbl.config(text=f"${self._session_cost:.3f} · {tot:,} tok")
        self._refresh_stats()
        self._refresh_out()
        self._refresh_insights()

    def _bulk_worker(self, section: str, hint: str, n: int):
        self.after(0, lambda: self._bulk_run_started(n))
        succeeded = 0
        failed    = 0
        for i in range(1, n + 1):
            started_at = datetime.now().strftime("%H:%M:%S")
            self.after(0, lambda idx=i, t=started_at: self._bulk_set_row(
                idx, status="running", started=t))
            self.after(0, lambda idx=i, total=n: self._bulk_progress_lbl.config(
                text=f"{idx - 1} / {total} — generating set {idx}…"))

            try:
                result = self.rag.generate(
                    section, hint,
                    on_progress=lambda m, idx=i: self.after(0, lambda msg=m, _i=idx:
                        self._bulk_set_row(_i, status="running", progress=msg)),
                    on_delta=None,
                    variation_seed=str(uuid.uuid4())[:8],
                )
                self.after(0, lambda idx=i, r=result: self._bulk_after_success(idx, r))
                succeeded += 1
            except Exception as e:
                logger.exception(f"Bulk set {i} failed")
                err = str(e)
                self.after(0, lambda idx=i, msg=err: self._bulk_set_row(
                    idx, status="failed", error=msg))
                failed += 1

        self.after(0, lambda: self._bulk_run_finished(succeeded, failed))
```

- [ ] **Step 2: Wire `_bulk_start` to launch the worker (no confirmation modal yet — comes in Task 8)**

Replace the `_bulk_start` stub body with:

```python
    def _bulk_start(self):
        ok, msg = api_status()
        if not ok:
            messagebox.showerror("API Not Ready",
                f"{msg}\n\nCopy .env.example → .env and fill in your keys.")
            return

        section = self._bulk_sec.get()
        raw = self._bulk_qty.get().strip()
        if not raw.isdigit():
            return
        n = min(int(raw), BULK_MAX_QUANTITY)
        if n < 1:
            return

        if self.db.count(section, indexed_only=True) == 0:
            if not messagebox.askyesno("No Indexed Documents",
                f"No indexed documents for {SECTIONS[section]}.\n\n"
                "Add docs and click Index, or generate without RAG context?"):
                return

        hint = self._bulk_hint.get()
        self._bulk_stop.clear()
        self._bulk_thread = threading.Thread(
            target=self._bulk_worker, args=(section, hint, n), daemon=True
        )
        self._bulk_thread.start()
```

- [ ] **Step 3: Manual smoke test (real API call — costs a few cents)**

Pre-conditions: `.env` is configured with valid Anthropic + Voyage keys. The KB has ≥ 1 indexed doc for at least one section (or you accept the "no indexed docs" warning).

Run: `cd "/Users/ibutlerking/Documents/question generation app" && ./venv/bin/python ucat_trainer.py`

Verify:
- Open Bulk tab. Set quantity to `2` (small, cheap test). Pick a section that has indexed docs.
- Click Start. Two queued rows appear.
- Row 1 transitions: `queued → ⟳ running … → ✓ done` with verdict, cost, and difficulty filled in.
- Row 2 follows the same pattern.
- Progress label reads `2 / 2 — generating set 2…` then `Bulk run finished: 2 succeeded, 0 failed.`
- Header `$X.XXX · N tok` meter increments.
- Switch to History tab — 2 new rows are present.

Close the app.

- [ ] **Step 4: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/ui.py
git commit -m "ui: bulk worker (happy path) — sequential loop, History integration"
```

---

## Task 6: Add retry-once policy

**Files:**
- Modify: `ucat/ui.py`

- [ ] **Step 1: Wrap the `rag.generate` call in a retry loop**

In `ucat/ui.py`, find `_bulk_worker`. Replace this inner block:

```python
            try:
                result = self.rag.generate(
                    section, hint,
                    on_progress=lambda m, idx=i: self.after(0, lambda msg=m, _i=idx:
                        self._bulk_set_row(_i, status="running", progress=msg)),
                    on_delta=None,
                    variation_seed=str(uuid.uuid4())[:8],
                )
                self.after(0, lambda idx=i, r=result: self._bulk_after_success(idx, r))
                succeeded += 1
            except Exception as e:
                logger.exception(f"Bulk set {i} failed")
                err = str(e)
                self.after(0, lambda idx=i, msg=err: self._bulk_set_row(
                    idx, status="failed", error=msg))
                failed += 1
```

with a nested attempt loop:

```python
            attempts = 0
            done = False
            while attempts < 2 and not done:
                try:
                    result = self.rag.generate(
                        section, hint,
                        on_progress=lambda m, idx=i: self.after(0, lambda msg=m, _i=idx:
                            self._bulk_set_row(_i, status="running", progress=msg)),
                        on_delta=None,
                        variation_seed=str(uuid.uuid4())[:8],
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
```

- [ ] **Step 2: Manual test — induce a transient failure**

Run: `cd "/Users/ibutlerking/Documents/question generation app" && ./venv/bin/python ucat_trainer.py`

Easiest reproducible failure: temporarily edit `.env` and replace `ANTHROPIC_API_KEY` with a bogus value, save the file, then start a bulk run with quantity=1. The first attempt will 401; the retry will 401 too; the row should end as `✗ <error>`. Restore the key after.

For a successful retry path:
1. Set quantity=3 with a valid key.
2. Click Start.
3. While row 1 is `⟳ running`, briefly toggle WiFi off then on within ~2s.
4. Observe: the `✗`-then-`✓` flicker happens within row 1 (or it just succeeds because the SDK retries internally — both are acceptable).
5. The other rows should continue to process normally.

Close the app.

- [ ] **Step 3: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/ui.py
git commit -m "ui: bulk worker retries each failed set once before recording failure"
```

---

## Task 7: Stop button + skipped-row handling

**Files:**
- Modify: `ucat/ui.py`

- [ ] **Step 1: Insert a stop check at the top of each iteration**

In `ucat/ui.py`, find `_bulk_worker`. Locate the line `for i in range(1, n + 1):` and insert immediately inside the loop, before the `started_at = datetime.now()...` line, so the loop begins:

```python
        for i in range(1, n + 1):
            if self._bulk_stop.is_set():
                # Mark this and every later row as skipped, then exit.
                for j in range(i, n + 1):
                    self.after(0, lambda idx=j: self._bulk_set_row(idx, status="skipped"))
                self.after(0, lambda: self._bulk_run_finished(succeeded, failed, stopped=True))
                return

            started_at = datetime.now().strftime("%H:%M:%S")
            ...
```

(Keep the rest of the loop body unchanged.)

- [ ] **Step 2: Implement `_bulk_stop_clicked`**

Replace the `_bulk_stop_clicked` stub body with:

```python
    def _bulk_stop_clicked(self):
        if self._bulk_thread is None or not self._bulk_thread.is_alive():
            return
        self._bulk_stop.set()
        self._bulk_stop_btn.config(state="disabled")
        self._bulk_progress_lbl.config(text="Stopping after current set…")
        self._status("Stopping bulk run…")
```

- [ ] **Step 3: Manual test**

Run the app. Open Bulk tab. Set quantity to `5`. Click Start.

Verify:
- Once row 1 is `✓ done` and row 2 is `⟳ running`, click Stop.
- Status bar reads `Stopping bulk run…`. Stop button greys out.
- Row 2 finishes naturally (it was already in flight).
- Rows 3, 4, 5 immediately mark as `· skipped`.
- Final summary: `Stopped at 2 / 5.`
- Start button re-enables.
- Click Start again — a fresh run begins, Treeview is reseeded with new queued rows.

Close the app.

- [ ] **Step 4: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/ui.py
git commit -m "ui: bulk Stop halts after current set; queued rows mark skipped"
```

---

## Task 8: Cost confirmation modal

**Files:**
- Modify: `ucat/ui.py`

- [ ] **Step 1: Insert a threshold check before launching the worker**

In `ucat/ui.py`, find `_bulk_start`. Replace the tail-end of the method (everything from `hint = self._bulk_hint.get()` to the `.start()` call) with:

```python
        hint = self._bulk_hint.get()

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

        self._bulk_stop.clear()
        self._bulk_thread = threading.Thread(
            target=self._bulk_worker, args=(section, hint, n), daemon=True
        )
        self._bulk_thread.start()
```

- [ ] **Step 2: Manual test**

Run the app. Open Bulk tab.

Verify:
- With quantity = 2 and Opus, click Start → no modal, run begins immediately. Stop after row 1 to save money.
- Set quantity high enough that the high-end estimate exceeds $5 (visible in the banner — usually 50+ sets with Opus + verify). Click Start → modal appears with the four-line summary. Click No → no run starts. Click Yes → run begins. Stop right away.

Close the app.

- [ ] **Step 3: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/ui.py
git commit -m "ui: bulk-tab confirmation modal above cost threshold"
```

---

## Task 9: Row → preview pane wiring

**Files:**
- Modify: `ucat/ui.py`

- [ ] **Step 1: Replace `_bulk_row_selected` body**

In `ucat/ui.py`, replace the `_bulk_row_selected` stub body with:

```python
    def _bulk_row_selected(self, _e):
        sel = self._bulk_tree.selection()
        if not sel:
            return
        iid = sel[0]
        if not iid.startswith("bulk-"):
            return
        try:
            idx = int(iid.split("-", 1)[1])
        except ValueError:
            return
        if idx < 1 or idx > len(self._bulk_rows):
            return

        row = self._bulk_rows[idx - 1]
        self._bulk_preview.config(state="normal")
        self._bulk_preview.delete(1.0, tk.END)

        if row["status"] == "done" and row["result"]:
            self._bulk_preview.insert(tk.END, format_qset(row["result"]["data"]))
        elif row["status"] == "failed":
            err = row["error"] or "(no error message captured)"
            self._bulk_preview.insert(tk.END,
                f"Set {idx} failed after retry.\n\n{err}")
        elif row["status"] == "skipped":
            self._bulk_preview.insert(tk.END,
                f"Set {idx} was skipped (Stop pressed before it started).")
        elif row["status"] == "running":
            self._bulk_preview.insert(tk.END,
                f"Set {idx} is still being generated…")
        else:  # queued
            self._bulk_preview.insert(tk.END, f"Set {idx} is queued.")

        self._bulk_preview.config(state="disabled")
```

- [ ] **Step 2: Manual test**

Run the app. Run a bulk run with quantity 2. While the run is in progress, click on row 1 → preview shows "still being generated…" if running, or the formatted question set if done. After completion:

Verify:
- Click row 1 → preview shows the full formatted question set (passage / questions / answers).
- Click row 2 → preview switches to row 2's content.
- If you intentionally make row 2 fail (revoke the API key after row 1 completes), clicking row 2 should show the error message.

Close the app.

- [ ] **Step 3: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/ui.py
git commit -m "ui: bulk-tab row click renders preview / error / skipped state"
```

---

## Task 10: Concurrent-run guards + clean shutdown

**Files:**
- Modify: `ucat/ui.py`

- [ ] **Step 1: Block bulk launch while single-shot is active**

In `ucat/ui.py`, find `_bulk_start`. After the `api_status()` check and before parsing the quantity, insert:

```python
        if self._bulk_thread is not None and self._bulk_thread.is_alive():
            self._status("A bulk run is already in progress.")
            return
        if str(self._gbtn.cget("state")) == "disabled":
            self._status("A run is already in progress — wait or stop it first.")
            return
```

So `_bulk_start` now begins:

```python
    def _bulk_start(self):
        ok, msg = api_status()
        if not ok:
            messagebox.showerror("API Not Ready",
                f"{msg}\n\nCopy .env.example → .env and fill in your keys.")
            return

        if self._bulk_thread is not None and self._bulk_thread.is_alive():
            self._status("A bulk run is already in progress.")
            return
        if str(self._gbtn.cget("state")) == "disabled":
            self._status("A run is already in progress — wait or stop it first.")
            return

        section = self._bulk_sec.get()
        ...
```

- [ ] **Step 2: Block single-shot while bulk is active**

In `ucat/ui.py`, find `_launch_gen`. At the very top, just before the existing `ok, msg = api_status()` check, insert:

```python
    def _launch_gen(self, variation_seed: Optional[str]):
        if self._bulk_thread is not None and self._bulk_thread.is_alive():
            self._status("A bulk run is in progress — wait or stop it first.")
            return

        ok, msg = api_status()
        ...
```

- [ ] **Step 3: Update `on_close` to stop bulk cleanly**

In `ucat/ui.py`, replace the existing `on_close` method:

```python
    def on_close(self):
        self.db.close(); self.destroy()
```

with:

```python
    def on_close(self):
        # Halt any in-flight bulk run; give the worker up to 5s to exit before destroy.
        if self._bulk_thread is not None and self._bulk_thread.is_alive():
            self._bulk_stop.set()
            self._bulk_thread.join(timeout=5.0)
        self.db.close()
        self.destroy()
```

- [ ] **Step 4: Manual test — concurrent guards**

Run the app.

Verify:
- Start a single-shot Generate. While it's running, switch to Bulk and click Start → no run begins; status bar reads `A run is already in progress — wait or stop it first.`
- Wait for single-shot to finish. Start a bulk run with quantity 5. While it's running, switch to Generate and click Generate → no run begins; status bar reads `A bulk run is in progress — wait or stop it first.`

- [ ] **Step 5: Manual test — clean shutdown**

Start a bulk run with quantity 10. After row 2 completes and row 3 enters `⟳ running`, close the window (the X button or `Cmd-Q`).

Verify:
- The window closes within ~5 seconds (the in-flight API call finishes or times out, then the thread joins).
- No zombie Python process — check with `ps aux | grep ucat_trainer | grep -v grep` (should be empty).

- [ ] **Step 6: Commit**

```bash
cd "/Users/ibutlerking/Documents/question generation app"
git add ucat/ui.py
git commit -m "ui: bulk vs single-shot mutual exclusion + clean shutdown"
```

---

## Task 11: Telemetry verification

**Files:**
- (no code changes — verifying that Task 5's emit calls land correctly)

The `bulk_run_start` and `bulk_run_end` events were emitted in Task 5 via `_bulk_run_started` / `_bulk_run_finished`. This task confirms they reach the JSONL log with correct fields.

- [ ] **Step 1: Run a small bulk run**

Run the app. Bulk tab → quantity 2 → Start. Wait for completion. Close the app.

- [ ] **Step 2: Tail the telemetry file and verify the events**

Run: `cd "/Users/ibutlerking/Documents/question generation app" && tail -20 ucat_telemetry.jsonl | grep -E "bulk_run"`

Expected: at least one `bulk_run_start` line with fields `section`, `n`, `model`, `verify`, `multi_judge`, `estimated_cost_high`; and a `bulk_run_end` line with `section`, `n`, `succeeded`, `failed`, `stopped`, `actual_cost_usd`, `duration_s`.

- [ ] **Step 3: Stop test — verify `stopped=true`**

Run the app. Bulk tab → quantity 5 → Start. Click Stop after row 1 completes. Wait for finish. Close.

Run: `tail -10 ucat_telemetry.jsonl | grep bulk_run_end`

The latest line should have `"stopped": true` and `succeeded + failed < 5`.

If the telemetry is wrong, the bug is in `_bulk_run_finished` — re-check that the `stopped` parameter is passed correctly from `_bulk_worker`'s early-return path (Task 7).

No commit (no code changes in this task).

---

## Task 12: End-to-end smoke checklist

**Files:**
- (verification only)

Run the app. Walk through each scenario from the spec's testing section. This catches regressions and integration bugs that the per-task tests miss.

- [ ] **Smoke 1: Two-set happy path** — quantity=2, click Start, both land in History as `✓ done` rows in the Bulk Treeview.

- [ ] **Smoke 2: Stop mid-run** — quantity=5, Stop after row 1 completes; rows 3-5 show `· skipped`; status bar shows `Stopped at 2 / 5.`

- [ ] **Smoke 3: Retry on failure** — break the Anthropic API key in `.env`, save, run a quantity=1 bulk; row should attempt twice, then end as `✗ <error>`. Restore the key.

- [ ] **Smoke 4: Cost confirm modal** — set quantity high enough that estimated `high > $5` (visible in the banner). Click Start → modal appears. Click No → no run begins.

- [ ] **Smoke 5: Concurrent block** — start single-shot Generate; while running, click Bulk Start → blocked with status-bar message.

- [ ] **Smoke 6: App close mid-run** — quantity=10, close the app after row 2 completes. App exits within 5s; no zombie process.

- [ ] **Smoke 7: Empty KB** — pick a section with zero indexed docs (or temporarily clear an index). Click Start → existing `No Indexed Documents` dialog appears. Yes proceeds without RAG context. No aborts.

- [ ] **Smoke 8: Settings persist** — set quantity to 7 and section to QR with hint "ratios". Quit. Relaunch. Bulk tab should reopen with the same values.

If all smoke tests pass, the feature is complete.

- [ ] **Final step: ensure the unit tests still pass**

Run: `cd "/Users/ibutlerking/Documents/question generation app" && ./venv/bin/python tests/test_bulk_cost.py`

Expected: `All tests passed.`

---

## Notes

- **Why no progress bar widget?** A simple `K / N` text label was deemed sufficient at the spec stage — bulk runs are minutes long, not seconds, and the per-row status is the actual progress signal.
- **Why no DB table for runs?** The History tab is already the canonical record of every generation. A run is a transient session-level concept; persistent run state would require eviction policies and rebuild logic on app crash for no real user benefit.
- **Why retry once and not three times?** Two attempts catches the 90% case (transient HTTP / rate-limit blips) without dragging out runs. Persistent failures (bad config, schema bugs) won't be cured by more retries.
- **Why no streaming in bulk?** The single-shot streaming view is per-set; piping 50 sets of token deltas into one widget would be illegible noise. Per-row progress text from `on_progress` is enough signal.
