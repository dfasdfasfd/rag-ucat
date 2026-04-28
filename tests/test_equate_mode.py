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
