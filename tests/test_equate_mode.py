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
