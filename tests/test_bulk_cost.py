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
