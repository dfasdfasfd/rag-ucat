"""Tests for the upgraded symbolic_qr_check.

The check has three layers, each with its own failure-mode signature:

  1. Explanation→option agreement (last `=` number vs marked option)
  2. Explanation arithmetic (every `LHS = RHS` line evaluated by sympy)
  3. Chart misread (chart-value claims cross-checked against stim data)
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ucat.verification import symbolic_qr_check  # noqa: E402


def _qr_set(*, expl: str, answer: str, options: dict, stimulus: dict | None = None):
    return {
        "section": "QR",
        "stimulus": stimulus or {},
        "questions": [
            {
                "number": 1,
                "text": "Q1?",
                "options": options,
                "answer": answer,
                "explanation": expl,
                "difficulty": 3.0,
                "coverage": {
                    "topic": "test", "scenario_type": "everyday",
                    "contains_named_entities": False,
                },
            }
        ],
    }


class TestLayer1ExplanationAgreement:
    def test_explanation_final_number_matches_marked_option(self):
        result = symbolic_qr_check(_qr_set(
            expl="Total = 45 + 38 = 83",
            answer="C",
            options={"A": "70", "B": "75", "C": "83", "D": "90", "E": "100"},
        ))
        assert result["checked"] == 1
        assert result["agreed"] == 1
        assert result["disagreed"] == []

    def test_explanation_disagrees_with_marked_option(self):
        # Claude wrote a clean derivation = 83 but marked answer "A" (=70).
        result = symbolic_qr_check(_qr_set(
            expl="Total = 45 + 38 = 83",
            answer="A",
            options={"A": "70", "B": "75", "C": "83", "D": "90", "E": "100"},
        ))
        assert len(result["disagreed"]) == 1
        assert result["disagreed"][0]["computed_value"] == pytest.approx(83.0)
        assert result["disagreed"][0]["marked_value"] == pytest.approx(70.0)


class TestLayer2ExplanationArithmetic:
    def test_internally_consistent_arithmetic_passes(self):
        result = symbolic_qr_check(_qr_set(
            expl="Total = 45 + 38 = 83\nIncrease = 83 - 60 = 23",
            answer="B",
            options={"A": "20", "B": "23", "C": "25", "D": "30", "E": "35"},
        ))
        assert result["arithmetic_errors"] == []

    def test_broken_arithmetic_is_flagged(self):
        # 45 + 38 = 83, but Claude wrote 79. Layer 1 still passes (79
        # matches option C), but layer 2 catches the broken LHS.
        result = symbolic_qr_check(_qr_set(
            expl="Total = 45 + 38 = 79",
            answer="C",
            options={"A": "70", "B": "75", "C": "79", "D": "85", "E": "90"},
        ))
        assert len(result["arithmetic_errors"]) == 1
        err = result["arithmetic_errors"][0]
        assert err["lhs_computed"] == pytest.approx(83.0)
        assert err["rhs_claimed"] == pytest.approx(79.0)

    def test_prose_lines_with_equals_are_skipped(self):
        # "x is the same as y = 5" shouldn't be parsed as arithmetic.
        result = symbolic_qr_check(_qr_set(
            expl="The variable x = 5 represents revenue. Therefore total = 5 * 4 = 20.",
            answer="A",
            options={"A": "20", "B": "25"},
        ))
        # 5*4=20 should pass (no false positive on the prose line).
        assert result["arithmetic_errors"] == []


class TestLayer3ChartMisread:
    def test_claim_matches_chart_passes(self):
        result = symbolic_qr_check(_qr_set(
            expl="Sales in 2022 = 50. Sales in 2021 = 38. Increase = 12.",
            answer="A",
            options={"A": "12", "B": "15"},
            stimulus={
                "type": "bar", "title": "Sales",
                "categories": ["2021", "2022"],
                "series": [{"name": "Sales", "values": [38, 50]}],
            },
        ))
        assert result["chart_misreads"] == []

    def test_misread_chart_value_is_flagged(self):
        # Chart says 2022 = 50, but explanation reads 2022 = 45.
        # Both layer-1 and layer-2 might pass (Claude derives a
        # wrong-but-internally-consistent answer), but layer 3 catches
        # the chart misread itself.
        result = symbolic_qr_check(_qr_set(
            expl="Sales in 2022 = 45. Therefore total = 45 + 38 = 83.",
            answer="C",
            options={"A": "70", "B": "75", "C": "83", "D": "85", "E": "90"},
            stimulus={
                "type": "bar", "title": "Sales",
                "categories": ["2021", "2022"],
                "series": [{"name": "Sales", "values": [38, 50]}],
            },
        ))
        # Find the misread for the "2022" claim
        misreads = [m for m in result["chart_misreads"] if "2022" in m["label"]]
        assert len(misreads) >= 1
        m = misreads[0]
        assert m["claimed"] == pytest.approx(45.0)
        assert m["actual"] == pytest.approx(50.0)


class TestBackcompat:
    def test_skipped_sympy_returns_full_shape(self):
        # The new return shape should always include the new keys, even
        # when sympy isn't installed.
        from ucat import verification as v
        original = v._HAS_SYMPY
        v._HAS_SYMPY = False
        try:
            result = symbolic_qr_check(_qr_set(
                expl="x = 1", answer="A", options={"A": "1"},
            ))
            assert "arithmetic_errors" in result
            assert "chart_misreads" in result
        finally:
            v._HAS_SYMPY = original
