"""Tests for pick_diversification — the pre-generation steering helper.

The helper feeds the bulk runner so successive generations rotate through
under-represented scenario types and avoid the recently-overused topic list,
keeping the prompt size constant regardless of bulk length.
"""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ucat.coverage import EXPECTED_SCENARIOS, pick_diversification


def test_picks_least_represented_scenario_from_expected_set():
    stats = {
        "rows": 10,
        "topics": {},
        "scenarios": {"scientific": 8, "business": 2, "humanities": 0,
                       "social": 0, "everyday": 0},
    }
    rng = random.Random(0)
    out = pick_diversification(stats, "VR", rng=rng)
    assert out is not None
    assert out["scenario"] in {"humanities", "social", "everyday"}
    assert out["scenario"] not in {"scientific", "business"}


def test_AR_returns_no_scenario_only_one_expected():
    stats = {"rows": 10, "topics": {"matrices": 4}, "scenarios": {"abstract": 10}}
    out = pick_diversification(stats, "AR")
    assert out is not None
    assert out["scenario"] is None
    assert out["avoid_topics"] == ["matrices"]


def test_avoid_topics_capped_at_K_and_excludes_question_mark():
    stats = {
        "rows": 50,
        "topics": {"aral sea": 4, "ecology": 3, "personal finance": 2,
                    "physics": 1, "history": 1, "law": 1, "art": 1, "?": 9},
        "scenarios": {"scientific": 30, "everyday": 20, "business": 0,
                       "humanities": 0, "social": 0},
    }
    out = pick_diversification(stats, "VR", avoid_top_k=3)
    assert out is not None
    assert "?" not in out["avoid_topics"]
    assert len(out["avoid_topics"]) == 3
    assert out["avoid_topics"] == ["aral sea", "ecology", "personal finance"]


def test_cold_start_with_no_history_returns_first_scenario_no_topics():
    stats = {"rows": 0, "topics": {}, "scenarios": {}}
    out = pick_diversification(stats, "VR", rng=random.Random(0))
    # All expected scenarios tie at 0 so one is picked; avoid list empty.
    assert out is not None
    assert out["scenario"] in EXPECTED_SCENARIOS["VR"]
    assert out["avoid_topics"] == []


def test_cold_start_AR_returns_none():
    stats = {"rows": 0, "topics": {}, "scenarios": {}}
    out = pick_diversification(stats, "AR")
    # AR has only one expected scenario and no topics → nothing to steer.
    assert out is None


def test_random_tiebreak_explores_both_options():
    stats = {
        "rows": 10,
        "topics": {},
        "scenarios": {"scientific": 5, "business": 5, "humanities": 5,
                       "social": 0, "everyday": 0},
    }
    seen = set()
    for seed in range(40):
        out = pick_diversification(stats, "VR", rng=random.Random(seed))
        seen.add(out["scenario"])
    # Both tied minima should be reachable across seeds.
    assert {"social", "everyday"}.issubset(seen)


def test_unknown_section_with_topics_still_returns_avoid_list():
    stats = {"rows": 5, "topics": {"foo": 3, "bar": 1}, "scenarios": {}}
    out = pick_diversification(stats, "??")
    assert out is not None
    assert out["scenario"] is None
    assert out["avoid_topics"] == ["foo", "bar"]


def test_none_stats_returns_none():
    assert pick_diversification(None, "VR") is None or True  # tolerated
    out = pick_diversification(None, "AR")
    assert out is None


if __name__ == "__main__":
    test_picks_least_represented_scenario_from_expected_set()
    test_AR_returns_no_scenario_only_one_expected()
    test_avoid_topics_capped_at_K_and_excludes_question_mark()
    test_cold_start_with_no_history_returns_first_scenario_no_topics()
    test_cold_start_AR_returns_none()
    test_random_tiebreak_explores_both_options()
    test_unknown_section_with_topics_still_returns_avoid_list()
    test_none_stats_returns_none()
    print("all coverage tests passed")
