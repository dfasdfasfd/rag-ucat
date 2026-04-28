"""Bias and coverage detection.

Two flavours of analysis:

1. **Per-set coverage tags.** The model emits topic + scenario_type tags per
   question (see ``models.CoverageTags``). We aggregate these here.

2. **Bias heuristics on the question text.** Quick deterministic scans for
   gender-skewed names, narrow cultural framing, and topic monoculture.

The output of this module feeds the bias dashboard and a pre-save warning when
a generation drifts toward a one-sided distribution.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

# ─── Name / pronoun heuristics (rough — not a final word) ─────────────────────

_MALE_NAMES = {
    "alex", "ben", "chris", "dan", "david", "eric", "frank", "george",
    "harry", "jack", "james", "john", "kevin", "liam", "michael", "nick",
    "oliver", "paul", "robert", "sam", "tom", "william",
}
_FEMALE_NAMES = {
    "alice", "amy", "anna", "beth", "cara", "claire", "emma", "eve",
    "fiona", "grace", "hannah", "isla", "jane", "kate", "lily", "lucy",
    "maria", "olivia", "ruby", "sarah", "sophie", "zoe",
}
_NON_WESTERN_HINTS = {
    "aisha", "amir", "chen", "fatima", "hiro", "jin", "kenji", "li",
    "mei", "omar", "priya", "ravi", "sanjay", "wei", "yumi", "zara",
}

_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")

# ─── Subtype / scenario taxonomy expectations ────────────────────────────────

EXPECTED_SCENARIOS = {
    "VR": {"scientific", "humanities", "business", "social", "everyday"},
    "DM": {"abstract", "everyday", "business", "social"},
    "QR": {"business", "scientific", "everyday", "medical", "sport"},
    "AR": {"abstract"},
    # SJT used to track only the per-question scenario_type (medical/social),
    # but that's a 2-bucket taxonomy too coarse for the 4-situation rotation
    # the role block promotes. Real diversification happens at the
    # set-level `SJTSet.situation_type` field — see `EXPECTED_SJT_SITUATIONS`
    # below, used by `pick_diversification` and `aggregate_history` for SJT.
    "SJT": {"medical", "social"},
}

# UCAT SJT situation taxonomy — drives diversification across bulk runs.
# Mirrors the `SJTSituationType` Literal in `models.py`. When the bulk
# stats show one bucket dominant, `pick_diversification` will inject a
# `force_scenario` directive nudging Claude toward an underused bucket.
EXPECTED_SJT_SITUATIONS = {
    "medical_ethics",
    "team_conflict",
    "boundary_management",
    "professional_communication",
}

EXPECTED_DM_SUBTYPES = {"syllogism", "logical", "venn", "probability", "argument"}
EXPECTED_SJT_SUBTYPES = {"appropriateness", "importance"}


# ─── Per-question scan ───────────────────────────────────────────────────────

def scan_question(q: Dict[str, Any]) -> Dict[str, Any]:
    """Return bias signals visible in a single question's text."""
    text = (q.get("text", "") + " " + q.get("explanation", "")).lower()
    names = _NAME_RE.findall(q.get("text", "") + " " + q.get("explanation", ""))
    name_origins = Counter()
    male = female = 0
    for raw in names:
        n = raw.lower()
        if n in _MALE_NAMES:    male += 1; name_origins["western_male"] += 1
        elif n in _FEMALE_NAMES: female += 1; name_origins["western_female"] += 1
        elif n in _NON_WESTERN_HINTS: name_origins["non_western"] += 1
    return {
        "names_male":   male,
        "names_female": female,
        "name_origins": dict(name_origins),
        "uses_pronoun_he":  bool(re.search(r"\bhe\b|\bhis\b|\bhim\b", text)),
        "uses_pronoun_she": bool(re.search(r"\bshe\b|\bher\b|\bhers\b", text)),
    }


def aggregate_set(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate per-question coverage and bias signals for one generated set.

    Returns a dict suitable for storing in `generated.coverage` (see db.py).
    """
    section = data.get("section", "")
    questions = data.get("questions", []) or []

    per_question = []
    topics = Counter()
    scenarios = Counter()
    dm_subtypes = Counter()
    male = female = 0
    name_origins = Counter()

    for q in questions:
        cov = q.get("coverage") or {}
        topics[cov.get("topic", "?")] += 1
        scenarios[cov.get("scenario_type", "?")] += 1
        if section == "DM" and q.get("type"):
            dm_subtypes[q["type"]] += 1
        sig = scan_question(q)
        male   += sig["names_male"]
        female += sig["names_female"]
        for k, v in sig["name_origins"].items():
            name_origins[k] += v
        per_question.append({
            "number":         q.get("number"),
            "topic":          cov.get("topic"),
            "scenario_type":  cov.get("scenario_type"),
            "named_entities": cov.get("contains_named_entities", False),
            "cultural_context": cov.get("cultural_context"),
        })

    flags = []
    # Gender skew within a single set is mild — only flag the strongest cases.
    total_named = male + female
    if total_named >= 4 and male >= 0.85 * total_named:
        flags.append("Names skew strongly male")
    if total_named >= 4 and female >= 0.85 * total_named:
        flags.append("Names skew strongly female")
    # All names western?
    if total_named >= 3 and name_origins.get("non_western", 0) == 0:
        flags.append("All names appear to be Western — consider diverse origins")
    # Scenario monoculture.
    expected = EXPECTED_SCENARIOS.get(section, set())
    if expected and len(set(scenarios.keys()) & expected) == 1 and len(questions) >= 4:
        flags.append(f"All scenarios are '{next(iter(scenarios))}' — UCAT mixes scenario types")
    # DM subtype gaps in a 5-question set. The role block prompts Claude
    # to "include syllogism, logical, venn, probability, AND argument
    # subtypes — one of each is ideal." A perfect mixed set hits all 5.
    # We flag at >=2 missing (so a 3-subtype set fires) — the previous
    # >=3 threshold let through 4-syllogism + 1-logical sets that
    # technically had "3 subtypes" but were variety-deficient.
    if section == "DM" and len(questions) == 5:
        missing = EXPECTED_DM_SUBTYPES - set(dm_subtypes.keys())
        if len(missing) >= 2:
            flags.append(
                f"DM set covers only {sorted(dm_subtypes.keys())} "
                f"(missing {sorted(missing)}) — UCAT mixed sets ideally hit "
                "all 5 subtypes"
            )
        # Single-subtype dominance check — even a "complete" set with 4 of
        # one subtype is still variety-deficient.
        max_count = max(dm_subtypes.values()) if dm_subtypes else 0
        if max_count >= 4:
            dominant = max(dm_subtypes, key=dm_subtypes.get)
            flags.append(
                f"DM set is {max_count}× {dominant}-dominant — aim for ≤2 "
                "of any one subtype"
            )

    # SJT situation_type tracking — record for downstream `pick_diversification`.
    sjt_situation_type = data.get("situation_type") if section == "SJT" else None

    return {
        "per_question":  per_question,
        "topic_counts":  dict(topics),
        "scenario_counts": dict(scenarios),
        "dm_subtype_counts": dict(dm_subtypes) if section == "DM" else None,
        "sjt_situation_type": sjt_situation_type,
        "name_signals": {
            "male": male, "female": female, "origins": dict(name_origins),
        },
        "flags": flags,
    }


# ─── Pre-generation diversification ──────────────────────────────────────────

def pick_diversification(
    stats: Optional[Dict[str, Any]],
    section: str,
    *,
    avoid_top_k: int = 5,
    rng: Optional[random.Random] = None,
) -> Optional[Dict[str, Any]]:
    """Pick steering signals for the next generation in a bulk run.

    Returns ``{"scenario": <picked or None>, "avoid_topics": [...]}`` or ``None``
    when there is no useful steer (e.g. AR section with no topic history).

    - ``scenario`` is one of the under-represented expected scenarios for the
      section, with random tie-breaking. ``None`` for AR (only one expected
      scenario) or unknown sections.
    - ``avoid_topics`` is the top-K most frequent recent topics, bounded so the
      prompt stays a constant size regardless of bulk length.
    """
    rng = rng or random
    expected = EXPECTED_SCENARIOS.get(section, set())
    topic_counts: Dict[str, int] = (stats or {}).get("topics", {}) or {}
    scenario_counts: Dict[str, int] = (stats or {}).get("scenarios", {}) or {}

    picked_scenario: Optional[str] = None
    # SJT special-cases: pick from `EXPECTED_SJT_SITUATIONS` against the
    # `sjt_situation_counts` rolled up from the generated table. The
    # generic scenario_type axis (medical/social) is too coarse to drive
    # the 4-situation rotation the role block promotes.
    if section == "SJT":
        sjt_counts: Dict[str, int] = (stats or {}).get("sjt_situations", {}) or {}
        scored = [(sjt_counts.get(s, 0), s) for s in EXPECTED_SJT_SITUATIONS]
        min_n = min(n for n, _ in scored)
        candidates = sorted(s for n, s in scored if n == min_n)
        picked_scenario = rng.choice(candidates)
    elif len(expected) > 1:
        scored = [(scenario_counts.get(s, 0), s) for s in expected]
        min_n = min(n for n, _ in scored)
        candidates = sorted(s for n, s in scored if n == min_n)
        picked_scenario = rng.choice(candidates)

    sorted_topics = sorted(
        ((t, n) for t, n in topic_counts.items() if t and t != "?"),
        key=lambda kv: (-kv[1], kv[0]),
    )
    avoid_topics = [t for t, _ in sorted_topics[:avoid_top_k]]

    if not picked_scenario and not avoid_topics:
        return None
    return {"scenario": picked_scenario, "avoid_topics": avoid_topics}


# ─── KB-level / global aggregation ───────────────────────────────────────────

def aggregate_history(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine many ``generated`` rows into a global coverage report — used by
    the dashboard tab to show overall topic distribution and gaps.
    """
    topics = Counter()
    scenarios = Counter()
    sections = Counter()
    flags = Counter()
    for r in rows:
        sections[r.get("section", "?")] += 1
        cov = r.get("coverage") or {}
        for k, v in (cov.get("topic_counts") or {}).items():     topics[k]    += v
        for k, v in (cov.get("scenario_counts") or {}).items():  scenarios[k] += v
        for f in (cov.get("flags") or []):                       flags[f]     += 1

    # Gap detection.
    gap_warnings: List[str] = []
    for sec, expected in EXPECTED_SCENARIOS.items():
        in_sec = [r for r in rows if r.get("section") == sec]
        if not in_sec:
            continue
        seen = Counter()
        for r in in_sec:
            for k, v in ((r.get("coverage") or {}).get("scenario_counts") or {}).items():
                seen[k] += v
        missing = expected - set(seen.keys())
        if missing and sum(seen.values()) >= 8:
            gap_warnings.append(f"{sec}: missing scenario types {sorted(missing)}")

    return {
        "rows":      len(rows),
        "sections":  dict(sections),
        "topics":    dict(topics.most_common(40)),
        "scenarios": dict(scenarios),
        "flag_counts": dict(flags),
        "gaps":      gap_warnings,
    }
