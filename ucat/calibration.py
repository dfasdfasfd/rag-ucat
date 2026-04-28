"""Auto-calibrated difficulty (no student responses required).

We combine three signals:

1. **Model self-prediction**  — Claude predicts a 1.0-5.0 logit per question
   alongside the generation itself (via the schema).
2. **Multi-judge prediction** — when the verifier runs, it ALSO emits a
   difficulty estimate, giving a second opinion.
3. **Feature-based estimate** — deterministic heuristics on the question text
   and options: option-similarity entropy, math step count, vocabulary level,
   distractor proximity, passage length, etc.

The three estimates are combined via a weighted average to produce a single
calibrated difficulty score per question. The set-level difficulty is the mean
of its question difficulties.

This module exposes pure functions; the rag orchestrator decides when to call.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ─── Feature-based difficulty heuristics ──────────────────────────────────────

# Vocabulary level — rough proxy. UCAT VR rewards Latinate/abstract vocabulary.
_HARD_VOCAB_PATTERNS = [
    r"\b\w{12,}\b",                              # very long words
    r"\b(?:notwithstanding|paradoxically|concomitant|antithetical|"
    r"mitigation|equivocal|perfunctory|unequivocal|spurious|"
    r"tantamount|requisite|ostensibly|presupposes|axiomatic|"
    r"corollary|empirical|hypothetical|tautological)\b",
]
_HARD_VOCAB_RE = re.compile("|".join(_HARD_VOCAB_PATTERNS), re.IGNORECASE)

# Math operations — used for QR step counting.
_MATH_OP_RE = re.compile(r"[\+\-\×\*\/÷%]|\b(?:sum|mean|median|mode|ratio|"
                          r"percent|increase|decrease|fraction|times|multiply|divide)\b",
                          re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def feature_difficulty(question: Dict[str, Any], section: str) -> float:
    """
    Estimate difficulty (1.0-5.0) from observable features of a question dict.

    The function is intentionally section-aware: VR weights vocabulary heavily,
    QR weights math step count, AR weights option count, DM weights logical
    chain length.
    """
    text   = (question.get("text") or "")
    expl   = (question.get("explanation") or "")
    opts   = question.get("options") or {}

    # --- shared signals ---
    text_len  = len(text)
    n_options = len(opts)
    avg_opt_len = (sum(len(str(v)) for v in opts.values()) / max(n_options, 1)) if opts else 0
    hard_vocab = len(_HARD_VOCAB_RE.findall(text + " " + expl))

    # --- section-specific signal ---
    if section == "QR":
        # Math step count — number of operations + numbers in explanation.
        ops  = len(_MATH_OP_RE.findall(expl))
        nums = len(_NUMBER_RE.findall(expl))
        steps = ops + max(nums - 1, 0)
        score = 1.0 + min(steps / 2.5, 3.5)            # 0 steps → 1, ~9 steps → 4.6
        # Option proximity: numerically close options are harder.
        nums_in_opts = []
        for v in opts.values():
            m = _NUMBER_RE.findall(str(v))
            if m:
                try: nums_in_opts.append(float(m[0].replace(",", "")))
                except ValueError: pass
        if len(nums_in_opts) >= 2:
            spread = (max(nums_in_opts) - min(nums_in_opts)) / max(abs(max(nums_in_opts)), 1)
            if spread < 0.10:    score += 0.6      # very close options
            elif spread < 0.25:  score += 0.3
        return _clamp(score)

    if section == "VR":
        # Vocabulary + passage-inferential difficulty.
        score = 1.5 + min(hard_vocab * 0.25, 2.0)
        # T/F/Can't Tell items with subtle "Can't Tell" answers harder.
        if str(question.get("answer", "")).upper() == "C" and n_options == 3:
            score += 0.4
        # Long stems and longer options are subtler.
        if text_len > 120: score += 0.4
        if avg_opt_len > 60: score += 0.3
        return _clamp(score)

    if section == "DM":
        # Logical chain depth — count premises and constraints.
        premises = len(re.findall(r"[•\-\*]\s|premise\s*\d*[:.]\s", text, re.IGNORECASE))
        chain = len(re.findall(r"\b(?:if|then|all|some|no|none|only|unless)\b",
                                text, re.IGNORECASE))
        score = 1.5 + min(premises * 0.4 + chain * 0.2, 3.0)
        if "venn" in str(question.get("type", "")).lower():
            score += 0.3
        return _clamp(score)

    if section == "AR":
        # Number of overlapping rules in shapes increases difficulty.
        # Without the panel structure here we approximate from explanation length.
        score = 2.5
        if len(expl) > 80:  score += 0.5
        if "test shape" in text.lower() and " not " in expl.lower():
            score += 0.4   # negative-rule reasoning
        return _clamp(score)

    return 3.0  # fallback


def _clamp(x: float, lo: float = 1.0, hi: float = 5.0) -> float:
    return max(lo, min(hi, round(x, 2)))


# ─── Combine signals ──────────────────────────────────────────────────────────

def calibrate_question(
    question: Dict[str, Any],
    section: str,
    *,
    model_prediction: Optional[float] = None,
    judge_prediction: Optional[float] = None,
    weights: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, Any]:
    """
    Combine the three difficulty signals into a single calibrated logit.

    Weights are dynamic based on what's available:

    - When a judge prediction is present (verify enabled), reduce the
      feature heuristic's weight from 0.4 to 0.2 and bump both model and
      judge predictions to 0.4. The judge is an actual reasoning pass over
      the question, while the feature heuristic is a crude proxy
      (especially for AR, where it's literally hardcoded at 2.5).
      Reweighting in favour of the more-informative signals reduces
      calibration noise on verified sets.

    - With no judge available (verify disabled), fall back to 0.5 / 0.5
      between the feature heuristic and the model self-prediction. The
      old 0.4 / 0.3 / 0.3 default left 30% weight on a missing signal
      that re-normalised away anyway, so this is a wash mathematically
      but reads more cleanly.

    Callers can still override `weights` explicitly to test alternative
    blends (used by `tests/test_subtype_targeting.py`).

    Missing signals are renormalised away — never zeroed.

    Returns a dict with the calibrated value plus the contributing components,
    so the dashboard can show provenance.
    """
    if weights is None:
        if judge_prediction is not None:
            weights = (0.2, 0.4, 0.4)  # judge present: down-weight heuristic
        else:
            weights = (0.5, 0.5, 0.0)  # no judge: feature + model only
    feat = feature_difficulty(question, section)
    components = {
        "feature":           (feat,             weights[0]),
        "model_prediction":  (model_prediction, weights[1]),
        "judge_prediction":  (judge_prediction, weights[2]),
    }

    used = [(v, w) for (v, w) in components.values() if v is not None]
    if not used:
        return {"calibrated": 3.0, "components": components, "weights_used": 0.0}

    total_w = sum(w for _, w in used)
    cal = sum(v * w for v, w in used) / total_w
    return {
        "calibrated": _clamp(cal),
        "components": {k: {"value": v, "weight": w} for k, (v, w) in components.items()},
        "weights_used": round(total_w, 3),
    }


def calibrate_set(
    questions: Iterable[Dict[str, Any]],
    section: str,
    *,
    judge_predictions: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """Per-question calibration + set-level summary (mean difficulty)."""
    judge_predictions = judge_predictions or {}
    per_q = []
    for q in questions:
        n = q.get("number")
        cal = calibrate_question(
            q, section,
            model_prediction=q.get("difficulty"),
            judge_prediction=judge_predictions.get(n) if n is not None else None,
        )
        per_q.append({"number": n, **cal})

    vals = [c["calibrated"] for c in per_q]
    return {
        "per_question": per_q,
        "set_difficulty": round(sum(vals) / len(vals), 2) if vals else 3.0,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }


def difficulty_distance(target: float, actual: float) -> float:
    """How far the calibrated set is from a requested target — used by retry logic."""
    return abs(target - actual)
