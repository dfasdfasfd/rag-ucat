"""Verification of generated questions.

Two layers:

1. **LLM judges.** A cheap Haiku call by default; optionally a 3-LLM jury
   (Haiku + Sonnet + Opus) for high-stakes content. Each judge returns a
   per-question correctness flag, free-text reasoning, AND a difficulty
   estimate that feeds back into the calibration module.

2. **Symbolic QR check.** A deterministic verifier that re-evaluates each QR
   question's marked answer using sympy on the supplied chart. Catches
   arithmetic mistakes that LLMs sometimes miss.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, ConfigDict

from .config import SECTIONS
from .llm import LLMClient, extract_usage, pydantic_to_strict_schema
from .telemetry import logger, trace

try:
    import sympy
    _HAS_SYMPY = True
except ImportError:
    _HAS_SYMPY = False

# ─── Schemas (verification has its own — not in models.py) ────────────────────

class _QuestionVerdictWithDifficulty(BaseModel):
    model_config = ConfigDict(extra="forbid")
    number: int
    correct: bool
    reasoning: str = ""
    difficulty: float = Field(ge=1.0, le=5.0,
                               description="Independent difficulty estimate (1.0-5.0 logits).")


class _JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    per_question: List[_QuestionVerdictWithDifficulty] = Field(default_factory=list)
    overall_correct: bool
    confidence: str   # one of low/medium/high — kept loose for resilience
    notes: List[str] = Field(default_factory=list)


# ─── LLM judge ────────────────────────────────────────────────────────────────

def _judge_system_blocks(section: str = "") -> List[Dict[str, Any]]:
    """System prompt for the LLM judge. Branched per section because SJT
    has fundamentally different scoring semantics: there is no single
    correct answer, only a Likert ranking against UCAT marking guidance.
    Using the generic "work out the correct answer" prompt for SJT
    produces false negatives at high rate (the judge invents a rubric
    that differs from UCAT's official one) and pollutes the calibration
    signal."""
    if section == "SJT":
        text = (
            "You are a meticulous UCAT Situational Judgement verifier. UCAT SJT "
            "is rubric-scored, not single-correct-answer — your job is to check "
            "whether the marked answer aligns with UCAT's official marking "
            "guidance, not to invent a 'correct' answer.\n\n"
            "UCAT SJT credit-scoring rubric:\n"
            "  - Top-tier label (the rubric's 'best' answer per UCAT principles): "
            "FULL credit — vote `correct: true`.\n"
            "  - Adjacent label (one Likert step from the top-tier): PARTIAL "
            "credit — still vote `correct: true`. UCAT awards partial marks "
            "here; treating it as wrong inflates false-negative rates.\n"
            "  - Two steps from the top-tier (e.g. 'Inappropriate but not "
            "awful' when 'Very appropriate' is the rubric answer): NO credit "
            "— vote `correct: false`.\n"
            "  - Opposite-end label: NO credit — vote `correct: false`.\n"
            "  - Mismatched label type (e.g. 'Very important' on an "
            "appropriateness question, or 'Very appropriate' on an importance "
            "question): NO credit — vote `correct: false`. Flag this in "
            "`reasoning` so the writer can fix the schema mismatch.\n\n"
            "For each question:\n\n"
            "1. Read the scenario and the marked answer (a Likert label).\n"
            "2. Determine which label is the top-tier answer per UCAT "
            "principles (professionalism, patient safety, honesty, team-working, "
            "scope of practice).\n"
            "3. Apply the rubric above to decide `correct: true/false`.\n"
            "4. Use `reasoning` to flag: ambiguous wording, scenarios where the "
            "answer is culturally obvious rather than UCAT-principled, or label "
            "type mismatches.\n"
            "5. Estimate difficulty on the 1.0-5.0 IRT logit scale where 1.0 = "
            "near-universal consensus answer and 5.0 = a genuinely contested "
            "judgement call requiring careful weighing of competing principles."
        )
    else:
        text = (
            "You are a meticulous UCAT exam verifier. You will be given a generated "
            "UCAT question set and must judge each question independently:\n\n"
            "1. Work out the correct answer yourself from the source material.\n"
            "2. Compare to the marked answer. Mark `correct: false` if they differ.\n"
            "3. Note any ambiguity, factual errors, or unfair distractors in `reasoning`.\n"
            "4. Estimate difficulty on the 1.0-5.0 IRT logit scale based on:\n"
            "   • multi-step reasoning required (more steps → harder)\n"
            "   • how close the distractors are to the correct answer\n"
            "   • vocabulary level and inferential subtlety (VR)\n"
            "   • computational complexity (QR)\n\n"
            "Be strict but calibrated. False positives destroy student trust; "
            "false negatives waste review time."
        )
    return [{
        "type": "text",
        "text": text,
        "cache_control": {"type": "ephemeral"},
    }]


def llm_judge(section: str, data: Dict[str, Any], model: str
              ) -> Tuple[_JudgeVerdict, Dict[str, Any]]:
    """Run a single LLM judge over one generated set."""
    client = LLMClient.anthropic()
    user = (
        f"Verify this UCAT {SECTIONS[section]} question set. Return per-question "
        f"verdicts plus your difficulty estimates.\n\n```json\n"
        f"{json.dumps(data, indent=2)}\n```"
    )
    kwargs = {
        "model": model,
        "max_tokens": 1500,
        "system": _judge_system_blocks(section),
        "messages": [{"role": "user", "content": user}],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": pydantic_to_strict_schema(_JudgeVerdict),
            }
        },
    }
    with client.messages.stream(**kwargs) as stream:
        msg = stream.get_final_message()
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        v = _JudgeVerdict.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning("Judge output unparseable: %s", e)
        v = _JudgeVerdict(per_question=[], overall_correct=True, confidence="low",
                          notes=[f"(verifier output unparseable: {e})"])
    return v, extract_usage(msg, model)


def _verdicts_unanimous(v1: Dict[str, Any], v2: Dict[str, Any]) -> bool:
    """True iff two verdicts agree on overall_correct and on every shared question."""
    if v1["overall_correct"] != v2["overall_correct"]:
        return False
    by_n_1 = {pq["number"]: pq["correct"] for pq in v1["per_question"]}
    by_n_2 = {pq["number"]: pq["correct"] for pq in v2["per_question"]}
    common = set(by_n_1) & set(by_n_2)
    if not common:
        return False
    return all(by_n_1[n] == by_n_2[n] for n in common)


def jury_verify(
    section: str,
    data: Dict[str, Any],
    *,
    judges: List[str],
) -> Dict[str, Any]:
    """
    Multi-judge verification. Each judge votes per-question; majority wins.
    Returns a dict containing the aggregated verdict, per-judge verdicts, and
    judge_predictions (for the calibration module).

    Optimisations: the first two judges run concurrently, and the third is
    skipped entirely when those two agree on every question — a clean batch
    pays max(t1, t2) instead of t1 + t2 + t3.
    """
    individual: List[Dict[str, Any]] = []
    usages: List[Dict[str, Any]] = []
    judge_predictions_per_q: Dict[int, List[float]] = {}
    failed_judges: List[Dict[str, str]] = []
    early_exit = False

    def _run_one(m: str) -> Optional[Tuple[_JudgeVerdict, Dict[str, Any]]]:
        # Returns None on per-judge failure (overload, timeout, parse error) so
        # that a single judge going down doesn't take the whole jury with it.
        with trace("judge", model=m, section=section):
            try:
                return llm_judge(section, data, m)
            except Exception as e:
                logger.warning("Judge %s failed: %s", m, e)
                failed_judges.append({"model": m, "error": str(e)})
                return None

    def _record(m: str, result: Optional[Tuple[_JudgeVerdict, Dict[str, Any]]]) -> bool:
        if result is None:
            return False
        v, u = result
        individual.append({"model": m, "verdict": v.model_dump()})
        usages.append(u)
        for pq in v.per_question:
            judge_predictions_per_q.setdefault(pq.number, []).append(pq.difficulty)
        return True

    if len(judges) >= 2:
        # Run the first two judges in parallel; defer the rest pending their agreement.
        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = [(m, ex.submit(_run_one, m)) for m in judges[:2]]
            results = [(m, fut.result()) for m, fut in futures]
        for m, r in results:
            _record(m, r)

        remaining = judges[2:]
        # Only short-circuit if both first judges actually returned and agreed.
        if (remaining
                and len(individual) >= 2
                and _verdicts_unanimous(individual[0]["verdict"], individual[1]["verdict"])):
            early_exit = True
        else:
            for m in remaining:
                _record(m, _run_one(m))
    else:
        for m in judges:
            _record(m, _run_one(m))

    # Per-question majority vote.
    questions = data.get("questions") or []
    flagged: List[int] = []
    correct_majority: List[bool] = []
    for q in questions:
        n = q.get("number")
        votes = []
        for j in individual:
            for pq in j["verdict"]["per_question"]:
                if pq["number"] == n:
                    votes.append(pq["correct"])
                    break
        if not votes:
            continue
        is_correct = sum(votes) > (len(votes) / 2)
        correct_majority.append(is_correct)
        if not is_correct:
            flagged.append(n)
    overall_correct = all(correct_majority) if correct_majority else True
    unanimous = all(j["verdict"]["overall_correct"] == overall_correct for j in individual)

    # Average judge difficulty per question.
    judge_predictions = {n: round(sum(vs) / len(vs), 2)
                         for n, vs in judge_predictions_per_q.items() if vs}

    return {
        "judges": judges,
        "individual": individual,
        "failed_judges": failed_judges,
        "overall_correct": overall_correct,
        "unanimous": unanimous,
        "early_exit": early_exit,
        "flagged_questions": flagged,
        "judge_predictions": judge_predictions,
        "usage": usages,
    }


# ─── Symbolic QR check ────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")
_LETTER_RE = re.compile(r"^([A-E])\b")


def _to_float(s: Any) -> Optional[float]:
    if s is None: return None
    m = _NUMBER_RE.search(str(s).replace(",", ""))
    if not m: return None
    try: return float(m.group())
    except ValueError: return None


def _flatten_chart_values(stim: Dict[str, Any]) -> Dict[str, float]:
    """Build a {category: total} or {series.cat: value} map from a chart stim."""
    out: Dict[str, float] = {}
    if not isinstance(stim, dict):
        return out
    cats = stim.get("categories") or []
    series = stim.get("series") or []
    for s in series:
        name = s.get("name", "")
        for i, v in enumerate(s.get("values") or []):
            if i < len(cats):
                out[f"{name}.{cats[i]}"] = float(v)
    rows = stim.get("rows")
    if isinstance(rows, dict):
        for col, vals in rows.items():
            for i, v in enumerate(vals or []):
                if i < len(cats):
                    f = _to_float(v)
                    if f is not None:
                        out[f"{col}.{cats[i]}"] = f
    return out


def _evaluate_explanation_arithmetic(expl: str) -> List[Dict[str, Any]]:
    """Find `LHS = RHS` segments in the explanation and verify the arithmetic.

    Handles chained equations like "Total = 45 + 38 = 79" by splitting on
    each `=` and checking whether each adjacent pair of arithmetic-shaped
    segments evaluates to equal values. Segments that don't parse as
    arithmetic (variable references, prose, "Total" labels) are skipped
    silently.

    Catches the common QR failure: Claude writes "45 + 38 = 79". The
    last `=` number 79 matches the marked option, but 45+38 is 83. The
    original symbolic check missed this because it only looked at the
    last `=` number — this layer evaluates the LHS arithmetic.
    """
    mismatches: List[Dict[str, Any]] = []
    if not _HAS_SYMPY:
        return mismatches
    import sympy

    def _arith_eval(text: str) -> Optional[float]:
        """Try to evaluate `text` as a pure arithmetic expression. Returns
        None if it doesn't look like arithmetic at all."""
        cleaned = (text.strip().replace(",", "")
                   .replace("×", "*").replace("÷", "/")
                   .replace("%", "/100"))
        # Reject if it contains anything that isn't an arithmetic char.
        if not cleaned or not re.fullmatch(r"[0-9.()+\-*/\s]+", cleaned):
            return None
        # Reject if it has no digits (e.g. just spaces or operators).
        if not re.search(r"\d", cleaned):
            return None
        try:
            return float(sympy.sympify(cleaned).evalf())
        except (sympy.SympifyError, TypeError, ValueError, ArithmeticError):
            return None

    for line in expl.splitlines():
        if "=" not in line:
            continue
        # Split on `=` and pair adjacent segments that BOTH look arithmetic.
        # For "Total = 45 + 38 = 79":
        #   segments = ["Total ", " 45 + 38 ", " 79"]
        #   pair 0/1: "Total" not arithmetic, skip
        #   pair 1/2: 45+38 = 83, 79 = 79 → 83 ≠ 79 → flag
        segments = line.split("=")
        for i in range(len(segments) - 1):
            lhs = _arith_eval(segments[i])
            rhs = _arith_eval(segments[i + 1])
            if lhs is None or rhs is None:
                continue
            if abs(lhs) < 1e-9 and abs(rhs) < 1e-9:
                continue
            rel = abs(lhs - rhs) / max(abs(rhs), 1e-9)
            if rel > 0.01:
                mismatches.append({
                    "line": line.strip(),
                    "lhs_computed": round(lhs, 4),
                    "rhs_claimed": round(rhs, 4),
                    "relative_error": round(rel, 4),
                })
    return mismatches


_LABEL_STOPWORDS = frozenset({
    "in", "of", "for", "the", "a", "an", "is", "was", "were",
    "are", "to", "by", "at", "on", "from",
})


def _label_keywords(label: str) -> List[str]:
    """Strip prose stopwords from a label, leaving the identifying tokens.
    "Sales in 2022" → ["sales", "2022"]."""
    return [
        w for w in re.findall(r"[A-Za-z0-9]+", label.lower())
        if w not in _LABEL_STOPWORDS
    ]


def _check_chart_lookups(expl: str, chart_values: Dict[str, float]) -> List[Dict[str, Any]]:
    """Cross-check chart-value claims in the explanation against actual data.

    If the explanation says "Sales in 2022 = 45" but the chart's
    flattened values show 2022 → 50, that's a chart-misread error.

    Label disambiguation requires ALL non-stopword keywords from the
    label to appear in the chart key. "Sales in 2021" matches
    "Sales.2021" (has both "sales" and "2021") but not "Sales.2022"
    (missing "2021"). This is stricter than a single-keyword match,
    which would conflate "Sales in 2021" with any series named "Sales"
    regardless of category, producing spurious flags.

    False positives are still possible (we'll flag derived intermediates
    that happen to match a chart key), but they appear in the verdict's
    `chart_misreads` list for human review, not as a hard `correct:
    false` vote.
    """
    misreads: List[Dict[str, Any]] = []
    if not chart_values:
        return misreads
    # Pre-tokenise chart keys for keyword-set matching.
    chart_key_tokens: Dict[str, set[str]] = {
        chart_key: set(re.findall(r"[A-Za-z0-9]+", chart_key.lower()))
        for chart_key in chart_values
    }
    pattern = re.compile(r"([A-Za-z][A-Za-z0-9 \-]{1,40}?)\s*[:=]\s*([\d.,]+)")
    seen_labels: set[str] = set()
    for match in pattern.finditer(expl):
        label = match.group(1).strip()
        if label.lower() in seen_labels:
            continue
        seen_labels.add(label.lower())
        try:
            claimed = float(match.group(2).replace(",", ""))
        except ValueError:
            continue
        keywords = set(_label_keywords(label))
        if not keywords:
            continue
        # Find chart keys whose token set is a SUPERSET of the label's
        # keywords. Pick the most specific (smallest token set) on tie.
        candidates = [
            (chart_key, chart_values[chart_key], len(toks))
            for chart_key, toks in chart_key_tokens.items()
            if keywords.issubset(toks)
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda c: c[2])  # smallest token set wins
        chart_key, chart_val, _ = candidates[0]
        if abs(chart_val) < 1e-9:
            continue
        rel = abs(chart_val - claimed) / max(abs(chart_val), 1e-9)
        if rel > 0.05:
            misreads.append({
                "label": label,
                "claimed": claimed,
                "chart_key": chart_key,
                "actual": chart_val,
                "relative_error": round(rel, 4),
            })
    return misreads


def symbolic_qr_check(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Re-evaluate QR question answers symbolically where possible.

    Three layers of checks, each producing actionable signals:

    1. **Explanation→option agreement** (the original check): the last
       `=` number in the explanation should match the marked option.
       Catches transcription errors.

    2. **Explanation arithmetic** (NEW): every `LHS = RHS` line in the
       explanation has its LHS evaluated by sympy and compared to RHS.
       Catches arithmetic errors like "45 + 38 = 79" where the LHS
       computes to 83 but Claude wrote 79 — even if 79 happens to
       match the marked option.

    3. **Chart misread** (NEW): values mentioned in the explanation
       under chart-like labels (e.g. "Sales in 2022 = 45") are
       cross-checked against the actual chart data via
       `_flatten_chart_values`. Catches the case where Claude reads
       a chart value incorrectly and derives a wrong answer from it.

    The check is conservative: layer 1 is the only one that promotes a
    question into the `disagreed` list (which can drive `correct: false`
    in the overall verdict). Layers 2 and 3 surface their findings in
    `arithmetic_errors` and `chart_misreads` for human review without
    auto-failing the verdict — false positives in those layers are
    plausible enough that gating verdicts on them would over-flag.
    """
    if not _HAS_SYMPY:
        return {
            "checked": 0, "agreed": 0, "disagreed": [],
            "arithmetic_errors": [], "chart_misreads": [],
            "skipped": "sympy not installed",
        }

    questions = data.get("questions") or []
    chart_values = _flatten_chart_values(data.get("stimulus") or {})

    checked = 0
    agreed = 0
    disagreed: List[Dict[str, Any]] = []
    arithmetic_errors: List[Dict[str, Any]] = []
    chart_misreads: List[Dict[str, Any]] = []

    for q in questions:
        expl = q.get("explanation", "")
        ans  = q.get("answer", "")
        opts = q.get("options") or {}

        # Layer 2: every line with `LHS = RHS` arithmetic, validated by sympy.
        per_q_arith = _evaluate_explanation_arithmetic(expl)
        for err in per_q_arith:
            err["number"] = q.get("number")
            arithmetic_errors.append(err)

        # Layer 3: chart-value lookups in the explanation, cross-checked.
        per_q_misreads = _check_chart_lookups(expl, chart_values)
        for mr in per_q_misreads:
            mr["number"] = q.get("number")
            chart_misreads.append(mr)

        # Layer 1: the original check — explanation's final `=` number
        # vs marked option.
        last_num = None
        for line in reversed(expl.splitlines()):
            if "=" in line:
                m = _NUMBER_RE.findall(line.replace(",", ""))
                if m:
                    try:
                        last_num = float(m[-1])
                        break
                    except ValueError:
                        pass
        if last_num is None:
            m = _NUMBER_RE.findall(expl.replace(",", ""))
            if m:
                try: last_num = float(m[-1])
                except ValueError: last_num = None

        opt_letter = ans.strip()[:1].upper() if isinstance(ans, str) else ""
        marked_val = _to_float(opts.get(opt_letter))

        if last_num is None or marked_val is None:
            continue

        checked += 1
        rel = abs(last_num - marked_val) / max(abs(marked_val), 1e-9)
        if rel < 0.01:
            agreed += 1
        else:
            disagreed.append({
                "number": q.get("number"),
                "marked_answer": ans,
                "marked_value": marked_val,
                "computed_value": last_num,
                "relative_error": round(rel, 4),
            })

    return {
        "checked": checked,
        "agreed": agreed,
        "disagreed": disagreed,
        "arithmetic_errors": arithmetic_errors,
        "chart_misreads": chart_misreads,
    }
