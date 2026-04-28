"""
Quality assurance pipeline: schema validation, content validation,
KB dedup, session dedup, rule-based scoring, and independent LLM scoring.
"""

import json
import re

from src.config import (
    SECTION_SCHEMAS, SECTIONS, QUALITY_THRESHOLDS, DEFAULT_SCORER, cosine_sim,
    MINIGAME_HINT_REQUIREMENTS,
)


# ─── Minigame Hint Validators ────────────────────────────────────────────────
# Each validator returns a list of error strings (empty = valid).

def _validate_inference_hints(h: dict) -> list:
    statements = h.get("statements")
    if not isinstance(statements, list) or len(statements) != 4:
        return ["minigame_hints.statements: expected exactly 4 entries"]
    errs = []
    implied = 0
    for i, s in enumerate(statements, 1):
        if not isinstance(s, dict) or not s.get("text") or "is_implied" not in s:
            errs.append(f"minigame_hints.statements[{i}]: needs {{text, is_implied}}")
        elif s.get("is_implied") is True:
            implied += 1
    if implied != 1:
        errs.append(f"minigame_hints.statements: expected exactly 1 with is_implied=true, got {implied}")
    return errs


def _validate_syllogism_hints(h: dict) -> list:
    premises = h.get("premises")
    if not isinstance(premises, list) or len(premises) != 3:
        return ["minigame_hints.premises: expected exactly 3 string premises"]
    if not all(isinstance(p, str) and p.strip() for p in premises):
        return ["minigame_hints.premises: every premise must be a non-empty string"]
    return []


def _validate_argument_hints(h: dict) -> list:
    args = h.get("arguments")
    if not isinstance(args, list) or len(args) != 4:
        return ["minigame_hints.arguments: expected exactly 4 entries"]
    errs = []
    valid_verdicts = {"strong", "weak", "irrelevant"}
    for i, a in enumerate(args, 1):
        if not isinstance(a, dict) or not a.get("text"):
            errs.append(f"minigame_hints.arguments[{i}]: missing text")
        elif a.get("verdict") not in valid_verdicts:
            errs.append(f"minigame_hints.arguments[{i}]: verdict must be one of {sorted(valid_verdicts)}")
    return errs


def _validate_venn_hints(h: dict) -> list:
    if not h.get("set_a") or not h.get("set_b"):
        return ["minigame_hints: venn requires both set_a and set_b labels"]
    items = h.get("items")
    if not isinstance(items, list) or len(items) != 6:
        return ["minigame_hints.items: expected exactly 6 entries"]
    errs = []
    valid_regions = {"A", "B", "both", "neither"}
    for i, it in enumerate(items, 1):
        if not isinstance(it, dict) or not it.get("text"):
            errs.append(f"minigame_hints.items[{i}]: missing text")
        elif it.get("region") not in valid_regions:
            errs.append(f"minigame_hints.items[{i}]: region must be one of {sorted(valid_regions)}")
    return errs


def _validate_data_table_hints(h: dict) -> list:
    headers = h.get("headers")
    rows = h.get("rows")
    errs = []
    if not isinstance(headers, list) or len(headers) < 2:
        errs.append("minigame_hints.headers: expected ≥2 column names")
    if not isinstance(rows, list) or len(rows) < 1:
        errs.append("minigame_hints.rows: expected ≥1 row")
    elif headers and isinstance(headers, list):
        for i, row in enumerate(rows, 1):
            if not isinstance(row, list) or len(row) != len(headers):
                errs.append(f"minigame_hints.rows[{i}]: row width does not match headers")
                break
    return errs


def _validate_role_id_hints(h: dict) -> list:
    role = h.get("role")
    if role not in ("doctor", "nurse", "student"):
        return ["minigame_hints.role: must be 'doctor', 'nurse', or 'student'"]
    actions = h.get("actions")
    if not isinstance(actions, list) or len(actions) != 4:
        return ["minigame_hints.actions: expected exactly 4 entries"]
    errs = []
    for i, a in enumerate(actions, 1):
        if not isinstance(a, dict) or not a.get("text") or "in_role" not in a:
            errs.append(f"minigame_hints.actions[{i}]: needs {{text, in_role}}")
    return errs


def _validate_values_sorter_hints(h: dict) -> list:
    actions = h.get("actions")
    if not isinstance(actions, list) or len(actions) != 6:
        return ["minigame_hints.actions: expected exactly 6 entries"]
    errs = []
    valid_pillars = {"beneficence", "nonMaleficence", "justice", "autonomy"}
    for i, a in enumerate(actions, 1):
        if not isinstance(a, dict) or not a.get("text"):
            errs.append(f"minigame_hints.actions[{i}]: missing text")
        elif a.get("pillar") not in valid_pillars:
            errs.append(f"minigame_hints.actions[{i}]: pillar must be one of {sorted(valid_pillars)}")
    return errs


_HINT_VALIDATORS = {
    "inference": _validate_inference_hints,
    "syllogism": _validate_syllogism_hints,
    "argument-strength": _validate_argument_hints,
    "venn": _validate_venn_hints,
    "data-table": _validate_data_table_hints,
    "role-identification": _validate_role_id_hints,
    "values-sorter": _validate_values_sorter_hints,
}


def validate_minigame_hints(kind: str, hints: dict) -> list:
    """Returns errors from the kind-specific validator, or [] if no validator."""
    validator = _HINT_VALIDATORS.get(kind)
    if not validator or not isinstance(hints, dict):
        return []
    return validator(hints)


# ─── Session Dedup Cache ────────────────────────────────────────────────────

class SessionDedupCache:
    """Tracks embeddings from the current session to prevent near-duplicates."""

    def __init__(self):
        self.embeddings = []  # list of (section, embedding_vector)

    def check(self, section: str, new_embedding: list,
              threshold: float = None) -> tuple:
        """
        Check if new embedding is too similar to a recent session generation.
        Returns (is_duplicate, max_similarity).
        """
        if threshold is None:
            threshold = QUALITY_THRESHOLDS["dedup_session"]
        max_sim = 0.0
        for s, emb in self.embeddings:
            if s == section:
                sim = cosine_sim(new_embedding, emb)
                max_sim = max(max_sim, sim)
                if sim > threshold:
                    return True, sim
        return False, max_sim

    def add(self, section: str, embedding: list):
        self.embeddings.append((section, embedding))

    def clear(self):
        self.embeddings.clear()


# ─── Schema Validation (Pure Python) ─────────────────────────────────────────

def validate_schema(data: dict, section: str) -> list:
    """
    Validate generated data against section schema.
    Returns list of error strings (empty = valid).
    """
    errors = []
    schema = SECTION_SCHEMAS.get(section)
    if not schema:
        errors.append(f"Unknown section: {section}")
        return errors

    # Check required keys
    for key in schema.get("required_keys", []):
        if key not in data:
            errors.append(f"Missing required key: '{key}'")

    # Check question count
    expected_count = schema.get("question_count")
    questions = data.get("questions", [])
    if expected_count and len(questions) != expected_count:
        errors.append(
            f"Expected {expected_count} questions, got {len(questions)}"
        )

    # Section-specific validations
    if section == "VR":
        passage = data.get("passage", "")
        min_len = schema.get("min_passage_len", 100)
        if len(passage) < min_len:
            errors.append(f"Passage too short ({len(passage)} chars, need ≥{min_len})")

    elif section == "QR":
        stimulus = data.get("stimulus", "")
        min_len = schema.get("min_stimulus_len", 20)
        if len(stimulus) < min_len:
            errors.append(f"Stimulus too short ({len(stimulus)} chars, need ≥{min_len})")

    elif section == "SJT":
        scenario = data.get("scenario", "")
        min_len = schema.get("min_scenario_len", 50)
        if len(scenario) < min_len:
            errors.append(f"Scenario too short ({len(scenario)} chars, need ≥{min_len})")

    # Validate each question
    for i, q in enumerate(questions):
        qnum = q.get("number", i + 1)

        if not q.get("text"):
            errors.append(f"Q{qnum}: missing question text")

        options = q.get("options", {})
        if not options:
            errors.append(f"Q{qnum}: missing options")

        answer = q.get("answer")
        if answer and options and answer not in options:
            errors.append(f"Q{qnum}: answer '{answer}' not in options {list(options.keys())}")

        # Check for empty or placeholder options
        for opt_key, opt_val in options.items():
            if not opt_val or opt_val.strip() in ("...", ""):
                errors.append(f"Q{qnum}: option {opt_key} is empty or placeholder")

        # Check for duplicate option values
        opt_values = [str(v).strip().lower() for v in options.values()]
        if len(opt_values) != len(set(opt_values)):
            errors.append(f"Q{qnum}: has duplicate option values")

        # DM: validate question type
        if section == "DM":
            valid_types = SECTION_SCHEMAS["DM"].get("valid_types", [])
            qtype = q.get("type", "")
            if valid_types and qtype and qtype not in valid_types:
                errors.append(f"Q{qnum}: invalid type '{qtype}', expected one of {valid_types}")

        # All sections: minigame_kind, when present, must match the section's catalogue.
        # Missing minigame_kind is a warning (handled in validate_content), not an error,
        # so old generations remain valid.
        valid_kinds = schema.get("valid_minigame_kinds", [])
        kind = q.get("minigame_kind")
        if valid_kinds and kind and kind not in valid_kinds:
            errors.append(
                f"Q{qnum}: invalid minigame_kind '{kind}', expected one of {valid_kinds}"
            )

        # Structured hints: when minigame_kind requires hints, validate their shape.
        # Missing hints is a warning (handled in validate_content); malformed hints
        # are a hard error so the LLM gets a corrective retry.
        if kind and kind in MINIGAME_HINT_REQUIREMENTS:
            hints = q.get("minigame_hints")
            if hints is not None:
                hint_errs = validate_minigame_hints(kind, hints)
                for he in hint_errs:
                    errors.append(f"Q{qnum}: {he}")

    return errors


# ─── Content Validation (Pure Python) ────────────────────────────────────────

def validate_content(data: dict, section: str) -> list:
    """
    Check for placeholder text, suspicious patterns, and content issues.
    Returns list of warning strings.
    """
    warnings = []
    text_blob = json.dumps(data).lower()

    # Check for placeholder text
    placeholders = ["lorem ipsum", "example text", "placeholder", "todo",
                    "insert here", "your text", "sample passage"]
    for ph in placeholders:
        if ph in text_blob:
            warnings.append(f"Contains placeholder text: '{ph}'")

    # Check for ellipsis patterns suggesting incomplete generation
    if text_blob.count("...") > 3:
        warnings.append("Contains excessive ellipsis (...) — may be incomplete")

    # VR: check passage has sufficient words
    if section == "VR":
        passage = data.get("passage", "")
        word_count = len(passage.split())
        if word_count < 80:
            warnings.append(f"Passage only {word_count} words (target: 200-300)")
        elif word_count > 500:
            warnings.append(f"Passage is {word_count} words (target: 200-300)")

    # Check all questions have explanations
    questions = data.get("questions", [])
    missing_exp = sum(1 for q in questions if not q.get("explanation"))
    if missing_exp > 0:
        warnings.append(f"{missing_exp} question(s) missing explanations")

    # Warn (don't error) if minigame_kind is missing — keeps old generations valid
    # while nudging the LLM to populate the field on retry/new runs.
    schema = SECTION_SCHEMAS.get(section, {})
    if schema.get("valid_minigame_kinds"):
        missing_kind = sum(1 for q in questions if not q.get("minigame_kind"))
        if missing_kind > 0:
            warnings.append(
                f"{missing_kind} question(s) missing minigame_kind — Pocket UCAT importer "
                "will fall back to keyword heuristics for those"
            )

    # Warn when a question's minigame_kind requires hints but none are provided.
    # Without hints, the importer falls back to a generic MCQ rendering.
    missing_hints = []
    for q in questions:
        kind = q.get("minigame_kind")
        if kind in MINIGAME_HINT_REQUIREMENTS and not q.get("minigame_hints"):
            missing_hints.append(f"Q{q.get('number', '?')}({kind})")
    if missing_hints:
        warnings.append(
            "Missing minigame_hints (will render as generic MCQ): "
            + ", ".join(missing_hints)
        )

    return warnings


# ─── KB Dedup Check ──────────────────────────────────────────────────────────

def check_kb_dedup(new_embedding: list, db, section: str,
                   threshold: float = None) -> tuple:
    """
    Check if new embedding is too similar to existing KB documents.
    Returns (is_duplicate, max_similarity, closest_doc_id).
    """
    if threshold is None:
        threshold = QUALITY_THRESHOLDS["dedup_kb"]

    docs = db.get_all_docs(section)
    max_sim = 0.0
    closest_id = None

    for doc in docs:
        if doc.get("embedding"):
            sim = cosine_sim(new_embedding, doc["embedding"])
            if sim > max_sim:
                max_sim = sim
                closest_id = doc["id"]

    return max_sim > threshold, max_sim, closest_id


# ─── Rule-Based Scoring ─────────────────────────────────────────────────────

def rule_score(data: dict, section: str) -> dict:
    """
    Score a generated question set using code-checkable rules.
    Returns dict with individual scores and overall (0.0-1.0).
    """
    checks = {}

    # 1. Schema validity
    schema_errors = validate_schema(data, section)
    checks["schema_valid"] = 1.0 if not schema_errors else 0.0

    # 2. Content validity
    content_warnings = validate_content(data, section)
    checks["content_clean"] = 1.0 if not content_warnings else max(0.0, 1.0 - 0.2 * len(content_warnings))

    questions = data.get("questions", [])

    # 3. All questions have text
    has_text = sum(1 for q in questions if q.get("text")) / max(1, len(questions))
    checks["questions_have_text"] = has_text

    # 4. All questions have valid answers
    valid_answers = 0
    for q in questions:
        if q.get("answer") and q.get("options") and q["answer"] in q["options"]:
            valid_answers += 1
    checks["answers_valid"] = valid_answers / max(1, len(questions))

    # 5. All questions have explanations
    has_exp = sum(1 for q in questions if q.get("explanation")) / max(1, len(questions))
    checks["has_explanations"] = has_exp

    # 6. Option uniqueness (no duplicate values)
    unique_opts = 0
    for q in questions:
        opts = list(q.get("options", {}).values())
        if len(opts) == len(set(str(v).strip().lower() for v in opts)):
            unique_opts += 1
    checks["options_unique"] = unique_opts / max(1, len(questions))

    # 7. No placeholder text
    text_blob = json.dumps(data).lower()
    no_placeholder = 1.0 if "..." not in text_blob[:50] and "lorem" not in text_blob else 0.0
    checks["no_placeholders"] = no_placeholder

    # 8. Correct question count
    expected = SECTION_SCHEMAS.get(section, {}).get("question_count", 0)
    checks["correct_count"] = 1.0 if len(questions) == expected else 0.0

    # Overall: average of all checks
    overall = sum(checks.values()) / len(checks) if checks else 0.0
    checks["overall"] = overall

    return checks


# ─── LLM-Based Quality Scoring ──────────────────────────────────────────────

def build_scoring_prompt(data: dict, section: str) -> tuple:
    """Build prompt for the independent LLM scorer."""
    data_json = json.dumps(data, indent=2)[:3000]

    system = (
        f"You are a UCAT {SECTIONS[section]} question quality assessor. "
        "Rate the following question set on a 1-5 scale for each criterion. "
        "Be critical — only rate 5 for genuinely excellent work.\n\n"
        "Return JSON with these keys:\n"
        '{"passage_quality": N, "question_quality": N, "distractor_quality": N, '
        '"answer_correctness": N, "authenticity": N, "overall": N}\n\n'
        "Criteria:\n"
        "- passage_quality: Clarity, appropriate length, academic tone, information density\n"
        "- question_quality: Tests comprehension/reasoning, unambiguous, appropriate difficulty\n"
        "- distractor_quality: Each wrong option is plausible but definitively wrong\n"
        "- answer_correctness: The marked answer is unambiguously correct\n"
        "- authenticity: Feels like a real UCAT question (format, style, difficulty)\n"
        "- overall: Holistic assessment of the entire question set"
    )

    user = f"Rate this UCAT {SECTIONS[section]} question set:\n\n{data_json}"

    return system, user


def parse_llm_score(response: str) -> dict:
    """Parse the LLM scorer's response into scores."""
    try:
        scores = json.loads(response)
        expected_keys = ["passage_quality", "question_quality", "distractor_quality",
                         "answer_correctness", "authenticity", "overall"]
        for key in expected_keys:
            if key not in scores:
                scores[key] = 3.0  # Default to middle if missing
            scores[key] = max(1.0, min(5.0, float(scores[key])))
        return scores
    except Exception:
        return {
            "passage_quality": 3.0, "question_quality": 3.0,
            "distractor_quality": 3.0, "answer_correctness": 3.0,
            "authenticity": 3.0, "overall": 3.0,
        }


# ─── QR Answer Verification ─────────────────────────────────────────────────

def build_qr_verification_prompt(data: dict) -> tuple:
    """Build prompt to verify QR mathematical answers."""
    system = (
        "You are a mathematics verification assistant. "
        "For each question, re-solve the problem step by step using the provided data. "
        "Verify whether the claimed answer is correct.\n\n"
        "Return JSON: {\"verifications\": [{\"question\": N, \"correct\": true/false, \"expected\": \"X\", \"reason\": \"...\"}]}"
    )

    stimulus = data.get("stimulus", "")
    questions_text = []
    for q in data.get("questions", []):
        questions_text.append(
            f"Q{q.get('number', '?')}: {q.get('text', '')}\n"
            f"Options: {json.dumps(q.get('options', {}))}\n"
            f"Claimed answer: {q.get('answer', '?')}"
        )

    user = f"Data:\n{stimulus}\n\nQuestions:\n" + "\n\n".join(questions_text)
    return system, user


# ─── Full Quality Pipeline ───────────────────────────────────────────────────

class QualityPipeline:
    """Orchestrates the full quality assessment for generated questions."""

    def __init__(self, ollama_client, db, scorer_model: str = DEFAULT_SCORER):
        self.ollama = ollama_client
        self.db = db
        self.scorer_model = scorer_model
        self.session_cache = SessionDedupCache()

    def validate(self, data: dict, section: str) -> list:
        """Run schema + content validation. Returns error list."""
        errors = validate_schema(data, section)
        warnings = validate_content(data, section)
        return errors + warnings

    def check_dedup(self, embedding: list, section: str) -> dict:
        """Check both KB and session dedup."""
        kb_dup, kb_sim, kb_id = check_kb_dedup(embedding, self.db, section)
        session_dup, session_sim = self.session_cache.check(section, embedding)
        return {
            "kb_duplicate": kb_dup,
            "kb_similarity": kb_sim,
            "kb_closest_id": kb_id,
            "session_duplicate": session_dup,
            "session_similarity": session_sim,
        }

    def score_rules(self, data: dict, section: str) -> dict:
        """Rule-based scoring (no LLM call)."""
        return rule_score(data, section)

    def score_llm(self, data: dict, section: str) -> dict:
        """Independent LLM scoring using the scorer model."""
        try:
            system, user = build_scoring_prompt(data, section)
            response = self.ollama.generate(
                system, user, self.scorer_model,
                options={"temperature": 0.1, "num_predict": 500}
            )
            return parse_llm_score(response)
        except Exception:
            return {"overall": 3.0}

    def verify_qr_answers(self, data: dict) -> dict:
        """Verify QR mathematical answers using the scorer model."""
        try:
            system, user = build_qr_verification_prompt(data)
            response = self.ollama.generate(
                system, user, self.scorer_model,
                options={"temperature": 0.1, "num_predict": 1000}
            )
            return json.loads(response)
        except Exception:
            return {"verifications": []}

    def full_assessment(self, data: dict, section: str,
                        embedding: list = None) -> dict:
        """
        Run the complete quality pipeline.
        Returns comprehensive quality report.
        """
        # 1. Schema + content validation
        errors = validate_schema(data, section)
        warnings = validate_content(data, section)

        # 2. Dedup checks
        dedup = {}
        if embedding:
            dedup = self.check_dedup(embedding, section)

        # 3. Rule-based scoring
        rules = self.score_rules(data, section)

        # 4. LLM scoring (independent model)
        llm_scores = self.score_llm(data, section)

        # 5. QR answer verification
        qr_verification = None
        if section == "QR" and not errors:
            qr_verification = self.verify_qr_answers(data)

        # 6. Compute final score
        rule_overall = rules.get("overall", 0.0)
        llm_overall = llm_scores.get("overall", 3.0) / 5.0  # Normalize to 0-1
        final = (QUALITY_THRESHOLDS["rule_weight"] * rule_overall
                 + QUALITY_THRESHOLDS["llm_weight"] * llm_overall)

        return {
            "errors": errors,
            "warnings": warnings,
            "dedup": dedup,
            "rule_scores": rules,
            "rule_score": rule_overall,
            "llm_scores": llm_scores,
            "llm_score": llm_scores.get("overall", 3.0),
            "final_score": final,
            "qr_verification": qr_verification,
            "format_valid": len(errors) == 0,
        }
