"""
Configuration constants for UCAT Trainer · RAG Edition.
All tunable parameters, section schemas, model configs, retrieval strategies,
embedding prefixes, context budgets, and quality thresholds.
"""

import math

# ─── Application ─────────────────────────────────────────────────────────────

APP_TITLE     = "UCAT Trainer · RAG"
DB_FILE       = "ucat_rag.db"
OLLAMA_BASE   = "http://localhost:11434"

# ─── Default Models ──────────────────────────────────────────────────────────

DEFAULT_LLM       = "qwen2.5:14b"
DEFAULT_EMBED     = "mxbai-embed-large"
DEFAULT_SCORER    = "qwen2.5:7b"       # Independent from generator for quality scoring
DEFAULT_VISION    = "qwen2.5vl"        # Vision OCR for screenshot ingestion
FALLBACK_LLM      = "qwen2.5:7b"      # Low-RAM fallback
FALLBACK_SCORER   = "qwen2.5:1.5b"
FALLBACK_VISION   = "llava"

# ─── Sections ────────────────────────────────────────────────────────────────

SECTIONS = {
    "VR": "Verbal Reasoning",
    "DM": "Decision Making",
    "QR": "Quantitative Reasoning",
    "SJT": "Situational Judgement",
}

SECTION_COLORS = {
    "VR": "#4A90D9",
    "DM": "#E8943A",
    "QR": "#3FB950",
    "SJT": "#A78BFA",
}

SECTION_SCHEMAS = {
    "VR": {
        "desc": (
            "A passage (200-300 words) followed by exactly 4 questions. "
            "Each question is either True/False/Can't Tell OR 4-option multiple choice (A-D). "
            "Questions must be answerable ONLY from the passage."
        ),
        "shape": (
            '{"section":"VR","passage":"...","questions":['
            '{"number":1,"text":"...","type":"tf","options":{"A":"True","B":"False","C":"Can\'t Tell"},"answer":"A","explanation":"..."},'
            '{"number":2,"text":"...","type":"mc","options":{"A":"...","B":"...","C":"...","D":"..."},"answer":"C","explanation":"..."}]}'
        ),
        "required_keys": ["passage", "questions"],
        "question_count": 4,
        "min_passage_len": 100,
    },
    "DM": {
        "desc": (
            "Exactly 5 standalone questions. Each is one of: syllogism, logical (clue-based), "
            "venn (set relationships), probability, or argument (strongest argument for/against). "
            "Each has 5 options (A-E)."
        ),
        "shape": (
            '{"section":"DM","questions":['
            '{"number":1,"type":"syllogism","text":"...","options":{"A":"...","B":"...","C":"...","D":"...","E":"..."},"answer":"B","explanation":"..."}]}'
        ),
        "required_keys": ["questions"],
        "question_count": 5,
        "valid_types": ["syllogism", "logical", "venn", "probability", "argument"],
    },
    "QR": {
        "desc": (
            "One data stimulus (table or chart as markdown text) followed by exactly 4 calculation questions. "
            "Each question has 5 numerical options (A-E). Include step-by-step working in each explanation."
        ),
        "shape": (
            '{"section":"QR","stimulus":"Title\\n\\n| Col | Col |\\n|-----|-----|\\n| val | val |",'
            '"questions":[{"number":1,"text":"...","options":{"A":"12","B":"14","C":"16","D":"18","E":"20"},"answer":"C","explanation":"step...=16"}]}'
        ),
        "required_keys": ["stimulus", "questions"],
        "question_count": 4,
        "min_stimulus_len": 20,
    },
    "SJT": {
        "desc": (
            "A scenario (80-150 words) describing a realistic professional or interpersonal situation. "
            "Followed by exactly 4 questions. Each question asks what you would do or how appropriate "
            "an action is, with 4 options ranked from Most appropriate to Least appropriate (A-D)."
        ),
        "shape": (
            '{"section":"SJT","scenario":"...",'
            '"questions":[{"number":1,"text":"...","options":{"A":"...","B":"...","C":"...","D":"..."},'
            '"answer":"A","appropriateness":["most","appropriate","inappropriate","most_inappropriate"],"explanation":"..."}]}'
        ),
        "required_keys": ["scenario", "questions"],
        "question_count": 4,
        "min_scenario_len": 50,
    },
}

# ─── Section-Specific Generation Parameters ──────────────────────────────────
# Lower temperature for logic/math, higher for creative passages.

SECTION_GEN_PARAMS = {
    "VR": {
        "temperature": 0.78,
        "top_p": 0.90,
        "num_predict": 3200,
        "preferred_llm": "qwen2.5:14b",
    },
    "DM": {
        "temperature": 0.65,
        "top_p": 0.85,
        "num_predict": 3000,
        "preferred_llm": "deepseek-r1:14b",
    },
    "QR": {
        "temperature": 0.55,
        "top_p": 0.80,
        "num_predict": 3500,
        "preferred_llm": "deepseek-r1:14b",
    },
    "SJT": {
        "temperature": 0.75,
        "top_p": 0.88,
        "num_predict": 3000,
        "preferred_llm": "qwen2.5:14b",
    },
}

# ─── Per-Section Retrieval Strategies ────────────────────────────────────────

RETRIEVAL_STRATEGIES = {
    "VR": {
        "mmr_lambda": 0.5,       # High diversity — prevent repeated themes
        "diversity_key": "topic_cluster",
        "top_k": 8,
        "description": "High topic diversity to prevent repeated themes",
    },
    "DM": {
        "mmr_lambda": 0.7,
        "diversity_key": "data_type",
        "top_k": 8,
        "ensure_type_coverage": True,  # Ensure syllogism/probability/logical/argument/venn mix
        "description": "Ensure question-type coverage across all DM sub-types",
    },
    "QR": {
        "mmr_lambda": 0.7,
        "diversity_key": "data_type",
        "top_k": 6,
        "description": "Mix calculation types (%, ratio, rate, area)",
    },
    "SJT": {
        "mmr_lambda": 0.5,
        "diversity_key": "data_type",
        "top_k": 8,
        "description": "Mix scenario themes (teamwork, ethics, patient care, conflict)",
    },
}

# ─── Instruction Prefixes for Embeddings ─────────────────────────────────────
# mxbai-embed-large performs measurably better with instruction prefixes.

EMBED_INSTRUCTIONS = {
    "VR": "Represent this UCAT verbal reasoning passage for retrieval: ",
    "DM": "Represent this UCAT decision making logic problem for retrieval: ",
    "QR": "Represent this quantitative data table and calculation questions for retrieval: ",
    "SJT": "Represent this situational judgement scenario for retrieval: ",
}

# ─── Context Window Budget ───────────────────────────────────────────────────
# Prevents silent truncation when retrieved context exceeds model limits.

CONTEXT_BUDGET = {
    "system_overhead": 800,     # Tokens for system prompt scaffolding
    "user_prompt": 200,         # Tokens for user prompt
    "output_reserve": 3500,     # Tokens reserved for generation output
    "min_context_docs": 3,      # Minimum docs to include even if tight
    "chars_per_token": 4,       # Rough estimate for budget math
    "default_ctx": 4096,        # Ollama default context window
}

# ─── Quality Thresholds ─────────────────────────────────────────────────────

QUALITY_THRESHOLDS = {
    "dedup_kb": 0.92,           # Max similarity to KB doc before rejection
    "dedup_session": 0.85,      # Max similarity to session-generated doc
    "auto_promote_rule": 0.9,   # Minimum rule-based score for auto-promote
    "auto_promote_llm": 4.0,    # Minimum LLM score (out of 5) for auto-promote
    "rule_weight": 0.6,         # Weight of rule-based score in final score
    "llm_weight": 0.4,          # Weight of LLM score in final score
}

# ─── Calibration ─────────────────────────────────────────────────────────────

CALIBRATION_THRESHOLD = 5       # Consecutive approvals to graduate a section

# ─── Feedback ────────────────────────────────────────────────────────────────

FEEDBACK = {
    "ema_alpha": 0.2,           # EMA decay for implicit signals
    "explicit_weight": 0.6,     # Weight of explicit feedback in composite
    "implicit_weight": 0.4,     # Weight of implicit feedback in composite
    "long_view_threshold_ms": 30000,  # 30 seconds = positive implicit signal
}

# ─── Batch Export ────────────────────────────────────────────────────────────

BATCH_EXPORT_FORMATS = ["json", "csv", "pdf"]

DIFFICULTY_MAP = {
    "easy": 3,
    "medium": 5,
    "hard": 8,
}

# ─── PDF Runtime Guard ───────────────────────────────────────────────────────

try:
    import reportlab  # noqa: F401
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# ─── Utilities ───────────────────────────────────────────────────────────────

def cosine_sim(a: list, b: list) -> float:
    """Cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    ma = math.sqrt(sum(x * x for x in a))
    mb = math.sqrt(sum(x * x for x in b))
    return dot / (ma * mb) if ma and mb else 0.0
