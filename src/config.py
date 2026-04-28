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
            '{"number":1,"text":"...","type":"tf","minigame_kind":"tfc","options":{"A":"True","B":"False","C":"Can\'t Tell"},"answer":"A","explanation":"..."},'
            '{"number":2,"text":"...","type":"mc","minigame_kind":"main-idea","options":{"A":"...","B":"...","C":"...","D":"..."},"answer":"C","explanation":"..."}]}'
        ),
        "required_keys": ["passage", "questions"],
        "question_count": 4,
        "min_passage_len": 100,
        "valid_minigame_kinds": ["tfc", "main-idea", "paraphrase", "tone-purpose", "inference"],
    },
    "DM": {
        "desc": (
            "Exactly 5 standalone questions. Each is one of: syllogism, logical (clue-based), "
            "venn (set relationships), probability, or argument (strongest argument for/against). "
            "Each has 5 options (A-E)."
        ),
        "shape": (
            '{"section":"DM","questions":['
            '{"number":1,"type":"syllogism","minigame_kind":"syllogism","text":"...","options":{"A":"...","B":"...","C":"...","D":"...","E":"..."},"answer":"B","explanation":"..."}]}'
        ),
        "required_keys": ["questions"],
        "question_count": 5,
        "valid_types": ["syllogism", "logical", "venn", "probability", "argument"],
        "valid_minigame_kinds": [
            "syllogism", "logic-grid", "venn", "probability", "argument-strength", "assumption",
        ],
    },
    "QR": {
        "desc": (
            "One data stimulus (table or chart as markdown text) followed by exactly 4 calculation questions. "
            "Each question has 5 numerical options (A-E). Include step-by-step working in each explanation."
        ),
        "shape": (
            '{"section":"QR","stimulus":"Title\\n\\n| Col | Col |\\n|-----|-----|\\n| val | val |",'
            '"questions":[{"number":1,"text":"...","minigame_kind":"data-table","options":{"A":"12","B":"14","C":"16","D":"18","E":"20"},"answer":"C","explanation":"step...=16"}]}'
        ),
        "required_keys": ["stimulus", "questions"],
        "question_count": 4,
        "min_stimulus_len": 20,
        "valid_minigame_kinds": [
            "rapid-estimation", "data-table", "ratio", "fraction", "graph-grab", "chart-sprint",
        ],
    },
    "SJT": {
        "desc": (
            "A scenario (80-150 words) describing a realistic professional or interpersonal situation. "
            "Followed by exactly 4 questions. Each question asks what you would do or how appropriate "
            "an action is, with 4 options ranked from Most appropriate to Least appropriate (A-D)."
        ),
        "shape": (
            '{"section":"SJT","scenario":"...",'
            '"questions":[{"number":1,"text":"...","minigame_kind":"appropriateness","options":{"A":"...","B":"...","C":"...","D":"..."},'
            '"answer":"A","appropriateness":["most","appropriate","inappropriate","most_inappropriate"],"explanation":"..."}]}'
        ),
        "required_keys": ["scenario", "questions"],
        "question_count": 4,
        "min_scenario_len": 50,
        "valid_minigame_kinds": [
            "appropriateness", "importance", "empathy", "escalation", "role-identification",
        ],
    },
}

# ─── Minigame Kind Catalogue ─────────────────────────────────────────────────
# All valid `minigame_kind` values, exposed for prompt injection so the LLM
# is told the exact accepted vocabulary at generation time.

ALL_MINIGAME_KINDS = sorted({
    k for sch in SECTION_SCHEMAS.values() for k in sch.get("valid_minigame_kinds", [])
})

# Structured hint requirements per minigame kind. Kinds in this dict need
# extra fields under `minigame_hints` to render correctly in the Pocket UCAT
# app. Kinds NOT in this dict are pure MCQs and need no hints.
MINIGAME_HINT_REQUIREMENTS = {
    "inference": {
        "description": (
            "List exactly 4 candidate statements about the passage; tag each as "
            "directly stated (is_implied: false) or implied (is_implied: true). "
            "Exactly ONE should be implied."
        ),
        "shape": (
            '"minigame_hints":{"statements":['
            '{"text":"Statement 1","is_implied":false},'
            '{"text":"Statement 2","is_implied":true},'
            '{"text":"Statement 3","is_implied":false},'
            '{"text":"Statement 4","is_implied":false}]}'
        ),
    },
    "syllogism": {
        "description": (
            "Provide exactly 3 premises as a separate array. The standard 4-5 "
            "options remain as conclusion choices."
        ),
        "shape": (
            '"minigame_hints":{"premises":['
            '"All A are B.","Some C are A.","Therefore..."]}'
        ),
    },
    "argument-strength": {
        "description": (
            "Provide exactly 4 arguments, each tagged 'strong', 'weak', or "
            "'irrelevant' for the proposition in the question text."
        ),
        "shape": (
            '"minigame_hints":{"arguments":['
            '{"text":"Argument 1","verdict":"strong"},'
            '{"text":"Argument 2","verdict":"weak"},'
            '{"text":"Argument 3","verdict":"irrelevant"},'
            '{"text":"Argument 4","verdict":"weak"}]}'
        ),
    },
    "venn": {
        "description": (
            "Provide both set labels and exactly 6 items, each placed in one of: "
            "'A' (only set A), 'B' (only set B), 'both', or 'neither'."
        ),
        "shape": (
            '"minigame_hints":{"set_a":"Set A label","set_b":"Set B label",'
            '"items":[{"text":"Item 1","region":"A"},{"text":"Item 2","region":"B"},'
            '{"text":"Item 3","region":"both"},{"text":"Item 4","region":"neither"},'
            '{"text":"Item 5","region":"A"},{"text":"Item 6","region":"both"}]}'
        ),
    },
    "data-table": {
        "description": (
            "Provide the table as structured data: an array of column headers "
            "and a 2-D array of cell strings (one entry per row). Keep the "
            "options for the question itself (4 entries; one isCorrect)."
        ),
        "shape": (
            '"minigame_hints":{"title":"Table title",'
            '"headers":["Col 1","Col 2","Col 3"],'
            '"rows":[["a","1","x"],["b","2","y"],["c","3","z"]]}'
        ),
    },
    "role-identification": {
        "description": (
            "Provide the role being evaluated and exactly 4 candidate actions, "
            "each tagged in_role: true/false."
        ),
        "shape": (
            '"minigame_hints":{"role":"doctor",'
            '"actions":[{"text":"Action 1","in_role":true},'
            '{"text":"Action 2","in_role":false},'
            '{"text":"Action 3","in_role":true},'
            '{"text":"Action 4","in_role":false}]}'
        ),
    },
    "values-sorter": {
        "description": (
            "Provide exactly 6 actions, each tagged with one of the four medical "
            "ethics pillars: 'beneficence', 'nonMaleficence', 'justice', 'autonomy'."
        ),
        "shape": (
            '"minigame_hints":{"actions":['
            '{"text":"Action 1","pillar":"beneficence"},'
            '{"text":"Action 2","pillar":"nonMaleficence"},'
            '{"text":"Action 3","pillar":"justice"},'
            '{"text":"Action 4","pillar":"autonomy"},'
            '{"text":"Action 5","pillar":"beneficence"},'
            '{"text":"Action 6","pillar":"justice"}]}'
        ),
    },
}


MINIGAME_KIND_DESCRIPTIONS = {
    # VR
    "tfc": "True / False / Can't Tell — pick the verdict based on the passage",
    "main-idea": "Pick the option that best captures the passage's central message",
    "paraphrase": "Pick the option that most accurately restates information from the passage",
    "tone-purpose": "Identify the author's tone, intent, or purpose",
    "inference": "Pick the implied (vs directly stated) statement",
    # DM
    "syllogism": "Apply deductive reasoning to premises to pick the valid conclusion",
    "logic-grid": "Solve a clue-based grid (seating, ordering, matching)",
    "venn": "Categorise items by set membership / overlap",
    "probability": "Calculate or compare probabilities",
    "argument-strength": "Rate arguments as strong, weak, or irrelevant for a proposition",
    "assumption": "Identify the unstated assumption underpinning an argument",
    # QR
    "rapid-estimation": "Approximate a numerical answer under time pressure",
    "data-table": "Read a table to answer a calculation question",
    "ratio": "Solve a ratio or proportion problem",
    "fraction": "Convert, compare, or compute with fractions",
    "graph-grab": "Read a chart/graph to answer a calculation question",
    "chart-sprint": "Quickly extract a value from a bar/pie/line chart",
    # SJT
    "appropriateness": "Rate how appropriate a proposed action is",
    "importance": "Rate how important a consideration is in the scenario",
    "empathy": "Pick the most empathetic / least empathetic response",
    "escalation": "Rank options from best to worst response in the situation",
    "role-identification": "Identify which actions fall within a stated professional role",
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
