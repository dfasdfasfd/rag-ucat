"""Constants, environment, and persisted settings.

Secrets are loaded from environment (typically a ``.env`` file via
``python-dotenv``). Application settings — last-used model, sliders, etc. —
persist to a JSON file alongside the DB.
"""
from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

# ─── Load .env if present ─────────────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    # Walk up from cwd to find .env (so running from any subdir works)
    here = Path.cwd()
    for parent in (here, *here.parents):
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
            break
except ImportError:
    pass  # python-dotenv is optional — env vars can be set directly

# ─── App identity ────────────────────────────────────────────────────────────

APP_TITLE   = "UCAT Trainer  ·  RAG"
DB_FILE     = os.environ.get("UCAT_DB_FILE", "ucat_rag.db")
SETTINGS_FILE  = os.environ.get("UCAT_SETTINGS_FILE", "ucat_settings.json")
TELEMETRY_FILE = os.environ.get("UCAT_TELEMETRY_FILE", "ucat_telemetry.jsonl")

# ─── Models ──────────────────────────────────────────────────────────────────

DEFAULT_LLM    = "claude-opus-4-7"
DEFAULT_VERIFY_LLM = "claude-haiku-4-5"     # cheap second-opinion judge
DEFAULT_JUDGE2_LLM = "claude-sonnet-4-6"    # tie-breaker in 3-judge mode

# Per-section primary judge override. SJT scoring is rubric-based and
# subjective; Haiku tends to vote `correct: true` indiscriminately under
# the lenient prompt and the per-question difficulty estimates are noisy.
# A Sonnet primary catches more rubric-violation false positives at ~5×
# the per-call cost, which is acceptable for SJT's lower call volume.
# Other sections stay on the cheap Haiku default — they have objective
# answers Haiku can verify reliably.
PRIMARY_JUDGE_BY_SECTION: Dict[str, str] = {
    "SJT": "claude-sonnet-4-6",
}
DEFAULT_EMBED  = "voyage-3-large"

LLM_CHOICES   = ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"]
EMBED_CHOICES = ["voyage-3-large", "voyage-3", "voyage-3-lite"]

# Cost per 1M tokens (USD) — for live cost display.
MODEL_COSTS = {
    "claude-opus-4-7":   {"in": 5.00, "out": 25.00, "cache_read": 0.50, "cache_write": 6.25},
    "claude-sonnet-4-6": {"in": 3.00, "out": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    "claude-haiku-4-5":  {"in": 1.00, "out":  5.00, "cache_read": 0.10, "cache_write": 1.25},
}

# ─── Retrieval / generation defaults ─────────────────────────────────────────

DEFAULT_TOP_K        = 4
DEFAULT_MMR_LAMBDA   = 0.55
DEFAULT_VERIFY       = True
DEFAULT_MULTI_JUDGE  = False    # 3-LLM jury (more cost, higher confidence)
DEFAULT_TARGET_DIFFICULTY = 3.0  # IRT logits — middle of the 1.0-5.0 band
DUPLICATE_THRESHOLD  = 0.93
EMBED_BATCH_SIZE     = 64

# ─── Bulk generation defaults ────────────────────────────────────────────────

BULK_MAX_QUANTITY           = 100      # hard cap on a single bulk run
BULK_COST_CONFIRM_THRESHOLD = 5.00     # USD; above this, ask before launching

# ─── UCAT domain ─────────────────────────────────────────────────────────────

SECTIONS = {
    "VR": "Verbal Reasoning",
    "DM": "Decision Making",
    "QR": "Quantitative Reasoning",
    "AR": "Abstract Reasoning",
    "SJT": "Situational Judgement",
}
SECTION_COLORS = {
    "VR": "#4A90D9", "DM": "#E8943A",
    "QR": "#3FB950", "AR": "#A78BFA",
    "SJT": "#F778BA",
}
SECTION_DESC = {
    "VR": "A passage (200-300 words) followed by exactly 4 questions. Each question is either True/False/Can't Tell OR 4-option multiple choice (A-D). Questions answerable ONLY from the passage.",
    "DM": "Exactly 5 standalone questions. Each is one of: syllogism, logical (clue-based), venn (set relationships), probability, or argument (strongest argument for/against). Each has 5 options (A-E). Venn questions MUST include a structured set spec (sets[]) for visual rendering.",
    "QR": "One data stimulus (table, bar chart, line chart, stacked-bar chart, or pie chart) followed by exactly 4 calculation questions. Each has 5 numerical options (A-E). Step-by-step working in each explanation. Stimulus MUST be provided as a structured chart spec for visual rendering.",
    "AR": "Type 1 set. Set A (6 panels with shape sets, hidden rule). Set B (6 panels with shape sets, different rule). Then 5 test shapes answered Set A / Set B / Neither. Panels MUST be provided as structured shape specs for visual rendering.",
    "SJT": "A workplace/clinical scenario followed by exactly 4 questions. Each question asks about the appropriateness or importance of a candidate action/consideration. Options are typically a Likert scale (e.g. 'Very appropriate' / 'Appropriate but not ideal' / 'Inappropriate but not awful' / 'Very inappropriate', or 'Very important' / 'Important' / 'Of minor importance' / 'Not important at all'). No single 'correct' answer — judged against UCAT marking guidance.",
}

# Question count per generated set, mirroring the min/max in the Pydantic
# section models. Used by bulk generation to convert "N questions" → "M sets".
SET_SIZES = {"VR": 4, "DM": 5, "QR": 4, "AR": 5, "SJT": 4}

# Per-section subtype catalogue. Each entry is (storage_value, human_label).
# storage_value matches the field used by the schema:
#   - DM → Question.type
#   - VR → Question.minigame_kind (with type=tf when value=='tfc', type=mc otherwise)
#   - QR → QRChart.type on the stimulus
# AR is intentionally empty: subtype dropdown is disabled in the UI.
SUBTYPES_BY_SECTION = {
    "DM": [
        ("syllogism",    "Syllogism"),
        ("logical",      "Logical (logic puzzles)"),
        ("venn",         "Venn"),
        ("probability",  "Probability"),
        ("argument",     "Argument"),
    ],
    "VR": [
        ("tfc",          "True / False / Can't Tell"),
        ("main-idea",    "Main idea"),
        ("paraphrase",   "Paraphrase match"),
        ("tone-purpose", "Tone & purpose"),
        ("inference",    "Inference"),
    ],
    "QR": [
        ("table",        "Table"),
        ("bar",          "Bar chart"),
        ("line",         "Line chart"),
        ("stacked_bar",  "Stacked bar"),
        ("pie",          "Pie"),
    ],
    "AR": [],
    "SJT": [
        ("appropriateness", "Appropriateness"),
        ("importance",      "Importance"),
    ],
}


def compute_set_count(n_input: int, section: str,
                       subtype: Optional[str]) -> int:
    """Convert the user's quantity input into a number of sets to generate.

    With ``subtype=None`` (Any/mixed mode), the input is already in sets and
    is returned as-is. With a subtype selected, the input is in *questions*
    and is rounded UP to the next full set using ``SET_SIZES[section]``.

    Returns 0 for non-positive inputs.
    """
    if n_input <= 0:
        return 0
    if not subtype:
        return n_input
    per_set = SET_SIZES.get(section, 1) or 1
    return math.ceil(n_input / per_set)


# ─── Bulk equate mode ────────────────────────────────────────────────────────

# The four sections that participate in equate mode. AR is intentionally
# excluded — it's pattern-matching, not directly comparable to the four
# reasoning sections, and a balanced UCAT mock typically pairs the reasoning
# sections together. Order is the round-robin order used by equate_task_list.
EQUATE_SECTIONS: tuple[str, ...] = ("VR", "QR", "SJT", "DM")


def equate_task_list(n: int) -> list[str]:
    """Round-robin section list for an equate run.

    n=0 → []
    n=1 → ["VR", "QR", "SJT", "DM"]
    n=3 → ["VR", "QR", "SJT", "DM", "VR", "QR", "SJT", "DM", "VR", "QR", "SJT", "DM"]

    Returns an empty list for non-positive n.
    """
    if n <= 0:
        return []
    return list(EQUATE_SECTIONS) * n


# IRT difficulty bands (Rasch logits). Predicted by Claude per question;
# refined when student response data accumulates.
IRT_BANDS = {
    1.0: "Very easy — simple recall, single-step inference, obvious distractors.",
    2.0: "Easy — early-test difficulty, two-step reasoning, weak distractors.",
    3.0: "Medium — typical UCAT difficulty, multi-step reasoning, plausible distractors.",
    4.0: "Hard — top-decile UCAT difficulty, subtle inferences, near-tie distractors.",
    5.0: "Very hard — discriminator items, edge cases, requires deep insight.",
}

def difficulty_label(logits: float) -> str:
    """Snap a continuous IRT value to its band label."""
    nearest = min(IRT_BANDS.keys(), key=lambda b: abs(b - logits))
    return IRT_BANDS[nearest]

# ─── Settings (persisted JSON) ────────────────────────────────────────────────

class Settings:
    """User preferences that survive across runs.

    Thread-safe: a single ``threading.RLock`` serialises ``get`` / ``set`` /
    ``load`` / ``save`` so concurrent reads from background workers (e.g. the
    async verify thread) and writes from the UI thread can't tear the
    underlying dict mid-update or interleave with disk persistence.

    The lock is reentrant because ``set`` calls ``save`` while still holding
    the lock — that's the only safe way to keep on-disk state and in-memory
    state consistent across concurrent writers. The disk write happens under
    the lock; readers in other threads briefly block on a contended save,
    which is preferable to the alternative (two writers' snapshots racing
    onto disk in the wrong order).
    """

    DEFAULTS: Dict[str, Any] = {
        "llm":              DEFAULT_LLM,
        "embed":            DEFAULT_EMBED,
        "top_k":            DEFAULT_TOP_K,
        "mmr_lambda":       DEFAULT_MMR_LAMBDA,
        "target_difficulty": DEFAULT_TARGET_DIFFICULTY,
        "verify":           DEFAULT_VERIFY,
        "multi_judge":      DEFAULT_MULTI_JUDGE,
        "bulk_section":     "VR",
        "bulk_quantity":    10,
        "bulk_hint":        "",
        # ── Subtype targeting (added 2026-04-26) ───────────────────────────
        "bulk_subtype":            "",   # current dropdown value ("" == Any/mixed)
        "bulk_subtype_by_section": {     # remember per-section choice
            "VR": "", "DM": "", "QR": "", "AR": "", "SJT": "",
        },
        "bulk_quantity_unit":      "sets",  # "sets" or "questions" (derived from subtype)
        # ── Equate mode + configurable confirm threshold (added 2026-04-28) ─
        "bulk_equate":                  False,   # tick to run VR/QR/SJT/DM together
        "bulk_cost_confirm_threshold":  5.00,    # USD; overrides BULK_COST_CONFIRM_THRESHOLD
    }

    def __init__(self, path: str = SETTINGS_FILE):
        self.path = path
        self._lock = threading.RLock()
        self.data: Dict[str, Any] = dict(self.DEFAULTS)
        self.load()

    def load(self):
        with self._lock:
            if os.path.exists(self.path):
                try:
                    with open(self.path, encoding="utf-8") as f:
                        saved = json.load(f)
                    for k in self.DEFAULTS:
                        if k in saved:
                            self.data[k] = saved[k]
                except Exception:
                    pass

    def save(self):
        # Snapshot under the lock so we don't serialise a half-mutated dict,
        # then release before the disk write so other readers aren't blocked
        # on I/O.
        with self._lock:
            snapshot = dict(self.data)
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2)
        except Exception:
            pass

    def get(self, k: str) -> Any:
        with self._lock:
            return self.data.get(k, self.DEFAULTS.get(k))

    def set(self, k: str, v: Any):
        # Hold the lock across both the mutation AND the save() call. RLock
        # makes this safe — save() re-enters and snapshots the same dict
        # we just mutated. Without this, two concurrent set() calls can
        # race: A mutates, releases, then B mutates+saves before A's save
        # runs, so the on-disk state ends up reflecting A's value (older
        # snapshot) while in-memory has B's. The lock keeps disk and
        # memory consistent.
        with self._lock:
            self.data[k] = v
            self.save()


# ─── API key check ────────────────────────────────────────────────────────────

def api_status() -> tuple[bool, str]:
    """Return (ok, message) describing whether all required env vars are set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False, "ANTHROPIC_API_KEY not set (add to .env)"
    if not os.environ.get("VOYAGE_API_KEY"):
        return False, "VOYAGE_API_KEY not set (add to .env)"
    return True, "Connected"


# ─── Bulk cost estimator ──────────────────────────────────────────────────────

# Per-section cost multipliers, normalised to mean 1.0 across EQUATE_SECTIONS
# so that sum across the four equate sections is exactly 4.0 (i.e. 4× the
# flat baseline). These are educated guesses based on prompt structure:
#   - VR  carries a passage (200-300 words) + 4 short questions
#   - QR  carries a chart spec (table/bar/line/etc.) + 4 numerical questions
#         — chart spec adds output tokens, hence the bump above 1.0
#   - SJT carries a workplace scenario + 4 Likert items — short, hence below 1.0
#   - DM  carries 5 standalone items — used as the baseline (1.0)
# Refine via telemetry in a follow-up spec.
SECTION_COST_MULTIPLIERS: dict[str, float] = {
    "VR":  0.95,
    "QR":  1.20,
    "SJT": 0.85,
    "DM":  1.00,
    "AR":  1.10,   # not in EQUATE_SECTIONS today, included for forward-compat
}


def estimate_bulk_cost(
    n: int,
    llm: str,
    *,
    multi_judge: bool,
    verify: bool,
) -> tuple[float, float]:
    """Estimate total USD cost for an N-set bulk run.

    Returns ``(low, high)`` where:
      • low  assumes generation input tokens hit the prompt cache (typical after
        the first iteration).
      • high assumes a cold cache for every iteration (worst case).

    Token assumptions reflect telemetry observations from single-shot runs:
      gen:    ~3000 input + ~2000 output
      verify: ~1500 input + ~600 output  (Haiku, single-judge)
      jury:   add Sonnet pass + Opus second-pass at similar token shapes
    """
    if n <= 0:
        return 0.0, 0.0

    costs = MODEL_COSTS[llm]
    gen_low  = (3000 * costs["cache_read"] + 2000 * costs["out"]) / 1_000_000
    gen_high = (3000 * costs["in"]         + 2000 * costs["out"]) / 1_000_000
    per_low, per_high = gen_low, gen_high

    if verify:
        haiku = MODEL_COSTS["claude-haiku-4-5"]
        verify_per = (1500 * haiku["in"] + 600 * haiku["out"]) / 1_000_000
        per_low  += verify_per
        per_high += verify_per

    if multi_judge:
        sonnet = MODEL_COSTS["claude-sonnet-4-6"]
        opus   = MODEL_COSTS["claude-opus-4-7"]
        jury_per = ((1500 * sonnet["in"] + 600 * sonnet["out"]) / 1_000_000
                  + (1500 * opus["in"]   + 600 * opus["out"])   / 1_000_000)
        per_low  += jury_per
        per_high += jury_per

    return n * per_low, n * per_high


def estimate_section_cost(
    section: str,
    n_sets: int,
    llm: str,
    *,
    multi_judge: bool,
    verify: bool,
) -> tuple[float, float]:
    """Per-section cost estimate. Wraps ``estimate_bulk_cost`` and applies the
    section-specific multiplier from ``SECTION_COST_MULTIPLIERS``.

    Sections outside the table fall back to multiplier 1.0.
    """
    low, high = estimate_bulk_cost(
        n_sets, llm, multi_judge=multi_judge, verify=verify)
    m = SECTION_COST_MULTIPLIERS.get(section, 1.0)
    return low * m, high * m
