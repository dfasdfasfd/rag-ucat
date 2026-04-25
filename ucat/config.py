"""Constants, environment, and persisted settings.

Secrets are loaded from environment (typically a ``.env`` file via
``python-dotenv``). Application settings — last-used model, sliders, etc. —
persist to a JSON file alongside the DB.
"""
from __future__ import annotations

import json
import os
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
}
SECTION_COLORS = {
    "VR": "#4A90D9", "DM": "#E8943A",
    "QR": "#3FB950", "AR": "#A78BFA",
}
SECTION_DESC = {
    "VR": "A passage (200-300 words) followed by exactly 4 questions. Each question is either True/False/Can't Tell OR 4-option multiple choice (A-D). Questions answerable ONLY from the passage.",
    "DM": "Exactly 5 standalone questions. Each is one of: syllogism, logical (clue-based), venn (set relationships), probability, or argument (strongest argument for/against). Each has 5 options (A-E). Venn questions MUST include a structured set spec (sets[]) for visual rendering.",
    "QR": "One data stimulus (table, bar chart, or line chart) followed by exactly 4 calculation questions. Each has 5 numerical options (A-E). Step-by-step working in each explanation. Stimulus MUST be provided as a structured chart spec for visual rendering.",
    "AR": "Type 1 set. Set A (6 panels with shape sets, hidden rule). Set B (6 panels with shape sets, different rule). Then 5 test shapes answered Set A / Set B / Neither. Panels MUST be provided as structured shape specs for visual rendering.",
}

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
    """User preferences that survive across runs."""

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
    }

    def __init__(self, path: str = SETTINGS_FILE):
        self.path = path
        self.data: Dict[str, Any] = dict(self.DEFAULTS)
        self.load()

    def load(self):
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
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception:
            pass

    def get(self, k: str) -> Any:    return self.data.get(k, self.DEFAULTS.get(k))
    def set(self, k: str, v: Any):   self.data[k] = v; self.save()


# ─── API key check ────────────────────────────────────────────────────────────

def api_status() -> tuple[bool, str]:
    """Return (ok, message) describing whether all required env vars are set."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False, "ANTHROPIC_API_KEY not set (add to .env)"
    if not os.environ.get("VOYAGE_API_KEY"):
        return False, "VOYAGE_API_KEY not set (add to .env)"
    return True, "Connected"


# ─── Bulk cost estimator ──────────────────────────────────────────────────────

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
