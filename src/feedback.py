"""
Feedback engine: explicit + implicit signals, EMA decay for implicit scores,
composite weighting for retrieval boost, and auto-promotion.
"""

from src.config import FEEDBACK, QUALITY_THRESHOLDS


class FeedbackEngine:
    """Manages explicit and implicit feedback signals for retrieval weighting."""

    def __init__(self, db):
        self.db = db

    # ─── Explicit Signals ────────────────────────────────────────────────────

    def record_generation(self, context_doc_ids: list, output_data: dict,
                          quality_report: dict):
        """Record that these context docs were used in a generation."""
        self.db.increment_generation_count(context_doc_ids)

    def record_promotion(self, gen_id: int, context_doc_ids: list):
        """User promoted this output — reward the source docs."""
        self.db.increment_success_count(context_doc_ids)

    # ─── Implicit Signals (EMA Decay) ────────────────────────────────────────

    def update_implicit_score(self, doc_id: int, signal_type: str):
        """
        Update implicit score with EMA decay (α=0.2).
        Positive signals (long_view, exported) = 1.0
        Negative signals (regenerated) = 0.0
        Score fully reflects recent behaviour within ~5 interactions.
        """
        alpha = FEEDBACK["ema_alpha"]
        signal = 1.0 if signal_type in ("long_view", "exported") else 0.0
        old_score = self.db.get_implicit_score(doc_id)
        new_score = alpha * signal + (1 - alpha) * old_score
        self.db.set_implicit_score(doc_id, new_score)

    def record_view_duration(self, gen_id: int, duration_ms: int,
                             context_doc_ids: list):
        """Record how long user viewed a generated question set."""
        self.db.update_generated_signal(gen_id, "view_duration_ms", duration_ms)
        if duration_ms >= FEEDBACK["long_view_threshold_ms"]:
            for doc_id in context_doc_ids:
                self.update_implicit_score(doc_id, "long_view")

    def record_regenerate(self, gen_id: int, context_doc_ids: list):
        """User immediately regenerated — negative implicit signal."""
        self.db.update_generated_signal(gen_id, "was_regenerated", 1)
        for doc_id in context_doc_ids:
            self.update_implicit_score(doc_id, "regenerated")

    def record_export(self, gen_id: int, context_doc_ids: list):
        """User exported the output — strong positive signal."""
        self.db.update_generated_signal(gen_id, "was_exported", 1)
        for doc_id in context_doc_ids:
            self.update_implicit_score(doc_id, "exported")

    # ─── Composite Weight for Retrieval ──────────────────────────────────────

    def get_doc_weight(self, doc: dict) -> float:
        """
        Compute retrieval weight for a KB document.
        Range: 1.0 (neutral) to ~2.0 (highly successful).
        Used to boost good source docs in retrieval scoring.
        """
        gen_count = doc.get("generation_count", 0)
        if gen_count == 0:
            return 1.0

        # Explicit success rate
        explicit_rate = doc.get("success_count", 0) / gen_count

        # Implicit score (already EMA-decayed)
        implicit = doc.get("implicit_score", 0.0)

        # Composite weight
        weight = 1.0 + (
            FEEDBACK["explicit_weight"] * explicit_rate
            + FEEDBACK["implicit_weight"] * implicit
        )
        return min(weight, 2.0)  # Cap at 2x boost

    # ─── Auto-Promotion ──────────────────────────────────────────────────────

    def should_auto_promote(self, quality_report: dict) -> bool:
        """Check if a generated question set qualifies for auto-promotion."""
        if not quality_report.get("format_valid"):
            return False

        rule_score = quality_report.get("rule_score", 0)
        llm_score = quality_report.get("llm_score", 0)
        dedup = quality_report.get("dedup", {})

        if dedup.get("kb_duplicate"):
            return False

        return (rule_score >= QUALITY_THRESHOLDS["auto_promote_rule"]
                and llm_score >= QUALITY_THRESHOLDS["auto_promote_llm"])

    def auto_promote_if_qualified(self, gen_id: int, quality_report: dict,
                                  embed_text_fn=None) -> bool:
        """Auto-promote a generated question set if it meets quality thresholds."""
        if self.should_auto_promote(quality_report):
            self.db.promote_to_kb(gen_id, embed_text_fn=embed_text_fn)
            return True
        return False
