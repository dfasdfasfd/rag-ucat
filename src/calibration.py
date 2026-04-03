"""
Per-section calibration gates with param locking, prompt drift detection,
and batch mode unlock control.
"""

from src.config import CALIBRATION_THRESHOLD, SECTIONS, SECTION_GEN_PARAMS, RETRIEVAL_STRATEGIES
from src.prompts import PromptBuilder, prompt_version


class CalibrationManager:
    """
    Manages per-section calibration before batch mode.
    Each section must hit CALIBRATION_THRESHOLD consecutive approvals
    before batch generation is unlocked.
    """

    def __init__(self, db):
        self.db = db
        self._prompt_builder = PromptBuilder()

    def get_state(self, section: str) -> dict:
        """Get current calibration state for a section."""
        return self.db.get_calibration_state(section)

    def record_approval(self, section: str):
        """
        Thumbs up: increment streak.
        If threshold reached, graduate and lock params.
        Returns: (new_count, is_now_calibrated)
        """
        state = self.get_state(section)
        new_count = state["consecutive_approvals"] + 1

        if new_count >= CALIBRATION_THRESHOLD:
            # Graduate: lock current prompt + gen + retrieval params
            current_hash = self._current_prompt_hash(section)
            locked = {
                "prompt_version": current_hash,
                "gen_params": SECTION_GEN_PARAMS[section],
                "retrieval_strategy": RETRIEVAL_STRATEGIES[section],
            }
            self.db.update_calibration(
                section, new_count,
                is_calibrated=True, locked_params=locked
            )
            return new_count, True
        else:
            self.db.update_calibration(section, new_count, is_calibrated=False)
            return new_count, False

    def record_rejection(self, section: str) -> str:
        """
        Thumbs down: reset streak to 0.
        Returns penalty instruction for retry prompt.
        """
        self.db.update_calibration(section, consecutive_approvals=0)
        return (
            "The previous output was rejected during calibration. "
            "Generate a substantially different question set with higher quality."
        )

    def is_batch_unlocked(self, section: str) -> bool:
        """Check if batch mode is available for this section."""
        return self.get_state(section).get("is_calibrated", False)

    def check_prompt_drift(self, section: str) -> str:
        """
        Auto-reset calibration if prompt template changed since graduation.
        Called on every generation. If locked prompt_version differs from current,
        reset calibration and return a notification message.
        Returns None if no drift, or notification string if reset.
        """
        state = self.get_state(section)
        if not state.get("is_calibrated"):
            return None

        locked_hash = (state.get("locked_params") or {}).get("prompt_version")
        if not locked_hash:
            return None

        current_hash = self._current_prompt_hash(section)
        if locked_hash != current_hash:
            self.reset(section)
            return (
                f"{SECTIONS[section]} prompt changed — "
                f"recalibration required before batch mode."
            )
        return None

    def reset(self, section: str):
        """Re-enter calibration (e.g. after prompt changes)."""
        self.db.update_calibration(
            section, consecutive_approvals=0,
            is_calibrated=False, locked_params=None
        )

    def reset_all(self):
        """Reset calibration for all sections."""
        for section in SECTIONS:
            self.reset(section)

    def get_all_states(self) -> dict:
        """Get calibration state for all sections."""
        return {sec: self.get_state(sec) for sec in SECTIONS}

    def _current_prompt_hash(self, section: str) -> str:
        """Compute current prompt template hash for drift detection."""
        # Build a dummy prompt to get the template hash
        system, _, version_hash = self._prompt_builder.build(section, [])
        return version_hash
