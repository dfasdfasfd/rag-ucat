"""
Sidebar: model selection, KB stats, coverage summary, calibration status.
"""

import tkinter as tk
from tkinter import ttk

from src.config import SECTIONS, SECTION_COLORS, SECTION_GEN_PARAMS, DEFAULT_LLM, DEFAULT_EMBED
from src.gui.theme import (
    PANEL, BG, TEXT, MUTED, ACCENT, SUCCESS, DANGER, WARN, BORDER,
    FM, FB, FS, FSB, FH, mkbtn,
)


class Sidebar:
    """Left sidebar with model selection, stats, and calibration indicators."""

    def __init__(self, parent, app):
        self.app = app
        self.frame = tk.Frame(parent, bg=PANEL, width=230)
        self.frame.pack(side="left", fill="y")
        self.frame.pack_propagate(False)

        self._slabels = {}
        self._cal_labels = {}

        self._build()

    def _build(self):
        sb = self.frame

        # KB section counts
        tk.Label(sb, text="KNOWLEDGE BASE", bg=PANEL, fg=MUTED,
                 font=FSB).pack(anchor="w", padx=16, pady=(20, 6))

        for code in SECTIONS:
            row = tk.Frame(sb, bg=PANEL)
            row.pack(fill="x", padx=10, pady=3)
            tk.Label(row, text="■", bg=PANEL, fg=SECTION_COLORS[code],
                     font=("Courier New", 12)).pack(side="left", padx=(4, 8))
            tk.Label(row, text=code, bg=PANEL, fg=TEXT, font=FB).pack(side="left")

            # Calibration indicator
            cal_lbl = tk.Label(row, text="○", bg=PANEL, fg=MUTED, font=FS)
            cal_lbl.pack(side="right", padx=2)
            self._cal_labels[code] = cal_lbl

            lbl = tk.Label(row, text="0/0", bg=PANEL, fg=MUTED, font=FS)
            lbl.pack(side="right", padx=8)
            self._slabels[code] = lbl

        tk.Label(sb, text="indexed/total  ○=uncal ●=cal", bg=PANEL, fg=MUTED,
                 font=("Courier New", 8)).pack(anchor="e", padx=14, pady=(0, 4))

        ttk.Separator(sb, orient="horizontal").pack(fill="x", padx=12, pady=12)

        # LLM Model selection
        tk.Label(sb, text="LLM MODEL", bg=PANEL, fg=MUTED,
                 font=FSB).pack(anchor="w", padx=16, pady=(0, 4))
        self.llm_var = tk.StringVar(value=DEFAULT_LLM)
        self.llm_cb = ttk.Combobox(sb, textvariable=self.llm_var,
                                    width=19, state="readonly")
        self.llm_cb.pack(padx=14, pady=(0, 8))

        # Embed Model selection
        tk.Label(sb, text="EMBED MODEL", bg=PANEL, fg=MUTED,
                 font=FSB).pack(anchor="w", padx=16, pady=(0, 4))
        self.emb_var = tk.StringVar(value=DEFAULT_EMBED)
        self.emb_cb = ttk.Combobox(sb, textvariable=self.emb_var,
                                    width=19, state="readonly")
        self.emb_cb.pack(padx=14, pady=(0, 6))

        mkbtn(sb, "↻  Refresh Models", self.app.refresh_models,
              bg=PANEL, fg=MUTED, font=FS, pady=4
              ).pack(padx=14, pady=(0, 12), fill="x")

        # Per-section model overrides
        ttk.Separator(sb, orient="horizontal").pack(fill="x", padx=12, pady=4)
        tk.Label(sb, text="SECTION MODELS", bg=PANEL, fg=MUTED,
                 font=FSB).pack(anchor="w", padx=16, pady=(4, 2))
        tk.Label(sb, text="(override per section)", bg=PANEL, fg=MUTED,
                 font=("Courier New", 7)).pack(anchor="w", padx=16, pady=(0, 4))

        self.section_model_vars = {}
        for code in SECTIONS:
            row = tk.Frame(sb, bg=PANEL)
            row.pack(fill="x", padx=10, pady=1)
            tk.Label(row, text=f"{code}:", bg=PANEL, fg=SECTION_COLORS[code],
                     font=FSB, width=3).pack(side="left", padx=(4, 4))
            default_model = SECTION_GEN_PARAMS[code]["preferred_llm"]
            var = tk.StringVar(value=default_model)
            cb = ttk.Combobox(row, textvariable=var, width=14, state="readonly", font=("Courier New", 8))
            cb.pack(side="left", padx=(0, 4))
            self.section_model_vars[code] = (var, cb)

        ttk.Separator(sb, orient="horizontal").pack(fill="x", padx=12, pady=4)

        # Index button
        self.idx_btn = mkbtn(sb, "⊛  Index Knowledge Base",
                              self.app.do_index, bg="#1F6FEB", pady=8)
        self.idx_btn.pack(padx=14, pady=(10, 2), fill="x")
        self.idx_lbl = tk.Label(sb, text="", bg=PANEL, fg=MUTED, font=FS)
        self.idx_lbl.pack(padx=14, anchor="w", pady=(0, 10))

        ttk.Separator(sb, orient="horizontal").pack(fill="x", padx=12, pady=4)

        # Prompt A/B tracking
        mkbtn(sb, "📊  Prompt A/B", self.app.show_prompt_ab,
              bg=PANEL, fg=WARN, font=FS, pady=5).pack(padx=14, pady=(6, 4), fill="x")

        # Current prompt version label
        self.prompt_v_lbl = tk.Label(sb, text="Prompt: -", bg=PANEL, fg=MUTED,
                                      font=("Courier New", 7))
        self.prompt_v_lbl.pack(anchor="w", padx=16, pady=(0, 6))

        ttk.Separator(sb, orient="horizontal").pack(fill="x", padx=12, pady=4)

        # Import / Samples / Export
        mkbtn(sb, "⊕  Import JSON", self.app.do_import,
              bg="#2D7D46", pady=7).pack(padx=14, pady=3, fill="x")
        mkbtn(sb, "📷  Import Screenshot", self.app.do_screenshot_import,
              bg="#2D7D46", pady=7).pack(padx=14, pady=3, fill="x")
        mkbtn(sb, "⊞  Add Samples", self.app.add_samples,
              bg=PANEL, fg=ACCENT, font=FS, pady=6).pack(padx=14, pady=2, fill="x")
        mkbtn(sb, "↓  Export Output", self.app.do_export,
              bg=PANEL, fg=MUTED, font=FS, pady=6).pack(padx=14, pady=2, fill="x")

    def update_stats(self, db):
        for code in SECTIONS:
            idx = db.count(code, indexed_only=True)
            tot = db.count(code, indexed_only=False)
            self._slabels[code].config(text=f"{idx}/{tot}")

    def update_calibration(self, calibration_mgr):
        for code in SECTIONS:
            state = calibration_mgr.get_state(code)
            if state.get("is_calibrated"):
                self._cal_labels[code].config(text="●", fg=SUCCESS)
            elif state.get("consecutive_approvals", 0) > 0:
                self._cal_labels[code].config(text="◐", fg=WARN)
            else:
                self._cal_labels[code].config(text="○", fg=MUTED)

    def update_prompt_version(self, version: str):
        """Update the displayed prompt version hash."""
        self.prompt_v_lbl.config(text=f"Prompt: v{version[:8]}")

    def get_section_model(self, section: str) -> str:
        """Get the user-selected model for a specific section."""
        if section in self.section_model_vars:
            return self.section_model_vars[section][0].get()
        return self.llm_var.get()

    def update_models(self, models):
        if not models:
            return
        self.llm_cb["values"] = models
        self.emb_cb["values"] = models
        if self.llm_var.get() not in models:
            self.llm_var.set(models[0])
        pref = next((m for m in models if "embed" in m.lower()), models[0])
        if self.emb_var.get() not in models:
            self.emb_var.set(pref)
        # Update per-section model dropdowns
        for code, (var, cb) in self.section_model_vars.items():
            cb["values"] = models
            if var.get() not in models:
                # Keep the configured default if available, else first model
                default = SECTION_GEN_PARAMS[code]["preferred_llm"]
                match = next((m for m in models if default in m), models[0])
                var.set(match)
