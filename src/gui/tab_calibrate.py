"""
Calibration tab: per-section Yes/No approval gate, progress indicator,
calibration graduation, and reset controls.
"""

import tkinter as tk
from tkinter import scrolledtext
import threading

from src.config import SECTIONS, CALIBRATION_THRESHOLD
from src.gui.theme import (
    BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, SUCCESS, WARN, DANGER,
    FM, FB, FS, FSB, FT, FH, mkbtn,
)
from src.gui.tab_generate import format_qset


class TabCalibrate:
    """Calibration mode: binary Yes/No gate per section before batch mode."""

    def __init__(self, parent, app):
        self.app = app
        self.frame = tk.Frame(parent, bg=BG)
        self._current_data = None
        self._current_section = None
        self._build()

    def _build(self):
        p = tk.Frame(self.frame, bg=BG)
        p.pack(fill="both", expand=True, padx=36, pady=24)

        tk.Label(p, text="Quality Calibration", bg=BG, fg=TEXT,
                 font=FT).pack(anchor="w")
        tk.Label(p, text=(
            f"Each section must pass {CALIBRATION_THRESHOLD} consecutive approvals "
            "to unlock batch mode."),
            bg=BG, fg=MUTED, font=FS).pack(anchor="w", pady=(2, 16))

        # Section selector
        sr = tk.Frame(p, bg=BG)
        sr.pack(anchor="w", pady=(0, 12))
        tk.Label(sr, text="Section:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self.section_var = tk.StringVar(value="VR")
        for code in SECTIONS:
            tk.Radiobutton(
                sr, text=f" {code} ", variable=self.section_var, value=code,
                bg=BG, fg=TEXT, selectcolor=PANEL, activebackground=BG,
                activeforeground=ACCENT, font=FB, indicatoron=False,
                relief="flat", bd=1, padx=12, pady=6, cursor="hand2",
                command=self._update_status
            ).pack(side="left", padx=4)

        # Status
        self.status_lbl = tk.Label(p, text="", bg=BG, fg=MUTED, font=FM)
        self.status_lbl.pack(anchor="w", pady=(0, 8))

        # Progress bar (visual)
        prog_frame = tk.Frame(p, bg=BG)
        prog_frame.pack(fill="x", pady=(0, 12))
        self.prog_dots = []
        for i in range(CALIBRATION_THRESHOLD):
            dot = tk.Label(prog_frame, text="○", bg=BG, fg=MUTED,
                           font=("Courier New", 16))
            dot.pack(side="left", padx=4)
            self.prog_dots.append(dot)

        # Generate + approve buttons
        btn_row = tk.Frame(p, bg=BG)
        btn_row.pack(anchor="w", pady=(0, 10))
        self.gen_btn = mkbtn(btn_row, "⚡  Generate for Calibration",
                              self._do_generate, padx=20, pady=10)
        self.gen_btn.pack(side="left", padx=(0, 20))

        self.yes_btn = mkbtn(btn_row, "  ✓  YES  ", self._approve,
                              bg=SUCCESS, padx=30, pady=14,
                              font=("Courier New", 14, "bold"))
        self.yes_btn.pack(side="left", padx=(0, 10))
        self.yes_btn.config(state="disabled")

        self.no_btn = mkbtn(btn_row, "  ✕  NO  ", self._reject,
                             bg=DANGER, padx=30, pady=14,
                             font=("Courier New", 14, "bold"))
        self.no_btn.pack(side="left")
        self.no_btn.config(state="disabled")

        mkbtn(btn_row, "↺ Reset", self._reset,
              bg=PANEL, fg=MUTED, font=FS, pady=6).pack(side="left", padx=(20, 0))

        # Preview
        pf = tk.Frame(p, bg=BORDER)
        pf.pack(fill="both", expand=True)
        self.preview = scrolledtext.ScrolledText(
            pf, bg=PANEL, fg=TEXT, font=FS, relief="flat",
            padx=12, pady=10, wrap=tk.WORD, state="disabled"
        )
        self.preview.pack(fill="both", expand=True)

        self._update_status()

    def _update_status(self):
        section = self.section_var.get()
        state = self.app.calibration.get_state(section)
        approvals = state.get("consecutive_approvals", 0)
        is_cal = state.get("is_calibrated", False)

        if is_cal:
            self.status_lbl.config(
                text=f"✓ {SECTIONS[section]} CALIBRATED — Batch mode unlocked!",
                fg=SUCCESS
            )
        else:
            self.status_lbl.config(
                text=f"{approvals}/{CALIBRATION_THRESHOLD} consecutive approvals for {SECTIONS[section]}",
                fg=WARN if approvals > 0 else MUTED
            )

        # Update progress dots
        for i, dot in enumerate(self.prog_dots):
            if i < approvals:
                dot.config(text="●", fg=SUCCESS)
            else:
                dot.config(text="○", fg=MUTED)

    def _do_generate(self):
        section = self.section_var.get()
        self.gen_btn.config(state="disabled", text="Generating...")
        self.yes_btn.config(state="disabled")
        self.no_btn.config(state="disabled")

        def worker():
            try:
                data, retrieved, report = self.app.generator.generate(
                    section, difficulty=5
                )
                self.frame.after(0, lambda: self._show_result(data, report))
            except Exception as e:
                self.frame.after(0, lambda err=str(e): self._show_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _show_result(self, data, report):
        self._current_data = data
        self._current_section = data.get("section", self.section_var.get())

        self.preview.config(state="normal")
        self.preview.delete(1.0, tk.END)
        self.preview.insert(tk.END, format_qset(data))

        # Show quality info
        rule = report.get("rule_score", 0)
        llm = report.get("llm_score", 0)
        self.preview.insert(tk.END, f"\n\n{'─'*40}\n")
        self.preview.insert(tk.END, f"Quality: Rule={rule:.2f}  LLM={llm:.1f}/5\n")
        self.preview.config(state="disabled")

        self.gen_btn.config(state="normal", text="⚡  Generate for Calibration")
        self.yes_btn.config(state="normal")
        self.no_btn.config(state="normal")

    def _show_error(self, err):
        self.preview.config(state="normal")
        self.preview.delete(1.0, tk.END)
        self.preview.insert(tk.END, f"Error: {err}")
        self.preview.config(state="disabled")
        self.gen_btn.config(state="normal", text="⚡  Generate for Calibration")

    def _approve(self):
        section = self.section_var.get()
        new_count, is_calibrated = self.app.calibration.record_approval(section)
        self.yes_btn.config(state="disabled")
        self.no_btn.config(state="disabled")

        # Re-embed approved output as soft KB reference
        if self._current_data:
            from src.embeddings import EmbeddingEngine
            embed_text = EmbeddingEngine.embed_text_for(self._current_data, section)
            data_type = EmbeddingEngine.infer_data_type(self._current_data, section)
            self.app.db.add_doc(
                section, self._current_data, embed_text,
                source="calibration", data_type=data_type
            )
            self.app.refresh_stats()
            self.app.tab_kb.refresh()

        if is_calibrated:
            self.app.status(f"✓ {SECTIONS[section]} calibrated! Batch mode unlocked.")

        self._update_status()
        self.app.sidebar.update_calibration(self.app.calibration)

    def _reject(self):
        section = self.section_var.get()
        penalty = self.app.calibration.record_rejection(section)
        self.yes_btn.config(state="disabled")
        self.no_btn.config(state="disabled")
        self._update_status()
        self.app.sidebar.update_calibration(self.app.calibration)
        self.app.status(f"Rejected — streak reset. {penalty}")

    def _reset(self):
        section = self.section_var.get()
        self.app.calibration.reset(section)
        self._update_status()
        self.app.sidebar.update_calibration(self.app.calibration)
        self.app.status(f"Calibration reset for {SECTIONS[section]}")
