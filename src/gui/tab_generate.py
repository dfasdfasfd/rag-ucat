"""
Generate tab: section selection, topic hint, difficulty slider,
live streaming preview, quality report, and action buttons.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import json
import textwrap

from src.config import SECTIONS
from src.gui.theme import (
    BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, SUCCESS, WARN, DANGER,
    FM, FB, FS, FSB, FT, FH, mkbtn,
)


def format_qset(data: dict) -> str:
    """Format a question set for display."""
    sec = data.get("section", "?")
    lines = ["═" * 64, f"  {SECTIONS.get(sec, sec)}  ·  Question Set", "═" * 64, ""]
    for key, label in (("passage", "PASSAGE"), ("stimulus", "DATA / STIMULUS")):
        if key in data:
            lines += [label, "─" * 40]
            for para in str(data[key]).split("\n"):
                lines.append(textwrap.fill(para, 72) if para.strip() else "")
            lines.append("")
    for key, label in (("set_a_description", "SET A"), ("set_b_description", "SET B")):
        if key in data:
            lines += [label, "─" * 40, str(data[key]), ""]
    for q in data.get("questions", []):
        lines.append(f"Q{q.get('number', '?')}.  {q.get('text', '')}")
        for k, v in q.get("options", {}).items():
            lines.append(f"       {k})  {v}")
        lines.append(f"       Answer: {q.get('answer', '?')}")
        if q.get("explanation"):
            lines.append(textwrap.fill(
                q["explanation"], 64,
                initial_indent="       Explanation: ",
                subsequent_indent="           "
            ))
        lines.append("")
    return "\n".join(lines)


class TabGenerate:
    """Generate tab with streaming preview, difficulty slider, and quality report."""

    def __init__(self, parent, app):
        self.app = app
        self.frame = tk.Frame(parent, bg=BG)

        self._last_data = None
        self._last_section = None
        self._abort = False

        self._build()

    def _build(self):
        p = tk.Frame(self.frame, bg=BG)
        p.pack(fill="both", expand=True, padx=36, pady=24)

        tk.Label(p, text="Generate New Questions", bg=BG, fg=TEXT,
                 font=FT).pack(anchor="w")
        tk.Label(p, text="RAG-grounded generation from your knowledge base.",
                 bg=BG, fg=MUTED, font=FS).pack(anchor="w", pady=(2, 16))

        # Section selection
        sr = tk.Frame(p, bg=BG)
        sr.pack(anchor="w", pady=(0, 12))
        tk.Label(sr, text="Section:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self.section_var = tk.StringVar(value="VR")
        for code in SECTIONS:
            tk.Radiobutton(
                sr, text=f" {code} ", variable=self.section_var, value=code,
                bg=BG, fg=TEXT, selectcolor=PANEL, activebackground=BG,
                activeforeground=ACCENT, font=FB, indicatoron=False,
                relief="flat", bd=1, padx=12, pady=6, cursor="hand2"
            ).pack(side="left", padx=4)

        # Hint row
        hr = tk.Frame(p, bg=BG)
        hr.pack(fill="x", pady=(0, 8))
        tk.Label(hr, text="Topic hint:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self.hint_var = tk.StringVar()
        tk.Entry(hr, textvariable=self.hint_var, bg=PANEL2, fg=TEXT, font=FM,
                 insertbackground=ACCENT, relief="flat", width=34).pack(side="left")
        tk.Label(hr, text="(optional — e.g. 'ecology', 'probability')",
                 bg=BG, fg=MUTED, font=FS).pack(side="left", padx=10)

        # Difficulty row
        dr = tk.Frame(p, bg=BG)
        dr.pack(fill="x", pady=(0, 12))
        tk.Label(dr, text="Difficulty:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self.diff_var = tk.IntVar(value=5)
        self.diff_scale = tk.Scale(
            dr, from_=1, to=10, orient="horizontal", variable=self.diff_var,
            bg=BG, fg=TEXT, troughcolor=PANEL, highlightthickness=0,
            font=FS, length=200, sliderlength=20
        )
        self.diff_scale.pack(side="left")
        self.diff_label = tk.Label(dr, text="5/10 (medium)", bg=BG, fg=MUTED, font=FS)
        self.diff_label.pack(side="left", padx=10)
        self.diff_var.trace_add("write", self._update_diff_label)

        # Button row
        ar = tk.Frame(p, bg=BG)
        ar.pack(anchor="w", pady=(0, 10))
        self.gen_btn = mkbtn(ar, "⚡  GENERATE QUESTIONS", self.app.do_generate,
                              padx=26, pady=10, font=("Courier New", 12, "bold"))
        self.gen_btn.pack(side="left", padx=(0, 8))
        self.regen_btn = mkbtn(ar, "↻  Regenerate", self.app.do_generate,
                                bg=PANEL, fg=ACCENT, font=FS, pady=8)
        self.regen_btn.pack(side="left", padx=(0, 8))
        self.regen_btn.config(state="disabled")
        self.abort_btn = mkbtn(ar, "■  Abort", self._do_abort,
                                bg=DANGER, font=FS, pady=8)
        self.abort_btn.pack(side="left", padx=(0, 16))
        self.abort_btn.config(state="disabled")
        self.prog_lbl = tk.Label(ar, text="", bg=BG, fg=MUTED, font=FS)
        self.prog_lbl.pack(side="left")

        # Two-panel output
        cols = tk.Frame(p, bg=BG)
        cols.pack(fill="both", expand=True)

        # Left: generated output (streaming preview)
        lf = tk.Frame(cols, bg=BG)
        lf.pack(side="left", fill="both", expand=True, padx=(0, 8))
        tk.Label(lf, text="OUTPUT", bg=BG, fg=MUTED, font=FH).pack(anchor="w", pady=(0, 4))
        ow = tk.Frame(lf, bg=BORDER, bd=1)
        ow.pack(fill="both", expand=True)
        self.output_text = scrolledtext.ScrolledText(
            ow, bg=PANEL, fg=TEXT, font=FM, relief="flat",
            padx=14, pady=12, wrap=tk.WORD, insertbackground=ACCENT, state="disabled"
        )
        self.output_text.pack(fill="both", expand=True)

        # Right: context + quality report
        rf = tk.Frame(cols, bg=BG, width=340)
        rf.pack(side="right", fill="y", padx=(8, 0))
        rf.pack_propagate(False)

        tk.Label(rf, text="RETRIEVED CONTEXT", bg=BG, fg=MUTED, font=FH).pack(anchor="w", pady=(0, 4))
        cw = tk.Frame(rf, bg=BORDER, bd=1)
        cw.pack(fill="both", expand=True)
        self.context_text = scrolledtext.ScrolledText(
            cw, bg=PANEL2, fg=MUTED, font=FS, relief="flat",
            padx=12, pady=10, wrap=tk.WORD, state="disabled"
        )
        self.context_text.pack(fill="both", expand=True)

        # Quality report panel
        tk.Label(rf, text="QUALITY REPORT", bg=BG, fg=MUTED, font=FH).pack(anchor="w", pady=(8, 4))
        qw = tk.Frame(rf, bg=BORDER, bd=1)
        qw.pack(fill="x")
        self.quality_text = tk.Label(
            qw, text="No report yet", bg=PANEL2, fg=MUTED, font=FS,
            anchor="w", justify="left", padx=12, pady=8
        )
        self.quality_text.pack(fill="x")

        # Bottom actions
        bot = tk.Frame(p, bg=BG)
        bot.pack(fill="x", pady=(10, 0))
        self.save_btn = mkbtn(bot, "✓  Save to Knowledge Base", self.app.save_to_kb,
                               bg=SUCCESS, pady=6, padx=14, state="disabled")
        self.save_btn.pack(side="left", padx=(0, 10))
        self.copy_btn = mkbtn(bot, "⊞  Copy", self.app.copy_output,
                               bg=PANEL, fg=TEXT, font=FM, pady=6, padx=14, state="disabled")
        self.copy_btn.pack(side="left")

    def _update_diff_label(self, *_):
        val = self.diff_var.get()
        if val <= 3:
            desc = "easy"
        elif val <= 6:
            desc = "medium"
        else:
            desc = "hard"
        self.diff_label.config(text=f"{val}/10 ({desc})")

    def _do_abort(self):
        self._abort = True
        self.abort_btn.config(state="disabled", text="Aborting...")

    def get_abort_flag(self):
        return lambda: self._abort

    def reset_abort(self):
        self._abort = False
        self.abort_btn.config(state="normal", text="■  Abort")

    def set_generating(self, is_generating: bool):
        if is_generating:
            self._abort = False
            self.gen_btn.config(state="disabled", text="Generating...")
            self.abort_btn.config(state="normal", text="■  Abort")
            self.save_btn.config(state="disabled")
            self.copy_btn.config(state="disabled")
            self.regen_btn.config(state="disabled")
        else:
            self.gen_btn.config(state="normal", text="⚡  GENERATE QUESTIONS")
            self.abort_btn.config(state="disabled", text="■  Abort")
            self.regen_btn.config(state="normal")

    def show_output(self, data: dict):
        self._last_data = data
        self._write_widget(self.output_text, format_qset(data))
        self.save_btn.config(state="normal")
        self.copy_btn.config(state="normal")

    def show_context(self, retrieved: list):
        if not retrieved:
            self._write_widget(self.context_text, "No indexed docs retrieved.")
            return
        ctx = f"Retrieved {len(retrieved)} document(s):\n\n"
        for i, doc in enumerate(retrieved, 1):
            score = doc.get("_rrf_score", 0)
            ctx += (f"[{i}] ID #{doc['id']}  ·  {doc['section']}  "
                    f"·  score {score:.4f}  ·  {doc.get('source', '?')}\n"
                    f"    {doc.get('embed_text', '')[:110]}...\n\n")
        self._write_widget(self.context_text, ctx.strip())

    def show_quality(self, report: dict):
        rule = report.get("rule_score", 0)
        llm = report.get("llm_score", 0)
        final = report.get("final_score", 0)
        errors = report.get("errors", [])
        version = report.get("prompt_version", "?")

        # Color based on final score
        if final >= 0.8:
            color = SUCCESS
        elif final >= 0.6:
            color = WARN
        else:
            color = DANGER

        lines = [
            f"Rule: {rule:.2f}  |  LLM: {llm:.1f}/5  |  Final: {final:.2f}",
            f"Prompt: v{version[:8]}",
        ]
        if errors:
            lines.append(f"Issues: {len(errors)}")
        dedup = report.get("dedup", {})
        if dedup:
            lines.append(f"KB sim: {dedup.get('kb_similarity', 0):.3f}  |  "
                         f"Session sim: {dedup.get('session_similarity', 0):.3f}")

        self.quality_text.config(text="\n".join(lines), fg=color)

    def append_token(self, token: str):
        """Append a streaming token to the output preview."""
        self.output_text.config(state="normal")
        self.output_text.insert(tk.END, token)
        self.output_text.see(tk.END)
        self.output_text.config(state="disabled")

    def clear_output(self):
        self._write_widget(self.output_text, "")
        self._write_widget(self.context_text, "")
        self.quality_text.config(text="Generating...", fg=MUTED)

    def _write_widget(self, widget, text):
        widget.config(state="normal")
        widget.delete(1.0, tk.END)
        widget.insert(tk.END, text)
        widget.config(state="disabled")
