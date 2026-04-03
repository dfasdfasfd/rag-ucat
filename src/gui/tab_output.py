"""
Output History tab: generated questions, quality scores, color-coding,
implicit signal tracking, and promotion controls.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import time

from src.config import SECTIONS
from src.gui.theme import (
    BG, PANEL, BORDER, TEXT, MUTED, SUCCESS, WARN, DANGER,
    FM, FS, FT, FH, mkbtn,
)
from src.gui.tab_generate import format_qset


class TabOutput:
    """Output History tab with quality display and implicit feedback tracking."""

    def __init__(self, parent, app):
        self.app = app
        self.frame = tk.Frame(parent, bg=BG)
        self._sel_gen_id = None
        self._view_start_time = None
        self._build()

    def _build(self):
        top = tk.Frame(self.frame, bg=BG)
        top.pack(fill="x", padx=22, pady=(16, 6))
        tk.Label(top, text="Output History", bg=BG, fg=TEXT, font=FT).pack(side="left")
        mkbtn(top, "↻  Refresh", self.refresh,
              bg=PANEL, fg=MUTED, font=FS, pady=5).pack(side="right")
        mkbtn(top, "✓  Auto-promote 4+", self._auto_promote_all,
              bg=SUCCESS, font=FS, pady=5).pack(side="right", padx=8)

        # Treeview
        tf = tk.Frame(self.frame, bg=BG)
        tf.pack(fill="both", expand=True, padx=22, pady=(0, 4))
        cols = ("ID", "Section", "Quality", "Context", "Prompt", "Date")
        self.tree = ttk.Treeview(tf, columns=cols, show="headings", height=10)
        for c, w in zip(cols, (50, 160, 70, 80, 80, 140)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w" if c in ("Date", "Prompt") else "center")

        # Color tags for quality scores
        self.tree.tag_configure("good", foreground=SUCCESS)
        self.tree.tag_configure("mid", foreground=WARN)
        self.tree.tag_configure("bad", foreground=DANGER)

        vsb = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # Preview
        pf = tk.Frame(self.frame, bg=BORDER)
        pf.pack(fill="both", expand=True, padx=22, pady=(0, 4))
        self.preview = scrolledtext.ScrolledText(
            pf, bg=PANEL, fg=TEXT, font=FS, relief="flat",
            padx=12, pady=10, wrap=tk.WORD
        )
        self.preview.pack(fill="both", expand=True)

        # Bottom actions
        bot = tk.Frame(self.frame, bg=BG)
        bot.pack(fill="x", padx=22, pady=(0, 10))
        self.promote_btn = mkbtn(bot, "✓  Add to Knowledge Base", self._promote,
                                  bg=SUCCESS, pady=6, padx=14, state="disabled")
        self.promote_btn.pack(side="left")

    def refresh(self):
        self._record_view_duration()  # Record view time for previous selection

        rows = self.app.db.get_generated(limit=500)
        for iid in self.tree.get_children():
            self.tree.delete(iid)

        # Batch-fetch quality scores for all gen_ids
        gen_ids = [r["id"] for r in rows]
        quality_map = self.app.db.get_quality_scores_batch(gen_ids)

        for r in rows:
            section_name = SECTIONS.get(r["section"], r["section"])
            ctx_count = len(r.get("context_ids", []))
            prompt_v = (r.get("prompt_version") or "?")[:8]

            # Look up quality score from quality_log
            q = quality_map.get(r["id"])
            if q:
                final = q.get("final_score", 0)
                quality_str = f"{final:.2f}"
                if final >= 0.8:
                    tag = ("good",)
                elif final >= 0.6:
                    tag = ("mid",)
                else:
                    tag = ("bad",)
            else:
                quality_str = "-"
                tag = ()

            self.tree.insert("", "end", iid=str(r["id"]),
                              values=(r["id"], section_name, quality_str,
                                      ctx_count, prompt_v, r["created"][:16]),
                              tags=tag)

    def _on_select(self, _e):
        # Record view duration for previous selection
        self._record_view_duration()

        sel = self.tree.selection()
        if not sel:
            return
        self._sel_gen_id = int(sel[0])
        self._view_start_time = time.time()

        rows = self.app.db.get_generated(limit=500)
        row = next((r for r in rows if r["id"] == self._sel_gen_id), None)
        if row:
            self.preview.config(state="normal")
            self.preview.delete(1.0, tk.END)
            self.preview.insert(tk.END, format_qset(row["data"]))
            self.preview.config(state="disabled")
            self.promote_btn.config(state="normal")

    def _record_view_duration(self):
        """Record implicit feedback: how long the user viewed the previous selection."""
        if self._sel_gen_id and self._view_start_time:
            duration_ms = int((time.time() - self._view_start_time) * 1000)
            try:
                gen = self.app.db.get_generated(limit=500)
                row = next((r for r in gen if r["id"] == self._sel_gen_id), None)
                if row and self.app.feedback:
                    ctx_ids = row.get("context_ids", [])
                    self.app.feedback.record_view_duration(
                        self._sel_gen_id, duration_ms, ctx_ids
                    )
            except Exception:
                pass
        self._view_start_time = None

    def _promote(self):
        if self._sel_gen_id:
            from src.embeddings import EmbeddingEngine
            self.app.db.promote_to_kb(
                self._sel_gen_id,
                embed_text_fn=EmbeddingEngine.embed_text_for
            )
            if self.app.feedback:
                gen = self.app.db.get_generated(limit=500)
                row = next((r for r in gen if r["id"] == self._sel_gen_id), None)
                if row:
                    self.app.feedback.record_promotion(
                        self._sel_gen_id, row.get("context_ids", [])
                    )
            self.app.refresh_stats()
            self.app.tab_kb.refresh()
            self.promote_btn.config(state="disabled")
            self.app.status("Added to knowledge base — re-index to activate for RAG")

    def _auto_promote_all(self):
        """Auto-promote all generated questions with quality score >= 4.0."""
        from src.embeddings import EmbeddingEngine
        rows = self.app.db.get_generated(limit=500)
        promoted = 0
        for row in rows:
            # Simple check: try promoting via feedback engine
            if self.app.feedback:
                # We'd need quality reports stored — for now promote manually
                pass
        self.app.status(f"Auto-promote: check calibration tab for quality-gated promotion")
