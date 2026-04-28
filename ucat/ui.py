"""Tkinter UI — generation tab, knowledge base browser, output history,
and insights dashboard for coverage / bias / difficulty.
"""
from __future__ import annotations

import io
import json
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from .coverage import pick_diversification
from .config import (APP_TITLE, LLM_CHOICES, EMBED_CHOICES, SECTIONS, SECTION_COLORS,
                      DEFAULT_LLM, DEFAULT_EMBED, IRT_BANDS, api_status, Settings,
                      BULK_MAX_QUANTITY, BULK_COST_CONFIRM_THRESHOLD, estimate_bulk_cost,
                      SUBTYPES_BY_SECTION, SET_SIZES, compute_set_count)
from .db import Database
from .format import format_qset
from .rag import RAGEngine
from .rendering import render_visuals_for
from .samples import SAMPLES
from .telemetry import emit, aggregate, logger

# ─── Theme ────────────────────────────────────────────────────────────────────

BG = "#0C1117"; PANEL = "#161B22"; PANEL2 = "#1C2128"; BORDER = "#30363D"
ACCENT = "#58A6FF"; TEXT = "#E6EDF3"; MUTED = "#7D8590"
SUCCESS = "#3FB950"; DANGER = "#F85149"; WARN = "#D29922"

FM = ("Courier New", 11); FB = ("Courier New", 11, "bold")
FS = ("Courier New", 9);  FSB = ("Courier New", 9, "bold")
FT = ("Courier New", 15, "bold"); FH = ("Courier New", 10, "bold")


def mkbtn(parent, text, cmd, fg="white", bg=ACCENT, font=None, **kw):
    return tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                     font=font or FB, relief="flat", cursor="hand2",
                     activebackground=bg, activeforeground=fg, **kw)


# ─── Scrollable frame helper ──────────────────────────────────────────────────

class ScrollFrame(tk.Frame):
    """A vertically-scrollable Frame for the visuals panel."""

    def __init__(self, parent, bg=PANEL, **kw):
        super().__init__(parent, bg=bg, **kw)
        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self._sb = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._sb.set)
        self._sb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self._canvas, bg=bg)
        self._win = self._canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner_resize)
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self._canvas.bind_all("<Button-4>", self._on_btn4, add="+")
        self._canvas.bind_all("<Button-5>", self._on_btn5, add="+")

    def _on_inner_resize(self, _e):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_resize(self, e):
        self._canvas.itemconfigure(self._win, width=e.width)

    def _on_mousewheel(self, e):
        # Only scroll if pointer is over us.
        if not str(e.widget).startswith(str(self._canvas)):
            return
        delta = -1 if e.delta > 0 else 1
        self._canvas.yview_scroll(delta, "units")

    def _on_btn4(self, e):
        if not str(e.widget).startswith(str(self._canvas)):
            return
        self._canvas.yview_scroll(-1, "units")

    def _on_btn5(self, e):
        if not str(e.widget).startswith(str(self._canvas)):
            return
        self._canvas.yview_scroll(1, "units")

    def clear(self):
        for w in self.inner.winfo_children():
            w.destroy()


# ─── Application ─────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1480x960")
        self.minsize(1180, 760)
        self.configure(bg=BG)

        self.settings = Settings()
        self.db       = Database()
        self.rag      = RAGEngine(self.db, self.settings)

        self._last_data: Optional[Dict[str, Any]] = None
        self._last_section: Optional[str] = None
        self._last_retrieved: List = []
        self._sel_gen_id: Optional[int] = None
        self._session_cost = 0.0
        self._session_tokens = {"in": 0, "out": 0, "cache_r": 0, "cache_w": 0}
        # Keep refs to PhotoImages so Tk doesn't GC them.
        self._photo_refs: List[Any] = []

        # Bulk-run state.
        self._bulk_stop: threading.Event = threading.Event()
        self._bulk_thread: Optional[threading.Thread] = None
        self._bulk_rows: List[Dict[str, Any]] = []
        self._bulk_started_at: Optional[float] = None
        self._bulk_run_cost: float = 0.0  # accumulated USD for the active run

        self._style()
        self._ui()
        self.after(700, self._chk_api)

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=TEXT, font=FM)
        s.configure("TNotebook", background=BG, borderwidth=0)
        s.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, font=FM, padding=[18, 9])
        s.map("TNotebook.Tab", background=[("selected", BG)], foreground=[("selected", ACCENT)])
        s.configure("Treeview", background=PANEL, foreground=TEXT,
                    fieldbackground=PANEL, font=FS, rowheight=28, borderwidth=0)
        s.configure("Treeview.Heading", background=BG, foreground=MUTED, font=FH)
        s.map("Treeview", background=[("selected", "#1F6FEB")], foreground=[("selected", "white")])
        s.configure("TScrollbar", background=PANEL, troughcolor=BG, borderwidth=0)
        s.configure("TCombobox", font=FM)
        s.configure("TSeparator", background=BORDER)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _ui(self):
        # Header.
        hdr = tk.Frame(self, bg=PANEL, height=54)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="◈  UCAT TRAINER", bg=PANEL, fg=ACCENT,
                 font=("Courier New", 15, "bold")).pack(side="left", padx=22, pady=12)
        tk.Label(hdr, text="RAG", bg=PANEL, fg=WARN, font=FSB).pack(side="left", pady=12)

        self._cost_lbl = tk.Label(hdr, text="$0.00 · 0 tok", bg=PANEL, fg=MUTED, font=FS)
        self._cost_lbl.pack(side="right", padx=(0, 22), pady=12)
        self._dot  = tk.Label(hdr, text="●", bg=PANEL, fg=DANGER, font=("Courier New", 13))
        self._dlbl = tk.Label(hdr, text="Checking API…", bg=PANEL, fg=MUTED, font=FS)
        self._dot.pack(side="right", padx=(0, 18))
        self._dlbl.pack(side="right", padx=(0, 4))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)
        self._sidebar(body)

        self._nb = ttk.Notebook(body)
        self._nb.pack(side="left", fill="both", expand=True)
        self._t_gen      = tk.Frame(self._nb, bg=BG)
        self._t_bulk     = tk.Frame(self._nb, bg=BG)
        self._t_kb       = tk.Frame(self._nb, bg=BG)
        self._t_out      = tk.Frame(self._nb, bg=BG)
        self._t_insights = tk.Frame(self._nb, bg=BG)
        self._nb.add(self._t_gen,      text="  ⚡  GENERATE  ")
        self._nb.add(self._t_bulk,     text="  ⚡⚡ BULK  ")
        self._nb.add(self._t_kb,       text="  🗄  KNOWLEDGE BASE  ")
        self._nb.add(self._t_out,      text="  📋  HISTORY  ")
        self._nb.add(self._t_insights, text="  📊  INSIGHTS  ")
        self._tab_gen()
        self._tab_bulk()
        self._tab_kb()
        self._tab_out()
        self._tab_insights()

        self._sbar = tk.Label(self, text="Ready", bg="#090C10", fg=MUTED, font=FS, anchor="w")
        self._sbar.pack(fill="x", padx=12, pady=(2, 4))
        self._refresh_stats()

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _sidebar(self, parent):
        container = tk.Frame(parent, bg=PANEL, width=250)
        container.pack(side="left", fill="y"); container.pack_propagate(False)
        sf = ScrollFrame(container, bg=PANEL)
        sf.pack(fill="both", expand=True)
        sb = sf.inner

        tk.Label(sb, text="KNOWLEDGE BASE", bg=PANEL, fg=MUTED,
                 font=FSB).pack(anchor="w", padx=16, pady=(20, 6))
        self._slabels: Dict[str, tk.Label] = {}
        for code in SECTIONS:
            row = tk.Frame(sb, bg=PANEL)
            row.pack(fill="x", padx=10, pady=3)
            tk.Label(row, text="■", bg=PANEL, fg=SECTION_COLORS[code],
                     font=("Courier New", 12)).pack(side="left", padx=(4, 8))
            tk.Label(row, text=code, bg=PANEL, fg=TEXT, font=FB).pack(side="left")
            lbl = tk.Label(row, text="0/0", bg=PANEL, fg=MUTED, font=FS)
            lbl.pack(side="right", padx=8)
            self._slabels[code] = lbl
        tk.Label(sb, text="indexed / total", bg=PANEL, fg=MUTED,
                 font=("Courier New", 8)).pack(anchor="e", padx=14, pady=(0, 4))

        ttk.Separator(sb, orient="horizontal").pack(fill="x", padx=12, pady=12)

        tk.Label(sb, text="LLM MODEL", bg=PANEL, fg=MUTED, font=FSB).pack(anchor="w", padx=16, pady=(0, 4))
        self._llm_var = tk.StringVar(value=self.settings.get("llm"))
        ttk.Combobox(sb, textvariable=self._llm_var, width=22, state="readonly",
                      values=LLM_CHOICES).pack(padx=14, pady=(0, 8))
        self._llm_var.trace_add("write", lambda *_: self.settings.set("llm", self._llm_var.get()))

        tk.Label(sb, text="EMBED MODEL", bg=PANEL, fg=MUTED, font=FSB).pack(anchor="w", padx=16, pady=(0, 4))
        self._emb_var = tk.StringVar(value=self.settings.get("embed"))
        ttk.Combobox(sb, textvariable=self._emb_var, width=22, state="readonly",
                      values=EMBED_CHOICES).pack(padx=14, pady=(0, 6))
        self._emb_var.trace_add("write", lambda *_: self.settings.set("embed", self._emb_var.get()))

        ttk.Separator(sb, orient="horizontal").pack(fill="x", padx=12, pady=10)

        # Top-K.
        tk.Label(sb, text="CONTEXT DOCS  (TOP-K)", bg=PANEL, fg=MUTED, font=FSB).pack(anchor="w", padx=16)
        kf = tk.Frame(sb, bg=PANEL); kf.pack(fill="x", padx=14, pady=(2, 4))
        self._topk_var = tk.IntVar(value=self.settings.get("top_k"))
        ttk.Scale(kf, from_=1, to=8, variable=self._topk_var, orient="horizontal",
                   command=lambda _v: (self.settings.set("top_k", int(self._topk_var.get())),
                                        self._topk_lbl.config(text=str(self._topk_var.get())))
                   ).pack(side="left", fill="x", expand=True)
        self._topk_lbl = tk.Label(kf, text=str(self._topk_var.get()), bg=PANEL, fg=TEXT, font=FB, width=2)
        self._topk_lbl.pack(side="right", padx=(6, 0))

        # MMR λ.
        tk.Label(sb, text="DIVERSITY  (MMR λ)", bg=PANEL, fg=MUTED, font=FSB).pack(anchor="w", padx=16, pady=(4, 0))
        mf = tk.Frame(sb, bg=PANEL); mf.pack(fill="x", padx=14, pady=(2, 4))
        self._mmr_var = tk.DoubleVar(value=self.settings.get("mmr_lambda"))
        ttk.Scale(mf, from_=0.0, to=1.0, variable=self._mmr_var, orient="horizontal",
                   command=lambda _v: (self.settings.set("mmr_lambda", round(self._mmr_var.get(), 2)),
                                        self._mmr_lbl.config(text=f"{self._mmr_var.get():.2f}"))
                   ).pack(side="left", fill="x", expand=True)
        self._mmr_lbl = tk.Label(mf, text=f"{self._mmr_var.get():.2f}", bg=PANEL, fg=TEXT, font=FB, width=4)
        self._mmr_lbl.pack(side="right", padx=(6, 0))
        tk.Label(sb, text="0=diverse  · 1=relevant", bg=PANEL, fg=MUTED,
                  font=("Courier New", 8)).pack(anchor="e", padx=16)

        # Difficulty (IRT logits).
        tk.Label(sb, text="TARGET DIFFICULTY  (IRT logits)", bg=PANEL, fg=MUTED, font=FSB).pack(anchor="w", padx=16, pady=(8, 0))
        df = tk.Frame(sb, bg=PANEL); df.pack(fill="x", padx=14, pady=(2, 4))
        self._diff_var = tk.DoubleVar(value=self.settings.get("target_difficulty"))
        ttk.Scale(df, from_=1.0, to=5.0, variable=self._diff_var, orient="horizontal",
                   command=lambda _v: (self.settings.set("target_difficulty", round(self._diff_var.get(), 1)),
                                        self._diff_lbl.config(text=f"{self._diff_var.get():.1f}"))
                   ).pack(side="left", fill="x", expand=True)
        self._diff_lbl = tk.Label(df, text=f"{self._diff_var.get():.1f}", bg=PANEL, fg=TEXT, font=FB, width=4)
        self._diff_lbl.pack(side="right", padx=(6, 0))
        tk.Label(sb, text="1=easy · 3=medium · 5=very hard", bg=PANEL, fg=MUTED,
                  font=("Courier New", 8)).pack(anchor="e", padx=16, pady=(0, 4))

        # Verify checkboxes.
        self._verify_var = tk.BooleanVar(value=self.settings.get("verify"))
        tk.Checkbutton(sb, text="✓ Self-verify answers",
                        variable=self._verify_var, bg=PANEL, fg=TEXT, font=FS,
                        selectcolor=PANEL, activebackground=PANEL, activeforeground=ACCENT,
                        command=lambda: self.settings.set("verify", self._verify_var.get())
                        ).pack(anchor="w", padx=14, pady=(4, 0))
        self._jury_var = tk.BooleanVar(value=self.settings.get("multi_judge"))
        tk.Checkbutton(sb, text="🏛  3-judge jury",
                        variable=self._jury_var, bg=PANEL, fg=TEXT, font=FS,
                        selectcolor=PANEL, activebackground=PANEL, activeforeground=ACCENT,
                        command=lambda: self.settings.set("multi_judge", self._jury_var.get())
                        ).pack(anchor="w", padx=14, pady=(0, 4))

        ttk.Separator(sb, orient="horizontal").pack(fill="x", padx=12, pady=8)

        self._idx_btn  = mkbtn(sb, "⊛  Index Knowledge Base", self._do_index,
                                bg="#1F6FEB", pady=8)
        self._idx_btn.pack(padx=14, pady=(4, 2), fill="x")
        self._idx_lbl = tk.Label(sb, text="", bg=PANEL, fg=MUTED, font=FS)
        self._idx_lbl.pack(padx=14, anchor="w", pady=(0, 8))

        ttk.Separator(sb, orient="horizontal").pack(fill="x", padx=12, pady=4)

        mkbtn(sb, "📥  Import from Crawler", self._import_crawler,
              bg="#7E5BEF", pady=7).pack(padx=14, pady=3, fill="x")
        mkbtn(sb, "⊕  Import JSON",   self._import,      bg="#2D7D46", pady=7
              ).pack(padx=14, pady=3, fill="x")
        mkbtn(sb, "⊞  Add Samples",   self._add_samples, bg=PANEL, fg=ACCENT,
              font=FS, pady=6).pack(padx=14, pady=2, fill="x")
        mkbtn(sb, "↓  Export Output", self._export,      bg=PANEL, fg=MUTED,
              font=FS, pady=6).pack(padx=14, pady=2, fill="x")

    # ── Generate tab ──────────────────────────────────────────────────────────

    def _tab_gen(self):
        p = tk.Frame(self._t_gen, bg=BG)
        p.pack(fill="both", expand=True, padx=24, pady=20)

        tk.Label(p, text="Generate New Questions", bg=BG, fg=TEXT, font=FT).pack(anchor="w")
        tk.Label(p,
                 text="MMR-diverse retrieval → cached generation → multi-judge verification → "
                      "auto-calibrated difficulty → bias / coverage analysis. Visuals render "
                      "automatically for QR charts, AR shapes, and DM venns.",
                 bg=BG, fg=MUTED, font=FS, wraplength=1100, justify="left"
                 ).pack(anchor="w", pady=(2, 14))

        sr = tk.Frame(p, bg=BG); sr.pack(anchor="w", pady=(0, 10))
        tk.Label(sr, text="Section:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self._gsec = tk.StringVar(value="VR")
        for code in SECTIONS:
            tk.Radiobutton(sr, text=f" {code} ", variable=self._gsec, value=code,
                           bg=BG, fg=TEXT, selectcolor=PANEL, activebackground=BG,
                           activeforeground=ACCENT, font=FB, indicatoron=False,
                           relief="flat", bd=1, padx=12, pady=6, cursor="hand2"
                           ).pack(side="left", padx=4)

        hr = tk.Frame(p, bg=BG); hr.pack(fill="x", pady=(0, 10))
        tk.Label(hr, text="Topic hint:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self._hint = tk.StringVar()
        tk.Entry(hr, textvariable=self._hint, bg=PANEL2, fg=TEXT, font=FM,
                 insertbackground=ACCENT, relief="flat", width=46).pack(side="left")
        tk.Label(hr, text="(steers retrieval — e.g. 'percentage change', 'syllogisms', 'ecology')",
                 bg=BG, fg=MUTED, font=FS).pack(side="left", padx=10)

        ar = tk.Frame(p, bg=BG); ar.pack(anchor="w", pady=(0, 10))
        self._gbtn  = mkbtn(ar, "⚡  GENERATE QUESTIONS", self._do_gen,
                             padx=26, pady=10, font=("Courier New", 12, "bold"))
        self._gbtn.pack(side="left", padx=(0, 12))
        self._regenbtn = mkbtn(ar, "↻  Regenerate (variation)", self._do_regen,
                                bg=PANEL, fg=ACCENT, font=FS, padx=14, pady=10, state="disabled")
        self._regenbtn.pack(side="left", padx=(0, 14))
        self._gprog = tk.Label(ar, text="", bg=BG, fg=MUTED, font=FS)
        self._gprog.pack(side="left")

        # Three-column body: text output | visuals | retrieved context
        cols = tk.Frame(p, bg=BG); cols.pack(fill="both", expand=True)

        # Text output (left).
        lf = tk.Frame(cols, bg=BG); lf.pack(side="left", fill="both", expand=True, padx=(0, 6))
        oh = tk.Frame(lf, bg=BG); oh.pack(fill="x")
        tk.Label(oh, text="OUTPUT", bg=BG, fg=MUTED, font=FH).pack(side="left", pady=(0, 4))
        self._verdict_lbl = tk.Label(oh, text="", bg=BG, fg=SUCCESS, font=FS)
        self._verdict_lbl.pack(side="right", pady=(0, 4))
        ow = tk.Frame(lf, bg=BORDER, bd=1); ow.pack(fill="both", expand=True)
        self._gout = scrolledtext.ScrolledText(
            ow, bg=PANEL, fg=TEXT, font=FM, relief="flat",
            padx=14, pady=12, wrap=tk.WORD, insertbackground=ACCENT, state="disabled"
        )
        self._gout.pack(fill="both", expand=True)

        # Visuals (middle).
        mf = tk.Frame(cols, bg=BG, width=440); mf.pack(side="left", fill="both", padx=(6, 6))
        mf.pack_propagate(False)
        vh = tk.Frame(mf, bg=BG); vh.pack(fill="x")
        tk.Label(vh, text="VISUALS", bg=BG, fg=MUTED, font=FH).pack(side="left", pady=(0, 4))
        self._vis_status = tk.Label(vh, text="", bg=BG, fg=MUTED, font=FS)
        self._vis_status.pack(side="right", pady=(0, 4))
        vw = tk.Frame(mf, bg=BORDER, bd=1); vw.pack(fill="both", expand=True)
        self._visuals = ScrollFrame(vw, bg=PANEL2)
        self._visuals.pack(fill="both", expand=True)

        # Retrieved context (right).
        rf = tk.Frame(cols, bg=BG, width=320); rf.pack(side="right", fill="y", padx=(6, 0))
        rf.pack_propagate(False)
        tk.Label(rf, text="RETRIEVED CONTEXT", bg=BG, fg=MUTED, font=FH).pack(anchor="w", pady=(0, 4))
        cw = tk.Frame(rf, bg=BORDER, bd=1); cw.pack(fill="both", expand=True)
        self._cout = scrolledtext.ScrolledText(
            cw, bg=PANEL2, fg=MUTED, font=FS, relief="flat",
            padx=12, pady=10, wrap=tk.WORD, state="disabled"
        )
        self._cout.pack(fill="both", expand=True)

        # Bottom action row.
        bot = tk.Frame(p, bg=BG); bot.pack(fill="x", pady=(10, 0))
        self._savebtn = mkbtn(bot, "✓  Save to Knowledge Base", self._save_kb,
                               bg=SUCCESS, pady=6, padx=14, state="disabled")
        self._savebtn.pack(side="left", padx=(0, 10))
        self._copybtn = mkbtn(bot, "⊞  Copy", self._copy,
                               bg=PANEL, fg=TEXT, font=FM, pady=6, padx=14, state="disabled")
        self._copybtn.pack(side="left")
        self._dup_lbl = tk.Label(bot, text="", bg=BG, fg=WARN, font=FS)
        self._dup_lbl.pack(side="right")

    # ── Bulk tab ──────────────────────────────────────────────────────────────

    def _tab_bulk(self):
        p = tk.Frame(self._t_bulk, bg=BG)
        p.pack(fill="both", expand=True, padx=24, pady=20)

        tk.Label(p, text="Bulk Generate", bg=BG, fg=TEXT, font=FT).pack(anchor="w")
        tk.Label(p,
                 text="Generate multiple question sets in sequence. Results land "
                      "in History — promote good ones to the KB from there.",
                 bg=BG, fg=MUTED, font=FS, wraplength=1100, justify="left"
                 ).pack(anchor="w", pady=(2, 14))

        # Equate mode checkbox — inserted above the Section radios so it
        # visually frames the entire single/equate decision.
        er = tk.Frame(p, bg=BG); er.pack(anchor="w", pady=(0, 6))
        self._bulk_equate = tk.BooleanVar(value=bool(self.settings.get("bulk_equate")))
        tk.Checkbutton(
            er,
            text=" Equate across VR/QR/SJT/DM (same qty for each section)",
            variable=self._bulk_equate, bg=BG, fg=TEXT, selectcolor=PANEL,
            activebackground=BG, activeforeground=ACCENT, font=FM,
            command=self._bulk_equate_changed,
        ).pack(side="left")

        # Section radios. Stored so _bulk_equate_changed can toggle their state.
        sr = tk.Frame(p, bg=BG); sr.pack(anchor="w", pady=(0, 10))
        tk.Label(sr, text="Section:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self._bulk_sec = tk.StringVar(value=self.settings.get("bulk_section"))
        self._bulk_section_radios: list[tk.Radiobutton] = []
        for code in SECTIONS:
            rb = tk.Radiobutton(sr, text=f" {code} ", variable=self._bulk_sec, value=code,
                           bg=BG, fg=TEXT, selectcolor=PANEL, activebackground=BG,
                           activeforeground=ACCENT, font=FB, indicatoron=False,
                           relief="flat", bd=1, padx=12, pady=6, cursor="hand2",
                           command=self._bulk_section_changed
                           )
            rb.pack(side="left", padx=4)
            self._bulk_section_radios.append(rb)

        # Subtype dropdown — populated based on the selected section.
        # Stored on self so equate mode can hide/show the whole row.
        self._bulk_subtype_frame = tk.Frame(p, bg=BG)
        self._bulk_subtype_frame.pack(anchor="w", pady=(0, 10))
        sb = self._bulk_subtype_frame
        tk.Label(sb, text="Subtype:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self._bulk_subtype = tk.StringVar(value="")
        self._bulk_subtype_cb = ttk.Combobox(
            sb, textvariable=self._bulk_subtype,
            state="readonly", width=32, font=FM,
        )
        self._bulk_subtype_cb.pack(side="left")
        self._bulk_subtype_cb.bind(
            "<<ComboboxSelected>>", lambda _e: self._bulk_inputs_changed()
        )
        # Initial population uses the current section.
        self._bulk_refresh_subtype_choices()

        # Quantity + topic hint row.
        qr = tk.Frame(p, bg=BG); qr.pack(fill="x", pady=(0, 10))
        self._bulk_qty_lbl = tk.Label(qr, text="Sets:", bg=BG, fg=MUTED, font=FM)
        self._bulk_qty_lbl.pack(side="left", padx=(0, 14))
        self._bulk_qty = tk.StringVar(value=str(self.settings.get("bulk_quantity")))
        tk.Entry(qr, textvariable=self._bulk_qty, bg=PANEL2, fg=TEXT, font=FM,
                 insertbackground=ACCENT, relief="flat", width=6).pack(side="left")
        self._bulk_qty.trace_add("write", lambda *_: self._bulk_inputs_changed())

        tk.Label(qr, text="   Topic hint:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(20, 14))
        self._bulk_hint = tk.StringVar(value=self.settings.get("bulk_hint"))
        tk.Entry(qr, textvariable=self._bulk_hint, bg=PANEL2, fg=TEXT, font=FM,
                 insertbackground=ACCENT, relief="flat", width=46).pack(side="left", fill="x", expand=True)
        self._bulk_hint.trace_add("write", lambda *_: self._bulk_inputs_changed())

        # Yield helper line: "→ 2 sets × 5 questions = 10 questions". Hidden
        # when no subtype is selected.
        self._bulk_yield_lbl = tk.Label(
            p, text="", bg=BG, fg=MUTED, font=FS, anchor="w"
        )
        self._bulk_yield_lbl.pack(anchor="w", pady=(0, 4))

        # Cost preview banner.
        self._bulk_cost_lbl = tk.Label(
            p, text="", bg=BG, fg=ACCENT, font=FB, anchor="w"
        )
        self._bulk_cost_lbl.pack(anchor="w", pady=(2, 12))

        # Action row.
        ar = tk.Frame(p, bg=BG); ar.pack(anchor="w", pady=(0, 10))
        self._bulk_start_btn = mkbtn(
            ar, "⚡  START BULK RUN", self._bulk_start,
            padx=22, pady=10, font=("Courier New", 12, "bold")
        )
        self._bulk_start_btn.pack(side="left", padx=(0, 12))
        self._bulk_stop_btn = mkbtn(
            ar, "⏹  STOP", self._bulk_stop_clicked,
            bg=DANGER, padx=18, pady=10, state="disabled"
        )
        self._bulk_stop_btn.pack(side="left", padx=(0, 14))
        self._bulk_progress_lbl = tk.Label(ar, text="", bg=BG, fg=MUTED, font=FS)
        self._bulk_progress_lbl.pack(side="left")

        # Treeview of per-set rows.
        tf = tk.Frame(p, bg=BG); tf.pack(fill="both", expand=True, pady=(8, 0))
        cols = ("#", "Started", "Section", "Subtype", "Status", "Verdict", "Cost", "Difficulty")
        self._bulk_tree = ttk.Treeview(tf, columns=cols, show="headings", height=10)
        for c, w in zip(cols, (44, 90, 70, 110, 180, 100, 70, 80)):
            self._bulk_tree.heading(c, text=c)
            self._bulk_tree.column(c, width=w, anchor="w" if c == "Status" else "center")
        vsb = ttk.Scrollbar(tf, orient="vertical", command=self._bulk_tree.yview)
        self._bulk_tree.configure(yscrollcommand=vsb.set)
        self._bulk_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._bulk_tree.bind("<<TreeviewSelect>>", self._bulk_row_selected)

        # Preview pane.
        pf = tk.Frame(p, bg=BORDER); pf.pack(fill="both", expand=True, pady=(8, 0))
        self._bulk_preview = scrolledtext.ScrolledText(
            pf, bg=PANEL, fg=TEXT, font=FS, relief="flat",
            padx=12, pady=10, wrap=tk.WORD, height=8
        )
        self._bulk_preview.pack(fill="both", expand=True)
        self._bulk_preview.config(state="disabled")

        # If the user previously left equate ticked, apply the disabled/hidden
        # state now (the BooleanVar's initial value is set above; this call
        # propagates it through to the radios and subtype frame).
        if self._bulk_equate.get():
            self._bulk_equate_changed()

        # Initialise the cost banner.
        self._bulk_inputs_changed()

    def _bulk_equate_changed(self):
        """Called when the equate checkbox is toggled. Disables the Section
        radios + hides the Subtype row when on; restores them when off.
        Persists the setting and triggers a cost-banner refresh."""
        equate = self._bulk_equate.get()
        self.settings.set("bulk_equate", equate)

        # Section radios.
        radio_state = "disabled" if equate else "normal"
        for rb in self._bulk_section_radios:
            rb.config(state=radio_state)

        # Subtype row — entirely hidden when equate is on.
        if equate:
            self._bulk_subtype_frame.pack_forget()
        else:
            # Repack at its original position (between section radios and
            # quantity row). pack() without `before=` re-appends to the end,
            # which is fine here because nothing has been packed since.
            # If the layout changes in future, switch to `before=<next-frame>`.
            self._bulk_subtype_frame.pack(anchor="w", pady=(0, 10))
            # When restoring, also force the choices to refresh in case the
            # current section's catalogue has changed.
            self._bulk_refresh_subtype_choices()

        self._bulk_inputs_changed()

    def _bulk_section_changed(self):
        """Called when the Section radio changes. Refreshes subtype choices
        and runs the standard inputs-changed update."""
        self._bulk_refresh_subtype_choices()
        self._bulk_inputs_changed()

    def _bulk_refresh_subtype_choices(self):
        """Rebuild the Subtype combobox values for the current section,
        restoring the user's last choice for that section."""
        section = self._bulk_sec.get()
        entries = SUBTYPES_BY_SECTION.get(section, [])

        # Build the displayed list: "Any (mixed)" always first, then human labels.
        labels = ["Any (mixed)"] + [lbl for _v, lbl in entries]
        self._bulk_subtype_cb.config(values=labels)

        # Disable for sections with no subtypes (AR).
        if not entries:
            self._bulk_subtype_cb.config(state="disabled")
            self._bulk_subtype.set("Any (mixed)")
            return
        self._bulk_subtype_cb.config(state="readonly")

        # Restore last-used subtype for this section.
        by_section = self.settings.get("bulk_subtype_by_section") or {}
        stored_value = by_section.get(section, "")
        if stored_value:
            stored_label = next(
                (lbl for v, lbl in entries if v == stored_value),
                "Any (mixed)",
            )
            self._bulk_subtype.set(stored_label)
        else:
            self._bulk_subtype.set("Any (mixed)")

    def _bulk_inputs_changed(self):
        """Called whenever section / subtype / quantity / hint changes.
        Validates input, flips labels, computes the cost preview, persists
        settings, and gates the Start button."""
        section = self._bulk_sec.get()
        hint    = self._bulk_hint.get()

        # Resolve subtype: combobox stores the human label; we map back to the
        # internal storage value (e.g. 'venn'). 'Any (mixed)' → "" → None.
        label_to_value = {
            lbl: v for v, lbl in SUBTYPES_BY_SECTION.get(section, [])
        }
        chosen_label = self._bulk_subtype.get()
        subtype_value = label_to_value.get(chosen_label, "")
        subtype = subtype_value or None

        # Persist settings.
        self.settings.set("bulk_section", section)
        self.settings.set("bulk_hint",    hint)
        self.settings.set("bulk_subtype", subtype_value)
        by_section = dict(self.settings.get("bulk_subtype_by_section") or {})
        by_section[section] = subtype_value
        self.settings.set("bulk_subtype_by_section", by_section)
        self.settings.set("bulk_quantity_unit",
                            "questions" if subtype else "sets")

        # Flip the Quantity label.
        self._bulk_qty_lbl.config(text=("Questions:" if subtype else "Sets:"))

        # Parse quantity.
        raw = self._bulk_qty.get().strip()
        n_input: Optional[int] = None
        if raw.isdigit():
            n_input = int(raw)
            if n_input >= 1:
                self.settings.set("bulk_quantity", n_input)

        # Bail if quantity is invalid.
        if n_input is None or n_input < 1:
            self._bulk_yield_lbl.config(text="")
            max_input = (BULK_MAX_QUANTITY * SET_SIZES.get(section, 5)) if subtype \
                          else BULK_MAX_QUANTITY
            self._bulk_cost_lbl.config(
                text=f"Enter a number 1 - {max_input}.",
                fg=WARN,
            )
            self._bulk_start_btn.config(state="disabled")
            return

        # Compute set count and yield helper line.
        n_sets = compute_set_count(n_input, section, subtype)
        capped = False
        if n_sets > BULK_MAX_QUANTITY:
            n_sets = BULK_MAX_QUANTITY
            capped = True

        # Helper line — only when subtype is set.
        if subtype:
            per_set = SET_SIZES[section]
            yielded = n_sets * per_set
            extra = yielded - n_input
            extra_note = f"  ({extra} extra)" if extra > 0 else ""
            cap_note = "  (capped — split into multiple runs for more)" if capped else ""
            self._bulk_yield_lbl.config(
                text=f"→ {n_sets} sets × {per_set} questions = {yielded} questions{extra_note}{cap_note}",
                fg=MUTED,
            )
        else:
            self._bulk_yield_lbl.config(text="")

        # Cost preview (always in sets — that's what's billed).
        llm     = self.settings.get("llm")
        verify  = bool(self.settings.get("verify"))
        jury    = bool(self.settings.get("multi_judge"))
        low, high = estimate_bulk_cost(n_sets, llm, multi_judge=jury, verify=verify)

        suffix = "  (capped at the max — split into multiple runs for more)" if capped else ""
        self._bulk_cost_lbl.config(
            text=f"Estimated cost: ~${low:.2f} - ${high:.2f}   "
                 f"({n_sets} sets × {llm}{suffix})",
            fg=ACCENT,
        )

        # Don't override "running" state — we re-enable in _bulk_run_finished.
        if self._bulk_thread is None or not self._bulk_thread.is_alive():
            self._bulk_start_btn.config(state="normal")

    def _bulk_start(self):
        ok, msg = api_status()
        if not ok:
            messagebox.showerror("API Not Ready",
                f"{msg}\n\nCopy .env.example → .env and fill in your keys.")
            return

        if self._bulk_thread is not None and self._bulk_thread.is_alive():
            self._status("A bulk run is already in progress.")
            return
        if str(self._gbtn.cget("state")) == "disabled":
            self._status("A run is already in progress — wait or stop it first.")
            return

        section = self._bulk_sec.get()

        # Resolve subtype from the dropdown's human label back to its storage value.
        label_to_value = {
            lbl: v for v, lbl in SUBTYPES_BY_SECTION.get(section, [])
        }
        subtype_value = label_to_value.get(self._bulk_subtype.get(), "")
        subtype = subtype_value or None

        raw = self._bulk_qty.get().strip()
        if not raw.isdigit():
            return
        n_input = int(raw)
        if n_input < 1:
            return

        # Convert questions → sets when subtype is set; cap at the max set count.
        n = compute_set_count(n_input, section, subtype)
        n = min(n, BULK_MAX_QUANTITY)
        if n < 1:
            return

        if self.db.count(section, indexed_only=True) == 0:
            if not messagebox.askyesno("No Indexed Documents",
                f"No indexed documents for {SECTIONS[section]}.\n\n"
                "Add docs and click Index, or generate without RAG context?"):
                return

        hint = self._bulk_hint.get()

        # Cost preview + threshold check.
        llm    = self.settings.get("llm")
        verify = bool(self.settings.get("verify"))
        jury   = bool(self.settings.get("multi_judge"))
        low, high = estimate_bulk_cost(n, llm, multi_judge=jury, verify=verify)

        if high > BULK_COST_CONFIRM_THRESHOLD:
            ok = messagebox.askyesno(
                "Confirm bulk run",
                f"Estimated cost: ${low:.2f} - ${high:.2f}\n"
                f"Sets: {n}\n"
                f"Section: {SECTIONS[section]}\n"
                f"Model: {llm}\n\n"
                f"Continue?",
            )
            if not ok:
                return

        # Build the task list. Single-section mode = [section] * n; equate
        # mode (added in a later task) will branch here to call
        # equate_task_list(n) instead.
        task_list = [section] * n

        self._bulk_stop.clear()
        self._bulk_thread = threading.Thread(
            target=self._bulk_worker, args=(task_list, hint, subtype), daemon=True
        )
        self._bulk_thread.start()

    def _bulk_stop_clicked(self):
        if self._bulk_thread is None or not self._bulk_thread.is_alive():
            return
        self._bulk_stop.set()
        self._bulk_stop_btn.config(state="disabled")
        self._bulk_progress_lbl.config(text="Stopping after current set…")
        self._status("Stopping bulk run…")

    def _bulk_row_selected(self, _e):
        sel = self._bulk_tree.selection()
        if not sel:
            return
        iid = sel[0]
        if not iid.startswith("bulk-"):
            return
        try:
            idx = int(iid.split("-", 1)[1])
        except ValueError:
            return
        if idx < 1 or idx > len(self._bulk_rows):
            return

        row = self._bulk_rows[idx - 1]
        self._bulk_preview.config(state="normal")
        self._bulk_preview.delete(1.0, tk.END)

        if row["status"] == "done" and row["result"]:
            self._bulk_preview.insert(tk.END, format_qset(row["result"]["data"]))
        elif row["status"] == "failed":
            err = row["error"] or "(no error message captured)"
            self._bulk_preview.insert(tk.END,
                f"Set {idx} failed after retry.\n\n{err}")
        elif row["status"] == "skipped":
            self._bulk_preview.insert(tk.END,
                f"Set {idx} was skipped (Stop pressed before it started).")
        elif row["status"] == "running":
            self._bulk_preview.insert(tk.END,
                f"Set {idx} is still being generated…")
        else:  # queued
            self._bulk_preview.insert(tk.END, f"Set {idx} is queued.")

        self._bulk_preview.config(state="disabled")

    # ── Bulk worker helpers ───────────────────────────────────────────────────

    def _bulk_row_iid(self, idx: int) -> str:
        return f"bulk-{idx}"

    def _bulk_seed_rows(self, task_list: list[str]):
        """Initialise _bulk_rows + Treeview with one queued entry per task.
        Each task is the section code for that row (single-section mode passes
        [section] * n; equate mode passes a round-robin list)."""
        subtype_value = self.settings.get("bulk_subtype") or ""
        # Subtype label is per-section; computed once for single-section runs
        # (all rows share the section). For equate runs subtype is forced None,
        # so the per-row label is always "—".
        self._bulk_rows = [
            {"idx": i, "status": "queued", "result": None,
              "error": None, "started": "",
              "section": section,
              "subtype": subtype_value or None}
            for i, section in enumerate(task_list, start=1)
        ]
        for iid in self._bulk_tree.get_children():
            self._bulk_tree.delete(iid)
        for r in self._bulk_rows:
            section = r["section"]
            subtype_label = next(
                (lbl for v, lbl in SUBTYPES_BY_SECTION.get(section, [])
                 if v == subtype_value),
                "—",
            ) if subtype_value else "—"
            # Note: Section column isn't added to the Treeview yet — that's
            # Task 5. For now the row values match the existing 7-column shape.
            self._bulk_tree.insert(
                "", "end", iid=self._bulk_row_iid(r["idx"]),
                values=(r["idx"], "", section, subtype_label,
                          "queued", "—", "—", "—"),
            )

    def _bulk_set_row(self, idx: int, *,
                       status: Optional[str] = None,
                       result: Optional[Dict[str, Any]] = None,
                       error:  Optional[str] = None,
                       started: Optional[str] = None,
                       progress: Optional[str] = None):
        """Update an in-memory row + the Treeview cells. Main-thread only."""
        if idx < 1 or idx > len(self._bulk_rows):
            return
        row = self._bulk_rows[idx - 1]
        if started is not None: row["started"] = started
        if result  is not None: row["result"]  = result
        if error   is not None: row["error"]   = error
        if status  is not None: row["status"]  = status

        # Compute display cells.
        if status == "running" and progress:
            st_cell = f"⟳ {progress[:60]}"
        elif row["status"] == "running":
            st_cell = "⟳ running"
        elif row["status"] == "done":
            st_cell = "✓ done"
        elif row["status"] == "failed":
            st_cell = f"✗ {(row['error'] or '')[:60]}"
        elif row["status"] == "skipped":
            st_cell = "· skipped"
        else:  # queued
            st_cell = "queued"

        # Subtype cell — derived from the row's stored subtype + section.
        # Use the row's own section so equate-mode rows resolve their subtype
        # against the correct section's catalogue.
        subtype_value = row.get("subtype")
        row_section   = row.get("section") or self._bulk_sec.get()
        subtype_cell = next(
            (lbl for v, lbl in SUBTYPES_BY_SECTION.get(row_section, [])
             if v == subtype_value),
            "—",
        ) if subtype_value else "—"

        verdict_cell = "—"
        cost_cell    = "—"
        diff_cell    = "—"
        if row["result"]:
            v = row["result"].get("verdict") or {}
            sym_disagreed = len((v.get("symbolic_qr") or {}).get("disagreed") or [])
            drift = row["result"].get("subtype_drift")
            if drift:
                # Drift takes precedence over the normal verdict badge.
                verdict_cell = "⚠ drift"
            elif not v:
                verdict_cell = "—"
            elif v.get("pending"):
                # Async verify still in flight; symbolic_qr may already disagree.
                verdict_cell = f"⟳ ⚠{sym_disagreed}" if sym_disagreed else "⟳"
            elif v.get("overall_correct", True):
                verdict_cell = "✓" if sym_disagreed == 0 else f"⚠ {sym_disagreed}"
            else:
                fq = len(v.get("flagged_questions") or [])
                verdict_cell = f"⚠ {fq + sym_disagreed}"

            u = row["result"].get("usage") or {}
            if u.get("cost_usd") is not None:
                cost_cell = f"${u['cost_usd']:.3f}"

            cal = row["result"].get("difficulty") or {}
            sd  = cal.get("set_difficulty")
            if isinstance(sd, (int, float)):
                diff_cell = f"{sd:.1f}"

        self._bulk_tree.item(
            self._bulk_row_iid(idx),
            values=(idx, row["started"],
                      row.get("section") or "—",
                      subtype_cell, st_cell,
                      verdict_cell, cost_cell, diff_cell),
        )

    def _bulk_run_started(self, task_list: list[str]):
        n = len(task_list)
        self._bulk_started_at = time.perf_counter()
        self._bulk_run_cost   = 0.0
        self._bulk_start_btn.config(state="disabled", text="Generating…")
        self._bulk_stop_btn.config(state="normal")
        self._bulk_progress_lbl.config(text=f"0 / {n}")
        self._bulk_seed_rows(task_list)
        llm    = self.settings.get("llm")
        verify = bool(self.settings.get("verify"))
        jury   = bool(self.settings.get("multi_judge"))
        _, est_high = estimate_bulk_cost(n, llm, multi_judge=jury, verify=verify)
        # Section field reflects the *first* section in the task list — single-
        # section runs share one section across all rows; equate runs use the
        # task_list field on the bulk_run_end event instead.
        emit("bulk_run_start",
             section=task_list[0] if task_list else self._bulk_sec.get(),
             n=n,
             model=llm,
             verify=verify,
             multi_judge=jury,
             estimated_cost_high=round(est_high, 4),
             subtype=(self.settings.get("bulk_subtype") or None))

    def _bulk_run_finished(self, succeeded: int, failed: int, stopped: bool = False):
        n = len(self._bulk_rows)
        skipped = sum(1 for r in self._bulk_rows if r["status"] == "skipped")
        drift_count = sum(
            1 for r in self._bulk_rows
            if (r.get("result") or {}).get("subtype_drift")
        )
        elapsed = (time.perf_counter() - (self._bulk_started_at or time.perf_counter()))
        emit("bulk_run_end",
             section=self._bulk_sec.get(),
             n=n,
             succeeded=succeeded,
             failed=failed,
             stopped=stopped,
             actual_cost_usd=round(self._bulk_run_cost, 4),
             duration_s=round(elapsed, 1),
             subtype=(self.settings.get("bulk_subtype") or None),
             drift_count=drift_count)
        self._bulk_thread = None
        self._bulk_started_at = None
        self._bulk_start_btn.config(state="normal", text="⚡  START BULK RUN")
        self._bulk_stop_btn.config(state="disabled")
        if stopped:
            tail = f"Stopped at {succeeded + failed} / {n}."
        else:
            drift_note = f" ({drift_count} with subtype drift)" if drift_count else ""
            tail = f"Bulk run finished: {succeeded} succeeded{drift_note}, {failed} failed"
            if skipped: tail += f", {skipped} skipped"
            tail += "."
        self._bulk_progress_lbl.config(text=tail)
        self._status(tail)
        self._bulk_inputs_changed()  # re-evaluate Start button against new state

    def _bulk_verify_complete(self, idx: int, update: Dict[str, Any]):
        """Main-thread; merges async-verify outcome into a bulk row that already
        rendered as 'done'. Stale rows (idx out of range) are silently dropped.
        """
        if idx < 1 or idx > len(self._bulk_rows):
            return
        row = self._bulk_rows[idx - 1]
        result = row.get("result") or {}
        # Patch the stored result so the row re-renders with the final verdict,
        # final cost, and recalibrated difficulty.
        result["verdict"]    = update.get("verdict") or result.get("verdict")
        result["usage"]      = update.get("usage")   or result.get("usage")
        result["difficulty"] = update.get("difficulty") or result.get("difficulty")
        self._bulk_set_row(idx, result=result)

        # Add the verify-only delta to bulk + session totals so cost columns
        # converge on the true post-verify total without double-counting.
        vu = update.get("verify_usage") or {}
        delta_cost = vu.get("cost_usd", 0.0) or 0.0
        self._bulk_run_cost += delta_cost
        self._session_cost  += delta_cost
        self._session_tokens["in"]      += vu.get("input_tokens", 0) or 0
        self._session_tokens["out"]     += vu.get("output_tokens", 0) or 0
        self._session_tokens["cache_r"] += vu.get("cache_read_input_tokens", 0) or 0
        self._session_tokens["cache_w"] += vu.get("cache_creation_input_tokens", 0) or 0
        tot = sum(self._session_tokens.values())
        self._cost_lbl.config(text=f"${self._session_cost:.3f} · {tot:,} tok")
        self._refresh_stats()
        self._refresh_insights()

    def _bulk_after_success(self, idx: int, result: Dict[str, Any]):
        """Main-thread; updates row + global session counters + History."""
        self._bulk_set_row(idx, status="done", result=result)
        usage = result.get("usage") or {}
        self._bulk_run_cost += usage.get("cost_usd", 0.0) or 0.0
        self._session_cost  += usage.get("cost_usd", 0.0) or 0.0
        self._session_tokens["in"]      += usage.get("input_tokens", 0) or 0
        self._session_tokens["out"]     += usage.get("output_tokens", 0) or 0
        self._session_tokens["cache_r"] += usage.get("cache_read_input_tokens", 0) or 0
        self._session_tokens["cache_w"] += usage.get("cache_creation_input_tokens", 0) or 0
        tot = sum(self._session_tokens.values())
        self._cost_lbl.config(text=f"${self._session_cost:.3f} · {tot:,} tok")
        self._refresh_stats()
        self._refresh_out()
        self._refresh_insights()

    def _bulk_worker(self, task_list: list[str], hint: str,
                       subtype: Optional[str]):
        n = len(task_list)
        self.after(0, lambda: self._bulk_run_started(task_list))
        succeeded = 0
        failed    = 0
        for i, section in enumerate(task_list, start=1):
            if self._bulk_stop.is_set():
                # Mark this and every later row as skipped, then exit.
                for j in range(i, n + 1):
                    self.after(0, lambda idx=j: self._bulk_set_row(idx, status="skipped"))
                self.after(0, lambda: self._bulk_run_finished(succeeded, failed, stopped=True))
                return

            started_at = datetime.now().strftime("%H:%M:%S")
            self.after(0, lambda idx=i, t=started_at: self._bulk_set_row(
                idx, status="running", started=t))
            self.after(0, lambda idx=i, total=n: self._bulk_progress_lbl.config(
                text=f"{idx - 1} / {total} — generating set {idx}…"))

            # Pre-pick an under-represented scenario + recently-overused topics
            # to avoid, so the bulk run rotates through diverse scenarios instead
            # of clustering on whatever the LLM defaults to. Bounded prompt cost.
            stats = self.db.coverage_stats(section, last_n=200)
            diversify = pick_diversification(stats, section) or {}

            attempts = 0
            done = False
            while attempts < 2 and not done:
                try:
                    result = self.rag.generate(
                        section, hint,
                        subtype=subtype,
                        on_progress=lambda m, idx=i: self.after(0, lambda msg=m, _i=idx:
                            self._bulk_set_row(_i, status="running", progress=msg)),
                        on_delta=None,
                        on_verify_complete=lambda upd, idx=i: self.after(
                            0, lambda u=upd, _i=idx: self._bulk_verify_complete(_i, u)),
                        force_scenario=diversify.get("scenario"),
                        avoid_topics=diversify.get("avoid_topics"),
                    )
                    self.after(0, lambda idx=i, r=result: self._bulk_after_success(idx, r))
                    succeeded += 1
                    done = True
                except Exception as e:
                    attempts += 1
                    logger.exception(f"Bulk set {i} attempt {attempts} failed")
                    if attempts >= 2:
                        err = str(e)
                        self.after(0, lambda idx=i, msg=err: self._bulk_set_row(
                            idx, status="failed", error=msg))
                        failed += 1
                    else:
                        self.after(0, lambda idx=i, msg=str(e): self._bulk_set_row(
                            idx, status="running",
                            progress=f"retrying ({msg[:40]})"))
                        time.sleep(1.0)

        self.after(0, lambda: self._bulk_run_finished(succeeded, failed))

    # ── Knowledge base tab ────────────────────────────────────────────────────

    def _tab_kb(self):
        top = tk.Frame(self._t_kb, bg=BG); top.pack(fill="x", padx=22, pady=(16, 6))
        tk.Label(top, text="Knowledge Base", bg=BG, fg=TEXT, font=FT).pack(side="left")

        fr = tk.Frame(top, bg=BG); fr.pack(side="right")
        tk.Label(fr, text="Filter:", bg=BG, fg=MUTED, font=FS).pack(side="left", padx=(0, 8))
        self._kbf = tk.StringVar(value="ALL")
        for v in ["ALL"] + list(SECTIONS.keys()):
            tk.Radiobutton(fr, text=v, variable=self._kbf, value=v,
                           bg=BG, fg=MUTED, selectcolor=PANEL, activebackground=BG,
                           font=FS, cursor="hand2", command=self._refresh_kb
                           ).pack(side="left", padx=3)

        tf = tk.Frame(self._t_kb, bg=BG); tf.pack(fill="both", expand=True, padx=22, pady=(0, 4))
        cols = ("ID", "Sec", "Source", "Qs", "Indexed", "Date")
        self._kbt = ttk.Treeview(tf, columns=cols, show="headings", height=14)
        for c, w in zip(cols, (44, 54, 100, 40, 74, 150)):
            self._kbt.heading(c, text=c)
            self._kbt.column(c, width=w, anchor="w" if c == "Date" else "center")
        vsb = ttk.Scrollbar(tf, orient="vertical", command=self._kbt.yview)
        self._kbt.configure(yscrollcommand=vsb.set)
        self._kbt.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._kbt.bind("<<TreeviewSelect>>", self._kb_sel)

        pf = tk.Frame(self._t_kb, bg=BORDER); pf.pack(fill="both", expand=True, padx=22, pady=(0, 14))
        self._kbprev = scrolledtext.ScrolledText(
            pf, bg=PANEL, fg=TEXT, font=FS, relief="flat", padx=12, pady=10, wrap=tk.WORD, height=10
        )
        self._kbprev.pack(fill="both", expand=True)
        self._refresh_kb()

    # ── History tab ───────────────────────────────────────────────────────────

    def _tab_out(self):
        top = tk.Frame(self._t_out, bg=BG); top.pack(fill="x", padx=22, pady=(16, 6))
        tk.Label(top, text="Output History", bg=BG, fg=TEXT, font=FT).pack(side="left")
        mkbtn(top, "↻  Refresh", self._refresh_out, bg=PANEL, fg=MUTED, font=FS, pady=5
              ).pack(side="right")

        tf = tk.Frame(self._t_out, bg=BG); tf.pack(fill="both", expand=True, padx=22, pady=(0, 4))
        cols = ("ID", "Section", "Diff", "Cost", "Verdict", "Generated At")
        self._outt = ttk.Treeview(tf, columns=cols, show="headings", height=10)
        for c, w in zip(cols, (50, 170, 60, 80, 110, 160)):
            self._outt.heading(c, text=c)
            self._outt.column(c, width=w, anchor="w" if c == "Generated At" else "center")
        vsb = ttk.Scrollbar(tf, orient="vertical", command=self._outt.yview)
        self._outt.configure(yscrollcommand=vsb.set)
        self._outt.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._outt.bind("<<TreeviewSelect>>", self._out_sel)

        pf = tk.Frame(self._t_out, bg=BORDER); pf.pack(fill="both", expand=True, padx=22, pady=(0, 4))
        self._outprev = scrolledtext.ScrolledText(
            pf, bg=PANEL, fg=TEXT, font=FS, relief="flat", padx=12, pady=10, wrap=tk.WORD
        )
        self._outprev.pack(fill="both", expand=True)

        bot = tk.Frame(self._t_out, bg=BG); bot.pack(fill="x", padx=22, pady=(0, 10))
        self._promobtn = mkbtn(bot, "✓  Add to Knowledge Base", self._promote,
                                bg=SUCCESS, pady=6, padx=14, state="disabled")
        self._promobtn.pack(side="left")
        self._refresh_out()

    # ── Insights tab ──────────────────────────────────────────────────────────

    def _tab_insights(self):
        top = tk.Frame(self._t_insights, bg=BG); top.pack(fill="x", padx=22, pady=(16, 6))
        tk.Label(top, text="Coverage · Bias · Telemetry", bg=BG, fg=TEXT, font=FT).pack(side="left")
        mkbtn(top, "↻  Refresh", self._refresh_insights, bg=PANEL, fg=MUTED, font=FS, pady=5
              ).pack(side="right")

        body = tk.Frame(self._t_insights, bg=BG); body.pack(fill="both", expand=True, padx=22, pady=(0, 14))

        # Two columns.
        left  = tk.Frame(body, bg=BG); left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right = tk.Frame(body, bg=BG); right.pack(side="right", fill="both", expand=True, padx=(8, 0))

        tk.Label(left, text="COVERAGE / BIAS",  bg=BG, fg=MUTED, font=FH).pack(anchor="w", pady=(0, 4))
        self._cov_box = scrolledtext.ScrolledText(left, bg=PANEL, fg=TEXT, font=FS,
                                                    relief="flat", padx=12, pady=10,
                                                    wrap=tk.WORD)
        self._cov_box.pack(fill="both", expand=True)

        tk.Label(right, text="TELEMETRY",       bg=BG, fg=MUTED, font=FH).pack(anchor="w", pady=(0, 4))
        self._tel_box = scrolledtext.ScrolledText(right, bg=PANEL, fg=TEXT, font=FS,
                                                    relief="flat", padx=12, pady=10,
                                                    wrap=tk.WORD)
        self._tel_box.pack(fill="both", expand=True)

        self._refresh_insights()

    # ── Actions ───────────────────────────────────────────────────────────────

    def _chk_api(self):
        ok, msg = api_status()
        if ok:
            self._dot.config(fg=SUCCESS); self._dlbl.config(text="Claude + Voyage ✓", fg=SUCCESS)
        else:
            self._dot.config(fg=DANGER); self._dlbl.config(text=msg, fg=DANGER)
            self._status(f"⚠  {msg}  |  see .env.example for setup")

    def _do_index(self):
        self._idx_btn.config(state="disabled", text="Indexing…")

        def worker():
            total = len(self.db.get_unindexed())
            if total == 0:
                self.after(0, lambda: (
                    self._idx_lbl.config(text="Nothing to index"),
                    self._idx_btn.config(state="normal", text="⊛  Index Knowledge Base"),
                    self._status("All documents already indexed")
                ))
                return
            try:
                def prog(done, n):
                    self.after(0, lambda _d=done, _n=n: (
                        self._idx_lbl.config(text=f"{_d}/{_n}…"),
                        self._refresh_stats()
                    ))
                done = self.rag.index_all(on_progress=prog)
                self.after(0, lambda: (
                    self._idx_lbl.config(text=f"✓ {done} indexed"),
                    self._idx_btn.config(state="normal", text="⊛  Index Knowledge Base"),
                    self._refresh_stats(), self._refresh_kb(),
                    self._status(f"Indexed {done} document(s) (batched)")
                ))
            except Exception as e:
                logger.exception("Index error")
                self.after(0, lambda err=str(e): (
                    self._idx_lbl.config(text="Error"),
                    self._idx_btn.config(state="normal", text="⊛  Index Knowledge Base"),
                    self._status(f"Index error: {err[:100]}")
                ))

        threading.Thread(target=worker, daemon=True).start()

    def _do_gen(self):
        self._launch_gen()

    def _do_regen(self):
        # Regenerate is a fresh call. The retrieved KB pool will rotate
        # naturally (cosine + MMR is deterministic but the embedding cache
        # may have changed), and the difficulty target / coverage steers
        # provide the actual entropy. The old `variation_seed` kwarg was
        # theatre — Claude can't use an opaque hex string as sampling
        # entropy with structured outputs enabled.
        self._launch_gen()

    def _launch_gen(self):
        if self._bulk_thread is not None and self._bulk_thread.is_alive():
            self._status("A bulk run is in progress — wait or stop it first.")
            return

        ok, msg = api_status()
        if not ok:
            messagebox.showerror("API Not Ready",
                f"{msg}\n\nCopy .env.example → .env and fill in your keys.")
            return

        section = self._gsec.get()
        if self.db.count(section, indexed_only=True) == 0:
            if not messagebox.askyesno("No Indexed Documents",
                f"No indexed documents for {SECTIONS[section]}.\n\n"
                "Add docs and click Index, or generate without RAG context?"):
                return

        hint = self._hint.get()
        self._last_section = section

        self._gbtn.config(state="disabled", text="Generating…")
        self._regenbtn.config(state="disabled")
        self._savebtn.config(state="disabled"); self._copybtn.config(state="disabled")
        self._gprog.config(text="")
        self._verdict_lbl.config(text="", fg=SUCCESS)
        self._dup_lbl.config(text="")
        self._wout("Contacting Claude…\n", self._gout)
        self._wout("", self._cout)
        self._clear_visuals()
        self._stream_buf: List[str] = []

        def on_delta(text: str):
            self._stream_buf.append(text)
            self.after(0, lambda: self._set_streaming("".join(self._stream_buf)))

        # Coverage-driven diversity steering for single-generation mode.
        # Bulk runs got this from day one; single-gen used to be "manual
        # control mode" where the user supplied the variety via hints,
        # but the coverage tracker is now mature enough that auto-steer
        # adds value without surprising the user (the steer goes into
        # the user-turn, not the role block — visible in the trace).
        stats = self.db.coverage_stats(section, last_n=200)
        diversify = pick_diversification(stats, section) or {}

        def worker():
            try:
                result = self.rag.generate(
                    section, hint,
                    on_progress=lambda m: self.after(0, lambda msg=m: self._gprog.config(text=f"⟳  {msg}")),
                    on_delta=on_delta,
                    on_verify_complete=lambda upd: self.after(
                        0, lambda u=upd: self._on_verify_complete(u)),
                    force_scenario=diversify.get("scenario"),
                    avoid_topics=diversify.get("avoid_topics"),
                )
                self.after(0, lambda: self._gen_ok(result, section))
            except Exception as e:
                logger.exception("Generation failed")
                self.after(0, lambda err=str(e): self._gen_err(err))

        threading.Thread(target=worker, daemon=True).start()

    def _set_streaming(self, text: str):
        self._wout(f"⟳  Streaming…\n\n{text[:4000]}", self._gout)

    def _gen_ok(self, result: Dict[str, Any], section: str):
        data       = result["data"]
        retrieved  = result["retrieved"]
        usage      = result["usage"]
        verdict    = result["verdict"]
        coverage   = result["coverage"]
        difficulty = result["difficulty"]
        dup        = result["dup_warning"]

        self._last_data = data
        self._last_section = section
        self._last_retrieved = retrieved

        self._wout(format_qset(data), self._gout)
        self._render_visuals(data)

        # Retrieved context panel.
        ctx_lines = [f"Retrieved {len(retrieved)} doc(s) (MMR λ={self.rag.mmr_lambda}):", ""]
        for i, (sc, doc) in enumerate(retrieved, 1):
            ctx_lines.append(f"[{i}] ID #{doc['id']}  ·  {doc['section']}  ·  sim {sc:.3f}  ·  {doc['source']}")
            ctx_lines.append(f"    {doc['embed_text'][:120].replace(chr(10),' ')}…")
            ctx_lines.append("")
        if difficulty:
            ctx_lines.append("─" * 36)
            ctx_lines.append(f"Difficulty (calibrated): {difficulty.get('set_difficulty')}")
            ctx_lines.append(f"  range: {difficulty.get('min')} → {difficulty.get('max')}")
        if coverage:
            ctx_lines.append("─" * 36)
            tc = coverage.get("topic_counts", {})
            sc_ = coverage.get("scenario_counts", {})
            if tc: ctx_lines.append(f"Topics: {dict(tc)}")
            if sc_: ctx_lines.append(f"Scenarios: {dict(sc_)}")
            for f in coverage.get("flags", []):
                ctx_lines.append(f"⚠ {f}")
        if usage:
            ctx_lines.append("─" * 36)
            ctx_lines.append(f"Tokens: in {usage['input_tokens']} · out {usage['output_tokens']}")
            ctx_lines.append(f"Cache: read {usage['cache_read_input_tokens']} · "
                              f"write {usage['cache_creation_input_tokens']}")
            ctx_lines.append(f"Cost: ${usage['cost_usd']:.4f}")
        self._wout("\n".join(ctx_lines).strip(), self._cout)

        # Verdict badge.
        self._render_verdict_badge(verdict)
        # Remember which row this generation produced so the (possibly delayed)
        # async verify callback can detect a stale follow-up after regenerate.
        self._last_row_id = result.get("row_id")

        if dup:
            self._dup_lbl.config(text=f"⚠ {dup}", fg=WARN)

        # Session totals.
        if usage:
            self._session_cost += usage.get("cost_usd", 0) or 0
            self._session_tokens["in"]      += usage.get("input_tokens", 0) or 0
            self._session_tokens["out"]     += usage.get("output_tokens", 0) or 0
            self._session_tokens["cache_r"] += usage.get("cache_read_input_tokens", 0) or 0
            self._session_tokens["cache_w"] += usage.get("cache_creation_input_tokens", 0) or 0
            tot = sum(self._session_tokens.values())
            self._cost_lbl.config(text=f"${self._session_cost:.3f} · {tot:,} tok")

        self._gbtn.config(state="normal", text="⚡  GENERATE QUESTIONS")
        self._regenbtn.config(state="normal")
        self._savebtn.config(state="normal"); self._copybtn.config(state="normal")
        self._gprog.config(text="✓  Done!")
        self._refresh_stats(); self._refresh_out(); self._refresh_insights()
        self._status(f"Generated {SECTIONS[section]} using {len(retrieved)} context doc(s)")

    def _render_verdict_badge(self, verdict: Optional[Dict[str, Any]]):
        """Update the verdict label from a verdict dict. Handles three states:
        none, pending (async verify in flight), and final (pass/fail)."""
        if not verdict:
            self._verdict_lbl.config(text="", fg=SUCCESS)
            return
        if verdict.get("pending"):
            sym = verdict.get("symbolic_qr") or {}
            disagreed = len(sym.get("disagreed", []))
            if disagreed:
                self._verdict_lbl.config(
                    text=f"⚠ {disagreed} arithmetic mismatch · ⟳ verifying…", fg=DANGER)
            else:
                self._verdict_lbl.config(text="⟳ verifying answers…", fg=ACCENT)
            return
        ok = verdict.get("overall_correct", True)
        mode = verdict.get("mode", "?")
        sym = verdict.get("symbolic_qr") or {}
        disagreed = len(sym.get("disagreed", []))
        if ok and disagreed == 0:
            conf = verdict.get("confidence", "—") if mode == "single" else (
                "unanimous" if verdict.get("unanimous") else "majority")
            self._verdict_lbl.config(text=f"✓ verified ({mode} · {conf})", fg=SUCCESS)
        else:
            bits = []
            fq = verdict.get("flagged_questions", [])
            if fq: bits.append(f"{len(fq)} flagged")
            if disagreed: bits.append(f"{disagreed} arithmetic mismatch")
            self._verdict_lbl.config(text=f"⚠ {', '.join(bits) or 'issues'}", fg=DANGER)

    def _on_verify_complete(self, update: Dict[str, Any]):
        """Main-thread handler for async verify completion. Drops stale callbacks
        (user has regenerated since), updates the verdict badge, and adds the
        verify-only token/cost delta to session totals."""
        row_id = update.get("row_id")
        # Stale: a newer generation has happened. Skip UI mutation but keep the
        # db row updated (the worker already did that).
        if row_id is None or row_id != getattr(self, "_last_row_id", None):
            return
        self._render_verdict_badge(update.get("verdict"))
        vu = update.get("verify_usage") or {}
        self._session_cost += vu.get("cost_usd", 0) or 0
        self._session_tokens["in"]      += vu.get("input_tokens", 0) or 0
        self._session_tokens["out"]     += vu.get("output_tokens", 0) or 0
        self._session_tokens["cache_r"] += vu.get("cache_read_input_tokens", 0) or 0
        self._session_tokens["cache_w"] += vu.get("cache_creation_input_tokens", 0) or 0
        tot = sum(self._session_tokens.values())
        self._cost_lbl.config(text=f"${self._session_cost:.3f} · {tot:,} tok")
        self._refresh_stats()
        self._refresh_insights()

    def _gen_err(self, err: str):
        self._wout(f"ERROR\n{'─'*40}\n{err}", self._gout)
        self._gbtn.config(state="normal", text="⚡  GENERATE QUESTIONS")
        self._regenbtn.config(state="disabled")
        self._gprog.config(text="")
        self._verdict_lbl.config(text="")
        self._status(f"Error: {err[:80]}")

    # ── Visuals rendering helpers ─────────────────────────────────────────────

    def _clear_visuals(self):
        self._photo_refs.clear()
        self._visuals.clear()
        self._vis_status.config(text="")

    def _add_image_to_visuals(self, image, label: str = ""):
        if image is None or not _HAS_PIL:
            return
        if label:
            tk.Label(self._visuals.inner, text=label, bg=PANEL2, fg=ACCENT,
                      font=FH).pack(anchor="w", padx=10, pady=(10, 2))
        # Resize if too wide for the panel.
        canvas_width = self._visuals.winfo_width() or 420
        max_w = max(canvas_width - 30, 200)
        if hasattr(image, "size"):
            w, h = image.size
            if w > max_w:
                ratio = max_w / w
                new_size = (int(w * ratio), int(h * ratio))
                image = image.resize(new_size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self._photo_refs.append(photo)
        lbl = tk.Label(self._visuals.inner, image=photo, bg=PANEL2)
        lbl.pack(padx=10, pady=4)

    def _render_visuals(self, data: Dict[str, Any]):
        self._clear_visuals()
        if not _HAS_PIL:
            tk.Label(self._visuals.inner,
                      text="(install Pillow for visuals)",
                      bg=PANEL2, fg=MUTED, font=FS).pack(padx=10, pady=10)
            return

        try:
            visuals = render_visuals_for(data)
        except Exception as e:
            logger.exception("Visual render failed")
            tk.Label(self._visuals.inner, text=f"render error: {e}",
                      bg=PANEL2, fg=DANGER, font=FS).pack(padx=10, pady=10)
            return

        section = data.get("section")
        any_added = False

        if section == "QR" and "chart" in visuals:
            self._add_image_to_visuals(visuals["chart"], label="QR Stimulus Chart")
            any_added = True

        if section == "AR":
            if "set_a" in visuals:
                self._add_image_to_visuals(visuals["set_a"], label="Set A Panels")
                any_added = True
            if "set_b" in visuals:
                self._add_image_to_visuals(visuals["set_b"], label="Set B Panels")
                any_added = True
            for i, img in enumerate(visuals.get("tests") or []):
                self._add_image_to_visuals(img, label=f"Test Shape {i + 1}")
                any_added = True

        if section == "DM" and "venns" in visuals:
            for n, img in sorted(visuals["venns"].items()):
                self._add_image_to_visuals(img, label=f"Q{n} — Venn")
                any_added = True

        if not any_added:
            tk.Label(self._visuals.inner,
                      text=f"(no visuals for {SECTIONS.get(section, section)})",
                      bg=PANEL2, fg=MUTED, font=FS).pack(padx=10, pady=10)
        else:
            self._vis_status.config(text=f"{len(self._photo_refs)} image(s)")

    def _wout(self, text: str, widget):
        widget.config(state="normal"); widget.delete(1.0, tk.END)
        widget.insert(tk.END, text); widget.config(state="disabled")

    # ── KB / history / insights actions ──────────────────────────────────────

    def _save_kb(self):
        if self._last_data and self._last_section:
            self.db.add_doc(self._last_section, self._last_data, source="generated")
            self._refresh_stats(); self._refresh_kb()
            self._gprog.config(text="✓  Saved — re-index to activate for RAG")
            self._savebtn.config(state="disabled")
            emit("kb_promote", section=self._last_section)

    def _copy(self):
        self._gout.config(state="normal")
        t = self._gout.get(1.0, tk.END)
        self._gout.config(state="disabled")
        self.clipboard_clear(); self.clipboard_append(t)
        self._copybtn.config(text="✓  Copied!")
        self.after(2000, lambda: self._copybtn.config(text="⊞  Copy"))

    def _import(self):
        p = filedialog.askopenfilename(
            title="Import UCAT Questions (JSON)",
            filetypes=[("JSON", "*.json"), ("All", "*.*")]
        )
        if not p: return
        try:
            result = self.db.import_json(p)
            parts = [f"Imported {result['added']} new document(s)."]
            if result["skipped"] > 0:
                parts.append(
                    f"Skipped {result['skipped']} duplicate(s) already in the KB."
                )
            if result["ignored"] > 0:
                parts.append(
                    f"Ignored {result['ignored']} malformed entry(ies)."
                )
            if result["added"] > 0:
                parts.append("\nNow click '⊛ Index Knowledge Base'.")
            messagebox.showinfo("Import", "\n".join(parts))
            self._refresh_stats(); self._refresh_kb()
        except Exception as e:
            messagebox.showerror("Import Error", str(e))

    def _import_crawler(self):
        """Import a folder of crawler-captured questions (manifest.json + per-Q artifacts)."""
        p = filedialog.askdirectory(
            title="Select Crawler output folder (the one containing manifest.json)",
            mustexist=True,
        )
        if not p: return
        from pathlib import Path
        from .crawler_import import import_from_crawler

        def worker():
            try:
                result = import_from_crawler(self.db, Path(p))
                self.after(0, lambda: self._crawler_import_done(result))
            except Exception as e:
                logger.exception("Crawler import failed")
                self.after(0, lambda err=str(e): messagebox.showerror("Crawler Import Error", err))

        self._status("Importing from crawler…")
        threading.Thread(target=worker, daemon=True).start()

    def _crawler_import_done(self, result: Dict[str, Any]):
        c = result["counts"]
        lines = [
            f"Imported {result['total']} question set(s):",
            "",
            f"  VR:   {c['VR']} sets",
            f"  DM:   {c['DM']} sets",
            f"  QR:   {c['QR']} sets",
            f"  SJT:  {c['SJT']} sets",
        ]
        if result.get("ar_skipped"):
            lines.append(f"\n  AR:   {result['ar_skipped']} captures skipped (UCAT removed AR)")
        if result.get("skipped"):
            lines.append("")
            lines.append(f"  {len(result['skipped'])} group(s) had captures dropped:")
            for s in result["skipped"][:5]:
                lines.append(f"    • {s['bucket']} ({s['section']}): {s['reason']}")
            if len(result["skipped"]) > 5:
                lines.append(f"    … and {len(result['skipped']) - 5} more")
        if result.get("errors"):
            lines.append("")
            lines.append(f"  {len(result['errors'])} error(s):")
            for err in result["errors"][:3]:
                lines.append(f"    • {err[:120]}")
        lines.append("")
        lines.append("Now click '⊛ Index Knowledge Base' to embed them.")
        messagebox.showinfo("Crawler Import", "\n".join(lines))
        self._refresh_stats(); self._refresh_kb()
        self._status(f"Crawler import: {result['total']} sets added")

    def _add_samples(self):
        for q in SAMPLES:
            self.db.add_doc(q["section"], q, source="sample")
        messagebox.showinfo("Samples Added",
            f"Added {len(SAMPLES)} sample documents.\nNow click '⊛ Index Knowledge Base'.")
        self._refresh_stats(); self._refresh_kb()

    def _export(self):
        rows = self.db.get_generated(limit=10000)
        if not rows:
            messagebox.showinfo("Export", "No generated questions yet."); return
        p = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")],
            initialfile="ucat_generated.json"
        )
        if p:
            with open(p, "w", encoding="utf-8") as f:
                json.dump([r["data"] for r in rows], f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Exported", f"Saved {len(rows)} sets to:\n{p}")

    def _refresh_kb(self):
        filt = self._kbf.get()
        docs = self.db.get_all_docs(None if filt == "ALL" else filt, limit=2000)
        for iid in self._kbt.get_children(): self._kbt.delete(iid)
        for d in docs:
            self._kbt.insert("", "end", iid=str(d["id"]),
                              values=(d["id"], d["section"], d["source"],
                                      len(d["data"].get("questions", [])),
                                      "✓" if d["embedding"] else "○",
                                      d["created"][:16]))

    def _kb_sel(self, _e):
        sel = self._kbt.selection()
        if not sel: return
        docs = self.db.get_all_docs(limit=100000)
        doc  = next((d for d in docs if d["id"] == int(sel[0])), None)
        if doc:
            self._kbprev.config(state="normal")
            self._kbprev.delete(1.0, tk.END)
            self._kbprev.insert(tk.END, format_qset(doc["data"]))
            self._kbprev.config(state="disabled")

    def _refresh_out(self):
        rows = self.db.get_generated(limit=500)
        for iid in self._outt.get_children(): self._outt.delete(iid)
        for r in rows:
            cost = f"${r['usage']['cost_usd']:.3f}" if r.get("usage") else "—"
            d = r.get("difficulty")
            d_str = f"{d:.1f}" if isinstance(d, (int, float)) else "—"
            v = r.get("verdict") or {}
            if not v:
                badge = "—"
            elif v.get("overall_correct", True):
                badge = "✓"
            else:
                fq = len(v.get("flagged_questions") or [])
                sym = (v.get("symbolic_qr") or {}).get("disagreed") or []
                badge = f"⚠ {fq + len(sym)}"
            self._outt.insert("", "end", iid=str(r["id"]),
                               values=(r["id"], SECTIONS.get(r["section"], r["section"]),
                                       d_str, cost, badge,
                                       (r["created"] or "")[:16]))

    def _out_sel(self, _e):
        sel = self._outt.selection()
        if not sel: return
        self._sel_gen_id = int(sel[0])
        rows = self.db.get_generated(limit=500)
        row  = next((r for r in rows if r["id"] == self._sel_gen_id), None)
        if not row: return

        text = format_qset(row["data"])
        v = row.get("verdict") or {}
        if v and not v.get("overall_correct", True):
            text += "\n\n" + "─" * 40 + "\n⚠  VERDICT FLAGGED:\n"
            for fq in v.get("flagged_questions") or []:
                text += f"  • Q{fq} marked incorrect by majority\n"
            for d in (v.get("symbolic_qr") or {}).get("disagreed") or []:
                text += (f"  • Q{d['number']}: marked {d['marked_value']} but "
                          f"explanation computes {d['computed_value']}\n")
        c = row.get("coverage") or {}
        if c.get("flags"):
            text += "\n" + "─" * 40 + "\nCOVERAGE FLAGS:\n"
            for f in c["flags"]:
                text += f"  • {f}\n"
        u = row.get("usage")
        if u:
            text += (f"\n{'─'*40}\nUsage: in {u['input_tokens']}  ·  out {u['output_tokens']}  ·  "
                     f"cache_r {u['cache_read_input_tokens']}  ·  cost ${u['cost_usd']:.4f}\n")
        self._outprev.config(state="normal")
        self._outprev.delete(1.0, tk.END)
        self._outprev.insert(tk.END, text)
        self._outprev.config(state="disabled")
        self._promobtn.config(state="normal")

    def _promote(self):
        if self._sel_gen_id:
            # Pass embed_doc so the new KB row gets its embedding computed
            # in the same call. Without this, the row sits in the KB with
            # embedding=NULL and doesn't participate in retrieval until the
            # user manually clicks ⊛ Index Knowledge Base.
            from .llm import embed_doc
            self.db.promote_to_kb(
                self._sel_gen_id,
                embed_fn=embed_doc,
                embed_model=self.settings.get("embed"),
            )
            self._refresh_stats(); self._refresh_kb()
            self._promobtn.config(state="disabled")
            self._status("Added to knowledge base and indexed.")

    def _refresh_stats(self):
        for code in SECTIONS:
            idx = self.db.count(code, indexed_only=True)
            tot = self.db.count(code, indexed_only=False)
            self._slabels[code].config(text=f"{idx}/{tot}")

    def _refresh_insights(self):
        # Coverage / bias.
        rows = self.db.get_generated(limit=500)
        from .coverage import aggregate_history
        agg = aggregate_history(rows)
        cov_lines = [f"Generated runs: {agg['rows']}", "",
                      "By section:"]
        for sec, n in sorted(agg.get("sections", {}).items()):
            cov_lines.append(f"  {sec}: {n}")
        if agg.get("topics"):
            cov_lines.append(""); cov_lines.append("Top topics:")
            for t, n in sorted(agg["topics"].items(), key=lambda x: -x[1])[:25]:
                cov_lines.append(f"  {t:<30s}  {n}")
        if agg.get("scenarios"):
            cov_lines.append(""); cov_lines.append("Scenarios:")
            for s, n in sorted(agg["scenarios"].items(), key=lambda x: -x[1]):
                cov_lines.append(f"  {s:<20s}  {n}")
        if agg.get("flag_counts"):
            cov_lines.append(""); cov_lines.append("Bias / coverage flags raised:")
            for f, n in sorted(agg["flag_counts"].items(), key=lambda x: -x[1]):
                cov_lines.append(f"  ({n}×) {f}")
        if agg.get("gaps"):
            cov_lines.append(""); cov_lines.append("Gap warnings:")
            for g in agg["gaps"]:
                cov_lines.append(f"  • {g}")
        # Difficulty distribution.
        diffs = [r["difficulty"] for r in rows if r.get("difficulty") is not None]
        if diffs:
            cov_lines.append(""); cov_lines.append("Difficulty (calibrated):")
            cov_lines.append(f"  count {len(diffs)}  ·  mean {sum(diffs)/len(diffs):.2f}"
                              f"  ·  range {min(diffs):.1f}–{max(diffs):.1f}")
            buckets = [0] * 5
            for d in diffs:
                b = min(int(d) - 1, 4)
                if b < 0: b = 0
                buckets[b] += 1
            for i, n in enumerate(buckets, 1):
                bar = "█" * min(n, 40)
                cov_lines.append(f"  {i:.1f}–{i+1:.1f}  {bar}  ({n})")

        self._cov_box.config(state="normal")
        self._cov_box.delete(1.0, tk.END)
        self._cov_box.insert(tk.END, "\n".join(cov_lines))
        self._cov_box.config(state="disabled")

        # Telemetry summary.
        agg_t = aggregate(last_n=2000)
        tel_lines = [f"Events tracked: {agg_t.get('events', 0)}", ""]
        if agg_t.get("by_event"):
            tel_lines.append("By event type:")
            for e, n in sorted(agg_t["by_event"].items(), key=lambda x: -x[1]):
                tel_lines.append(f"  {e:<32s} {n}")
        if agg_t.get("tokens"):
            t = agg_t["tokens"]
            tel_lines.append(""); tel_lines.append("Tokens:")
            tel_lines.append(f"  input        {t['in']:>10,}")
            tel_lines.append(f"  output       {t['out']:>10,}")
            tel_lines.append(f"  cache read   {t['cache_read']:>10,}")
            tel_lines.append(f"  cache write  {t['cache_write']:>10,}")
        tel_lines.append(""); tel_lines.append(f"Total cost: ${agg_t.get('total_cost_usd', 0):.4f}")
        self._tel_box.config(state="normal")
        self._tel_box.delete(1.0, tk.END)
        self._tel_box.insert(tk.END, "\n".join(tel_lines))
        self._tel_box.config(state="disabled")

    def _status(self, msg: str):
        self._sbar.config(text=msg)
        self.after(7000, lambda: self._sbar.config(text="Ready"))

    def on_close(self):
        # Halt any in-flight bulk run; give the worker up to 5s to exit before destroy.
        if self._bulk_thread is not None and self._bulk_thread.is_alive():
            self._bulk_stop.set()
            self._bulk_thread.join(timeout=5.0)
        self.db.close()
        self.destroy()


def run():
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()
