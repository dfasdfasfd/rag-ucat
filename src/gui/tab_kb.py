"""
Knowledge Base tab: document management, corpus coverage map, screenshot import.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext

from src.config import SECTIONS
from src.gui.theme import BG, PANEL, BORDER, TEXT, MUTED, ACCENT, WARN, FM, FS, FSB, FT, FH, mkbtn
from src.gui.tab_generate import format_qset


class TabKB:
    """Knowledge Base management tab with corpus coverage display."""

    def __init__(self, parent, app):
        self.app = app
        self.frame = tk.Frame(parent, bg=BG)
        self._build()

    def _build(self):
        top = tk.Frame(self.frame, bg=BG)
        top.pack(fill="x", padx=22, pady=(16, 6))
        tk.Label(top, text="Knowledge Base", bg=BG, fg=TEXT, font=FT).pack(side="left")

        # Filter
        fr = tk.Frame(top, bg=BG)
        fr.pack(side="right")
        tk.Label(fr, text="Filter:", bg=BG, fg=MUTED, font=FS).pack(side="left", padx=(0, 8))
        self.filter_var = tk.StringVar(value="ALL")
        for v in ["ALL"] + list(SECTIONS.keys()):
            tk.Radiobutton(
                fr, text=v, variable=self.filter_var, value=v,
                bg=BG, fg=MUTED, selectcolor=PANEL, activebackground=BG,
                font=FS, cursor="hand2", command=self.refresh
            ).pack(side="left", padx=3)

        mkbtn(top, "📊 Coverage", self.app.show_coverage,
              bg=PANEL, fg=WARN, font=FS, pady=5).pack(side="right", padx=8)

        # Treeview
        tf = tk.Frame(self.frame, bg=BG)
        tf.pack(fill="both", expand=True, padx=22, pady=(0, 4))
        cols = ("ID", "Sec", "Source", "Type", "Qs", "Indexed", "Score", "Date")
        self.tree = ttk.Treeview(tf, columns=cols, show="headings", height=14)
        for c, w in zip(cols, (44, 54, 80, 80, 40, 60, 50, 130)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w" if c == "Date" else "center")
        vsb = ttk.Scrollbar(tf, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # Delete button
        btn_frame = tk.Frame(self.frame, bg=BG)
        btn_frame.pack(fill="x", padx=22, pady=(0, 4))
        self.del_btn = mkbtn(btn_frame, "✕  Delete Selected", self._delete_selected,
                              bg="#F85149", font=FS, pady=4, state="disabled")
        self.del_btn.pack(side="left")

        # Preview
        pf = tk.Frame(self.frame, bg=BORDER)
        pf.pack(fill="both", expand=True, padx=22, pady=(0, 14))
        self.preview = scrolledtext.ScrolledText(
            pf, bg=PANEL, fg=TEXT, font=FS, relief="flat",
            padx=12, pady=10, wrap=tk.WORD, height=10
        )
        self.preview.pack(fill="both", expand=True)

    def refresh(self):
        filt = self.filter_var.get()
        section = None if filt == "ALL" else filt
        docs = self.app.db.get_all_docs(section, limit=2000)

        for iid in self.tree.get_children():
            self.tree.delete(iid)

        for d in docs:
            qs = len(d["data"].get("questions", []))
            indexed = "✓" if d["embedding"] else "○"
            score = f"{d['quality_score']:.1f}" if d.get("quality_score") else "-"
            dtype = d.get("data_type", "-") or "-"
            self.tree.insert("", "end", iid=str(d["id"]),
                              values=(d["id"], d["section"], d["source"],
                                      dtype, qs, indexed, score,
                                      d["created"][:16]))

    def _on_select(self, _e):
        sel = self.tree.selection()
        if not sel:
            self.del_btn.config(state="disabled")
            return
        self.del_btn.config(state="normal")
        doc_id = int(sel[0])
        docs = self.app.db.get_all_docs(limit=100000)
        doc = next((d for d in docs if d["id"] == doc_id), None)
        if doc:
            self.preview.config(state="normal")
            self.preview.delete(1.0, tk.END)
            self.preview.insert(tk.END, format_qset(doc["data"]))
            self.preview.config(state="disabled")

    def _delete_selected(self):
        sel = self.tree.selection()
        if sel:
            from tkinter import messagebox
            if messagebox.askyesno("Delete", f"Delete {len(sel)} document(s)?"):
                for iid in sel:
                    self.app.db.delete_doc(int(iid))
                self.refresh()
                self.app.refresh_stats()
