"""
Main application: wires together all backend modules and GUI tabs.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import threading
import os

from src.config import (
    APP_TITLE, SECTIONS, SECTION_GEN_PARAMS,
    DEFAULT_EMBED, DEFAULT_SCORER, DEFAULT_VISION,
)
from src.database import Database
from src.ollama_client import OllamaClient
from src.embeddings import EmbeddingEngine
from src.retrieval import Retriever
from src.prompts import PromptBuilder
from src.quality import QualityPipeline
from src.generator import GenerationPipeline
from src.feedback import FeedbackEngine
from src.corpus import CorpusAnalyzer
from src.ingestion import ScreenshotIngester
from src.calibration import CalibrationManager
from src.batch import BatchGenerator

from src.gui.theme import BG, PANEL, MUTED, ACCENT, SUCCESS, DANGER, WARN, FS, FSB, apply_theme, mkbtn
from src.gui.sidebar import Sidebar
from src.gui.tab_generate import TabGenerate
from src.gui.tab_kb import TabKB
from src.gui.tab_output import TabOutput
from src.gui.tab_calibrate import TabCalibrate
from src.gui.tab_batch import TabBatch


# ─── Built-in sample documents ──────────────────────────────────────────────
# Loaded from sample_questions.json if available

def _load_samples():
    sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                "sample_questions.json")
    if os.path.exists(sample_path):
        with open(sample_path, encoding="utf-8") as f:
            return json.load(f)
    return []


class App(tk.Tk):
    """Main UCAT Trainer application."""

    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1380x920")
        self.minsize(1020, 680)
        self.configure(bg=BG)

        # Initialize backend
        self.ollama = OllamaClient()
        self.db = Database()
        self.embedding_engine = EmbeddingEngine(self.ollama, self.db)
        self.feedback = FeedbackEngine(self.db)
        self.retriever = Retriever(self.db, self.embedding_engine, self.feedback)
        self.quality_pipeline = QualityPipeline(self.ollama, self.db, DEFAULT_SCORER)
        self.generator = GenerationPipeline(
            self.db, self.ollama, self.embedding_engine,
            self.retriever, self.quality_pipeline, self.feedback
        )
        self.corpus = CorpusAnalyzer(self.db)
        self.ingester = ScreenshotIngester(
            self.ollama, self.db, self.embedding_engine,
            vision_model=DEFAULT_VISION
        )
        self.calibration = CalibrationManager(self.db)
        self.batch_gen = BatchGenerator(self.generator, self.calibration, self.db)

        # Apply theme
        apply_theme(self)

        # Build UI
        self._build_ui()

        # Check Ollama connection
        self.after(700, self._check_ollama)

        # Handle close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=PANEL, height=54)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="◈  UCAT TRAINER", bg=PANEL, fg=ACCENT,
                 font=("Courier New", 15, "bold")).pack(side="left", padx=22, pady=12)
        tk.Label(hdr, text="RAG v2", bg=PANEL, fg=WARN, font=FSB).pack(side="left", pady=12)

        self._dot = tk.Label(hdr, text="●", bg=PANEL, fg=DANGER, font=("Courier New", 13))
        self._dlbl = tk.Label(hdr, text="Checking Ollama...", bg=PANEL, fg=MUTED, font=FS)
        self._dot.pack(side="right", padx=(0, 18))
        self._dlbl.pack(side="right", padx=(0, 4))

        # Body
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # Sidebar
        self.sidebar = Sidebar(body, self)

        # Notebook with tabs
        nb = ttk.Notebook(body)
        nb.pack(side="left", fill="both", expand=True)

        self.tab_gen = TabGenerate(nb, self)
        self.tab_kb = TabKB(nb, self)
        self.tab_out = TabOutput(nb, self)
        self.tab_cal = TabCalibrate(nb, self)
        self.tab_batch = TabBatch(nb, self)

        nb.add(self.tab_gen.frame, text="  ⚡  GENERATE  ")
        nb.add(self.tab_kb.frame, text="  🗄  KNOWLEDGE BASE  ")
        nb.add(self.tab_out.frame, text="  📋  OUTPUT HISTORY  ")
        nb.add(self.tab_cal.frame, text="  🎯  CALIBRATE  ")
        nb.add(self.tab_batch.frame, text="  📦  BATCH  ")

        # Status bar
        self._sbar = tk.Label(self, text="Ready", bg="#090C10", fg=MUTED,
                               font=FS, anchor="w")
        self._sbar.pack(fill="x", padx=12, pady=(2, 4))

        self.refresh_stats()

    # ─── Ollama Connection ───────────────────────────────────────────────────

    def _check_ollama(self):
        if self.ollama.check_connection():
            self._dot.config(fg=SUCCESS)
            self._dlbl.config(text="Ollama connected ✓", fg=SUCCESS)
            self.refresh_models()
        else:
            self._dot.config(fg=DANGER)
            self._dlbl.config(text="Ollama offline", fg=DANGER)
            messagebox.showwarning(
                "Ollama Not Running",
                "Ollama is not running.\n\n"
                "Start it with: ollama serve\n\n"
                "Then pull required models:\n"
                "  ollama pull mxbai-embed-large\n"
                "  ollama pull qwen2.5:14b\n"
                "  ollama pull qwen2.5:7b"
            )

    # ─── Actions ─────────────────────────────────────────────────────────────

    def refresh_models(self):
        models = self.ollama.list_models()
        self.sidebar.update_models(models)

    def refresh_stats(self):
        self.sidebar.update_stats(self.db)
        self.sidebar.update_calibration(self.calibration)

    def status(self, msg):
        self._sbar.config(text=msg)
        self.after(7000, lambda: self._sbar.config(text="Ready"))

    def do_index(self):
        self.sidebar.idx_btn.config(state="disabled", text="Indexing...")
        emb_model = self.sidebar.emb_var.get()
        self.embedding_engine.model = emb_model

        def worker():
            try:
                def prog(i, n, phase):
                    self.after(0, lambda: (
                        self.sidebar.idx_lbl.config(text=f"{phase}: {i+1}/{n}"),
                        self.refresh_stats()
                    ))
                result = self.embedding_engine.index_and_reindex(on_progress=prog)

                # Rebuild FTS index
                self.db.rebuild_fts()

                # Infer data types for docs without them
                self._infer_data_types()

                total = result["new"] + result["reindexed"]
                self.after(0, lambda: (
                    self.sidebar.idx_lbl.config(text=f"✓ {total} indexed"),
                    self.sidebar.idx_btn.config(state="normal", text="⊛  Index Knowledge Base"),
                    self.refresh_stats(),
                    self.tab_kb.refresh(),
                    self.status(f"Indexed {result['new']} new, re-indexed {result['reindexed']}")
                ))
            except Exception as e:
                self.after(0, lambda err=str(e): (
                    self.sidebar.idx_lbl.config(text="Error"),
                    self.sidebar.idx_btn.config(state="normal", text="⊛  Index Knowledge Base"),
                    self.status(f"Index error: {err[:100]}")
                ))

        threading.Thread(target=worker, daemon=True).start()

    def _infer_data_types(self):
        """Infer data_type for all docs that don't have one."""
        docs = self.db.get_all_docs(limit=10000)
        for doc in docs:
            if not doc.get("data_type"):
                dt = self.embedding_engine.infer_data_type(doc["data"], doc["section"])
                self.db.set_data_type(doc["id"], dt)

    def do_generate(self):
        if not self.ollama.check_connection():
            messagebox.showerror("Ollama Not Running",
                                  "Start Ollama with: ollama serve")
            return

        # Record implicit regenerate signal if there was a previous generation
        if self.tab_gen._last_data and self.feedback:
            last_gen = self.db.get_generated(limit=1)
            if last_gen:
                self.feedback.record_regenerate(
                    last_gen[0]["id"], last_gen[0].get("context_ids", [])
                )

        section = self.tab_gen.section_var.get()
        hint = self.tab_gen.hint_var.get()
        difficulty = self.tab_gen.diff_var.get()

        # Check prompt drift
        drift_msg = self.calibration.check_prompt_drift(section)
        if drift_msg:
            self.status(drift_msg)
            self.sidebar.update_calibration(self.calibration)

        if self.db.count(section, indexed_only=True) == 0:
            if not messagebox.askyesno("No Indexed Documents",
                f"No indexed documents for {SECTIONS[section]}.\n\n"
                "Generate anyway (no RAG context)?"):
                return

        # Apply per-section model override from sidebar
        section_model = self.sidebar.get_section_model(section)
        from src.config import SECTION_GEN_PARAMS
        SECTION_GEN_PARAMS[section]["preferred_llm"] = section_model

        self.tab_gen.set_generating(True)
        self.tab_gen.clear_output()

        def on_token(token):
            self.after(0, lambda t=token: self.tab_gen.append_token(t))

        def on_progress(msg):
            self.after(0, lambda m=msg: self.tab_gen.prog_lbl.config(text=f"⟳  {m}"))

        def worker():
            try:
                data, retrieved, report = self.generator.generate(
                    section, hint, difficulty,
                    on_progress=on_progress,
                    stream=True, on_token=on_token,
                    abort_flag=self.tab_gen.get_abort_flag()
                )
                self.after(0, lambda: self._gen_ok(data, retrieved, report, section))
            except Exception as e:
                self.after(0, lambda err=str(e): self._gen_err(err))

        threading.Thread(target=worker, daemon=True).start()

    def _gen_ok(self, data, retrieved, report, section):
        self.tab_gen.set_generating(False)
        self.tab_gen.show_output(data)
        self.tab_gen.show_context(retrieved)
        self.tab_gen.show_quality(report)
        self.tab_gen.prog_lbl.config(text="✓  Done!")
        self.tab_gen._last_section = section
        # Update prompt version display in sidebar
        pv = report.get("prompt_version", "")
        if pv:
            self.sidebar.update_prompt_version(pv)
        self.refresh_stats()
        self.tab_out.refresh()
        self.status(
            f"Generated {SECTIONS[section]} · "
            f"Quality: {report.get('final_score', 0):.2f} · "
            f"{len(retrieved)} context docs"
        )

    def _gen_err(self, err):
        self.tab_gen.set_generating(False)
        self.tab_gen.output_text.config(state="normal")
        self.tab_gen.output_text.delete(1.0, tk.END)
        self.tab_gen.output_text.insert(tk.END, f"ERROR\n{'─'*40}\n{err}")
        self.tab_gen.output_text.config(state="disabled")
        self.tab_gen.prog_lbl.config(text="")
        self.status(f"Error: {err[:80]}")

    def save_to_kb(self):
        data = self.tab_gen._last_data
        section = self.tab_gen._last_section
        if data and section:
            embed_text = EmbeddingEngine.embed_text_for(data, section)
            data_type = EmbeddingEngine.infer_data_type(data, section)
            self.db.add_doc(section, data, embed_text,
                            source="generated", data_type=data_type)
            self.refresh_stats()
            self.tab_kb.refresh()
            self.tab_gen.prog_lbl.config(text="✓  Saved — re-index to activate for RAG")
            self.tab_gen.save_btn.config(state="disabled")

    def copy_output(self):
        self.tab_gen.output_text.config(state="normal")
        text = self.tab_gen.output_text.get(1.0, tk.END)
        self.tab_gen.output_text.config(state="disabled")
        self.clipboard_clear()
        self.clipboard_append(text)
        self.tab_gen.copy_btn.config(text="✓  Copied!")
        self.after(2000, lambda: self.tab_gen.copy_btn.config(text="⊞  Copy"))

    def do_import(self):
        path = filedialog.askopenfilename(
            title="Import UCAT Questions (JSON)",
            filetypes=[("JSON", "*.json"), ("All", "*.*")]
        )
        if not path:
            return
        try:
            n = self.db.import_json(path, embed_text_fn=EmbeddingEngine.embed_text_for)
            messagebox.showinfo("Import", f"Imported {n} document(s).\nNow click '⊛ Index Knowledge Base'.")
            self.refresh_stats()
            self.tab_kb.refresh()
        except Exception as e:
            messagebox.showerror("Import Error", str(e))

    def _ask_screenshot_section(self):
        """Pop up a dialog asking the user to pick ONE section for screenshot import."""
        dlg = tk.Toplevel(self)
        dlg.title("Select Question Section")
        dlg.geometry("420x300")
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.transient(self)

        result = {"section": None}

        tk.Label(dlg, text="Which section are these screenshots from?",
                 bg=BG, fg="#E6EDF3", font=("Courier New", 12, "bold"),
                 wraplength=380).pack(padx=20, pady=(24, 4))
        tk.Label(dlg, text="All screenshots in this batch must be the same\n"
                           "section type. Import each section separately.",
                 bg=BG, fg=MUTED, font=("Courier New", 9),
                 justify="center").pack(padx=20, pady=(0, 16))

        chosen = tk.StringVar(value="")
        for code, name in SECTIONS.items():
            tk.Radiobutton(
                dlg, text=f"  {code}  —  {name}",
                variable=chosen, value=code,
                bg=BG, fg="#E6EDF3", selectcolor=PANEL,
                activebackground=BG, activeforeground=ACCENT,
                font=("Courier New", 11), anchor="w",
                indicatoron=True, padx=8, pady=4
            ).pack(fill="x", padx=30)

        btn_row = tk.Frame(dlg, bg=BG)
        btn_row.pack(pady=(18, 12))

        def on_ok():
            if chosen.get():
                result["section"] = chosen.get()
                dlg.destroy()
            else:
                messagebox.showwarning("No Section Selected",
                                       "Please select a section before continuing.",
                                       parent=dlg)

        def on_cancel():
            dlg.destroy()

        mkbtn(btn_row, "Continue", on_ok, bg="#1F6FEB", padx=20, pady=7
              ).pack(side="left", padx=(0, 10))
        mkbtn(btn_row, "Cancel", on_cancel, bg=PANEL, fg=MUTED, padx=20, pady=7
              ).pack(side="left")

        self.wait_window(dlg)
        return result["section"]

    def do_screenshot_import(self):
        # Ask for section FIRST — before file picker
        section = self._ask_screenshot_section()
        if not section:
            return

        paths = filedialog.askopenfilenames(
            title=f"Import {SECTIONS[section]} Screenshots",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp"), ("All", "*.*")]
        )
        if not paths:
            return

        self.status(f"Importing {len(paths)} screenshot(s) for {SECTIONS[section]}...")

        def worker():
            result = self.ingester.ingest_batch(
                list(paths), section,
                on_progress=lambda msg: self.after(0, lambda m=msg: self.status(m))
            )
            self.after(0, lambda: (
                self.refresh_stats(),
                self.tab_kb.refresh(),
                messagebox.showinfo("Import",
                    f"Success: {len(result['success'])}\n"
                    f"Failed: {len(result['failed'])}\n\n"
                    + ("\n".join(f["error"][:80] for f in result["failed"][:5]) if result["failed"] else "")
                )
            ))

        threading.Thread(target=worker, daemon=True).start()

    def add_samples(self):
        samples = _load_samples()
        if not samples:
            messagebox.showinfo("Samples", "No sample_questions.json found.")
            return
        for q in samples:
            if isinstance(q, dict) and q.get("section") in SECTIONS:
                embed_text = EmbeddingEngine.embed_text_for(q, q["section"])
                data_type = EmbeddingEngine.infer_data_type(q, q["section"])
                self.db.add_doc(q["section"], q, embed_text,
                                source="sample", data_type=data_type)
        messagebox.showinfo("Samples",
            f"Added {len(samples)} sample documents.\n"
            "Now click '⊛ Index Knowledge Base' to embed them.")
        self.refresh_stats()
        self.tab_kb.refresh()

    def do_export(self):
        rows = self.db.get_generated(limit=10000)
        if not rows:
            messagebox.showinfo("Export", "No generated questions yet.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="ucat_generated.json"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump([r["data"] for r in rows], f, indent=2, ensure_ascii=False)
            # Record implicit export signal for all exported rows
            if self.feedback:
                for r in rows:
                    self.feedback.record_export(r["id"], r.get("context_ids", []))
            messagebox.showinfo("Exported", f"Saved {len(rows)} sets to:\n{path}")

    def show_prompt_ab(self):
        """Show prompt version A/B tracking — quality scores by prompt version."""
        versions = self.db.get_all_prompt_versions()
        top = tk.Toplevel(self)
        top.title("Prompt A/B Tracking")
        top.geometry("650x400")
        top.configure(bg=BG)
        from tkinter import scrolledtext as st
        text = st.ScrolledText(top, bg=PANEL, fg="#E6EDF3", font=FS,
                                relief="flat", padx=12, pady=10)
        text.pack(fill="both", expand=True, padx=10, pady=10)

        if not versions:
            text.insert(tk.END, "No prompt versions tracked yet.\n\n"
                        "Generate some questions first — each generation records\n"
                        "the prompt template hash for A/B comparison.")
        else:
            text.insert(tk.END, "═" * 55 + "\n")
            text.insert(tk.END, "  Prompt Version A/B Quality Tracking\n")
            text.insert(tk.END, "═" * 55 + "\n\n")
            text.insert(tk.END, f"{'Version':<14} {'Count':>6} {'Rule':>7} {'LLM':>7} {'Final':>7}  {'Period'}\n")
            text.insert(tk.END, "─" * 55 + "\n")
            for v in versions:
                ver = (v["version"] or "?")[:12]
                count = v["count"] or 0
                rule = f"{v['avg_rule']:.2f}" if v["avg_rule"] else "-"
                llm = f"{v['avg_llm']:.1f}" if v["avg_llm"] else "-"
                final = f"{v['avg_final']:.2f}" if v["avg_final"] else "-"
                first = (v["first_used"] or "")[:10]
                last = (v["last_used"] or "")[:10]
                period = f"{first} → {last}" if first != last else first
                text.insert(tk.END, f"{ver:<14} {count:>6} {rule:>7} {llm:>7} {final:>7}  {period}\n")
            text.insert(tk.END, "\n" + "─" * 55 + "\n")
            text.insert(tk.END, "Higher final scores = better prompt template.\n"
                        "If a new version regresses, revert the prompt change.")
        text.config(state="disabled")

    def show_coverage(self):
        summary = self.corpus.coverage_summary()
        top = tk.Toplevel(self)
        top.title("Corpus Coverage Analysis")
        top.geometry("600x500")
        top.configure(bg=BG)
        from tkinter import scrolledtext as st
        text = st.ScrolledText(top, bg=PANEL, fg="#E6EDF3", font=FS,
                                relief="flat", padx=12, pady=10)
        text.pack(fill="both", expand=True, padx=10, pady=10)
        text.insert(tk.END, summary)
        text.config(state="disabled")

    def _on_close(self):
        self.db.close()
        self.destroy()
