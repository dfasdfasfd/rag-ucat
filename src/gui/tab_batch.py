"""
Batch Generation tab: count, difficulty distribution, format selection,
background progress bar, failure review. Only active for calibrated sections.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading

from src.config import SECTIONS, HAS_REPORTLAB, CALIBRATION_THRESHOLD
from src.gui.theme import (
    BG, PANEL, PANEL2, BORDER, TEXT, MUTED, ACCENT, SUCCESS, WARN, DANGER,
    FM, FB, FS, FSB, FT, FH, mkbtn,
)


class TabBatch:
    """Batch generation tab, unlocked per-section after calibration."""

    def __init__(self, parent, app):
        self.app = app
        self.frame = tk.Frame(parent, bg=BG)
        self._running = False
        self._build()

    def _build(self):
        p = tk.Frame(self.frame, bg=BG)
        p.pack(fill="both", expand=True, padx=36, pady=24)

        tk.Label(p, text="Batch Generation", bg=BG, fg=TEXT,
                 font=FT).pack(anchor="w")
        tk.Label(p, text="Generate multiple question sets after calibration.",
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
                command=self._check_calibration
            ).pack(side="left", padx=4)

        # Calibration status
        self.cal_status = tk.Label(p, text="", bg=BG, fg=MUTED, font=FM)
        self.cal_status.pack(anchor="w", pady=(0, 12))

        # Count
        cr = tk.Frame(p, bg=BG)
        cr.pack(anchor="w", pady=(0, 8))
        tk.Label(cr, text="Count:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self.count_var = tk.IntVar(value=10)
        tk.Spinbox(cr, from_=1, to=200, textvariable=self.count_var,
                   width=6, font=FM, bg=PANEL2, fg=TEXT).pack(side="left")
        tk.Label(cr, text="question sets", bg=BG, fg=MUTED, font=FS).pack(side="left", padx=10)

        # Difficulty distribution
        dr = tk.Frame(p, bg=BG)
        dr.pack(anchor="w", pady=(0, 8))
        tk.Label(dr, text="Difficulty:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))

        self.easy_var = tk.IntVar(value=30)
        self.med_var = tk.IntVar(value=40)
        self.hard_var = tk.IntVar(value=30)

        for label, var in [("Easy %", self.easy_var), ("Med %", self.med_var), ("Hard %", self.hard_var)]:
            tk.Label(dr, text=label, bg=BG, fg=MUTED, font=FS).pack(side="left", padx=(10, 2))
            tk.Spinbox(dr, from_=0, to=100, textvariable=var, width=4,
                       font=FS, bg=PANEL2, fg=TEXT).pack(side="left")

        # Format
        fr = tk.Frame(p, bg=BG)
        fr.pack(anchor="w", pady=(0, 12))
        tk.Label(fr, text="Format:", bg=BG, fg=MUTED, font=FM).pack(side="left", padx=(0, 14))
        self.format_var = tk.StringVar(value="json")
        formats = [("JSON", "json"), ("CSV", "csv")]
        if HAS_REPORTLAB:
            formats.append(("PDF", "pdf"))
        else:
            formats.append(("PDF (install reportlab)", "pdf_disabled"))

        for label, val in formats:
            state = "disabled" if val == "pdf_disabled" else "normal"
            actual_val = "pdf" if val == "pdf_disabled" else val
            rb = tk.Radiobutton(
                fr, text=label, variable=self.format_var, value=actual_val,
                bg=BG, fg=TEXT if state == "normal" else MUTED,
                selectcolor=PANEL, activebackground=BG, font=FS,
                state=state
            )
            rb.pack(side="left", padx=6)

        # Generate button
        btn_row = tk.Frame(p, bg=BG)
        btn_row.pack(anchor="w", pady=(0, 10))
        self.gen_btn = mkbtn(btn_row, "⚡  Start Batch Generation",
                              self._start_batch, padx=20, pady=10)
        self.gen_btn.pack(side="left")
        self.prog_lbl = tk.Label(btn_row, text="", bg=BG, fg=MUTED, font=FM)
        self.prog_lbl.pack(side="left", padx=16)

        # Progress bar
        self.progress = ttk.Progressbar(p, length=400, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 12))

        # Results
        self.result_text = tk.Label(
            p, text="", bg=BG, fg=MUTED, font=FM,
            anchor="w", justify="left", wraplength=600
        )
        self.result_text.pack(anchor="w", pady=(0, 8))

        # Failure list
        tk.Label(p, text="FAILURES (for manual review)", bg=BG, fg=MUTED,
                 font=FH).pack(anchor="w", pady=(8, 4))
        self.fail_list = tk.Listbox(
            p, bg=PANEL, fg=DANGER, font=FS, height=6,
            selectbackground="#1F6FEB"
        )
        self.fail_list.pack(fill="x", pady=(0, 8))

        # Save button
        self.save_btn = mkbtn(p, "↓  Save Output", self._save_output,
                               bg=PANEL, fg=ACCENT, font=FM, pady=6, state="disabled")
        self.save_btn.pack(anchor="w")

        self._last_output = None
        self._last_format = None
        self._check_calibration()

    def _check_calibration(self):
        section = self.section_var.get()
        is_cal = self.app.calibration.is_batch_unlocked(section)
        if is_cal:
            self.cal_status.config(
                text=f"✓ {SECTIONS[section]} is calibrated — batch mode available",
                fg=SUCCESS
            )
            self.gen_btn.config(state="normal")
        else:
            state = self.app.calibration.get_state(section)
            approvals = state.get("consecutive_approvals", 0)
            self.cal_status.config(
                text=f"✕ {SECTIONS[section]} needs calibration "
                     f"({approvals}/{CALIBRATION_THRESHOLD} approvals)",
                fg=DANGER
            )
            self.gen_btn.config(state="disabled")

    def _start_batch(self):
        if self._running:
            return

        section = self.section_var.get()
        count = self.count_var.get()
        fmt = self.format_var.get()

        # Validate percentages
        total_pct = self.easy_var.get() + self.med_var.get() + self.hard_var.get()
        if total_pct != 100:
            messagebox.showwarning("Invalid", f"Difficulty percentages must sum to 100 (got {total_pct})")
            return

        difficulty_dist = {
            "easy": self.easy_var.get() / 100.0,
            "medium": self.med_var.get() / 100.0,
            "hard": self.hard_var.get() / 100.0,
        }

        self._running = True
        self.gen_btn.config(state="disabled", text="Running...")
        self.progress["maximum"] = count
        self.progress["value"] = 0
        self.fail_list.delete(0, tk.END)
        self.result_text.config(text="Starting batch generation...")

        def worker():
            try:
                def on_progress(successful, total, failed):
                    self.frame.after(0, lambda s=successful, t=total, f=failed: (
                        self.progress.configure(value=s + f),
                        self.prog_lbl.config(text=f"{s}/{t} done ({f} failed)"),
                    ))

                result = self.app.batch_gen.generate_batch(
                    section, count, difficulty_dist, fmt,
                    on_progress=on_progress
                )
                self.frame.after(0, lambda r=result: self._show_results(r))
            except Exception as e:
                self.frame.after(0, lambda err=str(e): self._show_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _show_results(self, result):
        self._running = False
        self.gen_btn.config(state="normal", text="⚡  Start Batch Generation")
        self._last_output = result.get("output")
        self._last_format = result.get("format", "json")

        self.result_text.config(
            text=f"Done! {result['successful']} successful, {result['failed']} failed "
                 f"out of {result['total_attempted']} attempted.",
            fg=SUCCESS if result["failed"] == 0 else WARN
        )

        # Show failures
        for f in result.get("failures", []):
            reason = f.get("reason", "Unknown")
            self.fail_list.insert(tk.END, f"  {reason}")

        if self._last_output:
            self.save_btn.config(state="normal")

        self.app.refresh_stats()
        self.app.tab_kb.refresh()

    def _show_error(self, err):
        self._running = False
        self.gen_btn.config(state="normal", text="⚡  Start Batch Generation")
        self.result_text.config(text=f"Error: {err}", fg=DANGER)

    def _save_output(self):
        if not self._last_output:
            return
        ext = {"json": ".json", "csv": ".csv", "pdf": ".pdf"}.get(self._last_format, ".json")
        path = filedialog.asksaveasfilename(
            defaultextension=ext,
            filetypes=[(ext.upper().strip("."), f"*{ext}")],
            initialfile=f"ucat_batch{ext}"
        )
        if path:
            mode = "wb" if self._last_format == "pdf" else "w"
            with open(path, mode) as f:
                f.write(self._last_output)
            messagebox.showinfo("Saved", f"Batch output saved to:\n{path}")
