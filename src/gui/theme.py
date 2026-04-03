"""
GUI theme: colors, fonts, and widget factory methods.
Dark GitHub-inspired theme.
"""

import tkinter as tk
from tkinter import ttk

# ─── Colors ──────────────────────────────────────────────────────────────────

BG      = "#0C1117"
PANEL   = "#161B22"
PANEL2  = "#1C2128"
BORDER  = "#30363D"
ACCENT  = "#58A6FF"
TEXT    = "#E6EDF3"
MUTED   = "#7D8590"
SUCCESS = "#3FB950"
DANGER  = "#F85149"
WARN    = "#D29922"

# ─── Fonts ───────────────────────────────────────────────────────────────────

FM  = ("Courier New", 11)
FB  = ("Courier New", 11, "bold")
FS  = ("Courier New", 9)
FSB = ("Courier New", 9, "bold")
FT  = ("Courier New", 15, "bold")
FH  = ("Courier New", 10, "bold")

# ─── Widget Factories ────────────────────────────────────────────────────────

def mkbtn(parent, text, cmd, fg="white", bg=ACCENT, font=None, **kw):
    """Create a styled flat button."""
    return tk.Button(
        parent, text=text, command=cmd, bg=bg, fg=fg,
        font=font or FB, relief="flat", cursor="hand2",
        activebackground=bg, activeforeground=fg, **kw
    )


def mklabel(parent, text, fg=TEXT, font=FM, **kw):
    """Create a styled label."""
    return tk.Label(parent, text=text, bg=kw.pop("bg", BG), fg=fg, font=font, **kw)


def mkentry(parent, textvariable, width=34, **kw):
    """Create a styled entry."""
    return tk.Entry(
        parent, textvariable=textvariable, bg=PANEL2, fg=TEXT,
        font=FM, insertbackground=ACCENT, relief="flat", width=width, **kw
    )


def apply_theme(root):
    """Apply the dark theme to ttk widgets."""
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure("TFrame", background=BG)
    s.configure("TLabel", background=BG, foreground=TEXT, font=FM)
    s.configure("TNotebook", background=BG, borderwidth=0)
    s.configure("TNotebook.Tab", background=PANEL, foreground=MUTED,
                font=FM, padding=[18, 9])
    s.map("TNotebook.Tab",
          background=[("selected", BG)],
          foreground=[("selected", ACCENT)])
    s.configure("Treeview", background=PANEL, foreground=TEXT,
                fieldbackground=PANEL, font=FS, rowheight=28, borderwidth=0)
    s.configure("Treeview.Heading", background=BG, foreground=MUTED, font=FH)
    s.map("Treeview",
          background=[("selected", "#1F6FEB")],
          foreground=[("selected", "white")])
    s.configure("TScrollbar", background=PANEL, troughcolor=BG, borderwidth=0)
    s.configure("TCombobox", font=FM)
    s.configure("TSeparator", background=BORDER)

    # Quality score colors (tags for treeview)
    s.configure("good.TLabel", foreground=SUCCESS)
    s.configure("mid.TLabel", foreground=WARN)
    s.configure("bad.TLabel", foreground=DANGER)
