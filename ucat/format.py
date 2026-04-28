"""Plain-text formatting for generated question sets — used in the output panel
and copy-to-clipboard. Visuals are rendered separately by ``rendering``.
"""
from __future__ import annotations

import textwrap
from typing import Any, Dict

from .config import SECTIONS


def format_qset(data: Dict[str, Any]) -> str:
    section = data.get("section", "?")
    lines = ["═" * 64, f"  {SECTIONS.get(section, section)}  ·  Question Set", "═" * 64, ""]

    # VR passage.
    if data.get("passage"):
        lines += ["PASSAGE", "─" * 40]
        for para in str(data["passage"]).split("\n"):
            lines.append(textwrap.fill(para, 72) if para.strip() else "")
        lines.append("")

    # SJT scenario.
    if section == "SJT" and data.get("scenario"):
        lines += ["SCENARIO", "─" * 40]
        for para in str(data["scenario"]).split("\n"):
            lines.append(textwrap.fill(para, 72) if para.strip() else "")
        lines.append("")

    # QR stimulus — show summary text.
    if section == "QR" and isinstance(data.get("stimulus"), dict):
        stim = data["stimulus"]
        lines += ["DATA / STIMULUS", "─" * 40]
        lines.append(f"{stim.get('title','')}  ({stim.get('type','')})")
        if stim.get("units"):     lines.append(f"Units: {stim['units']}")
        if stim.get("categories"):
            lines.append("Categories: " + ", ".join(map(str, stim["categories"])))
        for s in stim.get("series", []) or []:
            lines.append(f"  {s.get('name','?')}: {s.get('values', [])}")
        if stim.get("rows"):
            for col, vals in stim["rows"].items():
                lines.append(f"  {col}: {vals}")
        if stim.get("note"):  lines.append(f"Note: {stim['note']}")
        lines.append("(See chart panel for visual)")
        lines.append("")
    elif data.get("stimulus"):  # legacy text stim
        lines += ["DATA / STIMULUS", "─" * 40, str(data["stimulus"]), ""]

    # AR sets.
    if section == "AR":
        if data.get("set_a_rule"):
            lines += ["SET A — RULE", "─" * 40,
                       textwrap.fill(str(data["set_a_rule"]), 72), ""]
        if data.get("set_b_rule"):
            lines += ["SET B — RULE", "─" * 40,
                       textwrap.fill(str(data["set_b_rule"]), 72), ""]
        if data.get("set_a_description"):    # legacy
            lines += ["SET A (legacy)", "─" * 40, str(data["set_a_description"]), ""]
        if data.get("set_b_description"):
            lines += ["SET B (legacy)", "─" * 40, str(data["set_b_description"]), ""]
        lines.append("(See panel images for shape sets)")
        lines.append("")

    # Questions.
    cal = data.get("calibrated_difficulty", {})
    per_q_cal = {c["number"]: c for c in (cal.get("per_question") or [])}

    for q in data.get("questions", []) or []:
        n = q.get("number", "?")
        d = per_q_cal.get(n, {}).get("calibrated") or q.get("difficulty")
        d_str = f"  [d={d:.1f}]" if isinstance(d, (int, float)) else ""
        lines.append(f"Q{n}.{d_str}  {q.get('text','')}")

        if q.get("type"):
            lines.append(f"       (type: {q['type']})")
        for k, v in (q.get("options") or {}).items():
            lines.append(f"       {k})  {v}")
        lines.append(f"       ✓  Answer: {q.get('answer','?')}")

        if q.get("explanation"):
            lines.append(textwrap.fill(
                q["explanation"], 64,
                initial_indent="       💡  ", subsequent_indent="           "
            ))
        lines.append("")

    # Difficulty / verdict / coverage summary.
    if cal:
        lines.append("─" * 40)
        lines.append(f"Calibrated set difficulty: {cal.get('set_difficulty','?')}  "
                      f"(min {cal.get('min','?')}, max {cal.get('max','?')})")

    return "\n".join(lines)
