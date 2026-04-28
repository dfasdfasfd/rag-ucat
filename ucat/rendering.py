"""Visual rendering for QR charts, AR shape panels, and DM Venn diagrams.

All renderers return PIL Images. The Tkinter UI converts them to PhotoImage
objects for display. Headless callers can save them to disk via `.save(path)`.

Design notes:
  * matplotlib is the workhorse — same backend for charts, shapes, and venns.
  * Renderers are pure functions of their spec; no DB, no LLM. Easy to test.
  * AR panels use matplotlib.patches with a grid layout; rotation is honoured.
  * Falls back gracefully if matplotlib isn't installed (returns a placeholder).
"""
from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")  # headless-safe; tk integration overrides if needed
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp_patches
    from matplotlib.transforms import Affine2D
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from matplotlib_venn import venn2, venn3
    _HAS_VENN = True
except ImportError:
    _HAS_VENN = False

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# ─── Theme — match the dark UI ────────────────────────────────────────────────

_BG       = "#161B22"   # PANEL
_PANEL2   = "#1C2128"
_GRID     = "#30363D"
_TEXT     = "#E6EDF3"
_MUTED    = "#7D8590"
_PALETTE  = ["#58A6FF", "#3FB950", "#E8943A", "#A78BFA", "#F778BA",
              "#56D4DD", "#D29922", "#F85149"]

_AR_FILL = {"black": "#222222", "white": "#FFFFFF", "grey": "#9CA3AF"}
_AR_EDGE = "#000000"


def _placeholder(text: str, w: int = 400, h: int = 200):
    """Fallback when matplotlib/PIL aren't available."""
    if _HAS_PIL:
        img = Image.new("RGB", (w, h), _BG)
        draw = ImageDraw.Draw(img)
        draw.text((10, h // 2 - 8), text, fill=_TEXT)
        return img
    return None


def _fig_to_pil(fig, dpi: int = 100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    if _HAS_PIL:
        return Image.open(buf).convert("RGBA")
    return buf  # raw bytesio if PIL is missing


# ═══ QR CHARTS ════════════════════════════════════════════════════════════════

def render_qr_chart(spec: Dict[str, Any], width: int = 720, height: int = 420):
    """Render a QR data-stimulus chart from a structured spec.

    Spec shape mirrors `models.QRChart`:
      {type, title, x_label, y_label, categories, series[], rows{}, units, note}
    """
    if not _HAS_MPL:
        return _placeholder(f"[QR chart unavailable — install matplotlib]")

    plt.rcdefaults()
    fig = plt.figure(figsize=(width / 100, height / 100), facecolor=_BG)
    ax = fig.add_subplot(111, facecolor=_PANEL2)

    chart_type = spec.get("type", "bar")
    title  = spec.get("title", "")
    x_label = spec.get("x_label", "")
    y_label = spec.get("y_label", "")
    categories = list(spec.get("categories") or [])
    series = list(spec.get("series") or [])
    units  = spec.get("units")

    try:
        if chart_type == "table":
            _render_table(ax, spec)
        elif chart_type == "pie":
            _render_pie(ax, categories, series)
        elif chart_type == "line":
            _render_line(ax, categories, series, x_label, y_label, units)
        elif chart_type == "stacked_bar":
            _render_stacked_bar(ax, categories, series, x_label, y_label, units)
        else:  # bar (default)
            _render_bar(ax, categories, series, x_label, y_label, units)
    except Exception as e:
        ax.clear()
        ax.text(0.5, 0.5, f"render error: {e}",
                ha="center", va="center", color=_MUTED, fontsize=10)
        ax.axis("off")

    if chart_type != "table":
        ax.set_title(title, color=_TEXT, fontsize=13, pad=12, weight="bold")
        for spine in ax.spines.values():
            spine.set_color(_GRID)
        ax.tick_params(colors=_MUTED)
        if hasattr(ax, "yaxis"):
            ax.yaxis.label.set_color(_TEXT)
            ax.xaxis.label.set_color(_TEXT)

    note = spec.get("note")
    if note:
        fig.text(0.5, 0.02, note, ha="center", color=_MUTED, fontsize=9,
                  style="italic")

    return _fig_to_pil(fig)


def _bar_positions(n_cats: int, n_series: int) -> Tuple[Any, float]:
    import numpy as _np
    width = 0.8 / max(n_series, 1)
    x = _np.arange(n_cats)
    return x, width


def _render_bar(ax, categories, series, x_label, y_label, units):
    x, width = _bar_positions(len(categories), len(series))
    for i, s in enumerate(series):
        offsets = x + (i - (len(series) - 1) / 2) * width
        ax.bar(offsets, s.get("values", []),
               width=width, label=s.get("name", ""),
               color=_PALETTE[i % len(_PALETTE)],
               edgecolor=_GRID)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, color=_TEXT, fontsize=10)
    if x_label: ax.set_xlabel(x_label)
    if y_label: ax.set_ylabel(y_label + (f" ({units})" if units else ""))
    if len(series) > 1:
        ax.legend(facecolor=_PANEL2, edgecolor=_GRID, labelcolor=_TEXT, fontsize=9)
    ax.grid(True, axis="y", color=_GRID, alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)


def _render_stacked_bar(ax, categories, series, x_label, y_label, units):
    import numpy as _np
    x = _np.arange(len(categories))
    bottom = _np.zeros(len(categories))
    for i, s in enumerate(series):
        vals = _np.asarray(s.get("values", []), dtype=float)
        if len(vals) != len(categories): continue
        ax.bar(x, vals, bottom=bottom, label=s.get("name", ""),
               color=_PALETTE[i % len(_PALETTE)], edgecolor=_GRID)
        bottom = bottom + vals
    ax.set_xticks(x)
    ax.set_xticklabels(categories, color=_TEXT, fontsize=10)
    if x_label: ax.set_xlabel(x_label)
    if y_label: ax.set_ylabel(y_label + (f" ({units})" if units else ""))
    ax.legend(facecolor=_PANEL2, edgecolor=_GRID, labelcolor=_TEXT, fontsize=9)
    ax.grid(True, axis="y", color=_GRID, alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)


def _render_line(ax, categories, series, x_label, y_label, units):
    for i, s in enumerate(series):
        ax.plot(categories, s.get("values", []),
                marker="o", linewidth=2,
                color=_PALETTE[i % len(_PALETTE)],
                label=s.get("name", ""))
    if x_label: ax.set_xlabel(x_label)
    if y_label: ax.set_ylabel(y_label + (f" ({units})" if units else ""))
    if len(series) > 1:
        ax.legend(facecolor=_PANEL2, edgecolor=_GRID, labelcolor=_TEXT, fontsize=9)
    ax.grid(True, color=_GRID, alpha=0.4, linestyle="--")
    ax.set_axisbelow(True)
    for label in ax.get_xticklabels():
        label.set_color(_TEXT)


def _render_pie(ax, categories, series):
    if not series: return
    s0 = series[0]
    values = s0.get("values", [])
    ax.pie(values, labels=categories, autopct="%1.1f%%",
           colors=_PALETTE[:len(values)],
           textprops={"color": _TEXT, "fontsize": 10},
           wedgeprops={"edgecolor": _GRID, "linewidth": 1})
    ax.set_aspect("equal")


def _render_table(ax, spec):
    ax.axis("off")
    rows = spec.get("rows") or {}
    categories = list(spec.get("categories") or [])
    title = spec.get("title", "")

    if not rows:
        ax.text(0.5, 0.5, "(no table data)", ha="center", va="center", color=_MUTED)
        return

    col_labels = ["" ] + categories  # leftmost col is row name
    table_data: List[List[str]] = []
    for row_name, vals in rows.items():
        formatted = [str(v) for v in vals]
        # pad if needed
        while len(formatted) < len(categories):
            formatted.append("")
        table_data.append([row_name, *formatted])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(_GRID)
        if r == 0:    # header
            cell.set_facecolor("#1F6FEB")
            cell.set_text_props(color="white", weight="bold")
        elif c == 0:  # leftmost label col
            cell.set_facecolor("#1F2530")
            cell.set_text_props(color=_TEXT, weight="bold")
        else:
            cell.set_facecolor(_PANEL2)
            cell.set_text_props(color=_TEXT)
    if title:
        ax.set_title(title, color=_TEXT, fontsize=13, pad=12, weight="bold")


# ═══ AR SHAPE PANELS ══════════════════════════════════════════════════════════

def render_ar_panel(panel_spec: Dict[str, Any], cell_size: int = 110):
    """Render a single AR panel (a list of shapes) as a PNG image."""
    if not _HAS_MPL:
        return _placeholder("[AR panel unavailable]", 200, 200)

    shapes = panel_spec.get("shapes") or []
    if not shapes:
        return _placeholder("(empty)", cell_size, cell_size)

    # Lay out in a grid that's roughly square.
    cols = max(1, int(math.ceil(math.sqrt(len(shapes)))))
    rows = max(1, int(math.ceil(len(shapes) / cols)))

    fig_w = cols * cell_size / 100
    fig_h = rows * cell_size / 100 + 0.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=_BG)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows + 0.15)
    ax.set_aspect("equal")
    ax.set_facecolor(_PANEL2)
    ax.axis("off")

    for idx, sh in enumerate(shapes):
        r = idx // cols
        c = idx % cols
        cx = c + 0.5
        cy = (rows - 1 - r) + 0.5
        _draw_shape(ax, sh, cx, cy)

    # Subtle border around panel.
    border = mp_patches.Rectangle((0.02, 0.02), cols - 0.04, rows - 0.04,
                                    linewidth=1, edgecolor=_GRID,
                                    facecolor="none")
    ax.add_patch(border)

    label = panel_spec.get("label")
    if label:
        ax.text(cols / 2, rows + 0.05, label,
                ha="center", va="bottom", color=_TEXT, fontsize=9, weight="bold")

    return _fig_to_pil(fig, dpi=100)


def render_ar_set(panels: Sequence[Dict[str, Any]], title: str = "",
                   cell_size: int = 100, panels_per_row: int = 3):
    """Render a 6-panel AR set as a single composite image."""
    if not _HAS_MPL:
        return _placeholder("[AR set unavailable]", 600, 400)
    n = len(panels)
    rows = max(1, int(math.ceil(n / panels_per_row)))

    fig, axes = plt.subplots(rows, panels_per_row,
                              figsize=(panels_per_row * cell_size / 80,
                                       rows * cell_size / 80 + 0.3),
                              facecolor=_BG)
    if rows == 1 and panels_per_row == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(axes.flatten()) if hasattr(axes, "flatten") else list(axes)

    for ax in axes_flat:
        ax.axis("off"); ax.set_facecolor(_PANEL2)

    for i, panel in enumerate(panels):
        if i >= len(axes_flat): break
        ax = axes_flat[i]
        ax.set_xlim(0, 4); ax.set_ylim(0, 4); ax.set_aspect("equal")
        # Panel background.
        ax.add_patch(mp_patches.Rectangle((0.1, 0.1), 3.8, 3.8,
                                            facecolor=_PANEL2,
                                            edgecolor=_GRID, linewidth=1.2))
        shapes = panel.get("shapes") or []
        # Inner grid layout for shapes.
        cols = max(1, int(math.ceil(math.sqrt(len(shapes)))))
        rs   = max(1, int(math.ceil(len(shapes) / cols)))
        for idx, sh in enumerate(shapes):
            r = idx // cols
            c = idx % cols
            # Pad inside the panel border.
            cx = 0.4 + (c + 0.5) * (3.2 / cols)
            cy = 3.6 - (r + 0.5) * (3.2 / rs)
            _draw_shape(ax, sh, cx, cy, default_radius=min(1.4 / cols, 1.4 / rs))
        ax.text(2, -0.1, panel.get("label", f"Panel {i + 1}"),
                ha="center", va="top", color=_MUTED, fontsize=8)

    if title:
        fig.suptitle(title, color=_TEXT, fontsize=12, weight="bold", y=0.99)

    fig.subplots_adjust(wspace=0.05, hspace=0.20, top=0.92, bottom=0.05)
    return _fig_to_pil(fig, dpi=110)


def _draw_shape(ax, sh: Dict[str, Any], cx: float, cy: float,
                  default_radius: float = 0.35):
    kind  = sh.get("kind", "circle")
    color = sh.get("color", "black")
    size  = sh.get("size", "medium")
    rot   = sh.get("rotation_deg", 0) or 0

    fill = _AR_FILL.get(color, "#222222")
    edge = _AR_EDGE
    rmul = {"small": 0.7, "medium": 1.0, "large": 1.3}.get(size, 1.0)
    r = default_radius * rmul

    transform = Affine2D().rotate_deg_around(cx, cy, rot) + ax.transData

    if kind == "square":
        patch = mp_patches.Rectangle((cx - r, cy - r), 2 * r, 2 * r,
                                       facecolor=fill, edgecolor=edge,
                                       linewidth=1.4)
    elif kind == "diamond":
        verts = [(cx, cy + r), (cx + r, cy), (cx, cy - r), (cx - r, cy)]
        patch = mp_patches.Polygon(verts, facecolor=fill, edgecolor=edge, linewidth=1.4)
    elif kind == "triangle":
        verts = [(cx, cy + r),
                 (cx - r * math.sin(math.radians(60)), cy - r * math.cos(math.radians(60))),
                 (cx + r * math.sin(math.radians(60)), cy - r * math.cos(math.radians(60)))]
        patch = mp_patches.Polygon(verts, facecolor=fill, edgecolor=edge, linewidth=1.4)
    elif kind == "star":
        patch = _regular_star(cx, cy, r, 5, fill, edge)
    elif kind == "pentagon":
        patch = mp_patches.RegularPolygon((cx, cy), 5, radius=r,
                                            facecolor=fill, edgecolor=edge, linewidth=1.4)
    elif kind == "hexagon":
        patch = mp_patches.RegularPolygon((cx, cy), 6, radius=r,
                                            facecolor=fill, edgecolor=edge, linewidth=1.4)
    elif kind == "cross":
        # Plus sign — two overlapping rectangles.
        h = mp_patches.Rectangle((cx - r, cy - r * 0.3), 2 * r, r * 0.6,
                                   facecolor=fill, edgecolor=edge, linewidth=1.4)
        v = mp_patches.Rectangle((cx - r * 0.3, cy - r), r * 0.6, 2 * r,
                                   facecolor=fill, edgecolor=edge, linewidth=1.4)
        for p in (h, v):
            p.set_transform(transform)
            ax.add_patch(p)
        return
    elif kind == "arrow":
        arrow = mp_patches.FancyArrow(cx - r, cy, 2 * r, 0,
                                        width=r * 0.4, head_width=r * 0.9,
                                        head_length=r * 0.7,
                                        facecolor=fill, edgecolor=edge,
                                        length_includes_head=True)
        arrow.set_transform(transform)
        ax.add_patch(arrow)
        return
    else:  # circle (default)
        patch = mp_patches.Circle((cx, cy), r,
                                    facecolor=fill, edgecolor=edge, linewidth=1.4)
    patch.set_transform(transform)
    ax.add_patch(patch)


def _regular_star(cx: float, cy: float, r: float, points: int,
                   fill: str, edge: str):
    verts = []
    for i in range(points * 2):
        ang = math.pi / 2 - i * math.pi / points
        rad = r if i % 2 == 0 else r * 0.45
        verts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    return mp_patches.Polygon(verts, facecolor=fill, edgecolor=edge, linewidth=1.4)


# ═══ DM VENN DIAGRAMS ═════════════════════════════════════════════════════════

def render_dm_venn(spec: Dict[str, Any], width: int = 480, height: int = 360):
    """Render a 2- or 3-circle Venn from a structured spec.

    Spec shape:
      {"sets": [{"label": ..., "members": [...]}, ...], "universe_label": ...}
    """
    if not _HAS_MPL:
        return _placeholder("[Venn unavailable — install matplotlib]", width, height)

    sets = spec.get("sets") or []
    if not sets:
        return _placeholder("(no venn sets)", width, height)

    fig = plt.figure(figsize=(width / 100, height / 100), facecolor=_BG)
    ax  = fig.add_subplot(111, facecolor=_PANEL2)

    set_objs = [set(s.get("members") or []) for s in sets]
    labels   = [s.get("label", "") for s in sets]

    if len(set_objs) == 2 and _HAS_VENN:
        v = venn2(subsets=set_objs, set_labels=labels, ax=ax)
    elif len(set_objs) == 3 and _HAS_VENN:
        v = venn3(subsets=set_objs, set_labels=labels, ax=ax)
    else:
        # Fallback: draw plain circles by hand.
        _venn_fallback(ax, set_objs, labels)
        v = None

    # Theme tweaks — recolour text + patches for the dark theme.
    if v is not None:
        for i, _ in enumerate(set_objs):
            patch = v.get_patch_by_id("A" if i == 0 else "B" if i == 1 else "C")
            if patch is not None:
                patch.set_color(_PALETTE[i % len(_PALETTE)])
                patch.set_alpha(0.55)
                patch.set_edgecolor("#FFFFFF")
        for label in v.set_labels or []:
            if label is not None:
                label.set_color(_TEXT)
                label.set_fontsize(11)
                label.set_fontweight("bold")
        for label in v.subset_labels or []:
            if label is not None:
                label.set_color("#FFFFFF")
                label.set_fontsize(10)

    universe = spec.get("universe_label")
    if universe:
        ax.set_title(universe, color=_TEXT, fontsize=12, weight="bold", pad=10)

    ax.axis("off")
    return _fig_to_pil(fig)


def _venn_fallback(ax, set_objs: List[set], labels: List[str]):
    """Draw a basic 2-set Venn when matplotlib_venn isn't installed."""
    n = len(set_objs)
    if n == 2:
        positions = [(-0.5, 0), (0.5, 0)]
    elif n == 3:
        positions = [(-0.6, 0.4), (0.6, 0.4), (0.0, -0.5)]
    else:
        positions = [(0, 0)]
    for i, (cx, cy) in enumerate(positions):
        circle = mp_patches.Circle((cx, cy), 1.0,
                                     facecolor=_PALETTE[i % len(_PALETTE)],
                                     edgecolor="#FFFFFF", alpha=0.55)
        ax.add_patch(circle)
        ax.text(cx + (1.0 if cx >= 0 else -1.0), cy + 1.05,
                labels[i] if i < len(labels) else f"Set {i+1}",
                color=_TEXT, fontsize=11, weight="bold",
                ha="left" if cx >= 0 else "right")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")


# ═══ Convenience: best-image-for-set ══════════════════════════════════════════

def render_visuals_for(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a generated set, return a dict of section-appropriate rendered images.

    Keys:
      QR → {"chart": Image}
      AR → {"set_a": Image, "set_b": Image, "tests": [Image, ...]}
      DM → {"venns": {q_number: Image, ...}}
      VR → {} (no visuals)
    """
    section = data.get("section")
    out: Dict[str, Any] = {}

    if section == "QR" and isinstance(data.get("stimulus"), dict):
        out["chart"] = render_qr_chart(data["stimulus"])

    elif section == "AR":
        if data.get("set_a_panels"):
            out["set_a"] = render_ar_set(data["set_a_panels"], title="Set A")
        if data.get("set_b_panels"):
            out["set_b"] = render_ar_set(data["set_b_panels"], title="Set B")
        tests = data.get("test_panels") or []
        if tests:
            out["tests"] = [render_ar_panel(p, cell_size=140) for p in tests]

    elif section == "DM":
        venns: Dict[int, Any] = {}
        for q in data.get("questions") or []:
            v = q.get("venn")
            if isinstance(v, dict) and v.get("sets"):
                venns[q.get("number", -1)] = render_dm_venn(v)
        if venns:
            out["venns"] = venns

    return out
