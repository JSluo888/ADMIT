"""Shared style, data access and output helpers for the ADMIT data figures.

Every number drawn in a figure is read from a JSON under $ADMIT_RUNS and registered with
`Sources.add(...)`, which writes $ADMIT_FIG_OUT/FIGURE_SOURCES.tsv.

Paths (environment variables):
  ADMIT_RUNS               experiment outputs (default <repo>/runs)
  ADMIT_FIG_OUT            figure outputs (default <repo>/figures/out)
  ADMIT_FIG_ASSETS         tissue case-study assets (in-house; default <repo>/figures/assets):
                           orion_case/, piyadasa_case/, cords_case/
  ADMIT_ORION_CASE_REGION  sub-directory of orion_case/ holding the pre-specified Orion region
  ADMIT_ORION_CASE_LABEL   how panel titles name that region

Usage from a figure script:
    import admit_style as S
    S.setup()
    fig = S.figure(height_in)
    ax = S.ax_in(fig, x, y, w, h)        # inches from the top-left corner
    ...
    S.finish(fig, "fig2", "fig2_forced_labels", sources)
"""
from __future__ import annotations

import csv
import functools
import itertools
import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]                                   # repository root
RUNS = Path(os.environ.get("ADMIT_RUNS", str(ROOT / "runs")))
FIGDIR = Path(os.environ.get("ADMIT_FIG_OUT", str(ROOT / "figures" / "out")))     # outputs: <FIGDIR>/<fig_id>/
ASSETS = Path(os.environ.get("ADMIT_FIG_ASSETS", str(ROOT / "figures" / "assets")))
ORION_CASE_REGION = os.environ.get("ADMIT_ORION_CASE_REGION", "region")
ORION_CASE_LABEL = os.environ.get("ADMIT_ORION_CASE_LABEL", "Orion, held-out region")
SOURCES_TSV = FIGDIR / "FIGURE_SOURCES.tsv"

# ---------------------------------------------------------------- size and type
FULL_W = 6.5          # in, US Letter with 1-inch margins
HALF_W = 3.2
FS = 7.0              # smallest text allowed (pt)
FS_BODY = 7.0
FS_TITLE = 8.0
FS_LETTER = 10.0
MIN_PT = 7.0

# ---------------------------------------------------------------- ink (dataviz reference chrome)
INK = "#0b0b0b"       # primary text
INK2 = "#52514e"      # secondary text
MUTED = "#898781"     # muted text / axis labels of lesser rank
GRID = "#e1e0d9"      # hairline gridlines
AXIS = "#c3c2b7"      # baseline / axis
PLANE = "#f3f2ee"     # placeholder plane
NEUTRAL = "#f0efec"   # diverging midpoint

# ---------------------------------------------------------------- method colours (fixed, never cycle)
BLUE = "#2a78d6"        # ADMIT (final, with M1)
# #7fb0ee: the lightest blue that passes the OKLCH lightness/chroma checks next to ADMIT, aqua and violet
LIGHT_BLUE = "#7fb0ee"  # ADMIT without M1 (ablation only)
AQUA = "#1baf7a"        # marginal projected LAC
ORANGE = "#eb6834"      # full-panel calibration / old method
VIOLET = "#6250d6"      # flat LAC/APS/RAPS + filter
GREY = "#8a8984"        # hierarchical baselines (HCC, scConform, CRSVP); de-emphasis tone
RED = "#e34948"         # "unsupported / inadmissible" ONLY
RED_FILL = "#fdeded"    # light red plane, same meaning (matches Fig 1)
RED_FILL2 = "#f6b8b7"   # stronger red plane, same meaning (no admissible ancestor)

# sequential blue ramp (dataviz reference)
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

ARMS = {
    # arm key in the run summaries -> (label, colour, marker)
    "admit_m1": ("ADMIT", BLUE, "o"),
    "admit": ("ADMIT without M1", LIGHT_BLUE, "o"),
    "lac_node_proj": ("Marginal projected LAC", AQUA, "D"),
    "patch_src_backstop": ("Full-panel calibration", ORANGE, "s"),
    "patch_src": ("Full-panel calibration (old, no backstop)", ORANGE, "s"),
    "lac_filt": ("LAC + filter", VIOLET, "^"),
    "aps_filt": ("APS + filter", VIOLET, "v"),
    "raps_filt": ("RAPS + filter", VIOLET, "<"),
}
MARKER_SIZE = 3.2      # pt, data dots
LW_DATA = 1.0
LW_CI = 0.6
LW_AXIS = 0.5
LW_GRID = 0.4

# ---------------------------------------------------------------- cell types
CT_LABEL = {
    "T_Lineage": "T cell", "Cytotoxic_T_Cell": "Cytotoxic T", "Helper_T_Cell": "Helper T",
    "Regulatory_T_Cell": "Regulatory T", "NK_Cell": "NK", "B_Cell": "B cell",
    "Plasma_Cell": "Plasma", "Myeloid": "Myeloid", "Macrophage_CD163pos": "Macrophage CD163+",
    "Macrophage_CD163neg": "Macrophage CD163−", "Dendritic_Cell": "Dendritic",
    "Neutrophil": "Neutrophil", "Immune_Other": "Immune, other", "Immune": "Immune",
    "Epithelial": "Epithelial", "Endothelial": "Endothelial", "Stromal": "Stromal",
}
CT_SHORT = {  # for inside heatmap cells
    "T_Lineage": "T", "Immune": "Imm", "Myeloid": "Mye", None: "∅",
}
# lineage order: T, NK, B/plasma, myeloid, other immune, epithelial, endothelial, stromal
CT_ORDER = ["T_Lineage", "Cytotoxic_T_Cell", "Helper_T_Cell", "Regulatory_T_Cell", "NK_Cell",
            "B_Cell", "Plasma_Cell", "Myeloid", "Macrophage_CD163pos", "Macrophage_CD163neg",
            "Dendritic_Cell", "Neutrophil", "Immune_Other", "Immune", "Epithelial",
            "Endothelial", "Stromal"]


def ct_sorted(types):
    return sorted(types, key=lambda t: CT_ORDER.index(t) if t in CT_ORDER else 99)


# ---------------------------------------------------------------- matplotlib setup
def setup():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": FS_BODY,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
        "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
        "axes.linewidth": LW_AXIS, "axes.edgecolor": AXIS, "axes.labelcolor": INK,
        "axes.labelsize": FS_BODY, "axes.titlesize": FS_TITLE, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white",
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": FS_BODY, "ytick.labelsize": FS_BODY,
        "xtick.major.width": LW_AXIS, "ytick.major.width": LW_AXIS,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
        "lines.linewidth": LW_DATA, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.fontsize": FS_BODY, "legend.frameon": False, "legend.handlelength": 1.2,
        "legend.borderaxespad": 0.2, "legend.handletextpad": 0.4,
        "svg.hashsalt": "admit",
        "axes.unicode_minus": True,
    })


def figure(height_in, width_in=FULL_W):
    fig = plt.figure(figsize=(width_in, height_in))
    fig._admit_h = height_in
    fig._admit_w = width_in
    return fig


def ax_in(fig, x, y, w, h, **kw):
    """Axes at (x, y) inches from the top-left corner, size w x h inches."""
    W, H = fig._admit_w, fig._admit_h
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)


def fig_text(fig, x, y, s, **kw):
    """Text at (x, y) inches from the top-left corner."""
    W, H = fig._admit_w, fig._admit_h
    kw.setdefault("fontsize", FS_BODY)
    kw.setdefault("color", INK)
    kw.setdefault("va", "top")
    return fig.text(x / W, 1 - y / H, s, **kw)


def panel_letter(fig, x, y, letter):
    return fig_text(fig, x, y, letter, fontsize=FS_LETTER, fontweight="bold", ha="left", va="top")


def panel_title(fig, x, y, title, sub=None, width=None):
    t = fig_text(fig, x, y, title, fontsize=FS_TITLE, fontweight="bold", ha="left", va="top")
    s = None
    if sub:
        s = fig_text(fig, x, y + 0.15, sub, fontsize=FS_BODY, color=INK2, ha="left", va="top",
                     linespacing=1.15)
    return t, s


def style_axes(ax, grid="y", xlabel=None, ylabel=None):
    ax.tick_params(axis="both", which="both", length=2.0, width=LW_AXIS)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(LW_AXIS)
        ax.spines[sp].set_color(AXIS)
    if grid in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, lw=LW_GRID, ls="-", zorder=0)
    if grid in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, lw=LW_GRID, ls="-", zorder=0)
    ax.set_axisbelow(True)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=FS_BODY, color=INK, labelpad=2)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=FS_BODY, color=INK, labelpad=2)


def target_line(ax, y=0.9, orient="h", label=None, label_pos=None, **kw):
    """Reference line at 1 - alpha. Solid, thin, ink-coloured (not a gridline)."""
    kw.setdefault("color", INK2)
    kw.setdefault("lw", 0.7)
    kw.setdefault("zorder", 1.5)
    if orient == "h":
        ax.axhline(y, **kw)
    else:
        ax.axvline(y, **kw)
    if label:
        x0, y0 = label_pos if label_pos else ((1.0, y) if orient == "h" else (y, 1.0))
        ax.text(x0, y0, label, transform=ax.get_yaxis_transform() if orient == "h" else ax.get_xaxis_transform(),
                fontsize=FS_BODY, color=INK2, ha="right" if orient == "h" else "left",
                va="bottom" if orient == "h" else "top")


def legend_handles(arms, target=True, target_label="target 1−α = 0.90"):
    from matplotlib.lines import Line2D
    hs = []
    for a in arms:
        lab, col, mk = ARMS[a]
        hs.append(Line2D([], [], ls="none", marker=mk, ms=4.2, mfc=col, mec="white", mew=0.3,
                         label=lab))
    if target:
        hs.append(Line2D([], [], color=INK2, lw=0.7, label=target_label))
    return hs


def fig_legend(fig, x, y, handles, ncol=None, **kw):
    """Legend whose upper-left corner sits at (x, y) inches from the top-left."""
    W, H = fig._admit_w, fig._admit_h
    kw.setdefault("fontsize", FS_BODY)
    kw.setdefault("columnspacing", 1.4)
    kw.setdefault("handletextpad", 0.3)
    kw.setdefault("borderpad", 0.0)
    kw.setdefault("borderaxespad", 0.0)
    return fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x / W, 1 - y / H),
                      ncol=ncol or len(handles), frameon=False, **kw)


def diverging_cmap(neg=ORANGE, pos=BLUE_RAMP[-2], mid=NEUTRAL, name="admit_div"):
    """orange (negative) <-> neutral grey <-> blue (positive). Warm/cool poles, grey midpoint."""
    return LinearSegmentedColormap.from_list(
        name, [to_rgb(neg), to_rgb("#f6c3ad"), to_rgb(mid), to_rgb("#9ec5f4"), to_rgb(pos)])


def fmt_n(n):
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.1f}M".replace(".0M", "M")
    if n >= 1e4:
        return f"{n / 1e3:.0f}k"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{int(round(n))}"


def fmt_int(n):
    return f"{int(round(n)):,}"


# ---------------------------------------------------------------- data access + provenance
@functools.lru_cache(maxsize=None)
def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def exists(path):
    return Path(path).exists()


def rel(path):
    p = str(path)
    p = p.replace(str(RUNS) + "/", "$ADMIT_RUNS/").replace(str(ASSETS) + "/", "$ADMIT_FIG_ASSETS/")
    return p.replace(str(ROOT) + "/", "")


class Sources:
    """Collects (figure, panel, status, json, fields, note) rows."""

    def __init__(self, fig_id):
        self.fig_id = fig_id
        self.rows = []

    def add(self, panel, path, fields, status="final", note=""):
        self.rows.append({"figure": self.fig_id, "panel": panel, "status": status,
                          "json_path": rel(path), "fields": fields, "note": note})

    def write(self):
        cols = ["figure", "panel", "status", "json_path", "fields", "note"]
        old = []
        if SOURCES_TSV.exists():
            with open(SOURCES_TSV) as fh:
                old = [r for r in csv.DictReader(fh, delimiter="\t") if r["figure"] != self.fig_id]
        rows = old + self.rows
        rows.sort(key=lambda r: (r["figure"], r["panel"]))
        with open(SOURCES_TSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})


# ---------------------------------------------------------------- checks
EMDASH = "—"


def _visible_texts(fig):
    """Texts that are actually drawn (records Text.draw calls during a full canvas draw)."""
    drawn = set()
    orig = Text.draw

    def _draw(self, renderer):
        drawn.add(id(self))
        return orig(self, renderer)

    Text.draw = _draw
    try:
        fig.canvas.draw()
    finally:
        Text.draw = orig
    out = []
    for t in fig.findobj(Text):
        if id(t) not in drawn:
            continue
        if not t.get_visible() or not t.get_text().strip():
            continue
        ax = t.axes
        if ax is not None and not ax.axison and t not in ax.texts:
            continue  # tick labels of an axes drawn without axis
        out.append(t)
    return out


def _text_poly(t, r):
    """Corners (4x2, display px) of the rotated rectangle a Text occupies."""
    import numpy as np
    bb = t.get_window_extent(r)
    th = np.deg2rad(t.get_rotation() % 360)
    if min(abs(np.sin(th)), abs(np.cos(th))) < 1e-6:
        return np.array([[bb.x0, bb.y0], [bb.x1, bb.y0], [bb.x1, bb.y1], [bb.x0, bb.y1]])
    fp = t.get_fontproperties()
    sizes = [r.get_text_width_height_descent(l or " ", fp, ismath=False) for l in t.get_text().split("\n")]
    w = max(s[0] for s in sizes)
    h = sum(s[1] for s in sizes) * (1.0 if len(sizes) == 1 else t.get_linespacing())
    cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    c = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) @ R.T
    return c + [cx, cy]


def _polys_overlap(a, b, tol=1.0):
    """Separating-axis test for two convex quads; True if they overlap by more than tol px."""
    import numpy as np
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            nn = np.hypot(*n)
            if nn == 0:
                continue
            n = n / nn
            pa, pb = a @ n, b @ n
            if min(pa.max(), pb.max()) - max(pa.min(), pb.min()) <= tol:
                return False
    return True


def check_figure(fig, verbose=True):
    """Font-size floor, em dashes, text outside the canvas, text-text overlaps (rotation-aware)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    problems = []
    texts = _visible_texts(fig)
    boxes = []
    for t in texts:
        s = t.get_text()
        if EMDASH in s:
            problems.append(f"em dash in text: {s!r}")
        if t.get_fontsize() < MIN_PT - 1e-6:
            problems.append(f"text below {MIN_PT} pt ({t.get_fontsize():.1f}): {s!r}")
        bb = t.get_window_extent(r)
        if bb.width == 0 or bb.height == 0:
            continue
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
            problems.append(f"text outside canvas: {s!r}")
        boxes.append((t, _text_poly(t, r)))
    for (t1, p1), (t2, p2) in itertools.combinations(boxes, 2):
        if _polys_overlap(p1, p2):
            problems.append(f"overlap: {t1.get_text()!r} x {t2.get_text()!r}")
    if verbose:
        if problems:
            print(f"[check] {len(problems)} issue(s):")
            for p in problems:
                print("   ", p)
        else:
            print("[check] no text issues (size floor, em dash, canvas, overlap)")
    return problems


def finish(fig, fig_id, stem, sources, check=True):
    outdir = FIGDIR / fig_id
    outdir.mkdir(parents=True, exist_ok=True)
    if check:
        check_figure(fig)
    pdf = outdir / f"{stem}.pdf"
    svg = outdir / f"{stem}.svg"
    png = outdir / f"{stem}_preview.png"
    meta = {"Creator": "figures (matplotlib)", "Title": stem}
    fig.savefig(pdf, metadata=meta)
    fig.savefig(svg, metadata={"Title": stem})
    fig.savefig(png, dpi=300)
    sources.write()
    # no em dash may reach the vector output either
    if EMDASH in svg.read_text(encoding="utf-8"):
        print("[check] WARNING: em dash found in SVG output")
    print(f"[out] {pdf}\n[out] {svg}\n[out] {png}")
    return pdf, svg, png
