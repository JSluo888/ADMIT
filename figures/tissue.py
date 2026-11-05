"""Shared helpers for the tissue panels.

Images are embedded rasters (>= 300 dpi at print size; imshow with interpolation 'none' keeps native pixels in
PDF/SVG); every annotation (scale bars, zoom boxes, connectors, outlines, labels, legends) is vector.

Colour rules
  - one colour per cell type, lineage families: T blue, myeloid orange, B/plasma violet, epithelial magenta,
    stromal/endothelial green; coarser node = lighter tint of its family
  - ambiguous ADMIT set (more than one leaf, or members on different root paths): lighter tint of the
    colour of its finest member
  - abstention (empty set): light grey with a thin dark outline; that grey means nothing else
  - background cells in highlight maps: dark grey (darker than the abstention grey)
  - red: unsupported forced calls only
  - composites: black background; each marker keeps one colour in the paper; FOXP3 white
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
from scipy import ndimage

import admit_style as S

# ----------------------------------------------------------------------- colours
from case_common import NODE_COL as _SHARED  # noqa: E402  (single source of the lineage palette)

NODE_COL = {k: v for k, v in _SHARED.items() if k not in ("UNMAPPED", "UNRESOLVED", "EMPTY")}
NODE_COL["root"] = _SHARED["UNRESOLVED"]   # set whose members span lineages (pale warm tone, not grey)
ABSTAIN = "#d9d9d9"          # empty set; this grey means nothing else
ABSTAIN_EDGE = "#4a4a4a"
MAP_BG = "#1a1a1a"           # cell-map background and cell boundaries
OTHER_DARK = "#454545"       # non-highlighted cells in highlight maps (darker than the abstention grey)
NO_LABEL = "#2e2e2e"         # cells without a reference label
UNIDENT = "#7a7a7a"          # empty set because the revised label pi_P(y) is undefined (correct to output nothing);
                             # mid grey, no outline: distinct from abstain (light grey + outline) and NO_LABEL (dark)
UNSUPPORTED = S.RED          # forced call whose defining marker is missing
WRONG_EDGE = "#ffffff"       # zoom-only outline for a wrong but supported forced call
TINT = 0.5                   # ambiguous set: mix halfway to white

LABEL = {
    "T_Lineage": "T cell", "Cytotoxic_T_Cell": "Cytotoxic T", "Helper_T_Cell": "Helper T",
    "Regulatory_T_Cell": "Regulatory T", "NK_Cell": "NK", "Myeloid": "Myeloid",
    "Macrophage_CD163pos": "Macrophage CD163+", "Macrophage_CD163neg": "Macrophage CD163−",
    "Dendritic_Cell": "Dendritic", "Neutrophil": "Neutrophil", "B_Cell": "B cell", "Plasma_Cell": "Plasma",
    "Immune_Other": "Immune, other", "Immune": "Immune", "Epithelial": "Epithelial", "Stromal": "Stromal",
    "Endothelial": "Endothelial", "root": "any lineage",
}

# marker colours for composites (one colour per marker across the paper; FOXP3 white)
MARKER_COL = {
    "Hoechst": "#3d6dff", "DNA": "#3d6dff", "Nuclear": "#3d6dff",
    "PanCK": "#ff2bd6", "CD3": "#2bff4a", "CD8": "#ffe600", "FOXP3": "#ffffff",
    "CD4": "#30ffff", "CD68": "#ff9a2e", "CD163": "#ff9a2e", "CD20": "#b07cff", "CD45": "#a0a0a0",
    "SMA": "#00c08b", "GFAP": "#ff2bd6", "Olig2": "#30ffff",
}
LABEL_TEXT_COL = {"Hoechst": "#7d9bff", "Nuclear": "#7d9bff", "DNA": "#7d9bff"}   # brighter text for blue


def rgb(c):
    return np.asarray(to_rgb(c), float)


def tint(c, f=TINT):
    """Lighter tint (mix with white) for ambiguous sets / coarser nodes."""
    return 1 - (1 - rgb(c)) * (1 - f)


def dim(c, f=0.42):
    """Darker shade (mix with black): wrong-but-supported forced calls at field scale."""
    return rgb(c) * f


# ----------------------------------------------------------------------- image processing
def pct_limits(img, mask=None, p=(1, 99.5)):
    v = img[mask > 0] if mask is not None else img.ravel()
    v = v[np.isfinite(v)]
    return np.percentile(v, p[0]), np.percentile(v, p[1])


def stretch(img, lo, hi):
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-12), 0, 1)


def median3(img):
    return ndimage.median_filter(img, size=3)


def composite(chans, colors):
    """Additive pseudocolour composite on black. chans: list of [0,1] arrays."""
    out = np.zeros(chans[0].shape + (3,), np.float32)
    for c, col in zip(chans, colors):
        out += c[..., None] * rgb(col)[None, None, :]
    return np.clip(out, 0, 1)


def block_mean(img, f):
    if f == 1:
        return img
    h, w = img.shape[:2]
    h2, w2 = h // f * f, w // f * f
    img = img[:h2, :w2]
    if img.ndim == 2:
        return img.reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3))
    return img.reshape(h2 // f, f, w2 // f, f, img.shape[2]).mean(axis=(1, 3))


def label_rgb(mask, colors_by_label, bg=None, boundaries=True):
    """mask: int label image; colors_by_label: dict label -> rgb. Background and inner cell boundaries in MAP_BG.
    Returns a float RGB image."""
    from skimage.segmentation import find_boundaries
    bgc = rgb(MAP_BG) if bg is None else np.asarray(bg, np.float32)
    n = int(mask.max()) + 1
    lut = np.tile(bgc.astype(np.float32), (n, 1))
    for lab, col in colors_by_label.items():
        if 0 < lab < n:
            lut[lab] = col
    img = lut[mask]
    if boundaries:
        img[find_boundaries(mask, mode="inner")] = bgc
    return img


# ----------------------------------------------------------------------- axes helpers
def image_ax(fig, x, y, w, h):
    ax = S.ax_in(fig, x, y, w, h)
    ax.set_axis_off()
    return ax


def show(ax, img, extent_um, origin="upper"):
    """extent_um = (width_um, height_um); image coordinates in um, y down."""
    W, H = extent_um
    ax.imshow(img, extent=(0, W, H, 0), interpolation="none", origin=origin)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")


def scale_bar(ax, length_um, extent_um, color="white", label=None, frac_pad=0.05, lw=1.2):
    """Lower-right scale bar with the label above it (rule 5)."""
    W, H = extent_um
    x1 = W * (1 - frac_pad)
    x0 = x1 - length_um
    y = H * (1 - frac_pad)
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=10)
    ax.text(x1, y - H * 0.025, label or f"{length_um:g} {chr(0xb5)}m", color=color, ha="right", va="bottom",
            fontsize=S.FS, zorder=10,
            bbox=dict(boxstyle="square,pad=0.05", facecolor="black", edgecolor="none") if color == "white" else None)


def zoom_box(ax, x0, y0, w, h, color="white", lw=0.75, number=None):
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=color, lw=lw, zorder=11))
    if number is not None:
        ax.text(x0, y0, f"{number}", color=color, ha="left", va="bottom", fontsize=S.FS, fontweight="bold", zorder=11)


def connect(fig, ax_from, box, ax_to, color="#8a8a8a", lw=0.5, corners=("bl", "br"), side="below"):
    """Connector lines from the box (in ax_from data coords) to ax_to: its top corners when the inset sits below
    (side='below'), or its left corners when the inset sits to the right (side='right')."""
    x0, y0, w, h = box
    pts_from = {"bl": (x0, y0 + h), "br": (x0 + w, y0 + h), "tl": (x0, y0), "tr": (x0 + w, y0)}
    if side == "right":
        pairs = [((x0 + w, y0), (0, 1)), ((x0 + w, y0 + h), (0, 0))]
    else:
        pts_to = {"bl": (0, 1), "br": (1, 1)}
        pairs = [(pts_from[c], pts_to[c]) for c in corners]
    for pa, pb in pairs:
        cp = ConnectionPatch(xyA=pa, coordsA=ax_from.transData, xyB=pb, coordsB=ax_to.transAxes,
                             color=color, lw=lw, zorder=0)
        fig.add_artist(cp)


def channel_legend(ax, names, colors, y=0.97, x=0.03, fs=None):
    """Marker names in their channel colours, top-left (rule 1)."""
    fs = fs or S.FS
    xx = x
    for n, c in zip(names, colors):
        t = ax.text(xx, y, n, color=c, transform=ax.transAxes, ha="left", va="top", fontsize=fs,
                    fontweight="bold", zorder=12)
        ax.figure.canvas.draw() if False else None
        xx += 0.035 + 0.055 * len(n) * (fs / 7.0) * (1.1 / max(ax.get_position().width * ax.figure.get_figwidth(), 0.5))
    return xx


def corner_text(ax, lines, colors=None, x=0.03, y=0.975, gap=4, line_gap=1.5, bold=True, extra_lines=()):
    """Words laid out left to right from the top-left corner, wrapping at the image edge, each on a small black box
    (rule 1: channel names in channel colours, top-left). `extra_lines`: (text, colour) rows below, white by default."""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    bb = ax.get_window_extent(r)
    inv = ax.transAxes.inverted()
    colors = colors or ["white"] * len(lines)
    xpx, ypx = bb.x0 + x * bb.width, bb.y0 + y * bb.height
    line_h = None
    for word, col in zip(lines, colors):
        col = LABEL_TEXT_COL.get(word, col)
        t = ax.text(*inv.transform((xpx, ypx)), word, color=col, transform=ax.transAxes, ha="left", va="top",
                    fontsize=S.FS, fontweight="bold" if bold else "normal", zorder=12,
                    bbox=dict(boxstyle="square,pad=0.08", facecolor="black", edgecolor="none"))
        e = t.get_window_extent(r)
        line_h = line_h or e.height
        if e.x1 > bb.x1 - 2 and xpx > bb.x0 + x * bb.width + 1:
            xpx = bb.x0 + x * bb.width
            ypx -= line_h + line_gap
            t.set_position(inv.transform((xpx, ypx)))
            e = t.get_window_extent(r)
        xpx = e.x1 + gap
    for txt, col in extra_lines:
        ypx -= (line_h or 9) + line_gap
        ax.text(*inv.transform((bb.x0 + x * bb.width, ypx)), txt, color=col, transform=ax.transAxes, ha="left",
                va="top", fontsize=S.FS, zorder=12,
                bbox=dict(boxstyle="square,pad=0.08", facecolor="black", edgecolor="none"))


def channel_legend_stack(ax, names, colors, x=0.03, y=0.97, dy=0.085):
    for i, (n, c) in enumerate(zip(names, colors)):
        ax.text(x, y - i * dy, n, color=c, transform=ax.transAxes, ha="left", va="top", fontsize=S.FS,
                fontweight="bold", zorder=12)


# ----------------------------------------------------------------------- vector outlines
def outlines(ax, mask, labels, um_per_px, color, lw=0.75, offset=(0, 0), zorder=9):
    """Vector contours of the given labels in a (small) mask crop; coordinates in um (offset in px)."""
    from skimage.measure import find_contours
    labels = set(int(l) for l in labels)
    if not labels:
        return
    objs = ndimage.find_objects(mask)
    for lab in labels:
        if lab <= 0 or lab > len(objs) or objs[lab - 1] is None:
            continue
        sl = objs[lab - 1]
        sub = np.pad(mask[sl] == lab, 1)
        for cnt in find_contours(sub.astype(float), 0.5):
            yy = (cnt[:, 0] - 1 + sl[0].start + offset[1]) * um_per_px
            xx = (cnt[:, 1] - 1 + sl[1].start + offset[0]) * um_per_px
            ax.plot(xx, yy, color=color, lw=lw, solid_joinstyle="round", zorder=zorder)


# ----------------------------------------------------------------------- legends
def swatch(fig, x, y, color, label, edge=None, size=0.085, text_color=None):
    """Square swatch + label at (x, y) inches from top-left; returns x after the label."""
    ax = S.ax_in(fig, x, y + 0.012, size, size)
    ax.set_axis_off()
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=edge or "none", lw=0.6 if edge else 0,
                           transform=ax.transAxes))
    S.fig_text(fig, x + size + 0.04, y, label, fontsize=S.FS, color=text_color or S.INK)


TREE = [  # (node, depth) in display order, depth starting at 0
    ("Immune", 0), ("T_Lineage", 1), ("Cytotoxic_T_Cell", 2), ("Helper_T_Cell", 2), ("Regulatory_T_Cell", 2),
    ("Myeloid", 1), ("Macrophage_CD163pos", 2), ("Macrophage_CD163neg", 2), ("B_Cell", 1), ("Neutrophil", 1),
    ("Immune_Other", 1),
    ("Epithelial", 0), ("Endothelial", 0), ("Stromal", 0),
]


def tree_legend(fig, x, y, title=None, pitch=0.106, indent=0.11, sw=0.075, split_at=None, col_w=1.25,
                nodes=None):
    """Tree-shaped cell-type legend: a swatch at each node, elbow connectors from parent
    to children. split_at: node that starts a second column (e.g. 'Epithelial'). Returns the bottom y (inches)."""
    ax_lines = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ax_lines.set_axis_off()
    ax_lines.set_xlim(0, fig._admit_w)
    ax_lines.set_ylim(fig._admit_h, 0)
    ax_lines.set_zorder(-1)
    y0 = y
    if title:
        S.fig_text(fig, x, y, title, fontsize=S.FS, color=S.INK, fontweight="bold")
        y0 = y + 0.17
    rows = [r for r in TREE if (r[0] != "Neutrophil" if nodes is None else r[0] in nodes)]   # default node set
    col_x, k, last, bottom = x, 0, {}, y0
    for n, dpt in rows:
        splits = split_at if isinstance(split_at, (tuple, list)) else ((split_at,) if split_at else ())
        if n in splits:
            col_x, k, last = col_x + col_w, 0, {}
        yy = y0 + k * pitch + sw / 2
        xx = col_x + dpt * indent
        if dpt > 0 and (dpt - 1) in last:
            px, py = last[dpt - 1]
            ax_lines.plot([px + sw / 2, px + sw / 2], [py + sw / 2 + 0.005, yy], color=S.AXIS, lw=0.6)
            ax_lines.plot([px + sw / 2, xx - 0.012], [yy, yy], color=S.AXIS, lw=0.6)
        sax = S.ax_in(fig, xx, yy - sw / 2, sw, sw)
        sax.set_axis_off()
        sax.add_patch(Rectangle((0, 0), 1, 1, facecolor=NODE_COL[n], edgecolor="none", transform=sax.transAxes))
        S.fig_text(fig, xx + sw + 0.045, yy, LABEL[n], fontsize=S.FS, color=S.INK, va="center")
        last[dpt] = (xx, yy - sw / 2)
        k += 1
        bottom = max(bottom, yy + pitch)
    return bottom


def status_keys(fig, x, y, keys, pitch=0.13, sw=0.075):
    """Rows of (colour, label, edge) status swatches (e.g. unsupported, abstain, ambiguous tint).
    A label with a line break takes extra height (0.105 in per added line); the swatch sits on its first line."""
    yy = y
    for c, lab, edge in keys:
        sax = S.ax_in(fig, x, yy, sw, sw)
        sax.set_axis_off()
        inner = edge[len("inner:"):] if isinstance(edge, str) and edge.startswith("inner:") else None
        if inner:        # dark swatch with an inner outline (zoom-only outlines, e.g. white = wrong forced call)
            edge = None
        sax.add_patch(Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=edge or "none", lw=0.5 if edge else 0,
                                transform=sax.transAxes))
        if inner:
            sax.add_patch(Rectangle((0.2, 0.2), 0.6, 0.6, facecolor="none", edgecolor=inner, lw=0.7,
                                    transform=sax.transAxes))
        nl = lab.count("\n")
        ty = yy + sw / 2 if nl == 0 else yy - 0.012
        S.fig_text(fig, x + sw + 0.045, ty, lab, fontsize=S.FS, color=S.INK, va="center" if nl == 0 else "top",
                   linespacing=1.05)
        yy += pitch + 0.105 * nl
    return yy
