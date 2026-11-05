"""Fig 1 (earlier layout; make_fig1_v9.py imports its data loaders, colour rule and panels e and g).

    python figures/make_fig1_v8.py

No model runs and no GPU. Real data: the Orion field and its per-cell typing ($ADMIT_FIG_ASSETS/orion_case),
calibration score histograms ($ADMIT_FIG_ASSETS/orion_case/fig1_v8_calibration_scores.npz, from make_fig1_v8_scores.py)
and Cords core cell maps ($ADMIT_RUNS/regions, as in Fig 4a). Drawn by hand: the tree, the flows in e and f, the ticks in g.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from matplotlib.patches import FancyArrowPatch, Rectangle
from skimage.segmentation import find_boundaries

sys.path.insert(0, str(Path(__file__).resolve().parent))
import admit_style as S  # noqa: E402
import make_fig4_regions as F4  # noqa: E402
import tissue as T  # noqa: E402

PANEL = "del_CD8"                       # panel of the maps in d; struck-through markers in a follow it
ASSET = S.ASSETS / "orion_case"
FIELD = ASSET / S.ORION_CASE_REGION
SCORES = ASSET / "fig1_v8_calibration_scores.npz"
REGIONS = S.RUNS / "regions" / "cords2024_nsclc_imc"
OUT_DIR = S.FIGDIR
STEM = "fig1_v8"

W, H = 6.5, 3.62
S.MIN_PT = 6.0
FS_LET, FS_TTL, FS_LAB, FS_XS = 8.5, 7.0, 6.5, 6.0
ZOOM = (1262, 1077, 616)                # x0, y0, size in full-resolution px (0.325 um/px): the 200 um window of Fig 1e
UM_PX = 0.325
GREY = S.MUTED
LINE = "#6f6e69"

NC = T.NODE_COL
MARKERS = ["CD45", "CD3", "CD4", "CD8", "FOXP3", "CD20", "CD68", "CD163", "CD31", "αSMA", "PanCK", "E-cadherin",
           "CD45RO", "Ki67", "PD1", "PDL1"]
PANEL_NAME = {"CD3": "CD3e", "αSMA": "aSMA", "E-cadherin": "Ecadherin"}

# Real requirement map (Ontology.from_repo): markers each node adds to its parent. NK, plasma, dendritic and neutrophil
# need markers Orion lacks and are left out.
TREE = [("Immune", "Immune", 0, "CD45"), ("T_Lineage", "T cell", 1, "CD3"), ("Cytotoxic_T_Cell", "Cytotoxic T", 2, "CD8"),
        ("Helper_T_Cell", "Helper T", 2, "CD4, FOXP3"), ("Regulatory_T_Cell", "Regulatory T", 2, "CD4, FOXP3"),
        ("B_Cell", "B cell", 1, "CD20"), ("Myeloid", "Myeloid", 1, "CD68"),
        ("Macrophage_CD163pos", "CD163+ macroph.", 2, "CD163"), ("Macrophage_CD163neg", "CD163− macroph.", 2, "CD163"),
        ("Immune_Other", "Immune, other", 1, ""), ("Epithelial", "Epithelial", 0, "PanCK"),
        ("Endothelial", "Endothelial", 0, "CD31"), ("Stromal", "Stromal", 0, "αSMA")]
SCORE_ROWS = [("Epithelial", "Epithelial"), ("Stromal", "Stromal"), ("B_Cell", "B cell"), ("T_Lineage", "T cell"),
              ("Helper_T_Cell", "Helper T"), ("Macrophage_CD163neg", "CD163− mac.")]


# ======================================================================= overlay canvas, inches, y down
class Canvas:
    def __init__(self):
        self.fig = S.figure(H, W)
        self.ax = self.fig.add_axes([0, 0, 1, 1], zorder=10)
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        self.ax.set_axis_off()

    def text(self, x, y, s, size=FS_LAB, color=S.INK, ha="left", va="center", weight="normal", ls=1.05):
        return self.ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, linespacing=ls, zorder=20)

    def line(self, xs, ys, color=LINE, lw=0.5, z=5):
        self.ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_capstyle="butt")

    def rect(self, x, y, w, h, fc="none", ec="none", lw=0.5, z=3):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw, zorder=z))

    def arrow(self, x0, y0, x1, y1, color=S.INK, lw=0.5, ms=4.5, rad=0.0, style="-|>"):
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                          shrinkA=0, shrinkB=0, zorder=6, connectionstyle=f"arc3,rad={rad}"))

    def title(self, x, y, letter, s):
        self.text(x, y, letter, size=FS_LET, weight="bold")
        self.text(x + 0.125, y, s, size=FS_TTL)

    def image(self, img, x, y, s, outline=None):
        a = S.ax_in(self.fig, x, y, s, s)
        a.imshow(img, interpolation="none")
        a.set_axis_off()
        if outline:
            self.rect(x, y, s, s, ec=outline, lw=0.9, z=8)

    def scalebar(self, x_right, y_bottom, length, color="white", lw=1.0):
        self.line([x_right - length, x_right], [y_bottom, y_bottom], color=color, lw=lw, z=9)


# ======================================================================= data
def load_field():
    d = {}
    d["cells"] = pd.read_parquet(FIELD / "cells.parquet").set_index("cell_id")
    d["meta"] = json.loads((FIELD / "admit_cells_meta.json").read_text())
    x0, y0, s = ZOOM
    d["he"] = tifffile.imread(FIELD / "he_crop_full.tif")[y0:y0 + s, x0:x0 + s]
    d["mask"] = np.array(np.load(FIELD / "mask_crop_full.npy", mmap_mode="r")[y0:y0 + s, x0:x0 + s])
    d["if"] = np.load(FIELD / "if_crop_full.npy", mmap_mode="r")
    d["if2"] = np.load(FIELD / "if_crop_2x.npy", mmap_mode="r")
    d["ch"] = json.loads((FIELD / "if_channels.json").read_text())["channels"]
    return d


def composite(d, channels, window):
    """Pseudocolour crop. Contrast: 1-99.5 percentile of each channel over the whole field, as in Fig 1a."""
    x0, y0, s = window
    layers, cols = [], []
    for name, disp, gain in channels:
        i = d["ch"].index(name)
        lo, hi = np.percentile(np.asarray(d["if2"][i], np.float32), (1, 99.5))
        layers.append(T.stretch(np.asarray(d["if"][i, y0:y0 + s, x0:x0 + s], np.float32), lo, hi) * gain)
        cols.append(T.MARKER_COL[disp])
    return T.composite(layers, cols)


def cell_map(mask, colours):
    ids = np.unique(mask)
    lut = np.tile(T.rgb(T.MAP_BG), (len(ids), 1))
    for k, i in enumerate(ids):
        if i in colours:
            lut[k] = colours[i]
    img = lut[np.searchsorted(ids, mask)]
    img[find_boundaries(mask, mode="inner")] = T.rgb(T.MAP_BG)
    return img


def colours(df, kind):
    out = {}
    if kind == "reference":
        for i, g in zip(df.index, df["gating_node17"]):
            out[i] = T.rgb(NC[g]) if isinstance(g, str) and g in NC else T.rgb(T.NO_LABEL)
    elif kind == "forced":
        for i, lab, ok in zip(df.index, df[f"{PANEL}__forced"], df[f"{PANEL}__forced_supported"]):
            out[i] = T.rgb(NC[lab]) if ok else T.rgb(S.RED)
    else:
        for i, fin, amb, lca in zip(df.index, df[f"{PANEL}__finest"], df[f"{PANEL}__ambiguous"], df[f"{PANEL}__lca"]):
            if fin == "abstain":
                out[i] = T.rgb(T.ABSTAIN)
            elif lca == "root":
                out[i] = T.rgb(NC["root"])
            else:
                out[i] = T.tint(NC[fin]) if amb else T.rgb(NC[fin])
    return out


def cd8_window(d, size=308):
    """Sub-window of the zoom with the most CD8 signal (fixed 32 px grid), for panel e."""
    x0, y0, s = ZOOM
    a = np.asarray(d["if"][d["ch"].index("CD8a"), y0:y0 + s, x0:x0 + s], np.float32)
    best, win = -1.0, None
    for yy in range(0, s - size + 1, 32):
        for xx in range(0, s - size + 1, 32):
            v = float(a[yy:yy + size, xx:xx + size].sum())
            if v > best:
                best, win = v, (x0 + xx, y0 + yy, size)
    return win


def load_cores():
    """The core-selection rule of Fig 4a; returns two unselected and two selected cores."""
    z = np.load(REGIONS / "regions_xgb_seed0.npz", allow_pickle=True)
    nodes = [str(n) for n in z["nodes"]]
    cells = pd.read_parquet(REGIONS / "regions_xgb_seed0_cells_L6.parquet")
    reg = cells.groupby("region").agg(sel=("sel_uncert", "max"), n=("uL", "size"))
    key = {str(k): i for i, k in enumerate(z["region_key"])}
    reg["score"] = [float(z["scores"][0, 0, key[r]]) for r in reg.index]
    reg = reg[reg.n >= 200].sort_values("score")
    pick = []
    for flag in (0, 1):
        sub = reg[reg.sel == flag]
        pick += [sub.index[int(round(q * (len(sub) - 1)))] for q in (1 / 8, 3 / 8, 5 / 8, 7 / 8)]
    use = [pick[1], pick[3], pick[6], pick[7]]
    ext = max(max(float(np.ptp(cells.loc[cells.region == r, "x"])), float(np.ptp(cells.loc[cells.region == r, "y"]))) for r in use)
    return {r: cells[cells.region == r] for r in use}, use, nodes, ext


def core_map(c, d, col, nodes, ext, x, y, s, outline=None):
    a = S.ax_in(c.fig, x, y, s, s)
    cols = np.array([F4.set_colour(F4.decode(m, nodes)) for m in d[col].values])
    a.set_facecolor(T.MAP_BG)
    a.scatter(d["x"] - d["x"].min(), d["y"] - d["y"].min(), s=0.62, c=cols, linewidths=0, rasterized=True)
    a.set_xlim(0, ext)
    a.set_ylim(ext, 0)
    a.set_xticks([])
    a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)
    if outline:
        c.rect(x, y, s, s, ec=outline, lw=0.9, z=8)


# ======================================================================= top row
def panel_a(c, d, x, y):
    c.title(x, y, "a", "Field and panel")
    s = 0.95
    top = y + 0.1
    c.image(d["he"][::2, ::2], x, top, s)
    c.image(composite(d, [("Hoechst", "Hoechst", 0.8), ("Pan-CK", "PanCK", 1.0), ("CD3e", "CD3", 1.0), ("CD8a", "CD8", 1.0)],
                      ZOOM)[::2, ::2], x, top + s + 0.02, s)
    c.text(x + 0.03, top + 0.06, "H&E", size=FS_XS, color=S.INK, weight="bold")
    for name, disp, dx, row in (("Hoechst", "Hoechst", 0.03, 0), ("PanCK", "PanCK", 0.39, 0), ("CD3", "CD3", 0.03, 1),
                                ("CD8", "CD8", 0.23, 1)):
        col = T.LABEL_TEXT_COL.get(disp, T.MARKER_COL[disp])
        t = c.text(x + dx, top + s + 0.08 + row * 0.105, name, size=FS_XS, color=col, weight="bold")
        t.set_bbox(dict(boxstyle="square,pad=0.08", facecolor="black", edgecolor="none"))
    length = 50 / (ZOOM[2] * UM_PX) * s
    c.scalebar(x + s - 0.05, top + 2 * s - 0.03, length)
    c.scalebar(x + s - 0.05, top + s - 0.05, length, color=S.INK)

    # marker list: Orion panel as plain text, markers of the deleted set struck through
    missing = set(d["meta"]["panels"][PANEL]["missing_vs_full"])
    lx = x + s + 0.06
    pitch = (2 * s + 0.02 - 0.08) / (len(MARKERS) - 1)
    for i, m in enumerate(MARKERS):
        yy = top + 0.04 + i * pitch
        gone = PANEL_NAME.get(m, m) in missing
        t = c.text(lx, yy, "E-cad" if m == "E-cadherin" else m, size=FS_XS, color=GREY if gone else S.INK)
        if gone:
            c.line([lx - 0.01, lx + 0.19], [yy, yy], color=S.INK, lw=0.6, z=21)


def panel_b(c, x, y):
    c.title(x, y, "b", "Requirement map")
    top = y + 0.17
    pitch = 0.146
    ind = 0.095
    sw = 0.062
    mx = x + 1.06
    rows = {}
    last_at = {}
    for i, (key, name, depth, _) in enumerate(TREE):
        rows[key] = (top + i * pitch, depth)
    for i, (key, name, depth, mk) in enumerate(TREE):
        yy = top + i * pitch
        xx = x + depth * ind
        off = key == "Cytotoxic_T_Cell"
        if depth > 0:
            px = x + (depth - 1) * ind + sw / 2
            py = last_at[depth - 1]
            c.line([px, px, xx], [py + sw / 2, yy, yy])
        last_at[depth] = yy
        c.rect(xx, yy - sw / 2, sw, sw, fc="white" if off else NC[key], ec=GREY if off else "none", lw=0.5, z=6)
        c.text(xx + sw + 0.03, yy, name, size=FS_LAB, color=GREY if off else S.INK)
        if mk:
            c.text(mx, yy, mk, size=FS_XS, color=LINE)
        if off:
            c.line([mx - 0.01, mx + 0.17], [yy, yy], color=S.INK, lw=0.6, z=21)
    # thin arrow: cytotoxic T is revised to its T cell ancestor
    y_c, y_t = rows["Cytotoxic_T_Cell"][0], rows["T_Lineage"][0]
    c.arrow(x + 0.76, y_c - 0.02, x + 0.44, y_t + 0.005, rad=0.42, lw=0.6)


def panel_c(c, x, y):
    c.title(x, y, "c", "Per-type thresholds")
    z = np.load(SCORES, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    edges, hist, lam = z["edges"], z["hist"], z["lam"]
    lo, hi = -4.0, 0.0
    px, pw = x + 0.58, 0.90
    rh = 0.232
    top = y + 0.30

    def X(v):
        return px + (np.clip(v, lo, hi) - lo) / (hi - lo) * pw

    centres = 0.5 * (edges[:-1] + edges[1:])
    for i, (key, label) in enumerate(SCORE_ROWS):
        base = top + (i + 1) * rh
        k = names.index(key)
        v = hist[k].astype(float).copy()
        under = centres < lo
        v = v[~under]
        v[0] += hist[k][under].sum()              # cells below the axis go in the first visible bin
        xs = centres[~under]
        v = v / v.max()
        c.ax.fill_between(X(xs), base, base - v * (rh - 0.03), step="mid", color=NC[key], linewidth=0, zorder=4)
        c.line([px, px + pw], [base, base], color=S.AXIS, lw=0.4, z=3)
        c.text(px - 0.04, base - 0.05, label, size=FS_XS, ha="right")
        xt = float(X(np.log10(1.0 - lam[k])))
        c.line([xt, xt], [base - rh + 0.015, base + 0.015], color=S.BLUE, lw=1.1, z=7)
    bottom = top + len(SCORE_ROWS) * rh
    xs = float(X(np.log10(1.0 - float(z["shared"]))))
    c.line([xs, xs], [top - 0.01, bottom + 0.015], color=S.AQUA, lw=0.9, z=6)
    c.text(xs, top - 0.065, "shared", size=FS_XS, color="#0f7a55", ha="center")
    c.text(px + pw, top - 0.065, "per type", size=FS_XS, color=S.BLUE, ha="right")
    # axis: classifier probability of the cell's own (revised) type
    for v, lab in ((-4, "0.0001"), (-2, "0.01"), (0, "1")):
        xv = float(X(v))
        c.line([xv, xv], [bottom + 0.015, bottom + 0.045], color=S.AXIS, lw=0.5)
        c.text(xv, bottom + 0.11, lab, size=FS_XS, ha="center", color=S.INK2)
    c.line([px, px + pw], [bottom + 0.015, bottom + 0.015], color=S.AXIS, lw=0.5)
    c.text(px + pw / 2, bottom + 0.235, "probability of own type", size=FS_XS, ha="center", color=S.INK2)


def panel_d(c, d, x, y):
    c.title(x, y, "d", "Output without CD8")
    s, g = 0.86, 0.03
    x0, y0, size = ZOOM
    ids = np.unique(d["mask"])
    df = d["cells"].loc[[i for i in ids if i in d["cells"].index]]
    mask = d["mask"][::2, ::2]
    y1 = y + 0.20
    y2 = y1 + s + 0.14
    bar = 50 / (size * UM_PX) * s
    for kind, label, xx, yy in (("reference", "Reference", x, y1), ("forced", "Forced label", x + s + g, y1),
                                ("admit", "ADMIT set", x, y2)):
        c.text(xx, yy - 0.055, label, size=FS_LAB)
        c.image(cell_map(mask, colours(df, kind)), xx, yy, s)
        c.scalebar(xx + s - 0.05, yy + s - 0.05, bar)
    # key for the ADMIT map; type colours are those of panel b
    kx = x + s + g + 0.03
    ky = y2 + 0.13
    for i, (col, lab) in enumerate(((T.tint(NC["T_Lineage"]), "lighter: set"), (T.rgb(NC["root"]), "beige: spans"),
                                    (T.rgb(NC["root"]), "lineages"), (T.rgb(T.ABSTAIN), "grey: empty set"))):
        yy = ky + i * 0.135
        if lab != "lineages":
            c.rect(kx, yy - 0.031, 0.062, 0.062, fc=col, z=6)
        c.text(kx + 0.09, yy, lab, size=FS_XS, color=S.INK2)
    c.text(kx, ky + 4 * 0.135 + 0.02, "colours as in b", size=FS_XS, color=S.INK2)


# ======================================================================= bottom row
def panel_e(c, d, x, y):
    c.title(x, y, "e", "Marker deletion")
    s = 0.62
    win = cd8_window(d)
    full = composite(d, [("Hoechst", "Hoechst", 0.6), ("CD3e", "CD3", 1.0), ("CD8a", "CD8", 1.0)], win)
    masked = composite(d, [("Hoechst", "Hoechst", 0.6), ("CD3e", "CD3", 1.0)], win)
    top = y + 0.24
    x2 = x + 1.12
    c.text(x, top - 0.055, "held-out cells", size=FS_LAB)
    c.text(x2, top - 0.055, "same cells", size=FS_LAB)
    c.image(full, x, top, s)
    c.image(masked, x2, top, s)
    bar = 50 / (win[2] * UM_PX) * s
    c.scalebar(x + s - 0.04, top + s - 0.04, bar)
    c.scalebar(x2 + s - 0.04, top + s - 0.04, bar)
    c.arrow(x + s + 0.03, top + s / 2, x2 - 0.03, top + s / 2)
    c.text((x + s + x2) / 2, top + s / 2 - 0.07, "mask CD8", size=FS_LAB, ha="center")
    # outputs
    mid = x2 + s / 2
    yo = top + s + 0.03
    c.arrow(mid, yo, mid, yo + 0.19)
    c.arrow(x2 + 0.08, yo, x + 0.82, yo + 0.19)
    c.text(mid, yo + 0.25, "ADMIT: set", size=FS_LAB, ha="center", color=S.BLUE)
    c.text(x + 0.72, yo + 0.25, "forced: one label", size=FS_LAB, ha="center")


def patients(c, x, y, n, s=0.085, gap=0.02):
    for i in range(n):
        c.rect(x + i * (s + gap), y - s / 2, s, s, ec=LINE, lw=0.5, z=5)


def panel_f(c, x, y):
    c.title(x, y, "f", "Panel estimated in advance")
    y1, y2 = y + 0.44, y + 0.98
    ax0, ax1 = x + 0.46, x + 1.06
    for yy, n, who, res in ((y1, 4, "development\npatients", "predicted\nresolution"),
                            (y2, 3, "held-out\npatients", "realised\nresolution")):
        patients(c, x, yy, n)
        c.text(x, yy + 0.155, who, size=FS_LAB, va="center")
        c.arrow(ax0, yy, ax1, yy)
        c.text((ax0 + ax1) / 2, yy - 0.115, "masked to\ncandidate panel", size=FS_XS, ha="center", va="center")
        c.text(ax1 + 0.04, yy, res, size=FS_LAB)
    xm = ax1 + 0.22
    c.arrow(xm, y1 + 0.135, xm, y2 - 0.135, style="<|-|>")
    c.text(xm + 0.04, (y1 + y2) / 2, "compare", size=FS_XS, color=S.INK2)


def panel_g(c, x, y):
    c.title(x, y, "g", "Region selection")
    cores, use, nodes, ext = load_cores()
    s, g = 0.50, 0.03
    y1 = y + 0.20
    y2 = y1 + s + 0.11
    c.text(x + 0.0, y1 - 0.055, "6-plex, all regions", size=FS_LAB)
    for j, r in enumerate(use):
        xx = x + j * (s + g)
        sel = j >= 2
        core_map(c, cores[r], "CL_pooled", nodes, ext, xx, y1, s, outline=S.BLUE if sel else None)
        if sel:
            core_map(c, cores[r], "final_uncert_cond", nodes, ext, xx, y2, s, outline=S.BLUE)
            c.arrow(xx + s / 2, y1 + s + 0.012, xx + s / 2, y2 - 0.012)
    c.scalebar(x + s - 0.03, y1 + s - 0.03, 200 / ext * s, lw=0.9)
    c.text(x + 2 * (s + g) - 0.05, y2 + s / 2, "full panel on\nselected regions", size=FS_LAB, ha="right")

    # calibration: one threshold for all regions, or one per selection stratum (drawn by hand)
    bx, bw = x + 4 * (s + g) + 0.08, 0.72
    yp = y1 + 0.03
    c.text(bx, yp, "pooled", size=FS_LAB, color="#c9501f")
    c.line([bx, bx + bw], [yp + 0.11, yp + 0.11], color=S.AXIS, lw=0.5)
    c.line([bx + 0.40] * 2, [yp + 0.055, yp + 0.165], color=S.ORANGE, lw=1.1, z=7)
    yq = yp + 0.33
    c.text(bx, yq, "selection-\nconditional", size=FS_LAB, color=S.BLUE)
    for k, (lab, xt) in enumerate((("selected", 0.22), ("unselected", 0.54))):
        yy = yq + 0.165 + k * 0.22
        c.text(bx, yy, lab, size=FS_XS, color=S.INK2)
        c.line([bx, bx + bw], [yy + 0.10, yy + 0.10], color=S.AXIS, lw=0.5)
        c.line([bx + xt] * 2, [yy + 0.055, yy + 0.145], color=S.BLUE, lw=1.1, z=7)


def main():
    S.setup()
    c = Canvas()
    d = load_field()
    panel_a(c, d, 0.0, 0.06)
    panel_b(c, 1.47, 0.06)
    panel_c(c, 3.03, 0.06)
    panel_d(c, d, 4.71, 0.06)
    y = 2.28
    panel_e(c, d, 0.0, y)
    panel_f(c, 1.86, y)
    panel_g(c, 3.58, y)

    problems = S.check_figure(c.fig)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"Creator": "figures/make_fig1_v8.py (matplotlib)", "Title": STEM}
    c.fig.savefig(OUT_DIR / f"{STEM}.pdf", metadata=meta, dpi=450)
    c.fig.savefig(OUT_DIR / f"{STEM}.png", dpi=300)
    print(f"[out] {OUT_DIR / (STEM + '.pdf')}\n[out] {OUT_DIR / (STEM + '.png')}")
    return c, problems


if __name__ == "__main__":
    main()
