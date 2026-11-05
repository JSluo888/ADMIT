"""Fig 1: ADMIT on one real Orion field (top row, a -> b -> c -> d) and the three experiments that use it (bottom row).

    python figures/make_fig1_v9.py

No model runs and no GPU. Real data: the Orion field and its per-cell typing, calibration score histograms and Cords
core cell maps (loaded as in make_fig1_v8.py). Drawn by hand: the tree, the arrows, the flows in f, the ticks in g.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import admit_style as S  # noqa: E402
import make_fig1_v8 as F1  # noqa: E402
import tissue as T  # noqa: E402

STEM = "fig1_v9"
F1.H = 3.72
NC = F1.NC
GREY, LINE = F1.GREY, F1.LINE
FS_LAB, FS_XS = F1.FS_LAB, F1.FS_XS
Y0 = 0.20                                   # titles of the top row; the ADMIT bracket sits above
BUS = 2.16                                  # the scorer line under the top row


def arrow_label(c, x0, x1, y, s):
    c.arrow(x0, y, x1, y, lw=0.6)
    c.text((x0 + x1) / 2, y - 0.075, s, size=FS_LAB, ha="center")


def panel_a(c, d, x, y):
    c.title(x, y, "a", "Measured panel $P$")
    s = 0.84
    top = y + 0.12
    c.image(d["he"][::2, ::2], x, top, s)
    c.image(F1.composite(d, [("Hoechst", "Hoechst", 0.8), ("Pan-CK", "PanCK", 1.0), ("CD3e", "CD3", 1.0),
                             ("CD8a", "CD8", 1.0)], F1.ZOOM)[::2, ::2], x, top + s + 0.02, s)
    c.text(x + 0.03, top + 0.06, "H&E", size=FS_XS, color=S.INK, weight="bold")
    for name, disp, dx, row in (("Hoechst", "Hoechst", 0.03, 0), ("PanCK", "PanCK", 0.37, 0), ("CD3", "CD3", 0.03, 1),
                                ("CD8", "CD8", 0.22, 1)):
        col = T.LABEL_TEXT_COL.get(disp, T.MARKER_COL[disp])
        t = c.text(x + dx, top + s + 0.08 + row * 0.105, name, size=FS_XS, color=col, weight="bold")
        t.set_bbox(dict(boxstyle="square,pad=0.08", facecolor="black", edgecolor="none"))
    length = 50 / (F1.ZOOM[2] * F1.UM_PX) * s
    c.scalebar(x + s - 0.05, top + 2 * s - 0.03, length)
    c.scalebar(x + s - 0.05, top + s - 0.05, length, color=S.INK)
    missing = set(d["meta"]["panels"][F1.PANEL]["missing_vs_full"])
    lx = x + s + 0.05
    pitch = (2 * s + 0.02 - 0.08) / (len(F1.MARKERS) - 1)
    for i, m in enumerate(F1.MARKERS):
        yy = top + 0.04 + i * pitch
        gone = F1.PANEL_NAME.get(m, m) in missing
        c.text(lx, yy, "E-cad" if m == "E-cadherin" else m, size=FS_XS, color=GREY if gone else S.INK)
        if gone:
            c.line([lx - 0.01, lx + 0.19], [yy, yy], color=S.INK, lw=0.6, z=21)
    return top + 2 * s + 0.02


SHORT = {"CD163+ macroph.": "CD163+ mac.", "CD163− macroph.": "CD163− mac."}


def panel_b(c, x, y):
    c.title(x, y, "b", "Requirement map $R(c)$")
    top = y + 0.19
    pitch, ind, sw = 0.128, 0.09, 0.058
    mx = x + 0.90
    rows, last_at = {}, {}
    for i, (key, name, depth, mk) in enumerate(F1.TREE):
        yy = top + i * pitch
        rows[key] = yy
        xx = x + depth * ind
        off = key == "Cytotoxic_T_Cell"
        if depth > 0:
            px = x + (depth - 1) * ind + sw / 2
            c.line([px, px, xx], [last_at[depth - 1] + sw / 2, yy, yy])
        last_at[depth] = yy
        c.rect(xx, yy - sw / 2, sw, sw, fc="white" if off else NC[key], ec=GREY if off else "none", lw=0.5, z=6)
        c.text(xx + sw + 0.03, yy, SHORT.get(name, name), size=FS_LAB, color=GREY if off else S.INK)
        if mk:
            c.text(mx, yy, mk, size=FS_XS, color=LINE)
        if off:
            c.line([mx - 0.01, mx + 0.17], [yy, yy], color=S.INK, lw=0.6, z=21)
    c.arrow(x + 0.70, rows["Cytotoxic_T_Cell"] - 0.02, x + 0.42, rows["T_Lineage"] + 0.005, rad=0.42, lw=0.6)
    return rows


def panel_c(c, x, y):
    c.title(x, y, "c", "Per-type calibration under $P$")
    z = np.load(F1.SCORES, allow_pickle=True)
    names = [str(n) for n in z["names"]]
    edges, hist, lam = z["edges"], z["hist"], z["lam"]
    lo, hi = -4.0, 0.0
    px, pw = x + 0.56, 0.88
    rh = 0.212
    top = y + 0.30

    def X(v):
        return px + (np.clip(v, lo, hi) - lo) / (hi - lo) * pw

    centres = 0.5 * (edges[:-1] + edges[1:])
    for i, (key, label) in enumerate(F1.SCORE_ROWS):
        base = top + (i + 1) * rh
        k = names.index(key)
        v = hist[k].astype(float).copy()
        under = centres < lo
        v = v[~under]
        v[0] += hist[k][under].sum()
        v = v / v.max()
        c.ax.fill_between(X(centres[~under]), base, base - v * (rh - 0.03), step="mid", color=NC[key], linewidth=0,
                          zorder=4)
        c.line([px, px + pw], [base, base], color=S.AXIS, lw=0.4, z=3)
        c.text(px - 0.04, base - 0.05, label, size=FS_XS, ha="right")
        xt = float(X(np.log10(1.0 - lam[k])))
        c.line([xt, xt], [base - rh + 0.015, base + 0.015], color=S.BLUE, lw=1.1, z=7)
    bottom = top + len(F1.SCORE_ROWS) * rh
    xs = float(X(np.log10(1.0 - float(z["shared"]))))
    c.line([xs, xs], [top - 0.01, bottom + 0.015], color=S.AQUA, lw=0.9, z=6)
    c.text(xs - 0.03, top - 0.065, "shared", size=FS_XS, color="#0f7a55", ha="right")
    c.text(xs + 0.03, top - 0.065, r"per type $\lambda_c$", size=FS_XS, color=S.BLUE, ha="left")
    for v, lab in ((-4, "0.0001"), (-2, "0.01"), (0, "1")):
        xv = float(X(v))
        c.line([xv, xv], [bottom + 0.015, bottom + 0.045], color=S.AXIS, lw=0.5)
        c.text(xv, bottom + 0.11, lab, size=FS_XS, ha="center", color=S.INK2)
    c.line([px, px + pw], [bottom + 0.015, bottom + 0.015], color=S.AXIS, lw=0.5)
    c.text(px + pw / 2, bottom + 0.225, "probability of own type", size=FS_XS, ha="center", color=S.INK2)
    return top, bottom


def panel_d(c, d, x, y):
    c.title(x, y, "d", "Prediction set $C_P(x)$")
    s, g = 0.78, 0.03
    ids = np.unique(d["mask"])
    df = d["cells"].loc[[i for i in ids if i in d["cells"].index]]
    mask = d["mask"][::2, ::2]
    y1 = y + 0.21
    y2 = y1 + s + 0.12
    bar = 50 / (F1.ZOOM[2] * F1.UM_PX) * s
    for kind, label, xx, yy in (("reference", "Reference", x, y1), ("forced", "Forced label", x + s + g, y1),
                                ("admit", "ADMIT set", x, y2)):
        c.text(xx, yy - 0.055, label, size=FS_LAB, color=S.BLUE if kind == "admit" else S.INK)
        c.image(F1.cell_map(mask, F1.colours(df, kind)), xx, yy, s)
        c.scalebar(xx + s - 0.05, yy + s - 0.05, bar)
    kx = x + s + g + 0.03
    ky = y2 + 0.10
    for i, (col, lab) in enumerate(((T.tint(NC["T_Lineage"]), "lighter: set"), (T.rgb(NC["root"]), "beige: spans"),
                                    (T.rgb(NC["root"]), "lineages"), (T.rgb(T.ABSTAIN), "grey: empty set"))):
        yy = ky + i * 0.13
        if lab != "lineages":
            c.rect(kx, yy - 0.031, 0.062, 0.062, fc=col, z=6)
        c.text(kx + 0.09, yy, lab, size=FS_XS, color=S.INK2)
    c.text(kx, ky + 4 * 0.13 + 0.02, "colours as in b", size=FS_XS, color=S.INK2)
    return y1, y2, s


def panel_f(c, x, y):
    """As make_fig1_v8.panel_f, with a different title."""
    c.title(x, y, "f", "Panel design")
    y1, y2 = y + 0.44, y + 0.98
    ax0, ax1 = x + 0.46, x + 1.06
    for yy, n, who, res in ((y1, 4, "development\npatients", "predicted\nresolution"),
                            (y2, 3, "held-out\npatients", "realised\nresolution")):
        F1.patients(c, x, yy, n)
        c.text(x, yy + 0.155, who, size=FS_LAB, va="center")
        c.arrow(ax0, yy, ax1, yy)
        c.text((ax0 + ax1) / 2, yy - 0.115, "masked to\ncandidate panel", size=FS_XS, ha="center", va="center")
        c.text(ax1 + 0.04, yy, res, size=FS_LAB)
    xm = ax1 + 0.22
    c.arrow(xm, y1 + 0.135, xm, y2 - 0.135, style="<|-|>")
    c.text(xm + 0.04, (y1 + y2) / 2, "compare", size=FS_XS, color=S.INK2)


def main():
    S.setup()
    c = F1.Canvas()
    d = F1.load_field()
    xa, xb, xc, xd = 0.0, 1.45, 2.97, 4.70
    a_bottom = panel_a(c, d, xa, Y0)
    rows = panel_b(c, xb, Y0)
    c_top, c_bottom = panel_c(c, xc, Y0)
    y1, y2, s = panel_d(c, d, xd, Y0)

    # flow of the model: the panel fixes the target, the target groups the calibration cells, the thresholds give sets
    ym = Y0 + 0.95
    arrow_label(c, 1.20, 1.40, ym, "$P$")
    arrow_label(c, 2.73, 2.95, ym, "$g_P$")
    arrow_label(c, 4.47, 4.66, ym, r"$\lambda_c$")
    # ADMIT: everything from the requirement map to the set; the scorer is an input
    yb = 0.07
    c.line([xb, 6.5], [yb, yb], color=S.INK, lw=0.6)
    for xx in (xb, 6.5):
        c.line([xx, xx], [yb, yb + 0.05], color=S.INK, lw=0.6)
    t = c.text((xb + 6.5) / 2, yb, "ADMIT", size=FS_LAB, ha="center", weight="bold")
    t.set_bbox(dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none"))
    # scorer bus: any scorer f is applied to calibration cells (c) and to the deployment cells (d)
    xs0 = 0.42
    xc_in, xd_in = xc + 0.56 + 0.44, xd + 0.40
    c.line([xs0, xs0], [a_bottom + 0.01, BUS], color=S.INK, lw=0.6)
    c.line([xs0, xd_in], [BUS, BUS], color=S.INK, lw=0.6)
    c.arrow(xc_in, BUS, xc_in, c_bottom + 0.28, lw=0.6)
    c.arrow(xd_in, BUS, xd_in, y2 + s + 0.01, lw=0.6)
    t = c.text(1.95, BUS, "any scorer $f$", size=FS_LAB, ha="center")
    t.set_bbox(dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none"))

    y = 2.38
    F1.panel_e(c, d, 0.0, y)
    panel_f(c, 1.86, y)
    F1.panel_g(c, 3.58, y)

    problems = S.check_figure(c.fig)
    F1.OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"Creator": "figures/make_fig1_v9.py (matplotlib)", "Title": STEM}
    c.fig.savefig(F1.OUT_DIR / f"{STEM}.pdf", metadata=meta, dpi=450)
    c.fig.savefig(F1.OUT_DIR / f"{STEM}.png", dpi=300)
    print(f"[out] {F1.OUT_DIR / (STEM + '.pdf')}")
    return c, problems


if __name__ == "__main__":
    main()
