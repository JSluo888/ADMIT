"""Fig 4 (earlier layout; make_fig4_v9.py reuses panels a and b): choosing regions for the full panel (20% budget,
6-plex, uncertainty rule).

    python figures/make_fig4_v8.py

No model runs and no GPU. Real data, all from $ADMIT_RUNS/regions/ (experiments/regions):
a  cords2024_nsclc_imc/regions_xgb_seed0_cells_L6.parquet + regions_xgb_seed0.npz (one split, held-out cores). Same core
   rule as make_fig4_regions.py, fixed before plotting and without labels: cores with >= 200 evaluation cells, split into
   unselected and selected, at quantiles 1/8, 3/8, 5/8, 7/8 of the score within each group. Top row: CL_pooled; second
   row: final_uncert_cond (full panel, selection-conditional thresholds). Each map shows the evaluation cells of the core
   (at most 1,000 per core).
b  summary_regions.json: variants uncert_cond and uncert_pooled, cov_F_sel.
c  summary_regions.json: paired E1_uncert_cond_vs_random_k and B4_uncert_pooled_vs_random_k, corC.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import admit_style as S  # noqa: E402
import make_fig4_regions as F4  # noqa: E402
import tissue as T  # noqa: E402

REG = S.RUNS / "regions"
CORDS = REG / "cords2024_nsclc_imc"
OUT_DIR = S.FIGDIR
STEM = "fig4_v8"
BUDGET = "0.20"
PANEL = "L6"
SCALE_UM = 200                     # Cords IMC: 1 um per pixel

W, H = 6.5, 3.62
S.MIN_PT = 6.0
FS_LET, FS_TTL, FS_LAB, FS_XS = 8.5, 7.0, 6.5, 6.0     # as in Fig 1
COND, POOLED = S.BLUE, S.ORANGE                        # as in Fig 1g
ROW_DY = 0.17                      # vertical offset of the two estimates within a cohort, in rows
AX_DX, AX_W = 0.70, 2.35           # left offset (room for cohort names) and width of the axes in b and c


class Canvas:
    """Overlay in inches, y down (as in Fig 1)."""

    def __init__(self):
        self.fig = S.figure(H, W)
        self.ax = self.fig.add_axes([0, 0, 1, 1], zorder=10)
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)
        self.ax.set_axis_off()

    def text(self, x, y, s, size=FS_LAB, color=S.INK, ha="left", va="center", weight="normal"):
        return self.ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight, linespacing=1.05, zorder=20)

    def title(self, x, y, letter, s):
        self.text(x, y + 0.035, letter, size=FS_LET, weight="bold", va="baseline")
        self.text(x + 0.125, y + 0.035, s, size=FS_TTL, va="baseline")

    def key(self, x, y, entries):
        """One legend row: dot and label in ink."""
        for xx, col, label in entries:
            self.ax.plot([x + xx], [y + 0.012], "o", ms=3.0, color=col, mew=0, zorder=20)
            self.text(x + xx + 0.05, y + 0.035, label, va="baseline")


# ======================================================================= a
def load_cores():
    z = np.load(CORDS / "regions_xgb_seed0.npz", allow_pickle=True)
    nodes = [str(n) for n in z["nodes"]]
    cells = pd.read_parquet(CORDS / "regions_xgb_seed0_cells_L6.parquet",
                            columns=["region", "x", "y", "uL", "sel_uncert", "CL_pooled", "final_uncert_cond"])
    reg = cells.groupby("region").agg(sel=("sel_uncert", "max"), n=("uL", "size"))
    key = {str(k): i for i, k in enumerate(z["region_key"])}
    reg["score"] = [float(z["scores"][0, 0, key[r]]) for r in reg.index]     # panel L6, rule uncert
    reg = reg[reg.n >= 200].sort_values("score")
    pick = []
    for flag in (0, 1):
        sub = reg[reg.sel == flag]
        pick += [sub.index[int(round(q * (len(sub) - 1)))] for q in (1 / 8, 3 / 8, 5 / 8, 7 / 8)]
    cores = [cells[cells.region == r] for r in pick]
    ext = max(max(float(np.ptp(d["x"])), float(np.ptp(d["y"]))) for d in cores)
    return cores, nodes, ext


def core_map(c, d, col, nodes, ext, x, y, s, outline=None):
    a = S.ax_in(c.fig, x, y, s, s)
    cols = np.array([F4.set_colour(F4.decode(m, nodes)) for m in d[col].values])
    a.set_facecolor(T.MAP_BG)
    xx = d["x"].values - 0.5 * (d["x"].min() + d["x"].max()) + ext / 2      # core centred in its tile
    yy = d["y"].values - 0.5 * (d["y"].min() + d["y"].max()) + ext / 2
    a.scatter(xx, yy, s=1.25, c=cols, linewidths=0, rasterized=True)
    a.set_xlim(0, ext)
    a.set_ylim(ext, 0)
    a.set_xticks([])
    a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)
    if outline:
        c.ax.add_patch(Rectangle((x, y), s, s, facecolor="none", edgecolor=outline, lw=0.9, zorder=8))


def panel_a(c, x, y):
    c.title(x, y, "a", "Cords held-out cores")
    cores, nodes, ext = load_cores()
    s = 0.76
    g = (W - 0.01 - 8 * s) / 7           # 0.01 in keeps the last blue outline inside the page
    y1 = y + 0.20
    y2 = y1 + s + 0.11
    c.text(x, y1 - 0.055, "6-plex, ADMIT sets")
    for j, d in enumerate(cores):
        xx = x + j * (s + g)
        selected = j >= 4
        core_map(c, d, "CL_pooled", nodes, ext, xx, y1, s, outline=S.BLUE if selected else None)
        if selected:
            core_map(c, d, "final_uncert_cond", nodes, ext, xx, y2, s, outline=S.BLUE)
            c.ax.add_patch(FancyArrowPatch((xx + s / 2, y1 + s + 0.012), (xx + s / 2, y2 - 0.012), arrowstyle="-|>",
                                           mutation_scale=4.5, color=S.INK, lw=0.5, shrinkA=0, shrinkB=0, zorder=6))
    xb = x + s - 0.04
    c.ax.plot([xb - SCALE_UM / ext * s, xb], [y1 + s - 0.04] * 2, color="white", lw=0.9, solid_capstyle="butt", zorder=9)
    c.text(x + 4 * (s + g) - 0.06, y2 + s / 2, "full panel,\nselected cores", ha="right")


# ======================================================================= b, c
def forest(c, x, y, w, h, rows, xlim, ticks, tick_fmt, xlabel):
    """rows: one (conditional, pooled) pair per cohort, each (estimate, low, high)."""
    a = S.ax_in(c.fig, x, y, w, h)
    for i, pair in enumerate(rows):
        for (v, lo, hi), dy, col in zip(pair, (-ROW_DY, ROW_DY), (COND, POOLED)):
            assert xlim[0] < lo <= hi < xlim[1], f"interval outside the axis: {lo}, {hi}"
            a.plot([lo, hi], [i + dy] * 2, color=col, lw=S.LW_CI, solid_capstyle="butt", zorder=3)
            a.plot([v], [i + dy], "o", ms=3.0, color=col, mew=0, zorder=4)
    a.set_xlim(*xlim)
    a.set_ylim(len(rows) - 0.5, -0.5)
    a.set_yticks(range(len(rows)))
    a.set_yticklabels([label for _, label in F4.COHORTS], fontsize=FS_LAB, color=S.INK)
    a.set_xticks(ticks)
    a.set_xticklabels([tick_fmt(t) for t in ticks], fontsize=FS_XS)
    a.tick_params(axis="x", length=2.0, width=S.LW_AXIS, pad=1.5)
    a.tick_params(axis="y", length=0, pad=3)
    a.spines["left"].set_visible(False)
    a.spines["bottom"].set_linewidth(S.LW_AXIS)
    a.set_xlabel(xlabel, fontsize=FS_LAB, color=S.INK, labelpad=1.5)
    return a


def panel_b(c, data, x, y):
    c.title(x, y, "b", "Coverage in selected regions")
    c.key(x + 1.82, y, [(0, COND, "selection-conditional"), (0.95, POOLED, "pooled")])
    rows = []
    for cohort, _ in F4.COHORTS:
        v = data["cohorts"][cohort]["panels"][PANEL]["variants"]
        pair = []
        for name in ("uncert_cond", "uncert_pooled"):
            m = v[name][BUDGET]["cov_F_sel"]
            pair.append((m["value"], m["ci"][0], m["ci"][1]))
        rows.append(pair)
    a = forest(c, x + AX_DX, y + 0.13, AX_W, 1.16, rows, (0.78, 0.93), [0.80, 0.85, 0.90], lambda t: f"{t:.2f}", "exact coverage")
    a.axvline(0.90, color=S.INK2, lw=0.6, ls=(0, (3, 2)), zorder=2)


def panel_c(c, data, x, y):
    c.title(x, y, "c", "Resolution gain over random")
    c.key(x + 1.62, y, [(0, COND, "conditional (primary)"), (0.95, POOLED, "pooled")])
    rows = []
    for cohort, _ in F4.COHORTS:
        p = data["cohorts"][cohort]["panels"][PANEL]["paired"]
        pair = []
        for name in ("E1_uncert_cond_vs_random_k", "B4_uncert_pooled_vs_random_k"):
            m = p[name][BUDGET]["corC"]
            pair.append((m["diff"], m["ci"][0], m["ci"][1]))
        rows.append(pair)

    def fmt(t):
        return "0" if t == 0 else f"{t:+.2f}".replace("-", "−").replace("+", "")

    a = forest(c, x + AX_DX, y + 0.13, AX_W, 1.16, rows, (-0.06, 0.06), [-0.04, -0.02, 0, 0.02, 0.04], fmt,
               "Δ correct leaf resolution")
    a.axvline(0, color=S.INK2, lw=0.6, zorder=2)


def main():
    S.setup()
    c = Canvas()
    data = json.loads((REG / "summary_regions.json").read_text())
    panel_a(c, 0.0, 0.06)
    y = 2.03
    panel_b(c, data, 0.0, y)
    panel_c(c, data, 3.43, y)

    problems = S.check_figure(c.fig)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"Creator": "figures/make_fig4_v8.py (matplotlib)", "Title": STEM}
    c.fig.savefig(OUT_DIR / f"{STEM}.pdf", metadata=meta, dpi=450)
    c.fig.savefig(OUT_DIR / f"{STEM}.png", dpi=300)
    print(f"[out] {OUT_DIR / (STEM + '.pdf')}\n[out] {OUT_DIR / (STEM + '.png')}")
    return c, problems


if __name__ == "__main__":
    main()
