"""Fig 3: the same window typed under four panels, predicted vs realised resolution, greedy vs comparators.

    python figures/make_fig3_v8.py

No model runs and no GPU. Real data:
a  the 200 um Orion window of Fig 1 under four panels, per-cell ADMIT sets from $ADMIT_FIG_ASSETS/orion_case (colour
   rule, cell map and window shared with make_fig1_v8.py). greedy_tb12/greedy_tb8 are the type-balanced greedy panels
   of experiments/round2 (round2/greedy_tb_lineage_orion.json).
b  Schuerch, predicted vs realised correct leaf resolution per (split, panel, type), bank and cell-weighted greedy panels
   ($ADMIT_RUNS/round3/panel_design). Points are per type, so they are not type-balanced values.
c  realised type-balanced resolution on held-out patients from summary_design.json. Greedy = empty start.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import admit_style as S  # noqa: E402
import make_fig1_v8 as F1  # noqa: E402

PD = S.RUNS / "round3" / "panel_design"
OUT_DIR = S.FIGDIR
STEM = "fig3_v8"

H = 3.85
F1.H = H                               # F1.Canvas reads the module height
PANELS = [("full", "Full 16"), ("greedy_tb12", "Greedy 12"), ("greedy_tb8", "Greedy 8"), ("rand50_s0", "Random 8")]
SIZE = 1.54                            # 616 px at 400 dpi: one image pixel per PNG pixel
GAP = (F1.W - 4 * SIZE) / 3
YLIM = (0, 0.35)
DX = 0.9                               # horizontal offset of the marks at one k, in markers


def load(name):
    return json.loads((PD / name).read_text())


def panel_a(c, y):
    x0, y0, size = F1.ZOOM
    mask = np.array(np.load(F1.FIELD / "mask_crop_full.npy", mmap_mode="r")[y0:y0 + size, x0:x0 + size])
    cells = pd.read_parquet(F1.FIELD / "cells.parquet").set_index("cell_id")
    df = cells.loc[cells.index.intersection(np.unique(mask))]
    bar = 50 / (size * F1.UM_PX) * SIZE
    c.text(0, y, "a", size=F1.FS_LET, weight="bold", va="baseline")
    top = y + 0.05
    for k, (panel, title) in enumerate(PANELS):
        x = k * (SIZE + GAP)
        F1.PANEL = panel               # F1.colours reads the module panel
        c.image(F1.cell_map(mask, F1.colours(df, "admit")), x, top, SIZE)
        c.text(x + SIZE / 2, y, title, size=F1.FS_TTL, ha="center", va="baseline")
    c.scalebar(SIZE - 0.06, top + SIZE - 0.06, bar)      # one bar: all four maps share the scale
    return top + SIZE


def style(ax, xlabel=None, ylabel=None):
    S.style_axes(ax, grid=None)
    ax.tick_params(labelsize=F1.FS_XS)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=F1.FS_LAB, color=S.INK, labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=F1.FS_LAB, color=S.INK, labelpad=2)


def scatter_points():
    pred, real = [], []
    for seed in range(5):
        bank = load(f"design_schurch_seed{seed}_bank.json")["bank"].values()
        greedy = load(f"design_schurch_seed{seed}_greedy_cw.json")["greedy"]
        for e in list(bank) + list(greedy):
            for t, pv in e["pred"].items():
                if not t.startswith("_") and t in e["real"]:
                    pred.append(pv["leafres_correct"])
                    real.append(e["real"][t]["leafres_correct"])
    return np.asarray(pred), np.asarray(real)


def panel_b(c, x, y, ytop, side):
    ag = load("summary_design.json")["schurch"]["agreement"]
    pred, real = scatter_points()
    r = float(np.corrcoef(pred, real)[0, 1])
    assert len(pred) == ag["n_points"] and abs(r - ag["pooled"]["r"]) < 1e-9, (len(pred), r)
    print(f"[b] n={len(pred)} r={r:.4f} mae={np.abs(pred - real).mean():.4f} zero={np.mean(real == 0):.3f} "
          f"r(real>0)={np.corrcoef(pred[real > 0], real[real > 0])[0, 1]:.4f}")
    c.text(x, y, "b", size=F1.FS_LET, weight="bold", va="baseline")
    ax_x = x + 0.36
    c.text(ax_x + side / 2, y, "Schürch, per type", size=F1.FS_TTL, ha="center", va="baseline")
    ax = S.ax_in(c.fig, ax_x, ytop, side, side)
    ax.plot([0, 1], [0, 1], color=S.INK2, lw=0.5, zorder=1)
    ax.scatter(pred, real, s=1.6, color=S.BLUE, alpha=0.25, edgecolors="none", zorder=2, rasterized=True,
               clip_on=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    style(ax, "Predicted resolution", "Realised resolution")


def matched(ax, val, ks):
    for k in ks:
        v = val["k"][str(k)]
        ax.plot([k, k], [v["random_min"], v["random_max"]], color=S.AXIS, lw=1.6, solid_capstyle="butt", zorder=1)
        ax.plot([k - 0.45, k + 0.45], [v["random_mean"]] * 2, color=S.INK2, lw=0.7, solid_capstyle="butt", zorder=2)
        ax.plot(k - DX, v["greedy_empty"], ls="none", marker="o", ms=3.4, mfc=S.BLUE, mec="none", zorder=3)
        ax.plot(k + DX, v["heuristic_empty"], ls="none", marker="D", ms=2.8, mfc="white", mec=S.INK2, mew=0.6, zorder=3)
    ax.set_xticks(ks)
    ax.set_ylim(*YLIM)
    ax.set_yticks([0, 0.1, 0.2, 0.3], ["0", "0.1", "0.2", "0.3"])


def panel_c(c, x, y, ytop, h):
    summ = load("summary_design.json")
    c.text(x, y, "c", size=F1.FS_LET, weight="bold", va="baseline")
    x_s = x + 0.46
    w_s = (F1.W - x_s - 0.18) * 2 / 3      # same scale per marker in both axes
    w_o = w_s / 2
    x_o = F1.W - w_o
    axs = S.ax_in(c.fig, x_s, ytop, w_s, h)
    axo = S.ax_in(c.fig, x_o, ytop, w_o, h)
    matched(axs, summ["schurch"]["validation"], (4, 8, 12, 16, 20, 24))
    matched(axo, summ["orion"]["validation"], (4, 8, 12))
    axs.set_xlim(1.5, 26.5)
    axo.set_xlim(1.5, 14)
    axo.set_yticklabels([])
    style(axs, "Markers", "Realised type-balanced\nresolution")
    axs.yaxis.label.set_linespacing(1.0)
    style(axo, "Markers")
    c.text(x_s + w_s / 2, y, "Schürch", size=F1.FS_TTL, ha="center", va="baseline")
    c.text(x_o + w_o / 2, y, "Orion", size=F1.FS_TTL, ha="center", va="baseline")
    greedy = Line2D([], [], ls="none", marker="o", ms=3.4, mfc=S.BLUE, mec="none")
    rand = (Line2D([], [], ls="none", marker="|", ms=7, mew=1.6, color=S.AXIS),
            Line2D([], [], ls="none", marker="_", ms=5, mew=0.7, color=S.INK2))
    heur = Line2D([], [], ls="none", marker="D", ms=2.8, mfc="white", mec=S.INK2, mew=0.6)
    axs.legend([greedy, rand, heur], ["greedy", "random (range, mean)", "heuristic"], loc="upper left",
               bbox_to_anchor=(0.02, 1.02), fontsize=F1.FS_XS, frameon=False, labelspacing=0.45, handletextpad=0.5,
               borderaxespad=0, borderpad=0, handler_map={tuple: HandlerTuple(ndivide=1, pad=0)})


def main():
    S.setup()
    c = F1.Canvas()
    bottom = panel_a(c, 0.105)
    y = bottom + 0.19
    ytop = y + 0.06
    side = 1.55
    panel_b(c, 0.0, y, ytop, side)
    panel_c(c, 2.16, y, ytop, side)
    S.check_figure(c.fig)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"Creator": "figures/make_fig3_v8.py (matplotlib)", "Title": STEM}
    c.fig.savefig(OUT_DIR / f"{STEM}.pdf", metadata=meta, dpi=450)
    c.fig.savefig(OUT_DIR / f"{STEM}.png", dpi=400)
    print(f"[out] {OUT_DIR / (STEM + '.pdf')}\n[out] {OUT_DIR / (STEM + '.png')}\n[size] {F1.W} x {H} in")


if __name__ == "__main__":
    main()
