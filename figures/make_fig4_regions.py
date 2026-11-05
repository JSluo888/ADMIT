#!/usr/bin/env python3
"""Region-selection figure (earlier draft layout). Fig 1 and Fig 4 import its colour rule (set_colour, decode) and the
cohort list (COHORTS).

    python figures/make_fig4_regions.py

Data: $ADMIT_RUNS/regions/ (experiments/regions: summary_regions.json; Cords seed-0 per-cell parquet).
a  Cords, held-out patients (seed 0): 4 unselected and 4 selected test cores (>= 200 evaluation cells), at quantiles
   1/8, 3/8, 5/8, 7/8 of the uncertainty score within each group (rule fixed before plotting; no labels used); top row 6-plex ADMIT sets, bottom row after imaging the full panel on the
   cores selected at a 20% budget (selection-conditional calibration). Colour = finest member of the set.
b  Coverage in profiled and unprofiled regions: pooled calibration vs selection-conditional calibration.
c  Correct leaf resolution gained over cost-matched random choice (same number of regions), b = 0.20.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import admit_style as S  # noqa: E402
import tissue as T  # noqa: E402

REG = S.RUNS / "regions"
COHORTS = [("cords2024_nsclc_imc", "Cords"), ("danenberg2022_metabric_imc", "Danenberg"),
           ("fischer2023_breast_imc", "Fischer"), ("meyer2025_tnbc_imc", "Meyer"),
           ("neotrip_tnbc_imc", "NeoTRIP"), ("quiche_tnbc_mibi_spain", "QUICHE Spain")]
B = "0.20"
POOLED = S.ORANGE

PARENT = {"Immune": None, "T_Lineage": "Immune", "Helper_T_Cell": "T_Lineage", "Regulatory_T_Cell": "T_Lineage",
          "Cytotoxic_T_Cell": "T_Lineage", "B_Cell": "Immune", "Plasma_Cell": "Immune", "NK_Cell": "Immune",
          "Myeloid": "Immune", "Macrophage_CD163pos": "Myeloid", "Macrophage_CD163neg": "Myeloid",
          "Dendritic_Cell": "Immune", "Neutrophil": "Immune", "Immune_Other": "Immune", "Epithelial": None,
          "Endothelial": None, "Stromal": None}


def path(n):
    out = [n]
    while PARENT[out[-1]] is not None:
        out.append(PARENT[out[-1]])
    return out


def set_colour(members):
    """Fig 2/3 rule: colour of the finest member; lighter tint if ambiguous; beige if lineages differ; grey if empty."""
    if not members:
        return T.rgb(T.ABSTAIN)
    paths = {m: path(m) for m in members}
    minimal = [m for m in members if not any(m != o and m in paths[o] for o in members)]
    if len({paths[m][-1] for m in members}) > 1:
        return T.rgb(T.NODE_COL["root"])
    if len(minimal) == 1:
        return T.rgb(T.NODE_COL[minimal[0]])
    # several minimal members within one lineage: lighter tint of the deepest one
    deepest = max(minimal, key=lambda m: len(paths[m]))
    return T.tint(T.NODE_COL[deepest])


def decode(mask, nodes):
    return [nodes[i] for i in range(len(nodes)) if (int(mask) >> i) & 1]


def resolves_to(members, leaf):
    if not members or leaf not in members:
        return False
    p = set(path(leaf))
    return all(m in p for m in members)


def core_map(ax, d, col, nodes, ext, outline=None):
    cols = np.array([set_colour(decode(m, nodes)) for m in d[col].values])
    ax.set_facecolor(T.MAP_BG)
    ax.scatter(d["x"] - d["x"].min(), d["y"] - d["y"].min(), s=0.55, c=cols, linewidths=0, rasterized=True)
    ax.set_xlim(0, ext); ax.set_ylim(ext, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(outline is not None)
        if outline is not None:
            s.set_color(outline); s.set_linewidth(1.6)


def get(S_, c, variant, key):
    v = S_["cohorts"][c]["panels"]["L6"]["variants"][variant][B][key]
    return v["value"], (v["ci"] or [np.nan, np.nan])


def paired(S_, c, name):
    p = S_["cohorts"][c]["panels"]["L6"]["paired"][name][B]["corC"]
    return p["diff"], p["ci"]


def main():
    S.setup()
    SUM = json.loads((REG / "summary_regions.json").read_text())
    src = S.Sources("fig4_regions")
    fig = S.figure(7.15)

    # ---------------------------------------------------------------- a: qualitative
    S.panel_letter(fig, 0.0, 0.0, "a")
    S.panel_title(fig, 0.22, 0.02, "Where to image the full panel: one held-out set of Cords cores",
                  sub="Four unselected and four selected test cores (blue outline: selected by the uncertainty rule at a 20% budget);\n"
                      "cores chosen at fixed quantiles of the score, without labels. NSCLC, IMC.")
    z = np.load(REG / "cords2024_nsclc_imc" / "regions_xgb_seed0.npz", allow_pickle=True)
    nodes = [str(n) for n in z["nodes"]]
    cells = pd.read_parquet(REG / "cords2024_nsclc_imc" / "regions_xgb_seed0_cells_L6.parquet")
    reg = cells.groupby("region").agg(u=("uL", "mean"), sel=("sel_uncert", "max"), n=("uL", "size"))
    key = {str(k): i for i, k in enumerate(z["region_key"])}
    reg["score"] = [float(z["scores"][0, 0, key[r]]) for r in reg.index]     # panel L6, rule uncert
    reg = reg[reg.n >= 200].sort_values("score")
    pick = []
    for flag in (0, 1):
        sub = reg[reg.sel == flag]
        pick += [sub.index[int(round(q * (len(sub) - 1)))] for q in (1 / 8, 3 / 8, 5 / 8, 7 / 8)]
    ext = max(float((cells[cells.region == r].x.max() - cells[cells.region == r].x.min())) for r in pick)
    ext = max(ext, max(float((cells[cells.region == r].y.max() - cells[cells.region == r].y.min())) for r in pick))
    x0, w, g = 0.62, 0.68, 0.05
    ytop, ybot = 0.82, 1.6
    S.fig_text(fig, 0.05, ytop + w / 2, "6-plex,\nADMIT sets", fontsize=S.FS, color=S.INK, va="center")
    S.fig_text(fig, 0.05, ybot + w / 2, "after full\npanel on\nselected", fontsize=S.FS, color=S.INK, va="center")
    for j, r in enumerate(pick):
        d = cells[cells.region == r]
        sel = bool(reg.loc[r, "sel"])
        xx = x0 + j * (w + g)
        ax = S.ax_in(fig, xx, ytop, w, w)
        core_map(ax, d, "CL_pooled", nodes, ext, outline=S.BLUE if sel else None)
        nr = int(z["region_N"][key[r]])
        S.fig_text(fig, xx + w / 2, ytop - 0.03, f"u {reg.loc[r, 'u']:.2f}\n{nr / 1000:.1f}k cells", fontsize=S.FS,
                   color=S.INK2, ha="center", va="bottom")
        ax = S.ax_in(fig, xx, ybot, w, w)
        core_map(ax, d, "final_uncert_cond", nodes, ext, outline=S.BLUE if sel else None)
        before = np.mean([resolves_to(decode(m, nodes), l) for m, l in zip(d.CL_pooled, d.label)])
        after = np.mean([resolves_to(decode(m, nodes), l) for m, l in zip(d.final_uncert_cond, d.label)])
        S.fig_text(fig, xx + w / 2, ybot + w + 0.03, f"{before:.2f}→{after:.2f}" if sel else f"{before:.2f}",
                   fontsize=S.FS, color=S.INK2 if not sel else S.INK, ha="center", va="top")
        if j == 0:
            ax.plot([ext * 0.08, ext * 0.08 + 200], [ext * 0.93] * 2, color="white", lw=1.0)
    S.fig_text(fig, x0, ybot + w + 0.17, "u: fraction of cells whose 6-plex set does not resolve to a leaf type; the score ranks "
               "expected unresolved cells (u × cells).\nBelow: fraction of cells whose set resolves to their reference "
               "type (before → after the full panel). Scale bar 200 µm.",
               fontsize=S.FS, color=S.INK2)
    src.add("a", REG / "cords2024_nsclc_imc" / "regions_xgb_seed0_cells_L6.parquet",
            "CL_pooled, final_uncert_cond, uL, sel_uncert; npz scores[L6, uncert]", note="seed 0 test cores")
    leg = [("Epithelial", "Epithelial"), ("T_Lineage", "T cell"), ("Cytotoxic_T_Cell", "Cytotoxic T"),
           ("Helper_T_Cell", "Helper T"), ("Regulatory_T_Cell", "Regulatory T"), ("B_Cell", "B cell"),
           ("Myeloid", "Myeloid"), ("Macrophage_CD163pos", "Macroph. CD163+"),
           ("Macrophage_CD163neg", "Macroph. CD163−"), ("Neutrophil", "Neutrophil"), ("Immune", "Immune"),
           ("Immune_Other", "Immune, other"), ("Stromal", "Stromal"), ("Endothelial", "Endothelial"),
           ("root", "set spans lineages")]
    ly = ybot + w + 0.5
    for i, (k, name) in enumerate(leg):
        cx, cy = 0.62 + (i % 6) * 0.97, ly + (i // 6) * 0.15
        S.fig_text(fig, cx, cy, "●", color=T.NODE_COL[k], fontsize=S.FS + 1, va="center")
        S.fig_text(fig, cx + 0.11, cy, name, fontsize=S.FS, color=S.INK2, va="center")
    S.fig_text(fig, 0.62, ly + 0.46, "Lighter tint: ambiguous set within one lineage; light grey: empty set.",
               fontsize=S.FS, color=S.INK2, va="center")

    # ---------------------------------------------------------------- b: validity
    yb = 3.5
    S.panel_letter(fig, 0.0, yb, "b")
    S.panel_title(fig, 0.22, yb + 0.02, "Coverage after choosing regions by uncertainty",
                  sub="Uncertainty rule, 20% budget; 95% patient-cluster intervals; dashed: 0.90 target")
    labels = [lab for _, lab in COHORTS]
    for k, (key, title) in enumerate([("cov_F_sel", "profiled regions (full panel)"),
                                      ("cov_L_unsel", "unprofiled regions (6-plex)")]):
        ax = S.ax_in(fig, 0.85 + k * 1.35, yb + 0.6, 1.2, 1.75)
        for i, (c, _) in enumerate(COHORTS):
            for dy, var, col in [(-0.14, "uncert_pooled", POOLED), (0.14, "uncert_cond", S.BLUE)]:
                v, ci = get(SUM, c, var, key)
                ax.plot(ci, [i + dy] * 2, color=col, lw=S.LW_CI)
                ax.plot([v], [i + dy], "o", ms=3.2, color=col)
        ax.axvline(0.90, color=S.INK2, lw=0.6, ls=(0, (3, 2)))
        ax.set_ylim(len(COHORTS) - 0.5, -0.5); ax.set_xlim(0.78, 0.95)
        ax.set_yticks(range(len(COHORTS)))
        ax.set_yticklabels(labels if k == 0 else [])
        ax.set_xticks([0.80, 0.85, 0.90, 0.95])
        S.style_axes(ax, grid="x", xlabel="exact coverage")
        ax.set_title(title, fontsize=S.FS, color=S.INK, pad=3)
    S.fig_text(fig, 0.45, yb + 2.75, "●", color=POOLED, fontsize=S.FS + 1, va="center")
    S.fig_text(fig, 0.56, yb + 2.75, "pooled calibration (all calibration cells)", fontsize=S.FS, color=S.INK2, va="center")
    S.fig_text(fig, 0.45, yb + 2.9, "●", color=S.BLUE, fontsize=S.FS + 1, va="center")
    S.fig_text(fig, 0.56, yb + 2.9, "selection-conditional calibration (Prop. 2)", fontsize=S.FS, color=S.INK2,
               va="center")
    lo = min(SUM["cohorts"][c]["panels"]["L6"]["paired"]["cond_vs_pooled_uncert"][B]["cov_F_sel"]["diff"] for c, _ in COHORTS)
    hi = max(SUM["cohorts"][c]["panels"]["L6"]["paired"]["cond_vs_pooled_uncert"][B]["cov_F_sel"]["diff"] for c, _ in COHORTS)
    nd = sum(SUM["cohorts"][c]["panels"]["L6"]["paired"]["cond_vs_pooled_uncert"][B]["mean_deficit_F_sel"]["ci"][1] < 0
             for c, _ in COHORTS)
    S.fig_text(fig, 0.45, yb + 3.1, f"Same profiled cells, conditional minus pooled: coverage +{lo:.3f} to +{hi:.3f};\n"
               f"mean type deficit in profiled regions significantly lower in {nd} of {len(COHORTS)} cohorts.", fontsize=S.FS, color=S.INK2,
               va="top")
    src.add("b", REG / "summary_regions.json", "variants.{uncert_pooled,uncert_cond}.0.20.cov_F_sel/cov_L_unsel",
            note="L6 panel")

    # ---------------------------------------------------------------- c: efficiency
    S.panel_letter(fig, 3.55, yb, "c")
    S.panel_title(fig, 3.77, yb + 0.02, "Resolution gained over random choice",
                  sub="Correct leaf resolution: rule minus random choice\nof the same number of regions; 20% budget")
    ax = S.ax_in(fig, 4.45, yb + 0.6, 1.35, 1.75)
    rows = [("E1_uncert_cond_vs_random_k", S.BLUE, "o", "uncertainty, conditional (primary)", True),
            ("voi_cond_vs_random_k", S.BLUE, "D", "expected gain, conditional (secondary)", False),
            ("B4_uncert_pooled_vs_random_k", POOLED, "o", "uncertainty, pooled (under-covers)", False)]
    for i, (c, _) in enumerate(COHORTS):
        for j, (name, col, mk, _, filled) in enumerate(rows):
            d, ci = paired(SUM, c, name)
            y = i + (j - 1) * 0.22
            ax.plot(ci, [y, y], color=col, lw=S.LW_CI)
            ax.plot([d], [y], mk, ms=3.0, color=col, mfc=col if filled else "white", mew=0.8)
    ax.axvline(0, color=S.INK2, lw=0.6)
    ax.set_ylim(len(COHORTS) - 0.5, -0.5); ax.set_yticks(range(len(COHORTS)))
    ax.set_yticklabels(labels)
    ax.set_xlim(-0.06, 0.06); ax.set_xticks([-0.04, 0.0, 0.04])
    for i, (c, _) in enumerate(COHORTS):
        vu = SUM["cohorts"][c]["panels"]["L6"]["variants"]
        cu, cr = vu["uncert_cond"][B]["frac_cells"]["value"], vu["random_k_uncert"][B]["frac_cells"]["value"]
        ax.text(1.03, i, f"{cu:.2f} / {cr:.2f}", transform=ax.get_yaxis_transform(), fontsize=S.FS, color=S.INK2,
                va="center", ha="left")
    ax.text(1.03, -0.55, "cells profiled\nrule / random", transform=ax.get_yaxis_transform(), fontsize=S.FS,
            color=S.INK2, va="bottom", ha="left")
    S.style_axes(ax, grid="x", xlabel="Δ correct leaf resolution")
    lax = S.ax_in(fig, 3.77, yb + 2.68, 0.12, 0.45)
    lax.set_axis_off(); lax.set_xlim(0, 1); lax.set_ylim(2.5, -0.5)
    for j, (_, col, mk, name, filled) in enumerate(rows):
        lax.plot([0.5], [j], mk, ms=3.0, color=col, mfc=col if filled else "white", mew=0.8, clip_on=False)
        S.fig_text(fig, 3.95, yb + 2.68 + 0.45 * (j + 0.5) / 3, name, fontsize=S.FS, color=S.INK2, va="center")
    S.fig_text(fig, 3.77, yb + 3.22, "Pre-specified criterion not met: primary rule positive\nin 2 of 6 cohorts, negative in 1.",
               fontsize=S.FS, color=S.INK, va="top")
    src.add("c", REG / "summary_regions.json", "paired.{E1_uncert_cond_vs_random_k,voi_cond_vs_random_k,"
            "B4_uncert_pooled_vs_random_k}.0.20.corC", note="L6 panel")
    S.finish(fig, "fig4", "fig4_regions", src, check=False)


if __name__ == "__main__":
    main()
