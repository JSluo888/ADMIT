"""Supplementary Fig S1: label moves under single-marker deletions (a) and per-type coverage (b).

    python figures/make_figS1.py

Reads the experiments/round3 summaries ($ADMIT_RUNS/round3; target g_P, requirement map R = positive | measured).
a  Schurch, logistic head: label -> g_P for each single-deletion panel (first split), share of cells relabelled.
b  worst-type exact coverage per panel for ADMIT, marginal LAC on g_P and LAC + filter (Schurch logistic and XGBoost,
   Orion, Cords); strata significantly below 0.90; per-type coverage on the full Schurch panel.
Writes $ADMIT_FIG_OUT/supp/S1.pdf (+ .svg, preview .png).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import admit_style as S  # noqa: E402
import r3data as R  # noqa: E402

D_ARMS = ["admit_m1", "lac_node_gp", "lac_filt"]
GP = r"$g_P$"
TARGET = 0.90
COV_TICKS = [0, 0.25, 0.5, 0.75, 0.9]
COV_TICKLABELS = ["0", "0.25", "0.50", "0.75", "0.90"]
EMPTY = "∅"
CODE = {"T_Lineage": "T", "Immune": "Im", "Myeloid": "My"}
SHORT = {"Macrophage_CD163pos": "Macroph. CD163+", "Immune_Other": "Immune, other"}


def unique_panels(run, meta):
    alias = meta.get("panel_alias", {})
    return [p for p in run["panels"] if p not in alias]


def arm_at(run, panel, arm):
    a = run["panels"][panel]["arms"].get(arm)
    return None if a is None else a.get(R.ALPHA_KEY)


def cov_axis(ax, ylabel=None):
    ax.set_ylim(0, 1.03)
    ax.set_yticks(COV_TICKS)
    ax.set_yticklabels(COV_TICKLABELS)
    ax.spines["left"].set_bounds(0, 1.0)
    S.style_axes(ax, grid="y", ylabel=ylabel)
    ax.axhline(1.0, color=S.GRID, lw=S.LW_GRID, zorder=0)


def strip(ax, run, panels, arms, key, rng, width=0.30, base=0.0, s=7):
    """Dots per panel (jittered) and a median bar, one column per arm."""
    n = None
    for k, arm in enumerate(arms):
        lab, col, mk = S.ARMS[arm]
        v = [arm_at(run, p, arm)[key] for p in panels
             if arm_at(run, p, arm) is not None and arm_at(run, p, arm)[key] is not None]
        v = np.asarray(v, float)
        n = len(v)
        xs = base + k + (rng.random(len(v)) - 0.5) * width * 2
        ax.scatter(xs, v, s=s, marker=mk, color=col, edgecolors="white", linewidths=0.3, zorder=3,
                   clip_on=False)
        med = float(np.median(v))
        ax.plot([base + k - 0.38, base + k + 0.38], [med, med], color=S.INK, lw=1.0, zorder=4)
    return n


def sig_count(run, panel_or_family, arm):
    if panel_or_family == "all_deletions":
        a = run["families"]["all_deletions"]["arms"][arm][R.ALPHA_KEY]
        return a["n_sig_below"], a["n_types"]
    a = arm_at(run, panel_or_family, arm)
    return len(a["sig_below"]), a["n_types"]


def supp_panel_a(fig, src, x0, y0):
    """Label -> g_P per single-deletion panel (frozen support map of the first split)."""
    meta = S.load_json(str(R.META["schurch_logistic"]))
    lc = meta["label_counts"]
    total = sum(lc.values())
    rep = meta["strata_reports"]["0"]
    singles = [p for p, f in meta["families"].items() if f == "single"]
    rows, still = [], []
    for p in singles:
        mp = rep[p]["g_P"]
        moved = {k: v for k, v in mp.items() if v != k}
        (rows if moved else still).append((p, moved))
    types = S.ct_sorted(lc.keys())
    rows.sort(key=lambda r: (min(types.index(t) for t in r[1]), -len(r[1])))
    src.add("a", R.META["schurch_logistic"], "strata_reports['0'][del_*].g_P; label_counts; families; n_cells; "
            "n_patients",
            note="label -> g_P per single-deletion panel, first split (frozen support map; R = positive | "
                 "measured). Identical across splits except the rare T_Lineage label (189 cells), which some splits' "
                 "frozen maps merge to Immune or leave without a supported target; % relabelled = label_counts of "
                 "moved types / all labelled cells")
    S.panel_letter(fig, x0, y0, "a")
    S.panel_title(fig, x0 + 0.17, y0 + 0.01, "One missing marker forces labels up the tree",
                  f"Schürch CODEX, {meta['n_patients']} patients, {meta['n_cells']:,} cells; single-marker deletions; "
                  f"target {GP} (first split)")
    cw, rh = 0.17, 0.088
    gx, gy = x0 + 0.47, y0 + 0.98
    nr = len(rows)
    ax = S.ax_in(fig, gx, gy, cw * len(types), rh * nr)
    ax.set_xlim(0, len(types))
    ax.set_ylim(nr, 0)
    ax.set_axis_off()
    for j, t in enumerate(types):
        ax.text(j + 0.45, -0.3, SHORT.get(t, S.CT_LABEL[t]), rotation=45, rotation_mode="anchor", ha="left",
                va="center", fontsize=S.FS, color=S.INK)
    for i, (p, moved) in enumerate(rows):
        ax.text(-0.3, i + 0.5, "−" + p[4:], ha="right", va="center", fontsize=S.FS, color=S.INK)
        for j, t in enumerate(types):
            if t in moved:
                tgt = moved[t]
                fc = S.RED_FILL2 if tgt is None else S.RED_FILL
                ax.add_patch(S.Rectangle((j + 0.05, i + 0.07), 0.9, 0.86, facecolor=fc, edgecolor="none", zorder=1))
                if tgt is None:
                    ax.text(j + 0.5, i + 0.52, EMPTY, ha="center", va="center", fontsize=S.FS, color=S.INK,
                            zorder=2, fontfamily="DejaVu Sans")
                else:
                    ax.text(j + 0.5, i + 0.55, CODE.get(tgt, tgt[:1]), ha="center", va="center", fontsize=S.FS,
                            color=S.INK, zorder=2)
            else:
                ax.add_patch(S.Rectangle((j + 0.05, i + 0.07), 0.9, 0.86, facecolor="#f7f7f5", edgecolor="none",
                                         zorder=1))
    for j in range(1, len(types)):
        if (types[j - 1], types[j]) in {("Regulatory_T_Cell", "NK_Cell"), ("NK_Cell", "B_Cell"),
                                        ("Plasma_Cell", "Myeloid"), ("Immune_Other", "Epithelial")}:
            ax.plot([j, j], [0, nr], color=S.AXIS, lw=0.5, zorder=3)
    bx = S.ax_in(fig, gx + cw * len(types) + 0.08, gy, 0.40, rh * nr)
    fr = [sum(lc[t] for t in moved) / total for _, moved in rows]
    bx.barh(np.arange(nr) + 0.5, fr, height=0.62, color=S.RED, edgecolor="none")
    bx.set_ylim(nr, 0)
    bx.set_xlim(0, 0.25)
    for i, f in enumerate(fr):
        bx.text(f + 0.01, i + 0.5, f"{100 * f:.0f}" if f >= 0.005 else "<1", va="center", ha="left", fontsize=S.FS,
                color=S.INK2)
    bx.set_yticks([])
    bx.set_xticks([])
    for sp in ("left", "bottom"):
        bx.spines[sp].set_visible(False)
    bx.text(0.004, -0.3, "% cells relabelled", rotation=45, rotation_mode="anchor", ha="left", va="center",
            fontsize=S.FS, color=S.INK)
    fy = gy + rh * nr + 0.08
    lx = x0 + 0.05
    for k, fc in enumerate([S.RED_FILL, S.RED_FILL2]):
        sw = S.ax_in(fig, lx, fy + 0.14 * k + 0.012, 0.13, 0.085)
        sw.set_axis_off()
        sw.add_patch(S.Rectangle((0, 0), 1, 1, facecolor=fc, edgecolor="none"))
    sw.text(0.5, 0.45, EMPTY, ha="center", va="center", fontsize=S.FS, fontfamily="DejaVu Sans",
            transform=sw.transAxes)
    S.fig_text(fig, lx + 0.18, fy, "label unsupported; new target: T = T cell, Im = Immune, My = Myeloid",
               fontsize=S.FS, color=S.INK2)
    S.fig_text(fig, lx + 0.18, fy + 0.14, "no admissible ancestor: the cell is unidentifiable", fontsize=S.FS,
               color=S.INK2)
    S.fig_text(fig, lx, fy + 0.29, ", ".join("−" + p[4:] for p, _ in still)
               + " move no type (any one CD45 isoform suffices).", fontsize=S.FS, color=S.INK2)


def supp_panel_d(fig, src, x0, y0):
    """Worst type per panel, strata below 0.90 and per-type coverage (target g_P; ADMIT, marginal LAC on g_P, LAC + filter)."""
    keys = [("Sch. log.", "schurch_logistic"), ("Sch. XGB", "schurch_xgb"), ("Orion", "orion_lopo13"),
            ("Cords", "cords2024_nsclc_imc")]
    runs = {k: R.run(k) for _, k in keys}
    S.panel_letter(fig, x0, y0, "b")
    S.panel_title(fig, x0 + 0.17, y0 + 0.01, "Per-type calibration reduces undercoverage",
                  f"Scored on {GP}. Whiskers: 95% CI, patient-cluster bootstrap. Significantly below: CI upper bound "
                  "< 0.90.\nTypes and strata with ≥100 pooled test cells.")
    S.fig_legend(fig, x0 + 3.45, y0 + 0.0, S.legend_handles(D_ARMS, target=False))
    S.fig_text(fig, x0 + 0.42, y0 + 0.50, "Worst type per panel; bar: median", fontsize=S.FS, color=S.INK)
    lw_, lh_ = 2.32, 0.92
    lax = S.ax_in(fig, x0 + 0.42, y0 + 0.70, lw_, lh_)
    rng = np.random.default_rng(11)
    gap = 0.8
    for g, (gname, k) in enumerate(keys):
        rr = runs[k]
        base = g * (len(D_ARMS) + gap)
        strip(lax, rr, unique_panels(rr, rr["meta"]), D_ARMS, "worst_cov", rng, width=0.26, base=base, s=5)
        lax.text(base + 1.0, -0.05, gname, ha="center", va="top", fontsize=S.FS, color=S.INK,
                 transform=lax.get_xaxis_transform())
        src.add("d", rr["src"], f"{rr['field']}.panels[*].arms[admit_m1|lac_node_gp|lac_filt]['0.10'].worst_cov; "
                f"{rr['field']}.families.all_deletions.arms[...]['0.10'].{{n_sig_below,n_types}}"
                + ("; .panels.full.arms[...]['0.10'].{sig_below,n_types}" if k.startswith("schurch") else ""),
                note=f"{gname}: target g_P; unique panels incl. full; bar = median")
        src.add("d", rr["meta_src"], "panel_alias")
    S.target_line(lax, TARGET)
    lax.set_xlim(-0.6, len(keys) * len(D_ARMS) + (len(keys) - 1) * gap - 0.4)
    cov_axis(lax, ylabel="Exact coverage, worst type")
    lax.set_xticks([])
    lax.spines["bottom"].set_visible(False)
    ty = y0 + 0.70 + lh_ + 0.19
    cols = [("full panel: types", [("log.", "schurch_logistic", "full"), ("XGB", "schurch_xgb", "full")]),
            ("all deletion panels: strata", [("log.", "schurch_logistic", "all_deletions"),
                                             ("XGB", "schurch_xgb", "all_deletions"),
                                             ("Orion", "orion_lopo13", "all_deletions"),
                                             ("Cords", "cords2024_nsclc_imc", "all_deletions")])]
    S.fig_text(fig, x0 + 0.05, ty, "Significantly below 0.90", fontsize=S.FS, color=S.INK, fontweight="bold")
    cx = x0 + 0.42
    widths = [0.32, 0.32] + [0.46] * 4
    xs = np.cumsum([cx] + widths)[:-1] + np.array(widths) / 2
    S.fig_text(fig, (xs[0] + xs[1]) / 2, ty + 0.13, cols[0][0], ha="center", fontsize=S.FS, color=S.INK2)
    S.fig_text(fig, (xs[2] + xs[-1]) / 2, ty + 0.13, cols[1][0], ha="center", fontsize=S.FS, color=S.INK2)
    flat = cols[0][1] + cols[1][1]
    for k, (lab, _, _) in enumerate(flat):
        S.fig_text(fig, xs[k], ty + 0.25, lab, ha="center", fontsize=S.FS, color=S.INK2)
    for r_, arm in enumerate(D_ARMS):
        yy = ty + 0.38 + 0.125 * r_
        lab, col, mk = S.ARMS[arm]
        kax = S.ax_in(fig, x0 + 0.17, yy - 0.012, 0.1, 0.1)
        kax.set_axis_off(), kax.set_xlim(0, 1), kax.set_ylim(0, 1)
        kax.scatter([0.5], [0.5], s=12, marker=mk, color=col, edgecolors="white", linewidths=0.3)
        for k, (_, rk, pf) in enumerate(flat):
            n, m = sig_count(runs[rk], pf, arm)
            S.fig_text(fig, xs[k], yy, f"{n}/{m}", ha="center", fontsize=S.FS, color=S.INK,
                       fontweight="bold" if n == 0 else "normal")
    # right: each type, full panel, logistic head
    run = runs["schurch_logistic"]
    full = {arm: arm_at(run, "full", arm) for arm in D_ARMS}
    types = S.ct_sorted({t for a in full.values() for t, s in a["strata"].items() if s["n"] >= 100})
    n_tot = full["admit_m1"]["n_ident"]
    rx = x0 + 3.18
    S.fig_text(fig, rx, y0 + 0.50, "Each type, Schürch full panel, logistic; n: k test cells; bold: rare",
               fontsize=S.FS, color=S.INK)
    rax = S.ax_in(fig, rx, y0 + 0.70, 6.45 - rx, lh_)
    off = {a: (k - 1) * 0.24 for k, a in enumerate(D_ARMS)}
    for arm in D_ARMS:
        lab, col, mk = S.ARMS[arm]
        st = full[arm]["strata"]
        for j, t in enumerate(types):
            if t not in st or st[t]["n"] < 100:
                continue
            e = st[t]
            rax.vlines(j + off[arm], e["ci"][0], e["ci"][1], color=col, lw=S.LW_CI, zorder=2)
            rax.scatter([j + off[arm]], [e["cov"]], s=9, marker=mk, color=col, edgecolors="white", linewidths=0.3,
                        zorder=3, clip_on=False)
    S.target_line(rax, TARGET)
    rax.set_xlim(-0.6, len(types) - 0.4)
    cov_axis(rax, ylabel="Exact coverage")
    rax.set_xticks(range(len(types)))
    rax.set_xticklabels([SHORT.get(t, S.CT_LABEL[t]) for t in types], rotation=40, ha="right",
                        rotation_mode="anchor", fontsize=S.FS)
    rax.tick_params(axis="x", length=0, pad=10.5)
    for j, t in enumerate(types):
        n = max(full[a]["strata"].get(t, {}).get("n", 0) for a in D_ARMS)
        rare = n < 0.01 * n_tot
        rax.text(j, -0.035, f"{n / 1e3:.1f}" if n < 1e4 else f"{n / 1e3:.0f}", ha="center", va="top", fontsize=S.FS,
                 color=S.INK if rare else S.INK2, fontweight="bold" if rare else "normal",
                 transform=rax.get_xaxis_transform())
        if rare:
            rax.get_xticklabels()[j].set_fontweight("bold")
    rax.text(-0.06, -0.035, "n (k)", ha="right", va="top", fontsize=S.FS, color=S.INK2, transform=rax.transAxes)
    src.add("d", run["src"], f"{run['field']}.panels.full.arms[admit_m1|lac_node_gp|lac_filt]['0.10'].strata[*]."
            "{n,cov,ci}; .n_ident",
            note="target g_P: every arm has the same strata (no extra merged group); CI = patient-cluster "
                 "bootstrap; n = max over arms of stratum n")


def supplement_markermap():
    src = S.Sources("S1")
    fig = S.figure(5.76)
    supp_panel_a(fig, src, 0.0, 0.0)
    supp_panel_d(fig, src, 0.0, 2.95)
    S.finish(fig, "supp", "S1", src)


def main():
    S.setup()
    supplement_markermap()


if __name__ == "__main__":
    main()
