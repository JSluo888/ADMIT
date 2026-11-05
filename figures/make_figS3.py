"""Supplementary Fig S3: resolution vs markers retained (a), coverage and set size over alpha (b), marker value (c).

    python figures/make_figS3.py

Schurch, logistic head; experiments/round3 outputs ($ADMIT_RUNS/round3; target g_P, R = positive | measured).
a  depth of the finest node and correct leaf resolution of ADMIT on the full panel and random 75/50/25% panels.
b  worst type and mean |C| over alpha for ADMIT, ADMIT without M1, marginal LAC on g_P and LAC + filter.
c  marker value from the panel-design runs (round3/panel_design, mean of 5 splits).
Writes $ADMIT_FIG_OUT/supp/S3.pdf (+ .svg, preview .png) and prints the numbers the caption quotes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import admit_style as S  # noqa: E402
import r3data as R  # noqa: E402

PD = R.R3 / "panel_design"
SEEDS = range(5)
GP = r"$g_P$"
D_ARMS = ["admit_m1", "admit", "lac_node_gp", "lac_filt"]
SHORT = {"Macrophage_CD163pos": "Macroph. CD163+", "Immune_Other": "Immune, other"}
MINUS = "−"


def arm_at(run, panel, arm, alpha=R.ALPHA_KEY):
    a = run["panels"][panel]["arms"].get(arm)
    return None if a is None else a.get(alpha)


def _load(name):
    f = PD / name
    return (S.load_json(str(f)), f) if f.exists() else (None, f)


def supp_a(fig, src, x0, y0, w):
    run = R.run("schurch_logistic")
    meta = run["meta"]
    bank = S.RUNS / "schurch" / "schurch_panel_bank.json"
    n_all = len(S.load_json(str(bank))["full_panel"])
    pans = [p for p in run["panels"] if run["panels"][p]["family"] in ("full", "random75", "random50", "random25")]
    rows = [(p, n_all - len(meta["deleted"][p]), arm_at(run, p, "admit_m1")) for p in pans]
    ok = [r for r in rows if r[2] is not None]
    na = [r[0] for r in rows if r[2] is None]
    covs = [r[2]["cov"] for r in ok]
    src.add("a", run["src"], f"{run['field']}.panels[full|rand*].arms.admit_m1['0.10'].{{depth,leafres_correct,cov}}",
            note="target g_P; random-retention bank; panels with no identifiable cells omitted: " + ",".join(na))
    src.add("a", run["meta_src"], "deleted (markers retained = |full_panel| - len(deleted))")
    src.add("a", bank, "full_panel (56 markers)")
    S.panel_letter(fig, x0, y0, "a")
    S.panel_title(fig, x0 + 0.17, y0 + 0.01, "Resolution follows the markers retained",
                  f"Schürch, ADMIT, {len(run['splits'])} patient splits; dots: random panels, line: mean.\n"
                  f"Exact coverage ({GP}) stays {min(covs):.3f} to {max(covs):.3f} on every panel shown.")
    ax1 = S.ax_in(fig, x0 + 0.45, y0 + 0.52, w - 0.55, 0.80)
    ax2 = S.ax_in(fig, x0 + 0.45, y0 + 1.44, w - 0.55, 0.80)
    rng = np.random.default_rng(5)
    for ax, key, lab, yl in ((ax1, "depth", "Finest depth in C", (0, 3.05)),
                             (ax2, "leafres_correct", "Correctly leaf-\nresolved fraction", (0, 0.42))):
        xs = np.array([r[1] for r in ok], float)
        ys = np.array([r[2][key] for r in ok], float)
        jit = (rng.random(len(xs)) - 0.5) * 2.2 * (xs < n_all)
        lv = sorted(set(xs))
        ax.plot(lv, [ys[xs == v].mean() for v in lv], color=S.BLUE, lw=S.LW_DATA, zorder=2)
        ax.scatter(xs + jit, ys, s=10, color=S.BLUE, edgecolors="white", linewidths=0.4, zorder=3, clip_on=False)
        ax.set_xlim(8, 60)
        ax.set_ylim(*yl)
        ax.set_xticks([14, 28, 42, 56])
        S.style_axes(ax, grid="y", ylabel=lab)
        ax.yaxis.label.set_linespacing(1.0)
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_xticklabels([])
    for y, t in ((1, "lineage"), (3, "subtype")):
        ax1.text(60, y + 0.03, t, ha="right", va="bottom", fontsize=S.FS, color=S.MUTED)
    ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax2.set_xlabel(f"Markers retained (of {n_all})", fontsize=S.FS, labelpad=2)
    low = [r for r in ok if r[2]["depth"] < 1.0]
    if low:
        xm = np.mean([r[1] for r in low])
        ax1.annotate("no CD45 isoform, CD31 or aSMA:\nonly epithelial cells identifiable",
                     xy=(xm + 1.6, 0.52), xytext=(31, 0.12), fontsize=S.FS, color=S.INK2, ha="left", va="bottom",
                     linespacing=1.0, arrowprops=dict(arrowstyle="-", color=S.INK2, lw=0.5, shrinkA=1, shrinkB=2))
    return min(covs), max(covs)


def supp_b(fig, src, x0, y0, w):
    run = R.run("schurch_logistic")
    fam = run["families"]["all_deletions"]
    alphas = [float(a) for a in run["alphas"]]
    akeys = [f"{a:.2f}" for a in alphas]
    nom = [1 - a for a in alphas]
    S.panel_letter(fig, x0, y0, "b")
    S.panel_title(fig, x0 + 0.17, y0 + 0.01, "Coverage and set size over α",
                  f"Schürch, mean over {fam['n_panels']} deletion panels; scored on {GP}")
    aw = w - 0.75
    ax1 = S.ax_in(fig, x0 + 0.62, y0 + 0.45, aw, 0.72)
    ax2 = S.ax_in(fig, x0 + 0.62, y0 + 1.30, aw, 0.72)
    for arm in D_ARMS:
        lab, col, mk = S.ARMS[arm]
        a = fam["arms"].get(arm)
        if not a:
            continue
        ks = [k for k in akeys if a.get(k)]
        xs = [1 - float(k) for k in ks]
        z = 6 if arm == "admit_m1" else 3
        for ax, f in ((ax1, "worst_cov"), (ax2, "size")):
            ax.plot(xs, [a[k][f] for k in ks], color=col, lw=S.LW_DATA, marker=mk,
                    ms=3.4 if arm == "admit_m1" else 4.2, mec="white", mew=0.3, zorder=z, clip_on=False)
    ax1.plot([0.78, 0.97], [0.78, 0.97], color=S.INK2, lw=0.7, zorder=1)
    for ax in (ax1, ax2):
        ax.set_xlim(0.78, 0.97)
        ax.set_xticks(nom)
        ax.set_xticklabels([f"{x:.2f}" for x in nom])
    ax1.set_xticklabels([])
    ax1.set_ylim(0, 1.0)
    ax1.set_yticks([0, 0.5, 1.0])
    S.style_axes(ax1, grid="y", ylabel="Worst type")
    ax1.text(0.785, 0.80, "target", fontsize=S.FS, color=S.INK2, ha="left", va="bottom")
    ax2.set_ylim(0, 6)
    ax2.set_yticks([0, 2, 4, 6])
    S.style_axes(ax2, grid="y", xlabel="Nominal coverage 1−α", ylabel="Mean |C|")
    S.fig_legend(fig, x0 + 0.17, y0 + 2.42, S.legend_handles(D_ARMS, target=False), ncol=2, columnspacing=0.6)
    src.add("b", run["src"], f"{run['field']}.families.all_deletions.arms[admit_m1|admit|lac_node_gp|lac_filt][alpha]."
            "{worst_cov,size}; .n_panels; runs.schurch_logistic.alphas",
            note="target g_P; worst_cov = mean over panels of the per-panel worst-type coverage")


def supp_c(fig, src, x0, y0, w):
    """Marker value from the panel-design runs: delta_full_minus_removed[marker][type].leafres_correct, mean of 5 splits."""
    per = []
    for k in SEEDS:
        d, f = _load(f"design_schurch_seed{k}_bank.json")
        if d is not None:
            per.append(d["marker_value"]["delta_full_minus_removed"])
    markers = sorted(set.intersection(*[set(p) for p in per]))
    mv = {}
    for m in markers:
        types = set.intersection(*[{t for t in p[m] if not t.startswith("_")} for p in per])
        mv[m] = {t: float(np.mean([p[m][t]["leafres_correct"] for p in per])) for t in types}
    types = S.ct_sorted({t for m in mv.values() for t in m})
    mx = {m: max(abs(v) for v in row.values()) for m, row in mv.items()}
    sel = sorted([m for m in mv if mx[m] >= 0.05], key=lambda m: -mx[m])[:18]

    def key(m):
        row = mv[m]
        best = max(types, key=lambda t: row.get(t, 0))
        return (types.index(best), -row.get(best, 0))
    sel.sort(key=key)
    M = np.array([[mv[m].get(t, np.nan) for t in types] for m in sel])
    src.add("c", PD / "design_schurch_seed{0-4}_bank.json",
            "marker_value.delta_full_minus_removed[marker][type].leafres_correct",
            note=f"panel design (frozen-map ADMIT-M1, R = positive | measured): predicted(full) - "
                 f"predicted(full minus marker), cross-fitted on calibration patients, mean of {len(per)} splits; "
                 "rows: markers with max |value| >= 0.05")
    S.panel_letter(fig, x0, y0, "c")
    S.panel_title(fig, x0 + 0.17, y0 + 0.01, "What each marker buys, per cell type",
                  "Predicted loss in correct leaf resolution when a marker is removed\n"
                  f"(percentage points; calibration patients only, mean of {len(per)} splits)")
    cw, rh = 0.150, 0.083
    gx, gy = x0 + 0.50, y0 + 1.04
    ax = S.ax_in(fig, gx, gy, cw * len(types), rh * len(sel))
    vmax = 0.5
    cmap = S.diverging_cmap()
    ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest",
              extent=(0, len(types), len(sel), 0))
    ax.set_xlim(0, len(types))
    ax.set_ylim(len(sel), 0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks(np.arange(len(sel)) + 0.5)
    ax.set_yticklabels([m.replace("beta-", "β-") for m in sel], fontsize=S.FS, color=S.INK)
    ax.tick_params(axis="y", length=0, pad=2)
    for j, t in enumerate(types):
        ax.text(j + 0.45, -0.3, SHORT.get(t, S.CT_LABEL[t]), rotation=45, rotation_mode="anchor", ha="left",
                va="center", fontsize=S.FS, color=S.INK)
    for i in range(len(sel)):
        for j in range(len(types)):
            v = M[i, j]
            if np.isfinite(v) and v >= 0.10:
                ax.text(j + 0.5, i + 0.55, f"{100 * v:.0f}", ha="center", va="center", fontsize=S.FS,
                        color="white" if abs(v) >= 0.3 else S.INK)
    for j in range(1, len(types)):
        ax.axvline(j, color="white", lw=0.6)
    for i in range(1, len(sel)):
        ax.axhline(i, color="white", lw=0.6)
    cax = S.ax_in(fig, gx + cw * len(types) + 0.10, gy + 0.22, 0.07, rh * len(sel) - 0.44)
    grad = np.linspace(vmax, -vmax, 256)[:, None]
    cax.imshow(grad, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto", extent=(0, 1, -vmax, vmax))
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.set_yticks([-0.5, 0, 0.5])
    cax.set_yticklabels([MINUS + "50", "0", "50"])
    for sp in cax.spines.values():
        sp.set_visible(False)
    cax.tick_params(axis="y", length=2, pad=1)
    cax.text(0.0, 1.04, "lost", transform=cax.transAxes, ha="left", va="bottom", fontsize=S.FS, color=S.INK2)
    cax.text(0.0, -0.09, "gained", transform=cax.transAxes, ha="left", va="top", fontsize=S.FS, color=S.INK2)
    return {m: mv[m] for m in sel}, gy + rh * len(sel)


def supplement():
    src = S.Sources("S3")
    fig = S.figure(5.62)
    cov_rng = supp_a(fig, src, 0.0, 0.0, 3.0)
    supp_b(fig, src, 3.45, 0.0, 3.05)
    mv, _ = supp_c(fig, src, 0.0, 2.85, 3.3)
    S.finish(fig, "supp", "S3", src)
    return cov_rng, mv


def main():
    S.setup()
    cov_rng, mv = supplement()
    print("[S3] coverage range", cov_rng)
    for m, t in (("CD8", "Cytotoxic_T_Cell"), ("PanCK", "Epithelial"), ("CD31", "Endothelial"), ("CD34", "Endothelial"),
                 ("FOXP3", "Helper_T_Cell"), ("FOXP3", "Regulatory_T_Cell")):
        if m in mv:
            print(f"[S3] marker value {m} -> {t}: {100 * mv[m].get(t, float('nan')):.1f} pp")


if __name__ == "__main__":
    main()
