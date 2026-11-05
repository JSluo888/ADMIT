"""Fig 2: forced labels on one field, coverage by panel group, deficit and set size by cohort.

    python figures/make_fig2_v8.py

No model runs and no GPU. Real data only (experiments/round3 outputs; requirement map R = positive | measured):
a     the held-out Orion field of Fig 1 under one random 50% panel ($ADMIT_FIG_ASSETS/orion_case); colour rule of Fig 1
b     Schurch, logistic head (summary_round3.json): pooled exact coverage per unique panel; pilot panels left out; a
      panel with no identifiable cells has no value and is left out
c, d  ten cohorts, XGBoost head, one Schurch column (summary_round3.json, cohorts/*/summary_within_xgb.json), checked
      against paired.tsv; deficit and set size are the all-deletions family at alpha 0.10; columns sorted by the
      deficit of marginal LAC
The numbers the caption quotes are printed at the end of the run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import admit_style as S  # noqa: E402
import make_fig1_v8 as F1  # noqa: E402  (shares its colour rule and cell map; sets S.MIN_PT = 6)
import r3data as R  # noqa: E402
import tissue as T  # noqa: E402

PANEL = "rand50_s0"                     # 8 of the 16 Orion markers missing
F1.PANEL = PANEL                        # F1.colours reads the panel from this name
SCHURCH_B = "schurch_logistic"          # head of panel b; schurch_xgb is the one-line alternative
OUT_DIR = S.FIGDIR
STEM = "fig2_v8"

W, H = 6.5, 3.80
FS_LET, FS_TTL, FS_LAB, FS_XS = F1.FS_LET, F1.FS_TTL, F1.FS_LAB, F1.FS_XS
LINE = F1.LINE

ADMIT = ("admit_m1", S.BLUE, "o", "ADMIT")
FULLCAL = ("patch_src_backstop", S.ORANGE, "s", "Full-panel calibration")
LAC = ("lac_node_gp", S.AQUA, "D", "Marginal LAC")

GROUPS = [("full", "full"), ("single", "single"), ("lineage_group_extension", "group"), ("random75", "75%"),
          ("random50", "50%"), ("random25", "25%")]
COHORTS = [("Schürch", "schurch_xgb"), ("Orion", "orion_lopo13"), ("Cords", "cords2024_nsclc_imc"),
           ("Danenberg", "danenberg2022_metabric_imc"), ("Fischer", "fischer2023_breast_imc"),
           ("Meyer", "meyer2025_tnbc_imc"), ("QUICHE\nSpain", "quiche_tnbc_mibi_spain"),
           ("QUICHE\nStanford", "quiche_tnbc_mibi_stanford"), ("NeoTRIP", "neotrip_tnbc_imc"),
           ("Launonen", "launonen2022_hgsc_tcycif")]


# ======================================================================= overlay canvas, inches, y down (as Fig 1)
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

    def title(self, x, y, letter, s):
        self.text(x, y, letter, size=FS_LET, weight="bold")
        self.text(x + 0.125, y, s, size=FS_TTL)

    def swatch(self, x, y, col, label):
        self.ax.add_patch(S.Rectangle((x, y - 0.031), 0.062, 0.062, facecolor=col, edgecolor="none", zorder=6))
        return self.text(x + 0.09, y, label, size=FS_XS, color=S.INK2)

    def key(self, x, y, arm):
        _, col, mk, label = arm
        self.ax.plot([x], [y], ls="none", marker=mk, ms=3.2, mfc=col, mec="white", mew=0.25, zorder=20)
        self.text(x + 0.06, y, label, size=FS_XS, color=S.INK2)


def stats(run, fam, arm):
    return run["families"][fam]["arms"][arm][R.ALPHA_KEY]


def thin_axes(ax):
    ax.tick_params(axis="both", which="both", labelsize=FS_XS, length=2.0, width=S.LW_AXIS, pad=1.5)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(S.LW_AXIS)
        ax.spines[side].set_color(S.AXIS)
    ax.yaxis.grid(True, color=S.GRID, lw=S.LW_GRID, zorder=0)
    ax.set_axisbelow(True)


# ======================================================================= a
def panel_a(c, x, y, s, g=0.03):
    c.title(x, y, "a", "Output, 8 markers missing")
    df = pd.read_parquet(F1.FIELD / "cells.parquet").set_index("cell_id")
    mask = np.load(F1.FIELD / "mask_crop_2x.npy")
    field_um = mask.shape[1] * 0.65
    top = y + 0.21
    for i, (kind, label) in enumerate((("reference", "Reference"), ("forced", "Forced label"), ("admit", "ADMIT set"))):
        xx = x + i * (s + g)
        img = T.block_mean(F1.cell_map(mask, F1.colours(df, kind)), 2)
        a = S.ax_in(c.fig, xx, top, s, s)
        a.imshow((img * 255 + 0.5).astype(np.uint8), interpolation="none")
        a.set_axis_off()
        c.text(xx, top - 0.055, label)
        bar = 200 / field_um * s
        c.line([xx + s - 0.05 - bar, xx + s - 0.05], [top + s - 0.05] * 2, color="white", lw=1.0, z=9)
    ky = top + s + 0.085
    kx = x
    for col, label, step in ((S.RED, "red: unsupported", 0.86), (T.tint(F1.NC["T_Lineage"]), "lighter: set", 0.68),
                             (T.rgb(T.ABSTAIN), "grey: empty set", 0.82)):
        c.swatch(kx, ky, col, label)
        kx += step
    c.text(kx, ky, "colours as in Fig. 1", size=FS_XS, color=S.INK2)
    c.text(x + 3 * s + 2 * g, ky, "bars: 200 µm", size=FS_XS, color=S.INK2, ha="right")
    return df


# ======================================================================= b
def panel_b(c, x, y, w, h):
    c.title(x, y, "b", "Exact coverage, Schürch")
    run = R.run(SCHURCH_B)
    alias = run["meta"].get("panel_alias", {})
    px, py = x + 0.24, y + 0.21
    ax = S.ax_in(c.fig, px, py, w - 0.24, h)
    rng = np.random.default_rng(11)
    arms = (ADMIT, FULLCAL, LAC)
    offs = (-0.27, 0.0, 0.27)
    for g, (fam, _) in enumerate(GROUPS):
        panels = [p for p in run["panels"] if p not in alias and run["panels"][p]["family"] == fam]
        for (arm, col, mk, _), off in zip(arms, offs):
            v = [run["panels"][p]["arms"][arm][R.ALPHA_KEY]["cov"] for p in panels
                 if run["panels"][p]["arms"].get(arm, {}).get(R.ALPHA_KEY)]
            mean = stats(run, fam, arm)["cov"]
            assert abs(mean - float(np.mean(v))) < 1e-6, (fam, arm)
            xs = g + off + (rng.random(len(v)) - 0.5) * (0.22 if len(v) > 1 else 0.0)
            ax.scatter(xs, v, s=5.5, marker=mk, color=col, edgecolors="white", linewidths=0.2, zorder=3, clip_on=False)
            ax.plot([g + off - 0.12, g + off + 0.12], [mean, mean], color=S.INK, lw=0.7, zorder=4,
                    solid_capstyle="butt")
    ax.axhline(0.9, color=S.INK2, lw=0.6, zorder=1.5)
    ax.set_xlim(-0.5, len(GROUPS) - 0.5)
    ax.set_ylim(0.3, 1.03)
    ax.set_yticks([0.4, 0.6, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.4", "0.6", "0.8", "0.9", "1.0"])
    ax.spines["left"].set_bounds(0.3, 1.0)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([lab for _, lab in GROUPS])
    thin_axes(ax)
    ax.tick_params(axis="x", length=0, pad=2.0, labelcolor=S.INK)
    # the three random groups share one label
    xin = lambda v: px + (v + 0.5) / len(GROUPS) * (w - 0.24)  # noqa: E731
    yb = py + h + 0.135
    c.line([xin(2.68), xin(5.32)], [yb, yb], color=S.AXIS, lw=0.5)
    c.text((xin(2.68) + xin(5.32)) / 2, yb + 0.065, "markers kept", size=FS_XS, ha="center", color=S.INK2)
    # key in the empty lower left of the plot, between the 0.6 and 0.4 gridlines
    y06, y04 = (py + (1.03 - v) / 0.73 * h for v in (0.6, 0.4))
    for i, arm in enumerate(arms):
        c.key(px + 0.07, y06 + (y04 - y06) * (0.22 + 0.28 * i), arm)


# ======================================================================= c, d
def lollipops(ax, cols, key, arms, offs, s=8.0):
    for (arm, col, mk, _), off in zip(arms, offs):
        v = [stats(r, "all_deletions", arm)[key] for _, r in cols]
        xs = np.arange(len(cols)) + off
        ax.vlines(xs, 0, v, color=col, lw=0.7, zorder=2)
        ax.scatter(xs, v, s=s, marker=mk, color=col, edgecolors="white", linewidths=0.2, zorder=3, clip_on=False)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_xticks([])
    thin_axes(ax)


def panels_cd(c, x, y, w):
    cols = [(name, R.run(key)) for name, key in COHORTS]
    for (name, key), (_, r) in zip(COHORTS, cols):     # the summaries must agree with the paired comparison
        p = R.paired(R.paired_name(key))
        a, b = stats(r, "all_deletions", "admit_m1"), stats(r, "all_deletions", "lac_node_gp")
        assert abs(a["mean_deficit"] - float(p["mean_deficit_A"])) < 1e-4, key
        assert abs(b["mean_deficit"] - float(p["mean_deficit_B"])) < 1e-4, key
        assert a["n_types"] == b["n_types"] == int(p["n_strata"]), key
    cols.sort(key=lambda nr: -stats(nr[1], "all_deletions", "lac_node_gp")["mean_deficit"])
    arms, offs = (ADMIT, LAC), (-0.09, 0.09)
    px, pw = x + 0.28, w - 0.28

    c.title(x, y, "c", "Mean coverage deficit")
    c.key(x + w - 1.03, y, ADMIT)
    c.key(x + w - 0.62, y, LAC)
    yc, hc = y + 0.11, 0.86
    ax = S.ax_in(c.fig, px, yc, pw, hc)
    lollipops(ax, cols, "mean_deficit", arms, offs)
    ax.set_ylim(0, 0.18)
    ax.set_yticks([0, 0.05, 0.10, 0.15])
    ax.set_yticklabels(["0", "0.05", "0.10", "0.15"])

    yd = yc + hc + 0.11
    c.title(x, yd, "d", "Mean set size")
    ytop, hd = yd + 0.09, 0.36
    bx = S.ax_in(c.fig, px, ytop, pw, hd)
    lollipops(bx, cols, "size", arms, offs, s=6.5)
    bx.set_ylim(0, 2.5)
    bx.set_yticks([0, 1, 2])
    for i, (name, _) in enumerate(cols):
        for j, part in enumerate(name.split("\n")):       # one baseline for every first line
            c.text(px + (i + 0.5) / len(cols) * pw, ytop + hd + 0.115 + j * 0.095, part, size=FS_LAB, ha="center",
                   va="baseline")
    return cols


# ======================================================================= numbers quoted in the caption
def report(df, cols):
    meta = S.load_json(str(S.ASSETS / "orion_case" / "case_study.json"))["cases"][0]["panels"][PANEL]
    lab = df["gating_node17"].notna()
    no_target = int((lab & df[f"{PANEL}__piP"].isna()).sum())
    unsup = float((~df[f"{PANEL}__forced_supported"].astype(bool)).mean())
    print(f"[a] {len(meta['missing_vs_full'])} missing; unsupported {meta['forced']['frac_unsupported']:.4f} "
          f"(parquet {unsup:.4f}); labelled with no target {no_target}/{int(lab.sum())} = {no_target / lab.sum():.4f}; "
          f"coverage {meta['admit']['coverage_M1_target_on_gating_cells']:.4f}; empty sets {meta['admit']['frac_abstain']:.4f}")
    run = R.run(SCHURCH_B)
    for fam, name in GROUPS:
        print(f"[b] {name:7s} " + "  ".join(f"{a[3]} {stats(run, fam, a[0])['cov']:.4f}" for a in (ADMIT, FULLCAL, LAC)))
    n = sum(stats(r, "all_deletions", "admit_m1")["n_types"] for _, r in cols)
    pool = {a[3]: sum(stats(r, "all_deletions", a[0])["n_types"] * stats(r, "all_deletions", a[0])["mean_deficit"]
                      for _, r in cols) / n for a in (ADMIT, LAC)}
    print(f"[c] pooled over {n} strata: ADMIT {pool['ADMIT']:.4f}, Marginal LAC {pool['Marginal LAC']:.4f}")
    for name, r in cols:
        a, b = stats(r, "all_deletions", "admit_m1"), stats(r, "all_deletions", "lac_node_gp")
        print(f"[c,d] {name.replace(chr(10), ' '):16s} deficit {a['mean_deficit']:.4f} {b['mean_deficit']:.4f}  "
              f"size {a['size']:.3f} {b['size']:.3f}  diff {a['size'] - b['size']:+.3f}")


def main():
    S.setup()
    c = Canvas()
    df = panel_a(c, 0.0, 0.06, 1.44)
    panel_b(c, 4.50, 0.06, 2.00, 1.33)
    cols = panels_cd(c, 0.0, 1.98, 6.5)

    S.check_figure(c.fig)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"Creator": "figures/make_fig2_v8.py (matplotlib)", "Title": STEM}
    c.fig.savefig(OUT_DIR / f"{STEM}.pdf", metadata=meta, dpi=450)
    c.fig.savefig(OUT_DIR / f"{STEM}.png", dpi=300)
    print(f"[out] {OUT_DIR / (STEM + '.pdf')}\n[out] {OUT_DIR / (STEM + '.png')}")
    report(df, cols)


if __name__ == "__main__":
    main()
