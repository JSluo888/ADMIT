"""Fig 4: panels a (cores) and b (coverage in selected regions) from make_fig4_v8.py; panel c is the per-type coverage
deficit of selected regions, selection-conditional minus pooled calibration.

    python figures/make_fig4_v9.py

c  summary_regions.json: paired cond_vs_pooled_uncert, mean_deficit_F_sel (20% budget, 6-plex).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import admit_style as S  # noqa: E402
import make_fig4_regions as F4  # noqa: E402
import make_fig4_v8 as V8  # noqa: E402

STEM = "fig4_v9"


def panel_c(c, data, x, y):
    c.title(x, y, "c", "Per-type deficit, selected regions")
    rows = []
    for cohort, _ in F4.COHORTS:
        m = data["cohorts"][cohort]["panels"][V8.PANEL]["paired"]["cond_vs_pooled_uncert"][V8.BUDGET]["mean_deficit_F_sel"]
        rows.append((m["diff"], m["ci"][0], m["ci"][1]))
    a = S.ax_in(c.fig, x + V8.AX_DX, y + 0.13, V8.AX_W, 1.16)
    for i, (v, lo, hi) in enumerate(rows):
        assert -0.04 < lo <= hi < 0.01
        a.plot([lo, hi], [i, i], color=V8.COND, lw=S.LW_CI, solid_capstyle="butt", zorder=3)
        a.plot([v], [i], "o", ms=3.0, color=V8.COND, mew=0, zorder=4)
    a.set_xlim(-0.035, 0.008)
    a.set_ylim(len(rows) - 0.5, -0.5)
    a.set_yticks(range(len(rows)))
    a.set_yticklabels([label for _, label in F4.COHORTS], fontsize=V8.FS_LAB, color=S.INK)
    ticks = [-0.03, -0.02, -0.01, 0]
    a.set_xticks(ticks)
    a.set_xticklabels(["0" if t == 0 else f"{t:.2f}".replace("-", "−") for t in ticks], fontsize=V8.FS_XS)
    a.tick_params(axis="x", length=2.0, width=S.LW_AXIS, pad=1.5)
    a.tick_params(axis="y", length=0, pad=3)
    a.spines["left"].set_visible(False)
    a.spines["bottom"].set_linewidth(S.LW_AXIS)
    a.set_xlabel("conditional minus pooled", fontsize=V8.FS_LAB, color=S.INK, labelpad=1.5)
    a.axvline(0, color=S.INK2, lw=0.6, zorder=2)


def main():
    S.setup()
    c = V8.Canvas()
    data = json.loads((V8.REG / "summary_regions.json").read_text())
    V8.panel_a(c, 0.0, 0.06)
    y = 2.03
    V8.panel_b(c, data, 0.0, y)
    panel_c(c, data, 3.43, y)
    S.check_figure(c.fig)
    V8.OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"Creator": "figures/make_fig4_v9.py (matplotlib)", "Title": STEM}
    c.fig.savefig(V8.OUT_DIR / f"{STEM}.pdf", metadata=meta, dpi=450)
    c.fig.savefig(V8.OUT_DIR / f"{STEM}.png", dpi=300)
    print(f"[out] {V8.OUT_DIR / (STEM + '.pdf')}")


if __name__ == "__main__":
    main()
