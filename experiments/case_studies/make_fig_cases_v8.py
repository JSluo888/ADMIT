#!/usr/bin/env python3
"""Case-study figures. Tissue and plots only; findings go in the captions.
fig_cases_v8: a Cords core without CD8 and HR forest; b rank-extreme patients with their share, and share rho per cohort.
fig_cases_tissue_supp: a Danenberg windows; b neighbour difference per cohort; c Kulasinghe HR forest.
Numbers from tables_cases.json (current requirement map); cell maps from round3map/. No model runs, no GPU, no patient
ids. Only cohort-level values of NeoTRIP, Meyer, Gupta, Girnius and cHL appear, as unnamed points.
Inputs: $ADMIT_RUNS/case_studies (cases_analysis.py, cases_r3map.py). Outputs: $ADMIT_FIG_OUT.

    python experiments/case_studies/make_fig_cases_v8.py"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'figures'))
import admit_style as S  # noqa: E402
import tissue as Ti  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402
from admit.ontology import NODE17  # noqa: E402

CS = Path(os.environ.get('ADMIT_RUNS', str(REPO / 'runs'))) / 'case_studies'
OUTDIR = S.FIGDIR
R3DIR = CS / 'round3map'   # per-cell ADMIT under the current requirement map
T = json.loads((CS / 'tables_cases.json').read_text())
IDX = {t: i for i, t in enumerate(NODE17)}
T_SET = [IDX[n] for n in ('T_Lineage', 'Cytotoxic_T_Cell', 'Helper_T_Cell', 'Regulatory_T_Cell')]
MAC_SET = [IDX[n] for n in ('Macrophage_CD163pos', 'Macrophage_CD163neg', 'Myeloid')]
CY, TR, POS = IDX['Cytotoxic_T_Cell'], IDX['Regulatory_T_Cell'], IDX['Macrophage_CD163pos']


def node_rgb(c):
    if c >= 0:
        return Ti.rgb(Ti.NODE_COL[NODE17[int(c)]])
    if c == -1:
        return Ti.rgb(Ti.NODE_COL['root'])
    if c == -2:
        return Ti.rgb(Ti.ABSTAIN)
    return Ti.rgb(Ti.NO_LABEL)


def map_ax(fig, x, y, s, r):
    ax = S.ax_in(fig, x, y, s, s)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor(Ti.MAP_BG)
    span = max(r.x.max() - r.x.min(), r.y.max() - r.y.min()) * 1.03
    cx, cy = (r.x.max() + r.x.min()) / 2, (r.y.max() + r.y.min()) / 2
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy + span / 2, cy - span / 2)
    ax.set_aspect('equal')
    return ax


def dots(ax, r, labels, focus, red=(), admit=False, sb=None, big=2.0, small=0.16):
    """Cell positions. Focus lineage in the paper's colours; red: call whose defining marker is missing; other cells dark.
    ADMIT: sets spanning lineages in the pale tone, empty sets in the abstention grey."""
    lab = np.asarray(labels)
    is_focus, is_red = np.isin(lab, focus), np.isin(lab, list(red))
    un, em = lab == -1, lab == -2
    rest = ~(is_focus | is_red | un | em)
    ax.scatter(r.x[rest], r.y[rest], s=small, c=Ti.OTHER_DARK, linewidths=0, rasterized=True, zorder=2)
    if admit:
        ax.scatter(r.x[un], r.y[un], s=small * 1.8, c=[Ti.rgb(Ti.NODE_COL['root'])], linewidths=0, rasterized=True, zorder=3)
        ax.scatter(r.x[em], r.y[em], s=small * 1.8, c=[Ti.rgb(Ti.ABSTAIN)], linewidths=0, rasterized=True, zorder=3)
    k = np.flatnonzero(is_focus & ~is_red)
    ax.scatter(r.x.iloc[k], r.y.iloc[k], s=big, c=[node_rgb(c) for c in lab[k]], edgecolors='#9fb4cc', linewidths=0.05, rasterized=True, zorder=4)
    k = np.flatnonzero(is_red)
    ax.scatter(r.x.iloc[k], r.y.iloc[k], s=big, c=Ti.UNSUPPORTED, linewidths=0, rasterized=True, zorder=5)
    if sb:
        x0, x1 = ax.get_xlim()
        y1, y0 = ax.get_ylim()
        w, h = x1 - x0, y1 - y0
        ax.plot([x1 - 0.05 * w - sb, x1 - 0.05 * w], [y1 - 0.06 * h] * 2, color='white', lw=0.9, solid_capstyle='butt', zorder=8)
        ax.text(x1 - 0.05 * w, y1 - 0.09 * h, f'{sb} µm', color='white', ha='right', va='bottom', fontsize=S.FS, zorder=8)


def pct(v, d=0):
    return f'{v * 100:.{d}f}%'


FS_LET, FS_TTL, FS_LAB, FS_XS = 8.5, 7.0, 6.5, 6.0   # as Fig 1
LINE_H = 0.105
BAN = ('not run', 'n/a', 'not estimable', 'not available', 'none', 'nan')

FOREST_LABEL = {('GT', 'cyto_T'): 'Reference, cytotoxic T', ('GT', 'pooled_T'): 'Reference, T cells', ('forced_noCD8', 'cyto_T'): 'Forced, cytotoxic T',
                ('forcedRT_noCD8', 'cyto_T'): 'Retrained, cytotoxic T', ('admit_noCD8', 'pooled_T'): 'ADMIT, T cells',
                ('admitRT_noCD8', 'pooled_T'): 'ADMIT retrained, T cells'}
ARM_COL = {'GT': S.INK, 'forced_noCD8': S.RED, 'forcedRT_noCD8': S.RED, 'admit_noCD8': S.BLUE, 'admitRT_noCD8': S.BLUE}   # red: calls without the marker
SHORT = {'Cytotoxic_T_Cell': 'Cytotoxic T', 'Helper_T_Cell': 'Helper T', 'Regulatory_T_Cell': 'Treg', 'T_Lineage': 'T cell', 'Macrophage_CD163pos': 'CD163+ mac.', 'Macrophage_CD163neg': 'CD163− mac.',
         'Myeloid': 'Myeloid', 'Neutrophil': 'Neutrophil', 'Dendritic_Cell': 'Dendritic', 'NK_Cell': 'NK', 'B_Cell': 'B', 'Plasma_Cell': 'Plasma', 'Immune_Other': 'Immune other',
         'Immune': 'Immune', 'Epithelial': 'Epithelial', 'Endothelial': 'Endothelial', 'Stromal': 'Stromal'}


def label(fig, x, y, s, size=FS_LAB, **kw):
    kw.setdefault('va', 'center')
    return S.fig_text(fig, x, y, s, fontsize=size, **kw)


def title(fig, x, y, letter, s):
    label(fig, x, y, letter, size=FS_LET, fontweight='bold')
    label(fig, x + 0.125, y, s, size=FS_TTL)


def bar(ax, um, text=True):
    """White scale bar, lower right, length above it."""
    x0, x1 = ax.get_xlim()
    y1, y0 = ax.get_ylim()
    w, h = x1 - x0, y1 - y0
    ax.plot([x1 - 0.05 * w - um, x1 - 0.05 * w], [y1 - 0.05 * h] * 2, color='white', lw=0.8, solid_capstyle='butt', zorder=8)
    if text:
        ax.text(x1 - 0.05 * w, y1 - 0.07 * h, f'{um} µm', color='white', ha='right', va='bottom', fontsize=FS_XS, zorder=8)


def ref_labels(lab):
    """Cells outside the hierarchy are -1 in the reference and -9 elsewhere; draw them alike, as other cells."""
    lab = np.asarray(lab)
    return np.where(lab == -1, -9, lab)


def legend(fig, x0, x1, y, items):
    """Squares with short labels on lines between x0 and x1; None starts a new line. Returns the bottom."""
    r_ = fig.canvas.get_renderer()
    W, H = fig._admit_w, fig._admit_h
    x, yy = x0, y
    for it in items:
        if it is None:
            x, yy = x0, yy + LINE_H
            continue
        col, s, edge = it
        t = label(fig, x + 0.085, yy, s, size=FS_XS)
        wd = t.get_window_extent(r_).width / fig.dpi
        assert x + 0.085 + wd <= x1, s
        if col:
            fig.add_artist(Rectangle((x / W, 1 - (yy + 0.03) / H), 0.06 / W, 0.06 / H, transform=fig.transFigure, facecolor=col, edgecolor=edge or 'none', lw=0.4))
        x += 0.085 + wd + 0.095
    return yy + 0.05


def forest(fig, rows, lx, px, pw, y, pitch, xlim, ticks):
    """Adjusted HR per SD, one row per arm, in the order of the table. A row with no HR has 0 calls written where the points sit."""
    ax = S.ax_in(fig, px, y, pw, pitch * len(rows))
    ax.set_xscale('log')
    for i, x in enumerate(rows):
        label(fig, lx, y + pitch * (i + 0.5), FOREST_LABEL[(x['arm'], x['celltype'])])
        if x['hr'] is None:
            ax.text(rows[0]['hr'], i, '0 calls', ha='center', va='center', fontsize=FS_LAB, color=S.INK, zorder=6)
            continue
        col = ARM_COL[x['arm']]
        ax.plot([max(x['lo'], xlim[0]), min(x['hi'], xlim[1])], [i, i], color=col, lw=S.LW_CI + 0.1, zorder=3)
        ax.plot(x['hr'], i, ls='none', marker='o', ms=2.8, mfc='white' if x['arm'] == 'admitRT_noCD8' else col, mec=col, mew=0.7, zorder=4)
    ax.axvline(1, color=S.INK2, lw=0.5, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:g}' for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([], minor=True)
    S.style_axes(ax, grid=None)
    ax.tick_params(axis='x', labelsize=FS_XS)
    ax.set_xlabel('Adjusted HR per SD', fontsize=FS_LAB, color=S.INK, labelpad=1.5)
    return ax


def swarm(vals, dx, step):
    """Vertical offsets so that dots dx apart in value do not overlap; no randomness."""
    placed = []
    for v in sorted(vals):
        for k in range(len(vals) * 2 + 1):
            yo = (k + 1) // 2 * step * (1 if k % 2 else -1)
            if all(abs(v - a) >= dx or abs(yo - b) >= step * 0.99 for a, b in placed):
                break
        placed.append((v, yo))
    return placed


# ------------------------------------------------------------------------------------------ main
def panel_a(fig, x, y):
    title(fig, x, y, 'a', 'No CD8, Cords')
    r = pd.read_parquet(R3DIR / 'cells_case1_core.parquet')
    s, g = 0.78, 0.02
    ty = y + 0.19
    for k, (ttl, lab, red, adm) in enumerate((('Reference', ref_labels(r.ref), [], False), ('Forced', r.forced_MI, [], False), ('Retrained forced', r.forced_RT, [CY], False), ('ADMIT', r.cons_MI, [], True))):
        xx = x + k * (s + g)
        ax = map_ax(fig, xx, ty, s, r)
        dots(ax, r, lab, T_SET, red=red, admit=adm, big=2.0, small=0.16)
        if k == 0:
            bar(ax, 200)
        label(fig, xx, ty - 0.055, ttl)
    rows = T['case1']['arms']['cords2024_nsclc_imc']
    assert T['case1']['call_rates_noCD8']['cords2024_nsclc_imc']['total_calls'] == 0, '0 calls is written at the forced row'
    fy = ty + s + 0.12
    pitch = 0.17
    forest(fig, rows, x, x + 1.06, 4 * s + 3 * g - 1.06, fy, pitch, (0.76, 1.1), [0.8, 0.9, 1])
    return fy + pitch * len(rows) + 0.25, x + 4 * s + 3 * g


def panel_b(fig, x, y):
    c = T['case2']
    title(fig, x, y, 'b', 'Subtype claims, missing marker')
    lx, mx = x, x + 0.70
    s, g, step = 0.66, 0.02, 0.78
    px, pw = mx + 2 * s + g + 0.12, 6.46 - (mx + 2 * s + g + 0.12)
    ty = y + 0.26
    label(fig, mx, y + 0.11, 'Reference')
    label(fig, mx + s + g, y + 0.11, 'Retrained forced')
    rows = (('CD8', 'CD8 of T', 'Patient B', 'Cords', T_SET, [CY], 'cells_vig_CD8.parquet', 100),
            ('FOXP3', 'Treg of T', 'Patient C', 'Cords', T_SET, [TR], 'cells_vig_FOXP3.parquet', 100),
            ('CD163', 'CD163+ of mac.', 'Patient D', 'QUICHE Spain', MAC_SET, [POS], 'cells_vig_CD163.parquet', None))   # Patient D: coordinates in pixels only
    for i, (m, what, pat, coh, focus, red, fn, sb) in enumerate(rows):
        r = pd.read_parquet(R3DIR / fn)
        p = c['vignettes'][m]['patient']
        yy = ty + i * step
        for k, (lab, rd, share) in enumerate(((ref_labels(r.ref), [], p['ref_share']), (r.forced_RT, red, p['forced_share']))):
            ax = map_ax(fig, mx + k * (s + g), yy, s, r)
            dots(ax, r, lab, focus, red=rd, big=1.7, small=0.09)
            if k == 0 and sb:
                bar(ax, sb)
            label(fig, mx + k * (s + g), yy - 0.05, pct(share))
        label(fig, lx, yy + s / 2 - LINE_H, what)
        label(fig, lx, yy + s / 2, pat, color=S.INK2)
        label(fig, lx, yy + s / 2 + LINE_H, coh, color=S.INK2)
    # share rho per cohort, rows aligned with the maps; y in inches from the top of the first map
    bottom = ty + 2 * step + s
    ax = S.ax_in(fig, px, ty, pw, bottom - ty)
    for i, (m, *_rest) in enumerate(rows):
        d = c['summary'][m]
        vals = [c['share_by_cohort'][m][k]['spearman'] for k in d['cohorts_used']]
        yc = i * step + s / 2
        for v, yo in swarm(vals, 0.052, 0.05):
            ax.plot(v, yc + yo, ls='none', marker='o', ms=2.6, mfc=S.INK2, mec='white', mew=0.2, zorder=3)
        ax.plot([d['median_spearman']] * 2, [yc - 0.20, yc + 0.20], color=S.INK, lw=1.0, zorder=4, solid_capstyle='butt')
    for v_ in (0.6, 0.8):
        ax.axvline(v_, color=S.GRID, lw=0.6, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom - ty, 0)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([0, 0.6, 0.8, 1])
    ax.set_xticklabels(['0', '0.6', '0.8', '1'])
    S.style_axes(ax, grid=None)
    ax.tick_params(axis='x', labelsize=FS_XS)
    ax.set_xlabel('Spearman ρ, per cohort', fontsize=FS_LAB, color=S.INK, labelpad=1.5)
    return bottom + 0.25


def build_main(h):
    fig = S.figure(h)
    fig.canvas.draw()
    y = 0.06
    ya, xr = panel_a(fig, 0.0, y)
    yb = panel_b(fig, xr + 0.12, y)
    items = [(Ti.NODE_COL[k], SHORT[k], None) for k in ('Cytotoxic_T_Cell', 'Helper_T_Cell', 'Regulatory_T_Cell', 'T_Lineage', 'Macrophage_CD163pos', 'Macrophage_CD163neg')]
    items += [None, (Ti.NODE_COL['Myeloid'], SHORT['Myeloid'], None), (Ti.NODE_COL['root'], 'Spans lineages', None), (Ti.ABSTAIN, 'Empty set', None), (Ti.UNSUPPORTED, 'Subtype, no marker', None), (Ti.OTHER_DARK, 'Other', None)]
    yl = legend(fig, 0.0, xr, max(ya + 0.09, yb - 0.05 - LINE_H), items)
    return fig, max(yl, yb)


# ------------------------------------------------------------------------------------------ supplement
def windows(fig, x, y):
    c = T['case3']
    title(fig, x, y, 'a', 'Danenberg windows')
    s, g, step = 0.78, 0.02, 0.89
    xs = [x + k * (s + g) for k in range(4)]
    ty = y + 0.27
    for x_, t_ in zip(xs, ('Channels', 'Reference', 'Forced', 'ADMIT')):
        label(fig, x_, y + 0.115, t_)
    lims = {}
    for w in c['windows']:
        for ch, (lo, hi) in w['limits'].items():
            a = lims.setdefault(ch, [lo, hi])
            a[0], a[1] = min(a[0], lo), max(a[1], hi)
    used, cons = set(), set()
    for i, w in enumerate(c['windows']):
        z = np.load(R3DIR / f'case3_window_{i}.npz')
        r = pd.read_parquet(R3DIR / f'cells_case3_{i}.parquet').set_index('obj')
        obj = z['obj']
        yy = ty + i * step
        chans = [Ti.stretch(np.arcsinh(z[ch]), np.arcsinh(lims[ch][0]), np.arcsinh(lims[ch][1])) for ch in ('DNA', 'PanCK', 'CD3', 'CD68')]
        comp = Ti.composite(chans, [Ti.MARKER_COL[k] for k in ('DNA', 'PanCK', 'CD3', 'CD68')])
        lab = obj.copy()
        lab[~np.isin(obj, r.index[r.unres].to_numpy())] = 0
        edge_all, edge_un = find_boundaries(obj, mode='inner'), find_boundaries(lab, mode='inner')
        comp[edge_un] = 1.0
        a_ = Ti.image_ax(fig, xs[0], yy, s, s)
        a_.imshow(comp, extent=(0, 100, 100, 0), interpolation='none')
        a_.plot([70, 95], [94, 94], color='white', lw=0.8, solid_capstyle='butt')
        in_win = r.loc[r.index.isin(np.unique(obj))]
        for k, col in enumerate(('ref', 'forced', 'cons')):
            lut = np.tile(Ti.rgb(Ti.NO_LABEL), (int(obj.max()) + 1, 1))
            for ob_id, row in r.iterrows():
                if ob_id < len(lut) and row.ref >= 0:
                    lut[ob_id] = node_rgb(row[col])
            img = lut[obj]
            img[obj == 0] = Ti.rgb(Ti.MAP_BG)
            img[edge_all] = Ti.rgb(Ti.MAP_BG)
            img[edge_un] = 1.0
            a_ = Ti.image_ax(fig, xs[k + 1], yy, s, s)
            a_.imshow(img, extent=(0, 100, 100, 0), interpolation='none')
            shown = in_win.loc[in_win.ref >= 0, col]
            used |= {int(v) for v in shown[shown >= 0].unique()}
            if col == 'cons':
                cons |= set(shown[shown < 0].astype(int))
        cv = w['coverage']['image']
        ly = yy - 0.05
        label(fig, xs[0], ly, f"P{int(w['q'] * 100)}, {pct(w['unres_frac_image_rerun'])} unresolved")
        label(fig, xs[2], ly, f"{pct(cv['forced_correct'])} correct")
        label(fig, xs[3], ly, f"{pct(cv['set_contains_reference'])} has reference")
    order = ['Cytotoxic_T_Cell', 'Helper_T_Cell', 'Regulatory_T_Cell', 'T_Lineage', 'Macrophage_CD163pos', 'Macrophage_CD163neg', 'Myeloid', 'Dendritic_Cell', 'Neutrophil', 'NK_Cell', 'B_Cell', 'Plasma_Cell',
             'Immune_Other', 'Immune', 'Epithelial', 'Endothelial', 'Stromal']
    items = [(Ti.NODE_COL[k], SHORT[k], None) for k in order if IDX[k] in used]
    if -1 in cons:
        items.append((Ti.NODE_COL['root'], 'Spans lineages', None))
    if -2 in cons:
        items.append((Ti.ABSTAIN, 'Empty set', None))
    items.append(('white', 'Unresolved', S.INK2))
    items += [None] + [(Ti.MARKER_COL[k], k, None) for k in ('DNA', 'PanCK', 'CD3', 'CD68')] + [(None, 'Bar 25 µm', None)]
    return ty + 2 * step + s, xs[3] + s, items


def neighbours(fig, lx, x1, y):
    W = T['case3']['W1_by_cohort']
    order = sorted(W, key=lambda k: W[k]['diff'])
    title(fig, lx, y, 'b', 'Neighbours, per cohort')
    ty = y + 0.15
    pitch = 0.072
    px = lx + 0.56
    ax = S.ax_in(fig, px, ty, x1 - px, pitch * len(order))
    for i, k in enumerate(order):
        v = W[k]
        col = S.BLUE if k == 'danenberg2022_metabric_imc' else (S.INK2 if v['lo'] > 0 else S.MUTED)
        ax.plot([v['lo'], v['hi']], [i, i], color=col, lw=S.LW_CI, zorder=3)
        ax.plot(v['diff'], i, ls='none', marker='o', ms=2.0, mfc=col, mec='none', zorder=4)
    ax.axvline(0, color=S.INK2, lw=0.5, zorder=1)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xticks([0, 0.05, 0.1])
    ax.set_xticklabels(['0', '0.05', '0.1'])
    S.style_axes(ax, grid=None)
    ax.tick_params(axis='x', labelsize=FS_XS)
    ax.set_xlabel('Neighbour difference', fontsize=FS_LAB, color=S.INK, labelpad=1.5)
    label(fig, lx, ty + pitch * (order.index('danenberg2022_metabric_imc') + 0.5), 'Danenberg', color=S.BLUE)
    return ty + pitch * len(order) + 0.25


def build_supp(h):
    fig = S.figure(h)
    fig.canvas.draw()
    y = 0.06
    ya, xr, items = windows(fig, 0.0, y)
    lx = xr + 0.16
    px = lx + 1.12
    pw = 6.46 - px
    yn = neighbours(fig, lx, 6.46, y)
    title(fig, lx, yn + 0.06, 'c', 'Kulasinghe, no CD8')
    pitch = 0.125
    ky = yn + 0.06 + 0.11
    forest(fig, T['case1']['arms']['kulasinghe2026_nsclc_pcf'], lx, px, pw, ky, pitch, (0.35, 2.8), [0.5, 1, 2])
    yk = ky + 6 * pitch + 0.25
    yl = legend(fig, 0.0, 6.46, max(ya, yk) + 0.09, items)
    return fig, yl


# ------------------------------------------------------------------------------------------ build
def check(fig):
    """The text checks of admit_style, plus: no placeholders, labels of at most 4 words."""
    problems = S.check_figure(fig)
    for t in S._visible_texts(fig):
        s = t.get_text().strip()
        if any(b in s.lower().split() or (' ' in b and b in s.lower()) for b in BAN) or '–' in s or s == '-' or len(s.split()) > 4:
            problems.append(f'label: {s!r}')
    assert not problems, problems


def main():
    S.setup()
    S.MIN_PT = 6.0
    import matplotlib.pyplot as plt
    for build, stem, hmax in ((build_main, 'fig_cases_v8', 3.4), (build_supp, 'fig_cases_tissue_supp', 3.2)):
        fig, bottom = build(4.5)
        plt.close(fig)
        h = round(bottom + 0.03, 2)
        assert h <= hmax, (stem, h)
        fig, _ = build(h)
        check(fig)
        OUTDIR.mkdir(parents=True, exist_ok=True)
        meta = {'Title': stem, 'Creator': 'experiments/case_studies/make_fig_cases_v8.py'}
        fig.savefig(OUTDIR / f'{stem}.pdf', dpi=600, metadata=meta)
        fig.savefig(OUTDIR / f'{stem}.png', dpi=300)
        plt.close(fig)
        print(stem, '6.5 x', h, 'in')


if __name__ == '__main__':
    main()
