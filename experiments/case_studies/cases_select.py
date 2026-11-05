"""Apply the case-study selection rules, before any per-cell typing or new retrain, and write cases_selection.json.
Only integer indices are written. The CD163 vignette needs the retrain, so cases_cd163.py applies the same rule later.

Choices fixed here before running:
- Vignette: rank within each cohort's patients with >= 50 reference parent cells and a defined forced share (average ranks);
  the largest integer rank difference across cohorts where individual maps are allowed. Ties: more parent cells, then lower pidx.
- Median core: sort by (count, core index), take index (n - 1) // 2.
- 'Test fold' = out of fold in the seed-0 cross-fit, so every eligible patient qualifies in exactly one direction.
- Percentile image: sort ascending by (fraction, image index), take index round(q * (n - 1)).
- Window: 100 x 100 px (1 um/px), fully inside the image, 1 px steps; cell inside by rounded centroid; ties: first in row-major order.
"""
import json
import sys
from pathlib import Path

from cases_common import *  # noqa: F403
from scipy.stats import rankdata
import pyarrow.parquet as pq

SHARES = {'CD8': ('cyto_T', 'noCD8'), 'FOXP3': ('treg', 'noFOXP3')}


def sample_ids_of(name):
    """sample_id of mapped cells (in memory only), with part_b's sample filter applied."""
    d = cohort_dir(name)
    s = pq.read_table(d / 'cells.parquet', columns=['sample_id']).column('sample_id').to_pandas().astype(str).to_numpy()
    sf = COHORTS[name].get('sample_filter')
    if sf:
        smp = pd.read_csv(d / sf['file'], low_memory=False)
        s = s[np.isin(s, list(set(smp.loc[smp[sf['col']] == sf['keep'], 'sample_id'].astype(str))))]
    return s


def vignette_candidates(name, marker):
    ct, cf = SHARES[marker]
    t = scatter_table(name)
    a, b, c = f'gt_{ct}', f'forcedRT_{cf}_{ct}', f'forcedRT_{cf}_pooled_T'
    if a not in t or b not in t or c not in t:
        return None
    co = Cohort(name, with_markers=False)
    t = restore_order(t, co.elig, co.cnt['n_mapped'])
    ref_parent = t.gt_pooled_T * t.n_mapped
    fp = t[c] * t.n_mapped
    g = t[(ref_parent.round() >= 50) & (fp.round() > 0)].copy()
    if len(g) < 2:
        return None
    g['ref_share'] = g[a] / g.gt_pooled_T
    g['forced_share'] = g[b] / g[c]
    if g.ref_share.std() == 0:
        return None
    g['r_ref'], g['r_for'] = rankdata(g.ref_share), rankdata(g.forced_share)
    g['d'] = (g.r_ref - g.r_for).abs()
    g['parent'] = (g.gt_pooled_T * g.n_mapped).round()
    g = g.sort_values(['d', 'parent', 'pidx'], ascending=[False, False, True])
    x = g.iloc[0]
    return {'cohort': name, 'marker': marker, 'pidx': int(x.pidx), 'n_ranked': int(len(g)), 'rank_diff': float(x.d),
            'rank_ref': float(x.r_ref), 'rank_forced': float(x.r_for), 'ref_share_table': float(x.ref_share),
            'forced_share_table': float(x.forced_share), 'ref_parent_cells': int(x.parent),
            'n_mapped': int(round(x.n_mapped))}


def largest_core(name, pidx):
    co = Cohort(name, with_markers=False)
    sid = sample_ids_of(name)[co.keep]
    uniq = sorted(np.unique(sid).tolist())
    s = pd.Series(sid[co.cpk == pidx]).value_counts()
    top = s.sort_index().sort_values(ascending=False, kind='stable').index[0]
    sp = co.splits()
    direction = '1' if pidx in set(sp['1']['ev'].tolist()) else '2'
    return {'sample_idx': int(uniq.index(top)), 'n_cores_of_patient': int(len(s)), 'n_cells_core': int(s.iloc[0]),
            'direction': direction}


def main():
    out = {'rules': __doc__}
    # ---- Case 1: Cords core
    name = 'cords2024_nsclc_imc'
    co = Cohort(name, with_markers=False)
    sid = sample_ids_of(name)[co.keep]
    uniq = sorted(np.unique(sid).tolist())
    idx = pd.Series(np.arange(len(uniq)), index=uniq)[sid].to_numpy()
    elig = np.isin(co.cpk, co.elig)
    n_ct = np.bincount(idx[elig & (co.y == IDX['Cytotoxic_T_Cell'])], minlength=len(uniq))
    n_all = np.bincount(idx[elig], minlength=len(uniq))
    cand = np.flatnonzero(n_ct >= 20)
    order = cand[np.lexsort((cand, n_ct[cand]))]
    pick = int(order[(len(order) - 1) // 2])
    pat = co.cpk[idx == pick]
    assert len(np.unique(pat)) == 1
    pidx = int(pat[0])
    sp = co.splits()
    out['case1_core'] = {'cohort': name, 'sample_idx': pick, 'pidx': pidx, 'n_candidates': int(len(cand)),
                         'n_cores_total': int(len(uniq)), 'ref_cyto_T_in_core': int(n_ct[pick]),
                         'n_cells_core': int(n_all[pick]),
                         'direction': '1' if pidx in set(sp['1']['ev'].tolist()) else '2'}
    log('case1 core', {k: v for k, v in out['case1_core'].items() if k not in ('sample_idx', 'pidx')})
    del co, sid, idx

    # ---- Case 2: vignettes for CD8, FOXP3 (table only)
    out['case2_vignettes'] = {}
    out['case2_all_candidates'] = {}
    for marker in SHARES:
        rows = []
        for f in sorted(DOWN.glob('*/fig_scatter_seed0_crossfit.tsv')):
            c = f.parent.name
            if c in RESTRICTED:
                continue
            r = vignette_candidates(c, marker)
            if r:
                rows.append(r)
        rows.sort(key=lambda r: (-r['rank_diff'], -r['ref_parent_cells']))
        out['case2_all_candidates'][marker] = [{k: r[k] for k in ('cohort', 'n_ranked', 'rank_diff')} for r in rows]
        top = rows[0]
        top.update(largest_core(top['cohort'], top['pidx']))
        out['case2_vignettes'][marker] = top
        log(marker, 'vignette', {k: v for k, v in top.items() if k not in ('sample_idx', 'pidx')})

    # ---- Case 3: Danenberg images and windows
    name = 'danenberg2022_metabric_imc'
    z = np.load(UNCERT_BIO / 'cells' / f'{name}.npz')
    t = pq.read_table(cohort_dir(name) / 'cells.parquet', columns=['sample_id', 'x', 'y']).to_pandas()
    all_ids = sorted(t.sample_id.astype(str).unique().tolist())   # image index = position in the sorted sample ids
    img = pd.Series(np.arange(len(all_ids)), index=all_ids)[t.sample_id.astype(str)].to_numpy()[z['keep']]
    xy = t[['x', 'y']].to_numpy(float)[z['keep']]
    ok = ~np.isnan(z['unres'])
    g = pd.DataFrame({'img': img[ok], 'u': z['unres'][ok]}).groupby('img').u.agg(['size', 'mean'])
    g = g[g['size'] >= 500].reset_index().sort_values(['mean', 'img'])
    res = {'n_images_ge500': int(len(g)), 'n_images_total': int(len(all_ids)), 'unresolved_rate_all': float(z['unres'][ok].mean()),
           'images': []}
    co = Cohort(name, with_markers=False)
    assert np.array_equal(co.keep, z['keep'])
    sp = co.splits()
    ev1 = set(sp['1']['ev'].tolist())
    import tifffile
    RAW = DANENBERG_RAW
    for q in (0.25, 0.50, 0.75):
        r = g.iloc[int(round(q * (len(g) - 1)))]
        i = int(r.img)
        shape = tifffile.imread(RAW / f'img{i}_cellmask.tiff').shape
        m = (img == i) & ok
        H = np.zeros(shape, dtype=np.int32)
        px, py = np.round(xy[m]).astype(int).T
        un = z['unres'][m] > 0
        np.add.at(H, (np.clip(py[un], 0, shape[0] - 1), np.clip(px[un], 0, shape[1] - 1)), 1)
        C = np.zeros((shape[0] + 1, shape[1] + 1), dtype=np.int64)
        C[1:, 1:] = H.cumsum(0).cumsum(1)
        W = 100
        S = C[W:, W:] - C[:-W, W:] - C[W:, :-W] + C[:-W, :-W]
        y0, x0 = np.unravel_index(np.argmax(S), S.shape)
        inw = (px >= x0) & (px < x0 + W) & (py >= y0) & (py < y0 + W)
        p = np.unique(z['cpk'][np.flatnonzero(m)])
        assert len(p) == 1
        res['images'].append({'q': q, 'img': i, 'pidx': int(p[0]), 'n_cells': int(r['size']), 'unres_frac': float(r['mean']),
                              'shape': [int(v) for v in shape], 'window_y0': int(y0), 'window_x0': int(x0),
                              'window_cells': int(inw.sum()), 'window_unres': int(un[inw].sum()),
                              'direction': '1' if int(p[0]) in ev1 else '2'})
    out['case3'] = res
    log('case3', json.dumps(res))
    (CS / 'cases_selection.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
