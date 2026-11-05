"""Per-cell typing of the selected cores and images (cases_selection.json, cases_selection_cd163.json). Seed 0, part_b's split, CPU.
Writes cells_<key>.parquet (x, y, labels; no cell or patient ids), case3_window_<k>.npz (window crops of the read-only Danenberg stacks)
and cells_summary.json (patient and core counts, and how each fold compares with the published tables)."""
import json
import re
import sys

from cases_common import *  # noqa: F403
import pyarrow.parquet as pq
import tifffile
from scipy.stats import spearmanr
from admit.panel_matched import AdmitM1, Ontology

onto = Ontology.from_repo(NODE17)
SEL = json.loads((CS / 'cases_selection.json').read_text())
SEL_163 = json.loads((CS / 'cases_selection_cd163.json').read_text())
RAW = DANENBERG_RAW
CTYPE = {'CD8': 'Cytotoxic_T_Cell', 'FOXP3': 'Regulatory_T_Cell'}
OUT = {}

# Layer order of the Danenberg image stacks (markerStackOrder.csv of the public deposit), one epitope per layer.
STACK = [
    'Histone H3', 'SMA', 'CK5', 'CD38', 'HLA-DR', 'CK8-18', 'CD15', 'FSP1', 'CD163', 'ICOS', 'OX40', 'CD68',
    'HER2 (3B5)', 'CD3', 'Podoplanin', 'CD11c', 'PD-1', 'GITR', 'CD16', 'HER2 (D8F12)', 'CD45RA', 'B2M', 'CD45RO',
    'FOXP3', 'CD20', 'ER', 'CD8', 'CD57', 'Ki-67', 'PDGFRB', 'Caveolin-1', 'CD4', 'CD31-vWF', 'CXCL12', 'HLA-ABC',
    'panCK', 'c-Caspase3c-PARP', 'DNA1', 'DNA2',
]


def object_numbers(mask: np.ndarray) -> np.ndarray:
    """Lookup stored mask value -> table ObjectNumber (0 = background). The stored values are rescaled to the uint16
    range and do not sort by ObjectNumber; object k is the k-th distinct object met in a row-major scan of the mask."""
    vals, first = np.unique(mask, return_index=True)         # first occurrence in the flattened, row-major mask
    vals, first = vals[vals > 0], first[vals > 0]
    number = np.zeros(int(vals.max()) + 1, dtype=np.int64)
    number[vals[np.argsort(first)]] = np.arange(1, len(vals) + 1)
    return number


def bits(sets):
    return (sets.astype(np.int64) * (1 << np.arange(sets.shape[1]))).sum(1)


class Typed:
    """One (cohort, direction, scorer, masked marker): forced, ADMIT consensus and set on all evaluation cells."""

    def __init__(self, co, direction, scorer_marker, mask_marker):
        sp = co.splits()[direction]
        t0 = time.time()
        self.fitted = co.model(sp, scorer_marker)
        panel = [m for m in co.panel if m != mask_marker]
        Pc = co.score(self.fitted, sp['ca'], mask_marker)
        adm = AdmitM1(onto, ALPHA).fit(onto.aggregate(Pc), co.y[sp['ca']], panel)
        del Pc
        self.evi = np.concatenate([co.by_p[p] for p in sp['ev'] if p in co.by_p])
        n = len(self.evi)
        self.forced = np.empty(n, dtype=np.int16)
        self.cons = np.empty(n, dtype=np.int16)
        self.sets = np.empty(n, dtype=np.int64)
        for i in range(0, n, 500_000):
            P = co.score(self.fitted, self.evi[i:i + 500_000], mask_marker)
            self.forced[i:i + 500_000] = P.argmax(1)
            s = adm.predict(onto.aggregate(P))
            self.cons[i:i + 500_000] = consensus_node(s, onto)
            self.sets[i:i + 500_000] = bits(s)
        self.pe = co.cpk[self.evi]
        self.unres = ~((self.cons >= 0) & onto.is_leaf[np.clip(self.cons, 0, None)])
        self.admissible = [NODE17[c] for c in np.flatnonzero(adm.adm)]
        self.supported = [NODE17[c] for c in np.flatnonzero(adm.S)]
        self.accuracy = float((self.forced == co.y[self.evi]).mean())
        log(co.name, 'dir', direction, 'scorer without', scorer_marker, 'masked', mask_marker, 'eval', n,
            'acc', round(self.accuracy, 3), f'{time.time()-t0:.0f}s')

    def counts(self, labels, nodes, n):
        return np.bincount(self.pe[np.isin(labels, nodes)], minlength=n).astype(float)


def against_table(co, ty, sp, arm, nodes, labels):
    """Per-patient proportion on this fold's patients vs the published seed-0 table."""
    t = restore_order(scatter_table(co.name), co.elig, co.cnt['n_mapped']).set_index('pidx')
    ev = np.asarray(sp['ev'])
    c = ty.counts(labels, nodes, len(co.clin))[ev] / co.cnt['n_mapped'][ev]
    tab = t.loc[ev, arm].to_numpy()
    r = {'arm': arm, 'n_patients': int(len(ev)), 'exact_equal_patients': int((np.abs(c - tab) < 1e-9).sum()),
         'mean_abs_diff_pp': float(np.abs(c - tab).mean() * 100), 'mean_cpu_pct': float(c.mean() * 100), 'mean_table_pct': float(tab.mean() * 100)}
    if np.std(c) > 0 and np.std(tab) > 0:
        r['spearman'] = float(spearmanr(c, tab).statistic)
    return r


def core_table(name, co):
    """x, y, node17 and sample index of mapped cells, plus all-cell positions, in the filtered table's order."""
    d = cohort_dir(name)
    t = pq.read_table(d / 'cells.parquet', columns=['sample_id', 'x', 'y', 'node17']).to_pandas()
    sf = COHORTS[name].get('sample_filter')
    if sf:
        smp = pd.read_csv(d / sf['file'], low_memory=False)
        t = t[t.sample_id.astype(str).isin(set(smp.loc[smp[sf['col']] == sf['keep'], 'sample_id'].astype(str)))].reset_index(drop=True)
    uniq = sorted(t.sample_id.astype(str).unique().tolist())
    t['s'] = pd.Series(np.arange(len(uniq)), index=uniq)[t.sample_id.astype(str)].to_numpy()
    return t.drop(columns=['sample_id']), uniq


def core_cells(tab, co, ty_by_arm, sample_idx, key):
    """One core: every cell's position; labels on mapped, evaluated cells. Unmapped cells stay as ref=-1."""
    tcore = tab[tab.s == sample_idx]
    pos_in_keep = pd.Series(np.arange(len(co.keep)), index=co.keep)
    rows = pd.DataFrame({'x': tcore.x.to_numpy(float), 'y': tcore.y.to_numpy(float), 'ref': -1}, index=tcore.index)
    mapped = tcore.index.intersection(co.keep)
    rows.loc[mapped, 'ref'] = co.y[pos_in_keep[mapped].to_numpy()]
    for arm, ty in ty_by_arm.items():
        inv = pd.Series(np.arange(len(ty.evi)), index=ty.evi)
        rows[f'forced_{arm}'], rows[f'cons_{arm}'], rows[f'set_{arm}'] = -9, -9, np.int64(0)
        ii = pos_in_keep[mapped].to_numpy()
        hit = np.isin(ii, ty.evi)
        at = inv[ii[hit]].to_numpy()
        rows.loc[mapped[hit], f'forced_{arm}'] = ty.forced[at]
        rows.loc[mapped[hit], f'cons_{arm}'] = ty.cons[at]
        rows.loc[mapped[hit], f'set_{arm}'] = ty.sets[at]
    rows.reset_index(drop=True).to_parquet(CS / f'cells_{key}.parquet', index=False)
    return rows


def vignette_card(co, ty, v, marker, sp, rows):
    """Report card numbers from this rerun, whole patient and core."""
    n = len(co.clin)
    p = v['pidx']
    ev = np.asarray(sp['ev'])
    nm = co.cnt['n_mapped'].astype(float)
    y_all = co.y[ty.evi]
    if marker == 'CD163':
        sub, par = [POS], [POS, NEG]
    else:
        sub, par = [IDX[CTYPE[marker]]], T_NODES
    ref_s, ref_p = ty.counts(y_all, sub, n), ty.counts(y_all, par, n)
    for_s, for_p = ty.counts(ty.forced, sub, n), ty.counts(ty.forced, par, n)
    adm_p = ty.counts(ty.cons, par if marker != 'CD163' else [POS, NEG, MYE], n)
    adm_s = ty.counts(ty.cons, sub, n)
    adm_neg = ty.counts(ty.cons, [NEG], n)
    unres = np.bincount(ty.pe[ty.unres], minlength=n).astype(float)
    card = {'marker': marker, 'cohort': co.cfg['display'], 'rank_rule_from_tables': {k: v[k] for k in ('n_ranked', 'rank_diff', 'rank_ref', 'rank_forced')},
            'patient': {'n_mapped': int(nm[p]), 'ref_subtype': int(ref_s[p]), 'ref_parent': int(ref_p[p]), 'forced_subtype': int(for_s[p]), 'forced_parent': int(for_p[p]),
                        'admit_subtype': int(adm_s[p]), 'admit_parent_level': int(adm_p[p]), 'admit_cd163neg_calls': int(adm_neg[p]), 'admit_unresolved': int(unres[p]),
                        'ref_share': float(ref_s[p] / max(ref_p[p], 1)), 'forced_share': float(for_s[p] / max(for_p[p], 1)),
                        'ref_parent_frac': float(ref_p[p] / nm[p]), 'admit_parent_frac': float(adm_p[p] / nm[p])},
            'core': {'n_cells': int(len(rows)), 'n_mapped': int((rows.ref >= 0).sum()), 'n_cores_of_patient': v['n_cores_of_patient']}}
    return card


def fold_of(co, pidx):
    sp = co.splits()
    return '1' if pidx in set(sp['1']['ev'].tolist()) else '2'


def run_cords():
    name = 'cords2024_nsclc_imc'
    co = Cohort(name)
    sp = co.splits()
    tab, _ = core_table(name, co)
    log('loaded', name)
    c1 = SEL['case1_core']
    v8 = SEL['case2_vignettes']['CD8']
    vf = SEL['case2_vignettes']['FOXP3']
    assert c1['cohort'] == name and v8['cohort'] == name and vf['cohort'] == name, 'vignette not in Cords: edit the job list'
    checks = []
    mi = Typed(co, c1['direction'], None, 'CD8')
    checks.append(against_table(co, mi, sp[c1['direction']], 'forced_noCD8_cyto_T', [IDX['Cytotoxic_T_Cell']], mi.forced))
    checks.append(against_table(co, mi, sp[c1['direction']], 'admit_noCD8_pooled_T', T_NODES, mi.cons))
    rt = Typed(co, c1['direction'], 'CD8', 'CD8')
    checks.append(against_table(co, rt, sp[c1['direction']], 'forcedRT_noCD8_cyto_T', [IDX['Cytotoxic_T_Cell']], rt.forced))
    checks.append(against_table(co, rt, sp[c1['direction']], 'admitRT_noCD8_pooled_T', T_NODES, rt.cons))
    rows = core_cells(tab, co, {'MI': mi, 'RT': rt}, c1['sample_idx'], 'case1_core')
    m = rows.ref >= 0
    cy = IDX['Cytotoxic_T_Cell']
    OUT['case1_core'] = {'n_cells': int(len(rows)), 'n_mapped': int(m.sum()), 'ref_cyto_T': int((rows.ref == cy).sum()),
                         'ref_T': int(rows.ref.isin(T_NODES).sum()),
                         'forced_MI_cyto_T': int((rows.forced_MI == cy).sum()), 'forced_MI_T': int(rows.forced_MI.isin(T_NODES).sum()),
                         'forced_RT_cyto_T': int((rows.forced_RT == cy).sum()), 'forced_RT_T': int(rows.forced_RT.isin(T_NODES).sum()),
                         'admit_MI_cyto_T': int((rows.cons_MI == cy).sum()), 'admit_MI_T': int(rows.cons_MI.isin(T_NODES).sum()),
                         'admit_RT_cyto_T': int((rows.cons_RT == cy).sum()), 'admit_RT_T': int(rows.cons_RT.isin(T_NODES).sum()),
                         'forced_MI_of_ref_cyto': pd.Series(np.array(NODE17)[rows.forced_MI[rows.ref == cy]]).value_counts().head(5).to_dict(),
                         'admit_MI_of_ref_cyto': pd.Series(rows.cons_MI[rows.ref == cy].map(lambda c: NODE17[c] if c >= 0 else ('unresolved' if c == -1 else 'empty'))).value_counts().head(5).to_dict(),
                         'admissible_MI': mi.admissible, 'admissible_RT': rt.admissible}
    # FOXP3 vignette is in direction vf['direction'], CD8 vignette in v8['direction']
    OUT['checks_cords'] = checks
    del mi
    if v8['direction'] == c1['direction']:
        rt8 = rt
    else:
        rt8 = Typed(co, v8['direction'], 'CD8', 'CD8')
        checks.append(against_table(co, rt8, sp[v8['direction']], 'forcedRT_noCD8_cyto_T', [cy], rt8.forced))
    r8 = core_cells(tab, co, {'RT': rt8}, v8['sample_idx'], 'vig_CD8')
    OUT['vig_CD8'] = vignette_card(co, rt8, v8, 'CD8', sp[v8['direction']], r8)
    del rt, rt8
    rtf = Typed(co, vf['direction'], 'FOXP3', 'FOXP3')
    checks.append(against_table(co, rtf, sp[vf['direction']], 'forcedRT_noFOXP3_treg', [IDX['Regulatory_T_Cell']], rtf.forced))
    rf = core_cells(tab, co, {'RT': rtf}, vf['sample_idx'], 'vig_FOXP3')
    OUT['vig_FOXP3'] = vignette_card(co, rtf, vf, 'FOXP3', sp[vf['direction']], rf)


def run_cd163():
    v = SEL_163['vignette']
    co = Cohort(v['cohort'])
    sp = co.splits()
    tab, _ = core_table(v['cohort'], co)
    ty = Typed(co, v['direction'], 'CD163', 'CD163')
    rows = core_cells(tab, co, {'RT': ty}, v['sample_idx'], 'vig_CD163')
    OUT['vig_CD163'] = vignette_card(co, ty, v, 'CD163', sp[v['direction']], rows)
    # the per-patient table written by cases_cd163.py should be reproduced exactly
    t = pd.read_csv(CS / f"cd163_{v['cohort']}.tsv", sep='\t').set_index('pidx')
    ev = np.asarray(sp[v['direction']]['ev'])
    c = ty.counts(ty.forced, [POS], len(co.clin))[ev]
    OUT['check_cd163_rerun_equal_patients'] = [int((c == t.loc[ev, 'rt_pos'].to_numpy()).sum()), int(len(ev))]


def run_danenberg():
    name = 'danenberg2022_metabric_imc'
    co = Cohort(name)
    sp = co.splits()
    z = np.load(UNCERT_BIO / 'cells' / f'{name}.npz')
    assert np.array_equal(co.keep, z['keep'])
    t = pq.read_table(cohort_dir(name) / 'cells.parquet', columns=['sample_id', 'cell_id', 'x', 'y', 'node17']).to_pandas()
    all_ids = sorted(t.sample_id.astype(str).unique().tolist())
    t['img'] = pd.Series(np.arange(len(all_ids)), index=all_ids)[t.sample_id.astype(str)].to_numpy()
    t['obj'] = t.cell_id.astype(str).map(lambda s: int(re.search(r'(\d+)$', s).group(1)))
    t = t.drop(columns=['sample_id', 'cell_id'])
    pos = pd.Series(np.arange(len(co.keep)), index=co.keep)
    typed = {}
    agree = {}
    for d in sorted({im['direction'] for im in SEL['case3']['images']}):
        ty = Typed(co, d, None, None)
        typed[d] = ty
        u_old = z['unres'][ty.evi]
        agree[d] = {'n_cells': int(len(ty.evi)), 'unresolved_agree_with_saved_npz': float(((u_old > 0) == ty.unres).mean()),
                    'accuracy': ty.accuracy}
    OUT['danenberg_rerun_vs_saved_sets'] = agree
    names = list(STACK)
    want = {'DNA': names.index('DNA1'), 'PanCK': names.index('panCK'), 'CD3': names.index('CD3'), 'CD68': names.index('CD68')}
    stacks, wins = {}, []
    for k, im in enumerate(SEL['case3']['images']):
        i = im['img']
        stack = tifffile.imread(RAW / f'img{i}_stack.tiff')
        assert stack.shape[0] == len(STACK), stack.shape
        mask = tifffile.imread(RAW / f'img{i}_cellmask.tiff')
        lookup = object_numbers(mask)
        obj = lookup[mask]
        tc = t[t.img == i]
        # join check: mask-object centroid vs the table position
        nobj = int(obj.max())
        yy, xx = np.indices(obj.shape)
        cnt = np.bincount(obj.ravel(), minlength=nobj + 1).astype(float)
        cx = np.bincount(obj.ravel(), weights=xx.ravel(), minlength=nobj + 1) / np.maximum(cnt, 1)
        cy = np.bincount(obj.ravel(), weights=yy.ravel(), minlength=nobj + 1) / np.maximum(cnt, 1)
        inr = tc.obj.to_numpy() <= nobj
        dist = np.hypot(cx[tc.obj.to_numpy()[inr]] - tc.x.to_numpy()[inr], cy[tc.obj.to_numpy()[inr]] - tc.y.to_numpy()[inr])
        join = {'n_table_cells': int(len(tc)), 'n_mask_objects': nobj, 'n_table_obj_beyond_mask': int((~inr).sum()),
                'median_offset_px': float(np.median(dist)), 'frac_within_1p5px': float((dist <= 1.5).mean()), 'max_offset_px': float(dist.max())}
        ty = typed[im['direction']]
        inv = pd.Series(np.arange(len(ty.evi)), index=ty.evi)
        mapped = tc.index.intersection(co.keep)
        ii = pos[mapped].to_numpy()
        at = inv[ii].to_numpy()
        rows = pd.DataFrame({'obj': tc.obj.to_numpy(), 'x': tc.x.to_numpy(float), 'y': tc.y.to_numpy(float), 'ref': -1, 'forced': -9, 'cons': -9, 'set': np.int64(0), 'unres': False}, index=tc.index)
        rows.loc[mapped, 'ref'] = co.y[ii]
        rows.loc[mapped, 'forced'] = ty.forced[at]
        rows.loc[mapped, 'cons'] = ty.cons[at]
        rows.loc[mapped, 'set'] = ty.sets[at]
        rows.loc[mapped, 'unres'] = ty.unres[at]
        rows.reset_index(drop=True).to_parquet(CS / f'cells_case3_{k}.parquet', index=False)
        y0, x0 = im['window_y0'], im['window_x0']
        crop = {ch: np.asarray(stack[j, y0:y0 + 100, x0:x0 + 100], np.float32) for ch, j in want.items()}
        inside = mask > 0
        lim = {ch: [float(v) for v in np.percentile(stack[j][inside], (1, 99.5))] for ch, j in want.items()}
        np.savez_compressed(CS / f'case3_window_{k}.npz', obj=obj[y0:y0 + 100, x0:x0 + 100], **crop)
        wx, wy = np.round(rows.x).astype(int), np.round(rows.y).astype(int)
        inw = (wx >= x0) & (wx < x0 + 100) & (wy >= y0) & (wy < y0 + 100)
        w = rows[inw]
        ms = w[w.ref >= 0]
        un = ms[ms.unres]
        def nm(c):
            return NODE17[c] if c >= 0 else ('unresolved' if c == -1 else 'empty')
        def setnames(b):
            return '{' + ', '.join(NODE17[j] for j in range(len(NODE17)) if (int(b) >> j) & 1) + '}'
        wins.append({'q': im['q'], 'join': join, 'limits': lim, 'unres_frac_image_rerun': float(rows.loc[rows.ref >= 0, 'unres'].mean()),
                     'window': {'n_mapped': int(len(ms)), 'n_unresolved': int(len(un)), 'n_forced_correct': int((ms.forced == ms.ref).sum()),
                                'n_unres_forced_correct': int((un.forced == un.ref).sum()), 'n_res_forced_correct': int(((ms[~ms.unres]).forced == (ms[~ms.unres]).ref).sum()),
                                'unresolved_sets_top': pd.Series(un.set.map(setnames)).value_counts().head(6).to_dict(),
                                'unresolved_consensus': pd.Series(un.cons.map(nm)).value_counts().to_dict(),
                                'unresolved_ref': pd.Series(un.ref.map(nm)).value_counts().to_dict(),
                                'unresolved_forced': pd.Series(un.forced.map(nm)).value_counts().to_dict()}})
        del stack, mask, obj
    OUT['case3_windows'] = wins


if __name__ == '__main__':
    jobs = sys.argv[1:] or ['danenberg', 'cd163', 'cords']
    path = CS / 'cells_summary.json'
    if path.exists():
        OUT.update(json.loads(path.read_text()))
    for j in jobs:
        {'danenberg': run_danenberg, 'cd163': run_cd163, 'cords': run_cords}[j]()
        path.write_text(json.dumps(OUT, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
        log('job done', j)
