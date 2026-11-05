"""Tables for the case studies: tables_cases.json. No patient ids, and no per-patient rows from restricted cohorts.
Reads existing seed-0 tables, the CD163 retrain, and the per-cell summary. Selection happened earlier, in cases_select.py."""
import json

from cases_common import *  # noqa: F403
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import spearmanr

R3DIR = CS / 'round3map'   # ADMIT arms under the current requirement map (cases_r3map.py); forced and reference labels do not depend on the map
HRN = json.loads((R3DIR / 'hr_old_new.json').read_text())
COH = sorted(f.parent.name for f in DOWN.glob('*/fig_scatter_seed0_crossfit.tsv'))
ROWS_C1 = [('GT', 'cyto_T', 'Reference, cytotoxic T'), ('GT', 'pooled_T', 'Reference, T cells'),
           ('forced_noCD8', 'cyto_T', 'Forced, imputed, cytotoxic T'), ('forcedRT_noCD8', 'cyto_T', 'Forced, retrained, cytotoxic T'),
           ('admit_noCD8', 'pooled_T', 'ADMIT, imputed scorer, T cells'), ('admitRT_noCD8', 'pooled_T', 'ADMIT, retrained scorer, T cells')]


def surv_rows(name, arm, ct):
    B = pd.read_csv(DOWN / name / 'partB_survival.tsv', sep='\t')
    prim = [e for e, v in COHORTS[name]['endpoints'].items() if v['primary']][0]
    return B[(B.endpoint == prim) & (B.direction == 'crossfit') & (B.arm == arm) & (B.celltype == ct)]


def with_round3_admit(t, name):
    """Replace the saved no-CD8 ADMIT T fractions by those under the current requirement map."""
    t = t.copy()
    for tag, col in (('imputed', 'admit_noCD8_pooled_T'), ('retrained', 'admitRT_noCD8_pooled_T')):
        n = pd.read_csv(R3DIR / f'admit_T_{tag}_{name}.tsv', sep='\t').set_index('pidx').T_frac_new
        t[col] = n.reindex(t.pidx).to_numpy()
    return t


def case1():
    out = {'arms': {}}
    for name in ('cords2024_nsclc_imc', 'kulasinghe2026_nsclc_pcf'):
        rows = []
        for arm, ct, lab in ROWS_C1:
            g = surv_rows(name, arm, ct)
            r0 = g[g.seed == 0].iloc[0]
            by_seed = {int(r.seed): (None if pd.isna(r.hr) else float(r.hr)) for r in g.itertuples()}
            rows.append({'label': lab, 'arm': arm, 'celltype': ct, 'n': int(r0.n), 'events': int(r0.events),
                         'hr': None if pd.isna(r0.hr) else float(r0.hr), 'lo': None if pd.isna(r0.ci_lo) else float(r0.ci_lo),
                         'hi': None if pd.isna(r0.ci_hi) else float(r0.ci_hi), 'p': None if pd.isna(r0.p) else float(r0.p),
                         'error': r0.error if isinstance(r0.error, str) else None, 'hr_by_seed': by_seed,
                         'ci_by_seed': {int(r.seed): (None if pd.isna(r.hr) else [float(r.ci_lo), float(r.ci_hi)]) for r in g.itertuples()}})
        for x, tag in ((rows[4], 'imputed'), (rows[5], 'retrained')):
            a = HRN[name]['arms'][tag]
            x['old_map_saved'] = {'hr': x['hr'], 'lo': x['lo'], 'hi': x['hi'], 'p': x['p']}
            x['old_map_cpu_same_fit'] = {k: a['old'][k] for k in ('hr', 'ci_lo', 'ci_hi', 'p')}
            x['hr'], x['lo'], x['hi'], x['p'] = a['new']['hr'], a['new']['ci_lo'], a['new']['ci_hi'], a['new']['p']
            x['n'], x['events'] = a['new']['n'], a['new']['events']
            x['hr_by_seed'], x['ci_by_seed'] = {0: x['hr']}, {0: [x['lo'], x['hi']]}
            x['cells_set_identical'], x['patients_T_count_changed'] = a['cells_set_identical'], a['patients_T_count_changed']
        out['arms'][name] = rows
    calls = json.loads((CS / 'check_kulasinghe_calls.json').read_text())
    out['what_the_call_rate_column_means'] = ('summary_compact.forcedMI_noCD8_pct_cells_cytoT = median over 10 seed x direction splits of '
                                              'the pooled fraction of evaluation cells; the HR is from the seed-0 cross-fit')
    out['call_rates_noCD8'] = {k: v for k, v in calls.items() if not k.startswith('_')}
    out['kulasinghe_detail'] = calls['_kulasinghe_detail']
    # how common: seed-0 not estimable, recomputed on every cohort with the arm
    common = {}
    for marker, ct, arm in (('CD8', 'cyto_T', 'forced_noCD8'), ('FOXP3', 'treg', 'forced_noFOXP3')):
        rows = {}
        for name in COH:
            t = scatter_table(name)
            if f'{arm}_{ct}' not in t:
                continue
            g = surv_rows(name, arm, ct)
            if not len(g):
                continue
            calls_ = (t[f'{arm}_{ct}'] * t.n_mapped).round()
            r0 = g[g.seed == 0]
            rows[name] = {'patients': int(len(t)), 'patients_any_call': int((calls_ > 0).sum()), 'calls': int(calls_.sum()),
                          'seed0_estimable': bool(len(r0) and pd.notna(r0.iloc[0].hr)),
                          'seeds_estimable_of_5': int(g.hr.notna().sum())}
        common[marker] = {'n_cohorts': len(rows), 'seed0_not_estimable': sorted(k for k, v in rows.items() if not v['seed0_estimable']),
                          'never_estimable_any_seed': sorted(k for k, v in rows.items() if v['seeds_estimable_of_5'] == 0), 'by_cohort': rows}
    out['how_common'] = common
    # KM for Cords, tertiles of the out-of-fold T-cell proportion
    name = 'cords2024_nsclc_imc'
    co = Cohort(name, with_markers=False)
    t = restore_order(scatter_table(name), co.elig, co.cnt['n_mapped']).merge(co.frame[['pidx', 'T', 'E']], on='pidx')
    t = with_round3_admit(t, name)
    km = {'n': int(len(t)), 'events': int(t.E.sum()), 'time_unit': 'as deposited (os_time)', 'arms': {}}
    cuts = {}
    for key, col in (('reference', 'gt_pooled_T'), ('admit_imputed', 'admit_noCD8_pooled_T'), ('admit_retrained', 'admitRT_noCD8_pooled_T'),
                     ('forced_retrained_cytoT', 'forcedRT_noCD8_cyto_T')):
        a, b = np.quantile(t[col], [1 / 3, 2 / 3])
        g = np.where(t[col] <= a, 'low', np.where(t[col] > b, 'high', 'mid'))
        cuts[key] = g
        arm = {'cuts': [float(a), float(b)], 'groups': {}}
        for lv in ('low', 'mid', 'high'):
            m = g == lv
            f = KaplanMeierFitter().fit(t['T'][m], t.E[m])
            sf = f.survival_function_.iloc[:, 0]
            arm['groups'][lv] = {'n': int(m.sum()), 'events': int(t.E[m].sum()), 'median': float(f.median_survival_time_),
                                 't': sf.index.to_numpy().round(3).tolist(), 's': sf.to_numpy().round(5).tolist()}
        lr = logrank_test(t['T'][g == 'high'], t['T'][g == 'low'], t.E[g == 'high'], t.E[g == 'low'])
        arm['logrank_high_vs_low_p'] = float(lr.p_value)
        km['arms'][key] = arm
    km['same_tertile_reference_vs_admit_imputed'] = float((cuts['reference'] == cuts['admit_imputed']).mean())
    km['opposite_tertile_reference_vs_admit_imputed'] = float((((cuts['reference'] == 'high') & (cuts['admit_imputed'] == 'low')) |
                                                               ((cuts['reference'] == 'low') & (cuts['admit_imputed'] == 'high'))).mean())
    km['spearman_reference_vs_admit_imputed'] = float(spearmanr(t.gt_pooled_T, t.admit_noCD8_pooled_T).statistic)
    km['forced_imputed_note'] = 'no tertiles: the imputed forced classifier calls no cytotoxic T cell in any patient'
    km['patients_any_forced_imputed_call'] = int((t.forced_noCD8_cyto_T > 0).sum())
    out['km_cords'] = km
    return out


SHARE = {'CD8': dict(sub='cyto_T', cf='noCD8'), 'FOXP3': dict(sub='treg', cf='noFOXP3')}


def share_tables():
    """Per marker, per cohort. Cohorts with >= 30 patients (the CD163 eligibility rule; it gives 14 cohorts for CD8)."""
    res = {}
    for marker, d in SHARE.items():
        sub, cf = d['sub'], d['cf']
        rows = {}
        for name in COH:
            t = scatter_table(name)
            a, b, c = f'gt_{sub}', f'forcedRT_{cf}_{sub}', f'forcedRT_{cf}_pooled_T'
            if a not in t or b not in t or c not in t:
                continue
            n = t.n_mapped
            st = share_stats(t[a] * n, t.gt_pooled_T * n, t[b] * n, t[c] * n)
            st['n_table_patients'] = int(len(t))
            st['abundance_spearman'] = float(spearmanr(t[a], t[b]).statistic) if t[a].std() > 0 and t[b].std() > 0 else None
            st['abundance_tertile_flip'] = tertile_flips(t[a], t[b])
            st['T_count_only_spearman'] = float(spearmanr(t[a], t.gt_pooled_T).statistic) if t[a].std() > 0 else None
            st['forcedT_count_only_spearman'] = float(spearmanr(t[a], t[c]).statistic) if t[a].std() > 0 and t[c].std() > 0 else None
            st['pooled_ref_abundance_pct'] = float((t[a] * n).sum() / n.sum() * 100)
            st['pooled_forced_abundance_pct'] = float((t[b] * n).sum() / n.sum() * 100)
            rows[name] = st
        res[marker] = rows
    cd = json.loads((R3DIR / 'cd163_aggregates.json').read_text())
    res['CD163'] = {k: {**v['share'], 'n_table_patients': v['n_eligible']} for k, v in cd.items()}
    return res, cd


def reading(rho):
    return 'proxy holds, refusal conservative' if rho >= 0.8 else ('proxy fails, refusal justified' if rho <= 0.6 else 'partial')


def summarise(rows):
    use = {k: v for k, v in rows.items() if v['n_table_patients'] >= 30 and v.get('spearman') is not None}
    rho = [v['spearman'] for v in use.values()]
    out = {'cohorts_used': sorted(use), 'n_cohorts': len(use), 'median_spearman': float(np.median(rho)), 'min_spearman': float(np.min(rho)),
           'max_spearman': float(np.max(rho)), 'median_tertile_flip': float(np.median([v['tertile_flip'] for v in use.values()])),
           'max_tertile_flip': float(np.max([v['tertile_flip'] for v in use.values()])),
           'reading': reading(float(np.median(rho))),
           'not_used': {k: ('under 30 patients' if v['n_table_patients'] < 30 else 'share constant in an arm') for k, v in rows.items() if k not in use}}
    big = [v['spearman_ge50_parent'] for v in use.values() if v.get('spearman_ge50_parent') is not None]
    if big:
        out['median_spearman_ge50_parent'] = float(np.median(big))
        out['n_cohorts_ge50_parent'] = len(big)
    ab = [v['abundance_spearman'] for v in use.values() if v.get('abundance_spearman') is not None]
    if ab:
        out['median_abundance_spearman'] = float(np.median(ab))
        out['max_abundance_tertile_flip'] = float(np.max([v['abundance_tertile_flip'] for v in use.values() if 'abundance_tertile_flip' in v]))
        ct = [v['T_count_only_spearman'] for v in use.values() if v.get('T_count_only_spearman') is not None]
        out['median_T_count_only_spearman'] = float(np.median(ct))
    return out


def survival(marker):
    """Cohorts with a survival primary endpoint and >= 30 events. Same patients in both arms: >= 1 parent cell in each."""
    rows = {}
    for name in COH if marker != 'CD163' else ['quiche_tnbc_mibi_spain', 'quiche_tnbc_mibi_stanford', 'kulasinghe2026_nsclc_pcf']:
        cfg = COHORTS[name]
        prim = [e for e, v in cfg['endpoints'].items() if v['primary']][0]
        if cfg['endpoints'][prim].get('kind', 'surv') != 'surv':
            continue
        co = Cohort(name, with_markers=False)
        if marker == 'CD163':
            t = pd.read_csv(R3DIR / f'cd163_{name}.tsv', sep='\t')
            d = pd.DataFrame({'pidx': t.pidx, 'ref_n': t.gt_pos, 'ref_p': t.gt_pos + t.gt_neg, 'for_n': t.rt_pos, 'for_p': t.rt_pos + t.rt_neg, 'n': t.n_mapped})
        else:
            sub, cf = SHARE[marker]['sub'], SHARE[marker]['cf']
            t = scatter_table(name)
            if f'forcedRT_{cf}_{sub}' not in t:
                continue
            t = restore_order(t, co.elig, co.cnt['n_mapped'])
            n = t.n_mapped
            d = pd.DataFrame({'pidx': t.pidx, 'ref_n': t[f'gt_{sub}'] * n, 'ref_p': t.gt_pooled_T * n, 'for_n': t[f'forcedRT_{cf}_{sub}'] * n,
                              'for_p': t[f'forcedRT_{cf}_pooled_T'] * n, 'n': n})
        ev = int(co.frame[co.frame.pidx.isin(d.pidx)].E.sum())
        if ev < 30:
            rows[name] = {'events_primary': ev, 'skipped': 'under 30 events'}
            continue
        d = d.merge(co.frame, on='pidx')
        d = d[(d.ref_p > 0) & (d.for_p > 0)].reset_index(drop=True)
        d['ref_share'], d['for_share'] = d.ref_n / d.ref_p, d.for_n / d.for_p
        d['ref_parent'], d['for_parent'] = d.ref_p / d.n, d.for_p / d.n
        res = {'events_primary': ev, 'adjusted_for': list(cfg['covariates']) + ['parent proportion']}
        for tag, x, par in (('reference', 'ref_share', 'ref_parent'), ('forced_retrained', 'for_share', 'for_parent')):
            if d[x].std() == 0:
                res[tag] = {'error': 'constant share'}
                continue
            D, _, _ = design(d, x, list(cfg['covariates']), 'T', 'E')
            D['parent'] = zs(d.loc[D.index, par])
            res[tag] = fit_model(D, 'surv')
        rows[name] = res
    return rows


def window_detail(k, im):
    """Set coverage of the reference label and forced accuracy, in the window and the whole image, split by unresolved."""
    r = pd.read_parquet(R3DIR / f'cells_case3_{k}.parquet')
    r = r[r.ref >= 0].copy()
    r['hit'] = [(int(b) >> int(y)) & 1 == 1 for b, y in zip(r.set, r.ref)]
    r['ok'] = r.forced == r.ref
    wx, wy = np.round(r.x).astype(int), np.round(r.y).astype(int)
    inw = (wx >= im['window_x0']) & (wx < im['window_x0'] + 100) & (wy >= im['window_y0']) & (wy < im['window_y0'] + 100)

    def blk(d):
        u, v = d[d.unres], d[~d.unres]
        return {'n': int(len(d)), 'set_contains_reference': float(d.hit.mean()), 'forced_correct': float(d.ok.mean()),
                'n_unres': int(len(u)), 'unres_set_contains_reference': float(u.hit.mean()) if len(u) else None,
                'unres_forced_correct': float(u.ok.mean()) if len(u) else None, 'n_res': int(len(v)),
                'res_set_contains_reference': float(v.hit.mean()) if len(v) else None, 'res_forced_correct': float(v.ok.mean()) if len(v) else None}
    return {'window': blk(r[inw]), 'image': blk(r)}


def vignette_context(sel, marker):
    """Where the patient sits on the level ADMIT does report, among all the cohort's patients (percent of patients at or below)."""
    v = sel
    if marker == 'CD163':
        t = pd.read_csv(R3DIR / f"cd163_{v['cohort']}.tsv", sep='\t')
        ref, adm = (t.gt_pos + t.gt_neg) / t.n_mapped, (t.adm_pos + t.adm_neg + t.adm_mye) / t.n_mapped
        ids = t.pidx.to_numpy()
    else:
        co = Cohort(v['cohort'], with_markers=False)
        t = restore_order(scatter_table(v['cohort']), co.elig, co.cnt['n_mapped'])
        cf = SHARE[marker]['cf']
        ref, adm = t.gt_pooled_T, t[f'admitRT_{cf}_pooled_T']
        ids = t.pidx.to_numpy()
    i = int(np.flatnonzero(ids == v['pidx'])[0])
    imp = {}
    if marker != 'CD163':
        cf, sub = SHARE[marker]['cf'], SHARE[marker]['sub']
        imp = {'forced_imputed_subtype_calls': int(round(t[f'forced_{cf}_{sub}'].iloc[i] * t.n_mapped.iloc[i])),
               'forced_imputed_parent_pct_of_cells': float(t[f'forced_{cf}_pooled_T'].iloc[i] * 100)}
    tagged = 'admit_parent_pct_of_cells' if marker == 'CD163' else 'admit_parent_pct_of_cells_saved_old_map'
    return {**imp, 'n_patients': int(len(t)), 'ref_parent_pct_of_cells': float(ref.iloc[i] * 100), tagged: float(adm.iloc[i] * 100),
            'ref_parent_percentile': float((ref <= ref.iloc[i]).mean() * 100), 'admit_parent_percentile': float((adm <= adm.iloc[i]).mean() * 100),
            'spearman_parent_ref_vs_admit_cohort': float(spearmanr(ref, adm).statistic)}


def case1_patient(sel):
    """Patient A, the Case 1 core's patient: percent of mapped cells and tertile per arm (seed-0 table)."""
    co = Cohort(sel['cohort'], with_markers=False)
    t = with_round3_admit(restore_order(scatter_table(sel['cohort']), co.elig, co.cnt['n_mapped']), sel['cohort'])
    i = int(np.flatnonzero(t.pidx.to_numpy() == sel['pidx'])[0])
    out = {'n_mapped': int(round(t.n_mapped.iloc[i])), 'n_patients': int(len(t))}
    for key, col in (('ref_cyto', 'gt_cyto_T'), ('ref_T', 'gt_pooled_T'), ('forced_imputed_cyto', 'forced_noCD8_cyto_T'),
                     ('forced_retrained_cyto', 'forcedRT_noCD8_cyto_T'), ('admit_imputed_T', 'admit_noCD8_pooled_T'), ('admit_retrained_T', 'admitRT_noCD8_pooled_T')):
        v = t[col]
        a, b = np.quantile(v, [1 / 3, 2 / 3])
        x = v.iloc[i]
        out[key] = {'pct': float(x * 100), 'tertile': None if a == b else ('low' if x <= a else ('high' if x > b else 'mid'))}
    return out


def main():
    T = {'seed': 0, 'requirement_map': 'R = required_positive | required_measured; ADMIT class AdmitM1 (not AdmitM1Frozen)',
         'danenberg_sets_check': json.loads((R3DIR / 'danenberg_sets_check.json').read_text())}
    T['case1'] = case1()
    log('case1 done')
    shares, cd = share_tables()
    T['case2'] = {'share_by_cohort': {m: {k: v for k, v in r.items()} for m, r in shares.items()},
                  'summary': {m: summarise(r) for m, r in shares.items()},
                  'cd163_detail': {k: {kk: v[kk] for kk in ('folds', 'consensus_of_reference_cells', 'pooled', 'n_mapped_total', 'admit_frac_patients_any_CD163neg_call',
                                                             'admit_pooled_neg_of_macrophage_level')} for k, v in cd.items()},
                  'survival': {m: survival(m) for m in ('CD8', 'FOXP3', 'CD163')}}
    log('case2 done')
    cs = json.loads((R3DIR / 'cells_summary.json').read_text())
    sel = json.loads((CS / 'cases_selection.json').read_text())
    sel163 = json.loads((CS / 'cases_selection_cd163.json').read_text())
    T['case1']['core'] = cs['case1_core']
    T['case1']['patient_A'] = case1_patient(sel['case1_core'])
    T['case1']['core_rule'] = {k: sel['case1_core'][k] for k in ('n_candidates', 'n_cores_total', 'ref_cyto_T_in_core', 'n_cells_core')}
    T['case2']['vignettes'] = {m: cs[f'vig_{m}'] for m in ('CD8', 'FOXP3', 'CD163')}
    T['case2']['vignette_candidates'] = {**sel['case2_all_candidates'], 'CD163': sel163['all_candidates']}
    for m in ('CD8', 'FOXP3', 'CD163'):
        vs = sel163['vignette'] if m == 'CD163' else sel['case2_vignettes'][m]
        T['case2']['vignettes'][m]['context'] = vignette_context(vs, m)
        T['case2']['vignettes'][m]['table_values_at_selection'] = {k: vs[k] for k in ('ref_share_table', 'forced_share_table')}
        rc = pd.read_parquet(R3DIR / f'cells_vig_{m}.parquet')
        rc = rc[rc.ref >= 0]
        T['case2']['vignettes'][m]['core_consensus'] = {'n_mapped': int(len(rc)), 'empty': int((rc.cons_RT == -2).sum()), 'set_spans_lineages': int((rc.cons_RT == -1).sum()),
                                                        'empty_and_reference_epithelial': int(((rc.cons_RT == -2) & (rc.ref == IDX['Epithelial'])).sum())}
    W = {}
    for f in sorted((UNCERT_BIO / 'cohorts').glob('*_W.json')):
        d = json.loads(f.read_text())
        w = d.get('W1_foreign_neighbour_diff')
        W[f.name[:-7]] = {'diff': w[0], 'lo': w[1], 'hi': w[2], 'base_resolved': d.get('W1_mean_foreign_resolved'), 'unresolved_rate': d['unresolved_rate']}
    T['case3'] = {'W1_by_cohort': W, 'W1_above_zero': sum(v['lo'] > 0 for v in W.values()), 'W1_n': len(W),
                  'image_rule': {k: sel['case3'][k] for k in ('n_images_ge500', 'n_images_total', 'unresolved_rate_all')},
                  'images': [{k: im[k] for k in ('q', 'n_cells', 'unres_frac', 'window_cells', 'window_unres')} for im in sel['case3']['images']],
                  'windows': cs['case3_windows'], 'rerun_vs_saved_sets': cs['danenberg_rerun_vs_saved_sets']}
    for k, im in enumerate(sel['case3']['images']):
        T['case3']['windows'][k]['coverage'] = window_detail(k, im)
    T['case3']['images_summary'] = json.loads((CS / 'case3_images_summary.json').read_text())
    T['reproduction'] = {'kulasinghe_rt_cpu_vs_table': json.loads((CS / 'check_cpu_rerun.json').read_text()), 'cords_folds': cs['checks_cords'],
                         'cd163_vignette_rerun_equal_patients': cs['check_cd163_rerun_equal_patients']}
    (CS / 'tables_cases.json').write_text(json.dumps(T, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    log('written')


if __name__ == '__main__':
    main()
