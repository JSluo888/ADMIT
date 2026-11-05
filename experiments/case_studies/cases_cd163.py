"""Case 2, new retrain: CD163+ share of macrophages without CD163. Same cross-fit as run_cohort.part_b's RT arm:
seed 0, both directions, CD163 set to 0 in training and test, XGBoost on CPU, ADMIT + M1 on the same scorer.
Eligible by the inventory of all 33 standardised cohorts: CD163 in the panel, both macrophage labels, >= 30 patients with >= 200
mapped cells. Below the patient rule: Lin 17, Einhaus 24 and 24, McCaffrey 21, Meylan 16. Usage: cases_cd163.py <cohort> [...]"""
import json
import sys

from cases_common import *  # noqa: F403
from scipy.stats import rankdata
from admit.panel_matched import AdmitM1, Ontology

ORDER = ['kulasinghe2026_nsclc_pcf', 'quiche_tnbc_mibi_stanford', 'quiche_tnbc_mibi_spain', 'gupta2026_normal_breast_imc']
OUTJ = CS / 'cd163_aggregates.json'


def run(name):
    t0 = time.time()
    co = Cohort(name)
    assert 'CD163' in co.panel, name
    onto = Ontology.from_repo(NODE17)
    panel = [m for m in co.panel if m != 'CD163']
    n = len(co.clin)
    keys = ['gt_pos', 'gt_neg', 'rt_pos', 'rt_neg', 'rt_mye', 'adm_pos', 'adm_neg', 'adm_mye', 'adm_unres']
    C = {k: np.zeros(n) for k in keys}
    conf = {'gt_pos_cells': {}, 'gt_neg_cells': {}}
    folds = []
    log(name, 'cells', len(co.keep), 'patients', len(co.elig), 'features', co.X.shape[1])
    for direction, sp in co.splits().items():
        ts = time.time()
        fitted = co.model(sp, 'CD163')
        log(name, 'dir', direction, 'fit', len(sp['tr']), f'{time.time()-ts:.0f}s')
        Pc = co.score(fitted, sp['ca'], 'CD163')
        adm = AdmitM1(onto, ALPHA).fit(onto.aggregate(Pc), co.y[sp['ca']], panel)
        del Pc
        evi = np.concatenate([co.by_p[p] for p in sp['ev'] if p in co.by_p])
        ye, pe = co.y[evi], co.cpk[evi]
        forced = np.empty(len(evi), dtype=np.int16)
        cons = np.empty(len(evi), dtype=np.int16)
        size = np.empty(len(evi), dtype=np.int16)
        has_neg = np.empty(len(evi), dtype=bool)
        for i in range(0, len(evi), 500_000):
            P = co.score(fitted, evi[i:i + 500_000], 'CD163')
            forced[i:i + 500_000] = P.argmax(1)
            sets = adm.predict(onto.aggregate(P))
            cons[i:i + 500_000] = consensus_node(sets, onto)
            size[i:i + 500_000] = sets.sum(1)
            has_neg[i:i + 500_000] = sets[:, NEG]
        for k, m in (('gt_pos', ye == POS), ('gt_neg', ye == NEG), ('rt_pos', forced == POS), ('rt_neg', forced == NEG),
                     ('rt_mye', forced == MYE), ('adm_pos', cons == POS), ('adm_neg', cons == NEG), ('adm_mye', cons == MYE),
                     ('adm_unres', ~((cons >= 0) & onto.is_leaf[np.clip(cons, 0, None)]))):
            C[k] += np.bincount(pe[m], minlength=n)
        for lab, tag in ((POS, 'gt_pos_cells'), (NEG, 'gt_neg_cells')):
            m = ye == lab
            names = np.array(NODE17 + ['unresolved', 'empty'])
            fv = pd.Series(names[forced[m]]).value_counts(normalize=True).head(5).round(4).to_dict()
            cv = pd.Series(names[np.where(cons[m] >= 0, cons[m], np.where(cons[m] == -1, len(NODE17), len(NODE17) + 1))]
                           ).value_counts(normalize=True).head(5).round(4).to_dict()
            conf[tag][direction] = {'n': int(m.sum()), 'forced': fv, 'admit_consensus': cv,
                                    'frac_set_contains_CD163neg': float(has_neg[m].mean())}
        folds.append({'direction': direction, 'n_fit_cells': int(len(sp['tr'])), 'n_cal_cells': int(len(sp['ca'])),
                      'n_eval_cells': int(len(evi)), 'accuracy_forced': float((forced == ye).mean()),
                      'mean_set_size': float(size.mean()),
                      'admissible_macrophage_nodes': [NODE17[c] for c in (POS, NEG, MYE) if adm.adm[c]],
                      'supported_macrophage_nodes': [NODE17[c] for c in (POS, NEG, MYE) if adm.S[c]],
                      'y_pos_maps_to': (NODE17[adm.gmap[POS]] if adm.gmap[POS] >= 0 else None),
                      'y_neg_maps_to': (NODE17[adm.gmap[NEG]] if adm.gmap[NEG] >= 0 else None),
                      'wall_s': round(time.time() - ts)})
        log(name, 'dir', direction, 'eval', len(evi), 'acc', round(folds[-1]['accuracy_forced'], 3), f'{time.time()-ts:.0f}s')
        del forced, cons, size, has_neg, fitted
    el = co.elig
    t = pd.DataFrame({k: v[el] for k, v in C.items()})
    t.insert(0, 'pidx', el)
    t.insert(1, 'n_mapped', co.cnt['n_mapped'][el])
    st = share_stats(t.gt_pos, t.gt_pos + t.gt_neg, t.rt_pos, t.rt_pos + t.rt_neg)
    res = {'cohort': name, 'seed': SEED, 'n_eligible': int(len(el)), 'folds': folds, 'share': st, 'consensus_of_reference_cells': conf,
           'pooled': {k: float(t[k].sum()) for k in keys}, 'n_mapped_total': float(t.n_mapped.sum()),
           'wall_s': round(time.time() - t0)}
    # ADMIT's own macrophage reading on the same patients
    ap = t.adm_pos + t.adm_neg + t.adm_mye
    res['admit_frac_patients_any_CD163neg_call'] = float((t.adm_neg > 0).mean())
    res['admit_pooled_neg_of_macrophage_level'] = float(t.adm_neg.sum() / max(ap.sum(), 1))
    if name not in RESTRICTED:     # per-patient table only where individual results are allowed
        t.to_csv(CS / f'cd163_{name}.tsv', sep='\t', index=False)
    d = json.loads(OUTJ.read_text()) if OUTJ.exists() else {}
    d[name] = res
    OUTJ.write_text(json.dumps(d, indent=1))
    log(name, 'done', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()}, f'{time.time()-t0:.0f}s')


def vignette():
    """Same rule as cases_select: largest integer rank difference, among cohorts where maps are allowed."""
    from cases_select import largest_core
    rows = []
    for name in ORDER:
        if name in RESTRICTED or not (CS / f'cd163_{name}.tsv').exists():
            continue
        t = pd.read_csv(CS / f'cd163_{name}.tsv', sep='\t')
        t['parent'] = t.gt_pos + t.gt_neg
        t['fp'] = t.rt_pos + t.rt_neg
        g = t[(t.parent >= 50) & (t.fp > 0)].copy()
        g['ref_share'], g['forced_share'] = g.gt_pos / g.parent, g.rt_pos / g.fp
        g['r_ref'], g['r_for'] = rankdata(g.ref_share), rankdata(g.forced_share)
        g['d'] = (g.r_ref - g.r_for).abs()
        x = g.sort_values(['d', 'parent', 'pidx'], ascending=[False, False, True]).iloc[0]
        rows.append({'cohort': name, 'marker': 'CD163', 'pidx': int(x.pidx), 'n_ranked': int(len(g)), 'rank_diff': float(x.d),
                     'rank_ref': float(x.r_ref), 'rank_forced': float(x.r_for), 'ref_share_table': float(x.ref_share),
                     'forced_share_table': float(x.forced_share), 'ref_parent_cells': int(x.parent), 'n_mapped': int(x.n_mapped)})
    rows.sort(key=lambda r: (-r['rank_diff'], -r['ref_parent_cells']))
    top = rows[0]
    top.update(largest_core(top['cohort'], top['pidx']))
    (CS / 'cases_selection_cd163.json').write_text(json.dumps(
        {'vignette': top, 'all_candidates': [{k: r[k] for k in ('cohort', 'n_ranked', 'rank_diff')} for r in rows]}, indent=1))
    log('CD163 vignette', {k: v for k, v in top.items() if k not in ('sample_idx', 'pidx')})


if __name__ == '__main__':
    for c in sys.argv[1:]:
        if c == 'vignette':
            vignette()
            continue
        try:
            run(c)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            log(c, 'FAILED', repr(e)[:300])
