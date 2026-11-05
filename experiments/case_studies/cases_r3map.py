"""Recompute every ADMIT arm of the case studies under the current requirement map (r3map.onto_r3). Selections stay
fixed: cases_selection*.json are only read.
Writes to round3map/; earlier files are not touched. Usage: cases_r3map.py <job> [...]; jobs: hr:<cohort>, cd163:<cohort>, danenberg_sets, cards_danenberg, cards_cd163, cards_cords."""
import json
import sys

from r3map import *  # noqa: F403
from ds_common import assoc

import cases_cells as CC
import cases_cd163 as C163

R3DIR = CS / 'round3map'
OLD, NEW = onto_old(), onto_r3()
CC.onto = NEW
CC.CS = R3DIR
C163.CS = R3DIR
C163.OUTJ = R3DIR / 'cd163_aggregates.json'
CARDS = R3DIR / 'cells_summary.json'
if CARDS.exists():
    CC.OUT.clear()
    CC.OUT.update(json.loads(CARDS.read_text()))
else:
    CC.OUT.clear()


def memo(co):
    """One fit per (direction, scorer): both maps then score the same model."""
    sp = co.splits()
    co.splits = lambda seed=0: sp
    cache = {}
    orig = co.model

    def model(s, marker=None):
        k = (id(s), marker)
        if k not in cache:
            cache[k] = orig(s, marker)
        return cache[k]
    co.model = model
    return sp


def hr_job(name):
    """Two no-CD8 ADMIT HRs (imputed and retrained scorer): saved, CPU old map, CPU new map, from the same fitted models."""
    co = Cohort(name)
    sp = memo(co)
    n = len(co.clin)
    assert co.mask_cols('CD8'), name
    out = {'cohort': name, 'admissible_changed_no_CD8': [], 'arms': {}}
    panel = [m for m in co.panel if m != 'CD8']
    out['admissible_changed_no_CD8'] = [NODE17[c] for c in np.flatnonzero(OLD.admissible(panel) != NEW.admissible(panel))]
    y_present = set(np.unique(co.y).tolist())
    out['changed_nodes_with_labels'] = [x for x in out['admissible_changed_no_CD8'] if IDX[x] in y_present]
    frame = co.frame.copy()
    adj = list(co.cfg['covariates'])
    B = pd.read_csv(DOWN / name / 'partB_survival.tsv', sep='\t')
    B = B[(B.endpoint == co.prim) & (B.direction == 'crossfit') & (B.seed == 0) & (B.celltype == 'pooled_T')].set_index('arm')
    for tag, scorer, arm in (('imputed', None, 'admit_noCD8'), ('retrained', 'CD8', 'admitRT_noCD8')):
        cols = {}
        same_sets = same_cons = tot = 0
        for mp, o in (('old', OLD), ('new', NEW)):
            CC.onto = o
            c_ = np.full(n, np.nan)
            keep = {}
            for d, s in sp.items():
                ty = CC.Typed(co, d, scorer, 'CD8')
                c_[s['ev']] = ty.counts(ty.cons, T_NODES, n)[s['ev']]
                keep[d] = (ty.sets.copy(), ty.cons.copy())
            cols[mp] = c_
            if mp == 'old':
                prev = keep
        for d in sp:
            same_sets += int((prev[d][0] == keep[d][0]).sum())
            same_cons += int((prev[d][1] == keep[d][1]).sum())
            tot += len(keep[d][0])
        CC.onto = NEW
        df = frame.copy()
        nm = co.cnt['n_mapped'][df.pidx].astype(float)
        res = {}
        for mp in cols:
            df[f'T_{mp}'] = cols[mp][df.pidx] / nm
            r = assoc(df, f'T_{mp}', adj, 'surv', ph_test=False)
            res[mp] = {k: r.get(k) for k in ('n', 'events', 'hr', 'ci_lo', 'ci_hi', 'p', 'error')}
        s0 = B.loc[arm]
        res['saved'] = {'hr': float(s0.hr), 'ci_lo': float(s0.ci_lo), 'ci_hi': float(s0.ci_hi), 'p': float(s0.p)}
        res['cells_set_identical'] = [same_sets, tot]
        res['cells_consensus_identical'] = [same_cons, tot]
        res['patients_T_count_changed'] = int((cols['old'][co.elig] != cols['new'][co.elig]).sum())
        out['arms'][tag] = res
        pd.DataFrame({'pidx': co.elig, 'T_frac_new': cols['new'][co.elig] / co.cnt['n_mapped'][co.elig], 'T_frac_old_cpu': cols['old'][co.elig] / co.cnt['n_mapped'][co.elig]}).to_csv(
            R3DIR / f'admit_T_{tag}_{name}.tsv', sep='\t', index=False)
    p = R3DIR / 'hr_old_new.json'
    d = json.loads(p.read_text()) if p.exists() else {}
    d[name] = out
    p.write_text(json.dumps(d, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    log('hr', name, json.dumps({k: {m: (round(v[m]['hr'], 3) if v[m].get('hr') else None) for m in ('saved', 'old', 'new')} for k, v in out['arms'].items()}), 'changed nodes', out['admissible_changed_no_CD8'])


def danenberg_sets():
    """Whole evaluation folds: are the full-panel sets identical under the two maps? Confirmed, not assumed."""
    name = 'danenberg2022_metabric_imc'
    co = Cohort(name)
    sp = memo(co)
    panel = co.panel
    out = {'admissible_changed_full_panel': [NODE17[c] for c in np.flatnonzero(OLD.admissible(panel) != NEW.admissible(panel))], 'FOXP3_in_panel': 'FOXP3' in panel, 'CD163_in_panel': 'CD163' in panel, 'folds': {}}
    for d in sp:
        CC.onto = OLD
        a = CC.Typed(co, d, None, None)
        CC.onto = NEW
        b = CC.Typed(co, d, None, None)
        out['folds'][d] = {'cells': int(len(a.sets)), 'sets_identical': int((a.sets == b.sets).sum()), 'consensus_identical': int((a.cons == b.cons).sum())}
    (R3DIR / 'danenberg_sets_check.json').write_text(json.dumps(out, indent=1))
    log('danenberg sets', json.dumps(out))


def save_cards():
    CARDS.write_text(json.dumps(CC.OUT, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))


if __name__ == '__main__':
    log('requirement maps checked: current = old + required_measured,', len(NODE17), 'nodes')
    for j in sys.argv[1:]:
        CC.onto = NEW
        if j.startswith('hr:'):
            hr_job(j[3:])
        elif j.startswith('cd163:'):
            C163.run(j[6:])
        elif j == 'danenberg_sets':
            danenberg_sets()
        elif j == 'cards_danenberg':
            CC.run_danenberg()
            save_cards()
        elif j == 'cards_cd163':
            CC.run_cd163()
            save_cards()
        elif j == 'cards_cords':
            CC.run_cords()
            save_cards()
        log('job done', j)
