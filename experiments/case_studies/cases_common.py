"""Shared code for the case studies. Patient ids stay in memory; outputs use integer indices only.

Reuses experiments/downstream (run_cohort, ds_common). XGBoost runs on CPU: run_cohort.fit_xgb uses device='cuda', so
fit_cpu copies it with only the device changed, and the GPU is hidden before xgboost loads.

Paths: $ADMIT_RUNS (default <repo>/runs) holds downstream/, uncert_bio/ and the case-study outputs (case_studies/);
$ADMIT_DATA_ROOT (default <repo>/data) holds the standard-schema cohorts and the Danenberg image stacks."""
from __future__ import annotations

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['ADMIT_XGB_DEVICE'] = 'cpu'
os.environ.setdefault('OMP_NUM_THREADS', '8')

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'experiments' / 'downstream'))
from ds_common import COHORTS, cohort_dir, design, endpoint_frame, load_cells, load_clinical  # noqa: E402
from run_cohort import ALPHA, MIN_CELLS, NODE17, consensus_node, patient_counts, predict17, sample_cells  # noqa: E402

RUNS = Path(os.environ.get('ADMIT_RUNS', str(REPO / 'runs')))
DATA = Path(os.environ.get('ADMIT_DATA_ROOT', str(REPO / 'data')))
CS = RUNS / 'case_studies'
DOWN = RUNS / 'downstream'
UNCERT_BIO = RUNS / 'uncert_bio'          # per-cohort uncertainty outputs: cells/<cohort>.npz, cohorts/<cohort>_W.json
DANENBERG_RAW = DATA / 'tier3_raw' / 'danenberg2022_images' / 'extracted'   # img<k>_stack.tiff, img<k>_cellmask.tiff
SEED = 0
# data-use terms: no individual maps for these cohorts, aggregate numbers only
RESTRICTED = {'neotrip_tnbc_imc', 'meyer2025_tnbc_imc', 'gupta2026_normal_breast_imc', 'girnius2026_tcycif',
              'chl_mibi_stanford'}
IDX = {t: i for i, t in enumerate(NODE17)}
T_NODES = [IDX[n] for n in ('T_Lineage', 'Cytotoxic_T_Cell', 'Helper_T_Cell', 'Regulatory_T_Cell')]
POS, NEG, MYE = IDX['Macrophage_CD163pos'], IDX['Macrophage_CD163neg'], IDX['Myeloid']

# Gupta has CD163 and both macrophage labels but no run_cohort config (no outcome). Minimal in-memory config, part_b's
# non-count transform; patients: all with >= 200 mapped cells.
if 'gupta2026_normal_breast_imc' not in COHORTS:
    COHORTS['gupta2026_normal_breast_imc'] = {
        'tier': 'tier3', 'display': 'Gupta 2026 normal breast (IMC)', 'endpoints': {}, 'covariates': [],
        'exclude_flag_regex': None, 'transform': 'arcsinh_p95', 'transform_cofactor': None, 'drop_regex': 'DNA|Histone|Iridium',
        'celltype_overrides': {'treg': {'unavailable': True, 'label': 'no FOXP3'}}, 'celltype_notes': {},
        'counterfactuals': ['noCD163']}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def fit_cpu(X, y, seed=SEED):
    """run_cohort.fit_xgb with device='cpu'; everything else unchanged."""
    import xgboost as xgb
    import xgboost.callback as cb
    # With the GPU hidden, some xgboost builds raise 'no CUDA device' in the end-of-round callback even on device='cpu'.
    # fit() here has no eval sets and no callbacks, so skipping the callback does not change training.
    cb.CallbackContainer.after_iteration = lambda self, model, epoch, dtrain, evals: False
    present = np.unique(y)
    remap = np.full(max(present) + 1, -1)
    remap[present] = np.arange(len(present))
    m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, objective='multi:softprob',
                          tree_method='hist', device='cpu', random_state=seed, n_jobs=8)
    m.fit(X, remap[y], verbose=False)
    return m, present


class Cohort:
    """Arrays exactly as run_cohort.part_b builds them, restricted to mapped cells."""

    def __init__(self, name, with_markers=True):
        self.name = name
        cfg = COHORTS[name]
        self.cfg = cfg
        self.clin = load_clinical(name)
        cells = load_cells(name, with_markers=with_markers)
        self.cnt, cp, mapped = patient_counts(name, self.clin, cells)
        self.keep = np.flatnonzero(mapped)
        self.y = np.array([IDX[v] for v in cells['node17'][self.keep]], dtype=np.int64)
        self.cpk = cp[self.keep]
        if with_markers:
            if cfg.get('transform', 'arcsinh') == 'arcsinh_p95':
                Xr = cells['X'][self.keep]
                co = np.maximum(np.percentile(Xr[np.random.default_rng(0).choice(len(Xr), min(len(Xr), 500000),
                                                                                  replace=False)], 95, axis=0), 1e-6)
                self.X = np.arcsinh(Xr / co[None, :]).astype(np.float32)
                del Xr
            else:
                self.X = np.arcsinh(cells['X'][self.keep] / cfg['transform_cofactor']).astype(np.float32)
            self.vocab = cells['marker_vocab']
            self.panel = sorted({v for v in self.vocab if v != 'NA'})
        del cells
        eps = cfg['endpoints']
        if eps:
            self.prim = [e for e, d in eps.items() if d['primary']][0]
            f = endpoint_frame(name, self.clin, self.prim)
            self.frame = f[self.cnt['n_mapped'][f.pidx] >= MIN_CELLS].drop(columns=['patient_id']).reset_index(drop=True)
            self.elig = np.sort(self.frame.pidx.to_numpy())
        else:
            self.prim, self.frame = None, None
            self.elig = np.flatnonzero(self.cnt['n_mapped'] >= MIN_CELLS)
        self.by_p = pd.Series(np.arange(len(self.cpk))).groupby(self.cpk).apply(lambda s: s.to_numpy()).to_dict()

    def splits(self, seed=SEED):
        """Replays part_b's rng path. {direction: dict(fit_p, cal_p, ev, tr, ca)}. Direction 1 develops on A, evaluates on B."""
        rng = np.random.default_rng(1000 + seed)
        perm = rng.permutation(self.elig)
        h = len(perm) // 2
        halves = {'A': perm[:h], 'B': perm[h:]}
        out = {}
        for direction, (dev_k, ev_k) in {'1': ('A', 'B'), '2': ('B', 'A')}.items():
            dev, ev = halves[dev_k], halves[ev_k]
            dperm = rng.permutation(dev)
            nf = int(round(0.6 * len(dperm)))
            fit_p, cal_p = dperm[:nf], dperm[nf:]
            tr = sample_cells(rng, self.cpk, self.by_p, fit_p, 2000)
            ca = sample_cells(rng, self.cpk, self.by_p, cal_p, 2000)
            out[direction] = dict(fit_p=fit_p, cal_p=cal_p, ev=ev, tr=tr, ca=ca)
        return out

    def mask_cols(self, marker):
        return [j for j, v in enumerate(self.vocab) if v == marker]

    def model(self, sp, marker=None):
        """Fit on the direction's fit cells. marker=None: full-panel scorer ('MI'); else retrained with it set to 0 ('RT').
        Returns (model, present, mu, sd)."""
        tr = sp['tr']
        mu, sd = self.X[tr].mean(0), self.X[tr].std(0)
        sd[sd == 0] = 1.0
        Z = (self.X[tr] - mu) / sd
        if marker:
            Z[:, self.mask_cols(marker)] = 0.0
        m, present = fit_cpu(Z, self.y[tr])
        return m, present, mu, sd

    def score(self, fitted, idx, marker=None):
        """Node-17 probabilities on cells idx with `marker` (if any) masked to 0."""
        m, present, mu, sd = fitted
        out = np.empty((len(idx), len(NODE17)), dtype=np.float32)
        for i in range(0, len(idx), 500_000):
            Z = (self.X[idx[i:i + 500_000]] - mu) / sd
            if marker:
                Z[:, self.mask_cols(marker)] = 0.0
            out[i:i + 500_000] = predict17(m, present, Z, len(NODE17))
        return out


def scatter_table(name):
    """fig_scatter_seed0_crossfit.tsv with pidx restored in memory (row order = sample(frac=1, random_state=0) of sorted
    eligible patients), checked against recomputed cell counts. No id column exists in these files."""
    t = pd.read_csv(DOWN / name / 'fig_scatter_seed0_crossfit.tsv', sep='\t')
    return t


def restore_order(t, elig, n_mapped):
    order = pd.DataFrame({'i': np.arange(len(elig))}).sample(frac=1.0, random_state=0).i.to_numpy()
    t = t.copy()
    t['pidx'] = np.asarray(elig)[order]
    frac = float((t.n_mapped.round().astype(int).to_numpy() == n_mapped[t.pidx.to_numpy()]).mean())
    assert frac == 1.0, ('row order not recovered', frac)
    return t


def fit_model(D: pd.DataFrame, kind: str) -> dict:
    """Coefficient of column 'z' (per +1 SD) with every other non-T/E column as an adjustment (Cox or logistic)."""
    import warnings
    D = D.dropna()
    res = {'n': int(len(D)), 'events': int(D['E'].sum())}
    ev = D['E'].sum() if kind == 'surv' else min(D['E'].sum(), len(D) - D['E'].sum())
    if len(D) < 10 or ev < 10 or D['z'].std() == 0:
        res['error'] = 'too few patients/events or constant covariate'
        return res
    X = D.drop(columns=['T', 'E'])
    X = X.loc[:, X.std(ddof=0) > 0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            if kind == 'surv':
                from lifelines import CoxPHFitter
                m = None
                for pen in (0.0, 0.01):
                    try:
                        m = CoxPHFitter(penalizer=pen).fit(pd.concat([X, D[['T', 'E']]], axis=1), 'T', 'E')
                        break
                    except Exception:  # noqa: BLE001
                        continue
                s = m.summary.loc['z']
                b, se, p = float(s['coef']), float(s['se(coef)']), float(s['p'])
            else:
                import statsmodels.api as sm
                m = sm.Logit(D['E'], sm.add_constant(X, has_constant='add')).fit(disp=0, maxiter=200)
                b, se, p = float(m.params['z']), float(m.bse['z']), float(m.pvalues['z'])
        except Exception as e:  # noqa: BLE001
            res['error'] = 'fit failed: ' + repr(e)[:120]
            return res
    res.update({'log_hr': b, 'se': se, 'p': p, 'hr': float(np.exp(b)),
                'ci_lo': float(np.exp(b - 1.96 * se)), 'ci_hi': float(np.exp(b + 1.96 * se))})
    return res


def zs(x):
    x = x.astype(float)
    return (x - x.mean()) / x.std(ddof=1)


def tertile_flips(ref, forced):
    """Share of patients in the high tertile by one and the low tertile by the other (cuts at the 1/3 and 2/3 quantiles)."""
    ref, forced = np.asarray(ref, float), np.asarray(forced, float)
    a, b = np.quantile(ref, [1 / 3, 2 / 3])
    c, d = np.quantile(forced, [1 / 3, 2 / 3])
    return float((((ref <= a) & (forced > d)) | ((ref > b) & (forced <= c))).mean())


def share_stats(ref_n, ref_parent, for_n, for_parent):
    """Per-cohort readout for a subtype share. Patients need >= 1 parent cell in both arms."""
    from scipy.stats import spearmanr
    ref_n, ref_parent, for_n, for_parent = [np.asarray(v, float) for v in (ref_n, ref_parent, for_n, for_parent)]
    ok = (ref_parent > 0) & (for_parent > 0)
    r, f = ref_n[ok] / ref_parent[ok], for_n[ok] / for_parent[ok]
    big = ref_parent[ok] >= 50
    out = {'n_patients_input': int(len(ok)), 'n_patients': int(ok.sum()), 'n_dropped_no_parent': int((~ok).sum()),
           'pooled_share_ref': float(ref_n[ok].sum() / ref_parent[ok].sum()),
           'pooled_share_forced': float(for_n[ok].sum() / for_parent[ok].sum()),
           'median_share_ref': float(np.median(r)), 'median_share_forced': float(np.median(f)),
           'n_forced_share_zero': int((f == 0).sum()), 'n_ref_share_zero': int((r == 0).sum())}
    if ok.sum() >= 3 and np.std(r) > 0 and np.std(f) > 0:
        out['spearman'] = float(spearmanr(r, f).statistic)
        out['tertile_flip'] = tertile_flips(r, f)
    if big.sum() >= 3 and np.std(r[big]) > 0 and np.std(f[big]) > 0:
        out['n_ge50_parent'] = int(big.sum())
        out['spearman_ge50_parent'] = float(spearmanr(r[big], f[big]).statistic)
        out['tertile_flip_ge50_parent'] = tertile_flips(r[big], f[big])
    return out
