#!/usr/bin/env python3
"""Round-2 outcome analyses (plan: $ADMIT_RUNS/downstream/round2/analysis_plan_round2.json).

  r2_analyze.py e11sim  --cohort C          E11 within-cohort simulated cross-panel deployment
  r2_analyze.py e11real --pair A:B          E11 real other-panel / other-site deployment
  r2_analyze.py e13                          E13 harmonised pooling
  r2_analyze.py e12     --cohort C          E12 proportion bounds
  r2_analyze.py e10     --cohort C          E10 within-cohort prediction with uncertainty features
Outputs under $ADMIT_RUNS/downstream/round2/<exp>/. No patient identifiers are written.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds_common import COHORTS, OUT, cox, endpoint_frame, endpoint_kind, load_clinical, write_json  # noqa: E402
from run_cohort import NODE17  # noqa: E402
from admit.panel_matched import Ontology  # noqa: E402

R2 = OUT / 'round2'
TYP = R2 / 'typing'
ONTO = Ontology.from_repo(NODE17)
IDX = {t: i for i, t in enumerate(NODE17)}
CT6 = ['cyto_T', 'treg', 'helper_T', 'pooled_T', 'B', 'myeloid_mac']
CT_NODES = {'cyto_T': ['Cytotoxic_T_Cell'], 'treg': ['Regulatory_T_Cell'], 'helper_T': ['Helper_T_Cell'],
            'pooled_T': ['T_Lineage', 'Cytotoxic_T_Cell', 'Helper_T_Cell', 'Regulatory_T_Cell'],
            'B': ['B_Cell'], 'myeloid_mac': ['Myeloid', 'Macrophage_CD163neg', 'Macrophage_CD163pos']}
warnings.simplefilter('ignore')


def log(*a):
    import time
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


# ------------------------------------------------------------------------------------------------ features
def map_counts(counts17, node_to_feature, nF):
    out = np.zeros((counts17.shape[0], nF))
    for n in range(17):
        out[:, node_to_feature[n]] += counts17[:, n]
    return out


def build(z, key_prefix, arm, fm, rows, contains_key=None):
    """Feature matrix (rows = pidx) for an arm. fm = (names, node_to_feature) of the feature panel."""
    n = z[f'{key_prefix}|n'][rows].astype(float)
    n[n == 0] = np.nan
    names, n2f = fm
    nF = len(names)
    if arm == 'F1':
        X = z[f'{key_prefix}|forced17'][rows] / n[:, None]
        cols = [f'p:{t}' for t in NODE17]
    elif arm == 'F1p':
        X = map_counts(z[f'{key_prefix}|forced17'][rows], n2f, nF) / n[:, None]
        cols = [f'p:{t}' for t in names]
    elif arm in ('F2', 'F3'):
        c19 = z[f'{key_prefix}|cons19'][rows]
        P = map_counts(c19[:, :17], n2f, nF) / n[:, None]
        X, cols = P, [f'p:{t}' for t in names]
        if arm == 'F3':
            cont = z[contains_key][rows] / n[:, None]
            unc = np.c_[c19[:, 18] / n, c19[:, 17] / n, z[f'{key_prefix}|setsize'][rows] / n, cont - P]
            X = np.c_[X, unc]
            cols = cols + ['u:abstain', 'u:unresolved', 'u:setsize'] + [f'u:amb:{t}' for t in names]
    else:
        raise ValueError(arm)
    return pd.DataFrame(X, columns=cols)


def oracle(z, rows):
    g = z['gt17'][rows].astype(float)
    return pd.DataFrame(g / g.sum(1, keepdims=True), columns=[f'p:{t}' for t in NODE17])


def prep(Xtr, Xte):
    """sqrt for proportions, z-score on train, drop constant features."""
    A, B = Xtr.copy(), Xte.copy()
    for c in A.columns:
        if c.startswith('p:') or c.startswith('u:abstain') or c.startswith('u:unresolved'):
            A[c], B[c] = np.sqrt(A[c].clip(lower=0)), np.sqrt(B[c].clip(lower=0))
    mu, sd = A.mean(), A.std(ddof=0)
    ok = sd > 1e-9
    A = (A.loc[:, ok] - mu[ok]) / sd[ok]
    B = (B.loc[:, ok] - mu[ok]) / sd[ok]
    return A.fillna(0.0), B.fillna(0.0)


# ------------------------------------------------------------------------------------------------ models
def fit_predict(model, Xtr, Ttr, Etr, Xte, seed, kind='surv'):
    if Xtr.shape[1] == 0:          # nothing identifiable under the panel: constant risk (C = 0.5)
        return np.zeros(len(Xte)), np.zeros(len(Xtr))
    if kind == 'surv':
        if model == 'ridge':
            from lifelines import CoxPHFitter
            df = Xtr.copy()
            df['T'], df['E'] = Ttr, Etr
            m = CoxPHFitter(penalizer=0.1, l1_ratio=0.0).fit(df, 'T', 'E')
            return Xte.to_numpy() @ m.params_.reindex(Xte.columns).fillna(0).to_numpy(), \
                Xtr.to_numpy() @ m.params_.reindex(Xtr.columns).fillna(0).to_numpy()
        import xgboost as xgb
        y = np.where(np.asarray(Etr) == 1, np.asarray(Ttr), -np.asarray(Ttr))
        dtr = xgb.DMatrix(Xtr.to_numpy(), label=y)
        p = {'objective': 'survival:cox', 'eta': 0.05, 'max_depth': 2, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'min_child_weight': 5, 'tree_method': 'hist', 'device': 'cuda', 'seed': seed, 'verbosity': 0}
        b = xgb.train(p, dtr, 300)
        return (b.predict(xgb.DMatrix(Xte.to_numpy()), output_margin=True),
                b.predict(dtr, output_margin=True))
    if model == 'ridge':
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr.to_numpy(), np.asarray(Etr))
        return m.decision_function(Xte.to_numpy()), m.decision_function(Xtr.to_numpy())
    import xgboost as xgb
    dtr = xgb.DMatrix(Xtr.to_numpy(), label=np.asarray(Etr))
    p = {'objective': 'binary:logistic', 'eta': 0.05, 'max_depth': 2, 'subsample': 0.8, 'colsample_bytree': 0.8,
         'min_child_weight': 5, 'tree_method': 'hist', 'device': 'cuda', 'seed': seed, 'verbosity': 0}
    b = xgb.train(p, dtr, 300)
    return b.predict(xgb.DMatrix(Xte.to_numpy()), output_margin=True), b.predict(dtr, output_margin=True)


def cindex(T, E, risk, kind='surv'):
    if kind == 'surv':
        # Harrell's C (vectorised; same definition as lifelines: comparable pairs T_i < T_j with E_i = 1,
        # risk ties count 1/2; pairs with tied times are not comparable)
        T, E, r = np.asarray(T, float), np.asarray(E, int), np.asarray(risk, float)
        comp = ((T[:, None] < T[None, :]) | ((T[:, None] == T[None, :]) & (E[None, :] == 0))) & (E[:, None] == 1)
        if not comp.any():
            return np.nan
        conc = (r[:, None] > r[None, :]) + 0.5 * (r[:, None] == r[None, :])
        return float((conc * comp).sum() / comp.sum())
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(E, risk)) if 0 < np.sum(E) < len(E) else np.nan


def cal_slope(T, E, lp, kind='surv'):
    try:
        if kind == 'surv':
            from lifelines import CoxPHFitter
            m = CoxPHFitter().fit(pd.DataFrame({'lp': lp, 'T': T, 'E': E}), 'T', 'E')
            return float(m.params_['lp'])
        import statsmodels.api as sm
        m = sm.Logit(np.asarray(E), sm.add_constant(np.asarray(lp))).fit(disp=0)
        return float(m.params[1])
    except Exception:  # noqa: BLE001
        return np.nan


def shift(Xfull, XP):
    sd = Xfull.std(ddof=0).replace(0, np.nan)
    common = [c for c in Xfull.columns if c in XP.columns and c.startswith('p:')]
    return float(np.nanmean(((XP[common].mean() - Xfull[common].mean()).abs() / sd[common]).to_numpy()))


# ------------------------------------------------------------------------------------------------ E11 sim
def frame(name, ep=None):
    cfg = COHORTS[name]
    ep = ep or [e for e, v in cfg['endpoints'].items() if v['primary']][0]
    clin = load_clinical(name)
    return endpoint_frame(name, clin, ep).set_index('pidx'), endpoint_kind(name, ep), ep


def e11sim(name):
    fr, kind, ep = frame(name)
    files = sorted(TYP.glob(f'sim_{name}_s*.npz'))
    rows, boots = [], []
    for f in files:
        seed = int(f.stem.split('_s')[-1])
        z = np.load(f, allow_pickle=True)
        meta = json.loads(f.with_name(f.stem + '_meta.json').read_text())
        grp = z['group']
        tr = np.array([p for p in np.flatnonzero(grp == 'train') if p in fr.index])
        de = np.array([p for p in np.flatnonzero(grp == 'deploy') if p in fr.index])
        Ttr, Etr = fr.loc[tr, 'T'].to_numpy(), fr.loc[tr, 'E'].to_numpy()
        Tde, Ede = fr.loc[de, 'T'].to_numpy(), fr.loc[de, 'E'].to_numpy()
        risks = {}
        for model in ('ridge', 'xgb'):
            # oracle (panel independent)
            A, B = prep(oracle(z, tr), oracle(z, de))
            r, rtr = fit_predict(model, A, Ttr, Etr, B, seed, kind)
            rows.append({'cohort': name, 'seed': seed, 'model': model, 'panel': 'any', 'arm': 'oracle',
                         'C': cindex(Tde, Ede, r, kind), 'C_full': np.nan, 'slope': cal_slope(Tde, Ede, r, kind),
                         'n_train': len(tr), 'n_deploy': len(de), 'events_deploy': int(Ede.sum())})
            for pn in meta['panels']:
                fmP = (meta['feature_maps'][pn]['F'], np.array(meta['feature_maps'][pn]['node_to_feature']))
                for arm in ('F1', 'F1p', 'F2', 'F3'):
                    Xtr = build(z, 'MI|full', arm, fmP, tr, f'MI|full|contains@{pn}')
                    Xfull = build(z, 'MI|full', arm, fmP, de, f'MI|full|contains@{pn}')
                    A, Bfull = prep(Xtr, Xfull)
                    r_full, _ = fit_predict(model, A, Ttr, Etr, Bfull, seed, kind)
                    c_full = cindex(Tde, Ede, r_full, kind)
                    for sc in (('MI',) if pn == 'full' else ('MI', 'RT')):
                        XP = build(z, f'{sc}|{pn}', arm, fmP, de, f'{sc}|{pn}|contains@{pn}')
                        _, BP = prep(Xtr, XP)
                        r_P, _ = fit_predict(model, A, Ttr, Etr, BP, seed, kind)
                        risks[(model, pn, arm, sc)] = r_P
                        rows.append({'cohort': name, 'seed': seed, 'model': model, 'panel': pn, 'arm': arm,
                                     'scorer': sc, 'C': cindex(Tde, Ede, r_P, kind), 'C_full': c_full,
                                     'dC': cindex(Tde, Ede, r_P, kind) - c_full,
                                     'slope': cal_slope(Tde, Ede, r_P, kind), 'shift': shift(Xfull, XP),
                                     'n_features': A.shape[1], 'n_train': len(tr), 'n_deploy': len(de),
                                     'events_deploy': int(Ede.sum())})
        if seed == 0:
            rng = np.random.default_rng(7000)
            idx = [rng.integers(0, len(de), len(de)) for _ in range(500)]
            for (model, pn, arm, sc), r in risks.items():
                if arm not in ('F2', 'F3'):
                    continue
                ref = risks[(model, pn, 'F1' if arm == 'F2' else 'F2', sc)]
                d = [cindex(Tde[i], Ede[i], r[i], kind) - cindex(Tde[i], Ede[i], ref[i], kind) for i in idx]
                boots.append({'cohort': name, 'model': model, 'panel': pn, 'scorer': sc,
                              'contrast': f'{arm} - {"F1" if arm == "F2" else "F2"}',
                              'delta': cindex(Tde, Ede, r, kind) - cindex(Tde, Ede, ref, kind),
                              'ci_lo': float(np.nanpercentile(d, 2.5)), 'ci_hi': float(np.nanpercentile(d, 97.5))})
        log(name, 'seed', seed, 'done')
    od = R2 / 'e11'
    od.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(rows)
    R.to_csv(od / f'e11sim_{name}.tsv', sep='\t', index=False)
    pd.DataFrame(boots).to_csv(od / f'e11sim_{name}_bootstrap_seed0.tsv', sep='\t', index=False)
    R['family'] = R.panel.map(lambda p: 'full' if p == 'full' else ('rand50' if p.startswith('rand') else
                                                                    ('any' if p == 'any' else 'single')))
    S = (R.groupby(['model', 'family', 'panel', 'arm', 'scorer'], dropna=False)
         .agg(C=('C', 'mean'), C_min=('C', 'min'), C_max=('C', 'max'), dC=('dC', 'mean'), slope=('slope', 'mean'),
              shift=('shift', 'mean'), n_seeds=('C', 'size')).reset_index())
    S.to_csv(od / f'e11sim_{name}_summary.tsv', sep='\t', index=False)
    print(S[S.model == 'ridge'].round(3).to_string(index=False))


# ------------------------------------------------------------------------------------------------ E11 real
def e11real(A, B):
    frA, kindA, epA = frame(A)
    frB, kindB, epB = frame(B)
    rows = []
    for f in sorted(TYP.glob(f'xfit_{A}_s*.npz')):
        if '__target_' in f.stem:
            continue
        seed = int(f.stem.split('_s')[-1])
        fb = f.with_name(f'{f.stem}__target_{B}.npz')
        if not fb.exists():
            continue
        zA, zB = np.load(f, allow_pickle=True), np.load(fb, allow_pickle=True)
        mA = json.loads(f.with_name(f.stem + '_meta.json').read_text())
        pn = f'to:{B}'
        fm = mA['feature_maps']['fold0'][pn]
        fmP = (fm['F'], np.array(fm['node_to_feature']))
        tr = np.array([p for p in zA['elig'] if p in frA.index and zA['MI|full|n'][p] > 0])
        de = np.array([p for p in zB['elig'] if p in frB.index and zB[f'MI|{pn}|n'][p] > 0])
        Ttr, Etr = frA.loc[tr, 'T'].to_numpy(), frA.loc[tr, 'E'].to_numpy()
        Tde, Ede = frB.loc[de, 'T'].to_numpy(), frB.loc[de, 'E'].to_numpy()
        for model in ('ridge', 'xgb'):
            Ao, Bo = prep(oracle(zA, tr), oracle(zB, de))
            r, _ = fit_predict(model, Ao, Ttr, Etr, Bo, seed, kindA)
            rows.append({'pair': f'{A}->{B}', 'seed': seed, 'model': model, 'arm': 'oracle', 'scorer': '',
                         'C_deploy': cindex(Tde, Ede, r, kindB), 'slope': cal_slope(Tde, Ede, r, kindB),
                         'C_cv_A': cv_c(Ao, Ttr, Etr, model, seed, kindA), 'n_A': len(tr), 'events_A': int(Etr.sum()),
                         'n_B': len(de), 'events_B': int(Ede.sum())})
            for arm in ('F1', 'F1p', 'F2', 'F3'):
                Xtr = build(zA, 'MI|full', arm, fmP, tr, f'MI|full|contains@{pn}')
                for sc in ('MI', 'RT'):
                    XB = build(zB, f'{sc}|{pn}', arm, fmP, de, f'{sc}|{pn}|contains@{pn}')
                    a_, b_ = prep(Xtr, XB)
                    r, _ = fit_predict(model, a_, Ttr, Etr, b_, seed, kindA)
                    rows.append({'pair': f'{A}->{B}', 'seed': seed, 'model': model, 'arm': arm, 'scorer': sc,
                                 'C_deploy': cindex(Tde, Ede, r, kindB), 'slope': cal_slope(Tde, Ede, r, kindB),
                                 'C_cv_A': cv_c(a_, Ttr, Etr, model, seed, kindA), 'shift_vs_A': shift(Xtr, XB),
                                 'n_A': len(tr), 'events_A': int(Etr.sum()), 'n_B': len(de),
                                 'events_B': int(Ede.sum())})
        log(A, '->', B, 'seed', seed)
    od = R2 / 'e11'
    od.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(rows)
    R.to_csv(od / f'e11real_{A}__{B}.tsv', sep='\t', index=False)
    S = (R.groupby(['model', 'arm', 'scorer']).agg(C_deploy=('C_deploy', 'mean'), C_min=('C_deploy', 'min'),
                                                    C_max=('C_deploy', 'max'), slope=('slope', 'mean'),
                                                    C_cv_A=('C_cv_A', 'mean'), n_seeds=('C_deploy', 'size'))
         .reset_index())
    S.to_csv(od / f'e11real_{A}__{B}_summary.tsv', sep='\t', index=False)
    print(S.round(3).to_string(index=False))


def cv_c(X, T, E, model, seed, kind, k=5):
    rng = np.random.default_rng(5000 + seed)
    fold = rng.permutation(np.arange(len(X)) % k)
    cs = []
    for j in range(k):
        te = fold == j
        if te.sum() < 5 or np.sum(np.asarray(E)[te]) < 1:
            continue
        r, _ = fit_predict(model, X[~te].reset_index(drop=True), np.asarray(T)[~te], np.asarray(E)[~te],
                           X[te].reset_index(drop=True), seed, kind)
        cs.append(cindex(np.asarray(T)[te], np.asarray(E)[te], r, kind))
    return float(np.nanmean(cs)) if cs else np.nan


# ------------------------------------------------------------------------------------------------ E10
def e10(name, ep=None):
    fr, kind, ep = frame(name, ep)
    rows = []
    for f in sorted(TYP.glob(f'xfit_{name}_s*.npz')):
        if '__target_' in f.stem:
            continue
        seed = int(f.stem.split('_s')[-1])
        z = np.load(f, allow_pickle=True)
        meta = json.loads(f.with_name(f.stem + '_meta.json').read_text())
        fm = meta['feature_maps']['fold0']['full']
        fmP = (fm['F'], np.array(fm['node_to_feature']))
        pts = np.array([p for p in z['elig'] if p in fr.index and z['MI|full|n'][p] > 0])
        T, E = fr.loc[pts, 'T'].to_numpy(), fr.loc[pts, 'E'].to_numpy()
        feats = {'oracle': oracle(z, pts)}
        for arm in ('F1', 'F2', 'F3'):
            feats[arm] = build(z, 'MI|full', arm, fmP, pts, 'MI|full|contains@full')
        rng = np.random.default_rng(5000 + seed)
        fold = rng.permutation(np.arange(len(pts)) % 5)
        for model in ('ridge', 'xgb'):
            for arm, X in feats.items():
                cs = []
                for j in range(5):
                    te = fold == j
                    if kind == 'surv' and E[te].sum() < 2:
                        continue
                    a_, b_ = prep(X[~te].reset_index(drop=True), X[te].reset_index(drop=True))
                    r, _ = fit_predict(model, a_, T[~te], E[~te], b_, seed, kind)
                    cs.append(cindex(T[te], E[te], r, kind))
                rows.append({'cohort': name, 'endpoint': ep, 'kind': kind, 'seed': seed, 'model': model, 'arm': arm,
                             'C': float(np.nanmean(cs)), 'n': len(pts), 'events': int(E.sum())})
        log(name, ep, 'seed', seed)
    od = R2 / 'e10'
    od.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(rows)
    R.to_csv(od / f'e10_{name}_{ep}.tsv', sep='\t', index=False)
    P = R.pivot_table(index=['seed', 'model'], columns='arm', values='C').reset_index()
    P['F3_minus_F2'] = P['F3'] - P['F2']
    P['F2_minus_F1'] = P['F2'] - P['F1']
    S = P.groupby('model').agg(['mean', 'min', 'max']).drop(columns='seed', level=0)
    S.to_csv(od / f'e10_{name}_{ep}_summary.tsv', sep='\t')
    print(name, ep)
    print(S.round(3).to_string())


# ------------------------------------------------------------------------------------------------ E12
def e12(name):
    cfg = COHORTS[name]
    fr, kind, ep = frame(name)
    clin_fr = fr
    rows, sens = [], []
    for f in sorted(TYP.glob(f'sim_{name}_s*.npz')):
        seed = int(f.stem.split('_s')[-1])
        z = np.load(f, allow_pickle=True)
        meta = json.loads(f.with_name(f.stem + '_meta.json').read_text())
        typed = np.flatnonzero(np.isin(z['group'], ['train', 'deploy']))
        for pn in meta['panels']:
            for sc in (('MI',) if pn == 'full' else ('MI', 'RT')):
                k = f'{sc}|{pn}'
                n = z[f'{k}|n'][typed].astype(float)
                LU = z[f'{k}|LU6'][typed] / n[:, None, None]
                c19 = z[f'{k}|cons19'][typed]
                for j, ct in enumerate(CT6):
                    if f'gt6|{ct}' not in z.files:
                        continue
                    truth = z[f'gt6|{ct}'][typed] / n
                    L, U = LU[:, j, 0], LU[:, j, 1]
                    pt = c19[:, [IDX[x] for x in CT_NODES[ct]]].sum(1) / n
                    rows.append({'cohort': name, 'seed': seed, 'panel': pn, 'scorer': sc, 'celltype': ct,
                                 'coverage': float(np.mean((L <= truth + 1e-12) & (truth <= U + 1e-12))),
                                 'width_pp': float(np.mean(U - L) * 100), 'mean_L_pct': float(np.mean(L) * 100),
                                 'mean_U_pct': float(np.mean(U) * 100), 'mean_truth_pct': float(np.mean(truth) * 100),
                                 'mean_point_pct': float(np.mean(pt) * 100), 'n_patients': len(typed)})
                    if seed == 0 and pn in ('full', 'del_CD8', 'del_FOXP3', 'del_CD20', 'del_CD68'):
                        d = clin_fr.reindex(typed).copy()
                        d = d.assign(L=L, U=U, P=pt, G=truth).dropna(subset=['T', 'E'])
                        adj = [c for c in cfg['covariates']]
                        res = {v: cox(d, v, adj, 'T', 'E', ph_test=False) for v in ('G', 'P', 'L', 'U')}

                        def st(r):
                            return (np.sign(r.get('log_hr', np.nan)), r.get('p', 1) < 0.05) if 'hr' in r else None
                        sens.append({'cohort': name, 'panel': pn, 'scorer': sc, 'celltype': ct,
                                     **{f'hr_{v}': r.get('hr') for v, r in res.items()},
                                     **{f'lo_{v}': r.get('ci_lo') for v, r in res.items()},
                                     **{f'hi_{v}': r.get('ci_hi') for v, r in res.items()},
                                     **{f'p_{v}': r.get('p') for v, r in res.items()},
                                     'stable_L_U': (st(res['L']) == st(res['U'])) if st(res['L']) and st(res['U'])
                                     else None,
                                     'stable_with_truth': (st(res['L']) == st(res['U']) == st(res['G']))
                                     if st(res['L']) and st(res['U']) and st(res['G']) else None})
        log(name, 'e12 seed', seed)
    od = R2 / 'e12'
    od.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(rows)
    R.to_csv(od / f'e12_{name}.tsv', sep='\t', index=False)
    pd.DataFrame(sens).to_csv(od / f'e12_{name}_sensitivity_seed0.tsv', sep='\t', index=False)
    S = R.groupby(['panel', 'scorer', 'celltype']).agg(coverage=('coverage', 'mean'), width_pp=('width_pp', 'mean'),
                                                       truth=('mean_truth_pct', 'mean')).reset_index()
    S.to_csv(od / f'e12_{name}_summary.tsv', sep='\t', index=False)
    print(S[S.panel.isin(['full', 'del_CD8', 'del_FOXP3', 'rand50_s0'])].round(3).to_string(index=False))
    print(pd.DataFrame(sens)[['panel', 'scorer', 'celltype', 'hr_G', 'hr_P', 'hr_L', 'hr_U', 'p_G', 'p_L', 'p_U',
                              'stable_L_U']].round(3).to_string(index=False))


# ------------------------------------------------------------------------------------------------ E13
POOLS = {
    'NSCLC_OS': {'ref': 'cords2024_nsclc_imc', 'members': [('cords2024_nsclc_imc', 'os'),
                                                           ('kulasinghe2026_nsclc_pcf', 'os'),
                                                           ('monkman2026_nsclc_pcf', 'os')]},
    'TNBC_OS': {'ref': 'quiche_tnbc_mibi_spain', 'members': [('fischer2023_breast_imc', 'os'),
                                                             ('meyer2025_tnbc_imc', 'os')]},
    'TNBC_recurrence': {'ref': 'quiche_tnbc_mibi_spain', 'members': [('quiche_tnbc_mibi_spain', 'rfs'),
                                                                     ('quiche_tnbc_mibi_stanford', 'rfs'),
                                                                     ('meyer2025_tnbc_imc', 'dfs')]},
}


def common_map(ref_labels, shared_lists):
    pcap = set(shared_lists[0])
    for s in shared_lists[1:]:
        pcap &= set(s)
    adm = ONTO.admissible(sorted(pcap))
    proj = ONTO.projection_map(adm)
    F = sorted({int(proj[l]) for l in ref_labels if proj[l] >= 0})
    n2f = []
    for n in range(17):
        c = proj[n]
        while c >= 0 and c not in F:
            c = ONTO.parent_idx[c]
        n2f.append(c)
    return sorted(pcap), [NODE17[c] for c in F], np.array(n2f)


def e13(seed=0, scorers=('MI', 'RT')):
    from lifelines import CoxPHFitter
    out, rows = {}, []
    for pool, spec in POOLS.items():
        ref = spec['ref']
        fref = TYP / f'xfit_{ref}_s{seed}.npz'
        zref = np.load(fref, allow_pickle=True)
        ref_labels = np.flatnonzero(zref['gt17'].sum(0) > 0)
        shared = {}
        for c, _ in spec['members']:
            if c == ref:
                continue
            m = json.loads((TYP / f'xfit_{ref}_s{seed}__target_{c}_meta.json').read_text())
            shared[c] = m['shared_vocab']
        pcap, Fc, n2f = common_map(ref_labels, list(shared.values()))
        out[pool] = {'reference': ref, 'P_cap': pcap, 'F_common': Fc}
        for sc in scorers:
            recs = []
            for c, ep in spec['members']:
                fr, kind, _ = frame(c, ep)
                if c == ref:
                    z, key = zref, 'MI|full'
                    if sc == 'RT':
                        key = 'MI|full'     # reference typed under its own full panel (no retraining needed)
                else:
                    z = np.load(TYP / f'xfit_{ref}_s{seed}__target_{c}.npz', allow_pickle=True)
                    key = f'{sc}|to:{c}'
                pts = np.array([p for p in z['elig'] if p in fr.index and z[f'{key}|n'][p] > 0])
                if c == 'fischer2023_breast_imc':
                    cl = load_clinical(c)
                    tn = set(cl.pidx[cl.mol_signature == 'Triple negative'])
                    pts = np.array([p for p in pts if p in tn])
                n = z[f'{key}|n'][pts].astype(float)
                cons = z[f'{key}|cons19'][pts][:, :17]
                forced = z[f'{key}|forced17'][pts]
                gt = z['gt17'][pts]
                d = fr.loc[pts, ['T', 'E']].copy()
                d['cohort'] = c
                for ct in CT6:
                    nodes = [IDX[x] for x in CT_NODES[ct]]
                    # ADMIT at F_common: cells whose common feature lies in subtree(ct); only if ct is resolvable
                    ok_nodes = [IDX[x] for x in CT_NODES[ct] if x in Fc]
                    if ct == 'pooled_T':
                        ok_nodes = [IDX[x] for x in CT_NODES[ct] if x in Fc] or ([IDX['T_Lineage']]
                                                                                 if 'T_Lineage' in Fc else [])
                    feat_counts = np.zeros(len(pts))
                    for nn in range(17):
                        if n2f[nn] >= 0 and n2f[nn] in ok_nodes:
                            feat_counts += cons[:, nn]
                    d[f'admit|{ct}'] = feat_counts / n if ok_nodes else np.nan
                    d[f'forced|{ct}'] = forced[:, nodes].sum(1) / n
                    gkey = f'gt6|{ct}'
                    d[f'naive|{ct}'] = (z[gkey][pts] / n) if gkey in z.files else np.nan
                recs.append(d)
            D = pd.concat(recs)
            for ct in CT6:
                for ver in ('admit', 'forced', 'naive'):
                    col = f'{ver}|{ct}'
                    sub = D[['T', 'E', 'cohort', col]].dropna()
                    sub = sub[sub.groupby('cohort')[col].transform('std') > 0]
                    cohorts_in = [c for c, g in sub.groupby('cohort') if g.E.sum() >= 10]
                    sub = sub[sub.cohort.isin(cohorts_in)]
                    if len(cohorts_in) < 2:
                        rows.append({'pool': pool, 'scorer': sc, 'celltype': ct, 'version': ver,
                                     'n_cohorts': len(cohorts_in), 'note': 'fewer than 2 cohorts'})
                        continue
                    for zs in ('pooled', 'within'):
                        s2 = sub.copy()
                        if zs == 'pooled':
                            s2['z'] = (s2[col] - s2[col].mean()) / s2[col].std()
                        else:
                            s2['z'] = s2.groupby('cohort')[col].transform(lambda v: (v - v.mean()) / v.std())
                        try:
                            m = CoxPHFitter().fit(s2[['T', 'E', 'cohort', 'z']], 'T', 'E', strata=['cohort'])
                            b, se = float(m.params_['z']), float(m.standard_errors_['z'])
                        except Exception:  # noqa: BLE001
                            b, se = np.nan, np.nan
                        per = []
                        for cc, g in s2.groupby('cohort'):
                            try:
                                mm = CoxPHFitter().fit(g[['T', 'E', 'z']], 'T', 'E')
                                per.append((cc, float(mm.params_['z']), float(mm.standard_errors_['z'])))
                            except Exception:  # noqa: BLE001
                                pass
                        bs = np.array([p[1] for p in per])
                        ses = np.array([p[2] for p in per])
                        w = 1 / ses ** 2
                        biv = float(np.sum(w * bs) / np.sum(w))
                        Q = float(np.sum(w * (bs - biv) ** 2))
                        I2 = max(0.0, (Q - (len(bs) - 1)) / Q) if Q > 0 else 0.0
                        rows.append({'pool': pool, 'scorer': sc, 'celltype': ct, 'version': ver, 'zscore': zs,
                                     'n_cohorts': len(cohorts_in), 'n': len(s2), 'events': int(s2.E.sum()),
                                     'hr_strat': np.exp(b), 'se_strat': se, 'precision': 1 / se ** 2 if se else np.nan,
                                     'lo_strat': np.exp(b - 1.96 * se), 'hi_strat': np.exp(b + 1.96 * se),
                                     'hr_iv': np.exp(biv), 'Q': Q, 'I2': I2,
                                     'per_cohort': json.dumps({p[0]: [round(np.exp(p[1]), 3), round(p[2], 3)]
                                                               for p in per})})
        log('e13', pool)
    od = R2 / 'e13'
    od.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(rows)
    R.to_csv(od / f'e13_seed{seed}.tsv', sep='\t', index=False)
    write_json(out, od / f'e13_seed{seed}_common_resolution.json')
    show = R[(R.get('zscore') == 'pooled')] if 'zscore' in R else R
    print(json.dumps(out, indent=0)[:1500])
    print(show[['pool', 'scorer', 'celltype', 'version', 'n_cohorts', 'events', 'hr_strat', 'lo_strat', 'hi_strat',
                'se_strat', 'I2']].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exp', choices=['e11sim', 'e11real', 'e13', 'e12', 'e10'])
    ap.add_argument('--cohort')
    ap.add_argument('--pair')
    ap.add_argument('--endpoint')
    a = ap.parse_args()
    if a.exp == 'e11sim':
        e11sim(a.cohort)
    elif a.exp == 'e11real':
        A, B = a.pair.split(':')
        e11real(A, B)
    elif a.exp == 'e13':
        e13()
    elif a.exp == 'e12':
        e12(a.cohort)
    else:
        e10(a.cohort, a.endpoint)


if __name__ == '__main__':
    main()
