#!/usr/bin/env python3
"""E14: uncertainty as QC. Simulated staining failure in 10% of test samples; per-sample ADMIT set statistics vs
forced-classifier confidence as failure detectors (AUROC). Plan: round2/analysis_plan_round2.json (E14).

  r2_qc.py --cohort cords2024_nsclc_imc --seed 0
  r2_qc.py --cohort schurch --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # experiments/ (schurch_common)
from ds_common import OUT, TIER1, write_json  # noqa: E402
from run_cohort import NODE17, consensus_node, fit_xgb, predict17  # noqa: E402
from admit.panel_matched import AdmitM1, Ontology  # noqa: E402

R2 = OUT / 'round2'
ONTO = Ontology.from_repo(NODE17)
IDX = {t: i for i, t in enumerate(NODE17)}
R_MARKERS = sorted({m for n in NODE17 for m in ONTO.R[n]} | {'CD45RO', 'CD45RA'})


def load_cords():
    import pyarrow.parquet as pq
    from r2_type import load
    d = load('cords2024_nsclc_imc')
    t = pq.read_table(TIER1 / 'cords2024_nsclc_imc' / 'cells.parquet',
                      columns=['node17', 'sample_id'])
    n17 = np.asarray(t.column('node17').to_numpy(zero_copy_only=False)).astype(str)
    sid = np.asarray(t.column('sample_id').to_numpy(zero_copy_only=False)).astype(str)
    # same row filter as load(): mapped cells of patients with a clinical row
    from ds_common import load_clinical
    clin = load_clinical('cords2024_nsclc_imc')
    import pyarrow.parquet as pq2
    pid = np.asarray(pq2.read_table(TIER1 / 'cords2024_nsclc_imc' / 'cells.parquet',
                                    columns=['patient_id']).column('patient_id').to_numpy(zero_copy_only=False))
    ok = (n17 != 'UNMAPPED') & np.isin(pid.astype(str), clin.patient_id.astype(str).to_numpy())
    sid = sid[ok]
    assert len(sid) == len(d['y'])
    raw = lambda X: np.sinh(X)            # noqa: E731  arcsinh(raw/1)
    back = lambda R: np.arcsinh(R)        # noqa: E731
    return {'X': d['X'], 'y': d['y'], 'pat': d['pat'].astype(str), 'sample': sid, 'vocab': d['vocab'],
            'raw': raw, 'back': back}


def load_schurch_qc():
    from schurch_common import SRC, load_schurch
    d = load_schurch()
    df = pd.read_csv(SRC, usecols=['ClusterName', 'File Name'], low_memory=False)
    mp = json.loads((OUT.parent / 'schurch' / 'schurch_label_mapping.json').read_text())
    df = df.loc[~df.ClusterName.isin(mp['excluded_unmapped_labels'])].reset_index(drop=True)
    assert len(df) == len(d['x'])
    y = np.array([IDX[v] for v in d['label_name']])
    return {'X': d['x'].astype(np.float32), 'y': y, 'pat': np.asarray(d['patient']).astype(str),
            'sample': df['File Name'].astype(str).to_numpy(), 'vocab': list(d['markers']),
            'raw': np.expm1, 'back': np.log1p}


def entropy(P):
    P = np.clip(P, 1e-12, 1)
    return -(P * np.log(P)).sum(1)


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', required=True)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    d = load_cords() if a.cohort.startswith('cords') else load_schurch_qc()
    rng = np.random.default_rng(6000 + a.seed)
    pats = rng.permutation(np.unique(d['pat']))
    n = len(pats)
    fitp, calp, testp = pats[: int(0.25 * n)], pats[int(0.25 * n): int(0.40 * n)], pats[int(0.40 * n):]
    by_p = pd.Series(np.arange(len(d['y']))).groupby(d['pat']).apply(lambda s: s.to_numpy()).to_dict()

    def samp(ps, cap):
        return np.sort(np.concatenate([by_p[p] if len(by_p[p]) <= cap else rng.choice(by_p[p], cap, replace=False)
                                       for p in ps]))
    tr, ca = samp(fitp, 2000), samp(calp, 2000)
    mu, sd = d['X'][tr].mean(0), d['X'][tr].std(0)
    sd[sd == 0] = 1
    model, present = fit_xgb((d['X'][tr] - mu) / sd, d['y'][tr], a.seed)
    panel = [v for v in d['vocab'] if v != 'NA']
    Pc = predict17(model, present, (d['X'][ca] - mu) / sd, ONTO.K)
    adm = AdmitM1(ONTO, 0.10).fit(ONTO.aggregate(Pc), d['y'][ca], panel)
    test_cells = np.flatnonzero(np.isin(d['pat'], testp))
    samples = np.unique(d['sample'][test_cells])
    fail = rng.random(len(samples)) < 0.10
    rmk = [j for j, v in enumerate(d['vocab']) if v in set(R_MARKERS)]
    fail_marker = {s: int(rng.choice(rmk)) for s in samples[fail]}
    rows = []
    for mode in ('dim0.2', 'noise1sd'):
        for s in samples:
            ci = test_cells[d['sample'][test_cells] == s]
            if len(ci) > 2000:
                ci = np.sort(rng.choice(ci, 2000, replace=False))
            X = d['X'][ci].copy()
            failed = s in fail_marker
            if failed:
                j = fail_marker[s]
                if mode == 'dim0.2':
                    X[:, j] = d['back'](d['raw'](X[:, j]) * 0.2)
            Z = (X - mu) / sd
            if failed and mode == 'noise1sd':
                Z[:, fail_marker[s]] += rng.normal(0, 1, len(ci)).astype(np.float32)
            P = predict17(model, present, Z.astype(np.float32), ONTO.K)
            sets = adm.predict(ONTO.aggregate(P))
            size = sets.sum(1)
            cons = consensus_node(sets, ONTO)
            rows.append({'cohort': a.cohort, 'seed': a.seed, 'mode': mode, 'failed': int(failed),
                         'marker': d['vocab'][fail_marker[s]] if failed else '', 'n_cells': len(ci),
                         'admit_nonsingleton': float(np.mean(size != 1)), 'admit_setsize': float(size.mean()),
                         'admit_abstain': float(np.mean(size == 0)), 'admit_unresolved': float(np.mean(cons == -1)),
                         'forced_1mmaxprob': float(np.mean(1 - P.max(1))), 'forced_entropy': float(entropy(P).mean())})
    R = pd.DataFrame(rows)
    od = R2 / 'e14'
    od.mkdir(parents=True, exist_ok=True)
    R.to_csv(od / f'e14_{a.cohort}_s{a.seed}_samples.tsv', sep='\t', index=False)
    res = []
    for mode, g in R.groupby('mode'):
        y = g.failed.to_numpy()
        for sc in ('admit_nonsingleton', 'admit_setsize', 'admit_abstain', 'admit_unresolved', 'forced_1mmaxprob',
                   'forced_entropy'):
            res.append({'cohort': a.cohort, 'seed': a.seed, 'mode': mode, 'score': sc, 'auroc': auroc(y, g[sc]),
                        'n_samples': len(g), 'n_failed': int(y.sum())})
            for mk, gg in g[g.failed == 1].groupby('marker'):
                yy = np.r_[np.ones(len(gg)), np.zeros(int((g.failed == 0).sum()))]
                ss = np.r_[gg[sc], g.loc[g.failed == 0, sc]]
                res.append({'cohort': a.cohort, 'seed': a.seed, 'mode': mode, 'score': sc, 'marker': mk,
                            'auroc': auroc(yy, ss), 'n_failed': len(gg)})
    pd.DataFrame(res).to_csv(od / f'e14_{a.cohort}_s{a.seed}_auroc.tsv', sep='\t', index=False)
    print(pd.DataFrame(res).query('marker != marker')[['mode', 'score', 'auroc', 'n_samples', 'n_failed']]
          .round(3).to_string(index=False))
    print('done', round(time.time() - t0, 1))


if __name__ == '__main__':
    main()
