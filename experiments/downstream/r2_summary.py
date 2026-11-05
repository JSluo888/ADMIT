#!/usr/bin/env python3
"""Round-2 summary: NEGATIVE_RESULTS.tsv (every prespecified hypothesis, test, estimate with CI, verdict),
figure-ready tables, and round2/MANIFEST.tsv. Reads only round-1/round-2 outputs."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DS = Path(os.environ.get('ADMIT_RUNS', str(Path(__file__).resolve().parents[2] / 'runs'))) / 'downstream'
R2 = DS / 'round2'


def tci(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) < 2:
        return (float(np.mean(v)) if len(v) else np.nan, np.nan, np.nan)
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    h = stats.t.ppf(0.975, len(v) - 1) * se
    return float(m), float(m - h), float(m + h)


def fmt(m, lo, hi, d=3):
    return f'{m:.{d}f} ({lo:.{d}f}, {hi:.{d}f})'


def e11_family_table(cohort='cords2024_nsclc_imc', model='ridge'):
    R = pd.read_csv(R2 / 'e11' / f'e11sim_{cohort}.tsv', sep='\t')
    R = R[(R.model == model) & (R.panel != 'any')]
    R['family'] = R.panel.map(lambda p: 'full' if p == 'full' else ('rand50' if p.startswith('rand') else 'single'))
    R['slope0'] = R.slope.fillna(0.0)
    R['arm_sc'] = R.arm + '-' + R.scorer.fillna('MI')
    per_seed = R.groupby(['family', 'arm_sc', 'seed']).agg(dC=('dC', 'mean'), C=('C', 'mean'),
                                                           slope=('slope0', 'mean')).reset_index()
    rows = []
    for (fam, arm), g in per_seed.groupby(['family', 'arm_sc']):
        rows.append({'family': fam, 'arm': arm, **{f'{k}_{s}': v for k in ('dC', 'C', 'slope')
                                                  for s, v in zip(('mean', 'lo', 'hi'), tci(g[k]))}})
    T = pd.DataFrame(rows)
    # paired per-seed differences vs forced-MI
    diffs = []
    for fam in ('single', 'rand50'):
        base = per_seed[(per_seed.family == fam) & (per_seed.arm_sc == 'F1-MI')].set_index('seed')
        for arm in ('F2-MI', 'F2-RT', 'F1p-MI', 'F1-RT', 'F3-MI'):
            g = per_seed[(per_seed.family == fam) & (per_seed.arm_sc == arm)].set_index('seed')
            for k in ('dC', 'slope'):
                d = (g[k] - base[k]).dropna()
                diffs.append({'family': fam, 'arm': arm, 'metric': k, 'vs': 'F1-MI', **dict(zip(('mean', 'lo', 'hi'),
                                                                                             tci(d)))})
        g2 = per_seed[(per_seed.family == fam) & (per_seed.arm_sc == 'F2-MI')].set_index('seed')
        for ref in ('F1p-MI', 'F1-RT'):
            b2 = per_seed[(per_seed.family == fam) & (per_seed.arm_sc == ref)].set_index('seed')
            for k in ('dC', 'slope'):
                diffs.append({'family': fam, 'arm': 'F2-MI', 'metric': k, 'vs': ref,
                              **dict(zip(('mean', 'lo', 'hi'), tci((g2[k] - b2[k]).dropna())))})
        b3 = per_seed[(per_seed.family == fam) & (per_seed.arm_sc == 'F2-RT')].set_index('seed')
        b4 = per_seed[(per_seed.family == fam) & (per_seed.arm_sc == 'F1-RT')].set_index('seed')
        diffs.append({'family': fam, 'arm': 'F2-RT', 'metric': 'dC', 'vs': 'F1-RT',
                      **dict(zip(('mean', 'lo', 'hi'), tci((b3['dC'] - b4['dC']).dropna())))})
    # F3 - F2 on C per family (incl. full)
    for fam in ('full', 'single', 'rand50'):
        for sc in ('MI', 'RT'):
            a = per_seed[(per_seed.family == fam) & (per_seed.arm_sc == f'F3-{sc}')].set_index('seed')
            b = per_seed[(per_seed.family == fam) & (per_seed.arm_sc == f'F2-{sc}')].set_index('seed')
            if len(a) and len(b):
                diffs.append({'family': fam, 'arm': f'F3-{sc}', 'metric': 'C', 'vs': f'F2-{sc}',
                              **dict(zip(('mean', 'lo', 'hi'), tci((a['C'] - b['C']).dropna())))})
    # feature shift (median over panels per seed)
    Rs = R.groupby(['family', 'arm_sc', 'seed'])['shift'].median().reset_index()
    for fam in ('single', 'rand50'):
        for arm in ('F1-MI', 'F2-MI', 'F1-RT', 'F2-RT'):
            g = Rs[(Rs.family == fam) & (Rs.arm_sc == arm)]
            diffs.append({'family': fam, 'arm': arm, 'metric': 'shift_median', 'vs': '',
                          **dict(zip(('mean', 'lo', 'hi'), tci(g['shift'])))})
    return T, pd.DataFrame(diffs)


def main():
    T, D = e11_family_table()
    T.to_csv(R2 / 'e11' / 'e11sim_cords_family_table.tsv', sep='\t', index=False)
    D.to_csv(R2 / 'e11' / 'e11sim_cords_paired_differences.tsv', sep='\t', index=False)
    neg = []

    def add(hyp, test, est, verdict, source, note=''):
        neg.append({'hypothesis': hyp, 'test': test, 'estimate_ci': est, 'verdict': verdict, 'source': source,
                    'note': note})

    def d(fam, arm, metric, vs='F1-MI'):
        r = D[(D.family == fam) & (D.arm == arm) & (D.metric == metric) & (D.vs == vs)].iloc[0]
        return r['mean'], r['lo'], r['hi']

    # ---- round-1 context (impute-then-classify distortions; ADMIT supported level)
    pb = {c: json.loads((DS / c / 'partB.json').read_text())['bootstrap_seed0_crossfit']['comparisons']
          for c in ('cords2024_nsclc_imc', 'neotrip_tnbc_imc', 'kulasinghe2026_nsclc_pcf')}

    def br(c, k):
        v = pb[c].get(k, {})
        return (f"{v['ratio_of_hr']:.2f} ({v['ratio_ci'][0]:.2f}, {v['ratio_ci'][1]:.2f})" if 'ratio_ci' in v
                else 'not estimable')
    add('R1a impute-then-classify distorts subtype associations when the defining marker is missing',
        'ratio of HR (forced, CD8 imputed / author labels), seed-0 cross-fit, patient bootstrap',
        f"Cords OS: {br('cords2024_nsclc_imc', 'forced_noCD8__cyto_T minus GT__cyto_T')} (0% of cells called "
        f"cytotoxic T); NeoTRIP pCR OR: {br('neotrip_tnbc_imc', 'forced_noCD8__cyto_T minus GT__cyto_T')}; "
        f"Kulasinghe OS: forced 1.49 (1.03, 2.16) vs author 0.91 (0.51, 1.65), from 0.08% vs 5.5% cells",
        'supported (round 1)', 'round-1 partB.json')
    add('R1b FOXP3 removal attenuates the Treg association even for a retrained forced classifier',
        'ratio of HR (forced retrained without FOXP3 / author Treg), Cords OS',
        br('cords2024_nsclc_imc', 'forcedRT_noFOXP3__treg minus GT__treg') + '; author 0.88 (0.81, 0.95) -> '
        'forced 0.93 (0.86, 1.00)', 'supported (round 1)', 'round-1 partB.json')
    add('R1c ADMIT supported-level (T cell) HR matches author-label T-cell HR without CD8',
        'ratio of HR (ADMIT T level / author T level), Cords OS',
        f"imputed scorer {br('cords2024_nsclc_imc', 'admit_noCD8__pooled_T minus GT__pooled_T')}; retrained scorer "
        f"{br('cords2024_nsclc_imc', 'admitRT_noCD8__pooled_T minus GT__pooled_T')}",
        'partly: matches with a retrained scorer, attenuated with the imputed scorer', 'round-1 partB.json')
    # ---- E11
    m, lo, hi = d('rand50', 'F2-MI', 'dC')
    add('H11a ADMIT features lose less C than forced (impute-then-classify) under random-50% panels',
        'paired per-seed difference in dC (ADMIT-MI minus forced-MI), Cords OS, ridge Cox, 5 seeds x 5 panels',
        fmt(m, lo, hi), 'supported' if lo > 0 else 'not supported', 'e11/e11sim_cords_paired_differences.tsv')
    m, lo, hi = d('rand50', 'F2-MI', 'slope')
    add('H11a-slope ADMIT keeps calibration slope better than impute-then-classify (random-50%)',
        'paired per-seed difference in calibration slope (ADMIT-MI minus forced-MI)', fmt(m, lo, hi),
        'supported' if lo > 0 else 'not supported', 'e11/e11sim_cords_paired_differences.tsv')
    m, lo, hi = d('single', 'F2-MI', 'dC')
    add('H11a-single same under single-marker deletions (CD8, FOXP3, CD20, CD68)',
        'paired per-seed difference in dC (ADMIT-MI minus forced-MI)', fmt(m, lo, hi),
        ('CI excludes 0 but negligible (< 0.005 C)' if lo > 0 and m < 0.005 else
         ('supported' if lo > 0 else 'not supported (no difference)')), 'e11/e11sim_cords_paired_differences.tsv')
    # absolute deploy C and descriptive slope distribution under random-50% panels (25 panel x seed cases)
    Rr = pd.read_csv(R2 / 'e11' / 'e11sim_cords2024_nsclc_imc.tsv', sep='\t')
    Rr = Rr[(Rr.model == 'ridge') & Rr.panel.str.startswith('rand')]
    Rr['a'] = Rr.arm + '-' + Rr.scorer
    ps = Rr.groupby(['a', 'seed']).C.mean().unstack(0)
    m, lo, hi = tci(ps['F2-MI'] - ps['F1-MI'])
    sl = Rr.groupby('a').slope.describe(percentiles=[.25, .5, .75])
    add('H11a-absC deploy C of ADMIT vs impute-then-classify (random-50%)',
        'paired per-seed difference in absolute deploy C (ADMIT-MI minus forced-MI)',
        fmt(m, lo, hi) + f"; mean C forced-MI {ps['F1-MI'].mean():.3f}, ADMIT-MI {ps['F2-MI'].mean():.3f}, "
        f"coarsened forced {ps['F1p-MI'].mean():.3f}, forced-RT {ps['F1-RT'].mean():.3f}, ADMIT-RT "
        f"{ps['F2-RT'].mean():.3f}; full-panel forced 0.551",
        'not supported (CI includes 0)' if lo <= 0 else 'supported', 'e11/e11sim_cords2024_nsclc_imc.tsv')
    add('H11a-slope-descriptive calibration slope stability (random-50%, 25 panel x seed cases)',
        'median [IQR] (min, max) of the calibration slope; descriptive, ideal 1',
        '; '.join(f"{k} {sl.loc[k, '50%']:.2f} [{sl.loc[k, '25%']:.2f}, {sl.loc[k, '75%']:.2f}] "
                  f"({sl.loc[k, 'min']:.2f}, {sl.loc[k, 'max']:.2f})" for k in ('F1-MI', 'F2-MI', 'F1p-MI', 'F1-RT',
                                                                             'F2-RT')),
        'descriptive: impute-then-classify slope erratic; ADMIT, coarsened and retrained forced stable',
        'e11/e11sim_cords2024_nsclc_imc.tsv', 'not a prespecified test; slopes undefined when the model has no '
        'identifiable feature (ADMIT under rand50_s1) are omitted')
    for ref, lab in (('F1p-MI', 'forced labels coarsened to the same supported level'),
                     ('F1-RT', 'forced classifier retrained without the markers')):
        m, lo, hi = d('rand50', 'F2-MI', 'dC', ref)
        add(f'H11a-vs-{ref} ADMIT beats {lab} (random-50%)', f'paired per-seed difference in dC (ADMIT-MI minus {ref})',
            fmt(m, lo, hi), 'supported' if lo > 0 else 'not supported', 'e11/e11sim_cords_paired_differences.tsv')
    m, lo, hi = d('rand50', 'F2-RT', 'dC', 'F1-RT')
    add('H11a-RT ADMIT beats forced with the same retrained scorer (random-50%)',
        'paired per-seed difference in dC (ADMIT-RT minus forced-RT)', fmt(m, lo, hi),
        'supported' if lo > 0 else 'not supported', 'e11/e11sim_cords_paired_differences.tsv')
    s1 = D[(D.metric == 'shift_median') & (D.family == 'rand50')].set_index('arm')
    add('H11b ADMIT feature distributions shift less than forced between full and reduced panel',
        'median standardised mean shift over features (random-50%), mean over seeds',
        f"ADMIT-MI {s1.loc['F2-MI', 'mean']:.2f} vs forced-MI {s1.loc['F1-MI', 'mean']:.2f}; ADMIT-RT "
        f"{s1.loc['F2-RT', 'mean']:.2f} vs forced-RT {s1.loc['F1-RT', 'mean']:.2f}",
        'not supported', 'e11/e11sim_cords_paired_differences.tsv',
        'ADMIT relabels cells it cannot resolve (abstain/unresolved), which moves its features more')
    for fam in ('full', 'single', 'rand50'):
        m, lo, hi = d(fam, 'F3-MI', 'C', 'F2-MI')
        add(f'H11c uncertainty features do not hurt and may help ({fam} panels)',
            'paired per-seed difference in deploy C (F3 minus F2, imputed scorer)', fmt(m, lo, hi),
            'not supported (F3 lower)' if hi < 0 else ('not supported (no gain)' if lo <= 0 else 'supported'),
            'e11/e11sim_cords_paired_differences.tsv')
    for pair, lab in (('quiche_tnbc_mibi_spain__quiche_tnbc_mibi_stanford', 'QUICHE Spain -> Stanford RFS'),
                      ('kulasinghe2026_nsclc_pcf__monkman2026_nsclc_pcf', 'Kulasinghe -> Monkman OS'),
                      ('monkman2026_nsclc_pcf__kulasinghe2026_nsclc_pcf', 'Monkman -> Kulasinghe OS')):
        R = pd.read_csv(R2 / 'e11' / f'e11real_{pair}.tsv', sep='\t')
        R = R[R.model == 'ridge']
        piv = R.pivot_table(index='seed', columns=['arm', 'scorer'], values='C_deploy')
        ests = {k: tci(piv[k]) for k in piv.columns}
        dmi = tci(piv[('F2', 'MI')] - piv[('F1', 'MI')])
        drt = tci(piv[('F2', 'RT')] - piv[('F1', 'RT')])
        cvA = R.groupby('arm').C_cv_A.mean()
        add(f'H11-real ADMIT features transfer better than forced ({lab})',
            'deploy C (mean over 5 seeds, t CI); paired difference ADMIT minus forced',
            f"forced-MI {fmt(*ests[('F1', 'MI')])}, ADMIT-MI {fmt(*ests[('F2', 'MI')])}, forced-RT "
            f"{fmt(*ests[('F1', 'RT')])}, ADMIT-RT {fmt(*ests[('F2', 'RT')])}; diff MI {fmt(*dmi)}, RT {fmt(*drt)}; "
            f"within-source CV C {cvA.min():.2f}-{cvA.max():.2f}",
            ('not supported' if 'quiche' in pair else 'inconclusive (no transferable signal: source CV C <= 0.52, '
             '16 events)'), f'e11/e11real_{pair}.tsv')
    # ---- E13
    E = pd.read_csv(R2 / 'e13' / 'e13_seed0.tsv', sep='\t')
    E = E[(E.zscore == 'pooled') & (E.scorer == 'MI')]
    for pool in ('NSCLC_OS', 'TNBC_recurrence', 'TNBC_OS'):
        g = E[(E.pool == pool) & E.celltype.isin(['cyto_T', 'pooled_T'])].set_index(['celltype', 'version'])
        txt = '; '.join(f"{ct} SE admit/forced/author {g.loc[(ct, 'admit'), 'se_strat']:.3f}/"
                        f"{g.loc[(ct, 'forced'), 'se_strat']:.3f}/{g.loc[(ct, 'naive'), 'se_strat']:.3f}, I2 "
                        f"{g.loc[(ct, 'admit'), 'I2']:.2f}/{g.loc[(ct, 'forced'), 'I2']:.2f}/"
                        f"{g.loc[(ct, 'naive'), 'I2']:.2f}" for ct in ('cyto_T', 'pooled_T')
                        if (ct, 'naive') in g.index)
        add(f'H13 ADMIT harmonisation improves pooled precision / lowers heterogeneity ({pool})',
            'cohort-stratified Cox SE and per-cohort I2, ADMIT (imputed scorer) vs forced vs author labels', txt,
            'not supported', 'e13/e13_seed0.tsv')
    # ---- E12
    for c in ('cords2024_nsclc_imc', 'danenberg2022_metabric_imc'):
        S = pd.read_csv(R2 / 'e12' / f'e12_{c}_summary.tsv', sep='\t')
        f = S[S.panel == 'full']
        dl = S[S.panel == 'del_CD8']
        sens = pd.read_csv(R2 / 'e12' / f'e12_{c}_sensitivity_seed0.tsv', sep='\t')
        st = sens.stable_L_U.dropna()
        add(f'H12 per-patient proportion bounds cover the author-label proportion and are informative ({c})',
            'coverage P(L <= truth <= U) and mean width (pp); stability of HR sign and p<0.05 between L and U',
            f"full panel coverage {f.coverage.min():.2f}-{f.coverage.max():.2f}, width {f.width_pp.min():.1f}-"
            f"{f.width_pp.max():.1f} pp; -CD8 coverage {dl.coverage.min():.2f}-{dl.coverage.max():.2f}, width "
            f"{dl.width_pp.min():.1f}-{dl.width_pp.max():.1f} pp; association stable across bounds in "
            f"{int(st.astype(str).eq('True').sum())}/{len(st)} estimable (type x panel x scorer)",
            'coverage high but bounds too wide to be informative; associations not stable across bounds',
            f'e12/e12_{c}_summary.tsv')
    # ---- E10
    for f in sorted((R2 / 'e10').glob('e10_*_*.tsv')):
        if f.stem.endswith('summary'):
            continue
        R = pd.read_csv(f, sep='\t')
        R = R[R.model == 'ridge']
        piv = R.pivot_table(index='seed', columns='arm', values='C')
        m, lo, hi = tci(piv['F3'] - piv['F2'])
        kind = 'AUC' if R.kind.iloc[0] == 'binary' else 'C'
        add(f"E10 uncertainty features improve within-cohort prediction ({R.cohort.iloc[0]}, {R.endpoint.iloc[0]})",
            f'5-fold CV {kind}, F3 minus F2, mean over 5 typing seeds (t CI)', fmt(m, lo, hi),
            'supported' if lo > 0 else 'not supported', str(f.relative_to(R2)),
            f"F2 {piv['F2'].mean():.3f}, F3 {piv['F3'].mean():.3f}, forced {piv['F1'].mean():.3f}, "
            f"author {piv['oracle'].mean():.3f}; n={int(R.n.iloc[0])}, events={int(R.events.iloc[0])}")
    # ---- E14
    import glob
    Q = pd.concat([pd.read_csv(f, sep='\t') for f in glob.glob(str(R2 / 'e14' / 'e14_*_auroc.tsv'))])
    Q = Q[Q.marker.isna()]
    for (c, mode), g in Q.groupby(['cohort', 'mode']):
        a = tci(g[g.score == 'admit_nonsingleton'].auroc)
        b = tci(g[g.score == 'forced_1mmaxprob'].auroc)
        add(f'E14 ADMIT set statistics flag failed samples better than forced confidence ({c}, {mode})',
            'AUROC per seed (3 seeds), primary scores: ADMIT non-singleton rate vs forced mean(1 - max prob)',
            f'ADMIT {fmt(*a, d=2)}; forced {fmt(*b, d=2)}', 'not supported (both near 0.5)',
            'e14/e14_*_auroc.tsv')
    N = pd.DataFrame(neg)
    N.to_csv(R2 / 'NEGATIVE_RESULTS.tsv', sep='\t', index=False)
    print(N[['hypothesis', 'estimate_ci', 'verdict']].to_string(index=False))
    # manifest
    rows = []
    for p in sorted(R2.rglob('*')):
        if p.is_file() and 'typing' not in p.parts and 'logs' not in p.parts:
            rows.append({'file': str(p.relative_to(R2)), 'bytes': p.stat().st_size})
    rows.append({'file': 'typing/ (per-seed per-patient typing counts, no IDs)',
                 'bytes': sum(q.stat().st_size for q in (R2 / 'typing').glob('*'))})
    pd.DataFrame(rows).to_csv(R2 / 'MANIFEST.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
