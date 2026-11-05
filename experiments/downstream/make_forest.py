#!/usr/bin/env python3
"""Assemble the Fig 4 forest table (Part C) and MANIFEST.tsv from the per-cohort outputs.

Writes  $ADMIT_RUNS/downstream/<cohort>/forest.tsv, $ADMIT_RUNS/downstream/forest_all.tsv,
        $ADMIT_RUNS/downstream/partB_errors_all.tsv, $ADMIT_RUNS/downstream/MANIFEST.tsv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds_common import COHORTS, OUT  # noqa: E402

FOREST_CT = ['cyto_T', 'treg', 'pooled_T']
# (arm in partB_survival, celltype, display role)
B_ROWS = [('GT', 'cyto_T', 'GT (same patients)'), ('forced_noCD8', 'cyto_T', 'forced, CD8 masked (mean-imputed)'),
          ('forcedRT_noCD8', 'cyto_T', 'forced, retrained without CD8'),
          ('forced_full', 'cyto_T', 'forced, full panel'),
          ('GT', 'treg', 'GT (same patients)'), ('forced_noFOXP3', 'treg', 'forced, FOXP3 masked (mean-imputed)'),
          ('forcedRT_noFOXP3', 'treg', 'forced, retrained without FOXP3'),
          ('GT', 'pooled_T', 'GT (same patients)'), ('admit_noCD8', 'pooled_T', 'ADMIT T level, CD8 masked'),
          ('admitRT_noCD8', 'pooled_T', 'ADMIT T level, retrained without CD8'),
          ('forced_noCD8', 'pooled_T', 'forced pooled T, CD8 masked'),
          ('admit_noFOXP3', 'pooled_T', 'ADMIT T level, FOXP3 masked'),
          ('admitRT_noFOXP3', 'pooled_T', 'ADMIT T level, retrained without FOXP3'),
          ('forcedRT_noCD8', 'pooled_T', 'forced pooled T, retrained without CD8')]


def main():
    allrows, errs, manifest = [], [], []
    for name, cfg in COHORTS.items():
        d = OUT / name
        if not (d / 'partA.tsv').exists():
            continue
        prim = [e for e, v in cfg['endpoints'].items() if v['primary']][0]
        A = pd.read_csv(d / 'partA.tsv', sep='\t')
        pa = json.loads((d / 'partA.json').read_text())
        prov = pa['provenance']
        rows = []
        a = A[(A.endpoint == prim) & (A.denominator == 'mapped') & (A.model == 'adjusted')]
        for _, r in a.iterrows():
            rows.append({'cohort': name, 'display': cfg['display'], 'endpoint': prim,
                         'effect': 'OR' if r.get('kind') == 'binary' else 'HR', 'celltype': r.celltype,
                         'celltype_label': r.celltype_label, 'source': 'partA_all_patients', 'arm': 'GT',
                         'role': 'GT, all eligible patients', 'hr': r.hr, 'ci_lo': r.ci_lo, 'ci_hi': r.ci_hi,
                         'p': r.p, 'p_bh': r.p_bh, 'n': r.n, 'events': r.events, 'c_model': r.c_model,
                         'note': r.error if isinstance(r.error, str) else ''})
        if (d / 'partB_survival.tsv').exists():
            B = pd.read_csv(d / 'partB_survival.tsv', sep='\t')
            B = B[B.endpoint == prim]
            seed0 = int(B.seed.min())
            for arm, ct, role in B_ROWS:
                g = B[(B.arm == arm) & (B.celltype == ct)]
                if g.empty:
                    continue
                cf0 = g[(g.direction == 'crossfit') & (g.seed == seed0)]
                d1 = g[g.direction.astype(str) == '1']
                cfa = g[g.direction == 'crossfit']
                r = cf0.iloc[0] if len(cf0) else None
                rows.append({'cohort': name, 'display': cfg['display'], 'endpoint': prim,
                             'effect': 'OR' if (r is not None and r.get('kind') == 'binary') else 'HR',
                             'celltype': ct, 'celltype_label': ct, 'source': f'partB_crossfit_seed{seed0}',
                             'arm': arm, 'role': role,
                             'hr': r.hr if r is not None else np.nan, 'ci_lo': r.ci_lo if r is not None else np.nan,
                             'ci_hi': r.ci_hi if r is not None else np.nan, 'p': r.p if r is not None else np.nan,
                             'p_bh': np.nan, 'n': r.n if r is not None else np.nan,
                             'events': r.events if r is not None else np.nan,
                             'c_model': r.c_model if r is not None else np.nan,
                             'hr_per_gt_sd': r.hr_per_gt_sd if r is not None else np.nan,
                             'crossfit_hr_min_5seeds': cfa.hr.min(), 'crossfit_hr_max_5seeds': cfa.hr.max(),
                             'split_half_hr_median': d1.hr.median(), 'split_half_hr_min': d1.hr.min(),
                             'split_half_hr_max': d1.hr.max(), 'split_half_n_p_lt_05': int((d1.p < 0.05).sum()),
                             'split_half_n': int(d1.hr.notna().sum()),
                             'split_half_events_median': d1.events.median(),
                             'note': (r.error if (r is not None and isinstance(r.error, str)) else
                                      ('not estimable' if r is None or pd.isna(r.hr) else ''))})
            pb = json.loads((d / 'partB.json').read_text()) if (d / 'partB.json').exists() else {}
            boot = pb.get('bootstrap_seed0_crossfit', {}).get('comparisons', {})
            for k, v in boot.items():
                rows.append({'cohort': name, 'display': cfg['display'], 'endpoint': prim, 'effect': 'ratio',
                             'celltype': k, 'source': f'bootstrap_seed{seed0}_crossfit', 'arm': 'delta',
                             'role': 'ratio of HRs (paired patient bootstrap)', 'hr': v.get('ratio_of_hr'),
                             'ci_lo': (v.get('ratio_ci') or [np.nan, np.nan])[0],
                             'ci_hi': (v.get('ratio_ci') or [np.nan, np.nan])[1], 'note': v.get('error', '')})
            if (d / 'partB_errors.tsv').exists():
                e = pd.read_csv(d / 'partB_errors.tsv', sep='\t')
                e.insert(0, 'cohort', name)
                errs.append(e)
        f = pd.DataFrame(rows)
        f.to_csv(d / 'forest.tsv', sep='\t', index=False)
        allrows.append(f)
        for fn, desc in (('analysis_plan.json', 'prespecified cohort plan'), ('partA.json', 'Part A full results'),
                         ('partA.tsv', 'Part A table (all endpoints/denominators/models)'),
                         ('partB.json', 'Part B splits, diagnostics, bootstrap'),
                         ('partB_survival.tsv', 'Part B Cox/logistic per seed x direction x arm x cell type'),
                         ('partB_errors.tsv', 'Part B per-patient proportion error (cross-fit, per seed)'),
                         ('fig_scatter_seed0_crossfit.tsv', 'per-patient proportions, seed-0 cross-fit (no IDs)'),
                         ('forest.tsv', 'Part C forest rows')):
            if (d / fn).exists():
                manifest.append({'cohort': name, 'file': str((d / fn).relative_to(OUT)), 'description': desc,
                                 'ready_at_run': prov.get('ready'), 'cohort_plan_hash': prov.get('cohort_plan_hash'),
                                 'cells_sha256': prov.get('cells_sha256', '')[:16],
                                 'clinical_sha256': prov.get('clinical_sha256', '')[:16],
                                 'run_at': prov.get('run_at')})
    F = pd.concat(allrows, ignore_index=True)
    F.to_csv(OUT / 'forest_all.tsv', sep='\t', index=False)
    if errs:
        pd.concat(errs, ignore_index=True).to_csv(OUT / 'partB_errors_all.tsv', sep='\t', index=False)
    for fn, desc in (('analysis_plan.json', 'generic prespecified plan + amendments'),
                     ('forest_all.tsv', 'Part C forest table, all cohorts'),
                     ('partB_errors_all.tsv', 'Part B error metrics, all cohorts')):
        if (OUT / fn).exists():
            manifest.append({'cohort': 'ALL', 'file': fn, 'description': desc})
    pd.DataFrame(manifest).to_csv(OUT / 'MANIFEST.tsv', sep='\t', index=False)
    print(F[F.celltype.isin(FOREST_CT)][['cohort', 'celltype', 'role', 'hr', 'ci_lo', 'ci_hi', 'p']].round(3)
          .to_string(index=False))


if __name__ == '__main__':
    main()
