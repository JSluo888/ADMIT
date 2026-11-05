#!/usr/bin/env python3
"""Compact cross-cohort summary (printed + $ADMIT_RUNS/downstream/summary_compact.tsv). No patient identifiers."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds_common import COHORTS, OUT  # noqa: E402


def f(r):
    if r is None or pd.isna(r.get('hr')):
        return 'NE'
    return f"{r['hr']:.2f} ({r['ci_lo']:.2f}-{r['ci_hi']:.2f})"


rows = []
for name, cfg in COHORTS.items():
    d = OUT / name
    if not (d / 'partB_survival.tsv').exists():
        continue
    prim = [e for e, v in cfg['endpoints'].items() if v['primary']][0]
    A = pd.read_csv(d / 'partA.tsv', sep='\t')
    a = A[(A.endpoint == prim) & (A.denominator == 'mapped') & (A.model == 'adjusted')].set_index('celltype')
    B = pd.read_csv(d / 'partB_survival.tsv', sep='\t')
    b = B[(B.endpoint == prim) & (B.direction == 'crossfit') & (B.seed == B.seed.min())]
    E = pd.read_csv(d / 'partB_errors.tsv', sep='\t')
    pb = json.loads((d / 'partB.json').read_text())
    boot = pb['bootstrap_seed0_crossfit']['comparisons']
    d1 = B[(B.endpoint == prim) & (B.direction.astype(str) == '1')]

    def g(arm, ct):
        x = b[(b.arm == arm) & (b.celltype == ct)]
        return x.iloc[0].to_dict() if len(x) else None

    def err(arm, ct, col):
        x = E[(E.arm == arm) & (E.celltype == ct)]
        return float(x[col].median()) if len(x) else np.nan

    def sh(arm, ct):
        x = d1[(d1.arm == arm) & (d1.celltype == ct)].hr
        return f'{x.min():.2f}-{x.max():.2f}' if x.notna().any() else 'NE'

    def bt(k):
        v = boot.get(k, {})
        return (f"{v['ratio_of_hr']:.2f} ({v['ratio_ci'][0]:.2f}-{v['ratio_ci'][1]:.2f})" if 'ratio_ci' in v
                else 'NE')
    mi = [s['panels'].get('MI:noCD8', {}).get('forced_frac_cells_cytotoxic_T') for s in pb['splits']]
    gtf = [s['panels'].get('MI:noCD8', {}).get('gt_frac_cells_cytotoxic_T') for s in pb['splits']]
    cov = [s['panels'].get('MI:noCD8', {}).get('admit_exact_coverage') for s in pb['splits']]
    inad = [s['panels'].get('MI:noCD8', {}).get('admit_frac_sets_with_cytotoxic_T') for s in pb['splits']]
    rows.append({
        'cohort': name, 'endpoint': prim, 'n': int(a.loc['cyto_T', 'n']), 'events': int(a.loc['cyto_T', 'events']),
        'GT_cytoT_partA': f(a.loc['cyto_T'].to_dict()), 'GT_cytoT_p_bh': a.loc['cyto_T', 'p_bh'],
        'GT_pooledT_partA': f(a.loc['pooled_T'].to_dict()), 'GT_pooledT_p_bh': a.loc['pooled_T', 'p_bh'],
        'GT_treg_partA': f(a.loc['treg'].to_dict()) if 'treg' in a.index else 'n/a',
        'GT_treg_p_bh': a.loc['treg', 'p_bh'] if 'treg' in a.index else np.nan,
        'forcedMI_noCD8_cytoT': f(g('forced_noCD8', 'cyto_T')),
        'forcedMI_noCD8_pct_cells_cytoT': np.nanmedian([v for v in mi if v is not None]) * 100,
        'gt_pct_cells_cytoT': np.nanmedian([v for v in gtf if v is not None]) * 100,
        'forcedRT_noCD8_cytoT': f(g('forcedRT_noCD8', 'cyto_T')),
        'ADMIT_MI_noCD8_pooledT': f(g('admit_noCD8', 'pooled_T')),
        'ADMIT_RT_noCD8_pooledT': f(g('admitRT_noCD8', 'pooled_T')),
        'forcedRT_noFOXP3_treg': f(g('forcedRT_noFOXP3', 'treg')),
        'forcedMI_noFOXP3_treg': f(g('forced_noFOXP3', 'treg')),
        'ratio_forcedMI_vs_GT_cytoT': bt('forced_noCD8__cyto_T minus GT__cyto_T'),
        'ratio_forcedRT_vs_GT_cytoT': bt('forcedRT_noCD8__cyto_T minus GT__cyto_T'),
        'ratio_ADMIT_MI_vs_GT_pooledT': bt('admit_noCD8__pooled_T minus GT__pooled_T'),
        'ratio_ADMIT_RT_vs_GT_pooledT': bt('admitRT_noCD8__pooled_T minus GT__pooled_T'),
        'ratio_GT_pooledT_vs_GT_cytoT': bt('GT__pooled_T minus GT__cyto_T'),
        'ratio_forcedRT_vs_GT_treg': bt('forcedRT_noFOXP3__treg minus GT__treg'),
        'mae_pp_forcedMI_cytoT': err('forced_noCD8', 'cyto_T', 'mae_pp'),
        'mae_pp_forcedRT_cytoT': err('forcedRT_noCD8', 'cyto_T', 'mae_pp'),
        'median_ae_pp_forcedRT_cytoT': err('forcedRT_noCD8', 'cyto_T', 'median_ae_pp'),
        'relerr25_forcedRT_cytoT': err('forcedRT_noCD8', 'cyto_T', 'frac_rel_err_gt25'),
        'spearman_forcedRT_cytoT': err('forcedRT_noCD8', 'cyto_T', 'spearman'),
        'spearman_forcedMI_cytoT': err('forced_noCD8', 'cyto_T', 'spearman'),
        'gt_mean_pct_cytoT': err('forced_noCD8', 'cyto_T', 'gt_mean_pct'),
        'admit_noCD8_exact_cov': np.nanmedian([v for v in cov if v is not None]),
        'admit_noCD8_frac_sets_with_cytoT': np.nanmax([v for v in inad if v is not None]),
        'split_half_GT_cytoT': sh('GT', 'cyto_T'), 'split_half_forcedRT_cytoT': sh('forcedRT_noCD8', 'cyto_T'),
        'split_half_events_median': d1.events.median(),
    })
S = pd.DataFrame(rows)
S.to_csv(OUT / 'summary_compact.tsv', sep='\t', index=False)
with pd.option_context('display.width', 250, 'display.max_columns', 50):
    print(S.T.to_string())
