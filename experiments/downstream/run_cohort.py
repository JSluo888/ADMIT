#!/usr/bin/env python3
"""Fig 4 downstream analyses for one cohort (plan: $ADMIT_RUNS/downstream/<cohort>/analysis_plan.json).

  python run_cohort.py --cohort cords2024_nsclc_imc --part A
  python run_cohort.py --cohort cords2024_nsclc_imc --part B --seeds 0-4

Part A: ground-truth (author-label) proportions vs survival, all eligible patients.
Part B: counterfactual panel without CD8 / FOXP3: forced classifier vs ADMIT, 50/50 patient splits x 5 seeds
        (both directions -> cross-fit), paired bootstrap of delta log HR on the seed-0 cross-fit.
No patient identifiers are printed or written (pidx = row of clinical.csv).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds_common import (COHORTS, CT_ORDER, CELLTYPES, IMMUNE_SUBTREE, OUT, PLAN_GENERIC, T_SUBTREE, bh,  # noqa: E402
                       assoc, available_celltypes, celltype_masks, cohort_dir, cox, cox_logHR_fast, design,
                       endpoint_frame, endpoint_kind, km_split, load_cells, load_clinical, min_events_ok, plan_for,
                       plan_hash, ready_info, sha256, write_json)

MIN_CELLS = 200
MIN_IMMUNE = 20
ALPHA = 0.10
from admit.ontology import NODE17  # noqa: E402,F401  (17-node column order shared by every experiment)


def log(*a):
    print(f'[{datetime.now().strftime("%H:%M:%S")}]', *a, flush=True)


def check_plan(name: str, require_ready: bool) -> dict:
    gp = json.loads((OUT / 'analysis_plan.json').read_text())
    cp = json.loads((OUT / name / 'analysis_plan.json').read_text())
    if gp['plan_hash'] != plan_hash(PLAN_GENERIC) or cp['plan_hash'] != plan_hash(plan_for(name)):
        sys.exit('plan on disk differs from code: write an amendment with write_plan.py --amend first')
    ri = ready_info(name)
    if require_ready and not ri['ready']:
        sys.exit(f'{name}: READY missing; not analysing')
    d = cohort_dir(name)
    prov = {'generic_plan_hash': gp['plan_hash'], 'cohort_plan_hash': cp['plan_hash'],
            'plan_written_at': cp['written_at'], **ri,
            'cells_sha256': sha256(d / 'cells.parquet'), 'clinical_sha256': sha256(d / 'clinical.csv'),
            'run_at': datetime.now().isoformat(timespec='seconds')}
    return prov


def patient_counts(name, clin, cells):
    """Per-pidx counts: mapped, immune, and each GT cell type (mapped cells only)."""
    pid2idx = pd.Series(clin.pidx.values, index=clin.patient_id.astype(str))
    cp = pd.Series(cells['patient_id']).map(pid2idx).to_numpy()
    if np.isnan(cp.astype(float)).any():
        log('cells without clinical row:', int(np.isnan(cp.astype(float)).sum()))
    cp = np.nan_to_num(cp.astype(float), nan=-1).astype(int)
    mapped = (cells['node17'] != 'UNMAPPED') & (cp >= 0)
    n = len(clin)
    out = {'n_mapped': np.bincount(cp[mapped], minlength=n),
           'n_all': np.bincount(cp[cp >= 0], minlength=n),
           'n_immune': np.bincount(cp[mapped & np.isin(cells['node17'], IMMUNE_SUBTREE)], minlength=n)}
    for k, m in celltype_masks(name, cells['node17'], cells['label_fine']).items():
        out[k] = np.bincount(cp[mapped & m], minlength=n)
    return out, cp, mapped


# ---------------------------------------------------------------------------------------------------------------
# Part A
# ---------------------------------------------------------------------------------------------------------------

def part_a(name: str, prov: dict):
    cfg = COHORTS[name]
    clin = load_clinical(name)
    cells = load_cells(name, with_markers=False)
    cnt, _, _ = patient_counts(name, clin, cells)
    del cells
    adjust = list(cfg['covariates'])
    CTS = available_celltypes(name)
    res = {'cohort': name, 'provenance': prov, 'adjust': adjust, 'celltypes_analysed': CTS, 'endpoints': {}}
    rows = []
    for ep, epd in cfg['endpoints'].items():
        kind = endpoint_kind(name, ep)
        df = endpoint_frame(name, clin, ep)
        n0 = len(df)
        df['n_mapped'] = cnt['n_mapped'][df.pidx]
        df['n_immune'] = cnt['n_immune'][df.pidx]
        df = df[df.n_mapped >= MIN_CELLS].reset_index(drop=True)
        E = {'label': epd['label'], 'primary': epd['primary'], 'kind': kind, 'n_endpoint_eligible': n0,
             'n_after_cell_rule': len(df), 'events': int(df.E.sum()), 'models': {}}
        if not min_events_ok(df.E, kind):
            E['not_estimable'] = 'fewer than 10 events (or 10 in the smaller class)'
        for ct in CTS:
            df[f'{ct}__mapped'] = cnt[ct][df.pidx] / df.n_mapped
            df[f'{ct}__immune'] = np.where(df.n_immune >= MIN_IMMUNE, cnt[ct][df.pidx] / df.n_immune.clip(1), np.nan)
        E['proportion_summary'] = {ct: {'mean': float(df[f'{ct}__mapped'].mean()),
                                        'median': float(df[f'{ct}__mapped'].median()),
                                        'frac_zero': float((df[f'{ct}__mapped'] == 0).mean())} for ct in CTS}
        for den in ('mapped', 'immune'):
            for model, adj in (('adjusted', adjust), ('unadjusted', [])):
                block = {}
                for ct in CTS:
                    r = assoc(df, f'{ct}__{den}', adj, kind)
                    if den == 'mapped' and model == 'adjusted' and kind == 'surv' and 'error' not in r:
                        r['km'] = km_split(df, f'{ct}__mapped', 'T', 'E',
                                           keep_curve=ct in ('cyto_T', 'treg', 'pooled_T'))
                    block[ct] = r
                padj = bh([block[ct].get('p') for ct in CTS])
                for ct, pa in zip(CTS, padj):
                    block[ct]['p_bh'] = pa
                fam = 'primary' if (epd['primary'] and den == 'mapped' and model == 'adjusted') else 'secondary'
                E['models'][f'{den}__{model}'] = {'bh_family': fam, 'by_celltype': block}
                for ct in CTS:
                    r = block[ct]
                    rows.append({'cohort': name, 'endpoint': ep, 'kind': kind, 'primary_endpoint': epd['primary'],
                                 'denominator': den, 'model': model, 'bh_family': fam, 'celltype': ct,
                                 'celltype_label': plan_for(name)['celltypes'][ct]['label'],
                                 'n': r.get('n'), 'events': r.get('events'), 'hr': r.get('hr'),
                                 'ci_lo': r.get('ci_lo'), 'ci_hi': r.get('ci_hi'), 'p': r.get('p'),
                                 'p_bh': r.get('p_bh'), 'c_model': r.get('c_model'),
                                 'c_celltype_only': r.get('c_celltype_only'), 'ph_test_p': r.get('ph_test_p'),
                                 'km_logrank_p': r.get('km', {}).get('logrank_p'),
                                 'km_n_high': r.get('km', {}).get('n_high'), 'km_n_low': r.get('km', {}).get('n_low'),
                                 'error': r.get('error')})
        res['endpoints'][ep] = E
        log(name, ep, 'n', len(df), 'events', int(df.E.sum()))
    od = OUT / name
    write_json(res, od / 'partA.json')
    pd.DataFrame(rows).to_csv(od / 'partA.tsv', sep='\t', index=False)
    t = pd.DataFrame(rows)
    show = t[(t.denominator == 'mapped') & (t.model == 'adjusted')][['endpoint', 'celltype', 'n', 'events', 'hr',
                                                                     'ci_lo', 'ci_hi', 'p', 'p_bh', 'c_model',
                                                                     'km_logrank_p']]
    print(show.round(4).to_string(index=False))


# ---------------------------------------------------------------------------------------------------------------
# Part B
# ---------------------------------------------------------------------------------------------------------------

def consensus_node(sets: np.ndarray, onto) -> np.ndarray:
    """Finest consensus node per row (prespecified). -1 = unresolved (no common node), -2 = empty set."""
    anc = onto.anc                         # anc[c, d]: c ancestor-or-self of d
    comp = anc | anc.T
    K = onto.K
    best = np.full(len(sets), -1, dtype=np.int64)
    bestd = np.full(len(sets), -1, dtype=np.int64)
    for c in range(K):
        ok = ~(sets & ~comp[c][None, :]).any(axis=1)
        ok &= (sets & anc[c][None, :]).any(axis=1)
        upd = ok & (onto.depth[c] > bestd)
        best[upd] = c
        bestd[upd] = onto.depth[c]
    best[sets.sum(axis=1) == 0] = -2
    return best


def fit_xgb(X, y, seed):
    import xgboost as xgb
    present = np.unique(y)
    remap = np.full(max(present) + 1, -1)
    remap[present] = np.arange(len(present))
    m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, objective='multi:softprob',
                          tree_method='hist', device='cuda', random_state=seed, n_jobs=8)
    m.fit(X, remap[y], verbose=False)
    return m, present


def predict17(m, present, Z, K, batch=1_000_000):
    P = np.zeros((len(Z), K), dtype=np.float32)
    for i in range(0, len(Z), batch):
        P[i:i + batch, present] = m.predict_proba(Z[i:i + batch])
    P /= P.sum(1, keepdims=True)
    return P


def sample_cells(rng, cell_p, idx_pool_by_p, patients, cap):
    out = []
    for p in patients:
        ii = idx_pool_by_p.get(p)
        if ii is None or len(ii) == 0:
            continue
        out.append(ii if len(ii) <= cap else rng.choice(ii, cap, replace=False))
    return np.concatenate(out)


def part_b(name: str, prov: dict, seeds: list, n_boot: int = 1000):
    from admit.panel_matched import AdmitM1, Ontology
    cfg = COHORTS[name]
    t0 = time.time()
    clin = load_clinical(name)
    cells = load_cells(name, with_markers=True)
    cnt, cp, mapped = patient_counts(name, clin, cells)
    onto = Ontology.from_repo(NODE17)
    idx17 = {t: i for i, t in enumerate(NODE17)}
    keep = np.flatnonzero(mapped)
    y = np.array([idx17[v] for v in cells['node17'][keep]], dtype=np.int64)
    cpk = cp[keep]
    if cfg.get('transform', 'arcsinh') == 'arcsinh_p95':
        Xr = cells['X'][keep]
        co = np.maximum(np.percentile(Xr[np.random.default_rng(0).choice(len(Xr), min(len(Xr), 500000),
                                                                           replace=False)], 95, axis=0), 1e-6)
        X = np.arcsinh(Xr / co[None, :]).astype(np.float32)
        del Xr
    else:
        co = cfg['transform_cofactor']
        X = np.arcsinh(cells['X'][keep] / co).astype(np.float32)
    vocab = cells['marker_vocab']
    gt_masks = {k: m[keep] for k, m in celltype_masks(name, cells['node17'], cells['label_fine']).items()}
    del cells
    log(name, 'cells kept', len(keep), 'features', X.shape[1], f'{time.time()-t0:.0f}s')
    full_panel = sorted({v for v in vocab if v != 'NA'})
    panels = {'full': full_panel}
    for cf in cfg['counterfactuals']:
        mk = cf.replace('no', '')
        assert mk in full_panel, (mk, full_panel)
        panels[cf] = [v for v in full_panel if v != mk]
    mask_cols = {pn: [j for j, v in enumerate(vocab) if v != 'NA' and v not in set(pl)] for pn, pl in panels.items()}
    log('panels', {k: [vocab[j] for j in v] for k, v in mask_cols.items()})

    # eligibility (primary endpoint) and per-endpoint frames
    eps = cfg['endpoints']
    prim = [e for e, d in eps.items() if d['primary']][0]
    frames = {}
    for ep in eps:
        f = endpoint_frame(name, clin, ep)
        f = f[cnt['n_mapped'][f.pidx] >= MIN_CELLS].reset_index(drop=True)
        frames[ep] = f
    elig = np.sort(frames[prim].pidx.to_numpy())
    adjust = list(cfg['covariates'])
    kinds = {ep: endpoint_kind(name, ep) for ep in eps}
    by_p = pd.Series(np.arange(len(cpk))).groupby(cpk).apply(lambda s: s.to_numpy()).to_dict()
    n_pat = len(clin)

    # node sets per cell type for forced/ADMIT labels (node17 definitions; overridden types -> undefined)
    ct_nodes = {}
    for ct in [c for c in ('cyto_T', 'treg', 'helper_T', 'pooled_T') if c in available_celltypes(name)]:
        d = plan_for(name)['celltypes'][ct]
        ct_nodes[ct] = [idx17[n] for n in d['node17']] if 'node17' in d else None
    # (arm, panel, kind, scorer): scorer 'MI' = full-panel classifier with mean imputation (primary);
    # 'RT' = classifier retrained on the reduced panel with the full label set (amendment 1, secondary)
    arms_def = [('forced_full', 'full', 'forced', 'MI'), ('admit_full', 'full', 'admit', 'MI')]
    for cf in cfg['counterfactuals']:
        arms_def += [(f'forced_{cf}', cf, 'forced', 'MI'), (f'admit_{cf}', cf, 'admit', 'MI'),
                     (f'forcedRT_{cf}', cf, 'forced', 'RT'), (f'admitRT_{cf}', cf, 'admit', 'RT')]

    results = {'cohort': name, 'provenance': prov, 'seeds': seeds, 'n_eligible': int(len(elig)),
               'primary_endpoint': prim, 'adjust': adjust, 'panels': panels, 'splits': []}
    rows = []
    crossfit = {}
    fig_scatter = None
    for seed in seeds:
        rng = np.random.default_rng(1000 + seed)
        perm = rng.permutation(elig)
        h = len(perm) // 2
        halves = {'A': perm[:h], 'B': perm[h:]}
        # per-pidx counts for every arm (cross-fit fills both halves)
        CF = {a[0]: {ct: np.full(n_pat, np.nan) for ct in ct_nodes} for a in arms_def}
        for direction, (dev_k, ev_k) in {'1': ('A', 'B'), '2': ('B', 'A')}.items():
            ts = time.time()
            dev, ev = halves[dev_k], halves[ev_k]
            dperm = rng.permutation(dev)
            nf = int(round(0.6 * len(dperm)))
            fit_p, cal_p = dperm[:nf], dperm[nf:]
            tr = sample_cells(rng, cpk, by_p, fit_p, 2000)
            ca = sample_cells(rng, cpk, by_p, cal_p, 2000)
            evi = np.concatenate([by_p[p] for p in ev if p in by_p])
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd == 0] = 1.0
            Ztr = (X[tr] - mu) / sd
            models = {('MI', 'full'): fit_xgb(Ztr, y[tr], seed)}
            for cf in cfg['counterfactuals']:
                zt = Ztr.copy()
                zt[:, mask_cols[cf]] = 0.0
                models[('RT', cf)] = fit_xgb(zt, y[tr], seed)
            del Ztr
            Zc = (X[ca] - mu) / sd
            Ze = (X[evi] - mu) / sd
            split = {'seed': seed, 'direction': direction, 'n_fit_patients': len(fit_p), 'n_cal_patients': len(cal_p),
                     'n_eval_patients': len(ev), 'n_fit_cells': len(tr), 'n_cal_cells': len(ca),
                     'n_eval_cells': len(evi), 'classes': [NODE17[c] for c in models[('MI', 'full')][1]],
                     'panels': {}}
            ye = y[evi]
            pe = cpk[evi]
            scorer_panels = [('MI', pn) for pn in panels] + [('RT', cf) for cf in cfg['counterfactuals']]
            for scorer, pn in scorer_panels:
                model, present = models[(scorer, 'full' if scorer == 'MI' else pn)]
                zc, ze = Zc.copy(), Ze.copy()
                zc[:, mask_cols[pn]] = 0.0
                ze[:, mask_cols[pn]] = 0.0
                Pc = predict17(model, present, zc, onto.K)
                Pe = predict17(model, present, ze, onto.K)
                forced = Pe.argmax(1)
                adm = AdmitM1(onto, ALPHA).fit(onto.aggregate(Pc), y[ca], panels[pn])
                sets = adm.predict(onto.aggregate(Pe))
                cons = consensus_node(sets, onto)
                admv = onto.admissible(panels[pn])
                tgt = adm.target(ye)
                ident = tgt >= 0
                hit = np.zeros(len(ye), bool)
                hit[ident] = sets[np.flatnonzero(ident), tgt[ident]]
                ctl = ye == idx17['Cytotoxic_T_Cell']
                cons_ctl = pd.Series(np.where(cons[ctl] >= 0, np.array(NODE17 + [''])[np.clip(cons[ctl], 0, None)],
                                              np.where(cons[ctl] == -2, 'abstain', 'unresolved'))).value_counts(
                    normalize=True).round(4).to_dict()
                pinfo = {'accuracy_forced_vs_node17': float((forced == ye).mean()),
                         'forced_frac_cells_cytotoxic_T': float((forced == idx17['Cytotoxic_T_Cell']).mean()),
                         'gt_frac_cells_cytotoxic_T': float(ctl.mean()),
                         'forced_frac_cells_treg': float((forced == idx17['Regulatory_T_Cell']).mean()),
                         'gt_frac_cells_treg_node17': float((ye == idx17['Regulatory_T_Cell']).mean()),
                         'recall_forced': {NODE17[c]: float((forced[ye == c] == c).mean())
                                           for c in np.unique(ye) if (ye == c).sum() > 0},
                         'admit_exact_coverage': float(hit[ident].mean()), 'admit_mean_set_size': float(sets.sum(1).mean()),
                         'admit_frac_sets_with_inadmissible': float((sets & ~admv[None, :]).any(1).mean()),
                         'admit_frac_sets_with_cytotoxic_T': float(sets[:, idx17['Cytotoxic_T_Cell']].mean()),
                         'admit_coverage_gt_ctl_cells': float(hit[ctl & ident].mean()) if (ctl & ident).any() else None,
                         'admit_consensus_of_gt_ctl_cells': cons_ctl,
                         'forced_label_of_gt_ctl_cells': pd.Series(np.array(NODE17)[forced[ctl]]).value_counts(
                             normalize=True).round(4).head(6).to_dict(),
                         'admit_groups': adm.strata_report(),
                         'admissible_nodes': [NODE17[c] for c in np.flatnonzero(admv)]}
                split['panels'][f'{scorer}:{pn}'] = pinfo
                for arm, apn, kind, sc_ in arms_def:
                    if apn != pn or sc_ != scorer:
                        continue
                    lab = forced if kind == 'forced' else cons
                    for ct, nodes in ct_nodes.items():
                        if nodes is None:
                            continue
                        c_ = np.bincount(pe[np.isin(lab, nodes)], minlength=n_pat).astype(float)
                        CF[arm][ct][ev] = c_[ev]
                del Pc, Pe, sets, cons, forced
            # GT arm
            gt_ct = {ct: np.bincount(pe[gt_masks[ct][evi]], minlength=n_pat) for ct in ct_nodes}
            results['splits'].append(split)
            log(name, f'seed {seed} dir {direction}', f'{time.time()-ts:.0f}s',
                'acc full', round(split['panels']['MI:full']['accuracy_forced_vs_node17'], 3),
                {k: (round(v['forced_frac_cells_cytotoxic_T'], 4), round(v['gt_frac_cells_cytotoxic_T'], 4))
                 for k, v in split['panels'].items()})
            # survival on the evaluation half
            rows += survival_rows(name, seed, direction, ev, frames, cnt, CF, gt_ct, adjust, arms_def, ct_nodes)
        # cross-fit: all eligible patients
        gt_all = {ct: cnt[ct].astype(float) for ct in ct_nodes}
        rows += survival_rows(name, seed, 'crossfit', elig, frames, cnt, CF, gt_all, adjust, arms_def, ct_nodes)
        crossfit[seed] = CF
        if seed == seeds[0]:
            fig_scatter = {'gt': gt_all, 'cf': CF}
    res_tab = pd.DataFrame(rows)
    od = OUT / name
    res_tab.to_csv(od / 'partB_survival.tsv', sep='\t', index=False)

    # error metrics (cross-fit, per seed) and bootstrap (seed 0 cross-fit)
    err = []
    for seed, CF in crossfit.items():
        nm = cnt['n_mapped'][elig].astype(float)
        for arm, *_ in arms_def:
            for ct in ct_nodes:
                if ct_nodes[ct] is None:
                    continue
                g = cnt[ct][elig] / nm
                f = CF[arm][ct][elig] / nm
                e = (f - g) * 100
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel = np.abs(f - g) / np.where(g > 0, g, np.nan)
                from scipy.stats import spearmanr
                err.append({'seed': seed, 'arm': arm, 'celltype': ct, 'n_patients': int(len(g)),
                            'mae_pp': float(np.mean(np.abs(e))), 'median_ae_pp': float(np.median(np.abs(e))),
                            'p90_ae_pp': float(np.percentile(np.abs(e), 90)), 'mean_signed_pp': float(np.mean(e)),
                            'gt_mean_pct': float(np.mean(g) * 100), 'pred_mean_pct': float(np.mean(f) * 100),
                            'spearman': float(spearmanr(f, g).statistic),
                            'frac_rel_err_gt25': float(np.nanmean(rel > 0.25))})
    pd.DataFrame(err).to_csv(od / 'partB_errors.tsv', sep='\t', index=False)
    boot = bootstrap(name, crossfit[seeds[0]], elig, frames[prim], cnt, adjust, seeds[0], n_boot, ct_nodes,
                     kinds[prim])
    results['bootstrap_seed0_crossfit'] = boot
    results['summary'] = summarize(res_tab, prim)
    write_json(results, od / 'partB.json')
    # figure-ready per-patient scatter (no identifiers): seed-0 cross-fit
    nm = cnt['n_mapped'][elig].astype(float)
    sc = {'n_mapped': nm}
    for ct in ct_nodes:
        sc[f'gt_{ct}'] = fig_scatter['gt'][ct][elig] / nm
        for arm, *_ in arms_def:
            if ct_nodes[ct] is not None:
                sc[f'{arm}_{ct}'] = fig_scatter['cf'][arm][ct][elig] / nm
    pd.DataFrame(sc).sample(frac=1.0, random_state=0).to_csv(od / 'fig_scatter_seed0_crossfit.tsv', sep='\t',
                                                             index=False)
    log(name, 'part B done', f'{time.time()-t0:.0f}s')
    print(results['summary'])


def survival_rows(name, seed, direction, pats, frames, cnt, CF, gt_ct, adjust, arms_def, ct_nodes):
    rows = []
    for ep, f in frames.items():
        df = f[f.pidx.isin(pats)].reset_index(drop=True)
        nm = cnt['n_mapped'][df.pidx].astype(float)
        cols = {}
        for ct in ct_nodes:
            cols[f'GT__{ct}'] = gt_ct[ct][df.pidx] / nm
            for arm, *_ in arms_def:
                if ct_nodes[ct] is not None:
                    cols[f'{arm}__{ct}'] = CF[arm][ct][df.pidx] / nm
        df = pd.concat([df, pd.DataFrame(cols)], axis=1)
        for col in cols:
            arm, ct = col.split('__')
            if df[col].isna().all():
                continue
            r = assoc(df, col, adjust, endpoint_kind(name, ep), ph_test=False)
            gsd = df[f'GT__{ct}'].std(ddof=1)
            rows.append({'cohort': name, 'seed': seed, 'direction': direction, 'endpoint': ep,
                         'kind': endpoint_kind(name, ep), 'arm': arm,
                         'celltype': ct, 'n': r.get('n'), 'events': r.get('events'), 'hr': r.get('hr'),
                         'log_hr': r.get('log_hr'), 'se': r.get('se'), 'ci_lo': r.get('ci_lo'),
                         'ci_hi': r.get('ci_hi'), 'p': r.get('p'), 'c_model': r.get('c_model'),
                         'c_celltype_only': r.get('c_celltype_only'),
                         'hr_per_gt_sd': (float(np.exp(r['log_hr'] * gsd / r['sd_of_covariate']))
                                          if r.get('log_hr') is not None and r.get('sd_of_covariate') else None),
                         'error': r.get('error')})
    return rows


def _boot_chunk(df, pairs, adjust, idxs, kind='surv'):
    out = []
    for idx in idxs:
        d = df.iloc[idx].reset_index(drop=True)
        rr = []
        for a, b in pairs:
            Da, _, _ = design(d, a, adjust, 'T', 'E')
            Db, _, _ = design(d, b, adjust, 'T', 'E')
            Da = Da.loc[:, (Da.std(ddof=0) > 0) | Da.columns.isin(['T', 'E'])]
            Db = Db.loc[:, (Db.std(ddof=0) > 0) | Db.columns.isin(['T', 'E'])]
            rr.append(cox_logHR_fast(Da, kind) - cox_logHR_fast(Db, kind))
        out.append(rr)
    return out


def bootstrap(name, CF, elig, frame, cnt, adjust, seed, B, ct_nodes, kind='surv'):
    from joblib import Parallel, delayed
    df = frame[frame.pidx.isin(elig)].reset_index(drop=True)
    nm = cnt['n_mapped'][df.pidx].astype(float)
    for ct in ct_nodes:
        df[f'GT__{ct}'] = cnt[ct][df.pidx] / nm
        for arm in CF:
            if ct_nodes[ct] is not None:
                df[f'{arm}__{ct}'] = CF[arm][ct][df.pidx] / nm
    pairs = [('forced_full__cyto_T', 'GT__cyto_T')]
    if 'forced_noCD8__cyto_T' in df:
        pairs += [('forced_noCD8__cyto_T', 'GT__cyto_T'), ('admit_noCD8__pooled_T', 'GT__pooled_T'),
                  ('forced_noCD8__pooled_T', 'GT__pooled_T')]
    if 'forced_noFOXP3__treg' in df:
        pairs += [('forced_noFOXP3__treg', 'GT__treg')]
    if 'forcedRT_noCD8__cyto_T' in df:
        pairs += [('forcedRT_noCD8__cyto_T', 'GT__cyto_T'), ('admitRT_noCD8__pooled_T', 'GT__pooled_T')]
    if 'forcedRT_noFOXP3__treg' in df:
        pairs += [('forcedRT_noFOXP3__treg', 'GT__treg')]
    pairs += [('GT__pooled_T', 'GT__cyto_T')]
    if 'admit_noCD8__pooled_T' in df:
        pairs += [('admit_noCD8__pooled_T', 'GT__cyto_T')]
    rng = np.random.default_rng(5000 + seed)
    idxs = [rng.integers(0, len(df), len(df)) for _ in range(B)]
    chunks = [idxs[i::16] for i in range(16)]
    res = Parallel(n_jobs=16)(delayed(_boot_chunk)(df, pairs, adjust, c, kind) for c in chunks)
    arr = np.array([r for ch in res for r in ch], dtype=float)
    out = {'B': B, 'n_patients': len(df), 'resample_unit': 'patient', 'comparisons': {}}
    for j, (a, b) in enumerate(pairs):
        v = arr[:, j]
        v = v[np.isfinite(v)]
        Da, _, _ = design(df, a, adjust, 'T', 'E')
        Db, _, _ = design(df, b, adjust, 'T', 'E')
        Da = Da.loc[:, (Da.std(ddof=0) > 0) | Da.columns.isin(['T', 'E'])]
        Db = Db.loc[:, (Db.std(ddof=0) > 0) | Db.columns.isin(['T', 'E'])]
        point = cox_logHR_fast(Da, kind) - cox_logHR_fast(Db, kind)
        if len(v) < 0.9 * B:
            out['comparisons'][f'{a} minus {b}'] = {'error': f'only {len(v)} of {B} bootstrap fits valid '
                                                             '(covariate constant, e.g. a label never predicted)'}
            continue
        out['comparisons'][f'{a} minus {b}'] = {
            'delta_log_hr': point, 'ratio_of_hr': float(np.exp(point)),
            'ci_lo': float(np.percentile(v, 2.5)), 'ci_hi': float(np.percentile(v, 97.5)),
            'ratio_ci': [float(np.exp(np.percentile(v, 2.5))), float(np.exp(np.percentile(v, 97.5)))],
            'n_valid': int(len(v))}
    return out


def summarize(t: pd.DataFrame, prim: str) -> dict:
    s = {}
    for (direction, arm, ct), g in t[t.endpoint == prim].groupby(['direction', 'arm', 'celltype']):
        hr = g.hr.dropna()
        if len(hr) == 0:
            continue
        s[f'{direction}|{arm}|{ct}'] = {'hr_median': float(hr.median()), 'hr_min': float(hr.min()),
                                       'hr_max': float(hr.max()), 'n_p_lt_05': int((g.p < 0.05).sum()),
                                       'n_splits': int(len(hr))}
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', required=True)
    ap.add_argument('--part', required=True, choices=['A', 'B'])
    ap.add_argument('--seeds', default='0-4')
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--allow-not-ready', action='store_true', help='development only; results marked provisional')
    a = ap.parse_args()
    prov = check_plan(a.cohort, require_ready=not a.allow_not_ready)
    if not prov['ready']:
        prov['PROVISIONAL'] = 'READY missing at run time'
    s = a.seeds
    seeds = list(range(int(s.split('-')[0]), int(s.split('-')[1]) + 1)) if '-' in s else [int(v) for v in s.split(',')]
    (OUT / a.cohort).mkdir(parents=True, exist_ok=True)
    if a.part == 'A':
        part_a(a.cohort, prov)
    else:
        part_b(a.cohort, prov, seeds, a.boot)


if __name__ == '__main__':
    main()
