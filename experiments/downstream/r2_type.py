#!/usr/bin/env python3
"""Round-2 typing (outcome-blind): per-patient feature counts under forced / ADMIT typing.

  r2_type.py sim   --cohort C --seed S            D1 split, every bank panel (E11 sim, E12, part of E10)
  r2_type.py xfit  --cohort A --seed S [--targets B1,B2 ...]   D2 cross-fit of A (+ X_cross_cohort typing of targets)

Writes $ADMIT_RUNS/downstream/round2/typing/<mode>_<cohort>_s<seed>.npz (+ _meta.json). No outcome data is read
except the eligibility frame of the primary endpoint (who is eligible), exactly as in round 1. No patient IDs
are written (pidx = clinical.csv row).
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
from ds_common import (COHORTS, OUT, celltype_masks, endpoint_frame, load_cells, load_clinical,  # noqa: E402
                       write_json)
from run_cohort import NODE17, consensus_node, fit_xgb, patient_counts, predict17  # noqa: E402
from admit.panel_matched import AdmitM1, Ontology  # noqa: E402

R2 = OUT / 'round2'
TYP = R2 / 'typing'
ALPHA = 0.10
MIN_CELLS = 200
TYPE_CAP = 3000
FITCAL_CAP = 2000
BANK_PANELS = ['full', 'del_CD8', 'del_FOXP3', 'del_CD20', 'del_CD68'] + [f'rand50_s{i}' for i in range(5)]
CT6 = ['cyto_T', 'treg', 'helper_T', 'pooled_T', 'B', 'myeloid_mac']
ONTO = Ontology.from_repo(NODE17)
IDX = {t: i for i, t in enumerate(NODE17)}


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


# ---------------------------------------------------------------------------------------------------- data
def load(name):
    cfg = COHORTS[name]
    clin = load_clinical(name)
    cells = load_cells(name, with_markers=True)
    cnt, cp, mapped = patient_counts(name, clin, cells)
    keep = np.flatnonzero(mapped)
    y = np.array([IDX[v] for v in cells['node17'][keep]], dtype=np.int64)
    Xr = cells['X'][keep]
    if cfg.get('transform', 'arcsinh') == 'arcsinh_p95':
        co = np.maximum(np.percentile(Xr[np.random.default_rng(0).choice(len(Xr), min(len(Xr), 500000),
                                                                       replace=False)], 95, axis=0), 1e-6)
        X = np.arcsinh(Xr / co[None, :]).astype(np.float32)
    else:
        X = np.arcsinh(Xr / cfg['transform_cofactor']).astype(np.float32)
    names = [v if v != 'NA' else s for s, v in zip(cells['marker_cols'], cells['marker_vocab'])]
    bankf = OUT.parent / 'cohorts' / name / 'panel_bank.json'
    bank = json.loads(bankf.read_text()) if bankf.exists() else None
    if bank:   # restrict to the bank's full panel (drops non-protein columns exactly as the bank did)
        ok = np.array([n in set(bank['full_panel']) for n in names])
        X, names = X[:, ok], [n for n, o in zip(names, ok) if o]
        vocab = [v for v, o in zip(cells['marker_vocab'], ok) if o]
    else:
        vocab = list(cells['marker_vocab'])
    gt6 = {k: m[keep] for k, m in celltype_masks(name, cells['node17'], cells['label_fine']).items()}
    prim = [e for e, v in cfg['endpoints'].items() if v['primary']][0]
    f = endpoint_frame(name, clin, prim)
    elig = np.sort(f.pidx[cnt['n_mapped'][f.pidx] >= MIN_CELLS].to_numpy())
    by_p = pd.Series(np.arange(len(keep))).groupby(cp[keep]).apply(lambda s: s.to_numpy()).to_dict()
    return {'name': name, 'X': X, 'y': y, 'pat': cp[keep], 'names': names, 'vocab': vocab, 'gt6': gt6,
            'elig': elig, 'by_p': by_p, 'n_pat': len(clin), 'bank': bank}


def sample(rng, by_p, pats, cap):
    out = [by_p[p] if len(by_p[p]) <= cap else rng.choice(by_p[p], cap, replace=False) for p in pats if p in by_p]
    return np.sort(np.concatenate(out)) if out else np.array([], dtype=np.int64)


# ---------------------------------------------------------------------------------------------------- mapping
def feature_map(adm: AdmitM1, adm_mask: np.ndarray, labels_present):
    """F_P = supported groups of the labels; node -> feature key (first ancestor-or-self of pi_P(n) in F_P,
    else 'coarse:<pi_P(n)>')."""
    F = sorted({int(adm.gmap[l]) for l in labels_present if adm.gmap[l] >= 0})
    proj = ONTO.projection_map(adm_mask)
    keys = []
    for n in range(ONTO.K):
        c = proj[n]
        while c >= 0 and c not in F:
            c = ONTO.parent_idx[c]
        keys.append(NODE17[c] if c >= 0 else ('coarse:' + (NODE17[proj[n]] if proj[n] >= 0 else 'root')))
    names = sorted(set(keys), key=lambda k: (k.startswith('coarse:'), k))
    return names, np.array([names.index(k) for k in keys])


def per_patient(pat, n_pat, idx, minlength):
    return np.bincount(pat * minlength + idx, minlength=n_pat * minlength).reshape(n_pat, minlength)


def summarize_typing(pat, n_pat, forced, sets, cons, gt6_masks_cells, fmaps):
    """Per-patient counts. fmaps: {panel: (names, node->feature idx)} for set-containment features."""
    out = {}
    out['forced17'] = per_patient(pat, n_pat, forced, 17)
    c19 = np.where(cons >= 0, cons, np.where(cons == -1, 17, 18))
    out['cons19'] = per_patient(pat, n_pat, c19, 19)
    size = sets.sum(1)
    out['setsize'] = np.bincount(pat, weights=size, minlength=n_pat)
    out['nonsingleton'] = np.bincount(pat, weights=(size != 1), minlength=n_pat)
    out['n'] = np.bincount(pat, minlength=n_pat)
    # E12 bounds for the six round-1 types (node17 definitions; 'treg' etc. by node)
    LU = np.zeros((n_pat, len(CT6), 2))
    empty = size == 0
    for j, ct in enumerate(CT6):
        nodes = {'cyto_T': ['Cytotoxic_T_Cell'], 'treg': ['Regulatory_T_Cell'], 'helper_T': ['Helper_T_Cell'],
                 'pooled_T': ['T_Lineage'], 'B': ['B_Cell'], 'myeloid_mac': ['Myeloid']}[ct]
        sub = np.zeros(17, bool)
        anc = np.zeros(17, bool)
        for nd in nodes:
            sub |= ONTO.anc[IDX[nd]]                # descendants-or-self of nd
            anc |= ONTO.anc[:, IDX[nd]]             # ancestors-or-self of nd
        if ct == 'myeloid_mac':
            sub = np.isin(np.arange(17), [IDX[n] for n in ('Myeloid', 'Macrophage_CD163neg', 'Macrophage_CD163pos')])
        inside = ~(sets & ~sub[None, :]).any(1) & ~empty
        touch = (sets & (sub | anc)[None, :]).any(1) | empty
        LU[:, j, 0] = np.bincount(pat, weights=inside, minlength=n_pat)
        LU[:, j, 1] = np.bincount(pat, weights=touch, minlength=n_pat)
    out['LU6'] = LU
    for pn, (fnames, fidx) in fmaps.items():
        M = np.zeros((17, len(fnames)), dtype=np.float32)
        M[np.arange(17), fidx] = 1
        cont = (sets.astype(np.float32) @ M) > 0
        out[f'contains@{pn}'] = np.stack([np.bincount(pat, weights=cont[:, j], minlength=n_pat)
                                          for j in range(len(fnames))], 1)
    return out


# ---------------------------------------------------------------------------------------------------- core
def fit_models(d, tr, panels_mask, seed, rt_panels):
    mu, sd = d['X'][tr].mean(0), d['X'][tr].std(0)
    sd[sd == 0] = 1.0
    Ztr = (d['X'][tr] - mu) / sd
    models = {'MI': fit_xgb(Ztr, d['y'][tr], seed)}
    for pn in rt_panels:
        z = Ztr.copy()
        z[:, panels_mask[pn]] = 0.0
        models[('RT', pn)] = fit_xgb(z, d['y'][tr], seed)
    return models, mu, sd


def scorer_for(models, scorer, pn):
    return models['MI'] if (scorer == 'MI' or pn == 'full') else models[('RT', pn)]


def type_block(models, Zc, yc, Zt, pat_t, n_pat, panels_mask, panels_vocab, gt6, typed_panels, fmap_panels,
               scorers=('MI', 'RT'), fixed_fmaps=None):
    """Calibrate ADMIT per (scorer, panel) on Zc, type Zt. Returns results dict and per-panel feature maps."""
    res, fmaps, gmaps = {}, {}, {}
    adm_fit = {}
    for pn in fmap_panels:
        for sc in scorers:
            if sc == 'RT' and pn == 'full':
                continue
            m, present = scorer_for(models, sc, pn)
            zc = Zc.copy()
            zc[:, panels_mask[pn]] = 0.0
            Pc = predict17(m, present, zc, ONTO.K)
            adm = AdmitM1(ONTO, ALPHA).fit(ONTO.aggregate(Pc), yc, panels_vocab[pn])
            adm_fit[(sc, pn)] = adm
        a = adm_fit[('MI', pn)]
        fmaps[pn] = fixed_fmaps[pn] if fixed_fmaps else feature_map(a, a.adm, np.unique(yc))
        gmaps[pn] = {'F': fmaps[pn][0], 'node_to_feature': fmaps[pn][1].tolist(),
                     'groups': a.strata_report(), 'admissible': [NODE17[c] for c in np.flatnonzero(a.adm)]}
    for pn in typed_panels:
        for sc in scorers:
            if sc == 'RT' and pn == 'full':
                continue
            m, present = scorer_for(models, sc, pn)
            zt = Zt.copy()
            zt[:, panels_mask[pn]] = 0.0
            Pt = predict17(m, present, zt, ONTO.K)
            adm = adm_fit[(sc, pn)]
            sets = adm.predict(ONTO.aggregate(Pt))
            cons = consensus_node(sets, ONTO)
            forced = Pt.argmax(1)
            fm = fmap_panels if pn == 'full' else [pn]
            r = summarize_typing(pat_t, n_pat, forced, sets, cons, gt6, {q: fmaps[q] for q in fm})
            r['maxprob_sum'] = np.bincount(pat_t, weights=Pt.max(1), minlength=n_pat)
            for k, v in r.items():
                res[f'{sc}|{pn}|{k}'] = v
            del Pt, sets, cons
    return res, gmaps, fmaps


def gt_counts(d, cells, n_pat):
    pat = d['pat'][cells]
    out = {'gt17': per_patient(pat, n_pat, d['y'][cells], 17)}
    for k, m in d['gt6'].items():
        out[f'gt6|{k}'] = np.bincount(pat, weights=m[cells], minlength=n_pat)
    return out


def panel_masks(d, panel_names, extra=None):
    masks, vocab = {}, {}
    full_vocab = [v for v in d['vocab'] if v != 'NA']
    for pn in panel_names:
        if pn == 'full':
            deleted = set()
        else:
            p = [q for q in d['bank']['panels'] if q['name'] == pn]
            if not p:
                continue
            deleted = set(p[0]['deleted'])
        masks[pn] = np.array([n in deleted for n in d['names']])
        vocab[pn] = [v for v, n in zip(d['vocab'], d['names']) if v != 'NA' and n not in deleted]
    if extra:
        for pn, keep_vocab in extra.items():
            masks[pn] = np.array([(v == 'NA') or (v not in set(keep_vocab)) for v in d['vocab']])
            vocab[pn] = [v for v in full_vocab if v in set(keep_vocab)]
    return masks, vocab


def run_sim(name, seed):
    t0 = time.time()
    d = load(name)
    rng = np.random.default_rng(2000 + seed)
    perm = rng.permutation(d['elig'])
    n = len(perm)
    a, b, c = int(0.25 * n), int(0.40 * n), int(0.70 * n)
    groups = {'fit': perm[:a], 'cal': perm[a:b], 'train': perm[b:c], 'deploy': perm[c:]}
    masks, vocab = panel_masks(d, BANK_PANELS)
    panels = [p for p in BANK_PANELS if p in masks]
    tr = sample(rng, d['by_p'], groups['fit'], FITCAL_CAP)
    ca = sample(rng, d['by_p'], groups['cal'], FITCAL_CAP)
    ty = sample(rng, d['by_p'], np.r_[groups['train'], groups['deploy']], TYPE_CAP)
    models, mu, sd = fit_models(d, tr, masks, seed, [p for p in panels if p != 'full'])
    log(name, 'seed', seed, 'models', f'{time.time()-t0:.0f}s')
    Zc = ((d['X'][ca] - mu) / sd).astype(np.float32)
    Zt = ((d['X'][ty] - mu) / sd).astype(np.float32)
    res, gmaps, _ = type_block(models, Zc, d['y'][ca], Zt, d['pat'][ty], d['n_pat'], masks, vocab, d['gt6'],
                               panels, panels)
    res.update(gt_counts(d, ty, d['n_pat']))
    grp = np.full(d['n_pat'], '', dtype=object)
    for g, ps in groups.items():
        grp[ps] = g
    res['group'] = grp.astype(str)
    TYP.mkdir(parents=True, exist_ok=True)
    tag = f'sim_{name}_s{seed}'
    np.savez_compressed(TYP / f'{tag}.npz', **res)
    write_json({'cohort': name, 'mode': 'sim', 'seed': seed, 'panels': panels, 'feature_maps': gmaps,
                'n_groups': {g: int(len(v)) for g, v in groups.items()}, 'n_typed_cells': int(len(ty)),
                'n_fit_cells': int(len(tr)), 'n_cal_cells': int(len(ca)), 'deleted': {
                    p: [n for n, m in zip(d['names'], masks[p]) if m] for p in panels},
                'wall_s': round(time.time() - t0, 1)}, TYP / f'{tag}_meta.json')
    log(name, 'seed', seed, 'done', f'{time.time()-t0:.0f}s')


def place_target(dA, dB):
    """B cells placed into A's columns by vocabulary name, z-scored on all B cells (label-free)."""
    muB, sdB = dB['X'].mean(0), dB['X'].std(0)
    sdB[sdB == 0] = 1.0
    Z = np.zeros((len(dB['X']), len(dA['names'])), dtype=np.float32)
    shared = []
    for j, v in enumerate(dA['vocab']):
        if v != 'NA' and v in dB['vocab']:
            jb = dB['vocab'].index(v)
            Z[:, j] = (dB['X'][:, jb] - muB[jb]) / sdB[jb]
            shared.append(v)
    return Z, shared


def run_xfit(name, seed, targets):
    t0 = time.time()
    d = load(name)
    rng = np.random.default_rng(3000 + seed)
    perm = rng.permutation(d['elig'])
    halves = [perm[: len(perm) // 2], perm[len(perm) // 2:]]
    tds = {t: load(t) for t in targets}
    extra = {}
    placed = {}
    for t, dB in tds.items():
        Z, shared = place_target(d, dB)
        placed[t] = Z
        extra[f'to:{t}'] = shared
    masks, vocab = panel_masks(d, ['full'], extra)
    res = {}
    gm_all = {}
    fixed = None
    for k in (0, 1):
        dev, ev = halves[k], halves[1 - k]
        dperm = rng.permutation(dev)
        nf = int(round(0.6 * len(dperm)))
        tr = sample(rng, d['by_p'], dperm[:nf], FITCAL_CAP)
        ca = sample(rng, d['by_p'], dperm[nf:], FITCAL_CAP)
        ty = sample(rng, d['by_p'], ev, TYPE_CAP)
        rt = list(extra) if k == 0 else []
        models, mu, sd = fit_models(d, tr, masks, seed, rt)
        Zc = ((d['X'][ca] - mu) / sd).astype(np.float32)
        Zt = ((d['X'][ty] - mu) / sd).astype(np.float32)
        fm_p = ['full'] + list(extra)
        r, gm, fm = type_block(models, Zc, d['y'][ca], Zt, d['pat'][ty], d['n_pat'], masks, vocab, d['gt6'],
                               ['full'], fm_p, scorers=('MI',), fixed_fmaps=fixed)
        if fixed is None:
            fixed = fm      # fold-0 feature maps are used for both folds (one feature space per panel)
        for kk, v in r.items():
            res[kk] = res.get(kk, 0) + v
        g = gt_counts(d, ty, d['n_pat'])
        for kk, v in g.items():
            res[kk] = res.get(kk, 0) + v
        gm_all[f'fold{k}'] = gm
        if k == 0:     # targets typed by the fold-0 classifier / calibration
            for t, dB in tds.items():
                pn = f'to:{t}'
                rngB = np.random.default_rng(4000 + seed)
                tyB = sample(rngB, dB['by_p'], dB['elig'], TYPE_CAP)
                ZB = placed[t][tyB]
                maskB = dict(masks)
                rB, gmB, _ = type_block(models, Zc, d['y'][ca], ZB, dB['pat'][tyB], dB['n_pat'], maskB, vocab,
                                        None, [pn], [pn], fixed_fmaps={pn: fixed[pn]})
                gB = gt_counts(dB, tyB, dB['n_pat'])
                outB = {**rB, **gB, 'elig': dB['elig']}
                np.savez_compressed(TYP / f'xfit_{name}_s{seed}__target_{t}.npz', **outB)
                write_json({'reference': name, 'target': t, 'seed': seed, 'shared_vocab': extra[pn],
                            'feature_maps': gmB, 'n_typed_cells': int(len(tyB))},
                           TYP / f'xfit_{name}_s{seed}__target_{t}_meta.json')
                log(name, '->', t, 'typed', len(tyB))
        log(name, 'seed', seed, 'fold', k, f'{time.time()-t0:.0f}s')
    res['elig'] = d['elig']
    TYP.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TYP / f'xfit_{name}_s{seed}.npz', **res)
    write_json({'cohort': name, 'mode': 'xfit', 'seed': seed, 'targets': targets, 'feature_maps': gm_all,
                'wall_s': round(time.time() - t0, 1)}, TYP / f'xfit_{name}_s{seed}_meta.json')
    log(name, 'xfit seed', seed, 'done', f'{time.time()-t0:.0f}s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['sim', 'xfit'])
    ap.add_argument('--cohort', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--targets', default='')
    a = ap.parse_args()
    TYP.mkdir(parents=True, exist_ok=True)
    if a.mode == 'sim':
        run_sim(a.cohort, a.seed)
    else:
        run_xfit(a.cohort, a.seed, [t for t in a.targets.split(',') if t])


if __name__ == '__main__':
    main()
