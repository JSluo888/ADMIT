"""Round-2 engine: arms at several alphas, additive per-(patient, node) sufficient statistics.

Every run (Orion protocol, Schurch head, generic cohort) fills one accumulator
    acc[panel, arm, alpha, patient, node, stat]
of sums that are additive over splits/folds, so a patient recurring across splits is one cluster
and every summary (pooled / per-stratum / per-type / equal-patient coverage, resolution, cluster
bootstrap) is computed later from acc alone (summarize_round2.py).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from admit.panel_matched import (ADMIT, AdmitM1, Ontology, SourcePATCH, admit_m1_crc, aps_flat,  # noqa: E402
                                       chain_to_leaf, crc_threshold, finest_depth, lac_flat, lac_node_projected,
                                       nmin_rule, patch_pm_crc, resolved_leaf)

ALPHAS = [0.05, 0.10, 0.15, 0.20]
A01 = 1                                      # index of alpha 0.10
BASE_ARMS = ["admit", "admit_m1", "admit_m1_n50", "lac_node_proj", "lac_filt", "aps_filt", "raps_filt",
             "patch_src_backstop", "patch_src"]
CRC_ARMS = ["admit_crc", "admit_m1_crc", "lac_crc_filt"]
SWEEP_ARMS = ["admit", "admit_m1", "lac_node_proj", "lac_filt", "aps_filt"]
STATS = ["g_hit", "g_n", "l_hit", "l_n_ident", "l_n", "l_size", "l_depth", "l_leafres", "l_single", "l_abst",
         "l_inadm", "l_leafres_correct"]
# l_leafres = prespecified E2 metric (chain ending in ANY leaf). l_leafres_correct (added before results, see
# deviations): the chain ends in the cell's own label. Needed because Immune_Other is a leaf requiring only CD45,
# so {Immune, Immune_Other} counts as "leaf-resolved" for every cell under any CD45 panel.
NSTAT = len(STATS)


def arms_for(alpha_idx: int, with_crc: bool) -> List[str]:
    if alpha_idx == A01:
        return BASE_ARMS + (CRC_ARMS if with_crc else [])
    return list(SWEEP_ARMS)


def run_arms_round2(onto: Ontology, panel, Pc, yc, Pt, Pc_full, alpha: float, arms: List[str],
                    gc: Optional[np.ndarray] = None, src_cache: Optional[Dict] = None):
    """Returns {arm: (sets (n_test, K) bool, tmap (K,) label->target node or -1)} and a strata report."""
    adm = onto.admissible(panel)
    proj = onto.projection_map(adm)
    Nc, Nt = onto.aggregate(Pc), onto.aggregate(Pt)
    out, rep = {}, {}
    want = set(arms)
    if "admit" in want:
        m = ADMIT(onto, alpha).fit(Nc, yc, panel)
        out["admit"] = (m.predict(Nt), proj)
        rep["admit_infinite"] = [onto.names[c] for c in np.where(m.infinite)[0]]
    for arm, nm in (("admit_m1", None), ("admit_m1_n50", 50)):
        if arm in want:
            m = AdmitM1(onto, alpha, nm).fit(Nc, yc, panel)
            out[arm] = (m.predict(Nt), m.gmap)
            rep[arm] = {"n_min": m.n_min_used, "groups": m.strata_report(),
                        "n_cal_unidentifiable": int((m.target(yc) < 0).sum()),
                        "map": {onto.names[d]: (onto.names[m.gmap[d]] if m.gmap[d] >= 0 else None)
                                for d in np.unique(yc)}}
    if "lac_node_proj" in want:
        out["lac_node_proj"] = (lac_node_projected(Nc, yc, Nt, onto, panel, alpha), proj)
    flt = adm[None, :]
    if "lac_filt" in want:
        out["lac_filt"] = (lac_flat(Pc, yc, Pt, alpha) & flt, proj)
    if "aps_filt" in want:
        out["aps_filt"] = (aps_flat(Pc, yc, Pt, alpha) & flt, proj)
    if "raps_filt" in want:
        out["raps_filt"] = (aps_flat(Pc, yc, Pt, alpha, k_reg=3, lam=0.05) & flt, proj)
    if want & {"patch_src_backstop", "patch_src"}:
        key = ("src", alpha)
        if src_cache is not None and key in src_cache:
            src = src_cache[key]
        else:
            src = SourcePATCH(onto, alpha, 10).fit(onto.aggregate(Pc_full), yc)
            if src_cache is not None:
                src_cache[key] = src
        if "patch_src_backstop" in want:
            out["patch_src_backstop"] = (src.predict(Nt, panel, backstop=True), proj)
        if "patch_src" in want:
            out["patch_src"] = (src.predict(Nt, panel, backstop=False), proj)
    if gc is not None:
        if "admit_crc" in want:
            out["admit_crc"] = (patch_pm_crc(Nc, yc, gc, Nt, onto, panel, alpha)[0], proj)
        if "admit_m1_crc" in want:
            s, gmap = admit_m1_crc(Nc, yc, gc, Nt, onto, panel, alpha)
            out["admit_m1_crc"] = (s, gmap)
        if "lac_crc_filt" in want:
            t = crc_threshold(1.0 - Pc[np.arange(len(yc)), yc], alpha, gc)
            out["lac_crc_filt"] = (((1.0 - Pt) <= t) & flt, proj)
    return out, rep


def suff_stats(sets: np.ndarray, yt: np.ndarray, tmap: np.ndarray, pslot: np.ndarray, n_pat: int,
               onto: Ontology, adm: np.ndarray) -> np.ndarray:
    """(n_pat, K, NSTAT) sums for one arm on one block of test cells."""
    K = onto.K
    n = len(yt)
    tgt = tmap[yt]
    ident = tgt >= 0
    hit = np.zeros(n, dtype=bool)
    hit[ident] = sets[np.where(ident)[0], tgt[ident]]
    size = sets.sum(axis=1)
    depth = finest_depth(sets, onto)
    lr = chain_to_leaf(sets, onto)
    inadm = (sets & ~adm[None, :]).sum(axis=1)
    out = np.zeros((n_pat * K, NSTAT))
    gi = pslot[ident] * K + tgt[ident]
    out[:, 0] = np.bincount(gi, weights=hit[ident], minlength=n_pat * K)
    out[:, 1] = np.bincount(gi, minlength=n_pat * K)
    li = pslot * K + yt
    out[:, 2] = np.bincount(li, weights=(hit & ident), minlength=n_pat * K)
    out[:, 3] = np.bincount(li, weights=ident, minlength=n_pat * K)
    out[:, 4] = np.bincount(li, minlength=n_pat * K)
    out[:, 5] = np.bincount(li, weights=size, minlength=n_pat * K)
    out[:, 6] = np.bincount(li, weights=depth, minlength=n_pat * K)
    out[:, 7] = np.bincount(li, weights=lr, minlength=n_pat * K)
    out[:, 8] = np.bincount(li, weights=(size == 1), minlength=n_pat * K)
    out[:, 9] = np.bincount(li, weights=(size == 0), minlength=n_pat * K)
    out[:, 10] = np.bincount(li, weights=inadm, minlength=n_pat * K)
    out[:, 11] = np.bincount(li, weights=(resolved_leaf(sets, onto) == yt), minlength=n_pat * K)
    return out.reshape(n_pat, K, NSTAT)


def all_arm_names(with_crc: bool) -> List[str]:
    return BASE_ARMS + (CRC_ARMS if with_crc else [])


def evaluate_block(acc, panel_i, onto, panel, Pc, yc, Pt, yt, pslot, n_pat, Pc_full, with_crc, gc=None,
                   arm_index=None, alphas=ALPHAS, src_cache=None):
    """Run all arms at all alphas on one calibration/test block and add into acc[panel_i]."""
    adm = onto.admissible(panel)
    reps = {}
    for ai, a in enumerate(alphas):
        sets, rep = run_arms_round2(onto, panel, Pc, yc, Pt, Pc_full, a, arms_for(ai, with_crc), gc, src_cache)
        if ai == A01:
            reps = rep
        for arm, (S, tmap) in sets.items():
            acc[panel_i, arm_index[arm], ai] += suff_stats(S, yt, tmap, pslot, n_pat, onto, adm)
    return reps
