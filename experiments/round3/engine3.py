"""Evaluation engine: frozen support map g_P, exact ADMIT-M1, every arm scored on pi_P and g_P.

Accumulators (all additive over splits / folds / draws, per test patient):
  acc  [panel, arm, alpha, patient, node, STATS]   arm-specific; target-node stats are indexed by the target node,
                                                   l_* stats by the cell's original label
  acct [panel, alpha, patient, label, TSTATS]      arm-independent target bookkeeping (coarsening, no support, ...)
  grp  [panel, arm, alpha, node, GSTATS]           calibration groups: times a node was a group, times lambda=+inf,
                                                   calibration cells in the group (summed over blocks)
Targets: own = the arm's own target (pi_P for every arm, g_P for admit_m1 / lac_node_gp / admit_m1_crc, the
calibration-chosen map for admit_m1cal); pi = pi_P(y); gp = g_P(y) = s_P(pi_P(y)) (frozen from training labels).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from admit.panel_matched import (ADMIT, AdmitM1, AdmitM1Frozen, Ontology, SourcePATCH,  # noqa: E402
                                 admit_m1_frozen_crc, aps_flat, chain_to_leaf, finest_depth, frozen_support_map,
                                 lac_flat, lac_node_projected, lac_node_target, nmin_rule, resolved_leaf)

ALPHAS = [0.05, 0.10, 0.15, 0.20]
A01 = 1
BASE_ARMS = ["admit", "admit_m1", "admit_m1cal", "lac_node_proj", "lac_node_gp", "lac_filt", "aps_filt",
             "patch_src_backstop"]
CRC_ARMS = ["admit_m1_crc"]
SWEEP_ARMS = ["admit", "admit_m1", "admit_m1cal", "lac_node_proj", "lac_node_gp", "lac_filt"]
STATS = ["own_hit", "own_n", "own_inf", "pi_hit", "pi_n", "gp_hit", "gp_n",
         "l_n", "l_own_hit", "l_own_n", "l_pi_hit", "l_gp_hit", "l_size", "l_depth", "l_leafres", "l_single",
         "l_abst", "l_inadm", "l_leafres_correct"]
TSTATS = ["t_n", "t_pi_ident", "t_gp_ident", "t_coarsened", "t_depth_lost", "t_nosup", "t_gp_ne_a10",
          "t_m1cal_ne_gp", "t_pi_changed_R"]
# t_pi_changed_R: pi_P under the requirement map R(c) = required_positive | required_measured differs from
# pi_P under required_positive only (or under the default map when a requirement override is given)
GSTATS = ["is_group", "is_inf", "n_cal"]
NSTAT, NTSTAT, NGSTAT = len(STATS), len(TSTATS), len(GSTATS)
SI = {s: i for i, s in enumerate(STATS)}
TI = {s: i for i, s in enumerate(TSTATS)}


def arms_for(alpha_idx: int, with_crc: bool) -> List[str]:
    if alpha_idx == A01:
        return BASE_ARMS + (CRC_ARMS if with_crc else [])
    return list(SWEEP_ARMS)


def all_arm_names(with_crc: bool) -> List[str]:
    return BASE_ARMS + (CRC_ARMS if with_crc else [])


def run_arms3(onto: Ontology, panel, Pc, yc, Pt, Pc_full, alpha: float, arms: Sequence[str], gmap, S,
              gc=None, src_cache: Optional[Dict] = None):
    """-> out {arm: (sets (n_test, K) bool, own target map (K,), +inf nodes (K,) bool)},
           groups {arm: (is_group (K,), is_inf (K,), n_cal (K,))}, extra (M1-cal map)."""
    adm = onto.admissible(panel)
    proj = onto.projection_map(adm)
    Nc, Nt = onto.aggregate(Pc), onto.aggregate(Pt)
    K = onto.K
    none = np.zeros(K, dtype=bool)
    out, groups, extra = {}, {}, {}
    want = set(arms)
    if "admit" in want:
        m = ADMIT(onto, alpha).fit(Nc, yc, panel)
        out["admit"] = (m.predict(Nt), proj, m.infinite.copy())
        groups["admit"] = (adm.copy(), m.infinite.copy(), m.n_c.astype(float))
    if "admit_m1" in want:
        m = AdmitM1Frozen(onto, alpha, gmap, S).fit(Nc, yc)
        out["admit_m1"] = (m.predict(Nt), gmap, m.infinite.copy())
        groups["admit_m1"] = (S.copy(), m.infinite.copy(), m.n_c.astype(float))
    if "admit_m1cal" in want:
        m = AdmitM1(onto, alpha, None).fit(Nc, yc, panel)
        out["admit_m1cal"] = (m.predict(Nt), m.gmap, none)
        groups["admit_m1cal"] = (m.S.copy(), none, m.n_c.astype(float))
        extra["m1cal_gmap"] = m.gmap
    if "lac_node_proj" in want:
        out["lac_node_proj"] = (lac_node_projected(Nc, yc, Nt, onto, panel, alpha), proj, none)
    if "lac_node_gp" in want:
        out["lac_node_gp"] = (lac_node_target(Nc, yc, Nt, gmap, S, alpha), gmap, none)
    flt = adm[None, :]
    if "lac_filt" in want:
        out["lac_filt"] = (lac_flat(Pc, yc, Pt, alpha) & flt, proj, none)
    if "aps_filt" in want:
        out["aps_filt"] = (aps_flat(Pc, yc, Pt, alpha) & flt, proj, none)
    if "patch_src_backstop" in want:
        key = ("src", alpha)
        if src_cache is not None and key in src_cache:
            src = src_cache[key]
        else:
            src = SourcePATCH(onto, alpha, 10).fit(onto.aggregate(Pc_full), yc)
            if src_cache is not None:
                src_cache[key] = src
        out["patch_src_backstop"] = (src.predict(Nt, panel, backstop=True), proj, none)
    if "admit_m1_crc" in want and gc is not None:
        s, lam = admit_m1_frozen_crc(Nc, yc, gc, Nt, gmap, S, alpha)
        inf = S & np.isposinf(lam)
        out["admit_m1_crc"] = (s, gmap, inf)
        groups["admit_m1_crc"] = (S.copy(), inf, np.bincount(gmap[yc][gmap[yc] >= 0], minlength=K).astype(float))
    return out, groups, extra


class TestBlock:
    """Per-block precomputation shared by all arms and alphas."""

    def __init__(self, onto: Ontology, yt: np.ndarray, pslot: np.ndarray, n_pat: int, adm: np.ndarray):
        self.onto, self.K = onto, onto.K
        self.yt = np.asarray(yt, dtype=np.int64)
        self.n = len(self.yt)
        self.pslot = np.asarray(pslot, dtype=np.int64)
        self.n_pat = n_pat
        self.adm = adm
        self.li = self.pslot * self.K + self.yt
        self.M = n_pat * self.K
        self.rows = np.arange(self.n)

    def lab(self, w=None):
        return np.bincount(self.li, weights=w, minlength=self.M)

    def tgt(self, t, w=None):
        ok = t >= 0
        return np.bincount(self.pslot[ok] * self.K + t[ok], weights=None if w is None else w[ok], minlength=self.M)

    def hits(self, sets, tmap):
        t = tmap[self.yt]
        h = np.zeros(self.n, dtype=bool)
        ok = t >= 0
        h[ok] = sets[self.rows[ok], t[ok]]
        return t, h


def suff_stats3(B: TestBlock, sets, own_map, own_inf, proj, gmap) -> np.ndarray:
    """(n_pat, K, NSTAT) sums for one arm on one block."""
    out = np.zeros((B.M, NSTAT))
    t_own, h_own = B.hits(sets, own_map)
    t_pi, h_pi = B.hits(sets, proj)
    t_gp, h_gp = B.hits(sets, gmap)
    out[:, SI["own_hit"]] = B.tgt(t_own, h_own.astype(float))
    out[:, SI["own_n"]] = B.tgt(t_own)
    inf_cell = np.zeros(B.n, dtype=float)
    ok = t_own >= 0
    inf_cell[ok] = own_inf[t_own[ok]]
    out[:, SI["own_inf"]] = B.tgt(t_own, inf_cell)
    out[:, SI["pi_hit"]] = B.tgt(t_pi, h_pi.astype(float))
    out[:, SI["pi_n"]] = B.tgt(t_pi)
    out[:, SI["gp_hit"]] = B.tgt(t_gp, h_gp.astype(float))
    out[:, SI["gp_n"]] = B.tgt(t_gp)
    size = sets.sum(axis=1)
    out[:, SI["l_n"]] = B.lab()
    out[:, SI["l_own_hit"]] = B.lab(h_own.astype(float))
    out[:, SI["l_own_n"]] = B.lab(ok.astype(float))
    out[:, SI["l_pi_hit"]] = B.lab(h_pi.astype(float))
    out[:, SI["l_gp_hit"]] = B.lab(h_gp.astype(float))
    out[:, SI["l_size"]] = B.lab(size.astype(float))
    out[:, SI["l_depth"]] = B.lab(finest_depth(sets, B.onto).astype(float))
    out[:, SI["l_leafres"]] = B.lab(chain_to_leaf(sets, B.onto).astype(float))
    out[:, SI["l_single"]] = B.lab((size == 1).astype(float))
    out[:, SI["l_abst"]] = B.lab((size == 0).astype(float))
    out[:, SI["l_inadm"]] = B.lab((sets & ~B.adm[None, :]).sum(axis=1).astype(float))
    out[:, SI["l_leafres_correct"]] = B.lab((resolved_leaf(sets, B.onto) == B.yt).astype(float))
    return out.reshape(B.n_pat, B.K, NSTAT)


def target_stats(B: TestBlock, proj, gmap, ref_gmap, m1cal_gmap=None, proj_legacy=None) -> np.ndarray:
    """(n_pat, K, NTSTAT) arm-independent sums, indexed by the original label."""
    onto = B.onto
    K = B.K
    lab_ok_pi = proj >= 0
    lab_ok_gp = gmap >= 0
    both = lab_ok_pi & lab_ok_gp
    per_label = np.zeros((K, NTSTAT))
    per_label[:, TI["t_n"]] = 1.0
    per_label[:, TI["t_pi_ident"]] = lab_ok_pi
    per_label[:, TI["t_gp_ident"]] = lab_ok_gp
    per_label[:, TI["t_coarsened"]] = both & (gmap != proj)
    dl = np.zeros(K)
    dl[both] = onto.depth[proj[both]] - onto.depth[gmap[both]]
    per_label[:, TI["t_depth_lost"]] = dl
    per_label[:, TI["t_nosup"]] = lab_ok_pi & ~lab_ok_gp
    per_label[:, TI["t_gp_ne_a10"]] = gmap != ref_gmap
    if m1cal_gmap is not None:
        per_label[:, TI["t_m1cal_ne_gp"]] = m1cal_gmap != gmap
    if proj_legacy is not None:
        per_label[:, TI["t_pi_changed_R"]] = proj_legacy != proj
    cnt = B.lab().reshape(B.n_pat, K)                      # cells per (patient, label)
    return cnt[:, :, None] * per_label[None, :, :]


def maps_for(onto, panel, y_train, n_cal, alphas, vocab=None):
    adm = onto.admissible(panel)
    return {a: frozen_support_map(onto, adm, y_train, n_cal, nmin_rule(a), vocab) for a in alphas}


def evaluate_block3(acc, acct, grp, panel_i, onto: Ontology, panel, Pc, yc, Pt, yt, pslot, n_pat, Pc_full, y_train,
                    with_crc, gc=None, arm_index=None, alphas=None, src_cache=None, n_cal=None, vocab=None,
                    arm_list=None):
    """Run every arm at every alpha on one calibration/test block and add into acc/acct/grp[panel_i].

    alphas: list of (alpha index in the accumulator, alpha); default enumerate(ALPHAS).
    y_train: training labels (after caps) for the frozen map; n_cal: calibration cells for the expected counts
    (default len(yc)); vocab: target annotation vocabulary (node indices) for cross-cohort maps.
    Returns a compact report of the alpha 0.10 maps / groups for the run metadata."""
    alphas = list(enumerate(ALPHAS)) if alphas is None else alphas
    adm = onto.admissible(panel)
    proj = onto.projection_map(adm)
    proj_legacy = onto.projection_map(onto.admissible(panel, legacy=True))
    ncal = len(yc) if n_cal is None else n_cal
    amaps = maps_for(onto, panel, y_train, ncal, sorted({a for _, a in alphas} | {0.10}), vocab)
    ref = amaps[0.10][0]
    B = TestBlock(onto, yt, pslot, n_pat, adm)
    rep = {}
    for ai, a in alphas:
        gmap, S, e = amaps[a]
        arms = arm_list if arm_list is not None else arms_for(ALPHAS.index(a) if a in ALPHAS else A01, with_crc)
        sets, groups, extra = run_arms3(onto, panel, Pc, yc, Pt, Pc_full, a, arms, gmap, S, gc, src_cache)
        for arm, (Sx, own_map, own_inf) in sets.items():
            acc[panel_i, arm_index[arm], ai] += suff_stats3(B, Sx, own_map, own_inf, proj, gmap)
        if grp is not None:
            for arm, (isg, isinf, ncal_c) in groups.items():
                grp[panel_i, arm_index[arm], ai] += np.stack([isg.astype(float), isinf.astype(float), ncal_c], 1)
        acct[panel_i, ai] += target_stats(B, proj, gmap, ref, extra.get("m1cal_gmap"), proj_legacy)
        if abs(a - 0.10) < 1e-9:
            nm = onto.names
            rep = {"n_min": nmin_rule(a), "n_cal": int(ncal), "n_train": int(len(y_train)),
                   "pi_P_changed_by_R": {nm[d]: [nm[proj_legacy[d]] if proj_legacy[d] >= 0 else None,
                                                 nm[proj[d]] if proj[d] >= 0 else None]
                                         for d in np.unique(yt) if proj_legacy[d] != proj[d]},
                   "S_P": [nm[c] for c in np.where(S)[0]],
                   "expected_counts": {nm[c]: round(float(e[c]), 2) for c in np.where(adm)[0]},
                   "g_P": {nm[d]: (nm[gmap[d]] if gmap[d] >= 0 else None) for d in np.unique(yt)},
                   "infinite_groups": {arm: [nm[c] for c in np.where(g[1])[0]] for arm, g in groups.items()
                                       if g[1].any()},
                   "n_cal_per_group_admit_m1": ({nm[c]: int(groups["admit_m1"][2][c]) for c in np.where(S)[0]}
                                               if "admit_m1" in groups else None),
                   "m1cal_map": ({nm[d]: (nm[extra["m1cal_gmap"][d]] if extra["m1cal_gmap"][d] >= 0 else None)
                                  for d in np.unique(yt)} if "m1cal_gmap" in extra else None)}
    return rep


def new_accumulators(n_blocks, n_arms, n_alpha, n_pat, K, dtype=np.float64):
    acc = np.zeros((n_blocks, n_arms, n_alpha, n_pat, K, NSTAT), dtype=dtype)
    acct = np.zeros((n_blocks, n_alpha, n_pat, K, NTSTAT), dtype=dtype)
    grp = np.zeros((n_blocks, n_arms, n_alpha, K, NGSTAT), dtype=dtype)
    return acc, acct, grp


def ep_from_block(tmp_acc, test_slots, target="gp"):
    """Equal-patient and pooled coverage of one block accumulator tmp_acc[arm, alpha, patient, node, stat] for one
    target ('own' | 'pi' | 'gp'): -> (ep (arm, alpha), pooled (arm, alpha))."""
    h = tmp_acc[..., SI[f"{target}_hit"]].sum(axis=3)
    n = tmp_acc[..., SI[f"{target}_n"]].sum(axis=3)
    with np.errstate(invalid="ignore", divide="ignore"):
        ep = np.nanmean(np.where(n[:, :, test_slots] > 0, h[:, :, test_slots] / n[:, :, test_slots], np.nan), axis=2)
        pooled = h.sum(axis=2) / n.sum(axis=2)
    return ep, pooled
