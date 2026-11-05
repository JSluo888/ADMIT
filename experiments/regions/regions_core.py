"""Region selection core: choose the regions of a low-plex (panel L) slide to re-image with the full panel F.

Pure functions on precomputed node probabilities; no I/O. The cohort runner (run_regions.py) and the synthetic
check (tests/test_region_selection.py) both call ``run_panel``.

Conformal machinery is the ADMIT-M1 engine, unchanged: ``frozen_support_map`` (deepest-first merging of expected
counts, "no supported target" = -1) and ``AdmitM1Frozen`` (k-th order statistic, +inf when k > n_c, no backstop).
What changes here is only what they are fed:
- C_L^dev and the VOI cross-fit: the training-label map with N_cal = the calibrating cells;
- routed strata: one map per stratum, frozen from DEV labels of that stratum, n_cal = N_CAL,s =
  N_CAL x (fraction of DEV calibration cells in s); ``frozen_support_map(onto, adm, y_dev_s, N_CAL_s, n_min)``
  computes e_c = n^DEV_{s,c} x N_CAL,s / N_DEV,s;
- `cond` fits AdmitM1Frozen on the stratum's CAL cells, `pooled` on all CAL cells, both with g_s;
- data-independent / TEST-only arms (random_k, random_bern, density_k, oracle_k) and the b = 0 / b = 1 references
  use the population map of each panel (the stratum = all DEV calibration cells, n_cal = N_CAL) and thresholds from
  all CAL cells.

Resolution: a set RESOLVES TO LEAF l if it is non-empty, every member lies on the root-to-l path and l is a member
(engine3 `leafres_correct`). Primary utility `corC` (resolves to the reference leaf, fixed leaf universe); selection
score u(x) = 0 iff C_L^dev(x) resolves to some leaf (``u_def='chain'``, default), and the same definition for u_F in
the VOI cross-fit. The singleton metrics (`cor1`, `unres1`: set == {leaf}) are kept as a secondary diagnostic.
"""
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from admit.panel_matched import AdmitM1Frozen, Ontology, frozen_support_map, nmin_rule, resolved_leaf  # noqa: E402

ALPHA = 0.10
BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.50)
B_PRIMARY = 0.20
EVAL_CAP = 1000          # one cap for calibration/evaluation cells in every role
CONTEXT_CAP = 3000       # context cells per region, only if a census is not used
MIN_REGION_CELLS = 50    # regions with fewer cells are dropped
Q_MIN = 20               # minimum cells per VOI lookup entry
SALT_CONTEXT, SALT_EVAL = 5, 7
RULES = ("uncert", "voi", "density")       # rule_index for the random_k draws
VARIANTS = ("uncert_cond", "uncert_pooled", "voi_cond", "voi_pooled", "density_cond", "density_pooled",
            "random_k_uncert", "random_k_voi", "random_k_density", "random_bern", "density_k",
            "oracle_k_uncert", "oracle_k_voi", "ref_L", "ref_F")
VI = {v: i for i, v in enumerate(VARIANTS)}
ORACLE_VARIANTS = ("oracle_k_uncert", "oracle_k_voi")
# per TEST region (evaluation cells): n_eval, leaf-universe size, correct singleton leaf (diagnostic), correct
# chain-to-leaf (primary), unresolved (not a singleton leaf / not chain-to-leaf; all eval cells), sum |C|,
# empty sets, coverage hits / denominators on the routed stratum's own target, panel-unidentifiable (pi = -1),
# identifiable but no supported target (g = -1)
RST = ("n_eval", "n_leafU", "cor1", "corC", "unres1", "unresC", "size", "abst", "cov_hit", "cov_n", "unident",
       "nosup")
# per (stratum, TEST patient, node): by reference label l_*, by target node t_*
TST = ("l_n", "l_cor1", "l_corC", "t_n", "t_hit")
SINFO = ("n_dev_regions", "n_cal_regions", "n_test_regions", "N_dev", "N_cal_A2", "n_cal_cells", "n_supported",
         "n_inf_cond", "n_inf_pooled")
RI = {s: i for i, s in enumerate(RST)}
TI = {s: i for i, s in enumerate(TST)}


# ----------------------------------------------------------------------------------------------------------------
# Splits and samples
# ----------------------------------------------------------------------------------------------------------------


def dev_cal_split(cal_patients: Sequence, s: int):
    """Calibration patients of the split (sorted) permuted with default_rng([s, 2028]); the first
    ceil(n/3) are DEV, the rest CAL. Orion fold f: the 12 non-test patients, s = f (ceil(12/3) = 4 DEV)."""
    c = np.sort(np.asarray(cal_patients))
    perm = np.random.default_rng([int(s), 2028]).permutation(c)
    n_dev = int(math.ceil(len(c) / 3))
    return perm[:n_dev], perm[n_dev:]


def dev_folds(dev_patients: Sequence, s: int):
    """DEV patients (sorted) permuted with default_rng([s, 2029]); fold A = first floor(n/2), fold B = rest."""
    d = np.sort(np.asarray(dev_patients))
    perm = np.random.default_rng([int(s), 2029]).permutation(d)
    h = len(d) // 2
    return perm[:h], perm[h:]


def sample_rows(rows: np.ndarray, region_index: int, cap: Optional[int], s: int, salt: int) -> np.ndarray:
    """Uniform sample without replacement of at most ``cap`` rows of one region, default_rng([s, region_index,
    salt]) (salt 7 = evaluation cells, 5 = context cells); sorted. cap None/0 = all rows."""
    rows = np.asarray(rows)
    if not cap or len(rows) <= cap:
        return np.sort(rows)
    return np.sort(np.random.default_rng([int(s), int(region_index), int(salt)]).choice(rows, cap, replace=False))


def bern_u(cohort: str, region_id: str, s: int) -> float:
    """random_bern: frozen hash of (cohort, region id, s) mapped to [0, 1)."""
    h = hashlib.sha256(f"{cohort}|{region_id}|{int(s)}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def convex_hull_area(xy: np.ndarray) -> float:
    xy = np.asarray(xy, dtype=np.float64)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 3:
        return 0.0
    from scipy.spatial import ConvexHull, QhullError
    try:
        return float(ConvexHull(xy).volume)
    except (QhullError, ValueError):
        return 0.0


# ----------------------------------------------------------------------------------------------------------------
# Set utilities
# ----------------------------------------------------------------------------------------------------------------


def set_bits(sets: np.ndarray) -> np.ndarray:
    return sets.astype(np.int64) @ (np.int64(1) << np.arange(sets.shape[1], dtype=np.int64))


def finest_bits(sets: np.ndarray, onto: Ontology) -> np.ndarray:
    """Members of the set at its maximal depth (the finest node; ties keep every member at that depth); 0 = empty."""
    d = np.where(sets, onto.depth[None, :], 0)
    mx = d.max(axis=1)
    return set_bits(sets & (d == mx[:, None]) & (mx[:, None] > 0))


def single_leaf(sets: np.ndarray, onto: Ontology) -> np.ndarray:
    """The set is exactly {one leaf of the hierarchy}."""
    return (sets.sum(axis=1) == 1) & onto.is_leaf[np.argmax(sets, axis=1)]


def unresolved(sets: np.ndarray, onto: Ontology, u_def: str = "chain") -> np.ndarray:
    """u = 0 iff the set resolves to some leaf (non-empty, every member on that leaf's root path, the leaf a member);
    empty sets count as unresolved. u_def='singleton' (set == {leaf}) is kept as a diagnostic."""
    if u_def == "singleton":
        return ~single_leaf(sets, onto)
    if u_def == "chain":
        return resolved_leaf(sets, onto) < 0
    raise ValueError(u_def)


def q_table(key: np.ndarray, fin: np.ndarray, u_f: np.ndarray, min_n: int = Q_MIN) -> Dict:
    """VOI lookup: mean out-of-fold u_F per out-of-fold L-set key (>= min_n cells), else per finest node
    (>= min_n), else the overall mean."""
    out = {"overall": float(np.mean(u_f)) if len(u_f) else float("nan"), "min_n": min_n}
    for name, k in (("key", key), ("fin", fin)):
        uk, inv = np.unique(k, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(uk))
        mean = np.bincount(inv, weights=u_f, minlength=len(uk)) / np.maximum(cnt, 1)
        ok = cnt >= min_n
        out[name] = (uk[ok], mean[ok])
    return out


def q_lookup(tab: Dict, key: np.ndarray, fin: np.ndarray) -> np.ndarray:
    q = np.full(len(key), tab["overall"], dtype=np.float64)
    done = np.zeros(len(key), dtype=bool)
    for name, k in (("key", key), ("fin", fin)):
        uk, mean = tab[name]
        if len(uk) == 0:
            continue
        pos = np.clip(np.searchsorted(uk, k), 0, len(uk) - 1)
        hit = (uk[pos] == k) & ~done
        q[hit] = mean[pos[hit]]
        done |= hit
    return q


# ----------------------------------------------------------------------------------------------------------------
# Selection rules
# ----------------------------------------------------------------------------------------------------------------


def dev_threshold(dev_scores: np.ndarray, b: float) -> float:
    """tau_b = (1 - b) quantile (numpy default, linear) of the rule's score over DEV regions."""
    return float(np.quantile(np.asarray(dev_scores, dtype=np.float64), 1.0 - b))


def select(scores: np.ndarray, tau: float) -> np.ndarray:
    """S_b(r) = 1 iff score(r) > tau_b: a fixed function of the region's own L-data given training + DEV."""
    return np.asarray(scores) > tau


def random_k(n_test: int, k: int, s: int, b_index: int, rule_index: int) -> np.ndarray:
    """random_k: exactly k of the n_test TEST regions, default_rng([s, b_index, rule_index, 11])."""
    sel = np.zeros(n_test, dtype=bool)
    if k > 0:
        sel[np.random.default_rng([int(s), int(b_index), int(rule_index), 11]).choice(n_test, int(k),
                                                                                       replace=False)] = True
    return sel


def top_k(values: np.ndarray, k: int, tiebreak: np.ndarray) -> np.ndarray:
    """k largest values (ties: smaller tiebreak first)."""
    sel = np.zeros(len(values), dtype=bool)
    if k > 0:
        sel[np.lexsort((tiebreak, -np.asarray(values, dtype=np.float64)))[:k]] = True
    return sel


def stratum_maps(onto: Ontology, adm_F: np.ndarray, adm_L: np.ndarray, y_dev: np.ndarray, dev_selected: np.ndarray,
                 n_cal: int, n_min: int):
    """Per routed stratum (1 = selected -> F, 0 = unselected -> L) the frozen support rule on DEV labels:
    e_c = n^DEV_{s,c} x N_CAL,s / N_DEV,s with N_CAL,s = N_CAL x N_DEV,s / N_DEV. Uses DEV labels, the DEV
    selection and the CAL cell COUNT only. -> {s: (gmap, S, e, N_dev_s, N_cal_s)}"""
    y_dev = np.asarray(y_dev, dtype=np.int64)
    n_dev = len(y_dev)
    out = {}
    for st, adm in ((1, adm_F), (0, adm_L)):
        ys = y_dev[np.asarray(dev_selected, dtype=bool) == bool(st)]
        n_cal_s = float(n_cal) * len(ys) / n_dev if n_dev else 0.0
        g, S, e = frozen_support_map(onto, adm, ys, n_cal_s, n_min)
        out[st] = (g, S, e, len(ys), n_cal_s)
    return out


def m1_fit(onto: Ontology, gmap: np.ndarray, S: np.ndarray, Pn_cal: np.ndarray, y_cal: np.ndarray) -> AdmitM1Frozen:
    return AdmitM1Frozen(onto, ALPHA, gmap, S).fit(Pn_cal, y_cal)


# ----------------------------------------------------------------------------------------------------------------
# Evaluation accumulators
# ----------------------------------------------------------------------------------------------------------------


class TestEval:
    """TEST evaluation cells of one fold: region position (0..n_treg-1), patient slot, reference label."""

    def __init__(self, onto: Ontology, y: np.ndarray, treg: np.ndarray, n_treg: int, pslot: np.ndarray, n_pat: int):
        self.onto, self.K = onto, onto.K
        self.y = np.asarray(y, dtype=np.int64)
        self.n = len(self.y)
        self.treg, self.nR = np.asarray(treg, dtype=np.int64), n_treg
        self.pslot, self.nP = np.asarray(pslot, dtype=np.int64), n_pat
        self.rows = np.arange(self.n)
        self.leafU = onto.is_leaf[self.y]

    def correct1(self, sets: np.ndarray) -> np.ndarray:
        return (sets.sum(axis=1) == 1) & sets[self.rows, self.y] & self.leafU

    def correct_chain(self, sets: np.ndarray) -> np.ndarray:
        return (resolved_leaf(sets, self.onto) == self.y) & self.leafU

    def stats(self, sets: np.ndarray, t: np.ndarray, pi: np.ndarray, strat: np.ndarray):
        size = sets.sum(axis=1)
        cor1 = self.correct1(sets)
        sl = (size == 1) & self.onto.is_leaf[np.argmax(sets, axis=1)]
        rl = resolved_leaf(sets, self.onto)
        cor_c = (rl == self.y) & self.leafU
        ok = t >= 0
        hit = np.zeros(self.n, dtype=bool)
        hit[ok] = sets[self.rows[ok], t[ok]]
        cols = (np.ones(self.n), self.leafU, cor1, cor_c, ~sl, rl < 0, size, size == 0, hit, ok, pi < 0,
                (pi >= 0) & ~ok)
        R = np.stack([np.bincount(self.treg, weights=np.asarray(c, dtype=np.float64), minlength=self.nR)
                      for c in cols], axis=1)
        M = 2 * self.nP * self.K
        base = (np.asarray(strat, dtype=np.int64) * self.nP + self.pslot) * self.K
        T = np.zeros((M, len(TST)))
        li = base + self.y
        T[:, TI["l_n"]] = np.bincount(li, minlength=M)
        T[:, TI["l_cor1"]] = np.bincount(li, weights=cor1.astype(np.float64), minlength=M)
        T[:, TI["l_corC"]] = np.bincount(li, weights=cor_c.astype(np.float64), minlength=M)
        ti = base[ok] + t[ok]
        T[:, TI["t_n"]] = np.bincount(ti, minlength=M)
        T[:, TI["t_hit"]] = np.bincount(ti, weights=hit[ok].astype(np.float64), minlength=M)
        return R.astype(np.float32), T.reshape(2, self.nP, self.K, len(TST)).astype(np.float32)


# ----------------------------------------------------------------------------------------------------------------
# One fold x one low-plex panel
# ----------------------------------------------------------------------------------------------------------------


def run_panel(onto: Ontology, panel_F: Sequence[str], panel_L: Sequence[str], y_train: np.ndarray, s: int, cohort: str,
              R: Dict[str, np.ndarray], ctx_reg: np.ndarray, PnL_ctx: np.ndarray, ev_reg: np.ndarray,
              ev_y: np.ndarray, PnL_ev: np.ndarray, PnF_ev: np.ndarray, P17L_ev: Optional[np.ndarray] = None,
              u_def: str = "chain", budgets: Sequence[float] = BUDGETS, keep_cells: bool = False,
              debug_hash: bool = False, selection_only: bool = False) -> Dict:
    """Scores, selection, routed calibration and evaluation for one fold (seed) and one low-plex panel L
    (F = full panel).

    R (per kept region, positions 0..nR-1): role (0 DEV, 1 CAL, 2 TEST), patient (int id; DEV folds and TEST
    patient slots use it), N_r (all cells of the region in the cohort table), region_id (str, hash rule),
    region_index (int, tie-break). Context cells (census or capped sample, every role the same way):
    ctx_reg (region position), PnL_ctx (node probabilities under L). Evaluation/calibration cells (<= 1,000 per
    region, labelled): ev_reg, ev_y, PnL_ev, PnF_ev, P17L_ev (label probabilities under L, forced call).
    Labels of CAL/TEST cells are used only by the calibrators (CAL) and the metrics and oracle (TEST)."""
    K = onto.K
    n_min = nmin_rule(ALPHA)
    role = np.asarray(R["role"])
    nR = len(role)
    nb = len(budgets)
    adm_L, adm_F = onto.admissible(panel_L), onto.admissible(panel_F)
    projL, projF = onto.projection_map(adm_L), onto.projection_map(adm_F)
    ev_reg = np.asarray(ev_reg, dtype=np.int64)
    ctx_reg = np.asarray(ctx_reg, dtype=np.int64)
    ev_y = np.asarray(ev_y, dtype=np.int64)
    ev_role = role[ev_reg]
    D, C, T = ev_role == 0, ev_role == 1, ev_role == 2
    yD, yC, yT = ev_y[D], ev_y[C], ev_y[T]
    PnL_D, PnF_D, PnL_C, PnF_C, PnL_T, PnF_T = PnL_ev[D], PnF_ev[D], PnL_ev[C], PnF_ev[C], PnL_ev[T], PnF_ev[T]
    N_dev, N_cal = int(D.sum()), int(C.sum())
    n_ctx = np.bincount(ctx_reg, minlength=nR)
    N_r = np.asarray(R["N_r"], dtype=np.float64)
    scale = np.where(n_ctx > 0, N_r / np.maximum(n_ctx, 1), 0.0)
    info: Dict = {"N_dev_cells": N_dev, "N_cal_cells": N_cal, "N_test_cells": int(T.sum()),
                  "n_regions": {k: int((role == i).sum()) for i, k in enumerate(("dev", "cal", "test"))}}

    # -- C_L^dev (training-label map, N_cal = |DEV calibration cells|, thresholds on all DEV) ----------
    g_tr, S_tr, _ = frozen_support_map(onto, adm_L, y_train, N_dev, n_min)
    m_dev = m1_fit(onto, g_tr, S_tr, PnL_D, yD)
    sets_ctx = m_dev.predict(PnL_ctx)
    uL_ctx = unresolved(sets_ctx, onto, u_def).astype(np.float64)
    key_ctx, fin_ctx = set_bits(sets_ctx), finest_bits(sets_ctx, onto)
    info["CLdev_supported"] = int(S_tr.sum())
    info["CLdev_inf_groups"] = int(m_dev.infinite.sum())

    # -- out-of-fold L and F cross-fit on DEV -----------------------------------------------------------------------
    dev_pat = np.unique(np.asarray(R["patient"])[role == 0])
    fa, fb = dev_folds(dev_pat, s)
    voi_ok = min(len(fa), len(fb)) >= 2
    info["voi_run"] = bool(voi_ok)
    info["dev_fold_patients"] = [int(len(fa)), int(len(fb))]
    q_ctx = np.full(len(ctx_reg), np.nan)
    if voi_ok:
        pat_D = np.asarray(R["patient"])[ev_reg[D]]
        in_a = np.isin(pat_D, fa)
        parts = []
        for fold in (in_a, ~in_a):
            other = ~fold
            n_o = int(other.sum())
            gL, SL, _ = frozen_support_map(onto, adm_L, y_train, n_o, n_min)
            gF, SF, _ = frozen_support_map(onto, adm_F, y_train, n_o, n_min)
            sL = m1_fit(onto, gL, SL, PnL_D[other], yD[other]).predict(PnL_D[fold])
            sF = m1_fit(onto, gF, SF, PnF_D[other], yD[other]).predict(PnF_D[fold])
            parts.append((set_bits(sL), finest_bits(sL, onto), unresolved(sF, onto, u_def).astype(np.float64),
                          unresolved(sL, onto, u_def).astype(np.float64)))
        key_o, fin_o, uF_o, uL_o = (np.concatenate(z) for z in zip(*parts))
        tab = q_table(key_o, fin_o, uF_o)
        q_ctx = q_lookup(tab, key_ctx, fin_ctx)
        info["oof"] = {"n": int(len(uF_o)), "n_keys_supported": int(len(tab["key"][0])),
                       "n_fin_supported": int(len(tab["fin"][0]))}

    # -- region scores, scaled by N_r / n_r ------------------------------------------------------------------------
    scores = np.full((len(RULES), nR), np.nan)
    scores[0] = np.bincount(ctx_reg, weights=uL_ctx, minlength=nR) * scale
    if voi_ok:
        scores[1] = np.bincount(ctx_reg, weights=uL_ctx - q_ctx, minlength=nR) * scale
    scores[2] = N_r
    ran = np.array([True, voi_ok, True])

    # -- DEV thresholds, fixed rule applied to every region ----------------------------------------------------------
    dev_r = role == 0
    test_r = np.flatnonzero(role == 2)
    nTR = len(test_r)
    taus = np.full((len(RULES), nb), np.nan)
    sel = np.zeros((len(RULES), nb, nR), dtype=bool)
    for ri in range(len(RULES)):
        if not ran[ri]:
            continue
        for bi, b in enumerate(budgets):
            taus[ri, bi] = dev_threshold(scores[ri][dev_r], b)
            sel[ri, bi] = select(scores[ri], taus[ri, bi])

    if selection_only:
        # blind phase: selections and stratum bookkeeping only (DEV labels for the maps, CAL label COUNTS for the
        # +inf groups); no TEST label is touched and no endpoint is computed
        sinfo = np.full((len(RULES), nb, 2, len(SINFO)), np.nan)
        sbits = np.zeros((len(RULES), nb, 2), dtype=np.int64)
        for ri in range(len(RULES)):
            if not ran[ri]:
                continue
            for bi in range(nb):
                sr = sel[ri, bi]
                dsel, csel = sr[ev_reg[D]], sr[ev_reg[C]]
                mp = stratum_maps(onto, adm_F, adm_L, yD, dsel, N_cal, n_min)
                for k_s, (Pc, cm) in enumerate(((PnL_C, ~csel), (PnF_C, csel))):
                    g_, S_, _, nd, nc = mp[k_s]
                    mc = m1_fit(onto, g_, S_, Pc[cm], yC[cm])
                    in_s = sr == bool(k_s)
                    sinfo[ri, bi, k_s] = (int((in_s & (role == 0)).sum()), int((in_s & (role == 1)).sum()),
                                          int((in_s & (role == 2)).sum()), nd, nc, int(cm.sum()), int(S_.sum()),
                                          int(mc.infinite.sum()), np.nan)
                    sbits[ri, bi, k_s] = int(set_bits(S_[None, :])[0])
        return {"taus": taus, "scores": scores, "sel": sel, "sinfo": sinfo, "sbits": sbits, "ran": ran,
                "test_regions": test_r, "info": info}

    # -- TEST bookkeeping --------------------------------------------------------------------------------------------
    tpos = np.full(nR, -1, dtype=np.int64)
    tpos[test_r] = np.arange(nTR)
    treg = tpos[ev_reg[T]]
    pat = np.asarray(R["patient"])
    tpat = np.unique(pat[test_r])
    pslot = np.searchsorted(tpat, pat[ev_reg[T]])
    Ev = TestEval(onto, yT, treg, nTR, pslot, len(tpat))
    tie = np.asarray(R["region_index"])[test_r]

    rstat = np.full((len(VARIANTS), nb, nTR, len(RST)), np.nan, dtype=np.float32)
    tstat = np.zeros((len(VARIANTS), nb, 2, len(tpat), K, len(TST)), dtype=np.float32)
    selT = np.zeros((len(VARIANTS), nb, nTR), dtype=bool)
    vran = np.zeros((len(VARIANTS), nb), dtype=bool)
    hashes = {}
    cells: Dict[str, np.ndarray] = {}

    def route(vname, bi, sel_t, sF, sL, gF_, gL_):
        c = sel_t[treg]
        sets = np.where(c[:, None], sF, sL)
        t = np.where(c, gF_[yT], gL_[yT])
        pi = np.where(c, projF[yT], projL[yT])
        vi = VI[vname]
        rstat[vi, bi], tstat[vi, bi] = Ev.stats(sets, t, pi, c.astype(np.int64))
        selT[vi, bi] = sel_t
        vran[vi, bi] = True
        if debug_hash:
            hashes[(vname, bi)] = hashlib.sha1(np.packbits(sets, axis=1).tobytes()).hexdigest()
        return sets

    # -- population maps + pooled thresholds (references, random / density_k / oracle arms) -----------------------
    gLp, SLp, _ = frozen_support_map(onto, adm_L, yD, N_cal, n_min)
    gFp, SFp, _ = frozen_support_map(onto, adm_F, yD, N_cal, n_min)
    mLp, mFp = m1_fit(onto, gLp, SLp, PnL_C, yC), m1_fit(onto, gFp, SFp, PnF_C, yC)
    sLp, sFp = mLp.predict(PnL_T), mFp.predict(PnF_T)
    info["pop_supported"] = {"L": int(SLp.sum()), "F": int(SFp.sum())}
    info["pop_inf_groups"] = {"L": int(mLp.infinite.sum()), "F": int(mFp.infinite.sum())}
    corr = Ev.correct1 if u_def == "singleton" else Ev.correct_chain
    gain = (np.bincount(treg, weights=corr(sFp).astype(np.float64), minlength=nTR)
            - np.bincount(treg, weights=corr(sLp).astype(np.float64), minlength=nTR))   # oracle gain, uses TEST labels

    # -- routed strata, cond and pooled -------------------------------------------------------------------------------
    sinfo = np.full((len(RULES), nb, 2, len(SINFO)), np.nan)
    sbits = np.zeros((len(RULES), nb, 2), dtype=np.int64)
    for ri, rule in enumerate(RULES):
        if not ran[ri]:
            continue
        for bi, b in enumerate(budgets):
            sr = sel[ri, bi]
            dsel, csel = sr[ev_reg[D]], sr[ev_reg[C]]
            mp = stratum_maps(onto, adm_F, adm_L, yD, dsel, N_cal, n_min)
            (g1, S1, _, nd1, nc1), (g0, S0, _, nd0, nc0) = mp[1], mp[0]
            m1c, m0c = m1_fit(onto, g1, S1, PnF_C[csel], yC[csel]), m1_fit(onto, g0, S0, PnL_C[~csel], yC[~csel])
            m1p, m0p = m1_fit(onto, g1, S1, PnF_C, yC), m1_fit(onto, g0, S0, PnL_C, yC)
            st = sr[test_r]
            sets_c = route(f"{rule}_cond", bi, st, m1c.predict(PnF_T), m0c.predict(PnL_T), g1, g0)
            route(f"{rule}_pooled", bi, st, m1p.predict(PnF_T), m0p.predict(PnL_T), g1, g0)
            for k_s, (S_, nd, nc, mc, mpo, cnt_cal) in enumerate(((S0, nd0, nc0, m0c, m0p, int((~csel).sum())),
                                                                 (S1, nd1, nc1, m1c, m1p, int(csel.sum())))):
                in_s = sr == bool(k_s)
                sinfo[ri, bi, k_s] = (int((in_s & (role == 0)).sum()), int((in_s & (role == 1)).sum()),
                                      int((in_s & (role == 2)).sum()), nd, nc, cnt_cal, int(S_.sum()),
                                      int(mc.infinite.sum()), int(mpo.infinite.sum()))
                sbits[ri, bi, k_s] = int(set_bits(S_[None, :])[0])
            if keep_cells and rule == "uncert" and abs(b - B_PRIMARY) < 1e-9:
                cells["final_uncert_cond"] = set_bits(sets_c)

    # -- TEST-side arms on the population maps with pooled thresholds ------------------------------------------------
    bern = np.array([bern_u(cohort, str(rid), s) for rid in np.asarray(R["region_id"])[test_r]])
    for bi, b in enumerate(budgets):
        k_rule = {RULES[ri]: int(sel[ri, bi][test_r].sum()) for ri in range(len(RULES)) if ran[ri]}
        for ri, rule in enumerate(RULES):
            if rule in k_rule:
                route(f"random_k_{rule}", bi, random_k(nTR, k_rule[rule], s, bi, ri), sFp, sLp, gFp, gLp)
        route("random_bern", bi, bern < b, sFp, sLp, gFp, gLp)
        route("density_k", bi, top_k(N_r[test_r], k_rule["uncert"], tie), sFp, sLp, gFp, gLp)
        for rule in ("uncert", "voi"):
            if rule in k_rule:
                route(f"oracle_k_{rule}", bi, top_k(gain, k_rule[rule], tie), sFp, sLp, gFp, gLp)
        route("ref_L", bi, np.zeros(nTR, dtype=bool), sFp, sLp, gFp, gLp)
        route("ref_F", bi, np.ones(nTR, dtype=bool), sFp, sLp, gFp, gLp)

    forced = np.full(nTR, np.nan)
    if P17L_ev is not None:
        forced = np.bincount(treg, weights=(np.argmax(P17L_ev[T], axis=1) != yT).astype(np.float64), minlength=nTR)
    if keep_cells:
        sdev_T = m_dev.predict(PnL_T)
        cells.update({"treg": treg, "y": yT, "CLdev": set_bits(sdev_T),
                      "uL": unresolved(sdev_T, onto, u_def).astype(np.int8),
                      "uL_singleton": unresolved(sdev_T, onto, "singleton").astype(np.int8),
                      "CL_pooled": set_bits(sLp), "CF_pooled": set_bits(sFp)})
        bp = int(np.argmin(np.abs(np.asarray(budgets) - B_PRIMARY)))
        for ri, rule in enumerate(RULES):
            cells[f"sel_{rule}"] = sel[ri, bp][ev_reg[T]].astype(np.int8)
    return {"rstat": rstat, "tstat": tstat, "selT": selT, "vran": vran, "taus": taus, "scores": scores, "sel": sel,
            "sinfo": sinfo, "sbits": sbits, "gain": gain, "forced_err_L": forced, "test_patients": tpat,
            "test_regions": test_r, "ran": ran, "info": info, "hashes": hashes, "cells": cells}
