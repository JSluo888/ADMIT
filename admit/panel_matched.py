"""ADMIT: panel-matched, projected-stratum conformal calibration over a cell-type hierarchy.

The deployment panel P is known before any target label exists. This module

1. builds the admissible label space A(P) = {c : R(c) subset P} over the 17-node
   ontology (``admit.ontology``; R(c) = required_positive | required_measured, isoform-aware CD45);
2. projects every reference label y to pi_P(y), the finest ancestor-or-self of y
   that lies in A(P) (-1 = unidentifiable);
3. masks reference (calibration) features to P with the same encoding the test
   cohort uses, rescoring with the frozen model;
4. calibrates one LAC tolerance per projected stratum c in A(P) and returns
   C(x) = {c in A(P) : s(x,c) <= lambda_c} (``ADMIT``). ``AdmitM1Frozen`` (ADMIT-M1, the primary method)
   first fixes the supported groups S_P from training labels only: strata whose expected calibration count
   is below n_min = ceil((1-alpha)/alpha) are merged into their nearest admissible ancestor
   (``frozen_support_map``); it then calibrates one tolerance per group and returns
   C(x) = {c in S_P : s(x,c) <= lambda_c}. ``AdmitM1`` (M1-cal) chooses the groups from the calibration
   labels instead and is kept as an ablation.
   There is no anti-empty backstop by default: an empty set is an abstention.

Everything is vectorised over cells; all set-valued outputs are (n, K) boolean
matrices over the K ontology nodes (the same 17 columns everywhere), so
every arm is scored by the same metric code (``set_metrics``).

Node scores. The scorer emits a probability for each *label* column (labels may be
internal nodes, e.g. ``T_Lineage``). The node probability is the mass of the node's
subtree: p(c|x) = sum_{d in subtree(c)} p_label(d|x). For a leaf this is its own
mass; ``aggregate`` implements it as one matrix product.

Naming: ``PanelMatchedPATCH`` is the original class name of ADMIT (``ADMIT`` is an alias);
``SourcePATCH`` is the full-panel-calibration comparator (thresholds fitted on the full panel,
then the A(P) filter).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ISOFORM_SATISFIERS = {"CD45": {"CD45", "CD45RA", "CD45RO"}}

# --------------------------------------------------------------------------
# Ontology, requirement map, admissibility and projection
# --------------------------------------------------------------------------


class Ontology:
    """Tree over K label/node columns with a requirement map R(c)."""

    def __init__(self, names: Sequence[str], parent: Dict[str, Optional[str]],
                 requirements: Dict[str, Iterable[str]],
                 legacy_requirements: Optional[Dict[str, Iterable[str]]] = None):
        self.names = [str(n) for n in names]
        self.K = len(self.names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.parent_idx = np.array(
            [self.idx.get(parent.get(n)) if parent.get(n) in self.idx else -1
             for n in self.names], dtype=np.int64)
        self.R = {n: frozenset(requirements.get(n, ())) for n in self.names}
        # reference requirement map (default: required_positive only), kept to count how many cells' pi_P the
        # required_measured correction (or a requirement override) changes; never used for prediction
        self.R_legacy = ({n: frozenset(legacy_requirements.get(n, ())) for n in self.names}
                         if legacy_requirements is not None else dict(self.R))
        # anc[c, d] = c is an ancestor-or-self of d
        anc = np.zeros((self.K, self.K), dtype=bool)
        for d in range(self.K):
            c = d
            while c >= 0:
                anc[c, d] = True
                c = self.parent_idx[c]
        self.anc = anc
        self.is_leaf = ~np.array([(self.parent_idx == c).any() for c in range(self.K)])
        self.depth = anc.sum(axis=0)  # 1 for top-level nodes

    @classmethod
    def from_repo(cls, target_types: Sequence[str],
                  req_override: Optional[Dict[str, Iterable[str]]] = None) -> "Ontology":
        """R(c) = required_positive | required_measured (a node defined by a marker's absence against a sibling
        also needs that marker measured; Helper_T_Cell needs FOXP3, Macrophage_CD163neg needs CD163).
        R_legacy = required_positive only.

        req_override (requirement-map sensitivity; None = no change): {node: [markers]} replaces R(c) of the named
        nodes. R_legacy then holds the default map, so pi_P_changed_by_R / t_pi_changed_R count cells changed by
        the override."""
        from admit.ontology import CELL_TYPE_ONTOLOGY as O
        parent, req, legacy = {}, {}, {}
        for t in target_types:
            node = O.get(str(t))
            parent[str(t)] = getattr(node, "parent", None) if node is not None else None
            pos = list(getattr(node, "required_positive", []) or []) if node else []
            meas = list(getattr(node, "required_measured", []) or []) if node else []
            req[str(t)] = pos + [m for m in meas if m not in pos]
            legacy[str(t)] = pos
        if req_override is not None:
            vocab = {m for n in O.values() for m in (n.required_positive + n.required_negative + n.required_measured
                                                     + n.optional_markers)}
            bad = [n for n, ms in req_override.items() if n not in req or not ms or not set(ms) <= vocab]
            if bad:  # an empty R(c) is always admissible; an unknown marker makes a node silently never admissible
                raise ValueError(f"requirement override: unknown node, empty list or unknown marker in {bad}")
            legacy = dict(req)
            req.update({n: list(ms) for n, ms in req_override.items()})
        return cls(target_types, parent, req, legacy)

    # ---- panel-dependent objects ----------------------------------------
    def admissible(self, panel: Iterable[str], legacy: bool = False) -> np.ndarray:
        ps = set(panel)
        R = self.R_legacy if legacy else self.R
        def ok(m: str) -> bool:
            return bool(ISOFORM_SATISFIERS.get(m, {m}) & ps)
        return np.array([all(ok(m) for m in R[n]) for n in self.names], dtype=bool)

    def projection_map(self, adm: np.ndarray) -> np.ndarray:
        """proj[d] = finest ancestor-or-self of d in A(P), or -1."""
        proj = np.full(self.K, -1, dtype=np.int64)
        for d in range(self.K):
            c = d
            while c >= 0 and not adm[c]:
                c = self.parent_idx[c]
            proj[d] = c
        return proj

    def project(self, y: np.ndarray, adm: np.ndarray) -> np.ndarray:
        return self.projection_map(adm)[np.asarray(y, dtype=np.int64)]

    def aggregate(self, P_label: np.ndarray) -> np.ndarray:
        """Node probability = subtree mass. (n,K) @ (K,K)^T."""
        return P_label @ self.anc.T.astype(P_label.dtype)

    def ancestor_lists(self) -> Dict[str, List[str]]:
        """leaf -> [self, parent, ..., 'root'] (format the comparator ports use)."""
        out = {}
        for d, n in enumerate(self.names):
            chain, c = [n], self.parent_idx[d]
            while c >= 0:
                chain.append(self.names[c]); c = self.parent_idx[c]
            chain.append("root")
            out[n] = chain
        return out


# --------------------------------------------------------------------------
# Feature masking (same encoding as at test)
# --------------------------------------------------------------------------


def mask_arc(X: np.ndarray, markers: Sequence[str], panel: Iterable[str]) -> np.ndarray:
    """X_arc encoding: an unmeasured marker is 0.0 (the z-score mean). No presence bits."""
    keep = np.array([m in set(panel) for m in markers], dtype=bool)
    Xm = np.array(X, dtype=np.float32, copy=True)
    Xm[:, ~keep] = 0.0
    return Xm


def mask_gmm160(X: np.ndarray, markers: Sequence[str], panel: Iterable[str],
                drop_nimbus: bool = False) -> np.ndarray:
    """X_gmm 160-D encoding [GMM | GMM_present | Nimbus | Nimbus_present].

    Unmeasured marker: GMM posterior 0.5, presence 0, Nimbus 0.5, Nimbus presence 0
    (encoding of the secondary v6 backbone). ``drop_nimbus`` blanks the Nimbus half for
    every marker, which is how external 40-D caches are expanded to 160-D.
    """
    m = len(markers)
    keep = np.array([mk in set(panel) for mk in markers], dtype=bool)
    Xm = np.array(X, dtype=np.float32, copy=True)
    Xm[:, :m][:, ~keep] = 0.5
    Xm[:, m:2 * m][:, ~keep] = 0.0
    nim = np.ones(m, dtype=bool) if drop_nimbus else ~keep
    Xm[:, 2 * m:3 * m][:, nim] = 0.5
    Xm[:, 3 * m:4 * m][:, nim] = 0.0
    return Xm


# --------------------------------------------------------------------------
# Conformal quantile
# --------------------------------------------------------------------------


def conformal_quantile(scores: np.ndarray, alpha: float, method: str = "order_stat") -> float:
    """Split-conformal tolerance at level ceil((n+1)(1-alpha))/n.

    ``order_stat``: the k-th smallest score, k = ceil((n+1)(1-alpha)); +inf if k > n.
    ``numpy_higher``: legacy repo convention ``np.quantile(s, min(k/n,1), 'higher')``
    (one order statistic more conservative when k < n; never +inf).
    """
    s = np.asarray(scores, dtype=np.float64)
    n = len(s)
    if n == 0:
        return float("inf")
    k = int(math.ceil((n + 1) * (1.0 - alpha) - 1e-12))
    if method == "numpy_higher":
        return float(np.quantile(s, min(k / n, 1.0), method="higher"))
    if k > n:
        return float("inf")
    return float(np.partition(s, k - 1)[k - 1])


# --------------------------------------------------------------------------
# ADMIT and the full-panel comparator
# --------------------------------------------------------------------------


def _backstop(sets: np.ndarray, S_node: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not valid.any():
        return sets
    m = np.where(valid[None, :], S_node, np.inf)
    j = m.argmin(axis=1)
    out = sets.copy()
    out[np.arange(len(j)), j] = True
    return out


@dataclass
class PanelMatchedPATCH:
    """ADMIT: calibration on reference cells rescored under P, strata pi_P(y).

    Every c in A(P) gets a tolerance (no minimum stratum size): with n_c calibration cells in
    the stratum and k = ceil((n_c+1)(1-alpha)), lambda_c is the k-th smallest score, or +inf
    when k > n_c (including n_c = 0), in which case c is always included. The returned set is
    the union of all passing nodes (no ancestor pruning). ``infinite`` marks +inf strata.
    """
    onto: Ontology
    alpha: float = 0.10
    quantile: str = "order_stat"
    lam: np.ndarray = field(default=None)
    n_c: np.ndarray = field(default=None)
    adm: np.ndarray = field(default=None)
    infinite: np.ndarray = field(default=None)

    def fit(self, Pnode_cal: np.ndarray, y_cal: np.ndarray, panel: Iterable[str]) -> "PanelMatchedPATCH":
        K = self.onto.K
        self.adm = self.onto.admissible(panel)
        strat = self.onto.project(y_cal, self.adm)
        self.lam = np.full(K, np.nan)
        self.n_c = np.zeros(K, dtype=np.int64)
        for c in np.where(self.adm)[0]:
            sel = strat == c
            self.n_c[c] = int(sel.sum())
            self.lam[c] = conformal_quantile(1.0 - Pnode_cal[sel, c], self.alpha, self.quantile)
        self.infinite = self.adm & ~np.isfinite(np.nan_to_num(self.lam, nan=0.0, posinf=np.inf))
        return self

    @property
    def finite(self) -> np.ndarray:
        return self.adm & ~self.infinite

    def predict(self, Pnode_test: np.ndarray, backstop: bool = False) -> np.ndarray:
        valid = self.adm
        S = 1.0 - Pnode_test
        lam = np.where(valid, self.lam, -np.inf)
        sets = (S <= lam[None, :]) & valid[None, :]
        return _backstop(sets, S, valid) if backstop else sets

    def strata_report(self) -> Dict:
        return {self.onto.names[c]: {"n_c": int(self.n_c[c]),
                                     "lambda": (float(self.lam[c]) if np.isfinite(self.lam[c]) else "inf")}
                for c in np.where(self.adm)[0]}


@dataclass
class SourcePATCH:
    """Full-panel calibration (earlier construction): source tolerances fit on reference labels y
    with the full panel, then the A(P) filter.

    In the 17-node space: aggregated node scores, n_min support, filter after thresholding, optional
    unconditional argmax backstop over S cap A(P).
    """
    onto: Ontology
    alpha: float = 0.10
    n_min: int = 10
    quantile: str = "order_stat"
    lam: np.ndarray = field(default=None)
    n_c: np.ndarray = field(default=None)
    supported: np.ndarray = field(default=None)

    def fit(self, Pnode_cal_full: np.ndarray, y_cal: np.ndarray) -> "SourcePATCH":
        K = self.onto.K
        self.lam = np.full(K, np.nan)
        self.n_c = np.bincount(np.asarray(y_cal, dtype=np.int64), minlength=K)
        self.supported = self.n_c >= self.n_min
        for c in np.where(self.supported)[0]:
            self.lam[c] = conformal_quantile(1.0 - Pnode_cal_full[y_cal == c, c], self.alpha, self.quantile)
        return self

    def predict(self, Pnode_test: np.ndarray, panel: Iterable[str], backstop: bool = True) -> np.ndarray:
        adm = self.onto.admissible(panel)
        valid = adm & self.supported
        S = 1.0 - Pnode_test
        lam = np.where(valid, self.lam, -np.inf)
        sets = (S <= lam[None, :]) & valid[None, :]
        return _backstop(sets, S, valid) if backstop else sets


# --------------------------------------------------------------------------
# Flat comparators (label space, vectorised; deterministic like the b5 benchmark)
# --------------------------------------------------------------------------


def lac_flat(P_cal, y_cal, P_test, alpha, quantile="order_stat", backstop=True):
    """Global LAC split-CP (argmax backstop)."""
    q = conformal_quantile(1.0 - P_cal[np.arange(len(y_cal)), y_cal], alpha, quantile)
    sets = (1.0 - P_test) <= q
    if backstop:
        empty = ~sets.any(axis=1)
        sets[empty, P_test[empty].argmax(axis=1)] = True
    return sets


def _aps_scores_all(P: np.ndarray, k_reg: Optional[int] = None, lam: float = 0.0) -> np.ndarray:
    """APS/RAPS score of every label for every row (deterministic, argsort tie order)."""
    order = np.argsort(-P, axis=1, kind="quicksort")
    sorted_p = np.take_along_axis(P, order, axis=1)
    cums = np.cumsum(sorted_p, axis=1)
    if k_reg is not None:
        ranks1 = np.arange(1, P.shape[1] + 1)[None, :]
        cums = cums + lam * np.maximum(ranks1 - k_reg, 0)
    out = np.empty_like(cums)
    np.put_along_axis(out, order, cums, axis=1)
    return out


def aps_flat(P_cal, y_cal, P_test, alpha, k_reg=None, lam=0.0, quantile="order_stat"):
    """APS (k_reg=None) or RAPS; argmax backstop."""
    s_cal = _aps_scores_all(P_cal, k_reg, lam)[np.arange(len(y_cal)), y_cal]
    q = conformal_quantile(s_cal, alpha, quantile)
    sets = _aps_scores_all(P_test, k_reg, lam) <= q
    empty = ~sets.any(axis=1)
    sets[empty, P_test[empty].argmax(axis=1)] = True
    return sets


def lac_node_projected(Pnode_cal, y_cal, Pnode_test, onto: Ontology, panel, alpha,
                       quantile="order_stat"):
    """Ablation: ONE marginal LAC tolerance over A(P) node scores, labels pi_P(y).

    Same projection and panel-matched calibration as PanelMatchedPATCH but not Mondrian.
    No backstop. Cells with pi_P(y) = -1 are excluded from calibration.
    """
    adm = onto.admissible(panel)
    strat = onto.project(y_cal, adm)
    ok = strat >= 0
    q = conformal_quantile(1.0 - Pnode_cal[np.where(ok)[0], strat[ok]], alpha, quantile)
    return ((1.0 - Pnode_test) <= q) & adm[None, :]


# --------------------------------------------------------------------------
# Hierarchical comparators (vectorised ports of HCC, scConform and CRSVP)
# --------------------------------------------------------------------------


def _port_quantile(scores, alpha, n_min, quantile):
    if len(scores) < n_min:
        return float("inf")
    return conformal_quantile(scores, alpha, quantile)


def hcc_vec(P_cal, y_cal, P_test, onto: Ontology, alpha, quantile="order_stat", n_min=10):
    """Vectorised ``HCCPort``: per-node thresholds on subtree mass over cells whose label
    lies in the subtree; ascend from the top-1 label to the first passing ancestor;
    return subtree(chosen) as label columns (root = all columns)."""
    K = onto.K
    A = onto.anc
    mass_cal = P_cal @ A.T
    thr = np.full(K, np.inf)
    for v in range(K):
        sel = A[v, y_cal]
        thr[v] = _port_quantile(1.0 - mass_cal[sel, v], alpha, n_min, quantile)
    mass = P_test @ A.T
    top = P_test.argmax(axis=1)
    n = len(top)
    chosen = np.full(n, -1, dtype=np.int64)  # -1 = root
    cur = top.copy()
    undecided = np.ones(n, dtype=bool)
    while undecided.any():
        idx = np.where(undecided & (cur >= 0))[0]
        if len(idx) == 0:
            break
        ok = (1.0 - mass[idx, cur[idx]]) <= thr[cur[idx]]
        chosen[idx[ok]] = cur[idx[ok]]
        undecided[idx[ok]] = False
        rest = idx[~ok]
        cur[rest] = onto.parent_idx[cur[rest]]
    sets = np.ones((n, K), dtype=bool)
    has = chosen >= 0
    sets[has] = A[chosen[has]]
    return sets


def scconform_vec(P_cal, y_cal, P_test, onto: Ontology, alpha, quantile="order_stat", n_min=10):
    """Vectorised ``scConformPort`` (laplacian_alpha=0): class-conditional LAC thresholds
    on label columns; closure to subtree(LCA of members); top-1 if empty."""
    K = onto.K
    A = onto.anc
    s_cal = 1.0 - P_cal[np.arange(len(y_cal)), y_cal]
    thr = np.full(K, np.inf)
    for c in range(K):
        thr[c] = _port_quantile(s_cal[y_cal == c], alpha, n_min, quantile)
    init = (1.0 - P_test) <= thr[None, :]
    n = len(P_test)
    empty = ~init.any(axis=1)
    init[empty, P_test[empty].argmax(axis=1)] = True
    # LCA = deepest node that is an ancestor-or-self of every member
    # common[i, v] = all members d of row i satisfy A[v, d]
    viol = init.astype(np.int32) @ (~A).T.astype(np.int32)  # (n, K): #members not under v
    common = viol == 0
    depth = np.where(common, onto.depth[None, :], 0)
    lca = depth.argmax(axis=1)
    has = common.any(axis=1)
    sets = np.ones((n, K), dtype=bool)
    sets[has] = A[lca[has]]
    sets[empty] = init[empty]  # port: backstop row is top-1 only, no closure
    return sets


def crsvp_vec(P_cal, y_cal, P_test, onto: Ontology, alpha, quantile="order_stat", n_min=10):
    """Vectorised ``CRSVPPort`` (r=1): global LAC threshold; smallest single node whose
    subtree contains the raw set (argmax backstop); root = all columns."""
    K = onto.K
    A = onto.anc
    tau = _port_quantile(1.0 - P_cal[np.arange(len(y_cal)), y_cal], alpha, n_min, quantile)
    raw = (1.0 - P_test) <= tau
    empty = ~raw.any(axis=1)
    raw[empty, P_test[empty].argmax(axis=1)] = True
    viol = raw.astype(np.int32) @ (~A).T.astype(np.int32)
    contains = viol == 0
    size = A.sum(axis=1)
    sz = np.where(contains, size[None, :], 10 ** 6)
    best = sz.argmin(axis=1)
    has = contains.any(axis=1)
    sets = np.ones((len(P_test), K), dtype=bool)
    sets[has] = A[best[has]]
    return sets


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def wilson(k: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(centre - half), float(centre + half))


def _rate(mask: np.ndarray, groups: Optional[np.ndarray] = None,
          bootstrap: Optional[int] = None, seed: int = 0) -> Dict:
    """Mean of a boolean vector with a cell-level Wilson interval.

    Hook: when ``groups`` (e.g. per-cell patient ids) is given and ``bootstrap`` > 0, a
    cluster (patient) bootstrap percentile interval is added under ``ci_cluster``.
    """
    n = int(mask.size)
    k = float(mask.sum())
    out = {"value": (k / n) if n else float("nan"), "n": n,
           "ci_wilson_cell": list(wilson(k, n)), "ci_note": "cell-level, not patient-clustered"}
    if groups is not None and bootstrap:
        rng = np.random.default_rng(seed)
        g_ids, inv = np.unique(groups, return_inverse=True)
        sums = np.bincount(inv, weights=mask.astype(float), minlength=len(g_ids))
        cnts = np.bincount(inv, minlength=len(g_ids)).astype(float)
        pick = rng.integers(0, len(g_ids), size=(bootstrap, len(g_ids)))
        vals = sums[pick].sum(axis=1) / np.maximum(cnts[pick].sum(axis=1), 1)
        out["ci_cluster"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        out["n_clusters"] = int(len(g_ids))
        out["per_cluster"] = {str(g): float(sums[i] / cnts[i]) for i, g in enumerate(g_ids)}
        out["ci_note"] = "cluster (patient/sample) bootstrap percentile interval in ci_cluster; Wilson is cell-level"
    return out


def set_metrics(sets: np.ndarray, y: np.ndarray, onto: Ontology, panel: Iterable[str],
                groups: Optional[np.ndarray] = None, per_stratum: bool = True,
                supported: Optional[np.ndarray] = None, per_label: bool = False,
                bootstrap: int = 0, seed: int = 0, target: Optional[np.ndarray] = None) -> Dict:
    """All evaluation metrics for one arm on one panel. ``sets`` is (n, K) bool over nodes.

    Patient hook: pass ``groups`` (per-cell sample / patient id, e.g. the ``sample`` array of
    the crc_*_identity.npz sidecars) and ``bootstrap`` > 0 to add cluster-bootstrap percentile
    intervals (``ci_cluster``) to every coverage rate and to each stratum's exact coverage.
    Without them every interval is a cell-level Wilson interval (not patient-clustered).
    """
    B = bootstrap
    G = (lambda m: None if groups is None else np.asarray(groups)[m])
    y = np.asarray(y, dtype=np.int64)
    n, K = sets.shape
    adm = onto.admissible(panel)
    # ``target`` overrides pi_P(y) (ADMIT+M1 evaluates its coarsened groups g(y))
    proj = np.asarray(target, dtype=np.int64) if target is not None else onto.project(y, adm)
    ident = proj >= 0
    rows = np.arange(n)
    size = sets.sum(axis=1)
    exact_hit = np.zeros(n, dtype=bool)
    exact_hit[ident] = sets[rows[ident], proj[ident]]
    leaf_in_A = adm[y]
    leaf_hit = sets[rows, y]
    anc_hit = (sets & onto.anc[:, y].T).any(axis=1)
    singleton = size == 1
    single_node = sets.argmax(axis=1)
    leaf_resolved = singleton & onto.is_leaf[single_node]
    # chain: all members on one root path (pairwise comparable); deepest member is a leaf
    deepest = np.where(sets, onto.depth[None, :], -1).argmax(axis=1)
    chain = (sets & ~onto.anc[:, deepest].T).sum(axis=1) == 0
    chain_leaf = chain & (size > 0) & onto.is_leaf[deepest]
    bad = sets & ~adm[None, :]
    total = float(size.sum())
    out = {
        "n_cells": int(n),
        "frac_identifiable": float(ident.mean()),
        "exact_coverage": _rate(exact_hit[ident], G(ident), B, seed),
        "exact_coverage_all_cells": _rate(exact_hit, groups, B, seed),
        # success = identifiable and pi_P(y) in C, or unidentifiable and C empty (abstained)
        "exact_or_correct_abstention": _rate(np.where(ident, exact_hit, size == 0), groups, B, seed),
        "leaf_coverage_y_in_A": _rate(leaf_hit[leaf_in_A], G(leaf_in_A), B, seed),
        "ancestor_coverage": _rate(anc_hit, groups, B, seed),
        "abstention": _rate(size == 0, groups, B, seed),
        "mean_set_size": float(size.mean()),
        "singleton_rate": float(singleton.mean()),
        "leaf_resolved_rate": float(leaf_resolved.mean()),
        "chain_to_leaf_rate": float(chain_leaf.mean()),
        "abstention_rate": float((size == 0).mean()),
        "abstention_rate_identifiable": float((size[ident] == 0).mean()) if ident.any() else float("nan"),
        "abstention_rate_unidentifiable": float((size[~ident] == 0).mean()) if (~ident).any() else float("nan"),
        "inadmissible_set_mass": float(bad.sum() / total) if total else 0.0,
        "frac_cells_with_inadmissible": float(bad.any(axis=1).mean()),
    }
    out["n_unidentifiable"] = int((~ident).sum())
    out["n_identifiable"] = int(ident.sum())
    if supported is not None:
        # ``supported`` = strata with a finite tolerance (ADMIT) or n_c >= n_min (full-panel calibration)
        sup_cells = ident & supported[np.where(proj >= 0, proj, 0)]
        out["exact_coverage_supported_strata"] = _rate(exact_hit[sup_cells])
        out["frac_cells_in_supported_strata"] = float(sup_cells.mean())
        out["n_test_cells_in_unsupported_strata"] = int((ident & ~sup_cells).sum())
    if per_label:
        pl = {}
        for d in np.unique(y):
            sel = y == d
            pl[onto.names[d]] = {
                "n": int(sel.sum()),
                "pi_P": onto.names[proj[sel][0]] if proj[sel][0] >= 0 else None,
                "exact_coverage": float(exact_hit[sel].mean()) if proj[sel][0] >= 0 else None,
                "ancestor_coverage": float(anc_hit[sel].mean()),
                "mean_set_size": float(size[sel].mean()),
                "singleton_rate": float(singleton[sel].mean()),
                "leaf_resolved_rate": float(leaf_resolved[sel].mean()),
                "abstention_rate": float((size[sel] == 0).mean())}
        out["per_label"] = pl
    if per_stratum:
        ps = {}
        for c in np.unique(proj[ident]):
            sel = proj == c
            k = int(exact_hit[sel].sum()); m = int(sel.sum())
            ps[onto.names[c]] = {"n": m, "exact_coverage": k / m, "ci_wilson_cell": list(wilson(k, m)),
                                 "mean_set_size": float(size[sel].mean()),
                                 "abstention_rate": float((size[sel] == 0).mean())}
            if groups is not None and B:
                r = _rate(exact_hit[sel], G(sel), B, seed)
                ps[onto.names[c]]["ci_cluster"] = r.get("ci_cluster")
                ps[onto.names[c]]["n_clusters"] = r.get("n_clusters")
        if (~ident).any():
            ps["<unidentifiable>"] = {"n": int((~ident).sum()),
                                     "abstention_rate": float((size[~ident] == 0).mean()),
                                     "mean_set_size": float(size[~ident].mean())}
        out["per_stratum"] = ps
    return out


def headline(m: Dict) -> Dict:
    """Flat summary of a ``set_metrics`` dict (for tables)."""
    return {
        "exact_cov": m["exact_coverage"]["value"],
        "exact_cov_ci": m["exact_coverage"]["ci_wilson_cell"],
        "exact_cov_ci_cluster": m["exact_coverage"].get("ci_cluster"),
        "leaf_cov": m["leaf_coverage_y_in_A"]["value"],
        "anc_cov": m["ancestor_coverage"]["value"],
        "size": m["mean_set_size"],
        "singleton": m["singleton_rate"],
        "leaf_resolved": m["leaf_resolved_rate"],
        "chain_leaf": m["chain_to_leaf_rate"],
        "abstain": m["abstention_rate"],
        "inadm_mass": m["inadmissible_set_mass"],
        "frac_ident": m["frac_identifiable"],
        "exact_cov_supported": m.get("exact_coverage_supported_strata", {}).get("value"),
        "exact_or_abstain": m["exact_or_correct_abstention"]["value"],
        "n_unident": m["n_unidentifiable"],
        "n_test_unsupported": m.get("n_test_cells_in_unsupported_strata"),
        "abstain_unident": m["abstention_rate_unidentifiable"],
    }


# --------------------------------------------------------------------------
# Patient-level tools (Schurch task 5b): equal-patient metrics, equal-patient CRC
# --------------------------------------------------------------------------


def crc_threshold(scores: np.ndarray, alpha: float, groups: np.ndarray) -> float:
    """Equal-patient conformal risk control for exact-label miscoverage (pilot/calibration.py).

    With G patients: choose the smallest t with sum_j L_j(t) + 1 <= (G + 1) alpha, L_j the
    patient's mean miscoverage. Infeasible ((G+1)alpha < 1) or empty -> +inf.
    """
    s = np.asarray(scores, dtype=float)
    if len(s) == 0:
        return float("inf")
    _, inverse, counts = np.unique(np.asarray(groups), return_inverse=True, return_counts=True)
    g = len(counts)
    allowed = ((g + 1) * alpha - 1) / g
    if allowed < 0:
        return float("inf")
    order = np.argsort(s, kind="stable")
    weights = 1.0 / (g * counts[inverse])
    cumulative = np.cumsum(weights[order])
    idx = min(int(np.searchsorted(cumulative, 1 - allowed)), len(s) - 1)
    return float(s[order[idx]])


def patch_pm_crc(Pnode_cal, y_cal, g_cal, Pnode_test, onto: Ontology, panel, alpha):
    """Projected-stratum ADMIT with an equal-patient CRC tolerance per stratum (secondary arm)."""
    adm = onto.admissible(panel)
    strat = onto.project(y_cal, adm)
    lam = np.full(onto.K, -np.inf)
    for c in np.where(adm)[0]:
        sel = strat == c
        lam[c] = crc_threshold(1.0 - Pnode_cal[sel, c], alpha, g_cal[sel]) if sel.any() else np.inf
    return ((1.0 - Pnode_test) <= lam[None, :]) & adm[None, :], lam


def equal_patient(sets: np.ndarray, y: np.ndarray, onto: Ontology, panel, groups: np.ndarray,
                  bootstrap: int = 2000, seed: int = 0, target: Optional[np.ndarray] = None) -> Dict:
    """Equal-patient averages (each test patient weighs 1/G) with patient-bootstrap 95% intervals."""
    adm = onto.admissible(panel)
    # ``target`` overrides pi_P(y) (ADMIT+M1 evaluates its coarsened groups g(y))
    proj = np.asarray(target, dtype=np.int64) if target is not None else onto.project(y, adm)
    ident = proj >= 0
    n = len(y)
    hit = np.zeros(n, dtype=bool)
    hit[ident] = sets[np.where(ident)[0], proj[ident]]
    size = sets.sum(axis=1)
    pats = np.unique(groups)
    rows = {}
    for p in pats:
        m = groups == p
        mi = m & ident
        rows[str(p)] = {"n": int(m.sum()), "n_ident": int(mi.sum()),
                        "exact_cov": float(hit[mi].mean()) if mi.any() else float("nan"),
                        "size": float(size[m].mean()), "singleton": float((size[m] == 1).mean()),
                        "abstain": float((size[m] == 0).mean())}
    rng = np.random.default_rng(seed)
    out = {"n_patients": int(len(pats)), "per_patient": rows}
    for k in ("exact_cov", "size", "singleton", "abstain"):
        v = np.array([rows[str(p)][k] for p in pats], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            out[k] = {"value": None, "ci_patient_bootstrap": None}
            continue
        b = v[rng.integers(0, len(v), size=(bootstrap, len(v)))].mean(axis=1)
        out[k] = {"value": float(v.mean()), "ci_patient_bootstrap": [float(np.percentile(b, 2.5)),
                                                                     float(np.percentile(b, 97.5))]}
    return out


# --------------------------------------------------------------------------
# ADMIT + M1: supported-ancestor projection
# --------------------------------------------------------------------------
# ADMIT is the current name of PanelMatchedPATCH.
ADMIT = PanelMatchedPATCH


def nmin_rule(alpha: float) -> int:
    """Smallest n with a finite split-conformal quantile: ceil((n+1)(1-alpha)) <= n  <=>  n >= (1-alpha)/alpha."""
    return int(math.ceil((1.0 - alpha) / alpha - 1e-9))


def supported_group_map(onto: Ontology, adm: np.ndarray, y_cal: np.ndarray, n_min: int):
    """M1 projection. Start from pi_P(y) (finest admissible ancestor-or-self); visit admissible nodes from the
    deepest level up; a node holding 0 < n < n_min calibration cells passes them to its nearest admissible
    ancestor. S_P = nodes left with >= n_min cells. Every label maps to the first node of S_P met walking up
    from pi_P(label) (inclusive), or -1 (unidentifiable under M1).
    Returns (gmap[label] -> group or -1, supported mask S_P, final counts per node)."""
    proj = onto.projection_map(adm)
    lab = np.asarray(y_cal, dtype=np.int64)
    base = proj[lab]
    counts = np.bincount(base[base >= 0], minlength=onto.K).astype(np.int64)
    order = sorted(np.where(adm)[0], key=lambda c: -onto.depth[c])
    for c in order:
        if 0 < counts[c] < n_min:
            a = onto.parent_idx[c]
            while a >= 0 and not adm[a]:
                a = onto.parent_idx[a]
            if a >= 0:
                counts[a] += counts[c]
            counts[c] = 0
    S = adm & (counts >= n_min)
    gmap = np.full(onto.K, -1, dtype=np.int64)
    for d in range(onto.K):
        c = proj[d]
        while c >= 0 and not S[c]:
            c = onto.parent_idx[c]
        gmap[d] = c if (c >= 0 and S[c]) else -1
    return gmap, S, counts


@dataclass
class AdmitM1:
    """ADMIT + M1 with groups chosen from calibration labels (**M1-cal**, ablation only).

    Strata are the supported groups S_P (n >= n_min after supported-ancestor projection), where S_P is chosen
    from the calibration labels; each gets the finite k-th order statistic tolerance on the same calibration
    cells; C(x) = {g in S_P : s(x,g) <= lambda_g}. No +inf strata, no backstop. Because the groups depend on
    the calibration labels, the fixed-group Mondrian argument does not cover this construction; use
    ``AdmitM1Frozen`` (support map frozen from training labels) for the guarantee."""
    onto: Ontology
    alpha: float = 0.10
    n_min: Optional[int] = None          # None -> nmin_rule(alpha)
    quantile: str = "order_stat"
    lam: np.ndarray = field(default=None)
    gmap: np.ndarray = field(default=None)
    S: np.ndarray = field(default=None)
    n_c: np.ndarray = field(default=None)
    adm: np.ndarray = field(default=None)

    def fit(self, Pnode_cal, y_cal, panel) -> "AdmitM1":
        nm = self.n_min if self.n_min is not None else nmin_rule(self.alpha)
        self.n_min_used = nm
        self.adm = self.onto.admissible(panel)
        self.gmap, self.S, _ = supported_group_map(self.onto, self.adm, y_cal, nm)
        g = self.gmap[np.asarray(y_cal, dtype=np.int64)]
        self.lam = np.full(self.onto.K, -np.inf)
        self.n_c = np.zeros(self.onto.K, dtype=np.int64)
        for c in np.where(self.S)[0]:
            sel = g == c
            self.n_c[c] = int(sel.sum())
            self.lam[c] = conformal_quantile(1.0 - Pnode_cal[sel, c], self.alpha, self.quantile)
        return self

    def target(self, y) -> np.ndarray:
        return self.gmap[np.asarray(y, dtype=np.int64)]

    def predict(self, Pnode_test) -> np.ndarray:
        return ((1.0 - Pnode_test) <= self.lam[None, :]) & self.S[None, :]

    def strata_report(self) -> Dict:
        return {self.onto.names[c]: {"n_c": int(self.n_c[c]), "lambda": float(self.lam[c])}
                for c in np.where(self.S)[0]}


def admit_m1_crc(Pnode_cal, y_cal, g_cal, Pnode_test, onto: Ontology, panel, alpha, n_min=None):
    """M1 groups with an equal-patient CRC tolerance per group (secondary arm). Infeasible -> +inf."""
    nm = n_min if n_min is not None else nmin_rule(alpha)
    adm = onto.admissible(panel)
    gmap, S, _ = supported_group_map(onto, adm, y_cal, nm)
    g = gmap[np.asarray(y_cal, dtype=np.int64)]
    lam = np.full(onto.K, -np.inf)
    for c in np.where(S)[0]:
        sel = g == c
        lam[c] = crc_threshold(1.0 - Pnode_cal[sel, c], alpha, g_cal[sel])
    return ((1.0 - Pnode_test) <= lam[None, :]) & S[None, :], gmap


# --------------------------------------------------------------------------
# ADMIT-M1: frozen support map (training labels only) + exact calibration
# --------------------------------------------------------------------------


def restrict_to_vocab(onto: Ontology, y: np.ndarray, vocab: Optional[Iterable[int]]) -> np.ndarray:
    """Replace every label by its finest ancestor-or-self in ``vocab`` (node indices); labels without one are
    dropped. ``vocab`` None -> labels unchanged. Used for cross-cohort maps: the target annotation scheme can only
    express labels in its vocabulary."""
    y = np.asarray(y, dtype=np.int64)
    if vocab is None:
        return y
    V = np.zeros(onto.K, dtype=bool)
    V[np.asarray(list(vocab), dtype=np.int64)] = True
    to = np.full(onto.K, -1, dtype=np.int64)
    for d in range(onto.K):
        c = d
        while c >= 0 and not V[c]:
            c = onto.parent_idx[c]
        to[d] = c
    r = to[y]
    return r[r >= 0]


def frozen_support_map(onto: Ontology, adm: np.ndarray, y_train: np.ndarray, n_cal: int, n_min: int,
                       vocab: Optional[Iterable[int]] = None):
    """Supported-ancestor map frozen before calibration, from training labels only.

    pi_P(y_train) counts give the expected calibration count e_c = n_train_c * n_cal / N_train (N_train = all
    training labels after the optional vocabulary restriction, n_cal = all calibration cells). Admissible nodes are
    visited deepest first; a node with 0 < e_c < n_min passes e_c to its nearest admissible ancestor.
    S_P = {c in A(P): e_c >= n_min}; g_P(d) = first node of S_P on the path from pi_P(d) upward, else -1
    ("no supported target").
    Returns (gmap (K,) label -> group or -1, S (K,) bool, expected counts (K,) after merging)."""
    proj = onto.projection_map(adm)
    y = restrict_to_vocab(onto, y_train, vocab)
    n_tr = len(y)
    e = np.zeros(onto.K, dtype=np.float64)
    if n_tr:
        base = proj[y]
        e = np.bincount(base[base >= 0], minlength=onto.K).astype(np.float64) * (float(n_cal) / n_tr)
    for c in sorted(np.where(adm)[0], key=lambda c: -onto.depth[c]):
        if 0 < e[c] < n_min:
            a = onto.parent_idx[c]
            while a >= 0 and not adm[a]:
                a = onto.parent_idx[a]
            if a >= 0:
                e[a] += e[c]
            e[c] = 0.0
    S = adm & (e >= n_min)
    gmap = np.full(onto.K, -1, dtype=np.int64)
    for d in range(onto.K):
        c = proj[d]
        while c >= 0 and not S[c]:
            c = onto.parent_idx[c]
        gmap[d] = c if (c >= 0 and S[c]) else -1
    return gmap, S, e


@dataclass
class AdmitM1Frozen:
    """ADMIT-M1 (primary method). Groups S_P and the target map g_P are given (frozen from training labels by
    ``frozen_support_map``); calibration groups cells by g_P(y); lambda_c = k-th smallest score, k = ceil((n_c+1)
    (1-alpha)), +inf when k > n_c (including n_c = 0: c then enters every set). C(x) = {c in S_P: s(x,c) <=
    lambda_c}; no backstop. Calibration cells with g_P(y) = -1 are not used. Proposition 1 applies to every
    c in S_P because S_P and g_P do not depend on the calibration cells."""
    onto: Ontology
    alpha: float
    gmap: np.ndarray
    S: np.ndarray
    quantile: str = "order_stat"
    lam: np.ndarray = field(default=None)
    n_c: np.ndarray = field(default=None)
    infinite: np.ndarray = field(default=None)

    def fit(self, Pnode_cal: np.ndarray, y_cal: np.ndarray) -> "AdmitM1Frozen":
        g = self.gmap[np.asarray(y_cal, dtype=np.int64)]
        self.lam = np.full(self.onto.K, -np.inf)
        self.n_c = np.zeros(self.onto.K, dtype=np.int64)
        for c in np.where(self.S)[0]:
            sel = g == c
            self.n_c[c] = int(sel.sum())
            self.lam[c] = conformal_quantile(1.0 - Pnode_cal[sel, c], self.alpha, self.quantile)
        self.infinite = self.S & np.isposinf(self.lam)
        return self

    def target(self, y) -> np.ndarray:
        return self.gmap[np.asarray(y, dtype=np.int64)]

    def predict(self, Pnode_test: np.ndarray) -> np.ndarray:
        return ((1.0 - Pnode_test) <= self.lam[None, :]) & self.S[None, :]


def lac_node_target(Pnode_cal, y_cal, Pnode_test, tmap: np.ndarray, candidates: np.ndarray, alpha,
                    quantile="order_stat"):
    """Marginal LAC on a fixed target map: ONE tolerance from the calibration scores 1 - p(tmap[y]|x) over the
    cells with tmap[y] >= 0; C(x) = {c in candidates: 1 - p(c|x) <= q}. With tmap = g_P and candidates = S_P
    this is the target-matched comparator of ADMIT-M1 (same target, candidate nodes, cells and denominator)."""
    t = tmap[np.asarray(y_cal, dtype=np.int64)]
    ok = t >= 0
    q = conformal_quantile(1.0 - Pnode_cal[np.where(ok)[0], t[ok]], alpha, quantile)
    return ((1.0 - Pnode_test) <= q) & candidates[None, :]


def admit_m1_frozen_crc(Pnode_cal, y_cal, g_cal, Pnode_test, gmap, S, alpha):
    """Frozen M1 groups with an equal-patient CRC tolerance per group (secondary arm). Empty or infeasible -> +inf."""
    g = gmap[np.asarray(y_cal, dtype=np.int64)]
    lam = np.full(len(S), -np.inf)
    for c in np.where(S)[0]:
        sel = g == c
        lam[c] = crc_threshold(1.0 - Pnode_cal[sel, c], alpha, g_cal[sel]) if sel.any() else np.inf
    return ((1.0 - Pnode_test) <= lam[None, :]) & S[None, :], lam


def finest_depth(sets: np.ndarray, onto: Ontology) -> np.ndarray:
    """Depth of the deepest node in each set (0 for an empty set). Depth 1 = Immune/Epithelial/Endothelial/Stromal."""
    return np.where(sets, onto.depth[None, :], 0).max(axis=1)


def chain_to_leaf(sets: np.ndarray, onto: Ontology) -> np.ndarray:
    """Leaf-resolved: the set is non-empty, all members lie on one root path, and its deepest member is a leaf."""
    size = sets.sum(axis=1)
    deepest = np.where(sets, onto.depth[None, :], -1).argmax(axis=1)
    chain = (sets & ~onto.anc[:, deepest].T).sum(axis=1) == 0
    return chain & (size > 0) & onto.is_leaf[deepest]


def resolved_leaf(sets: np.ndarray, onto: Ontology) -> np.ndarray:
    """Leaf the set resolves to (deepest member of a chain ending in a leaf), else -1."""
    deepest = np.where(sets, onto.depth[None, :], -1).argmax(axis=1)
    return np.where(chain_to_leaf(sets, onto), deepest, -1)
