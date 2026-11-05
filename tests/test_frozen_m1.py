"""Synthetic check of ADMIT-M1 (frozen support map) versus M1-cal (groups chosen from the calibration labels).

Hierarchy: lineage L with children A (common, easy) and B (rare, hard for L); cells labelled L directly (common,
easy); lineage M with child C. Calibration n = 800 cells per repetition: B's expected count (9.6) sits at
n_min = 9 (alpha 0.10), so whether B is its own group or merges into L is a coin flip.
- frozen map: chosen from 2,000 training labels, then calibrated on independent cells (+inf when k > n_c);
- M1-cal: chosen from the calibration labels themselves.
Group-conditional coverage is measured on 20,000 fresh test cells per repetition and averaged over repetitions,
conditionally on the group existing. The frozen map must reach >= 0.90 (up to Monte Carlo error) for every group;
for M1-cal the numbers are reported.

  python tests/test_frozen_m1.py          (or pytest tests/test_frozen_m1.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from admit.panel_matched import (AdmitM1, AdmitM1Frozen, Ontology, frozen_support_map,  # noqa: E402
                                 lac_node_target, nmin_rule, restrict_to_vocab)

NAMES = ["L", "A", "B", "M", "C"]
PARENT = {"A": "L", "B": "L", "C": "M"}
PROBS = np.array([0.15, 0.55, 0.012, 0.0, 0.288])     # label frequencies (L labelled directly, no M-labelled cells)
ALPHA = 0.10


def onto():
    return Ontology(NAMES, PARENT, {})


def sample(rng, n):
    y = rng.choice(len(NAMES), size=n, p=PROBS / PROBS.sum())
    pL = np.empty(n)
    frac = np.empty(n)       # share of the L mass on A (vs B)
    for lab, (a, b), (fa, fb) in ((0, (10, 1), (5, 5)), (1, (9, 1), (8, 2)), (2, (1, 4), (3, 3)), (4, (1, 9), (5, 5))):
        s = y == lab
        pL[s] = rng.beta(a, b, s.sum())
        frac[s] = rng.beta(fa, fb, s.sum())
    P = np.zeros((n, len(NAMES)))
    P[:, 1] = pL * frac
    P[:, 2] = pL * (1 - frac)
    P[:, 4] = 1 - pL
    return P, y


def group_cov(sets, tgt):
    out = {}
    for c in np.unique(tgt[tgt >= 0]):
        s = tgt == c
        out[int(c)] = float(sets[np.where(s)[0], c].mean())
    return out


def simulate(reps=1000, n_train=2000, n_cal=800, n_test=20000, seed=0):
    O = onto()
    adm = np.ones(O.K, dtype=bool)
    rng = np.random.default_rng(seed)
    nmin = nmin_rule(ALPHA)
    res = {"frozen": {}, "m1cal": {}, "lac_gp": {}}
    inf_groups = 0
    n_groups = 0
    for _ in range(reps):
        _, ytr = sample(rng, n_train)
        Pc, yc = sample(rng, n_cal)
        Pt, yt = sample(rng, n_test)
        Nc, Nt = O.aggregate(Pc), O.aggregate(Pt)
        gmap, S, _ = frozen_support_map(O, adm, ytr, n_cal, nmin)
        fz = AdmitM1Frozen(O, ALPHA, gmap, S).fit(Nc, yc)
        n_groups += int(S.sum())
        inf_groups += int(fz.infinite.sum())
        key = lambda m, c: f"{NAMES[c]}{'+B' if (NAMES[c] == 'L' and m[2] == 0) else ''}"  # noqa: E731
        for c, v in group_cov(fz.predict(Nt), fz.target(yt)).items():
            res["frozen"].setdefault(key(gmap, c), []).append(v)
        for c, v in group_cov(lac_node_target(Nc, yc, Nt, gmap, S, ALPHA), gmap[yt]).items():
            res["lac_gp"].setdefault(key(gmap, c), []).append(v)
        mc = AdmitM1(O, ALPHA, None).fit(Nc, yc, [])
        for c, v in group_cov(mc.predict(Nt), mc.target(yt)).items():
            res["m1cal"].setdefault(key(mc.gmap, c), []).append(v)
    summ = {m: {g: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))), len(v)) for g, v in r.items()}
            for m, r in res.items()}
    summ["frozen_inf_fraction"] = inf_groups / max(n_groups, 1)
    return summ


def test_frozen_map_basic():
    O = onto()
    adm = np.ones(O.K, dtype=bool)
    # 1000 training labels: 3 B cells -> expected 0.3 calibration cells at n_cal 100 -> B merges into L
    y = np.array([1] * 600 + [2] * 3 + [0] * 97 + [4] * 300)
    gmap, S, e = frozen_support_map(O, adm, y, 100, 9)
    assert gmap[2] == 0 and S[0] and not S[2]
    assert abs(e[0] - (97 + 3) * 0.1) < 1e-9
    # lineage M with expected count < n_min and no ancestor -> no supported target
    gmap, S, _ = frozen_support_map(O, adm, np.array([1] * 990 + [4] * 10), 100, 9)
    assert gmap[4] == -1 and gmap[3] == -1 and not S[3]
    # vocabulary restriction: target annotates only L and M -> A, B coarsened to L, C to M
    r = restrict_to_vocab(O, np.array([1, 2, 4, 0]), [0, 3])
    assert list(r) == [0, 0, 3, 0]
    # exact calibration: a group with fewer than n_min calibration cells gets +inf
    gmap, S, _ = frozen_support_map(O, adm, np.array([1] * 500 + [2] * 100 + [4] * 400), 100, 9)
    P, yy = sample(np.random.default_rng(1), 300)
    keep = np.flatnonzero(yy != 2)[:200]
    m = AdmitM1Frozen(O, ALPHA, gmap, S).fit(O.aggregate(P[keep]), yy[keep])
    assert S[2] and np.isposinf(m.lam[2]) and m.infinite[2]


def test_group_conditional_coverage(reps=600):
    s = simulate(reps=reps)
    for g, (mean, se, n) in s["frozen"].items():
        assert mean >= 1 - ALPHA - 3 * se - 1e-3, (g, mean, se, n)


if __name__ == "__main__":
    test_frozen_map_basic()
    s = simulate(reps=int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
    print("group-conditional coverage averaged over repetitions where the group exists (mean, s.e., reps)")
    for m in ("frozen", "lac_gp", "m1cal"):
        for g, (mean, se, n) in sorted(s[m].items()):
            flag = "" if mean >= 1 - ALPHA - 3 * se else "  <-- below 0.90 beyond MC error"
            print(f"  {m:7s} group {g:4s} {mean:.4f} ({se:.4f}, n={n}){flag}")
    print(f"  frozen: fraction of groups with lambda=+inf {s['frozen_inf_fraction']:.3f}")
    for g, (mean, se, n) in s["frozen"].items():
        assert mean >= 1 - ALPHA - 3 * se - 1e-3, (g, mean, se, n)
    print("OK: frozen map holds group-conditional coverage >= 0.90 for every group")
