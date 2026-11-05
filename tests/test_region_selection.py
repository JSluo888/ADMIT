"""Synthetic check and unit tests of the region-selection pipeline (experiments/regions/regions_core.py).

  python tests/test_region_selection.py                    (plain runner; prints the synthetic coverage tables)
  python -m pytest -s -q tests/test_region_selection.py

Worlds (7-node hierarchy: Epi, Str, Imm > {B, T > {CD8T, CD4T}}; L = {PanCK, CD45, CD3, CD8}, F = all 7 markers):
- hetero: i.i.d. regions with heterogeneous composition (Dirichlet) and classifier difficulty (log-normal noise
  scale), 3 regions per patient, <= 60 evaluation cells per region;
- clustered: patient random effects shared by the patient's regions on composition and difficulty;
- iid: one evaluation cell per independent region, so calibration and test points are exactly exchangeable within
  every (stratum, group).
Selection by the region uncertainty score (`uncert`) at b = 0.20 through the production code (regions_core.run_panel).
Pass: for `cond`, every (stratum, group) has coverage averaged over repetitions >= 0.90 - 3 MC SE.
`pooled` is reported whatever it shows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:                                        # no pytest: plain runner below
    pytest = None

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "regions"))
sys.path.insert(0, str(REPO))
import regions_core as rc  # noqa: E402
from admit.panel_matched import Ontology  # noqa: E402

NAMES = ["Epi", "Imm", "T", "CD8T", "CD4T", "B", "Str"]
PARENT = {"Epi": None, "Imm": None, "T": "Imm", "CD8T": "T", "CD4T": "T", "B": "Imm", "Str": None}
REQ = {"Epi": ["PanCK"], "Imm": ["CD45"], "T": ["CD45", "CD3"], "CD8T": ["CD45", "CD3", "CD8"],
       "CD4T": ["CD45", "CD3", "CD4"], "B": ["CD45", "CD20"], "Str": ["SMA"]}
F = ["PanCK", "CD45", "CD3", "CD8", "CD4", "CD20", "SMA"]
L = ["PanCK", "CD45", "CD3", "CD8"]
ONTO = Ontology(NAMES, PARENT, REQ)
BASE = np.array([0.40, 0.04, 0.0, 0.12, 0.14, 0.10, 0.20])
IMMUNE_DIR = np.array([-1.0, 1.0, 0.0, 1.0, 1.0, 1.0, -0.5])
# under L the classifier cannot separate CD4T from CD8T well, B from generic immune, stroma from epithelium
M_L = np.eye(7)
M_L[4] = 0
M_L[4, [4, 3]] = [0.6, 0.4]
M_L[5] = 0
M_L[5, [5, 1]] = [0.5, 0.5]
M_L[6] = 0
M_L[6, [6, 0]] = [0.5, 0.5]
B_IDX = rc.BUDGETS.index(0.20)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def world(kind: str, seed: int, budgets=(0.20,), n_pat=None):
    """n_pat: (DEV, CAL, TEST) patients; default (20, 40, 40) with 3 regions per patient, iid (300, 900, 900) x 1."""
    rng = np.random.default_rng([seed, {"hetero": 1, "clustered": 2, "iid": 3}[kind]])
    if kind == "iid":
        n_pat, rpp, cap, size = n_pat or (300, 900, 900), 1, 1, (40, 120)
    else:
        n_pat, rpp, cap, size = n_pat or (20, 40, 40), 3, 60, (80, 240)
    role_pat = np.repeat([0, 1, 2], n_pat)
    reg_role, reg_pat, comp, diff, Nr = [], [], [], [], []
    for p, rl in enumerate(role_pat):
        u = rng.normal()
        for _ in range(rpp):
            if kind == "clustered":
                w = BASE * np.exp(0.9 * u * IMMUNE_DIR + 0.3 * rng.normal(size=7))
                c = rng.dirichlet(8 * w / w.sum() + 1e-3)
                d = np.exp(0.35 * u + 0.25 * rng.normal())
            else:
                c = rng.dirichlet(4 * BASE + 1e-3)
                d = np.exp(0.45 * rng.normal())
            c[2] = 0.0
            reg_role.append(rl)
            reg_pat.append(p)
            comp.append(c / c.sum())
            diff.append(d)
            Nr.append(int(rng.integers(*size)))
    nR = len(reg_role)
    Nr = np.array(Nr)
    reg = np.repeat(np.arange(nR), Nr)
    y = np.concatenate([rng.choice(7, n, p=comp[r]) for r, n in enumerate(Nr)])
    d = np.array(diff)[reg][:, None]
    eps = rng.normal(size=(len(y), 7))
    P17F = softmax(3.0 * np.eye(7)[y] + d * eps)
    P17L = softmax(3.0 * M_L[y] + d * (0.7 * eps + 0.71 * rng.normal(size=(len(y), 7))))
    starts = np.concatenate([[0], np.cumsum(Nr)[:-1]])
    ev = np.concatenate([np.sort(rng.choice(n, min(cap, n), replace=False)) + s0 for s0, n in zip(starts, Nr)])
    R = {"role": np.array(reg_role), "patient": np.array(reg_pat), "N_r": Nr,
         "region_id": np.array([f"r{i}" for i in range(nR)]), "region_index": np.arange(nR)}
    y_train = rng.choice(7, 20000, p=BASE / BASE.sum())
    PnL = ONTO.aggregate(P17L)
    return dict(onto=ONTO, panel_F=F, panel_L=L, y_train=y_train, s=seed, cohort="synthetic", R=R, ctx_reg=reg,
                PnL_ctx=PnL, ev_reg=reg[ev], ev_y=y[ev], PnL_ev=PnL[ev], PnF_ev=ONTO.aggregate(P17F[ev]),
                P17L_ev=P17L[ev], budgets=budgets)


def coverage_reps(kind: str, reps: int, variants=("uncert_cond", "uncert_pooled"), n_pat=None):
    """-> {variant: (per-rep coverage (reps, 2, K) with NaN where undefined, per-rep test counts)}"""
    out = {v: (np.full((reps, 2, ONTO.K), np.nan), np.zeros((reps, 2, ONTO.K))) for v in variants}
    sel_frac = []
    for r in range(reps):
        res = rc.run_panel(**world(kind, 1000 + r, n_pat=n_pat))
        t = res["tstat"][:, 0].sum(axis=2)                     # (V, 2, K, TST)
        for v in variants:
            n = t[rc.VI[v], :, :, rc.TI["t_n"]]
            h = t[rc.VI[v], :, :, rc.TI["t_hit"]]
            out[v][1][r] = n
            with np.errstate(invalid="ignore", divide="ignore"):
                out[v][0][r] = np.where(n > 0, h / n, np.nan)
        sel_frac.append(res["selT"][rc.VI["uncert_cond"], 0].mean())
    return out, float(np.mean(sel_frac))


def summarise(kind, out, min_frac=0.3):
    rows, ok = [], True
    for v, (cov, n) in out.items():
        reps = cov.shape[0]
        for st in (0, 1):
            for c in range(ONTO.K):
                x = cov[:, st, c]
                x = x[np.isfinite(x)]
                if len(x) < max(10, min_frac * reps):
                    continue
                m, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))
                passed = m >= 0.90 - 3 * se
                rows.append((v, ("L/unsel", "F/sel")[st], NAMES[c], len(x), float(n[:, st, c].mean()), m, se, passed))
                if v.endswith("_cond") and not passed:
                    ok = False
    print(f"\n[{kind}] variant stratum group n_reps mean_test_n mean_cov mc_se pass(>=0.90-3SE)")
    for r_ in rows:
        print(f"  {r_[0]:14s} {r_[1]:8s} {r_[2]:5s} {r_[3]:4d} {r_[4]:8.1f} {r_[5]:.4f} {r_[6]:.4f} {r_[7]}")
    return ok, rows


SYN_CASES = [("hetero", 120), ("clustered", 120), ("iid", 120)]


def synthetic_coverage(kind, reps):
    out, frac = coverage_reps(kind, reps)
    print(f"[{kind}] mean realised fraction of TEST regions selected by uncert at b=0.20: {frac:.3f}")
    ok, rows = summarise(kind, out)
    assert rows, "no (stratum, group) cell had enough repetitions"
    assert ok, f"{kind}: cond coverage below 0.90 - 3 MC SE in some (stratum, group)"


# ------------------------------------------------------------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------------------------------------------------------------


def make_base():
    w = world("hetero", 7, budgets=rc.BUDGETS)
    return w, rc.run_panel(**w, debug_hash=True)


if pytest is not None:
    @pytest.mark.parametrize("kind,reps", SYN_CASES)
    def test_synthetic_coverage(kind, reps):
        synthetic_coverage(kind, reps)

    @pytest.fixture(scope="module")
    def base():
        return make_base()


def subset_world(w, keep_regions):
    """Drop regions (and their cells) from a world; keep everything else identical."""
    keep_regions = np.sort(keep_regions)
    new = -np.ones(len(w["R"]["role"]), dtype=np.int64)
    new[keep_regions] = np.arange(len(keep_regions))
    cm, em = new[w["ctx_reg"]] >= 0, new[w["ev_reg"]] >= 0
    v = dict(w)
    v["R"] = {k: np.asarray(a)[keep_regions] for k, a in w["R"].items()}
    v["ctx_reg"], v["PnL_ctx"] = new[w["ctx_reg"][cm]], w["PnL_ctx"][cm]
    for k in ("ev_y", "PnL_ev", "PnF_ev", "P17L_ev"):
        v[k] = w[k][em]
    v["ev_reg"] = new[w["ev_reg"][em]]
    return v, keep_regions


def test_threshold_rule_is_a_fixed_function(base):
    w, res = base
    # S_b(r) = 1{score(r) > tau_b} for every region of every role
    for ri in range(len(rc.RULES)):
        for bi in range(len(rc.BUDGETS)):
            assert np.array_equal(res["sel"][ri, bi], res["scores"][ri] > res["taus"][ri, bi])
    # tau_b and a region's score / selection do not depend on which other CAL / TEST regions exist
    role = w["R"]["role"]
    rng = np.random.default_rng(0)
    drop = np.concatenate([rng.choice(np.flatnonzero(role == r), (role == r).sum() // 2, replace=False)
                           for r in (1, 2)])
    keep = np.setdiff1d(np.arange(len(role)), drop)
    v, kept = subset_world(w, keep)
    res2 = rc.run_panel(**v)
    np.testing.assert_allclose(res2["taus"], res["taus"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(res2["scores"], res["scores"][:, kept], rtol=0, atol=1e-9)
    assert np.array_equal(res2["sel"], res["sel"][:, :, kept])
    # the same TEST region keeps its decision when other TEST regions are removed
    n_dropped_test = int(np.isin(drop, np.flatnonzero(role == 2)).sum())
    print(f"\n[fixed rule] taus identical after dropping {len(drop) - n_dropped_test} CAL and {n_dropped_test} TEST "
          f"regions; {len(kept)} regions keep their scores and decisions")


def test_random_matched_in_count(base):
    _, res = base
    S = res["selT"]
    for bi in range(len(rc.BUDGETS)):
        k = {r: int(S[rc.VI[f"{r}_cond"], bi].sum()) for r in rc.RULES}
        for r in rc.RULES:
            assert int(S[rc.VI[f"{r}_pooled"], bi].sum()) == k[r]
            assert int(S[rc.VI[f"random_k_{r}"], bi].sum()) == k[r]
        assert int(S[rc.VI["density_k"], bi].sum()) == k["uncert"]
        assert int(S[rc.VI["oracle_k_uncert"], bi].sum()) == k["uncert"]
        assert int(S[rc.VI["oracle_k_voi"], bi].sum()) == k["voi"]
        assert not S[rc.VI["ref_L"], bi].any() and S[rc.VI["ref_F"], bi].all()
    # separate draws per compared rule: same k gives different region sets for different rule indices
    a, b_ = rc.random_k(100, 20, 0, 2, 0), rc.random_k(100, 20, 0, 2, 1)
    assert a.sum() == b_.sum() == 20 and not np.array_equal(a, b_)
    assert np.array_equal(a, rc.random_k(100, 20, 0, 2, 0))
    print("\n[random_k] counts per budget (uncert, voi, density):",
          [tuple(int(S[rc.VI[f"{r}_cond"], bi].sum()) for r in rc.RULES) for bi in range(len(rc.BUDGETS))],
          "of", S.shape[-1], "TEST regions; every random_k / density_k / oracle_k arm matched")


def test_random_bern_is_a_frozen_hash():
    u = [rc.bern_u("c", f"r{i}", 3) for i in range(2000)]
    assert all(0 <= x < 1 for x in u) and u == [rc.bern_u("c", f"r{i}", 3) for i in range(2000)]
    assert abs(np.mean(np.array(u) < 0.2) - 0.2) < 0.03
    assert u != [rc.bern_u("c", f"r{i}", 4) for i in range(2000)]


def test_stratum_map_uses_dev_only(base):
    w, res = base
    role_ev = w["R"]["role"][w["ev_reg"]]
    rng = np.random.default_rng(1)
    v = dict(w)
    y = w["ev_y"].copy()
    for r in (1, 2):                                       # shuffle CAL and TEST labels
        idx = np.flatnonzero(role_ev == r)
        y[idx] = y[rng.permutation(idx)]
    v["ev_y"] = y
    res2 = rc.run_panel(**v)
    assert np.array_equal(res2["sbits"], res["sbits"])
    keep_cols = [rc.SINFO.index(c) for c in ("n_dev_regions", "n_cal_regions", "n_test_regions", "N_dev", "N_cal_A2",
                                             "n_cal_cells", "n_supported")]
    np.testing.assert_array_equal(res2["sinfo"][..., keep_cols], res["sinfo"][..., keep_cols])
    # changing DEV labels changes the maps (the map is a function of DEV labels)
    v2 = dict(w)
    y2 = w["ev_y"].copy()
    dev_b = np.flatnonzero((role_ev == 0) & (y2 == NAMES.index("B")))
    y2[dev_b[: int(0.9 * len(dev_b))]] = NAMES.index("CD4T")
    v2["ev_y"] = y2
    res3 = rc.run_panel(**v2)
    assert not np.array_equal(res3["sbits"], res["sbits"])
    # direct: the function signature takes DEV labels, the DEV selection and a CAL cell COUNT only
    adm_F, adm_L = ONTO.admissible(F), ONTO.admissible(L)
    yd = w["ev_y"][role_ev == 0]
    sel_d = np.random.default_rng(2).random(len(yd)) < 0.3
    m1 = rc.stratum_maps(ONTO, adm_F, adm_L, yd, sel_d, 5000, 9)
    m2 = rc.stratum_maps(ONTO, adm_F, adm_L, yd, sel_d, 5000, 9)
    for st in (0, 1):
        assert np.array_equal(m1[st][0], m2[st][0]) and np.array_equal(m1[st][1], m2[st][1])
        assert np.isclose(m1[st][4], 5000 * m1[st][3] / len(yd))
    empty = rc.stratum_maps(ONTO, adm_F, adm_L, yd, np.zeros(len(yd), bool), 5000, 9)
    assert not empty[1][1].any() and (empty[1][0] == -1).all()     # no DEV cells -> nothing supported
    print("\n[stratum map] identical under shuffled CAL+TEST labels; changes when DEV labels change; "
          "empty DEV stratum supports nothing")


def test_oracle_never_feeds_a_method(base):
    w, res = base
    role_ev = w["R"]["role"][w["ev_reg"]]
    rng = np.random.default_rng(3)
    v = dict(w)
    y = w["ev_y"].copy()
    idx = np.flatnonzero(role_ev == 2)
    y[idx] = rng.integers(0, 7, len(idx))                  # replace every TEST label
    v["ev_y"] = y
    res2 = rc.run_panel(**v, debug_hash=True)
    np.testing.assert_array_equal(res2["scores"], res["scores"])
    np.testing.assert_array_equal(res2["taus"], res["taus"])
    np.testing.assert_array_equal(res2["sel"], res["sel"])
    np.testing.assert_array_equal(res2["sbits"], res["sbits"])
    for vname in rc.VARIANTS:
        vi = rc.VI[vname]
        for bi in range(len(rc.BUDGETS)):
            if vname in rc.ORACLE_VARIANTS:
                continue
            assert np.array_equal(res2["selT"][vi, bi], res["selT"][vi, bi]), vname
            assert res2["hashes"].get((vname, bi)) == res["hashes"].get((vname, bi)), vname
    assert not np.array_equal(res2["gain"], res["gain"])      # the oracle does use TEST labels
    print("\n[oracle] TEST labels replaced: every non-oracle selection, threshold and final set identical; "
          "oracle gains changed")


def test_selection_only_matches_full_run(base):
    w, res = base
    sel = rc.run_panel(**w, selection_only=True)
    for k in ("taus", "scores", "sel", "sbits", "ran"):
        np.testing.assert_array_equal(sel[k], res[k])
    np.testing.assert_array_equal(sel["sinfo"][..., :-1], res["sinfo"][..., :-1])


def test_q_table_fallbacks():
    key = np.array([1] * 25 + [2] * 5 + [6] * 30)
    fin = np.array([1] * 25 + [2] * 5 + [2] * 30)
    uf = np.array([0.0] * 25 + [1.0] * 5 + [1.0] * 30)
    tab = rc.q_table(key, fin, uf)
    q = rc.q_lookup(tab, np.array([1, 2, 6, 99, 98]), np.array([1, 2, 2, 2, 77]))
    np.testing.assert_allclose(q, [0.0, 1.0, 1.0, 1.0, uf.mean()])


def test_singleton_vs_chain_definitions():
    s = np.zeros((4, 7), bool)
    s[0, NAMES.index("Epi")] = True                                         # {leaf}
    s[1, [NAMES.index("Imm"), NAMES.index("T"), NAMES.index("CD8T")]] = True  # chain to a leaf
    s[2, NAMES.index("T")] = True                                           # internal node
    assert list(rc.unresolved(s, ONTO, "singleton")) == [False, True, True, True]
    assert list(rc.unresolved(s, ONTO, "chain")) == [False, False, True, True]


def _main(argv):
    import time
    import traceback
    only_units = "--units" in argv
    results = []
    t0 = time.time()
    b = make_base()
    units = [test_threshold_rule_is_a_fixed_function, test_random_matched_in_count, test_stratum_map_uses_dev_only,
             test_oracle_never_feeds_a_method, test_selection_only_matches_full_run]
    plain = [test_random_bern_is_a_frozen_hash, test_q_table_fallbacks, test_singleton_vs_chain_definitions]
    jobs = [(f.__name__, (lambda f=f: f(b))) for f in units] + [(f.__name__, f) for f in plain]
    if not only_units:
        jobs += [(f"synthetic_coverage[{k}]", (lambda k=k, r=r: synthetic_coverage(k, r))) for k, r in SYN_CASES]
    if "--diag" in argv:
        # post hoc diagnostic (not a pass criterion): the clustered world with 4x more CAL / TEST patients,
        # to tell a finite-cluster effect from a construction error
        jobs.append(("diag_clustered_160_patients",
                     lambda: summarise("clustered n_pat=(20,160,160)",
                                       {"uncert_cond": coverage_reps("clustered", 120, ("uncert_cond",),
                                                                     n_pat=(20, 160, 160))[0]["uncert_cond"]})))
    for name, fn in jobs:
        ts = time.time()
        try:
            fn()
            results.append((name, "PASS", time.time() - ts))
        except Exception:                                  # noqa: BLE001
            traceback.print_exc()
            results.append((name, "FAIL", time.time() - ts))
    print("\n=== summary ===")
    for name, st, dt in results:
        print(f"{st}  {name}  ({dt:.1f}s)")
    print(f"total {time.time() - t0:.1f}s")
    return int(any(st == "FAIL" for _, st, _ in results))


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
