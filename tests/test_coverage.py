"""On exchangeable synthetic data, coverage of every revised-label group is at least 1 - alpha (up to Monte Carlo
error), averaged over many random calibration / test splits.

A group is checked when it is a calibration group in at least half of the splits (rare types are merged into their
parent by ADMIT+M1 in most small-calibration splits, and a handful of splits gives no precise estimate)."""
import warnings

import numpy as np
import pytest

from admit import ADMIT, NODE17, AdmitM1, Ontology, mask_arc
from admit.synthetic import make_toy_cohort, predict17, zscore

ONTO = Ontology.from_repo(NODE17)
REPS = 200
TOL = 0.03


@pytest.fixture(scope="module")
def pool():
    """A frozen classifier and a pool of i.i.d. cells (no patient effect) scored under a panel without CD8."""
    from sklearn.linear_model import LogisticRegression
    d = make_toy_cohort(n_patients=1, cells_per_patient=26000, patient_sd=0.0, signal=2.5, seed=7)
    Xtr, Xpool = zscore(d["x"][:6000], d["x"][6000:])
    ytr, ypool = d["y"][:6000], d["y"][6000:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    panel = [m for m in d["markers"] if m != "CD8"]
    N = ONTO.aggregate(predict17(clf, mask_arc(Xpool, d["markers"], panel)))
    return N, ypool, panel


def group_coverage(method, N, y, panel, alpha, n_cal, seed0=0):
    """Pooled over REPS random splits: per target group, covered test cells / test cells."""
    hits = np.zeros(ONTO.K)
    count = np.zeros(ONTO.K)
    n_reps = np.zeros(ONTO.K)
    for r in range(REPS):
        perm = np.random.default_rng(seed0 + r).permutation(len(y))
        cal, te = perm[:n_cal], perm[n_cal:]
        if method == "m1":
            m = AdmitM1(ONTO, alpha).fit(N[cal], y[cal], panel)
            tgt = m.target(y[te])
        else:
            m = ADMIT(ONTO, alpha).fit(N[cal], y[cal], panel)
            tgt = ONTO.project(y[te], m.adm)
        S = m.predict(N[te])
        ok = tgt >= 0
        hit = S[np.where(ok)[0], tgt[ok]]
        hits += np.bincount(tgt[ok], weights=hit, minlength=ONTO.K)
        c = np.bincount(tgt[ok], minlength=ONTO.K)
        count += c
        n_reps += c > 0
    return hits, count, n_reps


@pytest.mark.parametrize("method", ["admit", "m1"])
@pytest.mark.parametrize("alpha,n_cal", [(0.10, 400), (0.10, 3000), (0.20, 400)])
def test_group_conditional_coverage(pool, method, alpha, n_cal):
    N, y, panel = pool
    hits, count, n_reps = group_coverage(method, N, y, panel, alpha, n_cal)
    assert count[NODE17.index("T_Lineage")] > 0 and count[NODE17.index("Cytotoxic_T_Cell")] == 0   # revised labels
    seen = n_reps >= REPS // 2
    assert seen.sum() >= 8
    cov = hits[seen] / count[seen]
    names = [NODE17[c] for c in np.where(seen)[0]]
    low = {n: round(float(c), 3) for n, c in zip(names, cov) if c < 1 - alpha - TOL}
    assert not low, f"groups below {1 - alpha} - {TOL}: {low}"
    total = hits.sum() / count.sum()
    assert total >= 1 - alpha - 0.01
