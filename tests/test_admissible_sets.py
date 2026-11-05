"""Every type returned by ADMIT / ADMIT+M1 is admissible under the deployment panel (C_P(x) is a subset of A(P))."""
import warnings

import numpy as np
import pytest

from admit import (ADMIT, NODE17, AdmitM1, AdmitM1Frozen, Ontology, frozen_support_map, lac_node_projected,
                   mask_arc, nmin_rule)
from admit.synthetic import make_toy_cohort, predict17, zscore

ONTO = Ontology.from_repo(NODE17)


@pytest.fixture(scope="module")
def scored():
    from sklearn.linear_model import LogisticRegression
    d = make_toy_cohort(n_patients=12, cells_per_patient=800, seed=1)
    tr, rest = d["patient"] < 6, d["patient"] >= 6
    Xtr, Xrest = zscore(d["x"][tr], d["x"][rest])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=1000).fit(Xtr, d["y"][tr])
    return clf, Xrest, d["y"][rest], d["markers"]


@pytest.mark.parametrize("seed", range(25))
def test_sets_are_admissible(scored, seed):
    clf, X, y, markers = scored
    rng = np.random.default_rng(seed)
    panel = [m for m in markers if rng.random() < rng.uniform(0.3, 1.0)]
    adm = ONTO.admissible(panel)
    N = ONTO.aggregate(predict17(clf, mask_arc(X, markers, panel)))
    half = rng.permutation(len(y))
    cal, te = half[: len(y) // 4], half[len(y) // 4:]            # small calibration set: M1 merges happen
    for alpha in (0.05, 0.10, 0.20):
        m1 = AdmitM1(ONTO, alpha).fit(N[cal], y[cal], panel)
        s1 = m1.predict(N[te])
        assert not (s1 & ~adm[None, :]).any()
        assert not (s1 & ~m1.S[None, :]).any()                   # only supported groups are returned
        assert (m1.target(y[te])[m1.target(y[te]) >= 0] >= 0).all()
        gmap, S, _ = frozen_support_map(ONTO, adm, y[cal], n_cal=len(cal), n_min=nmin_rule(alpha))
        sf = AdmitM1Frozen(ONTO, alpha, gmap, S).fit(N[cal], y[cal]).predict(N[te])
        assert not (sf & ~adm[None, :]).any() and not (sf & ~S[None, :]).any()
        s0 = ADMIT(ONTO, alpha).fit(N[cal], y[cal], panel).predict(N[te])
        assert not (s0 & ~adm[None, :]).any()
        sm = lac_node_projected(N[cal], y[cal], N[te], ONTO, panel, alpha)
        assert not (sm & ~adm[None, :]).any()
