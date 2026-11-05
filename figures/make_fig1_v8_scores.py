"""Calibration scores for Fig 1, panel c. CPU only, about a minute; no training.

    python figures/make_fig1_v8_scores.py

Same calibration as the per-cell Orion case assets for the panel without CD8: the frozen round0 scorer and the
13-patient leave-one-patient-out pool of experiments/round3/run_orion3.py --protocol lopo13, minus the patient of the
Orion field (12 calibration patients); ADMIT-M1 with the support map frozen from the training labels. Thresholds are
asserted equal to groups_S_P in that field's admit_cells_meta.json. Writes histograms of log10 p(true group | cell) per
group, not cells, to $ADMIT_FIG_ASSETS/orion_case/fig1_v8_calibration_scores.npz.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "experiments"))
sys.path.insert(0, str(HERE))
import admit_style as S  # noqa: E402
from common import RUNS, get_backbone, load_identity, load_split  # noqa: E402
from admit.panel_matched import (AdmitM1Frozen, Ontology, conformal_quantile, frozen_support_map,  # noqa: E402
                                 nmin_rule)

ALPHA = 0.10
PANEL = "del_CD8"
ASSETS = S.ASSETS / "orion_case"


def lopo13_pool(bb):
    """X, y and patient of the lopo13 calibration pool (crc_cal + crc_test + rebuilt extra cells, <= 20,000 per patient)."""
    cal, test = load_split("crc_cal"), load_split("crc_test")
    ic, it = load_identity("crc_cal"), load_identity("crc_test")
    X = np.concatenate([bb.features(cal), bb.features(test)])
    y = np.concatenate([cal["y"], test["y"]]).astype(np.int64)
    g = np.concatenate([ic["sample"], it["sample"]])
    ex = np.load(RUNS / "crc_rebuild" / "crc_test_extra.npz", allow_pickle=True)
    X = np.concatenate([X, ex["X_arc"].astype(np.float32)])
    y = np.concatenate([y, ex["y"].astype(np.int64)])
    g = np.concatenate([g, ex["sample"].astype(str)])
    rng = np.random.default_rng(0)
    keep = np.sort(np.concatenate([rng.choice(np.flatnonzero(g == s), min(20000, int((g == s).sum())),
                                              replace=False) for s in sorted(np.unique(g))]))
    return X[keep], y[keep], g[keep]


def main():
    meta = json.loads((ASSETS / S.ORION_CASE_REGION / "admit_cells_meta.json").read_text())
    tt = [str(t) for t in load_split("crc_cal")["target_types"]]
    onto = Ontology.from_repo(tt)
    bb = get_backbone("round0", tt)
    bank = json.loads((RUNS / "panel_bank.json").read_text())
    panel = [m for m in bank["orion_panel"] if m != "CD8"]
    assert panel == next(p["markers"] for p in bank["panels"] if p["name"] == PANEL)
    y_train = load_split("crc_train_full")["y"].astype(np.int64)
    Xp, yp, gp = lopo13_pool(bb)
    keep = gp != meta["sample"]
    Xc, yc = Xp[keep], yp[keep].astype(np.int64)

    Pnode = onto.aggregate(bb.predict17(bb.mask(Xc, panel)))
    gmap, Sgrp, _ = frozen_support_map(onto, onto.admissible(panel), y_train, len(yc), nmin_rule(ALPHA))
    m = AdmitM1Frozen(onto, ALPHA, gmap, Sgrp).fit(Pnode, yc)
    g = gmap[yc]
    ok = g >= 0
    shared = float(conformal_quantile(1.0 - Pnode[np.where(ok)[0], g[ok]], ALPHA, "order_stat"))
    for name, v in meta["panels"][PANEL]["admit"]["groups_S_P"].items():
        c = tt.index(name)
        assert abs(v["lambda"] - float(m.lam[c])) < 1e-6 and v["n_c"] == int(m.n_c[c]), name

    edges = np.linspace(-5.0, 0.0, 61)     # log10 p; values below 1e-5 go in the first bin
    names, hist, lam, n_c, cov_shared = [], [], [], [], []
    for c in np.where(Sgrp)[0]:
        p = Pnode[g == c, c]
        logp = np.log10(np.clip(p, 1e-5, 1.0))
        names.append(onto.names[c])
        hist.append(np.histogram(logp, edges)[0])
        lam.append(float(m.lam[c]))
        n_c.append(int(m.n_c[c]))
        cov_shared.append(float((1.0 - p <= shared).mean()))
        print(f"{names[-1]:22s} n={n_c[-1]:6d} lambda={lam[-1]:.4f} p_tick={1 - lam[-1]:.4f} "
              f"coverage_at_shared={cov_shared[-1]:.3f}")
    print(f"shared threshold {shared:.4f} (p = {1 - shared:.4f}), {int(ok.sum())} calibration cells")
    np.savez(ASSETS / "fig1_v8_calibration_scores.npz", names=np.array(names), edges=edges, hist=np.array(hist),
             lam=np.array(lam), n_c=np.array(n_c), cov_at_shared=np.array(cov_shared), shared=shared,
             n_all=int(ok.sum()))


if __name__ == "__main__":
    main()
