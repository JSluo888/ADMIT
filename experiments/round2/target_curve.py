#!/usr/bin/env python3
"""Target-calibration curves on the existing external cohorts (round 2).

For each cohort k and n in {50, 100, 200, 500, 1000, 2000} labelled target cells (10 seeds): a random slice of
n cohort cells calibrates each arm (cohort features as measured, panel P_k); evaluation on up to 50,000 other
cells of the cohort (same seed). n = 0 is the label-free reference: Orion crc_cal masked to P_k cap Orion
(panel-matched). Arms: ADMIT, ADMIT+M1 (n_min rule), ADMIT+M1 (n_min 50), marginal projected LAC, LAC+filter,
APS+filter. Metrics (cell-level; the caches carry no patient ids): exact coverage of the arm's target,
worst type (strata >= 100 test cells), types with Wilson upper < 0.90, |C|, correct leaf-resolution.

  python experiments/round2/target_curve.py --backbone round0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (ALL_COHORTS, COHORT_FLAGS, RUNS, XARC_PLACEHOLDER, get_backbone, load_cohort,  # noqa: E402
                    load_split, orion_panel, write_json)
from admit.panel_matched import (ADMIT, AdmitM1, Ontology, aps_flat, lac_flat, lac_node_projected,  # noqa: E402
                                       resolved_leaf, wilson)

NS = [0, 50, 100, 200, 500, 1000, 2000]
SEEDS = list(range(10))
ALPHA = 0.10
TEST_CAP = 50000
ARMS = ["admit", "admit_m1", "admit_m1_n50", "lac_node_proj", "lac_filt", "aps_filt"]


def arm_sets(onto, panel, Pc, yc, Pt):
    adm = onto.admissible(panel)
    proj = onto.projection_map(adm)
    Nc, Nt = onto.aggregate(Pc), onto.aggregate(Pt)
    out = {"admit": (ADMIT(onto, ALPHA).fit(Nc, yc, panel).predict(Nt), proj)}
    for nm, nmin in (("admit_m1", None), ("admit_m1_n50", 50)):
        m = AdmitM1(onto, ALPHA, nmin).fit(Nc, yc, panel)
        out[nm] = (m.predict(Nt), m.gmap)
    out["lac_node_proj"] = (lac_node_projected(Nc, yc, Nt, onto, panel, ALPHA), proj)
    out["lac_filt"] = (lac_flat(Pc, yc, Pt, ALPHA) & adm[None, :], proj)
    out["aps_filt"] = (aps_flat(Pc, yc, Pt, ALPHA) & adm[None, :], proj)
    return out


def metrics(S, yt, tmap, onto):
    tgt = tmap[yt]
    ident = tgt >= 0
    if not ident.any():
        return None
    hit = S[np.where(ident)[0], tgt[ident]]
    worst, nsig, ntypes = None, 0, 0
    for c in np.unique(tgt[ident]):
        sel = tgt[ident] == c
        if sel.sum() >= 100:
            ntypes += 1
            cov = hit[sel].mean()
            worst = cov if worst is None else min(worst, cov)
            nsig += wilson(hit[sel].sum(), int(sel.sum()))[1] < 0.90
    return {"cov": float(hit.mean()), "worst": None if worst is None else float(worst), "n_sig": int(nsig),
            "n_types": ntypes, "size": float(S.sum(1).mean()),
            "leafres_correct": float((resolved_leaf(S, onto) == yt).mean()), "n_unident": int((~ident).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="round0", choices=["round0", "v6crc"])
    args = ap.parse_args()
    t0 = time.time()
    cal = load_split("crc_cal")
    tt = [str(t) for t in cal["target_types"]]
    onto = Ontology.from_repo(tt)
    bb = get_backbone(args.backbone, tt)
    orion = orion_panel()
    Xc, yc = bb.features(cal), cal["y"].astype(np.int64)
    out = {"backbone": bb.name, "alpha": ALPHA, "ns": NS, "seeds": SEEDS, "test_cap": TEST_CAP, "arms": ARMS,
           "cohorts": {}}
    for k in ALL_COHORTS:
        tk = time.time()
        d = load_cohort(k)
        Pk, yk = d["panel"], d["y"]
        Ptk = bb.predict17(bb.features(d, external=True))
        Pc_k = bb.predict17(bb.mask(Xc, [m for m in Pk if m in set(orion)], external=True))
        rec = {"flag": COHORT_FLAGS.get(k), "placeholder": args.backbone == "round0" and k in XARC_PLACEHOLDER,
               "n_cells": int(len(yk)), "curve": {}}
        for n in NS:
            runs = {a: [] for a in ARMS}
            for seed in SEEDS:
                perm = np.random.default_rng(seed).permutation(len(yk))
                ci = np.sort(perm[:n]) if n else np.array([], dtype=int)
                ti = np.sort(perm[n:n + TEST_CAP])
                if n == 0:
                    sets = arm_sets(onto, Pk, Pc_k, yc, Ptk[ti])
                else:
                    sets = arm_sets(onto, Pk, Ptk[ci], yk[ci], Ptk[ti])
                for a, (S, tmap) in sets.items():
                    m = metrics(S, yk[ti], tmap, onto)
                    if m is not None:
                        runs[a].append(m)
            summ = {}
            for a, rs in runs.items():
                if not rs:
                    continue
                summ[a] = {key: {"mean": float(np.mean([r[key] for r in rs if r[key] is not None])),
                                 "sd": float(np.std([r[key] for r in rs if r[key] is not None], ddof=1))
                                 if len(rs) > 1 else 0.0}
                           for key in ("cov", "worst", "size", "leafres_correct", "n_sig")
                           if any(r[key] is not None for r in rs)}
                summ[a]["seeds"] = rs
            rec["curve"][str(n)] = summ
        rec["wall_clock_s"] = round(time.time() - tk, 1)
        out["cohorts"][k] = rec
        c = rec["curve"]
        print(f"[{k:9s}] admit_m1 cov n=0 {c['0']['admit_m1']['cov']['mean']:.3f} n=200 "
              f"{c['200']['admit_m1']['cov']['mean']:.3f} n=2000 {c['2000']['admit_m1']['cov']['mean']:.3f} "
              f"{rec['wall_clock_s']}s", flush=True)
    out["wall_clock_s"] = round(time.time() - t0, 1)
    write_json(RUNS / "round2" / "target_curve" / f"target_curve_{args.backbone}.json", out)


if __name__ == "__main__":
    main()
