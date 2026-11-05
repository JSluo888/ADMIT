#!/usr/bin/env python3
"""Task 6: external coverage decomposition for the local cohorts.

For cohort k with panel P_k (markers_present of its 40-D cache), with alpha = 0.10:
  (i)   old PATCH: source full-panel tolerances (crc_cal, labels y), A(P_k) filter; backstop on/off
  (ii)  panel-matched PATCH calibrated on Orion crc_cal masked to P_k (external encoding),
        projected strata pi_{P_k}(y), no backstop (+ backstop ablation), evaluated on cohort k
  (iii) the same fitted (ii) evaluated on Orion crc_test masked to P_k (pure-panel reference);
        old PATCH (i) on the same masked Orion cells for reference
  (iv)  target recalibration with the same predictor / filter / output rule (panel-matched
        PATCH with strata pi_{P_k}(y)) on a labeled target slice: 500 random cells, and a
        50/50 split; 10 seeds each; (i) and (ii) re-scored on the identical held-out cells.
Comparator arms (flat/hierarchical, +-filter) are also run with Orion-cal panel-matched
calibration on cohort k, on identical cells.

Labels: the caches already carry the 17-node harmonized labels (cache_*_features.py). A label
that is an internal node (e.g. liu T_Lineage / Myeloid, hoch Myeloid) is treated as that node,
exactly as hier_coverage does: exact coverage needs pi_{P_k}(label) in C.

  python experiments/run_external.py --backbone round0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (ALL_COHORTS, COHORT_FLAGS, LOCAL_COHORTS, RUNS, XARC_PLACEHOLDER, get_backbone, load_cohort, load_identity,  # noqa: E402
                    load_split, orion_panel, write_json)
from arms import ARMS, run_arms  # noqa: E402
from admit.panel_matched import (AdmitM1, Ontology, PanelMatchedPATCH, SourcePATCH, headline,  # noqa: E402
                                       set_metrics)
M1_VARIANTS = (("admit_m1", None), ("admit_m1_n50", 50))   # round 2: n_min rule (9 at alpha 0.1) and 50

ALPHA, N_MIN, QUANT = 0.10, 10, "order_stat"
SEEDS = list(range(10))
SLICE = 500


def summarize(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"mean": None, "sd": None, "min": None, "max": None, "n": 0}
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="round0", choices=["round0", "round0rt", "v6crc"])
    ap.add_argument("--cohorts", default=",".join(ALL_COHORTS))
    ap.add_argument("--orion", default="cal", choices=["cal", "pooled"],
                    help="cal: Orion reference = crc_cal (cal) / crc_test (iii). pooled: crc_cal+crc_test, 7 "
                         "samples; (i),(ii) calibrate on all 7; (iii) = leave-one-sample-out on the pool")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--round2", action="store_true", help="write under round2/external (adds ADMIT+M1 arms)")
    args = ap.parse_args()
    t0 = time.time()
    cal, test = load_split("crc_cal"), load_split("crc_test")
    tt = [str(t) for t in cal["target_types"]]
    onto = Ontology.from_repo(tt)
    par = onto.ancestor_lists()
    bb = get_backbone(args.backbone, tt)
    orion = orion_panel()
    yc, yt = cal["y"].astype(np.int64), test["y"].astype(np.int64)
    Xc, Xt = bb.features(cal), bb.features(test)
    g_pool = None
    if args.orion == "pooled":
        ic, it = load_identity("crc_cal"), load_identity("crc_test")
        if ic is None or it is None:
            raise SystemExit("--orion pooled needs $ADMIT_IDENTITY_DIR (sample ids for the LOSO reference)")
        g_pool = np.concatenate([ic["sample"], it["sample"]])
        Xc = np.concatenate([Xc, Xt]); yc = np.concatenate([yc, yt])
        Xt, yt = Xc, yc
    Pc_full = bb.predict17(bb.mask(Xc, orion))
    src = SourcePATCH(onto, ALPHA, N_MIN, QUANT).fit(onto.aggregate(Pc_full), yc)
    out = {"backbone": bb.name, "orion_reference": (
               "crc_cal (2 Orion patients) calibrates (i)/(ii); crc_test (5 Orion patients) is the (iii) reference"
               if args.orion == "cal" else
               "crc_cal+crc_test pooled (7 samples) calibrates (i)/(ii); (iii) is leave-one-sample-out on the pool"),
           "alpha": ALPHA, "n_min": N_MIN, "quantile": QUANT, "slice": SLICE,
           "seeds": SEEDS, "orion_panel": orion, "cohorts": {}}
    for k in args.cohorts.split(","):
        tk = time.time()
        d = load_cohort(k)
        assert d["target_types"] == tt, k
        Pk = d["panel"]
        yk = d["y"]
        Xk = bb.features(d, external=True)
        Ptk = bb.predict17(Xk)
        Ntk = onto.aggregate(Ptk)
        adm = onto.admissible(Pk)
        Pk_cal = [m for m in Pk if m in set(orion)]   # masking is only defined for markers Orion measured
        rec = {"panel": Pk, "calibration_mask_panel_Pk_cap_orion": Pk_cal,
               "markers_unused_by_scorer": [m for m in Pk if m not in set(orion)],
               "local": k in LOCAL_COHORTS, "flag": COHORT_FLAGS.get(k),
               "n_cells": int(len(yk)), "admissible": [onto.names[c] for c in np.where(adm)[0]],
               "label_counts": {onto.names[c]: int(n) for c, n in enumerate(np.bincount(yk, minlength=onto.K)) if n},
               "projected_counts": {}, "xarc_placeholder": (args.backbone in ("round0", "round0rt") and k in XARC_PLACEHOLDER)}
        proj = onto.project(yk, adm)
        for c, n in zip(*np.unique(proj, return_counts=True)):
            rec["projected_counts"][onto.names[c] if c >= 0 else "<unidentifiable>"] = int(n)
        if rec["xarc_placeholder"]:
            rec["warning"] = ("X_arc in this cache is an all-zero placeholder (cache script writes np.zeros); "
                              "the X_arc scorer sees one constant input for every cell")
        # (i) old PATCH on cohort k
        S_i = {bs: src.predict(Ntk, Pk, backstop=bs) for bs in (False, True)}
        rec["i_source_patch"] = {f"backstop_{'on' if bs else 'off'}":
                                 {"headline": headline(m := set_metrics(S_i[bs], yk, onto, Pk)), "metrics": m}
                                 for bs in (False, True)}
        # (ii) panel-matched on Orion crc_cal masked to P_k (external encoding)
        Pc_k = bb.predict17(bb.mask(Xc, Pk_cal, external=True))
        pm = PanelMatchedPATCH(onto, ALPHA, QUANT).fit(onto.aggregate(Pc_k), yc, Pk)
        rec["pm_strata_orion"] = pm.strata_report()
        rec["pm_infinite_strata"] = [onto.names[c] for c in np.where(pm.infinite)[0]]
        S_ii = {bs: pm.predict(Ntk, backstop=bs) for bs in (False, True)}
        rec["ii_panel_matched"] = {f"backstop_{'on' if bs else 'off'}":
                                   {"headline": headline(m := set_metrics(S_ii[bs], yk, onto, Pk, supported=pm.finite)),
                                    "metrics": m} for bs in (False, True)}
        # round 2: ADMIT+M1 on the cohort (same panel-matched Orion calibration)
        m1s = {nm: AdmitM1(onto, ALPHA, nmin).fit(onto.aggregate(Pc_k), yc, Pk) for nm, nmin in M1_VARIANTS}
        S_ii_m1 = {nm: m.predict(Ntk) for nm, m in m1s.items()}
        rec["ii_admit_m1"] = {nm: {"headline": headline(mm := set_metrics(S_ii_m1[nm], yk, onto, Pk,
                                                                            target=m.target(yk))),
                                   "metrics": mm, "groups": m.strata_report(),
                                   "n_cal_unidentifiable": int((m.target(yc) < 0).sum())}
                              for nm, m in m1s.items()}
        # (iii) same PM-PATCH on Orion crc_test masked to P_k
        if g_pool is None:
            Pt_ok = bb.predict17(bb.mask(Xt, Pk_cal, external=True))
            Nt_ok = onto.aggregate(Pt_ok)
            m3 = set_metrics(pm.predict(Nt_ok), yt, onto, Pk, supported=pm.finite)
            m3s = set_metrics(src.predict(Nt_ok, Pk, backstop=True), yt, onto, Pk)
            m3m1 = {nm: set_metrics(m.predict(Nt_ok), yt, onto, Pk, target=m.target(yt)) for nm, m in m1s.items()}
        else:  # leave-one-sample-out on the pooled Orion reference, masked to P_k
            Nk_all = onto.aggregate(Pc_k)
            Nfull = onto.aggregate(Pc_full)
            S3 = np.zeros((len(yc), onto.K), dtype=bool); S3s = np.zeros_like(S3)
            S3m = {nm: np.zeros_like(S3) for nm, _ in M1_VARIANTS}
            T3m = {nm: np.full(len(yc), -1) for nm, _ in M1_VARIANTS}
            for smp in sorted(np.unique(g_pool)):
                ci, ti = g_pool != smp, g_pool == smp
                pm_s = PanelMatchedPATCH(onto, ALPHA, QUANT).fit(Nk_all[ci], yc[ci], Pk)
                S3[ti] = pm_s.predict(Nk_all[ti])
                for nm, nmin in M1_VARIANTS:
                    ms = AdmitM1(onto, ALPHA, nmin).fit(Nk_all[ci], yc[ci], Pk)
                    S3m[nm][ti] = ms.predict(Nk_all[ti])
                    T3m[nm][ti] = ms.target(yc[ti])
                src_s = SourcePATCH(onto, ALPHA, N_MIN, QUANT).fit(Nfull[ci], yc[ci])
                S3s[ti] = src_s.predict(Nk_all[ti], Pk, backstop=True)
            m3 = set_metrics(S3, yc, onto, Pk, groups=g_pool, bootstrap=args.bootstrap)
            m3s = set_metrics(S3s, yc, onto, Pk, groups=g_pool, bootstrap=args.bootstrap)
            m3m1 = {nm: set_metrics(S3m[nm], yc, onto, Pk, groups=g_pool, bootstrap=args.bootstrap, target=T3m[nm])
                    for nm, _ in M1_VARIANTS}
        rec["iii_orion_test_masked_to_Pk"] = {"panel_matched": {"headline": headline(m3), "metrics": m3},
                                              "source_patch_backstop_on": {"headline": headline(m3s), "metrics": m3s},
                                              **{nm: {"headline": headline(v), "metrics": v} for nm, v in m3m1.items()}}
        # comparator arms: Orion-cal panel-matched calibration, cohort-k test cells
        sets, info = run_arms(onto, par, Pk, Pc_k, yc, Ptk, Pc_full, src, ALPHA, QUANT, N_MIN)
        rec["arms_orion_cal_on_cohort"] = {a: headline(set_metrics(S, yk, onto, Pk, supported=pm.finite,
                                                                   per_stratum=False)) for a, S in sets.items()}
        # (iv) target recalibration
        iv = {}
        for mode in ("slice500", "split50"):
            runs = []
            for seed in SEEDS:
                perm = np.random.default_rng(seed).permutation(len(yk))
                ncal = SLICE if mode == "slice500" else len(yk) // 2
                ci, ti = np.sort(perm[:ncal]), np.sort(perm[ncal:])
                pmt = PanelMatchedPATCH(onto, ALPHA, QUANT).fit(Ntk[ci], yk[ci], Pk)
                St = pmt.predict(Ntk[ti])
                mt = set_metrics(St, yk[ti], onto, Pk, supported=pmt.finite)
                r = {"seed": seed, "n_cal": int(len(ci)), "n_test": int(len(ti)),
                     "infinite_strata": [onto.names[c] for c in np.where(pmt.infinite)[0]],
                     "target_recal": headline(mt),
                     "target_recal_backstop": headline(set_metrics(pmt.predict(Ntk[ti], backstop=True), yk[ti], onto, Pk,
                                                                   per_stratum=False)),
                     "i_source_same_cells": headline(set_metrics(S_i[True][ti], yk[ti], onto, Pk, per_stratum=False)),
                     "ii_pm_same_cells": headline(set_metrics(S_ii[False][ti], yk[ti], onto, Pk, per_stratum=False))}
                for nm, nmin in M1_VARIANTS:
                    mt1 = AdmitM1(onto, ALPHA, nmin).fit(Ntk[ci], yk[ci], Pk)
                    r[f"target_recal_{nm}"] = headline(set_metrics(mt1.predict(Ntk[ti]), yk[ti], onto, Pk,
                                                                   per_stratum=False, target=mt1.target(yk[ti])))
                    r[f"ii_{nm}_same_cells"] = headline(set_metrics(S_ii_m1[nm][ti], yk[ti], onto, Pk, per_stratum=False,
                                                                    target=m1s[nm].target(yk[ti])))
                if seed == 0:
                    r["target_recal_per_stratum"] = mt["per_stratum"]
                runs.append(r)
            summ = {}
            for arm in ("target_recal", "target_recal_backstop", "i_source_same_cells", "ii_pm_same_cells",
                        "target_recal_admit_m1", "target_recal_admit_m1_n50", "ii_admit_m1_same_cells",
                        "ii_admit_m1_n50_same_cells"):
                summ[arm] = {key: summarize([r[arm][key] for r in runs])
                             for key in ("exact_cov", "leaf_cov", "anc_cov", "size", "singleton", "abstain",
                                         "exact_or_abstain", "exact_cov_supported")}
            iv[mode] = {"runs": runs, "summary": summ}
        rec["iv_target_recalibration"] = iv
        rec["wall_clock_s"] = round(time.time() - tk, 1)
        out["cohorts"][k] = rec
        h = lambda x: f"{x['exact_cov']:.3f}/{x['size']:.2f}/ab{x['abstain']:.2f}"
        print(f"[{k:9s}] i(bs) {h(rec['i_source_patch']['backstop_on']['headline'])}  "
              f"ii {h(rec['ii_panel_matched']['backstop_off']['headline'])}  "
              f"iii {h(rec['iii_orion_test_masked_to_Pk']['panel_matched']['headline'])}  "
              f"iv500 {iv['slice500']['summary']['target_recal']['exact_cov']['mean']:.3f}  "
              f"iv50 {iv['split50']['summary']['target_recal']['exact_cov']['mean']:.3f}  {rec['wall_clock_s']}s",
              flush=True)
    out["wall_clock_total_s"] = round(time.time() - t0, 1)
    suffix = "" if args.orion == "cal" else "_pooled"
    outdir = RUNS / ("round2/external" if args.round2 else "external")
    write_json(outdir / f"external_decomposition_{args.backbone}{suffix}.json", out)


if __name__ == "__main__":
    main()
