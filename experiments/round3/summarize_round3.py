#!/usr/bin/env python3
"""Reduce engine3 accumulators to figure-ready JSON/TSV.

Inputs  runs/round3/cohort_<name>_<head>_s*.npz      Schurch (run_cohort3.py)
        runs/round3/orion_<protocol>.npz            Orion (run_orion3.py)
        runs/round3/cohorts/<cohort>/within_<head>_seed<k>.npz   (run_within3.py)
        runs/round3/cohorts/<pair>/pair_<dir>_<head>_seed<k>.npz (run_pair3.py)
Outputs summary_round3.json (Schurch + Orion; per run: targets.{gp,pi,own} = the round2 summary structure panels -> arms ->
        alpha -> cell summary, families, e4 split distribution; plus bookkeeping, groups, paired), cohorts/<c>/
        summary_within_<head>.json and summary_pair_<dir>_<head>.json (same layout), cohorts/summary.tsv (round-2
        columns + target and new fields), summary_all.tsv (every run), paired.tsv, strata_common.tsv, headline.json/md

  python experiments/round3/summarize_round3.py [--only schurch,orion,within,pairs,headline] [--heads xgb]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import RUNS, write_json  # noqa: E402
from engine3 import SI, TI  # noqa: E402

R3 = RUNS / "round3"
C3 = R3 / "cohorts"
B = 1000
MIN_N = 100
TARGET_COV = 0.90
TARGETS = ("gp", "pi", "own")
LABEL = {"admit": "ADMIT (pi_P groups, +inf)", "admit_m1": "ADMIT-M1 (frozen map)", "admit_m1cal": "M1-cal (round 2)",
         "lac_node_proj": "marginal LAC on pi_P", "lac_node_gp": "marginal LAC on g_P", "lac_filt": "LAC+filter",
         "aps_filt": "APS+filter", "patch_src_backstop": "full-panel cal. (old)",
         "admit_m1_crc": "ADMIT-M1 eq-patient CRC"}
TARGET_LABEL = {"gp": "g_P (supported, frozen)", "pi": "pi_P (panel only)", "own": "arm's own (round-2 convention)"}
# (name, target, A, B): paired deficit difference A - B on the common universe of the target
COMPARISONS = [("m1_vs_lacgp", "gp", "admit_m1", "lac_node_gp"),          # primary
               ("admit_vs_lacproj", "pi", "admit", "lac_node_proj"),      # fixed pi_P target, no merging
               ("m1_vs_lacproj", "pi", "admit_m1", "lac_node_proj"),      # what merging costs on pi_P
               ("m1cal_vs_lacgp", "gp", "admit_m1cal", "lac_node_gp"),
               ("admit_vs_lacgp", "gp", "admit", "lac_node_gp"),
               ("m1_vs_m1cal", "gp", "admit_m1", "admit_m1cal")]
WITHIN = ["cords2024_nsclc_imc", "danenberg2022_metabric_imc", "fischer2023_breast_imc", "meyer2025_tnbc_imc",
          "neotrip_tnbc_imc", "quiche_tnbc_mibi_spain", "quiche_tnbc_mibi_stanford", "launonen2022_hgsc_tcycif",
          "launonen2022_hgsc_tcycif__synonym"]
PAIRS = ["piyadasa_immune_vs_tumor", "kulasinghe_vs_monkman", "mcc_discovery_vs_validation", "quiche_spain_to_stanford"]
# headline runs: Schurch logistic + Orion lopo13 + 8 cohorts (XGBoost), all deletion panels
HEADLINE_RUNS = ["schurch_logistic", "orion_lopo13"] + [f"{c}:xgb" for c in WITHIN[:8]]
SUMMARY_COLS = ["cohort", "protocol", "head", "panels", "method", "method_label", "n_panels", "exact_cov",
                "cov_ci_lo", "cov_ci_hi", "ci_note", "ep_cov", "worst_type", "worst_cov", "min_worst_cov",
                "n_sig_below", "n_types", "size", "depth", "leaf_resolved", "leaf_resolved_anyleaf", "abstain",
                "inadm_mass", "n_patients", "n_seeds", "status",
                # engine3 fields
                "target", "mean_deficit", "p5_stratum_cov", "frac_coarsened", "depth_lost", "frac_no_supported_target",
                "frac_inf_groups", "frac_test_in_inf_group", "groups_per_block", "n_pi_changed_R", "frac_pi_changed_R",
                "n_ident"]


def boot_weights(n_pat, seed=0):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_pat, np.full(n_pat, 1.0 / n_pat), size=B).astype(float)


def _f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


# ------------------------------------------------------------------------------------------------------------------
# cell-level summaries
# ------------------------------------------------------------------------------------------------------------------


def cell_summary3(G, Gt, nodes, W, t, keep_labels=True, level=TARGET_COV):
    """G (n_pat, K, NSTAT) one panel/arm/alpha; Gt (n_pat, K, NTSTAT) the same panel/alpha; t in gp|pi|own.
    level = nominal coverage 1 - alpha (deficits and significance are relative to it)."""
    G = np.asarray(G, dtype=np.float64)
    gh, gn = G[:, :, SI[f"{t}_hit"]], G[:, :, SI[f"{t}_n"]]
    L = G[:, :, SI["l_n"]].sum(0)
    tot = gn.sum()
    if L.sum() == 0 or tot == 0:
        return None
    out = {"target": t, "n_cells": float(L.sum()), "n_ident": float(tot), "n_unident": float(L.sum() - tot)}
    out["cov"] = float(gh.sum() / tot)
    with np.errstate(invalid="ignore", divide="ignore"):
        cb = (W @ gh.sum(1)) / (W @ gn.sum(1))
        sb = (W @ gh) / (W @ gn)
    out["cov_ci"] = [float(np.nanpercentile(cb, 2.5)), float(np.nanpercentile(cb, 97.5))]
    pats = gn.sum(1) > 0
    pc = gh.sum(1)[pats] / gn.sum(1)[pats]
    out["ep_cov"] = float(pc.mean())
    out["min_patient_cov"] = float(pc.min())
    strata = {}
    for k in range(len(nodes)):
        n = gn[:, k].sum()
        if n == 0:
            continue
        c = float(gh[:, k].sum() / n)
        strata[nodes[k]] = {"n": int(n), "n_patients": int((gn[:, k] > 0).sum()), "cov": c,
                            "ci": [float(np.nanpercentile(sb[:, k], 2.5)), float(np.nanpercentile(sb[:, k], 97.5))],
                            "deficit": max(0.0, level - c)}
    out["strata"] = strata
    big = {k: v for k, v in strata.items() if v["n"] >= MIN_N}
    if big:
        wk = min(big, key=lambda k: big[k]["cov"])
        out["worst_type"], out["worst_cov"] = wk, big[wk]["cov"]
        out["sig_below"] = sorted([k for k, v in big.items() if v["ci"][1] < level])
        out["n_types"] = len(big)
        out["mean_deficit"] = float(np.mean([v["deficit"] for v in big.values()]))
    if t == "own":
        out["frac_test_in_inf_group"] = float(G[:, :, SI["own_inf"]].sum() / tot)
    Gt = np.asarray(Gt, dtype=np.float64)
    den = {"own": G[:, :, SI["l_own_n"]].sum(0), "pi": Gt[:, :, TI["t_pi_ident"]].sum(0),
           "gp": Gt[:, :, TI["t_gp_ident"]].sum(0)}[t]
    lh = G[:, :, SI[f"l_{t}_hit"]].sum(0)
    if keep_labels:
        lab = {}
        for k in range(len(nodes)):
            if L[k] == 0:
                continue
            lab[nodes[k]] = {"n": int(L[k]), "cov": (float(lh[k] / den[k]) if den[k] else None),
                             "n_ident": int(den[k]),
                             "size": float(G[:, k, SI["l_size"]].sum() / L[k]),
                             "depth": float(G[:, k, SI["l_depth"]].sum() / L[k]),
                             "leafres": float(G[:, k, SI["l_leafres"]].sum() / L[k]),
                             "leafres_correct": float(G[:, k, SI["l_leafres_correct"]].sum() / L[k]),
                             "single": float(G[:, k, SI["l_single"]].sum() / L[k]),
                             "abst": float(G[:, k, SI["l_abst"]].sum() / L[k])}
        out["labels"] = lab
    N = L.sum()
    for key, st in (("size", "l_size"), ("depth", "l_depth"), ("leafres", "l_leafres"),
                    ("leafres_correct", "l_leafres_correct"), ("single", "l_single"), ("abst", "l_abst")):
        out[key] = float(G[:, :, SI[st]].sum() / N)
    out["inadm_mass"] = float(G[:, :, SI["l_inadm"]].sum() / max(G[:, :, SI["l_size"]].sum(), 1))
    return out


def bookkeeping(Gt, nodes):
    """Arm-independent target bookkeeping for one panel/alpha (or a sum over panels)."""
    Gt = np.asarray(Gt, dtype=np.float64)
    s = Gt.sum(axis=(0, 1))
    n, pi, gp = s[TI["t_n"]], s[TI["t_pi_ident"]], s[TI["t_gp_ident"]]
    dv = lambda a, b: float(a / b) if b else None  # noqa: E731
    out = {"n_cells": float(n), "n_pi_ident": float(pi), "n_gp_ident": float(gp),
           "frac_coarsened": dv(s[TI["t_coarsened"]], pi), "depth_lost": dv(s[TI["t_depth_lost"]], gp),
           "depth_lost_per_coarsened_cell": dv(s[TI["t_depth_lost"]], s[TI["t_coarsened"]]),
           "frac_no_supported_target": dv(s[TI["t_nosup"]], pi), "frac_gp_ne_alpha010": dv(s[TI["t_gp_ne_a10"]], n),
           "frac_m1cal_target_ne_gp": dv(s[TI["t_m1cal_ne_gp"]], n),
           "n_pi_changed_R": float(s[TI["t_pi_changed_R"]]), "frac_pi_changed_R": dv(s[TI["t_pi_changed_R"]], n)}
    per = Gt.sum(0)
    out["labels"] = {nodes[k]: {"n": float(per[k, TI["t_n"]]),
                                "frac_coarsened": dv(per[k, TI["t_coarsened"]], per[k, TI["t_pi_ident"]]),
                                "frac_no_supported_target": dv(per[k, TI["t_nosup"]], per[k, TI["t_pi_ident"]]),
                                "frac_pi_changed_R": dv(per[k, TI["t_pi_changed_R"]], per[k, TI["t_n"]])}
                     for k in range(len(nodes)) if per[k, TI["t_n"]] > 0}
    return out


def group_summary(grp_pa, nodes, n_blocks):
    """grp_pa (K, 3) for one panel/arm/alpha summed over blocks."""
    isg, isinf = grp_pa[:, 0], grp_pa[:, 1]
    if isg.sum() == 0:
        return None
    return {"groups_per_block": float(isg.sum() / max(n_blocks, 1)), "frac_inf_groups": float(isinf.sum() / isg.sum()),
            "inf_nodes": {nodes[k]: int(isinf[k]) for k in np.where(isinf > 0)[0]}, "n_blocks": int(n_blocks)}


# ------------------------------------------------------------------------------------------------------------------
# paired comparison on a common universe
# ------------------------------------------------------------------------------------------------------------------


def paired(acc, pidx, ia, ib, ai, t, W, panels, nodes, keep_strata=False, level=TARGET_COV):
    """Paired A - B on the strata (panel, target node) with >= MIN_N pooled test cells; same strata, cells and
    denominators for both arms (asserted). Patient-cluster bootstrap with one weight vector for every stratum."""
    Ha, Hb, N, meta = [], [], [], []
    for p in pidx:
        Ga, Gb = acc[p, ia, ai], acc[p, ib, ai]
        na, nb = Ga[:, :, SI[f"{t}_n"]], Gb[:, :, SI[f"{t}_n"]]
        if Ga[:, :, SI["l_n"]].sum() == 0 or Gb[:, :, SI["l_n"]].sum() == 0:
            return None
        assert np.array_equal(na, nb), "compared arms do not share the target universe"
        tot = na.sum(0)
        for k in np.where(tot >= MIN_N)[0]:
            Ha.append(Ga[:, k, SI[f"{t}_hit"]]); Hb.append(Gb[:, k, SI[f"{t}_hit"]]); N.append(na[:, k])
            meta.append((panels[p], nodes[k]))
    if not N:
        return None
    Ha, Hb, N = (np.stack(v, 1).astype(np.float64) for v in (Ha, Hb, N))
    ca, cb = Ha.sum(0) / N.sum(0), Hb.sum(0) / N.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        WN = W @ N
        WA, WB = (W @ Ha) / WN, (W @ Hb) / WN
    da, db = np.maximum(0, level - ca), np.maximum(0, level - cb)
    DA, DB = np.maximum(0, level - WA), np.maximum(0, level - WB)
    dd = DA - DB
    dcov = WA - WB
    upA, upB = np.nanpercentile(WA, 97.5, axis=0), np.nanpercentile(WB, 97.5, axis=0)
    npat = (N > 0).sum(0)
    S = len(meta)
    out = {"n_strata": S, "n_cells": float(N.sum()),
           "n_patients_per_stratum": {"median": float(np.median(npat)), "min": int(npat.min()),
                                      "p10": float(np.percentile(npat, 10))},
           "mean_cov_A": float(ca.mean()), "mean_cov_B": float(cb.mean()),
           "mean_deficit_A": float(da.mean()), "mean_deficit_B": float(db.mean()),
           "mean_deficit_A_ci": [float(np.nanpercentile(np.nanmean(DA, 1), q)) for q in (2.5, 97.5)],
           "mean_deficit_B_ci": [float(np.nanpercentile(np.nanmean(DB, 1), q)) for q in (2.5, 97.5)],
           "diff_deficit": float((da - db).mean()),
           "diff_deficit_ci": [float(np.nanpercentile(np.nanmean(dd, 1), q)) for q in (2.5, 97.5)],
           "diff_cov": float((ca - cb).mean()),
           "diff_cov_ci": [float(np.nanpercentile(np.nanmean(dcov, 1), q)) for q in (2.5, 97.5)],
           "n_sig_below_A": int((upA < level).sum()), "n_sig_below_B": int((upB < level).sum()),
           "n_deficit_A_gt_B": int((da > db + 1e-12).sum()), "n_deficit_B_gt_A": int((db > da + 1e-12).sum()),
           "worst_A": {"panel": meta[int(ca.argmin())][0], "node": meta[int(ca.argmin())][1], "cov": float(ca.min())},
           "worst_B": {"panel": meta[int(cb.argmin())][0], "node": meta[int(cb.argmin())][1], "cov": float(cb.min())},
           "p5_cov_A": float(np.percentile(ca, 5)), "p5_cov_B": float(np.percentile(cb, 5)),
           "frac_strata_below_level_A": float((ca < level).mean()),
           "frac_strata_below_level_B": float((cb < level).mean())}
    rep = {"S": S, "sum_dd": da.sum() - db.sum(), "sum_da": da.sum(), "sum_db": db.sum(),
           "b_sum_dd": np.nansum(dd, 1), "b_cnt": np.isfinite(dd).sum(1), "b_sum_da": np.nansum(DA, 1),
           "b_sum_db": np.nansum(DB, 1), "sig_A": out["n_sig_below_A"], "sig_B": out["n_sig_below_B"]}
    if keep_strata:
        out["strata"] = [{"panel": m[0], "node": m[1], "n": int(N[:, j].sum()), "n_patients": int(npat[j]),
                          "cov_A": float(ca[j]), "cov_B": float(cb[j]),
                          "ci_A": [float(np.nanpercentile(WA[:, j], 2.5)), float(upA[j])],
                          "ci_B": [float(np.nanpercentile(WB[:, j], 2.5)), float(upB[j])],
                          "deficit_A": float(da[j]), "deficit_B": float(db[j])} for j, m in enumerate(meta)]
    return out, rep


# ------------------------------------------------------------------------------------------------------------------
# run-level reduction (Schurch, Orion, within-cohort)
# ------------------------------------------------------------------------------------------------------------------


def family_of(pn, fam):
    f = fam.get(pn)
    return f"random{pn.split('_')[0][4:]}" if f == "random_retention" else f


def summarize_run3(d, meta, ep=None, n_blocks=1, strata_rows=None, run_name=""):
    acc, acct, grp = d["acc"], d["acct"], d["grp"]
    panels, arms, nodes = list(d["panels"]), list(d["arms"]), list(d["nodes"])
    alphas = [float(a) for a in d["alphas"]]
    n_pat = acc.shape[3]
    W = boot_weights(n_pat)
    fam = meta["families"]
    res = {"alphas": alphas, "n_patients": int(n_pat), "n_blocks": int(n_blocks), "arm_labels": LABEL,
           "target_labels": TARGET_LABEL, "targets": {}, "bookkeeping": {"panels": {}, "families": {}},
           "groups": {}, "paired": {}}
    pfam = {pn: family_of(pn, fam) for pn in panels}
    fams = {}
    for pn in panels:
        fams.setdefault(pfam[pn], []).append(pn)
    fams["all_deletions"] = [pn for pn in panels if pfam[pn] != "full"]
    fams["all_panels"] = list(panels)
    # bookkeeping and groups
    for li, al in enumerate(alphas):
        a_ = f"{al:.2f}"
        for pi, pn in enumerate(panels):
            res["bookkeeping"]["panels"].setdefault(pn, {})[a_] = bookkeeping(acct[pi, li], nodes)
            for ai_, a in enumerate(arms):
                g = group_summary(np.asarray(grp[pi, ai_, li], dtype=np.float64), nodes, n_blocks)
                if g:
                    res["groups"].setdefault(pn, {}).setdefault(a, {})[a_] = g
        for f, pns in fams.items():
            idx = [panels.index(p) for p in pns]
            bk = bookkeeping(np.asarray(acct[idx, li], dtype=np.float64).sum(0), nodes)
            bk["n_panels"] = len(pns)
            gsum = {}
            for ai_, a in enumerate(arms):
                gg = np.asarray(grp[idx, ai_, li], dtype=np.float64).sum(0)
                g = group_summary(gg, nodes, n_blocks * len(pns))
                if g:
                    gsum[a] = g
            bk["groups"] = gsum
            res["bookkeeping"]["families"].setdefault(f, {})[a_] = bk
    # per-target round-2-like trees
    for t in TARGETS:
        R = {"panels": {}, "families": {}}
        for pi, pn in enumerate(panels):
            R["panels"][pn] = {"family": pfam[pn], "arms": {}}
            for ai_, a in enumerate(arms):
                R["panels"][pn]["arms"][a] = {}
                for li, al in enumerate(alphas):
                    G = acc[pi, ai_, li]
                    if np.asarray(G[:, :, SI["l_n"]]).sum() == 0:
                        continue
                    cs = cell_summary3(G, acct[pi, li], nodes, W, t, keep_labels=(li == 1), level=1.0 - al)
                    R["panels"][pn]["arms"][a][f"{al:.2f}"] = cs
        for alias, target in meta.get("panel_alias", {}).items():
            if target in R["panels"]:
                R["panels"][alias] = {**R["panels"][target], "family": fam.get(alias), "alias_of": target}
        for f, pns in fams.items():
            R["families"][f] = {"n_panels": len(pns), "arms": {}}
            for a in arms:
                R["families"][f]["arms"][a] = {}
                for al in alphas:
                    rows = [R["panels"][p]["arms"][a].get(f"{al:.2f}") for p in pns]
                    rows = [r for r in rows if r]
                    if not rows:
                        continue
                    m = lambda k: (float(np.mean([r[k] for r in rows if r.get(k) is not None]))  # noqa: E731
                                   if any(r.get(k) is not None for r in rows) else None)
                    fail = {}
                    for r in rows:
                        for tt in r.get("sig_below", []):
                            fail[tt] = fail.get(tt, 0) + 1
                    defs = [v["deficit"] for r in rows for v in r["strata"].values() if v["n"] >= MIN_N]
                    covs = [v["cov"] for r in rows for v in r["strata"].values() if v["n"] >= MIN_N]
                    ai_ = arms.index(a)
                    vals = []
                    for p_ in pns:
                        pi_ = panels.index(p_) if p_ in panels else panels.index(meta["panel_alias"][p_])
                        G_ = np.asarray(acc[pi_, ai_, alphas.index(al)], dtype=np.float64)
                        gh_, gn_ = G_[:, :, SI[f"{t}_hit"]].sum(1), G_[:, :, SI[f"{t}_n"]].sum(1)
                        if gn_.sum() > 0:
                            with np.errstate(invalid="ignore", divide="ignore"):
                                vals.append((W @ gh_) / (W @ gn_))
                    fci = ([float(np.nanpercentile(np.nanmean(np.vstack(vals), 0), q)) for q in (2.5, 97.5)]
                           if vals else None)
                    R["families"][f]["arms"][a][f"{al:.2f}"] = {
                        "cov": m("cov"), "cov_ci": fci, "ep_cov": m("ep_cov"), "worst_cov": m("worst_cov"),
                        "min_worst_cov": (float(min(r["worst_cov"] for r in rows if "worst_cov" in r))
                                          if any("worst_cov" in r for r in rows) else None),
                        "n_sig_below": int(sum(len(r.get("sig_below", [])) for r in rows)),
                        "n_types": int(sum(r.get("n_types", 0) for r in rows)), "failing_type_counts": fail,
                        "mean_deficit": float(np.mean(defs)) if defs else None,
                        "p5_stratum_cov": float(np.percentile(covs, 5)) if covs else None,
                        "size": m("size"), "depth": m("depth"), "leafres": m("leafres"),
                        "leafres_correct": m("leafres_correct"), "abst": m("abst"), "inadm_mass": m("inadm_mass"),
                        "frac_test_in_inf_group": m("frac_test_in_inf_group")}
        if ep is not None:
            k = {"own": 0, "pi": 1, "gp": 2}[t]
            E = ep[..., k]                      # split, panel, arm, alpha
            dist = {}
            for pi, pn in enumerate(panels):
                dist[pn] = {}
                for ai_, a in enumerate(arms):
                    v = E[:, pi, ai_, 1]
                    v = v[np.isfinite(v)]
                    if len(v):
                        dist[pn][a] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                       "min": float(v.min()), "p5": float(np.percentile(v, 5)),
                                       "frac_below_0.90": float((v < TARGET_COV).mean()), "n_splits": int(len(v))}
            R["e4_split_distribution"] = dist
            dels = [panels.index(p) for p in fams["all_deletions"]]
            R["e4_all_deletions"] = {a: {"mean_over_splits_and_panels": _f(np.nanmean(E[:, dels, ai_, 1])),
                                         "frac_split_panel_below_0.90": _f(np.nanmean(E[:, dels, ai_, 1] < TARGET_COV))}
                                     for ai_, a in enumerate(arms) if np.isfinite(E[:, dels, ai_, 1]).any()}
        res["targets"][t] = R
    # paired comparisons
    reps = {}
    for name, t, A, Bm in COMPARISONS:
        if A not in arms or Bm not in arms:
            continue
        ia, ib = arms.index(A), arms.index(Bm)
        res["paired"][name] = {"target": t, "A": A, "B": Bm, "families": {}}
        for f, pns in fams.items():
            if f == "all_panels":
                continue
            pidx = [panels.index(p) for p in pns]
            for li, al in enumerate(alphas):
                r = paired(acc, pidx, ia, ib, li, t, W, panels, nodes, keep_strata=(f == "full" and li == 1),
                           level=1.0 - al)
                if r is None:
                    continue
                res["paired"][name]["families"].setdefault(f, {})[f"{al:.2f}"] = r[0]
                if li == 1:
                    reps[(name, f)] = r[1]
    # common-universe strata table (alpha 0.10, every arm, both targets)
    if strata_rows is not None:
        for t in ("gp", "pi"):
            for pi, pn in enumerate(panels):
                n_ = np.asarray(acc[pi, 0, 1, :, :, SI[f"{t}_n"]], dtype=np.float64)
                tot = n_.sum(0)
                for k in np.where(tot >= MIN_N)[0]:
                    row = {"run": run_name, "target": t, "family": pfam[pn], "panel": pn, "node": nodes[k],
                           "n": int(tot[k]), "n_patients": int((n_[:, k] > 0).sum())}
                    for ai_, a in enumerate(arms):
                        h = np.asarray(acc[pi, ai_, 1, :, k, SI[f"{t}_hit"]], dtype=np.float64)
                        if np.asarray(acc[pi, ai_, 1, :, :, SI["l_n"]]).sum() == 0:
                            continue
                        with np.errstate(invalid="ignore", divide="ignore"):
                            wb = (W @ h) / (W @ n_[:, k])
                        row[f"cov_{a}"] = float(h.sum() / tot[k])
                        row[f"lo_{a}"] = float(np.nanpercentile(wb, 2.5))
                        row[f"hi_{a}"] = float(np.nanpercentile(wb, 97.5))
                    strata_rows.append(row)
    return res, reps


# ------------------------------------------------------------------------------------------------------------------
# loaders
# ------------------------------------------------------------------------------------------------------------------


def load_schurch(name):
    files = sorted(f for f in glob.glob(str(R3 / f"cohort_{name}_s*.npz")) if "debug" not in f)
    if not files:
        return None
    acc = acct = grp = None
    eps, seeds, metas = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if acc is None:
            acc, acct, grp = (z[k].astype(np.float64) for k in ("acc", "acct", "grp"))
            base = {k: z[k] for k in ("panels", "patients", "arms", "alphas", "nodes")}
        else:
            acc += z["acc"]; acct += z["acct"]; grp += z["grp"]
        eps.append(z["ep_cov"]); seeds += z["seeds"].tolist()
        metas.append(json.loads(Path(f.replace(".npz", "_meta.json")).read_text()))
    base.update(acc=acc, acct=acct, grp=grp)
    meta = metas[0]
    meta["splits"] = [s for m in metas for s in m["splits"]]
    return base, meta, np.concatenate(eps), sorted(seeds)


def load_within(cohort, head):
    files = sorted(f for f in glob.glob(str(C3 / cohort / f"within_{head}_seed*.npz"))
                   if "debug" not in f and Path(f.replace(".npz", "_meta.json")).exists())
    if not files:
        return None
    pats = set()
    for f in files:
        pats.update(np.load(f, allow_pickle=True)["patients"].tolist())
    patients = np.array(sorted(pats))
    acc = acct = grp = None
    eps, seeds, metas = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if acc is None:
            a = z["acc"]
            acc = np.zeros(a.shape[:3] + (len(patients),) + a.shape[4:], dtype=np.float32)
            acct = np.zeros(z["acct"].shape[:2] + (len(patients),) + z["acct"].shape[3:], dtype=np.float32)
            grp = np.zeros(z["grp"].shape, dtype=np.float64)
            base = {k: z[k] for k in ("panels", "arms", "alphas", "nodes")}
        assert list(z["panels"]) == list(base["panels"])
        sl = np.searchsorted(patients, z["patients"])
        acc[:, :, :, sl] += z["acc"]
        acct[:, :, sl] += z["acct"]
        grp += z["grp"]
        eps.append(z["ep_cov"]); seeds += z["seeds"].tolist()
        metas.append(json.loads(Path(f.replace(".npz", "_meta.json")).read_text()))
    base.update(acc=acc, acct=acct, grp=grp, patients=patients)
    meta = metas[0]
    meta["splits"] = [s for m in metas for s in m["splits"]]
    meta["test_accuracy_full_panel_by_seed"] = {m["seed"]: m["test_accuracy_full_panel"] for m in metas}
    return base, meta, np.concatenate(eps), sorted(seeds)


def load_orion(prot):
    f = R3 / f"orion_{prot}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    for k in ("acc", "acct", "grp"):
        d[k] = d[k].astype(np.float64)
    meta = json.loads((R3 / f"orion_{prot}_meta.json").read_text())
    return d, meta, None, int(d["n_blocks"])


# ------------------------------------------------------------------------------------------------------------------
# TSV rows
# ------------------------------------------------------------------------------------------------------------------


def family_rows(cohort, head, protocol, res, n_seeds, status="final", npat=None):
    rows = []
    for t in ("gp", "pi", "own"):
        R = res["targets"][t]
        for f, fa in R["families"].items():
            if f == "all_panels":
                continue
            bk = res["bookkeeping"]["families"][f]["0.10"]
            for a, al in fa["arms"].items():
                agg = al.get("0.10")
                if not agg:
                    continue
                if f == "full":
                    r = R["panels"]["full"]["arms"][a]["0.10"]
                    ci, note, worst, wt = r["cov_ci"], "patient-cluster bootstrap (B=1000), pooled over seeds", \
                        r.get("worst_cov"), r.get("worst_type")
                else:
                    ci, note, worst, wt = agg.get("cov_ci"), "patient-cluster bootstrap of the mean over panels (B=1000)", \
                        agg.get("worst_cov"), ""
                g = bk["groups"].get(a, {})
                rows.append({"cohort": cohort, "protocol": protocol, "head": head, "panels": f, "status": status,
                             "method": a, "method_label": LABEL.get(a, a), "n_panels": fa["n_panels"],
                             "exact_cov": agg["cov"], "cov_ci_lo": ci[0] if ci else None,
                             "cov_ci_hi": ci[1] if ci else None, "ci_note": note, "ep_cov": agg.get("ep_cov"),
                             "worst_type": wt, "worst_cov": worst, "min_worst_cov": agg.get("min_worst_cov"),
                             "n_sig_below": agg.get("n_sig_below"), "n_types": agg.get("n_types"),
                             "size": agg.get("size"), "depth": agg.get("depth"),
                             "leaf_resolved": agg.get("leafres_correct"), "leaf_resolved_anyleaf": agg.get("leafres"),
                             "abstain": agg.get("abst"), "inadm_mass": agg.get("inadm_mass"),
                             "n_patients": npat if npat is not None else res["n_patients"], "n_seeds": n_seeds,
                             "target": t, "mean_deficit": agg.get("mean_deficit"),
                             "p5_stratum_cov": agg.get("p5_stratum_cov"), "frac_coarsened": bk["frac_coarsened"],
                             "depth_lost": bk["depth_lost"], "frac_no_supported_target": bk["frac_no_supported_target"],
                             "frac_inf_groups": g.get("frac_inf_groups"),
                             "frac_test_in_inf_group": (res["targets"]["own"]["families"][f]["arms"].get(a, {})
                                                        .get("0.10", {}).get("frac_test_in_inf_group")),
                             "groups_per_block": g.get("groups_per_block"), "n_pi_changed_R": bk["n_pi_changed_R"],
                             "frac_pi_changed_R": bk["frac_pi_changed_R"], "n_ident": None})
    return rows


def paired_rows(run, cohort, head, protocol, res):
    rows = []
    for name, pr in res["paired"].items():
        for f, byal in pr["families"].items():
            for al, r in byal.items():
                rows.append({"run": run, "cohort": cohort, "head": head, "protocol": protocol, "comparison": name,
                             "target": pr["target"], "A": pr["A"], "B": pr["B"], "family": f, "alpha": al,
                             **{k: v for k, v in r.items() if not isinstance(v, (dict, list))},
                             "npat_median": r["n_patients_per_stratum"]["median"],
                             "npat_min": r["n_patients_per_stratum"]["min"],
                             "diff_deficit_lo": r["diff_deficit_ci"][0], "diff_deficit_hi": r["diff_deficit_ci"][1],
                             "diff_cov_lo": r["diff_cov_ci"][0], "diff_cov_hi": r["diff_cov_ci"][1],
                             "worst_A_cov": r["worst_A"]["cov"], "worst_A_node": r["worst_A"]["node"],
                             "worst_B_cov": r["worst_B"]["cov"], "worst_B_node": r["worst_B"]["node"]})
    return rows


def write_tsv(path, rows, cols=None):
    if not rows:
        return
    cols = cols or list(dict.fromkeys(k for r in rows for k in r))
    fmt = lambda v: (f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v)))  # noqa: E731
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\t".join(cols) + "\n" + "".join("\t".join(fmt(r.get(c)) for c in cols) + "\n" for r in rows))


# ------------------------------------------------------------------------------------------------------------------
# pairs
# ------------------------------------------------------------------------------------------------------------------


def _merge_pair(files, key, pkey):
    """Sum one array over seed files on the union of patients. Patient axis: 3 for acc_* and tcal, 2 for acct_* and
    tcal_t."""
    ax = 2 if (key.startswith("acct_") or key == "tcal_t") else 3
    pats = set()
    for f in files:
        z = np.load(f, allow_pickle=True)
        if key in z.files:
            pats.update(z[pkey].tolist())
    if not pats:
        return None, None
    patients = np.array(sorted(pats))
    arr = None
    for f in files:
        z = np.load(f, allow_pickle=True)
        if key not in z.files:
            continue
        a = z[key]
        if arr is None:
            shp = list(a.shape)
            shp[ax] = len(patients)
            arr = np.zeros(shp, dtype=np.float64)
        idx = [slice(None)] * arr.ndim
        idx[ax] = np.searchsorted(patients, z[pkey])
        arr[tuple(idx)] += a
    return arr, patients


def summarize_pair3(pair, heads, all_rows, paired_all, strata_rows):
    reg = json.loads((HERE.parents[1] / "configs" / "cohort_registry.json").read_text())["pairs"][pair]
    status = reg.get("status", "final")
    for dr in ("ab", "ba"):
        for head in heads:
            files = sorted(f for f in glob.glob(str(C3 / pair / f"pair_{dr}_{head}_seed*.npz"))
                           if Path(f.replace(".npz", "_meta.json")).exists())
            if not files:
                continue
            meta = json.loads(Path(files[0].replace(".npz", "_meta.json")).read_text())
            z0 = np.load(files[0], allow_pickle=True)
            arms, nodes = list(z0["arms"]), list(z0["nodes"])
            alphas = [float(a) for a in z0["alphas"]]
            src, tgt = meta["source"], meta["target"]
            res = {"pair": pair, "protocol": reg.get("protocol"), "status": status, "direction": dr, "head": head,
                   "engine": "engine3", "source": src, "target": tgt,
                   "shared_panel": meta["shared_panel"], "target_vocabulary_V_B": meta["target_vocabulary_V_B"],
                   "admissible_under_P": meta["admissible_under_P"],
                   "admissible_under_P_round2_map": meta["admissible_under_P_round2_map"],
                   "seeds": [], "blocks": {}, "arm_labels": LABEL, "target_labels": TARGET_LABEL}
            for f in files:
                m = json.loads(Path(f.replace(".npz", "_meta.json")).read_text())
                res["seeds"].append(m["seed"])
                res.setdefault("accuracy_by_seed", {})[m["seed"]] = m["accuracy"]
            nseed = len(files)
            for blk, pk in (("transfer", "patients_B"), ("within_source_masked", "patients_A"),
                            ("target_full", "patients_B")):
                acc, pats = _merge_pair(files, f"acc_{blk}", pk)
                if acc is None:
                    continue
                acct, _ = _merge_pair(files, f"acct_{blk}", pk)
                grp = sum(np.load(f, allow_pickle=True)[f"grp_{blk}"].astype(np.float64) for f in files)
                d = {"acc": acc, "acct": acct, "grp": grp, "panels": np.array([blk]), "arms": np.array(arms),
                     "alphas": np.array(alphas), "nodes": np.array(nodes)}
                r, reps = summarize_run3(d, {"families": {blk: "full"}}, None, nseed, strata_rows,
                                         f"{pair}:{dr}:{head}:{blk}")
                # the block is the 'full' family; drop the empty all_deletions entries
                for t in r["targets"].values():
                    t["families"].pop("all_deletions", None)
                r["bookkeeping"]["families"].pop("all_deletions", None)
                res["blocks"][blk] = {"n_patients": int(len(pats)), **r}
                for t in ("gp", "pi", "own"):
                    for a in arms:
                        cs = r["targets"][t]["panels"][blk]["arms"][a].get("0.10")
                        if cs is None:
                            continue
                        bk = r["bookkeeping"]["panels"][blk]["0.10"]
                        g = r["groups"].get(blk, {}).get(a, {}).get("0.10", {})
                        all_rows.append(_pair_row(pair, reg, head, f"{src}->{tgt}:{blk}", status, a, t, cs, bk, g,
                                                  len(pats), nseed, r["targets"]["own"]["panels"][blk]["arms"][a]
                                                  .get("0.10", {}) or {}))
                for name, pr in r["paired"].items():
                    for al, x in pr["families"].get("full", {}).items():
                        paired_all.append({"run": f"{pair}:{dr}:{head}", "cohort": pair, "head": head,
                                           "protocol": reg.get("protocol"), "comparison": name, "target": pr["target"],
                                           "A": pr["A"], "B": pr["B"], "family": blk, "alpha": al,
                                           **{k: v for k, v in x.items() if not isinstance(v, (dict, list))},
                                           "npat_median": x["n_patients_per_stratum"]["median"],
                                           "npat_min": x["n_patients_per_stratum"]["min"],
                                           "diff_deficit_lo": x["diff_deficit_ci"][0],
                                           "diff_deficit_hi": x["diff_deficit_ci"][1],
                                           "diff_cov_lo": x["diff_cov_ci"][0], "diff_cov_hi": x["diff_cov_ci"][1],
                                           "worst_A_cov": x["worst_A"]["cov"], "worst_A_node": x["worst_A"]["node"],
                                           "worst_B_cov": x["worst_B"]["cov"], "worst_B_node": x["worst_B"]["node"]})
            # target-calibration curve
            tc, pats = _merge_pair(files, "tcal", "patients_B")
            if tc is not None:
                tct, _ = _merge_pair(files, "tcal_t", "patients_B")
                tcg = sum(np.load(f, allow_pickle=True)["tcal_g"].astype(np.float64) for f in files)
                ns, tarms = [int(n) for n in z0["tcal_ns"]], list(z0["tcal_arms"])
                W = boot_weights(len(pats))
                curve = {}
                for ni, n in enumerate(ns):
                    Gt = tct[ni].sum(0)
                    bk = bookkeeping(Gt, nodes)
                    curve[n] = {"bookkeeping": {k: v for k, v in bk.items() if k != "labels"}, "arms": {}}
                    for k, a in enumerate(tarms):
                        G = tc[ni, :, k].sum(0)
                        g = group_summary(tcg[ni, :, k].sum(0), nodes, nseed * tc.shape[1])
                        curve[n]["arms"][a] = {"groups": g}
                        for t in ("gp", "pi", "own"):
                            cs = cell_summary3(G, Gt, nodes, W, t, keep_labels=False)
                            if cs is None:
                                continue
                            draws = []
                            for f in files:
                                tt_ = np.load(f, allow_pickle=True)["tcal"][ni, :, k]
                                gh, gn = tt_[..., SI[f"{t}_hit"]].sum((1, 2)), tt_[..., SI[f"{t}_n"]].sum((1, 2))
                                draws += list(gh[gn > 0] / gn[gn > 0])
                            cs["draw_cov_mean"] = float(np.mean(draws)) if draws else None
                            cs["draw_cov_p10_p90"] = ([float(np.percentile(draws, 10)), float(np.percentile(draws, 90))]
                                                      if draws else None)
                            cs["n_draws"] = len(draws)
                            curve[n]["arms"][a][t] = cs
                            all_rows.append(_pair_row(pair, reg, head, f"{src}->{tgt}:n={n}", status, a, t, cs, bk,
                                                      g or {}, len(pats), nseed, curve[n]["arms"][a].get("own", {}),
                                                      protocol="target_calibration_curve"))
                res["target_calibration_curve"] = curve
            nexp = len(reg["splits"]["seeds"])
            res["completeness"] = f"{len(res['seeds'])}/{nexp} seeds"
            write_json(C3 / pair / f"summary_pair_{dr}_{head}.json", res)


def _pair_row(pair, reg, head, panels, status, a, t, cs, bk, g, npat, nseed, own, protocol=None):
    return {"cohort": pair, "protocol": protocol or reg.get("protocol"), "head": head, "panels": panels,
            "status": status, "method": a, "method_label": LABEL.get(a, a), "n_panels": 1, "exact_cov": cs.get("cov"),
            "cov_ci_lo": cs.get("cov_ci", [None, None])[0], "cov_ci_hi": cs.get("cov_ci", [None, None])[1],
            "ci_note": "patient-cluster bootstrap (B=1000), pooled over seeds" + (
                f"; {cs['n_draws']} draws, p10-p90 {cs['draw_cov_p10_p90'][0]:.3f}-{cs['draw_cov_p10_p90'][1]:.3f}"
                if cs.get("draw_cov_p10_p90") else ""),
            "ep_cov": cs.get("ep_cov"), "worst_type": cs.get("worst_type"), "worst_cov": cs.get("worst_cov"),
            "min_worst_cov": cs.get("worst_cov"), "n_sig_below": len(cs.get("sig_below", [])),
            "n_types": cs.get("n_types"), "size": cs.get("size"), "depth": cs.get("depth"),
            "leaf_resolved": cs.get("leafres_correct"), "leaf_resolved_anyleaf": cs.get("leafres"),
            "abstain": cs.get("abst"), "inadm_mass": cs.get("inadm_mass"), "n_patients": npat, "n_seeds": nseed,
            "target": t, "mean_deficit": cs.get("mean_deficit"), "p5_stratum_cov": None,
            "frac_coarsened": bk.get("frac_coarsened"), "depth_lost": bk.get("depth_lost"),
            "frac_no_supported_target": bk.get("frac_no_supported_target"), "frac_inf_groups": g.get("frac_inf_groups"),
            "frac_test_in_inf_group": own.get("frac_test_in_inf_group") if own else None,
            "groups_per_block": g.get("groups_per_block"), "n_pi_changed_R": bk.get("n_pi_changed_R"),
            "frac_pi_changed_R": bk.get("frac_pi_changed_R"), "n_ident": cs.get("n_ident")}


# ------------------------------------------------------------------------------------------------------------------
# headline
# ------------------------------------------------------------------------------------------------------------------


def old_round2():
    """Per-run numbers of the earlier round2 engine (all deletions, alpha 0.10), for comparison."""
    out = {}
    S = RUNS / "round2" / "summary_round2.json"
    if S.exists():
        s = json.loads(S.read_text())
        for run in ("schurch_logistic", "schurch_xgb", "orion_lopo13"):
            if run in s["runs"]:
                out[run] = s["runs"][run]["families"]["all_deletions"]["arms"]
    for c in WITHIN:
        f = RUNS / "cohorts" / c / "summary_within_xgb.json"
        if f.exists():
            out[f"{c}:xgb"] = json.loads(f.read_text())["families"]["all_deletions"]["arms"]
    return out


def headline(all_res, all_reps):
    old = old_round2()
    rows, pooled = [], {}
    for run, res in all_res.items():
        fam = "all_deletions"
        g = res["targets"]["gp"]["families"][fam]["arms"]
        p = res["targets"]["pi"]["families"][fam]["arms"]
        o = res["targets"]["own"]["families"][fam]["arms"]
        bk = res["bookkeeping"]["families"][fam]["0.10"]
        pr = res["paired"].get("m1_vs_lacgp", {}).get("families", {}).get(fam, {}).get("0.10")
        pp = res["paired"].get("admit_vs_lacproj", {}).get("families", {}).get(fam, {}).get("0.10")
        pm = res["paired"].get("m1_vs_lacproj", {}).get("families", {}).get(fam, {}).get("0.10")
        oa = old.get(run, {})
        get = lambda D, a, k: D.get(a, {}).get("0.10", {}).get(k)  # noqa: E731
        row = {"run": run, "n_patients": res["n_patients"], "n_panels": bk["n_panels"],
               "M1_cov_gp": get(g, "admit_m1", "cov"), "lacgp_cov_gp": get(g, "lac_node_gp", "cov"),
               "M1_sig_gp": pr and pr["n_sig_below_A"], "lacgp_sig_gp": pr and pr["n_sig_below_B"],
               "strata_gp": pr and pr["n_strata"],
               "M1_meandef_gp": pr and pr["mean_deficit_A"], "lacgp_meandef_gp": pr and pr["mean_deficit_B"],
               "diff_def_gp": pr and pr["diff_deficit"], "diff_def_gp_lo": pr and pr["diff_deficit_ci"][0],
               "diff_def_gp_hi": pr and pr["diff_deficit_ci"][1],
               "M1_size": get(g, "admit_m1", "size"), "lacgp_size": get(g, "lac_node_gp", "size"),
               "M1_leafres": get(g, "admit_m1", "leafres_correct"),
               "lacgp_leafres": get(g, "lac_node_gp", "leafres_correct"),
               "M1_abst": get(g, "admit_m1", "abst"), "lacgp_abst": get(g, "lac_node_gp", "abst"),
               "frac_coarsened": bk["frac_coarsened"], "depth_lost": bk["depth_lost"],
               "frac_no_supported_target": bk["frac_no_supported_target"],
               "M1_frac_inf_groups": bk["groups"].get("admit_m1", {}).get("frac_inf_groups"),
               "M1_frac_test_inf": get(o, "admit_m1", "frac_test_in_inf_group"),
               "frac_pi_changed_R": bk["frac_pi_changed_R"], "n_pi_changed_R": bk["n_pi_changed_R"],
               "ADMIT_cov_pi": get(p, "admit", "cov"), "lacproj_cov_pi": get(p, "lac_node_proj", "cov"),
               "ADMIT_sig_pi": pp and pp["n_sig_below_A"], "lacproj_sig_pi": pp and pp["n_sig_below_B"],
               "strata_pi": pp and pp["n_strata"], "diff_def_pi_admit": pp and pp["diff_deficit"],
               "diff_def_pi_admit_lo": pp and pp["diff_deficit_ci"][0],
               "diff_def_pi_admit_hi": pp and pp["diff_deficit_ci"][1],
               "ADMIT_size": get(p, "admit", "size"), "lacproj_size": get(p, "lac_node_proj", "size"),
               "M1_sig_pi": pm and pm["n_sig_below_A"], "M1_cov_pi": get(p, "admit_m1", "cov"),
               "diff_def_pi_m1": pm and pm["diff_deficit"],
               "M1cal_own_sig": get(o, "admit_m1cal", "n_sig_below"), "M1cal_own_types": get(o, "admit_m1cal", "n_types"),
               "old_M1_sig": get(oa, "admit_m1", "n_sig_below"), "old_M1_types": get(oa, "admit_m1", "n_types"),
               "old_lacproj_sig": get(oa, "lac_node_proj", "n_sig_below"),
               "old_lacproj_types": get(oa, "lac_node_proj", "n_types"),
               "old_M1_cov": get(oa, "admit_m1", "cov"), "old_M1_size": get(oa, "admit_m1", "size"),
               "old_lacproj_size": get(oa, "lac_node_proj", "size")}
        rows.append(row)
    # pooled over the headline runs
    for name in ("m1_vs_lacgp", "admit_vs_lacproj", "m1_vs_lacproj", "m1cal_vs_lacgp"):
        reps = [all_reps[r].get((name, "all_deletions")) for r in HEADLINE_RUNS if r in all_reps]
        reps = [r for r in reps if r]
        if len(reps) == 0:
            continue
        S = sum(r["S"] for r in reps)
        bdd = sum(r["b_sum_dd"] for r in reps) / sum(r["b_cnt"] for r in reps)
        pooled[name] = {"runs": [r for r in HEADLINE_RUNS if r in all_reps and all_reps[r].get((name, "all_deletions"))],
                        "n_strata": int(S), "sig_A": int(sum(r["sig_A"] for r in reps)),
                        "sig_B": int(sum(r["sig_B"] for r in reps)),
                        "mean_deficit_A": float(sum(r["sum_da"] for r in reps) / S),
                        "mean_deficit_B": float(sum(r["sum_db"] for r in reps) / S),
                        "diff_deficit": float(sum(r["sum_dd"] for r in reps) / S),
                        "diff_deficit_ci": [float(np.percentile(bdd, 2.5)), float(np.percentile(bdd, 97.5))]}
    oldp = {}
    for a in ("admit_m1", "lac_node_proj"):
        vals = [old.get(r, {}).get(a, {}).get("0.10") for r in HEADLINE_RUNS]
        vals = [v for v in vals if v]
        oldp[a] = {"n_sig_below": int(sum(v["n_sig_below"] for v in vals)), "n_types": int(sum(v["n_types"] for v in vals)),
                   "n_runs": len(vals)}
    sweep, fullp = [], []
    for run, res in all_res.items():
        for al in ("0.05", "0.10", "0.15", "0.20"):
            g = res["targets"]["gp"]["families"]["all_deletions"]["arms"]
            bk = res["bookkeeping"]["families"]["all_deletions"].get(al, {})
            pr = res["paired"].get("m1_vs_lacgp", {}).get("families", {}).get("all_deletions", {}).get(al)
            gg = lambda a, k: g.get(a, {}).get(al, {}).get(k)  # noqa: E731
            sweep.append({"run": run, "alpha": al, "frac_gp_ne_alpha010": bk.get("frac_gp_ne_alpha010"),
                          "M1_groups_per_block": bk.get("groups", {}).get("admit_m1", {}).get("groups_per_block"),
                          "M1_frac_inf_groups": bk.get("groups", {}).get("admit_m1", {}).get("frac_inf_groups"),
                          "frac_coarsened": bk.get("frac_coarsened"),
                          "frac_no_supported_target": bk.get("frac_no_supported_target"),
                          "M1_cov_gp": gg("admit_m1", "cov"), "lacgp_cov_gp": gg("lac_node_gp", "cov"),
                          "M1_size": gg("admit_m1", "size"), "lacgp_size": gg("lac_node_gp", "size"),
                          "M1_sig": pr and pr["n_sig_below_A"], "lacgp_sig": pr and pr["n_sig_below_B"],
                          "strata": pr and pr["n_strata"], "diff_def": pr and pr["diff_deficit"],
                          "diff_def_ci": pr and pr["diff_deficit_ci"]})
        for name in ("m1_vs_lacgp", "admit_vs_lacproj", "m1_vs_lacproj"):
            pr = res["paired"].get(name, {}).get("families", {}).get("full", {}).get("0.10")
            if pr:
                fullp.append({"run": run, "comparison": name, "n_strata": pr["n_strata"],
                              "sig_A": pr["n_sig_below_A"], "sig_B": pr["n_sig_below_B"],
                              "mean_deficit_A": pr["mean_deficit_A"], "mean_deficit_B": pr["mean_deficit_B"],
                              "diff_deficit": pr["diff_deficit"], "diff_deficit_ci": pr["diff_deficit_ci"],
                              "worst_A": pr["worst_A"], "worst_B": pr["worst_B"],
                              "strata": pr.get("strata")})
    return {"per_run": rows, "pooled_headline_runs": pooled, "old_round2_pooled": oldp,
            "headline_runs": HEADLINE_RUNS, "alpha_sweep": sweep, "full_panel": fullp}


def render_headline(H):
    f3 = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    f2 = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    md = ["<!-- generated by experiments/round3/summarize_round3.py; do not edit -->\n",
          "# Headline (alpha 0.10, all deletion panels)\n\n",
          "Primary: ADMIT-M1 (frozen map) vs marginal LAC on g_P, both scored on g_P; strata = (panel, g_P node) with "
          ">= 100 pooled test cells, identical for both arms. Deficit = max(0, 0.90 - coverage); diff = mean over "
          "strata of deficit(M1) - deficit(LAC on g_P), 95% patient-cluster bootstrap. Old = round 2 (ADMIT+M1 on its "
          "own target vs marginal projected LAC on pi_P).\n\n"]
    h = ["run", "pats", "M1 cov", "LACgp cov", "sig< M1/LACgp/strata", "mean def M1 / LACgp", "diff [95% CI]",
         "|C| M1/LACgp", "leaf M1/LACgp", "coarsened", "depth lost", "no sup.", "+inf groups / test", "pi_P chg (R)",
         "old sig< M1 / mLAC"]
    md.append("| " + " | ".join(h) + " |\n|" + "---|" * len(h) + "\n")
    for r in H["per_run"]:
        md.append("| " + " | ".join([
            r["run"], str(r["n_patients"]), f3(r["M1_cov_gp"]), f3(r["lacgp_cov_gp"]),
            f"{r['M1_sig_gp']}/{r['lacgp_sig_gp']}/{r['strata_gp']}",
            f"{f3(r['M1_meandef_gp'])} / {f3(r['lacgp_meandef_gp'])}",
            f"{f3(r['diff_def_gp'])} [{f3(r['diff_def_gp_lo'])}, {f3(r['diff_def_gp_hi'])}]",
            f"{f2(r['M1_size'])}/{f2(r['lacgp_size'])}", f"{f3(r['M1_leafres'])}/{f3(r['lacgp_leafres'])}",
            f3(r["frac_coarsened"]), f2(r["depth_lost"]), f3(r["frac_no_supported_target"]),
            f"{f3(r['M1_frac_inf_groups'])} / {f3(r['M1_frac_test_inf'])}", f3(r["frac_pi_changed_R"]),
            f"{r['old_M1_sig']}/{r['old_M1_types']} vs {r['old_lacproj_sig']}/{r['old_lacproj_types']}"]) + " |\n")
    md.append("\nSecondary on pi_P: ADMIT (no merging) vs marginal projected LAC; ADMIT-M1 on pi_P.\n\n")
    h = ["run", "ADMIT cov", "mLAC cov", "sig< ADMIT/mLAC/strata", "diff def ADMIT-mLAC [CI]", "|C| ADMIT/mLAC",
         "M1 cov on pi_P", "M1 sig< on pi_P", "M1-cal own sig<"]
    md.append("| " + " | ".join(h) + " |\n|" + "---|" * len(h) + "\n")
    for r in H["per_run"]:
        md.append("| " + " | ".join([
            r["run"], f3(r["ADMIT_cov_pi"]), f3(r["lacproj_cov_pi"]),
            f"{r['ADMIT_sig_pi']}/{r['lacproj_sig_pi']}/{r['strata_pi']}",
            f"{f3(r['diff_def_pi_admit'])} [{f3(r['diff_def_pi_admit_lo'])}, {f3(r['diff_def_pi_admit_hi'])}]",
            f"{f2(r['ADMIT_size'])}/{f2(r['lacproj_size'])}", f3(r["M1_cov_pi"]), str(r["M1_sig_pi"]),
            f"{r['M1cal_own_sig']}/{r['M1cal_own_types']}"]) + " |\n")
    md.append("\nPooled over the headline runs (" + ", ".join(H["headline_runs"]) + "):\n\n")
    for name, p in H["pooled_headline_runs"].items():
        md.append(f"- {name}: sig< {p['sig_A']} vs {p['sig_B']} of {p['n_strata']} strata; mean deficit "
                  f"{p['mean_deficit_A']:.4f} vs {p['mean_deficit_B']:.4f}; diff {p['diff_deficit']:.4f} "
                  f"[{p['diff_deficit_ci'][0]:.4f}, {p['diff_deficit_ci'][1]:.4f}] (runs: {len(p['runs'])})\n")
    o = H["old_round2_pooled"]
    md.append(f"- old (round 2): ADMIT+M1 {o['admit_m1']['n_sig_below']}/{o['admit_m1']['n_types']} vs marginal "
              f"projected LAC {o['lac_node_proj']['n_sig_below']}/{o['lac_node_proj']['n_types']}\n")
    md.append("\n## Full panel (alpha 0.10)\n\n| run | comparison | sig< A/B/strata | mean def A / B | diff [CI] | "
              "worst A | worst B |\n|---|---|---|---|---|---|---|\n")
    for r in H["full_panel"]:
        md.append(f"| {r['run']} | {r['comparison']} | {r['sig_A']}/{r['sig_B']}/{r['n_strata']} | "
                  f"{r['mean_deficit_A']:.3f} / {r['mean_deficit_B']:.3f} | {r['diff_deficit']:.3f} "
                  f"[{r['diff_deficit_ci'][0]:.3f}, {r['diff_deficit_ci'][1]:.3f}] | {r['worst_A']['node']} "
                  f"{r['worst_A']['cov']:.3f} | {r['worst_B']['node']} {r['worst_B']['cov']:.3f} |\n")
    md.append("\n## Alpha sweep (all deletion panels): how the frozen g_P moves with n_min\n\n| run | alpha | g_P != "
              "g_P(0.10) | M1 groups/block | +inf groups | coarsened | no sup. | M1 cov | LACgp cov | |C| M1/LACgp | "
              "sig< M1/LACgp/strata | diff def |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in H["alpha_sweep"]:
        md.append(f"| {r['run']} | {r['alpha']} | {f3(r['frac_gp_ne_alpha010'])} | {f2(r['M1_groups_per_block'])} | "
                  f"{f3(r['M1_frac_inf_groups'])} | {f3(r['frac_coarsened'])} | {f3(r['frac_no_supported_target'])} | "
                  f"{f3(r['M1_cov_gp'])} | {f3(r['lacgp_cov_gp'])} | {f2(r['M1_size'])}/{f2(r['lacgp_size'])} | "
                  f"{r['M1_sig']}/{r['lacgp_sig']}/{r['strata']} | {f3(r['diff_def'])} |\n")
    (R3 / "headline.md").write_text("".join(md))


# ------------------------------------------------------------------------------------------------------------------
# external decomposition + target curves (cell-level; read from round3/external/ when present)
# ------------------------------------------------------------------------------------------------------------------


EXT_ARMS = ["admit", "admit_m1", "admit_m1cal", "lac_node_proj", "lac_node_gp", "lac_filt", "aps_filt",
            "patch_src_backstop"]
EXT_KEYS = ("cov", "worst", "n_sig", "n_types", "mean_deficit")


def _ext_cell(m):
    """metrics3 dict (one arm) -> flat {gp: {...}, pi: {...}, size, leafres_correct, abst}"""
    out = {}
    for t in ("gp", "pi"):
        x = m.get(t) or {}
        out[t] = {k: x.get(k) for k in EXT_KEYS}
    for k in ("size", "leafres_correct", "abst"):
        out[k] = m.get(k)
    return out


def _ext_cell_mean(m):
    """summarize_runs dict (one arm, mean over seeds) -> the same flat layout"""
    out = {}
    for t in ("gp", "pi"):
        x = m.get(t) or {}
        out[t] = {k: (x.get(k) or {}).get("mean") if isinstance(x.get(k), dict) else None for k in EXT_KEYS}
    for k in ("size", "leafres_correct", "abst"):
        out[k] = (m.get(k) or {}).get("mean")
    return out


def summarize_external():
    """Every arm on both targets (g_P and pi_P), per cohort and averaged over the clean cohorts, for (ii) Orion-cal on
    the cohort, (iii) Orion test masked to P_k, (iv) 500-cell and 50/50 target recalibration, and the curve."""
    out, tsv = {}, []
    for f in sorted((R3 / "external").glob("external_decomposition_*.json")):
        bb = f.stem.replace("external_decomposition_", "")
        d = json.loads(f.read_text())
        cfile = R3 / "external" / f"target_curve_{bb}.json"
        cur = json.loads(cfile.read_text()) if cfile.exists() else None
        clean = [k for k, r in d["cohorts"].items() if not r["placeholder"] and k not in ("danenberg", "risom")]
        per = {}
        for k, r in d["cohorts"].items():
            blocks = {}
            for blk, key in (("ii", "ii_orion_cal_on_cohort"), ("iii", "iii_orion_test_masked_to_Pk")):
                m = r[key]
                blocks[blk] = {a: _ext_cell(m["arms"][a]) for a in EXT_ARMS if a in m["arms"]}
                inf = m["info"]["infinite_groups"]
                blocks[blk]["_info"] = {"no_supported_target": m["info"]["book"]["frac_no_supported_target"],
                                        "coarsened": m["info"]["book"]["frac_coarsened"],
                                        "S_P": m["info"]["S_P"], "n_supported_groups": len(m["info"]["S_P"]),
                                        "n_inf_groups": {a: len(v) for a, v in inf.items()},
                                        "frac_inf_groups_admit_m1": m["info"]["frac_inf_groups_admit_m1"],
                                        "pi_P_changed_by_R": m["info"]["pi_P_changed_by_R"]}
            for blk, mode in (("iv500", "slice500"), ("iv_split50", "split50")):
                sm = r["iv_target_recalibration"][mode]["summary"]
                blocks[blk] = {a: _ext_cell_mean(sm[a]) for a in EXT_ARMS if a in sm}
                blocks[blk]["_info"] = {"no_supported_target": sm["_book"].get("frac_no_supported_target"),
                                        "coarsened": sm["_book"].get("frac_coarsened"),
                                        "n_supported_groups": sm["_n_supported_groups"],
                                        "frac_inf_groups_admit_m1": sm["_frac_inf_groups_admit_m1"]}
            if cur and k in cur["cohorts"]:
                for n, c in cur["cohorts"][k]["curve"].items():
                    sm = c["summary"]
                    blocks[f"curve_n{n}"] = {a: _ext_cell_mean(sm[a]) for a in EXT_ARMS if a in sm}
                    blocks[f"curve_n{n}"]["_info"] = {"no_supported_target": sm["_book"].get("frac_no_supported_target"),
                                                      "coarsened": sm["_book"].get("frac_coarsened"),
                                                      "n_supported_groups": sm["_n_supported_groups"],
                                                      "frac_inf_groups_admit_m1": sm["_frac_inf_groups_admit_m1"]}
            per[k] = blocks
            for blk, B_ in blocks.items():
                for a_, m in B_.items():
                    if a_.startswith("_"):
                        continue
                    for t in ("gp", "pi"):
                        tsv.append({"backbone": bb, "cohort": k, "clean": k in clean, "block": blk, "arm": a_,
                                    "target": t, **m[t], "size": m["size"], "leafres_correct": m["leafres_correct"],
                                    "abst": m["abst"], "no_supported_target": B_["_info"]["no_supported_target"],
                                    "frac_inf_groups_admit_m1": B_["_info"].get("frac_inf_groups_admit_m1"),
                                    "n_inf_groups_arm": B_["_info"].get("n_inf_groups", {}).get(a_)})
        mean = {}
        for blk in per[clean[0]]:
            mean[blk] = {}
            for a_ in EXT_ARMS:
                if a_ not in per[clean[0]][blk]:
                    continue
                rows_ = [per[k][blk][a_] for k in clean if a_ in per[k].get(blk, {})]
                avg = lambda vals: float(np.mean(vals)) if vals else None  # noqa: E731
                mean[blk][a_] = {t: {key: avg([x[t][key] for x in rows_ if x[t][key] is not None]) for key in EXT_KEYS}
                                 for t in ("gp", "pi")}
                for key in ("size", "leafres_correct", "abst"):
                    mean[blk][a_][key] = avg([x[key] for x in rows_ if x[key] is not None])
            mean[blk]["_info"] = {key: float(np.mean([per[k][blk]["_info"][key] for k in clean
                                                      if per[k][blk]["_info"].get(key) is not None]))
                                  for key in ("no_supported_target", "coarsened", "n_supported_groups",
                                              "frac_inf_groups_admit_m1")
                                  if any(per[k][blk]["_info"].get(key) is not None for k in clean)}
        out[bb] = {"clean_cohorts": clean, "cohorts": per, "mean": mean,
                   "layout": "mean.<block>.<arm>.{gp,pi}.{cov,worst,n_sig,n_types,mean_deficit} + size, "
                             "leafres_correct, abst (target-independent); <block>._info. Blocks: ii (Orion crc_cal "
                             "calibrates, cohort tested), iii (Orion crc_test masked to P_k), iv500 / iv_split50 (target "
                             "recalibration), curve_n<n>. patch_src_backstop = full-panel calibration (old method)."}
    write_json(R3 / "external" / "external_summary.json", out)
    write_tsv(R3 / "external" / "external_per_cohort.tsv", tsv)
    f3 = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    f2 = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    md = ["\n## External cohorts (cell-level; mean over clean cohorts; every arm on both targets)\n\n"]
    for bb, e in out.items():
        md.append(f"### {bb} ({len(e['clean_cohorts'])} clean cohorts: {', '.join(e['clean_cohorts'])})\n\n")
        for blk in ["ii", "iii", "iv500"] + [f"curve_n{n}" for n in (0, 50, 100, 200, 500, 2000)]:
            if blk not in e["mean"]:
                continue
            inf = e["mean"][blk]["_info"]
            md.append(f"**{blk}** (no supported target {f3(inf.get('no_supported_target'))}, supported groups "
                      f"{f2(inf.get('n_supported_groups'))}, ADMIT-M1 +inf groups {f3(inf.get('frac_inf_groups_admit_m1'))})"
                      "\n\n| arm | cov g_P | worst g_P | sig g_P | cov pi_P | worst pi_P | sig pi_P | |C| | leaf |\n"
                      "|---|---|---|---|---|---|---|---|---|\n")
            for a_ in EXT_ARMS:
                x = e["mean"][blk].get(a_)
                if not x:
                    continue
                md.append(f"| {a_} | {f3(x['gp']['cov'])} | {f3(x['gp']['worst'])} | {f2(x['gp']['n_sig'])} | "
                          f"{f3(x['pi']['cov'])} | {f3(x['pi']['worst'])} | {f2(x['pi']['n_sig'])} | {f2(x['size'])} | "
                          f"{f3(x['leafres_correct'])} |\n")
            md.append("\n")
    return out, "".join(md)


# ------------------------------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="schurch,orion,within,pairs,headline")
    ap.add_argument("--heads", default="xgb,mlp,logistic")
    ap.add_argument("--cohorts", default=",".join(WITHIN))
    args = ap.parse_args()
    only = set(args.only.split(","))
    heads = args.heads.split(",")
    all_res, all_reps = {}, {}
    rows_all, paired_all, strata_rows = [], [], []
    S = {"runs": {}, "engine": "engine3"}
    S3 = R3 / "summary_round3.json"
    if "schurch" in only or "orion" in only:
        if "schurch" in only:
            for head in ("logistic", "xgb"):
                L = load_schurch(f"schurch_{head}")
                if L is None:
                    continue
                d, meta, ep, seeds = L
                res, reps = summarize_run3(d, meta, ep, len(seeds), strata_rows, f"schurch_{head}")
                res["seeds"], res["splits"] = seeds, meta["splits"]
                S["runs"][f"schurch_{head}"] = res
                all_res[f"schurch_{head}"], all_reps[f"schurch_{head}"] = res, reps
                rows_all += family_rows("schurch", head, "within_patient_split", res, len(seeds))
                paired_all += paired_rows(f"schurch_{head}", "schurch", head, "within_patient_split", res)
                print("summarized", f"schurch_{head}", len(seeds), "splits", flush=True)
        if "orion" in only:
            for prot in ("lopo13", "lopo7", "exchangeable"):
                L = load_orion(prot)
                if L is None:
                    continue
                d, meta, _, nb = L
                res, reps = summarize_run3(d, meta, None, nb, strata_rows, f"orion_{prot}")
                S["runs"][f"orion_{prot}"] = res
                all_res[f"orion_{prot}"], all_reps[f"orion_{prot}"] = res, reps
                rows_all += family_rows("orion", "round0", prot, res, 1)
                paired_all += paired_rows(f"orion_{prot}", "orion", "round0", prot, res)
                print("summarized", f"orion_{prot}", flush=True)
        write_json(S3, S)
    cohort_rows = []
    if "within" in only:
        for c in args.cohorts.split(","):
            for head in heads:
                L = load_within(c, head)
                if L is None:
                    continue
                d, meta, ep, seeds = L
                res, reps = summarize_run3(d, meta, ep, len(seeds), strata_rows, f"{c}:{head}")
                res.update({"cohort": c, "head": head, "protocol": "within_patient_split", "seeds": seeds,
                            "test_accuracy_full_panel_by_seed": meta["test_accuracy_full_panel_by_seed"],
                            "transform": meta["transform"], "markers": meta["markers"],
                            "excluded_labels": meta["excluded_labels"], "label_counts": meta["label_counts"]})
                reg = json.loads((HERE.parents[1] / "configs" / "cohort_registry.json").read_text())[c]
                nexp = len(reg["splits"]["seeds"])
                res["completeness"] = f"{len(seeds)}/{nexp} seeds"
                status = "final" if len(seeds) >= nexp else f"partial {len(seeds)}/{nexp} seeds"
                write_json(C3 / c / f"summary_within_{head}.json", res)
                all_res[f"{c}:{head}"], all_reps[f"{c}:{head}"] = res, reps
                rr = family_rows(c, head, "within_patient_split", res, len(seeds), status)
                cohort_rows += rr
                rows_all += rr
                paired_all += paired_rows(f"{c}:{head}", c, head, "within_patient_split", res)
                print("summarized", c, head, status, flush=True)
    if "pairs" in only:
        for p in PAIRS:
            pr_rows = []
            summarize_pair3(p, heads, pr_rows, paired_all, strata_rows)
            cohort_rows += pr_rows
            rows_all += pr_rows
            print("summarized pair", p, len(pr_rows), "rows", flush=True)
    if cohort_rows:
        write_tsv(C3 / "summary.tsv", cohort_rows, SUMMARY_COLS)
    if rows_all and only >= {"schurch", "orion", "within", "pairs"}:
        write_tsv(R3 / "summary_all.tsv", rows_all, SUMMARY_COLS)
    if paired_all:
        write_tsv(R3 / ("paired.tsv" if only >= {"schurch", "orion", "within", "pairs"} else
                        f"paired_{'_'.join(sorted(only))}.tsv"), paired_all)
    if strata_rows:
        write_tsv(R3 / ("strata_common.tsv" if only >= {"schurch", "orion", "within", "pairs"} else
                        f"strata_common_{'_'.join(sorted(only))}.tsv"), strata_rows)
    if "headline" in only and all_res:
        H = headline({k: v for k, v in all_res.items() if "all_deletions" in v["targets"]["gp"]["families"]},
                     all_reps)
        write_json(R3 / "headline.json", H)
        render_headline(H)
        if (R3 / "external").exists():
            _, md = summarize_external()
            with open(R3 / "headline.md", "a") as fh:
                fh.write(md)
        print((R3 / "headline.md").read_text())


if __name__ == "__main__":
    main()
