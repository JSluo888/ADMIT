#!/usr/bin/env python3
"""Reduce the deletion / external JSONs to the summaries the report tables read.

Writes $ADMIT_RUNS/summary/deletion_summary.json and external_summary.json.
Every derived number (family means, worst strata, gaps) is computed here, from the run JSONs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, write_json  # noqa: E402

KEYS = ["exact_cov", "leaf_cov", "anc_cov", "size", "singleton", "leaf_resolved", "chain_leaf", "abstain",
        "inadm_mass", "frac_ident", "exact_or_abstain", "exact_cov_supported", "abstain_unident"]
MIN_STRATUM_N = 100


def _ok(v):
    return v is not None and np.isfinite(v)


def _nm(vals):
    v = [x for x in vals if _ok(x)]
    return float(np.mean(v)) if v else None


def beta_lo(n_c, alpha=0.10, q=0.025):
    """Lower q-quantile of the calibration-conditional coverage of split conformal with n_c
    calibration points: Beta(k, n_c + 1 - k), k = ceil((n_c + 1)(1 - alpha))."""
    if not n_c or n_c < 1:
        return None
    k = int(np.ceil((n_c + 1) * (1 - alpha) - 1e-12))
    if k > n_c:
        return None
    return float(beta_dist.ppf(q, k, n_c + 1 - k))


def stratum_stats(metrics, n_cal=None, target=0.90):
    """Worst supported stratum, and strata whose coverage is significantly below what split
    conformal predicts: test Wilson upper bound < lower 2.5% of Beta(k, n_c+1-k), with n_c the
    stratum's panel-matched calibration count (strata the calibration does not support are
    compared with the nominal 0.90)."""
    ps = metrics.get("per_stratum", {})
    rows = [(k, v) for k, v in ps.items() if k != "<unidentifiable>" and v["n"] >= MIN_STRATUM_N]
    if not rows:
        return {"min_stratum_cov": None, "argmin_stratum": None, "n_strata": 0, "n_sig_under": 0}
    k, v = min(rows, key=lambda kv: kv[1]["exact_coverage"])
    sig = []
    for kk, vv in rows:
        lo = beta_lo((n_cal or {}).get(kk, {}).get("n_c")) if n_cal else None
        thr = lo if lo is not None else target
        if vv["ci_wilson_cell"][1] < thr:
            sig.append(kk)
    clu = [kk for kk, vv in rows if vv.get("ci_cluster") and vv["ci_cluster"][1] < target]
    return {"min_stratum_cov": v["exact_coverage"], "argmin_stratum": k, "n_strata": len(rows),
            "n_sig_under": len(sig), "sig_under": sig,
            "n_cluster_under": len(clu), "cluster_under": clu}


def family_of(entry):
    fam = entry["family"]
    if fam == "random_retention":
        return f"random{entry['name'].split('_')[0][4:]}"
    return fam


def deletion():
    out = {}
    for f in sorted((RUNS / "deletion").glob("deletion_*_*.json")):
        if "debug" in f.stem or "FAKE" in f.stem:
            continue
        d = json.loads(f.read_text())
        key = f"{f.stem.split('_')[1]}|{d['protocol']}"
        panels = {}
        fam_rows = {}
        for name, e in d["panels"].items():
            row = {"family": family_of(e), "deleted": e["deleted"], "alias_of": e.get("alias_of"), "arms": {}}
            for arm, a in e["arms"].items():
                row["arms"][arm] = {**{k: a["headline"].get(k) for k in KEYS}, **stratum_stats(a["metrics"], e.get("pm_strata")),
                                    "exact_cov_ci_cluster": a["headline"].get("exact_cov_ci_cluster")}
            panels[name] = row
            # family means include aliased panels (e.g. lineage group {CD8} == single deletion of CD8)
            fam_rows.setdefault(row["family"], []).append(row)
        fams = {}
        for fam, rows in fam_rows.items():
            fams[fam] = {"n_panels": len(rows), "arms": {}}
            for arm in rows[0]["arms"]:
                agg = {}
                for k in KEYS + ["min_stratum_cov", "n_sig_under"]:
                    vals = [r["arms"][arm][k] for r in rows if _ok(r["arms"][arm][k])]
                    agg[k] = float(np.mean(vals)) if vals else None
                agg["worst_min_stratum_cov"] = float(min(r["arms"][arm]["min_stratum_cov"] for r in rows
                                                         if r["arms"][arm]["min_stratum_cov"] is not None)) \
                    if any(r["arms"][arm]["min_stratum_cov"] is not None for r in rows) else None
                agg["total_sig_under"] = int(sum(r["arms"][arm]["n_sig_under"] for r in rows))
                agg["total_strata"] = int(sum(r["arms"][arm]["n_strata"] for r in rows))
                fams[fam]["arms"][arm] = agg
        # all non-full unique panels
        allrows = [r for rows in fam_rows.values() for r in rows if r["family"] != "full" and r["alias_of"] is None]
        fams["all_deletions"] = {"n_panels": len(allrows), "arms": {}}
        for arm in allrows[0]["arms"]:
            agg = {}
            for k in KEYS + ["min_stratum_cov"]:
                vals = [r["arms"][arm][k] for r in allrows if _ok(r["arms"][arm][k])]
                agg[k] = float(np.mean(vals)) if vals else None
            agg["total_sig_under"] = int(sum(r["arms"][arm]["n_sig_under"] for r in allrows))
            agg["total_cluster_under"] = int(sum(r["arms"][arm].get("n_cluster_under", 0) for r in allrows))
            agg["total_strata"] = int(sum(r["arms"][arm]["n_strata"] for r in allrows))
            fams["all_deletions"]["arms"][arm] = agg
        out[key] = {"source": str(f), "n_cal": d["n_cal"], "n_test": d["n_test"], "split": d["split"],
                    "panels": panels, "families": fams,
                    "marker_by_celltype_moved_up": d["marker_by_celltype_moved_up"],
                    "wall_clock_total_s": d["wall_clock_total_s"]}
    write_json(RUNS / "summary" / "deletion_summary.json",
               {"min_stratum_n": MIN_STRATUM_N, "runs": out})


def external():
    out = {}
    for f in sorted((RUNS / "external").glob("external_decomposition_*.json")):
        d = json.loads(f.read_text())
        rows = {}
        for k, r in d["cohorts"].items():
            i_on = r["i_source_patch"]["backstop_on"]["headline"]
            i_off = r["i_source_patch"]["backstop_off"]["headline"]
            ii = r["ii_panel_matched"]["backstop_off"]["headline"]
            ii_bs = r["ii_panel_matched"]["backstop_on"]["headline"]
            iii = r["iii_orion_test_masked_to_Pk"]["panel_matched"]["headline"]
            iv5 = r["iv_target_recalibration"]["slice500"]["summary"]
            iv50 = r["iv_target_recalibration"]["split50"]["summary"]
            rows[k] = {
                "panel_size": len(r["panel"]), "n_cells": r["n_cells"], "xarc_placeholder": r["xarc_placeholder"],
                "frac_identifiable": ii["frac_ident"],
                "i_src_bs_on": i_on, "i_src_bs_off": i_off, "ii_pm": ii, "ii_pm_bs_on": ii_bs, "iii_pm_orion": iii,
                "iv_slice500": {a: {m: iv5[a][m] for m in ("exact_cov", "size", "abstain", "exact_cov_supported")}
                                for a in iv5},
                "iv_split50": {a: {m: iv50[a][m] for m in ("exact_cov", "size", "abstain", "exact_cov_supported")}
                               for a in iv50},
                # decomposition of the exact-coverage gap of old PATCH to the 0.90 target
                "gap_total_i": 0.90 - i_on["exact_cov"],
                "panel_part_recovered_ii_minus_i": ii["exact_cov"] - i_on["exact_cov"],
                "transfer_part_iii_minus_ii": iii["exact_cov"] - ii["exact_cov"],
                "target_recal500_minus_ii": iv5["target_recal"]["exact_cov"]["mean"] - ii["exact_cov"],
                "arms_orion_cal_on_cohort": r["arms_orion_cal_on_cohort"],
                "pm_infinite_strata": r.get("pm_infinite_strata", []),
                "projected_counts": r["projected_counts"],
                "calibration_mask_panel": r.get("calibration_mask_panel_Pk_cap_orion"),
                "markers_unused_by_scorer": r.get("markers_unused_by_scorer"),
                "flag": r.get("flag"), "local": r.get("local"),
                "n_unidentifiable": ii.get("n_unident"),
                "iii_ci_cluster": iii.get("exact_cov_ci_cluster"),
            }
        from common import SEPARATE_COHORTS, LOCAL_COHORTS
        B5_ROSTER = ["hoch", "schurch", "immucan", "liu", "cords", "hartmann", "maps", "tietscher"]
        cohorts = [k for k in rows if not rows[k]["xarc_placeholder"] and k not in SEPARATE_COHORTS]
        unf = ["lac", "aps", "raps", "hcc", "scconform", "crsvp"]
        inadm = {}
        for name, roster in (("b5_roster_8", B5_ROSTER), ("local_7", LOCAL_COHORTS),
                             ("all_available", list(rows))):
            ks = [k for k in roster if k in rows]
            ks_np = [k for k in ks if not rows[k]["xarc_placeholder"]]
            inadm[name] = {"cohorts": ks, "cohorts_no_placeholder": ks_np, "by_method": {
                m: {"mean_all": float(np.mean([rows[k]["arms_orion_cal_on_cohort"][m]["inadm_mass"] for k in ks])),
                    "mean_no_placeholder": float(np.mean([rows[k]["arms_orion_cal_on_cohort"][m]["inadm_mass"] for k in ks_np]))
                    if ks_np else None,
                    "per_cohort": {k: rows[k]["arms_orion_cal_on_cohort"][m]["inadm_mass"] for k in ks}}
                for m in unf}}
        mean = lambda key, sub: float(np.mean([rows[k][key][sub] for k in cohorts])) if cohorts else None
        out[f.stem.replace("external_decomposition_", "")] = {
                              "backbone": d["backbone"], "orion_reference": d.get("orion_reference"),
                              "frac_set_inadmissible_unfiltered": inadm,
                              "source": str(f), "rows": rows, "cohorts_in_mean": cohorts,
                              "mean": {"i_src_bs_on": mean("i_src_bs_on", "exact_cov"),
                                       "i_src_bs_off": mean("i_src_bs_off", "exact_cov"),
                                       "ii_pm": mean("ii_pm", "exact_cov"),
                                       "iii_pm_orion": mean("iii_pm_orion", "exact_cov"),
                                       "iv_slice500": float(np.mean([rows[k]["iv_slice500"]["target_recal"]["exact_cov"]["mean"]
                                                                     for k in cohorts])) if cohorts else None,
                                       "iv_split50": float(np.mean([rows[k]["iv_split50"]["target_recal"]["exact_cov"]["mean"]
                                                                    for k in cohorts])) if cohorts else None},
                              "wall_clock_total_s": d["wall_clock_total_s"]}
    write_json(RUNS / "summary" / "external_summary.json", out)




def schurch(B=2000, seed=0):
    out = {}
    for f in sorted((RUNS / "schurch").glob("schurch_deletion_*.json")):
        if "debug" in f.stem:
            continue
        d = json.loads(f.read_text())
        head = d["head"]
        names = list(d["splits"][0]["panels"].keys())
        panels = {}
        for pn in names:
            ents = []
            for sp in d["splits"]:
                e = sp["panels"][pn]
                if "alias_of" in e:
                    e = {**sp["panels"][e["alias_of"]], "family": e["family"], "alias_of": e["alias_of"]}
                ents.append(e)
            fam = ents[0]["family"]
            if fam == "random_retention":
                fam = f"random{pn.split('_')[0][4:]}"
            row = {"family": fam, "alias_of": ents[0].get("alias_of"), "deleted": ents[0]["deleted"],
                   "point_accuracy": float(np.mean([e["point_accuracy"] for e in ents])), "arms": {}}
            for arm in ents[0]["arms"]:
                per_split = [e["arms"][arm] for e in ents]
                pp = [v["exact_cov"] for e in per_split for v in e["equal_patient"]["per_patient"].values()
                      if v["exact_cov"] is not None and np.isfinite(v["exact_cov"])]
                rng = np.random.default_rng(seed)
                ci = None
                if pp:
                    a = np.array(pp)
                    bs = a[rng.integers(0, len(a), size=(B, len(a)))].mean(1)
                    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
                row["arms"][arm] = {
                    "ep_exact_cov_mean_over_splits": _nm([s["equal_patient"]["exact_cov"]["value"] for s in per_split]),
                    "ep_exact_cov_by_split": [s["equal_patient"]["exact_cov"]["value"] for s in per_split],
                    "ep_exact_cov_ci_by_split": [s["equal_patient"]["exact_cov"]["ci_patient_bootstrap"] for s in per_split],
                    "ep_exact_cov_pooled_pairs": float(np.mean(pp)) if pp else None,
                    "ep_exact_cov_pooled_pairs_ci": ci, "n_patient_split_pairs": len(pp),
                    "min_patient_exact_cov": float(np.min(pp)) if pp else None,
                    "ep_size": _nm([s["equal_patient"]["size"]["value"] for s in per_split]),
                    "ep_singleton": _nm([s["equal_patient"]["singleton"]["value"] for s in per_split]),
                    "ep_abstain": _nm([s["equal_patient"]["abstain"]["value"] for s in per_split]),
                    "cell_exact_cov": _nm([s["headline"]["exact_cov"] for s in per_split]),
                    "cell_exact_cov_ci_cluster_by_split": [s["headline"]["exact_cov_ci_cluster"] for s in per_split],
                    "inadm_mass": _nm([s["headline"]["inadm_mass"] for s in per_split]),
                    "leaf_cov": _nm([s["headline"]["leaf_cov"] for s in per_split]),
                    "anc_cov": _nm([s["headline"]["anc_cov"] for s in per_split]),
                }
            panels[pn] = row
        fams = {}
        for pn, r in panels.items():
            fams.setdefault(r["family"], []).append(r)
        uniq = [r for r in panels.values() if r["family"] != "full" and r["alias_of"] is None]
        fams["all_deletions"] = uniq
        fam_out = {}
        for fam, rows in fams.items():
            fam_out[fam] = {"n_panels": len(rows), "arms": {
                arm: {k: _nm([r["arms"][arm][k] for r in rows])
                      for k in ("ep_exact_cov_mean_over_splits", "ep_size", "ep_abstain", "ep_singleton",
                                "inadm_mass", "min_patient_exact_cov")}
                for arm in rows[0]["arms"]}}
        out[head] = {"source": str(f), "panels": panels, "families": fam_out,
                     "splits": [{k: s[k] for k in ("seed", "n_train", "n_cal", "n_test", "head_note",
                                                   "point_accuracy_full", "wall_clock_s")} for s in d["splits"]],
                     "excluded_labels": d["excluded_labels"], "label_counts_17": d["label_counts_17"],
                     "wall_clock_total_s": d["wall_clock_total_s"]}
    write_json(RUNS / "summary" / "schurch_summary.json", out)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "schurch":
    schurch()


if __name__ == "__main__" and len(sys.argv) == 1:
    deletion()
    external()
    if any((RUNS / "schurch").glob("schurch_deletion_*.json")):
        schurch()
