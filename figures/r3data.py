"""Data access for the figure scripts: summaries written by experiments/round3 ($ADMIT_RUNS/round3).

A run summary has targets.{gp,pi,own}; each target holds panels -> arms -> alpha -> cell summary and
families -> arms -> alpha. `run(key)` returns that layout plus provenance (src path, field prefix).
"""
from __future__ import annotations

import csv
from pathlib import Path

import admit_style as S

R3 = S.RUNS / "round3"
SUMMARY = R3 / "summary_round3.json"
# optional fallback for the Orion run when summary_round3.json lacks it (written by summarize_round3.summarize_run3)
ORION_CACHE = S.ASSETS / "round3_cache" / "summary_round3_orion_lopo13.json"
PAIRED = R3 / "paired.tsv"
META = {"schurch_logistic": R3 / "cohort_schurch_logistic_s0-9_meta.json",
        "schurch_xgb": R3 / "cohort_schurch_xgb_s0-4_meta.json",
        "orion_lopo13": R3 / "orion_lopo13_meta.json"}
TARGET = "gp"
ALPHA_KEY = "0.10"

# target-matched marginal comparator
S.ARMS.setdefault("lac_node_gp", ("Marginal LAC, same target", S.AQUA, "D"))


def cohort_file(c, head="xgb"):
    return R3 / "cohorts" / c / f"summary_within_{head}.json"


def cohort_meta_file(c, head="xgb"):
    return R3 / "cohorts" / c / f"within_{head}_seed0_meta.json"


def run(key, target=TARGET):
    """-> dict(panels, families, n_patients, splits, alphas, src, field, meta, meta_src)."""
    if key in META:
        summ = S.load_json(str(SUMMARY))
        if key in summ["runs"]:
            r, src, field = summ["runs"][key], SUMMARY, f"runs.{key}.targets.{target}"
        elif key == "orion_lopo13" and ORION_CACHE.exists():
            r, src, field = S.load_json(str(ORION_CACHE)), ORION_CACHE, f"targets.{target}"
        else:
            raise KeyError(f"{key} not in {SUMMARY}")
        meta, meta_src = S.load_json(str(META[key])), META[key]
    else:
        src = cohort_file(key)
        r, field = S.load_json(str(src)), f"targets.{target}"
        meta_src = cohort_meta_file(key)
        meta = S.load_json(str(meta_src)) if meta_src.exists() else {}
    t = r["targets"][target]
    return {"panels": t["panels"], "families": t["families"], "n_patients": r["n_patients"],
            "splits": r.get("splits", []), "seeds": r.get("seeds", []), "alphas": r["alphas"],
            "n_blocks": r.get("n_blocks"), "src": src, "field": field, "meta": meta, "meta_src": meta_src,
            "raw": r, "bookkeeping": r.get("bookkeeping", {})}


def paired(run_name, comparison="m1_vs_lacgp", family="all_deletions", alpha="0.1"):
    with open(PAIRED) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if (r["run"] == run_name and r["comparison"] == comparison and r["family"] == family
                    and abs(float(r["alpha"]) - float(alpha)) < 1e-9):
                return r
    return None


def paired_name(key):
    return key if key in META else f"{key}:xgb"
