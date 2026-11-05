#!/usr/bin/env python3
"""Within-cohort paired deletion with patient splits (Fig 2 replication) on a standard-schema cohort.

  python experiments/cohorts/run_within.py --cohort cords2024_nsclc_imc --head xgb --seed 0

One process = one (cohort, head, split seed). The fixed panel bank (cohorts/<cohort>/panel_bank.json, committed as
configs/panel_banks/cohorts/<cohort>.json; copy it with experiments/seed_runs.py) must exist. For every bank panel P the frozen head rescoring
the calibration patients' cells masked to P feeds the round-2 engine (all arms, alphas 0.05-0.20); the test cells are
masked the same way. Output: cohorts/<cohort>/within_<head>_seed<k>.npz with the additive accumulator over the test
patients of this split (acc[panel, arm, alpha, test patient, node, stat]) + _meta.json.

Optional --test-cohort/--site-col style site shift is handled by run_pair.py, not here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import std  # noqa: E402
from engine import ALPHAS, NSTAT, STATS, all_arm_names, evaluate_block  # noqa: E402
from admit.panel_matched import Ontology  # noqa: E402


def unique_panels(bank):
    uniq, alias = [], {}
    for p in bank["panels"]:
        key = tuple(p["markers"])
        for q in uniq:
            if tuple(q["markers"]) == key:
                alias[p["name"]] = q["name"]
                break
        else:
            uniq.append(p)
    return uniq, alias


def run_panels(onto, head, markers, panels, Zc, yc, gc, Zt, yt, pslot_t, n_pat_t, arms, arm_index, progress=""):
    """-> acc (n_panels, n_arms, n_alpha, n_pat_t, K, NSTAT) float32, ep (n_panels, n_arms, n_alpha) equal-patient
    coverage, reports of the first block."""
    Pc_full = head.predict17(Zc)
    acc = np.zeros((len(panels), len(arms), len(ALPHAS), n_pat_t, onto.K, NSTAT), dtype=np.float32)
    ep = np.full((len(panels), len(arms), len(ALPHAS)), np.nan)
    epsize = np.full_like(ep, np.nan)
    reports = {}
    src_cache = {}
    for pi, p in enumerate(panels):
        keep = set(p["markers"])
        miss = [j for j, m in enumerate(markers) if m not in keep]
        if miss:
            zc, zt = Zc.copy(), Zt.copy()
            zc[:, miss] = 0.0
            zt[:, miss] = 0.0
        else:
            zc, zt = Zc, Zt
        Pc, Pt = head.predict17(zc), head.predict17(zt)
        tmp = np.zeros((1, len(arms), len(ALPHAS), n_pat_t, onto.K, NSTAT))
        rep = evaluate_block(tmp, 0, onto, p["markers"], Pc, yc, Pt, yt, pslot_t, n_pat_t, Pc_full, True, gc=gc,
                             arm_index=arm_index, src_cache=src_cache)
        acc[pi] = tmp[0]
        reports[p["name"]] = rep
        h, n = tmp[0, :, :, :, :, 0].sum(3), tmp[0, :, :, :, :, 1].sum(3)
        sz, la = tmp[0, :, :, :, :, 5].sum(3), tmp[0, :, :, :, :, 4].sum(3)
        with np.errstate(invalid="ignore", divide="ignore"):
            ep[pi] = np.nanmean(np.where(n > 0, h / n, np.nan), axis=2)
            epsize[pi] = np.nanmean(np.where(la > 0, sz / la, np.nan), axis=2)
        if progress and pi % 10 == 0:
            print(f"{progress} panel {pi + 1}/{len(panels)}", flush=True)
    return acc, ep, epsize, reports


def compute(d, bank, reg, cohort, head_kind, seed, onto, limit=0):
    """Pure computation. d: load_std() dict. Returns (arrays for np.savez, meta dict)."""
    t0 = time.time()
    assert bank["full_panel"] == d["markers"], "bank full panel does not match the cohort features"
    panels, alias = unique_panels(bank)
    if limit:
        panels = panels[:limit]
    x, y, pcode, pkeys = d["x"], d["y"], d["pcode"], d["pkeys"]
    groups = std.patient_groups(pcode, len(pkeys))
    sp = std.patient_splits(np.arange(len(pkeys)), reg["splits"]["fractions"], [seed])[0]
    rng = np.random.default_rng(seed)
    tr = std.train_index(groups, sp["train"], reg.get("train_cap", 2000), rng)
    cap = reg.get("eval_cap") if d["n_cells"] > reg.get("eval_cap_above", 1_000_000) else None
    erng = np.random.default_rng([seed, 1])
    ci = std.capped_index(groups, sp["cal"], cap, erng)
    ti = std.capped_index(groups, sp["test"], cap, erng)
    tfit = std.fit_transform(x, reg["transform"], tr)
    mu, sd = x[tr].mean(0), x[tr].std(0)
    sd[sd == 0] = 1.0
    z = lambda rows: ((x[rows] - mu) / sd).astype(np.float32)  # noqa: E731
    head = std.Head(head_kind).fit(z(tr), y[tr], seed)
    Zc, Zt = z(ci), z(ti)
    test_codes = np.unique(pcode[ti])
    pslot = np.searchsorted(test_codes, pcode[ti])
    arms = all_arm_names(True)
    arm_index = {a: i for i, a in enumerate(arms)}
    acc, ep, epsize, reports = run_panels(onto, head, d["markers"], panels, Zc, y[ci], pkeys[pcode[ci]], Zt, y[ti],
                                          pslot, len(test_codes), arms, arm_index,
                                          progress=f"[{cohort} {head_kind} s{seed}]")
    Pfull = head.predict17(Zt)
    arrays = dict(acc=acc, ep_cov=ep[None], ep_size=epsize[None], panels=np.array([p["name"] for p in panels]),
                  patients=pkeys[test_codes], arms=np.array(arms), alphas=np.array(ALPHAS), stats=np.array(STATS),
                  nodes=np.array(onto.names), seeds=np.array([seed]))
    meta = {
        "cohort": cohort, "protocol": "within", "head": head_kind, "seed": seed,
        "transform": d["transform"], "transform_fit": tfit, "markers": d["markers"],
        "dropped_columns": d["dropped_columns"],
        "splits": [{"seed": seed, "train": pkeys[sp["train"]].tolist(), "cal": pkeys[sp["cal"]].tolist(),
                    "test": pkeys[sp["test"]].tolist(), "n_train": int(len(tr)), "n_cal": int(len(ci)),
                    "n_test": int(len(ti)), "eval_cap_per_patient": cap}],
        "head_fit_s": head.fit_s, "head_device": getattr(head, "device", None), "test_accuracy_full_panel": float((Pfull.argmax(1) == y[ti]).mean()),
        "panel_alias": alias, "families": {p["name"]: p["family"] for p in bank["panels"]},
        "deleted": {p["name"]: p["deleted"] for p in bank["panels"]},
        "excluded_labels": d["excluded"], "n_cells": d["n_cells"], "n_patients": int(len(pkeys)),
        "label_counts": {d["target_types"][c]: int(n) for c, n in enumerate(np.bincount(y, minlength=17)) if n},
        "strata_reports": reports, "wall_clock_s": round(time.time() - t0, 1), "host": os.uname().nodename}
    return arrays, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--head", required=True, choices=["xgb", "mlp", "logistic"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N panels, output tagged _debug")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    t0 = time.time()
    reg = std.registry()[args.cohort]
    if not std.is_ready(args.cohort):
        sys.exit(f"{args.cohort}: no READY file")
    out = Path(args.outdir) if args.outdir else std.OUT / args.cohort
    bank_path = std.OUT / args.cohort / "panel_bank.json"
    if not bank_path.exists():
        sys.exit(f"{bank_path} missing: run make_banks.py and commit the bank first")
    bank = json.loads(bank_path.read_text())
    d = std.load_std(args.cohort, reg["transform"], cohort_key=reg.get("cohort_key"),
                     sample_as_patient=reg.get("sample_as_patient", False), synonyms=reg.get("synonyms"))
    onto = Ontology.from_repo(d["target_types"])
    arrays, meta = compute(d, bank, reg, args.cohort, args.head, args.seed, onto, args.limit)
    tag = f"within_{args.head}_seed{args.seed}{'_debug' if args.limit else ''}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{tag}.npz", **arrays)
    meta["wall_clock_s"] = round(time.time() - t0, 1)
    std.write_json(out / f"{tag}_meta.json", meta)
    print(f"[{args.cohort} {args.head} s{args.seed}] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
