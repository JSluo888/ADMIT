#!/usr/bin/env python3
"""Within-cohort paired deletion on a standard-schema cohort: same loaders, splits, caps, heads and frozen panel
banks as cohorts/run_within.py, scored with engine3.

  python experiments/round3/run_within3.py --cohort cords2024_nsclc_imc --head xgb --seed 0
Output: runs/round3/cohorts/<cohort>/within_<head>_seed<k>.npz (+ _meta.json). The bank is read from
runs/cohorts/<cohort>/panel_bank.json (unchanged).

Requirement-map sensitivity (default off): ADMIT_REQMAP_JSON=<map.json>, a JSON file {node: [markers]}, replaces R(c)
of the named nodes and sends the output to runs/reqmap/<map stem>/cohorts/<cohort>/. --out sets another root.
configs/map_strong.json drops CD45 from the T-lineage, T-subtype, B-cell, myeloid and macrophage nodes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cohorts"))
sys.path.insert(0, str(HERE))
import std  # noqa: E402
from engine3 import (ALPHAS, GSTATS, STATS, TSTATS, all_arm_names, ep_from_block, evaluate_block3,  # noqa: E402
                     new_accumulators)
from admit.panel_matched import Ontology  # noqa: E402

OUT3 = std.OUT.parent / "round3" / "cohorts"


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


def compute3(d, bank, reg, cohort, head_kind, seed, onto, limit=0):
    """Pure computation, no file output. d: std.load_std() dict."""
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
    n_pat = len(test_codes)
    acc, acct, grp = new_accumulators(len(panels), len(arms), len(ALPHAS), n_pat, onto.K, dtype=np.float32)
    ep = np.full((len(panels), len(arms), len(ALPHAS), 3), np.nan)
    Pc_full = head.predict17(Zc)
    yc, yt, ytr = y[ci], y[ti], y[tr]
    gc = pkeys[pcode[ci]]
    reports, src_cache = {}, {}
    slots = np.arange(n_pat)
    for pi, p in enumerate(panels):
        keep = set(p["markers"])
        miss = [j for j, m in enumerate(d["markers"]) if m not in keep]
        if miss:
            zc, zt = Zc.copy(), Zt.copy()
            zc[:, miss] = 0.0
            zt[:, miss] = 0.0
        else:
            zc, zt = Zc, Zt
        Pc, Pt = head.predict17(zc), head.predict17(zt)
        ta, tt_, tg = new_accumulators(1, len(arms), len(ALPHAS), n_pat, onto.K)
        reports[p["name"]] = evaluate_block3(ta, tt_, tg, 0, onto, p["markers"], Pc, yc, Pt, yt, pslot, n_pat,
                                             Pc_full, ytr, True, gc=gc, arm_index=arm_index, src_cache=src_cache)
        acc[pi], acct[pi], grp[pi] = ta[0], tt_[0], tg[0]
        for k, tgt in enumerate(("own", "pi", "gp")):
            ep[pi, :, :, k] = ep_from_block(ta[0], slots, tgt)[0]
        if pi % 10 == 0:
            print(f"[{cohort} {head_kind} s{seed}] panel {pi + 1}/{len(panels)} {time.time() - t0:.0f}s", flush=True)
    Pfull = head.predict17(Zt)
    arrays = dict(acc=acc, acct=acct, grp=grp, ep_cov=ep[None], ep_targets=np.array(["own", "pi", "gp"]),
                  panels=np.array([p["name"] for p in panels]), patients=pkeys[test_codes], arms=np.array(arms),
                  alphas=np.array(ALPHAS), stats=np.array(STATS), tstats=np.array(TSTATS), gstats=np.array(GSTATS),
                  nodes=np.array(onto.names), seeds=np.array([seed]))
    meta = {
        "cohort": cohort, "protocol": "within", "head": head_kind, "seed": seed, "engine": "engine3",
        "transform": d["transform"], "transform_fit": tfit, "markers": d["markers"],
        "dropped_columns": d["dropped_columns"],
        "splits": [{"seed": seed, "train": pkeys[sp["train"]].tolist(), "cal": pkeys[sp["cal"]].tolist(),
                    "test": pkeys[sp["test"]].tolist(), "n_train": int(len(tr)), "n_cal": int(len(ci)),
                    "n_test": int(len(ti)), "eval_cap_per_patient": cap}],
        "head_fit_s": head.fit_s, "head_device": getattr(head, "device", None),
        "test_accuracy_full_panel": float((Pfull.argmax(1) == yt).mean()),
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
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="output root (default: round3/cohorts, or reqmap/<map>/cohorts)")
    args = ap.parse_args()
    t0 = time.time()
    map_path = os.environ.get("ADMIT_REQMAP_JSON")
    override = json.loads(Path(map_path).read_text()) if map_path else None
    root = Path(args.out) if args.out else (std.OUT.parent / "reqmap" / Path(map_path).stem / "cohorts"
                                            if map_path else OUT3)
    if override is not None and root.resolve() == OUT3.resolve():
        sys.exit("a requirement override must not write into round3/cohorts")
    reg = std.registry()[args.cohort]
    if not std.is_ready(args.cohort):
        sys.exit(f"{args.cohort}: no READY file")
    bank = json.loads((std.OUT / args.cohort / "panel_bank.json").read_text())
    d = std.load_std(args.cohort, reg["transform"], cohort_key=reg.get("cohort_key"),
                     sample_as_patient=reg.get("sample_as_patient", False), synonyms=reg.get("synonyms"))
    onto = Ontology.from_repo(d["target_types"], override)
    arrays, meta = compute3(d, bank, reg, args.cohort, args.head, args.seed, onto, args.limit)
    if override is not None:
        meta["requirement_override"] = {"path": map_path, "map": override}
    out = root / args.cohort
    out.mkdir(parents=True, exist_ok=True)
    tag = f"within_{args.head}_seed{args.seed}{'_debug' if args.limit else ''}"
    np.savez_compressed(out / f"{tag}.npz", **arrays)
    meta["wall_clock_s"] = round(time.time() - t0, 1)
    std.write_json(out / f"{tag}_meta.json", meta)
    print(f"[{args.cohort} {args.head} s{args.seed}] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
