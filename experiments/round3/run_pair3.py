#!/usr/bin/env python3
"""Real panel change / new cohort / site shift between two standard-schema cohorts: same roles, caps, heads,
z-scoring and blocks as cohorts/run_pair.py, scored with engine3.

  python experiments/round3/run_pair3.py --pair piyadasa_immune_vs_tumor --direction ab --head xgb --seed 0

Frozen support map: labels of the head's training cells (source A, train role, after the 2,000-per-patient cap),
restricted to the target cohort's annotation vocabulary V_B = the 17-node labels occurring in cohort B (each source
label -> finest ancestor-or-self in V_B), projected with pi_P (P = shared panel). Expected counts use the block's
calibration cell count: transfer / within_source_masked = A cal-role cells; target_full = B cal-role cells; the
target-calibration curve = the n drawn B cells.
Output: runs/round3/cohorts/<pair>/pair_<dir>_<head>_seed<k>.npz (+ _meta.json).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cohorts"))
sys.path.insert(0, str(HERE))
import std  # noqa: E402
from engine3 import (ALPHAS, GSTATS, NGSTAT, NSTAT, NTSTAT, STATS, TSTATS, all_arm_names,  # noqa: E402
                     evaluate_block3, new_accumulators)
from admit.panel_matched import Ontology  # noqa: E402

NS = [50, 100, 200, 500, 1000, 2000]
NDRAW = 10
TCAL_ARMS = ["admit_m1", "admit", "admit_m1cal", "lac_node_proj", "lac_node_gp", "lac_filt", "aps_filt"]
OUT3 = std.OUT.parent / "round3" / "cohorts"


def load_side(name, key, loaders=None):
    reg = std.registry()
    if loaders and name in loaders:
        return loaders[name](key)
    r = reg[name]
    return std.load_std(name, r["transform"], cohort_key=key, synonyms=r.get("synonyms"),
                        name_map=std.registry_name_map(name), sample_as_patient=r.get("sample_as_patient", False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--direction", required=True, choices=["ab", "ba"])
    ap.add_argument("--head", required=True, choices=["xgb", "mlp", "logistic"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--no-tcal", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    reg = std.registry()
    pr = reg["pairs"][args.pair]
    names = (pr["a"], pr["b"]) if args.direction == "ab" else (pr["b"], pr["a"])
    for n in names:
        if n != "orion" and not std.is_ready(n) and not pr.get("allow_not_ready"):
            sys.exit(f"{n}: no READY file")
    key = args.pair if pr.get("same_patients") else None
    loaders = {}
    if "orion" in names or "lin2023_crc_cycif" in names:
        import lin_orion
        loaders = lin_orion.LOADERS
    SA, SB = (load_side(n, key, loaders) for n in names)
    onto = Ontology.from_repo(SA["target_types"])
    if pr.get("same_patients"):
        sps = [std.patient_splits(np.unique(np.concatenate([SA["pkeys"], SB["pkeys"]])), pr["splits"]["fractions"],
                                  [args.seed])[0]]
    else:
        sps = [std.patient_splits(S["pkeys"], pr["splits"]["fractions"], [args.seed])[0] for S in (SA, SB)]
    role = {k: r for sp_ in sps for r in ("train", "cal", "test") for k in sp_[r]}
    sp = {r: np.concatenate([sp_[r] for sp_ in sps]) for r in ("train", "cal", "test")}

    def side_rows(S, tag):
        groups = std.patient_groups(S["pcode"], len(S["pkeys"]))
        codes = {r: [i for i, k in enumerate(S["pkeys"]) if role[k] == r] for r in ("train", "cal", "test")}
        rr = np.random.default_rng([args.seed, tag])
        cap = 3000 if S["n_cells"] > 1_000_000 else None
        tr = std.train_index(groups, codes["train"], 2000, rr) if codes["train"] else np.zeros(0, np.int64)
        ci = std.capped_index(groups, codes["cal"], cap, rr)
        ti = std.capped_index(groups, codes["test"], cap, rr)
        return tr, ci, ti, cap

    trA, ciA, tiA, capA = side_rows(SA, 1)
    trB, ciB, tiB, capB = side_rows(SB, 2)
    if min(len(trA), len(ciA), len(tiB)) == 0:
        sys.exit("empty role (too few patients)")
    tfA = std.fit_transform(SA["x"], reg.get(names[0], {}).get("transform", "none"), trA)
    tfB = std.fit_transform(SB["x"], reg.get(names[1], {}).get("transform", "none"), trB if len(trB) else tiB)
    fitB = trB if len(trB) else tiB
    muA, sdA = SA["x"][trA].mean(0), SA["x"][trA].std(0)
    muB, sdB = SB["x"][fitB].mean(0), SB["x"][fitB].std(0)
    sdA[sdA == 0] = 1.0
    sdB[sdB == 0] = 1.0
    mA, mB = SA["markers"], SB["markers"]
    P = [m for m in mA if m in set(mB)]
    jB = {m: j for j, m in enumerate(mB)}
    inP = np.array([m in set(P) for m in mA])

    def zA(rows, mask=False):
        z = ((SA["x"][rows] - muA) / sdA).astype(np.float32)
        if mask:
            z[:, ~inP] = 0.0
        return z

    def zB(rows):
        zb = ((SB["x"][rows] - muB) / sdB).astype(np.float32)
        out = np.zeros((len(rows), len(mA)), dtype=np.float32)
        for j, m in enumerate(mA):
            if m in jB:
                out[:, j] = zb[:, jB[m]]
        return out

    head = std.Head(args.head).fit(zA(trA), SA["y"][trA], args.seed)
    Pc_full = head.predict17(zA(ciA))
    Pc = head.predict17(zA(ciA, mask=True))
    PtB = head.predict17(zB(tiB))
    PtA = head.predict17(zA(tiA, mask=True))
    PcB = head.predict17(zB(ciB)) if len(ciB) else None
    ycA, ytA, ycB, ytB = SA["y"][ciA], SA["y"][tiA], SB["y"][ciB], SB["y"][tiB]
    gcA, gcB = SA["pkeys"][SA["pcode"][ciA]], SB["pkeys"][SB["pcode"][ciB]]
    patsB = np.unique(SB["pcode"][tiB])
    slotB = np.searchsorted(patsB, SB["pcode"][tiB])
    patsA = np.unique(SA["pcode"][tiA])
    slotA = np.searchsorted(patsA, SA["pcode"][tiA])
    ytrain = SA["y"][trA]
    vocabB = np.unique(SB["y"]).tolist()
    arms = all_arm_names(True)
    ai = {a: i for i, a in enumerate(arms)}
    blocks, reps = {}, {}

    def block(name, Pc_, yc_, Pt_, yt_, slot, npat, Pfull, gc):
        acc, acct, grp = new_accumulators(1, len(arms), len(ALPHAS), npat, onto.K)
        reps[name] = evaluate_block3(acc, acct, grp, 0, onto, P, Pc_, yc_, Pt_, yt_, slot, npat, Pfull, ytrain, True,
                                     gc=gc, arm_index=ai, src_cache={}, vocab=vocabB)
        blocks[name] = (acc.astype(np.float32), acct.astype(np.float32), grp.astype(np.float32))

    block("transfer", Pc, ycA, PtB, ytB, slotB, len(patsB), Pc_full, gcA)
    if len(tiA):
        block("within_source_masked", Pc, ycA, PtA, ytA, slotA, len(patsA), Pc_full, gcA)
    if PcB is not None and len(ciB):
        block("target_full", PcB, ycB, PtB, ytB, slotB, len(patsB), PcB, gcB)
    tcal = None
    if not args.no_tcal and PcB is not None and len(ciB):
        ti_ = {a: i for i, a in enumerate(TCAL_ARMS)}
        tcal = np.zeros((len(NS), NDRAW, len(TCAL_ARMS), len(patsB), onto.K, NSTAT), dtype=np.float32)
        tcal_t = np.zeros((len(NS), NDRAW, len(patsB), onto.K, NTSTAT), dtype=np.float32)
        tcal_g = np.zeros((len(NS), NDRAW, len(TCAL_ARMS), onto.K, NGSTAT), dtype=np.float32)
        reps["tcal_first_draw"] = {}
        for ni, n in enumerate(NS):
            for r in range(NDRAW):
                rr = np.random.default_rng([args.seed, n, r])
                sel = rr.choice(len(ciB), min(n, len(ciB)), replace=False)
                a_, t_, g_ = new_accumulators(1, len(TCAL_ARMS), 1, len(patsB), onto.K)
                rep = evaluate_block3(a_, t_, g_, 0, onto, P, PcB[sel], ycB[sel], PtB, ytB, slotB, len(patsB),
                                      PcB[sel], ytrain, False, arm_index=ti_, alphas=[(0, 0.10)], vocab=vocabB,
                                      arm_list=TCAL_ARMS, n_cal=len(sel))
                tcal[ni, r], tcal_t[ni, r], tcal_g[ni, r] = a_[0, :, 0], t_[0, 0], g_[0, :, 0]
                if r == 0:
                    reps["tcal_first_draw"][n] = rep
    tag = f"pair_{args.direction}_{args.head}_seed{args.seed}"
    out = OUT3 / args.pair
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for k, (a_, t_, g_) in blocks.items():
        arrays[f"acc_{k}"], arrays[f"acct_{k}"], arrays[f"grp_{k}"] = a_, t_, g_
    if tcal is not None:
        arrays.update(tcal=tcal, tcal_t=tcal_t, tcal_g=tcal_g)
    np.savez_compressed(out / f"{tag}.npz", **arrays, patients_B=SB["pkeys"][patsB], patients_A=SA["pkeys"][patsA],
                        arms=np.array(arms), alphas=np.array(ALPHAS), stats=np.array(STATS), tstats=np.array(TSTATS),
                        gstats=np.array(GSTATS), nodes=np.array(onto.names), tcal_ns=np.array(NS),
                        tcal_arms=np.array(TCAL_ARMS), seeds=np.array([args.seed]))
    acc_full = lambda P_, y_: float((P_.argmax(1) == y_).mean()) if len(y_) else None  # noqa: E731
    std.write_json(out / f"{tag}_meta.json", {
        "pair": args.pair, "protocol": pr.get("protocol"), "status": pr.get("status", "final"),
        "engine": "engine3", "direction": args.direction, "source": names[0], "target": names[1],
        "head": args.head, "seed": args.seed, "shared_panel": P,
        "source_only_markers": [m for m in mA if m not in set(mB)],
        "target_only_markers": [m for m in mB if m not in set(mA)],
        "target_vocabulary_V_B": [onto.names[c] for c in vocabB],
        "admissible_under_P": [onto.names[c] for c in np.where(onto.admissible(P))[0]],
        "admissible_under_P_round2_map": [onto.names[c] for c in np.where(onto.admissible(P, legacy=True))[0]],
        "roles": {"train": sp["train"].tolist(), "cal": sp["cal"].tolist(), "test": sp["test"].tolist()},
        "n_cells": {"src_train": int(len(trA)), "src_cal": int(len(ciA)), "src_test": int(len(tiA)),
                    "tgt_train_for_scaling": int(len(trB)), "tgt_cal": int(len(ciB)), "tgt_test": int(len(tiB))},
        "eval_cap": {"source": capA, "target": capB}, "transform_fit": {"source": tfA, "target": tfB},
        "accuracy": {"src_test_full_panel": acc_full(head.predict17(zA(tiA)), ytA),
                     "src_test_masked_P": acc_full(PtA, ytA), "tgt_test_under_P": acc_full(PtB, ytB)},
        "label_counts": {"source": {onto.names[c]: int(n) for c, n in enumerate(np.bincount(SA["y"], minlength=17)) if n},
                         "target": {onto.names[c]: int(n) for c, n in enumerate(np.bincount(SB["y"], minlength=17)) if n}},
        "excluded_labels": {"source": SA["excluded"], "target": SB["excluded"]},
        "strata_reports": reps, "head_fit_s": head.fit_s, "wall_clock_s": round(time.time() - t0, 1),
        "host": os.uname().nodename})
    print(f"[{args.pair} {args.direction} {args.head} s{args.seed}] done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
