#!/usr/bin/env python3
"""Paired deletion on a cohort config (Schurch): same splits, caps, heads and bank as round2/run_cohort.py, scored
with engine3 (frozen support map from the training cells, exact ADMIT-M1, pi_P and g_P targets).

  python experiments/round3/run_cohort3.py --config configs/schurch.json --head logistic --seeds 0-9
Writes runs/round3/cohort_<name>_<head>_s<seeds>.npz (+ _meta.json).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "round2"))
sys.path.insert(0, str(HERE))
from common import RUNS, write_json  # noqa: E402
from cohort import load_cohort_table, load_config, make_bank, make_splits  # noqa: E402
from run_cohort import fit_head, parse_seeds  # noqa: E402
from engine3 import (ALPHAS, GSTATS, STATS, TSTATS, all_arm_names, ep_from_block, evaluate_block3,  # noqa: E402
                     new_accumulators)
from admit.ontology import NODE17  # noqa: E402
from admit.panel_matched import Ontology  # noqa: E402

OUT = RUNS / "round3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--head", required=True, choices=["logistic", "xgb"])
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config(args.config)
    d = load_cohort_table(cfg)
    tt = list(NODE17)
    onto = Ontology.from_repo(tt)
    idx17 = {t: i for i, t in enumerate(tt)}
    y = np.array([idx17[n] for n in d["label_name"]], dtype=np.int64)
    x, patient, markers = d["x"], np.asarray(d["patient"]).astype(str), d["markers"]
    bank = make_bank(markers, Path(cfg["panel_bank"]))          # existing frozen bank, never rewritten
    uniq, alias = [], {}
    for p in bank["panels"]:
        key = tuple(p["markers"])
        for q in uniq:
            if tuple(q["markers"]) == key:
                alias[p["name"]] = q["name"]
                break
        else:
            uniq.append(p)
    if args.limit:
        uniq = uniq[: args.limit]
    patients = sorted(np.unique(patient).tolist())
    pidx = {p: i for i, p in enumerate(patients)}
    pslot = np.array([pidx[p] for p in patient])
    arms = all_arm_names(True)
    arm_index = {a: i for i, a in enumerate(arms)}
    want = set(parse_seeds(args.seeds))
    splits = [s for s in make_splits(patient, cfg["splits"]) if s["seed"] in want]
    acc, acct, grp = new_accumulators(len(uniq), len(arms), len(ALPHAS), len(patients), onto.K)
    nS = len(splits)
    ep = np.full((nS, len(uniq), len(arms), len(ALPHAS), 3), np.nan)      # last axis: own, pi, gp
    pooled = np.full_like(ep, np.nan)
    split_meta, reports = [], {}
    for si, sp in enumerate(splits):
        ts = time.time()
        rng = sp["rng"]
        tr = np.concatenate([rng.choice(np.flatnonzero(patient == g), min(cfg.get("train_cap", 2000),
                                                                           int((patient == g).sum())), replace=False)
                             for g in sp["train"]])
        ci = np.flatnonzero(np.isin(patient, sp["cal"]))
        ti = np.flatnonzero(np.isin(patient, sp["test"]))
        mu, sd = x[tr].mean(0), x[tr].std(0)
        sd[sd == 0] = 1.0
        head, present, note = fit_head(args.head, (x[tr] - mu) / sd, y[tr], sp["seed"])

        def p17(Z):
            P = np.zeros((len(Z), len(tt)))
            P[:, present] = head.predict_proba(Z)
            return P / P.sum(1, keepdims=True)

        Zc, Zt = (x[ci] - mu) / sd, (x[ti] - mu) / sd
        Pc_full = p17(Zc)
        split_meta.append({"seed": sp["seed"], "train": sp["train"].tolist(), "cal": sp["cal"].tolist(),
                           "test": sp["test"].tolist(), "n_train": int(len(tr)), "n_cal": int(len(ci)),
                           "n_test": int(len(ti)), "head_note": note,
                           "cal_accuracy_full": float((Pc_full.argmax(1) == y[ci]).mean())})
        tp = [pidx[g] for g in sp["test"]]
        reports[int(sp["seed"])] = {}
        for pi, p in enumerate(uniq):
            miss = [j for j, m in enumerate(markers) if m not in set(p["markers"])]
            zc, zt = Zc.copy(), Zt.copy()
            zc[:, miss] = 0.0
            zt[:, miss] = 0.0
            Pc, Pt = p17(zc), p17(zt)
            ta, tt_, tg = new_accumulators(1, len(arms), len(ALPHAS), len(patients), onto.K)
            rep = evaluate_block3(ta, tt_, tg, 0, onto, p["markers"], Pc, y[ci], Pt, y[ti], pslot[ti],
                                  len(patients), Pc_full, y[tr], True, gc=patient[ci], arm_index=arm_index,
                                  src_cache={})
            acc[pi] += ta[0]
            acct[pi] += tt_[0]
            grp[pi] += tg[0]
            reports[int(sp["seed"])][p["name"]] = rep
            for k, tgt in enumerate(("own", "pi", "gp")):
                ep[si, pi, :, :, k], pooled[si, pi, :, :, k] = ep_from_block(ta[0], tp, tgt)
        print(f"[{cfg['name']} {args.head} seed {sp['seed']}] {time.time()-ts:.1f}s  EP full admit_m1 gp "
              f"{ep[si, 0, arm_index['admit_m1'], 1, 2]:.3f} lac_node_gp {ep[si, 0, arm_index['lac_node_gp'], 1, 2]:.3f}",
              flush=True)
    tag = f"cohort_{cfg['name']}_{args.head}_s{args.seeds}{'_debug' if args.limit else ''}"
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{tag}.npz", acc=acc.astype(np.float32), acct=acct.astype(np.float32),
                        grp=grp.astype(np.float32), ep_cov=ep, pooled_cov=pooled, ep_targets=np.array(["own", "pi", "gp"]),
                        panels=np.array([p["name"] for p in uniq]), patients=np.array(patients), arms=np.array(arms),
                        alphas=np.array(ALPHAS), stats=np.array(STATS), tstats=np.array(TSTATS),
                        gstats=np.array(GSTATS), nodes=np.array(onto.names),
                        seeds=np.array([s["seed"] for s in splits]))
    write_json(OUT / f"{tag}_meta.json",
               {"cohort": cfg["name"], "head": args.head, "config": args.config, "engine": "engine3",
                "splits": split_meta, "panel_alias": alias,
                "families": {p["name"]: p["family"] for p in bank["panels"]},
                "deleted": {p["name"]: p["deleted"] for p in bank["panels"]},
                "excluded_labels": d["excluded"], "n_cells": int(len(y)), "n_patients": len(patients),
                "label_counts": {tt[c]: int(n) for c, n in enumerate(np.bincount(y, minlength=len(tt))) if n},
                "strata_reports": reports, "wall_clock_s": round(time.time() - t0, 1)})
    print("done", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
