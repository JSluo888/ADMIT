#!/usr/bin/env python3
"""Round 2 paired deletion with patient splits on a generic cohort config (Schurch E4; tier1 cohorts later).

  python experiments/round2/run_cohort.py --config configs/round2/schurch.json --head logistic --seeds 0-4
Writes round2/cohort_<name>_<head>_s<a>-<b>.npz (additive accumulator over its splits + per-split
equal-patient / pooled coverage) and a JSON of strata reports. Partial files are summed by summarize_round2.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, load_split, write_json  # noqa: E402
from cohort import load_cohort_table, load_config, make_bank, make_splits  # noqa: E402
from engine import ALPHAS, NSTAT, STATS, all_arm_names, evaluate_block  # noqa: E402
from admit.ontology import NODE17  # noqa: E402
from admit.panel_matched import Ontology  # noqa: E402


def fit_head(kind, X, y, seed, n_jobs=4):
    present = np.unique(y)
    remap = {c: i for i, c in enumerate(present)}
    yy = np.array([remap[c] for c in y])
    note = None
    if kind == "logistic":
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=1.0, max_iter=1500, solver="lbfgs", random_state=seed)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            m.fit(X, yy)
            if any(issubclass(x.category, ConvergenceWarning) for x in w):
                note = "ConvergenceWarning"
    else:
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, objective="multi:softprob",
                              tree_method="hist", n_jobs=n_jobs, random_state=seed)
        m.fit(X, yy, verbose=False)
    return m, present, note


def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--head", required=True, choices=["logistic", "xgb"])
    ap.add_argument("--seeds", required=True, help="subset of the config seeds, e.g. 0-4")
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
    bank = make_bank(markers, Path(cfg["panel_bank"]))
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
    acc = np.zeros((len(uniq), len(arms), len(ALPHAS), len(patients), onto.K, NSTAT))
    nS = len(splits)
    ep_cov = np.full((nS, len(uniq), len(arms), len(ALPHAS)), np.nan)
    pooled_cov = np.full_like(ep_cov, np.nan)
    ep_size = np.full_like(ep_cov, np.nan)
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
        for pi, p in enumerate(uniq):
            miss = [j for j, m in enumerate(markers) if m not in set(p["markers"])]
            zc, zt = Zc.copy(), Zt.copy()
            zc[:, miss] = 0.0
            zt[:, miss] = 0.0
            Pc, Pt = p17(zc), p17(zt)
            tmp = np.zeros((1,) + acc.shape[1:])
            rep = evaluate_block(tmp, 0, onto, p["markers"], Pc, y[ci], Pt, y[ti], pslot[ti], len(patients),
                                 Pc_full, True, gc=patient[ci], arm_index=arm_index, src_cache={})
            acc[pi] += tmp[0]
            if si == 0:
                reports[p["name"]] = rep
            hits = tmp[0, :, :, :, :, 0].sum(axis=3)       # arm, alpha, patient
            ns = tmp[0, :, :, :, :, 1].sum(axis=3)
            lsize = tmp[0, :, :, :, :, 5].sum(axis=3)
            lall = tmp[0, :, :, :, :, 4].sum(axis=3)
            tp = [pidx[g] for g in sp["test"]]
            with np.errstate(invalid="ignore", divide="ignore"):
                ep_cov[si, pi] = np.nanmean(np.where(ns[:, :, tp] > 0, hits[:, :, tp] / ns[:, :, tp], np.nan), axis=2)
                pooled_cov[si, pi] = hits.sum(axis=2) / ns.sum(axis=2)
                ep_size[si, pi] = np.nanmean(np.where(lall[:, :, tp] > 0, lsize[:, :, tp] / lall[:, :, tp], np.nan),
                                             axis=2)
        print(f"[{cfg['name']} {args.head} seed {sp['seed']}] {time.time()-ts:.1f}s  "
              f"EP admit/full {ep_cov[si, 0, arm_index['admit'], 1]:.3f}", flush=True)
    tag = f"cohort_{cfg['name']}_{args.head}_s{args.seeds}{'_debug' if args.limit else ''}"
    np.savez_compressed(RUNS / "round2" / f"{tag}.npz", acc=acc.astype(np.float32), ep_cov=ep_cov,
                        pooled_cov=pooled_cov, ep_size=ep_size, panels=np.array([p["name"] for p in uniq]),
                        patients=np.array(patients), arms=np.array(arms), alphas=np.array(ALPHAS),
                        stats=np.array(STATS), nodes=np.array(onto.names),
                        seeds=np.array([s["seed"] for s in splits]))
    write_json(RUNS / "round2" / f"{tag}_meta.json",
               {"cohort": cfg["name"], "head": args.head, "config": args.config, "splits": split_meta,
                "panel_alias": alias, "families": {p["name"]: p["family"] for p in bank["panels"]},
                "deleted": {p["name"]: p["deleted"] for p in bank["panels"]},
                "excluded_labels": d["excluded"], "n_cells": int(len(y)), "n_patients": len(patients),
                "label_counts": {tt[c]: int(n) for c, n in enumerate(np.bincount(y, minlength=len(tt))) if n},
                "strata_reports_first_split": reports, "wall_clock_s": round(time.time() - t0, 1)})
    print("done", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
