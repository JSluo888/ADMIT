#!/usr/bin/env python3
"""Task 5b: paired marker deletion on Schurch CRC CODEX with PATIENT splits.

35 patients; the three pilot splits (seeds 20260928-30): 15 train / 12 calibration / 8 test
patients. Heads (logistic, pilot spec; XGBoost) are trained on training patients only (2,000
cells per patient, pilot rng), on 17-node labels harmonized with the recovered ProteoRanker map.
Missing markers = training mean in log1p space = 0 after the train-only StandardScaler (the
same encoding is applied to calibration and test cells). Every arm of arms.ARMS runs on every
panel of schurch_panel_bank.json, plus two equal-patient conformal-risk-control arms (pilot):
lac_crc_filt (flat LAC + filter) and patch_pm_crc (projected strata, per-stratum CRC).
Reported per split: pooled-cell metrics with patient-cluster bootstrap CIs, and equal-patient
metrics with patient-bootstrap CIs.

  python experiments/run_schurch.py --head logistic
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, write_json  # noqa: E402
from arms import ARMS, run_arms  # noqa: E402
from schurch_common import PILOT_SPLIT_SEEDS, load_schurch  # noqa: E402
from admit.panel_matched import (Ontology, SourcePATCH, conformal_quantile, crc_threshold,  # noqa: E402
                                       equal_patient, headline, patch_pm_crc, set_metrics)

ALPHA, N_MIN, QUANT, TRAIN_CAP, B = 0.10, 10, "order_stat", 2000, 2000
EXTRA_ARMS = ["patch_pm_crc", "lac_crc_filt"]


def fit_head(kind, X, y, seed):
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
                              tree_method="hist", n_jobs=8, random_state=seed)
        m.fit(X, yy, verbose=False)
    return m, present, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="logistic", choices=["logistic", "xgb"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    d = load_schurch()
    bank = json.loads((RUNS / "schurch" / "schurch_panel_bank.json").read_text())
    from admit.ontology import NODE17
    tt = list(NODE17)          # the 17-node column order shared by every experiment
    onto = Ontology.from_repo(tt)
    par = onto.ancestor_lists()
    idx17 = {t: i for i, t in enumerate(tt)}
    y = np.array([idx17[n] for n in d["label_name"]], dtype=np.int64)
    x, patient, markers = d["x"], d["patient"], d["markers"]
    pid = np.unique(patient)
    res = {"head": args.head, "alpha": ALPHA, "n_min_old_patch": N_MIN, "quantile": QUANT, "train_cap": TRAIN_CAP,
           "bootstrap": B, "n_cells": int(len(y)), "n_patients": int(len(pid)),
           "excluded_labels": d["excluded"], "label_mapping": d["mapping"],
           "label_counts_17": {tt[c]: int(n) for c, n in enumerate(np.bincount(y, minlength=len(tt))) if n},
           "arms": ARMS + EXTRA_ARMS, "splits": []}
    panels = bank["panels"][: args.limit] if args.limit else bank["panels"]
    for seed in PILOT_SPLIT_SEEDS:
        ts = time.time()
        rng = np.random.default_rng(seed)
        ids = rng.permutation(pid)
        part = dict(train=ids[:15], cal=ids[15:27], test=ids[27:])
        tr = np.concatenate([rng.choice(np.flatnonzero(patient == g), min(TRAIN_CAP, int((patient == g).sum())),
                                        replace=False) for g in part["train"]])
        ci = np.flatnonzero(np.isin(patient, part["cal"]))
        ti = np.flatnonzero(np.isin(patient, part["test"]))
        mu, sd = x[tr].mean(0), x[tr].std(0)
        sd[sd == 0] = 1.0
        Z = lambda i: (x[i] - mu) / sd  # noqa: E731
        head, present, note = fit_head(args.head, Z(tr), y[tr], seed)

        def p17(Zm):
            P = np.zeros((len(Zm), len(tt)))
            P[:, present] = head.predict_proba(Zm)
            return P / P.sum(1, keepdims=True)

        Zc, Zt = Z(ci), Z(ti)
        yc, yt, gc, gt = y[ci], y[ti], patient[ci], patient[ti]
        Pc_full = p17(Zc)
        src = SourcePATCH(onto, ALPHA, N_MIN, QUANT).fit(onto.aggregate(Pc_full), yc)
        split = {"seed": seed, "train_patients": part["train"].tolist(), "cal_patients": part["cal"].tolist(),
                 "test_patients": part["test"].tolist(), "n_train": int(len(tr)), "n_cal": int(len(ci)),
                 "n_test": int(len(ti)), "head_note": note, "present_classes": [tt[c] for c in present],
                 "point_accuracy_full": float((Pc_full.argmax(1) == yc).mean()), "panels": {}}
        cache = {}
        for p in panels:
            key = tuple(p["markers"])
            if key in cache:
                split["panels"][p["name"]] = {"alias_of": cache[key], "family": p["family"]}
                continue
            miss = [j for j, m in enumerate(markers) if m not in set(p["markers"])]
            zc, zt = Zc.copy(), Zt.copy()
            zc[:, miss] = 0.0
            zt[:, miss] = 0.0
            Pc, Pt = p17(zc), p17(zt)
            P = p["markers"]
            sets, info = run_arms(onto, par, P, Pc, yc, Pt, Pc_full, src, ALPHA, QUANT, N_MIN)
            Nc, Nt = onto.aggregate(Pc), onto.aggregate(Pt)
            sets["patch_pm_crc"], lam_crc = patch_pm_crc(Nc, yc, gc, Nt, onto, P, ALPHA)
            t_crc = crc_threshold(1.0 - Pc[np.arange(len(yc)), yc], ALPHA, gc)
            sets["lac_crc_filt"] = ((1.0 - Pt) <= t_crc) & onto.admissible(P)[None, :]
            entry = {"family": p["family"], "deleted": p["deleted"], "pm_strata": info["pm_strata"],
                     "pm_infinite_strata": info["pm_infinite_strata"],
                     "crc_lambda": {onto.names[c]: (float(v) if np.isfinite(v) else "inf")
                                    for c, v in enumerate(lam_crc) if v > -np.inf},
                     "lac_crc_threshold": float(t_crc) if np.isfinite(t_crc) else "inf",
                     "point_accuracy": float((Pt.argmax(1) == yt).mean()), "arms": {}}
            for arm, S in sets.items():
                m = set_metrics(S, yt, onto, P, groups=gt, bootstrap=B, supported=info["pm_supported"],
                                per_stratum=arm in ("patch_pm", "patch_src_backstop", "patch_src", "lac_filt",
                                                    "patch_pm_crc"))
                ep = equal_patient(S, yt, onto, P, gt, B)
                entry["arms"][arm] = {"headline": headline(m), "equal_patient": ep,
                                      **({"per_stratum": m["per_stratum"]} if "per_stratum" in m else {})}
            split["panels"][p["name"]] = entry
            cache[key] = p["name"]
            a = entry["arms"]
            fv = lambda v: "  nan" if v is None else f"{v:.3f}"  # noqa: E731
            ep = lambda arm: fv(a[arm]["equal_patient"]["exact_cov"]["value"])  # noqa: E731
            print(f"[s{seed} {p['name']:26s}] EP exact: pm {ep('patch_pm')} src+bs {ep('patch_src_backstop')} "
                  f"lac_filt {ep('lac_filt')} pm_crc {ep('patch_pm_crc')} | |C| pm {a['patch_pm']['headline']['size']:.2f}",
                  flush=True)
        split["wall_clock_s"] = round(time.time() - ts, 1)
        res["splits"].append(split)
    res["wall_clock_total_s"] = round(time.time() - t0, 1)
    write_json(RUNS / "schurch" / f"schurch_deletion_{args.head}{'_debug' if args.limit else ''}.json", res)
    print("done", res["wall_clock_total_s"])


if __name__ == "__main__":
    main()
