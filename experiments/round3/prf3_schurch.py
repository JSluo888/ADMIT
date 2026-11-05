#!/usr/bin/env python3
"""prf3.py for Schurch: same config, bank, split and head as run_cohort3.py (split seed 0, XGBoost); scoring as prf3.

  python experiments/round3/prf3_schurch.py [--limit 2]
Output: runs/prf/schurch_xgb_seed0.json
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
from common import REPO, write_json  # noqa: E402
from cohort import load_cohort_table, load_config, make_bank, make_splits  # noqa: E402
from run_cohort import fit_head  # noqa: E402
from engine3 import maps_for, run_arms3  # noqa: E402
from prf3 import OUT, MIN_TYPE, admit_calls, scores  # noqa: E402
from admit.ontology import NODE17  # noqa: E402
from admit.panel_matched import Ontology  # noqa: E402

CONFIG = REPO / "configs" / "schurch.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config(CONFIG)
    d = load_cohort_table(cfg)
    tt = list(NODE17)
    onto = Ontology.from_repo(tt)
    idx17 = {t: i for i, t in enumerate(tt)}
    y = np.array([idx17[n] for n in d["label_name"]], dtype=np.int64)
    x, patient, markers = d["x"], np.asarray(d["patient"]).astype(str), d["markers"]
    bank = make_bank(markers, Path(cfg["panel_bank"]))
    uniq = []
    for p in bank["panels"]:
        if all(tuple(q["markers"]) != tuple(p["markers"]) for q in uniq):
            uniq.append(p)
    if args.limit:
        uniq = uniq[: args.limit]
    sp = [s for s in make_splits(patient, cfg["splits"]) if s["seed"] == 0][0]
    rng = sp["rng"]
    tr = np.concatenate([rng.choice(np.flatnonzero(patient == g), min(cfg.get("train_cap", 2000),
                                                                       int((patient == g).sum())), replace=False)
                         for g in sp["train"]])
    ci = np.flatnonzero(np.isin(patient, sp["cal"]))
    ti = np.flatnonzero(np.isin(patient, sp["test"]))
    mu, sd = x[tr].mean(0), x[tr].std(0)
    sd[sd == 0] = 1.0
    head, present, _ = fit_head("xgb", (x[tr] - mu) / sd, y[tr], sp["seed"])

    def p17(Z):
        P = np.zeros((len(Z), len(tt)))
        P[:, present] = head.predict_proba(Z)
        return P / P.sum(1, keepdims=True)

    Zc, Zt = (x[ci] - mu) / sd, (x[ti] - mu) / sd
    yc, yt, ytr = y[ci], y[ti], y[tr]
    Pc_full = p17(Zc)
    out = []
    for pi, p in enumerate(uniq):
        miss = [j for j, m in enumerate(markers) if m not in set(p["markers"])]
        zc, zt = Zc.copy(), Zt.copy()
        zc[:, miss] = 0.0
        zt[:, miss] = 0.0
        Pc, Pt = p17(zc), p17(zt)
        gmap, S, _ = maps_for(onto, p["markers"], ytr, len(yc), [0.10])[0.10]
        sets, _, _ = run_arms3(onto, p["markers"], Pc, yc, Pt, Pc_full, 0.10, ["admit_m1"], gmap, S)
        adm = onto.admissible(p["markers"])
        forced = Pt.argmax(1)
        admit = admit_calls(sets["admit_m1"][0], onto)
        ident = onto.projection_map(adm)[yt] >= 0          # reference cell has an admissible ancestor under the panel
        out.append({"panel": p["name"], "family": p["family"], "n_test": int(len(yt)),
                    "forced_unsupported": float((~adm[forced]).mean()), "identifiable": float(ident.mean()),
                    "forced": scores(forced, yt, onto), "admit": scores(admit, yt, onto),
                    "forced_ident": scores(forced[ident], yt[ident], onto) if ident.any() else None,
                    "admit_ident": scores(admit[ident], yt[ident], onto) if ident.any() else None})
        print(f"[schurch s0] panel {pi + 1}/{len(uniq)} {time.time() - t0:.0f}s", flush=True)
    res = {"cohort": "schurch", "seed": 0, "head": "xgb", "alpha": 0.10, "min_type_cells": MIN_TYPE,
           "n_test_patients": int(len(sp["test"])), "panels": out, "wall_clock_s": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"schurch_xgb_seed0{'_debug' if args.limit else ''}"
    write_json(OUT / f"{tag}.json", res)
    print(f"[schurch] done {res['wall_clock_s']}s", flush=True)


if __name__ == "__main__":
    main()
