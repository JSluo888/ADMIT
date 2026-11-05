#!/usr/bin/env python3
"""Task 5: paired marker deletion on Orion crc_test (same cells, same labels, panel varies).

  python experiments/run_deletion.py --backbone round0 --protocol primary
  protocol primary      : calibrate crc_cal masked to P, evaluate crc_test masked to P
  protocol exchangeable : random 50/50 split of crc_test (seed 0) -> cal half / test half

Every arm in ``arms.ARMS`` runs on every panel of $ADMIT_RUNS/panel_bank.json.
Output: $ADMIT_RUNS/deletion/deletion_<backbone>_<protocol>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, get_backbone, load_identity, load_split, write_json  # noqa: E402
from arms import ARMS, run_arms  # noqa: E402
from admit.panel_matched import Ontology, SourcePATCH, headline, set_metrics  # noqa: E402

ALPHA, N_MIN, QUANT = 0.10, 10, "order_stat"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="round0", choices=["round0", "round0rt", "v6crc"])
    ap.add_argument("--protocol", default="primary", choices=["primary", "exchangeable", "patient_loso", "patient_loso13"])
    ap.add_argument("--limit", type=int, default=0, help="debug: first N panels only")
    ap.add_argument("--bootstrap", type=int, default=2000, help="cluster bootstrap reps when sample ids exist")
    ap.add_argument("--fake-groups", action="store_true", help="debug only: random sample ids (never for results)")
    args = ap.parse_args()
    if args.protocol in ("patient_loso", "patient_loso13"):
        return main_loso(args)
    t0 = time.time()
    bank = json.loads((RUNS / "panel_bank.json").read_text())
    cal, test = load_split("crc_cal"), load_split("crc_test")
    tt = [str(t) for t in cal["target_types"]]
    onto = Ontology.from_repo(tt)
    par = onto.ancestor_lists()
    bb = get_backbone(args.backbone, tt)
    id_test = load_identity("crc_test")
    g_test = id_test["sample"] if id_test is not None else None
    if args.protocol == "primary":
        Xc, yc = bb.features(cal), cal["y"].astype(np.int64)
        Xt, yt = bb.features(test), test["y"].astype(np.int64)
        gt = g_test
        split_desc = ("cal=crc_cal (100,000 cells, 2 patients); test=crc_test (200,000 cells, "
                      "5 patients)")
    else:
        X, y = bb.features(test), test["y"].astype(np.int64)
        perm = np.random.default_rng(0).permutation(len(y))
        a, b = np.sort(perm[: len(y) // 2]), np.sort(perm[len(y) // 2:])
        Xc, yc, Xt, yt = X[a], y[a], X[b], y[b]
        gt = g_test[b] if g_test is not None else None
        split_desc = "random 50/50 split of crc_test, seed 0 (cell-level exchangeable; NOT patient-disjoint)"
    boot = args.bootstrap if gt is not None else 0
    full = bank["orion_panel"]
    Pc_full = bb.predict17(bb.mask(Xc, full))
    src = SourcePATCH(onto, ALPHA, N_MIN, QUANT).fit(onto.aggregate(Pc_full), yc)
    res = {"backbone": bb.name, "protocol": args.protocol, "split": split_desc, "alpha": ALPHA,
           "patient_ids": ("crc_test identity sidecar `sample`; cluster bootstrap B=%d" % boot) if boot
           else "none (cell-level Wilson intervals only)",
           "n_min": N_MIN, "quantile": QUANT, "arms": ARMS, "n_cal": int(len(yc)), "n_test": int(len(yt)),
           "source_patch": {"supported": [onto.names[c] for c in np.where(src.supported)[0]],
                            "lambda": {onto.names[c]: float(src.lam[c]) for c in np.where(src.supported)[0]}},
           "panels": {}}
    cache = {}
    panels = bank["panels"][: args.limit] if args.limit else bank["panels"]
    for p in panels:
        tp = time.time()
        key = tuple(p["markers"])
        if key in cache:
            res["panels"][p["name"]] = {**res["panels"][cache[key]], "name": p["name"], "family": p["family"],
                                        "alias_of": cache[key]}
            continue
        P = p["markers"]
        Pc = bb.predict17(bb.mask(Xc, P))
        Pt = bb.predict17(bb.mask(Xt, P))
        sets, info = run_arms(onto, par, P, Pc, yc, Pt, Pc_full, src, ALPHA, QUANT, N_MIN)
        adm = onto.admissible(P)
        entry = {"name": p["name"], "family": p["family"], "markers": P, "deleted": p["deleted"],
                 "admissible": [onto.names[c] for c in np.where(adm)[0]],
                 "pm_strata": info["pm_strata"], "arms": {}}
        for arm, S in sets.items():
            m = set_metrics(S, yt, onto, P, supported=info["pm_supported"],
                            per_label=arm in ("patch_pm", "patch_src_backstop", "lac_filt"),
                            groups=gt, bootstrap=boot)
            entry["arms"][arm] = {"headline": headline(m), "metrics": m}
        entry["wall_clock_s"] = round(time.time() - tp, 2)
        res["panels"][p["name"]] = entry
        cache[key] = p["name"]
        h = entry["arms"]
        print(f"[{p['name']:24s}] pm {h['patch_pm']['headline']['exact_cov']:.3f}/"
              f"{h['patch_pm']['headline']['size']:.2f}/abst {h['patch_pm']['headline']['abstain']:.3f} | "
              f"src+bs {h['patch_src_backstop']['headline']['exact_cov']:.3f}/{h['patch_src_backstop']['headline']['size']:.2f} | "
              f"lac_filt {h['lac_filt']['headline']['exact_cov']:.3f} | {entry['wall_clock_s']}s", flush=True)

    # marker x cell-type table: for each single deletion, does pi_P(y) move up the tree?
    labels_present = [int(d) for d in np.unique(yt)]
    adm_full = onto.admissible(full)
    proj_full = onto.projection_map(adm_full)
    table = {}
    for p in bank["panels"]:
        if p["family"] != "single":
            continue
        pm = onto.projection_map(onto.admissible(p["markers"]))
        row = {}
        for d in labels_present:
            moved = pm[d] != proj_full[d]
            row[onto.names[d]] = {"frac_cells_moved_up": float(moved), "n_cells": int((yt == d).sum()),
                                  "pi_full": onto.names[proj_full[d]] if proj_full[d] >= 0 else None,
                                  "pi_P": onto.names[pm[d]] if pm[d] >= 0 else None}
        row["_frac_all_cells_moved"] = float(np.mean(pm[yt] != proj_full[yt]))
        table[p["deleted"][0]] = row
    res["marker_by_celltype_moved_up"] = table
    res["wall_clock_total_s"] = round(time.time() - t0, 1)
    write_json(RUNS / "deletion" / f"deletion_{args.backbone}_{args.protocol}{'_debug' if args.limit else ''}.json", res)
    print(f"done in {res['wall_clock_total_s']} s")


def main_loso(args):
    """Patient-level protocol: pool crc_cal + crc_test (7 samples) and hold out one sample at
    a time (leave-one-sample-out); calibrate on the other six, test on the held-out one. Every
    pooled cell is tested exactly once; pooled metrics carry sample-cluster bootstrap CIs, and
    per-fold headlines are stored."""
    t0 = time.time()
    bank = json.loads((RUNS / "panel_bank.json").read_text())
    cal, test = load_split("crc_cal"), load_split("crc_test")
    tt = [str(t) for t in cal["target_types"]]
    onto = Ontology.from_repo(tt)
    par = onto.ancestor_lists()
    bb = get_backbone(args.backbone, tt)
    ic, it = load_identity("crc_cal"), load_identity("crc_test")
    if args.fake_groups:
        rng = np.random.default_rng(0)
        g = np.concatenate([rng.choice(["F1", "F2"], len(cal["y"])),
                            rng.choice(["F3", "F4", "F5", "F6", "F7"], len(test["y"]))])
        tag = "_FAKEGROUPS_debug"
    else:
        if ic is None or it is None:
            raise SystemExit("patient_loso needs $ADMIT_IDENTITY_DIR with crc_cal/crc_test identity sidecars")
        g = np.concatenate([ic["sample"], it["sample"]])
        tag = ""
    X = np.concatenate([bb.features(cal), bb.features(test)])
    y = np.concatenate([cal["y"], test["y"]]).astype(np.int64)
    split_desc = "crc_cal + crc_test pooled (300,000 cells, 7 samples); leave-one-sample-out"
    if args.protocol == "patient_loso13":
        # + rebuilt X_arc cells of the 6 further Orion patients (crc_test_extra.npz, built in-house);
        # X_arc only, so round0 backbones only. Cap 20,000 cells per patient (seed 0) for runtime.
        if args.backbone not in ("round0", "round0rt"):
            raise SystemExit("patient_loso13 needs an X_arc backbone")
        ex = np.load(RUNS / "crc_rebuild" / "crc_test_extra.npz", allow_pickle=True)
        X = np.concatenate([X, ex["X_arc"].astype(np.float32)])
        y = np.concatenate([y, ex["y"].astype(np.int64)])
        g = np.concatenate([g, ex["sample"].astype(str)])
        rng = np.random.default_rng(0)
        keep = np.sort(np.concatenate([rng.choice(np.flatnonzero(g == s_), min(20000, int((g == s_).sum())),
                                                  replace=False) for s_ in sorted(np.unique(g))]))
        X, y, g = X[keep], y[keep], g[keep]
        split_desc = ("crc_cal + crc_test + 6 rebuilt patients (13 patients), <=20,000 cells per patient "
                      "(seed 0); leave-one-patient-out")
        tag = tag + "13"
    folds = sorted(np.unique(g).tolist())
    full = bank["orion_panel"]
    Pfull = bb.predict17(bb.mask(X, full))
    srcs = {s: SourcePATCH(onto, ALPHA, N_MIN, QUANT).fit(onto.aggregate(Pfull[g != s]), y[g != s]) for s in folds}
    res = {"backbone": bb.name, "protocol": "patient_loso" + tag,
           "split": split_desc,
           "folds": {s: int((g == s).sum()) for s in folds}, "alpha": ALPHA, "n_min": N_MIN, "quantile": QUANT,
           "arms": ARMS, "n_cal": None, "n_test": int(len(y)), "bootstrap": args.bootstrap, "panels": {}}
    cache = {}
    panels = bank["panels"][: args.limit] if args.limit else bank["panels"]
    for p in panels:
        tp = time.time()
        key = tuple(p["markers"])
        if key in cache:
            res["panels"][p["name"]] = {**res["panels"][cache[key]], "name": p["name"], "family": p["family"],
                                        "alias_of": cache[key]}
            continue
        P = p["markers"]
        Pall = bb.predict17(bb.mask(X, P))
        pooled = {}
        fold_rows, nc = {}, {}
        for s in folds:
            ci, ti = g != s, g == s
            sets, info = run_arms(onto, par, P, Pall[ci], y[ci], Pall[ti], Pfull[ci], srcs[s], ALPHA, QUANT, N_MIN)
            for arm, S in sets.items():
                pooled.setdefault(arm, np.zeros((len(y), onto.K), dtype=bool))[ti] = S
            fold_rows[s] = {arm: headline(set_metrics(sets[arm], y[ti], onto, P, per_stratum=False))
                            for arm in ("patch_pm", "patch_src_backstop", "patch_src", "lac_filt", "aps_filt")}
            for k2, v in info["pm_strata"].items():
                nc.setdefault(k2, []).append(v["n_c"])
        adm = onto.admissible(P)
        entry = {"name": p["name"], "family": p["family"], "markers": P, "deleted": p["deleted"],
                 "admissible": [onto.names[c] for c in np.where(adm)[0]],
                 "pm_strata": {k2: {"n_c": int(np.min(v)), "n_c_by_fold": v} for k2, v in nc.items()},
                 "per_fold": fold_rows, "arms": {}}
        for arm, S in pooled.items():
            m = set_metrics(S, y, onto, P, per_label=arm in ("patch_pm", "patch_src_backstop", "lac_filt"),
                            groups=g, bootstrap=args.bootstrap)
            entry["arms"][arm] = {"headline": headline(m), "metrics": m}
        entry["wall_clock_s"] = round(time.time() - tp, 2)
        res["panels"][p["name"]] = entry
        cache[key] = p["name"]
        h = entry["arms"]
        print(f"[{p['name']:24s}] pm {h['patch_pm']['headline']['exact_cov']:.3f} "
              f"{h['patch_pm']['headline']['exact_cov_ci_cluster']} | src+bs "
              f"{h['patch_src_backstop']['headline']['exact_cov']:.3f} | {entry['wall_clock_s']}s", flush=True)
    res["marker_by_celltype_moved_up"] = {}
    res["wall_clock_total_s"] = round(time.time() - t0, 1)
    write_json(RUNS / "deletion" / f"deletion_{args.backbone}_patient_loso{tag}{'_debug' if args.limit else ''}.json", res)
    print(f"done in {res['wall_clock_total_s']} s")


if __name__ == "__main__":
    main()
