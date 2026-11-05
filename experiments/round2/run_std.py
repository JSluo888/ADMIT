#!/usr/bin/env python3
"""Generic ADMIT runner for a cohort directory in the standard schema (data/tier1/<c>/, data/tier2_std/<c>/).

Standard schema (see README.md):
  cells.parquet   cell_id, patient_id, sample_id, x, y, one float column per marker (markers.tsv source_name),
                  label columns, and `node17` (17-node label; anything else = excluded and counted)
  markers.tsv     source_name, vocabulary_name ("NA" = not in the 40-marker vocabulary), is_lineage, notes
  label_mapping.json, QC.json, clinical.csv (not needed by the runner)

Protocols
  deletion     paired marker deletion with patient splits (train/cal/test patients, default 15:12:8 of patients,
               20 seeds); panel bank generated from the cohort panel by the round-1 rules and written BEFORE results
  calib_curve  full cohort panel; calibrate on n labelled cells drawn from the calibration patients,
               n in {50,100,200,500,1000,2000} x 10 draws, evaluate on the test patients (per split)
  cross_panel  two cohort dirs with the same patients and different panels (A = --cohort-dir, B = --cohort-dir-b):
               head trained on A train patients; ADMIT/ADMIT+M1/comparators calibrated on A cal patients masked to
               A cap B (vocabulary markers), tested on B cells of the test patients under panel B (features placed by
               vocabulary name; each cohort z-scored on its own training-patient statistics); full-panel (A)
               calibration = old PATCH; run it twice with A and B swapped for the reverse direction
All protocols write an additive accumulator acc[panel, arm, alpha, patient, node, stat] (engine.py) that
summarize_round2.py reduces (per-stratum and per-type coverage with patient-cluster bootstrap CIs, equal-patient
coverage, resolution, alpha curves).

Usage: see README.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, load_split, write_json  # noqa: E402
from cohort import make_bank  # noqa: E402
from engine import ALPHAS, NSTAT, STATS, all_arm_names, evaluate_block  # noqa: E402
from run_cohort import fit_head  # noqa: E402
from admit.ontology import NODE17  # noqa: E402
from admit.panel_matched import Ontology  # noqa: E402

N17 = list(NODE17)          # the 17-node column order shared by every experiment


def load_std(cdir: Path, transform: str, cell_cap: int, seed: int = 0) -> dict:
    mk = pd.read_csv(cdir / "markers.tsv", sep="\t", keep_default_na=False)
    src = mk["source_name"].tolist()
    voc = [v if v not in ("", "NA") else None for v in mk["vocabulary_name"].tolist()]
    names, seen = [], set()
    for s, v in zip(src, voc):
        n = v if v else f"raw:{s}"
        while n in seen:
            n = n + "_dup"
        seen.add(n)
        names.append(n)
    df = pd.read_parquet(cdir / "cells.parquet", columns=["patient_id", "node17"] + src)
    lab = df["node17"].astype(str)
    keep = lab.isin(N17).to_numpy()
    excluded = lab[~keep].value_counts().to_dict()
    df = df.loc[keep]
    # per-patient cell cap (seeded) to keep runs tractable; recorded in the output
    if cell_cap:
        rng = np.random.default_rng(seed)
        idx = []
        for _, g in df.groupby("patient_id", observed=True).indices.items():
            idx.append(g if len(g) <= cell_cap else rng.choice(g, cell_cap, replace=False))
        df = df.iloc[np.sort(np.concatenate(idx))]
    x = df[src].to_numpy(dtype=np.float64)
    if transform == "log1p":
        x = np.log1p(np.clip(x, 0, None))
    elif transform.startswith("arcsinh"):
        x = np.arcsinh(x / float(transform.split(":")[1]))
    return {"x": x, "markers": names, "vocab_markers": [n for n in names if not n.startswith("raw:")],
            "y": np.array([N17.index(v) for v in df["node17"].astype(str)], dtype=np.int64),
            "patient": df["patient_id"].astype(str).to_numpy(), "excluded": {str(k): int(v) for k, v in excluded.items()},
            "transform": transform, "cell_cap": cell_cap}


def splits_for(patients, seeds, fractions):
    ids = np.unique(patients)
    f = np.array(fractions, dtype=float)
    n_tr, n_ca = (np.floor(f[:2] / f.sum() * len(ids))).astype(int)
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        perm = rng.permutation(ids)
        out.append({"seed": s, "rng": rng, "train": perm[:n_tr], "cal": perm[n_tr:n_tr + n_ca], "test": perm[n_tr + n_ca:]})
    return out


def train_split(d, sp, head, train_cap, max_train):
    rng = sp["rng"]
    tr = np.concatenate([rng.choice(np.flatnonzero(d["patient"] == g), min(train_cap, int((d["patient"] == g).sum())),
                                    replace=False) for g in sp["train"]])
    if max_train and len(tr) > max_train:
        tr = np.sort(rng.choice(tr, max_train, replace=False))
    mu, sd = d["x"][tr].mean(0), d["x"][tr].std(0)
    sd[sd == 0] = 1.0
    m, present, note = fit_head(head, (d["x"][tr] - mu) / sd, d["y"][tr], int(sp["seed"]))
    return m, present, note, mu, sd, len(tr)


def p17_fn(m, present):
    def f(Z):
        P = np.zeros((len(Z), len(N17)))
        P[:, present] = m.predict_proba(Z)
        return P / P.sum(1, keepdims=True)
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort-dir", required=True)
    ap.add_argument("--cohort-dir-b", help="cross_panel: second panel on the same patients")
    ap.add_argument("--protocol", required=True, choices=["deletion", "calib_curve", "cross_panel"])
    ap.add_argument("--head", default="logistic", choices=["logistic", "xgb"])
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--fractions", default="15,12,8", help="train,cal,test patient proportions")
    ap.add_argument("--transform", default="log1p", help="log1p | arcsinh:<cofactor> | none")
    ap.add_argument("--cell-cap", type=int, default=2000, help="max cells per patient (0 = all)")
    ap.add_argument("--train-cap", type=int, default=2000, help="training cells per training patient")
    ap.add_argument("--max-train", type=int, default=60000, help="max training cells overall")
    ap.add_argument("--out", default=str(RUNS / "round2" / "std"))
    ap.add_argument("--limit", type=int, default=0, help="debug: first N panels")
    args = ap.parse_args()
    t0 = time.time()
    a, b = (args.seeds.split("-") + [None])[:2]
    seeds = list(range(int(a), int(b) + 1)) if b else [int(v) for v in args.seeds.split(",")]
    fr = [float(v) for v in args.fractions.split(",")]
    cdir = Path(args.cohort_dir)
    name = cdir.name
    outdir = Path(args.out) / name
    outdir.mkdir(parents=True, exist_ok=True)
    onto = Ontology.from_repo(N17)
    d = load_std(cdir, args.transform, args.cell_cap)
    if args.protocol == "cross_panel":
        db = load_std(Path(args.cohort_dir_b), args.transform, args.cell_cap)
        common_v = [m for m in d["vocab_markers"] if m in set(db["vocab_markers"])]
        tag_panel = f"{name}__to__{Path(args.cohort_dir_b).name}"
        panels = [{"name": "A_cap_B", "family": "cross_panel", "markers": common_v,
                   "deleted": [m for m in d["markers"] if m not in set(common_v)]}]
        bank = {"panels": panels, "full_panel": d["markers"]}
    else:
        bank = make_bank(d["markers"], outdir / "panel_bank.json")      # fixed before results
        panels = bank["panels"]
        tag_panel = name
    uniq, alias = [], {}
    for p in panels:
        for q in uniq:
            if q["markers"] == p["markers"]:
                alias[p["name"]] = q["name"]
                break
        else:
            uniq.append(p)
    if args.protocol == "calib_curve":
        uniq = [p for p in uniq if p["name"] == "full"]
        alias = {}
    if args.limit:
        uniq = uniq[: args.limit]
    pats_all = np.unique(np.concatenate([d["patient"]] + ([db["patient"]] if args.protocol == "cross_panel" else [])))
    pidx = {p: i for i, p in enumerate(pats_all.tolist())}
    arms = all_arm_names(True)
    arm_index = {x: i for i, x in enumerate(arms)}
    NS = [50, 100, 200, 500, 1000, 2000]
    n_blocks = len(NS) if args.protocol == "calib_curve" else len(uniq)
    acc = np.zeros((n_blocks, len(arms), len(ALPHAS), len(pats_all), onto.K, NSTAT))
    split_meta = []
    for sp in splits_for(d["patient"], seeds, fr):
        ts = time.time()
        m, present, note, mu, sd, ntr = train_split(d, sp, args.head, args.train_cap, args.max_train)
        p17 = p17_fn(m, present)
        ci = np.flatnonzero(np.isin(d["patient"], sp["cal"]))
        if args.protocol == "cross_panel":
            # B cells of the test patients, placed into A's columns by vocabulary name, z-scored on B's own
            # training-patient statistics; unmeasured columns = 0 (training mean)
            shared = np.intersect1d(np.unique(d["patient"]), np.unique(db["patient"]))
            if len(shared) >= 3:      # same patients on both panels: test = B cells of A's test patients
                tB = np.flatnonzero(np.isin(db["patient"], sp["test"]))
                trB = np.flatnonzero(np.isin(db["patient"], sp["train"]))
            else:                     # different patients (two labs): test on all B patients, B z-scored on itself
                tB = np.arange(len(db["y"]))
                trB = tB
            muB, sdB = db["x"][trB].mean(0), db["x"][trB].std(0)
            sdB[sdB == 0] = 1.0
            ZB = np.zeros((len(tB), len(d["markers"])))
            for j, mname in enumerate(d["markers"]):
                if mname in set(common_v):
                    jb = db["markers"].index(mname)
                    ZB[:, j] = (db["x"][tB, jb] - muB[jb]) / sdB[jb]
            yt, gt = db["y"][tB], db["patient"][tB]
        else:
            ti = np.flatnonzero(np.isin(d["patient"], sp["test"]))
            yt, gt = d["y"][ti], d["patient"][ti]
        Zc = (d["x"][ci] - mu) / sd
        Pc_full = p17(Zc)
        split_meta.append({"seed": int(sp["seed"]), "n_train": int(ntr), "n_cal": int(len(ci)), "n_test": int(len(yt)),
                           "train": sp["train"].tolist(), "cal": sp["cal"].tolist(), "test": sp["test"].tolist(),
                           "head_note": note})
        pslot = np.array([pidx[g] for g in gt])
        if args.protocol == "calib_curve":
            Pt = p17((d["x"][ti] - mu) / sd)
            rng = np.random.default_rng(1000 + int(sp["seed"]))
            for bi, n in enumerate(NS):
                for draw in range(10):
                    sub = np.sort(rng.choice(len(ci), min(n, len(ci)), replace=False))
                    evaluate_block(acc, bi, onto, d["markers"], Pc_full[sub], d["y"][ci][sub], Pt, yt, pslot,
                                   len(pats_all), Pc_full[sub], True, gc=d["patient"][ci][sub], arm_index=arm_index,
                                   src_cache={})
        else:
            for pi, p in enumerate(uniq):
                keep = np.array([mm in set(p["markers"]) for mm in d["markers"]])
                zc = Zc.copy()
                zc[:, ~keep] = 0.0
                if args.protocol == "cross_panel":
                    zt = ZB
                    test_panel = db["markers"]                 # deployment panel = B (A(.) uses B's markers)
                else:
                    zt = (d["x"][ti] - mu) / sd
                    zt[:, ~keep] = 0.0
                    test_panel = p["markers"]
                Pc, Pt = p17(zc), p17(zt)
                if args.protocol == "cross_panel":
                    # A(P_B) for the test panel; calibration strata use the same A(P_B) via the panel argument
                    evaluate_block(acc, pi, onto, [mm for mm in test_panel if not mm.startswith("raw:")], Pc,
                                   d["y"][ci], Pt, yt, pslot, len(pats_all), Pc_full, True, gc=d["patient"][ci],
                                   arm_index=arm_index, src_cache={})
                else:
                    evaluate_block(acc, pi, onto, test_panel, Pc, d["y"][ci], Pt, yt, pslot, len(pats_all), Pc_full,
                                   True, gc=d["patient"][ci], arm_index=arm_index, src_cache={})
        print(f"[{name} {args.protocol} {args.head} seed {sp['seed']}] {time.time()-ts:.1f}s", flush=True)
    block_names = [f"n{n}" for n in NS] if args.protocol == "calib_curve" else [p["name"] for p in uniq]
    tag = f"{tag_panel}_{args.protocol}_{args.head}_s{args.seeds}{'_debug' if args.limit else ''}"
    np.savez_compressed(outdir / f"{tag}.npz", acc=acc.astype(np.float32), panels=np.array(block_names),
                        patients=pats_all, arms=np.array(arms), alphas=np.array(ALPHAS), stats=np.array(STATS),
                        nodes=np.array(onto.names), seeds=np.array(seeds))
    write_json(outdir / f"{tag}_meta.json",
               {"cohort_dir": str(cdir), "cohort_dir_b": args.cohort_dir_b, "protocol": args.protocol,
                "head": args.head, "transform": args.transform, "cell_cap": args.cell_cap, "train_cap": args.train_cap,
                "max_train": args.max_train, "fractions": fr, "splits": split_meta, "panel_alias": alias,
                "families": {p["name"]: p["family"] for p in panels} if args.protocol != "calib_curve"
                else {f"n{n}": "calib_curve" for n in NS},
                "deleted": {p["name"]: p["deleted"] for p in panels}, "excluded_labels": d["excluded"],
                "markers": d["markers"], "n_cells_used": int(len(d["y"])), "n_patients": int(len(np.unique(d["patient"]))),
                "wall_clock_s": round(time.time() - t0, 1)})
    print("done", round(time.time() - t0, 1), outdir / f"{tag}.npz")


if __name__ == "__main__":
    main()
