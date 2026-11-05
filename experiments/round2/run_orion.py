#!/usr/bin/env python3
"""Round 2 on Orion (round0 backbone): exchangeable / LOPO-7 / LOPO-13 with ADMIT, ADMIT+M1 and comparators
at 4 alphas; writes the additive accumulator (round2/orion_<protocol>.npz) + a small JSON of strata reports.

  python experiments/round2/run_orion.py --protocol exchangeable|lopo7|lopo13
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, get_backbone, load_identity, load_split, write_json  # noqa: E402
from engine import ALPHAS, NSTAT, STATS, all_arm_names, evaluate_block  # noqa: E402
from admit.panel_matched import Ontology  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, choices=["exchangeable", "lopo7", "lopo13"])
    ap.add_argument("--backbone", default="round0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    bank = json.loads((RUNS / "panel_bank.json").read_text())
    cal, test = load_split("crc_cal"), load_split("crc_test")
    tt = [str(t) for t in cal["target_types"]]
    onto = Ontology.from_repo(tt)
    bb = get_backbone(args.backbone, tt)
    ic, it = load_identity("crc_cal"), load_identity("crc_test")
    if args.protocol == "exchangeable":
        X, y, g = bb.features(test), test["y"].astype(np.int64), it["sample"]
        perm = np.random.default_rng(0).permutation(len(y))
        a, b = np.sort(perm[: len(y) // 2]), np.sort(perm[len(y) // 2:])
        blocks = [(a, b)]
        desc = "random 50/50 split of crc_test (seed 0); clusters = 5 test patients"
    else:
        X = np.concatenate([bb.features(cal), bb.features(test)])
        y = np.concatenate([cal["y"], test["y"]]).astype(np.int64)
        g = np.concatenate([ic["sample"], it["sample"]])
        desc = "crc_cal + crc_test (7 patients), leave-one-patient-out"
        if args.protocol == "lopo13":
            ex = np.load(RUNS / "crc_rebuild" / "crc_test_extra.npz", allow_pickle=True)
            X = np.concatenate([X, ex["X_arc"].astype(np.float32)])
            y = np.concatenate([y, ex["y"].astype(np.int64)])
            g = np.concatenate([g, ex["sample"].astype(str)])
            rng = np.random.default_rng(0)
            keep = np.sort(np.concatenate([rng.choice(np.flatnonzero(g == s), min(20000, int((g == s).sum())),
                                                      replace=False) for s in sorted(np.unique(g))]))
            X, y, g = X[keep], y[keep], g[keep]
            desc = "13 patients (<=20,000 cells each, seed 0), leave-one-patient-out"
        blocks = [(np.flatnonzero(g != s), np.flatnonzero(g == s)) for s in sorted(np.unique(g))]
    patients = sorted(np.unique(g).tolist())
    pidx = {p: i for i, p in enumerate(patients)}
    pslot_all = np.array([pidx[s] for s in g])
    arms = all_arm_names(False)
    arm_index = {a: i for i, a in enumerate(arms)}
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
    acc = np.zeros((len(uniq), len(arms), len(ALPHAS), len(patients), onto.K, NSTAT))
    full = bank["orion_panel"]
    Pfull = bb.predict17(bb.mask(X, full))
    reports = {}
    for pi, p in enumerate(uniq):
        tp = time.time()
        P = p["markers"]
        Pall = bb.predict17(bb.mask(X, P))
        reps = []
        for ci, ti in blocks:
            r = evaluate_block(acc, pi, onto, P, Pall[ci], y[ci], Pall[ti], y[ti], pslot_all[ti], len(patients),
                               Pfull[ci], False, arm_index=arm_index, src_cache={})
            reps.append(r)
        reports[p["name"]] = reps[0] if len(reps) == 1 else {"first_fold": reps[0]}
        cov = acc[pi, arm_index["admit"], 1, :, :, 0].sum() / max(acc[pi, arm_index["admit"], 1, :, :, 1].sum(), 1)
        cov1 = acc[pi, arm_index["admit_m1"], 1, :, :, 0].sum() / max(acc[pi, arm_index["admit_m1"], 1, :, :, 1].sum(), 1)
        print(f"[{args.protocol} {p['name']:24s}] admit {cov:.3f} m1 {cov1:.3f} {time.time()-tp:.1f}s", flush=True)
    sfx = "_debug" if args.limit else ""
    out = RUNS / "round2" / f"orion_{args.protocol}{sfx}.npz"
    np.savez_compressed(out, acc=acc.astype(np.float32), panels=np.array([p["name"] for p in uniq]),
                        patients=np.array(patients), arms=np.array(arms), alphas=np.array(ALPHAS),
                        stats=np.array(STATS), nodes=np.array(onto.names))
    write_json(RUNS / "round2" / f"orion_{args.protocol}{sfx}_meta.json",
               {"protocol": args.protocol, "backbone": bb.name, "desc": desc, "patients": patients,
                "panel_alias": alias, "families": {p["name"]: p["family"] for p in bank["panels"]},
                "deleted": {p["name"]: p["deleted"] for p in bank["panels"]},
                "strata_reports": reports, "wall_clock_s": round(time.time() - t0, 1)})
    print("done", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
