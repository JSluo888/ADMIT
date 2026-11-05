#!/usr/bin/env python3
"""Precision, recall and F1 of forced labels and ADMIT calls under the paired-deletion panels (post hoc, one split).

Same loaders, splits, caps, panels and head as run_within3.compute3; ADMIT is the admit_m1 arm at alpha 0.10.
Each ADMIT set becomes one call: its finest type if the set lies on one path of the hierarchy, otherwise the lowest
common ancestor of its types; an empty set, or a set spanning lineages, is no call. Both methods are scored against the
original reference label (not the panel's revised target), on all test cells and on the cells whose
reference label has an admissible ancestor under the panel (identifiable):
  flat          call == reference label; per-type precision, recall, F1 and their macro mean (types >= 50 test cells)
  hierarchical  Kiritchenko et al.: hP = sum|A(call) & A(ref)| / sum|A(call)|, hR = sum|A(call) & A(ref)| / sum|A(ref)|,
                A(.) = ancestors-or-self; a correct but coarse call keeps full precision and loses recall.

  ADMIT_XGB_DEVICE=cpu python experiments/round3/prf3.py --cohort launonen2022_hgsc_tcycif --seed 0 [--limit 2]
Output: runs/prf/<cohort>_xgb_seed<k>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cohorts"))
sys.path.insert(0, str(HERE))
import std  # noqa: E402
from engine3 import maps_for, run_arms3  # noqa: E402
from run_within3 import unique_panels  # noqa: E402
from admit.panel_matched import Ontology  # noqa: E402

OUT = std.OUT.parent / "prf"
MIN_TYPE = 50


def admit_calls(sets: np.ndarray, onto: Ontology) -> np.ndarray:
    anc, depth = onto.anc, onto.depth
    nonempty = sets.any(1)
    deepest = np.argmax(np.where(sets, depth[None, :], 0), 1)
    chain = ~(sets & ~anc[:, deepest].T).any(1)
    bad = (sets.astype(np.int32) @ (~anc).T.astype(np.int32)) > 0      # bad[n, a]: some set type is not under a
    common = ~bad & nonempty[:, None]
    lca = np.where(common.any(1), np.argmax(np.where(common, depth[None, :], 0), 1), -1)
    return np.where(~nonempty, -1, np.where(chain, deepest, lca))


def scores(call: np.ndarray, y: np.ndarray, onto: Ontology) -> dict:
    anc = onto.anc.astype(np.int32)
    made = call >= 0
    inter = np.zeros(len(y))
    inter[made] = (anc[:, call[made]] * anc[:, y[made]]).sum(0)
    hp = inter.sum() / max(anc[:, call[made]].sum(), 1)
    hr = inter.sum() / anc[:, y].sum()
    per = {}
    for t in np.unique(y):
        n_t = int((y == t).sum())
        if n_t < MIN_TYPE:
            continue
        tp = int(((call == t) & (y == t)).sum())
        n_call = int((call == t).sum())
        p = tp / n_call if n_call else 0.0
        r = tp / n_t
        per[onto.names[t]] = {"n": n_t, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}
    vals = list(per.values())
    return {"macro_precision": float(np.mean([v["precision"] for v in vals])),
            "macro_recall": float(np.mean([v["recall"] for v in vals])),
            "macro_f1": float(np.mean([v["f1"] for v in vals])),
            "call_rate": float(made.mean()),
            "precision_of_calls": float((call[made] == y[made]).mean()) if made.any() else float("nan"),
            "h_precision": float(hp), "h_recall": float(hr), "h_f1": float(2 * hp * hr / (hp + hr)) if hp + hr else 0.0,
            "per_type": per}


def compute(d, bank, reg, cohort, seed, onto, limit=0):
    t0 = time.time()
    assert bank["full_panel"] == d["markers"]
    panels, _ = unique_panels(bank)
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
    std.fit_transform(x, reg["transform"], tr)
    mu, sd = x[tr].mean(0), x[tr].std(0)
    sd[sd == 0] = 1.0
    z = lambda rows: ((x[rows] - mu) / sd).astype(np.float32)  # noqa: E731
    head = std.Head("xgb").fit(z(tr), y[tr], seed)
    Zc, Zt = z(ci), z(ti)
    yc, yt, ytr = y[ci], y[ti], y[tr]
    Pc_full = head.predict17(Zc)
    out = []
    for pi, p in enumerate(panels):
        miss = [j for j, m in enumerate(d["markers"]) if m not in set(p["markers"])]
        zc, zt = Zc.copy(), Zt.copy()
        zc[:, miss] = 0.0
        zt[:, miss] = 0.0
        Pc, Pt = head.predict17(zc), head.predict17(zt)
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
        print(f"[{cohort} s{seed}] panel {pi + 1}/{len(panels)} {time.time() - t0:.0f}s", flush=True)
    return {"cohort": cohort, "seed": seed, "head": "xgb", "alpha": 0.10, "min_type_cells": MIN_TYPE,
            "n_test_patients": int(len(np.unique(pcode[ti]))), "panels": out, "wall_clock_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    reg = std.registry()[args.cohort]
    bank = json.loads((std.OUT / args.cohort / "panel_bank.json").read_text())
    d = std.load_std(args.cohort, reg["transform"], cohort_key=reg.get("cohort_key"),
                     sample_as_patient=reg.get("sample_as_patient", False), synonyms=reg.get("synonyms"))
    onto = Ontology.from_repo(d["target_types"])
    res = compute(d, bank, reg, args.cohort, args.seed, onto, args.limit)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.cohort}_xgb_seed{args.seed}{'_debug' if args.limit else ''}"
    std.write_json(OUT / f"{tag}.json", res)
    print(f"[{args.cohort}] done {res['wall_clock_s']}s -> {OUT / (tag + '.json')}", flush=True)


if __name__ == "__main__":
    main()
