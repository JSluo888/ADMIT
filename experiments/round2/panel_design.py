#!/usr/bin/env python3
"""M6: panel design from reference (calibration) data only, with ADMIT+M1 (n_min rule) at alpha 0.10.

Prediction for a candidate panel P uses calibration patients only: leave-one-calibration-patient-out
cross-fitting (fit ADMIT+M1 on the other calibration patients, predict the held-out one). Realized =
fit on all calibration patients, evaluate on held-out test patients. Per cell type: expected finest depth,
leaf-resolved fraction, mean |C|, coverage of the M1 target.
  (a) marker value table: predicted(full) - predicted(full minus m) per marker and cell type
  (b) greedy forward selection from the empty panel; objective = cross-fitted cell-weighted leaf-resolved
      fraction on calibration cells; tie-break: higher finest depth, lower mean |C|, panel order
  (c) validation: predicted vs realized for every bank panel and every greedy panel

  python experiments/round2/panel_design.py --design schurch --seed 0
  python experiments/round2/panel_design.py --design orion
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
from admit.ontology import NODE17  # noqa: E402
from admit.panel_matched import AdmitM1, Ontology, chain_to_leaf, finest_depth, resolved_leaf  # noqa: E402

ALPHA = 0.10


def label_stats(sets, y, target, onto):
    """Per original label and overall: n, coverage (identifiable), leaf-resolved, finest depth, |C|, abstention."""
    size = sets.sum(1)
    dep = finest_depth(sets, onto)
    lr = chain_to_leaf(sets, onto)
    lrc = resolved_leaf(sets, onto) == y
    ident = target >= 0
    hit = np.zeros(len(y), dtype=bool)
    hit[ident] = sets[np.where(ident)[0], target[ident]]
    out = {}
    for d in np.unique(y):
        m = y == d
        mi = m & ident
        out[onto.names[d]] = {"n": int(m.sum()), "n_ident": int(mi.sum()),
                              "cov": float(hit[mi].mean()) if mi.any() else None,
                              "leafres": float(lr[m].mean()), "leafres_correct": float(lrc[m].mean()),
                              "depth": float(dep[m].mean()),
                              "size": float(size[m].mean()), "abst": float((size[m] == 0).mean())}
    out["_overall"] = {"n": int(len(y)), "cov": float(hit[ident].mean()) if ident.any() else None,
                       "leafres": float(lr.mean()), "leafres_correct": float(lrc.mean()),
                       "depth": float(dep.mean()), "size": float(size.mean())}
    types = [k for k in out if not k.startswith("_")]
    out["_type_balanced"] = {"leafres": float(np.mean([out[k]["leafres"] for k in types])),
                             "leafres_correct": float(np.mean([out[k]["leafres_correct"] for k in types])),
                             "depth": float(np.mean([out[k]["depth"] for k in types])),
                             "size": float(np.mean([out[k]["size"] for k in types]))}
    return out


class Design:
    """Holds calibration/test cells and a function producing 17-column label probabilities for a panel."""

    def __init__(self, onto, predict_fn, yc, gc, yt):
        self.onto, self.predict_fn, self.yc, self.gc, self.yt = onto, predict_fn, yc, gc, yt
        self.cal_pats = sorted(np.unique(gc).tolist())

    def predicted(self, panel):
        Pc, _ = self.predict_fn(panel, test=False)
        Nc = self.onto.aggregate(Pc)
        sets = np.zeros_like(Nc, dtype=bool)
        tgt = np.full(len(self.yc), -1)
        for p in self.cal_pats:
            ho = self.gc == p
            m = AdmitM1(self.onto, ALPHA).fit(Nc[~ho], self.yc[~ho], panel)
            sets[ho] = m.predict(Nc[ho])
            tgt[ho] = m.target(self.yc[ho])
        return label_stats(sets, self.yc, tgt, self.onto)

    def realized(self, panel):
        Pc, Pt = self.predict_fn(panel, test=True)
        m = AdmitM1(self.onto, ALPHA).fit(self.onto.aggregate(Pc), self.yc, panel)
        return label_stats(m.predict(self.onto.aggregate(Pt)), self.yt, m.target(self.yt), self.onto)


LINEAGE_START = ["CD45", "PanCK", "CD31", "aSMA"]   # root-level lineage markers


def types_T(des: Design, full, min_n=100):
    """Types T: labels whose full-panel ADMIT+M1 stratum is the label itself with >= min_n calibration cells."""
    Pc, _ = des.predict_fn(full, test=False)
    m = AdmitM1(des.onto, ALPHA).fit(des.onto.aggregate(Pc), des.yc, full)
    g = m.target(des.yc)
    return [des.onto.names[d] for d in np.unique(des.yc)
            if m.gmap[d] == d and int((g == d).sum()) >= min_n]


def tb_score(stats, T):
    lr = [stats[t]["leafres_correct"] for t in T if t in stats]
    dp = [stats[t]["depth"] for t in T if t in stats]
    return (float(np.mean(lr)) if lr else 0.0, float(np.mean(dp)) if dp else 0.0)


def greedy_tb(des: Design, full, T, start=()):
    """Type-balanced greedy: objective = mean over T of per-type correct leaf-resolution;
    tie-break: type-balanced mean depth, then lower cell-weighted |C|, then panel order."""
    chosen = [m for m in full if m in set(start)]
    st0 = des.predicted(list(chosen))
    steps = [{"step": 0, "added": None, "panel": list(chosen), "pred": st0, "real": des.realized(list(chosen)),
              "objective_tb": tb_score(st0, T)[0]}]
    remaining = [m for m in full if m not in set(chosen)]
    while remaining:
        best, best_key, best_stats = None, None, None
        for m in remaining:
            st = des.predicted(chosen + [m])
            lr, dp = tb_score(st, T)
            key = (lr, dp, -st["_overall"]["size"], -full.index(m))
            if best_key is None or key > best_key:
                best, best_key, best_stats = m, key, st
        chosen.append(best)
        remaining.remove(best)
        real = des.realized(list(chosen))
        steps.append({"step": len(chosen), "added": best, "panel": list(chosen), "pred": best_stats, "real": real,
                      "objective_tb": best_key[0], "objective_tb_real": tb_score(real, T)[0]})
        print(f"  tb-greedy {len(chosen):2d} +{best:12s} pred tb {best_key[0]:.3f} real tb {tb_score(real, T)[0]:.3f}",
              flush=True)
    return steps


def greedy(des: Design, full, start=()):
    chosen, steps = [m for m in full if m in set(start)], []
    st0 = des.predicted(list(chosen))
    steps.append({"step": 0, "added": None, "panel": list(chosen), "pred": st0, "real": des.realized(list(chosen))})
    remaining = [m for m in full if m not in set(chosen)]
    while remaining:
        best, best_key, best_stats = None, None, None
        for m in remaining:
            s = des.predicted(chosen + [m])
            o = s["_overall"]
            # objective: correctly leaf-resolved fraction (deviation from the prespecified any-leaf version,
            # which is degenerate: CD45 alone makes every set {Immune, Immune_Other} 'leaf-resolved')
            key = (o["leafres_correct"], o["depth"], -o["size"], -full.index(m))
            if best_key is None or key > best_key:
                best, best_key, best_stats = m, key, s
        chosen.append(best)
        remaining.remove(best)
        steps.append({"step": len(chosen), "added": best, "panel": list(chosen), "pred": best_stats,
                      "real": des.realized(list(chosen))})
        print(f"  greedy step {len(chosen):2d} +{best:12s} pred correct-leafres {best_stats['_overall']['leafres_correct']:.3f} "
              f"depth {best_stats['_overall']['depth']:.2f}", flush=True)
    return steps


def build_schurch(seed, onto, tt):
    from cohort import load_cohort_table, load_config, make_splits
    from run_cohort import fit_head
    cfg = load_config(Path(__file__).resolve().parents[2] / "configs" / "schurch.json")
    d = load_cohort_table(cfg)
    idx17 = {t: i for i, t in enumerate(tt)}
    y = np.array([idx17[n] for n in d["label_name"]], dtype=np.int64)
    x, patient, markers = d["x"], np.asarray(d["patient"]).astype(str), d["markers"]
    sp = [s for s in make_splits(patient, cfg["splits"]) if s["seed"] == seed][0]
    rng = sp["rng"]
    tr = np.concatenate([rng.choice(np.flatnonzero(patient == g), min(2000, int((patient == g).sum())), replace=False)
                         for g in sp["train"]])
    ci = np.flatnonzero(np.isin(patient, sp["cal"]))
    ti = np.flatnonzero(np.isin(patient, sp["test"]))
    mu, sd = x[tr].mean(0), x[tr].std(0)
    sd[sd == 0] = 1.0
    head, present, _ = fit_head("logistic", (x[tr] - mu) / sd, y[tr], seed)
    Zc, Zt = (x[ci] - mu) / sd, (x[ti] - mu) / sd

    def p17(Z):
        P = np.zeros((len(Z), len(tt)))
        P[:, present] = head.predict_proba(Z)
        return P / P.sum(1, keepdims=True)

    def predict_fn(panel, test):
        keep = np.array([m in set(panel) for m in markers])
        zc = Zc.copy(); zc[:, ~keep] = 0.0
        Pc = p17(zc)
        if not test:
            return Pc, None
        zt = Zt.copy(); zt[:, ~keep] = 0.0
        return Pc, p17(zt)

    des = Design(onto, predict_fn, y[ci], patient[ci], y[ti])
    bank = json.loads(Path(cfg["panel_bank"]).read_text())
    return des, markers, bank, {"seed": seed, "cal_patients": sp["cal"].tolist(), "test_patients": sp["test"].tolist()}


def build_orion(onto, tt):
    bb = get_backbone("round0", tt)
    cal, test = load_split("crc_cal"), load_split("crc_test")
    ic, it = load_identity("crc_cal"), load_identity("crc_test")
    Xc = np.concatenate([bb.features(cal), bb.features(test)])
    yc = np.concatenate([cal["y"], test["y"]]).astype(np.int64)
    gc = np.concatenate([ic["sample"], it["sample"]])
    ex = np.load(RUNS / "crc_rebuild" / "crc_test_extra.npz", allow_pickle=True)
    Xt, yt = ex["X_arc"].astype(np.float32), ex["y"].astype(np.int64)
    cache = {}

    def predict_fn(panel, test):
        key = tuple(panel)
        if key not in cache:
            cache.clear()
            cache[key] = {"c": bb.predict17(bb.mask(Xc, panel))}
        if test and "t" not in cache[key]:
            cache[key]["t"] = bb.predict17(bb.mask(Xt, panel))
        return cache[key]["c"], cache[key].get("t")

    bank = json.loads((RUNS / "panel_bank.json").read_text())
    des = Design(onto, predict_fn, yc, gc, yt)
    return des, list(bank["orion_panel"]), {"panels": bank["panels"]}, {
        "cal_patients": sorted(np.unique(gc).tolist()), "test_patients": sorted(np.unique(ex["sample"]).tolist())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True, choices=["schurch", "orion"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default="empty", choices=["empty", "lineage"])
    ap.add_argument("--objective", default="cellweighted", choices=["cellweighted", "typebalanced"])
    args = ap.parse_args()
    t0 = time.time()
    tt = list(NODE17)          # the 17-node column order shared by every experiment
    onto = Ontology.from_repo(tt)
    if args.design == "schurch":
        des, full, bank, info = build_schurch(args.seed, onto, tt)
    else:
        des, full, bank, info = build_orion(onto, tt)
    res = {"design": args.design, "info": info, "alpha": ALPHA, "method": "ADMIT+M1 (n_min rule)",
           "full_panel": full, "greedy_start": args.start}
    if args.objective == "typebalanced":   # greedy only + budgets
        T = types_T(des, full)
        res["objective"] = "type-balanced mean over T of correct leaf-resolution"
        res["types_T"] = T
        steps = greedy_tb(des, full, T, LINEAGE_START if args.start == "lineage" else ())
        res["greedy"] = steps
        res["budget"] = {}
        for k in (8, 12, 16, 24):
            hit = [st for st in steps if len(st["panel"]) == k]
            res["budget"][str(k)] = ({"panel": hit[0]["panel"], "pred_tb": hit[0]["objective_tb"],
                                      "real_tb": hit[0].get("objective_tb_real", tb_score(hit[0]["real"], T)[0]),
                                      "pred": hit[0]["pred"], "real": hit[0]["real"]}
                                     if hit else "not applicable (panel has fewer markers or start is larger)")
        res["wall_clock_s"] = round(time.time() - t0, 1)
        tag = f"greedy_tb_{args.start}_{args.design}" + (f"_seed{args.seed}" if args.design == "schurch" else "")
        write_json(RUNS / "round2" / f"{tag}.json", res)
        print("done", res["wall_clock_s"])
        return
    if args.start == "lineage":   # greedy only (marker value and bank are in the empty-start run)
        res["greedy"] = greedy(des, full, LINEAGE_START)
        res["wall_clock_s"] = round(time.time() - t0, 1)
        tag = f"greedy_lineage_{args.design}" + (f"_seed{args.seed}" if args.design == "schurch" else "")
        write_json(RUNS / "round2" / f"{tag}.json", res)
        print("done", res["wall_clock_s"])
        return
    # (a) marker value
    pf = des.predicted(full)
    mv = {}
    for m in full:
        pm = des.predicted([x for x in full if x != m])
        mv[m] = {k: {s: (None if (pf[k].get(s) is None or pm.get(k, {}).get(s) is None) else pf[k][s] - pm[k][s])
                     for s in ("leafres", "leafres_correct", "depth", "size", "cov")}
                 for k in pf if not k.startswith("_type")}
    res["marker_value"] = {"pred_full": pf, "delta_full_minus_removed": mv}
    print("marker value done", round(time.time() - t0, 1), flush=True)
    # (c) bank panels: predicted vs realized
    res["bank"] = {}
    for p in bank["panels"]:
        res["bank"][p["name"]] = {"panel": p["markers"], "pred": des.predicted(p["markers"]),
                                  "real": des.realized(p["markers"])}
    print("bank done", round(time.time() - t0, 1), flush=True)
    # (b) greedy
    res["greedy"] = greedy(des, full)
    res["wall_clock_s"] = round(time.time() - t0, 1)
    tag = f"panel_design_{args.design}" + (f"_seed{args.seed}" if args.design == "schurch" else "")
    write_json(RUNS / "round2" / f"{tag}.json", res)
    print("done", res["wall_clock_s"])


if __name__ == "__main__":
    main()
