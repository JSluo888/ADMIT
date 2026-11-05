#!/usr/bin/env python3
"""Panel design with ADMIT-M1.

Same designs, splits, heads and objectives as round2/panel_design.py, with ADMIT-M1 = frozen support map (training
labels: the split's head-training cells for Schurch, crc_train_full for Orion) + exact calibration (+inf rule), and the
requirement map R(c) = required_positive | required_measured. Predictions: leave-one-calibration-patient-out cross-fit (the frozen map's expected counts
use the cells each fold calibrates on). Realised: calibrate on all calibration patients, evaluate on held-out test
patients.

parts (comma list, default all):
  bank        marker value table + every bank panel predicted vs realised   -> design_<d>[_seedS]_bank.json
  greedy_cw   cell-weighted greedy from the empty panel                       -> ..._greedy_cw.json
  tb_empty    type-balanced greedy from the empty panel                       -> ..._tb_empty.json
  tb_lineage  type-balanced greedy from the lineage start                     -> ..._tb_lineage.json
  validate    matched-marker-count validation: greedy (each start) vs 20 random panels per k vs the R(c)-completion
              heuristic vs the bank's random-retention panels, realised on test patients with per-patient counts
              (needs tb_empty / tb_lineage)                                   -> ..._validate.json + _validate_pp.npz

  python experiments/round3/panel_design3.py --design schurch --seed 0 [--parts bank,tb_empty]
  python experiments/round3/panel_design3.py --design orion --parts tb_lineage
The Orion design needs the in-house Orion caches (see common.py).
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
from common import RUNS, get_backbone, load_identity, load_split, write_json  # noqa: E402
from admit.ontology import NODE17  # noqa: E402
from admit.panel_matched import (ISOFORM_SATISFIERS, AdmitM1Frozen, Ontology, chain_to_leaf,  # noqa: E402
                                 finest_depth, frozen_support_map, nmin_rule, resolved_leaf)

ALPHA = 0.10
OUT = RUNS / "round3" / "panel_design"
LINEAGE_START = ["CD45", "PanCK", "CD31", "aSMA"]
KS = [4, 8, 12, 16, 20, 24]
N_RANDOM = 20


def label_stats(sets, y, target, onto):
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
                              "depth": float(dep[m].mean()), "size": float(size[m].mean()),
                              "abst": float((size[m] == 0).mean())}
    out["_overall"] = {"n": int(len(y)), "cov": float(hit[ident].mean()) if ident.any() else None,
                       "leafres": float(lr.mean()), "leafres_correct": float(lrc.mean()),
                       "depth": float(dep.mean()), "size": float(size.mean()),
                       "frac_no_supported_target": float((~ident).mean())}
    types = [k for k in out if not k.startswith("_")]
    out["_type_balanced"] = {k: float(np.mean([out[t][k] for t in types]))
                             for k in ("leafres", "leafres_correct", "depth", "size")}
    return out, lrc


class Design3:
    def __init__(self, onto, predict_fn, yc, gc, yt, gt, y_train):
        self.onto, self.predict_fn = onto, predict_fn
        self.yc, self.gc, self.yt, self.gt, self.y_train = yc, gc, yt, gt, y_train
        self.cal_pats = sorted(np.unique(gc).tolist())
        self.test_pats = np.array(sorted(np.unique(gt).tolist()))
        self.tslot = np.searchsorted(self.test_pats, gt)

    def _fit(self, panel, Nc, yc):
        adm = self.onto.admissible(panel)
        gmap, S, _ = frozen_support_map(self.onto, adm, self.y_train, len(yc), nmin_rule(ALPHA))
        return AdmitM1Frozen(self.onto, ALPHA, gmap, S).fit(Nc, yc), gmap

    def predicted(self, panel):
        Pc, _ = self.predict_fn(panel, test=False)
        Nc = self.onto.aggregate(Pc)
        sets = np.zeros_like(Nc, dtype=bool)
        tgt = np.full(len(self.yc), -1)
        for p in self.cal_pats:
            ho = self.gc == p
            m, gmap = self._fit(panel, Nc[~ho], self.yc[~ho])
            sets[ho] = m.predict(Nc[ho])
            tgt[ho] = gmap[self.yc[ho]]
        return label_stats(sets, self.yc, tgt, self.onto)[0]

    def realized(self, panel, per_patient=False):
        Pc, Pt = self.predict_fn(panel, test=True)
        m, gmap = self._fit(panel, self.onto.aggregate(Pc), self.yc)
        st, lrc = label_stats(m.predict(self.onto.aggregate(Pt)), self.yt, gmap[self.yt], self.onto)
        if not per_patient:
            return st
        K = self.onto.K
        idx = self.tslot * K + self.yt
        pp = np.stack([np.bincount(idx, weights=lrc.astype(float), minlength=len(self.test_pats) * K),
                       np.bincount(idx, minlength=len(self.test_pats) * K).astype(float)], 1)
        return st, pp.reshape(len(self.test_pats), K, 2)


def types_T(des: Design3, full, min_n=100):
    Pc, _ = des.predict_fn(full, test=False)
    m, gmap = des._fit(full, des.onto.aggregate(Pc), des.yc)
    g = gmap[des.yc]
    return [des.onto.names[d] for d in np.unique(des.yc) if gmap[d] == d and int((g == d).sum()) >= min_n]


def tb_score(stats, T):
    lr = [stats[t]["leafres_correct"] for t in T if t in stats]
    dp = [stats[t]["depth"] for t in T if t in stats]
    return (float(np.mean(lr)) if lr else 0.0, float(np.mean(dp)) if dp else 0.0)


def greedy_tb(des, full, T, start=()):
    chosen = [m for m in full if m in set(start)]
    st0 = des.predicted(list(chosen))
    r0 = des.realized(list(chosen))
    steps = [{"step": 0, "added": None, "panel": list(chosen), "pred": st0, "real": r0,
              "objective_tb": tb_score(st0, T)[0], "objective_tb_real": tb_score(r0, T)[0]}]
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
        print(f"  tb-greedy {len(chosen):2d} +{best:12s} pred {best_key[0]:.3f} real {tb_score(real, T)[0]:.3f}",
              flush=True)
    return steps


def greedy_cw(des, full):
    chosen, steps = [], []
    st0 = des.predicted([])
    steps.append({"step": 0, "added": None, "panel": [], "pred": st0, "real": des.realized([])})
    remaining = list(full)
    while remaining:
        best, best_key, best_stats = None, None, None
        for m in remaining:
            s = des.predicted(chosen + [m])
            o = s["_overall"]
            key = (o["leafres_correct"], o["depth"], -o["size"], -full.index(m))
            if best_key is None or key > best_key:
                best, best_key, best_stats = m, key, s
        chosen.append(best)
        remaining.remove(best)
        steps.append({"step": len(chosen), "added": best, "panel": list(chosen), "pred": best_stats,
                      "real": des.realized(list(chosen))})
        print(f"  cw-greedy {len(chosen):2d} +{best:12s} pred {best_key[0]:.3f}", flush=True)
    return steps


def heuristic_order(onto: Ontology, full, start=()):
    """Data-free: repeatedly add the marker that completes the most R(c) sets (c over the ontology nodes); ties: the
    marker appearing in the most not-yet-complete R(c) sets (isoforms count for CD45), then panel order."""
    chosen = [m for m in full if m in set(start)]
    remaining = [m for m in full if m not in set(chosen)]
    order = []

    def satisfies(m, req):
        return m in ISOFORM_SATISFIERS.get(req, {req})

    while remaining:
        adm0 = onto.admissible(chosen)
        best, best_key = None, None
        for m in remaining:
            gain = int(onto.admissible(chosen + [m]).sum() - adm0.sum())
            touch = sum(1 for c, n in enumerate(onto.names) if not adm0[c]
                        and any(satisfies(m, r) for r in onto.R[n]))
            key = (gain, touch, -full.index(m))
            if best_key is None or key > best_key:
                best, best_key = m, key
        chosen.append(best)
        remaining.remove(best)
        order.append({"added": best, "completes": best_key[0], "touches": best_key[1], "panel": list(chosen)})
    return order


# ------------------------------------------------------------------------------------------------------------------


def build_schurch(seed, onto, tt):
    from cohort import load_cohort_table, load_config, make_splits
    from run_cohort import fit_head
    cfg = load_config(HERE.parents[1] / "configs" / "schurch.json")
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

    des = Design3(onto, predict_fn, y[ci], patient[ci], y[ti], patient[ti], y[tr])
    bank = json.loads(Path(cfg["panel_bank"]).read_text())
    return des, markers, bank, {"seed": seed, "n_cal_patients": int(len(sp["cal"])),
                                "n_test_patients": int(len(sp["test"])), "n_train_labels": int(len(tr))}


def build_orion(onto, tt):
    bb = get_backbone("round0", tt)
    cal, test = load_split("crc_cal"), load_split("crc_test")
    ic, it = load_identity("crc_cal"), load_identity("crc_test")
    Xc = np.concatenate([bb.features(cal), bb.features(test)])
    yc = np.concatenate([cal["y"], test["y"]]).astype(np.int64)
    gc = np.concatenate([ic["sample"], it["sample"]])
    ex = np.load(RUNS / "crc_rebuild" / "crc_test_extra.npz", allow_pickle=True)
    Xt, yt, gt = ex["X_arc"].astype(np.float32), ex["y"].astype(np.int64), ex["sample"].astype(str)
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
    y_train = load_split("crc_train_full")["y"].astype(np.int64)
    des = Design3(onto, predict_fn, yc, gc, yt, gt, y_train)
    return des, list(bank["orion_panel"]), {"panels": bank["panels"]}, {
        "n_cal_patients": int(len(np.unique(gc))), "n_test_patients": int(len(np.unique(gt))),
        "n_train_labels": int(len(y_train))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True, choices=["schurch", "orion"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parts", default="bank,greedy_cw,tb_empty,tb_lineage,validate")
    args = ap.parse_args()
    t0 = time.time()
    parts = args.parts.split(",")
    tt = list(NODE17)          # the 17-node column order shared by every experiment
    onto = Ontology.from_repo(tt)
    if args.design == "schurch":
        des, full, bank, info = build_schurch(args.seed, onto, tt)
    else:
        des, full, bank, info = build_orion(onto, tt)
    tag = f"design_{args.design}" + (f"_seed{args.seed}" if args.design == "schurch" else "")
    T = types_T(des, full)
    base = {"design": args.design, "info": info, "alpha": ALPHA, "engine": "ADMIT-M1 (frozen map)",
            "full_panel": full, "types_T": T}
    OUT.mkdir(parents=True, exist_ok=True)
    if "bank" in parts:
        res = dict(base)
        pf = des.predicted(full)
        mv = {}
        for m in full:
            pm = des.predicted([x for x in full if x != m])
            mv[m] = {k: {s: (None if (pf[k].get(s) is None or pm.get(k, {}).get(s) is None) else pf[k][s] - pm[k][s])
                         for s in ("leafres", "leafres_correct", "depth", "size", "cov")}
                     for k in pf if not k.startswith("_type")}
        res["marker_value"] = {"pred_full": pf, "delta_full_minus_removed": mv}
        res["bank"] = {p["name"]: {"panel": p["markers"], "family": p.get("family"), "pred": des.predicted(p["markers"]),
                                   "real": des.realized(p["markers"])} for p in bank["panels"]}
        res["wall_clock_s"] = round(time.time() - t0, 1)
        write_json(OUT / f"{tag}_bank.json", res)
        print("bank done", res["wall_clock_s"], flush=True)
    if "greedy_cw" in parts:
        res = dict(base, objective="cell-weighted correct leaf-resolution", start="empty")
        res["greedy"] = greedy_cw(des, full)
        res["wall_clock_s"] = round(time.time() - t0, 1)
        write_json(OUT / f"{tag}_greedy_cw.json", res)
    for start in ("empty", "lineage"):
        if f"tb_{start}" not in parts:
            continue
        res = dict(base, objective="type-balanced mean over T of correct leaf-resolution", start=start)
        steps = greedy_tb(des, full, T, LINEAGE_START if start == "lineage" else ())
        res["greedy"] = steps
        res["budget"] = {}
        for k in (8, 12, 16, 24):
            hit = [st for st in steps if len(st["panel"]) == k]
            res["budget"][str(k)] = ({"panel": hit[0]["panel"], "pred_tb": hit[0]["objective_tb"],
                                      "real_tb": hit[0]["objective_tb_real"], "pred": hit[0]["pred"],
                                      "real": hit[0]["real"]} if hit else "not applicable")
        res["wall_clock_s"] = round(time.time() - t0, 1)
        write_json(OUT / f"{tag}_tb_{start}.json", res)
    if "validate" in parts:
        res = dict(base, ks=[k for k in KS if k < len(full)], n_random=N_RANDOM)
        panels = {}
        for start in ("empty", "lineage"):
            f = OUT / f"{tag}_tb_{start}.json"
            if f.exists():
                steps = json.loads(f.read_text())["greedy"]
                for k in res["ks"]:
                    hit = [s for s in steps if len(s["panel"]) == k]
                    if hit:
                        panels[f"greedy_{start}_k{k}"] = {"kind": f"greedy_{start}", "k": k, "panel": hit[0]["panel"]}
            ho = heuristic_order(onto, full, LINEAGE_START if start == "lineage" else ())
            res[f"heuristic_order_{start}"] = ho
            for k in res["ks"]:
                hit = [o for o in ho if len(o["panel"]) == k]
                if hit:
                    panels[f"heuristic_{start}_k{k}"] = {"kind": f"heuristic_{start}", "k": k, "panel": hit[0]["panel"]}
        for k in res["ks"]:
            rng = np.random.default_rng(1000 + k)
            for j in range(N_RANDOM):
                pick = sorted(rng.choice(len(full), k, replace=False).tolist())
                panels[f"random_k{k}_{j}"] = {"kind": "random", "k": k, "panel": [full[i] for i in pick]}
        for p in bank["panels"]:
            if p.get("family") == "random_retention":
                panels[f"bank_{p['name']}"] = {"kind": "bank_random_retention", "k": len(p["markers"]),
                                               "panel": p["markers"]}
        names = list(panels)
        pp = np.zeros((len(names), len(des.test_pats), onto.K, 2))
        for i, nm in enumerate(names):
            st, pp[i] = des.realized(panels[nm]["panel"], per_patient=True)
            panels[nm]["real_tb"] = tb_score(st, T)[0]
            panels[nm]["real"] = {t: st[t] for t in T if t in st}
        res["panels"] = panels
        res["wall_clock_s"] = round(time.time() - t0, 1)
        write_json(OUT / f"{tag}_validate.json", res)
        np.savez_compressed(OUT / f"{tag}_validate_pp.npz", pp=pp, panel_names=np.array(names),
                            patients=des.test_pats, nodes=np.array(onto.names))
    print("done", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
