#!/usr/bin/env python3
"""Panel-design summary: runs/round3/panel_design/ -> summary_design.json + .md

Agreement (predicted on calibration patients by leave-one-calibration-patient-out cross-fit vs realised on held-out
test patients), metric = correct leaf-resolution per (split, panel, type) over the bank and cell-weighted greedy
panels: Pearson r and MAE pooled over all points, mean of per-split values, per type (across
panels and splits within a type), and on points with realised resolution > 0. Type-balanced objective along the
type-balanced greedy paths: pred vs real.
Validation at matched marker count k: realised type-balanced correct leaf-resolution (types T = union of the
per-split T) of the greedy panel (each start), the mean of 20 random panels, the R(c)-completion heuristic, pooled
over splits per test patient; differences with 95% patient-cluster bootstrap intervals (B = 1000, one weight vector
over unique test patients, a patient recurring across splits is one cluster).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import RUNS, write_json  # noqa: E402

D = RUNS / "round3" / "panel_design"
B = 1000


def rmae(p, r):
    p, r = np.asarray(p, float), np.asarray(r, float)
    if len(p) < 3:
        return {"n": int(len(p)), "r": None, "mae": float(np.mean(np.abs(p - r))) if len(p) else None}
    return {"n": int(len(p)), "r": float(np.corrcoef(p, r)[0, 1]) if p.std() > 0 and r.std() > 0 else None,
            "mae": float(np.mean(np.abs(p - r)))}


def agreement(tags):
    pts = []                                   # (split, source, panel, type, pred, real)
    tb = []                                    # (split, start, step, pred_tb, real_tb)
    for s, tag in tags:
        f = D / f"{tag}_bank.json"
        if f.exists():
            d = json.loads(f.read_text())
            for pn, e in d["bank"].items():
                for t, pv in e["pred"].items():
                    if t.startswith("_") or t not in e["real"]:
                        continue
                    pts.append((s, "bank", pn, t, pv["leafres_correct"], e["real"][t]["leafres_correct"]))
        f = D / f"{tag}_greedy_cw.json"
        if f.exists():
            for st in json.loads(f.read_text())["greedy"]:
                for t, pv in st["pred"].items():
                    if t.startswith("_") or t not in st["real"]:
                        continue
                    pts.append((s, "greedy_cw", "+".join(st["panel"]) or "(empty)", t, pv["leafres_correct"],
                                st["real"][t]["leafres_correct"]))
        for start in ("empty", "lineage"):
            f = D / f"{tag}_tb_{start}.json"
            if f.exists():
                for st in json.loads(f.read_text())["greedy"]:
                    tb.append((s, start, st["step"], st["objective_tb"], st["objective_tb_real"]))
    out = {"n_points": len(pts)}
    if pts:
        P = np.array([p[4] for p in pts]); R = np.array([p[5] for p in pts])
        out["pooled"] = rmae(P, R)
        splits = sorted({p[0] for p in pts})
        per = {s: rmae([p[4] for p in pts if p[0] == s], [p[5] for p in pts if p[0] == s]) for s in splits}
        out["per_split"] = per
        rs = [v["r"] for v in per.values() if v["r"] is not None]
        out["mean_of_per_split"] = {"r": float(np.mean(rs)) if rs else None,
                                    "mae": float(np.mean([v["mae"] for v in per.values()]))}
        types = sorted({p[3] for p in pts})
        out["per_type"] = {t: rmae([p[4] for p in pts if p[3] == t], [p[5] for p in pts if p[3] == t]) for t in types}
        nz = R > 0
        out["realised_gt0"] = rmae(P[nz], R[nz])
        out["frac_points_realised_zero"] = float((~nz).mean())
    if tb:
        out["tb_objective"] = {}
        for start in ("empty", "lineage"):
            rows = [x for x in tb if x[1] == start]
            if not rows:
                continue
            per = {s: rmae([x[3] for x in rows if x[0] == s], [x[4] for x in rows if x[0] == s])
                   for s in sorted({x[0] for x in rows})}
            rs = [v["r"] for v in per.values() if v["r"] is not None]
            out["tb_objective"][start] = {"pooled": rmae([x[3] for x in rows], [x[4] for x in rows]),
                                          "mean_of_per_split": {"r": float(np.mean(rs)) if rs else None,
                                                                "mae": float(np.mean([v["mae"] for v in per.values()]))}}
    return out


def validation(tags):
    vals, pps = [], []
    for s, tag in tags:
        f = D / f"{tag}_validate.json"
        if f.exists():
            vals.append(json.loads(f.read_text()))
            pps.append(np.load(D / f"{tag}_validate_pp.npz", allow_pickle=True))
    if not vals:
        return None
    nodes = list(pps[0]["nodes"])
    T = sorted({t for v in vals for t in v["types_T"]})
    Ti = [nodes.index(t) for t in T]
    pats = np.array(sorted({p for z in pps for p in z["patients"].tolist()}))
    W = np.random.default_rng(0).multinomial(len(pats), np.full(len(pats), 1.0 / len(pats)), size=B).astype(float)

    def pooled(kind_key):
        """sum per-patient counts over splits for the panels selected by kind_key(panel entry) -> (n_pat, K, 2)"""
        acc = np.zeros((len(pats), len(nodes), 2))
        found = 0
        for v, z in zip(vals, pps):
            names = list(z["panel_names"])
            sl = np.searchsorted(pats, z["patients"])
            for nm, e in v["panels"].items():
                if kind_key(nm, e):
                    acc[sl] += z["pp"][names.index(nm)]
                    found += 1
        return acc if found else None

    def tb(acc):
        H, N = acc[:, Ti, 0], acc[:, Ti, 1]
        point = float(np.nanmean(np.where(N.sum(0) > 0, H.sum(0) / np.maximum(N.sum(0), 1), np.nan)))
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = np.nanmean((W @ H) / (W @ N), axis=1)
            per_pat = np.nanmean(H / N, axis=1)          # each test patient's own type-balanced resolution
        return point, boot, per_pat

    rng_p = np.random.default_rng(1)
    ks = vals[0]["ks"]
    out = {"types_T_union": T, "n_test_patients_pooled": int(len(pats)), "n_splits": len(vals), "k": {},
           "ci_note": ("ci_patients: percentile bootstrap over test patients only (random-panel set fixed); "
                       "ci_patients_and_panels: each replicate also resamples the 20 random panels with replacement "
                       "(the primary interval for greedy - random); t95_patients: mean +- t(0.975, n-1) sd/sqrt(n) of "
                       "per-patient differences (unweighted), a check for designs with few patients")}
    for k in ks:
        rnd = []
        for j in range(vals[0]["n_random"]):
            a = pooled(lambda nm, e, j=j: nm == f"random_k{k}_{j}")
            if a is not None:
                rnd.append(tb(a))
        if not rnd:
            continue
        r_point = float(np.mean([x[0] for x in rnd]))
        RB = np.vstack([x[1] for x in rnd])                  # (n_random, B)
        r_boot = RB.mean(axis=0)
        pick = rng_p.integers(0, len(rnd), size=(B, len(rnd)))
        r_boot2 = np.take_along_axis(RB.T, pick, axis=1).mean(axis=1)
        r_pat = np.nanmean(np.vstack([x[2] for x in rnd]), axis=0)
        row = {"random_mean": r_point, "random_min": float(min(x[0] for x in rnd)),
               "random_max": float(max(x[0] for x in rnd)), "n_random": len(rnd)}
        for start in ("empty", "lineage"):
            g = pooled(lambda nm, e: nm == f"greedy_{start}_k{k}")
            h = pooled(lambda nm, e: nm == f"heuristic_{start}_k{k}")
            if g is None:
                continue
            gp_, gb, gpat = tb(g)
            d_pat = gpat - r_pat
            d_pat = d_pat[np.isfinite(d_pat)]
            from scipy.stats import t as tdist
            tq = float(tdist.ppf(0.975, max(len(d_pat) - 1, 1)))
            row[f"greedy_{start}"] = gp_
            row[f"greedy_{start}_minus_random"] = gp_ - r_point
            row[f"greedy_{start}_minus_random_ci_patients"] = [float(np.nanpercentile(gb - r_boot, q)) for q in (2.5, 97.5)]
            row[f"greedy_{start}_minus_random_ci"] = [float(np.nanpercentile(gb - r_boot2, q)) for q in (2.5, 97.5)]
            row[f"greedy_{start}_minus_random_per_patient"] = {
                "n": int(len(d_pat)), "min": float(d_pat.min()), "median": float(np.median(d_pat)), "max": float(d_pat.max()),
                "t95": [float(d_pat.mean() - tq * d_pat.std(ddof=1) / np.sqrt(len(d_pat))),
                        float(d_pat.mean() + tq * d_pat.std(ddof=1) / np.sqrt(len(d_pat)))]}
            row[f"greedy_{start}_rank_among_random"] = int(sum(x[0] >= gp_ for x in rnd))
            if h is not None:
                hp, hb, _ = tb(h)
                row[f"heuristic_{start}"] = hp
                row[f"greedy_{start}_minus_heuristic"] = gp_ - hp
                row[f"greedy_{start}_minus_heuristic_ci"] = [float(np.nanpercentile(gb - hb, q)) for q in (2.5, 97.5)]
        out["k"][str(k)] = row
    bank = {}
    for v in vals:
        for nm, e in v["panels"].items():
            if e["kind"] == "bank_random_retention":
                bank.setdefault(nm, {"k": e["k"], "real_tb_by_split": []})["real_tb_by_split"].append(e["real_tb"])
    out["bank_random_retention"] = {nm: {"k": b["k"], "real_tb_mean_over_splits": float(np.mean(b["real_tb_by_split"]))}
                                    for nm, b in bank.items()}
    return out


def main():
    res = {}
    designs = {"schurch": [(s, f"design_schurch_seed{s}") for s in range(5)], "orion": [(0, "design_orion")]}
    for dname, tags in designs.items():
        res[dname] = {"agreement": agreement(tags), "validation": validation(tags)}
    write_json(D / "summary_design.json", res)
    md = ["<!-- generated by experiments/round3/summarize_design3.py -->\n# Panel design (ADMIT-M1)\n\n"]
    f3 = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    for dname, r in res.items():
        a = r["agreement"]
        md.append(f"## {dname}\n\n")
        if a.get("pooled"):
            md.append(f"Agreement, correct leaf-resolution per (split, panel, type), bank + cell-weighted greedy panels "
                      f"({a['n_points']} points; {a['frac_points_realised_zero']:.2f} with realised 0): pooled r "
                      f"{f3(a['pooled']['r'])}, MAE {f3(a['pooled']['mae'])}; mean of per-split r "
                      f"{f3(a['mean_of_per_split']['r'])}, MAE {f3(a['mean_of_per_split']['mae'])}; realised > 0: r "
                      f"{f3(a['realised_gt0']['r'])}, MAE {f3(a['realised_gt0']['mae'])}.\n\n")
            md.append("| type | n | r | MAE |\n|---|---|---|---|\n" + "".join(
                f"| {t} | {v['n']} | {f3(v['r'])} | {f3(v['mae'])} |\n" for t, v in a["per_type"].items()) + "\n")
        for start, v in a.get("tb_objective", {}).items():
            md.append(f"Type-balanced objective along the {start}-start greedy path: pooled r {f3(v['pooled']['r'])}, "
                      f"MAE {f3(v['pooled']['mae'])}; mean of per-split r {f3(v['mean_of_per_split']['r'])}.\n\n")
        val = r["validation"]
        if val:
            md.append(f"Matched-marker-count validation (realised type-balanced correct leaf-resolution over T = "
                      f"{', '.join(val['types_T_union'])}; {val['n_test_patients_pooled']} test patients pooled over "
                      f"{val['n_splits']} split(s)):\n\n| k | random mean [min, max] | greedy empty | - random [CI] | "
                      f"heuristic empty | greedy - heur. [CI] | greedy lineage | - random [CI] | heuristic lineage | "
                      f"greedy - heur. [CI] |\n|---|---|---|---|---|---|---|---|---|---|\n")
            for k, x in val["k"].items():
                cells = [k, f"{f3(x['random_mean'])} [{f3(x['random_min'])}, {f3(x['random_max'])}]"]
                for st in ("empty", "lineage"):
                    if f"greedy_{st}" in x:
                        ci = x[f"greedy_{st}_minus_random_ci"]
                        cp = x[f"greedy_{st}_minus_random_ci_patients"]
                        pp_ = x[f"greedy_{st}_minus_random_per_patient"]
                        hci = x.get(f"greedy_{st}_minus_heuristic_ci", [None, None])
                        cells += [f3(x[f"greedy_{st}"]),
                                  f"{f3(x[f'greedy_{st}_minus_random'])} [{f3(ci[0])}, {f3(ci[1])}] (patients only "
                                  f"[{f3(cp[0])}, {f3(cp[1])}]; per-patient {f3(pp_['min'])}..{f3(pp_['max'])}, t95 "
                                  f"[{f3(pp_['t95'][0])}, {f3(pp_['t95'][1])}])",
                                  f3(x.get(f"heuristic_{st}")),
                                  f"{f3(x.get(f'greedy_{st}_minus_heuristic'))} [{f3(hci[0])}, {f3(hci[1])}]"]
                    else:
                        cells += ["-"] * 4
                md.append("| " + " | ".join(cells) + " |\n")
            md.append("\n" + val["ci_note"] + "\n\n")
    (D / "summary_design.md").write_text("".join(md))
    print("".join(md))


if __name__ == "__main__":
    main()
