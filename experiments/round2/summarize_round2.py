#!/usr/bin/env python3
"""Reduce round-2 accumulators to figure-ready JSON + TSV + a generated Markdown table file.

Inputs ($ADMIT_RUNS/round2): orion_<protocol>.npz (+_meta.json), cohort_<name>_<head>_s*.npz (+_meta.json),
panel_design_*.json, external/*.json. Outputs: summary_round2.json, tables/*.tsv, report_tables_round2.md.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RUNS, write_json  # noqa: E402

R2 = RUNS / "round2"
B = 1000
MIN_N = 100
ARMS_E1 = ["admit", "admit_m1", "admit_m1_n50", "lac_node_proj", "lac_filt", "aps_filt", "raps_filt",
           "patch_src_backstop", "patch_src", "admit_crc", "admit_m1_crc", "lac_crc_filt"]
SWEEP = ["admit", "admit_m1", "lac_node_proj", "lac_filt", "aps_filt"]
LABEL = {"admit": "ADMIT", "admit_m1": "ADMIT+M1 (n_min rule)", "admit_m1_n50": "ADMIT+M1 (n_min 50)",
         "lac_node_proj": "marginal proj. LAC", "lac_filt": "LAC+filter", "aps_filt": "APS+filter",
         "raps_filt": "RAPS+filter", "patch_src_backstop": "old PATCH bs on", "patch_src": "old PATCH bs off",
         "admit_crc": "ADMIT eq-patient CRC", "admit_m1_crc": "ADMIT+M1 eq-patient CRC",
         "lac_crc_filt": "LAC eq-patient CRC+filter"}


def load_run(kind, name):
    if kind == "orion":
        z = np.load(R2 / f"orion_{name}.npz", allow_pickle=True)
        meta = json.loads((R2 / f"orion_{name}_meta.json").read_text())
        return {k: z[k] for k in z.files}, meta, None
    files = sorted(glob.glob(str(R2 / f"cohort_{name}_s*.npz")))
    files = [f for f in files if "debug" not in f]
    if not files:
        return None, None, None
    acc, eps, metas = None, [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        acc = d["acc"].astype(np.float64) if acc is None else acc + d["acc"]
        eps.append({k: d[k] for k in ("ep_cov", "pooled_cov", "ep_size", "seeds")})
        metas.append(json.loads(Path(f.replace(".npz", "_meta.json")).read_text()))
        base = d
    base["acc"] = acc
    split = {k: np.concatenate([e[k] for e in eps]) for k in ("ep_cov", "pooled_cov", "ep_size", "seeds")}
    meta = metas[0]
    meta["splits"] = [s for m in metas for s in m["splits"]]
    return base, meta, split


def boot_weights(n_pat, seed=0):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_pat, np.full(n_pat, 1.0 / n_pat), size=B).astype(float)


def cell_summary(G, nodes, W):
    """G: (n_pat, K, NSTAT) for one panel/arm/alpha."""
    gh, gn = G[:, :, 0], G[:, :, 1]
    out = {}
    pats = gn.sum(1) > 0
    tot_n = gn.sum()
    out["n_ident"] = float(tot_n)
    out["n_unident"] = float(G[:, :, 4].sum() - G[:, :, 3].sum())
    if tot_n == 0:
        return None
    out["cov"] = float(gh.sum() / tot_n)
    with np.errstate(invalid="ignore", divide="ignore"):
        cb = (W @ gh.sum(1)) / (W @ gn.sum(1))
    out["cov_ci"] = [float(np.nanpercentile(cb, 2.5)), float(np.nanpercentile(cb, 97.5))]
    pc = gh.sum(1)[pats] / gn.sum(1)[pats]
    out["ep_cov"] = float(pc.mean())
    out["min_patient_cov"] = float(pc.min())
    strata = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        sb = (W @ gh) / (W @ gn)
    for k in range(len(nodes)):
        n = gn[:, k].sum()
        if n == 0:
            continue
        strata[nodes[k]] = {"n": int(n), "n_patients": int((gn[:, k] > 0).sum()), "cov": float(gh[:, k].sum() / n),
                            "ci": [float(np.nanpercentile(sb[:, k], 2.5)), float(np.nanpercentile(sb[:, k], 97.5))]}
    out["strata"] = strata
    big = {k: v for k, v in strata.items() if v["n"] >= MIN_N}
    if big:
        wk = min(big, key=lambda k: big[k]["cov"])
        out["worst_type"], out["worst_cov"] = wk, big[wk]["cov"]
        out["sig_below"] = sorted([k for k, v in big.items() if v["ci"][1] < 0.90])
        out["n_types"] = len(big)
    L = G[:, :, 4].sum(0)
    lab = {}
    for k in range(len(nodes)):
        if L[k] == 0:
            continue
        ni = G[:, k, 3].sum()
        lab[nodes[k]] = {"n": int(L[k]), "cov": (float(G[:, k, 2].sum() / ni) if ni else None),
                         "size": float(G[:, k, 5].sum() / L[k]), "depth": float(G[:, k, 6].sum() / L[k]),
                         "leafres": float(G[:, k, 7].sum() / L[k]), "leafres_correct": float(G[:, k, 11].sum() / L[k]),
                         "single": float(G[:, k, 8].sum() / L[k]), "abst": float(G[:, k, 9].sum() / L[k])}
    out["labels"] = lab
    N = L.sum()
    out["size"] = float(G[:, :, 5].sum() / N)
    out["depth"] = float(G[:, :, 6].sum() / N)
    out["leafres"] = float(G[:, :, 7].sum() / N)
    out["leafres_correct"] = float(G[:, :, 11].sum() / N)
    out["single"] = float(G[:, :, 8].sum() / N)
    out["abst"] = float(G[:, :, 9].sum() / N)
    out["inadm_mass"] = float(G[:, :, 10].sum() / max(G[:, :, 5].sum(), 1))
    return out


def summarize_run(d, meta, split):
    acc, panels, arms, nodes = d["acc"], list(d["panels"]), list(d["arms"]), list(d["nodes"])
    alphas = [float(a) for a in d["alphas"]]
    W = boot_weights(acc.shape[3])
    fam = meta["families"]
    res = {"panels": {}, "alphas": alphas, "patients": list(d["patients"])}
    for pi, pn in enumerate(panels):
        res["panels"][pn] = {"family": (f"random{pn.split('_')[0][4:]}" if fam.get(pn) == "random_retention"
                                        else fam.get(pn)), "arms": {}}
        for ai, a in enumerate(arms):
            res["panels"][pn]["arms"][a] = {}
            for li, al in enumerate(alphas):
                G = acc[pi, ai, li]
                if G[:, :, 4].sum() == 0:
                    continue
                cs = cell_summary(G, nodes, W)
                if cs is not None and li != 1:   # keep detail only at alpha 0.10
                    cs = {k: v for k, v in cs.items() if k not in ("labels",)}
                res["panels"][pn]["arms"][a][f"{al:.2f}"] = cs
    for alias, target in meta.get("panel_alias", {}).items():
        if target in res["panels"]:
            res["panels"][alias] = {**res["panels"][target], "family": fam.get(alias), "alias_of": target}
    # family / all-deletion aggregates at alpha 0.10 (+ sweep)
    fams = {}
    for pn, e in res["panels"].items():
        fams.setdefault(e["family"], []).append(pn)
    uniq_del = [pn for pn in panels if res["panels"][pn]["family"] != "full"]
    fams["all_deletions"] = uniq_del
    agg = {}
    for f, pns in fams.items():
        agg[f] = {"n_panels": len(pns), "arms": {}}
        for a in arms:
            agg[f]["arms"][a] = {}
            for al in alphas:
                rows = [res["panels"][p]["arms"][a].get(f"{al:.2f}") for p in pns]
                rows = [r for r in rows if r]
                if not rows:
                    continue
                m = lambda k: float(np.mean([r[k] for r in rows if r.get(k) is not None])) if any(r.get(k) is not None for r in rows) else None  # noqa: E731
                fail = {}
                for r in rows:
                    for t in r.get("sig_below", []):
                        fail[t] = fail.get(t, 0) + 1
                agg[f]["arms"][a][f"{al:.2f}"] = {
                    "cov": m("cov"), "ep_cov": m("ep_cov"), "worst_cov": m("worst_cov"),
                    "min_worst_cov": float(min(r["worst_cov"] for r in rows if "worst_cov" in r)) if any("worst_cov" in r for r in rows) else None,
                    "n_sig_below": int(sum(len(r.get("sig_below", [])) for r in rows)),
                    "n_types": int(sum(r.get("n_types", 0) for r in rows)), "failing_type_counts": fail,
                    "size": m("size"), "depth": m("depth"), "leafres": m("leafres"),
                    "leafres_correct": m("leafres_correct"), "abst": m("abst"), "inadm_mass": m("inadm_mass")}
    res["families"] = agg
    if split is not None:      # E4: per-split equal-patient coverage distribution
        ep = split["ep_cov"]   # split, panel, arm, alpha
        dist = {}
        for pi, pn in enumerate(panels):
            dist[pn] = {}
            for ai, a in enumerate(arms):
                v = ep[:, pi, ai, 1]
                v = v[np.isfinite(v)]
                if len(v) == 0:
                    continue
                dist[pn][a] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                               "min": float(v.min()), "p5": float(np.percentile(v, 5)),
                               "frac_below_0.90": float((v < 0.90).mean()), "n_splits": int(len(v)),
                               "size_mean": float(np.nanmean(split["ep_size"][:, pi, ai, 1]))}
        res["e4_split_distribution"] = dist
        alld = {}
        for ai, a in enumerate(arms):
            vals = ep[:, [panels.index(p) for p in uniq_del], ai, 1]
            alld[a] = {"mean_over_splits_and_panels": float(np.nanmean(vals)),
                       "frac_split_panel_below_0.90": float(np.nanmean(vals < 0.90)) if np.isfinite(vals).any() else None}
        res["e4_all_deletions"] = alld
        res["splits"] = meta["splits"]
    return res


def design_summary():
    out = {}
    for f in sorted(list(R2.glob("panel_design_*.json")) + list(R2.glob("greedy_lineage_*.json"))
                    + list(R2.glob("greedy_tb_*.json"))):
        d = json.loads(f.read_text())
        key = f.stem.replace("panel_design_", "")
        d.setdefault("bank", {})
        d.setdefault("marker_value", {"delta_full_minus_removed": {}})
        pairs = {m: ([], []) for m in ("leafres_correct", "leafres", "depth", "size")}
        rows = []
        for src, entries in (("bank", list(d["bank"].values())), ("greedy", d["greedy"])):
            for e in entries:
                for lab, pv in e["pred"].items():
                    if lab.startswith("_"):
                        continue
                    rv = e["real"].get(lab)
                    if rv is None:
                        continue
                    for m in pairs:
                        pairs[m][0].append(pv[m]); pairs[m][1].append(rv[m])
                rows.append({"source": src, "panel": e.get("panel"),
                             "pred": e["pred"]["_overall"], "real": e["real"]["_overall"]})
        val = {}
        for m, (p, r) in pairs.items():
            p, r = np.array(p), np.array(r)
            val[m] = {"n_pairs": int(len(p)), "mae": float(np.mean(np.abs(p - r))),
                      "pearson_r": float(np.corrcoef(p, r)[0, 1]) if p.std() > 0 and r.std() > 0 else None}
        ov = {}
        for m in ("leafres_correct", "depth", "size"):
            p = np.array([x["pred"][m] for x in rows]); r = np.array([x["real"][m] for x in rows])
            ov[m] = {"mae": float(np.mean(np.abs(p - r))),
                     "pearson_r": float(np.corrcoef(p, r)[0, 1]) if p.std() > 0 and r.std() > 0 else None}
        curve = [{"step": g["step"], "added": g["added"], "n_markers": len(g["panel"]),
                  "pred_leafres_correct": g["pred"]["_overall"]["leafres_correct"],
                  "real_leafres_correct": g["real"]["_overall"]["leafres_correct"],
                  "pred_type_balanced": g["pred"]["_type_balanced"]["leafres_correct"],
                  "real_type_balanced": g["real"]["_type_balanced"]["leafres_correct"],
                  "pred_depth": g["pred"]["_overall"]["depth"], "real_depth": g["real"]["_overall"]["depth"],
                  "pred_size": g["pred"]["_overall"]["size"], "real_size": g["real"]["_overall"]["size"]}
                 for g in d["greedy"]]
        mv = d["marker_value"]["delta_full_minus_removed"] or {}
        mv_overall = {m: {s: v["_overall"].get(s) for s in ("leafres_correct", "leafres", "depth", "size")}
                      for m, v in mv.items()}
        mv_types = {m: {lab: v[lab]["leafres_correct"] for lab in v if not lab.startswith("_")} for m, v in mv.items()}
        if "types_T" in d:      # type-balanced objective over T
            T = d["types_T"]
            tbm = lambda st: float(np.mean([st[t]["leafres_correct"] for t in T if t in st])) if T else None  # noqa: E731
            for c, g in zip(curve, d["greedy"]):
                c["pred_tb_T"] = tbm(g["pred"]); c["real_tb_T"] = tbm(g["real"])
            pT = np.array([c["pred_tb_T"] for c in curve]); rT = np.array([c["real_tb_T"] for c in curve])
            ov["tb_objective_T"] = {"mae": float(np.mean(np.abs(pT - rT))),
                                    "pearson_r": float(np.corrcoef(pT, rT)[0, 1]) if pT.std() > 0 and rT.std() > 0 else None}
            budget = {k: (v if isinstance(v, str) else {"panel": v["panel"], "pred_tb": v["pred_tb"], "real_tb": v["real_tb"],
                                                        "pred_size": v["pred"]["_overall"]["size"],
                                                        "real_size": v["real"]["_overall"]["size"],
                                                        "pred_depth": v["pred"]["_overall"]["depth"],
                                                        "real_depth": v["real"]["_overall"]["depth"]})
                      for k, v in d["budget"].items()}
        out[key] = {"validation_per_type": val, "validation_overall": ov, "greedy_curve": curve,
                    **({"types_T": d["types_T"], "budget": budget} if "types_T" in d else {}),
                    "marker_value_overall": mv_overall, "marker_value_by_type_leafres_correct": mv_types,
                    "info": d["info"], "wall_clock_s": d.get("wall_clock_s")}
    # mean over the 5 Schurch seeds (curves aligned by step; marker values by marker)
    for kind in ("schurch_seed", "greedy_lineage_schurch_seed", "greedy_tb_empty_schurch_seed",
                 "greedy_tb_lineage_schurch_seed"):
        keys = [k for k in out if k.startswith(kind)]
        if not keys:
            continue
        agg = {"seeds": keys}
        steps = min(len(out[k]["greedy_curve"]) for k in keys)
        agg["greedy_curve_mean"] = [{m: (float(np.mean([out[k]["greedy_curve"][i][m] for k in keys]))
                                         if m not in ("added", "step") else None)
                                     for m in out[keys[0]]["greedy_curve"][0]} for i in range(steps)]
        for i, row in enumerate(agg["greedy_curve_mean"]):
            row["step"] = i
            row["added_by_seed"] = [out[k]["greedy_curve"][i]["added"] for k in keys]
        if out[keys[0]]["marker_value_overall"]:
            mk = out[keys[0]]["marker_value_overall"].keys()
            agg["marker_value_mean"] = {m: {s2: float(np.mean([out[k]["marker_value_overall"][m][s2] for k in keys]))
                                            for s2 in ("leafres_correct", "depth", "size")} for m in mk}
            types = sorted({t for k in keys for m in mk for t in out[k]["marker_value_by_type_leafres_correct"][m]})
            agg["marker_value_by_type_mean"] = {m: {t: float(np.mean([out[k]["marker_value_by_type_leafres_correct"][m].get(t, np.nan) or 0 for k in keys]))
                                                    for t in types} for m in mk}
        agg["validation_per_type_mean"] = {m: {"mae": float(np.mean([out[k]["validation_per_type"][m]["mae"] for k in keys])),
                                               "r": float(np.mean([out[k]["validation_per_type"][m]["pearson_r"] for k in keys
                                                                    if out[k]["validation_per_type"][m]["pearson_r"] is not None]))}
                                           for m in out[keys[0]]["validation_per_type"]}
        if "budget" in out[keys[0]]:
            agg["budget_by_seed"] = {k: out[k]["budget"] for k in keys}
            agg["budget_mean"] = {}
            for b in ("8", "12", "16", "24"):
                vals = [out[k]["budget"][b] for k in keys if isinstance(out[k]["budget"].get(b), dict)]
                if vals:
                    agg["budget_mean"][b] = {m: float(np.mean([v[m] for v in vals]))
                                             for m in ("pred_tb", "real_tb", "pred_size", "real_size", "pred_depth", "real_depth")}
                    from collections import Counter
                    agg["budget_mean"][b]["marker_frequency"] = dict(Counter(m for v in vals for m in v["panel"]).most_common())
            agg["types_T_by_seed"] = {k: out[k]["types_T"] for k in keys}
            agg["tb_validation_mean"] = {"mae": float(np.mean([out[k]["validation_overall"]["tb_objective_T"]["mae"] for k in keys])),
                                         "r": float(np.nanmean([out[k]["validation_overall"]["tb_objective_T"]["pearson_r"] or np.nan for k in keys]))}
        out[kind[: -len("_seed")] + "_mean"] = agg
    return out


def external_summary():
    out = {}
    for f in sorted((R2 / "external").glob("external_decomposition_*.json")):
        d = json.loads(f.read_text())
        rows = {}
        for k, r in d["cohorts"].items():
            iv5 = r["iv_target_recalibration"]["slice500"]["summary"]
            iv50 = r["iv_target_recalibration"]["split50"]["summary"]
            row = {"flag": r.get("flag"), "placeholder": r["xarc_placeholder"],
                   "i_old_bs_on": r["i_source_patch"]["backstop_on"]["headline"],
                   "ii_admit": r["ii_panel_matched"]["backstop_off"]["headline"],
                   "iii_admit": r["iii_orion_test_masked_to_Pk"]["panel_matched"]["headline"],
                   "iv500_admit": iv5["target_recal"], "iv50_admit": iv50["target_recal"]}
            for nm in ("admit_m1", "admit_m1_n50"):
                row[f"ii_{nm}"] = r["ii_admit_m1"][nm]["headline"]
                row[f"iii_{nm}"] = r["iii_orion_test_masked_to_Pk"][nm]["headline"]
                row[f"iv500_{nm}"] = iv5[f"target_recal_{nm}"]
                row[f"iv50_{nm}"] = iv50[f"target_recal_{nm}"]
            rows[k] = row
        clean = [k for k, v in rows.items() if not v["placeholder"] and k not in ("danenberg", "risom")]
        mean = {}
        for col in rows[clean[0]]:
            if col in ("flag", "placeholder"):
                continue
            val = lambda v: v["mean"] if isinstance(v, dict) else v  # noqa: E731
            vals = [val(rows[k][col]["exact_cov"]) for k in clean]
            sizes = [val(rows[k][col]["size"]) for k in clean]
            mean[col] = {"exact_cov": float(np.mean(vals)), "size": float(np.mean(sizes))}
        out[f.stem.replace("external_decomposition_", "")] = {"rows": rows, "clean_cohorts": clean, "mean": mean}
    return out


def target_curve_summary():
    out = {}
    for f in sorted((R2 / "target_curve").glob("target_curve_*.json")):
        d = json.loads(f.read_text())
        clean = [k for k, r in d["cohorts"].items() if not r["placeholder"] and k not in ("danenberg", "risom")]
        curve = {}
        for n in [str(x) for x in d["ns"]]:
            curve[n] = {}
            for a in d["arms"]:
                rows = [d["cohorts"][k]["curve"][n].get(a) for k in clean]
                rows = [r for r in rows if r]
                if not rows:
                    continue
                curve[n][a] = {m: float(np.mean([r[m]["mean"] for r in rows if m in r])) for m in
                               ("cov", "worst", "size", "leafres_correct", "n_sig") if any(m in r for r in rows)}
        per_cohort = {k: {n: {a: {m: v["mean"] for m, v in r.items() if isinstance(v, dict) and "mean" in v}
                              for a, r in d["cohorts"][k]["curve"][n].items()} for n in d["cohorts"][k]["curve"]}
                      for k in d["cohorts"]}
        out[f.stem.replace("target_curve_", "")] = {"clean_cohorts": clean, "mean_curve": curve,
                                                     "per_cohort": per_cohort,
                                                     "flags": {k: r["flag"] for k, r in d["cohorts"].items()},
                                                     "placeholder": [k for k, r in d["cohorts"].items() if r["placeholder"]]}
    return out


def f3(x):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def f2(x):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.2f}"


def table(h, rows):
    return "| " + " | ".join(h) + " |\n|" + "|".join(["---"] * len(h)) + "|\n" + "".join(
        "| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def render(S):
    md = ["<!-- generated by experiments/round2/summarize_round2.py from round-2 JSON/NPZ; do not edit -->\n"]
    tsv = {}
    for run, R in S["runs"].items():
        md.append(f"## E1 / E2: {run}\n\n")
        ref = "full" if "full" in R["panels"] else list(R["panels"])[0]
        md.append(f"Reference panel/block for the detailed tables: `{ref}`.\n\n")
        full = R["panels"][ref]["arms"]
        rows, trow = [], []
        for a in [a for a in ARMS_E1 if a in full and full[a].get("0.10")]:
            x = full[a]["0.10"]
            fa = R["families"].get("all_deletions", {"arms": {}})["arms"].get(a, {}).get("0.10", {})
            top = sorted(fa.get("failing_type_counts", {}).items(), key=lambda kv: -kv[1])[:5]
            rows.append([LABEL[a], f"{f3(x['cov'])} [{f3(x['cov_ci'][0])}, {f3(x['cov_ci'][1])}]", f3(x["ep_cov"]),
                         f"{x.get('worst_type', '–')} {f3(x.get('worst_cov'))}",
                         f"{len(x.get('sig_below', []))}/{x.get('n_types', 0)} " + (",".join(x.get("sig_below", [])) or ""),
                         f2(x["size"]), f3(x["leafres_correct"]), f2(x["depth"]), f3(x["abst"]),
                         f3(fa.get("cov")), f3(fa.get("worst_cov")), f"{fa.get('n_sig_below')}/{fa.get('n_types')}",
                         "; ".join(f"{t}:{c}" for t, c in top), f2(fa.get("size")), f3(fa.get("leafres_correct"))])
        md.append("Full panel (alpha 0.10): pooled exact coverage of the arm's target [patient-cluster bootstrap], "
                  "equal-patient coverage, worst type (strata >= 100 cells), types significantly below 0.90 / types, "
                  "|C|, correctly leaf-resolved, finest depth, abstention. Then the mean over all unique deletion panels: "
                  "coverage, worst-type coverage, total significant failures / strata, most frequent failing types, "
                  "|C|, correct leaf-resolved.\n\n")
        h = ["arm", "cov [CI]", "EP cov", "worst type", "sig<0.9", "|C|", "leaf-res (correct)", "depth", "abstain",
             "del: cov", "del: worst", "del: sig<0.9", "del: failing types", "del: |C|", "del: leaf-res"]
        md.append(table(h, rows) + "\n")
        tsv[f"E1_{run}"] = (h, rows)
        # per-stratum detail, full panel
        arms5 = [a for a in ["admit", "admit_m1", "admit_m1_n50", "lac_node_proj", "lac_filt", "aps_filt",
                             "patch_src_backstop"] if a in full and full[a].get("0.10")]
        strata = sorted({s for a in arms5 for s in full[a]["0.10"]["strata"]})
        rows = []
        for s in strata:
            r = [s]
            for a in arms5:
                v = full[a]["0.10"]["strata"].get(s)
                r.append("–" if v is None else f"{f3(v['cov'])} [{f3(v['ci'][0])},{f3(v['ci'][1])}] n={v['n']}")
            rows.append(r)
        md.append("Per projected stratum (target node), full panel, alpha 0.10: coverage [patient-cluster CI] n.\n\n")
        md.append(table(["stratum"] + [LABEL[a] for a in arms5], rows) + "\n")
        tsv[f"E1_strata_full_{run}"] = (["stratum"] + [LABEL[a] for a in arms5], rows)
        # E2 resolution profile per cell type, full panel
        rows = []
        labs = sorted({l for a in arms5 for l in full[a]["0.10"]["labels"]})
        for l in labs:
            r = [l, full[arms5[0]]["0.10"]["labels"].get(l, {}).get("n", "–")]
            for a in arms5:
                v = full[a]["0.10"]["labels"].get(l)
                r.append("–" if v is None else f"{f2(v['depth'])} / {f3(v['leafres_correct'])} / {f2(v['size'])}")
            rows.append(r)
        md.append("E2 resolution per cell type (original label), full panel, alpha 0.10: finest depth / correctly "
                  "leaf-resolved / |C|.\n\n")
        md.append(table(["cell type", "n"] + [LABEL[a] for a in arms5], rows) + "\n")
        tsv[f"E2_types_full_{run}"] = (["cell type", "n"] + [LABEL[a] for a in arms5], rows)
        # E2 per panel family
        rows = []
        for fam, fa in R["families"].items():
            r = [fam, fa["n_panels"]]
            for a in arms5:
                v = fa["arms"].get(a, {}).get("0.10")
                r.append("–" if not v else f"{f3(v['cov'])} / {f2(v['depth'])} / {f3(v['leafres_correct'])} / {f2(v['size'])}")
            rows.append(r)
        md.append("E2 per panel family, alpha 0.10: coverage / finest depth / correct leaf-resolved / |C| (means over panels).\n\n")
        md.append(table(["family", "panels"] + [LABEL[a] for a in arms5], rows) + "\n")
        # alpha curves
        rows = []
        for a in SWEEP:
            for al in ("0.05", "0.10", "0.15", "0.20"):
                x = full.get(a, {}).get(al)
                fa = R["families"].get("all_deletions", {"arms": {}})["arms"].get(a, {}).get(al)
                if not x:
                    continue
                rows.append([LABEL[a], al, f3(x["cov"]), f3(x.get("worst_cov")), f2(x["size"]), f3(x["leafres_correct"]),
                             f2(x["depth"]), f3(fa["cov"]) if fa else "–", f3(fa["worst_cov"]) if fa else "–",
                             f2(fa["size"]) if fa else "–", f3(fa["leafres_correct"]) if fa else "–"])
        md.append("E2 coverage vs size over alpha (full panel; then mean over deletion panels): coverage, worst type, "
                  "|C|, correct leaf-resolved, depth.\n\n")
        h = ["arm", "alpha", "cov", "worst", "|C|", "leaf-res", "depth", "del cov", "del worst", "del |C|", "del leaf-res"]
        md.append(table(h, rows) + "\n")
        tsv[f"E2_alpha_{run}"] = (h, rows)
        if "e4_split_distribution" in R:
            md.append(f"### E4 split distribution ({run}): equal-patient coverage over the 20 splits\n\n")
            rows = []
            for pn in ("full", "del_CD3e", "grp_CD3e+CD4+CD8+FOXP3", "del_CD68", "pilot_random_keep_half"):
                for a in ("admit", "admit_m1", "lac_node_proj", "lac_filt", "aps_filt", "patch_src_backstop",
                          "admit_crc", "admit_m1_crc", "lac_crc_filt"):
                    v = R["e4_split_distribution"].get(pn, {}).get(a)
                    if v:
                        rows.append([pn, LABEL[a], f3(v["mean"]), f3(v["sd"]), f3(v["min"]), f3(v["p5"]),
                                     f3(v["frac_below_0.90"]), f2(v["size_mean"])])
            h = ["panel", "arm", "mean", "sd", "min", "p5", "frac splits < 0.90", "EP |C|"]
            md.append(table(h, rows) + "\n")
            tsv[f"E4_{run}"] = (h, rows)
            rows = [[LABEL[a], f3(v["mean_over_splits_and_panels"]), f3(v["frac_split_panel_below_0.90"])]
                    for a, v in R["e4_all_deletions"].items() if a in LABEL]
            md.append("All deletion panels x 20 splits: mean equal-patient coverage, fraction of (split, panel) below 0.90.\n\n")
            md.append(table(["arm", "mean EP cov", "frac < 0.90"], rows) + "\n")
    if S.get("design"):
        md.append("## M6 panel design\n\n")
        for k, d in S["design"].items():
            md.append(f"### {k}\n\n")
            if k.endswith("_mean"):
                md.append("Mean over seeds " + ", ".join(d["seeds"]) + ".\n\n")
                rows = [[c["step"], ",".join(str(a) for a in c["added_by_seed"]), f3(c["pred_leafres_correct"]),
                         f3(c["real_leafres_correct"]), f3(c["pred_type_balanced"]), f3(c["real_type_balanced"]),
                         f2(c["pred_size"]), f2(c["real_size"])] for c in d["greedy_curve_mean"]]
                md.append(table(["step", "added (per seed)", "pred leaf-res", "real leaf-res", "pred type-bal",
                                 "real type-bal", "pred |C|", "real |C|"], rows) + "\n")
                if "marker_value_mean" in d:
                    mvo = sorted(d["marker_value_mean"].items(), key=lambda kv: -kv[1]["leafres_correct"])
                    md.append(table(["marker", "mean d correct leaf-res", "mean d depth", "mean d |C|"],
                                    [[m, f3(v["leafres_correct"]), f2(v["depth"]), f2(v["size"])] for m, v in mvo]) + "\n")
                if "budget_mean" in d:
                    rows = [[b, f3(v["pred_tb"]), f3(v["real_tb"]), f2(v["pred_depth"]), f2(v["real_depth"]),
                             f2(v["pred_size"]), f2(v["real_size"]),
                             ", ".join(f"{m}:{c}" for m, c in list(v["marker_frequency"].items())[:12])]
                            for b, v in d["budget_mean"].items()]
                    md.append("Budgeted recommendation (type-balanced objective over T; mean over seeds; marker: number "
                              "of seeds whose size-k panel contains it):\n\n")
                    md.append(table(["k", "pred tb", "real tb", "pred depth", "real depth", "pred |C|", "real |C|",
                                     "markers (seed count)"], rows) + "\n")
                    md.append(f"Type-balanced objective, predicted vs realized along the greedy path (mean over seeds): "
                              f"MAE {f3(d['tb_validation_mean']['mae'])}, r {f3(d['tb_validation_mean']['r'])}.\n\n")
                md.append("Validation (mean over seeds): " + "; ".join(
                    f"{m}: MAE {f3(x['mae'])}, r {f3(x['r'])}" for m, x in d["validation_per_type_mean"].items()) + ".\n\n")
                continue
            if "budget" in d:
                rows = [[b, ("n/a" if isinstance(v, str) else f3(v["pred_tb"])), ("n/a" if isinstance(v, str) else f3(v["real_tb"])),
                         ("n/a" if isinstance(v, str) else ", ".join(v["panel"]))] for b, v in d["budget"].items()]
                md.append(f"Types T: {', '.join(d['types_T'])}. Budgeted panels (type-balanced objective):\n\n")
                md.append(table(["k", "pred tb", "real tb", "panel"], rows) + "\n")
                if d["greedy_curve"] and "pred_tb_T" in d["greedy_curve"][0]:
                    md.append("Type-balanced curve (pred / real): " + "; ".join(
                        f"{c['step']}:{c['added'] or '-'} {f3(c['pred_tb_T'])}/{f3(c['real_tb_T'])}" for c in d["greedy_curve"]) + "\n\n")
            v = d["validation_per_type"]
            md.append("Validation, predicted (calibration cross-fit) vs realized (held-out test patients), per (panel, "
                      "cell type) over bank + greedy panels: " + "; ".join(
                          f"{m}: MAE {f3(x['mae'])}, r {f3(x['pearson_r'])} (n={x['n_pairs']})" for m, x in v.items()) + ".\n\n")
            rows = [[c["step"], c["added"] or "(empty)", f3(c["pred_leafres_correct"]), f3(c["real_leafres_correct"]),
                     f3(c["pred_type_balanced"]), f3(c["real_type_balanced"]), f2(c["pred_depth"]), f2(c["real_depth"]),
                     f2(c["pred_size"]), f2(c["real_size"])] for c in d["greedy_curve"]]
            md.append("Greedy forward selection (objective: cross-fitted correct leaf-resolved fraction):\n\n")
            md.append(table(["step", "added", "pred leaf-res", "real leaf-res", "pred type-bal", "real type-bal",
                             "pred depth", "real depth", "pred |C|", "real |C|"], rows) + "\n")
            mvo = sorted(d["marker_value_overall"].items(), key=lambda kv: -(kv[1]["leafres_correct"] or 0))
            rows = [[m, f3(v["leafres_correct"]), f3(v["leafres"]), f2(v["depth"]), f2(v["size"])] for m, v in mvo]
            md.append("Marker value (predicted full minus full-without-marker; positive = resolution lost when removed):\n\n")
            md.append(table(["marker", "d correct leaf-res", "d any-leaf res", "d depth", "d |C|"], rows) + "\n")
    if S.get("target_curve"):
        md.append("## Target-calibration curves (external cohorts; n labelled target cells, 10 seeds; n=0 = Orion "
                  "panel-matched, no target labels)\n\n")
        for k, e in S["target_curve"].items():
            md.append(f"### {k} (mean over clean cohorts: {', '.join(e['clean_cohorts'])})\n\n")
            arms = ["admit", "admit_m1", "admit_m1_n50", "lac_node_proj", "lac_filt", "aps_filt"]
            rows = []
            for n, row in e["mean_curve"].items():
                rows.append([n] + [("–" if a not in row else f"{f3(row[a]['cov'])} / {f3(row[a].get('worst'))} / "
                                    f"{f2(row[a]['size'])} / {f3(row[a]['leafres_correct'])}") for a in arms])
            md.append("cell: exact coverage / worst type / |C| / correct leaf-res (cell-level; no patient ids in caches).\n\n")
            md.append(table(["n target cells"] + [LABEL[a] for a in arms], rows) + "\n")
    if S.get("external"):
        md.append("## External decomposition with ADMIT+M1\n\n")
        for k, e in S["external"].items():
            m = e["mean"]
            cols = [c for c in m]
            md.append(f"### {k} (mean over clean cohorts: {', '.join(e['clean_cohorts'])})\n\n")
            md.append(table(["quantity", "exact cov", "|C|"], [[c, f3(m[c]["exact_cov"]), f2(m[c]["size"])] for c in cols]) + "\n")
    (R2 / "report_tables_round2.md").write_text("".join(md))
    (R2 / "tables").mkdir(exist_ok=True)
    for k, (h, rows) in tsv.items():
        (R2 / "tables" / f"{k}.tsv").write_text("\t".join(map(str, h)) + "\n" + "".join(
            "\t".join(map(str, r)) + "\n" for r in rows))


def main():
    S = {"runs": {}}
    for prot in ("exchangeable", "lopo7", "lopo13"):
        if (R2 / f"orion_{prot}.npz").exists():
            d, meta, _ = load_run("orion", prot)
            S["runs"][f"orion_{prot}"] = summarize_run(d, meta, None)
    import re
    names = sorted({re.sub(r"_s[0-9,\-]+$", "", Path(f).stem[len("cohort_"):])
                    for f in glob.glob(str(R2 / "cohort_*_s*.npz")) if "debug" not in f})
    for nm in names:     # e.g. schurch_logistic, schurch_xgb, later <tier1 cohort>_<head>
        d, meta, split = load_run("cohort", nm)
        if d is not None:
            S["runs"][nm] = summarize_run(d, meta, split)
    std_root = Path(sys.argv[sys.argv.index("--std-root") + 1]) if "--std-root" in sys.argv else R2 / "std"
    for f in sorted(std_root.glob("*/*.npz")):          # generic standard-schema runs (run_std.py)
        if "debug" in f.name:
            continue
        z = np.load(f, allow_pickle=True)
        meta = json.loads(Path(str(f).replace(".npz", "_meta.json")).read_text())
        S["runs"][f"std:{f.stem}"] = summarize_run({k: z[k] for k in z.files}, meta, None)
    S["design"] = design_summary()
    if (R2 / "external").exists():
        S["external"] = external_summary()
    if (R2 / "target_curve").exists():
        S["target_curve"] = target_curve_summary()
    write_json(R2 / "summary_round2.json", S)
    render(S)
    print("wrote", R2 / "summary_round2.json", R2 / "report_tables_round2.md")


if __name__ == "__main__":
    main()
