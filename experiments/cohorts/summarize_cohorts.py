#!/usr/bin/env python3
"""Reduce new-cohort accumulators to per-cohort JSON, cohorts/summary.tsv and cohorts/MANIFEST.tsv.

Within-cohort (run_within.py): cohorts/<cohort>/within_<head>_seed<k>.npz are summed over seeds on the union of test
patients (a patient recurring across seeds is one cluster) and reduced with summarize_round2.summarize_run
(pooled exact coverage with patient-cluster bootstrap CI, per-type coverage, worst type, sig-below strata,
resolution: finest depth, leaf-resolved fraction, |C|, abstention). Family rows add a patient-cluster bootstrap CI
of the MEAN coverage over the family's panels (same bootstrap weights for every panel).
Pairs (run_pair.py) and target-calibration curves write their own summaries; this script indexes them.

  python experiments/cohorts/summarize_cohorts.py [cohort ...]
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import std  # noqa: E402
from summarize_round2 import LABEL, boot_weights, cell_summary, summarize_run  # noqa: E402

LABEL = dict(LABEL)
LABEL.update({"admit_m1": "ADMIT+M1", "patch_src_backstop": "full-panel cal. (old, backstop)",
              "patch_src": "full-panel cal. (old, no backstop)", "admit_tgt": "ADMIT target-cal.",
              "admit_m1_tgt": "ADMIT+M1 target-cal."})
A01 = 1
SUMMARY_COLS = ["cohort", "protocol", "head", "panels", "method", "method_label", "n_panels", "exact_cov",
                "cov_ci_lo", "cov_ci_hi", "ci_note", "ep_cov", "worst_type", "worst_cov", "min_worst_cov",
                "n_sig_below", "n_types", "size", "depth", "leaf_resolved", "leaf_resolved_anyleaf", "abstain",
                "inadm_mass", "n_patients", "n_seeds", "status"]


def merge_within(files):
    metas, pats = [], set()
    for f in files:
        z = np.load(f, allow_pickle=True)
        pats.update(z["patients"].tolist())
        metas.append(json.loads(Path(f.replace(".npz", "_meta.json")).read_text()))
    patients = np.array(sorted(pats))
    acc, eps, epsz, seeds = None, [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        a = z["acc"]
        if acc is None:
            acc = np.zeros(a.shape[:3] + (len(patients),) + a.shape[4:], dtype=np.float64)
            base = {k: z[k] for k in ("panels", "arms", "alphas", "stats", "nodes")}
        assert list(z["panels"]) == list(base["panels"])
        acc[:, :, :, np.searchsorted(patients, z["patients"])] += a
        eps.append(z["ep_cov"]); epsz.append(z["ep_size"]); seeds.extend(z["seeds"].tolist())
    base.update({"acc": acc, "patients": patients})
    split = {"ep_cov": np.concatenate(eps), "ep_size": np.concatenate(epsz), "seeds": np.array(seeds)}
    meta = metas[0]
    meta["splits"] = [s for m in metas for s in m["splits"]]
    meta["test_accuracy_full_panel_by_seed"] = {m["seed"]: m["test_accuracy_full_panel"] for m in metas}
    return base, meta, split


def family_ci(acc, panels_idx, arm_i, W):
    """patient-cluster bootstrap CI of the mean (over panels) pooled exact coverage at alpha 0.10."""
    vals = []
    for pi in panels_idx:
        G = acc[pi, arm_i, A01]
        gh, gn = G[:, :, 0].sum(1), G[:, :, 1].sum(1)
        if gn.sum() == 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            vals.append((W @ gh) / (W @ gn))
    if not vals:
        return None
    m = np.nanmean(np.vstack(vals), axis=0)
    return [float(np.nanpercentile(m, 2.5)), float(np.nanpercentile(m, 97.5))]


def within_rows(cohort, head, d, res, n_seeds):
    rows = []
    acc, panels, arms = d["acc"], list(d["panels"]), list(d["arms"])
    W = boot_weights(acc.shape[3])
    npat = int(acc.shape[3])
    fam_panels = {}
    for pn in panels:
        fam_panels.setdefault(res["panels"][pn]["family"], []).append(pn)
    fam_panels["all_deletions"] = [p for p in panels if res["panels"][p]["family"] != "full"]
    for fam, pns in fam_panels.items():
        pidx = [panels.index(p) for p in pns]
        for ai, a in enumerate(arms):
            agg = res["families"].get(fam, {}).get("arms", {}).get(a, {}).get("0.10")
            if not agg:
                continue
            if fam == "full":
                r = res["panels"]["full"]["arms"][a]["0.10"]
                ci, note = r["cov_ci"], "patient-cluster bootstrap (B=1000), pooled over seeds"
                worst, wt = r.get("worst_cov"), r.get("worst_type")
            else:
                ci = family_ci(acc, pidx, ai, W)
                note = "patient-cluster bootstrap of the mean over panels (B=1000)"
                worst, wt = agg.get("worst_cov"), ""
            rows.append({"cohort": cohort, "protocol": "within_patient_split", "head": head, "panels": fam, "status": "final",
                         "method": a, "method_label": LABEL.get(a, a), "n_panels": len(pns), "exact_cov": agg["cov"],
                         "cov_ci_lo": ci[0] if ci else None, "cov_ci_hi": ci[1] if ci else None, "ci_note": note,
                         "ep_cov": agg.get("ep_cov"), "worst_type": wt, "worst_cov": worst,
                         "min_worst_cov": agg.get("min_worst_cov"), "n_sig_below": agg.get("n_sig_below"),
                         "n_types": agg.get("n_types"), "size": agg.get("size"), "depth": agg.get("depth"),
                         "leaf_resolved": agg.get("leafres_correct"), "leaf_resolved_anyleaf": agg.get("leafres"),
                         "abstain": agg.get("abst"), "inadm_mass": agg.get("inadm_mass"), "n_patients": npat,
                         "n_seeds": n_seeds})
    return rows


def _merge_block(files, key, pkey):
    pats = set()
    for f in files:
        z = np.load(f, allow_pickle=True)
        if key in z.files:
            pats.update(z[pkey].tolist())
    if not pats:
        return None, None
    patients = np.array(sorted(pats))
    acc = None
    for f in files:
        z = np.load(f, allow_pickle=True)
        if key not in z.files:
            continue
        a = z[key]
        if acc is None:
            acc = np.zeros(a.shape[:-3] + (len(patients),) + a.shape[-2:], dtype=np.float64)
        acc[..., np.searchsorted(patients, z[pkey]), :, :] += a
    return acc, patients


def _row(base, arm, cs, npat, nseeds, extra=None):
    r = {**base, "method": arm, "method_label": LABEL.get(arm, arm), "n_panels": 1, "exact_cov": cs.get("cov"),
         "cov_ci_lo": cs.get("cov_ci", [None, None])[0], "cov_ci_hi": cs.get("cov_ci", [None, None])[1],
         "ci_note": "patient-cluster bootstrap (B=1000), pooled over seeds", "ep_cov": cs.get("ep_cov"),
         "worst_type": cs.get("worst_type"), "worst_cov": cs.get("worst_cov"), "min_worst_cov": cs.get("worst_cov"),
         "n_sig_below": len(cs.get("sig_below", [])), "n_types": cs.get("n_types"), "size": cs.get("size"),
         "depth": cs.get("depth"), "leaf_resolved": cs.get("leafres_correct"), "leaf_resolved_anyleaf": cs.get("leafres"),
         "abstain": cs.get("abst"), "inadm_mass": cs.get("inadm_mass"), "n_patients": npat, "n_seeds": nseeds}
    r.update(extra or {})
    return r


def summarize_pair(pair):
    reg = std.registry()["pairs"].get(pair)
    if reg is None:
        return [], []
    rows, written = [], []
    cdir = std.OUT / pair
    status = reg.get("status", "final")
    for d in ("ab", "ba"):
        for head in ("xgb", "mlp", "logistic"):
            files = sorted(f for f in glob.glob(str(cdir / f"pair_{d}_{head}_seed*.npz")) if Path(f.replace(".npz", "_meta.json")).exists())
            if not files:
                continue
            meta = json.loads(Path(files[0].replace(".npz", "_meta.json")).read_text())
            z0 = np.load(files[0], allow_pickle=True)
            arms, nodes = list(z0["arms"]), list(z0["nodes"])
            alphas = [float(a) for a in z0["alphas"]]
            src, tgt = meta["source"], meta["target"]
            res = {"pair": pair, "protocol": reg.get("protocol"), "status": status, "direction": d, "head": head,
                   "source": src, "target": tgt, "shared_panel": meta["shared_panel"],
                   "source_only_markers": meta["source_only_markers"], "target_only_markers": meta["target_only_markers"],
                   "admissible_under_P": meta["admissible_under_P"], "seeds": [], "blocks": {}, "arm_labels": LABEL}
            accs = {}
            for f in files:
                res["seeds"].append(int(np.load(f, allow_pickle=True)["seeds"][0]))
                m = json.loads(Path(f.replace(".npz", "_meta.json")).read_text())
                res.setdefault("accuracy_by_seed", {})[m["seed"]] = m["accuracy"]
            for blk, pk in (("transfer", "patients_B"), ("within_source_masked", "patients_A"),
                            ("target_full", "patients_B")):
                acc, pats = _merge_block(files, f"acc_{blk}", pk)
                if acc is None:
                    continue
                W = boot_weights(len(pats))
                res["blocks"][blk] = {"n_patients": int(len(pats)), "arms": {}}
                for ai, a in enumerate(arms):
                    res["blocks"][blk]["arms"][a] = {}
                    for li, al in enumerate(alphas):
                        G = acc[0, ai, li]
                        if G[:, :, 4].sum() == 0:
                            continue
                        cs = cell_summary(G, nodes, W)
                        if cs is None:
                            continue
                        res["blocks"][blk]["arms"][a][f"{al:.2f}"] = cs
                        if li == A01:
                            base = {"cohort": pair, "protocol": reg.get("protocol"), "head": head,
                                    "panels": f"{src}->{tgt}:{blk}", "status": status}
                            rows.append(_row(base, a, cs, len(pats), len(files)))
            acc, pats = _merge_block(files, "tcal", "patients_B")
            if acc is not None:
                ns, tarms = [int(n) for n in z0["tcal_ns"]], list(z0["tcal_arms"])
                W = boot_weights(len(pats))
                res["target_calibration_curve"] = {}
                for ni, n in enumerate(ns):
                    res["target_calibration_curve"][n] = {}
                    for k, a in enumerate(tarms):
                        G = acc[ni, :, k].sum(0)
                        cs = cell_summary(G, nodes, W)
                        if cs is None:
                            continue
                        draws = []          # per (seed, draw) coverage: calibration-sample variability
                        for f in files:
                            t = np.load(f, allow_pickle=True)["tcal"][ni, :, k]
                            gh, gn = t[..., 0].sum((1, 2)), t[..., 1].sum((1, 2))
                            draws += list(gh[gn > 0] / gn[gn > 0])
                        cs = {kk: v for kk, v in cs.items() if kk not in ("labels",)}
                        cs["draw_cov_mean"] = float(np.mean(draws))
                        cs["draw_cov_p10_p90"] = [float(np.percentile(draws, 10)), float(np.percentile(draws, 90))]
                        cs["n_draws"] = len(draws)
                        res["target_calibration_curve"][n][a] = cs
                        base = {"cohort": pair, "protocol": "target_calibration_curve", "head": head,
                                "panels": f"{src}->{tgt}:n={n}", "status": status}
                        rows.append(_row(base, a, cs, len(pats), len(files),
                                         {"ci_note": f"patient-cluster bootstrap, pooled over {len(draws)} draws x seeds; "
                                                     f"draw p10-p90 {cs['draw_cov_p10_p90'][0]:.3f}-{cs['draw_cov_p10_p90'][1]:.3f}"}))
            out = cdir / f"summary_pair_{d}_{head}.json"
            nexp = len(reg["splits"]["seeds"])
            res["completeness"] = f"{len(res['seeds'])}/{nexp} seeds"
            std.write_json(out, res)
            st = ("provisional" if "PROVISIONAL" in status else "final") if len(res["seeds"]) >= nexp else \
                f"partial {len(res['seeds'])}/{nexp} seeds" + (" (provisional)" if "PROVISIONAL" in status else "")
            written.append((out, f"{reg.get('protocol')}: {src} -> {tgt}, {head}, seeds {sorted(res['seeds'])}; "
                                 f"blocks transfer/within_source_masked/target_full + target-calibration curve", st))
    return rows, written


def summarize_within(cohort):
    rows, written = [], []
    cdir = std.OUT / cohort
    for head in ("xgb", "mlp", "logistic"):
        files = sorted(f for f in glob.glob(str(cdir / f"within_{head}_seed*.npz")) if "debug" not in f and Path(f.replace(".npz", "_meta.json")).exists())
        if not files:
            continue
        d, meta, split = merge_within(files)
        res = summarize_run(d, meta, split)
        res["cohort"], res["head"], res["protocol"] = cohort, head, "within_patient_split"
        res["seeds"] = sorted(int(s) for s in split["seeds"])
        res["test_accuracy_full_panel_by_seed"] = meta["test_accuracy_full_panel_by_seed"]
        res["transform"], res["markers"] = meta["transform"], meta["markers"]
        res["excluded_labels"], res["label_counts"] = meta["excluded_labels"], meta["label_counts"]
        res["arm_labels"] = LABEL
        out = cdir / f"summary_within_{head}.json"
        nexp = len(std.registry()[cohort]["splits"]["seeds"])
        res["completeness"] = f"{len(res['seeds'])}/{nexp} seeds"
        std.write_json(out, res)
        st = "final" if len(res["seeds"]) >= nexp else f"partial {len(res['seeds'])}/{nexp} seeds"
        written.append((out, f"within-cohort paired deletion, {head} head, seeds {res['seeds']}, "
                             f"{len(d['panels'])} unique panels, {d['acc'].shape[3]} test patients", st))
        rows += within_rows(cohort, head, d, res, len(files))
    return rows, written


def write_tables(all_rows, manifest):
    import csv
    S = std.OUT / "summary.tsv"
    old = []
    if S.exists():
        with open(S) as f:
            old = list(csv.DictReader(f, delimiter="\t"))
    keys = {r["cohort"] for r in all_rows}
    keep = [r for r in old if r["cohort"] not in keys]
    rows = keep + all_rows
    fmt = lambda v: (f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v)))  # noqa: E731
    with open(S, "w") as f:
        f.write("\t".join(SUMMARY_COLS) + "\n")
        for r in rows:
            f.write("\t".join(fmt(r.get(c)) for c in SUMMARY_COLS) + "\n")
    update_manifest(manifest)


def update_manifest(entries):
    """entries: [(path, contents, status)]; replaces rows with the same file."""
    M = std.OUT / "MANIFEST.tsv"
    cur = {}
    if M.exists():
        for line in M.read_text().splitlines()[1:]:
            if line.strip():
                f, c, s = (line.split("\t") + ["", "", ""])[:3]
                cur[f] = (c, s)
    for p, c, s in entries:
        cur[str(Path(p)).replace(str(std.OUT) + "/", "")] = (c, s)
    M.write_text("file\tcontents\tstatus\n" + "".join(f"{f}\t{c}\t{s}\n" for f, (c, s) in sorted(cur.items())))


def main():
    cohorts = sys.argv[1:] or sorted(p.name for p in std.OUT.iterdir() if p.is_dir() and not p.name.startswith("_"))
    all_rows, man = [], []
    pairs = std.registry()["pairs"]
    for c in cohorts:
        if c in pairs:
            rows, written = summarize_pair(c)
            man.append((std.OUT / c / "summary_pair_<ab|ba>_<head>.json#target_calibration_curve",
                        "TARGET RECALIBRATION arm (Fig 5a/5c): same source-trained classifier, calibrated on n labelled "
                        "target cal-patient cells (n=50/100/200/500/1000/2000; 10 draws x 5 split seeds), tested on target "
                        "held-out patients; arms admit_m1/admit/lac_node_proj/lac_filt/aps_filt. Also in summary.tsv: "
                        "protocol=target_calibration_curve, panels '<src>-><tgt>:n=<n>'", "see pair summary status"))
            man.append((std.OUT / c / "pair_<ab|ba>_<head>_seed<k>.npz", "raw additive accumulators per run "
                        "(acc_transfer / acc_within_source_masked / acc_target_full [panel, arm, alpha, patient, node, "
                        "stat]; tcal [n, draw, arm, patient, node, stat]) + _meta.json", "raw"))
        else:
            rows, written = summarize_within(c)
            man.append((std.OUT / c / "within_<head>_seed<k>.npz", "raw additive accumulators per run "
                        "(acc [panel, arm, alpha, test patient, node, stat]) + _meta.json; panel_bank.json = fixed bank",
                        "raw"))
        man += written
        all_rows += rows
    man.append((std.OUT / "summary.tsv", "one row per cohort x protocol x head x panel family x method (alpha 0.10)",
                "updated"))
    write_tables(all_rows, man)
    print("summarized", cohorts, len(all_rows), "rows")


if __name__ == "__main__":
    main()
