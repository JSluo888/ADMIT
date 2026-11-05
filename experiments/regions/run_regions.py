#!/usr/bin/env python3
"""Region-selection runner: one cohort x one seed (Orion: one lopo13 fold), primary low-plex panel L and every
rand50_* bank panel as L, the full panel as F.

  python experiments/regions/run_regions.py --cohort cords2024_nsclc_imc --seed 0
  python experiments/regions/run_regions.py --cohort schurch --seed 0 --head xgb
  python experiments/regions/run_regions.py --cohort orion --seed 3           # needs the in-house Orion caches

Training patients, training-cell caps, transform, z-score, mean imputation (absent marker = 0 after z-scoring) and
the head are those of the paired-deletion runs (run_within3.compute3 for the standard-schema cohorts, run_cohort3
for Schurch, the round0 backbone for Orion). Their per-patient evaluation caps are replaced by per-region caps.
Regions: Cords / Danenberg sample_id, Schurch spots, Orion (sample, region). N_r = all rows of the region in the
cohort table (labelled or not); context cells for the scores = every row of the region (census) unless
--context-cap is given; calibration / evaluation cells = labelled rows, <= 1,000 per region.
Writes $ADMIT_RUNS/regions/<cohort>/regions_<head>_seed<k>{_tag}.npz + _meta.json (+ seed 0: cells parquet).
Nothing printed here contains a patient or region identifier; endpoints are not printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent, HERE.parent / "cohorts", HERE.parent / "round2", HERE.parents[1]):
    sys.path.insert(0, str(p))
import regions_core as rc  # noqa: E402
from admit.ontology import NODE17  # noqa: E402
from admit.panel_matched import Ontology  # noqa: E402

RUNS = Path(os.environ.get("ADMIT_RUNS", str(HERE.parents[1] / "runs")))
OUT = Path(os.environ.get("ADMIT_REGIONS_OUT", str(RUNS / "regions")))
L_BASE = ["PanCK", "CD45", "CD3e", "CD8", "CD20", "CD68"]      # low-plex panel, fixed before results
CD45_ISOFORMS = ["CD45RA", "CD45RO", "CD45RB"]
STD_COHORTS = ["cords2024_nsclc_imc", "danenberg2022_metabric_imc", "fischer2023_breast_imc", "meyer2025_tnbc_imc",
               "neotrip_tnbc_imc", "quiche_tnbc_mibi_spain", "quiche_tnbc_mibi_stanford", "launonen2022_hgsc_tcycif"]
# additional standard-schema cohorts, same design
STD_COHORTS += ["jackson2020_breast_imc", "girnius2026_tcycif", "gupta2026_normal_breast_imc",
                "piyadasa2025_glioma_mibi_immune", "piyadasa2025_glioma_mibi_tumor", "mcc_discovery_haist_codex",
                "chl_mibi_stanford", "hoch2022_melanoma_imc", "knott2024_tnbc_codex"]


def low_plex(markers: List[str]):
    """L in the cohort's standardised names; CD45 absent -> every CD45 isoform present; else dropped."""
    out, notes = [], []
    for m in L_BASE:
        if m in markers:
            out.append(m)
        elif m == "CD45":
            iso = [x for x in CD45_ISOFORMS if x in markers]
            out += iso
            notes.append(f"CD45 absent; isoforms used: {iso}" if iso else "CD45 and isoforms absent; L has 5 markers")
        else:
            notes.append(f"{m} absent from the cohort panel")
    return out, notes


def target_types():
    return list(NODE17)          # the 17-node column order shared by every experiment


def chunked(fn: Callable, rows: np.ndarray, size: int = 400_000) -> np.ndarray:
    out = None
    for i in range(0, len(rows), size):
        P = fn(rows[i:i + size])
        if out is None:
            out = np.empty((len(rows), P.shape[1]), dtype=np.float64)
        out[i:i + size] = P
    return out if out is not None else np.zeros((0, 17))


# ----------------------------------------------------------------------------------------------------------------
# Loaders: table (all rows) + per-seed fold (roles, training labels, predictor)
# ----------------------------------------------------------------------------------------------------------------


class StdCohort:
    """Standard-schema cohorts (Cords, Danenberg, Fischer, Meyer, NeoTRIP, QUICHE Spain / Stanford, Launonen, ...):
    run_within3.compute3 replicated on the labelled rows."""

    def __init__(self, cohort: str):
        import std
        self.std, self.cohort = std, cohort
        self.reg = std.registry()[cohort]
        if not std.is_ready(cohort):
            raise SystemExit(f"{cohort}: no READY file")
        d = std.cohort_dir(cohort)
        src_cols, names, dropped = std.feature_table(d, self.reg.get("synonyms"))
        self.markers, self.dropped = names, dropped
        df = pd.read_parquet(d / "cells.parquet", columns=["patient_id", "sample_id", "x", "y", "node17"] + src_cols)
        tt = target_types()
        idx = {t: i for i, t in enumerate(tt)}
        lab = df["node17"].astype(str)
        keep = lab.isin(idx).to_numpy()
        self.X = std.transform(df[src_cols].to_numpy(dtype=np.float32), self.reg["transform"]).astype(np.float32)
        self.y = np.full(len(df), -1, dtype=np.int64)
        self.y[keep] = lab[keep].map(idx).to_numpy(np.int64)
        raw = df["patient_id"].astype(str).to_numpy()
        uq, inv = np.unique(raw, return_inverse=True)
        self.patient = np.array([std.anon(self.reg.get("cohort_key") or cohort, p) for p in uq])[inv]
        self.region = df["sample_id"].astype(str).to_numpy()
        self.xy = df[["x", "y"]].to_numpy(dtype=np.float64)
        self.lab_rows = np.flatnonzero(keep)
        self.pkeys = np.sort(np.unique(self.patient[self.lab_rows]))
        self.transform_spec = self.reg["transform"]
        self.bank = json.loads((std.OUT / cohort / "panel_bank.json").read_text())
        assert self.bank["full_panel"] == self.markers, "bank full panel does not match the cohort features"
        self.F = list(self.bank["full_panel"])
        del df

    def fold(self, s: int, head_kind: str):
        std = self.std
        pcode = np.searchsorted(self.pkeys, self.patient[self.lab_rows])
        groups = std.patient_groups(pcode, len(self.pkeys))
        sp = std.patient_splits(np.arange(len(self.pkeys)), self.reg["splits"]["fractions"], [s])[0]
        rng = np.random.default_rng(s)
        tr = self.lab_rows[std.train_index(groups, sp["train"], self.reg.get("train_cap", 2000), rng)]
        tfit = std.fit_transform(self.X, self.transform_spec, tr)
        if tfit:
            self.transform_spec = "none"          # fitted in place once; never refit on a second fold
        mu, sd = self.X[tr].mean(0), self.X[tr].std(0)
        sd[sd == 0] = 1.0
        t0 = time.time()
        head = std.Head(head_kind).fit(((self.X[tr] - mu) / sd).astype(np.float32), self.y[tr], s)
        if head_kind == "xgb":
            head.m.set_params(n_jobs=int(os.environ.get("REGIONS_PREDICT_NJOBS", "16")))
        fit_s = time.time() - t0
        markers = self.markers

        def predict(panel, rows):
            miss = [j for j, m in enumerate(markers) if m not in set(panel)]

            def f(r):
                Z = ((self.X[r] - mu) / sd).astype(np.float32)
                Z[:, miss] = 0.0
                return head.predict17(Z)
            return chunked(f, np.asarray(rows))

        roles = {"train": self.pkeys[sp["train"]], "cal": self.pkeys[sp["cal"]], "test": self.pkeys[sp["test"]]}
        return roles, self.y[tr], predict, {"head_fit_s": round(fit_s, 1), "n_train_cells": int(len(tr)),
                                            "head_device": getattr(head, "device", None), "transform_fit": tfit}


class Schurch:
    """Schurch CRC CODEX: run_cohort3 replicated (make_splits, training cap, z-score, fit_head)."""

    def __init__(self):
        from cohort import load_config
        from schurch_common import SRC, VOCAB
        from std import anon
        self.cohort = "schurch"
        self.cfg = load_config(HERE.parents[1] / "configs" / "schurch.json")
        df = pd.read_csv(SRC, low_memory=False)
        mp = json.loads((RUNS / "schurch" / "schurch_label_mapping.json").read_text())
        keep = (~df.ClusterName.isin(mp["excluded_unmapped_labels"])).to_numpy()
        mcols = [c for c in df if ":Cyc_" in c and not c.startswith(("HOECHST", "DRAQ5"))]
        self.markers = [VOCAB.get(c.split(" - ")[0].split(":")[0], c.split(" - ")[0].split(":")[0]) for c in mcols]
        self.dropped = []
        with np.errstate(invalid="ignore", divide="ignore"):
            self.X = np.log1p(df[mcols].to_numpy(dtype=np.float64))
        tt = target_types()
        idx = {t: i for i, t in enumerate(tt)}
        label = df.ClusterName.map({k: v["target"] for k, v in mp["mapping"].items()})
        assert label[keep].notna().all()
        self.y = np.full(len(df), -1, dtype=np.int64)
        self.y[keep] = label[keep].map(idx).to_numpy(np.int64)
        self.raw_patient = df["patients"].astype(str).to_numpy()
        uq, inv = np.unique(self.raw_patient, return_inverse=True)
        self.patient = np.array([anon("schurch", p) for p in uq])[inv]
        self.region = df["spots"].astype(str).to_numpy()
        self.xy = df[["X:X", "Y:Y"]].to_numpy(dtype=np.float64)
        self.lab_rows = np.flatnonzero(keep)
        bank = json.loads(Path(self.cfg["panel_bank"]).read_text())
        assert bank["full_panel"] == self.markers
        self.bank, self.F = bank, list(bank["full_panel"])
        self.n_nonfinite_rows = int((~np.isfinite(self.X).all(axis=1)).sum())
        del df

    def fold(self, s: int, head_kind: str):
        from cohort import make_splits
        from run_cohort import fit_head
        pl = self.raw_patient[self.lab_rows]
        sp = make_splits(pl, {**self.cfg["splits"], "seeds": [s]})[0]
        rng = sp["rng"]
        cap = self.cfg.get("train_cap", 2000)
        tr = self.lab_rows[np.concatenate([rng.choice(np.flatnonzero(pl == g), min(cap, int((pl == g).sum())),
                                                      replace=False) for g in sp["train"]])]
        mu, sd = self.X[tr].mean(0), self.X[tr].std(0)
        sd[sd == 0] = 1.0
        t0 = time.time()
        head, present, note = fit_head(head_kind, (self.X[tr] - mu) / sd, self.y[tr], s)
        if head_kind == "xgb":
            head.set_params(n_jobs=int(os.environ.get("REGIONS_PREDICT_NJOBS", "16")))
        fit_s = time.time() - t0
        markers = self.markers

        def predict(panel, rows):
            miss = [j for j, m in enumerate(markers) if m not in set(panel)]

            def f(r):
                Z = (self.X[r] - mu) / sd
                Z[:, miss] = 0.0
                P = np.zeros((len(Z), 17))
                P[:, present] = head.predict_proba(Z)
                return P / P.sum(1, keepdims=True)
            return chunked(f, np.asarray(rows))

        key = dict(zip(self.raw_patient[self.lab_rows], self.patient[self.lab_rows]))
        roles = {k: np.array([key[g] for g in sp[k]]) for k in ("train", "cal", "test")}
        return roles, self.y[tr], predict, {"head_fit_s": round(fit_s, 1), "n_train_cells": int(len(tr)),
                                            "head_note": note}


class Orion:
    """Orion lopo13 (round0 backbone, frozen; training labels crc_train_full); needs the in-house caches. No
    20,000-cell patient cap (the per-region caps replace it). Region = (sample, region field). Centroids: identity
    sidecars (crc_cal, crc_test); crc_test_extra has none unless --orion-xy-join (in-memory join on cell_id with the
    per-region cell tables in $ADMIT_ORION_REGIONS_DIR, read-only)."""

    def __init__(self, xy_join: bool = False):
        from common import get_backbone, load_identity, load_split
        from std import anon
        self.cohort = "orion"
        tt = target_types()
        self.bb = get_backbone("round0", tt)
        cal, test = load_split("crc_cal"), load_split("crc_test")
        ic, it = load_identity("crc_cal"), load_identity("crc_test")
        if ic is None or it is None:
            raise SystemExit("ADMIT_IDENTITY_DIR not set or sidecars missing")
        ex = np.load(RUNS / "crc_rebuild" / "crc_test_extra.npz", allow_pickle=True)
        self.X = np.concatenate([self.bb.features(cal), self.bb.features(test), ex["X_arc"].astype(np.float32)])
        self.y = np.concatenate([cal["y"], test["y"], ex["y"]]).astype(np.int64)
        self.raw_patient = np.concatenate([ic["sample"], it["sample"], ex["sample"].astype(str)])
        reg = np.concatenate([ic["region"].astype(str), it["region"].astype(str), ex["region"].astype(str)])
        # imaging field = (slide, region field); one patient of crc_test_extra has two slides, so its region
        # fields are keyed by slide (the sidecars' sample is the slide for every other patient)
        slide = np.concatenate([ic["sample"], it["sample"], ex["slide"].astype(str)]).astype(str)
        self.region = np.char.add(np.char.add(slide, "|"), reg)
        xy_ex = np.full((len(ex["y"]), 2), np.nan)
        self.xy_source = "identity sidecars (crc_cal, crc_test); crc_test_extra: none"
        if xy_join:
            xy_ex = self._join_xy(ex)
            self.xy_source = "identity sidecars + in-memory cell_id join for crc_test_extra"
        self.xy = np.concatenate([np.stack([ic["x_centroid"], ic["y_centroid"]], 1),
                                  np.stack([it["x_centroid"], it["y_centroid"]], 1), xy_ex]).astype(np.float64)
        uq, inv = np.unique(self.raw_patient, return_inverse=True)
        self.patient = np.array([anon("orion", p) for p in uq])[inv]
        self.patients_sorted = np.array(sorted(uq.tolist()))
        self.markers = list(self.bb.markers)
        self.dropped = []
        self.bank = json.loads((RUNS / "panel_bank.json").read_text())
        self.F = list(self.bank["orion_panel"])
        self.y_train = load_split("crc_train_full")["y"].astype(np.int64)

    @staticmethod
    def _join_xy(ex):
        root = Path(os.environ.get("ADMIT_ORION_REGIONS_DIR",
                                   os.path.join(os.environ.get("ADMIT_DATA_ROOT", str(HERE.parents[1] / "data")),
                                                "subsampled_regions")))
        xy = np.full((len(ex["y"]), 2), np.nan)
        smp, reg, cid = ex["slide"].astype(str), ex["region"].astype(str), ex["cell_id"].astype(np.int64)
        for s_ in np.unique(smp):
            for r_ in np.unique(reg[smp == s_]):
                p = root / s_ / r_ / "cells_in_region.csv"
                if not p.exists():
                    continue
                t = pd.read_csv(p, usecols=["CellID", "X_centroid", "Y_centroid"]).drop_duplicates("CellID")
                m = np.flatnonzero((smp == s_) & (reg == r_))
                j = pd.Index(t["CellID"].to_numpy()).get_indexer(cid[m])
                ok = j >= 0
                xy[m[ok]] = t[["X_centroid", "Y_centroid"]].to_numpy()[j[ok]]
        return xy

    def fold(self, f: int, head_kind: str):
        from std import anon
        test_raw = self.patients_sorted[f]
        others = [p for p in self.patients_sorted if p != test_raw]
        bb = self.bb

        def predict(panel, rows):
            return chunked(lambda r: bb.predict17(bb.mask(self.X[r], panel)), np.asarray(rows))

        roles = {"train": np.array([]), "cal": np.array([anon("orion", p) for p in others]),
                 "test": np.array([anon("orion", test_raw)])}
        return roles, self.y_train, predict, {"head_fit_s": 0.0, "backbone": bb.name,
                                              "n_train_cells": int(len(self.y_train))}


def load_cohort(name: str, args):
    if name in STD_COHORTS:
        return StdCohort(name)
    if name == "schurch":
        return Schurch()
    if name == "orion":
        return Orion(args.orion_xy_join)
    raise SystemExit(f"unknown cohort {name}")


# ----------------------------------------------------------------------------------------------------------------
# Fold assembly
# ----------------------------------------------------------------------------------------------------------------


def build_regions(tab, roles: Dict[str, np.ndarray], s: int, context_cap, eval_cap):
    """-> R (per kept region), ctx_rows, ctx_reg, ev_rows, ev_reg, bookkeeping counts."""
    dev, cal = rc.dev_cal_split(roles["cal"], s)
    role_of = {**{k: 0 for k in dev}, **{k: 1 for k in cal}, **{k: 2 for k in roles["test"]}}
    rid, rcode = np.unique(tab.region, return_inverse=True)            # region_index = position in sorted ids
    N_all = np.bincount(rcode, minlength=len(rid))
    pid, pcode = np.unique(tab.patient, return_inverse=True)
    pmin = np.full(len(rid), np.iinfo(np.int64).max)
    pmax = np.full(len(rid), -1)
    np.minimum.at(pmin, rcode, pcode)
    np.maximum.at(pmax, rcode, pcode)
    if (pmin != pmax).any():
        raise ValueError(f"{int((pmin != pmax).sum())} regions span more than one patient")
    r_role = np.array([role_of.get(pid[c], -1) for c in pmin])
    in_role = r_role >= 0
    small = in_role & (N_all < rc.MIN_REGION_CELLS)
    kept = np.flatnonzero(in_role & ~small)
    order = np.argsort(rcode, kind="stable")
    bounds = np.searchsorted(rcode[order], np.arange(len(rid) + 1))
    ctx_rows, ctx_reg, ev_rows, ev_reg, area = [], [], [], [], np.zeros(len(kept))
    for i, c in enumerate(kept):
        rows = order[bounds[c]:bounds[c + 1]]
        cr = rc.sample_rows(rows, c, context_cap, s, rc.SALT_CONTEXT)
        lab = rows[tab.y[rows] >= 0]
        er = rc.sample_rows(lab, c, eval_cap, s, rc.SALT_EVAL)
        ctx_rows.append(cr)
        ctx_reg.append(np.full(len(cr), i))
        ev_rows.append(er)
        ev_reg.append(np.full(len(er), i))
        area[i] = rc.convex_hull_area(tab.xy[rows])
    kp = pmin[kept]
    upat, pslot = np.unique(kp, return_inverse=True)
    from std import anon
    R = {"role": r_role[kept], "patient": pslot, "N_r": N_all[kept], "region_id": rid[kept], "region_index": kept,
         "area": area, "region_key": np.array([anon(tab.cohort + "|region", str(x)) for x in rid[kept]]),
         "slot_keys": pid[upat]}
    book = {"n_patients": {"dev": int(len(dev)), "cal": int(len(cal)), "test": int(len(roles["test"]))},
            "n_regions_in_roles": {k: int((r_role == i).sum()) for i, k in enumerate(("dev", "cal", "test"))},
            "n_regions_dropped_lt50": {k: int((small & (r_role == i)).sum()) for i, k in enumerate(("dev", "cal",
                                                                                                    "test"))},
            "n_regions_kept": {k: int((R["role"] == i).sum()) for i, k in enumerate(("dev", "cal", "test"))}}
    cat = lambda L: np.concatenate(L) if L else np.zeros(0, np.int64)  # noqa: E731
    return R, cat(ctx_rows), cat(ctx_reg), cat(ev_rows), cat(ev_reg), book


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, choices=STD_COHORTS + ["schurch", "orion"])
    ap.add_argument("--seed", type=int, required=True, help="seed s (Orion: lopo13 fold index 0..12)")
    ap.add_argument("--head", default="xgb", choices=["xgb", "logistic"])
    ap.add_argument("--panels", default="all", choices=["all", "primary"])
    ap.add_argument("--context-cap", type=int, default=0, help="0 = census (default)")
    ap.add_argument("--u-def", default="chain", choices=["chain", "singleton"], help="chain (primary) or singleton")
    ap.add_argument("--orion-xy-join", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--timing-only", action="store_true", help="stop after the F and primary-L predictions")
    ap.add_argument("--selection-only", action="store_true",
                    help="blind phase: selections + stratum bookkeeping only (no TEST label, no endpoint)")
    args = ap.parse_args()
    t0 = time.time()
    tm = {}
    tab = load_cohort(args.cohort, args)
    tm["load_s"] = round(time.time() - t0, 1)
    onto = Ontology.from_repo(target_types())
    L, notes = low_plex(tab.markers)
    panels = [("L6", L)]
    if args.panels == "all":
        panels += [(p["name"], list(p["markers"])) for p in tab.bank["panels"] if p["name"].startswith("rand50")]
    roles, y_train, predict, head_info = tab.fold(args.seed, args.head)
    tm["head_fit_s"] = head_info.get("head_fit_s")
    t1 = time.time()
    R, ctx_rows, ctx_reg, ev_rows, ev_reg, book = build_regions(tab, roles, args.seed, args.context_cap or None,
                                                                rc.EVAL_CAP)
    tm["regions_s"] = round(time.time() - t1, 1)
    t1 = time.time()
    P17F = predict(tab.F, ev_rows)
    PnF = onto.aggregate(P17F)
    tm["predict_F_s"] = round(time.time() - t1, 1)
    union = np.unique(np.concatenate([ctx_rows, ev_rows]))
    ci, ei = np.searchsorted(union, ctx_rows), np.searchsorted(union, ev_rows)
    ev_y = tab.y[ev_rows]
    print(f"[{args.cohort} s{args.seed}] regions kept {book['n_regions_kept']} context cells {len(ctx_rows)} "
          f"eval cells {len(ev_rows)} union {len(union)}; load {tm['load_s']}s head {tm['head_fit_s']}s "
          f"predict F {tm['predict_F_s']}s", flush=True)
    res, tm["panels"] = [], {}
    for pi_, (pname, pm) in enumerate(panels):
        t1 = time.time()
        P17L_u = predict(pm, union)
        tp = round(time.time() - t1, 1)
        if args.timing_only:
            tm["predict_L_union_s"] = tp
            print(f"[timing-only] predict L on {len(union)} rows {tp}s", flush=True)
            break
        PnL_u = onto.aggregate(P17L_u)
        out = rc.run_panel(onto, tab.F, pm, y_train, args.seed, args.cohort, R, ctx_reg, PnL_u[ci], ev_reg, ev_y,
                           PnL_u[ei], PnF, P17L_u[ei], u_def=args.u_def,
                           keep_cells=(pi_ == 0 and args.seed == 0 and not args.selection_only),
                           selection_only=args.selection_only)
        del P17L_u, PnL_u
        tm["panels"][pname] = {"predict_s": tp, "core_s": round(time.time() - t1 - tp, 1)}
        print(f"[{args.cohort} s{args.seed}] panel {pname} ({len(pm)} markers) predict {tp}s core "
              f"{tm['panels'][pname]['core_s']}s", flush=True)
        res.append(out)
    outdir = Path(args.outdir) if args.outdir else OUT / args.cohort
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"regions_{args.head}_seed{args.seed}{('_' + args.tag) if args.tag else ''}"
    meta = {"cohort": args.cohort, "seed": args.seed, "head": args.head, "u_def": args.u_def,
            "alpha": rc.ALPHA, "budgets": list(rc.BUDGETS), "eval_cap": rc.EVAL_CAP,
            "context": f"cap {args.context_cap}" if args.context_cap else "census (all rows of the region)",
            "L_primary": L, "L_notes": notes, "F": tab.F, "panels": {n: m for n, m in panels},
            "dropped_columns": tab.dropped, "head_info": head_info, "regions": book,
            "n_context_cells": int(len(ctx_rows)), "n_eval_cells": int(len(ev_rows)),
            "n_eval_by_role": {k: int((R["role"][ev_reg] == i).sum()) for i, k in enumerate(("dev", "cal", "test"))},
            "n_unlabelled_context": int((tab.y[ctx_rows] < 0).sum()),
            "xy_finite_frac_test": float(np.isfinite(tab.xy[ctx_rows][R["role"][ctx_reg] == 2]).all(1).mean())
            if len(ctx_rows) else None,
            "timings": tm, "host": os.uname().nodename}
    if hasattr(tab, "xy_source"):
        meta["xy_source"] = tab.xy_source
    if args.timing_only:
        meta["wall_clock_s"] = round(time.time() - t0, 1)
        (outdir / f"{tag}_timing.json").write_text(json.dumps(meta, indent=1, default=str))
        print(json.dumps(tm), flush=True)
        return
    if args.selection_only:
        nm = onto.names
        bits = lambda v: [nm[k] for k in range(len(nm)) if (int(v) >> k) & 1]  # noqa: E731
        role = R["role"]
        sel_out = {}
        for (pname, _), r in zip(panels, res):
            d = {"voi_run": r["info"]["voi_run"], "dev_fold_patients": r["info"]["dev_fold_patients"],
                 "selected": {}, "strata": {}}
            for ri, rule in enumerate(rc.RULES):
                if not r["ran"][ri]:
                    continue
                d["selected"][rule] = {f"{b_:.2f}": {k: int(r["sel"][ri, bi][role == i].sum())
                                                    for i, k in enumerate(("dev", "cal", "test"))}
                                       for bi, b_ in enumerate(rc.BUDGETS)}
                d["strata"][rule] = {f"{b_:.2f}": {st: dict(zip(rc.SINFO[:-1], [int(x) if np.isfinite(x) else None
                                                                                for x in r["sinfo"][ri, bi, k][:-1]]))
                                                  for k, st in enumerate(("L_unsel", "F_sel"))}
                                     for bi, b_ in enumerate(rc.BUDGETS)}
                d["strata"][rule]["supported_b020"] = {st: bits(r["sbits"][ri, rc.BUDGETS.index(0.2), k])
                                                      for k, st in enumerate(("L_unsel", "F_sel"))}
            sel_out[pname] = d
        meta["selection"] = sel_out
        meta["wall_clock_s"] = round(time.time() - t0, 1)
        (outdir / f"{tag}_selection.json").write_text(json.dumps(meta, indent=1, default=str))
        p0 = sel_out["L6"]["selected"]
        print(f"[{args.cohort} s{args.seed}] TEST regions {int((role == 2).sum())}; selected TEST regions at b=0.20 "
              f"(L6): " + ", ".join(f"{k} {v['0.20']['test']}" for k, v in p0.items()), flush=True)
        return
    stack = lambda k: np.stack([r[k] for r in res])  # noqa: E731
    test_r = res[0]["test_regions"]
    arrays = dict(rstat=stack("rstat"), tstat=stack("tstat"), selT=stack("selT"), vran=stack("vran"),
                  taus=stack("taus"), scores=stack("scores"), sinfo=stack("sinfo"), sbits=stack("sbits"),
                  gain=stack("gain"), forced_err_L=stack("forced_err_L"), ran=stack("ran"),
                  region_role=R["role"], region_N=R["N_r"], region_area=R["area"], region_key=R["region_key"],
                  region_n_ctx=np.bincount(ctx_reg, minlength=len(R["role"])),
                  region_n_eval=np.bincount(ev_reg, minlength=len(R["role"])), test_regions=test_r,
                  test_region_pslot=np.searchsorted(res[0]["test_patients"], R["patient"][test_r]),
                  test_patients=R["slot_keys"][res[0]["test_patients"]],
                  panels=np.array([n for n, _ in panels]), variants=np.array(rc.VARIANTS), rules=np.array(rc.RULES),
                  rst=np.array(rc.RST), tst=np.array(rc.TST), sinfo_names=np.array(rc.SINFO),
                  budgets=np.array(rc.BUDGETS), nodes=np.array(onto.names))
    np.savez_compressed(outdir / f"{tag}.npz", **arrays)
    meta["panel_info"] = {n: r["info"] for (n, _), r in zip(panels, res)}
    nm = onto.names
    bits = lambda v: [nm[k] for k in range(len(nm)) if (int(v) >> k) & 1]  # noqa: E731
    bp = rc.BUDGETS.index(rc.B_PRIMARY)
    meta["supported_b020"] = {n: {rule: {st: bits(r["sbits"][ri, bp, k]) for k, st in enumerate(("L_unsel", "F_sel"))}
                                  for ri, rule in enumerate(rc.RULES) if r["ran"][ri]}
                              for (n, _), r in zip(panels, res)}
    meta["realised_selected_test_regions"] = {
        n: {v: [int(x) for x in r["selT"][rc.VI[v]].sum(axis=1)] for v in rc.VARIANTS if r["vran"][rc.VI[v]].all()}
        for (n, _), r in zip(panels, res)}
    meta["wall_clock_s"] = round(time.time() - t0, 1)
    (outdir / f"{tag}_meta.json").write_text(json.dumps(meta, indent=1, default=lambda o: o.tolist()
                                                        if hasattr(o, "tolist") else str(o)))
    cells = res[0]["cells"]
    if cells:
        T = R["role"][ev_reg] == 2
        rows = ev_rows[T]
        df = pd.DataFrame({"region": R["region_key"][ev_reg[T]], "x": tab.xy[rows, 0], "y": tab.xy[rows, 1],
                           "label": np.array(nm)[cells["y"]]})
        for k in ("CLdev", "uL", "uL_singleton", "CL_pooled", "CF_pooled", "sel_uncert", "sel_voi", "sel_density",
                  "final_uncert_cond"):
            if k in cells:
                df[k] = cells[k]
        df.attrs["set_bit_order"] = ",".join(nm)
        df.to_parquet(outdir / f"{tag}_cells_L6.parquet", index=False)
    print(f"[{args.cohort} s{args.seed}] done {meta['wall_clock_s']}s -> {outdir / tag}.npz", flush=True)


if __name__ == "__main__":
    main()
