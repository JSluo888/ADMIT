#!/usr/bin/env python3
"""Summaries of the region-selection runs. Writes files only and prints paths and counts, never endpoint values.

  python experiments/regions/summarize_regions.py --cohorts cords2024_nsclc_imc danenberg2022_metabric_imc schurch

Per cohort, panel (L6 primary, rand50_* secondary), variant and budget, pooled over seeds (Orion: folds):
- primary utility `corC` = fraction of the fixed universe (TEST evaluation cells with a leaf reference label;
  failures stay in the denominator) whose final set resolves to the reference leaf (engine3 `leafres_correct`);
  `cor1` = the singleton definition (diagnostic); type-balanced (leaves with >= 100 pooled TEST cells)
  and equal-region (mean over TEST regions) versions;
- unresolved fraction (`unresC`, all TEST evaluation cells; secondary), mean |C|, abstention, R(b) on `unresC`
  (only when U(0) - U(1) > 0), forced-error capture, achieved cost (fractions of TEST regions, cells, area);
- coverage on each routed stratum's own target, overall and per (stratum, target type) with >= 100 pooled TEST
  cells, deficit, strata significantly below 0.90 (bootstrap upper bound < 0.90), panel-unidentifiable and
  no-supported-target fractions per stratum.
Intervals: 95% percentile patient-cluster bootstrap, B = 1,000, multinomial weights over unique TEST patients pooled
over seeds (one weight vector shared by every panel, variant, budget and comparison, so all differences are paired).
Comparisons (paired differences A - B): `uncert_cond` - `random_k_uncert` on corC (primary, b = 0.20, key E1_...);
secondary `uncert_cond` - `density_k` (B3b_...); `uncert_pooled` vs `random_k` / `density_k` (B4_...); and the rest
of PAIRS. Evidence rule (``evidence``) over the eligible cohorts (eligibility.py).
Outputs: $ADMIT_RUNS/regions/summary_regions.json, summary_regions.tsv.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import warnings
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import regions_core as rc  # noqa: E402

RUNS = Path(os.environ.get("ADMIT_RUNS", str(HERE.parents[1] / "runs")))
OUT = Path(os.environ.get("ADMIT_REGIONS_OUT", str(RUNS / "regions")))
B_BOOT = 1000
BOOT_SEED = 20271030
MIN_STRATUM = 100
LEVEL = 1 - rc.ALPHA
PAIRS = [  # (name, A, B, metrics); corC = primary utility, cor1 = singleton diagnostic
    ("E1_uncert_cond_vs_random_k", "uncert_cond", "random_k_uncert", ("corC", "cor1", "unresC")),
    ("B3b_uncert_cond_vs_density_k", "uncert_cond", "density_k", ("corC", "cor1", "unresC")),
    ("B4_uncert_pooled_vs_random_k", "uncert_pooled", "random_k_uncert", ("corC", "cor1", "unresC")),
    ("B4_uncert_pooled_vs_density_k", "uncert_pooled", "density_k", ("corC", "cor1", "unresC")),
    ("voi_cond_vs_random_k", "voi_cond", "random_k_voi", ("corC", "unresC")),
    ("density_cond_vs_random_k", "density_cond", "random_k_density", ("corC", "unresC")),
    ("uncert_cond_vs_random_bern", "uncert_cond", "random_bern", ("corC", "unresC")),
    ("voi_cond_vs_uncert_cond", "voi_cond", "uncert_cond", ("corC", "unresC")),
    ("oracle_gap_uncert", "oracle_k_uncert", "uncert_cond", ("corC",)),
    ("oracle_gap_voi", "oracle_k_voi", "voi_cond", ("corC",)),
    ("cond_vs_pooled_uncert", "uncert_cond", "uncert_pooled",
     ("corC", "mean_deficit", "mean_deficit_L_unsel", "mean_deficit_F_sel", "cov_L_unsel", "cov_F_sel")),
    ("cond_vs_pooled_voi", "voi_cond", "voi_pooled",
     ("corC", "mean_deficit", "mean_deficit_L_unsel", "mean_deficit_F_sel", "cov_L_unsel", "cov_F_sel")),
]
RATIO = {"cor1": ("cor1", "n_leafU"), "corC": ("corC", "n_leafU"), "unres1": ("unres1", "n_eval"),
         "unresC": ("unresC", "n_eval"), "size": ("size", "n_eval"), "abst": ("abst", "n_eval"),
         "cov": ("cov_hit", "cov_n"), "unident": ("unident", "n_eval"), "nosup": ("nosup", "n_eval")}


def ci(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [None, None]


def fl(x):
    return None if x is None or not np.isfinite(x) else float(x)


class Cohort:
    """Per-seed arrays mapped onto one global TEST patient index."""

    def __init__(self, cohort: str, head: str, rundir: Path, tag: str = ""):
        sfx = f"_{tag}" if tag else ""
        files = sorted(f for f in rundir.glob(f"regions_{head}_seed*.npz") if re.fullmatch(
            rf"regions_{head}_seed\d+{sfx}\.npz", f.name))
        if not files:
            raise FileNotFoundError(f"{rundir}: no regions_{head}_seed*.npz")
        self.cohort, self.files = cohort, files
        self.z = [np.load(f, allow_pickle=True) for f in files]
        self.meta = [json.loads(Path(str(f)[:-4] + "_meta.json").read_text()) for f in files]
        z0 = self.z[0]
        self.panels, self.variants = list(z0["panels"]), list(z0["variants"])
        self.budgets, self.rst, self.tst = list(z0["budgets"]), list(z0["rst"]), list(z0["tst"])
        self.nodes = list(z0["nodes"])
        for z in self.z:
            assert list(z["panels"]) == self.panels and list(z["variants"]) == self.variants
        self.patients = np.unique(np.concatenate([z["test_patients"] for z in self.z]))
        self.nP, self.nS = len(self.patients), len(self.z)
        self.gp = [np.searchsorted(self.patients, z["test_patients"])[z["test_region_pslot"]] for z in self.z]
        self.gslot = [np.searchsorted(self.patients, z["test_patients"]) for z in self.z]

    def panel_arrays(self, p: int):
        """-> dict of per-seed patient-level sums for panel p."""
        V, nb, K = len(self.variants), len(self.budgets), len(self.nodes)
        nR, nT = len(self.rst), len(self.tst)
        RS = np.zeros((self.nS, self.nP, V, nb, nR))
        RSst = np.zeros((self.nS, self.nP, V, nb, 2, nR))
        TS = np.zeros((self.nS, self.nP, V, nb, 2, K, nT), dtype=np.float32)
        cost = np.zeros((self.nS, self.nP, V, nb, 3))          # selected regions, cells, area
        tot = np.zeros((self.nS, self.nP, 3))
        fe = np.zeros((self.nS, self.nP, V, nb, 2))            # forced errors in selected regions, all
        ran = np.zeros((self.nS, V, nb), dtype=bool)
        regs = []
        for s, z in enumerate(self.z):
            tr = z["test_regions"]
            gp = self.gp[s]
            r = np.nan_to_num(z["rstat"][p].astype(np.float64))         # (V, nb, nTR, nR)
            sel = z["selT"][p]                                          # (V, nb, nTR)
            ran[s] = z["vran"][p]
            Nr, area = z["region_N"][tr].astype(np.float64), z["region_area"][tr].astype(np.float64)
            for v in range(V):
                for b in range(nb):
                    np.add.at(RS[s, :, v, b], gp, r[v, b])
                    for st in (0, 1):
                        m = sel[v, b] == bool(st)
                        np.add.at(RSst[s, :, v, b, st], gp[m], r[v, b][m])
                    w = sel[v, b].astype(np.float64)
                    np.add.at(cost[s, :, v, b], gp, np.stack([w, w * Nr, w * area], 1))
                    f = np.nan_to_num(z["forced_err_L"][p])
                    np.add.at(fe[s, :, v, b], gp, np.stack([w * f, f], 1))
            np.add.at(tot[s], gp, np.stack([np.ones(len(tr)), Nr, area], 1))
            T = z["tstat"][p]                                           # (V, nb, 2, nTP, K, nT)
            TS[s][self.gslot[s]] += np.moveaxis(T, 3, 0)
            regs.append((gp, r[..., self.rst.index("cor1")], r[..., self.rst.index("corC")],
                         r[..., self.rst.index("n_leafU")][0, 0]))
        return dict(RS=RS, RSst=RSst, TS=TS, cost=cost, tot=tot, fe=fe, ran=ran, regs=regs)


def summarise_panel(C: Cohort, p: int, W: np.ndarray):
    A = C.panel_arrays(p)
    V, nb = len(C.variants), len(C.budgets)
    ri = {k: C.rst.index(k) for k in C.rst}
    ti = {k: C.tst.index(k) for k in C.tst}
    out, rows, reps = {}, [], {}
    Wf = np.vstack([np.ones((1, C.nP)), W])                    # row 0 = point estimate
    for v, vn in enumerate(C.variants):
        for b, bv in enumerate(C.budgets):
            seeds = A["ran"][:, v, b]
            if not seeds.any():
                continue
            RS = A["RS"][seeds, :, v, b].sum(0)                     # (P, nR)
            RSst = A["RSst"][seeds, :, v, b].sum(0)                 # (P, 2, nR)
            TS = A["TS"][seeds, :, v, b].sum(0).astype(np.float64)  # (P, 2, K, nT)
            wR = Wf @ RS                                            # (1+B, nR)
            d = {}
            for m, (num, den) in RATIO.items():
                with np.errstate(invalid="ignore", divide="ignore"):
                    d[m] = wR[:, ri[num]] / wR[:, ri[den]]
            wst = np.einsum("bp,psr->bsr", Wf, RSst)
            for st, sn in enumerate(("L_unsel", "F_sel")):
                with np.errstate(invalid="ignore", divide="ignore"):
                    d[f"cov_{sn}"] = wst[:, st, ri["cov_hit"]] / wst[:, st, ri["cov_n"]]
                    d[f"unident_{sn}"] = wst[:, st, ri["unident"]] / wst[:, st, ri["n_eval"]]
                    d[f"nosup_{sn}"] = wst[:, st, ri["nosup"]] / wst[:, st, ri["n_eval"]]
                    d[f"cor1_{sn}"] = wst[:, st, ri["cor1"]] / wst[:, st, ri["n_leafU"]]
                    d[f"corC_{sn}"] = wst[:, st, ri["corC"]] / wst[:, st, ri["n_leafU"]]
                    d[f"cells_{sn}"] = wst[:, st, ri["n_eval"]]
            cst = Wf @ A["cost"][seeds, :, v, b].sum(0)
            tt = Wf @ A["tot"][seeds].sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                d["frac_regions"], d["frac_cells"], d["frac_area"] = (cst[:, k] / tt[:, k] for k in range(3))
                f = Wf @ A["fe"][seeds, :, v, b].sum(0)
                d["forced_error_capture"] = f[:, 0] / f[:, 1]
            # type-balanced cor1 (label stats summed over strata; leaves with >= 100 pooled TEST cells)
            lab_n = TS[:, :, :, ti["l_n"]].sum(1)                   # (P, K)
            keep_t = (lab_n.sum(0) >= MIN_STRATUM) & C_leaf(C)
            for nm_, col in (("corC", "l_corC"), ("cor1", "l_cor1")):
                lab_c = TS[:, :, :, ti[col]].sum(1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    d[f"{nm_}_type_balanced"] = (np.nanmean((Wf @ lab_c)[:, keep_t] / (Wf @ lab_n)[:, keep_t], axis=1)
                                                 if keep_t.any() else np.full(len(Wf), np.nan))
            # equal-region cor1 (mean over TEST regions with >= 1 leaf-universe cell)
            for k_, nm_ in ((2, "corC"), (1, "cor1")):
                num = np.zeros(len(Wf))
                den = np.zeros(len(Wf))
                for s in np.flatnonzero(seeds):
                    reg = A["regs"][s]
                    gp, cc, nl = reg[0], reg[k_], reg[3]
                    ok = nl > 0
                    num += Wf[:, gp[ok]] @ (cc[v, b][ok] / nl[ok])
                    den += Wf[:, gp[ok]].sum(1)
                d[f"{nm_}_equal_region"] = num / den
            # validity per (stratum, target type) on the routed stratum's own target
            tn, th = TS[:, :, :, ti["t_n"]], TS[:, :, :, ti["t_hit"]]
            keep = tn.sum(0) >= MIN_STRATUM                         # (2, K)
            strata = {}
            if keep.any():
                wn = np.einsum("bp,psk->bsk", Wf, tn)[:, keep]
                wh = np.einsum("bp,psk->bsk", Wf, th)[:, keep]
                with np.errstate(invalid="ignore", divide="ignore"):
                    cov = wh / wn                                   # (1+B, n_strata)
                deficit = np.maximum(0.0, LEVEL - cov)
                d["mean_deficit"] = np.nanmean(deficit, axis=1)
                up = np.nanpercentile(cov[1:], 97.5, axis=0)
                lo = np.nanpercentile(cov[1:], 2.5, axis=0)
                idx = np.argwhere(keep)
                for j, (st, k) in enumerate(idx):
                    strata[f"{('L_unsel', 'F_sel')[st]}:{C.nodes[k]}"] = {
                        "n": int(tn.sum(0)[st, k]), "cov": fl(cov[0, j]), "ci": [fl(lo[j]), fl(up[j])],
                        "deficit": fl(deficit[0, j]), "sig_below": bool(up[j] < LEVEL)}
                d["n_sig_below"] = np.array([float((up < LEVEL).sum())] * len(Wf))
                d["n_strata"] = np.array([float(keep.sum())] * len(Wf))
                for st, sn in enumerate(("L_unsel", "F_sel")):
                    m_ = idx[:, 0] == st
                    if m_.any():
                        d[f"mean_deficit_{sn}"] = np.nanmean(deficit[:, m_], axis=1)
                        d[f"n_sig_below_{sn}"] = np.array([float((up[m_] < LEVEL).sum())] * len(Wf))
                        d[f"n_strata_{sn}"] = np.array([float(m_.sum())] * len(Wf))
            cell = {"n_seeds": int(seeds.sum()), "strata": strata}
            for m, x in d.items():
                cell[m] = {"value": fl(x[0]), "ci": ci(x[1:]) if not m.startswith(("n_sig_below", "n_strata")) else None}
                rows.append((C.cohort, C.panels[p], vn, bv, "metric", m, fl(x[0]), *ci(x[1:])))
            cell["selects_nothing"] = bool(d["frac_regions"][0] == 0)
            cell["validity_failure_088"] = any(
                (cell.get(f"cov_{sn}", {}).get("ci", [None, None])[1] or 1.0) < 0.88 for sn in ("L_unsel", "F_sel"))
            out.setdefault(vn, {})[f"{bv:.2f}"] = cell
            reps[(vn, b)] = (d, seeds)
    # R(b) (descriptive; only when U(0) - U(1) > 0)
    for vn in out:
        for b, bv in enumerate(C.budgets):
            if (vn, b) in reps and ("ref_L", b) in reps and ("ref_F", b) in reps:
                u0, u1, ub = reps[("ref_L", b)][0]["unresC"], reps[("ref_F", b)][0]["unresC"], reps[(vn, b)][0]["unresC"]
                if u0[0] - u1[0] > 0:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        rb = (u0 - ub) / (u0 - u1)
                    out[vn][f"{bv:.2f}"]["R_b"] = {"value": fl(rb[0]), "ci": ci(rb[1:])}
                    rows.append((C.cohort, C.panels[p], vn, bv, "metric", "R_b", fl(rb[0]), *ci(rb[1:])))
    # paired comparisons (same seeds for both arms; shared bootstrap weights)
    paired = {}
    for name, a, bb, mets in PAIRS:
        for b, bv in enumerate(C.budgets):
            if (a, b) not in reps or (bb, b) not in reps:
                continue
            same = np.array_equal(reps[(a, b)][1], reps[(bb, b)][1])
            for m in mets:
                xa, xb = reps[(a, b)][0].get(m), reps[(bb, b)][0].get(m)
                if xa is None or xb is None:
                    continue
                dd = xa - xb
                lo, hi = ci(dd[1:])
                paired.setdefault(name, {}).setdefault(f"{bv:.2f}", {})[m] = {
                    "A": a, "B": bb, "diff": fl(dd[0]), "ci": [lo, hi], "same_seeds": bool(same),
                    "above0": bool(lo is not None and lo > 0), "below0": bool(hi is not None and hi < 0)}
                rows.append((C.cohort, C.panels[p], f"{a}-{bb}", bv, "paired", m, fl(dd[0]), lo, hi))
    return out, paired, rows


_LEAF = {}


def C_leaf(C: Cohort):
    if C.cohort not in _LEAF:
        sys.path.insert(0, str(HERE.parents[1]))
        from admit.ontology import NODE17
        from admit.panel_matched import Ontology
        onto = Ontology.from_repo(list(NODE17))
        assert onto.names == C.nodes
        _LEAF[C.cohort] = onto.is_leaf.copy()
    return _LEAF[C.cohort]


def evidence(paired_by_cohort, key, eligible, budget="0.20", metric="corC"):
    """Claim if the interval is above 0 in at least 75% of eligible cohorts (rounded up) and below 0 in none."""
    res = {c: paired_by_cohort.get(c, {}).get(key, {}).get(budget, {}).get(metric) for c in eligible}
    missing = [c for c, r in res.items() if r is None]
    res = {c: r for c, r in res.items() if r is not None}
    n = len(eligible)
    up = sorted(c for c, r in res.items() if r["above0"])
    down = sorted(c for c, r in res.items() if r["below0"])
    need = int(np.ceil(0.75 * n))
    return {"eligible": list(eligible), "n_eligible": n, "above0": up, "below0": down, "missing": missing,
            "needed_above0": need, "claim": bool(n > 0 and not missing and len(up) >= need and not down),
            "rule": "interval above 0 in >= ceil(0.75 x eligible) cohorts and below 0 in none (b = 0.20, L6, corC)"}


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", required=True)
    ap.add_argument("--head", default="xgb")
    ap.add_argument("--rundir", default=str(OUT))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tag", default="", help="file tag (e.g. smoke); default = the untagged full-run files")
    ap.add_argument("--eligible", nargs="*", default=None, help="eligible cohorts (eligibility.py; default: all --cohorts)")
    args = ap.parse_args()
    S = {"B": B_BOOT, "boot_seed": BOOT_SEED, "min_stratum": MIN_STRATUM, "cohorts": {}}
    rows, paired_primary = [], {}
    for c in args.cohorts:
        C = Cohort(c, args.head, Path(args.rundir) / c, args.tag)
        W = np.random.default_rng([BOOT_SEED]).multinomial(C.nP, np.full(C.nP, 1.0 / C.nP), size=B_BOOT).astype(float)
        S["cohorts"][c] = {"n_seeds": C.nS, "n_test_patients": int(C.nP), "panels": {}}
        for p, pn in enumerate(C.panels):
            o, pr, r = summarise_panel(C, p, W)
            S["cohorts"][c]["panels"][pn] = {"variants": o, "paired": pr}
            rows += r
            if pn == "L6":
                paired_primary[c] = pr
        print(f"[summarize] {c}: {C.nS} seeds, {C.nP} unique TEST patients, {len(C.panels)} panels", flush=True)
    elig = args.eligible if args.eligible is not None else args.cohorts
    S["eligible_B2"] = elig
    S["evidence_B3"] = {"E1_uncert_cond_vs_random_k": evidence(paired_primary, "E1_uncert_cond_vs_random_k", elig),
                        "secondary_uncert_cond_vs_density_k": evidence(paired_primary, "B3b_uncert_cond_vs_density_k",
                                                                       elig)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary_regions.json").write_text(json.dumps(S, indent=1))
    with open(out / "summary_regions.tsv", "w") as fh:
        fh.write("cohort\tpanel\tvariant\tbudget\tkind\tmetric\tvalue\tci_lo\tci_hi\n")
        for r in rows:
            fh.write("\t".join("" if x is None else (f"{x:.6g}" if isinstance(x, float) else str(x)) for x in r) + "\n")
    print(f"[summarize] wrote {out / 'summary_regions.json'} and {out / 'summary_regions.tsv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
