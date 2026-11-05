#!/usr/bin/env python3
"""Headline lines from cohorts/summary.tsv (alpha 0.10): within = all_deletions (+ full); pairs = transfer, target_full."""
import csv
import os
from collections import defaultdict
from pathlib import Path

S = Path(os.environ.get("ADMIT_RUNS", str(Path(__file__).resolve().parents[2] / "runs"))) / "cohorts" / "summary.tsv"
M = [("admit_m1", "M1"), ("admit", "ADMIT"), ("patch_src_backstop", "old"), ("lac_node_proj", "mLAC"), ("lac_filt", "LAC+f"),
     ("aps_filt", "APS+f")]
rows = list(csv.DictReader(open(S), delimiter="\t"))
g = defaultdict(dict)
for r in rows:
    if r["protocol"] == "target_calibration_curve":
        continue
    blk = r["panels"].split(":")[-1] if ":" in r["panels"] else r["panels"]
    if blk not in ("all_deletions", "full", "transfer", "target_full", "within_source_masked"):
        continue
    dirn = r["panels"].split(":")[0] if ":" in r["panels"] else ""
    g[(r["cohort"], r["head"], dirn, blk)][r["method"]] = r
f = lambda v: f"{float(v):.3f}" if v not in ("", None) else "-"  # noqa: E731
for k in sorted(g):
    c, h, dirn, blk = k
    d = g[k]
    if "admit_m1" not in d:
        continue
    a = d["admit_m1"]
    parts = []
    for m, lab in M:
        if m in d:
            r = d[m]
            w = r["worst_cov"] if blk != "all_deletions" else r["min_worst_cov"]
            parts.append(f"{lab} {f(r['exact_cov'])} w{f(w)}")
    dd = (" " + dirn.replace("_", "")[:40]) if dirn else ""
    print(f"{c[:28]:28s} {h:3s} {blk[:13]:13s}{dd} | CI[{f(a['cov_ci_lo'])},{f(a['cov_ci_hi'])}] |C|M1 {float(a['size']):.2f} "
          f"|C|AD {float(d['admit']['size']):.2f} leafM1 {f(a['leaf_resolved'])} sig {a['n_sig_below']}/{a['n_types']} | "
          + "; ".join(parts) + f" | seeds {a['n_seeds']} {a.get('status','')}")
