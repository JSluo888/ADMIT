#!/usr/bin/env python3
"""Compact view of cohorts/summary.tsv:  show.py <cohort-substring> [panels-substring] [head]"""
import csv
import os
import sys
from pathlib import Path

M = ["admit_m1", "admit", "patch_src_backstop", "lac_node_proj", "lac_filt", "aps_filt"]
A = {"admit_m1": "ADMIT+M1", "admit": "ADMIT", "patch_src_backstop": "fullcal(old)", "lac_node_proj": "margLAC",
     "lac_filt": "LAC+f", "aps_filt": "APS+f"}
c, p = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
h = sys.argv[3] if len(sys.argv) > 3 else None
f = lambda v, n=3: (f"{float(v):.{n}f}" if v not in ("", None) else "-")  # noqa: E731
rows = [r for r in csv.DictReader(open(Path(os.environ.get("ADMIT_RUNS", str(Path(__file__).resolve().parents[2] / "runs"))) / "cohorts" / "summary.tsv"), delimiter="\t")
        if c in r["cohort"] and p in r["panels"] and r["method"] in M and (h is None or r["head"] == h)]
key = lambda r: (r["protocol"], r["head"], r["panels"], M.index(r["method"]))  # noqa: E731
last = None
for r in sorted(rows, key=key):
    g = (r["protocol"], r["head"], r["panels"])
    if g != last:
        ab = lambda n: n.split("_")[0][:10] + "/" + (n.split("_")[1][:9] if n.startswith("mcc") else n.split("_")[-1][:8])  # noqa: E731
        pn = r["panels"]
        if "->" in pn:
            st, blk = pn.split(":", 1)
            s_, t_ = st.split("->")
            pn = f"{ab(s_)} -> {ab(t_)} : {blk}"
        print(f"## {r['cohort']} | {r['protocol']} | {r['head']} | {pn}"
              f" | pats {r['n_patients']} seeds {r['n_seeds']} {r.get('status','')}")
        last = g
    wt = f"{r['worst_type'][:14]}:{f(r['worst_cov'])}" if r["worst_type"] else f"worst(mean) {f(r['worst_cov'])} min {f(r['min_worst_cov'])}"
    print(f"  {A[r['method']]:13s} cov {f(r['exact_cov'])} [{f(r['cov_ci_lo'])},{f(r['cov_ci_hi'])}]  {wt:28s} "
          f"sig<{r['n_sig_below']}/{r['n_types']}  |C| {f(r['size'],2)} depth {f(r['depth'],2)} leaf {f(r['leaf_resolved'])} abst {f(r['abstain'])}")
