#!/usr/bin/env python3
"""ADMIT on synthetic data, end to end (CPU, well under a minute).

1. Simulate 30 patients on the 17-node hierarchy; split patients 15 train / 8 calibration / 7 test.
2. Train a classifier once on the full panel (logistic regression on z-scored intensities).
3. Delete CD8: the deployment panel P is every marker except CD8.
4. ADMIT-M1: fix the supported groups from training labels only (revise labels to pi_P(y), merge groups whose
   expected calibration count is below n_min = ceil((1-alpha)/alpha) into their parent), then mask calibration
   cells to P, rescore with the frozen classifier and fit one threshold per group.
5. Report exact coverage of the revised label (overall and per group), set size, and what a forced classifier and
   unfiltered LAC return on the same cells.

    python examples/toy_demo.py [--delete CD8] [--alpha 0.1] [--seed 0] [--cal-cells 200]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from admit import NODE17, AdmitM1Frozen, Ontology, frozen_support_map, lac_flat, mask_arc, nmin_rule  # noqa: E402
from admit.synthetic import make_toy_cohort, predict17, zscore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", default="CD8", help="comma-separated markers removed from the deployment panel")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cal-cells", type=int, default=0, help="use only this many labelled calibration cells")
    args = ap.parse_args()
    t0 = time.time()
    from sklearn.linear_model import LogisticRegression

    d = make_toy_cohort(n_patients=30, cells_per_patient=1500, seed=args.seed)
    onto = Ontology.from_repo(NODE17)
    markers = d["markers"]
    perm = np.random.default_rng(args.seed).permutation(30)
    role = {"train": perm[:15], "cal": perm[15:23], "test": perm[23:]}
    sel = {r: np.isin(d["patient"], p) for r, p in role.items()}
    Xtr, Xcal, Xte = zscore(d["x"][sel["train"]], d["x"][sel["cal"]], d["x"][sel["test"]])
    ytr, ycal, yte = (d["y"][sel[r]] for r in ("train", "cal", "test"))
    if args.cal_cells:
        keep = np.sort(np.random.default_rng(args.seed).choice(len(ycal), args.cal_cells, replace=False))
        Xcal, ycal = Xcal[keep], ycal[keep]

    # 1) classifier trained once, on the full panel, on training patients only
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=2000).fit(Xtr, ytr)

    # 2) deployment panel P and its admissible types A(P)
    deleted = [m.strip() for m in args.delete.split(",") if m.strip()]
    panel = [m for m in markers if m not in deleted]
    adm = onto.admissible(panel)
    proj = onto.projection_map(adm)
    print(f"panel P: {len(panel)} of {len(markers)} markers (deleted: {', '.join(deleted)})")
    print("not admissible under P:", [n for n, a in zip(NODE17, adm) if not a])
    for leaf in ("Cytotoxic_T_Cell", "Helper_T_Cell"):
        r = proj[NODE17.index(leaf)]
        print(f"revised label pi_P({leaf}) = {NODE17[r] if r >= 0 else 'unidentifiable'}")

    # 3) rescore calibration and test cells masked to P with the frozen classifier; node scores = subtree mass
    Pcal = predict17(clf, mask_arc(Xcal, markers, panel))
    Pte = predict17(clf, mask_arc(Xte, markers, panel))
    Ncal, Nte = onto.aggregate(Pcal), onto.aggregate(Pte)

    # 4) ADMIT-M1: supported groups frozen from training labels, then one threshold per group
    n_min = nmin_rule(args.alpha)
    gmap, S, _ = frozen_support_map(onto, adm, ytr, n_cal=len(ycal), n_min=n_min)
    m = AdmitM1Frozen(onto, args.alpha, gmap, S).fit(Ncal, ycal)
    sets = m.predict(Nte)
    tgt = m.target(yte)                       # coarsened revised label g(y) (-1 = unidentifiable)
    ident = tgt >= 0
    hit = np.zeros(len(yte), bool)
    hit[ident] = sets[np.where(ident)[0], tgt[ident]]
    size = sets.sum(1)
    print(f"\nADMIT-M1 (alpha {args.alpha}, n_min {n_min}): {int(m.S.sum())} calibrated groups")
    print(f"  {'group':22s} {'n_cal':>6s} {'n_test':>6s} {'coverage':>9s}")
    for c in np.where(m.S)[0]:
        s = tgt == c
        if s.any():
            print(f"  {NODE17[c]:22s} {m.n_c[c]:6d} {int(s.sum()):6d} {hit[s].mean():9.3f}")
    merged = [NODE17[c] for c in np.where(adm & ~m.S)[0] if (proj[ytr] == c).any()]   # had cells, too few
    if merged:
        print(f"  merged into their parent (expected fewer than {n_min} calibration cells): {', '.join(merged)}")
    print(f"  overall exact coverage {hit[ident].mean():.3f} (target {1 - args.alpha:.2f}); "
          f"mean set size {size.mean():.2f}; empty sets {np.mean(size == 0):.3f}")
    print(f"  returned types outside A(P): {int((sets & ~adm[None, :]).sum())}")
    ex = np.where(yte == NODE17.index("Cytotoxic_T_Cell"))[0][:3]
    for i in ex:
        print(f"  a cytotoxic T cell -> {{{', '.join(NODE17[j] for j in np.where(sets[i])[0])}}}")

    # 5) the same cells under a forced classifier and unfiltered LAC
    forced = Pte.argmax(1)
    lac = lac_flat(Pcal, ycal, Pte, args.alpha)
    print(f"\nforced classifier: {np.mean(~adm[forced]):.3f} of calls name a type P cannot support; "
          f"{np.mean(forced == NODE17.index('Cytotoxic_T_Cell')):.4f} of cells called cytotoxic T")
    print(f"unfiltered LAC: {(lac & ~adm[None, :]).sum() / lac.sum():.3f} of set entries unsupported")
    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
