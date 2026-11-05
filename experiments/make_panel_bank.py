#!/usr/bin/env python3
"""Fix the paired-deletion panel bank (and the run protocol) BEFORE any deletion result.

Writes $ADMIT_RUNS/panel_bank.json. Only marker names and cohort panels are read; no
model, no label, no score.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LOCAL_COHORTS, RUNS, load_cohort, orion_panel, write_json  # noqa: E402

LINEAGE_GROUPS = [["CD8"], ["CD4"], ["FOXP3"], ["CD20"], ["CD68"], ["CD163"],
                  ["PanCK", "Ecadherin"], ["CD31"], ["aSMA"], ["CD45"]]
# Pre-declared extension (not in the task spec): under the repo's isoform rule CD45 is
# satisfied by CD45RO, which Orion measures, so deleting {CD45} alone cannot remove the
# Immune lineage from A(P). {CD45, CD45RO} is the deletion that does.
EXTRA_GROUPS = [["CD45", "CD45RO"]]
RETAIN = [0.75, 0.50, 0.25]
SEEDS = [0, 1, 2, 3, 4]


def main():
    orion = orion_panel()
    bank = []

    def add(name, family, keep, note=""):
        keep = [m for m in orion if m in set(keep)]
        bank.append({"name": name, "family": family, "markers": keep,
                     "deleted": [m for m in orion if m not in keep], "n_markers": len(keep),
                     **({"note": note} if note else {})})

    add("full", "full", orion)
    for m in orion:
        add(f"del_{m}", "single", [x for x in orion if x != m])
    for g in LINEAGE_GROUPS:
        gg = [m for m in g if m in orion]
        add("grp_" + "+".join(gg), "lineage_group", [x for x in orion if x not in gg])
    for g in EXTRA_GROUPS:
        gg = [m for m in g if m in orion]
        add("grp_" + "+".join(gg), "lineage_group_extension", [x for x in orion if x not in gg],
            note="pre-declared extension; CD45 isoform rule")
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(len(orion))
        for f in RETAIN:
            k = int(round(f * len(orion)))
            add(f"rand{int(f * 100)}_s{seed}", "random_retention", [orion[i] for i in sorted(perm[:k])],
                note=f"keep {k}/{len(orion)}; nested within seed (first k of one permutation)")
    cohort_panels = {}
    for c in LOCAL_COHORTS:
        pk = load_cohort(c)["panel"]
        cohort_panels[c] = pk
        add(f"orion_cap_{c}", "external_intersection", [m for m in orion if m in set(pk)])

    payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "orion_panel": orion,
        "orion_panel_source": "columns of X_arc with any nonzero value in crc_train_full, crc_cal and crc_test",
        "external_cohort_panels": cohort_panels,
        "n_panels": len(bank),
        "panels": bank,
        "protocol": {
            "alpha": 0.10, "n_min": 10,
            "quantile": "k-th smallest score, k = ceil((n+1)(1-alpha)); +inf if k > n",
            "primary": "calibrate on crc_cal (2 Orion patients) masked to P, evaluate on crc_test (5 Orion patients) masked to P",
            "exchangeable_control": ("calibrate on a random half of crc_test (seed 0) masked to P, evaluate on the "
                                     "other half; added after task 2 showed crc_cal->crc_test is not exchangeable "
                                     "for round0 even on the full panel; reported separately, never mixed"),
            "backbones": {"round0": "primary (retrained round0_xgb_v6_leakfree, X_arc)",
                          "v6crc": "secondary (v6.0_crconly, X_gmm 160-D with presence bits)"},
            "metrics": ["exact coverage Pr[pi_P(y) in C] (identifiable cells) overall and per stratum",
                        "leaf coverage Pr[y in C] on cells with y in A(P)", "ancestor-credited coverage",
                        "mean |C|", "singleton rate", "leaf-resolved rate", "abstention (empty) rate",
                        "inadmissible set mass", "Wilson cell-level CIs (not patient-clustered)"],
        },
    }
    out = RUNS / "panel_bank.json"
    if out.exists():
        raise SystemExit(f"{out} exists; the bank is fixed. Delete it deliberately to rebuild.")
    write_json(out, payload)
    print(f"wrote {out}: {len(bank)} panels")
    for p in bank:
        print(f"  {p['name']:28s} {p['family']:24s} n={p['n_markers']:2d} deleted={p['deleted']}")


if __name__ == "__main__":
    main()
