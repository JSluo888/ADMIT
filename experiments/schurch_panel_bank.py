#!/usr/bin/env python3
"""Fix the Schurch (task 5b) panel bank before any Schurch result. Reads only marker names."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, write_json  # noqa: E402
from schurch_common import load_schurch  # noqa: E402

R_MARKERS = ["CD45", "CD20", "CD3e", "CD8", "CD4", "FOXP3", "CD68", "CD163", "CD11c", "HLADR", "CD31", "PanCK",
             "aSMA", "CD56", "CD15", "CD138"]          # every marker some R(c) uses (17-node ontology)
ISOFORMS = ["CD45RO", "CD45RA"]                      # satisfy CD45 under the isoform rule
GROUPS = [["CD8"], ["CD4"], ["FOXP3"], ["CD20"], ["CD68"], ["CD163"], ["PanCK", "Ecadherin"], ["CD31"],
          ["aSMA"], ["CD45"]]
EXT_GROUPS = [["CD45", "CD45RO", "CD45RA"], ["CD3e", "CD4", "CD8", "FOXP3"]]


def main():
    d = load_schurch()
    full = d["markers"]
    bank = []

    def add(name, family, keep, note=""):
        keep = [m for m in full if m in set(keep)]
        bank.append({"name": name, "family": family, "markers": keep,
                     "deleted": [m for m in full if m not in keep], "n_markers": len(keep),
                     **({"note": note} if note else {})})

    add("full", "full", full)
    for m in R_MARKERS + ISOFORMS:
        add(f"del_{m}", "single", [x for x in full if x != m])
    for g in GROUPS:
        gg = [m for m in g if m in full]
        add("grp_" + "+".join(gg), "lineage_group", [x for x in full if x not in gg],
            note=("Ecadherin not measured in Schurch" if "Ecadherin" in g else ""))
    for g in EXT_GROUPS:
        add("grp_" + "+".join(g), "lineage_group_extension", [x for x in full if x not in g],
            note="all CD45 isoforms" if "CD45RO" in g else "pilot T-lineage deletion (CD3/CD4/CD8/FOXP3)")
    for seed in range(5):
        perm = np.random.default_rng(seed).permutation(len(full))
        for f in (0.75, 0.50, 0.25):
            k = int(round(f * len(full)))
            add(f"rand{int(f * 100)}_s{seed}", "random_retention", [full[i] for i in sorted(perm[:k])],
                note=f"keep {k}/{len(full)}; nested within seed")
    rng = np.random.default_rng(701)                 # pilot/run_codex.py panels, for continuity
    drop4 = sorted(rng.choice(len(full), len(full) - 4, replace=False).tolist())
    half = sorted(rng.choice(len(full), len(full) // 2, replace=False).tolist())
    add("pilot_random_drop_4", "pilot", [full[i] for i in drop4])
    add("pilot_random_keep_half", "pilot", [full[i] for i in half])
    out = RUNS / "schurch" / "schurch_panel_bank.json"
    if out.exists():
        raise SystemExit(f"{out} exists; the bank is fixed")
    write_json(out, {"created": datetime.now().isoformat(timespec="seconds"), "full_panel": full,
                     "n_panels": len(bank), "panels": bank,
                     "protocol": {"alpha": 0.10, "splits": "pilot seeds 20260928-30: 15 train / 12 cal / 8 test patients",
                                  "heads": ["logistic (pilot spec)", "xgboost"],
                                  "missing_encoding": "train mean in log1p space = 0 after train-only StandardScaler"}})
    print(f"wrote {out}: {len(bank)} panels")
    for p in bank:
        print(f"  {p['name']:30s} {p['family']:24s} n={p['n_markers']:2d} deleted={p['deleted'][:8]}{'...' if len(p['deleted'])>8 else ''}")


if __name__ == "__main__":
    main()
