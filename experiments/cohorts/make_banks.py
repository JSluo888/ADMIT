#!/usr/bin/env python3
"""Fix the deletion panel bank of a cohort from its markers.tsv alone (no cells, no labels, no results).

Round-1 Schurch-bank logic (experiments/round2/cohort.make_bank): full panel; single deletions of the R(c) lineage
markers and CD45 isoforms present; lineage groups; random retention 75/50/25 % with seeds 0-4. Written to
$ADMIT_RUNS/cohorts/<cohort>/panel_bank.json (refuses to overwrite) and copied to
configs/panel_banks/cohorts/<cohort>.json (the committed banks used in the paper live there).

  python experiments/cohorts/make_banks.py cords2024_nsclc_imc [more cohorts]
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import std  # noqa: E402
from cohort import make_bank  # noqa: E402

REPO_BANKS = Path(__file__).resolve().parents[2] / "configs" / "panel_banks" / "cohorts"

for name in sys.argv[1:]:
    _, names, dropped = std.feature_table(std.cohort_dir(name), std.registry()[name].get("synonyms"))
    out = std.OUT / name / "panel_bank.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    existed = out.exists()
    bank = make_bank(names, out)
    if bank["full_panel"] != names:
        sys.exit(f"{name}: existing bank's full panel differs from the current markers.tsv features")
    bank["dropped_nonprotein_columns"] = dropped
    if not existed:
        out.write_text(json.dumps(bank, indent=1))
    REPO_BANKS.mkdir(parents=True, exist_ok=True)
    shutil.copy(out, REPO_BANKS / f"{name}.json")
    fam = {}
    for p in bank["panels"]:
        fam[p["family"]] = fam.get(p["family"], 0) + 1
    print(name, "existing" if existed else "new", len(names), "markers;", bank["n_panels"], "panels", fam,
          "dropped:", dropped)
