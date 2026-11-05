#!/usr/bin/env python3
"""Write $ADMIT_RUNS/round2/MANIFEST.tsv (file, what it contains, status) for the figure scripts."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RUNS  # noqa: E402

R2 = RUNS / "round2"
DESC = [
    ("spec_round2.json", "pre-specification of round 2 (ADMIT, M1, E1/E2/E4/M6), committed before results"),
    ("spec_round2_addendum.json", "addendum 1: correct-leaf-resolution objective/metric (before results)"),
    ("spec_round2_addendum2.json", "addendum 2: lineage-start greedy curve for M6"),
    ("orion_exchangeable.npz", "Orion round0 random halves of crc_test: acc[panel,arm,alpha,patient,node,stat] sufficient statistics"),
    ("orion_lopo7.npz", "Orion round0 leave-one-patient-out over 7 patients: same accumulator"),
    ("orion_lopo13.npz", "Orion round0 leave-one-patient-out over 13 patients (incl. 6 rebuilt patients): same accumulator"),
    ("orion_*_meta.json", "per-protocol metadata: patients, panel aliases/families, ADMIT/M1 strata reports"),
    ("cohort_schurch_logistic_s*.npz", "Schurch 20 patient splits (E4), logistic head: additive accumulator + per-split equal-patient coverage"),
    ("cohort_schurch_xgb_s*.npz", "Schurch 20 patient splits (E4), XGBoost head: same"),
    ("cohort_*_meta.json", "per-run metadata: splits (train/cal/test patients), label counts, M1 groups of the first split"),
    ("panel_design_schurch_seed*.json", "M6 Schurch (logistic, E4 seeds 0-4): marker value table, bank predicted-vs-realized, greedy from empty panel"),
    ("greedy_lineage_schurch_seed*.json", "M6 Schurch greedy from {CD45,PanCK,CD31,aSMA}"),
    ("spec_round2_addendum3.json", "addendum 3: type-balanced M6 objective over T, budgets k=8/12/16/24 (before running)"),
    ("greedy_tb_*_schurch_seed*.json", "M6 Schurch type-balanced greedy (empty and lineage starts), budgets, pred vs real"),
    ("greedy_tb_*_orion.json", "M6 Orion type-balanced greedy (empty and lineage starts), budgets, pred vs real"),
    ("panel_design_orion.json", "M6 Orion (cal pool 7 patients, 6 held-out patients): marker value, bank, greedy from empty"),
    ("greedy_lineage_orion.json", "M6 Orion greedy from the lineage panel"),
    ("external/external_decomposition_*.json", "external decomposition (i)-(iv) with ADMIT and ADMIT+M1 (n rule, n50), 12 cohorts"),
    ("target_curve/target_curve_*.json", "target-calibration curves on external cohorts: n labelled target cells 50..2000 x 10 seeds"),
    ("summary_round2.json", "FIGURE INPUT: all reduced numbers (E1 per-stratum/per-type coverage with patient-cluster CIs, E2 resolution, alpha curves, E4 split distributions, M6 curves/validation, external)"),
    ("report_tables_round2.md", "generated tables (from summary_round2.json)"),
    ("tables/*.tsv", "generated TSV tables (same numbers)"),
]


def status(pattern: str) -> str:
    files = sorted(R2.glob(pattern))
    if not files:
        return "missing"
    logs = R2 / "logs"
    stem = pattern.split("/")[-1].replace("*", "").replace(".npz", "").replace(".json", "")
    for lf in logs.glob("*.log"):
        txt = lf.read_text(errors="ignore")
        if "Traceback" in txt and stem and stem.strip("_") in lf.stem:
            return "error (see logs/" + lf.name + ")"
    return "final"


def main(partial=()):
    rows = ["file\tcontents\tstatus\tn_files\tupdated"]
    for pat, desc in DESC:
        st = status(pat)
        if any(p in pat for p in partial) and st == "final":
            st = "partial"
        n = len(list(R2.glob(pat)))
        rows.append(f"{pat}\t{desc}\t{st}\t{n}\t{datetime.now().isoformat(timespec='seconds')}")
    (R2 / "MANIFEST.tsv").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]))
