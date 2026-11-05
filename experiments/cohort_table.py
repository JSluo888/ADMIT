#!/usr/bin/env python3
"""Patients and cells per cohort (Table 1), counted from the standardised cohort files under $ADMIT_DATA_ROOT.

Cells = all rows of cells.parquet (including cells whose label lies outside the hierarchy); Schurch = cells left
after removing the dirt / undefined / mixed-label clusters (configs/schurch_label_mapping.json). The Orion row
(in-house gating labels) is not computed here.

  python experiments/cohort_table.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ADMIT_DATA_ROOT", str(REPO / "data")))


def main():
    import pandas as pd
    import pyarrow.parquet as pq
    rows = []
    src = DATA / "schurch_crc_codex" / "CRC_clusters_neighborhoods_markers.csv"
    if src.exists():
        excluded = json.loads((REPO / "configs" / "schurch_label_mapping.json").read_text())["excluded_unmapped_labels"]
        df = pd.read_csv(src, usecols=["ClusterName", "patients"], low_memory=False)
        df = df.loc[~df.ClusterName.isin(excluded)]
        rows.append(("schurch_crc_codex", df.patients.nunique(), len(df)))
    for tier in ("tier1", "tier2_std"):
        for f in sorted((DATA / tier).glob("*/cells.parquet")):
            n = pq.ParquetFile(f).metadata.num_rows
            pid = pq.read_table(f, columns=["patient_id"]).column("patient_id").to_pandas()
            rows.append((f.parent.name, int(pid.nunique()), int(n)))
    if not rows:
        sys.exit(f"no cohort found under {DATA}")
    print(f"{'cohort':40s} {'patients':>9s} {'cells':>12s}")
    for name, p, n in rows:
        print(f"{name:40s} {p:9,d} {n:12,d}")


if __name__ == "__main__":
    main()
