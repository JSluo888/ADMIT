"""Schurch CRC CODEX (35 patients) loader for task 5b: labels harmonized to the 17-node space with
the mapping recovered in schurch_infer_mapping.py, all 56 protein columns as the full panel, marker
names mapped onto the ProteoRanker vocabulary where R(c) needs them."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS  # noqa: E402

# Schurch et al. 2020 single-cell table (public; see README), under $ADMIT_DATA_ROOT/schurch_crc_codex/
SRC = (Path(os.environ.get("ADMIT_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data")))
       / "schurch_crc_codex" / "CRC_clusters_neighborhoods_markers.csv")
# Schurch antibody name (text before " - ") -> ProteoRanker marker vocabulary (only where it differs)
VOCAB = {"CD3": "CD3e", "Cytokeratin": "PanCK", "HLA-DR": "HLADR", "PD-L1": "PDL1", "PD-1": "PD1",
         "LAG-3": "LAG3", "Granzyme B": "GrzB"}
PILOT_SPLIT_SEEDS = [20260928, 20260929, 20260930]   # pilot/run_codex.py: 15 train / 12 cal / 8 test


def load_schurch():
    df = pd.read_csv(SRC, low_memory=False)
    mp = json.loads((RUNS / "schurch" / "schurch_label_mapping.json").read_text())
    excluded = mp["excluded_unmapped_labels"]
    df = df.loc[~df.ClusterName.isin(excluded)].reset_index(drop=True)
    marker_cols = [c for c in df if ":Cyc_" in c and not c.startswith(("HOECHST", "DRAQ5"))]
    names = [VOCAB.get(c.split(" - ")[0].split(":")[0], c.split(" - ")[0].split(":")[0]) for c in marker_cols]
    assert len(set(names)) == len(names)
    label = df.ClusterName.map({k: v["target"] for k, v in mp["mapping"].items()})
    assert label.notna().all()
    x = np.log1p(df[marker_cols].to_numpy(dtype=np.float64))
    return {"x": x, "markers": names, "marker_cols": marker_cols, "label_name": label.to_numpy(),
            "patient": df.patients.to_numpy(), "cluster": df.ClusterName.to_numpy(), "excluded": excluded,
            "mapping": {k: v["target"] for k, v in mp["mapping"].items()}}


def pilot_splits(patient):
    ids = np.unique(patient)
    assert len(ids) == 35
    out = []
    for seed in PILOT_SPLIT_SEEDS:
        perm = np.random.default_rng(seed).permutation(ids)
        out.append({"seed": seed, "train": perm[:15], "cal": perm[15:27], "test": perm[27:]})
    return out
