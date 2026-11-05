"""Lin 2023 CyCIF <-> Orion pair loaders (provisional pair: labels inferred for CRC02-17).

CyCIF: $ADMIT_DATA_ROOT/tier1/lin2023_crc_cycif (no READY file; the pair sets allow_not_ready). Ambiguous label codes are coarsened to the coarsest
node common to the readings named in label_mapping.json (fixed before any result):
  CRC2-17_inferred: 3 (Keratin+Ki67+aSMA+, epithelial vs stromal) -> excluded; 5 (mixed immune) -> Immune;
                    6 (T helper, CD3 not required) -> Immune; 10 (Macrophage(I), CD163+CD68-) -> Myeloid;
                    12 (DP lymphocyte) -> T_Lineage; 18 (CD31+aSMA+Desmin+, stromal vs endothelial) -> excluded
  S4_TypeID (CRC01):  7 (Macrophage(I)) -> Myeloid; 10 (T helper) -> Immune; 12 (DP lymphocyte) -> T_Lineage
Transform arcsinh(x/150) (the Orion X_arc construction), then z-score.
CDX2 excluded (Table S3 PassQC = No).
Orion: in-house caches crc_train_full (CRC01-15) + crc_cal (CRC16-17) from $ADMIT_CACHE_DIR, X_arc (arcsinh(x/150)
z-scored) restricted to the 16 populated markers, node17 labels y; patients from the identity sidecars in
$ADMIT_IDENTITY_DIR. CyCIF CRC(n) = Orion CRC(n-1)
(orion_overlap.csv), so both map to one shared key 'orion_<nn>'.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import std

LIN = std.DATA / "tier1" / "lin2023_crc_cycif"
COARSEN = {("CRC2-17_inferred", 3): None, ("CRC2-17_inferred", 5): "Immune", ("CRC2-17_inferred", 6): "Immune",
           ("CRC2-17_inferred", 10): "Myeloid", ("CRC2-17_inferred", 12): "T_Lineage",
           ("CRC2-17_inferred", 18): None,
           ("S4_TypeID", 7): "Myeloid", ("S4_TypeID", 10): "Immune", ("S4_TypeID", 12): "T_Lineage"}


def _num(s):
    return int(re.sub(r"\D", "", str(s)))


def _relabel(df):
    lab = df["node17"].astype(str).copy()
    sch = df["code_scheme"].astype(str).to_numpy()
    code = df["author_newtype"].astype(int).to_numpy()
    for (sc, c), node in COARSEN.items():
        m = (sch == sc) & (code == c)
        lab[m] = node if node is not None else "UNMAPPED"
    return lab


def load_lin(key):
    ov = pd.read_csv(LIN / "orion_overlap.csv").dropna(subset=["orion_patient"])
    idm = {str(a): f"orion_{_num(b):02d}" for a, b in zip(ov.cycif_patient, ov.orion_patient)}
    d = std.load_std("lin2023_crc_cycif", "arcsinh:150", cohort_key=key, id_map=idm, exclude_columns=("CDX2",),
                     extra_cols=("code_scheme", "author_newtype"), relabel=_relabel)
    d["name"] = "lin2023_crc_cycif"
    return d


def load_orion(key):
    from common import load_identity, load_split
    xs, ys, ps = [], [], []
    um = None
    for s in ("crc_train_full", "crc_cal"):
        z = load_split(s)
        ident = load_identity(s)            # row alignment with the cache is checked there
        if ident is None:
            raise FileNotFoundError(f"set ADMIT_IDENTITY_DIR to the folder holding {s}_identity.npz")
        xs.append(z["X_arc"].astype(np.float32)); ys.append(z["y"].astype(np.int64))
        ps.append(np.array([f"orion_{_num(v):02d}" for v in ident["sample"]]))
        um = [str(m) for m in z["universal_markers"]]
        tt = [str(t) for t in z["target_types"]]
    x, y, raw = np.concatenate(xs), np.concatenate(ys), np.concatenate(ps)
    pop = [j for j in range(x.shape[1]) if all((xx[:, j] != 0).any() for xx in xs)]
    uniq, inv = np.unique(raw, return_inverse=True)
    keys = np.array([std.anon(key or "orion", p) for p in uniq])
    pkeys = np.sort(keys)
    return {"x": x[:, pop].copy(), "markers": [um[j] for j in pop], "source_columns": [um[j] for j in pop],
            "dropped_columns": [], "y": y, "patient": keys[inv], "pkeys": pkeys,
            "pcode": np.searchsorted(pkeys, keys)[inv], "excluded": {"n": 0}, "target_types": tt,
            "transform": "X_arc (arcsinh(x/150), z-scored)", "n_cells": int(len(y)), "name": "orion"}


LOADERS = {"lin2023_crc_cycif": load_lin, "orion": load_orion}
