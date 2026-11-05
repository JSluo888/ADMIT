"""Shared loaders for the Orion / external-cohort experiments (frozen colorectal backbones).

Paths (environment variables; see README):
  ADMIT_RUNS          output root (default <repo>/runs)
  ADMIT_CACHE_DIR     feature caches: Orion crc_train_full/crc_cal/crc_test.npz and the external
                      <cohort>_40d.npz caches (in-house; not distributed)
  ADMIT_MODEL_DIR     frozen backbones (in-house weights; not distributed)
  ADMIT_IDENTITY_DIR  optional row-aligned <split>_identity.npz sidecars (patient/sample of each cache row)

Two frozen scorers are exposed behind one interface:

* ``round0``  -- ``round0_xgb_v6_leakfree``, X_arc 40-D, absent = 0.0.
* ``v6crc``   -- ``v6.0_crconly``, X_gmm 160-D with presence bits and
                 mask augmentation; v6 labels mapped onto the 17-node space.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import xgboost as xgb

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from admit.panel_matched import Ontology, mask_arc, mask_gmm160  # noqa: E402

UQ = Path(os.environ.get("ADMIT_CACHE_DIR", str(REPO / "data" / "cache")))
MODELS = Path(os.environ.get("ADMIT_MODEL_DIR", str(REPO / "models")))
RUNS = Path(os.environ.get("ADMIT_RUNS", str(REPO / "runs")))
LOCAL_COHORTS = ["hartmann", "hoch", "immucan", "liu", "maps", "schurch", "tietscher"]
# Second batch of external 40-D caches. Shift-target roster: schurch, hoch, immucan, tietscher, liu, maps,
# hartmann, sorin, risom, danenberg, karimi, plus cords. danenberg (uncurated GT mapping) and risom
# (DeepCell-derived labels) are reported separately from the clean set.
O2_COHORTS = ["cords", "sorin", "karimi", "danenberg", "risom"]
ALL_COHORTS = LOCAL_COHORTS + O2_COHORTS
COHORT_FLAGS = {"danenberg": "GT mapping uncurated (gt_mappings status=missing)",
                "risom": "labels DeepCell-derived (no primary GT on disk)",
                "sorin": "panel lacks CD45 (and isoforms): Immune lineage not admissible",
                "karimi": "panel lacks CD45 (and isoforms): Immune lineage not admissible"}
SEPARATE_COHORTS = {"danenberg", "risom"}
# Cohorts whose cached X_arc is an all-zero placeholder (cache_{hartmann,maps,tietscher}
# _features.py write np.zeros for X_arc). Any X_arc-based scorer sees a constant input there.
XARC_PLACEHOLDER = {"hartmann", "maps", "tietscher"}

V6_TO_17 = {
    "B_cell": "B_Cell", "CD4_T": "Helper_T_Cell", "CD8_T": "Cytotoxic_T_Cell",
    "Endothelial": "Endothelial", "Epithelial": "Epithelial", "Fibroblast": "Stromal",
    "ImmOther": "Immune_Other", "Lymphoid": "Immune", "Macrophage": "Myeloid",
    "Macrophage_CD163neg": "Macrophage_CD163neg", "Macrophage_CD163pos": "Macrophage_CD163pos",
    "Myeloid": "Myeloid", "T_cell": "T_Lineage", "Treg": "Regulatory_T_Cell",
    "DC": "Dendritic_Cell", "NK": "NK_Cell", "Neutrophil": "Neutrophil", "Plasma": "Plasma_Cell",
    "Immune": "Immune",
}


def load_split(name: str) -> Dict[str, np.ndarray]:
    d = np.load(UQ / f"{name}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_identity(split: str) -> Optional[Dict[str, np.ndarray]]:
    """Row-aligned identity sidecar ``<split>_identity.npz`` (keys sample, region, cell_id,
    x_centroid, y_centroid, y) from $ADMIT_IDENTITY_DIR, or None if not available.
    Alignment is verified: the sidecar's y must equal the cache's y row for row."""
    d = os.environ.get("ADMIT_IDENTITY_DIR")
    if not d:
        return None
    p = Path(d) / f"{split}_identity.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    ident = {k: z[k] for k in z.files}
    y = load_split(split)["y"]
    if not np.array_equal(np.asarray(ident["y"]).astype(np.int64), np.asarray(y).astype(np.int64)):
        raise ValueError(f"{p}: y does not match {split}.npz row for row")
    ident["sample"] = np.asarray(ident["sample"]).astype(str)
    return ident


def load_cohort(slug: str) -> Dict:
    d = np.load(UQ / f"{slug}_40d.npz", allow_pickle=True)
    um = [str(m) for m in d["universal_markers"]]
    pres = d["markers_present"].astype(bool)
    return {"X_arc": d["X_arc"].astype(np.float32), "X_gmm40": d["X_gmm"].astype(np.float32),
            "y": d["y"].astype(np.int64), "markers": um, "present": pres,
            "panel": [m for m, p in zip(um, pres) if p],
            "target_types": [str(t) for t in d["target_types"]]}


def orion_panel() -> List[str]:
    """Markers populated (any nonzero X_arc) in all three Orion splits, universal order."""
    cols = None
    for s in ("crc_train_full", "crc_cal", "crc_test"):
        d = np.load(UQ / f"{s}.npz", allow_pickle=True)
        nz = (d["X_arc"] != 0).any(axis=0)
        cols = nz if cols is None else (cols & nz)
        um = [str(m) for m in d["universal_markers"]]
    return [m for m, k in zip(um, cols) if k]


class Backbone:
    name: str
    markers: List[str]
    target_types: List[str]

    def features(self, split: Dict, external: bool = False) -> np.ndarray: ...
    def mask(self, X: np.ndarray, panel: Sequence[str], external: bool = False) -> np.ndarray: ...
    def predict17(self, X: np.ndarray) -> np.ndarray: ...


def _booster(path: Path, device: str) -> xgb.Booster:
    b = xgb.Booster()
    b.load_model(str(path))
    b.set_param({"device": device, "nthread": 20})
    return b


class Round0(Backbone):
    def __init__(self, model_dir: Optional[Path] = None, device: str = "cpu"):
        model_dir = Path(model_dir or MODELS / "round0_xgb_v6_leakfree_O2")
        meta = json.loads((model_dir / "metadata.json").read_text())
        self.meta = meta
        self.model_dir = str(model_dir)
        self.name = model_dir.name
        self.markers = list(meta["universal_markers"])
        self.target_types = list(meta["target_types_full"])
        self.m2o = {int(k): int(v) for k, v in meta["label_remap_model_to_orig"].items()}
        self.booster = _booster(model_dir / "model.ubj", device)

    def features(self, split, external=False):
        return np.asarray(split["X_arc"], dtype=np.float32)

    def mask(self, X, panel, external=False):
        return mask_arc(X, self.markers, panel)

    def predict17(self, X):
        Pp = self.booster.predict(xgb.DMatrix(X))
        P = np.zeros((len(X), len(self.target_types)), dtype=np.float64)
        for mc, oc in self.m2o.items():
            P[:, oc] = Pp[:, mc]
        P /= np.clip(P.sum(1, keepdims=True), 1e-12, None)
        return P


class V6CRC(Backbone):
    """v6.0_crconly on 160-D X_gmm; outputs mapped to the 17-node label columns."""

    def __init__(self, target_types17: Sequence[str], device: str = "cpu"):
        model_dir = MODELS / "v6.0_crconly" / "xgboost"
        meta = json.loads((model_dir / "metadata.json").read_text())
        self.meta = meta
        self.name = "v6.0_crconly"
        self.markers = list(meta["universal_markers"])
        self.target_types = list(target_types17)
        v6 = meta["target_types"]
        idx17 = {t: i for i, t in enumerate(self.target_types)}
        self.col_map = {int(k): idx17[V6_TO_17[v6[int(v)]]] for k, v in meta["contig_to_orig"].items()}
        self.booster = _booster(model_dir / "model.ubj", device)

    def features(self, split, external=False):
        if external:  # 40-D GMM cache -> 160-D (lodo_evaluation_v6.load_npz)
            X40, pres = split["X_gmm40"], split["present"].astype(np.float32)
            n = len(X40)
            X = np.zeros((n, 160), dtype=np.float32)
            X[:, :40] = X40
            X[:, 40:80] = pres[None, :]
            X[:, 80:120] = 0.5
            return X
        return np.asarray(split["X_gmm"], dtype=np.float32)

    def mask(self, X, panel, external=False):
        # ``external`` = encode as the external caches are encoded (no Nimbus at all).
        return mask_gmm160(X, self.markers, panel, drop_nimbus=external)

    def predict17(self, X):
        Pp = self.booster.predict(xgb.DMatrix(X))
        P = np.zeros((len(X), len(self.target_types)), dtype=np.float64)
        for mc, oc in self.col_map.items():
            P[:, oc] += Pp[:, mc]
        P /= np.clip(P.sum(1, keepdims=True), 1e-12, None)
        return P


BACKBONES = {
    "round0": "round0_xgb_v6_leakfree_O2 -- original frozen colorectal model (headline backbone)",
    "round0rt": "round0_xgb_v6_leakfree retrained here (reproducibility check only)",
    "v6crc": "v6.0_crconly (X_gmm 160-D, presence bits, mask augmentation; secondary backbone)",
}


def get_backbone(name: str, target_types17: Sequence[str], device: str = "cpu") -> Backbone:
    if name == "round0":
        return Round0(MODELS / "round0_xgb_v6_leakfree_O2", device=device)
    if name == "round0rt":
        return Round0(MODELS / "round0_xgb_v6_leakfree", device=device)
    if name == "v6crc":
        return V6CRC(target_types17, device=device)
    raise ValueError(name)


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    tmp.replace(path)


class Timer:
    def __init__(self):
        self.t0 = time.time(); self.marks = {}

    def mark(self, k):
        self.marks[k] = round(time.time() - self.t0, 2)
        return self.marks[k]
