"""Synthetic multiplexed-imaging cohort on the 17-node hierarchy (for the toy demo and the tests).

Each cell type expresses its required markers R(c) (plus a few extra lineage markers) above background, with
Gaussian noise and a per-patient intensity shift. Features are z-scored on training cells, so an unmeasured marker
is encoded as 0.0 (the training mean), the encoding used throughout the experiments (``admit.mask_arc``).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from admit.ontology import NODE17

TOY_MARKERS = ["CD45", "CD3e", "CD4", "CD8", "FOXP3", "CD20", "CD138", "CD68", "CD163", "CD56", "CD11c", "HLADR",
               "CD15", "PanCK", "Ecadherin", "CD31", "aSMA", "Vimentin"]

# leaf types and the markers they express (R(c) plus extra lineage markers)
TOY_PROFILES: Dict[str, Sequence[str]] = {
    "Helper_T_Cell": ["CD45", "CD3e", "CD4"],
    "Regulatory_T_Cell": ["CD45", "CD3e", "CD4", "FOXP3"],
    "Cytotoxic_T_Cell": ["CD45", "CD3e", "CD8"],
    "B_Cell": ["CD45", "CD20", "HLADR"],
    "Plasma_Cell": ["CD45", "CD138"],
    "Macrophage_CD163pos": ["CD45", "CD68", "CD163", "HLADR"],
    "Macrophage_CD163neg": ["CD45", "CD68", "HLADR"],
    "NK_Cell": ["CD45", "CD56"],
    "Dendritic_Cell": ["CD45", "CD11c", "HLADR"],
    "Neutrophil": ["CD45", "CD15"],
    "Immune_Other": ["CD45"],
    "Epithelial": ["PanCK", "Ecadherin"],
    "Endothelial": ["CD31"],
    "Stromal": ["aSMA", "Vimentin"],
}

# relative abundance (with a small calibration set the rare types are merged into their parent by ADMIT+M1)
TOY_WEIGHTS: Dict[str, float] = {
    "Epithelial": 30, "Stromal": 12, "Endothelial": 6, "Helper_T_Cell": 8, "Cytotoxic_T_Cell": 8,
    "Regulatory_T_Cell": 1.5, "B_Cell": 5, "Plasma_Cell": 2, "Macrophage_CD163pos": 6, "Macrophage_CD163neg": 4,
    "NK_Cell": 1, "Dendritic_Cell": 1.5, "Neutrophil": 3, "Immune_Other": 3,
}


def make_toy_cohort(n_patients: int = 30, cells_per_patient: int = 1500, signal: float = 3.0,
                    patient_sd: float = 0.15, seed: int = 0, markers: Optional[Sequence[str]] = None) -> Dict:
    """Returns dict with raw intensities ``x`` (n, m), ``markers``, ``y`` (index into NODE17) and ``patient``."""
    markers = list(markers or TOY_MARKERS)
    rng = np.random.default_rng(seed)
    types = list(TOY_WEIGHTS)
    w = np.array([TOY_WEIGHTS[t] for t in types], float)
    w /= w.sum()
    idx17 = {t: i for i, t in enumerate(NODE17)}
    col = {m: j for j, m in enumerate(markers)}
    prof = np.zeros((len(types), len(markers)))
    for i, t in enumerate(types):
        for m in TOY_PROFILES[t]:
            if m in col:
                prof[i, col[m]] = signal
    xs, ys, ps = [], [], []
    for p in range(n_patients):
        t = rng.choice(len(types), size=cells_per_patient, p=w)
        shift = rng.normal(0.0, patient_sd, size=len(markers))
        xs.append(prof[t] + shift[None, :] + rng.normal(0.0, 1.0, size=(cells_per_patient, len(markers))))
        ys.append(np.array([idx17[types[k]] for k in t], dtype=np.int64))
        ps.append(np.full(cells_per_patient, p, dtype=np.int64))
    return {"x": np.concatenate(xs).astype(np.float32), "markers": markers, "y": np.concatenate(ys),
            "patient": np.concatenate(ps)}


def zscore(x_train: np.ndarray, *xs: np.ndarray):
    """z-score every array with the training mean / sd (an absent marker is then 0.0 = the training mean)."""
    mu, sd = x_train.mean(0), x_train.std(0) + 1e-8
    return [((x - mu) / sd).astype(np.float32) for x in (x_train,) + xs]


def predict17(clf, X: np.ndarray, K: int = len(NODE17)) -> np.ndarray:
    """(n, 17) label-column probabilities from a fitted scikit-learn classifier (absent classes get 0)."""
    P = np.zeros((len(X), K))
    P[:, np.asarray(clf.classes_, dtype=np.int64)] = clf.predict_proba(X)
    return P / P.sum(1, keepdims=True)
