"""Shared lineage palette and helpers for the tissue case-study panels."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# lineage colours (T blue family, myeloid orange, B/plasma violet, epithelial/tumour magenta,
# stromal/endothelial green; red reserved for "unsupported")
NODE_COL = {
    "Cytotoxic_T_Cell": "#104281", "Helper_T_Cell": "#5598e7", "Regulatory_T_Cell": "#9ec5f4",
    "T_Lineage": "#2a78d6", "NK_Cell": "#86b6ef",
    "B_Cell": "#6250d6", "Plasma_Cell": "#9b8ee6",
    "Myeloid": "#eb6834", "Macrophage_CD163pos": "#b8461a", "Macrophage_CD163neg": "#f39a6b",
    "Dendritic_Cell": "#f6b27f", "Neutrophil": "#f7c59f",
    "Immune": "#9aa7b8", "Immune_Other": "#2a9d8f",   # teal (was grey #c3cad4; light grey = abstention only)
    "Epithelial": "#e87ba4", "Endothelial": "#7fc97f", "Stromal": "#008300",
    # UNRESOLVED (set members on different root paths) was #d9d9d9, the abstention grey; now a pale warm tone
    "UNMAPPED": "#5a5a5a", "UNRESOLVED": "#cdbf9f", "EMPTY": "#d9d9d9",
}
RED = "#e34948"


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def match_centroids(src_xy: np.ndarray, dst_xy: np.ndarray, max_d: float = 3.0):
    """One-to-one nearest-centroid matching within max_d px (greedy by distance over all candidate pairs).
    Returns (dst index per src row or -1, distance per src row or nan, stats)."""
    tree = cKDTree(dst_xy)
    pairs = tree.query_ball_point(src_xy, r=max_d)
    cand = [(float(np.hypot(*(src_xy[i] - dst_xy[j]))), i, j) for i, js in enumerate(pairs) for j in js]
    cand.sort()
    m = np.full(len(src_xy), -1, dtype=np.int64)
    dist = np.full(len(src_xy), np.nan)
    used = np.zeros(len(dst_xy), bool)
    for dd, i, j in cand:
        if m[i] < 0 and not used[j]:
            m[i], dist[i], used[j] = j, dd, True
    d1, _ = tree.query(src_xy)
    stats = {"n_src": int(len(src_xy)), "n_dst": int(len(dst_xy)), "n_matched": int((m >= 0).sum()),
             "match_rate_src": float((m >= 0).mean()), "match_rate_dst": float(used.mean()),
             "n_src_with_any_candidate": int((d1 <= max_d).sum()),
             "n_lost_to_one_to_one": int(((d1 <= max_d) & (m < 0)).sum()),
             "median_match_distance_px": float(np.nanmedian(dist)) if (m >= 0).any() else None}
    return m, dist, stats


def label_rgb(mask: np.ndarray, lab2rgb: dict, bg=(0.12, 0.12, 0.12)):
    """RGB image of mask labels coloured by lab2rgb {label: rgb}; others background."""
    lut = np.tile(np.array(bg, float), (int(mask.max()) + 1, 1))
    for lab, rgb in lab2rgb.items():
        lut[lab] = rgb
    return lut[mask]


def composite(chans, colours, lo_hi):
    """Additive pseudocolour composite on black from arcsinh arrays, per-channel (lo, hi) stretch."""
    out = np.zeros(chans[0].shape + (3,))
    for a, c, (lo, hi) in zip(chans, colours, lo_hi):
        v = np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)
        out += v[..., None] * hex2rgb(c)[None, None, :]
    return np.clip(out, 0, 1)
