"""Standard-schema cohort loader, classifier heads and splits for the new-cohort ADMIT runs.

Schema ($ADMIT_DATA_ROOT/tier1/<cohort>/ or $ADMIT_DATA_ROOT/tier2_std/<cohort>/): cells.parquet, markers.tsv,
clinical.csv, label_mapping.json, QC.json, READY. A cohort is used only once READY exists.

Fixed choices (pre-specified before any result; per-cohort values in configs/cohort_registry.json):
- features: every markers.tsv row except nuclear/DNA/background channels (Schurch precedent: all protein columns);
  feature name = vocabulary_name when mapped, else the source name (so R(c) uses the vocabulary);
- transform: registry value per cohort (IMC/MIBI counts: arcsinh(x/5); fluorescence: the QC.json transform, else
  log1p), then a z-score fit on training-patient cells only; absent marker = 0.0 after z-scoring (mean imputation,
  the X_arc / round-1 encoding);
- labels: node17 column; UNMAPPED rows excluded and counted;
- patient ids are replaced by anonymous keys (sha256 of cohort + id) before anything is written.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "round2"))
sys.path.insert(0, str(HERE.parents[1]))
from admit.ontology import NODE17  # noqa: E402

DATA = Path(os.environ.get("ADMIT_DATA_ROOT", str(HERE.parents[1] / "data")))
OUT = Path(os.environ.get("ADMIT_RUNS", str(HERE.parents[1] / "runs"))) / "cohorts"
REGISTRY = HERE.parents[1] / "configs" / "cohort_registry.json"
NONPROTEIN = re.compile(r"(?i)(^|\b)(dna\d*|ir\d{3}|iridium|histone|dapi|hoechst|draq5|nuclear|background|"
                        r"empty|blank|autofluor|af\d{3}|bg\d*|barcode|noodle|chan_\d+)(\b|_|$)")


def registry() -> Dict:
    return json.loads(REGISTRY.read_text())


def cohort_dir(name: str) -> Path:
    alt = registry().get(name, {}).get("dir")
    if alt:
        return DATA / alt          # relative to $ADMIT_DATA_ROOT
    for tier in ("tier1", "tier2_std"):
        p = DATA / tier / name
        if (p / "cells.parquet").exists():
            return p
    raise FileNotFoundError(name)


def is_ready(name: str) -> bool:
    alt = registry().get(name, {}).get("dir")
    if alt:
        return (DATA / alt / "READY").exists()
    for tier in ("tier1", "tier2_std"):
        if (DATA / tier / name / "READY").exists():
            return True
    return False


def anon(cohort: str, pid: str) -> str:
    return "p" + hashlib.sha256(f"{cohort}|{pid}".encode()).hexdigest()[:10]


NOTE_NONPROTEIN = re.compile(r"(?i)(^\s*(dna|nuclear)\s*$|^\s*nuclear stain|^\s*dna\b|\bbackground\b|\bsecondary-only\b|\bautofluorescence\b)")


def feature_table(d: Path, synonyms: Optional[Dict[str, str]] = None, name_map: Optional[Dict[str, str]] = None):
    """synonyms (sensitivity analyses only): source_name -> vocabulary name used for R(c).
    name_map: source_name -> harmonised name for NON-vocabulary markers (pairs; e.g. Kulasinghe author -> CORAL)."""
    synonyms = synonyms or {}
    name_map = name_map or {}
    mk = pd.read_csv(d / "markers.tsv", sep="\t", keep_default_na=False)
    rows, names, dropped = [], [], []
    for r in mk.itertuples(index=False):
        voc = str(r.vocabulary_name)
        src = str(r.source_name)
        note = str(getattr(r, "notes", ""))
        if src in synonyms:
            voc = synonyms[src]
        if voc in ("NA", "", "nan") and (NONPROTEIN.search(src) or NOTE_NONPROTEIN.search(note)):
            dropped.append(src)
            continue
        nm = voc if voc not in ("NA", "", "nan") else name_map.get(src, src)
        if nm in names:              # a second column with the same vocabulary name keeps its source name
            nm = src
        rows.append(src)
        names.append(nm)
    return rows, names, dropped


def transform(x: np.ndarray, spec: str) -> np.ndarray:
    if spec == "none" or spec.startswith("arcsinh_p99"):   # arcsinh_p99 is fit later on training rows
        return x
    if spec == "log1p":
        return np.log1p(np.clip(x, 0, None))
    if spec.startswith("arcsinh"):
        return np.arcsinh(x / float(spec.split(":")[1]))
    raise ValueError(spec)


def load_std(name: str, transform_spec: str, cohort_key: Optional[str] = None,
             id_map: Optional[Dict[str, str]] = None, sample_as_patient: bool = False,
             synonyms: Optional[Dict[str, str]] = None, name_map: Optional[Dict[str, str]] = None,
             exclude_columns=(), extra_cols=(), relabel=None) -> Dict:
    """-> x (float32, transformed, not z-scored), markers, y (17-node index), patient (anon key), sample,
    excluded label counts. cohort_key: salt for the anonymous patient key (pairs share one key space);
    id_map: raw patient id -> shared id (crosswalk) before hashing; raw ids are never returned or written."""
    d = cohort_dir(name)
    src_cols, names, dropped = feature_table(d, synonyms, name_map)
    keepc = [i for i, c in enumerate(src_cols) if c not in set(exclude_columns)]
    dropped = dropped + [c for c in src_cols if c in set(exclude_columns)]
    src_cols, names = [src_cols[i] for i in keepc], [names[i] for i in keepc]
    pid_col = "sample_id" if sample_as_patient else "patient_id"
    cols = [pid_col, "node17", "label_fine"] + list(extra_cols) + src_cols
    df = pd.read_parquet(d / "cells.parquet", columns=cols)
    if relabel is not None:
        df["node17"] = relabel(df)
    tt = list(NODE17)          # the 17-node column order shared by every experiment
    idx = {t: i for i, t in enumerate(tt)}
    lab = df["node17"].astype(str)
    keep = lab.isin(idx).to_numpy()
    excluded = df.loc[~keep, "label_fine"].astype(str).value_counts().to_dict()
    x = transform(df.loc[keep, src_cols].to_numpy(dtype=np.float32), transform_spec).astype(np.float32)
    raw_pid = df.loc[keep, pid_col].astype(str).to_numpy()
    if id_map is not None:
        raw_pid = np.array([id_map.get(p, "unmatched:" + p) for p in raw_pid])
    uniq, inv = np.unique(raw_pid, return_inverse=True)
    keys = np.array([anon(cohort_key or name, p) for p in uniq])
    pkeys = np.sort(keys)
    pcode = np.searchsorted(pkeys, keys)[inv]
    return {"x": x, "markers": names, "source_columns": src_cols, "dropped_columns": dropped,
            "y": lab[keep].map(idx).to_numpy(np.int64), "patient": keys[inv], "pkeys": pkeys, "pcode": pcode,
            "excluded": {"unmapped_by_label_fine": excluded, "n": int((~keep).sum())},
            "target_types": tt, "transform": transform_spec, "n_cells": int(keep.sum())}


def fit_transform(x: np.ndarray, spec: str, train_rows: np.ndarray) -> Dict:
    """In place, for specs fit on training-patient cells. ``arcsinh_p99:<f>`` (size-normalised, non-count MIBI/IMC
    tables): x_m <- arcsinh(x_m / c_m), c_m = f * (99th percentile of marker m over training cells); if that is 0
    (censored marker), f * max over training cells; floor 1e-8. Returns the fitted cofactors."""
    if not spec.startswith("arcsinh_p99"):
        return {}
    f = float(spec.split(":")[1])
    xt = x[train_rows]
    c = f * np.percentile(xt, 99, axis=0)
    mx = f * xt.max(axis=0)
    c = np.where(c > 0, c, mx)
    c = np.maximum(c, 1e-8).astype(np.float32)
    np.arcsinh(x / c, out=x)
    return {"cofactors": c.tolist()}


# --------------------------------------------------------------------------------------------------------------
# Heads
# --------------------------------------------------------------------------------------------------------------


class Head:
    """Frozen classifier: predict17(Z) -> (n, 17) probabilities (columns of absent classes are 0)."""

    def __init__(self, kind: str, K: int = 17):
        self.kind, self.K = kind, K

    def fit(self, Z: np.ndarray, y: np.ndarray, seed: int):
        self.present = np.unique(y)
        remap = np.full(self.K, -1)
        remap[self.present] = np.arange(len(self.present))
        yy = remap[y]
        t = time.time()
        if self.kind == "xgb":
            import xgboost as xgb
            import os
            self.device = os.environ.get("ADMIT_XGB_DEVICE", "cuda")
            self.m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, objective="multi:softprob",
                                       tree_method="hist", device=self.device, random_state=seed,
                                       n_jobs=int(os.environ.get("ADMIT_XGB_NJOBS", "2")))
            self.m.fit(Z, yy, verbose=False)
        elif self.kind == "mlp":
            self._fit_mlp(Z, yy, seed)
            self.device = self.dev
        elif self.kind == "logistic":
            from sklearn.linear_model import LogisticRegression
            self.m = LogisticRegression(C=1.0, max_iter=1500, random_state=seed).fit(Z, yy)
        else:
            raise ValueError(self.kind)
        self.fit_s = round(time.time() - t, 1)
        return self

    def _fit_mlp(self, Z, yy, seed):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        d, C = Z.shape[1], len(self.present)
        net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(),
                            nn.Linear(128, C)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        X = torch.as_tensor(Z, dtype=torch.float32, device=dev)
        Y = torch.as_tensor(yy, dtype=torch.long, device=dev)
        g = torch.Generator(device="cpu").manual_seed(seed)
        n, bs, epochs = len(X), 4096, 20
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * ((n + bs - 1) // bs))
        lossf = nn.CrossEntropyLoss()
        net.train()
        for _ in range(epochs):
            perm = torch.randperm(n, generator=g).to(dev)
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                opt.zero_grad(set_to_none=True)
                loss = lossf(net(X[b]), Y[b])
                loss.backward()
                opt.step()
                sched.step()
        net.eval()
        self.m, self.dev = net, dev

    def proba(self, Z: np.ndarray) -> np.ndarray:
        if self.kind == "mlp":
            import torch
            out = []
            with torch.no_grad():
                for i in range(0, len(Z), 1 << 18):
                    t = torch.as_tensor(Z[i:i + (1 << 18)], dtype=torch.float32, device=self.dev)
                    out.append(torch.softmax(self.m(t), dim=1).float().cpu().numpy())
            return np.concatenate(out) if out else np.zeros((0, len(self.present)), np.float32)
        return self.m.predict_proba(Z)

    def predict17(self, Z: np.ndarray) -> np.ndarray:
        P = np.zeros((len(Z), self.K), dtype=np.float64)
        if len(Z):
            P[:, self.present] = self.proba(Z)
            P /= P.sum(1, keepdims=True)
        return P


# --------------------------------------------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------------------------------------------


def patient_splits(patients: np.ndarray, fractions, seeds) -> List[Dict]:
    """40/20/40 by patient: n_train = floor(.4 N), n_cal = floor(.2 N), test = the rest.
    rng = default_rng(seed); perm = rng.permutation(sorted unique patient keys)."""
    ids = np.unique(patients)
    f = np.asarray(fractions, dtype=float)
    n_tr, n_ca = (np.floor(f[:2] / f.sum() * len(ids))).astype(int)
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        perm = rng.permutation(ids)
        out.append({"seed": int(s), "train": perm[:n_tr], "cal": perm[n_tr:n_tr + n_ca], "test": perm[n_tr + n_ca:]})
    return out


def patient_groups(pcode: np.ndarray, n: int) -> List[np.ndarray]:
    """rows of each patient code 0..n-1 (sorted)."""
    order = np.argsort(pcode, kind="stable")
    bounds = np.searchsorted(pcode[order], np.arange(n + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(n)]


def capped_index(groups: List[np.ndarray], members, cap: Optional[int], rng) -> np.ndarray:
    """Rows of the given patient codes, at most ``cap`` per patient (random, without replacement), sorted."""
    out = []
    for g in sorted(int(m) for m in members):
        rows = groups[g]
        out.append(rows if (not cap or len(rows) <= cap) else rng.choice(rows, cap, replace=False))
    return np.sort(np.concatenate(out)) if out else np.zeros(0, np.int64)


def train_index(groups: List[np.ndarray], members, cap: int, rng) -> np.ndarray:
    """Round-2 rule: up to ``cap`` cells per training patient, drawn with the split rng in train order."""
    return np.concatenate([rng.choice(groups[int(g)], min(cap, len(groups[int(g)])), replace=False)
                           for g in members])


def registry_name_map(name: str) -> Optional[Dict[str, str]]:
    """Name map documented in the cohort's QC.json (registry key name_map_from_qc = dotted path)."""
    if registry().get(name, {}).get("name_map"):
        return dict(registry()[name]["name_map"])
    path = registry().get(name, {}).get("name_map_from_qc")
    if not path:
        return None
    q = json.loads((cohort_dir(name) / "QC.json").read_text())
    for k in path.split("."):
        q = q[k]
    return dict(q)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    tmp.replace(path)
