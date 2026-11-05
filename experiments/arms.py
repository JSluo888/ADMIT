"""Every arm of the comparison, on identical cells, for one panel P.

Inputs are label-column probabilities from the frozen scorer:
  Pc_P, Pt_P   calibration / test cells rescored with features masked to P
  Pc_full      calibration cells scored on the full source panel (old PATCH only)
Node scores (PATCH arms, node-projected LAC) are subtree sums of label columns.
Flat and hierarchical comparators run on label columns with panel-matched calibration
(calibration masked to P, native labels y), each with and without the A(P) filter.

Release note: the original arm list also contained ``patch_legacy_b5port_src`` (a port of an earlier benchmark's
construction, kept for continuity with older code and not reported in the paper). It depended on code that is not
part of this release and has been removed; every other arm is unchanged.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from admit.panel_matched import (Ontology, PanelMatchedPATCH, SourcePATCH, aps_flat, crsvp_vec,
                                 hcc_vec, lac_flat, lac_node_projected, scconform_vec)

ARMS = [
    "patch_pm", "patch_pm_backstop",                       # new: panel-matched, projected strata, no n_min
    "patch_pm_nc_pos",                                     # ABLATION (not the specified method): strata with n_c = 0 excluded
    "patch_src", "patch_src_backstop",                     # old: source tolerances on leaf labels
    "lac", "lac_filt", "aps", "aps_filt", "raps", "raps_filt",
    "lac_node_proj",                                       # ablation: marginal (non-Mondrian) projected LAC
    "hcc", "hcc_filt", "scconform", "scconform_filt", "crsvp", "crsvp_filt",
]


def run_arms(onto: Ontology, par: Dict[str, List[str]], panel: List[str],
             Pc_P: np.ndarray, yc: np.ndarray, Pt_P: np.ndarray,
             Pc_full: np.ndarray, src: SourcePATCH, alpha: float = 0.10,
             quantile: str = "order_stat", n_min: int = 10, arms=None):
    arms = arms or ARMS
    tt = onto.names
    adm = onto.admissible(panel)
    Nc, Nt = onto.aggregate(Pc_P), onto.aggregate(Pt_P)
    out, info = {}, {}
    pm = PanelMatchedPATCH(onto, alpha, quantile).fit(Nc, yc, panel)
    info["pm_strata"] = pm.strata_report()
    info["pm_supported"] = pm.finite            # strata with a finite tolerance
    info["pm_n_infinite_strata"] = int(pm.infinite.sum())
    info["pm_infinite_strata"] = [onto.names[c] for c in np.where(pm.infinite)[0]]
    want = set(arms)
    if "patch_pm" in want:
        out["patch_pm"] = pm.predict(Nt, backstop=False)
    if "patch_pm_backstop" in want:
        out["patch_pm_backstop"] = pm.predict(Nt, backstop=True)
    if "patch_pm_nc_pos" in want:
        # post-hoc ablation: drop admissible strata that received no calibration cell at all
        # (their lambda is +inf only because n_c = 0, so they are always included)
        out["patch_pm_nc_pos"] = out.get("patch_pm", pm.predict(Nt)) & (pm.n_c > 0)[None, :]
    if "patch_src" in want:
        out["patch_src"] = src.predict(Nt, panel, backstop=False)
    if "patch_src_backstop" in want:
        out["patch_src_backstop"] = src.predict(Nt, panel, backstop=True)
    flt = adm[None, :]
    if want & {"lac", "lac_filt"}:
        s = lac_flat(Pc_P, yc, Pt_P, alpha, quantile)
        out["lac"], out["lac_filt"] = s, s & flt
    if want & {"aps", "aps_filt"}:
        s = aps_flat(Pc_P, yc, Pt_P, alpha, quantile=quantile)
        out["aps"], out["aps_filt"] = s, s & flt
    if want & {"raps", "raps_filt"}:
        s = aps_flat(Pc_P, yc, Pt_P, alpha, k_reg=3, lam=0.05, quantile=quantile)
        out["raps"], out["raps_filt"] = s, s & flt
    if "lac_node_proj" in want:
        out["lac_node_proj"] = lac_node_projected(Nc, yc, Nt, onto, panel, alpha, quantile)
    for name, fn in (("hcc", hcc_vec), ("scconform", scconform_vec), ("crsvp", crsvp_vec)):
        if want & {name, name + "_filt"}:
            s = fn(Pc_P, yc, Pt_P, onto, alpha, quantile, n_min)
            out[name], out[name + "_filt"] = s, s & flt
    return {k: out[k] for k in arms if k in out}, info
