"""ADMIT: admissibility-aware conformal cell typing for spatial proteomics.

Core objects (see ``admit.panel_matched`` for the full set of comparators and metrics)::

    from admit import Ontology, AdmitM1Frozen, NODE17, frozen_support_map, nmin_rule
    onto = Ontology.from_repo(NODE17)          # 17-node hierarchy with marker requirements R(c)
    adm = onto.admissible(panel)               # A(P): boolean mask over the 17 nodes
    revised = onto.project(y, adm)             # pi_P(y): finest admissible ancestor-or-self (-1 = unidentifiable)
    gmap, S, _ = frozen_support_map(onto, adm, y_train, n_cal=len(y_cal), n_min=nmin_rule(0.10))
    m = AdmitM1Frozen(onto, 0.10, gmap, S).fit(onto.aggregate(P_cal), y_cal)
    sets = m.predict(onto.aggregate(P_test))   # (n, 17) boolean prediction sets
"""
from admit.ontology import CELL_TYPE_ONTOLOGY, MARKER_VOCABULARY, NODE17
from admit.panel_matched import (ADMIT, AdmitM1, AdmitM1Frozen, Ontology, PanelMatchedPATCH, SourcePATCH,
                                 admit_m1_frozen_crc, aps_flat, chain_to_leaf, conformal_quantile, crsvp_vec,
                                 finest_depth, frozen_support_map, hcc_vec, lac_flat, lac_node_projected,
                                 lac_node_target, mask_arc, nmin_rule, resolved_leaf, restrict_to_vocab,
                                 scconform_vec, set_metrics, supported_group_map)

__version__ = "0.1.0"

__all__ = [
    "ADMIT", "AdmitM1", "AdmitM1Frozen", "CELL_TYPE_ONTOLOGY", "MARKER_VOCABULARY", "NODE17", "Ontology",
    "PanelMatchedPATCH", "SourcePATCH", "admit_m1_frozen_crc", "aps_flat", "chain_to_leaf", "conformal_quantile",
    "crsvp_vec", "finest_depth", "frozen_support_map", "hcc_vec", "lac_flat", "lac_node_projected",
    "lac_node_target", "mask_arc", "nmin_rule", "resolved_leaf", "restrict_to_vocab", "scconform_vec",
    "set_metrics", "supported_group_map",
]
