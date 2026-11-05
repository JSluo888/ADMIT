"""Hand cases for A(P), pi_P, the M1 supported-ancestor projection and the conformal quantile."""
import numpy as np
import pytest

from admit import NODE17, Ontology, conformal_quantile, nmin_rule, supported_group_map
from admit.synthetic import TOY_MARKERS

ONTO = Ontology.from_repo(NODE17)
I = {n: i for i, n in enumerate(NODE17)}
FULL = list(TOY_MARKERS)


def revised(label, panel):
    r = ONTO.project(np.array([I[label]]), ONTO.admissible(panel))[0]
    return NODE17[r] if r >= 0 else None


def without(*ms):
    return [m for m in FULL if m not in ms]


def test_seventeen_nodes_and_requirements():
    assert len(NODE17) == 17 and ONTO.K == 17
    assert ONTO.R["Cytotoxic_T_Cell"] == {"CD45", "CD3e", "CD8"}
    assert ONTO.R["T_Lineage"] == {"CD45", "CD3e"}
    # requirements are cumulative along the ancestry
    for d in range(ONTO.K):
        p = ONTO.parent_idx[d]
        if p >= 0:
            assert ONTO.R[NODE17[p]] <= ONTO.R[NODE17[d]]


def test_full_panel_everything_admissible():
    assert ONTO.admissible(FULL).all()
    for n in NODE17:
        assert revised(n, FULL) == n


def test_without_cd8_cytotoxic_t_becomes_t_cell():
    adm = ONTO.admissible(without("CD8"))
    assert [NODE17[i] for i in np.where(~adm)[0]] == ["Cytotoxic_T_Cell"]
    assert revised("Cytotoxic_T_Cell", without("CD8")) == "T_Lineage"
    assert revised("Helper_T_Cell", without("CD8")) == "Helper_T_Cell"


def test_without_cd3_all_t_types_become_immune():
    for t in ("T_Lineage", "Helper_T_Cell", "Regulatory_T_Cell", "Cytotoxic_T_Cell"):
        assert revised(t, without("CD3e")) == "Immune"


def test_foxp3_and_cd163_and_cd138():
    assert revised("Regulatory_T_Cell", without("FOXP3")) == "T_Lineage"
    assert revised("Helper_T_Cell", without("FOXP3")) == "T_Lineage"              # FOXP3 must be measured
    assert revised("Macrophage_CD163pos", without("CD163")) == "Myeloid"
    assert revised("Macrophage_CD163neg", without("CD163")) == "Myeloid"          # CD163 must be measured
    assert revised("Plasma_Cell", without("CD138")) == "Immune"


def test_requirements_include_required_measured():
    assert ONTO.R["Helper_T_Cell"] == {"CD45", "CD3e", "CD4", "FOXP3"}
    assert ONTO.R["Macrophage_CD163neg"] == {"CD45", "CD68", "CD163"}
    assert ONTO.R_legacy["Helper_T_Cell"] == {"CD45", "CD3e", "CD4"}
    o = Ontology.from_repo(NODE17, req_override={"Helper_T_Cell": ["CD3e", "CD4", "FOXP3"]})
    assert o.R["Helper_T_Cell"] == {"CD3e", "CD4", "FOXP3"}
    with pytest.raises(ValueError):
        Ontology.from_repo(NODE17, req_override={"Helper_T_Cell": ["NOT_A_MARKER"]})


def test_cd45_isoforms_and_unidentifiable():
    assert revised("Cytotoxic_T_Cell", ["CD45RO", "CD3e", "CD8"]) == "Cytotoxic_T_Cell"   # any CD45 isoform
    no45 = without("CD45")
    assert revised("Cytotoxic_T_Cell", no45) is None       # no admissible ancestor: unidentifiable
    assert revised("Epithelial", no45) == "Epithelial"
    assert not ONTO.admissible([]).any()


def test_admissible_set_closed_under_ancestors():
    rng = np.random.default_rng(0)
    for _ in range(200):
        panel = [m for m in FULL if rng.random() < 0.5]
        adm = ONTO.admissible(panel)
        for d in np.where(adm)[0]:
            p = ONTO.parent_idx[d]
            assert p < 0 or adm[p]
        proj = ONTO.projection_map(adm)
        for d in range(ONTO.K):           # pi_P(y) is admissible and an ancestor-or-self of y
            if proj[d] >= 0:
                assert adm[proj[d]] and ONTO.anc[proj[d], d]


def test_m1_merges_small_group_into_parent():
    y = np.array([I["Regulatory_T_Cell"]] * 5 + [I["T_Lineage"]] * 6 + [I["Helper_T_Cell"]] * 20
                 + [I["Epithelial"]] * 3)
    gmap, S, counts = supported_group_map(ONTO, ONTO.admissible(FULL), y, n_min=9)
    assert NODE17[gmap[I["Regulatory_T_Cell"]]] == "T_Lineage"      # 5 < 9 cells: merged into the parent
    assert NODE17[gmap[I["T_Lineage"]]] == "T_Lineage"              # 5 + 6 = 11 >= 9: supported
    assert NODE17[gmap[I["Helper_T_Cell"]]] == "Helper_T_Cell"
    assert gmap[I["Epithelial"]] == -1                              # 3 cells and no admissible parent
    assert S[I["T_Lineage"]] and not S[I["Regulatory_T_Cell"]] and not S[I["Epithelial"]]
    assert counts[I["T_Lineage"]] == 11


@pytest.mark.parametrize("alpha,n_min", [(0.05, 19), (0.10, 9), (0.15, 6), (0.20, 4)])
def test_nmin_rule(alpha, n_min):
    assert nmin_rule(alpha) == n_min
    assert np.isfinite(conformal_quantile(np.linspace(0, 1, n_min), alpha))
    assert np.isinf(conformal_quantile(np.linspace(0, 1, n_min - 1), alpha))


def test_conformal_quantile_order_statistic():
    s = np.arange(1, 20, dtype=float)               # n = 19, alpha 0.1: k = ceil(20 * 0.9) = 18
    assert conformal_quantile(s, 0.10) == 18.0
    assert np.isinf(conformal_quantile(np.array([]), 0.10))
