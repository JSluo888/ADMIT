"""Cell-type hierarchy and marker requirements R(c) used by ADMIT.

Each node of the 17-node hierarchy lists ``required_positive`` markers (the markers an expert needs to call
that type) and ``required_measured`` markers (markers that must be on the panel, positive or not, because the
node is told from a sibling by their absence: Helper T needs FOXP3, CD163- macrophage needs CD163). The
requirement set of the paper is R(c) = required_positive | required_measured (cumulative along the ancestry; CD45
is satisfied by any of its isoforms, see ``admit.panel_matched.ISOFORM_SATISFIERS``). Only ``parent``,
``required_positive`` and ``required_measured`` enter the method; the other fields are documentation.

Sources: Cell Ontology (CL), CellMarker 2.0, PanglaoDB and published multiplex panels
(Schurch 2020, Keren 2018, Jackson 2020, Hoch 2022, IMMUcan 2024).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class CellTypeNode:
    """A node in the cell type ontology tree."""
    name: str
    parent: Optional[str] = None
    required_positive: List[str] = field(default_factory=list)
    required_negative: List[str] = field(default_factory=list)
    optional_markers: List[str] = field(default_factory=list)
    # Markers that must be measured (not positive): the absence-defining marker that separates the node from a
    # sibling. Lineage-level exclusions (e.g. Epithelial CD45-) are not listed here.
    required_measured: List[str] = field(default_factory=list)
    level: str = "L1"  # L1=lineage, L2=subtype, L3=state
    cl_id: Optional[str] = None  # Cell Ontology ID
    description: str = ""
    is_leaf: bool = True
    children: List[str] = field(default_factory=list)


# ============================================================================
# Cell Type Ontology
# ============================================================================

CELL_TYPE_ONTOLOGY: Dict[str, CellTypeNode] = {
    # --- Root ---
    "Root": CellTypeNode(
        name="Root", parent=None, level="L0", is_leaf=False,
        children=["Immune", "Epithelial", "Endothelial", "Stromal", "Unknown", "Doublet_Artifact"],
    ),

    # --- Major Lineages (L1) ---
    "Immune": CellTypeNode(
        name="Immune", parent="Root", level="L1", is_leaf=False,
        required_positive=["CD45"],
        required_negative=["PanCK"],
        cl_id="CL:0000738",  # leukocyte
        children=["T_Lineage", "B_Cell", "Myeloid", "NK_Cell", "Dendritic_Cell", "Plasma_Cell", "Neutrophil", "Immune_Other"],
        description="All hematopoietic/immune cells (CD45+)",
    ),
    "Epithelial": CellTypeNode(
        name="Epithelial", parent="Root", level="L1", is_leaf=True,
        required_positive=["PanCK"],
        required_negative=["CD45"],
        optional_markers=["Ecadherin", "Ki67"],
        cl_id="CL:0000066",
        description="Epithelial/tumor cells (PanCK+ or Ecadherin+, CD45-)",
    ),
    "Endothelial": CellTypeNode(
        name="Endothelial", parent="Root", level="L1", is_leaf=True,
        required_positive=["CD31"],
        required_negative=["PanCK", "CD45"],
        cl_id="CL:0000115",
        description="Blood vessel endothelial cells (CD31+)",
    ),
    "Stromal": CellTypeNode(
        name="Stromal", parent="Root", level="L1", is_leaf=True,
        required_positive=["aSMA"],
        required_negative=["CD45", "CD31", "PanCK"],
        optional_markers=["Vimentin"],
        cl_id="CL:0000499",  # stromal cell
        description="Myofibroblast-like stromal cells (aSMA+). Note: aSMA marks myofibroblasts, not all fibroblasts",
    ),
    "Unknown": CellTypeNode(
        name="Unknown", parent="Root", level="L1", is_leaf=True,
        description="Cells with no clear marker pattern",
    ),
    "Doublet_Artifact": CellTypeNode(
        name="Doublet_Artifact", parent="Root", level="L1", is_leaf=True,
        description="Segmentation artifacts with biologically impossible marker co-expression",
    ),

    # --- Immune Subtypes ---
    "T_Lineage": CellTypeNode(
        name="T_Lineage", parent="Immune", level="L1.1", is_leaf=False,
        required_positive=["CD45", "CD3e"],
        required_negative=["CD20", "CD68"],
        cl_id="CL:0000084",  # T cell
        children=["Helper_T_Cell", "Regulatory_T_Cell", "Cytotoxic_T_Cell"],
        description="T cell lineage (CD45+ CD3e+). Returned when CD4/CD8/FOXP3 markers are absent or ambiguous.",
    ),
    "Helper_T_Cell": CellTypeNode(
        name="Helper_T_Cell", parent="T_Lineage", level="L2",
        required_positive=["CD45", "CD3e", "CD4"],
        required_negative=["FOXP3", "CD8"],
        required_measured=["FOXP3"],
        optional_markers=["PD1", "CD45RO", "Ki67"],
        cl_id="CL:0000492",  # CD4-positive helper T cell
        description="CD4+ conventional T helper cell (CD3e+ CD4+ FOXP3-)",
    ),
    "Regulatory_T_Cell": CellTypeNode(
        name="Regulatory_T_Cell", parent="T_Lineage", level="L2",
        required_positive=["CD45", "CD3e", "CD4", "FOXP3"],
        required_negative=["CD8"],
        optional_markers=["PD1", "CD45RO", "Ki67"],
        cl_id="CL:0000815",  # regulatory T cell
        description="Regulatory T cell (CD3e+ CD4+ FOXP3+)",
    ),
    "Cytotoxic_T_Cell": CellTypeNode(
        name="Cytotoxic_T_Cell", parent="T_Lineage", level="L2",
        required_positive=["CD45", "CD3e", "CD8"],
        required_negative=["CD4"],
        optional_markers=["PD1", "CD45RO", "Ki67", "PDL1"],
        cl_id="CL:0000794",  # CD8-positive cytotoxic T cell
        description="CD8+ cytotoxic T cell (CD3e+ CD8+)",
    ),

    "B_Cell": CellTypeNode(
        name="B_Cell", parent="Immune", level="L1.1", is_leaf=True,
        required_positive=["CD45", "CD20"],
        required_negative=["CD3e", "CD68"],
        optional_markers=["Ki67"],
        cl_id="CL:0000236",  # B cell
        description="B cell (CD45+ CD20+)",
    ),

    "Myeloid": CellTypeNode(
        name="Myeloid", parent="Immune", level="L1.1", is_leaf=False,
        required_positive=["CD45", "CD68"],
        required_negative=["CD3e", "CD20"],
        cl_id="CL:0000763",  # myeloid cell
        children=["Macrophage_CD163pos", "Macrophage_CD163neg"],
        description="Myeloid/macrophage lineage (CD45+ CD68+). Returned when CD163 is absent.",
    ),
    "Macrophage_CD163pos": CellTypeNode(
        name="Macrophage_CD163pos", parent="Myeloid", level="L2",
        required_positive=["CD45", "CD68", "CD163"],
        required_negative=["CD3e", "CD20"],
        optional_markers=["PDL1", "Ki67"],
        cl_id="CL:0000235",  # macrophage
        description="CD163+ macrophage / TAM-like (CD68+ CD163+). Often called M2 but represents a spectrum.",
    ),
    "Macrophage_CD163neg": CellTypeNode(
        name="Macrophage_CD163neg", parent="Myeloid", level="L2",
        required_positive=["CD45", "CD68"],
        required_negative=["CD163", "CD3e", "CD20"],
        required_measured=["CD163"],
        optional_markers=["PDL1", "Ki67"],
        description="CD163-negative macrophage (CD68+ CD163-). Often called M1 but represents a spectrum.",
    ),

    "NK_Cell": CellTypeNode(
        name="NK_Cell", parent="Immune", level="L1.1", is_leaf=True,
        required_positive=["CD45", "CD56"],
        required_negative=["CD3e", "CD20", "CD68"],
        optional_markers=["CD16", "GrzB"],
        cl_id="CL:0000623",  # natural killer cell
        description="Natural killer cell (CD45+ CD56+ CD3e-). Collapses to Immune_Other when CD56 absent.",
    ),

    "Dendritic_Cell": CellTypeNode(
        name="Dendritic_Cell", parent="Immune", level="L1.1", is_leaf=True,
        required_positive=["CD45", "CD11c", "HLADR"],
        required_negative=["CD3e", "CD20", "CD68"],
        optional_markers=["CD303", "CD1c"],
        cl_id="CL:0000451",  # dendritic cell
        description="Dendritic cell (CD45+ CD11c+ HLA-DR+ CD68-). Reported in 5/8 papers. Collapses to Immune_Other when CD11c or HLA-DR absent.",
    ),

    "Plasma_Cell": CellTypeNode(
        name="Plasma_Cell", parent="Immune", level="L1.1", is_leaf=True,
        required_positive=["CD45", "CD138"],
        required_negative=["CD3e", "CD20"],
        optional_markers=["CD38"],
        cl_id="CL:0000786",  # plasma cell
        description="Plasma cell (CD45+ CD138+ CD20-). Terminally differentiated B-lineage. Collapses to B_Cell when CD138 absent.",
    ),

    "Neutrophil": CellTypeNode(
        name="Neutrophil", parent="Immune", level="L1.1", is_leaf=True,
        required_positive=["CD45", "CD15"],  # CD66b is alternative (checked separately)
        required_negative=["CD3e", "CD20"],
        optional_markers=["CD66b", "MPO"],
        cl_id="CL:0000775",  # neutrophil
        description="Neutrophil (CD45+ CD15+ or CD66b+). Collapses to Immune_Other when CD15/CD66b absent.",
    ),

    "Immune_Other": CellTypeNode(
        name="Immune_Other", parent="Immune", level="L1.1", is_leaf=True,
        required_positive=["CD45"],
        required_negative=[],
        cl_id="CL:0000738",
        description="CD45+ immune cell without specific lineage markers. Fallback when DC/NK/Neutrophil markers unavailable.",
    ),
}


# The 17 nodes ADMIT works with, in the column order used by every experiment (label / probability columns).
# Root, Unknown and Doublet_Artifact above are not part of the 17-node label space.
NODE17: List[str] = [
    "B_Cell", "Cytotoxic_T_Cell", "Dendritic_Cell", "Endothelial", "Epithelial", "Helper_T_Cell", "Immune",
    "Immune_Other", "Macrophage_CD163neg", "Macrophage_CD163pos", "Myeloid", "NK_Cell", "Neutrophil",
    "Plasma_Cell", "Regulatory_T_Cell", "Stromal", "T_Lineage",
]

# The 40-marker vocabulary that markers.tsv ``vocabulary_name`` refers to (R(c) only uses these names).
MARKER_VOCABULARY: List[str] = [
    "CD3e", "CD4", "CD8", "CD20", "CD45", "CD45RO", "CD68", "Ki67", "FOXP3", "PD1", "PDL1", "CD11c", "CD163",
    "CD31", "CD45RA", "PanCK", "aSMA", "Ecadherin", "Vimentin", "HLADR", "CD56", "CD138", "CD15", "CD16",
    "CD206", "MPO", "CD14", "CD25", "CD38", "CD19", "CD11b", "CD44", "GrzB", "ICOS", "LAG3", "CD57", "CD303",
    "FAP", "PDGFRb", "Podoplanin",
]
