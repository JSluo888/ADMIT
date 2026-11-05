"""Old and current requirement maps for the case studies.

Current map (the default of Ontology.from_repo): R(c) = required_positive | required_measured.
Old map: required_positive only (Ontology.R_legacy). Both are used with AdmitM1, so old vs new isolates the map and the
results stay comparable with the downstream analyses."""
from cases_common import *  # noqa: F403
from admit.panel_matched import Ontology

EXTRA = {'Helper_T_Cell': 'FOXP3', 'Macrophage_CD163neg': 'CD163'}


def onto_old():
    o = Ontology.from_repo(NODE17)
    o.R = dict(o.R_legacy)
    return o


def onto_r3():
    o = Ontology.from_repo(NODE17)
    old = onto_old()
    assert all(o.R[n] == old.R[n] | ({EXTRA[n]} if n in EXTRA else set()) for n in o.names), \
        'current map is not the old map plus the required_measured markers'
    return o
