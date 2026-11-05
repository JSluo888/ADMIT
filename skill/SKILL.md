---
name: admit
description: Calibrated cell typing for spatial proteomics when markers are missing. Use when asked which cell types a marker panel can support, to turn classifier probabilities into calibrated prediction sets for a given panel, or to compare candidate panels.
---

# ADMIT

Install from the repository root: `pip install -e .`

```python
from admit import Ontology, AdmitM1Frozen, NODE17, frozen_support_map, nmin_rule, mask_arc

onto = Ontology.from_repo(NODE17)            # 17 cell types with their required markers
adm = onto.admissible(panel)                 # types the panel can support (boolean, 17)
gmap, S, _ = frozen_support_map(onto, adm, y_train, n_cal=len(y_cal), n_min=nmin_rule(0.10))
m = AdmitM1Frozen(onto, 0.10, gmap, S).fit(onto.aggregate(P_cal), y_cal)
sets = m.predict(onto.aggregate(P_test))     # (n, 17) boolean prediction sets
```

- `panel`: list of measured marker names. `y_train`, `y_cal`: integer labels indexing `NODE17`.
- `P_cal`, `P_test`: class probabilities (n × 17) from any classifier, computed with the markers outside the panel
  masked (`mask_arc`). Calibration cells must come from patients not used for training.
- An empty set means no supported type passed its threshold.
- Worked example: `python examples/toy_demo.py --delete CD8`.
