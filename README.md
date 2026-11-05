# ADMIT

Calibrated cell typing for spatial proteomics when markers are missing.

Each cell type has markers that must be measured before it can be called, such as CD8 for a cytotoxic T cell.
Given the measured panel, ADMIT keeps the cell types the panel supports, rescores calibration cells under that
panel and fits one conformal threshold per type. The output is a prediction set over a 17-node cell-type
hierarchy. ADMIT works on top of any classifier that returns class probabilities.

## Install

```bash
pip install -e .                          # method (numpy)
pip install -e ".[test]"                  # + demo and tests
pip install -e ".[experiments,figures]"   # + experiment and figure scripts
```

## Example

```bash
python examples/toy_demo.py --delete CD8
```

This runs ADMIT on synthetic data with CD8 removed from the panel and reports per-type coverage.

## Paper experiments

| Experiment | Script |
|---|---|
| Marker deletion | `experiments/round3/run_within3.py --cohort <name> --head xgb --seed 0` |
| Panel design | `experiments/round3/panel_design3.py` |
| Precision and recall | `experiments/round3/prf3.py` |
| Region selection | `experiments/regions/run_regions.py` |
| Case studies | `experiments/case_studies/` |
| Figures | `figures/make_fig*.py` |

Scripts read cohorts from `$ADMIT_DATA_ROOT` (default `data/`) and write results to `$ADMIT_RUNS` (default
`runs/`). Cohort settings are in `configs/cohort_registry.json`. Data sources are listed in the paper's
supplement. Some cohorts are access-controlled or in-house and are not redistributed.

## Layout

```
admit/         method: marker requirements, admissible types, calibration, prediction sets
experiments/   experiment scripts
figures/       figure scripts
configs/       cohort registry, marker panels, requirement maps
tests/         unit tests (pytest)
```

## Citation

Paper under review. Citation to follow.

## License

MIT
