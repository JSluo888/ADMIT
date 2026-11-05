"""Fig 4 downstream biology (ADMIT): shared cohort configs, cell-type definitions, survival helpers.

Everything analysis-defining lives in PLAN_GENERIC and COHORTS below; write_plan.py serialises them to
$ADMIT_RUNS/downstream/analysis_plan.json and <cohort>/analysis_plan.json BEFORE any outcome model is fit, and
run_cohort.py refuses to run if the on-disk plan hash differs from the code (no silent post-hoc changes).

No patient identifiers are printed or written: patients are referred to by their row index in clinical.csv (pidx).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get('ADMIT_DATA_ROOT', str(REPO / 'data')))          # standard-schema cohorts
OUT = Path(os.environ.get('ADMIT_RUNS', str(REPO / 'runs'))) / 'downstream'  # outputs
TIER1 = DATA / 'tier1'
TIER2 = DATA / 'tier2_std'
TIER3 = DATA / 'tier3_std'  # additional cohorts
sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------------------------------------------
# 17-node ontology (same as ADMIT engine): parent map
# ---------------------------------------------------------------------------------------------------------------
T_SUBTREE = ['T_Lineage', 'Cytotoxic_T_Cell', 'Helper_T_Cell', 'Regulatory_T_Cell']
IMMUNE_SUBTREE = ['Immune', 'Immune_Other', 'T_Lineage', 'Cytotoxic_T_Cell', 'Helper_T_Cell', 'Regulatory_T_Cell',
                  'B_Cell', 'Plasma_Cell', 'NK_Cell', 'Dendritic_Cell', 'Myeloid', 'Macrophage_CD163neg',
                  'Macrophage_CD163pos', 'Neutrophil']
MYELOID_SUBTREE = ['Myeloid', 'Macrophage_CD163neg', 'Macrophage_CD163pos']

# default cell-type definitions on node17 (cohort configs may override with author labels; see COHORTS)
CELLTYPES = {
    'cyto_T': {'label': 'Cytotoxic T (CD8)', 'node17': ['Cytotoxic_T_Cell']},
    'treg': {'label': 'Regulatory T', 'node17': ['Regulatory_T_Cell']},
    'helper_T': {'label': 'Helper T', 'node17': ['Helper_T_Cell']},
    'pooled_T': {'label': 'T lineage (pooled)', 'node17': T_SUBTREE},
    'B': {'label': 'B cell', 'node17': ['B_Cell']},
    'myeloid_mac': {'label': 'Macrophage / myeloid', 'node17': MYELOID_SUBTREE},
}
CT_ORDER = ['cyto_T', 'treg', 'helper_T', 'pooled_T', 'B', 'myeloid_mac']

PLAN_GENERIC = {
    'title': 'ADMIT, Fig 4: does the cell-type resolution a panel supports change biological conclusions?',
    'status': 'prespecified before any outcome model was fit on any cohort in this analysis',
    'data_rule': 'a cohort is analysed only when data/<tier>/<cohort>/READY exists (checked at run time; the READY '
                 'mtime and file sha256 of cells.parquet/clinical.csv are recorded in every result)',
    'patients': {
        'eligibility': 'endpoint time > 0 and event in {0,1}; cohort-specific exclusions (see cohort plan); '
                       '>= 200 ontology-mapped cells (node17 != UNMAPPED) per patient',
        'aggregation': 'all cells of a patient pooled over its samples/cores',
        'identifiers': 'never printed or written; patients are indexed by row of clinical.csv (pidx)',
    },
    'covariate': {
        'primary': 'proportion of the cell type among the patient\'s ontology-mapped cells (node17 != UNMAPPED); '
                   'UNMAPPED cells (author "Other"/unassigned) are excluded from numerator and denominator in all '
                   'arms so that author labels, forced labels and ADMIT labels are compared on identical cells',
        'secondary_denominator': 'proportion among immune cells (node17 in the Immune subtree); patients with < 20 '
                                 'immune cells are set missing for this specification',
        'scaling': 'standardised to mean 0, SD 1 within the analysis set of each model; HR is per +1 SD',
    },
    'cox': {
        'software': 'lifelines CoxPHFitter, Efron ties, no penalty (penalizer 0.01 only if the unpenalised fit '
                    'fails to converge; flagged)',
        'adjustment': 'age (per 10 years, continuous) and stage (categorical dummies; any level with < 20 patients '
                      'in the analysis set is merged into its adjacent level with more patients) where available; '
                      'complete-case on age/stage; unadjusted model reported as sensitivity',
        'outputs': 'HR per +1 SD with 95% Wald CI and p; Harrell C of the adjusted linear predictor and of the '
                   'cell-type covariate alone (both in-sample); Schoenfeld-residual PH test p for the covariate '
                   '(rank time transform; diagnostic only)',
    },
    'km': 'secondary: median split (high = proportion strictly above the median of the analysis set), log-rank p, '
          'KM median survival per group; group sizes reported (a zero median makes the split zero vs non-zero)',
    'multiplicity': {
        'primary_family': 'Benjamini-Hochberg over the cell types available in the cohort (cyto_T, treg, helper_T, '
                          'pooled_T, B, myeloid_mac) for the primary endpoint, primary denominator, adjusted Cox p',
        'secondary_families': 'BH applied separately within every other (endpoint, denominator, model) block; '
                              'labelled secondary',
        'reporting': 'raw and BH-adjusted p always reported together; Part B and the forest plot are estimation '
                     '(bias/attenuation with CIs), not additional hypothesis tests; all of Fig 4 is exploratory',
    },
    'part_B_counterfactual': {
        'panels': {'full': 'all markers', 'noCD8': 'the CD8 channel masked', 'noFOXP3': 'the FOXP3 channel masked'},
        'masking': 'features = arcsinh(intensity / cofactor) z-scored with classifier-training means/SDs; a masked '
                   'marker is set to 0 (= training mean; the same mean-imputation encoding as ADMIT engine X_arc)',
        'splits': '5 seeds (0-4). Eligible patients (primary endpoint) are permuted (numpy default_rng(1000+seed)) '
                  'and halved (A, B). Direction 1: develop on A, evaluate on B (the prespecified 50/50 primary). '
                  'Direction 2: develop on B, evaluate on A. Within the development half, 60% of patients fit the '
                  'classifier and 40% calibrate ADMIT (patient-level).',
        'classifier': 'XGBoost multi:softprob on the 17-node labels present in the fit cells (UNMAPPED excluded), '
                      '300 trees, depth 6, eta 0.1, hist, CUDA, random_state = seed; up to 2,000 random mapped cells '
                      'per fit patient; trained on the full panel only',
        'admit': 'ADMIT + M1 (admit.panel_matched.AdmitM1, alpha = 0.10, n_min = 9): calibration cells (up to '
                 '2,000 mapped cells per calibration patient) masked to the same panel and rescored by the frozen '
                 'classifier; prediction sets on all mapped cells of the evaluation patients',
        'admit_point_label_rule': 'finest consensus node of the prediction set: the deepest node c that is an '
                                  'ancestor-or-self of at least one set member and is comparable (ancestor-or-self '
                                  'or descendant-or-self) with every set member. A chain {Immune, T_Lineage} reads '
                                  'T_Lineage; {T_Lineage, Helper_T, Treg} reads T_Lineage; {Helper_T, Treg} reads '
                                  'T_Lineage; {Immune, T_Lineage, Myeloid} reads Immune; sets spanning two top-level '
                                  'lineages are unresolved; empty sets abstain. ADMIT pooled T = cells whose '
                                  'consensus node is in the T subtree. Abstained/unresolved cells stay in the '
                                  'denominator.',
        'arms': {
            'GT': 'author labels (ground truth) on the evaluation patients',
            'forced_full': 'argmax of the full-panel classifier (sanity: classifier error without masking)',
            'forced_noCD8': 'argmax with CD8 masked: the forced classifier still outputs cytotoxic T',
            'admit_noCD8': 'ADMIT consensus labels with CD8 masked: cytotoxic T is inadmissible, so only pooled T '
                           '(and Treg/helper T where admissible) is reported',
            'forced_noFOXP3': 'argmax with FOXP3 masked (Treg)',
            'admit_noFOXP3': 'ADMIT consensus labels with FOXP3 masked (Treg inadmissible; pooled T reported)',
            'admit_full': 'ADMIT consensus labels on the full panel (reference)',
        },
        'estimands': 'adjusted Cox HR per +1 SD on the evaluation half for cyto_T, treg, pooled_T under every arm '
                     'where defined; same model specification as Part A',
        'comparisons': {
            'cyto_T': 'forced_noCD8 vs GT (bias/attenuation of the forced subtype estimate)',
            'treg': 'forced_noFOXP3 vs GT',
            'pooled_T': 'admit_noCD8 vs GT pooled T (ADMIT reports the T-level quantity faithfully)',
            'resolution': 'GT cyto_T vs GT pooled_T (what is lost by reporting at T level; the biological question)',
        },
        'error_metrics': 'per-patient absolute error of the cyto_T (and treg, pooled_T) fraction vs GT in percentage '
                         'points (mean, median, 90th percentile), mean signed error, Spearman rho, fraction of '
                         'patients with relative error > 25%',
        'honesty_metrics': 'cell level on evaluation cells: fraction of ADMIT sets containing an inadmissible node '
                           '(must be 0), exact coverage of the revised label for GT cytotoxic T cells, distribution '
                           'of ADMIT consensus nodes for GT cytotoxic T cells, mean |C|',
        'uncertainty': 'per split: Cox 95% CI. Spread over the 5 seeds (direction 1 = primary; both directions '
                       'reported). Paired patient bootstrap (B = 1000, resample evaluation patients with '
                       'replacement, refit both models, re-standardise) of delta log HR for each comparison on '
                       'the seed-0 cross-fitted full cohort',
        'cross_fit': 'secondary: combining both directions gives every eligible patient a prediction from a model '
                     'not trained or calibrated on them; Cox on the whole cohort (prespecified display estimate for '
                     'the forest plot = seed 0 cross-fit; spread over seeds 0-4 shown)',
    },
    'amendment_1': {
        'at': 'see amendments[0].at in analysis_plan.json (written before the Part B run started)',
        'trigger': 'classifier diagnostic only (no Part B survival output had been viewed): on Cords seed 0, the '
                   'full-panel classifier with CD8 mean-imputed assigned no evaluation cell to cytotoxic T (GT CD8 '
                   'cells went to Stromal 31%, Helper T 22%, Epithelial 20%, Myeloid 17%), so the forced-label '
                   'cytotoxic-T proportion is identically ~0 and its HR is not estimable',
        'added_arms': {
            'forcedRT_<cf>': 'a second forced classifier retrained on the reduced panel (the masked marker set to 0 '
                             'in training too) with the full label set, so it still outputs cytotoxic T (or Treg) '
                             'from the remaining markers; same fit cells, hyper-parameters and seed',
            'admitRT_<cf>': 'ADMIT + M1 on top of that retrained classifier (same calibration patients)',
        },
        'status': 'the mean-imputed arms stay the prespecified primary; the retrained arms are secondary and are '
                  'reported alongside, with the same estimands, splits, metrics and bootstrap comparisons',
    },
    'amendment_2': {
        'reason': 'extend the prespecified rules to the Tier-2 / later cohorts before any of them is analysed',
        'binary_endpoints': 'binary response (pCR, ICI response): logistic regression (statsmodels Logit), OR per +1 SD '
                            'with Wald 95% CI and p, adjusted for the cohort covariates (treatment arm where the cohort '
                            'is a trial); AUC of the linear predictor and of the covariate alone replace Harrell C; no KM',
        'minimum_events': 'an outcome model is fit only with >= 10 events (survival) or >= 10 patients in the smaller '
                          'outcome class (binary); otherwise reported as not estimable. Cohorts whose every endpoint '
                          'fails this are not analysed (MCC validation: 5 OS events)',
        'stage_parsing': {'numeric': 'numeric code used as categorical', 'roman': 'I*/II*/III*/IV* -> 1/2/3/4 '
                          '(Unknown/Unstaged/Error -> missing)', 'tnm_breast': 'AJCC-7 anatomic stage group from '
                          'the (y)pT/(y|c)pN/M string: M1 -> 4; T1N0 -> 1; T1N1, T2N0, T2N1, T3N0 -> 2; any N2/N3, '
                          'T3N1+, T4 -> 3; Tis/T0 treated as T1; unknown T or N (and not M1) -> missing'},
        'transform_arcsinh_p95': 'cohorts whose intensities are not raw IMC counts use arcsinh(x / s_m) with s_m the '
                                 'per-marker 95th percentile over all mapped cells (label-free; s_m floored at 1e-6); '
                                 'monotone, so tree classifiers are unaffected except for where mean imputation lands',
        'unavailable_celltypes': 'a cell type without a clean author label (e.g. no FOXP3 in the panel) is not '
                                 'analysed and not counted in the BH family',
        'small_cohorts': 'Part B is run on every analysed cohort; split-half Cox models with < 30 events in the '
                         'evaluation half are reported but flagged underpowered; the cross-fit estimate uses all '
                         'patients',
    },
    'part_C_forest': 'per cohort x {cyto_T, treg, pooled_T}: GT HR (Part A, all eligible patients), GT HR and '
                     'forced-label HR and ADMIT pooled-T HR from the seed-0 cross-fit (same patients), plus seed '
                     'spread. Exploratory; multiplicity stated in the caption.',
    'replication_rule': 'a finding "replicates" in a cohort if the adjusted HR has the same direction and BH-adjusted '
                        'p < 0.05 in the primary family; same direction with raw p < 0.05 is "nominal"; otherwise '
                        '"not replicated". Non-replication is reported.',
    'earlier_reference': 'Orion PFS (n=39, 24 events) from v5.3 model proportions (trained on 15 of the 39 patients): '
                         'cyto_T log-rank p=0.0077, Treg p=0.044, pooled T p=0.70. Shown only as an earlier reference, '
                         'labelled so; not re-estimated until gating for the remaining Orion patients is available.',
}

# ---------------------------------------------------------------------------------------------------------------
# cohort configs (written into the cohort plan before the cohort is analysed)
# ---------------------------------------------------------------------------------------------------------------
COHORTS = {
    'cords2024_nsclc_imc': {
        'tier': 'tier1', 'display': 'Cords 2024 NSCLC (IMC)',
        'endpoints': {'os': {'primary': True, 'label': 'OS'},
                      'dfs': {'primary': False, 'label': 'DFS (recurrence only; relapses without DFS time '
                                                         'excluded: outcome-dependent missingness, see QC)'}},
        'covariates': ['age', 'stage'],
        'stage_note': 'numeric code 1-7 without a dictionary in the deposit; used as categorical',
        'exclude_flag_regex': 'mesothelioma',
        'transform_cofactor': 1.0,
        'drop_markers': ['Iridium_191', 'Iridium_193'],
        'celltype_overrides': {},
        'celltype_notes': {'myeloid_mac': 'author "Myeloid" (CD68/HLA-DR high; no macrophage split; panel lacks '
                                          'CD163/CD11c)',
                           'helper_T': 'all non-Treg CD4 subsets (CD4, IDO_CD4, PD1_CD4, TCF1/7_CD4, ki67_CD4)'},
        'counterfactuals': ['noCD8', 'noFOXP3'],
    },
    'danenberg2022_metabric_imc': {
        'tier': 'tier1', 'display': 'Danenberg 2022 METABRIC breast (IMC)',
        'endpoints': {'os': {'primary': True, 'label': 'OS'},
                      'dss': {'primary': False, 'label': 'DSS (deaths of unknown cause excluded)'}},
        'covariates': ['age', 'stage'],
        'stage_note': 'Rueda stage 0-4, NA common; categorical with sparse-level merging; complete case',
        'exclude_flag_regex': None,
        'transform_cofactor': 1.0,
        'drop_markers': ['DNA1', 'DNA2'],
        'celltype_overrides': {'treg': {'label_fine': ['T_{Reg} & T_{Ex}'],
                                        'label': 'Treg & Tex (author mixed class)'}},
        'celltype_notes': {'treg': 'no clean Treg label: the author class "T_{Reg} & T_{Ex}" (node17 T_Lineage) mixes '
                                   'regulatory and exhausted T cells; analysed as such and flagged',
                           'myeloid_mac': 'author "Macrophages" (node17 Myeloid)',
                           'pooled_T': 'CD8 + CD4 + Treg&Tex; the mixed author class "CD4+ T cells & APCs" maps to '
                                       'Immune and is not counted as T'},
        'counterfactuals': ['noCD8'],
        'counterfactual_notes': 'noFOXP3 not run: no clean Treg ground truth',
    },
    'fischer2023_breast_imc': {
        'tier': 'tier2', 'display': 'Fischer 2023 breast (IMC; Basel/Zurich)',
        'endpoints': {'os': {'primary': True, 'label': 'OS (all-cause; os_event = 1 - Status_OS, builder-coded)'}},
        'covariates': ['age', 'stage'], 'stage_parser': 'tnm_breast',
        'exclude_flag_regex': None, 'transform': 'arcsinh_p95', 'transform_cofactor': None,
        'drop_regex': 'Histone|Iridium|DNA',
        'celltype_overrides': {'treg': {'unavailable': True, 'label': 'Regulatory T (not available: no FOXP3 / no Treg label)'}},
        'celltype_notes': {'myeloid_mac': 'author stroma clusters mapped to Myeloid', 'cyto_T': 'stroma_cluster_22',
                           'helper_T': 'stroma_cluster_10'},
        'counterfactuals': ['noCD8'],
        'note': 'Meyer 2025 kept as a separate cohort with its own OS definition',
    },
    'meyer2025_tnbc_imc': {
        'tier': 'tier2', 'display': 'Meyer 2025 TNBC (IMC)',
        'endpoints': {'os': {'primary': True, 'label': 'OS (Meyer status_OS)'},
                      'dfs': {'primary': False, 'label': 'DFS'}},
        'covariates': ['age', 'stage'], 'stage_parser': 'tnm_breast',
        'exclude_flag_regex': None, 'transform': 'arcsinh_p95', 'transform_cofactor': None,
        'drop_regex': 'Histone|Iridium|DNA',
        'celltype_overrides': {'treg': {'unavailable': True, 'label': 'Regulatory T (not available: no FOXP3 / no Treg label)'}},
        'celltype_notes': {'myeloid_mac': 'author Monocyte (node17 Myeloid)'},
        'counterfactuals': ['noCD8'],
    },
    'quiche_tnbc_mibi_spain': {
        'tier': 'tier2', 'display': 'QUICHE TNBC MIBI (Spain)',
        'endpoints': {'rfs': {'primary': True, 'label': 'RFS'}},
        'covariates': ['age'], 'stage_note': 'no stage in the deposit',
        'exclude_flag_regex': None, 'transform': 'arcsinh_p95', 'transform_cofactor': None, 'drop_regex': None,
        'celltype_overrides': {},
        'celltype_notes': {'treg': 'very sparse (249 cells cohort-wide); flagged',
                           'myeloid_mac': 'CD14 monocytes + CD68/CD163 macrophages (Myeloid subtree)'},
        'counterfactuals': ['noCD8', 'noFOXP3'],
    },
    'quiche_tnbc_mibi_stanford': {
        'tier': 'tier2', 'display': 'QUICHE TNBC MIBI (Stanford)',
        'endpoints': {'rfs': {'primary': True, 'label': 'RFS'}},
        'covariates': ['age', 'stage'], 'stage_parser': 'roman', 'stage_col': 'stage_simple',
        'exclude_flag_regex': None, 'transform': 'arcsinh_p95', 'transform_cofactor': None, 'drop_regex': None,
        'celltype_overrides': {},
        'celltype_notes': {'treg': 'sparse (750 cells cohort-wide); flagged',
                           'myeloid_mac': 'CD14 monocytes + CD68/CD163 macrophages (Myeloid subtree)'},
        'counterfactuals': ['noCD8', 'noFOXP3'],
    },
    'launonen2022_hgsc_tcycif': {
        'tier': 'tier1', 'display': 'Launonen 2022 HGSC (t-CyCIF)',
        'endpoints': {'os': {'primary': True, 'label': 'OS'}, 'pfi': {'primary': False, 'label': 'PFI'}},
        'covariates': [], 'stage_note': 'no stage/age in the public deposit',
        'exclude_flag_regex': None, 'transform': 'arcsinh_p95', 'transform_cofactor': None, 'drop_regex': None,
        'celltype_overrides': {},
        'celltype_notes': {'myeloid_mac': 'IBA1/CD11c/CD163 macrophages (Myeloid subtree)'},
        'counterfactuals': ['noCD8', 'noFOXP3'],
    },
    'mcc_discovery_haist_codex': {
        'tier': 'tier2', 'display': 'MCC discovery (CODEX)',
        'endpoints': {'pfs': {'primary': True, 'label': 'PFS on first-line ICB'},
                      'os': {'primary': False, 'label': 'OS (9 events: below the 10-event rule)'},
                      'response': {'primary': False, 'label': 'ICI objective response', 'kind': 'binary',
                                   'col': 'response_binary'}},
        'covariates': ['age'], 'stage_note': 'no stage in the deposit',
        'exclude_flag_regex': None, 'transform': 'arcsinh_p95', 'transform_cofactor': None, 'drop_regex': 'DAPI',
        'celltype_overrides': {},
        'celltype_notes': {'myeloid_mac': 'author macrophages (node17 Myeloid)'},
        'counterfactuals': ['noCD8', 'noFOXP3'],
        'note': 'primary endpoint PFS because OS has 9 events (< 10); MCC validation (5 OS events) not analysed',
    },
    'kulasinghe2026_nsclc_pcf': {
        'tier': 'tier2', 'display': 'Kulasinghe 2026 NSCLC (PhenoCycler)',
        'endpoints': {'os': {'primary': True, 'label': 'OS'}, 'rfs': {'primary': False, 'label': 'RFS'}},
        'covariates': ['stage'], 'stage_parser': 'roman', 'stage_note': 'no age in the deposit',
        'exclude_flag_regex': None, 'transform': 'arcsinh_p95', 'transform_cofactor': None, 'drop_regex': 'DAPI',
        'celltype_overrides': {'treg': {'unavailable': True, 'label': 'Regulatory T (not available: no FOXP3 / no Treg label)'},
                               'helper_T': {'label_fine': ['CD4_cells', 'CD4_TFH'],
                                            'label': 'Helper T (author CD4 classes; node17 T_Lineage)'}},
        'celltype_notes': {'myeloid_mac': 'Macrophages (CD163neg/pos)'},
        'counterfactuals': ['noCD8'],
    },
    'monkman2026_nsclc_pcf': {
        'tier': 'tier2', 'display': 'Monkman 2026 NSCLC (PhenoCycler; MAPS-transferred labels)',
        'endpoints': {'os': {'primary': True, 'label': 'OS'}},
        'covariates': [], 'stage_note': 'no stage/age in the deposit',
        'exclude_flag_regex': None, 'transform': 'arcsinh_p95', 'transform_cofactor': None, 'drop_regex': 'dapi',
        'celltype_overrides': {'treg': {'unavailable': True, 'label': 'Regulatory T (not available: no Treg label)'},
                               'helper_T': {'label_fine': ['CD4_cells'],
                                            'label': 'Helper T (author CD4 class; node17 T_Lineage)'}},
        'celltype_notes': {'myeloid_mac': 'Macrophages (node17 Myeloid)'},
        'counterfactuals': ['noCD8'],
        'note': 'added in round 2 (E11 real pair, E13 NSCLC pool); not analysed in round 1',
    },
    # ---- additional cohorts: configs written before any outcome model was fit on these cohorts ----
    'piyadasa2025_glioma_mibi_immune': {
        'tier': 'tier2', 'display': 'Piyadasa 2025 glioma (MIBI, immune panel)',
        'endpoints': {'os': {'primary': True, 'label': 'OS'}},
        'covariates': ['age', 'stage'], 'stage_col': 'grade_who', 'stage_parser': 'numeric',
        'stage_note': 'no stage in glioma: WHO grade (2/3/4) used as the stage covariate; mixed grades (e.g. "2;3") '
                      'are missing (complete case)',
        'exclude_flag_regex': None, 'transform': 'arcsinh_p95', 'transform_cofactor': None, 'drop_regex': None,
        'celltype_overrides': {},
        'celltype_notes': {'treg': 'very sparse (373 cells cohort-wide); flagged',
                           'myeloid_mac': 'myeloid / microglia-macrophage classes (Myeloid subtree)'},
        'counterfactuals': ['noCD8', 'noFOXP3'],
        'note': 'adult + paediatric gliomas pooled; all samples of a patient pooled (primary and recurrence)',
    },
    'chl_mibi_stanford': {
        'tier': 'tier2', 'display': 'Classic Hodgkin lymphoma (MIBI, Stanford)',
        'endpoints': {'pfs': {'primary': True, 'label': 'PFS (10 events: at the minimum)'}},
        'covariates': ['stage'], 'stage_col': 'stage_num', 'stage_parser': 'numeric',
        'stage_note': 'Ann Arbor stage number; no age in the deposit',
        'exclude_flag_regex': None, 'transform': 'arcsinh_p95', 'transform_cofactor': None, 'drop_regex': None,
        'celltype_overrides': {},
        'celltype_notes': {'myeloid_mac': 'M1 (Myeloid) + M2 (Macrophage_CD163pos)'},
        'counterfactuals': ['noCD8', 'noFOXP3'],
        'note': 'small (52 patients, 10 PFS events): underpowered, reported as such',
    },
    'jackson2020_breast_imc': {
        'tier': 'tier3', 'display': 'Jackson 2020 breast (IMC, Basel)',
        'endpoints': {'os': {'primary': True, 'label': 'OS'},
                      'dss': {'primary': False, 'label': 'DSS (deaths of unknown cause excluded)'}},
        'covariates': ['age', 'stage'], 'stage_parser': 'tnm_breast',
        'sample_filter': {'file': 'samples.csv', 'col': 'core_type', 'keep': 'tumour'},
        'exclude_flag_regex': None, 'transform': 'arcsinh_p95', 'transform_cofactor': None, 'drop_regex': None,
        'celltype_overrides': {k: {'unavailable': True, 'label': v} for k, v in {
            'cyto_T': 'Cytotoxic T (not available: no CD8 in panel)', 'treg': 'Regulatory T (not available: no FOXP3)',
            'helper_T': 'Helper T (not available: no CD4)',
            'pooled_T': 'T lineage (not available: author T metaclusters lack CD3/CD45 signal -> UNMAPPED)',
            'B': 'B cell (not available: author B metaclusters contradict CD20/CD3 -> Immune)'}.items()},
        'celltype_notes': {'myeloid_mac': 'author MC4/MC6 macrophages (node17 Myeloid)'},
        'counterfactuals': [],
        'note': 'Basel tumour cores only (Zurich has no outcomes); panel has no CD4/CD8/FOXP3: Part A only',
    },
    'girnius2026_tcycif': {
        'tier': 'tier3', 'display': 'Girnius 2026 carcinoma TMA (t-CyCIF)',
        'endpoints': {'os': {'primary': True, 'label': 'OS'}},
        'covariates': ['age', 'stage'], 'stage_parser': 'numeric',
        'exclude_flag_regex': None, 'transform': 'arcsinh_p95', 'transform_cofactor': None, 'drop_regex': None,
        'celltype_overrides': {'B': {'unavailable': True, 'label': 'B cell (not available: no B label / no CD20)'}},
        'celltype_notes': {'pooled_T': 'author CD4 / CD8 / Treg classes (panel has no CD3: T nodes not admissible '
                                       'under the panel, so ADMIT cannot report T cells at all)',
                           'myeloid_mac': 'CD68 / CD206 macrophages (node17 Myeloid)'},
        'counterfactuals': ['noCD8', 'noFOXP3'],
        'note': 'values author-rescaled to [0, 1]; LN-metastasis cores kept in cells (pooled per patient)',
    },
    'meylan2026_gbm_codex': {
        'tier': 'tier3', 'display': 'Meylan 2026 recurrent GBM (CODEX, pre-treatment)',
        'endpoints': {'os': {'primary': True, 'label': 'OS from CAN-3110 injection'},
                      'pfs': {'primary': False, 'label': 'PFS'}},
        'covariates': ['age'], 'stage_note': 'no stage in glioma',
        'exclude_flag_regex': None, 'transform': 'arcsinh_p95', 'transform_cofactor': None, 'drop_regex': 'dapi',
        'celltype_overrides': {},
        'celltype_notes': {'myeloid_mac': 'Mac_CD163 / Mac_CD68 / Mac_HLADR'},
        'counterfactuals': ['noCD8', 'noFOXP3'],
        'note': '16 patients: underpowered, reported as such; CD45 not stained for 5 patients',
    },
    'hoch2022_melanoma_imc': {
        'tier': 'tier3', 'display': 'Hoch 2022 metastatic melanoma (IMC, protein panel)',
        'endpoints': {'pfs': {'primary': True, 'label': 'PFS from surgery (progression or last PET)'},
                      'os': {'primary': False, 'label': 'OS from surgery'}},
        'covariates': ['stage'], 'stage_parser': 'roman', 'stage_note': 'AJCC III / IV; "III or IV" missing; no '
                                                                         'numeric age (age range only)',
        'exclude_flag_regex': None, 'transform': 'arcsinh_p95', 'transform_cofactor': None, 'drop_regex': 'DNA',
        'celltype_overrides': {},
        'celltype_notes': {'myeloid_mac': 'author myeloid (node17 Myeloid)'},
        'counterfactuals': ['noCD8', 'noFOXP3'],
        'note': 'mixed pre-surgery treatments (ICI / targeted / naive); 5 patients with image-inconsistent outcomes '
                'excluded; control-tissue images have no outcome',
    },
    'neotrip_tnbc_imc': {
        'tier': 'tier2', 'display': 'NeoTRIP TNBC trial (IMC), baseline biopsies',
        'endpoints': {'pcr': {'primary': True, 'label': 'pCR', 'kind': 'binary', 'col': 'pcr'}},
        'covariates': ['arm'], 'arm_col': 'treatment',
        'sample_filter': {'file': 'samples.csv', 'col': 'biopsy_phase', 'keep': 'Baseline'},
        'exclude_flag_regex': None, 'transform': 'arcsinh_p95', 'transform_cofactor': None, 'drop_regex': 'DNA',
        'celltype_overrides': {},
        'celltype_notes': {'myeloid_mac': 'M2 macrophages (node17 Macrophage_CD163pos)'},
        'counterfactuals': ['noCD8', 'noFOXP3'],
        'note': 'baseline (pre-treatment) images only; adjusted for randomised arm (chemo vs chemo + atezolizumab)',
    },
}


def cohort_dir(name: str) -> Path:
    c = COHORTS[name]
    return {'tier1': TIER1, 'tier2': TIER2, 'tier3': TIER3}[c['tier']] / name


def ready_info(name: str) -> dict:
    d = cohort_dir(name)
    r = d / 'READY'
    return {'ready': r.exists(), 'ready_mtime': (r.stat().st_mtime if r.exists() else None)}


def sha256(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def plan_for(name: str) -> dict:
    c = dict(COHORTS[name])
    cts = {}
    for k in CT_ORDER:
        d = dict(CELLTYPES[k])
        if k in c['celltype_overrides']:
            d = {**{kk: vv for kk, vv in d.items() if kk != 'node17'}, **c['celltype_overrides'][k]}
        if k in c.get('celltype_notes', {}):
            d['note'] = c['celltype_notes'][k]
        cts[k] = d
    return {'cohort': name, 'generic_plan': 'runs/downstream/analysis_plan.json', 'config': c,
            'celltypes': cts, 'immune_denominator_node17': IMMUNE_SUBTREE}


def plan_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if not math.isfinite(v) else v
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))

    def clean(o):
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o

    path.write_text(json.dumps(clean(json.loads(json.dumps(obj, default=conv))), indent=1))


# ---------------------------------------------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------------------------------------------

def load_clinical(name: str) -> pd.DataFrame:
    clin = pd.read_csv(cohort_dir(name) / 'clinical.csv', low_memory=False)
    clin['pidx'] = np.arange(len(clin))
    return clin


def load_cells(name: str, with_markers: bool):
    import re
    import pyarrow.parquet as pq
    d = cohort_dir(name)
    mdf = pd.read_csv(d / 'markers.tsv', sep='\t', keep_default_na=False)
    cfg = COHORTS[name]
    drop = set(cfg.get('drop_markers', []))
    rx = cfg.get('drop_regex')
    marker_cols = [m for m in mdf.source_name if m not in drop and not (rx and re.search(rx, m, re.I))]
    sf = cfg.get('sample_filter')
    cols = ['patient_id', 'node17', 'label_fine'] + (['sample_id'] if sf else []) + (marker_cols if with_markers else [])
    t = pq.read_table(d / 'cells.parquet', columns=cols)
    keep = None
    if sf:
        smp = pd.read_csv(d / sf['file'], low_memory=False)
        ok = set(smp.loc[smp[sf['col']] == sf['keep'], 'sample_id'].astype(str))
        sid = np.asarray(t.column('sample_id').to_numpy(zero_copy_only=False)).astype(str)
        keep = np.isin(sid, list(ok))

    def col(c, dtype=str):
        v = np.asarray(t.column(c).to_numpy(zero_copy_only=False))
        v = v.astype(dtype)
        return v[keep] if keep is not None else v
    out = {'patient_id': col('patient_id'), 'node17': col('node17'), 'label_fine': col('label_fine')}
    if with_markers:
        n = int(keep.sum()) if keep is not None else t.num_rows
        X = np.empty((n, len(marker_cols)), dtype=np.float32)
        for j, m in enumerate(marker_cols):
            X[:, j] = col(m, np.float32)
        out['X'] = X
        vocab = {k: (str(v) if str(v) not in ('', 'nan', 'None') else 'NA')
                 for k, v in zip(mdf.source_name, mdf.vocabulary_name)}
        out['marker_cols'] = marker_cols
        out['marker_vocab'] = [vocab.get(m, 'NA') for m in marker_cols]
    return out


def available_celltypes(name: str) -> list:
    p = plan_for(name)['celltypes']
    return [k for k in CT_ORDER if not p[k].get('unavailable')]


def celltype_masks(name: str, node17: np.ndarray, label_fine: np.ndarray) -> dict:
    """Boolean cell masks per available cell type (ground truth), per plan_for(name)."""
    p = plan_for(name)['celltypes']
    out = {}
    for k, d in p.items():
        if d.get('unavailable'):
            continue
        if 'label_fine' in d:
            out[k] = np.isin(label_fine, d['label_fine'])
        else:
            out[k] = np.isin(node17, d['node17'])
    return out


def _roman_stage(v):
    if not isinstance(v, str):
        return np.nan
    v = v.strip().upper().replace('STAGE', '')
    if v.startswith('1'):
        v = 'I' + v[1:]
    for pre, g in (('IV', 4), ('III', 3), ('II', 2), ('I', 1)):
        if v.startswith(pre):
            return g
    return np.nan


def _tnm_breast_stage(v):
    import re
    if not isinstance(v, str):
        return np.nan
    if re.search(r'\bM1\b', v):
        return 4
    mt = re.search(r'T(is|\d)', v)
    mn = re.search(r'N(\d)', v)
    if not mt or not mn:
        return np.nan
    T = 1 if mt.group(1) == 'is' else max(1, int(mt.group(1)))
    N = int(mn.group(1))
    if T == 4 or N >= 2 or (T == 3 and N >= 1):
        return 3
    if T == 1 and N == 0:
        return 1
    return 2


def parse_stage(series: pd.Series, parser: str) -> pd.Series:
    if parser == 'roman':
        return series.map(_roman_stage).astype(float)
    if parser == 'tnm_breast':
        return series.map(_tnm_breast_stage).astype(float)
    return pd.to_numeric(series, errors='coerce')


# ---------------------------------------------------------------------------------------------------------------
# survival
# ---------------------------------------------------------------------------------------------------------------

def merge_sparse_levels(s: pd.Series, min_n: int = 20) -> pd.Series:
    """Categorical stage: merge any level with < min_n patients into its adjacent level with more patients."""
    s = s.copy()
    while True:
        vc = s.dropna().value_counts()
        if len(vc) <= 1:
            break
        levels = sorted(vc.index)
        small = [l for l in levels if vc[l] < min_n]
        if not small:
            break
        l = min(small, key=lambda v: vc[v])
        i = levels.index(l)
        nb = [levels[j] for j in (i - 1, i + 1) if 0 <= j < len(levels)]
        tgt = max(nb, key=lambda v: vc[v])
        s[s == l] = tgt
    return s


def design(df: pd.DataFrame, xcol: str, adjust: list, tcol: str, ecol: str):
    """Complete-case design frame: z (standardised xcol), age10, stage / arm dummies, T, E."""
    need = [xcol, tcol, ecol] + list(adjust)
    d = df[need].dropna().copy()
    out = pd.DataFrame(index=d.index)
    x = d[xcol].astype(float)
    sd = x.std(ddof=1)
    out['z'] = (x - x.mean()) / sd if sd > 0 else np.nan
    if 'age' in adjust:
        out['age10'] = d['age'].astype(float) / 10.0
    if 'stage' in adjust:
        st = merge_sparse_levels(d['stage'].astype(float))
        dm = pd.get_dummies(st.astype(int).astype(str), prefix='stage', drop_first=True, dtype=float)
        out = pd.concat([out, dm], axis=1)
    if 'arm' in adjust:
        dm = pd.get_dummies(d['arm'].astype(str), prefix='arm', drop_first=True, dtype=float)
        out = pd.concat([out, dm], axis=1)
    out['T'] = d[tcol].astype(float)
    out['E'] = d[ecol].astype(int)
    return out, float(sd), float(x.mean())


def assoc(df, xcol, adjust, kind='surv', ph_test=True) -> dict:
    return cox(df, xcol, adjust, 'T', 'E', ph_test) if kind == 'surv' else logit(df, xcol, adjust)


def min_events_ok(E: pd.Series, kind: str) -> bool:
    E = E.dropna()
    return (E.sum() >= 10) if kind == 'surv' else (min(E.sum(), len(E) - E.sum()) >= 10)


def logit(df: pd.DataFrame, xcol: str, adjust: list) -> dict:
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score
    D, sd, mu = design(df, xcol, adjust, 'T', 'E')
    res = {'n': int(len(D)), 'events': int(D['E'].sum()), 'sd_of_covariate': sd, 'mean_of_covariate': mu,
           'adjusted_for': list(adjust), 'model': 'logistic', 'penalized': False}
    if not min_events_ok(D['E'], 'binary') or not np.isfinite(D['z']).all():
        res['error'] = 'too few patients/events or constant covariate'
        return res
    Xd = D.drop(columns=['T', 'E'])
    Xd = Xd.loc[:, Xd.std(ddof=0) > 0]
    if 'z' not in Xd:
        res['error'] = 'constant covariate'
        return res
    Xd = sm.add_constant(Xd, has_constant='add')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = sm.Logit(D['E'], Xd).fit(disp=0, maxiter=200)
    except Exception as e:  # noqa: BLE001
        res['error'] = 'fit failed: ' + repr(e)[:150]
        return res
    b, se = float(m.params['z']), float(m.bse['z'])
    res.update({'log_hr': b, 'se': se, 'hr': float(np.exp(b)), 'ci_lo': float(np.exp(b - 1.96 * se)),
                'ci_hi': float(np.exp(b + 1.96 * se)), 'p': float(m.pvalues['z']),
                'c_model': float(roc_auc_score(D['E'], m.predict(Xd))),
                'c_celltype_only': float(roc_auc_score(D['E'], np.sign(b) * D['z'])), 'effect': 'OR per +1 SD'})
    return res


def cox(df: pd.DataFrame, xcol: str, adjust: list, tcol: str, ecol: str, ph_test: bool = True) -> dict:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    D, sd, mu = design(df, xcol, adjust, tcol, ecol)
    res = {'n': int(len(D)), 'events': int(D['E'].sum()), 'sd_of_covariate': sd, 'mean_of_covariate': mu,
           'adjusted_for': list(adjust), 'penalized': False}
    if len(D) < 10 or not min_events_ok(D['E'], 'surv') or not np.isfinite(D['z']).all():
        res['error'] = 'too few patients/events or constant covariate'
        return res
    D = D.loc[:, (D.std(ddof=0) > 0) | (D.columns.isin(['T', 'E']))]
    if 'z' not in D:
        res['error'] = 'constant covariate'
        return res
    if 'E' not in D:
        D['E'] = 1
    m = None
    for pen in (0.0, 0.01):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m = CoxPHFitter(penalizer=pen).fit(D, 'T', 'E')
            res['penalized'] = pen > 0
            break
        except Exception as e:  # noqa: BLE001
            res['fit_warning'] = repr(e)[:200]
    if m is None:
        res['error'] = 'fit failed'
        return res
    s = m.summary.loc['z']
    res.update({'log_hr': float(s['coef']), 'se': float(s['se(coef)']), 'hr': float(s['exp(coef)']),
                'ci_lo': float(s['exp(coef) lower 95%']), 'ci_hi': float(s['exp(coef) upper 95%']),
                'p': float(s['p']), 'c_model': float(m.concordance_index_),
                # C of the covariate alone, oriented in the fitted direction (risk score = sign(log HR) * z)
                'c_celltype_only': float(concordance_index(D['T'], -np.sign(s['coef']) * D['z'], D['E']))})
    if ph_test:
        try:
            from lifelines.statistics import proportional_hazard_test
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                pt = proportional_hazard_test(m, D, time_transform='rank')
            res['ph_test_p'] = float(pt.summary.loc['z', 'p'])
        except Exception as e:  # noqa: BLE001
            res['ph_test_p'] = None
    return res


def cox_logHR_fast(D: pd.DataFrame, kind: str = 'surv') -> float:
    """log HR (or log OR) of z only, for bootstrap (design already built)."""
    if 'z' not in D or not np.isfinite(D['z']).all():
        return float('nan')
    if kind != 'surv':
        import statsmodels.api as sm
        Xd = sm.add_constant(D.drop(columns=['T', 'E']), has_constant='add')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return float(sm.Logit(D['E'], Xd).fit(disp=0, maxiter=200).params['z'])
        except Exception:  # noqa: BLE001
            return float('nan')
    from lifelines import CoxPHFitter
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            m = CoxPHFitter().fit(D, 'T', 'E')
        except Exception:  # noqa: BLE001
            try:
                m = CoxPHFitter(penalizer=0.01).fit(D, 'T', 'E')
            except Exception:  # noqa: BLE001
                return float('nan')
    return float(m.params_['z'])


def km_split(df: pd.DataFrame, xcol: str, tcol: str, ecol: str, keep_curve: bool = False) -> dict:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    d = df[[xcol, tcol, ecol]].dropna()
    med = float(d[xcol].median())
    hi = d[xcol] > med
    res = {'median_cut': med, 'n_high': int(hi.sum()), 'n_low': int((~hi).sum()),
           'events_high': int(d.loc[hi, ecol].sum()), 'events_low': int(d.loc[~hi, ecol].sum())}
    if hi.sum() < 5 or (~hi).sum() < 5:
        res['error'] = 'degenerate split'
        return res
    lr = logrank_test(d.loc[hi, tcol], d.loc[~hi, tcol], d.loc[hi, ecol], d.loc[~hi, ecol])
    res['logrank_p'] = float(lr.p_value)
    for g, m in (('high', hi), ('low', ~hi)):
        k = KaplanMeierFitter().fit(d.loc[m, tcol], d.loc[m, ecol])
        res[f'median_surv_{g}'] = float(k.median_survival_time_)
        if keep_curve:
            sf = k.survival_function_.iloc[:, 0]
            res[f'curve_{g}'] = {'t': sf.index.to_numpy().round(2).tolist(), 's': sf.to_numpy().round(5).tolist()}
    return res


def bh(p: list) -> list:
    p = np.asarray([np.nan if v is None else v for v in p], dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out.tolist()
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    tmp = np.empty(n)
    tmp[order] = adj
    out[ok] = tmp
    return out.tolist()


def endpoint_kind(name: str, ep: str) -> str:
    return COHORTS[name]['endpoints'][ep].get('kind', 'surv')


def endpoint_frame(name: str, clin: pd.DataFrame, ep: str) -> pd.DataFrame:
    """Per-patient frame for one endpoint after the generic + cohort eligibility rules (cell count rule applied
    later). Columns: pidx, patient_id, T, E, age, stage, arm (binary endpoints: T = 1)."""
    cfg = COHORTS[name]
    epc = cfg['endpoints'][ep]
    d = clin.copy()
    if epc.get('kind', 'surv') == 'binary':
        d['T'] = 1.0
        d['E'] = pd.to_numeric(d[epc['col']], errors='coerce')
        ok = d['E'].isin([0, 1])
    else:
        d['T'] = pd.to_numeric(d[f'{ep}_time'], errors='coerce')
        d['E'] = pd.to_numeric(d[f'{ep}_event'], errors='coerce')
        ok = (d['T'] > 0) & d['E'].isin([0, 1])
    if cfg.get('exclude_flag_regex'):
        ok &= ~d['flags'].fillna('').str.contains(cfg['exclude_flag_regex'], case=False)
    keep = ['pidx', 'patient_id', 'T', 'E']
    if 'age' in d:
        d['age'] = pd.to_numeric(d['age'], errors='coerce')
        keep.append('age')
    sc = cfg.get('stage_col', 'stage')
    if sc in d:
        d['stage'] = parse_stage(d[sc], cfg.get('stage_parser', 'numeric'))
        keep.append('stage')
    if cfg.get('arm_col'):
        d['arm'] = d[cfg['arm_col']]
        keep.append('arm')
    return d.loc[ok, keep].reset_index(drop=True)
