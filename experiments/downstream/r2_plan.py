#!/usr/bin/env python3
"""Round-2 (usefulness of ADMIT's calibrated uncertainty downstream) prespecified plan.

Writes $ADMIT_RUNS/downstream/round2/analysis_plan_round2.json (and a copy at $ADMIT_RUNS/downstream/) BEFORE any
round-2 outcome model is fit. Never overwrites a different plan without --amend "<reason>".
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

OUT = Path(os.environ.get('ADMIT_RUNS', str(Path(__file__).resolve().parents[2] / 'runs'))) / 'downstream'

PLAN = {
    'title': 'ADMIT round 2: is ADMIT\'s calibrated uncertainty useful for downstream / clinical tasks?',
    'status': 'prespecified before any round-2 outcome model was fit (round-1 Fig 4 results were known: Part A/B '
              'survival associations in 10 cohorts; no round-2 typing or outcome analysis had been run)',
    'method': {
        'admit': 'ADMIT + M1 (admit.panel_matched.AdmitM1, supported-ancestor projection, '
                 'n_min = ceil((1-alpha)/alpha) = 9), alpha = 0.10, finite tolerances only, no backstop',
        'scorer': 'XGBoost multi:softprob on node17 labels (300 trees, depth 6, eta 0.1, hist, CUDA), as round 1; '
                  'features arcsinh(x/c) z-scored on classifier-training cells (cofactor rule of the round-1 '
                  'cohort plan); a masked marker = 0 (training mean)',
        'scorers': {'MI': 'full-panel classifier, deployment markers missing mean-imputed (impute-then-classify)',
                    'RT': 'classifier retrained on the deployment panel (missing markers absent in training), full '
                          'label set'},
        'consensus_rule': 'round-1 finest consensus node of the prediction set (-1 unresolved, -2 empty/abstain)',
        'primary_contrast': 'ADMIT-MI vs forced-MI (one full-panel model deployed everywhere, the paper protocol); '
                            'key secondary: ADMIT-RT vs forced-RT (same scorer, retrained)',
    },
    'typing_designs': {
        'D1_sim_split': 'within-cohort simulated deployment. 5 seeds (rng 2000+seed): eligible patients (primary '
                        'endpoint, >= 200 mapped cells) permuted -> classifier-fit 25%, ADMIT-calibration 15%, '
                        'outcome-train 30%, deploy 30%. Fit/cal cells: <= 2,000 mapped cells per patient; outcome-'
                        'train and deploy patients typed on <= 3,000 random mapped cells per patient. Outcome-train '
                        'patients are typed under the FULL panel; deploy patients under every bank panel.',
        'D2_crossfit': 'every patient typed out-of-sample under its own full panel: 5 seeds (rng 3000+seed), '
                       'patients halved; each half typed by a classifier fit on 60% / ADMIT calibrated on 40% of '
                       'the other half; <= 3,000 cells per typed patient.',
        'X_cross_cohort': 'reference cohort A typed by D2; target cohort B (different patients) typed by the '
                          'fold-1 classifier / calibration of A: B features placed by vocabulary name and z-scored on '
                          'B\'s own cells (label-free), A-only markers = 0 (MI); RT = classifier retrained on the A '
                          'markers whose vocabulary name is in B; ADMIT calibrated on A calibration cells masked to '
                          'A cap B with A(P) evaluated on A cap B. B labels are never used for typing.',
        'panels': 'within-cohort panels from the fixed bank runs/cohorts/<cohort>/panel_bank.json (written '
                  'before any result): full, del_CD8, del_FOXP3, del_CD20, del_CD68 (where present) and '
                  'rand50_s0..s4',
    },
    'features': {
        'node_map_m_P': 'for deployment panel P, F_P = the M1 supported groups (AdmitM1.gmap of the labels, fit on '
                        'the calibration cells under P). Any node n (forced label, consensus node) maps to the first '
                        'ancestor-or-self of pi_P(n) in F_P; a node coarser than every F_P node on its path keeps its '
                        'own name as a separate "coarse" feature; unresolved and abstained cells are separate counts',
        'F1_forced': 'forced argmax proportions at full (17-node) resolution; outcome model trained on full-panel '
                     'forced proportions of the outcome-train patients; deployed on forced-MI(P) or forced-RT(P)',
        'F1p_forced_projected': 'control: forced argmax mapped by m_P (same feature space as F2); trained on '
                                'full-panel forced mapped by m_P; isolates coarsening from conformal sets',
        'F2_admit': 'ADMIT consensus proportions mapped by m_P (coarsest common supported level of the full '
                    'training panel and P); trained on full-panel ADMIT mapped by m_P',
        'F3_admit_unc': 'F2 + per-patient abstention rate, unresolved rate, mean |C|, and for every F_P feature the '
                        'ambiguous fraction (cells whose set contains a node mapping to f minus cells whose consensus '
                        'maps to f)',
        'oracle': 'author labels (node17), full resolution, panel-independent ceiling',
        'denominator': 'typed (mapped) cells of the patient; sqrt-transformed proportions, z-scored with outcome-'
                       'train mean/SD; constant features dropped',
    },
    'outcome_models': {
        'ridge_cox': 'lifelines CoxPHFitter(penalizer=0.1, l1_ratio=0) on the transformed features (primary)',
        'xgb_cox': 'XGBoost survival:cox, CUDA, max_depth 2, eta 0.05, 300 rounds, subsample 0.8, colsample 0.8, '
                   'min_child_weight 5, seed = split seed (secondary)',
        'binary': 'L2 logistic regression (sklearn, C = 1.0) and XGBoost binary:logistic with the same tree '
                  'settings; AUC replaces C',
        'no clinical covariates': 'cell features only, to isolate the effect of typing',
    },
    'E11_cross_panel_deployment': {
        'priority': 1,
        'sim_cohorts': {'primary': 'cords2024_nsclc_imc (OS)',
                        'secondary': ['danenberg2022_metabric_imc (OS)', 'fischer2023_breast_imc (OS)',
                                      'quiche_tnbc_mibi_spain (RFS)'],
                        'note': 'secondary cohorts run only if compute allows; any not run is reported as not run'},
        'real_pairs': {'kulasinghe2026_nsclc_pcf -> monkman2026_nsclc_pcf': 'OS, primary real pair (different '
                       'patients, panels differ by HLADR); reverse direction secondary',
                       'quiche_tnbc_mibi_spain -> quiche_tnbc_mibi_stanford': 'RFS (same panel, different site)',
                       'mcc_discovery_haist_codex -> mcc_validation_sizun_codex': 'NOT RUN: validation has 5 OS '
                       'events (< 10)'},
        'real_pair_design': 'A typed by D2 (all A patients out-of-sample), outcome model trained on all A patients; '
                            'B typed by X_cross_cohort; metric on all B patients. "Full-panel C" reference = 5-fold '
                            'CV C within A; oracle = author labels A -> author labels B',
        'metrics': 'Harrell C on deploy patients (risk = model linear predictor / margin), calibration slope '
                   '(Cox regression of deploy outcome on the linear predictor), dC = C(P) - C(full panel) for the '
                   'same feature set and model on the same deploy patients, and C(P) - C(F1, full panel); feature '
                   'shift = mean over features of |mean_P - mean_full| / SD_full on the same deploy patients',
        'uncertainty': 'mean and range over 5 seeds; paired patient bootstrap (B = 500, seed 0) CI for C(F2,P) - '
                       'C(F1,P) and C(F3,P) - C(F2,P)',
        'hypotheses': {'H11a': 'forced (F1) features lose more C under deletion than ADMIT (F2) features',
                       'H11b': 'ADMIT feature shift between full-panel and P typing is smaller than forced',
                       'H11c': 'F3 >= F2 (uncertainty features do not hurt and may help under panel change)'},
    },
    'E13_harmonized_pooling': {
        'priority': 2,
        'pools': {'NSCLC_OS': ['cords2024_nsclc_imc', 'kulasinghe2026_nsclc_pcf', 'monkman2026_nsclc_pcf'],
                  'TNBC_OS': ['fischer2023_breast_imc (Triple negative only, mol_signature)', 'meyer2025_tnbc_imc'],
                  'TNBC_recurrence': ['quiche_tnbc_mibi_spain (RFS)', 'quiche_tnbc_mibi_stanford (RFS)',
                                      'meyer2025_tnbc_imc (DFS)']},
        'reference': {'NSCLC': 'cords2024_nsclc_imc', 'TNBC': 'quiche_tnbc_mibi_spain'},
        'typing': 'reference cohort by D2 (seed 0); other cohorts by X_cross_cohort from the reference (seed 0)',
        'common_resolution': 'F_common = pi_{P_cap}(reference labels) with P_cap = intersection over pool cohorts '
                             'of (reference panel cap cohort panel); every cohort\'s ADMIT consensus mapped to '
                             'F_common (first ancestor-or-self in F_common)',
        'versions': {'admit': 'ADMIT consensus at F_common (MI scorer primary, RT secondary)',
                     'forced': 'reference-classifier argmax at full resolution (MI primary, RT secondary)',
                     'author_naive': 'each cohort\'s own author node17 labels matched by name'},
        'cell_types': 'cyto_T, treg, helper_T, pooled_T, B, myeloid_mac (round-1 definitions); for ADMIT a type is '
                      'analysed only if its node is in F_common (pooled_T always)',
        'model': 'per pool x endpoint x type x version: Cox stratified by cohort, proportion z-scored over the '
                 'pooled patients (primary; secondary: within-cohort z), unadjusted; per-cohort Cox for Cochran Q '
                 'and I^2; inverse-variance pooled log HR; precision = 1/SE^2 of the stratified estimate',
        'endpoint_rule': 'cohorts enter a pool only for their own endpoint; >= 10 events per cohort',
    },
    'E12_proportion_bounds': {
        'priority': 3,
        'source': 'D1 typing of the E11 sim cohorts (deploy + outcome-train patients), every bank panel, MI and RT',
        'bounds': 'type t (round-1 definitions on node17): lower L = fraction of cells whose set is non-empty and '
                  'contained in subtree(t); upper U = fraction whose set intersects subtree(t) or ancestors(t), '
                  'plus abstained cells; truth = author-label proportion over the same typed cells',
        'metrics': 'patient-level coverage P(L <= truth <= U), mean width (percentage points)',
        'sensitivity': 'Cox (round-1 adjusted specification) of the primary endpoint on L, on U, on the ADMIT '
                       'point estimate and on truth (all typed patients): is the HR direction and p < 0.05 status '
                       'the same across L and U ("stable")?',
    },
    'E10_within_cohort_prediction': {
        'priority': 4,
        'cohorts': ['cords2024_nsclc_imc OS', 'danenberg2022_metabric_imc OS', 'fischer2023_breast_imc OS',
                    'quiche_tnbc_mibi_spain RFS', 'neotrip_tnbc_imc pCR (AUC)', 'mcc_discovery_haist_codex ICI '
                    'response (AUC)'],
        'design': 'D2 typing (full panel); outcome models by patient-level 5-fold CV repeated over the 5 typing '
                  'seeds; features F1, F2, F3, oracle; metric C (AUC for binary) on held-out folds',
        'contrast': 'F3 - F2 (paired, per seed; mean and range); expectation stated in advance: little gain at full '
                    'panel',
    },
    'E14_uncertainty_as_QC': {
        'priority': 5,
        'cohorts': ['cords2024_nsclc_imc (samples = cores)', 'schurch CRC CODEX (samples = File Name / spot)'],
        'design': 'classifier fit 25% / ADMIT calibration 15% / test 60% of patients (full panel, 3 seeds); 10% of '
                  'test samples fail: one lineage marker (uniform over the panel\'s R(c) markers) has its raw '
                  'intensity x 0.2 in every cell of the sample (primary) or Gaussian noise of 1 SD added in z '
                  'space (secondary)',
        'scores': {'admit_primary': 'non-singleton rate (fraction of cells whose set is empty or has > 1 node)',
                   'admit_secondary': ['mean |C|', 'abstention rate', 'unresolved rate'],
                   'forced_primary': 'mean (1 - max probability)',
                   'forced_secondary': ['mean entropy']},
        'metric': 'AUROC (failed vs intact test samples), per seed; mean and range; also per failed marker',
    },
    'multiplicity': 'all round-2 analyses are exploratory estimation; CIs and ranges, no claims from p-values',
    'negative_results': 'reported as found, in the same tables',
    'compute': 'one GPU workstation (XGBoost)',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--amend', default=None)
    a = ap.parse_args()
    h = hashlib.sha256(json.dumps(PLAN, sort_keys=True).encode()).hexdigest()[:16]
    path = OUT / 'round2' / 'analysis_plan_round2.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    amend = []
    if path.exists():
        old = json.loads(path.read_text())
        if old['plan_hash'] == h:
            print('unchanged', h)
            return
        if not a.amend:
            sys.exit('plan exists with a different hash; use --amend')
        amend = old.get('amendments', []) + [{'at': datetime.now().isoformat(timespec='seconds'), 'reason': a.amend,
                                              'previous_hash': old['plan_hash']}]
    obj = {'written_at': datetime.now().isoformat(timespec='seconds'), 'plan_hash': h, 'amendments': amend,
           'plan': PLAN}
    txt = json.dumps(obj, indent=1)
    path.write_text(txt)
    (OUT / 'analysis_plan_round2.json').write_text(txt)
    print('wrote', path, h)


if __name__ == '__main__':
    main()
