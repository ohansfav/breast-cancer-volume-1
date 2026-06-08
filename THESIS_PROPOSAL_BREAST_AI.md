# Thesis Proposal: Risk-Calibrated Breast Cancer Screening from FNA Morphology

## Working Title
Risk-Calibrated Machine Learning for Breast Cancer Triage: A Clinically Tunable Baseline Using Wisconsin FNA Morphology Features

## Problem Statement
Early breast cancer triage demands high sensitivity while controlling false positives that overload diagnostic pathways. Many ML papers optimize global accuracy, but clinical settings require threshold-aware decision policies and calibrated risk estimates. This thesis studies whether a transparent baseline model, combined with calibration and threshold optimization, can produce clinically useful screening behavior.

## Core Research Questions
1. How does threshold tuning change recall-precision tradeoffs for malignant case detection?
2. Can a simple interpretable model remain competitive when evaluation is aligned to clinical objectives?
3. Does probability calibration improve decision reliability for referral triage?
4. Which morphology features provide the strongest malignant-vs-benign separation in this dataset?

## Hypotheses
1. Lower thresholds will increase malignant recall substantially with acceptable precision loss for screening contexts.
2. A calibrated logistic baseline can provide stable, reproducible performance suitable for low-resource clinics.
3. Risk calibration diagnostics (Brier score and reliability curves) will reveal decision confidence quality beyond accuracy.

## Methodology
1. Dataset: scikit-learn Wisconsin Breast Cancer (FNA morphology).
2. Preprocessing: stratified train/test split and standard scaling.
3. Baseline model: logistic regression with probability outputs.
4. Evaluation: confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, calibration curve.
5. Threshold study: evaluate multiple decision thresholds and identify policy-specific operating points.
6. Statistical interpretation: compute effect-size ranking using Cohen's d for top differentiating features.

## Experimental Plan
1. Run baseline with fixed seed for reproducibility.
2. Sweep thresholds from 0.1 to 0.9 and record metrics.
3. Compare at least two policies:
   - High-recall screening policy
   - Balanced triage policy
4. Optional extension: compare logistic regression against random forest and gradient boosting under same split protocol.

## Thesis-Ready Deliverables
1. `thesis_outputs/feature_summary.csv`
2. `thesis_outputs/effect_sizes.csv`
3. `thesis_outputs/evaluation_metrics.json`
4. Interactive interface screenshots and threshold policy analysis
5. Error analysis section using confusion matrix and misclassification discussion

## Publication/Defense Angle
This thesis emphasizes translational value: interpretable ML, calibrated risk, and threshold policies tied to clinical workflow constraints. The same framework can be extended to larger institutional datasets and external validation in future work.

## Limitations and Ethics
1. Dataset size is modest and from a single benchmark source.
2. Results are not a substitute for medical diagnosis.
3. External validation is required before deployment.
4. Model bias and subgroup fairness should be evaluated on richer demographic datasets.

## Next Extensions
1. Add uncertainty bands and abstention policy for ambiguous predictions.
2. Introduce temporal follow-up and longitudinal risk updates.
3. Validate on external breast pathology or imaging-linked cohorts.
