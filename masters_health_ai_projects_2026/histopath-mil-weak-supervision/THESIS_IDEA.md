# Thesis Concept: histopath-mil-weak-supervision

## Working title
Weakly supervised multiple-instance learning for histopath triage in low-label oncology settings

## Research question
Can we improve clinical utility and trustworthiness for Breast/General Oncology tasks using a compact, reproducible pipeline that prioritizes calibration, sensitivity, and interpretability?

## Hypothesis
A carefully engineered lightweight model, with clinically motivated features and threshold tuning, can achieve robust performance and deployment readiness without requiring massive datasets.

## Dataset plan
- Phase 1: synthetic data for method development and stress testing.
- Phase 2: public datasets (e.g., TCGA, UCI, Kaggle oncology subsets, or institution-approved de-identified data).
- Phase 3: external validation and subgroup fairness checks.

## Method outline
1. Data preprocessing and leakage-safe split strategy.
2. Baseline model + one advanced variant.
3. Probability calibration and decision-threshold optimization.
4. Explainability (feature importance or SHAP-like analysis).
5. Sensitivity analysis and failure case audit.

## Evaluation
- Primary: Bag-level AUC and sensitivity at 95% specificity
- Secondary: subgroup stability, calibration across risk deciles.
- Practical: decision-curve style utility analysis.

## Expected contribution
- Reproducible implementation with transparent assumptions.
- Clinically interpretable performance reporting.
- A thesis-to-paper pathway with feasible compute requirements.

## Publishable structure
- Introduction: clinical pain point + AI gap.
- Methods: reproducible pipeline and governance checks.
- Results: quantitative metrics + subgroup and calibration.
- Discussion: deployment limitations and ethical implications.
