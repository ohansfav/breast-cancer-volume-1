# Breast Cancer Wisconsin Diagnostic Screening

An academic-grade, reproducible medical AI portfolio project for exploratory clinical data analysis and baseline classification on the authentic Breast Cancer Wisconsin dataset from scikit-learn.

## Repository Layout

```text
breast cancer/
├── medical_ai_loader.py
└── README.md
```

## Project Overview

This repository implements a compact but production-oriented diagnostic screening workflow for breast cancer risk stratification. The pipeline loads the Wisconsin Breast Cancer dataset directly from scikit-learn, performs structured data inspection with Pandas and NumPy, standardizes the clinical feature space, trains a baseline classifier, and reports screening-oriented evaluation metrics in the terminal.

The design goal is not novelty for its own sake. It is technical clarity: a clean, auditable implementation that demonstrates the core competencies expected in Digital Health and Oncology AI projects, including dataset governance, feature interpretation, reproducible preprocessing, and model evaluation.

## Clinical Context

The dataset contains measurements extracted from digitized fine needle aspirate (FNA) images of breast masses. These features encode morphology that is clinically meaningful:

- `mean radius`: a proxy for lesion size; larger masses may warrant greater suspicion.
- `mean texture`: reflects intensity variation and tissue heterogeneity.
- `mean perimeter`: approximates boundary extent and contributes to shape characterization.
- `mean area`: captures the two-dimensional footprint of the lesion.
- `mean smoothness`: describes local boundary regularity; irregular edges can indicate abnormal growth patterns.
- `mean compactness`, `mean concavity`, and `mean concave points`: summarize contour distortion and architectural irregularity, both of which are important visual cues in malignancy assessment.

In a screening setting, the primary clinical objective is not merely classification accuracy. False negatives are especially consequential because they correspond to malignant cases incorrectly labeled as benign. For that reason, the pipeline reports precision, recall, F1-score, and a confusion matrix rather than relying on accuracy alone.

## Installation Guide

1. Create and activate a Python environment.
1. Install the required scientific Python stack.

```bash
pip install numpy pandas scikit-learn
```

1. Run the analysis script.

```bash
python medical_ai_loader.py
```

The terminal output will include dataset loading status, feature interpretation notes, train/test split confirmation, a clinical confusion matrix, and the final classification metrics.

## Mathematical and Statistical Logic Used

This project follows a standard diagnostic machine learning workflow:

1. **Dataset ingestion**: the Breast Cancer Wisconsin data is loaded from `sklearn.datasets.load_breast_cancer`, ensuring that the source is authentic and versioned with scikit-learn.
2. **Exploratory statistics**: Pandas descriptive statistics are used to summarize central tendency, dispersion, and missingness across features.
3. **Stratified split**: the dataset is partitioned into training and testing subsets with preserved class proportions, which is essential when evaluating medical classifiers under class imbalance.
4. **Feature scaling**: `StandardScaler` transforms each feature to zero mean and unit variance. This normalizes the geometry of the feature space and stabilizes the optimizer for linear models.
5. **Baseline classifier**: a classical machine learning model is trained on the standardized features. The implementation is intentionally interpretable and suitable for a first-pass screening benchmark.
6. **Evaluation metrics**:
   - **Precision**: $\frac{TP}{TP + FP}$, indicating how many predicted malignant cases are truly malignant.
   - **Recall**: $\frac{TP}{TP + FN}$, indicating how many malignant cases are successfully detected.
   - **F1-score**: $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$, balancing false positives and false negatives.
   - **Confusion matrix**: provides the full error structure for clinical interpretation.

This statistical framing mirrors a real diagnostic workflow: measure the lesion, normalize the measurements, train a classifier, and inspect error modes rather than treating the problem as a generic benchmark.

## Future Roadmap

This repository is designed as Volume 1 in a multi-volume oncology AI portfolio.

- **Volume 2: Lung Cancer Screening**
  - CNN-based radiology feature learning
  - CT slice classification and uncertainty estimation
- **Volume 3: Melanoma Detection**
  - Vision Transformer models for dermoscopy analysis
  - Explainability overlays for lesion localization
- **Volume 4: Multi-Modal Oncology AI**
  - Fusion of imaging, pathology, and structured clinical variables
  - Risk calibration and longitudinal modeling
- **Volume 5: Medical API Deployment**
  - FastAPI-based inference services
  - Secure model serving with audit logging and version control

## Why This Project Matters

A strong oncology AI portfolio should demonstrate more than model training. It should show disciplined handling of clinical data, a clear understanding of evaluation tradeoffs, and code that can be read by researchers, clinicians, and engineering reviewers alike. This repository is intentionally concise, but it establishes that standard from the first volume onward.
