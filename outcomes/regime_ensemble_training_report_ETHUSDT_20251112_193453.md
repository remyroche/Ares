# Regime Ensemble Training Report

**Symbol:** ETHUSDT
**Model:** stacker_lgbm_calibrated (LightGBM Meta-Learner)
**Generated:** 2025-11-12T19:34:53.378929
**Report Version:** 1.0

## Ensemble Configuration

- **Meta-Learner:** LightGBM with Probability Calibration
- **Number of Base Models:** 3
- **Base Models:** extratrees, lightgbm, random_forest

## Ensemble Performance Metrics

### Overall Performance

| Metric | Value |
|--------|-------|
| Accuracy | 0.9990 |
| Precision (Weighted) | 0.9991 |
| Recall (Weighted) | 0.9990 |
| F1-Score (Weighted) | 0.9990 |
| Precision (Macro) | 0.9995 |
| Recall (Macro) | 0.9988 |
| F1-Score (Macro) | 0.9991 |
| Prediction Confidence | 0.9972 ± 0.0263 |
| Calibration Method | isotonic |

### Advanced Classification Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Macro F1 (Equal Weight) | 0.9991 | Treats each regime equally |
| Balanced Accuracy | 0.9988 | Average recall per class |
| Cohen's Kappa | 0.9988 | Almost perfect agreement |
| ROC-AUC (class_0) | 1.0000 | One-vs-rest ROC-AUC |
| ROC-AUC (class_1) | 1.0000 | One-vs-rest ROC-AUC |
| ROC-AUC (class_2) | 1.0000 | One-vs-rest ROC-AUC |
| ROC-AUC (class_3) | 1.0000 | One-vs-rest ROC-AUC |
| ROC-AUC (class_4) | 1.0000 | One-vs-rest ROC-AUC |
| ROC-AUC (class_5) | 1.0000 | One-vs-rest ROC-AUC |
| PR-AUC (class_0) | 1.0000 | One-vs-rest PR-AUC |
| PR-AUC (class_1) | 1.0000 | One-vs-rest PR-AUC |
| PR-AUC (class_2) | 1.0000 | One-vs-rest PR-AUC |
| PR-AUC (class_3) | 1.0000 | One-vs-rest PR-AUC |
| PR-AUC (class_4) | 1.0000 | One-vs-rest PR-AUC |
| PR-AUC (class_5) | 1.0000 | One-vs-rest PR-AUC |
| Log Loss | 0.0048 | Logarithmic loss |
| Brier Score | 0.0025 | Probability calibration |

### Temporal Regime Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Detection Delay (Mean) | 0.0039 | Average time to detect regime change |
| Detection Delay (Std) | 0.0625 | Std deviation of detection delay |
| True Regime Persistence | 4.1255 | Average duration of true regimes |
| Predicted Regime Persistence | 4.1255 | Average duration of predicted regimes |
| Persistence Ratio | 1.0000 | Ratio of predicted to true persistence |
| Transition Accuracy | 0.9981 | Accuracy of regime transition prediction |

### Segmentation Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Adjusted Rand Index | 0.9976 | Segmentation quality measure |
| Boundary Precision | 0.9961 | Precision of boundary detection |
| Boundary Recall | 0.9961 | Recall of boundary detection |
| Boundary F1-Score | 0.9961 | F1-score of boundary detection |

### Change-Point Detection Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Change-Point Precision | 1.0000 | Precision of change-point detection |
| Change-Point Recall | 1.0000 | Recall of change-point detection |
| Change-Point F1-Score | 1.0000 | F1-score of change-point detection |
| True Change Points | 254 | Number of true change points |
| Predicted Change Points | 254 | Number of predicted change points |
| Detected Change Points | 254 | Number of correctly detected change points |

### Sequence Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Hamming Loss | 0.0010 | Fraction of misclassified time points |

### Per-Regime Performance

| Regime | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 0 | 1.0000 | 0.9926 | 0.9963 | 135.0 |
| 1 | 1.0000 | 1.0000 | 1.0000 | 92.0 |
| 2 | 0.9968 | 1.0000 | 0.9984 | 312.0 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 52.0 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 267.0 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 194.0 |

## Temporal Analysis

- **Transition Entropy:** 0.0000
- **Average Regime Duration:** 0.00 periods
- **Number of Transitions:** 0
