# Regime Ensemble Training Report

**Symbol:** ETHUSDT
**Model:** stacker_lgbm_calibrated (LightGBM Meta-Learner)
**Generated:** 2025-11-11T23:14:48.109069
**Report Version:** 1.0

## Ensemble Configuration

- **Meta-Learner:** LightGBM with Probability Calibration
- **Number of Base Models:** 3
- **Base Models:** extratrees, lightgbm, random_forest

## Ensemble Performance Metrics

### Overall Performance

| Metric | Value |
|--------|-------|
| Accuracy | 0.9981 |
| Precision (Weighted) | 0.9981 |
| Recall (Weighted) | 0.9981 |
| F1-Score (Weighted) | 0.9981 |
| Precision (Macro) | 0.9989 |
| Recall (Macro) | 0.9975 |
| F1-Score (Macro) | 0.9982 |
| Prediction Confidence | 0.9966 ± 0.0312 |
| Calibration Method | isotonic |

### Advanced Classification Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Macro F1 (Equal Weight) | 0.9982 | Treats each regime equally |
| Balanced Accuracy | 0.9975 | Average recall per class |
| Cohen's Kappa | 0.9976 | Almost perfect agreement |
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
| Log Loss | 0.0055 | Logarithmic loss |
| Brier Score | 0.0031 | Probability calibration |

### Temporal Regime Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Detection Delay (Mean) | 0.0039 | Average time to detect regime change |
| Detection Delay (Std) | 0.0625 | Std deviation of detection delay |
| True Regime Persistence | 4.1255 | Average duration of true regimes |
| Predicted Regime Persistence | 4.0934 | Average duration of predicted regimes |
| Persistence Ratio | 0.9922 | Ratio of predicted to true persistence |
| Transition Accuracy | 0.9962 | Accuracy of regime transition prediction |

### Segmentation Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Adjusted Rand Index | 0.9952 | Segmentation quality measure |
| Boundary Precision | 0.9883 | Precision of boundary detection |
| Boundary Recall | 0.9961 | Recall of boundary detection |
| Boundary F1-Score | 0.9922 | F1-score of boundary detection |

### Change-Point Detection Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Change-Point Precision | 0.9922 | Precision of change-point detection |
| Change-Point Recall | 1.0000 | Recall of change-point detection |
| Change-Point F1-Score | 0.9961 | F1-score of change-point detection |
| True Change Points | 254 | Number of true change points |
| Predicted Change Points | 256 | Number of predicted change points |
| Detected Change Points | 254 | Number of correctly detected change points |

### Sequence Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Hamming Loss | 0.0019 | Fraction of misclassified time points |

### Per-Regime Performance

| Regime | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| 0 | 1.0000 | 0.9852 | 0.9925 | 135.0 |
| 1 | 1.0000 | 1.0000 | 1.0000 | 92.0 |
| 2 | 0.9936 | 1.0000 | 0.9968 | 312.0 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 52.0 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 267.0 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 194.0 |

## Temporal Analysis

- **Transition Entropy:** 0.0000
- **Average Regime Duration:** 0.00 periods
- **Number of Transitions:** 0
