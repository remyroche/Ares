# ML Mean-Reversion (v2) Summary for ETHUSDT (15m)

**Model Type**: XGBoost Classifier with Isotonic Calibration
**Target**: Directional (0=up, 1=down)
**Version**: v2 with relaxed thresholds, enhanced features, and proper calibration

## Teacher (OU/Hurst GMM) - IMPROVED

- Components: 3
- Mean-reversion cluster: 2
- Cluster counts: {0: 1453, 2: 527, -1: 287, 1: 173}
- Thresholds (RELAXED for 15m):
  - Hurst: 0.5
  - Half-life: 12.0 bars
  - ADF p-value: 0.15
  - Variance ratio: 1.2
- **Teacher positive rate: 0.0418** (IMPROVED from ~0.0)

## Student (XGB Classifier) - RAW vs CALIBRATED

### Raw Model Performance

**TRAIN**: ACC=0.0000, F1=0.0000, Precision=0.0000, Recall=0.0000, AUC=0.0000, LogLoss=0.0000
**VAL**: ACC=0.0000, F1=0.0000, Precision=0.0000, Recall=0.0000, AUC=0.0000, LogLoss=0.0000
**TEST**: ACC=0.5272, F1=0.6904, Precision=0.5272, Recall=1.0000, AUC=0.3520, LogLoss=0.7681

### Calibrated Model Performance

**TRAIN**: ACC=0.0000, F1=0.0000, Precision=0.0000, Recall=0.0000, AUC=0.0000, LogLoss=0.0000
**VAL**: ACC=0.0000, F1=0.0000, Precision=0.0000, Recall=0.0000, AUC=0.0000, LogLoss=0.0000
**TEST**: ACC=0.5387, F1=0.6945, Precision=0.5335, Recall=0.9946, AUC=0.5332, LogLoss=0.6803

### Class Balance

- Train positive rate (bearish): 0.0000
- Val positive rate (bearish): 0.0000
- Test positive rate (bearish): 0.5272

## Forward-Return Diagnostics

### Horizon 4 bars (60 minutes)

- n_samples: 349
- mean_fwd_return: 0.003409
- std_fwd_return: 0.064273
- **corr_prob_fwd**: 0.1300 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5072
- Returns by probability bucket:
  - bucket_0: 0.000649
  - bucket_1: 0.069438

### Horizon 8 bars (120 minutes)

- n_samples: 349
- mean_fwd_return: 0.007021
- std_fwd_return: 0.090946
- **corr_prob_fwd**: 0.0793 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5415
- Returns by probability bucket:
  - bucket_0: 0.004478
  - bucket_1: 0.067862

### Horizon 12 bars (180 minutes)

- n_samples: 349
- mean_fwd_return: 0.010613
- std_fwd_return: 0.112461
- **corr_prob_fwd**: 0.0665 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5358
- Returns by probability bucket:
  - bucket_0: 0.008276
  - bucket_1: 0.066537

## Signal Statistics

- Bullish signals (prob < 0.4): 0.0115
- Neutral signals (0.4 ≤ prob ≤ 0.6): 0.6279
- Bearish signals (prob > 0.6): 0.0267
- Mean calibrated probability: 0.5272
- Std calibrated probability: 0.0714

## Top 15 Feature Importances

