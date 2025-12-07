# ML Mean-Reversion (v2) Summary for ETHUSDT (15m)

**Model Type**: XGBoost Classifier with Isotonic Calibration
**Target**: Directional (0=up, 1=down)
**Version**: v2 with relaxed thresholds, enhanced features, and proper calibration

## Teacher (OU/Hurst GMM) - IMPROVED

- Components: 3
- Mean-reversion cluster: 1
- Cluster counts: {0: 18400, 1: 9230, 2: 5183, -1: 2228}
- Thresholds (RELAXED for 15m):
  - Hurst: 0.5
  - Half-life: 12.0 bars
  - ADF p-value: 0.15
  - Variance ratio: 1.2
- **Teacher positive rate: 0.0939** (IMPROVED from ~0.0)

## Student (XGB Classifier) - RAW vs CALIBRATED

### Raw Model Performance

**TRAIN**: ACC=0.0000, F1=0.0000, Precision=0.0000, Recall=0.0000, AUC=0.0000, LogLoss=0.0000
**VAL**: ACC=0.5005, F1=0.5686, Precision=0.5756, Recall=0.5618, AUC=0.5063, LogLoss=0.7051
**TEST**: ACC=0.5764, F1=0.6300, Precision=0.5760, Recall=0.6953, AUC=0.5915, LogLoss=0.6816

### Calibrated Model Performance

**TRAIN**: ACC=0.0000, F1=0.0000, Precision=0.0000, Recall=0.0000, AUC=0.0000, LogLoss=0.0000
**VAL**: ACC=0.5027, F1=0.5635, Precision=0.5802, Recall=0.5478, AUC=0.5159, LogLoss=0.6885
**TEST**: ACC=0.5828, F1=0.6289, Precision=0.5839, Recall=0.6815, AUC=0.6030, LogLoss=0.6765

### Class Balance

- Train positive rate (bearish): 0.0000
- Val positive rate (bearish): 0.5861
- Test positive rate (bearish): 0.5188

## Forward-Return Diagnostics

### Horizon 4 bars (60 minutes)

- n_samples: 7619
- mean_fwd_return: -0.000574
- std_fwd_return: 0.013172
- **corr_prob_fwd**: -0.0843 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5327
- Returns by probability bucket:
  - bucket_0: 0.000694
  - bucket_1: -0.001192
  - bucket_2: -0.002181

### Horizon 8 bars (120 minutes)

- n_samples: 7619
- mean_fwd_return: -0.000988
- std_fwd_return: 0.019402
- **corr_prob_fwd**: -0.1024 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5482
- Returns by probability bucket:
  - bucket_0: 0.001404
  - bucket_1: -0.002162
  - bucket_2: -0.003997

### Horizon 12 bars (180 minutes)

- n_samples: 7619
- mean_fwd_return: -0.001189
- std_fwd_return: 0.024106
- **corr_prob_fwd**: -0.1000 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5527
- Returns by probability bucket:
  - bucket_0: 0.001715
  - bucket_1: -0.002594
  - bucket_2: -0.004904

## Signal Statistics

- Bullish signals (prob < 0.4): 0.0000
- Neutral signals (0.4 ≤ prob ≤ 0.6): 0.7852
- Bearish signals (prob > 0.6): 0.1169
- Mean calibrated probability: 0.5371
- Std calibrated probability: 0.0695

## Top 15 Feature Importances

