# ML Mean-Reversion (v2) Summary for ETHUSDT (15m)

**Model Type**: XGBoost Classifier with Isotonic Calibration
**Target**: Directional (0=up, 1=down)
**Version**: v2 with relaxed thresholds, enhanced features, and proper calibration

## Teacher (OU/Hurst GMM) - IMPROVED

- Components: 3
- Mean-reversion cluster: 0
- Cluster counts: {1: 2991, 0: 975, 2: 397, -1: 309}
- Thresholds (RELAXED for 15m):
  - Hurst: 0.5
  - Half-life: 12.0 bars
  - ADF p-value: 0.15
  - Variance ratio: 1.2
- **Teacher positive rate: 0.0574** (IMPROVED from ~0.0)

## Student (XGB Classifier) - RAW vs CALIBRATED

### Raw Model Performance

**TRAIN**: ACC=0.0000, F1=0.0000, Precision=0.0000, Recall=0.0000, AUC=0.0000, LogLoss=0.0000
**VAL**: ACC=0.5926, F1=0.7442, Precision=0.5926, Recall=1.0000, AUC=0.3521, LogLoss=0.8037
**TEST**: ACC=0.5341, F1=0.6948, Precision=0.5323, Recall=1.0000, AUC=0.4588, LogLoss=0.7102

### Calibrated Model Performance

**TRAIN**: ACC=0.0000, F1=0.0000, Precision=0.0000, Recall=0.0000, AUC=0.0000, LogLoss=0.0000
**VAL**: ACC=0.5926, F1=0.7442, Precision=0.5926, Recall=1.0000, AUC=0.5417, LogLoss=0.6643
**TEST**: ACC=0.5530, F1=0.7035, Precision=0.5426, Recall=1.0000, AUC=0.5741, LogLoss=0.6475

### Class Balance

- Train positive rate (bearish): 0.0000
- Val positive rate (bearish): 0.5926
- Test positive rate (bearish): 0.5303

## Forward-Return Diagnostics

### Horizon 4 bars (60 minutes)

- n_samples: 408
- mean_fwd_return: 0.004912
- std_fwd_return: 0.092547
- **corr_prob_fwd**: -0.0293 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5417
- Returns by probability bucket:
  - bucket_0: 0.005822
  - bucket_1: -0.010332

### Horizon 8 bars (120 minutes)

- n_samples: 408
- mean_fwd_return: 0.008180
- std_fwd_return: 0.131899
- **corr_prob_fwd**: -0.0614 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5515
- Returns by probability bucket:
  - bucket_0: 0.011089
  - bucket_1: -0.040511

### Horizon 12 bars (180 minutes)

- n_samples: 408
- mean_fwd_return: 0.007870
- std_fwd_return: 0.136375
- **corr_prob_fwd**: -0.0581 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5588
- Returns by probability bucket:
  - bucket_0: 0.010579
  - bucket_1: -0.037467

## Signal Statistics

- Bullish signals (prob < 0.4): 0.0092
- Neutral signals (0.4 ≤ prob ≤ 0.6): 0.5795
- Bearish signals (prob > 0.6): 0.0352
- Mean calibrated probability: 0.5490
- Std calibrated probability: 0.1053

## Top 15 Feature Importances

