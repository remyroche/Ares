# ML Mean-Reversion (v2) Summary for ETHUSDT (15m)

**Model Type**: XGBoost Classifier with Isotonic Calibration
**Target**: Directional (0=up, 1=down)
**Version**: v2 with relaxed thresholds, enhanced features, and proper calibration

## Teacher (OU/Hurst GMM) - IMPROVED

- Components: 3
- Mean-reversion cluster: 2
- Cluster counts: {1: 1256, 2: 699, -1: 287, 0: 101}
- Thresholds (RELAXED for 15m):
  - Hurst: 0.5
  - Half-life: 12.0 bars
  - ADF p-value: 0.15
  - Variance ratio: 1.2
- **Teacher positive rate: 0.0435** (IMPROVED from ~0.0)

## Student (XGB Classifier) - RAW vs CALIBRATED

### Raw Model Performance

**TRAIN**: ACC=0.0000, F1=0.0000, Precision=0.0000, Recall=0.0000, AUC=0.0000, LogLoss=0.0000
**VAL**: ACC=0.0000, F1=0.0000, Precision=0.0000, Recall=0.0000, AUC=0.0000, LogLoss=0.0000
**TEST**: ACC=0.4985, F1=0.5986, Precision=0.5060, Recall=0.7326, AUC=0.4237, LogLoss=0.7661

### Calibrated Model Performance

**TRAIN**: ACC=0.0000, F1=0.0000, Precision=0.0000, Recall=0.0000, AUC=0.0000, LogLoss=0.0000
**VAL**: ACC=0.0000, F1=0.0000, Precision=0.0000, Recall=0.0000, AUC=0.0000, LogLoss=0.0000
**TEST**: ACC=0.5312, F1=0.6815, Precision=0.5216, Recall=0.9826, AUC=0.5223, LogLoss=0.6843

### Class Balance

- Train positive rate (bearish): 0.0000
- Val positive rate (bearish): 0.0000
- Test positive rate (bearish): 0.5104

## Forward-Return Diagnostics

### Horizon 4 bars (60 minutes)

- n_samples: 337
- mean_fwd_return: 0.003802
- std_fwd_return: 0.065324
- **corr_prob_fwd**: -0.0161 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5312
- Returns by probability bucket:
  - bucket_0: 0.003802

### Horizon 8 bars (120 minutes)

- n_samples: 337
- mean_fwd_return: 0.007718
- std_fwd_return: 0.092446
- **corr_prob_fwd**: 0.0610 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5341
- Returns by probability bucket:
  - bucket_0: 0.007718

### Horizon 12 bars (180 minutes)

- n_samples: 337
- mean_fwd_return: 0.011575
- std_fwd_return: 0.114319
- **corr_prob_fwd**: 0.0504 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5282
- Returns by probability bucket:
  - bucket_0: 0.011575

## Signal Statistics

- Bullish signals (prob < 0.4): 0.0255
- Neutral signals (0.4 ≤ prob ≤ 0.6): 0.6365
- Bearish signals (prob > 0.6): 0.0000
- Mean calibrated probability: 0.5104
- Std calibrated probability: 0.0606

## Top 15 Feature Importances

