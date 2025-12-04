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
**TEST**: ACC=0.5163, F1=0.6720, Precision=0.5138, Recall=0.9709, AUC=0.4943, LogLoss=0.7663

### Calibrated Model Performance

**TRAIN**: ACC=0.0000, F1=0.0000, Precision=0.0000, Recall=0.0000, AUC=0.0000, LogLoss=0.0000
**VAL**: ACC=0.0000, F1=0.0000, Precision=0.0000, Recall=0.0000, AUC=0.0000, LogLoss=0.0000
**TEST**: ACC=0.5163, F1=0.6772, Precision=0.5135, Recall=0.9942, AUC=0.5665, LogLoss=0.6790

### Class Balance

- Train positive rate (bearish): 0.0000
- Val positive rate (bearish): 0.0000
- Test positive rate (bearish): 0.5104

## Forward-Return Diagnostics

### Horizon 4 bars (60 minutes)

- n_samples: 337
- mean_fwd_return: 0.003802
- std_fwd_return: 0.065324
- **corr_prob_fwd**: 0.1511 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5697
- Returns by probability bucket:
  - bucket_0: 0.000945
  - bucket_1: 0.075016

### Horizon 8 bars (120 minutes)

- n_samples: 337
- mean_fwd_return: 0.007718
- std_fwd_return: 0.092446
- **corr_prob_fwd**: 0.1570 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5668
- Returns by probability bucket:
  - bucket_0: 0.005033
  - bucket_1: 0.074637

### Horizon 12 bars (180 minutes)

- n_samples: 337
- mean_fwd_return: 0.011575
- std_fwd_return: 0.114319
- **corr_prob_fwd**: 0.1692 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5430
- Returns by probability bucket:
  - bucket_0: 0.009115
  - bucket_1: 0.072901

## Signal Statistics

- Bullish signals (prob < 0.4): 0.1186
- Neutral signals (0.4 ≤ prob ≤ 0.6): 0.5217
- Bearish signals (prob > 0.6): 0.0257
- Mean calibrated probability: 0.5104
- Std calibrated probability: 0.0823

## Top 15 Feature Importances

