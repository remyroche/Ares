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
**VAL**: ACC=0.4344, F1=0.1154, Precision=0.9000, Recall=0.0616, AUC=0.4205, LogLoss=0.7880
**TEST**: ACC=0.4226, F1=0.2633, Precision=0.3889, Recall=0.1991, AUC=0.3862, LogLoss=0.7552

### Calibrated Model Performance

**TRAIN**: ACC=0.0000, F1=0.0000, Precision=0.0000, Recall=0.0000, AUC=0.0000, LogLoss=0.0000
**VAL**: ACC=0.6066, F1=0.7405, Precision=0.6116, Recall=0.9384, AUC=0.5253, LogLoss=0.6753
**TEST**: ACC=0.5184, F1=0.6828, Precision=0.5184, Recall=1.0000, AUC=0.5000, LogLoss=0.6962

### Class Balance

- Train positive rate (bearish): 0.7551
- Val positive rate (bearish): 0.5984
- Test positive rate (bearish): 0.5184

## Forward-Return Diagnostics

### Horizon 4 bars (60 minutes)

- n_samples: 682
- mean_fwd_return: 0.004108
- std_fwd_return: 0.074292
- **corr_prob_fwd**: 0.0234 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5235
- Returns by probability bucket:
  - bucket_0: 0.004108

### Horizon 8 bars (120 minutes)

- n_samples: 682
- mean_fwd_return: 0.007084
- std_fwd_return: 0.105959
- **corr_prob_fwd**: 0.0258 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5513
- Returns by probability bucket:
  - bucket_0: 0.007084

### Horizon 12 bars (180 minutes)

- n_samples: 682
- mean_fwd_return: 0.010171
- std_fwd_return: 0.131227
- **corr_prob_fwd**: 0.0229 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5528
- Returns by probability bucket:
  - bucket_0: 0.010171

## Signal Statistics

- Bullish signals (prob < 0.4): 0.0000
- Neutral signals (0.4 ≤ prob ≤ 0.6): 0.6806
- Bearish signals (prob > 0.6): 0.0000
- Mean calibrated probability: 0.5601
- Std calibrated probability: 0.0072

## Top 15 Feature Importances

