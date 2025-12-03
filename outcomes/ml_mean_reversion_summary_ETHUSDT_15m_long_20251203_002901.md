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
**VAL**: ACC=0.4713, F1=0.3518, Precision=0.6604, Recall=0.2397, AUC=0.4661, LogLoss=0.8455
**TEST**: ACC=0.4472, F1=0.4156, Precision=0.4598, Recall=0.3791, AUC=0.3926, LogLoss=0.8375

### Calibrated Model Performance

**TRAIN**: ACC=0.0000, F1=0.0000, Precision=0.0000, Recall=0.0000, AUC=0.0000, LogLoss=0.0000
**VAL**: ACC=0.4303, F1=0.0915, Precision=1.0000, Recall=0.0479, AUC=0.5240, LogLoss=0.6633
**TEST**: ACC=0.4889, F1=0.0280, Precision=1.0000, Recall=0.0142, AUC=0.5071, LogLoss=0.6886

### Class Balance

- Train positive rate (bearish): 0.0000
- Val positive rate (bearish): 0.5984
- Test positive rate (bearish): 0.5184

## Forward-Return Diagnostics

### Horizon 4 bars (60 minutes)

- n_samples: 633
- mean_fwd_return: 0.003167
- std_fwd_return: 0.074412
- **corr_prob_fwd**: -0.0234 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5103
- Returns by probability bucket:
  - bucket_0: 0.003167

### Horizon 8 bars (120 minutes)

- n_samples: 633
- mean_fwd_return: 0.005120
- std_fwd_return: 0.106113
- **corr_prob_fwd**: -0.0272 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5371
- Returns by probability bucket:
  - bucket_0: 0.005120

### Horizon 12 bars (180 minutes)

- n_samples: 633
- mean_fwd_return: 0.007080
- std_fwd_return: 0.131434
- **corr_prob_fwd**: -0.0268 (negative = good, higher prob → lower returns)
- **directional_accuracy**: 0.5403
- Returns by probability bucket:
  - bucket_0: 0.007080

## Signal Statistics

- Bullish signals (prob < 0.4): 0.0000
- Neutral signals (0.4 ≤ prob ≤ 0.6): 0.6218
- Bearish signals (prob > 0.6): 0.0100
- Mean calibrated probability: 0.5450
- Std calibrated probability: 0.0577

## Top 15 Feature Importances

