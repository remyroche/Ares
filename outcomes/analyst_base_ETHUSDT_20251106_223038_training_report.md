# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251106_223038
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-06T22:30:38.908642
**Total Training Time:** 29.58s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 1,000
- **Features:** 3

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**

---

## Model Training Details

### analyst_lightgbm (lightgbm)

#### Pre-HPO Metrics
- mae_mean: 0.3003
- mae_std: 0.0041
- f1_score_mean: 0.4493
- f1_score_std: 0.0026
- mse_mean: 0.1502
- mse_std: 0.0054
- recall_mean: 0.5000
- recall_std: 0.0000
- rmse_mean: 0.3875
- rmse_std: 0.0070
- accuracy_mean: 0.8160
- accuracy_std: 0.0086
- precision_mean: 0.4080
- precision_std: 0.0043
- r2_mean: -0.0008
- r2_std: 0.0008

**Fold Stability (Pre-HPO):**
- mae_cv: 0.0135
- mae_range: 0.0118
- f1_score_cv: 0.0058
- f1_score_range: 0.0076
- mse_cv: 0.0361
- mse_range: 0.0158
- recall_cv: 0.0000
- recall_range: 0.0000
- rmse_cv: 0.0181
- rmse_range: 0.0204
- accuracy_cv: 0.0105
- accuracy_range: 0.0250
- precision_cv: 0.0105
- precision_range: 0.0125
- r2_cv: -1.0284
- r2_range: 0.0022

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 21.64s
- **Best Parameters:** {'num_leaves': 62, 'learning_rate': 0.06224758646198911, 'feature_fraction': 0.9034691541034627, 'bagging_fraction': 0.8073105321497905, 'min_child_samples': 27}

#### Post-HPO Metrics

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 94.8581
- **Sharpe Ratio:** 94.8581
- **Sortino Ratio:** inf

---
