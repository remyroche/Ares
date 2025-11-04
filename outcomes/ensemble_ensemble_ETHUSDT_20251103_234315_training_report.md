# Training Report: ensemble_ensemble

**Session ID:** ensemble_ensemble_ETHUSDT_20251103_234315
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-03T23:43:15.129329
**Total Training Time:** 326.25s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 1,000
- **Features:** 8

---

## Best Model
**Name:** ensemble_ensemble_meta

**Metrics:**
- f1_score_mean: 0.5062
- f1_score_std: 0.0146
- r2_mean: -0.0319
- r2_std: 0.0546
- mse_mean: 0.1548
- mse_std: 0.0097
- rmse_mean: 0.3933
- rmse_std: 0.0124
- accuracy_mean: 0.8030
- accuracy_std: 0.0163
- precision_mean: 0.6044
- precision_std: 0.0681
- recall_mean: 0.5213
- recall_std: 0.0121
- mae_mean: 0.2859
- mae_std: 0.0165

---

## Model Training Details

### ensemble_ensemble_meta (lightgbm)

#### Pre-HPO Metrics
- f1_score_mean: 0.5246
- f1_score_std: 0.0275
- r2_mean: -0.0979
- r2_std: 0.0762
- mse_mean: 0.1647
- mse_std: 0.0116
- rmse_mean: 0.4055
- rmse_std: 0.0144
- accuracy_mean: 0.7900
- accuracy_std: 0.0089
- precision_mean: 0.5681
- precision_std: 0.0435
- recall_mean: 0.5303
- recall_std: 0.0192
- mae_mean: 0.2927
- mae_std: 0.0156

**Fold Stability (Pre-HPO):**
- f1_score_cv: 0.0524
- f1_score_range: 0.0772
- r2_cv: -0.7791
- r2_range: 0.2011
- mse_cv: 0.0707
- mse_range: 0.0277
- rmse_cv: 0.0355
- rmse_range: 0.0342
- accuracy_cv: 0.0113
- accuracy_range: 0.0200
- precision_cv: 0.0766
- precision_range: 0.1223
- recall_cv: 0.0363
- recall_range: 0.0521
- mae_cv: 0.0534
- mae_range: 0.0452

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- f1_score_mean: 0.5062
- f1_score_std: 0.0146
- r2_mean: -0.0319
- r2_std: 0.0546
- mse_mean: 0.1548
- mse_std: 0.0097
- rmse_mean: 0.3933
- rmse_std: 0.0124
- accuracy_mean: 0.8030
- accuracy_std: 0.0163
- precision_mean: 0.6044
- precision_std: 0.0681
- recall_mean: 0.5213
- recall_std: 0.0121
- mae_mean: 0.2859
- mae_std: 0.0165

**Fold Stability (Post-HPO):**
- f1_score_cv: 0.0289
- f1_score_range: 0.0406
- r2_cv: -1.7116
- r2_range: 0.1359
- mse_cv: 0.0625
- mse_range: 0.0253
- rmse_cv: 0.0316
- rmse_range: 0.0325
- accuracy_cv: 0.0203
- accuracy_range: 0.0500
- precision_cv: 0.1126
- precision_range: 0.1969
- recall_cv: 0.0232
- recall_range: 0.0321
- mae_cv: 0.0577
- mae_range: 0.0508

**Improvement:**
- f1_score_abs_improvement: -0.0184
- f1_score_rel_improvement: -3.5001
- r2_abs_improvement: +0.0660
- r2_rel_improvement: -67.4025
- mse_abs_improvement: -0.0098
- mse_rel_improvement: -5.9588
- rmse_abs_improvement: -0.0122
- rmse_rel_improvement: -3.0125
- accuracy_abs_improvement: +0.0130
- accuracy_rel_improvement: +1.6456
- precision_abs_improvement: +0.0363
- precision_rel_improvement: +6.3858
- recall_abs_improvement: -0.0090
- recall_rel_improvement: -1.6884
- mae_abs_improvement: -0.0068
- mae_rel_improvement: -2.3120

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 49.2351
- **Sharpe Ratio:** 49.2351
- **Sortino Ratio:** inf

---
