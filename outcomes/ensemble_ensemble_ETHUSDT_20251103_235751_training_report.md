# Training Report: ensemble_ensemble

**Session ID:** ensemble_ensemble_ETHUSDT_20251103_235751
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-03T23:57:51.777622
**Total Training Time:** 138.99s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 1,000
- **Features:** 8

---

## Best Model
**Name:** ensemble_ensemble_meta

**Metrics:**
- mse_mean: 0.1573
- mse_std: 0.0157
- f1_score_mean: 0.4759
- f1_score_std: 0.0256
- precision_mean: 0.4886
- precision_std: 0.0553
- mae_mean: 0.2839
- mae_std: 0.0180
- r2_mean: -0.0464
- r2_std: 0.0758
- rmse_mean: 0.3961
- rmse_std: 0.0194
- recall_mean: 0.5033
- recall_std: 0.0131
- accuracy_mean: 0.7940
- accuracy_std: 0.0128

---

## Model Training Details

### ensemble_ensemble_meta (lightgbm)

#### Pre-HPO Metrics
- mse_mean: 0.1675
- mse_std: 0.0123
- f1_score_mean: 0.5244
- f1_score_std: 0.0215
- precision_mean: 0.5731
- precision_std: 0.0394
- mae_mean: 0.2941
- mae_std: 0.0157
- r2_mean: -0.1159
- r2_std: 0.0604
- rmse_mean: 0.4090
- rmse_std: 0.0150
- recall_mean: 0.5294
- recall_std: 0.0154
- accuracy_mean: 0.7910
- accuracy_std: 0.0162

**Fold Stability (Pre-HPO):**
- mse_cv: 0.0733
- mse_range: 0.0335
- f1_score_cv: 0.0410
- f1_score_range: 0.0603
- precision_cv: 0.0687
- precision_range: 0.1078
- mae_cv: 0.0534
- mae_range: 0.0445
- r2_cv: -0.5216
- r2_range: 0.1658
- rmse_cv: 0.0366
- rmse_range: 0.0408
- recall_cv: 0.0292
- recall_range: 0.0439
- accuracy_cv: 0.0205
- accuracy_range: 0.0450

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- mse_mean: 0.1573
- mse_std: 0.0157
- f1_score_mean: 0.4759
- f1_score_std: 0.0256
- precision_mean: 0.4886
- precision_std: 0.0553
- mae_mean: 0.2839
- mae_std: 0.0180
- r2_mean: -0.0464
- r2_std: 0.0758
- rmse_mean: 0.3961
- rmse_std: 0.0194
- recall_mean: 0.5033
- recall_std: 0.0131
- accuracy_mean: 0.7940
- accuracy_std: 0.0128

**Fold Stability (Post-HPO):**
- mse_cv: 0.1000
- mse_range: 0.0472
- f1_score_cv: 0.0538
- f1_score_range: 0.0772
- precision_cv: 0.1133
- precision_range: 0.1744
- mae_cv: 0.0636
- mae_range: 0.0536
- r2_cv: -1.6351
- r2_range: 0.2120
- rmse_cv: 0.0490
- rmse_range: 0.0586
- recall_cv: 0.0261
- recall_range: 0.0339
- accuracy_cv: 0.0161
- accuracy_range: 0.0350

**Improvement:**
- mse_abs_improvement: -0.0103
- mse_rel_improvement: -6.1305
- f1_score_abs_improvement: -0.0485
- f1_score_rel_improvement: -9.2541
- precision_abs_improvement: -0.0845
- precision_rel_improvement: -14.7456
- mae_abs_improvement: -0.0102
- mae_rel_improvement: -3.4652
- r2_abs_improvement: +0.0695
- r2_rel_improvement: -59.9928
- rmse_abs_improvement: -0.0129
- rmse_rel_improvement: -3.1652
- recall_abs_improvement: -0.0262
- recall_rel_improvement: -4.9432
- accuracy_abs_improvement: +0.0030
- accuracy_rel_improvement: +0.3793

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 62.0010
- **Sharpe Ratio:** 62.0010
- **Sortino Ratio:** inf

---
