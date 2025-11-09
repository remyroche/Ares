# Training Report: ensemble_ensemble

**Session ID:** ensemble_ensemble_ETHUSDT_20251109_172620
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-09T17:26:20.265601
**Total Training Time:** 4.70s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 1,000
- **Features:** 100

---

## Best Model
**Name:** ensemble_ensemble_meta

**Metrics:**
- precision_mean: 0.9216
- precision_std: 0.0133
- accuracy_mean: 0.9399
- accuracy_std: 0.0091
- mse_mean: 0.0529
- mse_std: 0.0032
- f1_score_mean: 0.9126
- f1_score_std: 0.0141
- rmse_mean: 0.2298
- rmse_std: 0.0069
- r2_mean: 0.6991
- r2_std: 0.0179
- recall_mean: 0.9045
- recall_std: 0.0154
- mae_mean: 0.1182
- mae_std: 0.0037

---

## Model Training Details

### ensemble_ensemble_meta (lightgbm)

#### Pre-HPO Metrics
- precision_mean: 0.9314
- precision_std: 0.0102
- accuracy_mean: 0.9432
- accuracy_std: 0.0079
- mse_mean: 0.0478
- mse_std: 0.0039
- f1_score_mean: 0.9163
- f1_score_std: 0.0130
- rmse_mean: 0.2184
- rmse_std: 0.0088
- r2_mean: 0.7281
- r2_std: 0.0218
- recall_mean: 0.9033
- recall_std: 0.0154
- mae_mean: 0.0938
- mae_std: 0.0042

**Fold Stability (Pre-HPO):**
- precision_cv: 0.0109
- precision_range: 0.0298
- accuracy_cv: 0.0084
- accuracy_range: 0.0226
- mse_cv: 0.0818
- mse_range: 0.0111
- f1_score_cv: 0.0142
- f1_score_range: 0.0353
- rmse_cv: 0.0405
- rmse_range: 0.0251
- r2_cv: 0.0299
- r2_range: 0.0652
- recall_cv: 0.0170
- recall_range: 0.0409
- mae_cv: 0.0452
- mae_range: 0.0112

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- precision_mean: 0.9216
- precision_std: 0.0133
- accuracy_mean: 0.9399
- accuracy_std: 0.0091
- mse_mean: 0.0529
- mse_std: 0.0032
- f1_score_mean: 0.9126
- f1_score_std: 0.0141
- rmse_mean: 0.2298
- rmse_std: 0.0069
- r2_mean: 0.6991
- r2_std: 0.0179
- recall_mean: 0.9045
- recall_std: 0.0154
- mae_mean: 0.1182
- mae_std: 0.0037

**Fold Stability (Post-HPO):**
- precision_cv: 0.0145
- precision_range: 0.0389
- accuracy_cv: 0.0097
- accuracy_range: 0.0241
- mse_cv: 0.0601
- mse_range: 0.0091
- f1_score_cv: 0.0154
- f1_score_range: 0.0390
- rmse_cv: 0.0298
- rmse_range: 0.0196
- r2_cv: 0.0257
- r2_range: 0.0539
- recall_cv: 0.0171
- recall_range: 0.0411
- mae_cv: 0.0309
- mae_range: 0.0094

**Improvement:**
- precision_abs_improvement: -0.0099
- precision_rel_improvement: -1.0583
- accuracy_abs_improvement: -0.0033
- accuracy_rel_improvement: -0.3505
- mse_abs_improvement: +0.0051
- mse_rel_improvement: +10.6330
- f1_score_abs_improvement: -0.0037
- f1_score_rel_improvement: -0.4060
- rmse_abs_improvement: +0.0114
- rmse_rel_improvement: +5.2215
- r2_abs_improvement: -0.0289
- r2_rel_improvement: -3.9756
- recall_abs_improvement: +0.0012
- recall_rel_improvement: +0.1281
- mae_abs_improvement: +0.0244
- mae_rel_improvement: +26.0408

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 93.9900
- **Sharpe Ratio:** 93.9900
- **Sortino Ratio:** 93.9900

---
