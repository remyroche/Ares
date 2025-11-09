# Training Report: ensemble_ensemble

**Session ID:** ensemble_ensemble_ETHUSDT_20251109_172335
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-09T17:23:35.902734
**Total Training Time:** 4.45s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 1,000
- **Features:** 100

---

## Best Model
**Name:** ensemble_ensemble_meta

**Metrics:**
- mae_mean: 0.1182
- mae_std: 0.0037
- rmse_mean: 0.2298
- rmse_std: 0.0069
- mse_mean: 0.0529
- mse_std: 0.0032
- recall_mean: 0.9045
- recall_std: 0.0154
- accuracy_mean: 0.9399
- accuracy_std: 0.0091
- r2_mean: 0.6991
- r2_std: 0.0179
- f1_score_mean: 0.9126
- f1_score_std: 0.0141
- precision_mean: 0.9216
- precision_std: 0.0133

---

## Model Training Details

### ensemble_ensemble_meta (lightgbm)

#### Pre-HPO Metrics
- mae_mean: 0.0938
- mae_std: 0.0042
- rmse_mean: 0.2184
- rmse_std: 0.0088
- mse_mean: 0.0478
- mse_std: 0.0039
- recall_mean: 0.9033
- recall_std: 0.0154
- accuracy_mean: 0.9432
- accuracy_std: 0.0079
- r2_mean: 0.7281
- r2_std: 0.0218
- f1_score_mean: 0.9163
- f1_score_std: 0.0130
- precision_mean: 0.9314
- precision_std: 0.0102

**Fold Stability (Pre-HPO):**
- mae_cv: 0.0452
- mae_range: 0.0112
- rmse_cv: 0.0405
- rmse_range: 0.0251
- mse_cv: 0.0818
- mse_range: 0.0111
- recall_cv: 0.0170
- recall_range: 0.0409
- accuracy_cv: 0.0084
- accuracy_range: 0.0226
- r2_cv: 0.0299
- r2_range: 0.0652
- f1_score_cv: 0.0142
- f1_score_range: 0.0353
- precision_cv: 0.0109
- precision_range: 0.0298

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- mae_mean: 0.1182
- mae_std: 0.0037
- rmse_mean: 0.2298
- rmse_std: 0.0069
- mse_mean: 0.0529
- mse_std: 0.0032
- recall_mean: 0.9045
- recall_std: 0.0154
- accuracy_mean: 0.9399
- accuracy_std: 0.0091
- r2_mean: 0.6991
- r2_std: 0.0179
- f1_score_mean: 0.9126
- f1_score_std: 0.0141
- precision_mean: 0.9216
- precision_std: 0.0133

**Fold Stability (Post-HPO):**
- mae_cv: 0.0309
- mae_range: 0.0094
- rmse_cv: 0.0298
- rmse_range: 0.0196
- mse_cv: 0.0601
- mse_range: 0.0091
- recall_cv: 0.0171
- recall_range: 0.0411
- accuracy_cv: 0.0097
- accuracy_range: 0.0241
- r2_cv: 0.0257
- r2_range: 0.0539
- f1_score_cv: 0.0154
- f1_score_range: 0.0390
- precision_cv: 0.0145
- precision_range: 0.0389

**Improvement:**
- mae_abs_improvement: +0.0244
- mae_rel_improvement: +26.0408
- rmse_abs_improvement: +0.0114
- rmse_rel_improvement: +5.2215
- mse_abs_improvement: +0.0051
- mse_rel_improvement: +10.6330
- recall_abs_improvement: +0.0012
- recall_rel_improvement: +0.1281
- accuracy_abs_improvement: -0.0033
- accuracy_rel_improvement: -0.3505
- r2_abs_improvement: -0.0289
- r2_rel_improvement: -3.9756
- f1_score_abs_improvement: -0.0037
- f1_score_rel_improvement: -0.4060
- precision_abs_improvement: -0.0099
- precision_rel_improvement: -1.0583

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 93.9900
- **Sharpe Ratio:** 93.9900
- **Sortino Ratio:** 93.9900

---
