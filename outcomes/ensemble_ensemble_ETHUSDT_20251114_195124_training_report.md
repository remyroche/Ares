# Training Report: ensemble_ensemble

**Session ID:** ensemble_ensemble_ETHUSDT_20251114_195124
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-14T19:51:24.215783
**Total Training Time:** 2.60s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 1,000
- **Features:** 77

---

## Best Model
**Name:** ensemble_ensemble_meta

**Metrics:**
- rmse_mean: 0.0890
- rmse_std: 0.0706
- r2_mean: 0.5627
- r2_std: 0.4951
- mse_mean: 0.0129
- mse_std: 0.0157
- mae_mean: 0.0354
- mae_std: 0.0313

---

## Model Training Details

### ensemble_ensemble_meta (lightgbm)

#### Pre-HPO Metrics
- rmse_mean: 0.0883
- rmse_std: 0.0719
- r2_mean: 0.5634
- r2_std: 0.5161
- mse_mean: 0.0130
- mse_std: 0.0163
- mae_mean: 0.0346
- mae_std: 0.0332

**Fold Stability (Pre-HPO):**
- rmse_cv: 0.8150
- rmse_range: 0.2000
- r2_cv: 0.9160
- r2_range: 1.3583
- mse_cv: 1.2604
- mse_range: 0.0433
- mae_cv: 0.9597
- mae_range: 0.0794

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- rmse_mean: 0.0890
- rmse_std: 0.0706
- r2_mean: 0.5627
- r2_std: 0.4951
- mse_mean: 0.0129
- mse_std: 0.0157
- mae_mean: 0.0354
- mae_std: 0.0313

**Fold Stability (Post-HPO):**
- rmse_cv: 0.7931
- rmse_range: 0.1964
- r2_cv: 0.8799
- r2_range: 1.3036
- mse_cv: 1.2145
- mse_range: 0.0415
- mae_cv: 0.8849
- mae_range: 0.0755

**Improvement:**
- rmse_abs_improvement: +0.0007
- rmse_rel_improvement: +0.7892
- r2_abs_improvement: -0.0008
- r2_rel_improvement: -0.1343
- mse_abs_improvement: -0.0001
- mse_rel_improvement: -0.5629
- mae_abs_improvement: +0.0008
- mae_rel_improvement: +2.3495

---
