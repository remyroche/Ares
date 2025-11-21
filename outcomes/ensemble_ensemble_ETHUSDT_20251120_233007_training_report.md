# Training Report: ensemble_ensemble

**Session ID:** ensemble_ensemble_ETHUSDT_20251120_233007
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-20T23:30:07.942036
**Total Training Time:** 0.97s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 1,204
- **Features:** 74

---

## Best Model
**Name:** ensemble_ensemble_meta

**Metrics:**
- mse_mean: 0.0000
- mse_std: 0.0000
- r2_mean: -0.1548
- r2_std: 0.3963
- mae_mean: 0.0004
- mae_std: 0.0007
- rmse_mean: 0.0008
- rmse_std: 0.0010

---

## Model Training Details

### ensemble_ensemble_meta (lightgbm)

#### Pre-HPO Metrics
- mse_mean: 0.0000
- mse_std: 0.0000
- r2_mean: -0.2048
- r2_std: 0.5218
- mae_mean: 0.0004
- mae_std: 0.0007
- rmse_mean: 0.0009
- rmse_std: 0.0011

**Fold Stability (Pre-HPO):**
- mse_cv: 1.3641
- mse_range: 0.0000
- r2_cv: -2.5475
- r2_range: 1.3992
- mae_cv: 1.6699
- mae_range: 0.0019
- rmse_cv: 1.2140
- rmse_range: 0.0025

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- mse_mean: 0.0000
- mse_std: 0.0000
- r2_mean: -0.1548
- r2_std: 0.3963
- mae_mean: 0.0004
- mae_std: 0.0007
- rmse_mean: 0.0008
- rmse_std: 0.0010

**Fold Stability (Post-HPO):**
- mse_cv: 1.3218
- mse_range: 0.0000
- r2_cv: -2.5606
- r2_range: 1.0526
- mae_cv: 1.6432
- mae_range: 0.0018
- rmse_cv: 1.1962
- rmse_range: 0.0024

**Improvement:**
- mse_abs_improvement: -0.0000
- mse_rel_improvement: -9.0619
- r2_abs_improvement: +0.0500
- r2_rel_improvement: -24.4348
- mae_abs_improvement: -0.0000
- mae_rel_improvement: -7.4538
- rmse_abs_improvement: -0.0000
- rmse_rel_improvement: -3.7996

---
