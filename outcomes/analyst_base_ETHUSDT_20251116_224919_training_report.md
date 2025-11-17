# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251116_224919
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-16T22:49:19.109868
**Total Training Time:** 0.30s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 1,204
- **Features:** 72

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**
- r2_mean: 0.7980
- r2_std: 0.4040
- mse_mean: 0.0020
- mse_std: 0.0040
- mae_mean: 0.0020
- mae_std: 0.0040
- rmse_mean: 0.0200
- rmse_std: 0.0400

---

## Model Training Details

### analyst_lightgbm (lightgbm)

#### Pre-HPO Metrics
- r2_mean: 0.7980
- r2_std: 0.4040
- mse_mean: 0.0020
- mse_std: 0.0040
- mae_mean: 0.0020
- mae_std: 0.0040
- rmse_mean: 0.0200
- rmse_std: 0.0400

**Fold Stability (Pre-HPO):**
- r2_cv: 0.5063
- r2_range: 1.0101
- mse_cv: 2.0000
- mse_range: 0.0100
- mae_cv: 2.0000
- mae_range: 0.0100
- rmse_cv: 2.0000
- rmse_range: 0.0999

#### Hyperparameter Optimization
- **Trials:** 5
- **Time:** 0.00s

#### Post-HPO Metrics
- r2_mean: 0.7980
- r2_std: 0.4040
- mse_mean: 0.0020
- mse_std: 0.0040
- mae_mean: 0.0020
- mae_std: 0.0040
- rmse_mean: 0.0200
- rmse_std: 0.0400

**Fold Stability (Post-HPO):**
- r2_cv: 0.5063
- r2_range: 1.0101
- mse_cv: 2.0000
- mse_range: 0.0100
- mae_cv: 2.0000
- mae_range: 0.0100
- rmse_cv: 2.0000
- rmse_range: 0.0999

**Improvement:**
- r2_abs_improvement: +0.0000
- r2_rel_improvement: +0.0000
- mse_abs_improvement: +0.0000
- mse_rel_improvement: +0.0000
- mae_abs_improvement: +0.0000
- mae_rel_improvement: +0.0000
- rmse_abs_improvement: +0.0000
- rmse_rel_improvement: +0.0000

---
