# Training Report: ensemble_ensemble

**Session ID:** ensemble_ensemble_ETHUSDT_20251114_194331
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-14T19:43:31.431015
**Total Training Time:** 2.24s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 1,000
- **Features:** 77

---

## Best Model
**Name:** ensemble_ensemble_meta

**Metrics:**
- mse_mean: 0.0121
- mse_std: 0.0140
- rmse_mean: 0.0878
- rmse_std: 0.0666
- mae_mean: 0.0342
- mae_std: 0.0294
- r2_mean: 0.5836
- r2_std: 0.4436

---

## Model Training Details

### ensemble_ensemble_meta (lightgbm)

#### Pre-HPO Metrics
- mse_mean: 0.0117
- mse_std: 0.0143
- rmse_mean: 0.0851
- rmse_std: 0.0670
- mae_mean: 0.0316
- mae_std: 0.0293
- r2_mean: 0.6031
- r2_std: 0.4521

**Fold Stability (Pre-HPO):**
- mse_cv: 1.2218
- mse_range: 0.0379
- rmse_cv: 0.7870
- rmse_range: 0.1859
- mae_cv: 0.9271
- mae_range: 0.0694
- r2_cv: 0.7497
- r2_range: 1.1853

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- mse_mean: 0.0121
- mse_std: 0.0140
- rmse_mean: 0.0878
- rmse_std: 0.0666
- mae_mean: 0.0342
- mae_std: 0.0294
- r2_mean: 0.5836
- r2_std: 0.4436

**Fold Stability (Post-HPO):**
- mse_cv: 1.1517
- mse_range: 0.0368
- rmse_cv: 0.7583
- rmse_range: 0.1832
- mae_cv: 0.8608
- mae_range: 0.0665
- r2_cv: 0.7600
- r2_range: 1.1495

**Improvement:**
- mse_abs_improvement: +0.0004
- mse_rel_improvement: +3.6404
- rmse_abs_improvement: +0.0027
- rmse_rel_improvement: +3.2252
- mae_abs_improvement: +0.0026
- mae_rel_improvement: +8.1198
- r2_abs_improvement: -0.0194
- r2_rel_improvement: -3.2205

---
