# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251117_234125
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-17T23:41:25.830787
**Total Training Time:** 18.09s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 738
- **Features:** 72

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**
- rmse_mean: 0.0000
- rmse_std: 0.0000
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: -0.0023
- r2_std: 0.0035

---

## Model Training Details

### analyst_lightgbm (lightgbm)

#### Pre-HPO Metrics
- rmse_mean: 0.0000
- rmse_std: 0.0000
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: -1.0705
- r2_std: 2.1322

**Fold Stability (Pre-HPO):**
- rmse_cv: 0.7123
- rmse_range: 0.0001
- mse_cv: 1.1442
- mse_range: 0.0000
- mae_cv: 0.7543
- mae_range: 0.0000
- r2_cv: -1.9918
- r2_range: 5.3349

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- rmse_mean: 0.0000
- rmse_std: 0.0000
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: -0.0023
- r2_std: 0.0035

**Fold Stability (Post-HPO):**
- rmse_cv: 1.3559
- rmse_range: 0.0001
- mse_cv: 1.8322
- mse_range: 0.0000
- mae_cv: 0.7449
- mae_range: 0.0000
- r2_cv: -1.5158
- r2_range: 0.0090

**Improvement:**
- rmse_abs_improvement: -0.0000
- rmse_rel_improvement: -40.9148
- mse_abs_improvement: -0.0000
- mse_rel_improvement: -34.2574
- mae_abs_improvement: -0.0000
- mae_rel_improvement: -71.6651
- r2_abs_improvement: +1.0682
- r2_rel_improvement: -99.7841

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- rmse_mean: 0.0000
- rmse_std: 0.0000
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: -2.0264
- r2_std: 3.5098

**Fold Stability (Pre-HPO):**
- rmse_cv: 1.1608
- rmse_range: 0.0001
- mse_cv: 1.5062
- mse_range: 0.0000
- mae_cv: 1.0747
- mae_range: 0.0000
- r2_cv: -1.7321
- r2_range: 8.1055

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- rmse_mean: 0.0000
- rmse_std: 0.0000
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: -5.7505
- r2_std: 9.9601

**Fold Stability (Post-HPO):**
- rmse_cv: 1.2585
- rmse_range: 0.0001
- mse_cv: 1.6329
- mse_range: 0.0000
- mae_cv: 1.3137
- mae_range: 0.0001
- r2_cv: -1.7321
- r2_range: 23.0020

**Improvement:**
- rmse_abs_improvement: +0.0000
- rmse_rel_improvement: +49.9715
- mse_abs_improvement: +0.0000
- mse_rel_improvement: +147.5646
- mae_abs_improvement: +0.0000
- mae_rel_improvement: +111.7750
- r2_abs_improvement: -3.7241
- r2_rel_improvement: +183.7827

**Top 10 Important Features:**
- vectorbt_acceleration_momentum_10_20_price_returns: 21.0195
- resistance_level_1_5_price_returns: 17.1346
- sma_10_returns_vwap: 5.2928
- fibonacci_0.236_5_price_returns_vwap_x_wavelet_energy_vwap_x_27x: 4.5217
- volume_oscillator_5_15: 4.1568
- hurst_exponent: 4.0484
- fibonacci_0.5_10_price_returns_vwap_log_ratio_fibonacci_0.786_10_price_returns_vwap_x_9x: 3.6415
- fibonacci_0.236_5_price_returns_vwap_log_wavelet_energy_vwap_x_27x: 2.9544
- macd_12_26_9_returns_vwap: 2.5496
- ar_1_coefficients_20_base_9x_ratio: 2.3572

---
