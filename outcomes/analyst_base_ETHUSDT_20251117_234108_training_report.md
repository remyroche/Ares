# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251117_234108
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-17T23:41:08.778170
**Total Training Time:** 17.04s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 572
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
- r2_mean: -0.0035
- r2_std: 0.0070

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
- r2_mean: 0.1940
- r2_std: 0.4032

**Fold Stability (Pre-HPO):**
- rmse_cv: 1.1758
- rmse_range: 0.0001
- mse_cv: 1.7766
- mse_range: 0.0000
- mae_cv: 0.5782
- mae_range: 0.0000
- r2_cv: 2.0781
- r2_range: 1.0300

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
- r2_mean: -0.0035
- r2_std: 0.0070

**Fold Stability (Post-HPO):**
- rmse_cv: 1.6496
- rmse_range: 0.0001
- mse_cv: 1.9835
- mse_range: 0.0000
- mae_cv: 0.9043
- mae_range: 0.0000
- r2_cv: -2.0000
- r2_range: 0.0174

**Improvement:**
- rmse_abs_improvement: -0.0000
- rmse_rel_improvement: -23.8994
- mse_abs_improvement: -0.0000
- mse_rel_improvement: -9.5495
- mae_abs_improvement: -0.0000
- mae_rel_improvement: -32.6247
- r2_abs_improvement: -0.1975
- r2_rel_improvement: -101.7952

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- rmse_mean: 0.0000
- rmse_std: 0.0000
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: 0.0000
- r2_std: 0.0000

**Fold Stability (Pre-HPO):**
- rmse_cv: 1.2852
- rmse_range: 0.0000
- mse_cv: 1.4098
- mse_range: 0.0000
- mae_cv: 1.3022
- mae_range: 0.0000
- r2_cv: inf
- r2_range: 0.0000

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
- r2_mean: 0.0000
- r2_std: 0.0000

**Fold Stability (Post-HPO):**
- rmse_cv: 0.9392
- rmse_range: 0.0000
- mse_cv: 1.2976
- mse_range: 0.0000
- mae_cv: 1.0641
- mae_range: 0.0000
- r2_cv: inf
- r2_range: 0.0000

**Improvement:**
- rmse_abs_improvement: +0.0000
- rmse_rel_improvement: +19.8597
- mse_abs_improvement: +0.0000
- mse_rel_improvement: +1.9663
- mae_abs_improvement: +0.0000
- mae_rel_improvement: +31.5571
- r2_abs_improvement: +0.0000
- r2_rel_improvement: +0.0000

**Top 10 Important Features:**
- vectorbt_acceleration_momentum_10_20_price_returns: 16.0447
- resistance_level_1_5_price_returns: 12.4384
- sma_10_returns_vwap: 7.1908
- fibonacci_0.5_10_price_returns_vwap_log_ratio_fibonacci_0.786_10_price_returns_vwap_x_9x: 5.7324
- hurst_exponent: 5.4109
- fibonacci_0.236_5_price_returns_vwap_log_wavelet_energy_vwap_x_27x: 3.8019
- fibonacci_0.236_5_price_returns_vwap_x_wavelet_energy_vwap_x_27x: 3.6478
- volume_momentum_20: 2.7822
- vectorbt_acceleration_volatility_10_20_price_returns: 2.7430
- macd_12_26_9_returns_vwap: 2.7301

---
