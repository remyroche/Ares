# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251121_004459
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-21T00:44:59.964951
**Total Training Time:** 65.42s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 1,204
- **Features:** 58

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0001
- mae_std: 0.0001
- r2_mean: -0.0104
- r2_std: 0.0122
- rmse_mean: 0.0007
- rmse_std: 0.0008

---

## Model Training Details

### analyst_lightgbm (lightgbm)

#### Pre-HPO Metrics
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0002
- mae_std: 0.0003
- r2_mean: -0.0784
- r2_std: 0.1072
- rmse_mean: 0.0007
- rmse_std: 0.0008

**Fold Stability (Pre-HPO):**
- mse_cv: 1.2224
- mse_range: 0.0000
- mae_cv: 1.2485
- mae_range: 0.0007
- r2_cv: -1.3668
- r2_range: 0.2907
- rmse_cv: 1.1465
- rmse_range: 0.0017

#### Hyperparameter Optimization
- **Trials:** 5
- **Time:** 0.00s

#### Post-HPO Metrics
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0001
- mae_std: 0.0001
- r2_mean: -0.0104
- r2_std: 0.0122
- rmse_mean: 0.0007
- rmse_std: 0.0008

**Fold Stability (Post-HPO):**
- mse_cv: 1.2228
- mse_range: 0.0000
- mae_cv: 1.1453
- mae_range: 0.0003
- r2_cv: -1.1712
- r2_range: 0.0260
- rmse_cv: 1.1614
- rmse_range: 0.0017

**Improvement:**
- mse_abs_improvement: -0.0000
- mse_rel_improvement: -0.9463
- mae_abs_improvement: -0.0001
- mae_rel_improvement: -43.1097
- r2_abs_improvement: +0.0680
- r2_rel_improvement: -86.7294
- rmse_abs_improvement: -0.0000
- rmse_rel_improvement: -1.2029

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0002
- mae_std: 0.0003
- r2_mean: -0.2303
- r2_std: 0.3731
- rmse_mean: 0.0007
- rmse_std: 0.0008

**Fold Stability (Pre-HPO):**
- mse_cv: 1.2242
- mse_range: 0.0000
- mae_cv: 1.3099
- mae_range: 0.0007
- r2_cv: -1.6200
- r2_range: 0.9717
- rmse_cv: 1.1540
- rmse_range: 0.0018

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0002
- mae_std: 0.0003
- r2_mean: -0.7497
- r2_std: 1.3754
- rmse_mean: 0.0008
- rmse_std: 0.0009

**Fold Stability (Post-HPO):**
- mse_cv: 1.2275
- mse_range: 0.0000
- mae_cv: 1.2843
- mae_range: 0.0008
- r2_cv: -1.8345
- r2_range: 3.4957
- rmse_cv: 1.1377
- rmse_range: 0.0019

**Improvement:**
- mse_abs_improvement: +0.0000
- mse_rel_improvement: +4.3665
- mae_abs_improvement: +0.0000
- mae_rel_improvement: +9.1300
- r2_abs_improvement: -0.5194
- r2_rel_improvement: +225.5089
- rmse_abs_improvement: +0.0000
- rmse_rel_improvement: +2.9852

**Top 10 Important Features:**
- volume_price_trend_vwap_div_candlestick_piercing_line_pattern_vwap_3x_ratio: 18.8689
- volume_roc_1: 10.2130
- volume_price_trend_vwap_log_wavelet_energy_vwap_x_9x: 10.1593
- vectorbt_acceleration_trend_strength_5_20_price_returns: 8.2623
- hurst_exponent: 7.0856
- volume_price_trend_vwap_x_volume_entropy_10_volume_returns_vwap: 4.8631
- vectorbt_acceleration_consistency_5_10_price_returns: 4.7644
- volume_price_divergence_10: 4.1278
- vwma_20_price_returns_vwap_div_cycle_length_vwap_6x_ratio: 3.9018
- sma_50_returns_vwap: 2.8383

---
