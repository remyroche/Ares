# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251121_004413
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-21T00:44:13.143856
**Total Training Time:** 46.80s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 908
- **Features:** 58

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: -0.0004
- r2_std: 0.0007
- rmse_mean: 0.0000
- rmse_std: 0.0000

---

## Model Training Details

### analyst_lightgbm (lightgbm)

#### Pre-HPO Metrics
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: -0.1283
- r2_std: 0.1858
- rmse_mean: 0.0001
- rmse_std: 0.0000

**Fold Stability (Pre-HPO):**
- mse_cv: 1.2407
- mse_range: 0.0000
- mae_cv: 0.7467
- mae_range: 0.0000
- r2_cv: -1.4483
- r2_range: 0.4900
- rmse_cv: 0.6890
- rmse_range: 0.0001

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: -0.0004
- r2_std: 0.0007
- rmse_mean: 0.0000
- rmse_std: 0.0000

**Fold Stability (Post-HPO):**
- mse_cv: 1.1687
- mse_range: 0.0000
- mae_cv: 0.4174
- mae_range: 0.0000
- r2_cv: -1.8953
- r2_range: 0.0017
- rmse_cv: 0.8422
- rmse_range: 0.0001

**Improvement:**
- mse_abs_improvement: -0.0000
- mse_rel_improvement: -28.0402
- mae_abs_improvement: -0.0000
- mae_rel_improvement: -63.0146
- r2_abs_improvement: +0.1279
- r2_rel_improvement: -99.7176
- rmse_abs_improvement: -0.0000
- rmse_rel_improvement: -21.2055

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: -0.2525
- r2_std: 0.4222
- rmse_mean: 0.0001
- rmse_std: 0.0000

**Fold Stability (Pre-HPO):**
- mse_cv: 0.9414
- mse_range: 0.0000
- mae_cv: 0.7488
- mae_range: 0.0000
- r2_cv: -1.6721
- r2_range: 1.0933
- rmse_cv: 0.6628
- rmse_range: 0.0001

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: -0.6533
- r2_std: 0.9321
- rmse_mean: 0.0001
- rmse_std: 0.0001

**Fold Stability (Post-HPO):**
- mse_cv: 0.9932
- mse_range: 0.0000
- mae_cv: 0.8978
- mae_range: 0.0001
- r2_cv: -1.4268
- r2_range: 2.4034
- rmse_cv: 0.6771
- rmse_range: 0.0002

**Improvement:**
- mse_abs_improvement: +0.0000
- mse_rel_improvement: +62.7862
- mae_abs_improvement: +0.0000
- mae_rel_improvement: +45.5383
- r2_abs_improvement: -0.4008
- r2_rel_improvement: +158.7076
- rmse_abs_improvement: +0.0000
- rmse_rel_improvement: +26.7457

**Top 10 Important Features:**
- vectorbt_trend_consistency_10_price_returns: 13.7800
- resistance_level_1_5_price_returns: 13.5286
- volume_roc_1: 10.6584
- vectorbt_parkinson_volatility_50_vwap_27x_ratio_log_ratio_candlestick_engulfing_pattern_base_9x_ratio: 8.2728
- vectorbt_parabolic_sar_0.1_0.3: 6.3305
- ultimate_oscillator_7_14_28_returns_vwap: 6.0777
- volume_price_trend_vwap_div_candlestick_piercing_line_pattern_vwap_3x_ratio: 5.8778
- volume_price_trend_vwap_log_wavelet_energy_vwap_x_9x: 5.6697
- momentum_features: 4.8535
- volume_roc_5: 2.3062

---
