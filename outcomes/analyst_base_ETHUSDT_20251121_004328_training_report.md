# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251121_004328
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-21T00:43:28.343341
**Total Training Time:** 44.79s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 708
- **Features:** 58

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: -0.0010
- r2_std: 0.0013
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
- r2_mean: -0.0248
- r2_std: 0.0342
- rmse_mean: 0.0000
- rmse_std: 0.0000

**Fold Stability (Pre-HPO):**
- mse_cv: 0.9280
- mse_range: 0.0000
- mae_cv: 0.3290
- mae_range: 0.0000
- r2_cv: -1.3785
- r2_range: 0.0870
- rmse_cv: 0.5636
- rmse_range: 0.0001

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: -0.0010
- r2_std: 0.0013
- rmse_mean: 0.0000
- rmse_std: 0.0000

**Fold Stability (Post-HPO):**
- mse_cv: 1.2195
- mse_range: 0.0000
- mae_cv: 0.5741
- mae_range: 0.0000
- r2_cv: -1.2247
- r2_range: 0.0026
- rmse_cv: 1.1050
- rmse_range: 0.0001

**Improvement:**
- mse_abs_improvement: -0.0000
- mse_rel_improvement: -19.4270
- mae_abs_improvement: -0.0000
- mae_rel_improvement: -64.8530
- r2_abs_improvement: +0.0238
- r2_rel_improvement: -95.8347
- rmse_abs_improvement: -0.0000
- rmse_rel_improvement: -30.8634

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: -0.1064
- r2_std: 0.1691
- rmse_mean: 0.0000
- rmse_std: 0.0000

**Fold Stability (Pre-HPO):**
- mse_cv: 1.1156
- mse_range: 0.0000
- mae_cv: 1.0463
- mae_range: 0.0000
- r2_cv: -1.5897
- r2_range: 0.4364
- rmse_cv: 0.9437
- rmse_range: 0.0001

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: -0.6440
- r2_std: 1.2522
- rmse_mean: 0.0001
- rmse_std: 0.0001

**Fold Stability (Post-HPO):**
- mse_cv: 1.4209
- mse_range: 0.0000
- mae_cv: 1.3876
- mae_range: 0.0001
- r2_cv: -1.9443
- r2_range: 3.1477
- rmse_cv: 1.0453
- rmse_range: 0.0001

**Improvement:**
- mse_abs_improvement: +0.0000
- mse_rel_improvement: +104.0364
- mae_abs_improvement: +0.0000
- mae_rel_improvement: +45.9482
- r2_abs_improvement: -0.5377
- r2_rel_improvement: +505.4111
- rmse_abs_improvement: +0.0000
- rmse_rel_improvement: +35.7645

**Top 10 Important Features:**
- vectorbt_trend_consistency_10_price_returns: 21.4101
- volume_roc_1: 11.6638
- vectorbt_parabolic_sar_0.1_0.3: 10.2624
- resistance_level_1_5_price_returns: 9.5193
- ultimate_oscillator_7_14_28_returns_vwap: 8.6514
- momentum_features: 3.1700
- momentum_21_price_returns: 3.1177
- volume_roc_5: 2.9790
- volume_price_trend_vwap_div_candlestick_piercing_line_pattern_vwap_3x_ratio: 2.9392
- macd_12_26_9_returns_vwap: 2.4466

---
