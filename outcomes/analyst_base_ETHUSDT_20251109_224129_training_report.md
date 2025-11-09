# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251109_224129
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-09T22:41:29.503224
**Total Training Time:** 143.04s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 75
- **Features:** 69

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**
- mse_mean: 0.0000
- mse_std: 0.0000
- recall_mean: 1.0000
- recall_std: 0.0000
- precision_mean: 1.0000
- precision_std: 0.0000
- r2_mean: 1.0000
- r2_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- accuracy_mean: 1.0000
- accuracy_std: 0.0000
- rmse_mean: 0.0000
- rmse_std: 0.0000
- f1_score_mean: 1.0000
- f1_score_std: 0.0000

---

## Model Training Details

### analyst_lightgbm (lightgbm)

#### Pre-HPO Metrics
- mse_mean: 0.0252
- mse_std: 0.0114
- recall_mean: 1.0000
- recall_std: 0.0000
- precision_mean: 1.0000
- precision_std: 0.0000
- r2_mean: 0.7482
- r2_std: 0.1135
- mae_mean: 0.1118
- mae_std: 0.0309
- accuracy_mean: 1.0000
- accuracy_std: 0.0000
- rmse_mean: 0.1549
- rmse_std: 0.0348
- f1_score_mean: 1.0000
- f1_score_std: 0.0000

**Fold Stability (Pre-HPO):**
- mse_cv: 0.4525
- mse_range: 0.0312
- recall_cv: 0.0000
- recall_range: 0.0000
- precision_cv: 0.0000
- precision_range: 0.0000
- r2_cv: 0.1517
- r2_range: 0.2703
- mae_cv: 0.2761
- mae_range: 0.0807
- accuracy_cv: 0.0000
- accuracy_range: 0.0000
- rmse_cv: 0.2250
- rmse_range: 0.0952
- f1_score_cv: 0.0000
- f1_score_range: 0.0000

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 13.11s
- **Best Parameters:** {'num_leaves': 32, 'learning_rate': 0.0895108713994946, 'feature_fraction': 0.7740774739667493, 'bagging_fraction': 0.9056757566556675, 'min_child_samples': 13}

#### Post-HPO Metrics
- mse_mean: 0.0000
- mse_std: 0.0000
- recall_mean: 1.0000
- recall_std: 0.0000
- precision_mean: 1.0000
- precision_std: 0.0000
- r2_mean: 1.0000
- r2_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- accuracy_mean: 1.0000
- accuracy_std: 0.0000
- rmse_mean: 0.0000
- rmse_std: 0.0000
- f1_score_mean: 1.0000
- f1_score_std: 0.0000

**Fold Stability (Post-HPO):**
- mse_cv: 0.1936
- mse_range: 0.0000
- recall_cv: 0.0000
- recall_range: 0.0000
- precision_cv: 0.0000
- precision_range: 0.0000
- r2_cv: 0.0000
- r2_range: 0.0000
- mae_cv: 0.0980
- mae_range: 0.0000
- accuracy_cv: 0.0000
- accuracy_range: 0.0000
- rmse_cv: 0.1064
- rmse_range: 0.0000
- f1_score_cv: 0.0000
- f1_score_range: 0.0000

**Improvement:**
- mse_abs_improvement: -0.0252
- mse_rel_improvement: -100.0000
- recall_abs_improvement: +0.0000
- recall_rel_improvement: +0.0000
- precision_abs_improvement: +0.0000
- precision_rel_improvement: +0.0000
- r2_abs_improvement: +0.2518
- r2_rel_improvement: +33.6598
- mae_abs_improvement: -0.1118
- mae_rel_improvement: -99.9842
- accuracy_abs_improvement: +0.0000
- accuracy_rel_improvement: +0.0000
- rmse_abs_improvement: -0.1548
- rmse_rel_improvement: -99.9823
- f1_score_abs_improvement: +0.0000
- f1_score_rel_improvement: +0.0000

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 100.0000
- **Sharpe Ratio:** 100.0000
- **Sortino Ratio:** 100.0000

---

### analyst_depthwise_cnn (depthwise_cnn)

#### Pre-HPO Metrics

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 0.00s

#### Post-HPO Metrics
- mse_mean: 0.1039
- mse_std: 0.0330
- recall_mean: 0.5000
- recall_std: 0.0000
- precision_mean: 0.4400
- precision_std: 0.0133
- r2_mean: 0.0172
- r2_std: 0.2028
- mae_mean: 0.1497
- mae_std: 0.0144
- accuracy_mean: 0.8800
- accuracy_std: 0.0267
- rmse_mean: 0.3177
- rmse_std: 0.0542
- f1_score_mean: 0.4680
- f1_score_std: 0.0074

**Fold Stability (Post-HPO):**
- mse_cv: 0.3181
- mse_range: 0.0801
- recall_cv: 0.0000
- recall_range: 0.0000
- precision_cv: 0.0303
- precision_range: 0.0333
- r2_cv: 11.8076
- r2_range: 0.5617
- mae_cv: 0.0959
- mae_range: 0.0388
- accuracy_cv: 0.0303
- accuracy_range: 0.0667
- rmse_cv: 0.1705
- rmse_range: 0.1320
- f1_score_cv: 0.0158
- f1_score_range: 0.0185

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 33.0000
- **Sharpe Ratio:** 33.0000
- **Sortino Ratio:** 88.0000

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- mse_mean: 0.0005
- mse_std: 0.0007
- recall_mean: 1.0000
- recall_std: 0.0000
- precision_mean: 1.0000
- precision_std: 0.0000
- r2_mean: 0.9953
- r2_std: 0.0062
- mae_mean: 0.0057
- mae_std: 0.0048
- accuracy_mean: 1.0000
- accuracy_std: 0.0000
- rmse_mean: 0.0166
- rmse_std: 0.0161
- f1_score_mean: 1.0000
- f1_score_std: 0.0000

**Fold Stability (Pre-HPO):**
- mse_cv: 1.3409
- mse_range: 0.0019
- recall_cv: 0.0000
- recall_range: 0.0000
- precision_cv: 0.0000
- precision_range: 0.0000
- r2_cv: 0.0063
- r2_range: 0.0161
- mae_cv: 0.8418
- mae_range: 0.0120
- accuracy_cv: 0.0000
- accuracy_range: 0.0000
- rmse_cv: 0.9699
- rmse_range: 0.0413
- f1_score_cv: 0.0000
- f1_score_range: 0.0000

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 109.40s
- **Best Parameters:** {'depth': 8, 'learning_rate': 0.09726261649881028, 'l2_leaf_reg': 7.976195410250031, 'border_count': 193}

#### Post-HPO Metrics
- mse_mean: 0.0010
- mse_std: 0.0012
- recall_mean: 1.0000
- recall_std: 0.0000
- precision_mean: 1.0000
- precision_std: 0.0000
- r2_mean: 0.9912
- r2_std: 0.0105
- mae_mean: 0.0111
- mae_std: 0.0084
- accuracy_mean: 1.0000
- accuracy_std: 0.0000
- rmse_mean: 0.0239
- rmse_std: 0.0208
- f1_score_mean: 1.0000
- f1_score_std: 0.0000

**Fold Stability (Post-HPO):**
- mse_cv: 1.2213
- mse_range: 0.0030
- recall_cv: 0.0000
- recall_range: 0.0000
- precision_cv: 0.0000
- precision_range: 0.0000
- r2_cv: 0.0106
- r2_range: 0.0262
- mae_cv: 0.7559
- mae_range: 0.0193
- accuracy_cv: 0.0000
- accuracy_range: 0.0000
- rmse_cv: 0.8699
- rmse_range: 0.0498
- f1_score_cv: 0.0000
- f1_score_range: 0.0000

**Improvement:**
- mse_abs_improvement: +0.0005
- mse_rel_improvement: +86.2355
- recall_abs_improvement: +0.0000
- recall_rel_improvement: +0.0000
- precision_abs_improvement: +0.0000
- precision_rel_improvement: +0.0000
- r2_abs_improvement: -0.0042
- r2_rel_improvement: -0.4178
- mae_abs_improvement: +0.0054
- mae_rel_improvement: +93.9353
- accuracy_abs_improvement: +0.0000
- accuracy_rel_improvement: +0.0000
- rmse_abs_improvement: +0.0072
- rmse_rel_improvement: +43.4345
- f1_score_abs_improvement: +0.0000
- f1_score_rel_improvement: +0.0000

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 100.0000
- **Sharpe Ratio:** 100.0000
- **Sortino Ratio:** 100.0000

**Top 10 Important Features:**
- candlestick_harami_cross_pattern_trend_adj_27x_ratio: 25.2043
- returns_volatility_20_price_returns_vwap_x_directional_signal_base_27x_ratio: 8.7252
- vectorbt_enhanced_obv_10_base_3x_ratio: 6.3227
- vectorbt_enhanced_obv_10_base_9x_ratio: 5.1932
- catboost_regime_1_prob: 4.0770
- wavelet_energy_base_6x_ratio: 3.6797
- wavelet_energy_base_6x_ratio_x_fibonacci_0.618_20_price_returns_vwap_x_27x: 3.6159
- fibonacci_0.236_5_price_returns_vwap_minus_volume_vwap_20_base_3x_ratio: 3.0700
- wavelet_energy_base_6x_ratio_log_fibonacci_0.786_10_price_returns_vwap_x_9x: 2.4899
- entropy_rate_20_base_3x_ratio_div_fractal_dimension_base_27x_ratio: 2.3326

---
