# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251110_202843
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-10T20:28:43.001918
**Total Training Time:** 526.07s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 75
- **Features:** 60

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**
- precision_mean: 1.0000
- precision_std: 0.0000
- rmse_mean: 0.0000
- rmse_std: 0.0000
- accuracy_mean: 1.0000
- accuracy_std: 0.0000
- r2_mean: 1.0000
- r2_std: 0.0000
- mse_mean: 0.0000
- mse_std: 0.0000
- recall_mean: 1.0000
- recall_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- f1_score_mean: 1.0000
- f1_score_std: 0.0000

---

## Model Training Details

### analyst_lightgbm (lightgbm)

#### Pre-HPO Metrics
- precision_mean: 1.0000
- precision_std: 0.0000
- rmse_mean: 0.1568
- rmse_std: 0.0357
- accuracy_mean: 1.0000
- accuracy_std: 0.0000
- r2_mean: 0.7446
- r2_std: 0.1093
- mse_mean: 0.0259
- mse_std: 0.0118
- recall_mean: 1.0000
- recall_std: 0.0000
- mae_mean: 0.1131
- mae_std: 0.0286
- f1_score_mean: 1.0000
- f1_score_std: 0.0000

**Fold Stability (Pre-HPO):**
- precision_cv: 0.0000
- precision_range: 0.0000
- rmse_cv: 0.2277
- rmse_range: 0.0923
- accuracy_cv: 0.0000
- accuracy_range: 0.0000
- r2_cv: 0.1469
- r2_range: 0.2660
- mse_cv: 0.4562
- mse_range: 0.0307
- recall_cv: 0.0000
- recall_range: 0.0000
- mae_cv: 0.2531
- mae_range: 0.0799
- f1_score_cv: 0.0000
- f1_score_range: 0.0000

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 37.77s
- **Best Parameters:** {'num_leaves': 98, 'learning_rate': 0.09561180251751898, 'feature_fraction': 0.8363395062976672, 'bagging_fraction': 0.9161970900689447, 'min_child_samples': 10}

#### Post-HPO Metrics
- precision_mean: 1.0000
- precision_std: 0.0000
- rmse_mean: 0.0000
- rmse_std: 0.0000
- accuracy_mean: 1.0000
- accuracy_std: 0.0000
- r2_mean: 1.0000
- r2_std: 0.0000
- mse_mean: 0.0000
- mse_std: 0.0000
- recall_mean: 1.0000
- recall_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- f1_score_mean: 1.0000
- f1_score_std: 0.0000

**Fold Stability (Post-HPO):**
- precision_cv: 0.0000
- precision_range: 0.0000
- rmse_cv: 0.1064
- rmse_range: 0.0000
- accuracy_cv: 0.0000
- accuracy_range: 0.0000
- r2_cv: 0.0000
- r2_range: 0.0000
- mse_cv: 0.1936
- mse_range: 0.0000
- recall_cv: 0.0000
- recall_range: 0.0000
- mae_cv: 0.0980
- mae_range: 0.0000
- f1_score_cv: 0.0000
- f1_score_range: 0.0000

**Improvement:**
- precision_abs_improvement: +0.0000
- precision_rel_improvement: +0.0000
- rmse_abs_improvement: -0.1568
- rmse_rel_improvement: -99.9911
- accuracy_abs_improvement: +0.0000
- accuracy_rel_improvement: +0.0000
- r2_abs_improvement: +0.2554
- r2_rel_improvement: +34.3003
- mse_abs_improvement: -0.0259
- mse_rel_improvement: -100.0000
- recall_abs_improvement: +0.0000
- recall_rel_improvement: +0.0000
- mae_abs_improvement: -0.1131
- mae_rel_improvement: -99.9920
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
- precision_mean: 0.4400
- precision_std: 0.0133
- rmse_mean: 0.3239
- rmse_std: 0.0396
- accuracy_mean: 0.8800
- accuracy_std: 0.0267
- r2_mean: -0.0110
- r2_std: 0.0461
- mse_mean: 0.1065
- mse_std: 0.0235
- recall_mean: 0.5000
- recall_std: 0.0000
- mae_mean: 0.1680
- mae_std: 0.0264
- f1_score_mean: 0.4680
- f1_score_std: 0.0074

**Fold Stability (Post-HPO):**
- precision_cv: 0.0303
- precision_range: 0.0333
- rmse_cv: 0.1222
- rmse_range: 0.1104
- accuracy_cv: 0.0303
- accuracy_range: 0.0667
- r2_cv: -4.1805
- r2_range: 0.1264
- mse_cv: 0.2205
- mse_range: 0.0665
- recall_cv: 0.0000
- recall_range: 0.0000
- mae_cv: 0.1574
- mae_range: 0.0793
- f1_score_cv: 0.0158
- f1_score_range: 0.0185

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 33.0000
- **Sharpe Ratio:** 33.0000
- **Sortino Ratio:** 88.0000

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- precision_mean: 1.0000
- precision_std: 0.0000
- rmse_mean: 0.0162
- rmse_std: 0.0171
- accuracy_mean: 1.0000
- accuracy_std: 0.0000
- r2_mean: 0.9951
- r2_std: 0.0072
- mse_mean: 0.0006
- mse_std: 0.0008
- recall_mean: 1.0000
- recall_std: 0.0000
- mae_mean: 0.0055
- mae_std: 0.0051
- f1_score_mean: 1.0000
- f1_score_std: 0.0000

**Fold Stability (Pre-HPO):**
- precision_cv: 0.0000
- precision_range: 0.0000
- rmse_cv: 1.0562
- rmse_range: 0.0464
- accuracy_cv: 0.0000
- accuracy_range: 0.0000
- r2_cv: 0.0073
- r2_range: 0.0191
- mse_cv: 1.5171
- mse_range: 0.0022
- recall_cv: 0.0000
- recall_range: 0.0000
- mae_cv: 0.9315
- mae_range: 0.0139
- f1_score_cv: 0.0000
- f1_score_range: 0.0000

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 364.34s
- **Best Parameters:** {'depth': 8, 'learning_rate': 0.06371651421518383, 'l2_leaf_reg': 5.01249477568232, 'border_count': 246}

#### Post-HPO Metrics
- precision_mean: 1.0000
- precision_std: 0.0000
- rmse_mean: 0.0191
- rmse_std: 0.0169
- accuracy_mean: 1.0000
- accuracy_std: 0.0000
- r2_mean: 0.9943
- r2_std: 0.0077
- mse_mean: 0.0007
- mse_std: 0.0009
- recall_mean: 1.0000
- recall_std: 0.0000
- mae_mean: 0.0083
- mae_std: 0.0062
- f1_score_mean: 1.0000
- f1_score_std: 0.0000

**Fold Stability (Post-HPO):**
- precision_cv: 0.0000
- precision_range: 0.0000
- rmse_cv: 0.8857
- rmse_range: 0.0431
- accuracy_cv: 0.0000
- accuracy_range: 0.0000
- r2_cv: 0.0078
- r2_range: 0.0201
- mse_cv: 1.3753
- mse_range: 0.0023
- recall_cv: 0.0000
- recall_range: 0.0000
- mae_cv: 0.7438
- mae_range: 0.0157
- f1_score_cv: 0.0000
- f1_score_range: 0.0000

**Improvement:**
- precision_abs_improvement: +0.0000
- precision_rel_improvement: +0.0000
- rmse_abs_improvement: +0.0029
- rmse_rel_improvement: +17.9561
- accuracy_abs_improvement: +0.0000
- accuracy_rel_improvement: +0.0000
- r2_abs_improvement: -0.0008
- r2_rel_improvement: -0.0784
- mse_abs_improvement: +0.0001
- mse_rel_improvement: +17.3523
- recall_abs_improvement: +0.0000
- recall_rel_improvement: +0.0000
- mae_abs_improvement: +0.0028
- mae_rel_improvement: +50.9950
- f1_score_abs_improvement: +0.0000
- f1_score_rel_improvement: +0.0000

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 100.0000
- **Sharpe Ratio:** 100.0000
- **Sortino Ratio:** 100.0000

**Top 10 Important Features:**
- candlestick_harami_cross_pattern_trend_adj_27x_ratio: 14.1423
- returns_volatility_20_price_returns_vwap_x_directional_signal_base_27x_ratio: 11.5244
- candlestick_harami_cross_pattern_vwap_27x_ratio: 6.1119
- fibonacci_0.236_5_price_returns_base_div_vectorbt_enhanced_obv_10_base_9x_ratio: 6.0375
- vectorbt_enhanced_obv_10_base_9x_ratio: 5.9951
- directional_signal_vwap: 4.7486
- volume_price_trend_vwap: 4.2285
- vectorbt_enhanced_obv_10_base_3x_ratio: 4.1896
- fibonacci_0.236_5_price_returns_vwap_minus_volume_vwap_20_base_3x_ratio: 4.1028
- candlestick_dark_cloud_cover_pattern_vwap_3x_ratio_minus_fibonacci_0.786_10_price_returns_vwap: 3.1888

---
