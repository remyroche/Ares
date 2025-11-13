# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251112_231610
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-12T23:16:10.115984
**Total Training Time:** 119.76s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 9,253
- **Features:** 71

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**
- rmse_mean: 0.3655
- rmse_std: 0.0019
- accuracy_mean: 0.8072
- accuracy_std: 0.0026
- recall_mean: 0.5473
- recall_std: 0.0092
- mae_mean: 0.2832
- mae_std: 0.0026
- precision_mean: 0.8562
- precision_std: 0.0273
- f1_score_mean: 0.5330
- f1_score_std: 0.0157
- r2_mean: 0.1979
- r2_std: 0.0084
- mse_mean: 0.1336
- mse_std: 0.0014

---

## Model Training Details

### analyst_lightgbm (lightgbm)

#### Pre-HPO Metrics
- rmse_mean: 0.3224
- rmse_std: 0.0052
- accuracy_mean: 0.8564
- accuracy_std: 0.0057
- recall_mean: 0.6916
- recall_std: 0.0086
- mae_mean: 0.2318
- mae_std: 0.0047
- precision_mean: 0.8422
- precision_std: 0.0142
- f1_score_mean: 0.7295
- f1_score_std: 0.0105
- r2_mean: 0.3756
- r2_std: 0.0192
- mse_mean: 0.1040
- mse_std: 0.0033

**Fold Stability (Pre-HPO):**
- rmse_cv: 0.0160
- rmse_range: 0.0118
- accuracy_cv: 0.0066
- accuracy_range: 0.0156
- recall_cv: 0.0124
- recall_range: 0.0239
- mae_cv: 0.0202
- mae_range: 0.0113
- precision_cv: 0.0168
- precision_range: 0.0394
- f1_score_cv: 0.0144
- f1_score_range: 0.0292
- r2_cv: 0.0511
- r2_range: 0.0527
- mse_cv: 0.0319
- mse_range: 0.0076

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- rmse_mean: 0.3655
- rmse_std: 0.0019
- accuracy_mean: 0.8072
- accuracy_std: 0.0026
- recall_mean: 0.5473
- recall_std: 0.0092
- mae_mean: 0.2832
- mae_std: 0.0026
- precision_mean: 0.8562
- precision_std: 0.0273
- f1_score_mean: 0.5330
- f1_score_std: 0.0157
- r2_mean: 0.1979
- r2_std: 0.0084
- mse_mean: 0.1336
- mse_std: 0.0014

**Fold Stability (Post-HPO):**
- rmse_cv: 0.0051
- rmse_range: 0.0050
- accuracy_cv: 0.0032
- accuracy_range: 0.0075
- recall_cv: 0.0169
- recall_range: 0.0270
- mae_cv: 0.0093
- mae_range: 0.0061
- precision_cv: 0.0319
- precision_range: 0.0849
- f1_score_cv: 0.0294
- f1_score_range: 0.0465
- r2_cv: 0.0424
- r2_range: 0.0249
- mse_cv: 0.0102
- mse_range: 0.0036

**Improvement:**
- rmse_abs_improvement: +0.0431
- rmse_rel_improvement: +13.3573
- accuracy_abs_improvement: -0.0492
- accuracy_rel_improvement: -5.7422
- recall_abs_improvement: -0.1443
- recall_rel_improvement: -20.8647
- mae_abs_improvement: +0.0514
- mae_rel_improvement: +22.1890
- precision_abs_improvement: +0.0140
- precision_rel_improvement: +1.6570
- f1_score_abs_improvement: -0.1965
- f1_score_rel_improvement: -26.9344
- r2_abs_improvement: -0.1778
- r2_rel_improvement: -47.3287
- mse_abs_improvement: +0.0296
- mse_rel_improvement: +28.4694

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 80.7198
- **Sharpe Ratio:** 80.7198
- **Sortino Ratio:** 80.7198

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- rmse_mean: 0.3121
- rmse_std: 0.0059
- accuracy_mean: 0.8687
- accuracy_std: 0.0036
- recall_mean: 0.7294
- recall_std: 0.0108
- mae_mean: 0.2224
- mae_std: 0.0045
- precision_mean: 0.8460
- precision_std: 0.0083
- f1_score_mean: 0.7658
- f1_score_std: 0.0102
- r2_mean: 0.4150
- r2_std: 0.0183
- mse_mean: 0.0974
- mse_std: 0.0037

**Fold Stability (Pre-HPO):**
- rmse_cv: 0.0188
- rmse_range: 0.0152
- accuracy_cv: 0.0042
- accuracy_range: 0.0097
- recall_cv: 0.0148
- recall_range: 0.0283
- mae_cv: 0.0203
- mae_range: 0.0104
- precision_cv: 0.0098
- precision_range: 0.0253
- f1_score_cv: 0.0134
- f1_score_range: 0.0266
- r2_cv: 0.0442
- r2_range: 0.0435
- mse_cv: 0.0375
- mse_range: 0.0095

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- rmse_mean: 0.3854
- rmse_std: 0.0034
- accuracy_mean: 0.7948
- accuracy_std: 0.0052
- recall_mean: 0.5161
- recall_std: 0.0030
- mae_mean: 0.3101
- mae_std: 0.0025
- precision_mean: 0.8310
- precision_std: 0.0494
- f1_score_mean: 0.4748
- f1_score_std: 0.0061
- r2_mean: 0.1084
- r2_std: 0.0048
- mse_mean: 0.1485
- mse_std: 0.0026

**Fold Stability (Post-HPO):**
- rmse_cv: 0.0088
- rmse_range: 0.0090
- accuracy_cv: 0.0066
- accuracy_range: 0.0124
- recall_cv: 0.0059
- recall_range: 0.0086
- mae_cv: 0.0082
- mae_range: 0.0065
- precision_cv: 0.0594
- precision_range: 0.1288
- f1_score_cv: 0.0129
- f1_score_range: 0.0168
- r2_cv: 0.0442
- r2_range: 0.0142
- mse_cv: 0.0176
- mse_range: 0.0070

**Improvement:**
- rmse_abs_improvement: +0.0732
- rmse_rel_improvement: +23.4682
- accuracy_abs_improvement: -0.0739
- accuracy_rel_improvement: -8.5096
- recall_abs_improvement: -0.2133
- recall_rel_improvement: -29.2418
- mae_abs_improvement: +0.0878
- mae_rel_improvement: +39.4663
- precision_abs_improvement: -0.0149
- precision_rel_improvement: -1.7666
- f1_score_abs_improvement: -0.2910
- f1_score_rel_improvement: -37.9986
- r2_abs_improvement: -0.3066
- r2_rel_improvement: -73.8888
- mse_abs_improvement: +0.0511
- mse_rel_improvement: +52.4019

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 79.4770
- **Sharpe Ratio:** 79.4770
- **Sortino Ratio:** 79.4770

**Top 10 Important Features:**
- trend_score_14: 19.4062
- directional_signal: 5.4707
- vectorbt_sma_5: 5.2457
- volume_trend_strength_20_50: 3.5190
- enhanced_volatility_50: 2.9174
- vectorbt_enhanced_ad_line_50: 2.7785
- resistance_level_1_20_price_returns: 2.4281
- enhanced_volatility_20: 2.2843
- resistance_level_1_10_price_returns: 2.2372
- vectorbt_momentum_acceleration_10_10_price_returns: 2.1086

---
