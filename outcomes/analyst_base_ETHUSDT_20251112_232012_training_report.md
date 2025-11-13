# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251112_232012
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-12T23:20:12.027563
**Total Training Time:** 212.11s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 14,023
- **Features:** 71

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**
- rmse_mean: 0.3847
- rmse_std: 0.0046
- accuracy_mean: 0.7953
- accuracy_std: 0.0087
- recall_mean: 0.5061
- recall_std: 0.0024
- mae_mean: 0.3171
- mae_std: 0.0040
- precision_mean: 0.8571
- precision_std: 0.0481
- f1_score_mean: 0.4552
- f1_score_std: 0.0074
- r2_mean: 0.0977
- r2_std: 0.0066
- mse_mean: 0.1480
- mse_std: 0.0035

---

## Model Training Details

### analyst_lightgbm (lightgbm)

#### Pre-HPO Metrics
- rmse_mean: 0.3396
- rmse_std: 0.0059
- accuracy_mean: 0.8378
- accuracy_std: 0.0056
- recall_mean: 0.6309
- recall_std: 0.0073
- mae_mean: 0.2520
- mae_std: 0.0043
- precision_mean: 0.8290
- precision_std: 0.0081
- f1_score_mean: 0.6602
- f1_score_std: 0.0095
- r2_mean: 0.2969
- r2_std: 0.0142
- mse_mean: 0.1154
- mse_std: 0.0040

**Fold Stability (Pre-HPO):**
- rmse_cv: 0.0173
- rmse_range: 0.0168
- accuracy_cv: 0.0067
- accuracy_range: 0.0171
- recall_cv: 0.0115
- recall_range: 0.0210
- mae_cv: 0.0169
- mae_range: 0.0121
- precision_cv: 0.0097
- precision_range: 0.0210
- f1_score_cv: 0.0144
- f1_score_range: 0.0276
- r2_cv: 0.0477
- r2_range: 0.0415
- mse_cv: 0.0346
- mse_range: 0.0114

#### Hyperparameter Optimization
- **Trials:** 5
- **Time:** 45.15s
- **Best Parameters:** {'num_leaves': 80, 'learning_rate': 0.06513789033645549, 'feature_fraction': 0.771397283549618, 'bagging_fraction': 0.8830876978246638, 'min_child_samples': 85, 'reg_alpha': 0.7010730144935106, 'reg_lambda': 2.2051147887475997}

#### Post-HPO Metrics
- rmse_mean: 0.3847
- rmse_std: 0.0046
- accuracy_mean: 0.7953
- accuracy_std: 0.0087
- recall_mean: 0.5061
- recall_std: 0.0024
- mae_mean: 0.3171
- mae_std: 0.0040
- precision_mean: 0.8571
- precision_std: 0.0481
- f1_score_mean: 0.4552
- f1_score_std: 0.0074
- r2_mean: 0.0977
- r2_std: 0.0066
- mse_mean: 0.1480
- mse_std: 0.0035

**Fold Stability (Post-HPO):**
- rmse_cv: 0.0119
- rmse_range: 0.0141
- accuracy_cv: 0.0110
- accuracy_range: 0.0263
- recall_cv: 0.0048
- recall_range: 0.0063
- mae_cv: 0.0127
- mae_range: 0.0125
- precision_cv: 0.0562
- precision_range: 0.1134
- f1_score_cv: 0.0163
- f1_score_range: 0.0208
- r2_cv: 0.0675
- r2_range: 0.0160
- mse_cv: 0.0239
- mse_range: 0.0109

**Improvement:**
- rmse_abs_improvement: +0.0451
- rmse_rel_improvement: +13.2814
- accuracy_abs_improvement: -0.0425
- accuracy_rel_improvement: -5.0727
- recall_abs_improvement: -0.1248
- recall_rel_improvement: -19.7831
- mae_abs_improvement: +0.0651
- mae_rel_improvement: +25.8194
- precision_abs_improvement: +0.0281
- precision_rel_improvement: +3.3863
- f1_score_abs_improvement: -0.2050
- f1_score_rel_improvement: -31.0544
- r2_abs_improvement: -0.1992
- r2_rel_improvement: -67.0873
- mse_abs_improvement: +0.0327
- mse_rel_improvement: +28.3066

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 79.5337
- **Sharpe Ratio:** 79.5337
- **Sortino Ratio:** 79.5337

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- rmse_mean: 0.3220
- rmse_std: 0.0067
- accuracy_mean: 0.8581
- accuracy_std: 0.0076
- recall_mean: 0.6883
- recall_std: 0.0144
- mae_mean: 0.2352
- mae_std: 0.0049
- precision_mean: 0.8434
- precision_std: 0.0078
- f1_score_mean: 0.7266
- f1_score_std: 0.0159
- r2_mean: 0.3679
- r2_std: 0.0179
- mse_mean: 0.1037
- mse_std: 0.0043

**Fold Stability (Pre-HPO):**
- rmse_cv: 0.0208
- rmse_range: 0.0179
- accuracy_cv: 0.0088
- accuracy_range: 0.0199
- recall_cv: 0.0209
- recall_range: 0.0431
- mae_cv: 0.0210
- mae_range: 0.0126
- precision_cv: 0.0092
- precision_range: 0.0210
- f1_score_cv: 0.0219
- f1_score_range: 0.0473
- r2_cv: 0.0487
- r2_range: 0.0533
- mse_cv: 0.0415
- mse_range: 0.0115

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- rmse_mean: 0.3506
- rmse_std: 0.0054
- accuracy_mean: 0.8222
- accuracy_std: 0.0073
- recall_mean: 0.5821
- recall_std: 0.0083
- mae_mean: 0.2688
- mae_std: 0.0037
- precision_mean: 0.8321
- precision_std: 0.0188
- f1_score_mean: 0.5924
- f1_score_std: 0.0126
- r2_mean: 0.2506
- r2_std: 0.0148
- mse_mean: 0.1229
- mse_std: 0.0038

**Fold Stability (Post-HPO):**
- rmse_cv: 0.0153
- rmse_range: 0.0148
- accuracy_cv: 0.0088
- accuracy_range: 0.0206
- recall_cv: 0.0143
- recall_range: 0.0243
- mae_cv: 0.0138
- mae_range: 0.0099
- precision_cv: 0.0226
- precision_range: 0.0588
- f1_score_cv: 0.0213
- f1_score_range: 0.0362
- r2_cv: 0.0591
- r2_range: 0.0453
- mse_cv: 0.0305
- mse_range: 0.0104

**Improvement:**
- rmse_abs_improvement: +0.0286
- rmse_rel_improvement: +8.8802
- accuracy_abs_improvement: -0.0359
- accuracy_rel_improvement: -4.1802
- recall_abs_improvement: -0.1062
- recall_rel_improvement: -15.4295
- mae_abs_improvement: +0.0336
- mae_rel_improvement: +14.2901
- precision_abs_improvement: -0.0113
- precision_rel_improvement: -1.3386
- f1_score_abs_improvement: -0.1342
- f1_score_rel_improvement: -18.4689
- r2_abs_improvement: -0.1172
- r2_rel_improvement: -31.8731
- mse_abs_improvement: +0.0192
- mse_rel_improvement: +18.5253

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 82.2222
- **Sharpe Ratio:** 82.2222
- **Sortino Ratio:** 82.2222

**Top 10 Important Features:**
- trend_score_14: 11.2225
- enhanced_volatility_50: 3.0378
- directional_signal: 2.9235
- resistance_level_1_20_price_returns: 2.5799
- support_level_1_5_price_returns: 2.5460
- enhanced_volatility_100: 2.2091
- volume_std_10: 2.1881
- volume_trend_strength_20_50: 2.0479
- enhanced_volatility_20: 1.9895
- volume_price_trend: 1.8680

---
