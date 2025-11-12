# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251112_001106
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-12T00:11:06.099308
**Total Training Time:** 152.80s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 14,023
- **Features:** 71

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**
- accuracy_mean: 0.7931
- accuracy_std: 0.0080
- recall_mean: 0.5000
- recall_std: 0.0000
- r2_mean: 0.0553
- r2_std: 0.0058
- precision_mean: 0.3965
- precision_std: 0.0040
- mae_mean: 0.3236
- mae_std: 0.0044
- rmse_mean: 0.3936
- rmse_std: 0.0055
- mse_mean: 0.1550
- mse_std: 0.0044
- f1_score_mean: 0.4423
- f1_score_std: 0.0025

---

## Model Training Details

### analyst_lightgbm (lightgbm)

#### Pre-HPO Metrics
- accuracy_mean: 0.8388
- accuracy_std: 0.0067
- recall_mean: 0.6331
- recall_std: 0.0094
- r2_mean: 0.2992
- r2_std: 0.0109
- precision_mean: 0.8312
- precision_std: 0.0071
- mae_mean: 0.2516
- mae_std: 0.0045
- rmse_mean: 0.3390
- rmse_std: 0.0053
- mse_mean: 0.1150
- mse_std: 0.0036
- f1_score_mean: 0.6630
- f1_score_std: 0.0121

**Fold Stability (Pre-HPO):**
- accuracy_cv: 0.0080
- accuracy_range: 0.0196
- recall_cv: 0.0149
- recall_range: 0.0257
- r2_cv: 0.0363
- r2_range: 0.0333
- precision_cv: 0.0086
- precision_range: 0.0164
- mae_cv: 0.0179
- mae_range: 0.0124
- rmse_cv: 0.0156
- rmse_range: 0.0157
- mse_cv: 0.0313
- mse_range: 0.0106
- f1_score_cv: 0.0183
- f1_score_range: 0.0317

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- accuracy_mean: 0.7931
- accuracy_std: 0.0080
- recall_mean: 0.5000
- recall_std: 0.0000
- r2_mean: 0.0553
- r2_std: 0.0058
- precision_mean: 0.3965
- precision_std: 0.0040
- mae_mean: 0.3236
- mae_std: 0.0044
- rmse_mean: 0.3936
- rmse_std: 0.0055
- mse_mean: 0.1550
- mse_std: 0.0044
- f1_score_mean: 0.4423
- f1_score_std: 0.0025

**Fold Stability (Post-HPO):**
- accuracy_cv: 0.0101
- accuracy_range: 0.0245
- recall_cv: 0.0000
- recall_range: 0.0000
- r2_cv: 0.1052
- r2_range: 0.0175
- precision_cv: 0.0101
- precision_range: 0.0123
- mae_cv: 0.0137
- mae_range: 0.0136
- rmse_cv: 0.0140
- rmse_range: 0.0168
- mse_cv: 0.0282
- mse_range: 0.0133
- f1_score_cv: 0.0057
- f1_score_range: 0.0076

**Improvement:**
- accuracy_abs_improvement: -0.0457
- accuracy_rel_improvement: -5.4498
- recall_abs_improvement: -0.1331
- recall_rel_improvement: -21.0258
- r2_abs_improvement: -0.2439
- r2_rel_improvement: -81.5156
- precision_abs_improvement: -0.4346
- precision_rel_improvement: -52.2928
- mae_abs_improvement: +0.0720
- mae_rel_improvement: +28.6292
- rmse_abs_improvement: +0.0546
- rmse_rel_improvement: +16.1092
- mse_abs_improvement: +0.0400
- mse_rel_improvement: +34.8071
- f1_score_abs_improvement: -0.2207
- f1_score_rel_improvement: -33.2904

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 79.3055
- **Sharpe Ratio:** 79.3055
- **Sortino Ratio:** 79.3055

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- accuracy_mean: 0.8561
- accuracy_std: 0.0052
- recall_mean: 0.6842
- recall_std: 0.0077
- r2_mean: 0.3650
- r2_std: 0.0149
- precision_mean: 0.8400
- precision_std: 0.0066
- mae_mean: 0.2351
- mae_std: 0.0049
- rmse_mean: 0.3227
- rmse_std: 0.0060
- mse_mean: 0.1042
- mse_std: 0.0039
- f1_score_mean: 0.7220
- f1_score_std: 0.0080

**Fold Stability (Pre-HPO):**
- accuracy_cv: 0.0061
- accuracy_range: 0.0149
- recall_cv: 0.0113
- recall_range: 0.0215
- r2_cv: 0.0408
- r2_range: 0.0457
- precision_cv: 0.0079
- precision_range: 0.0147
- mae_cv: 0.0208
- mae_range: 0.0129
- rmse_cv: 0.0185
- rmse_range: 0.0160
- mse_cv: 0.0370
- mse_range: 0.0104
- f1_score_cv: 0.0110
- f1_score_range: 0.0219

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- accuracy_mean: 0.8500
- accuracy_std: 0.0064
- recall_mean: 0.6657
- recall_std: 0.0094
- r2_mean: 0.3453
- r2_std: 0.0126
- precision_mean: 0.8373
- precision_std: 0.0127
- mae_mean: 0.2412
- mae_std: 0.0048
- rmse_mean: 0.3277
- rmse_std: 0.0058
- mse_mean: 0.1074
- mse_std: 0.0038
- f1_score_mean: 0.7019
- f1_score_std: 0.0112

**Fold Stability (Post-HPO):**
- accuracy_cv: 0.0076
- accuracy_range: 0.0199
- recall_cv: 0.0141
- recall_range: 0.0251
- r2_cv: 0.0365
- r2_range: 0.0383
- precision_cv: 0.0152
- precision_range: 0.0380
- mae_cv: 0.0197
- mae_range: 0.0135
- rmse_cv: 0.0176
- rmse_range: 0.0159
- mse_cv: 0.0354
- mse_range: 0.0104
- f1_score_cv: 0.0160
- f1_score_range: 0.0313

**Improvement:**
- accuracy_abs_improvement: -0.0061
- accuracy_rel_improvement: -0.7164
- recall_abs_improvement: -0.0185
- recall_rel_improvement: -2.7070
- r2_abs_improvement: -0.0196
- r2_rel_improvement: -5.3764
- precision_abs_improvement: -0.0027
- precision_rel_improvement: -0.3212
- mae_abs_improvement: +0.0062
- mae_rel_improvement: +2.6231
- rmse_abs_improvement: +0.0050
- rmse_rel_improvement: +1.5359
- mse_abs_improvement: +0.0032
- mse_rel_improvement: +3.0923
- f1_score_abs_improvement: -0.0201
- f1_score_rel_improvement: -2.7852

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 84.9962
- **Sharpe Ratio:** 84.9962
- **Sortino Ratio:** 84.9962

**Top 10 Important Features:**
- trend_score_14: 8.8408
- enhanced_volatility_50: 2.9443
- directional_signal: 2.6638
- enhanced_volatility_20: 2.4434
- resistance_level_1_20_price_returns: 2.2202
- lightgbm_regime_3_prob: 2.1927
- enhanced_volatility_100: 2.0868
- volume_std_10: 2.0787
- support_level_1_5_price_returns: 2.0455
- volume_std_50: 2.0345

---
