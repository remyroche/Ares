# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251112_231809
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-12T23:18:09.912245
**Total Training Time:** 122.09s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 11,590
- **Features:** 71

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**
- rmse_mean: 0.3788
- rmse_std: 0.0053
- accuracy_mean: 0.7994
- accuracy_std: 0.0071
- recall_mean: 0.5181
- recall_std: 0.0052
- mae_mean: 0.3022
- mae_std: 0.0042
- precision_mean: 0.8106
- precision_std: 0.0189
- f1_score_mean: 0.4803
- f1_score_std: 0.0104
- r2_mean: 0.1250
- r2_std: 0.0073
- mse_mean: 0.1435
- mse_std: 0.0040

---

## Model Training Details

### analyst_lightgbm (lightgbm)

#### Pre-HPO Metrics
- rmse_mean: 0.3309
- rmse_std: 0.0061
- accuracy_mean: 0.8493
- accuracy_std: 0.0081
- recall_mean: 0.6642
- recall_std: 0.0091
- mae_mean: 0.2417
- mae_std: 0.0045
- precision_mean: 0.8362
- precision_std: 0.0114
- f1_score_mean: 0.7002
- f1_score_std: 0.0112
- r2_mean: 0.3323
- r2_std: 0.0146
- mse_mean: 0.1095
- mse_std: 0.0041

**Fold Stability (Pre-HPO):**
- rmse_cv: 0.0186
- rmse_range: 0.0160
- accuracy_cv: 0.0095
- accuracy_range: 0.0203
- recall_cv: 0.0137
- recall_range: 0.0241
- mae_cv: 0.0187
- mae_range: 0.0134
- precision_cv: 0.0136
- precision_range: 0.0290
- f1_score_cv: 0.0161
- f1_score_range: 0.0293
- r2_cv: 0.0440
- r2_range: 0.0371
- mse_cv: 0.0371
- mse_range: 0.0106

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- rmse_mean: 0.3788
- rmse_std: 0.0053
- accuracy_mean: 0.7994
- accuracy_std: 0.0071
- recall_mean: 0.5181
- recall_std: 0.0052
- mae_mean: 0.3022
- mae_std: 0.0042
- precision_mean: 0.8106
- precision_std: 0.0189
- f1_score_mean: 0.4803
- f1_score_std: 0.0104
- r2_mean: 0.1250
- r2_std: 0.0073
- mse_mean: 0.1435
- mse_std: 0.0040

**Fold Stability (Post-HPO):**
- rmse_cv: 0.0141
- rmse_range: 0.0155
- accuracy_cv: 0.0089
- accuracy_range: 0.0203
- recall_cv: 0.0101
- recall_range: 0.0151
- mae_cv: 0.0140
- mae_range: 0.0117
- precision_cv: 0.0234
- precision_range: 0.0475
- f1_score_cv: 0.0217
- f1_score_range: 0.0304
- r2_cv: 0.0586
- r2_range: 0.0168
- mse_cv: 0.0281
- mse_range: 0.0117

**Improvement:**
- rmse_abs_improvement: +0.0479
- rmse_rel_improvement: +14.4775
- accuracy_abs_improvement: -0.0499
- accuracy_rel_improvement: -5.8722
- recall_abs_improvement: -0.1460
- recall_rel_improvement: -21.9886
- mae_abs_improvement: +0.0605
- mae_rel_improvement: +25.0246
- precision_abs_improvement: -0.0256
- precision_rel_improvement: -3.0615
- f1_score_abs_improvement: -0.2199
- f1_score_rel_improvement: -31.4005
- r2_abs_improvement: -0.2073
- r2_rel_improvement: -62.3837
- mse_abs_improvement: +0.0340
- mse_rel_improvement: +31.0320

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 79.9396
- **Sharpe Ratio:** 79.9396
- **Sortino Ratio:** 79.9396

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- rmse_mean: 0.3148
- rmse_std: 0.0052
- accuracy_mean: 0.8655
- accuracy_std: 0.0048
- recall_mean: 0.7107
- recall_std: 0.0115
- mae_mean: 0.2264
- mae_std: 0.0041
- precision_mean: 0.8462
- precision_std: 0.0096
- f1_score_mean: 0.7492
- f1_score_std: 0.0115
- r2_mean: 0.3956
- r2_std: 0.0161
- mse_mean: 0.0991
- mse_std: 0.0033

**Fold Stability (Pre-HPO):**
- rmse_cv: 0.0166
- rmse_range: 0.0137
- accuracy_cv: 0.0056
- accuracy_range: 0.0125
- recall_cv: 0.0162
- recall_range: 0.0341
- mae_cv: 0.0182
- mae_range: 0.0115
- precision_cv: 0.0113
- precision_range: 0.0257
- f1_score_cv: 0.0153
- f1_score_range: 0.0326
- r2_cv: 0.0406
- r2_range: 0.0426
- mse_cv: 0.0333
- mse_range: 0.0086

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- rmse_mean: 0.3864
- rmse_std: 0.0050
- accuracy_mean: 0.7965
- accuracy_std: 0.0068
- recall_mean: 0.5098
- recall_std: 0.0021
- mae_mean: 0.3092
- mae_std: 0.0030
- precision_mean: 0.8191
- precision_std: 0.0460
- f1_score_mean: 0.4633
- f1_score_std: 0.0031
- r2_mean: 0.0899
- r2_std: 0.0050
- mse_mean: 0.1493
- mse_std: 0.0039

**Fold Stability (Post-HPO):**
- rmse_cv: 0.0130
- rmse_range: 0.0134
- accuracy_cv: 0.0085
- accuracy_range: 0.0164
- recall_cv: 0.0041
- recall_range: 0.0060
- mae_cv: 0.0097
- mae_range: 0.0072
- precision_cv: 0.0562
- precision_range: 0.1227
- f1_score_cv: 0.0067
- f1_score_range: 0.0096
- r2_cv: 0.0555
- r2_range: 0.0120
- mse_cv: 0.0258
- mse_range: 0.0104

**Improvement:**
- rmse_abs_improvement: +0.0715
- rmse_rel_improvement: +22.7278
- accuracy_abs_improvement: -0.0689
- accuracy_rel_improvement: -7.9653
- recall_abs_improvement: -0.2009
- recall_rel_improvement: -28.2666
- mae_abs_improvement: +0.0828
- mae_rel_improvement: +36.5985
- precision_abs_improvement: -0.0271
- precision_rel_improvement: -3.1993
- f1_score_abs_improvement: -0.2858
- f1_score_rel_improvement: -38.1542
- r2_abs_improvement: -0.3057
- r2_rel_improvement: -77.2731
- mse_abs_improvement: +0.0502
- mse_rel_improvement: +50.6049

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 79.6549
- **Sharpe Ratio:** 79.6549
- **Sortino Ratio:** 79.6549

**Top 10 Important Features:**
- trend_score_14: 21.7528
- directional_signal: 5.2164
- vectorbt_sma_5: 4.9459
- resistance_level_1_20_price_returns: 4.1683
- enhanced_volatility_20: 3.0174
- enhanced_volatility_50: 2.9906
- vectorbt_enhanced_ad_line_50: 2.8111
- support_level_1_5_price_returns: 2.4533
- volume_std_10: 2.2295
- vectorbt_smoothed_obv_50: 2.0726

---
