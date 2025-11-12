# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251112_002727
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-12T00:27:27.302679
**Total Training Time:** 97.86s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 14,023
- **Features:** 71

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**
- f1_score_mean: 0.5549
- f1_score_std: 0.0102
- accuracy_mean: 0.8102
- accuracy_std: 0.0060
- r2_mean: 0.1816
- r2_std: 0.0133
- recall_mean: 0.5580
- recall_std: 0.0072
- mae_mean: 0.2955
- mae_std: 0.0036
- mse_mean: 0.1342
- mse_std: 0.0028
- precision_mean: 0.7753
- precision_std: 0.0325
- rmse_mean: 0.3663
- rmse_std: 0.0038

---

## Model Training Details

### analyst_lightgbm (lightgbm)

#### Pre-HPO Metrics
- f1_score_mean: 0.6630
- f1_score_std: 0.0121
- accuracy_mean: 0.8388
- accuracy_std: 0.0067
- r2_mean: 0.2992
- r2_std: 0.0109
- recall_mean: 0.6331
- recall_std: 0.0094
- mae_mean: 0.2516
- mae_std: 0.0045
- mse_mean: 0.1150
- mse_std: 0.0036
- precision_mean: 0.8312
- precision_std: 0.0071
- rmse_mean: 0.3390
- rmse_std: 0.0053

**Fold Stability (Pre-HPO):**
- f1_score_cv: 0.0183
- f1_score_range: 0.0317
- accuracy_cv: 0.0080
- accuracy_range: 0.0196
- r2_cv: 0.0363
- r2_range: 0.0333
- recall_cv: 0.0149
- recall_range: 0.0257
- mae_cv: 0.0179
- mae_range: 0.0124
- mse_cv: 0.0313
- mse_range: 0.0106
- precision_cv: 0.0086
- precision_range: 0.0164
- rmse_cv: 0.0156
- rmse_range: 0.0157

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- f1_score_mean: 0.5549
- f1_score_std: 0.0102
- accuracy_mean: 0.8102
- accuracy_std: 0.0060
- r2_mean: 0.1816
- r2_std: 0.0133
- recall_mean: 0.5580
- recall_std: 0.0072
- mae_mean: 0.2955
- mae_std: 0.0036
- mse_mean: 0.1342
- mse_std: 0.0028
- precision_mean: 0.7753
- precision_std: 0.0325
- rmse_mean: 0.3663
- rmse_std: 0.0038

**Fold Stability (Post-HPO):**
- f1_score_cv: 0.0184
- f1_score_range: 0.0289
- accuracy_cv: 0.0074
- accuracy_range: 0.0160
- r2_cv: 0.0735
- r2_range: 0.0367
- recall_cv: 0.0129
- recall_range: 0.0190
- mae_cv: 0.0121
- mae_range: 0.0106
- mse_cv: 0.0209
- mse_range: 0.0081
- precision_cv: 0.0419
- precision_range: 0.0958
- rmse_cv: 0.0105
- rmse_range: 0.0110

**Improvement:**
- f1_score_abs_improvement: -0.1081
- f1_score_rel_improvement: -16.3076
- accuracy_abs_improvement: -0.0286
- accuracy_rel_improvement: -3.4093
- r2_abs_improvement: -0.1176
- r2_rel_improvement: -39.3028
- recall_abs_improvement: -0.0751
- recall_rel_improvement: -11.8656
- mae_abs_improvement: +0.0439
- mae_rel_improvement: +17.4704
- mse_abs_improvement: +0.0193
- mse_rel_improvement: +16.7490
- precision_abs_improvement: -0.0558
- precision_rel_improvement: -6.7184
- rmse_abs_improvement: +0.0273
- rmse_rel_improvement: +8.0577

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 81.0170
- **Sharpe Ratio:** 81.0170
- **Sortino Ratio:** 81.0170

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- f1_score_mean: 0.7220
- f1_score_std: 0.0080
- accuracy_mean: 0.8561
- accuracy_std: 0.0052
- r2_mean: 0.3650
- r2_std: 0.0149
- recall_mean: 0.6842
- recall_std: 0.0077
- mae_mean: 0.2351
- mae_std: 0.0049
- mse_mean: 0.1042
- mse_std: 0.0039
- precision_mean: 0.8400
- precision_std: 0.0066
- rmse_mean: 0.3227
- rmse_std: 0.0060

**Fold Stability (Pre-HPO):**
- f1_score_cv: 0.0110
- f1_score_range: 0.0219
- accuracy_cv: 0.0061
- accuracy_range: 0.0149
- r2_cv: 0.0408
- r2_range: 0.0457
- recall_cv: 0.0113
- recall_range: 0.0215
- mae_cv: 0.0208
- mae_range: 0.0129
- mse_cv: 0.0370
- mse_range: 0.0104
- precision_cv: 0.0079
- precision_range: 0.0147
- rmse_cv: 0.0185
- rmse_range: 0.0160

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- f1_score_mean: 0.4615
- f1_score_std: 0.0047
- accuracy_mean: 0.7963
- accuracy_std: 0.0079
- r2_mean: 0.0800
- r2_std: 0.0060
- recall_mean: 0.5090
- recall_std: 0.0020
- mae_mean: 0.3113
- mae_std: 0.0041
- mse_mean: 0.1509
- mse_std: 0.0043
- precision_mean: 0.8182
- precision_std: 0.0267
- rmse_mean: 0.3885
- rmse_std: 0.0055

**Fold Stability (Post-HPO):**
- f1_score_cv: 0.0102
- f1_score_range: 0.0125
- accuracy_cv: 0.0099
- accuracy_range: 0.0238
- r2_cv: 0.0754
- r2_range: 0.0187
- recall_cv: 0.0040
- recall_range: 0.0063
- mae_cv: 0.0132
- mae_range: 0.0126
- mse_cv: 0.0285
- mse_range: 0.0133
- precision_cv: 0.0326
- precision_range: 0.0790
- rmse_cv: 0.0142
- rmse_range: 0.0171

**Improvement:**
- f1_score_abs_improvement: -0.2605
- f1_score_rel_improvement: -36.0743
- accuracy_abs_improvement: -0.0598
- accuracy_rel_improvement: -6.9887
- r2_abs_improvement: -0.2850
- r2_rel_improvement: -78.0889
- recall_abs_improvement: -0.1752
- recall_rel_improvement: -25.6065
- mae_abs_improvement: +0.0762
- mae_rel_improvement: +32.4219
- mse_abs_improvement: +0.0467
- mse_rel_improvement: +44.8721
- precision_abs_improvement: -0.0218
- precision_rel_improvement: -2.5948
- rmse_abs_improvement: +0.0657
- rmse_rel_improvement: +20.3713

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 79.6264
- **Sharpe Ratio:** 79.6264
- **Sortino Ratio:** 79.6264

**Top 10 Important Features:**
- trend_score_14: 22.2666
- directional_signal: 5.2762
- vectorbt_sma_5: 4.2426
- resistance_level_1_20_price_returns: 3.2474
- enhanced_volatility_50: 2.8349
- enhanced_volatility_20: 2.4895
- lightgbm_regime_3_prob: 2.3072
- vectorbt_atr_30: 2.2585
- vectorbt_enhanced_ad_line_50: 2.1987
- support_level_1_5_price_returns: 2.0178

---
