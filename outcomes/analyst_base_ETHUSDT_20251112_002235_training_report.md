# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251112_002235
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-12T00:22:35.492285
**Total Training Time:** 46.45s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 14,023
- **Features:** 71

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**
- mse_mean: 0.1610
- mse_std: 0.0043
- r2_mean: 0.0188
- r2_std: 0.0028
- rmse_mean: 0.4012
- rmse_std: 0.0053
- f1_score_mean: 0.4423
- f1_score_std: 0.0025
- precision_mean: 0.3965
- precision_std: 0.0040
- mae_mean: 0.3295
- mae_std: 0.0044
- recall_mean: 0.5000
- recall_std: 0.0000
- accuracy_mean: 0.7931
- accuracy_std: 0.0080

---

## Model Training Details

### analyst_lightgbm (lightgbm)

#### Pre-HPO Metrics
- mse_mean: 0.1150
- mse_std: 0.0036
- r2_mean: 0.2992
- r2_std: 0.0109
- rmse_mean: 0.3390
- rmse_std: 0.0053
- f1_score_mean: 0.6630
- f1_score_std: 0.0121
- precision_mean: 0.8312
- precision_std: 0.0071
- mae_mean: 0.2516
- mae_std: 0.0045
- recall_mean: 0.6331
- recall_std: 0.0094
- accuracy_mean: 0.8388
- accuracy_std: 0.0067

**Fold Stability (Pre-HPO):**
- mse_cv: 0.0313
- mse_range: 0.0106
- r2_cv: 0.0363
- r2_range: 0.0333
- rmse_cv: 0.0156
- rmse_range: 0.0157
- f1_score_cv: 0.0183
- f1_score_range: 0.0317
- precision_cv: 0.0086
- precision_range: 0.0164
- mae_cv: 0.0179
- mae_range: 0.0124
- recall_cv: 0.0149
- recall_range: 0.0257
- accuracy_cv: 0.0080
- accuracy_range: 0.0196

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- mse_mean: 0.1610
- mse_std: 0.0043
- r2_mean: 0.0188
- r2_std: 0.0028
- rmse_mean: 0.4012
- rmse_std: 0.0053
- f1_score_mean: 0.4423
- f1_score_std: 0.0025
- precision_mean: 0.3965
- precision_std: 0.0040
- mae_mean: 0.3295
- mae_std: 0.0044
- recall_mean: 0.5000
- recall_std: 0.0000
- accuracy_mean: 0.7931
- accuracy_std: 0.0080

**Fold Stability (Post-HPO):**
- mse_cv: 0.0264
- mse_range: 0.0131
- r2_cv: 0.1482
- r2_range: 0.0069
- rmse_cv: 0.0132
- rmse_range: 0.0163
- f1_score_cv: 0.0057
- f1_score_range: 0.0076
- precision_cv: 0.0101
- precision_range: 0.0123
- mae_cv: 0.0133
- mae_range: 0.0135
- recall_cv: 0.0000
- recall_range: 0.0000
- accuracy_cv: 0.0101
- accuracy_range: 0.0245

**Improvement:**
- mse_abs_improvement: +0.0460
- mse_rel_improvement: +40.0115
- r2_abs_improvement: -0.2804
- r2_rel_improvement: -93.7146
- rmse_abs_improvement: +0.0621
- rmse_rel_improvement: +18.3307
- f1_score_abs_improvement: -0.2207
- f1_score_rel_improvement: -33.2904
- precision_abs_improvement: -0.4346
- precision_rel_improvement: -52.2928
- mae_abs_improvement: +0.0779
- mae_rel_improvement: +30.9805
- recall_abs_improvement: -0.1331
- recall_rel_improvement: -21.0258
- accuracy_abs_improvement: -0.0457
- accuracy_rel_improvement: -5.4498

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 79.3055
- **Sharpe Ratio:** 79.3055
- **Sortino Ratio:** 79.3055

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- mse_mean: 0.1042
- mse_std: 0.0039
- r2_mean: 0.3650
- r2_std: 0.0149
- rmse_mean: 0.3227
- rmse_std: 0.0060
- f1_score_mean: 0.7220
- f1_score_std: 0.0080
- precision_mean: 0.8400
- precision_std: 0.0066
- mae_mean: 0.2351
- mae_std: 0.0049
- recall_mean: 0.6842
- recall_std: 0.0077
- accuracy_mean: 0.8561
- accuracy_std: 0.0052

**Fold Stability (Pre-HPO):**
- mse_cv: 0.0370
- mse_range: 0.0104
- r2_cv: 0.0408
- r2_range: 0.0457
- rmse_cv: 0.0185
- rmse_range: 0.0160
- f1_score_cv: 0.0110
- f1_score_range: 0.0219
- precision_cv: 0.0079
- precision_range: 0.0147
- mae_cv: 0.0208
- mae_range: 0.0129
- recall_cv: 0.0113
- recall_range: 0.0215
- accuracy_cv: 0.0061
- accuracy_range: 0.0149

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- mse_mean: 0.1509
- mse_std: 0.0043
- r2_mean: 0.0800
- r2_std: 0.0060
- rmse_mean: 0.3885
- rmse_std: 0.0055
- f1_score_mean: 0.4615
- f1_score_std: 0.0047
- precision_mean: 0.8182
- precision_std: 0.0267
- mae_mean: 0.3113
- mae_std: 0.0041
- recall_mean: 0.5090
- recall_std: 0.0020
- accuracy_mean: 0.7963
- accuracy_std: 0.0079

**Fold Stability (Post-HPO):**
- mse_cv: 0.0285
- mse_range: 0.0133
- r2_cv: 0.0754
- r2_range: 0.0187
- rmse_cv: 0.0142
- rmse_range: 0.0171
- f1_score_cv: 0.0102
- f1_score_range: 0.0125
- precision_cv: 0.0326
- precision_range: 0.0790
- mae_cv: 0.0132
- mae_range: 0.0126
- recall_cv: 0.0040
- recall_range: 0.0063
- accuracy_cv: 0.0099
- accuracy_range: 0.0238

**Improvement:**
- mse_abs_improvement: +0.0467
- mse_rel_improvement: +44.8721
- r2_abs_improvement: -0.2850
- r2_rel_improvement: -78.0889
- rmse_abs_improvement: +0.0657
- rmse_rel_improvement: +20.3713
- f1_score_abs_improvement: -0.2605
- f1_score_rel_improvement: -36.0743
- precision_abs_improvement: -0.0218
- precision_rel_improvement: -2.5948
- mae_abs_improvement: +0.0762
- mae_rel_improvement: +32.4219
- recall_abs_improvement: -0.1752
- recall_rel_improvement: -25.6065
- accuracy_abs_improvement: -0.0598
- accuracy_rel_improvement: -6.9887

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
