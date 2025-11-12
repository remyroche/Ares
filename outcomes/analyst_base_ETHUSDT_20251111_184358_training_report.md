# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251111_184358
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-11T18:43:58.763538
**Total Training Time:** 4805.59s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 14,023
- **Features:** 71

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**
- accuracy_mean: 0.9676
- accuracy_std: 0.0522
- precision_mean: 0.9640
- precision_std: 0.0638
- recall_mean: 0.9286
- recall_std: 0.1128
- rmse_mean: 0.1762
- rmse_std: 0.0716
- mse_mean: 0.0362
- mse_std: 0.0329
- f1_score_mean: 0.9404
- f1_score_std: 0.0998
- r2_mean: 0.7782
- r2_std: 0.2040
- mae_mean: 0.1214
- mae_std: 0.0508

---

## Model Training Details

### analyst_lightgbm (lightgbm)

#### Pre-HPO Metrics
- accuracy_mean: 0.8383
- accuracy_std: 0.0070
- precision_mean: 0.8301
- precision_std: 0.0129
- recall_mean: 0.6315
- recall_std: 0.0139
- rmse_mean: 0.3394
- rmse_std: 0.0060
- mse_mean: 0.1152
- mse_std: 0.0041
- f1_score_mean: 0.6608
- f1_score_std: 0.0178
- r2_mean: 0.2978
- r2_std: 0.0152
- mae_mean: 0.2517
- mae_std: 0.0050

**Fold Stability (Pre-HPO):**
- accuracy_cv: 0.0084
- accuracy_range: 0.0182
- precision_cv: 0.0155
- precision_range: 0.0347
- recall_cv: 0.0220
- recall_range: 0.0390
- rmse_cv: 0.0177
- rmse_range: 0.0163
- mse_cv: 0.0356
- mse_range: 0.0111
- f1_score_cv: 0.0269
- f1_score_range: 0.0492
- r2_cv: 0.0510
- r2_range: 0.0408
- mae_cv: 0.0199
- mae_range: 0.0127

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 856.22s
- **Best Parameters:** {'num_leaves': 100, 'learning_rate': 0.08351232024786236, 'feature_fraction': 0.7942097465889634, 'bagging_fraction': 0.7432650731591364, 'min_child_samples': 5}

#### Post-HPO Metrics
- accuracy_mean: 0.9676
- accuracy_std: 0.0522
- precision_mean: 0.9640
- precision_std: 0.0638
- recall_mean: 0.9286
- recall_std: 0.1128
- rmse_mean: 0.1762
- rmse_std: 0.0716
- mse_mean: 0.0362
- mse_std: 0.0329
- f1_score_mean: 0.9404
- f1_score_std: 0.0998
- r2_mean: 0.7782
- r2_std: 0.2040
- mae_mean: 0.1214
- mae_std: 0.0508

**Fold Stability (Post-HPO):**
- accuracy_cv: 0.0540
- accuracy_range: 0.1315
- precision_cv: 0.0662
- precision_range: 0.1604
- recall_cv: 0.1215
- recall_range: 0.2840
- rmse_cv: 0.4066
- rmse_range: 0.1825
- mse_cv: 0.9099
- mse_range: 0.0833
- f1_score_cv: 0.1061
- f1_score_range: 0.2510
- r2_cv: 0.2622
- r2_range: 0.5157
- mae_cv: 0.4185
- mae_range: 0.1292

**Improvement:**
- accuracy_abs_improvement: +0.1293
- accuracy_rel_improvement: +15.4236
- precision_abs_improvement: +0.1338
- precision_rel_improvement: +16.1228
- recall_abs_improvement: +0.2971
- recall_rel_improvement: +47.0506
- rmse_abs_improvement: -0.1632
- rmse_rel_improvement: -48.0857
- mse_abs_improvement: -0.0790
- mse_rel_improvement: -68.6038
- f1_score_abs_improvement: +0.2796
- f1_score_rel_improvement: +42.3112
- r2_abs_improvement: +0.4804
- r2_rel_improvement: +161.3423
- mae_abs_improvement: -0.1303
- mae_rel_improvement: -51.7519

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 18.5246
- **Sharpe Ratio:** 18.5246
- **Sortino Ratio:** 96.7557

---

### analyst_depthwise_cnn (depthwise_cnn)

#### Pre-HPO Metrics

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 0.00s

#### Post-HPO Metrics
- accuracy_mean: 0.7931
- accuracy_std: 0.0080
- precision_mean: 0.3965
- precision_std: 0.0040
- recall_mean: 0.5000
- recall_std: 0.0000
- rmse_mean: 0.4052
- rmse_std: 0.0063
- mse_mean: 0.1642
- mse_std: 0.0051
- f1_score_mean: 0.4423
- f1_score_std: 0.0025
- r2_mean: -0.0009
- r2_std: 0.0035
- mae_mean: 0.3226
- mae_std: 0.0022

**Fold Stability (Post-HPO):**
- accuracy_cv: 0.0101
- accuracy_range: 0.0245
- precision_cv: 0.0101
- precision_range: 0.0123
- recall_cv: 0.0000
- recall_range: 0.0000
- rmse_cv: 0.0154
- rmse_range: 0.0189
- mse_cv: 0.0311
- mse_range: 0.0154
- f1_score_cv: 0.0057
- f1_score_range: 0.0076
- r2_cv: -3.9914
- r2_range: 0.0097
- mae_cv: 0.0068
- mae_range: 0.0067

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 79.3055
- **Sharpe Ratio:** 79.3055
- **Sortino Ratio:** 79.3055

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- accuracy_mean: 0.8590
- accuracy_std: 0.0074
- precision_mean: 0.8450
- precision_std: 0.0075
- recall_mean: 0.6907
- recall_std: 0.0101
- rmse_mean: 0.3215
- rmse_std: 0.0055
- mse_mean: 0.1034
- mse_std: 0.0035
- f1_score_mean: 0.7293
- f1_score_std: 0.0114
- r2_mean: 0.3697
- r2_std: 0.0134
- mae_mean: 0.2343
- mae_std: 0.0041

**Fold Stability (Pre-HPO):**
- accuracy_cv: 0.0087
- accuracy_range: 0.0206
- precision_cv: 0.0089
- precision_range: 0.0194
- recall_cv: 0.0146
- recall_range: 0.0236
- rmse_cv: 0.0170
- rmse_range: 0.0155
- mse_cv: 0.0341
- mse_range: 0.0100
- f1_score_cv: 0.0157
- f1_score_range: 0.0283
- r2_cv: 0.0363
- r2_range: 0.0365
- mae_cv: 0.0173
- mae_range: 0.0100

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 2879.34s
- **Best Parameters:** {'depth': 8, 'learning_rate': 0.08219772826786358, 'l2_leaf_reg': 1.6709557931179373, 'border_count': 102}

#### Post-HPO Metrics
- accuracy_mean: 0.8634
- accuracy_std: 0.0061
- precision_mean: 0.8387
- precision_std: 0.0135
- recall_mean: 0.7094
- recall_std: 0.0132
- rmse_mean: 0.3168
- rmse_std: 0.0063
- mse_mean: 0.1004
- mse_std: 0.0040
- f1_score_mean: 0.7466
- f1_score_std: 0.0142
- r2_mean: 0.3879
- r2_std: 0.0197
- mae_mean: 0.2266
- mae_std: 0.0050

**Fold Stability (Post-HPO):**
- accuracy_cv: 0.0070
- accuracy_range: 0.0156
- precision_cv: 0.0160
- precision_range: 0.0348
- recall_cv: 0.0186
- recall_range: 0.0354
- rmse_cv: 0.0200
- rmse_range: 0.0147
- mse_cv: 0.0401
- mse_range: 0.0094
- f1_score_cv: 0.0190
- f1_score_range: 0.0392
- r2_cv: 0.0509
- r2_range: 0.0571
- mae_cv: 0.0221
- mae_range: 0.0108

**Improvement:**
- accuracy_abs_improvement: +0.0044
- accuracy_rel_improvement: +0.5147
- precision_abs_improvement: -0.0063
- precision_rel_improvement: -0.7403
- recall_abs_improvement: +0.0187
- recall_rel_improvement: +2.7024
- rmse_abs_improvement: -0.0047
- rmse_rel_improvement: -1.4628
- mse_abs_improvement: -0.0030
- mse_rel_improvement: -2.8935
- f1_score_abs_improvement: +0.0173
- f1_score_rel_improvement: +2.3715
- r2_abs_improvement: +0.0182
- r2_rel_improvement: +4.9219
- mae_abs_improvement: -0.0076
- mae_rel_improvement: -3.2654

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 86.3439
- **Sharpe Ratio:** 86.3439
- **Sortino Ratio:** 86.3439

**Top 10 Important Features:**
- trend_score_14: 5.6965
- enhanced_volatility_100: 2.4410
- enhanced_volatility_50: 2.3785
- resistance_level_1_20_price_returns: 2.2302
- volume_std_10: 2.1759
- volume_std_50: 2.1333
- volume_trend_strength_20_50: 2.0385
- support_level_1_5_price_returns: 2.0340
- vectorbt_acceleration_momentum_10_10_price_returns: 1.9542
- directional_signal: 1.8390

---
