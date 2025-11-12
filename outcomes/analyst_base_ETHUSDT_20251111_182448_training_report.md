# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251111_182448
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-11T18:24:48.259261
**Total Training Time:** 5489.21s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 14,023
- **Features:** 71

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**
- precision_mean: 0.9549
- precision_std: 0.0703
- f1_score_mean: 0.9286
- f1_score_std: 0.1013
- accuracy_mean: 0.9603
- accuracy_std: 0.0527
- rmse_mean: 0.1986
- rmse_std: 0.0630
- mae_mean: 0.1386
- mae_std: 0.0460
- recall_mean: 0.9138
- recall_std: 0.1113
- mse_mean: 0.0434
- mse_std: 0.0309
- r2_mean: 0.7341
- r2_std: 0.1920

---

## Model Training Details

### analyst_lightgbm (lightgbm)

#### Pre-HPO Metrics
- precision_mean: 0.8301
- precision_std: 0.0129
- f1_score_mean: 0.6608
- f1_score_std: 0.0178
- accuracy_mean: 0.8383
- accuracy_std: 0.0070
- rmse_mean: 0.3394
- rmse_std: 0.0060
- mae_mean: 0.2517
- mae_std: 0.0050
- recall_mean: 0.6315
- recall_std: 0.0139
- mse_mean: 0.1152
- mse_std: 0.0041
- r2_mean: 0.2978
- r2_std: 0.0152

**Fold Stability (Pre-HPO):**
- precision_cv: 0.0155
- precision_range: 0.0347
- f1_score_cv: 0.0269
- f1_score_range: 0.0492
- accuracy_cv: 0.0084
- accuracy_range: 0.0182
- rmse_cv: 0.0177
- rmse_range: 0.0163
- mae_cv: 0.0199
- mae_range: 0.0127
- recall_cv: 0.0220
- recall_range: 0.0390
- mse_cv: 0.0356
- mse_range: 0.0111
- r2_cv: 0.0510
- r2_range: 0.0408

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 542.24s
- **Best Parameters:** {'num_leaves': 92, 'learning_rate': 0.09819970327530758, 'feature_fraction': 0.6095396174433924, 'bagging_fraction': 0.7482233934812235, 'min_child_samples': 41}

#### Post-HPO Metrics
- precision_mean: 0.9549
- precision_std: 0.0703
- f1_score_mean: 0.9286
- f1_score_std: 0.1013
- accuracy_mean: 0.9603
- accuracy_std: 0.0527
- rmse_mean: 0.1986
- rmse_std: 0.0630
- mae_mean: 0.1386
- mae_std: 0.0460
- recall_mean: 0.9138
- recall_std: 0.1113
- mse_mean: 0.0434
- mse_std: 0.0309
- r2_mean: 0.7341
- r2_std: 0.1920

**Fold Stability (Post-HPO):**
- precision_cv: 0.0737
- precision_range: 0.1767
- f1_score_cv: 0.1091
- f1_score_range: 0.2562
- accuracy_cv: 0.0549
- accuracy_range: 0.1340
- rmse_cv: 0.3170
- rmse_range: 0.1614
- mae_cv: 0.3322
- mae_range: 0.1186
- recall_cv: 0.1218
- recall_range: 0.2826
- mse_cv: 0.7120
- mse_range: 0.0786
- r2_cv: 0.2615
- r2_range: 0.4878

**Improvement:**
- precision_abs_improvement: +0.1248
- precision_rel_improvement: +15.0365
- f1_score_abs_improvement: +0.2679
- f1_score_rel_improvement: +40.5357
- accuracy_abs_improvement: +0.1220
- accuracy_rel_improvement: +14.5559
- rmse_abs_improvement: -0.1407
- rmse_rel_improvement: -41.4690
- mae_abs_improvement: -0.1131
- mae_rel_improvement: -44.9428
- recall_abs_improvement: +0.2823
- recall_rel_improvement: +44.7122
- mse_abs_improvement: -0.0718
- mse_rel_improvement: -62.3104
- r2_abs_improvement: +0.4364
- r2_rel_improvement: +146.5404

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 18.2175
- **Sharpe Ratio:** 18.2175
- **Sortino Ratio:** 96.0284

---

### analyst_depthwise_cnn (depthwise_cnn)

#### Pre-HPO Metrics

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 0.00s

#### Post-HPO Metrics
- precision_mean: 0.3965
- precision_std: 0.0040
- f1_score_mean: 0.4423
- f1_score_std: 0.0025
- accuracy_mean: 0.7931
- accuracy_std: 0.0080
- rmse_mean: 0.4048
- rmse_std: 0.0064
- mae_mean: 0.3234
- mae_std: 0.0010
- recall_mean: 0.5000
- recall_std: 0.0000
- mse_mean: 0.1639
- mse_std: 0.0052
- r2_mean: 0.0009
- r2_std: 0.0053

**Fold Stability (Post-HPO):**
- precision_cv: 0.0101
- precision_range: 0.0123
- f1_score_cv: 0.0057
- f1_score_range: 0.0076
- accuracy_cv: 0.0101
- accuracy_range: 0.0245
- rmse_cv: 0.0158
- rmse_range: 0.0191
- mae_cv: 0.0031
- mae_range: 0.0026
- recall_cv: 0.0000
- recall_range: 0.0000
- mse_cv: 0.0318
- mse_range: 0.0156
- r2_cv: 5.9050
- r2_range: 0.0157

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 79.3055
- **Sharpe Ratio:** 79.3055
- **Sortino Ratio:** 79.3055

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- precision_mean: 0.8450
- precision_std: 0.0075
- f1_score_mean: 0.7293
- f1_score_std: 0.0114
- accuracy_mean: 0.8590
- accuracy_std: 0.0074
- rmse_mean: 0.3215
- rmse_std: 0.0055
- mae_mean: 0.2343
- mae_std: 0.0041
- recall_mean: 0.6907
- recall_std: 0.0101
- mse_mean: 0.1034
- mse_std: 0.0035
- r2_mean: 0.3697
- r2_std: 0.0134

**Fold Stability (Pre-HPO):**
- precision_cv: 0.0089
- precision_range: 0.0194
- f1_score_cv: 0.0157
- f1_score_range: 0.0283
- accuracy_cv: 0.0087
- accuracy_range: 0.0206
- rmse_cv: 0.0170
- rmse_range: 0.0155
- mae_cv: 0.0173
- mae_range: 0.0100
- recall_cv: 0.0146
- recall_range: 0.0236
- mse_cv: 0.0341
- mse_range: 0.0100
- r2_cv: 0.0363
- r2_range: 0.0365

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 3171.04s
- **Best Parameters:** {'depth': 8, 'learning_rate': 0.08219772826786358, 'l2_leaf_reg': 1.6709557931179373, 'border_count': 102}

#### Post-HPO Metrics
- precision_mean: 0.8387
- precision_std: 0.0135
- f1_score_mean: 0.7466
- f1_score_std: 0.0142
- accuracy_mean: 0.8634
- accuracy_std: 0.0061
- rmse_mean: 0.3168
- rmse_std: 0.0063
- mae_mean: 0.2266
- mae_std: 0.0050
- recall_mean: 0.7094
- recall_std: 0.0132
- mse_mean: 0.1004
- mse_std: 0.0040
- r2_mean: 0.3879
- r2_std: 0.0197

**Fold Stability (Post-HPO):**
- precision_cv: 0.0160
- precision_range: 0.0348
- f1_score_cv: 0.0190
- f1_score_range: 0.0392
- accuracy_cv: 0.0070
- accuracy_range: 0.0156
- rmse_cv: 0.0200
- rmse_range: 0.0147
- mae_cv: 0.0221
- mae_range: 0.0108
- recall_cv: 0.0186
- recall_range: 0.0354
- mse_cv: 0.0401
- mse_range: 0.0094
- r2_cv: 0.0509
- r2_range: 0.0571

**Improvement:**
- precision_abs_improvement: -0.0063
- precision_rel_improvement: -0.7403
- f1_score_abs_improvement: +0.0173
- f1_score_rel_improvement: +2.3715
- accuracy_abs_improvement: +0.0044
- accuracy_rel_improvement: +0.5147
- rmse_abs_improvement: -0.0047
- rmse_rel_improvement: -1.4628
- mae_abs_improvement: -0.0076
- mae_rel_improvement: -3.2654
- recall_abs_improvement: +0.0187
- recall_rel_improvement: +2.7024
- mse_abs_improvement: -0.0030
- mse_rel_improvement: -2.8935
- r2_abs_improvement: +0.0182
- r2_rel_improvement: +4.9219

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
