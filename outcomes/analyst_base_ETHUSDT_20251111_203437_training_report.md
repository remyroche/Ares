# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251111_203437
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-11T20:34:37.751619
**Total Training Time:** 1821.72s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 14,023
- **Features:** 71

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**
- mse_mean: 0.0346
- mse_std: 0.0338
- r2_mean: 0.7880
- r2_std: 0.2091
- f1_score_mean: 0.9439
- f1_score_std: 0.0955
- precision_mean: 0.9651
- precision_std: 0.0627
- mae_mean: 0.1157
- mae_std: 0.0527
- accuracy_mean: 0.9691
- accuracy_std: 0.0509
- recall_mean: 0.9327
- recall_std: 0.1086
- rmse_mean: 0.1703
- rmse_std: 0.0746

---

## Model Training Details

### analyst_lightgbm (lightgbm)

#### Pre-HPO Metrics
- mse_mean: 0.1153
- mse_std: 0.0039
- r2_mean: 0.2971
- r2_std: 0.0135
- f1_score_mean: 0.6630
- f1_score_std: 0.0140
- precision_mean: 0.8330
- precision_std: 0.0062
- mae_mean: 0.2520
- mae_std: 0.0044
- accuracy_mean: 0.8391
- accuracy_std: 0.0065
- recall_mean: 0.6331
- recall_std: 0.0108
- rmse_mean: 0.3395
- rmse_std: 0.0057

**Fold Stability (Pre-HPO):**
- mse_cv: 0.0337
- mse_range: 0.0112
- r2_cv: 0.0455
- r2_range: 0.0398
- f1_score_cv: 0.0211
- f1_score_range: 0.0409
- precision_cv: 0.0074
- precision_range: 0.0163
- mae_cv: 0.0175
- mae_range: 0.0112
- accuracy_cv: 0.0078
- accuracy_range: 0.0181
- recall_cv: 0.0171
- recall_range: 0.0325
- rmse_cv: 0.0168
- rmse_range: 0.0164

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 199.94s
- **Best Parameters:** {'num_leaves': 97, 'learning_rate': 0.09713207052906694, 'feature_fraction': 0.8165285012348362, 'bagging_fraction': 0.8553155015664131, 'min_child_samples': 6}

#### Post-HPO Metrics
- mse_mean: 0.0346
- mse_std: 0.0338
- r2_mean: 0.7880
- r2_std: 0.2091
- f1_score_mean: 0.9439
- f1_score_std: 0.0955
- precision_mean: 0.9651
- precision_std: 0.0627
- mae_mean: 0.1157
- mae_std: 0.0527
- accuracy_mean: 0.9691
- accuracy_std: 0.0509
- recall_mean: 0.9327
- recall_std: 0.1086
- rmse_mean: 0.1703
- rmse_std: 0.0746

**Fold Stability (Post-HPO):**
- mse_cv: 0.9766
- mse_range: 0.0852
- r2_cv: 0.2654
- r2_range: 0.5274
- f1_score_cv: 0.1012
- f1_score_range: 0.2403
- precision_cv: 0.0649
- precision_range: 0.1573
- mae_cv: 0.4556
- mae_range: 0.1338
- accuracy_cv: 0.0525
- accuracy_range: 0.1280
- recall_cv: 0.1165
- recall_range: 0.2740
- rmse_cv: 0.4383
- rmse_range: 0.1895

**Improvement:**
- mse_abs_improvement: -0.0808
- mse_rel_improvement: -70.0295
- r2_abs_improvement: +0.4909
- r2_rel_improvement: +165.2320
- f1_score_abs_improvement: +0.2809
- f1_score_rel_improvement: +42.3699
- precision_abs_improvement: +0.1321
- precision_rel_improvement: +15.8541
- mae_abs_improvement: -0.1363
- mae_rel_improvement: -54.0859
- accuracy_abs_improvement: +0.1301
- accuracy_rel_improvement: +15.5026
- recall_abs_improvement: +0.2997
- recall_rel_improvement: +47.3328
- rmse_abs_improvement: -0.1693
- rmse_rel_improvement: -49.8526

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 19.0485
- **Sharpe Ratio:** 19.0485
- **Sortino Ratio:** 96.9126

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- mse_mean: 0.1043
- mse_std: 0.0039
- r2_mean: 0.3640
- r2_std: 0.0146
- f1_score_mean: 0.7238
- f1_score_std: 0.0140
- precision_mean: 0.8459
- precision_std: 0.0124
- mae_mean: 0.2356
- mae_std: 0.0052
- accuracy_mean: 0.8576
- accuracy_std: 0.0076
- recall_mean: 0.6852
- recall_std: 0.0120
- rmse_mean: 0.3230
- rmse_std: 0.0060

**Fold Stability (Pre-HPO):**
- mse_cv: 0.0373
- mse_range: 0.0108
- r2_cv: 0.0400
- r2_range: 0.0442
- f1_score_cv: 0.0193
- f1_score_range: 0.0376
- precision_cv: 0.0147
- precision_range: 0.0339
- mae_cv: 0.0219
- mae_range: 0.0131
- accuracy_cv: 0.0088
- accuracy_range: 0.0217
- recall_cv: 0.0175
- recall_range: 0.0327
- rmse_cv: 0.0186
- rmse_range: 0.0167

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 1404.11s
- **Best Parameters:** {'depth': 9, 'learning_rate': 0.07946113033845845, 'l2_leaf_reg': 1.342381455150235, 'border_count': 251}

#### Post-HPO Metrics
- mse_mean: 0.0987
- mse_std: 0.0043
- r2_mean: 0.3986
- r2_std: 0.0179
- f1_score_mean: 0.7590
- f1_score_std: 0.0153
- precision_mean: 0.8391
- precision_std: 0.0125
- mae_mean: 0.2215
- mae_std: 0.0049
- accuracy_mean: 0.8673
- accuracy_std: 0.0086
- recall_mean: 0.7231
- recall_std: 0.0147
- rmse_mean: 0.3140
- rmse_std: 0.0068

**Fold Stability (Post-HPO):**
- mse_cv: 0.0432
- mse_range: 0.0115
- r2_cv: 0.0449
- r2_range: 0.0509
- f1_score_cv: 0.0201
- f1_score_range: 0.0426
- precision_cv: 0.0149
- precision_range: 0.0369
- mae_cv: 0.0220
- mae_range: 0.0118
- accuracy_cv: 0.0099
- accuracy_range: 0.0213
- recall_cv: 0.0204
- recall_range: 0.0421
- rmse_cv: 0.0216
- rmse_range: 0.0183

**Improvement:**
- mse_abs_improvement: -0.0057
- mse_rel_improvement: -5.4426
- r2_abs_improvement: +0.0347
- r2_rel_improvement: +9.5219
- f1_score_abs_improvement: +0.0352
- f1_score_rel_improvement: +4.8669
- precision_abs_improvement: -0.0069
- precision_rel_improvement: -0.8117
- mae_abs_improvement: -0.0141
- mae_rel_improvement: -5.9796
- accuracy_abs_improvement: +0.0097
- accuracy_rel_improvement: +1.1309
- recall_abs_improvement: +0.0378
- recall_rel_improvement: +5.5199
- rmse_abs_improvement: -0.0089
- rmse_rel_improvement: -2.7652

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 86.7291
- **Sharpe Ratio:** 86.7291
- **Sortino Ratio:** 86.7291

**Top 10 Important Features:**
- trend_score_14: 4.9998
- enhanced_volatility_50: 2.4512
- volume_volatility_elasticity_20: 2.3870
- support_level_1_5_price_returns: 2.2884
- volume_std_10: 2.1885
- vectorbt_acceleration_momentum_5_10_price_returns: 2.1666
- enhanced_volatility_100: 2.0886
- directional_signal: 2.0858
- vectorbt_momentum_acceleration_10_10_price_returns: 1.9341
- stochastic_30_3_price_returns: 1.9096

---
