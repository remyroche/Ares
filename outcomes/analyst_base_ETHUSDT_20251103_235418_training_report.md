# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251103_235418
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-03T23:54:18.148314
**Total Training Time:** 213.63s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 1,000
- **Features:** 8

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**

---

## Model Training Details

### analyst_lightgbm (lightgbm)

#### Pre-HPO Metrics
- mse_mean: 0.1610
- mse_std: 0.0135
- f1_score_mean: 0.5585
- f1_score_std: 0.0306
- precision_mean: 0.6058
- precision_std: 0.0568
- mae_mean: 0.2940
- mae_std: 0.0150
- r2_mean: -0.0723
- r2_std: 0.0677
- rmse_mean: 0.4009
- rmse_std: 0.0164
- recall_mean: 0.5537
- recall_std: 0.0234
- accuracy_mean: 0.7910
- accuracy_std: 0.0258

**Fold Stability (Pre-HPO):**
- mse_cv: 0.0835
- mse_range: 0.0371
- f1_score_cv: 0.0547
- f1_score_range: 0.0835
- precision_cv: 0.0937
- precision_range: 0.1555
- mae_cv: 0.0509
- mae_range: 0.0397
- r2_cv: -0.9367
- r2_range: 0.1971
- rmse_cv: 0.0409
- rmse_range: 0.0453
- recall_cv: 0.0422
- recall_range: 0.0640
- accuracy_cv: 0.0326
- accuracy_range: 0.0800

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 47.83s
- **Best Parameters:** {'num_leaves': 78, 'learning_rate': 0.019993990982057376, 'feature_fraction': 0.9076872663033737, 'bagging_fraction': 0.651658085262521, 'min_child_samples': 50}

#### Post-HPO Metrics

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 30.6968
- **Sharpe Ratio:** 30.6968
- **Sortino Ratio:** inf

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- mse_mean: 0.1543
- mse_std: 0.0133
- f1_score_mean: 0.5366
- f1_score_std: 0.0329
- precision_mean: 0.5919
- precision_std: 0.0682
- mae_mean: 0.2817
- mae_std: 0.0159
- r2_mean: -0.0279
- r2_std: 0.0726
- rmse_mean: 0.3925
- rmse_std: 0.0165
- recall_mean: 0.5378
- recall_std: 0.0238
- accuracy_mean: 0.7930
- accuracy_std: 0.0250

**Fold Stability (Pre-HPO):**
- mse_cv: 0.0863
- mse_range: 0.0377
- f1_score_cv: 0.0614
- f1_score_range: 0.1011
- precision_cv: 0.1152
- precision_range: 0.1845
- mae_cv: 0.0565
- mae_range: 0.0438
- r2_cv: -2.6039
- r2_range: 0.2028
- rmse_cv: 0.0420
- rmse_range: 0.0470
- recall_cv: 0.0443
- recall_range: 0.0709
- accuracy_cv: 0.0316
- accuracy_range: 0.0750

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 157.63s
- **Best Parameters:** {'depth': 4, 'learning_rate': 0.015709470241058423, 'l2_leaf_reg': 8.760618856671115, 'border_count': 116}

#### Post-HPO Metrics
- mse_mean: 0.1448
- mse_std: 0.0110
- f1_score_mean: 0.4587
- f1_score_std: 0.0181
- precision_mean: 0.4586
- precision_std: 0.0998
- mae_mean: 0.2882
- mae_std: 0.0145
- r2_mean: 0.0357
- r2_std: 0.0505
- rmse_mean: 0.3803
- rmse_std: 0.0142
- recall_mean: 0.5040
- recall_std: 0.0081
- accuracy_mean: 0.8160
- accuracy_std: 0.0086

**Fold Stability (Post-HPO):**
- mse_cv: 0.0759
- mse_range: 0.0313
- f1_score_cv: 0.0394
- f1_score_range: 0.0486
- precision_cv: 0.2177
- precision_range: 0.2557
- mae_cv: 0.0503
- mae_range: 0.0396
- r2_cv: 1.4142
- r2_range: 0.1445
- rmse_cv: 0.0373
- rmse_range: 0.0406
- recall_cv: 0.0160
- recall_range: 0.0201
- accuracy_cv: 0.0105
- accuracy_range: 0.0250

**Improvement:**
- mse_abs_improvement: -0.0095
- mse_rel_improvement: -6.1549
- f1_score_abs_improvement: -0.0779
- f1_score_rel_improvement: -14.5091
- precision_abs_improvement: -0.1332
- precision_rel_improvement: -22.5116
- mae_abs_improvement: +0.0066
- mae_rel_improvement: +2.3339
- r2_abs_improvement: +0.0636
- r2_rel_improvement: -228.1621
- rmse_abs_improvement: -0.0122
- rmse_rel_improvement: -3.1080
- recall_abs_improvement: -0.0337
- recall_rel_improvement: -6.2711
- accuracy_abs_improvement: +0.0230
- accuracy_rel_improvement: +2.9004

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 94.8581
- **Sharpe Ratio:** 94.8581
- **Sortino Ratio:** inf

**Top 10 Important Features:**
- volume: 26.7132
- close: 21.6960
- open: 18.8194
- low: 17.0098
- high: 15.7617
- quality_scores_t_50bps: 0.0000
- base_threshold: 0.0000
- lookahead_periods: 0.0000

---
