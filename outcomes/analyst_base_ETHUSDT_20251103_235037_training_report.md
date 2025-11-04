# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251103_235037
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-03T23:50:37.254165
**Total Training Time:** 193.97s

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
- mae_mean: 0.2940
- mae_std: 0.0150
- mse_mean: 0.1610
- mse_std: 0.0135
- precision_mean: 0.6058
- precision_std: 0.0568
- recall_mean: 0.5537
- recall_std: 0.0234
- f1_score_mean: 0.5585
- f1_score_std: 0.0306
- accuracy_mean: 0.7910
- accuracy_std: 0.0258
- r2_mean: -0.0723
- r2_std: 0.0677
- rmse_mean: 0.4009
- rmse_std: 0.0164

**Fold Stability (Pre-HPO):**
- mae_cv: 0.0509
- mae_range: 0.0397
- mse_cv: 0.0835
- mse_range: 0.0371
- precision_cv: 0.0937
- precision_range: 0.1555
- recall_cv: 0.0422
- recall_range: 0.0640
- f1_score_cv: 0.0547
- f1_score_range: 0.0835
- accuracy_cv: 0.0326
- accuracy_range: 0.0800
- r2_cv: -0.9367
- r2_range: 0.1971
- rmse_cv: 0.0409
- rmse_range: 0.0453

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 64.00s
- **Best Parameters:** {'num_leaves': 54, 'learning_rate': 0.015071143209552544, 'feature_fraction': 0.676307833521113, 'bagging_fraction': 0.8981989871422092, 'min_child_samples': 40}

#### Post-HPO Metrics

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 30.6968
- **Sharpe Ratio:** 30.6968
- **Sortino Ratio:** inf

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- mae_mean: 0.2817
- mae_std: 0.0159
- mse_mean: 0.1543
- mse_std: 0.0133
- precision_mean: 0.5919
- precision_std: 0.0682
- recall_mean: 0.5378
- recall_std: 0.0238
- f1_score_mean: 0.5366
- f1_score_std: 0.0329
- accuracy_mean: 0.7930
- accuracy_std: 0.0250
- r2_mean: -0.0279
- r2_std: 0.0726
- rmse_mean: 0.3925
- rmse_std: 0.0165

**Fold Stability (Pre-HPO):**
- mae_cv: 0.0565
- mae_range: 0.0438
- mse_cv: 0.0863
- mse_range: 0.0377
- precision_cv: 0.1152
- precision_range: 0.1845
- recall_cv: 0.0443
- recall_range: 0.0709
- f1_score_cv: 0.0614
- f1_score_range: 0.1011
- accuracy_cv: 0.0316
- accuracy_range: 0.0750
- r2_cv: -2.6039
- r2_range: 0.2028
- rmse_cv: 0.0420
- rmse_range: 0.0470

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 122.16s
- **Best Parameters:** {'depth': 6, 'learning_rate': 0.019321035233660962, 'l2_leaf_reg': 8.349411134435398, 'border_count': 53}

#### Post-HPO Metrics
- mae_mean: 0.2854
- mae_std: 0.0170
- mse_mean: 0.1445
- mse_std: 0.0110
- precision_mean: 0.5594
- precision_std: 0.2021
- recall_mean: 0.5090
- recall_std: 0.0125
- f1_score_mean: 0.4694
- f1_score_std: 0.0254
- accuracy_mean: 0.8170
- accuracy_std: 0.0117
- r2_mean: 0.0377
- r2_std: 0.0507
- rmse_mean: 0.3799
- rmse_std: 0.0143

**Fold Stability (Post-HPO):**
- mae_cv: 0.0596
- mae_range: 0.0462
- mse_cv: 0.0764
- mse_range: 0.0303
- precision_cv: 0.3614
- precision_range: 0.5121
- recall_cv: 0.0246
- recall_range: 0.0309
- f1_score_cv: 0.0542
- f1_score_range: 0.0612
- accuracy_cv: 0.0143
- accuracy_range: 0.0300
- r2_cv: 1.3444
- r2_range: 0.1556
- rmse_cv: 0.0375
- rmse_range: 0.0392

**Improvement:**
- mae_abs_improvement: +0.0037
- mae_rel_improvement: +1.3203
- mse_abs_improvement: -0.0098
- mse_rel_improvement: -6.3488
- precision_abs_improvement: -0.0325
- precision_rel_improvement: -5.4926
- recall_abs_improvement: -0.0288
- recall_rel_improvement: -5.3534
- f1_score_abs_improvement: -0.0671
- f1_score_rel_improvement: -12.5119
- accuracy_abs_improvement: +0.0240
- accuracy_rel_improvement: +3.0265
- r2_abs_improvement: +0.0656
- r2_rel_improvement: -235.3611
- rmse_abs_improvement: -0.0126
- rmse_rel_improvement: -3.2091

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 70.0572
- **Sharpe Ratio:** 70.0572
- **Sortino Ratio:** inf

**Top 10 Important Features:**
- volume: 31.3707
- open: 19.0421
- close: 18.6694
- low: 18.6105
- high: 12.3073
- quality_scores_t_50bps: 0.0000
- base_threshold: 0.0000
- lookahead_periods: 0.0000

---
