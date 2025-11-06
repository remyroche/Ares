# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251106_215850
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-06T21:58:50.472683
**Total Training Time:** 158.28s

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
- accuracy_mean: 0.7910
- accuracy_std: 0.0258
- f1_score_mean: 0.5585
- f1_score_std: 0.0306
- mae_mean: 0.2940
- mae_std: 0.0150
- mse_mean: 0.1610
- mse_std: 0.0135
- precision_mean: 0.6058
- precision_std: 0.0568
- rmse_mean: 0.4009
- rmse_std: 0.0164
- r2_mean: -0.0723
- r2_std: 0.0677
- recall_mean: 0.5537
- recall_std: 0.0234

**Fold Stability (Pre-HPO):**
- accuracy_cv: 0.0326
- accuracy_range: 0.0800
- f1_score_cv: 0.0547
- f1_score_range: 0.0835
- mae_cv: 0.0509
- mae_range: 0.0397
- mse_cv: 0.0835
- mse_range: 0.0371
- precision_cv: 0.0937
- precision_range: 0.1555
- rmse_cv: 0.0409
- rmse_range: 0.0453
- r2_cv: -0.9367
- r2_range: 0.1971
- recall_cv: 0.0422
- recall_range: 0.0640

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 60.95s
- **Best Parameters:** {'num_leaves': 83, 'learning_rate': 0.014789614571204736, 'feature_fraction': 0.8401674562853072, 'bagging_fraction': 0.8202792980374377, 'min_child_samples': 42}

#### Post-HPO Metrics

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 30.6968
- **Sharpe Ratio:** 30.6968
- **Sortino Ratio:** inf

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- accuracy_mean: 0.7930
- accuracy_std: 0.0250
- f1_score_mean: 0.5366
- f1_score_std: 0.0329
- mae_mean: 0.2817
- mae_std: 0.0159
- mse_mean: 0.1543
- mse_std: 0.0133
- precision_mean: 0.5919
- precision_std: 0.0682
- rmse_mean: 0.3925
- rmse_std: 0.0165
- r2_mean: -0.0279
- r2_std: 0.0726
- recall_mean: 0.5378
- recall_std: 0.0238

**Fold Stability (Pre-HPO):**
- accuracy_cv: 0.0316
- accuracy_range: 0.0750
- f1_score_cv: 0.0614
- f1_score_range: 0.1011
- mae_cv: 0.0565
- mae_range: 0.0438
- mse_cv: 0.0863
- mse_range: 0.0377
- precision_cv: 0.1152
- precision_range: 0.1845
- rmse_cv: 0.0420
- rmse_range: 0.0470
- r2_cv: -2.6039
- r2_range: 0.2028
- recall_cv: 0.0443
- recall_range: 0.0709

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 90.00s
- **Best Parameters:** {'depth': 6, 'learning_rate': 0.014194971691441677, 'l2_leaf_reg': 8.660731150244658, 'border_count': 103}

#### Post-HPO Metrics
- accuracy_mean: 0.8170
- accuracy_std: 0.0093
- f1_score_mean: 0.4692
- f1_score_std: 0.0167
- mae_mean: 0.2865
- mae_std: 0.0153
- mse_mean: 0.1448
- mse_std: 0.0114
- precision_mean: 0.6094
- precision_std: 0.1868
- rmse_mean: 0.3803
- rmse_std: 0.0147
- r2_mean: 0.0355
- r2_std: 0.0536
- recall_mean: 0.5087
- recall_std: 0.0079

**Fold Stability (Post-HPO):**
- accuracy_cv: 0.0114
- accuracy_range: 0.0250
- f1_score_cv: 0.0355
- f1_score_range: 0.0455
- mae_cv: 0.0532
- mae_range: 0.0407
- mse_cv: 0.0789
- mse_range: 0.0316
- precision_cv: 0.3066
- precision_range: 0.5046
- rmse_cv: 0.0386
- rmse_range: 0.0407
- r2_cv: 1.5089
- r2_range: 0.1568
- recall_cv: 0.0155
- recall_range: 0.0201

**Improvement:**
- accuracy_abs_improvement: +0.0240
- accuracy_rel_improvement: +3.0265
- f1_score_abs_improvement: -0.0674
- f1_score_rel_improvement: -12.5576
- mae_abs_improvement: +0.0049
- mae_rel_improvement: +1.7258
- mse_abs_improvement: -0.0095
- mse_rel_improvement: -6.1320
- precision_abs_improvement: +0.0175
- precision_rel_improvement: +2.9534
- rmse_abs_improvement: -0.0122
- rmse_rel_improvement: -3.1010
- r2_abs_improvement: +0.0634
- r2_rel_improvement: -227.4305
- recall_abs_improvement: -0.0290
- recall_rel_improvement: -5.3932

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 88.0994
- **Sharpe Ratio:** 88.0994
- **Sortino Ratio:** inf

**Top 10 Important Features:**
- volume: 30.6828
- close: 19.6479
- high: 18.3495
- open: 16.2459
- low: 15.0739
- quality_scores_t_50bps: 0.0000
- base_threshold: 0.0000
- lookahead_periods: 0.0000

---
