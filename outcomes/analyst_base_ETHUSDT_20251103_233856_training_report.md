# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251103_233856
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-03T23:38:56.471513
**Total Training Time:** 258.66s

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
- f1_score_mean: 0.5585
- f1_score_std: 0.0306
- r2_mean: -0.0723
- r2_std: 0.0677
- mse_mean: 0.1610
- mse_std: 0.0135
- rmse_mean: 0.4009
- rmse_std: 0.0164
- accuracy_mean: 0.7910
- accuracy_std: 0.0258
- precision_mean: 0.6058
- precision_std: 0.0568
- recall_mean: 0.5537
- recall_std: 0.0234
- mae_mean: 0.2940
- mae_std: 0.0150

**Fold Stability (Pre-HPO):**
- f1_score_cv: 0.0547
- f1_score_range: 0.0835
- r2_cv: -0.9367
- r2_range: 0.1971
- mse_cv: 0.0835
- mse_range: 0.0371
- rmse_cv: 0.0409
- rmse_range: 0.0453
- accuracy_cv: 0.0326
- accuracy_range: 0.0800
- precision_cv: 0.0937
- precision_range: 0.1555
- recall_cv: 0.0422
- recall_range: 0.0640
- mae_cv: 0.0509
- mae_range: 0.0397

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 79.83s
- **Best Parameters:** {'num_leaves': 22, 'learning_rate': 0.016924462453161013, 'feature_fraction': 0.6056366766136184, 'bagging_fraction': 0.7492819119098154, 'min_child_samples': 45}

#### Post-HPO Metrics

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 30.6968
- **Sharpe Ratio:** 30.6968
- **Sortino Ratio:** inf

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- f1_score_mean: 0.5366
- f1_score_std: 0.0329
- r2_mean: -0.0279
- r2_std: 0.0726
- mse_mean: 0.1543
- mse_std: 0.0133
- rmse_mean: 0.3925
- rmse_std: 0.0165
- accuracy_mean: 0.7930
- accuracy_std: 0.0250
- precision_mean: 0.5919
- precision_std: 0.0682
- recall_mean: 0.5378
- recall_std: 0.0238
- mae_mean: 0.2817
- mae_std: 0.0159

**Fold Stability (Pre-HPO):**
- f1_score_cv: 0.0614
- f1_score_range: 0.1011
- r2_cv: -2.6039
- r2_range: 0.2028
- mse_cv: 0.0863
- mse_range: 0.0377
- rmse_cv: 0.0420
- rmse_range: 0.0470
- accuracy_cv: 0.0316
- accuracy_range: 0.0750
- precision_cv: 0.1152
- precision_range: 0.1845
- recall_cv: 0.0443
- recall_range: 0.0709
- mae_cv: 0.0565
- mae_range: 0.0438

#### Hyperparameter Optimization
- **Trials:** 50
- **Time:** 170.53s
- **Best Parameters:** {'depth': 6, 'learning_rate': 0.012852938621133736, 'l2_leaf_reg': 6.667222834422276, 'border_count': 69}

#### Post-HPO Metrics
- f1_score_mean: 0.4487
- f1_score_std: 0.0032
- r2_mean: 0.0457
- r2_std: 0.0414
- mse_mean: 0.1433
- mse_std: 0.0098
- rmse_mean: 0.3783
- rmse_std: 0.0128
- accuracy_mean: 0.8140
- accuracy_std: 0.0107
- precision_mean: 0.4078
- precision_std: 0.0044
- recall_mean: 0.4988
- recall_std: 0.0025
- mae_mean: 0.2863
- mae_std: 0.0147

**Fold Stability (Post-HPO):**
- f1_score_cv: 0.0072
- f1_score_range: 0.0091
- r2_cv: 0.9064
- r2_range: 0.1205
- mse_cv: 0.0687
- mse_range: 0.0284
- rmse_cv: 0.0338
- rmse_range: 0.0369
- accuracy_cv: 0.0131
- accuracy_range: 0.0300
- precision_cv: 0.0109
- precision_range: 0.0125
- recall_cv: 0.0050
- recall_range: 0.0062
- mae_cv: 0.0515
- mae_range: 0.0407

**Improvement:**
- f1_score_abs_improvement: -0.0879
- f1_score_rel_improvement: -16.3765
- r2_abs_improvement: +0.0736
- r2_rel_improvement: -263.9979
- mse_abs_improvement: -0.0110
- mse_rel_improvement: -7.1337
- rmse_abs_improvement: -0.0141
- rmse_rel_improvement: -3.6026
- accuracy_abs_improvement: +0.0210
- accuracy_rel_improvement: +2.6482
- precision_abs_improvement: -0.1841
- precision_rel_improvement: -31.0987
- recall_abs_improvement: -0.0390
- recall_rel_improvement: -7.2498
- mae_abs_improvement: +0.0046
- mae_rel_improvement: +1.6471

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 76.2381
- **Sharpe Ratio:** 76.2381
- **Sortino Ratio:** inf

**Top 10 Important Features:**
- volume: 31.3051
- open: 18.6936
- low: 18.2253
- high: 16.1864
- close: 15.5896
- quality_scores_t_50bps: 0.0000
- base_threshold: 0.0000
- lookahead_periods: 0.0000

---
