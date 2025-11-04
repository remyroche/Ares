# Training Report: ensemble_base

**Session ID:** ensemble_base_ETHUSDT_20251103_235751
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-03T23:57:51.783770
**Total Training Time:** 43.00s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 1,000
- **Features:** 8

---

## Best Model
**Name:** ensemble_lightgbm

**Metrics:**
- mse_mean: 0.1319
- mse_std: 0.0105
- f1_score_mean: 0.4493
- f1_score_std: 0.0026
- precision_mean: 0.4080
- precision_std: 0.0043
- mae_mean: 0.2765
- mae_std: 0.0144
- r2_mean: 0.1222
- r2_std: 0.0452
- rmse_mean: 0.3629
- rmse_std: 0.0145
- recall_mean: 0.5000
- recall_std: 0.0000
- accuracy_mean: 0.8160
- accuracy_std: 0.0086

---

## Model Training Details

### ensemble_lightgbm (lightgbm)

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
- **Time:** 40.89s
- **Best Parameters:** {'num_leaves': 84, 'learning_rate': 0.011433250288000681, 'feature_fraction': 0.8633337279889601, 'bagging_fraction': 0.6212119041143672, 'min_child_samples': 40}

#### Post-HPO Metrics
- mse_mean: 0.1319
- mse_std: 0.0105
- f1_score_mean: 0.4493
- f1_score_std: 0.0026
- precision_mean: 0.4080
- precision_std: 0.0043
- mae_mean: 0.2765
- mae_std: 0.0144
- r2_mean: 0.1222
- r2_std: 0.0452
- rmse_mean: 0.3629
- rmse_std: 0.0145
- recall_mean: 0.5000
- recall_std: 0.0000
- accuracy_mean: 0.8160
- accuracy_std: 0.0086

**Fold Stability (Post-HPO):**
- mse_cv: 0.0798
- mse_range: 0.0295
- f1_score_cv: 0.0058
- f1_score_range: 0.0076
- precision_cv: 0.0105
- precision_range: 0.0125
- mae_cv: 0.0519
- mae_range: 0.0366
- r2_cv: 0.3694
- r2_range: 0.1223
- rmse_cv: 0.0399
- rmse_range: 0.0406
- recall_cv: 0.0000
- recall_range: 0.0000
- accuracy_cv: 0.0105
- accuracy_range: 0.0250

**Improvement:**
- mse_abs_improvement: -0.0291
- mse_rel_improvement: -18.0963
- f1_score_abs_improvement: -0.1092
- f1_score_rel_improvement: -19.5521
- precision_abs_improvement: -0.1978
- precision_rel_improvement: -32.6530
- mae_abs_improvement: -0.0175
- mae_rel_improvement: -5.9584
- r2_abs_improvement: +0.1945
- r2_rel_improvement: -269.1549
- rmse_abs_improvement: -0.0381
- rmse_rel_improvement: -9.4957
- recall_abs_improvement: -0.0537
- recall_rel_improvement: -9.6958
- accuracy_abs_improvement: +0.0250
- accuracy_rel_improvement: +3.1606

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 94.8581
- **Sharpe Ratio:** 94.8581
- **Sortino Ratio:** inf

---
