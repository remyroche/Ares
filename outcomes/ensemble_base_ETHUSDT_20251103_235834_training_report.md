# Training Report: ensemble_base

**Session ID:** ensemble_base_ETHUSDT_20251103_235834
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-03T23:58:34.787505
**Total Training Time:** 91.38s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 1,000
- **Features:** 8

---

## Best Model
**Name:** ensemble_catboost

**Metrics:**
- mse_mean: 0.1451
- mse_std: 0.0121
- f1_score_mean: 0.4516
- f1_score_std: 0.0122
- precision_mean: 0.4408
- precision_std: 0.0687
- mae_mean: 0.2849
- mae_std: 0.0180
- r2_mean: 0.0341
- r2_std: 0.0598
- rmse_mean: 0.3805
- rmse_std: 0.0155
- recall_mean: 0.4966
- recall_std: 0.0071
- accuracy_mean: 0.8070
- accuracy_std: 0.0147

---

## Model Training Details

### ensemble_catboost (catboost)

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
- **Time:** 87.16s
- **Best Parameters:** {'depth': 7, 'learning_rate': 0.01982544804277281, 'l2_leaf_reg': 8.047013723402433, 'border_count': 39}

#### Post-HPO Metrics
- mse_mean: 0.1451
- mse_std: 0.0121
- f1_score_mean: 0.4516
- f1_score_std: 0.0122
- precision_mean: 0.4408
- precision_std: 0.0687
- mae_mean: 0.2849
- mae_std: 0.0180
- r2_mean: 0.0341
- r2_std: 0.0598
- rmse_mean: 0.3805
- rmse_std: 0.0155
- recall_mean: 0.4966
- recall_std: 0.0071
- accuracy_mean: 0.8070
- accuracy_std: 0.0147

**Fold Stability (Post-HPO):**
- mse_cv: 0.0833
- mse_range: 0.0342
- f1_score_cv: 0.0270
- f1_score_range: 0.0346
- precision_cv: 0.1558
- precision_range: 0.1773
- mae_cv: 0.0633
- mae_range: 0.0508
- r2_cv: 1.7518
- r2_range: 0.1825
- rmse_cv: 0.0408
- rmse_range: 0.0440
- recall_cv: 0.0143
- recall_range: 0.0202
- accuracy_cv: 0.0182
- accuracy_range: 0.0400

**Improvement:**
- mse_abs_improvement: -0.0093
- mse_rel_improvement: -5.9968
- f1_score_abs_improvement: -0.0850
- f1_score_rel_improvement: -15.8326
- precision_abs_improvement: -0.1511
- precision_rel_improvement: -25.5255
- mae_abs_improvement: +0.0032
- mae_rel_improvement: +1.1447
- r2_abs_improvement: +0.0620
- r2_rel_improvement: -222.3289
- rmse_abs_improvement: -0.0119
- rmse_rel_improvement: -3.0396
- recall_abs_improvement: -0.0411
- recall_rel_improvement: -7.6488
- accuracy_abs_improvement: +0.0140
- accuracy_rel_improvement: +1.7654

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 54.9094
- **Sharpe Ratio:** 54.9094
- **Sortino Ratio:** inf

**Top 10 Important Features:**
- volume: 33.5249
- open: 20.4256
- low: 18.4954
- close: 14.9692
- high: 12.5849
- quality_scores_t_50bps: 0.0000
- base_threshold: 0.0000
- lookahead_periods: 0.0000

---
