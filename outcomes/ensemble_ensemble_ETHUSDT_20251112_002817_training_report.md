# Training Report: ensemble_ensemble

**Session ID:** ensemble_ensemble_ETHUSDT_20251112_002817
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-12T00:28:17.066943
**Total Training Time:** 31.96s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 14,023
- **Features:** 71

---

## Best Model
**Name:** ensemble_ensemble_meta

**Metrics:**
- recall_mean: 0.6219
- recall_std: 0.0157
- r2_mean: 0.1816
- r2_std: 0.0382
- mse_mean: 0.1342
- mse_std: 0.0063
- mae_mean: 0.2649
- mae_std: 0.0048
- precision_mean: 0.7217
- precision_std: 0.0232
- accuracy_mean: 0.8142
- accuracy_std: 0.0108
- rmse_mean: 0.3662
- rmse_std: 0.0085
- f1_score_mean: 0.6425
- f1_score_std: 0.0191

---

## Model Training Details

### ensemble_ensemble_meta (lightgbm)

#### Pre-HPO Metrics
- recall_mean: 0.6221
- recall_std: 0.0140
- r2_mean: 0.1795
- r2_std: 0.0397
- mse_mean: 0.1345
- mse_std: 0.0064
- mae_mean: 0.2646
- mae_std: 0.0046
- precision_mean: 0.7227
- precision_std: 0.0250
- accuracy_mean: 0.8144
- accuracy_std: 0.0108
- rmse_mean: 0.3667
- rmse_std: 0.0086
- f1_score_mean: 0.6431
- f1_score_std: 0.0172

**Fold Stability (Pre-HPO):**
- recall_cv: 0.0226
- recall_range: 0.0440
- r2_cv: 0.2212
- r2_range: 0.1108
- mse_cv: 0.0477
- mse_range: 0.0178
- mae_cv: 0.0175
- mae_range: 0.0116
- precision_cv: 0.0345
- precision_range: 0.0693
- accuracy_cv: 0.0132
- accuracy_range: 0.0289
- rmse_cv: 0.0236
- rmse_range: 0.0240
- f1_score_cv: 0.0268
- f1_score_range: 0.0539

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- recall_mean: 0.6219
- recall_std: 0.0157
- r2_mean: 0.1816
- r2_std: 0.0382
- mse_mean: 0.1342
- mse_std: 0.0063
- mae_mean: 0.2649
- mae_std: 0.0048
- precision_mean: 0.7217
- precision_std: 0.0232
- accuracy_mean: 0.8142
- accuracy_std: 0.0108
- rmse_mean: 0.3662
- rmse_std: 0.0085
- f1_score_mean: 0.6425
- f1_score_std: 0.0191

**Fold Stability (Post-HPO):**
- recall_cv: 0.0252
- recall_range: 0.0458
- r2_cv: 0.2104
- r2_range: 0.1051
- mse_cv: 0.0467
- mse_range: 0.0174
- mae_cv: 0.0180
- mae_range: 0.0119
- precision_cv: 0.0321
- precision_range: 0.0634
- accuracy_cv: 0.0132
- accuracy_range: 0.0285
- rmse_cv: 0.0231
- rmse_range: 0.0235
- f1_score_cv: 0.0297
- f1_score_range: 0.0555

**Improvement:**
- recall_abs_improvement: -0.0003
- recall_rel_improvement: -0.0432
- r2_abs_improvement: +0.0021
- r2_rel_improvement: +1.1893
- mse_abs_improvement: -0.0003
- mse_rel_improvement: -0.2569
- mae_abs_improvement: +0.0003
- mae_rel_improvement: +0.0961
- precision_abs_improvement: -0.0010
- precision_rel_improvement: -0.1393
- accuracy_abs_improvement: -0.0003
- accuracy_rel_improvement: -0.0350
- rmse_abs_improvement: -0.0005
- rmse_rel_improvement: -0.1274
- f1_score_abs_improvement: -0.0006
- f1_score_rel_improvement: -0.0896

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 75.5392
- **Sharpe Ratio:** 75.5392
- **Sortino Ratio:** 81.4162

---
