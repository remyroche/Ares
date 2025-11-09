# Training Report: ensemble_base

**Session ID:** ensemble_base_ETHUSDT_20251109_145303
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-09T14:53:03.370166
**Total Training Time:** 0.75s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 20
- **Features:** 42

---

## Best Model
**Name:** ensemble_catboost

**Metrics:**
- mse_mean: 0.1532
- mse_std: 0.0363
- recall_mean: 0.4722
- recall_std: 0.0278
- precision_mean: 0.4222
- precision_std: 0.0222
- accuracy_mean: 0.8000
- accuracy_std: 0.0000
- mae_mean: 0.2087
- mae_std: 0.0242
- r2_mean: -0.2418
- r2_std: 0.0576
- rmse_mean: 0.3886
- rmse_std: 0.0467
- f1_score_mean: 0.4444
- f1_score_std: 0.0000

---

## Model Training Details

### ensemble_catboost (catboost)

#### Pre-HPO Metrics

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- mse_mean: 0.1532
- mse_std: 0.0363
- recall_mean: 0.4722
- recall_std: 0.0278
- precision_mean: 0.4222
- precision_std: 0.0222
- accuracy_mean: 0.8000
- accuracy_std: 0.0000
- mae_mean: 0.2087
- mae_std: 0.0242
- r2_mean: -0.2418
- r2_std: 0.0576
- rmse_mean: 0.3886
- rmse_std: 0.0467
- f1_score_mean: 0.4444
- f1_score_std: 0.0000

**Fold Stability (Post-HPO):**
- mse_cv: 0.2367
- mse_range: 0.0725
- recall_cv: 0.0588
- recall_range: 0.0556
- precision_cv: 0.0526
- precision_range: 0.0444
- accuracy_cv: 0.0000
- accuracy_range: 0.0000
- mae_cv: 0.1160
- mae_range: 0.0484
- r2_cv: -0.2381
- r2_range: 0.1152
- rmse_cv: 0.1201
- rmse_range: 0.0933
- f1_score_cv: 0.0000
- f1_score_range: 0.0000

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 80.0000
- **Sharpe Ratio:** 80.0000
- **Sortino Ratio:** 80.0000

**Top 10 Important Features:**
- vectorbt_sma_100: 12.6552
- vectorbt_garman_klass_volatility_50: 10.0531
- fibonacci_0.5_20_price_returns_vwap: 9.1912
- sma_50_returns_vwap: 6.7211
- vectorbt_momentum_acceleration_10_20_price_returns: 6.2656
- vectorbt_rogers_satchell_volatility_30_base_9x_ratio: 4.2293
- vectorbt_garman_klass_volatility_20: 4.0300
- pivot_point_5_price_returns: 3.9662
- tema_21_price_returns: 3.6327
- vectorbt_enhanced_obv_20: 3.5990

---
