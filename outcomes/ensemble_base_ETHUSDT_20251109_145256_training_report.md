# Training Report: ensemble_base

**Session ID:** ensemble_base_ETHUSDT_20251109_145256
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-09T14:52:56.682144
**Total Training Time:** 6.50s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 20
- **Features:** 42

---

## Best Model
**Name:** ensemble_catboost

**Metrics:**
- mse_mean: 0.1453
- mse_std: 0.0392
- recall_mean: 0.5000
- recall_std: 0.0000
- precision_mean: 0.4250
- precision_std: 0.0250
- accuracy_mean: 0.8500
- accuracy_std: 0.0500
- mae_mean: 0.1899
- mae_std: 0.0320
- r2_mean: -0.1663
- r2_std: 0.0126
- rmse_mean: 0.3777
- rmse_std: 0.0519
- f1_score_mean: 0.4591
- f1_score_std: 0.0146

---

## Model Training Details

### ensemble_catboost (catboost)

#### Pre-HPO Metrics

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- mse_mean: 0.1453
- mse_std: 0.0392
- recall_mean: 0.5000
- recall_std: 0.0000
- precision_mean: 0.4250
- precision_std: 0.0250
- accuracy_mean: 0.8500
- accuracy_std: 0.0500
- mae_mean: 0.1899
- mae_std: 0.0320
- r2_mean: -0.1663
- r2_std: 0.0126
- rmse_mean: 0.3777
- rmse_std: 0.0519
- f1_score_mean: 0.4591
- f1_score_std: 0.0146

**Fold Stability (Post-HPO):**
- mse_cv: 0.2700
- mse_range: 0.0785
- recall_cv: 0.0000
- recall_range: 0.0000
- precision_cv: 0.0588
- precision_range: 0.0500
- accuracy_cv: 0.0588
- accuracy_range: 0.1000
- mae_cv: 0.1688
- mae_range: 0.0641
- r2_cv: -0.0760
- r2_range: 0.0253
- rmse_cv: 0.1375
- rmse_range: 0.1039
- f1_score_cv: 0.0318
- f1_score_range: 0.0292

#### Risk-Reward Metrics
- **Risk-Reward Ratio:** 17.0000
- **Sharpe Ratio:** 17.0000
- **Sortino Ratio:** 85.0000

**Top 10 Important Features:**
- vectorbt_garman_klass_volatility_20: 11.8848
- sma_50_returns_vwap: 9.9208
- volume_vwap_20_base_3x_ratio: 9.6051
- vectorbt_momentum_acceleration_10_20_price_returns: 8.8862
- vectorbt_sma_100: 7.6306
- pred_catboost: 6.5007
- advanced_support_resistance_features: 5.2598
- vectorbt_enhanced_obv_20: 3.6431
- fibonacci_0.5_20_price_returns_trend_adj: 3.4178
- volume_std_50: 3.3096

---
