# Training Report: analyst_base

**Session ID:** analyst_base_ETHUSDT_20251117_234143
**Symbol:** ETHUSDT
**Timeframe:** 15m
**Timestamp:** 2025-11-17T23:41:43.941737
**Total Training Time:** 20.88s

---

## Data Quality
- **Quality Score:** 85.00%
- **Samples:** 1,000
- **Features:** 72

---

## Best Model
**Name:** analyst_lightgbm

**Metrics:**
- rmse_mean: 0.0000
- rmse_std: 0.0000
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: -0.0023
- r2_std: 0.0038

---

## Model Training Details

### analyst_lightgbm (lightgbm)

#### Pre-HPO Metrics
- rmse_mean: 0.0000
- rmse_std: 0.0000
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: -0.5664
- r2_std: 1.1244

**Fold Stability (Pre-HPO):**
- rmse_cv: 0.7550
- rmse_range: 0.0001
- mse_cv: 1.3365
- mse_range: 0.0000
- mae_cv: 0.4694
- mae_range: 0.0000
- r2_cv: -1.9851
- r2_range: 2.8152

#### Hyperparameter Optimization
- **Trials:** 20
- **Time:** 0.00s

#### Post-HPO Metrics
- rmse_mean: 0.0000
- rmse_std: 0.0000
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: -0.0023
- r2_std: 0.0038

**Fold Stability (Post-HPO):**
- rmse_cv: 1.3933
- rmse_range: 0.0001
- mse_cv: 1.8445
- mse_range: 0.0000
- mae_cv: 0.8276
- mae_range: 0.0000
- r2_cv: -1.6403
- r2_range: 0.0098

**Improvement:**
- rmse_abs_improvement: -0.0000
- rmse_rel_improvement: -36.0716
- mse_abs_improvement: -0.0000
- mse_rel_improvement: -23.4374
- mae_abs_improvement: -0.0000
- mae_rel_improvement: -64.4127
- r2_abs_improvement: +0.5641
- r2_rel_improvement: -99.5885

---

### analyst_catboost (catboost)

#### Pre-HPO Metrics
- rmse_mean: 0.0000
- rmse_std: 0.0000
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: 0.0379
- r2_std: 0.0656

**Fold Stability (Pre-HPO):**
- rmse_cv: 0.4349
- rmse_range: 0.0000
- mse_cv: 0.7132
- mse_range: 0.0000
- mae_cv: 0.3092
- mae_range: 0.0000
- r2_cv: 1.7321
- r2_range: 0.1515

#### Hyperparameter Optimization
- **Trials:** 0
- **Time:** 0.00s

#### Post-HPO Metrics
- rmse_mean: 0.0000
- rmse_std: 0.0000
- mse_mean: 0.0000
- mse_std: 0.0000
- mae_mean: 0.0000
- mae_std: 0.0000
- r2_mean: -0.1094
- r2_std: 0.1894

**Fold Stability (Post-HPO):**
- rmse_cv: 0.3225
- rmse_range: 0.0000
- mse_cv: 0.5615
- mse_range: 0.0000
- mae_cv: 0.4619
- mae_range: 0.0000
- r2_cv: -1.7321
- r2_range: 0.4375

**Improvement:**
- rmse_abs_improvement: +0.0000
- rmse_rel_improvement: +66.9612
- mse_abs_improvement: +0.0000
- mse_rel_improvement: +158.8014
- mae_abs_improvement: +0.0000
- mae_rel_improvement: +105.5540
- r2_abs_improvement: -0.1472
- r2_rel_improvement: -388.8020

**Top 10 Important Features:**
- vectorbt_acceleration_momentum_10_20_price_returns: 31.0700
- vectorbt_acceleration_volatility_10_20_price_returns: 7.2914
- fibonacci_0.5_10_price_returns_vwap_log_ratio_fibonacci_0.786_10_price_returns_vwap_x_9x: 6.8994
- vwma_20_price_returns_vwap_div_cycle_length_vwap_6x_ratio: 5.7508
- fibonacci_0.236_5_price_returns_vwap_x_wavelet_energy_vwap_x_27x: 4.5934
- macd_12_26_9_returns_vwap: 3.5362
- ar_1_coefficients_20_base_9x_ratio: 2.9931
- momentum_features: 2.6718
- sma_10_returns_vwap: 2.3307
- fibonacci_0.786_10_price_returns_vwap_x_9x_log_ratio_vectorbt_parkinson_volatility_50_vwap_x_27x: 2.1485

---
