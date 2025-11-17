# Analyst Base - Comprehensive Training Report

**Generated:** 2025-11-16 20:16:32 UTC

## 📋 Execution Summary

- **Training Type:** analyst_base
- **Success:** ✅ Yes
- **Execution Time:** 0.00 seconds
- **Models Trained:** 2
- **Model Names:** lightgbm, catboost

---

## ⚙️ Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Direction:** long
- **Execution Mode:** blank
- **HPO Enabled:** True

---

## 📊 Overall Performance Metrics

*No overall performance metrics available.*

---

## 📈 Split-Based Performance Metrics

### Training Set Metrics

| Metric | Value |
|--------|-------|
| Mae | 0.051850 |
| Mse | 0.015703 |
| R2 | 0.149253 |
| Rmse | 0.125310 |

### Validation Set Metrics

| Metric | Value |
|--------|-------|
| Mae | 0.098932 |
| Mse | 0.041270 |
| R2 | -0.028485 |
| Rmse | 0.203150 |

### Test Set Metrics

| Metric | Value |
|--------|-------|
| Mae | 0.098599 |
| Mse | 0.038760 |
| R2 | -0.046052 |
| Rmse | 0.196873 |

---

## 🤖 Per-Model Detailed Metrics

**Total Models:** 2

### LIGHTGBM

| Metric | Value |
|--------|-------|
| Best Iteration | 5.000000 |
| Generalization Score | -0.352979 |
| Iterations Used | 5.000000 |
| Mae | 0.106413 |
| Mse | 0.039042 |
| Overfitting Ratio | 1.352979 |
| R2 | -0.053678 |
| Rmse | 0.197591 |
| Test Mae | 0.106413 |
| Test Mse | 0.039042 |
| Test R2 | -0.053678 |
| Test Rmse | 0.197591 |
| Train Mae | 0.052331 |
| Train Mse | 0.015651 |
| Train R2 | 0.152071 |
| Train Rmse | 0.125103 |
| Train Test R2 Gap | 0.205748 |
| Val Mae | 0.100855 |
| Val Mse | 0.041068 |
| Val R2 | -0.023452 |
| Val Rmse | 0.202653 |

### CATBOOST

| Metric | Value |
|--------|-------|
| Best Iteration | 19.000000 |
| Generalization Score | -0.262411 |
| Iterations Used | 19.000000 |
| Mae | 0.090785 |
| Mse | 0.038477 |
| Overfitting Ratio | 1.262411 |
| R2 | -0.038427 |
| Rmse | 0.196156 |
| Test Mae | 0.090785 |
| Test Mse | 0.038477 |
| Test R2 | -0.038427 |
| Test Rmse | 0.196156 |
| Train Mae | 0.051368 |
| Train Mse | 0.015755 |
| Train R2 | 0.146436 |
| Train Rmse | 0.125518 |
| Train Test R2 Gap | 0.184863 |
| Val Mae | 0.097010 |
| Val Mse | 0.041472 |
| Val R2 | -0.033518 |
| Val Rmse | 0.203647 |

---

## 🔍 Hyperparameter Optimization (HPO) Results

*No HPO results available or HPO was disabled.*

---

## 📅 Walk-Forward Validation Results

**Number of Folds:** 3
**Strategy:** expanding
**Embargo Days:** 0

---

## 📋 Feature Importance

### Top 20 Most Important Features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | fibonacci_0.786_10_price_returns_vwap_x_9x_log_ratio_vectorbt_parkinson_volatility_50_vwap_x_27x | 6.572042 |
| 2 | vectorbt_acceleration_momentum_10_20_price_returns | 5.963562 |
| 3 | fibonacci_0.236_5_price_returns_vwap_log_wavelet_energy_vwap_x_27x | 5.288416 |
| 4 | ultimate_oscillator_7_14_28_returns_vwap | 4.528706 |
| 5 | hurst_exponent | 3.612702 |
| 6 | returns_kurtosis_20_price_returns | 3.351281 |
| 7 | macd_entropy_20_12_26 | 3.117851 |
| 8 | fibonacci_0.5_10_price_returns_vwap_log_ratio_fibonacci_0.786_10_price_returns_vwap_x_9x | 2.353890 |
| 9 | vectorbt_acceleration_5_price_returns | 2.050798 |
| 10 | sma_50_returns_vwap | 1.799570 |
| 11 | volume_price_trend_vwap_x_volume_entropy_10_volume_returns_vwap | 1.599521 |
| 12 | volume_momentum_20 | 1.448868 |
| 13 | vectorbt_momentum_5_price_returns | 1.414246 |
| 14 | fibonacci_0.786_20_price_returns | 1.344577 |
| 15 | vectorbt_enhanced_obv_50_base_27x_ratio | 1.282885 |
| 16 | macd_12_26_9_returns_vwap | 1.210452 |
| 17 | vectorbt_parabolic_sar_0.1_0.3 | 1.203694 |
| 18 | lightgbm_regime_3_prob | 1.145288 |
| 19 | vectorbt_zigzag_3.0_2 | 1.101714 |
| 20 | momentum_features | 1.012620 |

---

## 📊 Data Quality Metrics

| Metric | Value |
|--------|-------|
| Feature Count | 72.000000 |
| Mean Feature Std | 3.474572 |
| Mean Feature Variance | 124.602173 |
| Missing Values Count | 0.000000 |
| Missing Values Pct | 0.000000 |
| Numeric Features Count | 72.000000 |
| Sample Count | 1204.000000 |
| Target Max | 0.685767 |
| Target Mean | 0.043253 |
| Target Min | 0.000000 |
| Target Range | 0.685767 |
| Target Std | 0.157693 |

---

## 🧮 Model Complexity Metrics

| Metric | Value |
|--------|-------|
| Catboost | {'num_trees': 400, 'depth': 5} |
| Lightgbm | {'num_trees': 5, 'num_leaves': 32, 'max_depth': 5} |

---

## 📊 Prediction Statistics

| Statistic | Value |
|-----------|-------|
| Prediction Kurtosis | 2.099367 |
| Prediction Max | 0.133048 |
| Prediction Mean | 0.042045 |
| Prediction Median | 0.034542 |
| Prediction Min | 0.007025 |
| Prediction Skewness | 1.575816 |
| Prediction Std | 0.024757 |

---

## ⚠️ Error Analysis

| Metric | Value |
|--------|-------|
| Avg Mae Rmse Ratio | 0.500686 |
| Catboost Mae Rmse Ratio | 0.462822 |
| Lightgbm Mae Rmse Ratio | 0.538550 |

---

## 📊 Data Drift & Distribution Shift Checks

*Detects if train/val/test distributions differ significantly (KS tests, PSI, chi-square)*

*No data drift checks available.*

---

## 🎯 Uncertainty & Confidence Calibration

*Measures how well predicted probabilities match actual outcomes (Brier Score, ECE)*

*No uncertainty/calibration metrics available.*

---

## 🔍 SHAP Explanations & Model Interpretability

*Shapley values, PDP/ICE curves, and feature attribution*

*No SHAP explanations available.*

---

## ⚖️ Decision Threshold Optimization

*ROC/PR curves, F-beta optimization, cost-weighted thresholds*

*No threshold optimization metrics available.*

---

## 💾 Generated Artifacts

| Artifact Name | Path |
|---------------|------|
| analyst_base_catboost | `artifacts/analyst_base_catboost.pkl` |
| analyst_base_config | `artifacts/analyst_base_config.json` |
| analyst_base_lightgbm | `artifacts/analyst_base_lightgbm.pkl` |
| analyst_base_metrics | `artifacts/analyst_base_metrics.pkl` |
| analyst_base_metrics_report | `outcomes/analyst_base_ETHUSDT_15m_long_report_20251116_201631.md` |
| analyst_base_predictions | `versioned_artifacts/ETHUSDT_binance_15m_long_Analyst/analyst_base_predictions_20251116_201631_936.h5` |
| analyst_base_predictions_oof | `versioned_artifacts/ETHUSDT_binance_15m_long_Analyst/analyst_base_predictions_oof_20251116_201631_967.h5` |
| ml_scored_historical_data_oos | `ml_scored_historical_data_analyst_long_oos` |
| training_report_json | `outcomes/analyst_base_ETHUSDT_15m_long_metrics_20251116_201631.json` |
| training_report_markdown | `outcomes/analyst_base_ETHUSDT_15m_long_report_20251116_201631.md` |

---

*Comprehensive report generated by Ares Unified Training Pipeline v3.0 on 20251116_201632*
*Training Type: ANALYST_BASE | Symbol: ETHUSDT | Timeframe: 15m | Direction: long*
