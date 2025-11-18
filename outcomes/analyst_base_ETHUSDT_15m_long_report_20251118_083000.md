# Analyst Base - Comprehensive Training Report

**Generated:** 2025-11-18 08:30:00 UTC

## 📋 Execution Summary

- **Training Type:** analyst_base
- **Success:** ✅ Yes
- **Execution Time:** 0.00 seconds
- **Models Trained:** 1
- **Model Names:** lightgbm

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

| Metric | Value |
|--------|-------|
| Avg Generalization Score | 1.000000 |
| Avg Overfitting Ratio | 0.000000 |
| Generalization Score | 1.000000 |
| Overfitting Ratio | 0.000000 |
| Std Generalization Score | 0.000000 |
| Std Overfitting Ratio | 0.000000 |
| Train Test R2 Gap | 0.000000 |

---

## 📈 Split-Based Performance Metrics

### Training Set Metrics

| Metric | Value |
|--------|-------|
| Mae | 0.000000 |
| Mse | 0.000000 |
| R2 | 1.000000 |
| Rmse | 0.000000 |

### Validation Set Metrics

| Metric | Value |
|--------|-------|
| Mae | 0.000000 |
| Mse | 0.000000 |
| R2 | 1.000000 |
| Rmse | 0.000000 |

### Test Set Metrics

| Metric | Value |
|--------|-------|
| Mae | 0.000000 |
| Mse | 0.000000 |
| R2 | 1.000000 |
| Rmse | 0.000000 |

---

## 📚 Learnability & Generalization Diagnostics

This section summarizes how the model learns from data and how robustly it generalizes.

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| R² | 1.0000 | 1.0000 | 1.0000 |

### Overfitting & Generalization Indicators

- **Train–Test R² Gap:** 0.0000  \n  Larger gaps indicate that the model fits the training data much better than unseen data.
- **Overfitting Ratio:** 0.0000  \n  Approximate relative gap between train and test performance → **low (good)**.
- **Generalization Score:** 1.0000  \n  Ratio of test to train performance; values near 1.0 indicate similar train/test behaviour.

---

## 🤖 Per-Model Detailed Metrics

**Total Models:** 1

### LIGHTGBM

| Metric | Value |
|--------|-------|
| Best Iteration | 1.000000 |
| Generalization Score | 1.000000 |
| Iterations Used | 1.000000 |
| Mae | 0.000000 |
| Mse | 0.000000 |
| Overfitting Ratio | 0.000000 |
| R2 | 1.000000 |
| Rmse | 0.000000 |
| Test Mae | 0.000000 |
| Test Mse | 0.000000 |
| Test R2 | 1.000000 |
| Test Rmse | 0.000000 |
| Train Mae | 0.000000 |
| Train Mse | 0.000000 |
| Train R2 | 1.000000 |
| Train Rmse | 0.000000 |
| Train Test R2 Gap | 0.000000 |
| Val Mae | 0.000000 |
| Val Mse | 0.000000 |
| Val R2 | 1.000000 |
| Val Rmse | 0.000000 |

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
| 1 | acceleration_features | 0.000000 |
| 2 | ar_1_coefficients_20_base_9x_ratio | 0.000000 |
| 3 | candlestick_harami_cross_pattern_vwap_3x_ratio | 0.000000 |
| 4 | candlestick_harami_cross_pattern_vwap_3x_ratio_log_ratio_vectorbt_rogers_satchell_volatility_30_vwap_x_3x | 0.000000 |
| 5 | enhanced_volatility_50 | 0.000000 |
| 6 | fibonacci_0.236_5_price_returns_vwap_log_wavelet_energy_vwap_x_27x | 0.000000 |
| 7 | fibonacci_0.236_5_price_returns_vwap_x_wavelet_energy_vwap_x_27x | 0.000000 |
| 8 | fibonacci_0.5_10_price_returns_vwap_log_ratio_fibonacci_0.786_10_price_returns_vwap_x_9x | 0.000000 |
| 9 | fibonacci_0.5_10_price_returns_vwap_x_vectorbt_volatility_comprehensive_50_vwap | 0.000000 |
| 10 | fibonacci_0.786_10_price_returns | 0.000000 |
| 11 | fibonacci_0.786_10_price_returns_vwap_x_9x_log_ratio_vectorbt_parkinson_volatility_50_vwap_x_27x | 0.000000 |
| 12 | fibonacci_0.786_20_price_returns | 0.000000 |
| 13 | hurst_exponent | 0.000000 |
| 14 | macd_12_26_9_returns_vwap | 0.000000 |
| 15 | macd_entropy_20_12_26 | 0.000000 |
| 16 | momentum_21_price_returns | 0.000000 |
| 17 | momentum_features | 0.000000 |
| 18 | resistance_level_1_5_price_returns | 0.000000 |
| 19 | returns_kurtosis_20_price_returns | 0.000000 |
| 20 | rsi_21_returns_vwap | 0.000000 |

---

## 📊 Data Quality Metrics

| Metric | Value |
|--------|-------|
| Feature Count | 56.000000 |
| Mean Feature Std | 4.430800 |
| Mean Feature Variance | 160.199112 |
| Missing Values Count | 0.000000 |
| Missing Values Pct | 0.000000 |
| Numeric Features Count | 56.000000 |
| Sample Count | 1204.000000 |
| Target Max | 0.000000 |
| Target Mean | 0.000000 |
| Target Min | 0.000000 |
| Target Range | 0.000000 |
| Target Std | 0.000000 |

---

## 🧮 Model Complexity Metrics

| Metric | Value |
|--------|-------|
| Lightgbm | {'num_trees': 1, 'num_leaves': 32, 'max_depth': 5} |

---

## 📊 Prediction Statistics

| Statistic | Value |
|-----------|-------|
| Prediction Kurtosis | nan |
| Prediction Max | 0.000000 |
| Prediction Mean | 0.000000 |
| Prediction Median | 0.000000 |
| Prediction Min | 0.000000 |
| Prediction Skewness | nan |
| Prediction Std | 0.000000 |

---

## ⚠️ Error Analysis

*No error analysis metrics available.*

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
| analyst_base_config | `artifacts/analyst_base_config.json` |
| analyst_base_lightgbm | `artifacts/analyst_base_lightgbm.pkl` |
| analyst_base_metrics | `artifacts/analyst_base_metrics.pkl` |
| analyst_base_metrics_report | `outcomes/analyst_base_ETHUSDT_15m_long_report_20251118_083000.md` |
| analyst_base_predictions | `versioned_artifacts/ETHUSDT_binance_15m_long_Analyst/analyst_base_predictions_20251118_082959_986.h5` |
| analyst_base_predictions_oof | `versioned_artifacts/ETHUSDT_binance_15m_long_Analyst/analyst_base_predictions_oof_20251118_083000_016.h5` |
| ml_scored_historical_data_oos | `ml_scored_historical_data_analyst_long_oos` |
| training_report_json | `outcomes/analyst_base_ETHUSDT_15m_long_metrics_20251118_083000.json` |
| training_report_markdown | `outcomes/analyst_base_ETHUSDT_15m_long_report_20251118_083000.md` |

---

*Comprehensive report generated by Ares Unified Training Pipeline v3.0 on 20251118_083000*
*Training Type: ANALYST_BASE | Symbol: ETHUSDT | Timeframe: 15m | Direction: long*
