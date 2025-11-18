# Analyst Base - Comprehensive Training Report

**Generated:** 2025-11-17 23:42:04 UTC

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
- **Execution Mode:** light
- **HPO Enabled:** True

---

## 📊 Overall Performance Metrics

| Metric | Value |
|--------|-------|
| Avg Generalization Score | 0.000000 |
| Avg Overfitting Ratio | 0.500000 |
| Generalization Score | 0.000000 |
| Overfitting Ratio | 0.500000 |
| Std Generalization Score | 0.000000 |
| Std Overfitting Ratio | 0.500000 |
| Train Test R2 Gap | 0.148210 |

---

## 📈 Split-Based Performance Metrics

### Training Set Metrics

| Metric | Value |
|--------|-------|
| Mae | 0.000004 |
| Mse | 0.000000 |
| R2 | 0.148210 |
| Rmse | 0.000037 |

### Validation Set Metrics

| Metric | Value |
|--------|-------|
| Mae | 0.000004 |
| Mse | 0.000000 |
| R2 | 0.007063 |
| Rmse | 0.000022 |

### Test Set Metrics

| Metric | Value |
|--------|-------|
| Mae | 0.000002 |
| Mse | 0.000000 |
| R2 | 0.000000 |
| Rmse | 0.000002 |

---

## 📚 Learnability & Generalization Diagnostics

This section summarizes how the model learns from data and how robustly it generalizes.

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| R² | 0.1482 | 0.0071 | 0.0000 |

### Overfitting & Generalization Indicators

- **Train–Test R² Gap:** 0.1482  \n  Larger gaps indicate that the model fits the training data much better than unseen data.
- **Overfitting Ratio:** 0.5000  \n  Approximate relative gap between train and test performance → **high (risk of overfitting)**.
- **Generalization Score:** 0.0000  \n  Ratio of test to train performance; values near 1.0 indicate similar train/test behaviour.

---

## 🤖 Per-Model Detailed Metrics

**Total Models:** 2

### LIGHTGBM

| Metric | Value |
|--------|-------|
| Best Iteration | 1.000000 |
| Generalization Score | 0.000000 |
| Iterations Used | 1.000000 |
| Mae | 0.000003 |
| Mse | 0.000000 |
| Overfitting Ratio | 0.000000 |
| R2 | 0.000000 |
| Rmse | 0.000003 |
| Test Mae | 0.000003 |
| Test Mse | 0.000000 |
| Test R2 | 0.000000 |
| Test Rmse | 0.000003 |
| Train Mae | 0.000005 |
| Train Mse | 0.000000 |
| Train R2 | 0.000000 |
| Train Rmse | 0.000040 |
| Train Test R2 Gap | 0.000000 |
| Val Mae | 0.000004 |
| Val Mse | 0.000000 |
| Val R2 | -0.001057 |
| Val Rmse | 0.000022 |

### CATBOOST

| Metric | Value |
|--------|-------|
| Best Iteration | 37.000000 |
| Generalization Score | 0.000000 |
| Iterations Used | 37.000000 |
| Mae | 0.000001 |
| Mse | 0.000000 |
| Overfitting Ratio | 1.000000 |
| R2 | 0.000000 |
| Rmse | 0.000001 |
| Test Mae | 0.000001 |
| Test Mse | 0.000000 |
| Test R2 | 0.000000 |
| Test Rmse | 0.000001 |
| Train Mae | 0.000003 |
| Train Mse | 0.000000 |
| Train R2 | 0.296420 |
| Train Rmse | 0.000034 |
| Train Test R2 Gap | 0.296420 |
| Val Mae | 0.000004 |
| Val Mse | 0.000000 |
| Val R2 | 0.015182 |
| Val Rmse | 0.000022 |

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
| 1 | vectorbt_acceleration_momentum_10_20_price_returns | 11.994473 |
| 2 | sma_10_returns_vwap | 6.556174 |
| 3 | fibonacci_0.5_10_price_returns_vwap_log_ratio_fibonacci_0.786_10_price_returns_vwap_x_9x | 3.549912 |
| 4 | volume_price_divergence_10 | 3.255554 |
| 5 | macd_12_26_9_returns_vwap | 2.998468 |
| 6 | sma_50_returns_vwap | 2.606648 |
| 7 | ar_1_coefficients_20_base_9x_ratio | 1.943992 |
| 8 | momentum_features | 1.592709 |
| 9 | fibonacci_0.236_5_price_returns_vwap_log_wavelet_energy_vwap_x_27x | 1.538286 |
| 10 | macd_entropy_20_12_26 | 1.467966 |
| 11 | resistance_level_1_5_price_returns | 1.282034 |
| 12 | vwma_20_price_returns_vwap_div_cycle_length_vwap_6x_ratio | 0.977124 |
| 13 | volume_oscillator_5_15 | 0.913184 |
| 14 | lightgbm_regime_3_prob | 0.884598 |
| 15 | vectorbt_acceleration_5_price_returns | 0.707463 |
| 16 | volume_price_trend_vwap_log_wavelet_energy_vwap_x_9x | 0.689887 |
| 17 | acceleration_features | 0.634508 |
| 18 | vectorbt_enhanced_obv_50_base_27x_ratio | 0.627535 |
| 19 | rsi_21_returns_vwap | 0.610149 |
| 20 | fibonacci_0.236_5_price_returns_vwap_x_wavelet_energy_vwap_x_27x | 0.495151 |

---

## 📊 Data Quality Metrics

| Metric | Value |
|--------|-------|
| Feature Count | 72.000000 |
| Mean Feature Std | 3.645299 |
| Mean Feature Variance | 146.775940 |
| Missing Values Count | 0.000000 |
| Missing Values Pct | 0.000000 |
| Numeric Features Count | 72.000000 |
| Sample Count | 1000.000000 |
| Target Max | 0.000962 |
| Target Mean | 0.000002 |
| Target Min | 0.000000 |
| Target Range | 0.000962 |
| Target Std | 0.000035 |

---

## 🧮 Model Complexity Metrics

| Metric | Value |
|--------|-------|
| Catboost | {'num_trees': 400, 'depth': 5} |
| Lightgbm | {'num_trees': 1, 'num_leaves': 32, 'max_depth': 5} |

---

## 📊 Prediction Statistics

| Statistic | Value |
|-----------|-------|
| Prediction Kurtosis | -1.678413 |
| Prediction Max | 0.000003 |
| Prediction Mean | 0.000002 |
| Prediction Median | 0.000003 |
| Prediction Min | 0.000000 |
| Prediction Skewness | -0.286469 |
| Prediction Std | 0.000001 |

---

## ⚠️ Error Analysis

| Metric | Value |
|--------|-------|
| Avg Mae Rmse Ratio | 0.943475 |
| Catboost Mae Rmse Ratio | 0.886950 |
| Lightgbm Mae Rmse Ratio | 1.000000 |

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
| analyst_base_metrics_report | `outcomes/analyst_base_ETHUSDT_15m_long_report_20251117_234204.md` |
| analyst_base_predictions | `versioned_artifacts/ETHUSDT_binance_15m_long_Analyst/analyst_base_predictions_20251117_234204_855.h5` |
| analyst_base_predictions_oof | `versioned_artifacts/ETHUSDT_binance_15m_long_Analyst/analyst_base_predictions_oof_20251117_234204_890.h5` |
| ml_scored_historical_data_oos | `ml_scored_historical_data_analyst_long_oos` |
| training_report_json | `outcomes/analyst_base_ETHUSDT_15m_long_metrics_20251117_234204.json` |
| training_report_markdown | `outcomes/analyst_base_ETHUSDT_15m_long_report_20251117_234204.md` |

---

*Comprehensive report generated by Ares Unified Training Pipeline v3.0 on 20251117_234204*
*Training Type: ANALYST_BASE | Symbol: ETHUSDT | Timeframe: 15m | Direction: long*
