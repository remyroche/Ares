# Analyst Base - Comprehensive Training Report

**Generated:** 2025-11-21 00:46:05 UTC

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

| Metric | Value |
|--------|-------|
| Avg Generalization Score | -1.088688 |
| Avg Overfitting Ratio | 1.588688 |
| Generalization Score | -1.088688 |
| Overfitting Ratio | 1.588688 |
| Std Generalization Score | 1.075769 |
| Std Overfitting Ratio | 0.575769 |
| Train Test R2 Gap | 0.399057 |

---

## 📈 Split-Based Performance Metrics

### Training Set Metrics

| Metric | Value |
|--------|-------|
| Mae | 0.000007 |
| Mse | 0.000000 |
| R2 | 0.383283 |
| Rmse | 0.000058 |

### Validation Set Metrics

| Metric | Value |
|--------|-------|
| Mae | 0.000392 |
| Mse | 0.000004 |
| R2 | -0.028864 |
| Rmse | 0.001983 |

### Test Set Metrics

| Metric | Value |
|--------|-------|
| Mae | 0.000274 |
| Mse | 0.000003 |
| R2 | -0.015774 |
| Rmse | 0.001618 |

---

## 📚 Learnability & Generalization Diagnostics

This section summarizes how the model learns from data and how robustly it generalizes.

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| R² | 0.3833 | -0.0289 | -0.0158 |

### Overfitting & Generalization Indicators

- **Train–Test R² Gap:** 0.3991  \n  Larger gaps indicate that the model fits the training data much better than unseen data.
- **Overfitting Ratio:** 1.5887  \n  Approximate relative gap between train and test performance → **high (risk of overfitting)**.
- **Generalization Score:** -1.0887  \n  Ratio of test to train performance; values near 1.0 indicate similar train/test behaviour.

---

## 🤖 Per-Model Detailed Metrics

**Total Models:** 2

### LIGHTGBM

| Metric | Value |
|--------|-------|
| Best Iteration | 1.000000 |
| Generalization Score | -2.164457 |
| Iterations Used | 1.000000 |
| Mae | 0.000246 |
| Mse | 0.000003 |
| Overfitting Ratio | 2.164457 |
| R2 | -0.021645 |
| Rmse | 0.001623 |
| Test Mae | 0.000246 |
| Test Mse | 0.000003 |
| Test R2 | -0.021645 |
| Test Rmse | 0.001623 |
| Train Mae | 0.000010 |
| Train Mse | 0.000000 |
| Train R2 | -0.000000 |
| Train Rmse | 0.000078 |
| Train Test R2 Gap | 0.021645 |
| Val Mae | 0.000367 |
| Val Mse | 0.000004 |
| Val R2 | -0.033317 |
| Val Rmse | 0.001987 |

### CATBOOST

| Metric | Value |
|--------|-------|
| Best Iteration | 50.000000 |
| Generalization Score | -0.012919 |
| Iterations Used | 50.000000 |
| Mae | 0.000301 |
| Mse | 0.000003 |
| Overfitting Ratio | 1.012919 |
| R2 | -0.009903 |
| Rmse | 0.001613 |
| Test Mae | 0.000301 |
| Test Mse | 0.000003 |
| Test R2 | -0.009903 |
| Test Rmse | 0.001613 |
| Train Mae | 0.000005 |
| Train Mse | 0.000000 |
| Train R2 | 0.766565 |
| Train Rmse | 0.000038 |
| Train Test R2 Gap | 0.776469 |
| Val Mae | 0.000418 |
| Val Mse | 0.000004 |
| Val R2 | -0.024412 |
| Val Rmse | 0.001979 |

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
| 1 | candlestick_harami_cross_pattern_vwap_3x_ratio | 5.571678 |
| 2 | vectorbt_trend_consistency_10_price_returns | 5.407165 |
| 3 | volume_price_trend_vwap_div_candlestick_piercing_line_pattern_vwap_3x_ratio | 4.987172 |
| 4 | volume_price_trend_vwap | 4.661139 |
| 5 | resistance_level_1_5_price_returns | 4.598034 |
| 6 | vectorbt_parabolic_sar_0.1_0.3 | 2.924838 |
| 7 | shannon_entropy_20_10 | 2.572380 |
| 8 | volume_price_trend_vwap_log_wavelet_energy_vwap_x_9x | 2.211262 |
| 9 | candlestick_harami_cross_pattern_vwap_3x_ratio_log_ratio_vectorbt_rogers_satchell_volatility_30_vwap_x_3x | 1.963525 |
| 10 | volume_roc_1 | 1.837848 |
| 11 | fibonacci_0.236_5_price_returns_vwap_x_wavelet_energy_vwap_x_27x | 1.520672 |
| 12 | macd_12_26_9_returns_vwap | 1.150695 |
| 13 | vectorbt_parkinson_volatility_50_vwap_27x_ratio_log_ratio_candlestick_engulfing_pattern_base_9x_ratio | 0.882382 |
| 14 | fibonacci_0.236_5_price_returns_vwap_log_wavelet_energy_vwap_x_27x | 0.815927 |
| 15 | volume_price_divergence_10 | 0.814611 |
| 16 | vectorbt_acceleration_trend_strength_5_20_price_returns | 0.803973 |
| 17 | fibonacci_0.5_10_price_returns_vwap_x_vectorbt_volatility_comprehensive_50_vwap | 0.784613 |
| 18 | momentum_features | 0.731912 |
| 19 | volume_roc_5 | 0.726606 |
| 20 | vectorbt_rogers_satchell_volatility_30_vwap_6x_ratio | 0.613222 |

---

## 📊 Data Quality Metrics

| Metric | Value |
|--------|-------|
| Feature Count | 58.000000 |
| Mean Feature Std | 4.266488 |
| Mean Feature Variance | 154.667328 |
| Missing Values Count | 0.000000 |
| Missing Values Pct | 0.000000 |
| Numeric Features Count | 58.000000 |
| Sample Count | 1204.000000 |
| Target Max | 0.010920 |
| Target Mean | 0.000094 |
| Target Min | 0.000000 |
| Target Range | 0.010920 |
| Target Std | 0.000993 |

---

## 🧮 Model Complexity Metrics

| Metric | Value |
|--------|-------|
| Catboost | {'num_trees': 639, 'depth': 5} |
| Lightgbm | {'num_trees': 1, 'num_leaves': 32, 'max_depth': 5} |

---

## 📊 Prediction Statistics

| Statistic | Value |
|-----------|-------|
| Prediction Kurtosis | 2.341656 |
| Prediction Max | 0.000241 |
| Prediction Mean | 0.000034 |
| Prediction Median | 0.000005 |
| Prediction Min | -0.000004 |
| Prediction Skewness | 1.583175 |
| Prediction Std | 0.000042 |

---

## ⚠️ Error Analysis

| Metric | Value |
|--------|-------|
| Avg Mae Rmse Ratio | 0.169262 |
| Catboost Mae Rmse Ratio | 0.186761 |
| Lightgbm Mae Rmse Ratio | 0.151762 |

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
| analyst_base_metrics_report | `outcomes/analyst_base_ETHUSDT_15m_long_report_20251121_004605.md` |
| analyst_base_predictions | `versioned_artifacts/ETHUSDT_binance_15m_long_Analyst/analyst_base_predictions_20251121_004605_435.h5` |
| analyst_base_predictions_oof | `versioned_artifacts/ETHUSDT_binance_15m_long_Analyst/analyst_base_predictions_oof_20251121_004605_460.h5` |
| ml_scored_historical_data_oos | `ml_scored_historical_data_analyst_long_oos` |
| training_report_json | `outcomes/analyst_base_ETHUSDT_15m_long_metrics_20251121_004605.json` |
| training_report_markdown | `outcomes/analyst_base_ETHUSDT_15m_long_report_20251121_004605.md` |

---

*Comprehensive report generated by Ares Unified Training Pipeline v3.0 on 20251121_004605*
*Training Type: ANALYST_BASE | Symbol: ETHUSDT | Timeframe: 15m | Direction: long*
