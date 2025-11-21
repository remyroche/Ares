# Analyst Base - Comprehensive Training Report

**Generated:** 2025-11-20 23:27:22 UTC

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
| Avg Generalization Score | -1.088197 |
| Avg Overfitting Ratio | 1.588197 |
| Generalization Score | -1.088197 |
| Overfitting Ratio | 1.588197 |
| Std Generalization Score | 1.076260 |
| Std Overfitting Ratio | 0.576260 |
| Train Test R2 Gap | 0.443437 |

---

## 📈 Split-Based Performance Metrics

### Training Set Metrics

| Metric | Value |
|--------|-------|
| Mae | 0.000007 |
| Mse | 0.000000 |
| R2 | 0.427512 |
| Rmse | 0.000054 |

### Validation Set Metrics

| Metric | Value |
|--------|-------|
| Mae | 0.000387 |
| Mse | 0.000004 |
| R2 | -0.029688 |
| Rmse | 0.001984 |

### Test Set Metrics

| Metric | Value |
|--------|-------|
| Mae | 0.000274 |
| Mse | 0.000003 |
| R2 | -0.015926 |
| Rmse | 0.001618 |

---

## 📚 Learnability & Generalization Diagnostics

This section summarizes how the model learns from data and how robustly it generalizes.

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| R² | 0.4275 | -0.0297 | -0.0159 |

### Overfitting & Generalization Indicators

- **Train–Test R² Gap:** 0.4434  \n  Larger gaps indicate that the model fits the training data much better than unseen data.
- **Overfitting Ratio:** 1.5882  \n  Approximate relative gap between train and test performance → **high (risk of overfitting)**.
- **Generalization Score:** -1.0882  \n  Ratio of test to train performance; values near 1.0 indicate similar train/test behaviour.

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
| Best Iteration | 398.000000 |
| Generalization Score | -0.011937 |
| Iterations Used | 398.000000 |
| Mae | 0.000302 |
| Mse | 0.000003 |
| Overfitting Ratio | 1.011937 |
| R2 | -0.010207 |
| Rmse | 0.001613 |
| Test Mae | 0.000302 |
| Test Mse | 0.000003 |
| Test R2 | -0.010207 |
| Test Rmse | 0.001613 |
| Train Mae | 0.000005 |
| Train Mse | 0.000000 |
| Train R2 | 0.855023 |
| Train Rmse | 0.000030 |
| Train Test R2 Gap | 0.865230 |
| Val Mae | 0.000406 |
| Val Mse | 0.000004 |
| Val R2 | -0.026059 |
| Val Rmse | 0.001980 |

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
| 1 | lightgbm_regime_0_prob | 7.724911 |
| 2 | candlestick_harami_cross_pattern_vwap_3x_ratio | 4.443788 |
| 3 | volume_price_trend_vwap | 2.967592 |
| 4 | extratrees_regime_1_prob | 2.837232 |
| 5 | shannon_entropy_20_10 | 2.803165 |
| 6 | vectorbt_trend_consistency_5_price_returns | 2.426332 |
| 7 | vectorbt_parkinson_volatility_50_vwap_27x_ratio_log_ratio_candlestick_engulfing_pattern_base_9x_ratio | 2.370744 |
| 8 | volume_price_trend_vwap_div_candlestick_piercing_line_pattern_vwap_3x_ratio | 2.305979 |
| 9 | resistance_level_1_5_price_returns | 2.229306 |
| 10 | volume_roc_5 | 1.618649 |
| 11 | vectorbt_parabolic_sar_0.1_0.3 | 1.481876 |
| 12 | lightgbm_regime_4_prob | 1.219410 |
| 13 | candlestick_harami_cross_pattern_vwap_3x_ratio_log_ratio_vectorbt_rogers_satchell_volatility_30_vwap_x_3x | 1.043556 |
| 14 | vectorbt_trend_consistency_10_price_returns | 0.979363 |
| 15 | ultimate_oscillator_7_14_28_returns_vwap | 0.843742 |
| 16 | vectorbt_zigzag_3.0_2 | 0.813555 |
| 17 | volume_roc_1 | 0.700994 |
| 18 | vectorbt_zigzag_5.0_2 | 0.643524 |
| 19 | acceleration_features | 0.614193 |
| 20 | volume_price_trend_vwap_log_ratio_vectorbt_rogers_satchell_volatility_30_vwap_x_3x | 0.598192 |

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
| Target Max | 0.010920 |
| Target Mean | 0.000094 |
| Target Min | 0.000000 |
| Target Range | 0.010920 |
| Target Std | 0.000993 |

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
| Prediction Kurtosis | 3.142478 |
| Prediction Max | 0.000214 |
| Prediction Mean | 0.000035 |
| Prediction Median | 0.000012 |
| Prediction Min | 0.000005 |
| Prediction Skewness | 1.338989 |
| Prediction Std | 0.000035 |

---

## ⚠️ Error Analysis

| Metric | Value |
|--------|-------|
| Avg Mae Rmse Ratio | 0.169442 |
| Catboost Mae Rmse Ratio | 0.187121 |
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
| analyst_base_metrics_report | `outcomes/analyst_base_ETHUSDT_15m_long_report_20251120_232722.md` |
| analyst_base_predictions | `versioned_artifacts/ETHUSDT_binance_15m_long_Analyst/analyst_base_predictions_20251120_232722_000.h5` |
| analyst_base_predictions_oof | `versioned_artifacts/ETHUSDT_binance_15m_long_Analyst/analyst_base_predictions_oof_20251120_232722_018.h5` |
| ml_scored_historical_data_oos | `ml_scored_historical_data_analyst_long_oos` |
| training_report_json | `outcomes/analyst_base_ETHUSDT_15m_long_metrics_20251120_232722.json` |
| training_report_markdown | `outcomes/analyst_base_ETHUSDT_15m_long_report_20251120_232722.md` |

---

*Comprehensive report generated by Ares Unified Training Pipeline v3.0 on 20251120_232722*
*Training Type: ANALYST_BASE | Symbol: ETHUSDT | Timeframe: 15m | Direction: long*
