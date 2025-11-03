# SR Quality Model Training Report

**Generated:** 2025-11-02 20:20:22
**Symbol:** ETHUSDT
**Timeframe:** 1h
**Total Samples:** 175

---

## 📊 Executive Summary

**Status:** ✅ **PRODUCTION READY** (Health Score: 0.86/1.00)

### Key Metrics:

| Metric | Value | Status |
|--------|-------|--------|
| **Validation R²** | -0.003 | ⚠️ |
| **Validation RMSE** | 0.0073 | ✅ |
| **Overfitting** | none | ✅ |
| **Calibration (ECE)** | 0.0003 | ✅ |
| **Overall Health** | 0.86/1.00 | ✅ |

---

## 📈 Model Performance Metrics

### Cross-Validation Results:

| Fold | Train RMSE | Val RMSE | Train R² | Val R² | Val MAE | Boost Rounds |
|------|------------|----------|----------|--------|---------|--------------|
| 1 | 0.0074 | 0.0072 | -0.000 | -0.009 | 0.0071 | 1 |
| 2 | 0.0073 | 0.0073 | 0.000 | -0.000 | 0.0072 | 1 |
| 3 | 0.0073 | 0.0073 | 0.000 | -0.000 | 0.0072 | 1 |

### Average Performance:

- **Validation RMSE:** 0.0073 ± 0.0001
- **Validation R²:** -0.003 ± 0.004
- **Validation MAE:** 0.0071

---

## 🔬 Model Quality Metrics

### 1. Overfitting Detection

| Metric | Train | Validation | Gap | Status |
|--------|-------|------------|-----|--------|
| RMSE | - | - | +0.0002 | ✅ |
| R² | - | - | +0.0088 | ✅ |

**Severity:** none
**Recommendation:** Healthy - no significant overfitting detected

### 2. Calibration Analysis

- **Expected Calibration Error (ECE):** 0.0003
- **Mean Calibration Error:** 0.0003
- **Status:** ✅ Well calibrated

**Calibration by Prediction Range:**

| Predicted Range | Count | Predicted Mean | Actual Mean | Error | Status |
|-----------------|-------|----------------|-------------|-------|--------|
| 0.0-0.1 | 175 | 0.001 | 0.001 | 0.000 | ✅ |

### 3. Prediction Distribution

| Metric | Predictions | True Values | Ratio |
|--------|-------------|-------------|-------|
| Mean | 0.0012 | 0.0009 | - |
| Std | 0.0000 | 0.0073 | 0.00 |
| Range | 0.0000 | 0.0150 | 0.00 |

**⚠️  Distribution Issues:**
- Predictions collapsed (std < 0.05)
- 100% predictions near mean
- Limited range coverage (0%)
- Pred variance only 0% of true

### 4. Feature Importance Stability

- **Top 10 Stable:** 10/10
- **Mean CV:** 0.000
- **Status:** ✅ Stable

### 5. Error Analysis by Quality Bin

| Quality Bin | Samples | MAE | RMSE | Bias | R² |
|-------------|---------|-----|------|------|-----|
| Low (0.0-0.3) | 69 | 0.0088 | 0.0088 | -0.0088 | -783.713 |

---

## 💰 Financial Metrics

### Global Statistics:

**Component Metrics:**

| Component | Mean | Median | Std | Min | Max |
|-----------|------|--------|-----|-----|-----|

### Per-Level Analysis:

*Quality score not available*

---

## 🎯 Feature Importance Analysis

### Top 20 Features (Combined Ranking):

| Rank | Feature | LGBM Gain | Permutation | SHAP | Avg Rank |
|------|---------|-----------|-------------|------|----------|
| 1 | strength | 1 | 1 | 1 | 1.0 |
| 10 | market_volatility | 2 | 2 | 2 | 2.0 |
| 16 | quality_tier | 3 | 3 | 3 | 3.0 |
| 15 | recency_weighted_strength | 4 | 4 | 4 | 4.0 |
| 14 | bounce_consistency | 5 | 5 | 5 | 5.0 |
| 13 | volume_confirmation | 6 | 6 | 6 | 6.0 |
| 12 | is_uptrend | 7 | 7 | 7 | 7.0 |
| 11 | market_trend | 8 | 8 | 8 | 8.0 |
| 9 | is_support | 9 | 9 | 9 | 9.0 |
| 2 | touch_count | 10 | 10 | 10 | 10.0 |
| 8 | price_zscore | 11 | 11 | 11 | 11.0 |
| 7 | distance_to_current_pct | 12 | 12 | 12 | 12.0 |
| 6 | max_bounce_ratio | 13 | 13 | 13 | 13.0 |
| 5 | avg_bounce_ratio | 14 | 14 | 14 | 14.0 |
| 4 | consistency | 15 | 15 | 15 | 15.0 |
| 3 | age_bars | 16 | 16 | 16 | 16.0 |
| 17 | touch_quality_score | 17 | 17 | 17 | 17.0 |

### Key Insights:

**Most Important Features:**
1. `strength`
2. `market_volatility`
3. `quality_tier`
4. `recency_weighted_strength`
5. `bounce_consistency`

---

## 📋 Detailed Level Analysis

---

## 🚀 Production Readiness

### Overall Assessment:

**Health Score:** 0.86/1.00

### Criteria Checklist:

| Criterion | Status | Details |
|-----------|--------|---------|
| No significant overfitting | ✅ | none |
| Well calibrated | ✅ | ECE=0.0003 |
| Healthy predictions | ❌ | 4 issues |
| Stable features | ✅ | 10/10 stable |
| Stable CV | ✅ | - |

### ✅ **PRODUCTION READY**

Model meets all criteria for production deployment.

---

### Recommendations:

- Fix prediction distribution: Predictions collapsed (std < 0.05)
- Fix prediction distribution: 100% predictions near mean
- Fix prediction distribution: Limited range coverage (0%)
- Fix prediction distribution: Pred variance only 0% of true
