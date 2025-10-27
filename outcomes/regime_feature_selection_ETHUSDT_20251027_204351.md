# Economic Regime Feature Selection Report

**Generated**: 2025-10-27 20:43:51  
**Symbol**: ETHUSDT  
**Exchange**: binance  
**Timeframe**: 1h  

**Multi-Target Approach**: ✅ Enabled  
**Targets**: close_return, volume_log_return, price_range_pct, body_size_pct, volume_return, close_log_return, price_range, trades, volatility_20, cmf  
**Target Weights**: close_return: 8.0%, volume_log_return: 5.0%, price_range_pct: 30.0%, body_size_pct: 5.0%, volume_return: 5.0%, close_log_return: 5.0%, price_range: 25.0%, trades: 0.0%, volatility_20: 20.0%, cmf: 2.0%  

---


## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Selected Features** | 25 |
| **Regime Transition Features** | 0 |
| **Economic Distinctiveness** | 0.000 |
| **Overall Validation Score** | 0.000 |
| **Silhouette Score** | 0.000 |
| **Noise Ratio** | 0.0% |
| **Execution Time** | 81.77s |

---

## 🏷️ Feature Categories Breakdown

| Category | Count | Description |
|----------|-------|-------------|
| **Structured Features** | 0 | Features with clear regime transitions (score ≥ 0.8) |
| **Random Features** | 0 | Features with moderate transitions (0.4 ≤ score < 0.8) |
| **Slightly Varying** | 0 | Features with small variations (0.1 ≤ score < 0.4) |
| **Constant Features** | 0 | Features with minimal changes (score < 0.1) |

---

## 📈 Per-Feature Transition Scores

### Structured Features (High Regime Transition Detection)
No features in this category.

### Random Features (Moderate Regime Transition Detection)
No features in this category.

### Slightly Varying Features (Low Regime Transition Detection)
No features in this category.

### Constant Features (Minimal Regime Transition Detection)
No features in this category.

---

## 🎯 Selected Features

### Regime Transition Features
These features are particularly good at detecting regime transitions and clear regime boundaries:

No regime transition features identified.

### Top Features by Economic Score

| Rank | Feature Name | Category | Economic Score | Regime Disc. | Clustering | Stability | Composite |
|------|--------------|----------|----------------|--------------|------------|-----------|-----------|
| 1 | simple_returns_1_price_returns | returns | 0.079 | 0.985 | 0.000 | 0.991 | 0.402 |
| 2 | log_returns_10_price_returns | returns | 0.028 | 0.954 | 0.000 | 0.990 | 0.396 |
| 3 | log_returns_1_price_returns | returns | 0.106 | 0.967 | 0.000 | 0.993 | 0.423 |
| 4 | log_returns_5_price_returns | returns | 0.071 | 0.960 | 0.000 | 0.992 | 0.410 |
| 5 | cumulative_returns_10_price_returns | returns | 0.050 | 0.953 | 0.000 | 0.991 | 0.382 |
| 6 | simple_returns_10_price_returns | returns | 0.022 | 0.984 | 0.000 | 0.992 | 0.393 |
| 7 | simple_returns_5_price_returns | returns | 0.051 | 0.977 | 0.000 | 0.991 | 0.392 |
| 8 | rolling_returns_10_price_returns | returns | 0.050 | 0.951 | 0.000 | 0.991 | 0.382 |
| 9 | returns_volatility_20_price_returns | volatility | 0.060 | 0.952 | 0.000 | 0.976 | 0.376 |
| 10 | rolling_returns_20_price_returns | returns | 0.094 | 0.968 | 0.000 | 0.992 | 0.402 |
| 11 | sharpe_ratio_20_0.0_price_returns | returns | 0.135 | 0.937 | 0.000 | 0.960 | 0.421 |
| 12 | advanced_cumulative_returns_10 | returns | 0.222 | 0.965 | 0.000 | 0.970 | 0.453 |
| 13 | cumulative_returns_20_price_returns | returns | 0.094 | 0.970 | 0.000 | 0.992 | 0.402 |
| 14 | returns_kurtosis_20_price_returns | returns | 0.123 | 0.924 | 0.000 | 0.973 | 0.412 |
| 15 | returns_skewness_20_price_returns | returns | 0.078 | 0.939 | 0.000 | 0.957 | 0.404 |
| 16 | rolling_zscore_returns_20 | returns | 0.183 | 0.954 | 0.013 | 0.992 | 0.445 |
| 17 | advanced_cumulative_returns_20 | returns | 0.227 | 0.945 | 0.000 | 0.962 | 0.450 |
| 18 | momentum_features | momentum | 0.220 | 0.990 | 0.000 | 0.960 | 0.460 |
| 19 | ar_1_coefficients_20 | entropy | 0.043 | 0.924 | 0.000 | 0.965 | 0.390 |
| 20 | ljung_box_pvalue_20_10 | other | 0.023 | 0.904 | 0.000 | 0.960 | 0.379 |


---

## 📈 Economic Metrics by Regime

| Regime ID | Sharpe Ratio | Sortino Ratio | Calmar Ratio | Win Rate | Avg Return | Volatility | Max DD | Samples |
|-----------|--------------|---------------|--------------|----------|------------|------------|--------|---------|


---

## 🔍 Clustering Quality Metrics

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| **Silhouette Score** | 0.000 | > 0.2 | ❌ FAIL |
| **Calinski-Harabasz** | 0.0 | > 100 | ❌ FAIL |
| **Davies-Bouldin** | 0.000 | < 2.0 | ✅ PASS |
| **Noise Ratio** | 0.0% | < 30% | ✅ PASS |

---

## 📊 Feature Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| volume | 11 | 44.0% |
| returns | 8 | 32.0% |
| momentum | 3 | 12.0% |
| volatility | 3 | 12.0% |


---

## ⚡ Computational Performance

| Metric | Value |
|--------|-------|
| **Execution Time** | 81.77s |
| **VectorBT Available** | ✅ YES |
| **Hardware Optimization** | ✅ YES |
| **Cheap Proxies Enabled** | ✅ YES |
| **CV Folds** | 3 |
| **Silhouette Sample Ratio** | 10.0% |

---

## 💡 Recommendations

### Feature Selection Quality
- **Overall Score**: 0.000 (POOR)

### Economic Distinctiveness
- **Sharpe Variance**: 0.000 (POOR)

### Clustering Quality
- **Silhouette Score**: 0.000 (POOR)

### Next Steps
1. **Proceed with HDBSCAN**: Selected features show moderate economic distinctiveness
2. **Monitor Performance**: Track regime-specific Sharpe ratios during clustering
3. **Feature Refinement**: Consider current selection is optimal

---

## 🔧 Configuration Used

- **Target Feature Count**: 25
- **Economic Significance Weight**: 30.0%
- **Regime Discrimination Weight**: 25.0%
- **Clustering Quality Weight**: 15.0%
- **Stability Weight**: 10.0%
- **Excluded Categories**: microstructure, support_resistance
- **CV Folds**: 3
- **Silhouette Sample Ratio**: 10.0%

---

*Report generated by Economic Regime Feature Selector v1.0*
