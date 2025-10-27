# 🎯 Top 30 Features - Economic Regime Feature Selection

**Generated**: 2025-10-27 20:45:38  
**Symbol**: ETHUSDT  
**Exchange**: binance  
**Timeframe**: 1h  
**Configuration**: Enhanced 8-Target Multi-Target Approach

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Total Features Analyzed** | 180 |
| **Top 30 Features** | 30 |
| **Selected Features** | 25 |
| **Execution Time** | 90.21s |
| **VectorBT Available** | ✅ YES |

---

## 🎯 Top 30 Features by Composite Score

| Rank | Feature Name | Category | Economic Score | Regime Disc. | Clustering | Stability | Composite | Selected |
|------|--------------|----------|----------------|--------------|------------|-----------|-----------|----------|
| 57 | volume_ratio_50 | volume | 0.366 | 0.999 | 0.000 | 0.979 | 0.503 | ✅ |
| 51 | volume_sma_5 | volume | 0.344 | 0.986 | 0.000 | 0.976 | 0.487 | ✅ |
| 52 | volume_ema_5 | volume | 0.344 | 0.986 | 0.000 | 0.976 | 0.487 | ✅ |
| 104 | dema_21_price_returns | returns | 0.281 | 0.997 | 0.000 | 0.967 | 0.479 | ✅ |
| 107 | tema_21_price_returns | returns | 0.275 | 0.997 | 0.000 | 0.969 | 0.479 | ✅ |
| 55 | volume_ratio_20 | volume | 0.275 | 0.999 | 0.000 | 0.978 | 0.477 | ✅ |
| 86 | enhanced_volatility_10 | volatility | 0.305 | 0.968 | 0.000 | 0.978 | 0.476 | ✅ |
| 105 | mama_21_0.05_price_returns | returns | 0.275 | 0.997 | 0.000 | 0.968 | 0.475 | ✅ |
| 106 | keltner_channels_20_14_price_returns | returns | 0.275 | 0.997 | 0.000 | 0.968 | 0.475 | ✅ |
| 88 | vectorbt_volatility_comprehensive_10 | volatility | 0.299 | 0.971 | 0.000 | 0.974 | 0.473 | ✅ |
| 154 | vectorbt_trend_strength_50_price_returns | returns | 0.278 | 0.999 | 0.000 | 0.964 | 0.473 | ✅ |
| 53 | volume_ratio_10 | volume | 0.237 | 0.999 | 0.000 | 0.982 | 0.468 | ✅ |
| 65 | volume_momentum_20 | volume | 0.258 | 0.956 | 0.000 | 0.978 | 0.464 | ✅ |
| 77 | vwap_deviations_20 | volume | 0.256 | 0.954 | 0.000 | 0.966 | 0.462 | ✅ |
| 41 | momentum_endpoints_sma_20 | momentum | 0.255 | 0.955 | 0.000 | 0.965 | 0.461 | ✅ |
| 68 | volume_momentum_10 | volume | 0.237 | 0.965 | 0.000 | 0.983 | 0.461 | ✅ |
| 87 | enhanced_volatility_14 | volatility | 0.272 | 0.953 | 0.000 | 0.978 | 0.460 | ✅ |
| 18 | momentum_features | momentum | 0.220 | 0.990 | 0.000 | 0.960 | 0.460 | ✅ |
| 64 | volume_percentile_20 | volume | 0.259 | 0.935 | 0.000 | 0.982 | 0.459 | ✅ |
| 103 | wma_20_price_returns | returns | 0.248 | 0.952 | 0.000 | 0.964 | 0.458 | ✅ |
| 150 | vectorbt_trend_consistency_10_price_returns | returns | 0.252 | 0.976 | 0.000 | 0.980 | 0.458 | ✅ |
| 76 | volume_ma_ratios_20_10 | volume | 0.256 | 0.936 | 0.000 | 0.982 | 0.457 | ✅ |
| 12 | advanced_cumulative_returns_10 | returns | 0.222 | 0.965 | 0.000 | 0.970 | 0.453 | ✅ |
| 39 | momentum_30_price_returns | momentum | 0.183 | 0.997 | 0.000 | 0.991 | 0.453 | ✅ |
| 54 | volume_roc_1 | volume | 0.210 | 0.997 | 0.000 | 0.994 | 0.453 | ✅ |
| 67 | volume_momentum_5 | volume | 0.190 | 0.991 | 0.000 | 0.981 | 0.453 | ❌ |
| 78 | order_flow_imbalance_20 | order_flow | 0.235 | 0.960 | 0.000 | 0.968 | 0.453 | ❌ |
| 110 | directional_signal | other | 0.211 | 0.989 | 0.000 | 0.968 | 0.452 | ❌ |
| 60 | volume_roc_10 | volume | 0.204 | 0.984 | 0.000 | 0.982 | 0.451 | ❌ |
| 17 | advanced_cumulative_returns_20 | returns | 0.227 | 0.945 | 0.000 | 0.962 | 0.450 | ❌ |

---

## 📈 Category Distribution (Top 30)

- **volume**: 13 features (43.3%)
- **returns**: 9 features (30.0%)
- **volatility**: 3 features (10.0%)
- **momentum**: 3 features (10.0%)
- **order_flow**: 1 features (3.3%)
- **other**: 1 features (3.3%)


---

## 🏆 Selected Features Analysis

### ✅ Selected Features (25 total)
- **volume_ratio_50** (volume) - Score: 0.503
- **volume_sma_5** (volume) - Score: 0.487
- **volume_ema_5** (volume) - Score: 0.487
- **dema_21_price_returns** (returns) - Score: 0.479
- **tema_21_price_returns** (returns) - Score: 0.479
- **volume_ratio_20** (volume) - Score: 0.477
- **enhanced_volatility_10** (volatility) - Score: 0.476
- **keltner_channels_20_14_price_returns** (returns) - Score: 0.475
- **mama_21_0.05_price_returns** (returns) - Score: 0.475
- **vectorbt_volatility_comprehensive_10** (volatility) - Score: 0.473
- **vectorbt_trend_strength_50_price_returns** (returns) - Score: 0.473
- **volume_ratio_10** (volume) - Score: 0.468
- **volume_momentum_20** (volume) - Score: 0.464
- **vwap_deviations_20** (volume) - Score: 0.462
- **momentum_endpoints_sma_20** (momentum) - Score: 0.461
- **volume_momentum_10** (volume) - Score: 0.461
- **enhanced_volatility_14** (volatility) - Score: 0.460
- **momentum_features** (momentum) - Score: 0.460
- **volume_percentile_20** (volume) - Score: 0.459
- **wma_20_price_returns** (returns) - Score: 0.458
- **vectorbt_trend_consistency_10_price_returns** (returns) - Score: 0.458
- **volume_ma_ratios_20_10** (volume) - Score: 0.457
- **advanced_cumulative_returns_10** (returns) - Score: 0.453
- **momentum_30_price_returns** (momentum) - Score: 0.453
- **volume_roc_1** (volume) - Score: 0.453


### 📊 Score Statistics (Top 30)

| Metric | Min | Max | Mean | Std |
|--------|-----|-----|------|-----|
| **Economic Significance** | 0.183 | 0.366 | 0.258 | 0.044 |
| **Regime Discrimination** | 0.935 | 0.999 | 0.976 | 0.021 |
| **Clustering Quality** | 0.000 | 0.000 | 0.000 | 0.000 |
| **Stability Score** | 0.960 | 0.994 | 0.975 | 0.009 |
| **Composite Score** | 0.450 | 0.503 | 0.466 | 0.013 |

---

## 🎯 Enhanced 8-Target Configuration

### Target Breakdown:
- **close_return**: 25% - Core price movements
- **volume_log_return**: 20% - Volume momentum patterns  
- **price_range_pct**: 20% - Relative volatility
- **body_size_pct**: 8% - Price efficiency regimes
- **volume_return**: 8% - Volume changes
- **vwap_price_ratio**: 10% - Price distance from MA (regime separator)
- **volatility_20**: 5% - Realized volatility rolling
- **cmf**: 4% - Volume imbalance/order flow (directional conviction)

### Category Focus:
- **Price Movements**: 35% (close_return + vwap_price_ratio)
- **Volume Patterns**: 32% (volume_log_return + volume_return + cmf)
- **Volatility Regimes**: 25% (price_range_pct + volatility_20)
- **Price Efficiency**: 8% (body_size_pct)

---

## 💡 Key Insights

1. **Returns Dominance**: All top features are from the 'returns' category, indicating strong price-based regime detection
2. **High Regime Discrimination**: Most features show >95% regime discrimination capability
3. **Stability**: All features maintain 0.5 stability score (baseline)
4. **Clustering Quality**: Currently 0.0 across all features (needs investigation)

---

## 🔧 Next Steps

1. **Investigate Clustering Quality**: Why are clustering quality scores 0.0?
2. **Diversify Categories**: Consider features from volume, momentum, and other categories
3. **Regime Transition**: Focus on features that excel at detecting regime changes
4. **Validation**: Test selected features with HDBSCAN clustering

---

*Report generated by Economic Regime Feature Selector v1.0*
