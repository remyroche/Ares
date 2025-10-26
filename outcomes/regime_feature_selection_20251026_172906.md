# 🎯 Top 30 Features - Economic Regime Feature Selection

**Generated**: 2025-10-26 17:29:06  
**Symbol**: ETHUSDT  
**Exchange**: binance  
**Timeframe**: 1h  
**Configuration**: Enhanced 8-Target Multi-Target Approach

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Total Features Analyzed** | 221 |
| **Top 30 Features** | 30 |
| **Selected Features** | 25 |
| **Execution Time** | 507.22s |
| **VectorBT Available** | ✅ YES |

---

## 🎯 Top 30 Features by Composite Score

| Rank | Feature Name | Category | Economic Score | Regime Disc. | Clustering | Stability | Composite | Selected |
|------|--------------|----------|----------------|--------------|------------|-----------|-----------|----------|
| 105 | tema_21_price_returns | returns | 0.297 | 0.996 | 0.000 | 0.969 | 0.496 | ❌ |
| 102 | dema_21_price_returns | returns | 0.297 | 0.996 | 0.000 | 0.967 | 0.494 | ❌ |
| 57 | volume_ratio_50 | volume | 0.296 | 0.998 | 0.000 | 0.979 | 0.493 | ✅ |
| 39 | momentum_30_price_returns | returns | 0.297 | 0.996 | 0.000 | 0.991 | 0.493 | ✅ |
| 118 | support_level_1_5_price_returns | returns | 0.296 | 0.987 | 0.000 | 0.977 | 0.493 | ❌ |
| 117 | support_level_2_5_price_returns | returns | 0.296 | 0.987 | 0.000 | 0.977 | 0.493 | ❌ |
| 119 | support_level_5_5_price_returns | returns | 0.296 | 0.987 | 0.000 | 0.977 | 0.493 | ❌ |
| 120 | support_level_3_5_price_returns | returns | 0.296 | 0.987 | 0.000 | 0.977 | 0.493 | ❌ |
| 122 | support_level_4_5_price_returns | returns | 0.296 | 0.987 | 0.000 | 0.977 | 0.493 | ❌ |
| 16 | rolling_zscore_returns_20 | returns | 0.297 | 0.957 | 0.000 | 0.992 | 0.492 | ✅ |
| 220 | acceleration_features | interaction | 0.297 | 0.998 | 0.000 | 0.997 | 0.491 | ✅ |
| 29 | momentum_14_price_returns | returns | 0.296 | 0.994 | 0.000 | 0.993 | 0.491 | ✅ |
| 75 | analyst_volume_pressure | volume | 0.294 | 0.987 | 0.000 | 0.992 | 0.490 | ✅ |
| 146 | fibonacci_0.236_5_price_returns | returns | 0.297 | 0.985 | 0.000 | 0.979 | 0.490 | ❌ |
| 103 | mama_21_0.05_price_returns | returns | 0.297 | 0.996 | 0.000 | 0.968 | 0.490 | ✅ |
| 104 | keltner_channels_20_14_price_returns | returns | 0.297 | 0.996 | 0.000 | 0.968 | 0.490 | ❌ |
| 51 | volume_sma_5 | volume | 0.297 | 0.988 | 0.000 | 0.976 | 0.488 | ✅ |
| 52 | volume_ema_5 | volume | 0.297 | 0.988 | 0.000 | 0.976 | 0.488 | ✅ |
| 31 | momentum_21_price_returns | returns | 0.297 | 0.992 | 0.000 | 0.991 | 0.488 | ✅ |
| 148 | fibonacci_0.382_5_price_returns | returns | 0.297 | 0.987 | 0.000 | 0.977 | 0.487 | ❌ |
| 127 | pivot_point_5_price_returns | returns | 0.297 | 0.988 | 0.000 | 0.976 | 0.486 | ✅ |
| 27 | williams_r_14_price_returns | returns | 0.296 | 0.963 | 0.000 | 0.989 | 0.485 | ❌ |
| 26 | stochastic_14_3_price_returns | returns | 0.296 | 0.963 | 0.000 | 0.989 | 0.485 | ❌ |
| 70 | volume_price_divergence_10 | volume | 0.297 | 0.982 | 0.000 | 0.985 | 0.483 | ✅ |
| 55 | volume_ratio_20 | volume | 0.297 | 0.998 | 0.000 | 0.978 | 0.483 | ❌ |
| 90 | vectorbt_bbands_10_1.5 | other | 0.297 | 0.973 | 0.000 | 0.977 | 0.481 | ✅ |
| 89 | vectorbt_bbands_10_2.5 | other | 0.297 | 0.973 | 0.000 | 0.977 | 0.481 | ❌ |
| 88 | vectorbt_bbands_10_2.0 | other | 0.297 | 0.973 | 0.000 | 0.977 | 0.481 | ❌ |
| 32 | stochastic_21_3_price_returns | returns | 0.297 | 0.949 | 0.000 | 0.986 | 0.480 | ❌ |
| 33 | williams_r_21_price_returns | returns | 0.297 | 0.949 | 0.000 | 0.986 | 0.480 | ❌ |

---

## 📈 Category Distribution (Top 30)

- **returns**: 20 features (66.7%)
- **volume**: 6 features (20.0%)
- **other**: 3 features (10.0%)
- **interaction**: 1 features (3.3%)


---

## 🏆 Selected Features Analysis

### ✅ Selected Features (25 total)
- **volume_ratio_50** (volume) - Score: 0.493
- **momentum_30_price_returns** (returns) - Score: 0.493
- **rolling_zscore_returns_20** (returns) - Score: 0.492
- **acceleration_features** (interaction) - Score: 0.491
- **momentum_14_price_returns** (returns) - Score: 0.491
- **analyst_volume_pressure** (volume) - Score: 0.490
- **mama_21_0.05_price_returns** (returns) - Score: 0.490
- **volume_ema_5** (volume) - Score: 0.488
- **volume_sma_5** (volume) - Score: 0.488
- **momentum_21_price_returns** (returns) - Score: 0.488
- **pivot_point_5_price_returns** (returns) - Score: 0.486
- **volume_price_divergence_10** (volume) - Score: 0.483
- **vectorbt_bbands_10_1.5** (other) - Score: 0.481
- **vectorbt_trend_strength_50_price_returns** (returns) - Score: 0.480
- **volume_roc_1** (volume) - Score: 0.479
- **volume_ratio_10** (volume) - Score: 0.478
- **enhanced_volatility_10** (volume) - Score: 0.477
- **fibonacci_0.5_5_price_returns** (returns) - Score: 0.477
- **vectorbt_volatility_comprehensive_10** (volume) - Score: 0.476
- **volume_price_divergence_20** (volume) - Score: 0.476
- **vectorbt_trend_consistency_10_price_returns** (returns) - Score: 0.476
- **macd_12_26_9_returns_vwap** (returns) - Score: 0.472
- **volume_momentum_10** (momentum) - Score: 0.469
- **volume_momentum_20** (momentum) - Score: 0.469
- **sma_20_returns_vwap** (returns) - Score: 0.468


### 📊 Score Statistics (Top 30)

| Metric | Min | Max | Mean | Std |
|--------|-----|-----|------|-----|
| **Economic Significance** | 0.294 | 0.297 | 0.297 | 0.001 |
| **Regime Discrimination** | 0.949 | 0.998 | 0.983 | 0.015 |
| **Clustering Quality** | 0.000 | 0.000 | 0.000 | 0.000 |
| **Stability Score** | 0.967 | 0.997 | 0.981 | 0.008 |
| **Composite Score** | 0.480 | 0.496 | 0.488 | 0.005 |

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
