# 🎯 Top 30 Features - Economic Regime Feature Selection

**Generated**: 2025-10-26 15:03:00  
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
| **Selected Features** | 15 |
| **Execution Time** | 448.63s |
| **VectorBT Available** | ✅ YES |

---

## 🎯 Top 30 Features by Composite Score

| Rank | Feature Name | Category | Economic Score | Regime Disc. | Clustering | Stability | Composite | Selected |
|------|--------------|----------|----------------|--------------|------------|-----------|-----------|----------|
| 186 | vectorbt_jerk_5_price_returns | returns | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 35 | roc_21_price_returns | returns | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 174 | volume_entropy_20_volume_returns | returns | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 168 | volume_entropy_5_volume_returns | returns | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 195 | vectorbt_trend_strength_50_price_returns | returns | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 182 | vectorbt_momentum_10_price_returns | returns | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 171 | volume_entropy_10_volume_returns | returns | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 204 | vectorbt_acceleration_regime_5_10_price_returns | returns | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 167 | macd_entropy_20_12_26 | momentum | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 18 | momentum_features | momentum | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 185 | vectorbt_acceleration_5_price_returns | returns | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 184 | vectorbt_momentum_50_price_returns | returns | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 57 | volume_ratio_50 | volume | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 180 | vectorbt_momentum_5_price_returns | returns | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 107 | trend_score_14 | trend | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 108 | directional_signal | other | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 30 | roc_14_price_returns | returns | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 53 | volume_ratio_10 | volume | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 190 | vectorbt_trend_strength_5_price_returns | returns | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 183 | vectorbt_momentum_20_price_returns | returns | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 55 | volume_ratio_20 | volume | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 194 | vectorbt_trend_strength_20_price_returns | returns | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 220 | acceleration_features | interaction | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 188 | vectorbt_jerk_10_price_returns | returns | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 40 | roc_30_price_returns | returns | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 187 | vectorbt_acceleration_10_price_returns | returns | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 75 | analyst_volume_pressure | volume | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 43 | macd_delta_12_26_9 | momentum | 0.300 | 0.998 | 0.000 | 0.500 | 0.469 | ❌ |
| 189 | vectorbt_trend_consistency_5_price_returns | returns | 0.300 | 0.997 | 0.000 | 0.500 | 0.469 | ❌ |
| 192 | vectorbt_trend_strength_10_price_returns | returns | 0.300 | 0.997 | 0.000 | 0.500 | 0.469 | ❌ |

---

## 📈 Category Distribution (Top 30)

- **returns**: 20 features (66.7%)
- **volume**: 4 features (13.3%)
- **momentum**: 3 features (10.0%)
- **trend**: 1 features (3.3%)
- **other**: 1 features (3.3%)
- **interaction**: 1 features (3.3%)


---

## 🏆 Selected Features Analysis

### ✅ Selected Features (15 total)
- **simple_returns_1_price_returns** (returns) - Score: 0.468
- **log_returns_1_price_returns** (returns) - Score: 0.468
- **simple_returns_5_price_returns** (returns) - Score: 0.466
- **log_returns_5_price_returns** (returns) - Score: 0.466
- **advanced_cumulative_returns_10** (returns) - Score: 0.463
- **cumulative_returns_10_price_returns** (returns) - Score: 0.463
- **simple_returns_10_price_returns** (returns) - Score: 0.463
- **log_returns_10_price_returns** (returns) - Score: 0.463
- **rolling_returns_10_price_returns** (returns) - Score: 0.463
- **returns_volatility_20_price_returns** (returns) - Score: 0.457
- **sharpe_ratio_20_0.0_price_returns** (returns) - Score: 0.457
- **cumulative_returns_20_price_returns** (returns) - Score: 0.457
- **returns_skewness_20_price_returns** (returns) - Score: 0.457
- **returns_kurtosis_20_price_returns** (returns) - Score: 0.457
- **rolling_returns_20_price_returns** (returns) - Score: 0.456


### 📊 Score Statistics (Top 30)

| Metric | Min | Max | Mean | Std |
|--------|-----|-----|------|-----|
| **Economic Significance** | 0.300 | 0.300 | 0.300 | 0.000 |
| **Regime Discrimination** | 0.997 | 0.998 | 0.998 | 0.000 |
| **Clustering Quality** | 0.000 | 0.000 | 0.000 | 0.000 |
| **Stability Score** | 0.500 | 0.500 | 0.500 | 0.000 |
| **Composite Score** | 0.469 | 0.469 | 0.469 | 0.000 |

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
