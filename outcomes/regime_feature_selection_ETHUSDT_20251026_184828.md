# Economic Regime Feature Selection Report

**Generated**: 2025-10-26 18:48:28  
**Symbol**: ETHUSDT  
**Exchange**: binance  
**Timeframe**: 1h  

**Multi-Target Approach**: ✅ Enabled  
**Targets**: close_return, volume_log_return, price_range_pct, body_size_pct, volume_return, close_log_return, price_range, trades  
**Target Weights**: close_return: 18.0%, volume_log_return: 15.0%, price_range_pct: 18.0%, body_size_pct: 10.0%, volume_return: 10.0%, close_log_return: 15.0%, price_range: 12.0%, trades: 2.0%  

---


## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Selected Features** | 221 |
| **Regime Transition Features** | 221 |
| **Economic Distinctiveness** | 0.000 |
| **Overall Validation Score** | 0.295 |
| **Silhouette Score** | -0.016 |
| **Noise Ratio** | 0.0% |
| **Execution Time** | 127.83s |

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

| Rank | Feature Name | Category |
|------|--------------|----------|
| 1 | simple_returns_1_price_returns | returns |
| 2 | log_returns_10_price_returns | returns |
| 3 | log_returns_1_price_returns | returns |
| 4 | log_returns_5_price_returns | returns |
| 5 | cumulative_returns_10_price_returns | returns |
| 6 | simple_returns_10_price_returns | returns |
| 7 | simple_returns_5_price_returns | returns |
| 8 | rolling_returns_10_price_returns | returns |
| 9 | returns_volatility_20_price_returns | returns |
| 10 | rolling_returns_20_price_returns | returns |
| 11 | sharpe_ratio_20_0.0_price_returns | returns |
| 12 | advanced_cumulative_returns_10 | returns |
| 13 | cumulative_returns_20_price_returns | returns |
| 14 | returns_kurtosis_20_price_returns | returns |
| 15 | returns_skewness_20_price_returns | returns |
| 16 | rolling_zscore_returns_20 | returns |
| 17 | advanced_cumulative_returns_20 | returns |
| 18 | momentum_features | momentum |
| 19 | ar_1_coefficients_20 | entropy |
| 20 | ljung_box_pvalue_20_10 | other |
| 21 | vectorbt_momentum_comprehensive_21 | momentum |
| 22 | vectorbt_momentum_comprehensive_9 | momentum |
| 23 | vectorbt_momentum_comprehensive_14 | momentum |
| 24 | vectorbt_momentum_comprehensive_30 | momentum |
| 25 | macd_12_26_9_returns_vwap | returns |
| 26 | stochastic_14_3_price_returns | returns |
| 27 | williams_r_14_price_returns | returns |
| 28 | rsi_14_returns_vwap | returns |
| 29 | momentum_14_price_returns | returns |
| 30 | roc_14_price_returns | returns |
| 31 | momentum_21_price_returns | returns |
| 32 | stochastic_21_3_price_returns | returns |
| 33 | williams_r_21_price_returns | returns |
| 34 | rsi_21_returns_vwap | returns |
| 35 | roc_21_price_returns | returns |
| 36 | rsi_30_returns_vwap | returns |
| 37 | stochastic_30_3_price_returns | returns |
| 38 | williams_r_30_price_returns | returns |
| 39 | momentum_30_price_returns | returns |
| 40 | roc_30_price_returns | returns |
| 41 | momentum_endpoints_sma_20 | momentum |
| 42 | rsi_zscore_14_20 | momentum |
| 43 | macd_delta_12_26_9 | momentum |
| 44 | stochastic_kd_14_3 | oscillator |
| 45 | donchian_channel_20 | other |
| 46 | advanced_momentum_5_20 | momentum |
| 47 | analyst_momentum_5m | momentum |
| 48 | analyst_momentum_15m | momentum |
| 49 | advanced_momentum_10_30 | momentum |
| 50 | analyst_momentum_1h | momentum |
| 51 | volume_sma_5 | volume |
| 52 | volume_ema_5 | volume |
| 53 | volume_ratio_10 | volume |
| 54 | volume_roc_1 | volume |
| 55 | volume_ratio_20 | volume |
| 56 | volume_roc_5 | volume |
| 57 | volume_ratio_50 | volume |
| 58 | volume_std_10 | volume |
| 59 | volume_roc_20 | volume |
| 60 | volume_roc_10 | volume |
| 61 | volume_trend_strength_10_30 | volume |
| 62 | volume_trend_strength_20_50 | volume |
| 63 | volume_oscillator_10_20 | volume |
| 64 | volume_percentile_20 | volume |
| 65 | volume_momentum_20 | momentum |
| 66 | volume_oscillator_5_15 | volume |
| 67 | volume_momentum_5 | momentum |
| 68 | volume_momentum_10 | momentum |
| 69 | volume_price_correlation_20 | volume |
| 70 | volume_price_divergence_10 | volume |
| 71 | volume_price_correlation_10 | volume |
| 72 | volume_price_divergence_20 | volume |
| 73 | price_volume_oscillator_5_15 | volume |
| 74 | price_volume_oscillator_10_20 | volume |
| 75 | analyst_volume_pressure | volume |
| 76 | volume_ma_ratios_20_10 | volume |
| 77 | vwap_deviations_20 | volume |
| 78 | order_flow_imbalance_20 | order_flow |
| 79 | cmf_20 | other |
| 80 | vectorbt_enhanced_obv_20 | other |
| 81 | vectorbt_enhanced_ad_line_10 | other |
| 82 | vectorbt_enhanced_ad_line_20 | other |
| 83 | volume_volatility_elasticity_20 | volume |
| 84 | analyst_volume_trend | volume |
| 85 | enhanced_volatility_10 | volume |
| 86 | enhanced_volatility_14 | volume |
| 87 | vectorbt_volatility_comprehensive_10 | volume |
| 88 | vectorbt_bbands_10_2.0 | other |
| 89 | vectorbt_bbands_10_2.5 | other |
| 90 | vectorbt_bbands_10_1.5 | other |
| 91 | vectorbt_bbands_14_1.5 | other |
| 92 | vectorbt_bbands_14_2.0 | other |
| 93 | vectorbt_bbands_20_1.5 | other |
| 94 | vectorbt_bbands_20_2.0 | other |
| 95 | vectorbt_bbands_14_2.5 | other |
| 96 | vectorbt_bbands_20_2.5 | other |
| 97 | sma_5_returns_vwap | returns |
| 98 | sma_20_returns_vwap | returns |
| 99 | sma_10_returns_vwap | returns |
| 100 | ema_12_returns_vwap | returns |
| 101 | wma_20_price_returns | returns |
| 102 | dema_21_price_returns | returns |
| 103 | mama_21_0.05_price_returns | returns |
| 104 | keltner_channels_20_14_price_returns | returns |
| 105 | tema_21_price_returns | returns |
| 106 | vwma_20_price_returns | returns |
| 107 | trend_score_14 | trend |
| 108 | directional_signal | other |
| 109 | adx_14_returns_vwap | returns |
| 110 | cci_20_returns_vwap | returns |
| 111 | apo_12_26_returns_vwap | returns |
| 112 | cmo_14_returns_vwap | returns |
| 113 | aroon_25_returns_vwap | returns |
| 114 | pfe_12_returns_vwap | returns |
| 115 | natr_14_returns_vwap | returns |
| 116 | t3_14_0.7_returns_vwap | returns |
| 117 | support_level_2_5_price_returns | returns |
| 118 | support_level_1_5_price_returns | returns |
| 119 | support_level_5_5_price_returns | returns |
| 120 | support_level_3_5_price_returns | returns |
| 121 | resistance_level_1_5_price_returns | returns |
| 122 | support_level_4_5_price_returns | returns |
| 123 | resistance_level_2_5_price_returns | returns |
| 124 | resistance_level_5_5_price_returns | returns |
| 125 | resistance_level_4_5_price_returns | returns |
| 126 | resistance_level_3_5_price_returns | returns |
| 127 | pivot_point_5_price_returns | returns |
| 128 | support_level_1_10_price_returns | returns |
| 129 | support_level_2_10_price_returns | returns |
| 130 | support_level_4_10_price_returns | returns |
| 131 | support_level_3_10_price_returns | returns |
| 132 | resistance_level_1_10_price_returns | returns |
| 133 | resistance_level_2_10_price_returns | returns |
| 134 | support_level_5_10_price_returns | returns |
| 135 | resistance_level_3_10_price_returns | returns |
| 136 | resistance_level_4_10_price_returns | returns |
| 137 | resistance_level_5_10_price_returns | returns |
| 138 | support_level_1_20_price_returns | returns |
| 139 | support_level_2_20_price_returns | returns |
| 140 | pivot_point_10_price_returns | returns |
| 141 | support_level_5_20_price_returns | returns |
| 142 | support_level_4_20_price_returns | returns |
| 143 | support_level_3_20_price_returns | returns |
| 144 | pivot_point_20_price_returns | returns |
| 145 | fibonacci_0.236_10_price_returns | returns |
| 146 | fibonacci_0.236_5_price_returns | returns |
| 147 | fibonacci_0.382_10_price_returns | returns |
| 148 | fibonacci_0.382_5_price_returns | returns |
| 149 | fibonacci_0.236_20_price_returns | returns |
| 150 | fibonacci_0.382_20_price_returns | returns |
| 151 | fibonacci_0.5_5_price_returns | returns |
| 152 | fibonacci_0.618_5_price_returns | returns |
| 153 | fibonacci_0.5_20_price_returns | returns |
| 154 | fibonacci_0.5_10_price_returns | returns |
| 155 | fibonacci_0.786_10_price_returns | returns |
| 156 | fibonacci_0.786_5_price_returns | returns |
| 157 | fibonacci_0.618_10_price_returns | returns |
| 158 | fibonacci_0.618_20_price_returns | returns |
| 159 | fibonacci_0.786_20_price_returns | returns |
| 160 | candlestick_doji_pattern | candlestick |
| 161 | candlestick_harami_cross_pattern | candlestick |
| 162 | candlestick_long_legged_doji_pattern | candlestick |
| 163 | candlestick_three_white_soldiers_pattern | candlestick |
| 164 | candlestick_three_black_crows_pattern | candlestick |
| 165 | candlestick_dark_cloud_cover_pattern | candlestick |
| 166 | candlestick_piercing_line_pattern | candlestick |
| 167 | macd_entropy_20_12_26 | momentum |
| 168 | volume_entropy_5_volume_returns | returns |
| 169 | volume_entropy_ma_5_5_volume_returns | returns |
| 170 | volume_entropy_ma_5_10_volume_returns | returns |
| 171 | volume_entropy_10_volume_returns | returns |
| 172 | volume_entropy_ma_10_5_volume_returns | returns |
| 173 | volume_entropy_ma_10_10_volume_returns | returns |
| 174 | volume_entropy_20_volume_returns | returns |
| 175 | volume_entropy_ma_20_10_volume_returns | returns |
| 176 | volume_entropy_ma_20_5_volume_returns | returns |
| 177 | lempel_ziv_complexity_20 | other |
| 178 | entropy_rate_20 | entropy |
| 179 | spectral_entropy_20 | entropy |
| 180 | vectorbt_momentum_5_price_returns | returns |
| 181 | shannon_entropy_20_10 | entropy |
| 182 | vectorbt_momentum_10_price_returns | returns |
| 183 | vectorbt_momentum_20_price_returns | returns |
| 184 | vectorbt_momentum_50_price_returns | returns |
| 185 | vectorbt_acceleration_5_price_returns | returns |
| 186 | vectorbt_jerk_5_price_returns | returns |
| 187 | vectorbt_acceleration_10_price_returns | returns |
| 188 | vectorbt_jerk_10_price_returns | returns |
| 189 | vectorbt_trend_consistency_5_price_returns | returns |
| 190 | vectorbt_trend_strength_5_price_returns | returns |
| 191 | vectorbt_trend_consistency_10_price_returns | returns |
| 192 | vectorbt_trend_strength_10_price_returns | returns |
| 193 | vectorbt_trend_consistency_20_price_returns | returns |
| 194 | vectorbt_trend_strength_20_price_returns | returns |
| 195 | vectorbt_trend_strength_50_price_returns | returns |
| 196 | vectorbt_volume_acceleration_5_volume_returns | returns |
| 197 | vectorbt_volatility_acceleration_5_20_price_returns | returns |
| 198 | vectorbt_momentum_acceleration_5_10_price_returns | returns |
| 199 | vectorbt_acceleration_momentum_5_10_price_returns | returns |
| 200 | vectorbt_acceleration_consistency_5_10_price_returns | returns |
| 201 | vectorbt_acceleration_volatility_5_10_price_returns | returns |
| 202 | vectorbt_momentum_acceleration_5_20_price_returns | returns |
| 203 | vectorbt_acceleration_trend_strength_5_10_price_returns | returns |
| 204 | vectorbt_acceleration_regime_5_10_price_returns | returns |
| 205 | vectorbt_acceleration_momentum_5_20_price_returns | returns |
| 206 | vectorbt_acceleration_consistency_5_20_price_returns | returns |
| 207 | vectorbt_acceleration_volatility_5_20_price_returns | returns |
| 208 | vectorbt_acceleration_regime_5_20_price_returns | returns |
| 209 | vectorbt_acceleration_trend_strength_5_20_price_returns | returns |
| 210 | vectorbt_acceleration_trend_strength_10_10_price_returns | returns |
| 211 | vectorbt_acceleration_volatility_10_10_price_returns | returns |
| 212 | vectorbt_acceleration_momentum_10_10_price_returns | returns |
| 213 | vectorbt_momentum_acceleration_10_10_price_returns | returns |
| 214 | vectorbt_acceleration_consistency_10_10_price_returns | returns |
| 215 | vectorbt_acceleration_momentum_10_20_price_returns | returns |
| 216 | vectorbt_acceleration_consistency_10_20_price_returns | returns |
| 217 | vectorbt_acceleration_volatility_10_20_price_returns | returns |
| 218 | vectorbt_acceleration_trend_strength_10_20_price_returns | returns |
| 219 | vectorbt_acceleration_divergence_20_price_returns | returns |
| 220 | acceleration_features | interaction |
| 221 | dfa_slopes | other |

### Top Features by Economic Score

| Rank | Feature Name | Category | Economic Score | Regime Disc. | Clustering | Stability | Composite |
|------|--------------|----------|----------------|--------------|------------|-----------|-----------|
| 1 | simple_returns_1_price_returns |  | 0.072 | 0.991 | 0.000 | 0.991 | 0.416 |
| 2 | log_returns_10_price_returns |  | 0.038 | 0.968 | 0.000 | 0.990 | 0.410 |
| 3 | log_returns_1_price_returns |  | 0.090 | 0.983 | 0.000 | 0.993 | 0.440 |
| 4 | log_returns_5_price_returns |  | 0.059 | 0.977 | 0.000 | 0.992 | 0.423 |
| 5 | cumulative_returns_10_price_returns |  | 0.052 | 0.968 | 0.000 | 0.991 | 0.397 |
| 6 | simple_returns_10_price_returns |  | 0.041 | 0.980 | 0.000 | 0.992 | 0.405 |
| 7 | simple_returns_5_price_returns |  | 0.053 | 0.984 | 0.000 | 0.991 | 0.405 |
| 8 | rolling_returns_10_price_returns |  | 0.052 | 0.966 | 0.000 | 0.991 | 0.396 |
| 9 | returns_volatility_20_price_returns |  | 0.040 | 0.955 | 0.000 | 0.976 | 0.379 |
| 10 | rolling_returns_20_price_returns |  | 0.073 | 0.960 | 0.000 | 0.992 | 0.408 |
| 11 | sharpe_ratio_20_0.0_price_returns |  | 0.098 | 0.949 | 0.000 | 0.960 | 0.432 |
| 12 | advanced_cumulative_returns_10 |  | 0.198 | 0.973 | 0.000 | 0.970 | 0.488 |
| 13 | cumulative_returns_20_price_returns |  | 0.072 | 0.962 | 0.000 | 0.992 | 0.408 |
| 14 | returns_kurtosis_20_price_returns |  | 0.085 | 0.943 | 0.000 | 0.973 | 0.422 |
| 15 | returns_skewness_20_price_returns |  | 0.057 | 0.949 | 0.000 | 0.957 | 0.412 |
| 16 | rolling_zscore_returns_20 |  | 0.316 | 0.957 | 0.028 | 0.992 | 0.550 |
| 17 | advanced_cumulative_returns_20 |  | 0.191 | 0.953 | 0.000 | 0.962 | 0.479 |
| 18 | momentum_features |  | 0.189 | 0.995 | 0.000 | 0.960 | 0.489 |
| 19 | ar_1_coefficients_20 |  | 0.027 | 0.943 | 0.000 | 0.965 | 0.396 |
| 20 | ljung_box_pvalue_20_10 |  | 0.026 | 0.924 | 0.000 | 0.960 | 0.390 |


---

## 📈 Economic Metrics by Regime

| Regime ID | Sharpe Ratio | Sortino Ratio | Calmar Ratio | Win Rate | Avg Return | Volatility | Max DD | Samples |
|-----------|--------------|---------------|--------------|----------|------------|------------|--------|---------|


---

## 🔍 Clustering Quality Metrics

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| **Silhouette Score** | -0.016 | > 0.2 | ❌ FAIL |
| **Calinski-Harabasz** | 1.3 | > 100 | ❌ FAIL |
| **Davies-Bouldin** | 0.741 | < 2.0 | ✅ PASS |
| **Noise Ratio** | 0.0% | < 30% | ✅ PASS |

---

## 📊 Feature Category Distribution

| Category | Count | Percentage |
|----------|-------|------------|
|  | 221 | 100.0% |


---

## ⚡ Computational Performance

| Metric | Value |
|--------|-------|
| **Execution Time** | 127.83s |
| **VectorBT Available** | ✅ YES |
| **Hardware Optimization** | ✅ YES |
| **Cheap Proxies Enabled** | ✅ YES |
| **CV Folds** | 3 |
| **Silhouette Sample Ratio** | 10.0% |

---

## 💡 Recommendations

### Feature Selection Quality
- **Overall Score**: 0.295 (POOR)

### Economic Distinctiveness
- **Sharpe Variance**: 0.000 (POOR)

### Clustering Quality
- **Silhouette Score**: -0.016 (POOR)

### Next Steps
1. **Proceed with HDBSCAN**: Selected features show moderate economic distinctiveness
2. **Monitor Performance**: Track regime-specific Sharpe ratios during clustering
3. **Feature Refinement**: Consider reducing features

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
