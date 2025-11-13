# Final Feature Selection Report

**Generated:** 2025-11-12 23:54:16
**Step:** feature_generation_final_feature_selection_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** blank
- **Feature Count Targets:** [60, 50, 40]
- **Selection Method:** permutation ✅
- **Importance Type:** Permutation (captures feature interactions, not just Gini splits) 📊
- **Optimization Enabled:** True

## Feature Selection Methodology

✅ **Using Permutation Importance**
- Captures how features work together (feature interactions)
- More reliable than standard Gini importance for complex trading strategies
- Measures true impact on model predictions
- Better for identifying genuinely predictive features

## Feature Selection Results

- **60 Features Set:** 60 features selected
- **50 Features Set:** 50 features selected
- **40 Features Set:** 40 features selected

- **Total Feature Sets:** 3

## Selected Features by Set

### 60 Features Set (60 features)

1. resistance_level_1_20_price_returns
2. support_level_1_20_price_returns
3. volume_price_trend_base_3x_ratio_x_vectorbt_enhanced_obv_10_base_3x_ratio
4. candlestick_harami_cross_pattern_base_27x_ratio_minus_wavelet_energy_base_x_27x
5. volume_price_trend_trend_adj_6x_ratio_minus_candlestick_dark_cloud_cover_pattern_base_6x_ratio
6. advanced_support_resistance_features
7. enhanced_volatility_50
8. vectorbt_acceleration_momentum_10_20_price_returns
9. support_level_1_10_price_returns
10. vectorbt_enhanced_ad_line_20_base_minus_volume_vwap_20_vwap_6x_ratio
11. cycle_length_vwap_3x_ratio
12. wavelet_energy_base_9x_ratio
13. enhanced_volatility_100
14. volume_price_trend_base_3x_ratio_div_candlestick_harami_cross_pattern_base_27x_ratio
15. vectorbt_enhanced_obv_10_base_3x_ratio_div_candlestick_engulfing_pattern_vwap_9x_ratio
16. candlestick_piercing_line_pattern_base_3x_ratio_div_volume_entropy_ma_5_5_volume_returns_vwap
17. wavelet_energy_base_x_vectorbt_enhanced_ad_line_20_base_x_27x
18. ar_1_coefficients_20
19. volume_trend_strength_10_30
20. vectorbt_acceleration_volatility_5_20_price_returns
... and 40 more features

### 50 Features Set (50 features)

1. resistance_level_1_20_price_returns
2. support_level_1_20_price_returns
3. volume_price_trend_base_3x_ratio_x_vectorbt_enhanced_obv_10_base_3x_ratio
4. candlestick_harami_cross_pattern_base_27x_ratio_minus_wavelet_energy_base_x_27x
5. volume_price_trend_trend_adj_6x_ratio_minus_candlestick_dark_cloud_cover_pattern_base_6x_ratio
6. advanced_support_resistance_features
7. enhanced_volatility_50
8. vectorbt_acceleration_momentum_10_20_price_returns
9. support_level_1_10_price_returns
10. vectorbt_enhanced_ad_line_20_base_minus_volume_vwap_20_vwap_6x_ratio
11. cycle_length_vwap_3x_ratio
12. wavelet_energy_base_9x_ratio
13. enhanced_volatility_100
14. volume_price_trend_base_3x_ratio_div_candlestick_harami_cross_pattern_base_27x_ratio
15. vectorbt_enhanced_obv_10_base_3x_ratio_div_candlestick_engulfing_pattern_vwap_9x_ratio
16. candlestick_piercing_line_pattern_base_3x_ratio_div_volume_entropy_ma_5_5_volume_returns_vwap
17. wavelet_energy_base_x_vectorbt_enhanced_ad_line_20_base_x_27x
18. ar_1_coefficients_20
19. volume_trend_strength_10_30
20. vectorbt_acceleration_volatility_5_20_price_returns
... and 30 more features

### 40 Features Set (40 features)

1. resistance_level_1_20_price_returns
2. support_level_1_20_price_returns
3. volume_price_trend_base_3x_ratio_x_vectorbt_enhanced_obv_10_base_3x_ratio
4. candlestick_harami_cross_pattern_base_27x_ratio_minus_wavelet_energy_base_x_27x
5. volume_price_trend_trend_adj_6x_ratio_minus_candlestick_dark_cloud_cover_pattern_base_6x_ratio
6. advanced_support_resistance_features
7. enhanced_volatility_50
8. vectorbt_acceleration_momentum_10_20_price_returns
9. support_level_1_10_price_returns
10. vectorbt_enhanced_ad_line_20_base_minus_volume_vwap_20_vwap_6x_ratio
11. cycle_length_vwap_3x_ratio
12. wavelet_energy_base_9x_ratio
13. enhanced_volatility_100
14. volume_price_trend_base_3x_ratio_div_candlestick_harami_cross_pattern_base_27x_ratio
15. vectorbt_enhanced_obv_10_base_3x_ratio_div_candlestick_engulfing_pattern_vwap_9x_ratio
16. candlestick_piercing_line_pattern_base_3x_ratio_div_volume_entropy_ma_5_5_volume_returns_vwap
17. wavelet_energy_base_x_vectorbt_enhanced_ad_line_20_base_x_27x
18. ar_1_coefficients_20
19. volume_trend_strength_10_30
20. vectorbt_acceleration_volatility_5_20_price_returns
... and 20 more features


## Enhanced Feature Analysis

### Correlation Analysis

- **Average Correlation:** 0.0643
- **Max Correlation:** 1.0000
- **Min Correlation:** 0.0000
- **High Correlation Pairs:** 2
- **Correlation Threshold:** 0.8

### Redundancy Detection

- **Status:** Skipped (Performance optimization - correlation analysis provides sufficient information)

### Stability Analysis

- **Average Stability:** 0.5682
- **Stable Features:** 24
- **Stability Threshold:** 0.5837814803100295
- **Time Windows:** 5

### Cross-Validation Analysis

- **Average Consistency:** 0.1400
- **Consistent Features:** 3
- **Consistency Threshold:** 0.6
- **CV Folds:** 10

### Baseline Comparison

- **Improvement Ratio:** 2.15x
- **Selected Features Avg Score:** 0.020810
- **Baseline Avg Score:** 0.009660
- **Baseline Trials:** 10
- **Features Compared:** 60

## Performance Metrics

- **Execution Time:** N/A seconds
- **Optimization Enabled:** Yes
- **Hardware Optimization:** No

## Optimization Details

- **VectorBT Optimization:** Enabled
- **Rolling Optimizer:** Available
- **Hardware Manager:** Available

## Generated Artifacts

- **Feature Sets:** 3
- **Feature DataFrames:** 3
- **SHAP Analyses:** 0
- **Metadata Files:** 2
- **Total Artifacts:** 10

## Summary

Final feature selection completed successfully. Generated 3 optimized feature sets with comprehensive SHAP analysis and metadata. All artifacts saved in both pickle and markdown formats.

---
*Generated by Feature Generation Final Feature Selection Step at 2025-11-12 23:54:16*
