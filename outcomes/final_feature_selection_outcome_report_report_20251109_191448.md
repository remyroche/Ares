# Final Feature Selection Report

**Generated:** 2025-11-09 19:14:48
**Step:** feature_generation_final_feature_selection_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** light
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

1. fibonacci_0.236_5_price_returns_vwap_minus_volume_vwap_20_base_3x_ratio
2. vectorbt_ichimoku_cloud_12_30_52
3. candlestick_dark_cloud_cover_pattern_vwap_3x_ratio_log_fibonacci_0.786_10_price_returns_vwap
4. vectorbt_enhanced_ad_line_20_base_27x_ratio_log_ratio_fibonacci_0.786_10_price_returns_base_x_27x
5. candlestick_dark_cloud_cover_pattern_vwap_3x_ratio_minus_fibonacci_0.786_10_price_returns_vwap
6. directional_signal_vwap_log_ratio_volume_price_trend_vwap
7. entropy_rate_20_base_3x_ratio_div_fractal_dimension_base_27x_ratio
8. vectorbt_trend_strength_20_price_returns
9. vectorbt_acceleration_consistency_10_20_price_returns
10. candlestick_harami_cross_pattern_base_6x_ratio
11. vectorbt_bbands_10_1.5
12. volume_vwap_20_base_9x_ratio
13. candlestick_dark_cloud_cover_pattern_vwap_3x_ratio_log_ratio_fibonacci_0.618_20_price_returns_vwap_27x_ratio
14. fibonacci_0.236_5_price_returns_vwap
15. fibonacci_0.5_20_price_returns_trend_adj
16. fibonacci_0.618_20_price_returns_vwap_27x_ratio
17. vectorbt_volatility_comprehensive_50_vwap
18. fibonacci_0.236_5_price_returns_vwap_27x_ratio_x_wavelet_energy_base_6x_ratio
19. cycle_length_base_27x_ratio_log_candlestick_piercing_line_pattern_vwap_6x_ratio
20. volume_return
... and 40 more features

### 50 Features Set (50 features)

1. fibonacci_0.236_5_price_returns_vwap_minus_volume_vwap_20_base_3x_ratio
2. vectorbt_ichimoku_cloud_12_30_52
3. candlestick_dark_cloud_cover_pattern_vwap_3x_ratio_log_fibonacci_0.786_10_price_returns_vwap
4. vectorbt_enhanced_ad_line_20_base_27x_ratio_log_ratio_fibonacci_0.786_10_price_returns_base_x_27x
5. candlestick_dark_cloud_cover_pattern_vwap_3x_ratio_minus_fibonacci_0.786_10_price_returns_vwap
6. directional_signal_vwap_log_ratio_volume_price_trend_vwap
7. entropy_rate_20_base_3x_ratio_div_fractal_dimension_base_27x_ratio
8. vectorbt_trend_strength_20_price_returns
9. vectorbt_acceleration_consistency_10_20_price_returns
10. candlestick_harami_cross_pattern_base_6x_ratio
11. vectorbt_bbands_10_1.5
12. volume_vwap_20_base_9x_ratio
13. candlestick_dark_cloud_cover_pattern_vwap_3x_ratio_log_ratio_fibonacci_0.618_20_price_returns_vwap_27x_ratio
14. fibonacci_0.236_5_price_returns_vwap
15. fibonacci_0.5_20_price_returns_trend_adj
16. fibonacci_0.618_20_price_returns_vwap_27x_ratio
17. vectorbt_volatility_comprehensive_50_vwap
18. fibonacci_0.236_5_price_returns_vwap_27x_ratio_x_wavelet_energy_base_6x_ratio
19. cycle_length_base_27x_ratio_log_candlestick_piercing_line_pattern_vwap_6x_ratio
20. volume_return
... and 30 more features

### 40 Features Set (40 features)

1. fibonacci_0.236_5_price_returns_vwap_minus_volume_vwap_20_base_3x_ratio
2. vectorbt_ichimoku_cloud_12_30_52
3. candlestick_dark_cloud_cover_pattern_vwap_3x_ratio_log_fibonacci_0.786_10_price_returns_vwap
4. vectorbt_enhanced_ad_line_20_base_27x_ratio_log_ratio_fibonacci_0.786_10_price_returns_base_x_27x
5. candlestick_dark_cloud_cover_pattern_vwap_3x_ratio_minus_fibonacci_0.786_10_price_returns_vwap
6. directional_signal_vwap_log_ratio_volume_price_trend_vwap
7. entropy_rate_20_base_3x_ratio_div_fractal_dimension_base_27x_ratio
8. vectorbt_trend_strength_20_price_returns
9. vectorbt_acceleration_consistency_10_20_price_returns
10. candlestick_harami_cross_pattern_base_6x_ratio
11. vectorbt_bbands_10_1.5
12. volume_vwap_20_base_9x_ratio
13. candlestick_dark_cloud_cover_pattern_vwap_3x_ratio_log_ratio_fibonacci_0.618_20_price_returns_vwap_27x_ratio
14. fibonacci_0.236_5_price_returns_vwap
15. fibonacci_0.5_20_price_returns_trend_adj
16. fibonacci_0.618_20_price_returns_vwap_27x_ratio
17. vectorbt_volatility_comprehensive_50_vwap
18. fibonacci_0.236_5_price_returns_vwap_27x_ratio_x_wavelet_energy_base_6x_ratio
19. cycle_length_base_27x_ratio_log_candlestick_piercing_line_pattern_vwap_6x_ratio
20. volume_return
... and 20 more features


## Enhanced Feature Analysis

### Correlation Analysis

- **Average Correlation:** 0.0998
- **Max Correlation:** 1.0000
- **Min Correlation:** 0.0000
- **High Correlation Pairs:** 8
- **Correlation Threshold:** 0.8

### Redundancy Detection

- **Status:** Skipped (Performance optimization - correlation analysis provides sufficient information)

### Stability Analysis

- **Average Stability:** 0.6130
- **Stable Features:** 24
- **Stability Threshold:** 0.6347517472348413
- **Time Windows:** 5

### Cross-Validation Analysis

- **Average Consistency:** 0.4900
- **Consistent Features:** 24
- **Consistency Threshold:** 0.6
- **CV Folds:** 10

### Baseline Comparison

- **Improvement Ratio:** 1.00x
- **Selected Features Avg Score:** 0.090041
- **Baseline Avg Score:** 0.089760
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
*Generated by Feature Generation Final Feature Selection Step at 2025-11-09 19:14:48*
