# Final Feature Selection Report

**Generated:** 2025-11-09 19:07:46
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

1. vectorbt_ichimoku_cloud_12_30_52
2. vwma_20_price_returns_vwap_log_candlestick_dark_cloud_cover_pattern_vwap_3x_ratio
3. vectorbt_rogers_satchell_volatility_30_base_9x_ratio
4. vectorbt_rogers_satchell_volatility_30_base_9x_ratio_x_wavelet_energy_base_9x_ratio
5. enhanced_volatility_20_vwap
6. fibonacci_0.618_20_price_returns_vwap_27x_ratio_minus_cycle_length_vwap_3x_ratio
7. enhanced_volatility_20_vwap_div_vwma_20_price_returns_trend_adj
8. vwma_20_price_returns_vwap_log_candlestick_harami_cross_pattern_base_6x_ratio
9. fibonacci_0.5_20_price_returns_vwap
10. vectorbt_parkinson_volatility_50_vwap_9x_ratio
11. directional_signal_vwap_log_volume_price_trend_vwap
12. body_size
13. wavelet_energy_trend_adj_9x_ratio
14. vectorbt_enhanced_ad_line_20_base_27x_ratio_log_ratio_fibonacci_0.236_5_price_returns_vwap_27x_ratio
15. vwma_20_price_returns_trend_adj
16. fibonacci_0.786_10_price_returns_base
17. vectorbt_bbands_10_1.5
18. fibonacci_0.236_5_price_returns_base
19. vectorbt_trend_strength_20_price_returns
20. vwma_20_price_returns_vwap_x_vectorbt_volatility_comprehensive_50_vwap
... and 40 more features

### 50 Features Set (50 features)

1. vectorbt_ichimoku_cloud_12_30_52
2. vwma_20_price_returns_vwap_log_candlestick_dark_cloud_cover_pattern_vwap_3x_ratio
3. vectorbt_rogers_satchell_volatility_30_base_9x_ratio
4. vectorbt_rogers_satchell_volatility_30_base_9x_ratio_x_wavelet_energy_base_9x_ratio
5. enhanced_volatility_20_vwap
6. fibonacci_0.618_20_price_returns_vwap_27x_ratio_minus_cycle_length_vwap_3x_ratio
7. enhanced_volatility_20_vwap_div_vwma_20_price_returns_trend_adj
8. vwma_20_price_returns_vwap_log_candlestick_harami_cross_pattern_base_6x_ratio
9. fibonacci_0.5_20_price_returns_vwap
10. vectorbt_parkinson_volatility_50_vwap_9x_ratio
11. directional_signal_vwap_log_volume_price_trend_vwap
12. body_size
13. wavelet_energy_trend_adj_9x_ratio
14. vectorbt_enhanced_ad_line_20_base_27x_ratio_log_ratio_fibonacci_0.236_5_price_returns_vwap_27x_ratio
15. vwma_20_price_returns_trend_adj
16. fibonacci_0.786_10_price_returns_base
17. vectorbt_bbands_10_1.5
18. fibonacci_0.236_5_price_returns_base
19. vectorbt_trend_strength_20_price_returns
20. vwma_20_price_returns_vwap_x_vectorbt_volatility_comprehensive_50_vwap
... and 30 more features

### 40 Features Set (40 features)

1. vectorbt_ichimoku_cloud_12_30_52
2. vwma_20_price_returns_vwap_log_candlestick_dark_cloud_cover_pattern_vwap_3x_ratio
3. vectorbt_rogers_satchell_volatility_30_base_9x_ratio
4. vectorbt_rogers_satchell_volatility_30_base_9x_ratio_x_wavelet_energy_base_9x_ratio
5. enhanced_volatility_20_vwap
6. fibonacci_0.618_20_price_returns_vwap_27x_ratio_minus_cycle_length_vwap_3x_ratio
7. enhanced_volatility_20_vwap_div_vwma_20_price_returns_trend_adj
8. vwma_20_price_returns_vwap_log_candlestick_harami_cross_pattern_base_6x_ratio
9. fibonacci_0.5_20_price_returns_vwap
10. vectorbt_parkinson_volatility_50_vwap_9x_ratio
11. directional_signal_vwap_log_volume_price_trend_vwap
12. body_size
13. wavelet_energy_trend_adj_9x_ratio
14. vectorbt_enhanced_ad_line_20_base_27x_ratio_log_ratio_fibonacci_0.236_5_price_returns_vwap_27x_ratio
15. vwma_20_price_returns_trend_adj
16. fibonacci_0.786_10_price_returns_base
17. vectorbt_bbands_10_1.5
18. fibonacci_0.236_5_price_returns_base
19. vectorbt_trend_strength_20_price_returns
20. vwma_20_price_returns_vwap_x_vectorbt_volatility_comprehensive_50_vwap
... and 20 more features


## Enhanced Feature Analysis

### Correlation Analysis

- **Average Correlation:** 0.1233
- **Max Correlation:** 1.0000
- **Min Correlation:** 0.0001
- **High Correlation Pairs:** 14
- **Correlation Threshold:** 0.8

### Redundancy Detection

- **Status:** Skipped (Performance optimization - correlation analysis provides sufficient information)

### Stability Analysis

- **Average Stability:** 0.6170
- **Stable Features:** 24
- **Stability Threshold:** 0.6278815307757066
- **Time Windows:** 5

### Cross-Validation Analysis

- **Average Consistency:** 0.5017
- **Consistent Features:** 22
- **Consistency Threshold:** 0.6
- **CV Folds:** 10

### Baseline Comparison

- **Improvement Ratio:** 1.06x
- **Selected Features Avg Score:** 0.094097
- **Baseline Avg Score:** 0.088546
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
*Generated by Feature Generation Final Feature Selection Step at 2025-11-09 19:07:46*
