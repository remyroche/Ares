# Final Feature Selection Report

**Generated:** 2025-11-09 21:08:39
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

1. fibonacci_0.236_5_price_returns_vwap_27x_ratio
2. enhanced_volatility_20_vwap_minus_fibonacci_0.786_10_price_returns_vwap_x_9x
3. vwma_20_price_returns_trend_adj
4. volume_return
5. volume_vwap_20_vwap_3x_ratio
6. returns_volatility_20_price_returns_vwap_x_directional_signal_base_27x_ratio
7. fibonacci_0.786_10_price_returns_vwap_9x_ratio
8. returns_volatility_20_price_returns_base_minus_cycle_length_vwap_27x_ratio
9. vectorbt_volatility_comprehensive_50_vwap_log_wavelet_energy_base_6x_ratio
10. price_range
11. entropy_rate_20_vwap
12. returns_volatility_20_price_returns_vwap_log_fibonacci_0.786_10_price_returns_base_x_9x
13. vectorbt_enhanced_ad_line_20_base_27x_ratio_log_candlestick_engulfing_pattern_base_3x_ratio
14. vwma_20_price_returns_vwap_x_vectorbt_volatility_comprehensive_50_vwap
15. candlestick_doji_pattern_vwap_6x_ratio
16. entropy_rate_20_vwap_minus_cycle_length_vwap_3x_ratio
17. vectorbt_ichimoku_cloud_12_30_52
18. vwma_20_price_returns_vwap_log_ratio_candlestick_engulfing_pattern_base_9x_ratio
19. fibonacci_0.236_5_price_returns_base
20. fibonacci_0.618_20_price_returns_vwap_27x_ratio_div_fibonacci_0.786_10_price_returns_vwap_x_9x
... and 40 more features

### 50 Features Set (50 features)

1. fibonacci_0.236_5_price_returns_vwap_27x_ratio
2. enhanced_volatility_20_vwap_minus_fibonacci_0.786_10_price_returns_vwap_x_9x
3. vwma_20_price_returns_trend_adj
4. volume_return
5. volume_vwap_20_vwap_3x_ratio
6. returns_volatility_20_price_returns_vwap_x_directional_signal_base_27x_ratio
7. fibonacci_0.786_10_price_returns_vwap_9x_ratio
8. returns_volatility_20_price_returns_base_minus_cycle_length_vwap_27x_ratio
9. vectorbt_volatility_comprehensive_50_vwap_log_wavelet_energy_base_6x_ratio
10. price_range
11. entropy_rate_20_vwap
12. returns_volatility_20_price_returns_vwap_log_fibonacci_0.786_10_price_returns_base_x_9x
13. vectorbt_enhanced_ad_line_20_base_27x_ratio_log_candlestick_engulfing_pattern_base_3x_ratio
14. vwma_20_price_returns_vwap_x_vectorbt_volatility_comprehensive_50_vwap
15. candlestick_doji_pattern_vwap_6x_ratio
16. entropy_rate_20_vwap_minus_cycle_length_vwap_3x_ratio
17. vectorbt_ichimoku_cloud_12_30_52
18. vwma_20_price_returns_vwap_log_ratio_candlestick_engulfing_pattern_base_9x_ratio
19. fibonacci_0.236_5_price_returns_base
20. fibonacci_0.618_20_price_returns_vwap_27x_ratio_div_fibonacci_0.786_10_price_returns_vwap_x_9x
... and 30 more features

### 40 Features Set (40 features)

1. enhanced_volatility_20_vwap_minus_fibonacci_0.786_10_price_returns_vwap_x_9x
2. fibonacci_0.236_5_price_returns_vwap_27x_ratio
3. vwma_20_price_returns_vwap_x_vectorbt_volatility_comprehensive_50_vwap
4. fibonacci_0.786_10_price_returns_vwap_9x_ratio
5. vectorbt_enhanced_ad_line_20_base_27x_ratio_log_candlestick_engulfing_pattern_base_3x_ratio
6. returns_volatility_20_price_returns_vwap_x_directional_signal_base_27x_ratio
7. fibonacci_0.618_10_price_returns_base
8. volume_vwap_20_vwap_3x_ratio
9. vectorbt_volatility_comprehensive_50_vwap_log_wavelet_energy_base_6x_ratio
10. entropy_rate_20_vwap
11. returns_volatility_20_price_returns_base_minus_cycle_length_vwap_27x_ratio
12. vectorbt_bbands_10_1.5
13. wavelet_energy_base_9x_ratio_log_fibonacci_0.786_10_price_returns_vwap_x_9x
14. entropy_rate_20_vwap_minus_cycle_length_vwap_3x_ratio
15. vectorbt_enhanced_obv_50_base
16. cycle_length_base_27x_ratio_minus_returns_volatility_20_price_returns_base
17. fibonacci_0.618_20_price_returns_base
18. vectorbt_enhanced_ad_line_20_base_27x_ratio_x_cycle_length_vwap_27x_ratio
19. returns_volatility_20_price_returns_vwap_log_fibonacci_0.786_10_price_returns_base_x_9x
20. fibonacci_0.5_20_price_returns_vwap
... and 20 more features


## Enhanced Feature Analysis

### Correlation Analysis

- **Average Correlation:** 0.1264
- **Max Correlation:** 1.0000
- **Min Correlation:** 0.0000
- **High Correlation Pairs:** 19
- **Correlation Threshold:** 0.8

### Redundancy Detection

- **Status:** Skipped (Performance optimization - correlation analysis provides sufficient information)

### Stability Analysis

- **Average Stability:** 0.6279
- **Stable Features:** 24
- **Stability Threshold:** 0.6518797483702635
- **Time Windows:** 5

### Cross-Validation Analysis

- **Average Consistency:** 0.4767
- **Consistent Features:** 23
- **Consistency Threshold:** 0.6
- **CV Folds:** 10

### Baseline Comparison

- **Improvement Ratio:** 1.15x
- **Selected Features Avg Score:** 0.083583
- **Baseline Avg Score:** 0.072442
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
*Generated by Feature Generation Final Feature Selection Step at 2025-11-09 21:08:39*
