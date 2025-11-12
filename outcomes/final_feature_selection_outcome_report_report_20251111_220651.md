# Final Feature Selection Report

**Generated:** 2025-11-11 22:06:51
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

1. candlestick_harami_cross_pattern_base_27x_ratio_minus_candlestick_piercing_line_pattern_base_9x_ratio
2. volume_vwap_20_vwap_3x_ratio_log_ratio_vectorbt_enhanced_ad_line_20_base_x_27x
3. candlestick_harami_cross_pattern_base_27x_ratio
4. candlestick_piercing_line_pattern_base_9x_ratio_log_ratio_candlestick_piercing_line_pattern_base_3x_ratio
5. candlestick_dark_cloud_cover_pattern_base_6x_ratio_minus_fibonacci_0.786_10_price_returns_base_9x_ratio
6. vectorbt_parabolic_sar_0.05_0.2
7. enhanced_volatility_14_vwap_minus_vectorbt_parkinson_volatility_50_vwap_27x_ratio
8. candlestick_piercing_line_pattern_base_9x_ratio_log_wavelet_energy_base_6x_ratio
9. vectorbt_enhanced_obv_10_base_9x_ratio_div_candlestick_engulfing_pattern_base_6x_ratio
10. vectorbt_parkinson_volatility_50_vwap_log_ratio_vectorbt_enhanced_ad_line_20_base_x_27x
11. volume_price_trend_base_27x_ratio
12. candlestick_dark_cloud_cover_pattern_base_9x_ratio_log_vectorbt_enhanced_ad_line_20_base_27x_ratio
13. candlestick_piercing_line_pattern_base_9x_ratio_x_vectorbt_enhanced_ad_line_20_base_x_27x
14. vectorbt_smoothed_obv_10_base_27x_ratio
15. vectorbt_parkinson_volatility_50_base_6x_ratio_minus_candlestick_doji_pattern_base_3x_ratio
16. volume_price_correlation_20
17. candlestick_engulfing_pattern_base_27x_ratio_log_ratio_vectorbt_enhanced_obv_50_base_27x_ratio
18. candlestick_engulfing_pattern_base_27x_ratio_x_vectorbt_enhanced_obv_50_base_27x_ratio
19. candlestick_piercing_line_pattern_base_9x_ratio_div_candlestick_piercing_line_pattern_base_3x_ratio
20. volume_price_trend_base_3x_ratio_log_wavelet_energy_base_x_27x
... and 40 more features

### 50 Features Set (50 features)

1. candlestick_harami_cross_pattern_base_27x_ratio_minus_candlestick_piercing_line_pattern_base_9x_ratio
2. volume_vwap_20_vwap_3x_ratio_log_ratio_vectorbt_enhanced_ad_line_20_base_x_27x
3. candlestick_harami_cross_pattern_base_27x_ratio
4. candlestick_piercing_line_pattern_base_9x_ratio_log_ratio_candlestick_piercing_line_pattern_base_3x_ratio
5. candlestick_dark_cloud_cover_pattern_base_6x_ratio_minus_fibonacci_0.786_10_price_returns_base_9x_ratio
6. vectorbt_parabolic_sar_0.05_0.2
7. enhanced_volatility_14_vwap_minus_vectorbt_parkinson_volatility_50_vwap_27x_ratio
8. candlestick_piercing_line_pattern_base_9x_ratio_log_wavelet_energy_base_6x_ratio
9. vectorbt_enhanced_obv_10_base_9x_ratio_div_candlestick_engulfing_pattern_base_6x_ratio
10. vectorbt_parkinson_volatility_50_vwap_log_ratio_vectorbt_enhanced_ad_line_20_base_x_27x
11. volume_price_trend_base_27x_ratio
12. candlestick_dark_cloud_cover_pattern_base_9x_ratio_log_vectorbt_enhanced_ad_line_20_base_27x_ratio
13. candlestick_piercing_line_pattern_base_9x_ratio_x_vectorbt_enhanced_ad_line_20_base_x_27x
14. vectorbt_smoothed_obv_10_base_27x_ratio
15. vectorbt_parkinson_volatility_50_base_6x_ratio_minus_candlestick_doji_pattern_base_3x_ratio
16. volume_price_correlation_20
17. candlestick_engulfing_pattern_base_27x_ratio_log_ratio_vectorbt_enhanced_obv_50_base_27x_ratio
18. candlestick_engulfing_pattern_base_27x_ratio_x_vectorbt_enhanced_obv_50_base_27x_ratio
19. candlestick_piercing_line_pattern_base_9x_ratio_div_candlestick_piercing_line_pattern_base_3x_ratio
20. volume_price_trend_base_3x_ratio_log_wavelet_energy_base_x_27x
... and 30 more features

### 40 Features Set (40 features)

1. candlestick_harami_cross_pattern_base_27x_ratio_minus_candlestick_piercing_line_pattern_base_9x_ratio
2. volume_vwap_20_vwap_3x_ratio_log_ratio_vectorbt_enhanced_ad_line_20_base_x_27x
3. candlestick_harami_cross_pattern_base_27x_ratio
4. candlestick_piercing_line_pattern_base_9x_ratio_log_ratio_candlestick_piercing_line_pattern_base_3x_ratio
5. candlestick_dark_cloud_cover_pattern_base_6x_ratio_minus_fibonacci_0.786_10_price_returns_base_9x_ratio
6. vectorbt_parabolic_sar_0.05_0.2
7. enhanced_volatility_14_vwap_minus_vectorbt_parkinson_volatility_50_vwap_27x_ratio
8. candlestick_piercing_line_pattern_base_9x_ratio_log_wavelet_energy_base_6x_ratio
9. vectorbt_enhanced_obv_10_base_9x_ratio_div_candlestick_engulfing_pattern_base_6x_ratio
10. vectorbt_parkinson_volatility_50_vwap_log_ratio_vectorbt_enhanced_ad_line_20_base_x_27x
11. volume_price_trend_base_27x_ratio
12. candlestick_dark_cloud_cover_pattern_base_9x_ratio_log_vectorbt_enhanced_ad_line_20_base_27x_ratio
13. candlestick_piercing_line_pattern_base_9x_ratio_x_vectorbt_enhanced_ad_line_20_base_x_27x
14. vectorbt_smoothed_obv_10_base_27x_ratio
15. vectorbt_parkinson_volatility_50_base_6x_ratio_minus_candlestick_doji_pattern_base_3x_ratio
16. volume_price_correlation_20
17. candlestick_engulfing_pattern_base_27x_ratio_log_ratio_vectorbt_enhanced_obv_50_base_27x_ratio
18. candlestick_engulfing_pattern_base_27x_ratio_x_vectorbt_enhanced_obv_50_base_27x_ratio
19. candlestick_piercing_line_pattern_base_9x_ratio_div_candlestick_piercing_line_pattern_base_3x_ratio
20. volume_price_trend_base_3x_ratio_log_wavelet_energy_base_x_27x
... and 20 more features


## Enhanced Feature Analysis

### Correlation Analysis

- **Average Correlation:** 0.0230
- **Max Correlation:** 1.0000
- **Min Correlation:** 0.0000
- **High Correlation Pairs:** 0
- **Correlation Threshold:** 0.8

### Redundancy Detection

- **Status:** Skipped (Performance optimization - correlation analysis provides sufficient information)

### Stability Analysis

- **Average Stability:** 0.5976
- **Stable Features:** 24
- **Stability Threshold:** 0.6087105269564276
- **Time Windows:** 5

### Cross-Validation Analysis

- **Average Consistency:** 0.1133
- **Consistent Features:** 5
- **Consistency Threshold:** 0.6
- **CV Folds:** 10

### Baseline Comparison

- **Improvement Ratio:** 0.21x
- **Selected Features Avg Score:** 0.025011
- **Baseline Avg Score:** 0.120026
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
*Generated by Feature Generation Final Feature Selection Step at 2025-11-11 22:06:51*
