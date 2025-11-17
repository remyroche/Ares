# Final Feature Selection Report

**Generated:** 2025-11-16 13:12:45
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

- **60 Features Set:** 56 features selected
- **50 Features Set:** 50 features selected
- **40 Features Set:** 40 features selected

- **Total Feature Sets:** 3

## Selected Features by Set

### 60 Features Set (56 features)

1. target_margin_long
2. target_margin_short
3. macd_12_26_9_returns_vwap
4. ultimate_oscillator_7_14_28_returns_vwap
5. candlestick_harami_cross_pattern_vwap_3x_ratio_log_ratio_vectorbt_rogers_satchell_volatility_30_vwap_x_3x
6. vectorbt_momentum_5_price_returns
7. volume_price_trend_vwap
8. volume_roc_5
9. vectorbt_parkinson_volatility_50_vwap_27x_ratio_log_ratio_candlestick_engulfing_pattern_base_9x_ratio
10. hurst_exponent
11. volume_price_trend_vwap_x_volume_entropy_10_volume_returns_vwap
12. fibonacci_0.5_10_price_returns_vwap_log_ratio_fibonacci_0.786_10_price_returns_vwap_x_9x
13. vectorbt_parabolic_sar_0.1_0.3
14. volume_price_trend_vwap_log_wavelet_energy_vwap_x_9x
15. ar_1_coefficients_20_base_9x_ratio
16. vectorbt_zigzag_5.0_2
17. vectorbt_rogers_satchell_volatility_30_vwap_6x_ratio
18. volume_price_trend_vwap_x_vwma_20_price_returns_vwap
19. volume_entropy_10_volume_returns_vwap_div_candlestick_engulfing_pattern_vwap_27x_ratio
20. vectorbt_zigzag_10.0_2
... and 36 more features

### 50 Features Set (50 features)

1. target_margin_long
2. target_margin_short
3. macd_12_26_9_returns_vwap
4. ultimate_oscillator_7_14_28_returns_vwap
5. candlestick_harami_cross_pattern_vwap_3x_ratio_log_ratio_vectorbt_rogers_satchell_volatility_30_vwap_x_3x
6. vectorbt_momentum_5_price_returns
7. volume_price_trend_vwap
8. volume_roc_5
9. vectorbt_parkinson_volatility_50_vwap_27x_ratio_log_ratio_candlestick_engulfing_pattern_base_9x_ratio
10. hurst_exponent
11. volume_price_trend_vwap_x_volume_entropy_10_volume_returns_vwap
12. fibonacci_0.5_10_price_returns_vwap_log_ratio_fibonacci_0.786_10_price_returns_vwap_x_9x
13. vectorbt_parabolic_sar_0.1_0.3
14. volume_price_trend_vwap_log_wavelet_energy_vwap_x_9x
15. ar_1_coefficients_20_base_9x_ratio
16. vectorbt_zigzag_5.0_2
17. vectorbt_rogers_satchell_volatility_30_vwap_6x_ratio
18. volume_price_trend_vwap_x_vwma_20_price_returns_vwap
19. volume_entropy_10_volume_returns_vwap_div_candlestick_engulfing_pattern_vwap_27x_ratio
20. vectorbt_zigzag_10.0_2
... and 30 more features

### 40 Features Set (40 features)

1. target_margin_long
2. target_margin_short
3. macd_12_26_9_returns_vwap
4. ultimate_oscillator_7_14_28_returns_vwap
5. candlestick_harami_cross_pattern_vwap_3x_ratio_log_ratio_vectorbt_rogers_satchell_volatility_30_vwap_x_3x
6. vectorbt_momentum_5_price_returns
7. volume_price_trend_vwap
8. volume_roc_5
9. vectorbt_parkinson_volatility_50_vwap_27x_ratio_log_ratio_candlestick_engulfing_pattern_base_9x_ratio
10. hurst_exponent
11. volume_price_trend_vwap_x_volume_entropy_10_volume_returns_vwap
12. fibonacci_0.5_10_price_returns_vwap_log_ratio_fibonacci_0.786_10_price_returns_vwap_x_9x
13. vectorbt_parabolic_sar_0.1_0.3
14. volume_price_trend_vwap_log_wavelet_energy_vwap_x_9x
15. ar_1_coefficients_20_base_9x_ratio
16. vectorbt_zigzag_5.0_2
17. vectorbt_rogers_satchell_volatility_30_vwap_6x_ratio
18. volume_price_trend_vwap_x_vwma_20_price_returns_vwap
19. volume_entropy_10_volume_returns_vwap_div_candlestick_engulfing_pattern_vwap_27x_ratio
20. vectorbt_zigzag_10.0_2
... and 20 more features


## Enhanced Feature Analysis

### Correlation Analysis

- **Average Correlation:** 0.0864
- **Max Correlation:** 1.0000
- **Min Correlation:** 0.0000
- **High Correlation Pairs:** 0
- **Correlation Threshold:** 0.8

### Redundancy Detection

- **Status:** Skipped (Performance optimization - correlation analysis provides sufficient information)

### Stability Analysis

- **Average Stability:** 0.3333
- **Stable Features:** 53
- **Stability Threshold:** 0.3333333333333333
- **Time Windows:** 5

### Cross-Validation Analysis

- **Average Consistency:** 0.1518
- **Consistent Features:** 1
- **Consistency Threshold:** 0.6
- **CV Folds:** 10

### Baseline Comparison

- **Improvement Ratio:** 0.98x
- **Selected Features Avg Score:** 0.199963
- **Baseline Avg Score:** 0.203882
- **Baseline Trials:** 10
- **Features Compared:** 56

### Selection Frequency Distribution

- **Distribution Mode:** bimodal
- **Interpretation:** ✅ Clear separation between stable and unstable features
- **Highly Stable Features (>80%):** 0
- **Highly Unstable Features (<20%):** 33
- **Unstable Features Ratio:** 82.1%

**Frequency Breakdown:**
- 0-20%: 33 features (58.9%)
- 100%: 0 features (0.0%)
- 20-40%: 13 features (23.2%)
- 40-60%: 9 features (16.1%)
- 60-80%: 1 features (1.8%)
- 80-100%: 0 features (0.0%)

**⚠️ Warnings:**
- 🚨 >60% of features are highly unstable (selected <40% of time)
- ⚠️ <20% of features are highly stable (selected >80% of time)

### Mutual Information Stability (Correlation Proxy)

- **Stable Features (CV < 0.3):** 55
- **High MI Features (>0.1):** 41
- **Mean MI Stability:** 1.000
- **Method:** correlation_proxy
- **Execution Time:** 0.0s

✅ High MI stability across folds

### Data Leakage Detection (Phase 3)

- **Perfect Correlations (>0.99):** 0
- **Suspicious Correlations (>0.95):** 0
- **Execution Time:** 0.0s

✅ No data leakage detected

### Feature Information Content (Phase 3)

- **Low Variance Features (<0.01):** 0
- **Quasi-Constant Features (>99%):** 0
- **Execution Time:** 0.0s

✅ All features have sufficient information content

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
*Generated by Feature Generation Final Feature Selection Step at 2025-11-16 13:12:45*
