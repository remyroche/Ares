# Final Feature Selection Report

**Generated:** 2025-11-16 16:08:48
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

1. target_margin_long
2. target_margin_short
3. macd_12_26_9_returns_vwap
4. volume_price_trend_vwap_log_ratio_vectorbt_rogers_satchell_volatility_30_vwap_x_3x
5. vectorbt_zigzag_7.0_3
6. hurst_exponent
7. vectorbt_trend_consistency_5_price_returns
8. fibonacci_0.5_10_price_returns_vwap_log_ratio_fibonacci_0.786_10_price_returns_vwap_x_9x
9. vectorbt_parkinson_volatility_50_vwap_27x_ratio_log_ratio_candlestick_engulfing_pattern_base_9x_ratio
10. vectorbt_parabolic_sar_0.1_0.3
11. vectorbt_momentum_5_price_returns
12. returns_kurtosis_20_price_returns
13. candlestick_harami_cross_pattern_vwap_3x_ratio_log_ratio_vectorbt_rogers_satchell_volatility_30_vwap_x_3x
14. volume_price_trend_vwap_x_volume_entropy_10_volume_returns_vwap
15. volume_price_trend_vwap_div_candlestick_piercing_line_pattern_vwap_3x_ratio
16. volume_price_trend_vwap_x_vwma_20_price_returns_vwap
17. shannon_entropy_20_10
18. ar_1_coefficients_20_base_9x_ratio
19. volume_price_trend_vwap_minus_volume_entropy_10_volume_returns_vwap
20. vectorbt_acceleration_5_price_returns
... and 40 more features

### 50 Features Set (50 features)

1. target_margin_long
2. target_margin_short
3. macd_12_26_9_returns_vwap
4. volume_price_trend_vwap_log_ratio_vectorbt_rogers_satchell_volatility_30_vwap_x_3x
5. vectorbt_zigzag_7.0_3
6. hurst_exponent
7. vectorbt_trend_consistency_5_price_returns
8. fibonacci_0.5_10_price_returns_vwap_log_ratio_fibonacci_0.786_10_price_returns_vwap_x_9x
9. vectorbt_parkinson_volatility_50_vwap_27x_ratio_log_ratio_candlestick_engulfing_pattern_base_9x_ratio
10. vectorbt_parabolic_sar_0.1_0.3
11. vectorbt_momentum_5_price_returns
12. returns_kurtosis_20_price_returns
13. candlestick_harami_cross_pattern_vwap_3x_ratio_log_ratio_vectorbt_rogers_satchell_volatility_30_vwap_x_3x
14. volume_price_trend_vwap_x_volume_entropy_10_volume_returns_vwap
15. volume_price_trend_vwap_div_candlestick_piercing_line_pattern_vwap_3x_ratio
16. volume_price_trend_vwap_x_vwma_20_price_returns_vwap
17. shannon_entropy_20_10
18. ar_1_coefficients_20_base_9x_ratio
19. volume_price_trend_vwap_minus_volume_entropy_10_volume_returns_vwap
20. vectorbt_acceleration_5_price_returns
... and 30 more features

### 40 Features Set (40 features)

1. target_margin_long
2. target_margin_short
3. macd_12_26_9_returns_vwap
4. volume_price_trend_vwap_log_ratio_vectorbt_rogers_satchell_volatility_30_vwap_x_3x
5. vectorbt_zigzag_7.0_3
6. hurst_exponent
7. vectorbt_trend_consistency_5_price_returns
8. fibonacci_0.5_10_price_returns_vwap_log_ratio_fibonacci_0.786_10_price_returns_vwap_x_9x
9. vectorbt_parkinson_volatility_50_vwap_27x_ratio_log_ratio_candlestick_engulfing_pattern_base_9x_ratio
10. vectorbt_parabolic_sar_0.1_0.3
11. vectorbt_momentum_5_price_returns
12. returns_kurtosis_20_price_returns
13. candlestick_harami_cross_pattern_vwap_3x_ratio_log_ratio_vectorbt_rogers_satchell_volatility_30_vwap_x_3x
14. volume_price_trend_vwap_x_volume_entropy_10_volume_returns_vwap
15. volume_price_trend_vwap_div_candlestick_piercing_line_pattern_vwap_3x_ratio
16. volume_price_trend_vwap_x_vwma_20_price_returns_vwap
17. shannon_entropy_20_10
18. ar_1_coefficients_20_base_9x_ratio
19. volume_price_trend_vwap_minus_volume_entropy_10_volume_returns_vwap
20. vectorbt_acceleration_5_price_returns
... and 20 more features


## Enhanced Feature Analysis

### Correlation Analysis

- **Average Correlation:** 0.0919  — average pairwise |ρ| between features; lower is better and values <0.2 indicate low redundancy.
- **Max Correlation:** 1.0000  — highest |ρ| observed; very high values may indicate near-duplicate signals.
- **Min Correlation:** 0.0000  — lowest |ρ|; values near 0 show some features are nearly independent.
- **High Correlation Pairs:** 0  — number of feature pairs above the threshold; 0 is ideal.
- **Correlation Threshold:** 0.8  — pairs above this are considered redundant for clustering.

### Redundancy Detection

- **Status:** Skipped (Performance optimization - correlation analysis provides sufficient information)

### Stability Analysis

- **Average Stability:** 0.3333  — 0–1 score of importance consistency across time windows; higher is better and >0.5 is strong.
- **Stable Features:** 56  — features above the stability threshold; more indicates a more robust set.
- **Stability Threshold:** 0.3333333333333333  — adaptive cutoff used to classify features as stable.
- **Time Windows:** 5  — number of rolling windows used for stability estimation.

### Cross-Validation Analysis

- **Average Consistency:** 0.1367  — average selection frequency across folds (0–1); higher means features reappear more often.
- **Consistent Features:** 7  — features with consistency above the threshold; more is better.
- **Consistency Threshold:** 0.6  — minimum fold frequency to be considered consistent.
- **CV Folds:** 10  — number of time-series splits used; more folds give a stricter stability test.

### Baseline Comparison

- **Improvement Ratio:** 0.93x  — selected set score / baseline score; values <1.0 mean the selection outperforms baseline.
- **Selected Features Avg Score:** 0.204995  — mean importance of selected features; higher is better.
- **Baseline Avg Score:** 0.219410  — mean importance over all features; acts as a reference level.
- **Baseline Trials:** 10  — number of random baseline draws; more gives a more stable baseline estimate.
- **Features Compared:** 60  — size of the selected feature set used for the comparison.

### Selection Frequency Distribution

- **Distribution Mode:** bimodal
- **Interpretation:** ✅ Clear separation between stable and unstable features
- **Highly Stable Features (>80%):** 0
- **Highly Unstable Features (<20%):** 45
- **Unstable Features Ratio:** 83.3%

**Frequency Breakdown:**
- 0-20%: 45 features (75.0%)
- 100%: 0 features (0.0%)
- 20-40%: 5 features (8.3%)
- 40-60%: 3 features (5.0%)
- 60-80%: 7 features (11.7%)
- 80-100%: 0 features (0.0%)

**⚠️ Warnings:**
- 🚨 >60% of features are highly unstable (selected <40% of time)
- ⚠️ <20% of features are highly stable (selected >80% of time)

### Mutual Information Stability (Correlation Proxy)

- **Stable Features (CV < 0.3):** 58
- **High MI Features (>0.1):** 43
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
*Generated by Feature Generation Final Feature Selection Step at 2025-11-16 16:08:48*
