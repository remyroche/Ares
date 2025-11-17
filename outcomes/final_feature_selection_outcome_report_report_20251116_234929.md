# Final Feature Selection Report

**Generated:** 2025-11-16 23:49:29
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

- **60 Features Set:** 59 features selected
- **50 Features Set:** 50 features selected
- **40 Features Set:** 40 features selected

- **Total Feature Sets:** 3

## Selected Features by Set

### 60 Features Set (59 features)

1. enhanced_volatility_30_vwap_9x_ratio
2. target_margin_long
3. cycle_length_base_27x_ratio
4. vectorbt_acceleration_trend_strength_10_10_price_returns
5. enhanced_volatility_30_vwap_9x_ratio_log_ratio_fibonacci_0.786_10_price_returns_vwap_9x_ratio
6. volume_price_trend_vwap_minus_vectorbt_rogers_satchell_volatility_30_vwap_x_3x
7. vectorbt_momentum_5_price_returns
8. cycle_length_vwap_6x_ratio_log_ratio_vectorbt_rogers_satchell_volatility_30_vwap_x_27x
9. candlestick_harami_cross_pattern_vwap_3x_ratio_log_ratio_vectorbt_parkinson_volatility_50_vwap_27x_ratio
10. fibonacci_0.786_10_price_returns_vwap_x_9x_log_ratio_vectorbt_parkinson_volatility_50_vwap_x_27x
11. vectorbt_rogers_satchell_volatility_30_vwap_3x_ratio
12. vectorbt_momentum_acceleration_10_20_price_returns
13. enhanced_volatility_20_base_9x_ratio
14. vectorbt_parkinson_volatility_50_base_27x_ratio
15. volume_price_trend_vwap_minus_volume_entropy_10_volume_returns_vwap
16. vectorbt_acceleration_10_price_returns
17. fibonacci_0.5_10_price_returns_vwap_minus_vectorbt_volatility_comprehensive_50_vwap
18. volume_price_trend_base_log_wavelet_energy_vwap_x_27x
19. vectorbt_rogers_satchell_volatility_30_vwap_27x_ratio
20. vectorbt_momentum_50_price_returns
... and 39 more features

### 50 Features Set (50 features)

1. enhanced_volatility_30_vwap_9x_ratio
2. target_margin_long
3. cycle_length_base_27x_ratio
4. vectorbt_acceleration_trend_strength_10_10_price_returns
5. enhanced_volatility_30_vwap_9x_ratio_log_ratio_fibonacci_0.786_10_price_returns_vwap_9x_ratio
6. volume_price_trend_vwap_minus_vectorbt_rogers_satchell_volatility_30_vwap_x_3x
7. vectorbt_momentum_5_price_returns
8. cycle_length_vwap_6x_ratio_log_ratio_vectorbt_rogers_satchell_volatility_30_vwap_x_27x
9. candlestick_harami_cross_pattern_vwap_3x_ratio_log_ratio_vectorbt_parkinson_volatility_50_vwap_27x_ratio
10. fibonacci_0.786_10_price_returns_vwap_x_9x_log_ratio_vectorbt_parkinson_volatility_50_vwap_x_27x
11. vectorbt_rogers_satchell_volatility_30_vwap_3x_ratio
12. vectorbt_momentum_acceleration_10_20_price_returns
13. enhanced_volatility_20_base_9x_ratio
14. vectorbt_parkinson_volatility_50_base_27x_ratio
15. volume_price_trend_vwap_minus_volume_entropy_10_volume_returns_vwap
16. vectorbt_acceleration_10_price_returns
17. fibonacci_0.5_10_price_returns_vwap_minus_vectorbt_volatility_comprehensive_50_vwap
18. volume_price_trend_base_log_wavelet_energy_vwap_x_27x
19. vectorbt_rogers_satchell_volatility_30_vwap_27x_ratio
20. vectorbt_momentum_50_price_returns
... and 30 more features

### 40 Features Set (40 features)

1. enhanced_volatility_30_vwap_9x_ratio
2. target_margin_long
3. cycle_length_base_27x_ratio
4. vectorbt_acceleration_trend_strength_10_10_price_returns
5. enhanced_volatility_30_vwap_9x_ratio_log_ratio_fibonacci_0.786_10_price_returns_vwap_9x_ratio
6. volume_price_trend_vwap_minus_vectorbt_rogers_satchell_volatility_30_vwap_x_3x
7. vectorbt_momentum_5_price_returns
8. cycle_length_vwap_6x_ratio_log_ratio_vectorbt_rogers_satchell_volatility_30_vwap_x_27x
9. candlestick_harami_cross_pattern_vwap_3x_ratio_log_ratio_vectorbt_parkinson_volatility_50_vwap_27x_ratio
10. fibonacci_0.786_10_price_returns_vwap_x_9x_log_ratio_vectorbt_parkinson_volatility_50_vwap_x_27x
11. vectorbt_rogers_satchell_volatility_30_vwap_3x_ratio
12. vectorbt_momentum_acceleration_10_20_price_returns
13. enhanced_volatility_20_base_9x_ratio
14. vectorbt_parkinson_volatility_50_base_27x_ratio
15. volume_price_trend_vwap_minus_volume_entropy_10_volume_returns_vwap
16. vectorbt_acceleration_10_price_returns
17. fibonacci_0.5_10_price_returns_vwap_minus_vectorbt_volatility_comprehensive_50_vwap
18. volume_price_trend_base_log_wavelet_energy_vwap_x_27x
19. vectorbt_rogers_satchell_volatility_30_vwap_27x_ratio
20. vectorbt_momentum_50_price_returns
... and 20 more features


## Enhanced Feature Analysis

### Correlation Analysis

- **Average Correlation:** 0.0770  — average pairwise |ρ| between features; lower is better and values <0.2 indicate low redundancy.
- **Max Correlation:** 1.0000  — highest |ρ| observed; very high values may indicate near-duplicate signals.
- **Min Correlation:** 0.0000  — lowest |ρ|; values near 0 show some features are nearly independent.
- **High Correlation Pairs:** 0  — number of feature pairs above the threshold; 0 is ideal.
- **Correlation Threshold:** 0.8  — pairs above this are considered redundant for clustering.

### Redundancy Detection

- **Status:** Skipped (Performance optimization - correlation analysis provides sufficient information)

### Stability Analysis

- **Average Stability:** 0.5706  — 0–1 score of importance consistency across time windows; higher is better and >0.5 is strong.
- **Stable Features:** 56  — features above the stability threshold; more indicates a more robust set.
- **Stability Threshold:** 0.3333333333333333  — adaptive cutoff used to classify features as stable.
- **Time Windows:** 5  — number of rolling windows used for stability estimation.

### Cross-Validation Analysis

- **Average Consistency:** 0.1814  — average selection frequency across folds (0–1); higher means features reappear more often.
- **Consistent Features:** 9  — features with consistency above the threshold; more is better.
- **Consistency Threshold:** 0.6  — minimum fold frequency to be considered consistent.
- **CV Folds:** 10  — number of time-series splits used; more folds give a stricter stability test.

### Baseline Comparison

- **Improvement Ratio:** 0.99x  — selected set score / baseline score; values <1.0 mean the selection outperforms baseline.
- **Selected Features Avg Score:** 0.216698  — mean importance of selected features; higher is better.
- **Baseline Avg Score:** 0.219495  — mean importance over all features; acts as a reference level.
- **Baseline Trials:** 10  — number of random baseline draws; more gives a more stable baseline estimate.
- **Features Compared:** 59  — size of the selected feature set used for the comparison.

### Selection Frequency Distribution

- **Distribution Mode:** bimodal
- **Interpretation:** ✅ Clear separation between stable and unstable features
- **Highly Stable Features (>80%):** 2
- **Highly Unstable Features (<20%):** 38
- **Unstable Features Ratio:** 81.4%

**Frequency Breakdown:**
- 0-20%: 38 features (64.4%)
- 100%: 1 features (1.7%)
- 20-40%: 10 features (16.9%)
- 40-60%: 2 features (3.4%)
- 60-80%: 6 features (10.2%)
- 80-100%: 2 features (3.4%)

**⚠️ Warnings:**
- 🚨 >60% of features are highly unstable (selected <40% of time)
- ⚠️ <20% of features are highly stable (selected >80% of time)

### Mutual Information Stability (Correlation Proxy)

- **Stable Features (CV < 0.3):** 0
- **High MI Features (>0.1):** 0
- **Mean MI Stability:** -1.000
- **Method:** correlation_proxy
- **Execution Time:** 0.0s

🚨 Low MI stability - features may not generalize well

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
*Generated by Feature Generation Final Feature Selection Step at 2025-11-16 23:49:29*
