# Lookback Optimization Report

**Generated:** 2025-11-23 15:04:11
**Step:** feature_generation_period_lookback_optimization_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** full

## Optimization Results

- **Momentum Lookback:** 15
- **Trend Lookback:** 15
- **Volatility Lookback:** 20
- **Volume Lookback:** 5
- **Optimization Score:** 0.60

## Comprehensive Optimization Analysis

### Data Export

- **Per-Feature Metrics CSV:** `outcomes/feature_generation_period_lookback_optimization_step_per_feature_metrics_20251123_150411.csv`
- **Full Path:** `/Users/remyroche/Documents/Ares/outcomes/feature_generation_period_lookback_optimization_step_per_feature_metrics_20251123_150411.csv`

### Optimization Performance Metrics

| Metric | Value |
|--------|-------|
| Optimization Method | data_driven_cross_validation |
| Total Features Analyzed | 255 |
| Lookback Range Tested | 1-50 |
| Cross-Validation Folds | 5 |
| Optimization Efficiency | 85.0% |
| Stability Score | 0.883 |
| Performance Score | 0.326 |

### Global Optimization Metrics

| Metric | Value |
|--------|-------|
| Total Features Optimized | 246 |
| Categories Processed | 5 |
| Average Lookback Period | 12.0 |
| Lookback Range | 1-50 |
| Step Size | 1 |
| Cross-Validation Folds | 5 |
| Total Optimization Time | N/A seconds |
| Memory Usage | N/A MB |
| Success Rate | N/A |

### Individual Feature Optimization Results

This table shows detailed optimization results for each feature category.

| Feature Category | Features | Optimal Lookback | Performance | Stability | Information | Composite | Best Feature | Method |
|------------------|----------|------------------|-------------|-----------|-------------|-----------|--------------|--------|
| Momentum | 30 | 15 | 0.313 | 0.877 | 0.595 | 0.522 | N/A | cv |
| Trend | 25 | 15 | 0.328 | 0.891 | 0.609 | 0.543 | N/A | cv |
| Volatility | 48 | 20 | 0.326 | 0.885 | 0.605 | 0.536 | N/A | cv |
| Volume | 31 | 5 | 0.337 | 0.881 | 0.609 | 0.536 | N/A | cv |

**Column Descriptions:**
- **Features**: Number of features optimized in this category
- **Optimal Lookback**: Best lookback period across all features in category
- **Performance**: Average performance score (higher is better)
- **Stability**: Average stability across market regimes (higher is better)
- **Information**: Average information content (non-redundancy)
- **Composite**: Stability × Information (quality metric for feature weighting)
- **Best Feature**: Top performing feature in this category
- **Method**: Optimization method used (cv=cross-validation)

### Feature Category Optimization

Summary of optimization results by category with all key metrics.

| Category | Features | Optimal Lookback | Lookback Range | Performance | Stability | Information | Composite | Success Rate |
|----------|----------|------------------|----------------|-------------|-----------|-------------|-----------|-------------|
| momentum | 30 | 15 | 1-50 | 0.313 | 0.877 | 0.595 | 0.522 | 100.0% |
| volatility | 48 | 20 | 1-50 | 0.326 | 0.885 | 0.605 | 0.536 | 100.0% |
| trend | 25 | 15 | 1-50 | 0.328 | 0.891 | 0.609 | 0.543 | 100.0% |
| volume | 31 | 5 | 1-50 | 0.337 | 0.881 | 0.609 | 0.536 | 100.0% |

**Column Descriptions:**
- **Features**: Total features in category
- **Optimal Lookback**: Best performing lookback period
- **Lookback Range**: Range of lookback periods tested
- **Performance**: Average cross-validated performance score
- **Stability**: Average stability across different market conditions
- **Information**: Non-redundancy / unique information content
- **Composite**: Combined quality score (Stability × Information)
- **Success Rate**: Percentage of features successfully optimized

### Stability Analysis

| Metric | Value |
|--------|-------|
| Overall Stability | 0.833 |
| Short-term Stability | 0.790 |
| Medium-term Stability | 0.835 |
| Long-term Stability | 0.875 |
| Stability Variance | 0.001 |

### Performance Analysis

Cross-validated performance metrics across all optimized features.

| Metric | Value | Description |
|--------|-------|-------------|
| Average Performance | 0.801 | Mean cross-validation score across all features |
| Best Performance | 0.867 | Highest performing feature's CV score |
| Worst Performance | 0.740 | Lowest performing feature's CV score |
| Performance Range | 0.127 | Difference between best and worst (diversity metric) |
| Performance Std | 0.052 | Standard deviation of performance scores |

**Understanding Performance Metrics:**

- **Average Performance**: Indicates overall feature quality. Higher values (>0.70) suggest strong predictive features.
- **Best Performance**: Shows the ceiling of feature quality. Values >0.85 indicate excellent features.
- **Worst Performance**: Identifies weakest features. Values <0.60 may need review or removal.
- **Performance Range**: Large ranges (>0.20) suggest diverse feature quality; consider feature selection.
- **Performance Std**: High std (>0.10) indicates inconsistent feature quality across categories.

**Performance Metric Calculation:**

Performance scores are computed using:
1. **Cross-Validation**: K-fold CV (typically 2-5 folds) to assess generalization
2. **Information Criterion**: Measures feature's unique information content
3. **Stability Score**: Consistency across different market regimes
4. **Final Score**: Weighted combination of CV score, information, and stability

**Quality Thresholds:**
- **Excellent** (≥0.85): High-quality features for model training
- **Good** (0.70-0.85): Solid features, suitable for most models
- **Acceptable** (0.60-0.70): May be useful but require validation
- **Poor** (<0.60): Consider excluding or investigating for issues

### Learnability and Validation Diagnostics

These diagnostics evaluate whether the optimized features are stable, statistically significant, and free of obvious leakage or data issues that could hurt downstream model learnability.

#### Walk-Forward Stability

- **Stable features across windows:** 254/254 (100.0% if total_features > 0 else 0.0)
Features that remain predictive across rolling windows are more likely to generalize to unseen periods.

#### Null/Shuffle Test

- **Features beating shuffled baseline (p<0.05):** 0/254 (0.0% if null_total > 0 else 0.0)
- **Null distribution mean |corr|:** nan
- **Null distribution std:** nan
If only a small fraction of features beat the shuffled-label baseline, most correlations are likely noise rather than learnable signal.

#### Index Alignment Audit

- **Potential issues detected:** 0
High counts may indicate NaN-heavy or suspiciously aligned features, which can hide lookahead bias or data leaks.

#### FDR-Corrected Significance

- **Status:** Skipped (no per-feature p-values available)

#### Metric Definition and Target Type

- **Target type:** binary
- **Binary target:** True
This confirms whether the target behaves as expected (binary vs continuous) and flags obvious definition issues.

#### Label Balance

- **Minority class ratio:** 0.500
- **Balanced classes:** True
- **Recommendation:** balanced
Severe class imbalance reduces effective learnability and often requires stratified sampling or reweighting.

#### Multicollinearity Check

- **Highly correlated feature pairs (|r| > 0.95):** 0
- **Recommendation:** monitor
Strongly collinear features can inflate variance and reduce the effective dimensionality of learnable signal.

#### Stability Computation

- **Status:** verified
- **Note:** Ensure stability computed on non-overlapping time folds
Confirms that stability metrics are computed on non-overlapping time folds, which is critical for realistic learnability estimates.

### Individual Feature Analysis by Category

#### Momentum Features

**Category Summary:**
- **Best Individual Feature Lookback:** 15
- **Average Performance Score:** 0.313
- **Average Stability Score:** 0.877
- **Features Optimized:** 30

#### Trend Features

**Category Summary:**
- **Best Individual Feature Lookback:** 15
- **Average Performance Score:** 0.328
- **Average Stability Score:** 0.891
- **Features Optimized:** 25

#### Volatility Features

**Category Summary:**
- **Best Individual Feature Lookback:** 20
- **Average Performance Score:** 0.326
- **Average Stability Score:** 0.885
- **Features Optimized:** 48

#### Volume Features

**Category Summary:**
- **Best Individual Feature Lookback:** 5
- **Average Performance Score:** 0.337
- **Average Stability Score:** 0.881
- **Features Optimized:** 31

### Per-Feature Optimization Results

| Feature Name | Optimal Lookback | Performance | Stability | Method |
|--------------|------------------|-------------|-----------|--------|
| acceleration_features | 5 | 0.267 | 0.880 | N/A |
| advanced_cumulative_returns_10 | 10 | 0.324 | 0.874 | N/A |
| advanced_cumulative_returns_20 | 15 | 0.371 | 0.871 | N/A |
| advanced_support_resistance_features | 20 | 0.361 | 0.910 | N/A |
| adx_14_returns_vwap | 8 | 0.315 | 0.892 | N/A |
| apo_12_26_returns_vwap | 15 | 0.354 | 0.867 | N/A |
| ar_1_coefficients_20 | 15 | 0.312 | 0.887 | N/A |
| aroon_25_returns_vwap | 15 | 0.337 | 0.862 | N/A |
| candlestick_dark_cloud_cover_pattern | 20 | 0.130 | 0.931 | N/A |
| candlestick_doji_pattern | 20 | 0.202 | 0.930 | N/A |
| candlestick_engulfing_pattern | 20 | 0.144 | 0.926 | N/A |
| candlestick_hammer_pattern | 20 | 0.066 | 0.945 | N/A |
| candlestick_harami_cross_pattern | 20 | 0.196 | 0.944 | N/A |
| candlestick_piercing_line_pattern | 20 | 0.120 | 0.935 | N/A |
| candlestick_three_black_crows_pattern | 20 | 0.198 | 0.924 | N/A |
| candlestick_three_white_soldiers_pattern | 20 | 0.204 | 0.929 | N/A |
| cci_20_returns_vwap | 10 | 0.324 | 0.870 | N/A |
| cmf_20 | 15 | 0.301 | 0.879 | N/A |
| cmo_14_returns_vwap | 8 | 0.336 | 0.881 | N/A |
| cumulative_returns_10_price_returns | 4 | 0.270 | 0.876 | N/A |
| cumulative_returns_20_price_returns | 5 | 0.267 | 0.878 | N/A |
| cycle_length | 20 | 0.253 | 0.935 | N/A |
| dfa_slopes | 4 | 0.272 | 0.878 | N/A |
| directional_signal | 15 | 0.385 | 0.865 | N/A |
| donchian_channel_20 | 15 | 0.337 | 0.878 | N/A |
| entropy_rate_20 | 8 | 0.301 | 0.878 | N/A |
| fibonacci_0.236_10_price_returns | 8 | 0.315 | 0.882 | N/A |
| fibonacci_0.236_20_price_returns | 15 | 0.347 | 0.895 | N/A |
| fibonacci_0.236_5_price_returns | 5 | 0.291 | 0.875 | N/A |
| fibonacci_0.382_10_price_returns | 8 | 0.310 | 0.875 | N/A |
| fibonacci_0.382_20_price_returns | 15 | 0.336 | 0.874 | N/A |
| fibonacci_0.382_5_price_returns | 5 | 0.294 | 0.872 | N/A |
| fibonacci_0.5_10_price_returns | 8 | 0.307 | 0.874 | N/A |
| fibonacci_0.5_20_price_returns | 15 | 0.335 | 0.870 | N/A |
| fibonacci_0.5_5_price_returns | 5 | 0.291 | 0.875 | N/A |
| fibonacci_0.618_10_price_returns | 8 | 0.306 | 0.880 | N/A |
| fibonacci_0.618_20_price_returns | 15 | 0.336 | 0.875 | N/A |
| fibonacci_0.618_5_price_returns | 5 | 0.292 | 0.879 | N/A |
| fibonacci_0.786_10_price_returns | 8 | 0.309 | 0.882 | N/A |
| fibonacci_0.786_20_price_returns | 15 | 0.345 | 0.919 | N/A |
| fibonacci_0.786_5_price_returns | 5 | 0.289 | 0.881 | N/A |
| fractal_dimension | 3 | 0.000 | 0.500 | N/A |
| hurst_exponent | 10 | 0.298 | 0.881 | N/A |
| kama_30_2_30_returns_vwap | 20 | 0.380 | 0.883 | N/A |
| kst_10_15_20_30_10_10_10_15_returns_vwap | 20 | 0.355 | 0.900 | N/A |
| lempel_ziv_complexity_20 | 4 | 0.257 | 0.887 | N/A |
| ljung_box_pvalue_20_10 | 5 | 0.280 | 0.872 | N/A |
| log_returns_10_price_returns | 5 | 0.267 | 0.882 | N/A |
| log_returns_1_price_returns | 5 | 0.259 | 0.889 | N/A |
| log_returns_5_price_returns | 5 | 0.266 | 0.879 | N/A |
| mama_21_0.05_price_returns | 15 | 0.354 | 0.878 | N/A |
| max_drawdown | 10 | 0.358 | 0.878 | N/A |
| order_flow_imbalance_20 | 15 | 0.335 | 0.856 | N/A |
| pfe_12_returns_vwap | 20 | 0.126 | 0.987 | N/A |
| resistance_level_1_10_price_returns | 8 | 0.318 | 0.879 | N/A |
| resistance_level_1_5_price_returns | 8 | 0.291 | 0.875 | N/A |
| resistance_level_3_20_price_returns | 15 | 0.347 | 0.879 | N/A |
| returns_kurtosis_20_price_returns | 10 | 0.310 | 0.861 | N/A |
| returns_skewness_20_price_returns | 15 | 0.308 | 0.857 | N/A |
| roc_14_price_returns | 5 | 0.270 | 0.878 | N/A |
| roc_21_price_returns | 5 | 0.269 | 0.877 | N/A |
| roc_30_price_returns | 5 | 0.270 | 0.872 | N/A |
| rolling_returns_10_price_returns | 5 | 0.270 | 0.877 | N/A |
| rolling_returns_20_price_returns | 4 | 0.268 | 0.878 | N/A |
| rolling_zscore_returns_20 | 8 | 0.270 | 0.879 | N/A |
| shannon_entropy_20_10 | 10 | 0.313 | 0.880 | N/A |
| sharpe_ratio_20_0.0_price_returns | 15 | 0.318 | 0.879 | N/A |
| simple_returns_10_price_returns | 4 | 0.270 | 0.873 | N/A |
| simple_returns_1_price_returns | 5 | 0.264 | 0.890 | N/A |
| simple_returns_5_price_returns | 4 | 0.270 | 0.880 | N/A |
| spectral_entropy_20 | 8 | 0.292 | 0.879 | N/A |
| stochastic_14_3_price_returns | 8 | 0.281 | 0.878 | N/A |
| stochastic_21_3_price_returns | 8 | 0.285 | 0.884 | N/A |
| stochastic_30_3_price_returns | 8 | 0.281 | 0.870 | N/A |
| stochastic_kd_14_3 | 5 | 0.314 | 0.875 | N/A |
| support_level_2_5_price_returns | 5 | 0.290 | 0.882 | N/A |
| support_level_4_10_price_returns | 10 | 0.321 | 0.896 | N/A |
| support_level_4_20_price_returns | 15 | 0.347 | 0.882 | N/A |
| t3_14_0.7_returns_vwap | 15 | 0.381 | 0.870 | N/A |
| ultimate_oscillator_7_14_28_returns_vwap | 20 | 0.353 | 0.900 | N/A |
| vectorbt_acceleration_10_price_returns | 5 | 0.270 | 0.883 | N/A |
| vectorbt_acceleration_5_price_returns | 5 | 0.269 | 0.877 | N/A |
| vectorbt_acceleration_consistency_10_10_price_returns | 10 | 0.312 | 0.869 | N/A |
| vectorbt_acceleration_consistency_10_20_price_returns | 5 | 0.294 | 0.878 | N/A |
| vectorbt_acceleration_consistency_5_10_price_returns | 4 | 0.274 | 0.880 | N/A |
| vectorbt_acceleration_consistency_5_20_price_returns | 5 | 0.276 | 0.871 | N/A |
| vectorbt_acceleration_correlation_20_price_returns | 4 | 0.006 | 0.982 | N/A |
| vectorbt_acceleration_divergence_20_price_returns | 4 | 0.270 | 0.880 | N/A |
| vectorbt_acceleration_regime_5_10_price_returns | 20 | 0.159 | 0.933 | N/A |
| vectorbt_acceleration_regime_5_20_price_returns | 20 | 0.162 | 0.929 | N/A |
| vectorbt_adx_14 | 20 | 0.324 | 0.872 | N/A |
| vectorbt_adx_21 | 15 | 0.330 | 0.865 | N/A |
| vectorbt_adx_9 | 8 | 0.304 | 0.870 | N/A |
| vectorbt_bbands_10_1.5 | 5 | 0.299 | 0.868 | N/A |
| vectorbt_bbands_14_1.5 | 5 | 0.310 | 0.866 | N/A |
| vectorbt_bbands_20_1.5 | 15 | 0.333 | 0.868 | N/A |
| vectorbt_ichimoku_cloud_9_26_52 | 20 | 0.394 | 0.882 | N/A |
| vectorbt_jerk_10_price_returns | 5 | 0.266 | 0.881 | N/A |
| vectorbt_jerk_5_price_returns | 5 | 0.270 | 0.878 | N/A |
| vectorbt_parabolic_sar_0.02_0.2 | 8 | 0.318 | 0.867 | N/A |
| vectorbt_parabolic_sar_0.02_0.3 | 8 | 0.318 | 0.867 | N/A |
| vectorbt_parabolic_sar_0.05_0.2 | 5 | 0.296 | 0.874 | N/A |
| vectorbt_parabolic_sar_0.05_0.3 | 5 | 0.294 | 0.878 | N/A |
| vectorbt_parabolic_sar_0.1_0.2 | 5 | 0.284 | 0.876 | N/A |
| vectorbt_parabolic_sar_0.1_0.3 | 5 | 0.276 | 0.881 | N/A |
| vectorbt_zigzag_10.0_2 | 8 | 0.273 | 0.888 | N/A |
| vectorbt_zigzag_3.0_2 | 8 | 0.293 | 0.898 | N/A |
| vectorbt_zigzag_5.0_2 | 8 | 0.279 | 0.877 | N/A |
| vectorbt_zigzag_7.0_2 | 8 | 0.278 | 0.886 | N/A |
| vwap_deviations_20 | 15 | 0.349 | 0.876 | N/A |
| wavelet_energy | 20 | 0.352 | 0.879 | N/A |
| wma_20_price_returns | 15 | 0.353 | 0.876 | N/A |
| advanced_momentum_10_30 | 15 | 0.353 | 0.879 | N/A |
| advanced_momentum_5_20 | 10 | 0.353 | 0.873 | N/A |
| macd_12_26_9_returns_vwap | 10 | 0.367 | 0.867 | N/A |
| macd_delta_12_26_9 | 8 | 0.334 | 0.871 | N/A |
| macd_entropy_20_12_26 | 20 | 0.242 | 0.920 | N/A |
| momentum_14_price_returns | 5 | 0.271 | 0.883 | N/A |
| momentum_21_price_returns | 5 | 0.269 | 0.880 | N/A |
| momentum_30_price_returns | 5 | 0.269 | 0.879 | N/A |
| momentum_endpoints_sma_20 | 15 | 0.349 | 0.876 | N/A |
| momentum_features | 15 | 0.371 | 0.871 | N/A |
| rsi_14_returns_vwap | 8 | 0.336 | 0.881 | N/A |
| rsi_21_returns_vwap | 10 | 0.343 | 0.878 | N/A |
| rsi_30_returns_vwap | 10 | 0.351 | 0.872 | N/A |
| rsi_zscore_14_20 | 8 | 0.304 | 0.879 | N/A |
| vectorbt_acceleration_momentum_10_10_price_returns | 8 | 0.299 | 0.881 | N/A |
| vectorbt_acceleration_momentum_10_20_price_returns | 8 | 0.314 | 0.880 | N/A |
| vectorbt_acceleration_momentum_5_10_price_returns | 5 | 0.287 | 0.868 | N/A |
| vectorbt_acceleration_momentum_5_20_price_returns | 8 | 0.286 | 0.871 | N/A |
| vectorbt_momentum_50_price_returns | 5 | 0.270 | 0.881 | N/A |
| vectorbt_momentum_5_price_returns | 4 | 0.270 | 0.878 | N/A |
| vectorbt_momentum_acceleration_10_10_price_returns | 5 | 0.291 | 0.877 | N/A |
| vectorbt_momentum_acceleration_10_20_price_returns | 8 | 0.303 | 0.876 | N/A |
| vectorbt_momentum_acceleration_5_10_price_returns | 5 | 0.280 | 0.873 | N/A |
| vectorbt_momentum_acceleration_5_20_price_returns | 5 | 0.275 | 0.882 | N/A |
| vectorbt_momentum_comprehensive_14 | 15 | 0.337 | 0.876 | N/A |
| vectorbt_momentum_comprehensive_21 | 15 | 0.358 | 0.876 | N/A |
| vectorbt_momentum_comprehensive_30 | 15 | 0.370 | 0.870 | N/A |
| vectorbt_momentum_comprehensive_9 | 5 | 0.321 | 0.870 | N/A |
| volume_momentum_10 | 10 | 0.324 | 0.874 | N/A |
| volume_momentum_5 | 5 | 0.299 | 0.876 | N/A |
| analyst_volume_pressure | 5 | 0.267 | 0.892 | N/A |
| analyst_volume_trend | 15 | 0.369 | 0.868 | N/A |
| price_volume_oscillator_10_20 | 15 | 0.300 | 0.884 | N/A |
| price_volume_oscillator_5_15 | 8 | 0.301 | 0.878 | N/A |
| vectorbt_enhanced_obv_10 | 20 | 0.162 | 0.970 | N/A |
| vectorbt_enhanced_obv_20 | 20 | 0.392 | 0.888 | N/A |
| vectorbt_smoothed_obv_20 | 20 | 0.371 | 0.882 | N/A |
| vectorbt_volume_acceleration_5_volume_returns | 5 | 0.264 | 0.887 | N/A |
| vectorbt_volume_weighted_ad_line_10 | 20 | 0.382 | 0.898 | N/A |
| vectorbt_volume_weighted_ad_line_50 | 15 | 0.364 | 0.868 | N/A |
| volume_accumulation_distribution | 20 | 0.356 | 0.854 | N/A |
| volume_ema_5 | 20 | 0.385 | 0.880 | N/A |
| volume_ema_50 | 20 | 0.360 | 0.876 | N/A |
| volume_oscillator_10_20 | 15 | 0.362 | 0.869 | N/A |
| volume_oscillator_5_15 | 10 | 0.337 | 0.876 | N/A |
| volume_percentile_100 | 15 | 0.385 | 0.886 | N/A |
| volume_percentile_20 | 15 | 0.330 | 0.868 | N/A |
| volume_percentile_50 | 15 | 0.379 | 0.878 | N/A |
| volume_price_correlation_10 | 5 | 0.275 | 0.878 | N/A |
| volume_price_correlation_20 | 4 | 0.280 | 0.879 | N/A |
| volume_price_divergence_10 | 8 | 0.301 | 0.871 | N/A |
| volume_price_divergence_20 | 8 | 0.317 | 0.870 | N/A |
| volume_price_trend | 20 | 0.384 | 0.877 | N/A |
| volume_ratio_10 | 5 | 0.314 | 0.872 | N/A |
| volume_ratio_50 | 15 | 0.391 | 0.883 | N/A |
| volume_roc_1 | 5 | 0.272 | 0.872 | N/A |
| volume_roc_5 | 5 | 0.299 | 0.876 | N/A |
| volume_trend_strength_10_30 | 15 | 0.383 | 0.864 | N/A |
| volume_trend_strength_20_50 | 20 | 0.368 | 0.896 | N/A |
| volume_vwap_50 | 15 | 0.391 | 0.883 | N/A |
| volume_zscore_60_252 | 20 | 0.395 | 0.880 | N/A |
| band_limited_volatility | 20 | 0.353 | 0.894 | N/A |
| enhanced_volatility_100 | 20 | 0.369 | 0.903 | N/A |
| enhanced_volatility_14 | 10 | 0.322 | 0.887 | N/A |
| enhanced_volatility_30 | 15 | 0.351 | 0.865 | N/A |
| returns_volatility_20_price_returns | 15 | 0.329 | 0.878 | N/A |
| vectorbt_acceleration_volatility_10_10_price_returns | 15 | 0.334 | 0.884 | N/A |
| vectorbt_acceleration_volatility_10_20_price_returns | 15 | 0.305 | 0.880 | N/A |
| vectorbt_acceleration_volatility_5_10_price_returns | 10 | 0.288 | 0.886 | N/A |
| vectorbt_acceleration_volatility_5_20_price_returns | 5 | 0.284 | 0.869 | N/A |
| vectorbt_atr_10 | 20 | 0.339 | 0.885 | N/A |
| vectorbt_atr_14 | 20 | 0.350 | 0.887 | N/A |
| vectorbt_atr_20 | 20 | 0.359 | 0.880 | N/A |
| vectorbt_atr_30 | 20 | 0.367 | 0.873 | N/A |
| vectorbt_atr_50 | 20 | 0.374 | 0.870 | N/A |
| vectorbt_garman_klass_volatility_10 | 15 | 0.338 | 0.899 | N/A |
| vectorbt_garman_klass_volatility_14 | 20 | 0.350 | 0.895 | N/A |
| vectorbt_garman_klass_volatility_20 | 20 | 0.362 | 0.895 | N/A |
| vectorbt_garman_klass_volatility_30 | 20 | 0.370 | 0.905 | N/A |
| vectorbt_garman_klass_volatility_50 | 20 | 0.376 | 0.906 | N/A |
| vectorbt_parkinson_volatility_20 | 20 | 0.362 | 0.901 | N/A |
| vectorbt_parkinson_volatility_30 | 20 | 0.369 | 0.909 | N/A |
| vectorbt_parkinson_volatility_50 | 20 | 0.374 | 0.910 | N/A |
| vectorbt_rogers_satchell_volatility_10 | 15 | 0.338 | 0.896 | N/A |
| vectorbt_rogers_satchell_volatility_14 | 20 | 0.348 | 0.892 | N/A |
| vectorbt_rogers_satchell_volatility_20 | 20 | 0.361 | 0.891 | N/A |
| vectorbt_rogers_satchell_volatility_30 | 20 | 0.369 | 0.904 | N/A |
| vectorbt_rogers_satchell_volatility_50 | 20 | 0.375 | 0.904 | N/A |
| vectorbt_volatility_acceleration_5_20_price_returns | 4 | 0.275 | 0.877 | N/A |
| vectorbt_volatility_comprehensive_10 | 4 | 0.271 | 0.879 | N/A |
| vectorbt_volatility_comprehensive_14 | 5 | 0.278 | 0.879 | N/A |
| vectorbt_volatility_comprehensive_20 | 8 | 0.280 | 0.876 | N/A |
| vectorbt_volatility_comprehensive_30 | 10 | 0.295 | 0.892 | N/A |
| vectorbt_volatility_comprehensive_50 | 15 | 0.322 | 0.881 | N/A |
| vectorbt_yang_zhang_volatility_10 | 15 | 0.339 | 0.904 | N/A |
| vectorbt_yang_zhang_volatility_14 | 20 | 0.349 | 0.898 | N/A |
| volatility_expansion_10_3 | 5 | 0.269 | 0.873 | N/A |
| volatility_expansion_10_5 | 5 | 0.277 | 0.869 | N/A |
| volatility_expansion_10_7 | 5 | 0.286 | 0.872 | N/A |
| volatility_expansion_14_3 | 8 | 0.267 | 0.867 | N/A |
| volatility_expansion_14_5 | 8 | 0.274 | 0.859 | N/A |
| volatility_expansion_14_7 | 8 | 0.285 | 0.871 | N/A |
| volatility_expansion_20_3 | 5 | 0.268 | 0.860 | N/A |
| volatility_expansion_20_5 | 5 | 0.276 | 0.866 | N/A |
| volatility_expansion_20_7 | 8 | 0.287 | 0.873 | N/A |
| volume_std_10 | 8 | 0.314 | 0.885 | N/A |
| volume_std_20 | 15 | 0.331 | 0.877 | N/A |
| volume_std_50 | 20 | 0.365 | 0.895 | N/A |
| volume_volatility_elasticity_20 | 20 | 0.312 | 0.875 | N/A |
| dema_21_price_returns | 15 | 0.324 | 0.870 | N/A |
| ema_12_returns_vwap | 15 | 0.381 | 0.865 | N/A |
| ema_26_returns_vwap | 20 | 0.379 | 0.881 | N/A |
| ema_50_returns_vwap | 20 | 0.380 | 0.890 | N/A |
| sma_100_returns_vwap | 20 | 0.368 | 0.901 | N/A |
| sma_10_returns_vwap | 15 | 0.376 | 0.864 | N/A |
| sma_20_returns_vwap | 15 | 0.360 | 0.887 | N/A |
| sma_50_returns_vwap | 20 | 0.375 | 0.908 | N/A |
| sma_5_returns_vwap | 15 | 0.379 | 0.864 | N/A |
| tema_21_price_returns | 5 | 0.313 | 0.869 | N/A |
| trend_score_14 | 15 | 0.376 | 0.867 | N/A |
| vectorbt_acceleration_trend_strength_10_10_price_returns | 3 | 0.266 | 0.880 | N/A |
| vectorbt_acceleration_trend_strength_10_20_price_returns | 4 | 0.264 | 0.887 | N/A |
| vectorbt_acceleration_trend_strength_5_10_price_returns | 5 | 0.267 | 0.884 | N/A |
| vectorbt_acceleration_trend_strength_5_20_price_returns | 4 | 0.263 | 0.875 | N/A |
| vectorbt_ema_100 | 20 | 0.367 | 0.882 | N/A |
| vectorbt_trend_comprehensive_100 | 20 | 0.382 | 0.887 | N/A |
| vectorbt_trend_consistency_10_price_returns | 15 | 0.301 | 0.947 | N/A |
| vectorbt_trend_consistency_20_price_returns | 20 | 0.328 | 0.955 | N/A |
| vectorbt_trend_consistency_50_price_returns | 20 | 0.336 | 0.949 | N/A |
| vectorbt_trend_consistency_5_price_returns | 8 | 0.275 | 0.951 | N/A |
| vectorbt_trend_strength_10_price_returns | 5 | 0.269 | 0.882 | N/A |
| vectorbt_trend_strength_20_price_returns | 5 | 0.290 | 0.874 | N/A |
| vectorbt_trend_strength_50_price_returns | 8 | 0.307 | 0.870 | N/A |
| vectorbt_trend_strength_5_price_returns | 5 | 0.270 | 0.882 | N/A |

### Optimization Recommendations

#### Recommended Actions
- Review feature selection for better performance
- Monitor lookback performance across different market regimes
- Consider adaptive lookback periods based on volatility
- Validate optimization results with out-of-sample testing

#### Lookback Optimization Strategy
- **Short-term Lookback:** 10
- **Medium-term Lookback:** 30
- **Long-term Lookback:** 200
- **Optimization Method:** data_driven_cross_validation

## Metrics

- **Lookback Periods Tested:** N/A
- **Best Momentum Features:** N/A
- **Best Trend Features:** N/A
- **Best Volatility Features:** N/A
- **Best Volume Features:** N/A
- **Best Oscillator Features:** N/A
- **Best Acceleration Features:** N/A
- **Best Order Flow Features:** N/A
- **Best Advanced Statistical Features:** N/A
- **Best Spectral Wavelet Features:** N/A
- **Best Candlestick Pattern Features:** N/A
- **Best Returns Features:** N/A
- **Best Support Resistance Features:** N/A
- **Best Entropy Features:** N/A
- **Execution Mode:** N/A
- **Success:** False

## Comprehensive Lookback Optimization by Category

Each feature gets 1 optimal lookback period + 2 informative & non-redundant alternatives:

**Total Features Optimized:** 246 features across 5 categories

### Other Features

Optimized 112 features with optimal + alternative lookback periods:

#### acceleration_features
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.267
- **Stability Score:** 0.880
- **Optimization Method:** per_feature_mi_curve

#### advanced_cumulative_returns_10
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 15, 20]
- **Performance Score:** 0.324
- **Stability Score:** 0.874
- **Optimization Method:** per_feature_mi_curve

#### advanced_cumulative_returns_20
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.371
- **Stability Score:** 0.871
- **Optimization Method:** per_feature_mi_curve

#### advanced_support_resistance_features
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [4, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 4, 15]
- **Performance Score:** 0.361
- **Stability Score:** 0.910
- **Optimization Method:** per_feature_mi_curve

#### adx_14_returns_vwap
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.315
- **Stability Score:** 0.892
- **Optimization Method:** per_feature_mi_curve

#### apo_12_26_returns_vwap
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.354
- **Stability Score:** 0.867
- **Optimization Method:** per_feature_mi_curve

#### ar_1_coefficients_20
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.312
- **Stability Score:** 0.887
- **Optimization Method:** per_feature_mi_curve

#### aroon_25_returns_vwap
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.337
- **Stability Score:** 0.862
- **Optimization Method:** per_feature_mi_curve

#### candlestick_dark_cloud_cover_pattern
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.130
- **Stability Score:** 0.931
- **Optimization Method:** per_feature_mi_curve

#### candlestick_doji_pattern
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.202
- **Stability Score:** 0.930
- **Optimization Method:** per_feature_mi_curve

#### candlestick_engulfing_pattern
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.144
- **Stability Score:** 0.926
- **Optimization Method:** per_feature_mi_curve

#### candlestick_hammer_pattern
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.066
- **Stability Score:** 0.945
- **Optimization Method:** per_feature_mi_curve

#### candlestick_harami_cross_pattern
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.196
- **Stability Score:** 0.944
- **Optimization Method:** per_feature_mi_curve

#### candlestick_piercing_line_pattern
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.120
- **Stability Score:** 0.935
- **Optimization Method:** per_feature_mi_curve

#### candlestick_three_black_crows_pattern
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.198
- **Stability Score:** 0.924
- **Optimization Method:** per_feature_mi_curve

#### candlestick_three_white_soldiers_pattern
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.204
- **Stability Score:** 0.929
- **Optimization Method:** per_feature_mi_curve

#### cci_20_returns_vwap
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 15, 20]
- **Performance Score:** 0.324
- **Stability Score:** 0.870
- **Optimization Method:** per_feature_mi_curve

#### cmf_20
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 8, 20]
- **Performance Score:** 0.301
- **Stability Score:** 0.879
- **Optimization Method:** per_feature_mi_curve

#### cmo_14_returns_vwap
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.336
- **Stability Score:** 0.881
- **Optimization Method:** per_feature_mi_curve

#### cumulative_returns_10_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.270
- **Stability Score:** 0.876
- **Optimization Method:** per_feature_mi_curve

#### cumulative_returns_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.267
- **Stability Score:** 0.878
- **Optimization Method:** per_feature_mi_curve

#### cycle_length
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.253
- **Stability Score:** 0.935
- **Optimization Method:** per_feature_mi_curve

#### dfa_slopes
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.272
- **Stability Score:** 0.878
- **Optimization Method:** per_feature_mi_curve

#### directional_signal
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.385
- **Stability Score:** 0.865
- **Optimization Method:** per_feature_mi_curve

#### donchian_channel_20
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.337
- **Stability Score:** 0.878
- **Optimization Method:** per_feature_mi_curve

#### entropy_rate_20
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.301
- **Stability Score:** 0.878
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.236_10_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.315
- **Stability Score:** 0.882
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.236_20_price_returns
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.347
- **Stability Score:** 0.895
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.236_5_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.291
- **Stability Score:** 0.875
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.382_10_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.310
- **Stability Score:** 0.875
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.382_20_price_returns
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.336
- **Stability Score:** 0.874
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.382_5_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.294
- **Stability Score:** 0.872
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.5_10_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.307
- **Stability Score:** 0.874
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.5_20_price_returns
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.335
- **Stability Score:** 0.870
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.5_5_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.291
- **Stability Score:** 0.875
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.618_10_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.306
- **Stability Score:** 0.880
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.618_20_price_returns
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.336
- **Stability Score:** 0.875
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.618_5_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.292
- **Stability Score:** 0.879
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.786_10_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.309
- **Stability Score:** 0.882
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.786_20_price_returns
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 20]
- **Performance Score:** 0.345
- **Stability Score:** 0.919
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.786_5_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.289
- **Stability Score:** 0.881
- **Optimization Method:** per_feature_mi_curve

#### fractal_dimension
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [4, 5] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 4, 5]
- **Performance Score:** 0.000
- **Stability Score:** 0.500
- **Optimization Method:** per_feature_mi_curve

#### hurst_exponent
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 15, 20]
- **Performance Score:** 0.298
- **Stability Score:** 0.881
- **Optimization Method:** per_feature_mi_curve

#### kama_30_2_30_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.380
- **Stability Score:** 0.883
- **Optimization Method:** per_feature_mi_curve

#### kst_10_15_20_30_10_10_10_15_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.355
- **Stability Score:** 0.900
- **Optimization Method:** per_feature_mi_curve

#### lempel_ziv_complexity_20
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.257
- **Stability Score:** 0.887
- **Optimization Method:** per_feature_mi_curve

#### ljung_box_pvalue_20_10
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.280
- **Stability Score:** 0.872
- **Optimization Method:** per_feature_mi_curve

#### log_returns_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.267
- **Stability Score:** 0.882
- **Optimization Method:** per_feature_mi_curve

#### log_returns_1_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.259
- **Stability Score:** 0.889
- **Optimization Method:** per_feature_mi_curve

#### log_returns_5_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.266
- **Stability Score:** 0.879
- **Optimization Method:** per_feature_mi_curve

#### mama_21_0.05_price_returns
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 20]
- **Performance Score:** 0.354
- **Stability Score:** 0.878
- **Optimization Method:** per_feature_mi_curve

#### max_drawdown
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 15, 20]
- **Performance Score:** 0.358
- **Stability Score:** 0.878
- **Optimization Method:** per_feature_mi_curve

#### order_flow_imbalance_20
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.335
- **Stability Score:** 0.856
- **Optimization Method:** per_feature_mi_curve

#### pfe_12_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.126
- **Stability Score:** 0.987
- **Optimization Method:** per_feature_mi_curve

#### resistance_level_1_10_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.318
- **Stability Score:** 0.879
- **Optimization Method:** per_feature_mi_curve

#### resistance_level_1_5_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.291
- **Stability Score:** 0.875
- **Optimization Method:** per_feature_mi_curve

#### resistance_level_3_20_price_returns
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 8, 20]
- **Performance Score:** 0.347
- **Stability Score:** 0.879
- **Optimization Method:** per_feature_mi_curve

#### returns_kurtosis_20_price_returns
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 15, 20]
- **Performance Score:** 0.310
- **Stability Score:** 0.861
- **Optimization Method:** per_feature_mi_curve

#### returns_skewness_20_price_returns
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [4, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 4, 20]
- **Performance Score:** 0.308
- **Stability Score:** 0.857
- **Optimization Method:** per_feature_mi_curve

#### roc_14_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.270
- **Stability Score:** 0.878
- **Optimization Method:** per_feature_mi_curve

#### roc_21_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.269
- **Stability Score:** 0.877
- **Optimization Method:** per_feature_mi_curve

#### roc_30_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.270
- **Stability Score:** 0.872
- **Optimization Method:** per_feature_mi_curve

#### rolling_returns_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.270
- **Stability Score:** 0.877
- **Optimization Method:** per_feature_mi_curve

#### rolling_returns_20_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.268
- **Stability Score:** 0.878
- **Optimization Method:** per_feature_mi_curve

#### rolling_zscore_returns_20
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [3, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 3, 20]
- **Performance Score:** 0.270
- **Stability Score:** 0.879
- **Optimization Method:** per_feature_mi_curve

#### shannon_entropy_20_10
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 15, 20]
- **Performance Score:** 0.313
- **Stability Score:** 0.880
- **Optimization Method:** per_feature_mi_curve

#### sharpe_ratio_20_0.0_price_returns
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 20]
- **Performance Score:** 0.318
- **Stability Score:** 0.879
- **Optimization Method:** per_feature_mi_curve

#### simple_returns_10_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.270
- **Stability Score:** 0.873
- **Optimization Method:** per_feature_mi_curve

#### simple_returns_1_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.264
- **Stability Score:** 0.890
- **Optimization Method:** per_feature_mi_curve

#### simple_returns_5_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.270
- **Stability Score:** 0.880
- **Optimization Method:** per_feature_mi_curve

#### spectral_entropy_20
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.292
- **Stability Score:** 0.879
- **Optimization Method:** per_feature_mi_curve

#### stochastic_14_3_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.281
- **Stability Score:** 0.878
- **Optimization Method:** per_feature_mi_curve

#### stochastic_21_3_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.285
- **Stability Score:** 0.884
- **Optimization Method:** per_feature_mi_curve

#### stochastic_30_3_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.281
- **Stability Score:** 0.870
- **Optimization Method:** per_feature_mi_curve

#### stochastic_kd_14_3
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.314
- **Stability Score:** 0.875
- **Optimization Method:** per_feature_mi_curve

#### support_level_2_5_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.290
- **Stability Score:** 0.882
- **Optimization Method:** per_feature_mi_curve

#### support_level_4_10_price_returns
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 15, 20]
- **Performance Score:** 0.321
- **Stability Score:** 0.896
- **Optimization Method:** per_feature_mi_curve

#### support_level_4_20_price_returns
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 8, 20]
- **Performance Score:** 0.347
- **Stability Score:** 0.882
- **Optimization Method:** per_feature_mi_curve

#### t3_14_0.7_returns_vwap
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 8, 20]
- **Performance Score:** 0.381
- **Stability Score:** 0.870
- **Optimization Method:** per_feature_mi_curve

#### ultimate_oscillator_7_14_28_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.353
- **Stability Score:** 0.900
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.270
- **Stability Score:** 0.883
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_5_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.269
- **Stability Score:** 0.877
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_consistency_10_10_price_returns
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 15, 20]
- **Performance Score:** 0.312
- **Stability Score:** 0.869
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_consistency_10_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 15, 20]
- **Performance Score:** 0.294
- **Stability Score:** 0.878
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_consistency_5_10_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.274
- **Stability Score:** 0.880
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_consistency_5_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.276
- **Stability Score:** 0.871
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_correlation_20_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [3, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 3, 20]
- **Performance Score:** 0.006
- **Stability Score:** 0.982
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_divergence_20_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.270
- **Stability Score:** 0.880
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_regime_5_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.159
- **Stability Score:** 0.933
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_regime_5_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.162
- **Stability Score:** 0.929
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_adx_14
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.324
- **Stability Score:** 0.872
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_adx_21
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 20]
- **Performance Score:** 0.330
- **Stability Score:** 0.865
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_adx_9
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.304
- **Stability Score:** 0.870
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_bbands_10_1.5
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.299
- **Stability Score:** 0.868
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_bbands_14_1.5
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.310
- **Stability Score:** 0.866
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_bbands_20_1.5
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.333
- **Stability Score:** 0.868
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_ichimoku_cloud_9_26_52
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.394
- **Stability Score:** 0.882
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_jerk_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.266
- **Stability Score:** 0.881
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_jerk_5_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.270
- **Stability Score:** 0.878
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.02_0.2
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.318
- **Stability Score:** 0.867
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.02_0.3
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.318
- **Stability Score:** 0.867
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.05_0.2
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.296
- **Stability Score:** 0.874
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.05_0.3
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.294
- **Stability Score:** 0.878
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.1_0.2
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.284
- **Stability Score:** 0.876
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.1_0.3
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.276
- **Stability Score:** 0.881
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_zigzag_10.0_2
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.273
- **Stability Score:** 0.888
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_zigzag_3.0_2
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.293
- **Stability Score:** 0.898
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_zigzag_5.0_2
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.279
- **Stability Score:** 0.877
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_zigzag_7.0_2
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.278
- **Stability Score:** 0.886
- **Optimization Method:** per_feature_mi_curve

#### vwap_deviations_20
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.349
- **Stability Score:** 0.876
- **Optimization Method:** per_feature_mi_curve

#### wavelet_energy
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.352
- **Stability Score:** 0.879
- **Optimization Method:** per_feature_mi_curve

#### wma_20_price_returns
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.353
- **Stability Score:** 0.876
- **Optimization Method:** per_feature_mi_curve

### Momentum Features

Optimized 30 features with optimal + alternative lookback periods:

#### advanced_momentum_10_30
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.353
- **Stability Score:** 0.879
- **Optimization Method:** per_feature_mi_curve

#### advanced_momentum_5_20
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 15, 20]
- **Performance Score:** 0.353
- **Stability Score:** 0.873
- **Optimization Method:** per_feature_mi_curve

#### macd_12_26_9_returns_vwap
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 15, 20]
- **Performance Score:** 0.367
- **Stability Score:** 0.867
- **Optimization Method:** per_feature_mi_curve

#### macd_delta_12_26_9
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.334
- **Stability Score:** 0.871
- **Optimization Method:** per_feature_mi_curve

#### macd_entropy_20_12_26
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.242
- **Stability Score:** 0.920
- **Optimization Method:** per_feature_mi_curve

#### momentum_14_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.271
- **Stability Score:** 0.883
- **Optimization Method:** per_feature_mi_curve

#### momentum_21_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.269
- **Stability Score:** 0.880
- **Optimization Method:** per_feature_mi_curve

#### momentum_30_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.269
- **Stability Score:** 0.879
- **Optimization Method:** per_feature_mi_curve

#### momentum_endpoints_sma_20
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.349
- **Stability Score:** 0.876
- **Optimization Method:** per_feature_mi_curve

#### momentum_features
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.371
- **Stability Score:** 0.871
- **Optimization Method:** per_feature_mi_curve

#### rsi_14_returns_vwap
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.336
- **Stability Score:** 0.881
- **Optimization Method:** per_feature_mi_curve

#### rsi_21_returns_vwap
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 15, 20]
- **Performance Score:** 0.343
- **Stability Score:** 0.878
- **Optimization Method:** per_feature_mi_curve

#### rsi_30_returns_vwap
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 15, 20]
- **Performance Score:** 0.351
- **Stability Score:** 0.872
- **Optimization Method:** per_feature_mi_curve

#### rsi_zscore_14_20
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.304
- **Stability Score:** 0.879
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_momentum_10_10_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [4, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 4, 20]
- **Performance Score:** 0.299
- **Stability Score:** 0.881
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_momentum_10_20_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.314
- **Stability Score:** 0.880
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_momentum_5_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.287
- **Stability Score:** 0.868
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_momentum_5_20_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [3, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 3, 20]
- **Performance Score:** 0.286
- **Stability Score:** 0.871
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_50_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.270
- **Stability Score:** 0.881
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_5_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.270
- **Stability Score:** 0.878
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_acceleration_10_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.291
- **Stability Score:** 0.877
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_acceleration_10_20_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [4, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 4, 20]
- **Performance Score:** 0.303
- **Stability Score:** 0.876
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_acceleration_5_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.280
- **Stability Score:** 0.873
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_acceleration_5_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.275
- **Stability Score:** 0.882
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_comprehensive_14
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.337
- **Stability Score:** 0.876
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_comprehensive_21
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.358
- **Stability Score:** 0.876
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_comprehensive_30
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.370
- **Stability Score:** 0.870
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_comprehensive_9
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.321
- **Stability Score:** 0.870
- **Optimization Method:** per_feature_mi_curve

#### volume_momentum_10
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 15, 20]
- **Performance Score:** 0.324
- **Stability Score:** 0.874
- **Optimization Method:** per_feature_mi_curve

#### volume_momentum_5
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.299
- **Stability Score:** 0.876
- **Optimization Method:** per_feature_mi_curve

### Volume Features

Optimized 31 features with optimal + alternative lookback periods:

#### analyst_volume_pressure
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.267
- **Stability Score:** 0.892
- **Optimization Method:** per_feature_mi_curve

#### analyst_volume_trend
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.369
- **Stability Score:** 0.868
- **Optimization Method:** per_feature_mi_curve

#### price_volume_oscillator_10_20
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [4, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 4, 10]
- **Performance Score:** 0.300
- **Stability Score:** 0.884
- **Optimization Method:** per_feature_mi_curve

#### price_volume_oscillator_5_15
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.301
- **Stability Score:** 0.878
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_enhanced_obv_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.162
- **Stability Score:** 0.970
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_enhanced_obv_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.392
- **Stability Score:** 0.888
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_smoothed_obv_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.371
- **Stability Score:** 0.882
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volume_acceleration_5_volume_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.264
- **Stability Score:** 0.887
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volume_weighted_ad_line_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.382
- **Stability Score:** 0.898
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volume_weighted_ad_line_50
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 20]
- **Performance Score:** 0.364
- **Stability Score:** 0.868
- **Optimization Method:** per_feature_mi_curve

#### volume_accumulation_distribution
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.356
- **Stability Score:** 0.854
- **Optimization Method:** per_feature_mi_curve

#### volume_ema_5
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.385
- **Stability Score:** 0.880
- **Optimization Method:** per_feature_mi_curve

#### volume_ema_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.360
- **Stability Score:** 0.876
- **Optimization Method:** per_feature_mi_curve

#### volume_oscillator_10_20
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.362
- **Stability Score:** 0.869
- **Optimization Method:** per_feature_mi_curve

#### volume_oscillator_5_15
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 15, 20]
- **Performance Score:** 0.337
- **Stability Score:** 0.876
- **Optimization Method:** per_feature_mi_curve

#### volume_percentile_100
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 20]
- **Performance Score:** 0.385
- **Stability Score:** 0.886
- **Optimization Method:** per_feature_mi_curve

#### volume_percentile_20
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.330
- **Stability Score:** 0.868
- **Optimization Method:** per_feature_mi_curve

#### volume_percentile_50
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 20]
- **Performance Score:** 0.379
- **Stability Score:** 0.878
- **Optimization Method:** per_feature_mi_curve

#### volume_price_correlation_10
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.275
- **Stability Score:** 0.878
- **Optimization Method:** per_feature_mi_curve

#### volume_price_correlation_20
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.280
- **Stability Score:** 0.879
- **Optimization Method:** per_feature_mi_curve

#### volume_price_divergence_10
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.301
- **Stability Score:** 0.871
- **Optimization Method:** per_feature_mi_curve

#### volume_price_divergence_20
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.317
- **Stability Score:** 0.870
- **Optimization Method:** per_feature_mi_curve

#### volume_price_trend
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.384
- **Stability Score:** 0.877
- **Optimization Method:** per_feature_mi_curve

#### volume_ratio_10
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.314
- **Stability Score:** 0.872
- **Optimization Method:** per_feature_mi_curve

#### volume_ratio_50
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 20]
- **Performance Score:** 0.391
- **Stability Score:** 0.883
- **Optimization Method:** per_feature_mi_curve

#### volume_roc_1
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.272
- **Stability Score:** 0.872
- **Optimization Method:** per_feature_mi_curve

#### volume_roc_5
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.299
- **Stability Score:** 0.876
- **Optimization Method:** per_feature_mi_curve

#### volume_trend_strength_10_30
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 8, 20]
- **Performance Score:** 0.383
- **Stability Score:** 0.864
- **Optimization Method:** per_feature_mi_curve

#### volume_trend_strength_20_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.368
- **Stability Score:** 0.896
- **Optimization Method:** per_feature_mi_curve

#### volume_vwap_50
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 20]
- **Performance Score:** 0.391
- **Stability Score:** 0.883
- **Optimization Method:** per_feature_mi_curve

#### volume_zscore_60_252
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.395
- **Stability Score:** 0.880
- **Optimization Method:** per_feature_mi_curve

### Volatility Features

Optimized 48 features with optimal + alternative lookback periods:

#### band_limited_volatility
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.353
- **Stability Score:** 0.894
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_100
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.369
- **Stability Score:** 0.903
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_14
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 15, 20]
- **Performance Score:** 0.322
- **Stability Score:** 0.887
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_30
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 20]
- **Performance Score:** 0.351
- **Stability Score:** 0.865
- **Optimization Method:** per_feature_mi_curve

#### returns_volatility_20_price_returns
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.329
- **Stability Score:** 0.878
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_volatility_10_10_price_returns
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.334
- **Stability Score:** 0.884
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_volatility_10_20_price_returns
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [4, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 4, 20]
- **Performance Score:** 0.305
- **Stability Score:** 0.880
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_volatility_5_10_price_returns
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [3, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 3, 20]
- **Performance Score:** 0.288
- **Stability Score:** 0.886
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_volatility_5_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.284
- **Stability Score:** 0.869
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.339
- **Stability Score:** 0.885
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_14
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.350
- **Stability Score:** 0.887
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.359
- **Stability Score:** 0.880
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_30
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.367
- **Stability Score:** 0.873
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.374
- **Stability Score:** 0.870
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_10
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 20]
- **Performance Score:** 0.338
- **Stability Score:** 0.899
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_14
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.350
- **Stability Score:** 0.895
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.362
- **Stability Score:** 0.895
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_30
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.370
- **Stability Score:** 0.905
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.376
- **Stability Score:** 0.906
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parkinson_volatility_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.362
- **Stability Score:** 0.901
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parkinson_volatility_30
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.369
- **Stability Score:** 0.909
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parkinson_volatility_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.374
- **Stability Score:** 0.910
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_10
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 20]
- **Performance Score:** 0.338
- **Stability Score:** 0.896
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_14
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.348
- **Stability Score:** 0.892
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.361
- **Stability Score:** 0.891
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_30
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.369
- **Stability Score:** 0.904
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.375
- **Stability Score:** 0.904
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_acceleration_5_20_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.275
- **Stability Score:** 0.877
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_10
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.271
- **Stability Score:** 0.879
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_14
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.278
- **Stability Score:** 0.879
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_20
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [3, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 3, 20]
- **Performance Score:** 0.280
- **Stability Score:** 0.876
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_30
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [3, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 3, 20]
- **Performance Score:** 0.295
- **Stability Score:** 0.892
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_50
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [4, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 4, 10]
- **Performance Score:** 0.322
- **Stability Score:** 0.881
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_yang_zhang_volatility_10
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 20]
- **Performance Score:** 0.339
- **Stability Score:** 0.904
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_yang_zhang_volatility_14
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.349
- **Stability Score:** 0.898
- **Optimization Method:** per_feature_mi_curve

#### volatility_expansion_10_3
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.269
- **Stability Score:** 0.873
- **Optimization Method:** per_feature_mi_curve

#### volatility_expansion_10_5
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.277
- **Stability Score:** 0.869
- **Optimization Method:** per_feature_mi_curve

#### volatility_expansion_10_7
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.286
- **Stability Score:** 0.872
- **Optimization Method:** per_feature_mi_curve

#### volatility_expansion_14_3
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [3, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 3, 20]
- **Performance Score:** 0.267
- **Stability Score:** 0.867
- **Optimization Method:** per_feature_mi_curve

#### volatility_expansion_14_5
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.274
- **Stability Score:** 0.859
- **Optimization Method:** per_feature_mi_curve

#### volatility_expansion_14_7
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.285
- **Stability Score:** 0.871
- **Optimization Method:** per_feature_mi_curve

#### volatility_expansion_20_3
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.268
- **Stability Score:** 0.860
- **Optimization Method:** per_feature_mi_curve

#### volatility_expansion_20_5
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.276
- **Stability Score:** 0.866
- **Optimization Method:** per_feature_mi_curve

#### volatility_expansion_20_7
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.287
- **Stability Score:** 0.873
- **Optimization Method:** per_feature_mi_curve

#### volume_std_10
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.314
- **Stability Score:** 0.885
- **Optimization Method:** per_feature_mi_curve

#### volume_std_20
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 20]
- **Performance Score:** 0.331
- **Stability Score:** 0.877
- **Optimization Method:** per_feature_mi_curve

#### volume_std_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.365
- **Stability Score:** 0.895
- **Optimization Method:** per_feature_mi_curve

#### volume_volatility_elasticity_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.312
- **Stability Score:** 0.875
- **Optimization Method:** per_feature_mi_curve

### Trend Features

Optimized 25 features with optimal + alternative lookback periods:

#### dema_21_price_returns
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [4, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 4, 10]
- **Performance Score:** 0.324
- **Stability Score:** 0.870
- **Optimization Method:** per_feature_mi_curve

#### ema_12_returns_vwap
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 8, 20]
- **Performance Score:** 0.381
- **Stability Score:** 0.865
- **Optimization Method:** per_feature_mi_curve

#### ema_26_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.379
- **Stability Score:** 0.881
- **Optimization Method:** per_feature_mi_curve

#### ema_50_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.380
- **Stability Score:** 0.890
- **Optimization Method:** per_feature_mi_curve

#### sma_100_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.368
- **Stability Score:** 0.901
- **Optimization Method:** per_feature_mi_curve

#### sma_10_returns_vwap
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 8, 20]
- **Performance Score:** 0.376
- **Stability Score:** 0.864
- **Optimization Method:** per_feature_mi_curve

#### sma_20_returns_vwap
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 20]
- **Performance Score:** 0.360
- **Stability Score:** 0.887
- **Optimization Method:** per_feature_mi_curve

#### sma_50_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.375
- **Stability Score:** 0.908
- **Optimization Method:** per_feature_mi_curve

#### sma_5_returns_vwap
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 8, 20]
- **Performance Score:** 0.379
- **Stability Score:** 0.864
- **Optimization Method:** per_feature_mi_curve

#### tema_21_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.313
- **Stability Score:** 0.869
- **Optimization Method:** per_feature_mi_curve

#### trend_score_14
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.376
- **Stability Score:** 0.867
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_trend_strength_10_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 10, 20]
- **Performance Score:** 0.266
- **Stability Score:** 0.880
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_trend_strength_10_20_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.264
- **Stability Score:** 0.887
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_trend_strength_5_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.267
- **Stability Score:** 0.884
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_trend_strength_5_20_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.263
- **Stability Score:** 0.875
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_ema_100
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.367
- **Stability Score:** 0.882
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_comprehensive_100
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.382
- **Stability Score:** 0.887
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_consistency_10_price_returns
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [4, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 4, 20]
- **Performance Score:** 0.301
- **Stability Score:** 0.947
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_consistency_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.328
- **Stability Score:** 0.955
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_consistency_50_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.336
- **Stability Score:** 0.949
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_consistency_5_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.275
- **Stability Score:** 0.951
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_strength_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.269
- **Stability Score:** 0.882
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_strength_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.290
- **Stability Score:** 0.874
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_strength_50_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.307
- **Stability Score:** 0.870
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_strength_5_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.270
- **Stability Score:** 0.882
- **Optimization Method:** per_feature_mi_curve

### Optimization Strategy

- **Strategy:** 1 optimal + 2 alternatives per feature
- **Redundancy Check:** Alternatives must be ≥3 periods apart
- **Score Threshold:** Alternatives must have ≥70% of optimal score
- **Optimization Method:** per_feature_mi_curve
- **VectorBT Available:** False
- **Vectorization Manager Available:** False

## Next Steps

- Use optimized lookback periods in subsequent feature generation
- Apply optimized lookbacks to feature generation step
- Use selected optimal features for model training
- Consider regime-aware lookback adaptation for different market conditions
- Validate lookback performance with out-of-sample testing

