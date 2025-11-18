# Lookback Optimization Report

**Generated:** 2025-11-18 08:21:54
**Step:** feature_generation_period_lookback_optimization_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** blank

## Optimization Results

- **Momentum Lookback:** 5
- **Trend Lookback:** 5
- **Volatility Lookback:** 5
- **Volume Lookback:** 5
- **Optimization Score:** 0.74

## Comprehensive Optimization Analysis

### Data Export

- **Per-Feature Metrics CSV:** `outcomes/feature_generation_period_lookback_optimization_step_per_feature_metrics_20251118_082154.csv`
- **Full Path:** `/Users/remyroche/Documents/Ares/outcomes/feature_generation_period_lookback_optimization_step_per_feature_metrics_20251118_082154.csv`

### Optimization Performance Metrics

| Metric | Value |
|--------|-------|
| Optimization Method | data_driven_cross_validation |
| Total Features Analyzed | 268 |
| Lookback Range Tested | 1-50 |
| Cross-Validation Folds | 2 |
| Optimization Efficiency | 85.0% |
| Stability Score | 0.957 |
| Performance Score | 0.548 |

### Global Optimization Metrics

| Metric | Value |
|--------|-------|
| Total Features Optimized | 261 |
| Categories Processed | 5 |
| Average Lookback Period | 4.8 |
| Lookback Range | 1-50 |
| Step Size | 1 |
| Cross-Validation Folds | 2 |
| Total Optimization Time | N/A seconds |
| Memory Usage | N/A MB |
| Success Rate | N/A |

### Individual Feature Optimization Results

This table shows detailed optimization results for each feature category.

| Feature Category | Features | Optimal Lookback | Performance | Stability | Information | Composite | Best Feature | Method |
|------------------|----------|------------------|-------------|-----------|-------------|-----------|--------------|--------|
| Momentum | 31 | 5 | 0.539 | 0.951 | 0.745 | 0.708 | N/A | cv |
| Trend | 24 | 5 | 0.559 | 0.959 | 0.759 | 0.728 | N/A | cv |
| Volatility | 42 | 5 | 0.574 | 0.958 | 0.766 | 0.734 | N/A | cv |
| Volume | 49 | 5 | 0.522 | 0.960 | 0.741 | 0.712 | N/A | cv |

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
| momentum | 31 | 5 | 1-50 | 0.539 | 0.951 | 0.745 | 0.708 | 100.0% |
| volatility | 42 | 5 | 1-50 | 0.574 | 0.958 | 0.766 | 0.734 | 100.0% |
| trend | 24 | 5 | 1-50 | 0.559 | 0.959 | 0.759 | 0.728 | 100.0% |
| volume | 49 | 5 | 1-50 | 0.522 | 0.960 | 0.741 | 0.712 | 100.0% |

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

- **Stable features across windows:** 267/267 (100.0% if total_features > 0 else 0.0)
Features that remain predictive across rolling windows are more likely to generalize to unseen periods.

#### Null/Shuffle Test

- **Features beating shuffled baseline (p<0.05):** 0/267 (0.0% if null_total > 0 else 0.0)
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
- **Warning:** imbalanced_classes
This confirms whether the target behaves as expected (binary vs continuous) and flags obvious definition issues.

#### Label Balance

- **Minority class ratio:** 0.194
- **Balanced classes:** False
- **Recommendation:** use_stratified_folds
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
- **Best Individual Feature Lookback:** 5
- **Average Performance Score:** 0.539
- **Average Stability Score:** 0.951
- **Features Optimized:** 31

#### Trend Features

**Category Summary:**
- **Best Individual Feature Lookback:** 5
- **Average Performance Score:** 0.559
- **Average Stability Score:** 0.959
- **Features Optimized:** 24

#### Volatility Features

**Category Summary:**
- **Best Individual Feature Lookback:** 5
- **Average Performance Score:** 0.574
- **Average Stability Score:** 0.958
- **Features Optimized:** 42

#### Volume Features

**Category Summary:**
- **Best Individual Feature Lookback:** 5
- **Average Performance Score:** 0.522
- **Average Stability Score:** 0.960
- **Features Optimized:** 49

### Per-Feature Optimization Results

| Feature Name | Optimal Lookback | Performance | Stability | Method |
|--------------|------------------|-------------|-----------|--------|
| acceleration_features | 4 | 0.482 | 0.898 | N/A |
| advanced_cumulative_returns_10 | 5 | 0.571 | 0.929 | N/A |
| advanced_cumulative_returns_20 | 5 | 0.602 | 0.963 | N/A |
| adx_14_returns_vwap | 5 | 0.571 | 0.948 | N/A |
| apo_12_26_returns_vwap | 5 | 0.582 | 0.950 | N/A |
| ar_1_coefficients_20 | 5 | 0.560 | 0.957 | N/A |
| aroon_25_returns_vwap | 5 | 0.575 | 0.969 | N/A |
| candlestick_dark_cloud_cover_pattern | 20 | 0.207 | 0.990 | N/A |
| candlestick_doji_pattern | 15 | 0.294 | 0.977 | N/A |
| candlestick_engulfing_pattern | 15 | 0.205 | 0.963 | N/A |
| candlestick_harami_cross_pattern | 20 | 0.276 | 0.906 | N/A |
| candlestick_piercing_line_pattern | 20 | 0.196 | 0.962 | N/A |
| candlestick_three_black_crows_pattern | 20 | 0.288 | 0.925 | N/A |
| candlestick_three_white_soldiers_pattern | 15 | 0.293 | 0.966 | N/A |
| cci_20_returns_vwap | 4 | 0.537 | 0.956 | N/A |
| cmf_20 | 5 | 0.531 | 0.988 | N/A |
| cumulative_returns_10_price_returns | 4 | 0.502 | 0.981 | N/A |
| cumulative_returns_20_price_returns | 5 | 0.487 | 0.947 | N/A |
| cycle_length | 15 | 0.418 | 0.985 | N/A |
| dfa_slopes | 4 | 0.503 | 0.994 | N/A |
| directional_signal | 5 | 0.622 | 0.932 | N/A |
| donchian_channel_20 | 5 | 0.577 | 0.944 | N/A |
| entropy_rate_20 | 5 | 0.535 | 0.957 | N/A |
| fibonacci_0.236_10_price_returns | 5 | 0.562 | 0.926 | N/A |
| fibonacci_0.236_20_price_returns | 5 | 0.577 | 0.942 | N/A |
| fibonacci_0.236_5_price_returns | 4 | 0.522 | 0.922 | N/A |
| fibonacci_0.382_10_price_returns | 5 | 0.547 | 0.934 | N/A |
| fibonacci_0.382_20_price_returns | 5 | 0.575 | 0.957 | N/A |
| fibonacci_0.382_5_price_returns | 4 | 0.515 | 0.926 | N/A |
| fibonacci_0.5_10_price_returns | 5 | 0.544 | 0.922 | N/A |
| fibonacci_0.5_20_price_returns | 5 | 0.565 | 0.957 | N/A |
| fibonacci_0.5_5_price_returns | 4 | 0.517 | 0.926 | N/A |
| fibonacci_0.618_10_price_returns | 4 | 0.548 | 0.948 | N/A |
| fibonacci_0.618_20_price_returns | 5 | 0.571 | 0.971 | N/A |
| fibonacci_0.618_5_price_returns | 5 | 0.504 | 0.937 | N/A |
| fibonacci_0.786_10_price_returns | 5 | 0.555 | 0.949 | N/A |
| fibonacci_0.786_20_price_returns | 5 | 0.558 | 0.962 | N/A |
| fibonacci_0.786_5_price_returns | 4 | 0.507 | 0.963 | N/A |
| fractal_dimension | 8 | 0.003 | 0.994 | N/A |
| hurst_exponent | 4 | 0.540 | 0.959 | N/A |
| kama_30_2_30_returns_vwap | 5 | 0.619 | 0.971 | N/A |
| kst_10_15_20_30_10_10_10_15_returns_vwap | 5 | 0.592 | 0.970 | N/A |
| lempel_ziv_complexity_20 | 3 | 0.460 | 0.958 | N/A |
| ljung_box_pvalue_20_10 | 4 | 0.502 | 0.938 | N/A |
| log_returns_10_price_returns | 4 | 0.490 | 0.966 | N/A |
| log_returns_1_price_returns | 4 | 0.467 | 0.962 | N/A |
| log_returns_5_price_returns | 5 | 0.474 | 0.943 | N/A |
| mama_21_0.05_price_returns | 5 | 0.583 | 0.938 | N/A |
| max_drawdown | 5 | 0.585 | 0.924 | N/A |
| order_flow_imbalance_20 | 5 | 0.584 | 0.954 | N/A |
| pfe_12_returns_vwap | 5 | 0.549 | 0.968 | N/A |
| pivot_point_5_price_returns | 4 | 0.504 | 0.924 | N/A |
| resistance_level_1_20_price_returns | 5 | 0.585 | 1.000 | N/A |
| resistance_level_3_10_price_returns | 5 | 0.548 | 0.940 | N/A |
| resistance_level_3_5_price_returns | 5 | 0.510 | 0.908 | N/A |
| returns_kurtosis_20_price_returns | 5 | 0.569 | 0.955 | N/A |
| returns_skewness_20_price_returns | 5 | 0.565 | 0.944 | N/A |
| roc_14_price_returns | 4 | 0.498 | 0.970 | N/A |
| roc_21_price_returns | 3 | 0.467 | 0.960 | N/A |
| roc_30_price_returns | 4 | 0.485 | 0.950 | N/A |
| rolling_returns_10_price_returns | 4 | 0.502 | 0.984 | N/A |
| rolling_returns_20_price_returns | 4 | 0.487 | 0.953 | N/A |
| rolling_zscore_returns_20 | 5 | 0.491 | 0.995 | N/A |
| shannon_entropy_20_10 | 5 | 0.551 | 0.994 | N/A |
| sharpe_ratio_20_0.0_price_returns | 5 | 0.555 | 0.963 | N/A |
| simple_returns_10_price_returns | 4 | 0.501 | 0.971 | N/A |
| simple_returns_1_price_returns | 4 | 0.506 | 0.944 | N/A |
| simple_returns_5_price_returns | 4 | 0.492 | 0.961 | N/A |
| spectral_entropy_20 | 4 | 0.528 | 0.957 | N/A |
| stochastic_14_3_price_returns | 4 | 0.501 | 0.979 | N/A |
| stochastic_30_3_price_returns | 5 | 0.510 | 0.949 | N/A |
| stochastic_kd_14_3 | 5 | 0.537 | 0.919 | N/A |
| support_level_1_10_price_returns | 5 | 0.570 | 0.971 | N/A |
| support_level_1_20_price_returns | 5 | 0.572 | 0.929 | N/A |
| support_level_1_5_price_returns | 4 | 0.531 | 0.973 | N/A |
| t3_14_0.7_returns_vwap | 5 | 0.617 | 0.958 | N/A |
| ultimate_oscillator_7_14_28_returns_vwap | 5 | 0.589 | 0.967 | N/A |
| vectorbt_acceleration_10_price_returns | 4 | 0.500 | 0.984 | N/A |
| vectorbt_acceleration_5_price_returns | 3 | 0.500 | 0.955 | N/A |
| vectorbt_acceleration_consistency_10_10_price_returns | 5 | 0.578 | 0.994 | N/A |
| vectorbt_acceleration_consistency_10_20_price_returns | 4 | 0.562 | 0.969 | N/A |
| vectorbt_acceleration_consistency_5_10_price_returns | 4 | 0.522 | 0.968 | N/A |
| vectorbt_acceleration_consistency_5_20_price_returns | 4 | 0.513 | 0.984 | N/A |
| vectorbt_acceleration_correlation_20_price_returns | 3 | 0.039 | 0.908 | N/A |
| vectorbt_acceleration_divergence_20_price_returns | 4 | 0.503 | 0.976 | N/A |
| vectorbt_acceleration_regime_5_10_price_returns | 15 | 0.265 | 0.952 | N/A |
| vectorbt_acceleration_regime_5_20_price_returns | 20 | 0.228 | 0.944 | N/A |
| vectorbt_adx_14 | 5 | 0.573 | 0.996 | N/A |
| vectorbt_adx_21 | 5 | 0.589 | 0.969 | N/A |
| vectorbt_adx_9 | 4 | 0.545 | 0.985 | N/A |
| vectorbt_bbands_10_2.0 | 4 | 0.502 | 0.916 | N/A |
| vectorbt_bbands_14_1.5 | 5 | 0.538 | 0.935 | N/A |
| vectorbt_bbands_20_1.5 | 5 | 0.573 | 0.933 | N/A |
| vectorbt_enhanced_ad_line_20 | 5 | 0.571 | 0.954 | N/A |
| vectorbt_enhanced_ad_line_50 | 5 | 0.578 | 0.998 | N/A |
| vectorbt_ichimoku_cloud_9_26_52 | 5 | 0.618 | 0.941 | N/A |
| vectorbt_jerk_10_price_returns | 5 | 0.485 | 0.988 | N/A |
| vectorbt_jerk_5_price_returns | 4 | 0.487 | 0.890 | N/A |
| vectorbt_parabolic_sar_0.02_0.2 | 4 | 0.537 | 0.926 | N/A |
| vectorbt_parabolic_sar_0.02_0.3 | 4 | 0.536 | 0.926 | N/A |
| vectorbt_parabolic_sar_0.05_0.2 | 4 | 0.500 | 0.925 | N/A |
| vectorbt_parabolic_sar_0.05_0.3 | 4 | 0.501 | 0.921 | N/A |
| vectorbt_parabolic_sar_0.1_0.2 | 4 | 0.489 | 0.933 | N/A |
| vectorbt_parabolic_sar_0.1_0.3 | 4 | 0.487 | 0.951 | N/A |
| vectorbt_zigzag_10.0_2 | 5 | 0.480 | 0.908 | N/A |
| vectorbt_zigzag_3.0_2 | 5 | 0.544 | 0.994 | N/A |
| vectorbt_zigzag_5.0_2 | 4 | 0.493 | 0.948 | N/A |
| vectorbt_zigzag_7.0_2 | 5 | 0.505 | 0.952 | N/A |
| vwap_deviations_20 | 5 | 0.569 | 0.924 | N/A |
| vwma_20_price_returns | 5 | 0.596 | 0.952 | N/A |
| wavelet_energy | 5 | 0.578 | 0.977 | N/A |
| williams_r_21_price_returns | 4 | 0.513 | 0.952 | N/A |
| wma_20_price_returns | 5 | 0.589 | 0.938 | N/A |
| binary_label | 20 | 0.791 | 0.977 | N/A |
| meta_probability | 5 | 0.551 | 0.951 | N/A |
| advanced_momentum_10_30 | 5 | 0.589 | 0.958 | N/A |
| advanced_momentum_5_20 | 5 | 0.591 | 0.941 | N/A |
| macd_12_26_9_returns_vwap | 5 | 0.583 | 0.943 | N/A |
| macd_delta_12_26_9 | 5 | 0.567 | 0.918 | N/A |
| macd_entropy_20_12_26 | 10 | 0.450 | 0.958 | N/A |
| momentum_14_price_returns | 4 | 0.498 | 0.963 | N/A |
| momentum_21_price_returns | 4 | 0.506 | 0.973 | N/A |
| momentum_30_price_returns | 4 | 0.490 | 0.982 | N/A |
| momentum_endpoints_sma_20 | 5 | 0.586 | 0.936 | N/A |
| momentum_features | 5 | 0.601 | 0.963 | N/A |
| rsi_14_returns_vwap | 5 | 0.534 | 0.945 | N/A |
| rsi_21_returns_vwap | 5 | 0.550 | 0.983 | N/A |
| rsi_30_returns_vwap | 5 | 0.545 | 0.946 | N/A |
| rsi_zscore_14_20 | 5 | 0.543 | 0.975 | N/A |
| vectorbt_acceleration_momentum_10_10_price_returns | 4 | 0.533 | 0.928 | N/A |
| vectorbt_acceleration_momentum_10_20_price_returns | 5 | 0.559 | 0.956 | N/A |
| vectorbt_acceleration_momentum_5_10_price_returns | 4 | 0.495 | 0.924 | N/A |
| vectorbt_acceleration_momentum_5_20_price_returns | 5 | 0.523 | 0.937 | N/A |
| vectorbt_momentum_50_price_returns | 4 | 0.489 | 0.950 | N/A |
| vectorbt_momentum_5_price_returns | 4 | 0.506 | 0.972 | N/A |
| vectorbt_momentum_acceleration_10_10_price_returns | 4 | 0.536 | 0.947 | N/A |
| vectorbt_momentum_acceleration_10_20_price_returns | 5 | 0.527 | 0.934 | N/A |
| vectorbt_momentum_acceleration_5_10_price_returns | 4 | 0.512 | 0.948 | N/A |
| vectorbt_momentum_acceleration_5_20_price_returns | 4 | 0.506 | 0.964 | N/A |
| vectorbt_momentum_comprehensive_14 | 5 | 0.581 | 0.927 | N/A |
| vectorbt_momentum_comprehensive_21 | 5 | 0.597 | 0.931 | N/A |
| vectorbt_momentum_comprehensive_30 | 5 | 0.606 | 0.928 | N/A |
| vectorbt_momentum_comprehensive_9 | 5 | 0.562 | 0.928 | N/A |
| volume_momentum_10 | 4 | 0.522 | 0.975 | N/A |
| volume_momentum_20 | 5 | 0.529 | 0.987 | N/A |
| volume_momentum_5 | 4 | 0.497 | 0.958 | N/A |
| analyst_volume_pressure | 5 | 0.447 | 0.915 | N/A |
| analyst_volume_trend | 4 | 0.550 | 0.953 | N/A |
| price_volume_oscillator_10_20 | 4 | 0.536 | 0.962 | N/A |
| price_volume_oscillator_5_15 | 4 | 0.518 | 0.955 | N/A |
| vectorbt_enhanced_obv_10 | 5 | 0.560 | 0.994 | N/A |
| vectorbt_enhanced_obv_20 | 5 | 0.588 | 0.967 | N/A |
| vectorbt_enhanced_obv_50 | 5 | 0.577 | 0.990 | N/A |
| vectorbt_smoothed_obv_20 | 5 | 0.596 | 0.991 | N/A |
| vectorbt_volume_acceleration_5_volume_returns | 4 | 0.470 | 0.963 | N/A |
| vectorbt_volume_weighted_ad_line_10 | 5 | 0.556 | 0.947 | N/A |
| vectorbt_volume_weighted_ad_line_20 | 5 | 0.594 | 0.963 | N/A |
| vectorbt_volume_weighted_ad_line_50 | 5 | 0.605 | 0.998 | N/A |
| volume_accumulation_distribution | 5 | 0.563 | 0.947 | N/A |
| volume_ema_10 | 5 | 0.579 | 0.977 | N/A |
| volume_ema_20 | 5 | 0.605 | 0.972 | N/A |
| volume_ema_5 | 5 | 0.565 | 0.980 | N/A |
| volume_ema_50 | 5 | 0.593 | 0.961 | N/A |
| volume_entropy_10_volume_returns | 15 | 0.364 | 0.958 | N/A |
| volume_entropy_20_volume_returns | 20 | 0.319 | 0.917 | N/A |
| volume_entropy_5_volume_returns | 8 | 0.360 | 0.883 | N/A |
| volume_entropy_ma_10_10_volume_returns | 8 | 0.443 | 0.984 | N/A |
| volume_entropy_ma_10_5_volume_returns | 8 | 0.416 | 0.978 | N/A |
| volume_entropy_ma_20_10_volume_returns | 8 | 0.396 | 0.916 | N/A |
| volume_entropy_ma_20_5_volume_returns | 10 | 0.374 | 0.934 | N/A |
| volume_entropy_ma_5_10_volume_returns | 5 | 0.503 | 0.969 | N/A |
| volume_entropy_ma_5_5_volume_returns | 8 | 0.479 | 0.958 | N/A |
| volume_ma_ratios_20_10 | 4 | 0.511 | 0.950 | N/A |
| volume_oscillator_10_20 | 4 | 0.552 | 0.954 | N/A |
| volume_oscillator_5_15 | 5 | 0.539 | 0.975 | N/A |
| volume_percentile_100 | 5 | 0.527 | 0.945 | N/A |
| volume_percentile_20 | 4 | 0.516 | 0.941 | N/A |
| volume_percentile_50 | 5 | 0.523 | 0.949 | N/A |
| volume_price_correlation_10 | 5 | 0.553 | 0.959 | N/A |
| volume_price_correlation_20 | 5 | 0.590 | 0.969 | N/A |
| volume_price_divergence_10 | 4 | 0.509 | 0.960 | N/A |
| volume_price_divergence_20 | 4 | 0.512 | 0.936 | N/A |
| volume_price_trend | 5 | 0.593 | 0.999 | N/A |
| volume_ratio_10 | 4 | 0.501 | 0.962 | N/A |
| volume_ratio_20 | 4 | 0.511 | 0.949 | N/A |
| volume_ratio_50 | 4 | 0.525 | 0.959 | N/A |
| volume_roc_1 | 3 | 0.480 | 0.951 | N/A |
| volume_roc_10 | 4 | 0.532 | 0.969 | N/A |
| volume_roc_20 | 5 | 0.536 | 0.990 | N/A |
| volume_roc_5 | 4 | 0.495 | 0.965 | N/A |
| volume_trend_strength_10_30 | 5 | 0.572 | 0.953 | N/A |
| volume_trend_strength_20_50 | 5 | 0.598 | 0.969 | N/A |
| volume_vwap_10 | 5 | 0.615 | 0.999 | N/A |
| volume_vwap_50 | 5 | 0.600 | 0.977 | N/A |
| volume_zscore_60_252 | 5 | 0.531 | 0.945 | N/A |
| band_limited_volatility | 5 | 0.578 | 0.996 | N/A |
| enhanced_volatility_10 | 5 | 0.540 | 0.936 | N/A |
| enhanced_volatility_100 | 5 | 0.569 | 0.888 | N/A |
| enhanced_volatility_14 | 5 | 0.566 | 0.957 | N/A |
| enhanced_volatility_20 | 5 | 0.587 | 0.965 | N/A |
| enhanced_volatility_30 | 5 | 0.581 | 0.924 | N/A |
| enhanced_volatility_50 | 8 | 0.579 | 0.929 | N/A |
| returns_volatility_20_price_returns | 5 | 0.576 | 0.982 | N/A |
| vectorbt_acceleration_volatility_10_10_price_returns | 5 | 0.584 | 0.995 | N/A |
| vectorbt_acceleration_volatility_10_20_price_returns | 5 | 0.563 | 0.970 | N/A |
| vectorbt_acceleration_volatility_5_10_price_returns | 5 | 0.521 | 0.977 | N/A |
| vectorbt_acceleration_volatility_5_20_price_returns | 4 | 0.515 | 0.984 | N/A |
| vectorbt_atr_10 | 5 | 0.579 | 0.961 | N/A |
| vectorbt_atr_14 | 5 | 0.584 | 0.946 | N/A |
| vectorbt_atr_20 | 5 | 0.584 | 0.921 | N/A |
| vectorbt_atr_30 | 5 | 0.590 | 0.929 | N/A |
| vectorbt_atr_50 | 5 | 0.601 | 0.939 | N/A |
| vectorbt_garman_klass_volatility_10 | 5 | 0.583 | 0.964 | N/A |
| vectorbt_garman_klass_volatility_14 | 5 | 0.602 | 0.967 | N/A |
| vectorbt_garman_klass_volatility_20 | 5 | 0.602 | 0.950 | N/A |
| vectorbt_garman_klass_volatility_30 | 5 | 0.598 | 0.953 | N/A |
| vectorbt_garman_klass_volatility_50 | 5 | 0.608 | 0.976 | N/A |
| vectorbt_parkinson_volatility_14 | 5 | 0.599 | 0.966 | N/A |
| vectorbt_parkinson_volatility_20 | 5 | 0.599 | 0.950 | N/A |
| vectorbt_parkinson_volatility_30 | 5 | 0.599 | 0.962 | N/A |
| vectorbt_parkinson_volatility_50 | 5 | 0.610 | 0.976 | N/A |
| vectorbt_rogers_satchell_volatility_10 | 5 | 0.579 | 0.956 | N/A |
| vectorbt_rogers_satchell_volatility_14 | 5 | 0.604 | 0.969 | N/A |
| vectorbt_rogers_satchell_volatility_20 | 5 | 0.601 | 0.946 | N/A |
| vectorbt_rogers_satchell_volatility_30 | 5 | 0.597 | 0.960 | N/A |
| vectorbt_rogers_satchell_volatility_50 | 5 | 0.605 | 0.976 | N/A |
| vectorbt_volatility_acceleration_5_20_price_returns | 3 | 0.491 | 0.933 | N/A |
| vectorbt_volatility_comprehensive_10 | 4 | 0.474 | 0.926 | N/A |
| vectorbt_volatility_comprehensive_14 | 4 | 0.506 | 0.962 | N/A |
| vectorbt_volatility_comprehensive_20 | 4 | 0.517 | 0.947 | N/A |
| vectorbt_volatility_comprehensive_30 | 4 | 0.543 | 0.971 | N/A |
| vectorbt_volatility_comprehensive_50 | 5 | 0.567 | 0.961 | N/A |
| vectorbt_yang_zhang_volatility_10 | 5 | 0.585 | 0.976 | N/A |
| volume_std_10 | 5 | 0.576 | 0.964 | N/A |
| volume_std_20 | 5 | 0.599 | 0.999 | N/A |
| volume_std_50 | 5 | 0.589 | 0.959 | N/A |
| volume_volatility_elasticity_20 | 5 | 0.578 | 0.974 | N/A |
| dema_21_price_returns | 5 | 0.549 | 0.939 | N/A |
| ema_12_returns_vwap | 5 | 0.618 | 0.960 | N/A |
| ema_26_returns_vwap | 5 | 0.618 | 0.967 | N/A |
| ema_50_returns_vwap | 5 | 0.615 | 0.973 | N/A |
| sma_100_returns_vwap | 5 | 0.595 | 0.977 | N/A |
| sma_10_returns_vwap | 5 | 0.610 | 0.956 | N/A |
| sma_20_returns_vwap | 5 | 0.610 | 0.987 | N/A |
| sma_50_returns_vwap | 5 | 0.602 | 0.980 | N/A |
| sma_5_returns_vwap | 5 | 0.607 | 0.951 | N/A |
| tema_21_price_returns | 5 | 0.525 | 0.934 | N/A |
| trend_score_14 | 5 | 0.618 | 0.936 | N/A |
| vectorbt_acceleration_trend_strength_10_10_price_returns | 4 | 0.490 | 0.965 | N/A |
| vectorbt_acceleration_trend_strength_10_20_price_returns | 4 | 0.471 | 0.926 | N/A |
| vectorbt_acceleration_trend_strength_5_10_price_returns | 4 | 0.500 | 0.983 | N/A |
| vectorbt_acceleration_trend_strength_5_20_price_returns | 4 | 0.485 | 0.956 | N/A |
| vectorbt_sma_100 | 5 | 0.619 | 1.000 | N/A |
| vectorbt_trend_consistency_10_price_returns | 5 | 0.566 | 0.974 | N/A |
| vectorbt_trend_consistency_20_price_returns | 5 | 0.583 | 0.952 | N/A |
| vectorbt_trend_consistency_50_price_returns | 5 | 0.599 | 0.964 | N/A |
| vectorbt_trend_consistency_5_price_returns | 5 | 0.523 | 0.934 | N/A |
| vectorbt_trend_strength_10_price_returns | 4 | 0.492 | 0.936 | N/A |
| vectorbt_trend_strength_20_price_returns | 4 | 0.499 | 0.926 | N/A |
| vectorbt_trend_strength_50_price_returns | 4 | 0.523 | 0.950 | N/A |
| vectorbt_trend_strength_5_price_returns | 4 | 0.494 | 0.990 | N/A |

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

**Total Features Optimized:** 261 features across 5 categories

### Other Features

Optimized 115 features with optimal + alternative lookback periods:

#### acceleration_features
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.482
- **Stability Score:** 0.898
- **Optimization Method:** per_feature_mi_curve

#### advanced_cumulative_returns_10
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.571
- **Stability Score:** 0.929
- **Optimization Method:** per_feature_mi_curve

#### advanced_cumulative_returns_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.602
- **Stability Score:** 0.963
- **Optimization Method:** per_feature_mi_curve

#### adx_14_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.571
- **Stability Score:** 0.948
- **Optimization Method:** per_feature_mi_curve

#### apo_12_26_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.582
- **Stability Score:** 0.950
- **Optimization Method:** per_feature_mi_curve

#### ar_1_coefficients_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.560
- **Stability Score:** 0.957
- **Optimization Method:** per_feature_mi_curve

#### aroon_25_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.575
- **Stability Score:** 0.969
- **Optimization Method:** per_feature_mi_curve

#### candlestick_dark_cloud_cover_pattern
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.207
- **Stability Score:** 0.990
- **Optimization Method:** per_feature_mi_curve

#### candlestick_doji_pattern
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 8, 20]
- **Performance Score:** 0.294
- **Stability Score:** 0.977
- **Optimization Method:** per_feature_mi_curve

#### candlestick_engulfing_pattern
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 8, 20]
- **Performance Score:** 0.205
- **Stability Score:** 0.963
- **Optimization Method:** per_feature_mi_curve

#### candlestick_harami_cross_pattern
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.276
- **Stability Score:** 0.906
- **Optimization Method:** per_feature_mi_curve

#### candlestick_piercing_line_pattern
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.196
- **Stability Score:** 0.962
- **Optimization Method:** per_feature_mi_curve

#### candlestick_three_black_crows_pattern
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.288
- **Stability Score:** 0.925
- **Optimization Method:** per_feature_mi_curve

#### candlestick_three_white_soldiers_pattern
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 8, 20]
- **Performance Score:** 0.293
- **Stability Score:** 0.966
- **Optimization Method:** per_feature_mi_curve

#### cci_20_returns_vwap
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.537
- **Stability Score:** 0.956
- **Optimization Method:** per_feature_mi_curve

#### cmf_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 8, 20]
- **Performance Score:** 0.531
- **Stability Score:** 0.988
- **Optimization Method:** per_feature_mi_curve

#### cumulative_returns_10_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.502
- **Stability Score:** 0.981
- **Optimization Method:** per_feature_mi_curve

#### cumulative_returns_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [3, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 3, 20]
- **Performance Score:** 0.487
- **Stability Score:** 0.947
- **Optimization Method:** per_feature_mi_curve

#### cycle_length
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 10]
- **Performance Score:** 0.418
- **Stability Score:** 0.985
- **Optimization Method:** per_feature_mi_curve

#### dfa_slopes
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.503
- **Stability Score:** 0.994
- **Optimization Method:** per_feature_mi_curve

#### directional_signal
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.622
- **Stability Score:** 0.932
- **Optimization Method:** per_feature_mi_curve

#### donchian_channel_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.577
- **Stability Score:** 0.944
- **Optimization Method:** per_feature_mi_curve

#### entropy_rate_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.535
- **Stability Score:** 0.957
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.236_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.562
- **Stability Score:** 0.926
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.236_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.577
- **Stability Score:** 0.942
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.236_5_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.522
- **Stability Score:** 0.922
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.382_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.547
- **Stability Score:** 0.934
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.382_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.575
- **Stability Score:** 0.957
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.382_5_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.515
- **Stability Score:** 0.926
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.5_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.544
- **Stability Score:** 0.922
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.5_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.565
- **Stability Score:** 0.957
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.5_5_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.517
- **Stability Score:** 0.926
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.618_10_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.548
- **Stability Score:** 0.948
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.618_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.571
- **Stability Score:** 0.971
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.618_5_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.504
- **Stability Score:** 0.937
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.786_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.555
- **Stability Score:** 0.949
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.786_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.558
- **Stability Score:** 0.962
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.786_5_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.507
- **Stability Score:** 0.963
- **Optimization Method:** per_feature_mi_curve

#### fractal_dimension
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [4, 5] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 4, 5]
- **Performance Score:** 0.003
- **Stability Score:** 0.994
- **Optimization Method:** per_feature_mi_curve

#### hurst_exponent
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.540
- **Stability Score:** 0.959
- **Optimization Method:** per_feature_mi_curve

#### kama_30_2_30_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.619
- **Stability Score:** 0.971
- **Optimization Method:** per_feature_mi_curve

#### kst_10_15_20_30_10_10_10_15_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.592
- **Stability Score:** 0.970
- **Optimization Method:** per_feature_mi_curve

#### lempel_ziv_complexity_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 10, 20]
- **Performance Score:** 0.460
- **Stability Score:** 0.958
- **Optimization Method:** per_feature_mi_curve

#### ljung_box_pvalue_20_10
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.502
- **Stability Score:** 0.938
- **Optimization Method:** per_feature_mi_curve

#### log_returns_10_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.490
- **Stability Score:** 0.966
- **Optimization Method:** per_feature_mi_curve

#### log_returns_1_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.467
- **Stability Score:** 0.962
- **Optimization Method:** per_feature_mi_curve

#### log_returns_5_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 8, 20]
- **Performance Score:** 0.474
- **Stability Score:** 0.943
- **Optimization Method:** per_feature_mi_curve

#### mama_21_0.05_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.583
- **Stability Score:** 0.938
- **Optimization Method:** per_feature_mi_curve

#### max_drawdown
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.585
- **Stability Score:** 0.924
- **Optimization Method:** per_feature_mi_curve

#### order_flow_imbalance_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.584
- **Stability Score:** 0.954
- **Optimization Method:** per_feature_mi_curve

#### pfe_12_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.549
- **Stability Score:** 0.968
- **Optimization Method:** per_feature_mi_curve

#### pivot_point_5_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.504
- **Stability Score:** 0.924
- **Optimization Method:** per_feature_mi_curve

#### resistance_level_1_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.585
- **Stability Score:** 1.000
- **Optimization Method:** per_feature_mi_curve

#### resistance_level_3_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.548
- **Stability Score:** 0.940
- **Optimization Method:** per_feature_mi_curve

#### resistance_level_3_5_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.510
- **Stability Score:** 0.908
- **Optimization Method:** per_feature_mi_curve

#### returns_kurtosis_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 8, 20]
- **Performance Score:** 0.569
- **Stability Score:** 0.955
- **Optimization Method:** per_feature_mi_curve

#### returns_skewness_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.565
- **Stability Score:** 0.944
- **Optimization Method:** per_feature_mi_curve

#### roc_14_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.498
- **Stability Score:** 0.970
- **Optimization Method:** per_feature_mi_curve

#### roc_21_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 10, 20]
- **Performance Score:** 0.467
- **Stability Score:** 0.960
- **Optimization Method:** per_feature_mi_curve

#### roc_30_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.485
- **Stability Score:** 0.950
- **Optimization Method:** per_feature_mi_curve

#### rolling_returns_10_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.502
- **Stability Score:** 0.984
- **Optimization Method:** per_feature_mi_curve

#### rolling_returns_20_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.487
- **Stability Score:** 0.953
- **Optimization Method:** per_feature_mi_curve

#### rolling_zscore_returns_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [3, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 3, 20]
- **Performance Score:** 0.491
- **Stability Score:** 0.995
- **Optimization Method:** per_feature_mi_curve

#### shannon_entropy_20_10
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.551
- **Stability Score:** 0.994
- **Optimization Method:** per_feature_mi_curve

#### sharpe_ratio_20_0.0_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.555
- **Stability Score:** 0.963
- **Optimization Method:** per_feature_mi_curve

#### simple_returns_10_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.501
- **Stability Score:** 0.971
- **Optimization Method:** per_feature_mi_curve

#### simple_returns_1_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.506
- **Stability Score:** 0.944
- **Optimization Method:** per_feature_mi_curve

#### simple_returns_5_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.492
- **Stability Score:** 0.961
- **Optimization Method:** per_feature_mi_curve

#### spectral_entropy_20
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.528
- **Stability Score:** 0.957
- **Optimization Method:** per_feature_mi_curve

#### stochastic_14_3_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 8, 20]
- **Performance Score:** 0.501
- **Stability Score:** 0.979
- **Optimization Method:** per_feature_mi_curve

#### stochastic_30_3_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.510
- **Stability Score:** 0.949
- **Optimization Method:** per_feature_mi_curve

#### stochastic_kd_14_3
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.537
- **Stability Score:** 0.919
- **Optimization Method:** per_feature_mi_curve

#### support_level_1_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.570
- **Stability Score:** 0.971
- **Optimization Method:** per_feature_mi_curve

#### support_level_1_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.572
- **Stability Score:** 0.929
- **Optimization Method:** per_feature_mi_curve

#### support_level_1_5_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.531
- **Stability Score:** 0.973
- **Optimization Method:** per_feature_mi_curve

#### t3_14_0.7_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.617
- **Stability Score:** 0.958
- **Optimization Method:** per_feature_mi_curve

#### ultimate_oscillator_7_14_28_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.589
- **Stability Score:** 0.967
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_10_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.500
- **Stability Score:** 0.984
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_5_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 10, 20]
- **Performance Score:** 0.500
- **Stability Score:** 0.955
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_consistency_10_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.578
- **Stability Score:** 0.994
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_consistency_10_20_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.562
- **Stability Score:** 0.969
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_consistency_5_10_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.522
- **Stability Score:** 0.968
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_consistency_5_20_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 8, 20]
- **Performance Score:** 0.513
- **Stability Score:** 0.984
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_correlation_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 10, 15]
- **Performance Score:** 0.039
- **Stability Score:** 0.908
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_divergence_20_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.503
- **Stability Score:** 0.976
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_regime_5_10_price_returns
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 8, 20]
- **Performance Score:** 0.265
- **Stability Score:** 0.952
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_regime_5_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.228
- **Stability Score:** 0.944
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_adx_14
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.573
- **Stability Score:** 0.996
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_adx_21
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.589
- **Stability Score:** 0.969
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_adx_9
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.545
- **Stability Score:** 0.985
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_bbands_10_2.0
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.502
- **Stability Score:** 0.916
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_bbands_14_1.5
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.538
- **Stability Score:** 0.935
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_bbands_20_1.5
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.573
- **Stability Score:** 0.933
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_enhanced_ad_line_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.571
- **Stability Score:** 0.954
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_enhanced_ad_line_50
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.578
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_ichimoku_cloud_9_26_52
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.618
- **Stability Score:** 0.941
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_jerk_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 8, 20]
- **Performance Score:** 0.485
- **Stability Score:** 0.988
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_jerk_5_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.487
- **Stability Score:** 0.890
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.02_0.2
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.537
- **Stability Score:** 0.926
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.02_0.3
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.536
- **Stability Score:** 0.926
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.05_0.2
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.500
- **Stability Score:** 0.925
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.05_0.3
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.501
- **Stability Score:** 0.921
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.1_0.2
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.489
- **Stability Score:** 0.933
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.1_0.3
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.487
- **Stability Score:** 0.951
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_zigzag_10.0_2
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.480
- **Stability Score:** 0.908
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_zigzag_3.0_2
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.544
- **Stability Score:** 0.994
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_zigzag_5.0_2
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.493
- **Stability Score:** 0.948
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_zigzag_7.0_2
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.505
- **Stability Score:** 0.952
- **Optimization Method:** per_feature_mi_curve

#### vwap_deviations_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.569
- **Stability Score:** 0.924
- **Optimization Method:** per_feature_mi_curve

#### vwma_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.596
- **Stability Score:** 0.952
- **Optimization Method:** per_feature_mi_curve

#### wavelet_energy
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.578
- **Stability Score:** 0.977
- **Optimization Method:** per_feature_mi_curve

#### williams_r_21_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.513
- **Stability Score:** 0.952
- **Optimization Method:** per_feature_mi_curve

#### wma_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.589
- **Stability Score:** 0.938
- **Optimization Method:** per_feature_mi_curve

#### binary_label
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.791
- **Stability Score:** 0.977
- **Optimization Method:** per_feature_mi_curve

#### meta_probability
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.551
- **Stability Score:** 0.951
- **Optimization Method:** per_feature_mi_curve

### Momentum Features

Optimized 31 features with optimal + alternative lookback periods:

#### advanced_momentum_10_30
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.589
- **Stability Score:** 0.958
- **Optimization Method:** per_feature_mi_curve

#### advanced_momentum_5_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.591
- **Stability Score:** 0.941
- **Optimization Method:** per_feature_mi_curve

#### macd_12_26_9_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.583
- **Stability Score:** 0.943
- **Optimization Method:** per_feature_mi_curve

#### macd_delta_12_26_9
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.567
- **Stability Score:** 0.918
- **Optimization Method:** per_feature_mi_curve

#### macd_entropy_20_12_26
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 15, 20]
- **Performance Score:** 0.450
- **Stability Score:** 0.958
- **Optimization Method:** per_feature_mi_curve

#### momentum_14_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.498
- **Stability Score:** 0.963
- **Optimization Method:** per_feature_mi_curve

#### momentum_21_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.506
- **Stability Score:** 0.973
- **Optimization Method:** per_feature_mi_curve

#### momentum_30_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.490
- **Stability Score:** 0.982
- **Optimization Method:** per_feature_mi_curve

#### momentum_endpoints_sma_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.586
- **Stability Score:** 0.936
- **Optimization Method:** per_feature_mi_curve

#### momentum_features
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.601
- **Stability Score:** 0.963
- **Optimization Method:** per_feature_mi_curve

#### rsi_14_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.534
- **Stability Score:** 0.945
- **Optimization Method:** per_feature_mi_curve

#### rsi_21_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.550
- **Stability Score:** 0.983
- **Optimization Method:** per_feature_mi_curve

#### rsi_30_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.545
- **Stability Score:** 0.946
- **Optimization Method:** per_feature_mi_curve

#### rsi_zscore_14_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.543
- **Stability Score:** 0.975
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_momentum_10_10_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 8, 20]
- **Performance Score:** 0.533
- **Stability Score:** 0.928
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_momentum_10_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.559
- **Stability Score:** 0.956
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_momentum_5_10_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.495
- **Stability Score:** 0.924
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_momentum_5_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.523
- **Stability Score:** 0.937
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_50_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.489
- **Stability Score:** 0.950
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_5_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.506
- **Stability Score:** 0.972
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_acceleration_10_10_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.536
- **Stability Score:** 0.947
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_acceleration_10_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.527
- **Stability Score:** 0.934
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_acceleration_5_10_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [3, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 3, 20]
- **Performance Score:** 0.512
- **Stability Score:** 0.948
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_acceleration_5_20_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.506
- **Stability Score:** 0.964
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_comprehensive_14
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.581
- **Stability Score:** 0.927
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_comprehensive_21
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.597
- **Stability Score:** 0.931
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_comprehensive_30
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.606
- **Stability Score:** 0.928
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_comprehensive_9
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.562
- **Stability Score:** 0.928
- **Optimization Method:** per_feature_mi_curve

#### volume_momentum_10
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.522
- **Stability Score:** 0.975
- **Optimization Method:** per_feature_mi_curve

#### volume_momentum_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.529
- **Stability Score:** 0.987
- **Optimization Method:** per_feature_mi_curve

#### volume_momentum_5
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.497
- **Stability Score:** 0.958
- **Optimization Method:** per_feature_mi_curve

### Volume Features

Optimized 49 features with optimal + alternative lookback periods:

#### analyst_volume_pressure
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.447
- **Stability Score:** 0.915
- **Optimization Method:** per_feature_mi_curve

#### analyst_volume_trend
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.550
- **Stability Score:** 0.953
- **Optimization Method:** per_feature_mi_curve

#### price_volume_oscillator_10_20
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.536
- **Stability Score:** 0.962
- **Optimization Method:** per_feature_mi_curve

#### price_volume_oscillator_5_15
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.518
- **Stability Score:** 0.955
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_enhanced_obv_10
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.560
- **Stability Score:** 0.994
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_enhanced_obv_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.588
- **Stability Score:** 0.967
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_enhanced_obv_50
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.577
- **Stability Score:** 0.990
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_smoothed_obv_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.596
- **Stability Score:** 0.991
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volume_acceleration_5_volume_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [3, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 3, 20]
- **Performance Score:** 0.470
- **Stability Score:** 0.963
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volume_weighted_ad_line_10
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.556
- **Stability Score:** 0.947
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volume_weighted_ad_line_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.594
- **Stability Score:** 0.963
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volume_weighted_ad_line_50
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.605
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_accumulation_distribution
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.563
- **Stability Score:** 0.947
- **Optimization Method:** per_feature_mi_curve

#### volume_ema_10
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.579
- **Stability Score:** 0.977
- **Optimization Method:** per_feature_mi_curve

#### volume_ema_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.605
- **Stability Score:** 0.972
- **Optimization Method:** per_feature_mi_curve

#### volume_ema_5
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.565
- **Stability Score:** 0.980
- **Optimization Method:** per_feature_mi_curve

#### volume_ema_50
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.593
- **Stability Score:** 0.961
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_10_volume_returns
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 5, 20]
- **Performance Score:** 0.364
- **Stability Score:** 0.958
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_20_volume_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.319
- **Stability Score:** 0.917
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_5_volume_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.360
- **Stability Score:** 0.883
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_ma_10_10_volume_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.443
- **Stability Score:** 0.984
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_ma_10_5_volume_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.416
- **Stability Score:** 0.978
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_ma_20_10_volume_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.396
- **Stability Score:** 0.916
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_ma_20_5_volume_returns
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 15, 20]
- **Performance Score:** 0.374
- **Stability Score:** 0.934
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_ma_5_10_volume_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.503
- **Stability Score:** 0.969
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_ma_5_5_volume_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 10, 20]
- **Performance Score:** 0.479
- **Stability Score:** 0.958
- **Optimization Method:** per_feature_mi_curve

#### volume_ma_ratios_20_10
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.511
- **Stability Score:** 0.950
- **Optimization Method:** per_feature_mi_curve

#### volume_oscillator_10_20
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.552
- **Stability Score:** 0.954
- **Optimization Method:** per_feature_mi_curve

#### volume_oscillator_5_15
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.539
- **Stability Score:** 0.975
- **Optimization Method:** per_feature_mi_curve

#### volume_percentile_100
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.527
- **Stability Score:** 0.945
- **Optimization Method:** per_feature_mi_curve

#### volume_percentile_20
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.516
- **Stability Score:** 0.941
- **Optimization Method:** per_feature_mi_curve

#### volume_percentile_50
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.523
- **Stability Score:** 0.949
- **Optimization Method:** per_feature_mi_curve

#### volume_price_correlation_10
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.553
- **Stability Score:** 0.959
- **Optimization Method:** per_feature_mi_curve

#### volume_price_correlation_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.590
- **Stability Score:** 0.969
- **Optimization Method:** per_feature_mi_curve

#### volume_price_divergence_10
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.509
- **Stability Score:** 0.960
- **Optimization Method:** per_feature_mi_curve

#### volume_price_divergence_20
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.512
- **Stability Score:** 0.936
- **Optimization Method:** per_feature_mi_curve

#### volume_price_trend
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.593
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_ratio_10
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.501
- **Stability Score:** 0.962
- **Optimization Method:** per_feature_mi_curve

#### volume_ratio_20
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.511
- **Stability Score:** 0.949
- **Optimization Method:** per_feature_mi_curve

#### volume_ratio_50
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.525
- **Stability Score:** 0.959
- **Optimization Method:** per_feature_mi_curve

#### volume_roc_1
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.480
- **Stability Score:** 0.951
- **Optimization Method:** per_feature_mi_curve

#### volume_roc_10
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.532
- **Stability Score:** 0.969
- **Optimization Method:** per_feature_mi_curve

#### volume_roc_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.536
- **Stability Score:** 0.990
- **Optimization Method:** per_feature_mi_curve

#### volume_roc_5
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.495
- **Stability Score:** 0.965
- **Optimization Method:** per_feature_mi_curve

#### volume_trend_strength_10_30
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.572
- **Stability Score:** 0.953
- **Optimization Method:** per_feature_mi_curve

#### volume_trend_strength_20_50
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.598
- **Stability Score:** 0.969
- **Optimization Method:** per_feature_mi_curve

#### volume_vwap_10
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.615
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_vwap_50
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.600
- **Stability Score:** 0.977
- **Optimization Method:** per_feature_mi_curve

#### volume_zscore_60_252
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.531
- **Stability Score:** 0.945
- **Optimization Method:** per_feature_mi_curve

### Volatility Features

Optimized 42 features with optimal + alternative lookback periods:

#### band_limited_volatility
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.578
- **Stability Score:** 0.996
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_10
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.540
- **Stability Score:** 0.936
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_100
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.569
- **Stability Score:** 0.888
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_14
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.566
- **Stability Score:** 0.957
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.587
- **Stability Score:** 0.965
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_30
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.581
- **Stability Score:** 0.924
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_50
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [4, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 4, 20]
- **Performance Score:** 0.579
- **Stability Score:** 0.929
- **Optimization Method:** per_feature_mi_curve

#### returns_volatility_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 8, 20]
- **Performance Score:** 0.576
- **Stability Score:** 0.982
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_volatility_10_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.584
- **Stability Score:** 0.995
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_volatility_10_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.563
- **Stability Score:** 0.970
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_volatility_5_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.521
- **Stability Score:** 0.977
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_volatility_5_20_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 8, 20]
- **Performance Score:** 0.515
- **Stability Score:** 0.984
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_10
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.579
- **Stability Score:** 0.961
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_14
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.584
- **Stability Score:** 0.946
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.584
- **Stability Score:** 0.921
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_30
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.590
- **Stability Score:** 0.929
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_50
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.601
- **Stability Score:** 0.939
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_10
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.583
- **Stability Score:** 0.964
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_14
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.602
- **Stability Score:** 0.967
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.602
- **Stability Score:** 0.950
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_30
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.598
- **Stability Score:** 0.953
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_50
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.608
- **Stability Score:** 0.976
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parkinson_volatility_14
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.599
- **Stability Score:** 0.966
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parkinson_volatility_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.599
- **Stability Score:** 0.950
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parkinson_volatility_30
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.599
- **Stability Score:** 0.962
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parkinson_volatility_50
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.610
- **Stability Score:** 0.976
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_10
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.579
- **Stability Score:** 0.956
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_14
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.604
- **Stability Score:** 0.969
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.601
- **Stability Score:** 0.946
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_30
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.597
- **Stability Score:** 0.960
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_50
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.605
- **Stability Score:** 0.976
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_acceleration_5_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 10, 20]
- **Performance Score:** 0.491
- **Stability Score:** 0.933
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_10
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.474
- **Stability Score:** 0.926
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_14
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.506
- **Stability Score:** 0.962
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_20
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.517
- **Stability Score:** 0.947
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_30
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.543
- **Stability Score:** 0.971
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_50
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.567
- **Stability Score:** 0.961
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_yang_zhang_volatility_10
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.585
- **Stability Score:** 0.976
- **Optimization Method:** per_feature_mi_curve

#### volume_std_10
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.576
- **Stability Score:** 0.964
- **Optimization Method:** per_feature_mi_curve

#### volume_std_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.599
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_std_50
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.589
- **Stability Score:** 0.959
- **Optimization Method:** per_feature_mi_curve

#### volume_volatility_elasticity_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.578
- **Stability Score:** 0.974
- **Optimization Method:** per_feature_mi_curve

### Trend Features

Optimized 24 features with optimal + alternative lookback periods:

#### dema_21_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.549
- **Stability Score:** 0.939
- **Optimization Method:** per_feature_mi_curve

#### ema_12_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.618
- **Stability Score:** 0.960
- **Optimization Method:** per_feature_mi_curve

#### ema_26_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.618
- **Stability Score:** 0.967
- **Optimization Method:** per_feature_mi_curve

#### ema_50_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.615
- **Stability Score:** 0.973
- **Optimization Method:** per_feature_mi_curve

#### sma_100_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.595
- **Stability Score:** 0.977
- **Optimization Method:** per_feature_mi_curve

#### sma_10_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.610
- **Stability Score:** 0.956
- **Optimization Method:** per_feature_mi_curve

#### sma_20_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.610
- **Stability Score:** 0.987
- **Optimization Method:** per_feature_mi_curve

#### sma_50_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.602
- **Stability Score:** 0.980
- **Optimization Method:** per_feature_mi_curve

#### sma_5_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.607
- **Stability Score:** 0.951
- **Optimization Method:** per_feature_mi_curve

#### tema_21_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.525
- **Stability Score:** 0.934
- **Optimization Method:** per_feature_mi_curve

#### trend_score_14
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.618
- **Stability Score:** 0.936
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_trend_strength_10_10_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 8, 20]
- **Performance Score:** 0.490
- **Stability Score:** 0.965
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_trend_strength_10_20_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.471
- **Stability Score:** 0.926
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_trend_strength_5_10_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.500
- **Stability Score:** 0.983
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_trend_strength_5_20_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.485
- **Stability Score:** 0.956
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_sma_100
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.619
- **Stability Score:** 1.000
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_consistency_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.566
- **Stability Score:** 0.974
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_consistency_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.583
- **Stability Score:** 0.952
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_consistency_50_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.599
- **Stability Score:** 0.964
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_consistency_5_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.523
- **Stability Score:** 0.934
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_strength_10_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.492
- **Stability Score:** 0.936
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_strength_20_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.499
- **Stability Score:** 0.926
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_strength_50_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 10, 20]
- **Performance Score:** 0.523
- **Stability Score:** 0.950
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_strength_5_price_returns
- **Optimal Lookback:** 4 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [4, 8, 20]
- **Performance Score:** 0.494
- **Stability Score:** 0.990
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

