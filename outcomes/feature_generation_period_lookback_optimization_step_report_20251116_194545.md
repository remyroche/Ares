# Lookback Optimization Report

**Generated:** 2025-11-16 19:45:45
**Step:** feature_generation_period_lookback_optimization_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** blank

## Optimization Results

- **Momentum Lookback:** 20
- **Trend Lookback:** 20
- **Volatility Lookback:** 20
- **Volume Lookback:** 20
- **Optimization Score:** 0.50

## Comprehensive Optimization Analysis

### Data Export

- **Per-Feature Metrics CSV:** `outcomes/feature_generation_period_lookback_optimization_step_per_feature_metrics_20251116_194545.csv`
- **Full Path:** `/Users/remyroche/Documents/Ares/outcomes/feature_generation_period_lookback_optimization_step_per_feature_metrics_20251116_194545.csv`

### Optimization Performance Metrics

| Metric | Value |
|--------|-------|
| Optimization Method | data_driven_cross_validation |
| Total Features Analyzed | 268 |
| Lookback Range Tested | 1-50 |
| Cross-Validation Folds | 2 |
| Optimization Efficiency | 85.0% |
| Stability Score | 0.998 |
| Performance Score | 0.004 |

### Global Optimization Metrics

| Metric | Value |
|--------|-------|
| Total Features Optimized | 259 |
| Categories Processed | 5 |
| Average Lookback Period | 20.0 |
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
| Momentum | 31 | 20 | 0.003 | 0.998 | 0.501 | 0.500 | N/A | cv |
| Trend | 24 | 20 | 0.004 | 0.999 | 0.501 | 0.500 | N/A | cv |
| Volatility | 42 | 20 | 0.004 | 0.998 | 0.501 | 0.500 | N/A | cv |
| Volume | 49 | 20 | 0.003 | 0.999 | 0.501 | 0.500 | N/A | cv |

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
| momentum | 31 | 20 | 1-50 | 0.003 | 0.998 | 0.501 | 0.500 | 100.0% |
| volatility | 42 | 20 | 1-50 | 0.004 | 0.998 | 0.501 | 0.500 | 100.0% |
| trend | 24 | 20 | 1-50 | 0.004 | 0.999 | 0.501 | 0.500 | 100.0% |
| volume | 49 | 20 | 1-50 | 0.003 | 0.999 | 0.501 | 0.500 | 100.0% |

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

### Individual Feature Analysis by Category

#### Momentum Features

**Category Summary:**
- **Best Individual Feature Lookback:** 20
- **Average Performance Score:** 0.003
- **Average Stability Score:** 0.998
- **Features Optimized:** 31

#### Trend Features

**Category Summary:**
- **Best Individual Feature Lookback:** 20
- **Average Performance Score:** 0.004
- **Average Stability Score:** 0.999
- **Features Optimized:** 24

#### Volatility Features

**Category Summary:**
- **Best Individual Feature Lookback:** 20
- **Average Performance Score:** 0.004
- **Average Stability Score:** 0.998
- **Features Optimized:** 42

#### Volume Features

**Category Summary:**
- **Best Individual Feature Lookback:** 20
- **Average Performance Score:** 0.003
- **Average Stability Score:** 0.999
- **Features Optimized:** 49

### Per-Feature Optimization Results

| Feature Name | Optimal Lookback | Performance | Stability | Method |
|--------------|------------------|-------------|-----------|--------|
| acceleration_features | 20 | 0.003 | 0.998 | N/A |
| advanced_cumulative_returns_10 | 20 | 0.003 | 0.998 | N/A |
| advanced_cumulative_returns_20 | 20 | 0.005 | 0.997 | N/A |
| adx_14_returns_vwap | 20 | 0.004 | 0.999 | N/A |
| apo_12_26_returns_vwap | 20 | 0.004 | 0.998 | N/A |
| ar_1_coefficients_20 | 20 | 0.004 | 0.999 | N/A |
| aroon_25_returns_vwap | 20 | 0.004 | 0.998 | N/A |
| candlestick_dark_cloud_cover_pattern | 20 | 0.000 | 1.000 | N/A |
| candlestick_doji_pattern | 20 | 0.001 | 0.999 | N/A |
| candlestick_engulfing_pattern | 20 | 0.001 | 1.000 | N/A |
| candlestick_harami_cross_pattern | 20 | 0.001 | 0.999 | N/A |
| candlestick_piercing_line_pattern | 20 | 0.001 | 1.000 | N/A |
| candlestick_three_black_crows_pattern | 20 | 0.002 | 0.999 | N/A |
| candlestick_three_white_soldiers_pattern | 20 | 0.002 | 0.999 | N/A |
| cci_20_returns_vwap | 20 | 0.003 | 0.998 | N/A |
| cmf_20 | 20 | 0.003 | 0.999 | N/A |
| cumulative_returns_10_price_returns | 20 | 0.002 | 0.999 | N/A |
| cumulative_returns_20_price_returns | 20 | 0.002 | 0.999 | N/A |
| cycle_length | 20 | 0.002 | 0.999 | N/A |
| dfa_slopes | 20 | 0.003 | 0.999 | N/A |
| directional_signal | 20 | 0.005 | 0.997 | N/A |
| donchian_channel_20 | 20 | 0.004 | 0.999 | N/A |
| entropy_rate_20 | 20 | 0.004 | 0.999 | N/A |
| fibonacci_0.236_10_price_returns | 20 | 0.003 | 0.999 | N/A |
| fibonacci_0.236_20_price_returns | 20 | 0.003 | 0.999 | N/A |
| fibonacci_0.236_5_price_returns | 20 | 0.003 | 0.999 | N/A |
| fibonacci_0.382_10_price_returns | 20 | 0.003 | 0.999 | N/A |
| fibonacci_0.382_20_price_returns | 20 | 0.004 | 0.999 | N/A |
| fibonacci_0.382_5_price_returns | 20 | 0.003 | 0.999 | N/A |
| fibonacci_0.5_10_price_returns | 20 | 0.003 | 0.999 | N/A |
| fibonacci_0.5_20_price_returns | 20 | 0.004 | 0.999 | N/A |
| fibonacci_0.5_5_price_returns | 20 | 0.003 | 0.999 | N/A |
| fibonacci_0.618_10_price_returns | 20 | 0.003 | 0.999 | N/A |
| fibonacci_0.618_20_price_returns | 20 | 0.004 | 0.999 | N/A |
| fibonacci_0.618_5_price_returns | 20 | 0.003 | 0.999 | N/A |
| fibonacci_0.786_10_price_returns | 15 | 0.003 | 0.998 | N/A |
| fibonacci_0.786_20_price_returns | 20 | 0.004 | 0.997 | N/A |
| fibonacci_0.786_5_price_returns | 8 | 0.003 | 0.998 | N/A |
| fractal_dimension | 3 | 0.000 | 0.500 | N/A |
| hurst_exponent | 20 | 0.004 | 0.999 | N/A |
| kama_30_2_30_returns_vwap | 20 | 0.005 | 0.998 | N/A |
| kst_10_15_20_30_10_10_10_15_returns_vwap | 20 | 0.004 | 0.999 | N/A |
| lempel_ziv_complexity_20 | 20 | 0.003 | 0.998 | N/A |
| ljung_box_pvalue_20_10 | 20 | 0.003 | 0.999 | N/A |
| log_returns_10_price_returns | 20 | 0.003 | 0.999 | N/A |
| log_returns_1_price_returns | 20 | 0.004 | 0.999 | N/A |
| log_returns_5_price_returns | 20 | 0.003 | 0.998 | N/A |
| mama_21_0.05_price_returns | 20 | 0.004 | 0.998 | N/A |
| max_drawdown | 20 | 0.004 | 0.998 | N/A |
| order_flow_imbalance_20 | 20 | 0.004 | 0.999 | N/A |
| pfe_12_returns_vwap | 20 | 0.004 | 0.998 | N/A |
| pivot_point_5_price_returns | 8 | 0.003 | 0.999 | N/A |
| resistance_level_1_10_price_returns | 20 | 0.004 | 0.999 | N/A |
| resistance_level_1_20_price_returns | 20 | 0.005 | 0.999 | N/A |
| resistance_level_2_5_price_returns | 8 | 0.003 | 0.999 | N/A |
| returns_kurtosis_20_price_returns | 20 | 0.004 | 0.998 | N/A |
| returns_skewness_20_price_returns | 20 | 0.004 | 0.998 | N/A |
| roc_14_price_returns | 20 | 0.002 | 1.000 | N/A |
| roc_21_price_returns | 20 | 0.003 | 0.999 | N/A |
| roc_30_price_returns | 20 | 0.003 | 0.999 | N/A |
| rolling_returns_10_price_returns | 20 | 0.002 | 0.999 | N/A |
| rolling_returns_20_price_returns | 20 | 0.002 | 0.999 | N/A |
| rolling_zscore_returns_20 | 20 | 0.003 | 0.999 | N/A |
| shannon_entropy_20_10 | 20 | 0.004 | 0.999 | N/A |
| sharpe_ratio_20_0.0_price_returns | 20 | 0.004 | 0.999 | N/A |
| simple_returns_10_price_returns | 20 | 0.003 | 0.999 | N/A |
| simple_returns_1_price_returns | 20 | 0.002 | 0.999 | N/A |
| simple_returns_5_price_returns | 20 | 0.002 | 0.999 | N/A |
| spectral_entropy_20 | 20 | 0.004 | 0.999 | N/A |
| stochastic_14_3_price_returns | 20 | 0.004 | 0.999 | N/A |
| stochastic_21_3_price_returns | 20 | 0.004 | 0.999 | N/A |
| stochastic_30_3_price_returns | 20 | 0.004 | 0.999 | N/A |
| stochastic_kd_14_3 | 20 | 0.003 | 0.999 | N/A |
| support_level_1_10_price_returns | 20 | 0.004 | 0.998 | N/A |
| support_level_1_20_price_returns | 20 | 0.004 | 0.999 | N/A |
| support_level_1_5_price_returns | 20 | 0.004 | 0.999 | N/A |
| t3_14_0.7_returns_vwap | 20 | 0.005 | 0.998 | N/A |
| ultimate_oscillator_7_14_28_returns_vwap | 20 | 0.004 | 0.999 | N/A |
| vectorbt_acceleration_10_price_returns | 20 | 0.002 | 0.999 | N/A |
| vectorbt_acceleration_5_price_returns | 20 | 0.002 | 0.999 | N/A |
| vectorbt_acceleration_consistency_10_10_price_returns | 20 | 0.004 | 0.999 | N/A |
| vectorbt_acceleration_consistency_10_20_price_returns | 20 | 0.004 | 0.998 | N/A |
| vectorbt_acceleration_consistency_5_10_price_returns | 20 | 0.004 | 0.999 | N/A |
| vectorbt_acceleration_consistency_5_20_price_returns | 20 | 0.004 | 0.998 | N/A |
| vectorbt_acceleration_correlation_20_price_returns | 20 | 0.000 | 1.000 | N/A |
| vectorbt_acceleration_divergence_20_price_returns | 20 | 0.002 | 0.999 | N/A |
| vectorbt_acceleration_regime_5_10_price_returns | 20 | 0.001 | 1.000 | N/A |
| vectorbt_acceleration_regime_5_20_price_returns | 20 | 0.001 | 1.000 | N/A |
| vectorbt_adx_14 | 20 | 0.004 | 0.998 | N/A |
| vectorbt_adx_21 | 20 | 0.004 | 0.999 | N/A |
| vectorbt_adx_9 | 20 | 0.004 | 0.998 | N/A |
| vectorbt_bbands_10_1.5 | 20 | 0.003 | 0.999 | N/A |
| vectorbt_bbands_14_1.5 | 20 | 0.003 | 0.999 | N/A |
| vectorbt_bbands_20_1.5 | 20 | 0.003 | 0.998 | N/A |
| vectorbt_enhanced_ad_line_20 | 20 | 0.004 | 0.998 | N/A |
| vectorbt_enhanced_ad_line_50 | 20 | 0.004 | 0.999 | N/A |
| vectorbt_ichimoku_cloud_9_26_52 | 20 | 0.005 | 0.998 | N/A |
| vectorbt_jerk_10_price_returns | 20 | 0.002 | 0.999 | N/A |
| vectorbt_jerk_5_price_returns | 20 | 0.003 | 0.998 | N/A |
| vectorbt_parabolic_sar_0.02_0.2 | 20 | 0.003 | 0.998 | N/A |
| vectorbt_parabolic_sar_0.02_0.3 | 20 | 0.003 | 0.998 | N/A |
| vectorbt_parabolic_sar_0.05_0.2 | 20 | 0.003 | 0.999 | N/A |
| vectorbt_parabolic_sar_0.05_0.3 | 20 | 0.003 | 0.998 | N/A |
| vectorbt_parabolic_sar_0.1_0.2 | 20 | 0.003 | 0.999 | N/A |
| vectorbt_parabolic_sar_0.1_0.3 | 20 | 0.003 | 0.998 | N/A |
| vectorbt_zigzag_10.0_2 | 20 | 0.003 | 0.998 | N/A |
| vectorbt_zigzag_3.0_2 | 20 | 0.003 | 0.999 | N/A |
| vectorbt_zigzag_5.0_3 | 20 | 0.004 | 0.999 | N/A |
| vectorbt_zigzag_7.0_3 | 20 | 0.003 | 0.999 | N/A |
| vwap_deviations_20 | 20 | 0.004 | 0.998 | N/A |
| vwma_20_price_returns | 20 | 0.005 | 0.998 | N/A |
| wavelet_energy | 20 | 0.004 | 0.999 | N/A |
| wma_20_price_returns | 20 | 0.004 | 0.998 | N/A |
| advanced_momentum_10_30 | 20 | 0.004 | 0.998 | N/A |
| advanced_momentum_5_20 | 20 | 0.004 | 0.998 | N/A |
| macd_12_26_9_returns_vwap | 20 | 0.004 | 0.997 | N/A |
| macd_delta_12_26_9 | 20 | 0.004 | 0.998 | N/A |
| macd_entropy_20_12_26 | 20 | 0.002 | 0.999 | N/A |
| momentum_14_price_returns | 20 | 0.003 | 0.998 | N/A |
| momentum_21_price_returns | 20 | 0.003 | 0.999 | N/A |
| momentum_30_price_returns | 20 | 0.003 | 0.998 | N/A |
| momentum_endpoints_sma_20 | 20 | 0.004 | 0.998 | N/A |
| momentum_features | 20 | 0.005 | 0.997 | N/A |
| rsi_14_returns_vwap | 20 | 0.003 | 0.998 | N/A |
| rsi_21_returns_vwap | 20 | 0.003 | 0.998 | N/A |
| rsi_30_returns_vwap | 20 | 0.003 | 0.998 | N/A |
| rsi_zscore_14_20 | 20 | 0.003 | 0.998 | N/A |
| vectorbt_acceleration_momentum_10_10_price_returns | 20 | 0.004 | 0.998 | N/A |
| vectorbt_acceleration_momentum_10_20_price_returns | 20 | 0.004 | 0.999 | N/A |
| vectorbt_acceleration_momentum_5_10_price_returns | 20 | 0.003 | 0.999 | N/A |
| vectorbt_acceleration_momentum_5_20_price_returns | 20 | 0.003 | 0.999 | N/A |
| vectorbt_momentum_50_price_returns | 20 | 0.002 | 0.999 | N/A |
| vectorbt_momentum_5_price_returns | 20 | 0.002 | 0.999 | N/A |
| vectorbt_momentum_acceleration_10_10_price_returns | 20 | 0.003 | 0.999 | N/A |
| vectorbt_momentum_acceleration_10_20_price_returns | 20 | 0.004 | 0.999 | N/A |
| vectorbt_momentum_acceleration_5_10_price_returns | 20 | 0.003 | 0.998 | N/A |
| vectorbt_momentum_acceleration_5_20_price_returns | 20 | 0.003 | 0.998 | N/A |
| vectorbt_momentum_comprehensive_14 | 20 | 0.004 | 0.998 | N/A |
| vectorbt_momentum_comprehensive_21 | 20 | 0.004 | 0.997 | N/A |
| vectorbt_momentum_comprehensive_30 | 20 | 0.004 | 0.997 | N/A |
| vectorbt_momentum_comprehensive_9 | 20 | 0.004 | 0.998 | N/A |
| volume_momentum_10 | 20 | 0.003 | 0.998 | N/A |
| volume_momentum_20 | 20 | 0.003 | 0.999 | N/A |
| volume_momentum_5 | 20 | 0.003 | 0.997 | N/A |
| analyst_volume_pressure | 20 | 0.004 | 0.998 | N/A |
| analyst_volume_trend | 20 | 0.004 | 0.998 | N/A |
| price_volume_oscillator_10_20 | 20 | 0.004 | 0.999 | N/A |
| price_volume_oscillator_5_15 | 20 | 0.003 | 0.999 | N/A |
| vectorbt_enhanced_obv_10 | 20 | 0.004 | 0.999 | N/A |
| vectorbt_enhanced_obv_20 | 20 | 0.004 | 0.999 | N/A |
| vectorbt_enhanced_obv_50 | 20 | 0.004 | 0.998 | N/A |
| vectorbt_smoothed_obv_10 | 20 | 0.004 | 0.998 | N/A |
| vectorbt_volume_acceleration_5_volume_returns | 20 | 0.003 | 0.997 | N/A |
| vectorbt_volume_weighted_ad_line_10 | 20 | 0.004 | 0.999 | N/A |
| vectorbt_volume_weighted_ad_line_20 | 20 | 0.004 | 0.999 | N/A |
| vectorbt_volume_weighted_ad_line_50 | 20 | 0.004 | 1.000 | N/A |
| volume_accumulation_distribution | 20 | 0.004 | 0.998 | N/A |
| volume_ema_10 | 20 | 0.004 | 0.999 | N/A |
| volume_ema_20 | 20 | 0.004 | 0.998 | N/A |
| volume_ema_5 | 20 | 0.004 | 0.998 | N/A |
| volume_ema_50 | 20 | 0.004 | 0.999 | N/A |
| volume_entropy_10_volume_returns | 20 | 0.002 | 0.999 | N/A |
| volume_entropy_20_volume_returns | 20 | 0.002 | 0.999 | N/A |
| volume_entropy_5_volume_returns | 20 | 0.003 | 0.999 | N/A |
| volume_entropy_ma_10_10_volume_returns | 20 | 0.003 | 1.000 | N/A |
| volume_entropy_ma_10_5_volume_returns | 20 | 0.003 | 1.000 | N/A |
| volume_entropy_ma_20_10_volume_returns | 20 | 0.002 | 1.000 | N/A |
| volume_entropy_ma_20_5_volume_returns | 20 | 0.002 | 0.999 | N/A |
| volume_entropy_ma_5_10_volume_returns | 20 | 0.003 | 0.999 | N/A |
| volume_entropy_ma_5_5_volume_returns | 20 | 0.003 | 1.000 | N/A |
| volume_ma_ratios_20_10 | 20 | 0.003 | 0.998 | N/A |
| volume_oscillator_10_20 | 20 | 0.004 | 0.998 | N/A |
| volume_oscillator_5_15 | 20 | 0.003 | 0.998 | N/A |
| volume_percentile_100 | 20 | 0.003 | 0.999 | N/A |
| volume_percentile_20 | 20 | 0.003 | 0.998 | N/A |
| volume_percentile_50 | 20 | 0.003 | 0.999 | N/A |
| volume_price_correlation_10 | 20 | 0.003 | 0.999 | N/A |
| volume_price_correlation_20 | 20 | 0.004 | 0.998 | N/A |
| volume_price_divergence_10 | 20 | 0.002 | 0.998 | N/A |
| volume_price_divergence_20 | 20 | 0.003 | 0.997 | N/A |
| volume_price_trend | 20 | 0.005 | 0.998 | N/A |
| volume_ratio_10 | 20 | 0.003 | 0.998 | N/A |
| volume_ratio_20 | 20 | 0.003 | 0.998 | N/A |
| volume_ratio_50 | 20 | 0.003 | 0.998 | N/A |
| volume_roc_1 | 20 | 0.003 | 0.998 | N/A |
| volume_roc_10 | 20 | 0.003 | 0.998 | N/A |
| volume_roc_20 | 20 | 0.003 | 0.999 | N/A |
| volume_roc_5 | 20 | 0.003 | 0.999 | N/A |
| volume_trend_strength_10_30 | 20 | 0.004 | 0.999 | N/A |
| volume_trend_strength_20_50 | 20 | 0.004 | 0.998 | N/A |
| volume_vwap_10 | 20 | 0.005 | 0.998 | N/A |
| volume_vwap_50 | 20 | 0.004 | 0.999 | N/A |
| volume_zscore_60_252 | 20 | 0.003 | 0.998 | N/A |
| band_limited_volatility | 20 | 0.004 | 0.998 | N/A |
| enhanced_volatility_10 | 20 | 0.003 | 0.999 | N/A |
| enhanced_volatility_100 | 20 | 0.004 | 0.999 | N/A |
| enhanced_volatility_14 | 20 | 0.004 | 0.998 | N/A |
| enhanced_volatility_20 | 20 | 0.004 | 0.998 | N/A |
| enhanced_volatility_30 | 20 | 0.005 | 0.997 | N/A |
| enhanced_volatility_50 | 20 | 0.004 | 0.998 | N/A |
| returns_volatility_20_price_returns | 20 | 0.004 | 0.999 | N/A |
| vectorbt_acceleration_volatility_10_10_price_returns | 20 | 0.004 | 0.999 | N/A |
| vectorbt_acceleration_volatility_10_20_price_returns | 20 | 0.004 | 0.998 | N/A |
| vectorbt_acceleration_volatility_5_10_price_returns | 20 | 0.004 | 0.998 | N/A |
| vectorbt_acceleration_volatility_5_20_price_returns | 20 | 0.004 | 0.998 | N/A |
| vectorbt_atr_10 | 20 | 0.004 | 0.998 | N/A |
| vectorbt_atr_14 | 20 | 0.004 | 0.998 | N/A |
| vectorbt_atr_20 | 20 | 0.004 | 0.998 | N/A |
| vectorbt_atr_30 | 20 | 0.004 | 0.998 | N/A |
| vectorbt_atr_50 | 20 | 0.004 | 0.998 | N/A |
| vectorbt_garman_klass_volatility_10 | 20 | 0.004 | 0.999 | N/A |
| vectorbt_garman_klass_volatility_14 | 20 | 0.004 | 0.998 | N/A |
| vectorbt_garman_klass_volatility_20 | 20 | 0.004 | 0.998 | N/A |
| vectorbt_garman_klass_volatility_30 | 20 | 0.004 | 0.999 | N/A |
| vectorbt_garman_klass_volatility_50 | 20 | 0.004 | 0.999 | N/A |
| vectorbt_parkinson_volatility_10 | 20 | 0.004 | 0.999 | N/A |
| vectorbt_parkinson_volatility_14 | 20 | 0.004 | 0.999 | N/A |
| vectorbt_parkinson_volatility_20 | 20 | 0.005 | 0.998 | N/A |
| vectorbt_parkinson_volatility_30 | 20 | 0.004 | 0.999 | N/A |
| vectorbt_parkinson_volatility_50 | 20 | 0.004 | 0.999 | N/A |
| vectorbt_rogers_satchell_volatility_10 | 20 | 0.004 | 0.999 | N/A |
| vectorbt_rogers_satchell_volatility_14 | 20 | 0.004 | 0.998 | N/A |
| vectorbt_rogers_satchell_volatility_20 | 20 | 0.004 | 0.998 | N/A |
| vectorbt_rogers_satchell_volatility_30 | 20 | 0.004 | 0.999 | N/A |
| vectorbt_rogers_satchell_volatility_50 | 20 | 0.004 | 0.999 | N/A |
| vectorbt_volatility_acceleration_5_20_price_returns | 20 | 0.003 | 0.997 | N/A |
| vectorbt_volatility_comprehensive_10 | 20 | 0.004 | 0.998 | N/A |
| vectorbt_volatility_comprehensive_14 | 20 | 0.004 | 0.998 | N/A |
| vectorbt_volatility_comprehensive_20 | 20 | 0.004 | 0.997 | N/A |
| vectorbt_volatility_comprehensive_30 | 20 | 0.003 | 0.998 | N/A |
| vectorbt_volatility_comprehensive_50 | 20 | 0.004 | 0.999 | N/A |
| volume_std_10 | 20 | 0.004 | 0.999 | N/A |
| volume_std_20 | 20 | 0.005 | 0.998 | N/A |
| volume_std_50 | 20 | 0.004 | 0.999 | N/A |
| volume_volatility_elasticity_20 | 20 | 0.004 | 0.999 | N/A |
| dema_21_price_returns | 20 | 0.003 | 0.999 | N/A |
| ema_12_returns_vwap | 20 | 0.005 | 0.998 | N/A |
| ema_26_returns_vwap | 20 | 0.005 | 0.998 | N/A |
| ema_50_returns_vwap | 20 | 0.005 | 0.999 | N/A |
| sma_100_returns_vwap | 20 | 0.004 | 1.000 | N/A |
| sma_10_returns_vwap | 20 | 0.004 | 0.998 | N/A |
| sma_20_returns_vwap | 20 | 0.005 | 0.998 | N/A |
| sma_50_returns_vwap | 20 | 0.004 | 0.999 | N/A |
| sma_5_returns_vwap | 20 | 0.004 | 0.998 | N/A |
| tema_21_price_returns | 20 | 0.003 | 0.999 | N/A |
| trend_score_14 | 20 | 0.004 | 0.997 | N/A |
| vectorbt_acceleration_trend_strength_10_10_price_returns | 20 | 0.003 | 0.999 | N/A |
| vectorbt_acceleration_trend_strength_10_20_price_returns | 20 | 0.003 | 0.999 | N/A |
| vectorbt_acceleration_trend_strength_5_10_price_returns | 20 | 0.003 | 0.999 | N/A |
| vectorbt_acceleration_trend_strength_5_20_price_returns | 20 | 0.003 | 0.999 | N/A |
| vectorbt_sma_100 | 20 | 0.004 | 0.999 | N/A |
| vectorbt_trend_consistency_10_price_returns | 20 | 0.004 | 0.998 | N/A |
| vectorbt_trend_consistency_20_price_returns | 20 | 0.005 | 0.999 | N/A |
| vectorbt_trend_consistency_50_price_returns | 20 | 0.006 | 0.999 | N/A |
| vectorbt_trend_consistency_5_price_returns | 20 | 0.004 | 0.998 | N/A |
| vectorbt_trend_strength_10_price_returns | 20 | 0.003 | 0.999 | N/A |
| vectorbt_trend_strength_20_price_returns | 20 | 0.003 | 0.998 | N/A |
| vectorbt_trend_strength_50_price_returns | 20 | 0.004 | 0.999 | N/A |
| vectorbt_trend_strength_5_price_returns | 20 | 0.003 | 0.998 | N/A |

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

**Total Features Optimized:** 259 features across 5 categories

### Other Features

Optimized 113 features with optimal + alternative lookback periods:

#### acceleration_features
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### advanced_cumulative_returns_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### advanced_cumulative_returns_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.005
- **Stability Score:** 0.997
- **Optimization Method:** per_feature_mi_curve

#### adx_14_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### apo_12_26_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### ar_1_coefficients_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### aroon_25_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### candlestick_dark_cloud_cover_pattern
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.000
- **Stability Score:** 1.000
- **Optimization Method:** per_feature_mi_curve

#### candlestick_doji_pattern
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.001
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### candlestick_engulfing_pattern
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.001
- **Stability Score:** 1.000
- **Optimization Method:** per_feature_mi_curve

#### candlestick_harami_cross_pattern
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.001
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### candlestick_piercing_line_pattern
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.001
- **Stability Score:** 1.000
- **Optimization Method:** per_feature_mi_curve

#### candlestick_three_black_crows_pattern
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.002
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### candlestick_three_white_soldiers_pattern
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.002
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### cci_20_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### cmf_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### cumulative_returns_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.002
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### cumulative_returns_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.002
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### cycle_length
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.002
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### dfa_slopes
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### directional_signal
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.005
- **Stability Score:** 0.997
- **Optimization Method:** per_feature_mi_curve

#### donchian_channel_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### entropy_rate_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.236_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.236_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.236_5_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.382_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.382_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.382_5_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.5_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.5_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.5_5_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.618_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.618_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.618_5_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.786_10_price_returns
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 10, 20]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.786_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.997
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.786_5_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### fractal_dimension
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.000
- **Stability Score:** 0.500
- **Optimization Method:** per_feature_mi_curve

#### hurst_exponent
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### kama_30_2_30_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.005
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### kst_10_15_20_30_10_10_10_15_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### lempel_ziv_complexity_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### ljung_box_pvalue_20_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### log_returns_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### log_returns_1_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### log_returns_5_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### mama_21_0.05_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### max_drawdown
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### order_flow_imbalance_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### pfe_12_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### pivot_point_5_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### resistance_level_1_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### resistance_level_1_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.005
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### resistance_level_2_5_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 15, 20]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### returns_kurtosis_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### returns_skewness_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### roc_14_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.002
- **Stability Score:** 1.000
- **Optimization Method:** per_feature_mi_curve

#### roc_21_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### roc_30_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### rolling_returns_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.002
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### rolling_returns_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.002
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### rolling_zscore_returns_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### shannon_entropy_20_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### sharpe_ratio_20_0.0_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### simple_returns_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### simple_returns_1_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.002
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### simple_returns_5_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.002
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### spectral_entropy_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### stochastic_14_3_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### stochastic_21_3_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### stochastic_30_3_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### stochastic_kd_14_3
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### support_level_1_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### support_level_1_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### support_level_1_5_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### t3_14_0.7_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.005
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### ultimate_oscillator_7_14_28_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 10]
- **Performance Score:** 0.002
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_5_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.002
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_consistency_10_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_consistency_10_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_consistency_5_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_consistency_5_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_correlation_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.000
- **Stability Score:** 1.000
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_divergence_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 10]
- **Performance Score:** 0.002
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_regime_5_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.001
- **Stability Score:** 1.000
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_regime_5_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.001
- **Stability Score:** 1.000
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_adx_14
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_adx_21
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_adx_9
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_bbands_10_1.5
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_bbands_14_1.5
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_bbands_20_1.5
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_enhanced_ad_line_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_enhanced_ad_line_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_ichimoku_cloud_9_26_52
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.005
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_jerk_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 10]
- **Performance Score:** 0.002
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_jerk_5_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.02_0.2
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.02_0.3
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.05_0.2
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.05_0.3
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.1_0.2
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.1_0.3
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_zigzag_10.0_2
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_zigzag_3.0_2
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_zigzag_5.0_3
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_zigzag_7.0_3
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vwap_deviations_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vwma_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.005
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### wavelet_energy
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### wma_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

### Momentum Features

Optimized 31 features with optimal + alternative lookback periods:

#### advanced_momentum_10_30
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### advanced_momentum_5_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### macd_12_26_9_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.997
- **Optimization Method:** per_feature_mi_curve

#### macd_delta_12_26_9
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### macd_entropy_20_12_26
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.002
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### momentum_14_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### momentum_21_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### momentum_30_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### momentum_endpoints_sma_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### momentum_features
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.005
- **Stability Score:** 0.997
- **Optimization Method:** per_feature_mi_curve

#### rsi_14_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### rsi_21_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### rsi_30_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### rsi_zscore_14_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_momentum_10_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_momentum_10_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_momentum_5_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_momentum_5_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_50_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.002
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_5_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.002
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_acceleration_10_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_acceleration_10_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_acceleration_5_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_acceleration_5_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_comprehensive_14
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_comprehensive_21
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.997
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_comprehensive_30
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.997
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_comprehensive_9
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_momentum_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_momentum_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_momentum_5
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.997
- **Optimization Method:** per_feature_mi_curve

### Volume Features

Optimized 49 features with optimal + alternative lookback periods:

#### analyst_volume_pressure
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### analyst_volume_trend
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### price_volume_oscillator_10_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### price_volume_oscillator_5_15
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_enhanced_obv_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_enhanced_obv_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_enhanced_obv_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_smoothed_obv_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volume_acceleration_5_volume_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.997
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volume_weighted_ad_line_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volume_weighted_ad_line_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volume_weighted_ad_line_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 1.000
- **Optimization Method:** per_feature_mi_curve

#### volume_accumulation_distribution
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_ema_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_ema_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_ema_5
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_ema_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_10_volume_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.002
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_20_volume_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.002
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_5_volume_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_ma_10_10_volume_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 1.000
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_ma_10_5_volume_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 1.000
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_ma_20_10_volume_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.002
- **Stability Score:** 1.000
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_ma_20_5_volume_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.002
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_ma_5_10_volume_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_ma_5_5_volume_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 1.000
- **Optimization Method:** per_feature_mi_curve

#### volume_ma_ratios_20_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_oscillator_10_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_oscillator_5_15
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_percentile_100
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_percentile_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_percentile_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_price_correlation_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_price_correlation_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_price_divergence_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.002
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_price_divergence_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.997
- **Optimization Method:** per_feature_mi_curve

#### volume_price_trend
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.005
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_ratio_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_ratio_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_ratio_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_roc_1
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [3, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 3, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_roc_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_roc_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_roc_5
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_trend_strength_10_30
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_trend_strength_20_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_vwap_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.005
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_vwap_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_zscore_60_252
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

### Volatility Features

Optimized 42 features with optimal + alternative lookback periods:

#### band_limited_volatility
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_100
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_14
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_30
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.005
- **Stability Score:** 0.997
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### returns_volatility_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_volatility_10_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_volatility_10_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_volatility_5_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_volatility_5_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_14
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_30
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_14
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_30
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parkinson_volatility_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parkinson_volatility_14
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parkinson_volatility_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.005
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parkinson_volatility_30
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parkinson_volatility_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_14
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_30
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_acceleration_5_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.997
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_14
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.997
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_30
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_std_10
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_std_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.005
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### volume_std_50
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_volatility_elasticity_20
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

### Trend Features

Optimized 24 features with optimal + alternative lookback periods:

#### dema_21_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### ema_12_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.005
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### ema_26_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.005
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### ema_50_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.005
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### sma_100_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 1.000
- **Optimization Method:** per_feature_mi_curve

#### sma_10_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### sma_20_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.005
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### sma_50_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### sma_5_returns_vwap
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### tema_21_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### trend_score_14
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.997
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_trend_strength_10_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_trend_strength_10_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_trend_strength_5_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_trend_strength_5_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_sma_100
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_consistency_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_consistency_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.005
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_consistency_50_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.006
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_consistency_5_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_strength_10_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_strength_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 8, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_strength_50_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.004
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_strength_5_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.003
- **Stability Score:** 0.998
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

