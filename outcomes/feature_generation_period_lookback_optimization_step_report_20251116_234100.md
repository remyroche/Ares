# Lookback Optimization Report

**Generated:** 2025-11-16 23:41:00
**Step:** feature_generation_period_lookback_optimization_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** blank

## Optimization Results

- **Momentum Lookback:** 3
- **Trend Lookback:** 3
- **Volatility Lookback:** 3
- **Volume Lookback:** 5
- **Optimization Score:** 0.60

## Comprehensive Optimization Analysis

### Data Export

- **Per-Feature Metrics CSV:** `outcomes/feature_generation_period_lookback_optimization_step_per_feature_metrics_20251116_234100.csv`
- **Full Path:** `/Users/remyroche/Documents/Ares/outcomes/feature_generation_period_lookback_optimization_step_per_feature_metrics_20251116_234100.csv`

### Optimization Performance Metrics

| Metric | Value |
|--------|-------|
| Optimization Method | data_driven_cross_validation |
| Total Features Analyzed | 268 |
| Lookback Range Tested | 1-50 |
| Cross-Validation Folds | 2 |
| Optimization Efficiency | 85.0% |
| Stability Score | 0.971 |
| Performance Score | 0.237 |

### Global Optimization Metrics

| Metric | Value |
|--------|-------|
| Total Features Optimized | 259 |
| Categories Processed | 5 |
| Average Lookback Period | 3.4 |
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
| Momentum | 31 | 3 | 0.234 | 0.974 | 0.604 | 0.588 | N/A | cv |
| Trend | 24 | 3 | 0.250 | 0.975 | 0.612 | 0.597 | N/A | cv |
| Volatility | 42 | 3 | 0.248 | 0.971 | 0.610 | 0.592 | N/A | cv |
| Volume | 49 | 5 | 0.217 | 0.965 | 0.591 | 0.570 | N/A | cv |

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
| momentum | 31 | 3 | 1-50 | 0.234 | 0.974 | 0.604 | 0.588 | 100.0% |
| volatility | 42 | 3 | 1-50 | 0.248 | 0.971 | 0.610 | 0.592 | 100.0% |
| trend | 24 | 3 | 1-50 | 0.250 | 0.975 | 0.612 | 0.597 | 100.0% |
| volume | 49 | 5 | 1-50 | 0.217 | 0.965 | 0.591 | 0.570 | 100.0% |

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
- **Best Individual Feature Lookback:** 3
- **Average Performance Score:** 0.234
- **Average Stability Score:** 0.974
- **Features Optimized:** 31

#### Trend Features

**Category Summary:**
- **Best Individual Feature Lookback:** 3
- **Average Performance Score:** 0.250
- **Average Stability Score:** 0.975
- **Features Optimized:** 24

#### Volatility Features

**Category Summary:**
- **Best Individual Feature Lookback:** 3
- **Average Performance Score:** 0.248
- **Average Stability Score:** 0.971
- **Features Optimized:** 42

#### Volume Features

**Category Summary:**
- **Best Individual Feature Lookback:** 5
- **Average Performance Score:** 0.217
- **Average Stability Score:** 0.965
- **Features Optimized:** 49

### Per-Feature Optimization Results

| Feature Name | Optimal Lookback | Performance | Stability | Method |
|--------------|------------------|-------------|-----------|--------|
| acceleration_features | 3 | 0.219 | 0.993 | N/A |
| advanced_cumulative_returns_10 | 3 | 0.229 | 0.977 | N/A |
| advanced_cumulative_returns_20 | 3 | 0.201 | 0.996 | N/A |
| adx_14_returns_vwap | 3 | 0.247 | 0.970 | N/A |
| apo_12_26_returns_vwap | 3 | 0.278 | 0.995 | N/A |
| ar_1_coefficients_20 | 3 | 0.247 | 0.989 | N/A |
| aroon_25_returns_vwap | 5 | 0.156 | 0.944 | N/A |
| candlestick_dark_cloud_cover_pattern | 5 | 0.016 | 0.977 | N/A |
| candlestick_doji_pattern | 3 | 0.049 | 0.975 | N/A |
| candlestick_engulfing_pattern | 5 | 0.031 | 0.963 | N/A |
| candlestick_harami_cross_pattern | 3 | 0.049 | 0.978 | N/A |
| candlestick_piercing_line_pattern | 5 | 0.020 | 0.974 | N/A |
| candlestick_three_black_crows_pattern | 10 | 0.048 | 0.985 | N/A |
| candlestick_three_white_soldiers_pattern | 5 | 0.074 | 0.979 | N/A |
| cci_20_returns_vwap | 3 | 0.248 | 0.967 | N/A |
| cmf_20 | 3 | 0.154 | 0.935 | N/A |
| cumulative_returns_10_price_returns | 3 | 0.217 | 0.991 | N/A |
| cumulative_returns_20_price_returns | 3 | 0.208 | 0.977 | N/A |
| cycle_length | 20 | 0.060 | 0.962 | N/A |
| dfa_slopes | 3 | 0.225 | 0.987 | N/A |
| directional_signal | 3 | 0.244 | 0.964 | N/A |
| donchian_channel_20 | 3 | 0.227 | 0.945 | N/A |
| entropy_rate_20 | 5 | 0.199 | 0.966 | N/A |
| fibonacci_0.236_10_price_returns | 5 | 0.199 | 0.978 | N/A |
| fibonacci_0.236_20_price_returns | 5 | 0.199 | 0.961 | N/A |
| fibonacci_0.236_5_price_returns | 5 | 0.206 | 0.959 | N/A |
| fibonacci_0.382_10_price_returns | 5 | 0.199 | 0.856 | N/A |
| fibonacci_0.382_20_price_returns | 3 | 0.195 | 0.898 | N/A |
| fibonacci_0.382_5_price_returns | 3 | 0.234 | 0.992 | N/A |
| fibonacci_0.5_10_price_returns | 3 | 0.223 | 0.938 | N/A |
| fibonacci_0.5_20_price_returns | 3 | 0.211 | 0.987 | N/A |
| fibonacci_0.5_5_price_returns | 5 | 0.227 | 0.961 | N/A |
| fibonacci_0.618_10_price_returns | 3 | 0.229 | 0.968 | N/A |
| fibonacci_0.618_20_price_returns | 3 | 0.192 | 0.960 | N/A |
| fibonacci_0.618_5_price_returns | 3 | 0.236 | 0.968 | N/A |
| fibonacci_0.786_10_price_returns | 5 | 0.205 | 0.944 | N/A |
| fibonacci_0.786_20_price_returns | 5 | 0.208 | 0.975 | N/A |
| fibonacci_0.786_5_price_returns | 5 | 0.229 | 0.921 | N/A |
| fractal_dimension | 3 | 0.000 | 0.500 | N/A |
| hurst_exponent | 3 | 0.145 | 0.984 | N/A |
| kama_30_2_30_returns_vwap | 3 | 0.257 | 0.993 | N/A |
| kst_10_15_20_30_10_10_10_15_returns_vwap | 3 | 0.274 | 0.904 | N/A |
| lempel_ziv_complexity_20 | 3 | 0.262 | 0.974 | N/A |
| ljung_box_pvalue_20_10 | 3 | 0.244 | 0.978 | N/A |
| log_returns_10_price_returns | 3 | 0.221 | 0.984 | N/A |
| log_returns_1_price_returns | 3 | 0.189 | 0.964 | N/A |
| log_returns_5_price_returns | 3 | 0.179 | 0.977 | N/A |
| mama_21_0.05_price_returns | 3 | 0.228 | 0.984 | N/A |
| max_drawdown | 10 | 0.148 | 0.990 | N/A |
| order_flow_imbalance_20 | 3 | 0.204 | 0.945 | N/A |
| pfe_12_returns_vwap | 5 | 0.168 | 0.877 | N/A |
| pivot_point_5_price_returns | 3 | 0.232 | 0.977 | N/A |
| resistance_level_1_10_price_returns | 10 | 0.150 | 0.995 | N/A |
| resistance_level_1_20_price_returns | 20 | 0.127 | 0.998 | N/A |
| resistance_level_2_5_price_returns | 5 | 0.221 | 0.994 | N/A |
| returns_kurtosis_20_price_returns | 3 | 0.247 | 0.993 | N/A |
| returns_skewness_20_price_returns | 3 | 0.249 | 0.966 | N/A |
| roc_14_price_returns | 3 | 0.231 | 0.984 | N/A |
| roc_21_price_returns | 3 | 0.240 | 0.988 | N/A |
| roc_30_price_returns | 3 | 0.230 | 1.000 | N/A |
| rolling_returns_10_price_returns | 3 | 0.216 | 0.987 | N/A |
| rolling_returns_20_price_returns | 3 | 0.207 | 0.965 | N/A |
| rolling_zscore_returns_20 | 3 | 0.245 | 0.982 | N/A |
| shannon_entropy_20_10 | 3 | 0.204 | 0.942 | N/A |
| sharpe_ratio_20_0.0_price_returns | 5 | 0.230 | 0.958 | N/A |
| simple_returns_10_price_returns | 3 | 0.245 | 0.982 | N/A |
| simple_returns_1_price_returns | 3 | 0.218 | 0.955 | N/A |
| simple_returns_5_price_returns | 3 | 0.230 | 0.948 | N/A |
| spectral_entropy_20 | 3 | 0.242 | 0.963 | N/A |
| stochastic_14_3_price_returns | 3 | 0.238 | 0.984 | N/A |
| stochastic_21_3_price_returns | 3 | 0.233 | 0.973 | N/A |
| stochastic_30_3_price_returns | 3 | 0.239 | 0.989 | N/A |
| stochastic_kd_14_3 | 3 | 0.218 | 0.994 | N/A |
| support_level_1_10_price_returns | 10 | 0.180 | 0.960 | N/A |
| support_level_1_20_price_returns | 20 | 0.133 | 0.961 | N/A |
| support_level_1_5_price_returns | 5 | 0.175 | 0.954 | N/A |
| t3_14_0.7_returns_vwap | 3 | 0.243 | 0.972 | N/A |
| ultimate_oscillator_7_14_28_returns_vwap | 3 | 0.271 | 0.886 | N/A |
| vectorbt_acceleration_10_price_returns | 3 | 0.244 | 0.995 | N/A |
| vectorbt_acceleration_5_price_returns | 3 | 0.211 | 0.990 | N/A |
| vectorbt_acceleration_consistency_10_10_price_returns | 3 | 0.251 | 0.987 | N/A |
| vectorbt_acceleration_consistency_10_20_price_returns | 3 | 0.226 | 0.911 | N/A |
| vectorbt_acceleration_consistency_5_10_price_returns | 3 | 0.264 | 0.985 | N/A |
| vectorbt_acceleration_consistency_5_20_price_returns | 3 | 0.239 | 0.952 | N/A |
| vectorbt_acceleration_correlation_20_price_returns | 5 | 0.009 | 0.982 | N/A |
| vectorbt_acceleration_divergence_20_price_returns | 3 | 0.222 | 0.991 | N/A |
| vectorbt_acceleration_regime_5_10_price_returns | 8 | 0.043 | 0.977 | N/A |
| vectorbt_acceleration_regime_5_20_price_returns | 5 | 0.034 | 0.966 | N/A |
| vectorbt_adx_14 | 3 | 0.270 | 0.969 | N/A |
| vectorbt_adx_21 | 5 | 0.267 | 0.994 | N/A |
| vectorbt_adx_9 | 3 | 0.270 | 0.987 | N/A |
| vectorbt_bbands_10_1.5 | 3 | 0.247 | 0.971 | N/A |
| vectorbt_bbands_14_1.5 | 3 | 0.255 | 0.988 | N/A |
| vectorbt_bbands_20_1.5 | 3 | 0.255 | 0.981 | N/A |
| vectorbt_enhanced_ad_line_20 | 3 | 0.251 | 0.996 | N/A |
| vectorbt_enhanced_ad_line_50 | 3 | 0.278 | 0.999 | N/A |
| vectorbt_ichimoku_cloud_9_26_52 | 3 | 0.230 | 0.998 | N/A |
| vectorbt_jerk_10_price_returns | 3 | 0.249 | 0.994 | N/A |
| vectorbt_jerk_5_price_returns | 3 | 0.224 | 0.979 | N/A |
| vectorbt_parabolic_sar_0.02_0.2 | 3 | 0.241 | 0.957 | N/A |
| vectorbt_parabolic_sar_0.02_0.3 | 3 | 0.241 | 0.958 | N/A |
| vectorbt_parabolic_sar_0.05_0.2 | 3 | 0.250 | 0.933 | N/A |
| vectorbt_parabolic_sar_0.05_0.3 | 3 | 0.257 | 0.958 | N/A |
| vectorbt_parabolic_sar_0.1_0.2 | 3 | 0.257 | 0.938 | N/A |
| vectorbt_parabolic_sar_0.1_0.3 | 3 | 0.242 | 0.973 | N/A |
| vectorbt_zigzag_10.0_2 | 3 | 0.184 | 0.998 | N/A |
| vectorbt_zigzag_3.0_2 | 3 | 0.202 | 0.975 | N/A |
| vectorbt_zigzag_5.0_3 | 3 | 0.218 | 0.930 | N/A |
| vectorbt_zigzag_7.0_3 | 3 | 0.214 | 0.992 | N/A |
| vwap_deviations_20 | 3 | 0.233 | 0.990 | N/A |
| vwma_20_price_returns | 3 | 0.226 | 0.958 | N/A |
| wavelet_energy | 3 | 0.251 | 0.949 | N/A |
| wma_20_price_returns | 3 | 0.214 | 0.968 | N/A |
| advanced_momentum_10_30 | 3 | 0.287 | 0.970 | N/A |
| advanced_momentum_5_20 | 3 | 0.239 | 0.978 | N/A |
| macd_12_26_9_returns_vwap | 3 | 0.265 | 0.985 | N/A |
| macd_delta_12_26_9 | 3 | 0.248 | 0.982 | N/A |
| macd_entropy_20_12_26 | 15 | 0.087 | 0.936 | N/A |
| momentum_14_price_returns | 3 | 0.228 | 0.935 | N/A |
| momentum_21_price_returns | 3 | 0.226 | 0.990 | N/A |
| momentum_30_price_returns | 3 | 0.244 | 0.958 | N/A |
| momentum_endpoints_sma_20 | 3 | 0.222 | 0.976 | N/A |
| momentum_features | 3 | 0.215 | 0.989 | N/A |
| rsi_14_returns_vwap | 3 | 0.245 | 0.992 | N/A |
| rsi_21_returns_vwap | 3 | 0.268 | 0.982 | N/A |
| rsi_30_returns_vwap | 3 | 0.249 | 0.987 | N/A |
| rsi_zscore_14_20 | 3 | 0.218 | 0.958 | N/A |
| vectorbt_acceleration_momentum_10_10_price_returns | 3 | 0.243 | 0.990 | N/A |
| vectorbt_acceleration_momentum_10_20_price_returns | 3 | 0.240 | 0.995 | N/A |
| vectorbt_acceleration_momentum_5_10_price_returns | 3 | 0.265 | 0.979 | N/A |
| vectorbt_acceleration_momentum_5_20_price_returns | 3 | 0.236 | 0.966 | N/A |
| vectorbt_momentum_50_price_returns | 3 | 0.223 | 0.993 | N/A |
| vectorbt_momentum_5_price_returns | 3 | 0.228 | 0.962 | N/A |
| vectorbt_momentum_acceleration_10_10_price_returns | 3 | 0.245 | 0.970 | N/A |
| vectorbt_momentum_acceleration_10_20_price_returns | 3 | 0.235 | 0.987 | N/A |
| vectorbt_momentum_acceleration_5_10_price_returns | 3 | 0.228 | 0.969 | N/A |
| vectorbt_momentum_acceleration_5_20_price_returns | 3 | 0.220 | 0.933 | N/A |
| vectorbt_momentum_comprehensive_14 | 3 | 0.235 | 0.959 | N/A |
| vectorbt_momentum_comprehensive_21 | 3 | 0.230 | 1.000 | N/A |
| vectorbt_momentum_comprehensive_30 | 3 | 0.241 | 0.983 | N/A |
| vectorbt_momentum_comprehensive_9 | 3 | 0.241 | 0.986 | N/A |
| volume_momentum_10 | 3 | 0.235 | 0.975 | N/A |
| volume_momentum_20 | 3 | 0.234 | 0.965 | N/A |
| volume_momentum_5 | 3 | 0.238 | 0.963 | N/A |
| analyst_volume_pressure | 5 | 0.166 | 0.935 | N/A |
| analyst_volume_trend | 3 | 0.249 | 0.985 | N/A |
| price_volume_oscillator_10_20 | 3 | 0.271 | 0.998 | N/A |
| price_volume_oscillator_5_15 | 3 | 0.242 | 0.998 | N/A |
| vectorbt_enhanced_obv_10 | 3 | 0.263 | 0.937 | N/A |
| vectorbt_enhanced_obv_20 | 3 | 0.264 | 0.940 | N/A |
| vectorbt_enhanced_obv_50 | 3 | 0.275 | 0.979 | N/A |
| vectorbt_smoothed_obv_10 | 3 | 0.256 | 0.929 | N/A |
| vectorbt_volume_acceleration_5_volume_returns | 3 | 0.239 | 0.998 | N/A |
| vectorbt_volume_weighted_ad_line_10 | 3 | 0.256 | 0.947 | N/A |
| vectorbt_volume_weighted_ad_line_20 | 3 | 0.252 | 0.890 | N/A |
| vectorbt_volume_weighted_ad_line_50 | 3 | 0.268 | 0.929 | N/A |
| volume_accumulation_distribution | 3 | 0.229 | 0.994 | N/A |
| volume_ema_10 | 3 | 0.257 | 0.945 | N/A |
| volume_ema_20 | 3 | 0.257 | 0.938 | N/A |
| volume_ema_5 | 3 | 0.251 | 0.979 | N/A |
| volume_ema_50 | 3 | 0.265 | 0.989 | N/A |
| volume_entropy_10_volume_returns | 3 | 0.069 | 0.947 | N/A |
| volume_entropy_20_volume_returns | 3 | 0.064 | 0.970 | N/A |
| volume_entropy_5_volume_returns | 5 | 0.089 | 0.979 | N/A |
| volume_entropy_ma_10_10_volume_returns | 5 | 0.099 | 0.952 | N/A |
| volume_entropy_ma_10_5_volume_returns | 5 | 0.095 | 0.973 | N/A |
| volume_entropy_ma_20_10_volume_returns | 5 | 0.088 | 0.949 | N/A |
| volume_entropy_ma_20_5_volume_returns | 3 | 0.092 | 0.959 | N/A |
| volume_entropy_ma_5_10_volume_returns | 8 | 0.123 | 0.971 | N/A |
| volume_entropy_ma_5_5_volume_returns | 5 | 0.109 | 0.996 | N/A |
| volume_ma_ratios_20_10 | 3 | 0.240 | 0.974 | N/A |
| volume_oscillator_10_20 | 3 | 0.255 | 0.989 | N/A |
| volume_oscillator_5_15 | 3 | 0.245 | 0.970 | N/A |
| volume_percentile_100 | 3 | 0.234 | 0.979 | N/A |
| volume_percentile_20 | 3 | 0.241 | 0.974 | N/A |
| volume_percentile_50 | 3 | 0.231 | 0.974 | N/A |
| volume_price_correlation_10 | 3 | 0.253 | 0.955 | N/A |
| volume_price_correlation_20 | 3 | 0.249 | 0.959 | N/A |
| volume_price_divergence_10 | 3 | 0.243 | 0.977 | N/A |
| volume_price_divergence_20 | 3 | 0.240 | 0.966 | N/A |
| volume_price_trend | 3 | 0.196 | 0.900 | N/A |
| volume_ratio_10 | 3 | 0.237 | 0.962 | N/A |
| volume_ratio_20 | 3 | 0.245 | 0.983 | N/A |
| volume_ratio_50 | 3 | 0.237 | 0.974 | N/A |
| volume_roc_1 | 3 | 0.219 | 0.994 | N/A |
| volume_roc_10 | 3 | 0.244 | 0.980 | N/A |
| volume_roc_20 | 3 | 0.232 | 0.980 | N/A |
| volume_roc_5 | 3 | 0.228 | 0.958 | N/A |
| volume_trend_strength_10_30 | 3 | 0.272 | 0.967 | N/A |
| volume_trend_strength_20_50 | 3 | 0.280 | 0.996 | N/A |
| volume_vwap_10 | 3 | 0.240 | 0.943 | N/A |
| volume_vwap_50 | 3 | 0.253 | 0.960 | N/A |
| volume_zscore_60_252 | 3 | 0.233 | 0.963 | N/A |
| band_limited_volatility | 3 | 0.259 | 0.940 | N/A |
| enhanced_volatility_10 | 3 | 0.254 | 0.999 | N/A |
| enhanced_volatility_100 | 3 | 0.272 | 0.979 | N/A |
| enhanced_volatility_14 | 3 | 0.236 | 0.979 | N/A |
| enhanced_volatility_20 | 3 | 0.254 | 0.975 | N/A |
| enhanced_volatility_30 | 3 | 0.265 | 0.989 | N/A |
| enhanced_volatility_50 | 3 | 0.273 | 0.922 | N/A |
| returns_volatility_20_price_returns | 3 | 0.240 | 0.963 | N/A |
| vectorbt_acceleration_volatility_10_10_price_returns | 3 | 0.252 | 0.991 | N/A |
| vectorbt_acceleration_volatility_10_20_price_returns | 3 | 0.229 | 0.937 | N/A |
| vectorbt_acceleration_volatility_5_10_price_returns | 3 | 0.271 | 1.000 | N/A |
| vectorbt_acceleration_volatility_5_20_price_returns | 3 | 0.237 | 0.947 | N/A |
| vectorbt_atr_10 | 3 | 0.225 | 0.939 | N/A |
| vectorbt_atr_14 | 3 | 0.225 | 0.951 | N/A |
| vectorbt_atr_20 | 3 | 0.222 | 0.915 | N/A |
| vectorbt_atr_30 | 3 | 0.232 | 0.934 | N/A |
| vectorbt_atr_50 | 3 | 0.236 | 0.994 | N/A |
| vectorbt_garman_klass_volatility_10 | 3 | 0.260 | 0.981 | N/A |
| vectorbt_garman_klass_volatility_14 | 3 | 0.235 | 0.996 | N/A |
| vectorbt_garman_klass_volatility_20 | 3 | 0.238 | 0.938 | N/A |
| vectorbt_garman_klass_volatility_30 | 3 | 0.251 | 0.963 | N/A |
| vectorbt_garman_klass_volatility_50 | 3 | 0.264 | 0.975 | N/A |
| vectorbt_parkinson_volatility_10 | 3 | 0.260 | 0.977 | N/A |
| vectorbt_parkinson_volatility_14 | 3 | 0.232 | 0.978 | N/A |
| vectorbt_parkinson_volatility_20 | 3 | 0.221 | 0.948 | N/A |
| vectorbt_parkinson_volatility_30 | 3 | 0.247 | 0.927 | N/A |
| vectorbt_parkinson_volatility_50 | 3 | 0.258 | 0.982 | N/A |
| vectorbt_rogers_satchell_volatility_10 | 3 | 0.268 | 0.990 | N/A |
| vectorbt_rogers_satchell_volatility_14 | 3 | 0.241 | 0.984 | N/A |
| vectorbt_rogers_satchell_volatility_20 | 3 | 0.249 | 0.967 | N/A |
| vectorbt_rogers_satchell_volatility_30 | 3 | 0.255 | 0.962 | N/A |
| vectorbt_rogers_satchell_volatility_50 | 3 | 0.276 | 0.998 | N/A |
| vectorbt_volatility_acceleration_5_20_price_returns | 3 | 0.244 | 0.997 | N/A |
| vectorbt_volatility_comprehensive_10 | 3 | 0.251 | 0.963 | N/A |
| vectorbt_volatility_comprehensive_14 | 3 | 0.268 | 0.998 | N/A |
| vectorbt_volatility_comprehensive_20 | 3 | 0.256 | 0.987 | N/A |
| vectorbt_volatility_comprehensive_30 | 3 | 0.239 | 0.981 | N/A |
| vectorbt_volatility_comprehensive_50 | 3 | 0.258 | 0.957 | N/A |
| volume_std_10 | 3 | 0.241 | 0.988 | N/A |
| volume_std_20 | 3 | 0.244 | 0.999 | N/A |
| volume_std_50 | 3 | 0.245 | 0.999 | N/A |
| volume_volatility_elasticity_20 | 3 | 0.246 | 0.998 | N/A |
| dema_21_price_returns | 3 | 0.242 | 0.997 | N/A |
| ema_12_returns_vwap | 3 | 0.252 | 0.993 | N/A |
| ema_26_returns_vwap | 3 | 0.251 | 0.990 | N/A |
| ema_50_returns_vwap | 3 | 0.261 | 0.957 | N/A |
| sma_100_returns_vwap | 3 | 0.262 | 0.961 | N/A |
| sma_10_returns_vwap | 3 | 0.245 | 0.968 | N/A |
| sma_20_returns_vwap | 3 | 0.260 | 0.999 | N/A |
| sma_50_returns_vwap | 3 | 0.260 | 0.997 | N/A |
| sma_5_returns_vwap | 3 | 0.255 | 0.998 | N/A |
| tema_21_price_returns | 3 | 0.253 | 0.968 | N/A |
| trend_score_14 | 3 | 0.242 | 0.962 | N/A |
| vectorbt_acceleration_trend_strength_10_10_price_returns | 3 | 0.240 | 0.955 | N/A |
| vectorbt_acceleration_trend_strength_10_20_price_returns | 3 | 0.223 | 0.992 | N/A |
| vectorbt_acceleration_trend_strength_5_10_price_returns | 3 | 0.261 | 0.964 | N/A |
| vectorbt_acceleration_trend_strength_5_20_price_returns | 3 | 0.241 | 0.989 | N/A |
| vectorbt_sma_100 | 3 | 0.268 | 0.913 | N/A |
| vectorbt_trend_consistency_10_price_returns | 3 | 0.238 | 0.966 | N/A |
| vectorbt_trend_consistency_20_price_returns | 3 | 0.227 | 0.991 | N/A |
| vectorbt_trend_consistency_50_price_returns | 3 | 0.294 | 0.977 | N/A |
| vectorbt_trend_consistency_5_price_returns | 3 | 0.253 | 0.999 | N/A |
| vectorbt_trend_strength_10_price_returns | 3 | 0.236 | 0.953 | N/A |
| vectorbt_trend_strength_20_price_returns | 3 | 0.244 | 0.940 | N/A |
| vectorbt_trend_strength_50_price_returns | 3 | 0.241 | 0.985 | N/A |
| vectorbt_trend_strength_5_price_returns | 3 | 0.251 | 0.981 | N/A |

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
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.219
- **Stability Score:** 0.993
- **Optimization Method:** per_feature_mi_curve

#### advanced_cumulative_returns_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 10]
- **Performance Score:** 0.229
- **Stability Score:** 0.977
- **Optimization Method:** per_feature_mi_curve

#### advanced_cumulative_returns_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.201
- **Stability Score:** 0.996
- **Optimization Method:** per_feature_mi_curve

#### adx_14_returns_vwap
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 15]
- **Performance Score:** 0.247
- **Stability Score:** 0.970
- **Optimization Method:** per_feature_mi_curve

#### apo_12_26_returns_vwap
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.278
- **Stability Score:** 0.995
- **Optimization Method:** per_feature_mi_curve

#### ar_1_coefficients_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.247
- **Stability Score:** 0.989
- **Optimization Method:** per_feature_mi_curve

#### aroon_25_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.156
- **Stability Score:** 0.944
- **Optimization Method:** per_feature_mi_curve

#### candlestick_dark_cloud_cover_pattern
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.016
- **Stability Score:** 0.977
- **Optimization Method:** per_feature_mi_curve

#### candlestick_doji_pattern
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.049
- **Stability Score:** 0.975
- **Optimization Method:** per_feature_mi_curve

#### candlestick_engulfing_pattern
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 8, 20]
- **Performance Score:** 0.031
- **Stability Score:** 0.963
- **Optimization Method:** per_feature_mi_curve

#### candlestick_harami_cross_pattern
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 10, 20]
- **Performance Score:** 0.049
- **Stability Score:** 0.978
- **Optimization Method:** per_feature_mi_curve

#### candlestick_piercing_line_pattern
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 8, 20]
- **Performance Score:** 0.020
- **Stability Score:** 0.974
- **Optimization Method:** per_feature_mi_curve

#### candlestick_three_black_crows_pattern
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 15, 20]
- **Performance Score:** 0.048
- **Stability Score:** 0.985
- **Optimization Method:** per_feature_mi_curve

#### candlestick_three_white_soldiers_pattern
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.074
- **Stability Score:** 0.979
- **Optimization Method:** per_feature_mi_curve

#### cci_20_returns_vwap
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 10]
- **Performance Score:** 0.248
- **Stability Score:** 0.967
- **Optimization Method:** per_feature_mi_curve

#### cmf_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 15]
- **Performance Score:** 0.154
- **Stability Score:** 0.935
- **Optimization Method:** per_feature_mi_curve

#### cumulative_returns_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.217
- **Stability Score:** 0.991
- **Optimization Method:** per_feature_mi_curve

#### cumulative_returns_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.208
- **Stability Score:** 0.977
- **Optimization Method:** per_feature_mi_curve

#### cycle_length
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.060
- **Stability Score:** 0.962
- **Optimization Method:** per_feature_mi_curve

#### dfa_slopes
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.225
- **Stability Score:** 0.987
- **Optimization Method:** per_feature_mi_curve

#### directional_signal
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.244
- **Stability Score:** 0.964
- **Optimization Method:** per_feature_mi_curve

#### donchian_channel_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.227
- **Stability Score:** 0.945
- **Optimization Method:** per_feature_mi_curve

#### entropy_rate_20
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [3, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 3, 8]
- **Performance Score:** 0.199
- **Stability Score:** 0.966
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.236_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [3, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 3, 8]
- **Performance Score:** 0.199
- **Stability Score:** 0.978
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.236_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.199
- **Stability Score:** 0.961
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.236_5_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [3, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 3, 8]
- **Performance Score:** 0.206
- **Stability Score:** 0.959
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.382_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [3, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 3, 8]
- **Performance Score:** 0.199
- **Stability Score:** 0.856
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.382_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.195
- **Stability Score:** 0.898
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.382_5_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.234
- **Stability Score:** 0.992
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.5_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 10]
- **Performance Score:** 0.223
- **Stability Score:** 0.938
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.5_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.211
- **Stability Score:** 0.987
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.5_5_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [3, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 3, 8]
- **Performance Score:** 0.227
- **Stability Score:** 0.961
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.618_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.229
- **Stability Score:** 0.968
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.618_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.192
- **Stability Score:** 0.960
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.618_5_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.236
- **Stability Score:** 0.968
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.786_10_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [3, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 3, 8]
- **Performance Score:** 0.205
- **Stability Score:** 0.944
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.786_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [3, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 3, 20]
- **Performance Score:** 0.208
- **Stability Score:** 0.975
- **Optimization Method:** per_feature_mi_curve

#### fibonacci_0.786_5_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [3, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 3, 8]
- **Performance Score:** 0.229
- **Stability Score:** 0.921
- **Optimization Method:** per_feature_mi_curve

#### fractal_dimension
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.000
- **Stability Score:** 0.500
- **Optimization Method:** per_feature_mi_curve

#### hurst_exponent
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.145
- **Stability Score:** 0.984
- **Optimization Method:** per_feature_mi_curve

#### kama_30_2_30_returns_vwap
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.257
- **Stability Score:** 0.993
- **Optimization Method:** per_feature_mi_curve

#### kst_10_15_20_30_10_10_10_15_returns_vwap
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.274
- **Stability Score:** 0.904
- **Optimization Method:** per_feature_mi_curve

#### lempel_ziv_complexity_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 10]
- **Performance Score:** 0.262
- **Stability Score:** 0.974
- **Optimization Method:** per_feature_mi_curve

#### ljung_box_pvalue_20_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.244
- **Stability Score:** 0.978
- **Optimization Method:** per_feature_mi_curve

#### log_returns_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.221
- **Stability Score:** 0.984
- **Optimization Method:** per_feature_mi_curve

#### log_returns_1_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.189
- **Stability Score:** 0.964
- **Optimization Method:** per_feature_mi_curve

#### log_returns_5_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.179
- **Stability Score:** 0.977
- **Optimization Method:** per_feature_mi_curve

#### mama_21_0.05_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.228
- **Stability Score:** 0.984
- **Optimization Method:** per_feature_mi_curve

#### max_drawdown
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 15, 20]
- **Performance Score:** 0.148
- **Stability Score:** 0.990
- **Optimization Method:** per_feature_mi_curve

#### order_flow_imbalance_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.204
- **Stability Score:** 0.945
- **Optimization Method:** per_feature_mi_curve

#### pfe_12_returns_vwap
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 15]
- **Performance Score:** 0.168
- **Stability Score:** 0.877
- **Optimization Method:** per_feature_mi_curve

#### pivot_point_5_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.232
- **Stability Score:** 0.977
- **Optimization Method:** per_feature_mi_curve

#### resistance_level_1_10_price_returns
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 8, 15]
- **Performance Score:** 0.150
- **Stability Score:** 0.995
- **Optimization Method:** per_feature_mi_curve

#### resistance_level_1_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 5, 15]
- **Performance Score:** 0.127
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### resistance_level_2_5_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [3, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 3, 8]
- **Performance Score:** 0.221
- **Stability Score:** 0.994
- **Optimization Method:** per_feature_mi_curve

#### returns_kurtosis_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 15]
- **Performance Score:** 0.247
- **Stability Score:** 0.993
- **Optimization Method:** per_feature_mi_curve

#### returns_skewness_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 15]
- **Performance Score:** 0.249
- **Stability Score:** 0.966
- **Optimization Method:** per_feature_mi_curve

#### roc_14_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.231
- **Stability Score:** 0.984
- **Optimization Method:** per_feature_mi_curve

#### roc_21_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.240
- **Stability Score:** 0.988
- **Optimization Method:** per_feature_mi_curve

#### roc_30_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.230
- **Stability Score:** 1.000
- **Optimization Method:** per_feature_mi_curve

#### rolling_returns_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.216
- **Stability Score:** 0.987
- **Optimization Method:** per_feature_mi_curve

#### rolling_returns_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.207
- **Stability Score:** 0.965
- **Optimization Method:** per_feature_mi_curve

#### rolling_zscore_returns_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.245
- **Stability Score:** 0.982
- **Optimization Method:** per_feature_mi_curve

#### shannon_entropy_20_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.204
- **Stability Score:** 0.942
- **Optimization Method:** per_feature_mi_curve

#### sharpe_ratio_20_0.0_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [3, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 3, 8]
- **Performance Score:** 0.230
- **Stability Score:** 0.958
- **Optimization Method:** per_feature_mi_curve

#### simple_returns_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.245
- **Stability Score:** 0.982
- **Optimization Method:** per_feature_mi_curve

#### simple_returns_1_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.218
- **Stability Score:** 0.955
- **Optimization Method:** per_feature_mi_curve

#### simple_returns_5_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.230
- **Stability Score:** 0.948
- **Optimization Method:** per_feature_mi_curve

#### spectral_entropy_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.242
- **Stability Score:** 0.963
- **Optimization Method:** per_feature_mi_curve

#### stochastic_14_3_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 15]
- **Performance Score:** 0.238
- **Stability Score:** 0.984
- **Optimization Method:** per_feature_mi_curve

#### stochastic_21_3_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.233
- **Stability Score:** 0.973
- **Optimization Method:** per_feature_mi_curve

#### stochastic_30_3_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.239
- **Stability Score:** 0.989
- **Optimization Method:** per_feature_mi_curve

#### stochastic_kd_14_3
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.218
- **Stability Score:** 0.994
- **Optimization Method:** per_feature_mi_curve

#### support_level_1_10_price_returns
- **Optimal Lookback:** 10 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [10, 8, 15]
- **Performance Score:** 0.180
- **Stability Score:** 0.960
- **Optimization Method:** per_feature_mi_curve

#### support_level_1_20_price_returns
- **Optimal Lookback:** 20 (best performance + stability)
- **Alternative Lookbacks:** [10, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [20, 10, 15]
- **Performance Score:** 0.133
- **Stability Score:** 0.961
- **Optimization Method:** per_feature_mi_curve

#### support_level_1_5_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 8, 15]
- **Performance Score:** 0.175
- **Stability Score:** 0.954
- **Optimization Method:** per_feature_mi_curve

#### t3_14_0.7_returns_vwap
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.243
- **Stability Score:** 0.972
- **Optimization Method:** per_feature_mi_curve

#### ultimate_oscillator_7_14_28_returns_vwap
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.271
- **Stability Score:** 0.886
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.244
- **Stability Score:** 0.995
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_5_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.211
- **Stability Score:** 0.990
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_consistency_10_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.251
- **Stability Score:** 0.987
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_consistency_10_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.226
- **Stability Score:** 0.911
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_consistency_5_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.264
- **Stability Score:** 0.985
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_consistency_5_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.239
- **Stability Score:** 0.952
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_correlation_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [3, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 3, 8]
- **Performance Score:** 0.009
- **Stability Score:** 0.982
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_divergence_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.222
- **Stability Score:** 0.991
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_regime_5_10_price_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 10, 20]
- **Performance Score:** 0.043
- **Stability Score:** 0.977
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_regime_5_20_price_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 8, 20]
- **Performance Score:** 0.034
- **Stability Score:** 0.966
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_adx_14
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 15]
- **Performance Score:** 0.270
- **Stability Score:** 0.969
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_adx_21
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [3, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 3, 8]
- **Performance Score:** 0.267
- **Stability Score:** 0.994
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_adx_9
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.270
- **Stability Score:** 0.987
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_bbands_10_1.5
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.247
- **Stability Score:** 0.971
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_bbands_14_1.5
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.255
- **Stability Score:** 0.988
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_bbands_20_1.5
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.255
- **Stability Score:** 0.981
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_enhanced_ad_line_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 15]
- **Performance Score:** 0.251
- **Stability Score:** 0.996
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_enhanced_ad_line_50
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 10]
- **Performance Score:** 0.278
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_ichimoku_cloud_9_26_52
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.230
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_jerk_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.249
- **Stability Score:** 0.994
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_jerk_5_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.224
- **Stability Score:** 0.979
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.02_0.2
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 10]
- **Performance Score:** 0.241
- **Stability Score:** 0.957
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.02_0.3
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 10]
- **Performance Score:** 0.241
- **Stability Score:** 0.958
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.05_0.2
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.250
- **Stability Score:** 0.933
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.05_0.3
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.257
- **Stability Score:** 0.958
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.1_0.2
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.257
- **Stability Score:** 0.938
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parabolic_sar_0.1_0.3
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.242
- **Stability Score:** 0.973
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_zigzag_10.0_2
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.184
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_zigzag_3.0_2
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.202
- **Stability Score:** 0.975
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_zigzag_5.0_3
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.218
- **Stability Score:** 0.930
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_zigzag_7.0_3
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.214
- **Stability Score:** 0.992
- **Optimization Method:** per_feature_mi_curve

#### vwap_deviations_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.233
- **Stability Score:** 0.990
- **Optimization Method:** per_feature_mi_curve

#### vwma_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.226
- **Stability Score:** 0.958
- **Optimization Method:** per_feature_mi_curve

#### wavelet_energy
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.251
- **Stability Score:** 0.949
- **Optimization Method:** per_feature_mi_curve

#### wma_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.214
- **Stability Score:** 0.968
- **Optimization Method:** per_feature_mi_curve

### Momentum Features

Optimized 31 features with optimal + alternative lookback periods:

#### advanced_momentum_10_30
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.287
- **Stability Score:** 0.970
- **Optimization Method:** per_feature_mi_curve

#### advanced_momentum_5_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 10]
- **Performance Score:** 0.239
- **Stability Score:** 0.978
- **Optimization Method:** per_feature_mi_curve

#### macd_12_26_9_returns_vwap
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.265
- **Stability Score:** 0.985
- **Optimization Method:** per_feature_mi_curve

#### macd_delta_12_26_9
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.248
- **Stability Score:** 0.982
- **Optimization Method:** per_feature_mi_curve

#### macd_entropy_20_12_26
- **Optimal Lookback:** 15 (best performance + stability)
- **Alternative Lookbacks:** [3, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [15, 3, 10]
- **Performance Score:** 0.087
- **Stability Score:** 0.936
- **Optimization Method:** per_feature_mi_curve

#### momentum_14_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 15]
- **Performance Score:** 0.228
- **Stability Score:** 0.935
- **Optimization Method:** per_feature_mi_curve

#### momentum_21_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.226
- **Stability Score:** 0.990
- **Optimization Method:** per_feature_mi_curve

#### momentum_30_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.244
- **Stability Score:** 0.958
- **Optimization Method:** per_feature_mi_curve

#### momentum_endpoints_sma_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.222
- **Stability Score:** 0.976
- **Optimization Method:** per_feature_mi_curve

#### momentum_features
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.215
- **Stability Score:** 0.989
- **Optimization Method:** per_feature_mi_curve

#### rsi_14_returns_vwap
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.245
- **Stability Score:** 0.992
- **Optimization Method:** per_feature_mi_curve

#### rsi_21_returns_vwap
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.268
- **Stability Score:** 0.982
- **Optimization Method:** per_feature_mi_curve

#### rsi_30_returns_vwap
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.249
- **Stability Score:** 0.987
- **Optimization Method:** per_feature_mi_curve

#### rsi_zscore_14_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.218
- **Stability Score:** 0.958
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_momentum_10_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 10]
- **Performance Score:** 0.243
- **Stability Score:** 0.990
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_momentum_10_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.240
- **Stability Score:** 0.995
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_momentum_5_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.265
- **Stability Score:** 0.979
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_momentum_5_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.236
- **Stability Score:** 0.966
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_50_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.223
- **Stability Score:** 0.993
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_5_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.228
- **Stability Score:** 0.962
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_acceleration_10_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.245
- **Stability Score:** 0.970
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_acceleration_10_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 15]
- **Performance Score:** 0.235
- **Stability Score:** 0.987
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_acceleration_5_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.228
- **Stability Score:** 0.969
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_acceleration_5_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.220
- **Stability Score:** 0.933
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_comprehensive_14
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.235
- **Stability Score:** 0.959
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_comprehensive_21
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.230
- **Stability Score:** 1.000
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_comprehensive_30
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.241
- **Stability Score:** 0.983
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_momentum_comprehensive_9
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.241
- **Stability Score:** 0.986
- **Optimization Method:** per_feature_mi_curve

#### volume_momentum_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.235
- **Stability Score:** 0.975
- **Optimization Method:** per_feature_mi_curve

#### volume_momentum_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 10]
- **Performance Score:** 0.234
- **Stability Score:** 0.965
- **Optimization Method:** per_feature_mi_curve

#### volume_momentum_5
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.238
- **Stability Score:** 0.963
- **Optimization Method:** per_feature_mi_curve

### Volume Features

Optimized 49 features with optimal + alternative lookback periods:

#### analyst_volume_pressure
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [3, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 3, 8]
- **Performance Score:** 0.166
- **Stability Score:** 0.935
- **Optimization Method:** per_feature_mi_curve

#### analyst_volume_trend
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 8, 15]
- **Performance Score:** 0.249
- **Stability Score:** 0.985
- **Optimization Method:** per_feature_mi_curve

#### price_volume_oscillator_10_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.271
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### price_volume_oscillator_5_15
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 10]
- **Performance Score:** 0.242
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_enhanced_obv_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.263
- **Stability Score:** 0.937
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_enhanced_obv_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 10]
- **Performance Score:** 0.264
- **Stability Score:** 0.940
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_enhanced_obv_50
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.275
- **Stability Score:** 0.979
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_smoothed_obv_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.256
- **Stability Score:** 0.929
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volume_acceleration_5_volume_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.239
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volume_weighted_ad_line_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 15]
- **Performance Score:** 0.256
- **Stability Score:** 0.947
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volume_weighted_ad_line_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.252
- **Stability Score:** 0.890
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volume_weighted_ad_line_50
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.268
- **Stability Score:** 0.929
- **Optimization Method:** per_feature_mi_curve

#### volume_accumulation_distribution
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.229
- **Stability Score:** 0.994
- **Optimization Method:** per_feature_mi_curve

#### volume_ema_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 10]
- **Performance Score:** 0.257
- **Stability Score:** 0.945
- **Optimization Method:** per_feature_mi_curve

#### volume_ema_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.257
- **Stability Score:** 0.938
- **Optimization Method:** per_feature_mi_curve

#### volume_ema_5
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 15]
- **Performance Score:** 0.251
- **Stability Score:** 0.979
- **Optimization Method:** per_feature_mi_curve

#### volume_ema_50
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.265
- **Stability Score:** 0.989
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_10_volume_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 10, 20]
- **Performance Score:** 0.069
- **Stability Score:** 0.947
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_20_volume_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.064
- **Stability Score:** 0.970
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_5_volume_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [3, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 3, 20]
- **Performance Score:** 0.089
- **Stability Score:** 0.979
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_ma_10_10_volume_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [15, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 15, 20]
- **Performance Score:** 0.099
- **Stability Score:** 0.952
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_ma_10_5_volume_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.095
- **Stability Score:** 0.973
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_ma_20_10_volume_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.088
- **Stability Score:** 0.949
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_ma_20_5_volume_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 10, 20]
- **Performance Score:** 0.092
- **Stability Score:** 0.959
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_ma_5_10_volume_returns
- **Optimal Lookback:** 8 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [8, 5, 20]
- **Performance Score:** 0.123
- **Stability Score:** 0.971
- **Optimization Method:** per_feature_mi_curve

#### volume_entropy_ma_5_5_volume_returns
- **Optimal Lookback:** 5 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [5, 10, 20]
- **Performance Score:** 0.109
- **Stability Score:** 0.996
- **Optimization Method:** per_feature_mi_curve

#### volume_ma_ratios_20_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.240
- **Stability Score:** 0.974
- **Optimization Method:** per_feature_mi_curve

#### volume_oscillator_10_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 10]
- **Performance Score:** 0.255
- **Stability Score:** 0.989
- **Optimization Method:** per_feature_mi_curve

#### volume_oscillator_5_15
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.245
- **Stability Score:** 0.970
- **Optimization Method:** per_feature_mi_curve

#### volume_percentile_100
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.234
- **Stability Score:** 0.979
- **Optimization Method:** per_feature_mi_curve

#### volume_percentile_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.241
- **Stability Score:** 0.974
- **Optimization Method:** per_feature_mi_curve

#### volume_percentile_50
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.231
- **Stability Score:** 0.974
- **Optimization Method:** per_feature_mi_curve

#### volume_price_correlation_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.253
- **Stability Score:** 0.955
- **Optimization Method:** per_feature_mi_curve

#### volume_price_correlation_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.249
- **Stability Score:** 0.959
- **Optimization Method:** per_feature_mi_curve

#### volume_price_divergence_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.243
- **Stability Score:** 0.977
- **Optimization Method:** per_feature_mi_curve

#### volume_price_divergence_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.240
- **Stability Score:** 0.966
- **Optimization Method:** per_feature_mi_curve

#### volume_price_trend
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.196
- **Stability Score:** 0.900
- **Optimization Method:** per_feature_mi_curve

#### volume_ratio_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.237
- **Stability Score:** 0.962
- **Optimization Method:** per_feature_mi_curve

#### volume_ratio_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.245
- **Stability Score:** 0.983
- **Optimization Method:** per_feature_mi_curve

#### volume_ratio_50
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.237
- **Stability Score:** 0.974
- **Optimization Method:** per_feature_mi_curve

#### volume_roc_1
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.219
- **Stability Score:** 0.994
- **Optimization Method:** per_feature_mi_curve

#### volume_roc_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.244
- **Stability Score:** 0.980
- **Optimization Method:** per_feature_mi_curve

#### volume_roc_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 10]
- **Performance Score:** 0.232
- **Stability Score:** 0.980
- **Optimization Method:** per_feature_mi_curve

#### volume_roc_5
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.228
- **Stability Score:** 0.958
- **Optimization Method:** per_feature_mi_curve

#### volume_trend_strength_10_30
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 10]
- **Performance Score:** 0.272
- **Stability Score:** 0.967
- **Optimization Method:** per_feature_mi_curve

#### volume_trend_strength_20_50
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 15]
- **Performance Score:** 0.280
- **Stability Score:** 0.996
- **Optimization Method:** per_feature_mi_curve

#### volume_vwap_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.240
- **Stability Score:** 0.943
- **Optimization Method:** per_feature_mi_curve

#### volume_vwap_50
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.253
- **Stability Score:** 0.960
- **Optimization Method:** per_feature_mi_curve

#### volume_zscore_60_252
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.233
- **Stability Score:** 0.963
- **Optimization Method:** per_feature_mi_curve

### Volatility Features

Optimized 42 features with optimal + alternative lookback periods:

#### band_limited_volatility
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.259
- **Stability Score:** 0.940
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 10]
- **Performance Score:** 0.254
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_100
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.272
- **Stability Score:** 0.979
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_14
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.236
- **Stability Score:** 0.979
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.254
- **Stability Score:** 0.975
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_30
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.265
- **Stability Score:** 0.989
- **Optimization Method:** per_feature_mi_curve

#### enhanced_volatility_50
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 15]
- **Performance Score:** 0.273
- **Stability Score:** 0.922
- **Optimization Method:** per_feature_mi_curve

#### returns_volatility_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 15]
- **Performance Score:** 0.240
- **Stability Score:** 0.963
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_volatility_10_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.252
- **Stability Score:** 0.991
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_volatility_10_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 15]
- **Performance Score:** 0.229
- **Stability Score:** 0.937
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_volatility_5_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.271
- **Stability Score:** 1.000
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_volatility_5_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.237
- **Stability Score:** 0.947
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.225
- **Stability Score:** 0.939
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_14
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.225
- **Stability Score:** 0.951
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.222
- **Stability Score:** 0.915
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_30
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.232
- **Stability Score:** 0.934
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_atr_50
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.236
- **Stability Score:** 0.994
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.260
- **Stability Score:** 0.981
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_14
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 8, 20]
- **Performance Score:** 0.235
- **Stability Score:** 0.996
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 10, 20]
- **Performance Score:** 0.238
- **Stability Score:** 0.938
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_30
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.251
- **Stability Score:** 0.963
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_garman_klass_volatility_50
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.264
- **Stability Score:** 0.975
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parkinson_volatility_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.260
- **Stability Score:** 0.977
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parkinson_volatility_14
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.232
- **Stability Score:** 0.978
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parkinson_volatility_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [8, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 8, 15]
- **Performance Score:** 0.221
- **Stability Score:** 0.948
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parkinson_volatility_30
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.247
- **Stability Score:** 0.927
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_parkinson_volatility_50
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.258
- **Stability Score:** 0.982
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.268
- **Stability Score:** 0.990
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_14
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [8, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 8, 20]
- **Performance Score:** 0.241
- **Stability Score:** 0.984
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.249
- **Stability Score:** 0.967
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_30
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.255
- **Stability Score:** 0.962
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_rogers_satchell_volatility_50
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.276
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_acceleration_5_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.244
- **Stability Score:** 0.997
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.251
- **Stability Score:** 0.963
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_14
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.268
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.256
- **Stability Score:** 0.987
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_30
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.239
- **Stability Score:** 0.981
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_volatility_comprehensive_50
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 10] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 10]
- **Performance Score:** 0.258
- **Stability Score:** 0.957
- **Optimization Method:** per_feature_mi_curve

#### volume_std_10
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 15]
- **Performance Score:** 0.241
- **Stability Score:** 0.988
- **Optimization Method:** per_feature_mi_curve

#### volume_std_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.244
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_std_50
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.245
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### volume_volatility_elasticity_20
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 15]
- **Performance Score:** 0.246
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

### Trend Features

Optimized 24 features with optimal + alternative lookback periods:

#### dema_21_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.242
- **Stability Score:** 0.997
- **Optimization Method:** per_feature_mi_curve

#### ema_12_returns_vwap
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.252
- **Stability Score:** 0.993
- **Optimization Method:** per_feature_mi_curve

#### ema_26_returns_vwap
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.251
- **Stability Score:** 0.990
- **Optimization Method:** per_feature_mi_curve

#### ema_50_returns_vwap
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.261
- **Stability Score:** 0.957
- **Optimization Method:** per_feature_mi_curve

#### sma_100_returns_vwap
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.262
- **Stability Score:** 0.961
- **Optimization Method:** per_feature_mi_curve

#### sma_10_returns_vwap
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.245
- **Stability Score:** 0.968
- **Optimization Method:** per_feature_mi_curve

#### sma_20_returns_vwap
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.260
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### sma_50_returns_vwap
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.260
- **Stability Score:** 0.997
- **Optimization Method:** per_feature_mi_curve

#### sma_5_returns_vwap
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.255
- **Stability Score:** 0.998
- **Optimization Method:** per_feature_mi_curve

#### tema_21_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.253
- **Stability Score:** 0.968
- **Optimization Method:** per_feature_mi_curve

#### trend_score_14
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.242
- **Stability Score:** 0.962
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_trend_strength_10_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.240
- **Stability Score:** 0.955
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_trend_strength_10_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.223
- **Stability Score:** 0.992
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_trend_strength_5_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.261
- **Stability Score:** 0.964
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_acceleration_trend_strength_5_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.241
- **Stability Score:** 0.989
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_sma_100
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 15]
- **Performance Score:** 0.268
- **Stability Score:** 0.913
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_consistency_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 15] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 15]
- **Performance Score:** 0.238
- **Stability Score:** 0.966
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_consistency_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [10, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 10, 20]
- **Performance Score:** 0.227
- **Stability Score:** 0.991
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_consistency_50_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.294
- **Stability Score:** 0.977
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_consistency_5_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.253
- **Stability Score:** 0.999
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_strength_10_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.236
- **Stability Score:** 0.953
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_strength_20_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.244
- **Stability Score:** 0.940
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_strength_50_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 20] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 20]
- **Performance Score:** 0.241
- **Stability Score:** 0.985
- **Optimization Method:** per_feature_mi_curve

#### vectorbt_trend_strength_5_price_returns
- **Optimal Lookback:** 3 (best performance + stability)
- **Alternative Lookbacks:** [5, 8] (informative & non-redundant)
- **All Optimized Lookbacks:** [3, 5, 8]
- **Performance Score:** 0.251
- **Stability Score:** 0.981
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

