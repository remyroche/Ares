# Feature Generation Report

**Generated:** 2025-10-26 13:10:47
**Step:** feature_generation_feature_generation_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 1h
- **Execution Mode:** light

## Summary

✅ **Successfully generated 300 features** from 480 rows of data.

## Feature Statistics

- **Total Features:** 300
- **Data Samples:** 480
- **Memory Usage:** 0.62 MB
- **Missing Values:** 0
- **Missing Value %:** 0.00%

## Comprehensive Feature Analysis

### Feature Quality Metrics

| Metric | Value |
|--------|-------|
| High Quality Features (>0.7 score) | 1 |
| Medium Quality Features (0.4-0.7) | 0 |
| Low Quality Features (<0.4) | 298 |
| Constant Features | 0 |
| Highly Correlated Pairs | 754 |
| Average Correlation | 0.167 |
| Feature Stability Score | 0.003 |

### Top 10 Performing Features

| Rank | Feature | Quality Score | Correlation | Stability | Information |
|------|---------|---------------|-------------|-----------|-------------|
| 1 | `log_returns_1_price_returns` | 0.851 | 0.931 | 0.000 | 2.115 |
| 2 | `fractal_dimension` | 0.200 | 0.078 | 0.792 | 0.008 |
| 3 | `volume_roc_1` | 0.133 | 0.190 | 0.000 | 0.097 |
| 4 | `vectorbt_trend_strength_5_price_returns` | 0.115 | 0.254 | 0.000 | 0.081 |
| 5 | `sharpe_ratio_20_0.0_price_returns` | 0.096 | 0.181 | 0.000 | 0.038 |
| 6 | `candlestick_three_black_crows_pattern` | 0.092 | 0.224 | 0.000 | 0.048 |
| 7 | `candlestick_three_white_soldiers_pattern` | 0.088 | 0.208 | 0.000 | 0.074 |
| 8 | `vectorbt_momentum_50_price_returns` | 0.085 | 0.101 | 0.000 | 0.139 |
| 9 | `vectorbt_momentum_20_price_returns` | 0.078 | 0.133 | 0.000 | 0.169 |
| 10 | `rolling_returns_20_price_returns` | 0.076 | 0.126 | 0.000 | 0.162 |

### Feature Distribution Analysis

| Statistic | Value |
|-----------|-------|
| Mean Quality Score | 0.030 |
| Median Quality Score | 0.022 |
| Std Quality Score | 0.052 |
| Min Quality Score | 0.001 |
| Max Quality Score | 0.851 |

### Feature Redundancy Analysis

| Metric | Value |
|--------|-------|
| Redundant Feature Pairs | 754 |
| Redundancy Rate | 1.7% |
| Unique Features | -454 |
| Redundancy Score | 0.983 |

### Feature Stability Analysis

| Metric | Value |
|--------|-------|
| Stable Features (>0.8) | 0 |
| Moderately Stable (0.5-0.8) | 1 |
| Unstable Features (<0.5) | 298 |
| Average Stability | 0.003 |

### Feature Information Content

| Metric | Value |
|--------|-------|
| High Information (>0.7) | 1 |
| Medium Information (0.4-0.7) | 0 |
| Low Information (<0.4) | 298 |
| Average Information | 0.029 |

### Feature Recommendations

#### Features to Keep (High Quality)
- `log_returns_1_price_returns`

#### Features to Consider Removing (Low Quality)
- `fractal_dimension`
- `volume_roc_1`
- `vectorbt_trend_strength_5_price_returns`
- `sharpe_ratio_20_0.0_price_returns`
- `candlestick_three_black_crows_pattern`
- `candlestick_three_white_soldiers_pattern`
- `vectorbt_momentum_50_price_returns`
- `vectorbt_momentum_20_price_returns`
- `rolling_returns_20_price_returns`
- `cumulative_returns_20_price_returns`
- `momentum_21_price_returns`
- `volume_ratio_50`
- `volume_percentile_100`
- `acceleration_features`
- `volume_ratio_10`
- `volume_percentile_50`
- `returns_skewness_20_price_returns`
- `analyst_momentum_alignment`
- `volume_percentile_20`
- `volume_ratio_20`
- `rolling_zscore_returns_20`
- `vectorbt_momentum_acceleration_5_20_price_returns`
- `vectorbt_acceleration_momentum_10_10_price_returns`
- `volume_ma_ratios_20_10`
- `spectral_entropy_20`
- `stochastic_30_3_price_returns`
- `volume_volatility_elasticity_20`
- `williams_r_30_price_returns`
- `resistance_level_3_10_price_returns`
- `vectorbt_trend_strength_10_price_returns`
- `resistance_level_1_10_price_returns`
- `stochastic_14_3_price_returns`
- `williams_r_14_price_returns`
- `resistance_level_2_10_price_returns`
- `ar_1_coefficients_20`
- `resistance_level_4_10_price_returns`
- `resistance_level_5_10_price_returns`
- `vectorbt_momentum_5_price_returns`
- `ljung_box_pvalue_20_10`
- `roc_21_price_returns`
- `candlestick_harami_cross_pattern`
- `vectorbt_volatility_comprehensive_10`
- `candlestick_doji_pattern`
- `vectorbt_trend_consistency_20_price_returns`
- `vectorbt_trend_consistency_10_price_returns`
- `simple_returns_5_price_returns`
- `roc_30_price_returns`
- `aroon_25_returns_vwap`
- `rsi_entropy_20_14`
- `candlestick_dark_cloud_cover_pattern`
- `volume_momentum_5`
- `volume_zscore_60_252`
- `fibonacci_0.786_20_price_returns`
- `williams_r_21_price_returns`
- `stochastic_21_3_price_returns`
- `candlestick_long_legged_doji_pattern`
- `vectorbt_acceleration_regime_5_20_price_returns`
- `vectorbt_acceleration_divergence_20_price_returns`
- `rsi_21_returns_vwap`
- `volume_trend_strength_20_50`
- `enhanced_volatility_10`
- `vectorbt_volume_acceleration_5_volume_returns`
- `roc_14_price_returns`
- `vectorbt_atr_14`
- `simple_returns_10_price_returns`
- `cci_20_returns_vwap`
- `fibonacci_0.236_10_price_returns`
- `volume_sma_5`
- `volume_ema_5`
- `vectorbt_atr_10`
- `donchian_channel_20`
- `resistance_level_3_5_price_returns`
- `vectorbt_volatility_acceleration_5_20_price_returns`
- `cumulative_returns_10_price_returns`
- `rolling_returns_10_price_returns`
- `resistance_level_1_5_price_returns`
- `log_returns_10_price_returns`
- `vectorbt_acceleration_volatility_10_10_price_returns`
- `resistance_level_5_5_price_returns`
- `resistance_level_2_5_price_returns`
- `vectorbt_momentum_10_price_returns`
- `volume_roc_5`
- `vectorbt_volatility_comprehensive_14`
- `resistance_level_4_5_price_returns`
- `fibonacci_0.5_20_price_returns`
- `vectorbt_momentum_acceleration_10_10_price_returns`
- `vectorbt_yang_zhang_volatility_14`
- `volume_momentum_10`
- `fibonacci_0.618_20_price_returns`
- `band_limited_volatility`
- `fibonacci_0.786_5_price_returns`
- `vectorbt_volatility_comprehensive_50`
- `fibonacci_0.786_10_price_returns`
- `volume_price_divergence_10`
- `sma_50_returns_vwap`
- `volume_roc_10`
- `vectorbt_acceleration_10_price_returns`
- `vectorbt_volatility_comprehensive_20`
- `rsi_14_returns_vwap`
- `cmo_14_returns_vwap`
- `fibonacci_0.382_5_price_returns`
- `vectorbt_trend_strength_50_price_returns`
- `vectorbt_parkinson_volatility_14`
- `momentum_30_price_returns`
- `analyst_volume_pressure`
- `enhanced_volatility_30`
- `vectorbt_acceleration_momentum_10_20_price_returns`
- `fibonacci_0.618_5_price_returns`
- `vectorbt_rogers_satchell_volatility_10`
- `volume_entropy_ma_20_10_volume_returns`
- `volume_entropy_ma_20_5_volume_returns`
- `vectorbt_enhanced_obv_10`
- `vectorbt_bbands_14_2.5`
- `vectorbt_trend_consistency_50_price_returns`
- `volume_entropy_10_volume_returns`
- `vectorbt_acceleration_consistency_5_10_price_returns`
- `vectorbt_acceleration_trend_strength_5_20_price_returns`
- `vectorbt_bbands_14_2.0`
- `vectorbt_bbands_14_1.5`
- `vectorbt_garman_klass_volatility_14`
- `volume_momentum_20`
- `vectorbt_yang_zhang_volatility_30`
- `vectorbt_rogers_satchell_volatility_14`
- `sma_20_returns_vwap`
- `vectorbt_jerk_5_price_returns`
- `vectorbt_smoothed_obv_10`
- `ema_50_returns_vwap`
- `vectorbt_acceleration_momentum_5_20_price_returns`
- `vectorbt_volatility_comprehensive_30`
- `volume_roc_20`
- `volume_std_20`
- `returns_volatility_20_price_returns`
- `order_flow_imbalance_20`
- `vectorbt_garman_klass_volatility_10`
- `vectorbt_garman_klass_volatility_30`
- `vectorbt_yang_zhang_volatility_10`
- `candlestick_piercing_line_pattern`
- `vectorbt_parkinson_volatility_30`
- `ultimate_oscillator_7_14_28_returns_vwap`
- `support_level_1_10_price_returns`
- `support_level_2_10_price_returns`
- `support_level_4_10_price_returns`
- `support_level_3_10_price_returns`
- `support_level_5_10_price_returns`
- `advanced_momentum_5_20`
- `vectorbt_yang_zhang_volatility_50`
- `vectorbt_parkinson_volatility_50`
- `vectorbt_parkinson_volatility_10`
- `kst_10_15_20_30_10_10_10_15_returns_vwap`
- `vectorbt_atr_20`
- `vectorbt_trend_consistency_5_price_returns`
- `price_volume_oscillator_5_15`
- `volume_entropy_20_volume_returns`
- `vectorbt_enhanced_ad_line_50`
- `vectorbt_bbands_10_2.5`
- `fibonacci_0.618_10_price_returns`
- `vectorbt_atr_50`
- `vectorbt_bbands_10_2.0`
- `vectorbt_bbands_10_1.5`
- `vectorbt_volume_weighted_ad_line_50`
- `returns_kurtosis_20_price_returns`
- `stochastic_kd_14_3`
- `analyst_volume_trend`
- `dfa_slopes`
- `vwap_deviations_20`
- `vectorbt_parkinson_volatility_20`
- `volume_ema_10`
- `volume_sma_10`
- `volume_ema_20`
- `volume_sma_20`
- `enhanced_volatility_50`
- `vectorbt_yang_zhang_volatility_20`
- `pivot_point_5_price_returns`
- `lempel_ziv_complexity_20`
- `volume_std_10`
- `enhanced_volatility_14`
- `cycle_length`
- `vectorbt_acceleration_volatility_5_10_price_returns`
- `volume_entropy_ma_10_10_volume_returns`
- `log_returns_5_price_returns`
- `vectorbt_enhanced_obv_20`
- `vectorbt_momentum_acceleration_10_20_price_returns`
- `vectorbt_smoothed_obv_50`
- `volume_std_50`
- `vectorbt_acceleration_volatility_5_20_price_returns`
- `directional_signal`
- `vectorbt_acceleration_trend_strength_10_20_price_returns`
- `enhanced_volatility_20`
- `macd_12_26_9_returns_vwap`
- `support_level_2_5_price_returns`
- `enhanced_volatility_100`
- `vectorbt_atr_30`
- `support_level_3_5_price_returns`
- `vectorbt_momentum_comprehensive_9`
- `volume_price_correlation_10`
- `support_level_5_5_price_returns`
- `vectorbt_acceleration_trend_strength_10_10_price_returns`
- `support_level_4_5_price_returns`
- `macd_entropy_20_12_26`
- `vectorbt_rogers_satchell_volatility_30`
- `vectorbt_acceleration_regime_5_10_price_returns`
- `fibonacci_0.5_5_price_returns`
- `volume_price_trend`
- `support_level_1_5_price_returns`
- `resistance_level_1_20_price_returns`
- `resistance_level_2_20_price_returns`
- `resistance_level_5_20_price_returns`
- `resistance_level_3_20_price_returns`
- `resistance_level_4_20_price_returns`
- `vectorbt_acceleration_consistency_10_10_price_returns`
- `vectorbt_trend_strength_20_price_returns`
- `vectorbt_jerk_10_price_returns`
- `momentum_14_price_returns`
- `advanced_cumulative_returns_10`
- `rsi_30_returns_vwap`
- `vectorbt_volume_weighted_ad_line_10`
- `volume_entropy_ma_10_5_volume_returns`
- `volume_price_divergence_20`
- `vectorbt_enhanced_ad_line_10`
- `rsi_zscore_14_20`
- `pfe_12_returns_vwap`
- `volume_accumulation_distribution`
- `fibonacci_0.382_10_price_returns`
- `vectorbt_acceleration_5_price_returns`
- `wma_20_price_returns`
- `volume_price_correlation_20`
- `volume_oscillator_5_15`
- `vectorbt_momentum_comprehensive_30`
- `macd_delta_12_26_9`
- `volume_oscillator_10_20`
- `vectorbt_garman_klass_volatility_20`
- `pivot_point_10_price_returns`
- `momentum_endpoints_sma_20`
- `vectorbt_multi_timeframe_acceleration_5_20_price_returns`
- `momentum_features`
- `volume_entropy_ma_5_10_volume_returns`
- `fibonacci_0.382_20_price_returns`
- `vectorbt_smoothed_obv_20`
- `vectorbt_volume_weighted_ad_line_20`
- `apo_12_26_returns_vwap`
- `adx_14_returns_vwap`
- `natr_14_returns_vwap`
- `vectorbt_enhanced_ad_line_20`
- `trend_score_14`
- `wavelet_energy`
- `vectorbt_enhanced_obv_50`
- `entropy_rate_20`
- `vectorbt_rogers_satchell_volatility_20`
- `vectorbt_garman_klass_volatility_50`
- `advanced_momentum_10_30`
- `volume_entropy_ma_5_5_volume_returns`
- `fibonacci_0.5_10_price_returns`
- `vwma_20_price_returns`
- `fibonacci_0.236_20_price_returns`
- `volume_vwap_50`
- `volume_sma_50`
- `volume_ema_50`
- `volume_vwap_10`
- `analyst_momentum_5m`
- `analyst_momentum_15m`
- `analyst_momentum_1h`
- `pivot_point_20_price_returns`
- `volume_entropy_5_volume_returns`
- `support_level_1_20_price_returns`
- `support_level_2_20_price_returns`
- `support_level_5_20_price_returns`
- `support_level_4_20_price_returns`
- `support_level_3_20_price_returns`
- `vectorbt_rogers_satchell_volatility_50`
- `price_volume_oscillator_10_20`
- `sma_5_returns_vwap`
- `cmf_20`
- `advanced_cumulative_returns_20`
- `t3_14_0.7_returns_vwap`
- `kama_30_2_30_returns_vwap`
- `vectorbt_acceleration_trend_strength_5_10_price_returns`
- `tema_21_price_returns`
- `vectorbt_acceleration_consistency_10_20_price_returns`
- `volume_vwap_20`
- `vectorbt_acceleration_consistency_5_20_price_returns`
- `ema_12_returns_vwap`
- `ema_26_returns_vwap`
- `sma_100_returns_vwap`
- `volume_trend_strength_10_30`
- `vectorbt_acceleration_volatility_10_20_price_returns`
- `sma_10_returns_vwap`
- `vectorbt_momentum_acceleration_5_10_price_returns`
- `mama_21_0.05_price_returns`
- `keltner_channels_20_14_price_returns`
- `dema_21_price_returns`
- `vectorbt_bbands_20_2.0`
- `vectorbt_bbands_20_2.5`
- `vectorbt_bbands_20_1.5`
- `vectorbt_momentum_comprehensive_14`
- `shannon_entropy_20_10`
- `vectorbt_momentum_comprehensive_21`
- `vectorbt_acceleration_momentum_5_10_price_returns`
- `fibonacci_0.236_5_price_returns`

#### Features to Investigate (Medium Quality)

## Feature Categories

### Returns (157 features)

- `simple_returns_1_price_returns`
- `log_returns_10_price_returns`
- `log_returns_1_price_returns`
- `log_returns_5_price_returns`
- `cumulative_returns_10_price_returns`
- ... and 152 more

### Momentum (46 features)

- `momentum_features`
- `vectorbt_momentum_comprehensive_21`
- `vectorbt_momentum_comprehensive_9`
- `vectorbt_momentum_comprehensive_14`
- `vectorbt_momentum_comprehensive_30`
- ... and 41 more

### Volume (95 features)

- `returns_volatility_20_price_returns`
- `volume_sma_5`
- `volume_ema_5`
- `volume_ema_10`
- `volume_sma_10`
- ... and 90 more

### Volatility (57 features)

- `returns_volatility_20_price_returns`
- `volume_std_10`
- `volume_std_20`
- `volume_std_50`
- `volume_volatility_elasticity_20`
- ... and 52 more

### Trend (57 features)

- `macd_12_26_9_returns_vwap`
- `momentum_endpoints_sma_20`
- `macd_delta_12_26_9`
- `volume_sma_5`
- `volume_ema_5`
- ... and 52 more

### Oscillator (9 features)

- `macd_12_26_9_returns_vwap`
- `macd_delta_12_26_9`
- `volume_oscillator_10_20`
- `volume_oscillator_5_15`
- `price_volume_oscillator_5_15`
- ... and 4 more

### Support_resistance (33 features)

- `support_level_2_5_price_returns`
- `support_level_1_5_price_returns`
- `support_level_5_5_price_returns`
- `support_level_3_5_price_returns`
- `resistance_level_1_5_price_returns`
- ... and 28 more

### Candlestick (7 features)

- `candlestick_doji_pattern`
- `candlestick_harami_cross_pattern`
- `candlestick_long_legged_doji_pattern`
- `candlestick_three_white_soldiers_pattern`
- `candlestick_three_black_crows_pattern`
- ... and 2 more

### Entropy (14 features)

- `rsi_entropy_20_14`
- `macd_entropy_20_12_26`
- `volume_entropy_5_volume_returns`
- `volume_entropy_ma_5_5_volume_returns`
- `volume_entropy_ma_5_10_volume_returns`
- ... and 9 more

### Acceleration (29 features)

- `vectorbt_acceleration_5_price_returns`
- `vectorbt_acceleration_10_price_returns`
- `vectorbt_volume_acceleration_5_volume_returns`
- `vectorbt_volatility_acceleration_5_20_price_returns`
- `vectorbt_momentum_acceleration_5_10_price_returns`
- ... and 24 more

## Data Quality

| Metric | Value |
|--------|-------|
| Total Columns | 300 |
| Total Rows | 480 |
| Non-Null Values | 144,000 |
| Null Values | 0 |
| Memory Usage (MB) | 0.62 |

## Artifacts

### generated_features

**Path:** `artifacts/pre_training/long/Analyst/feature_generation_feature_generation_step/feature_generation_feature_generation_step_generated_features_1h_long_Analyst_20251026_131017.parquet`
**Size:** 814.18 KB

## Next Steps

- Features are ready for feature selection and interaction generation
- Consider running lookback optimization for optimal feature parameters
- Proceed to labeling step for profit-target generation

