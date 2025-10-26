# Feature Generation Report

**Generated:** 2025-10-26 17:58:31
**Step:** feature_generation_feature_generation_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** light

## Summary

✅ **Successfully generated 306 features** from 1,920 rows of data.

## Feature Statistics

- **Total Features:** 306
- **Data Samples:** 1,920
- **Memory Usage:** 2.53 MB
- **Missing Values:** 0
- **Missing Value %:** 0.00%

## Comprehensive Feature Analysis

### Feature Quality Metrics

| Metric | Value |
|--------|-------|
| High Quality Features (>0.7 score) | 1 |
| Medium Quality Features (0.4-0.7) | 0 |
| Low Quality Features (<0.4) | 304 |
| Constant Features | 0 |
| Highly Correlated Pairs | 979 |
| Average Correlation | 0.169 |
| Feature Stability Score | 0.004 |

### Top 10 Performing Features

| Rank | Feature | Quality Score | Correlation | Stability | Information |
|------|---------|---------------|-------------|-----------|-------------|
| 1 | `simple_returns_1_price_returns` | 0.993 | 0.917 | 0.000 | 2.777 |
| 2 | `vectorbt_trend_strength_5_price_returns` | 0.188 | 0.315 | 0.000 | 0.040 |
| 3 | `fractal_dimension` | 0.183 | 0.006 | 0.886 | 0.014 |
| 4 | `candlestick_three_white_soldiers_pattern` | 0.149 | 0.245 | 0.000 | 0.055 |
| 5 | `candlestick_three_black_crows_pattern` | 0.144 | 0.231 | 0.000 | 0.053 |
| 6 | `volume_roc_1` | 0.107 | 0.183 | 0.000 | 0.023 |
| 7 | `ar_1_coefficients_20` | 0.102 | 0.166 | 0.000 | 0.026 |
| 8 | `vectorbt_trend_strength_10_price_returns` | 0.100 | 0.171 | 0.000 | 0.047 |
| 9 | `volume_ratio_10` | 0.092 | 0.149 | 0.000 | 0.039 |
| 10 | `volume_ma_ratios_20_10` | 0.090 | 0.150 | 0.000 | 0.024 |

### Feature Distribution Analysis

| Statistic | Value |
|-----------|-------|
| Mean Quality Score | 0.026 |
| Median Quality Score | 0.015 |
| Std Quality Score | 0.061 |
| Min Quality Score | 0.001 |
| Max Quality Score | 0.993 |

### Feature Redundancy Analysis

| Metric | Value |
|--------|-------|
| Redundant Feature Pairs | 979 |
| Redundancy Rate | 2.1% |
| Unique Features | -673 |
| Redundancy Score | 0.979 |

### Feature Stability Analysis

| Metric | Value |
|--------|-------|
| Stable Features (>0.8) | 1 |
| Moderately Stable (0.5-0.8) | 0 |
| Unstable Features (<0.5) | 304 |
| Average Stability | 0.004 |

### Feature Information Content

| Metric | Value |
|--------|-------|
| High Information (>0.7) | 1 |
| Medium Information (0.4-0.7) | 0 |
| Low Information (<0.4) | 304 |
| Average Information | 0.025 |

### Feature Recommendations

#### Features to Keep (High Quality)
- `simple_returns_1_price_returns`

#### Features to Consider Removing (Low Quality)
- `vectorbt_trend_strength_5_price_returns`
- `fractal_dimension`
- `candlestick_three_white_soldiers_pattern`
- `candlestick_three_black_crows_pattern`
- `volume_roc_1`
- `ar_1_coefficients_20`
- `vectorbt_trend_strength_10_price_returns`
- `volume_ratio_10`
- `volume_ma_ratios_20_10`
- `volume_ratio_50`
- `volume_zscore_60_252`
- `volume_ratio_20`
- `volume_percentile_100`
- `volume_percentile_50`
- `volume_percentile_20`
- `vectorbt_trend_strength_20_price_returns`
- `candlestick_dark_cloud_cover_pattern`
- `candlestick_piercing_line_pattern`
- `pfe_12_returns_vwap`
- `acceleration_features`
- `sharpe_ratio_20_0.0_price_returns`
- `volume_roc_5`
- `vectorbt_volume_acceleration_5_volume_returns`
- `candlestick_harami_cross_pattern`
- `volume_roc_10`
- `volume_roc_20`
- `volume_momentum_5`
- `volume_momentum_10`
- `candlestick_doji_pattern`
- `volume_momentum_20`
- `roc_14_price_returns`
- `adx_14_returns_vwap`
- `natr_14_returns_vwap`
- `fibonacci_0.786_5_price_returns`
- `roc_30_price_returns`
- `enhanced_volatility_10`
- `hurst_exponent`
- `volume_ema_10`
- `volume_sma_10`
- `enhanced_volatility_14`
- `rolling_returns_10_price_returns`
- `enhanced_volatility_30`
- `cumulative_returns_10_price_returns`
- `simple_returns_5_price_returns`
- `enhanced_volatility_20`
- `rolling_zscore_returns_20`
- `roc_21_price_returns`
- `volume_sma_5`
- `volume_ema_5`
- `vectorbt_acceleration_momentum_10_20_price_returns`
- `vectorbt_momentum_10_price_returns`
- `returns_skewness_20_price_returns`
- `lempel_ziv_complexity_20`
- `simple_returns_10_price_returns`
- `volume_std_10`
- `vectorbt_momentum_20_price_returns`
- `stochastic_14_3_price_returns`
- `entropy_rate_20`
- `williams_r_14_price_returns`
- `cumulative_returns_20_price_returns`
- `vectorbt_momentum_5_price_returns`
- `rolling_returns_20_price_returns`
- `vectorbt_momentum_50_price_returns`
- `volume_std_20`
- `vectorbt_parkinson_volatility_14`
- `vectorbt_garman_klass_volatility_14`
- `vectorbt_acceleration_trend_strength_10_20_price_returns`
- `fibonacci_0.618_5_price_returns`
- `volume_trend_strength_10_30`
- `vectorbt_yang_zhang_volatility_14`
- `fibonacci_0.236_5_price_returns`
- `vectorbt_atr_14`
- `volume_entropy_10_volume_returns`
- `volume_price_divergence_20`
- `volume_trend_strength_20_50`
- `vectorbt_atr_10`
- `ljung_box_pvalue_20_10`
- `cycle_length`
- `volume_sma_20`
- `volume_ema_20`
- `vectorbt_volatility_comprehensive_14`
- `vectorbt_yang_zhang_volatility_10`
- `analyst_volume_trend`
- `williams_r_30_price_returns`
- `vectorbt_parkinson_volatility_10`
- `stochastic_30_3_price_returns`
- `max_drawdown`
- `stochastic_21_3_price_returns`
- `vectorbt_acceleration_10_price_returns`
- `vectorbt_garman_klass_volatility_10`
- `vectorbt_rogers_satchell_volatility_14`
- `vectorbt_parkinson_volatility_30`
- `williams_r_21_price_returns`
- `vectorbt_momentum_acceleration_5_20_price_returns`
- `vectorbt_acceleration_divergence_20_price_returns`
- `vectorbt_volatility_comprehensive_30`
- `vectorbt_yang_zhang_volatility_30`
- `volume_oscillator_5_15`
- `vectorbt_trend_consistency_10_price_returns`
- `vectorbt_rogers_satchell_volatility_10`
- `vectorbt_volatility_comprehensive_10`
- `cci_20_returns_vwap`
- `momentum_30_price_returns`
- `vectorbt_atr_20`
- `log_returns_10_price_returns`
- `resistance_level_3_10_price_returns`
- `resistance_level_2_10_price_returns`
- `resistance_level_1_10_price_returns`
- `resistance_level_4_10_price_returns`
- `resistance_level_5_10_price_returns`
- `vectorbt_garman_klass_volatility_30`
- `volume_price_divergence_10`
- `vectorbt_rogers_satchell_volatility_30`
- `volume_oscillator_10_20`
- `rsi_21_returns_vwap`
- `kst_10_15_20_30_10_10_10_15_returns_vwap`
- `band_limited_volatility`
- `resistance_level_1_20_price_returns`
- `resistance_level_2_20_price_returns`
- `resistance_level_3_20_price_returns`
- `resistance_level_5_20_price_returns`
- `resistance_level_4_20_price_returns`
- `vectorbt_atr_30`
- `ultimate_oscillator_7_14_28_returns_vwap`
- `shannon_entropy_20_10`
- `vectorbt_trend_consistency_20_price_returns`
- `volume_entropy_20_volume_returns`
- `vectorbt_yang_zhang_volatility_20`
- `log_returns_5_price_returns`
- `vectorbt_volatility_comprehensive_20`
- `dfa_slopes`
- `vectorbt_parkinson_volatility_20`
- `vectorbt_trend_strength_50_price_returns`
- `ema_50_returns_vwap`
- `vectorbt_acceleration_momentum_10_10_price_returns`
- `vectorbt_volume_weighted_ad_line_10`
- `volume_entropy_ma_10_5_volume_returns`
- `vectorbt_acceleration_momentum_5_20_price_returns`
- `vectorbt_rogers_satchell_volatility_20`
- `vectorbt_multi_timeframe_acceleration_5_20_price_returns`
- `cvar`
- `rolling_skewness_kurtosis`
- `trend_persistence`
- `vectorbt_acceleration_momentum_5_10_price_returns`
- `vectorbt_garman_klass_volatility_20`
- `volume_price_correlation_20`
- `vectorbt_momentum_acceleration_10_20_price_returns`
- `volume_vwap_10`
- `volume_volatility_elasticity_20`
- `vectorbt_enhanced_ad_line_20`
- `pivot_point_5_price_returns`
- `spectral_entropy_20`
- `volume_entropy_ma_20_10_volume_returns`
- `candlestick_hammer_pattern`
- `tema_21_price_returns`
- `volume_accumulation_distribution`
- `vectorbt_enhanced_ad_line_10`
- `candlestick_hanging_man_pattern`
- `cmo_14_returns_vwap`
- `wavelet_energy`
- `rsi_14_returns_vwap`
- `volume_vwap_20`
- `vectorbt_volatility_acceleration_5_20_price_returns`
- `vectorbt_acceleration_trend_strength_5_20_price_returns`
- `vectorbt_atr_50`
- `vectorbt_enhanced_ad_line_50`
- `momentum_21_price_returns`
- `vectorbt_momentum_acceleration_5_10_price_returns`
- `support_level_2_5_price_returns`
- `vwma_20_price_returns`
- `support_level_3_5_price_returns`
- `macd_entropy_20_12_26`
- `support_level_4_5_price_returns`
- `volume_entropy_ma_10_10_volume_returns`
- `vectorbt_trend_consistency_5_price_returns`
- `support_level_1_5_price_returns`
- `support_level_5_5_price_returns`
- `kama_30_2_30_returns_vwap`
- `sma_100_returns_vwap`
- `ema_26_returns_vwap`
- `vectorbt_trend_consistency_50_price_returns`
- `vectorbt_jerk_10_price_returns`
- `keltner_channels_20_14_price_returns`
- `vectorbt_volume_weighted_ad_line_20`
- `returns_kurtosis_20_price_returns`
- `fibonacci_0.5_20_price_returns`
- `fibonacci_0.236_20_price_returns`
- `mama_21_0.05_price_returns`
- `returns_volatility_20_price_returns`
- `vectorbt_momentum_acceleration_10_10_price_returns`
- `volume_vwap_50`
- `stochastic_kd_14_3`
- `momentum_14_price_returns`
- `support_level_2_10_price_returns`
- `support_level_1_10_price_returns`
- `support_level_3_10_price_returns`
- `support_level_4_10_price_returns`
- `support_level_5_10_price_returns`
- `donchian_channel_20`
- `rsi_30_returns_vwap`
- `analyst_momentum_alignment`
- `vectorbt_bbands_10_2.5`
- `vectorbt_bbands_10_2.0`
- `volume_entropy_ma_5_10_volume_returns`
- `vectorbt_acceleration_volatility_10_20_price_returns`
- `fibonacci_0.5_5_price_returns`
- `resistance_level_1_5_price_returns`
- `resistance_level_2_5_price_returns`
- `resistance_level_5_5_price_returns`
- `resistance_level_3_5_price_returns`
- `resistance_level_4_5_price_returns`
- `vectorbt_bbands_10_1.5`
- `fibonacci_0.382_5_price_returns`
- `volume_entropy_ma_20_5_volume_returns`
- `price_volume_oscillator_10_20`
- `enhanced_volatility_100`
- `aroon_25_returns_vwap`
- `volume_sma_50`
- `volume_ema_50`
- `analyst_volume_pressure`
- `apo_12_26_returns_vwap`
- `fibonacci_0.382_20_price_returns`
- `vectorbt_rogers_satchell_volatility_50`
- `advanced_momentum_5_20`
- `sma_50_returns_vwap`
- `fibonacci_0.236_10_price_returns`
- `sma_20_returns_vwap`
- `advanced_momentum_10_30`
- `vectorbt_enhanced_obv_20`
- `vectorbt_volatility_comprehensive_50`
- `macd_delta_12_26_9`
- `rsi_zscore_14_20`
- `cmf_20`
- `volume_std_50`
- `advanced_cumulative_returns_10`
- `vectorbt_acceleration_volatility_10_10_price_returns`
- `t3_14_0.7_returns_vwap`
- `sma_5_returns_vwap`
- `vectorbt_acceleration_5_price_returns`
- `vectorbt_momentum_comprehensive_14`
- `vectorbt_momentum_comprehensive_9`
- `sma_10_returns_vwap`
- `pivot_point_10_price_returns`
- `vectorbt_parkinson_volatility_50`
- `vectorbt_yang_zhang_volatility_50`
- `ema_12_returns_vwap`
- `macd_12_26_9_returns_vwap`
- `dema_21_price_returns`
- `enhanced_volatility_50`
- `price_volume_oscillator_5_15`
- `support_level_4_20_price_returns`
- `vectorbt_acceleration_volatility_5_20_price_returns`
- `support_level_5_20_price_returns`
- `vectorbt_enhanced_obv_50`
- `fibonacci_0.382_10_price_returns`
- `vectorbt_acceleration_trend_strength_10_10_price_returns`
- `vectorbt_garman_klass_volatility_50`
- `vectorbt_smoothed_obv_10`
- `momentum_endpoints_sma_20`
- `wma_20_price_returns`
- `vwap_deviations_20`
- `fibonacci_0.618_20_price_returns`
- `vectorbt_acceleration_trend_strength_5_10_price_returns`
- `volume_price_correlation_10`
- `volume_price_trend`
- `support_level_1_20_price_returns`
- `volume_entropy_5_volume_returns`
- `directional_signal`
- `vectorbt_acceleration_consistency_5_10_price_returns`
- `trend_score_14`
- `vectorbt_enhanced_obv_10`
- `vectorbt_acceleration_regime_5_10_price_returns`
- `fibonacci_0.5_10_price_returns`
- `support_level_3_20_price_returns`
- `support_level_2_20_price_returns`
- `vectorbt_acceleration_volatility_5_10_price_returns`
- `order_flow_imbalance_20`
- `fibonacci_0.786_10_price_returns`
- `vectorbt_bbands_20_2.5`
- `vectorbt_bbands_20_1.5`
- `vectorbt_bbands_20_2.0`
- `momentum_features`
- `vectorbt_momentum_comprehensive_30`
- `analyst_momentum_15m`
- `analyst_momentum_5m`
- `analyst_momentum_1h`
- `pivot_point_20_price_returns`
- `vectorbt_smoothed_obv_20`
- `advanced_cumulative_returns_20`
- `vectorbt_acceleration_consistency_10_10_price_returns`
- `fibonacci_0.618_10_price_returns`
- `vectorbt_acceleration_consistency_5_20_price_returns`
- `fibonacci_0.786_20_price_returns`
- `vectorbt_acceleration_regime_5_20_price_returns`
- `vectorbt_smoothed_obv_50`
- `vectorbt_volume_weighted_ad_line_50`
- `vectorbt_jerk_5_price_returns`
- `vectorbt_acceleration_consistency_10_20_price_returns`
- `vectorbt_momentum_comprehensive_21`
- `volume_entropy_ma_5_5_volume_returns`
- `vectorbt_bbands_14_2.0`
- `vectorbt_bbands_14_1.5`
- `vectorbt_bbands_14_2.5`
- `rsi_entropy_20_14`

#### Features to Investigate (Medium Quality)

## Feature Categories

### Returns (157 features)

- `log_returns_1_price_returns`
- `simple_returns_1_price_returns`
- `log_returns_5_price_returns`
- `log_returns_10_price_returns`
- `cumulative_returns_20_price_returns`
- ... and 152 more

### Momentum (47 features)

- `momentum_features`
- `vectorbt_momentum_comprehensive_21`
- `vectorbt_momentum_comprehensive_14`
- `vectorbt_momentum_comprehensive_9`
- `rsi_14_returns_vwap`
- ... and 42 more

### Volume (95 features)

- `returns_volatility_20_price_returns`
- `volume_sma_5`
- `volume_ema_5`
- `volume_sma_10`
- `volume_ema_10`
- ... and 90 more

### Volatility (57 features)

- `returns_volatility_20_price_returns`
- `volume_std_10`
- `volume_std_50`
- `volume_std_20`
- `volume_volatility_elasticity_20`
- ... and 52 more

### Trend (60 features)

- `macd_12_26_9_returns_vwap`
- `momentum_endpoints_sma_20`
- `macd_delta_12_26_9`
- `volume_sma_5`
- `volume_ema_5`
- ... and 55 more

### Oscillator (9 features)

- `macd_12_26_9_returns_vwap`
- `macd_delta_12_26_9`
- `volume_oscillator_10_20`
- `volume_oscillator_5_15`
- `price_volume_oscillator_10_20`
- ... and 4 more

### Support_resistance (33 features)

- `support_level_1_5_price_returns`
- `support_level_4_5_price_returns`
- `resistance_level_1_5_price_returns`
- `support_level_3_5_price_returns`
- `support_level_2_5_price_returns`
- ... and 28 more

### Candlestick (8 features)

- `candlestick_doji_pattern`
- `candlestick_hammer_pattern`
- `candlestick_hanging_man_pattern`
- `candlestick_harami_cross_pattern`
- `candlestick_dark_cloud_cover_pattern`
- ... and 3 more

### Entropy (14 features)

- `rsi_entropy_20_14`
- `volume_entropy_5_volume_returns`
- `macd_entropy_20_12_26`
- `volume_entropy_ma_5_5_volume_returns`
- `volume_entropy_10_volume_returns`
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
| Total Columns | 306 |
| Total Rows | 1,920 |
| Non-Null Values | 587,520 |
| Null Values | 0 |
| Memory Usage (MB) | 2.53 |

## Artifacts

### generated_features

**Path:** `artifacts/pre_training/long/Analyst/feature_generation_feature_generation_step/feature_generation_feature_generation_step_generated_features_15m_long_Analyst_20251026_175651.parquet`
**Size:** 2897.20 KB

## Next Steps

- Features are ready for feature selection and interaction generation
- Consider running lookback optimization for optimal feature parameters
- Proceed to labeling step for profit-target generation

