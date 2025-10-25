# Feature Generation Report

**Generated:** 2025-10-25 17:03:29
**Step:** feature_generation_feature_generation_step

## Configuration

- **Symbol:** ETHUSDT
- **Exchange:** binance
- **Timeframe:** 15m
- **Execution Mode:** light

## Summary

✅ **Successfully generated 334 features** from 1,920 rows of data.

## Feature Statistics

- **Total Features:** 334
- **Data Samples:** 1,920
- **Memory Usage:** 2.46 MB
- **Missing Values:** 0
- **Missing Value %:** 0.00%

## Comprehensive Feature Analysis

### Feature Quality Metrics

| Metric | Value |
|--------|-------|
| High Quality Features (>0.7 score) | 333 |
| Medium Quality Features (0.4-0.7) | 0 |
| Low Quality Features (<0.4) | 0 |
| Constant Features | 0 |
| Highly Correlated Pairs | 55611 |
| Average Correlation | 1.000 |
| Feature Stability Score | 0.906 |

### Top 10 Performing Features

| Rank | Feature | Quality Score | Correlation | Stability | Information |
|------|---------|---------------|-------------|-----------|-------------|
| 1 | `price_entropy_ma_5_10_price_returns` | 2.039 | 1.000 | 0.906 | 6.291 |
| 2 | `rsi_entropy_20_14` | 2.039 | 1.000 | 0.906 | 6.290 |
| 3 | `volume_price_divergence_10` | 2.039 | 1.000 | 0.906 | 6.290 |
| 4 | `resistance_level_4_5_price_returns` | 2.039 | 1.000 | 0.906 | 6.290 |
| 5 | `price_entropy_ma_20_5_price_returns` | 2.039 | 1.000 | 0.906 | 6.290 |
| 6 | `resistance_level_5_5_price_returns` | 2.039 | 1.000 | 0.906 | 6.289 |
| 7 | `volume_price_correlation_20` | 2.039 | 1.000 | 0.906 | 6.289 |
| 8 | `momentum_features` | 2.039 | 1.000 | 0.906 | 6.289 |
| 9 | `macd_delta_12_26_9` | 2.039 | 1.000 | 0.906 | 6.289 |
| 10 | `volume_momentum_10` | 2.039 | 1.000 | 0.906 | 6.289 |

### Feature Distribution Analysis

| Statistic | Value |
|-----------|-------|
| Mean Quality Score | 2.038 |
| Median Quality Score | 2.038 |
| Std Quality Score | 0.000 |
| Min Quality Score | 2.038 |
| Max Quality Score | 2.039 |

### Feature Redundancy Analysis

| Metric | Value |
|--------|-------|
| Redundant Feature Pairs | 55611 |
| Redundancy Rate | 100.0% |
| Unique Features | -55277 |
| Redundancy Score | 0.000 |

### Feature Stability Analysis

| Metric | Value |
|--------|-------|
| Stable Features (>0.8) | 333 |
| Moderately Stable (0.5-0.8) | 0 |
| Unstable Features (<0.5) | 0 |
| Average Stability | 0.906 |

### Feature Information Content

| Metric | Value |
|--------|-------|
| High Information (>0.7) | 333 |
| Medium Information (0.4-0.7) | 0 |
| Low Information (<0.4) | 0 |
| Average Information | 6.287 |

### Feature Recommendations

#### Features to Keep (High Quality)
- `price_entropy_ma_5_10_price_returns`
- `rsi_entropy_20_14`
- `volume_price_divergence_10`
- `resistance_level_4_5_price_returns`
- `price_entropy_ma_20_5_price_returns`
- `resistance_level_5_5_price_returns`
- `volume_price_correlation_20`
- `momentum_features`
- `macd_delta_12_26_9`
- `volume_momentum_10`
- `wavelet_energy`
- `pivot_point_5_price_returns`
- `fibonacci_0.618_20_price_returns`
- `fibonacci_0.236_10_price_returns`
- `lempel_ziv_complexity_20`
- `vectorbt_volatility_comprehensive_30`
- `ema_12_returns_vwap`
- `vectorbt_bbands_14_2.5`
- `log_returns_1_price_returns`
- `enhanced_volatility_30`
- `vectorbt_momentum_50_price_returns`
- `vectorbt_volatility_comprehensive_14`
- `analyst_momentum_15m`
- `resistance_level_3_20_price_returns`
- `vectorbt_momentum_20_price_returns`
- `vectorbt_momentum_acceleration_10_20_price_returns`
- `support_level_3_10_price_returns`
- `resistance_level_1_10_price_returns`
- `fibonacci_0.5_5_price_returns`
- `vectorbt_enhanced_obv_10`
- `volume_vwap_50`
- `vectorbt_acceleration_volatility_5_10_price_returns`
- `support_level_2_10_price_returns`
- `macd_12_26_9_returns_vwap`
- `vectorbt_trend_strength_50_price_returns`
- `volume_sma_50`
- `volume_price_trend`
- `cmf_20`
- `band_limited_volatility`
- `return_entropy_ma_5_10_price_returns`
- `pfe_12_returns_vwap`
- `vectorbt_momentum_acceleration_5_20_price_returns`
- `candlestick_dragonfly_doji_pattern`
- `cycle_length`
- `vectorbt_bbands_14_2.0`
- `volume_entropy_ma_10_10_volume_returns`
- `analyst_momentum_1h`
- `spectral_entropy_20`
- `fibonacci_0.5_10_price_returns`
- `vectorbt_acceleration_5_price_returns`
- `return_entropy_5_price_returns`
- `fibonacci_0.618_10_price_returns`
- `vectorbt_acceleration_divergence_20_price_returns`
- `support_level_3_5_price_returns`
- `vectorbt_acceleration_trend_strength_5_10_price_returns`
- `vectorbt_parkinson_volatility_50`
- `vectorbt_smoothed_obv_20`
- `vectorbt_garman_klass_volatility_14`
- `donchian_channel_20`
- `natr_14_returns_vwap`
- `vwma_20_price_returns`
- `return_entropy_ma_20_5_price_returns`
- `price_entropy_ma_5_5_price_returns`
- `enhanced_volatility_50`
- `kama_30_2_30_returns_vwap`
- `vectorbt_acceleration_regime_5_20_price_returns`
- `returns_skewness_20_price_returns`
- `rsi_zscore_14_20`
- `vectorbt_trend_consistency_20_price_returns`
- `cci_20_returns_vwap`
- `candlestick_hammer_pattern`
- `volume_entropy_ma_5_10_volume_returns`
- `vectorbt_volatility_acceleration_5_20_price_returns`
- `fibonacci_0.382_20_price_returns`
- `sma_10_returns_vwap`
- `analyst_volume_pressure`
- `vectorbt_acceleration_volatility_10_20_price_returns`
- `momentum_30_price_returns`
- `resistance_level_3_10_price_returns`
- `vectorbt_enhanced_ad_line_50`
- `cumulative_returns_20_price_returns`
- `candlestick_engulfing_pattern`
- `return_entropy_10_price_returns`
- `fibonacci_0.618_5_price_returns`
- `directional_signal`
- `fibonacci_0.786_5_price_returns`
- `vectorbt_volume_weighted_ad_line_50`
- `volume_vwap_20`
- `vectorbt_enhanced_obv_20`
- `resistance_level_5_10_price_returns`
- `vectorbt_bbands_10_2.0`
- `vectorbt_rogers_satchell_volatility_10`
- `support_level_4_20_price_returns`
- `enhanced_volatility_14`
- `vectorbt_parkinson_volatility_30`
- `order_flow_imbalance_20`
- `volume_std_50`
- `volume_roc_10`
- `dfa_slopes`
- `vectorbt_momentum_acceleration_5_10_price_returns`
- `analyst_momentum_5m`
- `vectorbt_volatility_comprehensive_20`
- `fibonacci_0.786_10_price_returns`
- `vectorbt_acceleration_trend_strength_5_20_price_returns`
- `volume_percentile_20`
- `vectorbt_momentum_5_price_returns`
- `vectorbt_acceleration_consistency_5_10_price_returns`
- `vectorbt_volume_weighted_ad_line_10`
- `vectorbt_bbands_14_1.5`
- `vectorbt_acceleration_volatility_10_10_price_returns`
- `vectorbt_enhanced_obv_50`
- `trend_persistence`
- `wma_20_price_returns`
- `apo_12_26_returns_vwap`
- `roc_14_price_returns`
- `enhanced_volatility_100`
- `vectorbt_yang_zhang_volatility_10`
- `volume_price_divergence_20`
- `vectorbt_jerk_10_price_returns`
- `volume_entropy_10_volume_returns`
- `vectorbt_garman_klass_volatility_20`
- `resistance_level_2_20_price_returns`
- `vwap_deviations_20`
- `fibonacci_0.236_5_price_returns`
- `support_level_1_20_price_returns`
- `aroon_25_returns_vwap`
- `price_entropy_10_price_returns`
- `vectorbt_parkinson_volatility_20`
- `momentum_21_price_returns`
- `volume_accumulation_distribution`
- `support_level_2_5_price_returns`
- `returns_volatility_20_price_returns`
- `vectorbt_spectral_wavelet_batch`
- `vectorbt_bbands_20_1.5`
- `vectorbt_trend_consistency_10_price_returns`
- `t3_14_0.7_returns_vwap`
- `rsi_30_returns_vwap`
- `vectorbt_garman_klass_volatility_30`
- `vectorbt_bbands_10_2.5`
- `price_volume_oscillator_10_20`
- `rolling_returns_10_price_returns`
- `advanced_cumulative_returns_10`
- `vectorbt_acceleration_regime_5_10_price_returns`
- `support_level_4_10_price_returns`
- `enhanced_volatility_10`
- `vectorbt_bbands_20_2.0`
- `price_volume_oscillator_5_15`
- `dema_21_price_returns`
- `vectorbt_acceleration_momentum_10_20_price_returns`
- `volume_momentum_20`
- `vectorbt_acceleration_10_price_returns`
- `vectorbt_acceleration_regime_10_10_price_returns`
- `vectorbt_volume_acceleration_5_volume_returns`
- `volume_entropy_ma_10_5_volume_returns`
- `vectorbt_garman_klass_volatility_50`
- `vectorbt_rogers_satchell_volatility_14`
- `vectorbt_acceleration_momentum_10_10_price_returns`
- `vectorbt_volume_weighted_ad_line_20`
- `advanced_momentum_5_20`
- `ultimate_oscillator_7_14_28_returns_vwap`
- `pivot_point_20_price_returns`
- `vectorbt_acceleration_consistency_5_20_price_returns`
- `permutation_entropy_20_3_1`
- `vectorbt_momentum_10_price_returns`
- `sma_50_returns_vwap`
- `vectorbt_smoothed_obv_50`
- `support_level_1_10_price_returns`
- `volume_percentile_50`
- `vectorbt_yang_zhang_volatility_30`
- `volume_std_10`
- `fibonacci_0.236_20_price_returns`
- `vectorbt_acceleration_regime_10_20_price_returns`
- `candlestick_harami_pattern`
- `vectorbt_acceleration_volatility_5_20_price_returns`
- `price_entropy_ma_20_10_price_returns`
- `vectorbt_atr_30`
- `candlestick_inverted_hammer_pattern`
- `support_level_4_5_price_returns`
- `support_level_5_10_price_returns`
- `fibonacci_0.382_10_price_returns`
- `resistance_level_3_5_price_returns`
- `momentum_14_price_returns`
- `support_level_5_5_price_returns`
- `volume_entropy_ma_20_10_volume_returns`
- `williams_r_14_price_returns`
- `ar_1_coefficients_20`
- `vectorbt_bbands_20_2.5`
- `resistance_level_2_10_price_returns`
- `analyst_momentum_alignment`
- `shannon_entropy_20_10`
- `vectorbt_atr_14`
- `candlestick_piercing_line_pattern`
- `resistance_level_1_5_price_returns`
- `vectorbt_yang_zhang_volatility_50`
- `return_entropy_ma_20_10_price_returns`
- `support_level_3_20_price_returns`
- `volume_ratio_20`
- `mama_21_0.05_price_returns`
- `vectorbt_acceleration_momentum_5_20_price_returns`
- `vectorbt_acceleration_correlation_20_price_returns`
- `cmo_14_returns_vwap`
- `vectorbt_parkinson_volatility_10`
- `volume_entropy_20_volume_returns`
- `candlestick_shooting_star_pattern`
- `vectorbt_trend_strength_20_price_returns`
- `volume_roc_5`
- `vectorbt_yang_zhang_volatility_20`
- `volume_entropy_ma_5_5_volume_returns`
- `kst_10_15_20_30_10_10_10_15_returns_vwap`
- `adx_14_returns_vwap`
- `log_returns_5_price_returns`
- `volume_percentile_100`
- `ema_50_returns_vwap`
- `return_entropy_ma_5_5_price_returns`
- `candlestick_abandoned_baby_pattern`
- `candlestick_long_legged_doji_pattern`
- `candlestick_doji_pattern`
- `volume_volatility_elasticity_20`
- `vectorbt_trend_consistency_50_price_returns`
- `volume_oscillator_10_20`
- `vectorbt_bbands_10_1.5`
- `simple_returns_1_price_returns`
- `resistance_level_4_20_price_returns`
- `volume_oscillator_5_15`
- `returns_kurtosis_20_price_returns`
- `volume_ratio_10`
- `resistance_level_5_20_price_returns`
- `vectorbt_acceleration_consistency_10_10_price_returns`
- `candlestick_harami_cross_pattern`
- `vectorbt_atr_10`
- `sma_20_returns_vwap`
- `keltner_channels_20_14_price_returns`
- `vectorbt_acceleration_momentum_5_10_price_returns`
- `volume_roc_20`
- `fibonacci_0.786_20_price_returns`
- `entropy_rate_20`
- `volume_entropy_5_volume_returns`
- `price_entropy_ma_10_10_price_returns`
- `fractal_dimension`
- `stochastic_21_3_price_returns`
- `vectorbt_momentum_comprehensive_21`
- `vectorbt_rogers_satchell_volatility_30`
- `williams_r_21_price_returns`
- `vectorbt_garman_klass_volatility_10`
- `vectorbt_momentum_comprehensive_30`
- `volume_ema_20`
- `vectorbt_jerk_5_price_returns`
- `fibonacci_0.5_20_price_returns`
- `support_level_2_20_price_returns`
- `sma_100_returns_vwap`
- `volume_std_20`
- `return_entropy_20_price_returns`
- `volume_sma_10`
- `tema_21_price_returns`
- `vectorbt_momentum_comprehensive_9`
- `sma_5_returns_vwap`
- `sharpe_ratio_20_0.0_price_returns`
- `price_entropy_ma_10_5_price_returns`
- `vectorbt_trend_consistency_5_price_returns`
- `vectorbt_trend_strength_10_price_returns`
- `volume_sma_5`
- `volume_entropy_ma_20_5_volume_returns`
- `volume_ema_10`
- `advanced_cumulative_returns_20`
- `vectorbt_momentum_comprehensive_14`
- `price_entropy_20_price_returns`
- `vectorbt_atr_20`
- `vectorbt_enhanced_ad_line_20`
- `vectorbt_momentum_acceleration_10_10_price_returns`
- `vectorbt_acceleration_trend_strength_10_10_price_returns`
- `macd_entropy_20_12_26`
- `resistance_level_1_20_price_returns`
- `williams_r_30_price_returns`
- `candlestick_gravestone_doji_pattern`
- `acceleration_features`
- `vectorbt_rogers_satchell_volatility_20`
- `rolling_returns_20_price_returns`
- `vectorbt_parkinson_volatility_14`
- `ema_26_returns_vwap`
- `volume_sma_20`
- `volume_ema_5`
- `stochastic_30_3_price_returns`
- `volume_ratio_50`
- `vectorbt_smoothed_obv_10`
- `rsi_21_returns_vwap`
- `volume_trend_strength_20_50`
- `trend_score_14`
- `stochastic_14_3_price_returns`
- `sample_entropy_20_2_0.2`
- `advanced_momentum_10_30`
- `price_entropy_5_price_returns`
- `volume_roc_1`
- `return_entropy_ma_10_5_price_returns`
- `volume_trend_strength_10_30`
- `volume_zscore_60_252`
- `vectorbt_volatility_comprehensive_50`
- `candlestick_hanging_man_pattern`
- `roc_21_price_returns`
- `vectorbt_trend_strength_5_price_returns`
- `fibonacci_0.382_5_price_returns`
- `simple_returns_10_price_returns`
- `cumulative_returns_10_price_returns`
- `simple_returns_5_price_returns`
- `volume_ma_ratios_20_10`
- `vectorbt_volatility_comprehensive_10`
- `vectorbt_acceleration_trend_strength_10_20_price_returns`
- `vectorbt_enhanced_ad_line_10`
- `vectorbt_atr_50`
- `log_returns_10_price_returns`
- `volume_ema_50`
- `roc_30_price_returns`
- `vectorbt_yang_zhang_volatility_14`
- `resistance_level_4_10_price_returns`
- `volume_price_correlation_10`
- `candlestick_dark_cloud_cover_pattern`
- `enhanced_volatility_20`
- `resistance_level_2_5_price_returns`
- `candlestick_three_black_crows_pattern`
- `candlestick_three_white_soldiers_pattern`
- `support_level_5_20_price_returns`
- `pivot_point_10_price_returns`
- `volume_momentum_5`
- `support_level_1_5_price_returns`
- `vectorbt_acceleration_consistency_10_20_price_returns`
- `ljung_box_pvalue_20_10`
- `stochastic_kd_14_3`
- `vectorbt_rogers_satchell_volatility_50`
- `return_entropy_ma_10_10_price_returns`
- `rolling_zscore_returns_20`
- `momentum_endpoints_sma_20`
- `volume_vwap_10`
- `analyst_volume_trend`
- `vectorbt_multi_timeframe_acceleration_5_20_price_returns`

#### Features to Consider Removing (Low Quality)

#### Features to Investigate (Medium Quality)

## Feature Categories

### Returns (178 features)

- `rsi_14_returns_vwap`
- `rsi_21_returns_vwap`
- `macd_12_26_9_returns_vwap`
- `rsi_30_returns_vwap`
- `sma_5_returns_vwap`
- ... and 173 more

### Momentum (47 features)

- `rsi_14_returns_vwap`
- `rsi_21_returns_vwap`
- `rsi_30_returns_vwap`
- `momentum_endpoints_sma_20`
- `rsi_zscore_14_20`
- ... and 42 more

### Volume (95 features)

- `volume_ema_5`
- `volume_sma_5`
- `volume_sma_10`
- `volume_sma_20`
- `volume_sma_50`
- ... and 90 more

### Volatility (57 features)

- `volume_std_50`
- `volume_std_10`
- `volume_std_20`
- `natr_14_returns_vwap`
- `returns_volatility_20_price_returns`
- ... and 52 more

### Trend (71 features)

- `macd_12_26_9_returns_vwap`
- `momentum_endpoints_sma_20`
- `macd_delta_12_26_9`
- `volume_ema_5`
- `volume_sma_5`
- ... and 66 more

### Oscillator (9 features)

- `macd_12_26_9_returns_vwap`
- `macd_delta_12_26_9`
- `macd_entropy_20_12_26`
- `volume_oscillator_10_20`
- `volume_oscillator_5_15`
- ... and 4 more

### Support_resistance (33 features)

- `support_level_1_5_price_returns`
- `support_level_2_5_price_returns`
- `support_level_4_5_price_returns`
- `support_level_5_5_price_returns`
- `support_level_3_5_price_returns`
- ... and 28 more

### Candlestick (16 features)

- `candlestick_hammer_pattern`
- `candlestick_doji_pattern`
- `candlestick_engulfing_pattern`
- `candlestick_harami_pattern`
- `candlestick_shooting_star_pattern`
- ... and 11 more

### Entropy (34 features)

- `rsi_entropy_20_14`
- `macd_entropy_20_12_26`
- `price_entropy_5_price_returns`
- `volume_entropy_5_volume_returns`
- `return_entropy_5_price_returns`
- ... and 29 more

### Acceleration (32 features)

- `acceleration_features`
- `vectorbt_acceleration_correlation_20_price_returns`
- `vectorbt_acceleration_5_price_returns`
- `vectorbt_acceleration_10_price_returns`
- `vectorbt_volume_acceleration_5_volume_returns`
- ... and 27 more

## Data Quality

| Metric | Value |
|--------|-------|
| Total Columns | 334 |
| Total Rows | 1,920 |
| Non-Null Values | 641,280 |
| Null Values | 0 |
| Memory Usage (MB) | 2.46 |

## Artifacts

### generated_features

**Path:** `artifacts/pre_training/long/Analyst/feature_generation_feature_generation_step/feature_generation_feature_generation_step_generated_features_long_Analyst_20251025_170324.parquet`
**Size:** 3584.39 KB

## Next Steps

- Features are ready for feature selection and interaction generation
- Consider running lookback optimization for optimal feature parameters
- Proceed to labeling step for profit-target generation

