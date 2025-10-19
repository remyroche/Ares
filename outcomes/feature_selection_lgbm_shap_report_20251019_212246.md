# LGBM/SHAP Feature Selection Report

Generated: 2025-10-19T21:22:46.102717

## OOF Settings

- Enabled: True
- Splits: 5
- Total validation rows: 968640

## Summary

- Features considered: 60
- Top-K for SHAP: 60
- Final selection target: 60

## Top Features (by combined score)

| Feature | Combined | SHAP | Gain |
|---|---:|---:|---:|
| vectorbt_acceleration_correlation_20_price_returns | 0.800000 | 626.152996 | 803298433531.365601 |
| volume_percentile_50 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_jerk_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_acceleration_regime_5_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_enhanced_ad_line_20 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_acceleration_consistency_10_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_parkinson_volatility_30 | 0.000000 | 0.000000 | 0.000000 |
| entropy_rate_20 | 0.000000 | 0.000000 | 0.000000 |
| volume_momentum_20 | 0.000000 | 0.000000 | 0.000000 |
| volume_entropy_ma_20_10_volume_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_acceleration_momentum_5_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| return_entropy_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| ljung_box_pvalue_20_10 | 0.000000 | 0.000000 | 0.000000 |
| sma_5_returns_vwap | 0.000000 | 0.000000 | 0.000000 |
| fibonacci_0.618_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| volume_percentile_100 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_enhanced_ad_line_50 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_rogers_satchell_volatility_50 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_enhanced_obv_20 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_rogers_satchell_volatility_30 | 0.000000 | 0.000000 | 0.000000 |
| simple_returns_1_price_returns | 0.000000 | 0.000000 | 0.000000 |
| volume_std_50 | 0.000000 | 0.000000 | 0.000000 |
| t3_14_0.7_returns_vwap | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_bbands_20_2.0 | 0.000000 | 0.000000 | 0.000000 |
| pivot_point_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| support_level_1_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| price_entropy_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| stochastic_21_3_price_returns | 0.000000 | 0.000000 | 0.000000 |
| fibonacci_0.5_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| ctf_ratio_sma_10_50_price_returns | 0.000000 | 0.000000 | 0.000000 |
| log_returns_1_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_bbands_14_1.5 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_volume_weighted_ad_line_20 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_acceleration_consistency_10_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| kama_30_2_30_returns_vwap | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_volatility_comprehensive_50 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_spectral_wavelet_batch | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_smoothed_obv_50 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_momentum_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| price_entropy_ma_20_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| advanced_cumulative_returns_10 | 0.000000 | 0.000000 | 0.000000 |
| volume_trend_strength_10_30 | 0.000000 | 0.000000 | 0.000000 |
| ctf_divergence_momentum_5_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| ema_12_returns_vwap | 0.000000 | 0.000000 | 0.000000 |
| simple_returns_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| volume_sma_5 | 0.000000 | 0.000000 | 0.000000 |
| order_flow_imbalance_20 | 0.000000 | 0.000000 | 0.000000 |
| support_level_3_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| ar_1_coefficients_20 | 0.000000 | 0.000000 | 0.000000 |
| keltner_channels_20_14_price_returns | 0.000000 | 0.000000 | 0.000000 |

## Interaction Summary

- Total interactions: 0
- Strong interactions: 0
- Avg strength: 0.000000
- Max strength: 0.000000

## Top Interactions

| f1 | f2 | strength | imp1 | imp2 |
|---|---|---:|---:|---:|
