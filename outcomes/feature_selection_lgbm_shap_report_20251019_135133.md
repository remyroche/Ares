# LGBM/SHAP Feature Selection Report

Generated: 2025-10-19T13:51:33.054708

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
| vectorbt_trend_strength_10_price_returns | 0.800000 | 626.152996 | 803298433531.365601 |
| spectral_entropy_20 | 0.000000 | 0.000000 | 0.000000 |
| price_entropy_ma_10_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| candlestick_piercing_line_pattern | 0.000000 | 0.000000 | 0.000000 |
| price_volume_oscillator_5_15 | 0.000000 | 0.000000 | 0.000000 |
| price_entropy_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_smoothed_obv_20 | 0.000000 | 0.000000 | 0.000000 |
| advanced_momentum_10_30 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_acceleration_volatility_10_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| resistance_level_4_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| volume_entropy_ma_20_5_volume_returns | 0.000000 | 0.000000 | 0.000000 |
| volume_entropy_ma_5_5_volume_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_volatility_comprehensive_14 | 0.000000 | 0.000000 | 0.000000 |
| pivot_point_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| fibonacci_0.786_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| fibonacci_0.618_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| order_flow_imbalance_20 | 0.000000 | 0.000000 | 0.000000 |
| sma_10_returns_vwap | 0.000000 | 0.000000 | 0.000000 |
| fractal_dimension | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_momentum_acceleration_10_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| volume_price_divergence_20 | 0.000000 | 0.000000 | 0.000000 |
| ctf_corr_momentum_5_15_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_acceleration_consistency_5_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| macd_entropy_20_12_26 | 0.000000 | 0.000000 | 0.000000 |
| volume_vwap_20 | 0.000000 | 0.000000 | 0.000000 |
| momentum_21_price_returns | 0.000000 | 0.000000 | 0.000000 |
| williams_r_21_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_momentum_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| wma_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| price_entropy_ma_10_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| ctf_30m_volume_volume_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_bbands_14_1.5 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_smoothed_obv_50 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_bbands_14_2.5 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_trend_consistency_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_volume_weighted_ad_line_50 | 0.000000 | 0.000000 | 0.000000 |
| return_entropy_ma_10_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_yang_zhang_volatility_14 | 0.000000 | 0.000000 | 0.000000 |
| support_level_1_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_rogers_satchell_volatility_14 | 0.000000 | 0.000000 | 0.000000 |
| log_returns_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| resistance_level_2_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_enhanced_obv_10 | 0.000000 | 0.000000 | 0.000000 |
| momentum_14_price_returns | 0.000000 | 0.000000 | 0.000000 |
| analyst_momentum_1h | 0.000000 | 0.000000 | 0.000000 |
| volume_entropy_ma_10_10_volume_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_momentum_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_spectral_wavelet_batch | 0.000000 | 0.000000 | 0.000000 |
| volume_entropy_ma_5_10_volume_returns | 0.000000 | 0.000000 | 0.000000 |
| rsi_30_returns_vwap | 0.000000 | 0.000000 | 0.000000 |

## Interaction Summary

- Total interactions: 0
- Strong interactions: 0
- Avg strength: 0.000000
- Max strength: 0.000000

## Top Interactions

| f1 | f2 | strength | imp1 | imp2 |
|---|---|---:|---:|---:|
