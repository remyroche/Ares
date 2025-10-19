# LGBM/SHAP Feature Selection Report

Generated: 2025-10-19T12:32:40.885275

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
| ctf_15m_volatility_price_returns | 0.800000 | 626.152996 | 803298433531.365601 |
| fibonacci_0.382_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| sample_entropy_20_2_0.2 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_jerk_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| fibonacci_0.5_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| ctf_30m_trend_price_returns | 0.000000 | 0.000000 | 0.000000 |
| volume_entropy_ma_20_5_volume_returns | 0.000000 | 0.000000 | 0.000000 |
| support_level_3_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| candlestick_harami_pattern | 0.000000 | 0.000000 | 0.000000 |
| volume_entropy_ma_5_5_volume_returns | 0.000000 | 0.000000 | 0.000000 |
| candlestick_inverted_hammer_pattern | 0.000000 | 0.000000 | 0.000000 |
| price_entropy_ma_10_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| pfe_12_returns_vwap | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_trend_consistency_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_trend_strength_50_price_returns | 0.000000 | 0.000000 | 0.000000 |
| return_entropy_ma_10_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| ctf_5m_volatility_price_returns | 0.000000 | 0.000000 | 0.000000 |
| return_entropy_ma_20_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_spectral_wavelet_batch | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_momentum_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_acceleration_regime_10_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_volatility_comprehensive_10 | 0.000000 | 0.000000 | 0.000000 |
| aroon_25_returns_vwap | 0.000000 | 0.000000 | 0.000000 |
| volume_volatility_elasticity_20 | 0.000000 | 0.000000 | 0.000000 |
| enhanced_volatility_50 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_acceleration_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| volume_zscore_60_252 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_acceleration_consistency_5_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| volume_oscillator_5_15 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_acceleration_consistency_5_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| candlestick_dragonfly_doji_pattern | 0.000000 | 0.000000 | 0.000000 |
| support_level_4_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| kama_30_2_30_returns_vwap | 0.000000 | 0.000000 | 0.000000 |
| volume_sma_50 | 0.000000 | 0.000000 | 0.000000 |
| enhanced_volatility_100 | 0.000000 | 0.000000 | 0.000000 |
| ctf_30m_hl_price_levels | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_yang_zhang_volatility_14 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_acceleration_consistency_10_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_volume_weighted_ad_line_50 | 0.000000 | 0.000000 | 0.000000 |
| macd_entropy_20_12_26 | 0.000000 | 0.000000 | 0.000000 |
| simple_returns_1_price_returns | 0.000000 | 0.000000 | 0.000000 |
| macd_delta_12_26_9 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_volatility_comprehensive_14 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_acceleration_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_garman_klass_volatility_20 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_volatility_comprehensive_20 | 0.000000 | 0.000000 | 0.000000 |
| ctf_15m_trend_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_rogers_satchell_volatility_50 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_parkinson_volatility_20 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_momentum_5_price_returns | 0.000000 | 0.000000 | 0.000000 |

## Interaction Summary

- Total interactions: 0
- Strong interactions: 0
- Avg strength: 0.000000
- Max strength: 0.000000

## Top Interactions

| f1 | f2 | strength | imp1 | imp2 |
|---|---|---:|---:|---:|
