# LGBM/SHAP Feature Selection Report

Generated: 2025-10-19T14:10:38.716119

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
| vectorbt_parkinson_volatility_14 | 0.800000 | 626.152996 | 803298433531.365601 |
| vectorbt_acceleration_trend_strength_10_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| price_entropy_ma_5_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| ctf_30m_trend_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_acceleration_trend_strength_5_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| price_volume_oscillator_10_20 | 0.000000 | 0.000000 | 0.000000 |
| ctf_5m_hl_price_levels | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_acceleration_correlation_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| sma_20_returns_vwap | 0.000000 | 0.000000 | 0.000000 |
| candlestick_gravestone_doji_pattern | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_momentum_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| ljung_box_pvalue_20_10 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_acceleration_consistency_5_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_trend_consistency_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| volume_percentile_20 | 0.000000 | 0.000000 | 0.000000 |
| resistance_level_1_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| ctf_30m_momentum_price_returns | 0.000000 | 0.000000 | 0.000000 |
| rsi_zscore_14_20 | 0.000000 | 0.000000 | 0.000000 |
| volume_sma_20 | 0.000000 | 0.000000 | 0.000000 |
| volume_trend_strength_10_30 | 0.000000 | 0.000000 | 0.000000 |
| keltner_channels_20_14_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_acceleration_divergence_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_trend_consistency_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| fibonacci_0.236_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| candlestick_shooting_star_pattern | 0.000000 | 0.000000 | 0.000000 |
| resistance_level_1_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| momentum_30_price_returns | 0.000000 | 0.000000 | 0.000000 |
| candlestick_hammer_pattern | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_momentum_50_price_returns | 0.000000 | 0.000000 | 0.000000 |
| support_level_5_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| resistance_level_1_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| sma_100_returns_vwap | 0.000000 | 0.000000 | 0.000000 |
| roc_30_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_rogers_satchell_volatility_10 | 0.000000 | 0.000000 | 0.000000 |
| volume_momentum_20 | 0.000000 | 0.000000 | 0.000000 |
| volume_sma_5 | 0.000000 | 0.000000 | 0.000000 |
| candlestick_doji_pattern | 0.000000 | 0.000000 | 0.000000 |
| stochastic_14_3_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_volume_weighted_ad_line_20 | 0.000000 | 0.000000 | 0.000000 |
| volume_oscillator_10_20 | 0.000000 | 0.000000 | 0.000000 |
| candlestick_engulfing_pattern | 0.000000 | 0.000000 | 0.000000 |
| fibonacci_0.786_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_momentum_comprehensive_9 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_acceleration_consistency_10_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_enhanced_ad_line_10 | 0.000000 | 0.000000 | 0.000000 |
| ctf_ratio_sma_10_50_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_bbands_10_1.5 | 0.000000 | 0.000000 | 0.000000 |
| entropy_rate_20 | 0.000000 | 0.000000 | 0.000000 |
| roc_14_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_atr_14 | 0.000000 | 0.000000 | 0.000000 |

## Interaction Summary

- Total interactions: 1770
- Strong interactions: 0
- Avg strength: 0.000000
- Max strength: 0.000000

## Top Interactions

| f1 | f2 | strength | imp1 | imp2 |
|---|---|---:|---:|---:|
| vectorbt_parkinson_volatility_14 | vectorbt_acceleration_trend_strength_10_20_price_returns | 0.000000 | 10.678896 | 0.000000 |
| vectorbt_parkinson_volatility_14 | price_entropy_ma_5_5_price_returns | 0.000000 | 10.678896 | 0.000000 |
| vectorbt_parkinson_volatility_14 | ctf_30m_trend_price_returns | 0.000000 | 10.678896 | 0.000000 |
| vectorbt_parkinson_volatility_14 | vectorbt_acceleration_trend_strength_5_20_price_returns | 0.000000 | 10.678896 | 0.000000 |
| vectorbt_parkinson_volatility_14 | price_volume_oscillator_10_20 | 0.000000 | 10.678896 | 0.000000 |
| vectorbt_parkinson_volatility_14 | ctf_5m_hl_price_levels | 0.000000 | 10.678896 | 0.000000 |
| vectorbt_parkinson_volatility_14 | vectorbt_acceleration_correlation_20_price_returns | 0.000000 | 10.678896 | 0.000000 |
| vectorbt_parkinson_volatility_14 | sma_20_returns_vwap | 0.000000 | 10.678896 | 0.000000 |
| vectorbt_parkinson_volatility_14 | candlestick_gravestone_doji_pattern | 0.000000 | 10.678896 | 0.000000 |
| vectorbt_parkinson_volatility_14 | vectorbt_momentum_20_price_returns | 0.000000 | 10.678896 | 0.000000 |
| vectorbt_parkinson_volatility_14 | ljung_box_pvalue_20_10 | 0.000000 | 10.678896 | 0.000000 |
| vectorbt_parkinson_volatility_14 | vectorbt_acceleration_consistency_5_20_price_returns | 0.000000 | 10.678896 | 0.000000 |
| vectorbt_parkinson_volatility_14 | vectorbt_trend_consistency_5_price_returns | 0.000000 | 10.678896 | 0.000000 |
| vectorbt_parkinson_volatility_14 | volume_percentile_20 | 0.000000 | 10.678896 | 0.000000 |
| vectorbt_parkinson_volatility_14 | resistance_level_1_10_price_returns | 0.000000 | 10.678896 | 0.000000 |
| vectorbt_parkinson_volatility_14 | ctf_30m_momentum_price_returns | 0.000000 | 10.678896 | 0.000000 |
| vectorbt_parkinson_volatility_14 | rsi_zscore_14_20 | 0.000000 | 10.678896 | 0.000000 |
| vectorbt_parkinson_volatility_14 | volume_sma_20 | 0.000000 | 10.678896 | 0.000000 |
| vectorbt_parkinson_volatility_14 | volume_trend_strength_10_30 | 0.000000 | 10.678896 | 0.000000 |
| vectorbt_parkinson_volatility_14 | keltner_channels_20_14_price_returns | 0.000000 | 10.678896 | 0.000000 |
