# LGBM/SHAP Feature Selection Report

Generated: 2025-10-19T12:53:27.927775

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
| macd_delta_12_26_9 | 0.800000 | 626.152996 | 803298433531.365601 |
| momentum_features | 0.000000 | 0.000000 | 0.000000 |
| fibonacci_0.5_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_bbands_20_2.5 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_acceleration_regime_10_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| volume_entropy_ma_5_5_volume_returns | 0.000000 | 0.000000 | 0.000000 |
| sharpe_ratio_20_0.0_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_atr_20 | 0.000000 | 0.000000 | 0.000000 |
| shannon_entropy_20_10 | 0.000000 | 0.000000 | 0.000000 |
| pivot_point_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| candlestick_harami_cross_pattern | 0.000000 | 0.000000 | 0.000000 |
| acceleration_features | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_volume_acceleration_5_volume_returns | 0.000000 | 0.000000 | 0.000000 |
| advanced_cumulative_returns_10 | 0.000000 | 0.000000 | 0.000000 |
| ljung_box_pvalue_20_10 | 0.000000 | 0.000000 | 0.000000 |
| mama_21_0.05_price_returns | 0.000000 | 0.000000 | 0.000000 |
| volume_momentum_5 | 0.000000 | 0.000000 | 0.000000 |
| volume_vwap_50 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_momentum_comprehensive_30 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_volatility_comprehensive_30 | 0.000000 | 0.000000 | 0.000000 |
| dema_21_price_returns | 0.000000 | 0.000000 | 0.000000 |
| ctf_30m_trend_price_returns | 0.000000 | 0.000000 | 0.000000 |
| ctf_ratio_volatility_5_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_enhanced_ad_line_10 | 0.000000 | 0.000000 | 0.000000 |
| rolling_returns_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_parkinson_volatility_30 | 0.000000 | 0.000000 | 0.000000 |
| volume_price_divergence_20 | 0.000000 | 0.000000 | 0.000000 |
| volume_sma_20 | 0.000000 | 0.000000 | 0.000000 |
| ema_26_returns_vwap | 0.000000 | 0.000000 | 0.000000 |
| resistance_level_1_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| return_entropy_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| price_entropy_ma_10_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| volume_std_50 | 0.000000 | 0.000000 | 0.000000 |
| volume_entropy_5_volume_returns | 0.000000 | 0.000000 | 0.000000 |
| volume_ratio_10 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_smoothed_obv_20 | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_momentum_comprehensive_9 | 0.000000 | 0.000000 | 0.000000 |
| pivot_point_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_acceleration_consistency_10_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| fibonacci_0.382_10_price_returns | 0.000000 | 0.000000 | 0.000000 |
| analyst_volume_trend | 0.000000 | 0.000000 | 0.000000 |
| support_level_2_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| ctf_5m_momentum_price_returns | 0.000000 | 0.000000 | 0.000000 |
| cumulative_returns_20_price_returns | 0.000000 | 0.000000 | 0.000000 |
| vectorbt_momentum_comprehensive_14 | 0.000000 | 0.000000 | 0.000000 |
| volume_entropy_ma_5_10_volume_returns | 0.000000 | 0.000000 | 0.000000 |
| fibonacci_0.236_5_price_returns | 0.000000 | 0.000000 | 0.000000 |
| volume_volatility_elasticity_20 | 0.000000 | 0.000000 | 0.000000 |
| advanced_cumulative_returns_20 | 0.000000 | 0.000000 | 0.000000 |
| resistance_level_5_10_price_returns | 0.000000 | 0.000000 | 0.000000 |

## Interaction Summary

- Total interactions: 0
- Strong interactions: 0
- Avg strength: 0.000000
- Max strength: 0.000000

## Top Interactions

| f1 | f2 | strength | imp1 | imp2 |
|---|---|---:|---:|---:|
