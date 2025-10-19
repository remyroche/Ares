# LGBM/SHAP Feature Selection Report

Generated: 2025-10-19T15:13:39.449847

## OOF Settings

- Enabled: True
- Splits: 5
- Total validation rows: 968640

## Summary

- Features considered: 60
- Top-K for SHAP: 60
- Final selection target: 60
- Selection method: combined_shap_lgb_interaction

## Detailed Statistics

### SHAP Score Distribution

- Min: 0.000000
- Max: 626.152996
- Mean: 10.435883
- Median: 0.000000
- Std: 80.159540
- Q25: 0.000000
- Q75: 0.000000
- Q90: 0.000000
- Q95: 0.000000
- Q99: 256.722728
- Non-zero count: 1
- Zero count: 59

### LGB Score Distribution

- Min: 0.000000
- Max: 803298433531.365601
- Mean: 13388307225.522760
- Median: 0.000000
- Std: 102837539116.482758
- Q25: 0.000000
- Q75: 0.000000
- Q90: 0.000000
- Q95: 0.000000
- Q99: 329352357747.857178
- Non-zero count: 1
- Zero count: 59

### COMBINED Score Distribution

- Min: 0.000000
- Max: 0.800000
- Mean: 0.013333
- Median: 0.000000
- Std: 0.102415
- Q25: 0.000000
- Q75: 0.000000
- Q90: 0.000000
- Q95: 0.000000
- Q99: 0.328000
- Non-zero count: 1
- Zero count: 59

### Score Correlations

- SHAP ↔ LGB: 1.0000
- SHAP ↔ Combined: 1.0000
- LGB ↔ Combined: 1.0000

### Threshold Analysis

- Threshold value: 0.000100
- Features above threshold: 1
- Features below threshold: 59
- Threshold percentage: 1.67%
- Max score above threshold: 0.800000
- Min score above threshold: 0.800000
- Max score below threshold: 0.000000

## Methodology

- SHAP weight: 0.6
- LGB weight: 0.2
- Interaction weight: 0.2
- Normalization: max_normalization
- Threshold method: fixed_threshold
- Fallback method: top_n_selection

## Top Features (by combined score)

| Rank | Feature | Combined | SHAP | SHAP_Norm | LGB | LGB_Norm | Above_Thresh | SHAP_W | LGB_W | INTER_W |

|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ctf_divergence_volatility_5_20_price_returns | 0.800000 | 626.152996 | 1.000 | 803298433531.365601 | 1.000 | ✓ | 0.600000 | 0.200000 | -0.000000 |
| 2 | analyst_volume_trend | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 3 | stochastic_kd_14_3 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 4 | lempel_ziv_complexity_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 5 | ctf_5m_trend_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 6 | volume_sma_10 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 7 | sma_10_returns_vwap | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 8 | vectorbt_enhanced_ad_line_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 9 | vectorbt_acceleration_momentum_5_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 10 | volume_momentum_5 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 11 | vectorbt_volume_weighted_ad_line_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 12 | rolling_returns_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 13 | volume_percentile_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 14 | ctf_5m_volatility_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 15 | volume_vwap_10 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 16 | fractal_dimension | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 17 | vectorbt_parkinson_volatility_50 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 18 | analyst_momentum_15m | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 19 | vectorbt_trend_consistency_50_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 20 | price_volume_oscillator_5_15 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 21 | vectorbt_enhanced_obv_50 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 22 | price_entropy_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 23 | volume_sma_5 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 24 | volume_ratio_50 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 25 | donchian_channel_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 26 | vectorbt_atr_10 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 27 | enhanced_volatility_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 28 | volume_roc_10 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 29 | vectorbt_volatility_comprehensive_14 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 30 | enhanced_volatility_14 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 31 | candlestick_engulfing_pattern | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 32 | vectorbt_parkinson_volatility_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 33 | vectorbt_bbands_14_2.5 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 34 | vectorbt_rogers_satchell_volatility_50 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 35 | stochastic_30_3_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 36 | vectorbt_yang_zhang_volatility_14 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 37 | volume_ema_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 38 | volume_ema_5 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 39 | vectorbt_momentum_acceleration_5_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 40 | price_entropy_ma_10_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 41 | resistance_level_1_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 42 | log_returns_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 43 | pfe_12_returns_vwap | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 44 | ctf_ratio_sma_10_50_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 45 | candlestick_dragonfly_doji_pattern | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 46 | sma_50_returns_vwap | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 47 | vectorbt_bbands_20_2.5 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 48 | ctf_ratio_volatility_5_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 49 | advanced_momentum_10_30 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 50 | volume_price_divergence_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |

## Interaction Summary

- Total interactions: 1770
- Strong interactions: 0
- Avg strength: 0.000000
- Max strength: 0.000000
- Interaction coverage: 21 features
- Strong interaction ratio: 0.0000

## Top Interactions

| f1 | f2 | strength | imp1 | imp2 |
|---|---|---:|---:|---:|
| ctf_divergence_volatility_5_20_price_returns | analyst_volume_trend | 0.000000 | 10.494964 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | stochastic_kd_14_3 | 0.000000 | 10.494964 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | lempel_ziv_complexity_20 | 0.000000 | 10.494964 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | ctf_5m_trend_price_returns | 0.000000 | 10.494964 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | volume_sma_10 | 0.000000 | 10.494964 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | sma_10_returns_vwap | 0.000000 | 10.494964 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | vectorbt_enhanced_ad_line_20 | 0.000000 | 10.494964 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | vectorbt_acceleration_momentum_5_10_price_returns | 0.000000 | 10.494964 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | volume_momentum_5 | 0.000000 | 10.494964 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | vectorbt_volume_weighted_ad_line_20 | 0.000000 | 10.494964 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | rolling_returns_10_price_returns | 0.000000 | 10.494964 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | volume_percentile_20 | 0.000000 | 10.494964 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | ctf_5m_volatility_price_returns | 0.000000 | 10.494964 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | volume_vwap_10 | 0.000000 | 10.494964 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | fractal_dimension | 0.000000 | 10.494964 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | vectorbt_parkinson_volatility_50 | 0.000000 | 10.494964 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | analyst_momentum_15m | 0.000000 | 10.494964 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | vectorbt_trend_consistency_50_price_returns | 0.000000 | 10.494964 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | price_volume_oscillator_5_15 | 0.000000 | 10.494964 | 0.000000 |
| ctf_divergence_volatility_5_20_price_returns | vectorbt_enhanced_obv_50 | 0.000000 | 10.494964 | 0.000000 |
