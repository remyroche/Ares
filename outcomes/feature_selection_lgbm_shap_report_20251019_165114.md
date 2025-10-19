# LGBM/SHAP Feature Selection Report

Generated: 2025-10-19T16:51:14.259862

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
- Max: 0.000967
- Mean: 0.000016
- Median: 0.000000
- Std: 0.000124
- Q25: 0.000000
- Q75: 0.000000
- Q90: 0.000000
- Q95: 0.000000
- Q99: 0.000397
- Non-zero count: 1
- Zero count: 59

### LGB Score Distribution

- Min: 0.000000
- Max: 17.111414
- Mean: 0.285190
- Median: 0.000000
- Std: 2.190588
- Q25: 0.000000
- Q75: 0.000000
- Q90: 0.000000
- Q95: 0.000000
- Q99: 7.015680
- Non-zero count: 1
- Zero count: 59

### COMBINED Score Distribution

- Min: 0.000000
- Max: 0.799994
- Mean: 0.013333
- Median: 0.000000
- Std: 0.102414
- Q25: 0.000000
- Q75: 0.000000
- Q90: 0.000000
- Q95: 0.000000
- Q99: 0.327997
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
- Max score above threshold: 0.799994
- Min score above threshold: 0.799994
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
| 1 | ctf_15m_hl_price_levels | 0.799994 | 0.000967 | 1.000 | 17.111414 | 1.000 | ✓ | 0.600000 | 0.200000 | -0.000006 |
| 2 | volume_price_correlation_10 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 3 | ar_1_coefficients_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 4 | candlestick_abandoned_baby_pattern | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 5 | ctf_divergence_volatility_5_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 6 | rolling_returns_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 7 | support_level_4_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 8 | ctf_ratio_momentum_5_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 9 | vectorbt_enhanced_obv_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 10 | return_entropy_ma_10_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 11 | support_level_1_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 12 | fibonacci_0.5_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 13 | dema_21_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 14 | volume_momentum_10 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 15 | vectorbt_volatility_comprehensive_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 16 | tema_21_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 17 | vectorbt_bbands_14_2.5 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 18 | volume_percentile_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 19 | ctf_corr_momentum_5_15_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 20 | keltner_channels_20_14_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 21 | stochastic_14_3_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 22 | vectorbt_volatility_acceleration_5_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 23 | candlestick_piercing_line_pattern | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 24 | candlestick_three_white_soldiers_pattern | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 25 | support_level_1_20_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 26 | sharpe_ratio_20_0.0_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 27 | vectorbt_atr_30 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 28 | momentum_30_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 29 | resistance_level_5_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 30 | vectorbt_bbands_20_1.5 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 31 | dfa_slopes | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 32 | vectorbt_jerk_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 33 | enhanced_volatility_50 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 34 | log_returns_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 35 | ctf_30m_trend_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 36 | volume_momentum_20 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 37 | candlestick_harami_pattern | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 38 | cycle_length | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 39 | resistance_level_5_5_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 40 | vectorbt_bbands_14_1.5 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 41 | return_entropy_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 42 | t3_14_0.7_returns_vwap | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 43 | volume_std_50 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 44 | vectorbt_volume_acceleration_5_volume_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 45 | momentum_21_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 46 | volume_ratio_10 | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 47 | vectorbt_acceleration_regime_5_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 48 | log_returns_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 49 | simple_returns_10_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |
| 50 | roc_14_price_returns | 0.000000 | 0.000000 | 0.000 | 0.000000 | 0.000 | ✗ | 0.000000 | 0.000000 | 0.000000 |

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
| ctf_15m_hl_price_levels | volume_price_correlation_10 | 0.000000 | 0.000002 | 0.000000 |
| ctf_15m_hl_price_levels | ar_1_coefficients_20 | 0.000000 | 0.000002 | 0.000000 |
| ctf_15m_hl_price_levels | candlestick_abandoned_baby_pattern | 0.000000 | 0.000002 | 0.000000 |
| ctf_15m_hl_price_levels | ctf_divergence_volatility_5_20_price_returns | 0.000000 | 0.000002 | 0.000000 |
| ctf_15m_hl_price_levels | rolling_returns_20_price_returns | 0.000000 | 0.000002 | 0.000000 |
| ctf_15m_hl_price_levels | support_level_4_10_price_returns | 0.000000 | 0.000002 | 0.000000 |
| ctf_15m_hl_price_levels | ctf_ratio_momentum_5_20_price_returns | 0.000000 | 0.000002 | 0.000000 |
| ctf_15m_hl_price_levels | vectorbt_enhanced_obv_20 | 0.000000 | 0.000002 | 0.000000 |
| ctf_15m_hl_price_levels | return_entropy_ma_10_10_price_returns | 0.000000 | 0.000002 | 0.000000 |
| ctf_15m_hl_price_levels | support_level_1_5_price_returns | 0.000000 | 0.000002 | 0.000000 |
| ctf_15m_hl_price_levels | fibonacci_0.5_5_price_returns | 0.000000 | 0.000002 | 0.000000 |
| ctf_15m_hl_price_levels | dema_21_price_returns | 0.000000 | 0.000002 | 0.000000 |
| ctf_15m_hl_price_levels | volume_momentum_10 | 0.000000 | 0.000002 | 0.000000 |
| ctf_15m_hl_price_levels | vectorbt_volatility_comprehensive_20 | 0.000000 | 0.000002 | 0.000000 |
| ctf_15m_hl_price_levels | tema_21_price_returns | 0.000000 | 0.000002 | 0.000000 |
| ctf_15m_hl_price_levels | vectorbt_bbands_14_2.5 | 0.000000 | 0.000002 | 0.000000 |
| ctf_15m_hl_price_levels | volume_percentile_20 | 0.000000 | 0.000002 | 0.000000 |
| ctf_15m_hl_price_levels | ctf_corr_momentum_5_15_20_price_returns | 0.000000 | 0.000002 | 0.000000 |
| ctf_15m_hl_price_levels | keltner_channels_20_14_price_returns | 0.000000 | 0.000002 | 0.000000 |
| ctf_15m_hl_price_levels | stochastic_14_3_price_returns | 0.000000 | 0.000002 | 0.000000 |
