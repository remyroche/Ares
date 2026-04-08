# Training Report — 20260321_140000
Generated: 2026-04-07 22:40 UTC

## Configuration
- **Train lookback**: 35040 hours
- **Label horizons**: [3, 10]
- **Label method**: triple_barrier
- **Label quantiles**: lo=0.3, hi=0.65
- **OOS holdout**: 730 days
- **Min train samples**: 200
- **Feature selection**: MDI (min=30, cap=0.995)
- **15m precision**: True

## Dataset Sizes
| Dataset | Rows | Features |
|---------|------|----------|
| train_compression_ratio_1_0376017_dist_ema_fast_0_095782697_sin_hod_0_12940952_tail_asymmetry_q90_q10_atr_norm_0_47184935_10 | 13,342 | 919 |
| train_compression_ratio_1_0376017_dist_ema_fast_0_095782697_sin_hod_0_12940952_tail_asymmetry_q90_q10_atr_norm_0_47184935_10_tight | 6,671 | 919 |
| train_compression_ratio_1_0376017_dist_ema_fast_0_095782697_sin_hod_0_12940952_tail_asymmetry_q90_q10_atr_norm_0_47184935_10_wide | 6,671 | 919 |
| train_cos_hod_0_00000000000000000000000000000000001_slope_-0_83119529_volume_trend_48_-0_64375514_vov_mad_20_-0_37346116_5 | 11,602 | 919 |
| train_cos_hod_0_00000000000000000000000000000000001_slope_-0_83119529_volume_trend_48_-0_64375514_vov_mad_20_-0_37346116_5_tight | 5,801 | 919 |
| train_cos_hod_0_00000000000000000000000000000000001_slope_-0_83119529_volume_trend_48_-0_64375514_vov_mad_20_-0_37346116_5_wide | 5,801 | 919 |
| train_cos_hod_0_12940952_realized_volatility_24h_-0_25666356_rsi_slope_0_46084976_5 | 13,184 | 919 |
| train_cos_hod_0_12940952_realized_volatility_24h_-0_25666356_rsi_slope_0_46084976_5_tight | 6,592 | 919 |
| train_cos_hod_0_12940952_realized_volatility_24h_-0_25666356_rsi_slope_0_46084976_5_wide | 6,592 | 919 |
| train_loc_initial_balance_pos_24_0_32075533_loc_prev_day_range_pos_48_0_28101721_atr_compression_ratio_1_3097934_ret8h_-0_38833785_10 | 8,474 | 919 |
| train_loc_initial_balance_pos_24_0_32075533_loc_prev_day_range_pos_48_0_28101721_atr_compression_ratio_1_3097934_ret8h_-0_38833785_10_tight | 4,237 | 919 |
| train_loc_initial_balance_pos_24_0_32075533_loc_prev_day_range_pos_48_0_28101721_atr_compression_ratio_1_3097934_ret8h_-0_38833785_10_wide | 4,237 | 919 |

## Alpha Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Prec@10 | Prec@30 | Lift@30 | Prec@40 | AvgTr/day@10 | AvgTr/day@30 | Avg Return | Trades | Best Iter |
|-------|----------|-----|----|---------|----------|---------|---------|---------|---------|-------------|-------------|------------|--------|-----------|

### Per-Horizon Alpha Performance (Quality Gate)
| Model | Winner | AUC | IC | LogLoss | PR-AUC | Lift@20 | BrierImp | Passed |
|-------|--------|-----|----|---------|--------|---------|----------|--------|
| long_cos_hod_0_00000000000000000000000000000000001_slope_-0_83119529_volume_trend_48_-0_64375514_vov_mad_20_-0_37346116_H5:extratrees | — | 0.5885 | 0.1764 | 0.5751 | 0.3635 | nan | nan | False |
| short_cos_hod_0_12940952_realized_volatility_24h_-0_25666356_rsi_slope_0_46084976_H5:extratrees | — | 0.5700 | 0.1479 | 0.6010 | 0.3617 | nan | nan | False |
| long_loc_initial_balance_pos_24_0_32075533_loc_prev_day_range_pos_48_0_28101721_atr_compression_ratio_1_3097934_ret8h_-0_38833785_H10:extratrees | — | 0.5974 | 0.1895 | 0.5625 | 0.3661 | nan | nan | False |
| short_compression_ratio_1_0376017_dist_ema_fast_0_095782697_sin_hod_0_12940952_tail_asymmetry_q90_q10_atr_norm_0_47184935_H10:extratrees | — | 0.5838 | 0.1733 | 0.6154 | 0.4017 | nan | nan | False |

### Detailed Model Performance

#### LONG_MR: **NOT TRAINED**


#### LONG_TF: **NOT TRAINED**


#### SHORT_MR: **NOT TRAINED**


#### SHORT_TF: **NOT TRAINED**

## Specialist Models
