# Base Training Report — 20260321_140000
Generated: 2026-04-08 20:02 UTC

## Alpha Model Performance per Strategy / Horizon
| Strategy ID | Side | H | Model | Winner | AUC | IC(bin) | IC(ret) | LogLoss | PR-AUC | Lift@20 | Prec@10 | Prec@30 | N features |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| compression_ratio_1_0376017_dist_ema_fast_0_095782697_sin_hod_0_12940952_tail_asymmetry_q90_q10_atr_norm_0_47184935 | short | 10 | extratrees | ✓ | 0.5746 | 0.1561 | 0.1388 | 0.6158 | 0.3970 | 1.2380 | 0.4996 | 0.3836 | 48 |
| cos_hod_0_00000000000000000000000000000000001_slope_-0_83119529_volume_trend_48_-0_64375514_vov_mad_20_-0_37346116 | long | 5 | extratrees | ✓ | 0.5697 | 0.1385 | 0.1214 | 0.5860 | 0.3484 | 1.3052 | 0.4530 | 0.3405 | 48 |
| loc_initial_balance_pos_24_0_32075533_loc_prev_day_range_pos_48_0_28101721_atr_compression_ratio_1_3097934_ret8h_-0_38833785 | long | 10 | extratrees | ✓ | 0.5850 | 0.1639 | 0.1177 | 0.5688 | 0.3519 | 1.3062 | 0.4629 | 0.3425 | 48 |
| cos_hod_0_12940952_realized_volatility_24h_-0_25666356_rsi_slope_0_46084976 | short | 5 | extratrees | ✓ | 0.5545 | 0.1131 | 0.0943 | 0.5833 | 0.3209 | 1.2021 | 0.4079 | 0.3235 | 48 |

## Per-Strategy Summary
| Strategy ID | Side | Deployed Hs | Primary H | Median AUC | Median IC | Median PR-AUC |
|---|---|---|---|---|---|---|
| cos_hod_0_00000000000000000000000000000000001_slope_-0_83119529_volume_trend_48_-0_64375514_vov_mad_20_-0_37346116 | long | [5] | 5 | 0.5697 | 0.1385 | 0.3484 |
| loc_initial_balance_pos_24_0_32075533_loc_prev_day_range_pos_48_0_28101721_atr_compression_ratio_1_3097934_ret8h_-0_38833785 | long | [10] | 10 | 0.5850 | 0.1639 | 0.3519 |
| compression_ratio_1_0376017_dist_ema_fast_0_095782697_sin_hod_0_12940952_tail_asymmetry_q90_q10_atr_norm_0_47184935 | short | [10] | 10 | 0.5746 | 0.1561 | 0.3970 |
| cos_hod_0_12940952_realized_volatility_24h_-0_25666356_rsi_slope_0_46084976 | short | [5] | 5 | 0.5545 | 0.1131 | 0.3209 |
