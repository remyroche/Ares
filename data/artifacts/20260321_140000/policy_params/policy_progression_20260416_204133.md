# Policy Progression for bars_in_high_vol_state_log_norm_-0_61002535_dist_weekly_vwap_-0_38453856_loc_ema_stack_pos_48_0_66408622_loc_prev_day_range_pos_24_0_41419065_atr_compression_ratio_0_97724169_ema50_slope_0_00070510805

Run ID: 20260321_140000

Step            |    Score |      PnL | Rb_DownRatio | PnL_25pct | WkSortino | MoSortino | MedPnL/Win |    WkGtP |   Skew |  Ulcer |   TUW | NegMo% |   DD_Mag | Trades/Day | AvgNetPnL | Params
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Baseline        |   0.2192 |  -5.5614 |    -246.8596 |    1.5619 |   -0.3906 |   -0.3906 |     0.0020 |   0.2413 | -2.2511 | 83.8110 | 0.9953 | 100.0% |   0.9967 |       0.87 |  -0.00880 | {'tp': 1.0, 'sl': 1.0}
position_sizing |   1.7089 |  -0.4346 |    -236.1562 |    0.1303 |   -0.3737 |   -0.3737 |     0.0002 |   0.2491 | -2.2344 | 21.4964 | 0.9953 | 100.0% |   0.3524 |       0.87 |  -0.00069 | {'size_power': 2.0, 'tp': 1.0, 'sl': 1.0}
trailing_stop   |   2.8286 |  -0.3458 |    -186.5771 |    0.1851 |   -0.2952 |   -0.2952 |     0.0006 |   0.3657 | -2.1847 | 17.0509 | 0.9953 |  91.7% |   0.2921 |       0.87 |  -0.00055 | {'size_power': 2.0, 'trailing_power': 1.2, 'trailing_squash_divisor': 3.25, 'trailing_override_alpha': 0.5, 'giveback_beta': 0.3, 'tp': 1.0, 'sl': 1.0}

