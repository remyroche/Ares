# Policy Progression for bars_in_high_vol_state_log_norm_0_45356679_dist_prior_day_low_0_011585106_rvol_z_0_89040798_volume_trend_48_-1_0869701_tbm

Run ID: 20260321_140000

Step            |    Score |      PnL | Rb_DownRatio | PnL_25pct | WkSortino | MoSortino | MedPnL/Win |    WkGtP |   Skew |  Ulcer |   TUW | NegMo% |   DD_Mag | Trades/Day | AvgNetPnL | Params
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Baseline        |   2.8797 |  -0.1746 |    -585.2995 |    0.1049 |   -0.9246 |   -0.9246 |     0.0005 |   0.3785 | 1.1103 | 9.7036 | 0.9984 | 100.0% |   0.1607 |       0.87 |  -0.00028 | {'tp': 0.5, 'sl': 0.18}
tp_sl_geometry  |   5.7221 |  -0.0746 |    -117.6139 |    0.3186 |   -0.1858 |   -0.1858 |     0.0012 |   0.8271 | 0.8562 | 5.0343 | 0.9905 |  66.7% |   0.0914 |       0.87 |  -0.00012 | {'tp': 1.3, 'sl': 0.45}
trailing_stop   |   6.1189 |  -0.0660 |    -104.5289 |    0.3460 |   -0.1651 |   -0.1651 |     0.0012 |   0.8504 | 1.7395 | 5.3180 | 0.9826 |  75.0% |   0.0938 |       0.87 |  -0.00010 | {'trailing_power': 1.4, 'trailing_squash_divisor': 3.5, 'trailing_override_alpha': 1.5, 'giveback_beta': 0.3, 'tp': 1.3, 'sl': 0.45}
position_sizing |   6.1580 |  -0.0536 |     -95.5938 |    0.2907 |   -0.1510 |   -0.1510 |     0.0010 |   0.8542 | 1.7820 | 4.5692 | 0.9826 |  75.0% |   0.0777 |       0.87 |  -0.00008 | {'size_power': 2.0, 'trailing_power': 1.4, 'trailing_squash_divisor': 3.5, 'trailing_override_alpha': 1.5, 'giveback_beta': 0.3, 'tp': 1.3, 'sl': 0.45}

