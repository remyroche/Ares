# Policy Progression for bars_in_high_vol_state_log_norm_0_45356679_dist_prior_day_low_0_011585106_rvol_z_0_89040798_volume_trend_48_-1_0869701_tbm

Run ID: 20260321_140000

Step            |    Score |      PnL | Rb_DownRatio | PnL_25pct | WkSortino | MoSortino | MedPnL/Win |    WkGtP |   Skew |  Ulcer |   TUW | NegMo% |   DD_Mag | Trades/Day | AvgNetPnL | Params
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Baseline        |  12.1442 |   0.6339 |     108.8431 |    1.7634 |    1.1680 |   20.0000 |     0.0080 |   3.3126 | 0.2640 | 6.9810 | 0.8610 |  20.0% |   0.0969 |       2.95 |   0.00155 | {'tp': 1.0, 'sl': 1.0}
trailing_stop   |  13.3396 |   0.4406 |      72.1404 |    2.0232 |    0.8950 |   20.0000 |     0.0090 |   1.9562 | 0.5580 | 8.1499 | 0.9220 |  20.0% |   0.1192 |       2.95 |   0.00107 | {'trailing_power': 1.7, 'trailing_squash_divisor': 1.5, 'trailing_override_alpha': 1.5, 'giveback_beta': 1.0, 'tp': 1.0, 'sl': 1.0}
position_sizing |  34.1703 |   1.5681 |     285.9179 |    2.3068 |    7.0792 |   20.0000 |     0.0107 |  21.9311 | 0.5581 | 2.7579 | 0.7171 |   0.0% |   0.0505 |       2.95 |   0.00382 | {'size_power': 2.0, 'trailing_power': 1.7, 'trailing_squash_divisor': 1.5, 'trailing_override_alpha': 1.5, 'giveback_beta': 1.0, 'tp': 1.0, 'sl': 1.0}

