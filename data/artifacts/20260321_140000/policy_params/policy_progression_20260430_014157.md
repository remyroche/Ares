# Policy Progression for dist_prior_day_high_0_035536911_dist_prior_day_low_0_012770575_vol_z_0_6473701_tbm

Run ID: 20260321_140000

Step            |    Score |      PnL | Rb_DownRatio | PnL_25pct | WkSortino | MoSortino | MedPnL/Win |    WkGtP |   Skew |  Ulcer |   TUW | NegMo% |   DD_Mag | Trades/Day | AvgNetPnL | Params
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Baseline        |  -2.3274 |  -3.2171 |    -447.8379 |    1.6697 |   -1.2546 |   -2.9248 |     0.0078 |   0.0022 | 0.4338 | 71.4331 | 0.9965 | 100.0% |   0.4166 |       3.27 |  -0.00570 | {'tp': 1.0, 'sl': 1.0}
trailing_stop   |   6.2833 |  -2.7809 |    -389.2942 |    2.5564 |   -1.1132 |   -2.2030 |     0.0104 |   0.1082 | 3.8658 | 68.0640 | 0.9947 | 100.0% |   0.4123 |       3.27 |  -0.00493 | {'trailing_power': 1.5, 'trailing_squash_divisor': 3.5, 'trailing_override_alpha': 1.5, 'giveback_beta': 0.3, 'tp': 1.0, 'sl': 1.0}
position_sizing |   9.4874 |  -0.2238 |    -283.9190 |    0.2149 |   -1.1875 |   -2.1439 |     0.0009 |   0.1343 | 4.7137 | 12.1681 | 0.9947 | 100.0% |   0.0329 |       3.27 |  -0.00040 | {'size_power': 2.0, 'trailing_power': 1.5, 'trailing_squash_divisor': 3.5, 'trailing_override_alpha': 1.5, 'giveback_beta': 0.3, 'tp': 1.0, 'sl': 1.0}

