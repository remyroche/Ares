# Policy Progression for loc_prev_day_range_pos_48_0_51315737_realized_volatility_24h_0_3200236_ret24h_-0_31203175_rsi_slope_-1_8608685

Run ID: 20260321_140000

Step            |    Score |      PnL | Rb_DownRatio | PnL_25pct | WkSortino | MoSortino | MedPnL/Win |    WkGtP |   Skew |  Ulcer |   TUW | NegMo% |   DD_Mag | Trades/Day | AvgNetPnL | Params
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Baseline        |   1.3775 |  -5.0088 |    -270.0048 |    2.8953 |   -0.3037 |   -0.3037 |     0.0029 |   0.3904 | -2.0571 | 84.3367 | 0.9978 | 100.0% |   0.9947 |       1.23 |  -0.00563 | {'tp': 1.0, 'sl': 1.0}
position_sizing |   1.8030 |  -0.4328 |    -240.4848 |    0.2432 |   -0.2705 |   -0.2705 |     0.0003 |   0.3800 | -3.4509 | 23.4370 | 0.9978 | 100.0% |   0.3532 |       1.23 |  -0.00049 | {'size_power': 2.0, 'tp': 1.0, 'sl': 1.0}
trailing_stop   |   2.1599 |  -0.3829 |    -211.5328 |    0.2693 |   -0.2379 |   -0.2379 |     0.0005 |   0.4335 | -3.7027 | 22.1069 | 0.9944 | 100.0% |   0.3226 |       1.23 |  -0.00043 | {'size_power': 2.0, 'trailing_power': 1.3, 'trailing_squash_divisor': 2.5, 'trailing_override_alpha': 0.7, 'giveback_beta': 0.3, 'tp': 1.0, 'sl': 1.0}

