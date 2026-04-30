# Policy Progression for dist_prior_day_low_0_012672868_range_expansion_ratio_-0_26680329_variance_ratio_10_48_-0_37689748_vov_fast_slow_ratio_0_56139898

Run ID: 20260321_140000

Step            |    Score |      PnL | Rb_DownRatio | PnL_25pct | WkSortino | MoSortino | MedPnL/Win |    WkGtP |   Skew |  Ulcer |   TUW | NegMo% |   DD_Mag | Trades/Day | AvgNetPnL | Params
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Baseline        |   3.2088 |  -0.1829 |    -796.6952 |    0.0670 |   -1.2933 |   -1.2933 |     0.0003 |   0.2683 | 1.3348 | 10.7531 | 0.9968 | 100.0% |   0.1680 |       0.85 |  -0.00030 | {'tp': 0.5, 'sl': 0.18}
tp_sl_geometry  |   5.7786 |  -0.1536 |    -334.8780 |    0.2358 |   -0.5436 |   -0.5436 |     0.0011 |   0.6115 | 1.4146 | 9.1641 | 0.9968 |  91.7% |   0.1467 |       0.85 |  -0.00025 | {'tp': 1.3, 'sl': 0.45}
trailing_stop   |   6.1398 |  -0.1422 |    -309.7954 |    0.2510 |   -0.5029 |   -0.5029 |     0.0011 |   0.6419 | 1.7146 | 8.8201 | 0.9968 |  83.3% |   0.1380 |       0.85 |  -0.00023 | {'trailing_power': 1.8, 'trailing_squash_divisor': 3.25, 'trailing_override_alpha': 1.5, 'giveback_beta': 0.5, 'tp': 1.3, 'sl': 0.45}
position_sizing |   6.2848 |  -0.1161 |    -267.7811 |    0.2123 |   -0.4347 |   -0.4347 |     0.0009 |   0.6500 | 1.7833 | 7.4626 | 0.9968 |  83.3% |   0.1147 |       0.85 |  -0.00019 | {'size_power': 2.0, 'trailing_power': 1.8, 'trailing_squash_divisor': 3.25, 'trailing_override_alpha': 1.5, 'giveback_beta': 0.5, 'tp': 1.3, 'sl': 0.45}

