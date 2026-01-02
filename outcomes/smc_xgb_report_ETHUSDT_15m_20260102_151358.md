# SMC XGBoost Model Report

- **Symbol**: ETHUSDT
- **Exchange**: binance
- **Timeframe**: 15m
- **Generated**: 20260102_151358

## Model Performance

| Metric | Train | Val | Test |
| --- | --- | --- | --- |
| RMSE | 2.0907 | 2.1208 | 2.1363 |
| R² Score | -0.1931 | -0.2059 | -0.2210 |
| Brier Score | nan | nan | nan |
| Samples | 50873 | 26113 | 26113 |
| Features | 47 | 47 | 47 |

**Mean Return (Breakout Signals)**: 0.0008 (98774 signals)
**Mean Return (Breakdown Signals)**: -0.0109 (3 signals)

## Top 15 Features by Importance

| Rank | Feature | Importance | Correlation |
| --- | --- | --- | --- |
| 21 | smc_adr_filled_pct | 34.8093 | 0.0091 |
| 2 | smc_dist_to_pdl_atr | 29.6863 | -0.0397 |
| 3 | smc_dist_to_day_open | 28.5723 | -0.0181 |
| 33 | smc_session_NY_AM | 28.2414 | -0.0058 |
| 30 | smc_profile_skew | 26.6576 | 0.0376 |
| 14 | smc_fib_retracement_level | 24.8315 | 0.0310 |
| 37 | smc_dow_Tue | 24.0319 | -0.0133 |
| 15 | smc_break_of_structure_mag | 23.2947 | 0.0027 |
| 41 | smc_dow_Sat | 22.9942 | 0.0105 |
| 38 | smc_dow_Wed | 22.7345 | 0.0159 |
| 25 | smc_htf_trend_slope | 22.0978 | -0.0136 |
| 1 | smc_dist_to_pdh_atr | 21.9932 | -0.0084 |
| 26 | smc_daily_wick_rejection | 21.7580 | 0.0260 |
| 11 | smc_range_position | 21.6267 | -0.0310 |
| 40 | smc_dow_Fri | 21.2076 | 0.0068 |

## Prediction Distribution Analysis

| Statistic | Predicted | Actual |
| --- | --- | --- |
| Mean | 1.0130 | 0.0211 |
| Std | 0.1616 | 1.9232 |
| Min | 0.1462 | -3.0000 |
| 25th Percentile | 0.9185 | -1.4801 |
| Median | 1.0128 | 0.0508 |
| 75th Percentile | 1.1067 | 1.5190 |
| Max | 1.7601 | 3.0000 |

## Scalar Band Performance (Deciles)

| Band | Count | Mean Scalar | Mean Forward Return | Directional Hit Rate |
| --- | --- | --- | --- | --- |
| (0.145, 0.813] | 10450 | 0.7186 | -0.0079 | 17.60% |
| (0.813, 0.892] | 10449 | 0.8566 | -0.0043 | 27.46% |
| (0.892, 0.941] | 10449 | 0.9179 | -0.0028 | 35.73% |
| (0.941, 0.979] | 10449 | 0.9607 | -0.0015 | 42.49% |
| (0.979, 1.013] | 10449 | 0.9961 | -0.0002 | 48.81% |
| (1.013, 1.047] | 10449 | 1.0295 | 0.0009 | 54.43% |
| (1.047, 1.084] | 10449 | 1.0651 | 0.0020 | 60.40% |
| (1.084, 1.133] | 10449 | 1.1074 | 0.0033 | 66.64% |
| (1.133, 1.216] | 10449 | 1.1703 | 0.0050 | 73.72% |
| (1.216, 1.76] | 10450 | 1.3079 | 0.0079 | 84.23% |

**Breakout Prediction Accuracy**: 34.40% (98774 predictions)
**Breakdown Prediction Accuracy**: 100.00% (3 predictions)
