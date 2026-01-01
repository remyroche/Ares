# SMC XGBoost Model Report

- **Symbol**: ETHUSDT
- **Exchange**: binance
- **Timeframe**: 15m
- **Generated**: 20251231_172755

## Model Performance

| Metric | Train | Val | Test |
| --- | --- | --- | --- |
| RMSE | 1.9297 | 1.9545 | 1.9702 |
| R² Score | -0.0164 | -0.0242 | -0.0385 |
| Brier Score | 0.1840 | 0.1831 | 0.2001 |
| Samples | 50873 | 26113 | 26113 |
| Features | 47 | 47 | 47 |

**Mean Return (Breakout Signals)**: 0.0038 (56140 signals)
**Mean Return (Breakdown Signals)**: -0.0039 (48031 signals)

## Top 15 Features by Importance

| Rank | Feature | Importance | Correlation |
| --- | --- | --- | --- |
| 21 | smc_adr_filled_pct | 15.7977 | 0.0091 |
| 32 | smc_session_London | 13.8541 | -0.0325 |
| 3 | smc_dist_to_day_open | 13.3740 | -0.0181 |
| 2 | smc_dist_to_pdl_atr | 12.7828 | -0.0397 |
| 42 | smc_dow_Sun | 12.5339 | 0.0005 |
| 41 | smc_dow_Sat | 12.5185 | 0.0105 |
| 30 | smc_profile_skew | 12.4133 | 0.0376 |
| 37 | smc_dow_Tue | 12.1038 | -0.0133 |
| 33 | smc_session_NY_AM | 11.8663 | -0.0058 |
| 35 | smc_session_Dead | 11.8486 | 0.0071 |
| 40 | smc_dow_Fri | 11.8276 | 0.0068 |
| 1 | smc_dist_to_pdh_atr | 11.4505 | -0.0084 |
| 38 | smc_dow_Wed | 11.4339 | 0.0159 |
| 14 | smc_fib_retracement_level | 11.3672 | 0.0310 |
| 26 | smc_daily_wick_rejection | 11.3271 | 0.0260 |

## Prediction Distribution Analysis

| Statistic | Predicted | Actual |
| --- | --- | --- |
| Mean | 0.5069 | 0.0211 |
| Std | 0.0911 | 1.9232 |
| Min | 0.1388 | -3.0000 |
| 25th Percentile | 0.4512 | -1.4801 |
| Median | 0.5079 | 0.0508 |
| 75th Percentile | 0.5632 | 1.5190 |
| Max | 0.8562 | 3.0000 |

## Scalar Band Performance (Deciles)

| Band | Count | Mean Scalar | Mean Forward Return | Directional Hit Rate |
| --- | --- | --- | --- | --- |
| (0.138, 0.391] | 10450 | 0.3409 | -0.0084 | 85.83% |
| (0.391, 0.435] | 10449 | 0.4153 | -0.0047 | 74.15% |
| (0.435, 0.464] | 10449 | 0.4508 | -0.0030 | 65.54% |
| (0.464, 0.488] | 10449 | 0.4763 | -0.0018 | 58.46% |
| (0.488, 0.508] | 10449 | 0.4978 | -0.0003 | 51.75% |
| (0.508, 0.528] | 10449 | 0.5179 | 0.0011 | 55.31% |
| (0.528, 0.55] | 10449 | 0.5389 | 0.0022 | 61.01% |
| (0.55, 0.578] | 10449 | 0.5636 | 0.0037 | 68.56% |
| (0.578, 0.622] | 10449 | 0.5980 | 0.0053 | 75.49% |
| (0.622, 0.856] | 10450 | 0.6692 | 0.0081 | 85.33% |

**Breakout Prediction Accuracy**: 46.20% (56140 predictions)
**Breakdown Prediction Accuracy**: 68.66% (48031 predictions)
