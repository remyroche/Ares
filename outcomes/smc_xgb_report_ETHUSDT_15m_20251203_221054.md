# SMC XGBoost Model Report

- **Symbol**: ETHUSDT
- **Exchange**: binance
- **Timeframe**: 15m
- **Generated**: 20251203_221054

## Model Performance

| Metric | Train | Val | Test |
| --- | --- | --- | --- |
| RMSE | nan | 2.1079 | 1.7633 |
| R² Score | nan | -0.0362 | -0.0394 |
| Brier Score | nan | 0.1779 | 0.1860 |
| Samples | 0 | 287 | 2040 |
| Features | 47 | 47 | 47 |

**Mean Return (Breakout Signals)**: 0.0137 (1261 signals)
**Mean Return (Breakdown Signals)**: -0.0145 (1028 signals)

## Top 15 Features by Importance

| Rank | Feature | Importance | Correlation |
| --- | --- | --- | --- |
| 38 | smc_dow_Wed | 11.6136 | 0.0981 |
| 24 | smc_atr_compression | 9.6633 | 0.0471 |
| 3 | smc_dist_to_day_open | 9.5556 | -0.0337 |
| 39 | smc_dow_Thu | 9.3347 | -0.0073 |
| 8 | smc_consequent_encroachment | 9.2183 | 0.0094 |
| 21 | smc_adr_filled_pct | 9.1096 | -0.0206 |
| 15 | smc_break_of_structure_mag | 9.0800 | -0.0102 |
| 5 | smc_nwog_gap_size | 8.7433 | 0.0139 |
| 26 | smc_daily_wick_rejection | 8.4792 | 0.0240 |
| 25 | smc_htf_trend_slope | 8.1019 | -0.0242 |
| 32 | smc_session_London | 8.0816 | 0.0756 |
| 23 | smc_time_elapsed_session | 8.0255 | -0.0338 |
| 34 | smc_session_NY_PM | 8.0064 | -0.0325 |
| 30 | smc_profile_skew | 7.7018 | -0.0188 |
| 12 | smc_dist_to_swing_high | 7.6627 | -0.0965 |

## Prediction Distribution Analysis

| Statistic | Predicted | Actual |
| --- | --- | --- |
| Mean | 0.5107 | -0.0290 |
| Std | 0.0858 | 1.7786 |
| Min | 0.2975 | -3.0000 |
| 25th Percentile | 0.4402 | -1.1690 |
| Median | 0.5178 | -0.0007 |
| 75th Percentile | 0.5814 | 1.1885 |
| Max | 0.6854 | 3.0000 |

## Scalar Band Performance (Deciles)

| Band | Count | Mean Scalar | Mean Forward Return | Directional Hit Rate |
| --- | --- | --- | --- | --- |
| (0.296, 0.398] | 234 | 0.3640 | -0.0141 | 88.46% |
| (0.398, 0.426] | 233 | 0.4125 | -0.0265 | 88.84% |
| (0.426, 0.453] | 233 | 0.4399 | -0.0127 | 73.82% |
| (0.453, 0.486] | 234 | 0.4687 | -0.0093 | 59.83% |
| (0.486, 0.518] | 232 | 0.5019 | -0.0028 | 48.71% |
| (0.518, 0.546] | 233 | 0.5328 | 0.0031 | 61.37% |
| (0.546, 0.569] | 233 | 0.5583 | 0.0056 | 65.24% |
| (0.569, 0.592] | 233 | 0.5810 | 0.0145 | 71.67% |
| (0.592, 0.622] | 233 | 0.6063 | 0.0169 | 75.54% |
| (0.622, 0.685] | 233 | 0.6425 | 0.0356 | 89.27% |

**Breakout Prediction Accuracy**: 44.01% (1261 predictions)
**Breakdown Prediction Accuracy**: 75.29% (1028 predictions)
