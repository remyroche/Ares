# SMC XGBoost Model Report

- **Symbol**: ETHUSDT
- **Exchange**: binance
- **Timeframe**: 15m
- **Generated**: 20251203_231509

## Model Performance

| Metric | Train | Val | Test |
| --- | --- | --- | --- |
| RMSE | nan | 2.3528 | 1.9619 |
| R² Score | nan | -0.0177 | -0.0069 |
| Brier Score | nan | 0.1563 | 0.1768 |
| Samples | 0 | 287 | 2034 |
| Features | 47 | 47 | 47 |

**Mean Return (Breakout Signals)**: 0.0190 (1222 signals)
**Mean Return (Breakdown Signals)**: -0.0193 (1016 signals)

## Top 15 Features by Importance

| Rank | Feature | Importance | Correlation |
| --- | --- | --- | --- |
| 37 | smc_dow_Tue | 15.8316 | -0.0912 |
| 38 | smc_dow_Wed | 14.3988 | 0.1247 |
| 35 | smc_session_Dead | 13.9066 | -0.0416 |
| 5 | smc_nwog_gap_size | 13.4391 | -0.0121 |
| 3 | smc_dist_to_day_open | 12.0460 | -0.0287 |
| 30 | smc_profile_skew | 11.7620 | -0.0202 |
| 21 | smc_adr_filled_pct | 11.7244 | -0.0315 |
| 26 | smc_daily_wick_rejection | 11.0857 | 0.0826 |
| 32 | smc_session_London | 10.9801 | 0.0866 |
| 4 | smc_dist_to_week_open | 9.9108 | -0.0448 |
| 40 | smc_dow_Fri | 9.9059 | -0.1099 |
| 28 | smc_poc_dist_atr | 9.5613 | -0.0387 |
| 25 | smc_htf_trend_slope | 9.3042 | -0.0886 |
| 15 | smc_break_of_structure_mag | 9.2625 | -0.0107 |
| 24 | smc_atr_compression | 8.9625 | 0.0399 |

## Prediction Distribution Analysis

| Statistic | Predicted | Actual |
| --- | --- | --- |
| Mean | 0.5136 | -0.0495 |
| Std | 0.1182 | 2.0087 |
| Min | 0.2561 | -3.0000 |
| 25th Percentile | 0.4079 | -1.7167 |
| Median | 0.5233 | 0.0272 |
| 75th Percentile | 0.6173 | 1.5106 |
| Max | 0.7412 | 3.0000 |

## Scalar Band Performance (Deciles)

| Band | Count | Mean Scalar | Mean Forward Return | Directional Hit Rate |
| --- | --- | --- | --- | --- |
| (0.255, 0.352] | 233 | 0.3247 | -0.0316 | 97.85% |
| (0.352, 0.388] | 232 | 0.3708 | -0.0191 | 80.17% |
| (0.388, 0.43] | 233 | 0.4076 | -0.0103 | 72.53% |
| (0.43, 0.475] | 232 | 0.4518 | -0.0137 | 68.53% |
| (0.475, 0.523] | 233 | 0.5000 | -0.0134 | 60.52% |
| (0.523, 0.56] | 232 | 0.5408 | 0.0011 | 66.81% |
| (0.56, 0.598] | 232 | 0.5812 | 0.0118 | 63.79% |
| (0.598, 0.635] | 233 | 0.6168 | 0.0106 | 69.10% |
| (0.635, 0.666] | 232 | 0.6519 | 0.0260 | 81.47% |
| (0.666, 0.741] | 233 | 0.6905 | 0.0547 | 95.71% |

**Breakout Prediction Accuracy**: 54.01% (1222 predictions)
**Breakdown Prediction Accuracy**: 79.13% (1016 predictions)
