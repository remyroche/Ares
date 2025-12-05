# SMC XGBoost Model Report

- **Symbol**: ETHUSDT
- **Exchange**: binance
- **Timeframe**: 15m
- **Generated**: 20251204_224415

## Model Performance

| Metric | Train | Val | Test |
| --- | --- | --- | --- |
| RMSE | 1.9824 | 2.1281 | 1.9130 |
| R² Score | 0.0731 | -0.0049 | -0.0345 |
| Brier Score | 0.1463 | 0.1447 | 0.1880 |
| Samples | 1651 | 869 | 2036 |
| Features | 47 | 47 | 47 |

**Mean Return (Breakout Signals)**: 0.0222 (2600 signals)
**Mean Return (Breakdown Signals)**: -0.0196 (1700 signals)

## Top 15 Features by Importance

| Rank | Feature | Importance | Correlation |
| --- | --- | --- | --- |
| 32 | smc_session_London | 14.1241 | 0.0418 |
| 3 | smc_dist_to_day_open | 12.3923 | -0.0618 |
| 26 | smc_daily_wick_rejection | 11.7994 | 0.0139 |
| 21 | smc_adr_filled_pct | 11.1842 | -0.0178 |
| 42 | smc_dow_Sun | 10.0940 | 0.0692 |
| 24 | smc_atr_compression | 9.8563 | -0.0117 |
| 29 | smc_is_in_value_area | 9.7893 | -0.0401 |
| 15 | smc_break_of_structure_mag | 9.7320 | 0.0194 |
| 28 | smc_poc_dist_atr | 9.7020 | -0.0061 |
| 38 | smc_dow_Wed | 9.6515 | -0.0199 |
| 25 | smc_htf_trend_slope | 9.5545 | 0.0163 |
| 36 | smc_dow_Mon | 9.4788 | -0.0210 |
| 12 | smc_dist_to_swing_high | 9.3999 | -0.0448 |
| 5 | smc_nwog_gap_size | 9.3779 | 0.0223 |
| 1 | smc_dist_to_pdh_atr | 9.0702 | 0.0157 |

## Prediction Distribution Analysis

| Statistic | Predicted | Actual |
| --- | --- | --- |
| Mean | 0.5248 | 0.0384 |
| Std | 0.1242 | 2.0056 |
| Min | 0.1924 | -3.0000 |
| 25th Percentile | 0.4364 | -1.4683 |
| Median | 0.5346 | 0.0434 |
| 75th Percentile | 0.6230 | 1.5945 |
| Max | 0.8012 | 3.0000 |

## Scalar Band Performance (Deciles)

| Band | Count | Mean Scalar | Mean Forward Return | Directional Hit Rate |
| --- | --- | --- | --- | --- |
| (0.191, 0.349] | 466 | 0.2978 | -0.0275 | 97.21% |
| (0.349, 0.41] | 466 | 0.3793 | -0.0234 | 90.13% |
| (0.41, 0.46] | 465 | 0.4361 | -0.0135 | 77.63% |
| (0.46, 0.501] | 466 | 0.4820 | -0.0079 | 65.67% |
| (0.501, 0.535] | 465 | 0.5176 | -0.0033 | 52.26% |
| (0.535, 0.57] | 466 | 0.5522 | 0.0064 | 62.23% |
| (0.57, 0.604] | 465 | 0.5870 | 0.0131 | 68.39% |
| (0.604, 0.642] | 466 | 0.6232 | 0.0186 | 78.54% |
| (0.642, 0.679] | 465 | 0.6589 | 0.0270 | 81.72% |
| (0.679, 0.801] | 466 | 0.7142 | 0.0629 | 94.42% |

**Breakout Prediction Accuracy**: 54.08% (2600 predictions)
**Breakdown Prediction Accuracy**: 84.24% (1700 predictions)
