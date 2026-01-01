# SMC XGBoost Model Report

- **Symbol**: ETHUSDT
- **Exchange**: binance
- **Timeframe**: 15m
- **Generated**: 20260101_203403

## Model Performance

| Metric | Train | Val | Test |
| --- | --- | --- | --- |
| RMSE | 1.9233 | 1.9475 | 1.9671 |
| R² Score | -0.0096 | -0.0169 | -0.0353 |
| Brier Score | 0.1800 | 0.1791 | 0.1988 |
| Samples | 50873 | 26113 | 26113 |
| Features | 47 | 47 | 47 |

**Mean Return (Breakout Signals)**: 0.0040 (55928 signals)
**Mean Return (Breakdown Signals)**: -0.0042 (48103 signals)

## Top 15 Features by Importance

| Rank | Feature | Importance | Correlation |
| --- | --- | --- | --- |
| 21 | smc_adr_filled_pct | 14.7661 | 0.0091 |
| 32 | smc_session_London | 12.7618 | -0.0325 |
| 3 | smc_dist_to_day_open | 12.3646 | -0.0181 |
| 42 | smc_dow_Sun | 12.3370 | 0.0005 |
| 2 | smc_dist_to_pdl_atr | 12.0982 | -0.0397 |
| 33 | smc_session_NY_AM | 11.8422 | -0.0058 |
| 37 | smc_dow_Tue | 11.7182 | -0.0133 |
| 30 | smc_profile_skew | 11.6112 | 0.0376 |
| 41 | smc_dow_Sat | 11.4660 | 0.0105 |
| 39 | smc_dow_Thu | 11.3096 | -0.0167 |
| 40 | smc_dow_Fri | 11.0177 | 0.0068 |
| 38 | smc_dow_Wed | 10.9506 | 0.0159 |
| 1 | smc_dist_to_pdh_atr | 10.9216 | -0.0084 |
| 15 | smc_break_of_structure_mag | 10.8847 | 0.0027 |
| 26 | smc_daily_wick_rejection | 10.7771 | 0.0260 |

## Prediction Distribution Analysis

| Statistic | Predicted | Actual |
| --- | --- | --- |
| Mean | 0.5069 | 0.0211 |
| Std | 0.0999 | 1.9232 |
| Min | 0.1221 | -3.0000 |
| 25th Percentile | 0.4451 | -1.4801 |
| Median | 0.5085 | 0.0508 |
| 75th Percentile | 0.5696 | 1.5190 |
| Max | 0.8712 | 3.0000 |

## Scalar Band Performance (Deciles)

| Band | Count | Mean Scalar | Mean Forward Return | Directional Hit Rate |
| --- | --- | --- | --- | --- |
| (0.121, 0.379] | 10450 | 0.3248 | -0.0087 | 86.89% |
| (0.379, 0.428] | 10449 | 0.4055 | -0.0050 | 75.91% |
| (0.428, 0.46] | 10449 | 0.4447 | -0.0032 | 66.84% |
| (0.46, 0.486] | 10449 | 0.4730 | -0.0019 | 59.42% |
| (0.486, 0.508] | 10449 | 0.4971 | -0.0005 | 51.84% |
| (0.508, 0.531] | 10450 | 0.5196 | 0.0011 | 55.61% |
| (0.531, 0.556] | 10448 | 0.5431 | 0.0026 | 62.23% |
| (0.556, 0.586] | 10449 | 0.5700 | 0.0038 | 69.52% |
| (0.586, 0.633] | 10449 | 0.6075 | 0.0055 | 77.08% |
| (0.633, 0.871] | 10450 | 0.6836 | 0.0086 | 86.74% |

**Breakout Prediction Accuracy**: 47.08% (55928 predictions)
**Breakdown Prediction Accuracy**: 69.81% (48103 predictions)
