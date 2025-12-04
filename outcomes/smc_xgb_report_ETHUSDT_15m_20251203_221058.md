# SMC XGBoost Model Report

- **Symbol**: ETHUSDT
- **Exchange**: binance
- **Timeframe**: 15m
- **Generated**: 20251203_221058

## Model Performance

| Metric | Train | Val | Test |
| --- | --- | --- | --- |
| RMSE | nan | 2.7588 | 2.2581 |
| R² Score | nan | -0.0889 | -0.0048 |
| Brier Score | nan | 0.1734 | 0.1681 |
| Samples | 0 | 287 | 2020 |
| Features | 47 | 47 | 47 |

**Mean Return (Breakout Signals)**: 0.0173 (1569 signals)
**Mean Return (Breakdown Signals)**: -0.0267 (738 signals)

## Top 15 Features by Importance

| Rank | Feature | Importance | Correlation |
| --- | --- | --- | --- |
| 25 | smc_htf_trend_slope | 14.5089 | 0.0217 |
| 35 | smc_session_Dead | 14.4198 | -0.0276 |
| 10 | smc_gap_fill_ratio | 13.7119 | 0.0042 |
| 26 | smc_daily_wick_rejection | 12.3546 | 0.1717 |
| 3 | smc_dist_to_day_open | 12.2034 | -0.0257 |
| 38 | smc_dow_Wed | 12.2012 | 0.1717 |
| 37 | smc_dow_Tue | 11.6392 | -0.1485 |
| 5 | smc_nwog_gap_size | 10.5726 | -0.0530 |
| 8 | smc_consequent_encroachment | 10.4472 | 0.0215 |
| 4 | smc_dist_to_week_open | 9.0538 | -0.0672 |
| 36 | smc_dow_Mon | 9.0484 | 0.0566 |
| 15 | smc_break_of_structure_mag | 8.9184 | -0.0159 |
| 12 | smc_dist_to_swing_high | 8.7218 | -0.0971 |
| 1 | smc_dist_to_pdh_atr | 8.7003 | 0.0201 |
| 28 | smc_poc_dist_atr | 8.6670 | -0.0961 |

## Prediction Distribution Analysis

| Statistic | Predicted | Actual |
| --- | --- | --- |
| Mean | 0.5278 | -0.0952 |
| Std | 0.1298 | 2.3095 |
| Min | 0.1783 | -3.0000 |
| 25th Percentile | 0.4534 | -3.0000 |
| Median | 0.5472 | -0.0532 |
| 75th Percentile | 0.6036 | 2.2421 |
| Max | 0.8100 | 3.0000 |

## Scalar Band Performance (Deciles)

| Band | Count | Mean Scalar | Mean Forward Return | Directional Hit Rate |
| --- | --- | --- | --- | --- |
| (0.177, 0.332] | 232 | 0.2656 | -0.0149 | 93.53% |
| (0.332, 0.414] | 231 | 0.3794 | -0.0388 | 82.25% |
| (0.414, 0.494] | 231 | 0.4557 | -0.0316 | 71.00% |
| (0.494, 0.529] | 231 | 0.5125 | -0.0123 | 35.06% |
| (0.529, 0.547] | 231 | 0.5386 | 0.0200 | 60.17% |
| (0.547, 0.568] | 231 | 0.5577 | 0.0387 | 58.01% |
| (0.568, 0.59] | 231 | 0.5783 | 0.0433 | 48.92% |
| (0.59, 0.621] | 231 | 0.6041 | -0.0024 | 63.64% |
| (0.621, 0.687] | 231 | 0.6577 | 0.0159 | 87.45% |
| (0.687, 0.81] | 231 | 0.7294 | 0.0139 | 96.10% |

**Breakout Prediction Accuracy**: 48.76% (1569 predictions)
**Breakdown Prediction Accuracy**: 81.30% (738 predictions)
