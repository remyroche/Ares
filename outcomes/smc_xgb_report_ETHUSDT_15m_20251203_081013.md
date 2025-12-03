# SMC XGBoost Model Report

- **Symbol**: ETHUSDT
- **Exchange**: binance
- **Timeframe**: 15m
- **Generated**: 20251203_081013

## Model Performance

| Metric | Train | Val | Test |
| --- | --- | --- | --- |
| RMSE | nan | 2.2047 | 1.8835 |
| R² Score | nan | 0.0536 | -0.0029 |
| Brier Score | nan | 0.1370 | 0.1732 |
| Samples | 0 | 384 | 2036 |
| Features | 47 | 47 | 47 |

**Mean Return (Breakout Signals)**: 0.0175 (1345 signals)
**Mean Return (Breakdown Signals)**: -0.0217 (981 signals)

## Top 15 Features by Importance

| Rank | Feature | Importance | Correlation |
| --- | --- | --- | --- |
| 37 | smc_dow_Tue | 13.4397 | -0.0841 |
| 15 | smc_break_of_structure_mag | 11.5060 | -0.0124 |
| 39 | smc_dow_Thu | 11.2778 | -0.0088 |
| 21 | smc_adr_filled_pct | 11.2319 | -0.0279 |
| 25 | smc_htf_trend_slope | 11.0022 | -0.0798 |
| 30 | smc_profile_skew | 9.9294 | 0.0361 |
| 3 | smc_dist_to_day_open | 9.7742 | -0.0409 |
| 12 | smc_dist_to_swing_high | 9.5798 | -0.1065 |
| 41 | smc_dow_Sat | 9.2722 | -0.0033 |
| 10 | smc_gap_fill_ratio | 9.2627 | 0.0019 |
| 2 | smc_dist_to_pdl_atr | 9.2233 | 0.0090 |
| 4 | smc_dist_to_week_open | 9.1975 | -0.0501 |
| 26 | smc_daily_wick_rejection | 8.8141 | 0.0631 |
| 28 | smc_poc_dist_atr | 8.3054 | -0.0473 |
| 1 | smc_dist_to_pdh_atr | 8.2467 | 0.0390 |

## Prediction Distribution Analysis

| Statistic | Predicted | Actual |
| --- | --- | --- |
| Mean | 0.5233 | -0.0185 |
| Std | 0.1342 | 1.9497 |
| Min | 0.2447 | -3.0000 |
| 25th Percentile | 0.4039 | -1.4718 |
| Median | 0.5490 | 0.0434 |
| 75th Percentile | 0.6364 | 1.4139 |
| Max | 0.7606 | 3.0000 |

## Scalar Band Performance (Deciles)

| Band | Count | Mean Scalar | Mean Forward Return | Directional Hit Rate |
| --- | --- | --- | --- | --- |
| (0.244, 0.33] | 243 | 0.2978 | -0.0419 | 97.94% |
| (0.33, 0.376] | 242 | 0.3515 | -0.0225 | 85.54% |
| (0.376, 0.432] | 242 | 0.4049 | -0.0174 | 76.03% |
| (0.432, 0.49] | 243 | 0.4597 | -0.0058 | 63.37% |
| (0.49, 0.549] | 242 | 0.5224 | -0.0021 | 49.17% |
| (0.549, 0.59] | 242 | 0.5696 | 0.0001 | 57.02% |
| (0.59, 0.622] | 243 | 0.6080 | 0.0077 | 64.20% |
| (0.622, 0.656] | 242 | 0.6385 | 0.0396 | 80.17% |
| (0.656, 0.686] | 242 | 0.6721 | 0.0258 | 85.95% |
| (0.686, 0.761] | 243 | 0.7090 | 0.0254 | 93.83% |

**Breakout Prediction Accuracy**: 50.19% (1345 predictions)
**Breakdown Prediction Accuracy**: 79.71% (981 predictions)
