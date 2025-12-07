# SMC XGBoost Model Report

- **Symbol**: ETHUSDT
- **Exchange**: binance
- **Timeframe**: 15m
- **Generated**: 20251207_131500

## Model Performance

| Metric | Train | Val | Test |
| --- | --- | --- | --- |
| RMSE | nan | 1.9351 | 1.9389 |
| R² Score | nan | -0.0006 | -0.0092 |
| Brier Score | nan | 0.1724 | 0.1835 |
| Samples | 0 | 11515 | 23319 |
| Features | 47 | 47 | 47 |

**Mean Return (Breakout Signals)**: 0.0059 (19898 signals)
**Mean Return (Breakdown Signals)**: -0.0070 (15047 signals)

## Top 15 Features by Importance

| Rank | Feature | Importance | Correlation |
| --- | --- | --- | --- |
| 35 | smc_session_Dead | 17.1852 | 0.0083 |
| 32 | smc_session_London | 16.7220 | -0.0399 |
| 38 | smc_dow_Wed | 15.8985 | 0.0488 |
| 21 | smc_adr_filled_pct | 14.9356 | 0.0113 |
| 31 | smc_session_Asia | 12.5346 | 0.0246 |
| 30 | smc_profile_skew | 12.2896 | 0.0087 |
| 36 | smc_dow_Mon | 11.7747 | -0.0155 |
| 28 | smc_poc_dist_atr | 11.6546 | -0.0066 |
| 3 | smc_dist_to_day_open | 11.5320 | -0.0178 |
| 39 | smc_dow_Thu | 11.4424 | -0.0261 |
| 26 | smc_daily_wick_rejection | 11.2285 | 0.0008 |
| 41 | smc_dow_Sat | 11.2127 | 0.0139 |
| 4 | smc_dist_to_week_open | 11.1119 | 0.0209 |
| 2 | smc_dist_to_pdl_atr | 11.0620 | -0.0307 |
| 10 | smc_gap_fill_ratio | 10.8375 | 0.0016 |

## Prediction Distribution Analysis

| Statistic | Predicted | Actual |
| --- | --- | --- |
| Mean | 0.5183 | 0.0405 |
| Std | 0.0993 | 1.9327 |
| Min | 0.1841 | -3.0000 |
| 25th Percentile | 0.4532 | -1.4788 |
| Median | 0.5168 | 0.0808 |
| 75th Percentile | 0.5833 | 1.5777 |
| Max | 0.8697 | 3.0000 |

## Scalar Band Performance (Deciles)

| Band | Count | Mean Scalar | Mean Forward Return | Directional Hit Rate |
| --- | --- | --- | --- | --- |
| (0.183, 0.394] | 3503 | 0.3428 | -0.0147 | 93.66% |
| (0.394, 0.437] | 3502 | 0.4177 | -0.0079 | 80.75% |
| (0.437, 0.467] | 3503 | 0.4529 | -0.0049 | 70.34% |
| (0.467, 0.492] | 3502 | 0.4799 | -0.0024 | 60.65% |
| (0.492, 0.517] | 3503 | 0.5045 | 0.0000 | 51.96% |
| (0.517, 0.542] | 3502 | 0.5292 | 0.0020 | 57.28% |
| (0.542, 0.568] | 3502 | 0.5547 | 0.0039 | 64.48% |
| (0.568, 0.601] | 3503 | 0.5837 | 0.0063 | 76.05% |
| (0.601, 0.648] | 3502 | 0.6224 | 0.0085 | 82.27% |
| (0.648, 0.87] | 3503 | 0.6949 | 0.0124 | 89.78% |

**Breakout Prediction Accuracy**: 49.13% (19898 predictions)
**Breakdown Prediction Accuracy**: 74.75% (15047 predictions)
