# SMC XGBoost Model Report

- **Symbol**: ETHUSDT
- **Exchange**: binance
- **Timeframe**: 1h
- **Generated**: 20251124_233037

## Model Performance

| Metric | Train | Val | Test |
| --- | --- | --- | --- |
| RMSE | 4.2486 | 4.3137 | 4.9346 |
| R² Score | 0.0164 | -0.0256 | -0.0009 |
| Brier Score | 0.1892 | 0.2384 | 0.2431 |
| Samples | 5000 | 1249 | 1563 |
| Features | 47 | 47 | 47 |

**Mean Return (Breakout Signals)**: 0.0125 (2868 signals)
**Mean Return (Breakdown Signals)**: -0.0115 (2387 signals)

## Top 15 Features by Importance

| Rank | Feature | Importance | Correlation |
| --- | --- | --- | --- |
| 22 | smc_rel_volume | 0.0508 | -0.0004 |
| 16 | smc_displacement_strength | 0.0448 | 0.0021 |
| 4 | smc_dist_to_week_open | 0.0397 | 0.0469 |
| 30 | smc_profile_skew | 0.0396 | 0.0142 |
| 21 | smc_adr_filled_pct | 0.0368 | 0.0011 |
| 3 | smc_dist_to_day_open | 0.0358 | 0.0257 |
| 36 | smc_dow_Mon | 0.0352 | -0.0119 |
| 25 | smc_htf_trend_slope | 0.0340 | 0.0056 |
| 20 | smc_consecutive_candles | 0.0331 | 0.0745 |
| 26 | smc_daily_wick_rejection | 0.0326 | 0.0087 |
| 2 | smc_dist_to_pdl_atr | 0.0320 | 0.0125 |
| 42 | smc_dow_Sun | 0.0314 | -0.0403 |
| 28 | smc_poc_dist_atr | 0.0312 | -0.0305 |
| 5 | smc_nwog_gap_size | 0.0306 | -0.0524 |
| 18 | smc_close_position_in_candle | 0.0291 | 0.0646 |

## Conformal Prediction Calibration

Prediction intervals for uncertainty quantification:

| Confidence Level | Quantile |
| --- | --- |
| 50% | ±0.4267 |
| 60% | ±0.5161 |
| 70% | ±0.5837 |
| 80% | ±0.6317 |
| 90% | ±0.6807 |
| 95% | ±0.7253 |
| 99% | ±0.7814 |

## Prediction Distribution Analysis

| Statistic | Predicted | Actual |
| --- | --- | --- |
| Mean | 0.5066 | 0.6243 |
| Std | 0.0746 | 4.4269 |
| Min | 0.3023 | -44.9348 |
| 25th Percentile | 0.4570 | -1.0680 |
| Median | 0.5086 | 0.6595 |
| 75th Percentile | 0.5544 | 2.4476 |
| Max | 0.7924 | 48.0008 |

**Breakout Prediction Accuracy**: 62.03% (2868 predictions)
**Breakdown Prediction Accuracy**: 60.79% (2387 predictions)

## Confidence Score Analysis

| Confidence Level | Mean Score | Std Score |
| --- | --- | --- |
| 50% | 0.7368 | 0.1004 |
| 60% | 0.7704 | 0.0908 |
| 70% | 0.7905 | 0.0845 |
| 80% | 0.8028 | 0.0806 |
| 90% | 0.8139 | 0.0769 |
| 95% | 0.8229 | 0.0738 |
| 99% | 0.8332 | 0.0703 |
