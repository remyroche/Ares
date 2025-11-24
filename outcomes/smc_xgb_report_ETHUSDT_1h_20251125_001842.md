# SMC XGBoost Model Report

- **Symbol**: ETHUSDT
- **Exchange**: binance
- **Timeframe**: 1h
- **Generated**: 20251125_001842

## Model Performance

| Metric | Train | Val | Test |
| --- | --- | --- | --- |
| RMSE | 4.2411 | 4.3133 | 4.9346 |
| R² Score | 0.0199 | -0.0254 | -0.0009 |
| Brier Score | 0.1826 | 0.2396 | 0.2452 |
| Samples | 5000 | 1249 | 1563 |
| Features | 47 | 47 | 47 |

**Mean Return (Breakout Signals)**: 0.0131 (2797 signals)
**Mean Return (Breakdown Signals)**: -0.0114 (2493 signals)

## Top 15 Features by Importance

| Rank | Feature | Importance | Correlation |
| --- | --- | --- | --- |
| 22 | smc_rel_volume | 0.0486 | -0.0004 |
| 16 | smc_displacement_strength | 0.0447 | 0.0021 |
| 4 | smc_dist_to_week_open | 0.0385 | 0.0469 |
| 30 | smc_profile_skew | 0.0378 | 0.0142 |
| 36 | smc_dow_Mon | 0.0376 | -0.0119 |
| 21 | smc_adr_filled_pct | 0.0362 | 0.0011 |
| 3 | smc_dist_to_day_open | 0.0338 | 0.0257 |
| 25 | smc_htf_trend_slope | 0.0333 | 0.0056 |
| 5 | smc_nwog_gap_size | 0.0315 | -0.0524 |
| 28 | smc_poc_dist_atr | 0.0315 | -0.0305 |
| 42 | smc_dow_Sun | 0.0314 | -0.0403 |
| 2 | smc_dist_to_pdl_atr | 0.0309 | 0.0125 |
| 18 | smc_close_position_in_candle | 0.0295 | 0.0646 |
| 20 | smc_consecutive_candles | 0.0291 | 0.0745 |
| 26 | smc_daily_wick_rejection | 0.0278 | 0.0087 |

## Conformal Prediction Calibration

Prediction intervals for uncertainty quantification:

| Confidence Level | Quantile |
| --- | --- |
| 50% | ±0.4351 |
| 60% | ±0.5113 |
| 70% | ±0.5820 |
| 80% | ±0.6336 |
| 90% | ±0.6864 |
| 95% | ±0.7389 |
| 99% | ±0.7975 |

## Prediction Distribution Analysis

| Statistic | Predicted | Actual |
| --- | --- | --- |
| Mean | 0.5065 | 0.6243 |
| Std | 0.0876 | 4.4269 |
| Min | 0.2634 | -44.9348 |
| 25th Percentile | 0.4466 | -1.0680 |
| Median | 0.5095 | 0.6595 |
| 75th Percentile | 0.5647 | 2.4476 |
| Max | 0.8295 | 48.0008 |

## Scalar Band Performance (Deciles)

| Band | Count | Mean Scalar | Mean Forward Return | Directional Hit Rate |
| --- | --- | --- | --- | --- |
| (0.262, 0.391] | 782 | 0.3549 | -0.0252 | 77.49% |
| (0.391, 0.432] | 781 | 0.4134 | -0.0123 | 71.32% |
| (0.432, 0.46] | 781 | 0.4465 | -0.0057 | 61.72% |
| (0.46, 0.485] | 781 | 0.4721 | -0.0012 | 55.44% |
| (0.485, 0.51] | 781 | 0.4972 | 0.0007 | 49.68% |
| (0.51, 0.53] | 781 | 0.5201 | 0.0052 | 56.47% |
| (0.53, 0.553] | 781 | 0.5416 | 0.0089 | 64.79% |
| (0.553, 0.577] | 781 | 0.5648 | 0.0093 | 65.17% |
| (0.577, 0.611] | 781 | 0.5923 | 0.0153 | 72.34% |
| (0.611, 0.829] | 782 | 0.6620 | 0.0188 | 84.40% |

**Breakout Prediction Accuracy**: 63.03% (2797 predictions)
**Breakdown Prediction Accuracy**: 60.73% (2493 predictions)

## Confidence Score Analysis

| Confidence Level | Mean Score | Std Score |
| --- | --- | --- |
| 50% | 0.7405 | 0.1110 |
| 60% | 0.7685 | 0.1017 |
| 70% | 0.7896 | 0.0944 |
| 80% | 0.8026 | 0.0896 |
| 90% | 0.8144 | 0.0852 |
| 95% | 0.8248 | 0.0812 |
| 99% | 0.8351 | 0.0771 |
