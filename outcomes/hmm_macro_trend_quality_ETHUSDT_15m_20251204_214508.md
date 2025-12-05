# HMM Macro Trend Quality Report

**Symbol**: ETHUSDT | **Exchange**: binance | **Timeframe**: 15m

**Generated**: 20251204_214508

---

## Global Metrics

| Metric | Value |
|--------|-------|
| n_regimes | 4 |
| total_samples | 31039 |
| mean_run_length_bars | 41.2205 |
| median_run_length_bars | 25.0000 |
| mean_run_length_hours | 10.3051 |
| median_run_length_hours | 6.2500 |
| n_regime_changes | 752 |
| temporal_smoothness | 0.9758 |

### Regime Duration Analysis

- **Mean run length**: 41.2 bars â 10.31h
- **Median run length**: 25.0 bars â 6.25h
- â **Target achieved**: 10-16h regime duration

## Per-Regime Metrics

|   regime_id |   n_samples |   n_runs |   mean_duration_bars |   median_duration_bars |   mean_duration_hours |   median_duration_hours |   score_mean |   score_std |   mean_return_1h |   sharpe_1h |   mean_return_2h |   sharpe_2h |   mean_return_3h |   sharpe_3h |
|------------:|------------:|---------:|---------------------:|-----------------------:|----------------------:|------------------------:|-------------:|------------:|-----------------:|------------:|-----------------:|------------:|-----------------:|------------:|
|           0 |        7759 |      117 |              66.3162 |                     61 |              16.5791  |                   15.25 |     0.432682 |   0.032791  |     -0.000710144 |  -0.0794479 |     -0.00139497  |  -0.106362  |     -0.00209564  | -0.123575   |
|           1 |        8070 |      268 |              30.1119 |                     21 |               7.52799 |                    5.25 |     0.492134 |   0.0275831 |     -0.000216073 |  -0.0264537 |     -0.000420302 |  -0.0352902 |     -0.000580423 | -0.0391615  |
|           2 |        7140 |      259 |              27.5676 |                     19 |               6.89189 |                    4.75 |     0.528976 |   0.0196736 |      0.000202436 |   0.0249994 |      0.00047654  |   0.0401888 |      0.000772512 |  0.0516534  |
|           3 |        8070 |      109 |              74.0367 |                     44 |              18.5092  |                   11    |     0.552858 |   0.014013  |      0.000164109 |   0.0176469 |      0.000216181 |   0.0153937 |      0.000161287 |  0.00882439 |

## Per-Quantile Metrics (0-1 Scalar)

| quantile_range   |   n_samples |   mean_return_1h |   sharpe_1h |   mean_return_2h |   sharpe_2h |   mean_return_3h |   sharpe_3h |
|:-----------------|------------:|-----------------:|------------:|-----------------:|------------:|-----------------:|------------:|
| 0.2-0.4          |        1248 |     -0.000899258 |  -0.0865382 |     -0.00166339  |  -0.109544  |     -0.00297805  |  -0.147942  |
| 0.4-0.6          |       29780 |     -0.000112322 |  -0.0130954 |     -0.000237475 |  -0.0187137 |     -0.000358797 |  -0.0221623 |
| 0.6-0.8          |          11 |     -0.00152217  |  -0.105752  |      0.00723929  |   0.363244  |      0.0249881   |   1.11075   |

## Transition Matrix

Probability of transitioning from row regime to column regime:

|   row_0 |         0 |         1 |         2 |         3 |
|--------:|----------:|----------:|----------:|----------:|
|       0 | 0.984921  | 0.0150793 | 0         | 0         |
|       1 | 0.0144981 | 0.966791  | 0.0187113 | 0         |
|       2 | 0         | 0.0211485 | 0.963725  | 0.0151261 |
|       3 | 0         | 0         | 0.0133846 | 0.986615  |

## Training Metrics Summary

- **oof_r2**: -0.08040103559374856
- **oof_rmse**: 5.619730512607567
- **xgb_oof_windows**: 59
- **xgb_oof_predictions**: 19615
- **ic_pearson_correlation**: 0.5181277245232248
- **ic_spearman_correlation**: 0.4137535652643636
- **macro_trend_model_training_failed**: 0
- **macro_trend_fallback_used**: False

### Alpha Score Distribution (macro_trend_score_continuous)

| Stat | Value |
|------|-------|
| mean | 0.501535 |
| std | 0.051699 |
| min | 0.317729 |
| max | 0.622866 |
| q05 | 0.402393 |
| q50 | 0.515679 |
| q95 | 0.561447 |
| n | 31039 |

### Permutation Importance (Top Features)

| Rank | Feature | Importance (mean) |
|------|---------|-------------------|
| 1 | trend_price_slope | 0.032284 |
| 2 | macro_range_norm_12h | 0.021484 |
| 3 | macro_range_norm_24h | 0.015339 |
| 4 | hl_range | 0.014314 |
| 5 | macro_vwap_3h | 0.013887 |
| 6 | macro_vol_realized_24h | 0.013740 |
| 7 | macro_return_vol_norm_24h | 0.013570 |
| 8 | macro_vol_realized_12h | 0.012550 |
| 9 | trend_ema_slow | 0.011872 |
| 10 | macro_atr_norm_3h | 0.009801 |

