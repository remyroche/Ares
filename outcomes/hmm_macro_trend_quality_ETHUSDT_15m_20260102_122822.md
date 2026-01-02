# HMM Macro Trend Quality Report

**Symbol**: ETHUSDT | **Exchange**: binance | **Timeframe**: 15m

**Generated**: 20260102_122822

---

## Global Metrics

| Metric | Value |
|--------|-------|
| n_regimes | 4 |
| total_samples | 139523 |
| mean_run_length_bars | 40.3013 |
| median_run_length_bars | 23.0000 |
| mean_run_length_hours | 10.0753 |
| median_run_length_hours | 5.7500 |
| n_regime_changes | 3461 |
| temporal_smoothness | 0.9752 |

### Regime Duration Analysis

- **Mean run length**: 40.3 bars â 10.08h
- **Median run length**: 23.0 bars â 5.75h
- â **Target achieved**: 10-16h regime duration

## Per-Regime Metrics

|   regime_id |   n_samples |   n_runs |   mean_duration_bars |   median_duration_bars |   mean_duration_hours |   median_duration_hours |   score_mean |   score_std |   mean_return_1h |   sharpe_1h |   mean_return_2h |   sharpe_2h |   mean_return_3h |   sharpe_3h |
|------------:|------------:|---------:|---------------------:|-----------------------:|----------------------:|------------------------:|-------------:|------------:|-----------------:|------------:|-----------------:|------------:|-----------------:|------------:|
|           0 |       34880 |      536 |              65.0746 |                     45 |              16.2687  |                   11.25 |     0.477033 |  0.0196416  |     -0.000262869 | -0.0386984  |     -0.000532559 | -0.0544813  |     -0.000788066 |  -0.0648287 |
|           1 |       36276 |     1181 |              30.7163 |                     20 |               7.67909 |                    5    |     0.504778 |  0.0102543  |      9.80836e-05 |  0.0156626  |      0.0001927   |  0.0214086  |      0.000288106 |   0.0257587 |
|           2 |       32091 |     1195 |              26.8544 |                     17 |               6.7136  |                    4.25 |     0.514371 |  0.00704788 |      2.33863e-05 |  0.00394716 |      7.52547e-05 |  0.00894507 |      0.000103757 |   0.0100114 |
|           3 |       36276 |      550 |              65.9564 |                     47 |              16.4891  |                   11.75 |     0.52341  |  0.0100422  |      0.000132004 |  0.0201856  |      0.000220372 |  0.0233726  |      0.000300564 |   0.0256851 |

## Per-Quantile Metrics (0-1 Scalar)

| quantile_range   |   n_samples |   mean_return_1h |   sharpe_1h |   mean_return_2h |    sharpe_2h |   mean_return_3h |   sharpe_3h |
|:-----------------|------------:|-----------------:|------------:|-----------------:|-------------:|-----------------:|------------:|
| 0.2-0.4          |          79 |     -0.00465095  | -0.492678   |     -0.00883983  | -0.681088    |      -0.0122106  | -0.764874   |
| 0.4-0.6          |      139397 |      8.45579e-07 |  0.00013224 |     -5.83803e-06 | -0.000635871 |      -1.6608e-05 | -0.00145801 |
| 0.6-0.8          |          29 |      0.00116378  |  0.131493   |      0.00213854  |  0.1584      |       0.00419271 |  0.290479   |
| 0.8-1.0          |          18 |      0.00800573  |  0.81519    |      0.0152259   |  1.37097     |       0.0197033  |  1.62487    |

## Transition Matrix

Probability of transitioning from row regime to column regime:

|   row_0 |        0 |         1 |         2 |         3 |
|--------:|---------:|----------:|----------:|----------:|
|       0 | 0.984633 | 0.015367  | 0         | 0         |
|       1 | 0.014748 | 0.967444  | 0.0178079 | 0         |
|       2 | 0        | 0.0200991 | 0.962762  | 0.0171388 |
|       3 | 0        | 0         | 0.0151344 | 0.984866  |

## Training Metrics Summary

- **oof_r2**: -0.007786253355734907
- **oof_rmse**: 5.227923855196301
- **xgb_oof_windows**: 135
- **xgb_oof_predictions**: 127811
- **ic_pearson_correlation**: 0.34233432952239623
- **ic_spearman_correlation**: 0.27460704250410667
- **macro_trend_model_training_failed**: 0
- **macro_trend_fallback_used**: False

### Alpha Score Distribution (macro_trend_score_continuous)

| Stat | Value |
|------|-------|
| mean | 0.504893 |
| std | 0.021571 |
| min | 0.276020 |
| max | 0.897343 |
| q05 | 0.460856 |
| q50 | 0.512173 |
| q95 | 0.527035 |
| n | 139523 |

### Permutation Importance (Top Features)

| Rank | Feature | Importance (mean) |
|------|---------|-------------------|
| 1 | trend_price_slope | 0.010898 |
| 2 | macro_vol_realized_12h | 0.008257 |
| 3 | macro_return_vol_norm_24h | 0.007067 |
| 4 | macro_support_dist_24h | 0.006462 |
| 5 | trend_ema_fast | 0.005939 |
| 6 | macro_vwap_6h | 0.005418 |
| 7 | macro_atr_norm_24h | 0.004401 |
| 8 | macro_range_norm_3h | 0.004221 |
| 9 | hl_range | 0.004051 |
| 10 | macro_vol_realized_24h | 0.004028 |

