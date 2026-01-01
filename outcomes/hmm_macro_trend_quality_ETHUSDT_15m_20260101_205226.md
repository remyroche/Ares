# HMM Macro Trend Quality Report

**Symbol**: ETHUSDT | **Exchange**: binance | **Timeframe**: 15m

**Generated**: 20260101_205226

---

## Global Metrics

| Metric | Value |
|--------|-------|
| n_regimes | 3 |
| total_samples | 139523 |
| mean_run_length_bars | 56.9249 |
| median_run_length_bars | 36.0000 |
| mean_run_length_hours | 14.2312 |
| median_run_length_hours | 9.0000 |
| n_regime_changes | 2450 |
| temporal_smoothness | 0.9824 |

### Regime Duration Analysis

- **Mean run length**: 56.9 bars â 14.23h
- **Median run length**: 36.0 bars â 9.00h
- â **Target achieved**: 10-16h regime duration

## Per-Regime Metrics

|   regime_id |   n_samples |   n_runs |   mean_duration_bars |   median_duration_bars |   mean_duration_hours |   median_duration_hours |   score_mean |   score_std |   mean_return_1h |   sharpe_1h |   mean_return_2h |   sharpe_2h |   mean_return_3h |   sharpe_3h |
|------------:|------------:|---------:|---------------------:|-----------------------:|----------------------:|------------------------:|-------------:|------------:|-----------------:|------------:|-----------------:|------------:|-----------------:|------------:|
|           0 |       46507 |      601 |              77.3827 |                     54 |              19.3457  |                   13.5  |     0.484459 |  0.017304   |     -0.000153273 |  -0.0229048 |     -0.000312573 |  -0.0325234 |     -0.000475252 |  -0.0398083 |
|           1 |       43718 |     1225 |              35.6882 |                     24 |               8.92204 |                    6    |     0.507944 |  0.00712472 |      6.96906e-05 |   0.0113997 |      0.000126545 |   0.0144174 |      0.000198066 |   0.018191  |
|           2 |       49298 |      625 |              78.8768 |                     57 |              19.7192  |                   14.25 |     0.521472 |  0.00769915 |      8.13389e-05 |   0.0127856 |      0.000158799 |   0.0173894 |      0.000215831 |   0.0190787 |

## Per-Quantile Metrics (0-1 Scalar)

| quantile_range   |   n_samples |   mean_return_1h |    sharpe_1h |   mean_return_2h |   sharpe_2h |   mean_return_3h |   sharpe_3h |
|:-----------------|------------:|-----------------:|-------------:|-----------------:|------------:|-----------------:|------------:|
| 0.4-0.6          |      139502 |     -1.37225e-06 | -0.000214472 |     -1.02506e-05 |  -0.0011157 |       -2.241e-05 | -0.00196596 |
| 0.6-0.8          |          21 |      0.00570147  |  0.651108    |      0.0120911   |   1.09535   |        0.0153688 |  1.24426    |

## Transition Matrix

Probability of transitioning from row regime to column regime:

|   row_0 |         0 |         1 |         2 |
|--------:|----------:|----------:|----------:|
|       0 | 0.987077  | 0.0129228 | 0         |
|       1 | 0.0137243 | 0.97198   | 0.0142962 |
|       2 | 0         | 0.012658  | 0.987342  |

## Training Metrics Summary

- **oof_r2**: -0.014836008496843966
- **oof_rmse**: 5.246177405268942
- **xgb_oof_windows**: 135
- **xgb_oof_predictions**: 127811
- **ic_pearson_correlation**: 0.3119910903165905
- **ic_spearman_correlation**: 0.24541676719436376
- **macro_trend_model_training_failed**: 0
- **macro_trend_fallback_used**: False

### Alpha Score Distribution (macro_trend_score_continuous)

| Stat | Value |
|------|-------|
| mean | 0.504896 |
| std | 0.019387 |
| min | 0.404375 |
| max | 0.645984 |
| q05 | 0.464754 |
| q50 | 0.509757 |
| q95 | 0.529389 |
| n | 139523 |

### Permutation Importance (Top Features)

| Rank | Feature | Importance (mean) |
|------|---------|-------------------|
| 1 | trend_price_slope | 0.011360 |
| 2 | macro_vol_realized_12h | 0.009225 |
| 3 | macro_support_dist_24h | 0.008915 |
| 4 | macro_return_vol_norm_24h | 0.007373 |
| 5 | macro_range_norm_3h | 0.005886 |
| 6 | trend_ema_fast | 0.005507 |
| 7 | macro_vwap_6h | 0.005406 |
| 8 | trend_ema_slow | 0.005102 |
| 9 | macro_atr_norm_24h | 0.005085 |
| 10 | macro_return_24h | 0.003797 |

