# HMM Macro Trend Quality Report

**Symbol**: ETHUSDT | **Exchange**: binance | **Timeframe**: 15m

**Generated**: 20251203_234100

---

## Global Metrics

| Metric | Value |
|--------|-------|
| n_regimes | 4 |
| total_samples | 31039 |
| mean_run_length_bars | 45.0493 |
| median_run_length_bars | 21.0000 |
| mean_run_length_hours | 11.2623 |
| median_run_length_hours | 5.2500 |
| n_regime_changes | 688 |
| temporal_smoothness | 0.9778 |

### Regime Duration Analysis

- **Mean run length**: 45.0 bars â 11.26h
- **Median run length**: 21.0 bars â 5.25h
- â **Target achieved**: 10-16h regime duration

## Per-Regime Metrics

|   regime_id |   n_samples |   n_runs |   mean_duration_bars |   median_duration_bars |   mean_duration_hours |   median_duration_hours |   score_mean |   score_std |   mean_return_1h |   sharpe_1h |   mean_return_2h |   sharpe_2h |   mean_return_3h |   sharpe_3h |
|------------:|------------:|---------:|---------------------:|-----------------------:|----------------------:|------------------------:|-------------:|------------:|-----------------:|------------:|-----------------:|------------:|-----------------:|------------:|
|           0 |        8999 |      129 |              69.7597 |                     23 |              17.4399  |                    5.75 |     0.481989 |    0.183922 |     -0.00129767  |  -0.123745  |     -0.00161113  |  -0.105356  |     -0.00188618  | -0.0968506  |
|           1 |        6793 |      249 |              27.2811 |                     16 |               6.82028 |                    4    |     0.500448 |    0.125443 |     -0.000245205 |  -0.034283  |     -0.00043963  |  -0.0406079 |     -0.000528228 | -0.0377274  |
|           2 |       10590 |      219 |              48.3562 |                     26 |              12.089   |                    6.5  |     0.510022 |    0.136878 |      0.000300721 |   0.0385321 |      0.000297013 |   0.0247873 |      0.000132402 |  0.00847073 |
|           3 |        4657 |       92 |              50.6196 |                     23 |              12.6549  |                    5.75 |     0.52612  |    0.146202 |      0.00121856  |   0.14618   |      0.00113191  |   0.0965132 |      0.00108077  |  0.0750165  |

## Per-Quantile Metrics (0-1 Scalar)

| quantile_range   |   n_samples |   mean_return_1h |   sharpe_1h |   mean_return_2h |   sharpe_2h |   mean_return_3h |   sharpe_3h |
|:-----------------|------------:|-----------------:|------------:|-----------------:|------------:|-----------------:|------------:|
| 0.0-0.2          |        1118 |     -0.0239771   |  -5.85624   |     -0.0252456   |  -1.833     |     -0.0267575   |  -1.27233   |
| 0.2-0.4          |        4368 |     -0.00988773  |  -3.24571   |     -0.0100997   |  -0.916034  |     -0.010211    |  -0.653384  |
| 0.4-0.6          |       19737 |     -0.000128741 |  -0.0442094 |     -0.000274425 |  -0.0314446 |     -0.000465257 |  -0.0368453 |
| 0.6-0.8          |        4767 |      0.00911495  |   3.11034   |      0.00888782  |   0.896248  |      0.00886723  |   0.640476  |
| 0.8-1.0          |        1049 |      0.0234529   |   5.62648   |      0.0250905   |   1.90658   |      0.0260271   |   1.38388   |

## Transition Matrix

Probability of transitioning from row regime to column regime:

|   row_0 |           0 |         1 |           2 |          3 |
|--------:|------------:|----------:|------------:|-----------:|
|       0 | 0.985665    | 0.0141127 | 0.000222247 | 0          |
|       1 | 0.0181095   | 0.963486  | 0.018404    | 0          |
|       2 | 0.000566572 | 0.0115203 | 0.97932     | 0.00859301 |
|       3 | 0           | 0         | 0.0197552   | 0.980245   |

## Training Metrics Summary

- **macro_trend_model_training_failed**: 1
- **macro_trend_fallback_used**: True

### Alpha Score Distribution (macro_trend_score_continuous)

| Stat | Value |
|------|-------|
| mean | 0.502214 |
| std | 0.151902 |
| min | 0.000000 |
| max | 1.000000 |
| q05 | 0.247123 |
| q50 | 0.504279 |
| q95 | 0.741767 |
| n | 31039 |

