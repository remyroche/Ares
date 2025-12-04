# HMM Macro Trend Quality Report

**Symbol**: ETHUSDT | **Exchange**: binance | **Timeframe**: 15m

**Generated**: 20251203_221408

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

 regime_id  n_samples  n_runs  mean_duration_bars  median_duration_bars  mean_duration_hours  median_duration_hours  score_mean  score_std  mean_return_1h  sharpe_1h  mean_return_2h  sharpe_2h  mean_return_3h  sharpe_3h
         0       8999     129           69.759690                  23.0            17.439922                   5.75    0.481989   0.183922       -0.001298  -0.123745       -0.001611  -0.105356       -0.001886  -0.096851
         1       6793     249           27.281124                  16.0             6.820281                   4.00    0.500448   0.125443       -0.000245  -0.034283       -0.000440  -0.040608       -0.000528  -0.037727
         2      10590     219           48.356164                  26.0            12.089041                   6.50    0.510022   0.136878        0.000301   0.038532        0.000297   0.024787        0.000132   0.008471
         3       4657      92           50.619565                  23.0            12.654891                   5.75    0.526120   0.146202        0.001219   0.146180        0.001132   0.096513        0.001081   0.075016

## Per-Quantile Metrics (0-1 Scalar)

quantile_range  n_samples  mean_return_1h  sharpe_1h  mean_return_2h  sharpe_2h  mean_return_3h  sharpe_3h
       0.0-0.2       1118       -0.023977  -5.856238       -0.025246  -1.832995       -0.026757  -1.272328
       0.2-0.4       4368       -0.009888  -3.245710       -0.010100  -0.916034       -0.010211  -0.653384
       0.4-0.6      19737       -0.000129  -0.044209       -0.000274  -0.031445       -0.000465  -0.036845
       0.6-0.8       4767        0.009115   3.110337        0.008888   0.896248        0.008867   0.640476
       0.8-1.0       1049        0.023453   5.626476        0.025090   1.906578        0.026027   1.383884

## Transition Matrix

Probability of transitioning from row regime to column regime:

col_0         0         1         2         3
row_0                                        
0      0.985665  0.014113  0.000222  0.000000
1      0.018110  0.963486  0.018404  0.000000
2      0.000567  0.011520  0.979320  0.008593
3      0.000000  0.000000  0.019755  0.980245

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

