# Training Report — 20260212_190000
Generated: 2026-02-12 20:21 UTC

## Configuration
- **Train lookback**: 26280 hours
- **Timeframe**: 1h (signals are made on the last closed bar)
- **Label horizons**: [2, 4, 8]
- **Prediction horizons in bars**: 2h → 2 bars, 4h → 4 bars, 8h → 8 bars
- **Label method**: triple_barrier
- **Label quantiles**: lo=0.3, hi=0.7
- **OOS holdout**: 180 days
- **Min train samples**: 200
- **Feature selection**: MDI (min=30, cap=0.995)
- **15m precision**: True

## Dataset Sizes
| Dataset | Rows | Features |
|---------|------|----------|
| exh_down | 128,510 | 34 |
| exh_up | 28,532 | 34 |
| gamma_model | 2,373,695 | 28 |
| spike_anatomy_best | 10,285 | 12 |
| spike_anatomy_worst | 11,826 | 12 |
| train_long_mr_2 | 2,560 | 614 |
| train_long_mr_4 | 2,560 | 614 |
| train_long_mr_8 | 2,560 | 614 |
| train_long_tf_2 | 1,991 | 606 |
| train_long_tf_4 | 1,991 | 606 |
| train_long_tf_8 | 1,991 | 606 |
| train_short_mr_2 | 1,991 | 614 |
| train_short_mr_4 | 1,991 | 614 |
| train_short_mr_8 | 1,991 | 614 |
| train_short_tf_2 | 2,560 | 606 |
| train_short_tf_4 | 2,560 | 606 |
| train_short_tf_8 | 2,560 | 606 |
| trap_model | 8,688,943 | 11 |

## Alpha Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Prec@10 | Prec@40 | AvgTr/day@10 | AvgTr/day@30 | Avg Return | Trades | Best Iter |
|-------|----------|-----|----|---------| ---------|---------|---------|-------------|-------------|------------|--------|-----------|
| LONG_MR | 48 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| LONG_TF | 48 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| SHORT_MR | 48 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| SHORT_TF | 48 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### Detailed Model Performance

#### LONG_MR
- **Features**: 48
- **Top features**: mfe_4h_G_VOL_1, ft_8_G_VOL_1, retrace_12_G_VOL_1, dist_vwap_norm_G_VOL_1, mfe_8h_G_VOL_1, mfe_2h_G_VOL_1, accel_5h_G_VOL_1, G_EXH_GIVEBACK_G_VOL_1, sin_dow_G_VOL_1, meta_alignment_G_VOL_1


#### LONG_TF
- **Features**: 48
- **Top features**: stall_ext_corr_G_VOL_0, dist_ema_slow_base_G_VOL_0, G_LIQ_EXCEL_G_VOL_0, ft_4_G_VOL_0, breakout_confirmed_G_VOL_0, vov_mad_20_G_VOL_1, sin_hod_G_VOL_1, ft_2_G_VOL_0, spike_score_G_VOL_0, asym_ft_G_VOL_0


#### SHORT_MR
- **Features**: 48
- **Top features**: mae_4h_G_VOL_0, meta_abs_net_x_breakout_G_VOL_0, rsi_1h_slope_G_VOL_0, clv_mean_4_G_VOL_0, mfe_8h_G_VOL_0, accel_5h_G_VOL_0, mae_4h_G_VOL_1, clv_mean_2_G_VOL_0, momentum_accel_G_VOL_0, G_TF_GRIND_G_VOL_0


#### SHORT_TF
- **Features**: 48
- **Top features**: meta_signal_x_accel_G_VOL_1, cos_hod_G_VOL_1, shannon_entropy_ret_16_G_VOL_1, evr_6_G_VOL_1, accel_5h_G_VOL_1, accept_dir2h_abs_prod_G_VOL_1, rsi_lag1_G_VOL_1, decel_8_G_VOL_1, dir_path_long_2h_G_VOL_1, dir_path_risk_long_2h_G_VOL_1

## Meta Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Calibration |
|-------|----------|-----|----|---------| ---------|-------------|
| long_mr | 30 | N/A | N/A | N/A | N/A | N/A |
| long_tf | 40 | N/A | N/A | N/A | N/A | N/A |
| short_mr | 30 | N/A | N/A | N/A | N/A | N/A |
| short_tf | 40 | N/A | N/A | N/A | N/A | N/A |

### Detailed Meta Model Performance

#### long_mr
- **Features**: 30

#### long_tf
- **Features**: 40

#### short_mr
- **Features**: 30

#### short_tf
- **Features**: 40
