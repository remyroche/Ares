# Training Report — 20260204_220000
Generated: 2026-02-12 04:03 UTC

## Configuration
- **Train lookback**: 26280 hours
- **Label horizons**: [2, 4, 8]
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
| gamma_model | 2,531,934 | 28 |
| spike_anatomy_best | 8,491 | 12 |
| spike_anatomy_worst | 9,744 | 12 |
| train_long_mr_2 | 1,560 | 1125 |
| train_long_mr_4 | 1,260 | 1125 |
| train_long_mr_8 | 1,432 | 1125 |
| train_long_tf_2 | 1,394 | 1109 |
| train_long_tf_4 | 1,123 | 1109 |
| train_long_tf_8 | 1,292 | 1109 |
| train_short_mr_2 | 1,418 | 1125 |
| train_short_mr_4 | 1,144 | 1125 |
| train_short_mr_8 | 1,277 | 1125 |
| train_short_tf_2 | 1,567 | 1109 |
| train_short_tf_4 | 1,284 | 1109 |
| train_short_tf_8 | 1,427 | 1109 |
| trap_model | 7,872,140 | 11 |

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
- **Top features**: spike_score_G_VOL_0, meta_alignment_G_TREND_1, spike_score_G_TREND_1, prog_def_G_VOL_0, vol_range_shock_G_VOL_0, shannon_entropy_ret_8_G_VOL_1, retrace_12_G_VOL_1, comp_to_exp_G_VOL_0, mfe_4h_G_VOL_1, volume_entropy_24_G_TREND_1


#### LONG_TF
- **Features**: 48
- **Top features**: progress_G_TREND_1, rsi_1h_slope_G_TREND_1, ft_2_G_TREND_1, atr_pct_G_TREND_1, meta_alignment_G_TREND_1, G_LIQ_EXCEL_G_VOL_1, climax_decay_G_TREND_1, range_pct_G_TREND_1, meta_abs_net_x_breakout_G_TREND_1, ret1h_z_G_TREND_1


#### SHORT_MR
- **Features**: 48
- **Top features**: trend_snr_G_TREND_1, excess_12h_G_TREND_1, speed_G_TREND_1, rv_24h_G_TREND_1, spike_score_G_TREND_1, rvol_z_G_TREND_1, rsi_1h_slope_G_TREND_1, mfe_8h_G_TREND_1, vol_z_G_TREND_1, accept_bin3_G_TREND_1


#### SHORT_TF
- **Features**: 48
- **Top features**: meta_alignment_G_TREND_1, volu_z_G_TREND_1, meta_alignment_G_VOL_0, rv_8h_G_VOL_1, amihud_illiq_G_TREND_1, dir_path_short_2h_G_TREND_1, rv_ratio_6_24_G_TREND_1, vol_z_G_VOL_0, cos_hod_G_VOL_1, meta_signal_x_accel_G_TREND_1

## Meta Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Calibration |
|-------|----------|-----|----|---------| ---------|-------------|
| long_mr | 40 | N/A | N/A | N/A | N/A | N/A |
| long_tf | 40 | N/A | N/A | N/A | N/A | N/A |
| short_mr | 40 | N/A | N/A | N/A | N/A | N/A |
| short_tf | 40 | N/A | N/A | N/A | N/A | N/A |

### Detailed Meta Model Performance

#### long_mr
- **Features**: 40

#### long_tf
- **Features**: 40

#### short_mr
- **Features**: 40

#### short_tf
- **Features**: 40
