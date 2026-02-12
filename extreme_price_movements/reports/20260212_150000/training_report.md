# Training Report — 20260212_150000
Generated: 2026-02-12 16:22 UTC

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
| spike_anatomy_best | 11,325 | 12 |
| spike_anatomy_worst | 13,157 | 12 |
| train_long_mr_2 | 2,372 | 614 |
| train_long_mr_4 | 2,372 | 614 |
| train_long_mr_8 | 2,372 | 614 |
| train_long_tf_2 | 1,895 | 606 |
| train_long_tf_4 | 1,895 | 606 |
| train_long_tf_8 | 1,895 | 606 |
| train_short_mr_2 | 1,895 | 614 |
| train_short_mr_4 | 1,895 | 614 |
| train_short_mr_8 | 1,895 | 614 |
| train_short_tf_2 | 2,372 | 606 |
| train_short_tf_4 | 2,372 | 606 |
| train_short_tf_8 | 2,372 | 606 |
| trap_model | 8,688,943 | 11 |

## Alpha Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Prec@10 | Prec@40 | AvgTr/day@10 | AvgTr/day@30 | Avg Return | Trades | Best Iter |
|-------|----------|-----|----|---------| ---------|---------|---------|-------------|-------------|------------|--------|-----------|
| LONG_MR | 48 | 0.5350 | 0.0530 | -0.0461 | nan% | 0.4655 | 0.4425 | 2.8005 | 3.2898 | nan | N/A | N/A |
| LONG_TF | 48 | 0.5367 | 0.0445 | -0.0372 | nan% | 0.4966 | 0.4431 | 2.6566 | 2.7309 | nan | N/A | N/A |
| SHORT_MR | 48 | 0.5719 | 0.1365 | -0.1658 | nan% | 0.5124 | 0.4284 | 2.6566 | 2.7309 | nan | N/A | N/A |
| SHORT_TF | 48 | 0.5423 | 0.1449 | -0.1538 | nan% | 0.4156 | 0.4616 | 2.8005 | 3.2898 | nan | N/A | N/A |

### Detailed Model Performance

#### LONG_MR
- **Features**: 48
- **OOF AUC**: 0.5350
- **OOF IC**: 0.0530
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.0461
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.4655
- **OOF Prec@40**: 0.4425
- **OOF Avg Trades/Day @10%**: 2.8005
- **OOF Avg Trades/Day @30%**: 3.2898
- **OOF ECE@10**: 0.1034
- **OOF Calibration Profile**: overconfident

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | -0.1602 / 0.2382 / 0.5026 / 790 | -0.0953 / 0.2147 / 0.5621 / 792 | -0.0888 / 0.2458 / 0.5479 / 790 |
| vol_48h | -0.1130 / 0.2242 / 0.5502 / 790 | -0.0816 / 0.2252 / 0.5602 / 792 | -0.1358 / 0.2493 / 0.5009 / 790 |
| volume_12h | -0.1426 / 0.2250 / 0.5309 / 790 | -0.1055 / 0.2290 / 0.5336 / 792 | -0.0936 / 0.2447 / 0.5464 / 790 |
| volume_48h | -0.1321 / 0.2312 / 0.5217 / 790 | -0.1094 / 0.2338 / 0.5346 / 792 | -0.0874 / 0.2337 / 0.5505 / 790 |
| trend_12h | -0.1217 / 0.2609 / 0.5183 / 804 | -0.1073 / 0.2240 / 0.5464 / 778 | -0.1387 / 0.2131 / 0.5407 / 790 |
| trend_48h | -0.1313 / 0.2558 / 0.5079 / 790 | -0.0873 / 0.2258 / 0.5595 / 792 | -0.1262 / 0.2171 / 0.5492 / 790 |

- **Top features**: mfe_4h_G_VOL_1, meta_alignment_G_VOL_0, ft_8_G_VOL_1, excess_coh_G_VOL_0, clv_mean_24_G_VOL_0, dist_stack_G_VOL_0, mfe_8h_G_VOL_1, retrace_12_G_VOL_1, meta_alignment_G_VOL_1, breakout_confirmed_G_VOL_0


#### LONG_TF
- **Features**: 48
- **OOF AUC**: 0.5367
- **OOF IC**: 0.0445
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.0372
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.4966
- **OOF Prec@40**: 0.4431
- **OOF Avg Trades/Day @10%**: 2.6566
- **OOF Avg Trades/Day @30%**: 2.7309
- **OOF ECE@10**: 0.0446
- **OOF Calibration Profile**: overconfident

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | -0.0521 / 0.2227 / 0.5612 / 631 | -0.0818 / 0.2279 / 0.5515 / 633 | -0.1333 / 0.2356 / 0.4959 / 631 |
| vol_48h | -0.0523 / 0.2272 / 0.5571 / 631 | -0.1019 / 0.2264 / 0.5503 / 557 | -0.1139 / 0.2319 / 0.5059 / 707 |
| volume_12h | -0.0887 / 0.2167 / 0.5562 / 631 | -0.0764 / 0.2409 / 0.5202 / 633 | -0.1142 / 0.2287 / 0.5369 / 631 |
| volume_48h | -0.0928 / 0.2265 / 0.5347 / 631 | -0.1130 / 0.2210 / 0.5325 / 633 | -0.0728 / 0.2388 / 0.5357 / 631 |
| trend_12h | -0.1096 / 0.2151 / 0.5344 / 631 | -0.0668 / 0.2336 / 0.5464 / 633 | -0.1031 / 0.2375 / 0.5226 / 631 |
| trend_48h | -0.0933 / 0.2176 / 0.5411 / 631 | -0.0479 / 0.2306 / 0.5578 / 633 | -0.1346 / 0.2380 / 0.5060 / 631 |

- **Top features**: vol_z24_base_G_VOL_1, cvar_5pct_G_VOL_1, rv_12h_G_VOL_0, rv_ratio_6_24_G_VOL_1, dist_ema_slow_base_G_VOL_0, atr_pct_change_G_VOL_1, rsi_slope_G_VOL_1, evr_6_G_VOL_0, G_META_MR_QUAL_G_VOL_1, tfq_dir2h_prod_G_VOL_1


#### SHORT_MR
- **Features**: 48
- **OOF AUC**: 0.5719
- **OOF IC**: 0.1365
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.1658
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.5124
- **OOF Prec@40**: 0.4284
- **OOF Avg Trades/Day @10%**: 2.6566
- **OOF Avg Trades/Day @30%**: 2.7309
- **OOF ECE@10**: 0.0889
- **OOF Calibration Profile**: overconfident

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0009 / 0.2269 / 0.5888 / 631 | -0.0598 / 0.2226 / 0.5548 / 633 | -0.0958 / 0.2065 / 0.5431 / 631 |
| vol_48h | -0.0367 / 0.2296 / 0.5591 / 631 | -0.0185 / 0.2160 / 0.5926 / 557 | -0.0740 / 0.2111 / 0.5552 / 707 |
| volume_12h | -0.0362 / 0.2348 / 0.5557 / 631 | -0.0435 / 0.2152 / 0.5726 / 633 | -0.0631 / 0.2061 / 0.5706 / 631 |
| volume_48h | -0.0431 / 0.2383 / 0.5533 / 631 | -0.0419 / 0.2175 / 0.5632 / 633 | -0.0666 / 0.2002 / 0.5799 / 631 |
| trend_12h | 0.0029 / 0.2230 / 0.5922 / 631 | -0.0553 / 0.2275 / 0.5389 / 633 | -0.1001 / 0.2056 / 0.5713 / 631 |
| trend_48h | -0.0118 / 0.2257 / 0.5755 / 631 | -0.0528 / 0.2225 / 0.5619 / 633 | -0.0764 / 0.2079 / 0.5784 / 631 |

- **Top features**: meta_abs_net_x_breakout_G_VOL_0, mae_4h_G_VOL_0, mae_4h_G_VOL_1, mae_2h_G_VOL_1, body_ratio_G_VOL_0, mfe_8h_G_VOL_0, mae_2h_G_VOL_0, evr_slope_G_VOL_0, mae_8h_G_VOL_1, rsi_1h_slope_G_VOL_0


#### SHORT_TF
- **Features**: 48
- **OOF AUC**: 0.5423
- **OOF IC**: 0.1449
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.1538
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.4156
- **OOF Prec@40**: 0.4616
- **OOF Avg Trades/Day @10%**: 2.8005
- **OOF Avg Trades/Day @30%**: 3.2898
- **OOF ECE@10**: 0.0879
- **OOF Calibration Profile**: overconfident

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | -0.0433 / 0.2281 / 0.5168 / 790 | -0.0509 / 0.2275 / 0.4988 / 792 | -0.0217 / 0.1969 / 0.5908 / 790 |
| vol_48h | -0.0485 / 0.2363 / 0.4895 / 790 | -0.0506 / 0.2119 / 0.5320 / 792 | -0.0197 / 0.2043 / 0.5858 / 790 |
| volume_12h | -0.0341 / 0.2265 / 0.5186 / 790 | -0.0643 / 0.2258 / 0.5018 / 792 | -0.0135 / 0.2002 / 0.5893 / 790 |
| volume_48h | -0.0552 / 0.2172 / 0.5256 / 790 | -0.0503 / 0.2170 / 0.5228 / 792 | -0.0051 / 0.2183 / 0.5800 / 790 |
| trend_12h | -0.0673 / 0.1891 / 0.5742 / 804 | -0.0177 / 0.2233 / 0.5390 / 778 | -0.0627 / 0.2407 / 0.4886 / 790 |
| trend_48h | -0.0390 / 0.1965 / 0.5768 / 790 | -0.0447 / 0.2222 / 0.5253 / 792 | -0.0414 / 0.2339 / 0.4958 / 790 |

- **Top features**: pullback_8_G_VOL_1, cos_hod_G_VOL_1, accel_5h_G_VOL_1, meta_signal_x_accel_G_VOL_1, rv_8h_G_VOL_1, dir_path_long_2h_G_VOL_1, ret8h_G_VOL_1, decel_8_G_VOL_1, rsi_lag1_G_VOL_1, trend_pct_base_G_VOL_1

## Specialist Models
- **Trap (GMM)**: 8 features, clusters=GaussianMixture(covariance_type='diag', max_iter=200, n_components=4, n_init=3,
                random_state=42)
- **Gamma (ExtraTrees)**: 20 features
