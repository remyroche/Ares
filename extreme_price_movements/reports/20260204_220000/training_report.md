# Training Report — 20260204_220000
Generated: 2026-02-11 01:55 UTC

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
| exh_down | 127,633 | 34 |
| exh_up | 28,061 | 34 |
| gamma_model | 2,506,106 | 28 |
| spike_anatomy_best | 8,832 | 12 |
| spike_anatomy_worst | 9,948 | 12 |
| train_long_mr_2 | 1,562 | 853 |
| train_long_mr_4 | 1,617 | 853 |
| train_long_mr_8 | 1,448 | 853 |
| train_long_tf_2 | 1,493 | 837 |
| train_long_tf_4 | 1,544 | 837 |
| train_long_tf_8 | 1,420 | 837 |
| train_short_mr_2 | 1,515 | 853 |
| train_short_mr_4 | 1,520 | 853 |
| train_short_mr_8 | 1,417 | 853 |
| train_short_tf_2 | 1,572 | 837 |
| train_short_tf_4 | 1,628 | 837 |
| train_short_tf_8 | 1,451 | 837 |
| trap_model | 7,848,324 | 11 |

## Alpha Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Prec@10 | Prec@40 | AvgTr/day@10 | AvgTr/day@30 | Avg Return | Trades | Best Iter |
|-------|----------|-----|----|---------| ---------|---------|---------|-------------|-------------|------------|--------|-----------|
| LONG_MR | 48 | 0.5780 | 0.1035 | 0.0880 | nan% | 0.6560 | 0.5738 | 2.3130 | 2.5623 | nan | N/A | N/A |
| LONG_TF | 48 | 0.5469 | 0.1828 | 0.0642 | nan% | 0.5918 | 0.5573 | 2.3880 | 2.4191 | nan | N/A | N/A |
| SHORT_MR | 48 | 0.5945 | 0.1346 | -0.1507 | nan% | 0.5518 | 0.4722 | 2.4690 | 2.5155 | nan | N/A | N/A |
| SHORT_TF | 48 | 0.6302 | 0.2288 | 0.0809 | nan% | 0.7834 | 0.6418 | 2.3690 | 2.6394 | nan | N/A | N/A |

### Detailed Model Performance

#### LONG_MR
- **Features**: 48
- **OOF AUC**: 0.5780
- **OOF IC**: 0.1035
- **OOF Rank IC**: nan
- **OOF Sharpe**: 0.0880
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.6560
- **OOF Prec@40**: 0.5738
- **OOF Avg Trades/Day @10%**: 2.3130
- **OOF Avg Trades/Day @30%**: 2.5623
- **OOF ECE@10**: 0.0634
- **OOF Calibration Profile**: overconfident

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0149 / 0.2427 / 0.5570 / 482 | 0.0253 / 0.2363 / 0.5854 / 484 | 0.0322 / 0.2403 / 0.5978 / 482 |
| vol_48h | 0.0428 / 0.2313 / 0.6171 / 482 | 0.0211 / 0.2435 / 0.5677 / 484 | 0.0068 / 0.2445 / 0.5474 / 482 |
| volume_12h | 0.0200 / 0.2412 / 0.5732 / 482 | 0.0147 / 0.2415 / 0.5545 / 484 | 0.0414 / 0.2366 / 0.6124 / 482 |
| volume_48h | 0.0192 / 0.2435 / 0.5714 / 482 | 0.0123 / 0.2431 / 0.5481 / 484 | 0.0416 / 0.2327 / 0.6103 / 482 |
| trend_12h | -0.0068 / 0.2517 / 0.5563 / 515 | 0.0407 / 0.2342 / 0.6151 / 451 | 0.0219 / 0.2322 / 0.5926 / 482 |
| trend_48h | -0.0053 / 0.2513 / 0.5627 / 482 | 0.0415 / 0.2346 / 0.6215 / 484 | 0.0067 / 0.2334 / 0.5811 / 482 |

- **Top features**: meta_alignment_G_TREND_1, impulse_ratio_24_G_TREND_1, retrace_12_G_TREND_1, dist_ema_fast_base_G_TREND_1, tf_bias_G_VOL_0, spike_score_G_TREND_1, mfe_4h_G_VOL_1, accel_5h_G_TREND_1, retrace_12_G_VOL_1, pullback_4_G_TREND_1


#### LONG_TF
- **Features**: 48
- **OOF AUC**: 0.5469
- **OOF IC**: 0.1828
- **OOF Rank IC**: nan
- **OOF Sharpe**: 0.0642
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.5918
- **OOF Prec@40**: 0.5573
- **OOF Avg Trades/Day @10%**: 2.3880
- **OOF Avg Trades/Day @30%**: 2.4191
- **OOF ECE@10**: 0.1361
- **OOF Calibration Profile**: underconfident/conservative

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0069 / 0.2391 / 0.5393 / 473 | -0.0032 / 0.2414 / 0.5234 / 474 | 0.0121 / 0.2116 / 0.5463 / 473 |
| vol_48h | 0.0040 / 0.2406 / 0.5297 / 473 | 0.0034 / 0.2348 / 0.5325 / 384 | 0.0180 / 0.2196 / 0.5536 / 563 |
| volume_12h | 0.0062 / 0.2371 / 0.5366 / 473 | -0.0093 / 0.2458 / 0.5178 / 474 | 0.0126 / 0.2091 / 0.5555 / 473 |
| volume_48h | -0.0082 / 0.2371 / 0.5156 / 473 | 0.0177 / 0.2319 / 0.5549 / 474 | 0.0286 / 0.2231 / 0.5724 / 473 |
| trend_12h | 0.0051 / 0.2365 / 0.5273 / 473 | 0.0186 / 0.2296 / 0.5571 / 450 | 0.0136 / 0.2262 / 0.5528 / 497 |
| trend_48h | 0.0039 / 0.2394 / 0.5298 / 473 | 0.0087 / 0.2389 / 0.5428 / 474 | 0.0020 / 0.2138 / 0.5405 / 473 |

- **Top features**: momentum_accel_G_TREND_1, G_LIQ_EXCEL_G_TREND_1, atr_pct_base_G_TREND_1, range_pct_G_TREND_1, rv_6h_G_VOL_0, ft_2_G_TREND_1, prog_def_G_TREND_1, body_pct_G_TREND_1, evr_6_G_TREND_1, meta_alignment_G_TREND_1


#### SHORT_MR
- **Features**: 48
- **OOF AUC**: 0.5945
- **OOF IC**: 0.1346
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.1507
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.5518
- **OOF Prec@40**: 0.4722
- **OOF Avg Trades/Day @10%**: 2.4690
- **OOF Avg Trades/Day @30%**: 2.5155
- **OOF ECE@10**: 0.0248
- **OOF Calibration Profile**: well-calibrated

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0384 / 0.2342 / 0.6146 / 505 | 0.0146 / 0.2213 / 0.5528 / 505 | 0.0121 / 0.2022 / 0.5673 / 505 |
| vol_48h | 0.0416 / 0.2311 / 0.6115 / 505 | 0.0125 / 0.2281 / 0.5584 / 395 | 0.0186 / 0.2038 / 0.5765 / 615 |
| volume_12h | 0.0194 / 0.2403 / 0.5792 / 505 | 0.0098 / 0.2230 / 0.5442 / 505 | 0.0253 / 0.1944 / 0.5985 / 505 |
| volume_48h | 0.0290 / 0.2371 / 0.6027 / 505 | 0.0040 / 0.2267 / 0.5402 / 505 | 0.0235 / 0.1939 / 0.5966 / 505 |
| trend_12h | 0.0241 / 0.2386 / 0.6022 / 505 | 0.0228 / 0.2265 / 0.5673 / 463 | -0.0022 / 0.1952 / 0.5654 / 547 |
| trend_48h | 0.0207 / 0.2382 / 0.5877 / 505 | 0.0246 / 0.2184 / 0.5895 / 505 | 0.0255 / 0.2011 / 0.5787 / 505 |

- **Top features**: trend_snr_G_TREND_1, speed_G_TREND_1, excess_12h_G_TREND_1, coherence_24_G_TREND_1, evr6_x_volz_G_TREND_1, meta_abs_net_x_drawext_G_TREND_1, rv_24h_G_TREND_1, vov_interaction_G_TREND_1, sin_hod_G_TREND_1, signed_vol_G_TREND_1


#### SHORT_TF
- **Features**: 48
- **OOF AUC**: 0.6302
- **OOF IC**: 0.2288
- **OOF Rank IC**: nan
- **OOF Sharpe**: 0.0809
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.7834
- **OOF Prec@40**: 0.6418
- **OOF Avg Trades/Day @10%**: 2.3690
- **OOF Avg Trades/Day @30%**: 2.6394
- **OOF ECE@10**: 0.0651
- **OOF Calibration Profile**: well-calibrated

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0270 / 0.2430 / 0.5905 / 483 | 0.0721 / 0.2285 / 0.6370 / 485 | 0.0427 / 0.2189 / 0.6070 / 483 |
| vol_48h | 0.0691 / 0.2326 / 0.6354 / 483 | 0.0371 / 0.2369 / 0.6028 / 485 | 0.0414 / 0.2209 / 0.6124 / 483 |
| volume_12h | 0.0427 / 0.2378 / 0.6262 / 483 | 0.0137 / 0.2400 / 0.5569 / 485 | 0.0652 / 0.2126 / 0.6664 / 483 |
| volume_48h | 0.0242 / 0.2436 / 0.5856 / 483 | 0.0948 / 0.2200 / 0.6755 / 485 | 0.0621 / 0.2269 / 0.6225 / 483 |
| trend_12h | 0.0204 / 0.2101 / 0.6449 / 517 | -0.0044 / 0.2510 / 0.5509 / 451 | 0.0692 / 0.2322 / 0.6393 / 483 |
| trend_48h | 0.0574 / 0.2183 / 0.6386 / 483 | 0.0605 / 0.2332 / 0.6273 / 485 | 0.0436 / 0.2389 / 0.6019 / 483 |

- **Top features**: ret1h_z_G_TREND_1, meta_alignment_G_TREND_1, rvol_hod_base_G_TREND_1, amihud_illiq_G_TREND_1, excess_12h_G_VOL_1, meta_alignment_G_VOL_0, volu_z_G_TREND_1, amihud_z_G_TREND_1, stage_tf_G_TREND_1, G_LIQ_GREAT_G_VOL_0

## Meta Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Calibration |
|-------|----------|-----|----|---------| ---------|-------------|
| long_mr | 24 | N/A | N/A | N/A | N/A | N/A |
| long_tf | 24 | N/A | N/A | N/A | N/A | N/A |
| short_mr | 24 | N/A | N/A | N/A | N/A | N/A |
| short_tf | 24 | N/A | N/A | N/A | N/A | N/A |

### Detailed Meta Model Performance

#### long_mr
- **Features**: 24

#### long_tf
- **Features**: 24

#### short_mr
- **Features**: 24

#### short_tf
- **Features**: 24

## Specialist Models
- **Trap (GMM)**: 8 features, clusters=GaussianMixture(covariance_type='diag', max_iter=200, n_components=4, n_init=3,
                random_state=42)
- **Gamma (ExtraTrees)**: 20 features
