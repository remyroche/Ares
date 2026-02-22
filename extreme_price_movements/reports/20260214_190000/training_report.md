# Training Report — 20260214_190000
Generated: 2026-02-22 14:49 UTC

## Configuration
- **Train lookback**: 35040 hours
- **Label horizons**: [2, 4, 8]
- **Label method**: triple_barrier
- **Label quantiles**: lo=0.3, hi=0.65
- **OOS holdout**: 180 days
- **Min train samples**: 200
- **Feature selection**: MDI (min=30, cap=0.995)
- **15m precision**: True

## Dataset Sizes
| Dataset | Rows | Features |
|---------|------|----------|
| gamma_model | 10,967,338 | 28 |
| spike_anatomy_best | 103,713 | 12 |
| spike_anatomy_worst | 118,600 | 12 |
| train_long_mr_2 | 152,512 | 1057 |
| train_long_mr_4 | 152,512 | 1057 |
| train_long_mr_8 | 152,512 | 1057 |
| train_long_tf_2 | 69,804 | 1051 |
| train_long_tf_4 | 69,804 | 1051 |
| train_long_tf_8 | 69,804 | 1051 |
| train_short_mr_2 | 69,804 | 1057 |
| train_short_mr_4 | 69,804 | 1057 |
| train_short_mr_8 | 69,804 | 1057 |
| train_short_tf_2 | 152,512 | 1051 |
| train_short_tf_4 | 152,512 | 1051 |
| train_short_tf_8 | 152,512 | 1051 |
| trap_model | 11,252,363 | 11 |

## Alpha Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Prec@10 | Prec@40 | AvgTr/day@10 | AvgTr/day@30 | Avg Return | Trades | Best Iter |
|-------|----------|-----|----|---------| ---------|---------|---------|-------------|-------------|------------|--------|-----------|
| LONG_MR | 48 | 0.5571 | 0.0998 | -1.0485 | nan% | 0.0285 | 0.0340 | 12.5842 | 37.7508 | nan | N/A | N/A |
| LONG_TF | 48 | 0.5516 | 0.0809 | -1.2662 | nan% | 0.0276 | 0.0367 | 5.7174 | 17.1515 | nan | N/A | N/A |
| SHORT_MR | 48 | 0.5285 | 0.0659 | -0.6585 | nan% | 0.0680 | 0.0631 | 5.7174 | 17.1515 | nan | N/A | N/A |
| SHORT_TF | 48 | 0.5831 | 0.1984 | -0.8304 | nan% | 0.0868 | 0.0774 | 12.5842 | 37.7508 | nan | N/A | N/A |

### Detailed Model Performance

#### LONG_MR
- **Features**: 48
- **OOF AUC**: 0.5571
- **OOF IC**: 0.0998
- **OOF Rank IC**: nan
- **OOF Sharpe**: -1.0485
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.0285
- **OOF Prec@40**: 0.0340
- **OOF Avg Trades/Day @10%**: 12.5842
- **OOF Avg Trades/Day @30%**: 37.7508
- **OOF ECE@10**: 0.0134
- **OOF Calibration Profile**: flat

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | -0.1533 / 0.0344 / 0.5578 / 50787 | -0.1340 / 0.0349 / 0.5614 / 50938 | -0.1482 / 0.0344 / 0.5520 / 50787 |
| vol_48h | -0.1527 / 0.0349 / 0.5499 / 50787 | -0.1362 / 0.0342 / 0.5667 / 50938 | -0.1461 / 0.0348 / 0.5549 / 50787 |
| volume_12h | -0.1599 / 0.0339 / 0.5531 / 50787 | -0.1363 / 0.0346 / 0.5688 / 50938 | -0.1397 / 0.0353 / 0.5494 / 50787 |
| volume_48h | -0.1445 / 0.0346 / 0.5569 / 50787 | -0.1368 / 0.0358 / 0.5581 / 50938 | -0.1546 / 0.0334 / 0.5556 / 50787 |
| trend_12h | -0.1206 / 0.0389 / 0.5475 / 50787 | -0.1302 / 0.0361 / 0.5613 / 50938 | -0.2037 / 0.0288 / 0.5537 / 50787 |
| trend_48h | -0.1456 / 0.0344 / 0.5547 / 50787 | -0.1381 / 0.0344 / 0.5680 / 50938 | -0.1515 / 0.0349 / 0.5493 / 50787 |

- **Top features**: kf_atr_mean_G_VOL_1, kf_atr_mean_G_VOL_0, body_pct_G_VOL_1, atr_pct_base_G_VOL_0, accel_5h_G_VOL_1, ret48h_G_VOL_1, ret72h_G_VOL_1, churn_G_VOL_0, ret72h_G_VOL_0, ret48h_G_VOL_0


#### LONG_TF
- **Features**: 48
- **OOF AUC**: 0.5516
- **OOF IC**: 0.0809
- **OOF Rank IC**: nan
- **OOF Sharpe**: -1.2662
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.0276
- **OOF Prec@40**: 0.0367
- **OOF Avg Trades/Day @10%**: 5.7174
- **OOF Avg Trades/Day @30%**: 17.1515
- **OOF ECE@10**: 0.0132
- **OOF Calibration Profile**: well-calibrated

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | -0.1391 / 0.0364 / 0.5488 / 23245 | -0.1327 / 0.0365 / 0.5445 / 23314 | -0.1514 / 0.0328 / 0.5606 / 23245 |
| vol_48h | -0.1395 / 0.0360 / 0.5453 / 23245 | -0.1293 / 0.0362 / 0.5499 / 23314 | -0.1543 / 0.0335 / 0.5596 / 23245 |
| volume_12h | -0.1232 / 0.0380 / 0.5454 / 23245 | -0.1259 / 0.0363 / 0.5578 / 23314 | -0.1817 / 0.0315 / 0.5496 / 23245 |
| volume_48h | -0.1183 / 0.0371 / 0.5584 / 23245 | -0.1373 / 0.0357 / 0.5521 / 23314 | -0.1710 / 0.0329 / 0.5431 / 23245 |
| trend_12h | -0.0809 / 0.0465 / 0.5558 / 23245 | -0.1851 / 0.0316 / 0.5401 / 23314 | -0.2142 / 0.0276 / 0.5100 / 23245 |
| trend_48h | -0.0908 / 0.0406 / 0.5792 / 23245 | -0.1357 / 0.0356 / 0.5512 / 23314 | -0.2285 / 0.0295 / 0.5053 / 23245 |

- **Top features**: kf_atr_mean_G_VOL_0, kf_atr_mean_G_VOL_1, body_pct_G_VOL_1, v_power_G_VOL_0, ret120h_G_VOL_1, G_LIQ_GOOD_G_VOL_0, trapped_longs_96_G_VOL_1, kf_score_rm24_mean_G_VOL_0, upside_semivariance_24_G_VOL_1, volume_trend_alignment_G_VOL_1


#### SHORT_MR
- **Features**: 48
- **OOF AUC**: 0.5285
- **OOF IC**: 0.0659
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.6585
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.0680
- **OOF Prec@40**: 0.0631
- **OOF Avg Trades/Day @10%**: 5.7174
- **OOF Avg Trades/Day @30%**: 17.1515
- **OOF ECE@10**: 0.0046
- **OOF Calibration Profile**: well-calibrated

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | -0.0749 / 0.0475 / 0.5499 / 23245 | -0.0807 / 0.0459 / 0.5289 / 23314 | -0.1332 / 0.0383 / 0.4926 / 23245 |
| vol_48h | -0.0636 / 0.0498 / 0.5477 / 23245 | -0.0924 / 0.0429 / 0.5292 / 23314 | -0.1355 / 0.0390 / 0.4947 / 23245 |
| volume_12h | -0.0740 / 0.0480 / 0.5338 / 23245 | -0.1003 / 0.0417 / 0.5376 / 23314 | -0.1080 / 0.0420 / 0.5114 / 23245 |
| volume_48h | -0.0898 / 0.0447 / 0.5213 / 23245 | -0.0927 / 0.0434 / 0.5346 / 23314 | -0.0955 / 0.0436 / 0.5303 / 23245 |
| trend_12h | -0.0924 / 0.0481 / 0.5124 / 23245 | -0.0739 / 0.0469 / 0.5471 / 23314 | -0.1216 / 0.0366 / 0.5063 / 23245 |
| trend_48h | -0.0764 / 0.0473 / 0.5241 / 23245 | -0.0820 / 0.0455 / 0.5390 / 23314 | -0.1281 / 0.0389 / 0.5185 / 23245 |

- **Top features**: G_LIQ_GOOD_G_VOL_0, kf_atr_mean_G_VOL_0, kf_atr_mean_G_VOL_1, support_quality_score_G_VOL_0, ffd_amihud_04_G_VOL_0, vol_expansion_ratio_G_VOL_1, is_trending_G_VOL_1, ffd_amihud_04_G_VOL_1, chop_score_G_VOL_1, meta_alignment_G_VOL_0


#### SHORT_TF
- **Features**: 48
- **OOF AUC**: 0.5831
- **OOF IC**: 0.1984
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.8304
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.0868
- **OOF Prec@40**: 0.0774
- **OOF Avg Trades/Day @10%**: 12.5842
- **OOF Avg Trades/Day @30%**: 37.7508
- **OOF ECE@10**: 0.0047
- **OOF Calibration Profile**: flat

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | -0.0978 / 0.0446 / 0.6040 / 50787 | -0.1007 / 0.0391 / 0.5844 / 50938 | -0.0808 / 0.0376 / 0.5499 / 50787 |
| vol_48h | -0.0959 / 0.0454 / 0.6151 / 50787 | -0.0948 / 0.0397 / 0.5729 / 50938 | -0.0894 / 0.0361 / 0.5425 / 50787 |
| volume_12h | -0.1004 / 0.0380 / 0.6018 / 50787 | -0.0847 / 0.0407 / 0.5985 / 50938 | -0.0953 / 0.0425 / 0.5514 / 50787 |
| volume_48h | -0.1080 / 0.0378 / 0.5680 / 50787 | -0.0843 / 0.0409 / 0.5975 / 50938 | -0.0893 / 0.0426 / 0.5812 / 50787 |
| trend_12h | -0.0774 / 0.0420 / 0.5484 / 50787 | -0.1017 / 0.0420 / 0.5923 / 50938 | -0.1022 / 0.0373 / 0.6104 / 50787 |
| trend_48h | -0.0908 / 0.0359 / 0.5538 / 50787 | -0.1011 / 0.0400 / 0.5958 / 50938 | -0.0893 / 0.0454 / 0.5875 / 50787 |

- **Top features**: kf_atr_mean_G_VOL_0, atr_pct_base_G_VOL_0, body_pct_G_VOL_0, kf_atr_mean_G_VOL_1, asset_atr_level_G_VOL_1, body_pct_G_VOL_1, churn_G_VOL_0, asset_atr_level_G_VOL_0, G_VOL, ret72h_G_VOL_0

## Meta Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Calibration |
|-------|----------|-----|----|---------| ---------|-------------|
| long_mr_H8 | 68 | N/A | N/A | N/A | N/A | N/A |
| long_mr_clf | 217 | N/A | N/A | N/A | N/A | N/A |
| long_tf_H8 | 66 | N/A | N/A | N/A | N/A | N/A |
| long_tf_clf | 216 | N/A | N/A | N/A | N/A | N/A |
| short_mr_H8 | 68 | N/A | N/A | N/A | N/A | N/A |
| short_mr_clf | 216 | N/A | N/A | N/A | N/A | N/A |
| short_tf_H8 | 64 | N/A | N/A | N/A | N/A | N/A |
| short_tf_clf | 217 | N/A | N/A | N/A | N/A | N/A |

### Detailed Meta Model Performance

#### long_mr_H8
- **Features**: 68

#### long_mr_clf
- **Features**: 217

#### long_tf_H8
- **Features**: 66

#### long_tf_clf
- **Features**: 216

#### short_mr_H8
- **Features**: 68

#### short_mr_clf
- **Features**: 216

#### short_tf_H8
- **Features**: 64

#### short_tf_clf
- **Features**: 217

## Specialist Models
- **Trap (GMM)**: 8 features, clusters=GaussianMixture(covariance_type='diag', max_iter=200, n_components=4, n_init=3,
                random_state=42)
- **Gamma (ExtraTrees)**: 20 features
