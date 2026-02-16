# Training Report — 20260214_190000
Generated: 2026-02-15 00:28 UTC

## Configuration
- **Train lookback**: 35040 hours
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
| gamma_model | 10,967,338 | 28 |
| spike_anatomy_best | 18,023 | 12 |
| spike_anatomy_worst | 20,109 | 12 |
| train_long_mr_2 | 7,551 | 687 |
| train_long_mr_4 | 7,551 | 687 |
| train_long_mr_8 | 7,551 | 687 |
| train_long_tf_2 | 5,167 | 681 |
| train_long_tf_4 | 5,167 | 681 |
| train_long_tf_8 | 5,167 | 681 |
| train_short_mr_2 | 5,167 | 687 |
| train_short_mr_4 | 5,167 | 687 |
| train_short_mr_8 | 5,167 | 687 |
| train_short_tf_2 | 7,551 | 681 |
| train_short_tf_4 | 7,551 | 681 |
| train_short_tf_8 | 7,551 | 681 |
| trap_model | 10,967,338 | 11 |

## Alpha Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Prec@10 | Prec@40 | AvgTr/day@10 | AvgTr/day@30 | Avg Return | Trades | Best Iter |
|-------|----------|-----|----|---------| ---------|---------|---------|-------------|-------------|------------|--------|-----------|
| LONG_MR | 48 | 0.5181 | 0.0234 | -0.1269 | nan% | 0.4043 | 0.3610 | 4.7448 | 5.8961 | nan | N/A | N/A |
| LONG_TF | 48 | 0.5585 | 0.1401 | -0.1485 | nan% | 0.4682 | 0.3782 | 4.1466 | 4.3300 | nan | N/A | N/A |
| SHORT_MR | 48 | 0.5288 | 0.0692 | -0.2119 | nan% | 0.3491 | 0.3393 | 4.1466 | 4.3300 | nan | N/A | N/A |
| SHORT_TF | 48 | 0.5475 | 0.0547 | -0.1651 | nan% | 0.3300 | 0.3250 | 4.7448 | 5.8961 | nan | N/A | N/A |

### Detailed Model Performance

#### LONG_MR
- **Features**: 48
- **OOF AUC**: 0.5181
- **OOF IC**: 0.0234
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.1269
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.4043
- **OOF Prec@40**: 0.3610
- **OOF Avg Trades/Day @10%**: 4.7448
- **OOF Avg Trades/Day @30%**: 5.8961
- **OOF ECE@10**: 0.0497
- **OOF Calibration Profile**: underconfident/conservative

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0017 / 0.2111 / 0.5152 / 2515 | 0.0013 / 0.2109 / 0.5207 / 2521 | 0.0007 / 0.2072 / 0.5177 / 2515 |
| vol_48h | 0.0029 / 0.2064 / 0.5362 / 2515 | -0.0009 / 0.2158 / 0.5091 / 2521 | 0.0006 / 0.2070 / 0.5093 / 2515 |
| volume_12h | 0.0027 / 0.2119 / 0.5307 / 2515 | -0.0001 / 0.2085 / 0.5042 / 2521 | 0.0010 / 0.2089 / 0.5185 / 2515 |
| volume_48h | -0.0014 / 0.2169 / 0.5029 / 2515 | -0.0002 / 0.2063 / 0.5119 / 2521 | 0.0035 / 0.2061 / 0.5342 / 2515 |
| trend_12h | 0.0005 / 0.2168 / 0.5190 / 2871 | -0.0019 / 0.2094 / 0.4917 / 2165 | 0.0017 / 0.2021 / 0.5304 / 2515 |
| trend_48h | 0.0043 / 0.2115 / 0.5391 / 2515 | -0.0012 / 0.2113 / 0.4955 / 2521 | 0.0003 / 0.2064 / 0.5173 / 2515 |

- **Top features**: mae_4h_G_VOL_1, mfe_4h_G_VOL_1, breakout_confirmed_G_VOL_1, rsi_slope_base_G_VOL_1, ft_drop_8_G_VOL_1, rsi_slope_G_VOL_1, dist_ema_fast_base_G_VOL_1, retrace_12_G_VOL_1, breakout_t_G_VOL_1, thrust_decay_8_G_VOL_1


#### LONG_TF
- **Features**: 48
- **OOF AUC**: 0.5585
- **OOF IC**: 0.1401
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.1485
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.4682
- **OOF Prec@40**: 0.3782
- **OOF Avg Trades/Day @10%**: 4.1466
- **OOF Avg Trades/Day @30%**: 4.3300
- **OOF ECE@10**: 0.0331
- **OOF Calibration Profile**: underconfident/conservative

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0069 / 0.2209 / 0.5300 / 1721 | 0.0108 / 0.2112 / 0.5534 / 1725 | 0.0258 / 0.1859 / 0.5742 / 1721 |
| vol_48h | 0.0155 / 0.2220 / 0.5478 / 1721 | 0.0061 / 0.2189 / 0.5456 / 1689 | -0.0022 / 0.1781 / 0.5409 / 1757 |
| volume_12h | 0.0008 / 0.2280 / 0.5320 / 1721 | 0.0134 / 0.2109 / 0.5403 / 1725 | 0.0122 / 0.1793 / 0.5706 / 1721 |
| volume_48h | -0.0003 / 0.2218 / 0.5269 / 1721 | 0.0163 / 0.2015 / 0.5479 / 1725 | 0.0349 / 0.1948 / 0.5921 / 1721 |
| trend_12h | 0.0059 / 0.2252 / 0.5472 / 1721 | 0.0242 / 0.2072 / 0.5535 / 1725 | 0.0019 / 0.1856 / 0.5425 / 1721 |
| trend_48h | 0.0004 / 0.2284 / 0.5305 / 1721 | 0.0186 / 0.2144 / 0.5622 / 1725 | -0.0078 / 0.1753 / 0.5389 / 1721 |

- **Top features**: ft_4_G_VOL_0, prog_def_G_VOL_0, rv_4h_G_VOL_0, rv_120h_G_VOL_0, align_G_VOL_1, vol_state_G_VOL_1, prog_def_G_VOL_1, range_pct_G_VOL_0, slope_G_VOL_1, vov_mad_20_G_VOL_1


#### SHORT_MR
- **Features**: 48
- **OOF AUC**: 0.5288
- **OOF IC**: 0.0692
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.2119
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.3491
- **OOF Prec@40**: 0.3393
- **OOF Avg Trades/Day @10%**: 4.1466
- **OOF Avg Trades/Day @30%**: 4.3300
- **OOF ECE@10**: 0.0222
- **OOF Calibration Profile**: underconfident/conservative

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | -0.0004 / 0.2228 / 0.5249 / 1721 | 0.0038 / 0.2070 / 0.5300 / 1725 | -0.0032 / 0.1981 / 0.5171 / 1721 |
| vol_48h | 0.0049 / 0.2125 / 0.5306 / 1721 | 0.0011 / 0.2074 / 0.5218 / 1689 | 0.0035 / 0.2079 / 0.5320 / 1757 |
| volume_12h | 0.0015 / 0.2182 / 0.5207 / 1721 | 0.0025 / 0.2046 / 0.5279 / 1725 | 0.0027 / 0.2051 / 0.5312 / 1721 |
| volume_48h | 0.0027 / 0.2150 / 0.5343 / 1721 | 0.0006 / 0.2145 / 0.5086 / 1725 | 0.0021 / 0.1984 / 0.5408 / 1721 |
| trend_12h | 0.0005 / 0.2082 / 0.5178 / 1721 | 0.0017 / 0.2170 / 0.5290 / 1725 | 0.0052 / 0.2026 / 0.5399 / 1721 |
| trend_48h | 0.0026 / 0.2112 / 0.5278 / 1721 | -0.0004 / 0.2118 / 0.5183 / 1725 | 0.0076 / 0.2048 / 0.5408 / 1721 |

- **Top features**: mae_4h_G_VOL_1, mae_4h_G_VOL_0, pullback_48_G_VOL_1, mae_8h_G_VOL_1, meta_abs_net_x_breakout_G_VOL_0, pullback_8_G_VOL_1, pullback_72_G_VOL_1, impulse_ratio_24_G_VOL_1, pullback_120_G_VOL_1, pullback_4_G_VOL_1


#### SHORT_TF
- **Features**: 48
- **OOF AUC**: 0.5475
- **OOF IC**: 0.0547
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.1651
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.3300
- **OOF Prec@40**: 0.3250
- **OOF Avg Trades/Day @10%**: 4.7448
- **OOF Avg Trades/Day @30%**: 5.8961
- **OOF ECE@10**: 0.0261
- **OOF Calibration Profile**: well-calibrated

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0112 / 0.2071 / 0.5597 / 2515 | 0.0041 / 0.2120 / 0.5345 / 2521 | 0.0102 / 0.2055 / 0.5486 / 2515 |
| vol_48h | 0.0054 / 0.2111 / 0.5453 / 2515 | 0.0083 / 0.2060 / 0.5470 / 2521 | 0.0119 / 0.2076 / 0.5487 / 2515 |
| volume_12h | 0.0077 / 0.2086 / 0.5536 / 2515 | 0.0066 / 0.2138 / 0.5396 / 2521 | 0.0099 / 0.2023 / 0.5477 / 2515 |
| volume_48h | 0.0087 / 0.2032 / 0.5591 / 2515 | 0.0075 / 0.2113 / 0.5389 / 2521 | 0.0085 / 0.2102 / 0.5457 / 2515 |
| trend_12h | 0.0055 / 0.1994 / 0.5419 / 2871 | 0.0047 / 0.2137 / 0.5403 / 2165 | 0.0109 / 0.2135 / 0.5625 / 2515 |
| trend_48h | 0.0088 / 0.2017 / 0.5471 / 2515 | 0.0072 / 0.2102 / 0.5448 / 2521 | 0.0079 / 0.2129 / 0.5500 / 2515 |

- **Top features**: accel_5h_G_VOL_1, dist_ema_fast_base_G_VOL_1, reject_dir2h_prod_G_VOL_1, rsi_G_VOL_1, trend_accel_120h_G_VOL_1, dir_path_short_2h_G_VOL_1, accept_dir2h_abs_prod_G_VOL_1, thrust_decay_8_G_VOL_1, excess_12h_G_VOL_1, G_EXH_GIVEBACK_G_VOL_1

## Meta Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Calibration |
|-------|----------|-----|----|---------| ---------|-------------|
| long_mr_H2 | 66 | N/A | N/A | N/A | N/A | N/A |
| long_mr_H4 | 68 | N/A | N/A | N/A | N/A | N/A |
| long_mr_H8 | 67 | N/A | N/A | N/A | N/A | N/A |
| long_mr_clf | 150 | N/A | N/A | N/A | N/A | N/A |
| long_tf_H2 | 61 | N/A | N/A | N/A | N/A | N/A |
| long_tf_H4 | 62 | N/A | N/A | N/A | N/A | N/A |
| long_tf_H8 | 63 | N/A | N/A | N/A | N/A | N/A |
| long_tf_clf | 150 | N/A | N/A | N/A | N/A | N/A |
| short_mr_H2 | 58 | N/A | N/A | N/A | N/A | N/A |
| short_mr_H4 | 62 | N/A | N/A | N/A | N/A | N/A |
| short_mr_H8 | 60 | N/A | N/A | N/A | N/A | N/A |
| short_mr_clf | 150 | N/A | N/A | N/A | N/A | N/A |
| short_tf_H2 | 63 | N/A | N/A | N/A | N/A | N/A |
| short_tf_H4 | 67 | N/A | N/A | N/A | N/A | N/A |
| short_tf_H8 | 66 | N/A | N/A | N/A | N/A | N/A |
| short_tf_clf | 150 | N/A | N/A | N/A | N/A | N/A |

### Detailed Meta Model Performance

#### long_mr_H2
- **Features**: 66

#### long_mr_H4
- **Features**: 68

#### long_mr_H8
- **Features**: 67

#### long_mr_clf
- **Features**: 150

#### long_tf_H2
- **Features**: 61

#### long_tf_H4
- **Features**: 62

#### long_tf_H8
- **Features**: 63

#### long_tf_clf
- **Features**: 150

#### short_mr_H2
- **Features**: 58

#### short_mr_H4
- **Features**: 62

#### short_mr_H8
- **Features**: 60

#### short_mr_clf
- **Features**: 150

#### short_tf_H2
- **Features**: 63

#### short_tf_H4
- **Features**: 67

#### short_tf_H8
- **Features**: 66

#### short_tf_clf
- **Features**: 150

## Specialist Models
- **Trap (GMM)**: 8 features, clusters=GaussianMixture(covariance_type='diag', max_iter=200, n_components=4, n_init=3,
                random_state=42)
- **Gamma (ExtraTrees)**: 20 features
