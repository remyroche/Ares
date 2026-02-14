# Training Report — 20260213_080000
Generated: 2026-02-14 09:19 UTC

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
| exh_down | 115,486 | 34 |
| exh_up | 35,372 | 34 |
| gamma_model | 2,499,626 | 28 |
| spike_anatomy_best | 9,330 | 12 |
| spike_anatomy_worst | 10,592 | 12 |
| train_long_mr_2 | 2,282 | 675 |
| train_long_mr_4 | 2,282 | 675 |
| train_long_mr_8 | 2,282 | 675 |
| train_long_tf_2 | 1,830 | 667 |
| train_long_tf_4 | 1,830 | 667 |
| train_long_tf_8 | 1,830 | 667 |
| train_short_mr_2 | 1,830 | 675 |
| train_short_mr_4 | 1,830 | 675 |
| train_short_mr_8 | 1,830 | 675 |
| train_short_tf_2 | 2,282 | 667 |
| train_short_tf_4 | 2,282 | 667 |
| train_short_tf_8 | 2,282 | 667 |
| trap_model | 8,325,383 | 11 |

## Alpha Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Prec@10 | Prec@40 | AvgTr/day@10 | AvgTr/day@30 | Avg Return | Trades | Best Iter |
|-------|----------|-----|----|---------| ---------|---------|---------|-------------|-------------|------------|--------|-----------|
| LONG_MR | 48 | 0.5581 | 0.1355 | -0.1061 | nan% | 0.5509 | 0.4097 | 2.8173 | 3.3160 | nan | N/A | N/A |
| LONG_TF | 48 | 0.5473 | 0.0851 | -0.0026 | nan% | 0.4845 | 0.4769 | 2.6337 | 2.7037 | nan | N/A | N/A |
| SHORT_MR | 48 | 0.5956 | 0.1547 | -0.0590 | nan% | 0.6421 | 0.5148 | 2.6337 | 2.7037 | nan | N/A | N/A |
| SHORT_TF | 48 | 0.5503 | 0.1377 | -0.1160 | nan% | 0.4834 | 0.5039 | 2.8173 | 3.3160 | nan | N/A | N/A |

### Detailed Model Performance

#### LONG_MR
- **Features**: 48
- **OOF AUC**: 0.5581
- **OOF IC**: 0.1355
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.1061
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.5509
- **OOF Prec@40**: 0.4097
- **OOF Avg Trades/Day @10%**: 2.8173
- **OOF Avg Trades/Day @30%**: 3.3160
- **OOF ECE@10**: 0.0986
- **OOF Calibration Profile**: underconfident/conservative

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0106 / 0.2157 / 0.5645 / 760 | 0.0111 / 0.1883 / 0.5814 / 762 | 0.0103 / 0.2167 / 0.5208 / 760 |
| vol_48h | 0.0121 / 0.2015 / 0.5690 / 760 | 0.0202 / 0.2114 / 0.5727 / 762 | 0.0112 / 0.2077 / 0.5298 / 760 |
| volume_12h | 0.0145 / 0.2016 / 0.5727 / 760 | 0.0104 / 0.2064 / 0.5579 / 762 | 0.0184 / 0.2126 / 0.5455 / 760 |
| volume_48h | 0.0097 / 0.2105 / 0.5575 / 760 | 0.0101 / 0.2011 / 0.5482 / 762 | 0.0237 / 0.2090 / 0.5679 / 760 |
| trend_12h | 0.0004 / 0.2316 / 0.5518 / 792 | 0.0177 / 0.2019 / 0.5864 / 730 | -0.0091 / 0.1859 / 0.5620 / 760 |
| trend_48h | -0.0028 / 0.2293 / 0.5356 / 760 | 0.0278 / 0.1975 / 0.5900 / 762 | 0.0003 / 0.1939 / 0.5568 / 760 |

- **Top features**: mfe_4h_G_VOL_1, ft_drop_8_G_VOL_1, mfe_8h_G_VOL_1, thrust_decay_8_G_VOL_1, retrace_12_G_VOL_1, mae_4h_G_VOL_1, pullback_8_G_VOL_1, volume_entropy_12_G_VOL_1, cos_hod_G_VOL_1, donch_dist_8_G_VOL_0


#### LONG_TF
- **Features**: 48
- **OOF AUC**: 0.5473
- **OOF IC**: 0.0851
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.0026
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.4845
- **OOF Prec@40**: 0.4769
- **OOF Avg Trades/Day @10%**: 2.6337
- **OOF Avg Trades/Day @30%**: 2.7037
- **OOF ECE@10**: 0.1327
- **OOF Calibration Profile**: overconfident

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0184 / 0.2111 / 0.5685 / 610 | -0.0044 / 0.2135 / 0.5048 / 610 | 0.0104 / 0.1997 / 0.5640 / 610 |
| vol_48h | -0.0025 / 0.2253 / 0.5130 / 610 | 0.0032 / 0.2046 / 0.5308 / 551 | 0.0138 / 0.1953 / 0.5796 / 669 |
| volume_12h | 0.0091 / 0.2074 / 0.5444 / 610 | -0.0076 / 0.2254 / 0.5096 / 610 | 0.0142 / 0.1915 / 0.5957 / 610 |
| volume_48h | -0.0066 / 0.2087 / 0.4998 / 610 | 0.0143 / 0.2010 / 0.5660 / 610 | 0.0158 / 0.2146 / 0.5706 / 610 |
| trend_12h | 0.0117 / 0.2015 / 0.5541 / 610 | 0.0041 / 0.2085 / 0.5332 / 610 | 0.0091 / 0.2143 / 0.5563 / 610 |
| trend_48h | 0.0033 / 0.2039 / 0.5199 / 610 | 0.0090 / 0.2195 / 0.5536 / 610 | 0.0077 / 0.2009 / 0.5625 / 610 |

- **Top features**: rv_48h_G_VOL_0, ft_2_G_VOL_0, sin_hod_G_VOL_1, asym_ft_G_VOL_0, vov_mad_20_G_VOL_1, G_MR_SPIKE_G_VOL_1, tfq_x_dir_edge_2h_G_VOL_1, evr_slope_G_VOL_1, dist_ema_slow_base_G_VOL_0, momentum_accel_G_VOL_0


#### SHORT_MR
- **Features**: 48
- **OOF AUC**: 0.5956
- **OOF IC**: 0.1547
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.0590
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.6421
- **OOF Prec@40**: 0.5148
- **OOF Avg Trades/Day @10%**: 2.6337
- **OOF Avg Trades/Day @30%**: 2.7037
- **OOF ECE@10**: 0.0598
- **OOF Calibration Profile**: overconfident

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0721 / 0.2034 / 0.6431 / 610 | 0.0124 / 0.2027 / 0.5882 / 610 | 0.0054 / 0.2035 / 0.5500 / 610 |
| vol_48h | 0.0616 / 0.2018 / 0.6406 / 610 | 0.0309 / 0.2054 / 0.5954 / 551 | 0.0032 / 0.2027 / 0.5502 / 669 |
| volume_12h | 0.0315 / 0.2071 / 0.5868 / 610 | 0.0468 / 0.2014 / 0.6358 / 610 | 0.0168 / 0.2011 / 0.5642 / 610 |
| volume_48h | 0.0304 / 0.2061 / 0.5924 / 610 | 0.0463 / 0.2057 / 0.6201 / 610 | 0.0159 / 0.1978 / 0.5725 / 610 |
| trend_12h | 0.0314 / 0.2112 / 0.5985 / 610 | 0.0530 / 0.2013 / 0.6207 / 610 | 0.0049 / 0.1971 / 0.5683 / 610 |
| trend_48h | 0.0408 / 0.2057 / 0.6019 / 610 | 0.0428 / 0.1997 / 0.6132 / 610 | 0.0120 / 0.2042 / 0.5725 / 610 |

- **Top features**: mae_4h_G_VOL_1, meta_abs_net_x_breakout_G_VOL_0, speed_G_VOL_0, mae_8h_G_VOL_1, pullback_48_G_VOL_1, excess_12h_G_VOL_1, pullback_72_G_VOL_1, amihud_illiq_G_VOL_0, spike_score_G_VOL_0, speed_G_VOL_1


#### SHORT_TF
- **Features**: 48
- **OOF AUC**: 0.5503
- **OOF IC**: 0.1377
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.1160
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.4834
- **OOF Prec@40**: 0.5039
- **OOF Avg Trades/Day @10%**: 2.8173
- **OOF Avg Trades/Day @30%**: 3.3160
- **OOF ECE@10**: 0.1066
- **OOF Calibration Profile**: well-calibrated

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0216 / 0.2155 / 0.5663 / 760 | 0.0030 / 0.2131 / 0.5183 / 762 | 0.0025 / 0.1934 / 0.5472 / 760 |
| vol_48h | 0.0123 / 0.2247 / 0.5577 / 760 | 0.0033 / 0.1996 / 0.5235 / 762 | 0.0047 / 0.1979 / 0.5433 / 760 |
| volume_12h | 0.0172 / 0.2174 / 0.5460 / 760 | -0.0018 / 0.2074 / 0.4955 / 762 | 0.0160 / 0.1973 / 0.6006 / 760 |
| volume_48h | 0.0098 / 0.2042 / 0.5232 / 760 | 0.0175 / 0.2034 / 0.5556 / 762 | 0.0094 / 0.2145 / 0.5762 / 760 |
| trend_12h | 0.0014 / 0.1840 / 0.5691 / 792 | 0.0148 / 0.2121 / 0.5460 / 730 | -0.0036 / 0.2271 / 0.5006 / 760 |
| trend_48h | 0.0213 / 0.1968 / 0.5783 / 760 | -0.0087 / 0.2083 / 0.4940 / 762 | 0.0190 / 0.2170 / 0.5635 / 760 |

- **Top features**: meta_alignment_G_VOL_1, rv_120h_G_VOL_1, rv_48h_G_VOL_1, volatility_zscore_G_VOL_1, cos_hod_G_VOL_1, vol_z_G_VOL_0, down_up_vol_ratio_24_G_VOL_1, stage_tf_G_VOL_1, perm_entropy_ret_12_G_VOL_0, ft_2_G_VOL_1

## Meta Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Calibration |
|-------|----------|-----|----|---------| ---------|-------------|
| long_mr_H2 | 62 | N/A | N/A | N/A | N/A | N/A |
| long_mr_H4 | 57 | N/A | N/A | N/A | N/A | N/A |
| long_mr_H8 | 56 | N/A | N/A | N/A | N/A | N/A |
| long_mr_clf | 146 | N/A | N/A | N/A | N/A | N/A |
| long_tf_H2 | 62 | N/A | N/A | N/A | N/A | N/A |
| long_tf_H4 | 60 | N/A | N/A | N/A | N/A | N/A |
| long_tf_H8 | 67 | N/A | N/A | N/A | N/A | N/A |
| long_tf_clf | 146 | N/A | N/A | N/A | N/A | N/A |
| short_mr_H2 | 62 | N/A | N/A | N/A | N/A | N/A |
| short_mr_H4 | 60 | N/A | N/A | N/A | N/A | N/A |
| short_mr_H8 | 59 | N/A | N/A | N/A | N/A | N/A |
| short_mr_clf | 146 | N/A | N/A | N/A | N/A | N/A |
| short_tf_H2 | 64 | N/A | N/A | N/A | N/A | N/A |
| short_tf_H4 | 60 | N/A | N/A | N/A | N/A | N/A |
| short_tf_H8 | 61 | N/A | N/A | N/A | N/A | N/A |
| short_tf_clf | 146 | N/A | N/A | N/A | N/A | N/A |

### Detailed Meta Model Performance

#### long_mr_H2
- **Features**: 62

#### long_mr_H4
- **Features**: 57

#### long_mr_H8
- **Features**: 56

#### long_mr_clf
- **Features**: 146

#### long_tf_H2
- **Features**: 62

#### long_tf_H4
- **Features**: 60

#### long_tf_H8
- **Features**: 67

#### long_tf_clf
- **Features**: 146

#### short_mr_H2
- **Features**: 62

#### short_mr_H4
- **Features**: 60

#### short_mr_H8
- **Features**: 59

#### short_mr_clf
- **Features**: 146

#### short_tf_H2
- **Features**: 64

#### short_tf_H4
- **Features**: 60

#### short_tf_H8
- **Features**: 61

#### short_tf_clf
- **Features**: 146

## Specialist Models
- **Trap (GMM)**: 8 features, clusters=GaussianMixture(covariance_type='diag', max_iter=200, n_components=4, n_init=3,
                random_state=42)
- **Gamma (ExtraTrees)**: 20 features
