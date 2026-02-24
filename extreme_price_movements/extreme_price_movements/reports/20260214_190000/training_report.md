# Training Report — 20260214_190000
Generated: 2026-02-24 03:02 UTC

## Configuration
- **Train lookback**: 35040 hours
- **Label horizons**: [2, 4, 8]
- **Label method**: triple_barrier
- **Label quantiles**: lo=0.3, hi=0.65
- **OOS holdout**: 14 days
- **Min train samples**: 200
- **Feature selection**: MDI (min=30, cap=0.995)
- **15m precision**: True

## Dataset Sizes
| Dataset | Rows | Features |
|---------|------|----------|
| gamma_model | 10,967,338 | 28 |
| spike_anatomy_best | 86,387 | 12 |
| spike_anatomy_worst | 99,877 | 12 |
| train_long_mr_2 | 113,965 | 1132 |
| train_long_mr_4 | 113,965 | 1132 |
| train_long_mr_8 | 113,965 | 1132 |
| train_long_tf_2 | 54,247 | 1126 |
| train_long_tf_4 | 54,247 | 1126 |
| train_long_tf_8 | 54,247 | 1126 |
| train_short_mr_2 | 54,247 | 1132 |
| train_short_mr_4 | 54,247 | 1132 |
| train_short_mr_8 | 54,247 | 1132 |
| train_short_tf_2 | 113,965 | 1126 |
| train_short_tf_4 | 113,965 | 1126 |
| train_short_tf_8 | 113,965 | 1126 |
| trap_model | 5,924,353 | 11 |

## Alpha Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Prec@10 | Prec@40 | AvgTr/day@10 | AvgTr/day@30 | Avg Return | Trades | Best Iter |
|-------|----------|-----|----|---------| ---------|---------|---------|-------------|-------------|------------|--------|-----------|
| LONG_MR | 48 | 0.5361 | 0.0340 | -0.1034 | nan% | 0.8091 | 0.7944 | 8.4485 | 25.3447 | nan | N/A | N/A |
| LONG_TF | 48 | 0.5209 | 0.0281 | -0.1470 | nan% | 0.7898 | 0.7583 | 3.9541 | 11.8622 | nan | N/A | N/A |
| SHORT_MR | 48 | 0.5510 | 0.0678 | -0.1301 | nan% | 0.7887 | 0.7677 | 3.9541 | 11.8622 | nan | N/A | N/A |
| SHORT_TF | 48 | 0.5206 | 0.0260 | -0.1138 | nan% | 0.7984 | 0.7899 | 8.4485 | 25.3447 | nan | N/A | N/A |

### Detailed Model Performance

#### LONG_MR
- **Features**: 48
- **OOF AUC**: 0.5361
- **OOF IC**: 0.0340
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.1034
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.8091
- **OOF Prec@40**: 0.7944
- **OOF Avg Trades/Day @10%**: 8.4485
- **OOF Avg Trades/Day @30%**: 25.3447
- **OOF ECE@10**: 0.0017
- **OOF Calibration Profile**: well-calibrated

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0050 / 0.1871 / 0.5447 / 37951 | 0.0036 / 0.1784 / 0.5309 / 38063 | 0.0018 / 0.1731 / 0.5303 / 37951 |
| vol_48h | 0.0036 / 0.1825 / 0.5341 / 37951 | 0.0032 / 0.1803 / 0.5323 / 38063 | 0.0054 / 0.1757 / 0.5402 / 37951 |
| volume_12h | 0.0042 / 0.1843 / 0.5367 / 37951 | 0.0044 / 0.1832 / 0.5358 / 38063 | 0.0017 / 0.1711 / 0.5347 / 37951 |
| volume_48h | 0.0036 / 0.1878 / 0.5351 / 37951 | 0.0048 / 0.1824 / 0.5401 / 38063 | -0.0003 / 0.1683 / 0.5303 / 37951 |
| trend_12h | 0.0027 / 0.1861 / 0.5303 / 37951 | 0.0030 / 0.1834 / 0.5317 / 38063 | 0.0038 / 0.1691 / 0.5400 / 37951 |
| trend_48h | 0.0018 / 0.1810 / 0.5265 / 37951 | 0.0047 / 0.1827 / 0.5395 / 38063 | 0.0056 / 0.1749 / 0.5430 / 37951 |

- **Top features**: ret8h_G_VOL_1, hurst_proxy_x_regime_trend_48h_G_VOL_1, atr_pct_G_VOL_0, ret72h_G_VOL_1, dist_ema_slow_base_G_VOL_1, ffd_rv_24h_04_G_VOL_1, churn_G_VOL_1, kf_score_rm24_mean_G_VOL_1, ret48h_G_VOL_1, cos_hod_G_VOL_1


#### LONG_TF
- **Features**: 48
- **OOF AUC**: 0.5209
- **OOF IC**: 0.0281
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.1470
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.7898
- **OOF Prec@40**: 0.7583
- **OOF Avg Trades/Day @10%**: 3.9541
- **OOF Avg Trades/Day @30%**: 11.8622
- **OOF ECE@10**: 0.0187
- **OOF Calibration Profile**: well-calibrated

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | -0.0028 / 0.2130 / 0.5107 / 18064 | 0.0015 / 0.1991 / 0.5184 / 18119 | -0.0084 / 0.1773 / 0.5003 / 18064 |
| vol_48h | -0.0063 / 0.2177 / 0.5090 / 18064 | 0.0013 / 0.1948 / 0.5164 / 18119 | -0.0093 / 0.1769 / 0.5002 / 18064 |
| volume_12h | -0.0016 / 0.2092 / 0.5064 / 18064 | 0.0010 / 0.1961 / 0.5159 / 18119 | -0.0001 / 0.1840 / 0.5242 / 18064 |
| volume_48h | 0.0006 / 0.2021 / 0.5121 / 18064 | 0.0014 / 0.1988 / 0.5168 / 18119 | 0.0025 / 0.1885 / 0.5301 / 18064 |
| trend_12h | 0.0022 / 0.1971 / 0.5185 / 18064 | 0.0001 / 0.2059 / 0.5170 / 18119 | -0.0001 / 0.1864 / 0.5194 / 18064 |
| trend_48h | 0.0017 / 0.2020 / 0.5204 / 18064 | 0.0017 / 0.1995 / 0.5176 / 18119 | 0.0006 / 0.1879 / 0.5200 / 18064 |

- **Top features**: G_LIQ_GOOD_G_VOL_0, vol_state_G_VOL_1, thrust_decay_8_G_VOL_0, atr_state_G_VOL_0, vov_mad_20_G_VOL_0, vol_price_spread_G_VOL_0, vp_profile_concentration_G_VOL_0, pct_breakout_t_G_VOL_1, ffd_vol_price_corr_10h_04_G_VOL_1, vov_iqr_20_G_VOL_0


#### SHORT_MR
- **Features**: 48
- **OOF AUC**: 0.5510
- **OOF IC**: 0.0678
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.1301
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.7887
- **OOF Prec@40**: 0.7677
- **OOF Avg Trades/Day @10%**: 3.9541
- **OOF Avg Trades/Day @30%**: 11.8622
- **OOF ECE@10**: 0.0139
- **OOF Calibration Profile**: well-calibrated

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0066 / 0.2110 / 0.5442 / 18064 | 0.0060 / 0.1982 / 0.5414 / 18119 | 0.0000 / 0.1758 / 0.5259 / 18064 |
| vol_48h | 0.0047 / 0.2153 / 0.5451 / 18064 | 0.0044 / 0.1942 / 0.5365 / 18119 | -0.0013 / 0.1755 / 0.5231 / 18064 |
| volume_12h | 0.0089 / 0.2070 / 0.5508 / 18064 | 0.0065 / 0.1950 / 0.5409 / 18119 | 0.0059 / 0.1829 / 0.5431 / 18064 |
| volume_48h | 0.0050 / 0.2012 / 0.5422 / 18064 | 0.0108 / 0.1969 / 0.5526 / 18119 | 0.0111 / 0.1868 / 0.5536 / 18064 |
| trend_12h | 0.0096 / 0.1956 / 0.5515 / 18064 | 0.0086 / 0.2041 / 0.5493 / 18119 | 0.0061 / 0.1852 / 0.5423 / 18064 |
| trend_48h | 0.0107 / 0.2002 / 0.5549 / 18064 | 0.0078 / 0.1983 / 0.5444 / 18119 | 0.0077 / 0.1865 / 0.5468 / 18064 |

- **Top features**: vol_state_G_VOL_1, G_LIQ_GOOD_G_VOL_0, vol_price_spread_G_VOL_0, atr_state_G_VOL_0, vov_mad_20_G_VOL_0, cvar_5pct_G_VOL_0, vol_price_spread_G_VOL_1, vp_profile_concentration_G_VOL_0, mfe_4h_G_VOL_0, efficiency_G_VOL_0


#### SHORT_TF
- **Features**: 48
- **OOF AUC**: 0.5206
- **OOF IC**: 0.0260
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.1138
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.7984
- **OOF Prec@40**: 0.7899
- **OOF Avg Trades/Day @10%**: 8.4485
- **OOF Avg Trades/Day @30%**: 25.3447
- **OOF ECE@10**: 0.0050
- **OOF Calibration Profile**: well-calibrated

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0006 / 0.1879 / 0.5186 / 37951 | 0.0021 / 0.1787 / 0.5226 / 38063 | 0.0002 / 0.1734 / 0.5179 / 37951 |
| vol_48h | 0.0016 / 0.1829 / 0.5203 / 37951 | 0.0015 / 0.1807 / 0.5188 / 38063 | 0.0015 / 0.1764 / 0.5217 / 37951 |
| volume_12h | 0.0004 / 0.1850 / 0.5134 / 37951 | 0.0013 / 0.1838 / 0.5202 / 38063 | 0.0011 / 0.1712 / 0.5265 / 37951 |
| volume_48h | -0.0008 / 0.1886 / 0.5129 / 37951 | 0.0021 / 0.1829 / 0.5227 / 38063 | -0.0006 / 0.1684 / 0.5229 / 37951 |
| trend_12h | 0.0010 / 0.1864 / 0.5228 / 37951 | 0.0010 / 0.1838 / 0.5148 / 38063 | -0.0002 / 0.1698 / 0.5203 / 37951 |
| trend_48h | 0.0018 / 0.1810 / 0.5236 / 37951 | 0.0013 / 0.1833 / 0.5201 / 38063 | 0.0013 / 0.1757 / 0.5183 / 37951 |

- **Top features**: ret8h_G_VOL_1, ffd_rv_24h_04_G_VOL_1, churn_G_VOL_1, is_high_vol_regime_G_VOL_1, hurst_proxy_x_regime_trend_48h_G_VOL_1, cos_hod_G_VOL_1, body_pct_G_VOL_1, vp_in_hvn_above_zone_G_VOL_0, kf_score_rm24_mean_G_VOL_1, ffd_diff_8_04_G_VOL_1

## Meta Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Calibration |
|-------|----------|-----|----|---------| ---------|-------------|
| long_mr_H8 | 72 | N/A | N/A | N/A | N/A | N/A |
| long_mr_clf | 213 | N/A | N/A | N/A | N/A | N/A |
| long_mr_early_inval | 0 | N/A | N/A | N/A | N/A | N/A |
| long_tf_H8 | 72 | N/A | N/A | N/A | N/A | N/A |
| long_tf_clf | 213 | N/A | N/A | N/A | N/A | N/A |
| long_tf_early_inval | 0 | N/A | N/A | N/A | N/A | N/A |
| short_mr_H8 | 71 | N/A | N/A | N/A | N/A | N/A |
| short_mr_clf | 213 | N/A | N/A | N/A | N/A | N/A |
| short_mr_early_inval | 0 | N/A | N/A | N/A | N/A | N/A |
| short_tf_H8 | 76 | N/A | N/A | N/A | N/A | N/A |
| short_tf_clf | 213 | N/A | N/A | N/A | N/A | N/A |
| short_tf_early_inval | 0 | N/A | N/A | N/A | N/A | N/A |

### Detailed Meta Model Performance

#### long_mr_H8
- **Features**: 72

#### long_mr_clf
- **Features**: 213

#### long_mr_early_inval
- **Features**: 0

#### long_tf_H8
- **Features**: 72

#### long_tf_clf
- **Features**: 213

#### long_tf_early_inval
- **Features**: 0

#### short_mr_H8
- **Features**: 71

#### short_mr_clf
- **Features**: 213

#### short_mr_early_inval
- **Features**: 0

#### short_tf_H8
- **Features**: 76

#### short_tf_clf
- **Features**: 213

#### short_tf_early_inval
- **Features**: 0

## Specialist Models
- **Trap (GMM)**: 8 features, clusters=GaussianMixture(covariance_type='diag', max_iter=200, n_components=4, n_init=3,
                random_state=42)
- **Gamma (ExtraTrees)**: 20 features
