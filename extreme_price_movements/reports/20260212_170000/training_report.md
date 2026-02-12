# Training Report — 20260212_170000
Generated: 2026-02-12 19:01 UTC

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
| LONG_MR | 48 | 0.5375 | 0.1147 | -0.1449 | nan% | 0.4854 | 0.4259 | 2.8695 | 3.4079 | nan | N/A | N/A |
| LONG_TF | 48 | 0.5379 | 0.0729 | -0.0463 | nan% | 0.3724 | 0.4299 | 2.6974 | 2.7800 | nan | N/A | N/A |
| SHORT_MR | 48 | 0.5769 | 0.1204 | -0.0478 | nan% | 0.5334 | 0.4712 | 2.6974 | 2.7800 | nan | N/A | N/A |
| SHORT_TF | 48 | 0.5533 | 0.1325 | -0.1042 | nan% | 0.5386 | 0.4858 | 2.8695 | 3.4079 | nan | N/A | N/A |

### Detailed Model Performance

#### LONG_MR
- **Features**: 48
- **OOF AUC**: 0.5375
- **OOF IC**: 0.1147
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.1449
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.4854
- **OOF Prec@40**: 0.4259
- **OOF Avg Trades/Day @10%**: 2.8695
- **OOF Avg Trades/Day @30%**: 3.4079
- **OOF ECE@10**: 0.0727
- **OOF Calibration Profile**: overconfident

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0098 / 0.2125 / 0.5590 / 853 | -0.0018 / 0.1993 / 0.5325 / 854 | 0.0058 / 0.2142 / 0.5250 / 853 |
| vol_48h | 0.0057 / 0.2031 / 0.5576 / 853 | 0.0078 / 0.2128 / 0.5503 / 854 | 0.0040 / 0.2101 / 0.5082 / 853 |
| volume_12h | 0.0001 / 0.2037 / 0.5345 / 853 | -0.0016 / 0.2093 / 0.5094 / 854 | 0.0174 / 0.2130 / 0.5745 / 853 |
| volume_48h | 0.0034 / 0.2139 / 0.5365 / 853 | -0.0006 / 0.2037 / 0.5149 / 854 | 0.0142 / 0.2084 / 0.5619 / 853 |
| trend_12h | -0.0113 / 0.2352 / 0.5445 / 875 | 0.0121 / 0.2018 / 0.5564 / 832 | -0.0205 / 0.1882 / 0.5400 / 853 |
| trend_48h | -0.0099 / 0.2310 / 0.5193 / 853 | 0.0077 / 0.1979 / 0.5591 / 854 | 0.0017 / 0.1971 / 0.5520 / 853 |

- **Top features**: mfe_4h_G_VOL_1, ft_8_G_VOL_1, thrust_decay_8_G_VOL_1, pullback_8_G_VOL_1, G_EXH_GIVEBACK_G_VOL_1, mae_4h_G_VOL_1, mfe_8h_G_VOL_1, volume_price_corr_10h_G_VOL_1, reject_dir2h_abs_prod_G_VOL_1, accel_5h_G_VOL_1


#### LONG_TF
- **Features**: 48
- **OOF AUC**: 0.5379
- **OOF IC**: 0.0729
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.0463
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.3724
- **OOF Prec@40**: 0.4299
- **OOF Avg Trades/Day @10%**: 2.6974
- **OOF Avg Trades/Day @30%**: 2.7800
- **OOF ECE@10**: 0.1383
- **OOF Calibration Profile**: overconfident

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0022 / 0.2126 / 0.5198 / 663 | 0.0098 / 0.2117 / 0.5553 / 665 | 0.0053 / 0.2021 / 0.5419 / 663 |
| vol_48h | 0.0028 / 0.2141 / 0.5352 / 663 | 0.0113 / 0.2082 / 0.5429 / 597 | 0.0045 / 0.2044 / 0.5361 / 731 |
| volume_12h | 0.0054 / 0.2046 / 0.5396 / 663 | -0.0006 / 0.2250 / 0.5308 / 665 | 0.0019 / 0.1967 / 0.5492 / 663 |
| volume_48h | 0.0090 / 0.2076 / 0.5515 / 663 | 0.0031 / 0.2041 / 0.5286 / 665 | 0.0053 / 0.2147 / 0.5318 / 663 |
| trend_12h | 0.0063 / 0.2057 / 0.5401 / 663 | 0.0071 / 0.2045 / 0.5504 / 665 | 0.0038 / 0.2161 / 0.5191 / 663 |
| trend_48h | 0.0065 / 0.2063 / 0.5440 / 663 | 0.0128 / 0.2144 / 0.5592 / 665 | -0.0030 / 0.2057 / 0.5090 / 663 |

- **Top features**: stall_ext_corr_G_VOL_0, dist_ema_slow_base_G_VOL_0, G_MR_SPIKE_G_VOL_1, progress_G_VOL_0, ft_2_G_VOL_0, ft_drop_G_VOL_0, breakout_confirmed_G_VOL_0, spike_score_G_VOL_0, amihud_z_G_VOL_0, asym_ft_G_VOL_0


#### SHORT_MR
- **Features**: 48
- **OOF AUC**: 0.5769
- **OOF IC**: 0.1204
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.0478
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.5334
- **OOF Prec@40**: 0.4712
- **OOF Avg Trades/Day @10%**: 2.6974
- **OOF Avg Trades/Day @30%**: 2.7800
- **OOF ECE@10**: 0.0752
- **OOF Calibration Profile**: overconfident

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0246 / 0.2147 / 0.5913 / 663 | 0.0192 / 0.2020 / 0.5816 / 665 | 0.0056 / 0.2020 / 0.5506 / 663 |
| vol_48h | 0.0273 / 0.2116 / 0.5962 / 663 | 0.0229 / 0.2090 / 0.5895 / 597 | 0.0006 / 0.1992 / 0.5434 / 731 |
| volume_12h | 0.0126 / 0.2179 / 0.5554 / 663 | 0.0265 / 0.2029 / 0.6033 / 665 | 0.0095 / 0.1979 / 0.5631 / 663 |
| volume_48h | 0.0201 / 0.2121 / 0.5830 / 663 | 0.0237 / 0.2093 / 0.5876 / 665 | 0.0053 / 0.1974 / 0.5555 / 663 |
| trend_12h | 0.0072 / 0.2231 / 0.5671 / 663 | 0.0226 / 0.2026 / 0.5828 / 665 | 0.0128 / 0.1932 / 0.5794 / 663 |
| trend_48h | 0.0172 / 0.2179 / 0.5773 / 663 | 0.0154 / 0.2003 / 0.5744 / 665 | 0.0156 / 0.2006 / 0.5776 / 663 |

- **Top features**: speed_G_VOL_0, meta_abs_net_x_breakout_G_VOL_0, mae_8h_G_VOL_1, mfe_8h_G_VOL_0, speed_G_VOL_1, cos_hod_G_VOL_0, mae_8h_G_VOL_0, coherence_24_G_VOL_0, mae_4h_G_VOL_1, vol_range_shock_G_VOL_0


#### SHORT_TF
- **Features**: 48
- **OOF AUC**: 0.5533
- **OOF IC**: 0.1325
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.1042
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.5386
- **OOF Prec@40**: 0.4858
- **OOF Avg Trades/Day @10%**: 2.8695
- **OOF Avg Trades/Day @30%**: 3.4079
- **OOF ECE@10**: 0.0905
- **OOF Calibration Profile**: well-calibrated

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0161 / 0.2157 / 0.5592 / 853 | 0.0002 / 0.2179 / 0.5187 / 854 | 0.0091 / 0.1880 / 0.5596 / 853 |
| vol_48h | 0.0183 / 0.2221 / 0.5661 / 853 | -0.0001 / 0.2056 / 0.5232 / 854 | 0.0044 / 0.1939 / 0.5406 / 853 |
| volume_12h | 0.0032 / 0.2218 / 0.5270 / 853 | -0.0023 / 0.2061 / 0.5090 / 854 | 0.0319 / 0.1937 / 0.6134 / 853 |
| volume_48h | -0.0073 / 0.2052 / 0.5074 / 853 | 0.0234 / 0.2031 / 0.5719 / 854 | 0.0201 / 0.2132 / 0.5767 / 853 |
| trend_12h | 0.0004 / 0.1829 / 0.5755 / 875 | 0.0089 / 0.2138 / 0.5387 / 832 | 0.0027 / 0.2256 / 0.5217 / 853 |
| trend_48h | 0.0117 / 0.1919 / 0.5839 / 853 | 0.0035 / 0.2128 / 0.5113 / 854 | 0.0148 / 0.2168 / 0.5560 / 853 |

- **Top features**: meta_alignment_G_VOL_1, accept_G_VOL_1, evr_6_G_VOL_1, vol_shock_asym_4_12_G_VOL_1, ft_2_G_VOL_1, momentum_accel_G_VOL_1, stage_tf_G_VOL_1, rvol_hod_base_G_VOL_1, cos_hod_G_VOL_1, decel_8_G_VOL_1

## Specialist Models
- **Trap (GMM)**: 8 features, clusters=GaussianMixture(covariance_type='diag', max_iter=200, n_components=4, n_init=3,
                random_state=42)
- **Gamma (ExtraTrees)**: 20 features
