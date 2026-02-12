# Training Report — 20260212_160000
Generated: 2026-02-12 17:39 UTC

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
| spike_anatomy_best | 9,625 | 12 |
| spike_anatomy_worst | 11,082 | 12 |
| train_long_mr_2 | 2,372 | 614 |
| train_long_mr_4 | 2,372 | 614 |
| train_long_mr_8 | 2,372 | 614 |
| train_long_tf_2 | 1,896 | 606 |
| train_long_tf_4 | 1,896 | 606 |
| train_long_tf_8 | 1,896 | 606 |
| train_short_mr_2 | 1,896 | 614 |
| train_short_mr_4 | 1,896 | 614 |
| train_short_mr_8 | 1,896 | 614 |
| train_short_tf_2 | 2,372 | 606 |
| train_short_tf_4 | 2,372 | 606 |
| train_short_tf_8 | 2,372 | 606 |
| trap_model | 8,688,943 | 11 |

## Alpha Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Prec@10 | Prec@40 | AvgTr/day@10 | AvgTr/day@30 | Avg Return | Trades | Best Iter |
|-------|----------|-----|----|---------| ---------|---------|---------|-------------|-------------|------------|--------|-----------|
| LONG_MR | 48 | 0.5431 | 0.1095 | -0.0768 | nan% | 0.4801 | 0.4508 | 2.8005 | 3.2898 | nan | N/A | N/A |
| LONG_TF | 48 | 0.5941 | 0.1771 | -0.2206 | nan% | 0.5756 | 0.4653 | 2.6533 | 2.7275 | nan | N/A | N/A |
| SHORT_MR | 48 | 0.5785 | 0.1441 | -0.1813 | nan% | 0.4691 | 0.4215 | 2.6533 | 2.7275 | nan | N/A | N/A |
| SHORT_TF | 48 | 0.5745 | 0.1685 | -0.0319 | nan% | 0.5646 | 0.4886 | 2.8005 | 3.2898 | nan | N/A | N/A |

### Detailed Model Performance

#### LONG_MR
- **Features**: 48
- **OOF AUC**: 0.5431
- **OOF IC**: 0.1095
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.0768
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.4801
- **OOF Prec@40**: 0.4508
- **OOF Avg Trades/Day @10%**: 2.8005
- **OOF Avg Trades/Day @30%**: 3.2898
- **OOF ECE@10**: 0.0315
- **OOF Calibration Profile**: overconfident

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0027 / 0.2109 / 0.5357 / 790 | 0.0003 / 0.1954 / 0.5527 / 792 | 0.0051 / 0.2201 / 0.5566 / 790 |
| vol_48h | 0.0077 / 0.2058 / 0.5668 / 790 | 0.0089 / 0.2042 / 0.5464 / 792 | -0.0001 / 0.2163 / 0.5236 / 790 |
| volume_12h | 0.0012 / 0.2012 / 0.5460 / 790 | 0.0099 / 0.2061 / 0.5635 / 792 | 0.0020 / 0.2190 / 0.5350 / 790 |
| volume_48h | 0.0026 / 0.2074 / 0.5280 / 790 | 0.0135 / 0.2098 / 0.5830 / 792 | 0.0015 / 0.2092 / 0.5254 / 790 |
| trend_12h | -0.0188 / 0.2393 / 0.5330 / 804 | 0.0040 / 0.1997 / 0.5434 / 778 | -0.0113 / 0.1867 / 0.5711 / 790 |
| trend_48h | -0.0040 / 0.2286 / 0.5578 / 790 | -0.0009 / 0.1985 / 0.5293 / 792 | 0.0054 / 0.1993 / 0.5640 / 790 |

- **Top features**: mfe_4h_G_VOL_1, ft_drop_8_G_VOL_1, dist_ema_fast_base_G_VOL_0, retrace_12_G_VOL_1, trend_snr_G_VOL_0, meta_alignment_G_VOL_0, excess_coh_G_VOL_0, dist_stack_G_VOL_0, climax_decay_G_VOL_1, accel_5h_G_VOL_1


#### LONG_TF
- **Features**: 48
- **OOF AUC**: 0.5941
- **OOF IC**: 0.1771
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.2206
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.5756
- **OOF Prec@40**: 0.4653
- **OOF Avg Trades/Day @10%**: 2.6533
- **OOF Avg Trades/Day @30%**: 2.7275
- **OOF ECE@10**: 0.1018
- **OOF Calibration Profile**: overconfident

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0074 / 0.2130 / 0.5530 / 632 | 0.0365 / 0.2038 / 0.6146 / 632 | 0.0411 / 0.1953 / 0.6141 / 632 |
| vol_48h | 0.0193 / 0.2127 / 0.5726 / 632 | 0.0169 / 0.2156 / 0.5828 / 557 | 0.0383 / 0.1871 / 0.6151 / 707 |
| volume_12h | -0.0001 / 0.2240 / 0.5499 / 632 | 0.0255 / 0.2152 / 0.5988 / 632 | 0.0313 / 0.1729 / 0.6094 / 632 |
| volume_48h | 0.0034 / 0.2144 / 0.5580 / 632 | 0.0236 / 0.2041 / 0.5893 / 632 | 0.0589 / 0.1936 / 0.6343 / 632 |
| trend_12h | 0.0224 / 0.2179 / 0.5881 / 632 | 0.0214 / 0.1980 / 0.5799 / 632 | 0.0338 / 0.1961 / 0.5988 / 632 |
| trend_48h | 0.0111 / 0.2215 / 0.5639 / 632 | 0.0315 / 0.2128 / 0.6078 / 632 | 0.0131 / 0.1778 / 0.5789 / 632 |

- **Top features**: range_pct_G_VOL_0, rv_2h_G_VOL_0, sin_hod_G_VOL_1, rsi_1h_slope_G_VOL_0, rv_8h_G_VOL_0, G_LIQ_EXCEL_G_VOL_0, ret1h_G_VOL_1, body_pct_G_VOL_0, accept_G_VOL_0, up_vol_6_G_VOL_1


#### SHORT_MR
- **Features**: 48
- **OOF AUC**: 0.5785
- **OOF IC**: 0.1441
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.1813
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.4691
- **OOF Prec@40**: 0.4215
- **OOF Avg Trades/Day @10%**: 2.6533
- **OOF Avg Trades/Day @30%**: 2.7275
- **OOF ECE@10**: 0.0749
- **OOF Calibration Profile**: overconfident

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0160 / 0.2233 / 0.5746 / 632 | 0.0199 / 0.2029 / 0.5720 / 632 | 0.0095 / 0.1903 / 0.5698 / 632 |
| vol_48h | 0.0139 / 0.2203 / 0.5700 / 632 | 0.0282 / 0.2026 / 0.5804 / 557 | 0.0132 / 0.1946 / 0.5770 / 707 |
| volume_12h | 0.0035 / 0.2242 / 0.5516 / 632 | 0.0249 / 0.2032 / 0.5845 / 632 | 0.0230 / 0.1892 / 0.5937 / 632 |
| volume_48h | 0.0098 / 0.2197 / 0.5572 / 632 | 0.0296 / 0.2052 / 0.5868 / 632 | 0.0144 / 0.1916 / 0.5814 / 632 |
| trend_12h | 0.0133 / 0.2129 / 0.5721 / 632 | 0.0255 / 0.2177 / 0.5858 / 632 | 0.0040 / 0.1860 / 0.5734 / 632 |
| trend_48h | 0.0170 / 0.2126 / 0.5810 / 632 | 0.0283 / 0.2096 / 0.5757 / 632 | 0.0118 / 0.1943 / 0.5773 / 632 |

- **Top features**: meta_abs_net_x_breakout_G_VOL_0, mae_4h_G_VOL_0, evr_slope_G_VOL_0, rsi_1h_slope_G_VOL_0, clv_mean_4_G_VOL_0, mae_4h_G_VOL_1, grind_score_G_VOL_0, clv_mean_2_G_VOL_0, sin_hod_G_VOL_1, volume_price_corr_10h_G_VOL_0


#### SHORT_TF
- **Features**: 48
- **OOF AUC**: 0.5745
- **OOF IC**: 0.1685
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.0319
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.5646
- **OOF Prec@40**: 0.4886
- **OOF Avg Trades/Day @10%**: 2.8005
- **OOF Avg Trades/Day @30%**: 3.2898
- **OOF ECE@10**: 0.0181
- **OOF Calibration Profile**: overconfident

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0164 / 0.2224 / 0.5826 / 790 | 0.0077 / 0.2081 / 0.5374 / 792 | 0.0191 / 0.1867 / 0.5723 / 790 |
| vol_48h | 0.0114 / 0.2255 / 0.5526 / 790 | 0.0140 / 0.2087 / 0.5526 / 792 | 0.0088 / 0.1830 / 0.5614 / 790 |
| volume_12h | 0.0097 / 0.2228 / 0.5550 / 790 | 0.0046 / 0.2030 / 0.5404 / 792 | 0.0364 / 0.1914 / 0.6084 / 790 |
| volume_48h | 0.0069 / 0.2028 / 0.5520 / 790 | 0.0225 / 0.2035 / 0.5759 / 792 | 0.0292 / 0.2109 / 0.5966 / 790 |
| trend_12h | 0.0079 / 0.1758 / 0.5833 / 804 | 0.0160 / 0.2164 / 0.5679 / 778 | 0.0018 / 0.2257 / 0.5390 / 790 |
| trend_48h | 0.0169 / 0.1834 / 0.5799 / 790 | 0.0193 / 0.2105 / 0.5691 / 792 | 0.0056 / 0.2233 / 0.5512 / 790 |

- **Top features**: meta_alignment_G_VOL_1, stage_tf_G_VOL_1, cos_hod_G_VOL_1, asym_ft_G_VOL_1, rv_12h_G_VOL_1, ft_drop_8_G_VOL_1, accel_5h_G_VOL_1, trend_snr_G_VOL_1, accept_G_VOL_1, vol_shock_asym_4_12_G_VOL_1

## Specialist Models
- **Trap (GMM)**: 8 features, clusters=GaussianMixture(covariance_type='diag', max_iter=200, n_components=4, n_init=3,
                random_state=42)
- **Gamma (ExtraTrees)**: 20 features
