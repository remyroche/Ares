# Training Report — 20260204_220000
Generated: 2026-02-09 23:45 UTC

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
| exh_down | 127,433 | 34 |
| exh_up | 27,924 | 34 |
| gamma_model | 2,445,067 | 28 |
| spike_anatomy_best | 8,891 | 12 |
| spike_anatomy_worst | 10,496 | 12 |
| train_long_mr_2 | 1,675 | 681 |
| train_long_mr_4 | 1,753 | 681 |
| train_long_mr_8 | 1,574 | 681 |
| train_long_tf_2 | 1,441 | 665 |
| train_long_tf_4 | 1,497 | 665 |
| train_long_tf_8 | 1,377 | 665 |
| train_short_mr_2 | 1,464 | 681 |
| train_short_mr_4 | 1,474 | 681 |
| train_short_mr_8 | 1,369 | 681 |
| train_short_tf_2 | 1,669 | 665 |
| train_short_tf_4 | 1,744 | 665 |
| train_short_tf_8 | 1,547 | 665 |
| trap_model | 7,981,262 | 11 |

## Alpha Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Prec@10 | Prec@40 | AvgTr/day@10 | AvgTr/day@30 | Avg Return | Trades | Best Iter |
|-------|----------|-----|----|---------| ---------|---------|---------|-------------|-------------|------------|--------|-----------|
| LONG_MR | 48 | 0.5740 | 0.0923 | -0.0521 | nan% | 0.5044 | 0.4935 | 2.5182 | 2.8411 | nan | N/A | N/A |
| LONG_TF | 48 | 0.5623 | 0.1559 | -0.0182 | nan% | 0.4887 | 0.5216 | 2.4496 | 2.4890 | nan | N/A | N/A |
| SHORT_MR | 48 | 0.5672 | 0.1735 | 0.0266 | nan% | 0.5945 | 0.5312 | 2.3481 | 2.3836 | nan | N/A | N/A |
| SHORT_TF | 48 | 0.6257 | 0.1996 | 0.0983 | nan% | 0.7435 | 0.6280 | 2.3846 | 2.6813 | nan | N/A | N/A |

### Detailed Model Performance

#### LONG_MR
- **Features**: 48
- **OOF AUC**: 0.5740
- **OOF IC**: 0.0923
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.0521
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.5044
- **OOF Prec@40**: 0.4935
- **OOF Avg Trades/Day @10%**: 2.5182
- **OOF Avg Trades/Day @30%**: 2.8411
- **OOF ECE@10**: 0.0342
- **OOF Calibration Profile**: overconfident

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0119 / 0.2352 / 0.5740 / 584 | 0.0267 / 0.2253 / 0.5989 / 585 | 0.0174 / 0.2379 / 0.5814 / 584 |
| vol_48h | 0.0203 / 0.2261 / 0.6102 / 584 | 0.0197 / 0.2375 / 0.5720 / 585 | 0.0150 / 0.2348 / 0.5614 / 584 |
| volume_12h | 0.0262 / 0.2348 / 0.5932 / 584 | 0.0239 / 0.2302 / 0.5911 / 585 | 0.0081 / 0.2333 / 0.5521 / 584 |
| volume_48h | 0.0162 / 0.2418 / 0.5783 / 584 | 0.0258 / 0.2317 / 0.5919 / 585 | -0.0010 / 0.2249 / 0.5347 / 584 |
| trend_12h | 0.0021 / 0.2458 / 0.5620 / 651 | 0.0101 / 0.2250 / 0.5732 / 518 | 0.0301 / 0.2252 / 0.6020 / 584 |
| trend_48h | 0.0090 / 0.2451 / 0.5966 / 584 | 0.0073 / 0.2321 / 0.5556 / 585 | 0.0202 / 0.2212 / 0.6080 / 584 |

- **Top features**: retrace_12_G_TREND_1, dist_ema_fast_base_G_TREND_1, excess_12h_G_VOL_0, retrace_12_G_VOL_1, excess_coh_G_TREND_1, mae_4h_G_TREND_1, volume_price_corr_10h_G_TREND_1, dist_vwap_norm_G_VOL_0, dlog_vol_5h_G_TREND_1, dist_ema_fast_base_G_VOL_0


#### LONG_TF
- **Features**: 48
- **OOF AUC**: 0.5623
- **OOF IC**: 0.1559
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.0182
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.4887
- **OOF Prec@40**: 0.5216
- **OOF Avg Trades/Day @10%**: 2.4496
- **OOF Avg Trades/Day @30%**: 2.4890
- **OOF ECE@10**: 0.1241
- **OOF Calibration Profile**: well-calibrated

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0160 / 0.2241 / 0.5404 / 499 | 0.0020 / 0.2323 / 0.5555 / 499 | 0.0300 / 0.2084 / 0.5855 / 499 |
| vol_48h | 0.0103 / 0.2304 / 0.5301 / 499 | 0.0038 / 0.2253 / 0.5506 / 434 | 0.0318 / 0.2110 / 0.5923 / 564 |
| volume_12h | 0.0064 / 0.2280 / 0.5244 / 499 | -0.0109 / 0.2461 / 0.5279 / 499 | -0.0001 / 0.1907 / 0.5921 / 499 |
| volume_48h | -0.0027 / 0.2318 / 0.5048 / 499 | 0.0228 / 0.2182 / 0.5799 / 499 | 0.0324 / 0.2148 / 0.5951 / 499 |
| trend_12h | 0.0056 / 0.2227 / 0.5305 / 499 | 0.0156 / 0.2259 / 0.5604 / 499 | 0.0319 / 0.2162 / 0.5988 / 499 |
| trend_48h | 0.0038 / 0.2256 / 0.5206 / 499 | 0.0193 / 0.2321 / 0.5811 / 499 | 0.0152 / 0.2070 / 0.5743 / 499 |

- **Top features**: rsi_slope_base_G_TREND_1, rsi_slope_G_VOL_1, rsi_slope_G_TREND_1, prog_def_G_TREND_1, rsi_slope_base_G_VOL_1, meta_alignment_G_TREND_1, v_power_G_TREND_1, retest_accept_G_TREND_1, evr_6_G_TREND_1, vol_asym_6_G_TREND_1


#### SHORT_MR
- **Features**: 48
- **OOF AUC**: 0.5672
- **OOF IC**: 0.1735
- **OOF Rank IC**: nan
- **OOF Sharpe**: 0.0266
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.5945
- **OOF Prec@40**: 0.5312
- **OOF Avg Trades/Day @10%**: 2.3481
- **OOF Avg Trades/Day @30%**: 2.3836
- **OOF ECE@10**: 0.0576
- **OOF Calibration Profile**: overconfident

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0119 / 0.2438 / 0.5635 / 456 | 0.0088 / 0.2342 / 0.5442 / 457 | -0.0064 / 0.2124 / 0.5239 / 456 |
| vol_48h | 0.0064 / 0.2447 / 0.5410 / 456 | 0.0152 / 0.2328 / 0.5689 / 386 | 0.0039 / 0.2156 / 0.5448 / 527 |
| volume_12h | 0.0057 / 0.2467 / 0.5604 / 456 | 0.0191 / 0.2148 / 0.5878 / 457 | -0.0085 / 0.2290 / 0.5032 / 456 |
| volume_48h | 0.0119 / 0.2427 / 0.5696 / 456 | 0.0086 / 0.2299 / 0.5259 / 457 | 0.0133 / 0.2178 / 0.5746 / 456 |
| trend_12h | 0.0114 / 0.2442 / 0.5671 / 456 | 0.0094 / 0.2353 / 0.5457 / 453 | -0.0138 / 0.2111 / 0.5424 / 460 |
| trend_48h | 0.0180 / 0.2407 / 0.5786 / 456 | 0.0069 / 0.2342 / 0.5319 / 457 | 0.0058 / 0.2156 / 0.5648 / 456 |

- **Top features**: speed_G_TREND_1, trend_snr_G_TREND_1, speed_G_VOL_1, retrace_12_G_TREND_1, spike_score_G_TREND_1, vol_z_G_TREND_1, mae_4h_G_VOL_1, vov_iqr_20_G_TREND_1, excess_coh_G_VOL_1, vol_range_shock_G_TREND_1


#### SHORT_TF
- **Features**: 48
- **OOF AUC**: 0.6257
- **OOF IC**: 0.1996
- **OOF Rank IC**: nan
- **OOF Sharpe**: 0.0983
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.7435
- **OOF Prec@40**: 0.6280
- **OOF Avg Trades/Day @10%**: 2.3846
- **OOF Avg Trades/Day @30%**: 2.6813
- **OOF ECE@10**: 0.0868
- **OOF Calibration Profile**: well-calibrated

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0322 / 0.2411 / 0.6164 / 515 | 0.0427 / 0.2370 / 0.6111 / 517 | 0.0145 / 0.2211 / 0.5845 / 515 |
| vol_48h | 0.0484 / 0.2373 / 0.6285 / 515 | 0.0341 / 0.2381 / 0.6040 / 517 | 0.0234 / 0.2237 / 0.5892 / 515 |
| volume_12h | 0.0132 / 0.2443 / 0.6077 / 515 | 0.0196 / 0.2385 / 0.5746 / 517 | 0.0529 / 0.2164 / 0.6506 / 515 |
| volume_48h | 0.0199 / 0.2446 / 0.5858 / 515 | 0.0792 / 0.2243 / 0.6652 / 517 | 0.0554 / 0.2303 / 0.6248 / 515 |
| trend_12h | 0.0123 / 0.2098 / 0.6288 / 552 | -0.0078 / 0.2519 / 0.5466 / 480 | 0.0332 / 0.2404 / 0.6279 / 515 |
| trend_48h | 0.0556 / 0.2179 / 0.6412 / 515 | 0.0322 / 0.2404 / 0.6093 / 517 | 0.0367 / 0.2408 / 0.6100 / 515 |

- **Top features**: rvol_hod_base_G_TREND_1, ret1h_z_G_TREND_1, evr_6_G_TREND_1, rv_12h_G_VOL_1, rv_24h_G_TREND_1, accel_5h_G_VOL_1, vov_mad_60_G_TREND_1, vol_expansion_ratio_G_TREND_1, volu_z_G_VOL_0, trend_snr_G_VOL_1

## Meta Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Calibration |
|-------|----------|-----|----|---------| ---------|-------------|
| long_mr | 10 | N/A | N/A | N/A | N/A | N/A |
| long_tf | 10 | N/A | N/A | N/A | N/A | N/A |
| short_mr | 10 | N/A | N/A | N/A | N/A | N/A |
| short_tf | 10 | N/A | N/A | N/A | N/A | N/A |

### Detailed Meta Model Performance

#### long_mr
- **Features**: 10

#### long_tf
- **Features**: 10

#### short_mr
- **Features**: 10

#### short_tf
- **Features**: 10

## Specialist Models
- **Trap (GMM)**: 8 features, clusters=GaussianMixture(covariance_type='diag', max_iter=200, n_components=4, n_init=3,
                random_state=42)
- **Gamma (ExtraTrees)**: 20 features
