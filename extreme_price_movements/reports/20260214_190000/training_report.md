# Training Report — 20260214_190000
Generated: 2026-02-27 08:12 UTC

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
| gamma_model | 5,683,250 | 28 |
| spike_anatomy_best | 24,196 | 12 |
| spike_anatomy_worst | 35,503 | 12 |
| train_long_mr_2 | 44,443 | 1136 |
| train_long_mr_4 | 44,443 | 1136 |
| train_long_mr_8 | 44,443 | 1136 |
| train_long_tf_2 | 15,256 | 1130 |
| train_long_tf_4 | 15,256 | 1130 |
| train_long_tf_8 | 15,256 | 1130 |
| train_short_mr_2 | 15,256 | 1136 |
| train_short_mr_4 | 15,256 | 1136 |
| train_short_mr_8 | 15,256 | 1136 |
| train_short_tf_2 | 44,443 | 1130 |
| train_short_tf_4 | 44,443 | 1130 |
| train_short_tf_8 | 44,443 | 1130 |
| trap_model | 5,924,353 | 11 |

## Alpha Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Prec@10 | Prec@40 | AvgTr/day@10 | AvgTr/day@30 | Avg Return | Trades | Best Iter |
|-------|----------|-----|----|---------| ---------|---------|---------|-------------|-------------|------------|--------|-----------|
| LONG_MR | 48 | 0.5138 | 0.0126 | -0.0965 | nan% | 0.7525 | 0.7310 | 4.0594 | 12.1763 | nan | N/A | N/A |
| LONG_TF | 48 | 0.5120 | 0.0195 | -0.1053 | nan% | 0.7549 | 0.7337 | 1.4039 | 4.2107 | nan | N/A | N/A |
| SHORT_MR | 48 | 0.5139 | 0.0217 | -0.1038 | nan% | 0.7439 | 0.7215 | 1.4039 | 4.2107 | nan | N/A | N/A |
| SHORT_TF | 48 | 0.5291 | 0.0322 | -0.0879 | nan% | 0.7552 | 0.7265 | 4.0594 | 12.1763 | nan | N/A | N/A |

### Per-Horizon Alpha Performance (Quality Gate)
| Model | Winner | AUC | IC | LogLoss | PR-AUC | Lift@20 | BrierImp | Passed |
|-------|--------|-----|----|---------|--------|---------|----------|--------|
| long_mr_H2:extratrees | — | 0.4993 | -0.0052 | 0.6512 | 0.7152 | nan | nan | False |
| long_mr_H4:extratrees | — | 0.4993 | -0.0052 | 0.6512 | 0.7152 | nan | nan | False |
| long_mr_H8:extratrees | — | 0.4993 | -0.0052 | 0.6512 | 0.7152 | nan | nan | False |
| long_tf_H4:xgboost | — | 0.4948 | -0.0004 | 0.6530 | 0.7131 | nan | nan | False |
| long_tf_H2:lightgbm | — | 0.4901 | -0.0071 | 0.6542 | 0.7081 | nan | nan | False |
| short_mr_H8:xgboost | — | 0.4936 | -0.0002 | 0.6535 | 0.7131 | nan | nan | False |
| short_tf_H2:catboost | — | 0.4968 | -0.0068 | 0.6516 | 0.7141 | nan | nan | False |
| short_tf_H4:catboost | — | 0.4968 | -0.0068 | 0.6516 | 0.7141 | nan | nan | False |
| short_tf_H8:catboost | — | 0.4968 | -0.0068 | 0.6516 | 0.7141 | nan | nan | False |
| short_mr_H2:lightgbm | — | 0.4907 | -0.0085 | 0.6538 | 0.7093 | nan | nan | False |
| short_mr_H4:xgboost | — | 0.4913 | -0.0028 | 0.6540 | 0.7101 | nan | nan | False |
| long_tf_H8:lightgbm | — | 0.4901 | -0.0081 | 0.6539 | 0.7087 | nan | nan | False |
| short_mr_H2:catboost | — | 0.4940 | -0.0008 | 0.6533 | 0.7123 | nan | nan | False |
| long_tf_H8:xgboost | — | 0.4952 | 0.0007 | 0.6530 | 0.7142 | nan | nan | False |
| short_tf_H2:extratrees | — | 0.4998 | -0.0050 | 0.6512 | 0.7160 | nan | nan | False |
| short_tf_H4:extratrees | — | 0.4998 | -0.0050 | 0.6512 | 0.7160 | nan | nan | False |
| short_tf_H8:extratrees | — | 0.4998 | -0.0050 | 0.6512 | 0.7160 | nan | nan | False |
| short_mr_H8:extratrees | — | 0.4956 | 0.0022 | 0.6529 | 0.7150 | nan | nan | False |
| long_tf_H4:lightgbm | — | 0.4908 | -0.0071 | 0.6541 | 0.7096 | nan | nan | False |
| short_tf_H2:xgboost | — | 0.4982 | -0.0048 | 0.6516 | 0.7137 | nan | nan | False |
| short_tf_H4:xgboost | — | 0.4982 | -0.0048 | 0.6516 | 0.7137 | nan | nan | False |
| short_tf_H8:xgboost | — | 0.4982 | -0.0048 | 0.6516 | 0.7137 | nan | nan | False |
| long_tf_H2:xgboost | — | 0.4933 | 0.0009 | 0.6533 | 0.7120 | nan | nan | False |
| short_mr_H4:lightgbm | — | 0.4919 | -0.0080 | 0.6538 | 0.7098 | nan | nan | False |
| long_mr_H2:xgboost | — | 0.4972 | -0.0062 | 0.6516 | 0.7133 | nan | nan | False |
| long_mr_H4:xgboost | — | 0.4972 | -0.0062 | 0.6516 | 0.7133 | nan | nan | False |
| long_mr_H8:xgboost | — | 0.4972 | -0.0062 | 0.6516 | 0.7133 | nan | nan | False |
| long_tf_H2:extratrees | — | 0.4943 | 0.0030 | 0.6532 | 0.7148 | nan | nan | False |
| short_mr_H4:catboost | — | 0.4949 | 0.0009 | 0.6537 | 0.7091 | nan | nan | False |
| short_mr_H8:lightgbm | — | 0.4900 | -0.0086 | 0.6541 | 0.7085 | nan | nan | False |
| long_tf_H8:catboost | — | 0.4922 | -0.0012 | 0.6536 | 0.7103 | nan | nan | False |
| long_tf_H4:extratrees | — | 0.4946 | 0.0002 | 0.6531 | 0.7142 | nan | nan | False |
| short_mr_H4:extratrees | — | 0.4952 | 0.0035 | 0.6531 | 0.7144 | nan | nan | False |
| short_mr_H2:extratrees | — | 0.4948 | 0.0031 | 0.6532 | 0.7140 | nan | nan | False |
| long_tf_H8:extratrees | — | 0.4945 | 0.0009 | 0.6528 | 0.7145 | nan | nan | False |
| long_mr_H2:catboost | — | 0.4973 | -0.0062 | 0.6517 | 0.7142 | nan | nan | False |
| long_mr_H4:catboost | — | 0.4973 | -0.0062 | 0.6517 | 0.7142 | nan | nan | False |
| long_mr_H8:catboost | — | 0.4973 | -0.0062 | 0.6517 | 0.7142 | nan | nan | False |
| short_mr_H8:catboost | — | 0.4959 | 0.0039 | 0.6532 | 0.7123 | nan | nan | False |
| long_mr_H2:lightgbm | — | 0.4966 | -0.0060 | 0.6518 | 0.7139 | nan | nan | False |
| long_mr_H4:lightgbm | — | 0.4966 | -0.0060 | 0.6518 | 0.7139 | nan | nan | False |
| long_mr_H8:lightgbm | — | 0.4966 | -0.0060 | 0.6518 | 0.7139 | nan | nan | False |
| long_tf_H2:catboost | — | 0.4928 | 0.0012 | 0.6533 | 0.7113 | nan | nan | False |
| short_mr_H2:xgboost | — | 0.4943 | -0.0009 | 0.6534 | 0.7143 | nan | nan | False |
| short_tf_H2:lightgbm | — | 0.4964 | -0.0065 | 0.6518 | 0.7136 | nan | nan | False |
| short_tf_H4:lightgbm | — | 0.4964 | -0.0065 | 0.6518 | 0.7136 | nan | nan | False |
| short_tf_H8:lightgbm | — | 0.4964 | -0.0065 | 0.6518 | 0.7136 | nan | nan | False |
| long_tf_H4:catboost | — | 0.4917 | -0.0011 | 0.6535 | 0.7108 | nan | nan | False |

### Detailed Model Performance

#### LONG_MR
- **Features**: 48
- **OOF AUC**: 0.5138
- **OOF IC**: 0.0126
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.0965
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.7525
- **OOF Prec@40**: 0.7310
- **OOF Avg Trades/Day @10%**: 4.0594
- **OOF Avg Trades/Day @30%**: 12.1763
- **OOF ECE@10**: 0.0028
- **OOF Calibration Profile**: well-calibrated

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | -0.0007 / 0.2137 / 0.5081 / 14800 | 0.0009 / 0.2049 / 0.5153 / 14843 | 0.0003 / 0.2001 / 0.5121 / 14800 |
| vol_48h | -0.0000 / 0.2125 / 0.5136 / 14800 | 0.0012 / 0.2064 / 0.5150 / 14843 | -0.0004 / 0.1998 / 0.5075 / 14800 |
| volume_12h | 0.0001 / 0.2094 / 0.5082 / 14800 | 0.0004 / 0.2089 / 0.5129 / 14843 | 0.0014 / 0.2004 / 0.5194 / 14800 |
| volume_48h | -0.0004 / 0.2087 / 0.5062 / 14800 | 0.0006 / 0.2107 / 0.5168 / 14843 | 0.0009 / 0.1992 / 0.5175 / 14800 |
| trend_12h | 0.0006 / 0.2064 / 0.5138 / 14800 | 0.0010 / 0.2105 / 0.5172 / 14843 | 0.0003 / 0.2018 / 0.5094 / 14800 |
| trend_48h | 0.0008 / 0.2033 / 0.5152 / 14800 | 0.0008 / 0.2097 / 0.5177 / 14843 | 0.0008 / 0.2056 / 0.5081 / 14800 |

- **Top features**: ffd_cvar_5pct_06_G_VOL_1, asset_atr_level_G_VOL_1, asset_vol_level_G_VOL_1, clv_mean_24_G_VOL_1, cos_hod_G_VOL_1, vol_state_G_VOL_0, atr_state_G_VOL_1, ffd_cvar_5pct_06_G_VOL_0, rv_8h_G_VOL_0, vol_state_G_VOL_1


#### LONG_TF
- **Features**: 48
- **OOF AUC**: 0.5120
- **OOF IC**: 0.0195
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.1053
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.7549
- **OOF Prec@40**: 0.7337
- **OOF Avg Trades/Day @10%**: 1.4039
- **OOF Avg Trades/Day @30%**: 4.2107
- **OOF ECE@10**: 0.0106
- **OOF Calibration Profile**: well-calibrated

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | -0.0036 / 0.2219 / 0.5100 / 5080 | -0.0002 / 0.2013 / 0.5102 / 5096 | -0.0028 / 0.1964 / 0.4994 / 5080 |
| vol_48h | -0.0016 / 0.2195 / 0.5155 / 5080 | -0.0005 / 0.2032 / 0.5005 / 5096 | -0.0019 / 0.1969 / 0.5091 / 5080 |
| volume_12h | 0.0000 / 0.2107 / 0.5077 / 5080 | 0.0016 / 0.2079 / 0.5114 / 5096 | 0.0009 / 0.2009 / 0.5121 / 5080 |
| volume_48h | 0.0004 / 0.2111 / 0.5063 / 5080 | -0.0002 / 0.2066 / 0.5069 / 5096 | 0.0027 / 0.2019 / 0.5196 / 5080 |
| trend_12h | -0.0003 / 0.2104 / 0.5013 / 5080 | 0.0004 / 0.2112 / 0.5118 / 5096 | 0.0010 / 0.1980 / 0.5175 / 5080 |
| trend_48h | -0.0005 / 0.2140 / 0.5041 / 5080 | 0.0005 / 0.2095 / 0.5104 / 5096 | -0.0008 / 0.1960 / 0.5139 / 5080 |

- **Top features**: speed_G_VOL_1, rv_48h_G_VOL_0, trend_overextension_z_G_VOL_1, rv_8h_G_VOL_1, pullback_120_G_VOL_0, ffd_cvar_5pct_06_G_VOL_0, trend_overextension_z_G_VOL_0, breakout_t_G_VOL_0, wick_body_ratio_G_VOL_1, dn_vol_G_VOL_1


#### SHORT_MR
- **Features**: 48
- **OOF AUC**: 0.5139
- **OOF IC**: 0.0217
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.1038
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.7439
- **OOF Prec@40**: 0.7215
- **OOF Avg Trades/Day @10%**: 1.4039
- **OOF Avg Trades/Day @30%**: 4.2107
- **OOF ECE@10**: 0.0114
- **OOF Calibration Profile**: well-calibrated

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | -0.0036 / 0.2219 / 0.5130 / 5080 | 0.0011 / 0.2011 / 0.5140 / 5096 | -0.0021 / 0.1963 / 0.5006 / 5080 |
| vol_48h | -0.0012 / 0.2194 / 0.5210 / 5080 | 0.0008 / 0.2030 / 0.5087 / 5096 | -0.0018 / 0.1969 / 0.5019 / 5080 |
| volume_12h | 0.0005 / 0.2106 / 0.5052 / 5080 | 0.0012 / 0.2080 / 0.5104 / 5096 | 0.0029 / 0.2006 / 0.5219 / 5080 |
| volume_48h | 0.0003 / 0.2111 / 0.5047 / 5080 | 0.0006 / 0.2064 / 0.5052 / 5096 | 0.0040 / 0.2016 / 0.5286 / 5080 |
| trend_12h | 0.0003 / 0.2102 / 0.5034 / 5080 | 0.0013 / 0.2110 / 0.5155 / 5096 | 0.0014 / 0.1979 / 0.5192 / 5080 |
| trend_48h | -0.0001 / 0.2139 / 0.5078 / 5080 | 0.0021 / 0.2092 / 0.5146 / 5096 | -0.0010 / 0.1961 / 0.5134 / 5080 |

- **Top features**: wick_ratio_G_VOL_0, ffd_cvar_5pct_06_G_VOL_0, rv_48h_G_VOL_0, ffd_diff_8_05_G_VOL_0, G_LIQ_GOOD_G_VOL_0, breakout_confirmed_G_VOL_0, vol_state_G_VOL_1, ffd_range_24_06_G_VOL_0, breakout_t_G_VOL_0, ffd_rv_24_04_G_VOL_1


#### SHORT_TF
- **Features**: 48
- **OOF AUC**: 0.5291
- **OOF IC**: 0.0322
- **OOF Rank IC**: nan
- **OOF Sharpe**: -0.0879
- **OOF Win Rate**: nan%
- **OOF Avg Return**: nan
- **OOF Max Drawdown**: nan
- **OOF Sortino**: nan
- **OOF Calmar**: nan
- **OOF Trades**: 0
- **OOF Prec@10**: 0.7552
- **OOF Prec@40**: 0.7265
- **OOF Avg Trades/Day @10%**: 4.0594
- **OOF Avg Trades/Day @30%**: 12.1763
- **OOF ECE@10**: 0.0020
- **OOF Calibration Profile**: well-calibrated

##### Per-Regime BSS, Brier & AUC
| Regime | Low (BSS / Brier / AUC / N) | Mid (BSS / Brier / AUC / N) | High (BSS / Brier / AUC / N) |
|--------|----------------------------|----------------------------|------------------------------|
| vol_12h | 0.0012 / 0.2133 / 0.5235 / 14800 | 0.0039 / 0.2043 / 0.5280 / 14843 | 0.0040 / 0.1994 / 0.5291 / 14800 |
| vol_48h | 0.0020 / 0.2121 / 0.5226 / 14800 | 0.0040 / 0.2058 / 0.5307 / 14843 | 0.0034 / 0.1990 / 0.5235 / 14800 |
| volume_12h | 0.0036 / 0.2087 / 0.5243 / 14800 | 0.0046 / 0.2080 / 0.5294 / 14843 | 0.0021 / 0.2003 / 0.5326 / 14800 |
| volume_48h | 0.0036 / 0.2079 / 0.5232 / 14800 | 0.0041 / 0.2100 / 0.5307 / 14843 | 0.0019 / 0.1990 / 0.5332 / 14800 |
| trend_12h | 0.0051 / 0.2055 / 0.5308 / 14800 | 0.0028 / 0.2101 / 0.5254 / 14843 | 0.0027 / 0.2013 / 0.5293 / 14800 |
| trend_48h | 0.0017 / 0.2031 / 0.5193 / 14800 | 0.0066 / 0.2085 / 0.5385 / 14843 | 0.0026 / 0.2052 / 0.5277 / 14800 |

- **Top features**: ffd_cvar_5pct_06_G_VOL_1, asset_atr_level_G_VOL_1, asset_vol_level_G_VOL_1, ffd_rv_12_06_G_VOL_1, ffd_diff_1_04_G_VOL_1, cos_hod_G_VOL_1, ffd_rv_6h_06_G_VOL_1, clv_mean_24_G_VOL_1, mtf_divergence_x_regime_vol_12h_G_VOL_1, vol_state_G_VOL_1

## Specialist Models
- **Trap (GMM)**: 8 features, clusters=GaussianMixture(covariance_type='diag', max_iter=200, n_components=4, n_init=3,
                random_state=42)
- **Gamma (ExtraTrees)**: 20 features
