# Training Report — 20260321_140000
Generated: 2026-05-14 04:35 UTC

## Configuration
- **Train lookback**: 35040 hours
- **Label horizons**: [5, 10]
- **Label method**: triple_barrier
- **Label quantiles**: lo=0.3, hi=0.65
- **OOS holdout**: 730 days
- **Min train samples**: 200
- **Feature selection**: MDI (min=30, cap=0.995)
- **15m precision**: True

## Dataset Sizes
| Dataset | Rows | Features |
|---------|------|----------|
| train_bars_in_high_vol_state_log_norm_-0_23822941_dist_prior_day_high_0_098164082_dist_rolling_7d_high_0_063372016_volume_autocorr_48_-0_38587153_10 | 275,644 | 606 |
| train_bars_in_high_vol_state_log_norm_-0_66360348_dist_ema50_atr_0_015806628_dist_prior_day_high_0_10310593_loc_range_pos_24_0_12656839_zscore_price_200_1_67386_ema50_slope_0_50249243_10 | 259,145 | 606 |
| train_dist_rolling_7d_high_0_098617375_mkt_ret_eq_24h_0_71155429_symbol_minus_mkt_ret_24h_1_4400502_xasset_mkt_depth_z_0_96888983_xasset_mkt_ob_stress_14_609916_10 | 335,039 | 606 |
| train_dist_rolling_7d_high_0_13977644_mkt_ret_eq_24h_-0_56630391_rolling_range_20_-0_40672407_10 | 278,204 | 606 |
| train_dist_rolling_7d_high_0_18961641_loc_swing_range_pos_48_0_666668_mkt_ret_eq_24h_-0_620763_symbol_minus_mkt_ret_24h_0_39961502_tail_asymmetry_q90_q10_atr_norm_0_46123043_10 | 222,784 | 606 |
| train_dist_rolling_7d_high_0_24105695_dist_rolling_7d_high_0_090701826_zscore_price_200_0_53547198_atr_compression_ratio_-0_62282997_10 | 253,555 | 606 |
| train_dist_weekly_vwap_0_074823022_loc_prev_week_range_pos_48_0_48354843_mkt_ret_eq_24h_-0_43956268_volume_autocorr_48_-0_38378653_10 | 256,687 | 606 |
| train_loc_ema_stack_pos_24_0_43357179_compression_ratio_-0_33411601_mkt_ret_eq_24h_1_1280091_mkt_ret_eq_24h_-0_81129736_up_down_semivol_ratio_tanh_-0_39156261_10 | 325,967 | 606 |
| train_loc_ema_stack_pos_48_0_98896253_zscore_price_200_1_7867407_atr_compression_ratio_-0_44549009_xasset_mkt_ob_stress_19_055239_10 | 498,553 | 606 |
| train_loc_prev_week_range_pos_48_0_42586401_loc_vwap_dev_z_24_0_10701825_zscore_price_50_1_0128103_mkt_ret_eq_24h_-0_78752208_up_down_return_mass_ratio_tanh_1_1231147_10 | 250,758 | 606 |
| train_prior_range_0_84333628_range_24h_pct_0_10307747_5 | 243,525 | 606 |
| train_range_24h_pct_0_10477796_range_24h_pct_0_063665748_rv_24h_0_1002732_up_down_semivol_ratio_tanh_0_43211657_10 | 215,060 | 606 |

## Alpha Models

### Performance Summary
| Model | Features | AUC | IC | Sharpe | Win Rate | Prec@10 | Prec@30 | Lift@30 | Prec@40 | AvgTr/day@10 | AvgTr/day@30 | Avg Return | Trades | Best Iter |
|-------|----------|-----|----|---------|----------|---------|---------|---------|---------|-------------|-------------|------------|--------|-----------|

### Per-Horizon Alpha Performance (Quality Gate)
| Model | Winner | AUC | IC | LogLoss | PR-AUC | Lift@20 | BrierImp | PR-AUC/Rnd | ECE | Mean IC | IC_Stab | DecSpearman | med(ICm) | mos(ICm<-.01) | mos(Rho>0) | IR_wk | Passed |
|-------|--------|-----|----|---------|--------|---------|----------|------------|-----|---------|---------|-------------|----------|---------------|------------|-------|--------|
| long_prior_range_0_84333628_range_24h_pct_0_10307747_H5:ebm_on_lgbm | — | 0.6133 | 0.1749 | 0.5716 | 0.3775 | 1.4649 | 0.0309 | nan | nan | nan | nan | nan | nan | 0 | 0 | nan | False |
| long_range_24h_pct_0_10477796_range_24h_pct_0_063665748_rv_24h_0_1002732_up_down_semivol_ratio_tanh_0_43211657_H10:ebm_on_lgbm | — | 0.5921 | 0.1462 | 0.6062 | 0.3877 | 1.3662 | 0.0116 | nan | nan | nan | nan | nan | nan | 0 | 0 | nan | False |
| short_dist_rolling_7d_high_0_13977644_mkt_ret_eq_24h_-0_56630391_rolling_range_20_-0_40672407_H10:ebm_on_lgbm | — | 0.6391 | 0.2246 | 0.6046 | 0.4490 | 1.5090 | 0.0452 | nan | nan | nan | nan | nan | nan | 0 | 0 | nan | False |
| short_loc_ema_stack_pos_48_0_98896253_zscore_price_200_1_7867407_atr_compression_ratio_-0_44549009_xasset_mkt_ob_stress_19_055239_H10:ebm_on_lgbm | — | 0.5861 | 0.1389 | 0.6194 | 0.3874 | 1.3017 | 0.0128 | nan | nan | nan | nan | nan | nan | 0 | 0 | nan | False |
| long_dist_rolling_7d_high_0_098617375_mkt_ret_eq_24h_0_71155429_symbol_minus_mkt_ret_24h_1_4400502_xasset_mkt_depth_z_0_96888983_xasset_mkt_ob_stress_14_609916_H10:ebm_on_lgbm | — | 0.6032 | 0.1649 | 0.6059 | 0.4020 | 1.4084 | 0.0243 | nan | nan | nan | nan | nan | nan | 0 | 0 | nan | False |
| short_dist_rolling_7d_high_0_18961641_loc_swing_range_pos_48_0_666668_mkt_ret_eq_24h_-0_620763_symbol_minus_mkt_ret_24h_0_39961502_tail_asymmetry_q90_q10_atr_norm_0_46123043_H10:ebm_on_lgbm | — | 0.5840 | 0.1347 | 0.6179 | 0.3788 | 1.2914 | 0.0075 | nan | nan | nan | nan | nan | nan | 0 | 0 | nan | False |
| short_loc_ema_stack_pos_24_0_43357179_compression_ratio_-0_33411601_mkt_ret_eq_24h_1_1280091_mkt_ret_eq_24h_-0_81129736_up_down_semivol_ratio_tanh_-0_39156261_H10:ebm_on_lgbm | — | 0.6123 | 0.1827 | 0.6163 | 0.4336 | 1.4190 | 0.0337 | nan | nan | nan | nan | nan | nan | 0 | 0 | nan | False |
| long_loc_prev_week_range_pos_48_0_42586401_loc_vwap_dev_z_24_0_10701825_zscore_price_50_1_0128103_mkt_ret_eq_24h_-0_78752208_up_down_return_mass_ratio_tanh_1_1231147_H10:ebm_on_lgbm | — | 0.5797 | 0.1287 | 0.6222 | 0.3874 | 1.3118 | 0.0098 | nan | nan | nan | nan | nan | nan | 0 | 0 | nan | False |
| short_dist_rolling_7d_high_0_24105695_dist_rolling_7d_high_0_090701826_zscore_price_200_0_53547198_atr_compression_ratio_-0_62282997_H10:ebm_on_lgbm | — | 0.6265 | 0.2059 | 0.6170 | 0.4447 | 1.4387 | 0.0352 | nan | nan | nan | nan | nan | nan | 0 | 0 | nan | False |
| long_dist_weekly_vwap_0_074823022_loc_prev_week_range_pos_48_0_48354843_mkt_ret_eq_24h_-0_43956268_volume_autocorr_48_-0_38378653_H10:ebm_on_lgbm | — | 0.5925 | 0.1471 | 0.6052 | 0.3777 | 1.3425 | 0.0167 | nan | nan | nan | nan | nan | nan | 0 | 0 | nan | False |
| long_bars_in_high_vol_state_log_norm_-0_23822941_dist_prior_day_high_0_098164082_dist_rolling_7d_high_0_063372016_volume_autocorr_48_-0_38587153_H10:ebm_on_lgbm | — | 0.5824 | 0.1315 | 0.6114 | 0.3718 | 1.2945 | 0.0094 | nan | nan | nan | nan | nan | nan | 0 | 0 | nan | False |
| short_bars_in_high_vol_state_log_norm_-0_66360348_dist_ema50_atr_0_015806628_dist_prior_day_high_0_10310593_loc_range_pos_24_0_12656839_zscore_price_200_1_67386_ema50_slope_0_50249243_H10:ebm_on_lgbm | — | 0.5894 | 0.1442 | 0.6206 | 0.3950 | 1.3263 | 0.0123 | nan | nan | nan | nan | nan | nan | 0 | 0 | nan | False |

### Detailed Model Performance

#### LONG_MR: **NOT TRAINED**


#### LONG_TF: **NOT TRAINED**


#### SHORT_MR: **NOT TRAINED**


#### SHORT_TF: **NOT TRAINED**

## Specialist Models
