# Simple Position Sizer Regime Report

Run: `20260321_140000`

## Executive Summary
- Consolidated OOF file: `data/artifacts/20260321_140000/oof/simple_sizer_oof_all.parquet`.
- Strategies/models exported: 8 strategy rows; OOF output contains the winning `meta_clf_only` score per strategy.
- Regime adaptors fitted/persisted: 8; enabled after survivor gates: 0.
- Because no adaptor passed `std_ratio <= 0.95`, `dd_ratio <= 0.95`, and `net_ret_ratio >= 1.02`, deployment scores remain raw calibrated simple-sizer scores for this run. The diagnostics still identify weak regime/asset buckets for review.

## Best Model By Strategy
| strategy | best pipeline | wallet pnl | utility | regime enabled | regime score |
| --- | --- | --- | --- | --- | --- |
| bars_in_high_vol_state_log_norm_-0_61002535_dist_weekly_vwap_-0_38453... | meta_clf_only | 4.7516 | 0.1726 | False | 0.0000 |
| bars_in_high_vol_state_log_norm_0_45356679_dist_prior_day_low_0_01158... | meta_clf_only | 1.5331 | -0.0166 | False | 0.0000 |
| bars_in_high_vol_state_log_norm_0_45356679_dist_prior_day_low_0_01158... | meta_clf_only | 1.5331 | -0.0166 | False | 0.0000 |
| dist_ema_fast_-0_0008298126_range_24h_pct_0_036314368_rsi_slope_1_111... | meta_clf_only | 2.4917 | 0.1820 | False | 0.0000 |
| dist_ema_fast_-0_0008298126_range_24h_pct_0_036314368_rsi_slope_1_111... | meta_clf_only | 2.4917 | 0.1820 | False | 0.0000 |
| dist_prior_day_high_0_035536911_dist_prior_day_low_0_012770575_vol_z_... | meta_clf_only | 1.4796 | -0.0144 | False | 0.0000 |
| dist_prior_day_high_0_035536911_dist_prior_day_low_0_012770575_vol_z_... | meta_clf_only | 1.4796 | -0.0144 | False | 0.0000 |
| dist_prior_day_low_0_012672868_loc_prev_week_range_pos_48_0_22452639 | meta_clf_only | 0.8059 | -0.1749 | False | 0.0000 |

## Regime Calibration Impact At Top 10%
| strategy | model | enabled | raw lift | adj lift | d lift | raw hit | adj hit | d hit | raw gross | adj gross | d gross | raw weekly std | adj weekly std | raw maxDD | adj maxDD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bars_in_high_vol_state_log_norm_-0_61002535_dist_weekly_vwap_-0_38453... | MetaClfOnly_clf_ranknorm | False | 1.3950 | 1.3950 | 0.0000 | 0.6376 | 0.6376 | 0.0000 | 0.0138 | 0.0138 | 0.0000 | 0.5558 | 0.5558 | 0.5371 | 0.5371 |
| bars_in_high_vol_state_log_norm_0_45356679_dist_prior_day_low_0_01158... | MetaClfOnly_clf_ranknorm | False | 1.3335 | 1.3335 | 0.0000 | 0.6111 | 0.6111 | 0.0000 | 0.0097 | 0.0097 | 0.0000 | 0.2325 | 0.2325 | 0.5493 | 0.5493 |
| bars_in_high_vol_state_log_norm_0_45356679_dist_prior_day_low_0_01158... | MetaClfOnly_clf_ranknorm | False | 1.3335 | 1.3335 | 0.0000 | 0.6111 | 0.6111 | 0.0000 | 0.0097 | 0.0097 | 0.0000 | 0.2325 | 0.2325 | 0.5493 | 0.5493 |
| dist_ema_fast_-0_0008298126_range_24h_pct_0_036314368_rsi_slope_1_111... | MetaClfOnly_clf_ranknorm | False | 1.4438 | 1.4438 | 0.0000 | 0.6313 | 0.6313 | 0.0000 | 0.0140 | 0.0140 | 0.0000 | 0.4263 | 0.4263 | 0.7848 | 0.7848 |
| dist_ema_fast_-0_0008298126_range_24h_pct_0_036314368_rsi_slope_1_111... | MetaClfOnly_clf_ranknorm | False | 1.4438 | 1.4438 | 0.0000 | 0.6313 | 0.6313 | 0.0000 | 0.0140 | 0.0140 | 0.0000 | 0.4263 | 0.4263 | 0.7848 | 0.7848 |
| dist_prior_day_high_0_035536911_dist_prior_day_low_0_012770575_vol_z_... | MetaClfOnly_clf_calibrated | False | 1.3305 | 1.3305 | 0.0000 | 0.5595 | 0.5595 | 0.0000 | 0.0097 | 0.0097 | 0.0000 | 0.1860 | 0.1860 | 0.6816 | 0.6816 |
| dist_prior_day_high_0_035536911_dist_prior_day_low_0_012770575_vol_z_... | MetaClfOnly_clf_calibrated | False | 1.3305 | 1.3305 | 0.0000 | 0.5595 | 0.5595 | 0.0000 | 0.0097 | 0.0097 | 0.0000 | 0.1860 | 0.1860 | 0.6816 | 0.6816 |
| dist_prior_day_low_0_012672868_loc_prev_week_range_pos_48_0_22452639 | MetaClfOnly_clf_calibrated | False | 1.3213 | 1.3213 | 0.0000 | 0.5789 | 0.5789 | 0.0000 | 0.0070 | 0.0070 | 0.0000 | 0.1790 | 0.1790 | 0.9078 | 0.9078 |

## Raw Vs Regime Candidate Metrics By Top Fraction
| strategy | top | raw lift | adj lift | raw gross | adj gross | raw hit | adj hit | raw sortino | adj sortino |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bars_in_high_vol_state_log_norm_-0_61002535_dist_weekly_vwap_-0_38453... | 1% | 1.6912 | 1.6912 | 0.0203 | 0.0203 | 0.7729 | 0.7729 | 2.0337 | 2.0337 |
| bars_in_high_vol_state_log_norm_-0_61002535_dist_weekly_vwap_-0_38453... | 5% | 1.5076 | 1.5076 | 0.0163 | 0.0163 | 0.6890 | 0.6890 | 1.7166 | 1.7166 |
| bars_in_high_vol_state_log_norm_-0_61002535_dist_weekly_vwap_-0_38453... | 10% | 1.3950 | 1.3950 | 0.0138 | 0.0138 | 0.6376 | 0.6376 | 1.3925 | 1.3925 |
| bars_in_high_vol_state_log_norm_-0_61002535_dist_weekly_vwap_-0_38453... | 20% | 1.3006 | 1.3006 | 0.0117 | 0.0117 | 0.5944 | 0.5944 | 1.0873 | 1.0873 |
| bars_in_high_vol_state_log_norm_0_45356679_dist_prior_day_low_0_01158... | 1% | 1.6658 | 1.6658 | 0.0170 | 0.0170 | 0.7634 | 0.7634 | 1.6030 | 1.6030 |
| bars_in_high_vol_state_log_norm_0_45356679_dist_prior_day_low_0_01158... | 5% | 1.4246 | 1.4246 | 0.0118 | 0.0118 | 0.6528 | 0.6528 | 1.0885 | 1.0885 |
| bars_in_high_vol_state_log_norm_0_45356679_dist_prior_day_low_0_01158... | 10% | 1.3335 | 1.3335 | 0.0097 | 0.0097 | 0.6111 | 0.6111 | 0.7404 | 0.7404 |
| bars_in_high_vol_state_log_norm_0_45356679_dist_prior_day_low_0_01158... | 20% | 1.2630 | 1.2630 | 0.0083 | 0.0083 | 0.5788 | 0.5788 | 0.5628 | 0.5628 |
| bars_in_high_vol_state_log_norm_0_45356679_dist_prior_day_low_0_01158... | 1% | 1.6658 | 1.6658 | 0.0170 | 0.0170 | 0.7634 | 0.7634 | 1.6030 | 1.6030 |
| bars_in_high_vol_state_log_norm_0_45356679_dist_prior_day_low_0_01158... | 5% | 1.4246 | 1.4246 | 0.0118 | 0.0118 | 0.6528 | 0.6528 | 1.0885 | 1.0885 |
| bars_in_high_vol_state_log_norm_0_45356679_dist_prior_day_low_0_01158... | 10% | 1.3335 | 1.3335 | 0.0097 | 0.0097 | 0.6111 | 0.6111 | 0.7404 | 0.7404 |
| bars_in_high_vol_state_log_norm_0_45356679_dist_prior_day_low_0_01158... | 20% | 1.2630 | 1.2630 | 0.0083 | 0.0083 | 0.5788 | 0.5788 | 0.5628 | 0.5628 |
| dist_ema_fast_-0_0008298126_range_24h_pct_0_036314368_rsi_slope_1_111... | 1% | 1.5306 | 1.5306 | 0.0167 | 0.0167 | 0.6692 | 0.6692 | 1.2517 | 1.2517 |
| dist_ema_fast_-0_0008298126_range_24h_pct_0_036314368_rsi_slope_1_111... | 5% | 1.5095 | 1.5095 | 0.0158 | 0.0158 | 0.6600 | 0.6600 | 1.0673 | 1.0673 |
| dist_ema_fast_-0_0008298126_range_24h_pct_0_036314368_rsi_slope_1_111... | 10% | 1.4438 | 1.4438 | 0.0140 | 0.0140 | 0.6313 | 0.6313 | 1.0410 | 1.0410 |
| dist_ema_fast_-0_0008298126_range_24h_pct_0_036314368_rsi_slope_1_111... | 20% | 1.3510 | 1.3510 | 0.0118 | 0.0118 | 0.5907 | 0.5907 | 0.8237 | 0.8237 |
| dist_ema_fast_-0_0008298126_range_24h_pct_0_036314368_rsi_slope_1_111... | 1% | 1.5306 | 1.5306 | 0.0167 | 0.0167 | 0.6692 | 0.6692 | 1.2517 | 1.2517 |
| dist_ema_fast_-0_0008298126_range_24h_pct_0_036314368_rsi_slope_1_111... | 5% | 1.5095 | 1.5095 | 0.0158 | 0.0158 | 0.6600 | 0.6600 | 1.0673 | 1.0673 |
| dist_ema_fast_-0_0008298126_range_24h_pct_0_036314368_rsi_slope_1_111... | 10% | 1.4438 | 1.4438 | 0.0140 | 0.0140 | 0.6313 | 0.6313 | 1.0410 | 1.0410 |
| dist_ema_fast_-0_0008298126_range_24h_pct_0_036314368_rsi_slope_1_111... | 20% | 1.3510 | 1.3510 | 0.0118 | 0.0118 | 0.5907 | 0.5907 | 0.8237 | 0.8237 |
| dist_prior_day_high_0_035536911_dist_prior_day_low_0_012770575_vol_z_... | 1% | 1.3934 | 1.3934 | 0.0114 | 0.0114 | 0.5859 | 0.5859 | 0.9006 | 0.9006 |
| dist_prior_day_high_0_035536911_dist_prior_day_low_0_012770575_vol_z_... | 5% | 1.4216 | 1.4216 | 0.0116 | 0.0116 | 0.5978 | 0.5978 | 0.8930 | 0.8930 |
| dist_prior_day_high_0_035536911_dist_prior_day_low_0_012770575_vol_z_... | 10% | 1.3305 | 1.3305 | 0.0097 | 0.0097 | 0.5595 | 0.5595 | 0.6774 | 0.6774 |
| dist_prior_day_high_0_035536911_dist_prior_day_low_0_012770575_vol_z_... | 20% | 1.2779 | 1.2779 | 0.0084 | 0.0084 | 0.5374 | 0.5374 | 0.5314 | 0.5314 |
| dist_prior_day_high_0_035536911_dist_prior_day_low_0_012770575_vol_z_... | 1% | 1.3934 | 1.3934 | 0.0114 | 0.0114 | 0.5859 | 0.5859 | 0.9006 | 0.9006 |
| dist_prior_day_high_0_035536911_dist_prior_day_low_0_012770575_vol_z_... | 5% | 1.4216 | 1.4216 | 0.0116 | 0.0116 | 0.5978 | 0.5978 | 0.8930 | 0.8930 |
| dist_prior_day_high_0_035536911_dist_prior_day_low_0_012770575_vol_z_... | 10% | 1.3305 | 1.3305 | 0.0097 | 0.0097 | 0.5595 | 0.5595 | 0.6774 | 0.6774 |
| dist_prior_day_high_0_035536911_dist_prior_day_low_0_012770575_vol_z_... | 20% | 1.2779 | 1.2779 | 0.0084 | 0.0084 | 0.5374 | 0.5374 | 0.5314 | 0.5314 |
| dist_prior_day_low_0_012672868_loc_prev_week_range_pos_48_0_22452639 | 1% | 1.6405 | 1.6405 | 0.0128 | 0.0128 | 0.7188 | 0.7188 | 1.2847 | 1.2847 |
| dist_prior_day_low_0_012672868_loc_prev_week_range_pos_48_0_22452639 | 5% | 1.3766 | 1.3766 | 0.0076 | 0.0076 | 0.6031 | 0.6031 | 0.5131 | 0.5131 |
| dist_prior_day_low_0_012672868_loc_prev_week_range_pos_48_0_22452639 | 10% | 1.3213 | 1.3213 | 0.0070 | 0.0070 | 0.5789 | 0.5789 | 0.4477 | 0.4477 |
| dist_prior_day_low_0_012672868_loc_prev_week_range_pos_48_0_22452639 | 20% | 1.2473 | 1.2473 | 0.0056 | 0.0056 | 0.5465 | 0.5465 | 0.2960 | 0.2960 |

## Regime And Asset Gates
- Fixed bucket diagnostics rows: 64; gated rows: 0.
- Asset diagnostics rows: 2136; gated rows: 0.

## Artifact Inventory
- `/Users/remyroche/Documents/Ares/data/artifacts/20260321_140000/ridge_sizer/regime_diagnostics_fixed.parquet`
- `/Users/remyroche/Documents/Ares/data/artifacts/20260321_140000/ridge_sizer/regime_diagnostics_adaptive.parquet`
- `/Users/remyroche/Documents/Ares/data/artifacts/20260321_140000/ridge_sizer/regime_asset_diagnostics.parquet`
- `/Users/remyroche/Documents/Ares/data/artifacts/20260321_140000/ridge_sizer/regime_before_after_metrics.parquet`
- `/Users/remyroche/Documents/Ares/data/artifacts/20260321_140000/ridge_sizer/regime_adaptor_summary.json`
- `/Users/remyroche/Documents/Ares/data/artifacts/20260321_140000/ridge_sizer/strategy_params.json`
- `/Users/remyroche/Documents/Ares/data/artifacts/20260321_140000/oof/simple_sizer_oof_all.parquet`
