# Kraken Actual OI/Volume Backfill Report

## Coverage Old vs New

| metric         |   old_rows |   new_rows |   delta_rows |   old_pct_of_price_rows |   new_pct_of_price_rows |   delta_pct_points |
|:---------------|-----------:|-----------:|-------------:|------------------------:|------------------------:|-------------------:|
| missing_oi     |     356781 |     356785 |            4 |                  7.5410 |                  7.5403 |            -0.0007 |
| missing_volume |    1716932 |      22810 |     -1694122 |                 36.2894 |                  0.4821 |           -35.8073 |
| valid_all      |    2812489 |    4358903 |      1546414 |                 59.4453 |                 92.1214 |            32.6761 |


## Regenerated Features: Empty Rows Before/After

| feature              | sources                    |   empty_pct_before |   empty_pct_after |   delta_empty_pct_points |   empty_rows_before |   empty_rows_after |
|:---------------------|:---------------------------|-------------------:|------------------:|-------------------------:|--------------------:|-------------------:|
| dist_stack           | model_contract             |            48.8323 |           48.8485 |                   0.0162 |             4090420 |            4091780 |
| dist_vwap_12_atr     | model_contract             |            44.4039 |           44.3990 |                  -0.0048 |             3766165 |            3765755 |
| dist_vwap_24_atr     | model_contract             |            44.4039 |           44.3990 |                  -0.0048 |             3766165 |            3765755 |
| dist_vwap_96_atr     | model_contract             |            44.4039 |           44.3990 |                  -0.0048 |             3766165 |            3765755 |
| dist_vwap_norm       | model_contract             |            44.4366 |           44.3990 |                  -0.0376 |             3768944 |            3765755 |
| leverage_build       | oi_crowding                |            47.7118 |           46.6998 |                  -1.0119 |             4046729 |            3960900 |
| leverage_build_score | oi_crowding                |            46.6998 |           46.6998 |                   0.0000 |             3960900 |            3960900 |
| oi_rel_vol_2h        | model_contract;oi_crowding |            60.3695 |           60.3695 |                   0.0000 |             5120305 |            5120305 |
| oi_rel_vol_4h        | model_contract;oi_crowding |            55.8240 |           55.8240 |                   0.0000 |             4734779 |            4734779 |
| oi_rel_vol_8h        | model_contract;oi_crowding |            52.1419 |           52.1419 |                   0.0000 |             4422473 |            4422473 |
| trapped_longs_12     | model_contract             |            44.4039 |           44.3990 |                  -0.0048 |             3766165 |            3765755 |
| trapped_longs_24     | model_contract             |            44.4039 |           44.3990 |                  -0.0048 |             3766165 |            3765755 |
| trapped_longs_96     | model_contract             |            44.4039 |           44.3990 |                  -0.0048 |             3766165 |            3765755 |
| unwind_score         | model_contract;oi_crowding |            46.7988 |           46.7988 |                   0.0000 |             3969293 |            3969293 |
| vwap_zone_1d_atr     | model_contract             |            43.3037 |           43.2977 |                  -0.0060 |             3672854 |            3672344 |
| vwap_zone_7d_atr     | model_contract             |            42.4093 |           42.4023 |                  -0.0070 |             3596998 |            3596402 |


## Actual Volume Status Counts

| coverage_status     |    rows |     pct |
|:--------------------|--------:|--------:|
| actual_trades       | 1980714 | 53.4637 |
| confirmed_no_trades | 1693949 | 45.7233 |
| unavailable         |   30122 |  0.8131 |


## Actual Volume Quality Totals

- Sidecar rows: 3,704,785
- Duplicate timestamps: 0
- Non-hourly timestamps: 0
- Negative numeric rows: 0
- Bad actual-trade rows: 0
- Bad confirmed-no-trade rows: 0
- Unavailable rows carrying numeric volume/trade values: 0


## Per-Feature Post-Recompute Missing/Fallback Breakdown

| feature              |   nan_rows_pct |   missing_column_rows_pct |   inf_rows_pct |   synthetic_fallback_rows_pct |   actual_trade_overlay_rows_pct |   confirmed_no_trade_overlay_rows_pct |   unavailable_volume_rows_pct |
|:---------------------|---------------:|--------------------------:|---------------:|------------------------------:|--------------------------------:|--------------------------------------:|------------------------------:|
| oi_rel_vol_2h        |        60.3695 |                    0.0000 |         0.0000 |                        0.0000 |                         23.3078 |                               19.8899 |                        0.3284 |
| oi_rel_vol_4h        |        55.8240 |                    0.0000 |         0.0000 |                        0.0000 |                         23.3078 |                               19.8899 |                        0.3284 |
| oi_rel_vol_8h        |        52.1419 |                    0.0000 |         0.0000 |                        0.0000 |                         23.3078 |                               19.8899 |                        0.3284 |
| dist_stack           |        48.2951 |                    0.0000 |         0.0000 |                        0.0000 |                         23.3078 |                               19.8899 |                        0.3284 |
| unwind_score         |        46.7988 |                    0.0000 |         0.0000 |                        0.0000 |                         23.3078 |                               19.8899 |                        0.3284 |
| leverage_build       |        46.6998 |                    0.0000 |         0.0000 |                        0.0000 |                         23.3078 |                               19.8899 |                        0.3284 |
| leverage_build_score |        46.6998 |                    0.0000 |         0.0000 |                        0.0000 |                         23.3078 |                               19.8899 |                        0.3284 |
| dist_vwap_12_atr     |        44.3990 |                    0.0000 |         0.0000 |                        0.0000 |                         23.3078 |                               19.8899 |                        0.3284 |
| dist_vwap_24_atr     |        44.3990 |                    0.0000 |         0.0000 |                        0.0000 |                         23.3078 |                               19.8899 |                        0.3284 |
| dist_vwap_96_atr     |        44.3990 |                    0.0000 |         0.0000 |                        0.0000 |                         23.3078 |                               19.8899 |                        0.3284 |
| dist_vwap_norm       |        44.3990 |                    0.0000 |         0.0000 |                        0.0000 |                         23.3078 |                               19.8899 |                        0.3284 |
| trapped_longs_12     |        44.3990 |                    0.0000 |         0.0000 |                        0.0000 |                         23.3078 |                               19.8899 |                        0.3284 |
| trapped_longs_24     |        44.3990 |                    0.0000 |         0.0000 |                        0.0000 |                         23.3078 |                               19.8899 |                        0.3284 |
| trapped_longs_96     |        44.3990 |                    0.0000 |         0.0000 |                        0.0000 |                         23.3078 |                               19.8899 |                        0.3284 |
| vwap_zone_1d_atr     |        43.2977 |                    0.0000 |         0.0000 |                        0.0000 |                         23.3078 |                               19.8899 |                        0.3284 |
| vwap_zone_7d_atr     |        42.4023 |                    0.0000 |         0.0000 |                        0.0000 |                         23.3078 |                               19.8899 |                        0.3284 |


## CSV Artifacts

- `kraken_actual_coverage_old_vs_new.csv`
- `regenerated_features_empty_before_after.csv`
- `actual_volume_sidecar_quality_by_symbol.csv`
- `actual_volume_coverage_status_counts.csv`
- `actual_volume_source_counts.csv`
- `per_feature_post_recompute_missing_fallback_breakdown.csv`