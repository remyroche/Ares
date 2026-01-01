# Liquidity Cluster Quality Report

**Symbol:** ETHUSDT  \n**Assessment time:** 2025-12-31T17:22:13.938812

## Overall Quality

- Overall quality score: **0.4924**

## Regime Separation Algorithm

The regime classification uses a hierarchical decision tree based on two key metrics:

**Key Metrics:**
- **RVOL (Relative Volume):** Volume / Average Volume (20-bar lookback)
- **VER (Volume-Efficiency Ratio):** Volume / Candle Range (High - Low)

### Phase 1: The "Energy" Filter (Volume)

First, check if the market is awake using RVOL:

**If RVOL < 0.8 (Low Energy):**
- Check Price Range:
  - **Small Range** → **Apathy** (Dead Zone)
  - **Large Range** → **Ghost** (Liquidity Gap / Trap)

### Phase 2: The "Conflict" Filter (High Volume)

**If RVOL > 1.2 (High Energy):** Big players are active. Check Efficiency (VER):

- **Is Range Small relative to Volume? (Low VER)**
  - → **Absorption** (The harder they push, the less it moves)
- **Is Range Large relative to Volume? (High VER)**
  - → **Valid Trend** (Efficient price discovery)

### Phase 3: The "Anomaly" Filter (The Steamroller)

This is the outlier regime:

**If RVOL > 3.0 AND Range > 3x ATR (Average True Range):**
- → **Steamroller**
- Even though liquidity is thick, buying pressure is so immense it clears the book instantly
- Represents initiative momentum with low liquidity risk

---

## CoV-based Separation

- Effort/Result CoV separation score: 0.4729
- Returns CoV separation score: 0.4721

## Effort vs Result Separation

- Effort/Result separation score: 0.3423
- Ghost vs Valid contrast: -0.3077
- Absorption vs Valid contrast: 0.4056

## Trap / Ghost Behavior

- Ghost reversal rate: 0.3536
- Ghost false-trend rate: 0.2433

## Absorption Behavior

- Absorption reversal rate: 0.5000
- Absorption follow-through rate: 0.1552

## Trend Confirmation & Apathy

- Valid trend follow-through (mean fwd return): -0.000260
- Apathy noise fraction: 0.3972

## Class Balance

- Class balance score: 0.9521
- Number of regimes: 5
- Number of samples: 2880

## Per-Regime Metrics

### Regime 0

- n_samples: 574.000000
- ghost_ratio_mean: 0.119443
- ghost_ratio_std: 0.045357
- ghost_ratio_cov: 0.379736
- absorption_ratio_mean: 9.460928
- absorption_ratio_std: 3.347267
- absorption_ratio_cov: 0.353799
- rvol_24_mean: 0.504653
- rvol_24_std: 0.202450
- rvol_24_cov: 0.401167
- rvol_20_mean: 0.520985
- rvol_20_std: 0.212179
- rvol_20_cov: 0.407265
- volume_efficiency_ratio_mean: 247.224518
- volume_efficiency_ratio_std: 37.828957
- volume_efficiency_ratio_cov: 0.153015
- intraday_close_ratio_mean: 18323.030684
- intraday_close_ratio_std: 27470.563744
- intraday_close_ratio_cov: 1.499237
- amihud_spike_ratio_scaled_mean: -0.209095
- amihud_spike_ratio_scaled_std: 0.688527
- amihud_spike_ratio_scaled_cov: 3.292896
- rvol_168_scaled_mean: -0.552515
- rvol_168_scaled_std: 0.367562
- rvol_168_scaled_cov: 0.665252
- cumulative_delta_divergence_mean: 0.946520
- cumulative_delta_divergence_std: 0.658278
- cumulative_delta_divergence_cov: 0.695472
- volume_direction_conviction_mean: 0.493688
- volume_direction_conviction_std: 0.290810
- volume_direction_conviction_cov: 0.589056
- volume_direction_imbalance_mean: -0.018809
- volume_direction_imbalance_std: 0.573036
- volume_direction_imbalance_cov: 30.465400
- trend_confirmation_6h_mean: 0.215672
- trend_confirmation_6h_std: 0.136761
- trend_confirmation_6h_cov: 0.634119
- momentum_persistence_3h_mean: 0.409100
- momentum_persistence_3h_std: 15.985050
- momentum_persistence_3h_cov: 39.073743
- vol_momentum_sync_mean: 0.043201
- vol_momentum_sync_std: 0.184751
- vol_momentum_sync_cov: 4.276582
- range_momentum_divergence_mean: 0.999871
- range_momentum_divergence_std: 0.000082
- range_momentum_divergence_cov: 0.000082
- volume_concentration_ratio_3h_mean: 0.442144
- volume_concentration_ratio_3h_std: 0.159642
- volume_concentration_ratio_3h_cov: 0.361063
- pressure_ratio_mean: 197406326784.000000
- pressure_ratio_std: 2179601399808.000000
- pressure_ratio_cov: 11.041193
- kyle_lambda_proxy_mean: 126.389723
- kyle_lambda_proxy_std: 68.820719
- kyle_lambda_proxy_cov: 0.544512
- reversal_intensity_mean: 0.000697
- reversal_intensity_std: 0.000942
- reversal_intensity_cov: 1.353103
- whipsaw_count_mean: 6.705575
- whipsaw_count_std: 1.536776
- whipsaw_count_cov: 0.229179
- vol_clustering_mean: 0.346472
- vol_clustering_std: 0.116219
- vol_clustering_cov: 0.335436
- vol_regime_change_mean: -0.071583
- vol_regime_change_std: 0.209799
- vol_regime_change_cov: 2.930841
- efficiency_ratio_mean: 558.746865
- efficiency_ratio_std: 412.202649
- efficiency_ratio_cov: 0.737727
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: 0.000125
- forward_return_std: 0.002995
- forward_return_cov: 23.985946
- forward_return_positive_rate: 0.545296
- forward_return_negative_rate: 0.452962
- forward_return_sharpe_like: 0.041691
- forward_return_mar_like: 0.001790
- forward_return_tail_loss_p95: 0.003845
- adverse_selection_rate: 0.452962
- structural_low_fraction: 0.182927
- transient_gap_fraction: 0.000000

### Regime 1

- n_samples: 310.000000
- ghost_ratio_mean: 0.319038
- ghost_ratio_std: 0.095505
- ghost_ratio_cov: 0.299353
- absorption_ratio_mean: 3.398081
- absorption_ratio_std: 1.268737
- absorption_ratio_cov: 0.373369
- rvol_24_mean: 2.399519
- rvol_24_std: 0.891907
- rvol_24_cov: 0.371702
- rvol_20_mean: 2.737389
- rvol_20_std: 1.735784
- rvol_20_cov: 0.634102
- volume_efficiency_ratio_mean: 549.234253
- volume_efficiency_ratio_std: 342.735352
- volume_efficiency_ratio_cov: 0.624024
- intraday_close_ratio_mean: 12901.056169
- intraday_close_ratio_std: 21362.404494
- intraday_close_ratio_cov: 1.655865
- amihud_spike_ratio_scaled_mean: -0.432431
- amihud_spike_ratio_scaled_std: 0.581836
- amihud_spike_ratio_scaled_cov: 1.345500
- rvol_168_scaled_mean: 2.012220
- rvol_168_scaled_std: 0.958940
- rvol_168_scaled_cov: 0.476558
- cumulative_delta_divergence_mean: 1.282929
- cumulative_delta_divergence_std: 0.980629
- cumulative_delta_divergence_cov: 0.764367
- volume_direction_conviction_mean: 0.548332
- volume_direction_conviction_std: 0.273990
- volume_direction_conviction_cov: 0.499679
- volume_direction_imbalance_mean: -0.023721
- volume_direction_imbalance_std: 0.613309
- volume_direction_imbalance_cov: 25.854695
- trend_confirmation_6h_mean: 0.255939
- trend_confirmation_6h_std: 0.139087
- trend_confirmation_6h_cov: 0.543436
- momentum_persistence_3h_mean: -1.323352
- momentum_persistence_3h_std: 26.318053
- momentum_persistence_3h_cov: 19.887423
- vol_momentum_sync_mean: 0.322451
- vol_momentum_sync_std: 0.361717
- vol_momentum_sync_cov: 1.121774
- range_momentum_divergence_mean: 0.999835
- range_momentum_divergence_std: 0.000088
- range_momentum_divergence_cov: 0.000088
- volume_concentration_ratio_3h_mean: 0.469550
- volume_concentration_ratio_3h_std: 0.208144
- volume_concentration_ratio_3h_cov: 0.443284
- pressure_ratio_mean: 981800321024.000000
- pressure_ratio_std: 17286378291200.000000
- pressure_ratio_cov: 17.606817
- kyle_lambda_proxy_mean: 326.547080
- kyle_lambda_proxy_std: 162.475915
- kyle_lambda_proxy_cov: 0.497557
- reversal_intensity_mean: 0.003070
- reversal_intensity_std: 0.005251
- reversal_intensity_cov: 1.710250
- whipsaw_count_mean: 5.909091
- whipsaw_count_std: 1.667328
- whipsaw_count_cov: 0.282163
- vol_clustering_mean: 0.468042
- vol_clustering_std: 0.111086
- vol_clustering_cov: 0.237341
- vol_regime_change_mean: 0.113119
- vol_regime_change_std: 0.218173
- vol_regime_change_cov: 1.928703
- efficiency_ratio_mean: 1000.997801
- efficiency_ratio_std: 630.857736
- efficiency_ratio_cov: 0.630229
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.000260
- forward_return_std: 0.006650
- forward_return_cov: 25.570350
- forward_return_positive_rate: 0.448387
- forward_return_negative_rate: 0.551613
- forward_return_sharpe_like: -0.039108
- forward_return_mar_like: -0.001249
- forward_return_tail_loss_p95: 0.010045
- adverse_selection_rate: 0.551613
- structural_low_fraction: 0.000000
- transient_gap_fraction: 0.000000

### Regime 2

- n_samples: 348.000000
- ghost_ratio_mean: 0.143207
- ghost_ratio_std: 0.054659
- ghost_ratio_cov: 0.381680
- absorption_ratio_mean: 8.036141
- absorption_ratio_std: 3.118847
- absorption_ratio_cov: 0.388103
- rvol_24_mean: 1.288711
- rvol_24_std: 0.602946
- rvol_24_cov: 0.467867
- rvol_20_mean: 1.295849
- rvol_20_std: 0.630769
- rvol_20_cov: 0.486761
- volume_efficiency_ratio_mean: 600.854431
- volume_efficiency_ratio_std: 341.442200
- volume_efficiency_ratio_cov: 0.568261
- intraday_close_ratio_mean: 20941.840715
- intraday_close_ratio_std: 32410.642763
- intraday_close_ratio_cov: 1.547650
- amihud_spike_ratio_scaled_mean: -0.763511
- amihud_spike_ratio_scaled_std: 0.355946
- amihud_spike_ratio_scaled_cov: 0.466196
- rvol_168_scaled_mean: 0.688820
- rvol_168_scaled_std: 0.769496
- rvol_168_scaled_cov: 1.117123
- cumulative_delta_divergence_mean: 1.078080
- cumulative_delta_divergence_std: 0.836149
- cumulative_delta_divergence_cov: 0.775592
- volume_direction_conviction_mean: 0.477289
- volume_direction_conviction_std: 0.288989
- volume_direction_conviction_cov: 0.605479
- volume_direction_imbalance_mean: -0.044720
- volume_direction_imbalance_std: 0.556750
- volume_direction_imbalance_cov: 12.449761
- trend_confirmation_6h_mean: 0.224141
- trend_confirmation_6h_std: 0.148840
- trend_confirmation_6h_cov: 0.664044
- momentum_persistence_3h_mean: -0.566795
- momentum_persistence_3h_std: 6.747498
- momentum_persistence_3h_cov: 11.904653
- vol_momentum_sync_mean: 0.071243
- vol_momentum_sync_std: 0.221243
- vol_momentum_sync_cov: 3.105467
- range_momentum_divergence_mean: 0.999875
- range_momentum_divergence_std: 0.000082
- range_momentum_divergence_cov: 0.000082
- volume_concentration_ratio_3h_mean: 0.432374
- volume_concentration_ratio_3h_std: 0.154110
- volume_concentration_ratio_3h_cov: 0.356428
- pressure_ratio_mean: 2643695501312.000000
- pressure_ratio_std: 23533216333824.000000
- pressure_ratio_cov: 8.901636
- kyle_lambda_proxy_mean: 221.353288
- kyle_lambda_proxy_std: 126.939739
- kyle_lambda_proxy_cov: 0.573471
- reversal_intensity_mean: 0.000762
- reversal_intensity_std: 0.001277
- reversal_intensity_cov: 1.676415
- whipsaw_count_mean: 6.247126
- whipsaw_count_std: 1.705008
- whipsaw_count_cov: 0.272927
- vol_clustering_mean: 0.389128
- vol_clustering_std: 0.122186
- vol_clustering_cov: 0.314000
- vol_regime_change_mean: -0.049570
- vol_regime_change_std: 0.224960
- vol_regime_change_cov: 4.538258
- efficiency_ratio_mean: 751.557071
- efficiency_ratio_std: 530.814570
- efficiency_ratio_cov: 0.706286
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: 0.000064
- forward_return_std: 0.004485
- forward_return_cov: 70.509889
- forward_return_positive_rate: 0.500000
- forward_return_negative_rate: 0.500000
- forward_return_sharpe_like: 0.014182
- forward_return_mar_like: 0.001028
- forward_return_tail_loss_p95: 0.006196
- adverse_selection_rate: 0.500000
- structural_low_fraction: 0.002874
- transient_gap_fraction: 0.000000

### Regime 3

- n_samples: 789.000000
- ghost_ratio_mean: 0.168882
- ghost_ratio_std: 0.077446
- ghost_ratio_cov: 0.458580
- absorption_ratio_mean: 7.122926
- absorption_ratio_std: 3.047773
- absorption_ratio_cov: 0.427882
- rvol_24_mean: 0.444372
- rvol_24_std: 0.219935
- rvol_24_cov: 0.494934
- rvol_20_mean: 0.459928
- rvol_20_std: 0.228148
- rvol_20_cov: 0.496050
- volume_efficiency_ratio_mean: 157.243271
- volume_efficiency_ratio_std: 30.015606
- volume_efficiency_ratio_cov: 0.190886
- intraday_close_ratio_mean: 12803.793164
- intraday_close_ratio_std: 21698.965636
- intraday_close_ratio_cov: 1.694729
- amihud_spike_ratio_scaled_mean: 0.883326
- amihud_spike_ratio_scaled_std: 1.241744
- amihud_spike_ratio_scaled_cov: 1.405759
- rvol_168_scaled_mean: -0.689710
- rvol_168_scaled_std: 0.444174
- rvol_168_scaled_cov: 0.644002
- cumulative_delta_divergence_mean: 0.876975
- cumulative_delta_divergence_std: 0.648330
- cumulative_delta_divergence_cov: 0.739280
- volume_direction_conviction_mean: 0.539464
- volume_direction_conviction_std: 0.285875
- volume_direction_conviction_cov: 0.529923
- volume_direction_imbalance_mean: 0.034534
- volume_direction_imbalance_std: 0.609853
- volume_direction_imbalance_cov: 17.659445
- trend_confirmation_6h_mean: 0.239334
- trend_confirmation_6h_std: 0.142563
- trend_confirmation_6h_cov: 0.595665
- momentum_persistence_3h_mean: 0.057067
- momentum_persistence_3h_std: 9.766110
- momentum_persistence_3h_cov: 171.134436
- vol_momentum_sync_mean: 0.146794
- vol_momentum_sync_std: 0.307748
- vol_momentum_sync_cov: 2.096469
- range_momentum_divergence_mean: 0.999838
- range_momentum_divergence_std: 0.000084
- range_momentum_divergence_cov: 0.000084
- volume_concentration_ratio_3h_mean: 0.440237
- volume_concentration_ratio_3h_std: 0.154506
- volume_concentration_ratio_3h_cov: 0.350960
- pressure_ratio_mean: 502778167296.000000
- pressure_ratio_std: 6470257606656.000000
- pressure_ratio_cov: 12.869011
- kyle_lambda_proxy_mean: 134.605767
- kyle_lambda_proxy_std: 85.454963
- kyle_lambda_proxy_cov: 0.634854
- reversal_intensity_mean: 0.001526
- reversal_intensity_std: 0.001889
- reversal_intensity_cov: 1.237456
- whipsaw_count_mean: 6.719899
- whipsaw_count_std: 1.627452
- whipsaw_count_cov: 0.242184
- vol_clustering_mean: 0.360900
- vol_clustering_std: 0.123320
- vol_clustering_cov: 0.341701
- vol_regime_change_mean: -0.066247
- vol_regime_change_std: 0.201277
- vol_regime_change_cov: 3.038295
- efficiency_ratio_mean: 474.921436
- efficiency_ratio_std: 363.700239
- efficiency_ratio_cov: 0.765811
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.000030
- forward_return_std: 0.003128
- forward_return_cov: 102.759950
- forward_return_positive_rate: 0.493029
- forward_return_negative_rate: 0.506971
- forward_return_sharpe_like: -0.009731
- forward_return_mar_like: -0.000275
- forward_return_tail_loss_p95: 0.004811
- adverse_selection_rate: 0.506971
- structural_low_fraction: 0.129278
- transient_gap_fraction: 0.000000

### Regime 4

- n_samples: 859.000000
- ghost_ratio_mean: 0.208000
- ghost_ratio_std: 0.089488
- ghost_ratio_cov: 0.430231
- absorption_ratio_mean: 5.788511
- absorption_ratio_std: 2.661089
- absorption_ratio_cov: 0.459719
- rvol_24_mean: 0.952329
- rvol_24_std: 0.470722
- rvol_24_cov: 0.494284
- rvol_20_mean: 0.959234
- rvol_20_std: 0.479322
- rvol_20_cov: 0.499693
- volume_efficiency_ratio_mean: 299.636017
- volume_efficiency_ratio_std: 59.387676
- volume_efficiency_ratio_cov: 0.198199
- intraday_close_ratio_mean: 15386.255735
- intraday_close_ratio_std: 24733.235015
- intraday_close_ratio_cov: 1.607489
- amihud_spike_ratio_scaled_mean: -0.130809
- amihud_spike_ratio_scaled_std: 0.755546
- amihud_spike_ratio_scaled_cov: 5.775964
- rvol_168_scaled_mean: 0.239357
- rvol_168_scaled_std: 0.585159
- rvol_168_scaled_cov: 2.444712
- cumulative_delta_divergence_mean: 1.026284
- cumulative_delta_divergence_std: 0.765434
- cumulative_delta_divergence_cov: 0.745831
- volume_direction_conviction_mean: 0.499492
- volume_direction_conviction_std: 0.289153
- volume_direction_conviction_cov: 0.578894
- volume_direction_imbalance_mean: -0.004958
- volume_direction_imbalance_std: 0.577380
- volume_direction_imbalance_cov: 116.459109
- trend_confirmation_6h_mean: 0.230081
- trend_confirmation_6h_std: 0.143777
- trend_confirmation_6h_cov: 0.624900
- momentum_persistence_3h_mean: -1.381319
- momentum_persistence_3h_std: 33.980913
- momentum_persistence_3h_cov: 24.600341
- vol_momentum_sync_mean: 0.174749
- vol_momentum_sync_std: 0.315027
- vol_momentum_sync_cov: 1.802739
- range_momentum_divergence_mean: 0.999850
- range_momentum_divergence_std: 0.000086
- range_momentum_divergence_cov: 0.000086
- volume_concentration_ratio_3h_mean: 0.433879
- volume_concentration_ratio_3h_std: 0.154313
- volume_concentration_ratio_3h_cov: 0.355658
- pressure_ratio_mean: 189279010816.000000
- pressure_ratio_std: 3261171761152.000000
- pressure_ratio_cov: 17.229442
- kyle_lambda_proxy_mean: 219.276018
- kyle_lambda_proxy_std: 130.722934
- kyle_lambda_proxy_cov: 0.596157
- reversal_intensity_mean: 0.001577
- reversal_intensity_std: 0.002500
- reversal_intensity_cov: 1.585160
- whipsaw_count_mean: 6.475524
- whipsaw_count_std: 1.729687
- whipsaw_count_cov: 0.267111
- vol_clustering_mean: 0.399884
- vol_clustering_std: 0.114301
- vol_clustering_cov: 0.285834
- vol_regime_change_mean: -0.011928
- vol_regime_change_std: 0.226484
- vol_regime_change_cov: 18.986826
- efficiency_ratio_mean: 659.722873
- efficiency_ratio_std: 481.245175
- efficiency_ratio_cov: 0.729466
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.000061
- forward_return_std: 0.004284
- forward_return_cov: 70.101370
- forward_return_positive_rate: 0.491841
- forward_return_negative_rate: 0.506993
- forward_return_sharpe_like: -0.014265
- forward_return_mar_like: -0.000379
- forward_return_tail_loss_p95: 0.007418
- adverse_selection_rate: 0.506993
- structural_low_fraction: 0.017462
- transient_gap_fraction: 0.000000

