# Liquidity Cluster Quality Report

**Symbol:** ETHUSDT  \n**Assessment time:** 2026-01-02T19:36:07.005376

## Overall Quality

- Overall quality score: **0.5018**

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

- Effort/Result CoV separation score: 0.7505
- Returns CoV separation score: 0.4868

## Effort vs Result Separation

- Effort/Result separation score: 0.2003
- Ghost vs Valid contrast: -0.1747
- Absorption vs Valid contrast: 0.2314

## Trap / Ghost Behavior

- Ghost reversal rate: 0.3535
- Ghost false-trend rate: 0.2176

## Absorption Behavior

- Absorption reversal rate: 0.4950
- Absorption follow-through rate: 0.1894

## Trend Confirmation & Apathy

- Valid trend follow-through (mean fwd return): -0.000023
- Apathy noise fraction: 0.4877

## Class Balance

- Class balance score: 0.9337
- Number of regimes: 5
- Number of samples: 142619

## Per-Regime Metrics

### Regime 0

- n_samples: 25582.000000
- ghost_ratio_mean: 0.119870
- ghost_ratio_std: 0.059161
- ghost_ratio_cov: 0.493543
- absorption_ratio_mean: 10.172265
- absorption_ratio_std: 4.503628
- absorption_ratio_cov: 0.442736
- rvol_24_mean: 0.777464
- rvol_24_std: 0.447484
- rvol_24_cov: 0.575569
- rvol_20_mean: 0.785913
- rvol_20_std: 0.455395
- rvol_20_cov: 0.579447
- volume_efficiency_ratio_mean: 748.561401
- volume_efficiency_ratio_std: 163.519989
- volume_efficiency_ratio_cov: 0.218446
- intraday_close_ratio_mean: 9908.462770
- intraday_close_ratio_std: 16142.337017
- intraday_close_ratio_cov: 1.629146
- amihud_spike_ratio_scaled_mean: -0.318790
- amihud_spike_ratio_scaled_std: 0.747664
- amihud_spike_ratio_scaled_cov: 2.345320
- rvol_168_scaled_mean: -0.211042
- rvol_168_scaled_std: 0.732619
- rvol_168_scaled_cov: 3.471439
- cumulative_delta_divergence_mean: 1.094986
- cumulative_delta_divergence_std: 0.820546
- cumulative_delta_divergence_cov: 0.749367
- volume_direction_conviction_mean: 0.534979
- volume_direction_conviction_std: 0.307908
- volume_direction_conviction_cov: 0.575552
- volume_direction_imbalance_mean: 0.013356
- volume_direction_imbalance_std: 0.617125
- volume_direction_imbalance_cov: 46.204980
- trend_confirmation_6h_mean: 0.253001
- trend_confirmation_6h_std: 0.157002
- trend_confirmation_6h_cov: 0.620561
- momentum_persistence_3h_mean: 24.905955
- momentum_persistence_3h_std: 5308.087178
- momentum_persistence_3h_cov: 213.125218
- vol_momentum_sync_mean: 0.062601
- vol_momentum_sync_std: 0.213829
- vol_momentum_sync_cov: 3.415751
- range_momentum_divergence_mean: 0.999747
- range_momentum_divergence_std: 0.000177
- range_momentum_divergence_cov: 0.000177
- volume_concentration_ratio_3h_mean: 0.443092
- volume_concentration_ratio_3h_std: 0.154349
- volume_concentration_ratio_3h_cov: 0.348344
- pressure_ratio_mean: 192213286912.000000
- pressure_ratio_std: 2595812016128.000000
- pressure_ratio_cov: 13.504852
- kyle_lambda_proxy_mean: 234.909457
- kyle_lambda_proxy_std: 164.136925
- kyle_lambda_proxy_cov: 0.698724
- reversal_intensity_mean: 0.000514
- reversal_intensity_std: 0.000769
- reversal_intensity_cov: 1.497814
- whipsaw_count_mean: 6.314831
- whipsaw_count_std: 1.628851
- whipsaw_count_cov: 0.257941
- vol_clustering_mean: 0.367809
- vol_clustering_std: 0.113991
- vol_clustering_cov: 0.309919
- vol_regime_change_mean: -0.054710
- vol_regime_change_std: 0.240977
- vol_regime_change_cov: 4.404651
- efficiency_ratio_mean: 352.764170
- efficiency_ratio_std: 310.333120
- efficiency_ratio_cov: 0.879718
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: 0.000008
- forward_return_std: 0.002357
- forward_return_cov: 305.747223
- forward_return_positive_rate: 0.500782
- forward_return_negative_rate: 0.497889
- forward_return_sharpe_like: 0.003270
- forward_return_mar_like: 0.000028
- forward_return_tail_loss_p95: 0.003322
- adverse_selection_rate: 0.497889
- structural_low_fraction: 0.097960
- transient_gap_fraction: 0.000000

### Regime 1

- n_samples: 14450.000000
- ghost_ratio_mean: 0.267967
- ghost_ratio_std: 0.112772
- ghost_ratio_cov: 0.420844
- absorption_ratio_mean: 4.559980
- absorption_ratio_std: 2.464313
- absorption_ratio_cov: 0.540422
- rvol_24_mean: 1.632466
- rvol_24_std: 0.914281
- rvol_24_cov: 0.560061
- rvol_20_mean: 1.711791
- rvol_20_std: 1.199958
- rvol_20_cov: 0.700996
- volume_efficiency_ratio_mean: 538.440796
- volume_efficiency_ratio_std: 134.929871
- volume_efficiency_ratio_cov: 0.250594
- intraday_close_ratio_mean: 9419.475925
- intraday_close_ratio_std: 14711.346197
- intraday_close_ratio_cov: 1.561801
- amihud_spike_ratio_scaled_mean: -0.086280
- amihud_spike_ratio_scaled_std: 0.886335
- amihud_spike_ratio_scaled_cov: 10.272813
- rvol_168_scaled_mean: 1.072451
- rvol_168_scaled_std: 1.170114
- rvol_168_scaled_cov: 1.091065
- cumulative_delta_divergence_mean: 1.128600
- cumulative_delta_divergence_std: 0.846389
- cumulative_delta_divergence_cov: 0.749946
- volume_direction_conviction_mean: 0.529101
- volume_direction_conviction_std: 0.283305
- volume_direction_conviction_cov: 0.535446
- volume_direction_imbalance_mean: 0.010746
- volume_direction_imbalance_std: 0.600095
- volume_direction_imbalance_cov: 55.844949
- trend_confirmation_6h_mean: 0.257947
- trend_confirmation_6h_std: 0.148724
- trend_confirmation_6h_cov: 0.576570
- momentum_persistence_3h_mean: -18.998065
- momentum_persistence_3h_std: 3368.687876
- momentum_persistence_3h_cov: 177.317422
- vol_momentum_sync_mean: 0.261429
- vol_momentum_sync_std: 0.348209
- vol_momentum_sync_cov: 1.331947
- range_momentum_divergence_mean: 0.999769
- range_momentum_divergence_std: 0.000145
- range_momentum_divergence_cov: 0.000145
- volume_concentration_ratio_3h_mean: 0.453524
- volume_concentration_ratio_3h_std: 0.178081
- volume_concentration_ratio_3h_cov: 0.392660
- pressure_ratio_mean: 894723358720.000000
- pressure_ratio_std: 26825023553536.000000
- pressure_ratio_cov: 29.981360
- kyle_lambda_proxy_mean: 395.673462
- kyle_lambda_proxy_std: 285.877607
- kyle_lambda_proxy_cov: 0.722509
- reversal_intensity_mean: 0.002132
- reversal_intensity_std: 0.003380
- reversal_intensity_cov: 1.585343
- whipsaw_count_mean: 6.057855
- whipsaw_count_std: 1.689423
- whipsaw_count_cov: 0.278881
- vol_clustering_mean: 0.429565
- vol_clustering_std: 0.111862
- vol_clustering_cov: 0.260408
- vol_regime_change_mean: 0.050488
- vol_regime_change_std: 0.220974
- vol_regime_change_cov: 4.376779
- efficiency_ratio_mean: 638.228829
- efficiency_ratio_std: 546.123169
- efficiency_ratio_cov: 0.855686
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.000023
- forward_return_std: 0.005340
- forward_return_cov: 232.624284
- forward_return_positive_rate: 0.505052
- forward_return_negative_rate: 0.494325
- forward_return_sharpe_like: -0.004299
- forward_return_mar_like: -0.000030
- forward_return_tail_loss_p95: 0.007969
- adverse_selection_rate: 0.494325
- structural_low_fraction: 0.010657
- transient_gap_fraction: 0.000000

### Regime 2

- n_samples: 17655.000000
- ghost_ratio_mean: 0.198921
- ghost_ratio_std: 0.122013
- ghost_ratio_cov: 0.613373
- absorption_ratio_mean: 7.305231
- absorption_ratio_std: 4.755907
- absorption_ratio_cov: 0.651028
- rvol_24_mean: 1.595407
- rvol_24_std: 0.987237
- rvol_24_cov: 0.618800
- rvol_20_mean: 1.817719
- rvol_20_std: 1.688509
- rvol_20_cov: 0.928916
- volume_efficiency_ratio_mean: 1327.099854
- volume_efficiency_ratio_std: 994.924561
- volume_efficiency_ratio_cov: 0.749698
- intraday_close_ratio_mean: 8435.996760
- intraday_close_ratio_std: 14576.508523
- intraday_close_ratio_cov: 1.727894
- amihud_spike_ratio_scaled_mean: -0.483733
- amihud_spike_ratio_scaled_std: 0.672070
- amihud_spike_ratio_scaled_cov: 1.389341
- rvol_168_scaled_mean: 0.991363
- rvol_168_scaled_std: 1.258339
- rvol_168_scaled_cov: 1.269303
- cumulative_delta_divergence_mean: 1.117466
- cumulative_delta_divergence_std: 0.831930
- cumulative_delta_divergence_cov: 0.744479
- volume_direction_conviction_mean: 0.496391
- volume_direction_conviction_std: 0.289324
- volume_direction_conviction_cov: 0.582854
- volume_direction_imbalance_mean: -0.000976
- volume_direction_imbalance_std: 0.574565
- volume_direction_imbalance_cov: 588.688921
- trend_confirmation_6h_mean: 0.241289
- trend_confirmation_6h_std: 0.150673
- trend_confirmation_6h_cov: 0.624451
- momentum_persistence_3h_mean: -0.314343
- momentum_persistence_3h_std: 54.191933
- momentum_persistence_3h_cov: 172.397381
- vol_momentum_sync_mean: 0.151445
- vol_momentum_sync_std: 0.288409
- vol_momentum_sync_cov: 1.904372
- range_momentum_divergence_mean: 0.999710
- range_momentum_divergence_std: 0.000184
- range_momentum_divergence_cov: 0.000184
- volume_concentration_ratio_3h_mean: 0.470838
- volume_concentration_ratio_3h_std: 0.203557
- volume_concentration_ratio_3h_cov: 0.432330
- pressure_ratio_mean: 1136755867648.000000
- pressure_ratio_std: 59649193672704.000000
- pressure_ratio_cov: 52.473179
- kyle_lambda_proxy_mean: 685.855138
- kyle_lambda_proxy_std: 592.143617
- kyle_lambda_proxy_cov: 0.863365
- reversal_intensity_mean: 0.001520
- reversal_intensity_std: 0.003643
- reversal_intensity_cov: 2.397416
- whipsaw_count_mean: 6.068366
- whipsaw_count_std: 1.684240
- whipsaw_count_cov: 0.277544
- vol_clustering_mean: 0.403200
- vol_clustering_std: 0.117293
- vol_clustering_cov: 0.290906
- vol_regime_change_mean: 0.023287
- vol_regime_change_std: 0.242443
- vol_regime_change_cov: 10.411109
- efficiency_ratio_mean: 441.802932
- efficiency_ratio_std: 376.197122
- efficiency_ratio_cov: 0.851504
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: 0.000043
- forward_return_std: 0.005805
- forward_return_cov: 134.724892
- forward_return_positive_rate: 0.504956
- forward_return_negative_rate: 0.493685
- forward_return_sharpe_like: 0.007422
- forward_return_mar_like: 0.000064
- forward_return_tail_loss_p95: 0.007769
- adverse_selection_rate: 0.493685
- structural_low_fraction: 0.030926
- transient_gap_fraction: 0.000000

### Regime 3

- n_samples: 52704.000000
- ghost_ratio_mean: 0.188265
- ghost_ratio_std: 0.088585
- ghost_ratio_cov: 0.470532
- absorption_ratio_mean: 6.569448
- absorption_ratio_std: 3.229144
- absorption_ratio_cov: 0.491540
- rvol_24_mean: 0.673605
- rvol_24_std: 0.393139
- rvol_24_cov: 0.583634
- rvol_20_mean: 0.683678
- rvol_20_std: 0.400166
- rvol_20_cov: 0.585314
- volume_efficiency_ratio_mean: 207.058655
- volume_efficiency_ratio_std: 61.419983
- volume_efficiency_ratio_cov: 0.296631
- intraday_close_ratio_mean: 12741.209695
- intraday_close_ratio_std: 17316.516798
- intraday_close_ratio_cov: 1.359095
- amihud_spike_ratio_scaled_mean: 0.384165
- amihud_spike_ratio_scaled_std: 1.143763
- amihud_spike_ratio_scaled_cov: 2.977271
- rvol_168_scaled_mean: -0.456329
- rvol_168_scaled_std: 0.675374
- rvol_168_scaled_cov: 1.480017
- cumulative_delta_divergence_mean: 1.034246
- cumulative_delta_divergence_std: 0.775480
- cumulative_delta_divergence_cov: 0.749802
- volume_direction_conviction_mean: 0.562797
- volume_direction_conviction_std: 0.296483
- volume_direction_conviction_cov: 0.526802
- volume_direction_imbalance_mean: 0.027586
- volume_direction_imbalance_std: 0.635522
- volume_direction_imbalance_cov: 23.037817
- trend_confirmation_6h_mean: 0.264260
- trend_confirmation_6h_std: 0.152018
- trend_confirmation_6h_cov: 0.575260
- momentum_persistence_3h_mean: 0.462395
- momentum_persistence_3h_std: 1028.727909
- momentum_persistence_3h_cov: 2224.781854
- vol_momentum_sync_mean: 0.147603
- vol_momentum_sync_std: 0.310140
- vol_momentum_sync_cov: 2.101177
- range_momentum_divergence_mean: 0.999839
- range_momentum_divergence_std: 0.000101
- range_momentum_divergence_cov: 0.000101
- volume_concentration_ratio_3h_mean: 0.428099
- volume_concentration_ratio_3h_std: 0.141296
- volume_concentration_ratio_3h_cov: 0.330056
- pressure_ratio_mean: 364145803264.000000
- pressure_ratio_std: 6195317309440.000000
- pressure_ratio_cov: 17.013288
- kyle_lambda_proxy_mean: 121.668248
- kyle_lambda_proxy_std: 79.064838
- kyle_lambda_proxy_cov: 0.649840
- reversal_intensity_mean: 0.001209
- reversal_intensity_std: 0.001761
- reversal_intensity_cov: 1.456344
- whipsaw_count_mean: 6.405457
- whipsaw_count_std: 1.637280
- whipsaw_count_cov: 0.255607
- vol_clustering_mean: 0.384401
- vol_clustering_std: 0.109653
- vol_clustering_cov: 0.285257
- vol_regime_change_mean: -0.046936
- vol_regime_change_std: 0.219194
- vol_regime_change_cov: 4.670115
- efficiency_ratio_mean: 556.440170
- efficiency_ratio_std: 452.212548
- efficiency_ratio_cov: 0.812689
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.000006
- forward_return_std: 0.003057
- forward_return_cov: 553.230965
- forward_return_positive_rate: 0.503852
- forward_return_negative_rate: 0.495351
- forward_return_sharpe_like: -0.001807
- forward_return_mar_like: -0.000009
- forward_return_tail_loss_p95: 0.004600
- adverse_selection_rate: 0.495351
- structural_low_fraction: 0.092574
- transient_gap_fraction: 0.000000

### Regime 4

- n_samples: 32228.000000
- ghost_ratio_mean: 0.174185
- ghost_ratio_std: 0.085735
- ghost_ratio_cov: 0.492206
- absorption_ratio_mean: 7.228028
- absorption_ratio_std: 3.654571
- absorption_ratio_cov: 0.505611
- rvol_24_mean: 0.862572
- rvol_24_std: 0.514271
- rvol_24_cov: 0.596207
- rvol_20_mean: 0.869499
- rvol_20_std: 0.527583
- rvol_20_cov: 0.606766
- volume_efficiency_ratio_mean: 401.562195
- volume_efficiency_ratio_std: 65.973198
- volume_efficiency_ratio_cov: 0.164291
- intraday_close_ratio_mean: 11302.340939
- intraday_close_ratio_std: 16650.672066
- intraday_close_ratio_cov: 1.473206
- amihud_spike_ratio_scaled_mean: -0.046380
- amihud_spike_ratio_scaled_std: 0.938711
- amihud_spike_ratio_scaled_cov: 20.239722
- rvol_168_scaled_mean: -0.092762
- rvol_168_scaled_std: 0.813577
- rvol_168_scaled_cov: 8.770587
- cumulative_delta_divergence_mean: 1.083763
- cumulative_delta_divergence_std: 0.810983
- cumulative_delta_divergence_cov: 0.748302
- volume_direction_conviction_mean: 0.539626
- volume_direction_conviction_std: 0.298579
- volume_direction_conviction_cov: 0.553307
- volume_direction_imbalance_mean: 0.021372
- volume_direction_imbalance_std: 0.616359
- volume_direction_imbalance_cov: 28.838922
- trend_confirmation_6h_mean: 0.257468
- trend_confirmation_6h_std: 0.153717
- trend_confirmation_6h_cov: 0.597033
- momentum_persistence_3h_mean: -13.081138
- momentum_persistence_3h_std: 2276.450785
- momentum_persistence_3h_cov: 174.025444
- vol_momentum_sync_mean: 0.124486
- vol_momentum_sync_std: 0.285816
- vol_momentum_sync_cov: 2.295970
- range_momentum_divergence_mean: 0.999796
- range_momentum_divergence_std: 0.000138
- range_momentum_divergence_cov: 0.000138
- volume_concentration_ratio_3h_mean: 0.432092
- volume_concentration_ratio_3h_std: 0.149216
- volume_concentration_ratio_3h_cov: 0.345334
- pressure_ratio_mean: 390269763584.000000
- pressure_ratio_std: 6821490720768.000000
- pressure_ratio_cov: 17.478912
- kyle_lambda_proxy_mean: 191.157258
- kyle_lambda_proxy_std: 122.448203
- kyle_lambda_proxy_cov: 0.640563
- reversal_intensity_mean: 0.000946
- reversal_intensity_std: 0.001468
- reversal_intensity_cov: 1.551032
- whipsaw_count_mean: 6.257416
- whipsaw_count_std: 1.649534
- whipsaw_count_cov: 0.263613
- vol_clustering_mean: 0.389105
- vol_clustering_std: 0.113325
- vol_clustering_cov: 0.291246
- vol_regime_change_mean: -0.033309
- vol_regime_change_std: 0.227906
- vol_regime_change_cov: 6.842110
- efficiency_ratio_mean: 512.482432
- efficiency_ratio_std: 454.028109
- efficiency_ratio_cov: 0.885939
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.000018
- forward_return_std: 0.003144
- forward_return_cov: 171.273527
- forward_return_positive_rate: 0.501226
- forward_return_negative_rate: 0.497316
- forward_return_sharpe_like: -0.005838
- forward_return_mar_like: -0.000021
- forward_return_tail_loss_p95: 0.004652
- adverse_selection_rate: 0.497316
- structural_low_fraction: 0.066216
- transient_gap_fraction: 0.000000

