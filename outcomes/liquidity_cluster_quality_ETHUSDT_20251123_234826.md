# Liquidity Cluster Quality Report

**Symbol:** ETHUSDT  \n**Assessment time:** 2025-11-23T23:48:26.674895

## Overall Quality

- Overall quality score: **0.4459**

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

- Effort/Result CoV separation score: 0.9483
- Returns CoV separation score: 0.6828

## Effort vs Result Separation

- Effort/Result separation score: 0.0744
- Ghost vs Valid contrast: -0.0273
- Absorption vs Valid contrast: 0.1218

## Trap / Ghost Behavior

- Ghost reversal rate: 0.1385
- Ghost false-trend rate: 0.6039

## Absorption Behavior

- Absorption reversal rate: 0.5024
- Absorption follow-through rate: 0.3345

## Trend Confirmation & Apathy

- Valid trend follow-through (mean fwd return): 0.000134
- Apathy noise fraction: 0.2793

## Class Balance

- Class balance score: 0.9393
- Number of regimes: 5
- Number of samples: 33947

## Per-Regime Metrics

### Regime 0

- n_samples: 8533.000000
- ghost_ratio_mean: 0.143132
- ghost_ratio_std: 0.076546
- ghost_ratio_cov: 0.534791
- absorption_ratio_mean: 8.866799
- absorption_ratio_std: 4.319828
- absorption_ratio_cov: 0.487191
- rvol_24_mean: 0.886633
- rvol_24_std: 0.445983
- rvol_24_cov: 0.503008
- rvol_20_mean: 0.817100
- rvol_20_std: 0.466007
- rvol_20_cov: 0.570318
- volume_efficiency_ratio_mean: 1469.479741
- volume_efficiency_ratio_std: 456.784525
- volume_efficiency_ratio_cov: 0.310848
- intraday_close_ratio_mean: 11886.688277
- intraday_close_ratio_std: 20086.340019
- intraday_close_ratio_cov: 1.689818
- amihud_spike_ratio_scaled_mean: -0.164995
- amihud_spike_ratio_scaled_std: 0.878579
- amihud_spike_ratio_scaled_cov: 5.324889
- rvol_168_scaled_mean: -0.249052
- rvol_168_scaled_std: 0.636851
- rvol_168_scaled_cov: 2.557105
- cumulative_delta_divergence_mean: 0.950155
- cumulative_delta_divergence_std: 0.720118
- cumulative_delta_divergence_cov: 0.757895
- volume_direction_conviction_mean: 0.477719
- volume_direction_conviction_std: 0.283564
- volume_direction_conviction_cov: 0.593579
- volume_direction_imbalance_mean: 0.012808
- volume_direction_imbalance_std: 0.555416
- volume_direction_imbalance_cov: 43.364865
- trend_confirmation_6h_mean: 0.222111
- trend_confirmation_6h_std: 0.140742
- trend_confirmation_6h_cov: 0.633658
- momentum_persistence_3h_mean: -6.456614
- momentum_persistence_3h_std: 460.335220
- momentum_persistence_3h_cov: 71.296694
- vol_momentum_sync_mean: 0.049294
- vol_momentum_sync_std: 0.183942
- vol_momentum_sync_cov: 3.731501
- range_momentum_divergence_mean: 0.999777
- range_momentum_divergence_std: 0.000152
- range_momentum_divergence_cov: 0.000152
- volume_concentration_ratio_3h_mean: 0.422356
- volume_concentration_ratio_3h_std: 0.148478
- volume_concentration_ratio_3h_cov: 0.351548
- pressure_ratio_mean: 462228193280.000000
- pressure_ratio_std: 10756050386944.000000
- pressure_ratio_cov: 23.270001
- kyle_lambda_proxy_mean: 13842557.299052
- kyle_lambda_proxy_std: 6789118.379204
- kyle_lambda_proxy_cov: 0.490453
- reversal_intensity_mean: 0.001116
- reversal_intensity_std: 0.001644
- reversal_intensity_cov: 1.473127
- whipsaw_count_mean: 6.367632
- whipsaw_count_std: 1.647561
- whipsaw_count_cov: 0.258740
- vol_clustering_mean: 0.355522
- vol_clustering_std: 0.105126
- vol_clustering_cov: 0.295696
- vol_regime_change_mean: -0.056069
- vol_regime_change_std: 0.266360
- vol_regime_change_cov: 4.750550
- efficiency_ratio_mean: 398.455557
- efficiency_ratio_std: 348.456865
- efficiency_ratio_cov: 0.874519
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.000075
- forward_return_std: 0.005249
- forward_return_cov: 70.360954

### Regime 1

- n_samples: 3342.000000
- ghost_ratio_mean: 0.264055
- ghost_ratio_std: 0.109378
- ghost_ratio_cov: 0.414225
- absorption_ratio_mean: 4.711539
- absorption_ratio_std: 2.692963
- absorption_ratio_cov: 0.571568
- rvol_24_mean: 1.518744
- rvol_24_std: 0.730498
- rvol_24_cov: 0.480988
- rvol_20_mean: 1.803412
- rvol_20_std: 1.097360
- rvol_20_cov: 0.608491
- volume_efficiency_ratio_mean: 1094.112075
- volume_efficiency_ratio_std: 345.766881
- volume_efficiency_ratio_cov: 0.316025
- intraday_close_ratio_mean: 9357.642618
- intraday_close_ratio_std: 16456.677424
- intraday_close_ratio_cov: 1.758635
- amihud_spike_ratio_scaled_mean: 0.049780
- amihud_spike_ratio_scaled_std: 0.951941
- amihud_spike_ratio_scaled_cov: 19.122770
- rvol_168_scaled_mean: 1.042039
- rvol_168_scaled_std: 1.233059
- rvol_168_scaled_cov: 1.183314
- cumulative_delta_divergence_mean: 1.029584
- cumulative_delta_divergence_std: 0.775834
- cumulative_delta_divergence_cov: 0.753541
- volume_direction_conviction_mean: 0.508138
- volume_direction_conviction_std: 0.274177
- volume_direction_conviction_cov: 0.539573
- volume_direction_imbalance_mean: 0.015295
- volume_direction_imbalance_std: 0.577252
- volume_direction_imbalance_cov: 37.740384
- trend_confirmation_6h_mean: 0.243409
- trend_confirmation_6h_std: 0.142126
- trend_confirmation_6h_cov: 0.583897
- momentum_persistence_3h_mean: -0.087440
- momentum_persistence_3h_std: 17.361677
- momentum_persistence_3h_cov: 198.555241
- vol_momentum_sync_mean: 0.275324
- vol_momentum_sync_std: 0.344091
- vol_momentum_sync_cov: 1.249770
- range_momentum_divergence_mean: 0.999766
- range_momentum_divergence_std: 0.000137
- range_momentum_divergence_cov: 0.000137
- volume_concentration_ratio_3h_mean: 0.440809
- volume_concentration_ratio_3h_std: 0.168712
- volume_concentration_ratio_3h_cov: 0.382733
- pressure_ratio_mean: 2560724041728.000000
- pressure_ratio_std: 87220350353408.000000
- pressure_ratio_cov: 34.060816
- kyle_lambda_proxy_mean: 14308282.844298
- kyle_lambda_proxy_std: 8388896.730197
- kyle_lambda_proxy_cov: 0.586297
- reversal_intensity_mean: 0.004284
- reversal_intensity_std: 0.006702
- reversal_intensity_cov: 1.564571
- whipsaw_count_mean: 6.191801
- whipsaw_count_std: 1.641469
- whipsaw_count_cov: 0.265104
- vol_clustering_mean: 0.400746
- vol_clustering_std: 0.104256
- vol_clustering_cov: 0.260155
- vol_regime_change_mean: 0.023510
- vol_regime_change_std: 0.240465
- vol_regime_change_cov: 10.228424
- efficiency_ratio_mean: 608.763523
- efficiency_ratio_std: 493.476389
- efficiency_ratio_cov: 0.810621
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: 0.000134
- forward_return_std: 0.009823
- forward_return_cov: 73.182236

### Regime 2

- n_samples: 3722.000000
- ghost_ratio_mean: 0.222163
- ghost_ratio_std: 0.111657
- ghost_ratio_cov: 0.502589
- absorption_ratio_mean: 6.017951
- absorption_ratio_std: 3.654487
- absorption_ratio_cov: 0.607264
- rvol_24_mean: 1.522077
- rvol_24_std: 0.762720
- rvol_24_cov: 0.501105
- rvol_20_mean: 2.061461
- rvol_20_std: 1.773265
- rvol_20_cov: 0.860198
- volume_efficiency_ratio_mean: 2299.355050
- volume_efficiency_ratio_std: 953.807532
- volume_efficiency_ratio_cov: 0.414815
- intraday_close_ratio_mean: 8963.867589
- intraday_close_ratio_std: 16840.233939
- intraday_close_ratio_cov: 1.878679
- amihud_spike_ratio_scaled_mean: -0.117972
- amihud_spike_ratio_scaled_std: 0.893810
- amihud_spike_ratio_scaled_cov: 7.576480
- rvol_168_scaled_mean: 1.122545
- rvol_168_scaled_std: 1.357246
- rvol_168_scaled_cov: 1.209080
- cumulative_delta_divergence_mean: 1.013799
- cumulative_delta_divergence_std: 0.772935
- cumulative_delta_divergence_cov: 0.762414
- volume_direction_conviction_mean: 0.466901
- volume_direction_conviction_std: 0.272047
- volume_direction_conviction_cov: 0.582665
- volume_direction_imbalance_mean: -0.000043
- volume_direction_imbalance_std: 0.540430
- volume_direction_imbalance_cov: 12509.293578
- trend_confirmation_6h_mean: 0.223160
- trend_confirmation_6h_std: 0.138241
- trend_confirmation_6h_cov: 0.619472
- momentum_persistence_3h_mean: -1.673254
- momentum_persistence_3h_std: 183.161846
- momentum_persistence_3h_cov: 109.464457
- vol_momentum_sync_mean: 0.190565
- vol_momentum_sync_std: 0.302805
- vol_momentum_sync_cov: 1.588982
- range_momentum_divergence_mean: 0.999707
- range_momentum_divergence_std: 0.000183
- range_momentum_divergence_cov: 0.000183
- volume_concentration_ratio_3h_mean: 0.448612
- volume_concentration_ratio_3h_std: 0.188251
- volume_concentration_ratio_3h_cov: 0.419629
- pressure_ratio_mean: 320465207296.000000
- pressure_ratio_std: 10287807725568.000000
- pressure_ratio_cov: 32.102729
- kyle_lambda_proxy_mean: 18038137.052391
- kyle_lambda_proxy_std: 9149937.114528
- kyle_lambda_proxy_cov: 0.507255
- reversal_intensity_mean: 0.003799
- reversal_intensity_std: 0.007510
- reversal_intensity_cov: 1.976473
- whipsaw_count_mean: 6.163084
- whipsaw_count_std: 1.635570
- whipsaw_count_cov: 0.265382
- vol_clustering_mean: 0.391020
- vol_clustering_std: 0.100140
- vol_clustering_cov: 0.256099
- vol_regime_change_mean: 0.013703
- vol_regime_change_std: 0.248821
- vol_regime_change_cov: 18.158238
- efficiency_ratio_mean: 475.254850
- efficiency_ratio_std: 375.745610
- efficiency_ratio_cov: 0.790619
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: 0.000134
- forward_return_std: 0.011836
- forward_return_cov: 88.337551

### Regime 3

- n_samples: 7157.000000
- ghost_ratio_mean: 0.250040
- ghost_ratio_std: 0.102674
- ghost_ratio_cov: 0.410630
- absorption_ratio_mean: 4.889540
- absorption_ratio_std: 2.617018
- absorption_ratio_cov: 0.535228
- rvol_24_mean: 1.105861
- rvol_24_std: 0.541923
- rvol_24_cov: 0.490047
- rvol_20_mean: 1.101292
- rvol_20_std: 0.572844
- rvol_20_cov: 0.520157
- volume_efficiency_ratio_mean: 553.368148
- volume_efficiency_ratio_std: 179.056586
- volume_efficiency_ratio_cov: 0.323576
- intraday_close_ratio_mean: 12434.816586
- intraday_close_ratio_std: 18857.180832
- intraday_close_ratio_cov: 1.516482
- amihud_spike_ratio_scaled_mean: 0.113324
- amihud_spike_ratio_scaled_std: 1.011357
- amihud_spike_ratio_scaled_cov: 8.924479
- rvol_168_scaled_mean: 0.151568
- rvol_168_scaled_std: 0.764362
- rvol_168_scaled_cov: 5.043032
- cumulative_delta_divergence_mean: 0.984822
- cumulative_delta_divergence_std: 0.740376
- cumulative_delta_divergence_cov: 0.751786
- volume_direction_conviction_mean: 0.527793
- volume_direction_conviction_std: 0.280703
- volume_direction_conviction_cov: 0.531843
- volume_direction_imbalance_mean: 0.036102
- volume_direction_imbalance_std: 0.596737
- volume_direction_imbalance_cov: 16.529098
- trend_confirmation_6h_mean: 0.249156
- trend_confirmation_6h_std: 0.144577
- trend_confirmation_6h_cov: 0.580267
- momentum_persistence_3h_mean: 0.388485
- momentum_persistence_3h_std: 58.408704
- momentum_persistence_3h_cov: 150.349915
- vol_momentum_sync_mean: 0.222740
- vol_momentum_sync_std: 0.340415
- vol_momentum_sync_cov: 1.528311
- range_momentum_divergence_mean: 0.999832
- range_momentum_divergence_std: 0.000099
- range_momentum_divergence_cov: 0.000099
- volume_concentration_ratio_3h_mean: 0.414452
- volume_concentration_ratio_3h_std: 0.140077
- volume_concentration_ratio_3h_cov: 0.337982
- pressure_ratio_mean: 775297236992.000000
- pressure_ratio_std: 40418687320064.000000
- pressure_ratio_cov: 52.133150
- kyle_lambda_proxy_mean: 11723188.166065
- kyle_lambda_proxy_std: 7149908.414357
- kyle_lambda_proxy_cov: 0.609895
- reversal_intensity_mean: 0.003427
- reversal_intensity_std: 0.004508
- reversal_intensity_cov: 1.315687
- whipsaw_count_mean: 6.358261
- whipsaw_count_std: 1.671329
- whipsaw_count_cov: 0.262860
- vol_clustering_mean: 0.391379
- vol_clustering_std: 0.105099
- vol_clustering_cov: 0.268536
- vol_regime_change_mean: -0.007143
- vol_regime_change_std: 0.228996
- vol_regime_change_cov: 32.057483
- efficiency_ratio_mean: 702.116213
- efficiency_ratio_std: 558.696217
- efficiency_ratio_cov: 0.795732
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.000017
- forward_return_std: 0.008014
- forward_return_cov: 469.563989

### Regime 4

- n_samples: 11193.000000
- ghost_ratio_mean: 0.172330
- ghost_ratio_std: 0.081180
- ghost_ratio_cov: 0.471072
- absorption_ratio_mean: 7.157651
- absorption_ratio_std: 3.456022
- absorption_ratio_cov: 0.482843
- rvol_24_mean: 0.717398
- rvol_24_std: 0.315375
- rvol_24_cov: 0.439609
- rvol_20_mean: 0.570437
- rvol_20_std: 0.266327
- rvol_20_cov: 0.466883
- volume_efficiency_ratio_mean: 556.632045
- volume_efficiency_ratio_std: 211.373307
- volume_efficiency_ratio_cov: 0.379736
- intraday_close_ratio_mean: 15797.809656
- intraday_close_ratio_std: 22827.698634
- intraday_close_ratio_cov: 1.444991
- amihud_spike_ratio_scaled_mean: 0.077688
- amihud_spike_ratio_scaled_std: 1.101592
- amihud_spike_ratio_scaled_cov: 14.179622
- rvol_168_scaled_mean: -0.591461
- rvol_168_scaled_std: 0.356713
- rvol_168_scaled_cov: 0.603105
- cumulative_delta_divergence_mean: 0.936656
- cumulative_delta_divergence_std: 0.712311
- cumulative_delta_divergence_cov: 0.760483
- volume_direction_conviction_mean: 0.498422
- volume_direction_conviction_std: 0.286354
- volume_direction_conviction_cov: 0.574522
- volume_direction_imbalance_mean: 0.033261
- volume_direction_imbalance_std: 0.573881
- volume_direction_imbalance_cov: 17.253731
- trend_confirmation_6h_mean: 0.227908
- trend_confirmation_6h_std: 0.142045
- trend_confirmation_6h_cov: 0.623256
- momentum_persistence_3h_mean: -0.519767
- momentum_persistence_3h_std: 102.684097
- momentum_persistence_3h_cov: 197.557887
- vol_momentum_sync_mean: 0.062894
- vol_momentum_sync_std: 0.213612
- vol_momentum_sync_cov: 3.396391
- range_momentum_divergence_mean: 0.999844
- range_momentum_divergence_std: 0.000204
- range_momentum_divergence_cov: 0.000204
- volume_concentration_ratio_3h_mean: 0.414668
- volume_concentration_ratio_3h_std: 0.141805
- volume_concentration_ratio_3h_cov: 0.341971
- pressure_ratio_mean: 590407270400.000000
- pressure_ratio_std: 11045127061504.000000
- pressure_ratio_cov: 18.707641
- kyle_lambda_proxy_mean: 10562394.357799
- kyle_lambda_proxy_std: 5912415.554173
- kyle_lambda_proxy_cov: 0.559761
- reversal_intensity_mean: 0.001415
- reversal_intensity_std: 0.001831
- reversal_intensity_cov: 1.294748
- whipsaw_count_mean: 6.534977
- whipsaw_count_std: 1.690399
- whipsaw_count_cov: 0.258669
- vol_clustering_mean: 0.349943
- vol_clustering_std: 0.105439
- vol_clustering_cov: 0.301303
- vol_regime_change_mean: -0.056710
- vol_regime_change_std: 0.246048
- vol_regime_change_cov: 4.338730
- efficiency_ratio_mean: 544.419579
- efficiency_ratio_std: 473.002995
- efficiency_ratio_cov: 0.868821
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.000025
- forward_return_std: 0.005349
- forward_return_cov: 214.633116

