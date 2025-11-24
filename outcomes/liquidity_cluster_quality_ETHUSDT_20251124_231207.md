# Liquidity Cluster Quality Report

**Symbol:** ETHUSDT  \n**Assessment time:** 2025-11-24T23:12:07.364623

## Overall Quality

- Overall quality score: **0.4538**

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

- Effort/Result CoV separation score: 0.8186
- Returns CoV separation score: 0.5528

## Effort vs Result Separation

- Effort/Result separation score: 0.0853
- Ghost vs Valid contrast: -0.1131
- Absorption vs Valid contrast: 0.0580

## Trap / Ghost Behavior

- Ghost reversal rate: 0.2422
- Ghost false-trend rate: 0.3125

## Absorption Behavior

- Absorption reversal rate: 0.4872
- Absorption follow-through rate: 0.3333

## Trend Confirmation & Apathy

- Valid trend follow-through (mean fwd return): -0.000296
- Apathy noise fraction: 0.2071

## Class Balance

- Class balance score: 0.9772
- Number of regimes: 5
- Number of samples: 720

## Per-Regime Metrics

### Regime 0

- n_samples: 169.000000
- ghost_ratio_mean: 0.155025
- ghost_ratio_std: 0.076997
- ghost_ratio_cov: 0.496678
- absorption_ratio_mean: 7.891312
- absorption_ratio_std: 3.479865
- absorption_ratio_cov: 0.440974
- rvol_24_mean: 0.932513
- rvol_24_std: 0.401130
- rvol_24_cov: 0.430160
- rvol_20_mean: 0.820324
- rvol_20_std: 0.258792
- rvol_20_cov: 0.315476
- volume_efficiency_ratio_mean: 626.467825
- volume_efficiency_ratio_std: 147.427011
- volume_efficiency_ratio_cov: 0.235331
- intraday_close_ratio_mean: 27474.491692
- intraday_close_ratio_std: 47799.137689
- intraday_close_ratio_cov: 1.739764
- amihud_spike_ratio_scaled_mean: -0.470401
- amihud_spike_ratio_scaled_std: 0.603572
- amihud_spike_ratio_scaled_cov: 1.283103
- rvol_168_scaled_mean: -0.213318
- rvol_168_scaled_std: 0.337338
- rvol_168_scaled_cov: 1.581385
- cumulative_delta_divergence_mean: 1.142640
- cumulative_delta_divergence_std: 0.845204
- cumulative_delta_divergence_cov: 0.739694
- volume_direction_conviction_mean: 0.465357
- volume_direction_conviction_std: 0.260746
- volume_direction_conviction_cov: 0.560313
- volume_direction_imbalance_mean: 0.055683
- volume_direction_imbalance_std: 0.531710
- volume_direction_imbalance_cov: 9.548859
- trend_confirmation_6h_mean: 0.236515
- trend_confirmation_6h_std: 0.145761
- trend_confirmation_6h_cov: 0.616286
- momentum_persistence_3h_mean: 2.056644
- momentum_persistence_3h_std: 15.132333
- momentum_persistence_3h_cov: 7.357779
- vol_momentum_sync_mean: 0.049791
- vol_momentum_sync_std: 0.188001
- vol_momentum_sync_cov: 3.775820
- range_momentum_divergence_mean: 0.999911
- range_momentum_divergence_std: 0.000057
- range_momentum_divergence_cov: 0.000057
- volume_concentration_ratio_3h_mean: 0.417467
- volume_concentration_ratio_3h_std: 0.161755
- volume_concentration_ratio_3h_cov: 0.387468
- pressure_ratio_mean: 568893505536.000000
- pressure_ratio_std: 7395615440896.000000
- pressure_ratio_cov: 13.000000
- kyle_lambda_proxy_mean: 14119911.368462
- kyle_lambda_proxy_std: 7900910.841638
- kyle_lambda_proxy_cov: 0.559558
- reversal_intensity_mean: 0.000970
- reversal_intensity_std: 0.001718
- reversal_intensity_cov: 1.770796
- whipsaw_count_mean: 5.654762
- whipsaw_count_std: 1.659623
- whipsaw_count_cov: 0.293491
- vol_clustering_mean: 0.351578
- vol_clustering_std: 0.114633
- vol_clustering_cov: 0.326052
- vol_regime_change_mean: -0.010851
- vol_regime_change_std: 0.252485
- vol_regime_change_cov: 23.268352
- efficiency_ratio_mean: 949.958143
- efficiency_ratio_std: 788.631550
- efficiency_ratio_cov: 0.830175
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: 0.000527
- forward_return_std: 0.004922
- forward_return_cov: 9.332589

### Regime 1

- n_samples: 165.000000
- ghost_ratio_mean: 0.229110
- ghost_ratio_std: 0.114213
- ghost_ratio_cov: 0.498505
- absorption_ratio_mean: 6.281740
- absorption_ratio_std: 5.499596
- absorption_ratio_cov: 0.875489
- rvol_24_mean: 1.346727
- rvol_24_std: 0.545614
- rvol_24_cov: 0.405141
- rvol_20_mean: 1.293225
- rvol_20_std: 0.513137
- rvol_20_cov: 0.396788
- volume_efficiency_ratio_mean: 608.342710
- volume_efficiency_ratio_std: 175.970219
- volume_efficiency_ratio_cov: 0.289262
- intraday_close_ratio_mean: 20615.293235
- intraday_close_ratio_std: 36735.844826
- intraday_close_ratio_cov: 1.781971
- amihud_spike_ratio_scaled_mean: -0.197944
- amihud_spike_ratio_scaled_std: 0.793388
- amihud_spike_ratio_scaled_cov: 4.008151
- rvol_168_scaled_mean: 0.477911
- rvol_168_scaled_std: 0.660372
- rvol_168_scaled_cov: 1.381789
- cumulative_delta_divergence_mean: 1.069362
- cumulative_delta_divergence_std: 0.864012
- cumulative_delta_divergence_cov: 0.807969
- volume_direction_conviction_mean: 0.480369
- volume_direction_conviction_std: 0.280211
- volume_direction_conviction_cov: 0.583324
- volume_direction_imbalance_mean: -0.012462
- volume_direction_imbalance_std: 0.557246
- volume_direction_imbalance_cov: 44.714271
- trend_confirmation_6h_mean: 0.236570
- trend_confirmation_6h_std: 0.155531
- trend_confirmation_6h_cov: 0.657442
- momentum_persistence_3h_mean: 0.222683
- momentum_persistence_3h_std: 3.729038
- momentum_persistence_3h_cov: 16.745960
- vol_momentum_sync_mean: 0.173424
- vol_momentum_sync_std: 0.313022
- vol_momentum_sync_cov: 1.804954
- range_momentum_divergence_mean: 0.999895
- range_momentum_divergence_std: 0.000058
- range_momentum_divergence_cov: 0.000058
- volume_concentration_ratio_3h_mean: 0.418313
- volume_concentration_ratio_3h_std: 0.162487
- volume_concentration_ratio_3h_cov: 0.388434
- pressure_ratio_mean: 4.456707
- pressure_ratio_std: 16.742773
- pressure_ratio_cov: 3.756759
- kyle_lambda_proxy_mean: 15002524.859804
- kyle_lambda_proxy_std: 8754921.963004
- kyle_lambda_proxy_cov: 0.583563
- reversal_intensity_mean: 0.002538
- reversal_intensity_std: 0.003982
- reversal_intensity_cov: 1.568902
- whipsaw_count_mean: 5.812121
- whipsaw_count_std: 1.475771
- whipsaw_count_cov: 0.253913
- vol_clustering_mean: 0.341667
- vol_clustering_std: 0.128524
- vol_clustering_cov: 0.376168
- vol_regime_change_mean: -0.058841
- vol_regime_change_std: 0.294876
- vol_regime_change_cov: 5.011396
- efficiency_ratio_mean: 1321.897036
- efficiency_ratio_std: 889.600823
- efficiency_ratio_cov: 0.672973
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.000296
- forward_return_std: 0.008108
- forward_return_cov: 27.346725

### Regime 2

- n_samples: 78.000000
- ghost_ratio_mean: 0.249227
- ghost_ratio_std: 0.138162
- ghost_ratio_cov: 0.554363
- absorption_ratio_mean: 7.055053
- absorption_ratio_std: 7.252661
- absorption_ratio_cov: 1.028009
- rvol_24_mean: 2.053974
- rvol_24_std: 0.773085
- rvol_24_cov: 0.376385
- rvol_20_mean: 2.872919
- rvol_20_std: 1.864496
- rvol_20_cov: 0.648990
- volume_efficiency_ratio_mean: 882.471558
- volume_efficiency_ratio_std: 387.037724
- volume_efficiency_ratio_cov: 0.438584
- intraday_close_ratio_mean: 21478.089722
- intraday_close_ratio_std: 47290.322561
- intraday_close_ratio_cov: 2.201794
- amihud_spike_ratio_scaled_mean: -0.331128
- amihud_spike_ratio_scaled_std: 0.726778
- amihud_spike_ratio_scaled_cov: 2.194855
- rvol_168_scaled_mean: 1.986870
- rvol_168_scaled_std: 1.125909
- rvol_168_scaled_cov: 0.566675
- cumulative_delta_divergence_mean: 1.092484
- cumulative_delta_divergence_std: 0.864133
- cumulative_delta_divergence_cov: 0.790980
- volume_direction_conviction_mean: 0.495129
- volume_direction_conviction_std: 0.285967
- volume_direction_conviction_cov: 0.577561
- volume_direction_imbalance_mean: -0.068452
- volume_direction_imbalance_std: 0.570410
- volume_direction_imbalance_cov: 8.333008
- trend_confirmation_6h_mean: 0.238698
- trend_confirmation_6h_std: 0.152825
- trend_confirmation_6h_cov: 0.640245
- momentum_persistence_3h_mean: 0.554165
- momentum_persistence_3h_std: 2.753608
- momentum_persistence_3h_cov: 4.968934
- vol_momentum_sync_mean: 0.266714
- vol_momentum_sync_std: 0.329542
- vol_momentum_sync_cov: 1.235563
- range_momentum_divergence_mean: 0.999879
- range_momentum_divergence_std: 0.000060
- range_momentum_divergence_cov: 0.000060
- volume_concentration_ratio_3h_mean: 0.524363
- volume_concentration_ratio_3h_std: 0.227709
- volume_concentration_ratio_3h_cov: 0.434259
- pressure_ratio_mean: 3.602385
- pressure_ratio_std: 6.165216
- pressure_ratio_cov: 1.711427
- kyle_lambda_proxy_mean: 14495480.177762
- kyle_lambda_proxy_std: 7836452.125668
- kyle_lambda_proxy_cov: 0.540613
- reversal_intensity_mean: 0.004297
- reversal_intensity_std: 0.008383
- reversal_intensity_cov: 1.950596
- whipsaw_count_mean: 5.974359
- whipsaw_count_std: 1.216492
- whipsaw_count_cov: 0.203619
- vol_clustering_mean: 0.364316
- vol_clustering_std: 0.137692
- vol_clustering_cov: 0.377946
- vol_regime_change_mean: 0.092488
- vol_regime_change_std: 0.201415
- vol_regime_change_cov: 2.177750
- efficiency_ratio_mean: 1250.494887
- efficiency_ratio_std: 719.419871
- efficiency_ratio_cov: 0.575308
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.001294
- forward_return_std: 0.011191
- forward_return_cov: 8.650185

### Regime 3

- n_samples: 129.000000
- ghost_ratio_mean: 0.182565
- ghost_ratio_std: 0.096206
- ghost_ratio_cov: 0.526971
- absorption_ratio_mean: 7.562531
- absorption_ratio_std: 5.060500
- absorption_ratio_cov: 0.669154
- rvol_24_mean: 0.551041
- rvol_24_std: 0.260065
- rvol_24_cov: 0.471952
- rvol_20_mean: 0.385393
- rvol_20_std: 0.181416
- rvol_20_cov: 0.470729
- volume_efficiency_ratio_mean: 286.147719
- volume_efficiency_ratio_std: 57.824477
- volume_efficiency_ratio_cov: 0.202079
- intraday_close_ratio_mean: 21007.291617
- intraday_close_ratio_std: 42704.517706
- intraday_close_ratio_cov: 2.032843
- amihud_spike_ratio_scaled_mean: 0.931898
- amihud_spike_ratio_scaled_std: 1.246574
- amihud_spike_ratio_scaled_cov: 1.337673
- rvol_168_scaled_mean: -0.812810
- rvol_168_scaled_std: 0.264812
- rvol_168_scaled_cov: 0.325798
- cumulative_delta_divergence_mean: 0.824699
- cumulative_delta_divergence_std: 0.613562
- cumulative_delta_divergence_cov: 0.743984
- volume_direction_conviction_mean: 0.551582
- volume_direction_conviction_std: 0.279867
- volume_direction_conviction_cov: 0.507390
- volume_direction_imbalance_mean: 0.074389
- volume_direction_imbalance_std: 0.615928
- volume_direction_imbalance_cov: 8.279881
- trend_confirmation_6h_mean: 0.257194
- trend_confirmation_6h_std: 0.144877
- trend_confirmation_6h_cov: 0.563298
- momentum_persistence_3h_mean: 0.036322
- momentum_persistence_3h_std: 15.765893
- momentum_persistence_3h_cov: 434.063642
- vol_momentum_sync_mean: 0.150015
- vol_momentum_sync_std: 0.322665
- vol_momentum_sync_cov: 2.150887
- range_momentum_divergence_mean: 0.999755
- range_momentum_divergence_std: 0.001504
- range_momentum_divergence_cov: 0.001504
- volume_concentration_ratio_3h_mean: 0.411382
- volume_concentration_ratio_3h_std: 0.140060
- volume_concentration_ratio_3h_cov: 0.340462
- pressure_ratio_mean: 20.425991
- pressure_ratio_std: 192.880402
- pressure_ratio_cov: 9.442891
- kyle_lambda_proxy_mean: 11716859.279857
- kyle_lambda_proxy_std: 6302699.572061
- kyle_lambda_proxy_cov: 0.537917
- reversal_intensity_mean: 0.002078
- reversal_intensity_std: 0.002629
- reversal_intensity_cov: 1.265268
- whipsaw_count_mean: 6.578125
- whipsaw_count_std: 1.366698
- whipsaw_count_cov: 0.207764
- vol_clustering_mean: 0.301034
- vol_clustering_std: 0.134738
- vol_clustering_cov: 0.447584
- vol_regime_change_mean: -0.107948
- vol_regime_change_std: 0.319212
- vol_regime_change_cov: 2.957081
- efficiency_ratio_mean: 724.620664
- efficiency_ratio_std: 588.274402
- efficiency_ratio_cov: 0.811838
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.001667
- forward_return_std: 0.012866
- forward_return_cov: 7.718226

### Regime 4

- n_samples: 179.000000
- ghost_ratio_mean: 0.161653
- ghost_ratio_std: 0.078632
- ghost_ratio_cov: 0.486424
- absorption_ratio_mean: 7.973078
- absorption_ratio_std: 4.932814
- absorption_ratio_cov: 0.618684
- rvol_24_mean: 0.651091
- rvol_24_std: 0.264790
- rvol_24_cov: 0.406687
- rvol_20_mean: 0.589678
- rvol_20_std: 0.221896
- rvol_20_cov: 0.376300
- volume_efficiency_ratio_mean: 417.769549
- volume_efficiency_ratio_std: 50.045140
- volume_efficiency_ratio_cov: 0.119791
- intraday_close_ratio_mean: 23437.194880
- intraday_close_ratio_std: 40401.330879
- intraday_close_ratio_cov: 1.723813
- amihud_spike_ratio_scaled_mean: 0.099282
- amihud_spike_ratio_scaled_std: 0.898760
- amihud_spike_ratio_scaled_cov: 9.052555
- rvol_168_scaled_mean: -0.519150
- rvol_168_scaled_std: 0.309172
- rvol_168_scaled_cov: 0.595535
- cumulative_delta_divergence_mean: 0.892023
- cumulative_delta_divergence_std: 0.668957
- cumulative_delta_divergence_cov: 0.749933
- volume_direction_conviction_mean: 0.509577
- volume_direction_conviction_std: 0.281765
- volume_direction_conviction_cov: 0.552939
- volume_direction_imbalance_mean: 0.100385
- volume_direction_imbalance_std: 0.574791
- volume_direction_imbalance_cov: 5.725848
- trend_confirmation_6h_mean: 0.259831
- trend_confirmation_6h_std: 0.158056
- trend_confirmation_6h_cov: 0.608305
- momentum_persistence_3h_mean: -1.243039
- momentum_persistence_3h_std: 9.135394
- momentum_persistence_3h_cov: 7.349239
- vol_momentum_sync_mean: 0.094929
- vol_momentum_sync_std: 0.266578
- vol_momentum_sync_cov: 2.808171
- range_momentum_divergence_mean: 0.999863
- range_momentum_divergence_std: 0.000472
- range_momentum_divergence_cov: 0.000472
- volume_concentration_ratio_3h_mean: 0.423309
- volume_concentration_ratio_3h_std: 0.153929
- volume_concentration_ratio_3h_cov: 0.363633
- pressure_ratio_mean: 12.500749
- pressure_ratio_std: 137.583633
- pressure_ratio_cov: 11.006032
- kyle_lambda_proxy_mean: 13523885.903178
- kyle_lambda_proxy_std: 7614524.739205
- kyle_lambda_proxy_cov: 0.563043
- reversal_intensity_mean: 0.001857
- reversal_intensity_std: 0.002705
- reversal_intensity_cov: 1.456627
- whipsaw_count_mean: 6.056180
- whipsaw_count_std: 1.543051
- whipsaw_count_cov: 0.254790
- vol_clustering_mean: 0.342228
- vol_clustering_std: 0.109386
- vol_clustering_cov: 0.319629
- vol_regime_change_mean: -0.032848
- vol_regime_change_std: 0.297953
- vol_regime_change_cov: 9.070748
- efficiency_ratio_mean: 983.060919
- efficiency_ratio_std: 745.300099
- efficiency_ratio_cov: 0.758142
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: 0.000686
- forward_return_std: 0.005461
- forward_return_cov: 7.960568

