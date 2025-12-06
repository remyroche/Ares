# Liquidity Cluster Quality Report

**Symbol:** ETHUSDT  \n**Assessment time:** 2025-12-05T22:31:02.100844

## Overall Quality

- Overall quality score: **0.4920**

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

- Effort/Result CoV separation score: 0.7055
- Returns CoV separation score: 0.7482

## Effort vs Result Separation

- Effort/Result separation score: 0.1877
- Ghost vs Valid contrast: -0.1116
- Absorption vs Valid contrast: 0.2684

## Trap / Ghost Behavior

- Ghost reversal rate: 0.2975
- Ghost false-trend rate: 0.2886

## Absorption Behavior

- Absorption reversal rate: 0.5005
- Absorption follow-through rate: 0.2346

## Trend Confirmation & Apathy

- Valid trend follow-through (mean fwd return): -0.000354
- Apathy noise fraction: 0.3013

## Class Balance

- Class balance score: 0.8509
- Number of regimes: 5
- Number of samples: 34135

## Per-Regime Metrics

### Regime 0

- n_samples: 5575.000000
- ghost_ratio_mean: 0.200151
- ghost_ratio_std: 0.098744
- ghost_ratio_cov: 0.493348
- absorption_ratio_mean: 6.214991
- absorption_ratio_std: 2.967007
- absorption_ratio_cov: 0.477395
- rvol_24_mean: 0.873831
- rvol_24_std: 0.384452
- rvol_24_cov: 0.439961
- rvol_20_mean: 0.845489
- rvol_20_std: 0.585039
- rvol_20_cov: 0.691953
- volume_efficiency_ratio_mean: 745.201050
- volume_efficiency_ratio_std: 166.754059
- volume_efficiency_ratio_cov: 0.223771
- intraday_close_ratio_mean: 11007.240047
- intraday_close_ratio_std: 18410.577600
- intraday_close_ratio_cov: 1.672588
- amihud_spike_ratio_scaled_mean: -0.196323
- amihud_spike_ratio_scaled_std: 0.875409
- amihud_spike_ratio_scaled_cov: 4.459026
- rvol_168_scaled_mean: -0.177845
- rvol_168_scaled_std: 0.882705
- rvol_168_scaled_cov: 4.963333
- cumulative_delta_divergence_mean: 1.010649
- cumulative_delta_divergence_std: 0.756350
- cumulative_delta_divergence_cov: 0.748380
- volume_direction_conviction_mean: 0.489596
- volume_direction_conviction_std: 0.290114
- volume_direction_conviction_cov: 0.592559
- volume_direction_imbalance_mean: -0.005325
- volume_direction_imbalance_std: 0.569109
- volume_direction_imbalance_cov: 106.884208
- trend_confirmation_6h_mean: 0.234691
- trend_confirmation_6h_std: 0.148542
- trend_confirmation_6h_cov: 0.632924
- momentum_persistence_3h_mean: 20.094019
- momentum_persistence_3h_std: 1494.055652
- momentum_persistence_3h_cov: 74.353252
- vol_momentum_sync_mean: 0.067660
- vol_momentum_sync_std: 0.216201
- vol_momentum_sync_cov: 3.195418
- range_momentum_divergence_mean: 0.999732
- range_momentum_divergence_std: 0.000911
- range_momentum_divergence_cov: 0.000911
- volume_concentration_ratio_3h_mean: 0.417126
- volume_concentration_ratio_3h_std: 0.144246
- volume_concentration_ratio_3h_cov: 0.345810
- pressure_ratio_mean: 350769414144.000000
- pressure_ratio_std: 6571665391616.000000
- pressure_ratio_cov: 18.735001
- kyle_lambda_proxy_mean: 440.240277
- kyle_lambda_proxy_std: 287.338596
- kyle_lambda_proxy_cov: 0.652686
- reversal_intensity_mean: 0.001029
- reversal_intensity_std: 0.003854
- reversal_intensity_cov: 3.743901
- whipsaw_count_mean: 6.264395
- whipsaw_count_std: 1.609438
- whipsaw_count_cov: 0.256918
- vol_clustering_mean: 0.362803
- vol_clustering_std: 0.117817
- vol_clustering_cov: 0.324742
- vol_regime_change_mean: -0.052333
- vol_regime_change_std: 0.258593
- vol_regime_change_cov: 4.941305
- efficiency_ratio_mean: 428.940760
- efficiency_ratio_std: 415.292526
- efficiency_ratio_cov: 0.968182
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.000002
- forward_return_std: 0.004536
- forward_return_cov: 2226.777186
- forward_return_positive_rate: 0.504933
- forward_return_negative_rate: 0.493632
- forward_return_sharpe_like: -0.000449
- forward_return_mar_like: -0.000004
- forward_return_tail_loss_p95: 0.005775
- adverse_selection_rate: 0.493632
- structural_low_fraction: 0.090942
- transient_gap_fraction: 0.000000

### Regime 1

- n_samples: 1620.000000
- ghost_ratio_mean: 0.342908
- ghost_ratio_std: 0.121928
- ghost_ratio_cov: 0.355571
- absorption_ratio_mean: 3.466872
- absorption_ratio_std: 1.850110
- absorption_ratio_cov: 0.533654
- rvol_24_mean: 1.493141
- rvol_24_std: 0.594869
- rvol_24_cov: 0.398401
- rvol_20_mean: 2.299347
- rvol_20_std: 1.911166
- rvol_20_cov: 0.831178
- volume_efficiency_ratio_mean: 787.979736
- volume_efficiency_ratio_std: 335.720856
- volume_efficiency_ratio_cov: 0.426053
- intraday_close_ratio_mean: 6377.269116
- intraday_close_ratio_std: 11944.711445
- intraday_close_ratio_cov: 1.873014
- amihud_spike_ratio_scaled_mean: 0.267550
- amihud_spike_ratio_scaled_std: 1.044972
- amihud_spike_ratio_scaled_cov: 3.905712
- rvol_168_scaled_mean: 1.234959
- rvol_168_scaled_std: 1.228926
- rvol_168_scaled_cov: 0.995115
- cumulative_delta_divergence_mean: 1.201116
- cumulative_delta_divergence_std: 0.854232
- cumulative_delta_divergence_cov: 0.711198
- volume_direction_conviction_mean: 0.559656
- volume_direction_conviction_std: 0.269945
- volume_direction_conviction_cov: 0.482341
- volume_direction_imbalance_mean: -0.036026
- volume_direction_imbalance_std: 0.620468
- volume_direction_imbalance_cov: 17.222626
- trend_confirmation_6h_mean: 0.275869
- trend_confirmation_6h_std: 0.143614
- trend_confirmation_6h_cov: 0.520588
- momentum_persistence_3h_mean: -0.975785
- momentum_persistence_3h_std: 46.598814
- momentum_persistence_3h_cov: 47.755200
- vol_momentum_sync_mean: 0.370866
- vol_momentum_sync_std: 0.359863
- vol_momentum_sync_cov: 0.970331
- range_momentum_divergence_mean: 0.999663
- range_momentum_divergence_std: 0.000231
- range_momentum_divergence_cov: 0.000231
- volume_concentration_ratio_3h_mean: 0.452560
- volume_concentration_ratio_3h_std: 0.209929
- volume_concentration_ratio_3h_cov: 0.463870
- pressure_ratio_mean: 4906983358464.000000
- pressure_ratio_std: 115308094291968.000000
- pressure_ratio_cov: 23.498774
- kyle_lambda_proxy_mean: 1001.319375
- kyle_lambda_proxy_std: 837.162914
- kyle_lambda_proxy_cov: 0.836060
- reversal_intensity_mean: 0.004645
- reversal_intensity_std: 0.008961
- reversal_intensity_cov: 1.929115
- whipsaw_count_mean: 5.909877
- whipsaw_count_std: 1.593075
- whipsaw_count_cov: 0.269562
- vol_clustering_mean: 0.424100
- vol_clustering_std: 0.107673
- vol_clustering_cov: 0.253886
- vol_regime_change_mean: 0.067119
- vol_regime_change_std: 0.220094
- vol_regime_change_cov: 3.279158
- efficiency_ratio_mean: 535.493638
- efficiency_ratio_std: 529.800276
- efficiency_ratio_cov: 0.989368
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.000354
- forward_return_std: 0.009584
- forward_return_cov: 27.045643
- forward_return_positive_rate: 0.490123
- forward_return_negative_rate: 0.509259
- forward_return_sharpe_like: -0.036974
- forward_return_mar_like: -0.000494
- forward_return_tail_loss_p95: 0.014079
- adverse_selection_rate: 0.509259
- structural_low_fraction: 0.008025
- transient_gap_fraction: 0.000000

### Regime 2

- n_samples: 4246.000000
- ghost_ratio_mean: 0.210513
- ghost_ratio_std: 0.104881
- ghost_ratio_cov: 0.498218
- absorption_ratio_mean: 6.009997
- absorption_ratio_std: 3.069915
- absorption_ratio_cov: 0.510801
- rvol_24_mean: 1.091348
- rvol_24_std: 0.484341
- rvol_24_cov: 0.443801
- rvol_20_mean: 1.281480
- rvol_20_std: 0.897525
- rvol_20_cov: 0.700381
- volume_efficiency_ratio_mean: 1530.675049
- volume_efficiency_ratio_std: 683.212585
- volume_efficiency_ratio_cov: 0.446347
- intraday_close_ratio_mean: 8086.854208
- intraday_close_ratio_std: 14907.928690
- intraday_close_ratio_cov: 1.843477
- amihud_spike_ratio_scaled_mean: -0.407741
- amihud_spike_ratio_scaled_std: 0.729305
- amihud_spike_ratio_scaled_cov: 1.788650
- rvol_168_scaled_mean: 0.455469
- rvol_168_scaled_std: 1.038692
- rvol_168_scaled_cov: 2.280490
- cumulative_delta_divergence_mean: 1.071913
- cumulative_delta_divergence_std: 0.801445
- cumulative_delta_divergence_cov: 0.747677
- volume_direction_conviction_mean: 0.477562
- volume_direction_conviction_std: 0.279603
- volume_direction_conviction_cov: 0.585481
- volume_direction_imbalance_mean: -0.011723
- volume_direction_imbalance_std: 0.553317
- volume_direction_imbalance_cov: 47.200493
- trend_confirmation_6h_mean: 0.234415
- trend_confirmation_6h_std: 0.147476
- trend_confirmation_6h_cov: 0.629123
- momentum_persistence_3h_mean: -1.355754
- momentum_persistence_3h_std: 66.074772
- momentum_persistence_3h_cov: 48.736563
- vol_momentum_sync_mean: 0.110120
- vol_momentum_sync_std: 0.257341
- vol_momentum_sync_cov: 2.336915
- range_momentum_divergence_mean: 0.999658
- range_momentum_divergence_std: 0.001331
- range_momentum_divergence_cov: 0.001331
- volume_concentration_ratio_3h_mean: 0.427378
- volume_concentration_ratio_3h_std: 0.157198
- volume_concentration_ratio_3h_cov: 0.367821
- pressure_ratio_mean: 428558647296.000000
- pressure_ratio_std: 7983396290560.000000
- pressure_ratio_cov: 18.628480
- kyle_lambda_proxy_mean: 1075.779161
- kyle_lambda_proxy_std: 667.884785
- kyle_lambda_proxy_cov: 0.620838
- reversal_intensity_mean: 0.001473
- reversal_intensity_std: 0.004999
- reversal_intensity_cov: 3.393433
- whipsaw_count_mean: 6.032266
- whipsaw_count_std: 1.688915
- whipsaw_count_cov: 0.279980
- vol_clustering_mean: 0.384381
- vol_clustering_std: 0.110024
- vol_clustering_cov: 0.286236
- vol_regime_change_mean: -0.009775
- vol_regime_change_std: 0.239021
- vol_regime_change_cov: 24.452256
- efficiency_ratio_mean: 365.722751
- efficiency_ratio_std: 285.905512
- efficiency_ratio_cov: 0.781755
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: 0.000081
- forward_return_std: 0.006174
- forward_return_cov: 76.481986
- forward_return_positive_rate: 0.499529
- forward_return_negative_rate: 0.500000
- forward_return_sharpe_like: 0.013075
- forward_return_mar_like: 0.000174
- forward_return_tail_loss_p95: 0.008560
- adverse_selection_rate: 0.500000
- structural_low_fraction: 0.040509
- transient_gap_fraction: 0.000000

### Regime 3

- n_samples: 16195.000000
- ghost_ratio_mean: 0.274036
- ghost_ratio_std: 0.123378
- ghost_ratio_cov: 0.450226
- absorption_ratio_mean: 4.566797
- absorption_ratio_std: 2.437886
- absorption_ratio_cov: 0.533828
- rvol_24_mean: 0.842427
- rvol_24_std: 0.354592
- rvol_24_cov: 0.420917
- rvol_20_mean: 0.735221
- rvol_20_std: 0.413613
- rvol_20_cov: 0.562570
- volume_efficiency_ratio_mean: 197.185471
- volume_efficiency_ratio_std: 68.287735
- volume_efficiency_ratio_cov: 0.346312
- intraday_close_ratio_mean: 14415.524426
- intraday_close_ratio_std: 19453.852253
- intraday_close_ratio_cov: 1.349507
- amihud_spike_ratio_scaled_mean: 0.205215
- amihud_spike_ratio_scaled_std: 1.089413
- amihud_spike_ratio_scaled_cov: 5.308656
- rvol_168_scaled_mean: -0.339734
- rvol_168_scaled_std: 0.731962
- rvol_168_scaled_cov: 2.154517
- cumulative_delta_divergence_mean: 1.021158
- cumulative_delta_divergence_std: 0.770982
- cumulative_delta_divergence_cov: 0.755007
- volume_direction_conviction_mean: 0.550239
- volume_direction_conviction_std: 0.291131
- volume_direction_conviction_cov: 0.529100
- volume_direction_imbalance_mean: 0.013331
- volume_direction_imbalance_std: 0.622384
- volume_direction_imbalance_cov: 46.686576
- trend_confirmation_6h_mean: 0.258103
- trend_confirmation_6h_std: 0.148316
- trend_confirmation_6h_cov: 0.574637
- momentum_persistence_3h_mean: 3.205533
- momentum_persistence_3h_std: 397.115335
- momentum_persistence_3h_cov: 123.884331
- vol_momentum_sync_mean: 0.133422
- vol_momentum_sync_std: 0.295272
- vol_momentum_sync_cov: 2.213071
- range_momentum_divergence_mean: 0.999842
- range_momentum_divergence_std: 0.000595
- range_momentum_divergence_cov: 0.000595
- volume_concentration_ratio_3h_mean: 0.423516
- volume_concentration_ratio_3h_std: 0.138897
- volume_concentration_ratio_3h_cov: 0.327961
- pressure_ratio_mean: 367464742912.000000
- pressure_ratio_std: 7654696550400.000000
- pressure_ratio_cov: 20.831105
- kyle_lambda_proxy_mean: 124.963897
- kyle_lambda_proxy_std: 77.892684
- kyle_lambda_proxy_cov: 0.623321
- reversal_intensity_mean: 0.001429
- reversal_intensity_std: 0.004797
- reversal_intensity_cov: 3.356023
- whipsaw_count_mean: 6.403347
- whipsaw_count_std: 1.626869
- whipsaw_count_cov: 0.254065
- vol_clustering_mean: 0.380370
- vol_clustering_std: 0.115901
- vol_clustering_cov: 0.304706
- vol_regime_change_mean: -0.044320
- vol_regime_change_std: 0.234651
- vol_regime_change_cov: 5.294404
- efficiency_ratio_mean: 650.586337
- efficiency_ratio_std: 563.832687
- efficiency_ratio_cov: 0.866653
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.000008
- forward_return_std: 0.009163
- forward_return_cov: 1124.778829
- forward_return_positive_rate: 0.497777
- forward_return_negative_rate: 0.501853
- forward_return_sharpe_like: -0.000889
- forward_return_mar_like: -0.000010
- forward_return_tail_loss_p95: 0.005379
- adverse_selection_rate: 0.501853
- structural_low_fraction: 0.080086
- transient_gap_fraction: 0.000123

### Regime 4

- n_samples: 6499.000000
- ghost_ratio_mean: 0.273785
- ghost_ratio_std: 0.125141
- ghost_ratio_cov: 0.457078
- absorption_ratio_mean: 4.597995
- absorption_ratio_std: 2.466691
- absorption_ratio_cov: 0.536471
- rvol_24_mean: 1.095012
- rvol_24_std: 0.507326
- rvol_24_cov: 0.463307
- rvol_20_mean: 1.232304
- rvol_20_std: 0.866924
- rvol_20_cov: 0.703498
- volume_efficiency_ratio_mean: 400.111023
- volume_efficiency_ratio_std: 76.124458
- volume_efficiency_ratio_cov: 0.190258
- intraday_close_ratio_mean: 11741.950178
- intraday_close_ratio_std: 17863.550859
- intraday_close_ratio_cov: 1.521344
- amihud_spike_ratio_scaled_mean: -0.101891
- amihud_spike_ratio_scaled_std: 0.937213
- amihud_spike_ratio_scaled_cov: 9.198227
- rvol_168_scaled_mean: 0.379058
- rvol_168_scaled_std: 1.107092
- rvol_168_scaled_cov: 2.920643
- cumulative_delta_divergence_mean: 1.105016
- cumulative_delta_divergence_std: 0.818213
- cumulative_delta_divergence_cov: 0.740454
- volume_direction_conviction_mean: 0.522515
- volume_direction_conviction_std: 0.285604
- volume_direction_conviction_cov: 0.546595
- volume_direction_imbalance_mean: -0.001462
- volume_direction_imbalance_std: 0.595509
- volume_direction_imbalance_cov: 407.265905
- trend_confirmation_6h_mean: 0.255476
- trend_confirmation_6h_std: 0.149170
- trend_confirmation_6h_cov: 0.583893
- momentum_persistence_3h_mean: -0.149037
- momentum_persistence_3h_std: 55.303397
- momentum_persistence_3h_cov: 371.071161
- vol_momentum_sync_mean: 0.169499
- vol_momentum_sync_std: 0.311721
- vol_momentum_sync_cov: 1.839066
- range_momentum_divergence_mean: 0.999791
- range_momentum_divergence_std: 0.000773
- range_momentum_divergence_cov: 0.000773
- volume_concentration_ratio_3h_mean: 0.424760
- volume_concentration_ratio_3h_std: 0.156189
- volume_concentration_ratio_3h_cov: 0.367711
- pressure_ratio_mean: 523249221632.000000
- pressure_ratio_std: 10419833929728.000000
- pressure_ratio_cov: 19.913711
- kyle_lambda_proxy_mean: 266.398938
- kyle_lambda_proxy_std: 197.495053
- kyle_lambda_proxy_cov: 0.741351
- reversal_intensity_mean: 0.001649
- reversal_intensity_std: 0.005610
- reversal_intensity_cov: 3.402892
- whipsaw_count_mean: 6.057547
- whipsaw_count_std: 1.641100
- whipsaw_count_cov: 0.270918
- vol_clustering_mean: 0.396330
- vol_clustering_std: 0.123670
- vol_clustering_cov: 0.312037
- vol_regime_change_mean: -0.006749
- vol_regime_change_std: 0.240029
- vol_regime_change_cov: 35.565654
- efficiency_ratio_mean: 673.087099
- efficiency_ratio_std: 586.945428
- efficiency_ratio_cov: 0.872020
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: 0.000002
- forward_return_std: 0.005591
- forward_return_cov: 3662.185443
- forward_return_positive_rate: 0.498230
- forward_return_negative_rate: 0.500846
- forward_return_sharpe_like: 0.000273
- forward_return_mar_like: 0.000004
- forward_return_tail_loss_p95: 0.006931
- adverse_selection_rate: 0.500846
- structural_low_fraction: 0.038467
- transient_gap_fraction: 0.000000

