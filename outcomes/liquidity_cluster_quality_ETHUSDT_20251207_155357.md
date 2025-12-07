# Liquidity Cluster Quality Report

**Symbol:** ETHUSDT  \n**Assessment time:** 2025-12-07T15:53:57.137060

## Overall Quality

- Overall quality score: **0.4798**

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

- Effort/Result CoV separation score: 0.7984
- Returns CoV separation score: 0.3993

## Effort vs Result Separation

- Effort/Result separation score: 0.1258
- Ghost vs Valid contrast: -0.0586
- Absorption vs Valid contrast: 0.1943

## Trap / Ghost Behavior

- Ghost reversal rate: 0.3550
- Ghost false-trend rate: 0.2158

## Absorption Behavior

- Absorption reversal rate: 0.4956
- Absorption follow-through rate: 0.1881

## Trend Confirmation & Apathy

- Valid trend follow-through (mean fwd return): -0.000008
- Apathy noise fraction: 0.4847

## Class Balance

- Class balance score: 0.9317
- Number of regimes: 5
- Number of samples: 140354

## Per-Regime Metrics

### Regime 0

- n_samples: 26016.000000
- ghost_ratio_mean: 0.191213
- ghost_ratio_std: 0.103623
- ghost_ratio_cov: 0.541922
- absorption_ratio_mean: 6.782618
- absorption_ratio_std: 3.508380
- absorption_ratio_cov: 0.517260
- rvol_24_mean: 0.872710
- rvol_24_std: 0.376263
- rvol_24_cov: 0.431143
- rvol_20_mean: 0.790492
- rvol_20_std: 0.461018
- rvol_20_cov: 0.583204
- volume_efficiency_ratio_mean: 762.458313
- volume_efficiency_ratio_std: 164.363937
- volume_efficiency_ratio_cov: 0.215571
- intraday_close_ratio_mean: 9780.687374
- intraday_close_ratio_std: 15924.471262
- intraday_close_ratio_cov: 1.628155
- amihud_spike_ratio_scaled_mean: -0.316515
- amihud_spike_ratio_scaled_std: 0.745882
- amihud_spike_ratio_scaled_cov: 2.356547
- rvol_168_scaled_mean: -0.185674
- rvol_168_scaled_std: 0.755402
- rvol_168_scaled_cov: 4.068437
- cumulative_delta_divergence_mean: 1.095294
- cumulative_delta_divergence_std: 0.818776
- cumulative_delta_divergence_cov: 0.747540
- volume_direction_conviction_mean: 0.534986
- volume_direction_conviction_std: 0.307506
- volume_direction_conviction_cov: 0.574793
- volume_direction_imbalance_mean: 0.014765
- volume_direction_imbalance_std: 0.616898
- volume_direction_imbalance_cov: 41.780827
- trend_confirmation_6h_mean: 0.252893
- trend_confirmation_6h_std: 0.156690
- trend_confirmation_6h_cov: 0.619588
- momentum_persistence_3h_mean: 26.036701
- momentum_persistence_3h_std: 5257.931042
- momentum_persistence_3h_cov: 201.943062
- vol_momentum_sync_mean: 0.063875
- vol_momentum_sync_std: 0.215396
- vol_momentum_sync_cov: 3.372152
- range_momentum_divergence_mean: 0.999745
- range_momentum_divergence_std: 0.000169
- range_momentum_divergence_cov: 0.000169
- volume_concentration_ratio_3h_mean: 0.443691
- volume_concentration_ratio_3h_std: 0.155116
- volume_concentration_ratio_3h_cov: 0.349603
- pressure_ratio_mean: 187872542720.000000
- pressure_ratio_std: 2563150708736.000000
- pressure_ratio_cov: 13.643030
- kyle_lambda_proxy_mean: 242.083770
- kyle_lambda_proxy_std: 168.659250
- kyle_lambda_proxy_cov: 0.696698
- reversal_intensity_mean: 0.000519
- reversal_intensity_std: 0.000773
- reversal_intensity_cov: 1.490152
- whipsaw_count_mean: 6.315998
- whipsaw_count_std: 1.636659
- whipsaw_count_cov: 0.259129
- vol_clustering_mean: 0.367814
- vol_clustering_std: 0.113844
- vol_clustering_cov: 0.309516
- vol_regime_change_mean: -0.053422
- vol_regime_change_std: 0.240753
- vol_regime_change_cov: 4.506626
- efficiency_ratio_mean: 348.915514
- efficiency_ratio_std: 307.463569
- efficiency_ratio_cov: 0.881198
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: 0.000005
- forward_return_std: 0.002370
- forward_return_cov: 511.600481
- forward_return_positive_rate: 0.499962
- forward_return_negative_rate: 0.498616
- forward_return_sharpe_like: 0.001954
- forward_return_mar_like: 0.000014
- forward_return_tail_loss_p95: 0.003329
- adverse_selection_rate: 0.498616
- structural_low_fraction: 0.078298
- transient_gap_fraction: 0.000000

### Regime 1

- n_samples: 13414.000000
- ghost_ratio_mean: 0.311852
- ghost_ratio_std: 0.134066
- ghost_ratio_cov: 0.429903
- absorption_ratio_mean: 4.049843
- absorption_ratio_std: 2.376170
- absorption_ratio_cov: 0.586731
- rvol_24_mean: 1.310500
- rvol_24_std: 0.582143
- rvol_24_cov: 0.444214
- rvol_20_mean: 1.720744
- rvol_20_std: 1.198151
- rvol_20_cov: 0.696298
- volume_efficiency_ratio_mean: 529.667053
- volume_efficiency_ratio_std: 132.519485
- volume_efficiency_ratio_cov: 0.250194
- intraday_close_ratio_mean: 9154.767260
- intraday_close_ratio_std: 14182.291438
- intraday_close_ratio_cov: 1.549170
- amihud_spike_ratio_scaled_mean: -0.057026
- amihud_spike_ratio_scaled_std: 0.897377
- amihud_spike_ratio_scaled_cov: 15.736166
- rvol_168_scaled_mean: 0.961618
- rvol_168_scaled_std: 1.164416
- rvol_168_scaled_cov: 1.210892
- cumulative_delta_divergence_mean: 1.127432
- cumulative_delta_divergence_std: 0.843972
- cumulative_delta_divergence_cov: 0.748579
- volume_direction_conviction_mean: 0.530784
- volume_direction_conviction_std: 0.283477
- volume_direction_conviction_cov: 0.534071
- volume_direction_imbalance_mean: 0.013053
- volume_direction_imbalance_std: 0.601616
- volume_direction_imbalance_cov: 46.089270
- trend_confirmation_6h_mean: 0.258587
- trend_confirmation_6h_std: 0.148798
- trend_confirmation_6h_cov: 0.575428
- momentum_persistence_3h_mean: -20.454398
- momentum_persistence_3h_std: 3496.358021
- momentum_persistence_3h_cov: 170.934293
- vol_momentum_sync_mean: 0.270073
- vol_momentum_sync_std: 0.350780
- vol_momentum_sync_cov: 1.298837
- range_momentum_divergence_mean: 0.999767
- range_momentum_divergence_std: 0.000144
- range_momentum_divergence_cov: 0.000145
- volume_concentration_ratio_3h_mean: 0.453593
- volume_concentration_ratio_3h_std: 0.178495
- volume_concentration_ratio_3h_cov: 0.393512
- pressure_ratio_mean: 893644701696.000000
- pressure_ratio_std: 27548092203008.000000
- pressure_ratio_cov: 30.826672
- kyle_lambda_proxy_mean: 394.332270
- kyle_lambda_proxy_std: 285.742837
- kyle_lambda_proxy_cov: 0.724625
- reversal_intensity_mean: 0.002180
- reversal_intensity_std: 0.003417
- reversal_intensity_cov: 1.567767
- whipsaw_count_mean: 6.060236
- whipsaw_count_std: 1.690980
- whipsaw_count_cov: 0.279029
- vol_clustering_mean: 0.429747
- vol_clustering_std: 0.112188
- vol_clustering_cov: 0.261056
- vol_regime_change_mean: 0.051614
- vol_regime_change_std: 0.219972
- vol_regime_change_cov: 4.261877
- efficiency_ratio_mean: 635.549270
- efficiency_ratio_std: 545.223072
- efficiency_ratio_cov: 0.857877
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.000008
- forward_return_std: 0.005330
- forward_return_cov: 673.322667
- forward_return_positive_rate: 0.506635
- forward_return_negative_rate: 0.492694
- forward_return_sharpe_like: -0.001485
- forward_return_mar_like: -0.000010
- forward_return_tail_loss_p95: 0.007962
- adverse_selection_rate: 0.492694
- structural_low_fraction: 0.009393
- transient_gap_fraction: 0.000000

### Regime 2

- n_samples: 17869.000000
- ghost_ratio_mean: 0.233257
- ghost_ratio_std: 0.132375
- ghost_ratio_cov: 0.567508
- absorption_ratio_mean: 6.003687
- absorption_ratio_std: 3.704154
- absorption_ratio_cov: 0.616980
- rvol_24_mean: 1.277142
- rvol_24_std: 0.601973
- rvol_24_cov: 0.471343
- rvol_20_mean: 1.817028
- rvol_20_std: 1.683020
- rvol_20_cov: 0.926249
- volume_efficiency_ratio_mean: 1326.771118
- volume_efficiency_ratio_std: 991.501221
- volume_efficiency_ratio_cov: 0.747304
- intraday_close_ratio_mean: 8340.296794
- intraday_close_ratio_std: 14316.297678
- intraday_close_ratio_cov: 1.716521
- amihud_spike_ratio_scaled_mean: -0.478764
- amihud_spike_ratio_scaled_std: 0.674063
- amihud_spike_ratio_scaled_cov: 1.407925
- rvol_168_scaled_mean: 0.884580
- rvol_168_scaled_std: 1.220582
- rvol_168_scaled_cov: 1.379844
- cumulative_delta_divergence_mean: 1.111115
- cumulative_delta_divergence_std: 0.828595
- cumulative_delta_divergence_cov: 0.745733
- volume_direction_conviction_mean: 0.496439
- volume_direction_conviction_std: 0.289307
- volume_direction_conviction_cov: 0.582765
- volume_direction_imbalance_mean: -0.001812
- volume_direction_imbalance_std: 0.574596
- volume_direction_imbalance_cov: 317.182201
- trend_confirmation_6h_mean: 0.240803
- trend_confirmation_6h_std: 0.150321
- trend_confirmation_6h_cov: 0.624252
- momentum_persistence_3h_mean: -0.321724
- momentum_persistence_3h_std: 53.931595
- momentum_persistence_3h_cov: 167.633275
- vol_momentum_sync_mean: 0.152394
- vol_momentum_sync_std: 0.288685
- vol_momentum_sync_cov: 1.894337
- range_momentum_divergence_mean: 0.999709
- range_momentum_divergence_std: 0.000184
- range_momentum_divergence_cov: 0.000184
- volume_concentration_ratio_3h_mean: 0.470738
- volume_concentration_ratio_3h_std: 0.203405
- volume_concentration_ratio_3h_cov: 0.432099
- pressure_ratio_mean: 1118290706432.000000
- pressure_ratio_std: 59262277517312.000000
- pressure_ratio_cov: 52.993624
- kyle_lambda_proxy_mean: 684.391357
- kyle_lambda_proxy_std: 589.428351
- kyle_lambda_proxy_cov: 0.861245
- reversal_intensity_mean: 0.001519
- reversal_intensity_std: 0.003614
- reversal_intensity_cov: 2.378698
- whipsaw_count_mean: 6.076054
- whipsaw_count_std: 1.682646
- whipsaw_count_cov: 0.276931
- vol_clustering_mean: 0.403201
- vol_clustering_std: 0.116571
- vol_clustering_cov: 0.289113
- vol_regime_change_mean: 0.024278
- vol_regime_change_std: 0.242583
- vol_regime_change_cov: 9.991993
- efficiency_ratio_mean: 437.436270
- efficiency_ratio_std: 370.610756
- efficiency_ratio_cov: 0.847234
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: 0.000035
- forward_return_std: 0.005764
- forward_return_cov: 165.639147
- forward_return_positive_rate: 0.504449
- forward_return_negative_rate: 0.494320
- forward_return_sharpe_like: 0.006037
- forward_return_mar_like: 0.000054
- forward_return_tail_loss_p95: 0.007725
- adverse_selection_rate: 0.494320
- structural_low_fraction: 0.029548
- transient_gap_fraction: 0.000000

### Regime 3

- n_samples: 52372.000000
- ghost_ratio_mean: 0.277328
- ghost_ratio_std: 0.125267
- ghost_ratio_cov: 0.451694
- absorption_ratio_mean: 4.526881
- absorption_ratio_std: 2.453168
- absorption_ratio_cov: 0.541911
- rvol_24_mean: 0.814834
- rvol_24_std: 0.349862
- rvol_24_cov: 0.429365
- rvol_20_mean: 0.690849
- rvol_20_std: 0.404522
- rvol_20_cov: 0.585543
- volume_efficiency_ratio_mean: 211.290100
- volume_efficiency_ratio_std: 64.313599
- volume_efficiency_ratio_cov: 0.304385
- intraday_close_ratio_mean: 12621.475051
- intraday_close_ratio_std: 17033.640752
- intraday_close_ratio_cov: 1.349576
- amihud_spike_ratio_scaled_mean: 0.372558
- amihud_spike_ratio_scaled_std: 1.141782
- amihud_spike_ratio_scaled_cov: 3.064709
- rvol_168_scaled_mean: -0.399170
- rvol_168_scaled_std: 0.701902
- rvol_168_scaled_cov: 1.758404
- cumulative_delta_divergence_mean: 1.039326
- cumulative_delta_divergence_std: 0.778956
- cumulative_delta_divergence_cov: 0.749482
- volume_direction_conviction_mean: 0.564333
- volume_direction_conviction_std: 0.296794
- volume_direction_conviction_cov: 0.525920
- volume_direction_imbalance_mean: 0.027709
- volume_direction_imbalance_std: 0.637022
- volume_direction_imbalance_cov: 22.989602
- trend_confirmation_6h_mean: 0.265239
- trend_confirmation_6h_std: 0.152237
- trend_confirmation_6h_cov: 0.573962
- momentum_persistence_3h_mean: 0.441724
- momentum_persistence_3h_std: 1031.992536
- momentum_persistence_3h_cov: 2336.281645
- vol_momentum_sync_mean: 0.147673
- vol_momentum_sync_std: 0.310491
- vol_momentum_sync_cov: 2.102554
- range_momentum_divergence_mean: 0.999838
- range_momentum_divergence_std: 0.000103
- range_momentum_divergence_cov: 0.000103
- volume_concentration_ratio_3h_mean: 0.427602
- volume_concentration_ratio_3h_std: 0.140634
- volume_concentration_ratio_3h_cov: 0.328889
- pressure_ratio_mean: 375064723456.000000
- pressure_ratio_std: 6300259319808.000000
- pressure_ratio_cov: 16.797792
- kyle_lambda_proxy_mean: 121.861170
- kyle_lambda_proxy_std: 78.667750
- kyle_lambda_proxy_cov: 0.645552
- reversal_intensity_mean: 0.001195
- reversal_intensity_std: 0.001738
- reversal_intensity_cov: 1.454153
- whipsaw_count_mean: 6.398041
- whipsaw_count_std: 1.640901
- whipsaw_count_cov: 0.256469
- vol_clustering_mean: 0.384658
- vol_clustering_std: 0.109684
- vol_clustering_cov: 0.285147
- vol_regime_change_mean: -0.046827
- vol_regime_change_std: 0.220299
- vol_regime_change_cov: 4.704576
- efficiency_ratio_mean: 555.178553
- efficiency_ratio_std: 453.305910
- efficiency_ratio_cov: 0.816505
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.000009
- forward_return_std: 0.003042
- forward_return_cov: 349.285858
- forward_return_positive_rate: 0.502606
- forward_return_negative_rate: 0.496496
- forward_return_sharpe_like: -0.002863
- forward_return_mar_like: -0.000013
- forward_return_tail_loss_p95: 0.004583
- adverse_selection_rate: 0.496496
- structural_low_fraction: 0.076950
- transient_gap_fraction: 0.000000

### Regime 4

- n_samples: 30683.000000
- ghost_ratio_mean: 0.246890
- ghost_ratio_std: 0.120232
- ghost_ratio_cov: 0.486987
- absorption_ratio_mean: 5.190082
- absorption_ratio_std: 2.840859
- absorption_ratio_cov: 0.547363
- rvol_24_mean: 0.910437
- rvol_24_std: 0.399137
- rvol_24_cov: 0.438401
- rvol_20_mean: 0.862410
- rvol_20_std: 0.518457
- rvol_20_cov: 0.601172
- volume_efficiency_ratio_mean: 413.504547
- volume_efficiency_ratio_std: 67.285378
- volume_efficiency_ratio_cov: 0.162720
- intraday_close_ratio_mean: 11035.398696
- intraday_close_ratio_std: 16261.833893
- intraday_close_ratio_cov: 1.473606
- amihud_spike_ratio_scaled_mean: -0.039922
- amihud_spike_ratio_scaled_std: 0.945310
- amihud_spike_ratio_scaled_cov: 23.679085
- rvol_168_scaled_mean: -0.091777
- rvol_168_scaled_std: 0.814302
- rvol_168_scaled_cov: 8.872645
- cumulative_delta_divergence_mean: 1.084714
- cumulative_delta_divergence_std: 0.813095
- cumulative_delta_divergence_cov: 0.749593
- volume_direction_conviction_mean: 0.539576
- volume_direction_conviction_std: 0.298937
- volume_direction_conviction_cov: 0.554022
- volume_direction_imbalance_mean: 0.022056
- volume_direction_imbalance_std: 0.616464
- volume_direction_imbalance_cov: 27.950138
- trend_confirmation_6h_mean: 0.257474
- trend_confirmation_6h_std: 0.153973
- trend_confirmation_6h_cov: 0.598014
- momentum_persistence_3h_mean: -14.251479
- momentum_persistence_3h_std: 2335.194647
- momentum_persistence_3h_cov: 163.856302
- vol_momentum_sync_mean: 0.122814
- vol_momentum_sync_std: 0.284211
- vol_momentum_sync_cov: 2.314155
- range_momentum_divergence_mean: 0.999791
- range_momentum_divergence_std: 0.000141
- range_momentum_divergence_cov: 0.000141
- volume_concentration_ratio_3h_mean: 0.431891
- volume_concentration_ratio_3h_std: 0.148703
- volume_concentration_ratio_3h_cov: 0.344308
- pressure_ratio_mean: 375880876032.000000
- pressure_ratio_std: 6744722374656.000000
- pressure_ratio_cov: 17.943776
- kyle_lambda_proxy_mean: 193.508282
- kyle_lambda_proxy_std: 124.042057
- kyle_lambda_proxy_cov: 0.641017
- reversal_intensity_mean: 0.000924
- reversal_intensity_std: 0.001409
- reversal_intensity_cov: 1.525740
- whipsaw_count_mean: 6.260828
- whipsaw_count_std: 1.644220
- whipsaw_count_cov: 0.262620
- vol_clustering_mean: 0.388927
- vol_clustering_std: 0.113353
- vol_clustering_cov: 0.291451
- vol_regime_change_mean: -0.033512
- vol_regime_change_std: 0.227854
- vol_regime_change_cov: 6.799213
- efficiency_ratio_mean: 499.764637
- efficiency_ratio_std: 446.677796
- efficiency_ratio_cov: 0.893776
- return_autocorr_lag6_mean: 0.000000
- return_autocorr_lag6_std: 0.000000
- return_autocorr_lag6_cov: 0.000000
- forward_return_mean: -0.000009
- forward_return_std: 0.003123
- forward_return_cov: 336.958235
- forward_return_positive_rate: 0.504221
- forward_return_negative_rate: 0.494443
- forward_return_sharpe_like: -0.002967
- forward_return_mar_like: -0.000012
- forward_return_tail_loss_p95: 0.004554
- adverse_selection_rate: 0.494443
- structural_low_fraction: 0.060131
- transient_gap_fraction: 0.000000

