# Breakout/Bounce Regime Diagnostics

- Symbol: **ETHUSDT**
- Exchange: **binance**
- Timeframe: **1h**
- Direction: **long**
- Horizon (bars): **96**
- Samples (training window): **1332**

## Global Model Metrics
- Validation log loss: **nan**
- Precision (breakout class 1): **0.3072**

### Class Counts (training labels)
| Regime | Count |
|--------|-------|
| 0 | 137 |
| 1 | 617 |
| 2 | 578 |

## Forward Return Sharpe-like Ratios
| Scope | Regime | Mean Return | Std Return | Sharpe-like |
|-------|--------|-------------|------------|-------------|
| global | -1 | 0.037819 | 0.196923 | 0.1920 |
| regime | 0 | -0.016105 | 0.137015 | -0.1175 |
| regime | 1 | 0.035847 | 0.209218 | 0.1713 |
| regime | 2 | 0.040688 | 0.181319 | 0.2244 |

## Winsorised CV Between/Within Regimes
Definition: per metric, we winsorise values (5–95%), compute global and per-regime CV, and report the ratio of between-regime CV to within-regime CV.

| Metric | Scope | Regime | Count | Mean | Std | CV | CV_between | CV_within | CV_between/within |
|--------|-------|--------|-------|------|-----|----|-----------|----------|--------------------|
| breakout_level_strength | global | -1 | 1332 | 0.654134 | 0.073373 | 0.112168 | 0.026509 | 0.092022 | 0.288073 |
| breakout_bearish_prob | global | -1 | 1332 | 0.470769 | 0.420560 | 0.893347 | 0.262438 | 0.626138 | 0.419138 |
| breakout_bullish_prob | global | -1 | 1332 | 0.529231 | 0.420560 | 0.794662 | 0.233447 | 0.556970 | 0.419138 |
| breakout_short_edge_score | global | -1 | 1332 | -0.029059 | 0.550391 | 18.940141 | 5.411930 | 13.411134 | 0.403540 |
| breakout_long_edge_score | global | -1 | 1332 | 0.029059 | 0.550391 | 18.940141 | 5.411930 | 13.411134 | 0.403540 |
| breakout_regime_2_prob | global | -1 | 1332 | 0.463313 | 0.421560 | 0.909881 | 0.783347 | 0.183152 | 4.277034 |
| breakout_regime_1_prob | global | -1 | 1332 | 0.521854 | 0.421017 | 0.806772 | 0.693430 | 0.145797 | 4.756131 |
| breakout_regime_0_prob | global | -1 | 1332 | 0.009358 | 0.013003 | 1.389505 | 27.775948 | 4.820994 | 5.761456 |
| opposing_age_log_hours | global | -1 | 1332 | 4.019901 | 1.262464 | 0.314053 | 0.066305 | 0.395395 | 0.167694 |
| age_log_hours | global | -1 | 1332 | 3.335427 | 0.948205 | 0.284283 | 0.052865 | 0.274653 | 0.192478 |
| approach_velocity | global | -1 | 1332 | 0.290247 | 2.312763 | 7.968245 | 2.509499 | 12.094558 | 0.207490 |
| momentum_divergence | global | -1 | 1332 | -9.572825 | 114.830870 | 11.995505 | 4.460373 | 16.991960 | 0.262499 |
| opposing_volume_depth_ratio | global | -1 | 1332 | 1.495216 | 0.613530 | 0.410329 | 0.080285 | 0.514453 | 0.156059 |
| primary_volume_depth_ratio | global | -1 | 1332 | 1.465862 | 0.506131 | 0.345279 | 0.373332 | 0.363434 | 1.027237 |
| primary_prominence_z_score | global | -1 | 1332 | 0.021112 | 0.927235 | 43.919201 | 19.674852 | 29.293950 | 0.671635 |
| primary_dist_to_round_pct | global | -1 | 1332 | 0.000043 | 0.000028 | 0.653526 | 0.361815 | 0.725484 | 0.498722 |
| trend_strength_adx | global | -1 | 1332 | 18.514013 | 10.662035 | 0.575890 | 0.318006 | 0.523157 | 0.607859 |
| opposing_prominence_z_score | global | -1 | 1332 | 0.026957 | 0.941196 | 34.915226 | 6.831540 | 26.359387 | 0.259169 |
| test_count | global | -1 | 1332 | 19.300300 | 11.538871 | 0.597860 | 0.295214 | 0.475457 | 0.620905 |
| bollinger_squeeze | global | -1 | 1332 | 0.167593 | 0.126260 | 0.753377 | 0.428843 | 0.670376 | 0.639705 |
| dist_to_opposing_level_atr | global | -1 | 1332 | 3.333108 | 3.717591 | 1.115353 | 0.627331 | 1.172873 | 0.534867 |
| rubber_band_extension | global | -1 | 1332 | 0.011021 | 0.050224 | 4.556961 | 3.934463 | 5.271207 | 0.746406 |
| is_flip_candidate | global | -1 | 1332 | 0.211712 | 0.408522 | 1.929612 | 0.939289 | 1.995161 | 0.470783 |
| forward_return | global | -1 | 1236 | 0.038277 | 0.188732 | 4.930694 | 0.640308 | 4.216152 | 0.151870 |
| forward_return | regime | 0 | 5 | -0.013184 | 0.109672 | 8.318457 | nan | nan | nan |
| forward_return | regime | 1 | 674 | 0.036284 | 0.199800 | 5.506593 | nan | nan | nan |
| forward_return | regime | 2 | 557 | 0.041009 | 0.174674 | 4.259441 | nan | nan | nan |
| is_flip_candidate | regime | 0 | 5 | 0.600000 | 0.489898 | 0.816497 | nan | nan | nan |
| is_flip_candidate | regime | 1 | 722 | 0.285319 | 0.451566 | 1.582673 | nan | nan | nan |
| is_flip_candidate | regime | 2 | 605 | 0.120661 | 0.325733 | 2.699569 | nan | nan | nan |
| rubber_band_extension | regime | 0 | 5 | 0.099236 | 0.078705 | 0.793117 | nan | nan | nan |
| rubber_band_extension | regime | 1 | 722 | 0.022640 | 0.042988 | 1.898769 | nan | nan | nan |
| rubber_band_extension | regime | 2 | 605 | -0.002793 | 0.052596 | 18.830329 | nan | nan | nan |
| dist_to_opposing_level_atr | regime | 0 | 5 | 7.205874 | 5.352542 | 0.742803 | nan | nan | nan |
| dist_to_opposing_level_atr | regime | 1 | 722 | 4.321633 | 4.678561 | 1.082591 | nan | nan | nan |
| dist_to_opposing_level_atr | regime | 2 | 605 | 2.098321 | 1.696839 | 0.808665 | nan | nan | nan |
| bollinger_squeeze | regime | 0 | 5 | 0.319744 | 0.085185 | 0.266416 | nan | nan | nan |
| bollinger_squeeze | regime | 1 | 722 | 0.159438 | 0.126538 | 0.793653 | nan | nan | nan |
| bollinger_squeeze | regime | 2 | 605 | 0.176575 | 0.125327 | 0.709765 | nan | nan | nan |
| test_count | regime | 0 | 5 | 7.520000 | 4.582750 | 0.609408 | nan | nan | nan |
| test_count | regime | 1 | 722 | 20.857341 | 12.263853 | 0.587987 | nan | nan | nan |
| test_count | regime | 2 | 605 | 17.748760 | 10.682790 | 0.601889 | nan | nan | nan |
| opposing_prominence_z_score | regime | 0 | 5 | -0.341422 | 0.210866 | 0.617612 | nan | nan | nan |
| opposing_prominence_z_score | regime | 1 | 722 | -0.009593 | 0.920216 | 95.923985 | nan | nan | nan |
| opposing_prominence_z_score | regime | 2 | 605 | 0.089117 | 1.000596 | 11.227894 | nan | nan | nan |
| trend_strength_adx | regime | 0 | 5 | 30.801537 | 7.808695 | 0.253516 | nan | nan | nan |
| trend_strength_adx | regime | 1 | 722 | 19.525067 | 11.694042 | 0.598925 | nan | nan | nan |
| trend_strength_adx | regime | 2 | 605 | 17.377653 | 9.554474 | 0.549814 | nan | nan | nan |
| primary_dist_to_round_pct | regime | 0 | 5 | 0.000076 | 0.000037 | 0.487367 | nan | nan | nan |
| primary_dist_to_round_pct | regime | 1 | 722 | 0.000041 | 0.000028 | 0.691566 | nan | nan | nan |
| primary_dist_to_round_pct | regime | 2 | 605 | 0.000045 | 0.000028 | 0.621636 | nan | nan | nan |
| primary_prominence_z_score | regime | 0 | 5 | -0.860427 | 0.014187 | 0.016488 | nan | nan | nan |
| primary_prominence_z_score | regime | 1 | 722 | 0.066357 | 0.900753 | 13.574344 | nan | nan | nan |
| primary_prominence_z_score | regime | 2 | 605 | -0.033385 | 0.940449 | 28.169849 | nan | nan | nan |
| primary_volume_depth_ratio | regime | 0 | 5 | 2.623168 | 0.604990 | 0.230633 | nan | nan | nan |
| primary_volume_depth_ratio | regime | 1 | 722 | 1.420928 | 0.489783 | 0.344692 | nan | nan | nan |
| primary_volume_depth_ratio | regime | 2 | 605 | 1.508576 | 0.503457 | 0.333730 | nan | nan | nan |
| opposing_volume_depth_ratio | regime | 0 | 5 | 1.752684 | 0.989695 | 0.564674 | nan | nan | nan |
| opposing_volume_depth_ratio | regime | 1 | 722 | 1.554123 | 0.753660 | 0.484942 | nan | nan | nan |
| opposing_volume_depth_ratio | regime | 2 | 605 | 1.465581 | 0.564301 | 0.385036 | nan | nan | nan |
| momentum_divergence | regime | 0 | 5 | -101.145235 | 246.947784 | 2.441517 | nan | nan | nan |
| momentum_divergence | regime | 1 | 722 | -9.669970 | 100.612949 | 10.404680 | nan | nan | nan |
| momentum_divergence | regime | 2 | 605 | -11.494212 | 140.422445 | 12.216796 | nan | nan | nan |
| approach_velocity | regime | 0 | 5 | 1.654257 | 5.484215 | 3.315213 | nan | nan | nan |
| approach_velocity | regime | 1 | 722 | 0.383710 | 2.474912 | 6.449959 | nan | nan | nan |
| approach_velocity | regime | 2 | 605 | -0.065762 | 2.572116 | 39.112581 | nan | nan | nan |
| age_log_hours | regime | 0 | 5 | 2.996505 | 0.858002 | 0.286334 | nan | nan | nan |
| age_log_hours | regime | 1 | 722 | 3.278847 | 0.932458 | 0.284386 | nan | nan | nan |
| age_log_hours | regime | 2 | 605 | 3.420735 | 0.957798 | 0.279998 | nan | nan | nan |
| opposing_age_log_hours | regime | 0 | 5 | 3.801955 | 2.342353 | 0.616092 | nan | nan | nan |
| opposing_age_log_hours | regime | 1 | 722 | 4.300431 | 1.323447 | 0.307748 | nan | nan | nan |
| opposing_age_log_hours | regime | 2 | 605 | 3.686034 | 1.102552 | 0.299116 | nan | nan | nan |
| breakout_regime_0_prob | regime | 0 | 5 | 0.560434 | 0.110847 | 0.197788 | nan | nan | nan |
| breakout_regime_0_prob | regime | 1 | 722 | 0.009565 | 0.013287 | 1.389071 | nan | nan | nan |
| breakout_regime_0_prob | regime | 2 | 605 | 0.008491 | 0.011215 | 1.320847 | nan | nan | nan |
| breakout_regime_1_prob | regime | 0 | 5 | 0.209789 | 0.041545 | 0.198032 | nan | nan | nan |
| breakout_regime_1_prob | regime | 1 | 722 | 0.899876 | 0.106934 | 0.118832 | nan | nan | nan |
| breakout_regime_1_prob | regime | 2 | 605 | 0.073057 | 0.079776 | 1.091960 | nan | nan | nan |
| breakout_regime_2_prob | regime | 0 | 5 | 0.226435 | 0.070919 | 0.313199 | nan | nan | nan |
| breakout_regime_2_prob | regime | 1 | 722 | 0.086317 | 0.096840 | 1.121910 | nan | nan | nan |
| breakout_regime_2_prob | regime | 2 | 605 | 0.916655 | 0.086811 | 0.094704 | nan | nan | nan |
| breakout_long_edge_score | regime | 0 | 5 | 0.360520 | 0.067834 | 0.188156 | nan | nan | nan |
| breakout_long_edge_score | regime | 1 | 722 | 0.004813 | 0.541260 | 112.461898 | nan | nan | nan |
| breakout_long_edge_score | regime | 2 | 605 | 0.054591 | 0.560068 | 10.259414 | nan | nan | nan |
| breakout_short_edge_score | regime | 0 | 5 | -0.360520 | 0.067834 | 0.188156 | nan | nan | nan |
| breakout_short_edge_score | regime | 1 | 722 | -0.004813 | 0.541260 | 112.461898 | nan | nan | nan |
| breakout_short_edge_score | regime | 2 | 605 | -0.054591 | 0.560068 | 10.259414 | nan | nan | nan |
| breakout_bullish_prob | regime | 0 | 5 | 0.790211 | 0.041545 | 0.052574 | nan | nan | nan |
| breakout_bullish_prob | regime | 1 | 722 | 0.513184 | 0.412818 | 0.804425 | nan | nan | nan |
| breakout_bullish_prob | regime | 2 | 605 | 0.546199 | 0.429936 | 0.787141 | nan | nan | nan |
| breakout_bearish_prob | regime | 0 | 5 | 0.209789 | 0.041545 | 0.198032 | nan | nan | nan |
| breakout_bearish_prob | regime | 1 | 722 | 0.486816 | 0.412818 | 0.847995 | nan | nan | nan |
| breakout_bearish_prob | regime | 2 | 605 | 0.453801 | 0.429936 | 0.947411 | nan | nan | nan |
| breakout_level_strength | regime | 0 | 5 | 0.617071 | 0.033236 | 0.053861 | nan | nan | nan |
| breakout_level_strength | regime | 1 | 722 | 0.655611 | 0.072121 | 0.110006 | nan | nan | nan |
| breakout_level_strength | regime | 2 | 605 | 0.651804 | 0.075227 | 0.115414 | nan | nan | nan |

## Pairwise Winsorised CV Ratios for Breakout Regime Probabilities
Each row compares two regime groups (A,B) for a given probability metric, reporting between/within WCoV and their ratio.

| Metric | Regime A | Regime B | CV_between | CV_within | CV_between/within |
|--------|----------|----------|-----------|----------|--------------------|
| breakout_regime_0_prob | 0 | 1 | 27.101247 | 6.107048 | 4.437700 |
| breakout_regime_0_prob | 0 | 2 | 30.665514 | 6.781671 | 4.521823 |
| breakout_regime_0_prob | 1 | 2 | 0.059365 | 1.353791 | 0.043851 |
| breakout_regime_1_prob | 0 | 1 | 0.384745 | 0.082781 | 4.647720 |
| breakout_regime_1_prob | 0 | 2 | 0.922259 | 0.818312 | 1.127026 |
| breakout_regime_1_prob | 1 | 2 | 0.790423 | 0.178491 | 4.428366 |
| breakout_regime_2_prob | 0 | 1 | 0.802296 | 0.960567 | 0.835232 |
| breakout_regime_2_prob | 0 | 2 | 0.378231 | 0.086434 | 4.375957 |
| breakout_regime_2_prob | 1 | 2 | 0.894386 | 0.197817 | 4.521280 |

## Main Feature Drivers per Factor
For each factor, the table below lists the top features by absolute Spearman correlation with that factor.


### Factor: `forward_return_h96`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| trend_strength_adx | 0.3158 | 0.2186 | 1236 |
| bollinger_squeeze | 0.3104 | 0.1182 | 1236 |
| test_count | -0.2010 | -0.1573 | 1236 |
| primary_prominence_z_score | -0.1864 | -0.0930 | 1236 |
| opposing_prominence_z_score | -0.1130 | -0.0250 | 1236 |
| opposing_age_log_hours | -0.0656 | -0.0880 | 1236 |
| primary_dist_to_round_pct | 0.0533 | 0.1163 | 1236 |
| primary_volume_depth_ratio | 0.0520 | 0.1069 | 1236 |
| momentum_divergence | 0.0423 | 0.0997 | 1236 |
| dist_to_opposing_level_atr | -0.0409 | -0.0847 | 1236 |
| is_flip_candidate | -0.0400 | -0.0657 | 1236 |
| age_log_hours | 0.0304 | 0.0802 | 1236 |
| rubber_band_extension | 0.0070 | -0.0054 | 1236 |
| opposing_volume_depth_ratio | -0.0058 | -0.0072 | 1236 |
| approach_velocity | 0.0032 | 0.0218 | 1236 |

### Factor: `breakout_long_edge_score`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| bollinger_squeeze | 0.3373 | 0.2556 | 1332 |
| trend_strength_adx | 0.2993 | 0.2410 | 1332 |
| test_count | -0.2120 | -0.2154 | 1332 |
| primary_prominence_z_score | -0.1966 | -0.2093 | 1332 |
| opposing_prominence_z_score | -0.1774 | -0.2064 | 1332 |
| primary_dist_to_round_pct | 0.0787 | 0.1017 | 1332 |
| opposing_volume_depth_ratio | -0.0533 | 0.0027 | 1332 |
| rubber_band_extension | 0.0360 | 0.0230 | 1332 |
| opposing_age_log_hours | -0.0356 | -0.0222 | 1332 |
| approach_velocity | 0.0328 | 0.0414 | 1332 |
| is_flip_candidate | 0.0311 | 0.0714 | 1332 |
| dist_to_opposing_level_atr | 0.0242 | 0.0186 | 1332 |
| age_log_hours | 0.0236 | 0.0343 | 1332 |
| primary_volume_depth_ratio | -0.0079 | 0.0889 | 1332 |
| momentum_divergence | 0.0006 | 0.0351 | 1332 |

### Factor: `breakout_short_edge_score`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| bollinger_squeeze | -0.3373 | -0.2556 | 1332 |
| trend_strength_adx | -0.2993 | -0.2410 | 1332 |
| test_count | 0.2120 | 0.2154 | 1332 |
| primary_prominence_z_score | 0.1966 | 0.2093 | 1332 |
| opposing_prominence_z_score | 0.1774 | 0.2064 | 1332 |
| primary_dist_to_round_pct | -0.0787 | -0.1017 | 1332 |
| opposing_volume_depth_ratio | 0.0533 | -0.0027 | 1332 |
| rubber_band_extension | -0.0360 | -0.0230 | 1332 |
| opposing_age_log_hours | 0.0356 | 0.0222 | 1332 |
| approach_velocity | -0.0328 | -0.0414 | 1332 |
| is_flip_candidate | -0.0311 | -0.0714 | 1332 |
| dist_to_opposing_level_atr | -0.0242 | -0.0186 | 1332 |
| age_log_hours | -0.0236 | -0.0343 | 1332 |
| primary_volume_depth_ratio | 0.0079 | -0.0889 | 1332 |
| momentum_divergence | -0.0006 | -0.0351 | 1332 |

### Factor: `breakout_bullish_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| bollinger_squeeze | 0.3099 | 0.2481 | 1332 |
| trend_strength_adx | 0.3077 | 0.2473 | 1332 |
| test_count | -0.2160 | -0.2114 | 1332 |
| primary_prominence_z_score | -0.2011 | -0.1952 | 1332 |
| opposing_prominence_z_score | -0.1582 | -0.1848 | 1332 |
| age_log_hours | 0.0601 | 0.0373 | 1332 |
| primary_dist_to_round_pct | 0.0518 | 0.0890 | 1332 |
| opposing_volume_depth_ratio | -0.0515 | -0.0043 | 1332 |
| primary_volume_depth_ratio | 0.0465 | 0.1074 | 1332 |
| opposing_age_log_hours | -0.0440 | -0.0247 | 1332 |
| is_flip_candidate | 0.0419 | 0.0732 | 1332 |
| momentum_divergence | 0.0271 | 0.0468 | 1332 |
| approach_velocity | 0.0172 | 0.0382 | 1332 |
| rubber_band_extension | 0.0154 | 0.0212 | 1332 |
| dist_to_opposing_level_atr | 0.0119 | 0.0194 | 1332 |

### Factor: `breakout_bearish_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| bollinger_squeeze | -0.3099 | -0.2481 | 1332 |
| trend_strength_adx | -0.3077 | -0.2473 | 1332 |
| test_count | 0.2160 | 0.2114 | 1332 |
| primary_prominence_z_score | 0.2011 | 0.1952 | 1332 |
| opposing_prominence_z_score | 0.1582 | 0.1848 | 1332 |
| age_log_hours | -0.0601 | -0.0373 | 1332 |
| primary_dist_to_round_pct | -0.0518 | -0.0890 | 1332 |
| opposing_volume_depth_ratio | 0.0515 | 0.0043 | 1332 |
| primary_volume_depth_ratio | -0.0465 | -0.1074 | 1332 |
| opposing_age_log_hours | 0.0440 | 0.0247 | 1332 |
| is_flip_candidate | -0.0419 | -0.0732 | 1332 |
| momentum_divergence | -0.0271 | -0.0468 | 1332 |
| approach_velocity | -0.0172 | -0.0382 | 1332 |
| rubber_band_extension | -0.0154 | -0.0212 | 1332 |
| dist_to_opposing_level_atr | -0.0119 | -0.0194 | 1332 |

### Factor: `breakout_regime_0_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| test_count | -0.5600 | -0.1819 | 1332 |
| bollinger_squeeze | 0.5400 | 0.2403 | 1332 |
| trend_strength_adx | 0.4542 | 0.2573 | 1332 |
| primary_prominence_z_score | -0.4349 | -0.1382 | 1332 |
| rubber_band_extension | 0.3720 | 0.2182 | 1332 |
| primary_volume_depth_ratio | 0.3643 | 0.2186 | 1332 |
| age_log_hours | 0.2563 | 0.0317 | 1332 |
| primary_dist_to_round_pct | 0.2185 | 0.1309 | 1332 |
| approach_velocity | 0.1727 | 0.1192 | 1332 |
| opposing_prominence_z_score | -0.1088 | -0.0672 | 1332 |
| opposing_volume_depth_ratio | 0.1060 | 0.0141 | 1332 |
| opposing_age_log_hours | -0.1054 | -0.0143 | 1332 |
| dist_to_opposing_level_atr | -0.0872 | 0.0548 | 1332 |
| is_flip_candidate | -0.0692 | 0.0312 | 1332 |
| momentum_divergence | 0.0295 | 0.0061 | 1332 |

### Factor: `breakout_regime_1_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| rubber_band_extension | 0.3349 | 0.1880 | 1332 |
| dist_to_opposing_level_atr | 0.2587 | 0.2429 | 1332 |
| opposing_age_log_hours | 0.2569 | 0.2294 | 1332 |
| is_flip_candidate | 0.2151 | 0.2066 | 1332 |
| opposing_volume_depth_ratio | 0.1215 | 0.0615 | 1332 |
| primary_volume_depth_ratio | -0.1033 | -0.0548 | 1332 |
| primary_dist_to_round_pct | -0.0988 | -0.0883 | 1332 |
| trend_strength_adx | 0.0938 | 0.0997 | 1332 |
| age_log_hours | -0.0893 | -0.0693 | 1332 |
| bollinger_squeeze | -0.0837 | -0.0580 | 1332 |
| test_count | 0.0804 | 0.1134 | 1332 |
| approach_velocity | -0.0600 | 0.0511 | 1332 |
| primary_prominence_z_score | 0.0511 | 0.0436 | 1332 |
| opposing_prominence_z_score | -0.0350 | -0.0381 | 1332 |
| momentum_divergence | 0.0149 | 0.0261 | 1332 |

### Factor: `breakout_regime_2_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| rubber_band_extension | -0.3763 | -0.2121 | 1332 |
| dist_to_opposing_level_atr | -0.2718 | -0.2487 | 1332 |
| opposing_age_log_hours | -0.2613 | -0.2275 | 1332 |
| is_flip_candidate | -0.2230 | -0.2098 | 1332 |
| trend_strength_adx | -0.1511 | -0.1284 | 1332 |
| opposing_volume_depth_ratio | -0.1170 | -0.0630 | 1332 |
| age_log_hours | 0.0764 | 0.0657 | 1332 |
| primary_dist_to_round_pct | 0.0754 | 0.0736 | 1332 |
| primary_volume_depth_ratio | 0.0695 | 0.0303 | 1332 |
| approach_velocity | 0.0540 | -0.0643 | 1332 |
| opposing_prominence_z_score | 0.0424 | 0.0456 | 1332 |
| test_count | -0.0348 | -0.0929 | 1332 |
| bollinger_squeeze | 0.0259 | 0.0310 | 1332 |
| momentum_divergence | -0.0137 | -0.0267 | 1332 |
| primary_prominence_z_score | -0.0101 | -0.0281 | 1332 |