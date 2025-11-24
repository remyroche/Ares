# Breakout/Bounce Regime Diagnostics

- Symbol: **ETHUSDT**
- Exchange: **binance**
- Timeframe: **1h**
- Direction: **long**
- Horizon (bars): **96**
- Samples (training window): **5630**

## Global Model Metrics
- Validation log loss: **nan**
- Precision (breakout class 1): **0.4633**

### Class Counts (training labels)
| Regime | Count |
|--------|-------|
| 0 | 442 |
| 1 | 2528 |
| 2 | 2660 |

## Forward Return Sharpe-like Ratios
| Scope | Regime | Mean Return | Std Return | Sharpe-like |
|-------|--------|-------------|------------|-------------|
| global | -1 | 0.030688 | 0.140231 | 0.2188 |
| regime | 0 | 0.001068 | 0.045401 | 0.0235 |
| regime | 1 | 0.026980 | 0.136804 | 0.1972 |
| regime | 2 | 0.036314 | 0.147464 | 0.2463 |

## Winsorised CV Between/Within Regimes
Definition: per metric, we winsorise values (5–95%), compute global and per-regime CV, and report the ratio of between-regime CV to within-regime CV.

| Metric | Scope | Regime | Count | Mean | Std | CV | CV_between | CV_within | CV_between/within |
|--------|-------|--------|-------|------|-----|----|-----------|----------|--------------------|
| breakout_level_strength | global | -1 | 5630 | 0.646075 | 0.059164 | 0.091575 | 0.065183 | 0.099453 | 0.655419 |
| breakout_bearish_prob | global | -1 | 5630 | 0.466594 | 0.384050 | 0.823092 | 0.140000 | 0.805820 | 0.173736 |
| breakout_bullish_prob | global | -1 | 5630 | 0.533406 | 0.384050 | 0.719996 | 0.122465 | 0.704887 | 0.173736 |
| breakout_short_edge_score | global | -1 | 5630 | -0.043385 | 0.494516 | 11.398219 | 2.680801 | 11.542111 | 0.232263 |
| breakout_long_edge_score | global | -1 | 5630 | 0.043385 | 0.494516 | 11.398219 | 2.680801 | 11.542111 | 0.232263 |
| breakout_regime_2_prob | global | -1 | 5630 | 0.490566 | 0.384559 | 0.783909 | 0.702012 | 0.235726 | 2.978088 |
| breakout_regime_1_prob | global | -1 | 5630 | 0.464082 | 0.383906 | 0.827238 | 0.750064 | 0.246798 | 3.039186 |
| breakout_regime_0_prob | global | -1 | 5630 | 0.025429 | 0.057433 | 2.258579 | 13.694816 | 2.794865 | 4.899992 |
| trend_strength_adx | global | -1 | 5630 | 20.209757 | 10.882229 | 0.538464 | 0.060131 | 0.497026 | 0.120981 |
| age_log_hours | global | -1 | 5630 | 3.230563 | 0.984491 | 0.304743 | 0.044290 | 0.324149 | 0.136635 |
| momentum_divergence | global | -1 | 5630 | 2.365355 | 68.290518 | 28.871146 | 3.782474 | 27.967419 | 0.135246 |
| primary_dist_to_round_pct | global | -1 | 5630 | 0.000055 | 0.000036 | 0.656156 | 0.069036 | 0.687396 | 0.100431 |
| dist_to_opposing_level_atr | global | -1 | 5630 | 3.210167 | 3.008249 | 0.937100 | 0.137976 | 0.774076 | 0.178246 |
| bollinger_squeeze | global | -1 | 5630 | 0.121478 | 0.107816 | 0.887531 | 0.186070 | 0.806614 | 0.230680 |
| primary_volume_depth_ratio | global | -1 | 5630 | 1.458701 | 0.410670 | 0.281531 | 0.003167 | 0.303079 | 0.010451 |
| rubber_band_extension | global | -1 | 5630 | 0.006306 | 0.034401 | 5.455004 | 0.884687 | 5.104369 | 0.173320 |
| test_count | global | -1 | 5630 | 25.869627 | 16.098003 | 0.622274 | 0.299234 | 0.804939 | 0.371747 |
| opposing_dist_to_round_pct | global | -1 | 5630 | 0.000057 | 0.000039 | 0.681558 | 0.183768 | 0.663433 | 0.276995 |
| primary_prominence_z_score | global | -1 | 5630 | -0.101957 | 0.788940 | 7.737971 | 5.345462 | 9.204373 | 0.580752 |
| opposing_age_log_hours | global | -1 | 5630 | 3.929225 | 1.213634 | 0.308874 | 0.014075 | 0.300072 | 0.046904 |
| opposing_volume_depth_ratio | global | -1 | 5630 | 1.470897 | 0.461308 | 0.313624 | 0.016543 | 0.283844 | 0.058283 |
| opposing_prominence_z_score | global | -1 | 5630 | -0.019022 | 0.892441 | 46.917466 | 29.264264 | 51.992399 | 0.562857 |
| is_flip_candidate | global | -1 | 5630 | 0.200178 | 0.400133 | 1.998891 | 0.361882 | 1.767919 | 0.204694 |
| forward_return | global | -1 | 5534 | 0.027063 | 0.117365 | 4.336733 | 0.529671 | 3.478691 | 0.152262 |
| forward_return | regime | 0 | 208 | -0.000716 | 0.041177 | 57.513770 | nan | nan | nan |
| forward_return | regime | 1 | 2550 | 0.023403 | 0.114100 | 4.875497 | nan | nan | nan |
| forward_return | regime | 2 | 2776 | 0.033442 | 0.127153 | 3.802167 | nan | nan | nan |
| is_flip_candidate | regime | 0 | 208 | 0.072115 | 0.258679 | 3.587014 | nan | nan | nan |
| is_flip_candidate | regime | 1 | 2605 | 0.249520 | 0.432735 | 1.734270 | nan | nan | nan |
| is_flip_candidate | regime | 2 | 2817 | 0.164004 | 0.370279 | 2.257743 | nan | nan | nan |
| opposing_prominence_z_score | regime | 0 | 208 | 1.112818 | 1.286996 | 1.156520 | nan | nan | nan |
| opposing_prominence_z_score | regime | 1 | 2605 | -0.108160 | 0.820166 | 7.582862 | nan | nan | nan |
| opposing_prominence_z_score | regime | 2 | 2817 | -0.023293 | 0.859761 | 36.910801 | nan | nan | nan |
| opposing_volume_depth_ratio | regime | 0 | 208 | 1.455794 | 0.296026 | 0.203344 | nan | nan | nan |
| opposing_volume_depth_ratio | regime | 1 | 2605 | 1.448225 | 0.429233 | 0.296385 | nan | nan | nan |
| opposing_volume_depth_ratio | regime | 2 | 2817 | 1.503211 | 0.527256 | 0.350753 | nan | nan | nan |
| opposing_age_log_hours | regime | 0 | 208 | 3.918660 | 1.120899 | 0.286041 | nan | nan | nan |
| opposing_age_log_hours | regime | 1 | 2605 | 3.996069 | 1.363867 | 0.341302 | nan | nan | nan |
| opposing_age_log_hours | regime | 2 | 2817 | 3.861091 | 1.052385 | 0.272562 | nan | nan | nan |
| primary_prominence_z_score | regime | 0 | 208 | 1.005333 | 1.384721 | 1.377376 | nan | nan | nan |
| primary_prominence_z_score | regime | 1 | 2605 | -0.112511 | 0.781140 | 6.942778 | nan | nan | nan |
| primary_prominence_z_score | regime | 2 | 2817 | -0.185621 | 0.649487 | 3.499000 | nan | nan | nan |
| opposing_dist_to_round_pct | regime | 0 | 208 | 0.000077 | 0.000036 | 0.466604 | nan | nan | nan |
| opposing_dist_to_round_pct | regime | 1 | 2605 | 0.000053 | 0.000037 | 0.699900 | nan | nan | nan |
| opposing_dist_to_round_pct | regime | 2 | 2817 | 0.000059 | 0.000040 | 0.675917 | nan | nan | nan |
| test_count | regime | 0 | 208 | 41.947115 | 30.911973 | 0.736927 | nan | nan | nan |
| test_count | regime | 1 | 2605 | 26.036852 | 16.122235 | 0.619208 | nan | nan | nan |
| test_count | regime | 2 | 2817 | 25.058573 | 15.436212 | 0.616005 | nan | nan | nan |
| rubber_band_extension | regime | 0 | 208 | 0.017535 | 0.027601 | 1.574032 | nan | nan | nan |
| rubber_band_extension | regime | 1 | 2605 | 0.004935 | 0.031693 | 6.422559 | nan | nan | nan |
| rubber_band_extension | regime | 2 | 2817 | 0.006653 | 0.037275 | 5.602398 | nan | nan | nan |
| primary_volume_depth_ratio | regime | 0 | 208 | 1.465246 | 0.511871 | 0.349342 | nan | nan | nan |
| primary_volume_depth_ratio | regime | 1 | 2605 | 1.454469 | 0.385812 | 0.265260 | nan | nan | nan |
| primary_volume_depth_ratio | regime | 2 | 2817 | 1.462851 | 0.428623 | 0.293005 | nan | nan | nan |
| bollinger_squeeze | regime | 0 | 208 | 0.075576 | 0.077800 | 1.029426 | nan | nan | nan |
| bollinger_squeeze | regime | 1 | 2605 | 0.119016 | 0.104657 | 0.879353 | nan | nan | nan |
| bollinger_squeeze | regime | 2 | 2817 | 0.127025 | 0.111502 | 0.877792 | nan | nan | nan |
| dist_to_opposing_level_atr | regime | 0 | 208 | 2.984129 | 1.406196 | 0.471225 | nan | nan | nan |
| dist_to_opposing_level_atr | regime | 1 | 2605 | 3.769982 | 3.790758 | 1.005511 | nan | nan | nan |
| dist_to_opposing_level_atr | regime | 2 | 2817 | 2.729247 | 2.257783 | 0.827255 | nan | nan | nan |
| primary_dist_to_round_pct | regime | 0 | 208 | 0.000062 | 0.000041 | 0.662494 | nan | nan | nan |
| primary_dist_to_round_pct | regime | 1 | 2605 | 0.000053 | 0.000035 | 0.662461 | nan | nan | nan |
| primary_dist_to_round_pct | regime | 2 | 2817 | 0.000057 | 0.000038 | 0.657184 | nan | nan | nan |
| momentum_divergence | regime | 0 | 208 | 19.585113 | 55.782522 | 2.848210 | nan | nan | nan |
| momentum_divergence | regime | 1 | 2605 | -0.142438 | 76.318966 | 535.802970 | nan | nan | nan |
| momentum_divergence | regime | 2 | 2817 | 1.455207 | 66.357158 | 45.599792 | nan | nan | nan |
| age_log_hours | regime | 0 | 208 | 2.940510 | 1.170324 | 0.398000 | nan | nan | nan |
| age_log_hours | regime | 1 | 2605 | 3.227449 | 0.993563 | 0.307848 | nan | nan | nan |
| age_log_hours | regime | 2 | 2817 | 3.258268 | 0.977665 | 0.300057 | nan | nan | nan |
| trend_strength_adx | regime | 0 | 208 | 17.806052 | 8.220394 | 0.461663 | nan | nan | nan |
| trend_strength_adx | regime | 1 | 2605 | 19.875594 | 11.109636 | 0.558959 | nan | nan | nan |
| trend_strength_adx | regime | 2 | 2817 | 20.693732 | 10.804275 | 0.522104 | nan | nan | nan |
| breakout_regime_0_prob | regime | 0 | 208 | 0.751198 | 0.176575 | 0.235058 | nan | nan | nan |
| breakout_regime_0_prob | regime | 1 | 2605 | 0.012533 | 0.018405 | 1.468440 | nan | nan | nan |
| breakout_regime_0_prob | regime | 2 | 2817 | 0.012387 | 0.018231 | 1.471788 | nan | nan | nan |
| breakout_regime_1_prob | regime | 0 | 208 | 0.116790 | 0.107512 | 0.920557 | nan | nan | nan |
| breakout_regime_1_prob | regime | 1 | 2605 | 0.859109 | 0.122483 | 0.142569 | nan | nan | nan |
| breakout_regime_1_prob | regime | 2 | 2817 | 0.124665 | 0.113609 | 0.911314 | nan | nan | nan |
| breakout_regime_2_prob | regime | 0 | 208 | 0.131758 | 0.112446 | 0.853427 | nan | nan | nan |
| breakout_regime_2_prob | regime | 1 | 2605 | 0.122336 | 0.109559 | 0.895560 | nan | nan | nan |
| breakout_regime_2_prob | regime | 2 | 2817 | 0.857548 | 0.124912 | 0.145662 | nan | nan | nan |
| breakout_long_edge_score | regime | 0 | 208 | 0.277912 | 0.520097 | 1.871447 | nan | nan | nan |
| breakout_long_edge_score | regime | 1 | 2605 | 0.012682 | 0.485292 | 38.266709 | nan | nan | nan |
| breakout_long_edge_score | regime | 2 | 2817 | 0.055222 | 0.496888 | 8.997959 | nan | nan | nan |
| breakout_short_edge_score | regime | 0 | 208 | -0.277912 | 0.520097 | 1.871447 | nan | nan | nan |
| breakout_short_edge_score | regime | 1 | 2605 | -0.012682 | 0.485292 | 38.266709 | nan | nan | nan |
| breakout_short_edge_score | regime | 2 | 2817 | -0.055222 | 0.496888 | 8.997959 | nan | nan | nan |
| breakout_bullish_prob | regime | 0 | 208 | 0.663357 | 0.361125 | 0.544390 | nan | nan | nan |
| breakout_bullish_prob | regime | 1 | 2605 | 0.511268 | 0.378334 | 0.739992 | nan | nan | nan |
| breakout_bullish_prob | regime | 2 | 2817 | 0.544255 | 0.388513 | 0.713844 | nan | nan | nan |
| breakout_bearish_prob | regime | 0 | 208 | 0.336643 | 0.361125 | 1.072722 | nan | nan | nan |
| breakout_bearish_prob | regime | 1 | 2605 | 0.488732 | 0.378334 | 0.774115 | nan | nan | nan |
| breakout_bearish_prob | regime | 2 | 2817 | 0.455745 | 0.388513 | 0.852480 | nan | nan | nan |
| breakout_level_strength | regime | 0 | 208 | 0.731689 | 0.082874 | 0.113264 | nan | nan | nan |
| breakout_level_strength | regime | 1 | 2605 | 0.645272 | 0.058458 | 0.090595 | nan | nan | nan |
| breakout_level_strength | regime | 2 | 2817 | 0.639695 | 0.051430 | 0.080398 | nan | nan | nan |

## Pairwise Winsorised CV Ratios for Breakout Regime Probabilities
Each row compares two regime groups (A,B) for a given probability metric, reporting between/within WCoV and their ratio.

| Metric | Regime A | Regime B | CV_between | CV_within | CV_between/within |
|--------|----------|----------|-----------|----------|--------------------|
| breakout_regime_0_prob | 0 | 1 | 5.923115 | 1.563484 | 3.788409 |
| breakout_regime_0_prob | 0 | 2 | 6.585902 | 1.736542 | 3.792539 |
| breakout_regime_0_prob | 1 | 2 | 0.005899 | 1.475045 | 0.003999 |
| breakout_regime_1_prob | 0 | 1 | 0.460232 | 0.142595 | 3.227552 |
| breakout_regime_1_prob | 0 | 2 | 0.031791 | 0.892685 | 0.035613 |
| breakout_regime_1_prob | 1 | 2 | 0.769231 | 0.247274 | 3.110848 |
| breakout_regime_2_prob | 0 | 1 | 0.038353 | 0.903712 | 0.042440 |
| breakout_regime_2_prob | 0 | 2 | 0.447340 | 0.146295 | 3.057789 |
| breakout_regime_2_prob | 1 | 2 | 0.728955 | 0.232476 | 3.135616 |

## Main Feature Drivers per Factor
For each factor, the table below lists the top features by absolute Spearman correlation with that factor.


### Factor: `forward_return_h96`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| trend_strength_adx | 0.1118 | 0.1060 | 5534 |
| test_count | -0.1037 | -0.1102 | 5534 |
| opposing_dist_to_round_pct | 0.0852 | 0.0708 | 5534 |
| bollinger_squeeze | 0.0799 | 0.0281 | 5534 |
| age_log_hours | 0.0661 | 0.0795 | 5534 |
| opposing_prominence_z_score | -0.0596 | -0.0875 | 5534 |
| primary_prominence_z_score | -0.0465 | -0.0855 | 5534 |
| is_flip_candidate | 0.0399 | 0.0229 | 5534 |
| dist_to_opposing_level_atr | -0.0330 | -0.0136 | 5534 |
| primary_dist_to_round_pct | 0.0289 | 0.0388 | 5534 |
| momentum_divergence | -0.0257 | 0.0262 | 5534 |
| rubber_band_extension | 0.0131 | 0.0101 | 5534 |
| primary_volume_depth_ratio | 0.0111 | 0.0247 | 5534 |
| opposing_volume_depth_ratio | -0.0093 | -0.0193 | 5534 |
| opposing_age_log_hours | 0.0065 | -0.0014 | 5534 |

### Factor: `breakout_long_edge_score`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| momentum_divergence | -0.1225 | -0.0863 | 5630 |
| is_flip_candidate | 0.0957 | 0.0856 | 5630 |
| opposing_age_log_hours | 0.0704 | 0.0647 | 5630 |
| primary_volume_depth_ratio | -0.0543 | -0.0180 | 5630 |
| primary_prominence_z_score | 0.0530 | -0.0072 | 5630 |
| opposing_dist_to_round_pct | 0.0522 | 0.0703 | 5630 |
| primary_dist_to_round_pct | 0.0474 | 0.0812 | 5630 |
| rubber_band_extension | -0.0394 | 0.0018 | 5630 |
| opposing_volume_depth_ratio | -0.0262 | -0.0132 | 5630 |
| age_log_hours | 0.0253 | 0.0392 | 5630 |
| bollinger_squeeze | -0.0228 | -0.0038 | 5630 |
| dist_to_opposing_level_atr | 0.0159 | 0.0221 | 5630 |
| test_count | -0.0117 | -0.0351 | 5630 |
| opposing_prominence_z_score | 0.0095 | -0.0425 | 5630 |
| trend_strength_adx | 0.0072 | 0.0271 | 5630 |

### Factor: `breakout_short_edge_score`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| momentum_divergence | 0.1225 | 0.0863 | 5630 |
| is_flip_candidate | -0.0957 | -0.0856 | 5630 |
| opposing_age_log_hours | -0.0704 | -0.0647 | 5630 |
| primary_volume_depth_ratio | 0.0543 | 0.0180 | 5630 |
| primary_prominence_z_score | -0.0530 | 0.0072 | 5630 |
| opposing_dist_to_round_pct | -0.0522 | -0.0703 | 5630 |
| primary_dist_to_round_pct | -0.0474 | -0.0812 | 5630 |
| rubber_band_extension | 0.0394 | -0.0018 | 5630 |
| opposing_volume_depth_ratio | 0.0262 | 0.0132 | 5630 |
| age_log_hours | -0.0253 | -0.0392 | 5630 |
| bollinger_squeeze | 0.0228 | 0.0038 | 5630 |
| dist_to_opposing_level_atr | -0.0159 | -0.0221 | 5630 |
| test_count | 0.0117 | 0.0351 | 5630 |
| opposing_prominence_z_score | -0.0095 | 0.0425 | 5630 |
| trend_strength_adx | -0.0072 | -0.0271 | 5630 |

### Factor: `breakout_bullish_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| momentum_divergence | -0.1418 | -0.0940 | 5630 |
| is_flip_candidate | 0.1092 | 0.0873 | 5630 |
| opposing_age_log_hours | 0.0752 | 0.0630 | 5630 |
| opposing_dist_to_round_pct | 0.0574 | 0.0782 | 5630 |
| primary_volume_depth_ratio | -0.0567 | -0.0158 | 5630 |
| primary_dist_to_round_pct | 0.0500 | 0.0832 | 5630 |
| rubber_band_extension | -0.0440 | 0.0028 | 5630 |
| age_log_hours | 0.0388 | 0.0411 | 5630 |
| test_count | -0.0307 | -0.0370 | 5630 |
| primary_prominence_z_score | 0.0305 | -0.0046 | 5630 |
| opposing_volume_depth_ratio | -0.0268 | -0.0123 | 5630 |
| dist_to_opposing_level_atr | 0.0239 | 0.0245 | 5630 |
| opposing_prominence_z_score | -0.0181 | -0.0407 | 5630 |
| bollinger_squeeze | -0.0166 | -0.0030 | 5630 |
| trend_strength_adx | 0.0106 | 0.0264 | 5630 |

### Factor: `breakout_bearish_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| momentum_divergence | 0.1418 | 0.0940 | 5630 |
| is_flip_candidate | -0.1092 | -0.0873 | 5630 |
| opposing_age_log_hours | -0.0752 | -0.0630 | 5630 |
| opposing_dist_to_round_pct | -0.0574 | -0.0782 | 5630 |
| primary_volume_depth_ratio | 0.0567 | 0.0158 | 5630 |
| primary_dist_to_round_pct | -0.0500 | -0.0832 | 5630 |
| rubber_band_extension | 0.0440 | -0.0028 | 5630 |
| age_log_hours | -0.0388 | -0.0411 | 5630 |
| test_count | 0.0307 | 0.0370 | 5630 |
| primary_prominence_z_score | -0.0305 | 0.0046 | 5630 |
| opposing_volume_depth_ratio | 0.0268 | 0.0123 | 5630 |
| dist_to_opposing_level_atr | -0.0239 | -0.0245 | 5630 |
| opposing_prominence_z_score | 0.0181 | 0.0407 | 5630 |
| bollinger_squeeze | 0.0166 | 0.0030 | 5630 |
| trend_strength_adx | -0.0106 | -0.0264 | 5630 |

### Factor: `breakout_regime_0_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| rubber_band_extension | 0.2631 | 0.0833 | 5630 |
| opposing_dist_to_round_pct | 0.2433 | 0.1267 | 5630 |
| opposing_prominence_z_score | 0.2331 | 0.2643 | 5630 |
| is_flip_candidate | -0.0951 | -0.0611 | 5630 |
| primary_prominence_z_score | 0.0903 | 0.2657 | 5630 |
| momentum_divergence | 0.0817 | 0.0456 | 5630 |
| primary_dist_to_round_pct | 0.0815 | 0.0571 | 5630 |
| dist_to_opposing_level_atr | 0.0772 | -0.0088 | 5630 |
| age_log_hours | -0.0771 | -0.0467 | 5630 |
| bollinger_squeeze | -0.0629 | -0.0837 | 5630 |
| trend_strength_adx | 0.0578 | -0.0484 | 5630 |
| opposing_volume_depth_ratio | -0.0450 | -0.0197 | 5630 |
| primary_volume_depth_ratio | 0.0400 | -0.0117 | 5630 |
| test_count | 0.0359 | 0.1843 | 5630 |
| opposing_age_log_hours | -0.0341 | -0.0020 | 5630 |

### Factor: `breakout_regime_1_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| is_flip_candidate | 0.1498 | 0.1322 | 5630 |
| opposing_prominence_z_score | -0.1362 | -0.1105 | 5630 |
| opposing_dist_to_round_pct | -0.0967 | -0.0995 | 5630 |
| primary_dist_to_round_pct | -0.0718 | -0.0669 | 5630 |
| dist_to_opposing_level_atr | 0.0622 | 0.1519 | 5630 |
| rubber_band_extension | -0.0581 | -0.0328 | 5630 |
| trend_strength_adx | -0.0520 | -0.0359 | 5630 |
| opposing_age_log_hours | 0.0409 | 0.0638 | 5630 |
| momentum_divergence | -0.0290 | -0.0071 | 5630 |
| primary_prominence_z_score | -0.0268 | -0.0147 | 5630 |
| opposing_volume_depth_ratio | -0.0265 | -0.0422 | 5630 |
| bollinger_squeeze | 0.0212 | -0.0153 | 5630 |
| test_count | -0.0150 | -0.0136 | 5630 |
| primary_volume_depth_ratio | 0.0128 | -0.0122 | 5630 |
| age_log_hours | -0.0017 | -0.0055 | 5630 |

### Factor: `breakout_regime_2_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| is_flip_candidate | -0.1382 | -0.1082 | 5630 |
| dist_to_opposing_level_atr | -0.1213 | -0.1482 | 5630 |
| trend_strength_adx | 0.0783 | 0.0546 | 5630 |
| primary_prominence_z_score | -0.0549 | -0.0883 | 5630 |
| opposing_age_log_hours | -0.0549 | -0.0629 | 5630 |
| bollinger_squeeze | 0.0508 | 0.0477 | 5630 |
| primary_dist_to_round_pct | 0.0448 | 0.0446 | 5630 |
| test_count | -0.0425 | -0.0579 | 5630 |
| age_log_hours | 0.0418 | 0.0236 | 5630 |
| opposing_prominence_z_score | 0.0369 | 0.0079 | 5630 |
| opposing_dist_to_round_pct | 0.0287 | 0.0502 | 5630 |
| momentum_divergence | 0.0263 | -0.0106 | 5630 |
| opposing_volume_depth_ratio | 0.0233 | 0.0497 | 5630 |
| rubber_band_extension | 0.0198 | 0.0005 | 5630 |
| primary_volume_depth_ratio | 0.0037 | 0.0167 | 5630 |