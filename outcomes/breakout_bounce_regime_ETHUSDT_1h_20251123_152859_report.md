# Breakout/Bounce Regime Diagnostics

- Symbol: **ETHUSDT**
- Exchange: **binance**
- Timeframe: **1h**
- Direction: **long**
- Horizon (bars): **6**
- Samples (training window): **5692**

## Global Model Metrics
- Validation log loss: **1.174302**
- Precision (breakout class 1): **0.3650**

### Class Counts (training labels)
| Regime | Count |
|--------|-------|
| 0 | 2529 |
| 1 | 1280 |
| 2 | 1874 |
| 3 | 9 |

## Forward Return Sharpe-like Ratios
| Scope | Regime | Mean Return | Std Return | Sharpe-like |
|-------|--------|-------------|------------|-------------|
| global | -1 | 0.001802 | 0.035009 | 0.0515 |
| regime | 0 | 0.000742 | 0.023355 | 0.0318 |
| regime | 1 | 0.003380 | 0.046290 | 0.0730 |
| regime | 2 | 0.001770 | 0.035575 | 0.0498 |

## Winsorised CV Between/Within Regimes
Definition: per metric, we winsorise values (5–95%), compute global and per-regime CV, and report the ratio of between-regime CV to within-regime CV.

| Metric | Scope | Regime | Count | Mean | Std | CV | CV_between | CV_within | CV_between/within |
|--------|-------|--------|-------|------|-----|----|-----------|----------|--------------------|
| breakout_level_strength | global | -1 | 5692 | 0.645294 | 0.059099 | 0.091584 | 0.027759 | 0.081488 | 0.340655 |
| breakout_bearish_prob | global | -1 | 5692 | 0.474251 | 0.360123 | 0.759351 | 0.022576 | 0.719514 | 0.031376 |
| breakout_bullish_prob | global | -1 | 5692 | 0.523387 | 0.360130 | 0.688076 | 0.020725 | 0.651954 | 0.031788 |
| breakout_short_edge_score | global | -1 | 5692 | -0.033242 | 0.464512 | 13.973553 | 0.472316 | 13.290524 | 0.035538 |
| breakout_long_edge_score | global | -1 | 5692 | 0.033242 | 0.464512 | 13.973553 | 0.472316 | 13.290524 | 0.035538 |
| breakout_regime_3_prob | global | -1 | 5692 | 0.002162 | 0.001534 | 0.709621 | 0.182541 | 0.695566 | 0.262435 |
| breakout_regime_2_prob | global | -1 | 5692 | 0.378936 | 0.290646 | 0.767005 | 0.677242 | 0.320730 | 2.111566 |
| breakout_regime_1_prob | global | -1 | 5692 | 0.261059 | 0.269339 | 1.031714 | 1.038433 | 0.436180 | 2.380746 |
| breakout_regime_0_prob | global | -1 | 5692 | 0.353320 | 0.307842 | 0.871286 | 0.783256 | 0.352464 | 2.222227 |
| age_log_hours | global | -1 | 5692 | 3.245959 | 0.986612 | 0.303951 | 0.042361 | 0.304403 | 0.139162 |
| opposing_dist_to_round_pct | global | -1 | 5692 | 0.000057 | 0.000039 | 0.682189 | 0.128839 | 0.658879 | 0.195543 |
| primary_dist_to_round_pct | global | -1 | 5692 | 0.000056 | 0.000037 | 0.657278 | 0.100625 | 0.646742 | 0.155588 |
| dist_to_opposing_level_atr | global | -1 | 5692 | 3.517966 | 3.444628 | 0.979153 | 0.171219 | 0.986322 | 0.173594 |
| fakeout_ratio | global | -1 | 5692 | 0.020830 | 0.489958 | 23.521472 | 9.559360 | 21.873795 | 0.437023 |
| test_count | global | -1 | 5692 | 25.596100 | 16.012219 | 0.625573 | 0.155001 | 0.598921 | 0.258801 |
| primary_prominence_z_score | global | -1 | 5692 | -0.104456 | 0.781328 | 7.479936 | 2.066095 | 6.702469 | 0.308259 |
| opposing_age_log_hours | global | -1 | 5692 | 3.932071 | 1.208683 | 0.307391 | 0.065326 | 0.301905 | 0.216380 |
| opposing_prominence_z_score | global | -1 | 5692 | -0.020434 | 0.881747 | 43.150860 | 12.399769 | 38.393881 | 0.322962 |
| penetration_depth | global | -1 | 5692 | 0.089592 | 0.390293 | 4.356339 | 1.728980 | 4.111240 | 0.420549 |
| rubber_band_extension | global | -1 | 5692 | 0.001816 | 0.011277 | 6.209777 | 1.071621 | 6.723943 | 0.159374 |
| approach_velocity | global | -1 | 5692 | 0.002010 | 0.907172 | 451.417649 | 110.060340 | 446.485629 | 0.246504 |
| close_proximity | global | -1 | 5692 | -0.344697 | 0.254340 | 0.737865 | 0.317347 | 0.675508 | 0.469791 |
| bollinger_squeeze | global | -1 | 5692 | 0.045874 | 0.049390 | 1.076635 | 0.331128 | 1.061460 | 0.311956 |
| is_flip_candidate | global | -1 | 5692 | 0.198349 | 0.398756 | 2.010381 | 0.415398 | 2.003632 | 0.207323 |
| forward_return | global | -1 | 5686 | 0.000525 | 0.013189 | 25.138153 | 0.739972 | 26.207675 | 0.028235 |
| forward_return | regime | 0 | 2068 | 0.000129 | 0.007212 | 55.967693 | nan | nan | nan |
| forward_return | regime | 1 | 1432 | 0.001060 | 0.018544 | 17.495495 | nan | nan | nan |
| forward_return | regime | 2 | 2186 | 0.000762 | 0.015494 | 20.331524 | nan | nan | nan |
| is_flip_candidate | regime | 0 | 2068 | 0.159091 | 0.365761 | 2.299068 | nan | nan | nan |
| is_flip_candidate | regime | 1 | 1438 | 0.328929 | 0.469824 | 1.428345 | nan | nan | nan |
| is_flip_candidate | regime | 2 | 2186 | 0.149588 | 0.356667 | 2.384327 | nan | nan | nan |
| bollinger_squeeze | regime | 0 | 2068 | 0.027987 | 0.025692 | 0.917995 | nan | nan | nan |
| bollinger_squeeze | regime | 1 | 1438 | 0.064841 | 0.069396 | 1.070246 | nan | nan | nan |
| bollinger_squeeze | regime | 2 | 2186 | 0.050849 | 0.050993 | 1.002839 | nan | nan | nan |
| close_proximity | regime | 0 | 2068 | -0.487573 | 0.323067 | 0.662602 | nan | nan | nan |
| close_proximity | regime | 1 | 1438 | -0.229115 | 0.172741 | 0.753949 | nan | nan | nan |
| close_proximity | regime | 2 | 2186 | -0.297140 | 0.202728 | 0.682265 | nan | nan | nan |
| approach_velocity | regime | 0 | 2068 | 0.269863 | 0.852891 | 3.160452 | nan | nan | nan |
| approach_velocity | regime | 1 | 1438 | -0.260848 | 0.964679 | 3.698247 | nan | nan | nan |
| approach_velocity | regime | 2 | 2186 | -0.089822 | 0.874213 | 9.732704 | nan | nan | nan |
| rubber_band_extension | regime | 0 | 2068 | 0.002402 | 0.007017 | 2.920752 | nan | nan | nan |
| rubber_band_extension | regime | 1 | 1438 | -0.001297 | 0.017323 | 13.356347 | nan | nan | nan |
| rubber_band_extension | regime | 2 | 2186 | 0.003156 | 0.012293 | 3.894648 | nan | nan | nan |
| penetration_depth | regime | 0 | 2068 | -0.121142 | 0.353781 | 2.920376 | nan | nan | nan |
| penetration_depth | regime | 1 | 1438 | 0.213604 | 0.354358 | 1.658952 | nan | nan | nan |
| penetration_depth | regime | 2 | 2186 | 0.200943 | 0.396865 | 1.975007 | nan | nan | nan |
| opposing_prominence_z_score | regime | 0 | 2068 | 0.288003 | 1.099023 | 3.816010 | nan | nan | nan |
| opposing_prominence_z_score | regime | 1 | 1438 | -0.316950 | 0.543288 | 1.714113 | nan | nan | nan |
| opposing_prominence_z_score | regime | 2 | 2186 | -0.134578 | 0.711319 | 5.285531 | nan | nan | nan |
| opposing_age_log_hours | regime | 0 | 2068 | 3.785756 | 1.255347 | 0.331597 | nan | nan | nan |
| opposing_age_log_hours | regime | 1 | 1438 | 4.338925 | 1.210602 | 0.279010 | nan | nan | nan |
| opposing_age_log_hours | regime | 2 | 2186 | 3.802694 | 1.095391 | 0.288057 | nan | nan | nan |
| primary_prominence_z_score | regime | 0 | 2068 | 0.170010 | 0.958344 | 5.636995 | nan | nan | nan |
| primary_prominence_z_score | regime | 1 | 1438 | -0.329172 | 0.489930 | 1.488372 | nan | nan | nan |
| primary_prominence_z_score | regime | 2 | 2186 | -0.230279 | 0.652075 | 2.831674 | nan | nan | nan |
| test_count | regime | 0 | 2068 | 30.960348 | 17.995308 | 0.581237 | nan | nan | nan |
| test_count | regime | 1 | 1438 | 22.506954 | 13.575766 | 0.603181 | nan | nan | nan |
| test_count | regime | 2 | 2186 | 22.581885 | 14.419030 | 0.638522 | nan | nan | nan |
| fakeout_ratio | regime | 0 | 2068 | -0.250617 | 0.569835 | 2.273731 | nan | nan | nan |
| fakeout_ratio | regime | 1 | 1438 | 0.190609 | 0.374845 | 1.966562 | nan | nan | nan |
| fakeout_ratio | regime | 2 | 2186 | 0.150041 | 0.422229 | 2.814101 | nan | nan | nan |
| dist_to_opposing_level_atr | regime | 0 | 2068 | 3.678831 | 3.186206 | 0.866092 | nan | nan | nan |
| dist_to_opposing_level_atr | regime | 1 | 1438 | 4.307334 | 4.528292 | 1.051298 | nan | nan | nan |
| dist_to_opposing_level_atr | regime | 2 | 2186 | 2.837048 | 2.695044 | 0.949946 | nan | nan | nan |
| primary_dist_to_round_pct | regime | 0 | 2068 | 0.000062 | 0.000040 | 0.650258 | nan | nan | nan |
| primary_dist_to_round_pct | regime | 1 | 1438 | 0.000048 | 0.000032 | 0.671992 | nan | nan | nan |
| primary_dist_to_round_pct | regime | 2 | 2186 | 0.000055 | 0.000035 | 0.643170 | nan | nan | nan |
| opposing_dist_to_round_pct | regime | 0 | 2068 | 0.000066 | 0.000042 | 0.641964 | nan | nan | nan |
| opposing_dist_to_round_pct | regime | 1 | 1438 | 0.000048 | 0.000035 | 0.734363 | nan | nan | nan |
| opposing_dist_to_round_pct | regime | 2 | 2186 | 0.000054 | 0.000035 | 0.650075 | nan | nan | nan |
| age_log_hours | regime | 0 | 2068 | 3.094245 | 0.912201 | 0.294806 | nan | nan | nan |
| age_log_hours | regime | 1 | 1438 | 3.430862 | 1.054803 | 0.307446 | nan | nan | nan |
| age_log_hours | regime | 2 | 2186 | 3.272485 | 0.997236 | 0.304733 | nan | nan | nan |
| breakout_regime_0_prob | regime | 0 | 2068 | 0.726532 | 0.173353 | 0.238604 | nan | nan | nan |
| breakout_regime_0_prob | regime | 1 | 1438 | 0.131721 | 0.097309 | 0.738752 | nan | nan | nan |
| breakout_regime_0_prob | regime | 2 | 2186 | 0.147556 | 0.102935 | 0.697599 | nan | nan | nan |
| breakout_regime_1_prob | regime | 0 | 2068 | 0.100379 | 0.095296 | 0.949363 | nan | nan | nan |
| breakout_regime_1_prob | regime | 1 | 1438 | 0.690703 | 0.148080 | 0.214390 | nan | nan | nan |
| breakout_regime_1_prob | regime | 2 | 2186 | 0.132199 | 0.098231 | 0.743050 | nan | nan | nan |
| breakout_regime_2_prob | regime | 0 | 2068 | 0.168919 | 0.118827 | 0.703453 | nan | nan | nan |
| breakout_regime_2_prob | regime | 1 | 1438 | 0.171960 | 0.098298 | 0.571635 | nan | nan | nan |
| breakout_regime_2_prob | regime | 2 | 2186 | 0.714830 | 0.147484 | 0.206320 | nan | nan | nan |
| breakout_regime_3_prob | regime | 0 | 2068 | 0.001630 | 0.001365 | 0.837387 | nan | nan | nan |
| breakout_regime_3_prob | regime | 1 | 1438 | 0.002466 | 0.001621 | 0.657245 | nan | nan | nan |
| breakout_regime_3_prob | regime | 2 | 2186 | 0.002468 | 0.001526 | 0.618036 | nan | nan | nan |
| breakout_long_edge_score | regime | 0 | 2068 | 0.039773 | 0.550186 | 13.833097 | nan | nan | nan |
| breakout_long_edge_score | regime | 1 | 1438 | 0.007517 | 0.297632 | 39.593390 | nan | nan | nan |
| breakout_long_edge_score | regime | 2 | 2186 | 0.041783 | 0.477600 | 11.430452 | nan | nan | nan |
| breakout_short_edge_score | regime | 0 | 2068 | -0.039773 | 0.550186 | 13.833097 | nan | nan | nan |
| breakout_short_edge_score | regime | 1 | 1438 | -0.007517 | 0.297632 | 39.593390 | nan | nan | nan |
| breakout_short_edge_score | regime | 2 | 2186 | -0.041783 | 0.477600 | 11.430452 | nan | nan | nan |
| breakout_bullish_prob | regime | 0 | 2068 | 0.526186 | 0.407874 | 0.775152 | nan | nan | nan |
| breakout_bullish_prob | regime | 1 | 1438 | 0.506562 | 0.239351 | 0.472501 | nan | nan | nan |
| breakout_bullish_prob | regime | 2 | 2186 | 0.531886 | 0.376449 | 0.707762 | nan | nan | nan |
| breakout_bearish_prob | regime | 0 | 2068 | 0.472029 | 0.407882 | 0.864104 | nan | nan | nan |
| breakout_bearish_prob | regime | 1 | 1438 | 0.490783 | 0.239299 | 0.487587 | nan | nan | nan |
| breakout_bearish_prob | regime | 2 | 2186 | 0.465531 | 0.376509 | 0.808774 | nan | nan | nan |
| breakout_level_strength | regime | 0 | 2068 | 0.667099 | 0.072102 | 0.108083 | nan | nan | nan |
| breakout_level_strength | regime | 1 | 1438 | 0.624698 | 0.036631 | 0.058638 | nan | nan | nan |
| breakout_level_strength | regime | 2 | 2186 | 0.636122 | 0.049018 | 0.077057 | nan | nan | nan |

## Pairwise Winsorised CV Ratios for Breakout Regime Probabilities
Each row compares two regime groups (A,B) for a given probability metric, reporting between/within WCoV and their ratio.

| Metric | Regime A | Regime B | CV_between | CV_within | CV_between/within |
|--------|----------|----------|-----------|----------|--------------------|
| breakout_regime_0_prob | 0 | 1 | 0.616609 | 0.280582 | 2.197608 |
| breakout_regime_0_prob | 0 | 2 | 0.675219 | 0.322216 | 2.095549 |
| breakout_regime_0_prob | 1 | 2 | 0.056125 | 0.709742 | 0.079078 |
| breakout_regime_1_prob | 0 | 1 | 0.862308 | 0.355508 | 2.425567 |
| breakout_regime_1_prob | 0 | 2 | 0.136411 | 0.829622 | 0.164425 |
| breakout_regime_1_prob | 1 | 2 | 0.789926 | 0.348372 | 2.267477 |
| breakout_regime_2_prob | 0 | 1 | 0.008936 | 0.638104 | 0.014004 |
| breakout_regime_2_prob | 0 | 2 | 0.608049 | 0.296622 | 2.049909 |
| breakout_regime_2_prob | 1 | 2 | 0.543582 | 0.246104 | 2.208752 |
| breakout_regime_3_prob | 0 | 1 | 0.211985 | 0.757198 | 0.279960 |
| breakout_regime_3_prob | 0 | 2 | 0.203114 | 0.700252 | 0.290058 |
| breakout_regime_3_prob | 1 | 2 | 0.000519 | 0.637656 | 0.000814 |

## Main Feature Drivers per Factor
For each factor, the table below lists the top features by absolute Spearman correlation with that factor.


### Factor: `forward_return_h6`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| opposing_age_log_hours | 0.0457 | 0.0393 | 5686 |
| is_flip_candidate | 0.0417 | 0.0009 | 5686 |
| primary_dist_to_round_pct | 0.0391 | 0.0114 | 5686 |
| test_count | -0.0333 | -0.0286 | 5686 |
| dist_to_opposing_level_atr | 0.0207 | 0.0137 | 5686 |
| opposing_prominence_z_score | -0.0167 | -0.0309 | 5686 |
| approach_velocity | 0.0074 | 0.0163 | 5686 |
| bollinger_squeeze | 0.0056 | 0.0267 | 5686 |
| fakeout_ratio | -0.0035 | -0.0132 | 5686 |
| close_proximity | 0.0034 | 0.0089 | 5686 |
| primary_prominence_z_score | -0.0034 | -0.0285 | 5686 |
| rubber_band_extension | -0.0021 | 0.0394 | 5686 |
| opposing_dist_to_round_pct | 0.0013 | -0.0027 | 5686 |
| penetration_depth | 0.0013 | -0.0070 | 5686 |
| age_log_hours | 0.0004 | -0.0160 | 5686 |

### Factor: `breakout_long_edge_score`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| primary_dist_to_round_pct | 0.0956 | 0.0765 | 5692 |
| primary_prominence_z_score | 0.0910 | 0.1099 | 5692 |
| opposing_dist_to_round_pct | 0.0510 | 0.0539 | 5692 |
| test_count | 0.0456 | 0.0752 | 5692 |
| opposing_age_log_hours | 0.0439 | 0.0524 | 5692 |
| dist_to_opposing_level_atr | 0.0433 | 0.0124 | 5692 |
| is_flip_candidate | 0.0420 | 0.0505 | 5692 |
| opposing_prominence_z_score | 0.0408 | 0.0063 | 5692 |
| bollinger_squeeze | -0.0341 | -0.0142 | 5692 |
| rubber_band_extension | -0.0321 | 0.0029 | 5692 |
| approach_velocity | -0.0180 | -0.0234 | 5692 |
| close_proximity | -0.0103 | 0.0093 | 5692 |
| penetration_depth | 0.0053 | 0.0188 | 5692 |
| age_log_hours | -0.0012 | -0.0107 | 5692 |
| fakeout_ratio | -0.0001 | 0.0154 | 5692 |

### Factor: `breakout_short_edge_score`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| primary_dist_to_round_pct | -0.0956 | -0.0765 | 5692 |
| primary_prominence_z_score | -0.0910 | -0.1099 | 5692 |
| opposing_dist_to_round_pct | -0.0510 | -0.0539 | 5692 |
| test_count | -0.0456 | -0.0752 | 5692 |
| opposing_age_log_hours | -0.0439 | -0.0524 | 5692 |
| dist_to_opposing_level_atr | -0.0433 | -0.0124 | 5692 |
| is_flip_candidate | -0.0420 | -0.0505 | 5692 |
| opposing_prominence_z_score | -0.0408 | -0.0063 | 5692 |
| bollinger_squeeze | 0.0341 | 0.0142 | 5692 |
| rubber_band_extension | 0.0321 | -0.0029 | 5692 |
| approach_velocity | 0.0180 | 0.0234 | 5692 |
| close_proximity | 0.0103 | -0.0093 | 5692 |
| penetration_depth | -0.0053 | -0.0188 | 5692 |
| age_log_hours | 0.0012 | 0.0107 | 5692 |
| fakeout_ratio | 0.0001 | -0.0154 | 5692 |

### Factor: `breakout_bullish_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| primary_dist_to_round_pct | 0.1005 | 0.0765 | 5692 |
| primary_prominence_z_score | 0.0680 | 0.0851 | 5692 |
| opposing_dist_to_round_pct | 0.0519 | 0.0528 | 5692 |
| opposing_age_log_hours | 0.0403 | 0.0536 | 5692 |
| dist_to_opposing_level_atr | 0.0378 | 0.0121 | 5692 |
| is_flip_candidate | 0.0354 | 0.0479 | 5692 |
| opposing_prominence_z_score | 0.0289 | -0.0021 | 5692 |
| rubber_band_extension | -0.0288 | 0.0048 | 5692 |
| bollinger_squeeze | -0.0227 | -0.0089 | 5692 |
| test_count | 0.0179 | 0.0488 | 5692 |
| approach_velocity | -0.0123 | -0.0205 | 5692 |
| close_proximity | -0.0087 | 0.0116 | 5692 |
| penetration_depth | 0.0038 | 0.0193 | 5692 |
| fakeout_ratio | -0.0027 | 0.0163 | 5692 |
| age_log_hours | 0.0018 | -0.0116 | 5692 |

### Factor: `breakout_bearish_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| primary_dist_to_round_pct | -0.1005 | -0.0760 | 5692 |
| primary_prominence_z_score | -0.0673 | -0.0843 | 5692 |
| opposing_dist_to_round_pct | -0.0523 | -0.0525 | 5692 |
| opposing_age_log_hours | -0.0408 | -0.0544 | 5692 |
| dist_to_opposing_level_atr | -0.0378 | -0.0123 | 5692 |
| is_flip_candidate | -0.0356 | -0.0485 | 5692 |
| opposing_prominence_z_score | -0.0288 | 0.0028 | 5692 |
| rubber_band_extension | 0.0286 | -0.0044 | 5692 |
| bollinger_squeeze | 0.0202 | 0.0074 | 5692 |
| test_count | -0.0162 | -0.0474 | 5692 |
| approach_velocity | 0.0152 | 0.0223 | 5692 |
| close_proximity | 0.0090 | -0.0125 | 5692 |
| age_log_hours | -0.0043 | 0.0095 | 5692 |
| penetration_depth | -0.0035 | -0.0198 | 5692 |
| fakeout_ratio | 0.0029 | -0.0170 | 5692 |

### Factor: `breakout_regime_0_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| penetration_depth | -0.4356 | -0.4171 | 5692 |
| fakeout_ratio | -0.4101 | -0.3927 | 5692 |
| close_proximity | -0.4053 | -0.4684 | 5692 |
| bollinger_squeeze | -0.3804 | -0.2454 | 5692 |
| approach_velocity | 0.2783 | 0.1576 | 5692 |
| test_count | 0.2568 | 0.2821 | 5692 |
| primary_prominence_z_score | 0.2515 | 0.2989 | 5692 |
| opposing_prominence_z_score | 0.2196 | 0.2967 | 5692 |
| opposing_dist_to_round_pct | 0.1890 | 0.2007 | 5692 |
| dist_to_opposing_level_atr | 0.1729 | 0.0310 | 5692 |
| age_log_hours | -0.1611 | -0.1353 | 5692 |
| primary_dist_to_round_pct | 0.1226 | 0.1483 | 5692 |
| opposing_age_log_hours | -0.1194 | -0.1120 | 5692 |
| rubber_band_extension | 0.1100 | 0.0256 | 5692 |
| is_flip_candidate | -0.1064 | -0.1018 | 5692 |

### Factor: `breakout_regime_1_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| close_proximity | 0.4199 | 0.3243 | 5692 |
| penetration_depth | 0.3446 | 0.2207 | 5692 |
| fakeout_ratio | 0.3414 | 0.2173 | 5692 |
| bollinger_squeeze | 0.3008 | 0.1955 | 5692 |
| opposing_prominence_z_score | -0.2593 | -0.2261 | 5692 |
| is_flip_candidate | 0.2455 | 0.2462 | 5692 |
| rubber_band_extension | -0.2310 | -0.1053 | 5692 |
| opposing_age_log_hours | 0.2146 | 0.2256 | 5692 |
| primary_prominence_z_score | -0.2058 | -0.1738 | 5692 |
| approach_velocity | -0.1756 | -0.1283 | 5692 |
| opposing_dist_to_round_pct | -0.1735 | -0.1496 | 5692 |
| primary_dist_to_round_pct | -0.1708 | -0.1387 | 5692 |
| age_log_hours | 0.1499 | 0.1270 | 5692 |
| test_count | -0.1415 | -0.1380 | 5692 |
| dist_to_opposing_level_atr | -0.0168 | 0.1274 | 5692 |

### Factor: `breakout_regime_2_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| penetration_depth | 0.2634 | 0.2343 | 5692 |
| fakeout_ratio | 0.2467 | 0.2114 | 5692 |
| dist_to_opposing_level_atr | -0.2344 | -0.1522 | 5692 |
| bollinger_squeeze | 0.2149 | 0.0750 | 5692 |
| test_count | -0.1928 | -0.1677 | 5692 |
| close_proximity | 0.1584 | 0.1913 | 5692 |
| primary_prominence_z_score | -0.1419 | -0.1527 | 5692 |
| is_flip_candidate | -0.1329 | -0.1232 | 5692 |
| rubber_band_extension | 0.1158 | 0.0717 | 5692 |
| approach_velocity | -0.1124 | -0.0446 | 5692 |
| opposing_age_log_hours | -0.0933 | -0.0933 | 5692 |
| opposing_prominence_z_score | -0.0739 | -0.1018 | 5692 |
| opposing_dist_to_round_pct | -0.0657 | -0.0722 | 5692 |
| age_log_hours | 0.0361 | 0.0219 | 5692 |
| primary_dist_to_round_pct | -0.0296 | -0.0266 | 5692 |

### Factor: `breakout_regime_3_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| bollinger_squeeze | 0.4514 | 0.2583 | 5692 |
| approach_velocity | -0.4480 | -0.3043 | 5692 |
| age_log_hours | 0.3430 | 0.3537 | 5692 |
| test_count | -0.3018 | -0.2364 | 5692 |
| fakeout_ratio | 0.1890 | 0.1145 | 5692 |
| primary_prominence_z_score | -0.1813 | -0.1344 | 5692 |
| penetration_depth | 0.1597 | 0.0881 | 5692 |
| opposing_age_log_hours | 0.1469 | 0.1336 | 5692 |
| close_proximity | 0.1437 | 0.1503 | 5692 |
| opposing_prominence_z_score | -0.1412 | -0.1056 | 5692 |
| is_flip_candidate | 0.1055 | 0.1010 | 5692 |
| rubber_band_extension | -0.1006 | -0.0603 | 5692 |
| primary_dist_to_round_pct | -0.0723 | -0.0929 | 5692 |
| opposing_dist_to_round_pct | -0.0401 | -0.0408 | 5692 |
| dist_to_opposing_level_atr | -0.0265 | 0.0377 | 5692 |