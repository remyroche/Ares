# Breakout/Bounce Regime Diagnostics

- Symbol: **ETHUSDT**
- Exchange: **binance**
- Timeframe: **1h**
- Direction: **long**
- Horizon (bars): **6**
- Samples (training window): **1458**

## Global Model Metrics
- Validation log loss: **nan**
- Precision (breakout class 1): **0.2667**

### Class Counts (training labels)
| Regime | Count |
|--------|-------|
| 0 | 458 |
| 1 | 431 |
| 2 | 569 |

## Forward Return Sharpe-like Ratios
| Scope | Regime | Mean Return | Std Return | Sharpe-like |
|-------|--------|-------------|------------|-------------|
| global | -1 | 0.001962 | 0.047145 | 0.0416 |
| regime | 0 | 0.002778 | 0.023818 | 0.1166 |
| regime | 1 | 0.003237 | 0.064829 | 0.0499 |
| regime | 2 | 0.000436 | 0.040110 | 0.0109 |

## Winsorised CV Between/Within Regimes
Definition: per metric, we winsorise values (5–95%), compute global and per-regime CV, and report the ratio of between-regime CV to within-regime CV.

| Metric | Scope | Regime | Count | Mean | Std | CV | CV_between | CV_within | CV_between/within |
|--------|-------|--------|-------|------|-----|----|-----------|----------|--------------------|
| breakout_regime_2_prob | global | -1 | 1458 | 0.417217 | 0.355426 | 0.851897 | 0.766455 | 0.275978 | 2.777233 |
| breakout_regime_1_prob | global | -1 | 1458 | 0.336548 | 0.340532 | 1.011838 | 0.949917 | 0.331141 | 2.868619 |
| breakout_regime_0_prob | global | -1 | 1458 | 0.242840 | 0.282929 | 1.165084 | 1.195851 | 0.459392 | 2.603117 |
| momentum_divergence | global | -1 | 1458 | 0.202419 | 32.548995 | 160.799713 | 8.810584 | 163.065264 | 0.054031 |
| close_proximity | global | -1 | 1458 | -0.242382 | 0.165196 | 0.681555 | 0.165349 | 0.693042 | 0.238584 |
| opposing_age_log_hours | global | -1 | 1458 | 4.039971 | 1.258412 | 0.311490 | 0.063424 | 0.308488 | 0.205596 |
| opposing_volume_depth_ratio | global | -1 | 1458 | 1.474315 | 0.585980 | 0.397459 | 0.026816 | 0.428842 | 0.062530 |
| approach_velocity | global | -1 | 1458 | 0.064257 | 0.958642 | 14.918922 | 3.622930 | 14.881347 | 0.243454 |
| age_log_hours | global | -1 | 1458 | 3.331900 | 0.957420 | 0.287349 | 0.031951 | 0.279657 | 0.114249 |
| primary_volume_depth_ratio | global | -1 | 1458 | 1.442402 | 0.494338 | 0.342719 | 0.046253 | 0.333626 | 0.138638 |
| primary_dist_to_round_pct | global | -1 | 1458 | 0.000042 | 0.000027 | 0.646012 | 0.042068 | 0.642201 | 0.065506 |
| primary_prominence_z_score | global | -1 | 1458 | -0.019128 | 0.911747 | 47.664822 | 5.214117 | 48.090500 | 0.108423 |
| test_count | global | -1 | 1458 | 19.331962 | 11.574454 | 0.598721 | 0.050548 | 0.590686 | 0.085575 |
| opposing_dist_to_round_pct | global | -1 | 1458 | 0.000044 | 0.000031 | 0.693972 | 0.044706 | 0.691845 | 0.064619 |
| opposing_prominence_z_score | global | -1 | 1458 | 0.000787 | 0.920410 | 1169.429857 | 181.748962 | 1168.366552 | 0.155558 |
| rubber_band_extension | global | -1 | 1458 | 0.003291 | 0.016593 | 5.042666 | 0.765970 | 5.003216 | 0.153096 |
| dist_to_opposing_level_atr | global | -1 | 1458 | 3.822659 | 4.368371 | 1.142757 | 0.235427 | 1.099638 | 0.214095 |
| is_flip_candidate | global | -1 | 1458 | 0.206447 | 0.404755 | 1.960575 | 0.417111 | 1.902460 | 0.219248 |
| forward_return | global | -1 | 1452 | 0.000757 | 0.020715 | 27.382771 | 0.606083 | 26.271856 | 0.023070 |
| forward_return | regime | 0 | 362 | 0.001439 | 0.008919 | 6.199994 | nan | nan | nan |
| forward_return | regime | 1 | 488 | 0.000737 | 0.028825 | 39.098259 | nan | nan | nan |
| forward_return | regime | 2 | 602 | 0.000328 | 0.021880 | 66.657792 | nan | nan | nan |
| is_flip_candidate | regime | 0 | 362 | 0.162983 | 0.369351 | 2.266185 | nan | nan | nan |
| is_flip_candidate | regime | 1 | 493 | 0.328600 | 0.469704 | 1.429409 | nan | nan | nan |
| is_flip_candidate | regime | 2 | 603 | 0.132670 | 0.339218 | 2.556854 | nan | nan | nan |
| dist_to_opposing_level_atr | regime | 0 | 362 | 3.956636 | 4.624767 | 1.168863 | nan | nan | nan |
| dist_to_opposing_level_atr | regime | 1 | 493 | 4.929324 | 5.057624 | 1.026028 | nan | nan | nan |
| dist_to_opposing_level_atr | regime | 2 | 603 | 2.729779 | 2.928233 | 1.072700 | nan | nan | nan |
| rubber_band_extension | regime | 0 | 362 | 0.004940 | 0.011953 | 2.419829 | nan | nan | nan |
| rubber_band_extension | regime | 1 | 493 | -0.000590 | 0.022277 | 37.787503 | nan | nan | nan |
| rubber_band_extension | regime | 2 | 603 | 0.004554 | 0.015161 | 3.329124 | nan | nan | nan |
| opposing_prominence_z_score | regime | 0 | 362 | 0.142262 | 1.015259 | 7.136537 | nan | nan | nan |
| opposing_prominence_z_score | regime | 1 | 493 | -0.189323 | 0.835577 | 4.413497 | nan | nan | nan |
| opposing_prominence_z_score | regime | 2 | 603 | 0.074549 | 0.907884 | 12.178322 | nan | nan | nan |
| opposing_dist_to_round_pct | regime | 0 | 362 | 0.000048 | 0.000030 | 0.637372 | nan | nan | nan |
| opposing_dist_to_round_pct | regime | 1 | 493 | 0.000043 | 0.000031 | 0.725389 | nan | nan | nan |
| opposing_dist_to_round_pct | regime | 2 | 603 | 0.000043 | 0.000030 | 0.702404 | nan | nan | nan |
| test_count | regime | 0 | 362 | 20.837017 | 11.901690 | 0.571180 | nan | nan | nan |
| test_count | regime | 1 | 493 | 18.636917 | 11.419709 | 0.612747 | nan | nan | nan |
| test_count | regime | 2 | 603 | 18.920398 | 10.935980 | 0.577999 | nan | nan | nan |
| primary_prominence_z_score | regime | 0 | 362 | 0.125896 | 0.993678 | 7.892869 | nan | nan | nan |
| primary_prominence_z_score | regime | 1 | 493 | -0.117178 | 0.847176 | 7.229816 | nan | nan | nan |
| primary_prominence_z_score | regime | 2 | 603 | -0.016858 | 0.918814 | 54.504089 | nan | nan | nan |
| primary_dist_to_round_pct | regime | 0 | 362 | 0.000040 | 0.000024 | 0.599621 | nan | nan | nan |
| primary_dist_to_round_pct | regime | 1 | 493 | 0.000042 | 0.000028 | 0.659403 | nan | nan | nan |
| primary_dist_to_round_pct | regime | 2 | 603 | 0.000044 | 0.000030 | 0.671509 | nan | nan | nan |
| primary_volume_depth_ratio | regime | 0 | 362 | 1.335161 | 0.435054 | 0.325844 | nan | nan | nan |
| primary_volume_depth_ratio | regime | 1 | 493 | 1.484335 | 0.501085 | 0.337582 | nan | nan | nan |
| primary_volume_depth_ratio | regime | 2 | 603 | 1.467540 | 0.507529 | 0.345836 | nan | nan | nan |
| age_log_hours | regime | 0 | 362 | 3.181073 | 0.813950 | 0.255873 | nan | nan | nan |
| age_log_hours | regime | 1 | 493 | 3.440358 | 1.040853 | 0.302542 | nan | nan | nan |
| age_log_hours | regime | 2 | 603 | 3.334724 | 0.940570 | 0.282053 | nan | nan | nan |
| approach_velocity | regime | 0 | 362 | 0.394919 | 0.985313 | 2.494975 | nan | nan | nan |
| approach_velocity | regime | 1 | 493 | -0.163549 | 1.066943 | 6.523692 | nan | nan | nan |
| approach_velocity | regime | 2 | 603 | 0.015876 | 0.816427 | 51.425700 | nan | nan | nan |
| opposing_volume_depth_ratio | regime | 0 | 362 | 1.486786 | 0.677150 | 0.455445 | nan | nan | nan |
| opposing_volume_depth_ratio | regime | 1 | 493 | 1.539677 | 0.743687 | 0.483015 | nan | nan | nan |
| opposing_volume_depth_ratio | regime | 2 | 603 | 1.442980 | 0.475909 | 0.329810 | nan | nan | nan |
| opposing_age_log_hours | regime | 0 | 362 | 3.895184 | 1.301482 | 0.334126 | nan | nan | nan |
| opposing_age_log_hours | regime | 1 | 493 | 4.406250 | 1.313875 | 0.298185 | nan | nan | nan |
| opposing_age_log_hours | regime | 2 | 603 | 3.835200 | 1.123486 | 0.292941 | nan | nan | nan |
| close_proximity | regime | 0 | 362 | -0.307344 | 0.202206 | 0.657913 | nan | nan | nan |
| close_proximity | regime | 1 | 493 | -0.228829 | 0.164089 | 0.717084 | nan | nan | nan |
| close_proximity | regime | 2 | 603 | -0.217053 | 0.137647 | 0.634162 | nan | nan | nan |
| momentum_divergence | regime | 0 | 362 | -1.839367 | 33.426618 | 18.172890 | nan | nan | nan |
| momentum_divergence | regime | 1 | 493 | -0.389124 | 36.749513 | 94.441741 | nan | nan | nan |
| momentum_divergence | regime | 2 | 603 | 2.454432 | 28.846629 | 11.752873 | nan | nan | nan |
| breakout_regime_0_prob | regime | 0 | 362 | 0.707134 | 0.151840 | 0.214726 | nan | nan | nan |
| breakout_regime_0_prob | regime | 1 | 493 | 0.094176 | 0.091880 | 0.975622 | nan | nan | nan |
| breakout_regime_0_prob | regime | 2 | 603 | 0.088074 | 0.090956 | 1.032720 | nan | nan | nan |
| breakout_regime_1_prob | regime | 0 | 362 | 0.129047 | 0.108223 | 0.838629 | nan | nan | nan |
| breakout_regime_1_prob | regime | 1 | 493 | 0.787142 | 0.145227 | 0.184500 | nan | nan | nan |
| breakout_regime_1_prob | regime | 2 | 603 | 0.090538 | 0.080883 | 0.893368 | nan | nan | nan |
| breakout_regime_2_prob | regime | 0 | 362 | 0.162168 | 0.109156 | 0.673104 | nan | nan | nan |
| breakout_regime_2_prob | regime | 1 | 493 | 0.115647 | 0.090138 | 0.779422 | nan | nan | nan |
| breakout_regime_2_prob | regime | 2 | 603 | 0.816062 | 0.146134 | 0.179072 | nan | nan | nan |

## Pairwise Winsorised CV Ratios for Breakout Regime Probabilities
Each row compares two regime groups (A,B) for a given probability metric, reporting between/within WCoV and their ratio.

| Metric | Regime A | Regime B | CV_between | CV_within | CV_between/within |
|--------|----------|----------|-----------|----------|--------------------|
| breakout_regime_0_prob | 0 | 1 | 0.869774 | 0.345833 | 2.515008 |
| breakout_regime_0_prob | 0 | 2 | 0.969511 | 0.380243 | 2.549716 |
| breakout_regime_0_prob | 1 | 2 | 0.033605 | 1.006929 | 0.033374 |
| breakout_regime_1_prob | 0 | 1 | 0.647713 | 0.249452 | 2.596539 |
| breakout_regime_1_prob | 0 | 2 | 0.182464 | 0.896008 | 0.203641 |
| breakout_regime_1_prob | 1 | 2 | 0.860594 | 0.279341 | 3.080807 |
| breakout_regime_2_prob | 0 | 1 | 0.171732 | 0.735698 | 0.233428 |
| breakout_regime_2_prob | 0 | 2 | 0.573098 | 0.223746 | 2.561376 |
| breakout_regime_2_prob | 1 | 2 | 0.698639 | 0.235673 | 2.964440 |

## Main Feature Drivers per Factor
For each factor, the table below lists the top features by absolute Spearman correlation with that factor.


### Factor: `forward_return_h6`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| momentum_divergence | -0.1394 | -0.0416 | 1452 |
| primary_volume_depth_ratio | -0.0749 | 0.0273 | 1452 |
| age_log_hours | -0.0666 | -0.0972 | 1452 |
| primary_dist_to_round_pct | 0.0660 | 0.0756 | 1452 |
| opposing_volume_depth_ratio | 0.0599 | 0.0508 | 1452 |
| close_proximity | -0.0595 | -0.0493 | 1452 |
| opposing_prominence_z_score | -0.0491 | -0.0634 | 1452 |
| opposing_age_log_hours | 0.0338 | 0.0334 | 1452 |
| dist_to_opposing_level_atr | 0.0338 | 0.0316 | 1452 |
| approach_velocity | -0.0297 | -0.0219 | 1452 |
| is_flip_candidate | 0.0205 | -0.0001 | 1452 |
| opposing_dist_to_round_pct | -0.0162 | 0.0428 | 1452 |
| test_count | -0.0089 | -0.0133 | 1452 |
| primary_prominence_z_score | 0.0023 | 0.0045 | 1452 |
| rubber_band_extension | 0.0016 | 0.0113 | 1452 |

### Factor: `breakout_regime_0_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| close_proximity | -0.2123 | -0.2647 | 1458 |
| approach_velocity | 0.1821 | 0.1566 | 1458 |
| primary_volume_depth_ratio | -0.1725 | -0.1410 | 1458 |
| age_log_hours | -0.1460 | -0.0995 | 1458 |
| opposing_volume_depth_ratio | -0.1087 | -0.0346 | 1458 |
| primary_dist_to_round_pct | -0.0923 | -0.0587 | 1458 |
| test_count | 0.0853 | 0.0900 | 1458 |
| primary_prominence_z_score | 0.0804 | 0.1004 | 1458 |
| rubber_band_extension | 0.0571 | 0.0774 | 1458 |
| opposing_age_log_hours | -0.0384 | -0.0698 | 1458 |
| momentum_divergence | -0.0377 | -0.0316 | 1458 |
| dist_to_opposing_level_atr | 0.0327 | 0.0315 | 1458 |
| opposing_dist_to_round_pct | 0.0289 | 0.0446 | 1458 |
| opposing_prominence_z_score | 0.0161 | 0.1034 | 1458 |
| is_flip_candidate | -0.0127 | -0.0607 | 1458 |

### Factor: `breakout_regime_1_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| is_flip_candidate | 0.2745 | 0.2457 | 1458 |
| rubber_band_extension | -0.2561 | -0.1100 | 1458 |
| dist_to_opposing_level_atr | 0.2481 | 0.1620 | 1458 |
| opposing_prominence_z_score | -0.2169 | -0.1714 | 1458 |
| opposing_age_log_hours | 0.2040 | 0.2115 | 1458 |
| approach_velocity | -0.1699 | -0.1128 | 1458 |
| close_proximity | 0.0984 | 0.0814 | 1458 |
| opposing_dist_to_round_pct | -0.0950 | -0.0225 | 1458 |
| primary_prominence_z_score | -0.0804 | -0.1000 | 1458 |
| primary_volume_depth_ratio | 0.0748 | 0.0945 | 1458 |
| test_count | -0.0543 | -0.0636 | 1458 |
| age_log_hours | 0.0499 | 0.0773 | 1458 |
| opposing_volume_depth_ratio | 0.0270 | 0.0623 | 1458 |
| primary_dist_to_round_pct | -0.0252 | -0.0065 | 1458 |
| momentum_divergence | -0.0248 | 0.0184 | 1458 |

### Factor: `breakout_regime_2_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| dist_to_opposing_level_atr | -0.2517 | -0.1810 | 1458 |
| is_flip_candidate | -0.2048 | -0.1867 | 1458 |
| opposing_age_log_hours | -0.1491 | -0.1465 | 1458 |
| opposing_prominence_z_score | 0.1220 | 0.0808 | 1458 |
| rubber_band_extension | 0.1071 | 0.0429 | 1458 |
| close_proximity | 0.0900 | 0.1362 | 1458 |
| primary_volume_depth_ratio | 0.0507 | 0.0235 | 1458 |
| momentum_divergence | 0.0451 | 0.0079 | 1458 |
| opposing_volume_depth_ratio | 0.0316 | -0.0318 | 1458 |
| opposing_dist_to_round_pct | -0.0249 | -0.0145 | 1458 |
| age_log_hours | 0.0184 | 0.0064 | 1458 |
| primary_dist_to_round_pct | 0.0179 | 0.0538 | 1458 |
| test_count | -0.0041 | -0.0119 | 1458 |
| primary_prominence_z_score | 0.0008 | 0.0147 | 1458 |
| approach_velocity | 0.0006 | -0.0185 | 1458 |