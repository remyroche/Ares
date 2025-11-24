# Breakout/Bounce Regime Diagnostics

- Symbol: **ETHUSDT**
- Exchange: **binance**
- Timeframe: **1h**
- Direction: **long**
- Horizon (bars): **96**
- Samples (training window): **5630**

## Global Model Metrics
- Validation log loss: **nan**
- Test log loss: **nan**
- Generalization gap (test - val log loss): **nan**
- Macro ROC AUC (OvR, val): **0.5299**
- Macro ROC AUC (OvR, test): **0.5153**
- Macro F1-score (val): **0.3401**
- Macro F1-score (test): **0.3124**
- Generalization gap (Macro F1 test - val): **-0.0277**
- Weighted F1-score (val): **0.4887**
- Weighted F1-score (test): **0.4042**
- Precision (breakout class 1, val): **0.5134**
- Sample split: train=3940, val=845, test=845

### Class Counts (training labels)
| Regime | Count |
|--------|-------|
| 0 | 442 |
| 1 | 2528 |
| 2 | 2660 |

### Resistance/Support Hit Counts by Outcome
| Side | Class 1 | Class 2 | Total |
|--------|--------|--------|--------|
| resistance | 1137 | 1515 | 2652 |
| support | 1144 | 1834 | 2978 |

## Meta-Label Success Summary
- Meta-labeled events: **5630**, success=1: **2079** (36.927% success rate)

| Class | Meta Events | Success Count | Success Rate |
|-------|------------|---------------|--------------|
| 0 | 442 | 138 | 31.222% |
| 1 | 2528 | 874 | 34.573% |
| 2 | 2660 | 1067 | 40.113% |

## Breakout Success Probability & High-Confidence Gating
- Observations with breakout_success_prob: **5630** | mean=1.000, std=0.000, p25=1.000, median=1.000, p75=1.000
- High-confidence signals (high_conf=1): **5630** / 5630 (100.000%)

### Forward Return Sharpe by Meta/High-Confidence Subset
| Subset | Samples | Mean Return | Std Return | Sharpe-like |
|--------|---------|-------------|------------|-------------|
| all | 5534 | 0.030688 | 0.140231 | 0.2188 |
| meta_success==1 | 2043 | 0.107937 | 0.141235 | 0.7642 |
| high_conf==1 | 5534 | 0.030688 | 0.140231 | 0.2188 |

## Forward Return Sharpe-like Ratios
| Scope | Side | Regime | Mean Return | Std Return | Sharpe-like |
|-------|------|--------|-------------|------------|-------------|
| global | all | -1 | 0.030688 | 0.140231 | 0.2188 |
| regime | all | 1 | 0.025458 | 0.142580 | 0.1786 |
| regime | all | 2 | 0.034132 | 0.138554 | 0.2463 |
| global | resistance | -1 | 0.034193 | 0.139660 | 0.2448 |
| regime | resistance | 1 | 0.050868 | 0.136937 | 0.3715 |
| regime | resistance | 2 | 0.022041 | 0.140369 | 0.1570 |
| global | support | -1 | 0.027567 | 0.140663 | 0.1960 |
| regime | support | 1 | 0.000024 | 0.143578 | 0.0002 |
| regime | support | 2 | 0.044101 | 0.136235 | 0.3237 |

## Per-Regime Summary (Forward Returns & Edge Scores)
| Regime | Side | Count | Mean Forward Return | Sharpe-like | Mean Long Edge | Mean Short Edge | Mean Bullish Prob | Mean Bearish Prob |
|--------|------|-------|---------------------|------------|---------------|----------------|-------------------|--------------------|
| 1 | all | 2281 | 0.025458 | 0.1786 | 0.005680 | -0.005680 | 0.505329 | 0.494671 |
| 1 | resistance | 1137 | 0.050868 | 0.3715 | 0.037683 | -0.037683 | 0.530055 | 0.469945 |
| 1 | support | 1144 | 0.000024 | 0.0002 | -0.026128 | 0.026128 | 0.480755 | 0.519245 |
| 2 | all | 3349 | 0.034132 | 0.2463 | 0.013564 | -0.013564 | 0.510246 | 0.489754 |
| 2 | resistance | 1515 | 0.022041 | 0.1570 | -0.139867 | 0.139867 | 0.391664 | 0.608336 |
| 2 | support | 1834 | 0.044101 | 0.3237 | 0.140308 | -0.140308 | 0.608201 | 0.391799 |

## Winsorised CV Between/Within Regimes
Definition: per metric, we winsorise values (5–95%), compute global and per-regime CV, and report the ratio of between-regime CV to within-regime CV.

| Metric | Scope | Regime | Count | Mean | Std | CV | CV_between | CV_within | CV_between/within |
|--------|-------|--------|-------|------|-----|----|-----------|----------|--------------------|
| breakout_level_strength | global | -1 | 5630 | 0.646075 | 0.059164 | 0.091575 | 0.009666 | 0.093829 | 0.103011 |
| breakout_bearish_prob | global | -1 | 5630 | 0.491213 | 0.094882 | 0.193158 | 0.005002 | 0.171685 | 0.029136 |
| breakout_bullish_prob | global | -1 | 5630 | 0.508787 | 0.094882 | 0.186486 | 0.004829 | 0.165754 | 0.029136 |
| breakout_short_edge_score | global | -1 | 5630 | -0.011076 | 0.124083 | 11.202950 | 0.338343 | 9.979125 | 0.033905 |
| breakout_long_edge_score | global | -1 | 5630 | 0.011076 | 0.124083 | 11.202950 | 0.338343 | 9.979125 | 0.033905 |
| breakout_regime_2_prob | global | -1 | 5630 | 0.482215 | 0.077942 | 0.161634 | 0.134531 | 0.109033 | 1.233860 |
| breakout_regime_1_prob | global | -1 | 5630 | 0.444794 | 0.077737 | 0.174772 | 0.147413 | 0.118790 | 1.240952 |
| breakout_regime_0_prob | global | -1 | 5630 | 0.069689 | 0.023972 | 0.343987 | 0.007125 | 0.353828 | 0.020138 |
| primary_prominence_z_score | global | -1 | 5630 | -0.101957 | 0.788940 | 7.737971 | 0.831136 | 7.801648 | 0.106533 |
| penetration_depth | global | -1 | 5630 | 0.099344 | 0.376647 | 3.791340 | 0.225220 | 3.797805 | 0.059303 |
| primary_dist_to_round_pct | global | -1 | 5630 | 0.000055 | 0.000036 | 0.656156 | 0.044967 | 0.657162 | 0.068427 |
| int_dist_opp_trend | global | -1 | 5630 | 62.012984 | 66.200057 | 1.067519 | 0.104484 | 1.113489 | 0.093835 |
| primary_volume_depth_ratio | global | -1 | 5630 | 1.458701 | 0.410670 | 0.281531 | 0.017683 | 0.280575 | 0.063025 |
| trend_strength_adx | global | -1 | 5630 | 20.209757 | 10.882229 | 0.538464 | 0.061384 | 0.535037 | 0.114728 |
| int_primary_prom_squeeze | global | -1 | 5630 | -0.053703 | 0.092504 | 1.722513 | 0.199059 | 1.656129 | 0.120195 |
| int_opposing_prom_squeeze | global | -1 | 5630 | -0.043589 | 0.093495 | 2.144935 | 0.053374 | 2.063290 | 0.025869 |
| opposing_dist_to_round_pct | global | -1 | 5630 | 0.000057 | 0.000039 | 0.681558 | 0.089804 | 0.661784 | 0.135700 |
| momentum_divergence | global | -1 | 5630 | 2.365355 | 68.290518 | 28.871146 | 2.285669 | 30.595072 | 0.074707 |
| opposing_prominence_z_score | global | -1 | 5630 | -0.019022 | 0.892441 | 46.917466 | 5.500789 | 47.008884 | 0.117016 |
| dist_to_opposing_level_atr | global | -1 | 5630 | 3.210167 | 3.008249 | 0.937100 | 0.216485 | 0.970122 | 0.223153 |
| opposing_volume_depth_ratio | global | -1 | 5630 | 1.470897 | 0.461308 | 0.313624 | 0.005148 | 0.317735 | 0.016203 |
| opposing_age_log_hours | global | -1 | 5630 | 3.929225 | 1.213634 | 0.308874 | 0.018009 | 0.312560 | 0.057619 |
| is_flip_candidate | global | -1 | 5630 | 0.200178 | 0.400133 | 1.998891 | 0.236358 | 2.010678 | 0.117551 |
| forward_return_support | global | -1 | 2927 | 0.023505 | 0.115813 | 4.927188 | 0.940752 | 4.892121 | 0.192299 |
| forward_return_resistance | global | -1 | 2607 | 0.031286 | 0.119737 | 3.827147 | 0.496801 | 3.775646 | 0.131580 |
| forward_return | global | -1 | 5534 | 0.027063 | 0.117365 | 4.336733 | 0.167569 | 4.372024 | 0.038328 |
| forward_return | regime | 1 | 2197 | 0.020941 | 0.119330 | 5.698294 | nan | nan | nan |
| forward_return | regime | 2 | 3337 | 0.030011 | 0.117309 | 3.908855 | nan | nan | nan |
| forward_return_resistance | regime | 1 | 1099 | 0.049024 | 0.116838 | 2.383289 | nan | nan | nan |
| forward_return_resistance | regime | 2 | 1508 | 0.017938 | 0.119413 | 6.656960 | nan | nan | nan |
| forward_return_support | regime | 1 | 1098 | -0.004721 | 0.116112 | 24.592890 | nan | nan | nan |
| forward_return_support | regime | 2 | 1829 | 0.039503 | 0.113866 | 2.882441 | nan | nan | nan |
| is_flip_candidate | regime | 1 | 2281 | 0.256466 | 0.436682 | 1.702688 | nan | nan | nan |
| is_flip_candidate | regime | 2 | 3349 | 0.161839 | 0.368303 | 2.275734 | nan | nan | nan |
| opposing_age_log_hours | regime | 1 | 2281 | 3.839176 | 1.483755 | 0.386477 | nan | nan | nan |
| opposing_age_log_hours | regime | 2 | 3349 | 3.980702 | 0.972486 | 0.244300 | nan | nan | nan |
| opposing_volume_depth_ratio | regime | 1 | 2281 | 1.479929 | 0.471804 | 0.318802 | nan | nan | nan |
| opposing_volume_depth_ratio | regime | 2 | 3349 | 1.464784 | 0.462907 | 0.316024 | nan | nan | nan |
| dist_to_opposing_level_atr | regime | 1 | 2281 | 4.065066 | 4.213663 | 1.036555 | nan | nan | nan |
| dist_to_opposing_level_atr | regime | 2 | 3349 | 2.675157 | 2.014843 | 0.753168 | nan | nan | nan |
| opposing_prominence_z_score | regime | 1 | 2281 | -0.141908 | 0.911856 | 6.425661 | nan | nan | nan |
| opposing_prominence_z_score | regime | 2 | 3349 | 0.067358 | 0.876504 | 13.012590 | nan | nan | nan |
| momentum_divergence | regime | 1 | 2281 | -5.179842 | 79.488001 | 15.345643 | nan | nan | nan |
| momentum_divergence | regime | 2 | 3349 | 5.632996 | 65.248429 | 11.583254 | nan | nan | nan |
| opposing_dist_to_round_pct | regime | 1 | 2281 | 0.000051 | 0.000034 | 0.669548 | nan | nan | nan |
| opposing_dist_to_round_pct | regime | 2 | 3349 | 0.000061 | 0.000041 | 0.676426 | nan | nan | nan |
| int_opposing_prom_squeeze | regime | 1 | 2281 | -0.045830 | 0.075001 | 1.636505 | nan | nan | nan |
| int_opposing_prom_squeeze | regime | 2 | 3349 | -0.041177 | 0.104871 | 2.546831 | nan | nan | nan |
| int_primary_prom_squeeze | regime | 1 | 2281 | -0.040781 | 0.080757 | 1.980248 | nan | nan | nan |
| int_primary_prom_squeeze | regime | 2 | 3349 | -0.062162 | 0.097120 | 1.562387 | nan | nan | nan |
| trend_strength_adx | regime | 1 | 2281 | 18.738403 | 10.829245 | 0.577917 | nan | nan | nan |
| trend_strength_adx | regime | 2 | 3349 | 21.219500 | 10.796673 | 0.508809 | nan | nan | nan |
| primary_volume_depth_ratio | regime | 1 | 2281 | 1.428492 | 0.391540 | 0.274093 | nan | nan | nan |
| primary_volume_depth_ratio | regime | 2 | 3349 | 1.480081 | 0.427010 | 0.288504 | nan | nan | nan |
| int_dist_opp_trend | regime | 1 | 2281 | 70.081269 | 83.061518 | 1.185217 | nan | nan | nan |
| int_dist_opp_trend | regime | 2 | 3349 | 57.122499 | 55.039978 | 0.963543 | nan | nan | nan |
| primary_dist_to_round_pct | regime | 1 | 2281 | 0.000052 | 0.000034 | 0.650940 | nan | nan | nan |
| primary_dist_to_round_pct | regime | 2 | 3349 | 0.000057 | 0.000038 | 0.668181 | nan | nan | nan |
| penetration_depth | regime | 1 | 2281 | 0.125767 | 0.385215 | 3.062923 | nan | nan | nan |
| penetration_depth | regime | 2 | 3349 | 0.081019 | 0.369364 | 4.559005 | nan | nan | nan |
| primary_prominence_z_score | regime | 1 | 2281 | -0.005364 | 0.936085 | 174.500473 | nan | nan | nan |
| primary_prominence_z_score | regime | 2 | 3349 | -0.174844 | 0.654779 | 3.744921 | nan | nan | nan |
| breakout_regime_0_prob | regime | 1 | 2281 | 0.069179 | 0.028090 | 0.406054 | nan | nan | nan |
| breakout_regime_0_prob | regime | 2 | 3349 | 0.070172 | 0.021225 | 0.302474 | nan | nan | nan |
| breakout_regime_1_prob | regime | 1 | 2281 | 0.524186 | 0.053281 | 0.101646 | nan | nan | nan |
| breakout_regime_1_prob | regime | 2 | 3349 | 0.393049 | 0.052393 | 0.133299 | nan | nan | nan |
| breakout_regime_2_prob | regime | 1 | 2281 | 0.405080 | 0.055598 | 0.137251 | nan | nan | nan |
| breakout_regime_2_prob | regime | 2 | 3349 | 0.534827 | 0.049557 | 0.092660 | nan | nan | nan |
| breakout_long_edge_score | regime | 1 | 2281 | 0.006785 | 0.070468 | 10.385243 | nan | nan | nan |
| breakout_long_edge_score | regime | 2 | 3349 | 0.014280 | 0.150589 | 10.545218 | nan | nan | nan |
| breakout_short_edge_score | regime | 1 | 2281 | -0.006785 | 0.070468 | 10.385243 | nan | nan | nan |
| breakout_short_edge_score | regime | 2 | 3349 | -0.014280 | 0.150589 | 10.545218 | nan | nan | nan |
| breakout_bullish_prob | regime | 1 | 2281 | 0.505709 | 0.053696 | 0.106180 | nan | nan | nan |
| breakout_bullish_prob | regime | 2 | 3349 | 0.510624 | 0.114971 | 0.225158 | nan | nan | nan |
| breakout_bearish_prob | regime | 1 | 2281 | 0.494291 | 0.053696 | 0.108633 | nan | nan | nan |
| breakout_bearish_prob | regime | 2 | 3349 | 0.489376 | 0.114971 | 0.234933 | nan | nan | nan |
| breakout_level_strength | regime | 1 | 2281 | 0.653568 | 0.070900 | 0.108482 | nan | nan | nan |
| breakout_level_strength | regime | 2 | 3349 | 0.641078 | 0.050342 | 0.078526 | nan | nan | nan |

## Pairwise Winsorised CV Ratios for Breakout Regime Probabilities
Each row compares two regime groups (A,B) for a given probability metric, reporting between/within WCoV and their ratio.

| Metric | Regime A | Regime B | CV_between | CV_within | CV_between/within |
|--------|----------|----------|-----------|----------|--------------------|
| breakout_regime_0_prob | 1 | 2 | 0.007125 | 0.353828 | 0.020138 |
| breakout_regime_1_prob | 1 | 2 | 0.147413 | 0.118790 | 1.240952 |
| breakout_regime_2_prob | 1 | 2 | 0.134531 | 0.109033 | 1.233860 |

## Main Feature Drivers per Factor
For each factor, the table below lists the top features by absolute Spearman correlation with that factor.


### Factor: `forward_return_h96`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| trend_strength_adx | 0.1118 | 0.1060 | 5534 |
| opposing_dist_to_round_pct | 0.0852 | 0.0708 | 5534 |
| opposing_prominence_z_score | -0.0596 | -0.0875 | 5534 |
| int_dist_opp_trend | 0.0569 | 0.0446 | 5534 |
| int_opposing_prom_squeeze | -0.0467 | -0.0184 | 5534 |
| primary_prominence_z_score | -0.0465 | -0.0855 | 5534 |
| is_flip_candidate | 0.0399 | 0.0229 | 5534 |
| dist_to_opposing_level_atr | -0.0330 | -0.0136 | 5534 |
| int_primary_prom_squeeze | -0.0324 | -0.0191 | 5534 |
| primary_dist_to_round_pct | 0.0289 | 0.0388 | 5534 |
| momentum_divergence | -0.0257 | 0.0262 | 5534 |
| penetration_depth | -0.0137 | -0.0054 | 5534 |
| primary_volume_depth_ratio | 0.0111 | 0.0247 | 5534 |
| opposing_volume_depth_ratio | -0.0093 | -0.0193 | 5534 |
| opposing_age_log_hours | 0.0065 | -0.0014 | 5534 |

### Factor: `breakout_long_edge_score`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| momentum_divergence | -0.1892 | -0.0834 | 5630 |
| opposing_age_log_hours | 0.1083 | 0.1112 | 5630 |
| is_flip_candidate | 0.0768 | 0.0785 | 5630 |
| primary_dist_to_round_pct | 0.0574 | 0.0854 | 5630 |
| primary_prominence_z_score | 0.0488 | -0.0192 | 5630 |
| trend_strength_adx | -0.0477 | -0.0389 | 5630 |
| int_primary_prom_squeeze | 0.0471 | 0.0417 | 5630 |
| primary_volume_depth_ratio | -0.0414 | -0.0463 | 5630 |
| int_opposing_prom_squeeze | -0.0252 | -0.0350 | 5630 |
| opposing_dist_to_round_pct | 0.0207 | 0.0355 | 5630 |
| opposing_prominence_z_score | -0.0167 | -0.0728 | 5630 |
| int_dist_opp_trend | -0.0151 | -0.0052 | 5630 |
| opposing_volume_depth_ratio | -0.0113 | -0.0075 | 5630 |
| dist_to_opposing_level_atr | 0.0079 | 0.0068 | 5630 |
| penetration_depth | 0.0039 | 0.0217 | 5630 |

### Factor: `breakout_short_edge_score`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| momentum_divergence | 0.1892 | 0.0834 | 5630 |
| opposing_age_log_hours | -0.1083 | -0.1112 | 5630 |
| is_flip_candidate | -0.0768 | -0.0785 | 5630 |
| primary_dist_to_round_pct | -0.0574 | -0.0854 | 5630 |
| primary_prominence_z_score | -0.0488 | 0.0192 | 5630 |
| trend_strength_adx | 0.0477 | 0.0389 | 5630 |
| int_primary_prom_squeeze | -0.0471 | -0.0417 | 5630 |
| primary_volume_depth_ratio | 0.0414 | 0.0463 | 5630 |
| int_opposing_prom_squeeze | 0.0252 | 0.0350 | 5630 |
| opposing_dist_to_round_pct | -0.0207 | -0.0355 | 5630 |
| opposing_prominence_z_score | 0.0167 | 0.0728 | 5630 |
| int_dist_opp_trend | 0.0151 | 0.0052 | 5630 |
| opposing_volume_depth_ratio | 0.0113 | 0.0075 | 5630 |
| dist_to_opposing_level_atr | -0.0079 | -0.0068 | 5630 |
| penetration_depth | -0.0039 | -0.0217 | 5630 |

### Factor: `breakout_bullish_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| momentum_divergence | -0.1938 | -0.0908 | 5630 |
| opposing_age_log_hours | 0.1051 | 0.1078 | 5630 |
| is_flip_candidate | 0.0748 | 0.0798 | 5630 |
| primary_dist_to_round_pct | 0.0557 | 0.0835 | 5630 |
| trend_strength_adx | -0.0501 | -0.0451 | 5630 |
| primary_prominence_z_score | 0.0466 | -0.0116 | 5630 |
| int_primary_prom_squeeze | 0.0440 | 0.0401 | 5630 |
| primary_volume_depth_ratio | -0.0409 | -0.0467 | 5630 |
| int_opposing_prom_squeeze | -0.0284 | -0.0397 | 5630 |
| opposing_dist_to_round_pct | 0.0209 | 0.0384 | 5630 |
| opposing_prominence_z_score | -0.0188 | -0.0691 | 5630 |
| int_dist_opp_trend | -0.0174 | -0.0108 | 5630 |
| opposing_volume_depth_ratio | -0.0128 | -0.0097 | 5630 |
| dist_to_opposing_level_atr | 0.0075 | 0.0057 | 5630 |
| penetration_depth | 0.0030 | 0.0184 | 5630 |

### Factor: `breakout_bearish_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| momentum_divergence | 0.1938 | 0.0908 | 5630 |
| opposing_age_log_hours | -0.1051 | -0.1078 | 5630 |
| is_flip_candidate | -0.0748 | -0.0798 | 5630 |
| primary_dist_to_round_pct | -0.0557 | -0.0835 | 5630 |
| trend_strength_adx | 0.0501 | 0.0451 | 5630 |
| primary_prominence_z_score | -0.0466 | 0.0116 | 5630 |
| int_primary_prom_squeeze | -0.0440 | -0.0401 | 5630 |
| primary_volume_depth_ratio | 0.0409 | 0.0467 | 5630 |
| int_opposing_prom_squeeze | 0.0284 | 0.0397 | 5630 |
| opposing_dist_to_round_pct | -0.0209 | -0.0384 | 5630 |
| opposing_prominence_z_score | 0.0188 | 0.0691 | 5630 |
| int_dist_opp_trend | 0.0174 | 0.0108 | 5630 |
| opposing_volume_depth_ratio | 0.0128 | 0.0097 | 5630 |
| dist_to_opposing_level_atr | -0.0075 | -0.0057 | 5630 |
| penetration_depth | -0.0030 | -0.0184 | 5630 |

### Factor: `breakout_regime_0_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| opposing_prominence_z_score | 0.3923 | 0.5819 | 5630 |
| int_opposing_prom_squeeze | 0.3614 | 0.2916 | 5630 |
| primary_prominence_z_score | 0.3192 | 0.6638 | 5630 |
| int_primary_prom_squeeze | 0.3085 | 0.3082 | 5630 |
| opposing_dist_to_round_pct | 0.2412 | 0.2058 | 5630 |
| dist_to_opposing_level_atr | 0.1882 | -0.0051 | 5630 |
| opposing_volume_depth_ratio | 0.1210 | -0.0109 | 5630 |
| primary_dist_to_round_pct | 0.1114 | 0.1390 | 5630 |
| int_dist_opp_trend | 0.0947 | -0.0341 | 5630 |
| penetration_depth | -0.0714 | -0.0696 | 5630 |
| momentum_divergence | 0.0685 | 0.0329 | 5630 |
| primary_volume_depth_ratio | 0.0603 | -0.0738 | 5630 |
| is_flip_candidate | -0.0560 | -0.0722 | 5630 |
| opposing_age_log_hours | -0.0050 | 0.0126 | 5630 |
| trend_strength_adx | -0.0037 | -0.1330 | 5630 |

### Factor: `breakout_regime_1_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| opposing_prominence_z_score | -0.3307 | -0.3202 | 5630 |
| int_opposing_prom_squeeze | -0.2586 | -0.1743 | 5630 |
| is_flip_candidate | 0.1625 | 0.2513 | 5630 |
| opposing_dist_to_round_pct | -0.1564 | -0.1664 | 5630 |
| penetration_depth | 0.1101 | 0.0752 | 5630 |
| trend_strength_adx | -0.0971 | -0.0846 | 5630 |
| opposing_age_log_hours | -0.0937 | 0.0653 | 5630 |
| primary_dist_to_round_pct | -0.0877 | -0.1231 | 5630 |
| momentum_divergence | -0.0843 | -0.0719 | 5630 |
| primary_prominence_z_score | -0.0676 | -0.0156 | 5630 |
| dist_to_opposing_level_atr | 0.0475 | 0.3079 | 5630 |
| int_primary_prom_squeeze | -0.0471 | -0.0057 | 5630 |
| primary_volume_depth_ratio | -0.0277 | -0.0132 | 5630 |
| opposing_volume_depth_ratio | 0.0217 | 0.0171 | 5630 |
| int_dist_opp_trend | 0.0003 | 0.1294 | 5630 |

### Factor: `breakout_regime_2_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| opposing_prominence_z_score | 0.1773 | 0.0880 | 5630 |
| is_flip_candidate | -0.1604 | -0.2240 | 5630 |
| trend_strength_adx | 0.1449 | 0.1388 | 5630 |
| dist_to_opposing_level_atr | -0.1346 | -0.3081 | 5630 |
| int_primary_prom_squeeze | -0.1186 | -0.1185 | 5630 |
| int_opposing_prom_squeeze | 0.1108 | 0.0581 | 5630 |
| primary_prominence_z_score | -0.0999 | -0.2519 | 5630 |
| momentum_divergence | 0.0798 | 0.0592 | 5630 |
| opposing_dist_to_round_pct | 0.0779 | 0.0846 | 5630 |
| penetration_depth | -0.0696 | -0.0477 | 5630 |
| primary_volume_depth_ratio | 0.0659 | 0.0430 | 5630 |
| opposing_age_log_hours | 0.0576 | -0.0708 | 5630 |
| primary_dist_to_round_pct | 0.0494 | 0.0680 | 5630 |
| opposing_volume_depth_ratio | -0.0377 | -0.0129 | 5630 |
| int_dist_opp_trend | -0.0301 | -0.1166 | 5630 |