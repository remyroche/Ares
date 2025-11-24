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
- Macro ROC AUC (OvR, val): **0.4794**
- Macro ROC AUC (OvR, test): **0.4928**
- Macro F1-score (val): **0.3255**
- Macro F1-score (test): **0.3141**
- Generalization gap (Macro F1 test - val): **-0.0113**
- Weighted F1-score (val): **0.4439**
- Weighted F1-score (test): **0.3974**
- Precision (breakout class 1, val): **0.4571**
- Sample split: train=3940, val=845, test=845

### Class Counts (training labels)
| Regime | Count |
|--------|-------|
| 0 | 442 |
| 1 | 2528 |
| 2 | 2660 |

### Resistance/Support Hit Counts by Outcome
| Side | Class 0 | Class 1 | Class 2 | Total |
|--------|--------|--------|--------|--------|
| resistance | 30 | 1415 | 1207 | 2652 |
| support | 109 | 1184 | 1685 | 2978 |

## Forward Return Sharpe-like Ratios
| Scope | Side | Regime | Mean Return | Std Return | Sharpe-like |
|-------|------|--------|-------------|------------|-------------|
| global | all | -1 | 0.030688 | 0.140231 | 0.2188 |
| regime | all | 0 | 0.004141 | 0.044598 | 0.0929 |
| regime | all | 1 | 0.032843 | 0.138935 | 0.2364 |
| regime | all | 2 | 0.030063 | 0.144252 | 0.2084 |
| global | resistance | -1 | 0.034193 | 0.139660 | 0.2448 |
| regime | resistance | 0 | 0.009037 | 0.040693 | 0.2221 |
| regime | resistance | 1 | 0.080357 | 0.133908 | 0.6001 |
| regime | resistance | 2 | -0.018979 | 0.128304 | -0.1479 |
| global | support | -1 | 0.027567 | 0.140663 | 0.1960 |
| regime | support | 0 | 0.002794 | 0.045522 | 0.0614 |
| regime | support | 1 | -0.024265 | 0.122379 | -0.1983 |
| regime | support | 2 | 0.065135 | 0.144801 | 0.4498 |

## Per-Regime Summary (Forward Returns & Edge Scores)
| Regime | Side | Count | Mean Forward Return | Sharpe-like | Mean Long Edge | Mean Short Edge | Mean Bullish Prob | Mean Bearish Prob |
|--------|------|-------|---------------------|------------|---------------|----------------|-------------------|--------------------|
| 0 | all | 139 | 0.004141 | 0.0929 | 0.365333 | -0.365333 | 0.724436 | 0.275564 |
| 0 | resistance | 30 | 0.009037 | 0.2221 | -0.408528 | 0.408528 | 0.203591 | 0.796409 |
| 0 | support | 109 | 0.002794 | 0.0614 | 0.578322 | -0.578322 | 0.867788 | 0.132212 |
| 1 | all | 2599 | 0.032843 | 0.2364 | 0.050645 | -0.050645 | 0.543326 | 0.456674 |
| 1 | resistance | 1415 | 0.080357 | 0.6001 | 0.357909 | -0.357909 | 0.779711 | 0.220289 |
| 1 | support | 1184 | -0.024265 | -0.1983 | -0.316567 | 0.316567 | 0.260822 | 0.739178 |
| 2 | all | 2892 | 0.030063 | 0.2084 | 0.051557 | -0.051557 | 0.542325 | 0.457675 |
| 2 | resistance | 1207 | -0.018979 | -0.1479 | -0.386495 | 0.386495 | 0.202589 | 0.797411 |
| 2 | support | 1685 | 0.065135 | 0.4498 | 0.365343 | -0.365343 | 0.785684 | 0.214316 |

## Winsorised CV Between/Within Regimes
Definition: per metric, we winsorise values (5–95%), compute global and per-regime CV, and report the ratio of between-regime CV to within-regime CV.

| Metric | Scope | Regime | Count | Mean | Std | CV | CV_between | CV_within | CV_between/within |
|--------|-------|--------|-------|------|-----|----|-----------|----------|--------------------|
| breakout_level_strength | global | -1 | 5630 | 0.646075 | 0.059164 | 0.091575 | 0.089880 | 0.090351 | 0.994791 |
| breakout_bearish_prob | global | -1 | 5630 | 0.452501 | 0.306102 | 0.676467 | 0.191922 | 0.656064 | 0.292536 |
| breakout_bullish_prob | global | -1 | 5630 | 0.547499 | 0.306102 | 0.559091 | 0.158621 | 0.542229 | 0.292536 |
| breakout_short_edge_score | global | -1 | 5630 | -0.060484 | 0.395036 | 6.531276 | 2.469942 | 6.619385 | 0.373138 |
| breakout_long_edge_score | global | -1 | 5630 | 0.060484 | 0.395036 | 6.531276 | 2.469942 | 6.619385 | 0.373138 |
| breakout_regime_2_prob | global | -1 | 5630 | 0.497361 | 0.308276 | 0.619824 | 0.546939 | 0.261913 | 2.088249 |
| breakout_regime_1_prob | global | -1 | 5630 | 0.462235 | 0.307341 | 0.664901 | 0.598350 | 0.268107 | 2.231757 |
| breakout_regime_0_prob | global | -1 | 5630 | 0.024784 | 0.041461 | 1.672926 | 12.613323 | 2.961602 | 4.258953 |
| trend_strength_adx | global | -1 | 5630 | 20.209757 | 10.882229 | 0.538464 | 0.123930 | 0.457055 | 0.271148 |
| momentum_divergence | global | -1 | 5630 | 2.365355 | 68.290518 | 28.871146 | 1.849607 | 25.014639 | 0.073941 |
| age_log_hours | global | -1 | 5630 | 3.230563 | 0.984491 | 0.304743 | 0.064936 | 0.318526 | 0.203863 |
| primary_dist_to_round_pct | global | -1 | 5630 | 0.000055 | 0.000036 | 0.656156 | 0.071141 | 0.704376 | 0.100999 |
| bollinger_squeeze | global | -1 | 5630 | 0.121478 | 0.107816 | 0.887531 | 0.296609 | 0.708514 | 0.418636 |
| dist_to_opposing_level_atr | global | -1 | 5630 | 3.210167 | 3.008249 | 0.937100 | 0.114255 | 0.773401 | 0.147731 |
| primary_volume_depth_ratio | global | -1 | 5630 | 1.458701 | 0.410670 | 0.281531 | 0.052141 | 0.257465 | 0.202515 |
| rubber_band_extension | global | -1 | 5630 | 0.006306 | 0.034401 | 5.455004 | 0.542903 | 4.677840 | 0.116059 |
| test_count | global | -1 | 5630 | 25.869627 | 16.098003 | 0.622274 | 0.463027 | 0.787234 | 0.588170 |
| opposing_age_log_hours | global | -1 | 5630 | 3.929225 | 1.213634 | 0.308874 | 0.028319 | 0.296905 | 0.095382 |
| primary_prominence_z_score | global | -1 | 5630 | -0.101957 | 0.788940 | 7.737971 | 7.710040 | 8.530611 | 0.903809 |
| opposing_volume_depth_ratio | global | -1 | 5630 | 1.470897 | 0.461308 | 0.313624 | 0.008506 | 0.267915 | 0.031749 |
| opposing_dist_to_round_pct | global | -1 | 5630 | 0.000057 | 0.000039 | 0.681558 | 0.235103 | 0.664680 | 0.353709 |
| opposing_prominence_z_score | global | -1 | 5630 | -0.019022 | 0.892441 | 46.917466 | 39.585119 | 45.720628 | 0.865804 |
| is_flip_candidate | global | -1 | 5630 | 0.200178 | 0.400133 | 1.998891 | 0.396824 | 1.721949 | 0.230450 |
| forward_return_support | global | -1 | 2927 | 0.023505 | 0.115813 | 4.927188 | 1.593534 | 3.915043 | 0.407029 |
| forward_return_resistance | global | -1 | 2607 | 0.031286 | 0.119737 | 3.827147 | 1.422020 | 2.827800 | 0.502872 |
| forward_return | global | -1 | 5534 | 0.027063 | 0.117365 | 4.336733 | 0.446715 | 3.432938 | 0.130126 |
| forward_return | regime | 0 | 139 | 0.002474 | 0.041073 | 16.602652 | nan | nan | nan |
| forward_return | regime | 1 | 2541 | 0.030030 | 0.116127 | 3.867002 | nan | nan | nan |
| forward_return | regime | 2 | 2854 | 0.025643 | 0.121516 | 4.738853 | nan | nan | nan |
| forward_return_resistance | regime | 0 | 30 | 0.009630 | 0.039606 | 4.112759 | nan | nan | nan |
| forward_return_resistance | regime | 1 | 1387 | 0.083411 | 0.117310 | 1.406422 | nan | nan | nan |
| forward_return_resistance | regime | 2 | 1190 | -0.022937 | 0.108496 | 4.730238 | nan | nan | nan |
| forward_return_support | regime | 0 | 109 | 0.002026 | 0.043777 | 21.612393 | nan | nan | nan |
| forward_return_support | regime | 1 | 1154 | -0.025581 | 0.101716 | 3.976308 | nan | nan | nan |
| forward_return_support | regime | 2 | 1664 | 0.063996 | 0.130575 | 2.040353 | nan | nan | nan |
| is_flip_candidate | regime | 0 | 139 | 0.057554 | 0.232898 | 4.046604 | nan | nan | nan |
| is_flip_candidate | regime | 1 | 2599 | 0.252020 | 0.434173 | 1.722771 | nan | nan | nan |
| is_flip_candidate | regime | 2 | 2892 | 0.160443 | 0.367016 | 2.287522 | nan | nan | nan |
| opposing_prominence_z_score | regime | 0 | 139 | 1.525847 | 0.924759 | 0.606063 | nan | nan | nan |
| opposing_prominence_z_score | regime | 1 | 2599 | -0.144113 | 0.814000 | 5.648347 | nan | nan | nan |
| opposing_prominence_z_score | regime | 2 | 2892 | 0.012814 | 0.870268 | 67.916563 | nan | nan | nan |
| opposing_dist_to_round_pct | regime | 0 | 139 | 0.000084 | 0.000037 | 0.435955 | nan | nan | nan |
| opposing_dist_to_round_pct | regime | 1 | 2599 | 0.000054 | 0.000037 | 0.673380 | nan | nan | nan |
| opposing_dist_to_round_pct | regime | 2 | 2892 | 0.000058 | 0.000040 | 0.693949 | nan | nan | nan |
| opposing_volume_depth_ratio | regime | 0 | 139 | 1.449140 | 0.227042 | 0.156674 | nan | nan | nan |
| opposing_volume_depth_ratio | regime | 1 | 2599 | 1.476882 | 0.455686 | 0.308546 | nan | nan | nan |
| opposing_volume_depth_ratio | regime | 2 | 2892 | 1.474290 | 0.499497 | 0.338805 | nan | nan | nan |
| primary_prominence_z_score | regime | 0 | 139 | 1.520022 | 1.156898 | 0.761106 | nan | nan | nan |
| primary_prominence_z_score | regime | 1 | 2599 | -0.116255 | 0.803523 | 6.911715 | nan | nan | nan |
| primary_prominence_z_score | regime | 2 | 2892 | -0.177138 | 0.648843 | 3.662920 | nan | nan | nan |
| opposing_age_log_hours | regime | 0 | 139 | 4.157262 | 1.069847 | 0.257344 | nan | nan | nan |
| opposing_age_log_hours | regime | 1 | 2599 | 3.926681 | 1.345279 | 0.342600 | nan | nan | nan |
| opposing_age_log_hours | regime | 2 | 2892 | 3.916104 | 1.084693 | 0.276983 | nan | nan | nan |
| test_count | regime | 0 | 139 | 50.906475 | 29.206260 | 0.573724 | nan | nan | nan |
| test_count | regime | 1 | 2599 | 25.836091 | 16.611119 | 0.642942 | nan | nan | nan |
| test_count | regime | 2 | 2892 | 25.170124 | 15.278990 | 0.607029 | nan | nan | nan |
| rubber_band_extension | regime | 0 | 139 | 0.013362 | 0.019461 | 1.456477 | nan | nan | nan |
| rubber_band_extension | regime | 1 | 2599 | 0.005879 | 0.031400 | 5.341112 | nan | nan | nan |
| rubber_band_extension | regime | 2 | 2892 | 0.006341 | 0.037638 | 5.935557 | nan | nan | nan |
| primary_volume_depth_ratio | regime | 0 | 139 | 1.301219 | 0.302910 | 0.232789 | nan | nan | nan |
| primary_volume_depth_ratio | regime | 1 | 2599 | 1.452823 | 0.398425 | 0.274242 | nan | nan | nan |
| primary_volume_depth_ratio | regime | 2 | 2892 | 1.470795 | 0.425361 | 0.289205 | nan | nan | nan |
| dist_to_opposing_level_atr | regime | 0 | 139 | 3.202567 | 1.359931 | 0.424638 | nan | nan | nan |
| dist_to_opposing_level_atr | regime | 1 | 2599 | 3.699312 | 3.740109 | 1.011028 | nan | nan | nan |
| dist_to_opposing_level_atr | regime | 2 | 2892 | 2.802631 | 2.348204 | 0.837857 | nan | nan | nan |
| bollinger_squeeze | regime | 0 | 139 | 0.046910 | 0.042368 | 0.903179 | nan | nan | nan |
| bollinger_squeeze | regime | 1 | 2599 | 0.119997 | 0.103719 | 0.864347 | nan | nan | nan |
| bollinger_squeeze | regime | 2 | 2892 | 0.126301 | 0.112120 | 0.887725 | nan | nan | nan |
| primary_dist_to_round_pct | regime | 0 | 139 | 0.000063 | 0.000044 | 0.689770 | nan | nan | nan |
| primary_dist_to_round_pct | regime | 1 | 2599 | 0.000054 | 0.000036 | 0.652627 | nan | nan | nan |
| primary_dist_to_round_pct | regime | 2 | 2892 | 0.000056 | 0.000037 | 0.669435 | nan | nan | nan |
| age_log_hours | regime | 0 | 139 | 2.816908 | 1.120516 | 0.397782 | nan | nan | nan |
| age_log_hours | regime | 1 | 2599 | 3.159003 | 0.954767 | 0.302237 | nan | nan | nan |
| age_log_hours | regime | 2 | 2892 | 3.320011 | 1.011776 | 0.304751 | nan | nan | nan |
| momentum_divergence | regime | 0 | 139 | 6.154515 | 32.792423 | 5.328190 | nan | nan | nan |
| momentum_divergence | regime | 1 | 2599 | -3.635491 | 75.419778 | 20.745418 | nan | nan | nan |
| momentum_divergence | regime | 2 | 2892 | 5.034263 | 69.293325 | 13.764342 | nan | nan | nan |
| trend_strength_adx | regime | 0 | 139 | 15.040921 | 5.752818 | 0.382478 | nan | nan | nan |
| trend_strength_adx | regime | 1 | 2599 | 20.072728 | 11.386763 | 0.567275 | nan | nan | nan |
| trend_strength_adx | regime | 2 | 2892 | 20.596400 | 10.571355 | 0.513262 | nan | nan | nan |
| breakout_regime_0_prob | regime | 0 | 139 | 0.682106 | 0.166039 | 0.243421 | nan | nan | nan |
| breakout_regime_0_prob | regime | 1 | 2599 | 0.019346 | 0.028578 | 1.477235 | nan | nan | nan |
| breakout_regime_0_prob | regime | 2 | 2892 | 0.018602 | 0.025581 | 1.375185 | nan | nan | nan |
| breakout_regime_1_prob | regime | 0 | 139 | 0.146315 | 0.094566 | 0.646316 | nan | nan | nan |
| breakout_regime_1_prob | regime | 1 | 2599 | 0.761826 | 0.142322 | 0.186817 | nan | nan | nan |
| breakout_regime_1_prob | regime | 2 | 2892 | 0.208937 | 0.134898 | 0.645638 | nan | nan | nan |
| breakout_regime_2_prob | regime | 0 | 139 | 0.169466 | 0.113020 | 0.666921 | nan | nan | nan |
| breakout_regime_2_prob | regime | 1 | 2599 | 0.214582 | 0.136195 | 0.634698 | nan | nan | nan |
| breakout_regime_2_prob | regime | 2 | 2892 | 0.767753 | 0.141580 | 0.184408 | nan | nan | nan |
| breakout_long_edge_score | regime | 0 | 139 | 0.369979 | 0.419787 | 1.134623 | nan | nan | nan |
| breakout_long_edge_score | regime | 1 | 2599 | 0.052982 | 0.376933 | 7.114350 | nan | nan | nan |
| breakout_long_edge_score | regime | 2 | 2892 | 0.053162 | 0.404377 | 7.606445 | nan | nan | nan |
| breakout_short_edge_score | regime | 0 | 139 | -0.369979 | 0.419787 | 1.134623 | nan | nan | nan |
| breakout_short_edge_score | regime | 1 | 2599 | -0.052982 | 0.376933 | 7.114350 | nan | nan | nan |
| breakout_short_edge_score | regime | 2 | 2892 | -0.053162 | 0.404377 | 7.606445 | nan | nan | nan |
| breakout_bullish_prob | regime | 0 | 139 | 0.727290 | 0.282070 | 0.387837 | nan | nan | nan |
| breakout_bullish_prob | regime | 1 | 2599 | 0.543994 | 0.292328 | 0.537373 | nan | nan | nan |
| breakout_bullish_prob | regime | 2 | 2892 | 0.542148 | 0.316211 | 0.583257 | nan | nan | nan |
| breakout_bearish_prob | regime | 0 | 139 | 0.272710 | 0.282070 | 1.034322 | nan | nan | nan |
| breakout_bearish_prob | regime | 1 | 2599 | 0.456006 | 0.292328 | 0.641062 | nan | nan | nan |
| breakout_bearish_prob | regime | 2 | 2892 | 0.457852 | 0.316211 | 0.690640 | nan | nan | nan |
| breakout_level_strength | regime | 0 | 139 | 0.766073 | 0.064054 | 0.083614 | nan | nan | nan |
| breakout_level_strength | regime | 1 | 2599 | 0.645576 | 0.059385 | 0.091987 | nan | nan | nan |
| breakout_level_strength | regime | 2 | 2892 | 0.640369 | 0.051680 | 0.080704 | nan | nan | nan |

## Pairwise Winsorised CV Ratios for Breakout Regime Probabilities
Each row compares two regime groups (A,B) for a given probability metric, reporting between/within WCoV and their ratio.

| Metric | Regime A | Regime B | CV_between | CV_within | CV_between/within |
|--------|----------|----------|-----------|----------|--------------------|
| breakout_regime_0_prob | 0 | 1 | 7.829518 | 2.299108 | 3.405459 |
| breakout_regime_0_prob | 0 | 2 | 8.702481 | 2.513264 | 3.462620 |
| breakout_regime_0_prob | 1 | 2 | 0.019625 | 1.427973 | 0.013743 |
| breakout_regime_1_prob | 0 | 1 | 0.415426 | 0.159882 | 2.598322 |
| breakout_regime_1_prob | 0 | 2 | 0.151939 | 0.556745 | 0.272906 |
| breakout_regime_1_prob | 1 | 2 | 0.587822 | 0.294736 | 1.994406 |
| breakout_regime_2_prob | 0 | 1 | 0.106247 | 0.586896 | 0.181032 |
| breakout_regime_2_prob | 0 | 2 | 0.397936 | 0.169341 | 2.349908 |
| breakout_regime_2_prob | 1 | 2 | 0.546930 | 0.274640 | 1.991440 |

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
| momentum_divergence | -0.1314 | -0.1007 | 5630 |
| opposing_age_log_hours | 0.0736 | 0.0748 | 5630 |
| rubber_band_extension | -0.0698 | -0.0417 | 5630 |
| is_flip_candidate | 0.0697 | 0.0614 | 5630 |
| opposing_dist_to_round_pct | 0.0507 | 0.0559 | 5630 |
| age_log_hours | 0.0450 | 0.0597 | 5630 |
| test_count | -0.0408 | -0.0744 | 5630 |
| bollinger_squeeze | -0.0348 | -0.0250 | 5630 |
| primary_dist_to_round_pct | 0.0344 | 0.0625 | 5630 |
| primary_prominence_z_score | 0.0310 | -0.0521 | 5630 |
| opposing_volume_depth_ratio | -0.0150 | -0.0263 | 5630 |
| opposing_prominence_z_score | -0.0136 | -0.0861 | 5630 |
| primary_volume_depth_ratio | -0.0126 | -0.0148 | 5630 |
| trend_strength_adx | 0.0120 | 0.0242 | 5630 |
| dist_to_opposing_level_atr | 0.0099 | 0.0115 | 5630 |

### Factor: `breakout_short_edge_score`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| momentum_divergence | 0.1314 | 0.1007 | 5630 |
| opposing_age_log_hours | -0.0736 | -0.0748 | 5630 |
| rubber_band_extension | 0.0698 | 0.0417 | 5630 |
| is_flip_candidate | -0.0697 | -0.0614 | 5630 |
| opposing_dist_to_round_pct | -0.0507 | -0.0559 | 5630 |
| age_log_hours | -0.0450 | -0.0597 | 5630 |
| test_count | 0.0408 | 0.0744 | 5630 |
| bollinger_squeeze | 0.0348 | 0.0250 | 5630 |
| primary_dist_to_round_pct | -0.0344 | -0.0625 | 5630 |
| primary_prominence_z_score | -0.0310 | 0.0521 | 5630 |
| opposing_volume_depth_ratio | 0.0150 | 0.0263 | 5630 |
| opposing_prominence_z_score | 0.0136 | 0.0861 | 5630 |
| primary_volume_depth_ratio | 0.0126 | 0.0148 | 5630 |
| trend_strength_adx | -0.0120 | -0.0242 | 5630 |
| dist_to_opposing_level_atr | -0.0099 | -0.0115 | 5630 |

### Factor: `breakout_bullish_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| momentum_divergence | -0.1450 | -0.1152 | 5630 |
| rubber_band_extension | -0.0782 | -0.0458 | 5630 |
| opposing_age_log_hours | 0.0715 | 0.0720 | 5630 |
| is_flip_candidate | 0.0711 | 0.0623 | 5630 |
| test_count | -0.0556 | -0.0723 | 5630 |
| age_log_hours | 0.0532 | 0.0644 | 5630 |
| opposing_dist_to_round_pct | 0.0504 | 0.0620 | 5630 |
| primary_dist_to_round_pct | 0.0361 | 0.0629 | 5630 |
| opposing_prominence_z_score | -0.0309 | -0.0800 | 5630 |
| bollinger_squeeze | -0.0277 | -0.0265 | 5630 |
| primary_prominence_z_score | 0.0141 | -0.0454 | 5630 |
| opposing_volume_depth_ratio | -0.0128 | -0.0267 | 5630 |
| trend_strength_adx | 0.0119 | 0.0193 | 5630 |
| primary_volume_depth_ratio | -0.0113 | -0.0140 | 5630 |
| dist_to_opposing_level_atr | 0.0093 | 0.0119 | 5630 |

### Factor: `breakout_bearish_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| momentum_divergence | 0.1450 | 0.1152 | 5630 |
| rubber_band_extension | 0.0782 | 0.0458 | 5630 |
| opposing_age_log_hours | -0.0715 | -0.0720 | 5630 |
| is_flip_candidate | -0.0711 | -0.0623 | 5630 |
| test_count | 0.0556 | 0.0723 | 5630 |
| age_log_hours | -0.0532 | -0.0644 | 5630 |
| opposing_dist_to_round_pct | -0.0504 | -0.0620 | 5630 |
| primary_dist_to_round_pct | -0.0361 | -0.0629 | 5630 |
| opposing_prominence_z_score | 0.0309 | 0.0800 | 5630 |
| bollinger_squeeze | 0.0277 | 0.0265 | 5630 |
| primary_prominence_z_score | -0.0141 | 0.0454 | 5630 |
| opposing_volume_depth_ratio | 0.0128 | 0.0267 | 5630 |
| trend_strength_adx | -0.0119 | -0.0193 | 5630 |
| primary_volume_depth_ratio | 0.0113 | 0.0140 | 5630 |
| dist_to_opposing_level_atr | -0.0093 | -0.0119 | 5630 |

### Factor: `breakout_regime_0_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| opposing_prominence_z_score | 0.2977 | 0.3117 | 5630 |
| primary_prominence_z_score | 0.2619 | 0.3266 | 5630 |
| opposing_dist_to_round_pct | 0.2583 | 0.1531 | 5630 |
| rubber_band_extension | 0.2369 | 0.0918 | 5630 |
| bollinger_squeeze | -0.2274 | -0.1171 | 5630 |
| test_count | 0.1912 | 0.2449 | 5630 |
| age_log_hours | -0.1474 | -0.0809 | 5630 |
| dist_to_opposing_level_atr | 0.1358 | -0.0054 | 5630 |
| is_flip_candidate | -0.1019 | -0.0657 | 5630 |
| trend_strength_adx | -0.0747 | -0.0812 | 5630 |
| primary_dist_to_round_pct | 0.0641 | 0.0570 | 5630 |
| opposing_age_log_hours | -0.0550 | 0.0209 | 5630 |
| opposing_volume_depth_ratio | 0.0444 | -0.0117 | 5630 |
| momentum_divergence | 0.0112 | 0.0418 | 5630 |
| primary_volume_depth_ratio | 0.0040 | -0.0409 | 5630 |

### Factor: `breakout_regime_1_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| opposing_prominence_z_score | -0.1795 | -0.1700 | 5630 |
| is_flip_candidate | 0.1571 | 0.1579 | 5630 |
| age_log_hours | -0.0777 | -0.0683 | 5630 |
| opposing_dist_to_round_pct | -0.0626 | -0.0831 | 5630 |
| dist_to_opposing_level_atr | 0.0541 | 0.1673 | 5630 |
| primary_prominence_z_score | -0.0498 | -0.0214 | 5630 |
| trend_strength_adx | -0.0403 | -0.0188 | 5630 |
| momentum_divergence | -0.0363 | -0.0458 | 5630 |
| primary_dist_to_round_pct | -0.0312 | -0.0375 | 5630 |
| test_count | -0.0270 | -0.0183 | 5630 |
| primary_volume_depth_ratio | -0.0196 | -0.0300 | 5630 |
| bollinger_squeeze | 0.0109 | -0.0122 | 5630 |
| opposing_volume_depth_ratio | 0.0096 | -0.0319 | 5630 |
| rubber_band_extension | -0.0067 | -0.0153 | 5630 |
| opposing_age_log_hours | -0.0026 | 0.0305 | 5630 |

### Factor: `breakout_regime_2_prob`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| is_flip_candidate | -0.1391 | -0.1330 | 5630 |
| age_log_hours | 0.1204 | 0.0982 | 5630 |
| dist_to_opposing_level_atr | -0.1068 | -0.1649 | 5630 |
| opposing_prominence_z_score | 0.0797 | 0.0536 | 5630 |
| trend_strength_adx | 0.0694 | 0.0489 | 5630 |
| bollinger_squeeze | 0.0594 | 0.0557 | 5630 |
| primary_volume_depth_ratio | 0.0412 | 0.0451 | 5630 |
| primary_prominence_z_score | -0.0386 | -0.1002 | 5630 |
| momentum_divergence | 0.0371 | 0.0301 | 5630 |
| test_count | -0.0348 | -0.0729 | 5630 |
| rubber_band_extension | -0.0309 | -0.0189 | 5630 |
| opposing_volume_depth_ratio | -0.0249 | 0.0362 | 5630 |
| opposing_age_log_hours | -0.0151 | -0.0382 | 5630 |
| primary_dist_to_round_pct | 0.0119 | 0.0162 | 5630 |
| opposing_dist_to_round_pct | -0.0044 | 0.0259 | 5630 |