# Breakout/Bounce Regime Diagnostics

- Symbol: **ETHUSDT**
- Exchange: **binance**
- Timeframe: **15m**
- Direction: **long**
- Horizon (bars): **24**
- Samples (training window): **156**

## Model Output Explanation

The model predicts one of three regimes for each support/resistance level approach:

- **Regime 0 (Bounce)**: Price bounces off the level without breaking through
  - Resistance: Price bounces down
  - Support: Price bounces up

- **Regime 1 (Breakout)**: Price breaks cleanly through the level and holds
  - Resistance: Price breaks up and sustains above
  - Support: Price breaks down and sustains below

- **Regime 2 (Trap/Fakeout)**: Price briefly breaks through but quickly reverses
  - False breakout that traps traders on the wrong side

**Model outputs** are probability distributions over these three regimes:
- `breakout_regime_0_prob`: Probability of bounce (0.0 to 1.0)
- `breakout_regime_1_prob`: Probability of breakout (0.0 to 1.0)
- `breakout_regime_2_prob`: Probability of trap (0.0 to 1.0)

The predicted regime is the one with the highest probability (argmax).

**Success probability** (`breakout_success_prob`) is a weighted average of regime probabilities
using empirical meta-label success rates. Values closer to 1.0 indicate higher confidence
in profitable trade outcomes based on historical triple-barrier labeling.

## Global Model Metrics
- Validation log loss: **nan**
- Test log loss: **nan**
- Generalization gap (test - val log loss): **nan**
- Macro F1-score (val): **0.2069**
- Weighted F1-score (val): **0.1079**
- Sample split: train=109, val=23, test=24

### Class Counts (training labels)
| Regime | Count |
|--------|-------|
| 0 | 83 |
| 1 | 73 |

### Resistance/Support Hit Counts by Outcome
| Side | Class 0 | Total |
|--------|--------|--------|
| support | 156 | 156 |

## Meta-Label Success Summary
- Meta-labeled events: **156**, success=1: **23** (14.744% success rate)

| Class | Meta Events | Success Count | Success Rate |
|-------|------------|---------------|--------------|
| 0 | 83 | 20 | 24.096% |
| 1 | 73 | 3 | 4.110% |

## Breakout Success Probability & High-Confidence Gating
- Observations with breakout_success_prob: **156** | mean=0.500, std=0.000, p25=0.500, median=0.500, p75=0.500
- High-confidence signals (high_conf=1): **0** / 156 (0.000%)

### Forward Return Sharpe by Meta/High-Confidence Subset
| Subset | Samples | Mean Return | Std Return | Sharpe-like |
|--------|---------|-------------|------------|-------------|
| all | 132 | 0.155068 | 0.488889 | 0.3172 |
| meta_success==1 | 23 | 0.525005 | 0.632132 | 0.8305 |

## Forward Return Sharpe-like Ratios
| Scope | Side | Regime | Mean Return | Std Return | Sharpe-like |
|-------|------|--------|-------------|------------|-------------|
| global | all | -1 | 0.155068 | 0.488889 | 0.3172 |
| regime | all | 0 | 0.155068 | 0.488889 | 0.3172 |
| global | resistance | -1 | nan | nan | nan |
| regime | resistance | 0 | nan | nan | nan |
| global | support | -1 | 0.155068 | 0.488889 | 0.3172 |
| regime | support | 0 | 0.155068 | 0.488889 | 0.3172 |

## Per-Regime Summary (Forward Returns & Edge Scores)
| Regime | Side | Count | Mean Forward Return | Sharpe-like | Mean Long Edge | Mean Short Edge | Mean Bullish Prob | Mean Bearish Prob |
|--------|------|-------|---------------------|------------|---------------|----------------|-------------------|--------------------|
| 0 | all | 156 | 0.155068 | 0.3172 | 0.093221 | -0.093221 | 0.686442 | 0.313558 |
| 0 | support | 156 | 0.155068 | 0.3172 | 0.093221 | -0.093221 | 0.686442 | 0.313558 |

## Winsorised CV Between/Within Regimes
Definition: per metric, we winsorise values (5–95%), compute global and per-regime CV, and report the ratio of between-regime CV to within-regime CV.

| Metric | Scope | Regime | Count | Mean | Std | CV | CV_between | CV_within | CV_between/within |
|--------|-------|--------|-------|------|-----|----|-----------|----------|--------------------|
| breakout_level_strength | global | -1 | 156 | 0.250000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| breakout_bearish_prob | global | -1 | 156 | 0.313558 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| breakout_bullish_prob | global | -1 | 156 | 0.686442 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| breakout_short_edge_score | global | -1 | 156 | -0.093221 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| breakout_long_edge_score | global | -1 | 156 | 0.093221 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| breakout_regime_2_prob | global | -1 | 156 | 0.313558 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| breakout_regime_1_prob | global | -1 | 156 | 0.313558 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| breakout_regime_0_prob | global | -1 | 156 | 0.372885 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| order_book_imbalance_proxy | global | -1 | 156 | 0.047969 | 0.671232 | 13.993018 | 0.000000 | 13.993018 | 0.000000 |
| fakeout_ratio | global | -1 | 156 | 0.012543 | 0.446975 | 35.636387 | 0.000000 | 35.636387 | 0.000000 |
| volume_at_impact | global | -1 | 156 | -0.077001 | 0.748612 | 9.722088 | 0.000000 | 9.722088 | 0.000000 |
| close_proximity | global | -1 | 156 | -0.053871 | 0.618899 | 11.488579 | 0.000000 | 11.488579 | 0.000000 |
| rejection_wick_ratio | global | -1 | 156 | 0.030123 | 0.024647 | 0.818223 | 0.000000 | 0.818223 | 0.000000 |
| penetration_depth | global | -1 | 156 | -0.020853 | 0.593805 | 28.476102 | 0.000000 | 28.476102 | 0.000000 |
| age_log_hours | global | -1 | 156 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| test_count | global | -1 | 156 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| inside_bar_chain | global | -1 | 156 | 0.022611 | 0.710816 | 31.437278 | 0.000000 | 31.437278 | 0.000000 |
| volatility_compression | global | -1 | 156 | -0.028982 | 0.355578 | 12.269087 | 0.000000 | 12.269087 | 0.000000 |
| bollinger_squeeze | global | -1 | 156 | -0.341582 | 0.454519 | 1.330628 | 0.000000 | 1.330628 | 0.000000 |
| trend_strength_adx | global | -1 | 156 | -0.331365 | 0.697486 | 2.104885 | 0.000000 | 2.104885 | 0.000000 |
| momentum_divergence | global | -1 | 156 | -0.066246 | 0.093061 | 1.404786 | 0.000000 | 1.404786 | 0.000000 |
| rubber_band_extension | global | -1 | 156 | 0.003221 | 0.010401 | 3.228808 | 0.000000 | 3.228808 | 0.000000 |
| approach_velocity | global | -1 | 156 | 0.064084 | 0.205967 | 3.213984 | 0.000000 | 3.213984 | 0.000000 |
| forward_return_support | global | -1 | 132 | 0.151915 | 0.480398 | 3.162283 | 0.000000 | 3.162283 | 0.000000 |
| forward_return | global | -1 | 132 | 0.151915 | 0.480398 | 3.162283 | 0.000000 | 3.162283 | 0.000000 |
| forward_return | regime | 0 | 132 | 0.151915 | 0.480398 | 3.162283 | nan | nan | nan |
| forward_return_support | regime | 0 | 132 | 0.151915 | 0.480398 | 3.162283 | nan | nan | nan |
| approach_velocity | regime | 0 | 156 | 0.064084 | 0.205967 | 3.213984 | nan | nan | nan |
| rubber_band_extension | regime | 0 | 156 | 0.003221 | 0.010401 | 3.228808 | nan | nan | nan |
| momentum_divergence | regime | 0 | 156 | -0.066246 | 0.093061 | 1.404786 | nan | nan | nan |
| trend_strength_adx | regime | 0 | 156 | -0.331365 | 0.697486 | 2.104885 | nan | nan | nan |
| bollinger_squeeze | regime | 0 | 156 | -0.341582 | 0.454519 | 1.330628 | nan | nan | nan |
| volatility_compression | regime | 0 | 156 | -0.028982 | 0.355578 | 12.269087 | nan | nan | nan |
| inside_bar_chain | regime | 0 | 156 | 0.022611 | 0.710816 | 31.437278 | nan | nan | nan |
| test_count | regime | 0 | 156 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| age_log_hours | regime | 0 | 156 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| penetration_depth | regime | 0 | 156 | -0.020853 | 0.593805 | 28.476102 | nan | nan | nan |
| rejection_wick_ratio | regime | 0 | 156 | 0.030123 | 0.024647 | 0.818223 | nan | nan | nan |
| close_proximity | regime | 0 | 156 | -0.053871 | 0.618899 | 11.488579 | nan | nan | nan |
| volume_at_impact | regime | 0 | 156 | -0.077001 | 0.748612 | 9.722088 | nan | nan | nan |
| fakeout_ratio | regime | 0 | 156 | 0.012543 | 0.446975 | 35.636387 | nan | nan | nan |
| order_book_imbalance_proxy | regime | 0 | 156 | 0.047969 | 0.671232 | 13.993018 | nan | nan | nan |
| breakout_regime_0_prob | regime | 0 | 156 | 0.372885 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_regime_1_prob | regime | 0 | 156 | 0.313558 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_regime_2_prob | regime | 0 | 156 | 0.313558 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_long_edge_score | regime | 0 | 156 | 0.093221 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_short_edge_score | regime | 0 | 156 | -0.093221 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_bullish_prob | regime | 0 | 156 | 0.686442 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_bearish_prob | regime | 0 | 156 | 0.313558 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_level_strength | regime | 0 | 156 | 0.250000 | 0.000000 | 0.000000 | nan | nan | nan |

## Main Feature Drivers per Factor
For each factor, the table below lists the top features by absolute Spearman correlation with that factor.


### Factor: `forward_return_h24`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| trend_strength_adx | 0.3363 | 0.0918 | 132 |
| bollinger_squeeze | 0.3231 | 0.1621 | 132 |
| momentum_divergence | 0.2255 | 0.1121 | 132 |
| inside_bar_chain | 0.1993 | -0.0077 | 132 |
| approach_velocity | -0.1829 | 0.0021 | 132 |
| rubber_band_extension | -0.0696 | 0.2555 | 132 |
| close_proximity | 0.0471 | 0.0292 | 132 |
| fakeout_ratio | -0.0365 | 0.0114 | 132 |
| rejection_wick_ratio | 0.0330 | 0.2406 | 132 |
| penetration_depth | -0.0273 | -0.0199 | 132 |
| order_book_imbalance_proxy | -0.0264 | -0.0107 | 132 |
| volatility_compression | 0.0214 | 0.0001 | 132 |
| volume_at_impact | 0.0152 | 0.0140 | 132 |