# Breakout/Bounce Regime Diagnostics

- Symbol: **ETHUSDT**
- Exchange: **binance**
- Timeframe: **15m**
- Direction: **long**
- Horizon (bars): **16**
- Samples (training window): **158**

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
- Macro F1-score (val): **0.3514**
- Weighted F1-score (val): **0.3806**
- Sample split: train=110, val=24, test=24

### Class Counts (training labels)
| Regime | Count |
|--------|-------|
| 0 | 102 |
| 1 | 56 |

### Resistance/Support Hit Counts by Outcome
| Side | Class 0 | Total |
|--------|--------|--------|
| support | 158 | 158 |

## Meta-Label Success Summary
- Meta-labeled events: **158**, success=1: **17** (10.759% success rate)

| Class | Meta Events | Success Count | Success Rate |
|-------|------------|---------------|--------------|
| 0 | 102 | 14 | 13.725% |
| 1 | 56 | 3 | 5.357% |

## Breakout Success Probability & High-Confidence Gating
- Observations with breakout_success_prob: **158** | mean=0.500, std=0.000, p25=0.500, median=0.500, p75=0.500
- High-confidence signals (high_conf=1): **0** / 158 (0.000%)

### Forward Return Sharpe by Meta/High-Confidence Subset
| Subset | Samples | Mean Return | Std Return | Sharpe-like |
|--------|---------|-------------|------------|-------------|
| all | 142 | 0.079051 | 0.331943 | 0.2381 |
| meta_success==1 | 17 | 0.384138 | 0.430436 | 0.8924 |

## Forward Return Sharpe-like Ratios
| Scope | Side | Regime | Mean Return | Std Return | Sharpe-like |
|-------|------|--------|-------------|------------|-------------|
| global | all | -1 | 0.079051 | 0.331943 | 0.2381 |
| regime | all | 0 | 0.079051 | 0.331943 | 0.2381 |
| global | resistance | -1 | nan | nan | nan |
| regime | resistance | 0 | nan | nan | nan |
| global | support | -1 | 0.079051 | 0.331943 | 0.2381 |
| regime | support | 0 | 0.079051 | 0.331943 | 0.2381 |

## Per-Regime Summary (Forward Returns & Edge Scores)
| Regime | Side | Count | Mean Forward Return | Sharpe-like | Mean Long Edge | Mean Short Edge | Mean Bullish Prob | Mean Bearish Prob |
|--------|------|-------|---------------------|------------|---------------|----------------|-------------------|--------------------|
| 0 | all | 158 | 0.079051 | 0.2381 | 0.138785 | -0.138785 | 0.777571 | 0.222429 |
| 0 | support | 158 | 0.079051 | 0.2381 | 0.138785 | -0.138785 | 0.777571 | 0.222429 |

## Winsorised CV Between/Within Regimes
Definition: per metric, we winsorise values (5–95%), compute global and per-regime CV, and report the ratio of between-regime CV to within-regime CV.

| Metric | Scope | Regime | Count | Mean | Std | CV | CV_between | CV_within | CV_between/within |
|--------|-------|--------|-------|------|-----|----|-----------|----------|--------------------|
| breakout_level_strength | global | -1 | 158 | 0.250000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| breakout_bearish_prob | global | -1 | 158 | 0.222429 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| breakout_bullish_prob | global | -1 | 158 | 0.777571 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| breakout_short_edge_score | global | -1 | 158 | -0.138785 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| breakout_long_edge_score | global | -1 | 158 | 0.138785 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| breakout_regime_2_prob | global | -1 | 158 | 0.222429 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| breakout_regime_1_prob | global | -1 | 158 | 0.222429 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| breakout_regime_0_prob | global | -1 | 158 | 0.555141 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| order_book_imbalance_proxy | global | -1 | 158 | 0.047032 | 0.665048 | 14.140373 | 0.000000 | 14.140373 | 0.000000 |
| fakeout_ratio | global | -1 | 158 | 0.012564 | 0.442381 | 35.209820 | 0.000000 | 35.209820 | 0.000000 |
| volume_at_impact | global | -1 | 158 | -0.077090 | 0.739883 | 9.597630 | 0.000000 | 9.597630 | 0.000000 |
| close_proximity | global | -1 | 158 | -0.053780 | 0.613167 | 11.401392 | 0.000000 | 11.401392 | 0.000000 |
| rejection_wick_ratio | global | -1 | 158 | 0.030317 | 0.024509 | 0.808410 | 0.000000 | 0.808410 | 0.000000 |
| penetration_depth | global | -1 | 158 | -0.021171 | 0.588252 | 27.785169 | 0.000000 | 27.785169 | 0.000000 |
| age_log_hours | global | -1 | 158 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| test_count | global | -1 | 158 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| inside_bar_chain | global | -1 | 158 | 0.022319 | 0.706283 | 31.645202 | 0.000000 | 31.645202 | 0.000000 |
| volatility_compression | global | -1 | 158 | -0.028747 | 0.352887 | 12.275734 | 0.000000 | 12.275734 | 0.000000 |
| bollinger_squeeze | global | -1 | 158 | -0.337375 | 0.452886 | 1.342384 | 0.000000 | 1.342384 | 0.000000 |
| trend_strength_adx | global | -1 | 158 | -0.327180 | 0.693536 | 2.119738 | 0.000000 | 2.119738 | 0.000000 |
| momentum_divergence | global | -1 | 158 | -0.065394 | 0.092737 | 1.418121 | 0.000000 | 1.418121 | 0.000000 |
| rubber_band_extension | global | -1 | 158 | 0.003180 | 0.010339 | 3.250712 | 0.000000 | 3.250712 | 0.000000 |
| approach_velocity | global | -1 | 158 | 0.063205 | 0.204601 | 3.237113 | 0.000000 | 3.237113 | 0.000000 |
| forward_return_support | global | -1 | 142 | 0.069542 | 0.306345 | 4.405195 | 0.000000 | 4.405195 | 0.000000 |
| forward_return | global | -1 | 142 | 0.069542 | 0.306345 | 4.405195 | 0.000000 | 4.405195 | 0.000000 |
| forward_return | regime | 0 | 142 | 0.069542 | 0.306345 | 4.405195 | nan | nan | nan |
| forward_return_support | regime | 0 | 142 | 0.069542 | 0.306345 | 4.405195 | nan | nan | nan |
| approach_velocity | regime | 0 | 158 | 0.063205 | 0.204601 | 3.237113 | nan | nan | nan |
| rubber_band_extension | regime | 0 | 158 | 0.003180 | 0.010339 | 3.250712 | nan | nan | nan |
| momentum_divergence | regime | 0 | 158 | -0.065394 | 0.092737 | 1.418121 | nan | nan | nan |
| trend_strength_adx | regime | 0 | 158 | -0.327180 | 0.693536 | 2.119738 | nan | nan | nan |
| bollinger_squeeze | regime | 0 | 158 | -0.337375 | 0.452886 | 1.342384 | nan | nan | nan |
| volatility_compression | regime | 0 | 158 | -0.028747 | 0.352887 | 12.275734 | nan | nan | nan |
| inside_bar_chain | regime | 0 | 158 | 0.022319 | 0.706283 | 31.645202 | nan | nan | nan |
| test_count | regime | 0 | 158 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| age_log_hours | regime | 0 | 158 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| penetration_depth | regime | 0 | 158 | -0.021171 | 0.588252 | 27.785169 | nan | nan | nan |
| rejection_wick_ratio | regime | 0 | 158 | 0.030317 | 0.024509 | 0.808410 | nan | nan | nan |
| close_proximity | regime | 0 | 158 | -0.053780 | 0.613167 | 11.401392 | nan | nan | nan |
| volume_at_impact | regime | 0 | 158 | -0.077090 | 0.739883 | 9.597630 | nan | nan | nan |
| fakeout_ratio | regime | 0 | 158 | 0.012564 | 0.442381 | 35.209820 | nan | nan | nan |
| order_book_imbalance_proxy | regime | 0 | 158 | 0.047032 | 0.665048 | 14.140373 | nan | nan | nan |
| breakout_regime_0_prob | regime | 0 | 158 | 0.555141 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_regime_1_prob | regime | 0 | 158 | 0.222429 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_regime_2_prob | regime | 0 | 158 | 0.222429 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_long_edge_score | regime | 0 | 158 | 0.138785 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_short_edge_score | regime | 0 | 158 | -0.138785 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_bullish_prob | regime | 0 | 158 | 0.777571 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_bearish_prob | regime | 0 | 158 | 0.222429 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_level_strength | regime | 0 | 158 | 0.250000 | 0.000000 | 0.000000 | nan | nan | nan |

## Main Feature Drivers per Factor
For each factor, the table below lists the top features by absolute Spearman correlation with that factor.


### Factor: `forward_return_h16`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| trend_strength_adx | 0.3532 | 0.1083 | 142 |
| bollinger_squeeze | 0.2678 | 0.1474 | 142 |
| rubber_band_extension | -0.2566 | 0.0372 | 142 |
| momentum_divergence | 0.1690 | 0.0943 | 142 |
| inside_bar_chain | 0.1392 | -0.0115 | 142 |
| approach_velocity | -0.1244 | -0.0062 | 142 |
| fakeout_ratio | -0.0834 | 0.0015 | 142 |
| volume_at_impact | -0.0424 | 0.0144 | 142 |
| penetration_depth | -0.0288 | -0.0216 | 142 |
| rejection_wick_ratio | -0.0284 | 0.1021 | 142 |
| close_proximity | 0.0250 | 0.0172 | 142 |
| volatility_compression | -0.0203 | 0.0073 | 142 |
| order_book_imbalance_proxy | 0.0018 | -0.0107 | 142 |