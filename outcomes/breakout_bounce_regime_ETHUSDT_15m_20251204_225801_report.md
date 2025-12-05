# Breakout/Bounce Regime Diagnostics

- Symbol: **ETHUSDT**
- Exchange: **binance**
- Timeframe: **15m**
- Direction: **long**
- Horizon (bars): **24**
- Samples (training window): **124**

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
- Macro F1-score (val): **0.2083**
- Weighted F1-score (val): **0.1096**
- Sample split: train=86, val=19, test=19

### Class Counts (training labels)
| Regime | Count |
|--------|-------|
| 0 | 69 |
| 1 | 55 |

### Resistance/Support Hit Counts by Outcome
| Side | Class 0 | Total |
|--------|--------|--------|
| support | 124 | 124 |

## S/R Quality Metrics - Break vs Bounce Analysis

This section analyzes how SR level quality metrics (touch count, volume, prominence)
correlate with break vs bounce outcomes. Higher quantiles should show better bounce rates.

### Sr Touch Count

| Quantile | Count | Metric Range | Bounce Rate | Break Rate | Trap Rate | Bounce/Break Ratio |
|----------|-------|--------------|-------------|------------|-----------|-------------------|
| Q3 | 38 | 19.00 - 27.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q2 | 20 | 18.00 - 18.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q5 | 24 | 49.00 - 51.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q4 | 11 | 31.00 - 31.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q1 | 31 | 11.00 - 15.00 | 100.0% | 0.0% | 0.0% | ∞ |

**Interpretation:**
- Higher quantiles with higher bounce/break ratios indicate better quality S/R levels
- Use this data to adjust sr_combined_strength weights in `_add_sr_strength_features()`

### Sr Volume Depth Ratio

| Quantile | Count | Metric Range | Bounce Rate | Break Rate | Trap Rate | Bounce/Break Ratio |
|----------|-------|--------------|-------------|------------|-----------|-------------------|
| Q3 | 34 | 1.32 - 1.33 | 100.0% | 0.0% | 0.0% | ∞ |
| Q1 | 30 | 0.75 - 1.13 | 100.0% | 0.0% | 0.0% | ∞ |
| Q4 | 6 | 1.48 - 1.48 | 100.0% | 0.0% | 0.0% | ∞ |
| Q2 | 30 | 1.19 - 1.31 | 100.0% | 0.0% | 0.0% | ∞ |
| Q5 | 24 | 1.86 - 2.63 | 100.0% | 0.0% | 0.0% | ∞ |

**Interpretation:**
- Higher quantiles with higher bounce/break ratios indicate better quality S/R levels
- Use this data to adjust sr_combined_strength weights in `_add_sr_strength_features()`

### Sr Prominence

| Quantile | Count | Metric Range | Bounce Rate | Break Rate | Trap Rate | Bounce/Break Ratio |
|----------|-------|--------------|-------------|------------|-----------|-------------------|
| Q2 | 29 | 0.00 - 0.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q3 | 24 | 0.00 - 0.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q5 | 17 | 0.01 - 0.01 | 100.0% | 0.0% | 0.0% | ∞ |
| Q4 | 18 | 0.00 - 0.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q1 | 36 | 0.00 - 0.00 | 100.0% | 0.0% | 0.0% | ∞ |

**Interpretation:**
- Higher quantiles with higher bounce/break ratios indicate better quality S/R levels
- Use this data to adjust sr_combined_strength weights in `_add_sr_strength_features()`


## Meta-Label Success Summary
- Meta-labeled events: **124**, success=1: **2** (1.613% success rate)

| Class | Meta Events | Success Count | Success Rate |
|-------|------------|---------------|--------------|
| 0 | 69 | 2 | 2.899% |
| 1 | 55 | 0 | 0.000% |

## Breakout Success Probability & High-Confidence Gating
- Observations with breakout_success_prob: **124** | mean=0.500, std=0.000, p25=0.500, median=0.500, p75=0.500
- High-confidence signals (high_conf=1): **0** / 124 (0.000%)

### Forward Return Sharpe by Meta/High-Confidence Subset
| Subset | Samples | Mean Return | Std Return | Sharpe-like |
|--------|---------|-------------|------------|-------------|
| all | 100 | -0.015331 | 0.036092 | -0.4248 |
| meta_success==1 | 0 | nan | nan | nan |

## Forward Return Sharpe-like Ratios
| Scope | Side | Regime | Mean Return | Std Return | Sharpe-like |
|-------|------|--------|-------------|------------|-------------|
| global | all | -1 | -0.015331 | 0.036092 | -0.4248 |
| regime | all | 0 | -0.015331 | 0.036092 | -0.4248 |
| global | resistance | -1 | nan | nan | nan |
| regime | resistance | 0 | nan | nan | nan |
| global | support | -1 | -0.015331 | 0.036092 | -0.4248 |
| regime | support | 0 | -0.015331 | 0.036092 | -0.4248 |

## Per-Regime Summary (Forward Returns & Edge Scores)
| Regime | Side | Count | Mean Forward Return | Sharpe-like | Mean Long Edge | Mean Short Edge | Mean Bullish Prob | Mean Bearish Prob |
|--------|------|-------|---------------------|------------|---------------|----------------|-------------------|--------------------|
| 0 | all | 124 | -0.015331 | -0.4248 | 0.101865 | -0.101865 | 0.686442 | 0.313558 |
| 0 | support | 124 | -0.015331 | -0.4248 | 0.101865 | -0.101865 | 0.686442 | 0.313558 |

## Winsorised CV Between/Within Regimes
Definition: per metric, we winsorise values (5–95%), compute global and per-regime CV, and report the ratio of between-regime CV to within-regime CV.

| Metric | Scope | Regime | Count | Mean | Std | CV | CV_between | CV_within | CV_between/within |
|--------|-------|--------|-------|------|-----|----|-----------|----------|--------------------|
| breakout_level_strength | global | -1 | 124 | 0.272353 | 0.048289 | 0.177302 | 0.000000 | 0.177302 | 0.000000 |
| breakout_bearish_prob | global | -1 | 124 | 0.313558 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| breakout_bullish_prob | global | -1 | 124 | 0.686442 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| breakout_short_edge_score | global | -1 | 124 | -0.101556 | 0.018006 | 0.177302 | 0.000000 | 0.177302 | 0.000000 |
| breakout_long_edge_score | global | -1 | 124 | 0.101556 | 0.018006 | 0.177302 | 0.000000 | 0.177302 | 0.000000 |
| breakout_regime_2_prob | global | -1 | 124 | 0.313558 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| breakout_regime_1_prob | global | -1 | 124 | 0.313558 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| breakout_regime_0_prob | global | -1 | 124 | 0.372885 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| order_book_imbalance_proxy | global | -1 | 124 | -0.056150 | 0.307838 | 5.482384 | 0.000000 | 5.482384 | 0.000000 |
| fakeout_ratio | global | -1 | 124 | 0.077889 | 0.204575 | 2.626492 | 0.000000 | 2.626492 | 0.000000 |
| volume_at_impact | global | -1 | 124 | -0.017374 | 0.181371 | 10.439498 | 0.000000 | 10.439498 | 0.000000 |
| close_proximity | global | -1 | 124 | 0.111674 | 0.286850 | 2.568643 | 0.000000 | 2.568643 | 0.000000 |
| rejection_wick_ratio | global | -1 | 124 | 0.037988 | 0.026828 | 0.706220 | 0.000000 | 0.706220 | 0.000000 |
| penetration_depth | global | -1 | 124 | 0.064665 | 0.173715 | 2.686403 | 0.000000 | 2.686403 | 0.000000 |
| age_log_hours | global | -1 | 124 | 0.019694 | 0.075990 | 3.858528 | 0.000000 | 3.858528 | 0.000000 |
| test_count | global | -1 | 124 | -0.158741 | 0.349153 | 2.199514 | 0.000000 | 2.199514 | 0.000000 |
| inside_bar_chain | global | -1 | 124 | -0.061218 | 0.143915 | 2.350853 | 0.000000 | 2.350853 | 0.000000 |
| volatility_compression | global | -1 | 124 | -0.067774 | 0.242229 | 3.574074 | 0.000000 | 3.574074 | 0.000000 |
| bollinger_squeeze | global | -1 | 124 | 0.068996 | 0.167869 | 2.433018 | 0.000000 | 2.433018 | 0.000000 |
| trend_strength_adx | global | -1 | 124 | -0.041235 | 0.112934 | 2.738831 | 0.000000 | 2.738831 | 0.000000 |
| momentum_divergence | global | -1 | 124 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| rubber_band_extension | global | -1 | 124 | -0.000730 | 0.001455 | 1.993257 | 0.000000 | 1.993257 | 0.000000 |
| approach_velocity | global | -1 | 124 | 0.005063 | 0.027997 | 5.529836 | 0.000000 | 5.529836 | 0.000000 |
| forward_return_support | global | -1 | 100 | -0.014949 | 0.034542 | 2.310736 | 0.000000 | 2.310736 | 0.000000 |
| forward_return | global | -1 | 100 | -0.014949 | 0.034542 | 2.310736 | 0.000000 | 2.310736 | 0.000000 |
| forward_return | regime | 0 | 100 | -0.014949 | 0.034542 | 2.310736 | nan | nan | nan |
| forward_return_support | regime | 0 | 100 | -0.014949 | 0.034542 | 2.310736 | nan | nan | nan |
| approach_velocity | regime | 0 | 124 | 0.005063 | 0.027997 | 5.529836 | nan | nan | nan |
| rubber_band_extension | regime | 0 | 124 | -0.000730 | 0.001455 | 1.993257 | nan | nan | nan |
| momentum_divergence | regime | 0 | 124 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| trend_strength_adx | regime | 0 | 124 | -0.041235 | 0.112934 | 2.738831 | nan | nan | nan |
| bollinger_squeeze | regime | 0 | 124 | 0.068996 | 0.167869 | 2.433018 | nan | nan | nan |
| volatility_compression | regime | 0 | 124 | -0.067774 | 0.242229 | 3.574074 | nan | nan | nan |
| inside_bar_chain | regime | 0 | 124 | -0.061218 | 0.143915 | 2.350853 | nan | nan | nan |
| test_count | regime | 0 | 124 | -0.158741 | 0.349153 | 2.199514 | nan | nan | nan |
| age_log_hours | regime | 0 | 124 | 0.019694 | 0.075990 | 3.858528 | nan | nan | nan |
| penetration_depth | regime | 0 | 124 | 0.064665 | 0.173715 | 2.686403 | nan | nan | nan |
| rejection_wick_ratio | regime | 0 | 124 | 0.037988 | 0.026828 | 0.706220 | nan | nan | nan |
| close_proximity | regime | 0 | 124 | 0.111674 | 0.286850 | 2.568643 | nan | nan | nan |
| volume_at_impact | regime | 0 | 124 | -0.017374 | 0.181371 | 10.439498 | nan | nan | nan |
| fakeout_ratio | regime | 0 | 124 | 0.077889 | 0.204575 | 2.626492 | nan | nan | nan |
| order_book_imbalance_proxy | regime | 0 | 124 | -0.056150 | 0.307838 | 5.482384 | nan | nan | nan |
| breakout_regime_0_prob | regime | 0 | 124 | 0.372885 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_regime_1_prob | regime | 0 | 124 | 0.313558 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_regime_2_prob | regime | 0 | 124 | 0.313558 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_long_edge_score | regime | 0 | 124 | 0.101556 | 0.018006 | 0.177302 | nan | nan | nan |
| breakout_short_edge_score | regime | 0 | 124 | -0.101556 | 0.018006 | 0.177302 | nan | nan | nan |
| breakout_bullish_prob | regime | 0 | 124 | 0.686442 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_bearish_prob | regime | 0 | 124 | 0.313558 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_level_strength | regime | 0 | 124 | 0.272353 | 0.048289 | 0.177302 | nan | nan | nan |

## Main Feature Drivers per Factor
For each factor, the table below lists the top features by absolute Spearman correlation with that factor.


### Factor: `forward_return_h24`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| rubber_band_extension | -0.3885 | -0.3328 | 100 |
| rejection_wick_ratio | 0.1824 | 0.1582 | 100 |

### Factor: `breakout_long_edge_score`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| test_count | -0.9882 | -0.9472 | 124 |
| bollinger_squeeze | 0.9241 | 0.6893 | 124 |
| fakeout_ratio | 0.8258 | 0.6291 | 124 |
| inside_bar_chain | -0.7270 | -0.0896 | 124 |
| rubber_band_extension | 0.6561 | 0.6432 | 124 |
| close_proximity | 0.6525 | 0.7578 | 124 |
| penetration_depth | 0.6492 | 0.6040 | 124 |
| trend_strength_adx | -0.5351 | -0.3969 | 124 |
| volatility_compression | -0.3545 | -0.0563 | 124 |
| order_book_imbalance_proxy | -0.2018 | -0.2314 | 124 |
| volume_at_impact | -0.1859 | -0.0185 | 124 |
| approach_velocity | 0.1747 | 0.2037 | 124 |
| age_log_hours | 0.1525 | 0.1035 | 124 |
| rejection_wick_ratio | -0.1292 | -0.1335 | 124 |

### Factor: `breakout_short_edge_score`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| test_count | 0.9882 | 0.9472 | 124 |
| bollinger_squeeze | -0.9241 | -0.6893 | 124 |
| fakeout_ratio | -0.8258 | -0.6291 | 124 |
| inside_bar_chain | 0.7270 | 0.0896 | 124 |
| rubber_band_extension | -0.6561 | -0.6432 | 124 |
| close_proximity | -0.6525 | -0.7578 | 124 |
| penetration_depth | -0.6492 | -0.6040 | 124 |
| trend_strength_adx | 0.5351 | 0.3969 | 124 |
| volatility_compression | 0.3545 | 0.0563 | 124 |
| order_book_imbalance_proxy | 0.2018 | 0.2314 | 124 |
| volume_at_impact | 0.1859 | 0.0185 | 124 |
| approach_velocity | -0.1747 | -0.2037 | 124 |
| age_log_hours | -0.1525 | -0.1035 | 124 |
| rejection_wick_ratio | 0.1292 | 0.1335 | 124 |