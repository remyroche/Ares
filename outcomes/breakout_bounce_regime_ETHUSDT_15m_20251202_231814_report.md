# Breakout/Bounce Regime Diagnostics

- Symbol: **ETHUSDT**
- Exchange: **binance**
- Timeframe: **15m**
- Direction: **long**
- Horizon (bars): **48**
- Samples (training window): **246**

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
- Validation log loss: **0.547116**
- Test log loss: **nan**
- Generalization gap (test - val log loss): **nan**
- Macro F1-score (val): **0.4932**
- Weighted F1-score (val): **0.9596**
- Sample split: train=172, val=37, test=37

### Class Counts (training labels)
| Regime | Count |
|--------|-------|
| 0 | 73 |
| 1 | 80 |
| 2 | 47 |
| 3 | 46 |

### Resistance/Support Hit Counts by Outcome
| Side | Class 0 | Total |
|--------|--------|--------|
| resistance | 102 | 102 |
| support | 144 | 144 |

## S/R Quality Metrics - Break vs Bounce Analysis

This section analyzes how SR level quality metrics (touch count, volume, prominence)
correlate with break vs bounce outcomes. Higher quantiles should show better bounce rates.

### Sr Touch Count

| Quantile | Count | Metric Range | Bounce Rate | Break Rate | Trap Rate | Bounce/Break Ratio |
|----------|-------|--------------|-------------|------------|-----------|-------------------|
| Q4 | 67 | 16.00 - 25.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q5 | 28 | 42.00 - 42.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q2 | 35 | 9.00 - 9.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q1 | 81 | 2.00 - 8.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q3 | 35 | 13.00 - 13.00 | 100.0% | 0.0% | 0.0% | ∞ |

**Interpretation:**
- Higher quantiles with higher bounce/break ratios indicate better quality S/R levels
- Use this data to adjust sr_combined_strength weights in `_add_sr_strength_features()`

### Sr Volume Depth Ratio

| Quantile | Count | Metric Range | Bounce Rate | Break Rate | Trap Rate | Bounce/Break Ratio |
|----------|-------|--------------|-------------|------------|-----------|-------------------|
| Q1 | 50 | 0.98 - 1.08 | 100.0% | 0.0% | 0.0% | ∞ |
| Q2 | 63 | 1.12 - 1.20 | 100.0% | 0.0% | 0.0% | ∞ |
| Q4 | 35 | 1.57 - 1.96 | 100.0% | 0.0% | 0.0% | ∞ |
| Q5 | 29 | 2.05 - 10.07 | 100.0% | 0.0% | 0.0% | ∞ |
| Q3 | 69 | 1.22 - 1.35 | 100.0% | 0.0% | 0.0% | ∞ |

**Interpretation:**
- Higher quantiles with higher bounce/break ratios indicate better quality S/R levels
- Use this data to adjust sr_combined_strength weights in `_add_sr_strength_features()`

### Sr Prominence

| Quantile | Count | Metric Range | Bounce Rate | Break Rate | Trap Rate | Bounce/Break Ratio |
|----------|-------|--------------|-------------|------------|-----------|-------------------|
| Q2 | 82 | 0.00 - 0.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q4 | 46 | 0.00 - 0.01 | 100.0% | 0.0% | 0.0% | ∞ |
| Q5 | 30 | 0.01 - 0.01 | 100.0% | 0.0% | 0.0% | ∞ |
| Q1 | 53 | 0.00 - 0.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q3 | 35 | 0.00 - 0.00 | 100.0% | 0.0% | 0.0% | ∞ |

**Interpretation:**
- Higher quantiles with higher bounce/break ratios indicate better quality S/R levels
- Use this data to adjust sr_combined_strength weights in `_add_sr_strength_features()`


## Meta-Label Success Summary
- Meta-labeled events: **246**, success=1: **54** (21.951% success rate)

| Class | Meta Events | Success Count | Success Rate |
|-------|------------|---------------|--------------|
| 0 | 73 | 4 | 5.479% |
| 1 | 80 | 2 | 2.500% |
| 2 | 47 | 4 | 8.511% |
| 3 | 46 | 44 | 95.652% |

## Breakout Success Probability & High-Confidence Gating
- Observations with breakout_success_prob: **246** | mean=0.500, std=0.000, p25=0.500, median=0.500, p75=0.500
- High-confidence signals (high_conf=1): **0** / 246 (0.000%)

### Forward Return Sharpe by Meta/High-Confidence Subset
| Subset | Samples | Mean Return | Std Return | Sharpe-like |
|--------|---------|-------------|------------|-------------|
| all | 198 | -0.022400 | 0.048886 | -0.4582 |
| meta_success==1 | 52 | 0.020134 | 0.026655 | 0.7554 |

## Forward Return Sharpe-like Ratios
| Scope | Side | Regime | Mean Return | Std Return | Sharpe-like |
|-------|------|--------|-------------|------------|-------------|
| global | all | -1 | -0.022400 | 0.048886 | -0.4582 |
| regime | all | 0 | -0.022400 | 0.048886 | -0.4582 |
| global | resistance | -1 | -0.032171 | 0.061034 | -0.5271 |
| regime | resistance | 0 | -0.032171 | 0.061034 | -0.5271 |
| global | support | -1 | -0.016569 | 0.038756 | -0.4275 |
| regime | support | 0 | -0.016569 | 0.038756 | -0.4275 |

## Per-Regime Summary (Forward Returns & Edge Scores)
| Regime | Side | Count | Mean Forward Return | Sharpe-like | Mean Long Edge | Mean Short Edge | Mean Bullish Prob | Mean Bearish Prob |
|--------|------|-------|---------------------|------------|---------------|----------------|-------------------|--------------------|
| 0 | all | 246 | -0.022400 | -0.4582 | 0.025209 | -0.025209 | 0.550067 | 0.449933 |
| 0 | resistance | 102 | -0.032171 | -0.5271 | -0.143855 | 0.143855 | 0.206749 | 0.793251 |
| 0 | support | 144 | -0.016569 | -0.4275 | 0.144962 | -0.144962 | 0.793251 | 0.206749 |

## Winsorised CV Between/Within Regimes
Definition: per metric, we winsorise values (5–95%), compute global and per-regime CV, and report the ratio of between-regime CV to within-regime CV.

| Metric | Scope | Regime | Count | Mean | Std | CV | CV_between | CV_within | CV_between/within |
|--------|-------|--------|-------|------|-----|----|-----------|----------|--------------------|
| breakout_level_strength | global | -1 | 246 | 0.244303 | 0.013871 | 0.056777 | 0.000000 | 0.056777 | 0.000000 |
| breakout_bearish_prob | global | -1 | 246 | 0.449933 | 0.288946 | 0.642198 | 0.000000 | 0.642198 | 0.000000 |
| breakout_bullish_prob | global | -1 | 246 | 0.550067 | 0.288946 | 0.525292 | 0.000000 | 0.525292 | 0.000000 |
| breakout_short_edge_score | global | -1 | 246 | -0.026269 | 0.140967 | 5.366321 | 0.000000 | 5.366321 | 0.000000 |
| breakout_long_edge_score | global | -1 | 246 | 0.026269 | 0.140967 | 5.366321 | 0.000000 | 5.366321 | 0.000000 |
| breakout_regime_2_prob | global | -1 | 246 | 0.329159 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| breakout_regime_1_prob | global | -1 | 246 | 0.206749 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| breakout_regime_0_prob | global | -1 | 246 | 0.464093 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| order_book_imbalance_proxy | global | -1 | 246 | 0.030913 | 0.269213 | 8.708799 | 0.000000 | 8.708799 | 0.000000 |
| fakeout_ratio | global | -1 | 246 | 0.056494 | 0.155029 | 2.744163 | 0.000000 | 2.744163 | 0.000000 |
| volume_at_impact | global | -1 | 246 | 0.005709 | 0.133931 | 23.458272 | 0.000000 | 23.458272 | 0.000000 |
| close_proximity | global | -1 | 246 | 0.127338 | 0.303840 | 2.386089 | 0.000000 | 2.386089 | 0.000000 |
| rejection_wick_ratio | global | -1 | 246 | 0.031041 | 0.024000 | 0.773171 | 0.000000 | 0.773171 | 0.000000 |
| penetration_depth | global | -1 | 246 | 0.046760 | 0.152298 | 3.256977 | 0.000000 | 3.256977 | 0.000000 |
| age_log_hours | global | -1 | 246 | 0.028615 | 0.082979 | 2.899903 | 0.000000 | 2.899903 | 0.000000 |
| test_count | global | -1 | 246 | -0.107031 | 0.243343 | 2.273564 | 0.000000 | 2.273564 | 0.000000 |
| inside_bar_chain | global | -1 | 246 | -0.067440 | 0.155377 | 2.303911 | 0.000000 | 2.303911 | 0.000000 |
| volatility_compression | global | -1 | 246 | -0.095297 | 0.261411 | 2.743105 | 0.000000 | 2.743105 | 0.000000 |
| bollinger_squeeze | global | -1 | 246 | 0.223425 | 0.668780 | 2.993308 | 0.000000 | 2.993308 | 0.000000 |
| trend_strength_adx | global | -1 | 246 | 0.106149 | 0.270163 | 2.545130 | 0.000000 | 2.545130 | 0.000000 |
| momentum_divergence | global | -1 | 246 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| rubber_band_extension | global | -1 | 246 | -0.000573 | 0.001609 | 2.810259 | 0.000000 | 2.810259 | 0.000000 |
| approach_velocity | global | -1 | 246 | 0.038161 | 0.124845 | 3.271542 | 0.000000 | 3.271542 | 0.000000 |
| forward_return_support | global | -1 | 124 | -0.016471 | 0.038300 | 2.325253 | 0.000000 | 2.325253 | 0.000000 |
| forward_return_resistance | global | -1 | 74 | -0.032216 | 0.060482 | 1.877394 | 0.000000 | 1.877394 | 0.000000 |
| forward_return | global | -1 | 198 | -0.022071 | 0.047743 | 2.163139 | 0.000000 | 2.163139 | 0.000000 |
| forward_return | regime | 0 | 198 | -0.022071 | 0.047743 | 2.163139 | nan | nan | nan |
| forward_return_resistance | regime | 0 | 74 | -0.032216 | 0.060482 | 1.877394 | nan | nan | nan |
| forward_return_support | regime | 0 | 124 | -0.016471 | 0.038300 | 2.325253 | nan | nan | nan |
| approach_velocity | regime | 0 | 246 | 0.038161 | 0.124845 | 3.271542 | nan | nan | nan |
| rubber_band_extension | regime | 0 | 246 | -0.000573 | 0.001609 | 2.810259 | nan | nan | nan |
| momentum_divergence | regime | 0 | 246 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| trend_strength_adx | regime | 0 | 246 | 0.106149 | 0.270163 | 2.545130 | nan | nan | nan |
| bollinger_squeeze | regime | 0 | 246 | 0.223425 | 0.668780 | 2.993308 | nan | nan | nan |
| volatility_compression | regime | 0 | 246 | -0.095297 | 0.261411 | 2.743105 | nan | nan | nan |
| inside_bar_chain | regime | 0 | 246 | -0.067440 | 0.155377 | 2.303911 | nan | nan | nan |
| test_count | regime | 0 | 246 | -0.107031 | 0.243343 | 2.273564 | nan | nan | nan |
| age_log_hours | regime | 0 | 246 | 0.028615 | 0.082979 | 2.899903 | nan | nan | nan |
| penetration_depth | regime | 0 | 246 | 0.046760 | 0.152298 | 3.256977 | nan | nan | nan |
| rejection_wick_ratio | regime | 0 | 246 | 0.031041 | 0.024000 | 0.773171 | nan | nan | nan |
| close_proximity | regime | 0 | 246 | 0.127338 | 0.303840 | 2.386089 | nan | nan | nan |
| volume_at_impact | regime | 0 | 246 | 0.005709 | 0.133931 | 23.458272 | nan | nan | nan |
| fakeout_ratio | regime | 0 | 246 | 0.056494 | 0.155029 | 2.744163 | nan | nan | nan |
| order_book_imbalance_proxy | regime | 0 | 246 | 0.030913 | 0.269213 | 8.708799 | nan | nan | nan |
| breakout_regime_0_prob | regime | 0 | 246 | 0.464093 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_regime_1_prob | regime | 0 | 246 | 0.206749 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_regime_2_prob | regime | 0 | 246 | 0.329159 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_long_edge_score | regime | 0 | 246 | 0.026269 | 0.140967 | 5.366321 | nan | nan | nan |
| breakout_short_edge_score | regime | 0 | 246 | -0.026269 | 0.140967 | 5.366321 | nan | nan | nan |
| breakout_bullish_prob | regime | 0 | 246 | 0.550067 | 0.288946 | 0.525292 | nan | nan | nan |
| breakout_bearish_prob | regime | 0 | 246 | 0.449933 | 0.288946 | 0.642198 | nan | nan | nan |
| breakout_level_strength | regime | 0 | 246 | 0.244303 | 0.013871 | 0.056777 | nan | nan | nan |

## Main Feature Drivers per Factor
For each factor, the table below lists the top features by absolute Spearman correlation with that factor.


### Factor: `forward_return_h48`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| rejection_wick_ratio | 0.2543 | 0.2773 | 198 |
| rubber_band_extension | -0.0088 | -0.0124 | 198 |

### Factor: `breakout_long_edge_score`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| rubber_band_extension | -0.2740 | -0.1406 | 246 |
| test_count | 0.2033 | 0.2892 | 246 |
| close_proximity | -0.1778 | -0.2443 | 246 |
| trend_strength_adx | -0.1607 | -0.2827 | 246 |
| fakeout_ratio | -0.1575 | -0.1776 | 246 |
| bollinger_squeeze | -0.1574 | -0.2895 | 246 |
| rejection_wick_ratio | 0.1456 | 0.1706 | 246 |
| approach_velocity | 0.1043 | 0.2204 | 246 |
| order_book_imbalance_proxy | -0.1006 | -0.1374 | 246 |
| age_log_hours | -0.0795 | 0.0397 | 246 |
| inside_bar_chain | 0.0762 | -0.0716 | 246 |
| volatility_compression | 0.0612 | -0.1298 | 246 |
| volume_at_impact | -0.0545 | -0.2059 | 246 |
| penetration_depth | -0.0301 | -0.1504 | 246 |

### Factor: `breakout_short_edge_score`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| rubber_band_extension | 0.2740 | 0.1406 | 246 |
| test_count | -0.2033 | -0.2892 | 246 |
| close_proximity | 0.1778 | 0.2443 | 246 |
| trend_strength_adx | 0.1607 | 0.2827 | 246 |
| fakeout_ratio | 0.1575 | 0.1776 | 246 |
| bollinger_squeeze | 0.1574 | 0.2895 | 246 |
| rejection_wick_ratio | -0.1456 | -0.1706 | 246 |
| approach_velocity | -0.1043 | -0.2204 | 246 |
| order_book_imbalance_proxy | 0.1006 | 0.1374 | 246 |
| age_log_hours | 0.0795 | -0.0397 | 246 |
| inside_bar_chain | -0.0762 | 0.0716 | 246 |
| volatility_compression | -0.0612 | 0.1298 | 246 |
| volume_at_impact | 0.0545 | 0.2059 | 246 |
| penetration_depth | 0.0301 | 0.1504 | 246 |