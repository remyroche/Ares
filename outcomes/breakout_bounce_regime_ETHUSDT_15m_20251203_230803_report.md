# Breakout/Bounce Regime Diagnostics

- Symbol: **ETHUSDT**
- Exchange: **binance**
- Timeframe: **15m**
- Direction: **long**
- Horizon (bars): **96**
- Samples (training window): **225**

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
- Macro F1-score (val): **1.0000**
- Weighted F1-score (val): **1.0000**
- Sample split: train=157, val=34, test=34

### Class Counts (training labels)
| Regime | Count |
|--------|-------|
| 0 | 101 |
| 1 | 36 |
| 2 | 50 |
| 3 | 38 |

### Resistance/Support Hit Counts by Outcome
| Side | Class 0 | Total |
|--------|--------|--------|
| resistance | 126 | 126 |
| support | 99 | 99 |

## S/R Quality Metrics - Break vs Bounce Analysis

This section analyzes how SR level quality metrics (touch count, volume, prominence)
correlate with break vs bounce outcomes. Higher quantiles should show better bounce rates.

### Sr Touch Count

| Quantile | Count | Metric Range | Bounce Rate | Break Rate | Trap Rate | Bounce/Break Ratio |
|----------|-------|--------------|-------------|------------|-----------|-------------------|
| Q4 | 54 | 20.00 - 31.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q5 | 36 | 49.00 - 51.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q3 | 18 | 19.00 - 19.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q1 | 86 | 11.00 - 15.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q2 | 31 | 18.00 - 18.00 | 100.0% | 0.0% | 0.0% | ∞ |

**Interpretation:**
- Higher quantiles with higher bounce/break ratios indicate better quality S/R levels
- Use this data to adjust sr_combined_strength weights in `_add_sr_strength_features()`

### Sr Volume Depth Ratio

| Quantile | Count | Metric Range | Bounce Rate | Break Rate | Trap Rate | Bounce/Break Ratio |
|----------|-------|--------------|-------------|------------|-----------|-------------------|
| Q3 | 30 | 1.32 - 1.33 | 100.0% | 0.0% | 0.0% | ∞ |
| Q4 | 52 | 1.48 - 1.89 | 100.0% | 0.0% | 0.0% | ∞ |
| Q1 | 70 | 0.75 - 1.21 | 100.0% | 0.0% | 0.0% | ∞ |
| Q2 | 42 | 1.31 - 1.31 | 100.0% | 0.0% | 0.0% | ∞ |
| Q5 | 31 | 2.63 - 2.63 | 100.0% | 0.0% | 0.0% | ∞ |

**Interpretation:**
- Higher quantiles with higher bounce/break ratios indicate better quality S/R levels
- Use this data to adjust sr_combined_strength weights in `_add_sr_strength_features()`

### Sr Prominence

| Quantile | Count | Metric Range | Bounce Rate | Break Rate | Trap Rate | Bounce/Break Ratio |
|----------|-------|--------------|-------------|------------|-----------|-------------------|
| Q3 | 49 | 0.00 - 0.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q5 | 24 | 0.01 - 0.01 | 100.0% | 0.0% | 0.0% | ∞ |
| Q4 | 48 | 0.00 - 0.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q1 | 63 | 0.00 - 0.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q2 | 41 | 0.00 - 0.00 | 100.0% | 0.0% | 0.0% | ∞ |

**Interpretation:**
- Higher quantiles with higher bounce/break ratios indicate better quality S/R levels
- Use this data to adjust sr_combined_strength weights in `_add_sr_strength_features()`


## Meta-Label Success Summary
- Meta-labeled events: **225**, success=1: **11** (4.889% success rate)

| Class | Meta Events | Success Count | Success Rate |
|-------|------------|---------------|--------------|
| 0 | 101 | 4 | 3.960% |
| 1 | 36 | 0 | 0.000% |
| 2 | 50 | 0 | 0.000% |
| 3 | 38 | 7 | 18.421% |

## Breakout Success Probability & High-Confidence Gating
- Observations with breakout_success_prob: **225** | mean=0.500, std=0.000, p25=0.500, median=0.500, p75=0.500
- High-confidence signals (high_conf=1): **0** / 225 (0.000%)

### Forward Return Sharpe by Meta/High-Confidence Subset
| Subset | Samples | Mean Return | Std Return | Sharpe-like |
|--------|---------|-------------|------------|-------------|
| all | 129 | -0.052949 | 0.059943 | -0.8833 |
| meta_success==1 | 7 | 0.000702 | 0.001693 | 0.4148 |

## Forward Return Sharpe-like Ratios
| Scope | Side | Regime | Mean Return | Std Return | Sharpe-like |
|-------|------|--------|-------------|------------|-------------|
| global | all | -1 | -0.052949 | 0.059943 | -0.8833 |
| regime | all | 0 | -0.052949 | 0.059943 | -0.8833 |
| global | resistance | -1 | -0.060155 | 0.060488 | -0.9945 |
| regime | resistance | 0 | -0.060155 | 0.060488 | -0.9945 |
| global | support | -1 | -0.045401 | 0.058421 | -0.7771 |
| regime | support | 0 | -0.045401 | 0.058421 | -0.7771 |

## Per-Regime Summary (Forward Returns & Edge Scores)
| Regime | Side | Count | Mean Forward Return | Sharpe-like | Mean Long Edge | Mean Short Edge | Mean Bullish Prob | Mean Bearish Prob |
|--------|------|-------|---------------------|------------|---------------|----------------|-------------------|--------------------|
| 0 | all | 225 | -0.052949 | -0.8833 | -0.019771 | 0.019771 | 0.473314 | 0.526686 |
| 0 | resistance | 126 | -0.060155 | -0.9945 | -0.134564 | 0.134564 | 0.277614 | 0.722386 |
| 0 | support | 99 | -0.045401 | -0.7771 | 0.126329 | -0.126329 | 0.722386 | 0.277614 |

## Winsorised CV Between/Within Regimes
Definition: per metric, we winsorise values (5–95%), compute global and per-regime CV, and report the ratio of between-regime CV to within-regime CV.

| Metric | Scope | Regime | Count | Mean | Std | CV | CV_between | CV_within | CV_between/within |
|--------|-------|--------|-------|------|-----|----|-----------|----------|--------------------|
| breakout_level_strength | global | -1 | 225 | 0.293939 | 0.065483 | 0.222777 | 0.000000 | 0.222777 | 0.000000 |
| breakout_bearish_prob | global | -1 | 225 | 0.526686 | 0.220779 | 0.419185 | 0.000000 | 0.419185 | 0.000000 |
| breakout_bullish_prob | global | -1 | 225 | 0.473314 | 0.220779 | 0.466454 | 0.000000 | 0.466454 | 0.000000 |
| breakout_short_edge_score | global | -1 | 225 | 0.019831 | 0.132017 | 6.657053 | 0.000000 | 6.657053 | 0.000000 |
| breakout_long_edge_score | global | -1 | 225 | -0.019831 | 0.132017 | 6.657053 | 0.000000 | 6.657053 | 0.000000 |
| breakout_regime_2_prob | global | -1 | 225 | 0.245614 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| breakout_regime_1_prob | global | -1 | 225 | 0.277614 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| breakout_regime_0_prob | global | -1 | 225 | 0.476772 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| order_book_imbalance_proxy | global | -1 | 225 | -0.021702 | 0.455476 | 20.987874 | 0.000000 | 20.987874 | 0.000000 |
| fakeout_ratio | global | -1 | 225 | 0.113197 | 0.235607 | 2.081388 | 0.000000 | 2.081388 | 0.000000 |
| volume_at_impact | global | -1 | 225 | -0.011515 | 0.270728 | 23.511801 | 0.000000 | 23.511801 | 0.000000 |
| close_proximity | global | -1 | 225 | 0.169040 | 0.382332 | 2.261779 | 0.000000 | 2.261779 | 0.000000 |
| rejection_wick_ratio | global | -1 | 225 | 0.029500 | 0.019665 | 0.666617 | 0.000000 | 0.666617 | 0.000000 |
| penetration_depth | global | -1 | 225 | 0.064042 | 0.234384 | 3.659828 | 0.000000 | 3.659828 | 0.000000 |
| age_log_hours | global | -1 | 225 | 0.048114 | 0.166740 | 3.465555 | 0.000000 | 3.465555 | 0.000000 |
| test_count | global | -1 | 225 | -0.249816 | 0.388389 | 1.554701 | 0.000000 | 1.554701 | 0.000000 |
| inside_bar_chain | global | -1 | 225 | 0.010460 | 0.514541 | 49.189413 | 0.000000 | 49.189413 | 0.000000 |
| volatility_compression | global | -1 | 225 | -0.075815 | 0.359398 | 4.740436 | 0.000000 | 4.740436 | 0.000000 |
| bollinger_squeeze | global | -1 | 225 | 0.485669 | 1.031913 | 2.124726 | 0.000000 | 2.124726 | 0.000000 |
| trend_strength_adx | global | -1 | 225 | -0.023829 | 0.212843 | 8.932159 | 0.000000 | 8.932159 | 0.000000 |
| momentum_divergence | global | -1 | 225 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| rubber_band_extension | global | -1 | 225 | -0.000431 | 0.001183 | 2.744010 | 0.000000 | 2.744010 | 0.000000 |
| approach_velocity | global | -1 | 225 | -0.039686 | 0.176611 | 4.450151 | 0.000000 | 4.450151 | 0.000000 |
| forward_return_support | global | -1 | 63 | -0.045406 | 0.058316 | 1.284341 | 0.000000 | 1.284341 | 0.000000 |
| forward_return_resistance | global | -1 | 66 | -0.059985 | 0.059977 | 0.999855 | 0.000000 | 0.999855 | 0.000000 |
| forward_return | global | -1 | 129 | -0.052881 | 0.059636 | 1.127747 | 0.000000 | 1.127747 | 0.000000 |
| forward_return | regime | 0 | 129 | -0.052881 | 0.059636 | 1.127747 | nan | nan | nan |
| forward_return_resistance | regime | 0 | 66 | -0.059985 | 0.059977 | 0.999855 | nan | nan | nan |
| forward_return_support | regime | 0 | 63 | -0.045406 | 0.058316 | 1.284341 | nan | nan | nan |
| approach_velocity | regime | 0 | 225 | -0.039686 | 0.176611 | 4.450151 | nan | nan | nan |
| rubber_band_extension | regime | 0 | 225 | -0.000431 | 0.001183 | 2.744010 | nan | nan | nan |
| momentum_divergence | regime | 0 | 225 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| trend_strength_adx | regime | 0 | 225 | -0.023829 | 0.212843 | 8.932159 | nan | nan | nan |
| bollinger_squeeze | regime | 0 | 225 | 0.485669 | 1.031913 | 2.124726 | nan | nan | nan |
| volatility_compression | regime | 0 | 225 | -0.075815 | 0.359398 | 4.740436 | nan | nan | nan |
| inside_bar_chain | regime | 0 | 225 | 0.010460 | 0.514541 | 49.189413 | nan | nan | nan |
| test_count | regime | 0 | 225 | -0.249816 | 0.388389 | 1.554701 | nan | nan | nan |
| age_log_hours | regime | 0 | 225 | 0.048114 | 0.166740 | 3.465555 | nan | nan | nan |
| penetration_depth | regime | 0 | 225 | 0.064042 | 0.234384 | 3.659828 | nan | nan | nan |
| rejection_wick_ratio | regime | 0 | 225 | 0.029500 | 0.019665 | 0.666617 | nan | nan | nan |
| close_proximity | regime | 0 | 225 | 0.169040 | 0.382332 | 2.261779 | nan | nan | nan |
| volume_at_impact | regime | 0 | 225 | -0.011515 | 0.270728 | 23.511801 | nan | nan | nan |
| fakeout_ratio | regime | 0 | 225 | 0.113197 | 0.235607 | 2.081388 | nan | nan | nan |
| order_book_imbalance_proxy | regime | 0 | 225 | -0.021702 | 0.455476 | 20.987874 | nan | nan | nan |
| breakout_regime_0_prob | regime | 0 | 225 | 0.476772 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_regime_1_prob | regime | 0 | 225 | 0.277614 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_regime_2_prob | regime | 0 | 225 | 0.245614 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_long_edge_score | regime | 0 | 225 | -0.019831 | 0.132017 | 6.657053 | nan | nan | nan |
| breakout_short_edge_score | regime | 0 | 225 | 0.019831 | 0.132017 | 6.657053 | nan | nan | nan |
| breakout_bullish_prob | regime | 0 | 225 | 0.473314 | 0.220779 | 0.466454 | nan | nan | nan |
| breakout_bearish_prob | regime | 0 | 225 | 0.526686 | 0.220779 | 0.419185 | nan | nan | nan |
| breakout_level_strength | regime | 0 | 225 | 0.293939 | 0.065483 | 0.222777 | nan | nan | nan |

## Main Feature Drivers per Factor
For each factor, the table below lists the top features by absolute Spearman correlation with that factor.


### Factor: `forward_return_h96`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| rejection_wick_ratio | 0.1662 | 0.1385 | 129 |
| rubber_band_extension | -0.1623 | -0.0036 | 129 |

### Factor: `breakout_long_edge_score`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| rubber_band_extension | 0.5702 | 0.5294 | 225 |
| bollinger_squeeze | -0.3311 | -0.3100 | 225 |
| trend_strength_adx | -0.3107 | -0.1965 | 225 |
| age_log_hours | -0.2902 | -0.2215 | 225 |
| rejection_wick_ratio | 0.1632 | 0.1433 | 225 |
| test_count | 0.1528 | 0.0469 | 225 |
| close_proximity | -0.1448 | -0.1366 | 225 |
| inside_bar_chain | 0.1153 | -0.0888 | 225 |
| approach_velocity | 0.0889 | 0.2625 | 225 |
| order_book_imbalance_proxy | -0.0876 | -0.0922 | 225 |
| penetration_depth | 0.0813 | -0.0059 | 225 |
| fakeout_ratio | -0.0715 | 0.0040 | 225 |
| volatility_compression | 0.0581 | -0.0241 | 225 |
| volume_at_impact | -0.0284 | -0.0465 | 225 |

### Factor: `breakout_short_edge_score`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| rubber_band_extension | -0.5702 | -0.5294 | 225 |
| bollinger_squeeze | 0.3311 | 0.3100 | 225 |
| trend_strength_adx | 0.3107 | 0.1965 | 225 |
| age_log_hours | 0.2902 | 0.2215 | 225 |
| rejection_wick_ratio | -0.1632 | -0.1433 | 225 |
| test_count | -0.1528 | -0.0469 | 225 |
| close_proximity | 0.1448 | 0.1366 | 225 |
| inside_bar_chain | -0.1153 | 0.0888 | 225 |
| approach_velocity | -0.0889 | -0.2625 | 225 |
| order_book_imbalance_proxy | 0.0876 | 0.0922 | 225 |
| penetration_depth | -0.0813 | 0.0059 | 225 |
| fakeout_ratio | 0.0715 | -0.0040 | 225 |
| volatility_compression | -0.0581 | 0.0241 | 225 |
| volume_at_impact | 0.0284 | 0.0465 | 225 |