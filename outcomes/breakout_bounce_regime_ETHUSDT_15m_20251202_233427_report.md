# Breakout/Bounce Regime Diagnostics

- Symbol: **ETHUSDT**
- Exchange: **binance**
- Timeframe: **15m**
- Direction: **long**
- Horizon (bars): **96**
- Samples (training window): **224**

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
- Sample split: train=156, val=34, test=34

### Class Counts (training labels)
| Regime | Count |
|--------|-------|
| 0 | 101 |
| 1 | 35 |
| 2 | 50 |
| 3 | 38 |

### Resistance/Support Hit Counts by Outcome
| Side | Class 0 | Total |
|--------|--------|--------|
| resistance | 125 | 125 |
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
| Q1 | 85 | 11.00 - 15.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q2 | 31 | 18.00 - 18.00 | 100.0% | 0.0% | 0.0% | ∞ |

**Interpretation:**
- Higher quantiles with higher bounce/break ratios indicate better quality S/R levels
- Use this data to adjust sr_combined_strength weights in `_add_sr_strength_features()`

### Sr Volume Depth Ratio

| Quantile | Count | Metric Range | Bounce Rate | Break Rate | Trap Rate | Bounce/Break Ratio |
|----------|-------|--------------|-------------|------------|-----------|-------------------|
| Q3 | 30 | 1.32 - 1.33 | 100.0% | 0.0% | 0.0% | ∞ |
| Q4 | 51 | 1.48 - 1.89 | 100.0% | 0.0% | 0.0% | ∞ |
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
| Q2 | 40 | 0.00 - 0.00 | 100.0% | 0.0% | 0.0% | ∞ |

**Interpretation:**
- Higher quantiles with higher bounce/break ratios indicate better quality S/R levels
- Use this data to adjust sr_combined_strength weights in `_add_sr_strength_features()`


## Meta-Label Success Summary
- Meta-labeled events: **224**, success=1: **11** (4.911% success rate)

| Class | Meta Events | Success Count | Success Rate |
|-------|------------|---------------|--------------|
| 0 | 101 | 4 | 3.960% |
| 1 | 35 | 0 | 0.000% |
| 2 | 50 | 0 | 0.000% |
| 3 | 38 | 7 | 18.421% |

## Breakout Success Probability & High-Confidence Gating
- Observations with breakout_success_prob: **224** | mean=0.500, std=0.000, p25=0.500, median=0.500, p75=0.500
- High-confidence signals (high_conf=1): **0** / 224 (0.000%)

### Forward Return Sharpe by Meta/High-Confidence Subset
| Subset | Samples | Mean Return | Std Return | Sharpe-like |
|--------|---------|-------------|------------|-------------|
| all | 128 | -0.053062 | 0.060082 | -0.8832 |
| meta_success==1 | 7 | 0.000702 | 0.001693 | 0.4148 |

## Forward Return Sharpe-like Ratios
| Scope | Side | Regime | Mean Return | Std Return | Sharpe-like |
|-------|------|--------|-------------|------------|-------------|
| global | all | -1 | -0.053062 | 0.060082 | -0.8832 |
| regime | all | 0 | -0.053062 | 0.060082 | -0.8832 |
| global | resistance | -1 | -0.060568 | 0.060809 | -0.9960 |
| regime | resistance | 0 | -0.060568 | 0.060809 | -0.9960 |
| global | support | -1 | -0.045319 | 0.058319 | -0.7771 |
| regime | support | 0 | -0.045319 | 0.058319 | -0.7771 |

## Per-Regime Summary (Forward Returns & Edge Scores)
| Regime | Side | Count | Mean Forward Return | Sharpe-like | Mean Long Edge | Mean Short Edge | Mean Bullish Prob | Mean Bearish Prob |
|--------|------|-------|---------------------|------------|---------------|----------------|-------------------|--------------------|
| 0 | all | 224 | -0.053062 | -0.8832 | -0.019423 | 0.019423 | 0.473737 | 0.526263 |
| 0 | resistance | 125 | -0.060568 | -0.9960 | -0.136605 | 0.136605 | 0.273733 | 0.726267 |
| 0 | support | 99 | -0.045319 | -0.7771 | 0.128534 | -0.128534 | 0.726267 | 0.273733 |

## Winsorised CV Between/Within Regimes
Definition: per metric, we winsorise values (5–95%), compute global and per-regime CV, and report the ratio of between-regime CV to within-regime CV.

| Metric | Scope | Regime | Count | Mean | Std | CV | CV_between | CV_within | CV_between/within |
|--------|-------|--------|-------|------|-----|----|-----------|----------|--------------------|
| breakout_level_strength | global | -1 | 224 | 0.293526 | 0.065337 | 0.222592 | 0.000000 | 0.222592 | 0.000000 |
| breakout_bearish_prob | global | -1 | 224 | 0.526263 | 0.224738 | 0.427045 | 0.000000 | 0.427045 | 0.000000 |
| breakout_bullish_prob | global | -1 | 224 | 0.473737 | 0.224738 | 0.474394 | 0.000000 | 0.474394 | 0.000000 |
| breakout_short_edge_score | global | -1 | 224 | 0.019486 | 0.134222 | 6.888153 | 0.000000 | 6.888153 | 0.000000 |
| breakout_long_edge_score | global | -1 | 224 | -0.019486 | 0.134222 | 6.888153 | 0.000000 | 6.888153 | 0.000000 |
| breakout_regime_2_prob | global | -1 | 224 | 0.228571 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| breakout_regime_1_prob | global | -1 | 224 | 0.273733 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| breakout_regime_0_prob | global | -1 | 224 | 0.497696 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| order_book_imbalance_proxy | global | -1 | 224 | -0.017199 | 0.452433 | 26.305602 | 0.000000 | 26.305602 | 0.000000 |
| fakeout_ratio | global | -1 | 224 | 0.113767 | 0.236221 | 2.076362 | 0.000000 | 2.076362 | 0.000000 |
| volume_at_impact | global | -1 | 224 | -0.011509 | 0.271473 | 23.587787 | 0.000000 | 23.587787 | 0.000000 |
| close_proximity | global | -1 | 224 | 0.163634 | 0.375299 | 2.293523 | 0.000000 | 2.293523 | 0.000000 |
| rejection_wick_ratio | global | -1 | 224 | 0.029509 | 0.019711 | 0.667954 | 0.000000 | 0.667954 | 0.000000 |
| penetration_depth | global | -1 | 224 | 0.059345 | 0.225649 | 3.802321 | 0.000000 | 3.802321 | 0.000000 |
| age_log_hours | global | -1 | 224 | 0.051039 | 0.163211 | 3.197799 | 0.000000 | 3.197799 | 0.000000 |
| test_count | global | -1 | 224 | -0.246187 | 0.385441 | 1.565644 | 0.000000 | 1.565644 | 0.000000 |
| inside_bar_chain | global | -1 | 224 | 0.012627 | 0.516173 | 40.879662 | 0.000000 | 40.879662 | 0.000000 |
| volatility_compression | global | -1 | 224 | -0.078341 | 0.358481 | 4.575889 | 0.000000 | 4.575889 | 0.000000 |
| bollinger_squeeze | global | -1 | 224 | 0.487840 | 1.033710 | 2.118954 | 0.000000 | 2.118954 | 0.000000 |
| trend_strength_adx | global | -1 | 224 | -0.020395 | 0.207876 | 10.192708 | 0.000000 | 10.192708 | 0.000000 |
| momentum_divergence | global | -1 | 224 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| rubber_band_extension | global | -1 | 224 | -0.000428 | 0.001185 | 2.771526 | 0.000000 | 2.771526 | 0.000000 |
| approach_velocity | global | -1 | 224 | -0.040046 | 0.177774 | 4.439241 | 0.000000 | 4.439241 | 0.000000 |
| forward_return_support | global | -1 | 63 | -0.045352 | 0.058267 | 1.284781 | 0.000000 | 1.284781 | 0.000000 |
| forward_return_resistance | global | -1 | 65 | -0.060453 | 0.060391 | 0.998975 | 0.000000 | 0.998975 | 0.000000 |
| forward_return | global | -1 | 128 | -0.053031 | 0.059839 | 1.128372 | 0.000000 | 1.128372 | 0.000000 |
| forward_return | regime | 0 | 128 | -0.053031 | 0.059839 | 1.128372 | nan | nan | nan |
| forward_return_resistance | regime | 0 | 65 | -0.060453 | 0.060391 | 0.998975 | nan | nan | nan |
| forward_return_support | regime | 0 | 63 | -0.045352 | 0.058267 | 1.284781 | nan | nan | nan |
| approach_velocity | regime | 0 | 224 | -0.040046 | 0.177774 | 4.439241 | nan | nan | nan |
| rubber_band_extension | regime | 0 | 224 | -0.000428 | 0.001185 | 2.771526 | nan | nan | nan |
| momentum_divergence | regime | 0 | 224 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| trend_strength_adx | regime | 0 | 224 | -0.020395 | 0.207876 | 10.192708 | nan | nan | nan |
| bollinger_squeeze | regime | 0 | 224 | 0.487840 | 1.033710 | 2.118954 | nan | nan | nan |
| volatility_compression | regime | 0 | 224 | -0.078341 | 0.358481 | 4.575889 | nan | nan | nan |
| inside_bar_chain | regime | 0 | 224 | 0.012627 | 0.516173 | 40.879662 | nan | nan | nan |
| test_count | regime | 0 | 224 | -0.246187 | 0.385441 | 1.565644 | nan | nan | nan |
| age_log_hours | regime | 0 | 224 | 0.051039 | 0.163211 | 3.197799 | nan | nan | nan |
| penetration_depth | regime | 0 | 224 | 0.059345 | 0.225649 | 3.802321 | nan | nan | nan |
| rejection_wick_ratio | regime | 0 | 224 | 0.029509 | 0.019711 | 0.667954 | nan | nan | nan |
| close_proximity | regime | 0 | 224 | 0.163634 | 0.375299 | 2.293523 | nan | nan | nan |
| volume_at_impact | regime | 0 | 224 | -0.011509 | 0.271473 | 23.587787 | nan | nan | nan |
| fakeout_ratio | regime | 0 | 224 | 0.113767 | 0.236221 | 2.076362 | nan | nan | nan |
| order_book_imbalance_proxy | regime | 0 | 224 | -0.017199 | 0.452433 | 26.305602 | nan | nan | nan |
| breakout_regime_0_prob | regime | 0 | 224 | 0.497696 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_regime_1_prob | regime | 0 | 224 | 0.273733 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_regime_2_prob | regime | 0 | 224 | 0.228571 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_long_edge_score | regime | 0 | 224 | -0.019486 | 0.134222 | 6.888153 | nan | nan | nan |
| breakout_short_edge_score | regime | 0 | 224 | 0.019486 | 0.134222 | 6.888153 | nan | nan | nan |
| breakout_bullish_prob | regime | 0 | 224 | 0.473737 | 0.224738 | 0.474394 | nan | nan | nan |
| breakout_bearish_prob | regime | 0 | 224 | 0.526263 | 0.224738 | 0.427045 | nan | nan | nan |
| breakout_level_strength | regime | 0 | 224 | 0.293526 | 0.065337 | 0.222592 | nan | nan | nan |

## Main Feature Drivers per Factor
For each factor, the table below lists the top features by absolute Spearman correlation with that factor.


### Factor: `forward_return_h96`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| rejection_wick_ratio | 0.1800 | 0.1536 | 128 |
| rubber_band_extension | -0.1549 | 0.0060 | 128 |

### Factor: `breakout_long_edge_score`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| rubber_band_extension | 0.5642 | 0.5281 | 224 |
| bollinger_squeeze | -0.3342 | -0.3135 | 224 |
| trend_strength_adx | -0.3257 | -0.2160 | 224 |
| age_log_hours | -0.3043 | -0.2413 | 224 |
| rejection_wick_ratio | 0.1635 | 0.1431 | 224 |
| test_count | 0.1408 | 0.0367 | 224 |
| close_proximity | -0.1342 | -0.1264 | 224 |
| inside_bar_chain | 0.1040 | -0.0925 | 224 |
| order_book_imbalance_proxy | -0.1002 | -0.1029 | 224 |
| penetration_depth | 0.0947 | 0.0064 | 224 |
| approach_velocity | 0.0898 | 0.2641 | 224 |
| fakeout_ratio | -0.0732 | 0.0024 | 224 |
| volatility_compression | 0.0695 | -0.0205 | 224 |
| volume_at_impact | -0.0285 | -0.0469 | 224 |

### Factor: `breakout_short_edge_score`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| rubber_band_extension | -0.5642 | -0.5281 | 224 |
| bollinger_squeeze | 0.3342 | 0.3135 | 224 |
| trend_strength_adx | 0.3257 | 0.2160 | 224 |
| age_log_hours | 0.3043 | 0.2413 | 224 |
| rejection_wick_ratio | -0.1635 | -0.1431 | 224 |
| test_count | -0.1408 | -0.0367 | 224 |
| close_proximity | 0.1342 | 0.1264 | 224 |
| inside_bar_chain | -0.1040 | 0.0925 | 224 |
| order_book_imbalance_proxy | 0.1002 | 0.1029 | 224 |
| penetration_depth | -0.0947 | -0.0064 | 224 |
| approach_velocity | -0.0898 | -0.2641 | 224 |
| fakeout_ratio | 0.0732 | -0.0024 | 224 |
| volatility_compression | -0.0695 | 0.0205 | 224 |
| volume_at_impact | 0.0285 | 0.0469 | 224 |