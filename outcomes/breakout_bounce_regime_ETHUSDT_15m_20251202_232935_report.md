# Breakout/Bounce Regime Diagnostics

- Symbol: **ETHUSDT**
- Exchange: **binance**
- Timeframe: **15m**
- Direction: **long**
- Horizon (bars): **48**
- Samples (training window): **273**

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
- Validation log loss: **0.555095**
- Test log loss: **nan**
- Generalization gap (test - val log loss): **nan**
- Macro F1-score (val): **0.4605**
- Weighted F1-score (val): **0.7863**
- Sample split: train=191, val=41, test=41

### Class Counts (training labels)
| Regime | Count |
|--------|-------|
| 0 | 82 |
| 1 | 13 |
| 2 | 58 |
| 3 | 120 |

### Resistance/Support Hit Counts by Outcome
| Side | Class 0 | Total |
|--------|--------|--------|
| resistance | 148 | 148 |
| support | 125 | 125 |

## S/R Quality Metrics - Break vs Bounce Analysis

This section analyzes how SR level quality metrics (touch count, volume, prominence)
correlate with break vs bounce outcomes. Higher quantiles should show better bounce rates.

### Sr Touch Count

| Quantile | Count | Metric Range | Bounce Rate | Break Rate | Trap Rate | Bounce/Break Ratio |
|----------|-------|--------------|-------------|------------|-----------|-------------------|
| Q4 | 43 | 21.00 - 28.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q2 | 54 | 18.00 - 18.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q5 | 54 | 31.00 - 51.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q3 | 36 | 19.00 - 20.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q1 | 86 | 11.00 - 15.00 | 100.0% | 0.0% | 0.0% | ∞ |

**Interpretation:**
- Higher quantiles with higher bounce/break ratios indicate better quality S/R levels
- Use this data to adjust sr_combined_strength weights in `_add_sr_strength_features()`

### Sr Volume Depth Ratio

| Quantile | Count | Metric Range | Bounce Rate | Break Rate | Trap Rate | Bounce/Break Ratio |
|----------|-------|--------------|-------------|------------|-----------|-------------------|
| Q3 | 55 | 1.32 - 1.33 | 100.0% | 0.0% | 0.0% | ∞ |
| Q1 | 59 | 0.75 - 1.13 | 100.0% | 0.0% | 0.0% | ∞ |
| Q4 | 52 | 1.48 - 1.89 | 100.0% | 0.0% | 0.0% | ∞ |
| Q2 | 76 | 1.19 - 1.31 | 100.0% | 0.0% | 0.0% | ∞ |
| Q5 | 31 | 2.63 - 2.63 | 100.0% | 0.0% | 0.0% | ∞ |

**Interpretation:**
- Higher quantiles with higher bounce/break ratios indicate better quality S/R levels
- Use this data to adjust sr_combined_strength weights in `_add_sr_strength_features()`

### Sr Prominence

| Quantile | Count | Metric Range | Bounce Rate | Break Rate | Trap Rate | Bounce/Break Ratio |
|----------|-------|--------------|-------------|------------|-----------|-------------------|
| Q2 | 79 | 0.00 - 0.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q4 | 41 | 0.00 - 0.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q5 | 54 | 0.00 - 0.01 | 100.0% | 0.0% | 0.0% | ∞ |
| Q3 | 36 | 0.00 - 0.00 | 100.0% | 0.0% | 0.0% | ∞ |
| Q1 | 63 | 0.00 - 0.00 | 100.0% | 0.0% | 0.0% | ∞ |

**Interpretation:**
- Higher quantiles with higher bounce/break ratios indicate better quality S/R levels
- Use this data to adjust sr_combined_strength weights in `_add_sr_strength_features()`


## Meta-Label Success Summary
- Meta-labeled events: **273**, success=1: **7** (2.564% success rate)

| Class | Meta Events | Success Count | Success Rate |
|-------|------------|---------------|--------------|
| 0 | 82 | 3 | 3.659% |
| 1 | 13 | 0 | 0.000% |
| 2 | 58 | 1 | 1.724% |
| 3 | 120 | 3 | 2.500% |

## Breakout Success Probability & High-Confidence Gating
- Observations with breakout_success_prob: **273** | mean=0.500, std=0.000, p25=0.500, median=0.500, p75=0.500
- High-confidence signals (high_conf=1): **0** / 273 (0.000%)

### Forward Return Sharpe by Meta/High-Confidence Subset
| Subset | Samples | Mean Return | Std Return | Sharpe-like |
|--------|---------|-------------|------------|-------------|
| all | 225 | -0.024394 | 0.044092 | -0.5533 |
| meta_success==1 | 4 | -0.026189 | 0.063984 | -0.4093 |

## Forward Return Sharpe-like Ratios
| Scope | Side | Regime | Mean Return | Std Return | Sharpe-like |
|-------|------|--------|-------------|------------|-------------|
| global | all | -1 | -0.024394 | 0.044092 | -0.5533 |
| regime | all | 0 | -0.024394 | 0.044092 | -0.5533 |
| global | resistance | -1 | -0.025753 | 0.046316 | -0.5560 |
| regime | resistance | 0 | -0.025753 | 0.046316 | -0.5560 |
| global | support | -1 | -0.022947 | 0.041546 | -0.5523 |
| regime | support | 0 | -0.022947 | 0.041546 | -0.5523 |

## Per-Regime Summary (Forward Returns & Edge Scores)
| Regime | Side | Count | Mean Forward Return | Sharpe-like | Mean Long Edge | Mean Short Edge | Mean Bullish Prob | Mean Bearish Prob |
|--------|------|-------|---------------------|------------|---------------|----------------|-------------------|--------------------|
| 0 | all | 273 | -0.024394 | -0.5533 | -0.010796 | 0.010796 | 0.486208 | 0.513792 |
| 0 | resistance | 148 | -0.025753 | -0.5560 | -0.096501 | 0.096501 | 0.336292 | 0.663708 |
| 0 | support | 125 | -0.022947 | -0.5523 | 0.090679 | -0.090679 | 0.663708 | 0.336292 |

## Winsorised CV Between/Within Regimes
Definition: per metric, we winsorise values (5–95%), compute global and per-regime CV, and report the ratio of between-regime CV to within-regime CV.

| Metric | Scope | Regime | Count | Mean | Std | CV | CV_between | CV_within | CV_between/within |
|--------|-------|--------|-------|------|-----|----|-----------|----------|--------------------|
| breakout_level_strength | global | -1 | 273 | 0.286076 | 0.061477 | 0.214895 | 0.000000 | 0.214895 | 0.000000 |
| breakout_bearish_prob | global | -1 | 273 | 0.513792 | 0.163126 | 0.317494 | 0.000000 | 0.317494 | 0.000000 |
| breakout_bullish_prob | global | -1 | 273 | 0.486208 | 0.163126 | 0.335507 | 0.000000 | 0.335507 | 0.000000 |
| breakout_short_edge_score | global | -1 | 273 | 0.010883 | 0.094788 | 8.709609 | 0.000000 | 8.709609 | 0.000000 |
| breakout_long_edge_score | global | -1 | 273 | -0.010883 | 0.094788 | 8.709609 | 0.000000 | 8.709609 | 0.000000 |
| breakout_regime_2_prob | global | -1 | 273 | 0.199389 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | nan |
| breakout_regime_1_prob | global | -1 | 273 | 0.336292 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| breakout_regime_0_prob | global | -1 | 273 | 0.464319 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| order_book_imbalance_proxy | global | -1 | 273 | -0.021774 | 0.392219 | 18.012799 | 0.000000 | 18.012799 | 0.000000 |
| fakeout_ratio | global | -1 | 273 | 0.090612 | 0.209458 | 2.311594 | 0.000000 | 2.311594 | 0.000000 |
| volume_at_impact | global | -1 | 273 | -0.013209 | 0.228663 | 17.311168 | 0.000000 | 17.311168 | 0.000000 |
| close_proximity | global | -1 | 273 | 0.139431 | 0.339016 | 2.431427 | 0.000000 | 2.431427 | 0.000000 |
| rejection_wick_ratio | global | -1 | 273 | 0.033360 | 0.023729 | 0.711281 | 0.000000 | 0.711281 | 0.000000 |
| penetration_depth | global | -1 | 273 | 0.047216 | 0.192831 | 4.084009 | 0.000000 | 4.084009 | 0.000000 |
| age_log_hours | global | -1 | 273 | 0.041456 | 0.141331 | 3.409151 | 0.000000 | 3.409151 | 0.000000 |
| test_count | global | -1 | 273 | -0.205446 | 0.364203 | 1.772745 | 0.000000 | 1.772745 | 0.000000 |
| inside_bar_chain | global | -1 | 273 | -0.085012 | 0.163682 | 1.925388 | 0.000000 | 1.925388 | 0.000000 |
| volatility_compression | global | -1 | 273 | -0.061519 | 0.315712 | 5.131911 | 0.000000 | 5.131911 | 0.000000 |
| bollinger_squeeze | global | -1 | 273 | 0.398698 | 0.950247 | 2.383376 | 0.000000 | 2.383376 | 0.000000 |
| trend_strength_adx | global | -1 | 273 | -0.019296 | 0.183861 | 9.528667 | 0.000000 | 9.528667 | 0.000000 |
| momentum_divergence | global | -1 | 273 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| rubber_band_extension | global | -1 | 273 | -0.000338 | 0.001221 | 3.611421 | 0.000000 | 3.611421 | 0.000000 |
| approach_velocity | global | -1 | 273 | -0.023503 | 0.122316 | 5.204163 | 0.000000 | 5.204163 | 0.000000 |
| forward_return_support | global | -1 | 109 | -0.022714 | 0.040736 | 1.793465 | 0.000000 | 1.793465 | 0.000000 |
| forward_return_resistance | global | -1 | 116 | -0.024796 | 0.043608 | 1.758638 | 0.000000 | 1.758638 | 0.000000 |
| forward_return | global | -1 | 225 | -0.023715 | 0.042066 | 1.773853 | 0.000000 | 1.773853 | 0.000000 |
| forward_return | regime | 0 | 225 | -0.023715 | 0.042066 | 1.773853 | nan | nan | nan |
| forward_return_resistance | regime | 0 | 116 | -0.024796 | 0.043608 | 1.758638 | nan | nan | nan |
| forward_return_support | regime | 0 | 109 | -0.022714 | 0.040736 | 1.793465 | nan | nan | nan |
| approach_velocity | regime | 0 | 273 | -0.023503 | 0.122316 | 5.204163 | nan | nan | nan |
| rubber_band_extension | regime | 0 | 273 | -0.000338 | 0.001221 | 3.611421 | nan | nan | nan |
| momentum_divergence | regime | 0 | 273 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| trend_strength_adx | regime | 0 | 273 | -0.019296 | 0.183861 | 9.528667 | nan | nan | nan |
| bollinger_squeeze | regime | 0 | 273 | 0.398698 | 0.950247 | 2.383376 | nan | nan | nan |
| volatility_compression | regime | 0 | 273 | -0.061519 | 0.315712 | 5.131911 | nan | nan | nan |
| inside_bar_chain | regime | 0 | 273 | -0.085012 | 0.163682 | 1.925388 | nan | nan | nan |
| test_count | regime | 0 | 273 | -0.205446 | 0.364203 | 1.772745 | nan | nan | nan |
| age_log_hours | regime | 0 | 273 | 0.041456 | 0.141331 | 3.409151 | nan | nan | nan |
| penetration_depth | regime | 0 | 273 | 0.047216 | 0.192831 | 4.084009 | nan | nan | nan |
| rejection_wick_ratio | regime | 0 | 273 | 0.033360 | 0.023729 | 0.711281 | nan | nan | nan |
| close_proximity | regime | 0 | 273 | 0.139431 | 0.339016 | 2.431427 | nan | nan | nan |
| volume_at_impact | regime | 0 | 273 | -0.013209 | 0.228663 | 17.311168 | nan | nan | nan |
| fakeout_ratio | regime | 0 | 273 | 0.090612 | 0.209458 | 2.311594 | nan | nan | nan |
| order_book_imbalance_proxy | regime | 0 | 273 | -0.021774 | 0.392219 | 18.012799 | nan | nan | nan |
| breakout_regime_0_prob | regime | 0 | 273 | 0.464319 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_regime_1_prob | regime | 0 | 273 | 0.336292 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_regime_2_prob | regime | 0 | 273 | 0.199389 | 0.000000 | 0.000000 | nan | nan | nan |
| breakout_long_edge_score | regime | 0 | 273 | -0.010883 | 0.094788 | 8.709609 | nan | nan | nan |
| breakout_short_edge_score | regime | 0 | 273 | 0.010883 | 0.094788 | 8.709609 | nan | nan | nan |
| breakout_bullish_prob | regime | 0 | 273 | 0.486208 | 0.163126 | 0.335507 | nan | nan | nan |
| breakout_bearish_prob | regime | 0 | 273 | 0.513792 | 0.163126 | 0.317494 | nan | nan | nan |
| breakout_level_strength | regime | 0 | 273 | 0.286076 | 0.061477 | 0.214895 | nan | nan | nan |

## Main Feature Drivers per Factor
For each factor, the table below lists the top features by absolute Spearman correlation with that factor.


### Factor: `forward_return_h48`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| test_count | 0.4689 | 0.7092 | 225 |
| trend_strength_adx | 0.4569 | 0.6793 | 225 |
| bollinger_squeeze | -0.3689 | -0.5403 | 225 |
| inside_bar_chain | 0.3045 | -0.0157 | 225 |
| age_log_hours | 0.2732 | 0.4784 | 225 |
| fakeout_ratio | -0.2101 | -0.2302 | 225 |
| volume_at_impact | 0.1897 | 0.3324 | 225 |
| rejection_wick_ratio | 0.1802 | 0.1869 | 225 |
| penetration_depth | -0.0979 | -0.2695 | 225 |
| close_proximity | -0.0952 | -0.3072 | 225 |
| rubber_band_extension | 0.0871 | 0.0388 | 225 |
| approach_velocity | 0.0292 | -0.0415 | 225 |
| order_book_imbalance_proxy | -0.0209 | 0.0077 | 225 |
| volatility_compression | 0.0111 | -0.0809 | 225 |

### Factor: `breakout_long_edge_score`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| rubber_band_extension | 0.3461 | 0.3040 | 273 |
| bollinger_squeeze | -0.3052 | -0.2993 | 273 |
| trend_strength_adx | -0.2864 | -0.1791 | 273 |
| age_log_hours | -0.2688 | -0.2125 | 273 |
| test_count | 0.1621 | 0.0642 | 273 |
| rejection_wick_ratio | 0.1530 | 0.1418 | 273 |
| close_proximity | -0.1403 | -0.1381 | 273 |
| inside_bar_chain | 0.1096 | -0.0848 | 273 |
| approach_velocity | 0.0847 | 0.2489 | 273 |
| order_book_imbalance_proxy | -0.0823 | -0.0842 | 273 |
| fakeout_ratio | -0.0744 | -0.0075 | 273 |
| penetration_depth | 0.0724 | -0.0132 | 273 |
| volatility_compression | 0.0556 | -0.0226 | 273 |
| volume_at_impact | -0.0259 | -0.0450 | 273 |

### Factor: `breakout_short_edge_score`
| Feature | Spearman | Pearson | Samples |
|---------|----------|---------|---------|
| rubber_band_extension | -0.3461 | -0.3040 | 273 |
| bollinger_squeeze | 0.3052 | 0.2993 | 273 |
| trend_strength_adx | 0.2864 | 0.1791 | 273 |
| age_log_hours | 0.2688 | 0.2125 | 273 |
| test_count | -0.1621 | -0.0642 | 273 |
| rejection_wick_ratio | -0.1530 | -0.1418 | 273 |
| close_proximity | 0.1403 | 0.1381 | 273 |
| inside_bar_chain | -0.1096 | 0.0848 | 273 |
| approach_velocity | -0.0847 | -0.2489 | 273 |
| order_book_imbalance_proxy | 0.0823 | 0.0842 | 273 |
| fakeout_ratio | 0.0744 | 0.0075 | 273 |
| penetration_depth | -0.0724 | 0.0132 | 273 |
| volatility_compression | -0.0556 | 0.0226 | 273 |
| volume_at_impact | 0.0259 | 0.0450 | 273 |