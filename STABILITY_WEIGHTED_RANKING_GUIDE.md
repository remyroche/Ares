# Stability-Weighted Feature Ranking Guide

## Overview

The feature selection now supports **stability-weighted ranking** that combines SHAP importance with temporal stability scores to prioritize features that are both predictive AND stable across time.

## How It Works

### Formula
```
combined_score = (1 - stability_weight) × SHAP_importance + stability_weight × stability_score
```

### Components

1. **SHAP Importance** (0-1 normalized)
   - Measures feature's predictive power
   - Captures feature interactions
   - Based on game-theoretic interpretation

2. **Stability Score** (0-1)
   - Measures consistency across time windows
   - Formula: `1 / (1 + coefficient_of_variation)`
   - High score = low variation in importance over time

3. **Stability Weight** (0-1)
   - Controls the balance between importance and stability
   - `0.0` = Pure SHAP importance (default)
   - `0.3` = 30% stability, 70% importance (recommended)
   - `0.5` = Equal weight
   - `1.0` = Pure stability

## Usage

### Option 1: Via Configuration File

Edit your config YAML file (e.g., `config/analyst_multi_output_config.yaml`):

```yaml
feature_selection:
  max_features: 60
  min_features: 40
  selection_method: "permutation"
  use_permutation_importance: true
  stability_weight: 0.3  # Add this line
```

### Option 2: Programmatically

```python
from src.training.steps.pre_training.components.final_feature_selection import (
    FinalFeatureSelectionConfig,
    FinalFeatureSelectionComponent
)

# Create config with stability weighting
config = FinalFeatureSelectionConfig(
    max_features=60,
    min_features=40,
    selection_method='permutation',
    use_permutation_importance=True,
    stability_weight=0.3  # 30% stability, 70% importance
)

# Use the component
component = FinalFeatureSelectionComponent(config)
selected_features = component.select_features(X, y, feature_cols)
```

### Option 3: Via Step Configuration

In `feature_generation_final_feature_selection_step.py`:

```python
config = FinalFeatureSelectionConfig(
    max_features=size,
    min_features=max(5, size // 2),
    selection_method=config.get('selection_method', 'permutation'),
    scoring_threshold=config.get('scoring_threshold', 0.01),
    use_tree_based=config.get('use_tree_based', True),
    use_permutation_importance=config.get('use_permutation_importance', True),
    stability_weight=config.get('stability_weight', 0.3)  # Add this
)
```

## Recommended Settings

### Conservative (Prioritize Stability)
```yaml
stability_weight: 0.5  # 50% stability, 50% importance
```
- **Use when**: Market conditions change frequently
- **Effect**: Selects features that work consistently over time
- **Trade-off**: May miss some high-importance but volatile features

### Balanced (Recommended)
```yaml
stability_weight: 0.3  # 30% stability, 70% importance
```
- **Use when**: General trading scenarios
- **Effect**: Good balance between predictive power and reliability
- **Trade-off**: Optimal for most use cases

### Aggressive (Prioritize Importance)
```yaml
stability_weight: 0.1  # 10% stability, 90% importance
```
- **Use when**: Stable market conditions
- **Effect**: Maximizes predictive power
- **Trade-off**: May select features that work well in-sample but poorly out-of-sample

### Pure Importance (Default)
```yaml
stability_weight: 0.0  # 0% stability, 100% importance
```
- **Use when**: You want original SHAP-only ranking
- **Effect**: No stability adjustment
- **Trade-off**: Current behavior (no change)

## Benefits

### 1. Improved Out-of-Sample Performance
- Stable features generalize better to unseen data
- Reduces overfitting to specific time periods

### 2. Robustness to Market Regime Changes
- Features that work across different market conditions
- Less sensitive to temporary market anomalies

### 3. Better Feature Interpretability
- Stable features are more reliable for decision-making
- Easier to trust and explain

### 4. Reduced Model Degradation
- Model performance degrades slower over time
- Less frequent retraining required

## Example Output

When stability weighting is enabled, you'll see logs like:

```
📊 Step 3.5: Applying stability-weighted ranking (weight=0.3)
🔄 Computing stability scores for 176 features...
✅ Stability-weighted ranking complete:
   Weight: 30.0% stability, 70.0% importance
   Ranking changes in top 20: 8
   New top 5: ['feature_a', 'feature_b', 'feature_c', 'feature_d', 'feature_e']
```

## Comparison: Before vs After

### Without Stability Weighting (weight=0.0)
```
Top 5 Features:
1. high_importance_volatile_feature (SHAP: 0.95, Stability: 0.40)
2. medium_importance_feature (SHAP: 0.85, Stability: 0.60)
3. high_importance_unstable (SHAP: 0.80, Stability: 0.35)
4. medium_importance_stable (SHAP: 0.75, Stability: 0.85)
5. low_importance_feature (SHAP: 0.70, Stability: 0.50)
```

### With Stability Weighting (weight=0.3)
```
Top 5 Features:
1. medium_importance_feature (Combined: 0.78)  ← More balanced
2. medium_importance_stable (Combined: 0.77)  ← Promoted due to stability
3. high_importance_volatile_feature (Combined: 0.76)  ← Slightly demoted
4. high_importance_unstable (Combined: 0.67)  ← Demoted due to instability
5. stable_moderate_feature (Combined: 0.65)  ← New entry
```

## Technical Details

### Stability Calculation

1. **Split data into 5 time windows**
2. **For each window**:
   - Calculate feature-target correlation
   - Store importance score
3. **Calculate coefficient of variation**:
   ```python
   cv = std(importances) / mean(importances)
   ```
4. **Convert to stability score**:
   ```python
   stability = 1 / (1 + cv)
   ```
   - Low CV → High stability (score near 1.0)
   - High CV → Low stability (score near 0.0)

### Normalization

- SHAP importances are normalized to [0, 1] range
- Stability scores are already in [0, 1] range
- Combined scores are weighted averages in [0, 1] range

## Monitoring

Check the enhanced analysis report for stability metrics:

```
Stability analysis: 24/60 features stable (threshold=0.61)
```

This shows how many features meet the stability threshold, independent of the weighting.

## Best Practices

1. **Start with default (0.0)** to establish baseline performance
2. **Gradually increase** to 0.2-0.3 and monitor out-of-sample performance
3. **Compare results** between different weights using backtesting
4. **Monitor stability metrics** in the enhanced analysis report
5. **Adjust based on market conditions**:
   - Volatile markets → Higher weight (0.4-0.5)
   - Stable markets → Lower weight (0.1-0.2)

## Troubleshooting

### Issue: No ranking changes observed
- **Cause**: All features have similar stability scores
- **Solution**: Check if data has sufficient time variation

### Issue: Too many ranking changes
- **Cause**: Weight too high or features very unstable
- **Solution**: Reduce weight or investigate data quality

### Issue: Performance degraded
- **Cause**: Stability weight too high, losing important features
- **Solution**: Reduce weight to 0.1-0.2

## Summary

Stability-weighted ranking gives you control over the **importance vs. stability trade-off**:

- **0.0**: Maximum predictive power (current behavior)
- **0.3**: Balanced approach (recommended)
- **0.5**: Equal weight to both
- **1.0**: Maximum stability

Start with 0.3 and adjust based on your specific use case and backtesting results.
