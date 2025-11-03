# Regime Features Artifact Flow

**Date:** November 2, 2025  
**Purpose:** Document how regime features flow from regime_ensemble_training to model training

## Source: regime_ensemble_training

The `regime_ensemble_training` step trains ML models to classify market regimes and outputs regime probabilities for each regime type.

### Artifacts Created

**From `regime_ensemble_training` step:**
- `regime_ensemble_predictions` - Ensemble predictions for all samples
- `regime_ensemble_probabilities` - Probability matrix with columns:
  - `regime_prob_0` - Probability of Regime 0 (e.g., Trending Up)
  - `regime_prob_1` - Probability of Regime 1 (e.g., Ranging)
  - `regime_prob_2` - Probability of Regime 2 (e.g., Trending Down)
  - `regime_prob_3` - Probability of Regime 3 (e.g., Volatile)

### Regime Types (Example - may vary by implementation)

0. **Trending Up:** Strong upward price movement
1. **Ranging:** Sideways/consolidation price movement  
2. **Trending Down:** Strong downward price movement
3. **Volatile:** High volatility, unpredictable movement

## Usage in Feature Engineering

### Analyst Models

The `AnalystFeatureEngineer` takes regime probabilities and creates confidence features:

```python
# Input: regime_ensemble_probabilities artifact
# Columns: regime_prob_0, regime_prob_1, regime_prob_2, regime_prob_3

# Extract probabilities
regime_probs = {
    0: data['regime_prob_0'],  # Series with probability values
    1: data['regime_prob_1'],
    2: data['regime_prob_2'],
    3: data['regime_prob_3']
}

# Engineer features
analyst_engineer = AnalystFeatureEngineer()
features = analyst_engineer.engineer_features(
    base_features,  # 50 features from feature_generation_final_feature_selection_step
    regime_probabilities=regime_probs
)

# Output columns added:
# - regime_confidence_0 (same as regime_prob_0)
# - regime_confidence_1 (same as regime_prob_1)
# - regime_confidence_2 (same as regime_prob_2)
# - regime_confidence_3 (same as regime_prob_3)
```

### Tactician Models

The `TacticianFeatureEngineer` uses the same regime probabilities plus analyst signals:

```python
# Same regime probabilities as Analyst
regime_probs = {
    0: data['regime_prob_0'],
    1: data['regime_prob_1'],
    2: data['regime_prob_2'],
    3: data['regime_prob_3']
}

# Plus analyst ensemble output
analyst_signal = analyst_ensemble_predictions['prediction']  # Or aggregated value

# Engineer features
tactician_engineer = TacticianFeatureEngineer()
features = tactician_engineer.engineer_features(
    base_features,  # 50 features from feature_generation_final_feature_selection_step
    regime_probabilities=regime_probs,
    analyst_signal_strength=analyst_signal
)

# Output columns added:
# - regime_confidence_0
# - regime_confidence_1
# - regime_confidence_2
# - regime_confidence_3
# - analyst_signal_strength
```

## Complete Artifact Flow

```
┌─────────────────────────────────────────┐
│   regime_ensemble_training              │
│   (ML models classify regimes)          │
└────────────┬────────────────────────────┘
             │
             │ Creates artifacts:
             │ - regime_ensemble_probabilities
             │   (regime_prob_0, regime_prob_1, 
             │    regime_prob_2, regime_prob_3)
             │
             ↓
┌─────────────────────────────────────────┐
│   feature_generation_final_             │
│   feature_selection_step                │
│   (Selects top 50 features)             │
└────────────┬────────────────────────────┘
             │
             │ Creates artifacts:
             │ - selected_feature_dataframe_50
             │
             ↓
┌─────────────────────────────────────────┐
│   Merge: base features + regime probs   │
│   base_features (50)                    │
│   + regime_prob_0-3 (4)                 │
│   = 54 columns                          │
└────────────┬────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────┐
│   AnalystFeatureEngineer                │
│   Renames regime_prob_X →               │
│   regime_confidence_X                   │
└────────────┬────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────┐
│   Analyst Models Training               │
│   (54 features: 50 base + 4 regime)     │
└────────────┬────────────────────────────┘
             │
             │ Creates artifacts:
             │ - analyst_ensemble_predictions
             │
             ↓
┌─────────────────────────────────────────┐
│   TacticianFeatureEngineer              │
│   base (50) + regime (4) + analyst (1)  │
│   = 55 features                         │
└────────────┬────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────┐
│   Tactician Models Training             │
│   (55 features)                         │
└─────────────────────────────────────────┘
```

## Implementation Details

### Automatic Fallback Logic

The feature engineers have built-in fallback logic:

```python
# Priority 1: Use provided regime_probabilities dict
if regime_probabilities is not None:
    for regime_idx in range(4):
        result_data[f'regime_confidence_{regime_idx}'] = regime_probabilities[regime_idx]

# Priority 2: Extract from existing columns (regime_prob_0-3)
elif any(f'regime_prob_{i}' in data.columns for i in range(4)):
    for regime_idx in range(4):
        prob_col = f'regime_prob_{regime_idx}'
        if prob_col in data.columns:
            result_data[f'regime_confidence_{regime_idx}'] = data[prob_col]

# Priority 3: Use uniform distribution as fallback
else:
    for regime_idx in range(4):
        result_data[f'regime_confidence_{regime_idx}'] = 0.25  # Equal probability
```

### Expected Artifact Names

**From regime_ensemble_training:**
- `regime_ensemble_probabilities` (preferred)
- `regime_ml_outputs` (alternative)
- Columns: `regime_prob_0`, `regime_prob_1`, `regime_prob_2`, `regime_prob_3`

**From feature_generation_final_feature_selection_step:**
- `selected_feature_dataframe_50` (or 40/60)
- `final_dataset` (alternative)

**From analyst_ensemble_models:**
- `analyst_ensemble_predictions`
- `analyst_ensemble_confidence`

## Common Issues & Solutions

### Issue 1: Regime probabilities not found
```
Warning: "No regime probabilities provided or found in data, using uniform distribution"
```
**Solution:** Ensure `regime_ensemble_training` has run and artifacts are saved properly.

### Issue 2: Only uniform probabilities (0.25, 0.25, 0.25, 0.25)
**Cause:** Regime ensemble didn't output probabilities or artifact wasn't loaded.
**Solution:** Check regime_ensemble_training logs and artifact outputs.

### Issue 3: Feature count mismatch
```
Expected: 54 features (Analyst) or 55 features (Tactician)
Actual: Different count
```
**Solution:** 
1. Verify base features count from feature_generation_final_feature_selection_step
2. Check regime features were added (should have regime_confidence_0-3)
3. For Tactician, verify analyst_signal_strength was added

## Testing & Verification

### Verify Regime Artifacts Exist
```python
# Check regime ensemble outputs
regime_probs = get_artifact('regime_ensemble_probabilities')
assert 'regime_prob_0' in regime_probs.columns
assert 'regime_prob_1' in regime_probs.columns
assert 'regime_prob_2' in regime_probs.columns
assert 'regime_prob_3' in regime_probs.columns
print(f"✓ Regime probabilities loaded: {regime_probs.shape}")
```

### Verify Feature Engineering Output
```python
# Test Analyst
analyst_features = analyst_engineer.engineer_features(base_features, regime_probs)
assert 'regime_confidence_0' in analyst_features.columns
assert 'regime_confidence_1' in analyst_features.columns
assert 'regime_confidence_2' in analyst_features.columns
assert 'regime_confidence_3' in analyst_features.columns
print(f"✓ Analyst features: {analyst_features.shape}")
# Expected: (N, 54) = 50 base + 4 regime

# Test Tactician
tactician_features = tactician_engineer.engineer_features(
    base_features, regime_probs, analyst_signal
)
assert 'analyst_signal_strength' in tactician_features.columns
print(f"✓ Tactician features: {tactician_features.shape}")
# Expected: (N, 55) = 50 base + 4 regime + 1 analyst
```

## Summary

- **Regime features** come from `regime_ensemble_training` ML model outputs
- **4 regime probabilities** (regime_prob_0 through regime_prob_3) are renamed to regime_confidence
- **Feature engineers** automatically handle fallback to uniform distribution if regime data missing
- **Total features:** Analyst=54, Tactician=55

This ensures that models have access to current market regime information which cannot be pre-computed and must come from real-time regime classification.

