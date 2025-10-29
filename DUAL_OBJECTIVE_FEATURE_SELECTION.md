# Dual-Objective Feature Selection: Accuracy + Temporal Smoothness

## Overview

The feature selection process has been enhanced to optimize for **both accuracy and temporal smoothness**, selecting 60-80 features that maximize both classification performance and prediction stability.

## Feature Selection Algorithm

### Dual-Objective Selection Process

**Location**: `regime_models_training.py` → `_run_feature_selection()`

**Method**: `dual_objective_accuracy_temporal`

### Step-by-Step Process

#### Step 1: Accuracy-Based Importance
- Train LightGBM classifier on all features
- Extract feature importances (gain-based)
- Normalize to get accuracy scores (0-1 scale)

**Model Configuration**:
```python
lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    min_child_samples=50,  # Increased for stability
    min_data_in_leaf=50   # Increased for stability
)
```

#### Step 2: Temporal Smoothness Evaluation
- For top 150 features by accuracy:
  - Train quick LightGBM model on single feature
  - Use TimeSeriesSplit CV (3 folds)
  - Evaluate predictions for transition rate
  - Calculate smoothness score = 1 - transition_rate
- For remaining features:
  - Use accuracy score as proxy (with 0.7 discount)

**Smoothness Score**:
```
smoothness = 1.0 - min(transition_rate, 1.0)
```
- Lower transition rate → Higher smoothness score
- Higher smoothness → More stable predictions

#### Step 3: Combine Scores
**Weighted Combination**:
```
combined_score = 0.6 * accuracy_score + 0.4 * temporal_smoothness_score
```

**Weights**:
- Accuracy: 60% weight
- Temporal smoothness: 40% weight

#### Step 4: Select Top Features
- Sort features by combined score (descending)
- Select top 60-80 features
- Target count: `min(80, max(60, int(total_features * 0.3)))`

## Feature Count Logic

**Target Range**: 60-80 features

**Selection Rules**:
- If total features < 60: Use all features
- If total features ≥ 60: Select 60-80 features
- Formula: `min(80, max(60, int(total_features * 0.3)))`

**Examples**:
- 200 features → 60 features (30%)
- 300 features → 70 features (23%)
- 400 features → 80 features (20%)
- 500 features → 80 features (16%)

## Output Structure

The selection returns comprehensive information:

```python
{
    'selection_performed': True,
    'selection_method': 'dual_objective_accuracy_temporal',
    'selected_indices': [0, 1, 5, 10, ...],
    'selected_feature_names': ['feature_1', 'feature_2', ...],
    'retained_feature_count': 70,
    'total_feature_count': 245,
    'target_feature_count': 70,
    'accuracy_weight': 0.6,
    'temporal_weight': 0.4,
    'feature_importances': {
        'feature_1': 0.0123,  # Combined score
        ...
    },
    'accuracy_importances': {
        'feature_1': 0.0156,  # Accuracy-only score
        ...
    },
    'temporal_importances': {
        'feature_1': 0.0089,  # Temporal smoothness score
        ...
    },
    'importance_ranking': [
        {
            'feature': 'feature_1',
            'combined_score': 0.0123,
            'accuracy_score': 0.0156,
            'temporal_score': 0.0089,
            'rank': 1
        },
        ...
    ],
    'top_features_preview': 'feature_1 (acc:0.0156,temp:0.0089), ...'
}
```

## Fallback Strategy

If dual-objective selection fails:

1. **Primary Fallback**: Accuracy-based selection
   - Uses `SelectFromModel` with LightGBM
   - Still enforces 60-80 feature limit
   - Method: `accuracy_based_fallback`

2. **Ultimate Fallback**: Use all features
   - Only if both methods fail
   - Method: `all_features_fallback`

## Performance Optimization

**Temporal Smoothness Evaluation**:
- Only evaluates top 150 features by accuracy (not all features)
- Significantly reduces computation time
- Uses quick models (50 estimators vs 200)

**Time Complexity**:
- Accuracy evaluation: O(n_features) - single model training
- Temporal evaluation: O(150) - individual feature models
- Total: ~2-5 seconds for 200-500 features

## Benefits

1. **Optimized for Stability**: Features selected promote stable predictions
2. **Maintains Accuracy**: Still prioritizes classification accuracy
3. **Optimal Feature Count**: 60-80 features balances model complexity and performance
4. **Comprehensive Metrics**: Tracks both accuracy and temporal scores
5. **Transparent Selection**: Shows why each feature was selected

## Example Output

```
🎯 [REGIME_MODELS] Starting dual-objective feature selection (accuracy + temporal smoothness)
📊 [REGIME_MODELS] Target feature count: 70
🔍 [REGIME_MODELS] Step 1: Evaluating accuracy-based importance
🔍 [REGIME_MODELS] Step 2: Evaluating temporal smoothness
🔍 [REGIME_MODELS] Step 3: Combining accuracy and temporal smoothness scores
✅ [REGIME_MODELS] Dual-objective feature selection completed in 3.456s
🎯 [REGIME_MODELS] Retained 70/245 features (target: 70)
📊 [REGIME_MODELS] Top 5 features: feature_1 (acc:0.0156,temp:0.0089), feature_2 (acc:0.0142,temp:0.0078), ...
```

## Integration

The dual-objective selection is automatically used when:
- Feature count > 60 features
- Sufficient data available for temporal CV

If selection fails, falls back gracefully to accuracy-based selection with the same 60-80 feature limit.
