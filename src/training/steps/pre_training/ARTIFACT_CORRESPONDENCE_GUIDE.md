# Pre-Training Steps Artifact Correspondence Guide

This document shows the complete correspondence between what each step creates and what other steps need to retrieve.

## Step Dependencies Overview

```
feature_generation_data_validation_step
    ↓
feature_generation_labeling_integration_step
    ↓
feature_generation_feature_generation_step
    ↓
feature_generation_period_lookback_optimization_step
    ↓
feature_generation_feature_selection_step
    ↓
feature_generation_interaction_generation_step_analyst
feature_generation_interaction_generation_step_tactician
    ↓
feature_generation_final_feature_selection_step
    ↓
feature_generation_final_validation_step
```

## Detailed Artifact Correspondence

### 1. feature_generation_data_validation_step
**Creates:**
- `validated_dataframe` - Clean, validated OHLCV data
- `validation_metrics` - Data quality metrics
- `data_quality_scores` - Quality assessment scores

**Used by:**
- `feature_generation_labeling_integration_step`

### 2. feature_generation_labeling_integration_step
**Creates:**
- `labeled_dataframe` - OHLCV data with labels/targets
- `targets` - Target variables for ML
- `labeling_metrics` - Labeling quality metrics
- `labeling_quality_scores` - Quality scores for labels

**Used by:**
- `feature_generation_period_lookback_optimization_step`
- `feature_generation_feature_selection_step`
- `feature_generation_interaction_generation_step_analyst`
- `feature_generation_interaction_generation_step_tactician`
- `feature_generation_final_feature_selection_step`
- `feature_generation_final_validation_step`

### 3. feature_generation_feature_generation_step
**Creates:**
- `feature_dataframe` - Generated features DataFrame
- `feature_names` - List of feature names
- `feature_categories` - Feature categorization
- `feature_generation_metrics` - Generation metrics
- `feature_generation_stats` - Statistical summaries

**Used by:**
- `feature_generation_period_lookback_optimization_step`
- `feature_generation_feature_selection_step`
- `feature_generation_final_feature_selection_step`

### 4. feature_generation_period_lookback_optimization_step
**Creates:**
- `optimized_periods` - Best periods for each feature
- `optimized_lookbacks` - Best lookback windows
- `mi_best_lookbacks_per_feature` - MI-based best lookbacks
- `mrmr_top_lookbacks_per_feature` - mRMR-based top lookbacks
- `mi_scores_by_feature` - MI scores by feature
- `oos_sharpe_by_feature_window` - Out-of-sample Sharpe ratios
- `selected_features_metadata` - Metadata for selected features
- `family_diagnostics` - Feature family diagnostics

**Used by:**
- `feature_generation_feature_selection_step` (top1 periods/lookbacks)
- `feature_generation_interaction_generation_step_analyst` (top2-3 periods/lookbacks)
- `feature_generation_interaction_generation_step_tactician` (top2-3 periods/lookbacks)
- `feature_generation_final_feature_selection_step` (top1 periods/lookbacks)

### 5. feature_generation_feature_selection_step
**Creates:**
- `selected_features` - Selected feature set
- `feature_selection_scores` - Selection scores
- `feature_importance_rankings` - Importance rankings
- `selection_metrics` - Selection performance metrics
- `selection_performance` - Performance evaluation

**Used by:**
- `feature_generation_interaction_generation_step_analyst`
- `feature_generation_interaction_generation_step_tactician`

### 6. feature_generation_interaction_generation_step_analyst
**Creates:**
- `interaction_features` - Generated interaction features
- `interaction_metadata` - Interaction generation metadata
- `interaction_generation_metrics` - Generation metrics
- `interaction_performance` - Performance metrics
- `interaction_quality_scores` - Quality scores

**Used by:**
- `feature_generation_final_feature_selection_step`

### 7. feature_generation_interaction_generation_step_tactician
**Creates:**
- `interaction_features` - Generated interaction features
- `interaction_metadata` - Interaction generation metadata
- `interaction_generation_metrics` - Generation metrics
- `interaction_performance` - Performance metrics
- `interaction_quality_scores` - Quality scores

**Used by:**
- `feature_generation_final_feature_selection_step`

### 8. feature_generation_final_feature_selection_step
**Creates:**
- `selected_features_60` - Top 60 features
- `selected_features_50` - Top 50 features
- `selected_features_40` - Top 40 features
- `selected_feature_dataframe_60` - DataFrame with top 60 features
- `selected_feature_dataframe_50` - DataFrame with top 50 features
- `selected_feature_dataframe_40` - DataFrame with top 40 features
- `feature_scores` - Feature importance scores
- `shap_values_60/50/40` - SHAP values for each feature set
- `selection_metadata` - Selection process metadata

**Used by:**
- `feature_generation_final_validation_step`

### 9. feature_generation_final_validation_step
**Creates:**
- `final_dataset` - Final validated dataset
- `final_validation_metrics` - Final validation metrics
- `final_quality_scores` - Final quality scores
- `final_validation_warnings` - Validation warnings
- `final_performance_metrics` - Final performance metrics

**Used by:**
- None (final step)

## Usage Examples

### Retrieving Artifacts in a Step
```python
# In feature_generation_period_lookback_optimization_step
artifact_manager = get_pretraining_artifact_manager()

# Get features from feature_generation_feature_generation_step
feature_artifacts = artifact_manager.get_step_artifacts('feature_generation_feature_generation_step')
feature_df = feature_artifacts.get('feature_dataframe')
feature_names = feature_artifacts.get('feature_names')

# Get targets from feature_generation_labeling_integration_step
labeling_artifacts = artifact_manager.get_step_artifacts('feature_generation_labeling_integration_step')
labeled_df = labeling_artifacts.get('labeled_dataframe')
targets = labeling_artifacts.get('targets')
```

### Saving Artifacts in a Step
```python
# In feature_generation_period_lookback_optimization_step
artifact_manager.save(
    step_name='feature_generation_period_lookback_optimization_step',
    artifacts={
        'optimized_periods': optimized_periods,
        'optimized_lookbacks': optimized_lookbacks,
        'mi_best_lookbacks_per_feature': mi_results,
        'mrmr_top_lookbacks_per_feature': mrmr_results,
        'mi_scores_by_feature': mi_scores,
        'oos_sharpe_by_feature_window': sharpe_scores,
        'selected_features_metadata': metadata,
        'family_diagnostics': diagnostics
    }
)
```

### Manual Correspondence Validation
```python
# Use the separate validator to check correspondence
from src.training.steps.pre_training.utils.artifact_correspondence_validator import (
    check_step_correspondence, check_all_correspondences, get_retrieval_guide
)

# Check correspondence for a specific step
check_step_correspondence('feature_generation_period_lookback_optimization_step')

# Check all correspondences
check_all_correspondences()

# Get retrieval guide for a step
get_retrieval_guide('feature_generation_period_lookback_optimization_step')
```

## Key Benefits

1. **Clear Correspondence**: Each step knows exactly what it creates and what it needs
2. **Validation**: Built-in validation ensures artifacts exist before steps run
3. **Step-Agnostic**: Artifact manager remains generic and reusable
4. **Traceability**: Easy to track data flow through the pipeline
5. **Error Prevention**: Missing dependencies are caught early

This correspondence ensures that the pre-training pipeline runs smoothly with proper data flow between steps.
