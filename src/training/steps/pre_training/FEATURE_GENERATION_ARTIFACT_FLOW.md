# Feature Generation Artifact Flow

This document describes the artifact flow between the feature generation steps in the pre_training pipeline, ensuring proper artifact creation and consumption patterns.

## Overview

The feature generation pipeline consists of several interconnected steps that create and consume artifacts in a specific order. Each step produces artifacts that subsequent steps can consume, creating a clear dependency chain.

## Artifact Manager

The `utils/artifact_manager.py` provides a centralized artifact management system specifically designed for pre_training steps. It includes:

- **PreTrainingArtifactManager**: Main class for managing artifacts
- **Context Management**: Sets execution context (symbol, exchange, timeframe, etc.)
- **Artifact Storage**: Uses enhanced artifact manager with compression and metadata
- **Artifact Retrieval**: Loads artifacts with proper naming and versioning

## Step Dependencies and Artifact Flow

### 1. Feature Generation Period Lookback Optimization Step
**File**: `feature_generation_period_lookback_optimization_step.py`

**Consumes**:
- `feature_generation_feature_generation_step` → `feature_lists`
- `feature_generation_labeling_integration_step` → `labels`

**Produces**:
- `optimized_periods`: Dictionary with top performing periods (top1, top2-3)
- `period_performance_metrics`: Performance metrics for each tested period
- `optimization_report`: Detailed report of the optimization process

**Purpose**: Tests different lookback periods and selects the most effective ones based on feature performance against labels.

### 2. Feature Generation Feature Selection Step
**File**: `feature_generation_feature_selection_step.py`

**Consumes**:
- `feature_generation_feature_generation_step` → `feature_lists`
- `feature_generation_period_lookback_optimization_step` → `optimized_periods` (top1)
- `feature_generation_labeling_integration_step` → `labels`

**Produces**:
- `selected_features`: Dictionary of selected features by category
- `feature_importance_scores`: Importance scores for each feature
- `selection_report`: Detailed report of the selection process

**Purpose**: Selects the most relevant features using the top1 periods from optimization.

### 3. Feature Generation Interaction Generation Step - Analyst
**File**: `feature_generation_interaction_generation_step_analyst.py`

**Consumes**:
- `feature_generation_feature_selection_step` → `selected_features`
- `feature_generation_period_lookback_optimization_step` → `optimized_periods` (top2-3)
- `feature_generation_labeling_integration_step` → `labels`

**Produces**:
- `interaction_features`: Generated interaction features
- `interaction_performance_metrics`: Performance metrics for interactions
- `interaction_report`: Detailed report of the interaction generation process

**Purpose**: Generates interaction features for the Analyst model using top2-3 periods.

### 4. Feature Generation Interaction Generation Step - Tactician
**File**: `feature_generation_interaction_generation_step_tactician.py`

**Consumes**:
- `feature_generation_feature_selection_step` → `selected_features`
- `feature_generation_period_lookback_optimization_step` → `optimized_periods` (top2-3)
- `feature_generation_labeling_integration_step` → `labels`

**Produces**:
- `interaction_features`: Generated interaction features
- `interaction_performance_metrics`: Performance metrics for interactions
- `interaction_report`: Detailed report of the interaction generation process

**Purpose**: Generates interaction features for the Tactician model using top2-3 periods.

### 5. Feature Generation Final Feature Selection Step
**File**: `feature_generation_final_feature_selection_step.py`

**Consumes**:
- `feature_generation_feature_generation_step` → `feature_lists`
- `feature_generation_period_lookback_optimization_step` → `optimized_periods` (top1)
- `feature_generation_interaction_generation_step_analyst` → `interaction_features`
- `feature_generation_interaction_generation_step_tactician` → `interaction_features`
- `feature_generation_labeling_integration_step` → `labels`

**Produces**:
- `final_selected_features`: Final set of selected features
- `feature_ranking`: Ranking of all features by importance
- `final_selection_report`: Comprehensive report of the final selection process

**Purpose**: Performs the final feature selection by combining base features and interaction features.

## Artifact Naming Convention

Artifacts are named using the following pattern:
```
{information}_{step_name}_{artifact_name}_{symbol}_{exchange}_{timeframe}_{timestamp}
```

Example:
```
pre_training_feature_generation_period_lookback_optimization_step_optimized_periods_ETHUSDT_binance_15m_20240101_120000
```

## Usage Example

```python
from utils.artifact_manager import get_pretraining_artifact_manager

# Get artifact manager
am = get_pretraining_artifact_manager()

# Set context
am.set_context(
    symbol='ETHUSDT',
    exchange='binance',
    timeframe='15m',
    direction='long',
    model='Analyst'
)

# Save an artifact
am.save_artifact(
    'feature_generation_period_lookback_optimization_step',
    'optimized_periods',
    {'top1': [20], 'top2_3': [10, 30]},
    metadata={'test_periods': [5, 10, 15, 20, 30]}
)

# Load an artifact
optimized_periods = am.load_artifact(
    'feature_generation_period_lookback_optimization_step',
    'optimized_periods'
)
```

## Error Handling

Each step includes comprehensive error handling:

1. **Missing Artifacts**: Steps provide fallback data when required artifacts are not found
2. **Artifact Validation**: Artifacts are validated before use
3. **Logging**: Detailed logging for debugging and monitoring
4. **Graceful Degradation**: Steps continue with reduced functionality when possible

## Component Registration

All steps are automatically registered in the component registry (`component_registry.py`) and can be accessed through the ComponentFactory:

```python
from src.training.steps.pre_training.components.component_factory import ComponentFactory

# Register all components
from src.training.steps.pre_training.components.component_registry import ComponentRegistry
ComponentRegistry.register_all_components()

# Get a step
step = ComponentFactory.create_component('feature_generation_period_lookback_optimization_step', config)
```

## Testing

Each step includes comprehensive testing capabilities:

1. **Unit Tests**: Individual step functionality
2. **Integration Tests**: Artifact flow between steps
3. **Mock Data**: Synthetic data generation for testing
4. **Validation**: Artifact format and content validation

## Performance Considerations

1. **Caching**: Artifacts are cached for frequently accessed data
2. **Compression**: Large artifacts are compressed to save space
3. **Lazy Loading**: Artifacts are loaded only when needed
4. **Memory Management**: Large datasets are processed in chunks when possible

## Future Enhancements

1. **Parallel Processing**: Multiple steps can run in parallel when dependencies allow
2. **Artifact Versioning**: Support for multiple versions of the same artifact
3. **Artifact Cleanup**: Automatic cleanup of old artifacts
4. **Metrics Collection**: Performance metrics for each step
5. **Artifact Dependencies**: Automatic dependency resolution