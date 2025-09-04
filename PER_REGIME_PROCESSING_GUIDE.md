# Per-HMM Regime Processing Implementation Guide

## Overview

This guide documents the implementation of per-HMM regime processing for steps 4-21 in the trading pipeline. The implementation ensures that each task (training, optimizing, etc.) is performed on a per-regime basis using consistent methods.

## Key Components

### 1. Regime Handler (`regime_handler.py`)

The central component that provides consistent access to regime data across all steps:

- **`load_unified_regime_data()`**: Loads the unified regime dataset created by Step 4
- **`get_regime_ids()`**: Extracts unique regime IDs from the data
- **`filter_data_by_regime()`**: Filters data for a specific regime with optional temporal context
- **`process_per_regime()`**: Processes data for each regime using a provided function
- **`save_regime_results()`**: Saves per-regime processing results
- **`load_regime_results()`**: Loads previously saved per-regime results

### 2. Regime Processing Decorator (`regime_processing_decorator.py`)

Provides decorators and utilities for automatic per-regime processing:

- **`@per_regime_processing`**: Decorator that automatically handles per-regime execution
- **`aggregate_regime_results()`**: Aggregates per-regime results using various methods
- **`RegimeProcessingContext`**: Context manager for regime-specific processing

### 3. Pipeline Integration (`per_regime_pipeline_integration.py`)

Integrates per-regime processing into the existing pipeline:

- **`PerRegimePipelineIntegrator`**: Main integration class
- **`get_step_function()`**: Dynamically loads per-regime or standard step functions
- **`update_step_config_for_regime()`**: Updates configuration for regime-specific processing
- **`verify_regime_data_availability()`**: Verifies regime data is available

## Implementation Pattern

### For New Steps

```python
from src.training.steps.regime_processing_decorator import per_regime_processing

@per_regime_processing(result_type='your_results', parallel=True)
async def process_your_step_regime(
    data: pd.DataFrame,
    regime_id: int,
    **kwargs
) -> pd.DataFrame:
    """Process your step for a single regime."""
    # Your regime-specific logic here
    result = your_processing_function(data, **kwargs)
    result['regime_id'] = regime_id
    return result
```

### For Existing Steps

1. Create a new file: `step{N}_{name}_per_regime.py`
2. Import the original step class/functions
3. Create a per-regime version that inherits or wraps the original
4. Implement regime-specific logic
5. Export a `run_per_regime_step()` function

## Step-by-Step Implementation Status

| Step | Name | Per-Regime Status | Implementation File |
|------|------|------------------|-------------------|
| 4 | Regime Data Splitting | N/A (Creates regime data) | Original |
| 5 | Labeling | ✅ Implemented | `step05_labeling_per_regime.py` |
| 6 | Feature Engineering | ✅ Implemented | `step06_feature_engineering_per_regime.py` |
| 7 | Enhanced Matrix Operations | 📝 Template Created | Template available |
| 8 | Advanced Feature Selection | 📝 Template Created | Template available |
| 9 | HMM Based Training | 📝 Template Created | Template available |
| 10 | Unified Regime Intelligence | ⏳ Pending | - |
| 11 | Analyst Creation | ⏳ Pending | - |
| 12 | Analyst Enhancement | ⏳ Pending | - |
| 13 | Analyst Ensemble Creation | ⏳ Pending | - |
| 14 | Tactician Labeling | ⏳ Pending | - |
| 15 | Tactician Specialist Training | ⏳ Pending | - |
| 16 | Confidence Calibration | ⏳ Pending | - |
| 17 | Final Parameters Optimization | ⏳ Pending | - |
| 18 | Walk Forward Validation | ⏳ Pending | - |
| 19 | Monte Carlo Validation | ⏳ Pending | - |
| 20 | A/B Testing | ⏳ Pending | - |
| 21 | Saving | ⏳ Pending | - |

## Configuration

### Enable Per-Regime Processing

```json
{
  "per_regime_processing": true,
  "pipeline_settings": {
    "per_regime_processing_enabled": true,
    "regime_coherence_method": "unified_regime_handler",
    "parallel_regime_processing": true,
    "preserve_temporal_context": true,
    "context_window_size": 100
  }
}
```

### Regime-Specific Parameters

```json
{
  "regime_specific_params": {
    "regime_0": {
      "step05_labeling": {
        "time_barrier_minutes": 45,
        "max_lookahead": 150
      },
      "step06_feature_engineering": {
        "lookback_periods": [10, 20, 50, 100, 200],
        "emphasis": "trend"
      }
    }
  }
}
```

## Usage in Main Pipeline

### Update Pipeline File (e.g., `temp_fixed2.py`)

```python
from src.training.steps.per_regime_pipeline_integration import per_regime_integrator

# For each step:
step_func = await per_regime_integrator.get_step_function('step05_labeling')
step_config = per_regime_integrator.update_step_config_for_regime('step05_labeling', config)

success = await step_func(
    symbol=symbol,
    exchange=exchange,
    timeframe=timeframe,
    data_dir=data_dir,
    config=step_config
)
```

## Benefits

1. **Regime-Specific Optimization**: Each regime can have parameters optimized for its specific characteristics
2. **Better Model Performance**: Models trained on regime-specific data capture regime dynamics better
3. **Consistent Data Access**: All steps use the same unified regime handler
4. **Temporal Context Preservation**: Important for technical indicators that need lookback periods
5. **Parallel Processing**: Regimes can be processed in parallel for faster execution

## Best Practices

1. **Always Preserve Context**: Use `preserve_context=True` when filtering regime data for technical indicators
2. **Handle Context Rows**: Mark context rows appropriately (e.g., label=-999) to exclude from training
3. **Aggregate Intelligently**: Choose appropriate aggregation method based on the step's output
4. **Log Regime Statistics**: Always log per-regime statistics for debugging and analysis
5. **Save Regime Metadata**: Store regime-specific configurations and results for traceability

## Troubleshooting

### Common Issues

1. **Missing Regime Data**: Ensure Step 4 has been run successfully
2. **Empty Regime Results**: Check if regime has sufficient data after filtering
3. **Context Window Too Large**: Reduce context_window if regimes are too short
4. **Memory Issues**: Process regimes sequentially instead of in parallel

### Debugging

```python
# Verify regime data
from src.training.steps.regime_handler import regime_handler

data = await regime_handler.load_unified_regime_data(symbol, exchange, timeframe, data_dir)
regime_ids = regime_handler.get_regime_ids(data)
stats = regime_handler.get_regime_statistics(data)
```

## Next Steps

1. Implement per-regime versions for remaining steps (7-21)
2. Add regime transition handling for validation steps
3. Implement cross-regime model comparison
4. Add regime-specific performance metrics
5. Create visualization tools for per-regime results