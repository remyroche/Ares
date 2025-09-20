# Automatic Step Triggering for Market Analysis Pipeline

This system ensures that when one market analysis step finishes, it automatically triggers the next step, creating a seamless execution flow.

## Overview

The market analysis pipeline consists of 11 steps that run in sequence:

1. **sr_parameter_optimization** - Optimize SR detection levels
2. **sr_detection** - Detect Support/Resistance levels
3. **sr_clustering** - Generate SR clusters
4. **hmm_regime_discovery** - Discover market regimes
5. **hmm_clustering** - HMM-based regime clustering
6. **hmm_models_training** - Base models training, HPO, saving, metrics
7. **hmm_ensemble_training** - Meta-model, HPO, saving, metrics
8. **regime_data_splitting** - Tag data by regimes
9. **multi_horizon_profit_labeler** - Apply triple barrier method
10. **feature_lookback_optimization** - Optimize feature lookback periods
11. **pid_based_feature_generation** - Cross timeframe interaction features

## How It Works

The automatic triggering system is built on top of the existing `sub_pipeline.py` infrastructure. When a step completes successfully, the system automatically calls the next step in the sequence, ensuring continuous execution without manual intervention.

## Usage

### Method 1: Using the Auto Step Trigger Module

```python
from src.training.steps.market_analysis.auto_step_trigger import (
    auto_execute_all_market_analysis_steps,
    auto_execute_from_step
)

# Execute all steps from the beginning
result = await auto_execute_all_market_analysis_steps(
    symbol="ETHUSDT",
    exchange="BINANCE",
    timeframe="1m",
    data_dir="historical_data",
    force_rerun=True
)

# Execute from a specific step (will trigger all subsequent steps)
result = await auto_execute_from_step(
    step_name='hmm_clustering',
    symbol="ETHUSDT",
    exchange="BINANCE", 
    timeframe="1m",
    data_dir="historical_data",
    force_rerun=True
)
```

### Method 2: Using the Enhanced Orchestrator

```python
from src.training.steps.market_analysis.enhanced_market_analysis_orchestrator import (
    run_auto_triggering_market_analysis_pipeline
)

# Execute all steps from the beginning
result = await run_auto_triggering_market_analysis_pipeline(
    symbol="ETHUSDT",
    exchange="BINANCE",
    timeframe="1m",
    data_dir="historical_data",
    force_rerun=True
)

# Execute from a specific step
result = await run_auto_triggering_market_analysis_pipeline(
    symbol="ETHUSDT",
    exchange="BINANCE",
    timeframe="1m", 
    data_dir="historical_data",
    start_from_step="regime_data_splitting",
    force_rerun=True
)
```

### Method 3: Using the Sub-Pipeline Directly

```python
from src.training.steps.market_analysis.sub_pipeline import (
    MarketAnalysisSubPipeline,
    SubPipelineConfig,
    ExecutionMode
)

# Create configuration
config = SubPipelineConfig(
    mode=ExecutionMode.FULL,
    symbol="ETHUSDT",
    exchange="BINANCE",
    timeframe="1m",
    data_dir="historical_data",
    force_rerun=True,
    single_stage_only=False  # Enable automatic triggering
)

# Create and execute sub-pipeline
sub_pipeline = MarketAnalysisSubPipeline(config)

# Execute all steps from the beginning
result = await sub_pipeline.execute_all_steps_from_start(config)

# Execute from a specific step
result = await sub_pipeline.execute_sub_pipeline_with_next('hmm_clustering', config)
```

## Key Features

### Automatic Triggering
- When one step completes successfully, the next step is automatically triggered
- No manual intervention required between steps
- Seamless execution flow

### Error Handling
- If a step fails, the execution stops and reports the failure
- Detailed error messages and execution summaries
- Comprehensive logging throughout the process

### Flexible Execution
- Can start from any step in the sequence
- Can execute all steps from the beginning
- Supports different execution modes (FULL, LIGHT, BLANK)

### Progress Monitoring
- Real-time progress updates
- Execution time tracking
- Success/failure statistics
- Detailed execution summaries

## Configuration Options

### Basic Configuration
```python
config = {
    'force_rerun': True,           # Force rerun existing artifacts
    'parallel_processing': True,   # Enable parallel processing
    'validation_enabled': True,    # Enable validation
    'monitoring_enabled': True,    # Enable monitoring
    'fast_mode': False,            # Enable fast mode
}
```

### Advanced Configuration
```python
config = {
    'custom_params': {
        'multi_horizon_labeling': {
            'horizons': [5, 10, 20],
            'volatility_threshold': 0.02
        },
        'feature_lookback_optimization': {
            'max_lookback': 50,
            'optimization_method': 'bayesian'
        }
    }
}
```

## Example Output

```
🚀 Starting automatic execution of all 11 market analysis steps
================================================================================
📋 Steps to be executed automatically:
   1. sr_parameter_optimization - Optimize SR detection levels
   2. sr_detection - Detect Support/Resistance levels
   3. sr_clustering - Generate SR clusters
   4. hmm_regime_discovery - Discover market regimes
   5. hmm_clustering - HMM-based regime clustering
   6. hmm_models_training - Base models training, HPO, saving, metrics
   7. hmm_ensemble_training - Meta-model, HPO, saving, metrics
   8. regime_data_splitting - Tag data by regimes
   9. multi_horizon_profit_labeler - Apply triple barrier method
   10. feature_lookback_optimization - Optimize feature lookback periods
   11. pid_based_feature_generation - Cross timeframe interaction features
================================================================================

🎉 Automatic execution completed
✅ Successful steps: 11/11
⏱️ Total execution time: 1847.32 seconds
```

## Files Created/Modified

1. **sub_pipeline.py** - Enhanced with automatic triggering capabilities
2. **auto_step_trigger.py** - New module with convenience functions
3. **enhanced_market_analysis_orchestrator.py** - Updated with auto-triggering support
4. **example_auto_triggering.py** - Example usage demonstrations

## Testing

Run the example script to test the automatic triggering:

```bash
python src/training/steps/market_analysis/example_auto_triggering.py
```

## Benefits

1. **Seamless Execution**: No manual intervention required between steps
2. **Error Recovery**: Clear failure reporting and error handling
3. **Flexibility**: Can start from any step in the sequence
4. **Monitoring**: Comprehensive progress tracking and logging
5. **Integration**: Works with existing pipeline infrastructure
6. **Scalability**: Supports different execution modes and configurations

This system ensures that your market analysis pipeline runs smoothly from start to finish, automatically progressing through all 11 steps when each completes successfully.
