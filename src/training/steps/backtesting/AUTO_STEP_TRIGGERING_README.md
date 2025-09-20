# Automatic Step Triggering for Backtesting Pipeline

This system ensures that when one backtesting step finishes, it automatically triggers the next step, creating a seamless execution flow.

## Overview

The backtesting pipeline consists of 7 steps that run in sequence:

1. **basic_backtesting_pre** - Pre-optimization baseline backtesting
2. **final_parameters_optimization** - System-wide parameter optimization
3. **basic_backtesting_post** - Post-optimization comparison backtesting
4. **walk_forward_validation** - Walk-forward backtesting
5. **monte_carlo_simulation** - Monte Carlo backtesting
6. **ab_testing** - A/B testing for strategies
7. **reporting** - Comprehensive reporting

## How It Works

The automatic triggering system is built on top of the existing `sub_pipeline.py` infrastructure. When a step completes successfully, the system automatically calls the next step in the sequence, ensuring continuous execution without manual intervention.

## Usage

### Method 1: Using the Auto Step Trigger Module

```python
from src.training.steps.backtesting.auto_step_trigger import (
    auto_execute_all_backtesting_steps,
    auto_execute_from_step
)

# Execute all steps automatically from the beginning
result = await auto_execute_all_backtesting_steps(
    symbol="ETHUSDT",  # Any symbol: BTCUSDT, ADAUSDT, etc.
    exchange="BINANCE",  # Any exchange: BYBIT, KRAKEN, etc.
    timeframe="1m",  # Any timeframe: 5m, 15m, 1h, etc.
    force_rerun=True
)

# Or execute from a specific step (will trigger all subsequent steps)
result = await auto_execute_from_step(
    step_name='walk_forward_validation',  # Will trigger steps 4-7
    symbol="ETHUSDT",
    exchange="BINANCE",
    timeframe="1m",
    force_rerun=True
)
```

### Method 2: Using the Sub-Pipeline Directly

```python
from src.training.steps.backtesting.sub_pipeline import (
    BacktestingSubPipeline,
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
sub_pipeline = BacktestingSubPipeline(config)

# Execute all steps from the beginning
result = await sub_pipeline.execute_all_steps_from_start(config)

# Execute from a specific step
result = await sub_pipeline.execute_sub_pipeline_with_next('monte_carlo_simulation', config)
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

### Integration
- Works with existing pipeline infrastructure
- Compatible with ares_launcher parameters
- Supports different symbols, exchanges, and timeframes

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
        'walk_forward_validation': {
            'window_size': 30,
            'step_size': 7,
            'min_samples': 100
        },
        'monte_carlo_simulation': {
            'n_simulations': 1000,
            'confidence_level': 0.95
        },
        'ab_testing': {
            'test_duration_days': 30,
            'significance_level': 0.05
        }
    }
}
```

## Example Output

```
🚀 Starting automatic execution of all 7 backtesting steps
================================================================================
📋 Steps to be executed automatically:
   1. basic_backtesting_pre - Pre-optimization baseline backtesting
   2. final_parameters_optimization - System-wide parameter optimization
   3. basic_backtesting_post - Post-optimization comparison backtesting
   4. walk_forward_validation - Walk-forward backtesting
   5. monte_carlo_simulation - Monte Carlo backtesting
   6. ab_testing - A/B testing for strategies
   7. reporting - Comprehensive reporting
================================================================================

🎉 Automatic execution completed
✅ Successful steps: 7/7
⏱️ Total execution time: 1847.32 seconds
```

## Files Created/Modified

1. **sub_pipeline.py** - Enhanced with automatic triggering capabilities
2. **auto_step_trigger.py** - New module with convenience functions
3. **example_auto_triggering.py** - Example usage demonstrations

## Testing

Run the example script to test the automatic triggering:

```bash
python src/training/steps/backtesting/example_auto_triggering.py
```

## Benefits

1. **Seamless Execution**: No manual intervention required between steps
2. **Error Recovery**: Clear failure reporting and error handling
3. **Flexibility**: Can start from any step in the sequence
4. **Monitoring**: Comprehensive progress tracking and logging
5. **Integration**: Works with existing pipeline infrastructure
6. **Scalability**: Supports different execution modes and configurations
7. **Agnostic**: Works with any symbol, exchange, and timeframe

## Integration with ares_launcher

The system is designed to work seamlessly with `ares_launcher`, accepting parameters like:

- **symbol**: Trading symbol (e.g., 'ETHUSDT', 'BTCUSDT')
- **exchange**: Exchange name (e.g., 'BINANCE', 'BYBIT')
- **timeframe**: Data timeframe (e.g., '1m', '5m', '1h')

These parameters are passed through without hardcoded defaults, making the system truly agnostic to the specific trading pair or analysis timeframe.

This system ensures that your backtesting pipeline runs smoothly from start to finish, automatically progressing through all 7 steps when each completes successfully.
