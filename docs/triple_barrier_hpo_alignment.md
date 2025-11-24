# Triple Barrier HPO Alignment

## Overview

This document explains the programmatic alignment between `meta_labeling_hpo_experiment` results and triple barrier configuration.

## Problem

Previously, the HPO experiment would find optimal parameters for labeling, but there was no standardized way to apply these parameters to create a properly configured `OptimizedTripleBarrierLabeling` instance. This led to:

- Manual parameter copying between systems
- Potential misalignment between HPO results and production settings
- Inconsistent usage across different parts of the codebase

## Solution

The `create_triple_barrier_from_hpo()` function provides programmatic alignment by:

1. **Automatically loading** the latest HPO results from `outcomes/` directory
2. **Extracting** relevant parameters (profit_thr_base, stop_to_profit_ratio, horizon_bars, etc.)
3. **Converting** HPO parameters to triple barrier format
4. **Creating** a properly configured `OptimizedTripleBarrierLabeling` instance
5. **Falling back** to sensible defaults if HPO results are not found

## Usage

### Basic Usage

```python
from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    create_triple_barrier_from_hpo
)

# Create triple barrier labeler aligned with HPO results
labeler, hpo_params, used_hpo = create_triple_barrier_from_hpo(
    symbol='ETHUSDT',
    timeframe='15m',
    binary_classification=True,
    transaction_cost=0.0008
)

# Use the labeler
labels, profits, metadata = labeler.generate_labels(market_data)
```

### With Custom Fallbacks

```python
# Create labeler with custom fallback parameters (used if HPO not found)
labeler, hpo_params, used_hpo = create_triple_barrier_from_hpo(
    symbol='ETHUSDT',
    timeframe='15m',
    fallback_profit_take=0.006,    # 0.6% profit target
    fallback_stop_loss=0.004,      # 0.4% stop loss
    fallback_time_barrier=60,      # 60 minutes
    fallback_max_lookahead=80,     # 80 bars
    binary_classification=True,
    transaction_cost=0.001
)
```

## Parameter Mapping

The function maps HPO parameters to triple barrier settings as follows:

| HPO Parameter | Triple Barrier Parameter | Conversion |
|---------------|-------------------------|------------|
| `profit_thr_base` | `profit_take_multiplier` | Direct copy |
| `stop_to_profit_ratio` | `stop_loss_multiplier` | `profit_take × ratio` (min 0.05%) |
| `horizon_bars` | `time_barrier_minutes` | `horizon × timeframe_minutes` (cap at 240min) |
| `horizon_bars` | `max_lookahead` | Direct copy (bounded 10-200) |

Additional HPO parameters used elsewhere in the pipeline:
- `min_event_spacing`: Minimum bars between events
- `kalman_Q`, `kalman_R`: Kalman filter parameters for volatility smoothing
- `vol_baseline_window`: Baseline window for volatility calculation
- `profit_mult_min/max`, `stop_mult_min/max`: Multiplier bounds for adaptation

## HPO Results Location

The function looks for HPO results in:
```
outcomes/meta_labeling_hpo_best_params_{symbol}_{timeframe}_*.json
```

Example: `outcomes/meta_labeling_hpo_best_params_ETHUSDT_15m_20241120_153045.json`

## Parameter Priority

The function prefers parameters in this order:
1. **knee_params**: Pareto knee point (balanced trade-off)
2. **best_params**: Highest scoring configuration
3. **Fallback defaults**: If no HPO results found

## Return Values

The function returns a tuple of:
- `labeler`: Configured `OptimizedTripleBarrierLabeling` instance
- `hpo_params`: Dictionary of HPO parameters (or empty dict if not found)
- `used_hpo`: Boolean indicating whether HPO params were found and used

## Integration Points

The alignment function can be used in:

1. **Feature Generation**: During training to generate labels with HPO-aligned barriers
2. **Backtesting**: To ensure backtest uses same barriers as HPO-optimized settings
3. **Live Trading**: To apply HPO-discovered barriers in production
4. **Research**: To experiment with HPO-aligned parameters

## Example: Full Workflow

```python
# 1. Run HPO experiment to find optimal parameters
# This generates: outcomes/meta_labeling_hpo_best_params_ETHUSDT_15m_*.json

# 2. In training pipeline, use the aligned labeler
from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    create_triple_barrier_from_hpo
)

labeler, hpo_params, used_hpo = create_triple_barrier_from_hpo(
    symbol='ETHUSDT',
    timeframe='15m',
    binary_classification=True,
    transaction_cost=0.0008
)

if used_hpo:
    print(f"✅ Using HPO-aligned parameters:")
    print(f"  Profit: {labeler.profit_take_multiplier:.4f}")
    print(f"  Stop: {labeler.stop_loss_multiplier:.4f}")
else:
    print(f"ℹ️ Using fallback parameters")

# 3. Generate labels
labels, profits, metadata = labeler.generate_labels(market_data)

# 4. Train models using these labels
# ... rest of pipeline ...
```

## Validation

The `OptimizedTripleBarrierLabeling` class automatically validates all parameters:

- Profit take: must be 0.1% - 10%
- Stop loss: must be 0.05% - 5%
- Risk-reward ratio: must be >= 1.0
- Transaction cost: must be 0% - 1%
- Barriers must be sufficiently separated (> 0.1%)

If validation fails, the function falls back to very conservative settings:
- Profit: 0.5%
- Stop: 0.4%
- Time barrier: 30 minutes
- Max lookahead: 50 bars

## Benefits

1. **Consistency**: Ensures triple barrier settings match HPO-discovered optimal parameters
2. **Automation**: No manual parameter copying needed
3. **Traceability**: Clear logging shows which parameters are being used and from where
4. **Safety**: Automatic validation prevents invalid parameter combinations
5. **Flexibility**: Easy to override fallback parameters for different instruments
6. **Robustness**: Graceful fallback if HPO results unavailable

## Future Enhancements

Potential improvements:
- Support for per-regime triple barrier parameters
- Automatic parameter refresh when new HPO results are generated
- Parameter interpolation for instruments without HPO results
- Multi-symbol HPO result aggregation
- A/B testing framework for comparing HPO vs fallback parameters

## See Also

- `src/training/steps/labeling/meta_labeling_hpo_experiment_step.py` - HPO experiment implementation
- `src/feature_generation/utils/step06_labeling_components/optimized_triple_barrier_labeling.py` - Triple barrier implementation
- `examples/triple_barrier_hpo_alignment_example.py` - Complete usage examples
