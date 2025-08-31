# Regime-Specific Triple Barrier Optimization Guide

## Overview

This guide explains how to implement and use the per-HMM regime triple barrier thresholds and optimized TPSL (Take Profit/Stop Loss) parameters system. The implementation extends the existing Optuna optimization framework to provide regime-aware parameter tuning for each HMM regime.

## Key Features

- **Regime-Specific Optimization**: Each HMM regime gets its own optimized triple barrier thresholds and TPSL parameters
- **Multi-Objective Optimization**: Balances performance metrics (Sharpe ratio, win rate, profit factor) with regime-specific accuracy
- **Comprehensive Parameter Space**: Optimizes triple barrier thresholds, TPSL parameters, and regime-specific multipliers
- **Advanced Performance Tracking**: Tracks performance metrics per regime with statistical significance testing
- **Visualization and Reporting**: Generates comprehensive reports and visualizations for optimization results
- **Integration Ready**: Seamlessly integrates with existing pipeline components

## Architecture

### Core Components

1. **RegimeSpecificTripleBarrierOptimizer**: Main optimization engine
2. **RegimeAwareTripleBarrierLabeling**: Regime-aware labeling component
3. **RegimeSpecificOptimizationConfig**: Configuration management
4. **RegimeTripleBarrierConfig**: Regime-specific parameter configuration

### File Structure

```
src/
├── training/steps/step17_final_parameters_optimization/
│   └── regime_specific_triple_barrier_optimization.py
├── training/steps/step4_analyst_labeling_feature_engineering_components/
│   └── regime_aware_triple_barrier_labeling.py
└── config/
    └── regime_specific_optimization_config.py
```

## Quick Start

### 1. Basic Usage

```python
import asyncio
import pandas as pd
from src.training.steps.step17_final_parameters_optimization.regime_specific_triple_barrier_optimization import (
    optimize_regime_triple_barrier_parameters
)

# Load your data with regime information
data = pd.read_parquet("your_data_with_regimes.parquet")

# Create configuration
config = {
    "regime_specific_optimization": {
        "n_trials_per_regime": 100,
        "timeout_minutes_per_regime": 60,
        "objectives": ["sharpe_ratio", "win_rate", "profit_factor"],
        "objective_weights": {
            "sharpe_ratio": 0.4,
            "win_rate": 0.3,
            "profit_factor": 0.3
        }
    }
}

# Run optimization
results = await optimize_regime_triple_barrier_parameters(
    data=data,
    config=config,
    regime_column="composite_cluster_id"
)

# Use optimized parameters
for regime_name, result in results.items():
    print(f"Regime: {regime_name}")
    print(f"Optimized TP Multiplier: {result.triple_barrier_params.profit_take_multiplier}")
    print(f"Optimized SL Multiplier: {result.triple_barrier_params.stop_loss_multiplier}")
    print(f"Sharpe Ratio: {result.sharpe_ratio}")
```

### 2. Regime-Aware Labeling

```python
from src.training.steps.step4_analyst_labeling_feature_engineering_components.regime_aware_triple_barrier_labeling import (
    apply_regime_aware_triple_barrier_labeling,
    create_regime_aware_labeler_from_optimization_results
)

# Create labeler from optimization results
labeler = create_regime_aware_labeler_from_optimization_results(results)

# Apply regime-aware labeling
labeled_data = labeler.apply_regime_aware_triple_barrier_labeling(
    data, 
    regime_column="composite_cluster_id"
)

# Or use the utility function directly
labeled_data = apply_regime_aware_triple_barrier_labeling(
    data=data,
    optimization_results=results,
    regime_column="composite_cluster_id"
)
```

## Configuration

### Regime-Specific Constraints

Each regime can have different parameter constraints based on market characteristics:

```python
from src.config.regime_specific_optimization_config import (
    RegimeSpecificOptimizationConfig,
    RegimeSpecificConstraints
)

config = RegimeSpecificOptimizationConfig()

# Configure constraints for different regimes
config.regime_constraints = {
    "BULL_TREND": RegimeSpecificConstraints(
        tp_multiplier_range=[2.5, 5.0],
        sl_multiplier_range=[1.2, 2.5],
        position_size_range=[0.10, 0.25],
        profit_take_multiplier_range=[0.02, 0.04],
        stop_loss_multiplier_range=[0.01, 0.02],
    ),
    "BEAR_TREND": RegimeSpecificConstraints(
        tp_multiplier_range=[2.0, 4.5],
        sl_multiplier_range=[1.0, 2.2],
        position_size_range=[0.08, 0.20],
        profit_take_multiplier_range=[0.015, 0.035],
        stop_loss_multiplier_range=[0.008, 0.018],
    ),
    "SIDEWAYS_RANGE": RegimeSpecificConstraints(
        tp_multiplier_range=[1.5, 3.0],
        sl_multiplier_range=[0.8, 1.8],
        position_size_range=[0.06, 0.15],
        profit_take_multiplier_range=[0.01, 0.025],
        stop_loss_multiplier_range=[0.005, 0.015],
    ),
}
```

### Optimization Objectives

Configure multi-objective optimization with custom weights:

```python
config.objectives = ["sharpe_ratio", "win_rate", "profit_factor", "regime_accuracy"]
config.objective_weights = {
    "sharpe_ratio": 0.3,
    "win_rate": 0.25,
    "profit_factor": 0.25,
    "regime_accuracy": 0.2
}
```

## Advanced Usage

### 1. Custom Regime Mapping

```python
from src.training.steps.step4_analyst_labeling_feature_engineering_components.regime_aware_triple_barrier_labeling import (
    RegimeAwareTripleBarrierLabeling,
    RegimeTripleBarrierConfig
)

# Create custom configuration
config = RegimeTripleBarrierConfig()

# Set regime mapping
regime_mapping = {
    0: "BULL_TREND",
    1: "BEAR_TREND", 
    2: "SIDEWAYS_RANGE",
    3: "HIGH_VOLATILITY",
    4: "LOW_VOLATILITY"
}
config.regime_id_to_name = regime_mapping

# Set regime-specific parameters
config.regime_profit_take_multipliers["BULL_TREND"] = 0.03
config.regime_stop_loss_multipliers["BULL_TREND"] = 0.015
config.regime_tp_multipliers["BULL_TREND"] = 3.0
config.regime_sl_multipliers["BULL_TREND"] = 1.5

# Create labeler
labeler = RegimeAwareTripleBarrierLabeling(config)
```

### 2. Performance Analysis

```python
# Get performance summary by regime
performance_summary = labeler.get_regime_performance_summary(
    labeled_data, 
    regime_column="composite_cluster_id"
)

for regime_name, metrics in performance_summary.items():
    print(f"\nRegime: {regime_name}")
    print(f"  Total Samples: {metrics['total_samples']}")
    print(f"  Valid Samples: {metrics['valid_samples']}")
    print(f"  Win Rate: {metrics['win_rate']:.4f}")
    print(f"  Avg Profit: {metrics['avg_profit']:.4f}")
    print(f"  Total Return: {metrics['total_return']:.4f}")
```

### 3. Integration with Existing Pipeline

```python
# In your existing step4_triple_barrier_method.py
async def execute_triple_barrier_method(self, symbol, exchange, timeframe, data_dir):
    # Load data with regime information
    data = self._load_data_with_regimes(data_dir, symbol, exchange, timeframe)
    
    # Check if regime-specific optimization results exist
    optimization_results = self._load_optimization_results(symbol, exchange, timeframe)
    
    if optimization_results:
        # Use regime-aware labeling
        from src.training.steps.step4_analyst_labeling_feature_engineering_components.regime_aware_triple_barrier_labeling import (
            apply_regime_aware_triple_barrier_labeling
        )
        
        labeled_data = apply_regime_aware_triple_barrier_labeling(
            data=data,
            optimization_results=optimization_results,
            regime_column="composite_cluster_id"
        )
    else:
        # Fallback to standard labeling
        labeled_data = self._apply_standard_labeling(data)
    
    return labeled_data
```

## Optimization Process

### 1. Parameter Space

The optimizer explores the following parameter space for each regime:

**Triple Barrier Parameters:**
- `profit_take_multiplier`: [0.01, 0.05] (log scale)
- `stop_loss_multiplier`: [0.005, 0.03] (log scale)
- `time_barrier_minutes`: [15, 120]
- `max_lookahead`: [50, 200]

**Regime-Specific Multipliers:**
- `regime_volatility_multiplier`: [0.5, 2.0]
- `regime_trend_multiplier`: [0.5, 2.0]
- `regime_volume_multiplier`: [0.5, 2.0]

**TPSL Parameters:**
- `tp_multiplier`: [1.5, 4.0]
- `sl_multiplier`: [0.8, 2.0]
- `position_size`: [0.05, 0.25]
- `tp_atr_multiplier`: [1.0, 4.0]
- `sl_atr_multiplier`: [0.5, 2.0]
- `trailing_stop`: [0.0, 0.02]
- `break_even_threshold`: [0.005, 0.02]

### 2. Objective Function

The optimizer maximizes a composite score:

```
Score = w1 * normalized_sharpe_ratio + 
        w2 * normalized_win_rate + 
        w3 * normalized_profit_factor + 
        w4 * normalized_regime_accuracy
```

Where each metric is normalized to [0, 1] range.

### 3. Cross-Validation

Uses time-series cross-validation with regime-aware splits to ensure robust optimization.

## Performance Metrics

### Per-Regime Metrics

- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Total Return**: Cumulative return
- **Max Drawdown**: Maximum peak-to-trough decline
- **Sortino Ratio**: Downside risk-adjusted returns
- **Calmar Ratio**: Return to max drawdown ratio

### Regime-Specific Metrics

- **Regime Accuracy**: Accuracy of regime classification
- **Regime Precision**: Precision of regime-specific predictions
- **Regime Recall**: Recall of regime-specific predictions
- **Regime F1**: Harmonic mean of precision and recall

## Output and Visualization

### 1. Optimization Results

The optimizer generates:

- **Parameter importance plots** for each regime
- **Optimization history plots** showing convergence
- **Performance comparison charts** across regimes
- **Comprehensive reports** with detailed metrics

### 2. File Outputs

```
optimization_results/
├── regime_optimization_results.png
├── param_importance_REGIME_0.html
├── param_importance_REGIME_1.html
├── optimization_history_REGIME_0.html
├── optimization_history_REGIME_1.html
└── ...
```

### 3. Database Storage

Optimization studies are stored in SQLite database:
```
regime_triple_barrier_optuna_studies.db
```

## Best Practices

### 1. Data Preparation

- Ensure regime information is available in your data
- Use consistent regime column names (`composite_cluster_id`, `regime`, etc.)
- Validate regime distribution (avoid extremely imbalanced regimes)

### 2. Configuration

- Start with conservative parameter ranges
- Use regime-specific constraints based on market characteristics
- Balance optimization objectives based on your strategy goals

### 3. Optimization

- Use sufficient trials per regime (100+ recommended)
- Monitor convergence and adjust early stopping parameters
- Validate results with out-of-sample testing

### 4. Integration

- Test regime-aware labeling with small datasets first
- Implement fallback to standard labeling when optimization results unavailable
- Monitor regime-specific performance in production

## Troubleshooting

### Common Issues

1. **Insufficient Data per Regime**
   - Increase `min_sample_size` in configuration
   - Consider merging similar regimes
   - Use longer historical data

2. **Poor Optimization Convergence**
   - Increase `n_trials_per_regime`
   - Adjust parameter ranges
   - Check regime data quality

3. **Regime Mapping Issues**
   - Verify regime column exists in data
   - Check regime ID to name mapping
   - Ensure consistent regime naming

4. **Performance Degradation**
   - Validate optimization results with backtesting
   - Check for regime shift in data
   - Re-optimize periodically

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger("RegimeSpecificTripleBarrierOptimizer").setLevel(logging.DEBUG)
logging.getLogger("RegimeAwareTripleBarrierLabeling").setLevel(logging.DEBUG)
```

## Examples

### Complete Example

```python
import asyncio
import pandas as pd
from src.training.steps.step17_final_parameters_optimization.regime_specific_triple_barrier_optimization import (
    optimize_regime_triple_barrier_parameters
)
from src.training.steps.step4_analyst_labeling_feature_engineering_components.regime_aware_triple_barrier_labeling import (
    apply_regime_aware_triple_barrier_labeling
)

async def main():
    # Load data
    data = pd.read_parquet("data_with_regimes.parquet")
    
    # Configure optimization
    config = {
        "regime_specific_optimization": {
            "n_trials_per_regime": 100,
            "timeout_minutes_per_regime": 60,
            "objectives": ["sharpe_ratio", "win_rate", "profit_factor"],
            "objective_weights": {
                "sharpe_ratio": 0.4,
                "win_rate": 0.3,
                "profit_factor": 0.3
            }
        }
    }
    
    # Run optimization
    results = await optimize_regime_triple_barrier_parameters(
        data=data,
        config=config,
        regime_column="composite_cluster_id"
    )
    
    # Apply regime-aware labeling
    labeled_data = apply_regime_aware_triple_barrier_labeling(
        data=data,
        optimization_results=results,
        regime_column="composite_cluster_id"
    )
    
    # Save results
    labeled_data.to_parquet("regime_aware_labeled_data.parquet")
    
    # Print summary
    for regime_name, result in results.items():
        print(f"{regime_name}: Sharpe={result.sharpe_ratio:.4f}, "
              f"Win Rate={result.win_rate:.4f}, "
              f"Profit Factor={result.profit_factor:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Conclusion

The regime-specific triple barrier optimization system provides a powerful framework for adapting trading parameters to different market regimes. By optimizing triple barrier thresholds and TPSL parameters for each HMM regime, you can achieve better performance and more robust trading strategies.

Key benefits:
- **Adaptive Parameters**: Each regime gets optimized parameters
- **Better Performance**: Improved risk-adjusted returns
- **Robust Optimization**: Multi-objective optimization with cross-validation
- **Easy Integration**: Seamless integration with existing pipeline
- **Comprehensive Analysis**: Detailed performance tracking and visualization

For questions or issues, refer to the test script `test_regime_specific_triple_barrier_optimization.py` for working examples.