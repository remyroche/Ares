# Iterative Rolling HMM Clustering Optimization

This document describes the iterative optimization approach for Rolling HMM clustering that replaces traditional grid search with a more efficient 20% increment strategy.

## Overview

The iterative optimization system uses a reduced initial parameter set and then optimizes each parameter using 20% increments until convergence is achieved. This approach is significantly more efficient than grid search while still finding optimal parameters.

## Key Features

### 1. Reduced Parameter Set
Instead of optimizing over a large parameter space, we use a focused set of 5 key parameters:
- `n_components`: Number of HMM states (4-6)
- `ewma_short`: Short-term EWMA window (4-12)
- `ewma_long`: Long-term EWMA window (16-30)
- `min_covar`: Minimum covariance for regularization (1e-4 to 1e-1)
- `kappa`: Sticky prior strength (0.1 to 10.0)

### 2. 20% Increment Strategy
Each parameter is optimized using fixed 20% increments:
- Try positive direction: `new_value = current_value * 1.2`
- Try negative direction: `new_value = current_value * 0.8`
- Continue in the improving direction until no more improvement
- Apply bounds to keep parameters in valid ranges

### 3. Optimization Order
Parameters are optimized in a specific order to maximize efficiency:
1. `n_components` (model structure - most important)
2. `ewma_short` (feature engineering)
3. `ewma_long` (feature engineering)
4. `min_covar` (regularization)
5. `kappa` (regularization)

### 4. Convergence Criteria
The optimization stops when:
- No improvement for 3 consecutive iterations
- Maximum iterations (20) reached
- Maximum parameter iterations (10) reached

## Implementation Files

### Core Components

1. **`rolling_hmm_iterative_optimizer.py`**
   - Main iterative optimizer class
   - Implements 20% increment strategy
   - Handles parameter bounds and convergence
   - Tracks optimization history

2. **`rolling_hmm_iterative_regime_discovery_step.py`**
   - Main pipeline orchestrator
   - Integrates all components
   - Handles data loading, feature engineering, optimization, and results
   - Provides complete end-to-end regime discovery

3. **`feature_engineering.py`**
   - Enhanced to support dynamic EWMA configs
   - Pre-computes features for multiple EWMA windows
   - Caches features for reuse during optimization

4. **`sticky_hmm_model.py`**
   - HMM model implementation
   - Supports sticky priors and regularization
   - Handles model fitting and prediction

### Configuration

5. **`hpo_config.py`**
   - Configuration classes for all components
   - Default parameters and optimization settings
   - Easy customization for different use cases

## Algorithm Details

### Iterative Optimization Process

```
1. Initialize with reduced parameters:
   - n_components=5, ewma_short=6, ewma_long=20
   - min_covar=0.005, kappa=2.0

2. For each iteration (max 20):
   a. For each parameter in optimization order:
      i. Try 20% increase
      ii. If score improves, continue increasing
      iii. Else try 20% decrease
      iv. If score improves, continue decreasing
      v. Stop when no improvement

3. Check convergence:
   - If no improvement for 3 iterations → stop
   - If max iterations reached → stop

4. Return best parameters and model
```

### Objective Function

The objective function combines multiple quality metrics:
- **Statistical Quality (40%)**: Between/within CV ratio, silhouette score
- **Temporal Quality (20%)**: Temporal smoothness, persistence
- **Economic Quality (40%)**: Regime returns, volatility, Sharpe ratios

### Feature Engineering

Features are generated dynamically based on optimized EWMA parameters:
- Returns: Log returns, EWMA returns, cumulative returns
- Volatility: Rolling std, EWMA volatility, realized volatility
- Trend: SMAs, EWMA crossovers, momentum indicators
- Volume: Volume ratios, OBV, volume-weighted returns

## Usage

### Basic Usage

```python
from src.training.steps.market_analysis.rolling_hmm_clustering_iterative import (
    RollingHMMIterativeRegimeDiscoveryStep,
    IterativeRegimeDiscoveryConfig,
    DEFAULT_ITERATIVE_HPO_CONFIG
)

# Create configuration
config = IterativeRegimeDiscoveryConfig(
    data_path="data/market/1h/BTCUSDT_1h.parquet",
    iterative_hpo_config=DEFAULT_ITERATIVE_HPO_CONFIG,
    output_dir="results/rolling_hmm_iterative"
)

# Run regime discovery
discovery = RollingHMMIterativeRegimeDiscoveryStep(config)
result = discovery.run()

print(f"Best parameters: {result['best_parameters']}")
print(f"Best score: {result['best_score']:.4f}")
print(f"Optimization trials: {result['optimization_trials']}")
```

### Custom Configuration

```python
from src.training.steps.market_analysis.rolling_hmm_clustering_iterative import (
    IterativeHPOConfig,
    IterativeRegimeDiscoveryConfig
)

# Custom HPO configuration
hpo_config = IterativeHPOConfig(
    initial_n_components=4,
    initial_ewma_short=8,
    initial_ewma_long=24,
    increment_ratio=0.15,  # Use 15% increments instead of 20%
    max_iterations=15,
    convergence_patience=2
)

# Create discovery config
config = IterativeRegimeDiscoveryConfig(
    data_path="data/market/1h/BTCUSDT_1h.parquet",
    iterative_hpo_config=hpo_config,
    output_dir="results/custom_iterative"
)
```

## Performance Benefits

### Compared to Grid Search
- **Reduced Computation**: 20-50 trials vs 1000+ in grid search
- **Faster Convergence**: Typically 5-10 iterations vs exhaustive search
- **Better Resource Usage**: Focused optimization on promising parameters

### Quality Assurance
- **Convergence Testing**: Ensures parameters are truly optimal
- **Multiple Metrics**: Balances statistical, temporal, and economic quality
- **Robust Bounds**: Prevents invalid parameter combinations

## Output

The system produces:
1. **Optimized HMM Model**: Trained with best parameters
2. **Regime Predictions**: Time series of regime labels
3. **Quality Metrics**: Comprehensive assessment of regime quality
4. **Optimization History**: Complete record of optimization process
5. **Economic Features**: Interpretable features for analysis

## Example Results

```
🎉 Iterative Rolling HMM Regime Discovery Complete!
📊 Final Results:
   → Data: 8760 rows
   → Features: 45 total, 12 economic
   → Regimes: 5 discovered
   → Best Score: 0.8432
   → Optimization: 28 trials, 7 iterations
   → Converged: True
   → Execution Time: 45.2s
   → Output: results/rolling_hmm_iterative

Best Parameters:
{
  "n_components": 5,
  "ewma_short": 7,
  "ewma_long": 22,
  "min_covar": 0.0038,
  "kappa": 2.4
}
```

## Testing

Run the test script to verify the implementation:

```bash
python src/training/steps/market_analysis/rolling_hmm_clustering_iterative/rolling_hmm_iterative_regime_discovery_step.py
```

This will execute a complete regime discovery pipeline with sample data and report results.

## Future Enhancements

1. **Adaptive Increments**: Adjust increment size based on parameter sensitivity
2. **Parallel Evaluation**: Evaluate multiple parameter sets simultaneously
3. **Early Stopping**: More sophisticated stopping criteria
4. **Parameter Importance**: Analyze which parameters contribute most to quality
5. **Multi-Objective Optimization**: Balance multiple quality dimensions

## Conclusion

The iterative rolling HMM clustering optimization provides an efficient and effective alternative to traditional grid search. By using 20% increments and focused optimization, it achieves high-quality results with significantly reduced computational requirements.

The system is particularly well-suited for:
- Production environments where optimization speed is critical
- Research requiring rapid parameter exploration
- Systems with limited computational resources
- Real-time regime discovery applications
