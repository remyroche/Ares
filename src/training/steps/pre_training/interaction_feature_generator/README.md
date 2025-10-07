# Data-Driven Lookback Optimization System

A comprehensive three-stage Bayesian optimization system for selecting optimal lookback periods for feature families, replacing hardcoded ceilings with data-driven inference while maintaining production constraints.

## Overview

This system implements a rigorous, production-safe approach to lookback optimization that:

- **Replaces hardcoded lookback ceilings** with data-driven inference using Bayesian shrinkage
- **Maintains production constraints** (≤120 pre-selection features, ≤15 interactions, ≤50ms p99 latency)
- **Provides uncertainty quantification** with HAC standard errors and hierarchical shrinkage
- **Supports both discrete and blended approaches** for robust feature generation
- **Implements hysteresis and simplicity priors** for stable production deployment

## Architecture

The system consists of three main stages:

### Stage 1: IC Surface Estimation
- Estimates smooth information coefficient surfaces for each feature family
- Uses HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors
- Fits penalized splines to capture smoothness and reduce multiple testing issues
- Incorporates cost-aware optimization with CPU, staleness, and uncertainty penalties

### Stage 2: Walk-Forward Stability Testing
- Tests stability of optimal lookback choices across time using purged walk-forward validation
- Prevents data leakage with proper purging and embargo periods
- Evaluates fold-wise match rates and IC penalties
- Makes recommendations for discrete vs blended approaches

### Stage 3: Hierarchical Bayesian Shrinkage
- Applies hierarchical shrinkage across feature families and symbols
- Uses variational inference (ADVI) or NUTS sampling for efficiency
- Stabilizes estimates by borrowing strength across similar families
- Provides uncertainty quantification with credible intervals

## Key Features

### Cost-Aware Optimization
The system replaces simple lookback ceilings with a sophisticated cost function:

```
Score(ℓ) = IC_oos(ℓ) - λ_cost × CPU_cost(ℓ) - λ_stale × Staleness(ℓ) - λ_unc × SE_HAC(ℓ)
```

Where:
- `λ_cost`: Penalty for CPU cost (latency impact)
- `λ_stale`: Penalty for staleness (update lag)
- `λ_unc`: Penalty for estimation risk (HAC standard error)

### Feature Family Support
- **Momentum**: Price momentum with configurable lookback periods
- **Volatility**: EW volatility with halflife optimization
- **Garman-Klass**: High-frequency volatility estimation
- **VWAP Roll**: Volume-weighted average price rolling features
- **RSI**: Relative Strength Index with period optimization
- **Autocorrelation**: Time series autocorrelation features

### Production Constraints
- **Feature Budget**: ≤120 pre-selection features
- **Interaction Limit**: ≤15 total interactions
- **Latency Constraint**: ≤50ms p99 latency
- **Lookback Ceiling**: ≤120 minutes (configurable)

## Quick Start

### Basic Usage

```python
from src.training.steps.pre_training.interaction_feature_generator import (
    LookbackOptimizationOrchestrator,
    create_production_config
)

# Create configuration
config = create_production_config()

# Initialize orchestrator
orchestrator = LookbackOptimizationOrchestrator(config)

# Prepare data
data = {
    "SYMBOL_1": your_market_data_1,  # pd.DataFrame with OHLCV columns
    "SYMBOL_2": your_market_data_2
}

targets = {
    "SYMBOL_1": your_target_1,  # np.ndarray of future returns
    "SYMBOL_2": your_target_2
}

feature_names = {
    FamilyType.MOMENTUM: "momentum_feature",
    FamilyType.VOLATILITY: "volatility_feature",
    # ... other families
}

# Run optimization
result = orchestrator.optimize_lookbacks(data, targets, feature_names)

# Check results
if result.success:
    print(f"Optimization completed in {result.execution_time:.3f}s")
    
    # Access decisions
    for symbol, symbol_decisions in result.decisions.items():
        for family, decision in symbol_decisions.items():
            print(f"{symbol}-{family.value}: {decision.lookback_spec.decision_type.value}")
            print(f"  Lookback: {decision.lookback_spec.effective_lookback}")
            print(f"  Confidence: {decision.lookback_spec.confidence_score:.3f}")
else:
    print(f"Optimization failed: {result.error_message}")
```

### Advanced Usage

```python
# Use individual stages for fine-grained control
from .ic_surface import ICSurfaceEstimator
from .wf_stability import StabilityTester
from .hierarchical import HierarchicalBayesianShrinkage

# Stage 1: IC Surface Estimation
ic_estimator = ICSurfaceEstimator(config)
ic_result = ic_estimator.estimate_surface(data, target, FamilyType.MOMENTUM, "momentum")

# Stage 2: Stability Testing
stability_tester = StabilityTester(config)
stability_result = stability_tester.test_stability(data, target, ic_result, "momentum")

# Stage 3: Hierarchical Shrinkage
hierarchical_shrinkage = HierarchicalBayesianShrinkage(config)
hierarchical_result = hierarchical_shrinkage.apply_shrinkage(symbol_family_data)
```

## Configuration

### Development Configuration
```python
from .config import create_development_config

config = create_development_config()
# - Relaxed search grids
# - Fewer CV folds (3)
# - Reduced hierarchical samples (500)
# - Faster execution for testing
```

### Production Configuration
```python
from .config import create_production_config

config = create_production_config()
# - Strict cost penalties
# - Conservative hysteresis
# - More CV folds (7)
# - More hierarchical samples (2000)
```

### Custom Configuration
```python
from .config import LookbackOptimizationConfig, CostPenalties, SearchGrids

config = LookbackOptimizationConfig(
    penalties=CostPenalties(
        lambda_cost=0.1,      # Higher CPU penalty
        lambda_stale=0.05,    # Staleness penalty
        lambda_uncertainty=0.15  # Uncertainty penalty
    ),
    search_grids=SearchGrids(
        momentum_bars=[5, 12, 24, 48],  # Custom momentum grid
        sigma_halflife=[6, 12, 18, 36]  # Custom volatility grid
    )
)
```

## Feature Family Details

### Momentum Features
- **Purpose**: Capture price momentum and trend following
- **Calculation**: `pct_change(lookback)`
- **Search Grid**: [5, 12, 24, 48, 96, 192] bars
- **Cost Model**: Linear scaling with lookback

### Volatility Features
- **Purpose**: Measure market volatility using EW methods
- **Calculation**: `sqrt(ewm_var(returns, alpha=2/(lookback+1)))`
- **Search Grid**: [6, 12, 18, 36, 72, 144] halflife periods
- **Cost Model**: Logarithmic scaling (EW is efficient)

### Garman-Klass Volatility
- **Purpose**: High-frequency volatility using OHLC data
- **Calculation**: `sqrt(mean(0.5*log(high/low)^2 - (2*log(2)-1)*log(close/open)^2))`
- **Search Grid**: [6, 12, 24, 48, 96] bars
- **Cost Model**: Linear scaling

### VWAP Rolling Features
- **Purpose**: Volume-weighted price analysis
- **Calculation**: `close / vwap(typical_price, volume, lookback)`
- **Search Grid**: [12, 36] bars
- **Cost Model**: Linear scaling (expensive due to volume calculations)

### RSI Features
- **Purpose**: Relative strength momentum indicator
- **Calculation**: `100 - 100/(1 + avg_gain/avg_loss)`
- **Search Grid**: [7, 14, 28, 56] periods
- **Cost Model**: Linear scaling

### Autocorrelation Features
- **Purpose**: Time series dependency analysis
- **Calculation**: `autocorr(returns, lag=1, window=lookback)`
- **Search Grid**: [6, 12, 24, 48] bars
- **Cost Model**: Linear scaling (expensive due to correlation calculations)

## Decision Logic

### Discrete vs Blended Selection

The system automatically chooses between discrete and blended approaches based on:

1. **Stability Metrics**:
   - Fold match rate ≥ 60-70%
   - Average IC penalty ≤ 0.1-0.2σ
   - HDI width ≤ 4 bars

2. **Hysteresis Rules**:
   - Minimum change threshold: 22% in log lookback
   - Minimum IC improvement: 0.25σ
   - Prevents excessive switching

3. **Cost Considerations**:
   - CPU cost penalties
   - Staleness penalties
   - Uncertainty penalties

### Blended Features

When blending is recommended, the system:
- Selects 2-3 nearby lookback windows
- Optimizes blend weights using linear regression
- Maintains same computational footprint as discrete choice
- Exports single blended feature (not components)

## Performance Characteristics

### Runtime Profile
- **Stage 1 (IC Surface)**: 10-20 min per symbol
- **Stage 2 (Stability)**: 5-10 min per symbol  
- **Stage 3 (Hierarchical)**: 2-5 min total
- **End-to-end**: <1 hour per symbol on modest hardware

### Memory Usage
- **Peak Memory**: ~8GB for typical datasets
- **Feature Generation**: ~1-2GB per symbol
- **Hierarchical Inference**: ~2-4GB total

### Scalability
- **Symbols**: Linear scaling with parallel processing
- **Families**: Constant time per family
- **Data Size**: Sub-linear due to chunking

## Validation and Testing

### Statistical Validation
- **Purged Walk-Forward**: Prevents data leakage
- **HAC Standard Errors**: Accounts for autocorrelation
- **Block Bootstrap**: Robust uncertainty quantification
- **Convergence Diagnostics**: R-hat, effective sample size

### Production Validation
- **Feature Budget**: Enforces ≤120 pre-selection limit
- **Latency Testing**: Validates ≤50ms p99 constraint
- **Memory Limits**: Prevents OOM errors
- **CI Gates**: Automated quality checks

### Test Suite
```bash
# Run comprehensive tests
python -m src.training.steps.pre_training.interaction_feature_generator.test_optimization_system

# Run specific test categories
python -m unittest TestICSurfaceEstimation
python -m unittest TestStabilityTesting
python -m unittest TestDecisionMaking
```

## Monitoring and Debugging

### Logging
```python
import logging
logging.basicConfig(level=logging.INFO)

# System provides detailed logging at each stage
# - IC surface estimation progress
# - Stability testing results
# - Hierarchical shrinkage diagnostics
# - Decision making rationale
```

### Metrics and Reports
```python
# Generate comprehensive report
report = orchestrator.generate_comprehensive_report(result)

# Key metrics:
# - Execution time breakdown
# - Decision type distribution
# - Feature quality scores
# - Memory usage statistics
# - Recommendations
```

### Debugging Tools
```python
# Access intermediate results
ic_results = result.ic_surface_results
stability_results = result.stability_results
hierarchical_results = result.hierarchical_results
decisions = result.decisions

# Inspect individual components
for symbol, symbol_ic in ic_results.items():
    for family, ic_result in symbol_ic.items():
        print(f"{symbol}-{family.value}:")
        print(f"  Optimal lookback: {ic_result.optimal_lookback}")
        print(f"  Optimal IC: {ic_result.optimal_ic}")
        print(f"  R-squared: {ic_result.r_squared}")
```

## Integration with Existing Pipeline

### Feature Generation Integration
```python
# Replace hardcoded lookbacks in existing features
from .feature_families import MultiFamilyFeatureGenerator

# Use optimized lookbacks
feature_generator = MultiFamilyFeatureGenerator(config)
feature_results = feature_generator.generate_features(
    data, optimized_decisions, feature_names
)

# Create feature matrix
feature_matrix, feature_names = feature_generator.create_feature_matrix(feature_results)
```

### Interaction Feature Integration
```python
# Use optimized features in interactions
from .pid_based_feature_generation.interaction_feature_generator import InteractionFeatureGenerator

# Pass optimized lookbacks to interaction generator
interaction_generator = InteractionFeatureGenerator()
interaction_results = await interaction_generator.generate_interaction_features(
    feature_matrix, feature_names, optimized_lookback_periods, target
)
```

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce `memory_limit_gb` in config
   - Use smaller search grids
   - Process symbols sequentially

2. **Convergence Issues**
   - Increase hierarchical samples
   - Check data quality
   - Verify sufficient data length

3. **Low Quality Features**
   - Review data preprocessing
   - Check for data leakage
   - Adjust quality thresholds

4. **Slow Performance**
   - Enable parallel processing
   - Reduce CV folds
   - Use development config

### Performance Optimization

```python
# Optimize for speed
config = create_development_config()
config.cv.n_folds = 3
config.hierarchical.n_samples = 500
config.enable_parallel = True

# Optimize for quality
config = create_production_config()
config.cv.n_folds = 7
config.hierarchical.n_samples = 2000
config.penalties.lambda_uncertainty = 0.2
```

## Future Enhancements

### Planned Features
- **GPU Acceleration**: CUDA support for large-scale optimization
- **Online Learning**: Incremental updates without full retraining
- **Multi-Asset Optimization**: Cross-asset lookback sharing
- **Regime-Aware Optimization**: Different lookbacks for different market regimes

### Research Directions
- **Deep Learning Integration**: Neural network-based IC surface estimation
- **Causal Inference**: Causal lookback selection methods
- **Multi-Objective Optimization**: Pareto-optimal lookback selection
- **Federated Learning**: Distributed optimization across multiple systems

## Contributing

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest src/training/steps/pre_training/interaction_feature_generator/

# Run linting
flake8 src/training/steps/pre_training/interaction_feature_generator/

# Run type checking
mypy src/training/steps/pre_training/interaction_feature_generator/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all functions
- Document all public APIs
- Write comprehensive tests

## License

This project is part of the Ares Trading System and is proprietary software.

## Support

For questions, issues, or contributions, please contact the Ares development team.

---

*This system represents a significant advancement in feature engineering for quantitative trading, providing a rigorous, data-driven approach to lookback optimization that maintains production constraints while maximizing predictive power.*