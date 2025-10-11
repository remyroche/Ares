# VectorBT Integration Guide

This guide explains how to use the VectorBT-enhanced components in the `src/utils/ml_common/` directory.

## Overview

The VectorBT integration provides three main enhancements:

1. **VectorBT Backtesting Engine** - High-performance portfolio backtesting
2. **VectorBT Financial Metrics** - Comprehensive financial performance analysis
3. **VectorBT Portfolio Optimization** - Advanced portfolio optimization strategies

## Installation

### Prerequisites

```bash
# Install VectorBT
pip install vectorbt

# Optional: Install GPU acceleration
pip install cupy  # For CUDA support

# Install additional dependencies
pip install scipy scikit-learn
```

### Verify Installation

```python
import vectorbt as vbt
print(f"VectorBT version: {vbt.__version__}")
```

## Quick Start

### 1. VectorBT Backtesting

```python
from src.utils.ml_common.vectorbt_backtesting_engine import (
    run_vectorbt_backtest, create_vectorbt_config, BacktestMode
)
import numpy as np
import pandas as pd

# Generate sample data
n_periods = 1000
n_assets = 3
returns = np.random.normal(0.001, 0.02, (n_periods, n_assets))
prices = 100 * (1 + returns).cumprod(axis=0)
signals = np.random.choice([-1, 0, 1], size=(n_periods, n_assets))
timestamps = pd.date_range(start='2020-01-01', periods=n_periods, freq='1min')

# Create configuration
config = create_vectorbt_config(
    initial_capital=100000.0,
    commission_rate=0.001,
    slippage_rate=0.0005,
    use_gpu=True
)

# Run backtest
result = run_vectorbt_backtest(
    signals, prices, timestamps,
    config=config,
    mode=BacktestMode.VECTORBT_CPU
)

# Access results
print(f"Final portfolio value: ${result.portfolio_values[-1]:,.2f}")
print(f"Total return: {result.performance_metrics['total_return']:.2%}")
print(f"Sharpe ratio: {result.performance_metrics['sharpe_ratio']:.3f}")
print(f"Max drawdown: {result.performance_metrics['max_drawdown']:.2%}")
```

### 2. VectorBT Financial Metrics

```python
from src.utils.ml_common.vectorbt_financial_metrics import (
    calculate_financial_metrics, create_metrics_config
)

# Generate sample data
portfolio_values = 100000 * (1 + np.random.normal(0.001, 0.02, 1000)).cumprod()
returns = np.random.normal(0.001, 0.02, 1000)
benchmark_values = 100000 * (1 + np.random.normal(0.0008, 0.015, 1000)).cumprod()

# Create configuration
config = create_metrics_config(
    risk_free_rate=0.02,
    annualization_factor=252,
    enable_regime_analysis=True
)

# Calculate metrics
metrics = calculate_financial_metrics(
    portfolio_values=portfolio_values,
    returns=returns,
    benchmark_values=benchmark_values,
    config=config
)

# Access key metrics
print(f"Total return: {metrics['total_return']:.2%}")
print(f"Volatility: {metrics['volatility']:.2%}")
print(f"Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Max drawdown: {metrics['max_drawdown']:.2%}")
print(f"Alpha: {metrics.get('alpha', 0):.4f}")
print(f"Beta: {metrics.get('beta', 0):.3f}")
```

### 3. VectorBT Portfolio Optimization

```python
from src.utils.ml_common.vectorbt_portfolio_optimization import (
    optimize_portfolio, create_optimization_config, OptimizationMethod
)

# Generate sample data
n_periods = 1000
n_assets = 5
returns = np.random.normal(0.001, 0.02, (n_periods, n_assets))
asset_names = [f'Asset_{i+1}' for i in range(n_assets)]

# Create configuration
config = create_optimization_config(
    method=OptimizationMethod.MEAN_VARIANCE,
    risk_aversion=1.0,
    rebalancing_frequency='monthly'
)

# Optimize portfolio
result = optimize_portfolio(
    returns=returns,
    method=OptimizationMethod.MEAN_VARIANCE,
    asset_names=asset_names,
    config=config
)

# Access results
print(f"Expected return: {result.expected_return:.2%}")
print(f"Expected volatility: {result.expected_volatility:.2%}")
print(f"Sharpe ratio: {result.sharpe_ratio:.3f}")
print(f"Portfolio weights: {result.weights}")
```

## Advanced Usage

### Using the Unified Vectorization Manager

The unified vectorization manager automatically selects the optimal strategy for your operations:

```python
from src.utils.ml_common.unified_vectorization_manager import (
    optimize_vectorbt_backtesting, optimize_vectorbt_metrics, optimize_vectorbt_portfolio
)

# VectorBT backtesting through unified manager
backtest_result = optimize_vectorbt_backtesting(
    signals=signals,
    prices=prices,
    timestamps=timestamps
)

# VectorBT metrics through unified manager
metrics_result = optimize_vectorbt_metrics(
    portfolio_values=portfolio_values,
    returns=returns,
    benchmark_values=benchmark_values
)

# VectorBT portfolio optimization through unified manager
portfolio_result = optimize_vectorbt_portfolio(
    returns=returns,
    asset_names=asset_names
)
```

### Custom Configurations

#### Backtesting Configuration

```python
from src.utils.ml_common.vectorbt_backtesting_engine import VectorBTBacktestConfig

config = VectorBTBacktestConfig(
    initial_capital=500000.0,
    commission_rate=0.0005,
    slippage_rate=0.0002,
    max_position_size=0.2,
    min_position_size=0.01,
    rebalance_frequency='daily',
    risk_free_rate=0.03,
    use_gpu=True,
    enable_parallel=True,
    memory_limit_gb=16.0
)
```

#### Financial Metrics Configuration

```python
from src.utils.ml_common.vectorbt_financial_metrics import FinancialMetricsConfig

config = FinancialMetricsConfig(
    risk_free_rate=0.025,
    annualization_factor=252,
    lookback_periods=252,
    confidence_level=0.05,
    downside_threshold=0.0,
    enable_regime_analysis=True,
    regime_threshold=0.1
)
```

#### Portfolio Optimization Configuration

```python
from src.utils.ml_common.vectorbt_portfolio_optimization import (
    OptimizationConfig, OptimizationConstraints, OptimizationMethod
)

constraints = OptimizationConstraints(
    min_weight=0.0,
    max_weight=0.3,
    max_single_asset_weight=0.2,
    max_sector_weight=0.4,
    max_turnover=0.5,
    max_portfolio_volatility=0.25
)

config = OptimizationConfig(
    method=OptimizationMethod.MEAN_VARIANCE,
    risk_aversion=1.5,
    rebalancing_frequency='monthly',
    lookback_period=252,
    constraints=constraints,
    enable_regime_aware=True
)
```

## Performance Optimization

### GPU Acceleration

VectorBT supports GPU acceleration for large datasets:

```python
# Enable GPU acceleration
config = create_vectorbt_config(use_gpu=True)

# Use GPU mode for backtesting
result = run_vectorbt_backtest(
    signals, prices, timestamps,
    config=config,
    mode=BacktestMode.VECTORBT_GPU
)
```

### Parallel Processing

For CPU-bound operations, enable parallel processing:

```python
# Enable parallel processing
config = create_vectorbt_config(enable_parallel=True)

# Use parallel mode
result = run_vectorbt_backtest(
    signals, prices, timestamps,
    config=config,
    mode=BacktestMode.VECTORBT_PARALLEL
)
```

### Memory Optimization

For large datasets, use memory optimization:

```python
config = create_vectorbt_config(
    memory_limit_gb=8.0,
    chunk_size=10000,
    enable_memory_optimization=True
)
```

## Available Metrics

### Return Metrics
- `total_return` - Total return over the period
- `annualized_return` - Annualized return
- `cumulative_return` - Cumulative return
- `avg_rolling_return` - Average rolling return

### Risk Metrics
- `volatility` - Annualized volatility
- `downside_deviation` - Downside deviation
- `var_95` - 95% Value at Risk
- `cvar_95` - 95% Conditional Value at Risk
- `skewness` - Return skewness
- `kurtosis` - Return kurtosis

### Risk-Adjusted Metrics
- `sharpe_ratio` - Sharpe ratio
- `sortino_ratio` - Sortino ratio
- `calmar_ratio` - Calmar ratio
- `information_ratio` - Information ratio
- `treynor_ratio` - Treynor ratio

### Drawdown Metrics
- `max_drawdown` - Maximum drawdown
- `avg_drawdown` - Average drawdown
- `max_drawdown_duration` - Maximum drawdown duration
- `recovery_time` - Recovery time from max drawdown

### Trading Metrics
- `win_rate` - Win rate
- `profit_factor` - Profit factor
- `expectancy` - Expected value per trade
- `best_trade` - Best trade return
- `worst_trade` - Worst trade return

### Benchmark Metrics
- `alpha` - Jensen's alpha
- `beta` - Beta coefficient
- `tracking_error` - Tracking error
- `relative_performance` - Relative performance vs benchmark

## Available Optimization Methods

### Portfolio Optimization Methods
- `MEAN_VARIANCE` - Mean-variance optimization
- `RISK_PARITY` - Risk parity optimization
- `EQUAL_WEIGHT` - Equal weight portfolio
- `MIN_VARIANCE` - Minimum variance optimization
- `MAX_SHARPE` - Maximum Sharpe ratio optimization
- `BLACK_LITTERMAN` - Black-Litterman model
- `HIERARCHICAL_RISK_PARITY` - Hierarchical risk parity
- `MAX_DIVERSIFICATION` - Maximum diversification

### Rebalancing Frequencies
- `DAILY` - Daily rebalancing
- `WEEKLY` - Weekly rebalancing
- `MONTHLY` - Monthly rebalancing
- `QUARTERLY` - Quarterly rebalancing
- `ANNUALLY` - Annual rebalancing
- `ADAPTIVE` - Adaptive rebalancing

## Error Handling

All VectorBT components include comprehensive error handling:

```python
try:
    result = run_vectorbt_backtest(signals, prices, timestamps)
except ImportError as e:
    print(f"VectorBT not available: {e}")
except ValueError as e:
    print(f"Invalid input data: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Testing

Run the test suite to verify everything is working:

```bash
# Run all tests
python -m pytest src/utils/ml_common/test_vectorbt_integration.py -v

# Run specific test class
python -m pytest src/utils/ml_common/test_vectorbt_integration.py::TestVectorBTBacktesting -v
```

## Examples

See `vectorbt_integration_example.py` for comprehensive examples:

```bash
python src/utils/ml_common/vectorbt_integration_example.py
```

## Troubleshooting

### Common Issues

1. **VectorBT not found**: Install with `pip install vectorbt`
2. **GPU not available**: Install CuPy with `pip install cupy`
3. **Memory errors**: Reduce dataset size or enable memory optimization
4. **Slow performance**: Enable GPU acceleration or parallel processing

### Performance Tips

1. Use GPU acceleration for large datasets (>10,000 periods)
2. Enable parallel processing for CPU-bound operations
3. Use memory optimization for very large datasets
4. Consider chunking data for extremely large datasets

### Getting Help

- Check the VectorBT documentation: https://vectorbt.dev/
- Review the test files for usage examples
- Check the unified vectorization manager for automatic optimization

## Migration from Existing Code

### From Custom Backtesting

```python
# Old way
from src.utils.ml_common.vectorized_backtesting import run_vectorized_backtest
result = run_vectorized_backtest(signals, prices)

# New way
from src.utils.ml_common.vectorbt_backtesting_engine import run_vectorbt_backtest
result = run_vectorbt_backtest(signals, prices, timestamps)
```

### From Custom Metrics

```python
# Old way
from src.utils.ml_common.evaluation.unified_evaluator import compute_sharpe_ratio
sharpe = compute_sharpe_ratio(returns)

# New way
from src.utils.ml_common.vectorbt_financial_metrics import calculate_financial_metrics
metrics = calculate_financial_metrics(portfolio_values, returns)
sharpe = metrics['sharpe_ratio']
```

## Conclusion

The VectorBT integration provides significant performance improvements and enhanced functionality for financial analysis. The unified vectorization manager automatically selects the optimal strategy, making it easy to get the best performance without manual configuration.

For more advanced usage, refer to the individual module documentation and the comprehensive test suite.