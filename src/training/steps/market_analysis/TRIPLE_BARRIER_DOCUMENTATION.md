# MARKET_ANALYSIS Triple Barrier Labeling Documentation

## Overview

The MARKET_ANALYSIS Triple Barrier Labeling system provides a comprehensive implementation of the triple barrier method for financial time series labeling. This system integrates regime-aware optimization, performance validation, and seamless integration with the existing market analysis pipeline.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Components](#core-components)
3. [Configuration](#configuration)
4. [Usage Examples](#usage-examples)
5. [Advanced Features](#advanced-features)
6. [Performance Optimization](#performance-optimization)
7. [Validation and Testing](#validation-and-testing)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)

## Quick Start

### Basic Usage

```python
from src.training.steps.market_analysis.triple_barrier_labeling import apply_triple_barrier_labeling
import pandas as pd

# Load your market data
data = pd.read_parquet('market_data.parquet')

# Apply triple barrier labeling
labeled_data = apply_triple_barrier_labeling(data)

print(f"Generated {len(labeled_data)} labeled samples")
print(f"Label distribution: {labeled_data['label'].value_counts()}")
```

### Regime-Aware Labeling

```python
from src.training.steps.market_analysis.enhanced_market_analysis_with_triple_barrier import quick_triple_barrier_analysis

# Apply regime-aware triple barrier labeling
labeled_data = quick_triple_barrier_analysis(
    data, 
    regime_aware=True,
    regime_column='hmm_regime'
)
```

## Core Components

### 1. MarketAnalysisTripleBarrierLabeling

The main triple barrier labeling class that provides:

- **Standard Triple Barrier Labeling**: Basic implementation with configurable parameters
- **Regime-Aware Labeling**: Integration with HMM regime detection
- **Performance Optimization**: Numba acceleration and vectorization
- **Transaction Cost Modeling**: Realistic profit/loss calculations
- **Binary Classification**: Automatic filtering of HOLD samples

### 2. RegimeAwareTripleBarrierOptimizer

Advanced optimizer that provides:

- **Regime-Specific Parameter Optimization**: Optimizes parameters for each market regime
- **Performance-Based Adjustment**: Uses Sharpe ratio, win rate, and profit factor
- **Comprehensive Regime Analysis**: Detailed performance metrics per regime
- **Parameter Persistence**: Save and load optimization results

### 3. TripleBarrierValidator

Comprehensive validation framework that provides:

- **Data Quality Validation**: OHLC consistency, missing values, price anomalies
- **Labeling Quality Validation**: Class balance, label consistency
- **Performance Validation**: Win rate, Sharpe ratio, maximum drawdown
- **Temporal Validation**: Lookahead bias detection
- **Statistical Validation**: Normality tests, autocorrelation analysis

### 4. EnhancedMarketAnalysisWithTripleBarrier

Integrated pipeline that provides:

- **Seamless Workflow Integration**: Complete end-to-end analysis
- **Automated Optimization**: Regime parameter optimization
- **Comprehensive Validation**: Multi-level validation framework
- **Results Persistence**: Save intermediate and final results
- **Performance Monitoring**: Execution time and resource usage tracking

## Configuration

### TripleBarrierConfig

```python
from src.training.steps.market_analysis.triple_barrier_labeling import TripleBarrierConfig

config = TripleBarrierConfig(
    profit_take_multiplier=0.002,    # 0.2% profit take
    stop_loss_multiplier=0.001,      # 0.1% stop loss
    time_barrier_minutes=30,         # 30-minute time barrier
    max_lookahead=100,               # Maximum 100 points lookahead
    transaction_cost=0.0008,         # 0.08% transaction cost
    binary_classification=True,      # Filter out HOLD samples
    regime_aware=True,               # Enable regime-aware labeling
    regime_column='hmm_regime',      # Regime column name
    enable_validation=True,          # Enable validation
    enable_profiling=True            # Enable performance profiling
)
```

### MarketAnalysisTripleBarrierConfig

```python
from src.training.steps.market_analysis.enhanced_market_analysis_with_triple_barrier import MarketAnalysisTripleBarrierConfig

config = MarketAnalysisTripleBarrierConfig(
    # Triple barrier parameters
    profit_take_multiplier=0.002,
    stop_loss_multiplier=0.001,
    time_barrier_minutes=30,
    max_lookahead=100,
    transaction_cost=0.0008,
    binary_classification=True,
    
    # Regime awareness
    regime_aware=True,
    regime_column='hmm_regime',
    optimize_regime_parameters=True,
    
    # Validation
    enable_validation=True,
    validation_threshold=0.7,
    
    # Performance optimization
    enable_numba_acceleration=True,
    enable_vectorization=True,
    
    # Output settings
    save_intermediate_results=True,
    save_optimization_results=True,
    output_directory='data_cache/triple_barrier_results'
)
```

## Usage Examples

### Example 1: Basic Triple Barrier Labeling

```python
import pandas as pd
import numpy as np
from src.training.steps.market_analysis.triple_barrier_labeling import create_triple_barrier_labeler

# Create sample market data
dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
data = pd.DataFrame({
    'open': np.random.uniform(100, 110, 1000),
    'high': np.random.uniform(105, 115, 1000),
    'low': np.random.uniform(95, 105, 1000),
    'close': np.random.uniform(100, 110, 1000),
    'volume': np.random.uniform(1000, 10000, 1000)
}, index=dates)

# Create labeler with custom parameters
labeler = create_triple_barrier_labeler(
    profit_take_multiplier=0.003,  # 0.3% profit take
    stop_loss_multiplier=0.002,    # 0.2% stop loss
    time_barrier_minutes=45,       # 45-minute time barrier
    binary_classification=True
)

# Apply labeling
labeled_data = labeler.apply_triple_barrier_labeling(data)

# Analyze results
print(f"Total samples: {len(labeled_data)}")
print(f"Label distribution: {labeled_data['label'].value_counts()}")
print(f"Average profit: {labeled_data['net_profit_pct'].mean():.4f}")
print(f"Win rate: {(labeled_data['net_profit_pct'] > 0).mean():.3f}")
```

### Example 2: Regime-Aware Optimization

```python
from src.training.steps.market_analysis.regime_aware_triple_barrier_optimizer import optimize_regime_barriers

# Create data with regime information
data['hmm_regime'] = np.random.choice([0, 1, 2], 1000, p=[0.4, 0.4, 0.2])

# Optimize regime parameters
optimizer = optimize_regime_barriers(
    data,
    regime_column='hmm_regime',
    save_results='regime_optimization.json'
)

# Apply optimized labeling
labeled_data = optimizer.apply_optimized_labeling(data)

# Generate optimization report
report = optimizer.generate_optimization_report()
print(f"Optimization completed for {report['summary']['total_regimes']} regimes")
print(f"Average Sharpe ratio: {report['summary']['avg_sharpe_ratio']:.3f}")
```

### Example 3: Comprehensive Validation

```python
from src.training.steps.market_analysis.triple_barrier_validator import validate_triple_barrier_implementation

# Validate the labeling implementation
validation_report = validate_triple_barrier_implementation(data, labeled_data)

print(f"Overall validation score: {validation_report.overall_score:.3f}")
print(f"Passed checks: {validation_report.passed_checks}/{validation_report.total_checks}")

if validation_report.critical_issues:
    print("Critical issues found:")
    for issue in validation_report.critical_issues:
        print(f"  - {issue}")

if validation_report.recommendations:
    print("Recommendations:")
    for rec in validation_report.recommendations:
        print(f"  - {rec}")
```

### Example 4: Complete Pipeline Integration

```python
from src.training.steps.market_analysis.enhanced_market_analysis_with_triple_barrier import run_enhanced_market_analysis_with_triple_barrier

# Run complete analysis pipeline
results = run_enhanced_market_analysis_with_triple_barrier(
    data=data,
    symbol='ETHUSDT',
    exchange='BINANCE',
    timeframe='1m',
    config=config,
    output_dir='results/triple_barrier_analysis'
)

# Analyze results
print(f"Analysis completed in {results['execution_time']:.2f} seconds")
print(f"Labeled samples: {results['triple_barrier_labeling']['total_samples']}")
print(f"Win rate: {results['performance_metrics']['win_rate']:.3f}")
print(f"Sharpe ratio: {results['performance_metrics']['sharpe_ratio']:.3f}")
```

### Example 5: Performance Benchmarking

```python
from src.training.steps.market_analysis.triple_barrier_labeling import benchmark_triple_barrier_methods

# Benchmark different implementations
benchmark_results = benchmark_triple_barrier_methods(data)

print(f"Standard implementation: {benchmark_results['standard_time']:.3f}s")
print(f"Regime-aware implementation: {benchmark_results['regime_aware_time']:.3f}s")
print(f"Numba acceleration: {'Available' if benchmark_results['numba_available'] else 'Not available'}")
```

## Advanced Features

### Custom Objective Functions

```python
from src.training.steps.market_analysis.regime_aware_triple_barrier_optimizer import RegimeAwareTripleBarrierOptimizer

# Create optimizer with custom objective function
optimizer = RegimeAwareTripleBarrierOptimizer({
    'objective_function': 'profit_factor',  # Use profit factor instead of Sharpe ratio
    'profit_take_range': (0.001, 0.008),
    'stop_loss_range': (0.0005, 0.004),
    'max_iterations': 200
})

# Optimize parameters
regime_parameters = optimizer.optimize_regime_parameters(data)
```

### Custom Validation Rules

```python
from src.training.steps.market_analysis.triple_barrier_validator import TripleBarrierValidator

# Create validator with custom parameters
validator = TripleBarrierValidator({
    'min_win_rate': 0.4,           # Higher win rate requirement
    'max_drawdown_threshold': 0.15, # Lower drawdown threshold
    'min_sharpe_ratio': 0.8,       # Higher Sharpe ratio requirement
    'temporal_validation': True,
    'statistical_validation': True
})

# Validate implementation
report = validator.validate_triple_barrier_implementation(data, labeled_data)
```

### Batch Processing

```python
import os
from pathlib import Path

def process_multiple_symbols(data_directory: str, output_directory: str):
    """Process multiple symbols in batch."""
    
    data_path = Path(data_directory)
    output_path = Path(output_directory)
    output_path.mkdir(exist_ok=True)
    
    for file_path in data_path.glob('*.parquet'):
        symbol = file_path.stem.split('_')[1]  # Extract symbol from filename
        
        # Load data
        data = pd.read_parquet(file_path)
        
        # Run analysis
        results = run_enhanced_market_analysis_with_triple_barrier(
            data, symbol, 'BINANCE', '1m',
            output_dir=str(output_path / symbol)
        )
        
        print(f"Processed {symbol}: {results['performance_metrics']['win_rate']:.3f} win rate")

# Process all symbols
process_multiple_symbols('data/klines', 'results/batch_analysis')
```

## Performance Optimization

### Numba Acceleration

The system automatically uses Numba acceleration when available:

```python
# Check if Numba is available
try:
    import numba
    print("Numba acceleration available")
except ImportError:
    print("Numba not available - using Python implementation")
```

### Memory Optimization

For large datasets, consider chunking:

```python
def process_large_dataset(data: pd.DataFrame, chunk_size: int = 10000):
    """Process large dataset in chunks."""
    
    results = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i + chunk_size]
        
        # Process chunk
        labeled_chunk = apply_triple_barrier_labeling(chunk)
        results.append(labeled_chunk)
        
        print(f"Processed chunk {i//chunk_size + 1}/{(len(data)-1)//chunk_size + 1}")
    
    return pd.concat(results, ignore_index=True)

# Process large dataset
large_data = pd.read_parquet('large_dataset.parquet')
labeled_data = process_large_dataset(large_data, chunk_size=5000)
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def process_symbol_parallel(args):
    """Process a single symbol (for parallel execution)."""
    symbol, data_path = args
    
    data = pd.read_parquet(data_path)
    results = run_enhanced_market_analysis_with_triple_barrier(
        data, symbol, 'BINANCE', '1m'
    )
    
    return symbol, results

def parallel_batch_processing(data_directory: str):
    """Process multiple symbols in parallel."""
    
    data_path = Path(data_directory)
    file_paths = list(data_path.glob('*.parquet'))
    
    # Prepare arguments
    args = [(path.stem.split('_')[1], path) for path in file_paths]
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(process_symbol_parallel, args))
    
    return dict(results)

# Process symbols in parallel
results = parallel_batch_processing('data/klines')
```

## Validation and Testing

### Running Tests

```bash
# Run all tests
python -m pytest src/training/steps/market_analysis/test_triple_barrier_labeling.py -v

# Run specific test categories
python -m pytest src/training/steps/market_analysis/test_triple_barrier_labeling.py::TestTripleBarrierLabeling -v
python -m pytest src/training/steps/market_analysis/test_triple_barrier_labeling.py::TestPerformance -v

# Run with coverage
python -m pytest src/training/steps/market_analysis/test_triple_barrier_labeling.py --cov=src.training.steps.market_analysis --cov-report=html
```

### Custom Test Cases

```python
import pytest
from src.training.steps.market_analysis.triple_barrier_labeling import apply_triple_barrier_labeling

def test_custom_scenario():
    """Test a custom scenario."""
    
    # Create test data with specific characteristics
    data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [102, 103, 104, 105, 106],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    
    # Apply labeling
    labeled_data = apply_triple_barrier_labeling(data)
    
    # Assertions
    assert len(labeled_data) > 0
    assert 'label' in labeled_data.columns
    assert all(label in [-1, 1] for label in labeled_data['label'])
```

## API Reference

### Core Classes

#### MarketAnalysisTripleBarrierLabeling

```python
class MarketAnalysisTripleBarrierLabeling:
    def __init__(self, config: TripleBarrierConfig):
        """Initialize the triple barrier labeler."""
    
    def apply_triple_barrier_labeling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply triple barrier labeling to market data."""
```

#### RegimeAwareTripleBarrierOptimizer

```python
class RegimeAwareTripleBarrierOptimizer:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the regime optimizer."""
    
    def optimize_regime_parameters(self, data: pd.DataFrame) -> Dict[Union[int, str], RegimeBarrierParams]:
        """Optimize parameters for each regime."""
    
    def apply_optimized_labeling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply optimized regime-aware labeling."""
```

#### TripleBarrierValidator

```python
class TripleBarrierValidator:
    def __init__(self, config: Dict[str, Any]):
        """Initialize the validator."""
    
    def validate_triple_barrier_implementation(self, data: pd.DataFrame, labeled_data: pd.DataFrame) -> ValidationReport:
        """Validate triple barrier implementation."""
```

### Convenience Functions

```python
# Basic labeling
apply_triple_barrier_labeling(data, **kwargs) -> pd.DataFrame

# Create labeler
create_triple_barrier_labeler(**kwargs) -> MarketAnalysisTripleBarrierLabeling

# Regime optimization
optimize_regime_barriers(data, **kwargs) -> RegimeAwareTripleBarrierOptimizer

# Validation
validate_triple_barrier_implementation(data, labeled_data, **kwargs) -> ValidationReport
quick_validate_triple_barrier(data, labeled_data) -> bool

# Complete pipeline
run_enhanced_market_analysis_with_triple_barrier(data, symbol, exchange, timeframe, **kwargs) -> Dict[str, Any]
quick_triple_barrier_analysis(data, **kwargs) -> pd.DataFrame

# Benchmarking
benchmark_triple_barrier_methods(data) -> Dict[str, float]
```

## Troubleshooting

### Common Issues

#### 1. "Missing required OHLC columns"

**Problem**: The data doesn't contain the required OHLC columns.

**Solution**: Ensure your data has columns named 'open', 'high', 'low', 'close' (case-insensitive):

```python
# Rename columns if needed
data = data.rename(columns={
    'Open': 'open',
    'High': 'high', 
    'Low': 'low',
    'Close': 'close'
})
```

#### 2. "Regime column not found"

**Problem**: Regime-aware labeling is enabled but the regime column is missing.

**Solution**: Either add the regime column or disable regime-aware labeling:

```python
# Option 1: Add regime column
data['hmm_regime'] = np.random.choice([0, 1, 2], len(data))

# Option 2: Disable regime-aware labeling
labeled_data = apply_triple_barrier_labeling(data, regime_aware=False)
```

#### 3. "Validation score below threshold"

**Problem**: The labeling validation score is below the configured threshold.

**Solution**: Adjust parameters or investigate data quality:

```python
# Check validation details
report = validate_triple_barrier_implementation(data, labeled_data)
print("Validation issues:", report.critical_issues)
print("Recommendations:", report.recommendations)

# Adjust parameters
labeled_data = apply_triple_barrier_labeling(
    data,
    profit_take_multiplier=0.003,  # Increase profit take
    stop_loss_multiplier=0.002     # Increase stop loss
)
```

#### 4. "Insufficient data for optimization"

**Problem**: Not enough data for regime parameter optimization.

**Solution**: Use more data or disable optimization:

```python
# Option 1: Use more data
if len(data) < 1000:
    print("Warning: Consider using more data for reliable optimization")

# Option 2: Disable optimization
config = MarketAnalysisTripleBarrierConfig(optimize_regime_parameters=False)
```

#### 5. Performance Issues

**Problem**: Slow execution with large datasets.

**Solution**: Enable optimizations and consider chunking:

```python
# Enable Numba acceleration
import numba
print("Numba available:", numba is not None)

# Use chunking for large datasets
def process_in_chunks(data, chunk_size=5000):
    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i + chunk_size]
        labeled_chunk = apply_triple_barrier_labeling(chunk)
        results.append(labeled_chunk)
    return pd.concat(results, ignore_index=True)
```

### Debug Mode

Enable debug logging for detailed information:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Run analysis with debug information
results = run_enhanced_market_analysis_with_triple_barrier(data, 'TEST', 'TEST', '1m')
```

### Performance Profiling

Profile performance to identify bottlenecks:

```python
import cProfile
import pstats

# Profile the analysis
profiler = cProfile.Profile()
profiler.enable()

results = run_enhanced_market_analysis_with_triple_barrier(data, 'TEST', 'TEST', '1m')

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Print top 10 functions
```

## Contributing

### Development Setup

1. **Install Dependencies**:
   ```bash
   pip install pandas numpy scipy scikit-learn numba pytest
   ```

2. **Run Tests**:
   ```bash
   python -m pytest src/training/steps/market_analysis/test_triple_barrier_labeling.py -v
   ```

3. **Code Style**:
   ```bash
   # Format code
   black src/training/steps/market_analysis/
   
   # Lint code
   flake8 src/training/steps/market_analysis/
   ```

### Adding New Features

1. **Create Feature Branch**:
   ```bash
   git checkout -b feature/new-feature
   ```

2. **Add Tests**:
   ```python
   def test_new_feature():
       # Test implementation
       pass
   ```

3. **Update Documentation**:
   - Update this documentation
   - Add docstrings to new functions
   - Update examples if needed

4. **Submit Pull Request**:
   - Ensure all tests pass
   - Update documentation
   - Add examples for new features

### Performance Optimization Guidelines

1. **Use Numba for Loops**: Decorate performance-critical functions with `@numba.jit`
2. **Vectorize Operations**: Use NumPy vectorized operations instead of Python loops
3. **Memory Efficiency**: Use appropriate data types and avoid unnecessary copies
4. **Caching**: Cache expensive computations when possible

### Testing Guidelines

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test complete workflows
3. **Performance Tests**: Test with large datasets
4. **Edge Cases**: Test with problematic data and edge cases

## Conclusion

The MARKET_ANALYSIS Triple Barrier Labeling system provides a comprehensive, production-ready implementation of the triple barrier method with advanced features like regime-aware optimization, comprehensive validation, and seamless pipeline integration. 

Key benefits:

- **Flexibility**: Configurable parameters for different market conditions
- **Performance**: Optimized with Numba acceleration and vectorization
- **Reliability**: Comprehensive validation and error handling
- **Integration**: Seamless integration with existing market analysis pipeline
- **Extensibility**: Modular design for easy customization and extension

For questions, issues, or contributions, please refer to the project repository or contact the development team.