# Testing and Performance System

This directory contains comprehensive testing and performance monitoring systems for the data-driven clustering pipeline.

## 🧪 **Testing Framework**

### **1. Automated Regression Tests**
- **File**: `test_economic_validation_regression.py`
- **Purpose**: Ensures each release reproduces previous economic-validation scores within tolerance
- **Features**:
  - Synthetic data generation with known regime structures
  - Economic validation score comparison
  - Baseline score management
  - Tolerance-based pass/fail criteria
  - Comprehensive test reporting

### **2. Synthetic Dataset Testing**
- **File**: `test_synthetic_datasets.py`
- **Purpose**: Tests system behavior across different market scenarios
- **Features**:
  - Multiple market scenarios (bull market, crisis, sideways, high volatility)
  - Regime persistence testing
  - Economic coherence validation
  - Multi-objective optimization testing
  - Scenario-specific baseline comparison

### **3. Comprehensive Test Runner**
- **File**: `test_runner.py`
- **Purpose**: Unified test execution and reporting
- **Features**:
  - Orchestrates all testing activities
  - Performance profiling integration
  - Baseline management
  - Comprehensive reporting
  - Command-line interface

## 📊 **Performance Monitoring**

### **1. Performance Profiler**
- **File**: `../performance/performance_profiler.py`
- **Purpose**: Detailed performance analysis of system components
- **Features**:
  - Feature generation performance profiling
  - Multi-objective optimization profiling
  - Economic validation performance analysis
  - Memory usage tracking
  - CPU utilization monitoring
  - Parallelization effectiveness analysis
  - Caching performance evaluation

### **2. Performance Monitor**
- **File**: `../performance/performance_monitor.py`
- **Purpose**: Real-time performance monitoring and alerting
- **Features**:
  - Real-time metrics collection
  - Performance threshold monitoring
  - Automatic alert generation
  - Historical performance tracking
  - Intelligent caching system
  - Performance optimization recommendations

## 🚀 **Quick Start**

### **Run All Tests**
```bash
cd src/training/steps/market_analysis/hdbscan_clustering/tests
python test_runner.py
```

### **Run Specific Test Types**
```bash
# Regression tests only
python test_runner.py --no-performance --no-synthetic

# Performance profiling only
python test_runner.py --no-regression --no-synthetic

# Synthetic tests only
python test_runner.py --no-performance --no-regression
```

### **Run with Custom Parameters**
```bash
# Custom tolerance and save baseline
python test_runner.py --tolerance 0.03 --save-baseline --iterations 5

# Custom output directory
python test_runner.py --output-dir my_test_results
```

## 📋 **Test Configuration**

### **Regression Test Configuration**
```python
from test_economic_validation_regression import create_default_test_cases

# Default test cases
test_cases = create_default_test_cases()

# Custom test cases
custom_test_cases = [
    {
        'name': 'custom_test',
        'n_samples': 1500,
        'n_features': 75,
        'n_regimes': 4,
        'noise_level': 0.15,
        'regime_persistence': 0.75
    }
]
```

### **Synthetic Test Configuration**
```python
from test_synthetic_datasets import create_default_test_scenarios

# Default scenarios
test_scenarios = create_default_test_scenarios()

# Custom scenarios
custom_scenarios = [
    {
        'name': 'custom_scenario',
        'scenario': 'bull_market',
        'n_samples': 2000,
        'n_features': 100,
        'n_regimes': 5,
        'noise_level': 0.1,
        'regime_persistence': 0.8
    }
]
```

## 📊 **Performance Monitoring Usage**

### **Start Performance Monitoring**
```python
from performance.performance_monitor import start_performance_monitoring, get_performance_summary

# Start monitoring
start_performance_monitoring()

# Get current performance summary
summary = get_performance_summary()
print(f"Memory usage: {summary['monitor']['memory_usage']['current_mb']:.1f}MB")
print(f"CPU usage: {summary['monitor']['cpu_usage']['current_percent']:.1f}%")
print(f"Cache hit rate: {summary['cache']['hit_rate']:.1%}")
```

### **Use Intelligent Caching**
```python
from performance.performance_monitor import get_intelligent_cache, cached

# Get cache instance
cache = get_intelligent_cache()

# Cached function
@cached(ttl=300, cache=cache)
def expensive_computation(data):
    # Expensive operation
    return processed_data

# Use cached function
result = expensive_computation(my_data)
```

### **Get Performance Optimization Recommendations**
```python
from performance.performance_monitor import get_performance_optimizer

# Get optimizer
optimizer = get_performance_optimizer()

# Analyze performance
analysis = optimizer.analyze_performance()
print(f"Recommendations: {analysis['total_recommendations']}")

# Get optimization plan
plan = optimizer.get_optimization_plan()
print(f"High priority: {plan['overview']['high_priority']}")
```

## 📈 **Performance Profiling Usage**

### **Profile Feature Generation**
```python
from performance.performance_profiler import run_performance_profiling
import pandas as pd

# Generate sample data
market_data = pd.DataFrame({
    'timestamp': pd.date_range('2020-01-01', periods=1000, freq='1H'),
    'open': 100 + np.cumsum(np.random.normal(0, 0.01, 1000)),
    'close': 100 + np.cumsum(np.random.normal(0, 0.01, 1000)),
    'volume': np.random.lognormal(5, 0.5, 1000)
})

# Run profiling
results = run_performance_profiling(market_data, n_iterations=3)
print(f"Feature generation: {results['feature_generation']['avg_execution_time']:.3f}s")
print(f"Peak memory: {results['memory_usage']['peak_memory_mb']:.1f}MB")
```

### **Profile Multi-Objective Optimization**
```python
from performance.performance_profiler import PerformanceProfiler

# Create profiler
profiler = PerformanceProfiler()

# Profile optimization
optimization_stats = profiler.profile_multi_objective_optimization(
    market_data=market_data,
    features=features,
    feature_names=feature_names,
    n_trials=50
)
print(f"Optimization time: {optimization_stats['execution_time']:.3f}s")
```

## 📊 **Test Results and Reporting**

### **Regression Test Results**
```python
# Run regression tests
from test_economic_validation_regression import run_regression_tests

results = run_regression_tests(tolerance=0.05, save_baseline=False)

print(f"Total Tests: {results['total_tests']}")
print(f"Passed: {results['passed_tests']}")
print(f"Failed: {results['failed_tests']}")
print(f"Pass Rate: {results['pass_rate']:.1%}")

# Check specific test results
for test_name, comparison in results['comparison_results'].items():
    if comparison['status'] != 'PASS':
        print(f"{test_name}: {comparison['message']}")
```

### **Synthetic Test Results**
```python
# Run synthetic tests
from test_synthetic_datasets import run_synthetic_tests

results = run_synthetic_tests(tolerance=0.05, save_baseline=False)

print(f"Total Tests: {results['total_tests']}")
print(f"Passed: {results['passed_tests']}")
print(f"Pass Rate: {results['pass_rate']:.1%}")

# Check scenario-specific results
for test_name, comparison in results['comparison_results'].items():
    if comparison['status'] != 'PASS':
        print(f"{test_name}: {comparison['message']}")
```

## 🔧 **Configuration Options**

### **Test Tolerance**
- **Default**: 0.05 (5%)
- **Purpose**: Maximum allowed difference in scores
- **Usage**: `--tolerance 0.03` for stricter testing

### **Performance Profiling Iterations**
- **Default**: 3
- **Purpose**: Number of iterations for averaging
- **Usage**: `--iterations 5` for more accurate results

### **Baseline Management**
- **Purpose**: Save current results as new baseline
- **Usage**: `--save-baseline` to update baseline scores

### **Test Categories**
- **Performance Profiling**: `--no-performance` to disable
- **Regression Tests**: `--no-regression` to disable
- **Synthetic Tests**: `--no-synthetic` to disable

## 📁 **File Structure**

```
tests/
├── test_economic_validation_regression.py    # Regression testing
├── test_synthetic_datasets.py                # Synthetic dataset testing
├── test_runner.py                            # Comprehensive test runner
├── README_TESTING.md                         # This documentation
└── ../performance/
    ├── performance_profiler.py               # Performance profiling
    ├── performance_monitor.py                # Real-time monitoring
    └── performance_profiles/                 # Profile results directory
```

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify module structure

2. **Memory Issues**
   - Reduce test data size
   - Increase system memory
   - Enable garbage collection

3. **Performance Issues**
   - Check CPU utilization
   - Monitor memory usage
   - Review cache effectiveness

4. **Test Failures**
   - Check tolerance settings
   - Review baseline scores
   - Verify test data generation

### **Debug Mode**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run tests with debug logging
results = run_regression_tests(tolerance=0.05)
```

## 📊 **Performance Benchmarks**

### **Expected Performance**
- **Feature Generation**: < 2.0s for 1000 samples
- **Multi-Objective Optimization**: < 30.0s for 50 trials
- **Economic Validation**: < 5.0s for 1000 samples
- **Peak Memory**: < 1000MB for standard datasets
- **Cache Hit Rate**: > 70% for repeated operations

### **Scaling Considerations**
- **Memory**: Linear with data size
- **CPU**: Benefits from parallelization
- **Cache**: Improves with repeated operations
- **I/O**: Minimal for in-memory operations

## 🔄 **Continuous Integration**

### **GitHub Actions Example**
```yaml
name: Test Data-Driven Clustering
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: |
          cd src/training/steps/market_analysis/hdbscan_clustering/tests
          python test_runner.py --tolerance 0.05
```

## 📚 **Additional Resources**

- **Main Documentation**: `../README_DATA_DRIVEN.md`
- **System Summary**: `../SYSTEM_SUMMARY.md`
- **Migration Guide**: `../MIGRATION_GUIDE.md`
- **Examples**: `../examples/`

## 🎯 **Best Practices**

1. **Run tests regularly** to catch regressions early
2. **Update baselines** when making intentional changes
3. **Monitor performance** during development
4. **Use caching** for repeated operations
5. **Profile before optimizing** to identify bottlenecks
6. **Set appropriate tolerances** for your use case
7. **Document test scenarios** for team understanding
8. **Automate testing** in CI/CD pipeline