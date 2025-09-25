# Unified NAS-TAS System Guide

## Overview

The Unified NAS-TAS System consolidates all Neural Architecture Search (NAS) and Tree Architecture Search (TAS) components into a single, cohesive framework. This system eliminates code duplication, provides unified interfaces, and offers enhanced performance and functionality.

## Key Benefits

- **Unified Interface**: Single interface for both NAS and TAS operations
- **Code Consolidation**: Eliminates 1134+ duplicate utility imports across 208 files
- **Enhanced Performance**: Hardware optimization and parallel processing
- **Better Maintainability**: Centralized configuration and error handling
- **Future-Proof**: Extensible architecture for new algorithms and methods
- **Backward Compatibility**: Legacy components still work through adapters

## Architecture

```
Unified System
├── UnifiedSearchEngine (Search strategies)
├── UnifiedMultiObjectiveOptimizer (Optimization algorithms)
├── UnifiedEconomicEvaluator (Economic evaluation)
├── UnifiedRegimeDetector (Regime detection)
├── UnifiedUtilities (Common utilities)
├── UnifiedConfig (Configuration management)
└── BackwardCompatibility (Legacy support)
```

## Quick Start

### 1. Basic Usage

```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    UnifiedSearchEngine, SearchConfig, ArchitectureType, SearchStrategy
)

# Create unified search engine
config = SearchConfig(
    architecture_type=ArchitectureType.HYBRID,
    search_strategy=SearchStrategy.ENHANCED_BAYESIAN,
    max_iterations=100
)

engine = UnifiedSearchEngine(config)

# Perform search
result = engine.search(search_space, objective_function)

print(f"Best architecture: {result.best_architecture}")
print(f"Best score: {max(result.best_scores.values())}")
```

### 2. Complete System Setup

```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    create_unified_system, UnifiedConfig
)

# Create complete unified system
system = create_unified_system()

# Use components
search_result = system['search_engine'].search(search_space, objective_function)
optimization_result = system['optimizer'].optimize(objective_functions, parameter_bounds)
evaluation_result = system['evaluator'].evaluate(predictions, market_data, returns)
regime_result = system['regime_detector'].detect_regimes(data)
```

## Component Details

### 1. UnifiedSearchEngine

**Purpose**: Unified interface for all search strategies (Bayesian, Evolutionary, RL, etc.)

**Key Features**:
- Architecture-agnostic search strategies
- Advanced optimization techniques (NSGA-II, SPEA2, Bayesian TPE)
- Hardware optimization and parallel processing
- Comprehensive logging and monitoring

**Usage**:
```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    UnifiedSearchEngine, SearchConfig, SearchStrategy
)

config = SearchConfig(
    search_strategy=SearchStrategy.EVOLUTIONARY,
    max_iterations=100,
    population_size=50
)

engine = UnifiedSearchEngine(config)
result = engine.search(search_space, objective_function)
```

### 2. UnifiedMultiObjectiveOptimizer

**Purpose**: Multi-objective optimization with multiple algorithms

**Key Features**:
- NSGA-II, SPEA2, Bayesian optimization
- Pareto frontier management
- Architecture-agnostic optimization
- Advanced performance metrics

**Usage**:
```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    UnifiedMultiObjectiveOptimizer, UnifiedMultiObjectiveConfig, OptimizationAlgorithm
)

config = UnifiedMultiObjectiveConfig(
    algorithm=OptimizationAlgorithm.NSGA2,
    objectives=['accuracy', 'efficiency', 'profitability'],
    objective_weights=[0.4, 0.3, 0.3]
)

optimizer = UnifiedMultiObjectiveOptimizer(config)
result = optimizer.optimize(objective_functions, parameter_bounds)
```

### 3. UnifiedEconomicEvaluator

**Purpose**: Economic significance and trading viability evaluation

**Key Features**:
- Comprehensive financial metrics (Sharpe ratio, max drawdown, etc.)
- Economic significance calculation
- Trading viability assessment
- Architecture-specific performance metrics

**Usage**:
```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    UnifiedEconomicEvaluator, EconomicEvaluationConfig
)

config = EconomicEvaluationConfig(
    evaluation_types=['economic_significance', 'trading_viability'],
    significance_threshold=0.05,
    risk_free_rate=0.02
)

evaluator = UnifiedEconomicEvaluator(config)
result = evaluator.evaluate(predictions, market_data, returns)
```

### 4. UnifiedRegimeDetector

**Purpose**: Regime detection with multiple methods

**Key Features**:
- Clustering, HMM, neural networks, tree-based methods
- Regime validation and quality assessment
- Architecture-agnostic detection
- Comprehensive regime analysis

**Usage**:
```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    UnifiedRegimeDetector, RegimeDetectionConfig
)

config = RegimeDetectionConfig(
    method='hybrid',
    n_regimes=3,
    min_regime_duration=10
)

detector = UnifiedRegimeDetector(config)
result = detector.detect_regimes(data, features)
```

### 5. UnifiedUtilities

**Purpose**: Common utilities for data processing and validation

**Key Features**:
- Data validation and optimization
- Architecture-specific optimizations
- Hardware optimization integration
- Performance monitoring

**Usage**:
```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    UnifiedUtilities, DataType, ArchitectureType
)

utilities = UnifiedUtilities()

# Validate data
validation_result = utilities.validate_data(data, DataType.MARKET_DATA, ArchitectureType.NEURAL)

# Optimize data
optimized_data = utilities.optimize_data(data, ArchitectureType.NEURAL, DataType.MARKET_DATA)
```

### 6. UnifiedConfig

**Purpose**: Comprehensive configuration management

**Key Features**:
- Unified configuration for all components
- Multiple format support (JSON, YAML, Pickle)
- Configuration validation
- Easy component configuration management

**Usage**:
```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    UnifiedConfig, create_default_config
)

# Create default configuration
config = create_default_config()

# Modify component configurations
config.search_config.max_iterations = 200
config.optimization_config.algorithm = 'nsga2'

# Save configuration
config.save_to_file('config.yaml', ConfigFormat.YAML)
```

## Migration Guide

### From Legacy NAS Components

```python
# Old way
from src.training.steps.market_analysis.nas_regime.core.enhanced_nas_engine import EnhancedNASEngine

engine = EnhancedNASEngine(config)
result = engine.search(search_space, objective_function)

# New way
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    UnifiedSearchEngine, SearchConfig, ArchitectureType
)

config = SearchConfig(architecture_type=ArchitectureType.NEURAL)
engine = UnifiedSearchEngine(config)
result = engine.search(search_space, objective_function)
```

### From Legacy TAS Components

```python
# Old way
from src.training.steps.market_analysis.tas_regime.core.tas_engine import TASEngine

engine = TASEngine(config)
result = engine.search(search_space, objective_function)

# New way
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    UnifiedSearchEngine, SearchConfig, ArchitectureType
)

config = SearchConfig(architecture_type=ArchitectureType.TREE)
engine = UnifiedSearchEngine(config)
result = engine.search(search_space, objective_function)
```

### Using Backward Compatibility

If you need to maintain existing code temporarily:

```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils import (
    LegacyNASEngineAdapter, LegacyTASEngineAdapter
)

# Legacy NAS engine (uses unified system underneath)
nas_engine = LegacyNASEngineAdapter(config)
result = nas_engine.search(search_space, objective_function)

# Legacy TAS engine (uses unified system underneath)
tas_engine = LegacyTASEngineAdapter(config)
result = tas_engine.search(search_space, objective_function)
```

## Configuration Examples

### 1. Neural Architecture Search

```python
config = UnifiedConfig()

# Configure for neural architectures
config.search_config.architecture_type = ArchitectureType.NEURAL
config.search_config.search_strategy = SearchStrategy.ENHANCED_BAYESIAN
config.search_config.max_iterations = 150
config.search_config.population_size = 75

# Configure optimization
config.optimization_config.algorithm = OptimizationAlgorithm.NSGA2
config.optimization_config.objectives = ['accuracy', 'efficiency', 'complexity']
config.optimization_config.objective_weights = [0.5, 0.3, 0.2]

# Configure evaluation
config.evaluation_config.architecture_type = ArchitectureType.NEURAL
config.evaluation_config.evaluation_types = ['economic_significance', 'trading_viability']
```

### 2. Tree Architecture Search

```python
config = UnifiedConfig()

# Configure for tree architectures
config.search_config.architecture_type = ArchitectureType.TREE
config.search_config.search_strategy = SearchStrategy.EVOLUTIONARY
config.search_config.max_iterations = 100
config.search_config.population_size = 50

# Configure optimization
config.optimization_config.algorithm = OptimizationAlgorithm.NSGA2
config.optimization_config.objectives = ['accuracy', 'interpretability', 'speed']
config.optimization_config.objective_weights = [0.4, 0.4, 0.2]

# Configure evaluation
config.evaluation_config.architecture_type = ArchitectureType.TREE
config.evaluation_config.evaluation_types = ['economic_significance', 'performance_metrics']
```

### 3. Hybrid Architecture Search

```python
config = UnifiedConfig()

# Configure for hybrid architectures
config.search_config.architecture_type = ArchitectureType.HYBRID
config.search_config.search_strategy = SearchStrategy.HYBRID
config.search_config.max_iterations = 200
config.search_config.population_size = 100

# Configure optimization
config.optimization_config.algorithm = OptimizationAlgorithm.HYBRID
config.optimization_config.objectives = ['accuracy', 'efficiency', 'robustness', 'interpretability']
config.optimization_config.objective_weights = [0.3, 0.25, 0.25, 0.2]

# Configure evaluation
config.evaluation_config.architecture_type = ArchitectureType.HYBRID
config.evaluation_config.evaluation_types = ['economic_significance', 'trading_viability', 'performance_metrics']
```

## Performance Optimization

### 1. Hardware Optimization

```python
config = UnifiedConfig()

# Enable hardware optimization
config.search_config.enable_gpu_acceleration = True
config.search_config.enable_parallel_processing = True
config.search_config.n_jobs = -1  # Use all available cores

# Set memory limits
config.search_config.memory_limit_gb = 8.0
config.utility_config.memory_limit_gb = 4.0
```

### 2. Parallel Processing

```python
config = UnifiedConfig()

# Enable parallel processing for all components
config.search_config.enable_parallel_processing = True
config.optimization_config.enable_parallel_processing = True
config.utility_config.enable_parallel_processing = True

# Set number of jobs
config.search_config.n_jobs = -1
config.optimization_config.n_jobs = -1
config.utility_config.n_jobs = -1
```

### 3. Memory Optimization

```python
config = UnifiedConfig()

# Enable memory optimization
config.utility_config.enable_memory_optimization = True
config.utility_config.enable_hardware_optimization = True

# Set memory limits
config.max_memory_usage_gb = 16.0
config.search_config.memory_limit_gb = 8.0
config.utility_config.memory_limit_gb = 4.0
```

## Best Practices

### 1. Configuration Management

- Use `UnifiedConfig` for system-wide configuration
- Save configurations to files for reproducibility
- Validate configurations before use
- Use configuration templates for different scenarios

### 2. Error Handling

- Always check `result.success` before using results
- Handle `error_message` appropriately
- Use try-catch blocks for critical operations
- Log errors for debugging

### 3. Performance Monitoring

- Use performance reports for optimization
- Monitor memory usage and execution times
- Use caching for repeated operations
- Profile critical sections

### 4. Data Validation

- Always validate data before processing
- Use appropriate data types and architectures
- Check data quality scores
- Handle missing values appropriately

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce memory limits or enable optimization
3. **Performance Issues**: Enable parallel processing and hardware optimization
4. **Configuration Errors**: Validate configuration before use

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Create system with debug enabled
config = UnifiedConfig()
config.search_config.enable_logging = True
config.search_config.log_level = 'DEBUG'

system = create_unified_system(config)
```

### Performance Profiling

```python
config = UnifiedConfig()
config.enable_profiling = True
config.profiling_output_file = 'profile_output.txt'

system = create_unified_system(config)
```

## Future Roadmap

- **Enhanced Algorithms**: New search strategies and optimization algorithms
- **Advanced Hardware Support**: Better GPU and multi-core optimization
- **Cloud Integration**: Support for distributed computing
- **AutoML Features**: Automated hyperparameter tuning
- **Real-time Processing**: Streaming data support
- **Visualization Tools**: Interactive dashboards and plots

## Support

For questions, issues, or contributions:

- **Documentation**: This guide and inline documentation
- **Examples**: Check the examples directory
- **Issues**: Report bugs and request features
- **Contributions**: Submit pull requests for improvements

## License

This unified system is part of the larger project and follows the same licensing terms.