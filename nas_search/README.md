# Evolutionary Architecture Search (EAS)

A comprehensive evolutionary algorithm implementation for Neural Architecture Search (NAS), featuring advanced optimization techniques, hardware-specific optimizations, and integration with ML utilities.

## Features

### 🧬 Core Evolutionary Algorithm
- **Population-based search** with configurable population size and generations
- **Genetic operators**: Tournament selection, crossover, and mutation
- **Elite preservation** to maintain best solutions across generations
- **Convergence detection** with early stopping capabilities
- **Diversity tracking** to prevent premature convergence

### 🖥️ Hardware Optimization
- **M1 Apple Silicon optimization** with GPU and CPU acceleration
- **Memory management** with automatic cleanup and monitoring
- **Parallel processing** with optimized thread pools
- **Performance monitoring** with detailed metrics tracking

### 🤖 Machine Learning Integration
- **Cross-validation** for robust fitness evaluation
- **Hyperparameter optimization** with HPO, Grid Search, and Bayesian methods
- **Performance constraints** with training time and memory limits
- **Multiple metrics** support (accuracy, precision, recall, F1, AUC-ROC)

### 📊 Advanced Features
- **Structured logging** with timestamped output
- **Serialization** support for saving/loading architectures
- **Progress tracking** with real-time monitoring
- **Error handling** with graceful degradation
- **Memory optimization** with automatic garbage collection

## Architecture

### Core Classes

#### `EvolutionaryArchitectureSearch`
Main class that orchestrates the evolutionary search process.

```python
nas = EvolutionaryArchitectureSearch(
    architecture_config=ArchitectureConfig(),
    evolutionary_config=EvolutionaryConfig(),
    fitness_config=FitnessConfig(),
    data=(X, y),
    log_dir="search_results"
)

best_architecture = nas.run_evolution()
```

#### `Architecture`
Represents a neural network architecture with layers, parameters, and performance metrics.

```python
layers = [
    {'type': 'dense', 'neurons': 64, 'activation': 'relu'},
    {'type': 'dropout', 'neurons': 32, 'dropout': 0.2},
    {'type': 'dense', 'neurons': 16, 'activation': 'sigmoid'}
]

arch = Architecture(layers, config)
```

### Configuration Classes

#### `ArchitectureConfig`
Defines constraints for neural architectures:
- Layer count limits
- Neuron count limits
- Layer types and activation functions
- Parameter and FLOP constraints

#### `EvolutionaryConfig`
Controls the evolutionary algorithm:
- Population size and generations
- Genetic operator rates
- Selection pressure and diversity
- Parallel processing settings

#### `FitnessConfig`
Configures fitness evaluation:
- Evaluation metrics
- Cross-validation settings
- Training parameters
- Performance constraints

## Usage Examples

### Basic Usage

```python
import numpy as np
from evolutionary_search import (
    EvolutionaryArchitectureSearch,
    ArchitectureConfig,
    EvolutionaryConfig,
    FitnessConfig
)

# Create sample data
X = np.random.randn(1000, 20)
y = np.random.randint(0, 2, 1000)

# Configure search
arch_config = ArchitectureConfig(
    max_layers=8,
    min_layers=2,
    max_neurons_per_layer=512
)

evo_config = EvolutionaryConfig(
    population_size=50,
    max_generations=100,
    n_workers=4
)

fitness_config = FitnessConfig(
    cv_folds=5,
    max_training_epochs=100
)

# Run search
nas = EvolutionaryArchitectureSearch(
    architecture_config=arch_config,
    evolutionary_config=evo_config,
    fitness_config=fitness_config,
    data=(X, y),
    log_dir="nas_results"
)

best_architecture = nas.run_evolution()
print(f"Best fitness: {best_architecture.fitness:.4f}")
```

### Advanced Configuration

```python
# Custom architecture constraints
arch_config = ArchitectureConfig(
    max_layers=10,
    min_layers=3,
    max_neurons_per_layer=1024,
    min_neurons_per_layer=32,
    layer_types=['dense', 'conv1d', 'lstm', 'attention'],
    activation_functions=['relu', 'tanh', 'sigmoid', 'gelu'],
    max_parameters=1000000,
    min_parameters=10000
)

# Advanced evolutionary settings
evo_config = EvolutionaryConfig(
    population_size=100,
    max_generations=200,
    elite_size=10,
    tournament_size=5,
    crossover_rate=0.8,
    mutation_rate=0.2,
    mutation_strength=0.1,
    selection_pressure=2.0,
    diversity_weight=0.1,
    early_stopping_patience=30,
    convergence_threshold=1e-6,
    n_workers=8,
    use_parallel_evaluation=True
)

# Comprehensive fitness evaluation
fitness_config = FitnessConfig(
    primary_metric='accuracy',
    secondary_metrics=['precision', 'recall', 'f1_score', 'auc_roc'],
    cv_folds=10,
    use_stratified_cv=True,
    max_training_epochs=200,
    early_stopping_patience=20,
    learning_rate=0.001,
    batch_size=64,
    max_training_time=600.0,  # 10 minutes
    max_memory_usage=16.0,   # 16 GB
    min_accuracy_threshold=0.7
)
```

### Hardware Optimization

```python
# The system automatically detects and optimizes for M1 hardware
# GPU acceleration is used when available
# Memory monitoring prevents OOM errors
# CPU cores are optimized for parallel processing

# Manual hardware optimization
nas.gpu_manager.optimize_tensor_operations(data)
nas.memory_optimizer.start_monitoring()
nas.cpu_optimizer.create_optimized_thread_pool(max_workers=8)
```

### Monitoring and Logging

```python
# Real-time progress monitoring
with tprint_timer("Evolution"):
    best_architecture = nas.run_evolution()

# Structured logging
tprint_structured({
    'generation': nas.generation,
    'best_fitness': nas.best_architecture.fitness,
    'population_diversity': nas.calculate_diversity(nas.population)
})

# Search summary
summary = nas.get_search_summary()
print(f"Total evaluations: {summary['total_evaluations']}")
print(f"Average evaluation time: {summary['avg_evaluation_time']:.3f}s")
```

## Integration with Utility Modules

### Common Operations
- **Data validation** with `validate_dataframe_columns`
- **Safe operations** with `safe_dataframe_operation`
- **Memory optimization** with `optimize_memory`
- **File operations** with `safe_to_parquet`, `safe_read_parquet`

### Math Validation
- **Safe mathematical operations** with `safe_divide`, `safe_log`, `safe_sqrt`
- **Validation functions** with `validate_finite`, `validate_positive`
- **Statistical functions** with `safe_mean`, `safe_std`, `safe_correlation`

### Serialization
- **JSON serialization** for human-readable results
- **Pickle serialization** for complex objects
- **Parquet serialization** for large datasets
- **Universal serialization** with automatic format detection

### Logging and Monitoring
- **Timestamped logging** with `tprint` functions
- **Structured logging** with JSON output
- **Performance monitoring** with timing functions
- **Progress tracking** with progress bars

### ML Common Utilities
- **Hyperparameter optimization** with HPO, Grid Search, Bayesian methods
- **Cross-validation** with stratified and time series splits
- **Model evaluation** with comprehensive metrics
- **Feature selection** with advanced algorithms

### Hardware Optimization
- **M1 GPU acceleration** with Metal Performance Shaders
- **Memory optimization** with unified memory architecture
- **CPU optimization** with performance and efficiency cores
- **Parallel processing** with optimized thread pools

## Performance Characteristics

### Scalability
- **Population size**: 10-1000 individuals
- **Generations**: 10-1000 iterations
- **Parallel evaluation**: Up to 16 workers
- **Memory usage**: Automatic optimization and monitoring

### Optimization Features
- **Early stopping** to prevent overfitting
- **Convergence detection** for efficient search
- **Diversity maintenance** to avoid local optima
- **Elite preservation** to maintain best solutions

### Hardware Utilization
- **M1 GPU acceleration** when available
- **Memory monitoring** to prevent OOM
- **CPU optimization** for parallel processing
- **Automatic cleanup** to free resources

## File Structure

```
nas_search/
├── evolutionary_search.py      # Main implementation
├── test_evolutionary_search.py # Comprehensive tests
├── README.md                  # This documentation
└── logs/                      # Generated log files
    ├── nas_search_*.log       # Search logs
    ├── population_gen_*.json  # Population snapshots
    ├── best_architecture_*.json # Best architectures
    └── search_history_*.json  # Search history
```

## Dependencies

### Required
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `pathlib` - File system operations
- `concurrent.futures` - Parallel processing
- `threading` - Thread management
- `json` - Serialization
- `pickle` - Object serialization
- `time` - Timing operations
- `random` - Random number generation
- `logging` - Logging system

### Optional (for enhanced functionality)
- `torch` - PyTorch for neural networks
- `psutil` - System monitoring
- `colorama` - Colored output
- `sklearn` - Machine learning utilities
- `optuna` - Hyperparameter optimization

## Testing

Run the comprehensive test suite:

```bash
python test_evolutionary_search.py
```

The test suite includes:
- **Unit tests** for individual components
- **Integration tests** for complete workflows
- **Performance tests** with larger datasets
- **Error handling tests** for edge cases
- **Serialization tests** for data persistence

## Examples

### Simple Classification Task

```python
from sklearn.datasets import make_classification
from evolutionary_search import *

# Create classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Configure search
nas = EvolutionaryArchitectureSearch(
    architecture_config=ArchitectureConfig(max_layers=6),
    evolutionary_config=EvolutionaryConfig(population_size=30, max_generations=50),
    fitness_config=FitnessConfig(cv_folds=5),
    data=(X, y)
)

# Run search
best_architecture = nas.run_evolution()
print(f"Best architecture: {best_architecture}")
print(f"Fitness: {best_architecture.fitness:.4f}")
```

### Regression Task

```python
from sklearn.datasets import make_regression
from evolutionary_search import *

# Create regression dataset
X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)

# Configure for regression
fitness_config = FitnessConfig(
    primary_metric='neg_mean_squared_error',
    cv_folds=5
)

nas = EvolutionaryArchitectureSearch(
    architecture_config=ArchitectureConfig(max_layers=8),
    evolutionary_config=EvolutionaryConfig(population_size=40, max_generations=75),
    fitness_config=fitness_config,
    data=(X, y)
)

best_architecture = nas.run_evolution()
```

### Time Series Task

```python
# For time series data, you might want to include LSTM/GRU layers
arch_config = ArchitectureConfig(
    layer_types=['dense', 'lstm', 'gru', 'dropout', 'batch_norm'],
    max_layers=10,
    min_layers=3
)

# Configure for time series
fitness_config = FitnessConfig(
    primary_metric='neg_mean_absolute_error',
    cv_folds=3,  # Use time series CV
    max_training_epochs=150
)
```

## Advanced Features

### Custom Fitness Functions

```python
def custom_fitness_evaluator(architecture, X, y):
    """Custom fitness evaluation function."""
    # Your custom evaluation logic here
    # Return fitness score between 0 and 1
    return fitness_score

# Use custom evaluator
nas.fitness_evaluator = custom_fitness_evaluator
```

### Architecture Constraints

```python
# Define custom constraints
def custom_constraint(architecture):
    """Custom architecture constraint."""
    # Your constraint logic here
    return is_valid

# Apply custom constraint
nas.architecture_constraint = custom_constraint
```

### Parallel Evaluation

```python
# Configure parallel evaluation
evo_config = EvolutionaryConfig(
    use_parallel_evaluation=True,
    n_workers=8,  # Number of parallel workers
    batch_size=5   # Batch size for evaluation
)
```

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce population size or enable memory optimization
2. **Slow evaluation**: Use parallel evaluation or reduce CV folds
3. **Poor convergence**: Increase population size or adjust mutation rate
4. **Hardware issues**: Check M1 optimization settings

### Performance Tips

1. **Use parallel evaluation** for faster search
2. **Enable hardware optimization** for M1 systems
3. **Monitor memory usage** to prevent OOM
4. **Use early stopping** to save time
5. **Save intermediate results** for long searches

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by evolutionary algorithms and neural architecture search research
- Built with modern Python practices and hardware optimization
- Integrates with comprehensive utility modules for enhanced functionality