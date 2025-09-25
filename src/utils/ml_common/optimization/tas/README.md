# Advanced Tree Architecture Search (TAS) System

A comprehensive system for tree-based architecture search with advanced capabilities including meta-learning, hardware optimization, uncertainty estimation, regime analysis, and real-time adaptation.

## 🚀 Features

### Core Capabilities
- **Tree Architecture Search**: Advanced search strategies for tree-based models
- **Meta-Learning**: MAML, prototypical networks, and few-shot learning
- **Hardware Optimization**: CPU, GPU, and M1-specific optimizations
- **Uncertainty Estimation**: Ensemble methods, Bayesian uncertainty, confidence scoring
- **Regime Analysis**: Regime detection, regime-aware optimization, transition analysis
- **Real-time Adaptation**: Dynamic architecture adaptation, performance monitoring
- **Multi-objective Optimization**: Pareto optimization, NSGA-II, SPEA2
- **Continual Learning**: Episodic memory, catastrophic forgetting prevention

### Advanced Search Strategies
- **Evolutionary Algorithms**: Genetic algorithms, NSGA-II, SPEA2
- **Bayesian Optimization**: Gaussian processes, TPE, acquisition functions
- **Reinforcement Learning**: PPO, A2C, DQN for architecture search
- **Hybrid Methods**: Combined strategies for optimal performance

### Optimization Features
- **Hardware Acceleration**: Matrix operations, parallel processing
- **Memory Optimization**: Caching, memory pools, efficient data structures
- **Distributed Search**: Multi-processing, distributed computing
- **Real-time Performance**: Low-latency adaptation, performance monitoring

## 📁 Directory Structure

```
tas/
├── core/                    # Core TAS components
│   ├── tas_engine.py        # Main TAS engine
│   ├── tas_config.py        # Configuration classes
│   ├── tas_result.py        # Result classes
│   ├── tree_architecture.py # Tree architecture classes
│   └── search_space.py      # Search space definitions
├── meta_learning/           # Meta-learning capabilities
│   ├── tree_meta_learning.py    # MAML and prototypical networks
│   ├── few_shot_learning.py    # Few-shot learning
│   └── continual_learning.py   # Continual learning
├── search/                  # Search strategies
│   ├── evolutionary_search.py   # Evolutionary algorithms
│   ├── (moved to src/utils/nas_tas/bayesian_search.py)      # Bayesian optimization
│   ├── rl_search.py           # Reinforcement learning
│   └── multi_objective_search.py # Multi-objective optimization
├── optimization/            # Hardware and performance optimization
│   ├── hardware_optimization.py # Hardware acceleration
│   ├── memory_optimization.py   # Memory optimization
│   └── parallel_optimization.py # Parallel processing
├── uncertainty/             # Uncertainty estimation
│   ├── uncertainty_estimation.py # Uncertainty quantification
│   ├── confidence_scoring.py    # Confidence scoring
│   └── robustness_analysis.py   # Robustness analysis
├── regime_analysis/         # Regime analysis
│   ├── tree_regime_analyzer.py  # Regime detection
│   ├── regime_optimization.py   # Regime-aware optimization
│   └── regime_reporting.py      # Regime reporting
├── adaptation/              # Real-time adaptation
│   ├── real_time_adaptation.py  # Real-time adaptation
│   ├── dynamic_optimization.py  # Dynamic optimization
│   └── performance_tracking.py  # Performance tracking
├── evaluation/              # Evaluation capabilities
│   ├── tree_evaluator.py        # Tree evaluation
│   ├── multi_objective_evaluation.py # Multi-objective evaluation
│   └── regime_evaluation.py     # Regime evaluation
└── utils/                   # Utilities
    ├── tree_utils.py            # Tree utilities
    ├── visualization.py         # Visualization tools
    └── logging.py              # Logging utilities
```

## 🎯 Quick Start

### Basic Usage

```python
from src.utils.ml_common.optimization.tas import TreeArchitectureSearchEngine, TASEngineConfig

# Create TAS engine
config = TASEngineConfig(
    search_strategy=SearchStrategy.BAYESIAN,
    optimization_mode=OptimizationMode.SINGLE_OBJECTIVE,
    enable_meta_learning=True,
    enable_hardware_optimization=True
)

engine = TreeArchitectureSearchEngine(config)

# Perform search
result = engine.search(
    train_data=(X_train, y_train),
    validation_data=(X_val, y_val),
    test_data=(X_test, y_test)
)

print(f"Best architecture: {result.best_architecture}")
print(f"Best score: {result.best_score:.4f}")
```

### Advanced Usage with Meta-Learning

```python
from src.utils.ml_common.optimization.tas import TreeArchitectureSearchEngine, TASEngineConfig, SearchStrategy, OptimizationMode

# Create advanced TAS engine
config = TASEngineConfig(
    search_strategy=SearchStrategy.HYBRID,
    optimization_mode=OptimizationMode.REGIME_AWARE,
    enable_meta_learning=True,
    enable_hardware_optimization=True,
    enable_uncertainty_estimation=True,
    enable_regime_analysis=True,
    enable_real_time_adaptation=True
)

engine = TreeArchitectureSearchEngine(config)

# Perform regime-aware search
result = engine.search(
    train_data=(X_train, y_train),
    validation_data=(X_val, y_val),
    regime_data=regime_info
)

# Adapt to new data
new_architecture = engine.adapt_to_new_data(
    new_data=(X_new, y_new),
    current_architecture=result.best_architecture
)
```

### Meta-Learning for Few-Shot Adaptation

```python
from src.utils.ml_common.optimization.tas.meta_learning import TreeMetaLearning, MetaLearningConfig

# Configure meta-learning
config = MetaLearningConfig(
    meta_learning_rate=0.001,
    num_inner_steps=5,
    num_outer_steps=100,
    num_shots=5
)

meta_learner = TreeMetaLearning(config)

# Meta-train on multiple tasks
meta_train_tasks = [task1, task2, task3, ...]
meta_val_tasks = [val_task1, val_task2, ...]

meta_results = meta_learner.meta_train(meta_train_tasks, meta_val_tasks)

# Few-shot adaptation to new task
adaptation_results = meta_learner.few_shot_adaptation(
    support_data=(X_support, y_support),
    query_data=(X_query, y_query),
    adaptation_method="maml"
)
```

## 🔧 Configuration

### TAS Engine Configuration

```python
from src.utils.nas_tas.tas.tas_config import TASEngineConfig, SearchStrategy, OptimizationMode

config = TASEngineConfig(
    # Search strategy
    search_strategy=SearchStrategy.HYBRID,
    optimization_mode=OptimizationMode.REGIME_AWARE,
    
    # Advanced features
    enable_meta_learning=True,
    enable_hardware_optimization=True,
    enable_uncertainty_estimation=True,
    enable_regime_analysis=True,
    enable_real_time_adaptation=True,
    enable_continual_learning=True,
    
    # Performance settings
    max_search_time=3600,  # 1 hour
    max_evaluations=1000,
    parallel_evaluations=4,
    memory_limit_gb=8.0,
    
    # Output settings
    save_results=True,
    save_models=True,
    output_dir="tas_results",
    verbose=True
)
```

### Search Strategy Configuration

```python
from src.utils.nas_tas.tas.tas_config import TASSearchConfig, SearchMethod

search_config = TASSearchConfig(
    search_strategy=SearchMethod.BAYESIAN,
    search_budget=100,
    search_time_limit=3600,
    
    # Bayesian optimization
    bayesian_acquisition_function="expected_improvement",
    bayesian_n_initial_points=10,
    bayesian_alpha=1e-6,
    
    # Evolutionary algorithm
    evolutionary_population_size=50,
    evolutionary_generations=100,
    evolutionary_mutation_rate=0.1,
    evolutionary_crossover_rate=0.8,
    
    # Performance
    parallel_evaluations=4,
    memory_limit_gb=8.0,
    cache_evaluations=True
)
```

## 📊 Advanced Features

### Regime-Aware Search

```python
# Enable regime analysis
config.enable_regime_analysis = True

# Perform regime-aware search
result = engine.search(
    train_data=(X_train, y_train),
    validation_data=(X_val, y_val),
    regime_data={
        'regime_labels': regime_labels,
        'regime_characteristics': regime_chars,
        'regime_transitions': regime_transitions
    }
)

# Access regime-specific results
regime_analysis = result.regime_analysis
regime_architectures = result.regime_specific_architectures
```

### Uncertainty Estimation

```python
# Enable uncertainty estimation
config.enable_uncertainty_estimation = True

# Perform search with uncertainty
result = engine.search(train_data, validation_data)

# Access uncertainty estimates
uncertainty = result.uncertainty_estimates
confidence = uncertainty['confidence_score']
reliability = uncertainty['reliability_score']
```

### Real-time Adaptation

```python
# Enable real-time adaptation
config.enable_real_time_adaptation = True

# Perform search
result = engine.search(train_data, validation_data)

# Adapt to new data in real-time
new_architecture = engine.adapt_to_new_data(
    new_data=(X_new, y_new),
    current_architecture=result.best_architecture
)
```

## 🎨 Visualization

### Search Progress Visualization

```python
from src.utils.ml_common.optimization.tas.utils import TreeVisualizer

visualizer = TreeVisualizer()

# Visualize search progress
visualizer.plot_search_progress(result.search_history)

# Visualize architecture comparison
visualizer.plot_architecture_comparison([arch1, arch2, arch3])

# Visualize regime analysis
visualizer.plot_regime_analysis(result.regime_analysis)
```

### Performance Analytics

```python
from src.utils.ml_common.optimization.tas.adaptation import TreePerformanceTracker

tracker = TreePerformanceTracker()

# Track performance metrics
tracker.track_architecture_performance(architecture, performance_metrics)

# Generate performance report
report = tracker.generate_performance_report()

# Visualize performance trends
tracker.plot_performance_trends()
```

## 🔬 Research and Development

### Custom Search Strategies

```python
from src.utils.ml_common.optimization.tas.search import EvolutionaryTreeSearch

class CustomTreeSearch(EvolutionaryTreeSearch):
    def __init__(self, config):
        super().__init__(config)
        # Add custom search logic
    
    def custom_mutation(self, architecture):
        # Implement custom mutation
        pass
    
    def custom_crossover(self, parent1, parent2):
        # Implement custom crossover
        pass
```

### Custom Evaluation Metrics

```python
from src.utils.ml_common.optimization.tas.evaluation import TreeEvaluator

class CustomTreeEvaluator(TreeEvaluator):
    def __init__(self, config):
        super().__init__(config)
        # Add custom evaluation logic
    
    def custom_evaluation_metric(self, architecture, data):
        # Implement custom metric
        pass
```

## 📈 Performance Optimization

### Hardware Optimization

```python
from src.utils.ml_common.optimization.tas.optimization import TreeHardwareOptimizer

# Enable hardware optimization
config.enable_hardware_optimization = True

# Configure for specific hardware
hardware_optimizer = TreeHardwareOptimizer(
    target_device="m1",  # "cpu", "gpu", "m1", "auto"
    memory_optimization=True,
    parallel_processing=True
)
```

### Memory Optimization

```python
from src.utils.ml_common.optimization.tas.optimization import TreeMemoryOptimizer

# Configure memory optimization
memory_optimizer = TreeMemoryOptimizer(
    memory_limit_gb=8.0,
    cache_size=1000,
    enable_memory_pool=True
)
```

## 🧪 Testing and Validation

### Unit Testing

```python
import unittest
from src.utils.ml_common.optimization.tas import TreeArchitectureSearchEngine

class TestTAS(unittest.TestCase):
    def setUp(self):
        self.config = TASEngineConfig()
        self.engine = TreeArchitectureSearchEngine(self.config)
    
    def test_basic_search(self):
        # Test basic search functionality
        result = self.engine.search(train_data, validation_data)
        self.assertIsNotNone(result.best_architecture)
        self.assertGreater(result.best_score, 0.0)
    
    def test_meta_learning(self):
        # Test meta-learning functionality
        pass
```

### Integration Testing

```python
def test_end_to_end_workflow():
    """Test complete TAS workflow."""
    # Setup
    config = TASEngineConfig(enable_meta_learning=True)
    engine = TreeArchitectureSearchEngine(config)
    
    # Search
    result = engine.search(train_data, validation_data)
    
    # Adaptation
    new_architecture = engine.adapt_to_new_data(new_data, result.best_architecture)
    
    # Validation
    assert new_architecture is not None
    assert new_architecture.overall_score > 0.0
```

## 📚 Documentation

### API Reference

- [Core Components](docs/core.md)
- [Meta-Learning](docs/meta_learning.md)
- [Search Strategies](docs/search.md)
- [Optimization](docs/optimization.md)
- [Uncertainty Estimation](docs/uncertainty.md)
- [Regime Analysis](docs/regime_analysis.md)
- [Adaptation](docs/adaptation.md)
- [Evaluation](docs/evaluation.md)
- [Utilities](docs/utils.md)

### Examples

- [Basic Usage](examples/basic_usage.py)
- [Advanced Features](examples/advanced_features.py)
- [Meta-Learning](examples/meta_learning.py)
- [Regime Analysis](examples/regime_analysis.py)
- [Real-time Adaptation](examples/real_time_adaptation.py)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by Neural Architecture Search (NAS) research
- Built on scikit-learn and other open-source libraries
- Contributions from the machine learning community

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the development team
- Check the documentation and examples

---

**Advanced Tree Architecture Search (TAS) System** - Bringing the power of neural architecture search to tree-based models with advanced meta-learning, hardware optimization, and real-time adaptation capabilities.