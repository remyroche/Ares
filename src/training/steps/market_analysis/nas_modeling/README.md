# Neural Architecture Search (NAS) for Market Analysis

This directory contains a comprehensive Neural Architecture Search (NAS) implementation specifically designed for financial market analysis, regime detection, and HMM state modeling.

## 🚀 Overview

The NAS system provides true neural architecture search capabilities for:
- **Market Regime Detection**: Finding optimal architectures for identifying market states
- **HMM State Modeling**: Optimizing architectures for Hidden Markov Model state learning
- **Time Series Analysis**: Architectures for temporal market data analysis
- **Feature Learning**: Neural architectures for automatic feature extraction

## 🏗️ Architecture

### Core Components

```
nas_modeling/
├── core/                          # Core NAS functionality
│   ├── nas_search.py             # Main architecture search engine
│   ├── nas_model.py              # Neural network models
│   ├── nas_trainer.py            # Model training utilities
│   └── nas_evaluator.py          # Architecture evaluation
├── search/                       # Search strategies
│   ├── search_space.py           # Architecture search space
│   ├── random_search.py          # Random search strategy
│   ├── bayesian_search.py        # Bayesian optimization
│   └── evolutionary_search.py    # Evolutionary algorithms
├── evaluation/                   # Evaluation metrics
│   ├── nas_metrics.py            # Comprehensive NAS metrics
│   ├── regime_metrics.py         # Regime-specific metrics
│   └── hmm_metrics.py            # HMM-specific metrics
├── applications/                 # Domain-specific applications
│   ├── hmm_nas.py                # NAS for HMM optimization
│   └── regime_nas.py             # NAS for regime detection
└── utils/                        # Utilities
    ├── nas_utils.py              # NAS helper functions
    └── logging_utils.py          # Logging utilities
```

## 🎯 Key Features

### Neural Architecture Search
- **True NAS Implementation**: Actual neural architecture search, not clustering
- **Multiple Search Strategies**: Random search, Bayesian optimization, evolutionary algorithms
- **Flexible Search Space**: Configurable architecture components and constraints
- **Hardware Acceleration**: GPU support with mixed precision training

### Market-Specific Optimizations
- **Regime Detection**: Specialized architectures for market state identification
- **HMM Integration**: NAS-optimized architectures for Hidden Markov Models
- **Time Series Support**: LSTM, GRU, and attention-based temporal models
- **Financial Metrics**: Domain-specific evaluation metrics

### Comprehensive Evaluation
- **Multi-dimensional Metrics**: Performance, complexity, efficiency, generalization
- **Architecture Comparison**: Systematic comparison of different architectures
- **Complexity Analysis**: Parameter counting, FLOP estimation, memory usage
- **Stability Metrics**: Training stability and convergence analysis

## 🔧 Quick Start

### Basic NAS Search

```python
from nas_modeling.core.nas_search import NASArchitectureSearch, NASSearchConfig
from nas_modeling.core.nas_trainer import NASTrainer, TrainingConfig
from nas_modeling.core.nas_evaluator import NASEvaluator, EvaluationConfig

# Configure search
search_config = NASSearchConfig(
    max_iterations=100,
    search_strategy="random",
    primary_metric="accuracy",
    use_gpu=True
)

# Create NAS search engine
nas_search = NASArchitectureSearch(search_config)

# Perform search
result = nas_search.search(
    train_data=(X_train, y_train),
    validation_data=(X_val, y_val),
    problem_type="classification"
)

print(f"Best architecture: {result.best_architecture.name}")
print(f"Best score: {result.best_score:.4f}")
```

### HMM-Specific NAS

```python
from nas_modeling.applications.hmm_nas import HMM_NAS_Optimizer

# Create HMM NAS optimizer
hmm_nas = HMM_NAS_Optimizer(config)

# Optimize HMM architecture
best_hmm_arch = hmm_nas.optimize_hmm_architecture(
    market_data, n_states=5, n_iterations=50
)
```

### Regime Detection NAS

```python
from nas_modeling.applications.regime_nas import Regime_NAS_Detector

# Create regime detector
regime_nas = Regime_NAS_Detector(config)

# Find optimal regime detection architecture
best_regime_arch = regime_nas.find_optimal_regime_detector(
    market_features, regime_labels, n_regimes=10
)
```

## 🔍 Search Strategies

### Random Search
- **Description**: Randomly samples architectures from search space
- **Best for**: Initial exploration, baseline comparison
- **Speed**: Fast, scales well with search space size
- **Use case**: Quick prototyping, large search spaces

### Bayesian Optimization
- **Description**: Uses Gaussian processes to model performance landscape
- **Best for**: Expensive evaluations, small search spaces
- **Speed**: Slower but more intelligent
- **Use case**: Fine-tuning, limited computational budget

### Evolutionary Algorithms
- **Description**: Uses genetic operators (crossover, mutation, selection)
- **Best for**: Complex search spaces, multi-objective optimization
- **Speed**: Moderate, good for parallel evaluation
- **Use case**: Complex architectures, when diversity is important

## 📊 Evaluation Metrics

### Performance Metrics
- **Accuracy**: Classification accuracy or R² for regression
- **Precision/Recall/F1**: Standard classification metrics
- **Loss**: Training/validation loss

### Complexity Metrics
- **Parameters**: Number of trainable parameters
- **FLOPs**: Floating-point operations
- **Memory Usage**: Model memory requirements
- **Complexity Score**: Composite complexity measure

### Efficiency Metrics
- **Training Time**: Time per epoch
- **Inference Time**: Forward pass time
- **Throughput**: Samples per second
- **Parameters per Second**: Training efficiency

### Generalization Metrics
- **Overfitting Gap**: Train vs validation performance
- **Consistency Score**: Performance stability
- **Cross-validation Scores**: Robustness measures

## 🎨 Architecture Search Space

### Layer Types
- **Dense**: Fully connected layers
- **Convolution**: 1D/2D convolutions for time series
- **LSTM/GRU**: Recurrent layers for temporal data
- **Attention**: Multi-head attention mechanisms
- **BatchNorm/Dropout**: Regularization layers

### Activation Functions
- **ReLU**: Rectified Linear Unit
- **Tanh**: Hyperbolic tangent
- **Sigmoid**: Logistic function
- **Leaky ReLU**: Leaky Rectified Linear Unit
- **ELU**: Exponential Linear Unit
- **GELU**: Gaussian Error Linear Unit
- **Swish**: Self-gated activation

### Architecture Patterns
- **MLP**: Multi-layer perceptrons
- **CNN**: Convolutional neural networks
- **RNN**: Recurrent neural networks
- **Transformer**: Attention-based architectures
- **Hybrid**: Combined architectures

## 🔬 Problem Types

### Classification
- **Use case**: Market regime classification
- **Output**: Class probabilities
- **Loss**: Cross-entropy
- **Metrics**: Accuracy, F1-score

### Regression
- **Use case**: Price prediction, volatility forecasting
- **Output**: Continuous values
- **Loss**: Mean squared error
- **Metrics**: R², MAE, RMSE

### HMM State Modeling
- **Use case**: Hidden Markov Model state learning
- **Output**: State probabilities and transitions
- **Loss**: Negative log-likelihood
- **Metrics**: State accuracy, transition consistency

### Regime Detection
- **Use case**: Market regime identification
- **Output**: Regime probabilities
- **Loss**: Cross-entropy with regime labels
- **Metrics**: Regime accuracy, stability

### Time Series
- **Use case**: Temporal pattern recognition
- **Output**: Sequence predictions
- **Loss**: Sequence loss functions
- **Metrics**: Sequence accuracy, temporal consistency

## 🛠️ Configuration

### NAS Search Configuration
```python
search_config = NASSearchConfig(
    max_iterations=100,
    search_strategy="random",
    primary_metric="accuracy",
    minimize_metric=False,
    use_gpu=True,
    batch_size=32,
    validation_split=0.2
)
```

### Training Configuration
```python
train_config = TrainingConfig(
    epochs=100,
    batch_size=64,
    learning_rate=0.001,
    optimizer="adam",
    loss_function="cross_entropy",
    early_stopping_patience=10
)
```

### Evaluation Configuration
```python
eval_config = EvaluationConfig(
    batch_size=32,
    compute_confusion_matrix=True,
    compute_per_class_metrics=True,
    compute_complexity_metrics=True
)
```

## 📈 Usage Examples

### Complete NAS Pipeline
```python
# 1. Setup data
X_train, y_train, X_val, y_val = load_market_data()

# 2. Configure NAS
config = NASSearchConfig(max_iterations=50, search_strategy="bayesian")
nas = NASArchitectureSearch(config)

# 3. Perform search
result = nas.search(
    train_data=(X_train, y_train),
    validation_data=(X_val, y_val),
    problem_type="regime_detection"
)

# 4. Train best architecture
best_model = NASModel.create_from_config(result.best_architecture, "regime_detection")
trainer = NASTrainer(TrainingConfig(epochs=50))
trained_model = trainer.train(best_model, train_dataset, val_dataset)

# 5. Evaluate
evaluator = NASEvaluator(EvaluationConfig())
metrics = evaluator.evaluate_architecture(
    trained_model.model, test_dataset, "regime_detection"
)
```

### HMM Architecture Optimization
```python
from nas_modeling.applications.hmm_nas import HMM_NAS_Optimizer

hmm_nas = HMM_NAS_Optimizer()
optimal_hmm = hmm_nas.optimize_hmm_architecture(
    market_data, n_states=6, n_iterations=30
)
```

### Custom Search Space
```python
from nas_modeling.search.search_space import SearchSpace

search_space = SearchSpace()
custom_arch = search_space.generate_random_architecture(
    input_dim=50, output_dim=4, problem_type="classification"
)
```

## 🎯 Performance Tips

### Hardware Optimization
- **GPU Usage**: Enable `use_gpu=True` for faster training
- **Mixed Precision**: Set `mixed_precision=True` for memory efficiency
- **Batch Size**: Tune batch size based on available memory
- **Parallel Workers**: Increase `num_workers` for data loading

### Search Efficiency
- **Start Simple**: Begin with random search for exploration
- **Bayesian for Fine-tuning**: Use Bayesian optimization for refinement
- **Early Stopping**: Enable early stopping to save time
- **Warm Starting**: Use pre-trained models when possible

### Memory Management
- **Gradient Checkpointing**: For very deep architectures
- **Model Pruning**: Remove unnecessary parameters
- **Batch Size Adjustment**: Reduce if memory issues occur
- **Gradient Accumulation**: For large effective batch sizes

## 🔧 Integration with Market Analysis

### Pipeline Integration
The NAS system integrates seamlessly with the existing market analysis pipeline:

1. **Data Preparation**: Use existing feature extraction
2. **Architecture Search**: Find optimal models for specific tasks
3. **Model Training**: Train best architectures on full datasets
4. **Evaluation**: Comprehensive evaluation with financial metrics
5. **Deployment**: Deploy optimized models for live trading

### HMM Integration
- **State Learning**: NAS-optimized HMM state detection
- **Transition Modeling**: Learn optimal state transition patterns
- **Regime Classification**: Classify market regimes using NAS models
- **Sequence Prediction**: Predict future market states

### Feature Engineering
- **Automatic Feature Learning**: NAS models can learn features automatically
- **Feature Selection**: Identify most important features for regime detection
- **Feature Interaction**: Discover complex feature relationships
- **Dimensionality Reduction**: Learn compact representations

## 📋 Requirements

### Dependencies
- **PyTorch**: >= 1.9.0 (for neural networks)
- **NumPy**: >= 1.21.0 (for numerical operations)
- **SciPy**: >= 1.7.0 (for optimization)
- **Scikit-learn**: >= 1.0.0 (for metrics)
- **CUDA**: Optional (for GPU acceleration)

### Hardware Requirements
- **Minimum**: CPU with 4GB RAM
- **Recommended**: GPU with 8GB+ VRAM
- **Optimal**: Multi-GPU setup for parallel search

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```python
# Reduce batch size
config = NASSearchConfig(batch_size=16)

# Enable gradient checkpointing
config = TrainingConfig(gradient_checkpointing=True)
```

**Slow Search**
```python
# Use random search for faster exploration
config = NASSearchConfig(search_strategy="random")

# Reduce iterations
config = NASSearchConfig(max_iterations=50)
```

**Poor Performance**
```python
# Increase model capacity
config = SearchSpace(hidden_dims=[128, 64, 32])

# Use more sophisticated search
config = NASSearchConfig(search_strategy="bayesian")
```

**Overfitting**
```python
# Add regularization
config = TrainingConfig(weight_decay=1e-4)

# Use early stopping
config = TrainingConfig(early_stopping_patience=10)
```

## 📚 Advanced Usage

### Custom Search Strategy
```python
class CustomSearch:
    def generate_architecture(self, iteration):
        # Implement custom search logic
        pass

# Register custom strategy
search_strategies = {'custom': CustomSearch()}
```

### Multi-Objective Optimization
```python
# Optimize for multiple metrics
metrics = NASMetrics()
comparison = metrics.compare_architectures(
    architecture_list,
    primary_metric="accuracy",
    secondary_metrics=["efficiency", "complexity"]
)
```

### Ensemble NAS
```python
# Combine multiple search strategies
results = []
for strategy in ['random', 'bayesian', 'evolutionary']:
    config = NASSearchConfig(search_strategy=strategy)
    nas = NASArchitectureSearch(config)
    result = nas.search(train_data, val_data)
    results.append(result)

# Select best overall
best_result = max(results, key=lambda x: x.best_score)
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **New Search Strategies**: Implement additional search algorithms
2. **Architecture Types**: Add support for more architecture patterns
3. **Evaluation Metrics**: Enhance evaluation capabilities
4. **Hardware Support**: Add support for new accelerators
5. **Documentation**: Improve examples and tutorials

## 📄 License

This module is part of the Ares trading system and follows the same licensing terms.

## 📞 Support

For support and questions, please refer to the main project documentation or create an issue in the project repository.