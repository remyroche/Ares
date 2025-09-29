# Enhanced Tree Architecture Search (TAS)

A sophisticated tree-based machine learning optimization system that matches the sophistication of Neural Architecture Search (NAS) while staying within non-neural-network methods.

## 🌟 Features

### Modern Tree Algorithms
- **XGBoost**: Extreme Gradient Boosting with advanced regularization
- **LightGBM**: Light Gradient Boosting Machine with categorical feature support
- **CatBoost**: Categorical Boosting with built-in categorical handling
- **Random Forest**: Ensemble of decision trees with bootstrap aggregation
- **Extra Trees**: Extremely Randomized Trees for additional diversity
- **BART**: Bayesian Additive Regression Trees for uncertainty estimation

### Automated Optimization
- **AutoML Integration**: Automated model selection and hyperparameter tuning
- **Evolutionary Algorithms**: NSGA-II and SPEA2 for multi-objective optimization
- **Bayesian Optimization**: Efficient hyperparameter search using Gaussian processes
- **Genetic Programming**: Tree structure optimization using evolutionary methods

### Advanced Feature Engineering
- **Technical Indicators**: RSI, MACD, Bollinger Bands, and more
- **Price Features**: OHLCV transformations and technical patterns
- **Volume Features**: Volume-based indicators and momentum
- **Volatility Features**: GARCH models and volatility clustering
- **Momentum Features**: Rate of change and momentum indicators
- **Trend Features**: Moving averages and trend detection
- **Regime Features**: Market regime identification and adaptation
- **Interaction Features**: Feature combinations and polynomial terms
- **Cross-timeframe Features**: Multi-timeframe analysis

### Sophisticated Evaluation
- **Risk-adjusted Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Drawdown Analysis**: Maximum drawdown and recovery metrics
- **Hit Rate**: Success rate of predictions
- **Payoff Ratio**: Average win to average loss ratio
- **Regime-aware Evaluation**: Performance across different market conditions
- **Economic Significance**: Real-world trading viability assessment
- **Multi-objective Optimization**: Balancing accuracy, robustness, and efficiency

## 🚀 Quick Start

### Basic Usage

```python
from src.utils.ml_common.optimization.tas.enhanced_tas_engine import (
    EnhancedTASEngine, EnhancedTASConfig, quick_enhanced_tas_search
)
import numpy as np

# Create sample data
X_train = np.random.rand(1000, 20)
y_train = np.random.randint(0, 2, 1000)
X_val = np.random.rand(200, 20)
y_val = np.random.randint(0, 2, 200)

# Quick search with default settings
result = quick_enhanced_tas_search(
    X_train, y_train, X_val, y_val,
    model_types=["xgboost", "lightgbm", "catboost"],
    max_search_time=3600
)

print(f"Best model: {result.best_model}")
print(f"Best score: {result.best_score:.4f}")
```

### Advanced Configuration

```python
from src.utils.ml_common.optimization.tas.enhanced_tas_engine import (
    EnhancedTASEngine, EnhancedTASConfig
)

# Create custom configuration
config = EnhancedTASConfig(
    model_types=["xgboost", "lightgbm", "catboost", "random_forest"],
    enable_automl=True,
    enable_evolutionary_search=True,
    enable_advanced_metrics=True,
    enable_feature_engineering=True,
    enable_ensemble=True,
    max_search_time=7200,  # 2 hours
    max_evaluations=1000,
    parallel_evaluations=8
)

# Create engine and run search
engine = EnhancedTASEngine(config)
result = engine.search(X_train, y_train, X_val, y_val, X_test, y_test)
```

## 📊 Architecture

### Core Components

```
Enhanced TAS Engine
├── Enhanced Tree Models
│   ├── XGBoost Integration
│   ├── LightGBM Integration
│   ├── CatBoost Integration
│   ├── Random Forest
│   ├── Extra Trees
│   └── BART (Bayesian)
├── AutoML Framework
│   ├── Optuna Integration
│   ├── Grid Search
│   ├── Random Search
│   └── Bayesian Optimization
├── Evolutionary Search
│   ├── NSGA-II
│   ├── SPEA2
│   └── Genetic Programming
├── Advanced Evaluation
│   ├── Risk-adjusted Metrics
│   ├── Regime Analysis
│   ├── Economic Significance
│   └── Multi-objective Scoring
└── Feature Engineering
    ├── Technical Indicators
    ├── Feature Selection
    ├── Dimensionality Reduction
    └── Cross-timeframe Analysis
```

### Search Strategies

1. **Single-objective Optimization**: Focus on accuracy or specific metric
2. **Multi-objective Optimization**: Balance multiple objectives simultaneously
3. **Regime-aware Optimization**: Adapt to different market conditions
4. **Real-time Optimization**: Continuous adaptation to new data
5. **Continual Learning**: Incremental model updates

## 🔧 Configuration

### Model Configuration

```python
from src.utils.ml_common.optimization.tas.models.enhanced_tree_models import (
    TreeModelConfig, TreeModelType
)

# XGBoost configuration
xgb_config = TreeModelConfig(
    model_type=TreeModelType.XGBOOST,
    params={
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    },
    is_classifier=True
)

# LightGBM configuration
lgb_config = TreeModelConfig(
    model_type=TreeModelType.LIGHTGBM,
    params={
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'feature_fraction': 0.8
    },
    is_classifier=True
)
```

### AutoML Configuration

```python
from src.utils.ml_common.optimization.tas.automl.tree_automl import (
    AutoMLConfig, AutoMLMethod
)

config = AutoMLConfig(
    optimization_method=AutoMLMethod.OPTUNA,
    max_trials=200,
    timeout_seconds=3600,
    model_types=["xgboost", "lightgbm", "catboost"],
    enable_ensemble=True,
    ensemble_method="voting"
)
```

### Evolutionary Search Configuration

```python
from src.utils.ml_common.optimization.shared_utils.evolutionary_search import (
    EvolutionaryConfig, EvolutionaryAlgorithm
)

config = EvolutionaryConfig(
    population_size=100,
    max_generations=50,
    use_nsga2=True,
    use_spea2=True,
    crossover_probability=0.8,
    mutation_probability=0.1
)
```

## 📈 Evaluation Metrics

### Financial Metrics

- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return
- **Calmar Ratio**: Return to maximum drawdown ratio
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Hit Rate**: Percentage of successful predictions
- **Payoff Ratio**: Average win to average loss ratio

### Statistical Metrics

- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity to positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

### Regime-aware Metrics

- **Regime-specific Performance**: Performance across different market conditions
- **Regime Transition Analysis**: Model behavior during regime changes
- **Economic Significance**: Real-world trading viability
- **Trading Viability**: Practical implementation considerations

## 🎯 Multi-objective Optimization

### Objectives

1. **Accuracy**: Prediction accuracy and performance
2. **Robustness**: Stability across different conditions
3. **Efficiency**: Computational efficiency and speed
4. **Interpretability**: Model explainability and transparency

### Optimization Methods

- **NSGA-II**: Non-dominated Sorting Genetic Algorithm II
- **SPEA2**: Strength Pareto Evolutionary Algorithm 2
- **Weighted Sum**: Linear combination of objectives
- **Pareto Optimization**: Find Pareto-optimal solutions

## 🔍 Feature Engineering

### Technical Indicators

```python
from src.utils.ml_common.optimization.tas.data_pipeline.feature_engineering import (
    FeatureEngineer, FeatureConfig
)

# Create feature engineer
config = FeatureConfig(
    technical_indicators=True,
    price_features=True,
    volume_features=True,
    volatility_features=True,
    momentum_features=True,
    trend_features=True,
    regime_features=True,
    interaction_features=True,
    polynomial_features=True
)

engineer = FeatureEngineer(config)
X_enhanced = engineer.generate_features(X)
```

### Feature Selection

```python
from src.utils.ml_common.feature_selection import (
    FeatureSelectionFramework, FeatureSelectionMethod
)

# Create feature selector
selector = FeatureSelectionFramework(
    method=FeatureSelectionMethod.MUTUAL_INFO,
    max_features=100,
    importance_threshold=0.01
)

X_selected = selector.select_features(X, y)
```

## 📊 Results and Analysis

### Enhanced TAS Result

```python
result = engine.search(X_train, y_train, X_val, y_val, X_test, y_test)

# Access results
print(f"Best model: {result.best_model}")
print(f"Best score: {result.best_score:.4f}")
print(f"Search time: {result.total_search_time:.2f}s")
print(f"Total evaluations: {result.total_evaluations}")

# Model rankings
for model_type, score in result.model_rankings:
    print(f"{model_type}: {score:.4f}")

# Feature importance
for feature, importance in result.feature_importance.items():
    print(f"{feature}: {importance:.4f}")

# Multi-objective results
for solution in result.pareto_front:
    print(f"Objectives: {solution.objectives}")
```

### Visualization

```python
# Create visualizations
create_visualization(result)

# Plot search progress
plt.plot(result.search_history)
plt.title('Search Progress')
plt.xlabel('Generation')
plt.ylabel('Best Score')
plt.show()
```

## 🚀 Advanced Usage

### Custom Objective Functions

```python
def custom_objective(params):
    """Custom objective function for optimization."""
    # Train model with parameters
    model = create_model(params)
    model.fit(X_train, y_train)
    
    # Evaluate model
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    
    # Add custom metrics
    robustness = calculate_robustness(model, X_val, y_val)
    efficiency = calculate_efficiency(model)
    
    # Return multi-objective score
    return [accuracy, robustness, efficiency]
```

### Regime-aware Optimization

```python
# Create regime-aware configuration
config = EnhancedTASConfig(
    enable_regime_analysis=True,
    regime_adaptation=True,
    regime_specific_optimization=True
)

# Run regime-aware search
result = engine.search(
    X_train, y_train, X_val, y_val, X_test, y_test,
    regime_labels=regime_labels
)
```

### Ensemble Methods

```python
# Create ensemble configuration
config = EnhancedTASConfig(
    enable_ensemble=True,
    ensemble_method="stacking",
    ensemble_models=["xgboost", "lightgbm", "catboost"]
)

# Run ensemble search
result = engine.search(X_train, y_train, X_val, y_val, X_test, y_test)
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce `max_evaluations` or `parallel_evaluations`
3. **Timeout Issues**: Increase `max_search_time` or reduce search complexity
4. **Convergence Issues**: Adjust evolutionary algorithm parameters

### Performance Optimization

1. **Parallel Processing**: Use `parallel_evaluations` for faster search
2. **Early Stopping**: Enable `early_stopping` to avoid overfitting
3. **Feature Selection**: Reduce feature space for faster evaluation
4. **Model Caching**: Cache trained models for reuse

## 📚 Examples

See the `examples/` directory for comprehensive examples:

- `enhanced_tas_example.py`: Complete demonstration
- `automl_example.py`: AutoML usage
- `evolutionary_example.py`: Evolutionary search
- `evaluation_example.py`: Advanced evaluation
- `feature_engineering_example.py`: Feature engineering

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- XGBoost team for the excellent gradient boosting library
- LightGBM team for the efficient gradient boosting framework
- CatBoost team for the categorical boosting library
- Scikit-learn team for the machine learning foundation
- Optuna team for the hyperparameter optimization framework