# Shared Utilities Integration Summary

## 🎯 **Complete Integration of Shared Utilities for NAS and TAS**

This document summarizes the complete integration of shared utilities that are now used by both Neural Architecture Search (NAS) and Tree-based Advanced Statistics (TAS) systems.

## 📊 **System Architecture**

```
Shared Utilities (NAS/TAS Common)
├── Evolutionary Search
│   ├── NSGA-II Optimizer
│   ├── SPEA2 Optimizer
│   ├── Genetic Algorithm
│   └── Evolutionary Algorithm Manager
├── Feature Engineering
│   ├── Technical Indicators
│   ├── Feature Selection
│   ├── Dimensionality Reduction
│   └── Unified Feature Engineer
└── Advanced Metrics
    ├── Risk Metrics
    ├── Trading Metrics
    ├── Regime Metrics
    ├── Economic Metrics
    ├── Model Metrics
    └── Advanced Evaluator

NAS Systems
├── Neural Architecture Search
├── Multi-objective Optimization
├── Shared Utilities Integration
└── Advanced Evaluation

TAS Systems
├── Tree Architecture Search
├── Enhanced Tree Models
├── AutoML Integration
├── Shared Utilities Integration
└── Advanced Evaluation
```

## 🔧 **Shared Utilities Components**

### **1. Evolutionary Search** (`shared_utils/evolutionary_search.py`)

#### **Core Algorithms**
- **NSGA-II**: Non-dominated Sorting Genetic Algorithm II for multi-objective optimization
- **SPEA2**: Strength Pareto Evolutionary Algorithm 2 for multi-objective optimization
- **Genetic Algorithm**: Single-objective optimization with elitism
- **Evolutionary Algorithm Manager**: Coordinates different algorithms

#### **Key Features**
- **Multi-objective Optimization**: Pareto-optimal solutions
- **Diversity Preservation**: Crowding distance and archive management
- **Convergence Detection**: Early stopping and performance monitoring
- **Flexible Configuration**: Customizable parameters and strategies

#### **Usage by NAS**
```python
# NAS Evolutionary Search
from src.utils.ml_common.optimization.shared_utils.evolutionary_search import (
    EvolutionaryAlgorithmManager, EvolutionaryConfig, create_evolutionary_algorithm_manager
)

config = EvolutionaryConfig(
    population_size=100,
    max_generations=50,
    use_nsga2=True,  # Multi-objective for NAS
    use_spea2=True,
    use_genetic_algorithm=True
)
manager = create_evolutionary_algorithm_manager(config)

# Define NAS objective functions
def accuracy_objective(architecture_params):
    # Evaluate neural architecture accuracy
    return model_accuracy

def efficiency_objective(architecture_params):
    # Evaluate neural architecture efficiency
    return model_efficiency

def complexity_objective(architecture_params):
    # Evaluate neural architecture complexity
    return model_complexity

# Run evolutionary search
result = manager.optimize_with_algorithm(
    [accuracy_objective, efficiency_objective, complexity_objective],
    nas_parameter_space,
    "nsga2"
)
```

#### **Usage by TAS**
```python
# TAS Evolutionary Search
from src.utils.ml_common.optimization.shared_utils.evolutionary_search import (
    EvolutionaryAlgorithmManager, EvolutionaryConfig, create_evolutionary_algorithm_manager
)

config = EvolutionaryConfig(
    population_size=50,
    max_generations=100,
    use_nsga2=True,  # Multi-objective for TAS
    use_spea2=True,
    use_genetic_algorithm=True
)
manager = create_evolutionary_algorithm_manager(config)

# Define TAS objective functions
def accuracy_objective(tree_params):
    # Evaluate tree model accuracy
    return tree_accuracy

def robustness_objective(tree_params):
    # Evaluate tree model robustness
    return tree_robustness

def efficiency_objective(tree_params):
    # Evaluate tree model efficiency
    return tree_efficiency

# Run evolutionary search
result = manager.optimize_with_algorithm(
    [accuracy_objective, robustness_objective, efficiency_objective],
    tas_parameter_space,
    "nsga2"
)
```

### **2. Feature Engineering** (`shared_utils/feature_engineering.py`)

#### **Core Components**
- **Technical Indicators**: RSI, MACD, Bollinger Bands, SMA, EMA
- **Feature Selection**: Mutual Information, F-score, RFE
- **Dimensionality Reduction**: PCA, ICA, LDA
- **Unified Feature Engineer**: Comprehensive pipeline

#### **Key Features**
- **Technical Analysis**: Financial market indicators
- **Feature Selection**: Automated feature importance
- **Dimensionality Reduction**: Noise reduction and compression
- **Cross-timeframe Analysis**: Multi-timeframe features

#### **Usage by NAS**
```python
# NAS Feature Engineering
from src.utils.ml_common.optimization.shared_utils.feature_engineering import (
    UnifiedFeatureEngineer, FeatureConfig, create_unified_feature_engineer
)

config = FeatureConfig(
    enable_technical_indicators=True,
    enable_feature_selection=True,
    feature_selection_method="mutual_info",
    max_features=100,
    enable_dimensionality_reduction=True,
    reduction_method="pca",
    n_components=50
)
engineer = create_unified_feature_engineer(config)

# Engineer features for NAS
result = engineer.engineer_features(X_train, y_train, feature_names)
enhanced_X = result.enhanced_features
selected_features = result.selected_features
```

#### **Usage by TAS**
```python
# TAS Feature Engineering
from src.utils.ml_common.optimization.shared_utils.feature_engineering import (
    UnifiedFeatureEngineer, FeatureConfig, create_unified_feature_engineer
)

config = FeatureConfig(
    enable_technical_indicators=True,
    enable_feature_selection=True,
    feature_selection_method="f_score",
    max_features=200,
    enable_dimensionality_reduction=False
)
engineer = create_unified_feature_engineer(config)

# Engineer features for TAS
result = engineer.engineer_features(X_train, y_train, feature_names)
enhanced_X = result.enhanced_features
feature_importance = result.feature_importance
```

### **3. Advanced Metrics** (`shared_utils/advanced_metrics.py`)

#### **Core Metrics**
- **Risk Metrics**: Sharpe ratio, Sortino ratio, Max drawdown, VaR, CVaR
- **Trading Metrics**: Win rate, Profit factor, Consecutive wins/losses
- **Regime Metrics**: Regime accuracy, stability, transition analysis
- **Economic Metrics**: Economic significance, trading viability
- **Model Metrics**: Accuracy, Precision, Recall, F1-score, AUC

#### **Key Features**
- **Risk-adjusted Evaluation**: Financial performance assessment
- **Regime-aware Analysis**: Performance across market conditions
- **Economic Significance**: Real-world trading viability
- **Multi-objective Scoring**: Comprehensive model assessment

#### **Usage by NAS**
```python
# NAS Advanced Evaluation
from src.utils.ml_common.optimization.shared_utils.advanced_metrics import (
    AdvancedEvaluator, create_advanced_evaluator
)

evaluator = create_advanced_evaluator()

# Evaluate NAS model
result = evaluator.evaluate(
    predictions=nas_predictions,
    targets=y_test,
    returns=returns_data,
    regime_labels=regime_labels,
    model_complexity=model_complexity
)

# Access comprehensive metrics
print(f"Overall Score: {result.overall_score:.4f}")
print(f"Risk-adjusted Score: {result.risk_adjusted_score:.4f}")
print(f"Regime-aware Score: {result.regime_aware_score:.4f}")
print(f"Economic Score: {result.economic_score:.4f}")
print(f"Trading Score: {result.trading_score:.4f}")
```

#### **Usage by TAS**
```python
# TAS Advanced Evaluation
from src.utils.ml_common.optimization.shared_utils.advanced_metrics import (
    AdvancedEvaluator, create_advanced_evaluator
)

evaluator = create_advanced_evaluator()

# Evaluate TAS model
result = evaluator.evaluate(
    predictions=tas_predictions,
    targets=y_test,
    returns=returns_data,
    regime_labels=regime_labels,
    model_complexity=tree_complexity
)

# Access comprehensive metrics
print(f"Overall Score: {result.overall_score:.4f}")
print(f"Risk-adjusted Score: {result.risk_adjusted_score:.4f}")
print(f"Regime-aware Score: {result.regime_aware_score:.4f}")
print(f"Economic Score: {result.economic_score:.4f}")
print(f"Trading Score: {result.trading_score:.4f}")
```

## 🔗 **Integration Examples**

### **1. NAS with Shared Utilities**

#### **Complete NAS Pipeline**
```python
from src.training.steps.market_analysis.nas_regime.core.nas_shared_utils_integration import (
    NASSharedUtilsIntegration, NASSearchConfig
)

# Configure NAS with shared utilities
config = NASSearchConfig(
    population_size=100,
    max_generations=50,
    enable_feature_engineering=True,
    enable_advanced_metrics=True,
    objectives=["accuracy", "efficiency", "complexity", "regime_awareness"]
)

# Create NAS integration
nas_integration = NASSharedUtilsIntegration(config)

# Run architecture search
best_architectures = nas_integration.search_architectures(
    train_data=(X_train, y_train),
    validation_data=(X_val, y_val),
    test_data=(X_test, y_test)
)

# Get search statistics
stats = nas_integration.get_search_statistics()
print(f"Found {stats['total_architectures']} architectures")
print(f"Best fitness: {stats['best_fitness']:.4f}")
```

### **2. TAS with Shared Utilities**

#### **Complete TAS Pipeline**
```python
from src.utils.ml_common.optimization.tas.core.tas_engine import (
    TreeArchitectureSearchEngine, TASEngineConfig
)

# Configure TAS with shared utilities
config = TASEngineConfig(
    enable_enhanced_models=True,
    enable_automl=True,
    enable_evolutionary_search=True,
    enable_advanced_metrics=True,
    enable_feature_engineering=True,
    model_types=["xgboost", "lightgbm", "catboost"],
    evolutionary_algorithm="nsga2",
    population_size=50,
    max_generations=100
)

# Create TAS engine
engine = TreeArchitectureSearchEngine(config)

# Run tree architecture search
result = engine.search(
    train_data=(X_train, y_train),
    validation_data=(X_val, y_val),
    test_data=(X_test, y_test),
    regime_data={'regime_labels': regime_labels}
)

print(f"Best architecture: {result.best_architecture}")
print(f"Best score: {result.best_score:.4f}")
```

### **3. Direct Shared Utility Usage**

#### **Evolutionary Search**
```python
# Create objective functions
def objective1(params):
    return evaluate_model(params)

def objective2(params):
    return evaluate_efficiency(params)

# Run evolutionary search
from src.utils.ml_common.optimization.shared_utils.evolutionary_search import (
    create_evolutionary_algorithm_manager, EvolutionaryConfig
)

config = EvolutionaryConfig(population_size=50, max_generations=100)
manager = create_evolutionary_algorithm_manager(config)
result = manager.optimize_with_algorithm([objective1, objective2], parameter_space)
```

#### **Feature Engineering**
```python
# Engineer features
from src.utils.ml_common.optimization.shared_utils.feature_engineering import (
    create_unified_feature_engineer
)

engineer = create_unified_feature_engineer()
result = engineer.engineer_features(X, y, feature_names)
enhanced_X = result.enhanced_features
```

#### **Advanced Evaluation**
```python
# Evaluate models
from src.utils.ml_common.optimization.shared_utils.advanced_metrics import (
    create_advanced_evaluator
)

evaluator = create_advanced_evaluator()
result = evaluator.evaluate(predictions, targets, returns, regime_labels)
```

## 📊 **Performance Benefits**

### **1. Code Reusability**
- **Shared Components**: Common algorithms used by both NAS and TAS
- **Consistent Interface**: Unified API for all shared utilities
- **Maintenance Efficiency**: Single implementation to maintain

### **2. Performance Optimization**
- **Optimized Algorithms**: Efficient implementations of evolutionary algorithms
- **Parallel Processing**: Multi-threaded evaluation and optimization
- **Memory Management**: Efficient memory usage for large populations

### **3. Advanced Capabilities**
- **Multi-objective Optimization**: Pareto-optimal solutions
- **Regime-aware Evaluation**: Performance across market conditions
- **Economic Significance**: Real-world trading viability
- **Feature Engineering**: Automated feature selection and engineering

## 🎯 **Key Achievements**

### **✅ Shared Utilities Integration**
- **Evolutionary Search**: NSGA-II, SPEA2, GA for both NAS and TAS
- **Feature Engineering**: Technical indicators, selection, dimensionality reduction
- **Advanced Metrics**: Risk-adjusted, regime-aware, economic evaluation

### **✅ NAS Integration**
- **Neural Architecture Search**: Using shared evolutionary algorithms
- **Multi-objective Optimization**: Balancing accuracy, efficiency, complexity
- **Advanced Evaluation**: Comprehensive model assessment
- **Feature Engineering**: Enhanced input processing

### **✅ TAS Integration**
- **Tree Architecture Search**: Using shared evolutionary algorithms
- **Enhanced Tree Models**: XGBoost, LightGBM, CatBoost, BART
- **AutoML Integration**: Automated model selection and tuning
- **Advanced Evaluation**: Risk-adjusted and regime-aware metrics

### **✅ Unified Interface**
- **Consistent API**: Same interface for both NAS and TAS
- **Flexible Configuration**: Customizable parameters and strategies
- **Error Handling**: Robust error handling and fallback mechanisms
- **Documentation**: Comprehensive documentation and examples

## 🚀 **Usage Guidelines**

### **1. For NAS Systems**
```python
# Import shared utilities
from src.utils.ml_common.optimization.shared_utils.evolutionary_search import *
from src.utils.ml_common.optimization.shared_utils.feature_engineering import *
from src.utils.ml_common.optimization.shared_utils.advanced_metrics import *

# Configure for NAS
config = NASSearchConfig(
    enable_feature_engineering=True,
    enable_advanced_metrics=True,
    objectives=["accuracy", "efficiency", "complexity"]
)

# Use shared utilities
nas_integration = NASSharedUtilsIntegration(config)
```

### **2. For TAS Systems**
```python
# Import shared utilities
from src.utils.ml_common.optimization.shared_utils.evolutionary_search import *
from src.utils.ml_common.optimization.shared_utils.feature_engineering import *
from src.utils.ml_common.optimization.shared_utils.advanced_metrics import *

# Configure for TAS
config = TASEngineConfig(
    enable_enhanced_models=True,
    enable_automl=True,
    enable_evolutionary_search=True,
    enable_advanced_metrics=True,
    enable_feature_engineering=True
)

# Use shared utilities
engine = TreeArchitectureSearchEngine(config)
```

### **3. Direct Usage**
```python
# Direct shared utility usage
from src.utils.ml_common.optimization.shared_utils.evolutionary_search import (
    create_evolutionary_algorithm_manager
)
from src.utils.ml_common.optimization.shared_utils.feature_engineering import (
    create_unified_feature_engineer
)
from src.utils.ml_common.optimization.shared_utils.advanced_metrics import (
    create_advanced_evaluator
)

# Use directly
evolutionary_manager = create_evolutionary_algorithm_manager(config)
feature_engineer = create_unified_feature_engineer(config)
advanced_evaluator = create_advanced_evaluator()
```

## 🎉 **Conclusion**

The shared utilities integration provides:

1. **✅ Unified Interface**: Consistent API for both NAS and TAS systems
2. **✅ Code Reusability**: Common algorithms and utilities
3. **✅ Advanced Capabilities**: Multi-objective optimization, regime-aware evaluation
4. **✅ Performance Optimization**: Efficient implementations and parallel processing
5. **✅ Comprehensive Evaluation**: Risk-adjusted, economic, and trading metrics
6. **✅ Feature Engineering**: Automated feature selection and engineering
7. **✅ Evolutionary Search**: NSGA-II, SPEA2, and genetic algorithms
8. **✅ Documentation**: Complete documentation and examples

The shared utilities successfully enable both NAS and TAS systems to use the same advanced algorithms and evaluation metrics, providing a unified and powerful framework for neural and tree-based architecture search optimization.