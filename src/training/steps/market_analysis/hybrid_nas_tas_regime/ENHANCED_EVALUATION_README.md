# Enhanced NAS-TAS Evaluation and Clustering

This document describes the data-driven improvements implemented for the NAS-TAS evaluation and clustering system, making it more resilient and comprehensive.

## Overview

The enhanced system addresses the following key areas:

1. **Regime Evaluation Metrics** - Comprehensive risk-adjusted performance measures
2. **Feature Correlation Handling** - PCA and VIF-based feature selection
3. **Cross-Validation for Clustering** - Parameter optimization with out-of-sample validation
4. **Robust Scoring Models** - Machine learning-based regime quality prediction

## 1. Enhanced Regime Evaluation Metrics

### Location: `src/training/steps/market_analysis/hybrid_nas_tas_regime/evaluation/enhanced_regime_evaluator.py`

### Key Features:

#### Return and Volatility per Regime
- **Mean Return**: Average return for each regime
- **Volatility**: Standard deviation of returns
- **Risk-Adjusted Metrics**: Sharpe and Sortino ratios

#### Advanced Risk Metrics
- **Maximum Drawdown**: Largest peak-to-trough decline
- **VaR and CVaR**: Value at Risk and Conditional VaR at 95% confidence
- **Calmar Ratio**: Annual return divided by maximum drawdown
- **Information Ratio**: Excess return over benchmark divided by tracking error

#### Trading Performance Metrics
- **Hit Rate**: Percentage of profitable trades
- **Payoff Ratio**: Average gain to average loss ratio
- **Economic Significance**: Multi-factor economic importance score
- **Trading Viability**: Combined trading performance score

#### Distribution Analysis
- **Skewness**: Asymmetry of return distribution
- **Kurtosis**: Tail heaviness of return distribution
- **Stability Score**: Regime consistency measure

### Usage Example:

```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime.evaluation.enhanced_regime_evaluator import create_enhanced_regime_evaluator

# Configure evaluator
evaluator_config = {
    'risk_free_rate': 0.02,
    'confidence_level': 0.95,
    'min_regime_size': 10,
    'min_sharpe_threshold': 0.5,
    'min_sortino_threshold': 0.3,
    'max_drawdown_threshold': 0.2
}

# Create evaluator
evaluator = create_enhanced_regime_evaluator(evaluator_config)

# Evaluate regimes
result = evaluator.evaluate_regimes(market_data, regime_labels)

# Access comprehensive metrics
for metric in result.regime_metrics:
    print(f"Regime {metric.regime_id}:")
    print(f"  Sharpe Ratio: {metric.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio: {metric.sortino_ratio:.3f}")
    print(f"  Max Drawdown: {metric.max_drawdown:.3f}")
    print(f"  Hit Rate: {metric.hit_rate:.3f}")
    print(f"  Payoff Ratio: {metric.payoff_ratio:.3f}")
```

## 2. Feature Correlation Handling

### Location: `src/utils/feature_selection/pca_module.py` and `src/utils/feature_selection/vif_module.py`

### Key Features:

#### PCA Module
- **Dimensionality Reduction**: Transform correlated features into orthogonal components
- **Variance Threshold**: Retain components explaining specified variance
- **Correlation Filtering**: Remove highly correlated features before PCA
- **Feature Importance**: Extract feature importance from PCA loadings

#### VIF Module
- **Multicollinearity Detection**: Calculate Variance Inflation Factor for each feature
- **Stepwise Removal**: Iteratively remove features with high VIF
- **Correlation Analysis**: Identify and remove highly correlated feature pairs
- **Variance Filtering**: Remove low-variance features

### Usage Example:

```python
from src.utils.feature_selection import create_pca_module, create_vif_module

# Configure VIF module
vif_config = {
    'vif_threshold': 10.0,
    'correlation_threshold': 0.9,
    'enable_correlation_filtering': True,
    'stepwise_removal': True
}

# Apply VIF-based feature selection
vif_module = create_vif_module(vif_config)
vif_result = vif_module.apply_vif_feature_selection(features)

# Configure PCA module
pca_config = {
    'variance_threshold': 0.95,
    'correlation_threshold': 0.9,
    'enable_correlation_filtering': True
}

# Apply PCA for dimensionality reduction
pca_module = create_pca_module(pca_config)
pca_result = pca_module.apply_pca_feature_selection(features)
```

## 3. Cross-Validation for Clustering Parameters

### Location: `src/training/steps/market_analysis/hybrid_nas_tas_regime/evaluation/clustering_cross_validation.py`

### Key Features:

#### Parameter Optimization
- **Number of Regimes**: Optimize cluster count using cross-validation
- **Weight Optimization**: Find optimal economic significance, momentum, and volume weights
- **Algorithm Selection**: Choose best clustering algorithm (K-means, Hierarchical, GMM)

#### Validation Metrics
- **Silhouette Score**: Measure cluster separation and cohesion
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster dispersion
- **Davies-Bouldin Index**: Average similarity ratio of clusters
- **Stability Analysis**: Cross-fold consistency measures

#### Time Series Support
- **Time Series CV**: Respect temporal order in cross-validation
- **Out-of-Sample Validation**: Test on future data
- **Regime Transition Analysis**: Analyze regime change patterns

### Usage Example:

```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime.evaluation.clustering_cross_validation import create_clustering_cross_validator

# Configure cross-validation
cv_config = {
    'cv_folds': 5,
    'scoring_metric': 'silhouette',
    'enable_time_series_cv': True,
    'n_regimes_range': list(range(2, 11)),
    'weight_ranges': {
        'economic_significance_weight': np.arange(0.1, 0.6, 0.1),
        'momentum_weight': np.arange(0.1, 0.6, 0.1),
        'volume_weight': np.arange(0.1, 0.6, 0.1)
    }
}

# Optimize parameters
cv_validator = create_clustering_cross_validator(cv_config)
cv_result = cv_validator.optimize_clustering_parameters(features, market_data)

print(f"Best parameters: {cv_result.best_params}")
print(f"Best score: {cv_result.best_score:.3f}")
```

## 4. Robust Scoring Models

### Location: `src/training/steps/market_analysis/hybrid_nas_tas_regime/evaluation/robust_scoring_models.py`

### Key Features:

#### Model Types
- **Regression Models**: Random Forest, Gradient Boosting, Linear Regression, Ridge, Lasso, SVR, MLP
- **Classification Models**: Random Forest, SVM, MLP for regime quality classification
- **Ensemble Selection**: Choose best model based on cross-validation performance

#### Target Variables
- **Economic Significance**: Multi-factor economic importance
- **Trading Viability**: Combined trading performance score
- **Stability Score**: Regime consistency measure
- **Risk Score**: Risk assessment score
- **Performance Score**: Overall performance metric
- **Regime Quality Class**: High/Medium/Low quality classification

#### Feature Engineering
- **Market Features**: Returns, volatility, technical indicators
- **Volume Analysis**: Volume trends and patterns
- **Technical Indicators**: RSI, Bollinger Bands, moving averages
- **Regime Features**: Clustering-based feature extraction

### Usage Example:

```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime.evaluation.robust_scoring_models import create_robust_scoring_models

# Configure scoring models
scoring_config = {
    'test_size': 0.2,
    'cv_folds': 5,
    'enable_feature_scaling': True,
    'model_selection_strategy': 'ensemble'
}

# Train models
scoring_models = create_robust_scoring_models(scoring_config)
model_performances = scoring_models.train_scoring_models(
    historical_data, features, regime_labels, regime_metrics
)

# Predict regime scores
scoring_result = scoring_models.predict_regime_scores(
    regime_features, regime_market_data, regime_id
)

print(f"Economic Significance: {scoring_result.economic_significance:.3f}")
print(f"Trading Viability: {scoring_result.trading_viability:.3f}")
print(f"Stability Score: {scoring_result.stability_score:.3f}")
```

## 5. Integrated Enhanced Clustering System

### Location: `src/training/steps/market_analysis/hybrid_nas_tas_regime/core/enhanced_economic_clustering.py`

### Key Features:

#### Integrated Pipeline
1. **Feature Selection**: Apply VIF and PCA for correlation handling
2. **Parameter Optimization**: Use cross-validation to optimize clustering parameters
3. **Enhanced Clustering**: Apply clustering with optimized parameters
4. **Regime Evaluation**: Calculate comprehensive regime metrics
5. **Scoring Models**: Train and apply robust scoring models
6. **Economic Analysis**: Generate economic insights and rankings

#### Comprehensive Results
- **Clustering Results**: Labels, centers, probabilities
- **Regime Metrics**: Detailed performance metrics for each regime
- **Rankings**: Multiple ranking systems (performance, risk, economic, trading)
- **Feature Selection Info**: Details on feature selection process
- **Cross-Validation Results**: Parameter optimization results
- **Scoring Model Results**: Machine learning model predictions

### Usage Example:

```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime.core.enhanced_economic_clustering import create_enhanced_economic_clusterer

# Configure enhanced clustering
clustering_config = {
    'n_regimes': 4,
    'enable_feature_selection': True,
    'enable_cross_validation': True,
    'enable_scoring_models': True,
    'evaluator_config': {
        'risk_free_rate': 0.02,
        'min_regime_size': 10
    },
    'cv_config': {
        'cv_folds': 5,
        'scoring_metric': 'silhouette'
    },
    'scoring_config': {
        'test_size': 0.2,
        'enable_feature_scaling': True
    }
}

# Run enhanced clustering
clusterer = create_enhanced_economic_clusterer(clustering_config)
result = clusterer.cluster_with_enhanced_evaluation(
    features, market_data, historical_data
)

# Access comprehensive results
print(f"Number of clusters: {len(set(result.labels))}")
print(f"Overall quality score: {result.overall_quality_score:.3f}")
print(f"Algorithm used: {result.algorithm_used}")

# Access regime rankings
for ranking_type, rankings in result.regime_rankings.items():
    print(f"{ranking_type}: {rankings}")
```

## 6. Complete Example

### Location: `src/training/steps/market_analysis/hybrid_nas_tas_regime/examples/enhanced_clustering_example.py`

The complete example demonstrates all components working together:

```python
# Run the complete demonstration
python src/training/steps/market_analysis/hybrid_nas_tas_regime/examples/enhanced_clustering_example.py
```

## 7. Configuration Options

### Enhanced Regime Evaluator
```python
evaluator_config = {
    'risk_free_rate': 0.02,           # Annual risk-free rate
    'confidence_level': 0.95,          # VaR confidence level
    'min_regime_size': 10,             # Minimum regime size
    'min_sharpe_threshold': 0.5,       # Minimum Sharpe ratio
    'min_sortino_threshold': 0.3,      # Minimum Sortino ratio
    'max_drawdown_threshold': 0.2,     # Maximum drawdown threshold
    'min_hit_rate_threshold': 0.4,     # Minimum hit rate
    'min_payoff_ratio_threshold': 1.0  # Minimum payoff ratio
}
```

### Feature Selection
```python
pca_config = {
    'variance_threshold': 0.95,        # Variance explained threshold
    'correlation_threshold': 0.9,      # Correlation threshold
    'enable_correlation_filtering': True,
    'enable_variance_filtering': True
}

vif_config = {
    'vif_threshold': 10.0,             # VIF threshold
    'correlation_threshold': 0.9,      # Correlation threshold
    'stepwise_removal': True,          # Stepwise feature removal
    'standardize_features': True       # Standardize before VIF
}
```

### Cross-Validation
```python
cv_config = {
    'cv_folds': 5,                     # Number of CV folds
    'scoring_metric': 'silhouette',    # Scoring metric
    'enable_time_series_cv': True,     # Use time series CV
    'n_regimes_range': list(range(2, 11)),  # Range of regimes to test
    'weight_ranges': {                 # Weight optimization ranges
        'economic_significance_weight': np.arange(0.1, 0.6, 0.1),
        'momentum_weight': np.arange(0.1, 0.6, 0.1),
        'volume_weight': np.arange(0.1, 0.6, 0.1)
    }
}
```

### Robust Scoring Models
```python
scoring_config = {
    'test_size': 0.2,                  # Test set size
    'cv_folds': 5,                     # CV folds
    'enable_feature_scaling': True,    # Scale features
    'model_selection_strategy': 'ensemble'  # Model selection strategy
}
```

## 8. Benefits of Enhanced System

### Data-Driven Improvements
- **Comprehensive Metrics**: Sharpe, Sortino, max drawdown, hit rate, payoff ratio
- **Risk-Adjusted Performance**: Better assessment of regime quality
- **Feature Quality**: PCA and VIF ensure high-quality features
- **Parameter Optimization**: Cross-validation prevents overfitting
- **Machine Learning**: Robust scoring models replace fixed scores

### Resilience and Robustness
- **Correlation Handling**: Prevents multicollinearity issues
- **Out-of-Sample Validation**: Ensures generalizability
- **Multiple Algorithms**: Fallback options for clustering
- **Comprehensive Evaluation**: Multiple metrics for regime assessment
- **Adaptive Scoring**: Models learn from historical data

### Performance and Scalability
- **Efficient Feature Selection**: Reduces dimensionality while preserving information
- **Parallel Processing**: Cross-validation can be parallelized
- **Memory Optimization**: PCA reduces memory requirements
- **Configurable Thresholds**: Adjustable parameters for different use cases

## 9. Integration with Existing System

The enhanced components are designed to integrate seamlessly with the existing NAS-TAS system:

1. **Backward Compatibility**: Existing interfaces remain unchanged
2. **Optional Features**: All enhancements can be enabled/disabled via configuration
3. **Modular Design**: Components can be used independently
4. **Performance Monitoring**: Built-in logging and performance tracking
5. **Error Handling**: Robust error handling with fallback options

## 10. Future Enhancements

Potential future improvements:

1. **Deep Learning Models**: Neural networks for regime quality prediction
2. **Online Learning**: Incremental model updates with new data
3. **Multi-Asset Support**: Extend to multiple assets and markets
4. **Real-Time Processing**: Stream processing capabilities
5. **Advanced Feature Engineering**: More sophisticated feature extraction
6. **Ensemble Methods**: Combine multiple clustering algorithms
7. **Interpretability**: SHAP values and feature importance analysis

## Conclusion

The enhanced NAS-TAS evaluation and clustering system provides a comprehensive, data-driven approach to regime detection and analysis. By incorporating advanced metrics, feature selection, cross-validation, and machine learning models, the system is more resilient, accurate, and adaptable to different market conditions.

The modular design allows for easy integration and customization, while the comprehensive documentation ensures maintainability and extensibility for future enhancements.