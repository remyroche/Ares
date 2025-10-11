# VectorBT Integration for Feature Selection

This module provides VectorBT-enhanced feature selection capabilities for financial data analysis and trading system optimization.

## 🚀 Features

### 1. Enhanced Financial Feature Importance (`VectorBTImportanceAnalyzer`)
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic Oscillators
- **Risk Metrics**: Sharpe ratio, Sortino ratio, Maximum Drawdown, VaR
- **Performance Metrics**: Cumulative returns, rolling returns, volatility measures
- **Market Microstructure**: Bid-ask spread analysis, order flow metrics

### 2. Time Series Aware Feature Selection (`VectorBTDirectionalSelector`)
- **Regime Detection**: Market regime identification for adaptive feature selection
- **Temporal Pattern Recognition**: VectorBT's built-in time series analysis
- **Rolling Window Analysis**: Dynamic feature importance over time
- **Cross-Asset Correlation**: Multi-asset feature relationships

### 3. Advanced Correlation Analysis (`VectorBTCorrelationAnalyzer`)
- **Multiple Correlation Methods**: Pearson, Spearman, Kendall
- **Rolling Correlations**: Time-varying correlation analysis
- **Lagged Correlations**: Cross-lag feature relationships
- **Financial Correlations**: Returns, volatility, volume correlations
- **Correlation Clustering**: Group highly correlated features

## 📖 Usage Examples

### Basic Financial Feature Importance Analysis

```python
from src.feature_selection.vectorbt import VectorBTImportanceAnalyzer, VectorBTImportanceConfig

# Configure analyzer
config = VectorBTImportanceConfig(
    include_technical_indicators=True,
    include_risk_metrics=True,
    include_performance_metrics=True,
    rsi_period=14,
    macd_fast=12,
    macd_slow=26
)

# Create analyzer
analyzer = VectorBTImportanceAnalyzer(config)

# Analyze financial importance
result = analyzer.analyze_financial_importance(
    prices=price_data,  # Price data (numpy array or DataFrame)
    returns=returns_data,  # Returns data
    feature_names=feature_names  # Optional feature names
)

# Access results
print(f"Market regime: {result.market_regime}")
print(f"Technical indicators: {result.technical_indicators}")
print(f"Risk metrics: {result.risk_metrics}")
print(f"Combined scores: {result.combined_scores}")
```

### Time Series Aware Feature Selection

```python
from src.feature_selection.vectorbt import VectorBTDirectionalSelector, VectorBTDirectionalConfig

# Configure selector
config = VectorBTDirectionalConfig(
    enable_regime_detection=True,
    enable_temporal_analysis=True,
    enable_cross_asset=True,
    regime_window=50,
    max_features_per_regime=30
)

# Create selector
selector = VectorBTDirectionalSelector(config)

# Select features with time series awareness
result = selector.select_features(
    features=feature_data,  # Feature matrix
    prices=price_data,      # Price data for regime detection
    returns=returns_data,   # Returns data
    feature_names=feature_names
)

# Access results
print(f"Selected features: {result.selected_features}")
print(f"Regime: {result.regime_info.regime_type}")
print(f"Confidence: {result.regime_info.confidence}")
print(f"Temporal features: {len(result.temporal_features)}")
```

### Advanced Correlation Analysis

```python
from src.feature_selection.vectorbt import VectorBTCorrelationAnalyzer, VectorBTCorrelationConfig

# Configure analyzer
config = VectorBTCorrelationConfig(
    enable_pearson=True,
    enable_spearman=True,
    enable_rolling_correlation=True,
    enable_lagged_correlation=True,
    enable_correlation_clustering=True,
    rolling_window=30,
    correlation_threshold=0.8
)

# Create analyzer
analyzer = VectorBTCorrelationAnalyzer(config)

# Analyze correlations
result = analyzer.analyze_correlations(
    data=feature_data,
    feature_names=feature_names
)

# Access results
print(f"Correlation matrix shape: {result.correlation_matrix.shape}")
print(f"Correlation clusters: {result.correlation_clusters}")
print(f"High correlation pairs: {len(analyzer.get_highly_correlated_pairs(result.correlation_matrix))}")

# Get correlation summary
summary = analyzer.get_correlation_summary(result)
print(f"Summary: {summary}")
```

## 🔧 Integration with Existing Feature Selection

### Enhanced Feature Selection Pipeline

```python
from src.feature_selection import select_features
from src.feature_selection.vectorbt import (
    VectorBTImportanceAnalyzer,
    VectorBTDirectionalSelector,
    VectorBTCorrelationAnalyzer
)

def enhanced_feature_selection_pipeline(X, y, prices, returns, feature_names):
    """Enhanced feature selection using VectorBT capabilities."""
    
    # Step 1: Advanced correlation analysis
    corr_analyzer = VectorBTCorrelationAnalyzer()
    corr_result = corr_analyzer.analyze_correlations(X, feature_names)
    
    # Remove highly correlated features
    high_corr_pairs = corr_analyzer.get_highly_correlated_pairs(
        corr_result.correlation_matrix, threshold=0.9
    )
    features_to_remove = set()
    for pair in high_corr_pairs:
        features_to_remove.add(pair[1])  # Remove second feature in each pair
    
    filtered_features = [f for f in feature_names if f not in features_to_remove]
    X_filtered = X[:, [feature_names.index(f) for f in filtered_features]]
    
    # Step 2: Financial importance analysis
    importance_analyzer = VectorBTImportanceAnalyzer()
    importance_result = importance_analyzer.analyze_financial_importance(
        prices, returns, filtered_features
    )
    
    # Step 3: Time series aware selection
    directional_selector = VectorBTDirectionalSelector()
    directional_result = directional_selector.select_features(
        X_filtered, prices, returns, filtered_features
    )
    
    # Step 4: Combine results
    final_features = directional_result.selected_features
    
    return {
        'selected_features': final_features,
        'regime_info': directional_result.regime_info,
        'financial_importance': importance_result.combined_scores,
        'correlation_analysis': corr_result,
        'temporal_analysis': directional_result.temporal_features
    }
```

### Integration with Core Framework

```python
from src.feature_selection import get_feature_selection_framework
from src.feature_selection.vectorbt import VectorBTImportanceAnalyzer

# Get core framework
framework = get_feature_selection_framework()

# Add VectorBT capabilities
def vectorbt_enhanced_selection(X, y, method='comprehensive', **kwargs):
    """Enhanced selection with VectorBT analysis."""
    
    # Run standard selection
    result = framework.select_features(X, y, method=method, **kwargs)
    
    # Add VectorBT financial analysis
    if 'prices' in kwargs and 'returns' in kwargs:
        analyzer = VectorBTImportanceAnalyzer()
        financial_result = analyzer.analyze_financial_importance(
            kwargs['prices'], kwargs['returns'], result.get('selected_features', [])
        )
        
        # Enhance result with financial metrics
        result['financial_analysis'] = financial_result
        result['market_regime'] = financial_result.market_regime
        result['risk_metrics'] = financial_result.risk_metrics
    
    return result
```

## ⚙️ Configuration Options

### VectorBTImportanceConfig
- `include_technical_indicators`: Enable technical indicator analysis
- `include_risk_metrics`: Enable risk metric calculation
- `include_performance_metrics`: Enable performance metric analysis
- `rsi_period`, `macd_fast`, `macd_slow`: Technical indicator parameters
- `sharpe_lookback`, `var_confidence`: Risk metric parameters

### VectorBTDirectionalConfig
- `enable_regime_detection`: Enable market regime detection
- `enable_temporal_analysis`: Enable temporal feature analysis
- `enable_cross_asset`: Enable cross-asset correlation analysis
- `regime_window`: Window size for regime detection
- `max_features_per_regime`: Maximum features per regime

### VectorBTCorrelationConfig
- `enable_pearson`, `enable_spearman`, `enable_kendall`: Correlation methods
- `enable_rolling_correlation`: Enable rolling correlation analysis
- `enable_lagged_correlation`: Enable lagged correlation analysis
- `rolling_window`: Window size for rolling correlations
- `correlation_threshold`: Threshold for clustering

## 🔍 Performance Monitoring

All VectorBT components include performance monitoring:

```python
# Get performance statistics
stats = analyzer.get_performance_stats()
print(f"Analyses performed: {stats['analyses_performed']}")
print(f"Average time: {stats['avg_time_per_analysis']:.3f}s")
```

## 🚨 Error Handling

VectorBT components include robust error handling with fallback implementations:

- **VectorBT Unavailable**: Automatic fallback to standard implementations
- **Data Issues**: Graceful handling of insufficient data
- **Calculation Errors**: Fallback methods for failed calculations
- **Performance Issues**: Automatic optimization based on data size

## 📊 Output Formats

### FinancialImportanceResult
- `technical_indicators`: Technical indicator values and trends
- `risk_metrics`: Risk metric calculations
- `performance_metrics`: Performance metric analysis
- `combined_scores`: Weighted combination of all metrics
- `market_regime`: Detected market regime

### VectorBTDirectionalResult
- `selected_features`: Features selected for current regime
- `regime_info`: Detailed regime information
- `temporal_features`: Temporal analysis results
- `cross_asset_features`: Cross-asset correlation features
- `feature_scores`: Feature importance scores

### CorrelationResult
- `correlation_matrix`: Full correlation matrix
- `rolling_correlations`: Time-varying correlations
- `lagged_correlations`: Cross-lag correlations
- `correlation_clusters`: Grouped highly correlated features
- `correlation_strength`: Feature correlation strength

## 🔗 Dependencies

- **VectorBT**: Advanced financial analysis library
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **SciPy**: Statistical functions (fallback)

## 📈 Future Enhancements

- **Portfolio Optimization**: Feature selection for portfolio-level metrics
- **Backtesting Integration**: Historical validation of feature selection
- **Real-time Analysis**: Live market data integration
- **Advanced Regime Detection**: Machine learning-based regime detection
- **Multi-Asset Analysis**: Cross-asset feature relationships

---

**Version**: 1.0.0  
**Last Updated**: October 2025  
**Maintainers**: Ares Team