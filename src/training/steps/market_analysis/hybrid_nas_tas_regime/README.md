# Hybrid NAS-TAS Regime System

## Overview

The **Hybrid NAS-TAS Regime System** replaces the traditional HMM clustering approach with a sophisticated hybrid system that combines:

- **Neural Architecture Search (NAS)** from the `nas_regime/` system
- **Tree Architecture Search (TAS)** from the ML Common TAS system
- **Economic and Financial Relevance Evaluation** for meaningful regime interpretation
- **Advanced Regime Tagging** for existing data processing

This system provides a complete replacement for HMM-based clustering with enhanced economic significance and trading viability.

## Key Features

### 🔄 Hybrid Architecture
- **TAS Integration**: Tree-based feature extraction and clustering
- **NAS Integration**: Neural network-based pattern recognition
- **Adaptive Fusion**: Intelligent combination of TAS and NAS inputs
- **Economic Priority**: Focus on economically significant regimes

### 🔍 Economic Clustering
- **Economic Significance Integration**: Directly incorporates economic factors into clustering
- **Momentum Analysis**: Multi-timeframe momentum detection (5, 10, 20, 50 periods)
- **Volume Analysis**: Volume-price correlation and volume trend analysis
- **Advanced Economic Metrics**: Volatility, trend, efficiency, liquidity analysis
- **Economic Distance Metrics**: Specialized distance calculations for economic clustering
- **Economic-Aware Algorithms**: K-means, Hierarchical, GMM, and Adaptive clustering with economic awareness

### 🔍 Advanced Clustering with Economic Integration
- **Economic Distance Metrics**: Specialized distance calculations incorporating economic significance
- **Momentum-Weighted Clustering**: Clustering that prioritizes momentum-driven market moves
- **Volume-Adjusted Algorithms**: Clustering algorithms that account for volume patterns
- **Economic Frontier Analysis**: 4D frontier establishment between economically significant clusters
- **Regime Transfer Optimization**: CV similarity and size constraints with economic weighting
- **Multi-Algorithm Ensemble**: Combines economic K-means, hierarchical, and GMM clustering
- **Adaptive Economic Selection**: Automatically chooses best algorithm based on economic performance

### 📊 Economic Significance (Enhanced with Momentum & Volume)
- **Volatility Regime Analysis**: Identifies distinct volatility patterns with momentum/volume factors
- **Trend Strength Evaluation**: Measures regime trend characteristics with multi-timeframe analysis
- **Market Efficiency Assessment**: Evaluates information efficiency and autocorrelation patterns
- **Liquidity Regime Detection**: Identifies liquidity-based regimes with spread analysis
- **Momentum Regime Analysis**: Detects momentum patterns across multiple timeframes (5, 10, 20, 50 periods)
- **Volume-Momentum Analysis**: Analyzes volume-price correlations and volume trends
- **Price Action Analysis**: Evaluates candlestick patterns and price behavior
- **Market Microstructure Analysis**: Examines bid-ask spreads, order flow, and market impact
- **Inter-Market Analysis**: Studies cross-market relationships and lead-lag patterns
- **Sector Rotation Analysis**: Identifies sector rotation and market regime changes

### 💰 Financial Relevance
- **Trading Viability**: Ensures regimes are tradable
- **Risk-Return Profiles**: Evaluates regime risk characteristics
- **Sharpe Ratio Analysis**: Measures risk-adjusted returns
- **Drawdown Assessment**: Evaluates maximum drawdown per regime

### 🏷️ Regime Tagging
- **Historical Data Tagging**: Tags existing datasets with regime information
- **Unified Processing**: Creates single datasets with regime labels
- **Backward Compatibility**: Maintains compatibility with existing pipelines
- **Batch Processing**: Efficient processing of large datasets

## Architecture

```
Hybrid NAS-TAS Regime System
├── Core Components
│   ├── Hybrid Regime Detector      # Main detection engine with economic clustering
│   ├── Economic Clusterer         # Economic-aware clustering algorithms
│   ├── Coherent Regime Modeler    # Advanced economic/financial regime analysis
│   ├── TAS Integration             # Tree-based feature extraction with momentum
│   ├── NAS Integration             # Neural feature extraction with volume analysis
│   └── Economic Evaluator          # Enhanced economic significance analysis
├── Configuration System
│   ├── Hybrid Regime Config        # Main configuration with economic clustering
│   ├── Economic Evaluation Config  # Economic analysis settings with momentum/volume
│   └── Financial Relevance Config  # Financial analysis settings
├── Integration Layer
│   └── Hybrid Orchestrator         # Main orchestrator (replaces HMM)
└── Utilities
    ├── Regime Tagger               # Data tagging system
    └── Performance Monitor         # Execution tracking
```

## Usage

### Basic Regime Detection

```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime import (
    HybridNASTASRegimeDetector,
    HybridRegimeConfig
)

# Create configuration
config = HybridRegimeConfig(n_regimes=8)
detector = HybridNASTASRegimeDetector(config)

# Detect regimes
result = detector.detect_regimes(market_data)
print(f"Detected {len(set(result.regime_predictions))} regimes")
```

### Advanced Orchestrator Usage

```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime import HybridRegimeOrchestrator

# Create orchestrator
orchestrator = HybridRegimeOrchestrator()

# Complete regime detection with analysis
results = orchestrator.detect_regimes(
    market_data,
    symbol="ETHUSDT",
    exchange="binance",
    timeframe="1h"
)

# Access regime data
regime_predictions = results['regime_data']['predictions']
economic_scores = results['economic_analysis']['significance_scores']
financial_scores = results['financial_analysis']['relevance_scores']
```

#### Economic Clustering with Momentum and Volume

```python
from src.training.steps.market_analysis.hybrid_nas_tas_regime import (
    HybridNASTASRegimeDetector,
    HybridRegimeConfig,
    ClusteringAlgorithm,
    EconomicSignificanceType
)

# Create configuration for economic clustering
config = HybridRegimeConfig(
    n_regimes=4,
    clustering_config={
        "primary_algorithm": ClusteringAlgorithm.ECONOMIC_ADAPTIVE,
        "economic_clustering": True,
        "momentum_integration": True,
        "volume_integration": True,
        "momentum_threshold": 0.7,
        "volume_threshold": 0.6
    },
    economic_evaluation={
        "enabled": True,
        "significance_types": [
            EconomicSignificanceType.MOMENTUM_REGIME.value,
            EconomicSignificanceType.VOLUME_MOMENTUM.value,
            EconomicSignificanceType.VOLATILITY_REGIME.value
        ],
        "momentum_threshold": 0.7,
        "volume_threshold": 0.6,
        "momentum_periods": [5, 10, 20, 50]
    }
)

# Create detector with economic clustering
detector = HybridNASTASRegimeDetector(config)

# Detect regimes with economic clustering
result = detector.detect_regimes(market_data)

# Access economic clustering results
if result.success:
    print(f"Economic significance: {result.economic_significance_scores}")
    print(f"Momentum scores: {result.momentum_scores}")
    print(f"Volume profiles: {result.volume_profiles}")
    print(f"Economic clustering used: {result.metadata.get('economic_clustering_used', False)}")
```

### Data Tagging

```python
# Tag existing data with regime information
tagging_results = orchestrator.tag_existing_data(
    data_directory="/path/to/historical/data",
    output_directory="/path/to/tagged/data"
)

# Create regime-aware dataset
regime_dataset = orchestrator.create_regime_aware_dataset(
    market_data, results, split_by_regime=False
)
```

## Configuration

### Hybrid Regime Configuration

```python
config = HybridRegimeConfig(
    n_regimes=8,
    combination_strategy=RegimeCombinationStrategy.ADAPTIVE_FUSION,
    economic_evaluation={
        'enabled': True,
        'significance_types': ['volatility_regime', 'trend_strength'],
        'min_significance_score': 0.7
    },
    financial_relevance={
        'enabled': True,
        'sharpe_ratio_threshold': 0.5,
        'max_drawdown_threshold': 0.15
    }
)
```

### Economic-Focused Configuration

```python
config = create_economic_focused_config()
# Focuses on economic significance with higher NAS weight
```

### Trading-Focused Configuration

```python
config = create_trading_focused_config()
# Optimizes for trading viability and backtesting
```

## Integration with Existing Systems

### Replacing HMM Clustering

The hybrid system fully replaces HMM clustering functionality:

```python
# OLD: HMM-based approach
from src.training.steps.market_analysis.hmm_clustering import OptimalRegimeClusteringOrchestrator
hmm_orchestrator = OptimalRegimeClusteringOrchestrator()

# NEW: Hybrid NAS-TAS approach
from src.training.steps.market_analysis.hybrid_nas_tas_regime import HybridRegimeOrchestrator
hybrid_orchestrator = HybridRegimeOrchestrator()

# Same interface, enhanced functionality
results = hybrid_orchestrator.detect_regimes(market_data, symbol, exchange, timeframe)
```

### Regime-Aware Training Integration

```python
# Create tagged dataset for training
tagged_data = orchestrator.create_regime_aware_dataset(market_data, results)

# Use in downstream training steps
for regime_id in tagged_data['regime_id'].unique():
    regime_data = tagged_data[tagged_data['regime_id'] == regime_id]
    # Train regime-specific models
```

## Performance Characteristics

### Advantages over HMM Clustering

1. **Economic Relevance**: Regimes have clear economic interpretation with momentum/volume analysis
2. **Trading Viability**: Identified regimes are actually tradable with proven economic significance
3. **Economic Clustering**: Directly integrates economic factors into the clustering process itself
4. **Momentum Integration**: Analyzes momentum patterns across multiple timeframes (5, 10, 20, 50 periods)
5. **Volume Analysis**: Incorporates volume-price correlations and volume trends into regime detection
6. **Economic Distance Metrics**: Specialized distance calculations for economically meaningful clustering
7. **Multi-Algorithm Ensemble**: Combines economic K-means, hierarchical, and GMM clustering
8. **Adaptive Economic Selection**: Automatically chooses best algorithm based on economic performance
9. **Advanced Economic Metrics**: Comprehensive evaluation including microstructure and inter-market analysis
10. **Unified Processing**: Single dataset approach with tagging eliminates file management complexity

### Computational Complexity

- **Training Time**: Moderate (TAS + NAS feature extraction)
- **Inference Time**: Fast (optimized clustering algorithms)
- **Memory Usage**: Efficient (streaming and batch processing)
- **Scalability**: Handles large datasets with batch processing

## Validation and Testing

### Economic Significance Validation

- **Volatility Regimes**: Verified through statistical tests
- **Trend Strength**: Measured with R-squared analysis
- **Market Efficiency**: Assessed via autocorrelation tests
- **Liquidity Patterns**: Validated with spread analysis

### Financial Relevance Testing

- **Sharpe Ratio**: Risk-adjusted return analysis
- **Maximum Drawdown**: Risk management evaluation
- **Win Rate**: Trading success measurement
- **Profit Factor**: Overall profitability assessment

## Monitoring and Reporting

### Performance Tracking

```python
# Access performance history
summary = orchestrator.get_regime_summary()
print(f"Average economic significance: {summary['avg_economic_significance']}")
print(f"Average financial relevance: {summary['avg_financial_relevance']}")
```

### Comprehensive Reporting

- **JSON Reports**: Detailed regime analysis
- **CSV Exports**: Regime predictions and metadata
- **Performance Metrics**: Execution time and quality scores
- **Economic Analysis**: Significance scores and interpretations

## Migration Guide

### From HMM Clustering to Hybrid System

1. **Replace Imports**:
   ```python
   # Old
   from src.training.steps.market_analysis.hmm_clustering import OptimalRegimeClusteringOrchestrator

   # New
   from src.training.steps.market_analysis.hybrid_nas_tas_regime import HybridRegimeOrchestrator
   ```

2. **Update Configuration**:
   ```python
   # Use hybrid configuration instead of HMM config
   config = HybridRegimeConfig(n_regimes=8)
   orchestrator = HybridRegimeOrchestrator(config)
   ```

3. **Adapt Data Processing**:
   ```python
   # Results now include economic and financial analysis
   results = orchestrator.detect_regimes(market_data)
   economic_scores = results['economic_analysis']['significance_scores']
   financial_scores = results['financial_analysis']['relevance_scores']
   ```

## Future Enhancements

### Planned Features

- **Meta-Learning**: Adaptive regime detection
- **Multi-Timeframe**: Cross-timeframe regime analysis
- **Real-time Processing**: Streaming regime detection
- **Advanced Visualization**: Interactive regime analysis
- **Custom Evaluation Metrics**: Domain-specific significance measures

### Extension Points

- **Custom Economic Evaluators**: Plugin economic significance functions
- **Alternative Clustering Algorithms**: Support for specialized clustering
- **Enhanced Feature Engineering**: Custom feature extraction methods
- **Multi-Modal Integration**: Combine with other data sources

## Support and Maintenance

### Troubleshooting

- **Low Economic Significance**: Check data quality and market conditions
- **Poor Financial Relevance**: Review trading cost assumptions
- **Clustering Failures**: Verify feature extraction and data preprocessing
- **Performance Issues**: Monitor memory usage and batch sizes

### Configuration Tuning

- **Significance Thresholds**: Adjust based on market characteristics
- **Feature Weights**: Balance TAS vs NAS contributions
- **Regime Count**: Tune based on market complexity
- **Validation Settings**: Configure cross-validation parameters

---

The Hybrid NAS-TAS Regime System represents a significant advancement over traditional HMM clustering, providing economically meaningful and financially relevant regime detection with enhanced practical applicability.