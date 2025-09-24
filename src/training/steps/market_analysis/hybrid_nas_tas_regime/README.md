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

### 📊 Economic Significance
- **Volatility Regime Analysis**: Identifies distinct volatility patterns
- **Trend Strength Evaluation**: Measures regime trend characteristics
- **Market Efficiency Assessment**: Evaluates information efficiency
- **Liquidity Regime Detection**: Identifies liquidity-based regimes

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
│   ├── Hybrid Regime Detector      # Main detection engine
│   ├── TAS Integration             # Tree-based feature extraction
│   ├── NAS Integration             # Neural feature extraction
│   └── Economic Evaluator          # Economic significance analysis
├── Configuration System
│   ├── Hybrid Regime Config        # Main configuration
│   ├── Economic Evaluation Config  # Economic analysis settings
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

1. **Economic Relevance**: Regimes have clear economic interpretation
2. **Trading Viability**: Identified regimes are actually tradable
3. **Adaptive Learning**: System adapts to market conditions
4. **Robust Evaluation**: Comprehensive significance testing
5. **Unified Processing**: Single dataset approach with tagging

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