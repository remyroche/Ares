# Enhanced ML Monitoring System

A comprehensive monitoring system for ML models and ensembles with detailed explanations using SHAP/LIME, designed for backtesting, paper trading, and live trading environments.

## Overview

The Enhanced ML Monitoring System provides:

- **Comprehensive Trade Decision Tracking**: Captures complete context, indicators, and ML model details for every trade decision
- **SHAP/LIME Integration**: Detailed model explanations for understanding decision-making processes
- **Ensemble Monitoring**: Tracks individual model weights, performance, and contributions
- **Multi-Mode Support**: Works across backtesting, paper trading, and live trading
- **Automated CSV Export**: Monthly reports with detailed breakdowns and statistics
- **Real-time Performance Tracking**: Continuous monitoring of model and ensemble performance

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Monitoring Orchestrator                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Enhanced ML     │  │ Explainability  │  │ Ensemble     │ │
│  │ Monitor         │  │ Integrator      │  │ Monitor      │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ CSV Export      │  │ Trading System  │  │ Performance  │ │
│  │ Manager         │  │ Integrator      │  │ Tracker      │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Enhanced ML Monitor (`enhanced_ml_monitoring.py`)
- Records complete trade decisions with context and explanations
- Tracks model and ensemble performance metrics
- Manages memory and export scheduling

### 2. Explainability Integration (`explainability_integration.py`)
- Integrates SHAP and LIME for model explanations
- Generates feature-level importance scores
- Provides ensemble-level explanation aggregation

### 3. Ensemble Monitor (`ensemble_monitor.py`)
- Tracks individual model weights and performance
- Monitors ensemble diversity and consensus
- Provides weight optimization recommendations

### 4. CSV Export Manager (`csv_export_manager.py`)
- Exports comprehensive monthly reports
- Creates detailed breakdowns by component
- Generates summary statistics and analysis

### 5. Trading System Integration (`trading_integration.py`)
- Integrates with backtesting, paper trading, and live trading systems
- Automatically captures trade decisions
- Hooks into existing trading workflows

### 6. Monitoring Orchestrator (`monitoring_orchestrator.py`)
- Coordinates all monitoring components
- Provides unified interface for the entire system
- Manages initialization and shutdown

## Data Structures

### Trade Decision Hierarchy

```
Trade Decision
├── Context (exchange, token, time, price, volume, timeframe, regime)
├── Trading Indicators (RSI, MACD, Bollinger Bands, etc.)
│   ├── Name, Value, Weight, Confidence, Risk Score
├── Ensemble Decision
│   ├── Final Prediction, Confidence, Risk Score
│   ├── Model Weights (per-model contribution)
│   └── Individual Model Decisions
│       ├── Model ID, Type, Prediction, Confidence
│       ├── Feature Importance
│       ├── SHAP Values (if available)
│       └── LIME Explanation (if available)
└── Final Action (buy/sell/hold, position size, stop loss, take profit)
```

## Configuration

### Basic Configuration

```yaml
enhanced_monitoring:
  enable_monitoring: true
  enable_explanations: true
  enable_ensemble_monitoring: true
  enable_csv_export: true
  export_interval_days: 30
  max_memory_decisions: 10000
  export_directory: "monitoring_exports"
```

### SHAP/LIME Configuration

```yaml
explainability_integration:
  enable_shap: true
  enable_lime: true
  max_features_explained: 20
  explanation_cache_size: 1000
```

### Ensemble Monitoring Configuration

```yaml
ensemble_monitoring:
  weight_update_frequency_hours: 24
  performance_window_days: 30
  min_weight_threshold: 0.01
  max_weight_threshold: 0.8
  rebalance_threshold: 0.1
```

## Usage Examples

### Basic Usage

```python
from src.monitoring import create_monitoring_orchestrator

# Initialize monitoring system
config = {
    "enhanced_monitoring": {
        "enable_monitoring": True,
        "export_interval_days": 30
    }
}

orchestrator = await create_monitoring_orchestrator(config)

# Record a trade decision
from src.monitoring import TradeDecision, TradeContext, EnsembleDecision

context = TradeContext(
    exchange="binance",
    token="BTCUSDT",
    timestamp=datetime.now(),
    price=45000.0,
    volume=0.1,
    timeframe="1h"
)

ensemble_decision = EnsembleDecision(
    ensemble_id="main_ensemble",
    final_prediction=0.75,
    final_confidence=0.85,
    model_weights={"model_1": 0.6, "model_2": 0.4},
    model_decisions=[],
    voting_mechanism="weighted_average",
    consensus_score=0.8,
    disagreement_level=0.15
)

trade_decision = TradeDecision(
    decision_id="trade_001",
    context=context,
    trading_mode=TradingMode.PAPER,
    timestamp=datetime.now(),
    trading_indicators=[],
    overall_confidence=0.82,
    overall_risk_score=0.18,
    ensemble_decision=ensemble_decision,
    action="buy",
    position_size=0.1
)

await orchestrator.record_trade_decision(trade_decision)
```

### Trading System Integration

```python
# Integrate with existing trading systems
await orchestrator.integrate_trading_system(backtesting_system, "backtesting")
await orchestrator.integrate_trading_system(paper_trading_system, "paper_trading")
await orchestrator.integrate_trading_system(live_trading_system, "live_trading")

# All trades will now be automatically monitored
```

### Ensemble Weight Updates

```python
# Update ensemble weights based on performance
model_performances = {
    "model_1": {"accuracy": 0.78, "win_rate": 0.72},
    "model_2": {"accuracy": 0.75, "win_rate": 0.68}
}

current_weights = {"model_1": 0.6, "model_2": 0.4}
new_weights = await orchestrator.update_ensemble_weights(
    "main_ensemble", model_performances, current_weights
)
```

### Getting Analysis

```python
# Get comprehensive ensemble analysis
analysis = await orchestrator.get_ensemble_analysis("main_ensemble")

# Get monitoring statistics
stats = orchestrator.get_comprehensive_stats()

# Force export monitoring data
await orchestrator.export_monitoring_data()
```

## CSV Export Structure

The system generates comprehensive CSV reports with the following structure:

### Main Files
- `trade_decisions_YYYYMMDD_HHMMSS_main.csv` - Complete trade decisions
- `model_performances_YYYYMMDD_HHMMSS_main.csv` - Model performance metrics
- `ensemble_performances_YYYYMMDD_HHMMSS_main.csv` - Ensemble performance metrics

### Breakdown Files
- `trade_decisions_YYYYMMDD_HHMMSS_trading_indicators.csv` - Trading indicators breakdown
- `trade_decisions_YYYYMMDD_HHMMSS_ensemble_breakdown.csv` - Ensemble decision breakdown
- `trade_decisions_YYYYMMDD_HHMMSS_model_breakdown.csv` - Individual model breakdown
- `trade_decisions_YYYYMMDD_HHMMSS_context_analysis.csv` - Market context analysis

### Summary Files
- `trade_decisions_YYYYMMDD_HHMMSS_summary.csv` - Summary statistics
- `model_performances_YYYYMMDD_HHMMSS_summary.csv` - Model performance summary
- `monitoring_summary_YYYYMMDD_HHMMSS.json` - Overall monitoring summary

## Key Features

### 1. Comprehensive Context Capture
- Exchange, token, timestamp, price, volume
- Timeframe and market regime
- Market conditions (volatility, trend, volume profile)

### 2. Trading Indicators Tracking
- RSI, MACD, Bollinger Bands, Moving Averages
- Volume profile, Support/Resistance levels
- Momentum and volatility indicators
- Each with weight, confidence, and risk scores

### 3. ML Model Decision Details
- Individual model predictions and confidence
- Feature importance scores
- SHAP values for feature contributions
- LIME explanations for local interpretability
- Processing time and model version tracking

### 4. Ensemble Analysis
- Model weight distribution and stability
- Consensus and disagreement metrics
- Performance-based weight optimization
- Model diversity and contribution analysis

### 5. Performance Monitoring
- Accuracy, precision, recall, F1-score
- Trading performance (win rate, profit factor, Sharpe ratio)
- Model stability and drift detection
- Feature importance stability tracking

### 6. Automated Reporting
- Monthly CSV exports with detailed breakdowns
- Summary statistics and trend analysis
- Performance comparisons across models and ensembles
- Export metadata and compression options

## Integration Points

### Backtesting Integration
- Hooks into trade execution methods
- Captures prediction contexts
- Records performance metrics
- Tracks model behavior over time

### Paper Trading Integration
- Monitors simulated trades
- Records decision-making process
- Tracks performance without risk
- Validates model behavior

### Live Trading Integration
- Captures real trade decisions
- Monitors actual performance
- Tracks model reliability
- Provides real-time insights

## Dependencies

### Required Libraries
- `pandas` - Data manipulation and CSV export
- `numpy` - Numerical computations
- `asyncio` - Asynchronous operations
- `dataclasses` - Data structure definitions
- `pathlib` - File system operations

### Optional Libraries
- `shap` - SHAP explanations (install with `pip install shap`)
- `lime` - LIME explanations (install with `pip install lime`)

## Performance Considerations

### Memory Management
- Configurable memory limits for decision storage
- Automatic cleanup of old data
- Efficient data structures for large-scale monitoring

### Export Optimization
- Configurable export intervals
- Compression options for large datasets
- Batch processing for efficient I/O

### Caching
- Explanation caching to avoid recomputation
- Model performance caching
- Configurable cache sizes

## Error Handling

The system includes comprehensive error handling:
- Graceful degradation when SHAP/LIME unavailable
- Fallback mechanisms for missing data
- Detailed logging for debugging
- Recovery from component failures

## Monitoring and Debugging

### Logging
- Comprehensive logging at all levels
- Performance metrics tracking
- Error reporting and debugging information

### Statistics
- Real-time monitoring statistics
- Component health monitoring
- Performance trend analysis

### Debugging Tools
- Export functionality for data analysis
- Comprehensive statistics reporting
- Component status monitoring

## Future Enhancements

### Planned Features
- Real-time dashboard integration
- Advanced visualization tools
- Machine learning-based anomaly detection
- Automated model retraining triggers
- Integration with external monitoring systems

### Extensibility
- Plugin architecture for custom metrics
- Custom explanation methods
- Flexible export formats
- Integration with external databases

## Support and Maintenance

### Configuration Updates
- Hot-reloadable configuration
- Runtime parameter adjustments
- Component enable/disable options

### Data Management
- Automated cleanup of old data
- Backup and restore functionality
- Data migration tools

### Performance Tuning
- Configurable performance parameters
- Memory usage optimization
- Export scheduling optimization

This enhanced monitoring system provides a comprehensive solution for tracking and analyzing ML model performance across all trading modes, with detailed explanations and automated reporting capabilities.