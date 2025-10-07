# Enhanced Monitoring System

A comprehensive monitoring system for trading decisions that provides detailed tracking and explanations across backtesting, paper trading, and live trading modes.

## Overview

The Enhanced Monitoring System captures and analyzes every aspect of trading decisions, providing:

1. **Context Capture** - Exchange, token, time, price, and market conditions
2. **Trade Indicators** - Confidence, risk, and technical indicators
3. **Per-Ensemble Indicators** - Weight of each ML model in ensemble decisions
4. **Per-ML Indicators** - Individual model confidence, risk, and performance
5. **Per-ML Decision Making** - Weight of each trading indicator per model
6. **SHAP/LIME Explanations** - Detailed model interpretability
7. **Monthly CSV Exports** - Comprehensive monthly reports
8. **Daily Ongoing CSV** - Real-time daily metrics tracking

## Features

### 🎯 Comprehensive Decision Tracking
- **Context**: Exchange, token, timestamp, price, volume, timeframe
- **Market Conditions**: Technical indicators, volatility, volume analysis
- **HMM Regime Context**: Regime identification, probabilities, stability
- **Trading Signals**: Signal strength, confidence, quality, risk assessment
- **Model Decisions**: Individual model predictions, confidence, feature importance
- **Ensemble Decisions**: Model weights, consensus, disagreement analysis

### 🔍 Model Interpretability
- **SHAP Explanations**: Feature importance and contribution analysis
- **LIME Explanations**: Local interpretable model-agnostic explanations
- **Combined Explanations**: Consensus between SHAP and LIME
- **Feature Analysis**: Top contributing features per model
- **Decision Tracing**: Complete decision path from data to action

### 📊 Performance Monitoring
- **Model Performance**: Accuracy, precision, recall, F1-score, AUC
- **Trading Performance**: Win rate, profit factor, Sharpe ratio, drawdown
- **Ensemble Performance**: Diversity, consensus quality, weight stability
- **Risk Metrics**: VaR, expected shortfall, maximum drawdown
- **Regime Analysis**: Performance by HMM regime

### 📈 Export and Reporting
- **Monthly Reports**: Comprehensive monthly analysis with detailed breakdowns
- **Daily Ongoing CSV**: Real-time daily metrics (date, exchange, asset, trades, shorts vs long, HMM clusters, Sharpe, PnL)
- **Model Performance CSV**: Individual model performance tracking
- **Ensemble Analysis CSV**: Ensemble performance and weight analysis
- **Decision Traces**: Complete decision history with explanations

### 🔄 Trading Mode Integration
- **Backtesting**: Full integration with backtesting systems
- **Paper Trading**: Real-time monitoring of paper trading decisions
- **Live Trading**: Production monitoring with risk alerts

## Architecture

```
Enhanced Monitoring Orchestrator
├── Trade Decision Context Capture
│   ├── Market Conditions Analysis
│   ├── HMM Regime Context
│   ├── Trading Signal Context
│   ├── Model Decision Context
│   └── Ensemble Decision Context
├── SHAP/LIME Integration
│   ├── SHAP Analyzer
│   ├── LIME Analyzer
│   └── Explainability Integrator
├── Enhanced ML Monitoring
│   ├── Trade Decision Tracking
│   ├── Model Performance Tracking
│   └── Ensemble Performance Tracking
├── Daily Summary Tracker
│   ├── Daily Metrics Calculation
│   ├── Regime Performance Analysis
│   └── Real-time Updates
└── Trading System Integration
    ├── Backtesting Integration
    ├── Paper Trading Integration
    └── Live Trading Integration
```

## Installation

### Dependencies

```bash
pip install numpy pandas scikit-learn shap lime matplotlib seaborn
```

### Optional Dependencies

```bash
# For enhanced visualizations
pip install plotly dash

# For database storage
pip install sqlalchemy psycopg2-binary

# For API endpoints
pip install fastapi uvicorn
```

## Configuration

### Basic Configuration

```yaml
enhanced_monitoring:
  enable_monitoring: true
  enable_explanations: true
  enable_real_time_tracking: true
  monthly_export_enabled: true
  daily_export_enabled: true
  export_directory: "enhanced_monitoring_exports"
  max_decisions_in_memory: 50000
  data_retention_days: 365
```

### SHAP/LIME Configuration

```yaml
shap_analysis:
  enable_shap: true
  max_features: 50
  explanation_timeout: 30

lime_analysis:
  enable_lime: true
  max_features: 20
  num_samples: 1000
  explanation_timeout: 30
```

### Trading Integration Configuration

```yaml
trading_integration:
  enable_monitoring: true
  capture_explanations: true
  capture_performance_metrics: true
  real_time_export: false
  export_interval_minutes: 60
```

## Usage

### Basic Usage

```python
import asyncio
from src.monitoring.enhanced_monitoring_orchestrator import EnhancedMonitoringOrchestrator

# Load configuration
with open('enhanced_monitoring_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize orchestrator
orchestrator = EnhancedMonitoringOrchestrator(config)

# Record a comprehensive trade decision
decision = await orchestrator.record_comprehensive_decision(
    context=trade_context,
    trading_mode=TradingMode.LIVE,
    trading_indicators=trading_indicators,
    ensemble_decision=ensemble_decision,
    individual_model_decisions=model_decisions,
    model_indicator_weights=indicator_weights,
    action="buy",
    position_size=0.1,
    stop_loss=100.0,
    take_profit=110.0
)

# Export monitoring data
await orchestrator.export_monthly_report()
await orchestrator.export_daily_ongoing_csv()
```

### Trading System Integration

```python
# Integrate with trading systems
await orchestrator.integrate_trading_systems(
    backtesting_system=backtesting_system,
    paper_trading_system=paper_trading_system,
    live_trading_system=live_trading_system
)
```

### SHAP/LIME Explanations

```python
from src.monitoring.shap_lime_integration import ModelExplanationRequest

# Create explanation request
request = ModelExplanationRequest(
    model_id="hmm_model",
    model_type="hmm",
    features=np.array([100.0]),
    feature_names=['price'],
    prediction=0.7,
    model=hmm_model,
    training_data=training_data
)

# Generate explanations
explanations = await explainability_integrator.explain_model_prediction(request)
```

## Data Structures

### ComprehensiveTradeDecision

```python
@dataclass
class ComprehensiveTradeDecision:
    decision_id: str
    timestamp: datetime
    trading_mode: TradingMode
    context: TradeContext
    trading_indicators: List[TradingIndicator]
    overall_confidence: float
    overall_risk_score: float
    ensemble_decision: EnsembleDecision
    individual_model_decisions: List[MLModelDecision]
    model_indicator_weights: Dict[str, Dict[str, float]]
    shap_explanations: Optional[Dict[str, Any]]
    lime_explanations: Optional[Dict[str, Any]]
    action: str
    position_size: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    execution_time_ms: float
    success_metrics: Optional[Dict[str, float]]
```

### MarketConditions

```python
@dataclass
class MarketConditions:
    current_price: float
    price_change_1h: float
    price_change_24h: float
    price_change_7d: float
    current_volume: float
    volume_change_1h: float
    volume_avg_24h: float
    volatility_1h: float
    volatility_24h: float
    atr_14: float
    rsi_14: float
    macd_signal: float
    macd_histogram: float
    bollinger_position: float
    adx_14: float
    bid_ask_spread: float
    order_book_imbalance: float
    market_depth: float
```

### HMMRegimeContext

```python
@dataclass
class HMMRegimeContext:
    regime_id: str
    regime_name: str
    regime_probability: float
    regime_transition_probability: float
    regime_duration: int
    regime_stability_score: float
    next_regime_probabilities: Dict[str, float]
    regime_volatility: float
    regime_trend_strength: float
    regime_momentum: float
    regime_win_rate: float
    regime_avg_return: float
    regime_sharpe_ratio: float
```

## Export Formats

### Monthly Report Structure

```
enhanced_monitoring_exports/
├── monthly_reports_2024-01/
│   ├── comprehensive_decisions_2024-01.csv
│   ├── daily_summaries_2024-01.csv
│   ├── model_performance_2024-01.csv
│   ├── ensemble_analysis_2024-01.csv
│   └── monthly_report_summary_2024-01.json
└── ongoing_daily_metrics.csv
```

### Daily Ongoing CSV Columns

- `date`: Trading date
- `exchange`: Exchange name
- `asset`: Trading asset
- `total_trades`: Total number of trades
- `long_trades`: Number of long trades
- `short_trades`: Number of short trades
- `hold_trades`: Number of hold decisions
- `dominant_hmm_clusters`: Dominant HMM regime
- `sharpe_ratio`: Daily Sharpe ratio
- `pnl_absolute`: Absolute PnL
- `pnl_percentage`: Percentage PnL
- `win_rate`: Win rate
- `profit_factor`: Profit factor
- `max_drawdown`: Maximum drawdown
- `avg_confidence`: Average confidence
- `avg_risk_score`: Average risk score
- `model_accuracy_avg`: Average model accuracy
- `ensemble_consensus_avg`: Average ensemble consensus

## Performance Considerations

### Memory Management
- Configurable memory limits for in-memory storage
- Automatic cleanup of old data based on retention policy
- Efficient data structures for large-scale monitoring

### Processing Speed
- Asynchronous processing for non-blocking operations
- Caching of model explainers for faster explanations
- Batch processing for export operations

### Scalability
- Modular architecture for easy scaling
- Configurable export frequencies
- Optional database storage for large datasets

## Monitoring and Alerts

### Built-in Monitoring
- Real-time performance metrics
- Memory usage tracking
- Processing time monitoring
- Error rate tracking

### Alerting (Optional)
- Risk threshold breaches
- Model performance degradation
- System health issues
- Export failures

## Examples

### Complete Example

See `example_enhanced_monitoring_usage.py` for a comprehensive example that demonstrates:

1. System initialization and configuration
2. Mock trading system integration
3. Trade decision recording
4. SHAP/LIME explanation generation
5. Data export and reporting
6. Statistics and monitoring

### Backtesting Integration

```python
# Integrate with backtesting system
await orchestrator.integrate_trading_systems(backtesting_system=backtesting_system)

# Backtesting system will automatically capture all decisions
# and record them in the monitoring system
```

### Paper Trading Integration

```python
# Integrate with paper trading system
await orchestrator.integrate_trading_systems(paper_trading_system=paper_trading_system)

# Paper trading decisions are automatically monitored
# with real-time updates to daily summaries
```

### Live Trading Integration

```python
# Integrate with live trading system
await orchestrator.integrate_trading_systems(live_trading_system=live_trading_system)

# Live trading decisions are monitored with risk alerts
# and real-time performance tracking
```

## Troubleshooting

### Common Issues

1. **SHAP/LIME not available**: Install required dependencies
2. **Memory issues**: Reduce `max_decisions_in_memory` in config
3. **Export failures**: Check export directory permissions
4. **Performance issues**: Adjust export frequencies and cleanup intervals

### Debug Mode

```yaml
development:
  enable_debug_mode: true
  enable_profiling: true
```

### Logging

```yaml
logging:
  log_level: "DEBUG"
  enable_console_logging: true
  enable_file_logging: true
```

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure backward compatibility

## License

This enhanced monitoring system is part of the Ares trading framework and follows the same licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the example usage files
3. Check the configuration options
4. Create an issue with detailed information about your setup