# Comprehensive Monitoring System

This monitoring system provides complete observability for the Ares trading bot, offering detailed insights into every aspect of trading performance, ML model behavior, market conditions, and system health.

## 🎯 Overview

The enhanced monitoring system captures:

1. **What we did**: Every trade decision with complete context
2. **When**: Precise timing, market regime, and S/R level proximity  
3. **Why**: Detailed ML model predictions and reasoning
4. **In-depth analysis**: Multi-timeframe feature analysis (30m, 15m, 5m, 1m)
5. **Performance tracking**: Comprehensive ML model validation and comparison
6. **Error detection**: Real-time anomaly detection and alerting

## 🏗️ Architecture

### Core Components

1. **`TradeConditionsMonitor`** - Comprehensive trade decision tracking
2. **`EnhancedMLTracker`** - ML model performance and prediction analysis
3. **`RegimeSRTracker`** - Market regime detection and S/R level identification
4. **`ErrorDetectionSystem`** - Anomaly detection and alerting
5. **`MonitoringIntegrationManager`** - Unified coordination of all components

### Component Relationships

```
MonitoringIntegrationManager
├── TradeConditionsMonitor
│   ├── Multi-timeframe feature analysis
│   ├── Trade decision context recording
│   └── Performance outcome tracking
├── EnhancedMLTracker
│   ├── Individual model prediction tracking
│   ├── Ensemble performance monitoring
│   └── Model comparison and ranking
├── RegimeSRTracker
│   ├── Market regime detection
│   ├── Support/Resistance level identification
│   └── Regime-specific performance analysis
└── ErrorDetectionSystem
    ├── Real-time anomaly detection
    ├── System health monitoring
    └── Intelligent alerting
```

## 🚀 Key Features

### Trade Conditions Monitoring

- **Complete Decision Context**: Captures every factor influencing trade decisions
- **Multi-timeframe Analysis**: Features from 30m, 15m, 5m, 1m timeframes
- **Model Predictions**: Individual and ensemble model outputs with reasoning
- **Regime Context**: Market regime and S/R level proximity
- **Performance Validation**: Complete trade lifecycle tracking

### ML Model Performance Tracking

- **Real-time Tracking**: Every model prediction logged with context
- **Ensemble Analysis**: How model combinations perform vs individuals  
- **Performance Comparison**: Automated model ranking and recommendations
- **Confidence Calibration**: How well model confidence matches actual accuracy
- **Feature Importance**: Which features drive predictions across timeframes

### Market Regime & S/R Analysis

- **Regime Detection**: Automatic classification of market conditions
- **S/R Level Identification**: Dynamic support/resistance level tracking
- **Transition Monitoring**: Regime change detection and impact analysis
- **Performance by Regime**: How strategies perform in different conditions

### Error Detection & Alerting

- **Anomaly Detection**: Statistical and ML-based anomaly identification
- **System Health**: Real-time monitoring of CPU, memory, network
- **Intelligent Alerts**: Context-aware notifications with severity levels
- **Predictive Warnings**: Early detection of potential issues

## 📊 Data Captured

### Trade Decision Record
```python
TradeDecisionContext(
    decision_id="decision_ETHUSDT_1234567890",
    timestamp=datetime.now(),
    symbol="ETHUSDT",
    current_price=3500.0,
    current_regime=RegimeType.BULL_TREND,
    regime_confidence=0.85,
    
    # Multi-timeframe features
    timeframe_features={
        "30m": MultiTimeframeFeatures(...),
        "15m": MultiTimeframeFeatures(...),
        "5m": MultiTimeframeFeatures(...),
        "1m": MultiTimeframeFeatures(...)
    },
    
    # Model predictions
    ensemble_predictions=[
        EnsemblePrediction(
            individual_predictions=[...],
            aggregated_prediction=0.65,
            consensus_level=0.78
        )
    ],
    
    # S/R context
    nearby_sr_levels=[...],
    
    # Risk assessment
    risk_score=0.3,
    recommended_action=TradeAction.ENTER_LONG
)
```

### Model Performance Record
```python
ModelPerformanceAnalysis(
    model_id="xgb_trend_1h",
    analysis_period_days=7,
    total_predictions=245,
    directional_accuracy=0.67,
    confidence_calibration=0.72,
    feature_stability_score=0.85,
    performance_trend="improving",
    regime_performance={
        "bull_trend": 0.75,
        "bear_trend": 0.58,
        "sideways": 0.62
    }
)
```

## 🔧 Configuration

### Basic Setup
```python
config = {
    "monitoring": {
        "storage_backend": "sqlite",
        "database_path": "monitoring.db"
    },
    "trade_conditions_monitor": {
        "enable_detailed_logging": True,
        "enable_feature_analysis": True,
        "enable_model_tracking": True
    },
    "enhanced_ml_tracker": {
        "enable_real_time_tracking": True,
        "enable_ensemble_analysis": True,
        "performance_window_days": 7
    },
    "regime_sr_tracker": {
        "enable_regime_tracking": True,
        "enable_sr_tracking": True,
        "regime_detection_interval": 60
    },
    "error_detection": {
        "enable_anomaly_detection": True,
        "enable_email_alerts": True,
        "monitoring_interval": 30
    }
}
```

### Integration with Trading Pipeline
```python
from src.monitoring import MonitoringIntegrationManager

# Initialize monitoring
monitoring_manager = MonitoringIntegrationManager(config)
await monitoring_manager.initialize()

# Get components
trade_monitor = monitoring_manager.components.trade_conditions_monitor
ml_tracker = monitoring_manager.components.enhanced_ml_tracker
regime_tracker = monitoring_manager.components.regime_sr_tracker
error_detector = monitoring_manager.components.error_detection_system
```

## 📈 Usage Examples

### Recording a Trade Decision
```python
# Collect multi-timeframe features
timeframe_features = await trade_monitor.get_multi_timeframe_features(
    symbol="ETHUSDT",
    timestamp=datetime.now(),
    timeframes=["30m", "15m", "5m", "1m"]
)

# Create decision context
decision_context = TradeDecisionContext(
    # ... decision details
    timeframe_features=timeframe_features,
    ensemble_predictions=model_predictions,
    # ... other context
)

# Record decision
await trade_monitor.record_trade_decision(decision_context)
```

### Tracking ML Model Predictions
```python
# Track individual model prediction
prediction_id = await ml_tracker.track_model_prediction(
    model_id="xgb_trend_1h",
    model_type=ModelType.XGBOOST,
    ensemble_name="trend_following",
    prediction=0.65,
    confidence=0.82,
    features=feature_dict,
    feature_importance=importance_dict
)

# Record actual outcome later
await ml_tracker.record_actual_outcome(
    prediction_id=prediction_id,
    actual_outcome=0.045,  # 4.5% return
    outcome_timestamp=datetime.now()
)
```

### Monitoring Market Conditions
```python
# Detect current regime
regime = await regime_tracker.detect_current_regime("ETHUSDT", "1h")
print(f"Current regime: {regime.current_regime.value} (confidence: {regime.confidence:.2f})")

# Identify S/R levels
sr_levels = await regime_tracker.identify_sr_levels("ETHUSDT", "1h")
for level in sr_levels:
    print(f"{level.level_type.value}: ${level.price:.2f} (strength: {level.strength:.2f})")
```

### Error Detection and Alerting
```python
# Record error event
await error_detector.record_error_event(
    severity=AlertSeverity.ERROR,
    category=ErrorCategory.MODEL,
    error_message="Model accuracy dropped below threshold",
    component="MLPredictor",
    impact_score=0.7
)

# Detect anomalies
anomaly = await error_detector.detect_anomaly(
    metric_name="prediction_accuracy",
    current_value=0.35,
    expected_value=0.65
)
```

## 📊 Reporting and Analysis

### Generate Comprehensive Reports
```python
# Trade monitoring report
trade_report = await trade_monitor.generate_monitoring_report(days=7)

# ML model comparison
model_comparison = await ml_tracker.generate_model_comparison_report()

# System health summary
health_stats = await error_detector.get_detection_statistics()
```

### Key Metrics Available

#### Trade Performance
- Win rate by regime type
- Average PnL per trade
- Risk-adjusted returns
- Model prediction accuracy validation

#### ML Model Performance
- Individual model accuracy trends
- Ensemble effectiveness
- Feature importance stability
- Confidence calibration scores

#### System Health
- Error rates and anomaly counts
- Resource utilization trends
- Alert frequency and resolution times
- Performance degradation indicators

## 🔍 Debugging and Troubleshooting

### Common Issues

1. **High Memory Usage**: Check `health_metrics_history` size in monitoring components
2. **Database Locks**: Ensure proper async connection management
3. **Missing Features**: Verify data source integration in `_fetch_timeframe_data`
4. **Alert Fatigue**: Adjust `cooldown_minutes` and `max_alerts_per_hour` in alert rules

### Debug Logging
```python
import logging
logging.getLogger("src.monitoring").setLevel(logging.DEBUG)
```

### Performance Monitoring
```python
# Check monitoring statistics
stats = await monitoring_manager.get_unified_statistics()
print(f"Trade decisions tracked: {stats['trade_decisions_tracked']}")
print(f"ML predictions tracked: {stats['ml_predictions_tracked']}")
print(f"Anomalies detected: {stats['anomalies_detected']}")
```

## 🚀 Advanced Features

### Custom Alert Rules
```python
custom_rule = AlertRule(
    rule_id="custom_accuracy_check",
    rule_name="Custom Model Accuracy Check",
    category=ErrorCategory.MODEL,
    metric_name="model_accuracy",
    condition="less_than",
    threshold=0.5,
    severity=AlertSeverity.WARNING,
    evaluation_window_minutes=15,
    notify_email=True
)

error_detector.alert_rules["custom_accuracy_check"] = custom_rule
```

### Multi-Exchange Support
The system supports monitoring across multiple exchanges by including exchange context in all records.

### Regime-Specific Analysis
Performance can be analyzed by regime type to identify which strategies work best in different market conditions.

## 🛠️ Extension Points

### Adding New Metrics
1. Extend `SystemHealthMetrics` dataclass
2. Update `collect_system_health_metrics` method
3. Add baseline calculation in `_calculate_metric_baselines`

### Custom Anomaly Detection
1. Implement new detection method in `ErrorDetectionSystem`
2. Add to `AnomalyType` enum
3. Update `_classify_anomaly_type` method

### Additional Timeframes
Simply add new timeframes to the `timeframes` list in configuration - the system automatically adapts.

## 📚 Documentation

- See `monitoring_integration_example.py` for complete usage examples
- Check individual component docstrings for detailed API documentation
- Review test files for integration patterns

## 🤝 Contributing

When extending the monitoring system:

1. Follow the established dataclass patterns for new record types
2. Implement proper error handling with the `@handle_errors` decorator
3. Add comprehensive logging for debugging
4. Update database schemas for new data types
5. Include statistical validation for new metrics

## 📞 Support

For issues or questions:
1. Check the debug logs with `logging.DEBUG` level
2. Review the monitoring statistics for component health
3. Verify configuration against the example in `monitoring_integration_example.py`
4. Check database connectivity and schema integrity