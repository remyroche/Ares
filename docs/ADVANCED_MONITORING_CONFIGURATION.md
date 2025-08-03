# Advanced Monitoring System Configuration

This document provides comprehensive configuration examples for the advanced monitoring and tracking system.

## Overview

The advanced monitoring system consists of several integrated components:

1. **Metrics Dashboard** - Real-time metrics visualization
2. **Advanced Tracer** - Correlation ID tracking and distributed tracing
3. **Correlation Manager** - Request/response correlation management
4. **ML Monitor** - Machine learning monitoring and drift detection
5. **Report Scheduler** - Automated report generation and distribution
6. **Tracking System** - Enhanced tracking with ensemble data
7. **Integration Manager** - Unified coordination of all components

## Configuration Structure

### Main Configuration

```yaml
# Add to your existing config.py or config.yaml
monitoring:
  # Enable/disable monitoring system
  enabled: true
  
  # Metrics Dashboard Configuration
  metrics_dashboard:
    update_interval: 5  # seconds
    max_metric_history: 1000
    enable_real_time_updates: true
    enable_websocket_broadcast: true
    
  # Advanced Tracer Configuration
  advanced_tracer:
    enable_tracing: true
    correlation_id_header: "X-Correlation-ID"
    trace_sampling_rate: 1.0
    max_trace_history: 10000
    enable_performance_tracing: true
    enable_error_tracing: true
    
  # Correlation Manager Configuration
  correlation_manager:
    enable_correlation_tracking: true
    correlation_timeout: 300  # 5 minutes
    max_correlation_history: 10000
    enable_performance_analysis: true
    enable_debug_aggregation: true
    
  # ML Monitor Configuration
  ml_monitor:
    enable_online_learning: true
    drift_detection_enabled: true
    feature_importance_tracking: true
    auto_retraining_enabled: true
    drift_threshold: 0.1
    drift_check_interval: 300  # 5 minutes
    performance_check_interval: 60  # 1 minute
    feature_analysis_interval: 600  # 10 minutes
    
  # Report Scheduler Configuration
  report_scheduler:
    enable_automated_reports: true
    default_schedule: "daily"
    email_distribution: true
    report_formats: ["pdf", "html", "json"]
    default_recipients: ["admin@example.com"]
    
    # Email configuration
    email:
      smtp_server: "smtp.gmail.com"
      smtp_port: 587
      username: "your-email@gmail.com"
      password: "your-app-password"
      use_tls: true
      
  # Tracking System Configuration
  tracking_system:
    enable_correlation_tracking: true
    enable_ensemble_tracking: true
    enable_regime_tracking: true
    enable_decision_path_tracking: true
    max_tracking_history: 50000
    
  # Integration Manager Configuration
  monitoring_integration:
    enable_unified_monitoring: true
    enable_cross_component_tracking: true
    enable_performance_correlation: true
```

## Component-Specific Configuration

### Metrics Dashboard

```yaml
metrics_dashboard:
  # Real-time update interval
  update_interval: 5
  
  # Maximum number of metric history points
  max_metric_history: 1000
  
  # Enable real-time updates
  enable_real_time_updates: true
  
  # Enable WebSocket broadcasting
  enable_websocket_broadcast: true
  
  # Metric aggregation windows
  aggregation_windows:
    "1m": 60
    "5m": 300
    "15m": 900
    "1h": 3600
    "1d": 86400
```

### Advanced Tracer

```yaml
advanced_tracer:
  # Enable tracing
  enable_tracing: true
  
  # Correlation ID header name
  correlation_id_header: "X-Correlation-ID"
  
  # Trace sampling rate (0.0 to 1.0)
  trace_sampling_rate: 1.0
  
  # Maximum trace history
  max_trace_history: 10000
  
  # Enable performance tracing
  enable_performance_tracing: true
  
  # Enable error tracing
  enable_error_tracing: true
  
  # Component types to trace
  component_types:
    - "analyst"
    - "strategist"
    - "tactician"
    - "supervisor"
    - "exchange"
    - "database"
    - "gui"
    - "monitoring"
```

### ML Monitor

```yaml
ml_monitor:
  # Enable online learning monitoring
  enable_online_learning: true
  
  # Enable drift detection
  drift_detection_enabled: true
  
  # Enable feature importance tracking
  feature_importance_tracking: true
  
  # Enable automatic retraining
  auto_retraining_enabled: true
  
  # Drift detection threshold
  drift_threshold: 0.1
  
  # Monitoring intervals
  drift_check_interval: 300  # 5 minutes
  performance_check_interval: 60  # 1 minute
  feature_analysis_interval: 600  # 10 minutes
  
  # Drift detection methods
  drift_detection_methods:
    - "statistical_test"
    - "distribution_comparison"
    - "performance_degradation"
    - "feature_drift"
    
  # Auto-retraining triggers
  auto_retraining_triggers:
    critical_drift_threshold: 0.3
    performance_degradation_threshold: 0.2
    consecutive_failures: 5
```

### Report Scheduler

```yaml
report_scheduler:
  # Enable automated reports
  enable_automated_reports: true
  
  # Default report schedule
  default_schedule: "daily"
  
  # Enable email distribution
  email_distribution: true
  
  # Report formats
  report_formats: ["pdf", "html", "json"]
  
  # Default recipients
  default_recipients: ["admin@example.com"]
  
  # Report configurations
  reports:
    performance_summary:
      schedule: "daily"
      format: "pdf"
      recipients: ["admin@example.com", "trading@example.com"]
      enabled: true
      
    model_analysis:
      schedule: "weekly"
      format: "html"
      recipients: ["ml-team@example.com"]
      enabled: true
      
    risk_assessment:
      schedule: "daily"
      format: "pdf"
      recipients: ["risk@example.com"]
      enabled: true
      
    executive_summary:
      schedule: "weekly"
      format: "pdf"
      recipients: ["executive@example.com"]
      enabled: true
      
  # Email configuration
  email:
    smtp_server: "smtp.gmail.com"
    smtp_port: 587
    username: "your-email@gmail.com"
    password: "your-app-password"
    use_tls: true
    from_address: "ares-bot@example.com"
    reply_to: "support@example.com"
```

### Tracking System

```yaml
tracking_system:
  # Enable correlation tracking
  enable_correlation_tracking: true
  
  # Enable ensemble tracking
  enable_ensemble_tracking: true
  
  # Enable regime tracking
  enable_regime_tracking: true
  
  # Enable decision path tracking
  enable_decision_path_tracking: true
  
  # Maximum tracking history
  max_tracking_history: 50000
  
  # Tracking intervals
  correlation_update_interval: 10  # seconds
  ensemble_update_interval: 30  # seconds
  regime_update_interval: 60  # seconds
  decision_update_interval: 30  # seconds
  
  # Ensemble tracking configuration
  ensemble_tracking:
    track_individual_predictions: true
    track_ensemble_weights: true
    track_meta_learner: true
    track_confidence_scores: true
    
  # Regime tracking configuration
  regime_tracking:
    track_regime_probabilities: true
    track_regime_features: true
    track_regime_indicators: true
    track_regime_transitions: true
    track_regime_duration: true
    
  # Feature tracking configuration
  feature_tracking:
    track_importance_scores: true
    track_stability_scores: true
    track_drift_scores: true
    track_rankings: true
```

## Integration with Existing Components

### Performance Monitor Integration

```python
# In your existing performance monitor
from src.monitoring import AdvancedTracer, CorrelationManager

class PerformanceMonitor:
    def __init__(self, config):
        self.tracer = AdvancedTracer(config)
        self.correlation_manager = CorrelationManager(config)
        
    async def monitor_performance(self, correlation_id: str):
        async with self.tracer.trace_span(
            component_type=ComponentType.MONITORING,
            operation_name="performance_monitoring",
            correlation_id=correlation_id
        ) as span:
            # Your performance monitoring logic here
            pass
```

### Enhanced Model Monitor Integration

```python
# In your existing enhanced model monitor
from src.monitoring import MLMonitor, TrackingSystem

class EnhancedModelMonitor:
    def __init__(self, config):
        self.ml_monitor = MLMonitor(config)
        self.tracking_system = TrackingSystem(config)
        
    async def monitor_model_behavior(self, model_id: str):
        # Record model behavior tracking
        await self.tracking_system.record_model_behavior_tracking(
            model_id=model_id,
            prediction_consistency=0.85,
            confidence_trend=[0.8, 0.82, 0.79],
            feature_importance_stability=0.9,
            prediction_drift=0.05
        )
```

### Trade Tracker Integration

```python
# In your existing trade tracker
from src.monitoring import TrackingSystem, CorrelationManager

class TradeTracker:
    def __init__(self, config):
        self.tracking_system = TrackingSystem(config)
        self.correlation_manager = CorrelationManager(config)
        
    async def record_trade(self, trade_data: Dict[str, Any], correlation_id: str):
        # Record ensemble tracking
        await self.tracking_system.record_ensemble_tracking(
            ensemble_id="ensemble_1",
            ensemble_type="regime_ensemble",
            individual_predictions={"model_1": 0.8, "model_2": 0.75},
            ensemble_weights={"model_1": 0.6, "model_2": 0.4},
            final_prediction="buy",
            confidence=0.78
        )
        
        # Record regime tracking
        await self.tracking_system.record_regime_tracking(
            regime_type="BULL_TREND",
            regime_confidence=0.85,
            regime_probabilities={"BULL_TREND": 0.6, "BEAR_TREND": 0.2, "SIDEWAYS": 0.2},
            regime_features=["price_momentum", "volatility"],
            regime_indicators={"momentum_score": 0.7, "volatility_score": 0.3},
            regime_transition_probability=0.1
        )
```

## Database Schema Extensions

### SQLite Extensions

```sql
-- Correlation tracking table
CREATE TABLE correlation_tracking (
    correlation_id TEXT PRIMARY KEY,
    request_timestamp DATETIME,
    response_timestamp DATETIME,
    component_path TEXT,
    performance_metrics JSON,
    error_info JSON,
    metadata JSON
);

-- Model behavior tracking table
CREATE TABLE model_behavior_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT,
    timestamp DATETIME,
    performance_metrics JSON,
    feature_importance JSON,
    drift_scores JSON,
    ensemble_weights JSON,
    metadata JSON
);

-- Report history table
CREATE TABLE report_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_type TEXT,
    generated_at DATETIME,
    schedule_type TEXT,
    recipients JSON,
    file_path TEXT,
    status TEXT
);

-- Ensemble tracking table
CREATE TABLE ensemble_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ensemble_id TEXT,
    timestamp DATETIME,
    ensemble_type TEXT,
    individual_predictions JSON,
    ensemble_weights JSON,
    final_prediction TEXT,
    confidence REAL,
    meta_learner_prediction TEXT,
    meta_learner_confidence REAL
);

-- Regime tracking table
CREATE TABLE regime_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    regime_type TEXT,
    timestamp DATETIME,
    regime_confidence REAL,
    regime_probabilities JSON,
    regime_features JSON,
    regime_indicators JSON,
    regime_transition_probability REAL,
    regime_duration INTEGER
);
```

## API Endpoints

### Metrics Dashboard API

```
GET /api/metrics/dashboard - Get current dashboard data
GET /api/metrics/history/{metric_name} - Get metric history
GET /api/metrics/performance - Get performance metrics
GET /api/metrics/system-health - Get system health metrics
```

### Tracing API

```
GET /api/tracing/correlation/{correlation_id} - Get trace data
GET /api/tracing/performance - Get performance traces
GET /api/tracing/errors - Get error traces
```

### ML Monitoring API

```
GET /api/ml/models - Get model performance
GET /api/ml/drift - Get drift detection results
GET /api/ml/features - Get feature importance
GET /api/ml/ensembles - Get ensemble performance
```

### Reports API

```
GET /api/reports/schedule - Get report schedule
POST /api/reports/generate - Generate custom report
GET /api/reports/history - Get report history
```

### Tracking API

```
GET /api/tracking/ensembles - Get ensemble tracking data
GET /api/tracking/regimes - Get regime tracking data
GET /api/tracking/features - Get feature tracking data
GET /api/tracking/decisions - Get decision path tracking data
```

## Usage Examples

### Basic Setup

```python
from src.monitoring import MonitoringIntegrationManager

# Initialize monitoring system
config = {
    "monitoring": {
        "enabled": True,
        "metrics_dashboard": {"update_interval": 5},
        "advanced_tracer": {"enable_tracing": True},
        # ... other configurations
    }
}

# Setup and start monitoring
integration_manager = await setup_monitoring_integration_manager(config)
await integration_manager.start_integration()

# Get unified dashboard data
dashboard_data = integration_manager.get_unified_dashboard_data()
```

### Advanced Usage with Correlation IDs

```python
from src.monitoring import AdvancedTracer, ComponentType

# Initialize tracer
tracer = AdvancedTracer(config)

# Trace a complete request
with tracer.trace_request(correlation_id="req_123") as trace_request:
    # Trace analyst component
    async with tracer.trace_span(
        component_type=ComponentType.ANALYST,
        operation_name="feature_engineering",
        correlation_id="req_123"
    ) as span:
        # Your analyst logic here
        span.metadata["features_generated"] = 50
        
    # Trace strategist component
    async with tracer.trace_span(
        component_type=ComponentType.STRATEGIST,
        operation_name="strategy_execution",
        correlation_id="req_123"
    ) as span:
        # Your strategist logic here
        span.metadata["strategy_applied"] = "momentum_based"
```

### ML Monitoring with Drift Detection

```python
from src.monitoring import MLMonitor

# Initialize ML monitor
ml_monitor = MLMonitor(config)
await ml_monitor.start_monitoring()

# Get drift alerts
drift_alerts = ml_monitor.get_drift_alerts(severity="critical")

# Get model performance
performance = ml_monitor.get_model_performance_history("ensemble_1", limit=100)
```

### Automated Reporting

```python
from src.monitoring import ReportScheduler

# Initialize report scheduler
scheduler = ReportScheduler(config)
await scheduler.start_scheduling()

# Add custom report configuration
scheduler.add_report_config("custom_report", ReportConfig(
    report_type=ReportType.PERFORMANCE_SUMMARY,
    schedule=ReportSchedule.DAILY,
    format=ReportFormat.PDF,
    recipients=["custom@example.com"],
    enabled=True
))
```

## Performance Considerations

### Memory Management

- Set appropriate `max_trace_history` and `max_tracking_history` values
- Enable periodic cleanup of old data
- Use sampling for high-volume tracing

### Database Optimization

- Index frequently queried fields
- Use appropriate data types for JSON fields
- Implement data retention policies

### Network Optimization

- Use WebSocket compression for real-time updates
- Implement connection pooling for database connections
- Use caching for frequently accessed data

## Security Considerations

### Access Control

- Implement authentication for monitoring APIs
- Use role-based access control for sensitive data
- Encrypt sensitive configuration data

### Data Privacy

- Anonymize sensitive data in traces
- Implement data retention policies
- Use secure communication for email reports

### Audit Logging

- Log all monitoring system access
- Track configuration changes
- Monitor for suspicious activity

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Reduce history limits
   - Enable more aggressive cleanup
   - Use sampling for tracing

2. **Database Performance**
   - Add appropriate indexes
   - Implement data partitioning
   - Optimize query patterns

3. **Network Issues**
   - Check WebSocket connections
   - Verify email configuration
   - Monitor API response times

### Debugging

```python
# Enable debug logging
import logging
logging.getLogger("src.monitoring").setLevel(logging.DEBUG)

# Get detailed component status
status = integration_manager.get_integration_status()
print(json.dumps(status, indent=2))

# Export trace data for analysis
trace_data = tracer.export_trace_data(correlation_id="req_123")
print(trace_data)
```

This configuration provides a comprehensive foundation for implementing advanced monitoring and tracking capabilities in your trading bot. 