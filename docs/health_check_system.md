# Ares Health Check System

## Overview

The Ares Health Check System provides comprehensive monitoring and health assessment for all components in the trading system. It integrates with the existing Sentinel monitoring and Prometheus metrics to provide real-time health status, alerting, and performance monitoring.

## Architecture

### Core Components

1. **SystemHealthChecker** - Main orchestrator for health monitoring
2. **ComponentHealthChecker** - Individual component health assessment
3. **Sentinel Integration** - System-wide monitoring and alerting
4. **Prometheus Metrics** - Health metrics collection and exposure
5. **Health Mixins** - Component-specific health check implementations

### Component Health Mixins

- **AnalystHealthMixin** - Health checks for data analysis components
- **StrategistHealthMixin** - Health checks for trading strategy components  
- **TacticianHealthMixin** - Health checks for order execution components
- **ExchangeHealthMixin** - Health checks for exchange connectivity

## API Endpoints

### System Health Endpoints

#### `GET /api/health`
Get comprehensive system health summary including all components and system metrics.

**Response:**
```json
{
  "health": {
    "system": {
      "status": "healthy",
      "health_score": 95.2,
      "uptime_seconds": 3600,
      "total_components": 9,
      "healthy_components": 8,
      "critical_components": 0,
      "degraded_components": 1
    },
    "components": {
      "analyst": {
        "status": "healthy",
        "health_score": 100.0,
        "issues": [],
        "models_loaded": 3
      }
    }
  },
  "system_metrics": {
    "cpu": {"usage_percent": 25.4},
    "memory": {"percent": 68.2},
    "disk": {"percent": 45.7}
  }
}
```

#### `GET /api/health/components`
Get health status for all registered components.

#### `GET /api/health/components/{component_name}`
Get detailed health status for a specific component.

#### `GET /api/health/sentinel`
Get Sentinel-specific health information including alerts and monitoring status.

#### `GET /api/health/system`
Get system-level metrics (CPU, memory, disk, network).

#### `GET /api/health/status`
Get quick health status summary for dashboard display.

### Health Management Endpoints

#### `POST /api/health/register/{component_name}`
Register a component for health monitoring.

#### `DELETE /api/health/register/{component_name}`
Unregister a component from health monitoring.

#### `GET /api/health/alerts`
Get current health-related alerts from Sentinel.

#### `POST /api/health/alerts/clear`
Clear all health-related alerts.

### Metrics Endpoint

#### `GET /api/metrics`
Get Prometheus metrics in text format for external monitoring systems.

## Health Status Levels

### Status Values
- **healthy** (90-100 score) - Component operating normally
- **warning** (70-89 score) - Minor issues detected
- **degraded** (50-69 score) - Performance issues present
- **critical** (1-49 score) - Major problems requiring attention
- **error** (0 score) - Component non-functional

### Health Score Calculation

Health scores are calculated based on multiple factors:

1. **Component Status** - Whether the component is running/active
2. **Error Rates** - Recent error counts and frequencies
3. **Performance Metrics** - Response times, throughput
4. **Resource Usage** - Memory, CPU consumption
5. **Data Freshness** - Timestamp of last successful operations
6. **Configuration Issues** - Missing or invalid configurations

## Integration Guide

### Adding Health Checks to Existing Components

1. **Import health check mixins:**
```python
from src.sentinel.health_checker import AnalystHealthMixin
```

2. **Enhance component class:**
```python
class MyAnalyst(AnalystHealthMixin):
    def __init__(self):
        super().__init__()
        # Component initialization
        
    # Component-specific health status is automatically available
```

3. **Register with health checker:**
```python
from src.sentinel.health_checker import health_checker

# Register component instance
health_checker.register_component("my_analyst", analyst_instance)
```

### Custom Health Check Implementation

```python
def get_health_status(self) -> Dict[str, Any]:
    """Custom health check implementation."""
    health_score = 100.0
    status = "healthy"
    issues = []
    
    # Check component-specific conditions
    if not self.is_initialized:
        health_score -= 50
        status = "critical"
        issues.append("Not initialized")
        
    # Check data freshness
    if self.last_update_time < time.time() - 300:  # 5 minutes
        health_score -= 20
        status = "warning" if status == "healthy" else status
        issues.append("Stale data")
        
    return {
        "component": "my_component",
        "status": status,
        "health_score": max(0, health_score),
        "issues": issues,
        "custom_metrics": {
            "last_update": self.last_update_time,
            "data_count": len(self.data)
        }
    }
```

## Prometheus Metrics

### Health Check Metrics

- `component_health_score` - Health score (0-100) per component
- `component_status` - Status level (0-4) per component  
- `health_check_duration_seconds` - Time spent on health checks
- `component_uptime_seconds` - Component uptime
- `alert_count_total` - Number of active alerts by component and severity

### System Metrics

- `memory_usage_bytes` - Memory usage per component
- `cpu_usage_percent` - CPU usage per component
- `step_execution_duration_seconds` - Step execution times
- `validation_passed_total` / `validation_failed_total` - Validation results

## Monitoring and Alerting

### Automatic Alerts

The system automatically generates alerts for:

- Component status changes (healthy → warning → critical)
- Health score drops below thresholds
- System resource usage exceeding limits
- Component unavailability
- Execution errors and failures

### Alert Severity Levels

- **LOW** - Minor issues, informational
- **MEDIUM** - Performance degradation
- **HIGH** - Critical component issues

### Alert Configuration

Configure alert thresholds in system configuration:

```python
config = {
    "sentinel": {
        "alert_threshold": 0.8,  # Health score threshold
        "monitoring_interval": 60,  # Check interval in seconds
        "max_alerts": 100,  # Maximum stored alerts
        "enable_performance_monitoring": True,
        "enable_error_monitoring": True,
        "enable_system_monitoring": True
    }
}
```

## Usage Examples

### Initialize Health Monitoring

```python
from src.sentinel.health_integration import initialize_health_monitoring

# Initialize the health monitoring system
await initialize_health_monitoring()
```

### Check Component Health

```python
from src.sentinel.health_checker import health_checker

# Check specific component
health_data = await health_checker.check_component_health("analyst")
print(f"Analyst status: {health_data['status']}")

# Check all components
all_health = await health_checker.check_all_components()
```

### Generate Health Dashboard

```python
from src.sentinel.health_integration import generate_system_health_dashboard

# Get comprehensive dashboard data
dashboard = await generate_system_health_dashboard()
print(f"Overall system status: {dashboard['system_overview']['status']}")
```

### Integration with Existing Components

```python
from src.sentinel.health_integration import integrate_health_check_with_component

# Enhance existing component with health checks
enhanced_analyst = integrate_health_check_with_component(
    analyst_instance, 
    "analyst"
)

# Now the component has health check capabilities
health_status = enhanced_analyst.get_health_status()
```

## Best Practices

### Component Health Implementation

1. **Check Critical Dependencies** - Verify required resources are available
2. **Monitor Performance Metrics** - Track response times and throughput
3. **Validate Configuration** - Ensure all required settings are present
4. **Check Data Freshness** - Verify data is up-to-date
5. **Monitor Error Rates** - Track recent errors and failures

### Health Score Guidelines

- Start with 100 points
- Deduct points based on severity:
  - Critical issues: -40 to -60 points
  - Major issues: -20 to -30 points  
  - Minor issues: -5 to -15 points
- Ensure scores never go below 0
- Use consistent scoring across components

### Monitoring Strategy

1. **Regular Health Checks** - Run checks every 60 seconds
2. **Immediate Alerts** - Critical issues trigger instant notifications
3. **Trend Analysis** - Track health scores over time
4. **Proactive Monitoring** - Identify issues before they become critical
5. **Dashboard Integration** - Display health status in management UI

## Troubleshooting

### Common Issues

1. **Component Not Registered**
   - Ensure component is registered with health_checker
   - Check component instance is passed correctly

2. **Health Check Failures**
   - Verify component implements health check methods
   - Check for exceptions in health check logic

3. **Metrics Not Appearing**
   - Confirm Prometheus server is running (port 8000)
   - Verify metrics are being recorded properly

4. **Alerts Not Triggering**
   - Check Sentinel is initialized and monitoring
   - Verify alert thresholds are configured correctly

### Debug Commands

```python
# Check registered components
print(health_checker.component_checkers.keys())

# Check Sentinel status
if sentinel:
    print(sentinel.get_sentinel_status())

# Get Prometheus metrics
from src.utils.prometheus_metrics import metrics
print(metrics.get_metrics())
```

## Future Enhancements

1. **Historical Trending** - Track health metrics over time
2. **Predictive Alerting** - Alert on trend degradation
3. **Auto-Recovery** - Automatic component restart on failures
4. **Advanced Analytics** - ML-based anomaly detection
5. **External Integration** - Push metrics to external monitoring systems