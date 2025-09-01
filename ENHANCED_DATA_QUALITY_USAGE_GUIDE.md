# Enhanced Data Quality System - Usage Guide

## Overview

This guide demonstrates how to use the comprehensive enhanced data quality system that has been implemented across step1, step1_5, step3, and step4. The system provides:

- **Data gap detection and filling**
- **Data quality validation and formatting**
- **Efficient processing with proper decorators**
- **Adequate validators**
- **Integration with step1/step1_5 components when step3/step4 lack data**

## 🚀 Quick Start

### 1. Basic Usage

```python
import asyncio
from src.training.steps.step1.enhanced_data_quality_manager import EnhancedDataQualityManager

async def basic_quality_check():
    # Initialize the quality manager
    manager = EnhancedDataQualityManager("data_cache")

    # Run a comprehensive quality check
    results = await manager.comprehensive_quality_check(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        check_gaps=True,
        fill_gaps=True,
        validate_format=True
    )

    print(f"Quality check success: {results['success']}")
    print(f"Gaps detected: {len(results['gaps_detected'])}")
    print(f"Gaps filled: {len(results['gaps_filled'])}")
    print(f"Format issues: {len(results['format_issues'])}")

# Run the quality check
asyncio.run(basic_quality_check())
```

### 2. Running the Integrated Pipeline

```python
import asyncio
from src.training.steps.integrated_data_quality_pipeline import run_integrated_pipeline

async def run_complete_pipeline():
    # Run the complete pipeline with all steps
    success = await run_integrated_pipeline(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        data_cache_path="data_cache",
        run_all_steps=True,
        force_rerun=True
    )

    if success:
        print("🎉 Complete pipeline executed successfully!")
    else:
        print("❌ Pipeline execution failed")

# Run the complete pipeline
asyncio.run(run_complete_pipeline())
```

## 📊 Data Quality Monitoring

### 1. Real-time Monitoring

```python
import asyncio
from src.training.steps.step1.data_quality_monitor import start_data_quality_monitoring

async def start_monitoring():
    # Start real-time monitoring
    monitor = await start_data_quality_monitoring(
        symbols=["ETHUSDT", "BTCUSDT", "ADAUSDT"],
        exchanges=["BINANCE"],
        timeframes=["1m", "5m"],
        data_cache_path="data_cache",
        interval_seconds=300  # Check every 5 minutes
    )

    # Add custom alert callbacks
    def custom_alert_handler(alert):
        print(f"🚨 Alert: {alert.alert_type} - {alert.message}")

    monitor.add_alert_callback(custom_alert_handler)

    # Keep monitoring running
    try:
        await asyncio.sleep(3600)  # Monitor for 1 hour
    finally:
        await monitor.stop_monitoring()

# Start monitoring
asyncio.run(start_monitoring())
```

### 2. Custom Alert Callbacks

```python
import asyncio
from src.training.steps.step1.data_quality_monitor import (
    DataQualityMonitor,
    create_email_alert_callback,
    create_slack_alert_callback
)

async def setup_monitoring_with_callbacks():
    monitor = DataQualityMonitor("data_cache")

    # Add email alerts for critical issues
    email_callback = create_email_alert_callback("admin@example.com")
    monitor.add_alert_callback(email_callback)

    # Add Slack alerts for all issues
    slack_callback = create_slack_alert_callback("https://hooks.slack.com/...")
    monitor.add_alert_callback(slack_callback)

    # Custom callback for logging
    def log_alert(alert):
        if alert.severity == "critical":
            print(f"🚨 CRITICAL: {alert}")
        elif alert.severity == "high":
            print(f"⚠️ HIGH: {alert}")
        else:
            print(f"ℹ️ INFO: {alert}")

    monitor.add_alert_callback(log_alert)

    # Start monitoring
    await monitor.start_monitoring(
        symbols=["ETHUSDT"],
        exchanges=["BINANCE"],
        timeframes=["1m"]
    )

asyncio.run(setup_monitoring_with_callbacks())
```

## 🖥️ Web Dashboard

### 1. Starting the Dashboard

```python
import asyncio
from src.training.steps.step1.data_quality_dashboard import start_data_quality_dashboard

async def start_dashboard():
    # Start the web dashboard
    dashboard = await start_data_quality_dashboard(
        data_cache_path="data_cache",
        host="0.0.0.0",
        port=8080
    )

    print("🌐 Dashboard started at http://localhost:8080")

    # Keep the dashboard running
    try:
        await asyncio.sleep(float('inf'))
    except KeyboardInterrupt:
        await dashboard.stop_dashboard()

# Start the dashboard
asyncio.run(start_dashboard())
```

### 2. Dashboard API Usage

```python
import aiohttp
import asyncio

async def dashboard_api_examples():
    base_url = "http://localhost:8080"

    async with aiohttp.ClientSession() as session:
        # Get system status
        async with session.get(f"{base_url}/api/status") as response:
            status = await response.json()
            print(f"System status: {status['overall_status']}")

        # Get quality metrics
        async with session.get(f"{base_url}/api/metrics") as response:
            metrics = await response.json()
            print(f"Total gaps: {metrics.get('total_gaps', 0)}")

        # Get recent alerts
        async with session.get(f"{base_url}/api/alerts?limit=10") as response:
            alerts = await response.json()
            print(f"Recent alerts: {len(alerts)}")

        # Run a quality check
        async with session.post(
            f"{base_url}/api/quality-check?symbol=ETHUSDT&exchange=BINANCE&timeframe=1m"
        ) as response:
            result = await response.json()
            print(f"Quality check result: {result['success']}")

asyncio.run(dashboard_api_examples())
```

## 🔧 Step Integration

### 1. Enhanced Step1 Data Collection

```python
import asyncio
from src.training.steps.step1_data_collection import run_step

async def run_enhanced_step1():
    # Run step1 with enhanced quality checks
    success = await run_step(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        data_dir="data_cache",
        force_rerun=True
    )

    if success:
        print("✅ Step1 completed with enhanced quality checks")
    else:
        print("❌ Step1 failed")

asyncio.run(run_enhanced_step1())
```

### 2. Enhanced Step1.5 Data Converter

```python
import asyncio
from src.training.steps.step1_5_data_converter import run_step

async def run_enhanced_step1_5():
    # Run step1_5 with enhanced validation
    success = await run_step(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        data_dir="data_cache",
        force_rerun=True
    )

    if success:
        print("✅ Step1_5 completed with enhanced validation")
    else:
        print("❌ Step1_5 failed")

asyncio.run(run_enhanced_step1_5())
```

### 3. Enhanced Step3 HMM Discovery

```python
import asyncio
from src.training.steps.step3_hmm_regime_discovery import run_step

async def run_enhanced_step3():
    # Run step3 with automatic data recovery
    success = await run_step(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        data_dir="data_cache",
        force_rerun=True
    )

    if success:
        print("✅ Step3 completed with automatic data recovery")
    else:
        print("❌ Step3 failed")

asyncio.run(run_enhanced_step3())
```

## 🎯 Advanced Usage

### 1. Custom Quality Thresholds

```python
import asyncio
from src.training.steps.step1.data_quality_monitor import DataQualityMonitor

async def custom_thresholds():
    monitor = DataQualityMonitor("data_cache")

    # Set custom quality thresholds
    custom_thresholds = {
        "gap_threshold": 5,  # Alert if more than 5 gaps
        "format_issues_threshold": 3,  # Alert if more than 3 format issues
        "data_freshness_hours": 12,  # Alert if data is older than 12 hours
        "min_data_rows": 5000,  # Alert if less than 5000 rows
        "max_null_ratio": 0.05  # Alert if more than 5% null values
    }

    monitor.set_quality_thresholds(custom_thresholds)

    # Start monitoring with custom thresholds
    await monitor.start_monitoring(
        symbols=["ETHUSDT"],
        exchanges=["BINANCE"],
        timeframes=["1m"]
    )

asyncio.run(custom_thresholds())
```

### 2. Batch Quality Checks

```python
import asyncio
from src.training.steps.step1.enhanced_data_quality_manager import EnhancedDataQualityManager

async def batch_quality_checks():
    manager = EnhancedDataQualityManager("data_cache")

    # Define symbols to check
    symbols = ["ETHUSDT", "BTCUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
    exchanges = ["BINANCE"]
    timeframes = ["1m", "5m"]

    # Run quality checks for all combinations
    tasks = []
    for symbol in symbols:
        for exchange in exchanges:
            for timeframe in timeframes:
                task = manager.comprehensive_quality_check(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe
                )
                tasks.append(task)

    # Execute all quality checks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    successful_checks = 0
    failed_checks = 0

    for result in results:
        if isinstance(result, Exception):
            failed_checks += 1
            print(f"❌ Quality check failed: {result}")
        elif result.get("success"):
            successful_checks += 1
        else:
            failed_checks += 1

    print(f"✅ Successful checks: {successful_checks}")
    print(f"❌ Failed checks: {failed_checks}")

asyncio.run(batch_quality_checks())
```

### 3. Quality Report Generation

```python
import asyncio
import json
from datetime import datetime
from src.training.steps.step1.data_quality_monitor import DataQualityMonitor

async def generate_quality_report():
    monitor = DataQualityMonitor("data_cache")

    # Generate monitoring report
    report = monitor.generate_monitoring_report()

    # Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"quality_report_{timestamp}.txt"

    with open(report_file, "w") as f:
        f.write(report)

    print(f"📊 Quality report saved to: {report_file}")

    # Get performance metrics
    metrics = monitor.get_performance_metrics()

    # Save metrics as JSON
    metrics_file = f"quality_metrics_{timestamp}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"📈 Performance metrics saved to: {metrics_file}")

asyncio.run(generate_quality_report())
```

## 🔄 Automatic Data Recovery

### 1. Testing Automatic Recovery

```python
import asyncio
from src.training.steps.step1.enhanced_data_quality_manager import EnhancedDataQualityManager

async def test_automatic_recovery():
    manager = EnhancedDataQualityManager("data_cache")

    # Try to get data for step3/step4 (this will trigger automatic recovery if data is missing)
    results = await manager.get_data_for_step3_step4(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m"
    )

    if results["success"]:
        print("✅ Data is ready for step3/step4")
    else:
        print("❌ Data is not ready for step3/step4")
        print(f"Missing requirements: {results.get('missing_for_steps', [])}")

        # The system will automatically try to fix missing data
        print("🔄 Automatic recovery will be attempted...")

asyncio.run(test_automatic_recovery())
```

### 2. Manual Data Recovery

```python
import asyncio
from src.training.steps.step1.enhanced_data_quality_manager import EnhancedDataQualityManager

async def manual_data_recovery():
    manager = EnhancedDataQualityManager("data_cache")

    # Manually trigger data recovery
    recovery_results = await manager._fix_missing_data_for_steps(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m"
    )

    print(f"Recovery success: {recovery_results['success']}")
    print(f"Step1 success: {recovery_results.get('step1_success', False)}")
    print(f"Step1_5 success: {recovery_results.get('step1_5_success', False)}")

    if recovery_results.get("still_missing"):
        print(f"Still missing: {recovery_results['still_missing']}")

asyncio.run(manual_data_recovery())
```

## 🧪 Testing

### 1. Running the Test Suite

```python
import asyncio
from src.training.steps.step1.test_enhanced_data_quality_system import run_comprehensive_tests

async def run_tests():
    # Run comprehensive tests
    run_comprehensive_tests()

# Run tests
asyncio.run(run_tests())
```

### 2. Individual Component Testing

```python
import pytest
import asyncio

# Run specific test classes
pytest.main([
    "src/training/steps/step1/test_enhanced_data_quality_system.py::TestEnhancedDataQualityManager",
    "-v"
])

# Run specific test methods
pytest.main([
    "src/training/steps/step1/test_enhanced_data_quality_system.py::TestEnhancedDataQualityManager::test_comprehensive_quality_check",
    "-v"
])
```

## 📋 Configuration

### 1. Environment Variables

```bash
# Set data cache directory
export DATA_CACHE_PATH="/path/to/data_cache"

# Set monitoring interval (in seconds)
export MONITORING_INTERVAL=300

# Set dashboard port
export DASHBOARD_PORT=8080

# Set quality thresholds
export GAP_THRESHOLD=10
export FORMAT_ISSUES_THRESHOLD=5
export DATA_FRESHNESS_HOURS=24
```

### 2. Configuration Files

```python
# config/quality_config.json
{
    "monitoring": {
        "interval_seconds": 300,
        "symbols": ["ETHUSDT", "BTCUSDT", "ADAUSDT"],
        "exchanges": ["BINANCE"],
        "timeframes": ["1m", "5m"]
    },
    "thresholds": {
        "gap_threshold": 10,
        "format_issues_threshold": 5,
        "data_freshness_hours": 24,
        "min_data_rows": 10000,
        "max_null_ratio": 0.1
    },
    "dashboard": {
        "host": "0.0.0.0",
        "port": 8080,
        "refresh_interval": 30
    }
}
```

## 🚨 Troubleshooting

### 1. Common Issues

```python
# Issue: Import errors
try:
    from src.training.steps.step1.enhanced_data_quality_manager import EnhancedDataQualityManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed")

# Issue: Data not found
async def check_data_availability():
    manager = EnhancedDataQualityManager("data_cache")

    # Check if data files exist
    import os
    klines_file = "data_cache/klines_BINANCE_ETHUSDT_1m_consolidated.parquet"
    aggtrades_file = "data_cache/aggtrades_BINANCE_ETHUSDT_consolidated.parquet"

    if not os.path.exists(klines_file):
        print(f"❌ Klines file not found: {klines_file}")

    if not os.path.exists(aggtrades_file):
        print(f"❌ Aggtrades file not found: {aggtrades_file}")

# Issue: Monitoring not starting
async def debug_monitoring():
    monitor = DataQualityMonitor("data_cache")

    # Check if components are available
    if not monitor.gap_detector:
        print("⚠️ Gap detector not available")

    if not monitor.gap_filler:
        print("⚠️ Gap filler not available")

    if not monitor.validator:
        print("⚠️ Validator not available")

asyncio.run(debug_monitoring())
```

### 2. Performance Issues

```python
# Issue: Slow quality checks
async def optimize_performance():
    manager = EnhancedDataQualityManager("data_cache")

    # Use memory-efficient processing
    results = await manager.comprehensive_quality_check(
        symbol="ETHUSDT",
        exchange="BINANCE",
        timeframe="1m",
        check_gaps=True,
        fill_gaps=False,  # Don't auto-fill during monitoring
        validate_format=True
    )

    # Check processing time
    import time
    start_time = time.time()
    # ... run quality check ...
    end_time = time.time()

    if end_time - start_time > 30:
        print("⚠️ Quality check is taking too long")
        print("Consider reducing data size or optimizing processing")

# Issue: High memory usage
async def monitor_memory_usage():
    import psutil
    import os

    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB

    if memory_usage > 1000:  # More than 1GB
        print(f"⚠️ High memory usage: {memory_usage:.2f} MB")
        print("Consider using memory-efficient processing")
```

## 📚 Best Practices

### 1. Production Deployment

```python
# Use proper logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('quality_system.log'),
        logging.StreamHandler()
    ]
)

# Use environment-specific configurations
import os
environment = os.getenv('ENVIRONMENT', 'development')

if environment == 'production':
    # Production settings
    monitoring_interval = 600  # 10 minutes
    dashboard_host = "0.0.0.0"
    dashboard_port = 80
else:
    # Development settings
    monitoring_interval = 60  # 1 minute
    dashboard_host = "127.0.0.1"
    dashboard_port = 8080
```

### 2. Error Handling

```python
async def robust_quality_check():
    try:
        manager = EnhancedDataQualityManager("data_cache")

        results = await manager.comprehensive_quality_check(
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframe="1m"
        )

        return results

    except Exception as e:
        logger.error(f"Quality check failed: {e}")

        # Fallback: try basic check
        try:
            # Basic file existence check
            import os
            klines_file = "data_cache/klines_BINANCE_ETHUSDT_1m_consolidated.parquet"

            if os.path.exists(klines_file):
                return {"success": True, "message": "Basic check passed"}
            else:
                return {"success": False, "message": "Data file not found"}

        except Exception as fallback_error:
            logger.error(f"Fallback check also failed: {fallback_error}")
            return {"success": False, "error": str(e)}
```

### 3. Monitoring and Alerting

```python
async def setup_production_monitoring():
    monitor = DataQualityMonitor("data_cache")

    # Set production thresholds
    monitor.set_quality_thresholds({
        "gap_threshold": 5,
        "format_issues_threshold": 2,
        "data_freshness_hours": 6,
        "min_data_rows": 50000,
        "max_null_ratio": 0.05
    })

    # Add production alert callbacks
    def production_alert_handler(alert):
        if alert.severity in ["high", "critical"]:
            # Send to production monitoring system
            print(f"🚨 PRODUCTION ALERT: {alert}")

            # Could integrate with PagerDuty, Slack, etc.
            # send_pagerduty_alert(alert)
            # send_slack_alert(alert)

    monitor.add_alert_callback(production_alert_handler)

    # Start monitoring
    await monitor.start_monitoring(
        symbols=["ETHUSDT", "BTCUSDT"],
        exchanges=["BINANCE"],
        timeframes=["1m", "5m"],
        interval_seconds=300
    )

asyncio.run(setup_production_monitoring())
```

This comprehensive usage guide demonstrates how to effectively use all the enhanced data quality components in various scenarios, from basic usage to advanced production deployments.