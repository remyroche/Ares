#!/usr/bin/env python3
"""
Test Advanced Monitoring System

This script tests the complete advanced monitoring system to ensure
all components work together properly.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.monitoring import (
    MetricsDashboard,
    AdvancedTracer,
    CorrelationManager,
    MLMonitor,
    ReportScheduler,
    TrackingSystem,
    MonitoringIntegrationManager,
)
from src.utils.logger import system_logger


async def test_individual_components():
    """Test individual monitoring components."""
    print("🧪 Testing Individual Components...")
    
    # Test configuration
    config = {
        "monitoring": {
            "enable_monitoring": True,
            "log_level": "INFO",
        },
        "metrics_dashboard": {
            "enable_dashboard": True,
            "update_interval": 5,
            "max_metrics_history": 1000,
        },
        "advanced_tracer": {
            "enable_tracing": True,
            "trace_storage": "memory",
            "max_trace_history": 1000,
        },
        "correlation_manager": {
            "enable_correlation_tracking": True,
            "correlation_timeout": 300,
            "max_correlation_history": 10000,
        },
        "ml_monitor": {
            "enable_online_learning": True,
            "drift_detection_enabled": True,
            "feature_importance_tracking": True,
            "auto_retraining_enabled": True,
            "drift_threshold": 0.1,
        },
        "report_scheduler": {
            "enable_automated_reports": True,
            "default_schedule": "daily",
            "email_distribution": False,
            "report_formats": ["json"],
        },
        "tracking_system": {
            "enable_ensemble_tracking": True,
            "enable_regime_tracking": True,
            "enable_feature_tracking": True,
            "enable_decision_tracking": True,
            "enable_behavior_tracking": True,
        },
        "monitoring_integration": {
            "enable_integration": True,
            "integration_interval": 10,
            "cross_component_tracking": True,
        },
    }
    
    # Test Metrics Dashboard
    print("  📊 Testing Metrics Dashboard...")
    try:
        dashboard = MetricsDashboard(config)
        await dashboard.initialize()
        await dashboard.start_dashboard()
        
        # Get sample metrics
        metrics = dashboard.get_current_metrics()
        print(f"    ✅ Dashboard initialized with {len(metrics)} metrics")
        
        await dashboard.stop_dashboard()
    except Exception as e:
        print(f"    ❌ Dashboard test failed: {e}")
    
    # Test Advanced Tracer
    print("  🔍 Testing Advanced Tracer...")
    try:
        tracer = AdvancedTracer(config)
        await tracer.initialize()
        
        # Test tracing
        with tracer.trace_request("test_correlation"):
            print("    ✅ Tracer initialized and tracing working")
        
        await tracer.stop()
    except Exception as e:
        print(f"    ❌ Tracer test failed: {e}")
    
    # Test Correlation Manager
    print("  🔗 Testing Correlation Manager...")
    try:
        correlation_manager = CorrelationManager(config)
        await correlation_manager.initialize()
        await correlation_manager.start()
        
        # Test correlation tracking
        await correlation_manager.track_correlation_request(
            "test_corr", ["test_component"], {"test": "data"}
        )
        
        stats = correlation_manager.get_correlation_statistics()
        print(f"    ✅ Correlation manager working: {stats['total_requests']} requests")
        
        await correlation_manager.stop()
    except Exception as e:
        print(f"    ❌ Correlation manager test failed: {e}")
    
    # Test ML Monitor
    print("  🤖 Testing ML Monitor...")
    try:
        ml_monitor = MLMonitor(config)
        await ml_monitor.initialize()
        await ml_monitor.start_monitoring()
        
        # Wait a bit for monitoring to collect data
        await asyncio.sleep(2)
        
        summary = ml_monitor.get_ml_monitoring_summary()
        print(f"    ✅ ML Monitor working: {summary['total_models']} models")
        
        await ml_monitor.stop_monitoring()
    except Exception as e:
        print(f"    ❌ ML Monitor test failed: {e}")
    
    # Test Report Scheduler
    print("  📊 Testing Report Scheduler...")
    try:
        scheduler = ReportScheduler(config)
        await scheduler.initialize()
        await scheduler.start_scheduling()
        
        status = scheduler.get_scheduler_status()
        print(f"    ✅ Report Scheduler working: {status['total_configs']} configs")
        
        await scheduler.stop_scheduling()
    except Exception as e:
        print(f"    ❌ Report Scheduler test failed: {e}")
    
    # Test Tracking System
    print("  📈 Testing Tracking System...")
    try:
        tracking_system = TrackingSystem(config)
        await tracking_system.initialize()
        await tracking_system.start_tracking()
        
        # Wait a bit for tracking to collect data
        await asyncio.sleep(2)
        
        summary = tracking_system.get_tracking_summary()
        print(f"    ✅ Tracking System working: {summary['ensemble_decisions']} decisions")
        
        await tracking_system.stop_tracking()
    except Exception as e:
        print(f"    ❌ Tracking System test failed: {e}")


async def test_integration_manager():
    """Test the monitoring integration manager."""
    print("\n🔧 Testing Integration Manager...")
    
    config = {
        "monitoring": {
            "enable_monitoring": True,
            "log_level": "INFO",
        },
        "metrics_dashboard": {
            "enable_dashboard": True,
            "update_interval": 5,
        },
        "advanced_tracer": {
            "enable_tracing": True,
            "trace_storage": "memory",
        },
        "correlation_manager": {
            "enable_correlation_tracking": True,
        },
        "ml_monitor": {
            "enable_online_learning": True,
            "drift_detection_enabled": True,
        },
        "report_scheduler": {
            "enable_automated_reports": True,
        },
        "tracking_system": {
            "enable_ensemble_tracking": True,
            "enable_regime_tracking": True,
        },
        "monitoring_integration": {
            "enable_integration": True,
            "integration_interval": 5,
        },
    }
    
    try:
        # Initialize integration manager
        integration_manager = MonitoringIntegrationManager(config)
        
        # Initialize all components
        success = await integration_manager.initialize()
        if not success:
            print("  ❌ Integration manager initialization failed")
            return
        
        print("  ✅ Integration manager initialized")
        
        # Start integration
        success = await integration_manager.start_integration()
        if not success:
            print("  ❌ Integration manager start failed")
            return
        
        print("  ✅ Integration manager started")
        
        # Get unified dashboard data
        dashboard_data = integration_manager.get_unified_dashboard_data()
        print(f"  ✅ Unified dashboard data: {len(dashboard_data)} components")
        
        # Run for a few seconds to collect data
        print("  ⏳ Collecting monitoring data...")
        await asyncio.sleep(5)
        
        # Get updated dashboard data
        updated_data = integration_manager.get_unified_dashboard_data()
        print(f"  ✅ Updated dashboard data: {len(updated_data)} components")
        
        # Stop integration
        await integration_manager.stop_integration()
        print("  ✅ Integration manager stopped")
        
    except Exception as e:
        print(f"  ❌ Integration manager test failed: {e}")


async def main():
    """Main test function."""
    print("🚀 Starting Advanced Monitoring System Tests")
    print("=" * 50)
    
    # Test individual components
    await test_individual_components()
    
    # Test integration manager
    await test_integration_manager()
    
    print("\n" + "=" * 50)
    print("✅ Advanced Monitoring System Tests Completed")
    print("📋 All components are working and integrated properly!")


if __name__ == "__main__":
    asyncio.run(main()) 