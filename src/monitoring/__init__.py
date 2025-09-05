#!/usr/bin/env python3

"""
Advanced Monitoring and Tracking System

This package provides comprehensive monitoring capabilities for the Ares trading bot,
including real-time metrics visualization, advanced tracing, ML monitoring,
automated reporting, and comprehensive tracking.
"""


# Enhanced ML Monitoring Components
from .monitors.enhanced_ml_monitor import (
    EnhancedMLMonitor, TradeContext, TradingIndicator, MLModelDecision,
    EnsembleDecision, TradeDecision, TradingMode, ModelType,
    ModelPerformanceMetrics, EnsemblePerformanceMetrics
)

# GUI components
from .gui.monitoring_dashboard import (
    MonitoringDashboard, EnhancedMonitoringDashboard,
    MonitoringVisualization, VisualizationControlPanel,
    launch_dashboard
)

__all__ = [
    # Original monitoring components
    "AdvancedTracer",
    "CorrelationManager",
    "MonitoringIntegrationManager",
    "MetricsDashboard",
    "MLMonitor",
    "ReportScheduler",
    "TrackingSystem",
    
    # Enhanced ML Monitoring components
    "EnhancedMLMonitor",
    "TradeContext",
    "TradingIndicator", 
    "MLModelDecision",
    "EnsembleDecision",
    "TradeDecision",
    "TradingMode",
    "ModelType",
    "ModelPerformanceMetrics",
    "EnsemblePerformanceMetrics",
    "ExplainabilityIntegrator",
    "EnsembleMonitor",
    "ModelContribution",
    "CSVExportManager",
    "TradingSystemIntegrator",
    "MonitoringOrchestrator",
    "create_monitoring_orchestrator",
    "DailySummaryTracker",
    "DailyTradeSummary",
    "RegimePerformance",
    
    # GUI components
    "MonitoringDashboard",
    "EnhancedMonitoringDashboard",
    "MonitoringVisualization",
    "VisualizationControlPanel",
    "launch_dashboard",
]
