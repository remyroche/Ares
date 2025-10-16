#!/usr/bin/env python3

"""
Advanced Monitoring and Tracking System

This package provides comprehensive monitoring capabilities for the Ares trading bot,
including real-time metrics visualization, advanced tracing, ML monitoring,
automated reporting, and comprehensive tracking.
"""

# Enhanced ML Monitoring Components
from .enhanced_ml_monitoring import (
    EnhancedMLMonitor, TradeContext, TradingIndicator, MLModelDecision,
    EnsembleDecision, TradeDecision, TradingMode, ModelType,
    ModelPerformanceMetrics, EnsemblePerformanceMetrics
)

# Enhanced Monitoring Orchestrator
from .enhanced_monitoring_orchestrator import (
    EnhancedMonitoringOrchestrator, ComprehensiveTradeDecision,
    EnhancedMonitoringConfig, MonthlyReport
)

# Trade Decision Context Capture
from .trade_decision_capture import (
    TradeDecisionContextCapture, ComprehensiveTradeContext,
    MarketConditions, HMMRegimeContext, TradingSignalContext,
    ModelDecisionContext, EnsembleDecisionContext
)

# SHAP/LIME Integration
from .shap_lime_integration import (
    ExplainabilityIntegrator, SHAPAnalyzer, LIMEAnalyzer,
    SHAPExplanation, LIMEExplanation, ModelExplanationRequest
)

# Ensemble Monitoring
from .ensemble_monitor import (
    EnsembleMonitor, ModelWeight, EnsembleState, ModelContribution,
    EnsemblePerformanceSnapshot
)

# Daily Summary Tracker
from .daily_summary_tracker import (
    DailySummaryTracker, DailyTradeSummary, RegimePerformance
)

# Trading Integration
from .trading_integration import (
    TradingSystemIntegrator, TradingSystemConfig
)

# Trading Mode Monitoring Integration
from .trading_mode_monitoring_integration import (
    TradingModeMonitoringIntegration
)

# Auto Monitoring Launcher
from .auto_monitoring_launcher import (
    AutoMonitoringLauncher, launch_auto_monitoring, get_auto_monitoring, stop_auto_monitoring
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
    
    # Enhanced Monitoring Orchestrator
    "EnhancedMonitoringOrchestrator",
    "ComprehensiveTradeDecision",
    "EnhancedMonitoringConfig",
    "MonthlyReport",
    
    # Trade Decision Context Capture
    "TradeDecisionContextCapture",
    "ComprehensiveTradeContext",
    "MarketConditions",
    "HMMRegimeContext",
    "TradingSignalContext",
    "ModelDecisionContext",
    "EnsembleDecisionContext",
    
    # SHAP/LIME Integration
    "ExplainabilityIntegrator",
    "SHAPAnalyzer",
    "LIMEAnalyzer",
    "SHAPExplanation",
    "LIMEExplanation",
    "ModelExplanationRequest",
    
    # Ensemble Monitoring
    "EnsembleMonitor",
    "ModelWeight",
    "EnsembleState",
    "ModelContribution",
    "EnsemblePerformanceSnapshot",
    
    # Daily Summary Tracker
    "DailySummaryTracker",
    "DailyTradeSummary",
    "RegimePerformance",
    
    # Trading Integration
    "TradingSystemIntegrator",
    "TradingSystemConfig",
    
    # Trading Mode Monitoring Integration
    "TradingModeMonitoringIntegration",
    
    # Auto Monitoring Launcher
    "AutoMonitoringLauncher",
    "launch_auto_monitoring",
    "get_auto_monitoring",
    "stop_auto_monitoring",
    
    # GUI components
    "MonitoringDashboard",
    "EnhancedMonitoringDashboard",
    "MonitoringVisualization",
    "VisualizationControlPanel",
    "launch_dashboard",
]
