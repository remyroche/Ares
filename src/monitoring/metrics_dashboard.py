#!/usr/bin/env python3
"""
Real-time Metrics Dashboard

This module provides comprehensive real-time metrics visualization for the Ares trading bot,
including performance metrics, model behavior, system health, and trading analytics.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors


class MetricType(Enum):
    """Metric types for categorization."""
    PERFORMANCE = "performance"
    MODEL_BEHAVIOR = "model_behavior"
    SYSTEM_HEALTH = "system_health"
    TRADING_ANALYTICS = "trading_analytics"
    RISK_METRICS = "risk_metrics"
    ENSEMBLE_METRICS = "ensemble_metrics"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    metric_name: str
    metric_type: MetricType
    value: float
    timestamp: datetime
    metadata: Dict[str, Any]
    unit: Optional[str] = None


@dataclass
class DashboardMetric:
    """Dashboard metric with aggregation."""
    metric_name: str
    metric_type: MetricType
    current_value: float
    previous_value: Optional[float]
    change_percentage: Optional[float]
    trend: str  # "up", "down", "stable"
    last_updated: datetime
    metadata: Dict[str, Any]
    unit: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics aggregation."""
    total_pnl: float
    daily_pnl: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_duration: float
    best_trade: float
    worst_trade: float


@dataclass
class ModelBehaviorMetrics:
    """Model behavior metrics."""
    model_accuracy: float
    prediction_confidence: float
    feature_importance_stability: float
    concept_drift_score: float
    data_drift_score: float
    model_performance_trend: List[float]
    last_retraining: Optional[datetime] = None


@dataclass
class SystemHealthMetrics:
    """System health metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    error_rate: float
    uptime: float
    active_connections: int


@dataclass
class TradingAnalytics:
    """Trading analytics metrics."""
    active_positions: int
    total_exposure: float
    portfolio_value: float
    margin_usage: float
    leverage: float
    risk_per_trade: float
    correlation_matrix: Dict[str, Dict[str, float]]


class MetricsDashboard:
    """
    Real-time metrics dashboard providing comprehensive visualization
    of trading bot performance, model behavior, and system health.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metrics dashboard.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("MetricsDashboard")
        
        # Dashboard configuration
        self.dashboard_config = config.get("metrics_dashboard", {})
        self.update_interval = self.dashboard_config.get("update_interval", 5)  # seconds
        self.max_metric_history = self.dashboard_config.get("max_metric_history", 1000)
        self.enable_real_time_updates = self.dashboard_config.get("enable_real_time_updates", True)
        self.enable_websocket_broadcast = self.dashboard_config.get("enable_websocket_broadcast", True)
        
        # Metric storage
        self.metrics_history: Dict[str, List[MetricPoint]] = {}
        self.current_metrics: Dict[str, DashboardMetric] = {}
        self.performance_metrics: Optional[PerformanceMetrics] = None
        self.model_behavior_metrics: Dict[str, ModelBehaviorMetrics] = {}
        self.system_health_metrics: Optional[SystemHealthMetrics] = None
        self.trading_analytics: Optional[TradingAnalytics] = None
        
        # Dashboard state
        self.is_running = False
        self.dashboard_task: Optional[asyncio.Task] = None
        self.websocket_connections: List[Any] = []
        
        # Metric aggregation
        self.aggregation_windows = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "1d": 86400,
        }
        
        self.logger.info("📊 Metrics Dashboard initialized")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid dashboard configuration"),
            AttributeError: (False, "Missing required dashboard parameters"),
        },
        default_return=False,
        context="dashboard initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the metrics dashboard."""
        try:
            self.logger.info("Initializing Metrics Dashboard...")
            
            # Initialize metric storage
            await self._initialize_metric_storage()
            
            # Initialize real-time updates
            if self.enable_real_time_updates:
                await self._initialize_real_time_updates()
            
            # Initialize websocket broadcasting
            if self.enable_websocket_broadcast:
                await self._initialize_websocket_broadcast()
            
            self.logger.info("✅ Metrics Dashboard initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Metrics Dashboard initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="metric storage initialization",
    )
    async def _initialize_metric_storage(self) -> None:
        """Initialize metric storage structures."""
        try:
            # Initialize metric history for each type
            for metric_type in MetricType:
                self.metrics_history[metric_type.value] = []
            
            self.logger.info("Metric storage initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing metric storage: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="real-time updates initialization",
    )
    async def _initialize_real_time_updates(self) -> None:
        """Initialize real-time metric updates."""
        try:
            self.dashboard_task = asyncio.create_task(self._dashboard_update_loop())
            self.logger.info("Real-time updates initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing real-time updates: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="websocket broadcast initialization",
    )
    async def _initialize_websocket_broadcast(self) -> None:
        """Initialize websocket broadcasting for real-time updates."""
        try:
            # This would integrate with the existing GUI websocket system
            self.logger.info("WebSocket broadcasting initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing websocket broadcast: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="dashboard update loop",
    )
    async def _dashboard_update_loop(self) -> None:
        """Main dashboard update loop."""
        try:
            while self.is_running:
                await self._update_all_metrics()
                await self._broadcast_metrics()
                await asyncio.sleep(self.update_interval)
                
        except Exception as e:
            self.logger.error(f"Error in dashboard update loop: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="metrics update",
    )
    async def _update_all_metrics(self) -> None:
        """Update all dashboard metrics."""
        try:
            # Update performance metrics
            await self._update_performance_metrics()
            
            # Update model behavior metrics
            await self._update_model_behavior_metrics()
            
            # Update system health metrics
            await self._update_system_health_metrics()
            
            # Update trading analytics
            await self._update_trading_analytics()
            
            # Update current metrics
            await self._update_current_metrics()
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance metrics update",
    )
    async def _update_performance_metrics(self) -> None:
        """Update performance metrics."""
        try:
            # This would integrate with the existing performance monitor
            # For now, create sample metrics
            self.performance_metrics = PerformanceMetrics(
                total_pnl=0.0,
                daily_pnl=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                avg_trade_duration=0.0,
                best_trade=0.0,
                worst_trade=0.0,
            )
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="model behavior metrics update",
    )
    async def _update_model_behavior_metrics(self) -> None:
        """Update model behavior metrics."""
        try:
            # This would integrate with the enhanced model monitor
            # For now, create sample metrics
            for model_id in ["ensemble_1", "ensemble_2", "meta_learner"]:
                self.model_behavior_metrics[model_id] = ModelBehaviorMetrics(
                    model_accuracy=0.0,
                    prediction_confidence=0.0,
                    feature_importance_stability=0.0,
                    concept_drift_score=0.0,
                    data_drift_score=0.0,
                    model_performance_trend=[0.0],
                    last_retraining=None,
                )
            
        except Exception as e:
            self.logger.error(f"Error updating model behavior metrics: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="system health metrics update",
    )
    async def _update_system_health_metrics(self) -> None:
        """Update system health metrics."""
        try:
            # This would integrate with system monitoring
            # For now, create sample metrics
            self.system_health_metrics = SystemHealthMetrics(
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                network_latency=0.0,
                error_rate=0.0,
                uptime=0.0,
                active_connections=0,
            )
            
        except Exception as e:
            self.logger.error(f"Error updating system health metrics: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="trading analytics update",
    )
    async def _update_trading_analytics(self) -> None:
        """Update trading analytics."""
        try:
            # This would integrate with portfolio management
            # For now, create sample metrics
            self.trading_analytics = TradingAnalytics(
                active_positions=0,
                total_exposure=0.0,
                portfolio_value=0.0,
                margin_usage=0.0,
                leverage=0.0,
                risk_per_trade=0.0,
                correlation_matrix={},
            )
            
        except Exception as e:
            self.logger.error(f"Error updating trading analytics: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="current metrics update",
    )
    async def _update_current_metrics(self) -> None:
        """Update current dashboard metrics."""
        try:
            now = datetime.now()
            
            # Update performance metrics
            if self.performance_metrics:
                self._update_metric("total_pnl", MetricType.PERFORMANCE, 
                                  self.performance_metrics.total_pnl, now)
                self._update_metric("win_rate", MetricType.PERFORMANCE, 
                                  self.performance_metrics.win_rate, now)
                self._update_metric("sharpe_ratio", MetricType.PERFORMANCE, 
                                  self.performance_metrics.sharpe_ratio, now)
            
            # Update system health metrics
            if self.system_health_metrics:
                self._update_metric("cpu_usage", MetricType.SYSTEM_HEALTH, 
                                  self.system_health_metrics.cpu_usage, now)
                self._update_metric("memory_usage", MetricType.SYSTEM_HEALTH, 
                                  self.system_health_metrics.memory_usage, now)
                self._update_metric("error_rate", MetricType.SYSTEM_HEALTH, 
                                  self.system_health_metrics.error_rate, now)
            
            # Update trading analytics
            if self.trading_analytics:
                self._update_metric("active_positions", MetricType.TRADING_ANALYTICS, 
                                  self.trading_analytics.active_positions, now)
                self._update_metric("total_exposure", MetricType.TRADING_ANALYTICS, 
                                  self.trading_analytics.total_exposure, now)
            
        except Exception as e:
            self.logger.error(f"Error updating current metrics: {e}")

    def _update_metric(self, metric_name: str, metric_type: MetricType, 
                      value: float, timestamp: datetime) -> None:
        """Update a single metric."""
        try:
            # Create metric point
            metric_point = MetricPoint(
                metric_name=metric_name,
                metric_type=metric_type,
                value=value,
                timestamp=timestamp,
                metadata={},
            )
            
            # Add to history
            history_key = metric_type.value
            if history_key not in self.metrics_history:
                self.metrics_history[history_key] = []
            
            self.metrics_history[history_key].append(metric_point)
            
            # Limit history size
            if len(self.metrics_history[history_key]) > self.max_metric_history:
                self.metrics_history[history_key] = self.metrics_history[history_key][-self.max_metric_history:]
            
            # Update current metric
            previous_value = None
            change_percentage = None
            trend = "stable"
            
            if len(self.metrics_history[history_key]) > 1:
                previous_point = self.metrics_history[history_key][-2]
                previous_value = previous_point.value
                
                if previous_value != 0:
                    change_percentage = ((value - previous_value) / abs(previous_value)) * 100
                    
                    if change_percentage > 1:
                        trend = "up"
                    elif change_percentage < -1:
                        trend = "down"
            
            self.current_metrics[metric_name] = DashboardMetric(
                metric_name=metric_name,
                metric_type=metric_type,
                current_value=value,
                previous_value=previous_value,
                change_percentage=change_percentage,
                trend=trend,
                last_updated=timestamp,
                metadata={},
            )
            
        except Exception as e:
            self.logger.error(f"Error updating metric {metric_name}: {e}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="metrics broadcast",
    )
    async def _broadcast_metrics(self) -> None:
        """Broadcast metrics to connected clients."""
        try:
            if not self.enable_websocket_broadcast:
                return
            
            dashboard_data = self.get_dashboard_data()
            
            # Broadcast to websocket connections
            for connection in self.websocket_connections:
                try:
                    await connection.send_text(json.dumps(dashboard_data))
                except Exception as e:
                    self.logger.error(f"Error broadcasting to connection: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error broadcasting metrics: {e}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        try:
            return {
                "timestamp": datetime.now().isoformat(),
                "performance_metrics": asdict(self.performance_metrics) if self.performance_metrics else {},
                "model_behavior_metrics": {k: asdict(v) for k, v in self.model_behavior_metrics.items()},
                "system_health_metrics": asdict(self.system_health_metrics) if self.system_health_metrics else {},
                "trading_analytics": asdict(self.trading_analytics) if self.trading_analytics else {},
                "current_metrics": {k: asdict(v) for k, v in self.current_metrics.items()},
            }
        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {e}")
            return {}

    def get_metric_history(self, metric_name: str, 
                          window: str = "1h") -> List[Dict[str, Any]]:
        """Get metric history for a specific metric."""
        try:
            history = []
            window_seconds = self.aggregation_windows.get(window, 3600)
            cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
            
            for metric_type, points in self.metrics_history.items():
                for point in points:
                    if point.metric_name == metric_name and point.timestamp >= cutoff_time:
                        history.append(asdict(point))
            
            return sorted(history, key=lambda x: x["timestamp"])
            
        except Exception as e:
            self.logger.error(f"Error getting metric history: {e}")
            return []

    @handle_specific_errors(
        error_handlers={
            Exception: (False, "Dashboard start failed"),
        },
        default_return=False,
        context="dashboard start",
    )
    async def start(self) -> bool:
        """Start the metrics dashboard."""
        try:
            self.is_running = True
            self.logger.info("🚀 Metrics Dashboard started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting dashboard: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="dashboard stop",
    )
    async def stop(self) -> None:
        """Stop the metrics dashboard."""
        try:
            self.is_running = False
            
            if self.dashboard_task:
                self.dashboard_task.cancel()
                try:
                    await self.dashboard_task
                except asyncio.CancelledError:
                    pass
            
            self.logger.info("🛑 Metrics Dashboard stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping dashboard: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="metrics dashboard setup",
)
async def setup_metrics_dashboard(config: Dict[str, Any]) -> MetricsDashboard | None:
    """
    Setup and initialize metrics dashboard.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MetricsDashboard instance or None if setup failed
    """
    try:
        dashboard = MetricsDashboard(config)
        
        if await dashboard.initialize():
            return dashboard
        else:
            return None
            
    except Exception as e:
        system_logger.error(f"Error setting up metrics dashboard: {e}")
        return None 