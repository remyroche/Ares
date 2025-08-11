# src/monitoring/performance_monitor.py

"""
Performance Monitor for Dual Model System
Comprehensive monitoring of model performance, system metrics, trading performance, and optimization opportunities.
"""

import asyncio
import json
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import psutil

from src.monitoring.csv_exporter import CSVExporter
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
    execution_error,
    initialization_error,
    warning,
)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""

    timestamp: datetime = field(default_factory=datetime.now)
    model_accuracy: float = 0.0
    model_precision: float = 0.0
    model_recall: float = 0.0
    model_f1_score: float = 0.0
    model_auc: float = 0.0
    trading_win_rate: float = 0.0
    trading_profit_factor: float = 0.0
    trading_sharpe_ratio: float = 0.0
    trading_max_drawdown: float = 0.0
    trading_total_return: float = 0.0
    system_memory_usage: float = 0.0
    system_cpu_usage: float = 0.0
    system_response_time: float = 0.0
    system_throughput: float = 0.0
    confidence_analyst: float = 0.0
    confidence_tactician: float = 0.0
    confidence_final: float = 0.0
    order_execution_success_rate: float = 0.0
    order_execution_slippage: float = 0.0
    training_performance: float = 0.0
    training_duration: float = 0.0
    feature_engineering_performance: float = 0.0
    meta_labeling_performance: float = 0.0

    # Risk management metrics
    portfolio_var: float = 0.0
    portfolio_correlation: float = 0.0
    portfolio_concentration: float = 0.0
    portfolio_leverage: float = 1.0
    position_count: int = 0
    max_position_size: float = 0.0
    avg_position_size: float = 0.0
    position_duration: float = 0.0
    market_volatility: float = 0.0
    market_liquidity: float = 0.0
    market_stress: float = 0.0
    market_regime: str = "unknown"


@dataclass
class OptimizationOpportunity:
    """Optimization opportunity data structure."""

    timestamp: datetime = field(default_factory=datetime.now)
    category: str = ""
    metric: str = ""
    current_value: float = 0.0
    target_value: float = 0.0
    improvement_potential: float = 0.0
    priority: str = "low"  # low, medium, high, critical
    description: str = ""
    recommended_action: str = ""
    estimated_impact: str = ""


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("PerformanceMonitor")

        # Monitoring configuration
        self.monitoring_config = config.get(
            "performance_monitoring",
            {
                "enable_monitoring": True,
                "monitoring_interval_seconds": 60,
                "metrics_history_size": 1000,
                "alert_thresholds": {
                    "model_accuracy_min": 0.7,
                    "trading_win_rate_min": 0.6,
                    "system_memory_max": 0.8,
                    "system_cpu_max": 0.8,
                    "response_time_max": 5.0,
                },
                "optimization_thresholds": {
                    "performance_degradation": 0.1,
                    "confidence_drop": 0.15,
                    "memory_increase": 0.2,
                    "response_time_increase": 0.3,
                },
            },
        )

        # Metrics storage
        self.metrics_history: deque = deque(
            maxlen=self.monitoring_config["metrics_history_size"],
        )
        self.optimization_opportunities: list[OptimizationOpportunity] = []
        self.alerts: list[dict[str, Any]] = []

        # Performance tracking
        self.start_time = datetime.now()
        self.last_metrics_time = None
        self.monitoring_active = False

        # Component references
        self.dual_model_system = None
        self.ml_confidence_predictor = None
        self.enhanced_order_manager = None
        self.enhanced_training_manager = None

        # Performance baselines
        self.baseline_metrics = None
        self.performance_trends = {}

        # Monitoring tasks
        self.monitoring_task = None
        self.alert_task = None
        self.optimization_task = None

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="performance monitor initialization",
    )
    async def initialize(self) -> bool:
        """Initialize performance monitor."""
        try:
            self.logger.info("🔍 Initializing Performance Monitor...")

            # Initialize monitoring components
            await self._initialize_monitoring_components()

            # Start monitoring tasks
            await self._start_monitoring_tasks()

            # Initialize baseline metrics
            await self._initialize_baseline_metrics()

            self.monitoring_active = True
            self.logger.info("✅ Performance Monitor initialized successfully")
            return True

        except Exception:
            self.print(
                initialization_error("Error initializing performance monitor: {e}"),
            )
            return False

    async def _initialize_monitoring_components(self) -> None:
        """Initialize monitoring components."""
        # Set up monitoring intervals
        self.monitoring_interval = self.monitoring_config["monitoring_interval_seconds"]

        # Initialize performance tracking
        self.performance_trends = {
            "model_accuracy": deque(maxlen=100),
            "trading_win_rate": deque(maxlen=100),
            "system_memory": deque(maxlen=100),
            "system_cpu": deque(maxlen=100),
            "response_time": deque(maxlen=100),
            "confidence_final": deque(maxlen=100),
        }

        # CSV exporter
        self.csv_exporter: CSVExporter | None = None

        # Initialize baseline metrics
        await self._initialize_baseline_metrics()

        # Initialize CSV exporter
        await self._initialize_csv_exporter()

        # Initialize monitoring tasks
        await self._start_monitoring_tasks()

        self.logger.info("📊 Monitoring components initialized")

    async def _initialize_csv_exporter(self) -> None:
        """Initialize CSV exporter for data export."""
        try:
            self.csv_exporter = CSVExporter(self.config)
            await self.csv_exporter.initialize()
            self.logger.info("📊 CSV exporter initialized")
        except Exception:
            self.print(initialization_error("Error initializing CSV exporter: {e}"))

    async def _start_monitoring_tasks(self) -> None:
        """Start monitoring tasks."""
        # Start metrics collection task
        self.monitoring_task = asyncio.create_task(self._metrics_collection_loop())

        # Start alert monitoring task
        self.alert_task = asyncio.create_task(self._alert_monitoring_loop())

        # Start optimization analysis task
        self.optimization_task = asyncio.create_task(self._optimization_analysis_loop())

    async def _initialize_baseline_metrics(self) -> None:
        """Initialize baseline performance metrics."""
        # Create initial baseline
        baseline = PerformanceMetrics()
        baseline.model_accuracy = 0.75  # Default baseline
        baseline.trading_win_rate = 0.65
        baseline.system_memory_usage = 0.3
        baseline.system_cpu_usage = 0.2
        baseline.response_time = 1.0

        self.baseline_metrics = baseline
        self.logger.info("📊 Baseline metrics initialized")

    async def _metrics_collection_loop(self) -> None:
        """Continuous metrics collection loop."""
        while self.monitoring_active:
            try:
                await self._collect_performance_metrics()
                await asyncio.sleep(self.monitoring_interval)
            except Exception:
                self.print(error("Error in metrics collection loop: {e}"))
                await asyncio.sleep(self.monitoring_interval)

    async def _alert_monitoring_loop(self) -> None:
        """Continuous alert monitoring loop."""
        while self.monitoring_active:
            try:
                await self._check_alerts()
                await asyncio.sleep(self.monitoring_interval)
            except Exception:
                self.print(error("Error in alert monitoring loop: {e}"))
                await asyncio.sleep(self.monitoring_interval)

    async def _optimization_analysis_loop(self) -> None:
        """Continuous optimization analysis loop."""
        while self.monitoring_active:
            try:
                await self._analyze_optimization_opportunities()
                await asyncio.sleep(self.monitoring_interval * 5)  # Less frequent
            except Exception:
                self.print(error("Error in optimization analysis loop: {e}"))
                await asyncio.sleep(self.monitoring_interval * 5)

    async def _collect_performance_metrics(self) -> None:
        """Collect comprehensive performance metrics."""
        try:
            metrics = PerformanceMetrics()

            # Collect model performance metrics
            await self._collect_model_metrics(metrics)

            # Collect trading performance metrics
            await self._collect_trading_metrics(metrics)

            # Collect system performance metrics
            await self._collect_system_metrics(metrics)

            # Collect confidence metrics
            await self._collect_confidence_metrics(metrics)

            # Collect order execution metrics
            await self._collect_order_execution_metrics(metrics)

            # Collect training metrics
            await self._collect_training_metrics(metrics)

            # Store metrics
            self.metrics_history.append(metrics)
            self.last_metrics_time = datetime.now()

            # Update performance trends
            self._update_performance_trends(metrics)

        except Exception:
            self.print(error("Error collecting performance metrics: {e}"))

    async def _collect_model_metrics(self, metrics: PerformanceMetrics) -> None:
        """Collect model performance metrics."""
        try:
            if self.ml_confidence_predictor:
                # Get model performance from ML confidence predictor
                model_performance = self.ml_confidence_predictor.get_model_performance()

                metrics.model_accuracy = model_performance.get("accuracy", 0.0)
                metrics.model_precision = model_performance.get("precision", 0.0)
                metrics.model_recall = model_performance.get("recall", 0.0)
                metrics.model_f1_score = model_performance.get("f1_score", 0.0)
                metrics.model_auc = model_performance.get("auc", 0.0)

        except Exception:
            self.print(error("Error collecting model metrics: {e}"))

    async def _collect_trading_metrics(self, metrics: PerformanceMetrics) -> None:
        """Collect trading performance metrics."""
        try:
            # This would integrate with actual trading performance tracking
            # For now, using simulated metrics
            metrics.trading_win_rate = 0.65  # Simulated
            metrics.trading_profit_factor = 1.5
            metrics.trading_sharpe_ratio = 1.2
            metrics.trading_max_drawdown = -0.05
            metrics.trading_total_return = 0.25

            # Enhanced risk management metrics
            await self._collect_risk_management_metrics(metrics)

        except Exception:
            self.print(error("Error collecting trading metrics: {e}"))

    async def _collect_risk_management_metrics(
        self,
        metrics: PerformanceMetrics,
    ) -> None:
        """Collect enhanced risk management metrics."""
        try:
            # Get risk management data from configuration or external systems
            risk_system = self.config.get("risk_management", {})

            # Portfolio risk metrics
            portfolio_metrics = risk_system.get("portfolio", {})
            metrics.portfolio_var = portfolio_metrics.get("value_at_risk", 0.0)
            metrics.portfolio_correlation = portfolio_metrics.get("correlation", 0.0)
            metrics.portfolio_concentration = portfolio_metrics.get(
                "concentration",
                0.0,
            )
            metrics.portfolio_leverage = portfolio_metrics.get("leverage", 1.0)

            # Position risk metrics
            position_metrics = risk_system.get("positions", {})
            metrics.position_count = position_metrics.get("active_positions", 0)
            metrics.max_position_size = position_metrics.get("max_position_size", 0.0)
            metrics.avg_position_size = position_metrics.get("avg_position_size", 0.0)
            metrics.position_duration = position_metrics.get("avg_duration", 0.0)

            # Market risk metrics
            market_metrics = risk_system.get("market", {})
            metrics.market_volatility = market_metrics.get("volatility", 0.0)
            metrics.market_liquidity = market_metrics.get("liquidity_score", 0.0)
            metrics.market_stress = market_metrics.get("stress_score", 0.0)
            metrics.market_regime = market_metrics.get("regime", "unknown")

            # Risk alerts
            risk_alerts = risk_system.get("alerts", [])
            if risk_alerts:
                self.logger.warning(
                    f"Risk alerts detected: {len(risk_alerts)} active alerts",
                )

        except Exception:
            self.print(error("Error collecting risk management metrics: {e}"))

    async def _collect_system_metrics(self, metrics: PerformanceMetrics) -> None:
        """Collect system performance metrics."""
        try:
            # Memory usage
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            metrics.system_memory_usage = memory_info.rss / (1024 * 1024 * 1024)  # GB

            # CPU usage
            metrics.system_cpu_usage = psutil.cpu_percent(interval=1)

            # Response time (simulated)
            metrics.system_response_time = 1.5  # seconds

            # Throughput (simulated)
            metrics.system_throughput = 100.0  # operations per second

        except Exception:
            self.print(error("Error collecting system metrics: {e}"))

    async def _collect_confidence_metrics(self, metrics: PerformanceMetrics) -> None:
        """Collect confidence metrics."""
        try:
            if self.dual_model_system:
                # Get confidence metrics from dual model system
                self.dual_model_system.get_system_info()

                # Simulated confidence metrics
                metrics.confidence_analyst = 0.75
                metrics.confidence_tactician = 0.80
                metrics.confidence_final = 0.72

        except Exception:
            self.print(error("Error collecting confidence metrics: {e}"))

    async def _collect_order_execution_metrics(
        self,
        metrics: PerformanceMetrics,
    ) -> None:
        """Collect order execution metrics."""
        try:
            if self.enhanced_order_manager:
                # Get order execution performance
                performance = self.enhanced_order_manager.get_performance_metrics()

                metrics.order_execution_success_rate = performance.get(
                    "success_rate",
                    0.95,
                )
                metrics.order_execution_slippage = performance.get(
                    "average_slippage",
                    0.001,
                )

        except Exception:
            self.print(execution_error("Error collecting order execution metrics: {e}"))

    async def _collect_training_metrics(self, metrics: PerformanceMetrics) -> None:
        """Collect training performance metrics."""
        try:
            if self.enhanced_training_manager:
                # Get training performance
                training_status = (
                    self.enhanced_training_manager.get_enhanced_training_status()
                )

                metrics.training_performance = training_status.get("performance", 0.8)
                metrics.training_duration = training_status.get("duration", 300.0)

        except Exception:
            self.print(error("Error collecting training metrics: {e}"))

    def _update_performance_trends(self, metrics: PerformanceMetrics) -> None:
        """Update performance trends."""
        try:
            # Update trend data
            self.performance_trends["model_accuracy"].append(metrics.model_accuracy)
            self.performance_trends["trading_win_rate"].append(metrics.trading_win_rate)
            self.performance_trends["system_memory"].append(metrics.system_memory_usage)
            self.performance_trends["system_cpu"].append(metrics.system_cpu_usage)
            self.performance_trends["response_time"].append(
                metrics.system_response_time,
            )
            self.performance_trends["confidence_final"].append(metrics.confidence_final)

        except Exception:
            self.print(error("Error updating performance trends: {e}"))

    async def _check_alerts(self) -> None:
        """Check for performance alerts."""
        try:
            if not self.metrics_history:
                return

            latest_metrics = self.metrics_history[-1]
            thresholds = self.monitoring_config["alert_thresholds"]

            alerts = []

            # Check model performance alerts
            if latest_metrics.model_accuracy < thresholds["model_accuracy_min"]:
                alerts.append(
                    {
                        "type": "model_performance",
                        "severity": "high",
                        "message": f"Model accuracy below threshold: {latest_metrics.model_accuracy:.3f}",
                        "metric": "model_accuracy",
                        "value": latest_metrics.model_accuracy,
                        "threshold": thresholds["model_accuracy_min"],
                    },
                )

            # Check trading performance alerts
            if latest_metrics.trading_win_rate < thresholds["trading_win_rate_min"]:
                alerts.append(
                    {
                        "type": "trading_performance",
                        "severity": "medium",
                        "message": f"Trading win rate below threshold: {latest_metrics.trading_win_rate:.3f}",
                        "metric": "trading_win_rate",
                        "value": latest_metrics.trading_win_rate,
                        "threshold": thresholds["trading_win_rate_min"],
                    },
                )

            # Check system performance alerts
            if latest_metrics.system_memory_usage > thresholds["system_memory_max"]:
                alerts.append(
                    {
                        "type": "system_performance",
                        "severity": "high",
                        "message": f"Memory usage above threshold: {latest_metrics.system_memory_usage:.3f}",
                        "metric": "system_memory_usage",
                        "value": latest_metrics.system_memory_usage,
                        "threshold": thresholds["system_memory_max"],
                    },
                )

            if latest_metrics.system_cpu_usage > thresholds["system_cpu_max"]:
                alerts.append(
                    {
                        "type": "system_performance",
                        "severity": "medium",
                        "message": f"CPU usage above threshold: {latest_metrics.system_cpu_usage:.3f}",
                        "metric": "system_cpu_usage",
                        "value": latest_metrics.system_cpu_usage,
                        "threshold": thresholds["system_cpu_max"],
                    },
                )

            if latest_metrics.system_response_time > thresholds["response_time_max"]:
                alerts.append(
                    {
                        "type": "system_performance",
                        "severity": "medium",
                        "message": f"Response time above threshold: {latest_metrics.system_response_time:.3f}s",
                        "metric": "system_response_time",
                        "value": latest_metrics.system_response_time,
                        "threshold": thresholds["response_time_max"],
                    },
                )

            # Store alerts
            self.alerts.extend(alerts)

            # Log alerts
            for _alert in alerts:
                self.print(warning("🚨 ALERT: {alert['message']}"))

        except Exception:
            self.print(error("Error checking alerts: {e}"))

    async def _analyze_optimization_opportunities(self) -> None:
        """Analyze optimization opportunities."""
        try:
            if len(self.metrics_history) < 10:
                return

            # Clear previous opportunities
            self.optimization_opportunities.clear()

            # Analyze model performance optimization
            await self._analyze_model_optimization()

            # Analyze system performance optimization
            await self._analyze_system_optimization()

            # Analyze trading performance optimization
            await self._analyze_trading_optimization()

            # Analyze confidence optimization
            await self._analyze_confidence_optimization()

            # Analyze anomaly detection
            await self._analyze_anomaly_detection()

        except Exception:
            self.print(error("Error analyzing optimization opportunities: {e}"))

    async def _analyze_anomaly_detection(self) -> None:
        """Analyze system for anomalies and unusual patterns."""
        try:
            # Get recent metrics for anomaly analysis
            recent_metrics = (
                list(self.metrics_history)[-50:] if self.metrics_history else []
            )

            if len(recent_metrics) < 10:
                return

            # Calculate statistical baselines
            baseline_metrics = {
                "trading_win_rate": np.mean(
                    [m.trading_win_rate for m in recent_metrics],
                ),
                "trading_profit_factor": np.mean(
                    [m.trading_profit_factor for m in recent_metrics],
                ),
                "trading_sharpe_ratio": np.mean(
                    [m.trading_sharpe_ratio for m in recent_metrics],
                ),
                "model_accuracy": np.mean([m.model_accuracy for m in recent_metrics]),
                "confidence_final": np.mean(
                    [m.confidence_final for m in recent_metrics],
                ),
                "system_memory_usage": np.mean(
                    [m.system_memory_usage for m in recent_metrics],
                ),
                "system_cpu_usage": np.mean(
                    [m.system_cpu_usage for m in recent_metrics],
                ),
            }

            # Calculate standard deviations for threshold setting
            std_metrics = {
                "trading_win_rate": np.std(
                    [m.trading_win_rate for m in recent_metrics],
                ),
                "trading_profit_factor": np.std(
                    [m.trading_profit_factor for m in recent_metrics],
                ),
                "trading_sharpe_ratio": np.std(
                    [m.trading_sharpe_ratio for m in recent_metrics],
                ),
                "model_accuracy": np.std([m.model_accuracy for m in recent_metrics]),
                "confidence_final": np.std(
                    [m.confidence_final for m in recent_metrics],
                ),
                "system_memory_usage": np.std(
                    [m.system_memory_usage for m in recent_metrics],
                ),
                "system_cpu_usage": np.std(
                    [m.system_cpu_usage for m in recent_metrics],
                ),
            }

            # Check for anomalies in current metrics
            current_metrics = self.metrics_history[-1] if self.metrics_history else None
            if current_metrics:
                anomalies = []

                for metric_name, baseline in baseline_metrics.items():
                    current_value = getattr(current_metrics, metric_name, 0.0)
                    threshold = (
                        std_metrics.get(metric_name, 0.1) * 2
                    )  # 2 standard deviations

                    if abs(current_value - baseline) > threshold:
                        anomaly = OptimizationOpportunity(
                            category="anomaly_detection",
                            metric=metric_name,
                            current_value=current_value,
                            target_value=baseline,
                            improvement_potential=abs(current_value - baseline),
                            priority="high"
                            if abs(current_value - baseline) > threshold * 1.5
                            else "medium",
                            description=f"Anomaly detected in {metric_name}: {current_value:.3f} (baseline: {baseline:.3f})",
                            recommended_action="Investigate unusual pattern in system behavior",
                            estimated_impact="High - may indicate system issues or market changes",
                        )
                        anomalies.append(anomaly)

                # Add anomalies to optimization opportunities
                self.optimization_opportunities.extend(anomalies)

        except Exception:
            self.print(error("Error in anomaly detection: {e}"))

    async def _analyze_model_optimization(self) -> None:
        """Analyze model performance optimization opportunities."""
        try:
            recent_metrics = list(self.metrics_history)[-10:]
            avg_accuracy = np.mean([m.model_accuracy for m in recent_metrics])

            if avg_accuracy < 0.8:
                opportunity = OptimizationOpportunity(
                    category="model_performance",
                    metric="model_accuracy",
                    current_value=avg_accuracy,
                    target_value=0.85,
                    improvement_potential=0.85 - avg_accuracy,
                    priority="high" if avg_accuracy < 0.7 else "medium",
                    description="Model accuracy below optimal level",
                    recommended_action="Retrain models with additional data or hyperparameter optimization",
                    estimated_impact="High - Improved prediction accuracy",
                )
                self.optimization_opportunities.append(opportunity)

        except Exception:
            self.print(error("Error analyzing model optimization: {e}"))

    async def _analyze_system_optimization(self) -> None:
        """Analyze system performance optimization opportunities."""
        try:
            recent_metrics = list(self.metrics_history)[-10:]
            avg_memory = np.mean([m.system_memory_usage for m in recent_metrics])
            avg_cpu = np.mean([m.system_cpu_usage for m in recent_metrics])
            avg_response = np.mean([m.system_response_time for m in recent_metrics])

            # Memory optimization
            if avg_memory > 0.7:
                opportunity = OptimizationOpportunity(
                    category="system_performance",
                    metric="memory_usage",
                    current_value=avg_memory,
                    target_value=0.5,
                    improvement_potential=avg_memory - 0.5,
                    priority="high" if avg_memory > 0.8 else "medium",
                    description="High memory usage detected",
                    recommended_action="Optimize memory usage, implement garbage collection",
                    estimated_impact="Medium - Improved system stability",
                )
                self.optimization_opportunities.append(opportunity)

            # CPU optimization
            if avg_cpu > 0.7:
                opportunity = OptimizationOpportunity(
                    category="system_performance",
                    metric="cpu_usage",
                    current_value=avg_cpu,
                    target_value=0.5,
                    improvement_potential=avg_cpu - 0.5,
                    priority="medium",
                    description="High CPU usage detected",
                    recommended_action="Optimize algorithms, implement caching",
                    estimated_impact="Medium - Improved response time",
                )
                self.optimization_opportunities.append(opportunity)

            # Response time optimization
            if avg_response > 3.0:
                opportunity = OptimizationOpportunity(
                    category="system_performance",
                    metric="response_time",
                    current_value=avg_response,
                    target_value=1.5,
                    improvement_potential=avg_response - 1.5,
                    priority="high" if avg_response > 5.0 else "medium",
                    description="Slow response time detected",
                    recommended_action="Optimize algorithms, implement async processing",
                    estimated_impact="High - Improved user experience",
                )
                self.optimization_opportunities.append(opportunity)

        except Exception:
            self.print(error("Error analyzing system optimization: {e}"))

    async def _analyze_trading_optimization(self) -> None:
        """Analyze trading performance optimization opportunities."""
        try:
            recent_metrics = list(self.metrics_history)[-10:]
            avg_win_rate = np.mean([m.trading_win_rate for m in recent_metrics])
            avg_profit_factor = np.mean(
                [m.trading_profit_factor for m in recent_metrics],
            )

            if avg_win_rate < 0.6:
                opportunity = OptimizationOpportunity(
                    category="trading_performance",
                    metric="win_rate",
                    current_value=avg_win_rate,
                    target_value=0.7,
                    improvement_potential=0.7 - avg_win_rate,
                    priority="high" if avg_win_rate < 0.5 else "medium",
                    description="Low trading win rate detected",
                    recommended_action="Review trading strategy, improve entry/exit criteria",
                    estimated_impact="High - Improved profitability",
                )
                self.optimization_opportunities.append(opportunity)

            if avg_profit_factor < 1.2:
                opportunity = OptimizationOpportunity(
                    category="trading_performance",
                    metric="profit_factor",
                    current_value=avg_profit_factor,
                    target_value=1.5,
                    improvement_potential=1.5 - avg_profit_factor,
                    priority="medium",
                    description="Low profit factor detected",
                    recommended_action="Optimize position sizing, improve risk management",
                    estimated_impact="Medium - Improved risk-adjusted returns",
                )
                self.optimization_opportunities.append(opportunity)

        except Exception:
            self.print(error("Error analyzing trading optimization: {e}"))

    async def _analyze_confidence_optimization(self) -> None:
        """Analyze confidence optimization opportunities."""
        try:
            recent_metrics = list(self.metrics_history)[-10:]
            avg_confidence = np.mean([m.confidence_final for m in recent_metrics])

            if avg_confidence < 0.6:
                opportunity = OptimizationOpportunity(
                    category="confidence_performance",
                    metric="confidence_final",
                    current_value=avg_confidence,
                    target_value=0.75,
                    improvement_potential=0.75 - avg_confidence,
                    priority="medium",
                    description="Low confidence scores detected",
                    recommended_action="Improve model calibration, enhance feature engineering",
                    estimated_impact="Medium - Improved decision quality",
                )
                self.optimization_opportunities.append(opportunity)

        except Exception:
            self.print(error("Error analyzing confidence optimization: {e}"))

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary."""
        try:
            if not self.metrics_history:
                return {"error": "No metrics available"}

            latest_metrics = self.metrics_history[-1]

            # Calculate trends
            trends = {}
            for metric_name, trend_data in self.performance_trends.items():
                if len(trend_data) >= 2:
                    recent_values = list(trend_data)[-5:]
                    trends[metric_name] = {
                        "current": recent_values[-1],
                        "trend": "increasing"
                        if recent_values[-1] > recent_values[0]
                        else "decreasing",
                        "change": recent_values[-1] - recent_values[0],
                    }

            return {
                "current_metrics": {
                    "model_accuracy": latest_metrics.model_accuracy,
                    "trading_win_rate": latest_metrics.trading_win_rate,
                    "system_memory_usage": latest_metrics.system_memory_usage,
                    "system_cpu_usage": latest_metrics.system_cpu_usage,
                    "response_time": latest_metrics.system_response_time,
                    "confidence_final": latest_metrics.confidence_final,
                },
                "trends": trends,
                "alerts_count": len(self.alerts),
                "optimization_opportunities_count": len(
                    self.optimization_opportunities,
                ),
                "monitoring_duration": (
                    datetime.now() - self.start_time
                ).total_seconds(),
            }

        except Exception as e:
            self.print(error("Error getting performance summary: {e}"))
            return {"error": str(e)}

    def get_optimization_recommendations(self) -> list[dict[str, Any]]:
        """Get optimization recommendations."""
        try:
            recommendations = []

            for opportunity in self.optimization_opportunities:
                recommendations.append(
                    {
                        "category": opportunity.category,
                        "metric": opportunity.metric,
                        "current_value": opportunity.current_value,
                        "target_value": opportunity.target_value,
                        "improvement_potential": opportunity.improvement_potential,
                        "priority": opportunity.priority,
                        "description": opportunity.description,
                        "recommended_action": opportunity.recommended_action,
                        "estimated_impact": opportunity.estimated_impact,
                    },
                )

            return recommendations

        except Exception:
            self.print(error("Error getting optimization recommendations: {e}"))
            return []

    def get_alerts(self) -> list[dict[str, Any]]:
        """Get current alerts."""
        try:
            return self.alerts.copy()

        except Exception:
            self.print(error("Error getting alerts: {e}"))
            return []

    def export_performance_report(self, filepath: str) -> bool:
        """Export performance report to file."""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "performance_summary": self.get_performance_summary(),
                "optimization_recommendations": self.get_optimization_recommendations(),
                "alerts": self.get_alerts(),
                "metrics_history": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "model_accuracy": m.model_accuracy,
                        "trading_win_rate": m.trading_win_rate,
                        "system_memory_usage": m.system_memory_usage,
                        "system_cpu_usage": m.system_cpu_usage,
                        "response_time": m.system_response_time,
                        "confidence_final": m.confidence_final,
                    }
                    for m in self.metrics_history
                ],
            }

            with open(filepath, "w") as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"Performance report exported to: {filepath}")
            return True

        except Exception:
            self.print(error("Error exporting performance report: {e}"))
            return False

    async def export_all_monitoring_data(
        self,
        time_range: str = "24h",
    ) -> Dict[str, Optional[str]]:
        """Export all monitoring data to CSV files."""
        try:
            if not self.csv_exporter:
                self.print(initialization_error("CSV exporter not initialized"))
                return {}

            export_results = {}

            # Export performance metrics
            if self.metrics_history:
                performance_data = []
                for metrics in self.metrics_history:
                    performance_data.append(
                        {
                            "timestamp": metrics.timestamp.isoformat(),
                            "model_accuracy": metrics.model_accuracy,
                            "model_precision": metrics.model_precision,
                            "model_recall": metrics.model_recall,
                            "model_f1_score": metrics.model_f1_score,
                            "model_auc": metrics.model_auc,
                            "trading_win_rate": metrics.trading_win_rate,
                            "trading_profit_factor": metrics.trading_profit_factor,
                            "trading_sharpe_ratio": metrics.trading_sharpe_ratio,
                            "trading_max_drawdown": metrics.trading_max_drawdown,
                            "trading_total_return": metrics.trading_total_return,
                            "system_memory_usage": metrics.system_memory_usage,
                            "system_cpu_usage": metrics.system_cpu_usage,
                            "system_response_time": metrics.system_response_time,
                            "system_throughput": metrics.system_throughput,
                            "confidence_analyst": metrics.confidence_analyst,
                            "confidence_tactician": metrics.confidence_tactician,
                            "confidence_final": metrics.confidence_final,
                        },
                    )

                export_results[
                    "performance"
                ] = await self.csv_exporter.export_performance_metrics(
                    performance_data,
                    time_range,
                )

            # Export risk metrics
            if self.metrics_history:
                risk_data = []
                for metrics in self.metrics_history:
                    risk_data.append(
                        {
                            "timestamp": metrics.timestamp.isoformat(),
                            "portfolio_var": metrics.portfolio_var,
                            "portfolio_correlation": metrics.portfolio_correlation,
                            "portfolio_concentration": metrics.portfolio_concentration,
                            "portfolio_leverage": metrics.portfolio_leverage,
                            "position_count": metrics.position_count,
                            "max_position_size": metrics.max_position_size,
                            "avg_position_size": metrics.avg_position_size,
                            "position_duration": metrics.position_duration,
                            "market_volatility": metrics.market_volatility,
                            "market_liquidity": metrics.market_liquidity,
                            "market_stress": metrics.market_stress,
                            "market_regime": metrics.market_regime,
                        },
                    )

                export_results[
                    "risk_metrics"
                ] = await self.csv_exporter.export_risk_metrics(risk_data, time_range)

            # Export system health data
            if self.metrics_history:
                system_data = []
                for metrics in self.metrics_history:
                    system_data.append(
                        {
                            "timestamp": metrics.timestamp.isoformat(),
                            "health_score": (
                                metrics.model_accuracy + metrics.trading_win_rate
                            )
                            / 2,
                            "memory_usage": metrics.system_memory_usage,
                            "cpu_usage": metrics.system_cpu_usage,
                            "response_time": metrics.system_response_time,
                            "throughput": metrics.system_throughput,
                            "error_rate": 0.0,  # Would be calculated from actual errors
                            "uptime": 0.0,  # Would be calculated from system uptime
                            "active_connections": 0,  # Would be tracked from actual connections
                        },
                    )

                export_results[
                    "system_health"
                ] = await self.csv_exporter.export_system_health(
                    system_data,
                    time_range,
                )

            self.logger.info(f"✅ All monitoring data exported: {export_results}")
            return export_results

        except Exception:
            self.print(error("Error exporting all monitoring data: {e}"))
            return {}

    def get_csv_export_summary(self) -> Dict[str, Any]:
        """Get CSV export summary."""
        try:
            if not self.csv_exporter:
                return {"error": "CSV exporter not initialized"}

            return self.csv_exporter.get_export_summary()
        except Exception as e:
            self.print(error("Error getting CSV export summary: {e}"))
            return {"error": str(e)}

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance monitor cleanup",
    )
    async def stop(self) -> None:
        """Stop performance monitoring."""
        try:
            self.logger.info("🛑 Stopping Performance Monitor...")

            # Stop monitoring tasks
            self.monitoring_active = False

            if self.monitoring_task:
                self.monitoring_task.cancel()
            if self.alert_task:
                self.alert_task.cancel()
            if self.optimization_task:
                self.optimization_task.cancel()

            # Export final report
            report_path = (
                f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            self.export_performance_report(report_path)

            self.logger.info("✅ Performance Monitor stopped successfully")

        except Exception:
            self.print(error("Error stopping performance monitor: {e}"))


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="performance monitor setup",
)
async def setup_performance_monitor(
    config: dict[str, Any] | None = None,
) -> PerformanceMonitor | None:
    """
    Setup Performance Monitor.

    Args:
        config: Configuration dictionary

    Returns:
        Optional[PerformanceMonitor]: Initialized performance monitor or None
    """
    try:
        if config is None:
            config = {}

        monitor = PerformanceMonitor(config)
        if await monitor.initialize():
            return monitor
        return None

    except Exception as e:
        system_logger.exception(f"Error setting up Performance Monitor: {e}")
        return None
