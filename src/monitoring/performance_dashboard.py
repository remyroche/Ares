# src/monitoring/performance_dashboard.py

"""
Performance Dashboard for Dual Model System
Real-time monitoring and visualization of system performance metrics.
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.monitoring.performance_monitor import PerformanceMonitor
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


@dataclass
class DashboardMetrics:
    """Dashboard metrics data structure."""

    timestamp: datetime
    model_performance: dict[str, float]
    trading_performance: dict[str, float]
    system_performance: dict[str, float]
    confidence_metrics: dict[str, float]
    alerts: list[dict[str, Any]]
    optimization_opportunities: list[dict[str, Any]]


class PerformanceDashboard:
    """Real-time performance dashboard."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("PerformanceDashboard")

        # Dashboard configuration
        self.dashboard_config = config.get(
            "performance_dashboard",
            {
                "enable_dashboard": True,
                "update_interval_seconds": 30,
                "max_history_points": 100,
                "enable_alerts": True,
                "enable_optimization_recommendations": True,
                "enable_export": True,
                "export_interval_minutes": 60,
            },
        )

        # Dashboard state
        self.is_active = False
        self.update_task = None
        self.export_task = None

        # Performance monitor reference
        self.performance_monitor: PerformanceMonitor | None = None

        # Dashboard data
        self.metrics_history: list[DashboardMetrics] = []
        self.current_metrics: DashboardMetrics | None = None

        # Export configuration
        self.export_dir = Path("dashboard_exports")
        self.export_dir.mkdir(exist_ok=True)

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="performance dashboard initialization",
    )
    async def initialize(self, performance_monitor: PerformanceMonitor) -> bool:
        """Initialize performance dashboard."""
        self.logger.info("📊 Initializing Performance Dashboard...")

        # Set performance monitor reference
        self.performance_monitor = performance_monitor

        # Initialize dashboard components
        await self._initialize_dashboard_components()

        # Start dashboard tasks
        await self._start_dashboard_tasks()

        self.is_active = True
        self.logger.info("✅ Performance Dashboard initialized successfully")
        return True

    async def _initialize_dashboard_components(self) -> None:
        """Initialize dashboard components."""
        # Set up update interval
        self.update_interval = self.dashboard_config["update_interval_seconds"]

        # Initialize metrics history
        self.metrics_history = []

        self.logger.info("📊 Dashboard components initialized")

    async def _start_dashboard_tasks(self) -> None:
        """Start dashboard tasks."""
        # Start metrics update task
        self.update_task = asyncio.create_task(self._metrics_update_loop())

        # Start export task
        if self.dashboard_config["enable_export"]:
            self.export_task = asyncio.create_task(self._export_loop())

    async def _metrics_update_loop(self) -> None:
        """Continuous metrics update loop."""
        while self.is_active:
            await self._update_dashboard_metrics()
            await asyncio.sleep(self.update_interval)

    async def _export_loop(self) -> None:
        """Continuous export loop."""
        export_interval = self.dashboard_config["export_interval_minutes"] * 60

        while self.is_active:
            await self._export_dashboard_data()
            await asyncio.sleep(export_interval)

    async def _update_dashboard_metrics(self) -> None:
        """Update dashboard metrics."""
        if not self.performance_monitor:
            return

        # Get performance summary
        performance_summary = self.performance_monitor.get_performance_summary()

        if "error" in performance_summary:
            self.logger.warning(
                f"Error getting performance summary: {performance_summary['error']}",
            )
            return

        # Create dashboard metrics
        dashboard_metrics = DashboardMetrics(
            timestamp=datetime.now(),
            model_performance={
                "accuracy": performance_summary["current_metrics"][
                    "model_accuracy"
                ],
                "precision": 0.0,  # Would get from performance monitor
                "recall": 0.0,  # Would get from performance monitor
                "f1_score": 0.0,  # Would get from performance monitor
                "auc": 0.0,  # Would get from performance monitor
            },
            trading_performance={
                "win_rate": performance_summary["current_metrics"][
                    "trading_win_rate"
                ],
                "profit_factor": 1.5,  # Would get from performance monitor
                    "sharpe_ratio": 1.2,  # Would get from performance monitor
                    "max_drawdown": -0.05,  # Would get from performance monitor
                    "total_return": 0.25,  # Would get from performance monitor
                },
                system_performance={
                    "memory_usage": performance_summary["current_metrics"][
                        "system_memory_usage"
                    ],
                    "cpu_usage": performance_summary["current_metrics"][
                        "system_cpu_usage"
                    ],
                    "response_time": performance_summary["current_metrics"][
                        "response_time"
                    ],
                    "throughput": 100.0,  # Would get from performance monitor
                },
                confidence_metrics={
                    "analyst_confidence": 0.75,  # Would get from performance monitor
                    "tactician_confidence": 0.80,  # Would get from performance monitor
                    "final_confidence": performance_summary["current_metrics"][
                        "confidence_final"
                    ],
                },
                alerts=self.performance_monitor.get_alerts(),
                optimization_opportunities=self.performance_monitor.get_optimization_recommendations(),
            )

            # Update current metrics
            self.current_metrics = dashboard_metrics

            # Add to history
            self.metrics_history.append(dashboard_metrics)

            # Keep history within limits
            max_history = self.dashboard_config["max_history_points"]
            if len(self.metrics_history) > max_history:
                self.metrics_history = self.metrics_history[-max_history:]

            # Log dashboard update
            self.logger.debug(
                f"Dashboard updated: {len(self.metrics_history)} metrics points",
            )

        except Exception as e:
            self.logger.error(f"Error updating dashboard metrics: {e}")

    async def _export_dashboard_data(self) -> None:
        """Export dashboard data."""
        try:
            if not self.current_metrics:
                return

            # Create export filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = self.export_dir / f"dashboard_export_{timestamp}.json"

            # Prepare export data
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "current_metrics": {
                    "model_performance": self.current_metrics.model_performance,
                    "trading_performance": self.current_metrics.trading_performance,
                    "system_performance": self.current_metrics.system_performance,
                    "confidence_metrics": self.current_metrics.confidence_metrics,
                },
                "alerts": self.current_metrics.alerts,
                "optimization_opportunities": self.current_metrics.optimization_opportunities,
                "metrics_history": [
                    {
                        "timestamp": m.timestamp.isoformat(),
                        "model_accuracy": m.model_performance["accuracy"],
                        "trading_win_rate": m.trading_performance["win_rate"],
                        "system_memory": m.system_performance["memory_usage"],
                        "system_cpu": m.system_performance["cpu_usage"],
                        "response_time": m.system_performance["response_time"],
                        "confidence_final": m.confidence_metrics["final_confidence"],
                    }
                    for m in self.metrics_history
                ],
            }

            # Export to file
            with open(export_file, "w") as f:
                json.dump(export_data, f, indent=2)

            self.logger.info(f"📊 Dashboard data exported to: {export_file}")

        except Exception as e:
            self.logger.error(f"Error exporting dashboard data: {e}")

    def get_dashboard_summary(self) -> dict[str, Any]:
        """Get dashboard summary."""
        try:
            if not self.current_metrics:
                return {"error": "No dashboard data available"}

            # Calculate trends
            trends = self._calculate_trends()

            # Calculate health score
            health_score = self._calculate_health_score()

            return {
                "timestamp": self.current_metrics.timestamp.isoformat(),
                "current_metrics": {
                    "model_performance": self.current_metrics.model_performance,
                    "trading_performance": self.current_metrics.trading_performance,
                    "system_performance": self.current_metrics.system_performance,
                    "confidence_metrics": self.current_metrics.confidence_metrics,
                },
                "trends": trends,
                "health_score": health_score,
                "alerts_count": len(self.current_metrics.alerts),
                "optimization_opportunities_count": len(
                    self.current_metrics.optimization_opportunities,
                ),
                "metrics_history_count": len(self.metrics_history),
            }

        except Exception as e:
            self.logger.error(f"Error getting dashboard summary: {e}")
            return {"error": str(e)}

    def _calculate_trends(self) -> dict[str, Any]:
        """Calculate performance trends."""
        try:
            if len(self.metrics_history) < 2:
                return {}

            trends = {}

            # Get recent metrics
            recent_metrics = self.metrics_history[-5:]

            # Calculate trends for key metrics
            key_metrics = [
                ("model_accuracy", "model_performance", "accuracy"),
                ("trading_win_rate", "trading_performance", "win_rate"),
                ("system_memory", "system_performance", "memory_usage"),
                ("system_cpu", "system_performance", "cpu_usage"),
                ("response_time", "system_performance", "response_time"),
                ("confidence_final", "confidence_metrics", "final_confidence"),
            ]

            for metric_name, category, field in key_metrics:
                values = [getattr(m, category)[field] for m in recent_metrics]

                if len(values) >= 2:
                    current_value = values[-1]
                    previous_value = values[0]
                    change = current_value - previous_value
                    change_percent = (
                        (change / previous_value * 100) if previous_value != 0 else 0
                    )

                    trends[metric_name] = {
                        "current": current_value,
                        "previous": previous_value,
                        "change": change,
                        "change_percent": change_percent,
                        "trend": "improving"
                        if change > 0
                        else "declining"
                        if change < 0
                        else "stable",
                    }

            return trends

        except Exception as e:
            self.logger.error(f"Error calculating trends: {e}")
            return {}

    def _calculate_predictive_analytics(self) -> dict[str, Any]:
        """Calculate predictive analytics for future performance estimation."""
        try:
            if len(self.metrics_history) < 10:
                return {"error": "Insufficient data for predictive analytics"}

            # Get historical data for prediction
            historical_metrics = self.metrics_history[-50:]  # Last 50 data points
            
            predictions = {}
            
            # Simple linear regression for key metrics
            for metric_name in ["model_accuracy", "trading_win_rate", "system_memory_usage"]:
                values = []
                for metric in historical_metrics:
                    if metric_name == "model_accuracy":
                        values.append(metric.model_performance.get("accuracy", 0.0))
                    elif metric_name == "trading_win_rate":
                        values.append(metric.trading_performance.get("win_rate", 0.0))
                    elif metric_name == "system_memory_usage":
                        values.append(metric.system_performance.get("memory_usage", 0.0))
                
                timestamps = [i for i in range(len(values))]
                
                if len(values) >= 5:
                    # Simple linear trend prediction
                    try:
                        # Calculate linear regression coefficients
                        n = len(values)
                        sum_x = sum(timestamps)
                        sum_y = sum(values)
                        sum_xy = sum(x * y for x, y in zip(timestamps, values))
                        sum_x2 = sum(x * x for x in timestamps)
                        
                        # Linear regression: y = mx + b
                        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                        intercept = (sum_y - slope * sum_x) / n
                        
                        # Predict next 5 periods
                        future_predictions = []
                        for i in range(1, 6):
                            prediction = slope * (n + i - 1) + intercept
                            future_predictions.append(max(0, min(1, prediction)))  # Clamp between 0 and 1
                        
                        # Calculate prediction confidence based on R-squared
                        y_mean = sum_y / n
                        ss_tot = sum((y - y_mean) ** 2 for y in values)
                        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(timestamps, values))
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                        
                        predictions[metric_name] = {
                            "current_value": values[-1],
                            "predictions": future_predictions,
                            "confidence": r_squared,
                            "trend": "increasing" if slope > 0 else "decreasing",
                            "slope": slope
                        }
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to calculate prediction for {metric_name}: {e}")
                        continue

            return predictions

        except Exception as e:
            self.logger.error(f"Error calculating predictive analytics: {e}")
            return {"error": f"Predictive analytics failed: {e}"}

    def _calculate_health_score(self) -> dict[str, Any]:
        """Calculate system health score."""
        try:
            if not self.current_metrics:
                return {"score": 0, "status": "unknown"}

            # Define health thresholds
            thresholds = {
                "model_accuracy": {"min": 0.7, "weight": 0.3},
                "trading_win_rate": {"min": 0.6, "weight": 0.25},
                "system_memory": {"max": 0.8, "weight": 0.2},
                "system_cpu": {"max": 0.8, "weight": 0.15},
                "response_time": {"max": 5.0, "weight": 0.1},
            }

            # Calculate individual scores
            scores = {}
            total_score = 0
            total_weight = 0

            for metric, config in thresholds.items():
                if metric == "model_accuracy":
                    value = self.current_metrics.model_performance["accuracy"]
                    min_threshold = config["min"]
                    score = (
                        min(1.0, value / min_threshold)
                        if value < min_threshold
                        else 1.0
                    )
                elif metric == "trading_win_rate":
                    value = self.current_metrics.trading_performance["win_rate"]
                    min_threshold = config["min"]
                    score = (
                        min(1.0, value / min_threshold)
                        if value < min_threshold
                        else 1.0
                    )
                elif metric == "system_memory":
                    value = self.current_metrics.system_performance["memory_usage"]
                    max_threshold = config["max"]
                    score = (
                        1.0 - min(1.0, value / max_threshold)
                        if value > max_threshold
                        else 1.0
                    )
                elif metric == "system_cpu":
                    value = self.current_metrics.system_performance["cpu_usage"]
                    max_threshold = config["max"]
                    score = (
                        1.0 - min(1.0, value / max_threshold)
                        if value > max_threshold
                        else 1.0
                    )
                elif metric == "response_time":
                    value = self.current_metrics.system_performance["response_time"]
                    max_threshold = config["max"]
                    score = (
                        1.0 - min(1.0, value / max_threshold)
                        if value > max_threshold
                        else 1.0
                    )
                else:
                    score = 1.0

                scores[metric] = score
                total_score += score * config["weight"]
                total_weight += config["weight"]

            # Calculate overall health score
            overall_score = total_score / total_weight if total_weight > 0 else 0

            # Determine health status
            if overall_score >= 0.8:
                status = "healthy"
            elif overall_score >= 0.6:
                status = "warning"
            else:
                status = "critical"

            return {
                "score": overall_score,
                "status": status,
                "component_scores": scores,
                "alerts_count": len(self.current_metrics.alerts),
            }

        except Exception as e:
            self.logger.error(f"Error calculating health score: {e}")
            return {"score": 0, "status": "error"}

    def get_alerts_summary(self) -> dict[str, Any]:
        """Get alerts summary."""
        try:
            if not self.current_metrics:
                return {"alerts": [], "summary": {}}

            alerts = self.current_metrics.alerts

            # Categorize alerts
            alert_categories = {}
            severity_counts = {"high": 0, "medium": 0, "low": 0}

            for alert in alerts:
                alert_type = alert.get("type", "unknown")
                severity = alert.get("severity", "medium")

                if alert_type not in alert_categories:
                    alert_categories[alert_type] = []
                alert_categories[alert_type].append(alert)

                severity_counts[severity] += 1

            return {
                "alerts": alerts,
                "summary": {
                    "total_alerts": len(alerts),
                    "alert_categories": alert_categories,
                    "severity_counts": severity_counts,
                },
            }

        except Exception as e:
            self.logger.error(f"Error getting alerts summary: {e}")
            return {"alerts": [], "summary": {}}

    def get_optimization_summary(self) -> dict[str, Any]:
        """Get optimization opportunities summary."""
        try:
            if not self.current_metrics:
                return {"opportunities": [], "summary": {}}

            opportunities = self.current_metrics.optimization_opportunities

            # Categorize opportunities
            opportunity_categories = {}
            priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

            for opportunity in opportunities:
                category = opportunity.get("category", "unknown")
                priority = opportunity.get("priority", "medium")

                if category not in opportunity_categories:
                    opportunity_categories[category] = []
                opportunity_categories[category].append(opportunity)

                priority_counts[priority] += 1

            return {
                "opportunities": opportunities,
                "summary": {
                    "total_opportunities": len(opportunities),
                    "opportunity_categories": opportunity_categories,
                    "priority_counts": priority_counts,
                },
            }

        except Exception as e:
            self.logger.error(f"Error getting optimization summary: {e}")
            return {"opportunities": [], "summary": {}}

    def get_performance_chart_data(
        self,
        metric: str,
        time_range: str = "1h",
    ) -> dict[str, Any]:
        """Get performance chart data for a specific metric."""
        try:
            if not self.metrics_history:
                return {"error": "No metrics history available"}

            # Filter metrics based on time range
            filtered_metrics = self._filter_metrics_by_time_range(time_range)
            if not filtered_metrics:
                return {"error": "No metrics available for specified time range"}

            # Extract data for the specified metric
            timestamps = [m.timestamp.isoformat() for m in filtered_metrics]
            values = self._extract_metric_values(filtered_metrics, metric)
            
            if values is None:
                return {"error": f"Unknown metric: {metric}"}

            return {
                "metric": metric,
                "time_range": time_range,
                "timestamps": timestamps,
                "values": values,
                "data_points": len(values),
            }

        except Exception as e:
            self.logger.error(f"Error getting performance chart data: {e}")
            return {"error": str(e)}

    def _filter_metrics_by_time_range(self, time_range: str) -> list[DashboardMetrics]:
        """Filter metrics based on time range."""
        now = datetime.now()
        cutoff_time = self._get_cutoff_time(now, time_range)
        
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]

    def _get_cutoff_time(self, now: datetime, time_range: str) -> datetime:
        """Get cutoff time based on time range."""
        time_range_map = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "24h": timedelta(hours=24),
        }
        
        delta = time_range_map.get(time_range, timedelta(hours=1))  # Default to 1 hour
        return now - delta

    def _extract_metric_values(self, filtered_metrics: list[DashboardMetrics], metric: str) -> list[float] | None:
        """Extract metric values from filtered metrics."""
        metric_extractors = {
            "model_accuracy": lambda m: m.model_performance["accuracy"],
            "trading_win_rate": lambda m: m.trading_performance["win_rate"],
            "system_memory": lambda m: m.system_performance["memory_usage"],
            "system_cpu": lambda m: m.system_performance["cpu_usage"],
            "response_time": lambda m: m.system_performance["response_time"],
            "confidence_final": lambda m: m.confidence_metrics["final_confidence"],
        }
        
        extractor = metric_extractors.get(metric)
        if extractor is None:
            return None
            
        return [extractor(m) for m in filtered_metrics]

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="performance dashboard cleanup",
    )
    async def stop(self) -> None:
        """Stop performance dashboard."""
        try:
            self.logger.info("🛑 Stopping Performance Dashboard...")

            # Stop dashboard tasks
            self.is_active = False

            if self.update_task:
                self.update_task.cancel()
            if self.export_task:
                self.export_task.cancel()

            # Export final dashboard data
            await self._export_dashboard_data()

            self.logger.info("✅ Performance Dashboard stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping performance dashboard: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="performance dashboard setup",
)
async def setup_performance_dashboard(
    config: dict[str, Any] | None = None,
    performance_monitor: PerformanceMonitor | None = None,
) -> PerformanceDashboard | None:
    """
    Setup Performance Dashboard.

    Args:
        config: Configuration dictionary
        performance_monitor: Performance monitor instance

    Returns:
        Optional[PerformanceDashboard]: Initialized performance dashboard or None
    """
    try:
        if config is None:
            config = {}

        dashboard = PerformanceDashboard(config)
        if await dashboard.initialize(performance_monitor):
            return dashboard
        return None

    except Exception as e:
        system_logger.error(f"Error setting up Performance Dashboard: {e}")
        return None
