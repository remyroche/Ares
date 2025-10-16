
from src.core.domain import PerformanceLevel
from src.utils.logger import system_logger
from src.core.decorators import handles_errors, log_execution_time, cached

# src/monitoring/performance_dashboard.py

"""
Performance Dashboard for Dual Model System
Real-time monitoring and visualization of system performance metrics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncio
    from datetime import datetime
import logging
import time

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

    def __init__(self, config: dict[str, Any]) -> None:
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
                "enable_export": False,
                "export_interval_minutes": 60,
            },
        )

        # Dashboard state
        self.is_active: bool = False
        self.update_task: asyncio.Task | None = None
        self.export_task: asyncio.Task | None = None

        # Dashboard data
        self.metrics_history: list[DashboardMetrics] = []
        self.current_metrics: DashboardMetrics | None = None

        # Export configuration
        self.export_dir = Path("dashboard_exports")
        self.export_dir.mkdir(exist_ok = True)

    @log_execution_time(level = PerformanceLevel.DETAILED)
    @log_execution_time()
    @cached()
    @handles_errors(fallback = False)
    async def initialize(self) -> bool:
        """Initialize performance dashboard."""
        self.logger.info("📊 Initializing Performance Dashboard...")
        self.is_active = True
        self.logger.info("✅ Performance Dashboard initialized successfully")
        return True

# Global dashboard instance
performance_dashboard: PerformanceDashboard | None = None

async def setup_performance_dashboard(
    config: dict[str, Any] | None = None
) -> PerformanceDashboard | None:
    """
    Setup and initialize the performance dashboard.

    Args:
        config: Configuration dictionary for the dashboard

    Returns:
        PerformanceDashboard instance or None if setup fails
    """
    global performance_dashboard

    try:
        if performance_dashboard is None:
            default_config = config or {}
            performance_dashboard = PerformanceDashboard(default_config)
            success = await performance_dashboard.initialize()
            if success:
                system_logger.info("✅ Performance dashboard setup completed successfully")
                return performance_dashboard
            else:
                system_logger.error("❌ Performance dashboard initialization failed")
                return None
        else:
            system_logger.info("📊 Performance dashboard already initialized")
            return performance_dashboard

    except Exception as e:
        system_logger.exception(f"❌ Error setting up performance dashboard: {e}")
        return None
