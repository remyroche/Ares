from __future__ import annotations

from src.core.decorators import cached, log_execution_time
from src.core.domain import PerformanceLevel

# src/monitoring/performance_dashboard.py


"""
Performance Dashboard for Dual Model System
Real-time monitoring and visualization of system performance metrics.
"""


from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.decorators import handles_errors
from src.utils.logger import system_logger

if TYPE_CHECKING:
    import asyncio
    from datetime import datetime
from src.core.decorators.errors import handles_errors


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
        self.export_dir.mkdir(exist_ok=True)

    @log_execution_time(level=PerformanceLevel.DETAILED)
    @log_execution_time()
    @cached()
    @handles_errors(fallback=False)
    async def initialize(self) -> bool:
        """Initialize performance dashboard."""
        self.logger.info("📊 Initializing Performance Dashboard...")
        self.is_active = True
        self.logger.info("✅ Performance Dashboard initialized successfully")
        return True
