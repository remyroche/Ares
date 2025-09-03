# src/monitoring/performance_dashboard.py

from src.core.decorators import (
    cached,
    log_execution_time
)

from src.core.domain import PerformanceLevel

"""
Performance Dashboard for Dual Model System
Real-time monitoring and visualization of system performance metrics.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.decorators import handles_errors

from src.utils.logger import system_logger

@dataclass
class DashboardMetrics:
    """Dashboard metrics data structure."""

    timestamp: datetime
    model_performance: Dict[str, float]
    trading_performance: Dict[str, float]
    system_performance: Dict[str, float]
    confidence_metrics: Dict[str, float]
    alerts: List[Dict[str, Any]]
    optimization_opportunities: List[Dict[str, Any]]

class PerformanceDashboard:
    """Real-time performance dashboard."""

    def __init__(self, config: Dict[str, Any]) -> None:
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
                "enable_export": False,
                "export_interval_minutes": 60,
            },
        )

        # Dashboard state
        self.is_active: bool = False
        self.update_task: Optional[asyncio.Task] = None
        self.export_task: Optional[asyncio.Task] = None

        # Dashboard data
        self.metrics_history: List[DashboardMetrics] = []
        self.current_metrics: Optional[DashboardMetrics] = None

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
