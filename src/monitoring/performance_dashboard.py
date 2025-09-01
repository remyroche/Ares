# src/monitoring/performance_dashboard.py

"""
Performance Dashboard for Dual Model System
Real-time monitoring and visualization of system performance metrics.
"""


import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.error_handler import handle_errors
from src.utils.centralized_decorators import (
    performance_monitor,
    PerformanceLevel,
    resource_monitor,
    memory_efficient,
)
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


