# src/monitoring/performance_dashboard.py

"""
Performance Dashboard for Dual Model System
Real-time monitoring and visualization of system performance metrics.
"""


from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

    performance_monitor,
    PerformanceLevel,
    resource_monitor,
    memory_efficient,
)


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


