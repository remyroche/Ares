# src/monitoring/performance_monitor.py

"""
Performance Monitor for Dual Model System
Comprehensive monitoring of model performance, system metrics, trading performance, and optimization opportunities.
"""


from dataclasses import dataclass, field
from datetime import datetime

from src.utils.centralized_decorators import (
performance_monitor,
PerformanceLevel,
resource_monitor,
memory_efficient,
)


@dataclass


