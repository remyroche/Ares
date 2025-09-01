"""
Monitoring manager for pipeline components (minimal scaffold).
"""

from typing import Any, Dict
from src.utils.performance_monitor import (
    performance_monitor,
    PerformanceLevel,
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger

