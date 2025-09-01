#!/usr/bin/env python3
"""
Advanced Tracing System with Correlation IDs

This module provides comprehensive request/response tracing across all components
of the Ares trading bot with correlation IDs for debugging and performance analysis.
"""


import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.centralized_decorators import (
    performance_monitor,
    PerformanceLevel,
)
from src.utils.logger import system_logger


class TraceLevel(Enum):
    """Trace levels for different types of tracing."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComponentType(Enum):
    """Component types for tracing."""

    ANALYST = "analyst"
    STRATEGIST = "strategist"
    TACTICIAN = "tactician"
    SUPERVISOR = "supervisor"
    EXCHANGE = "exchange"
    DATABASE = "database"
    GUI = "gui"
    MONITORING = "monitoring"


@dataclass
class TraceSpan:
    """Individual trace span for a component operation."""

    span_id: str
    correlation_id: str
    component_type: ComponentType
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    status: str = "running"  # "running", "completed", "failed"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_span_id: Optional[str] = None
    child_span_ids: List[str] = field(default_factory=list)


@dataclass
class TraceRequest:
    """Complete trace request with all spans."""

    correlation_id: str
    request_timestamp: datetime
    component_path: List[ComponentType]
    spans: List[TraceSpan]
    response_timestamp: Optional[datetime] = None
    total_duration_ms: Optional[float] = None
    status: str = "running"  # "running", "completed", "failed"
    error_info: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass


