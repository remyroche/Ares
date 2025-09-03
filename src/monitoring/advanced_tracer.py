#!/usr/bin/env python3
"""Advanced Tracing System with Correlation IDs.

This module provides comprehensive request/response tracing across all components of the
Ares trading bot with correlation IDs for debugging and performance analysis.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from src.core.decorators import handles_errors
from src.utils.centralized_decorators import PerformanceLevel, performance_monitor
from src.utils.error_handler import handle_errors, handle_specific_errors
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
class PerformanceMetrics:
    """Performance metrics for tracing."""

    total_duration_ms: float
    component_durations: Dict[str, float]
    bottleneck_component: str
    throughput_ops_per_sec: float
    error_rate: float
    success_rate: float


class AdvancedTracer:
    """Advanced tracing system with correlation IDs for comprehensive request/response
    tracking across all components."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("AdvancedTracer")

        self.tracer_config = config.get("advanced_tracer", {})
        self.enable_tracing: bool = bool(self.tracer_config.get("enable_tracing", True))
        self.correlation_id_header: str = self.tracer_config.get(
            "correlation_id_header",
            "X-Correlation-ID",
        )
        self.trace_sampling_rate: float = float(
            self.tracer_config.get("trace_sampling_rate", 1.0)
        )
        self.max_trace_history: int = int(
            self.tracer_config.get("max_trace_history", 10000)
        )
        self.enable_performance_tracing: bool = bool(
            self.tracer_config.get("enable_performance_tracing", True)
        )
        self.enable_error_tracing: bool = bool(
            self.tracer_config.get("enable_error_tracing", True)
        )

        # Storage
        self._traces: Dict[str, TraceRequest] = {}

    @performance_monitor(level=PerformanceLevel.DETAILED)
    @handles_errors(
        error_handlers={
            ValueError: (False, "Invalid tracer configuration"),
            AttributeError: (False, "Missing required tracer parameters"),
        },
        default_return=False,
        context="advanced_tracer.initialize",
    )
    async def initialize(self) -> bool:
        """Initialize the advanced tracer."""
        self.logger.info("Initializing AdvancedTracer ...")
        # Minimal sanity checks
        if not 0.0 <= self.trace_sampling_rate <= 1.0:
            self.logger.error("Invalid trace_sampling_rate; must be within [0, 1]")
            return False
        self.logger.info("✅ AdvancedTracer initialization completed")
        return True

    def create_correlation_id(self) -> str:
        """Create a new correlation ID."""
        return str(uuid.uuid4())

    @handles_errors(fallback=None)
    def start_span(
        self,
        correlation_id: str,
        component_type: ComponentType,
        operation_name: str,
        parent_span_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceSpan | None:
        span = TraceSpan(
            span_id=str(uuid.uuid4()),
            correlation_id=correlation_id,
            component_type=component_type,
            operation_name=operation_name,
            start_time=datetime.now(),
            metadata=dict(metadata or {}),
            parent_span_id=parent_span_id,
        )
        return span

    @handle_errors(default_return=None, context="advanced_tracer.finish_span")
    def finish_span(
        self,
        span: TraceSpan,
        status: str = "completed",
        error_message: Optional[str] = None,
    ) -> TraceSpan | None:
        span.end_time = datetime.now()
        if span.end_time and span.start_time:
            span.duration_ms = (
                span.end_time - span.start_time
            ).total_seconds() * 1000.0
        span.status = status
        span.error_message = error_message
        return span

    @handles_errors(fallback=None)
    def record_trace(self, trace: TraceRequest) -> None:
        """Record a completed trace request."""
        self._traces[trace.correlation_id] = trace
        # Keep history bounded
        if len(self._traces) > self.max_trace_history:
            # Remove oldest by insertion order
            oldest_key = next(iter(self._traces.keys()))
            self._traces.pop(oldest_key, None)

    def get_trace(self, correlation_id: str) -> Optional[TraceRequest]:
        return self._traces.get(correlation_id)

    def get_traces_count(self) -> int:
        return len(self._traces)
