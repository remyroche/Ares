#!/usr/bin/env python3
"""
Advanced Tracing System with Correlation IDs

This module provides comprehensive request/response tracing across all components
of the Ares trading bot with correlation IDs for debugging and performance analysis.
"""

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error,
)


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
    end_time: datetime | None = None
    duration_ms: float | None = None
    status: str = "running"  # "running", "completed", "failed"
    error_message: str | None = None
    metadata: dict[str, Any] = None
    parent_span_id: str | None = None
    child_span_ids: list[str] = None


@dataclass
class TraceRequest:
    """Complete trace request with all spans."""

    correlation_id: str
    request_timestamp: datetime
    component_path: list[ComponentType]
    spans: list[TraceSpan]
    response_timestamp: datetime | None = None
    total_duration_ms: float | None = None
    status: str = "running"  # "running", "completed", "failed"
    error_info: dict[str, Any] | None = None
    performance_metrics: dict[str, float] = None
    metadata: dict[str, Any] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for tracing."""

    total_duration_ms: float
    component_durations: dict[str, float]
    bottleneck_component: str
    throughput_ops_per_sec: float
    error_rate: float
    success_rate: float


class AdvancedTracer:
    """
    Advanced tracing system with correlation IDs for comprehensive
    request/response tracking across all components.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize advanced tracer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("AdvancedTracer")

        # Tracer configuration
        self.tracer_config = config.get("advanced_tracer", {})
        self.enable_tracing = self.tracer_config.get("enable_tracing", True)
        self.correlation_id_header = self.tracer_config.get(
            "correlation_id_header",
            "X-Correlation-ID",
        )
        self.trace_sampling_rate = self.tracer_config.get("trace_sampling_rate", 1.0)
        self.max_trace_history = self.tracer_config.get("max_trace_history", 10000)
        self.enable_performance_tracing = self.tracer_config.get(
            "enable_performance_tracing",
            True,
        )
        self.enable_error_tracing = self.tracer_config.get("enable_error_tracing", True)

        # Trace storage
        self.trace_requests: dict[str, TraceRequest] = {}
        self.active_spans: dict[str, TraceSpan] = {}
        self.performance_metrics: dict[str, PerformanceMetrics] = {}

        # Tracing state
        self.is_tracing = False
        self.trace_cleanup_task: asyncio.Task | None = None

        # Correlation ID management
        self.correlation_id_counter = 0

        self.logger.info("🔍 Advanced Tracer initialized")

    def generate_correlation_id(self) -> str:
        """Generate a unique correlation ID."""
        self.correlation_id_counter += 1
        timestamp = int(time.time() * 1000)
        return f"ares-{timestamp}-{self.correlation_id_counter}-{uuid.uuid4().hex[:8]}"

    def generate_span_id(self) -> str:
        """Generate a unique span ID."""
        return f"span-{uuid.uuid4().hex}"

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid tracer configuration"),
            AttributeError: (False, "Missing required tracer parameters"),
        },
        default_return=False,
        context="tracer initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the advanced tracer."""
        self.logger.info("Initializing Advanced Tracer...")

        # Initialize trace storage
        await self._initialize_trace_storage()

        # Initialize performance tracking
        if self.enable_performance_tracing:
            await self._initialize_performance_tracking()

        # Initialize error tracking
        if self.enable_error_tracing:
            await self._initialize_error_tracking()

        # Start trace cleanup task
        self.trace_cleanup_task = asyncio.create_task(self._trace_cleanup_loop())

        self.logger.info("✅ Advanced Tracer initialization completed")
        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="trace storage initialization",
    )
    async def _initialize_trace_storage(self) -> None:
        """Initialize trace storage structures."""
        # Initialize trace storage
        self.trace_requests.clear()
        self.active_spans.clear()
        self.performance_metrics.clear()

        self.logger.info("Trace storage initialized")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="performance tracking initialization",
    )
    async def _initialize_performance_tracking(self) -> None:
        """Initialize performance tracking."""
        # Initialize performance tracking structures
        self.logger.info("Performance tracking initialized")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="error tracking initialization",
    )
    async def _initialize_error_tracking(self) -> None:
        """Initialize error tracking."""
        # Initialize error tracking structures
        self.logger.info("Error tracking initialized")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="trace cleanup loop",
    )
    async def _trace_cleanup_loop(self) -> None:
        """Cleanup old traces periodically."""
        while self.is_tracing:
            await self._cleanup_old_traces()
            await asyncio.sleep(300)  # Cleanup every 5 minutes

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="trace cleanup",
    )
    async def _cleanup_old_traces(self) -> None:
        """Cleanup old traces to prevent memory issues."""
        cutoff_time = datetime.now() - timedelta(hours=24)

        # Cleanup old trace requests
        old_correlation_ids = [
            corr_id
            for corr_id, trace in self.trace_requests.items()
            if trace.request_timestamp < cutoff_time
        ]

        for corr_id in old_correlation_ids:
            del self.trace_requests[corr_id]

        # Cleanup old active spans
        old_span_ids = [
            span_id
            for span_id, span in self.active_spans.items()
            if span.start_time < cutoff_time
        ]

        for span_id in old_span_ids:
            del self.active_spans[span_id]

        # Limit trace history
        if len(self.trace_requests) > self.max_trace_history:
            # Remove oldest traces
            sorted_traces = sorted(
                self.trace_requests.items(),
                key=lambda x: x[1].request_timestamp,
            )

            excess_count = len(self.trace_requests) - self.max_trace_history
            for corr_id, _ in sorted_traces[:excess_count]:
                del self.trace_requests[corr_id]

        if old_correlation_ids or old_span_ids:
            self.logger.info(
                f"Cleaned up {len(old_correlation_ids)} old traces and {len(old_span_ids)} old spans",
            )

    @contextmanager
    def trace_request(
        self,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Context manager for tracing a complete request.

        Args:
            correlation_id: Optional correlation ID, generates one if not provided
            metadata: Optional metadata for the request

        Yields:
            TraceRequest: The trace request object
        """
        if not self.enable_tracing:
            yield None
            return

        if correlation_id is None:
            correlation_id = self.generate_correlation_id()

        trace_request = TraceRequest(
            correlation_id=correlation_id,
            request_timestamp=datetime.now(),
            component_path=[],
            spans=[],
            metadata=metadata or {},
        )

        self.trace_requests[correlation_id] = trace_request

        try:
            yield trace_request
        finally:
            trace_request.response_timestamp = datetime.now()
            if trace_request.request_timestamp and trace_request.response_timestamp:
                trace_request.total_duration_ms = (
                    trace_request.response_timestamp - trace_request.request_timestamp
                ).total_seconds() * 1000

    @asynccontextmanager
    async def trace_span(
        self,
        component_type: ComponentType,
        operation_name: str,
        correlation_id: str,
        parent_span_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Async context manager for tracing a component span.

        Args:
            component_type: Type of component being traced
            operation_name: Name of the operation
            correlation_id: Correlation ID for the request
            parent_span_id: Optional parent span ID
            metadata: Optional metadata for the span

        Yields:
            TraceSpan: The trace span object
        """
        if not self.enable_tracing:
            yield None
            return

        span_id = self.generate_span_id()
        start_time = datetime.now()

        span = TraceSpan(
            span_id=span_id,
            correlation_id=correlation_id,
            component_type=component_type,
            operation_name=operation_name,
            start_time=start_time,
            parent_span_id=parent_span_id,
            child_span_ids=[],
            metadata=metadata or {},
        )

        self.active_spans[span_id] = span

        # Add to trace request
        if correlation_id in self.trace_requests:
            self.trace_requests[correlation_id].spans.append(span)
            self.trace_requests[correlation_id].component_path.append(component_type)

        try:
            yield span
        except Exception as e:
            span.status = "failed"
            span.error_message = str(e)
            raise
        finally:
            span.end_time = datetime.now()
            if span.start_time and span.end_time:
                span.duration_ms = (
                    span.end_time - span.start_time
                ).total_seconds() * 1000

            span.status = "completed"
            del self.active_spans[span_id]

    def trace_function(
        self,
        component_type: ComponentType,
        operation_name: str | None = None,
    ):
        """
        Decorator for tracing function calls.

        Args:
            component_type: Type of component being traced
            operation_name: Optional operation name, uses function name if not provided
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not self.enable_tracing:
                    return await func(*args, **kwargs)

                # Extract correlation ID from kwargs or generate one
                correlation_id = kwargs.pop(
                    "correlation_id",
                    self.generate_correlation_id(),
                )
                metadata = kwargs.pop("trace_metadata", {})

                op_name = operation_name or func.__name__

                async with self.trace_span(
                    component_type=component_type,
                    operation_name=op_name,
                    correlation_id=correlation_id,
                    metadata=metadata,
                ) as span:
                    try:
                        result = await func(*args, **kwargs)
                        span.metadata["result_type"] = type(result).__name__
                        return result
                    except Exception as e:
                        span.status = "failed"
                        span.error_message = str(e)
                        raise

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not self.enable_tracing:
                    return func(*args, **kwargs)

                # Extract correlation ID from kwargs or generate one
                correlation_id = kwargs.pop(
                    "correlation_id",
                    self.generate_correlation_id(),
                )
                metadata = kwargs.pop("trace_metadata", {})

                op_name = operation_name or func.__name__

                with self.trace_span(
                    component_type=component_type,
                    operation_name=op_name,
                    correlation_id=correlation_id,
                    metadata=metadata,
                ) as span:
                    try:
                        result = func(*args, **kwargs)
                        span.metadata["result_type"] = type(result).__name__
                        return result
                    except Exception as e:
                        span.status = "failed"
                        span.error_message = str(e)
                        raise

            # Return async wrapper for async functions, sync wrapper for sync functions
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper

        return decorator

    def get_trace_request(self, correlation_id: str) -> TraceRequest | None:
        """Get a trace request by correlation ID."""
        return self.trace_requests.get(correlation_id)

    def get_active_spans(self) -> list[TraceSpan]:
        """Get all currently active spans."""
        return list(self.active_spans.values())

    def get_trace_statistics(self) -> dict[str, Any]:
        """Get trace statistics."""
        try:
            total_requests = len(self.trace_requests)
            active_spans = len(self.active_spans)

            # Calculate performance metrics
            completed_requests = [
                req for req in self.trace_requests.values() if req.status == "completed"
            ]

            failed_requests = [
                req for req in self.trace_requests.values() if req.status == "failed"
            ]

            avg_duration = 0
            if completed_requests:
                durations = [
                    req.total_duration_ms
                    for req in completed_requests
                    if req.total_duration_ms
                ]
                avg_duration = sum(durations) / len(durations) if durations else 0

            return {
                "total_requests": total_requests,
                "active_spans": active_spans,
                "completed_requests": len(completed_requests),
                "failed_requests": len(failed_requests),
                "success_rate": len(completed_requests) / total_requests
                if total_requests > 0
                else 0,
                "average_duration_ms": avg_duration,
                "tracing_enabled": self.enable_tracing,
            }

        except Exception:
            self.logger.exception(error("Error getting trace statistics: {e}"))
            return {}

    def export_trace_data(
        self,
        correlation_id: str | None = None,
        format: str = "json",
    ) -> str:
        """Export trace data."""
        try:
            if correlation_id:
                trace_data = self.get_trace_request(correlation_id)
                if not trace_data:
                    return "Trace not found"

                if format == "json":
                    return json.dumps(asdict(trace_data), indent=2, default=str)
                return str(trace_data)
            # Export all traces
            all_traces = {
                corr_id: asdict(trace) for corr_id, trace in self.trace_requests.items()
            }

            if format == "json":
                return json.dumps(all_traces, indent=2, default=str)
            return str(all_traces)

        except Exception as e:
            self.logger.exception(error(f"Error exporting trace data: {e}"))
            return ""

    @handle_specific_errors(
        error_handlers={
            Exception: (False, "Tracer start failed"),
        },
        default_return=False,
        context="tracer start",
    )
    async def start(self) -> bool:
        """Start the advanced tracer."""
        try:
            self.is_tracing = True
            self.logger.info("🚀 Advanced Tracer started")
            return True

        except Exception:
            self.logger.exception(error("Error starting tracer: {e}"))
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="tracer stop",
    )
    async def stop(self) -> None:
        """Stop the advanced tracer."""
        try:
            self.is_tracing = False

            if self.trace_cleanup_task:
                self.trace_cleanup_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self.trace_cleanup_task

            self.logger.info("🛑 Advanced Tracer stopped")

        except Exception:
            self.logger.exception(error("Error stopping tracer: {e}"))


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="advanced tracer setup",
)
async def setup_advanced_tracer(config: dict[str, Any]) -> AdvancedTracer | None:
    """
    Setup and initialize advanced tracer.

    Args:
        config: Configuration dictionary

    Returns:
        AdvancedTracer instance or None if setup failed
    """
    try:
        tracer = AdvancedTracer(config)

        if await tracer.initialize():
            return tracer
        return None

    except Exception:
        system_logger.exception(error("Error setting up advanced tracer: {e}"))
        return None
