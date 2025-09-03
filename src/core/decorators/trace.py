from __future__ import annotations

"""
Distributed tracing decorators for observability.

Provides decorators for creating and managing trace spans,
compatible with OpenTelemetry and other tracing systems.
"""

import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .compose import P, R, uniform_wrapper
from .logging import get_correlation_id
import asyncio

# Context variable for current trace
current_trace_var: ContextVar[Optional["TraceContext"]] = ContextVar(
    "current_trace", default=None
)


class SpanKind(Enum):
    """Types of spans in distributed tracing."""

    INTERNAL = "internal"
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(Enum):
    """Status of a span."""

    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class Span:
    """Represents a single span in a trace."""

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    kind: SpanKind = SpanKind.INTERNAL
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[tuple[str, float, dict[str, Any]]] = field(default_factory=list)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute on the span."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] = None) -> None:
        """Add an event to the span."""
        self.events.append((name, time.time(), attributes or {}))

    def set_status(self, status: SpanStatus, description: str = None) -> None:
        """Set the span status."""
        self.status = status
        if description:
            self.attributes["status_description"] = description

    def end(self) -> None:
        """End the span."""
        if self.end_time is None:
            self.end_time = time.time()

    @property
    def duration_ms(self) -> float | None:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None


@dataclass
class TraceContext:
    """Context for a complete trace."""

    trace_id: str
    spans: list[Span] = field(default_factory=list)
    baggage: dict[str, str] = field(default_factory=dict)

    def create_span(
        self,
        name: str,
        parent_span: Span | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
    ) -> Span:
        """Create a new span in this trace."""
        import uuid

        span = Span(
            name=name,
            trace_id=self.trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_span.span_id if parent_span else None,
            kind=kind,
        )

        # Add correlation ID as attribute
        span.set_attribute("correlation_id", get_correlation_id())

        # Add baggage as attributes
        for key, value in self.baggage.items():
            span.set_attribute(f"baggage.{key}", value)

        self.spans.append(span)
        return span


# Simple in-memory trace storage (replace with real backend in production)
_trace_storage: dict[str, TraceContext] = {}


def get_current_trace() -> TraceContext | None:
    """Get the current trace context."""
    return current_trace_var.get()


def set_current_trace(trace: TraceContext) -> None:
    """Set the current trace context."""
    current_trace_var.set(trace)


def create_trace(trace_id: str | None = None) -> TraceContext:
    """Create a new trace context."""
    import uuid

    if trace_id is None:
        trace_id = str(uuid.uuid4())

    trace = TraceContext(trace_id=trace_id)
    _trace_storage[trace_id] = trace
    return trace


def traced(
    *,
    span_name: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] = None,
    record_args: bool = False,
    record_result: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Create a traced span for the decorated function.

    Args:
        span_name: Name for the span (defaults to function name)
        kind: Type of span
        attributes: Additional attributes to add to span
        record_args: Whether to record function arguments
        record_result: Whether to record function result

    Example:
        @traced(span_name="fetch_user_data", kind=SpanKind.CLIENT)
        def fetch_user(user_id: str) -> dict:
            return api.get_user(user_id)
    """

    def sync_handler(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        # Get or create trace context
        trace = get_current_trace()
        if trace is None:
            trace = create_trace()
            set_current_trace(trace)

        # Create span
        name = span_name or func.__name__
        parent_span = None  # Could be enhanced to track parent spans
        span = trace.create_span(name, parent_span, kind)

        # Add function metadata
        span.set_attribute("function", func.__name__)
        span.set_attribute("module", func.__module__)

        # Add custom attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        # Record arguments if requested
        if record_args:
            span.set_attribute("args", str(args))
            span.set_attribute("kwargs", str(kwargs))

        try:
            # Execute function
            result = func(*args, **kwargs)

            # Record result if requested
            if record_result:
                span.set_attribute("result", str(result))

            # Set success status
            span.set_status(SpanStatus.OK)

            return result

        except Exception as e:
            # Record error
            span.set_status(SpanStatus.ERROR, str(e))
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            span.add_event("exception", {"type": type(e).__name__, "message": str(e)})
            raise

        finally:
            # End span
            span.end()

    async def async_handler(
        func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        # Get or create trace context
        trace = get_current_trace()
        if trace is None:
            trace = create_trace()
            set_current_trace(trace)

        # Create span
        name = span_name or func.__name__
        parent_span = None
        span = trace.create_span(name, parent_span, kind)

        # Add function metadata
        span.set_attribute("function", func.__name__)
        span.set_attribute("module", func.__module__)

        # Add custom attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        # Record arguments if requested
        if record_args:
            span.set_attribute("args", str(args))
            span.set_attribute("kwargs", str(kwargs))

        try:
            # Execute function
            result = await func(*args, **kwargs)

            # Record result if requested
            if record_result:
                span.set_attribute("result", str(result))

            # Set success status
            span.set_status(SpanStatus.OK)

            return result

        except Exception as e:
            # Record error
            span.set_status(SpanStatus.ERROR, str(e))
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            span.add_event("exception", {"type": type(e).__name__, "message": str(e)})
            raise

        finally:
            # End span
            span.end()

    return uniform_wrapper(
        f"traced({span_name or 'auto'})", sync_handler, async_handler
    )


def span_event(name: str, attributes: dict[str, Any] = None) -> None:
    """
    Add an event to the current span.

    This should be called within a traced function to add events.

    Args:
        name: Event name
        attributes: Event attributes

    Example:
        @traced()
        def process_order(order_id: str):
            span_event("order_validated", {"order_id": order_id})
            # ... process order ...
            span_event("order_processed", {"status": "success"})
    """
    trace = get_current_trace()
    if trace and trace.spans:
        # Add event to the most recent span
        current_span = trace.spans[-1]
        current_span.add_event(name, attributes)


def span_attribute(key: str, value: Any) -> None:
    """
    Add an attribute to the current span.

    This should be called within a traced function to add attributes.

    Args:
        key: Attribute key
        value: Attribute value

    Example:
        @traced()
        def fetch_data(user_id: str):
            span_attribute("user_id", user_id)
            data = database.get_user(user_id)
            span_attribute("data_size", len(data))
            return data
    """
    trace = get_current_trace()
    if trace and trace.spans:
        # Add attribute to the most recent span
        current_span = trace.spans[-1]
        current_span.set_attribute(key, value)


def trace_method(
    cls: type = None,
    *,
    span_prefix: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Union[type, Callable[[type], type]]:
    """
    Class decorator to trace all methods.

    Args:
        span_prefix: Prefix for span names (defaults to class name)
        kind: Default span kind for methods

    Example:
        @trace_method(span_prefix="UserService")
        class UserService:
            def get_user(self, user_id: str) -> dict:
                return self.db.get_user(user_id)

            def update_user(self, user_id: str, data: dict) -> dict:
                return self.db.update_user(user_id, data)
    """

    def decorator(cls: type) -> type:
        prefix = span_prefix or cls.__name__

        # Trace all methods
        for name, method in cls.__dict__.items():
            if callable(method) and not name.startswith("_"):
                span_name = f"{prefix}.{name}"
                traced_method = traced(span_name=span_name, kind=kind)(method)
                setattr(cls, name, traced_method)

        return cls

    if cls is None:
        return decorator
    return decorator(cls)


def get_trace_summary(trace_id: str) -> dict[str, Any] | None:
    """
    Get a summary of a trace.

    Args:
        trace_id: ID of the trace

    Returns:
        Summary dict or None if trace not found
    """
    trace = _trace_storage.get(trace_id)
    if not trace:
        return None

    total_duration = 0
    error_count = 0
    span_summaries = []

    for span in trace.spans:
        span_summary = {
            "name": span.name,
            "span_id": span.span_id,
            "parent_span_id": span.parent_span_id,
            "kind": span.kind.value,
            "status": span.status.value,
            "duration_ms": span.duration_ms,
            "attributes": span.attributes,
            "events": [
                {"name": event[0], "timestamp": event[1], "attributes": event[2]}
                for event in span.events
            ],
        }
        span_summaries.append(span_summary)

        if span.duration_ms:
            total_duration += span.duration_ms

        if span.status == SpanStatus.ERROR:
            error_count += 1

    return {
        "trace_id": trace_id,
        "span_count": len(trace.spans),
        "error_count": error_count,
        "total_duration_ms": total_duration,
        "baggage": trace.baggage,
        "spans": span_summaries,
    }
