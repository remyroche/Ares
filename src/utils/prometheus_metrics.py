"""
Prometheus metrics collection utility for training step validators.
"""

import logging
import socket
from typing import Optional

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )
    from prometheus_client.exposition import start_http_server
    _PROM_AVAILABLE = True
    _PROM_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - optional dependency fallback
    Counter = Gauge = Histogram = None  # type: ignore[assignment]
    generate_latest = start_http_server = None  # type: ignore[assignment]
    _PROM_AVAILABLE = False
    _PROM_IMPORT_ERROR = e

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """Prometheus metrics collection for training step validators."""

    def __init__(self, port: int = 8000):
        """Initialize PrometheusMetrics."""
        self.port = port
        self.metrics_initialized = False
        self.logger = logger

        if not _PROM_AVAILABLE:
            self.logger.info(
                "Prometheus client not available; metrics disabled. Error: %s",
                str(_PROM_IMPORT_ERROR),
            )
            # Create no-op attribute placeholders to avoid attribute errors
            self.step_execution_duration = None
            self.step_success_counter = None
            self.step_failure_counter = None
            self.data_quality_score = None
            self.data_size_gauge = None
            self.data_completeness = None
            self.model_accuracy = None
            self.model_loss = None
            self.memory_usage = None
            self.cpu_usage = None
            self.validation_passed = None
            self.validation_failed = None
            return

        # Step execution metrics
        self.step_execution_duration = Histogram(
            "step_execution_duration_seconds",
            "Time spent executing training steps",
            ["step_name", "status"],
        )

        self.step_success_counter = Counter(
            "step_success_total",
            "Number of successful step executions",
            ["step_name"],
        )

        self.step_failure_counter = Counter(
            "step_failure_total",
            "Number of failed step executions",
            ["step_name", "error_type"],
        )

        # Data quality metrics
        self.data_quality_score = Gauge(
            "data_quality_score",
            "Data quality score (0-1)",
            ["step_name", "data_type"],
        )

        self.data_size_gauge = Gauge(
            "data_size_records",
            "Number of records in dataset",
            ["step_name", "data_type"],
        )

        self.data_completeness = Gauge(
            "data_completeness_ratio",
            "Ratio of complete data (0-1)",
            ["step_name", "data_type"],
        )

        # Model performance metrics
        self.model_accuracy = Gauge(
            "model_accuracy",
            "Model accuracy score",
            ["step_name", "model_type"],
        )

        self.model_loss = Gauge(
            "model_loss",
            "Model loss value",
            ["step_name", "model_type"],
        )

        # System metrics
        self.memory_usage = Gauge(
            "memory_usage_bytes",
            "Memory usage in bytes",
            ["step_name"],
        )

        self.cpu_usage = Gauge(
            "cpu_usage_percent",
            "CPU usage percentage",
            ["step_name"],
        )

        # Validation metrics
        self.validation_passed = Counter(
            "validation_passed_total",
            "Number of passed validations",
            ["step_name", "validation_type"],
        )

        self.validation_failed = Counter(
            "validation_failed_total",
            "Number of failed validations",
            ["step_name", "validation_type", "reason"],
        )

        self._start_metrics_server()

    def _start_metrics_server(self) -> None:
        """Start the Prometheus metrics server."""
        if not _PROM_AVAILABLE:
            return

        # Check if server is already running on this port
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(("localhost", self.port))
            sock.close()
            if result == 0:
                # Port is already in use, don't start another server
                self.logger.info(
                    f"Prometheus metrics server already running on port {self.port}"
                )
                self.metrics_initialized = True
                return
        except Exception as e:
            self.logger.warning(f"Port {self.port} check failed: {e}")

        try:
            start_http_server(self.port)
            self.logger.info(f"Prometheus metrics server started on port {self.port}")
            self.metrics_initialized = True
        except Exception as e:
            self.logger.error(f"Failed to start Prometheus metrics server: {e}")

    def record_step_execution(
        self, step_name: str, duration: float, status: str
    ) -> None:
        """Record step execution metrics."""
        if not _PROM_AVAILABLE:
            return
        
        self.step_execution_duration.labels(
            step_name=step_name, status=status
        ).observe(duration)

        if status == "SUCCESS":
            self.step_success_counter.labels(step_name=step_name).inc()
        else:
            self.step_failure_counter.labels(
                step_name=step_name,
                error_type=status,
            ).inc()

    def record_data_quality(
        self, step_name: str, data_type: str, quality_score: float
    ) -> None:
        """Record data quality metrics."""
        if not _PROM_AVAILABLE:
            return
        
        self.data_quality_score.labels(
            step_name=step_name, data_type=data_type
        ).set(quality_score)

    def record_data_size(
        self, step_name: str, data_type: str, size: int
    ) -> None:
        """Record data size metrics."""
        if not _PROM_AVAILABLE:
            return
        
        self.data_size_gauge.labels(
            step_name=step_name, data_type=data_type
        ).set(size)

    def record_data_completeness(
        self, step_name: str, data_type: str, completeness: float
    ) -> None:
        """Record data completeness metrics."""
        if not _PROM_AVAILABLE:
            return
        
        self.data_completeness.labels(
            step_name=step_name, data_type=data_type
        ).set(completeness)

    def record_model_performance(
        self, step_name: str, model_type: str, accuracy: float, loss: float
    ) -> None:
        """Record model performance metrics."""
        if not _PROM_AVAILABLE:
            return
        
        self.model_accuracy.labels(
            step_name=step_name, model_type=model_type
        ).set(accuracy)
        self.model_loss.labels(
            step_name=step_name, model_type=model_type
        ).set(loss)

    def record_system_metrics(
        self, step_name: str, memory_bytes: int, cpu_percent: float
    ) -> None:
        """Record system metrics."""
        if not _PROM_AVAILABLE:
            return
        
        self.memory_usage.labels(step_name=step_name).set(memory_bytes)
        self.cpu_usage.labels(step_name=step_name).set(cpu_percent)

    def record_validation_result(
        self, step_name: str, validation_type: str, passed: bool, reason: str = ""
    ) -> None:
        """Record validation results."""
        if not _PROM_AVAILABLE:
            return
        
        if passed:
            self.validation_passed.labels(
                step_name=step_name,
                validation_type=validation_type,
            ).inc()
        else:
            self.validation_failed.labels(
                step_name=step_name,
                validation_type=validation_type,
                reason=reason,
            ).inc()

    def get_metrics(self) -> str:
        """Get the latest metrics in Prometheus format."""
        if not _PROM_AVAILABLE or generate_latest is None:
            # Fallback implementation
            return ""
        return generate_latest()  # type: ignore[return-value]


# Global metrics instance (singleton)
_metrics_instance: Optional[PrometheusMetrics] = None


def get_metrics() -> PrometheusMetrics:
    """Get the global metrics instance (singleton pattern)."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = PrometheusMetrics()
    return _metrics_instance


# For backward compatibility
metrics = get_metrics()
