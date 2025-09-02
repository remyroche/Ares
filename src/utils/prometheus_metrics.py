"""
Prometheus Metrics Module

This module provides Prometheus metrics collection for the Ares trading system,
including step execution metrics, data quality metrics, and system performance metrics.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Try to import Prometheus client
try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest
    from prometheus_client.exposition import start_http_server
    _PROM_AVAILABLE = True
    _PROM_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - optional dependency fallback
    Counter = Gauge = Histogram = None  # type: ignore[assignment]
    generate_latest = None  # type: ignore[assignment]
    start_http_server = None  # type: ignore[assignment]
    _PROM_IMPORT_ERROR = e
    _PROM_AVAILABLE = False

from src.utils.logger import system_logger

logger = logging.getLogger(__name__)

class PrometheusMetrics:
    """Prometheus metrics collection for training step validators."""
    
    def __init__(self, port: int = 8000):
        """Initialize PrometheusMetrics."""
        self.port = port
        self.metrics_initialized = False
        
        if not _PROM_AVAILABLE:
            logger.info(
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
            "Data completeness ratio (0-1)",
            ["step_name", "data_type"],
        )
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            "model_accuracy",
            "Model accuracy score (0-1)",
            ["step_name", "model_type"],
        )
        
        self.model_loss = Gauge(
            "model_loss",
            "Model loss value",
            ["step_name", "model_type"],
        )
        
        # System performance metrics
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
            ["step_name", "validation_type"],
        )
        
        self.metrics_initialized = True
        logger.info("Prometheus metrics initialized successfully")
    
    def record_step_execution(self, step_name: str, duration: float, status: str = "completed") -> None:
        """Record step execution duration."""
        if not self.metrics_initialized or not _PROM_AVAILABLE:
            return
        
        try:
            self.step_execution_duration.labels(step_name=step_name, status=status).observe(duration)
            logger.debug(f"Recorded step execution: {step_name} - {duration:.3f}s - {status}")
        except Exception as e:
            logger.warning(f"Failed to record step execution metric: {e}")
    
    def record_step_success(self, step_name: str) -> None:
        """Record successful step execution."""
        if not self.metrics_initialized or not _PROM_AVAILABLE:
            return
        
        try:
            self.step_success_counter.labels(step_name=step_name).inc()
            logger.debug(f"Recorded step success: {step_name}")
        except Exception as e:
            logger.warning(f"Failed to record step success metric: {e}")
    
    def record_step_failure(self, step_name: str, error_type: str = "unknown") -> None:
        """Record failed step execution."""
        if not self.metrics_initialized or not _PROM_AVAILABLE:
            return
        
        try:
            self.step_failure_counter.labels(step_name=step_name, error_type=error_type).inc()
            logger.debug(f"Recorded step failure: {step_name} - {error_type}")
        except Exception as e:
            logger.warning(f"Failed to record step failure metric: {e}")
    
    def record_data_quality_score(self, step_name: str, data_type: str, score: float) -> None:
        """Record data quality score."""
        if not self.metrics_initialized or not _PROM_AVAILABLE:
            return
        
        try:
            self.data_quality_score.labels(step_name=step_name, data_type=data_type).set(score)
            logger.debug(f"Recorded data quality score: {step_name} - {data_type} - {score:.3f}")
        except Exception as e:
            logger.warning(f"Failed to record data quality score metric: {e}")
    
    def record_data_size(self, step_name: str, data_type: str, size: int) -> None:
        """Record data size."""
        if not self.metrics_initialized or not _PROM_AVAILABLE:
            return
        
        try:
            self.data_size_gauge.labels(step_name=step_name, data_type=data_type).set(size)
            logger.debug(f"Recorded data size: {step_name} - {data_type} - {size}")
        except Exception as e:
            logger.warning(f"Failed to record data size metric: {e}")
    
    def record_data_completeness(self, step_name: str, data_type: str, completeness: float) -> None:
        """Record data completeness ratio."""
        if not self.metrics_initialized or not _PROM_AVAILABLE:
            return
        
        try:
            self.data_completeness.labels(step_name=step_name, data_type=data_type).set(completeness)
            logger.debug(f"Recorded data completeness: {step_name} - {data_type} - {completeness:.3f}")
        except Exception as e:
            logger.warning(f"Failed to record data completeness metric: {e}")
    
    def record_model_accuracy(self, step_name: str, model_type: str, accuracy: float) -> None:
        """Record model accuracy."""
        if not self.metrics_initialized or not _PROM_AVAILABLE:
            return
        
        try:
            self.model_accuracy.labels(step_name=step_name, model_type=model_type).set(accuracy)
            logger.debug(f"Recorded model accuracy: {step_name} - {model_type} - {accuracy:.3f}")
        except Exception as e:
            logger.warning(f"Failed to record model accuracy metric: {e}")
    
    def record_model_loss(self, step_name: str, model_type: str, loss: float) -> None:
        """Record model loss."""
        if not self.metrics_initialized or not _PROM_AVAILABLE:
            return
        
        try:
            self.model_loss.labels(step_name=step_name, model_type=model_type).set(loss)
            logger.debug(f"Recorded model loss: {step_name} - {model_type} - {loss:.6f}")
        except Exception as e:
            logger.warning(f"Failed to record model loss metric: {e}")
    
    def record_memory_usage(self, step_name: str, usage_bytes: int) -> None:
        """Record memory usage."""
        if not self.metrics_initialized or not _PROM_AVAILABLE:
            return
        
        try:
            self.memory_usage.labels(step_name=step_name).set(usage_bytes)
            logger.debug(f"Recorded memory usage: {step_name} - {usage_bytes} bytes")
        except Exception as e:
            logger.warning(f"Failed to record memory usage metric: {e}")
    
    def record_cpu_usage(self, step_name: str, usage_percent: float) -> None:
        """Record CPU usage."""
        if not self.metrics_initialized or not _PROM_AVAILABLE:
            return
        
        try:
            self.cpu_usage.labels(step_name=step_name).set(usage_percent)
            logger.debug(f"Recorded CPU usage: {step_name} - {usage_percent:.2f}%")
        except Exception as e:
            logger.warning(f"Failed to record CPU usage metric: {e}")
    
    def record_validation_result(self, step_name: str, validation_type: str, passed: bool) -> None:
        """Record validation result."""
        if not self.metrics_initialized or not _PROM_AVAILABLE:
            return
        
        try:
            if passed:
                self.validation_passed.labels(step_name=step_name, validation_type=validation_type).inc()
                logger.debug(f"Recorded validation passed: {step_name} - {validation_type}")
            else:
                self.validation_failed.labels(step_name=step_name, validation_type=validation_type).inc()
                logger.debug(f"Recorded validation failed: {step_name} - {validation_type}")
        except Exception as e:
            logger.warning(f"Failed to record validation result metric: {e}")
    
    def record_validation_time(self, step_name: str, validation_time: float) -> None:
        """Record validation time."""
        if not self.metrics_initialized or not _PROM_AVAILABLE:
            return
        
        try:
            self.step_execution_duration.labels(step_name=step_name, status="validation").observe(validation_time)
            logger.debug(f"Recorded validation time: {step_name} - {validation_time:.3f}s")
        except Exception as e:
            logger.warning(f"Failed to record validation time metric: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        if not self.metrics_initialized or not _PROM_AVAILABLE:
            return {"error": "Prometheus metrics not available"}
        
        try:
            return {
                "prometheus_available": _PROM_AVAILABLE,
                "metrics_initialized": self.metrics_initialized,
                "port": self.port,
                "available_metrics": [
                    "step_execution_duration",
                    "step_success_counter",
                    "step_failure_counter",
                    "data_quality_score",
                    "data_size_gauge",
                    "data_completeness",
                    "model_accuracy",
                    "model_loss",
                    "memory_usage",
                    "cpu_usage",
                    "validation_passed",
                    "validation_failed"
                ]
            }
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {"error": str(e)}
    
    def start_metrics_server(self) -> bool:
        """Start the Prometheus metrics HTTP server."""
        if not _PROM_AVAILABLE:
            logger.warning("Cannot start metrics server: Prometheus client not available")
            return False
        
        try:
            start_http_server(self.port)
            logger.info(f"Started Prometheus metrics server on port {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            return False
    
    def generate_metrics(self) -> str:
        """Generate current metrics in Prometheus format."""
        if not _PROM_AVAILABLE:
            return "# Prometheus metrics not available\n"
        
        try:
            return generate_latest().decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to generate metrics: {e}")
            return f"# Error generating metrics: {e}\n"

# Global metrics instance
metrics = PrometheusMetrics()

# Convenience function for creating metrics instance
def create_metrics(port: int = 8000) -> PrometheusMetrics:
    """Create a new PrometheusMetrics instance."""
    return PrometheusMetrics(port)
