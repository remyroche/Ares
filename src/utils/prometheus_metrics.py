"""
Prometheus metrics collection utility for training step validators.
"""

import time
from typing import Any, Dict, Optional
from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.exposition import start_http_server
import threading
import logging

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """Prometheus metrics collection for training step validators."""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.metrics_initialized = False
        
        # Step execution metrics
        self.step_execution_duration = Histogram(
            'step_execution_duration_seconds',
            'Time spent executing training steps',
            ['step_name', 'status']
        )
        
        self.step_success_counter = Counter(
            'step_success_total',
            'Number of successful step executions',
            ['step_name']
        )
        
        self.step_failure_counter = Counter(
            'step_failure_total',
            'Number of failed step executions',
            ['step_name', 'error_type']
        )
        
        # Data quality metrics
        self.data_quality_score = Gauge(
            'data_quality_score',
            'Data quality score (0-1)',
            ['step_name', 'data_type']
        )
        
        self.data_size_gauge = Gauge(
            'data_size_records',
            'Number of records in dataset',
            ['step_name', 'data_type']
        )
        
        self.data_completeness = Gauge(
            'data_completeness_ratio',
            'Ratio of complete data (0-1)',
            ['step_name', 'data_type']
        )
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            'model_accuracy',
            'Model accuracy score',
            ['step_name', 'model_type']
        )
        
        self.model_loss = Gauge(
            'model_loss',
            'Model loss value',
            ['step_name', 'model_type']
        )
        
        # System metrics
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            ['step_name']
        )
        
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            ['step_name']
        )
        
        # Validation metrics
        self.validation_passed = Counter(
            'validation_passed_total',
            'Number of passed validations',
            ['step_name', 'validation_type']
        )
        
        self.validation_failed = Counter(
            'validation_failed_total',
            'Number of failed validations',
            ['step_name', 'validation_type', 'reason']
        )
        
        self._start_metrics_server()
    
    def _start_metrics_server(self):
        """Start the Prometheus metrics server."""
        try:
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
            self.metrics_initialized = True
        except Exception as e:
            logger.warning(f"Failed to start Prometheus metrics server: {e}")
    
    def record_step_execution(self, step_name: str, duration: float, status: str):
        """Record step execution metrics."""
        self.step_execution_duration.labels(step_name=step_name, status=status).observe(duration)
        
        if status == "SUCCESS":
            self.step_success_counter.labels(step_name=step_name).inc()
        else:
            self.step_failure_counter.labels(step_name=step_name, error_type=status).inc()
    
    def record_data_quality(self, step_name: str, data_type: str, quality_score: float):
        """Record data quality metrics."""
        self.data_quality_score.labels(step_name=step_name, data_type=data_type).set(quality_score)
    
    def record_data_size(self, step_name: str, data_type: str, size: int):
        """Record data size metrics."""
        self.data_size_gauge.labels(step_name=step_name, data_type=data_type).set(size)
    
    def record_data_completeness(self, step_name: str, data_type: str, completeness: float):
        """Record data completeness metrics."""
        self.data_completeness.labels(step_name=step_name, data_type=data_type).set(completeness)
    
    def record_model_performance(self, step_name: str, model_type: str, accuracy: float, loss: float):
        """Record model performance metrics."""
        self.model_accuracy.labels(step_name=step_name, model_type=model_type).set(accuracy)
        self.model_loss.labels(step_name=step_name, model_type=model_type).set(loss)
    
    def record_system_metrics(self, step_name: str, memory_bytes: int, cpu_percent: float):
        """Record system metrics."""
        self.memory_usage.labels(step_name=step_name).set(memory_bytes)
        self.cpu_usage.labels(step_name=step_name).set(cpu_percent)
    
    def record_validation_result(self, step_name: str, validation_type: str, passed: bool, reason: str = ""):
        """Record validation results."""
        if passed:
            self.validation_passed.labels(step_name=step_name, validation_type=validation_type).inc()
        else:
            self.validation_failed.labels(step_name=step_name, validation_type=validation_type, reason=reason).inc()
    
    def get_metrics(self) -> str:
        """Get current metrics in Prometheus format."""
        return generate_latest()


# Global metrics instance
metrics = PrometheusMetrics()
