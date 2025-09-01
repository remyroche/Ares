#!/usr/bin/env python3
"""
Surrogate Optimization Monitoring System

This module provides comprehensive monitoring capabilities for surrogate optimization:
- Real-time performance tracking
- Automated reporting
- Performance alerts
- Historical analysis
- Dashboard integration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import time
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import threading
import queue

# Utilities
from src.utils.logger import system_logger


@dataclass
class OptimizationMetrics:
    """Data class for optimization metrics."""
    timestamp: float
    trial_id: int
    surrogate_score: float
    actual_score: Optional[float]
    uncertainty: float
    evaluation_type: str
    model_type: str
    training_time: float
    prediction_time: float
    memory_usage: float
    cpu_usage: float


@dataclass
class PerformanceAlert:
    """Data class for performance alerts."""
    timestamp: float
    alert_type: str
    severity: str
    message: str
    metrics: Dict[str, Any]


class SurrogateOptimizationMonitor:
    """Comprehensive monitoring system for surrogate optimization."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("SurrogateOptimizationMonitor")

        # Monitoring state
        self.metrics_history: List[OptimizationMetrics] = []
        self.alerts: List[PerformanceAlert] = []
        self.performance_thresholds = config.get('performance_thresholds', {})
        self.monitoring_enabled = config.get('monitoring_enabled', True)

        # Real-time monitoring
        self.metrics_queue = queue.Queue()
        self.monitoring_thread = None
        self.is_monitoring = False

        # Performance tracking
        self.start_time = time.time()
        self.total_trials = 0
        self.expensive_evaluations = 0
        self.surrogate_evaluations = 0

        # Alert thresholds
        self.alert_thresholds = {
            'surrogate_accuracy_threshold': 0.7,
            'convergence_stall_threshold': 10,
            'memory_usage_threshold': 0.8,
            'training_time_threshold': 60.0,
            'uncertainty_threshold': 0.5
        }

        # Initialize monitoring
        if self.monitoring_enabled:
            self._start_monitoring()

    def _start_monitoring(self) -> None:
        """Start real-time monitoring thread."""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("Started real-time monitoring")

    def _process_metrics(self, metrics: OptimizationMetrics) -> None:
        """Process and analyze metrics."""
        # Check for immediate alerts
        self._check_immediate_alerts(metrics)

        # Update performance tracking
        self._update_performance_tracking(metrics)

    def _check_immediate_alerts(self, metrics: OptimizationMetrics) -> None:
        """Check for immediate alerts based on current metrics."""
        alerts = []

        # Check surrogate accuracy
        if metrics.actual_score is not None:
            accuracy = self._calculate_surrogate_accuracy(metrics)
            if accuracy < self.alert_thresholds['surrogate_accuracy_threshold']:
                alerts.append(PerformanceAlert(
                    timestamp=time.time(),
                    alert_type="low_surrogate_accuracy",
                    severity="warning",
                    message=f"Low surrogate accuracy: {accuracy:.3f}",
                    metrics={"accuracy": accuracy, "trial_id": metrics.trial_id}
                ))

        # Check training time
        if metrics.training_time > self.alert_thresholds['training_time_threshold']:
            alerts.append(PerformanceAlert(
                timestamp=time.time(),
                alert_type="slow_training",
                severity="warning",
                message=f"Slow training time: {metrics.training_time:.2f}s",
                metrics={"training_time": metrics.training_time, "trial_id": metrics.trial_id}
            ))

        # Check memory usage
        if metrics.memory_usage > self.alert_thresholds['memory_usage_threshold']:
            alerts.append(PerformanceAlert(
                timestamp=time.time(),
                alert_type="high_memory_usage",
                severity="critical",
                message=f"High memory usage: {metrics.memory_usage:.1%}",
                metrics={"memory_usage": metrics.memory_usage, "trial_id": metrics.trial_id}
            ))

        # Add alerts
        for alert in alerts:
            self.alerts.append(alert)
            self.logger.warning(f"Alert: {alert.message}")

    def _calculate_surrogate_accuracy(self, metrics: OptimizationMetrics) -> float:
        """Calculate surrogate accuracy for current trial."""
        if metrics.actual_score is None:
            return 0.0

        # Simple accuracy based on relative error
        relative_error = abs(metrics.surrogate_score - metrics.actual_score) / (abs(metrics.actual_score) + 1e-8)
        return max(0.0, 1.0 - relative_error)

    def _update_performance_tracking(self, metrics: OptimizationMetrics) -> None:
        """Update performance tracking statistics."""
        # This could be extended with more sophisticated tracking
        pass

    def _check_alerts(self) -> None:
        """Check for periodic alerts."""
        if len(self.metrics_history) < 10:
            return

        # Check for convergence stall
        recent_metrics = self.metrics_history[-10:]
        recent_scores = [m.actual_score or m.surrogate_score for m in recent_metrics]

        if len(recent_scores) >= 5:
            improvement = max(recent_scores) - min(recent_scores)
            if improvement < 0.001:  # No significant improvement
                stall_alert = PerformanceAlert(
                    timestamp=time.time(),
                    alert_type="convergence_stall",
                    severity="warning",
                    message="Possible convergence stall detected",
                    metrics={"improvement": improvement, "trials_checked": len(recent_scores)}
                )
                self.alerts.append(stall_alert)
                self.logger.warning("Convergence stall detected")

    def _calculate_convergence_rate(self, scores: List[float]) -> float:
        """Calculate convergence rate."""
        if len(scores) < 2:
            return 0.0

        # Calculate improvement rate
        improvements = []
        for i in range(1, len(scores)):
            improvement = scores[i] - scores[i-1]
            improvements.append(max(0, improvement))

        return np.mean(improvements) if improvements else 0.0

    def _calculate_performance_efficiency(self) -> float:
        """Calculate overall performance efficiency."""
        if not self.metrics_history:
            return 0.0

        # Combine multiple factors
        factors = []

        # Time efficiency (faster is better)
        avg_training_time = np.mean([m.training_time for m in self.metrics_history])
        time_efficiency = max(0, 1.0 - avg_training_time / 60.0)  # Normalize to 1 minute
        factors.append(time_efficiency)

        # Accuracy efficiency
        accuracy_metrics = []
        for metrics in self.metrics_history:
            if metrics.actual_score is not None:
                accuracy = self._calculate_surrogate_accuracy(metrics)
                accuracy_metrics.append(accuracy)
        accuracy_efficiency = np.mean(accuracy_metrics) if accuracy_metrics else 0.0
        factors.append(accuracy_efficiency)

        # Cost efficiency (fewer expensive evaluations)
        cost_efficiency = 1.0 - (self.expensive_evaluations / max(self.total_trials, 1))
        factors.append(cost_efficiency)

        return np.mean(factors)
