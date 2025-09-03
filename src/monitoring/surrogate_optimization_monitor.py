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

from src.core.decorators import handles_errors

import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import time
import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import threading
import queue

# Utilities
from src.utils.logger import system_logger

import os.path

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
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Process metrics from queue
                while not self.metrics_queue.empty():
                    metrics = self.metrics_queue.get_nowait()
                    self._process_metrics(metrics)
                
                # Check for alerts
                self._check_alerts()
                
                # Sleep briefly
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)
    
    def record_metrics(self, metrics: OptimizationMetrics) -> None:
        """Record optimization metrics."""
        if self.monitoring_enabled:
            self.metrics_queue.put(metrics)
        
        # Also store directly for immediate access
        self.metrics_history.append(metrics)
        
        # Update counters
        self.total_trials += 1
        if metrics.evaluation_type == 'expensive':
            self.expensive_evaluations += 1
        else:
            self.surrogate_evaluations += 1
    
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
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {}
        
        # Calculate basic statistics
        total_time = time.time() - self.start_time
        expensive_ratio = self.expensive_evaluations / max(self.total_trials, 1)
        
        # Surrogate accuracy
        accuracy_metrics = []
        for metrics in self.metrics_history:
            if metrics.actual_score is not None:
                accuracy = self._calculate_surrogate_accuracy(metrics)
                accuracy_metrics.append(accuracy)
        
        avg_accuracy = np.mean(accuracy_metrics) if accuracy_metrics else 0.0
        
        # Convergence analysis
        scores = [m.actual_score or m.surrogate_score for m in self.metrics_history]
        best_score = max(scores) if scores else 0.0
        convergence_rate = self._calculate_convergence_rate(scores)
        
        # Uncertainty analysis
        uncertainties = [m.uncertainty for m in self.metrics_history]
        avg_uncertainty = np.mean(uncertainties) if uncertainties else 0.0
        
        # Performance metrics
        training_times = [m.training_time for m in self.metrics_history]
        avg_training_time = np.mean(training_times) if training_times else 0.0
        
        prediction_times = [m.prediction_time for m in self.metrics_history]
        avg_prediction_time = np.mean(prediction_times) if prediction_times else 0.0
        
        return {
            'total_trials': self.total_trials,
            'expensive_evaluations': self.expensive_evaluations,
            'surrogate_evaluations': self.surrogate_evaluations,
            'expensive_evaluation_ratio': expensive_ratio,
            'total_time': total_time,
            'avg_surrogate_accuracy': avg_accuracy,
            'best_score': best_score,
            'convergence_rate': convergence_rate,
            'avg_uncertainty': avg_uncertainty,
            'avg_training_time': avg_training_time,
            'avg_prediction_time': avg_prediction_time,
            'alerts_count': len(self.alerts),
            'performance_efficiency': self._calculate_performance_efficiency()
        }
    
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
    
    def generate_performance_report(self, filepath: Optional[str] = None) -> str:
        """Generate comprehensive performance report."""
        summary = self.get_performance_summary()
        
        # Create report
        report_lines = [
            "=" * 80,
            "SURROGATE OPTIMIZATION PERFORMANCE REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "PERFORMANCE SUMMARY:",
            f"  Total Trials: {summary.get('total_trials', 0)}",
            f"  Expensive Evaluations: {summary.get('expensive_evaluations', 0)}",
            f"  Surrogate Evaluations: {summary.get('surrogate_evaluations', 0)}",
            f"  Expensive Evaluation Ratio: {summary.get('expensive_evaluation_ratio', 0):.2%}",
            f"  Total Time: {summary.get('total_time', 0):.2f}s",
            "",
            "ACCURACY METRICS:",
            f"  Average Surrogate Accuracy: {summary.get('avg_surrogate_accuracy', 0):.3f}",
            f"  Best Score Achieved: {summary.get('best_score', 0):.4f}",
            f"  Convergence Rate: {summary.get('convergence_rate', 0):.4f}",
            f"  Average Uncertainty: {summary.get('avg_uncertainty', 0):.3f}",
            "",
            "TIMING METRICS:",
            f"  Average Training Time: {summary.get('avg_training_time', 0):.3f}s",
            f"  Average Prediction Time: {summary.get('avg_prediction_time', 0):.3f}s",
            "",
            "EFFICIENCY METRICS:",
            f"  Overall Performance Efficiency: {summary.get('performance_efficiency', 0):.3f}",
            "",
            "ALERTS:",
            f"  Total Alerts: {summary.get('alerts_count', 0)}"
        ]
        
        # Add recent alerts
        if self.alerts:
            report_lines.append("  Recent Alerts:")
            for alert in self.alerts[-5:]:  # Last 5 alerts
                report_lines.append(f"    [{alert.severity.upper()}] {alert.message}")
        
        report = "\n".join(report_lines)
        
        # Save to file if specified
        if filepath:
            with open(filepath, 'w') as f:
                f.write(report)
            self.logger.info(f"Performance report saved to {filepath}")
        
        return report
    
    def create_performance_visualizations(self, save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """Create performance visualization plots."""
        if not self.metrics_history:
            return {}
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        figures = {}
        
        # 1. Score progression
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Surrogate Optimization Performance Analysis', fontsize=16)
        
        # Score progression
        scores = [m.actual_score or m.surrogate_score for m in self.metrics_history]
        trial_ids = [m.trial_id for m in self.metrics_history]
        
        axes[0, 0].plot(trial_ids, scores, 'b-', alpha=0.7, linewidth=2)
        axes[0, 0].set_title('Score Progression')
        axes[0, 0].set_xlabel('Trial ID')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Uncertainty progression
        uncertainties = [m.uncertainty for m in self.metrics_history]
        axes[0, 1].plot(trial_ids, uncertainties, 'r-', alpha=0.7, linewidth=2)
        axes[0, 1].set_title('Uncertainty Progression')
        axes[0, 1].set_xlabel('Trial ID')
        axes[0, 1].set_ylabel('Uncertainty')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Evaluation type distribution
        eval_types = [m.evaluation_type for m in self.metrics_history]
        expensive_count = eval_types.count('expensive')
        surrogate_count = eval_types.count('surrogate')
        
        axes[1, 0].pie([expensive_count, surrogate_count], 
                      labels=['Expensive', 'Surrogate'], 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Evaluation Type Distribution')
        
        # Training time distribution
        training_times = [m.training_time for m in self.metrics_history]
        axes[1, 1].hist(training_times, bins=20, alpha=0.7, color='green')
        axes[1, 1].set_title('Training Time Distribution')
        axes[1, 1].set_xlabel('Training Time (s)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['performance_analysis'] = fig
        
        # 2. Accuracy analysis
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Surrogate Accuracy Analysis', fontsize=16)
        
        # Accuracy over time
        accuracy_metrics = []
        accuracy_trials = []
        for i, metrics in enumerate(self.metrics_history):
            if metrics.actual_score is not None:
                accuracy = self._calculate_surrogate_accuracy(metrics)
                accuracy_metrics.append(accuracy)
                accuracy_trials.append(metrics.trial_id)
        
        if accuracy_metrics:
            axes[0].plot(accuracy_trials, accuracy_metrics, 'g-', alpha=0.7, linewidth=2)
            axes[0].axhline(y=self.alert_thresholds['surrogate_accuracy_threshold'], 
                           color='r', linestyle='--', alpha=0.7, label='Threshold')
            axes[0].set_title('Surrogate Accuracy Over Time')
            axes[0].set_xlabel('Trial ID')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Accuracy distribution
        if accuracy_metrics:
            axes[1].hist(accuracy_metrics, bins=15, alpha=0.7, color='orange')
            axes[1].axvline(x=self.alert_thresholds['surrogate_accuracy_threshold'], 
                           color='r', linestyle='--', alpha=0.7, label='Threshold')
            axes[1].set_title('Surrogate Accuracy Distribution')
            axes[1].set_xlabel('Accuracy')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        figures['accuracy_analysis'] = fig
        
        # Save figures if directory specified
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            for name, fig in figures.items():
                filepath = os.path.join(save_dir, f"{name}.png")
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                self.logger.info(f"Saved visualization: {filepath}")
        
        return figures
    
    def get_recent_alerts(self, hours: int = 24) -> List[PerformanceAlert]:
        """Get alerts from the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
    
    def clear_old_metrics(self, days: int = 7) -> None:
        """Clear metrics older than N days."""
        cutoff_time = time.time() - (days * 24 * 3600)
        self.metrics_history = [
            metrics for metrics in self.metrics_history 
            if metrics.timestamp >= cutoff_time
        ]
        self.logger.info(f"Cleared metrics older than {days} days")
    
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to CSV file."""
        if not self.metrics_history:
            self.logger.warning("No metrics to export")
            return
        
        # Convert to DataFrame
        data = []
        for metrics in self.metrics_history:
            data.append(asdict(metrics))
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Exported {len(data)} metrics to {filepath}")
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Stopped real-time monitoring")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop_monitoring()