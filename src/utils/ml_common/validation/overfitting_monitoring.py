"""
Overfitting Monitoring for ML Common

Real-time overfitting monitoring system with learning curve analysis,
performance gap detection, and early warning system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from collections import deque
import warnings

logger = logging.getLogger(__name__)

@dataclass
class OverfittingMonitoringConfig:
    """Configuration for overfitting monitoring."""

    # Performance gap analysis
    enable_performance_gap_analysis: bool = True
    train_val_gap_threshold: float = 0.1  # 10% gap triggers warning
    max_acceptable_gap: float = 0.25  # 25% gap triggers critical alert

    # Learning curve monitoring
    enable_learning_curve_monitoring: bool = True
    learning_curve_window: int = 10
    learning_curve_min_samples: int = 50

    # Early warning system
    enable_early_warning_system: bool = True
    warning_window: int = 5
    critical_window: int = 10
    divergence_threshold: float = 0.05  # 5% divergence rate

    # Trend analysis
    enable_trend_analysis: bool = True
    trend_window: int = 15
    trend_smoothing_alpha: float = 0.3

    # Anomaly detection
    enable_anomaly_detection: bool = True
    anomaly_detection_sensitivity: float = 2.0  # Standard deviations

    # Reporting
    save_monitoring_reports: bool = True
    report_directory: str = "reports/overfitting_monitoring"
    enable_visualization: bool = True
    monitoring_interval_seconds: int = 60

    # Thresholds
    critical_overfitting_threshold: float = 0.3  # 30% overfitting rate
    warning_overfitting_threshold: float = 0.15  # 15% overfitting rate

@dataclass
class MonitoringReport:
    """Real-time overfitting monitoring report."""

    # Basic information
    model_name: str = "unknown"
    monitoring_session_id: str = None
    total_epochs: int = 0
    current_epoch: int = 0

    # Performance metrics
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    train_loss: float = 0.0
    val_loss: float = 0.0

    # Overfitting indicators
    accuracy_gap: float = 0.0
    loss_gap: float = 0.0
    overfitting_detected: bool = False
    overfitting_severity: str = "none"  # none, low, medium, high, critical

    # Learning curve analysis
    learning_curve_trend: str = "stable"  # improving, stable, degrading
    learning_curve_health: str = "good"  # good, concerning, poor

    # Early warning indicators
    early_warning_triggered: bool = False
    warning_indicators: List[str] = None
    divergence_rate: float = 0.0

    # Trend analysis
    accuracy_trend: str = "stable"  # improving, stable, declining
    loss_trend: str = "stable"  # decreasing, stable, increasing

    # Anomaly detection
    anomaly_detected: bool = False
    anomaly_score: float = 0.0
    anomaly_description: str = ""

    # Recommendations
    recommendations: List[str] = None
    immediate_actions: List[str] = None
    training_suggestions: List[str] = None

    # Metadata
    monitoring_timestamp: str = None
    config_used: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default collections."""
        if self.warning_indicators is None:
            self.warning_indicators = []
        if self.recommendations is None:
            self.recommendations = []
        if self.immediate_actions is None:
            self.immediate_actions = []
        if self.training_suggestions is None:
            self.training_suggestions = []
        if self.monitoring_timestamp is None:
            self.monitoring_timestamp = datetime.now().isoformat()
        if self.config_used is None:
            self.config_used = {}
        if self.monitoring_session_id is None:
            self.monitoring_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

class OverfittingMonitor:
    """Real-time overfitting monitoring system."""

    def __init__(self, config: Optional[OverfittingMonitoringConfig] = None):
        """
        Initialize overfitting monitoring system.

        Args:
            config: Configuration for monitoring
        """
        self.config = config or OverfittingMonitoringConfig()
        self.monitoring_history = []
        self.active_sessions = {}
        self.performance_history = {}

        # Create report directory
        if self.config.save_monitoring_reports:
            Path(self.config.report_directory).mkdir(parents=True, exist_ok=True)

        logger.info("✅ Overfitting Monitoring initialized")

    def start_monitoring_session(self,
                                model_name: str,
                                model_type: str = "unknown") -> str:
        """
        Start a new monitoring session.

        Args:
            model_name: Name of the model to monitor
            model_type: Type of model

        Returns:
            Session ID
        """
        session_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.active_sessions[session_id] = {
            'model_name': model_name,
            'model_type': model_type,
            'start_time': datetime.now(),
            'metrics_history': deque(maxlen=self.config.trend_window * 2),
            'learning_curve_data': deque(maxlen=self.config.learning_curve_window * 2),
            'anomaly_scores': deque(maxlen=50)
        }

        self.performance_history[session_id] = []

        logger.info(f"📊 Started monitoring session {session_id} for {model_name}")
        return session_id

    def monitor_training_step(self,
                             session_id: str,
                             epoch: int,
                             train_accuracy: float,
                             val_accuracy: float,
                             train_loss: float,
                             val_loss: float,
                             additional_metrics: Optional[Dict[str, float]] = None) -> MonitoringReport:
        """
        Monitor a single training step.

        Args:
            session_id: Active monitoring session ID
            epoch: Current training epoch
            train_accuracy: Training accuracy
            val_accuracy: Validation accuracy
            train_loss: Training loss
            val_loss: Validation loss
            additional_metrics: Optional additional metrics

        Returns:
            MonitoringReport with analysis
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found. Start a session first.")

        report = MonitoringReport(model_name=self.active_sessions[session_id]['model_name'])
        report.monitoring_session_id = session_id
        report.total_epochs = epoch
        report.current_epoch = epoch

        try:
            # Store current metrics
            current_metrics = {
                'epoch': epoch,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'timestamp': datetime.now()
            }

            if additional_metrics:
                current_metrics.update(additional_metrics)

            # Update session data
            session = self.active_sessions[session_id]
            session['metrics_history'].append(current_metrics)

            # Basic performance calculations
            report.train_accuracy = train_accuracy
            report.val_accuracy = val_accuracy
            report.train_loss = train_loss
            report.val_loss = val_loss

            # Calculate performance gaps
            report.accuracy_gap = abs(train_accuracy - val_accuracy)
            report.loss_gap = abs(train_loss - val_loss)

            # Performance gap analysis
            if self.config.enable_performance_gap_analysis:
                self._analyze_performance_gaps(report, session)

            # Learning curve monitoring
            if self.config.enable_learning_curve_monitoring:
                self._analyze_learning_curve(report, session)

            # Early warning system
            if self.config.enable_early_warning_system:
                self._check_early_warnings(report, session)

            # Trend analysis
            if self.config.enable_trend_analysis:
                self._analyze_trends(report, session)

            # Anomaly detection
            if self.config.enable_anomaly_detection:
                self._detect_anomalies(report, session)

            # Generate recommendations
            self._generate_monitoring_recommendations(report)

            # Store report
            self.monitoring_history.append(report)
            self.performance_history[session_id].append(current_metrics)

            # Log critical issues
            if report.overfitting_severity in ['high', 'critical']:
                logger.warning(f"🚨 High overfitting detected for {report.model_name}: {report.overfitting_severity}")
                for issue in report.immediate_actions:
                    logger.warning(f"   Action: {issue}")

            return report

        except Exception as e:
            logger.error(f"Monitoring step failed: {e}")
            report.recommendations.append(f"Monitoring error: {str(e)}")
            return report

    def _analyze_performance_gaps(self, report: MonitoringReport, session: Dict[str, Any]):
        """Analyze performance gaps between train and validation."""
        try:
            metrics_history = list(session['metrics_history'])

            if len(metrics_history) < 2:
                return

            # Calculate average gaps over recent history
            recent_metrics = list(metrics_history)[-self.config.warning_window:]
            avg_accuracy_gap = np.mean([m['train_accuracy'] - m['val_accuracy'] for m in recent_metrics])
            avg_loss_gap = np.mean([m['val_loss'] - m['train_loss'] for m in recent_metrics])

            # Detect overfitting
            if avg_accuracy_gap > self.config.max_acceptable_gap:
                report.overfitting_detected = True
                report.overfitting_severity = "critical"
                report.warning_indicators.append(f"Critical accuracy gap: {avg_accuracy_gap:.4f}")
                report.immediate_actions.append("Consider early stopping")
            elif avg_accuracy_gap > self.config.train_val_gap_threshold:
                report.overfitting_detected = True
                report.overfitting_severity = "high"
                report.warning_indicators.append(f"High accuracy gap: {avg_accuracy_gap:.4f}")
                report.immediate_actions.append("Monitor closely for overfitting")

            # Check loss divergence
            if avg_loss_gap > 0.1:  # Training loss much lower than validation
                if report.overfitting_severity in ["none", "low"]:
                    report.overfitting_severity = "medium"
                report.warning_indicators.append(f"Training loss significantly lower than validation: {avg_loss_gap:.4f}")

        except Exception as e:
            logger.error(f"Performance gap analysis failed: {e}")

    def _analyze_learning_curve(self, report: MonitoringReport, session: Dict[str, Any]):
        """Analyze learning curve trends and health."""
        try:
            metrics_history = list(session['metrics_history'])

            if len(metrics_history) < self.config.learning_curve_min_samples:
                return

            # Extract learning curves
            epochs = [m['epoch'] for m in metrics_history]
            train_accuracies = [m['train_accuracy'] for m in metrics_history]
            val_accuracies = [m['val_accuracy'] for m in metrics_history]
            train_losses = [m['train_loss'] for m in metrics_history]
            val_losses = [m['val_loss'] for m in metrics_history]

            # Calculate learning curve health
            train_trend = self._calculate_trend(train_accuracies)
            val_trend = self._calculate_trend(val_accuracies)
            loss_trend = self._calculate_trend(train_losses, reverse=True)  # Lower is better for loss

            # Assess overall learning curve health
            if train_trend == "improving" and val_trend == "improving":
                report.learning_curve_health = "good"
                report.learning_curve_trend = "improving"
            elif train_trend == "improving" and val_trend == "stable":
                report.learning_curve_health = "concerning"
                report.learning_curve_trend = "diverging"
                report.warning_indicators.append("Training improving but validation stagnant")
            elif train_trend == "improving" and val_trend == "declining":
                report.learning_curve_health = "poor"
                report.learning_curve_trend = "overfitting"
                report.warning_indicators.append("Training improving but validation declining")
                report.overfitting_severity = "high"
            else:
                report.learning_curve_health = "stable"
                report.learning_curve_trend = "stable"

        except Exception as e:
            logger.error(f"Learning curve analysis failed: {e}")

    def _check_early_warnings(self, report: MonitoringReport, session: Dict[str, Any]):
        """Check for early warning indicators."""
        try:
            metrics_history = list(session['metrics_history'])

            if len(metrics_history) < self.config.warning_window:
                return

            recent_metrics = list(metrics_history)[-self.config.warning_window:]

            # Check for rapid divergence
            accuracy_divergence = []
            for i in range(1, len(recent_metrics)):
                prev_gap = abs(recent_metrics[i-1]['train_accuracy'] - recent_metrics[i-1]['val_accuracy'])
                curr_gap = abs(recent_metrics[i]['train_accuracy'] - recent_metrics[i]['val_accuracy'])
                accuracy_divergence.append(curr_gap - prev_gap)

            avg_divergence = np.mean(accuracy_divergence) if accuracy_divergence else 0

            if avg_divergence > self.config.divergence_threshold:
                report.early_warning_triggered = True
                report.divergence_rate = avg_divergence
                report.warning_indicators.append(f"Rapid performance divergence: {avg_divergence:.4f}")

                if report.overfitting_severity in ["none", "low"]:
                    report.overfitting_severity = "medium"

            # Check for loss divergence
            loss_divergence = []
            for i in range(1, len(recent_metrics)):
                prev_loss_gap = recent_metrics[i-1]['val_loss'] - recent_metrics[i-1]['train_loss']
                curr_loss_gap = recent_metrics[i]['val_loss'] - recent_metrics[i]['train_loss']
                loss_divergence.append(curr_loss_gap - prev_loss_gap)

            avg_loss_divergence = np.mean(loss_divergence) if loss_divergence else 0

            if avg_loss_divergence > 0.01:  # Validation loss increasing relative to train
                report.early_warning_triggered = True
                report.warning_indicators.append(f"Validation loss diverging: {avg_loss_divergence:.4f}")

        except Exception as e:
            logger.error(f"Early warning check failed: {e}")

    def _analyze_trends(self, report: MonitoringReport, session: Dict[str, Any]):
        """Analyze performance trends over time."""
        try:
            metrics_history = list(session['metrics_history'])

            if len(metrics_history) < self.config.trend_window:
                return

            recent_metrics = list(metrics_history)[-self.config.trend_window:]

            # Accuracy trend
            accuracies = [m['val_accuracy'] for m in recent_metrics]
            accuracy_trend_value = self._calculate_trend_slope(accuracies)

            if accuracy_trend_value > 0.01:
                report.accuracy_trend = "improving"
            elif accuracy_trend_value < -0.01:
                report.accuracy_trend = "declining"
            else:
                report.accuracy_trend = "stable"

            # Loss trend
            losses = [m['val_loss'] for m in recent_metrics]
            loss_trend_value = self._calculate_trend_slope(losses)

            if loss_trend_value < -0.01:
                report.loss_trend = "decreasing"
            elif loss_trend_value > 0.01:
                report.loss_trend = "increasing"
            else:
                report.loss_trend = "stable"

            # Cross-reference trends for overfitting detection
            if report.accuracy_trend == "declining" and report.loss_trend == "increasing":
                if report.overfitting_severity in ["none", "low"]:
                    report.overfitting_severity = "medium"
                report.warning_indicators.append("Validation accuracy declining while loss increasing")

        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")

    def _detect_anomalies(self, report: MonitoringReport, session: Dict[str, Any]):
        """Detect anomalies in training performance."""
        try:
            metrics_history = list(session['metrics_history'])

            if len(metrics_history) < 10:
                return

            # Calculate anomaly scores based on recent performance
            recent_accuracies = [m['val_accuracy'] for m in metrics_history[-20:]]
            recent_losses = [m['val_loss'] for m in metrics_history[-20:]]

            if len(recent_accuracies) > 5:
                accuracy_mean = np.mean(recent_accuracies)
                accuracy_std = np.std(recent_accuracies)
                loss_mean = np.mean(recent_losses)
                loss_std = np.std(recent_losses)

                # Current values
                current_accuracy = report.val_accuracy
                current_loss = report.val_loss

                # Z-scores
                accuracy_z_score = abs(current_accuracy - accuracy_mean) / (accuracy_std + 1e-8)
                loss_z_score = abs(current_loss - loss_mean) / (loss_std + 1e-8)

                # Anomaly score
                report.anomaly_score = max(accuracy_z_score, loss_z_score)

                if report.anomaly_score > self.config.anomaly_detection_sensitivity:
                    report.anomaly_detected = True
                    report.anomaly_description = f"Anomalous performance: accuracy_z={accuracy_z_score:.2f}, loss_z={loss_z_score:.2f}"

                    if report.overfitting_severity in ["none", "low"]:
                        report.overfitting_severity = "medium"

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")

    def _calculate_trend(self, values: List[float], reverse: bool = False) -> str:
        """Calculate trend direction from list of values."""
        if len(values) < 3:
            return "insufficient_data"

        # Calculate slope
        slope = self._calculate_trend_slope(values)

        # Apply reverse logic if needed (for loss, decreasing is good)
        if reverse:
            slope = -slope

        if slope > 0.001:
            return "improving"
        elif slope < -0.001:
            return "degrading"
        else:
            return "stable"

    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate slope of trend line."""
        try:
            if len(values) < 2:
                return 0.0

            x = np.arange(len(values))
            y = np.array(values)

            # Simple linear regression slope
            slope = np.polyfit(x, y, 1)[0]
            return slope

        except Exception as e:
            logger.error(f"Trend slope calculation failed: {e}")
            return 0.0

    def _generate_monitoring_recommendations(self, report: MonitoringReport):
        """Generate monitoring recommendations based on analysis."""
        try:
            # Immediate actions for critical overfitting
            if report.overfitting_severity == "critical":
                report.immediate_actions.extend([
                    "Stop training immediately",
                    "Reduce model complexity",
                    "Increase regularization",
                    "Check for data leakage"
                ])

            # Training suggestions for high overfitting
            if report.overfitting_severity == "high":
                report.training_suggestions.extend([
                    "Consider early stopping",
                    "Increase dropout rate",
                    "Add more training data",
                    "Use data augmentation"
                ])

            # General recommendations
            if report.learning_curve_health == "poor":
                report.recommendations.append("Model may be overfitting - consider regularization techniques")

            if report.early_warning_triggered:
                report.recommendations.append("Early warning signs detected - monitor closely")

            if report.anomaly_detected:
                report.recommendations.append("Anomalous performance detected - investigate training data")

            # Trend-based recommendations
            if report.accuracy_trend == "declining":
                report.recommendations.append("Validation accuracy is declining - check for overfitting")

            if report.loss_trend == "increasing":
                report.recommendations.append("Validation loss is increasing - consider adjusting learning rate")

            # Positive feedback
            if (report.overfitting_severity in ["none", "low"] and
                report.learning_curve_health == "good"):
                report.recommendations.append("Training appears healthy - continue monitoring")

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")

    def end_monitoring_session(self, session_id: str) -> Dict[str, Any]:
        """
        End a monitoring session and generate final report.

        Args:
            session_id: Session ID to end

        Returns:
            Final monitoring summary
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]
        model_name = session['model_name']

        # Generate final report
        final_report = {
            'session_id': session_id,
            'model_name': model_name,
            'duration': datetime.now() - session['start_time'],
            'total_epochs': len(session['metrics_history']),
            'final_status': 'completed'
        }

        # Save final report
        if self.config.save_monitoring_reports:
            self._save_monitoring_report(final_report, session_id)

        # Clean up
        del self.active_sessions[session_id]
        if session_id in self.performance_history:
            del self.performance_history[session_id]

        logger.info(f"📊 Ended monitoring session {session_id} for {model_name}")
        return final_report

    def _save_monitoring_report(self, report: Dict[str, Any], session_id: str):
        """Save monitoring report to file."""
        try:
            filename = f"monitoring_report_{session_id}.json"
            filepath = Path(self.config.report_directory) / filename

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"Monitoring report saved: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save monitoring report: {e}")

    def get_monitoring_history(self, session_id: Optional[str] = None) -> List[MonitoringReport]:
        """Get monitoring history."""
        if session_id:
            return [r for r in self.monitoring_history if r.monitoring_session_id == session_id]
        return self.monitoring_history.copy()

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.active_sessions.keys())

# Global instance
DEFAULT_OVERFITTING_MONITOR = OverfittingMonitor()

def get_overfitting_monitor(config: Optional[OverfittingMonitoringConfig] = None) -> OverfittingMonitor:
    """Get overfitting monitor instance."""
    if config is None:
        return DEFAULT_OVERFITTING_MONITOR
    return OverfittingMonitor(config)

def start_monitoring_session(model_name: str, model_type: str = "unknown") -> str:
    """Convenience function to start monitoring session."""
    monitor = get_overfitting_monitor()
    return monitor.start_monitoring_session(model_name, model_type)

def monitor_training_step(session_id: str,
                         epoch: int,
                         train_accuracy: float,
                         val_accuracy: float,
                         train_loss: float,
                         val_loss: float,
                         additional_metrics: Optional[Dict[str, float]] = None) -> MonitoringReport:
    """Convenience function to monitor training step."""
    monitor = get_overfitting_monitor()
    return monitor.monitor_training_step(
        session_id, epoch, train_accuracy, val_accuracy, train_loss, val_loss, additional_metrics
    )