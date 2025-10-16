"""
Monitoring and Retrain Decision System for End-to-End Roadmap

Implements:
- Calibration monitoring (MSE/Brier by bucket)
- PSI for feature drift detection
- Correlation drift monitoring
- Latency monitoring (p95/p99 per component)
- Retrain decision tree with graceful degradation
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from collections import deque
import logging
from zoneinfo import ZoneInfo

class MonitoringStatus(Enum):
    """Status of monitoring system."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"

class RetrainTrigger(Enum):
    """Triggers for retraining."""
    SCHEDULED = "scheduled"
    CALIBRATION_LOSS = "calibration_loss"
    PSI_DRIFT = "psi_drift"
    CORRELATION_DRIFT = "correlation_drift"
    LATENCY_BREACH = "latency_breach"
    MANUAL = "manual"

@dataclass
class MonitoringConfig:
    """Configuration for monitoring system."""
    calibration_loss_threshold: float = 2.0  # sigma
    psi_threshold: float = 0.3
    correlation_drift_threshold: float = 0.5
    latency_p99_threshold: float = 50.0  # ms
    missing_data_threshold: float = 0.05  # 5% of bars
    lookback_hours: int = 2
    monitoring_interval_minutes: int = 5
    retrain_check_interval_hours: int = 2
    scheduled_retrain_time: str = "02:00"  # ET
    scheduled_retrain_timezone: str = "America/New_York"
    fallback_model_latency_ms: float = 2.0
    psi_monitor_columns: Tuple[str, ...] = ("σ_EW", "vwap_dist")

@dataclass
class MonitoringMetrics:
    """Current monitoring metrics."""
    timestamp: datetime
    calibration_loss: float
    psi_scores: Dict[str, float]
    correlation_drift: float
    latency_p95: float
    latency_p99: float
    missing_data_pct: float
    status: MonitoringStatus
    alerts: List[str]

@dataclass
class RetrainDecision:
    """Retrain decision result."""
    should_retrain: bool
    trigger: Optional[RetrainTrigger]
    urgency: str  # 'low', 'medium', 'high', 'critical'
    reason: str
    fallback_required: bool
    estimated_duration_minutes: int

class CalibrationMonitor:
    """Monitors model calibration quality."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.calibration_history = deque(maxlen=1000)
        self.bucket_calibrations = {
            'open': deque(maxlen=100),
            'mid': deque(maxlen=100),
            'close': deque(maxlen=100)
        }
        self.consecutive_breaches = 0
        self.last_loss = 0.0

    def update_calibration(self,
                          predictions: np.ndarray,
                          actual: np.ndarray,
                          session_buckets: Optional[np.ndarray] = None) -> float:
        """Update calibration metrics."""

        if len(predictions) == 0 or len(actual) == 0:
            return 0.0

        # Calculate MSE
        mse = np.mean((predictions - actual) ** 2)
        self.calibration_history.append(mse)

        # Calculate bucket-specific calibration if buckets provided
        if session_buckets is not None:
            for bucket in ['open', 'mid', 'close']:
                bucket_mask = session_buckets == bucket
                if np.any(bucket_mask):
                    bucket_mse = np.mean((predictions[bucket_mask] - actual[bucket_mask]) ** 2)
                    self.bucket_calibrations[bucket].append(bucket_mse)

        loss = self._calculate_loss()
        self.last_loss = loss

        if loss > self.config.calibration_loss_threshold:
            self.consecutive_breaches += 1
        else:
            self.consecutive_breaches = 0

        return loss

    def get_calibration_loss(self) -> float:
        """Get current calibration loss (in sigma units)."""
        loss = self._calculate_loss()
        self.last_loss = loss
        return loss

    def _calculate_loss(self) -> float:
        """Calculate calibration loss using history."""
        if len(self.calibration_history) < 10:
            return 0.0

        recent_mse = np.mean(list(self.calibration_history)[-10:])
        historical_mse = np.mean(list(self.calibration_history)[:-10]) if len(self.calibration_history) > 10 else recent_mse

        if historical_mse == 0:
            return 0.0

        # Calculate sigma deviation
        sigma = np.std(list(self.calibration_history))
        if sigma == 0:
            return 0.0

        loss = (recent_mse - historical_mse) / sigma
        return loss

    def has_persistent_breach(self) -> bool:
        """Return True if calibration breach persisted for required duration."""
        return self.consecutive_breaches >= 3

    def get_bucket_calibration(self) -> Dict[str, float]:
        """Get calibration by session bucket."""
        bucket_losses = {}

        for bucket, history in self.bucket_calibrations.items():
            if len(history) < 5:
                bucket_losses[bucket] = 0.0
                continue

            recent_mse = np.mean(list(history)[-5:])
            historical_mse = np.mean(list(history)[:-5]) if len(history) > 5 else recent_mse

            if historical_mse == 0:
                bucket_losses[bucket] = 0.0
            else:
                sigma = np.std(list(history))
                if sigma == 0:
                    bucket_losses[bucket] = 0.0
                else:
                    bucket_losses[bucket] = (recent_mse - historical_mse) / sigma

        return bucket_losses

class PSIMonitor:
    """Monitors Population Stability Index for feature drift."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.reference_distributions = {}
        self.current_distributions = {}
        self.monitored_columns = set(config.psi_monitor_columns or [])

    def update_reference(self, features: pd.DataFrame):
        """Update reference distributions from training data."""
        monitored_columns = self.monitored_columns
        for col in features.columns:
            if monitored_columns and col not in monitored_columns:
                continue
            if features[col].dtype in ['float64', 'int64']:
                # Create histogram bins
                finite_data = features[col].dropna()
                if len(finite_data) > 0:
                    bins = np.histogram_bin_edges(finite_data, bins=10)
                    hist, _ = np.histogram(finite_data, bins=bins)
                    self.reference_distributions[col] = {
                        'hist': hist,
                        'bins': bins,
                        'total': len(finite_data)
                    }

    def calculate_psi(self, features: pd.DataFrame) -> Dict[str, float]:
        """Calculate PSI for all features."""
        psi_scores = {}

        monitored_columns = self.monitored_columns
        for col in features.columns:
            if monitored_columns and col not in monitored_columns:
                continue
            if col not in self.reference_distributions:
                continue

            finite_data = features[col].dropna()
            if len(finite_data) == 0:
                psi_scores[col] = 0.0
                continue

            ref_dist = self.reference_distributions[col]

            # Calculate current distribution
            hist, _ = np.histogram(finite_data, bins=ref_dist['bins'])

            # Calculate PSI
            psi = self._calculate_psi_score(
                ref_dist['hist'], hist,
                ref_dist['total'], len(finite_data)
            )

            psi_scores[col] = psi

        return psi_scores

    def _calculate_psi_score(self,
                            ref_hist: np.ndarray,
                            curr_hist: np.ndarray,
                            ref_total: int,
                            curr_total: int) -> float:
        """Calculate PSI score between two distributions."""

        # Normalize histograms
        ref_pct = ref_hist / ref_total
        curr_pct = curr_hist / curr_total

        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        ref_pct = np.maximum(ref_pct, epsilon)
        curr_pct = np.maximum(curr_pct, epsilon)

        # Calculate PSI
        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))

        return psi

class CorrelationDriftMonitor:
    """Monitors correlation drift in feature matrix."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.reference_correlation = None
        self.correlation_history = deque(maxlen=100)

    def update_reference(self, features: pd.DataFrame):
        """Update reference correlation matrix."""
        # Calculate correlation matrix for top 20 features by variance
        feature_vars = features.var()
        top_features = feature_vars.nlargest(20).index
        top_features_data = features[top_features].dropna()

        if len(top_features_data) > 0:
            self.reference_correlation = top_features_data.corr()

    def calculate_drift(self, features: pd.DataFrame) -> float:
        """Calculate correlation drift."""
        if self.reference_correlation is None:
            return 0.0

        # Get same features as reference
        ref_features = self.reference_correlation.columns
        available_features = [f for f in ref_features if f in features.columns]

        if len(available_features) < 5:
            return 0.0

        # Calculate current correlation matrix
        current_data = features[available_features].dropna()
        if len(current_data) < 10:
            return 0.0

        current_correlation = current_data.corr()

        # Calculate Frobenius norm difference
        ref_subset = self.reference_correlation.loc[available_features, available_features]
        diff = current_correlation - ref_subset
        frobenius_norm = np.sqrt(np.sum(diff ** 2))

        self.correlation_history.append(frobenius_norm)
        return frobenius_norm

class LatencyMonitor:
    """Monitors latency across components."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.latency_history = {
            'feature_compute': deque(maxlen=1000),
            'model_inference': deque(maxlen=1000),
            'io_orchestration': deque(maxlen=1000),
            'total': deque(maxlen=1000)
        }

    def record_latency(self,
                      component: str,
                      latency_ms: float):
        """Record latency for a component."""
        if component in self.latency_history:
            self.latency_history[component].append(latency_ms)

    def get_latency_stats(self) -> Dict[str, Dict[str, float]]:
        """Get latency statistics."""
        stats = {}

        for component, history in self.latency_history.items():
            if len(history) == 0:
                stats[component] = {'p95': 0.0, 'p99': 0.0, 'mean': 0.0}
                continue

            history_array = np.array(history)
            stats[component] = {
                'p95': np.percentile(history_array, 95),
                'p99': np.percentile(history_array, 99),
                'mean': np.mean(history_array)
            }

        return stats

    def is_latency_breach(self) -> bool:
        """Check if latency budget is breached."""
        stats = self.get_latency_stats()
        total_p99 = stats.get('total', {}).get('p99', 0.0)
        return total_p99 > self.config.latency_p99_threshold

class RetrainDecisionTree:
    """Decision tree for retrain triggers."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.calibration_monitor = CalibrationMonitor(config)
        self.psi_monitor = PSIMonitor(config)
        self.correlation_monitor = CorrelationDriftMonitor(config)
        self.latency_monitor = LatencyMonitor(config)
        self.last_retrain = None
        self.retrain_count = 0

    def should_retrain(self,
                      current_metrics: MonitoringMetrics,
                      predictions: Optional[np.ndarray] = None,
                      actual: Optional[np.ndarray] = None,
                      features: Optional[pd.DataFrame] = None) -> RetrainDecision:
        """Determine if retraining is needed."""

        triggers = []
        urgency_levels = []

        # Check calibration loss
        if (
            current_metrics.calibration_loss > self.config.calibration_loss_threshold
            and self.calibration_monitor.has_persistent_breach()
        ):
            triggers.append(RetrainTrigger.CALIBRATION_LOSS)
            urgency_levels.append('high')

        # Check PSI drift
        max_psi = max(current_metrics.psi_scores.values()) if current_metrics.psi_scores else 0.0
        if max_psi > self.config.psi_threshold:
            triggers.append(RetrainTrigger.PSI_DRIFT)
            urgency_levels.append('medium')

        # Check correlation drift
        if current_metrics.correlation_drift > self.config.correlation_drift_threshold:
            triggers.append(RetrainTrigger.CORRELATION_DRIFT)
            urgency_levels.append('medium')

        # Check latency breach
        if current_metrics.latency_p99 > self.config.latency_p99_threshold:
            triggers.append(RetrainTrigger.LATENCY_BREACH)
            urgency_levels.append('critical')

        # Check scheduled retrain
        if self._is_scheduled_retrain_time():
            triggers.append(RetrainTrigger.SCHEDULED)
            urgency_levels.append('low')

        # Determine overall decision
        if not triggers:
            return RetrainDecision(
                should_retrain=False,
                trigger=None,
                urgency='low',
                reason='All metrics within acceptable ranges',
                fallback_required=False,
                estimated_duration_minutes=0
            )

        # Determine urgency
        if 'critical' in urgency_levels:
            urgency = 'critical'
        elif 'high' in urgency_levels:
            urgency = 'high'
        elif 'medium' in urgency_levels:
            urgency = 'medium'
        else:
            urgency = 'low'

        # Determine if fallback is required
        fallback_required = (
            current_metrics.latency_p99 > self.config.latency_p99_threshold or
            current_metrics.status == MonitoringStatus.CRITICAL
        )

        # Estimate duration
        duration = self._estimate_retrain_duration(urgency)

        return RetrainDecision(
            should_retrain=True,
            trigger=triggers[0],  # Primary trigger
            urgency=urgency,
            reason=f"Triggered by: {', '.join([t.value for t in triggers])}",
            fallback_required=fallback_required,
            estimated_duration_minutes=duration
        )

    def _is_scheduled_retrain_time(self, current_time: Optional[datetime] = None) -> bool:
        """Check if it's time for scheduled retrain based on configured clock."""
        scheduled_time_str = self.config.scheduled_retrain_time
        if not scheduled_time_str:
            return False

        try:
            hour, minute = [int(part) for part in scheduled_time_str.split(":")]
        except ValueError:
            warnings.warn("Invalid scheduled_retrain_time format. Expected HH:MM.")
            return False

        try:
            tz = ZoneInfo(self.config.scheduled_retrain_timezone)
        except Exception:
            warnings.warn("Invalid timezone for scheduled retrain; defaulting to UTC.")
            tz = ZoneInfo("UTC")

        if current_time is None:
            now = datetime.now(tz)
        else:
            now = current_time.astimezone(tz) if current_time.tzinfo else current_time.replace(tzinfo=tz)

        scheduled_today = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if now < scheduled_today:
            return False

        if self.last_retrain is None:
            return True

        last_retrain = self.last_retrain
        if last_retrain.tzinfo is None:
            last_retrain = last_retrain.replace(tzinfo=tz)
        else:
            last_retrain = last_retrain.astimezone(tz)

        if last_retrain >= scheduled_today:
            return False

        if last_retrain.date() == now.date():
            return False

        return True

    def _estimate_retrain_duration(self, urgency: str) -> int:
        """Estimate retrain duration in minutes."""
        base_duration = 30  # Base duration in minutes

        if urgency == 'critical':
            return base_duration
        elif urgency == 'high':
            return base_duration * 2
        elif urgency == 'medium':
            return base_duration * 3
        else:
            return base_duration * 4

class MonitoringSystem:
    """Complete monitoring system."""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.decision_tree = RetrainDecisionTree(config)
        self.logger = logging.getLogger(__name__)
        self.metrics_history = deque(maxlen=1000)

    def update_metrics(self,
                      features: pd.DataFrame,
                      predictions: Optional[np.ndarray] = None,
                      actual: Optional[np.ndarray] = None,
                      session_buckets: Optional[np.ndarray] = None) -> MonitoringMetrics:
        """Update all monitoring metrics."""

        # Update calibration
        calibration_loss = 0.0
        if predictions is not None and actual is not None:
            calibration_loss = self.decision_tree.calibration_monitor.update_calibration(
                predictions, actual, session_buckets
            )
        else:
            calibration_loss = self.decision_tree.calibration_monitor.get_calibration_loss()

        # Update PSI
        psi_scores = self.decision_tree.psi_monitor.calculate_psi(features)

        # Update correlation drift
        correlation_drift = self.decision_tree.correlation_monitor.calculate_drift(features)

        # Get latency stats
        latency_stats = self.decision_tree.latency_monitor.get_latency_stats()
        latency_p95 = latency_stats.get('total', {}).get('p95', 0.0)
        latency_p99 = latency_stats.get('total', {}).get('p99', 0.0)

        # Calculate missing data percentage
        missing_data_pct = features.isnull().sum().sum() / (len(features) * len(features.columns))

        # Determine status
        status = self._determine_status(
            calibration_loss, max(psi_scores.values()) if psi_scores else 0.0,
            correlation_drift, latency_p99, missing_data_pct
        )

        # Create metrics object
        metrics = MonitoringMetrics(
            timestamp=datetime.now(),
            calibration_loss=calibration_loss,
            psi_scores=psi_scores,
            correlation_drift=correlation_drift,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            missing_data_pct=missing_data_pct,
            status=status,
            alerts=[]
        )

        # Add alerts
        if (
            calibration_loss > self.config.calibration_loss_threshold
            and self.decision_tree.calibration_monitor.has_persistent_breach()
        ):
            metrics.alerts.append(f"High calibration loss: {calibration_loss:.2f}σ")

        if max(psi_scores.values()) if psi_scores else 0.0 > self.config.psi_threshold:
            metrics.alerts.append("PSI drift detected")

        if correlation_drift > self.config.correlation_drift_threshold:
            metrics.alerts.append(f"Correlation drift: {correlation_drift:.2f}")

        if latency_p99 > self.config.latency_p99_threshold:
            metrics.alerts.append(f"Latency breach: {latency_p99:.1f}ms")

        if missing_data_pct > self.config.missing_data_threshold:
            metrics.alerts.append(f"High missing data: {missing_data_pct:.1%}")

        self.metrics_history.append(metrics)
        return metrics

    def _determine_status(self,
                         calibration_loss: float,
                         max_psi: float,
                         correlation_drift: float,
                         latency_p99: float,
                         missing_data_pct: float) -> MonitoringStatus:
        """Determine overall system status."""

        critical_conditions = [
            latency_p99 > self.config.latency_p99_threshold,
            missing_data_pct > self.config.missing_data_threshold * 2
        ]

        warning_conditions = [
            calibration_loss > self.config.calibration_loss_threshold,
            max_psi > self.config.psi_threshold,
            correlation_drift > self.config.correlation_drift_threshold
        ]

        if any(critical_conditions):
            return MonitoringStatus.CRITICAL
        elif any(warning_conditions):
            return MonitoringStatus.WARNING
        else:
            return MonitoringStatus.HEALTHY

    def get_retrain_decision(self, metrics: MonitoringMetrics) -> RetrainDecision:
        """Get retrain decision based on current metrics."""
        return self.decision_tree.should_retrain(metrics)

    def record_latency(self, component: str, latency_ms: float):
        """Record latency for a component."""
        self.decision_tree.latency_monitor.record_latency(component, latency_ms)

    def update_reference_data(self, features: pd.DataFrame):
        """Update reference distributions for drift detection."""
        self.decision_tree.psi_monitor.update_reference(features)
        self.decision_tree.correlation_monitor.update_reference(features)
