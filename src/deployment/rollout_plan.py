"""
Rollout Plan for End-to-End Roadmap System

Implements:
- Shadow mode (1-2 sessions with full logging, no trades)
- Canary deployment (10-20% risk for one session)
- Full deployment with retrain triggers and automatic fallback
- Risk management and rollback procedures
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

class DeploymentPhase(Enum):
    """Deployment phases."""
    SHADOW = "shadow"
    CANARY = "canary"
    FULL = "full"
    ROLLBACK = "rollback"

class RiskLevel(Enum):
    """Risk levels for deployment."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RolloutConfig:
    """Configuration for rollout plan."""
    shadow_duration_sessions: int = 2
    canary_risk_percentage: float = 0.15  # 15%
    canary_duration_sessions: int = 1
    full_deployment_trigger_conditions: List[str] = None
    rollback_triggers: List[str] = None
    monitoring_interval_minutes: int = 5
    risk_thresholds: Dict[str, float] = None

    def __post_init__(self):
        if self.full_deployment_trigger_conditions is None:
            self.full_deployment_trigger_conditions = [
                'calibration_within_1sigma',
                'latency_within_slo',
                'no_critical_alerts',
                'canary_success_rate > 0.95'
            ]

        if self.rollback_triggers is None:
            self.rollback_triggers = [
                'calibration_loss > 3sigma',
                'latency_p99 > 100ms',
                'critical_system_error',
                'data_quality_breach'
            ]

        if self.risk_thresholds is None:
            self.risk_thresholds = {
                'calibration_loss': 2.0,
                'latency_p99': 50.0,
                'error_rate': 0.05,
                'data_quality': 0.95
            }

@dataclass
class DeploymentStatus:
    """Current deployment status."""
    phase: DeploymentPhase
    start_time: datetime
    duration: timedelta
    risk_level: RiskLevel
    metrics: Dict[str, Any]
    alerts: List[str]
    canary_percentage: float = 0.0
    success_rate: float = 0.0

class ShadowModeManager:
    """Manages shadow mode deployment."""

    def __init__(self, config: RolloutConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_history = []
        self.start_time = None

    def start_shadow_mode(self) -> DeploymentStatus:
        """Start shadow mode deployment."""
        self.start_time = datetime.now()
        self.logger.info("Starting shadow mode deployment")

        return DeploymentStatus(
            phase=DeploymentPhase.SHADOW,
            start_time=self.start_time,
            duration=timedelta(0),
            risk_level=RiskLevel.LOW,
            metrics={},
            alerts=[]
        )

    def monitor_shadow_mode(self,
                           features: pd.DataFrame,
                           predictions: np.ndarray,
                           actual: np.ndarray) -> DeploymentStatus:
        """Monitor shadow mode performance."""

        if self.start_time is None:
            raise ValueError("Shadow mode not started")

        duration = datetime.now() - self.start_time

        # Calculate metrics
        metrics = self._calculate_shadow_metrics(features, predictions, actual)
        self.metrics_history.append(metrics)

        # Check for alerts
        alerts = self._check_shadow_alerts(metrics)

        # Determine risk level
        risk_level = self._assess_risk_level(metrics, alerts)

        return DeploymentStatus(
            phase=DeploymentPhase.SHADOW,
            start_time=self.start_time,
            duration=duration,
            risk_level=risk_level,
            metrics=metrics,
            alerts=alerts
        )

    def should_exit_shadow_mode(self, status: DeploymentStatus) -> bool:
        """Determine if shadow mode should exit."""

        # Check duration
        if status.duration >= timedelta(days=self.config.shadow_duration_sessions):
            return True

        # Check for critical issues
        if status.risk_level == RiskLevel.CRITICAL:
            return True

        # Check for sustained good performance
        if len(self.metrics_history) >= 10:
            recent_metrics = self.metrics_history[-10:]
            avg_calibration = np.mean([m.get('calibration_loss', 0) for m in recent_metrics])
            avg_latency = np.mean([m.get('latency_p99', 0) for m in recent_metrics])

            if avg_calibration < 1.0 and avg_latency < 30:
                return True

        return False

    def _calculate_shadow_metrics(self,
                                 features: pd.DataFrame,
                                 predictions: np.ndarray,
                                 actual: np.ndarray) -> Dict[str, Any]:
        """Calculate shadow mode metrics."""

        metrics = {}

        # Calibration metrics
        if len(predictions) > 0 and len(actual) > 0:
            mse = np.mean((predictions - actual) ** 2)
            metrics['calibration_loss'] = mse
            metrics['correlation'] = np.corrcoef(predictions, actual)[0, 1] if len(predictions) > 1 else 0.0

        # Latency metrics (simplified)
        metrics['latency_p95'] = 25.0  # Placeholder
        metrics['latency_p99'] = 45.0  # Placeholder

        # Data quality metrics
        metrics['data_quality'] = 1.0 - features.isnull().sum().sum() / (len(features) * len(features.columns))

        # Feature metrics
        metrics['feature_count'] = len(features.columns)
        metrics['sample_count'] = len(features)

        return metrics

    def _check_shadow_alerts(self, metrics: Dict[str, Any]) -> List[str]:
        """Check for shadow mode alerts."""
        alerts = []

        if metrics.get('calibration_loss', 0) > self.config.risk_thresholds['calibration_loss']:
            alerts.append(f"High calibration loss: {metrics['calibration_loss']:.2f}")

        if metrics.get('latency_p99', 0) > self.config.risk_thresholds['latency_p99']:
            alerts.append(f"High latency: {metrics['latency_p99']:.1f}ms")

        if metrics.get('data_quality', 1.0) < self.config.risk_thresholds['data_quality']:
            alerts.append(f"Low data quality: {metrics['data_quality']:.2f}")

        return alerts

    def _assess_risk_level(self, metrics: Dict[str, Any], alerts: List[str]) -> RiskLevel:
        """Assess risk level based on metrics and alerts."""

        if len(alerts) > 3:
            return RiskLevel.CRITICAL
        elif len(alerts) > 1:
            return RiskLevel.HIGH
        elif len(alerts) > 0:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

class CanaryManager:
    """Manages canary deployment."""

    def __init__(self, config: RolloutConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_history = []
        self.start_time = None
        self.canary_percentage = config.canary_risk_percentage

    def start_canary_mode(self) -> DeploymentStatus:
        """Start canary mode deployment."""
        self.start_time = datetime.now()
        self.logger.info(f"Starting canary mode deployment with {self.canary_percentage:.1%} risk")

        return DeploymentStatus(
            phase=DeploymentPhase.CANARY,
            start_time=self.start_time,
            duration=timedelta(0),
            risk_level=RiskLevel.MEDIUM,
            metrics={},
            alerts=[],
            canary_percentage=self.canary_percentage
        )

    def monitor_canary_mode(self,
                           features: pd.DataFrame,
                           predictions: np.ndarray,
                           actual: np.ndarray,
                           trade_results: Optional[Dict[str, Any]] = None) -> DeploymentStatus:
        """Monitor canary mode performance."""

        if self.start_time is None:
            raise ValueError("Canary mode not started")

        duration = datetime.now() - self.start_time

        # Calculate metrics
        metrics = self._calculate_canary_metrics(features, predictions, actual, trade_results)
        self.metrics_history.append(metrics)

        # Check for alerts
        alerts = self._check_canary_alerts(metrics)

        # Determine risk level
        risk_level = self._assess_risk_level(metrics, alerts)

        # Calculate success rate
        success_rate = self._calculate_success_rate(metrics)

        return DeploymentStatus(
            phase=DeploymentPhase.CANARY,
            start_time=self.start_time,
            duration=duration,
            risk_level=risk_level,
            metrics=metrics,
            alerts=alerts,
            canary_percentage=self.canary_percentage,
            success_rate=success_rate
        )

    def should_exit_canary_mode(self, status: DeploymentStatus) -> Tuple[bool, bool]:
        """Determine if canary mode should exit and whether to proceed to full deployment.

        Returns:
            (should_exit, should_proceed_to_full)
        """

        # Check duration
        if status.duration >= timedelta(days=self.config.canary_duration_sessions):
            return True, status.success_rate > 0.95

        # Check for critical issues
        if status.risk_level == RiskLevel.CRITICAL:
            return True, False

        # Check for sustained good performance
        if len(self.metrics_history) >= 5:
            recent_metrics = self.metrics_history[-5:]
            avg_success_rate = np.mean([m.get('success_rate', 0) for m in recent_metrics])

            if avg_success_rate > 0.95:
                return True, True

        return False, False

    def _calculate_canary_metrics(self,
                                 features: pd.DataFrame,
                                 predictions: np.ndarray,
                                 actual: np.ndarray,
                                 trade_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate canary mode metrics."""

        metrics = {}

        # Calibration metrics
        if len(predictions) > 0 and len(actual) > 0:
            mse = np.mean((predictions - actual) ** 2)
            metrics['calibration_loss'] = mse
            metrics['correlation'] = np.corrcoef(predictions, actual)[0, 1] if len(predictions) > 1 else 0.0

        # Latency metrics
        metrics['latency_p95'] = 30.0  # Placeholder
        metrics['latency_p99'] = 50.0  # Placeholder

        # Data quality metrics
        metrics['data_quality'] = 1.0 - features.isnull().sum().sum() / (len(features) * len(features.columns))

        # Trading metrics (if available)
        if trade_results:
            metrics['success_rate'] = trade_results.get('success_rate', 0.0)
            metrics['profit_loss'] = trade_results.get('profit_loss', 0.0)
            metrics['trade_count'] = trade_results.get('trade_count', 0)
        else:
            metrics['success_rate'] = 0.95  # Placeholder
            metrics['profit_loss'] = 0.0
            metrics['trade_count'] = 0

        # Feature metrics
        metrics['feature_count'] = len(features.columns)
        metrics['sample_count'] = len(features)

        return metrics

    def _check_canary_alerts(self, metrics: Dict[str, Any]) -> List[str]:
        """Check for canary mode alerts."""
        alerts = []

        if metrics.get('calibration_loss', 0) > self.config.risk_thresholds['calibration_loss']:
            alerts.append(f"High calibration loss: {metrics['calibration_loss']:.2f}")

        if metrics.get('latency_p99', 0) > self.config.risk_thresholds['latency_p99']:
            alerts.append(f"High latency: {metrics['latency_p99']:.1f}ms")

        if metrics.get('data_quality', 1.0) < self.config.risk_thresholds['data_quality']:
            alerts.append(f"Low data quality: {metrics['data_quality']:.2f}")

        if metrics.get('success_rate', 1.0) < 0.90:
            alerts.append(f"Low success rate: {metrics['success_rate']:.2f}")

        return alerts

    def _assess_risk_level(self, metrics: Dict[str, Any], alerts: List[str]) -> RiskLevel:
        """Assess risk level based on metrics and alerts."""

        if len(alerts) > 2:
            return RiskLevel.CRITICAL
        elif len(alerts) > 1:
            return RiskLevel.HIGH
        elif len(alerts) > 0:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _calculate_success_rate(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall success rate."""
        return metrics.get('success_rate', 0.0)

class FullDeploymentManager:
    """Manages full deployment."""

    def __init__(self, config: RolloutConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics_history = []
        self.start_time = None
        self.retrain_triggers_enabled = True
        self.automatic_fallback_enabled = True

    def start_full_deployment(self) -> DeploymentStatus:
        """Start full deployment."""
        self.start_time = datetime.now()
        self.logger.info("Starting full deployment")

        return DeploymentStatus(
            phase=DeploymentPhase.FULL,
            start_time=self.start_time,
            duration=timedelta(0),
            risk_level=RiskLevel.LOW,
            metrics={},
            alerts=[]
        )

    def monitor_full_deployment(self,
                               features: pd.DataFrame,
                               predictions: np.ndarray,
                               actual: np.ndarray,
                               trade_results: Optional[Dict[str, Any]] = None) -> DeploymentStatus:
        """Monitor full deployment performance."""

        if self.start_time is None:
            raise ValueError("Full deployment not started")

        duration = datetime.now() - self.start_time

        # Calculate metrics
        metrics = self._calculate_full_metrics(features, predictions, actual, trade_results)
        self.metrics_history.append(metrics)

        # Check for alerts
        alerts = self._check_full_alerts(metrics)

        # Determine risk level
        risk_level = self._assess_risk_level(metrics, alerts)

        # Check for rollback triggers
        if self._should_rollback(metrics, alerts):
            return DeploymentStatus(
                phase=DeploymentPhase.ROLLBACK,
                start_time=self.start_time,
                duration=duration,
                risk_level=RiskLevel.CRITICAL,
                metrics=metrics,
                alerts=alerts + ["Rollback triggered"]
            )

        return DeploymentStatus(
            phase=DeploymentPhase.FULL,
            start_time=self.start_time,
            duration=duration,
            risk_level=risk_level,
            metrics=metrics,
            alerts=alerts
        )

    def _calculate_full_metrics(self,
                               features: pd.DataFrame,
                               predictions: np.ndarray,
                               actual: np.ndarray,
                               trade_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate full deployment metrics."""

        metrics = {}

        # Calibration metrics
        if len(predictions) > 0 and len(actual) > 0:
            mse = np.mean((predictions - actual) ** 2)
            metrics['calibration_loss'] = mse
            metrics['correlation'] = np.corrcoef(predictions, actual)[0, 1] if len(predictions) > 1 else 0.0

        # Latency metrics
        metrics['latency_p95'] = 35.0  # Placeholder
        metrics['latency_p99'] = 55.0  # Placeholder

        # Data quality metrics
        metrics['data_quality'] = 1.0 - features.isnull().sum().sum() / (len(features) * len(features.columns))

        # Trading metrics (if available)
        if trade_results:
            metrics['success_rate'] = trade_results.get('success_rate', 0.0)
            metrics['profit_loss'] = trade_results.get('profit_loss', 0.0)
            metrics['trade_count'] = trade_results.get('trade_count', 0)
        else:
            metrics['success_rate'] = 0.90  # Placeholder
            metrics['profit_loss'] = 0.0
            metrics['trade_count'] = 0

        # Feature metrics
        metrics['feature_count'] = len(features.columns)
        metrics['sample_count'] = len(features)

        # System health metrics
        metrics['uptime_hours'] = (datetime.now() - self.start_time).total_seconds() / 3600
        metrics['retrain_triggers_enabled'] = self.retrain_triggers_enabled
        metrics['automatic_fallback_enabled'] = self.automatic_fallback_enabled

        return metrics

    def _check_full_alerts(self, metrics: Dict[str, Any]) -> List[str]:
        """Check for full deployment alerts."""
        alerts = []

        if metrics.get('calibration_loss', 0) > self.config.risk_thresholds['calibration_loss']:
            alerts.append(f"High calibration loss: {metrics['calibration_loss']:.2f}")

        if metrics.get('latency_p99', 0) > self.config.risk_thresholds['latency_p99']:
            alerts.append(f"High latency: {metrics['latency_p99']:.1f}ms")

        if metrics.get('data_quality', 1.0) < self.config.risk_thresholds['data_quality']:
            alerts.append(f"Low data quality: {metrics['data_quality']:.2f}")

        if metrics.get('success_rate', 1.0) < 0.85:
            alerts.append(f"Low success rate: {metrics['success_rate']:.2f}")

        return alerts

    def _assess_risk_level(self, metrics: Dict[str, Any], alerts: List[str]) -> RiskLevel:
        """Assess risk level based on metrics and alerts."""

        if len(alerts) > 3:
            return RiskLevel.CRITICAL
        elif len(alerts) > 2:
            return RiskLevel.HIGH
        elif len(alerts) > 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _should_rollback(self, metrics: Dict[str, Any], alerts: List[str]) -> bool:
        """Check if rollback should be triggered."""

        for trigger in self.config.rollback_triggers:
            if trigger == 'calibration_loss > 3sigma' and metrics.get('calibration_loss', 0) > 3.0:
                return True
            elif trigger == 'latency_p99 > 100ms' and metrics.get('latency_p99', 0) > 100.0:
                return True
            elif trigger == 'critical_system_error' and 'critical' in [a.lower() for a in alerts]:
                return True
            elif trigger == 'data_quality_breach' and metrics.get('data_quality', 1.0) < 0.90:
                return True

        return False

class RolloutOrchestrator:
    """Orchestrates the complete rollout process."""

    def __init__(self, config: RolloutConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.shadow_manager = ShadowModeManager(config)
        self.canary_manager = CanaryManager(config)
        self.full_manager = FullDeploymentManager(config)
        self.current_phase = None
        self.rollout_log = []

    def execute_rollout(self,
                       features: pd.DataFrame,
                       predictions: np.ndarray,
                       actual: np.ndarray,
                       trade_results: Optional[Dict[str, Any]] = None) -> DeploymentStatus:
        """Execute the complete rollout process."""

        # Phase 1: Shadow Mode
        if self.current_phase is None or self.current_phase == DeploymentPhase.SHADOW:
            if self.current_phase is None:
                status = self.shadow_manager.start_shadow_mode()
                self.current_phase = DeploymentPhase.SHADOW
            else:
                status = self.shadow_manager.monitor_shadow_mode(features, predictions, actual)

            self.rollout_log.append({
                'timestamp': datetime.now(),
                'phase': 'shadow',
                'status': status
            })

            if self.shadow_manager.should_exit_shadow_mode(status):
                self.logger.info("Exiting shadow mode, proceeding to canary")
                self.current_phase = DeploymentPhase.CANARY

        # Phase 2: Canary Mode
        elif self.current_phase == DeploymentPhase.CANARY:
            if not hasattr(self.canary_manager, 'start_time') or self.canary_manager.start_time is None:
                status = self.canary_manager.start_canary_mode()
            else:
                status = self.canary_manager.monitor_canary_mode(features, predictions, actual, trade_results)

            self.rollout_log.append({
                'timestamp': datetime.now(),
                'phase': 'canary',
                'status': status
            })

            should_exit, should_proceed = self.canary_manager.should_exit_canary_mode(status)
            if should_exit:
                if should_proceed:
                    self.logger.info("Exiting canary mode, proceeding to full deployment")
                    self.current_phase = DeploymentPhase.FULL
                else:
                    self.logger.warning("Exiting canary mode, rolling back due to poor performance")
                    self.current_phase = DeploymentPhase.ROLLBACK

        # Phase 3: Full Deployment
        elif self.current_phase == DeploymentPhase.FULL:
            if not hasattr(self.full_manager, 'start_time') or self.full_manager.start_time is None:
                status = self.full_manager.start_full_deployment()
            else:
                status = self.full_manager.monitor_full_deployment(features, predictions, actual, trade_results)

            self.rollout_log.append({
                'timestamp': datetime.now(),
                'phase': 'full',
                'status': status
            })

            if status.phase == DeploymentPhase.ROLLBACK:
                self.current_phase = DeploymentPhase.ROLLBACK

        # Phase 4: Rollback
        elif self.current_phase == DeploymentPhase.ROLLBACK:
            status = DeploymentStatus(
                phase=DeploymentPhase.ROLLBACK,
                start_time=datetime.now(),
                duration=timedelta(0),
                risk_level=RiskLevel.CRITICAL,
                metrics={},
                alerts=["System rolled back due to critical issues"]
            )

            self.rollout_log.append({
                'timestamp': datetime.now(),
                'phase': 'rollback',
                'status': status
            })

        return status

    def get_rollout_summary(self) -> Dict[str, Any]:
        """Get summary of rollout process."""

        if not self.rollout_log:
            return {'status': 'not_started'}

        phases = [entry['phase'] for entry in self.rollout_log]
        current_phase = phases[-1] if phases else 'unknown'

        # Calculate duration for each phase
        phase_durations = {}
        phase_starts = {}

        for entry in self.rollout_log:
            phase = entry['phase']
            if phase not in phase_starts:
                phase_starts[phase] = entry['timestamp']
            else:
                duration = entry['timestamp'] - phase_starts[phase]
                phase_durations[phase] = duration.total_seconds() / 3600  # hours

        return {
            'current_phase': current_phase,
            'phase_durations_hours': phase_durations,
            'total_entries': len(self.rollout_log),
            'last_update': self.rollout_log[-1]['timestamp'].isoformat() if self.rollout_log else None
        }

    def save_rollout_log(self, filepath: str):
        """Save rollout log to file."""
        log_data = {
            'config': self.config.__dict__,
            'rollout_log': [
                {
                    'timestamp': entry['timestamp'].isoformat(),
                    'phase': entry['phase'],
                    'status': {
                        'phase': entry['status'].phase.value,
                        'risk_level': entry['status'].risk_level.value,
                        'alerts': entry['status'].alerts,
                        'metrics': entry['status'].metrics
                    }
                }
                for entry in self.rollout_log
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)

def create_rollout_orchestrator(config: Optional[RolloutConfig] = None) -> RolloutOrchestrator:
    """Create rollout orchestrator with configuration."""
    if config is None:
        config = RolloutConfig()

    return RolloutOrchestrator(config)

def run_rollout(features: pd.DataFrame,
               predictions: np.ndarray,
               actual: np.ndarray,
               trade_results: Optional[Dict[str, Any]] = None,
               config: Optional[RolloutConfig] = None) -> DeploymentStatus:
    """Run rollout process."""

    orchestrator = create_rollout_orchestrator(config)
    return orchestrator.execute_rollout(features, predictions, actual, trade_results)
