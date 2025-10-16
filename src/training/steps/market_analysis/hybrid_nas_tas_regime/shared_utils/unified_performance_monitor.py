"""
Unified Performance Monitoring System

This module provides a comprehensive performance monitoring system that consolidates
performance tracking for both TAS and NAS architectures, enabling real-time monitoring,
adaptive optimization, and regime-specific performance analysis.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import threading
from pathlib import Path
import json
import pickle

from .unified_architecture_config import ArchitectureType, OptimizationObjective
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)

class PerformanceMetric(Enum):
    """Types of performance metrics to monitor."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFITABILITY = "profitability"
    ROBUSTNESS = "robustness"
    EFFICIENCY = "efficiency"
    ECONOMIC_SIGNIFICANCE = "economic_significance"
    TRADING_VIABILITY = "trading_viability"
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"
    REGIME_STABILITY = "regime_stability"
    ADAPTATION_SPEED = "adaptation_speed"

class MonitoringLevel(Enum):
    """Monitoring levels for different use cases."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    REAL_TIME = "real_time"

@dataclass
class PerformanceSnapshot:
    """Snapshot of performance at a specific time."""
    timestamp: datetime
    architecture_type: ArchitectureType
    iteration: int
    metrics: Dict[PerformanceMetric, float]
    regime_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceTrend:
    """Performance trend analysis."""
    metric: PerformanceMetric
    trend_direction: str  # "improving", "stable", "declining"
    trend_strength: float  # 0.0 to 1.0
    volatility: float
    recent_avg: float
    historical_avg: float
    confidence: float

@dataclass
class RegimePerformanceProfile:
    """Performance profile for a specific regime."""
    regime_id: str
    regime_type: str
    sample_count: int
    avg_performance: Dict[PerformanceMetric, float]
    performance_std: Dict[PerformanceMetric, float]
    stability_score: float
    adaptation_time: float
    last_updated: datetime

class UnifiedPerformanceMonitor:
    """Unified performance monitoring system for TAS and NAS architectures."""

    def __init__(self,
                 architecture_type: ArchitectureType,
                 monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD,
                 max_history_size: int = 10000,
                 monitoring_interval: float = 1.0,
                 enable_real_time: bool = True):
        """Initialize the unified performance monitor.

        Args:
            architecture_type: Type of architecture being monitored
            monitoring_level: Level of monitoring detail
            max_history_size: Maximum number of performance snapshots to keep
            monitoring_interval: Interval between monitoring checks (seconds)
            enable_real_time: Whether to enable real-time monitoring
        """
        tprint_info("🚀 Initializing Unified Performance Monitor")
        tprint_debug(f"Architecture type: {architecture_type}")
        tprint_debug(f"Monitoring level: {monitoring_level}")
        tprint_debug(f"Max history size: {max_history_size}")
        tprint_debug(f"Monitoring interval: {monitoring_interval}")
        tprint_debug(f"Real-time monitoring: {enable_real_time}")

        self.architecture_type = architecture_type
        self.monitoring_level = monitoring_level
        self.max_history_size = max_history_size
        self.monitoring_interval = monitoring_interval
        self.enable_real_time = enable_real_time

        self.logger = logging.getLogger(self.__class__.__name__)

        # Performance history
        tprint_debug("📊 Initializing performance history...")
        self.performance_history: deque = deque(maxlen=max_history_size)
        self.regime_performance: Dict[str, RegimePerformanceProfile] = {}
        tprint_success("✅ Performance history initialized")

        # Real-time monitoring
        tprint_debug("⏱️ Initializing real-time monitoring...")
        self.current_iteration = 0
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        tprint_success("✅ Real-time monitoring initialized")

        # Performance tracking
        tprint_debug("📈 Initializing performance tracking...")
        self.start_time = time.time()
        self.last_check_time = time.time()
        self.performance_trends: Dict[PerformanceMetric, PerformanceTrend] = {}
        tprint_success("✅ Performance tracking initialized")

        # Adaptive thresholds
        tprint_debug("⚙️ Initializing adaptive thresholds...")
        self.performance_thresholds = self._initialize_thresholds()
        self.adaptive_thresholds = True
        tprint_success("✅ Adaptive thresholds initialized")

        # Alert system
        tprint_debug("🚨 Initializing alert system...")
        self.alert_callbacks: List[callable] = []
        self.alert_history: deque = deque(maxlen=1000)
        tprint_success("✅ Alert system initialized")

        tprint_success(f"✅ Unified Performance Monitor initialized for {architecture_type.value}")
        tprint_info(f"   Monitoring Level: {monitoring_level.value}")
        self.logger.info(f"✅ Unified Performance Monitor initialized for {architecture_type.value}")
        self.logger.info(f"   Monitoring Level: {monitoring_level.value}")
        self.logger.info(f"   Real-time: {enable_real_time}")
        self.logger.info(f"   Max History: {max_history_size}")

    def _initialize_thresholds(self) -> Dict[PerformanceMetric, float]:
        """Initialize performance thresholds based on monitoring level."""
        base_thresholds = {
            PerformanceMetric.ACCURACY: 0.8,
            PerformanceMetric.F1_SCORE: 0.7,
            PerformanceMetric.SHARPE_RATIO: 1.0,
            PerformanceMetric.MAX_DRAWDOWN: 0.2,
            PerformanceMetric.WIN_RATE: 0.6,
            PerformanceMetric.ECONOMIC_SIGNIFICANCE: 0.7,
            PerformanceMetric.TRADING_VIABILITY: 0.6,
            PerformanceMetric.REGIME_STABILITY: 0.8
        }

        if self.monitoring_level == MonitoringLevel.BASIC:
            # Relaxed thresholds for basic monitoring
            return {k: v * 0.8 for k, v in base_thresholds.items()}
        elif self.monitoring_level == MonitoringLevel.COMPREHENSIVE:
            # Stricter thresholds for comprehensive monitoring
            return {k: v * 1.2 for k, v in base_thresholds.items()}
        else:
            return base_thresholds

    def start_monitoring(self):
        """Start real-time performance monitoring."""
        if not self.enable_real_time:
            self.logger.warning("Real-time monitoring is disabled")
            return

        if self.monitoring_active:
            self.logger.warning("Monitoring is already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info("🚀 Real-time performance monitoring started")

    def stop_monitoring(self):
        """Stop real-time performance monitoring."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        self.logger.info("⏹️ Real-time performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop for real-time performance tracking."""
        while self.monitoring_active:
            try:
                self._check_performance_trends()
                self._update_adaptive_thresholds()
                self._check_alerts()

                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def record_performance(self,
                          metrics: Dict[PerformanceMetric, float],
                          iteration: int = None,
                          regime_id: Optional[str] = None,
                          metadata: Dict[str, Any] = None) -> PerformanceSnapshot:
        """Record a performance snapshot.

        Args:
            metrics: Dictionary of performance metrics
            iteration: Current iteration number
            regime_id: Optional regime identifier
            metadata: Optional metadata

        Returns:
            Performance snapshot that was recorded
        """
        if iteration is None:
            iteration = self.current_iteration
            self.current_iteration += 1

        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            architecture_type=self.architecture_type,
            iteration=iteration,
            metrics=metrics,
            regime_id=regime_id,
            metadata=metadata or {}
        )

        # Add to history
        self.performance_history.append(snapshot)

        # Update regime performance if regime_id provided
        if regime_id:
            self._update_regime_performance(snapshot)

        # Update performance trends
        self._update_performance_trends()

        self.logger.debug(f"📊 Performance recorded for iteration {iteration}")
        return snapshot

    def _update_regime_performance(self, snapshot: PerformanceSnapshot):
        """Update regime-specific performance profile."""
        regime_id = snapshot.regime_id

        if regime_id not in self.regime_performance:
            self.regime_performance[regime_id] = RegimePerformanceProfile(
                regime_id=regime_id,
                regime_type="unknown",
                sample_count=0,
                avg_performance={},
                performance_std={},
                stability_score=0.0,
                adaptation_time=0.0,
                last_updated=datetime.now()
            )

        profile = self.regime_performance[regime_id]
        profile.sample_count += 1
        profile.last_updated = datetime.now()

        # Update average performance
        for metric, value in snapshot.metrics.items():
            if metric not in profile.avg_performance:
                profile.avg_performance[metric] = value
                profile.performance_std[metric] = 0.0
            else:
                # Running average
                alpha = 0.1  # Learning rate
                profile.avg_performance[metric] = (
                    alpha * value + (1 - alpha) * profile.avg_performance[metric]
                )

                # Running standard deviation (simplified)
                profile.performance_std[metric] = abs(
                    value - profile.avg_performance[metric]
                )

        # Calculate stability score
        if len(snapshot.metrics) > 0:
            stability_scores = []
            for metric in snapshot.metrics:
                if metric in profile.avg_performance:
                    std_score = 1.0 / (1.0 + profile.performance_std[metric])
                    stability_scores.append(std_score)

            if stability_scores:
                profile.stability_score = np.mean(stability_scores)

    def _update_performance_trends(self):
        """Update performance trend analysis."""
        if len(self.performance_history) < 10:
            return

        recent_snapshots = list(self.performance_history)[-50:]  # Last 50 snapshots
        historical_snapshots = list(self.performance_history)[-200:-50] if len(self.performance_history) > 200 else []

        for metric in PerformanceMetric:
            if not any(snapshot.metrics.get(metric) is not None for snapshot in recent_snapshots):
                continue

            # Get metric values
            recent_values = [s.metrics[metric] for s in recent_snapshots if s.metrics.get(metric) is not None]
            historical_values = [s.metrics[metric] for s in historical_snapshots if s.metrics.get(metric) is not None]

            if len(recent_values) < 5:
                continue

            # Calculate trend
            recent_avg = np.mean(recent_values)
            historical_avg = np.mean(historical_values) if historical_values else recent_avg

            # Trend direction
            if recent_avg > historical_avg * 1.05:
                trend_direction = "improving"
            elif recent_avg < historical_avg * 0.95:
                trend_direction = "declining"
            else:
                trend_direction = "stable"

            # Trend strength
            trend_strength = abs(recent_avg - historical_avg) / (historical_avg + 1e-8)
            trend_strength = min(trend_strength, 1.0)

            # Volatility
            volatility = np.std(recent_values) / (recent_avg + 1e-8)

            # Confidence
            confidence = min(len(recent_values) / 20.0, 1.0)

            self.performance_trends[metric] = PerformanceTrend(
                metric=metric,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                volatility=volatility,
                recent_avg=recent_avg,
                historical_avg=historical_avg,
                confidence=confidence
            )

    def _check_performance_trends(self):
        """Check for significant performance trends."""
        if not self.performance_trends:
            return

        for metric, trend in self.performance_trends.items():
            # Check for declining performance
            if (trend.trend_direction == "declining" and
                trend.trend_strength > 0.2 and
                trend.confidence > 0.7):

                self._trigger_alert(
                    f"Performance declining for {metric.value}",
                    {
                        'metric': metric.value,
                        'trend_direction': trend.trend_direction,
                        'trend_strength': trend.trend_strength,
                        'recent_avg': trend.recent_avg,
                        'historical_avg': trend.historical_avg
                    }
                )

            # Check for high volatility
            elif trend.volatility > 0.5 and trend.confidence > 0.5:
                self._trigger_alert(
                    f"High volatility detected for {metric.value}",
                    {
                        'metric': metric.value,
                        'volatility': trend.volatility,
                        'recent_avg': trend.recent_avg
                    }
                )

    def _update_adaptive_thresholds(self):
        """Update performance thresholds based on historical performance."""
        if not self.adaptive_thresholds or len(self.performance_history) < 100:
            return

        # Calculate adaptive thresholds based on historical performance
        recent_snapshots = list(self.performance_history)[-100:]

        for metric in PerformanceMetric:
            values = [s.metrics[metric] for s in recent_snapshots
                     if s.metrics.get(metric) is not None]

            if len(values) < 10:
                continue

            # Set threshold based on historical performance
            if metric in [PerformanceMetric.ACCURACY, PerformanceMetric.F1_SCORE,
                         PerformanceMetric.WIN_RATE, PerformanceMetric.ECONOMIC_SIGNIFICANCE,
                         PerformanceMetric.TRADING_VIABILITY]:
                # Higher is better
                self.performance_thresholds[metric] = np.percentile(values, 20)
            else:
                # Lower is better (like max_drawdown)
                self.performance_thresholds[metric] = np.percentile(values, 80)

    def _check_alerts(self):
        """Check for alert conditions."""
        if not self.performance_history:
            return

        latest_snapshot = self.performance_history[-1]

        for metric, threshold in self.performance_thresholds.items():
            if metric not in latest_snapshot.metrics:
                continue

            value = latest_snapshot.metrics[metric]

            # Check threshold violations
            if metric in [PerformanceMetric.ACCURACY, PerformanceMetric.F1_SCORE,
                         PerformanceMetric.WIN_RATE, PerformanceMetric.ECONOMIC_SIGNIFICANCE,
                         PerformanceMetric.TRADING_VIABILITY]:
                # Higher is better
                if value < threshold:
                    self._trigger_alert(
                        f"Performance below threshold for {metric.value}",
                        {
                            'metric': metric.value,
                            'value': value,
                            'threshold': threshold,
                            'iteration': latest_snapshot.iteration
                        }
                    )
            else:
                # Lower is better
                if value > threshold:
                    self._trigger_alert(
                        f"Performance above threshold for {metric.value}",
                        {
                            'metric': metric.value,
                            'value': value,
                            'threshold': threshold,
                            'iteration': latest_snapshot.iteration
                        }
                    )

    def _trigger_alert(self, message: str, data: Dict[str, Any]):
        """Trigger an alert with the given message and data."""
        alert = {
            'timestamp': datetime.now(),
            'message': message,
            'data': data,
            'architecture_type': self.architecture_type.value
        }

        self.alert_history.append(alert)

        # Call registered alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")

        self.logger.warning(f"🚨 ALERT: {message}")

    def register_alert_callback(self, callback: callable):
        """Register a callback function for alerts."""
        self.alert_callbacks.append(callback)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary."""
        if not self.performance_history:
            return {'error': 'No performance data available'}

        latest_snapshot = self.performance_history[-1]

        summary = {
            'architecture_type': self.architecture_type.value,
            'total_iterations': len(self.performance_history),
            'monitoring_duration': time.time() - self.start_time,
            'latest_performance': latest_snapshot.metrics,
            'performance_trends': {
                metric.value: {
                    'trend_direction': trend.trend_direction,
                    'trend_strength': trend.trend_strength,
                    'volatility': trend.volatility,
                    'recent_avg': trend.recent_avg,
                    'confidence': trend.confidence
                } for metric, trend in self.performance_trends.items()
            },
            'regime_performance': {
                regime_id: {
                    'sample_count': profile.sample_count,
                    'avg_performance': {k.value: v for k, v in profile.avg_performance.items()},
                    'stability_score': profile.stability_score,
                    'last_updated': profile.last_updated.isoformat()
                } for regime_id, profile in self.regime_performance.items()
            },
            'performance_thresholds': {
                metric.value: threshold for metric, threshold in self.performance_thresholds.items()
            },
            'alert_count': len(self.alert_history),
            'recent_alerts': list(self.alert_history)[-5:] if self.alert_history else []
        }

        return summary

    def get_regime_performance_comparison(self) -> Dict[str, Any]:
        """Compare performance across different regimes."""
        if not self.regime_performance:
            return {'error': 'No regime performance data available'}

        comparison = {}

        for regime_id, profile in self.regime_performance.items():
            comparison[regime_id] = {
                'sample_count': profile.sample_count,
                'stability_score': profile.stability_score,
                'avg_performance': {k.value: v for k, v in profile.avg_performance.items()},
                'performance_std': {k.value: v for k, v in profile.performance_std.items()},
                'last_updated': profile.last_updated.isoformat()
            }

        return comparison

    def get_performance_trend_analysis(self) -> Dict[str, Any]:
        """Get detailed performance trend analysis."""
        if not self.performance_trends:
            return {'error': 'No trend data available'}

        analysis = {}

        for metric, trend in self.performance_trends.items():
            analysis[metric.value] = {
                'trend_direction': trend.trend_direction,
                'trend_strength': trend.trend_strength,
                'volatility': trend.volatility,
                'recent_avg': trend.recent_avg,
                'historical_avg': trend.historical_avg,
                'confidence': trend.confidence,
                'improvement_ratio': trend.recent_avg / (trend.historical_avg + 1e-8)
            }

        return analysis

    def export_performance_data(self, filepath: str, format: str = "json"):
        """Export performance data to file.

        Args:
            filepath: Path to save the data
            format: Export format ("json" or "pickle")
        """
        try:
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                'performance_history': [
                    {
                        'timestamp': s.timestamp.isoformat(),
                        'architecture_type': s.architecture_type.value,
                        'iteration': s.iteration,
                        'metrics': {k.value: v for k, v in s.metrics.items()},
                        'regime_id': s.regime_id,
                        'metadata': s.metadata
                    } for s in self.performance_history
                ],
                'regime_performance': {
                    regime_id: {
                        'regime_id': profile.regime_id,
                        'regime_type': profile.regime_type,
                        'sample_count': profile.sample_count,
                        'avg_performance': {k.value: v for k, v in profile.avg_performance.items()},
                        'performance_std': {k.value: v for k, v in profile.performance_std.items()},
                        'stability_score': profile.stability_score,
                        'adaptation_time': profile.adaptation_time,
                        'last_updated': profile.last_updated.isoformat()
                    } for regime_id, profile in self.regime_performance.items()
                },
                'performance_trends': {
                    metric.value: {
                        'trend_direction': trend.trend_direction,
                        'trend_strength': trend.trend_strength,
                        'volatility': trend.volatility,
                        'recent_avg': trend.recent_avg,
                        'historical_avg': trend.historical_avg,
                        'confidence': trend.confidence
                    } for metric, trend in self.performance_trends.items()
                },
                'alert_history': list(self.alert_history),
                'performance_thresholds': {
                    metric.value: threshold for metric, threshold in self.performance_thresholds.items()
                },
                'monitoring_config': {
                    'architecture_type': self.architecture_type.value,
                    'monitoring_level': self.monitoring_level.value,
                    'max_history_size': self.max_history_size,
                    'monitoring_interval': self.monitoring_interval,
                    'enable_real_time': self.enable_real_time
                }
            }

            if format == "json":
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                with open(output_path, 'wb') as f:
                    pickle.dump(data, f)

            self.logger.info(f"✅ Performance data exported to {filepath}")

        except Exception as e:
            self.logger.error(f"❌ Failed to export performance data: {e}")
            raise

    def import_performance_data(self, filepath: str, format: str = "json"):
        """Import performance data from file."""
        try:
            with open(filepath, 'rb' if format == "pickle" else 'r') as f:
                if format == "json":
                    data = json.load(f)
                else:
                    data = pickle.load(f)

            # Import performance history
            for snapshot_data in data.get('performance_history', []):
                snapshot = PerformanceSnapshot(
                    timestamp=datetime.fromisoformat(snapshot_data['timestamp']),
                    architecture_type=ArchitectureType(snapshot_data['architecture_type']),
                    iteration=snapshot_data['iteration'],
                    metrics={PerformanceMetric(k): v for k, v in snapshot_data['metrics'].items()},
                    regime_id=snapshot_data.get('regime_id'),
                    metadata=snapshot_data.get('metadata', {})
                )
                self.performance_history.append(snapshot)

            # Import regime performance
            for regime_id, profile_data in data.get('regime_performance', {}).items():
                profile = RegimePerformanceProfile(
                    regime_id=profile_data['regime_id'],
                    regime_type=profile_data['regime_type'],
                    sample_count=profile_data['sample_count'],
                    avg_performance={PerformanceMetric(k): v for k, v in profile_data['avg_performance'].items()},
                    performance_std={PerformanceMetric(k): v for k, v in profile_data['performance_std'].items()},
                    stability_score=profile_data['stability_score'],
                    adaptation_time=profile_data['adaptation_time'],
                    last_updated=datetime.fromisoformat(profile_data['last_updated'])
                )
                self.regime_performance[regime_id] = profile

            self.logger.info(f"✅ Performance data imported from {filepath}")

        except Exception as e:
            self.logger.error(f"❌ Failed to import performance data: {e}")
            raise

# Convenience functions
def create_performance_monitor(architecture_type: ArchitectureType,
                             monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD,
                             **kwargs) -> UnifiedPerformanceMonitor:
    """Create a performance monitor with default settings."""
    return UnifiedPerformanceMonitor(
        architecture_type=architecture_type,
        monitoring_level=monitoring_level,
        **kwargs
    )

def create_basic_monitor(architecture_type: ArchitectureType) -> UnifiedPerformanceMonitor:
    """Create a basic performance monitor."""
    return UnifiedPerformanceMonitor(
        architecture_type=architecture_type,
        monitoring_level=MonitoringLevel.BASIC,
        enable_real_time=False
    )

def create_real_time_monitor(architecture_type: ArchitectureType) -> UnifiedPerformanceMonitor:
    """Create a real-time performance monitor."""
    return UnifiedPerformanceMonitor(
        architecture_type=architecture_type,
        monitoring_level=MonitoringLevel.REAL_TIME,
        enable_real_time=True,
        monitoring_interval=0.5
    )
