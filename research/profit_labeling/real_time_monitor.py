"""
Real-Time Labeling Performance Monitor

This module provides real-time monitoring of profit labeling performance with
automatic drift detection and recalibration capabilities. It tracks labeling
quality over time and triggers updates when performance degrades.

Key Monitoring Components:
1. Performance Tracking (Quality metrics over time)
2. Drift Detection (Statistical and ML-based)
3. Automatic Recalibration Triggers
4. Alert System for Quality Degradation
5. Adaptive Threshold Management
6. Historical Performance Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
from collections import deque
import warnings

# Statistical imports for drift detection
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger


class DriftDetectionMethod(Enum):
    """Enumeration of drift detection methods."""
    STATISTICAL_TEST = "statistical_test"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    ROLLING_CORRELATION = "rolling_correlation"
    DISTRIBUTION_CHANGE = "distribution_change"
    ML_BASED = "ml_based"
    ENSEMBLE = "ensemble"


class AlertLevel(Enum):
    """Enumeration of alert levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class PerformanceMetric(Enum):
    """Enumeration of performance metrics to monitor."""
    PREDICTIVE_POWER = "predictive_power"
    LABEL_STABILITY = "label_stability"
    HIT_RATE = "hit_rate"
    CORRELATION = "correlation"
    SHARPE_RATIO = "sharpe_ratio"
    INFORMATION_RATIO = "information_ratio"
    QUALITY_SCORE = "quality_score"


@dataclass
class MonitoringConfig:
    """Configuration for real-time performance monitoring."""
    # Monitoring parameters
    monitoring_window: int = 1000  # Number of samples to keep in memory
    update_frequency: int = 50     # Update metrics every N samples
    drift_detection_window: int = 200  # Window for drift detection
    
    # Performance thresholds
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'critical': 0.2,   # Below this triggers emergency recalibration
        'warning': 0.4,    # Below this triggers warning
        'good': 0.6,       # Above this is considered good
        'excellent': 0.8   # Above this is excellent
    })
    
    # Drift detection parameters
    drift_detection_methods: List[DriftDetectionMethod] = field(default_factory=lambda: [
        DriftDetectionMethod.STATISTICAL_TEST,
        DriftDetectionMethod.PERFORMANCE_THRESHOLD,
        DriftDetectionMethod.ROLLING_CORRELATION
    ])
    
    drift_sensitivity: float = 0.05  # P-value threshold for drift detection
    min_samples_for_drift: int = 100
    
    # Alert configuration
    enable_alerts: bool = True
    alert_cooldown_minutes: int = 30  # Minimum time between similar alerts
    
    # Recalibration triggers
    auto_recalibration: bool = True
    recalibration_threshold: float = 0.3  # Quality threshold for auto recalibration
    max_recalibration_frequency: int = 100  # Maximum once per N samples
    
    # Historical tracking
    save_monitoring_data: bool = True
    monitoring_data_retention_days: int = 30
    
    # Performance comparison
    benchmark_comparison: bool = True
    benchmark_window: int = 500


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time."""
    timestamp: datetime
    sample_count: int
    quality_metrics: Dict[PerformanceMetric, float]
    label_statistics: Dict[str, float]
    market_conditions: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""
    drift_detected: bool
    drift_confidence: float
    detection_method: DriftDetectionMethod
    drift_magnitude: float
    affected_metrics: List[PerformanceMetric]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MonitoringAlert:
    """Alert for performance issues."""
    alert_level: AlertLevel
    alert_type: str
    message: str
    affected_metrics: List[str]
    current_values: Dict[str, float]
    threshold_values: Dict[str, float]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class LabelingPerformanceTracker:
    """Tracks labeling performance metrics over time."""
    
    def __init__(self, config: MonitoringConfig):
        """Initialize performance tracker."""
        self.config = config
        self.logger = get_logger('LabelingPerformanceTracker')
        
        # Performance history
        self.performance_history: deque = deque(maxlen=config.monitoring_window)
        self.metric_history: Dict[PerformanceMetric, deque] = {
            metric: deque(maxlen=config.monitoring_window) 
            for metric in PerformanceMetric
        }
        
        # Current state
        self.current_snapshot: Optional[PerformanceSnapshot] = None
        self.sample_counter: int = 0
        
        self.logger.info('📊 Performance Tracker initialized')
    
    def track_performance(self,
                         labeled_data: pd.DataFrame,
                         market_data: pd.DataFrame,
                         predictions: Optional[pd.DataFrame] = None) -> PerformanceSnapshot:
        """Track performance for current labeling."""
        self.sample_counter += len(labeled_data)
        
        # Calculate current performance metrics
        quality_metrics = self._calculate_quality_metrics(labeled_data, market_data, predictions)
        label_statistics = self._calculate_label_statistics(labeled_data)
        market_conditions = self._extract_market_conditions(market_data)
        
        # Create performance snapshot
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            sample_count=self.sample_counter,
            quality_metrics=quality_metrics,
            label_statistics=label_statistics,
            market_conditions=market_conditions
        )
        
        # Store in history
        self.performance_history.append(snapshot)
        
        # Update metric-specific history
        for metric, value in quality_metrics.items():
            self.metric_history[metric].append(value)
        
        self.current_snapshot = snapshot
        
        return snapshot
    
    def _calculate_quality_metrics(self,
                                  labeled_data: pd.DataFrame,
                                  market_data: pd.DataFrame,
                                  predictions: Optional[pd.DataFrame]) -> Dict[PerformanceMetric, float]:
        """Calculate quality metrics."""
        metrics = {}
        
        # Predictive power (correlation with future returns)
        if 'close' in market_data.columns and 'overall_opportunity' in labeled_data.columns:
            returns = market_data['close'].pct_change().shift(-1).fillna(0)
            opportunity = labeled_data['overall_opportunity'].fillna(0)
            
            common_idx = opportunity.index.intersection(returns.index)
            if len(common_idx) > 10:
                corr = np.corrcoef(opportunity.loc[common_idx], returns.loc[common_idx])[0, 1]
                metrics[PerformanceMetric.PREDICTIVE_POWER] = abs(corr) if not np.isnan(corr) else 0.0
                metrics[PerformanceMetric.CORRELATION] = corr if not np.isnan(corr) else 0.0
        
        # Label stability
        prob_columns = [col for col in labeled_data.columns if col.endswith('_prob')]
        if prob_columns:
            stability_scores = []
            for col in prob_columns[:3]:  # Top 3 for efficiency
                values = labeled_data[col].dropna()
                if len(values) > 20:
                    rolling_std = values.rolling(20).std()
                    if rolling_std.mean() > 0 and values.std() > 0:
                        stability = 1.0 - (rolling_std.std() / rolling_std.mean())
                        stability_scores.append(max(0, min(1, stability)))
            
            metrics[PerformanceMetric.LABEL_STABILITY] = np.mean(stability_scores) if stability_scores else 0.5
        
        # Hit rate
        if 'overall_opportunity' in labeled_data.columns:
            opportunity = labeled_data['overall_opportunity'].fillna(0)
            hit_rate = (opportunity > 0.5).mean()
            metrics[PerformanceMetric.HIT_RATE] = hit_rate
        
        # Quality score (if available)
        quality_columns = [col for col in labeled_data.columns if 'quality' in col.lower()]
        if quality_columns:
            avg_quality = labeled_data[quality_columns].mean().mean()
            metrics[PerformanceMetric.QUALITY_SCORE] = avg_quality
        
        return metrics
    
    def _calculate_label_statistics(self, labeled_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate label distribution statistics."""
        stats = {}
        
        # Overall opportunity statistics
        if 'overall_opportunity' in labeled_data.columns:
            opportunity = labeled_data['overall_opportunity'].fillna(0)
            stats['opportunity_mean'] = float(opportunity.mean())
            stats['opportunity_std'] = float(opportunity.std())
            stats['opportunity_min'] = float(opportunity.min())
            stats['opportunity_max'] = float(opportunity.max())
            stats['high_opportunity_ratio'] = float((opportunity > 0.7).mean())
        
        # Probability column statistics
        prob_columns = [col for col in labeled_data.columns if col.endswith('_prob')]
        if prob_columns:
            prob_values = labeled_data[prob_columns].mean(axis=1)
            stats['avg_prob_mean'] = float(prob_values.mean())
            stats['avg_prob_std'] = float(prob_values.std())
        
        return stats
    
    def _extract_market_conditions(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract current market condition indicators."""
        conditions = {}
        
        if 'close' in market_data.columns:
            prices = market_data['close']
            returns = prices.pct_change()
            
            # Volatility
            recent_vol = returns.tail(20).std()
            conditions['volatility'] = float(recent_vol) if not np.isnan(recent_vol) else 0.0
            
            # Trend (last 50 periods)
            if len(prices) >= 50:
                recent_prices = prices.tail(50)
                slope, _, r_value, _, _ = stats.linregress(range(len(recent_prices)), recent_prices)
                conditions['trend_slope'] = float(slope / prices.iloc[-1])  # Normalized
                conditions['trend_strength'] = float(abs(r_value))
            
            # Momentum
            if len(prices) >= 10:
                momentum = (prices.iloc[-1] / prices.iloc[-10]) - 1
                conditions['momentum'] = float(momentum)
        
        return conditions
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of recent performance."""
        if not self.performance_history:
            return {}
        
        summary = {}
        
        # Recent snapshots
        recent_snapshots = list(self.performance_history)[-10:]  # Last 10 snapshots
        
        # Calculate trends for each metric
        for metric in PerformanceMetric:
            values = [s.quality_metrics.get(metric, 0) for s in recent_snapshots]
            if values:
                summary[metric.value] = {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0,
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Overall summary
        summary['monitoring_status'] = {
            'samples_processed': self.sample_counter,
            'snapshots_recorded': len(self.performance_history),
            'monitoring_duration_hours': (
                (recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp).total_seconds() / 3600
                if len(recent_snapshots) > 1 else 0
            )
        }
        
        return summary


class LabelingDriftDetector:
    """Detects drift in labeling performance."""
    
    def __init__(self, config: MonitoringConfig):
        """Initialize drift detector."""
        self.config = config
        self.logger = get_logger('LabelingDriftDetector')
        
        # Drift detection state
        self.baseline_performance: Optional[Dict[PerformanceMetric, float]] = None
        self.drift_history: List[DriftDetectionResult] = []
        
        self.logger.info('🔍 Drift Detector initialized')
    
    def detect_drift(self, performance_history: List[PerformanceSnapshot]) -> Optional[DriftDetectionResult]:
        """Detect drift in labeling performance."""
        if len(performance_history) < self.config.min_samples_for_drift:
            return None
        
        # Establish baseline if not set
        if self.baseline_performance is None:
            self._establish_baseline(performance_history[:self.config.drift_detection_window])
        
        # Get recent performance
        recent_performance = performance_history[-self.config.drift_detection_window:]
        
        # Run drift detection methods
        drift_results = []
        
        for method in self.config.drift_detection_methods:
            try:
                if method == DriftDetectionMethod.STATISTICAL_TEST:
                    result = self._detect_statistical_drift(recent_performance)
                elif method == DriftDetectionMethod.PERFORMANCE_THRESHOLD:
                    result = self._detect_threshold_drift(recent_performance)
                elif method == DriftDetectionMethod.ROLLING_CORRELATION:
                    result = self._detect_correlation_drift(recent_performance)
                elif method == DriftDetectionMethod.DISTRIBUTION_CHANGE:
                    result = self._detect_distribution_drift(recent_performance)
                else:
                    continue
                
                if result:
                    drift_results.append(result)
                    
            except Exception as e:
                self.logger.warning(f'Drift detection method {method.value} failed: {e}')
        
        # Combine drift detection results
        if drift_results:
            return self._combine_drift_results(drift_results)
        
        return None
    
    def _establish_baseline(self, baseline_snapshots: List[PerformanceSnapshot]):
        """Establish baseline performance metrics."""
        self.baseline_performance = {}
        
        for metric in PerformanceMetric:
            values = [s.quality_metrics.get(metric, 0) for s in baseline_snapshots]
            if values:
                self.baseline_performance[metric] = np.mean(values)
        
        self.logger.info(f'📊 Baseline performance established from {len(baseline_snapshots)} snapshots')
    
    def _detect_statistical_drift(self, recent_snapshots: List[PerformanceSnapshot]) -> Optional[DriftDetectionResult]:
        """Detect drift using statistical tests."""
        if not self.baseline_performance:
            return None
        
        drift_detected = False
        affected_metrics = []
        p_values = []
        
        for metric in PerformanceMetric:
            baseline_value = self.baseline_performance.get(metric, 0)
            recent_values = [s.quality_metrics.get(metric, 0) for s in recent_snapshots]
            
            if len(recent_values) > 10:
                # One-sample t-test against baseline
                t_stat, p_value = stats.ttest_1samp(recent_values, baseline_value)
                p_values.append(p_value)
                
                if p_value < self.config.drift_sensitivity:
                    drift_detected = True
                    affected_metrics.append(metric)
        
        if drift_detected:
            avg_p_value = np.mean(p_values)
            drift_confidence = 1.0 - avg_p_value
            drift_magnitude = np.mean([
                abs(np.mean([s.quality_metrics.get(m, 0) for s in recent_snapshots]) - 
                    self.baseline_performance.get(m, 0))
                for m in affected_metrics
            ])
            
            return DriftDetectionResult(
                drift_detected=True,
                drift_confidence=drift_confidence,
                detection_method=DriftDetectionMethod.STATISTICAL_TEST,
                drift_magnitude=drift_magnitude,
                affected_metrics=affected_metrics,
                recommendations=self._generate_drift_recommendations(affected_metrics),
                metadata={'p_values': p_values, 'avg_p_value': avg_p_value}
            )
        
        return None
    
    def _detect_threshold_drift(self, recent_snapshots: List[PerformanceSnapshot]) -> Optional[DriftDetectionResult]:
        """Detect drift based on performance thresholds."""
        if len(recent_snapshots) < 10:
            return None
        
        # Check if recent performance is below critical thresholds
        recent_quality = np.mean([
            s.quality_metrics.get(PerformanceMetric.QUALITY_SCORE, 0.5) 
            for s in recent_snapshots
        ])
        
        critical_threshold = self.config.quality_thresholds['critical']
        
        if recent_quality < critical_threshold:
            drift_magnitude = critical_threshold - recent_quality
            
            return DriftDetectionResult(
                drift_detected=True,
                drift_confidence=min(1.0, drift_magnitude / critical_threshold),
                detection_method=DriftDetectionMethod.PERFORMANCE_THRESHOLD,
                drift_magnitude=drift_magnitude,
                affected_metrics=[PerformanceMetric.QUALITY_SCORE],
                recommendations=['Immediate recalibration required', 'Review labeling parameters'],
                metadata={'recent_quality': recent_quality, 'threshold': critical_threshold}
            )
        
        return None
    
    def _detect_correlation_drift(self, recent_snapshots: List[PerformanceSnapshot]) -> Optional[DriftDetectionResult]:
        """Detect drift in correlation patterns."""
        if len(recent_snapshots) < 20:
            return None
        
        # Extract correlation values
        correlations = [
            s.quality_metrics.get(PerformanceMetric.CORRELATION, 0) 
            for s in recent_snapshots
        ]
        
        if len(correlations) < 20:
            return None
        
        # Test for trend in correlations
        x = np.arange(len(correlations))
        slope, _, r_value, p_value, _ = stats.linregress(x, correlations)
        
        # Drift if significant negative trend in correlation
        if p_value < self.config.drift_sensitivity and slope < -0.001:  # Declining correlation
            return DriftDetectionResult(
                drift_detected=True,
                drift_confidence=abs(r_value),
                detection_method=DriftDetectionMethod.ROLLING_CORRELATION,
                drift_magnitude=abs(slope),
                affected_metrics=[PerformanceMetric.CORRELATION],
                recommendations=['Monitor correlation decline', 'Consider parameter adjustment'],
                metadata={'slope': slope, 'r_value': r_value, 'p_value': p_value}
            )
        
        return None
    
    def _detect_distribution_drift(self, recent_snapshots: List[PerformanceSnapshot]) -> Optional[DriftDetectionResult]:
        """Detect drift in label distributions."""
        if len(recent_snapshots) < 30:
            return None
        
        # Split into two periods for comparison
        mid_point = len(recent_snapshots) // 2
        early_period = recent_snapshots[:mid_point]
        late_period = recent_snapshots[mid_point:]
        
        # Compare quality score distributions
        early_quality = [s.quality_metrics.get(PerformanceMetric.QUALITY_SCORE, 0.5) for s in early_period]
        late_quality = [s.quality_metrics.get(PerformanceMetric.QUALITY_SCORE, 0.5) for s in late_period]
        
        if len(early_quality) > 5 and len(late_quality) > 5:
            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(early_quality, late_quality)
            
            if p_value < self.config.drift_sensitivity:
                return DriftDetectionResult(
                    drift_detected=True,
                    drift_confidence=ks_stat,
                    detection_method=DriftDetectionMethod.DISTRIBUTION_CHANGE,
                    drift_magnitude=ks_stat,
                    affected_metrics=[PerformanceMetric.QUALITY_SCORE],
                    recommendations=['Distribution shift detected', 'Review labeling consistency'],
                    metadata={'ks_statistic': ks_stat, 'p_value': p_value}
                )
        
        return None
    
    def _combine_drift_results(self, drift_results: List[DriftDetectionResult]) -> DriftDetectionResult:
        """Combine multiple drift detection results."""
        # Take the most confident detection
        best_result = max(drift_results, key=lambda x: x.drift_confidence)
        
        # Combine affected metrics
        all_affected_metrics = []
        for result in drift_results:
            all_affected_metrics.extend(result.affected_metrics)
        
        unique_affected_metrics = list(set(all_affected_metrics))
        
        # Combine recommendations
        all_recommendations = []
        for result in drift_results:
            all_recommendations.extend(result.recommendations)
        
        unique_recommendations = list(set(all_recommendations))
        
        return DriftDetectionResult(
            drift_detected=True,
            drift_confidence=best_result.drift_confidence,
            detection_method=DriftDetectionMethod.ENSEMBLE,
            drift_magnitude=np.mean([r.drift_magnitude for r in drift_results]),
            affected_metrics=unique_affected_metrics,
            recommendations=unique_recommendations,
            metadata={
                'individual_results': len(drift_results),
                'detection_methods': [r.detection_method.value for r in drift_results]
            }
        )
    
    def _generate_drift_recommendations(self, affected_metrics: List[PerformanceMetric]) -> List[str]:
        """Generate recommendations for detected drift."""
        recommendations = []
        
        if PerformanceMetric.PREDICTIVE_POWER in affected_metrics:
            recommendations.extend([
                'Predictive power decline detected',
                'Consider retraining ML components',
                'Review feature engineering'
            ])
        
        if PerformanceMetric.LABEL_STABILITY in affected_metrics:
            recommendations.extend([
                'Label stability issues detected',
                'Review parameter stability',
                'Consider smoothing techniques'
            ])
        
        if PerformanceMetric.HIT_RATE in affected_metrics:
            recommendations.extend([
                'Hit rate changes detected',
                'Review target levels',
                'Consider market regime changes'
            ])
        
        if not recommendations:
            recommendations.append('General performance drift detected - review overall system')
        
        return recommendations


class AutoRecalibrator:
    """Automatic recalibration system for labeling parameters."""
    
    def __init__(self, config: MonitoringConfig):
        """Initialize auto recalibrator."""
        self.config = config
        self.logger = get_logger('AutoRecalibrator')
        
        # Recalibration state
        self.last_recalibration: Optional[datetime] = None
        self.recalibration_history: List[Dict[str, Any]] = []
        
        self.logger.info('🔧 Auto Recalibrator initialized')
    
    def should_recalibrate(self, 
                          drift_result: Optional[DriftDetectionResult],
                          current_performance: PerformanceSnapshot) -> bool:
        """Determine if recalibration should be triggered."""
        if not self.config.auto_recalibration:
            return False
        
        # Check time since last recalibration
        if self.last_recalibration:
            time_since_recal = datetime.now() - self.last_recalibration
            min_interval = timedelta(minutes=self.config.max_recalibration_frequency * 5)  # Assume 5min periods
            
            if time_since_recal < min_interval:
                return False
        
        # Check if drift is detected
        if drift_result and drift_result.drift_detected:
            if drift_result.drift_confidence > 0.7:  # High confidence drift
                return True
        
        # Check if performance is below threshold
        quality_score = current_performance.quality_metrics.get(PerformanceMetric.QUALITY_SCORE, 0.5)
        if quality_score < self.config.recalibration_threshold:
            return True
        
        return False
    
    def recalibrate(self, market_data: pd.DataFrame) -> Optional[Any]:
        """Perform automatic recalibration."""
        self.logger.info('🔄 Performing automatic recalibration')
        
        try:
            # Use dynamic optimization to find new parameters
            from .dynamic_target_optimizer import discover_optimal_targets_and_horizons
            
            optimization_result = discover_optimal_targets_and_horizons(market_data)
            
            if optimization_result.objective_score > 0.5:  # Accept if reasonable score
                # Record recalibration
                self.last_recalibration = datetime.now()
                self.recalibration_history.append({
                    'timestamp': self.last_recalibration,
                    'trigger': 'auto_recalibration',
                    'objective_score': optimization_result.objective_score,
                    'new_targets': optimization_result.optimal_targets,
                    'new_horizons': optimization_result.optimal_horizons
                })
                
                self.logger.info(f'✅ Recalibration completed with score: {optimization_result.objective_score:.3f}')
                return optimization_result
            
        except Exception as e:
            self.logger.error(f'Recalibration failed: {e}')
        
        return None


class RealTimeLabelingMonitor:
    """
    Main real-time monitoring system for profit labeling.
    
    Coordinates performance tracking, drift detection, and automatic recalibration
    to maintain optimal labeling performance over time.
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        """Initialize real-time monitor."""
        self.config = config or MonitoringConfig()
        self.logger = get_logger('RealTimeLabelingMonitor')
        
        # Initialize components
        self.performance_tracker = LabelingPerformanceTracker(self.config)
        self.drift_detector = LabelingDriftDetector(self.config)
        self.auto_recalibrator = AutoRecalibrator(self.config)
        
        # Alert system
        self.alerts: List[MonitoringAlert] = []
        self.last_alert_time: Dict[str, datetime] = {}
        
        # Monitoring state
        self.monitoring_active: bool = False
        self.update_counter: int = 0
        
        self.logger.info('📡 Real-Time Labeling Monitor initialized')
        self.logger.info(f'   → Update frequency: every {self.config.update_frequency} samples')
        self.logger.info(f'   → Auto recalibration: {"enabled" if self.config.auto_recalibration else "disabled"}')
    
    def monitor_labeling_performance(self,
                                   labeled_data: pd.DataFrame,
                                   market_data: pd.DataFrame,
                                   predictions: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Monitor labeling performance and trigger actions if needed.
        
        Args:
            labeled_data: Current labeled data
            market_data: Current market data
            predictions: Optional ML predictions
            
        Returns:
            Dictionary with monitoring results and any triggered actions
        """
        self.update_counter += 1
        
        # Track performance
        current_snapshot = self.performance_tracker.track_performance(
            labeled_data, market_data, predictions
        )
        
        monitoring_result = {
            'timestamp': current_snapshot.timestamp,
            'sample_count': current_snapshot.sample_count,
            'current_quality': current_snapshot.quality_metrics.get(PerformanceMetric.QUALITY_SCORE, 0.5),
            'alerts_generated': [],
            'recalibration_triggered': False,
            'drift_detected': False
        }
        
        # Periodic monitoring updates
        if self.update_counter % self.config.update_frequency == 0:
            self.logger.info(f'📊 Monitoring update #{self.update_counter // self.config.update_frequency}')
            
            # Check for performance issues
            alerts = self._check_performance_alerts(current_snapshot)
            if alerts:
                monitoring_result['alerts_generated'] = alerts
                self.alerts.extend(alerts)
            
            # Detect drift
            if len(self.performance_tracker.performance_history) >= self.config.min_samples_for_drift:
                drift_result = self.drift_detector.detect_drift(
                    list(self.performance_tracker.performance_history)
                )
                
                if drift_result and drift_result.drift_detected:
                    monitoring_result['drift_detected'] = True
                    monitoring_result['drift_confidence'] = drift_result.drift_confidence
                    
                    self.logger.warning(f'⚠️ Drift detected: {drift_result.detection_method.value}')
                    
                    # Check if recalibration should be triggered
                    if self.auto_recalibrator.should_recalibrate(drift_result, current_snapshot):
                        recalibration_result = self.auto_recalibrator.recalibrate(market_data)
                        
                        if recalibration_result:
                            monitoring_result['recalibration_triggered'] = True
                            monitoring_result['new_config'] = recalibration_result
                            
                            self.logger.info('🔄 Automatic recalibration triggered')
        
        return monitoring_result
    
    def _check_performance_alerts(self, snapshot: PerformanceSnapshot) -> List[MonitoringAlert]:
        """Check for performance alerts."""
        alerts = []
        
        if not self.config.enable_alerts:
            return alerts
        
        # Check quality score thresholds
        quality_score = snapshot.quality_metrics.get(PerformanceMetric.QUALITY_SCORE, 0.5)
        
        if quality_score < self.config.quality_thresholds['critical']:
            if self._should_generate_alert('critical_quality'):
                alert = MonitoringAlert(
                    alert_level=AlertLevel.CRITICAL,
                    alert_type='quality_degradation',
                    message=f'Critical quality degradation: {quality_score:.3f}',
                    affected_metrics=['quality_score'],
                    current_values={'quality_score': quality_score},
                    threshold_values={'critical': self.config.quality_thresholds['critical']},
                    recommendations=[
                        'Immediate attention required',
                        'Check data quality',
                        'Consider emergency recalibration'
                    ]
                )
                alerts.append(alert)
        
        elif quality_score < self.config.quality_thresholds['warning']:
            if self._should_generate_alert('warning_quality'):
                alert = MonitoringAlert(
                    alert_level=AlertLevel.WARNING,
                    alert_type='quality_decline',
                    message=f'Quality decline detected: {quality_score:.3f}',
                    affected_metrics=['quality_score'],
                    current_values={'quality_score': quality_score},
                    threshold_values={'warning': self.config.quality_thresholds['warning']},
                    recommendations=[
                        'Monitor closely',
                        'Review recent performance',
                        'Consider parameter adjustment'
                    ]
                )
                alerts.append(alert)
        
        # Check predictive power
        predictive_power = snapshot.quality_metrics.get(PerformanceMetric.PREDICTIVE_POWER, 0.5)
        if predictive_power < 0.3:  # Very low predictive power
            if self._should_generate_alert('low_predictive_power'):
                alert = MonitoringAlert(
                    alert_level=AlertLevel.WARNING,
                    alert_type='low_predictive_power',
                    message=f'Low predictive power: {predictive_power:.3f}',
                    affected_metrics=['predictive_power'],
                    current_values={'predictive_power': predictive_power},
                    threshold_values={'minimum': 0.3},
                    recommendations=[
                        'Labels may not be useful for ML',
                        'Review labeling methodology',
                        'Check feature engineering'
                    ]
                )
                alerts.append(alert)
        
        return alerts
    
    def _should_generate_alert(self, alert_type: str) -> bool:
        """Check if alert should be generated (considering cooldown)."""
        if alert_type in self.last_alert_time:
            time_since_last = datetime.now() - self.last_alert_time[alert_type]
            cooldown = timedelta(minutes=self.config.alert_cooldown_minutes)
            
            if time_since_last < cooldown:
                return False
        
        self.last_alert_time[alert_type] = datetime.now()
        return True
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        dashboard_data = {}
        
        # Performance summary
        dashboard_data['performance_summary'] = self.performance_tracker.get_performance_summary()
        
        # Recent alerts
        recent_alerts = [a for a in self.alerts if 
                        (datetime.now() - a.timestamp).total_seconds() < 3600]  # Last hour
        dashboard_data['recent_alerts'] = len(recent_alerts)
        dashboard_data['alert_levels'] = {
            level.value: sum(1 for a in recent_alerts if a.alert_level == level)
            for level in AlertLevel
        }
        
        # Drift history
        recent_drift = [d for d in self.drift_detector.drift_history if
                       (datetime.now() - d.timestamp).total_seconds() < 3600]
        dashboard_data['drift_detections'] = len(recent_drift)
        
        # Recalibration history
        dashboard_data['recalibrations'] = len(self.auto_recalibrator.recalibration_history)
        
        # Current status
        if self.performance_tracker.current_snapshot:
            current_quality = self.performance_tracker.current_snapshot.quality_metrics.get(
                PerformanceMetric.QUALITY_SCORE, 0.5
            )
            
            if current_quality >= self.config.quality_thresholds['excellent']:
                status = 'excellent'
            elif current_quality >= self.config.quality_thresholds['good']:
                status = 'good'
            elif current_quality >= self.config.quality_thresholds['warning']:
                status = 'acceptable'
            else:
                status = 'poor'
            
            dashboard_data['current_status'] = status
        else:
            dashboard_data['current_status'] = 'unknown'
        
        return dashboard_data
    
    def save_monitoring_state(self, output_path: Union[str, Path]):
        """Save monitoring state to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare serializable data
        state_data = {
            'config': self.config.__dict__,
            'performance_summary': self.performance_tracker.get_performance_summary(),
            'drift_history': [
                {
                    'timestamp': d.timestamp.isoformat(),
                    'drift_detected': d.drift_detected,
                    'drift_confidence': d.drift_confidence,
                    'detection_method': d.detection_method.value,
                    'affected_metrics': [m.value for m in d.affected_metrics]
                }
                for d in self.drift_detector.drift_history
            ],
            'recalibration_history': self.auto_recalibrator.recalibration_history,
            'alert_count': len(self.alerts),
            'monitoring_duration': self.update_counter
        }
        
        with open(output_path, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        self.logger.info(f'💾 Monitoring state saved to {output_path}')


# Convenience functions
def create_real_time_monitor(config: Optional[MonitoringConfig] = None) -> RealTimeLabelingMonitor:
    """Create real-time labeling monitor."""
    return RealTimeLabelingMonitor(config)


def monitor_labeling_quality(labeled_data: pd.DataFrame,
                           market_data: pd.DataFrame,
                           monitor: Optional[RealTimeLabelingMonitor] = None) -> Dict[str, Any]:
    """Convenience function to monitor labeling quality."""
    if monitor is None:
        monitor = RealTimeLabelingMonitor()
    
    return monitor.monitor_labeling_performance(labeled_data, market_data)