"""
Performance Tracker

Tracks and monitors model performance across different regimes and time periods
for the NAS-TAS system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Performance metrics to track."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC = "auc"
    CONFIDENCE = "confidence"
    PREDICTION_TIME = "prediction_time"
    THROUGHPUT = "throughput"


class AlertType(Enum):
    """Types of performance alerts."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    HIGH_ERROR_RATE = "high_error_rate"
    LOW_CONFIDENCE = "low_confidence"
    SLOW_PREDICTION = "slow_prediction"
    MODEL_DRIFT = "model_drift"


@dataclass
class PerformanceConfig:
    """Configuration for performance tracking."""
    
    # Tracking settings
    enable_performance_tracking: bool = True
    tracking_frequency: int = 100  # Track every N predictions
    metrics_to_track: List[PerformanceMetric] = field(default_factory=lambda: [
        PerformanceMetric.ACCURACY, PerformanceMetric.F1_SCORE, PerformanceMetric.CONFIDENCE
    ])
    
    # Performance thresholds
    performance_threshold: float = 0.6  # Minimum acceptable performance
    degradation_threshold: float = 0.1  # Performance degradation threshold
    confidence_threshold: float = 0.7  # Minimum confidence threshold
    prediction_time_threshold: float = 1.0  # Maximum prediction time in seconds
    
    # Alerting
    enable_alerts: bool = True
    alert_frequency: int = 50  # Check alerts every N predictions
    alert_cooldown: int = 300  # Seconds between same alerts
    
    # Data retention
    max_history_length: int = 10000  # Maximum number of performance records
    enable_data_compression: bool = True
    compression_ratio: float = 0.1  # Keep 10% of data after compression
    
    # Reporting
    enable_performance_reports: bool = True
    report_frequency: int = 1000  # Generate reports every N predictions
    report_format: str = "json"  # "json", "csv", "html"
    
    # Storage
    performance_data_path: str = "performance_data"
    report_path: str = "performance_reports"
    alert_log_path: str = "performance_alerts.log"
    
    # Advanced features
    enable_drift_detection: bool = True
    drift_detection_window: int = 1000  # Window for drift detection
    enable_anomaly_detection: bool = True
    anomaly_threshold: float = 2.0  # Standard deviations for anomaly detection


@dataclass
class PerformanceRecord:
    """Single performance record."""
    
    # Basic information
    model_id: str
    regime_id: int
    timestamp: datetime
    
    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confidence: float
    
    # System metrics
    prediction_time: float
    throughput: float
    
    # Context
    data_shape: Tuple[int, int]
    feature_names: List[str]
    
    # Metadata
    prediction_count: int
    model_version: str


@dataclass
class PerformanceAlert:
    """Performance alert."""
    
    alert_id: str
    model_id: str
    alert_type: AlertType
    severity: str  # "low", "medium", "high", "critical"
    message: str
    timestamp: datetime
    
    # Alert details
    current_value: float
    threshold_value: float
    trend: str  # "increasing", "decreasing", "stable"
    
    # Resolution
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None


@dataclass
class PerformanceReport:
    """Performance report."""
    
    # Report metadata
    report_id: str
    generated_at: datetime
    time_period: Tuple[datetime, datetime]
    
    # Model performance
    model_performance: Dict[str, Dict[str, float]]
    regime_performance: Dict[int, Dict[str, float]]
    overall_performance: Dict[str, float]
    
    # Trends and insights
    performance_trends: Dict[str, str]  # metric -> trend
    top_performing_models: List[str]
    underperforming_models: List[str]
    
    # Alerts and issues
    active_alerts: List[PerformanceAlert]
    resolved_alerts: List[PerformanceAlert]
    
    # Recommendations
    recommendations: List[str]


class PerformanceTracker:
    """
    Performance tracker for monitoring model performance across regimes.
    
    Tracks performance metrics, detects anomalies, generates alerts,
    and provides comprehensive performance reporting.
    """
    
    def __init__(self, config: PerformanceConfig):
        """Initialize performance tracker.
        
        Args:
            config: Performance tracking configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize storage
        self._initialize_storage()
        
        # Performance data
        self.performance_history = {}  # model_id -> [PerformanceRecord]
        self.regime_performance = {}  # regime_id -> [PerformanceRecord]
        self.model_statistics = {}  # model_id -> statistics
        
        # Alerting system
        self.active_alerts = {}  # model_id -> [PerformanceAlert]
        self.alert_history = []
        self.last_alert_time = {}  # model_id -> last_alert_time
        
        # Drift detection
        self.baseline_performance = {}  # model_id -> baseline_metrics
        self.drift_detectors = {}  # model_id -> drift_detector
        
        # Reporting
        self.report_counter = 0
        self.last_report_time = datetime.now()
        
        self.logger.info("✅ Performance Tracker initialized")
        self.logger.info(f"   Tracking frequency: {config.tracking_frequency}")
        self.logger.info(f"   Metrics tracked: {[m.value for m in config.metrics_to_track]}")
        self.logger.info(f"   Alerts enabled: {config.enable_alerts}")
        self.logger.info(f"   Drift detection: {config.enable_drift_detection}")
    
    def _initialize_storage(self):
        """Initialize storage directories."""
        try:
            Path(self.config.performance_data_path).mkdir(parents=True, exist_ok=True)
            Path(self.config.report_path).mkdir(parents=True, exist_ok=True)
            
            if self.config.enable_alerts:
                Path(self.config.alert_log_path).parent.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("✅ Performance storage initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Storage initialization failed: {e}")
            raise
    
    def setup_model_tracking(self, model_id: str, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Setup performance tracking for a model.
        
        Args:
            model_id: Model identifier
            model_info: Model information
            
        Returns:
            Setup result
        """
        try:
            # Initialize performance history
            self.performance_history[model_id] = []
            
            # Initialize model statistics
            self.model_statistics[model_id] = {
                'total_predictions': 0,
                'average_performance': {},
                'performance_trend': 'stable',
                'last_update': datetime.now()
            }
            
            # Initialize baseline performance
            if self.config.enable_drift_detection:
                self.baseline_performance[model_id] = {
                    'accuracy': model_info.get('val_metrics', {}).get('accuracy', 0.5),
                    'f1_score': model_info.get('val_metrics', {}).get('f1_score', 0.5),
                    'confidence': 0.7
                }
            
            # Initialize drift detector
            if self.config.enable_drift_detection:
                self.drift_detectors[model_id] = self._create_drift_detector()
            
            self.logger.info(f"✅ Performance tracking setup for {model_id}")
            return {'status': 'tracking_enabled', 'model_id': model_id}
            
        except Exception as e:
            self.logger.error(f"❌ Performance tracking setup failed for {model_id}: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _create_drift_detector(self):
        """Create drift detector for a model."""
        # Simple drift detector implementation
        return {
            'baseline_metrics': {},
            'recent_metrics': [],
            'drift_threshold': 0.1,
            'detected_drift': False
        }
    
    def record_performance(self, 
                         model_id: str,
                         regime_id: int,
                         performance_metrics: Dict[str, float],
                         prediction_time: float = 0.0,
                         data_shape: Tuple[int, int] = (0, 0),
                         feature_names: List[str] = None,
                         model_version: str = "1.0.0") -> bool:
        """
        Record performance metrics for a model.
        
        Args:
            model_id: Model identifier
            regime_id: Regime identifier
            performance_metrics: Performance metrics dictionary
            prediction_time: Time taken for prediction
            data_shape: Shape of input data
            feature_names: List of feature names
            model_version: Model version
            
        Returns:
            Success status
        """
        try:
            # Create performance record
            record = PerformanceRecord(
                model_id=model_id,
                regime_id=regime_id,
                timestamp=datetime.now(),
                accuracy=performance_metrics.get('accuracy', 0.0),
                precision=performance_metrics.get('precision', 0.0),
                recall=performance_metrics.get('recall', 0.0),
                f1_score=performance_metrics.get('f1_score', 0.0),
                confidence=performance_metrics.get('confidence', 0.0),
                prediction_time=prediction_time,
                throughput=1.0 / prediction_time if prediction_time > 0 else 0.0,
                data_shape=data_shape,
                feature_names=feature_names or [],
                prediction_count=self.model_statistics.get(model_id, {}).get('total_predictions', 0) + 1,
                model_version=model_version
            )
            
            # Add to performance history
            if model_id not in self.performance_history:
                self.performance_history[model_id] = []
            
            self.performance_history[model_id].append(record)
            
            # Add to regime performance
            if regime_id not in self.regime_performance:
                self.regime_performance[regime_id] = []
            self.regime_performance[regime_id].append(record)
            
            # Update model statistics
            self._update_model_statistics(model_id, record)
            
            # Check for alerts
            if self.config.enable_alerts:
                self._check_performance_alerts(model_id, record)
            
            # Check for drift
            if self.config.enable_drift_detection:
                self._check_model_drift(model_id, record)
            
            # Compress data if needed
            if len(self.performance_history[model_id]) > self.config.max_history_length:
                self._compress_performance_data(model_id)
            
            # Generate report if needed
            if self._should_generate_report():
                self._generate_performance_report()
            
            self.logger.debug(f"📊 Recorded performance for {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Performance recording failed for {model_id}: {e}")
            return False
    
    def _update_model_statistics(self, model_id: str, record: PerformanceRecord):
        """Update model statistics."""
        try:
            if model_id not in self.model_statistics:
                self.model_statistics[model_id] = {
                    'total_predictions': 0,
                    'average_performance': {},
                    'performance_trend': 'stable',
                    'last_update': datetime.now()
                }
            
            stats = self.model_statistics[model_id]
            stats['total_predictions'] += 1
            stats['last_update'] = datetime.now()
            
            # Update average performance
            if 'average_performance' not in stats:
                stats['average_performance'] = {}
            
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'confidence']
            for metric in metrics:
                current_avg = stats['average_performance'].get(metric, 0.0)
                new_value = getattr(record, metric, 0.0)
                n = stats['total_predictions']
                
                # Update running average
                stats['average_performance'][metric] = (current_avg * (n - 1) + new_value) / n
            
            # Update performance trend
            if len(self.performance_history[model_id]) >= 10:
                recent_f1 = [r.f1_score for r in self.performance_history[model_id][-10:]]
                if len(recent_f1) >= 2:
                    trend = 'improving' if recent_f1[-1] > recent_f1[0] else 'declining'
                    stats['performance_trend'] = trend
            
        except Exception as e:
            self.logger.error(f"❌ Statistics update failed for {model_id}: {e}")
    
    def _check_performance_alerts(self, model_id: str, record: PerformanceRecord):
        """Check for performance alerts."""
        try:
            # Check if enough time has passed since last alert
            if model_id in self.last_alert_time:
                time_since_last = (datetime.now() - self.last_alert_time[model_id]).total_seconds()
                if time_since_last < self.config.alert_cooldown:
                    return
            
            alerts_generated = []
            
            # Check performance degradation
            if record.f1_score < self.config.performance_threshold:
                alert = PerformanceAlert(
                    alert_id=f"{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    model_id=model_id,
                    alert_type=AlertType.PERFORMANCE_DEGRADATION,
                    severity='high',
                    message=f"Performance below threshold: {record.f1_score:.3f} < {self.config.performance_threshold}",
                    timestamp=datetime.now(),
                    current_value=record.f1_score,
                    threshold_value=self.config.performance_threshold,
                    trend='declining'
                )
                alerts_generated.append(alert)
            
            # Check low confidence
            if record.confidence < self.config.confidence_threshold:
                alert = PerformanceAlert(
                    alert_id=f"{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_confidence",
                    model_id=model_id,
                    alert_type=AlertType.LOW_CONFIDENCE,
                    severity='medium',
                    message=f"Low confidence: {record.confidence:.3f} < {self.config.confidence_threshold}",
                    timestamp=datetime.now(),
                    current_value=record.confidence,
                    threshold_value=self.config.confidence_threshold,
                    trend='declining'
                )
                alerts_generated.append(alert)
            
            # Check slow prediction
            if record.prediction_time > self.config.prediction_time_threshold:
                alert = PerformanceAlert(
                    alert_id=f"{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_time",
                    model_id=model_id,
                    alert_type=AlertType.SLOW_PREDICTION,
                    severity='low',
                    message=f"Slow prediction: {record.prediction_time:.3f}s > {self.config.prediction_time_threshold}s",
                    timestamp=datetime.now(),
                    current_value=record.prediction_time,
                    threshold_value=self.config.prediction_time_threshold,
                    trend='declining'
                )
                alerts_generated.append(alert)
            
            # Add alerts to system
            for alert in alerts_generated:
                if model_id not in self.active_alerts:
                    self.active_alerts[model_id] = []
                self.active_alerts[model_id].append(alert)
                self.alert_history.append(alert)
                self.last_alert_time[model_id] = datetime.now()
                
                self.logger.warning(f"🚨 Alert generated for {model_id}: {alert.message}")
            
        except Exception as e:
            self.logger.error(f"❌ Alert checking failed for {model_id}: {e}")
    
    def _check_model_drift(self, model_id: str, record: PerformanceRecord):
        """Check for model drift."""
        try:
            if model_id not in self.drift_detectors:
                return
            
            drift_detector = self.drift_detectors[model_id]
            
            # Update recent metrics
            drift_detector['recent_metrics'].append({
                'f1_score': record.f1_score,
                'accuracy': record.accuracy,
                'confidence': record.confidence,
                'timestamp': record.timestamp
            })
            
            # Keep only recent metrics
            max_recent = 100
            if len(drift_detector['recent_metrics']) > max_recent:
                drift_detector['recent_metrics'] = drift_detector['recent_metrics'][-max_recent:]
            
            # Check for drift if we have enough data
            if len(drift_detector['recent_metrics']) >= 20:
                baseline = self.baseline_performance.get(model_id, {})
                recent_f1 = [m['f1_score'] for m in drift_detector['recent_metrics'][-20:]]
                baseline_f1 = baseline.get('f1_score', 0.5)
                
                # Simple drift detection: significant drop in performance
                avg_recent_f1 = np.mean(recent_f1)
                drift_magnitude = baseline_f1 - avg_recent_f1
                
                if drift_magnitude > drift_detector['drift_threshold']:
                    if not drift_detector['detected_drift']:
                        drift_detector['detected_drift'] = True
                        
                        # Generate drift alert
                        alert = PerformanceAlert(
                            alert_id=f"{model_id}_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            model_id=model_id,
                            alert_type=AlertType.MODEL_DRIFT,
                            severity='critical',
                            message=f"Model drift detected: {drift_magnitude:.3f} drop in F1 score",
                            timestamp=datetime.now(),
                            current_value=avg_recent_f1,
                            threshold_value=baseline_f1,
                            trend='declining'
                        )
                        
                        if model_id not in self.active_alerts:
                            self.active_alerts[model_id] = []
                        self.active_alerts[model_id].append(alert)
                        self.alert_history.append(alert)
                        
                        self.logger.warning(f"🚨 Model drift detected for {model_id}: {drift_magnitude:.3f}")
            
        except Exception as e:
            self.logger.error(f"❌ Drift detection failed for {model_id}: {e}")
    
    def _compress_performance_data(self, model_id: str):
        """Compress performance data to save space."""
        try:
            if model_id not in self.performance_history:
                return
            
            history = self.performance_history[model_id]
            if len(history) <= self.config.max_history_length:
                return
            
            # Keep only recent data based on compression ratio
            keep_count = int(len(history) * self.config.compression_ratio)
            self.performance_history[model_id] = history[-keep_count:]
            
            self.logger.info(f"📦 Compressed performance data for {model_id}: {len(history)} -> {keep_count} records")
            
        except Exception as e:
            self.logger.error(f"❌ Data compression failed for {model_id}: {e}")
    
    def _should_generate_report(self) -> bool:
        """Check if performance report should be generated."""
        if not self.config.enable_performance_reports:
            return False
        
        self.report_counter += 1
        return self.report_counter >= self.config.report_frequency
    
    def _generate_performance_report(self):
        """Generate performance report."""
        try:
            self.logger.info("📊 Generating performance report...")
            
            # Create report
            report = PerformanceReport(
                report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                generated_at=datetime.now(),
                time_period=(self.last_report_time, datetime.now()),
                model_performance=self._calculate_model_performance(),
                regime_performance=self._calculate_regime_performance(),
                overall_performance=self._calculate_overall_performance(),
                performance_trends=self._calculate_performance_trends(),
                top_performing_models=self._get_top_performing_models(),
                underperforming_models=self._get_underperforming_models(),
                active_alerts=self._get_active_alerts(),
                resolved_alerts=self._get_resolved_alerts(),
                recommendations=self._generate_recommendations()
            )
            
            # Save report
            self._save_performance_report(report)
            
            # Update counters
            self.report_counter = 0
            self.last_report_time = datetime.now()
            
            self.logger.info(f"✅ Performance report generated: {report.report_id}")
            
        except Exception as e:
            self.logger.error(f"❌ Report generation failed: {e}")
    
    def _calculate_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for each model."""
        model_performance = {}
        
        for model_id, history in self.performance_history.items():
            if not history:
                continue
            
            recent_history = history[-100:]  # Last 100 records
            
            model_performance[model_id] = {
                'accuracy': np.mean([r.accuracy for r in recent_history]),
                'precision': np.mean([r.precision for r in recent_history]),
                'recall': np.mean([r.recall for r in recent_history]),
                'f1_score': np.mean([r.f1_score for r in recent_history]),
                'confidence': np.mean([r.confidence for r in recent_history]),
                'prediction_time': np.mean([r.prediction_time for r in recent_history]),
                'throughput': np.mean([r.throughput for r in recent_history])
            }
        
        return model_performance
    
    def _calculate_regime_performance(self) -> Dict[int, Dict[str, float]]:
        """Calculate performance metrics for each regime."""
        regime_performance = {}
        
        for regime_id, history in self.regime_performance.items():
            if not history:
                continue
            
            recent_history = history[-100:]  # Last 100 records
            
            regime_performance[regime_id] = {
                'accuracy': np.mean([r.accuracy for r in recent_history]),
                'precision': np.mean([r.precision for r in recent_history]),
                'recall': np.mean([r.recall for r in recent_history]),
                'f1_score': np.mean([r.f1_score for r in recent_history]),
                'confidence': np.mean([r.confidence for r in recent_history]),
                'prediction_time': np.mean([r.prediction_time for r in recent_history])
            }
        
        return regime_performance
    
    def _calculate_overall_performance(self) -> Dict[str, float]:
        """Calculate overall performance metrics."""
        all_records = []
        for history in self.performance_history.values():
            all_records.extend(history[-100:])  # Last 100 records per model
        
        if not all_records:
            return {}
        
        return {
            'accuracy': np.mean([r.accuracy for r in all_records]),
            'precision': np.mean([r.precision for r in all_records]),
            'recall': np.mean([r.recall for r in all_records]),
            'f1_score': np.mean([r.f1_score for r in all_records]),
            'confidence': np.mean([r.confidence for r in all_records]),
            'prediction_time': np.mean([r.prediction_time for r in all_records]),
            'throughput': np.mean([r.throughput for r in all_records])
        }
    
    def _calculate_performance_trends(self) -> Dict[str, str]:
        """Calculate performance trends."""
        trends = {}
        
        for model_id, history in self.performance_history.items():
            if len(history) < 20:
                continue
            
            recent_f1 = [r.f1_score for r in history[-20:]]
            older_f1 = [r.f1_score for r in history[-40:-20]] if len(history) >= 40 else recent_f1
            
            recent_avg = np.mean(recent_f1)
            older_avg = np.mean(older_f1)
            
            if recent_avg > older_avg + 0.05:
                trends[model_id] = 'improving'
            elif recent_avg < older_avg - 0.05:
                trends[model_id] = 'declining'
            else:
                trends[model_id] = 'stable'
        
        return trends
    
    def _get_top_performing_models(self) -> List[str]:
        """Get top performing models."""
        model_performance = self._calculate_model_performance()
        
        if not model_performance:
            return []
        
        # Sort by F1 score
        sorted_models = sorted(
            model_performance.items(),
            key=lambda x: x[1].get('f1_score', 0),
            reverse=True
        )
        
        return [model_id for model_id, _ in sorted_models[:5]]  # Top 5
    
    def _get_underperforming_models(self) -> List[str]:
        """Get underperforming models."""
        model_performance = self._calculate_model_performance()
        
        if not model_performance:
            return []
        
        # Find models below threshold
        underperforming = []
        for model_id, metrics in model_performance.items():
            f1_score = metrics.get('f1_score', 0)
            if f1_score < self.config.performance_threshold:
                underperforming.append(model_id)
        
        return underperforming
    
    def _get_active_alerts(self) -> List[PerformanceAlert]:
        """Get active alerts."""
        active_alerts = []
        for model_alerts in self.active_alerts.values():
            active_alerts.extend([alert for alert in model_alerts if not alert.resolved])
        return active_alerts
    
    def _get_resolved_alerts(self) -> List[PerformanceAlert]:
        """Get resolved alerts."""
        return [alert for alert in self.alert_history if alert.resolved]
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Check for underperforming models
        underperforming = self._get_underperforming_models()
        if underperforming:
            recommendations.append(f"Consider retraining or replacing underperforming models: {', '.join(underperforming)}")
        
        # Check for active alerts
        active_alerts = self._get_active_alerts()
        if len(active_alerts) > 5:
            recommendations.append("High number of active alerts - investigate system health")
        
        # Check for performance trends
        trends = self._calculate_performance_trends()
        declining_models = [model_id for model_id, trend in trends.items() if trend == 'declining']
        if declining_models:
            recommendations.append(f"Models showing declining performance: {', '.join(declining_models)}")
        
        return recommendations
    
    def _save_performance_report(self, report: PerformanceReport):
        """Save performance report to file."""
        try:
            report_path = Path(self.config.report_path) / f"{report.report_id}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert report to dictionary
            report_dict = {
                'report_id': report.report_id,
                'generated_at': report.generated_at.isoformat(),
                'time_period': [report.time_period[0].isoformat(), report.time_period[1].isoformat()],
                'model_performance': report.model_performance,
                'regime_performance': report.regime_performance,
                'overall_performance': report.overall_performance,
                'performance_trends': report.performance_trends,
                'top_performing_models': report.top_performing_models,
                'underperforming_models': report.underperforming_models,
                'active_alerts': [
                    {
                        'alert_id': alert.alert_id,
                        'model_id': alert.model_id,
                        'alert_type': alert.alert_type.value,
                        'severity': alert.severity,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat()
                    } for alert in report.active_alerts
                ],
                'recommendations': report.recommendations
            }

            with open(report_path, 'w') as f:
                json.dump(report_dict, f, indent=2)

            self.logger.info(f"💾 Performance report saved: {report_path}")

        except (IOError, OSError, json.JSONEncodeError) as e:
            self.logger.error(f"❌ Could not save performance report: {e}")
        except Exception as e:
            self.logger.error(f"❌ Unexpected error saving performance report: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'total_models': len(self.performance_history),
            'total_records': sum(len(history) for history in self.performance_history.values()),
            'active_alerts': len(self._get_active_alerts()),
            'resolved_alerts': len(self._get_resolved_alerts()),
            'overall_performance': self._calculate_overall_performance(),
            'top_performing_models': self._get_top_performing_models(),
            'underperforming_models': self._get_underperforming_models(),
            'performance_trends': self._calculate_performance_trends()
        }
    
    def get_model_performance(self, model_id: str) -> Dict[str, Any]:
        """Get performance for a specific model."""
        if model_id not in self.performance_history:
            return {'error': f'Model {model_id} not found'}
        
        history = self.performance_history[model_id]
        if not history:
            return {'error': f'No performance data for {model_id}'}
        
        recent_history = history[-100:]  # Last 100 records
        
        return {
            'model_id': model_id,
            'total_records': len(history),
            'recent_performance': {
                'accuracy': np.mean([r.accuracy for r in recent_history]),
                'precision': np.mean([r.precision for r in recent_history]),
                'recall': np.mean([r.recall for r in recent_history]),
                'f1_score': np.mean([r.f1_score for r in recent_history]),
                'confidence': np.mean([r.confidence for r in recent_history]),
                'prediction_time': np.mean([r.prediction_time for r in recent_history])
            },
            'performance_trend': self.model_statistics.get(model_id, {}).get('performance_trend', 'unknown'),
            'active_alerts': len([alert for alert in self.active_alerts.get(model_id, []) if not alert.resolved])
        }