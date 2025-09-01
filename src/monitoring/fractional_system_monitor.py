# src/monitoring/fractional_system_monitor.py

"""Fractional System Monitor: Production monitoring for combined fractional system.
Implements comprehensive monitoring, alerting, and performance tracking.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.error_handler import handle_errors


class FractionalSystemMonitor:
    """Monitor performance of combined fractional system in production."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize fractional system monitor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Monitoring parameters
        self.monitoring_window = self.config.get('monitoring_window', 1000)  # samples
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'feature_quality_min': 0.6,
            'label_quality_min': 0.5,
            'processing_time_max': 5.0,  # seconds
            'error_rate_max': 0.05,
            'regime_quality_min': 0.7
        })

        # Performance tracking
        self.performance_history = []
        self.alert_history = []
        self.regime_performance = {}

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_start_time = None

        # Alert channels
        self.alert_channels = self.config.get('alert_channels', ['log', 'file'])
        self.alert_output_dir = Path(self.config.get('alert_output_dir', 'data/monitoring/alerts'))
        self.alert_output_dir.mkdir(parents=True, exist_ok=True)

        # Performance metrics
        self.metrics = {
            'feature_quality': [],
            'label_quality': [],
            'processing_time': [],
            'error_rate': [],
            'regime_quality': [],
            'hmm_integration_quality': [],
            'overall_synergy': []
        }

        self.logger = get_logger("FractionalSystemMonitor")

        self.logger.info("✅ Fractional System Monitor initialized successfully")

    def start_monitoring(self):
        """Start monitoring the fractional system."""
        self.is_monitoring = True
        self.monitoring_start_time = datetime.now()
        self.logger.info("🚀 Started fractional system monitoring")

    @handle_errors("Fractional system monitoring")
    def _calculate_performance_metrics(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        hmm_regime: Optional[str],
        processing_time: float,
        error_occurred: bool
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics.

        Args:
            features: Features DataFrame
            labels: Labels Series
            hmm_regime: HMM regime label
            processing_time: Processing time
            error_occurred: Whether an error occurred

        Returns:
            Dictionary with performance metrics
        """
        try:
            metrics = {
                'timestamp': datetime.now(),
                'hmm_regime': hmm_regime,
                'processing_time': processing_time,
                'error_occurred': error_occurred,
                'feature_count': len(features.columns) if not features.empty else 0,
                'sample_count': len(features) if not features.empty else 0
            }

            # Feature quality metrics
            if not features.empty:
                feature_qualities = []
                for col in features.columns:
                    feature_series = features[col].dropna()
                    if len(feature_series) > 0:
                        variance = feature_series.var()
                        non_zero_ratio = (feature_series != 0).sum() / len(feature_series)
                        quality_score = min(1.0, variance * 100) * non_zero_ratio
                        feature_qualities.append(quality_score)

                metrics['feature_quality'] = np.mean(feature_qualities) if feature_qualities else 0.0
                metrics['feature_quality_std'] = np.std(feature_qualities) if feature_qualities else 0.0
            else:
                metrics['feature_quality'] = 0.0
                metrics['feature_quality_std'] = 0.0

            # Label quality metrics
            if not labels.empty:
                label_series = labels.dropna()
                if len(label_series) > 0:
                    metrics['label_quality'] = label_series.var()
                    metrics['label_range'] = label_series.max() - label_series.min()
                    metrics['label_mean'] = label_series.mean()

                    # Label distribution
                    positive_labels = (label_series > 0).sum()
                    negative_labels = (label_series < 0).sum()
                    neutral_labels = (label_series == 0).sum()
                    total_labels = len(label_series)

                    metrics['label_distribution'] = {
                        'positive_ratio': positive_labels / total_labels,
                        'negative_ratio': negative_labels / total_labels,
                        'neutral_ratio': neutral_labels / total_labels
                    }
                else:
                    metrics['label_quality'] = 0.0
                    metrics['label_distribution'] = {'positive_ratio': 0.0, 'negative_ratio': 0.0, 'neutral_ratio': 0.0}
            else:
                metrics['label_quality'] = 0.0
                metrics['label_distribution'] = {'positive_ratio': 0.0, 'negative_ratio': 0.0, 'neutral_ratio': 0.0}

            # HMM integration quality
            if hmm_regime and not features.empty:
                regime_features = [col for col in features.columns if col.startswith(f'regime_{hmm_regime}')]
                if regime_features:
                    regime_qualities = []
                    for col in regime_features:
                        feature_series = features[col].dropna()
                        if len(feature_series) > 0:
                            quality_score = min(1.0, feature_series.var() * 100)
                            regime_qualities.append(quality_score)

                    metrics['hmm_integration_quality'] = np.mean(regime_qualities) if regime_qualities else 0.0
                else:
                    metrics['hmm_integration_quality'] = 0.0
            else:
                metrics['hmm_integration_quality'] = 0.0

            # Overall synergy score
            if not features.empty and not labels.empty:
                synergy_score = self._calculate_synergy_score(features, labels)
                metrics['overall_synergy'] = synergy_score
            else:
                metrics['overall_synergy'] = 0.0

            return metrics

        except Exception as e:
            self.logger.warning(f"Error calculating performance metrics: {e}")
            return {
                'timestamp': datetime.now(),
                'hmm_regime': hmm_regime,
                'processing_time': processing_time,
                'error_occurred': error_occurred,
                'feature_quality': 0.0,
                'label_quality': 0.0,
                'hmm_integration_quality': 0.0,
                'overall_synergy': 0.0,
                'error': str(e)
            }

    def _calculate_synergy_score(self, features: pd.DataFrame, labels: pd.Series) -> float:
        """Calculate synergy score between features and labels.

        Args:
            features: Features DataFrame
            labels: Labels Series

        Returns:
            Synergy score (0-1)
        """
        try:
            # Calculate feature-label correlations
            correlations = []
            for col in features.columns:
                if col.startswith(('frac_diff', 'regime_')):
                    feature_series = features[col].dropna()
                    if len(feature_series) > 0 and len(labels) > 0:
                        # Align series
                        min_len = min(len(feature_series), len(labels))
                        feature_aligned = feature_series.iloc[-min_len:]
                        label_aligned = labels.iloc[-min_len:]

                        corr = abs(feature_aligned.corr(label_aligned))
                        if not pd.isna(corr):
                            correlations.append(corr)

            if correlations:
                # Higher average correlation indicates better synergy
                avg_correlation = np.mean(correlations)
                synergy_score = min(1.0, avg_correlation * 2)  # Scale to 0-1
                return synergy_score
            else:
                return 0.5

        except Exception as e:
            self.logger.warning(f"Error calculating synergy score: {e}")
            return 0.5

    def _store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics in monitoring history.

        Args:
            metrics: Performance metrics dictionary
        """
        try:
            # Store in metrics lists
            self.metrics['feature_quality'].append(metrics.get('feature_quality', 0.0))
            self.metrics['label_quality'].append(metrics.get('label_quality', 0.0))
            self.metrics['processing_time'].append(metrics.get('processing_time', 0.0))
            self.metrics['hmm_integration_quality'].append(metrics.get('hmm_integration_quality', 0.0))
            self.metrics['overall_synergy'].append(metrics.get('overall_synergy', 0.0))

            # Calculate error rate
            recent_errors = sum(1 for m in self.metrics['feature_quality'][-self.monitoring_window:]
                              if m == 0.0)
            error_rate = recent_errors / min(len(self.metrics['feature_quality']), self.monitoring_window)
            self.metrics['error_rate'].append(error_rate)

            # Keep only recent metrics
            for key in self.metrics:
                if len(self.metrics[key]) > self.monitoring_window:
                    self.metrics[key] = self.metrics[key][-self.monitoring_window:]

        except Exception as e:
            self.logger.warning(f"Error storing metrics: {e}")

    def _check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for alert conditions.

        Args:
            metrics: Performance metrics

        Returns:
            List of alert dictionaries
        """
        alerts = []

        try:
            # Feature quality alert
            if metrics.get('feature_quality', 0.0) < self.alert_thresholds['feature_quality_min']:
                alerts.append({
                    'type': 'feature_quality_low',
                    'severity': 'warning',
                    'message': f"Feature quality below threshold: {metrics['feature_quality']:.3f} < {self.alert_thresholds['feature_quality_min']}",
                    'metric': 'feature_quality',
                    'value': metrics['feature_quality'],
                    'threshold': self.alert_thresholds['feature_quality_min']
                })

            # Label quality alert
            if metrics.get('label_quality', 0.0) < self.alert_thresholds['label_quality_min']:
                alerts.append({
                    'type': 'label_quality_low',
                    'severity': 'warning',
                    'message': f"Label quality below threshold: {metrics['label_quality']:.3f} < {self.alert_thresholds['label_quality_min']}",
                    'metric': 'label_quality',
                    'value': metrics['label_quality'],
                    'threshold': self.alert_thresholds['label_quality_min']
                })

            # Processing time alert
            if metrics.get('processing_time', 0.0) > self.alert_thresholds['processing_time_max']:
                alerts.append({
                    'type': 'processing_time_high',
                    'severity': 'warning',
                    'message': f"Processing time above threshold: {metrics['processing_time']:.3f}s > {self.alert_thresholds['processing_time_max']}s",
                    'metric': 'processing_time',
                    'value': metrics['processing_time'],
                    'threshold': self.alert_thresholds['processing_time_max']
                })

            # Error rate alert
            if len(self.metrics['error_rate']) > 0:
                current_error_rate = self.metrics['error_rate'][-1]
                if current_error_rate > self.alert_thresholds['error_rate_max']:
                    alerts.append({
                        'type': 'error_rate_high',
                        'severity': 'critical',
                        'message': f"Error rate above threshold: {current_error_rate:.3f} > {self.alert_thresholds['error_rate_max']}",
                        'metric': 'error_rate',
                        'value': current_error_rate,
                        'threshold': self.alert_thresholds['error_rate_max']
                    })

            # HMM integration quality alert
            if metrics.get('hmm_integration_quality', 0.0) < self.alert_thresholds['regime_quality_min']:
                alerts.append({
                    'type': 'hmm_integration_quality_low',
                    'severity': 'warning',
                    'message': f"HMM integration quality below threshold: {metrics['hmm_integration_quality']:.3f} < {self.alert_thresholds['regime_quality_min']}",
                    'metric': 'hmm_integration_quality',
                    'value': metrics['hmm_integration_quality'],
                    'threshold': self.alert_thresholds['regime_quality_min']
                })

            # Error occurrence alert
            if metrics.get('error_occurred', False):
                alerts.append({
                    'type': 'processing_error',
                    'severity': 'critical',
                    'message': "Processing error occurred in fractional system",
                    'metric': 'error_occurred',
                    'value': True,
                    'threshold': False
                })

        except Exception as e:
            self.logger.warning(f"Error checking alerts: {e}")

        return alerts

    def _trigger_alerts(self, alerts: List[Dict[str, Any]], metrics: Dict[str, Any]):
        """Trigger alerts through configured channels.

        Args:
            alerts: List of alert dictionaries
            metrics: Performance metrics
        """
        try:
            for alert in alerts:
                alert['timestamp'] = datetime.now().isoformat()
                alert['hmm_regime'] = metrics.get('hmm_regime')

                # Store alert
                self.alert_history.append(alert)

                # Send to alert channels
                for channel in self.alert_channels:
                    if channel == 'log':
                        self._log_alert(alert)
                    elif channel == 'file':
                        self._file_alert(alert)

                self.logger.warning(f"🚨 Alert triggered: {alert['type']} - {alert['message']}")

        except Exception as e:
            self.logger.error(f"Error triggering alerts: {e}")

    def _log_alert(self, alert: Dict[str, Any]):
        """Log alert to logger.

        Args:
            alert: Alert dictionary
        """
        try:
            severity = alert.get('severity', 'info').upper()
            message = alert.get('message', 'Unknown alert')
            regime = alert.get('hmm_regime', 'unknown')

            if severity == 'CRITICAL':
                self.logger.critical(f"🚨 CRITICAL ALERT [{regime}]: {message}")
            elif severity == 'WARNING':
                self.logger.warning(f"⚠️ WARNING [{regime}]: {message}")
            else:
                self.logger.info(f"ℹ️ INFO [{regime}]: {message}")

        except Exception as e:
            self.logger.error(f"Error logging alert: {e}")

    def _file_alert(self, alert: Dict[str, Any]):
        """Write alert to file.

        Args:
            alert: Alert dictionary
        """
        try:
            alert_file = self.alert_output_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.json"

            # Load existing alerts
            existing_alerts = []
            if alert_file.exists():
                with open(alert_file, 'r') as f:
                    existing_alerts = json.load(f)

            # Add new alert
            existing_alerts.append(alert)

            # Write back to file
            with open(alert_file, 'w') as f:
                json.dump(existing_alerts, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Error writing alert to file: {e}")

    def _update_regime_performance(self, hmm_regime: str, metrics: Dict[str, Any]):
        """Update regime-specific performance tracking.

        Args:
            hmm_regime: HMM regime label
            metrics: Performance metrics
        """
        try:
            if hmm_regime not in self.regime_performance:
                self.regime_performance[hmm_regime] = {
                    'metrics': [],
                    'total_samples': 0,
                    'error_count': 0,
                    'avg_feature_quality': 0.0,
                    'avg_label_quality': 0.0,
                    'avg_processing_time': 0.0
                }

            regime_data = self.regime_performance[hmm_regime]
            regime_data['metrics'].append(metrics)
            regime_data['total_samples'] += metrics.get('sample_count', 0)

            if metrics.get('error_occurred', False):
                regime_data['error_count'] += 1

            # Update averages
            recent_metrics = regime_data['metrics'][-self.monitoring_window:]
            if recent_metrics:
                regime_data['avg_feature_quality'] = np.mean([m.get('feature_quality', 0.0) for m in recent_metrics])
                regime_data['avg_label_quality'] = np.mean([m.get('label_quality', 0.0) for m in recent_metrics])
                regime_data['avg_processing_time'] = np.mean([m.get('processing_time', 0.0) for m in recent_metrics])

            # Keep only recent metrics
            if len(regime_data['metrics']) > self.monitoring_window:
                regime_data['metrics'] = regime_data['metrics'][-self.monitoring_window:]

        except Exception as e:
            self.logger.warning(f"Error updating regime performance: {e}")

    def _store_performance_record(self, metrics: Dict[str, Any]):
        """Store performance record in history.

        Args:
            metrics: Performance metrics
        """
        try:
            self.performance_history.append(metrics)

            # Keep only recent records
            if len(self.performance_history) > self.monitoring_window:
                self.performance_history = self.performance_history[-self.monitoring_window:]

        except Exception as e:
            self.logger.warning(f"Error storing performance record: {e}")


# Configuration helper