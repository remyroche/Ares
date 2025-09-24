"""
Performance Tracking for Tree Architecture Search

Advanced performance tracking and analytics capabilities for tree-based models
including metrics collection, performance analysis, and reporting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from ..core.tas_config import TASConfig, TreeModelType
from ..core.tree_architecture import TreeArchitectureCandidate
from ..core.tas_result import TASResult

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Snapshot of model performance at a specific point in time."""
    timestamp: datetime
    model_type: str
    architecture_params: Dict[str, Any]
    metrics: Dict[str, float]
    system_metrics: Dict[str, float]
    data_characteristics: Dict[str, Any]
    performance_score: float = 0.0


@dataclass
class AnalyticsReport:
    """Analytics report with comprehensive performance analysis."""
    report_id: str
    generation_time: datetime
    time_range: Dict[str, datetime]
    summary_statistics: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    anomaly_detection: Dict[str, Any]
    recommendations: List[str]
    visualizations: Dict[str, Any] = field(default_factory=dict)


class TreePerformanceTracker:
    """
    Advanced Performance Tracker for Tree Architecture Search.

    Tracks and analyzes performance of tree models across multiple dimensions
    including accuracy, efficiency, robustness, and system impact.
    """

    def __init__(self, config: TASConfig):
        """Initialize performance tracker.

        Args:
            config: TAS configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Performance tracking
        self.performance_history: List[PerformanceSnapshot] = []
        self.current_session_id = self._generate_session_id()

        # Tracking configuration
        self.track_system_metrics = True
        self.track_data_characteristics = True
        self.track_predictions = False  # Enable for detailed prediction tracking

        # Analysis parameters
        self.anomaly_threshold = 2.0  # Standard deviations
        self.trend_window = 20
        self.reporting_interval = timedelta(hours=1)

        # Storage
        self.storage_path = "performance_reports"
        self._ensure_storage_path()

        self.logger.info("✅ Tree Performance Tracker initialized")
        self.logger.info(f"📊 Session ID: {self.current_session_id}")
        self.logger.info(f"📈 Anomaly threshold: {self.anomaly_threshold}σ")

    def track_performance(self,
                         model: Any,
                         architecture: TreeArchitectureCandidate,
                         train_data: Tuple[np.ndarray, np.ndarray],
                         test_data: Tuple[np.ndarray, np.ndarray],
                         training_time: float,
                         inference_time: float) -> PerformanceSnapshot:
        """Track comprehensive performance of a model.

        Args:
            model: The trained model
            architecture: Architecture used
            train_data: Training data
            test_data: Test data
            training_time: Training time in seconds
            inference_time: Inference time in seconds

        Returns:
            PerformanceSnapshot object
        """
        self.logger.debug("📊 Tracking model performance")

        try:
            # Get basic metrics
            metrics = self._calculate_model_metrics(model, test_data)

            # Get system metrics
            system_metrics = self._get_system_metrics() if self.track_system_metrics else {}

            # Get data characteristics
            data_characteristics = self._analyze_data_characteristics(train_data, test_data) if self.track_data_characteristics else {}

            # Calculate overall performance score
            performance_score = self._calculate_performance_score(metrics, system_metrics)

            # Create snapshot
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                model_type=architecture.model_type.value,
                architecture_params=architecture.to_dict(),
                metrics=metrics,
                system_metrics=system_metrics,
                data_characteristics=data_characteristics,
                performance_score=performance_score
            )

            # Add to history
            self.performance_history.append(snapshot)

            # Limit history size
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]

            self.logger.debug(f"✅ Performance tracked: Score = {performance_score:.4f}")
            return snapshot

        except Exception as e:
            self.logger.error(f"❌ Performance tracking failed: {e}")
            # Return minimal snapshot
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                model_type=architecture.model_type.value,
                architecture_params={},
                metrics={'error': str(e)},
                system_metrics={},
                data_characteristics={}
            )

    def generate_analytics_report(self,
                                 start_time: Optional[datetime] = None,
                                 end_time: Optional[datetime] = None) -> AnalyticsReport:
        """Generate comprehensive analytics report.

        Args:
            start_time: Start time for analysis period
            end_time: End time for analysis period

        Returns:
            AnalyticsReport object
        """
        self.logger.info("📈 Generating analytics report")

        try:
            # Set default time range
            if start_time is None:
                start_time = datetime.now() - timedelta(days=1)
            if end_time is None:
                end_time = datetime.now()

            # Filter relevant snapshots
            relevant_snapshots = [s for s in self.performance_history
                                if start_time <= s.timestamp <= end_time]

            if not relevant_snapshots:
                self.logger.warning("⚠️ No performance data available for report")
                return AnalyticsReport(
                    report_id=self._generate_report_id(),
                    generation_time=datetime.now(),
                    time_range={'start': start_time, 'end': end_time},
                    summary_statistics={},
                    trend_analysis={},
                    anomaly_detection={},
                    recommendations=["No performance data available"]
                )

            # Generate report components
            summary_stats = self._calculate_summary_statistics(relevant_snapshots)
            trend_analysis = self._analyze_trends(relevant_snapshots)
            anomaly_detection = self._detect_anomalies(relevant_snapshots)
            recommendations = self._generate_recommendations(relevant_snapshots, summary_stats)

            report = AnalyticsReport(
                report_id=self._generate_report_id(),
                generation_time=datetime.now(),
                time_range={'start': start_time, 'end': end_time},
                summary_statistics=summary_stats,
                trend_analysis=trend_analysis,
                anomaly_detection=anomaly_detection,
                recommendations=recommendations
            )

            # Save report
            self._save_report(report)

            self.logger.info(f"✅ Analytics report generated: {report.report_id}")
            return report

        except Exception as e:
            self.logger.error(f"❌ Report generation failed: {e}")
            return AnalyticsReport(
                report_id=self._generate_report_id(),
                generation_time=datetime.now(),
                time_range={'start': start_time, 'end': end_time},
                summary_statistics={},
                trend_analysis={},
                anomaly_detection={},
                recommendations=[f"Report generation failed: {str(e)}"]
            )

    def _calculate_model_metrics(self,
                                model: Any,
                                data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Calculate comprehensive model metrics."""
        try:
            X, y = data

            # Basic metrics
            predictions = model.predict(X)

            metrics = {}

            # Classification metrics
            if hasattr(model, 'predict_proba'):
                # Classification model
                metrics['accuracy'] = accuracy_score(y, predictions)

                if len(np.unique(y)) == 2:  # Binary classification
                    metrics['precision'] = precision_score(y, predictions, average='binary')
                    metrics['recall'] = recall_score(y, predictions, average='binary')
                    metrics['f1'] = f1_score(y, predictions, average='binary')
                else:  # Multi-class classification
                    metrics['precision'] = precision_score(y, predictions, average='weighted')
                    metrics['recall'] = recall_score(y, predictions, average='weighted')
                    metrics['f1'] = f1_score(y, predictions, average='weighted')

                # Probability metrics
                proba = model.predict_proba(X)
                metrics['confidence_mean'] = np.mean(np.max(proba, axis=1))
                metrics['confidence_std'] = np.std(np.max(proba, axis=1))

            else:
                # Regression model
                metrics['mse'] = mean_squared_error(y, predictions)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['mae'] = mean_absolute_error(y, predictions)
                metrics['r2'] = r2_score(y, predictions)

                # Prediction statistics
                metrics['prediction_mean'] = np.mean(predictions)
                metrics['prediction_std'] = np.std(predictions)

            return metrics

        except Exception as e:
            self.logger.warning(f"⚠️ Model metrics calculation failed: {e}")
            return {'error': str(e)}

    def _get_system_metrics(self) -> Dict[str, float]:
        """Get current system performance metrics."""
        try:
            import psutil

            metrics = {}

            # CPU metrics
            metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            metrics['cpu_count'] = psutil.cpu_count()

            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_used_gb'] = memory.used / (1024**3)
            metrics['memory_available_gb'] = memory.available / (1024**3)

            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics['disk_percent'] = disk.percent
            metrics['disk_used_gb'] = disk.used / (1024**3)

            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                if network:
                    metrics['network_bytes_sent'] = network.bytes_sent
                    metrics['network_bytes_recv'] = network.bytes_recv
            except:
                pass

            return metrics

        except ImportError:
            self.logger.warning("⚠️ psutil not available for system metrics")
            return {'note': 'psutil not available'}
        except Exception as e:
            self.logger.warning(f"⚠️ System metrics collection failed: {e}")
            return {'error': str(e)}

    def _analyze_data_characteristics(self,
                                    train_data: Tuple[np.ndarray, np.ndarray],
                                    test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """Analyze characteristics of training and test data."""
        try:
            X_train, y_train = train_data
            X_test, y_test = test_data

            characteristics = {}

            # Dataset sizes
            characteristics['train_samples'] = len(X_train)
            characteristics['test_samples'] = len(X_test)
            characteristics['feature_count'] = X_train.shape[1]

            # Label distribution (train)
            unique_labels, label_counts = np.unique(y_train, return_counts=True)
            characteristics['train_label_distribution'] = dict(zip(unique_labels, label_counts))
            characteristics['train_class_balance'] = label_counts / len(y_train)

            # Label distribution (test)
            unique_labels_test, label_counts_test = np.unique(y_test, return_counts=True)
            characteristics['test_label_distribution'] = dict(zip(unique_labels_test, label_counts_test))
            characteristics['test_class_balance'] = label_counts_test / len(y_test)

            # Feature statistics
            characteristics['feature_means'] = np.mean(X_train, axis=0).tolist()
            characteristics['feature_stds'] = np.std(X_train, axis=0).tolist()
            characteristics['feature_ranges'] = (np.max(X_train, axis=0) - np.min(X_train, axis=0)).tolist()

            # Data quality indicators
            characteristics['train_missing_values'] = np.sum(np.isnan(X_train))
            characteristics['test_missing_values'] = np.sum(np.isnan(X_test))

            return characteristics

        except Exception as e:
            self.logger.warning(f"⚠️ Data characteristics analysis failed: {e}")
            return {'error': str(e)}

    def _calculate_performance_score(self,
                                   metrics: Dict[str, float],
                                   system_metrics: Dict[str, float]) -> float:
        """Calculate overall performance score."""
        try:
            score_components = []

            # Model performance (weight: 0.7)
            if 'accuracy' in metrics:
                score_components.append(('accuracy', metrics['accuracy'], 0.4))
                score_components.append(('f1', metrics.get('f1', 0.0), 0.3))
            elif 'r2' in metrics:
                score_components.append(('r2', metrics['r2'], 0.4))
                score_components.append(('rmse', 1.0 / (1.0 + metrics.get('rmse', 1.0)), 0.3))

            # System efficiency (weight: 0.3)
            if 'cpu_percent' in system_metrics:
                cpu_efficiency = max(0.0, 1.0 - system_metrics['cpu_percent'] / 100.0)
                score_components.append(('cpu_efficiency', cpu_efficiency, 0.15))

            if 'memory_percent' in system_metrics:
                memory_efficiency = max(0.0, 1.0 - system_metrics['memory_percent'] / 100.0)
                score_components.append(('memory_efficiency', memory_efficiency, 0.15))

            # Calculate weighted score
            total_score = 0.0
            total_weight = 0.0

            for name, value, weight in score_components:
                if 0.0 <= value <= 1.0:  # Valid range
                    total_score += value * weight
                    total_weight += weight

            if total_weight > 0:
                return total_score / total_weight
            else:
                return 0.5  # Default score

        except Exception as e:
            self.logger.warning(f"⚠️ Performance score calculation failed: {e}")
            return 0.5

    def _calculate_summary_statistics(self, snapshots: List[PerformanceSnapshot]) -> Dict[str, Any]:
        """Calculate summary statistics from snapshots."""
        try:
            if not snapshots:
                return {}

            # Extract metrics
            accuracies = [s.metrics.get('accuracy', 0.0) for s in snapshots]
            performance_scores = [s.performance_score for s in snapshots]
            system_metrics = [s.system_metrics for s in snapshots]

            # Basic statistics
            summary = {
                'total_snapshots': len(snapshots),
                'time_range': {
                    'start': min(s.timestamp for s in snapshots),
                    'end': max(s.timestamp for s in snapshots)
                },
                'model_performance': {
                    'accuracy_mean': np.mean(accuracies),
                    'accuracy_std': np.std(accuracies),
                    'accuracy_min': np.min(accuracies),
                    'accuracy_max': np.max(accuracies),
                    'performance_score_mean': np.mean(performance_scores),
                    'performance_score_std': np.std(performance_scores)
                }
            }

            # System metrics summary
            if system_metrics:
                cpu_percents = [m.get('cpu_percent', 0.0) for m in system_metrics if m]
                memory_percents = [m.get('memory_percent', 0.0) for m in system_metrics if m]

                if cpu_percents:
                    summary['system_metrics'] = {
                        'cpu_percent_mean': np.mean(cpu_percents),
                        'cpu_percent_std': np.std(cpu_percents),
                        'memory_percent_mean': np.mean(memory_percents) if memory_percents else 0.0,
                        'memory_percent_std': np.std(memory_percents) if memory_percents else 0.0
                    }

            return summary

        except Exception as e:
            self.logger.warning(f"⚠️ Summary statistics calculation failed: {e}")
            return {}

    def _analyze_trends(self, snapshots: List[PerformanceSnapshot]) -> Dict[str, Any]:
        """Analyze performance trends."""
        try:
            if len(snapshots) < 10:
                return {'note': 'Insufficient data for trend analysis'}

            # Sort by timestamp
            sorted_snapshots = sorted(snapshots, key=lambda x: x.timestamp)

            # Extract time series data
            timestamps = [s.timestamp for s in sorted_snapshots]
            accuracies = [s.metrics.get('accuracy', 0.0) for s in sorted_snapshots]
            performance_scores = [s.performance_score for s in sorted_snapshots]

            # Calculate trends
            trends = {}

            # Accuracy trend
            if len(accuracies) > 1:
                accuracy_trend = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
                trends['accuracy_trend'] = 'increasing' if accuracy_trend > 0.001 else 'decreasing' if accuracy_trend < -0.001 else 'stable'

            # Performance score trend
            if len(performance_scores) > 1:
                performance_trend = np.polyfit(range(len(performance_scores)), performance_scores, 1)[0]
                trends['performance_trend'] = 'increasing' if performance_trend > 0.001 else 'decreasing' if performance_trend < -0.001 else 'stable'

            # Volatility analysis
            trends['accuracy_volatility'] = np.std(accuracies)
            trends['performance_volatility'] = np.std(performance_scores)

            return trends

        except Exception as e:
            self.logger.warning(f"⚠️ Trend analysis failed: {e}")
            return {'error': str(e)}

    def _detect_anomalies(self, snapshots: List[PerformanceSnapshot]) -> Dict[str, Any]:
        """Detect performance anomalies."""
        try:
            if len(snapshots) < 10:
                return {'note': 'Insufficient data for anomaly detection'}

            # Extract performance scores
            performance_scores = [s.performance_score for s in snapshots]

            # Calculate statistics
            mean_score = np.mean(performance_scores)
            std_score = np.std(performance_scores)

            if std_score == 0:
                return {'note': 'No variation in performance scores'}

            # Find anomalies
            anomalies = []
            threshold = mean_score - self.anomaly_threshold * std_score  # Lower threshold

            for i, snapshot in enumerate(snapshots):
                if snapshot.performance_score < threshold:
                    anomalies.append({
                        'index': i,
                        'timestamp': snapshot.timestamp,
                        'performance_score': snapshot.performance_score,
                        'deviation': (mean_score - snapshot.performance_score) / std_score
                    })

            return {
                'total_snapshots': len(snapshots),
                'mean_score': mean_score,
                'std_score': std_score,
                'threshold': threshold,
                'anomalies_count': len(anomalies),
                'anomalies': anomalies[:10]  # Limit to 10 anomalies
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Anomaly detection failed: {e}")
            return {'error': str(e)}

    def _generate_recommendations(self,
                                 snapshots: List[PerformanceSnapshot],
                                 summary_stats: Dict[str, Any]) -> List[str]:
        """Generate performance-based recommendations."""
        recommendations = []

        try:
            # Performance-based recommendations
            model_perf = summary_stats.get('model_performance', {})
            system_metrics = summary_stats.get('system_metrics', {})

            # Accuracy recommendations
            accuracy_mean = model_perf.get('accuracy_mean', 0.0)
            if accuracy_mean < 0.7:
                recommendations.append("Consider improving model architecture or feature engineering")
            elif accuracy_mean > 0.95:
                recommendations.append("Model may be overfitting - consider regularization")

            # System recommendations
            cpu_mean = system_metrics.get('cpu_percent_mean', 0.0)
            if cpu_mean > 80:
                recommendations.append("High CPU usage detected - consider optimization or hardware upgrade")

            memory_mean = system_metrics.get('memory_percent_mean', 0.0)
            if memory_mean > 80:
                recommendations.append("High memory usage detected - consider memory optimization")

            # Trend-based recommendations
            trend_analysis = self._analyze_trends(snapshots)
            accuracy_trend = trend_analysis.get('accuracy_trend', 'stable')
            if accuracy_trend == 'decreasing':
                recommendations.append("Performance is declining - investigate recent changes")

            # Anomaly-based recommendations
            anomaly_detection = self._detect_anomalies(snapshots)
            if anomaly_detection.get('anomalies_count', 0) > 0:
                recommendations.append("Performance anomalies detected - review model stability")

            return recommendations

        except Exception as e:
            self.logger.warning(f"⚠️ Recommendation generation failed: {e}")
            return ["Unable to generate recommendations"]

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return f"TAS_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"

    def _generate_report_id(self) -> str:
        """Generate unique report ID."""
        return f"REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"

    def _ensure_storage_path(self):
        """Ensure storage path exists."""
        try:
            if not os.path.exists(self.storage_path):
                os.makedirs(self.storage_path)
        except Exception as e:
            self.logger.warning(f"⚠️ Could not create storage path: {e}")

    def _save_report(self, report: AnalyticsReport):
        """Save analytics report to storage."""
        try:
            filename = f"{self.storage_path}/{report.report_id}.json"
            with open(filename, 'w') as f:
                json.dump({
                    'report_id': report.report_id,
                    'generation_time': report.generation_time.isoformat(),
                    'time_range': {
                        'start': report.time_range['start'].isoformat(),
                        'end': report.time_range['end'].isoformat()
                    },
                    'summary_statistics': report.summary_statistics,
                    'trend_analysis': report.trend_analysis,
                    'anomaly_detection': report.anomaly_detection,
                    'recommendations': report.recommendations
                }, f, indent=2)

            self.logger.debug(f"💾 Report saved: {filename}")

        except Exception as e:
            self.logger.warning(f"⚠️ Report saving failed: {e}")

    def get_performance_history(self,
                               limit: Optional[int] = None) -> List[PerformanceSnapshot]:
        """Get performance history."""
        if limit:
            return self.performance_history[-limit:]
        return self.performance_history


class TreeMetricsCollector:
    """
    Advanced Metrics Collector for Tree Architecture Search.

    Collects and aggregates metrics from multiple sources for comprehensive analysis.
    """

    def __init__(self, config: TASConfig):
        """Initialize metrics collector.

        Args:
            config: TAS configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Collection state
        self.collected_metrics = defaultdict(list)
        self.collection_intervals = {}
        self.aggregation_functions = {}

        # Default aggregation functions
        self._setup_default_aggregations()

        self.logger.info("✅ Tree Metrics Collector initialized")

    def collect_metrics(self,
                       source: str,
                       metrics: Dict[str, Any],
                       timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Collect metrics from a source.

        Args:
            source: Source of metrics (e.g., 'model', 'system', 'data')
            metrics: Dictionary of metric values
            timestamp: Collection timestamp

        Returns:
            Collection summary
        """
        try:
            timestamp = timestamp or datetime.now()

            # Add metadata
            metrics_with_meta = {
                'source': source,
                'timestamp': timestamp,
                'session_id': self._get_session_id(),
                **metrics
            }

            # Store metrics
            for key, value in metrics.items():
                self.collected_metrics[key].append({
                    'value': value,
                    'source': source,
                    'timestamp': timestamp
                })

            # Update collection intervals
            if source not in self.collection_intervals:
                self.collection_intervals[source] = []

            self.collection_intervals[source].append(timestamp)

            self.logger.debug(f"📊 Collected {len(metrics)} metrics from {source}")
            return {'collected': len(metrics), 'source': source}

        except Exception as e:
            self.logger.error(f"❌ Metrics collection failed: {e}")
            return {'error': str(e)}

    def get_aggregated_metrics(self, metric_name: str, aggregation: str = 'mean') -> Dict[str, Any]:
        """Get aggregated metrics for a specific metric.

        Args:
            metric_name: Name of the metric
            aggregation: Aggregation function ('mean', 'sum', 'max', 'min', 'std')

        Returns:
            Aggregated metric data
        """
        try:
            if metric_name not in self.collected_metrics:
                return {'error': f'Metric {metric_name} not found'}

            values = [entry['value'] for entry in self.collected_metrics[metric_name]]

            if not values:
                return {'error': f'No values for metric {metric_name}'}

            # Apply aggregation
            if aggregation == 'mean':
                result = np.mean(values)
            elif aggregation == 'sum':
                result = np.sum(values)
            elif aggregation == 'max':
                result = np.max(values)
            elif aggregation == 'min':
                result = np.min(values)
            elif aggregation == 'std':
                result = np.std(values)
            else:
                result = np.mean(values)  # Default to mean

            return {
                'metric': metric_name,
                'aggregation': aggregation,
                'value': result,
                'count': len(values),
                'values': values
            }

        except Exception as e:
            self.logger.error(f"❌ Metric aggregation failed: {e}")
            return {'error': str(e)}

    def _setup_default_aggregations(self):
        """Setup default aggregation functions."""
        self.aggregation_functions = {
            'accuracy': 'mean',
            'precision': 'mean',
            'recall': 'mean',
            'f1': 'mean',
            'cpu_percent': 'mean',
            'memory_percent': 'mean',
            'training_time': 'mean',
            'inference_time': 'mean'
        }

    def _get_session_id(self) -> str:
        """Get current session ID."""
        return f"METRICS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def clear_metrics(self, source: Optional[str] = None):
        """Clear collected metrics."""
        if source:
            if source in self.collected_metrics:
                del self.collected_metrics[source]
            if source in self.collection_intervals:
                del self.collection_intervals[source]
        else:
            self.collected_metrics.clear()
            self.collection_intervals.clear()

        self.logger.info(f"🧹 Cleared metrics for source: {source or 'all'}")


class TreeAnalytics:
    """
    Advanced Analytics Engine for Tree Architecture Search.

    Provides advanced analytics capabilities including statistical analysis,
    correlation analysis, and predictive modeling of performance.
    """

    def __init__(self, config: TASConfig):
        """Initialize analytics engine.

        Args:
            config: TAS configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Analytics state
        self.correlation_cache = {}
        self.prediction_models = {}
        self.statistical_tests = {}

        self.logger.info("✅ Tree Analytics Engine initialized")

    def calculate_correlations(self, metrics_data: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate correlations between metrics.

        Args:
            metrics_data: Dictionary of metric time series

        Returns:
            Correlation matrix as dictionary
        """
        try:
            metric_names = list(metrics_data.keys())

            if len(metric_names) < 2:
                return {}

            correlations = {}

            for i, name1 in enumerate(metric_names):
                for j, name2 in enumerate(metric_names):
                    if i < j:  # Avoid duplicate calculations
                        key = f"{name1}_{name2}"

                        if key not in self.correlation_cache:
                            values1 = metrics_data[name1]
                            values2 = metrics_data[name2]

                            if len(values1) == len(values2) and len(values1) > 1:
                                correlation = np.corrcoef(values1, values2)[0, 1]
                                self.correlation_cache[key] = correlation
                            else:
                                self.correlation_cache[key] = 0.0

                        correlations[key] = self.correlation_cache[key]

            return correlations

        except Exception as e:
            self.logger.error(f"❌ Correlation calculation failed: {e}")
            return {}

    def predict_performance(self,
                           historical_data: Dict[str, List[float]],
                           target_metric: str,
                           prediction_horizon: int = 5) -> Dict[str, Any]:
        """Predict future performance based on historical data.

        Args:
            historical_data: Historical metric data
            target_metric: Metric to predict
            prediction_horizon: Number of steps to predict

        Returns:
            Prediction results
        """
        try:
            if target_metric not in historical_data:
                return {'error': f'Target metric {target_metric} not found'}

            data = historical_data[target_metric]

            if len(data) < 10:
                return {'error': 'Insufficient data for prediction'}

            # Simple linear trend prediction
            x = np.arange(len(data))
            y = np.array(data)

            # Linear regression
            slope, intercept = np.polyfit(x, y, 1)

            # Generate predictions
            future_x = np.arange(len(data), len(data) + prediction_horizon)
            predictions = slope * future_x + intercept

            # Calculate confidence intervals (simplified)
            y_pred = slope * x + intercept
            mse = np.mean((y - y_pred) ** 2)
            std_error = np.sqrt(mse)

            confidence_interval = 1.96 * std_error  # 95% confidence

            return {
                'target_metric': target_metric,
                'predictions': predictions.tolist(),
                'trend_slope': slope,
                'trend_intercept': intercept,
                'confidence_interval': confidence_interval,
                'mse': mse,
                'prediction_horizon': prediction_horizon
            }

        except Exception as e:
            self.logger.error(f"❌ Performance prediction failed: {e}")
            return {'error': str(e)}


# Convenience functions
def create_performance_tracker(config: TASConfig) -> TreePerformanceTracker:
    """Create a performance tracker with default configuration."""
    return TreePerformanceTracker(config)


def create_metrics_collector(config: TASConfig) -> TreeMetricsCollector:
    """Create a metrics collector with default configuration."""
    return TreeMetricsCollector(config)


def create_analytics_engine(config: TASConfig) -> TreeAnalytics:
    """Create an analytics engine with default configuration."""
    return TreeAnalytics(config)