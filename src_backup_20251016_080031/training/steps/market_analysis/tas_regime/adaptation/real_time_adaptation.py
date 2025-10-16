"""
Real-Time Adaptation for Tree Architecture Search

Advanced real-time adaptation capabilities for tree-based models including
performance monitoring, dynamic optimization, and adaptive search strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
import psutil
import threading
from datetime import datetime, timedelta
from collections import defaultdict, deque
import warnings

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# DecisionTreeClassifier removed - only advanced tree models supported
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from ..core.tas_config import TASConfig, TreeModelType
from ..core.tree_architecture import TreeArchitectureCandidate
from ..core.tas_result import TASResult

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    gpu_usage: float = 0.0
    training_time: float = 0.0
    inference_time: float = 0.0
    model_accuracy: float = 0.0
    model_complexity: float = 0.0
    adaptation_score: float = 0.0


@dataclass
class SystemHealth:
    """System health status."""
    overall_health: str = "good"
    cpu_health: str = "good"
    memory_health: str = "good"
    storage_health: str = "good"
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class TreePerformanceMonitor:
    """
    Advanced Performance Monitor for Tree Architecture Search.

    Monitors system resources, model performance, and provides optimization
    recommendations for real-time adaptation.
    """

    def __init__(self, config: TASConfig):
        """Initialize performance monitor.

        Args:
            config: TAS configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Performance tracking
        self.metrics_history: deque = deque(maxlen=1000)
        self.performance_thresholds = self._initialize_thresholds()

        # System monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 1.0  # seconds

        # Performance analysis
        self.performance_trends = defaultdict(list)
        self.anomaly_detection_enabled = True

        # Optimization recommendations
        self.optimization_queue = deque(maxlen=50)
        self.last_optimization_time = datetime.now()

        self.logger.info("✅ Tree Performance Monitor initialized")
        self.logger.info(f"📊 Monitoring interval: {self.monitoring_interval}s")
        self.logger.info(f"🔍 Anomaly detection: {self.anomaly_detection_enabled}")

    def start_monitoring(self):
        """Start real-time performance monitoring."""
        if self.monitoring_active:
            self.logger.warning("⚠️ Monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info("🚀 Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        self.logger.info("🛑 Performance monitoring stopped")

    def record_metrics(self,
                      model: Any,
                      training_time: float,
                      inference_time: float,
                      accuracy: float,
                      complexity: float) -> PerformanceMetrics:
        """Record performance metrics for a model.

        Args:
            model: The trained model
            training_time: Training time in seconds
            inference_time: Inference time in seconds
            accuracy: Model accuracy score
            complexity: Model complexity score

        Returns:
            PerformanceMetrics object
        """
        try:
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory().percent

            # GPU usage (placeholder - would need GPU monitoring library)
            gpu_usage = 0.0

            # Create metrics
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                training_time=training_time,
                inference_time=inference_time,
                model_accuracy=accuracy,
                model_complexity=complexity,
                adaptation_score=self._calculate_adaptation_score(accuracy, training_time)
            )

            # Add to history
            self.metrics_history.append(metrics)

            # Update trends
            self._update_performance_trends(metrics)

            # Check for anomalies
            if self.anomaly_detection_enabled:
                self._detect_anomalies(metrics)

            self.logger.debug(f"📊 Metrics recorded: CPU={cpu_usage:.1f}%, "
                            f"Memory={memory_usage:.1f}%, Accuracy={accuracy:.4f}")

            return metrics

        except Exception as e:
            self.logger.error(f"❌ Metrics recording failed: {e}")
            # Return default metrics
            return PerformanceMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                model_accuracy=accuracy,
                model_complexity=complexity
            )

    def get_system_health(self) -> SystemHealth:
        """Get current system health status.

        Returns:
            SystemHealth object with health assessment
        """
        try:
            # Get current system state
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            health = SystemHealth()
            warnings = []
            recommendations = []

            # CPU health
            if cpu_percent > 90:
                health.cpu_health = "critical"
                warnings.append("High CPU usage")
                recommendations.append("Consider reducing model complexity")
            elif cpu_percent > 70:
                health.cpu_health = "warning"
                warnings.append("Elevated CPU usage")

            # Memory health
            if memory.percent > 90:
                health.memory_health = "critical"
                warnings.append("High memory usage")
                recommendations.append("Consider memory optimization")
            elif memory.percent > 70:
                health.memory_health = "warning"
                warnings.append("Elevated memory usage")

            # Storage health
            if disk.percent > 90:
                health.storage_health = "critical"
                warnings.append("Low disk space")
                recommendations.append("Free up disk space")
            elif disk.percent > 80:
                health.storage_health = "warning"
                warnings.append("Low disk space")

            # Overall health
            if any(h == "critical" for h in [health.cpu_health, health.memory_health, health.storage_health]):
                health.overall_health = "critical"
            elif any(h == "warning" for h in [health.cpu_health, health.memory_health, health.storage_health]):
                health.overall_health = "warning"

            health.warnings = warnings
            health.recommendations = recommendations

            return health

        except Exception as e:
            self.logger.error(f"❌ System health check failed: {e}")
            return SystemHealth(overall_health="unknown")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics.

        Returns:
            Dictionary with performance summary
        """
        if not self.metrics_history:
            return {}

        try:
            # Extract metrics
            cpu_usage = [m.cpu_usage for m in self.metrics_history]
            memory_usage = [m.memory_usage for m in self.metrics_history]
            accuracy_scores = [m.model_accuracy for m in self.metrics_history]
            training_times = [m.training_time for m in self.metrics_history]

            # Calculate statistics
            summary = {
                'total_metrics': len(self.metrics_history),
                'time_range': {
                    'start': self.metrics_history[0].timestamp if self.metrics_history else None,
                    'end': self.metrics_history[-1].timestamp if self.metrics_history else None
                },
                'cpu_usage': {
                    'mean': np.mean(cpu_usage),
                    'std': np.std(cpu_usage),
                    'min': np.min(cpu_usage),
                    'max': np.max(cpu_usage)
                },
                'memory_usage': {
                    'mean': np.mean(memory_usage),
                    'std': np.std(memory_usage),
                    'min': np.min(memory_usage),
                    'max': np.max(memory_usage)
                },
                'model_performance': {
                    'accuracy_mean': np.mean(accuracy_scores),
                    'accuracy_std': np.std(accuracy_scores),
                    'training_time_mean': np.mean(training_times),
                    'training_time_std': np.std(training_times)
                },
                'trends': dict(self.performance_trends),
                'system_health': self.get_system_health().__dict__
            }

            return summary

        except Exception as e:
            self.logger.error(f"❌ Performance summary failed: {e}")
            return {}

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Get system metrics
                cpu_usage = psutil.cpu_percent(interval=0.1)
                memory_usage = psutil.virtual_memory().percent

                # Create basic metrics
                metrics = PerformanceMetrics(
                    timestamp=datetime.now(),
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage
                )

                # Add to history
                self.metrics_history.append(metrics)

                # Check for optimization opportunities
                self._check_optimization_opportunities()

                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"❌ Monitoring loop error: {e}")
                time.sleep(1.0)

    def _initialize_thresholds(self) -> Dict[str, float]:
        """Initialize performance thresholds."""
        return {
            'cpu_warning': 70.0,
            'cpu_critical': 90.0,
            'memory_warning': 70.0,
            'memory_critical': 90.0,
            'accuracy_target': 0.8,
            'training_time_max': 300.0,  # 5 minutes
            'adaptation_score_min': 0.7
        }

    def _update_performance_trends(self, metrics: PerformanceMetrics):
        """Update performance trends."""
        try:
            # Update trend data
            self.performance_trends['cpu_usage'].append(metrics.cpu_usage)
            self.performance_trends['memory_usage'].append(metrics.memory_usage)
            self.performance_trends['model_accuracy'].append(metrics.model_accuracy)
            self.performance_trends['training_time'].append(metrics.training_time)

            # Keep only recent data (last 100 points)
            for key in self.performance_trends:
                if len(self.performance_trends[key]) > 100:
                    self.performance_trends[key] = self.performance_trends[key][-100:]

        except Exception as e:
            self.logger.warning(f"⚠️ Trend update failed: {e}")

    def _detect_anomalies(self, metrics: PerformanceMetrics):
        """Detect performance anomalies."""
        try:
            if len(self.metrics_history) < 10:
                return

            # Get recent metrics
            recent_metrics = list(self.metrics_history)[-10:]

            # Check for sudden changes
            recent_cpu = [m.cpu_usage for m in recent_metrics]
            recent_memory = [m.memory_usage for m in recent_metrics]
            recent_accuracy = [m.model_accuracy for m in recent_metrics]

            # Calculate thresholds
            cpu_threshold = np.mean(recent_cpu) + 2 * np.std(recent_cpu)
            memory_threshold = np.mean(recent_memory) + 2 * np.std(recent_memory)
            accuracy_threshold = np.mean(recent_accuracy) - 2 * np.std(recent_accuracy)

            # Check for anomalies
            anomalies = []

            if metrics.cpu_usage > cpu_threshold:
                anomalies.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")

            if metrics.memory_usage > memory_threshold:
                anomalies.append(f"High memory usage: {metrics.memory_usage:.1f}%")

            if metrics.model_accuracy < accuracy_threshold and metrics.model_accuracy > 0:
                anomalies.append(f"Low accuracy: {metrics.model_accuracy:.4f}")

            # Log anomalies
            if anomalies:
                self.logger.warning(f"⚠️ Performance anomalies detected: {anomalies}")

        except Exception as e:
            self.logger.warning(f"⚠️ Anomaly detection failed: {e}")

    def _calculate_adaptation_score(self, accuracy: float, training_time: float) -> float:
        """Calculate adaptation score based on performance."""
        try:
            # Normalize accuracy (0-1 scale)
            normalized_accuracy = min(max(accuracy, 0.0), 1.0)

            # Normalize training time (inverse - faster is better)
            if training_time <= 0:
                normalized_time = 1.0
            else:
                normalized_time = min(1.0, 300.0 / training_time)  # 300s as baseline

            # Weighted score
            adaptation_score = 0.7 * normalized_accuracy + 0.3 * normalized_time
            return min(max(adaptation_score, 0.0), 1.0)

        except Exception:
            return 0.5

    def _check_optimization_opportunities(self):
        """Check for optimization opportunities."""
        try:
            if len(self.metrics_history) < 20:
                return

            # Check if optimization is needed
            recent_metrics = list(self.metrics_history)[-10:]
            avg_cpu = np.mean([m.cpu_usage for m in recent_metrics])
            avg_memory = np.mean([m.memory_usage for m in recent_metrics])
            avg_accuracy = np.mean([m.model_accuracy for m in recent_metrics])

            # Check thresholds
            if (avg_cpu > self.performance_thresholds['cpu_warning'] or
                avg_memory > self.performance_thresholds['memory_warning'] or
                avg_accuracy < self.performance_thresholds['accuracy_target']):

                optimization = {
                    'timestamp': datetime.now(),
                    'reason': 'performance_degradation',
                    'recommendations': self._generate_optimization_recommendations(avg_cpu, avg_memory, avg_accuracy),
                    'metrics': {
                        'cpu_usage': avg_cpu,
                        'memory_usage': avg_memory,
                        'accuracy': avg_accuracy
                    }
                }

                self.optimization_queue.append(optimization)
                self.logger.info(f"🔧 Optimization opportunity detected: {optimization['reason']}")

        except Exception as e:
            self.logger.warning(f"⚠️ Optimization check failed: {e}")

    def _generate_optimization_recommendations(self,
                                             cpu_usage: float,
                                             memory_usage: float,
                                             accuracy: float) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        if cpu_usage > self.performance_thresholds['cpu_critical']:
            recommendations.append("Reduce model complexity or use sampling")
        elif cpu_usage > self.performance_thresholds['cpu_warning']:
            recommendations.append("Consider parallel processing")

        if memory_usage > self.performance_thresholds['memory_critical']:
            recommendations.append("Reduce batch size or use memory-efficient algorithms")
        elif memory_usage > self.performance_thresholds['memory_warning']:
            recommendations.append("Consider memory optimization techniques")

        if accuracy < self.performance_thresholds['accuracy_target']:
            recommendations.append("Consider model architecture improvements")

        return recommendations

    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get optimization suggestions."""
        return list(self.optimization_queue)


class TreeRealTimeAdapter:
    """
    Real-Time Adaptation System for Tree Architecture Search.

    Adapts tree architectures in real-time based on performance monitoring
    and changing data characteristics.
    """

    def __init__(self, config: TASConfig):
        """Initialize real-time adapter.

        Args:
            config: TAS configuration
        """
        tprint_info("🔄 Initializing Real-Time Adaptation System")
        tprint_debug(f"Configuration: {config}")
        tprint_debug(f"Adaptation enabled: {config.enable_real_time_adaptation}")
        tprint_debug(f"Adaptation threshold: {config.adaptation_threshold}")
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize performance tracking
        self.performance_metrics = {
            'initialization_time': 0.0,
            'adaptation_time': 0.0,
            'monitoring_time': 0.0,
            'total_execution_time': 0.0
        }

        # Adaptation state
        self.current_architecture = None
        self.adaptation_history = []
        self.performance_monitor = TreePerformanceMonitor(config)

        # Adaptation parameters
        self.adaptation_threshold = 0.1
        self.min_adaptation_interval = timedelta(minutes=5)
        self.max_adaptation_attempts = 10

        # Real-time learning
        self.online_learning_enabled = True
        self.incremental_training_enabled = True

        self.logger.info("✅ Tree Real-Time Adapter initialized")
        self.logger.info(f"🔄 Adaptation threshold: {self.adaptation_threshold}")
        self.logger.info(f"⏱️ Min adaptation interval: {self.min_adaptation_interval}")

    def adapt_architecture(self,
                          current_architecture: TreeArchitectureCandidate,
                          new_data: Tuple[np.ndarray, np.ndarray],
                          adaptation_method: str = "incremental") -> TreeArchitectureCandidate:
        """Adapt architecture to new data.

        Args:
            current_architecture: Current best architecture
            new_data: New data for adaptation
            adaptation_method: Adaptation method ("incremental", "retrain", "evolutionary")

        Returns:
            Adapted architecture
        """
        self.logger.info(f"🔄 Adapting architecture using {adaptation_method}")

        try:
            # Check if adaptation is needed
            if not self._should_adapt(current_architecture, new_data):
                self.logger.info("📊 No adaptation needed")
                return current_architecture

            # Perform adaptation based on method
            if adaptation_method == "incremental" and self.incremental_training_enabled:
                adapted_architecture = self._incremental_adaptation(current_architecture, new_data)
            elif adaptation_method == "retrain":
                adapted_architecture = self._retrain_adaptation(current_architecture, new_data)
            elif adaptation_method == "evolutionary":
                adapted_architecture = self._evolutionary_adaptation(current_architecture, new_data)
            else:
                self.logger.warning(f"⚠️ Unknown adaptation method: {adaptation_method}")
                adapted_architecture = current_architecture

            # Record adaptation
            self.adaptation_history.append({
                'timestamp': datetime.now(),
                'method': adaptation_method,
                'original_architecture': current_architecture,
                'adapted_architecture': adapted_architecture,
                'data_size': len(new_data[0])
            })

            # Limit adaptation history
            if len(self.adaptation_history) > 100:
                self.adaptation_history = self.adaptation_history[-100:]

            self.logger.info("✅ Architecture adaptation completed")
            return adapted_architecture

        except Exception as e:
            self.logger.error(f"❌ Architecture adaptation failed: {e}")
            return current_architecture

    def real_time_search(self,
                        train_data: Tuple[np.ndarray, np.ndarray],
                        validation_data: Tuple[np.ndarray, np.ndarray],
                        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> TASResult:
        """Perform real-time search with continuous adaptation.

        Args:
            train_data: Training data
            validation_data: Validation data
            test_data: Optional test data

        Returns:
            TAS search result
        """
        self.logger.info("🔄 Starting real-time search")

        try:
            # Initialize with basic architecture
            current_architecture = TreeArchitectureCandidate(
                model_type=TreeModelType.RANDOM_FOREST,
                n_trees=100,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1
            )

            # Start performance monitoring
            self.performance_monitor.start_monitoring()

            # Real-time learning loop
            best_result = None
            search_iterations = 0

            while search_iterations < 50:  # Limit iterations
                # Train current architecture
                model = self._train_architecture(current_architecture, train_data)
                accuracy = self._evaluate_architecture(model, validation_data)

                # Record performance
                training_time = 1.0  # Placeholder
                inference_time = 0.5  # Placeholder
                complexity = len(current_architecture.to_dict())

                self.performance_monitor.record_metrics(
                    model, training_time, inference_time, accuracy, complexity
                )

                # Check if we have a new best
                if best_result is None or accuracy > best_result.best_score:
                    best_result = TASResult(
                        best_architecture=current_architecture,
                        best_score=accuracy,
                        search_history=[],
                        execution_time=0.0,
                        success=True
                    )

                # Adapt if needed
                if self._should_adapt_for_real_time(current_architecture, accuracy):
                    current_architecture = self.adapt_architecture(
                        current_architecture, train_data, "incremental"
                    )

                search_iterations += 1

                # Small delay to simulate real-time operation
                time.sleep(0.1)

            # Stop monitoring
            self.performance_monitor.stop_monitoring()

            if best_result:
                best_result.execution_time = search_iterations * 0.1
                self.logger.info(f"✅ Real-time search completed with best score: {best_result.best_score:.4f}")

            return best_result or TASResult(success=False, error_message="No valid result")

        except Exception as e:
            self.logger.error(f"❌ Real-time search failed: {e}")
            self.performance_monitor.stop_monitoring()
            return TASResult(success=False, error_message=str(e))

    def _should_adapt(self,
                     architecture: TreeArchitectureCandidate,
                     new_data: Tuple[np.ndarray, np.ndarray]) -> bool:
        """Check if adaptation is needed."""
        try:
            # Check time since last adaptation
            if self.adaptation_history:
                last_adaptation = self.adaptation_history[-1]['timestamp']
                time_since_adaptation = datetime.now() - last_adaptation

                if time_since_adaptation < self.min_adaptation_interval:
                    return False

            # Check data size
            if len(new_data[0]) < 100:  # Need sufficient data
                return False

            # Check performance degradation
            if len(self.adaptation_history) >= 3:
                recent_scores = [a['adapted_architecture'].overall_score
                               for a in self.adaptation_history[-3:]
                               if hasattr(a['adapted_architecture'], 'overall_score')]

                if recent_scores and np.mean(recent_scores) < 0.7:
                    return True

            return False

        except Exception:
            return False

    def _should_adapt_for_real_time(self,
                                   architecture: TreeArchitectureCandidate,
                                   accuracy: float) -> bool:
        """Check if adaptation is needed for real-time search."""
        try:
            # Adapt if accuracy is low
            if accuracy < 0.8:
                return True

            # Adapt if performance has degraded
            if len(self.adaptation_history) >= 5:
                recent_accuracies = []
                for adaptation in self.adaptation_history[-5:]:
                    arch = adaptation['adapted_architecture']
                    if hasattr(arch, 'overall_score') and arch.overall_score > 0:
                        recent_accuracies.append(arch.overall_score)

                if recent_accuracies and np.mean(recent_accuracies) < accuracy - 0.05:
                    return True

            return False

        except Exception:
            return False

    def _incremental_adaptation(self,
                              architecture: TreeArchitectureCandidate,
                              new_data: Tuple[np.ndarray, np.ndarray]) -> TreeArchitectureCandidate:
        """Perform incremental adaptation."""
        try:
            # Create a modified version of the architecture
            adapted_architecture = TreeArchitectureCandidate(
                model_type=architecture.model_type,
                n_trees=max(50, architecture.n_trees - 10),  # Reduce complexity
                max_depth=min(15, architecture.max_depth + 1),  # Increase depth
                min_samples_split=architecture.min_samples_split,
                min_samples_leaf=architecture.min_samples_leaf,
                max_features=architecture.max_features
            )

            # Train on new data and evaluate
            model = self._train_architecture(adapted_architecture, new_data)
            accuracy = self._evaluate_architecture(model, new_data)

            adapted_architecture.overall_score = accuracy

            self.logger.info(f"✅ Incremental adaptation completed with score: {accuracy:.4f}")
            return adapted_architecture

        except Exception as e:
            self.logger.error(f"❌ Incremental adaptation failed: {e}")
            return architecture

    def _retrain_adaptation(self,
                          architecture: TreeArchitectureCandidate,
                          new_data: Tuple[np.ndarray, np.ndarray]) -> TreeArchitectureCandidate:
        """Perform retrain adaptation."""
        try:
            # Use original parameters but retrain
            adapted_architecture = architecture

            # Retrain on new data
            model = self._train_architecture(adapted_architecture, new_data)
            accuracy = self._evaluate_architecture(model, new_data)

            adapted_architecture.overall_score = accuracy

            self.logger.info(f"✅ Retrain adaptation completed with score: {accuracy:.4f}")
            return adapted_architecture

        except Exception as e:
            self.logger.error(f"❌ Retrain adaptation failed: {e}")
            return architecture

    def _evolutionary_adaptation(self,
                               architecture: TreeArchitectureCandidate,
                               new_data: Tuple[np.ndarray, np.ndarray]) -> TreeArchitectureCandidate:
        """Perform evolutionary adaptation."""
        try:
            # Simple evolutionary approach - create multiple variations
            variations = []

            # Original
            variations.append(architecture)

            # Variation 1: Reduce complexity
            var1 = TreeArchitectureCandidate(
                model_type=architecture.model_type,
                n_trees=max(20, architecture.n_trees // 2),
                max_depth=architecture.max_depth,
                min_samples_split=architecture.min_samples_split,
                min_samples_leaf=architecture.min_samples_leaf
            )
            variations.append(var1)

            # Variation 2: Increase complexity
            var2 = TreeArchitectureCandidate(
                model_type=architecture.model_type,
                n_trees=min(500, architecture.n_trees * 2),
                max_depth=min(20, architecture.max_depth + 2),
                min_samples_split=architecture.min_samples_split,
                min_samples_leaf=architecture.min_samples_leaf
            )
            variations.append(var2)

            # Evaluate all variations
            best_architecture = architecture
            best_score = 0.0

            for var in variations:
                model = self._train_architecture(var, new_data)
                score = self._evaluate_architecture(model, new_data)
                var.overall_score = score

                if score > best_score:
                    best_score = score
                    best_architecture = var

            self.logger.info(f"✅ Evolutionary adaptation completed with score: {best_score:.4f}")
            return best_architecture

        except Exception as e:
            self.logger.error(f"❌ Evolutionary adaptation failed: {e}")
            return architecture

    def _train_architecture(self,
                           architecture: TreeArchitectureCandidate,
                           data: Tuple[np.ndarray, np.ndarray]) -> Any:
        """Train architecture on data."""
        try:
            X, y = data

            if architecture.model_type == TreeModelType.RANDOM_FOREST:
                model = RandomForestClassifier(
                    n_estimators=architecture.n_trees,
                    max_depth=architecture.max_depth,
                    min_samples_split=architecture.min_samples_split,
                    min_samples_leaf=architecture.min_samples_leaf,
                    max_features=architecture.max_features,
                    random_state=42
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=architecture.n_trees,
                    max_depth=architecture.max_depth,
                    min_samples_split=architecture.min_samples_split,
                    min_samples_leaf=architecture.min_samples_leaf,
                    max_features=architecture.max_features,
                    random_state=42
                )

            model.fit(X, y)
            return model

        except Exception as e:
            self.logger.error(f"❌ Architecture training failed: {e}")
            raise

    def _evaluate_architecture(self,
                              model: Any,
                              data: Tuple[np.ndarray, np.ndarray]) -> float:
        """Evaluate architecture on data."""
        try:
            X, y = data
            return model.score(X, y)

        except Exception as e:
            self.logger.error(f"❌ Architecture evaluation failed: {e}")
            return 0.0

    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        if not self.adaptation_history:
            return {}

        return {
            'n_adaptations': len(self.adaptation_history),
            'methods_used': list(set([a['method'] for a in self.adaptation_history])),
            'avg_data_size': np.mean([a['data_size'] for a in self.adaptation_history]),
            'recent_adaptations': self.adaptation_history[-5:] if len(self.adaptation_history) > 5 else self.adaptation_history
        }


# Convenience functions
def create_performance_monitor(config: TASConfig) -> TreePerformanceMonitor:
    """Create a performance monitor with default configuration."""
    return TreePerformanceMonitor(config)


def create_real_time_adapter(config: TASConfig) -> TreeRealTimeAdapter:
    """Create a real-time adapter with default configuration."""
    return TreeRealTimeAdapter(config)


def quick_adaptation(current_architecture: TreeArchitectureCandidate,
                    new_data: Tuple[np.ndarray, np.ndarray]) -> TreeArchitectureCandidate:
    """Quick adaptation with default settings."""
    config = TASConfig()
    adapter = TreeRealTimeAdapter(config)
    return adapter.adapt_architecture(current_architecture, new_data)


class TreeAdaptiveSearch:
    """
    Adaptive Search Strategy for Tree Architecture Search.

    Implements adaptive search strategies that evolve based on performance feedback
    and environmental conditions.
    """

    def __init__(self, config: TASConfig):
        """Initialize adaptive search.

        Args:
            config: TAS configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Adaptive search state
        self.search_history = []
        self.performance_feedback = []
        self.adaptation_strategies = {
            'exploration': 0.3,
            'exploitation': 0.7,
            'diversification': 0.2
        }

        # Adaptive parameters
        self.learning_rate = 0.1
        self.adaptation_frequency = 5

        self.logger.info("✅ Tree Adaptive Search initialized")

    def adapt_search_strategy(self,
                             current_performance: float,
                             search_iteration: int) -> Dict[str, float]:
        """Adapt search strategy based on performance.

        Args:
            current_performance: Current model performance
            search_iteration: Current search iteration

        Returns:
            Updated strategy weights
        """
        try:
            # Record performance feedback
            self.performance_feedback.append({
                'iteration': search_iteration,
                'performance': current_performance,
                'timestamp': datetime.now()
            })

            # Adapt strategy if needed
            if len(self.performance_feedback) % self.adaptation_frequency == 0:
                # Analyze recent performance trend
                recent_feedback = self.performance_feedback[-10:] if len(self.performance_feedback) >= 10 else self.performance_feedback

                if recent_feedback:
                    performances = [f['performance'] for f in recent_feedback]
                    avg_performance = np.mean(performances)
                    performance_trend = np.polyfit(range(len(performances)), performances, 1)[0]

                    # Adapt strategy based on trend
                    if performance_trend < -0.01:  # Declining performance
                        # Increase exploration
                        self.adaptation_strategies['exploration'] = min(0.5, self.adaptation_strategies['exploration'] + 0.1)
                        self.adaptation_strategies['exploitation'] = max(0.3, self.adaptation_strategies['exploitation'] - 0.1)

                    elif performance_trend > 0.01:  # Improving performance
                        # Increase exploitation
                        self.adaptation_strategies['exploitation'] = min(0.8, self.adaptation_strategies['exploitation'] + 0.05)
                        self.adaptation_strategies['exploration'] = max(0.1, self.adaptation_strategies['exploration'] - 0.05)

            # Normalize strategies
            total_weight = sum(self.adaptation_strategies.values())
            if total_weight > 0:
                self.adaptation_strategies = {
                    k: v / total_weight for k, v in self.adaptation_strategies.items()
                }

            self.logger.debug(f"🔄 Adapted search strategy: {self.adaptation_strategies}")
            return self.adaptation_strategies.copy()

        except Exception as e:
            self.logger.error(f"❌ Search strategy adaptation failed: {e}")
            return self.adaptation_strategies.copy()

    def get_search_recommendations(self) -> Dict[str, Any]:
        """Get search strategy recommendations."""
        try:
            if not self.performance_feedback:
                return {'recommendations': ['Insufficient data for recommendations']}

            recent_feedback = self.performance_feedback[-20:] if len(self.performance_feedback) >= 20 else self.performance_feedback
            performances = [f['performance'] for f in recent_feedback]

            recommendations = []

            # Performance-based recommendations
            avg_performance = np.mean(performances)
            if avg_performance < 0.7:
                recommendations.append("Consider increasing exploration to find better architectures")
            elif avg_performance > 0.9:
                recommendations.append("Good performance - consider increasing exploitation")

            # Strategy diversity recommendation
            strategy_variance = np.var(list(self.adaptation_strategies.values()))
            if strategy_variance < 0.1:
                recommendations.append("Search strategy may be too focused - consider diversification")

            return {
                'current_strategies': self.adaptation_strategies,
                'avg_performance': avg_performance,
                'performance_trend': np.polyfit(range(len(performances)), performances, 1)[0] if len(performances) > 1 else 0.0,
                'recommendations': recommendations
            }

        except Exception as e:
            self.logger.error(f"❌ Search recommendations failed: {e}")
            return {'error': str(e)}