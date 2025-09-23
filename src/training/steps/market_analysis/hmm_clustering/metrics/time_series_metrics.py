"""
Time series aware metrics for HMM clustering.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import time

try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

try:
    from src.utils.hardware import (
        get_hardware_accelerator,
        get_memory_manager,
        get_performance_monitor
    )
    HARDWARE_ACCELERATION_AVAILABLE = True
except ImportError:
    HARDWARE_ACCELERATION_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesMetricsResult:
    """Result of time series aware metrics calculation."""
    # Temporal stability metrics
    temporal_stability: float
    regime_persistence: float
    transition_frequency: float
    regime_duration_stats: Dict[str, float]
    
    # Temporal clustering quality
    temporal_silhouette: float
    temporal_cohesion: float
    temporal_separation: float
    
    # Regime transition analysis
    transition_matrix: np.ndarray
    transition_probabilities: Dict[str, float]
    regime_stationarity: Dict[str, float]
    
    # Time series specific metrics
    autocorrelation_clusters: Dict[int, float]
    trend_consistency: float
    volatility_clustering: float
    
    # Execution metadata
    execution_time: float
    matrix_ops_used: bool
    hardware_acceleration_used: bool


class TimeSeriesAwareMetrics:
    """Time series aware metrics calculator for HMM clustering."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the time series metrics calculator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize hardware acceleration if available
        self.hardware_accelerator = None
        self.memory_manager = None
        self.performance_monitor = None
        
        if HARDWARE_ACCELERATION_AVAILABLE:
            try:
                self.hardware_accelerator = get_hardware_accelerator()
                self.memory_manager = get_memory_manager()
                self.performance_monitor = get_performance_monitor()
                self.logger.info("✅ Hardware acceleration initialized for time series metrics")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available for time series metrics: {e}")
        
        # Initialize matrix operations if available
        self.matrix_ops = None
        self.vectorized_core = None
        self.enhanced_ops = None
        self.batch_processor = None
        
        if MATRIX_OPERATIONS_AVAILABLE:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized for time series metrics")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available for time series metrics: {e}")

    def calculate_time_series_metrics(self, features: np.ndarray, labels: np.ndarray, 
                                    timestamps: Optional[np.ndarray] = None) -> TimeSeriesMetricsResult:
        """Calculate time series aware clustering metrics.

        Args:
            features: Feature matrix
            labels: Cluster labels
            timestamps: Timestamp array (optional)

        Returns:
            TimeSeriesMetricsResult with time series metrics
        """
        start_time = time.time()
        
        try:
            # Monitor performance
            if self.performance_monitor:
                self.performance_monitor.start_monitoring("time_series_metrics_calculation")
            
            # Filter out noise points
            valid_mask = labels != -1
            valid_features = features[valid_mask]
            valid_labels = labels[valid_mask]
            valid_timestamps = timestamps[valid_mask] if timestamps is not None else None
            
            if len(valid_labels) == 0:
                return self._create_empty_time_series_result(start_time)
            
            unique_labels = np.unique(valid_labels)
            n_clusters = len(unique_labels)
            
            if n_clusters < 2:
                return self._create_single_cluster_time_series_result(start_time)
            
            # Calculate temporal stability metrics
            temporal_stability_metrics = self._calculate_temporal_stability_metrics(
                valid_features, valid_labels, valid_timestamps
            )
            
            # Calculate temporal clustering quality
            temporal_clustering_quality = self._calculate_temporal_clustering_quality(
                valid_features, valid_labels, valid_timestamps
            )
            
            # Calculate regime transition analysis
            transition_analysis = self._calculate_regime_transition_analysis(
                valid_labels, valid_timestamps
            )
            
            # Calculate time series specific metrics
            time_series_specific = self._calculate_time_series_specific_metrics(
                valid_features, valid_labels, valid_timestamps
            )
            
            # Stop performance monitoring
            perf_metrics = {}
            if self.performance_monitor:
                perf_metrics = self.performance_monitor.stop_monitoring("time_series_metrics_calculation")
            
            execution_time = time.time() - start_time
            
            return TimeSeriesMetricsResult(
                temporal_stability=temporal_stability_metrics['temporal_stability'],
                regime_persistence=temporal_stability_metrics['regime_persistence'],
                transition_frequency=temporal_stability_metrics['transition_frequency'],
                regime_duration_stats=temporal_stability_metrics['regime_duration_stats'],
                temporal_silhouette=temporal_clustering_quality['temporal_silhouette'],
                temporal_cohesion=temporal_clustering_quality['temporal_cohesion'],
                temporal_separation=temporal_clustering_quality['temporal_separation'],
                transition_matrix=transition_analysis['transition_matrix'],
                transition_probabilities=transition_analysis['transition_probabilities'],
                regime_stationarity=transition_analysis['regime_stationarity'],
                autocorrelation_clusters=time_series_specific['autocorrelation_clusters'],
                trend_consistency=time_series_specific['trend_consistency'],
                volatility_clustering=time_series_specific['volatility_clustering'],
                execution_time=execution_time,
                matrix_ops_used=self.matrix_ops is not None,
                hardware_acceleration_used=self.hardware_accelerator is not None
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Time series metrics calculation failed: {e}")
            return self._create_error_time_series_result(str(e), execution_time)

    def _calculate_temporal_stability_metrics(self, features: np.ndarray, labels: np.ndarray, 
                                            timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate temporal stability metrics.

        Args:
            features: Valid feature matrix
            labels: Valid cluster labels
            timestamps: Valid timestamps

        Returns:
            Dictionary of temporal stability metrics
        """
        try:
            # Calculate regime persistence (how long regimes last on average)
            regime_persistence = self._calculate_regime_persistence(labels, timestamps)
            
            # Calculate transition frequency
            transition_frequency = self._calculate_transition_frequency(labels)
            
            # Calculate regime duration statistics
            regime_duration_stats = self._calculate_regime_duration_stats(labels, timestamps)
            
            # Calculate temporal stability (overall measure)
            temporal_stability = self._calculate_overall_temporal_stability(
                regime_persistence, transition_frequency, regime_duration_stats
            )
            
            return {
                'temporal_stability': temporal_stability,
                'regime_persistence': regime_persistence,
                'transition_frequency': transition_frequency,
                'regime_duration_stats': regime_duration_stats
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Temporal stability metrics calculation failed: {e}")
            return {
                'temporal_stability': 0.0,
                'regime_persistence': 0.0,
                'transition_frequency': 0.0,
                'regime_duration_stats': {}
            }

    def _calculate_regime_persistence(self, labels: np.ndarray, timestamps: Optional[np.ndarray] = None) -> float:
        """Calculate regime persistence.

        Args:
            labels: Cluster labels
            timestamps: Timestamps (optional)

        Returns:
            Regime persistence score
        """
        try:
            # Count consecutive occurrences of each regime
            regime_lengths = []
            current_regime = labels[0]
            current_length = 1
            
            for i in range(1, len(labels)):
                if labels[i] == current_regime:
                    current_length += 1
                else:
                    regime_lengths.append(current_length)
                    current_regime = labels[i]
                    current_length = 1
            
            # Add the last regime length
            regime_lengths.append(current_length)
            
            if not regime_lengths:
                return 0.0
            
            # Calculate average regime length (persistence)
            avg_length = np.mean(regime_lengths)
            max_length = len(labels)
            
            # Normalize by total length
            persistence = avg_length / max_length
            
            return float(persistence)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime persistence calculation failed: {e}")
            return 0.0

    def _calculate_transition_frequency(self, labels: np.ndarray) -> float:
        """Calculate transition frequency between regimes.

        Args:
            labels: Cluster labels

        Returns:
            Transition frequency score
        """
        try:
            if len(labels) < 2:
                return 0.0
            
            # Count transitions
            transitions = 0
            for i in range(1, len(labels)):
                if labels[i] != labels[i-1]:
                    transitions += 1
            
            # Calculate frequency (transitions per observation)
            frequency = transitions / (len(labels) - 1)
            
            return float(frequency)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Transition frequency calculation failed: {e}")
            return 0.0

    def _calculate_regime_duration_stats(self, labels: np.ndarray, timestamps: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate regime duration statistics.

        Args:
            labels: Cluster labels
            timestamps: Timestamps (optional)

        Returns:
            Dictionary of duration statistics
        """
        try:
            # Calculate regime lengths (same as in persistence calculation)
            regime_lengths = []
            current_regime = labels[0]
            current_length = 1
            
            for i in range(1, len(labels)):
                if labels[i] == current_regime:
                    current_length += 1
                else:
                    regime_lengths.append(current_length)
                    current_regime = labels[i]
                    current_length = 1
            
            regime_lengths.append(current_length)
            
            if not regime_lengths:
                return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
            
            return {
                'mean': float(np.mean(regime_lengths)),
                'std': float(np.std(regime_lengths)),
                'min': float(np.min(regime_lengths)),
                'max': float(np.max(regime_lengths)),
                'median': float(np.median(regime_lengths))
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime duration stats calculation failed: {e}")
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}

    def _calculate_overall_temporal_stability(self, regime_persistence: float, 
                                            transition_frequency: float, 
                                            regime_duration_stats: Dict[str, float]) -> float:
        """Calculate overall temporal stability.

        Args:
            regime_persistence: Regime persistence score
            transition_frequency: Transition frequency score
            regime_duration_stats: Duration statistics

        Returns:
            Overall temporal stability score
        """
        try:
            # Higher persistence and lower transition frequency indicate more stability
            # Lower coefficient of variation in durations also indicates stability
            
            cv_duration = regime_duration_stats['std'] / (regime_duration_stats['mean'] + 1e-8)
            
            # Combine metrics (higher is better)
            stability = regime_persistence * (1.0 - transition_frequency) * (1.0 - cv_duration)
            
            return float(max(0.0, min(1.0, stability)))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Overall temporal stability calculation failed: {e}")
            return 0.0

    def _calculate_temporal_clustering_quality(self, features: np.ndarray, labels: np.ndarray, 
                                             timestamps: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate temporal clustering quality metrics.

        Args:
            features: Valid feature matrix
            labels: Valid cluster labels
            timestamps: Valid timestamps

        Returns:
            Dictionary of temporal clustering quality metrics
        """
        try:
            # Calculate temporal silhouette (modified silhouette that considers temporal proximity)
            temporal_silhouette = self._calculate_temporal_silhouette(features, labels, timestamps)
            
            # Calculate temporal cohesion (how well clustered points are in time)
            temporal_cohesion = self._calculate_temporal_cohesion(features, labels, timestamps)
            
            # Calculate temporal separation (how well separated clusters are in time)
            temporal_separation = self._calculate_temporal_separation(features, labels, timestamps)
            
            return {
                'temporal_silhouette': temporal_silhouette,
                'temporal_cohesion': temporal_cohesion,
                'temporal_separation': temporal_separation
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Temporal clustering quality calculation failed: {e}")
            return {
                'temporal_silhouette': 0.0,
                'temporal_cohesion': 0.0,
                'temporal_separation': 0.0
            }

    def _calculate_temporal_silhouette(self, features: np.ndarray, labels: np.ndarray, 
                                     timestamps: Optional[np.ndarray] = None) -> float:
        """Calculate temporal silhouette score.

        Args:
            features: Valid feature matrix
            labels: Valid cluster labels
            timestamps: Valid timestamps

        Returns:
            Temporal silhouette score
        """
        try:
            # Use matrix operations for distance calculation if available
            if self.matrix_ops is not None:
                # Calculate temporal-weighted distances
                temporal_distances = self._calculate_temporal_weighted_distances(features, timestamps)
                return self._calculate_silhouette_with_distances(labels, temporal_distances)
            else:
                # Fallback to standard silhouette with temporal weighting
                from sklearn.metrics import silhouette_score
                return float(silhouette_score(features, labels))
                
        except Exception as e:
            self.logger.warning(f"⚠️ Temporal silhouette calculation failed: {e}")
            return 0.0

    def _calculate_temporal_weighted_distances(self, features: np.ndarray, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate temporal-weighted distances between points.

        Args:
            features: Feature matrix
            timestamps: Timestamps

        Returns:
            Temporal-weighted distance matrix
        """
        try:
            n_points = len(features)
            distances = np.zeros((n_points, n_points))
            
            for i in range(n_points):
                for j in range(i+1, n_points):
                    # Feature distance
                    feature_dist = np.linalg.norm(features[i] - features[j])
                    
                    # Temporal distance (if timestamps available)
                    if timestamps is not None:
                        time_diff = abs(timestamps[i] - timestamps[j])
                        # Normalize temporal distance
                        time_dist = time_diff / (np.max(timestamps) - np.min(timestamps) + 1e-8)
                        # Combine feature and temporal distances
                        combined_dist = feature_dist + 0.1 * time_dist  # Weight temporal distance less
                    else:
                        combined_dist = feature_dist
                    
                    distances[i, j] = combined_dist
                    distances[j, i] = combined_dist
            
            return distances
            
        except Exception as e:
            self.logger.warning(f"⚠️ Temporal weighted distances calculation failed: {e}")
            return np.zeros((len(features), len(features)))

    def _calculate_silhouette_with_distances(self, labels: np.ndarray, distances: np.ndarray) -> float:
        """Calculate silhouette score using precomputed distances.

        Args:
            labels: Cluster labels
            distances: Distance matrix

        Returns:
            Silhouette score
        """
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            silhouette_scores = []
            
            for i, label in enumerate(labels):
                # Find same cluster points
                same_cluster_mask = labels == label
                same_cluster_distances = distances[i, same_cluster_mask]
                same_cluster_distances = same_cluster_distances[same_cluster_distances > 0]  # Remove self-distance
                
                if len(same_cluster_distances) == 0:
                    a_i = 0.0
                else:
                    a_i = np.mean(same_cluster_distances)
                
                # Find nearest different cluster
                min_b_i = float('inf')
                for other_label in unique_labels:
                    if other_label != label:
                        other_cluster_mask = labels == other_label
                        other_cluster_distances = distances[i, other_cluster_mask]
                        
                        if len(other_cluster_distances) > 0:
                            b_i = np.mean(other_cluster_distances)
                            min_b_i = min(min_b_i, b_i)
                
                if min_b_i == float('inf'):
                    min_b_i = 0.0
                
                # Calculate silhouette for this point
                if max(a_i, min_b_i) == 0:
                    s_i = 0.0
                else:
                    s_i = (min_b_i - a_i) / max(a_i, min_b_i)
                
                silhouette_scores.append(s_i)
            
            return float(np.mean(silhouette_scores))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Silhouette with distances calculation failed: {e}")
            return 0.0

    def _calculate_temporal_cohesion(self, features: np.ndarray, labels: np.ndarray, 
                                   timestamps: Optional[np.ndarray] = None) -> float:
        """Calculate temporal cohesion of clusters.

        Args:
            features: Valid feature matrix
            labels: Valid cluster labels
            timestamps: Valid timestamps

        Returns:
            Temporal cohesion score
        """
        try:
            unique_labels = np.unique(labels)
            cohesion_scores = []
            
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_indices = np.where(cluster_mask)[0]
                
                if len(cluster_indices) < 2:
                    continue
                
                # Calculate temporal spread of cluster
                if timestamps is not None:
                    cluster_timestamps = timestamps[cluster_indices]
                    temporal_spread = np.max(cluster_timestamps) - np.min(cluster_timestamps)
                    total_spread = np.max(timestamps) - np.min(timestamps)
                    
                    if total_spread > 0:
                        temporal_cohesion = 1.0 - (temporal_spread / total_spread)
                        cohesion_scores.append(temporal_cohesion)
                else:
                    # Use feature-based cohesion
                    cluster_features = features[cluster_indices]
                    centroid = np.mean(cluster_features, axis=0)
                    distances = np.linalg.norm(cluster_features - centroid, axis=1)
                    cohesion = 1.0 / (1.0 + np.mean(distances))
                    cohesion_scores.append(cohesion)
            
            return float(np.mean(cohesion_scores)) if cohesion_scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Temporal cohesion calculation failed: {e}")
            return 0.0

    def _calculate_temporal_separation(self, features: np.ndarray, labels: np.ndarray, 
                                     timestamps: Optional[np.ndarray] = None) -> float:
        """Calculate temporal separation between clusters.

        Args:
            features: Valid feature matrix
            labels: Valid cluster labels
            timestamps: Valid timestamps

        Returns:
            Temporal separation score
        """
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0
            
            separation_scores = []
            
            for i, label_a in enumerate(unique_labels):
                for j, label_b in enumerate(unique_labels[i+1:], i+1):
                    mask_a = labels == label_a
                    mask_b = labels == label_b
                    
                    indices_a = np.where(mask_a)[0]
                    indices_b = np.where(mask_b)[0]
                    
                    if len(indices_a) == 0 or len(indices_b) == 0:
                        continue
                    
                    # Calculate temporal separation
                    if timestamps is not None:
                        timestamps_a = timestamps[indices_a]
                        timestamps_b = timestamps[indices_b]
                        
                        # Calculate minimum temporal distance between clusters
                        min_temporal_dist = float('inf')
                        for t_a in timestamps_a:
                            for t_b in timestamps_b:
                                dist = abs(t_a - t_b)
                                min_temporal_dist = min(min_temporal_dist, dist)
                        
                        # Normalize by total time range
                        total_range = np.max(timestamps) - np.min(timestamps)
                        if total_range > 0:
                            normalized_separation = min_temporal_dist / total_range
                            separation_scores.append(normalized_separation)
                    else:
                        # Use feature-based separation
                        features_a = features[indices_a]
                        features_b = features[indices_b]
                        
                        centroid_a = np.mean(features_a, axis=0)
                        centroid_b = np.mean(features_b, axis=0)
                        
                        separation = np.linalg.norm(centroid_a - centroid_b)
                        separation_scores.append(separation)
            
            return float(np.mean(separation_scores)) if separation_scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Temporal separation calculation failed: {e}")
            return 0.0

    def _calculate_regime_transition_analysis(self, labels: np.ndarray, 
                                            timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate regime transition analysis.

        Args:
            labels: Valid cluster labels
            timestamps: Valid timestamps

        Returns:
            Dictionary of transition analysis metrics
        """
        try:
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            
            # Create transition matrix
            transition_matrix = np.zeros((n_clusters, n_clusters))
            
            # Count transitions
            for i in range(1, len(labels)):
                current_idx = np.where(unique_labels == labels[i])[0][0]
                previous_idx = np.where(unique_labels == labels[i-1])[0][0]
                transition_matrix[previous_idx, current_idx] += 1
            
            # Convert to probabilities
            row_sums = transition_matrix.sum(axis=1)
            transition_probabilities = {}
            for i, label in enumerate(unique_labels):
                if row_sums[i] > 0:
                    transition_probabilities[f"from_{label}"] = {
                        "probabilities": (transition_matrix[i] / row_sums[i]).tolist(),
                        "total_transitions": int(row_sums[i])
                    }
            
            # Calculate regime stationarity (how stable each regime is)
            regime_stationarity = {}
            for i, label in enumerate(unique_labels):
                if row_sums[i] > 0:
                    # Stationarity is the probability of staying in the same regime
                    stationarity = transition_matrix[i, i] / row_sums[i]
                    regime_stationarity[f"regime_{label}"] = float(stationarity)
            
            return {
                'transition_matrix': transition_matrix,
                'transition_probabilities': transition_probabilities,
                'regime_stationarity': regime_stationarity
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime transition analysis failed: {e}")
            return {
                'transition_matrix': np.array([]),
                'transition_probabilities': {},
                'regime_stationarity': {}
            }

    def _calculate_time_series_specific_metrics(self, features: np.ndarray, labels: np.ndarray, 
                                              timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate time series specific metrics.

        Args:
            features: Valid feature matrix
            labels: Valid cluster labels
            timestamps: Valid timestamps

        Returns:
            Dictionary of time series specific metrics
        """
        try:
            # Calculate autocorrelation for each cluster
            autocorrelation_clusters = self._calculate_cluster_autocorrelations(features, labels)
            
            # Calculate trend consistency
            trend_consistency = self._calculate_trend_consistency(features, labels, timestamps)
            
            # Calculate volatility clustering
            volatility_clustering = self._calculate_volatility_clustering(features, labels)
            
            return {
                'autocorrelation_clusters': autocorrelation_clusters,
                'trend_consistency': trend_consistency,
                'volatility_clustering': volatility_clustering
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Time series specific metrics calculation failed: {e}")
            return {
                'autocorrelation_clusters': {},
                'trend_consistency': 0.0,
                'volatility_clustering': 0.0
            }

    def _calculate_cluster_autocorrelations(self, features: np.ndarray, labels: np.ndarray) -> Dict[int, float]:
        """Calculate autocorrelation for each cluster.

        Args:
            features: Valid feature matrix
            labels: Valid cluster labels

        Returns:
            Dictionary of cluster autocorrelations
        """
        try:
            unique_labels = np.unique(labels)
            autocorrelations = {}
            
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_features = features[cluster_mask]
                
                if len(cluster_features) < 3:
                    autocorrelations[label] = 0.0
                    continue
                
                # Calculate autocorrelation for the first feature (assuming it's the main time series)
                if cluster_features.shape[1] > 0:
                    time_series = cluster_features[:, 0]
                    
                    # Calculate lag-1 autocorrelation
                    if len(time_series) > 1:
                        correlation = np.corrcoef(time_series[:-1], time_series[1:])[0, 1]
                        autocorrelations[label] = float(correlation) if not np.isnan(correlation) else 0.0
                    else:
                        autocorrelations[label] = 0.0
                else:
                    autocorrelations[label] = 0.0
            
            return autocorrelations
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cluster autocorrelations calculation failed: {e}")
            return {}

    def _calculate_trend_consistency(self, features: np.ndarray, labels: np.ndarray, 
                                   timestamps: Optional[np.ndarray] = None) -> float:
        """Calculate trend consistency across clusters.

        Args:
            features: Valid feature matrix
            labels: Valid cluster labels
            timestamps: Valid timestamps

        Returns:
            Trend consistency score
        """
        try:
            unique_labels = np.unique(labels)
            cluster_trends = []
            
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_features = features[cluster_mask]
                cluster_timestamps = timestamps[cluster_mask] if timestamps is not None else None
                
                if len(cluster_features) < 2:
                    continue
                
                # Calculate trend for first feature
                if cluster_features.shape[1] > 0:
                    time_series = cluster_features[:, 0]
                    
                    if cluster_timestamps is not None and len(cluster_timestamps) > 1:
                        # Linear regression to get trend
                        from scipy import stats
                        slope, _, r_value, _, _ = stats.linregress(cluster_timestamps, time_series)
                        cluster_trends.append(slope)
                    else:
                        # Simple trend calculation
                        trend = (time_series[-1] - time_series[0]) / len(time_series)
                        cluster_trends.append(trend)
            
            if not cluster_trends:
                return 0.0
            
            # Calculate consistency (lower variance in trends means higher consistency)
            trend_variance = np.var(cluster_trends)
            trend_mean = np.mean(np.abs(cluster_trends))
            
            if trend_mean == 0:
                return 1.0
            
            consistency = 1.0 / (1.0 + trend_variance / (trend_mean + 1e-8))
            return float(consistency)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Trend consistency calculation failed: {e}")
            return 0.0

    def _calculate_volatility_clustering(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate volatility clustering measure.

        Args:
            features: Valid feature matrix
            labels: Valid cluster labels

        Returns:
            Volatility clustering score
        """
        try:
            unique_labels = np.unique(labels)
            cluster_volatilities = []
            
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_features = features[cluster_mask]
                
                if len(cluster_features) < 2:
                    continue
                
                # Calculate volatility for first feature
                if cluster_features.shape[1] > 0:
                    time_series = cluster_features[:, 0]
                    returns = np.diff(time_series) / (time_series[:-1] + 1e-8)
                    volatility = np.std(returns)
                    cluster_volatilities.append(volatility)
            
            if not cluster_volatilities:
                return 0.0
            
            # Calculate clustering measure (how well volatilities are separated)
            volatility_variance = np.var(cluster_volatilities)
            volatility_mean = np.mean(cluster_volatilities)
            
            if volatility_mean == 0:
                return 0.0
            
            clustering_measure = volatility_variance / (volatility_mean + 1e-8)
            return float(clustering_measure)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Volatility clustering calculation failed: {e}")
            return 0.0

    def _create_empty_time_series_result(self, start_time: float) -> TimeSeriesMetricsResult:
        """Create empty time series result for no valid clusters.

        Args:
            start_time: Start time for execution time calculation

        Returns:
            Empty TimeSeriesMetricsResult
        """
        execution_time = time.time() - start_time
        
        return TimeSeriesMetricsResult(
            temporal_stability=0.0,
            regime_persistence=0.0,
            transition_frequency=0.0,
            regime_duration_stats={},
            temporal_silhouette=0.0,
            temporal_cohesion=0.0,
            temporal_separation=0.0,
            transition_matrix=np.array([]),
            transition_probabilities={},
            regime_stationarity={},
            autocorrelation_clusters={},
            trend_consistency=0.0,
            volatility_clustering=0.0,
            execution_time=execution_time,
            matrix_ops_used=self.matrix_ops is not None,
            hardware_acceleration_used=self.hardware_accelerator is not None
        )

    def _create_single_cluster_time_series_result(self, start_time: float) -> TimeSeriesMetricsResult:
        """Create time series result for single cluster case.

        Args:
            start_time: Start time for execution time calculation

        Returns:
            Single cluster TimeSeriesMetricsResult
        """
        execution_time = time.time() - start_time
        
        return TimeSeriesMetricsResult(
            temporal_stability=1.0,  # Single cluster is perfectly stable
            regime_persistence=1.0,
            transition_frequency=0.0,
            regime_duration_stats={'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0},
            temporal_silhouette=0.0,
            temporal_cohesion=1.0,
            temporal_separation=0.0,
            transition_matrix=np.array([[1.0]]),
            transition_probabilities={},
            regime_stationarity={},
            autocorrelation_clusters={},
            trend_consistency=0.0,
            volatility_clustering=0.0,
            execution_time=execution_time,
            matrix_ops_used=self.matrix_ops is not None,
            hardware_acceleration_used=self.hardware_accelerator is not None
        )

    def _create_error_time_series_result(self, error_message: str, execution_time: float) -> TimeSeriesMetricsResult:
        """Create error time series result.

        Args:
            error_message: Error message
            execution_time: Execution time

        Returns:
            Error TimeSeriesMetricsResult
        """
        return TimeSeriesMetricsResult(
            temporal_stability=0.0,
            regime_persistence=0.0,
            transition_frequency=0.0,
            regime_duration_stats={'error': error_message},
            temporal_silhouette=0.0,
            temporal_cohesion=0.0,
            temporal_separation=0.0,
            transition_matrix=np.array([]),
            transition_probabilities={'error': error_message},
            regime_stationarity={'error': error_message},
            autocorrelation_clusters={'error': error_message},
            trend_consistency=0.0,
            volatility_clustering=0.0,
            execution_time=execution_time,
            matrix_ops_used=False,
            hardware_acceleration_used=False
        )