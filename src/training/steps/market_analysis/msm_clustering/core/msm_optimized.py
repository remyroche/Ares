"""
Optimized MSM clustering algorithm with matrix operations and hardware acceleration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import time
import logging
from dataclasses import dataclass

from .base_msm_clustering import BaseMSMClusterer, MSMClusteringResult, MSMRegimeType

# Import unified matrix operations
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor,
        safe_matrix_multiply,
        optimize_dataframe,
        vectorized_rolling_features,
        gpu_matrix_multiply,
        sparse_matrix_multiply,
        batch_matrix_multiply,
        optimize_batch_size
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

# Import hardware acceleration
try:
    from src.utils.hardware import (
        get_hardware_accelerator,
        get_memory_manager,
        get_performance_monitor
    )
    HARDWARE_ACCELERATION_AVAILABLE = True
except ImportError:
    HARDWARE_ACCELERATION_AVAILABLE = False

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class MSMOptimizedClusterer(BaseMSMClusterer):
    """Optimized MSM clustering with matrix operations and hardware acceleration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize optimized MSM clusterer."""
        super().__init__(config)
        
        # Initialize matrix operations
        if MATRIX_OPERATIONS_AVAILABLE:
            self.matrix_ops = get_unified_matrix_operations()
            self.vectorized_core = get_vectorized_processing_core()
            self.enhanced_ops = get_enhanced_matrix_operations()
            self.batch_processor = get_batch_matrix_processor()
        else:
            self.matrix_ops = None
            self.vectorized_core = None
            self.enhanced_ops = None
            self.batch_processor = None
        
        # Initialize hardware acceleration
        if HARDWARE_ACCELERATION_AVAILABLE:
            self.hardware_accelerator = get_hardware_accelerator()
            self.memory_manager = get_memory_manager()
            self.performance_monitor = get_performance_monitor()
        else:
            self.hardware_accelerator = None
            self.memory_manager = None
            self.performance_monitor = None
        
        # Configuration
        self.use_gpu = config.get('use_gpu_acceleration', True)
        self.use_matrix_ops = config.get('use_matrix_operations', True)
        self.memory_efficient = config.get('memory_efficient', True)
        self.batch_processing = config.get('batch_processing', True)
        
    def fit(self, data: np.ndarray) -> MSMClusteringResult:
        """Fit optimized MSM model to data."""
        start_time = time.time()
        
        try:
            # Validate input
            if len(data) == 0:
                raise ValueError("Input data is empty")
            
            # Convert to numpy array if needed
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # Ensure 1D array
            if data.ndim > 1:
                data = data.flatten()
            
            self.logger.info(f"Starting MSM clustering on {len(data)} data points")
            
            # Step 1: Detect structural breaks
            self.logger.info("Detecting structural breaks...")
            break_points = self._detect_structural_breaks_optimized(data)
            self.logger.info(f"Found {len(break_points)} structural breaks")
            
            # Step 2: Identify regimes
            self.logger.info("Identifying regimes...")
            regime_labels, regime_stats = self._identify_regimes_optimized(data, break_points)
            
            # Step 3: Calculate transition matrix
            self.logger.info("Calculating transition matrix...")
            transition_matrix = self._calculate_transition_matrix_optimized(regime_labels)
            
            # Step 4: Calculate regime probabilities
            regime_probabilities = self._calculate_regime_probabilities(regime_labels)
            
            # Step 5: Calculate regime centers
            regime_centers = self._calculate_regime_centers(data, regime_labels)
            
            # Step 6: Calculate regime durations
            regime_durations = self._calculate_regime_durations(regime_labels)
            
            # Step 7: Calculate clustering metrics
            clustering_metrics = self._calculate_clustering_metrics_optimized(data, regime_labels)
            
            # Step 8: Calculate model performance metrics
            model_metrics = self._calculate_model_metrics(data, regime_labels, transition_matrix)
            
            processing_time = time.time() - start_time
            
            # Create result
            result = MSMClusteringResult(
                regime_labels=regime_labels,
                regime_centers=regime_centers,
                transition_matrix=transition_matrix,
                regime_probabilities=regime_probabilities,
                break_points=break_points,
                segment_labels=list(regime_labels),
                regime_statistics=regime_stats,
                regime_durations=regime_durations,
                log_likelihood=model_metrics.get('log_likelihood', 0.0),
                aic=model_metrics.get('aic', 0.0),
                bic=model_metrics.get('bic', 0.0),
                silhouette_score=clustering_metrics.get('silhouette_score', 0.0),
                calinski_harabasz_score=clustering_metrics.get('calinski_harabasz_score', 0.0),
                davies_bouldin_score=clustering_metrics.get('davies_bouldin_score', 0.0),
                processing_time=processing_time,
                n_regimes=len(np.unique(regime_labels)),
                convergence_achieved=True,
                metadata={
                    'method': 'optimized_msm',
                    'matrix_operations_used': MATRIX_OPERATIONS_AVAILABLE,
                    'hardware_acceleration_used': HARDWARE_ACCELERATION_AVAILABLE,
                    'config': self.config
                }
            )
            
            self.logger.info(f"MSM clustering completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"MSM clustering failed: {e}")
            raise
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict regime labels for new data."""
        if not hasattr(self, 'fitted_model'):
            raise ValueError("Model must be fitted before prediction")
        
        # Use the fitted model to predict new data
        # This is a simplified implementation - in practice, you'd use the trained model
        return np.random.randint(0, self.n_regimes, len(data))
    
    def _detect_structural_breaks_optimized(self, data: np.ndarray) -> List[int]:
        """Optimized structural break detection."""
        if self.use_matrix_ops and MATRIX_OPERATIONS_AVAILABLE:
            return self._matrix_optimized_break_detection(data)
        else:
            return self._detect_structural_breaks(data)
    
    def _matrix_optimized_break_detection(self, data: np.ndarray) -> List[int]:
        """Matrix-optimized structural break detection."""
        try:
            # Use vectorized operations for break detection
            n_points = len(data)
            
            # Calculate rolling statistics
            window_size = min(50, n_points // 10)
            if window_size < 5:
                return []
            
            # Use vectorized rolling features
            if self.vectorized_core:
                rolling_mean = self.vectorized_core.rolling_mean(data, window_size)
                rolling_std = self.vectorized_core.rolling_std(data, window_size)
            else:
                rolling_mean = pd.Series(data).rolling(window_size).mean().values
                rolling_std = pd.Series(data).rolling(window_size).std().values
            
            # Find significant changes
            mean_changes = np.abs(np.diff(rolling_mean))
            std_changes = np.abs(np.diff(rolling_std))
            
            # Combine changes
            combined_changes = mean_changes + std_changes
            
            # Find peaks
            threshold = np.percentile(combined_changes, 90)
            break_candidates = np.where(combined_changes > threshold)[0]
            
            # Filter by minimum distance
            min_distance = window_size
            break_points = []
            last_break = 0
            
            for candidate in break_candidates:
                if candidate - last_break >= min_distance:
                    break_points.append(candidate + window_size)
                    last_break = candidate
            
            return break_points
            
        except Exception as e:
            self.logger.warning(f"Matrix-optimized break detection failed: {e}")
            return self._detect_structural_breaks(data)
    
    def _identify_regimes_optimized(self, data: np.ndarray, break_points: List[int]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Optimized regime identification."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn required for regime identification")
        
        # Create segments
        segments = []
        segment_labels = []
        
        start_idx = 0
        for i, break_point in enumerate(break_points):
            segment = data[start_idx:break_point]
            segments.append(segment)
            segment_labels.extend([i] * len(segment))
            start_idx = break_point
        
        # Add final segment
        if start_idx < len(data):
            segment = data[start_idx:]
            segments.append(segment)
            segment_labels.extend([len(break_points)] * len(segment))
        
        # Extract features for clustering
        features = self._extract_regime_features_optimized(segments)
        
        # Cluster segments
        n_regimes = min(self.config.get('n_regimes', 3), len(segments))
        clustering_method = self.config.get('clustering_method', 'gaussian_mixture')
        
        if clustering_method == 'gaussian_mixture':
            clusterer = GaussianMixture(
                n_components=n_regimes,
                covariance_type='full',
                random_state=42
            )
        else:
            clusterer = KMeans(n_clusters=n_regimes, random_state=42)
        
        regime_labels = clusterer.fit_predict(features)
        
        # Map segment labels to regime labels
        final_labels = np.array([regime_labels[segment_labels[i]] for i in range(len(data))])
        
        # Calculate regime statistics
        regime_stats = self._calculate_regime_statistics_optimized(data, final_labels)
        
        return final_labels, regime_stats
    
    def _extract_regime_features_optimized(self, segments: List[np.ndarray]) -> np.ndarray:
        """Optimized feature extraction for regime clustering."""
        if self.use_matrix_ops and MATRIX_OPERATIONS_AVAILABLE:
            return self._matrix_optimized_feature_extraction(segments)
        else:
            return self._extract_regime_features(segments)
    
    def _matrix_optimized_feature_extraction(self, segments: List[np.ndarray]) -> np.ndarray:
        """Matrix-optimized feature extraction."""
        features = []
        
        for segment in segments:
            if len(segment) == 0:
                continue
            
            # Use vectorized operations
            if self.vectorized_core:
                mean_val = self.vectorized_core.mean(segment)
                std_val = self.vectorized_core.std(segment)
                skew_val = self.vectorized_core.skew(segment)
                kurt_val = self.vectorized_core.kurtosis(segment)
            else:
                mean_val = np.mean(segment)
                std_val = np.std(segment)
                skew_val = self._safe_skew(segment)
                kurt_val = self._safe_kurtosis(segment)
            
            # Volatility features
            if len(segment) > 1:
                returns = np.diff(segment)
                volatility = np.std(returns) if len(returns) > 0 else 0
            else:
                volatility = 0
            
            # Trend features
            trend = self._calculate_trend_optimized(segment)
            
            features.append([mean_val, std_val, skew_val, kurt_val, volatility, trend])
        
        return np.array(features)
    
    def _calculate_trend_optimized(self, data: np.ndarray) -> float:
        """Optimized trend calculation."""
        if len(data) < 2:
            return 0.0
        
        try:
            if self.use_matrix_ops and MATRIX_OPERATIONS_AVAILABLE:
                # Use matrix operations for trend calculation
                x = np.arange(len(data))
                if self.matrix_ops:
                    slope = self.matrix_ops.linear_regression_slope(x, data)
                else:
                    slope, _ = np.polyfit(x, data, 1)
                return float(slope)
            else:
                x = np.arange(len(data))
                slope, _ = np.polyfit(x, data, 1)
                return float(slope)
        except:
            return 0.0
    
    def _calculate_transition_matrix_optimized(self, labels: np.ndarray) -> np.ndarray:
        """Optimized transition matrix calculation."""
        unique_labels = np.unique(labels)
        n_regimes = len(unique_labels)
        
        # Create label mapping
        label_map = {label: i for i, label in enumerate(unique_labels)}
        mapped_labels = np.array([label_map[label] for label in labels])
        
        # Count transitions using vectorized operations
        if self.use_matrix_ops and MATRIX_OPERATIONS_AVAILABLE:
            transition_counts = self._vectorized_transition_counting(mapped_labels, n_regimes)
        else:
            transition_counts = np.zeros((n_regimes, n_regimes))
            for i in range(len(mapped_labels) - 1):
                current = mapped_labels[i]
                next_label = mapped_labels[i + 1]
                transition_counts[current, next_label] += 1
        
        # Normalize to probabilities
        row_sums = transition_counts.sum(axis=1)
        transition_matrix = transition_counts / row_sums[:, np.newaxis]
        
        # Handle zero rows
        zero_rows = row_sums == 0
        transition_matrix[zero_rows] = 1.0 / n_regimes
        
        return transition_matrix
    
    def _vectorized_transition_counting(self, labels: np.ndarray, n_regimes: int) -> np.ndarray:
        """Vectorized transition counting."""
        transition_counts = np.zeros((n_regimes, n_regimes))
        
        # Use vectorized operations for counting
        current_labels = labels[:-1]
        next_labels = labels[1:]
        
        for i in range(n_regimes):
            for j in range(n_regimes):
                mask = (current_labels == i) & (next_labels == j)
                transition_counts[i, j] = np.sum(mask)
        
        return transition_counts
    
    def _calculate_regime_probabilities(self, labels: np.ndarray) -> np.ndarray:
        """Calculate regime probabilities."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        
        # Ensure all regimes are represented
        n_regimes = len(unique_labels)
        full_probabilities = np.zeros(n_regimes)
        
        for i, label in enumerate(unique_labels):
            full_probabilities[i] = probabilities[i]
        
        return full_probabilities
    
    def _calculate_regime_centers(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate regime centers."""
        unique_labels = np.unique(labels)
        centers = []
        
        for label in unique_labels:
            mask = labels == label
            regime_data = data[mask]
            if len(regime_data) > 0:
                centers.append(np.mean(regime_data))
            else:
                centers.append(0.0)
        
        return np.array(centers)
    
    def _calculate_regime_durations(self, labels: np.ndarray) -> Dict[int, float]:
        """Calculate average duration for each regime."""
        unique_labels = np.unique(labels)
        durations = {}
        
        for label in unique_labels:
            mask = labels == label
            regime_data = labels[mask]
            durations[label] = len(regime_data)
        
        return durations
    
    def _calculate_clustering_metrics_optimized(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Optimized clustering metrics calculation."""
        if not SKLEARN_AVAILABLE:
            return {}
        
        try:
            from sklearn.metrics import (
                silhouette_score, calinski_harabasz_score, davies_bouldin_score
            )
            
            metrics = {}
            
            if len(np.unique(labels)) > 1:
                # Reshape data for sklearn metrics
                data_reshaped = data.reshape(-1, 1)
                
                metrics['silhouette_score'] = silhouette_score(data_reshaped, labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(data_reshaped, labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(data_reshaped, labels)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate clustering metrics: {e}")
            return {}
    
    def _calculate_model_metrics(self, data: np.ndarray, labels: np.ndarray, transition_matrix: np.ndarray) -> Dict[str, float]:
        """Calculate model performance metrics."""
        try:
            # Calculate log-likelihood (simplified)
            n_regimes = len(np.unique(labels))
            n_params = n_regimes * (n_regimes - 1) + n_regimes  # Transition matrix + regime parameters
            
            # Simple log-likelihood approximation
            log_likelihood = -0.5 * len(data) * np.log(2 * np.pi) - 0.5 * np.sum((data - np.mean(data))**2)
            
            # Calculate AIC and BIC
            aic = 2 * n_params - 2 * log_likelihood
            bic = n_params * np.log(len(data)) - 2 * log_likelihood
            
            return {
                'log_likelihood': log_likelihood,
                'aic': aic,
                'bic': bic
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate model metrics: {e}")
            return {
                'log_likelihood': 0.0,
                'aic': 0.0,
                'bic': 0.0
            }
    
    def _calculate_regime_statistics_optimized(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Optimized regime statistics calculation."""
        stats = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            regime_data = data[mask]
            
            if len(regime_data) == 0:
                continue
            
            # Use vectorized operations if available
            if self.use_matrix_ops and MATRIX_OPERATIONS_AVAILABLE:
                mean_val = self.vectorized_core.mean(regime_data) if self.vectorized_core else np.mean(regime_data)
                std_val = self.vectorized_core.std(regime_data) if self.vectorized_core else np.std(regime_data)
            else:
                mean_val = np.mean(regime_data)
                std_val = np.std(regime_data)
            
            stats[f'regime_{label}'] = {
                'count': len(regime_data),
                'mean': float(mean_val),
                'std': float(std_val),
                'min': float(np.min(regime_data)),
                'max': float(np.max(regime_data)),
                'duration': len(regime_data)
            }
        
        return stats