"""
Sample Reallocator

This module provides post-clustering optimization capabilities for
HDBSCAN-based regime discovery, including sample reallocation,
cluster refinement, and quality improvement.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
import warnings

logger = logging.getLogger(__name__)

@dataclass
class SampleReallocatorConfig:
    """Configuration for sample reallocation."""
    # Reallocation parameters
    enable_reallocation: bool = True
    max_iterations: int = 10
    convergence_threshold: float = 1e-4
    
    # Quality metrics
    quality_metric: str = 'silhouette'  # 'silhouette', 'calinski_harabasz', 'davies_bouldin'
    min_improvement: float = 0.01
    
    # Sample selection
    reallocation_strategy: str = 'border'  # 'border', 'uncertain', 'all'
    border_threshold: float = 0.1
    uncertainty_threshold: float = 0.5
    
    # Cluster constraints
    min_cluster_size: int = 5
    max_cluster_size: Optional[int] = None
    preserve_cluster_count: bool = True
    
    # Distance metrics
    distance_metric: str = 'euclidean'
    use_approximate: bool = True
    n_neighbors: int = 10
    
    # Validation
    validate_reallocation: bool = True
    max_reallocation_ratio: float = 0.3
    
    # Regime-aware reallocation
    enable_regime_aware_reallocation: bool = True
    regime_detection_method: str = 'variance'  # 'variance', 'entropy', 'volatility'
    regime_window: int = 20
    regime_threshold: float = 0.1
    regime_specific_quality_metrics: bool = True
    regime_transition_aware: bool = True

class SampleReallocator:
    """
    Post-clustering sample reallocator for regime discovery.
    
    Provides optimization capabilities to improve clustering quality
    through intelligent sample reallocation and cluster refinement.
    """
    
    def __init__(self, config: Optional[SampleReallocatorConfig] = None):
        """
        Initialize sample reallocator.
        
        Args:
            config: Configuration for sample reallocation
        """
        self.config = config or SampleReallocatorConfig()
        self.reallocation_stats = {}
        self.original_labels = None
        self.original_features = None
        
    def reallocate_samples(self, 
                          cluster_labels: np.ndarray,
                          features: np.ndarray,
                          target_metric: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reallocate samples to improve clustering quality.
        
        Args:
            cluster_labels: Current cluster labels
            features: Feature matrix
            target_metric: Target quality metric (optional)
            
        Returns:
            Tuple of (optimized_labels, reallocation_info)
        """
        try:
            if not self.config.enable_reallocation:
                logger.info("Sample reallocation disabled")
                return cluster_labels, {'reallocation_performed': False}
            
            logger.info("🔄 Starting sample reallocation...")
            
            # Store original data
            self.original_labels = cluster_labels.copy()
            self.original_features = features.copy()
            
            # Validate input
            if self.config.validate_reallocation:
                cluster_labels, features = self._validate_input(cluster_labels, features)
            
            # Calculate initial quality
            initial_quality = self._calculate_quality_metric(cluster_labels, features, target_metric)
            
            # Perform reallocation
            if self.config.enable_regime_aware_reallocation:
                optimized_labels = self._perform_regime_aware_reallocation(cluster_labels, features, target_metric)
            else:
                optimized_labels = self._perform_reallocation(cluster_labels, features, target_metric)
            
            # Calculate final quality
            final_quality = self._calculate_quality_metric(optimized_labels, features, target_metric)
            
            # Calculate reallocation statistics
            reallocation_info = self._calculate_reallocation_stats(
                cluster_labels, optimized_labels, features, initial_quality, final_quality
            )
            
            self.reallocation_stats = reallocation_info
            
            logger.info(f"✅ Sample reallocation completed. Quality improvement: {final_quality - initial_quality:.4f}")
            
            return optimized_labels, reallocation_info
            
        except Exception as e:
            logger.error(f"❌ Sample reallocation failed: {e}")
            return cluster_labels, {'error': str(e)}
    
    def _validate_input(self, cluster_labels: np.ndarray, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Validate input data for reallocation."""
        try:
            # Check for sufficient samples
            if len(cluster_labels) < 10:
                logger.warning("⚠️ Insufficient samples for reallocation")
                return cluster_labels, features
            
            # Check for sufficient clusters
            unique_labels = np.unique(cluster_labels)
            unique_labels = unique_labels[unique_labels != -1]  # Remove noise
            if len(unique_labels) < 2:
                logger.warning("⚠️ Insufficient clusters for reallocation")
                return cluster_labels, features
            
            # Check for NaN or infinite values
            if np.isnan(features).any() or np.isinf(features).any():
                logger.warning("⚠️ Found NaN or infinite values, cleaning data")
                features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
            
            return cluster_labels, features
            
        except Exception as e:
            logger.error(f"❌ Input validation failed: {e}")
            return cluster_labels, features
    
    def _perform_reallocation(self, 
                            cluster_labels: np.ndarray, 
                            features: np.ndarray,
                            target_metric: Optional[str] = None) -> np.ndarray:
        """Perform iterative sample reallocation."""
        try:
            current_labels = cluster_labels.copy()
            best_labels = current_labels.copy()
            best_quality = self._calculate_quality_metric(current_labels, features, target_metric)
            
            iteration = 0
            improvement = float('inf')
            
            while iteration < self.config.max_iterations and improvement > self.config.convergence_threshold:
                iteration += 1
                logger.debug(f"Reallocation iteration {iteration}")
                
                # Select samples for reallocation
                samples_to_reallocate = self._select_samples_for_reallocation(current_labels, features)
                
                if len(samples_to_reallocate) == 0:
                    logger.info("No samples selected for reallocation, stopping")
                    break
                
                # Reallocate selected samples
                new_labels = self._reallocate_selected_samples(
                    current_labels, features, samples_to_reallocate, target_metric
                )
                
                # Calculate new quality
                new_quality = self._calculate_quality_metric(new_labels, features, target_metric)
                
                # Check for improvement
                improvement = new_quality - best_quality
                
                if improvement > self.config.min_improvement:
                    best_labels = new_labels.copy()
                    best_quality = new_quality
                    current_labels = new_labels
                    logger.debug(f"Iteration {iteration}: Quality improved by {improvement:.4f}")
                else:
                    logger.debug(f"Iteration {iteration}: No significant improvement ({improvement:.4f})")
                    break
            
            logger.info(f"Reallocation completed in {iteration} iterations")
            return best_labels
            
        except Exception as e:
            logger.error(f"❌ Reallocation process failed: {e}")
            return cluster_labels
    
    def _select_samples_for_reallocation(self, 
                                       cluster_labels: np.ndarray, 
                                       features: np.ndarray) -> np.ndarray:
        """Select samples for reallocation based on strategy."""
        try:
            if self.config.reallocation_strategy == 'border':
                return self._select_border_samples(cluster_labels, features)
            elif self.config.reallocation_strategy == 'uncertain':
                return self._select_uncertain_samples(cluster_labels, features)
            elif self.config.reallocation_strategy == 'all':
                return np.arange(len(cluster_labels))
            else:
                logger.warning(f"⚠️ Unknown reallocation strategy: {self.config.reallocation_strategy}")
                return self._select_border_samples(cluster_labels, features)
                
        except Exception as e:
            logger.error(f"❌ Sample selection failed: {e}")
            return np.array([])
    
    def _select_border_samples(self, cluster_labels: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Select samples near cluster boundaries."""
        try:
            # Calculate distances to cluster centers
            cluster_centers = self._calculate_cluster_centers(cluster_labels, features)
            
            if len(cluster_centers) == 0:
                return np.array([])
            
            # Calculate distances from each sample to all cluster centers
            distances = np.sqrt(((features[:, np.newaxis] - cluster_centers[np.newaxis, :]) ** 2).sum(axis=2))
            
            # Find minimum distance to own cluster
            own_cluster_distances = np.min(distances, axis=1)
            
            # Find minimum distance to other clusters
            other_cluster_distances = np.partition(distances, 1, axis=1)[:, 1]
            
            # Calculate border score (closeness to other clusters relative to own cluster)
            border_scores = other_cluster_distances / (own_cluster_distances + 1e-10)
            
            # Select samples with high border scores
            threshold = np.percentile(border_scores, (1 - self.config.border_threshold) * 100)
            border_samples = np.where(border_scores > threshold)[0]
            
            # Limit number of samples to reallocate
            max_samples = int(len(cluster_labels) * self.config.max_reallocation_ratio)
            if len(border_samples) > max_samples:
                # Select samples with highest border scores
                top_indices = np.argsort(border_scores[border_samples])[-max_samples:]
                border_samples = border_samples[top_indices]
            
            return border_samples
            
        except Exception as e:
            logger.error(f"❌ Border sample selection failed: {e}")
            return np.array([])
    
    def _select_uncertain_samples(self, cluster_labels: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Select samples with uncertain cluster assignments."""
        try:
            # Calculate distances to cluster centers
            cluster_centers = self._calculate_cluster_centers(cluster_labels, features)
            
            if len(cluster_centers) == 0:
                return np.array([])
            
            # Calculate distances from each sample to all cluster centers
            distances = np.sqrt(((features[:, np.newaxis] - cluster_centers[np.newaxis, :]) ** 2).sum(axis=2))
            
            # Calculate uncertainty as ratio of second closest to closest distance
            sorted_distances = np.sort(distances, axis=1)
            uncertainty_scores = sorted_distances[:, 1] / (sorted_distances[:, 0] + 1e-10)
            
            # Select samples with high uncertainty
            threshold = np.percentile(uncertainty_scores, (1 - self.config.uncertainty_threshold) * 100)
            uncertain_samples = np.where(uncertainty_scores > threshold)[0]
            
            # Limit number of samples to reallocate
            max_samples = int(len(cluster_labels) * self.config.max_reallocation_ratio)
            if len(uncertain_samples) > max_samples:
                # Select samples with highest uncertainty scores
                top_indices = np.argsort(uncertainty_scores[uncertain_samples])[-max_samples:]
                uncertain_samples = uncertain_samples[top_indices]
            
            return uncertain_samples
            
        except Exception as e:
            logger.error(f"❌ Uncertain sample selection failed: {e}")
            return np.array([])
    
    def _reallocate_selected_samples(self, 
                                   cluster_labels: np.ndarray,
                                   features: np.ndarray,
                                   samples_to_reallocate: np.ndarray,
                                   target_metric: Optional[str] = None) -> np.ndarray:
        """Reallocate selected samples to improve clustering quality."""
        try:
            new_labels = cluster_labels.copy()
            
            for sample_idx in samples_to_reallocate:
                # Get current cluster
                current_cluster = cluster_labels[sample_idx]
                
                # Find best alternative cluster
                best_cluster = self._find_best_cluster_for_sample(
                    sample_idx, cluster_labels, features, target_metric
                )
                
                # Reallocate if different from current cluster
                if best_cluster != current_cluster:
                    new_labels[sample_idx] = best_cluster
            
            return new_labels
            
        except Exception as e:
            logger.error(f"❌ Sample reallocation failed: {e}")
            return cluster_labels
    
    def _find_best_cluster_for_sample(self, 
                                    sample_idx: int,
                                    cluster_labels: np.ndarray,
                                    features: np.ndarray,
                                    target_metric: Optional[str] = None) -> int:
        """Find the best cluster for a specific sample."""
        try:
            # Get unique clusters (excluding noise)
            unique_clusters = np.unique(cluster_labels)
            unique_clusters = unique_clusters[unique_clusters != -1]
            
            if len(unique_clusters) == 0:
                return cluster_labels[sample_idx]
            
            best_cluster = cluster_labels[sample_idx]
            best_quality = -np.inf
            
            # Test each cluster
            for cluster in unique_clusters:
                # Create temporary labels with sample assigned to this cluster
                temp_labels = cluster_labels.copy()
                temp_labels[sample_idx] = cluster
                
                # Calculate quality with this assignment
                quality = self._calculate_quality_metric(temp_labels, features, target_metric)
                
                if quality > best_quality:
                    best_quality = quality
                    best_cluster = cluster
            
            return best_cluster
            
        except Exception as e:
            logger.error(f"❌ Best cluster finding failed: {e}")
            return cluster_labels[sample_idx]
    
    def _calculate_cluster_centers(self, cluster_labels: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Calculate cluster centers."""
        try:
            unique_labels = np.unique(cluster_labels)
            unique_labels = unique_labels[unique_labels != -1]  # Remove noise
            
            if len(unique_labels) == 0:
                return np.array([])
            
            centers = []
            for label in unique_labels:
                mask = cluster_labels == label
                if mask.sum() > 0:
                    center = np.mean(features[mask], axis=0)
                    centers.append(center)
            
            return np.array(centers) if centers else np.array([])
            
        except Exception as e:
            logger.error(f"❌ Cluster center calculation failed: {e}")
            return np.array([])
    
    def _calculate_quality_metric(self, 
                                cluster_labels: np.ndarray, 
                                features: np.ndarray,
                                target_metric: Optional[str] = None) -> float:
        """Calculate clustering quality metric."""
        try:
            metric = target_metric or self.config.quality_metric
            
            # Remove noise points for validation
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                return -np.inf
            
            valid_labels = cluster_labels[valid_mask]
            valid_features = features[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return -np.inf
            
            # Calculate metric
            if metric == 'silhouette':
                return silhouette_score(valid_features, valid_labels)
            elif metric == 'calinski_harabasz':
                return calinski_harabasz_score(valid_features, valid_labels)
            elif metric == 'davies_bouldin':
                return -davies_bouldin_score(valid_features, valid_labels)  # Negative because lower is better
            else:
                return silhouette_score(valid_features, valid_labels)
            
        except Exception as e:
            logger.debug(f"Quality metric calculation failed: {e}")
            return -np.inf
    
    def _calculate_reallocation_stats(self, 
                                    original_labels: np.ndarray,
                                    optimized_labels: np.ndarray,
                                    features: np.ndarray,
                                    initial_quality: float,
                                    final_quality: float) -> Dict[str, Any]:
        """Calculate reallocation statistics."""
        try:
            # Count reallocated samples
            reallocated_samples = np.sum(original_labels != optimized_labels)
            reallocation_ratio = reallocated_samples / len(original_labels)
            
            # Calculate quality improvement
            quality_improvement = final_quality - initial_quality
            
            # Calculate cluster size changes
            original_sizes = self._calculate_cluster_sizes(original_labels)
            optimized_sizes = self._calculate_cluster_sizes(optimized_labels)
            
            # Calculate cluster count changes
            original_clusters = len(set(original_labels)) - (1 if -1 in original_labels else 0)
            optimized_clusters = len(set(optimized_labels)) - (1 if -1 in optimized_labels else 0)
            
            stats = {
                'reallocation_performed': True,
                'reallocated_samples': reallocated_samples,
                'reallocation_ratio': reallocation_ratio,
                'initial_quality': initial_quality,
                'final_quality': final_quality,
                'quality_improvement': quality_improvement,
                'original_clusters': original_clusters,
                'optimized_clusters': optimized_clusters,
                'cluster_count_change': optimized_clusters - original_clusters,
                'original_cluster_sizes': original_sizes,
                'optimized_cluster_sizes': optimized_sizes
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Reallocation stats calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_cluster_sizes(self, cluster_labels: np.ndarray) -> Dict[int, int]:
        """Calculate cluster sizes."""
        try:
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            return dict(zip(unique_labels, counts))
        except Exception as e:
            logger.debug(f"Cluster size calculation failed: {e}")
            return {}
    
    def get_reallocation_stats(self) -> Dict[str, Any]:
        """Get reallocation statistics."""
        return self.reallocation_stats.copy()
    
    def get_original_labels(self) -> Optional[np.ndarray]:
        """Get original cluster labels."""
        return self.original_labels.copy() if self.original_labels is not None else None
    
    def get_original_features(self) -> Optional[np.ndarray]:
        """Get original features."""
        return self.original_features.copy() if self.original_features is not None else None
    
    def _perform_regime_aware_reallocation(self, 
                                         cluster_labels: np.ndarray, 
                                         features: np.ndarray,
                                         target_metric: Optional[str] = None) -> np.ndarray:
        """Perform regime-aware sample reallocation."""
        try:
            logger.info("🎯 Starting regime-aware reallocation...")
            
            # Detect regimes
            regimes = self._detect_regimes(features)
            
            if regimes is not None and len(np.unique(regimes)) > 1:
                # Apply regime-aware reallocation
                optimized_labels = self._apply_regime_aware_reallocation(cluster_labels, features, regimes, target_metric)
            else:
                # Fall back to standard reallocation
                optimized_labels = self._perform_reallocation(cluster_labels, features, target_metric)
            
            return optimized_labels
            
        except Exception as e:
            logger.error(f"❌ Regime-aware reallocation failed: {e}")
            return self._perform_reallocation(cluster_labels, features, target_metric)
    
    def _detect_regimes(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Detect regimes in the feature data."""
        try:
            # Use first feature for regime detection
            primary_feature = features[:, 0]
            
            if self.config.regime_detection_method == 'variance':
                regimes = self._detect_regimes_by_variance(primary_feature)
            elif self.config.regime_detection_method == 'entropy':
                regimes = self._detect_regimes_by_entropy(primary_feature)
            elif self.config.regime_detection_method == 'volatility':
                regimes = self._detect_regimes_by_volatility(primary_feature)
            else:
                regimes = self._detect_regimes_by_variance(primary_feature)
            
            return regimes
            
        except Exception as e:
            logger.error(f"❌ Regime detection failed: {e}")
            return None
    
    def _detect_regimes_by_variance(self, feature: np.ndarray) -> np.ndarray:
        """Detect regimes based on variance changes."""
        try:
            window = self.config.regime_window
            threshold = self.config.regime_threshold
            
            regimes = np.zeros(len(feature))
            rolling_var = pd.Series(feature).rolling(window=window).var().values
            
            # Find variance change points
            var_changes = np.abs(np.diff(rolling_var)) > (threshold * np.nanmean(rolling_var))
            change_points = np.where(var_changes)[0]
            
            # Assign regime labels
            current_regime = 0
            for i in range(len(regimes)):
                if i in change_points:
                    current_regime = (current_regime + 1) % 3
                regimes[i] = current_regime
            
            return regimes
            
        except Exception as e:
            logger.debug(f"Variance-based regime detection failed: {e}")
            return np.zeros(len(feature))
    
    def _detect_regimes_by_entropy(self, feature: np.ndarray) -> np.ndarray:
        """Detect regimes based on entropy changes."""
        try:
            window = self.config.regime_window
            threshold = self.config.regime_threshold
            
            regimes = np.zeros(len(feature))
            
            # Calculate rolling entropy
            rolling_entropy = []
            for i in range(window, len(feature)):
                window_data = feature[i-window:i]
                # Discretize data
                hist, _ = np.histogram(window_data, bins=10)
                hist = hist / hist.sum()
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log2(hist))
                rolling_entropy.append(entropy)
            
            rolling_entropy = np.array(rolling_entropy)
            
            # Find entropy change points
            entropy_changes = np.abs(np.diff(rolling_entropy)) > (threshold * np.std(rolling_entropy))
            change_points = np.where(entropy_changes)[0] + window
            
            # Assign regime labels
            current_regime = 0
            for i in range(len(regimes)):
                if i in change_points:
                    current_regime = (current_regime + 1) % 3
                regimes[i] = current_regime
            
            return regimes
            
        except Exception as e:
            logger.debug(f"Entropy-based regime detection failed: {e}")
            return np.zeros(len(feature))
    
    def _detect_regimes_by_volatility(self, feature: np.ndarray) -> np.ndarray:
        """Detect regimes based on volatility changes."""
        try:
            window = self.config.regime_window
            threshold = self.config.regime_threshold
            
            regimes = np.zeros(len(feature))
            rolling_vol = pd.Series(feature).rolling(window=window).std().values
            
            # Find volatility change points
            vol_changes = np.abs(np.diff(rolling_vol)) > (threshold * np.nanmean(rolling_vol))
            change_points = np.where(vol_changes)[0]
            
            # Assign regime labels
            current_regime = 0
            for i in range(len(regimes)):
                if i in change_points:
                    current_regime = (current_regime + 1) % 3
                regimes[i] = current_regime
            
            return regimes
            
        except Exception as e:
            logger.debug(f"Volatility-based regime detection failed: {e}")
            return np.zeros(len(feature))
    
    def _apply_regime_aware_reallocation(self, 
                                       cluster_labels: np.ndarray, 
                                       features: np.ndarray, 
                                       regimes: np.ndarray,
                                       target_metric: Optional[str] = None) -> np.ndarray:
        """Apply regime-aware reallocation."""
        try:
            unique_regimes = np.unique(regimes)
            optimized_labels = cluster_labels.copy()
            
            # Process each regime separately
            for regime in unique_regimes:
                regime_mask = regimes == regime
                regime_cluster_labels = cluster_labels[regime_mask]
                regime_features = features[regime_mask]
                
                if len(regime_cluster_labels) > 1:
                    # Apply regime-specific reallocation
                    regime_optimized = self._reallocate_regime_samples(
                        regime_cluster_labels, regime_features, regime, target_metric
                    )
                    optimized_labels[regime_mask] = regime_optimized
            
            # Apply regime transition awareness
            if self.config.regime_transition_aware:
                optimized_labels = self._apply_transition_aware_reallocation(
                    optimized_labels, features, regimes
                )
            
            return optimized_labels
            
        except Exception as e:
            logger.error(f"❌ Regime-aware reallocation application failed: {e}")
            return cluster_labels
    
    def _reallocate_regime_samples(self, 
                                 cluster_labels: np.ndarray, 
                                 features: np.ndarray, 
                                 regime: int,
                                 target_metric: Optional[str] = None) -> np.ndarray:
        """Reallocate samples within a specific regime."""
        try:
            # Use regime-specific quality metrics
            if self.config.regime_specific_quality_metrics:
                regime_metric = self._get_regime_specific_metric(regime, target_metric)
            else:
                regime_metric = target_metric
            
            # Apply standard reallocation within regime
            current_labels = cluster_labels.copy()
            best_labels = current_labels.copy()
            best_quality = self._calculate_quality_metric(current_labels, features, regime_metric)
            
            iteration = 0
            improvement = float('inf')
            
            while iteration < self.config.max_iterations and improvement > self.config.convergence_threshold:
                iteration += 1
                
                # Select samples for reallocation within regime
                samples_to_reallocate = self._select_regime_samples_for_reallocation(
                    current_labels, features, regime
                )
                
                if len(samples_to_reallocate) == 0:
                    break
                
                # Reallocate selected samples
                new_labels = self._reallocate_selected_samples(
                    current_labels, features, samples_to_reallocate, regime_metric
                )
                
                # Calculate new quality
                new_quality = self._calculate_quality_metric(new_labels, features, regime_metric)
                
                # Check for improvement
                improvement = new_quality - best_quality
                
                if improvement > self.config.min_improvement:
                    best_labels = new_labels.copy()
                    best_quality = new_quality
                    current_labels = new_labels
            
            return best_labels
            
        except Exception as e:
            logger.error(f"❌ Regime sample reallocation failed: {e}")
            return cluster_labels
    
    def _get_regime_specific_metric(self, regime: int, target_metric: Optional[str] = None) -> str:
        """Get regime-specific quality metric."""
        try:
            # Define regime-specific metrics based on regime characteristics
            if regime == 0:  # Low volatility regime
                return 'silhouette'  # Focus on cluster separation
            elif regime == 1:  # Medium volatility regime
                return target_metric or 'calinski_harabasz'  # Balanced metric
            else:  # High volatility regime
                return 'davies_bouldin'  # Focus on cluster compactness
            
        except Exception as e:
            logger.debug(f"Regime-specific metric selection failed: {e}")
            return target_metric or 'silhouette'
    
    def _select_regime_samples_for_reallocation(self, 
                                              cluster_labels: np.ndarray, 
                                              features: np.ndarray, 
                                              regime: int) -> np.ndarray:
        """Select samples for reallocation within a regime."""
        try:
            # Use regime-specific selection strategy
            if regime == 0:  # Low volatility regime
                # Focus on border samples
                return self._select_border_samples(cluster_labels, features)
            elif regime == 1:  # Medium volatility regime
                # Use uncertain samples
                return self._select_uncertain_samples(cluster_labels, features)
            else:  # High volatility regime
                # Use all samples (more aggressive reallocation)
                return np.arange(len(cluster_labels))
            
        except Exception as e:
            logger.error(f"❌ Regime sample selection failed: {e}")
            return np.array([])
    
    def _apply_transition_aware_reallocation(self, 
                                           cluster_labels: np.ndarray, 
                                           features: np.ndarray, 
                                           regimes: np.ndarray) -> np.ndarray:
        """Apply transition-aware reallocation."""
        try:
            optimized_labels = cluster_labels.copy()
            
            # Find regime transition points
            regime_changes = np.diff(regimes) != 0
            transition_points = np.where(regime_changes)[0]
            
            # Apply smoothing around transition points
            for transition_point in transition_points:
                # Define smoothing window around transition
                window_size = min(10, len(cluster_labels) // 10)
                start_idx = max(0, transition_point - window_size)
                end_idx = min(len(cluster_labels), transition_point + window_size)
                
                # Apply majority vote smoothing in transition window
                window_labels = cluster_labels[start_idx:end_idx]
                if len(window_labels) > 0:
                    unique_labels, counts = np.unique(window_labels, return_counts=True)
                    most_common_label = unique_labels[np.argmax(counts)]
                    optimized_labels[start_idx:end_idx] = most_common_label
            
            return optimized_labels
            
        except Exception as e:
            logger.error(f"❌ Transition-aware reallocation failed: {e}")
            return cluster_labels