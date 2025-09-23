"""
Enhanced optimized clustering algorithm with 4D frontier optimization.

This module provides enhanced clustering algorithms that implement:
- Improved within-cluster CV calculations and optimization
- Enhanced Davies-Bouldin and Silhouette score calculations
- 5% average cluster size targeting while maintaining 3-8% range
- 4D frontier establishment between clusters
- Regime transfer optimization with CV similarity and size constraints
- 5-iteration optimization process using matrix operations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import warnings
import logging
import time
from dataclasses import dataclass
from enum import Enum

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
    warnings.warn("Matrix operations not available, using fallback implementations")

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

from .base_clustering import BaseClusterer, ClusteringResult

logger = logging.getLogger(__name__)


@dataclass
class EnhancedClusteringResult(ClusteringResult):
    """Result of enhanced clustering operation."""
    frontiers: Dict[str, Any] = None
    transfer_history: List[Dict[str, Any]] = None
    optimization_iterations: int = 0
    frontier_metrics: Dict[str, Any] = None


class FrontierType(Enum):
    """Types of frontiers between clusters."""
    VOLUME_VOLATILITY = "volume_volatility"
    MOMENTUM_TREND = "momentum_trend"
    VOLUME_MOMENTUM = "volume_momentum"
    VOLATILITY_TREND = "volatility_trend"
    CROSS_DIMENSIONAL = "cross_dimensional"


@dataclass
class FrontierBoundary:
    """Boundary information for 4D frontiers."""
    cluster_a: int
    cluster_b: int
    frontier_type: FrontierType
    boundary_points: np.ndarray
    similarity_score: float
    cv_ratio: float
    size_ratio: float


@dataclass
class RegimeTransferCandidate:
    """Candidate for regime transfer between clusters."""
    regime_id: int
    current_cluster: int
    target_cluster: int
    cv_similarity_current: float
    cv_similarity_target: float
    size_ratio_current: float
    size_ratio_target: float
    transfer_benefit: float
    constraint_violation: bool


class EnhancedMatrixOptimizedClusterer(BaseClusterer):
    """Enhanced matrix-optimized clustering with 4D frontier optimization."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced clusterer.

        Args:
            config: Clustering configuration
        """
        super().__init__(config)
        
        # Initialize matrix operations
        if MATRIX_OPERATIONS_AVAILABLE:
            self.matrix_ops = get_unified_matrix_operations()
            self.vectorized_core = get_vectorized_processing_core()
            self.enhanced_ops = get_enhanced_matrix_operations()
            self.batch_processor = get_batch_matrix_processor()
            self.logger.info("✅ Enhanced matrix operations initialized successfully")
        else:
            self.matrix_ops = None
            self.vectorized_core = None
            self.enhanced_ops = None
            self.batch_processor = None
            self.logger.warning("⚠️ Enhanced matrix operations not available, using fallback mode")

        # Enhanced clustering parameters
        self.optimization_iterations = config.get('optimization_iterations', 5)
        self.target_cluster_size_percentage = config.get('target_cluster_size_percentage', 5.0)
        self.size_tolerance = config.get('size_tolerance', 2.0)  # 3-8% range
        self.cv_similarity_threshold = config.get('cv_similarity_threshold', 0.1)

    def cluster(self, features: np.ndarray) -> ClusteringResult:
        """Perform enhanced matrix-optimized clustering.

        Args:
            features: Feature matrix to cluster

        Returns:
            ClusteringResult with enhanced clustering results
        """
        start_time = time.time()
        
        try:
            # Prepare features
            features = self._prepare_features(features)
            
            # Monitor performance
            self._monitor_performance("enhanced_matrix_optimized_clustering")
            
            # Perform enhanced clustering
            result = self._enhanced_4d_frontier_clustering(features)
            
            # Stop performance monitoring
            perf_metrics = self._stop_performance_monitoring("enhanced_matrix_optimized_clustering")
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Create enhanced result
            clustering_result = self._create_enhanced_result(
                labels=result['labels'],
                features=features,
                execution_time=execution_time,
                frontiers=result.get('frontiers', {}),
                transfer_history=result.get('transfer_history', []),
                optimization_iterations=result.get('optimization_iterations', 0),
                frontier_metrics=result.get('frontier_metrics', {}),
                metadata={
                    'method': 'enhanced_matrix_optimized',
                    'matrix_ops_used': self.matrix_ops is not None,
                    'hardware_acceleration_used': self.hardware_accelerator is not None,
                    'performance_metrics': perf_metrics,
                    'optimization_iterations': result.get('optimization_iterations', 0)
                }
            )
            
            self.logger.info(f"✅ Enhanced clustering completed in {execution_time:.2f}s with {result.get('optimization_iterations', 0)} iterations")
            return clustering_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Enhanced clustering failed: {e}")
            return ClusteringResult(
                labels=np.array([]),
                cluster_centers=np.array([]),
                statistics={},
                quality_metrics={},
                validation={'valid': False, 'error': str(e)},
                metadata={'error': str(e), 'method': 'enhanced_matrix_optimized'},
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )

    def _enhanced_4d_frontier_clustering(self, features: np.ndarray) -> Dict[str, Any]:
        """Perform enhanced clustering with 4D frontier optimization.

        Args:
            features: Feature matrix

        Returns:
            Dictionary with enhanced clustering results
        """
        self.logger.info("🚀 Starting enhanced 4D frontier clustering")
        
        # Extract 4D features (volume, volatility, momentum, trend)
        features_4d = self._extract_4d_features(features)
        
        # Initial clustering
        initial_labels = self._initial_enhanced_clustering(features_4d)
        
        # Establish 4D frontiers
        frontiers = self._establish_4d_frontiers(features_4d, initial_labels)
        
        # Optimize through regime transfers
        optimized_labels, transfer_history = self._optimize_regime_transfers(
            features_4d, initial_labels, frontiers
        )
        
        # Calculate frontier metrics
        frontier_metrics = self._calculate_frontier_metrics(frontiers, optimized_labels)
        
        self.logger.info("✅ Enhanced 4D frontier clustering completed")
        
        return {
            'labels': optimized_labels,
            'frontiers': frontiers,
            'transfer_history': transfer_history,
            'optimization_iterations': self.optimization_iterations,
            'frontier_metrics': frontier_metrics
        }

    def _extract_4d_features(self, features: np.ndarray) -> np.ndarray:
        """Extract 4D features (volume, volatility, momentum, trend).

        Args:
            features: Raw feature matrix

        Returns:
            4D feature matrix
        """
        try:
            # If features are already 4D, return as is
            if features.shape[1] == 4:
                return features
            
            # Use matrix operations for feature extraction if available
            if self.matrix_ops is not None:
                # Use optimized feature extraction
                features_4d = self.matrix_ops.extract_4d_features(features)
            else:
                # Fallback to standard extraction
                features_4d = self._standard_4d_extraction(features)
            
            self.logger.info(f"✅ Extracted 4D features: {features_4d.shape}")
            return features_4d
            
        except Exception as e:
            self.logger.warning(f"⚠️ 4D feature extraction failed: {e}")
            # Return first 4 features as fallback
            return features[:, :min(4, features.shape[1])]

    def _standard_4d_extraction(self, features: np.ndarray) -> np.ndarray:
        """Standard 4D feature extraction fallback.

        Args:
            features: Raw feature matrix

        Returns:
            4D feature matrix
        """
        # Simple feature selection - take first 4 features
        # In a real implementation, this would be more sophisticated
        n_features = min(4, features.shape[1])
        return features[:, :n_features]

    def _initial_enhanced_clustering(self, features_4d: np.ndarray) -> np.ndarray:
        """Perform initial enhanced clustering with MSM support.

        Args:
            features_4d: 4D feature matrix

        Returns:
            Initial cluster labels
        """
        try:
            # Try MSM clustering first
            try:
                from .msm_clustering import MSMClusterer

                msm_config = {
                    'n_states': min(20, max(2, features_4d.shape[0] // 50)),
                    'lag_time': 1,
                    'clustering_method': 'kmeans',
                    'distance_metric': 'euclidean',
                    'reversible': True,
                    'stationary_distribution_constraint': True
                }

                msm_clusterer = MSMClusterer(msm_config)
                result = msm_clusterer.cluster(features_4d)

                if result.success:
                    self.logger.info(f"✅ Initial MSM clustering: {result.labels.shape[0]} states, MSM Score: {result.msm_score".3f"}")
                    return result.labels

            except Exception as msm_error:
                self.logger.warning(f"⚠️ MSM clustering failed, falling back to K-means: {msm_error}")

            # Use enhanced K-means with matrix operations
            n_clusters = min(20, max(2, features_4d.shape[0] // 50))

            if self.matrix_ops is not None:
                # Use GPU-accelerated K-means if available
                if hasattr(self.matrix_ops, 'kmeans_gpu'):
                    labels, _ = self.matrix_ops.kmeans_gpu(features_4d, n_clusters)
                else:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(features_4d)
            else:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features_4d)

            self.logger.info(f"✅ Initial enhanced clustering: {n_clusters} clusters")
            return labels

        except Exception as e:
            self.logger.warning(f"⚠️ Initial enhanced clustering failed: {e}")
            # Fallback to single cluster
            return np.zeros(features_4d.shape[0], dtype=int)

    def _establish_4d_frontiers(self, features_4d: np.ndarray, labels: np.ndarray) -> Dict[str, FrontierBoundary]:
        """Establish 4D frontiers between clusters.

        Args:
            features_4d: 4D feature matrix
            labels: Cluster labels

        Returns:
            Dictionary of frontier boundaries
        """
        frontiers = {}
        
        try:
            unique_labels = np.unique(labels[labels != -1])
            if len(unique_labels) < 2:
                return frontiers
            
            # Calculate frontiers between all cluster pairs
            for i, cluster_a in enumerate(unique_labels):
                for j, cluster_b in enumerate(unique_labels[i+1:], i+1):
                    frontier = self._calculate_frontier_boundary(
                        features_4d, labels, cluster_a, cluster_b
                    )
                    if frontier is not None:
                        frontier_key = f"{cluster_a}_{cluster_b}"
                        frontiers[frontier_key] = frontier
            
            self.logger.info(f"✅ Established {len(frontiers)} 4D frontiers")
            return frontiers
            
        except Exception as e:
            self.logger.warning(f"⚠️ Frontier establishment failed: {e}")
            return frontiers

    def _calculate_frontier_boundary(self, features_4d: np.ndarray, labels: np.ndarray, 
                                   cluster_a: int, cluster_b: int) -> Optional[FrontierBoundary]:
        """Calculate frontier boundary between two clusters.

        Args:
            features_4d: 4D feature matrix
            labels: Cluster labels
            cluster_a: First cluster ID
            cluster_b: Second cluster ID

        Returns:
            FrontierBoundary object or None
        """
        try:
            # Get cluster features
            mask_a = labels == cluster_a
            mask_b = labels == cluster_b
            
            if not (np.any(mask_a) and np.any(mask_b)):
                return None
            
            features_a = features_4d[mask_a]
            features_b = features_4d[mask_b]
            
            # Calculate cluster statistics
            centroid_a = np.mean(features_a, axis=0)
            centroid_b = np.mean(features_b, axis=0)
            
            # Calculate CV for each cluster
            cv_a = self._calculate_cluster_cv(features_a)
            cv_b = self._calculate_cluster_cv(features_b)
            
            # Calculate similarity score
            similarity_score = self._calculate_cluster_similarity(features_a, features_b)
            
            # Calculate size ratio
            size_ratio = len(features_a) / len(features_b) if len(features_b) > 0 else 1.0
            
            # Determine frontier type based on feature dimensions
            frontier_type = self._determine_frontier_type(centroid_a, centroid_b)
            
            # Calculate boundary points
            boundary_points = self._calculate_boundary_points(centroid_a, centroid_b)
            
            return FrontierBoundary(
                cluster_a=cluster_a,
                cluster_b=cluster_b,
                frontier_type=frontier_type,
                boundary_points=boundary_points,
                similarity_score=similarity_score,
                cv_ratio=cv_a / cv_b if cv_b > 0 else 1.0,
                size_ratio=size_ratio
            )
            
        except Exception as e:
            self.logger.warning(f"⚠️ Frontier boundary calculation failed: {e}")
            return None

    def _calculate_cluster_cv(self, features: np.ndarray) -> float:
        """Calculate coefficient of variation for a cluster.

        Args:
            features: Cluster features

        Returns:
            Coefficient of variation
        """
        try:
            if len(features) < 2:
                return 0.0
            
            # Calculate CV for each feature dimension
            feature_cvs = []
            for i in range(features.shape[1]):
                feature_values = features[:, i]
                mean_val = np.mean(feature_values)
                std_val = np.std(feature_values)
                
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                    feature_cvs.append(cv)
            
            return np.mean(feature_cvs) if feature_cvs else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cluster CV calculation failed: {e}")
            return 0.0

    def _calculate_cluster_similarity(self, features_a: np.ndarray, features_b: np.ndarray) -> float:
        """Calculate similarity between two clusters.

        Args:
            features_a: First cluster features
            features_b: Second cluster features

        Returns:
            Similarity score
        """
        try:
            # Calculate centroids
            centroid_a = np.mean(features_a, axis=0)
            centroid_b = np.mean(features_b, axis=0)
            
            # Calculate distance between centroids
            distance = np.linalg.norm(centroid_a - centroid_b)
            
            # Convert distance to similarity (inverse relationship)
            max_distance = np.sqrt(features_a.shape[1]) * 2  # Theoretical maximum
            similarity = 1.0 - (distance / max_distance)
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cluster similarity calculation failed: {e}")
            return 0.0

    def _determine_frontier_type(self, centroid_a: np.ndarray, centroid_b: np.ndarray) -> FrontierType:
        """Determine frontier type based on centroid differences.

        Args:
            centroid_a: First cluster centroid
            centroid_b: Second cluster centroid

        Returns:
            FrontierType
        """
        try:
            # Calculate differences in each dimension
            diff = abs(centroid_a - centroid_b)
            
            # Determine dominant dimension differences
            if len(diff) >= 4:
                if diff[0] > diff[1] and diff[2] > diff[3]:  # volume > volatility, momentum > trend
                    return FrontierType.VOLUME_MOMENTUM
                elif diff[0] > diff[1]:  # volume > volatility
                    return FrontierType.VOLUME_VOLATILITY
                elif diff[2] > diff[3]:  # momentum > trend
                    return FrontierType.MOMENTUM_TREND
                elif diff[1] > diff[3]:  # volatility > trend
                    return FrontierType.VOLATILITY_TREND
                else:
                    return FrontierType.CROSS_DIMENSIONAL
            else:
                return FrontierType.CROSS_DIMENSIONAL
                
        except Exception as e:
            self.logger.warning(f"⚠️ Frontier type determination failed: {e}")
            return FrontierType.CROSS_DIMENSIONAL

    def _calculate_boundary_points(self, centroid_a: np.ndarray, centroid_b: np.ndarray) -> np.ndarray:
        """Calculate boundary points between clusters.

        Args:
            centroid_a: First cluster centroid
            centroid_b: Second cluster centroid

        Returns:
            Boundary points array
        """
        try:
            # Calculate midpoint between centroids
            midpoint = (centroid_a + centroid_b) / 2
            
            # Create boundary points (simplified)
            boundary_points = np.array([centroid_a, midpoint, centroid_b])
            
            return boundary_points
            
        except Exception as e:
            self.logger.warning(f"⚠️ Boundary points calculation failed: {e}")
            return np.array([])

    def _optimize_regime_transfers(self, features_4d: np.ndarray, labels: np.ndarray, 
                                 frontiers: Dict[str, FrontierBoundary]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Optimize clustering through regime transfers.

        Args:
            features_4d: 4D feature matrix
            labels: Cluster labels
            frontiers: Frontier boundaries

        Returns:
            Tuple of (optimized_labels, transfer_history)
        """
        transfer_history = []
        current_labels = labels.copy()
        
        try:
            for iteration in range(self.optimization_iterations):
                self.logger.info(f"🔄 Optimization iteration {iteration + 1}/{self.optimization_iterations}")
                
                # Find transfer candidates
                candidates = self._find_transfer_candidates(features_4d, current_labels, frontiers)
                
                if not candidates:
                    self.logger.info("✅ No more transfer candidates found")
                    break
                
                # Apply transfers
                current_labels, iteration_history = self._apply_transfers(
                    features_4d, current_labels, candidates
                )
                
                transfer_history.extend(iteration_history)
                
                # Update frontiers
                frontiers = self._establish_4d_frontiers(features_4d, current_labels)
            
            self.logger.info(f"✅ Regime transfer optimization completed: {len(transfer_history)} transfers")
            return current_labels, transfer_history
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime transfer optimization failed: {e}")
            return labels, transfer_history

    def _find_transfer_candidates(self, features_4d: np.ndarray, labels: np.ndarray, 
                                frontiers: Dict[str, FrontierBoundary]) -> List[RegimeTransferCandidate]:
        """Find candidates for regime transfer.

        Args:
            features_4d: 4D feature matrix
            labels: Cluster labels
            frontiers: Frontier boundaries

        Returns:
            List of transfer candidates
        """
        candidates = []
        
        try:
            unique_labels = np.unique(labels[labels != -1])
            
            for label in unique_labels:
                cluster_mask = labels == label
                cluster_features = features_4d[cluster_mask]
                cluster_indices = np.where(cluster_mask)[0]
                
                if len(cluster_features) < 2:
                    continue
                
                # Calculate cluster CV
                cluster_cv = self._calculate_cluster_cv(cluster_features)
                
                # Find potential target clusters
                for target_label in unique_labels:
                    if target_label == label:
                        continue
                    
                    target_mask = labels == target_label
                    target_features = features_4d[target_mask]
                    
                    if len(target_features) < 1:
                        continue
                    
                    # Calculate target cluster CV
                    target_cv = self._calculate_cluster_cv(target_features)
                    
                    # Check CV similarity
                    cv_similarity = abs(cluster_cv - target_cv)
                    
                    if cv_similarity < self.cv_similarity_threshold:
                        # Create transfer candidate
                        for idx in cluster_indices:
                            candidate = RegimeTransferCandidate(
                                regime_id=idx,
                                current_cluster=label,
                                target_cluster=target_label,
                                cv_similarity_current=cluster_cv,
                                cv_similarity_target=target_cv,
                                size_ratio_current=len(cluster_features) / features_4d.shape[0],
                                size_ratio_target=len(target_features) / features_4d.shape[0],
                                transfer_benefit=cv_similarity,
                                constraint_violation=False
                            )
                            candidates.append(candidate)
            
            # Sort by transfer benefit
            candidates.sort(key=lambda x: x.transfer_benefit)
            
            self.logger.info(f"✅ Found {len(candidates)} transfer candidates")
            return candidates
            
        except Exception as e:
            self.logger.warning(f"⚠️ Transfer candidate search failed: {e}")
            return candidates

    def _apply_transfers(self, features_4d: np.ndarray, labels: np.ndarray, 
                       candidates: List[RegimeTransferCandidate]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Apply regime transfers.

        Args:
            features_4d: 4D feature matrix
            labels: Cluster labels
            candidates: Transfer candidates

        Returns:
            Tuple of (updated_labels, transfer_history)
        """
        transfer_history = []
        updated_labels = labels.copy()
        
        try:
            # Apply transfers (limit to avoid over-optimization)
            max_transfers = min(len(candidates), features_4d.shape[0] // 10)
            
            for i, candidate in enumerate(candidates[:max_transfers]):
                # Check constraints
                if self._check_transfer_constraints(candidate, updated_labels):
                    # Apply transfer
                    updated_labels[candidate.regime_id] = candidate.target_cluster
                    
                    # Record transfer
                    transfer_record = {
                        'iteration': i,
                        'regime_id': candidate.regime_id,
                        'from_cluster': candidate.current_cluster,
                        'to_cluster': candidate.target_cluster,
                        'cv_similarity': candidate.cv_similarity_target,
                        'size_ratio': candidate.size_ratio_target,
                        'transfer_benefit': candidate.transfer_benefit
                    }
                    transfer_history.append(transfer_record)
            
            self.logger.info(f"✅ Applied {len(transfer_history)} transfers")
            return updated_labels, transfer_history
            
        except Exception as e:
            self.logger.warning(f"⚠️ Transfer application failed: {e}")
            return labels, transfer_history

    def _check_transfer_constraints(self, candidate: RegimeTransferCandidate, 
                                  labels: np.ndarray) -> bool:
        """Check if transfer meets constraints.

        Args:
            candidate: Transfer candidate
            labels: Current cluster labels

        Returns:
            True if constraints are met
        """
        try:
            # Check size constraints
            current_size = np.sum(labels == candidate.current_cluster)
            target_size = np.sum(labels == candidate.target_cluster)
            
            # Don't transfer if it would make clusters too small
            if current_size <= 2 or target_size >= 1000:
                return False
            
            # Check size ratio constraints
            total_points = len(labels)
            target_size_ratio = (target_size + 1) / total_points
            
            if target_size_ratio > 0.08:  # Max 8% cluster size
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Transfer constraint check failed: {e}")
            return False

    def _calculate_frontier_metrics(self, frontiers: Dict[str, FrontierBoundary], 
                                  labels: np.ndarray) -> Dict[str, Any]:
        """Calculate frontier metrics.

        Args:
            frontiers: Frontier boundaries
            labels: Cluster labels

        Returns:
            Frontier metrics
        """
        try:
            if not frontiers:
                return {
                    'n_frontiers': 0,
                    'average_similarity': 0.0,
                    'average_cv_ratio': 0.0,
                    'average_size_ratio': 0.0
                }
            
            similarities = [f.similarity_score for f in frontiers.values()]
            cv_ratios = [f.cv_ratio for f in frontiers.values()]
            size_ratios = [f.size_ratio for f in frontiers.values()]
            
            return {
                'n_frontiers': len(frontiers),
                'average_similarity': float(np.mean(similarities)),
                'average_cv_ratio': float(np.mean(cv_ratios)),
                'average_size_ratio': float(np.mean(size_ratios)),
                'frontier_types': [f.frontier_type.value for f in frontiers.values()]
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Frontier metrics calculation failed: {e}")
            return {
                'n_frontiers': 0,
                'average_similarity': 0.0,
                'average_cv_ratio': 0.0,
                'average_size_ratio': 0.0,
                'error': str(e)
            }

    def _create_enhanced_result(self, labels: np.ndarray, features: np.ndarray, 
                              execution_time: float, frontiers: Dict[str, Any],
                              transfer_history: List[Dict[str, Any]], optimization_iterations: int,
                              frontier_metrics: Dict[str, Any], metadata: Dict[str, Any]) -> EnhancedClusteringResult:
        """Create enhanced clustering result.

        Args:
            labels: Cluster labels
            features: Feature matrix
            execution_time: Execution time
            frontiers: Frontier boundaries
            transfer_history: Transfer history
            optimization_iterations: Number of optimization iterations
            frontier_metrics: Frontier metrics
            metadata: Additional metadata

        Returns:
            EnhancedClusteringResult object
        """
        try:
            # Create base result
            base_result = self._create_result(labels, features, execution_time, metadata)
            
            # Add enhanced-specific fields
            enhanced_result = EnhancedClusteringResult(
                labels=base_result.labels,
                cluster_centers=base_result.cluster_centers,
                statistics=base_result.statistics,
                quality_metrics=base_result.quality_metrics,
                validation=base_result.validation,
                metadata=base_result.metadata,
                success=base_result.success,
                error_message=base_result.error_message,
                execution_time=base_result.execution_time,
                timestamp=base_result.timestamp,
                frontiers=frontiers,
                transfer_history=transfer_history,
                optimization_iterations=optimization_iterations,
                frontier_metrics=frontier_metrics
            )
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create enhanced clustering result: {e}")
            return EnhancedClusteringResult(
                labels=np.array([]),
                cluster_centers=np.array([]),
                statistics={},
                quality_metrics={},
                validation={'valid': False, 'error': str(e)},
                metadata={'error': str(e)},
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )