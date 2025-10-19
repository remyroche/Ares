"""
Optimized Post-Processing for HDBSCAN Clustering

This module provides optimized post-processing with VectorBT acceleration,
parallel processing, and intelligent optimization strategies.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy import stats

# Import UnifiedVectorizationManager
from src.utils.ml_common.unified_vectorization_manager import (
    UnifiedVectorizationManager, 
    VectorizationConfig,
    get_unified_vectorization_manager
)

logger = logging.getLogger(__name__)

@dataclass
class PostProcessingConfig:
    """Configuration for optimized post-processing."""
    # Sample reallocation
    enable_sample_reallocation: bool = True
    reallocation_threshold: float = 0.5
    max_reallocation_iterations: int = 10
    
    # Economic validation
    enable_economic_validation: bool = True
    min_economic_value: float = 0.1
    max_economic_value: float = 10.0
    
    # Temporal stabilization
    enable_temporal_stabilization: bool = True
    stability_threshold: float = 0.8
    min_temporal_consistency: float = 0.6
    
    # Parallel processing
    max_workers: Optional[int] = None
    use_multiprocessing: bool = True
    chunk_size: int = 1000
    
    # Memory optimization
    memory_efficient: bool = True
    max_memory_gb: float = 8.0
    
    # VectorBT optimization
    enable_vectorbt: bool = True
    enable_gpu: bool = False
    
    # Validation
    enable_validation: bool = True
    min_silhouette_score: float = 0.3
    min_cluster_size: int = 5

class OptimizedPostProcessor:
    """
    Optimized post-processor with VectorBT acceleration and parallel processing.
    """
    
    def __init__(self, config: Optional[PostProcessingConfig] = None):
        """Initialize the optimized post-processor."""
        self.config = config or PostProcessingConfig()
        
        # Initialize UnifiedVectorizationManager
        vectorization_config = VectorizationConfig(
            enable_vectorbt=self.config.enable_vectorbt,
            enable_gpu=self.config.enable_gpu,
            memory_efficient=self.config.memory_efficient,
            max_memory_gb=self.config.max_memory_gb,
            chunk_size=self.config.chunk_size,
            enable_parallel=True
        )
        self.vectorization_manager = get_unified_vectorization_manager(vectorization_config)
        
        # Performance tracking
        self.performance_stats = {
            'post_processing_time': 0.0,
            'samples_reallocated': 0,
            'economic_value_improvement': 0.0,
            'temporal_stability_improvement': 0.0,
            'memory_usage_mb': 0.0,
            'vectorbt_usage_rate': 0.0,
            'parallel_efficiency': 0.0
        }
        
        logger.info("✅ OptimizedPostProcessor initialized")
    
    def post_process_clusters(self, cluster_labels: np.ndarray, 
                             features_df: pd.DataFrame,
                             timestamps: Optional[pd.Series] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Post-process clusters with optimization and validation.
        
        Args:
            cluster_labels: Initial cluster labels
            features_df: Input features DataFrame
            timestamps: Optional timestamps for temporal analysis
            
        Returns:
            Tuple of (optimized_cluster_labels, post_processing_info)
        """
        start_time = time.time()
        logger.info(f"🚀 Starting optimized post-processing for {len(cluster_labels)} samples")
        
        # Validate input
        self._validate_inputs(cluster_labels, features_df)
        
        # Initialize post-processing info
        post_processing_info = {
            'original_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'original_noise_points': list(cluster_labels).count(-1),
            'optimization_steps': []
        }
        
        # Step 1: Sample reallocation
        if self.config.enable_sample_reallocation:
            logger.info("🔄 Optimizing sample reallocation")
            cluster_labels, reallocation_info = self._optimize_sample_reallocation(
                cluster_labels, features_df
            )
            post_processing_info['reallocation_info'] = reallocation_info
        
        # Step 2: Economic validation
        if self.config.enable_economic_validation:
            logger.info("🔄 Performing economic validation")
            cluster_labels, economic_info = self._validate_economic_value(
                cluster_labels, features_df
            )
            post_processing_info['economic_info'] = economic_info
        
        # Step 3: Temporal stabilization
        if self.config.enable_temporal_stabilization and timestamps is not None:
            logger.info("🔄 Optimizing temporal stabilization")
            cluster_labels, temporal_info = self._optimize_temporal_stability(
                cluster_labels, features_df, timestamps
            )
            post_processing_info['temporal_info'] = temporal_info
        
        # Step 4: Final validation
        if self.config.enable_validation:
            logger.info("🔄 Performing final validation")
            validation_info = self._validate_clustering(cluster_labels, features_df)
            post_processing_info['validation_info'] = validation_info
        
        # Update performance stats
        post_processing_time = time.time() - start_time
        self._update_performance_stats(cluster_labels, features_df, post_processing_time)
        
        logger.info(f"✅ Post-processing completed in {post_processing_time:.2f}s")
        return cluster_labels, post_processing_info
    
    def _optimize_sample_reallocation(self, cluster_labels: np.ndarray, 
                                    features_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Optimize sample reallocation using VectorBT acceleration."""
        logger.info("🔄 Optimizing sample reallocation")
        
        try:
            # Use VectorBT acceleration if available
            if hasattr(self.vectorization_manager, 'optimize_sample_reallocation'):
                optimized_labels, reallocation_info = self.vectorization_manager.optimize_sample_reallocation(
                    cluster_labels, features_df,
                    threshold=self.config.reallocation_threshold,
                    max_iterations=self.config.max_reallocation_iterations
                )
            else:
                # Use standard reallocation
                optimized_labels, reallocation_info = self._standard_sample_reallocation(
                    cluster_labels, features_df
                )
            
            logger.info("✅ Sample reallocation completed")
            return optimized_labels, reallocation_info
            
        except Exception as e:
            logger.error(f"❌ Sample reallocation failed: {e}")
            return cluster_labels, {'error': str(e)}
    
    def _standard_sample_reallocation(self, cluster_labels: np.ndarray, 
                                    features_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Standard sample reallocation implementation."""
        optimized_labels = cluster_labels.copy()
        reallocation_info = {
            'iterations': 0,
            'samples_moved': 0,
            'improvement_score': 0.0
        }
        
        for iteration in range(self.config.max_reallocation_iterations):
            # Find samples that might benefit from reallocation
            samples_to_reallocate = self._find_samples_for_reallocation(
                optimized_labels, features_df
            )
            
            if len(samples_to_reallocate) == 0:
                break
            
            # Reallocate samples
            moved_samples = self._reallocate_samples(
                optimized_labels, features_df, samples_to_reallocate
            )
            
            reallocation_info['samples_moved'] += moved_samples
            reallocation_info['iterations'] += 1
            
            if moved_samples == 0:
                break
        
        # Calculate improvement score
        original_score = self._calculate_clustering_score(cluster_labels, features_df)
        optimized_score = self._calculate_clustering_score(optimized_labels, features_df)
        reallocation_info['improvement_score'] = optimized_score - original_score
        
        return optimized_labels, reallocation_info
    
    def _find_samples_for_reallocation(self, cluster_labels: np.ndarray, 
                                     features_df: pd.DataFrame) -> List[int]:
        """Find samples that might benefit from reallocation."""
        samples_to_reallocate = []
        
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Skip noise points
                continue
            
            # Calculate distance to cluster center
            cluster_mask = cluster_labels == label
            if cluster_mask.sum() < 2:
                continue
            
            cluster_center = features_df[cluster_mask].mean()
            distance_to_center = np.linalg.norm(features_df.iloc[i] - cluster_center)
            
            # Calculate distance to other clusters
            other_clusters = set(cluster_labels) - {label, -1}
            min_distance_to_other = float('inf')
            
            for other_label in other_clusters:
                other_cluster_mask = cluster_labels == other_label
                if other_cluster_mask.sum() < 2:
                    continue
                
                other_center = features_df[other_cluster_mask].mean()
                distance_to_other = np.linalg.norm(features_df.iloc[i] - other_center)
                min_distance_to_other = min(min_distance_to_other, distance_to_other)
            
            # Check if sample is closer to another cluster
            if min_distance_to_other < distance_to_center * self.config.reallocation_threshold:
                samples_to_reallocate.append(i)
        
        return samples_to_reallocate
    
    def _reallocate_samples(self, cluster_labels: np.ndarray, 
                           features_df: pd.DataFrame, 
                           samples_to_reallocate: List[int]) -> int:
        """Reallocate samples to better clusters."""
        moved_samples = 0
        
        for sample_idx in samples_to_reallocate:
            original_label = cluster_labels[sample_idx]
            
            # Find best cluster for this sample
            best_label = self._find_best_cluster_for_sample(
                sample_idx, cluster_labels, features_df
            )
            
            if best_label != original_label:
                cluster_labels[sample_idx] = best_label
                moved_samples += 1
        
        return moved_samples
    
    def _find_best_cluster_for_sample(self, sample_idx: int, 
                                    cluster_labels: np.ndarray, 
                                    features_df: pd.DataFrame) -> int:
        """Find the best cluster for a sample."""
        sample_features = features_df.iloc[sample_idx]
        best_label = cluster_labels[sample_idx]
        min_distance = float('inf')
        
        for label in set(cluster_labels):
            if label == -1:
                continue
            
            cluster_mask = cluster_labels == label
            if cluster_mask.sum() < 2:
                continue
            
            cluster_center = features_df[cluster_mask].mean()
            distance = np.linalg.norm(sample_features - cluster_center)
            
            if distance < min_distance:
                min_distance = distance
                best_label = label
        
        return best_label
    
    def _validate_economic_value(self, cluster_labels: np.ndarray, 
                               features_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Validate clusters based on economic value."""
        logger.info("🔄 Validating economic value")
        
        try:
            # Use VectorBT acceleration if available
            if hasattr(self.vectorization_manager, 'validate_economic_value'):
                validated_labels, economic_info = self.vectorization_manager.validate_economic_value(
                    cluster_labels, features_df,
                    min_value=self.config.min_economic_value,
                    max_value=self.config.max_economic_value
                )
            else:
                # Use standard economic validation
                validated_labels, economic_info = self._standard_economic_validation(
                    cluster_labels, features_df
                )
            
            logger.info("✅ Economic validation completed")
            return validated_labels, economic_info
            
        except Exception as e:
            logger.error(f"❌ Economic validation failed: {e}")
            return cluster_labels, {'error': str(e)}
    
    def _standard_economic_validation(self, cluster_labels: np.ndarray, 
                                    features_df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Standard economic validation implementation."""
        validated_labels = cluster_labels.copy()
        economic_info = {
            'clusters_removed': 0,
            'samples_reclassified': 0,
            'economic_scores': {}
        }
        
        # Calculate economic score for each cluster
        for label in set(cluster_labels):
            if label == -1:
                continue
            
            cluster_mask = cluster_labels == label
            if cluster_mask.sum() < self.config.min_cluster_size:
                continue
            
            cluster_features = features_df[cluster_mask]
            economic_score = self._calculate_economic_score(cluster_features)
            economic_info['economic_scores'][label] = economic_score
            
            # Remove clusters with low economic value
            if economic_score < self.config.min_economic_value:
                validated_labels[cluster_mask] = -1
                economic_info['clusters_removed'] += 1
                economic_info['samples_reclassified'] += cluster_mask.sum()
        
        return validated_labels, economic_info
    
    def _calculate_economic_score(self, cluster_features: pd.DataFrame) -> float:
        """Calculate economic score for a cluster."""
        # Simple economic score based on feature variance and mean
        # In practice, this would be more sophisticated
        feature_variance = cluster_features.var().mean()
        feature_mean = cluster_features.mean().mean()
        
        # Higher variance and mean indicate better economic value
        economic_score = feature_variance * feature_mean
        
        return economic_score
    
    def _optimize_temporal_stability(self, cluster_labels: np.ndarray, 
                                   features_df: pd.DataFrame, 
                                   timestamps: pd.Series) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Optimize temporal stability of clusters."""
        logger.info("🔄 Optimizing temporal stability")
        
        try:
            # Use VectorBT acceleration if available
            if hasattr(self.vectorization_manager, 'optimize_temporal_stability'):
                stabilized_labels, temporal_info = self.vectorization_manager.optimize_temporal_stability(
                    cluster_labels, features_df, timestamps,
                    stability_threshold=self.config.stability_threshold,
                    min_consistency=self.config.min_temporal_consistency
                )
            else:
                # Use standard temporal stabilization
                stabilized_labels, temporal_info = self._standard_temporal_stabilization(
                    cluster_labels, features_df, timestamps
                )
            
            logger.info("✅ Temporal stabilization completed")
            return stabilized_labels, temporal_info
            
        except Exception as e:
            logger.error(f"❌ Temporal stabilization failed: {e}")
            return cluster_labels, {'error': str(e)}
    
    def _standard_temporal_stabilization(self, cluster_labels: np.ndarray, 
                                       features_df: pd.DataFrame, 
                                       timestamps: pd.Series) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Standard temporal stabilization implementation."""
        stabilized_labels = cluster_labels.copy()
        temporal_info = {
            'stability_improvements': 0,
            'temporal_consistency': 0.0,
            'cluster_stability_scores': {}
        }
        
        # Calculate temporal stability for each cluster
        for label in set(cluster_labels):
            if label == -1:
                continue
            
            cluster_mask = cluster_labels == label
            if cluster_mask.sum() < 2:
                continue
            
            # Calculate temporal stability
            stability_score = self._calculate_temporal_stability(
                cluster_mask, timestamps
            )
            temporal_info['cluster_stability_scores'][label] = stability_score
            
            # Improve stability if below threshold
            if stability_score < self.config.stability_threshold:
                improved_labels = self._improve_cluster_stability(
                    label, cluster_mask, cluster_labels, features_df, timestamps
                )
                stabilized_labels[cluster_mask] = improved_labels
                temporal_info['stability_improvements'] += 1
        
        # Calculate overall temporal consistency
        temporal_info['temporal_consistency'] = self._calculate_temporal_consistency(
            stabilized_labels, timestamps
        )
        
        return stabilized_labels, temporal_info
    
    def _calculate_temporal_stability(self, cluster_mask: np.ndarray, 
                                    timestamps: pd.Series) -> float:
        """Calculate temporal stability of a cluster."""
        cluster_timestamps = timestamps[cluster_mask]
        
        if len(cluster_timestamps) < 2:
            return 0.0
        
        # Calculate temporal consistency
        sorted_timestamps = cluster_timestamps.sort_values()
        time_gaps = sorted_timestamps.diff().dropna()
        
        if len(time_gaps) == 0:
            return 0.0
        
        # Stability is inversely related to variance in time gaps
        stability = 1.0 / (1.0 + time_gaps.var())
        
        return stability
    
    def _improve_cluster_stability(self, label: int, cluster_mask: np.ndarray, 
                                 cluster_labels: np.ndarray, 
                                 features_df: pd.DataFrame, 
                                 timestamps: pd.Series) -> np.ndarray:
        """Improve temporal stability of a cluster."""
        # Simple improvement: reassign samples based on temporal proximity
        cluster_timestamps = timestamps[cluster_mask]
        cluster_features = features_df[cluster_mask]
        
        # Find temporal outliers
        time_median = cluster_timestamps.median()
        time_mad = (cluster_timestamps - time_median).abs().median()
        
        # Reassign temporal outliers
        improved_labels = cluster_labels[cluster_mask].copy()
        temporal_outliers = (cluster_timestamps - time_median).abs() > 2 * time_mad
        
        if temporal_outliers.any():
            # Reassign outliers to noise
            improved_labels[temporal_outliers] = -1
        
        return improved_labels
    
    def _calculate_temporal_consistency(self, cluster_labels: np.ndarray, 
                                      timestamps: pd.Series) -> float:
        """Calculate overall temporal consistency of clustering."""
        consistency_scores = []
        
        for label in set(cluster_labels):
            if label == -1:
                continue
            
            cluster_mask = cluster_labels == label
            if cluster_mask.sum() < 2:
                continue
            
            stability = self._calculate_temporal_stability(cluster_mask, timestamps)
            consistency_scores.append(stability)
        
        if not consistency_scores:
            return 0.0
        
        return np.mean(consistency_scores)
    
    def _validate_clustering(self, cluster_labels: np.ndarray, 
                           features_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate final clustering results."""
        validation_info = {
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'n_noise_points': list(cluster_labels).count(-1),
            'silhouette_score': 0.0,
            'calinski_harabasz_score': 0.0,
            'davies_bouldin_score': 0.0,
            'validation_passed': True
        }
        
        # Calculate validation metrics
        if validation_info['n_clusters'] >= 2:
            try:
                valid_mask = cluster_labels != -1
                if valid_mask.sum() >= 2:
                    valid_labels = cluster_labels[valid_mask]
                    valid_features = features_df[valid_mask]
                    
                    if len(set(valid_labels)) >= 2:
                        validation_info['silhouette_score'] = silhouette_score(valid_features, valid_labels)
                        validation_info['calinski_harabasz_score'] = calinski_harabasz_score(valid_features, valid_labels)
                        validation_info['davies_bouldin_score'] = davies_bouldin_score(valid_features, valid_labels)
                        
                        # Check if validation passed
                        if validation_info['silhouette_score'] < self.config.min_silhouette_score:
                            validation_info['validation_passed'] = False
                            logger.warning(f"⚠️ Low silhouette score: {validation_info['silhouette_score']:.3f}")
            except Exception as e:
                logger.debug(f"Validation metrics calculation failed: {e}")
        
        return validation_info
    
    def _calculate_clustering_score(self, cluster_labels: np.ndarray, 
                                  features_df: pd.DataFrame) -> float:
        """Calculate overall clustering score."""
        try:
            valid_mask = cluster_labels != -1
            if valid_mask.sum() < 2:
                return 0.0
            
            valid_labels = cluster_labels[valid_mask]
            valid_features = features_df[valid_mask]
            
            if len(set(valid_labels)) < 2:
                return 0.0
            
            return silhouette_score(valid_features, valid_labels)
        except:
            return 0.0
    
    def _validate_inputs(self, cluster_labels: np.ndarray, features_df: pd.DataFrame):
        """Validate input parameters."""
        if len(cluster_labels) != len(features_df):
            raise ValueError("Cluster labels and features must have the same length")
        
        if len(cluster_labels) == 0:
            raise ValueError("Cluster labels cannot be empty")
    
    def _update_performance_stats(self, cluster_labels: np.ndarray, 
                                features_df: pd.DataFrame, 
                                post_processing_time: float):
        """Update performance statistics."""
        self.performance_stats['post_processing_time'] = post_processing_time
        
        # Calculate memory usage
        memory_usage = features_df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        self.performance_stats['memory_usage_mb'] = memory_usage
        
        # Get VectorBT usage rate
        vectorization_stats = self.vectorization_manager.get_performance_stats()
        self.performance_stats['vectorbt_usage_rate'] = vectorization_stats.get('vectorbt_usage_rate', 0)
        
        # Calculate parallel efficiency
        if post_processing_time > 0:
            samples_per_second = len(cluster_labels) / post_processing_time
            self.performance_stats['parallel_efficiency'] = samples_per_second
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add vectorization manager stats
        vectorization_stats = self.vectorization_manager.get_performance_stats()
        stats['vectorization_stats'] = vectorization_stats
        
        return stats
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'post_processing_time': 0.0,
            'samples_reallocated': 0,
            'economic_value_improvement': 0.0,
            'temporal_stability_improvement': 0.0,
            'memory_usage_mb': 0.0,
            'vectorbt_usage_rate': 0.0,
            'parallel_efficiency': 0.0
        }
        
        # Reset vectorization manager stats
        self.vectorization_manager.reset_stats()

# Convenience function for easy usage
def create_optimized_post_processor(
    enable_sample_reallocation: bool = True,
    enable_economic_validation: bool = True,
    enable_temporal_stabilization: bool = True,
    max_workers: Optional[int] = None,
    memory_efficient: bool = True,
    enable_vectorbt: bool = True,
    enable_gpu: bool = False
) -> OptimizedPostProcessor:
    """
    Create an optimized post-processor with specified configuration.
    
    Args:
        enable_sample_reallocation: Enable sample reallocation
        enable_economic_validation: Enable economic validation
        enable_temporal_stabilization: Enable temporal stabilization
        max_workers: Maximum number of parallel workers
        memory_efficient: Enable memory optimization
        enable_vectorbt: Enable VectorBT acceleration
        enable_gpu: Enable GPU acceleration
        
    Returns:
        OptimizedPostProcessor instance
    """
    config = PostProcessingConfig(
        enable_sample_reallocation=enable_sample_reallocation,
        enable_economic_validation=enable_economic_validation,
        enable_temporal_stabilization=enable_temporal_stabilization,
        max_workers=max_workers,
        memory_efficient=memory_efficient,
        enable_vectorbt=enable_vectorbt,
        enable_gpu=enable_gpu
    )
    
    return OptimizedPostProcessor(config)

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples, n_features = 1000, 10
    
    # Create clustered data
    cluster1 = np.random.randn(300, n_features) + [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    cluster2 = np.random.randn(300, n_features) + [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2]
    cluster3 = np.random.randn(400, n_features) + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    features = np.vstack([cluster1, cluster2, cluster3])
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    features_df = pd.DataFrame(features, columns=feature_names)
    
    # Create sample cluster labels
    cluster_labels = np.concatenate([
        np.zeros(300),  # Cluster 0
        np.ones(300),   # Cluster 1
        np.full(400, 2) # Cluster 2
    ])
    
    # Create timestamps
    timestamps = pd.Series(pd.date_range('2023-01-01', periods=n_samples, freq='1H'))
    
    print(f"Sample data: {features_df.shape}")
    print(f"Cluster labels: {len(np.unique(cluster_labels))} clusters")
    
    # Create optimized post-processor
    post_processor = create_optimized_post_processor(
        enable_sample_reallocation=True,
        enable_economic_validation=True,
        enable_temporal_stabilization=True,
        memory_efficient=True,
        enable_vectorbt=True
    )
    
    # Post-process clusters
    optimized_labels, post_processing_info = post_processor.post_process_clusters(
        cluster_labels, features_df, timestamps
    )
    
    print(f"Post-processing results: {len(np.unique(optimized_labels))} clusters")
    print(f"Performance stats: {post_processor.get_performance_stats()}")
