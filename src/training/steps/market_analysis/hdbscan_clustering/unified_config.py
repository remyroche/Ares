"""
Unified HDBSCAN Configuration Management

This module provides a single source of truth for all HDBSCAN clustering parameters
with proper validation, adaptive scaling, and comprehensive quality thresholds.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import pandas as pd
from enum import Enum


class ExecutionMode(Enum):
    """Execution modes for different use cases."""
    LIGHT = "light"      # Fast execution for small datasets
    STANDARD = "standard"  # Balanced execution
    FULL = "full"        # Comprehensive execution for large datasets
    BLANK = "blank"      # Minimal execution for testing


class DistanceMetric(Enum):
    """Supported distance metrics."""
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    CHEBYSHEV = "chebyshev"


class ClusterSelectionMethod(Enum):
    """HDBSCAN cluster selection methods."""
    EOM = "eom"    # Excess of Mass
    LEAF = "leaf"  # Leaf selection


@dataclass
class QualityThresholds:
    """Quality thresholds for clustering validation."""
    min_dbcv: float = 0.3
    min_silhouette: float = 0.2
    min_temporal_stability: float = 0.7
    min_economic_separation: float = 0.2
    max_noise_ratio: float = 0.4
    min_cluster_size_ratio: float = 0.02  # Minimum 2% of data
    max_cluster_size_ratio: float = 0.5   # Maximum 50% of data


@dataclass
class AdaptiveScalingConfig:
    """Configuration for adaptive parameter scaling."""
    enable_adaptive_scaling: bool = True
    min_cluster_size_factor: float = 0.02  # 2% of data size
    min_samples_factor: float = 0.01       # 1% of data size
    max_cluster_size: int = 100
    min_cluster_size: int = 5
    max_min_samples: int = 50
    min_min_samples: int = 3


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering pipeline."""
    correlation_threshold: float = 0.95
    enable_feature_selection: bool = True
    feature_selection_method: str = "mrmr"  # 'mrmr', 'lasso', 'mutual_info'
    max_features: int = 50
    enable_regime_features: bool = True
    enable_entropy_features: bool = True
    enable_spectral_features: bool = True
    enable_temporal_features: bool = True


@dataclass
class TemporalStabilityConfig:
    """Configuration for temporal stability validation."""
    min_regime_duration: int = 20  # Minimum bars for a regime
    max_transition_frequency: float = 0.1  # Max 10% transitions per window
    stability_window: int = 100  # Window for stability calculation
    enable_temporal_smoothing: bool = True
    smoothing_window: int = 5


@dataclass
class ChunkProcessingConfig:
    """Configuration for chunked processing."""
    enable_chunked_processing: bool = True
    chunk_size: int = 1000
    chunk_overlap: float = 0.1  # 10% overlap between chunks
    enable_temporal_continuity: bool = True
    merge_similar_clusters: bool = True
    similarity_threshold: float = 0.8


class UnifiedHDBSCANConfig:
    """
    Unified configuration for all HDBSCAN clustering components.
    
    This class provides a single source of truth for all clustering parameters
    with proper validation, adaptive scaling, and comprehensive quality control.
    """
    
    def __init__(
        self,
        execution_mode: ExecutionMode = ExecutionMode.STANDARD,
        quality_thresholds: Optional[QualityThresholds] = None,
        adaptive_scaling: Optional[AdaptiveScalingConfig] = None,
        feature_engineering: Optional[FeatureEngineeringConfig] = None,
        temporal_stability: Optional[TemporalStabilityConfig] = None,
        chunk_processing: Optional[ChunkProcessingConfig] = None
    ):
        self.execution_mode = execution_mode
        self.quality_thresholds = quality_thresholds or QualityThresholds()
        self.adaptive_scaling = adaptive_scaling or AdaptiveScalingConfig()
        self.feature_engineering = feature_engineering or FeatureEngineeringConfig()
        self.temporal_stability = temporal_stability or TemporalStabilityConfig()
        self.chunk_processing = chunk_processing or ChunkProcessingConfig()
        
        # Core HDBSCAN parameters (will be set adaptively)
        self.min_cluster_size: int = 15
        self.min_samples: int = 5
        self.cluster_selection_epsilon: float = 0.0
        self.cluster_selection_method: ClusterSelectionMethod = ClusterSelectionMethod.EOM
        self.metric: DistanceMetric = DistanceMetric.EUCLIDEAN
        self.alpha: float = 1.0
        
        # Performance settings
        self.n_jobs: int = -1
        self.memory_efficient: bool = True
        self.max_memory_gb: float = 8.0
        
        # Validation
        self._validate_config()
        self._apply_execution_mode_optimizations()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate quality thresholds
        assert 0 <= self.quality_thresholds.min_dbcv <= 1, "min_dbcv must be in [0, 1]"
        assert 0 <= self.quality_thresholds.min_silhouette <= 1, "min_silhouette must be in [0, 1]"
        assert 0 <= self.quality_thresholds.min_temporal_stability <= 1, "min_temporal_stability must be in [0, 1]"
        assert 0 <= self.quality_thresholds.min_economic_separation <= 1, "min_economic_separation must be in [0, 1]"
        assert 0 <= self.quality_thresholds.max_noise_ratio <= 1, "max_noise_ratio must be in [0, 1]"
        
        # Validate adaptive scaling
        assert 0 < self.adaptive_scaling.min_cluster_size_factor <= 0.5, "min_cluster_size_factor must be in (0, 0.5]"
        assert 0 < self.adaptive_scaling.min_samples_factor <= 0.1, "min_samples_factor must be in (0, 0.1]"
        assert self.adaptive_scaling.min_cluster_size < self.adaptive_scaling.max_cluster_size, "min_cluster_size must be < max_cluster_size"
        
        # Validate feature engineering
        assert 0 < self.feature_engineering.correlation_threshold <= 1, "correlation_threshold must be in (0, 1]"
        assert self.feature_engineering.max_features > 0, "max_features must be > 0"
        assert self.feature_engineering.feature_selection_method in ['mrmr', 'lasso', 'mutual_info'], "Invalid feature_selection_method"
    
    def _apply_execution_mode_optimizations(self):
        """Apply optimizations based on execution mode."""
        if self.execution_mode == ExecutionMode.LIGHT:
            # Light mode: faster execution
            self.adaptive_scaling.min_cluster_size_factor = 0.05  # 5% of data
            self.adaptive_scaling.min_samples_factor = 0.02       # 2% of data
            self.feature_engineering.max_features = 30
            self.feature_engineering.enable_entropy_features = False
            self.feature_engineering.enable_spectral_features = False
            self.chunk_processing.chunk_size = 500
            
        elif self.execution_mode == ExecutionMode.BLANK:
            # Blank mode: minimal execution for testing
            self.adaptive_scaling.min_cluster_size_factor = 0.1   # 10% of data
            self.adaptive_scaling.min_samples_factor = 0.05       # 5% of data
            self.feature_engineering.max_features = 20
            self.feature_engineering.enable_entropy_features = False
            self.feature_engineering.enable_spectral_features = False
            self.feature_engineering.enable_regime_features = False
            self.chunk_processing.chunk_size = 200
            
        elif self.execution_mode == ExecutionMode.FULL:
            # Full mode: comprehensive execution
            self.adaptive_scaling.min_cluster_size_factor = 0.01  # 1% of data
            self.adaptive_scaling.min_samples_factor = 0.005      # 0.5% of data
            self.feature_engineering.max_features = 100
            self.chunk_processing.chunk_size = 2000
    
    def get_adaptive_parameters(self, data_size: int, feature_count: int) -> Dict[str, Any]:
        """
        Get parameters adapted to data characteristics.
        
        Args:
            data_size: Number of samples in the dataset
            feature_count: Number of features in the dataset
            
        Returns:
            Dictionary of adaptive parameters
        """
        if not self.adaptive_scaling.enable_adaptive_scaling:
            return {
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'cluster_selection_epsilon': self.cluster_selection_epsilon,
                'cluster_selection_method': self.cluster_selection_method.value,
                'metric': self.metric.value,
                'alpha': self.alpha
            }
        
        # Calculate adaptive parameters
        adaptive_min_cluster_size = max(
            self.adaptive_scaling.min_cluster_size,
            min(
                int(data_size * self.adaptive_scaling.min_cluster_size_factor),
                self.adaptive_scaling.max_cluster_size
            )
        )
        
        adaptive_min_samples = max(
            self.adaptive_scaling.min_min_samples,
            min(
                int(data_size * self.adaptive_scaling.min_samples_factor),
                self.adaptive_scaling.max_min_samples
            )
        )
        
        # Choose metric based on feature characteristics
        if feature_count > 50:
            metric = DistanceMetric.MANHATTAN  # More robust for high dimensions
        else:
            metric = self.metric
        
        # Choose cluster selection method based on data size
        if data_size > 1000:
            cluster_selection_method = ClusterSelectionMethod.EOM  # Better for large datasets
        else:
            cluster_selection_method = ClusterSelectionMethod.LEAF  # Better for small datasets
        
        return {
            'min_cluster_size': adaptive_min_cluster_size,
            'min_samples': adaptive_min_samples,
            'cluster_selection_epsilon': self.cluster_selection_epsilon,
            'cluster_selection_method': cluster_selection_method.value,
            'metric': metric.value,
            'alpha': self.alpha
        }
    
    def validate_clustering_quality(self, cluster_labels: np.ndarray, 
                                  quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate clustering quality against thresholds.
        
        Args:
            cluster_labels: Cluster labels from HDBSCAN
            quality_metrics: Dictionary of quality metrics
            
        Returns:
            Validation results with pass/fail status
        """
        validation_results = {
            'overall_passed': True,
            'individual_checks': {},
            'recommendations': []
        }
        
        # Check DBCV score
        dbcv = quality_metrics.get('dbcv', 0.0)
        dbcv_passed = dbcv >= self.quality_thresholds.min_dbcv
        validation_results['individual_checks']['dbcv'] = {
            'passed': dbcv_passed,
            'value': dbcv,
            'threshold': self.quality_thresholds.min_dbcv
        }
        if not dbcv_passed:
            validation_results['recommendations'].append(
                f"DBCV score {dbcv:.3f} below threshold {self.quality_thresholds.min_dbcv}. "
                "Consider adjusting min_cluster_size or min_samples."
            )
        
        # Check silhouette score
        silhouette = quality_metrics.get('silhouette_score', 0.0)
        silhouette_passed = silhouette >= self.quality_thresholds.min_silhouette
        validation_results['individual_checks']['silhouette'] = {
            'passed': silhouette_passed,
            'value': silhouette,
            'threshold': self.quality_thresholds.min_silhouette
        }
        if not silhouette_passed:
            validation_results['recommendations'].append(
                f"Silhouette score {silhouette:.3f} below threshold {self.quality_thresholds.min_silhouette}. "
                "Consider using different distance metric or feature preprocessing."
            )
        
        # Check temporal stability
        temporal_stability = quality_metrics.get('temporal_stability', 0.0)
        temporal_passed = temporal_stability >= self.quality_thresholds.min_temporal_stability
        validation_results['individual_checks']['temporal_stability'] = {
            'passed': temporal_passed,
            'value': temporal_stability,
            'threshold': self.quality_thresholds.min_temporal_stability
        }
        if not temporal_passed:
            validation_results['recommendations'].append(
                f"Temporal stability {temporal_stability:.3f} below threshold {self.quality_thresholds.min_temporal_stability}. "
                "Consider enabling temporal smoothing or adjusting regime duration requirements."
            )
        
        # Check economic separation
        economic_separation = quality_metrics.get('economic_separation', 0.0)
        economic_passed = economic_separation >= self.quality_thresholds.min_economic_separation
        validation_results['individual_checks']['economic_separation'] = {
            'passed': economic_passed,
            'value': economic_separation,
            'threshold': self.quality_thresholds.min_economic_separation
        }
        if not economic_passed:
            validation_results['recommendations'].append(
                f"Economic separation {economic_separation:.3f} below threshold {self.quality_thresholds.min_economic_separation}. "
                "Consider adding more regime-specific features or adjusting clustering parameters."
            )
        
        # Check noise ratio
        noise_ratio = quality_metrics.get('noise_ratio', 0.0)
        noise_passed = noise_ratio <= self.quality_thresholds.max_noise_ratio
        validation_results['individual_checks']['noise_ratio'] = {
            'passed': noise_passed,
            'value': noise_ratio,
            'threshold': self.quality_thresholds.max_noise_ratio
        }
        if not noise_passed:
            validation_results['recommendations'].append(
                f"Noise ratio {noise_ratio:.3f} above threshold {self.quality_thresholds.max_noise_ratio}. "
                "Consider reducing min_cluster_size or min_samples."
            )
        
        # Check cluster size distribution
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels[unique_labels != -1])
        if n_clusters > 0:
            cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels if label != -1]
            cluster_size_ratios = [size / len(cluster_labels) for size in cluster_sizes]
            
            min_ratio = min(cluster_size_ratios)
            max_ratio = max(cluster_size_ratios)
            
            size_distribution_passed = (
                min_ratio >= self.quality_thresholds.min_cluster_size_ratio and
                max_ratio <= self.quality_thresholds.max_cluster_size_ratio
            )
            validation_results['individual_checks']['cluster_size_distribution'] = {
                'passed': size_distribution_passed,
                'min_ratio': min_ratio,
                'max_ratio': max_ratio,
                'min_threshold': self.quality_thresholds.min_cluster_size_ratio,
                'max_threshold': self.quality_thresholds.max_cluster_size_ratio
            }
            if not size_distribution_passed:
                validation_results['recommendations'].append(
                    f"Cluster size distribution unbalanced: min={min_ratio:.3f}, max={max_ratio:.3f}. "
                    "Consider adjusting clustering parameters or using weighted clustering."
                )
        
        # Overall validation
        validation_results['overall_passed'] = all(
            check['passed'] for check in validation_results['individual_checks'].values()
        )
        
        return validation_results
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'execution_mode': self.execution_mode.value,
            'quality_thresholds': {
                'min_dbcv': self.quality_thresholds.min_dbcv,
                'min_silhouette': self.quality_thresholds.min_silhouette,
                'min_temporal_stability': self.quality_thresholds.min_temporal_stability,
                'min_economic_separation': self.quality_thresholds.min_economic_separation,
                'max_noise_ratio': self.quality_thresholds.max_noise_ratio,
                'min_cluster_size_ratio': self.quality_thresholds.min_cluster_size_ratio,
                'max_cluster_size_ratio': self.quality_thresholds.max_cluster_size_ratio
            },
            'adaptive_scaling': {
                'enable_adaptive_scaling': self.adaptive_scaling.enable_adaptive_scaling,
                'min_cluster_size_factor': self.adaptive_scaling.min_cluster_size_factor,
                'min_samples_factor': self.adaptive_scaling.min_samples_factor,
                'max_cluster_size': self.adaptive_scaling.max_cluster_size,
                'min_cluster_size': self.adaptive_scaling.min_cluster_size,
                'max_min_samples': self.adaptive_scaling.max_min_samples,
                'min_min_samples': self.adaptive_scaling.min_min_samples
            },
            'feature_engineering': {
                'correlation_threshold': self.feature_engineering.correlation_threshold,
                'enable_feature_selection': self.feature_engineering.enable_feature_selection,
                'feature_selection_method': self.feature_engineering.feature_selection_method,
                'max_features': self.feature_engineering.max_features,
                'enable_regime_features': self.feature_engineering.enable_regime_features,
                'enable_entropy_features': self.feature_engineering.enable_entropy_features,
                'enable_spectral_features': self.feature_engineering.enable_spectral_features,
                'enable_temporal_features': self.feature_engineering.enable_temporal_features
            },
            'temporal_stability': {
                'min_regime_duration': self.temporal_stability.min_regime_duration,
                'max_transition_frequency': self.temporal_stability.max_transition_frequency,
                'stability_window': self.temporal_stability.stability_window,
                'enable_temporal_smoothing': self.temporal_stability.enable_temporal_smoothing,
                'smoothing_window': self.temporal_stability.smoothing_window
            },
            'chunk_processing': {
                'enable_chunked_processing': self.chunk_processing.enable_chunked_processing,
                'chunk_size': self.chunk_processing.chunk_size,
                'chunk_overlap': self.chunk_processing.chunk_overlap,
                'enable_temporal_continuity': self.chunk_processing.enable_temporal_continuity,
                'merge_similar_clusters': self.chunk_processing.merge_similar_clusters,
                'similarity_threshold': self.chunk_processing.similarity_threshold
            },
            'created_at': datetime.now().isoformat()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UnifiedHDBSCANConfig':
        """Create configuration from dictionary."""
        execution_mode = ExecutionMode(config_dict.get('execution_mode', 'standard'))
        
        quality_thresholds = QualityThresholds(**config_dict.get('quality_thresholds', {}))
        adaptive_scaling = AdaptiveScalingConfig(**config_dict.get('adaptive_scaling', {}))
        feature_engineering = FeatureEngineeringConfig(**config_dict.get('feature_engineering', {}))
        temporal_stability = TemporalStabilityConfig(**config_dict.get('temporal_stability', {}))
        chunk_processing = ChunkProcessingConfig(**config_dict.get('chunk_processing', {}))
        
        return cls(
            execution_mode=execution_mode,
            quality_thresholds=quality_thresholds,
            adaptive_scaling=adaptive_scaling,
            feature_engineering=feature_engineering,
            temporal_stability=temporal_stability,
            chunk_processing=chunk_processing
        )


def create_unified_config(
    execution_mode: str = "standard",
    **kwargs
) -> UnifiedHDBSCANConfig:
    """
    Factory function to create a unified HDBSCAN configuration.
    
    Args:
        execution_mode: Execution mode ('light', 'standard', 'full', 'blank')
        **kwargs: Additional configuration parameters
        
    Returns:
        UnifiedHDBSCANConfig instance
    """
    mode = ExecutionMode(execution_mode.lower())
    return UnifiedHDBSCANConfig(execution_mode=mode, **kwargs)