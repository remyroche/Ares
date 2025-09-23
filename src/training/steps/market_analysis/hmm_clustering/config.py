"""
Consolidated configuration for HMM clustering operations only.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class ClusteringMode(Enum):
    """Clustering execution modes."""
    STANDARD = "standard"
    ENHANCED = "enhanced"
    BOTH = "both"


class HardwareAcceleration(Enum):
    """Hardware acceleration options."""
    AUTO = "auto"
    GPU = "gpu"
    CPU = "cpu"
    MPS = "mps"  # Apple Silicon Metal Performance Shaders


@dataclass
class HMMClusteringConfig:
    """Configuration for HMM clustering operations only."""
    
    # Clustering configuration
    clustering_config: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics configuration
    metrics_config: Dict[str, Any] = field(default_factory=dict)
    
    # Integration configuration
    integration_config: Dict[str, Any] = field(default_factory=dict)
    
    # Hardware optimization
    hardware_config: Dict[str, Any] = field(default_factory=dict)
    
    # Fast fail configuration
    fast_fail_config: Dict[str, Any] = field(default_factory=dict)
    
    # Execution mode
    mode: ClusteringMode = ClusteringMode.BOTH
    
    # Hardware acceleration
    hardware_acceleration: HardwareAcceleration = HardwareAcceleration.AUTO
    
    # Feature engineering
    feature_engineering: Dict[str, Any] = field(default_factory=dict)
    
    # Memory management
    memory_config: Dict[str, Any] = field(default_factory=dict)
    
    # Timeout configurations
    timeouts: Dict[str, float] = field(default_factory=dict)
    
    # Quality thresholds
    quality_thresholds: Dict[str, float] = field(default_factory=dict)
    
    @classmethod
    def create_default(cls) -> 'HMMClusteringConfig':
        """Create default configuration."""
        return cls(
            clustering_config={
                'use_matrix_optimization': True,
                'use_enhanced_clustering': True,
                'n_clusters_range': (2, 20),
                'min_cluster_size': 10,
                'max_cluster_size': 1000,
                'silhouette_threshold': 0.3,
                'cv_threshold': 0.6
            },
            metrics_config={
                'calculate_basic_metrics': True,
                'calculate_detailed_metrics': True,
                'track_evolution': True,
                'export_reports': True,
                'report_formats': ['json', 'csv']
            },
            integration_config={
                'enable_fast_fail': True,
                'fallback_to_standard': False,
                'parallel_processing': True,
                'max_workers': 4
            },
            hardware_config={
                'use_gpu_acceleration': True,
                'use_matrix_operations': True,
                'memory_efficient': True,
                'batch_processing': True
            },
            fast_fail_config={
                'timeout_seconds': 300,
                'memory_limit_gb': 8.0,
                'quality_threshold': 0.3,
                'max_iterations': 100
            },
            feature_engineering={
                'extract_4d_features': True,
                'volume_features': True,
                'volatility_features': True,
                'momentum_features': True,
                'trend_features': True,
                'normalize_features': True
            },
            memory_config={
                'chunk_size': 10000,
                'max_memory_usage': 0.8,
                'enable_garbage_collection': True,
                'cache_intermediate_results': True
            },
            timeouts={
                'standard_clustering': 120.0,
                'enhanced_clustering': 300.0,
                'metrics_calculation': 60.0,
                'feature_engineering': 90.0
            },
            quality_thresholds={
                'min_silhouette': 0.2,
                'max_cluster_cv': 0.8,
                'min_cluster_count': 2,
                'max_cluster_count': 25
            }
        )
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key in ['clustering_config', 'metrics_config', 'integration_config', 
                        'hardware_config', 'fast_fail_config', 'feature_engineering',
                        'memory_config', 'timeouts', 'quality_thresholds']:
                getattr(self, key).update(value)
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware configuration for clustering operations."""
        return {
            'use_gpu_acceleration': self.hardware_config.get('use_gpu_acceleration', True),
            'use_matrix_operations': self.hardware_config.get('use_matrix_operations', True),
            'memory_efficient': self.hardware_config.get('memory_efficient', True),
            'batch_processing': self.hardware_config.get('batch_processing', True),
            'hardware_acceleration': self.hardware_acceleration.value
        }
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return {
            'extract_4d_features': self.feature_engineering.get('extract_4d_features', True),
            'volume_features': self.feature_engineering.get('volume_features', True),
            'volatility_features': self.feature_engineering.get('volatility_features', True),
            'momentum_features': self.feature_engineering.get('momentum_features', True),
            'trend_features': self.feature_engineering.get('trend_features', True),
            'normalize_features': self.feature_engineering.get('normalize_features', True)
        }
    
    def get_timeout_config(self) -> Dict[str, float]:
        """Get timeout configuration."""
        return {
            'standard_clustering': self.timeouts.get('standard_clustering', 120.0),
            'enhanced_clustering': self.timeouts.get('enhanced_clustering', 300.0),
            'metrics_calculation': self.timeouts.get('metrics_calculation', 60.0),
            'feature_engineering': self.timeouts.get('feature_engineering', 90.0)
        }