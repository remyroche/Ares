"""
Clustering module for regime discovery system.

This module now imports optimized clustering components from the optimization/
directory instead of maintaining legacy implementations.
"""

# Import optimized clustering components
from ..optimization.optimized_hdbscan_clusterer import (
    OptimizedHDBSCANClusterer,
    HDBSCANConfig,
    create_optimized_hdbscan_clusterer
)

from .noise_handler import NoiseHandler

# Legacy aliases for backward compatibility
HDBSCANClusterer = OptimizedHDBSCANClusterer
ClusterResult = dict  # Simple alias for backward compatibility

__all__ = [
    # Optimized components
    'OptimizedHDBSCANClusterer',
    'HDBSCANConfig',
    'create_optimized_hdbscan_clusterer',
    'NoiseHandler',
    
    # Legacy aliases for backward compatibility
    'HDBSCANClusterer',
    'ClusterResult'
]