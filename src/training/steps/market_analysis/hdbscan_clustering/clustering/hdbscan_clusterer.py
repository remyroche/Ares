"""
HDBSCAN Clusterer for regime discovery system.

This module provides an HDBSCANClusterer class that wraps the optimized clustering
components for backward compatibility with the HDBSCAN clustering pipeline.
"""

from ..optimization.optimized_hdbscan_clusterer import OptimizedHDBSCANClusterer

# Alias for backward compatibility with the HDBSCAN clustering system
HDBSCANClusterer = OptimizedHDBSCANClusterer

__all__ = ['HDBSCANClusterer']
