"""
Refactored NAS-TAS Clustering Component.

This module provides a clean, modular implementation of the NAS-TAS clustering functionality,
broken down into focused, single-responsibility components for better maintainability and testability.

Currently imports from the existing components directory until the modular structure is fully implemented.
"""

# Import main component from this directory
from .main_component import NASTASClusteringComponent

# Import configuration from clustering directory
from .config.clustering_config import NASTASClusteringConfig, ClusteringContext

# Import services from clusters directory
from ..clusters.clustering_service import ClusteringService
from ..clusters.feature_service import FeatureService
from ..clusters.optimization_service import OptimizationService
from ..clusters.hardware_service import HardwareService

__all__ = [
    'NASTASClusteringComponent',
    'NASTASClusteringConfig', 
    'ClusteringContext',
    'ClusteringService',
    'FeatureService',
    'OptimizationService',
    'HardwareService'
]
