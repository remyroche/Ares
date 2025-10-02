"""
Refactored NAS-TAS Clustering Component.

This module provides a clean, modular implementation of the NAS-TAS clustering functionality,
broken down into focused, single-responsibility components for better maintainability and testability.
"""

from .main_component import NASTASClusteringComponent
from .config.clustering_config import NASTASClusteringConfig, ClusteringContext
from .services.clustering_service import ClusteringService
from .services.feature_service import FeatureService
from .services.optimization_service import OptimizationService
from .services.hardware_service import HardwareService

__all__ = [
    'NASTASClusteringComponent',
    'NASTASClusteringConfig', 
    'ClusteringContext',
    'ClusteringService',
    'FeatureService',
    'OptimizationService',
    'HardwareService'
]
