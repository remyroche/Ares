"""
Hybrid NAS Clustering - Complementary Tree-Based and Neural Architecture Search

This module provides hybrid NAS clustering that combines tree-based and neural
approaches to complement the existing neural NAS system for market regime detection.

Key Features:
- Tree-based NAS for fast feature selection and regime detection
- Neural NAS for complex pattern recognition and sequential modeling
- Intelligent routing based on data characteristics
- Ensemble methods combining both approaches
- Integration with existing neural NAS pipeline
"""

from .core.hybrid_nas_clusterer import HybridNASClusterer
from .core.hybrid_nas_config import HybridNASClusteringConfig
from .components.hybrid_nas_clustering_component import HybridNASClusteringComponent
from .integration.hybrid_nas_orchestrator import HybridNASOrchestrator

__all__ = [
    'HybridNASClusterer',
    'HybridNASClusteringConfig', 
    'HybridNASClusteringComponent',
    'HybridNASOrchestrator'
]