"""
HMM Clustering Module - Consolidated HMM-based regime clustering functionality.

This module provides a unified interface for HMM clustering operations including:
- Matrix-optimized clustering with GPU acceleration
- Enhanced clustering with 4D frontier optimization
- Comprehensive metrics and reporting
- Fast fail mechanisms
"""

# Core clustering algorithms
from .core.matrix_optimized import MatrixOptimizedClusterer
from .core.enhanced_optimized import EnhancedMatrixOptimizedClusterer
from .core.base_clustering import BaseClusterer, ClusteringResult

# Integration and orchestration
from .integration.orchestrator import OptimalRegimeClusteringOrchestrator
from .integration.enhanced_integration import EnhancedClusteringIntegration
from .integration.fast_fail import FastFailManager

# Metrics and reporting
from .metrics.basic_metrics import BasicClusteringMetrics
from .metrics.detailed_metrics import DetailedClusteringMetrics
from .metrics.evolution_report import MetricsEvolutionReporter

# Component wrappers
from .components.clustering_component import OptimalRegimeClusteringComponent

# Configuration
from .config import HMMClusteringConfig

__all__ = [
    # Core clustering
    'MatrixOptimizedClusterer',
    'EnhancedMatrixOptimizedClusterer',
    'BaseClusterer',
    'ClusteringResult',
    
    # Integration
    'OptimalRegimeClusteringOrchestrator',
    'EnhancedClusteringIntegration',
    'FastFailManager',
    
    # Metrics
    'BasicClusteringMetrics',
    'DetailedClusteringMetrics',
    'MetricsEvolutionReporter',
    
    # Components
    'OptimalRegimeClusteringComponent',
    
    # Config
    'HMMClusteringConfig'
]