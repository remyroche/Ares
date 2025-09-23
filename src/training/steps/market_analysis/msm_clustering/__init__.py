"""
MSM Clustering Module - Markov State Model based regime clustering functionality.

This module provides a unified interface for MSM clustering operations including:
- Data-driven regime discovery with structural break detection
- Enhanced clustering with adaptive regime identification
- Comprehensive metrics and reporting
- Fast fail mechanisms
- Integration with existing HMM infrastructure
"""

# Core clustering algorithms
from .core.msm_optimized import MSMOptimizedClusterer
from .core.enhanced_msm import EnhancedMSMClusterer
from .core.base_msm_clustering import BaseMSMClusterer, MSMClusteringResult

# Integration and orchestration
from .integration.msm_orchestrator import MSMRegimeClusteringOrchestrator
from .integration.enhanced_msm_integration import EnhancedMSMIntegration
from .integration.msm_fast_fail import MSMFastFailManager

# Metrics and reporting
from .metrics.msm_metrics import MSMClusteringMetrics
from .metrics.detailed_msm_metrics import DetailedMSMMetrics
from .metrics.msm_evolution_report import MSMMetricsEvolutionReporter

# Component wrappers
from .components.msm_clustering_component import MSMRegimeClusteringComponent

# Configuration
from .config import MSMClusteringConfig

__all__ = [
    # Core clustering
    'MSMOptimizedClusterer',
    'EnhancedMSMClusterer',
    'BaseMSMClusterer',
    'MSMClusteringResult',
    
    # Integration
    'MSMRegimeClusteringOrchestrator',
    'EnhancedMSMIntegration',
    'MSMFastFailManager',
    
    # Metrics
    'MSMClusteringMetrics',
    'DetailedMSMMetrics',
    'MSMMetricsEvolutionReporter',
    
    # Components
    'MSMRegimeClusteringComponent',
    
    # Config
    'MSMClusteringConfig'
]