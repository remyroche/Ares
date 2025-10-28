"""
MS-DR Clustering Module

This module provides Markov-Switching Dynamic Regression clustering
for regime discovery with integrated quality assessment and artifact management.
"""

from .ms_dr_clusterer import (
    MSDRClusterer,
    MSDRConfig,
    MSDRResult,
    create_ms_dr_clusterer,
    MS_AVAILABLE,
    MS_LIBRARY
)

# Import auto-tuner
from .ms_dr_auto_tuner import (
    MSDRAutoTuner,
    MSDRTuningConfig,
    auto_tune_ms_dr_clustering
)

# Import hierarchical HPO extension
try:
    from .hierarchical_hpo_extension import (
        MSDRHierarchicalOptimizer,
        create_msdr_parameter_groups,
        create_msdr_optimization_stages
    )
    HIERARCHICAL_HPO_AVAILABLE = True
except ImportError:
    HIERARCHICAL_HPO_AVAILABLE = False
    MSDRHierarchicalOptimizer = None
    create_msdr_parameter_groups = None
    create_msdr_optimization_stages = None

# Import standalone function with artifact manager
try:
    from src.feature_generation.integration.enhanced_ms_dr_clustering_integration import (
        perform_ms_dr_clustering_with_artifact_manager,
        perform_enhanced_ms_dr_clustering
    )
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    perform_ms_dr_clustering_with_artifact_manager = None
    perform_enhanced_ms_dr_clustering = None

__all__ = [
    'MSDRClusterer',
    'MSDRConfig',
    'MSDRResult',
    'create_ms_dr_clusterer',
    'MS_AVAILABLE',
    'MS_LIBRARY',
    'MSDRAutoTuner',
    'MSDRTuningConfig',
    'auto_tune_ms_dr_clustering',
    'perform_ms_dr_clustering_with_artifact_manager',
    'perform_enhanced_ms_dr_clustering',
    'INTEGRATION_AVAILABLE',
    'MSDRHierarchicalOptimizer',
    'create_msdr_parameter_groups',
    'create_msdr_optimization_stages',
    'HIERARCHICAL_HPO_AVAILABLE'
]
