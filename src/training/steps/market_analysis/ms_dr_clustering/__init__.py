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

# Import artifact integration functions
try:
    from .artifact_integration import (
        perform_ms_dr_clustering_with_artifact_manager,
        perform_enhanced_ms_dr_clustering,
        load_market_data_for_msdr
    )
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    perform_ms_dr_clustering_with_artifact_manager = None
    perform_enhanced_ms_dr_clustering = None
    load_market_data_for_msdr = None

__all__ = [
    # Core clustering
    'MSDRClusterer',
    'MSDRConfig',
    'MSDRResult',
    'create_ms_dr_clusterer',
    'MS_AVAILABLE',
    'MS_LIBRARY',
    
    # Auto-tuning
    'MSDRAutoTuner',
    'MSDRTuningConfig',
    'auto_tune_ms_dr_clustering',
    
    # Hierarchical optimization
    'MSDRHierarchicalOptimizer',
    'create_msdr_parameter_groups',
    'create_msdr_optimization_stages',
    'HIERARCHICAL_HPO_AVAILABLE',
    
    # Integration functions (artifact management & data loading)
    'perform_ms_dr_clustering_with_artifact_manager',
    'perform_enhanced_ms_dr_clustering',
    'load_market_data_for_msdr',
    'INTEGRATION_AVAILABLE'
]
