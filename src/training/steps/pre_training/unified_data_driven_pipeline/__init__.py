"""
Unified Data-Driven Feature Pipeline

A comprehensive, data-driven feature engineering pipeline that consolidates
period optimization, interaction generation, and feature selection into a
single, coherent system.

Key Features:
- Purged & Embargoed Walk-Forward CV to prevent leakage
- Multi-objective feature selection with explicit objectives
- Data-driven approach with configurable guardrails
- VectorBT optimization for performance
- Comprehensive statistical analysis
- Advanced economic evaluation
- HTF-aware interaction generation
- GPU optimizations
- Advanced caching and serialization

This module has been consolidated to eliminate redundancy and provide a single,
comprehensive implementation that integrates all advanced features.
"""

# Import the consolidated pipeline as the main implementation
from .consolidated_pipeline import (
    UnifiedDataDrivenPipeline,
    ConsolidatedPipelineResult,
    create_unified_pipeline,
    process_with_unified_pipeline
)

# Import the refactored pipeline (recommended for new usage)
from .refactored_pipeline import (
    RefactoredUnifiedPipeline,
    RefactoredPipelineResult,
    create_refactored_pipeline,
    create_full_pipeline,
    create_blank_pipeline,
    create_light_pipeline
)

from .core.config import (
    UnifiedPipelineConfig,
    create_default_config,
    create_high_performance_config,
    create_memory_efficient_config,
    create_fast_config
)

# Import simplified configuration
from .core.simplified_config import (
    create_full_config,
    create_blank_config,
    create_light_config,
    create_config_by_intensity,
    list_available_intensities,
    PipelineIntensity
)

from .time_series_cv import (
    PurgedEmbargoedWalkForwardCV,
    PurgedEmbargoedConfig,
    TimeSeriesSplit,
    create_purged_embargoed_cv,
    validate_time_series_splits
)

from .statistical_analysis import (
    StatisticalAnalysisFramework,
    DataCharacteristics,
    PatternAnalysis,
    RelationshipAnalysis
)

from .feature_selection.multi_objective_selector import (
    MultiObjectiveFeatureSelector,
    create_default_objectives,
    create_performance_objectives,
    create_stability_objectives,
    create_balanced_objectives
)

__version__ = "1.0.0"
__author__ = "Ares Trading System"

__all__ = [
    # Main consolidated pipeline (legacy)
    'UnifiedDataDrivenPipeline',
    'ConsolidatedPipelineResult',
    'create_unified_pipeline',
    'process_with_unified_pipeline',
    
    # Refactored pipeline (recommended for new usage)
    'RefactoredUnifiedPipeline',
    'RefactoredPipelineResult',
    'create_refactored_pipeline',
    'create_full_pipeline',
    'create_blank_pipeline',
    'create_light_pipeline',
    
    # Configuration
    'UnifiedPipelineConfig',
    'create_default_config',
    'create_high_performance_config',
    'create_memory_efficient_config',
    'create_fast_config',
    
    # Simplified configuration
    'create_full_config',
    'create_blank_config',
    'create_light_config',
    'create_config_by_intensity',
    'list_available_intensities',
    'PipelineIntensity',
    
    # Time series CV
    'PurgedEmbargoedWalkForwardCV',
    'PurgedEmbargoedConfig',
    'TimeSeriesSplit',
    'create_purged_embargoed_cv',
    'validate_time_series_splits',
    
    # Statistical analysis
    'StatisticalAnalysisFramework',
    'DataCharacteristics',
    'PatternAnalysis',
    'RelationshipAnalysis',
    
    # Feature selection
    'MultiObjectiveFeatureSelector',
    'create_default_objectives',
    'create_performance_objectives',
    'create_stability_objectives',
    'create_balanced_objectives'
]