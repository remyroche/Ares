"""
Core components of the Unified Data-Driven Feature Pipeline
"""

from .unified_pipeline import (
    UnifiedDataDrivenPipeline,
    FeaturePipelineResult,
    create_unified_pipeline,
    process_features
)

from .config import (
    UnifiedPipelineConfig,
    create_default_config,
    create_high_performance_config,
    create_memory_efficient_config,
    create_fast_config
)

__all__ = [
    'UnifiedDataDrivenPipeline',
    'FeaturePipelineResult',
    'create_unified_pipeline',
    'process_features',
    'UnifiedPipelineConfig',
    'create_default_config',
    'create_high_performance_config',
    'create_memory_efficient_config',
    'create_fast_config'
]