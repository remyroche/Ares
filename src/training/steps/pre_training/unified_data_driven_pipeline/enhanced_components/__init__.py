"""
Enhanced Components for Unified Data-Driven Pipeline

This module provides enhanced components that integrate advanced features
for comprehensive data analysis, optimization, and validation.
"""

from .enhanced_walk_forward_validation import (
    AdvancedWalkForwardValidator,
    AdvancedWalkForwardConfig,
    AdvancedTimeSeriesSplit
)

from .enhanced_statistical_framework import (
    EnhancedStatisticalFramework,
    HypothesisTestResult,
    MultipleTestingResult,
    StatisticalAnalysisResult
)

from .enhanced_schema_validation import (
    EnhancedSchemaValidator,
    ValidationResult,
    SchemaDefinition,
    TemporalAlignmentResult
)

from .enhanced_caching_integration import (
    EnhancedCachingIntegration,
    CacheEntry,
    CacheStats,
    ArtifactMetadata
)

from .gpu_optimizations import (
    GPUOptimizer,
    GPUConfig,
    GPUOperationResult
)

# Note: Enhanced unified pipeline has been consolidated into the main pipeline
# The enhanced pipeline classes are no longer available as separate components

__all__ = [
    # Walk-forward validation
    'AdvancedWalkForwardValidator',
    'AdvancedWalkForwardConfig',
    'AdvancedTimeSeriesSplit',

    # Statistical framework
    'EnhancedStatisticalFramework',
    'HypothesisTestResult',
    'MultipleTestingResult',
    'StatisticalAnalysisResult',

    # Schema validation
    'EnhancedSchemaValidator',
    'ValidationResult',
    'SchemaDefinition',
    'TemporalAlignmentResult',

    # Caching integration
    'EnhancedCachingIntegration',
    'CacheEntry',
    'CacheStats',
    'ArtifactMetadata',

    # GPU optimizations
    'GPUOptimizer',
    'GPUConfig',
    'GPUOperationResult'
]
