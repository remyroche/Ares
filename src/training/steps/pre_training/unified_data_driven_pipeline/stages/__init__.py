"""
Pipeline Stages for Unified Data-Driven Pipeline

This module contains the modular stages for the unified pipeline:
- DataValidationStage: Comprehensive data validation and quality assessment
- FeatureGenerationStage: Feature generation and engineering
- FeatureSelectionStage: Feature selection and optimization
- OptimizationStage: Period optimization, lookback optimization, and interaction generation
"""

from .data_validation_stage import (
    DataValidationStage,
    DataValidationResult,
    create_data_validation_stage
)

from .feature_generation_stage import (
    FeatureGenerationStage,
    FeatureGenerationResult,
    create_feature_generation_stage
)

from .feature_selection_stage import (
    FeatureSelectionStage,
    FeatureSelectionStageResult,
    create_feature_selection_stage
)

from .optimization_stage import (
    OptimizationStage,
    OptimizationStageResult,
    create_optimization_stage
)

__all__ = [
    # Data validation stage
    'DataValidationStage',
    'DataValidationResult',
    'create_data_validation_stage',
    
    # Feature generation stage
    'FeatureGenerationStage',
    'FeatureGenerationResult',
    'create_feature_generation_stage',
    
    # Feature selection stage
    'FeatureSelectionStage',
    'FeatureSelectionStageResult',
    'create_feature_selection_stage',
    
    # Optimization stage
    'OptimizationStage',
    'OptimizationStageResult',
    'create_optimization_stage'
]