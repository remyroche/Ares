"""
Pre-Training Steps Package

This package contains the feature engineering steps that were moved from market_analysis:
- multi_horizon_profit_labeler: Apply multi-horizon profit labeling
- feature_lookback_optimization: Optimize feature lookback periods
- pid_based_feature_generation: PID-based feature generation with interaction, polynomial, and cross-timeframe features
- final_feature_selection: Final multi-stage feature selection (120→100→80→60)
"""

# Export main sub-pipeline
from .sub_pipeline import (
    PreTrainingSubPipeline,
    SubPipelineConfig,
    SubPipelineResult,
    execute_pre_training_pipeline
)

# Export individual components
from .multi_horizon_profit_labeler import (
    MultiHorizonProfitLabeler,
    MultiHorizonConfig
)

from .pid_based_feature_generation import (
    PIDBasedFeatureGeneration,
    PIDBasedFeatureGenerationConfig,
    PIDBasedFeatureGenerationResult,
    generate_pid_features
)

__all__ = [
    'PreTrainingSubPipeline',
    'SubPipelineConfig',
    'SubPipelineResult',
    'execute_pre_training_pipeline',
    'MultiHorizonProfitLabeler',
    'MultiHorizonConfig',
    'PIDBasedFeatureGeneration',
    'PIDBasedFeatureGenerationConfig',
    'PIDBasedFeatureGenerationResult',
    'generate_pid_features'
]