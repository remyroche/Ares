"""
Shared utilities for market analysis steps.
"""

from .core import (
    # Core functions
    get_logger,
    prepare_market_features,
    validate_regime_count,
    normalize_weights,
    validate_algorithm_type,
    create_default_config,
    
    # Classes
    FeatureConfig,
    FeaturePreparationResult,
    BaseConfig,
    ConfigValidator,
    LoggingContext,
    MetricsCalculator,
    CharacteristicsGenerator,
    
    # Logging functions
    log_execution,
    log_performance,
    
    # Metrics functions
    calculate_consensus_metrics,
    calculate_disagreement_metrics,
    calculate_economic_scores,
    calculate_trading_scores,
    calculate_stability_scores,
    create_regime_characteristics,
    generate_cluster_characteristics,
)

from .calibration_registry import (
    get_quality_thresholds,
    get_calibration_config,
    validate_quality_metrics,
    get_algorithm_recommendations,
)

__all__ = [
    # Core functions
    'get_logger',
    'prepare_market_features',
    'validate_regime_count',
    'normalize_weights',
    'validate_algorithm_type',
    'create_default_config',
    
    # Classes
    'FeatureConfig',
    'FeaturePreparationResult',
    'BaseConfig',
    'ConfigValidator',
    'LoggingContext',
    'MetricsCalculator',
    'CharacteristicsGenerator',
    
    # Logging functions
    'log_execution',
    'log_performance',
    
    # Metrics functions
    'calculate_consensus_metrics',
    'calculate_disagreement_metrics',
    'calculate_economic_scores',
    'calculate_trading_scores',
    'calculate_stability_scores',
    'create_regime_characteristics',
    'generate_cluster_characteristics',
    
    # Calibration functions
    'get_quality_thresholds',
    'get_calibration_config',
    'validate_quality_metrics',
    'get_algorithm_recommendations',
]