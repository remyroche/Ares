"""
Shared utilities for HDBSCAN regime detection and clustering.

This package provides common utilities that eliminate redundancy between
NAS and TAS components, including feature preparation, configuration validation,
logging utilities, metrics calculation, and regime characteristics.
"""

from .features import prepare_market_features, FeatureConfig, FeaturePreparationResult
from .feature_filters import (
    winsorize_frame,
    filter_low_variance,
    prune_correlated_features,
    apply_quality_thresholds,
)
from .config import (
    validate_regime_count, normalize_weights, validate_algorithm_type,
    create_default_config, create_adaptive_config, ConfigValidator, BaseConfig, NASConfig, TASConfig, HybridConfig
)
from .logging_utils import (
    log_execution, log_performance, LoggingContext,
    get_logger, log_info, log_warning, log_error, log_success, log_debug
)
from .metrics import (
    calculate_consensus_metrics, calculate_disagreement_metrics,
    calculate_economic_scores, calculate_trading_scores, calculate_stability_scores,
    MetricsCalculator
)
from .characteristics import (
    create_regime_characteristics, generate_cluster_characteristics,
    CharacteristicsGenerator
)

__all__ = [
    # Features
    'prepare_market_features', 'FeatureConfig', 'FeaturePreparationResult',
    'winsorize_frame', 'filter_low_variance', 'prune_correlated_features', 'apply_quality_thresholds',

    # Configuration
    'validate_regime_count', 'normalize_weights', 'validate_algorithm_type',
    'create_default_config', 'create_adaptive_config', 'ConfigValidator', 'BaseConfig', 'NASConfig', 'TASConfig', 'HybridConfig',

    # Logging
    'log_execution', 'log_performance', 'LoggingContext',
    'get_logger', 'log_info', 'log_warning', 'log_error', 'log_success', 'log_debug',

    # Metrics
    'calculate_consensus_metrics', 'calculate_disagreement_metrics',
    'calculate_economic_scores', 'calculate_trading_scores', 'calculate_stability_scores',
    'MetricsCalculator',

    # Characteristics
    'create_regime_characteristics', 'generate_cluster_characteristics',
    'CharacteristicsGenerator'
]
