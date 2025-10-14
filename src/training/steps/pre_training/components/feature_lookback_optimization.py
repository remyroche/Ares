"""
Feature Lookback Optimization Component Import.

This module is deprecated and will be removed.
"""

from ..logging_utils import PreTrainingEventLogger, configure_pre_training_logging

_event_logger = PreTrainingEventLogger(configure_pre_training_logging())
_event_logger.warning(
    "FeatureLookbackOptimizationComponent has been removed",
    context={'step': 'component.feature_lookback_import'},
)

# Component has been removed
