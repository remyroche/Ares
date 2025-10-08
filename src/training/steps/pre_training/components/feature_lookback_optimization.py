"""
Feature Lookback Optimization Component Import.

This module provides the FeatureLookbackOptimizationComponent for the market analysis pipeline.
"""

from ..feature_lookback_optimization.feature_lookback_optimization import FeatureLookbackOptimizationComponent
from ..logging_utils import PreTrainingEventLogger, configure_pre_training_logging

_event_logger = PreTrainingEventLogger(configure_pre_training_logging())
_event_logger.info(
    "FeatureLookbackOptimizationComponent imported",
    context={'step': 'component.feature_lookback_import'},
)

__all__ = ['FeatureLookbackOptimizationComponent']
