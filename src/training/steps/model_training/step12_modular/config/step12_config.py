"""
Step 12 Modular: Configuration

This module contains configuration classes and constants for Step 12.
"""

from typing import Dict, Any

from ..base.imports import validate_step12_imports
from ..base.logger import setup_step12_logger

class Step12Config:
    """Configuration class for Step 12 Analyst Enhancement."""

    def __init__(self, **kwargs):
        """Initialize Step 12 configuration."""
        # Base configuration
        self.blank_training_lookback_days = kwargs.get('blank_training_lookback_days', 1095)
        self.default_timeframe = kwargs.get('default_timeframe', '1m')
        self.default_exchange = kwargs.get('default_exchange', 'BINANCE')

        # Regime-specific settings
        self.regime_specific_optimization = kwargs.get('regime_specific_optimization', True)
        self.regime_specific_feature_selection = kwargs.get('regime_specific_feature_selection', True)
        self.regime_specific_hyperparameter_optimization = kwargs.get('regime_specific_hyperparameter_optimization', True)
        self.regime_specific_validation = kwargs.get('regime_specific_validation', True)
        self.regime_specific_logging = kwargs.get('regime_specific_logging', True)

        # Processing parameters
        self.min_regime_samples = kwargs.get('min_regime_samples', 1000)
        self.regime_validation_split = kwargs.get('regime_validation_split', 0.2)
        self.regime_optimization_trials = kwargs.get('regime_optimization_trials', 50)
        self.regime_feature_selection_threshold = kwargs.get('regime_feature_selection_threshold', 0.01)
        self.regime_parallel_processing = kwargs.get('regime_parallel_processing', True)
        self.regime_memory_optimization = kwargs.get('regime_memory_optimization', True)

        # Metadata columns
        self.metadata_columns = [
            'timestamp', 'exchange', 'symbol', 'timeframe', 'split',
            'year', 'month', 'day', 'day_of_week', 'day_of_month',
            'quarter', 'composite_cluster_id'
        ]

        # Label columns
        self.label_columns = {
            'label', 'target', 'y', 'class', 'signal', 'prediction'
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'blank_training_lookback_days': self.blank_training_lookback_days,
            'default_timeframe': self.default_timeframe,
            'default_exchange': self.default_exchange,
            'regime_specific_optimization': self.regime_specific_optimization,
            'regime_specific_feature_selection': self.regime_specific_feature_selection,
            'regime_specific_hyperparameter_optimization': self.regime_specific_hyperparameter_optimization,
            'regime_specific_validation': self.regime_specific_validation,
            'regime_specific_logging': self.regime_specific_logging,
            'min_regime_samples': self.min_regime_samples,
            'regime_validation_split': self.regime_validation_split,
            'regime_optimization_trials': self.regime_optimization_trials,
            'regime_feature_selection_threshold': self.regime_feature_selection_threshold,
            'regime_parallel_processing': self.regime_parallel_processing,
            'regime_memory_optimization': self.regime_memory_optimization,
            'metadata_columns': self.metadata_columns,
            'label_columns': self.label_columns
        }

def create_step12_config(**kwargs) -> Step12Config:
    """Create a Step 12 configuration instance."""
    return Step12Config(**kwargs)

DEFAULT_CONFIG = create_step12_config()

__all__ = ['Step12Config', 'create_step12_config', 'DEFAULT_CONFIG']
