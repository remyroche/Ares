"""
Pre-training Settings Module

This module provides configuration settings for pre-training steps.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
from src.training.config.data_locator import DataLocatorConfig

logger = logging.getLogger(__name__)

@dataclass
class PreTrainingSettings:
    """Configuration settings for pre-training steps."""
    
    # Feature generation settings
    max_features: int = 1000
    feature_selection_method: str = "mutual_info"
    min_feature_importance: float = 0.01
    
    # Data quality settings
    min_data_quality_score: float = 0.8
    max_missing_ratio: float = 0.1
    outlier_detection_enabled: bool = True
    
    # Performance settings
    parallel_processing: bool = True
    max_workers: int = 4
    memory_limit_gb: float = 8.0
    
    # Validation settings
    cross_validation_folds: int = 5
    validation_timeout_minutes: int = 30
    enable_early_stopping: bool = True
    
    # Regime settings
    regime: str = "bull"
    
    # Logging settings
    log_level: str = "INFO"
    enable_detailed_logging: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            'max_features': self.max_features,
            'feature_selection_method': self.feature_selection_method,
            'min_feature_importance': self.min_feature_importance,
            'min_data_quality_score': self.min_data_quality_score,
            'max_missing_ratio': self.max_missing_ratio,
            'outlier_detection_enabled': self.outlier_detection_enabled,
            'parallel_processing': self.parallel_processing,
            'max_workers': self.max_workers,
            'memory_limit_gb': self.memory_limit_gb,
            'cross_validation_folds': self.cross_validation_folds,
            'validation_timeout_minutes': self.validation_timeout_minutes,
            'enable_early_stopping': self.enable_early_stopping,
            'regime': self.regime,
            'log_level': self.log_level,
            'enable_detailed_logging': self.enable_detailed_logging
        }
    
    def to_data_locator_config(self) -> DataLocatorConfig:
        """Convert settings to DataLocatorConfig."""
        return DataLocatorConfig()

# Default settings instance
_default_settings = PreTrainingSettings()

def get_pre_training_settings() -> PreTrainingSettings:
    """Get pre-training settings."""
    return _default_settings

def update_pre_training_settings(**kwargs) -> PreTrainingSettings:
    """Update pre-training settings with new values."""
    global _default_settings
    
    # Create new settings with updated values
    new_settings = PreTrainingSettings(
        max_features=kwargs.get('max_features', _default_settings.max_features),
        feature_selection_method=kwargs.get('feature_selection_method', _default_settings.feature_selection_method),
        min_feature_importance=kwargs.get('min_feature_importance', _default_settings.min_feature_importance),
        min_data_quality_score=kwargs.get('min_data_quality_score', _default_settings.min_data_quality_score),
        max_missing_ratio=kwargs.get('max_missing_ratio', _default_settings.max_missing_ratio),
        outlier_detection_enabled=kwargs.get('outlier_detection_enabled', _default_settings.outlier_detection_enabled),
        parallel_processing=kwargs.get('parallel_processing', _default_settings.parallel_processing),
        max_workers=kwargs.get('max_workers', _default_settings.max_workers),
        memory_limit_gb=kwargs.get('memory_limit_gb', _default_settings.memory_limit_gb),
        cross_validation_folds=kwargs.get('cross_validation_folds', _default_settings.cross_validation_folds),
        validation_timeout_minutes=kwargs.get('validation_timeout_minutes', _default_settings.validation_timeout_minutes),
        enable_early_stopping=kwargs.get('enable_early_stopping', _default_settings.enable_early_stopping),
        regime=kwargs.get('regime', _default_settings.regime),
        log_level=kwargs.get('log_level', _default_settings.log_level),
        enable_detailed_logging=kwargs.get('enable_detailed_logging', _default_settings.enable_detailed_logging)
    )
    
    _default_settings = new_settings
    return _default_settings

def reset_pre_training_settings() -> PreTrainingSettings:
    """Reset pre-training settings to defaults."""
    global _default_settings
    _default_settings = PreTrainingSettings()
    return _default_settings