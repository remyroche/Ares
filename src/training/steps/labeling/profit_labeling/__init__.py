"""
Profit Labeling Package

This package provides profit labeling functionality for the pre-training pipeline.
"""

from .volatility_aware_labeler import (
    VolatilityAwareMultiHorizonLabeler,
    VolatilityAwareConfig,
    LabelingResult,
    LabelDefinitionType,
    create_enhanced_analyst_labeler
)

__all__ = [
    "VolatilityAwareMultiHorizonLabeler",
    "VolatilityAwareConfig", 
    "LabelingResult",
    "LabelDefinitionType",
    "create_enhanced_analyst_labeler"
]
