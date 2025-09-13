"""
Data processing module for ML common utilities.
"""

from .regime_processing import RegimeProcessor
from .feature_preparation import FeaturePreparator
# Import lightly to avoid heavy init costs; expose EnhancedDataLabeler directly
from .data_labeling import EnhancedDataLabeler

# Provide a lightweight LabelingConfig alias for callers to avoid ImportError from package init
try:
    from .data_labeling import TripleBarrierConfig as LabelingConfig  # Backward/alias
except Exception:
    LabelingConfig = None  # type: ignore

__all__ = [
    'RegimeProcessor',
    'FeaturePreparator',
    'EnhancedDataLabeler',
    'LabelingConfig'
]