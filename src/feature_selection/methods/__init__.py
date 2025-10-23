"""Feature selection methods (filter, wrapper, embedded, stability)."""

# MRMR is now available through VectorBT implementation
from ..vectorbt_extensions.vectorbt_mrmr_selector import VectorBTMRMRSelector as MRMRSelector
from .stability_selection import ElasticNetStabilitySelector, StabilityAnalyzer
from .wrapper_methods import RecursiveFeatureEliminator
from .importance import FeatureImportanceRanker
# Regularization is now available through VectorBT implementation
from ..vectorbt_extensions.vectorbt_regularization import VectorBTRegularizationSelector as FeatureRegularizationSelector
from ..vectorbt_extensions.vectorbt_config import VectorBTFeatureSelectionConfig as FeatureRegularizationConfig

def create_feature_regularization_selector(config=None):
    """Create feature regularization selector using VectorBT implementation."""
    return FeatureRegularizationSelector(config)

__all__ = [
    'MRMRSelector',
    'ElasticNetStabilitySelector',
    'StabilityAnalyzer',
    'RecursiveFeatureEliminator',
    'FeatureImportanceRanker',
    'FeatureRegularizationSelector',
    'FeatureRegularizationConfig',
    'create_feature_regularization_selector',
]
