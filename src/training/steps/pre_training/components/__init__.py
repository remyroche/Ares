"""
Pre-Training Components Package.

This package contains all the components for the pre-training pipeline stage.
"""

from .base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult
from .contracts import (
    PipelineState,
    MultiHorizonArtifacts,
    FeatureLookbackArtifacts,
    InteractiveFeatureArtifacts,
    FinalSelectionArtifacts,
)

__all__ = [
    'BasePreTrainingComponent',
    'ComponentConfig',
    'ComponentResult',
    'PipelineState',
    'MultiHorizonArtifacts',
    'FeatureLookbackArtifacts',
    'InteractiveFeatureArtifacts',
    'FinalSelectionArtifacts',
    'ComponentFactory',
    'FeatureLookbackOptimizationComponent',
    'FinalFeatureSelectionComponent'
]


def __getattr__(name):
    if name == 'ComponentFactory':
        from .component_factory import ComponentFactory  # type: ignore
        return ComponentFactory
    if name == 'FeatureLookbackOptimizationComponent':
        from .feature_lookback_optimization import FeatureLookbackOptimizationComponent  # type: ignore
        return FeatureLookbackOptimizationComponent
    if name == 'FinalFeatureSelectionComponent':
        from .final_feature_selection import FinalFeatureSelectionComponent  # type: ignore
        return FinalFeatureSelectionComponent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")