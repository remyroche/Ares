"""
Pre-Training Components Package.

This package contains all the components for the pre-training pipeline stage.
"""

from .base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult, ComponentError
from .contracts import (
    PipelineState,
    ValidationResults,
    MultiHorizonLabelingResult,
    StandardizedLabelingOutput,
    MultiHorizonArtifacts,
    FeatureLookbackSummary,
    FeatureLookbackOptimizationResult,
    FeatureLookbackArtifacts,
    FinalSelectionResult,
    FinalSelectionArtifacts,
)

__all__ = [
    'BasePreTrainingComponent',
    'ComponentConfig',
    'ComponentResult',
    'ComponentError',
    'ComponentFactory',
    'FeatureLookbackOptimizationComponent',
    'FinalFeatureSelectionComponent',
    'PipelineState',
    'ValidationResults',
    'MultiHorizonLabelingResult',
    'StandardizedLabelingOutput',
    'MultiHorizonArtifacts',
    'FeatureLookbackSummary',
    'FeatureLookbackOptimizationResult',
    'FeatureLookbackArtifacts',
    'FinalSelectionResult',
    'FinalSelectionArtifacts',
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
