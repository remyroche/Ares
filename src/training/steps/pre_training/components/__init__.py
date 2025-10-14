"""
Pre-Training Components Package.

This package contains all the components for the pre-training pipeline stage.
"""

from .base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult, ComponentError

__all__ = [
    'BasePreTrainingComponent',
    'ComponentConfig',
    'ComponentResult',
    'ComponentError',
    'ComponentFactory',
    'FinalFeatureSelectionComponent'
]


def __getattr__(name):
    if name == 'ComponentFactory':
        from .component_factory import ComponentFactory  # type: ignore
        return ComponentFactory
    if name == 'FinalFeatureSelectionComponent':
        from .final_feature_selection import FinalFeatureSelectionComponent  # type: ignore
        return FinalFeatureSelectionComponent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")