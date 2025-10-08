"""Registration helper for the PID-based feature generation component."""

from __future__ import annotations

from .component_factory import register_component, register_unavailable_component

try:
    from ..pid_based_feature_generation.pid_based_feature_generation_component import (
        PIDBasedFeatureGenerationComponent,
    )
except ImportError as exc:  # pragma: no cover - optional dependency guard
    register_unavailable_component(
        'pid_based_feature_generation',
        error=str(exc),
        source=__name__,
    )
else:
    register_component('pid_based_feature_generation')(PIDBasedFeatureGenerationComponent)
