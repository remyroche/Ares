"""Tests for the pre-training component registry and factory."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from src.training.steps.pre_training.components.base_component import (
    BasePreTrainingComponent,
    ComponentResult,
)
from src.training.steps.pre_training.components.component_factory import (
    ComponentFactory,
    register_component,
    register_unavailable_component,
)


class _BaseTestComponent(BasePreTrainingComponent):
    """Simple test component used for registry validations."""

    def get_required_artifacts(self) -> list[str]:
        return []

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        return ComponentResult(success=True, artifacts={}, metadata={})


def _cleanup(name: str) -> None:
    """Ensure test registrations do not leak between test cases."""

    ComponentFactory._unregister_for_testing(name)


def test_dynamic_component_registration_creates_component() -> None:
    """Dynamically registering a component should allow factory creation."""

    name = "test_dynamic_component"

    @register_component(name)
    class DynamicComponent(_BaseTestComponent):
        pass

    try:
        component = ComponentFactory.create_component(name)
        assert isinstance(component, DynamicComponent)
    finally:
        _cleanup(name)


def test_duplicate_component_registration_raises() -> None:
    """Registering the same component name twice should raise an error."""

    name = "test_duplicate_component"

    first_decorator = register_component(name)

    class FirstComponent(_BaseTestComponent):
        pass

    first_decorator(FirstComponent)

    second_decorator = register_component(name)

    class SecondComponent(_BaseTestComponent):
        pass

    try:
        with pytest.raises(ValueError):
            second_decorator(SecondComponent)
    finally:
        _cleanup(name)


def test_optional_component_unavailable_is_handled_gracefully() -> None:
    """Creating an unavailable optional component should raise a clear error."""

    name = "test_optional_component"
    register_unavailable_component(name, error="missing dependency")

    try:
        assert not ComponentFactory.is_component_available(name)
        with pytest.raises(ValueError) as excinfo:
            ComponentFactory.create_component(name)
        assert "not available" in str(excinfo.value)
    finally:
        _cleanup(name)
