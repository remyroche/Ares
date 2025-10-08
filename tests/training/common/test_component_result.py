import pytest

from src.training.common.component_result import ComponentError, ComponentResult


def test_component_result_success_has_no_error():
    result = ComponentResult(success=True)
    assert result.error is None
    assert result.metrics == {}
    assert result.warnings == []


def test_component_result_failure_requires_error():
    with pytest.raises(ValueError):
        ComponentResult(success=False)


def test_component_result_accepts_error_message():
    result = ComponentResult(success=False, error_message="boom")
    assert isinstance(result.error, ComponentError)
    assert str(result.error) == "boom"
    assert not result.success


def test_component_result_rejects_success_with_error():
    with pytest.raises(ValueError):
        ComponentResult(success=True, error=RuntimeError("bad"))
