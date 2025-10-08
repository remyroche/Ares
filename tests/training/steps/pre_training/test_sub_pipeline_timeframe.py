import pytest

from src.training.steps.pre_training import sub_pipeline as sub_pipeline_module
from src.training.steps.pre_training.sub_pipeline import SubPipelineConfig


def test_timeframe_defaults_to_primary(monkeypatch):
    monkeypatch.setattr(sub_pipeline_module, 'get_primary_timeframe', lambda: '42m')

    config = SubPipelineConfig()

    assert config.timeframe == '42m'


def test_custom_params_timeframe_overrides_global(monkeypatch):
    monkeypatch.setattr(sub_pipeline_module, 'get_primary_timeframe', lambda: '42m')

    config = SubPipelineConfig(custom_params={'timeframe': '5m'})

    assert config.timeframe == '5m'
    assert config.custom_params['timeframe'] == '5m'


def test_explicit_timeframe_takes_precedence(monkeypatch):
    monkeypatch.setattr(sub_pipeline_module, 'get_primary_timeframe', lambda: '42m')

    config = SubPipelineConfig(timeframe='30m', custom_params={'timeframe': '5m'})

    assert config.timeframe == '30m'
    assert config.custom_params['timeframe'] == '30m'


def test_pipeline_override_applies_when_explicit_missing(monkeypatch):
    monkeypatch.setattr(sub_pipeline_module, 'get_primary_timeframe', lambda: '42m')

    config = SubPipelineConfig(pipeline={'timeframe': '90m'})

    assert config.timeframe == '90m'


def test_analyst_role_forces_sixty_minutes(monkeypatch):
    monkeypatch.setattr(sub_pipeline_module, 'get_primary_timeframe', lambda: '42m')

    config = SubPipelineConfig(custom_params={'role': 'analyst', 'timeframe': '5m'})

    assert config.timeframe == '60m'
    assert config.custom_params['timeframe'] == '60m'


def test_analyst_flag_forces_sixty_minutes(monkeypatch):
    monkeypatch.setattr(sub_pipeline_module, 'get_primary_timeframe', lambda: '42m')

    config = SubPipelineConfig(pipeline={'analyst_mode': True, 'timeframe': '5m'})

    assert config.timeframe == '60m'
