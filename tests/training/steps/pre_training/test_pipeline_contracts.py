import pytest

from src.training.steps.pre_training.components.base_component import ComponentResult
from src.training.steps.pre_training.components.contracts import (
    PipelineState,
    MultiHorizonArtifacts,
    validate_multi_horizon_artifacts,
    pipeline_state_from_mapping,
)


def test_pipeline_state_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown pipeline state keys"):
        pipeline_state_from_mapping({'symbol': 'ETHUSDT', 'exchange': 'binance', 'timeframe': '1h', 'data_dir': 'historical_data', 'unexpected': 1})


def test_component_result_rejects_non_string_artifact_keys():
    with pytest.raises(ValueError, match="Artifact keys must be strings"):
        ComponentResult(success=True, artifacts={1: {}})


def test_pipeline_state_validates_strings():
    with pytest.raises(ValueError):
        PipelineState(symbol="", exchange="binance", timeframe="1h", data_dir="historical_data")


def test_multi_horizon_validator_requires_primary_payload():
    with pytest.raises(ValueError, match="multi_horizon_labeling_result"):
        validate_multi_horizon_artifacts({'labeling_report': {}})


def test_pipeline_state_from_mapping_legacy_artifacts():
    artifacts: MultiHorizonArtifacts = {
        'multi_horizon_labeling_result': {'labeled_data': {}},
        'labeling_report': {},
    }
    state = pipeline_state_from_mapping({
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '1h',
        'data_dir': 'historical_data',
        **artifacts,
    })
    assert state.multi_horizon is not None
    assert 'multi_horizon_labeling_result' in state
