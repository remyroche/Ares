import asyncio
from datetime import datetime
import sys
import types
from typing import Any, Dict

import pandas as pd
import pytest

from src.training.steps.pre_training.components.base_component import (
    BasePreTrainingComponent,
    ComponentResult,
    ComponentConfig,
)


components_stub = types.ModuleType("components_stub")


class _StubFactory:  # pragma: no cover - placeholder
    @staticmethod
    def create_component(*_args, **_kwargs):
        raise NotImplementedError


components_stub.ComponentFactory = _StubFactory
components_stub.ComponentConfig = ComponentConfig
components_stub.ComponentResult = ComponentResult
components_stub.BasePreTrainingComponent = BasePreTrainingComponent
sys.modules['src.training.steps.pre_training.components'] = components_stub

from src.training.steps.pre_training.sub_pipeline import (
    PreTrainingSubPipeline,
    SubPipelineConfig,
    SubPipelineResult,
    SubPipelineStatus,
    ComponentFactory,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _StubArtifactManager:
    def __init__(self) -> None:
        self.saved_payloads: list[Dict[str, Any]] = []

    def save_artifact(self, data: Any, base_name: str, extension: str = ".json", **_: Any) -> str:
        self.saved_payloads.append(data)
        return f"/tmp/{base_name}{extension}"


class _StubLabelerComponent(BasePreTrainingComponent):
    def get_required_artifacts(self) -> list[str]:
        return ['multi_horizon_labeling_result']

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        df = pd.DataFrame({'close': [1, 2, 3]})
        artifacts = {
            'multi_horizon_labeling_result': {
                'labeled_data': df,
                'metadata': {'source': 'stub'},
            }
        }
        metadata = {'label_distribution': {'positive': 2, 'negative': 1}}
        await self.save_artifacts(artifacts, metadata)
        return ComponentResult(success=True, artifacts=artifacts, metadata=metadata)


def _raise_unexpected(name: str):
    raise AssertionError(f"Unexpected component request: {name}")


def _simple_step(name: str, key: str) -> Any:
    async def _run(self, config: SubPipelineConfig, run_metadata: Dict[str, Any]) -> SubPipelineResult:
        result = SubPipelineResult(
            sub_pipeline_name=name,
            status=SubPipelineStatus.COMPLETED,
            start_time=datetime.utcnow(),
        )
        result.success = True
        result.end_time = result.start_time
        result.duration_seconds = 0.0
        result.artifacts = {key: {'status': 'ok'}}
        result.metadata = {'run_metadata': dict(run_metadata)}
        return result

    return _run


@pytest.mark.anyio("asyncio")
async def test_run_metadata_printed_and_persisted(monkeypatch, capfd):
    stub_manager = _StubArtifactManager()
    monkeypatch.setattr(
        'src.training.steps.pre_training.components.base_component.get_artifact_manager',
        lambda: stub_manager,
    )

    run_metadata = {
        'git_sha': 'test-sha',
        'config_hash': 'hash-123',
        'data_snapshot_id': 'snapshot-test',
        'rng_seed': 42,
        'host_name': 'ares-test-host',
        'start_time_utc': '2024-01-01T00:00:00Z',
        'end_time_utc': None,
        'duration_seconds': None,
    }

    monkeypatch.setattr(
        PreTrainingSubPipeline,
        '_gather_run_metadata',
        lambda self, cfg: dict(run_metadata),
    )

    monkeypatch.setattr(
        ComponentFactory,
        'create_component',
        lambda name, cfg: (
            _StubLabelerComponent(cfg)
            if name == 'multi_horizon_profit_labeler'
            else (_raise_unexpected(name))
        ),
    )

    monkeypatch.setattr(
        PreTrainingSubPipeline,
        '_execute_feature_lookback_optimization',
        _simple_step('feature_lookback_optimization', 'feature_lookback_optimization_result'),
    )
    monkeypatch.setattr(
        PreTrainingSubPipeline,
        '_execute_interactive_feature_generation',
        _simple_step('interactive_feature_generation', 'interactive_feature_generation_result'),
    )
    monkeypatch.setattr(
        PreTrainingSubPipeline,
        '_execute_final_feature_selection',
        _simple_step('final_feature_selection', 'final_feature_selection_result'),
    )

    pipeline = PreTrainingSubPipeline()
    config = SubPipelineConfig(custom_params={'rng_seed': 42, 'data_snapshot_id': 'snapshot-test'})

    result = await pipeline.execute_pipeline(config)

    assert result['success'] is True
    assert stub_manager.saved_payloads, 'artifact should be saved with metadata'

    saved_payload = stub_manager.saved_payloads[0]
    assert saved_payload['metadata']['run_metadata'] == run_metadata

    captured = capfd.readouterr()
    assert 'Run metadata' in captured.out
    assert '"git_sha": "test-sha"' in captured.out

    summary_line = next(
        (line for line in captured.out.splitlines() if 'Run metadata summary' in line),
        '',
    )
    assert summary_line
