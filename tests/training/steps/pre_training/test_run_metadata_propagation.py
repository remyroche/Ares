import asyncio
from datetime import datetime
import sys
import types
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from src.training.common.artifact_persistence import SaveReport
from src.training.steps.pre_training.components import base_component as base_component_module
from src.training.steps.pre_training.components.base_component import (
    BasePreTrainingComponent,
    ComponentResult,
    ComponentConfig,
)


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
    def __init__(self, base_dir: Path) -> None:
        self.saved_payloads: list[Dict[str, Any]] = []
        self.base_paths = {"artifacts": base_dir}


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
async def test_run_metadata_printed_and_persisted(monkeypatch, capfd, tmp_path):
    stub_manager = _StubArtifactManager(tmp_path)
    monkeypatch.setattr(base_component_module, 'get_artifact_manager', lambda: stub_manager)

    def _persist_artifacts(*, component_name, artifacts, metadata, base_dir, logger, **_):
        stub_manager.saved_payloads.append({'metadata': dict(metadata or {})})
        return SaveReport(
            paths={
                'artifact': str(base_dir / f"{component_name}_artifact.json"),
                'metadata': str(base_dir / f"{component_name}_metadata.json"),
            },
            bytes={'artifact': 1, 'metadata': 1},
            duration=0.0,
            checksum={'artifact': 'abc', 'metadata': 'def'},
            correlation_id='test-correlation',
        )

    monkeypatch.setattr(base_component_module, 'persist_artifacts', _persist_artifacts)

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
        lambda self, cfg, *args: dict(run_metadata),
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
    monkeypatch.setattr(
        PreTrainingSubPipeline,
        '_prepare_interactive_training_input',
        lambda self, pipeline_state: {},
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
