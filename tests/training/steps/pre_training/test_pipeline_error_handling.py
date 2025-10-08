import asyncio
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from src.training.steps.pre_training.sub_pipeline import (
    PreTrainingSubPipeline,
    SubPipelineConfig,
    SubPipelineFailure,
    SubPipelineResult,
    SubPipelineStatus,
)


class _StubLocator:
    def __init__(self, base_path):
        self._path = base_path

    def data_path(self, _key: str):
        return self._path

    def cache_path(self, _key: str):
        return self._path

    def artifacts_path(self, _key: str, ensure_exists: bool = False):
        return self._path

    def generated_path(self, _key: str, ensure_exists: bool = False):
        return self._path


def _seed_stub(seed: int) -> SimpleNamespace:
    return SimpleNamespace(seed=seed, numpy=SimpleNamespace(), python=SimpleNamespace())


def _build_result(
    name: str,
    *,
    success: bool,
    message: str | None = None,
    artifacts: Dict[str, Any] | None = None,
    errors: List[str] | None = None,
) -> SubPipelineResult:
    result = SubPipelineResult(
        sub_pipeline_name=name,
        status=SubPipelineStatus.COMPLETED if success else SubPipelineStatus.FAILED,
        start_time=datetime.utcnow(),
    )
    result.end_time = result.start_time
    result.duration_seconds = 0.0
    result.success = success
    result.artifacts = artifacts or {}
    result.metadata = {'run_metadata': {}}
    result.warnings = []
    result.errors = list(errors or [])
    if not success:
        failure_message = message or f"{name} failed"
        result.error_message = failure_message
        failure = SubPipelineFailure(
            error_code=f"{name.upper()}_ERR",
            message=failure_message,
            step=name,
        )
        result.failure = failure
        if failure_message not in result.errors:
            result.errors.append(failure_message)
    return result


@pytest.fixture(autouse=True)
def _patch_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(
        PreTrainingSubPipeline,
        '_gather_run_metadata',
        lambda self, cfg, seed=None: {'run_id': 'test', 'start_time_utc': '2024-01-01T00:00:00Z'},
    )
    monkeypatch.setattr(PreTrainingSubPipeline, '_resolve_random_seed', lambda self, cfg: 0)
    monkeypatch.setattr(
        'src.training.steps.pre_training.sub_pipeline.seed_rngs',
        lambda seed: _seed_stub(seed),
    )
    monkeypatch.setattr(PreTrainingSubPipeline, '_emit_effective_configuration', lambda self, cfg: None)
    monkeypatch.setattr(PreTrainingSubPipeline, '_create_metrics_sink', lambda self, cfg: None)
    monkeypatch.setattr(
        PreTrainingSubPipeline,
        '_resolve_data_locator',
        lambda self, cfg: _StubLocator(tmp_path),
    )
    yield


def test_pipeline_halts_on_first_failure(monkeypatch):
    async def _fail_step(self, config, run_metadata):
        return _build_result('multi_horizon_profit_labeler', success=False, message='step boom')

    async def _unexpected(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError('subsequent step should not run when continue_on_error is false')

    monkeypatch.setattr(
        PreTrainingSubPipeline,
        '_execute_multi_horizon_profit_labeler',
        _fail_step,
    )
    monkeypatch.setattr(
        PreTrainingSubPipeline,
        '_execute_feature_lookback_optimization',
        _unexpected,
    )
    monkeypatch.setattr(
        PreTrainingSubPipeline,
        '_execute_interactive_feature_generation',
        _unexpected,
    )
    monkeypatch.setattr(
        PreTrainingSubPipeline,
        '_execute_final_feature_selection',
        _unexpected,
    )

    pipeline = PreTrainingSubPipeline()
    result = asyncio.run(pipeline.execute_pipeline(SubPipelineConfig()))

    assert result['success'] is False
    assert result['completed_steps'] == 0
    assert any('step boom' in message for message in result['errors'])
    assert result['results']['multi_horizon_profit_labeler'] == {}
    assert 'feature_lookback_optimization' not in result['results']
    assert 'step boom' in (result.get('error_summary') or '')


def test_pipeline_continues_when_flag_enabled(monkeypatch):
    async def _fail_first(self, config, run_metadata):
        return _build_result('multi_horizon_profit_labeler', success=False, message='fail first')

    async def _succeed(name: str, artifacts_key: str, self, config, run_metadata):
        artifacts = {artifacts_key: {'status': 'ok'}}
        return _build_result(name, success=True, artifacts=artifacts)

    monkeypatch.setattr(
        PreTrainingSubPipeline,
        '_execute_multi_horizon_profit_labeler',
        _fail_first,
    )
    monkeypatch.setattr(
        PreTrainingSubPipeline,
        '_execute_feature_lookback_optimization',
        lambda self, cfg, meta: _succeed('feature_lookback_optimization', 'feature_lookback_optimization_result', self, cfg, meta),
    )
    monkeypatch.setattr(
        PreTrainingSubPipeline,
        '_execute_interactive_feature_generation',
        lambda self, cfg, meta: _succeed('interactive_feature_generation', 'interactive_feature_generation_result', self, cfg, meta),
    )
    monkeypatch.setattr(
        PreTrainingSubPipeline,
        '_execute_final_feature_selection',
        lambda self, cfg, meta: _succeed('final_feature_selection', 'final_feature_selection_result', self, cfg, meta),
    )

    pipeline = PreTrainingSubPipeline()
    config = SubPipelineConfig(pipeline={'continue_on_error': True})
    result = asyncio.run(pipeline.execute_pipeline(config))

    assert result['success'] is False
    assert result['completed_steps'] == 3
    assert 'feature_lookback_optimization' in result['results']
    assert 'interactive_feature_generation' in result['results']
    assert 'final_feature_selection' in result['results']
    assert any('fail first' in message for message in result['errors'])
    assert 'fail first' in (result.get('error_summary') or '')
    assert result['error_code']
