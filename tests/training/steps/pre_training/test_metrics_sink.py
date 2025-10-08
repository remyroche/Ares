import csv
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import asyncio
import pandas as pd
import pytest

# Stub heavy component imports before loading the pipeline module.
import sys
import types

components_stub = types.ModuleType("src.training.steps.pre_training.components")


class _StubFactory:
    @staticmethod
    def create_component(*args, **kwargs):  # pragma: no cover - not used in tests
        raise NotImplementedError


class _StubConfig:  # pragma: no cover - placeholder for type compatibility
    pass


components_stub.ComponentFactory = _StubFactory
components_stub.ComponentConfig = _StubConfig
sys.modules.setdefault("src.training.steps.pre_training.components", components_stub)

from src.training.config.data_locator import DataLocatorConfig
from src.training.steps.pre_training.sub_pipeline import (
    PreTrainingSubPipeline,
    SubPipelineConfig,
    SubPipelineResult,
    SubPipelineStatus,
)


def _build_result(
    name: str,
    artifact_key: str,
    artifact_payload: Dict[str, Any],
    duration: float,
    metadata: Optional[Dict[str, Any]] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
) -> SubPipelineResult:
    result = SubPipelineResult(
        sub_pipeline_name=name,
        status=SubPipelineStatus.COMPLETED,
        start_time=datetime.now(),
    )
    result.success = True
    result.end_time = result.start_time + timedelta(seconds=duration)
    result.duration_seconds = duration
    result.artifacts = {artifact_key: artifact_payload}
    merged_metadata = dict(metadata or {})
    if run_metadata is not None:
        merged_metadata['run_metadata'] = dict(run_metadata)
    result.metadata = merged_metadata
    return result


def _patch_pipeline_steps(monkeypatch, durations=None):
    durations = durations or {}

    async def _mh(self, config, run_metadata):
        df = pd.DataFrame({'value': [1, 2, 3]})
        metadata = {'label_distribution': {'positive': 2, 'negative': 1}}
        return _build_result(
            'multi_horizon_profit_labeler',
            'multi_horizon_labeling_result',
            {'labeled_data': df},
            durations.get('mh', 5.0),
            metadata,
            run_metadata,
        )

    async def _flo(self, config, run_metadata):
        df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
        payload = {
            'optimized_features': {'feature_a': 0.7},
            'optimization_table': df,
        }
        metadata = {'label_distribution': {'positive': 1, 'negative': 1}}
        return _build_result(
            'feature_lookback_optimization',
            'feature_lookback_optimization_result',
            payload,
            durations.get('flo', 4.0),
            metadata,
            run_metadata,
        )

    async def _interactive(self, config, run_metadata):
        df = pd.DataFrame({'value': [1, 2]})
        payload = {'features': {'f1': 1}, 'generated_table': df}
        return _build_result(
            'interactive_feature_generation',
            'interactive_feature_generation_result',
            payload,
            durations.get('interactive', 3.0),
            {'label_distribution': {'positive': 1, 'neutral': 1}},
            run_metadata,
        )

    async def _ffs(self, config, run_metadata):
        df = pd.DataFrame({'value': [1, 2, 3, 4]})
        payload = {'selected_features': ['f1', 'f2'], 'evaluation': df}
        return _build_result(
            'final_feature_selection',
            'final_feature_selection_result',
            payload,
            durations.get('ffs', 2.0),
            {},
            run_metadata,
        )

    monkeypatch.setattr(PreTrainingSubPipeline, "_execute_multi_horizon_profit_labeler", _mh)
    monkeypatch.setattr(PreTrainingSubPipeline, "_execute_feature_lookback_optimization", _flo)
    monkeypatch.setattr(PreTrainingSubPipeline, "_execute_interactive_feature_generation", _interactive)
    monkeypatch.setattr(PreTrainingSubPipeline, "_execute_final_feature_selection", _ffs)


def test_pre_training_metrics_sink_csv(tmp_path, monkeypatch):
    _patch_pipeline_steps(monkeypatch)
    pipeline = PreTrainingSubPipeline()

    metrics_path = tmp_path / "metrics.csv"
    config = SubPipelineConfig(
        metrics_output_path=str(metrics_path),
        metrics_output_format="csv",
        metrics_prometheus_enabled=False,
    )

    result = asyncio.run(pipeline.execute_pipeline(config))
    assert result['success'] is True

    with metrics_path.open() as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    assert len(rows) == 5  # four steps + aggregate pipeline record
    for column in ['duration_seconds', 'row_count_total', 'memory_peak_mb', 'label_distribution_skew']:
        assert column in rows[0]

    step_row = next(row for row in rows if row['step_name'] == 'multi_horizon_profit_labeler')
    assert step_row['record_type'] == 'step'
    assert step_row['row_count_total'] == '3'
    assert float(step_row['label_distribution_skew']) == pytest.approx(1 / 3)

    pipeline_row = next(row for row in rows if row['record_type'] == 'pipeline')
    assert pipeline_row['completed_steps'] == '4'
    assert pipeline_row['total_row_count'] == '14'
    assert pipeline_row['row_count_details']


def test_pre_training_metrics_sink_jsonl_and_prometheus(tmp_path, monkeypatch):
    durations = {'mh': 1.5, 'flo': 2.5, 'interactive': 1.0, 'ffs': 0.5}
    _patch_pipeline_steps(monkeypatch, durations=durations)
    pipeline = PreTrainingSubPipeline()

    metrics_path = tmp_path / "metrics.jsonl"
    config = SubPipelineConfig(
        metrics_output_path=str(metrics_path),
        metrics_output_format="jsonl",
        metrics_prometheus_enabled=True,
    )

    result = asyncio.run(pipeline.execute_pipeline(config))
    assert result['success'] is True
    assert pipeline._metrics_sink is not None
    assert pipeline._metrics_sink.registry is not None

    with metrics_path.open() as jsonl_file:
        records = [json.loads(line) for line in jsonl_file if line.strip()]

    assert len(records) == 5
    assert all('duration_seconds' in record for record in records)
    assert any(record['record_type'] == 'pipeline' for record in records)

    registry = pipeline._metrics_sink.registry
    duration_value = registry.get_sample_value(
        'pre_training_duration_seconds',
        {'record': 'multi_horizon_profit_labeler'},
    )
    assert duration_value == pytest.approx(durations['mh'])

    completed_value = registry.get_sample_value(
        'pre_training_completed_steps',
        {'record': 'pipeline_total'},
    )
    assert completed_value == pytest.approx(4.0)


def test_pre_training_metrics_sink_uses_locator_default(tmp_path, monkeypatch):
    _patch_pipeline_steps(monkeypatch)
    pipeline = PreTrainingSubPipeline()

    custom_artifacts = tmp_path / "artifacts"
    config = SubPipelineConfig(
        data_locator_config=DataLocatorConfig(
            base_artifacts_dir=str(custom_artifacts),
        ),
        metrics_output_path=None,
        metrics_output_format="csv",
        metrics_prometheus_enabled=False,
    )

    result = asyncio.run(pipeline.execute_pipeline(config))
    assert result['success'] is True

    metrics_path = custom_artifacts / "pre_training_metrics.csv"
    assert metrics_path.exists()
