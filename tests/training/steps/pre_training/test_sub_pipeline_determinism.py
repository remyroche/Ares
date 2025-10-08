import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
import sys
import types

import numpy as np
import pandas as pd
import pytest

# Stub heavy component imports before loading the pipeline module.
components_stub = types.ModuleType("src.training.steps.pre_training.components")


class _StubFactory:  # pragma: no cover - placeholder for compatibility
    @staticmethod
    def create_component(*args, **kwargs):
        raise NotImplementedError


class _StubConfig:  # pragma: no cover - placeholder for compatibility
    pass


components_stub.ComponentFactory = _StubFactory
components_stub.ComponentConfig = _StubConfig
sys.modules["src.training.steps.pre_training.components"] = components_stub

from src.training.steps.pre_training import sub_pipeline as sub_pipeline_module
from src.training.steps.pre_training.sub_pipeline import (
    PreTrainingSubPipeline,
    SubPipelineConfig,
    SubPipelineResult,
    SubPipelineStatus,
)


class _DeterministicDateTime(datetime):
    _base = datetime(2024, 1, 1, 0, 0, 0)
    _step = timedelta(seconds=1)
    _counter = 0

    @classmethod
    def now(cls, tz: Optional[Any] = None):
        value = cls._base + cls._step * cls._counter
        cls._counter += 1
        if tz is not None:
            return value.astimezone(tz)
        return value

    @classmethod
    def utcnow(cls):
        return cls.now()

    @classmethod
    def reset(cls):
        cls._counter = 0


def _build_result(
    name: str,
    artifacts: Dict[str, Any],
    duration: float,
    run_metadata: Dict[str, Any],
) -> SubPipelineResult:
    result = SubPipelineResult(
        sub_pipeline_name=name,
        status=SubPipelineStatus.COMPLETED,
        start_time=sub_pipeline_module.datetime.now(),
    )
    result.success = True
    result.end_time = result.start_time + timedelta(seconds=duration)
    result.duration_seconds = duration
    result.artifacts = artifacts
    result.metadata = {"run_metadata": dict(run_metadata)}
    return result


def _patch_deterministic_steps(monkeypatch):
    async def _mh(self, config, run_metadata):
        rng = self._seeded_rngs.numpy
        values = rng.integers(0, 100, size=4)
        df = pd.DataFrame({"label": values}).reset_index(drop=True)
        artifacts = {
            "multi_horizon_labeling_result": {
                "labeled_data": df,
                "method": "deterministic",
            }
        }
        return _build_result("multi_horizon_profit_labeler", artifacts, 5.0, run_metadata)

    async def _flo(self, config, run_metadata):
        rng = self._seeded_rngs.numpy
        scores = rng.normal(size=5)
        artifacts = {
            "feature_lookback_optimization_result": {
                "optimized_features": {"f1": float(scores[0])},
                "optimization_table": pd.DataFrame({"score": scores}),
            }
        }
        return _build_result("feature_lookback_optimization", artifacts, 4.0, run_metadata)

    async def _interactive(self, config, run_metadata):
        rng = self._seeded_rngs.numpy
        features = rng.integers(0, 50, size=3)
        artifacts = {
            "interactive_feature_generation_result": {
                "features": {"g1": int(features[0])},
                "generated_table": pd.DataFrame({"feature": features}),
            }
        }
        return _build_result("interactive_feature_generation", artifacts, 3.0, run_metadata)

    async def _ffs(self, config, run_metadata):
        rng = self._seeded_rngs.numpy
        selections = rng.choice(["a", "b", "c"], size=2, replace=False)
        artifacts = {
            "final_feature_selection_result": {
                "selected_features": selections.tolist(),
                "evaluation": pd.DataFrame({"feature": selections}),
            }
        }
        return _build_result("final_feature_selection", artifacts, 2.0, run_metadata)

    monkeypatch.setattr(PreTrainingSubPipeline, "_execute_multi_horizon_profit_labeler", _mh)
    monkeypatch.setattr(PreTrainingSubPipeline, "_execute_feature_lookback_optimization", _flo)
    monkeypatch.setattr(PreTrainingSubPipeline, "_execute_interactive_feature_generation", _interactive)
    monkeypatch.setattr(PreTrainingSubPipeline, "_execute_final_feature_selection", _ffs)


def _run_pipeline(config: SubPipelineConfig) -> Dict[str, Any]:
    pipeline = PreTrainingSubPipeline()
    return asyncio.run(pipeline.execute_pipeline(config))


def test_sub_pipeline_runs_are_deterministic(monkeypatch):
    _patch_deterministic_steps(monkeypatch)
    monkeypatch.setattr(sub_pipeline_module, "datetime", _DeterministicDateTime)

    config = SubPipelineConfig(custom_params={"random_seed": 777})

    _DeterministicDateTime.reset()
    first = _run_pipeline(config)

    _DeterministicDateTime.reset()
    second = _run_pipeline(config)

    assert first["metrics"] == second["metrics"]
    assert first["results"].keys() == second["results"].keys()

    first_labels = first["results"]["multi_horizon_profit_labeler"]["multi_horizon_labeling_result"]["labeled_data"]
    second_labels = second["results"]["multi_horizon_profit_labeler"]["multi_horizon_labeling_result"]["labeled_data"]
    pd.testing.assert_frame_equal(first_labels, second_labels)

    first_opt = first["results"]["feature_lookback_optimization"]["feature_lookback_optimization_result"]["optimization_table"]
    second_opt = second["results"]["feature_lookback_optimization"]["feature_lookback_optimization_result"]["optimization_table"]
    pd.testing.assert_frame_equal(first_opt, second_opt)

    assert first["metrics"]["random_seed"] == config.custom_params["random_seed"]
