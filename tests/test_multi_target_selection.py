"""Tests for corrected target selection in the multi-target scheme."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest


def _load_multi_target_scheme_module():
    """Load the multi_target_scheme module without triggering package side-effects."""

    # Stub problematic optional modules that trigger expansive import graphs.
    ml_common_key = "src.utils.ml_common"
    if ml_common_key not in sys.modules:
        ml_common_pkg = types.ModuleType(ml_common_key)
        ml_common_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules[ml_common_key] = ml_common_pkg

    optimization_key = "src.utils.ml_common.optimization"
    if optimization_key not in sys.modules:
        optimization_pkg = types.ModuleType(optimization_key)
        optimization_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules[optimization_key] = optimization_pkg

    bayesian_key = "src.utils.ml_common.optimization.bayesian_tpe_optimizer"
    if bayesian_key not in sys.modules:
        bayesian_module = types.ModuleType(bayesian_key)

        class _DummyBayesianTPEOptimizer:  # pragma: no cover - simple stub
            def __init__(self, *args, **kwargs):
                pass

            def optimize(self, *args, **kwargs):
                return None

        bayesian_module.BayesianTPEOptimizer = _DummyBayesianTPEOptimizer
        sys.modules[bayesian_key] = bayesian_module

    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "training"
        / "steps"
        / "pre_training"
        / "profit_labeling"
        / "multi_target_scheme.py"
    )

    spec = importlib.util.spec_from_file_location(
        "multi_target_scheme", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[assignment]
    return module


multi_target_scheme = _load_multi_target_scheme_module()
MultiTargetScheme = multi_target_scheme.MultiTargetScheme
MultiTargetConfig = multi_target_scheme.MultiTargetConfig
TargetBand = multi_target_scheme.TargetBand


@pytest.fixture
def multi_target_config() -> MultiTargetConfig:
    config = MultiTargetConfig()
    config.max_targets_per_band = 5
    config.max_targets_total = 5
    config.min_targets_total = 1
    config.min_lqs_score = 0.3
    config.max_correlation_threshold = 0.99
    return config


def _build_candidate_labels() -> Dict[str, pd.DataFrame]:
    index = pd.RangeIndex(0, 200)

    high_quality = pd.Series([1] * 100 + [-1] * 100, index=index)
    medium_quality = pd.Series(
        (np.tile([1, -1], 100))[:200], index=index
    )
    low_quality = pd.Series([1] * 160 + [-1] * 40, index=index)

    return {
        "target_high": pd.DataFrame({"labels": high_quality}),
        "target_medium": pd.DataFrame({"labels": medium_quality}),
        "target_low": pd.DataFrame({"labels": low_quality}),
    }


def _build_candidate_targets() -> Dict[str, Dict[str, Any]]:
    base_targets = [
        {
            "target_name": "target_high",
            "band": TargetBand.SMALL,
            "k_up": 0.5,
            "k_down": 0.5,
            "parameters": {},
        },
        {
            "target_name": "target_medium",
            "band": TargetBand.MEDIUM,
            "k_up": 0.8,
            "k_down": 0.8,
            "parameters": {},
        },
        {
            "target_name": "target_low",
            "band": TargetBand.HIGH,
            "k_up": 1.2,
            "k_down": 1.2,
            "parameters": {},
        },
    ]

    return {target["target_name"]: target for target in base_targets}


def test_select_optimal_targets_applies_multiple_testing_correction(multi_target_config):
    scheme = MultiTargetScheme(multi_target_config)
    candidate_labels = _build_candidate_labels()
    candidate_targets = list(_build_candidate_targets().values())

    selected_targets, metadata = scheme._select_optimal_targets(
        candidate_labels, candidate_targets
    )

    # Ensure the correction tracked the evaluated hypotheses
    assert metadata["total_candidates_evaluated"] == 3
    assert metadata["quality_thresholds"]["bonferroni"] >= multi_target_config.min_lqs_score

    bh_info = metadata["quality_thresholds"]["benjamini_hochberg"]
    assert bh_info["critical_thresholds"]
    assert bh_info["final_threshold"] >= multi_target_config.min_lqs_score

    # The low-quality candidate should be filtered out by the correction
    assert "target_low" not in selected_targets
    assert set(selected_targets.keys()).issubset({"target_high", "target_medium"})


def test_generate_targets_surfaces_selection_metadata(monkeypatch, multi_target_config):
    scheme = MultiTargetScheme(multi_target_config)

    candidate_labels = _build_candidate_labels()
    candidate_targets_map = _build_candidate_targets()
    candidate_targets = list(candidate_targets_map.values())

    bars_index = pd.RangeIndex(0, 20)
    bars = pd.DataFrame(
        {
            "open": np.linspace(100, 101, 20),
            "high": np.linspace(101, 102, 20),
            "low": np.linspace(99, 100, 20),
            "close": np.linspace(100, 101, 20),
        },
        index=bars_index,
    )
    volatility = pd.Series(np.linspace(0.5, 1.0, 20), index=bars_index)
    eligibility = pd.Series(True, index=bars_index)

    def fake_generate_candidate_targets(*_args, **_kwargs):
        return [dict(target) for target in candidate_targets]

    def fake_calculate_fpt_horizons(_candidates, *_args, **_kwargs):
        return {
            name: {"horizon": 5, "horizon_context": {"horizon": 5}}
            for name in candidate_labels
        }

    def fake_generate_candidate_labels(*_args, **_kwargs):
        return candidate_labels

    def fake_generate_final_labels(selected, *_args, **_kwargs):
        labels_df = pd.DataFrame(index=bars_index)
        confidence_df = pd.DataFrame(index=bars_index)
        eligibility_df = pd.DataFrame(index=bars_index)

        for idx, name in enumerate(selected.keys()):
            pattern = np.tile([1, -1], 10)[: len(bars_index)]
            labels_df[name] = pattern if idx % 2 == 0 else -pattern
            confidence_df[name] = 0.6
            eligibility_df[name] = True

        return {
            "labels": labels_df,
            "confidence_scores": confidence_df,
            "eligibility_masks": eligibility_df,
        }

    monkeypatch.setattr(scheme, "_generate_candidate_targets", fake_generate_candidate_targets)
    monkeypatch.setattr(scheme, "_calculate_fpt_horizons", fake_calculate_fpt_horizons)
    monkeypatch.setattr(scheme, "_generate_candidate_labels", fake_generate_candidate_labels)
    monkeypatch.setattr(scheme, "_generate_final_labels", fake_generate_final_labels)
    monkeypatch.setattr(scheme, "_resolve_label_conflicts", lambda df: df)
    monkeypatch.setattr(scheme, "_apply_label_smoothing", lambda labels, _conf: labels)

    result = scheme.generate_targets(bars, volatility, eligibility)

    assert result.selection_metadata
    assert result.selection_metadata["total_candidates_evaluated"] == 3
    assert result.selection_metadata["quality_thresholds"]["final"] >= multi_target_config.min_lqs_score
    assert result.selected_targets
