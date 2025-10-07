import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


if "cvxpy" not in sys.modules:
    cvxpy_stub = types.ModuleType("cvxpy")

    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

        def __matmul__(self, other):  # pragma: no cover - unused but kept for safety
            return 0

    class _DummyProblem:
        def __init__(self, *args, **kwargs):
            self.status = "optimal"

        def solve(self, *args, **kwargs):
            return None

    cvxpy_stub.Variable = _Dummy
    cvxpy_stub.Parameter = _Dummy
    cvxpy_stub.Problem = _DummyProblem
    cvxpy_stub.Maximize = lambda *args, **kwargs: None
    cvxpy_stub.Minimize = lambda *args, **kwargs: None
    cvxpy_stub.sum = lambda *args, **kwargs: 0
    cvxpy_stub.CBC = "CBC"
    cvxpy_stub.OPTIMAL = "optimal"
    cvxpy_stub.Constraint = _Dummy

    sys.modules["cvxpy"] = cvxpy_stub

from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.pipeline import (
    CrossTimeframePipeline,
    PipelineConfig,
)
from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.statistical_selection import (
    SelectionResult,
)
from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.evaluation import (
    EvaluationResult,
)


class DummyEvaluation:
    """Stub evaluation component that validates the received feature inputs."""

    def __init__(self):
        self.calls = []

    def evaluate_features(
        self,
        final_features,
        targets,
        regime_segments,
        materialized_htfs=None,
        interactions=None,
    ):
        assert isinstance(final_features, list)
        assert isinstance(materialized_htfs, dict)
        assert isinstance(interactions, list)
        self.calls.append((final_features, targets, regime_segments, materialized_htfs, interactions))
        return EvaluationResult(
            overall_ic=0.02,
            overall_ic_std=0.01,
            overall_ic_ci=(0.0, 0.04),
            regime_results={},
            ablation_results={},
            spa_test_result={},
            walk_forward_results=[],
            metadata={'n_features': len(final_features)},
        )

    def get_evaluation_summary(self, result):
        return {
            'overall_ic': result.overall_ic,
            'overall_ic_std': result.overall_ic_std,
            'overall_ic_ci': result.overall_ic_ci,
            'mean_sharpe': 0.25,
            'max_drawdown': -0.01,
            'feature_count': result.metadata.get('n_features', 0),
            'metadata': result.metadata,
        }


class DummyMonitoring:
    """Stub monitoring system that asserts input compatibility."""

    def __init__(self):
        self.calls = []

    def setup_monitoring(self, final_features, evaluation_summary, regime_segments):
        assert isinstance(final_features, list)
        assert evaluation_summary is None or isinstance(evaluation_summary, dict)
        self.calls.append((final_features, evaluation_summary, regime_segments))

    def get_penalty_parameters(self):
        return {}


class LowICEvaluation:
    """Evaluation stub that produces weak IC to trigger penalty increases."""

    def __init__(self):
        self.calls = []

    def evaluate_features(self, final_features, targets, regime_segments, feature_source=None):
        assert isinstance(final_features, list)
        assert isinstance(feature_source, pd.DataFrame)
        self.calls.append((final_features, targets, regime_segments, feature_source))
        return EvaluationResult(
            overall_ic=0.01,
            overall_ic_std=0.015,
            overall_ic_ci=(-0.01, 0.03),
            regime_results={},
            ablation_results={},
            spa_test_result={},
            walk_forward_results=[],
            metadata={'n_features': len(final_features), 'volatility_level': 0.4},
        )

    def get_evaluation_summary(self, result):
        return {
            'overall_ic': result.overall_ic,
            'overall_ic_std': result.overall_ic_std,
            'overall_ic_ci': result.overall_ic_ci,
            'mean_sharpe': 0.1,
            'max_drawdown': -0.05,
            'feature_count': result.metadata.get('n_features', 0),
            'metadata': {'volatility_level': result.metadata.get('volatility_level', 0.4)},
        }


def _build_dummy_series(index):
    values = np.linspace(0.0, 1.0, len(index))
    return pd.Series(values, index=index)


def test_pipeline_passes_feature_list_to_evaluation_and_monitoring():
    config = PipelineConfig()
    pipeline = CrossTimeframePipeline(config)

    # Replace heavy dependencies with lightweight stubs
    pipeline._sessionize_and_align = lambda ohlcv, optional: {"aligned_data": ohlcv}
    pipeline.regime_segmentation.segment_regimes = lambda sessionized, targets: {"segments": []}
    pipeline.phase1_probe.run_probe_stage = lambda sessionized, segments, targets: {"phase1": True}
    pipeline.phase2_optimization.optimize_lookbacks = lambda data, phase1, segments, targets: {"phase2": True}
    pipeline.ehu_rih_assignment.assign_htf_features = lambda phase2, data: {"feature_a": {}}
    pipeline.knapsack_selection.select_features = (
        lambda phase2, assignments, sessionized=None: ["feature_a"]
    )

    def _materialize_htfs(sessionized_data, selected_htfs):
        index = sessionized_data["aligned_data"].index
        feature = SimpleNamespace(feature_series=_build_dummy_series(index))
        return {"feature_a": feature}

    pipeline.htf_materialization.materialize_htfs = _materialize_htfs

    
    def _generate_interactions(materialized_htfs, base_feature_series, targets):
        if materialized_htfs:
            index = next(iter(materialized_htfs.values())).feature_series.index
        else:
            index = targets.index
        interaction = SimpleNamespace(
            name="interaction_feature",
            feature_series=_build_dummy_series(index),
        )
        return [interaction]

    pipeline.interaction_templates.generate_interactions = _generate_interactions

    selection_result = SelectionResult(
        selected_features=["feature_a", "interaction_feature"],
        selection_frequencies={"feature_a": 1.0, "interaction_feature": 0.8},
        p_values={"feature_a": 0.01, "interaction_feature": 0.02},
        fdr_corrected_p_values={"feature_a": 0.02, "interaction_feature": 0.03},
        conditional_ics={"feature_a": 0.1, "interaction_feature": 0.05},
        group_lasso_groups={},
        selection_method="stub",
        metadata={"source": "test"},
    )

    pipeline.statistical_selection.select_final_features = (
        lambda materialized_htfs, interactions, targets: selection_result
    )

    dummy_evaluation = DummyEvaluation()
    dummy_monitoring = DummyMonitoring()
    pipeline.evaluation = dummy_evaluation
    pipeline.monitoring = dummy_monitoring

    # Create simple OHLCV data within the configured session hours
    index = pd.date_range("2023-01-02 09:00:00", periods=32, freq="min")
    ohlcv = pd.DataFrame(
        {
            "open": np.random.rand(len(index)),
            "high": np.random.rand(len(index)),
            "low": np.random.rand(len(index)),
            "close": np.random.rand(len(index)),
            "volume": np.random.randint(100, 200, size=len(index)),
        },
        index=index,
    )
    targets = pd.Series(np.random.randn(len(index)), index=index)

    results = pipeline.run_pipeline(ohlcv_data=ohlcv, targets=targets)

    assert dummy_evaluation.calls, "Evaluation should be executed once"
    eval_features, _, _, materialized_htfs, interactions = dummy_evaluation.calls[0]
    assert eval_features == selection_result.selected_features
    assert set(materialized_htfs.keys()) == {"feature_a"}
    assert interactions and interactions[0].name == "interaction_feature"

    assert dummy_monitoring.calls, "Monitoring should receive pipeline outputs"
    monitoring_call = dummy_monitoring.calls[0]
    assert monitoring_call[0] == selection_result.selected_features
    assert monitoring_call[1]['overall_ic'] == pytest.approx(0.02)

    assert results["selected_feature_list"] == selection_result.selected_features


def test_monitoring_penalties_propagate_to_scoring():
    config = PipelineConfig(adaptive_penalties=True)
    pipeline = CrossTimeframePipeline(config)

    pipeline._sessionize_and_align = lambda ohlcv, optional: {"aligned_data": ohlcv}
    pipeline.regime_segmentation.segment_regimes = lambda sessionized, targets: {"segments": []}
    pipeline.phase1_probe.run_probe_stage = lambda sessionized, segments, targets: {"phase1": True}
    pipeline.phase2_optimization.optimize_lookbacks = lambda data, phase1, segments, targets: {"phase2": True}
    pipeline.ehu_rih_assignment.assign_htf_features = lambda phase2, data: {"feature_b": {}}
    pipeline.knapsack_selection.select_features = lambda phase2, assignments: ["feature_b"]

    def _materialize(sessionized_data, selected_htfs):
        index = sessionized_data["aligned_data"].index
        feature = SimpleNamespace(feature_series=_build_dummy_series(index))
        return {"feature_b": feature}

    pipeline.htf_materialization.materialize_htfs = _materialize

    pipeline.interaction_templates.generate_interactions = (
        lambda materialized, base_features, targets: []
    )

    selection_result = SelectionResult(
        selected_features=["feature_b"],
        selection_frequencies={"feature_b": 1.0},
        p_values={"feature_b": 0.05},
        fdr_corrected_p_values={"feature_b": 0.05},
        conditional_ics={"feature_b": 0.02},
        group_lasso_groups={},
        selection_method="stub",
        metadata={"source": "test"},
    )

    pipeline.statistical_selection.select_final_features = (
        lambda materialized_htfs, interactions, targets: selection_result
    )

    low_ic_evaluation = LowICEvaluation()
    pipeline.evaluation = low_ic_evaluation

    index = pd.date_range("2023-01-03 09:00:00", periods=32, freq="min")
    ohlcv = pd.DataFrame(
        {
            "open": np.random.rand(len(index)),
            "high": np.random.rand(len(index)),
            "low": np.random.rand(len(index)),
            "close": np.random.rand(len(index)),
            "volume": np.random.randint(100, 200, size=len(index)),
        },
        index=index,
    )
    targets = pd.Series(np.random.randn(len(index)), index=index)

    pipeline.run_pipeline(ohlcv_data=ohlcv, targets=targets)

    penalties_from_monitoring = pipeline.monitoring.get_penalty_parameters()
    assert penalties_from_monitoring["lambda_unc"] > 0.10

    scorer_penalties = pipeline.scoring_system.get_current_penalties()
    assert scorer_penalties["lambda_unc"] == pytest.approx(
        penalties_from_monitoring["lambda_unc"]
    )

    assert pipeline.monitoring.system_state.performance_metrics
