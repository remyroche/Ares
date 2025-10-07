import logging
from typing import List

import numpy as np
import pandas as pd
import pytest

from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.knapsack_selection import (
    FeatureCandidate,
    KnapsackSelection,
)
from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.pipeline import (
    PipelineConfig,
)


def _build_candidate(feature_id: str, feature_name: str, series: pd.Series) -> FeatureCandidate:
    return FeatureCandidate(
        feature_id=feature_id,
        feature_name=feature_name,
        family="trend_level_vol",
        utility=1.0,
        cost=1.0,
        lookback=60,
        update_style="ehu",
        metadata={"feature_series": series},
    )


def test_knapsack_selection_applies_correlation_constraint(caplog: pytest.LogCaptureFixture) -> None:
    config = PipelineConfig()
    config.max_correlation = 0.2

    selection = KnapsackSelection(config)

    # Force deterministic solver path for testing
    selection.solver.solve_knapsack = lambda features, matrix: selection.solver._solve_greedy(
        features, matrix
    )

    index = pd.date_range("2024-01-01", periods=120, freq="5min")
    base_series = pd.Series(np.linspace(0.0, 1.0, len(index)), index=index)
    correlated_series = base_series + 1e-6

    candidates: List[FeatureCandidate] = [
        _build_candidate("feat_a_60", "p/price_ema10_pct", base_series),
        _build_candidate("feat_b_60", "p/price_ema20_pct", correlated_series),
    ]

    selection._create_feature_candidates = lambda phase2, assignments: candidates

    with caplog.at_level(logging.INFO):
        result = selection.select_features({}, [], sessionized_data=None)

    assert len(result.selected_features) == 1
    assert not result.correlation_matrix.empty

    corr_value = result.correlation_matrix.loc[
        candidates[0].feature_name, candidates[1].feature_name
    ]
    assert abs(corr_value) >= config.max_correlation - 1e-6

    logged = [record.message.lower() for record in caplog.records]
    assert any("correlation" in message for message in logged)
