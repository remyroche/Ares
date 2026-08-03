from __future__ import annotations

import pandas as pd

from scripts.run_canonical_base_conversion_prediction_attribution import (
    _reference_edges,
    _stable_top,
    _two_state_shapley,
)


def test_stable_top_is_global_score_then_candidate_identity() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": ["b", "a", "c", "d"],
            "score_raw": [1.0, 1.0, 0.5, 0.0],
        }
    )
    selected = _stable_top(frame, 0.5)
    assert selected["candidate_id"].tolist() == ["a", "b"]


def test_reference_edges_are_fixed_and_cover_future_extremes() -> None:
    edges = _reference_edges(pd.Series(range(100)), bins=5)
    assert len(edges) == 6
    assert edges[0] == float("-inf")
    assert edges[-1] == float("inf")


def test_shapley_reconciles_composition_and_conversion_change() -> None:
    p_a = pd.Series([0.6, 0.4]).to_numpy(float)
    p_b = pd.Series([0.4, 0.6]).to_numpy(float)
    v_a = pd.Series([1.0, -1.0]).to_numpy(float)
    v_b = pd.Series([2.0, 0.0]).to_numpy(float)
    composition, conversion = _two_state_shapley(p_a, v_a, p_b, v_b)
    actual = float(p_b @ v_b - p_a @ v_a)
    assert abs(actual - composition - conversion) < 1e-12
