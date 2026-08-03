import numpy as np
import pandas as pd

from scripts.run_execution_ev_transition_context_overlay import (
    HORIZONS,
    SCORE,
    SIDE,
    _fit_side_overlay,
    build_transition_context_features,
)


def _frame(rows: int = 8) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            SIDE: ["long"] * (rows // 2) + ["short"] * (rows - rows // 2),
            SCORE: np.linspace(-0.01, 0.01, rows),
            "execution_net_ev_12h": np.linspace(-0.02, 0.02, rows),
        }
    )
    for horizon in HORIZONS:
        frame[f"transition_probability_h{horizon}"] = np.linspace(0.1, 0.9, rows)
    return frame


def test_transition_context_has_only_continuous_probabilities_uncertainty_and_interactions():
    features = build_transition_context_features(_frame())
    assert "transition_p_h1" in features
    assert "transition_uncertainty_h12" in features
    assert "direct_ev_x_transition_mean" in features
    assert not any("rank" in column or "decile" in column for column in features.columns)


def test_overlay_has_exact_zero_fallback_when_prior_side_support_is_insufficient():
    evaluation = _frame(8)
    features = build_transition_context_features(evaluation)
    evaluation = pd.concat([evaluation, features], axis=1)
    correction, abstention, report = _fit_side_overlay(
        evaluation.iloc[:0], evaluation, list(features.columns), min_rows=2, random_state=1
    )
    assert np.array_equal(correction, np.zeros(len(evaluation), dtype=np.float32))
    assert np.array_equal(abstention, np.zeros(len(evaluation), dtype=np.float32))
    assert report["long"]["status"] == "zero_fallback"
    assert report["short"]["status"] == "zero_fallback"
