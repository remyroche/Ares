from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.meta_residual_overlay import ResidualOverlayState


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-03-01", periods=8, freq="h", tz="UTC"),
            "side_name": ["long"] * 4 + ["short"] * 4,
            "archetype_policy_key": ["a", "a", "b", "b", "a", "a", "b", "b"],
            "meta_resid_arch_expected_hit_surprise": [
                -0.4,
                -0.2,
                -0.3,
                -0.1,
                -0.5,
                -0.2,
                -0.4,
                -0.1,
            ],
            "meta_resid_arch_expected_dirty_positive": [
                0.4,
                0.2,
                0.3,
                0.1,
                0.5,
                0.2,
                0.4,
                0.1,
            ],
        }
    )


def test_overlay_transform_needs_no_outcome_columns() -> None:
    frame = _frame()
    state = ResidualOverlayState(
        hit_alpha=0.2, dirty_lambda=0.025, local_hit_alpha=0.01
    )
    state.fit_normalization(frame)
    score = state.transform(frame, np.full(len(frame), 0.6, dtype=np.float32))
    assert score.shape == (len(frame),)
    assert np.isfinite(score).all()
    assert not np.allclose(score, 0.6)


def test_overlay_transform_rejects_outcomes() -> None:
    frame = _frame()
    state = ResidualOverlayState().fit_normalization(frame)
    with pytest.raises(ValueError, match="received outcomes"):
        state.transform(frame.assign(clean_exec=1), np.full(len(frame), 0.5))


def test_overlay_unknown_group_uses_frozen_fallback() -> None:
    train = _frame()
    state = ResidualOverlayState(local_hit_alpha=0.02).fit_normalization(train)
    oos = train.iloc[:2].copy()
    oos["side_name"] = "unknown"
    oos["archetype_policy_key"] = "new"
    score = state.transform(oos, np.full(len(oos), 0.5))
    assert np.isfinite(score).all()


def test_overlay_joblib_roundtrip_is_exact(tmp_path) -> None:
    frame = _frame()
    state = ResidualOverlayState(
        hit_alpha=0.2, dirty_lambda=0.025, local_hit_alpha=0.01
    )
    state.fit_normalization(frame)
    expected = state.transform(
        frame, np.linspace(0.4, 0.7, len(frame), dtype=np.float32)
    )
    path = tmp_path / "state.joblib"
    joblib.dump(state, path)
    restored = joblib.load(path)
    actual = restored.transform(
        frame, np.linspace(0.4, 0.7, len(frame), dtype=np.float32)
    )
    np.testing.assert_array_equal(actual, expected)
