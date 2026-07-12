from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.meta_residual_shock_overlay import (
    MARKET_SHOCK_COMPONENTS,
    ResidualShockOverlayState,
)


def _state() -> ResidualShockOverlayState:
    return ResidualShockOverlayState(
        references={
            name: np.asarray([0.0, 1.0, 2.0], dtype=np.float32)
            for name in MARKET_SHOCK_COMPONENTS
        },
        archetype_multipliers={"long||a": 1.0, "long||*": 0.5},
        train_end="2026-06-30",
    )


def _frame() -> pd.DataFrame:
    data = {name: [2.0, 1.0] for name in MARKET_SHOCK_COMPONENTS}
    for name, direction in MARKET_SHOCK_COMPONENTS.items():
        if direction < 0:
            data[name] = [-2.0, -1.0]
    data["side_name"] = ["long", "long"]
    data["archetype_policy_key"] = ["a", "missing"]
    return pd.DataFrame(data)


def test_shock_overlay_is_outcome_free_and_side_aware() -> None:
    state = _state()
    frame = _frame()
    raw = state.transform_raw(frame)
    local = state.transform(frame)
    assert raw[0] > raw[1]
    assert local[0] == pytest.approx(raw[0])
    assert local[1] == pytest.approx(0.5 * raw[1])
    adjusted, _, _ = state.adjust_scores(
        frame,
        np.asarray([0.8, 0.8], dtype=np.float32),
        {"long": {"variant": "raw", "threshold": 0.5, "alpha": 0.1}},
    )
    assert adjusted[0] < adjusted[1]


def test_shock_overlay_rejects_realized_outcomes() -> None:
    frame = _frame()
    frame["clean_exec"] = 1.0
    with pytest.raises(ValueError, match="received outcomes"):
        _state().transform(frame)
