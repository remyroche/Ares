from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.target_design import (
    TargetDesignSpec,
    build_target,
    build_training_weights,
    fit_target_reference,
    fit_training_weight_reference,
    resolve_net_return,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-01-01 00:00", "2026-01-01 00:00", "2026-01-01 00:15"], utc=True
            ),
            "__y_ret__": [-0.01, 0.00, 0.02],
            "__first_touch_capture_net__": [-0.01, 0.00, 0.02],
            "__first_touch_round_trip_cost__": [0.01, 0.01, 0.01],
            "__first_touch_target_soft__": [0.1, 0.5, 0.9],
            "__barrier_pct__": [0.01, 0.02, 0.04],
        }
    )


def test_already_net_target_never_subtracts_cost_twice() -> None:
    frame = _frame()
    net, meta = resolve_net_return(frame, TargetDesignSpec(name="raw", target_kind="raw_net"))
    assert np.allclose(net, frame["__y_ret__"])
    assert meta["additional_cost_subtracted"] == 0.0


def test_raw_and_vol_targets_keep_net_return_ordering() -> None:
    frame = _frame()
    raw, _ = build_target(frame, TargetDesignSpec(name="raw", target_kind="raw_net"))
    vol, _ = build_target(frame, TargetDesignSpec(name="vol", target_kind="vol_norm_net"))
    assert np.all(np.diff(raw) > 0.0)
    assert raw.shape == vol.shape == (3,)
    assert np.all((vol >= 0.0) & (vol <= 1.0))


def test_dual_target_is_convex_combination() -> None:
    frame = _frame()
    raw, _ = build_target(frame, TargetDesignSpec(name="raw", target_kind="raw_net"))
    vol, _ = build_target(frame, TargetDesignSpec(name="vol", target_kind="vol_norm_net"))
    dual, _ = build_target(
        frame,
        TargetDesignSpec(name="dual", target_kind="dual_raw_vol", dual_raw_weight=0.25),
    )
    assert np.allclose(dual, 0.25 * raw + 0.75 * vol)


def test_timestamp_balanced_volatility_weights_are_normalized_and_train_only() -> None:
    frame = _frame()
    reference = fit_training_weight_reference(frame)
    w7, _ = build_training_weights(
        frame,
        TargetDesignSpec(name="w7", target_kind="raw_net", weight_mode="timestamp_balanced"),
        reference,
    )
    weighted, meta = build_training_weights(
        frame,
        TargetDesignSpec(
            name="weighted", target_kind="raw_net", weight_mode="timestamp_balanced_vol_damped"
        ),
        reference,
    )
    assert np.allclose(w7, np.array([0.75, 0.75, 1.50], dtype=np.float32))
    assert np.isclose(weighted.mean(), 1.0)
    assert meta["volatility_train_median"] == 0.02
    assert not np.allclose(weighted, w7)


def test_side_relative_target_requires_train_reference_and_is_side_aware() -> None:
    train = _frame()
    train["side_name"] = ["long", "long", "short"]
    spec = TargetDesignSpec(name="side", target_kind="side_robust_net")
    reference = fit_target_reference(train, spec)
    target, meta = build_target(train, spec, reference=reference)
    assert target.dtype == np.float32
    assert meta["reference_schema"] == "side_economic_target_reference_v1"
    with pytest.raises(ValueError, match="train-fitted"):
        build_target(train, spec)


def test_side_ecdf_uses_frozen_train_distribution() -> None:
    train = _frame()
    train["side_name"] = ["long", "long", "short"]
    spec = TargetDesignSpec(name="side_ecdf", target_kind="side_net_ecdf", side_ecdf_knots=9)
    reference = fit_target_reference(train, spec)
    oos = train.copy()
    oos["__first_touch_capture_net__"] = 0.50
    score, _ = build_target(oos, spec, reference=reference)
    assert np.allclose(score, 1.0)
