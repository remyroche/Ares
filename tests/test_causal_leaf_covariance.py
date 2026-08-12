from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.causal_leaf_covariance import (
    CausalLeafCovarianceConfig,
    CausalLeafCovarianceError,
    build_causal_leaf_covariance_state,
)


def _frame(*, second_block_first: tuple[float, float] = (99.0, -99.0), future: tuple[float, float] = (7.0, -7.0)) -> pd.DataFrame:
    return pd.DataFrame({
        "source_utc": pd.date_range("2025-01-01", periods=7, freq="h", tz="UTC"),
        "evaluation_block": ["history"] * 5 + ["evaluation"] * 2,
        "family": ["price"] * 7,
        "side_name": ["long"] * 7,
        "head_name": ["p_clear"] * 7,
        "x": [0.0, 1.0, -1.0, 2.0, -2.0, second_block_first[0], future[0]],
        "y": [0.0, -1.0, 1.0, -2.0, 2.0, second_block_first[1], future[1]],
    })


def test_current_and_future_rows_do_not_enter_their_own_frozen_reference_or_state() -> None:
    first = build_causal_leaf_covariance_state(_frame(), ["x", "y"]).frame
    changed = build_causal_leaf_covariance_state(
        _frame(second_block_first=(-10_000.0, 10_000.0), future=(50_000.0, -50_000.0)), ["x", "y"]
    ).frame
    diagnostic = [name for name in first if name.startswith("leaf_covariance__")]
    # The first evaluation row is computed before either its value or a future
    # row can update state.  It is a zero distance from its frozen snapshot.
    np.testing.assert_allclose(
        first.loc[5, diagnostic].to_numpy(dtype=np.float32),
        changed.loc[5, diagnostic].to_numpy(dtype=np.float32),
        equal_nan=True,
    )
    assert first.loc[5, "leaf_covariance__24h__weighted_covariance_distance"] == pytest.approx(0.0)


def test_contemporaneous_rows_are_scored_before_any_same_timestamp_update() -> None:
    """Cross-sectional rows at one decision time must have identical history.

    The second row deliberately has an extreme value.  If the first were
    allowed to update before the second is scored, changing it would alter the
    second row's state and turn the cross-section into a look-ahead channel.
    """
    base = _frame().iloc[:5].copy()
    same_time = pd.DataFrame({
        "source_utc": [pd.Timestamp("2025-01-01 05:00:00", tz="UTC")] * 2,
        "evaluation_block": ["evaluation"] * 2,
        "family": ["price"] * 2,
        "side_name": ["long"] * 2,
        "head_name": ["p_clear"] * 2,
        "x": [2.0, -2.0],
        "y": [-2.0, 2.0],
    })
    first = build_causal_leaf_covariance_state(
        pd.concat((base, same_time), ignore_index=True), ["x", "y"]
    ).frame
    changed = same_time.copy()
    changed.loc[0, ["x", "y"]] = [10_000.0, -10_000.0]
    second = build_causal_leaf_covariance_state(
        pd.concat((base, changed), ignore_index=True), ["x", "y"]
    ).frame
    diagnostics = [name for name in first if name.startswith("leaf_covariance__")]
    np.testing.assert_allclose(
        first.loc[6, diagnostics].to_numpy(dtype=np.float32),
        second.loc[6, diagnostics].to_numpy(dtype=np.float32),
        equal_nan=True,
    )


def test_support_aware_hierarchy_shrinks_thin_family_to_side_head_and_global() -> None:
    timestamps = pd.date_range("2025-01-01", periods=34, freq="h", tz="UTC")
    history = pd.DataFrame({
        "source_utc": timestamps,
        "evaluation_block": ["history"] * 33 + ["evaluation"],
        "family": ["thin"] * 2 + ["deep"] * 31 + ["thin"],
        "side_name": ["long"] * 34,
        "head_name": ["p_clear"] * 34,
        "x": np.linspace(-2.0, 2.0, 34),
        "y": np.cos(np.linspace(0.0, 3.0, 34)),
    })
    output = build_causal_leaf_covariance_state(
        history, ["x", "y"], config=CausalLeafCovarianceConfig(shrinkage_support=10.0)
    ).frame.iloc[-1]
    assert output["leaf_covariance__reference_support"] == pytest.approx(2.0)
    assert output["leaf_covariance__side_head_weight"] > output["leaf_covariance__family_weight"]
    assert output["leaf_covariance__global_weight"] > output["leaf_covariance__family_weight"]
    assert output[[
        "leaf_covariance__family_weight",
        "leaf_covariance__side_head_weight",
        "leaf_covariance__global_weight",
    ]].sum() == pytest.approx(1.0)


def test_rejects_raw_leaf_ids_non_utc_order_and_unbounded_fields() -> None:
    frame = _frame()
    raw_leaf = frame.assign(raw_leaf_id="not-allowed")
    with pytest.raises(CausalLeafCovarianceError, match="raw leaf"):
        build_causal_leaf_covariance_state(raw_leaf, ["x", "y"])
    unordered = frame.iloc[[1, 0, 2, 3, 4, 5, 6]].reset_index(drop=True)
    with pytest.raises(CausalLeafCovarianceError, match="UTC order"):
        build_causal_leaf_covariance_state(unordered, ["x", "y"])
    with pytest.raises(CausalLeafCovarianceError, match="too many"):
        build_causal_leaf_covariance_state(frame.assign(**{f"f{i}": float(i) for i in range(16)}), [f"f{i}" for i in range(16)])
