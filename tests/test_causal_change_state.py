from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.causal_change_state import (
    build_causal_change_state,
    build_streaming_long_change_state,
)


def _as_dict(matrix: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    return {name: matrix[:, index] for index, name in enumerate(names)}


def test_change_state_preserves_signed_liquidation_mechanism() -> None:
    sequence = np.zeros((2, 16, 4), dtype=np.float32)
    # Row zero enters a price/OI contraction with expanding volume and vol.
    sequence[0, -4:, 0] = -2.0
    sequence[0, -4:, 1] = -1.5
    sequence[0, -4:, 2] = 2.0
    sequence[0, -4:, 3] = 1.0
    # Row one is the corresponding short-covering direction.
    sequence[1, -4:, 0] = 2.0
    sequence[1, -4:, 1] = -1.5
    sequence[1, -4:, 2] = 2.0
    sequence[1, -4:, 3] = 1.0
    matrix, names = build_causal_change_state(
        sequence,
        ["mkt_ret_1h", "mkt_oi_chg_1h", "mkt_volume_z_24h", "mkt_rv_1h"],
    )
    values = _as_dict(matrix, names)
    assert values["cp_price_medium__signed_shift"][0] < 0.0
    assert values["cp_leverage_medium__signed_shift"][0] < 0.0
    assert values["cp_mechanism__long_liquidation"][0] > values["cp_mechanism__short_covering"][0]
    assert values["cp_mechanism__short_covering"][1] > values["cp_mechanism__long_liquidation"][1]


def test_change_state_is_invariant_to_data_after_decision_sequence() -> None:
    rng = np.random.default_rng(17)
    history = rng.normal(size=(24, 3)).astype(np.float32)
    decision_sequence = history[:16][None, :, :]
    first, names = build_causal_change_state(
        decision_sequence, ["mkt_ret_1h", "mkt_oi_chg_1h", "market_breadth_1h"]
    )
    history[16:] = 1_000.0
    second, second_names = build_causal_change_state(
        history[:16][None, :, :], ["mkt_ret_1h", "mkt_oi_chg_1h", "market_breadth_1h"]
    )
    assert names == second_names
    np.testing.assert_allclose(first, second, equal_nan=True)


def test_change_state_contract_is_continuous_and_outcome_free() -> None:
    sequence = np.ones((5, 16, 2), dtype=np.float32)
    matrix, names = build_causal_change_state(
        sequence, ["mkt_funding_mean", "market_pc1_variance_share_12h"]
    )
    assert matrix.dtype == np.float32
    assert matrix.shape == (5, len(names))
    assert "cp_global__run_length_entropy" in names
    assert "cp_mechanism__unknown_transition" in names
    assert not any(token in name for name in names for token in ("label", "target", "realized", "outcome"))


def test_streaming_long_state_detects_signed_multi_day_shift() -> None:
    index = pd.date_range("2025-01-01", periods=1_200, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "mkt_ret_4h": np.r_[np.zeros(1_032), np.full(168, -2.0)],
        "mkt_oi_chg_4h_rz": np.r_[np.zeros(1_032), np.full(168, -1.5)],
        "market_breadth_1h": np.r_[np.zeros(1_032), np.full(168, -1.0)],
        "mkt_volume_z_24h": np.r_[np.zeros(1_032), np.full(168, 2.0)],
    }, index=index, dtype=np.float32)
    state = build_streaming_long_change_state(frame, list(frame.columns))
    assert state.iloc[-1]["cp_long_price_168h__signed_shift"] < 0.0
    assert state.iloc[-1]["cp_long_leverage_168h__signed_shift"] < 0.0
    assert state.iloc[-1]["cp_long_volume_liquidity_168h__signed_shift"] > 0.0
    assert state.iloc[-1]["cp_long_global_168h__mean_change_probability"] > 0.5


def test_streaming_long_state_is_causal_and_compact() -> None:
    rng = np.random.default_rng(91)
    index = pd.date_range("2025-01-01", periods=1_300, freq="h", tz="UTC")
    values = pd.DataFrame(
        rng.normal(size=(len(index), 4)).astype(np.float32),
        index=index,
        columns=["mkt_ret_1h", "mkt_oi_chg_1h", "mkt_funding_mean", "market_breadth_1h"],
    )
    first = build_streaming_long_change_state(values.iloc[:1_200], list(values.columns))
    changed = values.copy()
    changed.iloc[1_200:] = 1_000.0
    second = build_streaming_long_change_state(changed, list(changed.columns)).iloc[:1_200]
    pd.testing.assert_frame_equal(first, second)
    assert len(first.columns) < 250
    assert first.dtypes.eq(np.dtype("float32")).all()
    assert not any(token in name for name in first for token in ("label", "target", "realized", "outcome"))


def test_long_mechanism_bottleneck_separates_liquidation_and_covering() -> None:
    index = pd.date_range("2025-01-01", periods=1_200, freq="h", tz="UTC")
    base = np.zeros(1_200, dtype=np.float32)
    down = base.copy(); down[-168:] = -2.0
    up = base.copy(); up[-168:] = 2.0
    oi_down = base.copy(); oi_down[-168:] = -1.5
    volume = base.copy(); volume[-168:] = 2.0
    liquidating = build_streaming_long_change_state(pd.DataFrame({
        "mkt_ret_4h": down,
        "mkt_oi_chg_4h_rz": oi_down,
        "mkt_volume_z_24h": volume,
        "mkt_rv_4h": volume,
    }, index=index), ["mkt_ret_4h", "mkt_oi_chg_4h_rz", "mkt_volume_z_24h", "mkt_rv_4h"])
    covering = build_streaming_long_change_state(pd.DataFrame({
        "mkt_ret_4h": up,
        "mkt_oi_chg_4h_rz": oi_down,
        "mkt_volume_z_24h": volume,
        "mkt_rv_4h": volume,
    }, index=index), ["mkt_ret_4h", "mkt_oi_chg_4h_rz", "mkt_volume_z_24h", "mkt_rv_4h"])
    assert liquidating.iloc[-1]["cp_long_mechanism_long_liquidation_168h"] > 0.5
    assert liquidating.iloc[-1]["cp_long_mechanism_short_covering_168h"] < 0.1
    assert covering.iloc[-1]["cp_long_mechanism_short_covering_168h"] > 0.5
    mechanism_columns = [name for name in liquidating if name.startswith("cp_long_mechanism_")]
    assert len(mechanism_columns) == 42
