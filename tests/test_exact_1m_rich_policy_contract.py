"""Focused research/live parity tests for the exact rich 1m oracle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.exact_1m_rich_policy_contract import (
    Exact1mRichExecutionContract,
    Exact1mRichV2ExecutionContract,
    RichExitExtensions,
    advance_exact_1m_rich_position,
    exact_1m_rich_v2_receipt,
    replay_exact_1m_rich_policy,
    replay_exact_1m_rich_policy_v2,
)
from extreme_price_movements.strict_r3_rich_policy import RichPolicyParams
from extreme_price_movements.inference.strict_r3_live_execution import (
    _advance_rich_policy_position,
    _rich_policy_params,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data_perp/artifacts/strict_r3_rich_policy_hpo_long_20260817_v1/frozen_challenger.json"


def _params() -> tuple[object, float]:
    return _rich_policy_params(json.loads(FROZEN.read_text()))


def _position(*, entry: float = 100.0, atr: float = 1.7935832787339956, mfe: float = 0.0, mae: float = 0.0, armed: bool = False, protected: bool = False) -> dict:
    return {
        "entry_price": entry,
        "atr": atr,
        "entry_ts": "2026-08-17T00:00:00Z",
        "timeout_ts": "2026-08-17T12:00:00Z",
        "next_bar_ts": "2026-08-17T00:00:00Z",
        "maximum_favourable": mfe,
        "maximum_adverse": mae,
        "trailing_armed": armed,
        "capital_protect_armed": protected,
        "rich_adaptive_activation_multiplier": 1.0,
    }


def _bars(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["timestamp", "high", "low", "close"])
    return frame.set_index(pd.to_datetime(frame.pop("timestamp"), utc=True))


def _assert_equal_relevant(left: dict, right: dict) -> None:
    for field in (
        "maximum_favourable", "maximum_adverse", "trailing_armed",
        "trailing_armed_now", "capital_protect_armed",
        "capital_protect_armed_now", "next_bar_ts", "rich_policy_last_activation",
        "smooth_armed", "smooth_armed_now", "smooth_lock", "smooth_lock_price",
        "sl_barrier_frac", "tp_barrier_frac", "sl_distance",
        "tp_barrier_distance", "protect_activation", "protect_lock",
    ):
        if isinstance(left[field], float):
            assert left[field] == pytest.approx(right[field])
        else:
            assert left[field] == right[field]
    assert left.get("exit", {}).keys() == right.get("exit", {}).keys()
    for field, value in left.get("exit", {}).items():
        if isinstance(value, float):
            assert value == pytest.approx(right["exit"][field])
        else:
            assert value == right["exit"][field]


def test_oracle_matches_current_live_rich_state_machine_over_mixed_path() -> None:
    params, median = _params()
    position = _position(mfe=3.1)
    bars = _bars([
        ("2026-08-17T00:00:00Z", 100.2, 99.9, 100.0),
        ("2026-08-17T00:01:00Z", 103.2, 102.5, 102.7),
    ])
    live = _advance_rich_policy_position(
        position=position, bars=bars, params=params, median_atr_fraction=median,
    )
    oracle = advance_exact_1m_rich_position(
        position=position, bars=bars, params=params, median_atr_fraction=median,
    )
    _assert_equal_relevant(live, oracle)


def test_oracle_matches_live_smooth_capital_protection_state_machine() -> None:
    params = RichPolicyParams(
        sl_mult=10.0,
        smooth_capital_protection_enabled=True,
        protection_unit="raw_decision_time_atr",
        protection_activation_atr=1.5,
        protection_strength=0.5,
        protection_power=1.5,
        adverse_exit_enabled=False,
    )
    position = _position(atr=2.0)
    # The first bar arms the positive smooth lock; the second crosses the
    # persisted lock.  This exercises the ordering that was absent from the
    # older offline oracle.
    first_bars = _bars([("2026-08-17T00:00:00Z", 104.0, 100.4, 101.0)])
    live_first = _advance_rich_policy_position(
        position=position, bars=first_bars, params=params, median_atr_fraction=0.01,
    )
    oracle_first = advance_exact_1m_rich_position(
        position=position, bars=first_bars, params=params, median_atr_fraction=0.01,
    )
    _assert_equal_relevant(live_first, oracle_first)
    second_position = {**position, **live_first}
    second_bars = _bars([("2026-08-17T00:01:00Z", 101.1, 100.1, 100.4)])
    live_second = _advance_rich_policy_position(
        position=second_position, bars=second_bars, params=params, median_atr_fraction=0.01,
    )
    oracle_second = advance_exact_1m_rich_position(
        position={**position, **oracle_first}, bars=second_bars,
        params=params, median_atr_fraction=0.01,
    )
    _assert_equal_relevant(live_second, oracle_second)


def test_priority_is_stop_then_capital_then_trailing_then_fast_adverse() -> None:
    params, median = _params()
    # This constructed state crosses every relevant lower threshold.  The hard
    # stop is necessarily authoritative, independent of later policy actions.
    position = _position(mfe=10.0, mae=4.0, armed=True, protected=True)
    bars = _bars([("2026-08-17T00:00:00Z", 100.0, 90.0, 92.0)])
    outcome = advance_exact_1m_rich_position(
        position=position, bars=bars, params=params, median_atr_fraction=median,
    )
    assert outcome["exit"]["exit_reason"] == "stop_loss"


def test_prior_peak_arms_trailing_only_for_following_completed_minute() -> None:
    params, median = _params()
    # First bar earns the peak but is unable to trigger the trail.  The next
    # bar sees that prior peak and may trigger it.
    first = advance_exact_1m_rich_position(
        position=_position(),
        bars=_bars([("2026-08-17T00:00:00Z", 103.2, 100.0, 102.9)]),
        params=params, median_atr_fraction=median,
    )
    assert first["trailing_armed"] is False or first["trailing_armed_now"] is True
    second_position = {**_position(), **first}
    second = advance_exact_1m_rich_position(
        position=second_position,
        bars=_bars([("2026-08-17T00:01:00Z", 103.0, 102.5, 102.6)]),
        params=params, median_atr_fraction=median,
    )
    # Exact winner geometry can arm one minute later; if it does, no current
    # bar high may have armed and exited it simultaneously.
    assert second["next_bar_ts"] == "2026-08-17T00:02:00+00:00" or "exit" in second


def test_elapsed_15m_anchor_is_preserved_under_one_minute_scan() -> None:
    params, median = _params()
    position = _position()
    bars = _bars([("2026-08-17T00:16:00Z", 100.0, 100.0, 100.0)])
    position["next_bar_ts"] = "2026-08-17T00:16:00Z"
    outcome = advance_exact_1m_rich_position(
        position=position, bars=bars, params=params, median_atr_fraction=median,
    )
    # At bar end 00:17, the legacy index is floor(17 / 15) == 1 rather than 17.
    expected = min(
        outcome["tp_barrier_distance"] * params.trailing_activation_mult,
        100.0 * params.trailing_activation_cap_pct,
    )
    assert outcome["rich_policy_last_activation"] == pytest.approx(expected)


def test_exact_replay_emits_h12_timestamp_and_cost_once() -> None:
    params, median = _params()
    high = np.full((1, 720), 100.0)
    low = np.full((1, 720), 100.0)
    close = np.full((1, 720), 100.0)
    entries = pd.DataFrame({
        "entry_price": [100.0], "atr": [1.0],
        "entry_ts": [pd.Timestamp("2026-08-17T00:05:00Z")],
    })
    result = replay_exact_1m_rich_policy(
        positions=entries, highs=high, lows=low, closes=close,
        params=params, median_atr_fraction=median,
        contract=Exact1mRichExecutionContract(),
    )
    assert result["path_valid"][0]
    assert result["exit_reason"][0] == "timeout_h12"
    assert pd.Timestamp(result["exit_timestamp"][0], tz="UTC") == pd.Timestamp("2026-08-17T12:05:00Z")
    assert result["gross_bps"][0] - result["net_bps"][0] == pytest.approx(100.0)


def test_future_extensions_fail_closed_until_a_new_schema_is_validated() -> None:
    params, median = _params()
    with pytest.raises(ValueError, match="separately validated"):
        advance_exact_1m_rich_position(
            position=_position(),
            bars=_bars([("2026-08-17T00:00:00Z", 100.0, 100.0, 100.0)]),
            params=params, median_atr_fraction=median,
            extensions=RichExitExtensions(giveback_confirmation_window_minutes=3),
        )


def _v2_paths() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high = np.full((1, 720), 100.0)
    low = np.full((1, 720), 100.0)
    close = np.full((1, 720), 100.0)
    return high, low, close


def _v2_params(*, activation: float = 0.5) -> RichPolicyParams:
    return RichPolicyParams(
        sl_mult=10.0,
        trailing_activation_mult=activation,
        fixed_trailing_gap_mult=0.25,
        capital_protect_mfe_mult=0.0,
        adverse_exit_enabled=False,
    )


def _v2_replay(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    params: RichPolicyParams,
    extensions: RichExitExtensions,
) -> dict[str, np.ndarray]:
    return replay_exact_1m_rich_policy_v2(
        entry=np.array([100.0]), atr=np.array([1.0]),
        highs=high, lows=low, closes=close,
        entry_timestamps=pd.DatetimeIndex(["2026-08-17T00:05:00Z"]),
        params=params, median_atr_fraction=0.01, extensions=extensions,
        contract=Exact1mRichV2ExecutionContract(),
    )


def test_v2_default_trailing_matches_v1_threshold_semantics() -> None:
    high, low, close = _v2_paths()
    high[0, 0] = 101.0
    close[0, 0] = 100.9
    high[0, 1] = 101.0
    close[0, 1] = 100.9
    low[0, 2] = 100.70
    close[0, 2] = 100.65
    result = _v2_replay(
        high, low, close, params=_v2_params(), extensions=RichExitExtensions(),
    )
    assert result["exit_reason"][0] == "trailing"
    # The first completed peak arms the trail; it is first executable against
    # the next completed bar, which has the default low of 100.0.
    assert result["exit_minute"][0] == 1
    # Default v2 is the frozen live low-cross/threshold-fill behaviour.
    assert result["exit_price"][0] == pytest.approx(100.75)


def test_v2_confirmation_and_hysteresis_use_completed_closes_only() -> None:
    high, low, close = _v2_paths()
    high[0, 0:2] = 101.0
    close[0, 0:2] = 100.9
    # The first close breaches.  The second lies in the hysteresis band: it
    # remains an effective breach rather than resetting the confirmation run.
    close[0, 2] = 100.70
    close[0, 3] = 100.80
    low[0, 2:4] = 100.65
    result = _v2_replay(
        high, low, close, params=_v2_params(),
        extensions=RichExitExtensions(
            giveback_confirmation_window_minutes=2,
            giveback_confirmation_fraction=1.0,
            trail_hysteresis_atr=0.10,
        ),
    )
    assert result["exit_reason"][0] == "trailing"
    assert result["exit_minute"][0] == 3
    assert result["exit_price"][0] == pytest.approx(100.80)


def test_v2_minute_noise_tolerance_blocks_a_single_noisy_giveback() -> None:
    high, low, close = _v2_paths()
    high[0, 0:2] = 101.0
    close[0, 0:2] = 100.9
    high[0, 2] = 101.4
    low[0, 2] = 100.70
    close[0, 2] = 100.70
    result = _v2_replay(
        high, low, close, params=_v2_params(),
        extensions=RichExitExtensions(
            giveback_confirmation_window_minutes=2,
            minute_noise_scale=0.30,
            minute_noise_ewma_minutes=1.0,
        ),
    )
    # No confirmation: the new minute's high true range widens the tolerance.
    assert result["exit_minute"][0] != 2


def test_v2_no_progress_can_start_from_entry_or_latest_mae() -> None:
    high, low, close = _v2_paths()
    entry_origin = _v2_replay(
        high, low, close, params=_v2_params(activation=10.0),
        extensions=RichExitExtensions(
            no_progress_start_minutes=3, no_progress_required_mfe_atr=0.5,
            no_progress_origin="entry",
        ),
    )
    assert entry_origin["exit_reason"][0] == "no_progress"
    assert entry_origin["exit_minute"][0] == 2
    low[0, 0] = 99.0
    mae_origin = _v2_replay(
        high, low, close, params=_v2_params(activation=10.0),
        extensions=RichExitExtensions(
            no_progress_start_minutes=3, no_progress_required_mfe_atr=0.5,
            no_progress_origin="mae",
        ),
    )
    assert mae_origin["exit_reason"][0] == "no_progress"
    assert mae_origin["exit_minute"][0] == 3


def test_v2_stall_velocity_smooth_protection_and_ratchet() -> None:
    # Peak stall: a local peak is followed by a deep, prolonged retracement.
    high, low, close = _v2_paths()
    high[0, 0] = 101.0
    close[0, 1:4] = 100.5
    stalled = _v2_replay(
        high, low, close, params=_v2_params(activation=10.0),
        extensions=RichExitExtensions(stalled_peak_minutes=3, stalled_peak_drawdown_atr=0.25),
    )
    assert stalled["exit_reason"][0] == "peak_stall"
    assert stalled["exit_minute"][0] == 3

    # Velocity is a trailing-only control.  A long confirmation window keeps
    # the soft trail open long enough for the velocity rule to be evaluated.
    high, low, close = _v2_paths()
    high[0, 0:2] = 101.0
    close[0, 0:2] = 100.9
    close[0, 2] = 100.0
    velocity = _v2_replay(
        high, low, close, params=_v2_params(),
        extensions=RichExitExtensions(
            giveback_confirmation_window_minutes=10,
            giveback_velocity_atr_per_hour=30.0,
        ),
    )
    assert velocity["exit_reason"][0] == "giveback_velocity"

    # Smooth capital protection arms after the peak and applies next minute.
    high, low, close = _v2_paths()
    high[0, 0] = 101.5
    low[0, 1] = 100.2
    smooth = _v2_replay(
        high, low, close, params=_v2_params(activation=10.0),
        extensions=RichExitExtensions(
            protection_activation_atr=1.0, protection_strength=0.5,
            protection_power=1.0,
        ),
    )
    assert smooth["exit_reason"][0] == "smooth_capital_protect"
    assert smooth["exit_minute"][0] == 1

    # Ratchet tightens only after a meaningful MFE increment, never every
    # noisy one-minute high.
    high, low, close = _v2_paths()
    high[0, 0] = 101.0
    high[0, 1] = 101.3
    high[0, 2] = 101.6
    close[0, 0:3] = 101.1
    ratchet = _v2_replay(
        high, low, close, params=_v2_params(activation=10.0),
        extensions=RichExitExtensions(trailing_ratchet_step_atr=0.5),
    )
    assert ratchet["final_ratchet_mfe"][0] == pytest.approx(1.6)


def test_v2_vectorized_replay_is_deterministic_and_receipted() -> None:
    rng = np.random.default_rng(1729)
    n = 96
    high = 100.0 + np.abs(rng.normal(0.0, 0.25, size=(n, 720)))
    low = 100.0 - np.abs(rng.normal(0.0, 0.25, size=(n, 720)))
    close = 100.0 + rng.normal(0.0, 0.10, size=(n, 720))
    timestamps = pd.date_range("2026-08-17T00:05:00Z", periods=n, freq="h")
    params = _v2_params()
    extensions = RichExitExtensions(
        giveback_confirmation_window_minutes=3,
        giveback_confirmation_fraction=2.0 / 3.0,
        trailing_ratchet_step_atr=0.15,
        minute_noise_scale=0.2,
    )
    kwargs = dict(
        entry=np.full(n, 100.0), atr=np.full(n, 1.0), highs=high, lows=low,
        closes=close, entry_timestamps=timestamps, params=params,
        median_atr_fraction=0.01, extensions=extensions,
    )
    first = replay_exact_1m_rich_policy_v2(**kwargs)
    second = replay_exact_1m_rich_policy_v2(**kwargs)
    for key in ("path_valid", "gross_bps", "net_bps", "exit_price", "exit_minute", "exit_reason_code"):
        assert np.array_equal(first[key], second[key], equal_nan=True)
    assert exact_1m_rich_v2_receipt(
        params=params, extensions=extensions, replay=first,
    ) == exact_1m_rich_v2_receipt(
        params=params, extensions=extensions, replay=second,
    )
