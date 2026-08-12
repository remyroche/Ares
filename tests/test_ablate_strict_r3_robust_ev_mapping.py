from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ablate_strict_r3_robust_ev_mapping.py"
SPEC = importlib.util.spec_from_file_location("robust_ev_map_ablation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_robust_day_trim_removes_both_residual_tails() -> None:
    days = pd.date_range("2026-01-01", periods=20, freq="D", tz="UTC")
    daily = pd.DataFrame({"__day__": days, "robust_z": np.arange(20, dtype=float)})
    kept = MODULE._keep_trimmed_days(daily, 0.10)
    assert len(kept) == 16
    assert days[0] not in kept and days[1] not in kept
    assert days[-1] not in kept and days[-2] not in kept


def test_state_expected_fails_to_prior_without_three_resolved_days() -> None:
    state = pd.DataFrame({
        "ev_state_level21_bps": [100.0],
        "ev_state_trend_bps": [20.0],
        "ev_state_slope_bps_per_day": [5.0],
        "ev_state_std_bps": [50.0],
        "ev_state_sign_entropy": [0.2],
        "ev_state_reference_days": [2.0],
    })
    for arm in MODULE.STATE_ARMS:
        np.testing.assert_allclose(MODULE._state_expected(np.array([75.0]), state, arm), [75.0])


def test_future_outcomes_do_not_change_earlier_daily_state() -> None:
    decision = pd.date_range("2026-01-01", periods=8, freq="D", tz="UTC")
    base = pd.DataFrame({
        "candidate_id": [f"row-{i}" for i in range(8)],
        "__decision_ts__": decision,
        "policy_label_available_ts": decision + pd.Timedelta(hours=12),
        "policy_path_valid": True,
        "ev_bridge_policy_residual_bps": np.linspace(-50.0, 90.0, 8),
    })
    early = MODULE._daily_residual_state(base.iloc[:6].copy())
    full = MODULE._daily_residual_state(base.copy()).iloc[:6].reset_index(drop=True)
    columns = [column for column in early if column.startswith("ev_state_")]
    pd.testing.assert_frame_equal(early[columns], full[columns])


def test_reserve_history_seeds_state_on_first_held_day() -> None:
    reserve_day = pd.date_range("2025-12-20", periods=5, freq="D", tz="UTC")
    held_day = pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True)
    frame = pd.DataFrame({
        "candidate_id": [f"reserve-{i}" for i in range(5)] + ["held"],
        "__decision_ts__": reserve_day.append(held_day),
        "policy_label_available_ts": reserve_day.append(held_day) + pd.Timedelta(hours=12),
        "policy_path_valid": True,
        "ev_bridge_policy_residual_bps": [10.0, 20.0, 30.0, 40.0, 50.0, np.nan],
    })
    state = MODULE._daily_residual_state(frame).iloc[-1]
    assert state["ev_state_reference_days"] == 5.0
    assert np.isfinite(state["ev_state_level21_bps"])


def test_entropy_is_bounded_and_detects_one_sided_state() -> None:
    assert MODULE._entropy(np.array([1.0, 2.0, 3.0])) == 0.0
    assert np.isclose(MODULE._entropy(np.array([-1.0, 1.0])), 1.0)
