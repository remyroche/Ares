from __future__ import annotations

import ast
import numpy as np
import pandas as pd

from extreme_price_movements.strict_r3_frozen_policy_labels import (
    causal_hourly_atr_from_hourly,
    replay_frozen_policy_15m,
    replay_policy_hourly_proxy,
)
from scripts.materialize_strict_r3_frozen_policy_labels_v2 import (
    _causal_hourly_atr_from_15m,
)


def _bars() -> pd.DataFrame:
    index = pd.date_range("2026-08-01 01:00", periods=60, freq="15min", tz="UTC")
    return pd.DataFrame({
        "open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0,
    }, index=index)


def test_frozen_policy_uses_prior_bar_trailing_state_and_one_cost() -> None:
    bars = _bars()
    # Bar zero reaches >0.5 ATR.  The trailing exit may occur only on bar one.
    bars.iloc[0, bars.columns.get_loc("high")] = 101.0
    bars.iloc[1, bars.columns.get_loc("low")] = 100.70
    candidates = pd.DataFrame({
        "candidate_id": ["a"], "__decision_ts__": [bars.index[0]],
        "side_name": ["long"], "atr_1h": [1.0],
    })
    result = replay_frozen_policy_15m(candidates, bars)
    assert result["policy_path_valid"].iloc[0]
    assert result["policy_exit_bar_15m"].iloc[0] == 1
    assert result["policy_exit_reason"].iloc[0] == "trailing"
    assert result["policy_gross_bps"].iloc[0] == 75.0
    assert result["policy_net_bps"].iloc[0] == -25.0
    assert result["policy_cost_bps"].iloc[0] == 100.0


def test_frozen_policy_is_side_symmetric() -> None:
    bars = _bars()
    bars.iloc[0, bars.columns.get_loc("high")] = 104.0
    bars.iloc[0, bars.columns.get_loc("low")] = 96.0
    candidates = pd.DataFrame({
        "candidate_id": ["long", "short"],
        "__decision_ts__": [bars.index[0], bars.index[0]],
        "side_name": ["long", "short"], "atr_1h": [1.0, 1.0],
    })
    result = replay_frozen_policy_15m(candidates, bars)
    assert result["policy_exit_reason"].tolist() == ["stop_loss", "stop_loss"]
    np.testing.assert_allclose(result["policy_gross_bps"], [-300.0, -300.0])
    np.testing.assert_allclose(result["policy_net_bps"], [-400.0, -400.0])


def test_frozen_policy_accepts_selected_geometry() -> None:
    bars = _bars()
    bars.iloc[0, bars.columns.get_loc("low")] = 95.0
    candidates = pd.DataFrame({
        "candidate_id": ["a"], "__decision_ts__": [bars.index[0]],
        "side_name": ["long"], "atr_1h": [1.0],
    })
    result = replay_frozen_policy_15m(
        candidates, bars,
        stop_loss_atr=4.0,
        trailing_activation_atr=2.0,
        trailing_giveback_atr=0.1,
    )
    assert result.loc[0, "policy_exit_reason"] == "stop_loss"
    assert result.loc[0, "policy_gross_bps"] == -400.0


def test_zero_volume_flat_h12_path_is_invalid_not_minus_cost() -> None:
    bars = _bars()
    bars.loc[:, ["open", "high", "low", "close"]] = 100.0
    bars["volume"] = 0.0
    candidates = pd.DataFrame({
        "candidate_id": ["a"], "__decision_ts__": [bars.index[0]],
        "side_name": ["long"], "atr_1h": [1.0],
    })
    result = replay_frozen_policy_15m(candidates, bars)
    assert not bool(result.loc[0, "policy_path_valid"])
    assert result.loc[0, "policy_exit_reason"] == "invalid_path"
    assert np.isnan(result.loc[0, "policy_gross_bps"])
    assert np.isnan(result.loc[0, "policy_net_bps"])


def test_materializer_loads_policy_inside_main_before_replay() -> None:
    path = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "scripts/materialize_strict_r3_frozen_policy_labels_v2.py"
    )
    tree = ast.parse(path.read_text())
    main = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    names = {node.id for node in ast.walk(main) if isinstance(node, ast.Name)}
    assert "policy" in names and "policy_payload" in names


def test_causal_15m_atr_fallback_does_not_use_decision_or_future_bars() -> None:
    signal = pd.Timestamp("2026-08-02 00:00", tz="UTC")
    index = pd.date_range(signal - pd.Timedelta(hours=20), periods=132, freq="15min")
    bars = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
    }, index=index)
    baseline = _causal_hourly_atr_from_15m(bars).loc[signal]
    decision = signal + pd.Timedelta(hours=1)
    bars.loc[decision:, ["high", "low"]] = [250.0, 1.0]
    after = _causal_hourly_atr_from_15m(bars).loc[signal]
    assert np.isfinite(baseline) and baseline > 0.0
    assert after == baseline


def test_hourly_proxy_uses_only_completed_prior_bars_for_atr() -> None:
    signal = pd.Timestamp("2026-08-02 00:00", tz="UTC")
    index = pd.date_range(signal - pd.Timedelta(hours=20), periods=40, freq="1h")
    bars = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
    }, index=index)
    decision = signal + pd.Timedelta(hours=1)
    before = causal_hourly_atr_from_hourly(bars).loc[decision]
    changed = bars.copy()
    changed.loc[decision:, ["high", "low"]] = [250.0, 1.0]
    after = causal_hourly_atr_from_hourly(changed).loc[decision]
    assert before == after


def test_hourly_proxy_is_explicit_and_never_exits_before_hour_end() -> None:
    index = pd.date_range("2026-08-01 01:00", periods=20, freq="1h", tz="UTC")
    bars = pd.DataFrame({
        "open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0,
    }, index=index)
    bars.iloc[0, bars.columns.get_loc("high")] = 101.0
    bars.iloc[1, bars.columns.get_loc("low")] = 100.70
    candidates = pd.DataFrame({
        "candidate_id": ["a"], "__decision_ts__": [index[0]],
        "side_name": ["long"], "atr_1h": [1.0],
    })
    result = replay_policy_hourly_proxy(
        candidates,
        bars,
        stop_loss_atr=3.0,
        trailing_activation_atr=0.5,
        trailing_giveback_atr=0.25,
    )
    assert result.loc[0, "policy_path_valid"]
    assert result.loc[0, "policy_exit_bar_1h"] == 1
    assert result.loc[0, "policy_exit_bar_15m"] == 7
    assert result.loc[0, "policy_outcome_source"] == "hourly_ohlc_proxy"
    assert result.loc[0, "policy_net_bps"] == -25.0
