from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.simple_policy_stop import (
    SIMPLE_POLICY_GENERATOR,
    SIMPLE_POLICY_SCHEMA,
    SimplePolicyStopParamsError,
    compute_initial_simple_policy_stop_decision,
    compute_simple_policy_stop_decision,
)
from extreme_price_movements.simple_policy_optimiser import simulate_and_score
from extreme_price_movements.simple_policy_winner import WINNER_POLICY_PATHWAY_ID


def _winner_params(tmp_path, *, strategy_id: str, adverse_exit_enabled: bool) -> dict:
    params_source = (
        "artifacts/test-run/simple_policy_optimiser/deployment/"
        "best_policy_params.json"
    )
    artifact_path = tmp_path / params_source
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_text("{}", encoding="utf-8")
    sizing_state = {
        "policy_id": "raw_bayesian_v1",
        "pathway_id": WINNER_POLICY_PATHWAY_ID,
        "train_normalizer": 0.9,
    }
    return {
        "params_source": params_source,
        "params_hash": hashlib.sha256(artifact_path.read_bytes()).hexdigest()[:16],
        "_loaded_from_simple_policy_artifact": True,
        "_artifact_path": str(artifact_path),
        "generated_by": SIMPLE_POLICY_GENERATOR,
        "schema": SIMPLE_POLICY_SCHEMA,
        "strategy_id": strategy_id,
        "policy_pathway_id": WINNER_POLICY_PATHWAY_ID,
        "replay_timeframe": "1m",
        "trailing_activation_curve": "total_mfe",
        "capital_preservation_enabled": False,
        "sizing_policy_id": "raw_bayesian_v1",
        "raw_bayesian_sizing_state": sizing_state,
        "barrier_frac": 0.01,
        "sl_mult": 3.0,
        "trailing_activation_mult": 1.0,
        "trailing_power": 1.5,
        "trailing_squash_divisor": 10.0,
        "giveback_beta": 0.5,
        "atr_power": 1.0,
        "atr_multiplier": 1.0,
        "hard_tp_abs_pct": 0.0,
        "exit_pressure_enabled": False,
        # Deliberately nonzero: the winner contract must still disable this path.
        "capital_protect_mfe_mult": 0.1,
        "capital_protect_regression_frac": 0.45,
        "adverse_exit_enabled": adverse_exit_enabled,
        "adverse_exit_alpha": 1.0,
        "adverse_exit_beta": 1.0,
        "adverse_exit_delta": 1.0,
        "adverse_exit_theta_quantile": 0.75,
        "adverse_exit_theta": 2.4018619060516357 if adverse_exit_enabled else 1.0e9,
        "adverse_exit_fast_bars": 60,
        "adverse_exit_min_mae_atr": 1.4 if adverse_exit_enabled else 1.9,
        "adverse_exit_min_speed": 0.3 if adverse_exit_enabled else 1.4,
        "adverse_exit_max_mfe_atr": 0.25,
    }


def _completed_bar(timestamp: str, *, high: float, low: float, close: float) -> pd.DataFrame:
    bars = pd.DataFrame(
        {"open": [100.0], "high": [high], "low": [low], "close": [close], "is_complete": [True]},
        index=pd.DatetimeIndex([timestamp], tz="UTC"),
    )
    bars.attrs["timeframe"] = "1m"
    return bars


def _sim_rows(side: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-18T00:00:00Z"]),
            "symbol": ["BTC/USD:USD"],
            "rank_pct": [0.5],
            "side": [side],
            "barrier_pct": [0.01],
            "expected_spread_bps": [0.0],
        }
    )


def test_winner_long_adverse_exit_matches_optimizer_1m_simulator(tmp_path) -> None:
    params = _winner_params(tmp_path, strategy_id="long_winner", adverse_exit_enabled=True)
    opens = np.array([[100.0, 100.0]], dtype=np.float32)
    highs = np.array([[100.1, 100.1]], dtype=np.float32)
    lows = np.array([[99.9, 98.4]], dtype=np.float32)
    closes = np.array([[100.0, 98.5]], dtype=np.float32)
    metrics = simulate_and_score(
        _sim_rows(1.0),
        opens,
        highs,
        lows,
        closes,
        cost_pct=0.0,
        policy_pathway_id=WINNER_POLICY_PATHWAY_ID,
        replay_timeframe="1m",
        trailing_activation_curve="total_mfe",
        capital_preservation_enabled=False,
        sl_mult=params["sl_mult"],
        trailing_activation_mult=params["trailing_activation_mult"],
        trailing_power=params["trailing_power"],
        trailing_squash_divisor=params["trailing_squash_divisor"],
        giveback_beta=params["giveback_beta"],
        adverse_exit_enabled=True,
        adverse_exit_theta=params["adverse_exit_theta"],
        adverse_exit_fast_bars=60,
        adverse_exit_min_mae_atr=1.4,
        adverse_exit_min_speed=0.3,
        adverse_exit_max_mfe_atr=0.25,
        max_concurrent_trades=10,
        max_concurrent_per_asset=10,
    )
    decision = compute_simple_policy_stop_decision(
        state={
            "entry_price": 100.0,
            "stop_price": 97.0,
            "strategy_id": "long_winner",
            "barrier_frac": 0.01,
            "barrier_frac_is_effective": True,
            "rank_percentile": 0.5,
            "bars_in_trade": 0,
        },
        latest_market_state=_completed_bar(
            "2026-07-18T00:01:00Z", high=100.1, low=98.4, close=98.5
        ),
        policy_params=params,
        side="long",
    )

    assert list(metrics["exit_reason"]) == ["adverse_exit"]
    assert metrics["exit_bars"][0] == 1
    assert decision.should_exit is True
    assert decision.exit_reason == "adverse_excursion_exit"
    assert decision.policy_pathway_id == WINNER_POLICY_PATHWAY_ID
    assert decision.raw_bayesian_sizing_state == params["raw_bayesian_sizing_state"]


def test_winner_trailing_promotion_matches_total_mfe_simulator(tmp_path) -> None:
    params = _winner_params(tmp_path, strategy_id="short_winner", adverse_exit_enabled=False)
    opens = np.array([[100.0, 100.0, 100.0]], dtype=np.float32)
    highs = np.array([[100.1, 99.5, 99.5]], dtype=np.float32)
    lows = np.array([[99.9, 98.0, 98.5]], dtype=np.float32)
    closes = np.array([[100.0, 98.5, 99.0]], dtype=np.float32)
    metrics = simulate_and_score(
        _sim_rows(-1.0),
        opens,
        highs,
        lows,
        closes,
        cost_pct=0.0,
        policy_pathway_id=WINNER_POLICY_PATHWAY_ID,
        replay_timeframe="1m",
        trailing_activation_curve="total_mfe",
        capital_preservation_enabled=False,
        sl_mult=params["sl_mult"],
        trailing_activation_mult=params["trailing_activation_mult"],
        trailing_power=params["trailing_power"],
        trailing_squash_divisor=params["trailing_squash_divisor"],
        giveback_beta=params["giveback_beta"],
        adverse_exit_enabled=False,
        max_concurrent_trades=10,
        max_concurrent_per_asset=10,
    )
    decision = compute_simple_policy_stop_decision(
        state={
            "entry_price": 100.0,
            "stop_price": 103.0,
            "strategy_id": "short_winner",
            "barrier_frac": 0.01,
            "barrier_frac_is_effective": True,
            "bars_in_trade": 0,
        },
        latest_market_state=_completed_bar(
            "2026-07-18T00:01:00Z", high=99.5, low=98.0, close=98.5
        ),
        policy_params=params,
        side="short",
    )

    # The runtime promotion after the completed first bar is the stop the
    # optimizer uses for its next one-minute candle, which exits as trailing.
    assert list(metrics["exit_reason"]) == ["trailing"]
    assert metrics["exit_bars"][0] == 2
    assert decision.should_replace is True
    assert decision.reason == "trailing_profit"
    assert decision.stop_price == pytest.approx(98.911, rel=1e-5)
    assert decision.capital_protect_armed is False


def test_winner_contract_rejects_nonminute_and_nonwinner_exit_metadata(tmp_path) -> None:
    params = _winner_params(tmp_path, strategy_id="short_winner", adverse_exit_enabled=False)
    params["capital_preservation_enabled"] = True
    with pytest.raises(SimplePolicyStopParamsError, match="capital_preservation_enabled=false"):
        compute_initial_simple_policy_stop_decision(
            entry_price=100.0,
            policy_params=params,
            side="short",
            strategy_id="short_winner",
        )

    params["capital_preservation_enabled"] = False
    params["atr_power"] = 0.9
    with pytest.raises(SimplePolicyStopParamsError, match="raw ATR scaling"):
        compute_initial_simple_policy_stop_decision(
            entry_price=100.0,
            policy_params=params,
            side="short",
            strategy_id="short_winner",
        )

    params["atr_power"] = 1.0
    bars = pd.concat(
        [
            _completed_bar("2026-07-18T00:01:00Z", high=100.1, low=99.9, close=100.0),
            _completed_bar("2026-07-18T00:16:00Z", high=100.2, low=99.8, close=100.0),
        ]
    )
    bars.attrs["timeframe"] = "15m"
    with pytest.raises(SimplePolicyStopParamsError, match="completed 1m bars"):
        compute_simple_policy_stop_decision(
            state={
                "entry_price": 100.0,
                "stop_price": 103.0,
                "strategy_id": "short_winner",
                "barrier_frac": 0.01,
                "barrier_frac_is_effective": True,
            },
            latest_market_state=bars,
            policy_params=params,
            side="short",
        )
