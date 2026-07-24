import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import extreme_price_movements.inference.run_inference as run_inference
from extreme_price_movements.inference.candidate_selector import (
    build_strategy_candidate_masks,
    select_candidates,
)
from extreme_price_movements.inference.feature_generator import (
    _synthesize_live_safe_feature_keys,
)
from extreme_price_movements.inference.portfolio_policy import PortfolioPolicyConfig
from extreme_price_movements.inference.run_inference import (
    _ev_adjusted_prediction_after_entry_friction,
    _evaluate_oco_policy,
    _latest_closed_candle_start,
    _monitor_active_position_price_action,
    _position_policy_entry_price,
)
from extreme_price_movements.inference.simple_policy_stop import (
    MIN_TRAILING_GIVEBACK_FRAC,
    SIMPLE_POLICY_GENERATOR,
    SIMPLE_POLICY_SCHEMA,
    SimplePolicyStopDecision,
    SimplePolicyStopParamsError,
    compute_simple_policy_stop_decision,
    extract_simple_policy_stop_params_by_strategy,
    load_simple_policy_stop_params_by_strategy,
)
from extreme_price_movements.inference.trade_executor import (
    CANONICAL_STOP_POSITION_FIELDS,
    MODEL_AND_POLICY_CONTEXT_KEYS,
    OCOExecutor,
    TradeExecutor,
    _classify_exchange_error,
    _closed_trade_metrics,
    _create_reduce_stop_loss_order,
    _default_cross_margin_dust_quote_threshold,
    _enrich_order_from_exchange,
    _kraken_futures_last_stop_from_executable_stop,
    _protective_stop_coverage,
    _protective_stop_trigger_matches_policy,
    _stop_coverage_is_sufficient,
    _stop_is_at_least_as_protective,
    _stop_trigger_reference_price,
    _safety_take_profit_target,
    _update_live_path_extrema_from_price,
)
from extreme_price_movements.optimise import _select_candidate_trade_mask


def test_protective_stop_coverage_requires_full_exchange_position():
    orders = [
        {"amount": 100.0, "filled": 4.0, "remaining": 96.0},
        {"amount": 5.0, "filled": 0.0, "remaining": 5.0},
    ]
    coverage = _protective_stop_coverage(orders)
    assert coverage == pytest.approx(101.0)
    assert _stop_coverage_is_sufficient(coverage, 101.0)
    assert not _stop_coverage_is_sufficient(coverage, 102.0)


def test_live_stop_geometry_uses_realized_fill_without_overwriting_theoretical_entry():
    state = {
        "side": "long",
        "entry_price": 0.0600,
        "policy_entry_price": 0.0600,
        "theoretical_entry_price": 0.0600,
        "realized_entry_price": 0.05895,
        "peak_price": 0.05895,
        "mfe": 0.0,
        "mae": 0.0,
    }

    result = _update_live_path_extrema_from_price(
        state,
        side="long",
        price=0.05861,
        timestamp=pd.Timestamp("2026-07-17T19:08:53Z"),
    )

    assert result["entry_price"] == pytest.approx(0.05895)
    assert result["mae"] == pytest.approx((0.05895 - 0.05861) / 0.05895)
    assert state["stop_geometry_entry_price"] == pytest.approx(0.05895)
    assert state["policy_entry_price"] == pytest.approx(0.0600)
    assert state["theoretical_entry_price"] == pytest.approx(0.0600)


def test_trade_context_cannot_overwrite_realized_execution_entry_fields():
    assert {
        "entry_price",
        "realized_entry_price",
        "actual_entry_price",
        "stop_geometry_entry_price",
    }.issubset(CANONICAL_STOP_POSITION_FIELDS)


@pytest.mark.parametrize(
    ("side", "mfe", "expected_price", "expected_return"),
    [
        ("long", 0.0, 102.0, 0.02),
        ("long", 0.03, 103.5, 0.035),
        ("short", 0.0, 98.0, 0.02),
        ("short", 0.03, 96.5, 0.035),
    ],
)
def test_safety_take_profit_target_is_beyond_mfe_with_two_pct_floor(
    side, mfe, expected_price, expected_return
):
    target = _safety_take_profit_target(
        entry_price=100.0,
        side=side,
        mfe=mfe,
    )
    assert target["target_price"] == pytest.approx(expected_price)
    assert target["target_return"] == pytest.approx(expected_return)


class _SafetyTakeProfitExchange:
    def __init__(self):
        self.created = []
        self.cancelled = []

    def create_order(self, **kwargs):
        self.created.append(kwargs)
        return {"id": f"tp-{len(self.created)}", "status": "open", **kwargs}

    def cancel_order(self, order_id, symbol, params=None):
        self.cancelled.append((order_id, symbol, params or {}))
        return {"id": order_id, "status": "canceled"}


def test_safety_take_profit_only_ratchets_on_new_mfe():
    exchange = _SafetyTakeProfitExchange()
    executor = OCOExecutor(
        exchange,
        {},
        config={"execution_account": "perps", "safety_take_profit_enabled": True},
    )
    state = {
        "side": "long",
        "entry_price": 100.0,
        "realized_entry_price": 100.0,
        "size": 2.0,
    }

    initial = executor._ensure_safety_take_profit_order(
        "TEST/USD:USD", state, observed_mfe=0.0, mfe_improved=False
    )
    unchanged = executor._ensure_safety_take_profit_order(
        "TEST/USD:USD", state, observed_mfe=0.0, mfe_improved=False
    )
    improved = executor._ensure_safety_take_profit_order(
        "TEST/USD:USD", state, observed_mfe=0.03, mfe_improved=True
    )

    assert initial["updated"] is True
    assert unchanged == {"updated": False, "reason": "mfe_not_improved"}
    assert improved["updated"] is True
    assert [order["price"] for order in exchange.created] == pytest.approx(
        [102.0, 103.5]
    )
    assert exchange.cancelled[0][0] == "tp-1"
    assert state["take_profit_order_id"] == "tp-2"


def test_canonical_position_update_wires_safety_take_profit():
    exchange = _SafetyTakeProfitExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        config={"execution_account": "perps", "safety_take_profit_enabled": True},
    )
    assert executor.oco_executor is not None
    executor.oco_executor.active_positions["TEST/USD:USD"] = {
        "side": "short",
        "entry_price": 100.0,
        "realized_entry_price": 100.0,
        "size": 2.0,
        "mfe": 0.0,
    }

    executor.update_position_policy_state("TEST/USD:USD", mfe=0.03)
    executor.update_position_policy_state("TEST/USD:USD", mfe=0.03)

    assert len(exchange.created) == 1
    assert exchange.created[0]["side"] == "buy"
    assert exchange.created[0]["price"] == pytest.approx(96.5)


def test_raw_signal_close_reliability_rejects_zero_volume_without_substitution():
    idx = pd.date_range("2026-06-10 08:00", periods=1, freq="h", tz="UTC")
    symbol = "CYBER/USD:USD"
    panel = {
        "close": pd.DataFrame({symbol: [0.3559]}, index=idx),
        "volume": pd.DataFrame({symbol: [0.0]}, index=idx),
        "mark_close": pd.DataFrame({symbol: [0.341955]}, index=idx),
        "index_price": pd.DataFrame({symbol: [0.34226]}, index=idx),
    }

    snap = run_inference._raw_signal_close_reliability_snapshot(
        panel,
        symbol,
        max_reference_gap_bps=150.0,
    )

    assert snap["raw_signal_close_unreliable"] is True
    assert snap["raw_signal_close_unreliable_reason"] == "zero_volume_raw_close"
    assert snap["signal_price"] == pytest.approx(0.3559)
    assert snap["raw_signal_close"] == pytest.approx(0.3559)
    assert snap["raw_signal_close_reference_source"] in {
        "mark_close",
        "index_price",
    }
    assert snap["raw_signal_close_reference_gap_bps"] > 150.0


def test_raw_signal_close_reliability_rejects_large_reference_gap_without_substitution():
    idx = pd.date_range("2026-06-10 08:00", periods=1, freq="h", tz="UTC")
    symbol = "CYBER/USD:USD"
    panel = {
        "close": pd.DataFrame({symbol: [0.3559]}, index=idx),
        "volume": pd.DataFrame({symbol: [10.0]}, index=idx),
        "mark_close": pd.DataFrame({symbol: [0.341955]}, index=idx),
    }

    snap = run_inference._raw_signal_close_reliability_snapshot(
        panel,
        symbol,
        max_reference_gap_bps=150.0,
    )

    assert snap["raw_signal_close_unreliable"] is True
    assert (
        snap["raw_signal_close_unreliable_reason"]
        == "raw_close_reference_gap_too_large"
    )
    assert snap["signal_price"] == pytest.approx(0.3559)
    assert snap["raw_signal_close_reference_price"] == pytest.approx(0.341955)


def test_raw_signal_close_reliability_accepts_clean_raw_close():
    idx = pd.date_range("2026-06-10 08:00", periods=1, freq="h", tz="UTC")
    symbol = "ETH/USD:USD"
    panel = {
        "close": pd.DataFrame({symbol: [3500.0]}, index=idx),
        "volume": pd.DataFrame({symbol: [25.0]}, index=idx),
        "mark_close": pd.DataFrame({symbol: [3498.0]}, index=idx),
    }

    snap = run_inference._raw_signal_close_reliability_snapshot(
        panel,
        symbol,
        max_reference_gap_bps=150.0,
    )

    assert snap["raw_signal_close_unreliable"] is False
    assert snap["raw_signal_close_unreliable_reason"] == ""
    assert snap["signal_price"] == pytest.approx(3500.0)
    assert snap["raw_signal_close_reference_gap_bps"] < 150.0


class _PrescoreExchange:
    def __init__(self, *, bid=100.0, ask=100.1):
        self._bid = bid
        self._ask = ask

    def fetch_ticker(self, symbol):
        return {
            "bid": self._bid,
            "ask": self._ask,
            "last": (self._bid + self._ask) / 2.0,
        }

    def fetch_order_book(self, symbol):
        return {
            "asks": [[self._ask, 10.0]],
            "bids": [[self._bid, 10.0]],
        }


class _PrescoreExecutor:
    mode = "live-test"

    def __init__(self, exchange):
        self.exchange = exchange


def test_executable_stop_sentinel_updates_mfe_and_intrabar_trailing(tmp_path):
    artifact_path = tmp_path / "best_policy_params.json"
    artifact_path.write_text("{}", encoding="utf-8")
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()[:16]
    strategy_id = "long_test"
    params = {
        strategy_id: {
            "generated_by": SIMPLE_POLICY_GENERATOR,
            "schema": SIMPLE_POLICY_SCHEMA,
            "strategy_id": strategy_id,
            "params_source": artifact_path.as_posix(),
            "params_hash": artifact_hash,
            "_loaded_from_simple_policy_artifact": True,
            "_artifact_path": artifact_path.as_posix(),
            "barrier_frac": 0.02,
            "sl_mult": 1.0,
            "sl_abs_cap_pct": 0.0,
            "enable_trailing": True,
            "trailing_activation_mult": 0.5,
            "trailing_activation_cap_pct": 0.0,
            "trailing_activation_decay_half_life_bars": 0.0,
            "trailing_activation_decay_start_bars": 0,
            "trailing_activation_min_mult": 1.0,
            "trailing_power": 1.5,
            "trailing_squash_divisor": 2.0,
            "giveback_beta": 0.5,
            "round_trip_cost_pct": 0.002,
            "capital_protect_mfe_mult": 1.0,
            "capital_protect_regression_frac": 0.45,
            "capital_protect_min_lock_bps": 0.0,
            "adverse_exit_enabled": False,
            "adverse_exit_min_mae_atr": 1.0,
            "adverse_exit_min_speed": 0.3,
            "adverse_exit_fast_bars": 4,
            "adverse_exit_max_mfe_atr": 0.25,
        }
    }
    executor = OCOExecutor(
        _PrescoreExchange(bid=103.0, ask=103.1),
        {},
        config={"lightweight_stop_sentinel_fetch_orderbook": False},
    )
    executor.simple_policy_stop_params_by_strategy = params
    executor.active_positions["TEST/USD:USD"] = {
        "side": "long",
        "entry_price": 100.0,
        "policy_entry_price": 100.0,
        "size": 1.0,
        "bucket_key": strategy_id,
        "strategy_id": strategy_id,
        "stop_price": 98.0,
        "policy_stop_price": 98.0,
        "requested_policy_stop": 98.0,
        "exchange_stop_price": 98.0,
        "final_placed_stop": 98.0,
        "stop_order_id": "stub-stop",
        "stop_reason": "original_stop_loss",
        "barrier_frac": 0.02,
        "barrier_pct": 0.02,
        "sl_mult": 1.0,
        "stop_policy_params_source": artifact_path.as_posix(),
        "stop_policy_params_hash": artifact_hash,
        "stop_policy_schema": SIMPLE_POLICY_SCHEMA,
    }
    updates = []

    def fake_update(symbol, state, decision):
        updates.append(
            (
                symbol,
                bool(decision.should_replace),
                float(decision.stop_price),
                decision.reason,
                float(decision.mfe),
            )
        )
        state["stop_price"] = float(decision.stop_price)
        state["policy_stop_price"] = float(decision.stop_price)
        state["requested_policy_stop"] = float(decision.stop_price)
        state["stop_reason"] = decision.reason

    executor._update_stop_loss_from_policy_decision = fake_update

    status = executor.monitor_executable_stops_once()["TEST/USD:USD"]
    state = executor.active_positions["TEST/USD:USD"]

    assert state.get("mfe", 0.0) == pytest.approx(0.0)
    assert state["intrabar_trailing_mfe"] == pytest.approx(0.03)
    assert state["intrabar_trailing_peak_price"] == pytest.approx(103.0)
    assert updates
    assert updates[0][1] is True
    assert updates[0][3] == "trailing_profit"
    assert state["stop_price"] > 98.0
    assert status["mfe_updated"] is True
    assert status["intrabar_policy_update"]["evaluated"] is True
    assert status["intrabar_policy_update"]["capital_protection_allowed"] is True
    assert state.get("capital_protect_armed", False) is False


def test_lightweight_sentinel_updates_short_intrabar_mfe_and_trailing(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "best_policy_params.json"
    artifact_path.write_text("{}\n")
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()[:16]
    strategy_id = "short_test"
    params = {
        strategy_id: {
            "generated_by": SIMPLE_POLICY_GENERATOR,
            "schema": SIMPLE_POLICY_SCHEMA,
            "strategy_id": strategy_id,
            "params_source": artifact_path.as_posix(),
            "params_hash": artifact_hash,
            "_loaded_from_simple_policy_artifact": True,
            "_artifact_path": artifact_path.as_posix(),
            "barrier_frac": 0.02,
            "sl_mult": 1.0,
            "sl_abs_cap_pct": 0.0,
            "enable_trailing": True,
            "trailing_activation_mult": 0.5,
            "trailing_activation_cap_pct": 0.0,
            "trailing_activation_decay_half_life_bars": 0.0,
            "trailing_activation_decay_start_bars": 0,
            "trailing_activation_min_mult": 1.0,
            "trailing_power": 1.5,
            "trailing_squash_divisor": 2.0,
            "giveback_beta": 0.5,
            "round_trip_cost_pct": 0.002,
            "capital_protect_mfe_mult": 0.0,
            "capital_protect_regression_frac": 0.45,
            "capital_protect_min_lock_bps": 0.0,
            "adverse_exit_enabled": False,
            "adverse_exit_min_mae_atr": 1.0,
            "adverse_exit_min_speed": 0.3,
            "adverse_exit_fast_bars": 4,
            "adverse_exit_max_mfe_atr": 0.25,
        }
    }
    executor = OCOExecutor(
        _PrescoreExchange(bid=96.9, ask=97.0),
        {},
        config={"lightweight_stop_sentinel_fetch_orderbook": False},
    )
    executor.simple_policy_stop_params_by_strategy = params
    executor.active_positions["TEST/USD:USD"] = {
        "side": "short",
        "entry_price": 100.0,
        "policy_entry_price": 100.0,
        "size": 1.0,
        "bucket_key": strategy_id,
        "strategy_id": strategy_id,
        "stop_price": 102.0,
        "policy_stop_price": 102.0,
        "requested_policy_stop": 102.0,
        "exchange_stop_price": 102.0,
        "final_placed_stop": 102.0,
        "stop_order_id": "stub-stop",
        "stop_reason": "original_stop_loss",
        "barrier_frac": 0.02,
        "barrier_pct": 0.02,
        "sl_mult": 1.0,
        "stop_policy_params_source": artifact_path.as_posix(),
        "stop_policy_params_hash": artifact_hash,
        "stop_policy_schema": SIMPLE_POLICY_SCHEMA,
    }
    updates = []

    def fake_update(symbol, state, decision):
        updates.append(
            (
                symbol,
                bool(decision.should_replace),
                float(decision.stop_price),
                decision.reason,
                float(decision.mfe),
            )
        )
        state["stop_price"] = float(decision.stop_price)
        state["policy_stop_price"] = float(decision.stop_price)
        state["requested_policy_stop"] = float(decision.stop_price)
        state["stop_reason"] = decision.reason

    executor._update_stop_loss_from_policy_decision = fake_update

    status = executor.monitor_executable_stops_once()["TEST/USD:USD"]
    state = executor.active_positions["TEST/USD:USD"]

    assert state.get("mfe", 0.0) == pytest.approx(0.0)
    assert state["intrabar_trailing_mfe"] == pytest.approx(0.03)
    assert state["intrabar_trailing_peak_price"] == pytest.approx(97.0)
    assert updates
    assert updates[0][1] is True
    assert updates[0][3] == "trailing_profit"
    assert state["stop_price"] < 102.0
    assert status["mfe_updated"] is True
    assert status["intrabar_policy_update"]["evaluated"] is True
    assert status["intrabar_policy_update"]["capital_protection_allowed"] is True


def test_pre_score_market_mask_rejects_wide_spread_before_scoring():
    idx = pd.date_range("2026-06-10 08:00", periods=1, freq="h", tz="UTC")
    symbol = "ETH/USD:USD"
    panel = {
        "close": pd.DataFrame({symbol: [100.0]}, index=idx),
        "volume": pd.DataFrame({symbol: [10.0]}, index=idx),
        "mark_close": pd.DataFrame({symbol: [100.0]}, index=idx),
        "open_interest": pd.DataFrame({symbol: [1000000.0]}, index=idx),
    }

    snap = run_inference._pre_score_market_mask_snapshot(
        panel=panel,
        symbol=symbol,
        side="long",
        strategy_id="long_mr",
        executor=_PrescoreExecutor(_PrescoreExchange(bid=100.0, ask=103.0)),
        policy=run_inference.PortfolioPolicyConfig(max_spread_bps=25.0),
        runtime_config={
            "mode": "live-test",
            "market_mode": "perps",
            "live_prescore_orderbook_enabled": False,
        },
        now=pd.Timestamp("2026-06-10 09:05:00Z"),
        signal_bar_ts=idx[-1],
        raw_close_reference_gap_bps=150.0,
        max_signal_close_to_entry_seconds=900.0,
    )

    assert snap["prescore_market_mask_allowed"] is False
    assert snap["prescore_market_mask_reason"] == "ticker_spread_above_prescore_max"
    assert snap["prescore_ticker_spread_bps"] > 25.0


def test_live_prescore_market_mask_is_opt_in_to_preserve_replay_denominator(
    monkeypatch,
):
    monkeypatch.delenv("EPM_LIVE_PRESCORE_MARKET_MASK_ENABLED", raising=False)

    assert run_inference._live_prescore_market_mask_enabled({}, "live") is False
    assert (
        run_inference._live_prescore_market_mask_enabled(
            {"live_prescore_market_mask_enabled": True}, "live"
        )
        is True
    )


def test_pre_score_market_mask_accepts_fresh_liquid_candidate():
    idx = pd.date_range("2026-06-10 08:00", periods=1, freq="h", tz="UTC")
    symbol = "ETH/USD:USD"
    panel = {
        "close": pd.DataFrame({symbol: [100.0]}, index=idx),
        "volume": pd.DataFrame({symbol: [10.0]}, index=idx),
        "mark_close": pd.DataFrame({symbol: [100.0]}, index=idx),
        "open_interest": pd.DataFrame({symbol: [1000000.0]}, index=idx),
    }

    snap = run_inference._pre_score_market_mask_snapshot(
        panel=panel,
        symbol=symbol,
        side="long",
        strategy_id="long_mr",
        executor=_PrescoreExecutor(_PrescoreExchange(bid=100.0, ask=100.1)),
        policy=run_inference.PortfolioPolicyConfig(max_spread_bps=25.0),
        runtime_config={
            "mode": "live-test",
            "market_mode": "perps",
            "live_prescore_orderbook_enabled": True,
            "live_prescore_liquidity_probe_quote_notional": 50.0,
        },
        now=pd.Timestamp("2026-06-10 09:05:00Z"),
        signal_bar_ts=idx[-1],
        raw_close_reference_gap_bps=150.0,
        max_signal_close_to_entry_seconds=900.0,
    )

    assert snap["prescore_market_mask_allowed"] is True
    assert snap["prescore_market_mask_reason"] == ""
    assert snap["prescore_oi_value"] == pytest.approx(1000000.0)


def test_pre_score_market_mask_parallel_preserves_candidate_order():
    idx = pd.date_range("2026-06-10 08:00", periods=1, freq="h", tz="UTC")
    symbols = ["A/USD:USD", "B/USD:USD", "C/USD:USD"]
    panel = {
        "close": pd.DataFrame({symbol: [100.0] for symbol in symbols}, index=idx),
        "volume": pd.DataFrame({symbol: [10.0] for symbol in symbols}, index=idx),
        "mark_close": pd.DataFrame({symbol: [100.0] for symbol in symbols}, index=idx),
        "open_interest": pd.DataFrame(
            {symbol: [1000000.0] for symbol in symbols}, index=idx
        ),
    }
    side_metrics = {
        "prescore_market_mask_input": 0,
        "prescore_market_mask_pass": 0,
        "prescore_market_mask_block": 0,
        "prescore_market_mask_reasons": {},
        "non_fatal_issues": 0,
    }

    kept, snapshots = run_inference._apply_pre_score_market_masks(
        panel=panel,
        candidates=symbols,
        side="long",
        strategy_id="long_mr",
        executor=_PrescoreExecutor(_PrescoreExchange(bid=100.0, ask=100.1)),
        policy=run_inference.PortfolioPolicyConfig(max_spread_bps=25.0),
        runtime_config={
            "mode": "live-test",
            "market_mode": "perps",
            "live_prescore_orderbook_enabled": False,
            "live_prescore_market_mask_workers": 2,
        },
        now=pd.Timestamp("2026-06-10 09:05:00Z"),
        signal_bar_ts=idx[-1],
        raw_close_reference_gap_bps=150.0,
        max_signal_close_to_entry_seconds=900.0,
        side_metrics=side_metrics,
    )

    assert kept == symbols
    assert list(snapshots) == symbols
    assert side_metrics["prescore_market_mask_input"] == 3
    assert side_metrics["prescore_market_mask_pass"] == 3
    assert side_metrics["prescore_market_mask_block"] == 0


def test_trade_result_merge_derives_entry_notional_quote_for_fee_audit():
    features_log = {
        "symbol": "PUMP/USD:USD",
        "side": "long",
        "position_size_after_liquidity": 7.0,
    }
    trade_result = {
        "base_amount": 4900.0,
        "realized_entry_price": 0.001483,
        "entry_fee_estimate_quote": 0.0035,
        "entry_fee_estimate_bps": 5.0,
    }

    merged = run_inference._merge_trade_result_entry_log_fields(
        features_log, trade_result
    )

    assert merged["entry_notional_quote"] == pytest.approx(7.0)
    assert merged["entry_fee_estimate_quote"] == pytest.approx(0.0035)


def test_live_warmup_state_health_requires_panel_span_or_current_cache(tmp_path):
    idx = pd.date_range("2026-06-10 08:00", periods=6, freq="h", tz="UTC")
    symbol = "ETH/USD:USD"
    panel = {"close": pd.DataFrame({symbol: [100.0] * len(idx)}, index=idx)}
    cfg = {
        "live_raw_rolling_state_enabled": True,
        "live_raw_rolling_state_path": str(tmp_path / "raw_rolling_state.npz"),
        "live_causal_transform_state_enabled": True,
        "live_causal_transform_state_path": str(
            tmp_path / "causal_transform_state.npz"
        ),
        "live_feature_snapshot_cache_dir": str(tmp_path / "feature_cache"),
    }

    health = run_inference._live_warmup_state_health_snapshot(
        panel=panel,
        symbols=[symbol],
        lookback_hours=6,
        required_model_warmup_hours=24 * 45,
        latest_closed_hour=idx[-1],
        feature_runtime_cfg=cfg,
        config={"mode": "live-test", "live_min_panel_warmup_hours": 24},
    )

    assert health["ok"] is False
    assert health["reason"] == "insufficient_panel_warmup"


def test_live_warmup_state_health_accepts_current_panel_and_state(tmp_path):
    idx = pd.date_range("2026-06-01 00:00", periods=24 * 35, freq="h", tz="UTC")
    symbol = "ETH/USD:USD"
    panel = {"close": pd.DataFrame({symbol: [100.0] * len(idx)}, index=idx)}
    raw_state = tmp_path / "raw_rolling_state.npz"
    causal_state = tmp_path / "causal_transform_state.npz"
    raw_state.write_bytes(b"state")
    causal_state.write_bytes(b"state")
    cache_dir = tmp_path / "feature_cache" / "abc"
    cache_dir.mkdir(parents=True)
    (cache_dir / "rolling_meta.json").write_text(
        json.dumps(
            {
                "end_ts": pd.Timestamp(idx[-1]).isoformat(),
                "rows": 1,
                "features": ["x"],
            }
        )
    )
    cfg = {
        "live_raw_rolling_state_enabled": True,
        "live_raw_rolling_state_path": str(raw_state),
        "live_causal_transform_state_enabled": True,
        "live_causal_transform_state_path": str(causal_state),
        "live_feature_snapshot_cache_dir": str(tmp_path / "feature_cache"),
    }

    health = run_inference._live_warmup_state_health_snapshot(
        panel=panel,
        symbols=[symbol],
        lookback_hours=24 * 35,
        required_model_warmup_hours=24 * 45,
        latest_closed_hour=idx[-1],
        feature_runtime_cfg=cfg,
        config={"mode": "live-test", "live_min_panel_warmup_hours": 24 * 32},
    )

    assert health["ok"] is True
    assert health["panel_ok"] is True
    assert health["raw_rolling_state"]["ok"] is True


def test_live_warmup_state_health_accepts_hashed_feature_state_inventory(tmp_path):
    idx = pd.date_range("2026-06-01 00:00", periods=24 * 35, freq="h", tz="UTC")
    symbol = "ETH/USD:USD"
    panel = {"close": pd.DataFrame({symbol: [100.0] * len(idx)}, index=idx)}
    raw_state_root = tmp_path / "raw_rolling_state.npz"
    causal_state_root = tmp_path / "causal_transform_state.npz"
    (tmp_path / "raw_rolling_state.abc123.npz").write_bytes(b"state")
    (tmp_path / "causal_transform_state.def456.npz").write_bytes(b"state")
    cfg = {
        "live_raw_rolling_state_enabled": True,
        "live_raw_rolling_state_path": str(raw_state_root),
        "live_causal_transform_state_enabled": True,
        "live_causal_transform_state_path": str(causal_state_root),
        "live_feature_snapshot_cache_dir": str(tmp_path / "feature_cache"),
    }

    health = run_inference._live_warmup_state_health_snapshot(
        panel=panel,
        symbols=[symbol],
        lookback_hours=24 * 35,
        required_model_warmup_hours=24 * 45,
        latest_closed_hour=idx[-1],
        feature_runtime_cfg=cfg,
        config={"mode": "live-test", "live_min_panel_warmup_hours": 24 * 32},
    )

    assert health["ok"] is True
    assert health["raw_rolling_state"]["ok"] is True
    assert health["raw_rolling_state"]["exact_exists"] is False
    assert health["raw_rolling_state"]["hashed_count"] == 1
    assert health["raw_rolling_state"]["reason"] == "ok_hashed_state_inventory"
    assert health["causal_transform_state"]["ok"] is True
    assert health["causal_transform_state"]["exact_exists"] is False
    assert health["causal_transform_state"]["hashed_count"] == 1


def test_live_warmup_state_health_accepts_raw_state_container(tmp_path):
    idx = pd.date_range("2026-06-01 00:00", periods=24 * 35, freq="h", tz="UTC")
    symbol = "ETH/USD:USD"
    panel = {"close": pd.DataFrame({symbol: [100.0] * len(idx)}, index=idx)}
    raw_state_root = tmp_path / "raw_rolling_state.npz"
    raw_state_container = tmp_path / "raw_rolling_state.container.sqlite"
    causal_state = tmp_path / "causal_transform_state.npz"
    raw_state_container.write_bytes(b"state")
    causal_state.write_bytes(b"state")
    cfg = {
        "live_raw_rolling_state_enabled": True,
        "live_raw_rolling_state_path": str(raw_state_root),
        "live_raw_rolling_state_container_enabled": True,
        "live_raw_rolling_state_container_path": str(raw_state_container),
        "live_causal_transform_state_enabled": True,
        "live_causal_transform_state_path": str(causal_state),
        "live_feature_snapshot_cache_dir": str(tmp_path / "feature_cache"),
    }

    health = run_inference._live_warmup_state_health_snapshot(
        panel=panel,
        symbols=[symbol],
        lookback_hours=24 * 35,
        required_model_warmup_hours=24 * 45,
        latest_closed_hour=idx[-1],
        feature_runtime_cfg=cfg,
        config={"mode": "live-test", "live_min_panel_warmup_hours": 24 * 32},
    )

    assert health["raw_rolling_state"]["ok"] is True
    assert health["raw_rolling_state"]["container_exists"] is True
    assert health["raw_rolling_state"]["reason"] == "ok_state_container"


def _simple_policy_params(**overrides):
    strategy_id = str(overrides.get("strategy_id", "long_mr"))
    base = Path("/tmp/ares_inference_policy_tests")
    source = (
        "artifacts/test-run/simple_policy_optimiser/deployment/best_policy_params.json"
    )
    artifact_path = base / source
    row = {
        "generated_by": SIMPLE_POLICY_GENERATOR,
        "schema": SIMPLE_POLICY_SCHEMA,
        "strategy_id": strategy_id,
        "enable_trailing": True,
        "barrier_pct": 0.02,
        "sl_mult": 1.0,
        "trailing_activation_mult": 1.0,
        "trailing_power": 1.5,
        "trailing_squash_divisor": 2.0,
        "giveback_beta": 0.5,
        "atr_power": 1.0,
        "atr_multiplier": 1.0,
        "hard_tp_abs_pct": 0.0,
        "capital_protect_mfe_mult": 1.0,
        "capital_protect_regression_frac": 0.45,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "generated_by": SIMPLE_POLICY_GENERATOR,
                "schema_version": SIMPLE_POLICY_SCHEMA,
                "strategies": [row],
            },
            sort_keys=True,
        )
    )
    params = load_simple_policy_stop_params_by_strategy(str(base), run_id="test-run")[
        strategy_id
    ]
    params.update(overrides)
    return params


def test_ev_adjustment_haircuts_only_excess_execution_costs():
    calibration = {
        "long_demo": [
            {"mean_score": 0.70, "mean_net_return": 0.010, "count": 100},
            {"mean_score": 0.80, "mean_net_return": 0.020, "count": 100},
            {"mean_score": 0.90, "mean_net_return": 0.030, "count": 100},
        ]
    }

    within_baseline = _ev_adjusted_prediction_after_entry_friction(
        calibrated_score=0.80,
        strategy_id="long_demo",
        side="long",
        calibration=calibration,
        live_entry_friction_bps=88.0,
        observed_spread_bps=100.0,
        orderbook_slippage_bps=15.0,
        adverse_signal_gap_bps=25.0,
        spread_baseline_bps=100.0,
        delay_slippage_baseline_bps=40.0,
        policy_rank_reference_store=None,
    )

    assert within_baseline["ev_haircut_raw_live_entry_friction_bps"] == 88.0
    assert within_baseline["ev_haircut_bps"] == 0.0
    assert within_baseline["ev_haircut_spread_excess_bps"] == 0.0
    assert within_baseline["ev_haircut_delay_slippage_excess_bps"] == 0.0
    assert within_baseline["ev_adjusted_net_return_after_friction"] == pytest.approx(
        within_baseline["ev_adjusted_net_return_before_friction"]
    )

    above_baseline = _ev_adjusted_prediction_after_entry_friction(
        calibrated_score=0.80,
        strategy_id="long_demo",
        side="long",
        calibration=calibration,
        live_entry_friction_bps=120.0,
        observed_spread_bps=120.0,
        orderbook_slippage_bps=25.0,
        adverse_signal_gap_bps=35.0,
        spread_baseline_bps=100.0,
        delay_slippage_baseline_bps=40.0,
        policy_rank_reference_store=None,
    )

    assert above_baseline["ev_haircut_spread_excess_bps"] == pytest.approx(10.0)
    assert above_baseline["ev_haircut_delay_slippage_excess_bps"] == pytest.approx(20.0)
    assert above_baseline["ev_haircut_bps"] == pytest.approx(30.0)
    assert above_baseline["ev_adjusted_net_return_after_friction"] == pytest.approx(
        above_baseline["ev_adjusted_net_return_before_friction"] - 0.003
    )


def test_ev_adjustment_records_stop_exit_reserve_without_double_counting():
    calibration = {
        "short_demo": [
            {"mean_score": 0.70, "mean_net_return": 0.010, "count": 100},
            {"mean_score": 0.80, "mean_net_return": 0.020, "count": 100},
            {"mean_score": 0.90, "mean_net_return": 0.030, "count": 100},
        ]
    }

    adjusted = _ev_adjusted_prediction_after_entry_friction(
        calibrated_score=0.80,
        strategy_id="short_demo",
        side="short",
        calibration=calibration,
        live_entry_friction_bps=0.0,
        observed_spread_bps=0.0,
        orderbook_slippage_bps=0.0,
        adverse_signal_gap_bps=0.0,
        spread_baseline_bps=100.0,
        delay_slippage_baseline_bps=40.0,
        expected_stop_exit_friction_bps=90.0,
        stop_exit_baseline_bps=15.0,
        stop_exit_friction_source="test.stop_exit",
        policy_rank_reference_store=None,
    )

    assert adjusted["ev_haircut_stop_exit_excess_bps"] == pytest.approx(75.0)
    assert adjusted["ev_haircut_bps"] == pytest.approx(0.0)
    assert adjusted["ev_haircut_stop_exit_source"] == "test.stop_exit"
    assert adjusted["ev_adjusted_net_return_after_friction"] == pytest.approx(
        adjusted["ev_adjusted_net_return_before_friction"]
    )


def test_live_ev_haircut_uses_symbol_average_spread_baseline(tmp_path, monkeypatch):
    baseline_path = (
        tmp_path
        / "exchanges"
        / "krakenfutures"
        / "spread_model"
        / "per_asset_spread_baseline_latest.csv"
    )
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text(
        "symbol,rows,average_spread_bps\nBTC/USD:USD,10,20.0\nNMR/USD:USD,10,60.0\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("EPM_SIMPLE_POLICY_SPREAD_BASELINE_PATH", raising=False)
    run_inference._LIVE_SPREAD_BASELINE_CACHE.clear()

    spread_baseline, source = run_inference._live_ev_haircut_spread_baseline_bps(
        symbol="BTC/USD:USD",
        data_root=str(tmp_path),
        fallback_bps=100.0,
    )
    adjusted = _ev_adjusted_prediction_after_entry_friction(
        calibrated_score=0.80,
        strategy_id="long_demo",
        side="long",
        calibration={
            "long_demo": [
                {"mean_score": 0.70, "mean_net_return": 0.010, "count": 100},
                {"mean_score": 0.80, "mean_net_return": 0.020, "count": 100},
                {"mean_score": 0.90, "mean_net_return": 0.030, "count": 100},
            ]
        },
        live_entry_friction_bps=15.0,
        observed_spread_bps=30.0,
        orderbook_slippage_bps=0.0,
        adverse_signal_gap_bps=0.0,
        spread_baseline_bps=spread_baseline,
        spread_baseline_source=source,
        delay_slippage_baseline_bps=0.0,
        policy_rank_reference_store=None,
    )

    assert spread_baseline == pytest.approx(20.0)
    assert "per_asset_spread_baseline.average_spread_bps" in source
    assert adjusted["ev_haircut_observed_half_spread_bps"] == pytest.approx(15.0)
    assert adjusted["ev_haircut_half_spread_baseline_bps"] == pytest.approx(10.0)
    assert adjusted["ev_haircut_spread_excess_bps"] == pytest.approx(5.0)
    assert adjusted["ev_haircut_bps"] == pytest.approx(5.0)
    assert adjusted["ev_haircut_spread_baseline_source"] == source


def test_live_stop_exit_friction_estimate_uses_spread_and_orderbook():
    class _Book:
        spread_bps = 12.0
        expected_fill_slippage_bps = 7.5

    policy = PortfolioPolicyConfig(
        ev_haircut_expected_stop_exit_bps=50.0,
        ev_haircut_stop_exit_spread_multiplier=0.5,
        ev_haircut_stop_exit_orderbook_multiplier=2.0,
    )

    estimate, source = run_inference._estimate_live_stop_exit_friction_bps(
        portfolio_policy=policy,
        ticker_snapshot={"spread_bps": 999.0},
        book_snapshot=_Book(),
    )

    assert estimate == pytest.approx(50.0 + 6.0 + 15.0)
    assert "base=50.0000" in source
    assert "spread_component=6.0000" in source
    assert "orderbook_component=15.0000" in source

    estimate, source = run_inference._estimate_live_stop_exit_friction_bps(
        portfolio_policy=policy,
        ticker_snapshot={"spread_bps": 20.0},
        book_snapshot=None,
    )

    assert estimate == pytest.approx(60.0)
    assert "spread_component=10.0000" in source


def test_simple_policy_stop_params_honor_policy_artifact_root_override(
    tmp_path, monkeypatch
):
    active = tmp_path / "artifacts" / "run_a" / "simple_policy_optimiser" / "deployment"
    active.mkdir(parents=True)
    active_row = {
        "strategy_id": "long_stale",
        "enable_trailing": True,
        "barrier_pct": 0.02,
        "sl_mult": 1.0,
        "trailing_activation_mult": 1.0,
        "trailing_power": 1.5,
        "trailing_squash_divisor": 2.0,
        "giveback_beta": 0.5,
        "capital_protect_mfe_mult": 1.0,
        "capital_protect_regression_frac": 0.45,
    }
    (active / "best_policy_params.json").write_text(
        json.dumps(
            {
                "generated_by": SIMPLE_POLICY_GENERATOR,
                "schema_version": SIMPLE_POLICY_SCHEMA,
                "strategies": [active_row],
            }
        )
    )
    override = tmp_path / "policy_override" / "simple_policy_optimiser" / "deployment"
    override.mkdir(parents=True)
    override_row = dict(active_row)
    override_row["strategy_id"] = "long_current"
    override_row["sl_mult"] = 1.25
    (override / "best_policy_params.json").write_text(
        json.dumps(
            {
                "generated_by": SIMPLE_POLICY_GENERATOR,
                "schema_version": SIMPLE_POLICY_SCHEMA,
                "strategies": [override_row],
            }
        )
    )

    monkeypatch.setenv("EPM_INFERENCE_POLICY_ARTIFACT_ROOT", str(override.parents[1]))
    params = load_simple_policy_stop_params_by_strategy(str(tmp_path), run_id="run_a")

    assert set(params) == {"long_current"}
    assert params["long_current"]["sl_mult"] == 1.25


def _policy_decision(
    params,
    *,
    should_replace=True,
    stop_price=99.0,
    reason="capital_preservation",
    reason_detail="capital_preservation: test",
    **overrides,
):
    kwargs = {
        "should_replace": should_replace,
        "stop_price": stop_price,
        "reason": reason,
        "reason_detail": reason_detail,
        "strategy_id": params["strategy_id"],
        "params_source": params["params_source"],
        "params_hash": params["params_hash"],
        "barrier_frac": params.get("barrier_frac") or params.get("barrier_pct"),
        "sl_mult": params["sl_mult"],
    }
    kwargs.update(overrides)
    return SimplePolicyStopDecision(**kwargs)


def test_trade_executor_fetch_margin_balance_uses_kraken_futures_flex_account():
    class _Exchange:
        id = "krakenfutures"

        def __init__(self):
            self.balance_params = None

        def fetch_balance(self, params=None):
            self.balance_params = params
            return {"info": {"accounts": {"flex": {}}}}

    exchange = _Exchange()
    executor = TradeExecutor(mode="shadow", exchange=exchange)

    assert executor._fetch_margin_balance() == {"info": {"accounts": {"flex": {}}}}
    assert exchange.balance_params == {"type": "flex"}


def test_synthesize_live_safe_timestamp_dayofweek_feature():
    idx = pd.date_range("2026-03-06", periods=4, freq="1d", tz="UTC")
    close = pd.DataFrame({"BTC/USDT": [1.0, 1.0, 1.0, 1.0]}, index=idx)

    feats = _synthesize_live_safe_feature_keys(
        {},
        {"close": close},
        ["BTC/USDT"],
        {"timestamp.dayofweek>=5"},
    )

    assert "timestamp.dayofweek>=5" in feats
    assert feats["timestamp.dayofweek>=5"]["BTC/USDT"].tolist() == [
        0.0,
        1.0,
        1.0,
        0.0,
    ]


def test_synthesize_live_safe_repairs_nan_path_efficiency_residual():
    idx = pd.date_range("2026-03-01", periods=520, freq="1h", tz="UTC")
    base = 100.0 + np.sin(np.arange(len(idx)) / 7.0).cumsum()
    close = pd.DataFrame(
        {
            "BTC/USD:USD": base,
            "ETH/USD:USD": base * 0.8 + np.cos(np.arange(len(idx)) / 5.0),
        },
        index=idx,
        dtype=np.float32,
    )
    stale = pd.DataFrame(
        np.nan,
        index=idx,
        columns=["BTC/USD:USD", "ETH/USD:USD"],
        dtype=np.float32,
    )

    feats = _synthesize_live_safe_feature_keys(
        {"path_efficiency_24_ts_resid": stale},
        {"close": close},
        ["BTC/USD:USD", "ETH/USD:USD"],
        {"path_efficiency_24_ts_resid"},
    )

    repaired = feats["path_efficiency_24_ts_resid"].tail(1)
    assert np.isfinite(repaired.to_numpy(dtype=np.float32)).all()


def test_live_adverse_policy_exit_shadow_uses_completed_policy_bar(monkeypatch):
    params = _simple_policy_params(strategy_id="short_test")

    class _Executor:
        mode = "live"
        exchange = object()

        def __init__(self):
            self.closed = None

        def get_simple_policy_stop_params(self, bucket_key):
            return params

        def close_position(self, symbol, price, reason):
            self.closed = {"symbol": symbol, "price": price, "reason": reason}
            return {"closed_trade": {"symbol": symbol, "exit_price": price}}

    def _decision(**kwargs):
        return SimplePolicyStopDecision(
            should_replace=False,
            stop_price=None,
            reason="adverse_excursion_exit",
            reason_detail="adverse_excursion_exit: test",
            strategy_id=params["strategy_id"],
            params_source=params["params_source"],
            params_hash=params["params_hash"],
            barrier_frac=params.get("barrier_frac") or params["barrier_pct"],
            sl_mult=params["sl_mult"],
            should_exit=True,
            exit_reason="adverse_excursion_exit",
            mfe=0.0,
            mae=0.03,
        )

    monkeypatch.setattr(
        run_inference, "_shadow_execution_realism_enabled", lambda: True
    )
    monkeypatch.setattr(
        run_inference,
        "_fetch_live_closeable_price",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("completed-bar policy must not fetch a ticker")
        ),
    )
    monkeypatch.setattr(
        run_inference,
        "compute_simple_policy_stop_decision",
        lambda **kwargs: _decision(**kwargs),
    )

    position_state = {
        "side": "short",
        "entry_price": 100.0,
        "bucket_key": "short_test",
        "strategy_id": "short_test",
        "stop_price": 110.0,
        "stop_reason": "original_stop_loss",
        "peak_price": 100.0,
        "mfe": 0.0,
        "mae": 0.0,
        "barrier_frac": 0.047,
        "barrier_pct": 0.047,
        "barrier_frac_is_effective": True,
        "sl_mult": 1.4,
    }
    bars = pd.DataFrame(
        {
            "open": [99.0],
            "high": [101.0],
            "low": [98.5],
            "close": [99.0],
        },
        index=pd.DatetimeIndex(["2026-06-09T21:10:00Z"]),
    )
    executor = _Executor()

    result = _evaluate_oco_policy("PNUT/USD:USD", position_state, bars, executor)

    assert result == {"closed_trade": {"symbol": "PNUT/USD:USD", "exit_price": 99.0}}
    assert executor.closed == {
        "symbol": "PNUT/USD:USD",
        "price": 99.0,
        "reason": "adverse_excursion_exit",
    }
    shadow = position_state["simple_policy_shadow"]
    assert shadow["shadow_exit_price"] == 99.0
    assert shadow["shadow_policy_bar_exit_price"] == 99.0
    assert shadow["shadow_exit_price_source"] == "trade_1m_close"
    assert shadow["shadow_exit_reason"] == "adverse_excursion_exit"
    assert shadow["barrier_frac"] == pytest.approx(0.047)
    assert shadow["barrier_frac_is_effective"] is True
    assert shadow["sl_mult"] == pytest.approx(1.4)


def test_cross_margin_dust_threshold_defaults_by_mode():
    assert _default_cross_margin_dust_quote_threshold("live-test") == 2.5
    assert _default_cross_margin_dust_quote_threshold("live_test") == 2.5
    assert _default_cross_margin_dust_quote_threshold("live") == 5.0
    assert _default_cross_margin_dust_quote_threshold("shadow") == 5.0


def test_trade_executor_shutdown_preserves_positions_by_default():
    class _OCO:
        def __init__(self):
            self.close_calls = 0

        def close_all_positions(self):
            self.close_calls += 1

    executor = TradeExecutor(mode="shadow", exchange=None)
    executor.oco_executor = _OCO()

    executor.shutdown()

    assert executor.oco_executor.close_calls == 0


def test_trade_executor_shutdown_can_flatten_when_explicitly_enabled():
    class _OCO:
        def __init__(self):
            self.close_calls = 0

        def close_all_positions(self):
            self.close_calls += 1

    executor = TradeExecutor(
        mode="shadow",
        exchange=None,
        config={"close_positions_on_shutdown": True},
    )
    executor.oco_executor = _OCO()

    executor.shutdown()

    assert executor.oco_executor.close_calls == 1


def test_select_candidates_uses_ret12h_move_and_vol_thresholds():
    idx = pd.date_range("2026-03-01", periods=13, freq="1h", tz="UTC")
    symbols = ["A", "B", "C", "D"]
    close = pd.DataFrame(
        {
            "A": [100] * 12 + [108],  # +8%
            "B": [100] * 12 + [106],  # +6%
            "C": [100] * 12 + [94],  # -6%
            "D": [100] * 12 + [92],  # -8%
        },
        index=idx,
    )
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {
        "ret12h": close / close.shift(12) - 1.0,
        "volatility_zscore": pd.DataFrame(
            {
                "A": [1.7] * len(idx),
                "B": [1.3] * len(idx),  # below threshold
                "C": [1.8] * len(idx),
                "D": [1.9] * len(idx),
            },
            index=idx,
        ),
        "chop_score": pd.DataFrame(0.1, index=idx, columns=symbols),
    }

    import extreme_price_movements.inference.candidate_selector as cs

    cs._resolve_runtime_cfg = lambda: {
        "candidate_mask_params_by_mode": {
            "price_up_tf": {
                "family": "abs_move_threshold",
                "param": 7.0,
                "z_hours": 1.0,
                "duration_hours": 1.0,
            },
            "price_up_mr": {
                "family": "abs_move_threshold",
                "param": 999.0,
                "z_hours": 1.0,
                "duration_hours": 1.0,
            },
            "price_down_tf": {
                "family": "abs_move_threshold",
                "param": 7.0,
                "z_hours": 1.0,
                "duration_hours": 1.0,
            },
            "price_down_mr": {
                "family": "abs_move_threshold",
                "param": 999.0,
                "z_hours": 1.0,
                "duration_hours": 1.0,
            },
        }
    }

    long_cands, short_cands = select_candidates(
        panel=panel,
        feats=feats,
        metric="ret12h",
    )

    assert long_cands == ["A"]
    assert short_cands == ["D"]


def test_latest_closed_candle_respects_publication_delay():
    now = pd.Timestamp("2026-05-10 17:00:03", tz="UTC")
    assert _latest_closed_candle_start(
        now,
        timeframe_minutes=60,
        delay_seconds=5.0,
    ) == pd.Timestamp("2026-05-10 15:00:00", tz="UTC")

    now = pd.Timestamp("2026-05-10 17:00:06", tz="UTC")
    assert _latest_closed_candle_start(
        now,
        timeframe_minutes=60,
        delay_seconds=5.0,
    ) == pd.Timestamp("2026-05-10 16:00:00", tz="UTC")


def test_policy_optimiser_stop_decision_uses_max_favorable_giveback():
    params = _simple_policy_params(
        barrier_frac=0.02,
        sl_mult=1.2,
        trailing_activation_mult=1.5,
        trailing_power=1.4,
        trailing_squash_divisor=1.5,
        giveback_beta=0.8,
        capital_protect_mfe_mult=0.0,
    )
    decision = compute_simple_policy_stop_decision(
        side="long",
        state={
            "side": "long",
            "entry_price": 100.0,
            "peak_price": 104.0,
            "mfe": 0.04,
            "mae": 0.0,
            "stop_price": 97.6,
            "strategy_id": "long_mr",
            "barrier_frac": 0.02,
            "sl_mult": 1.2,
        },
        latest_market_state={},
        policy_params=params,
    )
    max_favorable_abs = 4.0
    dynamic = min((max_favorable_abs / (2.0 * 1.5)) ** 1.4, 1.0)
    trail_amount = max(
        max_favorable_abs * 0.8 * (1.0 - dynamic),
        100.0 * MIN_TRAILING_GIVEBACK_FRAC,
    )
    expected = 100.0 + max_favorable_abs - trail_amount

    assert decision.reason == "trailing_profit"
    assert decision.stop_price == pytest.approx(expected)


def test_policy_optimiser_stop_decision_does_not_call_sub_fee_lock_profit():
    params = _simple_policy_params(
        barrier_frac=0.02,
        sl_mult=1.2,
        trailing_activation_mult=0.05,
        trailing_power=1.5,
        trailing_squash_divisor=3.5,
        giveback_beta=0.3,
        capital_protect_mfe_mult=0.0,
        round_trip_cost_pct=0.01,
        cost_pct_per_side=0.005,
    )
    decision = compute_simple_policy_stop_decision(
        side="long",
        state={
            "side": "long",
            "entry_price": 100.0,
            "peak_price": 100.4,
            "mfe": 0.004,
            "mae": 0.0,
            "stop_price": 97.6,
            "strategy_id": "long_mr",
            "barrier_frac": 0.02,
            "sl_mult": 1.2,
        },
        latest_market_state={},
        policy_params=params,
    )

    assert decision.reason == "trailing_risk_reduction"
    assert "profit_cost_floor=0.01" in decision.reason_detail
    assert "net_lock_ret=-" in decision.reason_detail
    assert decision.stop_price > 100.0


def test_policy_optimiser_stop_decision_caps_trailing_activation():
    base_state = {
        "side": "long",
        "entry_price": 100.0,
        "peak_price": 103.5,
        "mfe": 0.035,
        "mae": 0.0,
        "stop_price": 98.0,
        "strategy_id": "long_mr",
        "barrier_frac": 0.02,
        "sl_mult": 1.0,
    }
    uncapped = compute_simple_policy_stop_decision(
        side="long",
        state=base_state,
        latest_market_state={},
        policy_params=_simple_policy_params(
            barrier_frac=0.02,
            sl_mult=1.0,
            trailing_activation_mult=2.5,
            trailing_activation_cap_pct=0.0,
            capital_protect_mfe_mult=0.0,
        ),
    )
    capped = compute_simple_policy_stop_decision(
        side="long",
        state=base_state,
        latest_market_state={},
        policy_params=_simple_policy_params(
            barrier_frac=0.02,
            sl_mult=1.0,
            trailing_activation_mult=2.5,
            trailing_activation_cap_pct=0.03,
            capital_protect_mfe_mult=0.0,
        ),
    )

    assert uncapped.reason != "trailing_profit"
    assert uncapped.effective_trailing_activation_return == pytest.approx(0.05)
    assert capped.reason == "trailing_profit"
    assert capped.trailing_activation_cap_pct == pytest.approx(0.03)
    assert capped.effective_trailing_activation_return == pytest.approx(0.03)
    assert capped.stop_price > base_state["stop_price"]


def test_policy_optimiser_stop_decision_applies_exit_pressure_tightening():
    params = _simple_policy_params(
        barrier_frac=0.02,
        sl_mult=1.0,
        trailing_activation_mult=10.0,
        hard_tp_abs_pct=0.0,
        capital_protect_mfe_mult=0.0,
        exit_pressure_enabled=True,
        exit_pressure_alpha=1.0,
        exit_pressure_beta=1.0,
        exit_pressure_delta=1.0,
        exit_pressure_kappa=1.0,
        exit_pressure_min_multiplier=0.25,
        target_holding_hours=0.25,
    )
    decision = compute_simple_policy_stop_decision(
        side="long",
        state={
            "side": "long",
            "entry_price": 100.0,
            "peak_price": 100.0,
            "mfe": 0.0,
            "mae": 0.0,
            "bars_in_trade": 4,
            "stop_price": 98.0,
            "strategy_id": "long_mr",
            "barrier_frac": 0.02,
        },
        latest_market_state={},
        policy_params=params,
    )

    assert decision.reason == "exit_pressure_stop_tightening"
    assert decision.stop_price > 98.0
    assert decision.exit_pressure > 0.0
    assert decision.tightening_multiplier < 1.0


def test_policy_optimiser_stop_decision_does_not_label_disabled_pressure():
    params = _simple_policy_params(
        barrier_frac=0.02,
        sl_mult=1.8,
        trailing_activation_mult=10.0,
        hard_tp_abs_pct=0.0,
        capital_protect_mfe_mult=0.0,
        exit_pressure_enabled=False,
        exit_pressure_beta=0.0,
        exit_pressure_kappa=0.0,
        exit_pressure_min_multiplier=1.0,
    )
    decision = compute_simple_policy_stop_decision(
        side="long",
        state={
            "side": "long",
            "entry_price": 100.0,
            "peak_price": 100.0,
            "mfe": 0.0,
            "mae": 0.0,
            "bars_in_trade": 1,
            "stop_price": 90.0,
            "strategy_id": "long_mr",
            "barrier_frac": 0.02,
        },
        latest_market_state={},
        policy_params=params,
    )

    assert decision.reason == "policy_stop_loss"
    assert decision.stop_price == pytest.approx(96.4)
    assert decision.exit_pressure == pytest.approx(0.0)
    assert decision.tightening_multiplier == pytest.approx(1.0)


def test_capital_protection_arms_before_it_can_tighten_stop():
    params = _simple_policy_params(
        barrier_frac=0.02,
        sl_mult=1.0,
        trailing_activation_mult=10.0,
        capital_protect_mfe_mult=1.0,
        capital_protect_lock_frac=0.5,
        capital_protect_min_lock_bps=0.0,
        capital_protect_spread_lock_mult=0.0,
    )
    state = {
        "side": "long",
        "entry_price": 100.0,
        "peak_price": 103.0,
        "mfe": 0.03,
        "mae": 0.01,
        "stop_price": 98.0,
        "strategy_id": "long_mr",
        "barrier_frac": 0.02,
    }

    armed = compute_simple_policy_stop_decision(
        side="long",
        state=state,
        latest_market_state={},
        policy_params=params,
    )

    assert armed.reason == "capital_preservation_armed"
    assert armed.capital_protect_armed
    assert armed.capital_protect_armed_now
    assert not armed.should_replace

    active = compute_simple_policy_stop_decision(
        side="long",
        state={**state, "capital_protect_armed": True},
        latest_market_state={},
        policy_params=params,
    )

    assert active.reason == "capital_preservation"
    assert active.capital_protect_armed
    assert not active.capital_protect_armed_now
    assert active.stop_price == pytest.approx(101.0)


def test_capital_protection_requires_ten_minutes_and_current_confirmation():
    params = _simple_policy_params(
        barrier_frac=0.02,
        sl_mult=1.0,
        trailing_activation_mult=10.0,
        capital_protect_mfe_mult=1.0,
        capital_protect_lock_frac=0.5,
        capital_protect_min_lock_bps=0.0,
        capital_protect_spread_lock_mult=0.0,
    )
    base_state = {
        "side": "long",
        "entry_price": 100.0,
        "peak_price": 103.0,
        "mfe": 0.03,
        "mae": 0.0,
        "stop_price": 98.0,
        "strategy_id": "long_mr",
        "barrier_frac": 0.02,
    }

    crossed = compute_simple_policy_stop_decision(
        side="long",
        state={
            **base_state,
            "capital_protect_current_price": 104.0,
            "capital_protect_observation_ts": "2026-07-17T10:00:00Z",
        },
        latest_market_state={},
        policy_params=params,
    )
    assert crossed.capital_protect_pending
    assert not crossed.capital_protect_armed
    assert not crossed.should_replace
    assert crossed.capital_protect_crossed_ts == "2026-07-17T10:00:00+00:00"

    below_before_confirmation = compute_simple_policy_stop_decision(
        side="long",
        state={
            **base_state,
            "capital_protect_crossed_ts": crossed.capital_protect_crossed_ts,
            "capital_protect_current_price": 100.0,
            "capital_protect_observation_ts": "2026-07-17T10:09:00Z",
        },
        latest_market_state={},
        policy_params=params,
    )
    assert below_before_confirmation.capital_protect_pending
    assert (
        below_before_confirmation.capital_protect_crossed_ts
        == crossed.capital_protect_crossed_ts
    )
    assert not below_before_confirmation.capital_protect_armed

    confirmed = compute_simple_policy_stop_decision(
        side="long",
        state={
            **base_state,
            "capital_protect_crossed_ts": crossed.capital_protect_crossed_ts,
            "capital_protect_current_price": 104.2,
            "capital_protect_observation_ts": "2026-07-17T10:10:00Z",
        },
        latest_market_state={},
        policy_params=params,
    )
    assert confirmed.capital_protect_armed
    assert confirmed.capital_protect_armed_now
    assert not confirmed.capital_protect_pending
    assert confirmed.reason == "capital_preservation"
    assert confirmed.stop_price == pytest.approx(101.0)


def test_completed_bar_observation_does_not_change_capital_timer():
    params = _simple_policy_params(
        barrier_frac=0.02,
        sl_mult=1.0,
        trailing_activation_mult=10.0,
        capital_protect_mfe_mult=1.0,
        capital_protect_lock_frac=0.5,
        capital_protect_min_lock_bps=0.0,
        capital_protect_spread_lock_mult=0.0,
    )
    decision = compute_simple_policy_stop_decision(
        side="long",
        state={
            "side": "long",
            "entry_price": 100.0,
            "peak_price": 103.0,
            "mfe": 0.03,
            "mae": 0.0,
            "stop_price": 98.0,
            "strategy_id": "long_mr",
            "barrier_frac": 0.02,
            "capital_protect_crossed_ts": "2026-07-17T10:00:00+00:00",
            "capital_protection_disabled_for_observation": True,
        },
        latest_market_state=pd.DataFrame(
            {"close": [103.0]},
            index=[pd.Timestamp("2026-07-17T10:15:00Z")],
        ),
        policy_params=params,
    )
    assert not decision.capital_protection_observation_enabled
    assert decision.capital_protect_crossed_ts == "2026-07-17T10:00:00+00:00"
    assert not decision.capital_protect_armed


def test_select_candidates_rejects_legacy_threshold_overrides():
    idx = pd.date_range("2026-03-01", periods=2, freq="1h", tz="UTC")
    close = pd.DataFrame({"A": [100, 101], "B": [100, 99]}, index=idx)
    panel = {
        "close": close,
        "high": close,
        "low": close,
        "open": close,
        "volume": close,
    }
    feats = {"ret12h": close.pct_change().fillna(0.0)}

    with pytest.raises(ValueError, match="Legacy threshold overrides"):
        select_candidates(
            panel=panel,
            feats=feats,
            extreme_pct=0.25,
            metric="ret12h",
        )


def test_lgbm_strategy_masks_align_latest_symbol_vectors_with_panel_features():
    idx = pd.date_range("2026-03-01", periods=4, freq="1h", tz="UTC")
    symbols = ["A", "B", "C"]
    close = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 103.0],
            "B": [100.0, 99.0, 98.0, 97.0],
            "C": [100.0, 100.0, 100.0, 100.0],
        },
        index=idx,
    )
    panel = {
        "close": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "open": close,
        "volume": pd.DataFrame(100.0, index=idx, columns=symbols),
    }
    feats = {
        "ret12h": close.pct_change().fillna(0.0),
        "latest_symbol_score": pd.Series({"A": 0.9, "B": 0.1, "C": 0.8}, dtype=float),
        "panel_score": pd.DataFrame(
            {
                "A": [0.1, 0.2, 0.3, 0.9],
                "B": [0.1, 0.2, 0.3, 0.9],
                "C": [0.1, 0.2, 0.3, 0.2],
            },
            index=idx,
        ),
    }
    strategies = [
        {
            "strategy_id": "long_test",
            "trade_side": "long",
            "base_event_trigger": "(*)|(latest_symbol_score>0.5&panel_score>0.5)|(*)",
        }
    ]

    masks = build_strategy_candidate_masks(panel, feats, strategies)

    assert masks["long_test"] == ["A"]


def test_live_candidate_selection_preserves_authoritative_pre_model_lgbm_masks(
    monkeypatch, tmp_path
):
    idx = pd.date_range("2026-03-01", periods=2, freq="1h", tz="UTC")
    symbols = ["A", "B"]
    close = pd.DataFrame(
        {"A": [100.0, 101.0], "B": [100.0, 99.0]},
        index=idx,
    )
    panel = {
        "close": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "open": close,
        "volume": pd.DataFrame(100.0, index=idx, columns=symbols),
    }
    mask_rows = {
        "long_test": {
            "strategy_id": "long_test",
            "trade_side": "long",
            "base_event_trigger": "(*)|(mask_signal>0.5)|(*)",
            "mask_params": {"canonical_key": "(*)|(mask_signal>0.5)|(*)"},
        }
    }

    def fake_load_or_compute_features(*, cfg, **kwargs):
        namespace = str((cfg or {}).get("live_feature_cache_namespace") or "")
        if namespace == "mask":
            return {
                "mask_signal": pd.DataFrame(
                    {"A": [1.0], "B": [0.0]},
                    index=pd.DatetimeIndex([idx[-1]]),
                )
            }
        if namespace == "model":
            return {
                "mask_signal": pd.DataFrame(
                    {"A": [np.nan], "B": [np.nan]},
                    index=pd.DatetimeIndex([idx[-1]]),
                )
            }
        return {}

    monkeypatch.setattr(
        run_inference,
        "load_or_compute_features",
        fake_load_or_compute_features,
    )

    _, long_cands, short_cands, _, strategy_masks = (
        run_inference._select_candidates_and_load_features(
            panel=panel,
            symbols=symbols,
            run_id="test-run",
            data_root=str(tmp_path),
            cfg={"live_feature_cycle_cache_enabled": False},
            lookback_hours=2,
            required_feature_keys=set(),
            lgbm_strategy_mask_rows=mask_rows,
            feature_context_symbols=symbols,
            strategy_feature_contracts=None,
            model_features_required=True,
        )
    )

    assert long_cands == ["A"]
    assert short_cands == []
    assert strategy_masks["long_test"] == ["A"]


def test_active_strategy_gap_guard_contract_uses_only_nonempty_masks():
    masks = {
        "long_active": ["A"],
        "short_inactive": [],
        "long_no_symbols": [""],
    }
    contracts = {
        "long_active": ["ret24h", "base_probability_long_active"],
        "short_inactive": ["ret1h", "z_r_12"],
    }

    active = run_inference._active_strategy_required_feature_keys_from_masks(
        masks,
        contracts,
    )

    assert "ret24h" in active
    assert "base_probability_long_active" in active
    assert "ret1h" not in active
    assert "z_r_12" not in active


def test_select_candidates_falls_back_when_optimized_masks_are_silent():
    idx = pd.date_range("2026-03-01", periods=13, freq="1h", tz="UTC")
    symbols = ["A", "B", "C", "D"]
    close = pd.DataFrame(
        {
            "A": [100] * 12 + [109],
            "B": [100] * 12 + [101],
            "C": [100] * 12 + [99],
            "D": [100] * 12 + [91],
        },
        index=idx,
    )
    panel = {
        "close": close,
        "high": close * 1.04,
        "low": close * 0.96,
        "open": close,
        "volume": close,
    }
    feats = {
        "ret12h": close / close.shift(12) - 1.0,
        "range_12h_pct": pd.DataFrame(0.08, index=idx, columns=symbols),
        "volatility_zscore": pd.DataFrame(1.7, index=idx, columns=symbols),
    }

    import extreme_price_movements.inference.candidate_selector as cs

    cs._resolve_runtime_cfg = lambda: {
        "train_extreme_pct_hourly": 0.25,
        "train_min_move_12h_pct": 0.06,
        "train_min_range_pct": 0.06,
        "train_min_vol_zscore": 1.5,
        "candidate_mask_empty_fallback_enabled": True,
        "candidate_mask_params_by_mode": {
            mode: {
                "family": "abs_move_threshold",
                "param": 999.0,
                "z_hours": 1.0,
                "duration_hours": 1.0,
            }
            for mode in (
                "price_up_tf",
                "price_up_mr",
                "price_down_tf",
                "price_down_mr",
            )
        },
    }

    long_cands, short_cands = select_candidates(
        panel=panel,
        feats=feats,
        metric="ret12h",
    )

    assert long_cands == ["A"]
    assert short_cands == ["D"]


def test_candidate_trade_mask_respects_side_specific_extremes():
    idx = pd.date_range("2026-03-01", periods=2, freq="1h", tz="UTC")
    ret12h = pd.DataFrame(
        {
            "A": [0.08, 0.07],
            "B": [0.05, 0.01],
            "C": [-0.07, -0.06],
            "D": [-0.02, -0.08],
        },
        index=idx,
    )
    vol_z = pd.DataFrame(1.7, index=idx, columns=ret12h.columns)
    trades = pd.DataFrame(
        {
            "entry_ts": [idx[0], idx[0], idx[1], idx[1]],
            "symbol": ["A", "C", "B", "D"],
            "side": ["long", "short", "long", "short"],
        }
    )
    mask = _select_candidate_trade_mask(
        trades,
        ret12h,
        vol_z,
        pct=0.25,
        min_move_12h_pct=0.05,
        min_vol_zscore=1.5,
    )
    assert mask.tolist() == [True, True, False, True]


def test_5m_exit_takes_priority_over_threshold_update():
    executor = TradeExecutor(
        mode="shadow",
        exchange=None,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params(barrier_pct=0.01)
            }
        },
    )
    rec = executor.execute_trade(
        "BTC/USDT", "long", 0.5, price=100.0, bucket_key="long_mr"
    )
    assert rec["status"] == "recorded"
    pos = executor.get_active_positions()["BTC/USDT"]
    assert pos["stop_price"] < 100.0

    bars = pd.DataFrame(
        {
            "open": [100.0],
            "high": [105.0],  # would improve trailing stop
            "low": [98.5],  # breaches the current stop first
            "close": [104.0],
        },
        index=pd.date_range("2026-03-01 01:00", periods=1, freq="5min", tz="UTC"),
    )
    _evaluate_oco_policy("BTC/USDT", pos, bars, executor)
    assert "BTC/USDT" not in executor.get_active_positions()


def test_restart_reconstructs_elapsed_policy_bars_before_fast_exit(monkeypatch):
    executor = TradeExecutor(
        mode="shadow",
        exchange=None,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params(barrier_pct=0.01)
            }
        },
    )
    state = {
        "side": "long",
        "entry_price": 100.0,
        "realized_entry_price": 100.0,
        "entry_time": pd.Timestamp("2026-03-01 00:00", tz="UTC"),
        "stop_price": 90.0,
        "peak_price": 100.0,
        "mfe": 0.0,
        "mae": 0.0,
        "bars_in_trade": 0,
        "policy_bar_minutes": 1,
        "bucket_key": "long_mr",
        "strategy_id": "long_mr",
    }
    observed = {}

    def _capture_decision(*, state, **kwargs):
        observed["bars_in_trade"] = state["bars_in_trade"]
        return SimpleNamespace(
            should_exit=False,
            should_replace=False,
            stop_price=None,
            reason="original_stop_loss",
            reason_detail="unchanged_original_stop_loss",
            peak_price=100.0,
            mfe=0.0,
            mae=0.0,
            capital_protect_armed=False,
            capital_protect_armed_now=False,
            capital_protect_crossed_ts=None,
            capital_protect_pending=False,
            capital_protect_activation_return=None,
        )

    monkeypatch.setattr(
        "extreme_price_movements.inference.run_inference.compute_simple_policy_stop_decision",
        _capture_decision,
    )
    bars = pd.DataFrame(
        {"open": [99.0], "high": [99.0], "low": [99.0], "close": [99.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-03-01 02:00", tz="UTC")]),
    )

    try:
        _evaluate_oco_policy("BTC/USDT", state, bars, executor)
    finally:
        executor.shutdown()

    assert observed["bars_in_trade"] == 120
    assert state["bars_in_trade_reconstructed_from_entry"] is True


def test_position_policy_entry_price_prefers_realized_fill():
    entry_price, source = _position_policy_entry_price(
        {
            "theoretical_entry_price": 100.0,
            "policy_entry_price": 100.2,
            "realized_entry_price": 101.5,
        }
    )

    assert entry_price == pytest.approx(101.5)
    assert source == "realized_entry_price"


def test_shadow_executor_exposes_monitorable_open_positions():
    executor = TradeExecutor(
        mode="shadow",
        exchange=None,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
    )

    rec = executor.execute_trade(
        "BTC/USDT", "long", 0.5, price=100.0, bucket_key="long_mr"
    )
    positions = executor.get_oco_positions()
    statuses = executor.monitor_orders_once()

    assert rec["status"] == "recorded"
    assert "BTC/USDT" in positions
    assert statuses["BTC/USDT"]["status"] == "open"
    assert statuses["BTC/USDT"]["mode"] == "shadow"
    assert statuses["BTC/USDT"]["stop_price"] < 100.0
    assert statuses["BTC/USDT"]["stop_order_id"].startswith("shadow-stop-")
    assert executor.get_active_positions()["BTC/USDT"]["last_order_status"] == "open"


def test_shadow_retry_missing_protective_stop_reattaches_synthetic_stop():
    executor = TradeExecutor(
        mode="shadow",
        exchange=None,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
    )

    rec = executor.execute_trade(
        "BTC/USDT", "long", 0.5, price=100.0, bucket_key="long_mr"
    )
    assert rec["status"] == "recorded"
    executor.positions["BTC/USDT"].pop("stop_order_id", None)
    retry = executor.retry_missing_protective_stop(
        "BTC/USDT", executor.positions["BTC/USDT"]
    )

    assert retry["success"] is True
    assert retry["mode"] == "shadow"
    assert retry["simulated"] is True
    assert retry["stop_order_id"].startswith("shadow-stop-")
    assert executor.positions["BTC/USDT"]["protective_stop_attached"] is True


def test_shadow_monitor_updates_stop_from_trailing_price_action():
    executor = TradeExecutor(
        mode="shadow",
        exchange=None,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
    )
    rec = executor.execute_trade(
        "BTC/USDT", "long", 0.5, price=100.0, bucket_key="long_mr"
    )
    assert rec["status"] == "recorded"
    initial_stop = executor.get_active_positions()["BTC/USDT"]["stop_price"]
    bars = pd.DataFrame(
        {
            "open": [100.0, 102.0],
            "high": [103.0, 106.0],
            "low": [100.0, 104.0],
            "close": [102.0, 105.0],
        },
        index=pd.date_range("2026-03-01 00:00", periods=2, freq="15min", tz="UTC"),
    )
    executor.update_position_policy_state(
        "BTC/USDT",
        last_5m_eval_ts=pd.Timestamp("2026-03-01 00:00", tz="UTC"),
    )
    executor.positions["BTC/USDT"]["entry_time"] = pd.Timestamp(
        "2026-03-01 00:00", tz="UTC"
    )
    executor.positions["BTC/USDT"]["ohlcv_5m_latest"] = bars

    statuses = _monitor_active_position_price_action(
        executor,
        exchange=None,
        now=pd.Timestamp("2026-03-01 00:30:06", tz="UTC"),
    )
    updated = executor.get_active_positions()["BTC/USDT"]

    assert updated["stop_price"] > initial_stop
    assert updated["peak_price"] == 106.0
    assert statuses["BTC/USDT"]["price_action"]["stop_price_before"] == initial_stop
    assert statuses["BTC/USDT"]["price_action"]["stop_price_after"] > initial_stop


def test_shadow_monitor_waits_for_closed_15m_price_action():
    executor = TradeExecutor(
        mode="shadow",
        exchange=None,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
    )
    rec = executor.execute_trade(
        "BTC/USDT", "long", 0.5, price=100.0, bucket_key="long_mr"
    )
    assert rec["status"] == "recorded"
    initial_stop = executor.get_active_positions()["BTC/USDT"]["stop_price"]
    incomplete_bars = pd.DataFrame(
        {
            "open": [100.0, 102.0],
            "high": [102.0, 106.0],
            "low": [100.0, 102.0],
            "close": [102.0, 105.0],
        },
        index=pd.date_range("2026-03-01 00:00", periods=2, freq="5min", tz="UTC"),
    )
    executor.positions["BTC/USDT"]["entry_time"] = pd.Timestamp(
        "2026-03-01 00:00", tz="UTC"
    )
    executor.positions["BTC/USDT"]["ohlcv_5m_latest"] = incomplete_bars

    _monitor_active_position_price_action(
        executor,
        exchange=None,
        now=pd.Timestamp("2026-03-01 00:10:06", tz="UTC"),
    )
    assert executor.get_active_positions()["BTC/USDT"]["stop_price"] == initial_stop

    complete_bars = pd.DataFrame(
        {
            "open": [100.0, 102.0, 105.0],
            "high": [102.0, 106.0, 107.0],
            "low": [100.0, 102.0, 104.0],
            "close": [102.0, 105.0, 106.0],
        },
        index=pd.date_range("2026-03-01 00:00", periods=3, freq="5min", tz="UTC"),
    )
    executor.positions["BTC/USDT"]["ohlcv_5m_latest"] = complete_bars
    statuses = _monitor_active_position_price_action(
        executor,
        exchange=None,
        now=pd.Timestamp("2026-03-01 00:15:06", tz="UTC"),
    )
    updated = executor.get_active_positions()["BTC/USDT"]

    assert updated["stop_price"] > initial_stop
    assert updated["last_15m_eval_ts"] == pd.Timestamp("2026-03-01 00:00", tz="UTC")
    assert statuses["BTC/USDT"]["price_action"]["bars_evaluated"] == 1


def test_shadow_monitor_keeps_updating_after_initial_eight_hour_window():
    executor = TradeExecutor(
        mode="shadow",
        exchange=None,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
    )
    rec = executor.execute_trade(
        "BTC/USDT", "long", 0.5, price=100.0, bucket_key="long_mr"
    )
    assert rec["status"] == "recorded"
    initial_stop = executor.get_active_positions()["BTC/USDT"]["stop_price"]
    bars = pd.DataFrame(
        {
            "open": [100.0, 105.0],
            "high": [106.0, 108.0],
            "low": [104.0, 106.0],
            "close": [105.0, 107.0],
        },
        index=pd.date_range("2026-03-01 12:00", periods=2, freq="15min", tz="UTC"),
    )
    executor.positions["BTC/USDT"]["entry_time"] = pd.Timestamp(
        "2026-03-01 00:00", tz="UTC"
    )
    executor.positions["BTC/USDT"]["last_5m_eval_ts"] = pd.Timestamp(
        "2026-03-01 11:45", tz="UTC"
    )
    executor.positions["BTC/USDT"]["ohlcv_5m_latest"] = bars

    statuses = _monitor_active_position_price_action(
        executor,
        exchange=None,
        now=pd.Timestamp("2026-03-01 12:30:06", tz="UTC"),
    )
    updated = executor.get_active_positions()["BTC/USDT"]

    assert updated["stop_price"] > initial_stop
    assert updated["peak_price"] == 108.0
    assert statuses["BTC/USDT"]["price_action"]["bars_evaluated"] == 2


def test_live_executor_places_stop_loss_only_not_oco_or_take_profit(monkeypatch):
    class _Exchange:
        def __init__(self):
            self.orders = []
            self.oco_calls = 0

        def fetch_ohlcv(self, symbol, timeframe="1h", limit=14):
            return [[i, 100, 101, 99, 100, 1] for i in range(14)]

        def fetch_ticker(self, symbol):
            return {"last": 100.0}

        def create_oco_order(self, *args, **kwargs):
            self.oco_calls += 1
            raise AssertionError("OCO must not be used")

        def create_order(self, **kwargs):
            self.orders.append(kwargs)
            return {"id": f"order-{len(self.orders)}"}

    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    exchange = _Exchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
    finally:
        executor.shutdown()

    assert result["success"]
    order_types = [order["type"] for order in exchange.orders]
    assert "STOP_LOSS" in order_types
    assert "market" in order_types  # entry order and emergency cleanup in test
    assert order_types.count("market") >= 1
    assert exchange.oco_calls == 0


class _FilterAwareExchange:
    def __init__(self, *, active=True, min_cost=10.0, cancel_raises=False):
        self.orders = []
        self.canceled = []
        self.oco_calls = 0
        self.cancel_raises = cancel_raises
        self.markets = {
            "BTC/USDT": {
                "active": active,
                "limits": {
                    "amount": {"min": 0.001, "max": 1000.0},
                    "cost": {"min": min_cost, "max": 1_000_000.0},
                },
                "info": {"status": "TRADING" if active else "BREAK"},
            }
        }

    def load_markets(self):
        return self.markets

    def market(self, symbol):
        return self.markets[symbol]

    def amount_to_precision(self, symbol, amount):
        return f"{float(amount):.6f}"

    def price_to_precision(self, symbol, price):
        return f"{float(price):.2f}"

    def fetch_ohlcv(self, symbol, timeframe="1h", limit=14):
        return [[i, 100.0, 101.0, 99.0, 100.0, 1.0] for i in range(14)]

    def fetch_ticker(self, symbol):
        return {"last": 100.0}

    def fetch_trades(self, symbol, since=None, limit=None):
        return []

    def create_oco_order(self, *args, **kwargs):
        self.oco_calls += 1
        raise AssertionError("OCO must not be used")

    def create_order(self, **kwargs):
        order = dict(kwargs)
        order["id"] = f"order-{len(self.orders) + 1}"
        if kwargs["type"] in {"limit", "market"}:
            order["amount"] = kwargs["amount"]
            order["filled"] = kwargs["amount"]
            order["average"] = kwargs.get("price", 100.0)
        self.orders.append(order)
        return order

    def cancel_order(self, order_id, symbol, params=None):
        if self.cancel_raises:
            raise RuntimeError("cancel rejected by exchange")
        self.canceled.append((order_id, symbol, params or {}))
        return {"id": order_id, "status": "canceled"}

    def fetch_order(self, order_id, symbol, params=None):
        for order in self.orders:
            if order.get("id") == order_id:
                return {**order, "status": order.get("status", "open")}
        raise RuntimeError("unknown order")


def test_exchange_error_classifier_covers_binance_failure_modes():
    cases = {
        "Account has insufficient balance": "insufficient_balance",
        "Filter failure: LOT_SIZE precision invalid": "invalid_precision_or_filter",
        "symbol halted or inactive: BTC/USDT": "symbol_halted",
        "Order rejected by exchange": "order_rejected",
        "ORDER_WOULD_IMMEDIATELY_TRIGGER": "trigger_price_rejected",
        "OrderImmediatelyFillable": "trigger_price_rejected",
        "STRATEGY_INVALID_TRIGGER_PRICE": "trigger_price_rejected",
        "CONDITIONAL_ORDER_TRIGGER_REJECT": "trigger_price_rejected",
        "krakenfutures fetchOrder() is not supported yet": "unsupported_exchange_method",
        "network timeout while sending order": "network_timeout",
        "cancel rejected by exchange": "cancel_failed",
        "Duplicate clientOrderId was sent": "duplicate_client_order_id",
        "krakenfutures: createOrder failed due to wouldNotReducePosition": "position_already_reduced",
    }
    for message, expected in cases.items():
        assert _classify_exchange_error(RuntimeError(message)) == expected


def test_kraken_futures_stop_orders_trigger_on_executable_side_price():
    class _KrakenFuturesExchange:
        id = "krakenfutures"

        def __init__(self):
            self.created_orders = []

        def create_order(self, **kwargs):
            self.created_orders.append(dict(kwargs))
            return {"id": "stop-1", **kwargs}

    exchange = _KrakenFuturesExchange()
    _create_reduce_stop_loss_order(
        exchange,
        symbol="XPL/USD:USD",
        side="buy",
        amount=79.0,
        stop_price=0.0804,
        config={"execution_account": "perps"},
    )
    _create_reduce_stop_loss_order(
        exchange,
        symbol="XPL/USD:USD",
        side="sell",
        amount=79.0,
        stop_price=0.0794,
        config={"execution_account": "perps"},
    )

    buy_stop, sell_stop = exchange.created_orders
    assert buy_stop["type"] == "market"
    assert buy_stop["params"]["reduceOnly"] is True
    assert buy_stop["params"]["triggerSignal"] == "last"
    assert buy_stop["params"]["stopLossPrice"] == pytest.approx(0.0804)
    assert sell_stop["params"]["triggerSignal"] == "last"
    assert sell_stop["params"]["stopLossPrice"] == pytest.approx(0.0794)


def test_kraken_futures_native_sendorder_uses_last_trigger():
    class _NativeKrakenFuturesExchange:
        id = "krakenfutures"

        def __init__(self):
            self.markets = {"XPL/USD:USD": {"id": "PF_XPLUSD", "active": True}}
            self.native_orders = []

        def load_markets(self):
            return self.markets

        def market(self, symbol):
            return self.markets[symbol]

        def amount_to_precision(self, symbol, amount):
            return f"{float(amount):.1f}"

        def price_to_precision(self, symbol, price):
            return f"{float(price):.4f}"

        def privatePostSendorder(self, payload):
            self.native_orders.append(dict(payload))
            return {"sendStatus": {"order_id": "native-stop-1"}, "result": "success"}

        def create_order(self, **kwargs):
            raise AssertionError("native Kraken Futures stop path should be used")

    exchange = _NativeKrakenFuturesExchange()

    order = _create_reduce_stop_loss_order(
        exchange,
        symbol="XPL/USD:USD",
        side="buy",
        amount=79.0,
        stop_price=0.0804,
        config={"execution_account": "perps"},
    )

    assert order["id"] == "native-stop-1"
    assert order["triggerSignal"] == "last"
    assert exchange.native_orders == [
        {
            "symbol": "PF_XPLUSD",
            "side": "buy",
            "size": "79.0",
            "orderType": "stp",
            "stopPrice": "0.0804",
            "reduceOnly": True,
            "triggerSignal": "last",
        }
    ]


def test_kraken_futures_last_stop_adjusts_from_executable_side_spread():
    short_stop, short_meta = _kraken_futures_last_stop_from_executable_stop(
        {"bid": 0.0418, "ask": 0.0420, "last": 0.0419},
        {},
        position_side="short",
        policy_stop_price=0.04267,
    )
    long_stop, long_meta = _kraken_futures_last_stop_from_executable_stop(
        {"bid": 0.0998, "ask": 0.1002, "last": 0.1000},
        {},
        position_side="long",
        policy_stop_price=0.0975,
    )

    assert short_stop == pytest.approx(0.04257)
    assert short_meta["gap_source"] == "ask_minus_last"
    assert long_stop == pytest.approx(0.0977)
    assert long_meta["gap_source"] == "last_minus_bid"


def test_kraken_futures_stop_trigger_reference_uses_executable_side_not_last():
    class _KrakenFuturesExchange:
        id = "krakenfutures"

    price, source = _stop_trigger_reference_price(
        _KrakenFuturesExchange(),
        {
            "last": 0.0810,
            "close": 0.0809,
            "bid": 0.0808,
            "ask": 0.0812,
            "markPrice": 0.0803,
            "info": {"markPrice": "0.0802"},
        },
        {"execution_account": "perps"},
        position_side="long",
    )

    assert price == pytest.approx(0.0808)
    assert source == "bid"

    price, source = _stop_trigger_reference_price(
        _KrakenFuturesExchange(),
        {
            "last": 0.0810,
            "close": 0.0809,
            "bid": 0.0808,
            "ask": 0.0812,
            "markPrice": 0.0803,
            "info": {"markPrice": "0.0802"},
        },
        {"execution_account": "perps"},
        position_side="short",
    )

    assert price == pytest.approx(0.0812)
    assert source == "ask"


def test_live_policy_does_not_close_from_ticker_when_policy_bar_does_not_breach(
    monkeypatch,
):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    class _ExecutableBreachExchange(_FilterAwareExchange):
        id = "krakenfutures"

        def __init__(self):
            super().__init__()
            self.markets["PNUT/USD:USD"] = {
                "id": "PF_PNUTUSD",
                "active": True,
                "limits": {
                    "amount": {"min": 1.0, "max": 1_000_000.0},
                    "cost": {"min": 1.0, "max": 1_000_000.0},
                },
                "contract": True,
                "swap": True,
                "quote": "USD",
                "settle": "USD",
                "info": {"status": "TRADING"},
            }

        def fetch_ticker(self, symbol):
            return {"last": 99.0, "bid": 98.8, "ask": 100.2}

    exchange = _ExecutableBreachExchange()
    executor = TradeExecutor(
        mode="live-test",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "short_mr": _simple_policy_params(strategy_id="short_mr")
            }
        },
        config={
            "execution_account": "perps",
            "market_mode": "perps",
            "live_quote_currency": "USD",
            "monitor_interval_seconds": 300,
        },
    )
    executor.positions["PNUT/USD:USD"] = {
        "side": "short",
        "entry_price": 99.0,
        "realized_entry_price": 99.0,
        "size": 100.0,
        "bucket_key": "short_mr",
        "strategy_id": "short_mr",
        "stop_price": 100.0,
        "stop_reason": "original_stop_loss",
        "stop_order_id": "stop-1",
        "entry_time": pd.Timestamp("2026-03-01 00:00", tz="UTC"),
        "peak_price": 99.0,
        "mfe": 0.0,
        "mae": 0.0,
    }
    bars = pd.DataFrame(
        {
            "open": [99.0],
            "high": [99.5],
            "low": [98.5],
            "close": [99.0],
        },
        index=pd.date_range("2026-03-01 00:05", periods=1, freq="5min", tz="UTC"),
    )

    result = _evaluate_oco_policy(
        "PNUT/USD:USD", executor.positions["PNUT/USD:USD"], bars, executor
    )

    assert result is None
    assert "PNUT/USD:USD" in executor.positions


def test_kraken_software_stop_breach_cancels_hosted_stop_before_market_close():
    class _KrakenCloseFirstExchange(_FilterAwareExchange):
        id = "krakenfutures"

        def __init__(self):
            super().__init__()
            self.events = []

        def create_order(self, **kwargs):
            self.events.append(("create_order", kwargs["type"], kwargs["side"]))
            order = super().create_order(**kwargs)
            order["average"] = 0.0008637
            order["filled"] = kwargs["amount"]
            return order

        def cancel_order(self, order_id, symbol, params=None):
            self.events.append(("cancel_order", order_id, symbol))
            return super().cancel_order(order_id, symbol, params)

    exchange = _KrakenCloseFirstExchange()
    executor = OCOExecutor(
        exchange,
        {},
        config={
            "execution_account": "perps",
            "market_mode": "perps",
        },
    )
    state = {
        "side": "short",
        "entry_price": 0.0008211,
        "size": 8900.0,
        "bucket_key": "short_asset",
        "stop_price": 0.0008445,
        "final_placed_stop": 0.0008445,
        "requested_policy_stop": 0.0008445,
        "stop_reason": "original_stop_loss",
        "stop_order_id": "stop-order",
        "mfe": 0.0,
        "mae": 0.0,
    }
    executor.active_positions["TURBO/USD:USD"] = state

    executor._close_position(
        "TURBO/USD:USD",
        state,
        0.0008562,
        "software_executable_stop_breach:original_stop_loss",
    )

    assert exchange.events[0][0] == "cancel_order"
    assert exchange.events[1] == ("create_order", "market", "buy")
    assert "TURBO/USD:USD" not in executor.active_positions
    assert (
        state["trade_recap_events"][0]["event"]
        == "protective_stops_cancelled_before_close"
    )
    assert state["last_close_metrics"]["exit_price"] == pytest.approx(0.0008637)


def test_lightweight_stop_sentinel_closes_short_on_executable_ask():
    class _KrakenSentinelExchange(_FilterAwareExchange):
        id = "krakenfutures"

        def __init__(self):
            super().__init__()
            self.events = []

        def fetch_ticker(self, symbol):
            return {
                "bid": 0.0008320,
                "ask": 0.0008500,
                "last": 0.0008310,
                "timestamp": 1_780_000_000_000,
            }

        def fetch_order_book(self, symbol):
            return {
                "bids": [[0.0008315, 10_000.0]],
                "asks": [[0.0008562, 10_000.0]],
                "timestamp": 1_780_000_000_100,
            }

        def create_order(self, **kwargs):
            self.events.append(("create_order", kwargs["type"], kwargs["side"]))
            order = super().create_order(**kwargs)
            order["average"] = kwargs.get("price")
            order["filled"] = kwargs["amount"]
            return order

        def cancel_order(self, order_id, symbol, params=None):
            self.events.append(("cancel_order", order_id, symbol))
            return super().cancel_order(order_id, symbol, params)

    exchange = _KrakenSentinelExchange()
    executor = OCOExecutor(
        exchange,
        {},
        config={"execution_account": "perps", "market_mode": "perps"},
    )
    state = {
        "side": "short",
        "entry_price": 0.0008211,
        "size": 8900.0,
        "bucket_key": "short_asset",
        "stop_price": 0.0008445,
        "final_placed_stop": 0.0008445,
        "requested_policy_stop": 0.0008445,
        "stop_reason": "original_stop_loss",
        "stop_order_id": "stop-order",
        "mfe": 0.0,
        "mae": 0.0,
    }
    executor.active_positions["TURBO/USD:USD"] = state

    statuses = executor.monitor_executable_stops_once()

    status = statuses["TURBO/USD:USD"]
    assert status["status"] == "closed"
    assert status["reason"] == "software_executable_stop_breach:original_stop_loss"
    assert status["executable_price"] == pytest.approx(0.0008562)
    assert status["executable_price_source"] == "orderbook_best_ask"
    assert status["stop_breach_overshoot_bps"] > 0.0
    assert exchange.events[0][0] == "cancel_order"
    assert exchange.events[1] == ("create_order", "market", "buy")
    assert "TURBO/USD:USD" not in executor.active_positions

    closed_trade = status["closed_trade"]
    assert closed_trade["exit_price"] == pytest.approx(0.0008562)
    assert closed_trade["sentinel_executable_price"] == pytest.approx(0.0008562)
    assert closed_trade["sentinel_executable_price_source"] == "orderbook_best_ask"
    assert closed_trade["sentinel_stop_breach_overshoot_bps"] > 0.0
    assert closed_trade["ticker_bid"] == pytest.approx(0.0008320)
    assert "lightweight_stop_sentinel_sample" in closed_trade["trade_recap"]
    assert "lightweight_stop_sentinel_breach" in closed_trade["trade_recap"]


def test_lightweight_stop_sentinel_pretriggers_short_near_executable_ask():
    class _KrakenSentinelExchange(_FilterAwareExchange):
        id = "krakenfutures"

        def __init__(self):
            super().__init__()
            self.events = []

        def fetch_ticker(self, symbol):
            return {
                "bid": 99.90,
                "ask": 99.995,
                "last": 99.80,
                "timestamp": 1_780_000_000_000,
            }

        def fetch_order_book(self, symbol):
            return {
                "bids": [[99.90, 10_000.0]],
                "asks": [[99.995, 10_000.0]],
                "timestamp": 1_780_000_000_100,
            }

        def create_order(self, **kwargs):
            self.events.append(("create_order", kwargs["type"], kwargs["side"]))
            order = super().create_order(**kwargs)
            order["average"] = kwargs.get("price")
            order["filled"] = kwargs["amount"]
            return order

        def cancel_order(self, order_id, symbol, params=None):
            self.events.append(("cancel_order", order_id, symbol))
            return super().cancel_order(order_id, symbol, params)

    exchange = _KrakenSentinelExchange()
    executor = OCOExecutor(
        exchange,
        {},
        config={
            "execution_account": "perps",
            "market_mode": "perps",
            "lightweight_stop_sentinel_pretrigger_buffer_bps": 1.0,
        },
    )
    state = {
        "side": "short",
        "entry_price": 98.0,
        "size": 1.0,
        "bucket_key": "short_asset",
        "stop_price": 100.0,
        "final_placed_stop": 100.0,
        "requested_policy_stop": 100.0,
        "stop_reason": "original_stop_loss",
        "stop_order_id": "stop-order",
        "mfe": 0.0,
        "mae": 0.0,
    }
    executor.active_positions["ZEC/USD:USD"] = state

    statuses = executor.monitor_executable_stops_once()

    status = statuses["ZEC/USD:USD"]
    assert status["status"] == "closed"
    assert (
        status["reason"]
        == "software_executable_stop_breach_pretrigger:original_stop_loss"
    )
    assert status["pretriggered"] is True
    assert status["pretrigger_buffer_bps"] == pytest.approx(1.0)
    assert status["stop_distance_bps"] == pytest.approx(0.5000250012)
    assert status["stop_breach_overshoot_bps"] == pytest.approx(0.0)
    assert exchange.events[0][0] == "cancel_order"
    assert exchange.events[1] == ("create_order", "market", "buy")
    assert "ZEC/USD:USD" not in executor.active_positions

    closed_trade = status["closed_trade"]
    assert closed_trade["sentinel_pretriggered"] is True
    assert closed_trade["sentinel_pretrigger_buffer_bps"] == pytest.approx(1.0)
    assert closed_trade["sentinel_stop_distance_bps"] == pytest.approx(0.5000250012)
    assert closed_trade["sentinel_stop_breach_overshoot_bps"] == pytest.approx(0.0)
    assert "lightweight_stop_sentinel_sample" in closed_trade["trade_recap"]
    assert "lightweight_stop_sentinel_pretrigger" in closed_trade["trade_recap"]


def test_lightweight_stop_sentinel_uses_wider_profit_lock_pretrigger():
    class _KrakenSentinelExchange(_FilterAwareExchange):
        id = "krakenfutures"

        def __init__(self):
            super().__init__()
            self.events = []

        def fetch_ticker(self, symbol):
            return {
                "bid": 99.70,
                "ask": 99.80,
                "last": 99.70,
                "timestamp": 1_780_000_000_000,
            }

        def fetch_order_book(self, symbol):
            return {
                "bids": [[99.70, 10_000.0]],
                "asks": [[99.80, 10_000.0]],
                "timestamp": 1_780_000_000_100,
            }

        def create_order(self, **kwargs):
            self.events.append(("create_order", kwargs["type"], kwargs["side"]))
            order = super().create_order(**kwargs)
            order["average"] = kwargs.get("price")
            order["filled"] = kwargs["amount"]
            return order

        def cancel_order(self, order_id, symbol, params=None):
            self.events.append(("cancel_order", order_id, symbol))
            return super().cancel_order(order_id, symbol, params)

    exchange = _KrakenSentinelExchange()
    executor = OCOExecutor(
        exchange,
        {},
        config={
            "execution_account": "perps",
            "market_mode": "perps",
            "lightweight_stop_sentinel_pretrigger_buffer_bps": 1.0,
            "lightweight_stop_sentinel_profit_lock_pretrigger_buffer_bps": 25.0,
        },
    )
    state = {
        "side": "short",
        "entry_price": 100.50,
        "size": 1.0,
        "bucket_key": "short_asset",
        "stop_price": 100.0,
        "final_placed_stop": 100.0,
        "requested_policy_stop": 100.0,
        "stop_reason": "trailing_profit",
        "stop_order_id": "stop-order",
        "mfe": 0.01,
        "mae": 0.0,
    }
    executor.active_positions["AAVE/USD:USD"] = state

    statuses = executor.monitor_executable_stops_once()

    status = statuses["AAVE/USD:USD"]
    assert status["status"] == "closed"
    assert (
        status["reason"] == "software_executable_stop_breach_pretrigger:trailing_profit"
    )
    assert status["pretriggered"] is True
    assert status["pretrigger_buffer_bps"] == pytest.approx(25.0)
    assert status["stop_distance_bps"] == pytest.approx(20.0400801603)
    assert exchange.events[0][0] == "cancel_order"
    assert exchange.events[1] == ("create_order", "market", "buy")

    closed_trade = status["closed_trade"]
    assert closed_trade["sentinel_pretriggered"] is True
    assert closed_trade["sentinel_pretrigger_buffer_bps"] == pytest.approx(25.0)
    assert closed_trade["sentinel_executable_price_source"] == "orderbook_best_ask"


def test_lightweight_stop_sentinel_does_not_close_on_last_only_breach():
    class _KrakenSentinelExchange(_FilterAwareExchange):
        id = "krakenfutures"

        def fetch_ticker(self, symbol):
            return {
                "bid": 0.0008500,
                "ask": 0.0008700,
                "last": 0.0008300,
                "timestamp": 1_780_000_000_000,
            }

        def fetch_order_book(self, symbol):
            return {
                "bids": [[0.0008500, 10_000.0]],
                "asks": [[0.0008700, 10_000.0]],
                "timestamp": 1_780_000_000_100,
            }

    exchange = _KrakenSentinelExchange()
    executor = OCOExecutor(
        exchange,
        {},
        config={"execution_account": "perps", "market_mode": "perps"},
    )
    state = {
        "side": "long",
        "entry_price": 0.0008600,
        "size": 8900.0,
        "bucket_key": "long_asset",
        "stop_price": 0.0008445,
        "policy_stop_price": 0.0008445,
        "requested_policy_stop": 0.0008445,
        "exchange_stop_price": 0.0008525,
        "final_placed_stop": 0.0008525,
        "stop_reason": "original_stop_loss",
        "stop_order_id": "stop-order",
        "mfe": 0.0,
        "mae": 0.0,
    }
    executor.active_positions["TURBO/USD:USD"] = state

    statuses = executor.monitor_executable_stops_once()

    status = statuses["TURBO/USD:USD"]
    assert status["status"] == "open"
    assert status["executable_price"] == pytest.approx(0.0008500)
    assert status["executable_price_source"] == "orderbook_best_bid"
    assert status["stop_basis"] == "policy_executable_bid_ask"
    assert status["policy_executable_stop_price"] == pytest.approx(0.0008445)
    assert status["exchange_trigger_stop_price"] == pytest.approx(0.0008525)
    assert status["exchange_trigger_is_more_protective"] is True
    assert status["stop_distance_bps"] > 0.0
    assert "TURBO/USD:USD" in executor.active_positions


def test_monitor_active_position_does_not_use_ticker_stop_sentinel(
    monkeypatch,
):
    monkeypatch.setattr(
        "extreme_price_movements.inference.run_inference.hf_data_loader.fetch_specific_period",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("5m OHLCV should not be fetched after sentinel close")
        ),
    )

    class _KrakenSentinelExchange(_FilterAwareExchange):
        id = "krakenfutures"

        def fetch_ticker(self, symbol):
            return {
                "bid": 0.0008320,
                "ask": 0.0008500,
                "last": 0.0008310,
                "timestamp": 1_780_000_000_000,
            }

        def fetch_order_book(self, symbol):
            return {
                "bids": [[0.0008315, 10_000.0]],
                "asks": [[0.0008562, 10_000.0]],
                "timestamp": 1_780_000_000_100,
            }

        def fetch_order(self, order_id, symbol, params=None):
            return {
                "id": order_id,
                "status": "open",
                "side": "buy",
                "type": "stop_loss",
                "info": {"triggerSignal": "last", "stopPrice": "0.0008445"},
            }

        def create_order(self, **kwargs):
            order = super().create_order(**kwargs)
            order["average"] = kwargs.get("price")
            order["filled"] = kwargs["amount"]
            return order

    exchange = _KrakenSentinelExchange()
    executor = TradeExecutor(
        mode="live-test",
        exchange=exchange,
        bucket_params={},
        config={
            "execution_account": "perps",
            "market_mode": "perps",
            "lightweight_stop_sentinel_enabled": True,
        },
    )
    state = {
        "side": "short",
        "entry_price": 0.0008211,
        "size": 8900.0,
        "bucket_key": "short_asset",
        "strategy_id": "short_asset",
        "stop_price": 0.0008445,
        "final_placed_stop": 0.0008445,
        "requested_policy_stop": 0.0008445,
        "stop_reason": "original_stop_loss",
        "stop_order_id": "stop-order",
        "stop_order_ids": ["stop-order"],
        "entry_time": pd.Timestamp("2026-03-01 00:00", tz="UTC"),
        "mfe": 0.0,
        "mae": 0.0,
    }
    executor.positions["TURBO/USD:USD"] = state
    executor.oco_executor.active_positions["TURBO/USD:USD"] = state

    statuses = _monitor_active_position_price_action(
        executor,
        exchange=exchange,
        now=pd.Timestamp("2026-03-01 00:01", tz="UTC"),
        config=executor.config,
    )

    assert "executable_stop_sentinel" not in statuses["TURBO/USD:USD"]
    assert "price_action" not in statuses["TURBO/USD:USD"]
    assert executor.get_position("TURBO/USD:USD") is not None


def test_kraken_futures_existing_mark_stop_does_not_match_policy():
    class _KrakenFuturesExchange:
        id = "krakenfutures"

    mark_stop = {"info": {"triggerSignal": "mark", "stopPrice": "0.0804"}}
    last_stop = {"info": {"triggerSignal": "last", "stopPrice": "0.0804"}}
    bid_stop = {"info": {"triggerSignal": "bid", "stopPrice": "0.0804"}}
    ask_stop = {"info": {"triggerSignal": "ask", "stopPrice": "0.0804"}}

    cfg = {"execution_account": "perps"}
    assert (
        _protective_stop_trigger_matches_policy(
            _KrakenFuturesExchange(), mark_stop, cfg, position_side="long"
        )
        is False
    )
    assert (
        _protective_stop_trigger_matches_policy(
            _KrakenFuturesExchange(), last_stop, cfg, position_side="long"
        )
        is True
    )
    assert (
        _protective_stop_trigger_matches_policy(
            _KrakenFuturesExchange(), bid_stop, cfg, position_side="long"
        )
        is False
    )
    assert (
        _protective_stop_trigger_matches_policy(
            _KrakenFuturesExchange(), ask_stop, cfg, position_side="short"
        )
        is False
    )


def test_live_executor_converts_quote_notional_to_base_amount(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
    finally:
        executor.shutdown()

    assert result["success"]
    assert result["base_amount"] == pytest.approx(0.999)
    entry_order = exchange.orders[0]
    stop_order = exchange.orders[1]
    assert entry_order["amount"] == 1.0
    assert stop_order["amount"] == pytest.approx(0.999)
    assert stop_order["type"] == "STOP_LOSS"


def test_live_executor_preserves_margin_order_params(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={
            "execution_account": "margin",
            "margin_mode": "cross",
            "margin_side_effect_type": "AUTO_BORROW_REPAY",
        },
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
    finally:
        executor.shutdown()

    assert result["success"]
    entry_order = exchange.orders[0]
    stop_order = exchange.orders[1]
    assert entry_order["params"]["marginMode"] == "cross"
    assert entry_order["params"]["sideEffectType"] == "NO_SIDE_EFFECT"
    assert stop_order["params"]["marginMode"] == "cross"
    assert stop_order["params"]["sideEffectType"] == "AUTO_REPAY"


def test_get_bucket_params_filters_stop_policy_fields():
    executor = TradeExecutor(
        mode="shadow",
        exchange=None,
        bucket_params={
            "global_rank_threshold": 0.4,
            "long_mr": {
                "max_hold_hours": 12,
                "cooldown_hours": 3,
                "tp_mult": 9.0,
                "rank_threshold": 0.8,
            },
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            },
        },
    )

    params = executor.get_bucket_params("long_mr")

    assert params == {"max_hold_hours": 12, "cooldown_hours": 3}


def test_place_oco_order_signature_has_no_atr_frac():
    assert "atr_frac" not in inspect.signature(OCOExecutor.place_oco_order).parameters


def test_stop_cleanup_removed_legacy_helpers_from_oco_executor():
    for method_name in (
        "start_" + "monitoring",
        "stop_" + "monitoring",
        "_monitor_" + "loop",
        "monitor_" + "positions",
        "_update_" + "oco",
        "_widen_stop_" + "away_from_market",
        "_update_stop_" + "loss",
        "_replace_stop_order_" + "raw",
    ):
        assert not hasattr(OCOExecutor, method_name)


def test_live_executor_accepts_live_policy_barrier(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params(barrier_pct=None)
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT",
            "long",
            100.0,
            price=100.0,
            bucket_key="long_mr",
            trade_context={"barrier_pct": 0.50},
        )
    finally:
        executor.shutdown()

    assert result["success"]
    assert result["barrier_frac"] == pytest.approx(0.50)
    assert result["stop_price"] == pytest.approx(50.0)


def test_live_executor_rejects_exchange_filter_failures(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange(min_cost=500.0)
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
    finally:
        executor.shutdown()

    assert not result["success"]
    assert result["error_category"] == "invalid_precision_or_filter"
    assert exchange.orders == []


def test_live_executor_treats_subunit_size_as_quote_not_fraction(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange(min_cost=0.01)
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params(barrier_pct=0.50)
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT",
            "long",
            0.5,
            price=100.0,
            bucket_key="long_mr",
            trade_context={"barrier_pct": 0.50},
        )
    finally:
        executor.shutdown()

    assert result["success"]
    entry_orders = [order for order in exchange.orders if order["type"] == "market"]
    assert len(entry_orders) == 1
    assert entry_orders[0]["amount"] == pytest.approx(0.005)
    assert executor.get_active_positions()["BTC/USDT"]["quote_size"] == pytest.approx(
        0.5
    )


def test_live_executor_rejects_halted_symbols(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange(active=False)
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
    finally:
        executor.shutdown()

    assert not result["success"]
    assert result["error_category"] == "symbol_halted"
    assert exchange.orders == []


@pytest.mark.parametrize(
    ("message", "expected_category"),
    [
        ("Account has insufficient balance", "insufficient_balance"),
        ("Order rejected by exchange", "order_rejected"),
        ("network timeout while sending order", "network_timeout"),
        ("Duplicate clientOrderId was sent", "duplicate_client_order_id"),
    ],
)
def test_live_executor_classifies_entry_order_failures(
    monkeypatch, message, expected_category
):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    class _FailingEntryExchange(_FilterAwareExchange):
        def create_order(self, **kwargs):
            if kwargs["type"] in {"limit", "market"}:
                raise RuntimeError(message)
            return super().create_order(**kwargs)

    exchange = _FailingEntryExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
    finally:
        executor.shutdown()

    assert not result["success"]
    assert result["error_category"] == expected_category
    assert executor.get_active_positions() == {}


def test_live_executor_uses_partial_fill_amount_for_stop_loss(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    class _PartialFillExchange(_FilterAwareExchange):
        def create_order(self, **kwargs):
            order = dict(kwargs)
            order["id"] = f"order-{len(self.orders) + 1}"
            if kwargs["type"] in {"limit", "market"}:
                order["amount"] = kwargs["amount"]
                order["filled"] = kwargs["amount"] / 2.0
                order["average"] = kwargs.get("price", 100.0)
            self.orders.append(order)
            return order

    exchange = _PartialFillExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
    finally:
        executor.shutdown()

    assert result["success"]
    assert result["partial_fill"] is True
    assert result["base_amount"] == pytest.approx(0.4995)
    stop_orders = [order for order in exchange.orders if order["type"] == "STOP_LOSS"]
    assert stop_orders[-1]["amount"] == pytest.approx(0.4995)


def test_stop_loss_cancel_replace_uses_existing_base_amount(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
        assert result["success"]
        state = executor.oco_executor.active_positions["BTC/USDT"]
        params = executor.oco_executor.get_simple_policy_stop_params("long_mr")
        decision = _policy_decision(params, stop_price=99.0)
        executor.oco_executor._update_stop_loss_from_policy_decision(
            "BTC/USDT", state, decision
        )
    finally:
        executor.shutdown()

    stop_orders = [order for order in exchange.orders if order["type"] == "STOP_LOSS"]
    assert len(stop_orders) >= 2
    assert stop_orders[-1]["amount"] == pytest.approx(0.999)
    assert exchange.canceled[0][0] == "order-2"
    assert exchange.oco_calls == 0


def test_stop_loss_replace_rejects_looser_short_stop_at_boundary(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    params = _simple_policy_params(strategy_id="short_mr")
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={"simple_policy_stop_params_by_strategy": {"short_mr": params}},
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "short", 100.0, price=100.0, bucket_key="short_mr"
        )
        assert result["success"]
        state = executor.oco_executor.active_positions["BTC/USDT"]
        state["stop_price"] = 103.0
        state["stop_order_id"] = "order-2"
        state["stop_order_ids"] = ["order-2"]
        decision = _policy_decision(params, stop_price=105.0)

        executor.oco_executor._replace_stop_order_from_decision(
            "BTC/USDT", state, decision
        )
    finally:
        executor.shutdown()

    assert state["stop_price"] == pytest.approx(103.0)
    assert state["stop_order_id"] == "order-2"
    assert state["stop_update_error_category"] == "policy_stop_not_improved"
    assert exchange.canceled == []


def test_stop_loss_cancel_replace_does_not_duplicate_on_cancel_failure(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange(cancel_raises=True)
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
        assert result["success"]
        state = executor.oco_executor.active_positions["BTC/USDT"]
        params = executor.oco_executor.get_simple_policy_stop_params("long_mr")
        decision = _policy_decision(params, stop_price=99.0)
        executor.oco_executor._update_stop_loss_from_policy_decision(
            "BTC/USDT", state, decision
        )
        stop_update_error_category = state.get("stop_update_error_category")
    finally:
        executor.shutdown()

    stop_orders = [order for order in exchange.orders if order["type"] == "STOP_LOSS"]
    assert len(stop_orders) == 1
    assert stop_update_error_category == "cancel_failed"


def test_stop_loss_replacement_rejects_immediate_trigger_without_widening(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    class _TriggerRejectOnceExchange(_FilterAwareExchange):
        def __init__(self):
            super().__init__()
            self.rejected_replacement = False

        def create_order(self, **kwargs):
            if (
                kwargs["type"] == "STOP_LOSS"
                and any(order["type"] == "STOP_LOSS" for order in self.orders)
                and not self.rejected_replacement
            ):
                self.rejected_replacement = True
                raise RuntimeError("ORDER_WOULD_IMMEDIATELY_TRIGGER")
            return super().create_order(**kwargs)

    exchange = _TriggerRejectOnceExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={
            "monitor_interval_seconds": 300,
            "stop_replace_retry_backoff_seconds": 0.0,
            "stop_replace_retry_gap_growths": [0.0, 0.10, 0.20, 0.30],
        },
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
        assert result["success"]
        state = executor.oco_executor.active_positions["BTC/USDT"]
        params = executor.oco_executor.get_simple_policy_stop_params("long_mr")
        decision = _policy_decision(params, stop_price=99.95)
        executor.oco_executor._update_stop_loss_from_policy_decision(
            "BTC/USDT", state, decision
        )
        stop_update_error_category = state.get("stop_update_error_category")
        events = list(state.get("trade_recap_events", []))
    finally:
        executor.shutdown()

    stop_orders = [order for order in exchange.orders if order["type"] == "STOP_LOSS"]
    assert stop_update_error_category == "policy_stop_rejected_by_exchange"
    assert len(stop_orders) == 2
    assert exchange.rejected_replacement is True
    assert stop_orders[-1]["params"]["stopPrice"] == pytest.approx(98.0)
    assert exchange.canceled[0][0] == "order-2"
    assert any(
        event["event"] == "stop_replace_failed"
        and event.get("reject_reason") == "ORDER_WOULD_IMMEDIATELY_TRIGGER"
        for event in events
    )


def test_margin_executor_routes_entry_stop_cancel_and_close_params(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={
            "monitor_interval_seconds": 300,
            "execution_account": "margin",
            "margin_mode": "cross",
            "margin_side_effect_type": "AUTO_BORROW_REPAY",
        },
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
        assert result["success"]
        state = executor.oco_executor.active_positions["BTC/USDT"]
        params = executor.oco_executor.get_simple_policy_stop_params("long_mr")
        decision = _policy_decision(params, stop_price=99.0)
        executor.oco_executor._update_stop_loss_from_policy_decision(
            "BTC/USDT", state, decision
        )
        close_result = executor.close_position("BTC/USDT", reason="test_close")
    finally:
        executor.shutdown()

    assert close_result["success"]
    entry_order = exchange.orders[0]
    stop_orders = [order for order in exchange.orders if order["type"] == "STOP_LOSS"]
    market_closes = [
        order
        for order in exchange.orders
        if order["type"] == "market" and order["side"] == "sell"
    ]
    assert entry_order["params"]["marginMode"] == "cross"
    assert entry_order["params"]["sideEffectType"] == "NO_SIDE_EFFECT"
    assert stop_orders[0]["params"]["marginMode"] == "cross"
    assert stop_orders[0]["params"]["sideEffectType"] == "AUTO_REPAY"
    assert exchange.canceled[0][2]["marginMode"] == "cross"
    assert market_closes[-1]["params"]["sideEffectType"] == "AUTO_REPAY"


def test_monitor_orders_once_removes_filled_stop(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    class _FilledStopExchange(_FilterAwareExchange):
        def fetch_order(self, order_id, symbol, params=None):
            order = super().fetch_order(order_id, symbol, params=params)
            if order["type"] == "STOP_LOSS":
                order["status"] = "closed"
            return order

    exchange = _FilledStopExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
        assert result["success"]
        statuses = executor.monitor_orders_once()
    finally:
        executor.shutdown()

    assert statuses["BTC/USDT"]["status"] == "closed"
    assert "BTC/USDT" not in executor.get_active_positions()


def test_monitor_orders_once_classifies_fetch_order_timeout(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    class _TimeoutFetchOrderExchange(_FilterAwareExchange):
        def fetch_order(self, order_id, symbol, params=None):
            raise TimeoutError("network timeout while monitoring order")

    exchange = _TimeoutFetchOrderExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
        assert result["success"]
        statuses = executor.monitor_orders_once()
        active_after_monitor = executor.get_active_positions()
    finally:
        executor.shutdown()

    assert statuses["BTC/USDT"]["status"] == "error"
    assert statuses["BTC/USDT"]["error_category"] == "network_timeout"
    assert "BTC/USDT" in active_after_monitor


def test_monitor_orders_once_falls_back_to_open_orders_when_fetch_order_unsupported(
    monkeypatch,
):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    class _ListOnlyStopExchange(_FilterAwareExchange):
        def fetch_order(self, order_id, symbol, params=None):
            raise RuntimeError(
                "krakenfutures fetchOrder() is not supported yet, "
                "consider using fetchOpenOrders() and fetchClosedOrders() instead"
            )

        def fetch_open_orders(self, symbol, since=None, limit=None, params=None):
            return [
                {**order, "status": "open"}
                for order in self.orders
                if order.get("type") == "STOP_LOSS"
            ]

    exchange = _ListOnlyStopExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
        assert result["success"]
        statuses = executor.monitor_orders_once()
        active_after_monitor = executor.get_active_positions()
    finally:
        executor.shutdown()

    status = statuses["BTC/USDT"]
    assert status["status"] == "open"
    assert status["resolved_via"] == "fetch_open_orders"
    assert status["reconciled_after_error"] is True
    assert status["fetch_order_error_category"] == "unsupported_exchange_method"
    assert "BTC/USDT" in active_after_monitor


def test_restart_absent_entry_recovers_exact_stop_fill(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    class _PrivateFillExchange(_FilterAwareExchange):
        def fetch_my_trades(self, symbol, since=None, limit=None):
            return [
                {
                    "id": "fill-1",
                    "order": "stop-1",
                    "side": "buy",
                    "amount": 100.0,
                    "price": 0.1343,
                    "timestamp": int(
                        pd.Timestamp("2026-07-18T22:40:00Z").timestamp() * 1000
                    ),
                    "fee": {"cost": 0.01, "currency": "USD"},
                }
            ]

    executor = TradeExecutor(
        mode="live",
        exchange=_PrivateFillExchange(),
        bucket_params={},
        config={"execution_account": "perps"},
    )
    try:
        closed = executor.reconcile_absent_logged_entry(
            {
                "symbol": "FARTCOIN/USD:USD",
                "side": "short",
                "timestamp": "2026-07-18T09:07:49Z",
                "entry_time": "2026-07-18T09:07:49Z",
                "actual_entry_price": 0.1286,
                "realized_entry_price": 0.1286,
                "entry_price": 0.1286,
                "entry_notional_quote": 12.86,
                "stop_price": 0.1345,
                "stop_order_id": "stop-1",
                "position_id": "position-1",
                "strategy_id": "short_breakout",
            }
        )
    finally:
        executor.shutdown()

    assert closed["symbol"] == "FARTCOIN/USD:USD"
    assert closed["reason"].startswith("stop_loss_filled")
    assert closed["exit_price"] == pytest.approx(0.1343)
    assert closed["filled"] == pytest.approx(100.0)
    assert closed["reconciliation_mode"] == "startup_absent_position"
    assert (
        closed["reconciliation_fill_resolution"]
        == "private_fill_exact_stop_order"
    )


def test_perps_reconciliation_imports_existing_position_and_stop(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    class _PerpsRestartExchange(_FilterAwareExchange):
        id = "krakenfutures"

        def __init__(self):
            super().__init__()
            self.markets = {
                "NIGHT/USD:USD": {
                    "active": True,
                    "limits": {
                        "amount": {"min": 1.0, "max": 1_000_000.0},
                        "cost": {"min": 1.0, "max": 1_000_000.0},
                    },
                    "contract": True,
                    "swap": True,
                    "quote": "USD",
                    "settle": "USD",
                    "info": {"status": "TRADING"},
                }
            }

        def fetch_positions(self, symbols=None, params=None):
            return [
                {
                    "symbol": "NIGHT/USD:USD",
                    "contracts": 201.0,
                    "side": "short",
                    "entryPrice": 0.0318,
                    "contractSize": 1.0,
                }
            ]

        def price_to_precision(self, symbol, price):
            return f"{float(price):.5f}"

        def fetch_ticker(self, symbol):
            return {"bid": 0.03270, "ask": 0.03285, "last": 0.03280}

        def fetch_open_orders(self, symbol, since=None, limit=None, params=None):
            return [
                {
                    "id": "stop-1",
                    "symbol": symbol,
                    "type": "stop",
                    "side": "buy",
                    "amount": 100.0,
                    "status": "open",
                    "reduceOnly": True,
                    "info": {
                        "order_id": "stop-1",
                        "stopPrice": "0.03280",
                        "triggerSignal": "last",
                    },
                },
                {
                    "id": "stop-2",
                    "symbol": symbol,
                    "type": "stop",
                    "side": "buy",
                    "amount": 101.0,
                    "status": "open",
                    "reduceOnly": True,
                    "info": {
                        "order_id": "stop-2",
                        "stopPrice": "0.03280",
                        "triggerSignal": "last",
                    },
                },
            ]

    params = _simple_policy_params(strategy_id="short_mr")
    exchange = _PerpsRestartExchange()
    executor = TradeExecutor(
        mode="live-test",
        exchange=exchange,
        bucket_params={"simple_policy_stop_params_by_strategy": {"short_mr": params}},
        config={
            "execution_account": "perps",
            "market_mode": "perps",
            "live_quote_currency": "USD",
            "monitor_interval_seconds": 300,
        },
    )
    executor._load_pending_entry_context = lambda symbol: {
        "symbol": symbol,
        "status": "pending",
        "action": "enter",
        "strategy_id": "short_mr",
        "actual_entry_price": 0.0318,
        "stop_price": 0.03285,
        "barrier_frac": 0.03301886792452832,
        "barrier_pct": 0.03301886792452832,
        "sl_mult": 1.0,
        "stop_policy_params_source": params["params_source"],
        "stop_policy_params_hash": "retired-policy-hash",
        "stop_policy_schema": SIMPLE_POLICY_SCHEMA,
        "timestamp": "2026-05-17T21:28:04Z",
    }
    try:
        report = executor.reconcile_cross_margin_account()
        with executor.oco_executor._positions_lock:
            live_state = executor.oco_executor.active_positions["NIGHT/USD:USD"]
            live_state["bars_in_trade"] = 137
            live_state["last_policy_eval_ts"] = pd.Timestamp(
                "2026-05-17T23:45:00Z"
            )
            live_state["last_15m_eval_ts"] = pd.Timestamp(
                "2026-05-17T23:45:00Z"
            )
            live_state["policy_bar_minutes"] = 1
        executor.reconcile_cross_margin_account()
        statuses = executor.monitor_orders_once()
        active = executor.get_active_positions()
    finally:
        executor.shutdown()

    assert report["summary"]["skipped"] is False
    assert report["summary"]["active_positions_after_reconcile"] == 1
    assert report["items"][0]["classification"] == "external_perp_position"
    assert report["items"][0]["imported_for_monitoring"] is True
    assert active["NIGHT/USD:USD"]["stop_order_id"] == "stop-1"
    assert active["NIGHT/USD:USD"]["stop_order_ids"] == ["stop-1", "stop-2"]
    assert active["NIGHT/USD:USD"]["stop_order_coverage"] == pytest.approx(201.0)
    assert active["NIGHT/USD:USD"]["stop_price"] == pytest.approx(0.03285)
    assert active["NIGHT/USD:USD"]["policy_stop_price"] == pytest.approx(0.03285)
    assert active["NIGHT/USD:USD"]["exchange_stop_price"] == pytest.approx(0.03280)
    assert active["NIGHT/USD:USD"]["final_placed_stop"] == pytest.approx(0.03280)
    assert active["NIGHT/USD:USD"]["external_position"] is True
    assert active["NIGHT/USD:USD"]["bars_in_trade"] == 137
    assert active["NIGHT/USD:USD"]["last_policy_eval_ts"] == pd.Timestamp(
        "2026-05-17T23:45:00Z"
    )
    assert active["NIGHT/USD:USD"]["last_15m_eval_ts"] == pd.Timestamp(
        "2026-05-17T23:45:00Z"
    )
    assert active["NIGHT/USD:USD"]["policy_bar_minutes"] == 1
    assert active["NIGHT/USD:USD"]["stop_policy_params_hash"] == params["params_hash"]
    assert active["NIGHT/USD:USD"]["reconciliation_policy_version_migrated"] is True
    assert (
        active["NIGHT/USD:USD"]["reconciliation_previous_policy_hash"]
        == "retired-policy-hash"
    )
    assert active["NIGHT/USD:USD"]["stop_price"] == pytest.approx(0.03285)
    assert statuses["NIGHT/USD:USD"]["status"] == "open"
    assert statuses["NIGHT/USD:USD"]["stop_order_coverage"] == pytest.approx(201.0)


def test_perps_reconciliation_preserves_tighter_recovered_stop(monkeypatch):
    assert _stop_is_at_least_as_protective("short", 0.125, 0.131)
    assert not _stop_is_at_least_as_protective("short", 0.132, 0.131)
    assert _stop_is_at_least_as_protective("long", 105.0, 100.0)
    assert not _stop_is_at_least_as_protective("long", 95.0, 100.0)
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    class _TighterRecoveredStopExchange(_FilterAwareExchange):
        id = "krakenfutures"

        def __init__(self):
            super().__init__()
            self.markets = {
                "SYRUP/USD:USD": {
                    "active": True,
                    "limits": {
                        "amount": {"min": 1.0, "max": 1_000_000.0},
                        "cost": {"min": 1.0, "max": 1_000_000.0},
                    },
                    "contract": True,
                    "swap": True,
                    "quote": "USD",
                    "settle": "USD",
                    "info": {"status": "TRADING"},
                }
            }

        def fetch_positions(self, symbols=None, params=None):
            return [
                {
                    "symbol": "SYRUP/USD:USD",
                    "contracts": 100.0,
                    "side": "short",
                    "entryPrice": 0.12865,
                    "contractSize": 1.0,
                }
            ]

        def fetch_open_orders(self, symbol, since=None, limit=None, params=None):
            return [
                {
                    "id": "tight-stop",
                    "symbol": symbol,
                    "type": "stop",
                    "side": "buy",
                    "amount": 100.0,
                    "status": "open",
                    "reduceOnly": True,
                    "info": {
                        "order_id": "tight-stop",
                        "stopPrice": "0.12524",
                        "triggerSignal": "last",
                    },
                }
            ]

    params = _simple_policy_params(strategy_id="short_mr")
    exchange = _TighterRecoveredStopExchange()
    executor = TradeExecutor(
        mode="live-test",
        exchange=exchange,
        bucket_params={"simple_policy_stop_params_by_strategy": {"short_mr": params}},
        config={
            "execution_account": "perps",
            "market_mode": "perps",
            "live_quote_currency": "USD",
            "monitor_interval_seconds": 300,
        },
    )
    executor._load_pending_entry_context = lambda symbol: {
        "symbol": symbol,
        "status": "pending",
        "action": "enter",
        "strategy_id": "short_mr",
        "actual_entry_price": 0.12865,
        "stop_price": 0.1312,
        "barrier_frac": 0.02,
        "barrier_pct": 0.02,
        "sl_mult": 1.0,
        "stop_policy_params_source": params["params_source"],
        "stop_policy_params_hash": params["params_hash"],
        "stop_policy_schema": SIMPLE_POLICY_SCHEMA,
        "timestamp": "2026-06-09T16:25:00Z",
    }
    try:
        report = executor.reconcile_cross_margin_account()
        active = executor.get_active_positions()
    finally:
        executor.shutdown()

    assert report["summary"]["skipped"] is False
    assert report["summary"]["active_positions_after_reconcile"] == 1
    assert exchange.canceled == []
    state = active["SYRUP/USD:USD"]
    assert state["stop_order_id"] == "tight-stop"
    assert state["stop_order_ids"] == ["tight-stop"]
    assert state["stop_order_coverage"] == pytest.approx(100.0)
    assert state["stop_price"] == pytest.approx(0.1312)
    assert state["policy_stop_price"] == pytest.approx(0.1312)
    assert state["exchange_stop_price"] == pytest.approx(0.12524)
    assert state["final_placed_stop"] == pytest.approx(0.12524)


def test_pending_entry_context_recovers_archetype_geometry_from_shadow(tmp_path):
    trade_log = tmp_path / "inference_trades.csv"
    archetype_bucket = "short__policy_archetype_short__short_breakout_precision"
    pd.DataFrame(
        [
            {
                "timestamp": "2026-07-17T14:09:01Z",
                "lifecycle_event": "entry_placed",
                "status": "reconciled_absent",
                "action": "enter",
                "symbol": "LPT/USD:USD",
                "side": "short",
                "strategy_id": "s52_meta_threshold_handoff",
                "entry_price": 1.496,
                "stop_price": 1.526,
                "simple_policy_shadow": json.dumps(
                    {
                        "schema": "simple_policy_execution_shadow_v1",
                        "bucket_key": archetype_bucket,
                        "strategy_id": archetype_bucket,
                        "policy_entry_price": 1.496,
                        "realized_entry_price": 1.496,
                        "shadow_stop_price": 1.526,
                        "barrier_frac": 0.05180812498272756,
                        "barrier_pct": 0.05180812498272756,
                        "barrier_frac_is_effective": True,
                        "sl_mult": 1.2,
                        "params_source": "artifact/best_policy_params_perps.json",
                        "params_hash": "geometry-hash",
                        "params_schema": SIMPLE_POLICY_SCHEMA,
                    }
                ),
            }
        ]
    ).to_csv(trade_log, index=False)
    executor = TradeExecutor(
        mode="shadow",
        exchange=None,
        bucket_params={},
        config={"trade_log_path": str(trade_log)},
    )
    try:
        context = executor._load_pending_entry_context("LPT/USD:USD")
    finally:
        executor.shutdown()

    assert context["model_strategy_id"] == "s52_meta_threshold_handoff"
    assert context["strategy_id"] == archetype_bucket
    assert context["bucket_key"] == archetype_bucket
    assert context["stop_price"] == pytest.approx(1.526)
    assert context["policy_stop_price"] == pytest.approx(1.526)
    assert context["requested_policy_stop"] == pytest.approx(1.526)
    assert context["barrier_frac"] == pytest.approx(0.05180812498272756)
    assert context["barrier_pct"] == pytest.approx(0.05180812498272756)
    assert context["barrier_frac_is_effective"] is True
    assert context["sl_mult"] == pytest.approx(1.2)
    assert context["stop_policy_params_hash"] == "geometry-hash"
    assert context["stop_policy_schema"] == SIMPLE_POLICY_SCHEMA
    assert context["reconciliation_shadow_contract_recovered"] is True


def test_pending_entry_context_restores_entry_provenance_after_restart(tmp_path):
    trade_log = tmp_path / "trades.csv"
    pd.DataFrame(
        [
            {
                "timestamp": "2026-07-18T09:00:00Z",
                "lifecycle_event": "entry_placed",
                "status": "pending",
                "action": "enter",
                "symbol": "SOL/USD:USD",
                "entry_provenance_json": json.dumps(
                    {
                        "schema_version": "entry_provenance_v1",
                        "fields": {
                            "policy_archetype": "long__compression_release",
                            "archetype_hit_surprise_actual_hit_rate": 0.7,
                            "meta_sel_ood_abs_z_p95": 2.2,
                            "meta_lgbm_leaf_count_p10": 8,
                        },
                    }
                ),
            }
        ]
    ).to_csv(trade_log, index=False)
    executor = TradeExecutor(
        mode="shadow",
        exchange=None,
        bucket_params={},
        config={"trade_log_path": str(trade_log)},
    )
    try:
        context = executor._load_pending_entry_context("SOL/USD:USD")
    finally:
        executor.shutdown()

    assert context["policy_archetype"] == "long__compression_release"
    assert context["archetype_hit_surprise_actual_hit_rate"] == pytest.approx(0.7)
    assert context["meta_sel_ood_abs_z_p95"] == pytest.approx(2.2)
    assert context["meta_lgbm_leaf_count_p10"] == 8


def test_perps_reconciliation_imports_orphan_position_with_artifact_stop(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    class _OrphanPerpsExchange(_FilterAwareExchange):
        id = "krakenfutures"

        def __init__(self):
            super().__init__()
            self.markets = {
                "STBL/USD:USD": {
                    "active": True,
                    "limits": {
                        "amount": {"min": 1.0, "max": 1_000_000.0},
                        "cost": {"min": 1.0, "max": 1_000_000.0},
                    },
                    "contract": True,
                    "swap": True,
                    "quote": "USD",
                    "settle": "USD",
                    "info": {"status": "TRADING"},
                }
            }

        def fetch_positions(self, symbols=None, params=None):
            return [
                {
                    "symbol": "STBL/USD:USD",
                    "contracts": 207.0,
                    "side": "short",
                    "entryPrice": 0.03085,
                    "contractSize": 1.0,
                }
            ]

        def fetch_open_orders(self, symbol, since=None, limit=None, params=None):
            return []

    params = _simple_policy_params(strategy_id="short_mr")
    exchange = _OrphanPerpsExchange()
    executor = TradeExecutor(
        mode="live-test",
        exchange=exchange,
        bucket_params={"simple_policy_stop_params_by_strategy": {"short_mr": params}},
        config={
            "execution_account": "perps",
            "market_mode": "perps",
            "live_quote_currency": "USD",
            "monitor_interval_seconds": 300,
        },
    )
    executor._load_pending_entry_context = lambda symbol: {}
    try:
        report = executor.reconcile_cross_margin_account()
        active = executor.get_active_positions()
    finally:
        executor.shutdown()

    assert report["summary"]["skipped"] is False
    assert report["summary"]["active_positions_after_reconcile"] == 1
    assert report["items"][0]["imported_for_monitoring"] is True
    state = active["STBL/USD:USD"]
    assert state["external_position"] is True
    assert state["monitoring_only"] is True
    assert state["strategy_id"] == "short_mr"
    assert state["barrier_frac"] == pytest.approx(0.02)
    assert state["sl_mult"] == pytest.approx(1.0)
    assert state["stop_policy_params_hash"] == params["params_hash"]
    assert (
        state["reconciliation_barrier_source"] == "artifact_simple_policy_stop_params"
    )
    assert (
        state["reconciliation_context_source"] == "artifact_fallback_external_position"
    )
    assert state.get("recovered_from_pending_trade_log") is not True


def test_perps_reconciliation_preserves_authoritative_shadow_geometry(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    class _ShadowPerpsExchange(_FilterAwareExchange):
        id = "krakenfutures"

        def __init__(self):
            super().__init__()
            self.markets = {
                "ETHFI/USD:USD": {
                    "active": True,
                    "limits": {
                        "amount": {"min": 0.1, "max": 1_000_000.0},
                        "cost": {"min": 1.0, "max": 1_000_000.0},
                    },
                    "contract": True,
                    "swap": True,
                    "quote": "USD",
                    "settle": "USD",
                    "info": {"status": "TRADING"},
                }
            }

        def fetch_positions(self, symbols=None, params=None):
            return [
                {
                    "symbol": "ETHFI/USD:USD",
                    "contracts": 21.3,
                    "side": "long",
                    "entryPrice": 0.4416,
                    "contractSize": 1.0,
                }
            ]

        def fetch_open_orders(self, symbol, since=None, limit=None, params=None):
            return []

    shadow_strategy = "long__policy_archetype_long__long_breakout_diagnostic_candidate"
    shadow_barrier = 0.007825422769182454
    shadow_sl_mult = 2.7
    exchange = _ShadowPerpsExchange()
    executor = TradeExecutor(
        mode="live-test",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                shadow_strategy: _simple_policy_params(
                    strategy_id=shadow_strategy,
                    barrier_pct=0.03,
                    barrier_frac=0.03,
                    sl_mult=2.7,
                )
            }
        },
        config={
            "execution_account": "perps",
            "market_mode": "perps",
            "live_quote_currency": "USD",
            "monitor_interval_seconds": 300,
        },
    )
    executor._load_pending_entry_context = lambda symbol: {
        "symbol": symbol,
        "status": "pending",
        "action": "enter",
        "strategy_id": shadow_strategy,
        "actual_entry_price": 0.4416,
        "realized_entry_price": 0.4416,
        "policy_entry_price": 0.4416,
        "stop_price": 0.4323,
        "barrier_frac": shadow_barrier,
        "barrier_pct": shadow_barrier,
        "sl_mult": shadow_sl_mult,
        "stop_policy_params_source": "entry-shadow",
        "stop_policy_params_hash": "entry-shadow-hash",
        "stop_policy_schema": SIMPLE_POLICY_SCHEMA,
        "timestamp": "2026-07-17T19:08:00Z",
        "reconciliation_shadow_contract_recovered": True,
    }
    executor.resolve_simple_policy_strategy_id = (
        lambda strategy_id, side: "long_s52_meta_threshold_handoff"
    )
    executor._fetch_reconciled_entry_fill = lambda *args, **kwargs: {
        "checked": True,
        "matched": False,
    }
    try:
        report = executor.reconcile_cross_margin_account()
        state = executor.get_active_positions()["ETHFI/USD:USD"]
    finally:
        executor.shutdown()

    assert report["summary"]["active_positions_after_reconcile"] == 1
    assert state["strategy_id"] == shadow_strategy
    assert state["barrier_frac"] == pytest.approx(shadow_barrier)
    assert state["sl_mult"] == pytest.approx(shadow_sl_mult)
    expected_stop = 0.4416 * (1.0 - shadow_sl_mult * shadow_barrier)
    assert state["stop_price"] == pytest.approx(float(exchange.price_to_precision(
        "ETHFI/USD:USD", expected_stop
    )))
    assert state["reconciliation_strategy_fallback_used"] is False
    assert state["stop_policy_params_source"].endswith("best_policy_params.json")
    assert state["stop_policy_params_hash"] != "entry-shadow-hash"


def test_perps_reconciliation_existing_stop_does_not_override_artifact_barrier(
    monkeypatch,
):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    class _WideStopLongPerpsExchange(_FilterAwareExchange):
        id = "krakenfutures"

        def __init__(self):
            super().__init__()
            self.markets = {
                "TRX/USD:USD": {
                    "active": True,
                    "limits": {
                        "amount": {"min": 1.0, "max": 1_000_000.0},
                        "cost": {"min": 1.0, "max": 1_000_000.0},
                    },
                    "contract": True,
                    "swap": True,
                    "quote": "USD",
                    "settle": "USD",
                    "info": {"status": "TRADING"},
                }
            }

        def fetch_positions(self, symbols=None, params=None):
            return [
                {
                    "symbol": "TRX/USD:USD",
                    "contracts": 10.0,
                    "side": "long",
                    "entryPrice": 100.0,
                    "contractSize": 1.0,
                }
            ]

        def fetch_open_orders(self, symbol, since=None, limit=None, params=None):
            return [
                {
                    "id": "wide-stop",
                    "symbol": symbol,
                    "type": "stop",
                    "side": "sell",
                    "amount": 10.0,
                    "status": "open",
                    "reduceOnly": True,
                    "info": {
                        "order_id": "wide-stop",
                        "stopPrice": "80.0",
                        "triggerSignal": "last",
                    },
                }
            ]

        def fetch_ticker(self, symbol):
            return {"bid": 101.0, "ask": 101.1, "last": 101.05}

        def fetch_order_book(self, symbol):
            return {"bids": [[101.0, 10.0]], "asks": [[101.1, 10.0]]}

    params = _simple_policy_params(
        strategy_id="long_s52_meta_threshold_handoff",
        barrier_pct=0.02,
        barrier_frac=0.02,
        sl_mult=1.0,
        capital_protect_mfe_mult=3.0,
    )
    exchange = _WideStopLongPerpsExchange()
    executor = TradeExecutor(
        mode="live-test",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_s52_meta_threshold_handoff": params
            }
        },
        config={
            "execution_account": "perps",
            "market_mode": "perps",
            "live_quote_currency": "USD",
            "monitor_interval_seconds": 300,
        },
    )
    executor._load_pending_entry_context = lambda symbol: {
        "symbol": symbol,
        "status": "pending",
        "action": "enter",
        "strategy_id": "s52_meta_threshold_handoff",
        "actual_entry_price": 100.0,
        "stop_price": 80.0,
        "barrier_frac": 0.20,
        "barrier_pct": 0.20,
        "sl_mult": 1.0,
        "stop_policy_params_source": "artifacts/stale/policy_params/best_policy_params.json",
        "stop_policy_params_hash": "stalehash",
        "stop_policy_schema": SIMPLE_POLICY_SCHEMA,
        "timestamp": "2026-07-10T13:00:00Z",
    }
    try:
        report = executor.reconcile_cross_margin_account()
        active = executor.get_active_positions()
        state = active["TRX/USD:USD"]
        assert report["summary"]["active_positions_after_reconcile"] == 1
        assert exchange.canceled[0][0] == "wide-stop"
        assert state["stop_price"] == pytest.approx(98.0)
        assert state["policy_stop_price"] == pytest.approx(98.0)
        assert state["barrier_frac"] == pytest.approx(0.02)
        assert state["reconciliation_barrier_source"] == (
            "artifact_simple_policy_stop_params"
        )
        assert state["stop_policy_params_source"] == params["params_source"]
        assert state["stop_policy_params_hash"] == params["params_hash"]
        assert state["reconciliation_previous_barrier_frac"] == pytest.approx(0.20)
        if state.get("reconciliation_existing_stop_implied_barrier_frac") is not None:
            assert state["reconciliation_existing_stop_implied_barrier_frac"] == (
                pytest.approx(0.20)
            )
    finally:
        executor.shutdown()


def test_raw_stop_replacement_api_removed_from_live_executor(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
        assert result["success"]
        assert not hasattr(executor.oco_executor, "_update_stop_" + "loss")
        for method_name in (
            "start_" + "monitoring",
            "stop_" + "monitoring",
            "monitor_" + "positions",
        ):
            assert not hasattr(executor.oco_executor, method_name)
    finally:
        executor.shutdown()


def test_missing_policy_decision_does_not_authorise_replacement(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
        assert result["success"]
        state = executor.oco_executor.active_positions["BTC/USDT"]
        old_stop = state["stop_price"]
        executor.update_position_policy_state("BTC/USDT")
        assert state["stop_price"] == old_stop
        assert len([o for o in exchange.orders if o["type"] == "STOP_LOSS"]) == 1
    finally:
        executor.shutdown()


def test_decision_missing_hash_blocks_live_replacement(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
        assert result["success"]
        state = executor.oco_executor.active_positions["BTC/USDT"]
        old_stop = state["stop_price"]
        decision = SimplePolicyStopDecision(
            should_replace=True,
            stop_price=99.0,
            reason="capital_preservation",
            reason_detail="capital_preservation: test",
            strategy_id="long_mr",
            params_source="artifacts/test-run/simple_policy_optimiser/deployment/best_policy_params.json",
            params_hash="",
            barrier_frac=0.02,
            sl_mult=1.0,
        )
        executor.oco_executor._update_stop_loss_from_policy_decision(
            "BTC/USDT", state, decision
        )
        assert state["stop_price"] == old_stop
        assert state["stop_update_error_category"] == "unauthorised_stop_update"
    finally:
        executor.shutdown()


def test_legacy_fields_alone_cannot_produce_stop_replacement():
    with pytest.raises(SimplePolicyStopParamsError, match="unknown simple-policy"):
        TradeExecutor(
            mode="shadow",
            exchange=None,
            bucket_params={
                "simple_policy_stop_params_by_strategy": {
                    "long_mr": {
                        **_simple_policy_params(),
                        "trail_mult": 0.25,
                        "giveback_pct": 0.01,
                        "profit_lock_amount": 0.003,
                        "fixed_stop_loss_pct": 0.02,
                    }
                }
            },
        )


def test_shadow_rejects_arbitrary_stop_price_without_policy_decision():
    executor = TradeExecutor(
        mode="shadow",
        exchange=None,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
    )
    result = executor.execute_trade(
        "BTC/USDT", "long", 0.5, price=100.0, bucket_key="long_mr"
    )
    assert result["status"] == "recorded"
    old_stop = executor.get_active_positions()["BTC/USDT"]["stop_price"]
    executor.update_position_policy_state("BTC/USDT")
    assert executor.get_active_positions()["BTC/USDT"]["stop_price"] == old_stop


def test_shadow_rejects_invalid_policy_decision_with_shared_validator():
    executor = TradeExecutor(
        mode="shadow",
        exchange=None,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
    )
    result = executor.execute_trade(
        "BTC/USDT", "long", 0.5, price=100.0, bucket_key="long_mr"
    )
    assert result["status"] == "recorded"
    state = executor.positions["BTC/USDT"]
    old_stop = state["stop_price"]
    decision = SimplePolicyStopDecision(
        should_replace=True,
        stop_price=99.0,
        reason="capital_preservation",
        reason_detail="missing hash",
        strategy_id="long_mr",
        params_source="artifacts/test-run/simple_policy_optimiser/deployment/best_policy_params.json",
        params_hash="",
        barrier_frac=0.02,
        sl_mult=1.0,
    )

    executor.update_position_policy_state("BTC/USDT", policy_stop_decision=decision)

    assert state["stop_price"] == old_stop
    assert state["stop_update_error_category"] == "unauthorised_stop_update"
    assert "params_hash" in state["stop_update_error"]


def test_policy_update_rejects_dict_decision_inputs():
    executor = TradeExecutor(
        mode="shadow",
        exchange=None,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
    )
    executor.execute_trade("BTC/USDT", "long", 0.5, price=100.0, bucket_key="long_mr")

    executor.update_position_policy_state(
        "BTC/USDT",
        policy_stop_decision={
            "should_replace": True,
            "stop_price": 99.0,
            "reason": "capital_preservation",
            "reason_detail": "dict input",
            "strategy_id": "long_mr",
            "params_source": "artifacts/test-run/simple_policy_optimiser/deployment/best_policy_params.json",
            "params_hash": "test-policy-hash",
            "barrier_frac": 0.02,
            "sl_mult": 1.0,
        },
    )
    state = executor.get_active_positions()["BTC/USDT"]
    assert state["stop_update_error_category"] == "unauthorised_stop_update"
    assert "SimplePolicyStopDecision" in state["stop_update_error"]


def test_short_policy_decision_replacement_improves_downward():
    executor = TradeExecutor(
        mode="shadow",
        exchange=None,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "short_mr": _simple_policy_params(strategy_id="short_mr")
            }
        },
    )
    result = executor.execute_trade(
        "BTC/USDT", "short", 0.5, price=100.0, bucket_key="short_mr"
    )
    assert result["status"] == "recorded"
    params = executor.get_simple_policy_stop_params("short_mr")
    decision = _policy_decision(
        params,
        stop_price=99.0,
        reason="trailing_profit",
        reason_detail="trailing_profit: test",
    )
    executor.update_position_policy_state("BTC/USDT", policy_stop_decision=decision)
    assert executor.get_active_positions()["BTC/USDT"]["stop_price"] == pytest.approx(
        99.0
    )


def test_execute_trade_hard_blocks_stale_signal_before_order_recording():
    executor = TradeExecutor(
        mode="shadow",
        exchange=None,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params(strategy_id="long_mr")
            }
        },
        config={
            "hard_stale_signal_entry_gate_enabled": True,
            "max_signal_close_to_entry_seconds": 900.0,
        },
    )
    now = pd.Timestamp.now(tz="UTC")
    result = executor.execute_trade(
        "BTC/USDT",
        "long",
        10.0,
        price=100.0,
        bucket_key="long_mr",
        trade_context={
            "signal_bar_ts": (now - pd.Timedelta(minutes=80)).isoformat(),
            "signal_bar_close_ts": (now - pd.Timedelta(minutes=20)).isoformat(),
        },
    )

    assert result["success"] is False
    assert result["status"] == "rejected"
    assert result["error_category"] == "stale_signal_age_exceeded"
    assert result["signal_close_to_entry_seconds"] > 900.0
    assert executor.get_active_positions() == {}


def test_reattach_requires_policy_provenance(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
        assert result["success"]
        state = executor.oco_executor.active_positions["BTC/USDT"]
        state.pop("stop_policy_params_hash", None)
        reattach = executor.oco_executor._reattach_protective_stop(
            "BTC/USDT", state, previous_status="rejected"
        )
        assert not reattach["success"]
        assert reattach["error_category"] == "missing_policy_provenance"
    finally:
        executor.shutdown()


def test_reattach_succeeds_for_policy_derived_stop_state(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
        assert result["success"]
        state = executor.oco_executor.active_positions["BTC/USDT"]
        state["stop_order_id"] = None
        reattach = executor.oco_executor._reattach_protective_stop(
            "BTC/USDT", state, previous_status="rejected"
        )
        assert reattach["success"]
    finally:
        executor.shutdown()


def test_trade_context_cannot_override_validated_simple_policy_fields():
    executor = TradeExecutor(
        mode="shadow",
        exchange=None,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
    )
    result = executor.execute_trade(
        "BTC/USDT",
        "long",
        0.5,
        price=100.0,
        bucket_key="long_mr",
        trade_context={
            "params_hash": "forged-generic",
            "stop_policy_params_hash": "forged-hash",
            "stop_policy_params_source": "forged-source",
            "stop_policy_schema": "forged-schema",
            "sl_mult": 99.0,
            "barrier_frac": 0.99,
            "barrier_pct": 0.99,
        },
    )
    assert result["status"] == "recorded"
    expected_hash = _simple_policy_params()["params_hash"]
    assert result["stop_policy_params_hash"] == expected_hash
    assert (
        result["stop_policy_params_source"]
        == "artifacts/test-run/simple_policy_optimiser/deployment/best_policy_params.json"
    )
    assert result["stop_policy_schema"] == SIMPLE_POLICY_SCHEMA
    assert result["sl_mult"] == pytest.approx(1.0)
    assert result["barrier_frac"] == pytest.approx(0.02)
    state = executor.get_active_positions()["BTC/USDT"]
    assert state["stop_policy_params_hash"] == expected_hash
    assert (
        state["stop_policy_params_source"]
        == "artifacts/test-run/simple_policy_optimiser/deployment/best_policy_params.json"
    )
    assert state["stop_policy_schema"] == SIMPLE_POLICY_SCHEMA
    assert state["sl_mult"] == pytest.approx(1.0)
    assert state["barrier_frac"] == pytest.approx(0.02)


@pytest.mark.parametrize(
    "override, expected",
    [
        ({"params_hash": ""}, "params_hash"),
        ({"params_source": ""}, "params_source"),
        ({"generated_by": "manual"}, "generated_by"),
        ({"schema": "wrong_schema"}, "schema"),
    ],
)
def test_live_entry_fails_closed_without_explicit_policy_metadata(
    monkeypatch, override, expected
):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    params = _simple_policy_params(**override)
    executor = TradeExecutor(
        mode="live",
        exchange=_FilterAwareExchange(),
        bucket_params={"simple_policy_stop_params_by_strategy": {"long_mr": params}},
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
        assert not result["success"]
        assert result["error_category"] == "invalid_simple_policy_stop_params"
        assert expected in result["error"]
    finally:
        executor.shutdown()


def test_raw_stop_replacement_method_is_removed():
    assert not hasattr(
        __import__(
            "extreme_price_movements.inference.trade_executor",
            fromlist=["OCOExecutor"],
        ).OCOExecutor,
        "_replace_stop_order_" + "raw",
    )


def test_strict_immediate_trigger_preflight_repairs_candidate_before_replace(
    monkeypatch,
):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
        assert result["success"]
        state = executor.oco_executor.active_positions["BTC/USDT"]
        old_order_id = state["stop_order_id"]
        old_stop = state["stop_price"]
        params = executor.oco_executor.get_simple_policy_stop_params("long_mr")
        decision = _policy_decision(
            params,
            stop_price=101.0,
            reason_detail="invalid local trigger side",
        )
        executor.update_position_policy_state("BTC/USDT", policy_stop_decision=decision)
        assert "BTC/USDT" not in executor.oco_executor.active_positions
        assert state["requested_policy_stop"] == pytest.approx(101.0)
        assert state["sentinel_pretriggered"] is True
        assert state["last_close_metrics"]["reason"].startswith(
            "software_executable_stop_breach_pre_replace:"
        )
        assert "stop_update_error_category" not in state
        assert any(
            event.get("event")
            == "software_policy_stop_breached_before_exchange_replace"
            for event in state.get("trade_recap_events", [])
        )
    finally:
        executor.shutdown()


def test_live_policy_update_returns_internal_close_metrics(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    params = _simple_policy_params()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={"simple_policy_stop_params_by_strategy": {"long_mr": params}},
        config={"monitor_interval_seconds": 300},
    )

    def _fake_policy_update_close(oco_executor, symbol, state, decision):
        state["last_close_metrics"] = {
            "symbol": symbol,
            "side": state.get("side"),
            "entry_price": state.get("entry_price"),
            "exit_price": 101.0,
            "filled": state.get("size"),
            "reason": "software_policy_stop_close",
        }
        oco_executor.active_positions.pop(symbol, None)

    monkeypatch.setattr(
        OCOExecutor,
        "_update_stop_loss_from_policy_decision",
        _fake_policy_update_close,
    )

    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
        assert result["success"]
        decision = _policy_decision(params, stop_price=101.0)

        update_result = executor.update_position_policy_state(
            "BTC/USDT", policy_stop_decision=decision
        )

        assert update_result["closed_trade"]["symbol"] == "BTC/USDT"
        assert update_result["closed_trade"]["reason"] == "software_policy_stop_close"
        assert executor.get_position("BTC/USDT") is None
    finally:
        executor.shutdown()


def test_stop_fill_close_metrics_marks_open_shadow_exit():
    state = {
        "side": "short",
        "entry_price": 0.3095,
        "size": 18.5,
        "bucket_key": "short_mr",
        "stop_price": 0.316,
        "final_placed_stop": 0.316,
        "requested_policy_stop": 0.316,
        "stop_reason": "original_stop_loss",
        "mfe": 0.0,
        "mae": 0.011,
        "simple_policy_shadow": {
            "schema": "simple_policy_execution_shadow_v1",
            "status": "open",
            "shadow_stop_price": 0.316,
            "shadow_stop_reason": "original_stop_loss",
            "initial_shadow_stop_price": 0.316,
            "policy_entry_price": 0.3095,
            "realized_entry_price": 0.3095,
            "entry_gap_bps": 0.0,
            "events": [],
        },
    }
    order = {
        "id": "stop-order",
        "average": 0.3208,
        "filled": 18.5,
        "status": "closed",
        "type": "market",
    }

    metrics = _closed_trade_metrics(
        "IP/USD:USD", state, order, reason="stop_loss_filled"
    )

    assert metrics["shadow_status"] == "shadow_exit_triggered"
    assert metrics["shadow_exit_price"] == pytest.approx(0.3208)
    assert metrics["shadow_exit_price_source"] == "observed_exchange_stop_fill"
    assert metrics["shadow_theoretical_exit_price"] == pytest.approx(0.316)
    assert metrics["shadow_stop_trigger_price"] == pytest.approx(0.316)
    assert metrics["shadow_trigger_vs_live_exit_gap_bps"] == pytest.approx(
        (1.0 - 0.3208 / 0.316) * 10000.0
    )
    assert metrics["shadow_exit_reason"] == "shadow_stop_loss_filled:original_stop_loss"
    assert metrics["simple_policy_shadow"]["events"][-1]["event"] == (
        "shadow_exchange_stop_filled"
    )
    assert metrics["simple_policy_shadow"]["events"][-1][
        "shadow_exit_price"
    ] == pytest.approx(0.3208)
    assert metrics["simple_policy_shadow"]["events"][-1][
        "shadow_theoretical_exit_price"
    ] == pytest.approx(0.316)


def test_stop_fill_close_metrics_reconciles_pretriggered_shadow_exit():
    state = {
        "side": "short",
        "entry_price": 0.0008211,
        "size": 8900.0,
        "bucket_key": "short_asset",
        "stop_price": 0.0008445,
        "final_placed_stop": 0.0008445,
        "requested_policy_stop": 0.0008445,
        "stop_reason": "original_stop_loss",
        "simple_policy_shadow": {
            "schema": "simple_policy_execution_shadow_v1",
            "status": "shadow_exit_triggered",
            "shadow_stop_price": 0.0008445,
            "shadow_stop_reason": "original_stop_loss",
            "shadow_exit_price": 0.0008562,
            "shadow_exit_reason": "software_executable_stop_breach:original_stop_loss",
            "events": [],
        },
    }
    order = {
        "id": "stop-order",
        "average": 0.0008637,
        "filled": 8900.0,
        "status": "closed",
        "type": "market",
    }

    metrics = _closed_trade_metrics(
        "TURBO/USD:USD", state, order, reason="stop_loss_filled"
    )

    assert metrics["shadow_status"] == "shadow_exit_triggered"
    assert metrics["shadow_exit_price"] == pytest.approx(0.0008637)
    assert metrics["shadow_exit_price_source"] == "observed_exchange_stop_fill"
    assert metrics["shadow_theoretical_exit_price"] == pytest.approx(0.0008562)
    assert metrics["shadow_trigger_vs_live_exit_gap_bps"] == pytest.approx(
        (1.0 - 0.0008637 / 0.0008562) * 10000.0
    )
    assert metrics["shadow_exit_reason"] == "shadow_stop_loss_filled:original_stop_loss"


def test_close_metrics_copy_entry_provenance_after_restart():
    provenance = json.dumps(
        {
            "schema_version": "entry_provenance_v1",
            "fields": {
                "policy_archetype": "short__late_run",
                "side": "short",
                "model_artifact_run_id": "model_20260718_v1",
                "policy_artifact_run_id": "policy_20260718_v1",
                "policy_pathway_id": "joint_trailing_total_mfe_raw_bayesian_v1",
                "sizing_policy_id": "raw_bayesian_sizing_v1",
                "v9_tail95_predecessor_rank": 0.97,
                "market_state_mlp_expected_net_ev_after_1pct": 0.012,
                "archetype_hit_surprise_actual_hit_rate": 0.51,
                "archetype_hit_surprise_expected_hit_rate": 0.63,
                "archetype_hit_surprise_hit_rate_delta": -0.12,
                "threshold_basis_dynamic_ev_target": 0.007,
                "threshold_basis_archetype_baseline_window_days": 28,
                "threshold_basis_archetype_baseline_ev_mean": 0.011,
                "threshold_basis_archetype_baseline_take_profit_rate": 0.71,
                "threshold_basis_archetype_baseline_stop_rate": 0.12,
                "threshold_basis_archetype_baseline_timeout_rate": 0.05,
                "threshold_basis_archetype_baseline_successful_trade_mae_to_sl_mean": 0.19,
                "meta_sel_ood_abs_z_p95": 2.7,
                "meta_lgbm_uncertainty_score": 0.24,
                "inference_drift_score": 0.08,
                "gmm_cluster_id": 4,
                "ae_reconstruction_error": 0.021,
                "meta_lgbm_leaf_count_p10": 11,
            },
        }
    )
    metrics = _closed_trade_metrics(
        "TURBO/USD:USD",
        {
            "side": "short",
            "entry_price": 1.0,
            "size": 2.0,
            "bucket_key": "short_asset",
            "stop_price": 1.1,
            "entry_provenance_json": provenance,
        },
        {"average": 1.05, "filled": 2.0, "type": "market"},
        reason="manual_close",
    )

    assert metrics["entry_provenance_json"] == provenance
    assert metrics["side"] == "short"
    assert metrics["model_artifact_run_id"] == "model_20260718_v1"
    assert metrics["policy_artifact_run_id"] == "policy_20260718_v1"
    assert metrics["policy_pathway_id"] == "joint_trailing_total_mfe_raw_bayesian_v1"
    assert metrics["sizing_policy_id"] == "raw_bayesian_sizing_v1"
    assert metrics["v9_tail95_predecessor_rank"] == pytest.approx(0.97)
    assert metrics["policy_archetype"] == "short__late_run"
    assert metrics["archetype_hit_surprise_hit_rate_delta"] == pytest.approx(-0.12)
    assert metrics["threshold_basis_dynamic_ev_target"] == pytest.approx(0.007)
    assert metrics["threshold_basis_archetype_baseline_window_days"] == 28
    assert metrics["threshold_basis_archetype_baseline_ev_mean"] == pytest.approx(0.011)
    assert metrics["threshold_basis_archetype_baseline_take_profit_rate"] == pytest.approx(0.71)
    assert metrics["threshold_basis_archetype_baseline_stop_rate"] == pytest.approx(0.12)
    assert metrics["threshold_basis_archetype_baseline_timeout_rate"] == pytest.approx(0.05)
    assert metrics[
        "threshold_basis_archetype_baseline_successful_trade_mae_to_sl_mean"
    ] == pytest.approx(0.19)
    assert metrics["meta_sel_ood_abs_z_p95"] == pytest.approx(2.7)
    assert metrics["meta_lgbm_uncertainty_score"] == pytest.approx(0.24)
    assert metrics["inference_drift_score"] == pytest.approx(0.08)
    assert metrics["gmm_cluster_id"] == 4
    assert metrics["ae_reconstruction_error"] == pytest.approx(0.021)
    assert metrics["meta_lgbm_leaf_count_p10"] == 11


def test_closed_trade_metrics_uses_exchange_fees_for_verified_net_pnl():
    state = {
        "side": "long",
        "entry_price": 100.0,
        "size": 1.0,
        "bucket_key": "long_bars",
        "stop_price": 95.0,
        "requested_policy_stop": 95.0,
        "entry_fee_quote": 0.01,
        "entry_fee_source": "order_fee",
        "entry_order_type": "market",
    }
    order = {
        "id": "close-order",
        "average": 110.0,
        "filled": 1.0,
        "status": "closed",
        "type": "market",
        "fee": {"cost": 0.02, "currency": "USD"},
    }

    metrics = _closed_trade_metrics(
        "BTC/USD:USD", state, order, reason="software_executable_stop_breach"
    )

    assert metrics["gross_pnl"] == pytest.approx(10.0)
    assert metrics["fees_verified"] is True
    assert metrics["fee_source"] == "verified_order_fees"
    assert metrics["fees_amount"] == pytest.approx(0.03)
    assert metrics["net_pnl"] == pytest.approx(9.97)
    assert metrics["net_pnl_pct"] == pytest.approx(9.97 / 100.0)
    assert metrics["net_pnl_verification_status"] == "verified_exchange_fees"


def test_order_fee_enrichment_uses_matching_private_trade_fill_fee():
    class _SparseFeeExchange:
        def fetch_order(self, order_id, symbol, params=None):
            raise RuntimeError("fetchOrder unsupported")

        def fetch_closed_orders(self, symbol, since=None, limit=None, params=None):
            return [
                {
                    "id": "close-order",
                    "symbol": symbol,
                    "side": "buy",
                    "type": "market",
                    "status": "closed",
                    "average": 1.578,
                    "filled": 4.6,
                    "fees": [],
                    "info": {"order_id": "close-order"},
                }
            ]

        def fetch_open_orders(self, symbol, since=None, limit=None, params=None):
            return []

        def fetch_my_trades(self, symbol, since=None, limit=None):
            return [
                {
                    "id": "fill-1",
                    "order": "other-order",
                    "symbol": symbol,
                    "side": "buy",
                    "price": 1.579,
                    "amount": 1.0,
                    "fee": {"cost": 0.01, "currency": "USD"},
                },
                {
                    "id": "fill-2",
                    "order": "close-order",
                    "symbol": symbol,
                    "side": "buy",
                    "price": 1.578,
                    "amount": 4.6,
                    "fee": {"cost": 0.0036, "currency": "USD"},
                },
            ]

    order = _enrich_order_from_exchange(
        _SparseFeeExchange(),
        {
            "id": "close-order",
            "symbol": "LPT/USD:USD",
            "side": "buy",
            "type": "market",
            "average": 1.578,
            "filled": 4.6,
        },
        symbol="LPT/USD:USD",
        config={"market_mode": "perps"},
        price=1.578,
    )

    assert order["fee_source_order_fetch"] == "fetch_my_trades"
    assert order["fees"] == [{"cost": pytest.approx(0.0036), "currency": "USD"}]
    metrics = _closed_trade_metrics(
        "LPT/USD:USD",
        {
            "side": "short",
            "entry_price": 1.569,
            "size": 4.6,
            "entry_fee_quote": 0.0036381,
            "entry_order_type": "market",
        },
        order,
        reason="software_executable_stop_breach_pretrigger:exit_pressure_stop_tightening",
        config={"market_mode": "perps"},
    )
    assert metrics["fees_verified"] is True
    assert metrics["exit_fee_quote"] == pytest.approx(0.0036)
    assert metrics["net_pnl_verification_status"] == "verified_exchange_fees"


def test_closed_trade_metrics_promotes_estimated_net_pnl_when_fees_missing():
    state = {
        "side": "long",
        "entry_price": 100.0,
        "size": 1.0,
        "bucket_key": "long_bars",
        "stop_price": 95.0,
        "requested_policy_stop": 95.0,
        "entry_order_type": "market",
    }
    order = {
        "id": "close-order",
        "average": 110.0,
        "filled": 1.0,
        "status": "closed",
        "type": "market",
    }

    metrics = _closed_trade_metrics(
        "BTC/USD:USD",
        state,
        order,
        reason="software_executable_stop_breach",
        config={"fee_bps_market": 5.0, "fee_bps_market_exit": 7.0},
    )

    assert metrics["fees_verified"] is False
    assert metrics["entry_fee_estimate_quote"] == pytest.approx(0.05)
    assert metrics["exit_fee_estimate_quote"] == pytest.approx(0.077)
    assert metrics["estimated_fees_amount"] == pytest.approx(0.127)
    assert metrics["fees_estimated"] is True
    assert metrics["fees_estimated_complete"] is True
    assert metrics["fees_amount"] == pytest.approx(0.127)
    assert metrics["gross_to_net_cost_quote"] == pytest.approx(0.127)
    assert metrics["gross_to_net_cost_pct"] == pytest.approx(0.127 / 100.0)
    assert metrics["net_pnl"] == pytest.approx(9.873)
    assert metrics["net_pnl_pct"] == pytest.approx(9.873 / 100.0)
    assert metrics["net_pnl_estimated"] == pytest.approx(9.873)
    assert metrics["net_pnl_pct_estimated"] == pytest.approx(9.873 / 100.0)
    assert metrics["net_pnl_verification_status"] == "estimated_missing_exchange_fees"


def test_closed_trade_metrics_flags_partial_exchange_fees():
    state = {
        "side": "long",
        "entry_price": 100.0,
        "size": 1.0,
        "bucket_key": "long_bars",
        "stop_price": 95.0,
        "requested_policy_stop": 95.0,
        "entry_fee_quote": 0.01,
        "entry_fee_source": "order_fee",
        "entry_order_type": "market",
    }
    order = {
        "id": "close-order",
        "average": 110.0,
        "filled": 1.0,
        "status": "closed",
        "type": "market",
    }

    metrics = _closed_trade_metrics(
        "BTC/USD:USD",
        state,
        order,
        reason="software_executable_stop_breach",
        config={"fee_bps_market": 5.0, "fee_bps_market_exit": 7.0},
    )

    assert metrics["fees_verified"] is False
    assert metrics["net_pnl_verification_status"] == (
        "partial_exchange_fees_estimated_missing_side"
    )
    assert metrics["entry_fee_quote"] == pytest.approx(0.01)
    assert metrics["exit_fee_estimate_quote"] == pytest.approx(0.077)
    assert metrics["fees_amount"] == pytest.approx(0.087)
    assert metrics["net_pnl"] == pytest.approx(9.913)
    assert metrics["net_pnl_pct"] == pytest.approx(9.913 / 100.0)


def test_closed_trade_metrics_estimates_missing_fees_with_live_perp_default(
    monkeypatch,
):
    for key in (
        "EPM_LIVE_FEE_FALLBACK_BPS",
        "EPM_LIVE_PERP_FEE_BPS",
        "EPM_KRAKEN_FUTURES_TAKER_FEE_BPS",
    ):
        monkeypatch.delenv(key, raising=False)
    state = {
        "side": "short",
        "entry_price": 312.7,
        "size": 0.02,
        "bucket_key": "short_asset",
        "stop_price": 314.65,
        "requested_policy_stop": 314.65,
        "entry_order_type": "limit",
    }
    order = {
        "id": "close-order",
        "average": 317.68,
        "filled": 0.02,
        "status": "closed",
        "type": "market",
    }

    metrics = _closed_trade_metrics(
        "XMR/USD:USD",
        state,
        order,
        reason="stop_loss_filled",
        config={"market_mode": "perps"},
    )

    gross = (312.7 - 317.68) * 0.02
    expected_fees = (312.7 * 0.02 + 317.68 * 0.02) * 5.0 / 10000.0
    assert metrics["gross_pnl"] == pytest.approx(gross)
    assert metrics["fees_verified"] is False
    assert metrics["entry_fee_estimate_bps"] == pytest.approx(5.0)
    assert metrics["exit_fee_estimate_bps"] == pytest.approx(5.0)
    assert metrics["entry_fee_estimate_source"] == (
        "default_live_perp_fee_bps_entry_limit"
    )
    assert metrics["exit_fee_estimate_source"] == (
        "default_live_perp_fee_bps_exit_market"
    )
    assert metrics["estimated_fees_amount"] == pytest.approx(expected_fees)
    assert metrics["fees_amount"] == pytest.approx(expected_fees)
    assert metrics["net_pnl"] == pytest.approx(gross - expected_fees)
    assert metrics["net_pnl_pct"] == pytest.approx(
        (gross - expected_fees) / (312.7 * 0.02)
    )
    assert metrics["net_pnl_estimated"] == pytest.approx(gross - expected_fees)
    assert metrics["net_pnl_verification_status"] == "estimated_missing_exchange_fees"


def test_closed_trade_metrics_uses_actual_exchange_leverage_for_levered_pnl():
    state = {
        "side": "short",
        "entry_price": 0.04938,
        "size": 1000.0,
        "bucket_key": "short_asset",
        "stop_price": 0.0500,
        "requested_policy_stop": 0.0500,
        "entry_order_type": "market",
        "requested_entry_leverage": 18.2939652165,
        "configured_entry_leverage": 18.2939652165,
        "actual_entry_leverage": 10.0,
        "max_entry_leverage": 10.0,
        "wallet_value_at_entry": 48.4392770384,
    }
    order = {
        "id": "close-order",
        "average": 0.04841,
        "filled": 1000.0,
        "status": "closed",
        "type": "market",
    }

    metrics = _closed_trade_metrics(
        "SEI/USD:USD",
        state,
        order,
        reason="software_executable_stop_breach_pre_replace:trailing_profit",
        config={
            "market_mode": "perps",
            "fee_bps_market": 0.0,
            "fee_bps_market_exit": 0.0,
        },
    )

    assert metrics["configured_entry_leverage"] == pytest.approx(10.0)
    assert metrics["actual_entry_leverage"] == pytest.approx(10.0)
    assert metrics["requested_entry_leverage"] == pytest.approx(18.2939652165)
    assert metrics["max_entry_leverage"] == pytest.approx(10.0)
    assert metrics["net_pnl_pct_configured_leverage_estimated"] == pytest.approx(
        metrics["net_pnl_pct_estimated"] * 10.0
    )


def test_closed_trade_metrics_caps_legacy_requested_leverage_for_perps():
    state = {
        "side": "short",
        "entry_price": 0.149,
        "size": 48.0,
        "bucket_key": "short_asset",
        "stop_price": 0.1526,
        "requested_policy_stop": 0.1526,
        "entry_order_type": "market",
        "configured_entry_leverage": 18.8142223551,
        "perp_effective_leverage": 18.8142223551,
        "perp_liquidation_guard_reason": "capped_to_keep_liquidation_beyond_stop",
        "perp_liquidation_requested_leverage": 18.8142223551,
        "perp_liquidation_guarded_leverage": 7.5,
        "perp_liquidation_safe_max_leverage": 7.5,
        "perp_liquidation_leverage_capped": True,
        "perp_liquidation_stop_distance_pct": 0.068,
        "wallet_value_at_entry": 48.3612,
    }
    order = {
        "id": "close-order",
        "average": 0.1526,
        "filled": 48.0,
        "status": "closed",
        "type": "market",
    }

    metrics = _closed_trade_metrics(
        "SUSHI/USD:USD",
        state,
        order,
        reason="software_executable_stop_breach:original_stop_loss",
        config={"market_mode": "perps"},
    )

    assert metrics["actual_entry_leverage"] == pytest.approx(10.0)
    assert metrics["configured_entry_leverage"] == pytest.approx(10.0)
    assert metrics["max_entry_leverage"] == pytest.approx(10.0)
    assert metrics["requested_entry_leverage"] == pytest.approx(18.8142223551)
    assert metrics["perp_liquidation_guard_reason"] == (
        "capped_to_keep_liquidation_beyond_stop"
    )
    assert metrics["perp_liquidation_guarded_leverage"] == pytest.approx(7.5)
    assert metrics["net_pnl_pct_configured_leverage_estimated"] == pytest.approx(
        metrics["net_pnl_pct_estimated"] * 10.0
    )


def test_trade_email_bodies_are_sectioned_and_skip_unwired_nan_values():
    close_body = run_inference._build_trade_close_email_body(
        closed_trade={
            "symbol": "SPX/USD:USD",
            "side": "long",
            "strategy_id": "long_bars",
            "reason": "stop_loss_filled:exchange_valid_giveback_fallback",
            "entry_price": 0.3434,
            "exit_price": 0.3408,
            "ticker_spread_bps": 8.7349,
            "ev_haircut_expected_stop_exit_friction_bps": 79.5,
            "ev_haircut_stop_exit_excess_bps": 64.5,
            "net_pnl_pct": -0.0123,
            "net_pnl_estimated": -0.0042,
            "net_pnl_pct_estimated": -0.0195,
            "estimated_fees_amount": 0.0007,
            "estimated_fee_source": "configured_entry_market_fee_bps+configured_exit_market_fee_bps",
            "fees_estimated": True,
            "fees_estimated_complete": True,
            "net_pnl_verification_status": "estimated_missing_exchange_fees",
            "entry_fee_quote": np.nan,
            "shadow_exit_price": 0.3408,
            "shadow_exit_price_source": "observed_exchange_stop_fill",
            "shadow_theoretical_exit_price": 0.3445,
            "shadow_trigger_vs_live_exit_gap_bps": -108.5,
            "sentinel_executable_price": 0.3408,
            "sentinel_executable_price_source": "orderbook_best_bid",
            "sentinel_stop_distance_bps": -108.5,
            "sentinel_stop_breach_overshoot_bps": 108.5,
            "trade_recap": "line 1\nline 2",
            "perp_liquidation_guard_reason": "capped_to_keep_liquidation_beyond_stop",
            "perp_liquidation_leverage_capped": True,
            "perp_liquidation_requested_leverage": 10.0,
            "perp_liquidation_guarded_leverage": 4.0,
            "perp_liquidation_safe_max_leverage": 4.0,
            "perp_liquidation_stop_distance_pct": 0.1845,
            "perp_liquidation_required_distance_pct": 0.1945,
            "perp_liquidation_distance_at_requested_pct": 0.045,
            "perp_liquidation_distance_at_guarded_pct": 0.195,
            "policy_archetype": "long__compression_release",
            "policy_archetype_source": "policy_archetype",
            "local_side_archetype": "long__compression_release",
            "source_archetype": "compression_release",
            "archetype_label_family": "compression_release",
            "auction_rank_pct": 0.9399,
            "normalized_rank_score": 0.9411,
            "threshold_rank_score": 0.9252,
            "threshold_rank_score_source": "threshold_rank_score_after_friction_ev",
            "adjusted_rank_score": 0.9252,
            "ev_adjusted_rank_score": 0.9252,
            "final_gate_rank_score": 0.9252,
            "final_gate_threshold": 0.9,
            "final_gate_rank_score_source": "portfolio_gate",
            "portfolio_priority": 0.9188,
            "portfolio_priority_multiplier": 1.03,
            "portfolio_priority_adjustment": 0.012,
            "portfolio_rank_adjustment": -0.006,
            "portfolio_priority_after_live_friction_ev": 0.9188,
            "archetype_hit_surprise_threshold": 0.86,
            "archetype_hit_surprise_mode": "hit_surprise_priority_rank_50",
            "archetype_hit_surprise_threshold_delta": -0.02,
            "archetype_hit_surprise_applied": True,
            "archetype_hit_surprise_reason": "applied",
            "archetype_hit_surprise_matched_key": "long__compression_release",
            "archetype_hit_surprise_actual_hit_rate": 0.72,
            "archetype_hit_surprise_expected_hit_rate": 0.64,
            "archetype_hit_surprise_hit_rate_delta": 0.08,
            "archetype_hit_surprise_hit_rate_surprise_z": 1.4,
            "archetype_hit_surprise_support_confidence": 0.75,
            "archetype_hit_surprise_n_eff": 24.0,
            "dynamic_hr_surprise_threshold": 0.805,
            "dynamic_hr_surprise_applied": True,
            "dynamic_hr_surprise_reason": "applied_recent_hr",
            "dynamic_hr_surprise_head": "long",
            "dynamic_hr_surprise_z_eff": -0.45,
            "dynamic_hr_surprise_guarded_y": 0.68,
            "dynamic_hr_surprise_w_lower": 0.11,
            "dynamic_hr_surprise_w_raise": 0.16,
            "dynamic_hr_surprise_state_age_days": 2.5,
            "strategy_ev_hit_rate": 0.70,
            "strategy_ev_avg_net_return": 0.004,
            "strategy_ev_gate_allowed": True,
            "strategy_ev_gate_reason": "pass",
            "estimated_ev_net_return": 0.014269,
            "estimated_ev_historical_net_return": 0.014269,
            "estimated_ev_historical_cost_bps": 101.22,
            "ev_adjusted_net_return_after_friction": 0.00628,
            "ev_adjusted_historical_net_return_before_rebase": 0.014269,
            "ev_inference_cost_rebase_enabled": True,
            "ev_inference_cost_rebase_applied": True,
            "ev_inference_fixed_round_trip_cost_bps": 20.0,
            "ev_inference_spread_multiplier": 1.5,
            "ev_inference_spread_model_bps": 73.2,
            "ev_inference_total_cost_bps": 93.2,
            "ev_inference_cost_model_contract": "fixed20bps_plus_1.5x_live_spread",
        },
        config={"market_mode": "perps"},
    )

    assert "Summary\n" in close_body
    assert "PnL and fees\n" in close_body
    assert "Entry execution\n" in close_body
    assert "Shadow and parity\n" in close_body
    assert "net_pnl_quote_estimated_fees: -0.00420000" in close_body
    assert "estimated_fees_amount: 0.00070000" in close_body
    assert "net_pnl_verification_status: estimated_missing_exchange_fees" in close_body
    assert "ev_haircut_expected_stop_exit_friction_bps: 79.5000" in close_body
    assert "ev_haircut_stop_exit_excess_bps: 64.5000" in close_body
    assert "sentinel_executable_price_source: orderbook_best_bid" in close_body
    assert "sentinel_stop_breach_overshoot_bps: 108.5000" in close_body
    assert "shadow_exit_price_source: observed_exchange_stop_fill" in close_body
    assert "shadow_trigger_vs_live_exit_gap_bps: -108.5000" in close_body
    assert "perp_liquidation_guard_reason" not in close_body
    assert "perp_liquidation_guarded_leverage" not in close_body
    assert "perp_liquidation_stop_distance_pct: 18.4500%" in close_body
    assert "policy_archetype: long__compression_release" in close_body
    assert "policy_archetype_source: policy_archetype" in close_body
    assert "auction_rank_pct: 0.939900" in close_body
    assert (
        "threshold_rank_score_source: threshold_rank_score_after_friction_ev"
        in close_body
    )
    assert "portfolio_priority_after_live_friction_ev: 0.918800" in close_body
    assert "archetype_recent_hit_rate: 72.0000%" in close_body
    assert "archetype_baseline_hit_rate: 64.0000%" in close_body
    assert "archetype_recent_vs_baseline_hit_rate_delta: 8.0000%" in close_body
    assert "archetype_hit_surprise_n_eff: 24.0000" in close_body
    assert "dynamic_hr_surprise_threshold" not in close_body
    assert "dynamic_hr_surprise_reason" not in close_body
    assert "dynamic_hr_surprise_state_age_days" not in close_body
    assert "strategy_ev_avg_net_return: 0.4000%" in close_body
    assert "ev_adjusted_net_return_after_friction: 0.6280%" in close_body
    assert "ev_inference_total_cost_bps: 93.2000" in close_body
    assert (
        "ev_inference_cost_model_contract: fixed20bps_plus_1.5x_live_spread"
        in close_body
    )
    assert "entry_fee_quote" not in close_body
    assert "nan" not in close_body.lower()

    close_html = run_inference._build_trade_close_email_html_body(
        closed_trade={
            "symbol": "SPX/USD:USD",
            "side": "long",
            "strategy_id": "long_bars",
            "reason": "stop_loss_filled:trailing_profit",
            "net_pnl_pct_estimated": -0.0195,
            "net_pnl_estimated": -0.0042,
            "configured_entry_leverage": 18.2939652165,
            "requested_entry_leverage": 18.2939652165,
            "max_entry_leverage": 10.0,
            "base_rank_pct": 0.93,
            "exit_vs_policy_stop_bps": -108.5,
            "exit_vs_expected_spread_bps": 12.3,
            "net_pnl_verification_status": "estimated_missing_exchange_fees",
            "policy_rank_pct": 0.91,
            "auction_rank_pct": 0.94,
            "threshold_rank_score": 0.9252,
            "threshold_rank_score_source": "threshold_rank_score_after_friction_ev",
            "adjusted_rank_score": 0.9252,
            "ev_adjusted_rank_score": 0.9252,
            "final_gate_rank_score": 0.9252,
            "final_gate_threshold": 0.9,
            "portfolio_priority": 0.9188,
            "portfolio_priority_multiplier": 1.03,
            "portfolio_rank_adjustment": -0.006,
            "deployment_rank_threshold": 0.71,
            "base_pred": 0.81,
            "base_rank_pct": 0.93,
            "meta_pred": 0.67,
            "base_train_rank_pct": 0.91,
            "meta_train_rank_pct": 0.89,
            "inference_drift_score": 0.12,
            "uncertainty_score": 0.34,
            "feature_drift_psi_core": 0.08,
            "feature_drift_ks_bin_mean": 0.11,
            "leaf_count_p10": 42.0,
            "leaf_count_min": 11.0,
            "rare_leaf_fraction": 0.05,
            "leaf_model_space_distance_mean": 0.17,
            "gmm_cluster_id": 3,
            "gmm_posterior_max": 0.82,
            "gmm_posterior_margin": 0.61,
            "gmm_entropy": 0.22,
            "gmm_ood_score": 0.09,
            "mahalanobis_distance": 1.18,
            "ae_reconstruction_error": 0.031,
            "email_env_volatility": 1.25,
            "email_env_vol_of_vol": 0.42,
            "email_env_entropy": 0.61,
            "email_env_signed_trend": -0.37,
            "email_env_volume_z": 2.14,
            "email_env_atr_percentile": 0.18,
            "email_env_amihud_z": -0.44,
            "email_env_vwap_distance_atr": 0.73,
            "email_precomputed_feature_sources_json": json.dumps(
                {
                    "email_env_volatility": "candidate_features:rvol_z",
                    "email_env_vol_of_vol": "candidate_features:vol_of_vol",
                    "email_env_entropy": "candidate_features:direction_entropy_20",
                    "email_env_signed_trend": "candidate_features:regime_trend_score",
                    "email_env_volume_z": "candidate_features:volume_z_24",
                    "email_env_atr_percentile": "candidate_features:atr_percentile",
                    "email_env_amihud_z": "candidate_features:amihud_z",
                    "email_env_vwap_distance_atr": "candidate_features:dist_vwap_atr",
                }
            ),
            "dynamic_hr_surprise_z_eff": -0.45,
            "dynamic_hr_surprise_threshold": 0.805,
            "dynamic_hr_surprise_applied": True,
            "dynamic_hr_surprise_reason": "applied_recent_hr",
            "dynamic_hr_surprise_head": "long",
            "dynamic_hr_surprise_guarded_y": 0.68,
            "dynamic_hr_surprise_state_age_days": 2.5,
            "dynamic_hr_threshold": 0.805,
            "policy_archetype": "long__compression_release",
            "policy_archetype_source": "policy_archetype",
            "source_archetype": "compression_release",
            "archetype_label_family": "compression_release",
            "archetype_hit_surprise_threshold": 0.86,
            "archetype_hit_surprise_threshold_delta": -0.02,
            "archetype_hit_surprise_actual_hit_rate": 0.72,
            "archetype_hit_surprise_expected_hit_rate": 0.64,
            "archetype_hit_surprise_hit_rate_delta": 0.08,
            "archetype_hit_surprise_hit_rate_surprise_z": 1.4,
            "archetype_hit_surprise_support_confidence": 0.75,
            "archetype_hit_surprise_n_eff": 24.0,
            "strategy_ev_hit_rate": 0.70,
            "strategy_ev_avg_net_return": 0.004,
            "estimated_ev_net_return": 0.014269,
            "estimated_ev_historical_net_return": 0.014269,
            "ev_adjusted_net_return_after_friction": 0.00628,
            "ev_adjusted_historical_net_return_before_rebase": 0.014269,
            "ev_inference_total_cost_bps": 93.2,
            "ev_inference_spread_multiplier": 1.5,
            "ev_inference_cost_model_contract": "fixed20bps_plus_1.5x_live_spread",
            "threshold_basis_mapped_expected_ev_side_archetype": 0.011,
            "threshold_basis_side_archetype_recent_ev_correction": 0.001,
            "threshold_basis_corrected_expected_ev": 0.012,
            "threshold_basis_dynamic_ev_target": 0.007,
            "threshold_basis_rank_score": 0.91,
            "threshold_basis_apply_cutoff": 0.90,
            "threshold_basis_ev_target_local_support": 146,
            "threshold_basis_archetype_baseline_window_days": 28,
            "threshold_basis_archetype_baseline_scope": "side_x_archetype",
            "threshold_basis_archetype_baseline_trim_fraction": 0.10,
            "threshold_basis_archetype_baseline_support": 126,
            "threshold_basis_archetype_baseline_retained_days": 23,
            "threshold_basis_archetype_baseline_trimmed_days": 3,
            "threshold_basis_archetype_baseline_ev_mean": 0.0118,
            "threshold_basis_archetype_baseline_ev_median": 0.0107,
            "threshold_basis_archetype_baseline_ev_iqr": 0.0082,
            "threshold_basis_archetype_baseline_positive_ev_rate": 0.74,
            "threshold_basis_archetype_baseline_take_profit_rate": 0.61,
            "threshold_basis_archetype_baseline_historical_scope": "side_x_archetype",
            "threshold_basis_archetype_baseline_historical_support": 418,
            "threshold_basis_archetype_baseline_historical_positive_ev_rate": 0.68,
            "threshold_basis_archetype_baseline_recent_vs_historical_positive_ev_rate": 0.06,
            "threshold_basis_archetype_baseline_successful_trade_mae_to_sl_mean": 0.23,
            "threshold_basis_archetype_baseline_successful_trade_mae_to_sl_support": 91,
            "threshold_basis_archetype_baseline_clean_rate": 0.69,
            "threshold_basis_archetype_baseline_dirty_positive_rate": 0.18,
            "threshold_basis_archetype_baseline_bad_mae_rate": 0.21,
            "threshold_basis_archetype_baseline_stop_rate": 0.17,
            "threshold_basis_archetype_baseline_timeout_rate": 0.04,
            "threshold_basis_archetype_baseline_mapped_ev_decile": 9,
            "threshold_basis_archetype_baseline_mapped_ev_decile_support": 18,
            "threshold_basis_archetype_baseline_mapped_ev_decile_calibration_residual": 0.0021,
            "threshold_basis_archetype_baseline_gmm_state_ev_mean": 0.0142,
            "threshold_basis_archetype_baseline_gmm_state_support": 29,
            "meta_sel_ood_abs_z_max": 3.10,
            "meta_sel_ood_abs_z_mean": 0.77,
            "meta_sel_ood_abs_z_p95": 2.22,
            "meta_sel_ood_iqr_exceed_frac": 0.08,
            "meta_sel_ood_missing_frac": 0.0,
            "meta_sel_ood_centroid_l2": 1.43,
            "perp_liquidation_guard_reason": "capped_to_keep_liquidation_beyond_stop",
            "perp_liquidation_leverage_capped": True,
            "perp_liquidation_requested_leverage": 10.0,
            "perp_liquidation_guarded_leverage": 4.0,
            "perp_liquidation_safe_max_leverage": 4.0,
            "perp_liquidation_stop_distance_pct": 0.1845,
        },
        config={"market_mode": "perps", "perp_default_leverage": 10.0},
    )
    assert "<!doctype html>" in close_html
    assert "EPM Trade Closed" in close_html
    assert "Outcome" in close_html
    assert "Prediction Versus Outcome" in close_html
    assert "Admission Rank / Threshold" in close_html
    assert "0.9100 / 0.9000" in close_html
    assert "Exchange Entry Leverage Used" in close_html
    assert "Base Rank pct" in close_html
    assert "Avg First-Touch MAE, Positive-EV Trades" in close_html
    assert "First-Touch TP Rate" in close_html
    assert "First-Touch Stop Rate" in close_html
    assert "First-Touch Timeout Rate" in close_html
    assert "Meta OOD Absolute z p95" in close_html
    assert "28d vs Historical Positive-EV Delta" in close_html
    assert "Base Pred" not in close_html
    assert "-7.8000%" in close_html
    assert "Configured Lev" not in close_html
    assert "Estimated Configured-Leverage Net PnL %" not in close_html
    assert "Estimated Wallet Net PnL %" not in close_html
    assert "Estimated Margin ROI % @ Entry Leverage" not in close_html
    assert "Stop Gap bps" not in close_html
    assert "Exit Spread Delta bps" not in close_html
    assert "Pred vs Outcome Gap" in close_html
    assert "Exit vs Policy Stop Quality" in close_html
    assert "Exit vs Expected Spread Quality" in close_html
    assert "Gap as % of Outcome" in close_html
    assert "trailing profit" in close_html
    assert "Model Health: Drift, Support, Uncertainty and OOD" in close_html
    assert "Dynamic HR z_eff" not in close_html
    assert "Dynamic HR Threshold" not in close_html
    assert "Dynamic HR Reason" not in close_html
    assert "Policy Archetype" in close_html
    assert "long__compression_release" in close_html
    assert "Policy Archetype Source" in close_html
    assert "Auction Rank pct" in close_html
    assert "EV Adjusted Rank Score" in close_html
    assert "Portfolio Priority" in close_html
    assert "Archetype Recent Performance" in close_html
    assert "Archetype Recent Hit Rate" in close_html
    assert "Archetype Baseline Hit Rate" in close_html
    assert "Recent vs Baseline Hit Rate Delta" in close_html
    assert "Archetype n_eff" in close_html
    assert "Strategy EV Net Return" in close_html
    assert "EV After Live Friction" in close_html
    assert "EV Inference Total Cost bps" in close_html
    assert "fixed20bps_plus_1.5x_live_spread" in close_html
    assert "Drift Score" in close_html
    assert "Uncertainty Score" in close_html
    assert "Feature Population Drift PSI" in close_html
    assert "Leaf Support P10" in close_html
    assert "Strategy and Latent State" in close_html
    assert "GMM Centroid Distance (Mahalanobis)" in close_html
    assert "GMM Posterior Max" in close_html
    assert "Market Environment at Entry" in close_html
    assert "Volatility (rvol_z)" in close_html
    assert "Signed Distance from VWAP (ATR-normalized) (dist_vwap_atr)" in close_html
    assert "Policy Context at Entry" in close_html
    assert "Side x Archetype Mapped EV" in close_html
    assert "Policy: 28-Day Archetype Evidence at Entry" in close_html
    assert "28d Net EV Mean" in close_html
    assert "Mapped-EV Decile Calibration Residual" in close_html
    assert "Current GMM-State 28d EV" in close_html
    assert "Fee Status" not in close_html
    assert "Model Drift / OOD" not in close_html
    assert "Liquidation Guard" in close_html
    assert "Guarded Leverage" not in close_html
    assert "Guard Enabled" not in close_html
    assert "Guard Reason" not in close_html
    assert "Required Liquidation Distance" not in close_html
    assert "Distance at Guarded Leverage" not in close_html
    assert "Maintenance Margin Assumption" not in close_html
    assert "Liquidation Fee Buffer" not in close_html
    assert "Safety Buffer" not in close_html
    assert "capped_to_keep_liquidation_beyond_stop" not in close_html
    assert "Full Audit Detail" in close_html
    assert "estimated_missing_exchange_fees" in close_html

    close_subject = run_inference._build_trade_close_email_subject(
        {
            "symbol": "SPX/USD:USD",
            "side": "long",
            "reason": "software_executable_stop_breach_pretrigger:trailing_profit",
            "close_execution_method": "market",
            "net_pnl_pct_estimated": 0.0123,
            "exit_vs_policy_stop_bps": 42.0,
            "stop_origin": "trailing_profit",
        }
    )
    assert close_subject == "EPM trade closed: SPX/USD:USD long Win 1.2300% via market"
    assert "(policy" not in close_subject

    open_body = run_inference._build_trade_open_email_body(
        symbol="TURBO/USD:USD",
        side="short",
        strategy_id="short_asset",
        size=25.0,
        decision={
            "policy_rank_pct": 0.92,
            "rank_threshold": 0.89,
            "policy_archetype": "short__late_run_continuation",
            "archetype_hit_surprise_threshold": 0.91,
            "archetype_hit_surprise_threshold_delta": 0.02,
            "archetype_hit_surprise_applied": True,
            "archetype_hit_surprise_reason": "applied",
            "archetype_hit_surprise_matched_key": "short__late_run_continuation",
            "archetype_hit_surprise_actual_hit_rate": 0.55,
            "archetype_hit_surprise_expected_hit_rate": 0.68,
            "archetype_hit_surprise_hit_rate_delta": -0.13,
            "strategy_ev_hit_rate": 0.61,
            "strategy_ev_avg_net_return": 0.002,
            "ev_haircut_expected_stop_exit_friction_bps": 79.5,
            "ev_haircut_stop_exit_excess_bps": 64.5,
        },
        trade_result={
            "realized_entry_price": 0.0008211,
            "ticker_spread_bps": 8.5215,
            "expected_total_entry_friction_bps": 4.2608,
            "entry_notional_quote": 25.0,
            "entry_fee_estimate_quote": 0.0125,
            "entry_fee_estimate_bps": 5.0,
            "entry_fee_estimate_source": "configured_entry_market_fee_bps",
            "stop_price": 0.0008445,
            "entry_order_type": "market",
            "order": {"id": "entry-order"},
            "base_amount": np.nan,
            "perp_liquidation_guard_reason": "requested_leverage_safe",
            "perp_liquidation_leverage_capped": False,
            "perp_liquidation_requested_leverage": 4.0,
            "perp_liquidation_guarded_leverage": 4.0,
            "perp_liquidation_safe_max_leverage": 7.5,
        },
        predictions={"base_pred": 0.81, "meta_pred": 0.67},
        config={"market_mode": "perps"},
    )

    assert "Decision and rank\n" in open_body
    assert "Entry execution\n" in open_body
    assert "Stops\n" in open_body
    assert "Entry fees\n" in open_body
    assert "entry_fee_estimate_quote: 0.01250000" in open_body
    assert "entry_fee_estimate_bps: 5.0000" in open_body
    assert "entry_fee_estimate_source: configured_entry_market_fee_bps" in open_body
    assert "ev_haircut_expected_stop_exit_friction_bps: 79.5000" in open_body
    assert "ev_haircut_stop_exit_excess_bps: 64.5000" in open_body
    assert "policy_archetype: short__late_run_continuation" in open_body
    assert "archetype_recent_hit_rate: 55.0000%" in open_body
    assert "archetype_baseline_hit_rate: 68.0000%" in open_body
    assert "strategy_ev_hit_rate: 61.0000%" in open_body
    assert "perp_liquidation_guard_reason" not in open_body
    assert "perp_liquidation_guarded_leverage" not in open_body
    assert "base_amount" not in open_body
    assert "nan" not in open_body.lower()

    open_html = run_inference._build_trade_open_email_html_body(
        symbol="TURBO/USD:USD",
        side="short",
        strategy_id="short_asset",
        size=25.0,
        decision={
            "policy_rank_pct": 0.92,
            "rank_threshold": 0.89,
            "policy_archetype": "short__late_run_continuation",
            "archetype_hit_surprise_threshold": 0.91,
            "archetype_hit_surprise_actual_hit_rate": 0.55,
            "archetype_hit_surprise_expected_hit_rate": 0.68,
            "strategy_ev_hit_rate": 0.61,
        },
        trade_result={
            "realized_entry_price": 0.0008211,
            "entry_notional_quote": 25.0,
            "entry_fee_estimate_quote": 0.0125,
            "entry_fee_estimate_bps": 5.0,
            "entry_fee_estimate_source": "configured_entry_market_fee_bps",
            "stop_price": 0.0008445,
            "entry_order_type": "market",
            "order": {"id": "entry-order"},
            "perp_liquidation_guard_reason": "requested_leverage_safe",
            "perp_liquidation_guarded_leverage": 4.0,
            "perp_liquidation_safe_max_leverage": 7.5,
        },
        predictions={"base_pred": 0.81, "meta_pred": 0.67},
        config={"market_mode": "perps"},
    )
    assert "EPM Trade Opened" in open_html
    assert "Action" in open_html
    assert "Entry Fee Est." in open_html
    assert "Policy Archetype" in open_html
    assert "short__late_run_continuation" in open_html
    assert "Archetype Recent Hit Rate" in open_html
    assert "Liquidation Guard" in open_html
    assert "Safe Max Leverage" in open_html
    assert "Full Audit Detail" in open_html


def test_close_email_uses_entry_provenance_and_legacy_archetype_fallback():
    provenance = json.dumps(
        {
            "schema_version": "entry_provenance_v1",
            "fields": {
                "policy_archetype": "long__compression_release",
                "archetype_hit_surprise_actual_hit_rate": 0.72,
                "archetype_hit_surprise_expected_hit_rate": 0.64,
                "archetype_hit_surprise_hit_rate_delta": 0.08,
                "meta_sel_ood_abs_z_p95": 2.3,
                "meta_lgbm_uncertainty_score": 0.17,
                "inference_drift_score": 0.06,
                "gmm_cluster_id": 2,
                "ae_reconstruction_error": 0.014,
                "meta_lgbm_leaf_count_p10": 9,
            },
        }
    )
    closed_trade = {
        "symbol": "SOL/USD:USD",
        "side": "long",
        "reason": "manual_close",
        "entry_provenance_json": provenance,
    }
    plain = run_inference._build_trade_close_email_body(
        closed_trade=closed_trade, config={}
    )
    html = run_inference._build_trade_close_email_html_body(
        closed_trade=closed_trade, config={}
    )
    legacy_plain = run_inference._build_trade_close_email_body(
        closed_trade={
            "symbol": "SOL/USD:USD",
            "side": "long",
            "model_prediction_audit": json.dumps(
                {"thresholds": {"policy_archetype": "long__legacy_arch"}}
            ),
        },
        config={},
    )

    assert "policy_archetype: long__compression_release" in plain
    assert "archetype_recent_hit_rate: 72.0000%" in plain
    assert "archetype_baseline_hit_rate: 64.0000%" in plain
    assert "archetype_recent_vs_baseline_hit_rate_delta: 8.0000%" in plain
    assert "meta_sel_ood_abs_z_p95: 2.300000" in plain
    assert "meta_lgbm_uncertainty_score: 0.170000" in plain
    assert "inference_drift_score: 0.060000" in plain
    assert "Model Health: Drift, Support, Uncertainty and OOD" in html
    assert "Policy Archetype" in html
    assert "Recent HR vs Baseline" in html
    assert "long__legacy_arch" in legacy_plain


def test_trade_close_email_cause_prefers_policy_reason_over_execution_mechanism():
    trailing = run_inference._email_close_cause_plain(
        {
            "side": "short",
            "reason": "software_executable_stop_breach_pre_replace:trailing_profit",
            "exit_reason_detail": "trailing_profit: mfe=0.005 lock_ret=0.002",
            "sentinel_executable_price": 100.2,
            "requested_policy_stop": 100.0,
            "sentinel_executable_price_source": "orderbook_best_ask",
        }
    )
    pressure = run_inference._email_close_cause_plain(
        {
            "side": "short",
            "reason": "software_executable_stop_breach_pretrigger:exit_pressure_stop_tightening",
            "exit_reason_detail": "exit_pressure_stop_tightening: exit_pressure=1.5",
            "sentinel_executable_price": 100.2,
            "requested_policy_stop": 100.0,
            "sentinel_executable_price_source": "orderbook_best_ask",
        }
    )

    assert trailing == "trailing stop locked profit above the configured cost floor."
    assert pressure == "policy tightened the stop because exit pressure increased."
    assert "buy-to-cover price crossed the stop" not in trailing
    assert "buy-to-cover price crossed the stop" not in pressure


def test_close_email_reports_residual_state_and_28d_admission_contract():
    source = {
        "meta_postprocessor_policy_id": "meta_residual_v9_tail95_market_state_mlp_hier_ev_v1",
        "meta_postprocessor_predecessor_id": (
            "meta_residual_extreme_local_champion_overlay_ooftrain_"
            "tieaware_downonly_20260712_v9::forced_local_tail_0.950"
        ),
        "meta_postprocessor_side_archetype": "long|compression_release",
        "resid_event_aegmm_gmm_cluster_id": 2,
        "resid_event_aegmm_gmm_entropy": 0.24,
        "resid_event_aegmm_expected_negative_residual_event": 0.18,
        "resid_event_aegmm_expected_positive_residual_event": 0.41,
        "threshold_basis_policy_id": (
            "ev_target_side_archetype_global_top10_before_mlp_28d_flat_v1"
        ),
        "threshold_basis_family": "ev_target_side_archetype_multiplier_before_mlp",
        "threshold_basis_window_days": 28,
        "threshold_basis_selected": True,
        "threshold_basis_reason": "selected",
        "threshold_basis_rank_score": 0.96,
        "threshold_basis_apply_cutoff": 0.91,
        "threshold_basis_dynamic_ev_target": 0.012,
        "threshold_basis_dynamic_score_threshold": 0.83,
        "threshold_basis_ev_target_multiplier": 1.08,
        "threshold_basis_ev_target_local_support": 143,
        "threshold_basis_ev_target_global_fallback": False,
        "threshold_basis_recent_reference_rows": 12000,
        "threshold_basis_reference_rows": 310000,
    }
    post_plain = "\n".join(
        line
        for line in run_inference._email_meta_postprocessor_plain_lines(source)
        if line
    )
    admission_plain = "\n".join(
        line
        for line in run_inference._email_threshold_basis_plain_lines(source)
        if line
    )
    post_html = str(run_inference._email_meta_postprocessor_html_rows(source))
    admission_html = str(run_inference._email_threshold_basis_html_rows(source))

    assert "tieaware_downonly_20260712_v9" in post_plain
    assert "residual_state_cluster_id: 2" in post_plain
    assert "expected_negative_residual_event: 18.0000%" in post_plain
    assert "Residual-State Cluster" in post_html
    assert "ev_target_side_archetype_global_top10_before_mlp_28d_flat_v1" in admission_plain
    assert "admission_window_days: 28" in admission_plain
    assert "side_archetype_ev_multiplier: 1.080000" in admission_plain
    assert "Side x Archetype Local Support" in admission_html


def test_close_email_context_uses_only_precomputed_feature_snapshots():
    symbol = "SOL/USD:USD"
    candidate_features = pd.DataFrame(
        {
            "rvol_z": [1.25],
            "volatility_of_volatility_48": [0.42],
            "direction_entropy_20": [0.61],
            "regime_trend_score": [-0.37],
            "volume_z_24": [2.14],
            "atr_percentile": [0.18],
            "amihud_z": [-0.44],
            "dist_vwap_atr": [0.73],
            "gmm_cluster_id": [3.0],
            "gmm_posterior_max": [0.82],
            "gmm_posterior_margin": [0.61],
            "gmm_entropy": [0.22],
            "gmm_ood_score": [0.09],
            "mahalanobis_distance": [1.18],
            "ae_reconstruction_error": [0.031],
            "future_outcome_should_not_be_copied": [99.0],
        },
        index=[symbol],
    )
    meta_model_input_features = pd.DataFrame(
        {
            "meta_sel_ood_abs_z_max": [3.6],
            "meta_sel_ood_abs_z_mean": [0.9],
            "meta_sel_ood_abs_z_p95": [2.4],
            "meta_sel_ood_iqr_exceed_frac": [0.14],
            "meta_sel_ood_missing_frac": [0.0],
            "meta_sel_ood_centroid_l2": [1.7],
        },
        index=[symbol],
    )
    snapshot = run_inference._snapshot_precomputed_email_context(
        symbol=symbol,
        candidate_features=candidate_features,
        meta_model_input_features=meta_model_input_features,
    )

    assert snapshot["email_env_volatility"] == pytest.approx(1.25)
    assert snapshot["email_env_vwap_distance_atr"] == pytest.approx(0.73)
    assert snapshot["gmm_posterior_max"] == pytest.approx(0.82)
    assert snapshot["mahalanobis_distance"] == pytest.approx(1.18)
    assert snapshot["meta_sel_ood_abs_z_p95"] == pytest.approx(2.4)
    assert snapshot["meta_sel_ood_iqr_exceed_frac"] == pytest.approx(0.14)
    assert "future_outcome_should_not_be_copied" not in snapshot
    assert json.loads(snapshot["email_precomputed_feature_sources_json"]) == {
        "ae_reconstruction_error": "candidate_features:ae_reconstruction_error",
        "email_env_amihud_z": "candidate_features:amihud_z",
        "email_env_atr_percentile": "candidate_features:atr_percentile",
        "email_env_entropy": "candidate_features:direction_entropy_20",
        "email_env_signed_trend": "candidate_features:regime_trend_score",
        "email_env_vol_of_vol": "candidate_features:volatility_of_volatility_48",
        "email_env_volatility": "candidate_features:rvol_z",
        "email_env_volume_z": "candidate_features:volume_z_24",
        "email_env_vwap_distance_atr": "candidate_features:dist_vwap_atr",
        "gmm_cluster_id": "candidate_features:gmm_cluster_id",
        "gmm_entropy": "candidate_features:gmm_entropy",
        "gmm_ood_score": "candidate_features:gmm_ood_score",
        "gmm_posterior_margin": "candidate_features:gmm_posterior_margin",
        "gmm_posterior_max": "candidate_features:gmm_posterior_max",
        "mahalanobis_distance": "candidate_features:mahalanobis_distance",
        "meta_sel_ood_abs_z_max": "meta_model_input:meta_sel_ood_abs_z_max",
        "meta_sel_ood_abs_z_mean": "meta_model_input:meta_sel_ood_abs_z_mean",
        "meta_sel_ood_abs_z_p95": "meta_model_input:meta_sel_ood_abs_z_p95",
        "meta_sel_ood_centroid_l2": "meta_model_input:meta_sel_ood_centroid_l2",
        "meta_sel_ood_iqr_exceed_frac": "meta_model_input:meta_sel_ood_iqr_exceed_frac",
        "meta_sel_ood_missing_frac": "meta_model_input:meta_sel_ood_missing_frac",
    }
    context = run_inference._model_context_from_scored_decision(
        {"chain_results": snapshot},
        refresh_reason="test",
        timestamp=pd.Timestamp("2026-07-17 10:00:00", tz="UTC"),
        signal_bar_ts=pd.Timestamp("2026-07-17 09:00:00", tz="UTC"),
    )
    for key in (
        "email_env_volatility",
        "email_env_vwap_distance_atr",
        "gmm_posterior_max",
        "mahalanobis_distance",
        "meta_sel_ood_abs_z_p95",
        "meta_sel_ood_iqr_exceed_frac",
        "email_precomputed_feature_sources_json",
    ):
        assert key in context
        assert key in MODEL_AND_POLICY_CONTEXT_KEYS


def test_model_health_email_contains_only_health_diagnostics():
    labels = {
        label
        for label, _, _ in run_inference._email_model_health_html_rows(
            {
                "base_pred": 0.4,
                "base_rank_pct": 0.9,
                "meta_pred": 0.7,
                "calibrated_score": 0.75,
                "policy_rank_pct": 1.0,
                "final_gate_rank_score": 0.91,
                "feature_drift_psi_core": 0.12,
                "meta_sel_ood_abs_z_p95": 2.1,
                "uncertainty_score": 0.2,
                "leaf_count_p10": 14.0,
            }
        )
    }
    assert {"Feature Population Drift PSI", "Meta OOD Absolute z p95", "Leaf Support P10"} <= labels
    assert not {
        "Base Score",
        "Base Batch Rank",
        "Meta Score",
        "Legacy Calibrated Score (diagnostic only)",
        "Legacy Policy Rank (diagnostic only)",
        "Final Gate Rank",
    } & labels


def test_trade_result_entry_fields_merge_into_trade_logger_context():
    features_log = {"symbol": "PUMP/USD:USD"}
    merged = run_inference._merge_trade_result_entry_log_fields(
        features_log,
        {
            "order": {"id": "entry-order"},
            "entry_notional_quote": 7.2635,
            "entry_fee_estimate_quote": 0.00363175,
            "entry_fee_estimate_bps": 5.0,
            "entry_fee_estimate_source": "configured_entry_market_fee_bps",
            "entry_fee_quote": np.nan,
            "entry_fee_source": "missing_exchange_fee",
            "oco_result": {"stop_order_id": "stop-order", "stop_price": 0.00142},
        },
    )

    assert merged["exchange_order_id"] == "entry-order"
    assert merged["entry_notional_quote"] == pytest.approx(7.2635)
    assert merged["entry_fee_estimate_quote"] == pytest.approx(0.00363175)
    assert merged["entry_fee_estimate_bps"] == pytest.approx(5.0)
    assert merged["entry_fee_estimate_source"] == "configured_entry_market_fee_bps"
    assert merged["entry_fee_source"] == "missing_exchange_fee"
    assert merged["stop_order_id"] == "stop-order"
    assert merged["stop_price"] == pytest.approx(0.00142)


def test_live_entry_uses_context_barrier_when_artifact_has_none(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    params = _simple_policy_params()
    params.pop("barrier_pct", None)
    params.pop("barrier_frac", None)
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={"simple_policy_stop_params_by_strategy": {"long_mr": params}},
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT",
            "long",
            100.0,
            price=100.0,
            bucket_key="long_mr",
            trade_context={"barrier_frac": 0.02},
        )
        assert result["success"]
        state = executor.oco_executor.active_positions["BTC/USDT"]
        assert state["barrier_frac"] == pytest.approx(0.02)
        assert state["stop_price"] == pytest.approx(98.0)
    finally:
        executor.shutdown()


def test_live_entry_artifact_barrier_wins_over_forged_context(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT",
            "long",
            100.0,
            price=100.0,
            bucket_key="long_mr",
            trade_context={"barrier_frac": 0.5, "sl_mult": 99.0},
        )
        assert result["success"]
        state = executor.oco_executor.active_positions["BTC/USDT"]
        assert state["barrier_frac"] == pytest.approx(0.02)
        assert state["sl_mult"] == pytest.approx(1.0)
        assert state["stop_price"] == pytest.approx(98.0)
    finally:
        executor.shutdown()


def test_live_replacement_uses_position_barrier_when_artifact_has_none(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
        assert result["success"]
        params = _simple_policy_params()
        params.pop("barrier_pct", None)
        params.pop("barrier_frac", None)
        executor.oco_executor.simple_policy_stop_params_by_strategy["long_mr"] = params
        state = executor.oco_executor.active_positions["BTC/USDT"]
        old_stop = state["stop_price"]
        old_order_id = state["stop_order_id"]
        bars = pd.DataFrame(
            {"open": [100.0], "high": [106.0], "low": [100.0], "close": [105.0]},
            index=pd.date_range("2026-01-01", periods=1, freq="15min", tz="UTC"),
        )
        _evaluate_oco_policy("BTC/USDT", state, bars, executor)
        assert state["barrier_frac"] == pytest.approx(0.02)
        assert "BTC/USDT" not in executor.oco_executor.active_positions
        assert state["sentinel_pretriggered"] is True
        assert state["last_close_metrics"]["reason"].startswith(
            "software_executable_stop_breach_pre_replace:"
        )
        assert "stop_update_error_category" not in state
        assert any(
            event.get("event")
            == "software_policy_stop_breached_before_exchange_replace"
            for event in state.get("trade_recap_events", [])
        )
    finally:
        executor.shutdown()


@pytest.mark.parametrize(
    "decision_overrides",
    [
        {"params_schema": "wrong_schema"},
        {"params_hash": ""},
        {"params_source": ""},
        {"barrier_frac": float("nan")},
    ],
)
def test_forged_policy_decision_metadata_is_rejected(monkeypatch, decision_overrides):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
        assert result["success"]
        state = executor.oco_executor.active_positions["BTC/USDT"]
        old_stop = state["stop_price"]
        params = executor.oco_executor.get_simple_policy_stop_params("long_mr")
        kwargs = _policy_decision(
            params,
            stop_price=99.0,
            reason_detail="forged",
        ).to_dict()
        kwargs.update(decision_overrides)
        decision = SimplePolicyStopDecision(**kwargs)
        executor.oco_executor._replace_stop_order_from_decision(
            "BTC/USDT", state, decision
        )
        assert state["stop_price"] == old_stop
        assert state["stop_update_error_category"] == "unauthorised_stop_update"
        assert exchange.canceled == []
    finally:
        executor.shutdown()


def test_reattach_rejects_nan_policy_provenance(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
        assert result["success"]
        state = executor.oco_executor.active_positions["BTC/USDT"]
        state["stop_order_id"] = None
        state["barrier_frac"] = float("nan")
        reattach = executor.oco_executor._reattach_protective_stop(
            "BTC/USDT", state, previous_status="rejected"
        )
        assert not reattach["success"]
        assert reattach["error_category"] == "missing_policy_provenance"
    finally:
        executor.shutdown()


def test_initial_live_stop_uses_artifact_barrier_without_context(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.oco_executor.place_oco_order(
            symbol="BTC/USDT",
            side="long",
            entry_price=100.0,
            size=1.0,
            bucket_key="long_mr",
        )
        assert result["success"]
        assert result["stop_price"] == pytest.approx(98.0)
        state = executor.oco_executor.active_positions["BTC/USDT"]
        assert state["barrier_frac"] == pytest.approx(0.02)
        assert state["sl_mult"] == pytest.approx(1.0)
    finally:
        executor.shutdown()


def test_live_trade_context_cannot_forge_mirror_stop_fields(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT",
            "long",
            100.0,
            price=100.0,
            bucket_key="long_mr",
            trade_context={
                "stop_price": 1.0,
                "barrier_frac": 0.50,
                "barrier_pct": 0.50,
                "sl_mult": 99.0,
                "strategy_id": "forged",
                "stop_policy_params_source": "forged",
                "stop_policy_params_hash": "forged",
                "stop_policy_schema": "forged",
            },
        )
        assert result["success"]
        mirror = executor.positions["BTC/USDT"]
        expected_hash = _simple_policy_params()["params_hash"]
        assert mirror["stop_price"] == pytest.approx(98.0)
        assert mirror["barrier_frac"] == pytest.approx(0.02)
        assert mirror["barrier_pct"] == pytest.approx(0.02)
        assert mirror["sl_mult"] == pytest.approx(1.0)
        assert mirror["strategy_id"] == "long_mr"
        assert (
            mirror["stop_policy_params_source"]
            == "artifacts/test-run/simple_policy_optimiser/deployment/best_policy_params.json"
        )
        assert mirror["stop_policy_params_hash"] == expected_hash
        assert mirror["stop_policy_schema"] == SIMPLE_POLICY_SCHEMA
    finally:
        executor.shutdown()


def test_shadow_trade_context_cannot_forge_initial_stop_fields():
    executor = TradeExecutor(
        mode="shadow",
        exchange=None,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
    )
    rec = executor.execute_trade(
        "BTC/USDT",
        "long",
        0.5,
        price=100.0,
        bucket_key="long_mr",
        trade_context={
            "stop_price": 1.0,
            "barrier_frac": 0.50,
            "barrier_pct": 0.50,
            "sl_mult": 99.0,
            "strategy_id": "forged",
            "stop_policy_params_source": "forged",
            "stop_policy_params_hash": "forged",
            "stop_policy_schema": "forged",
        },
    )

    assert rec["status"] == "recorded"
    expected_hash = _simple_policy_params()["params_hash"]
    assert rec["stop_price"] == pytest.approx(98.0)
    state = executor.positions["BTC/USDT"]
    assert state["stop_price"] == pytest.approx(98.0)
    assert state["barrier_frac"] == pytest.approx(0.02)
    assert state["barrier_pct"] == pytest.approx(0.02)
    assert state["sl_mult"] == pytest.approx(1.0)
    assert state["strategy_id"] == "long_mr"
    assert (
        state["stop_policy_params_source"]
        == "artifacts/test-run/simple_policy_optimiser/deployment/best_policy_params.json"
    )
    assert state["stop_policy_params_hash"] == expected_hash
    assert state["stop_policy_schema"] == SIMPLE_POLICY_SCHEMA


def test_initial_short_stop_uses_artifact_barrier_and_sl_mult(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "short_mr": _simple_policy_params(strategy_id="short_mr")
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT",
            "short",
            100.0,
            price=100.0,
            bucket_key="short_mr",
            trade_context={"barrier_pct": 0.50, "sl_mult": 99.0},
        )
        assert result["success"]
        state = executor.oco_executor.active_positions["BTC/USDT"]
        assert state["stop_price"] == pytest.approx(102.0)
        assert state["barrier_frac"] == pytest.approx(0.02)
        assert state["sl_mult"] == pytest.approx(1.0)
    finally:
        executor.shutdown()


def test_simple_policy_extraction_rejects_legacy_stop_fields():
    params = _simple_policy_params(trail_mult=0.25)
    with pytest.raises(SimplePolicyStopParamsError, match="unknown simple-policy"):
        extract_simple_policy_stop_params_by_strategy(
            {"simple_policy_stop_params_by_strategy": {"long_mr": params}}
        )


def test_simple_policy_extraction_aliases_and_filters_non_stop_fields():
    selected = _simple_policy_params(strategy_id="long_unique")
    strategies = _simple_policy_params(strategy_id="short_mr")
    legacy_bucket = _simple_policy_params(strategy_id="long_breakout")
    explicit = _simple_policy_params(strategy_id="short_breakout")
    payload = {
        "generated_by": SIMPLE_POLICY_GENERATOR,
        "schema": SIMPLE_POLICY_SCHEMA,
        "params_source": "artifacts/test-run/simple_policy_optimiser/deployment/best_policy_params.json",
        "params_hash": "artifact-hash",
        "selected": [selected],
        "strategies": [selected, {**strategies, "ridge_alpha": 0.7}],
        "buckets": {"Long_Breakout": {**legacy_bucket, "max_hold_hours": 99}},
        "simple_policy_stop_params_by_strategy": {"short_breakout": explicit},
        "ridge_global": 123,
    }
    extracted = extract_simple_policy_stop_params_by_strategy(payload)
    assert set(extracted) == {"short_breakout"}
    assert "long_unique" not in extracted
    assert "SHORT_MR" not in extracted
    assert "Long_Breakout" not in extracted
    assert "long_breakout" not in extracted
    assert extracted["short_breakout"]["strategy_id"] == "short_breakout"
    for row in extracted.values():
        assert "ridge_alpha" not in row
        assert "max_hold_hours" not in row
        assert "ridge_global" not in row


def test_reattach_rejects_open_tracked_stop(monkeypatch):
    monkeypatch.setattr(
        "extreme_price_movements.inference.trade_executor.hf_data_loader.fetch_ohlcv_5m",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    exchange = _FilterAwareExchange()
    executor = TradeExecutor(
        mode="live",
        exchange=exchange,
        bucket_params={
            "simple_policy_stop_params_by_strategy": {
                "long_mr": _simple_policy_params()
            }
        },
        config={"monitor_interval_seconds": 300},
    )
    try:
        result = executor.execute_trade(
            "BTC/USDT", "long", 100.0, price=100.0, bucket_key="long_mr"
        )
        assert result["success"]
        state = executor.oco_executor.active_positions["BTC/USDT"]
        original_order_id = state["stop_order_id"]
        reattach = executor.oco_executor._reattach_protective_stop(
            "BTC/USDT", state, previous_status="open"
        )
        assert not reattach["success"]
        assert reattach["error_category"] == "stop_order_still_active"
        assert state["stop_order_id"] == original_order_id
        assert exchange.canceled == []
    finally:
        executor.shutdown()


def test_simple_policy_geometry_resolves_archetype_before_side_parent():
    executor = OCOExecutor.__new__(OCOExecutor)
    executor.simple_policy_stop_params_by_strategy = {
        "long__parent": {"side": "long", "sl_mult": 2.7},
        "long__policy_archetype_long_mixed": {"side": "long", "sl_mult": 1.8},
        "long__policy_archetype_long__compression_release": {
            "side": "long",
            "sl_mult": 1.4,
        },
        "short__parent": {"side": "short", "sl_mult": 3.1},
    }

    assert executor.resolve_simple_policy_strategy_id(
        "canonical_meta_policy", "long", "policy_archetype_long_mixed"
    ) == "long__policy_archetype_long_mixed"
    assert executor.resolve_simple_policy_strategy_id(
        "canonical_meta_policy", "long", "policy_archetype_unknown"
    ) == "long__parent"
    assert executor.resolve_simple_policy_strategy_id(
        "canonical_meta_policy", "long", "long__compression_release"
    ) == "long__policy_archetype_long__compression_release"
    assert executor.resolve_simple_policy_strategy_id(
        "canonical_meta_policy", "short", None
    ) == "short__parent"
