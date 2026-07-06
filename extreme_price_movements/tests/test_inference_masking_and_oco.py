import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.candidate_selector import (
    build_strategy_candidate_masks,
    select_candidates,
)
from extreme_price_movements.inference.feature_generator import (
    _synthesize_live_safe_feature_keys,
)
from extreme_price_movements.inference.portfolio_policy import PortfolioPolicyConfig
import extreme_price_movements.inference.run_inference as run_inference
from extreme_price_movements.inference.run_inference import (
    _evaluate_oco_policy,
    _ev_adjusted_prediction_after_entry_friction,
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
    validate_simple_policy_stop_params,
)
from extreme_price_movements.inference.trade_executor import (
    OCOExecutor,
    STOP_MIN_CURRENT_DISTANCE_PCT,
    TradeExecutor,
    _classify_exchange_error,
    _closed_trade_metrics,
    _create_reduce_stop_loss_order,
    _default_cross_margin_dust_quote_threshold,
    _enrich_order_from_exchange,
    _kraken_futures_last_stop_from_executable_stop,
    _protective_stop_trigger_matches_policy,
    _stop_is_at_least_as_protective,
    _stop_trigger_reference_price,
)
from extreme_price_movements.optimise import _select_candidate_trade_mask


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
    assert snap["prescore_orderbook_capacity_quote_within_slippage"] >= 50.0


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


def test_ev_adjustment_includes_stop_exit_haircut():
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
    assert adjusted["ev_haircut_bps"] == pytest.approx(75.0)
    assert adjusted["ev_haircut_stop_exit_source"] == "test.stop_exit"
    assert adjusted["ev_adjusted_net_return_after_friction"] == pytest.approx(
        adjusted["ev_adjusted_net_return_before_friction"] - 0.0075
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
        "symbol,rows,average_spread_bps\n"
        "BTC/USD:USD,10,20.0\n"
        "NMR/USD:USD,10,60.0\n",
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


def test_live_adverse_policy_exit_shadow_uses_executable_close(monkeypatch):
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
        lambda symbol, side, executor: {
            "price": 103.0,
            "source": "ticker_ask",
            "bid": 102.8,
            "ask": 103.0,
            "last": 102.9,
        },
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

    assert result == {"closed_trade": {"symbol": "PNUT/USD:USD", "exit_price": 103.0}}
    assert executor.closed == {
        "symbol": "PNUT/USD:USD",
        "price": 103.0,
        "reason": "adverse_excursion_exit",
    }
    shadow = position_state["simple_policy_shadow"]
    assert shadow["shadow_exit_price"] == 103.0
    assert shadow["shadow_policy_bar_exit_price"] == 99.0
    assert shadow["shadow_exit_price_source"] == "ticker_ask"
    assert shadow["shadow_exit_reason"] == "adverse_excursion_exit"


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


def test_shadow_monitor_uses_closed_5m_price_action():
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
            "high": [102.0, 106.0],
            "low": [100.0, 102.0],
            "close": [102.0, 105.0],
        },
        index=pd.date_range("2026-03-01 00:00", periods=2, freq="5min", tz="UTC"),
    )
    executor.positions["BTC/USDT"]["entry_time"] = pd.Timestamp(
        "2026-03-01 00:00", tz="UTC"
    )
    executor.positions["BTC/USDT"]["ohlcv_5m_latest"] = bars

    statuses = _monitor_active_position_price_action(
        executor,
        exchange=None,
        now=pd.Timestamp("2026-03-01 00:10:06", tz="UTC"),
    )
    updated = executor.get_active_positions()["BTC/USDT"]

    assert updated["stop_price"] > initial_stop
    assert updated["last_5m_eval_ts"] == pd.Timestamp("2026-03-01 00:05", tz="UTC")
    assert statuses["BTC/USDT"]["price_action"]["bars_evaluated"] == 2


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


def test_live_policy_closes_when_executable_side_breaches_stop(monkeypatch):
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

    assert result["success"] is True
    assert result["reason"].startswith("software_executable_stop_breach")
    assert "PNUT/USD:USD" not in executor.get_active_positions()


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


def test_monitor_active_position_runs_lightweight_sentinel_before_5m_policy(
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

    sentinel = statuses["TURBO/USD:USD"]["executable_stop_sentinel"]
    assert sentinel["status"] == "closed"
    assert sentinel["executable_price"] == pytest.approx(0.0008562)
    assert "price_action" not in statuses["TURBO/USD:USD"]
    assert executor.get_position("TURBO/USD:USD") is None


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
    entry_orders = [order for order in exchange.orders if order["type"] == "limit"]
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
        "stop_policy_params_hash": params["params_hash"],
        "stop_policy_schema": SIMPLE_POLICY_SCHEMA,
        "timestamp": "2026-05-17T21:28:04Z",
    }
    try:
        report = executor.reconcile_cross_margin_account()
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
        assert state["stop_price"] == pytest.approx(
            100.0 * (1.0 - STOP_MIN_CURRENT_DISTANCE_PCT)
        )
        assert state["stop_price"] > old_stop
        assert state["stop_order_id"] != old_order_id
        assert exchange.canceled[0][0] == old_order_id
        assert "stop_update_error_category" not in state
        assert any(
            event.get("event") == "simple_policy_stop_min_current_distance_adjusted"
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


def test_closed_trade_metrics_separates_estimated_net_pnl_when_fees_missing():
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
    assert np.isnan(metrics["net_pnl"])
    assert metrics["entry_fee_estimate_quote"] == pytest.approx(0.05)
    assert metrics["exit_fee_estimate_quote"] == pytest.approx(0.077)
    assert metrics["estimated_fees_amount"] == pytest.approx(0.127)
    assert metrics["fees_estimated"] is True
    assert metrics["fees_estimated_complete"] is True
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
    assert (
        "perp_liquidation_guard_reason: capped_to_keep_liquidation_beyond_stop"
        in close_body
    )
    assert "perp_liquidation_guarded_leverage: 4.0000x" in close_body
    assert "perp_liquidation_stop_distance_pct: 18.4500%" in close_body
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
            "deployment_rank_threshold": 0.71,
            "base_pred": 0.81,
            "base_rank_pct": 0.93,
            "meta_pred": 0.67,
            "inference_drift_score": 0.12,
            "uncertainty_score": 0.34,
            "dynamic_hr_surprise_z_eff": -0.45,
            "dynamic_hr_threshold": 0.805,
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
    assert "Estimated Margin ROI % @ Entry Leverage" in close_html
    assert "Exchange Entry Leverage Used" in close_html
    assert "Base Rank pct" in close_html
    assert "Base Pred" not in close_html
    assert "-7.8000%" in close_html
    assert "Configured Lev" not in close_html
    assert "Estimated Configured-Leverage Net PnL %" not in close_html
    assert "Stop Gap bps" not in close_html
    assert "Exit Spread Delta bps" not in close_html
    assert "Pred vs Outcome Gap" in close_html
    assert "Exit vs Policy Stop Quality" in close_html
    assert "Exit vs Expected Spread Quality" in close_html
    assert "Gap as % of Outcome" in close_html
    assert "trailing profit" in close_html
    assert "Metrics" in close_html
    assert "Dynamic HR z_eff" in close_html
    assert "Drift Score" in close_html
    assert "Uncertainty Score" in close_html
    assert "Liquidation Guard" in close_html
    assert "Guarded Leverage" in close_html
    assert "capped_to_keep_liquidation_beyond_stop" in close_html
    assert "4.0000x" in close_html
    assert "Full Audit Detail" in close_html
    assert "estimated_missing_exchange_fees" in close_html

    open_body = run_inference._build_trade_open_email_body(
        symbol="TURBO/USD:USD",
        side="short",
        strategy_id="short_asset",
        size=25.0,
        decision={
            "policy_rank_pct": 0.92,
            "rank_threshold": 0.89,
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
    assert "perp_liquidation_guard_reason: requested_leverage_safe" in open_body
    assert "perp_liquidation_guarded_leverage: 4.0000x" in open_body
    assert "base_amount" not in open_body
    assert "nan" not in open_body.lower()

    open_html = run_inference._build_trade_open_email_html_body(
        symbol="TURBO/USD:USD",
        side="short",
        strategy_id="short_asset",
        size=25.0,
        decision={"policy_rank_pct": 0.92, "rank_threshold": 0.89},
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
    assert "Liquidation Guard" in open_html
    assert "Safe Max Leverage" in open_html
    assert "Full Audit Detail" in open_html


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
        assert state["stop_price"] == pytest.approx(
            100.0 * (1.0 - STOP_MIN_CURRENT_DISTANCE_PCT)
        )
        assert state["stop_price"] > old_stop
        assert state["stop_order_id"] != old_order_id
        assert exchange.canceled[0][0] == old_order_id
        assert "stop_update_error_category" not in state
        assert any(
            event.get("event") == "simple_policy_stop_min_current_distance_adjusted"
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
