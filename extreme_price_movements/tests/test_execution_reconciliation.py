import json

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.execution_fill_model import (
    stop_exit_fill_price,
    stop_exit_fill_price_array,
)
from extreme_price_movements.inference.execution_reconciliation import (
    build_ledger_replay_field_coverage,
    build_live_decision_replay_reconciliation,
    build_shadow_trade_reconciliation,
    build_spread_slippage_reconciliation,
    _logged_meta_prediction,
)
from extreme_price_movements.portfolio_policy_replay import PortfolioPolicyParams


class _DummyMetaModel:
    feature_columns = ["base_score", "drift_context"]

    def predict(self, X):
        return X["drift_context"].to_numpy(dtype=float)


class _DummyOrchestrator:
    meta_models = {"long_demo_tbm_clf": _DummyMetaModel()}


def test_logged_meta_prediction_scores_final_logged_matrix_directly():
    meta_features = pd.DataFrame(
        [{"base_score": 0.42, "drift_context": 0.73}],
        index=["BTC/USD:USD"],
    )

    pred, source = _logged_meta_prediction(
        _DummyOrchestrator(),
        meta_features,
        side="long",
        strategy_id="demo",
        meta_model_key="long_demo_tbm_clf",
    )

    assert pred == pytest.approx(0.73)
    assert source == "logged_final_meta_input"


def test_logged_meta_prediction_requires_complete_logged_contract():
    meta_features = pd.DataFrame([{"base_score": 0.42}], index=["BTC/USD:USD"])

    pred, source = _logged_meta_prediction(
        _DummyOrchestrator(),
        meta_features,
        side="long",
        strategy_id="demo",
        meta_model_key="long_demo_tbm_clf",
    )

    assert np.isnan(pred)
    assert source == "incomplete_logged_meta_features:1"


def test_spread_slippage_reconciliation_compares_policy_proxy_to_live():
    ledger = pd.DataFrame(
        [
            {
                "signal_bar_ts": pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                "symbol": "BTC/USD:USD",
                "side": "long",
                "strategy_id": "long_a",
                "portfolio_decision": "traded",
                "was_traded": True,
                "entry_slippage_proxy_bps": 4.0,
                "expected_fill_slippage_bps": 6.0,
                "ticker_spread_bps": 8.0,
                "expected_fill_price": 101.0,
                "realized_entry_price": 101.1,
                "theoretical_entry_price": 100.0,
            }
        ]
    )

    rows, summary = build_spread_slippage_reconciliation(ledger)

    assert rows["expected_policy_slippage_bps"].iloc[0] == 4.0
    assert rows["live_total_entry_friction_bps"].iloc[0] == 10.0
    assert rows["policy_vs_live_slippage_delta_bps"].iloc[0] == 2.0
    assert summary["traded_rows"] == 1


def test_decision_replay_flags_replay_accept_live_reject(tmp_path):
    config_path = tmp_path / "optimized_portfolio_policy_config.json"
    params = PortfolioPolicyParams(
        max_concurrent_positions=4,
        max_concurrent_per_side=None,
        max_concurrent_per_strategy=None,
        max_new_entries_per_bar=2,
        global_threshold_floor=0.50,
        threshold_viability_margin=0.0,
        min_position_size=0.01,
    )
    config_path.write_text(json.dumps(params.to_live_config()), encoding="utf-8")
    ledger = pd.DataFrame(
        [
            {
                "signal_bar_ts": pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                "symbol": "BTC/USD:USD",
                "side": "long",
                "strategy_id": "long_a",
                "threshold_rank_score": 0.90,
                "initial_rank_threshold": 0.50,
                "theoretical_entry_price": 100.0,
                "portfolio_decision": "portfolio_rejected",
                "portfolio_reject_reason": "global_auction_stale_signal_age:stale_signal_age_exceeded",
                "was_traded": False,
            }
        ]
    )

    rows, summary = build_live_decision_replay_reconciliation(
        ledger,
        portfolio_policy_config_path=config_path,
    )

    assert summary["replay_accepted"] == 1
    assert summary["live_traded"] == 0
    assert rows["replay_live_gap_class"].iloc[0] == "replay_accept_live_reject"
    assert rows["replay_live_gap_explanation"].iloc[0] == "live_stale_signal_or_data_gate"


def test_ledger_replay_field_coverage_flags_missing_traded_entry_fields():
    ledger = pd.DataFrame(
        [
            {
                "signal_bar_ts": pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                "symbol": "BTC/USD:USD",
                "side": "long",
                "strategy_id": "long_a",
                "portfolio_decision": "traded",
                "base_model_features_json": '["x"]',
                "base_model_feature_values_json": '{"x": 1.0}',
                "base_pred": 0.61,
                "meta_pred": 0.62,
                "calibrated_score": 0.62,
                "policy_rank_pct": 0.9,
                "auction_rank_pct": 0.9,
                "threshold_rank_score": 0.9,
                "threshold_rank_score_source": "policy",
                "passed_rank_gate": True,
                "decision_ts": pd.Timestamp("2026-01-01 00:05", tz="UTC"),
                "signal_bar_close_ts": pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                "policy_entry_price": 100.0,
                "expected_fill_price": 100.2,
                # Deliberately omit realized_entry_price and order_id.
                "signal_to_entry_seconds": 300.0,
                "decision_to_entry_seconds": 3.0,
                "hourly_close_to_latest_decision_price_bps": 1.0,
                "decision_price_to_fill_bps": 2.0,
                "ticker_spread_bps": 8.0,
                "expected_fill_slippage_bps": 3.0,
                "expected_total_entry_friction_bps": 7.0,
                "fee_bps": 7.0,
                "ev_haircut_bps": 0.0,
                "position_id": "p1",
                "was_traded": True,
            }
        ]
    )

    rows, summary = build_ledger_replay_field_coverage(ledger)

    assert summary["live_traded_rows"] == 1
    assert summary["failed_traded_field_checks"] >= 2
    missing = rows.loc[rows["missing_rows"].gt(0), "accepted_alternatives"].tolist()
    assert "realized_entry_price|entry_price_actual" in missing
    assert "order_id" in missing
    assert summary["exact_portfolio_state_replayable_rows"] == 0


def test_ledger_replay_field_coverage_accepts_portfolio_state_snapshot():
    ledger = pd.DataFrame(
        [
            {
                "signal_bar_ts": pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                "symbol": "BTC/USD:USD",
                "side": "long",
                "strategy_id": "long_a",
                "portfolio_decision": "portfolio_rejected",
                "base_model_features_json": '["x"]',
                "base_model_feature_values_json": '{"x": 1.0}',
                "base_pred": 0.61,
                "meta_pred": 0.62,
                "calibrated_score": 0.62,
                "policy_rank_pct": 0.9,
                "auction_rank_pct": 0.9,
                "threshold_rank_score": 0.9,
                "threshold_rank_score_source": "policy",
                "passed_rank_gate": True,
                "portfolio_state_snapshot_json": '{"positions":[],"cooldowns":{}}',
                "portfolio_state_snapshot_hash": "abc123",
                "wallet_before": 10000.0,
                "open_positions_before": 0,
                "cooldowns_before_json": "{}",
                "portfolio_priority": 0.5,
            }
        ]
    )

    rows, summary = build_ledger_replay_field_coverage(ledger)

    assert summary["exact_portfolio_state_replayable_rows"] == 1
    state_rows = rows.loc[rows["field_group"].eq("exact_portfolio_state_replay")]
    assert state_rows["missing_rows"].sum() == 0


def test_shadow_trade_reconciliation_reports_exit_parity_pass():
    trade_log = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                "symbol": "BIO/USD:USD",
                "side": "short",
                "strategy_id": "short_a",
                "action": "exit",
                "status": "closed",
                "exit_price": 0.0312,
                "shadow_exit_price": 0.0312,
                "shadow_entry_gap_bps": 0.0,
                "stop_price": 0.0318,
                "shadow_latest_stop_price": 0.0318,
                "shadow_status": "shadow_exit_triggered",
            }
        ]
    )

    rows, summary = build_shadow_trade_reconciliation(trade_log, tolerance_bps=1.0)

    assert len(rows) == 1
    assert summary["closed_shadow_rows"] == 1
    assert summary["exit_gap_mismatch_rows"] == 0
    assert summary["exit_execution_parity_status"] == "pass"


def test_shadow_trade_reconciliation_reports_open_positions_pending():
    trade_log = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01 00:00", tz="UTC"),
                "symbol": "PNUT/USD:USD",
                "side": "short",
                "strategy_id": "short_a",
                "action": "entry",
                "status": "open",
                "shadow_entry_gap_bps": 0.0,
                "stop_price": 0.0427,
                "shadow_latest_stop_price": 0.0427,
                "shadow_status": "open",
            }
        ]
    )

    _, summary = build_shadow_trade_reconciliation(trade_log, tolerance_bps=1.0)

    assert summary["closed_shadow_rows"] == 0
    assert summary["open_shadow_rows"] == 1
    assert summary["exit_execution_parity_status"] == "pending_open_positions"


def test_stop_exit_fill_model_scalar_and_array_match_long_short():
    long_hit, long_px = stop_exit_fill_price(
        side="long",
        stop_px=100.0,
        candle_high=101.0,
        candle_low=99.5,
        base_gap_bps=15.0,
        alpha_through=0.05,
        max_gap_bps=75.0,
    )
    short_hit, short_px = stop_exit_fill_price(
        side="short",
        stop_px=100.0,
        candle_high=100.5,
        candle_low=99.0,
        base_gap_bps=15.0,
        alpha_through=0.05,
        max_gap_bps=75.0,
    )

    hit, px = stop_exit_fill_price_array(
        side=np.array([1.0, -1.0], dtype=np.float32),
        stop_px=np.array([100.0, 100.0], dtype=np.float32),
        candle_high=np.array([101.0, 100.5], dtype=np.float32),
        candle_low=np.array([99.5, 99.0], dtype=np.float32),
        base_gap_bps=15.0,
        alpha_through=0.05,
        max_gap_bps=75.0,
    )

    assert long_hit is True
    assert short_hit is True
    assert hit.tolist() == [True, True]
    assert px[0] == pytest.approx(long_px, rel=1e-6)
    assert px[1] == pytest.approx(short_px, rel=1e-6)
