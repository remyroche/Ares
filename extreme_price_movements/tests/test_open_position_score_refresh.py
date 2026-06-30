import pandas as pd
import pytest

from extreme_price_movements.inference.run_inference import (
    _active_position_score_whitelist_entries,
    _apply_active_position_score_whitelist,
    _refresh_active_position_model_context_from_decision,
)
from extreme_price_movements.inference.trade_executor import TradeExecutor


def test_open_position_whitelist_adds_matching_strategy_candidate_and_mask():
    active_positions = {
        "BTC/USD:USD": {
            "side": "short",
            "strategy_id": "short_asset",
        }
    }

    entries = _active_position_score_whitelist_entries(
        active_positions,
        accepted_strategies={"short_asset"},
    )
    long_cands, short_cands, masks, diagnostics = _apply_active_position_score_whitelist(
        long_cands=[],
        short_cands=["ETH/USD:USD"],
        strategy_candidate_masks={"short_asset": ["ETH/USD:USD"]},
        whitelist_entries=entries,
    )

    assert long_cands == []
    assert short_cands == ["ETH/USD:USD", "BTC/USD:USD"]
    assert masks["short_asset"] == ["ETH/USD:USD", "BTC/USD:USD"]
    assert diagnostics["added"] == 1


def test_open_position_whitelist_does_not_cross_strategy_ids():
    active_positions = {
        "BTC/USD:USD": {
            "side": "short",
            "strategy_id": "short_asset",
        }
    }

    entries = _active_position_score_whitelist_entries(
        active_positions,
        accepted_strategies={"short_boll"},
    )

    assert entries == []


def test_open_position_model_context_refresh_updates_exit_rank_fields():
    executor = TradeExecutor(mode="shadow")
    executor.positions["BTC/USD:USD"] = {
        "side": "short",
        "strategy_id": "short_asset",
        "entry_price": 100.0,
        "stop_price": 102.0,
    }
    decision = {
        "symbol": "BTC/USD:USD",
        "side": "short",
        "strategy_id": "short_asset",
        "raw_score": 0.61,
        "calibrated_score": 0.61,
        "threshold_score": 0.82,
        "effective_threshold": 0.71,
        "policy_artifact_run_id": "policy_run",
        "chain_results": {
            "base_pred": 0.57,
            "meta_pred": 0.61,
            "base_train_rank_pct": 0.73,
            "meta_train_rank_pct": 0.82,
            "policy_rank_pct": 0.84,
            "sizer_rank_percentile": 0.82,
            "rank_score_source": "historical_meta_oof_percentile",
        },
    }

    refreshed = _refresh_active_position_model_context_from_decision(
        executor,
        decision,
        side="short",
        refresh_reason="test_score_refresh",
        timestamp=pd.Timestamp("2026-06-30T12:00:00Z"),
        signal_bar_ts=pd.Timestamp("2026-06-30T11:00:00Z"),
    )

    state = executor.get_position("BTC/USD:USD")
    assert refreshed is True
    assert state["base_pred"] == pytest.approx(0.57)
    assert state["meta_pred"] == pytest.approx(0.61)
    assert state["meta_train_rank_pct"] == pytest.approx(0.82)
    assert state["rank_percentile"] == pytest.approx(0.82)
    assert state["sizer_rank_percentile"] == pytest.approx(0.82)
    assert state["policy_rank_pct"] == pytest.approx(0.84)
    assert state["last_model_score_refresh_reason"] == "test_score_refresh"

