import json

from extreme_price_movements.inference.run_inference import _prediction_ledger_row


def test_prediction_ledger_persists_simple_policy_replay_fields():
    row = _prediction_ledger_row(
        {
            "symbol": "AAA/USD:USD",
            "strategy_id": "short_demo",
            "rank_threshold": 0.6,
            "chain_results": {
                "normalized_rank_score": 0.75,
                "effective_threshold": 0.65,
            },
        },
        timestamp="2026-01-01T00:00:00Z",
        side="short",
        portfolio_decision="traded",
        execution_snapshot={
            "policy_entry_price": 100.0,
            "barrier_pct": 0.01,
            "stop_price": 101.2,
            "stop_order_id": "stop-1",
        },
        trade_result={
            "sl_mult": 1.2,
            "trailing_activation_mult": 1.5,
            "trailing_power": 1.7,
            "trailing_squash_divisor": 3.0,
            "giveback_beta": 0.6,
            "stop_policy_params_source": "policy.json",
            "stop_policy_params_hash": "abc123",
            "stop_policy_schema": "simple_policy_v1",
        },
        was_traded=True,
    )

    assert row["barrier_pct"] == 0.01
    assert row["sl_mult"] == 1.2
    assert row["policy_sl_mult"] == 1.2
    assert row["policy_stop_price"] == 101.2
    assert row["stop_policy_params_hash"] == "abc123"
    replay_params = json.loads(row["policy_replay_params_json"])
    assert replay_params["barrier_pct"] == 0.01
    assert replay_params["sl_mult"] == 1.2
    assert replay_params["trailing_activation_mult"] == 1.5
    assert replay_params["stop_policy_schema"] == "simple_policy_v1"
