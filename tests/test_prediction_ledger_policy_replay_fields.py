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
                "policy_pathway_id": "joint_trailing_total_mfe_raw_bayesian_v1",
                "sizing_policy_id": "raw_bayesian_v1",
                "sizing_overlay_source": "raw_bayesian_v1_frozen_train_state",
                "raw_bayesian_size_multiplier": 1.08,
                "size_before_sizing_overlay": 100.0,
                "size_after_sizing_overlay": 108.0,
                "raw_bayesian_state_fit_rows": 22207,
                "sizing_policy_strategy_id": "short_demo",
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
    assert row["policy_pathway_id"] == "joint_trailing_total_mfe_raw_bayesian_v1"
    assert row["sizing_policy_id"] == "raw_bayesian_v1"
    assert row["sizing_overlay_source"] == "raw_bayesian_v1_frozen_train_state"
    assert row["raw_bayesian_size_multiplier"] == 1.08
    assert row["size_before_sizing_overlay"] == 100.0
    assert row["size_after_sizing_overlay"] == 108.0
    assert row["raw_bayesian_state_fit_rows"] == 22207
    assert row["sizing_policy_strategy_id"] == "short_demo"


def test_prediction_ledger_persists_close_email_policy_and_health_context():
    row = _prediction_ledger_row(
        {
            "symbol": "BTC/USD:USD",
            "strategy_id": "long_demo",
            "chain_results": {
                "threshold_basis_policy_id": "side_archetype_ev70_trim10_21d",
                "threshold_basis_archetype_baseline_ev_mean": 0.011,
                "threshold_basis_archetype_baseline_take_profit_rate": 0.73,
                "threshold_basis_archetype_baseline_stop_rate": 0.09,
                "threshold_basis_archetype_baseline_timeout_rate": 0.04,
                "threshold_basis_archetype_baseline_successful_trade_mae_to_sl_mean": 0.21,
                "archetype_hit_surprise_actual_hit_rate": 0.67,
                "meta_sel_ood_abs_z_p95": 2.4,
                "meta_sel_ood_centroid_l2": 1.7,
            },
        },
        timestamp="2026-07-18T12:00:00Z",
        side="long",
        portfolio_decision="traded",
    )

    assert row["threshold_basis_policy_id"] == "side_archetype_ev70_trim10_21d"
    assert row["threshold_basis_archetype_baseline_ev_mean"] == 0.011
    assert row["threshold_basis_archetype_baseline_take_profit_rate"] == 0.73
    assert row["threshold_basis_archetype_baseline_stop_rate"] == 0.09
    assert row["threshold_basis_archetype_baseline_timeout_rate"] == 0.04
    assert row["threshold_basis_archetype_baseline_successful_trade_mae_to_sl_mean"] == 0.21
    assert row["archetype_hit_surprise_actual_hit_rate"] == 0.67
    assert row["meta_sel_ood_abs_z_p95"] == 2.4
    assert row["meta_sel_ood_centroid_l2"] == 1.7
