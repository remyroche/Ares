from extreme_price_movements.simple_policy_optimiser import _build_deployment_payload


def _result(avg_pnl: float, *, holding: dict | None = None) -> dict:
    holding = holding or {}
    metrics = {
        "top_5": {
            "avg_pnl_bankroll": avg_pnl,
            "n_trades": 10,
            "start_date": "2026-01-01",
            "end_date": "2026-01-10",
            **holding,
        },
        "top_1": {
            "n_trades": 1,
            "start_date": "2026-01-01",
            "end_date": "2026-01-10",
        },
    }
    return {
        "validation_metrics": metrics,
        "deployment_threshold_metrics": {"deployment_rank_threshold": 0.7},
        "best_params": {"sl_mult": 1.2},
        "best_size_power": 1.1,
    }


def test_deployment_payload_requires_current_trained_meta_model():
    payload = _build_deployment_payload(
        run_id="test-run",
        oos_results_json={
            "strategies": {
                "long_available": _result(0.02),
                "short_available": _result(0.03),
                "short_missing": _result(0.99),
            }
        },
        available_strategy_ids={"long_available", "short_available"},
    )

    selected = {row["strategy_id"] for row in payload["strategies"]}
    assert selected == {"long_available", "short_available"}

    rejected = {
        row["strategy_id"]: row.get("reject_reasons", [])
        for row in payload["rejected_strategies"]
    }
    assert "missing_trained_meta_model" in rejected["short_missing"]
    assert payload["selection_rules"]["requires_current_trained_meta_model"] is True


def test_deployment_payload_persists_realized_holding_time_metrics():
    payload = _build_deployment_payload(
        run_id="test-run",
        oos_results_json={
            "strategies": {
                "long_available": _result(
                    0.02,
                    holding={
                        "avg_holding_bars": 18.0,
                        "median_holding_bars": 12.0,
                        "p90_holding_bars": 48.0,
                        "max_holding_bars": 96.0,
                        "avg_holding_time_hours": 4.5,
                        "median_holding_time_hours": 3.0,
                        "p90_holding_time_hours": 12.0,
                        "max_holding_time_hours": 24.0,
                    },
                )
            }
        },
        available_strategy_ids={"long_available"},
    )

    strategy = payload["strategies"][0]
    assert strategy["configured_max_holding_time_hours"] == 24.0
    assert strategy["avg_holding_time_hours"] == 4.5
    assert strategy["median_holding_time_hours"] == 3.0
    assert strategy["p90_holding_time_hours"] == 12.0
    assert strategy["max_holding_time_hours"] == 24.0
