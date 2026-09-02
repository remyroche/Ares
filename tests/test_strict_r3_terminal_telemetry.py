from extreme_price_movements.inference.strict_r3_live_execution import (
    _terminal_trade_telemetry,
)


def test_terminal_telemetry_preserves_execution_adjusted_prediction() -> None:
    position = {
        "candidate_id": "candidate-1",
        "symbol": "ABC/USD:USD",
        "side": "long",
        "entry_ts": "2026-09-01T00:00:00Z",
        "entry_price": 1.0,
        "amount": 2.0,
        "entry_reporting_context": {
            "bcf_mc1_expected_net_bps": 91.0,
            "current_mc1_expected_net_bps": 73.0,
            "final_score": 0.99,
            "c1_lva_adapter": "canonical_c1_lva_dual_mc1",
        },
        "execution_economics": {
            "raw_expected_gross_bps": 191.0,
            "mapped_expected_net_bps": 91.0,
            "execution_delay_gap_bps": 12.0,
            "live_spread_bps": 8.0,
            "expected_entry_impact_bps": 3.0,
            "live_round_trip_microstructure_bps": 15.6,
            "execution_microstructure_buffer_bps": 10.0,
            "execution_friction_bps": 25.6,
            "execution_adjusted_expected_net_bps": 153.4,
            "policy_cost_bps": 100.0,
        },
    }
    closed = {
        "candidate_id": "candidate-1",
        "symbol": "ABC/USD:USD",
        "side": "long",
        "actual_exit_filled_amount": 2.0,
        "gross_pnl_confirmed_fill_pct": 0.01,
        "net_pnl_pct": 0.005,
    }

    telemetry = _terminal_trade_telemetry(position=position, closed_trade=closed)

    assert telemetry["prediction"] == {
        "bcf_mc1_expected_net_bps": 91.0,
        "current_mc1_expected_net_bps": 73.0,
        "bcf_final_score": 0.99,
        "current_final_score": None,
        "c1_lva_adapter": "canonical_c1_lva_dual_mc1",
        "portfolio_priority_rank": None,
        "raw_expected_gross_bps": 191.0,
        "mapped_expected_net_bps": 91.0,
        "adverse_only_delay_gap_bps": 12.0,
        "live_full_spread_bps": 8.0,
        "entry_vwap_impact_bps": 3.0,
        "live_round_trip_microstructure_bps": 15.6,
        "execution_buffer_bps": 10.0,
        "execution_friction_bps": 25.6,
        "execution_adjusted_expected_net_bps": 153.4,
        "policy_cost_bps": 100.0,
    }
    assert telemetry["pnl"]["net_bps"] == 50.0
