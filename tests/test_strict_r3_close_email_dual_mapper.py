from __future__ import annotations

import pytest

from extreme_price_movements.inference.canonical_stack_reporting import (
    canonical_stack_rows,
)
from extreme_price_movements.inference.run_inference import (
    _build_trade_close_email_html_body,
)
from extreme_price_movements.inference.strict_r3_live_execution import (
    _close_reporting_payload,
)


def _closed_trade() -> dict:
    position = {
        "candidate_id": "ABC/USD:USD|long|2026-08-20T00:00:00Z",
        "symbol": "ABC/USD:USD",
        "exchange_symbol": "ABC/USD:USD",
        "side": "long",
        "entry_price": 100.0,
        "amount": 2.0,
        "contract_size": 1.0,
        "entry_ts": "2026-08-20T00:01:00Z",
        "entry_fill_ts": "2026-08-20T00:01:00Z",
        "entry_signal_atr": 2.0,
        "mfe": 4.0,
        "effective_leverage": 7.0,
        "entry_reporting_context": {
            "entry_wallet_equity": 200.0,
            "base_score": 0.80,
            "conditional_consensus_rank": 0.90,
            "final_score": 0.95,
            "bcf_mc1_expected_net_bps": 140.0,
            "mc1_d2_expected_net_bps": 110.0,
            "bcf_mc1_admitted_ge_30bps": True,
            "current_mc1_admitted_ge_30bps": True,
            "dual_bcf_current_admitted_ge_30bps": True,
            "portfolio_policy_schema": "strict_r3_bcf_current_dual_mc1_portfolio_v1",
        },
    }
    return _close_reporting_payload(
        position=position,
        exit_row={
            "candidate_id": position["candidate_id"],
            "exit_reason": "trailing",
            "exit_price": 109.0,
            "exit_ts": "2026-08-20T01:01:00Z",
        },
        close_execution_method="reduce_only_market_after_completed_1m_threshold_bar",
        exchange_order_id="exit-1",
        exchange_order={"status": "closed", "average": 110.0, "filled": 2.0, "info": {}},
        actual_exit_time="2026-08-20T01:01:00Z",
    )


def test_close_payload_uses_confirmed_fill_pnl_and_mfe_units() -> None:
    closed = _closed_trade()

    assert closed["actual_entry_fill_price"] == 100.0
    assert closed["actual_exit_price"] == 110.0
    assert closed["gross_pnl_confirmed_fill_quote"] == 20.0
    assert closed["gross_pnl_confirmed_fill_pct"] == pytest.approx(0.10)
    assert closed["gross_pnl_confirmed_fill_wallet_pct"] == pytest.approx(0.10)
    assert closed["gross_pnl_price_basis"] == "confirmed_kraken_entry_and_exit_fills"
    assert closed["mfe_atr"] == pytest.approx(2.0)
    assert closed["mfe_price_pct"] == pytest.approx(0.04)


def test_dual_mapper_close_email_uses_current_authority() -> None:
    closed = _closed_trade()
    rows = canonical_stack_rows(closed)

    assert "EV Map — Dual BCF + Current-v5 MC1" in rows
    labels = {label for label, _, _ in rows["EV Map — Dual BCF + Current-v5 MC1"]}
    assert "BCF MC1 Expected Net (auction priority)" in labels
    assert "Current-v5 MC1 Expected Net" in labels
    assert "Auction Rank (Final Score)" not in labels

    html = _build_trade_close_email_html_body(
        closed_trade=closed,
        config={"market_mode": "perps", "perp_max_leverage": 7.0},
    )
    assert "Corrected Expected EV" not in html
    assert "Gross Realized PnL % (confirmed fills; fees pending)" in html
    assert "MFE (ATR)" in html
    assert "MFE (price %)" in html
    assert "BCF / Current-v5 MC1 Expected EV" in html
