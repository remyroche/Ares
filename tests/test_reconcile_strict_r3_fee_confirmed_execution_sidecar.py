from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/reconcile_strict_r3_fee_confirmed_execution_sidecar.py"
)
SPEC = importlib.util.spec_from_file_location("fee_sidecar", MODULE_PATH)
assert SPEC and SPEC.loader
fee_sidecar = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fee_sidecar)


def _ledger() -> dict:
    return {
        "records": {
            "a": {
                "symbol": "TEST/USD:USD",
                "candidate_id": "TEST/USD:USD|long|2026-08-21T10:00:00Z",
                "trade_telemetry": {
                    "entry": {
                        "entry_fill_time": "2026-08-21T10:00:03Z",
                        "notional_quote": 100.0,
                    },
                    "exit": {"exit_time": "2026-08-21T10:10:00Z"},
                },
            },
        },
    }


def test_reconciles_fee_and_funding_from_exact_contract_window() -> None:
    logs = [
        {"id": "1", "date": "2026-08-21T10:00:03Z", "asset": "usd", "contract": "pf_testusd", "info": "futures trade", "fee": "1.0", "realized_pnl": "0", "realized_funding": None},
        {"id": "2", "date": "2026-08-21T10:05:00Z", "asset": "usd", "contract": "pf_testusd", "info": "funding rate change", "fee": "0", "realized_pnl": None, "realized_funding": "2.0"},
        {"id": "3", "date": "2026-08-21T10:10:01Z", "asset": "usd", "contract": "pf_testusd", "info": "futures trade", "fee": "1.5", "realized_pnl": "9", "realized_funding": "0"},
    ]
    result = fee_sidecar.reconcile(ledger=_ledger(), account_logs=logs, tolerance_seconds=10)
    row = result["rows"][0]
    assert row["status"] == "confirmed"
    assert row["realized_pnl_quote"] == 9.0
    assert row["fees_quote"] == 2.5
    assert row["funding_quote"] == 2.0
    assert row["net_quote"] == 8.5
    assert row["net_bps"] == 850.0


def test_fails_closed_without_exit_evidence() -> None:
    logs = [
        {"id": "1", "date": "2026-08-21T10:00:03Z", "asset": "usd", "contract": "pf_testusd", "info": "futures trade", "fee": "1.0", "realized_pnl": "0", "realized_funding": None},
    ]
    result = fee_sidecar.reconcile(ledger=_ledger(), account_logs=logs, tolerance_seconds=10)
    row = result["rows"][0]
    assert row["status"] == "unconfirmed"
    assert row["reason"] == "account_log_history_does_not_cover_trade_window"
