from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/refresh_strict_r3_execution_adjusted_ev_calibration.py"
)
SPEC = importlib.util.spec_from_file_location("calibration_refresh", MODULE_PATH)
assert SPEC and SPEC.loader
refresh_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(refresh_module)


def test_refresh_is_immutable_read_only_and_excludes_pending_fee(tmp_path) -> None:
    ledger = tmp_path / "ledger.json"
    fee = tmp_path / "fees.json"
    prediction = tmp_path / "predictions.json"
    ledger.write_text(json.dumps({"records": {
        "a": {"trade_telemetry": {"prediction": {}, "pnl": {"fees_verified": False}}},
    }}))
    fee.write_text(json.dumps({
        "schema": "strict_r3_fee_confirmed_execution_sidecar_v1",
        "rows": [{"record_key": "a", "status": "confirmed", "net_bps": 80.0}],
    }))
    prediction.write_text(json.dumps({
        "schema": "strict_r3_execution_prediction_recovery_sidecar_v1",
        "rows": [{
            "record_key": "a", "status": "confirmed",
            "execution_adjusted_expected_net_bps": 90.0,
        }],
    }))
    receipt = refresh_module.refresh(
        ledger_path=ledger,
        fee_sidecar_path=fee,
        prediction_sidecar_path=prediction,
        out_root=tmp_path / "out",
        as_of=pd.Timestamp("2026-09-01T00:00:00Z"),
    )
    data = json.loads(receipt.read_text())
    assert data["scope"].startswith("read-only calibration observer")
    assert data["calibration"]["confirmed_observations"] == 1
    assert data["calibration"]["observations"][0]["realised_net_bps"] == 80.0
    assert data["calibration"]["observations"][0]["prediction_source"] == "persisted_entry_receipt"
    try:
        refresh_module.refresh(
            ledger_path=ledger,
            fee_sidecar_path=fee,
            prediction_sidecar_path=prediction,
            out_root=tmp_path / "out",
            as_of=pd.Timestamp("2026-09-01T00:00:00Z"),
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("refresh allowed an immutable receipt overwrite")
