from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/recover_strict_r3_execution_prediction_sidecar.py"
)
SPEC = importlib.util.spec_from_file_location("prediction_sidecar", MODULE_PATH)
assert SPEC and SPEC.loader
prediction_sidecar = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prediction_sidecar)


def _ledger() -> dict:
    return {"records": {"key": {
        "symbol": "TEST/USD:USD",
        "candidate_id": "TEST/USD:USD|long|2026-08-21T10:00:00Z",
        "trade_telemetry": {"entry": {"entry_fill_time": "2026-08-21T11:00:01Z"}},
    }}}


def _receipt(path: Path, *, expected: float) -> Path:
    line = {
        "decision_ts": "2026-08-21T11:00:00Z",
        "actions": [{
            "action": "entry",
            "candidate_id": "TEST/USD:USD|long|2026-08-21T10:00:00Z",
            "actual_entry_fill_ts": "2026-08-21T11:00:01Z",
            "entry_order_id": "order",
            "execution_economics": {"execution_adjusted_expected_net_bps": expected},
        }],
    }
    path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    return path


def test_recovers_only_exact_candidate_and_fill_match(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path / "execution.log", expected=80.0)
    result = prediction_sidecar.recover(ledger=_ledger(), receipt_paths=[receipt])
    row = result["rows"][0]
    assert row["status"] == "confirmed"
    assert row["execution_adjusted_expected_net_bps"] == 80.0


def test_rejects_conflicting_receipts(tmp_path: Path) -> None:
    first = _receipt(tmp_path / "first.log", expected=80.0)
    second = _receipt(tmp_path / "second.log", expected=100.0)
    result = prediction_sidecar.recover(ledger=_ledger(), receipt_paths=[first, second])
    row = result["rows"][0]
    assert row["status"] == "unconfirmed"
    assert row["reason"] == "conflicting_persisted_entry_economics_matches"
