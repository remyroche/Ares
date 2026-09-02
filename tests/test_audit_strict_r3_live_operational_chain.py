from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts/audit_strict_r3_live_operational_chain.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("operational_chain_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_counts_only_complete_matching_cross_runtime_receipts(monkeypatch, tmp_path):
    module = _load_module()
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    artifacts.mkdir()
    reports.mkdir()
    decision = pd.Timestamp("2026-08-16T03:00:00Z")
    producer = artifacts / "strict_r3_live_hourly_producer_v56_20260816T030000Z_v3"
    producer.mkdir()
    (producer / "producer_lease.json").write_text("{}")
    (producer / "execution_attempt_started.json").write_text("{}")
    (producer / "run_manifest.json").write_text(json.dumps({
        "status": "pass", "mode": "live", "exchange_order_submission": True,
        "decision_ts": decision.isoformat(),
    }))
    report = reports / "strict_r3_live_candle_v56_20260816T030000Z_strict_r3_live_hourly_producer_v56_20260816T030000Z_v3.json"
    report.write_text(json.dumps({
        "runtime_tag": "v56", "status": "pass", "irregularities": [],
        "producer_receipt": str(producer.relative_to(tmp_path)),
        "position_monitor": {"receipt": "monitor/receipt"},
    }))
    supervisor = artifacts / "strict_r3_live_operations_supervisor_v56_20260816T030000Z_v3"
    supervisor.mkdir()
    (supervisor / "run_manifest.json").write_text(json.dumps({
        "terminal": True, "report_status": "pass",
    }))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "ARTIFACTS", artifacts)
    monkeypatch.setattr(module, "REPORTS", reports)

    result = module.audit(start=decision, required=8)

    assert result["valid_consecutive_candles"] == 1
    assert result["remaining"] == 7
