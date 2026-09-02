from __future__ import annotations

import json

import pandas as pd

from scripts import report_strict_r3_live_candle as live_report
from scripts.report_strict_r3_live_candle import live_state_entry_consistency


def test_entry_state_consistency_flags_duplicate_live_symbol(tmp_path):
    state_path = tmp_path / "live_state.json"
    state_path.write_text(json.dumps({
        "positions": [
            {"candidate_id": "PUMP/USD:USD|long|old", "exchange_symbol": "PUMP/USD:USD"},
            {"candidate_id": "PUMP/USD:USD|long|new", "exchange_symbol": "PUMP/USD:USD"},
        ],
    }))

    summary, issues = live_state_entry_consistency(
        {"actions": [{"action": "entry", "candidate_id": "PUMP/USD:USD|long|new"}]},
        state_path=state_path,
    )

    assert summary["entry_actions_present_in_state"] == 1
    assert issues == ["live_state_duplicate_symbols: PUMP/USD:USD"]


def test_entry_state_consistency_flags_unrecorded_entry(tmp_path):
    state_path = tmp_path / "live_state.json"
    state_path.write_text(json.dumps({
        "positions": [{"candidate_id": "SUSHI/USD:USD|long|known", "exchange_symbol": "SUSHI/USD:USD"}],
    }))

    _, issues = live_state_entry_consistency(
        {"actions": [{"action": "entry", "candidate_id": "PUMP/USD:USD|long|new"}]},
        state_path=state_path,
    )

    assert issues == [
        "entry_actions_absent_from_live_state: PUMP/USD:USD|long|new",
    ]


def test_any_runtime_reports_the_actual_successful_resealed_producer(monkeypatch, tmp_path):
    decision = pd.Timestamp("2026-08-16T03:00:00Z")
    old = tmp_path / "strict_r3_live_hourly_producer_v51_20260816T030000Z_v1"
    new = tmp_path / "strict_r3_live_hourly_producer_v56_20260816T030000Z_v3"
    old.mkdir()
    new.mkdir()
    (old / "run_manifest.json").write_text('{"status":"failed_closed"}')
    (new / "run_manifest.json").write_text('{"status":"pass"}')
    monkeypatch.setattr(live_report, "ARTIFACTS", tmp_path)

    selected = live_report.producer("any", decision)

    assert selected == new
    assert live_report.producer_runtime(selected) == "v56"
