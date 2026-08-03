"""Tests for the non-mutating Stage-C v3/v4 reconciliation author."""

from __future__ import annotations

from scripts.author_stage_c_v4_reconciliation import build_reconciliation


def _manifest(*, rows: dict[str, int], runner: str) -> dict[str, object]:
    return {
        "status": "COMPLETED_RESEARCH_ONLY_STAGE1",
        "target": "retain_h0_given_clear",
        "terminal_decision": "CURRENT_OHLCV_OI_FUNDING_CONTRACT_INSUFFICIENT_FOR_ENTRY_RETENTION",
        "rows": rows,
        "code_sha256": {"stage1_runner": runner},
    }


def test_reconciliation_preserves_history_and_marks_source_drift_without_promotion() -> None:
    payload = build_reconciliation(
        v3=_manifest(rows={"input_compatible": 10, "clear_first_support": 4, "predictions": 12}, runner="old"),
        v4=_manifest(rows={"input_compatible": 20, "clear_first_support": 8, "predictions": 24}, runner="sealed"),
        target_audit={"status": "STAGE_C_V4_PRIMARY_TARGET_CONTRACT_VERIFIED", "checks": {"passed": 55}},
        current_runner_sha256="current",
    )
    assert payload["promotion_eligible"] is False
    assert payload["historical_record"]["v3"]["purpose"].startswith("preserved")
    assert payload["historical_record"]["v4"]["role"].startswith("current")
    assert payload["reproducibility"]["status"] == "SOURCE_DRIFT_AFTER_SEALED_V4"
    assert any(section["section"] == "5. primary conditional target" and section["status"] == "VERIFIED" for section in payload["specification_sections"])
