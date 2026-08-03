from pathlib import Path


def test_h2_bridge_audit_fails_closed_on_score_lineage_not_labels_only():
    source = Path("scripts/audit_h2_2025_identical_row_oof_bridge.py").read_text()
    assert "FAIL_CLOSED_INCOMPATIBLE_H2_OOF_LINEAGE" in source
    assert "score_base_alpha" in source and "score_residual_expected_ev" in source
    assert "no_2026_fit_or_map_labels_used" in source
    assert '"model_sample_cadence": "1h"' in source


def test_h2_bridge_audit_records_exact_label_and_identity_requirements():
    source = Path("scripts/audit_h2_2025_identical_row_oof_bridge.py").read_text()
    assert "execution_label_end_utc" in source
    assert "exact 1m-derived economics labels" in source
    assert "final candidate identity" in source


def test_h2_bridge_audit_coalesces_resolved_label_timestamp_and_checks_score_aliases():
    source = Path("scripts/audit_h2_2025_identical_row_oof_bridge.py").read_text()
    assert 'combine_first(ledger["execution_label_available_at"])' in source
    assert '("score_base_alpha", "base_oof_score")' in source
    assert '("score_residual_expected_ev", "residual_expected_ev")' in source
    assert "historical_score_alias_audit.csv" in source
