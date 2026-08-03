from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from scripts.materialize_authoritative_soft_regime_transition_sidecars import (
    BOCPD_SCHEMA,
    BOCPD_STATUS,
    REGIME_CONTEXT,
    SidecarError,
    _sealed_manifest,
    assemble_sidecars,
    cadence_audit,
    validate_label_resolution_audit,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    hours = pd.date_range("2025-01-06", periods=2, freq="h", tz="UTC")
    lgbm = pd.DataFrame(
        {
            "source_utc": hours,
            "lgbm_transition_probability": [.2, .8],
            "lgbm_transition_available": [True, True],
            "lgbm_entropy": [.7, .7],
            "lgbm_margin": [.6, .6],
            "lgbm_ood_available": [False, False],
            "lgbm_ood_score": [None, None],
            "provenance_partition": ["blocked_oof_2022_2025"] * 2,
            "train_end_exclusive_utc": [pd.Timestamp("2025-01-01", tz="UTC")] * 2,
            "fit_label_resolution_max_utc": [pd.Timestamp("2024-12-31", tz="UTC")] * 2,
        }
    )
    bocpd = pd.DataFrame(
        {
            "source_utc": hours,
            **{field: [1.0, 2.0] for field in REGIME_CONTEXT},
            "bocpd_stable_vs_transition_probability": [.1, .9],
            "bocpd_stable_vs_transition_available": [True, True],
            "bocpd_stable_vs_transition_entropy": [.5, .5],
            "bocpd_stable_vs_transition_margin": [.8, .8],
            "bocpd_regime_available": [True, True],
            "bocpd_ood_available": [False, False],
            "bocpd_ood_score": [None, None],
            "provenance_partition": ["blocked_oof_2022_2025"] * 2,
            "train_end_exclusive_utc": [pd.Timestamp("2025-01-01", tz="UTC")] * 2,
            "fit_label_resolution_max_utc": [pd.Timestamp("2024-12-31", tz="UTC")] * 2,
        }
    )
    return lgbm, bocpd


def test_sidecars_keep_regime_and_transition_separate_and_exclude_rejected_ids() -> None:
    regime, transition = assemble_sidecars(*_inputs())
    assert "lgbm_transition_probability" not in regime
    assert "bocpd__run_length_entropy" in regime
    assert "lgbm_transition_probability" in transition
    assert all("gmm" not in name and "morphology" not in name for name in regime.columns)
    assert all("gmm" not in name and "morphology" not in name for name in transition.columns)
    assert regime["source_utc"].dt.minute.eq(0).all()


def test_label_resolution_audit_fails_if_a_fold_used_unresolved_labels() -> None:
    lgbm, bocpd = _inputs()
    lgbm.loc[0, "fit_label_resolution_max_utc"] = pd.Timestamp("2025-01-01", tz="UTC")
    with pytest.raises(SidecarError, match="strictly before train end"):
        assemble_sidecars(lgbm, bocpd)


def test_sealed_manifest_requires_expected_status_and_hashes(tmp_path) -> None:
    root = tmp_path / "bocpd"
    root.mkdir()
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"schema": BOCPD_SCHEMA, "status": BOCPD_STATUS, "outputs_sha256": {}}))
    (root / "manifest.sha256").write_text(hashlib.sha256(manifest.read_bytes()).hexdigest() + "  manifest.json\n")
    assert _sealed_manifest(root, schema=BOCPD_SCHEMA, status=BOCPD_STATUS)["status"] == BOCPD_STATUS
    manifest.write_text(json.dumps({"schema": BOCPD_SCHEMA, "status": "PENDING", "outputs_sha256": {}}))
    (root / "manifest.sha256").write_text(hashlib.sha256(manifest.read_bytes()).hexdigest() + "  manifest.json\n")
    with pytest.raises(SidecarError, match="required status"):
        _sealed_manifest(root, schema=BOCPD_SCHEMA, status=BOCPD_STATUS)


def test_audit_validator_tolerates_no_historical_rows() -> None:
    validate_label_resolution_audit(pd.DataFrame({"source_utc": pd.date_range("2026-01-01", periods=1, freq="h", tz="UTC")}))


def test_cadence_audit_keeps_native_15m_as_hourly_row_lookback_only() -> None:
    regime, transition = assemble_sidecars(*_inputs())
    audit = cadence_audit(regime=regime, transition=transition)
    assert set(audit["model_sample_cadence"]) == {"1h"}
    assert audit["non_hourly_timestamp_rows"].eq(0).all()
    assert audit["native_15m_contract"].str.contains("never a 15m model example").all()
