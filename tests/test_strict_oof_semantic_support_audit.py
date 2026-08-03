"""Tests for the fail-closed, semantic supportive-head OOF audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.strict_oof_semantic_support_audit import (
    audit_semantic_support,
    semantic_head_specs,
)
from extreme_price_movements.supportive_target_semantics import (
    materialize_supportive_target_semantics,
)
from scripts.audit_strict_oof_semantic_support import run


def _source() -> pd.DataFrame:
    decision = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
    return pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "symbol": ["AAA/USD:USD"] * 4,
        "side": ["long", "short", "long", "short"],
        "decision_ts": decision,
        "label_end_ts": decision + pd.Timedelta(hours=12),
        "label_available_ts": decision + pd.Timedelta(hours=12),
        "__path_auxiliary_target_valid__": [1, 1, 1, 1],
        "__time_to_first_meaningful_mfe_target_valid__": [1, 1, 1, 1],
        "__meaningful_mfe_reached_12h__": [1, 0, 1, 0],
        "__peak_mfe_atr_12h__": [2.0, 0.0, 3.0, 0.0],
        "__mae_before_meaningful_mfe_atr_12h__": [0.5, 1.0, 0.7, 1.1],
        "__time_to_first_meaningful_mfe_hours_12h__": [3.0, 12.0, 1.0, 12.0],
        "clean_economic_favorable_first": [1, 0, 1, 0],
        "adverse_first": [0, 1, 0, 1],
        "same_minute_favorable_adverse_conflict": [0, 0, 0, 0],
        "first_favorable_minute": [30.0, np.nan, 120.0, np.nan],
        "__mfe_persistence_path_efficiency_12h__": [0.7, 0.3, 0.9, 0.1],
        "__adverse_trough_atr_12h__": [0.0, 1.2, 0.5, 1.0],
        "__adverse_trough_recovery_50pct_confirmed_2bars_12h__": [0, 1, 1, 0],
    })


def _labels() -> pd.DataFrame:
    return materialize_supportive_target_semantics(_source())[0]


def _predictions(labels: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame({
        "candidate_id": labels.candidate_id,
        "__ts__": labels.decision_ts - pd.Timedelta(minutes=1),
        "__decision_ts__": labels.decision_ts,
        "__label_available_at__": labels.label_available_ts,
        "is_oof": True,
        "prediction_fit_end_ts": labels.decision_ts - pd.Timedelta(days=1),
        "prediction_generated_ts": labels.decision_ts,
        "prediction_model_id": ["model-1"] * len(labels),
        "prediction_fold_id": ["fold-1"] * len(labels),
    })
    for spec in semantic_head_specs(labels.columns):
        prediction = pd.to_numeric(labels[spec.target_column], errors="coerce")
        # Every named head is intentionally supplied in this valid fixture.  A
        # null conditional/censored target is irrelevant outside its mask.
        result[spec.prediction_aliases[0]] = prediction.fillna(0.5)
    return result


def _complete_audit() -> tuple[pd.DataFrame, pd.DataFrame, object]:
    labels = _labels()
    predictions = _predictions(labels)
    audit = audit_semantic_support(
        labels,
        predictions,
        semantic_contract_sha256="semantic-contract-sha",
        oof_manifest={"semantic_target_contract_sha256": "semantic-contract-sha"},
    )
    return labels, predictions, audit


def test_one_to_one_hash_bound_join_evaluates_only_valid_conditional_rows() -> None:
    labels, _, audit = _complete_audit()
    assert audit.status == "STRICT_OOF_SEMANTIC_AUDIT_COMPLETE"
    assert audit.joined is not None
    assert len(audit.joined) == len(labels)
    peak = audit.metrics.loc[audit.metrics["head"].eq("conditional_peak_mfe")].iloc[0]
    assert peak.status == "STRICT_OOF_METRIC"
    # The two unreached candidates must never be scored as zero-valued peak-MFE
    # regression examples.
    assert peak.valid_target_rows == 2
    assert peak.scored_rows == 2


def test_hazard_scoring_preserves_right_censoring_and_at_risk_mask() -> None:
    _, _, audit = _complete_audit()
    # a reaches at 3h and c at 1h, so at 4-8h only b/d are still at risk.
    late_hazard = audit.metrics.loc[
        audit.metrics["head"].eq("meaningful_mfe_hazard_4_8h")
    ].iloc[0]
    assert late_hazard.status == "STRICT_OOF_METRIC"
    assert late_hazard.valid_target_rows == 2
    assert late_hazard.scored_rows == 2


def test_duplicate_identity_or_non_strict_lineage_fails_closed_without_metrics() -> None:
    labels = _labels()
    duplicate = pd.concat([_predictions(labels), _predictions(labels).iloc[[0]]], ignore_index=True)
    audit = audit_semantic_support(
        labels, duplicate,
        semantic_contract_sha256="x",
        oof_manifest={"semantic_target_contract_sha256": "x"},
    )
    assert audit.status == "BLOCKED_FAIL_CLOSED_NO_SEMANTIC_METRICS"
    assert audit.metrics.empty
    assert audit.joined is None
    assert "BLOCKED_PREDICTION_CANDIDATE_IDENTITY_NOT_ONE_TO_ONE" in audit.readiness.status.tolist()

    late_fit = _predictions(labels)
    late_fit.loc[0, "prediction_fit_end_ts"] = labels.loc[0, "decision_ts"]
    audit = audit_semantic_support(
        labels, late_fit,
        semantic_contract_sha256="x",
        oof_manifest={"semantic_target_contract_sha256": "x"},
    )
    assert "BLOCKED_OOF_FIT_END_NOT_BEFORE_DECISION" in audit.readiness.status.tolist()
    assert audit.metrics.empty


def test_contract_hash_must_be_explicitly_bound_by_the_oof_manifest() -> None:
    labels = _labels()
    audit = audit_semantic_support(
        labels, _predictions(labels),
        semantic_contract_sha256="right-hash",
        oof_manifest={"semantic_target_contract_sha256": "wrong-hash"},
    )
    assert audit.status == "BLOCKED_FAIL_CLOSED_NO_SEMANTIC_METRICS"
    assert audit.metrics.empty
    assert audit.readiness.status.tolist() == ["BLOCKED_SEMANTIC_CONTRACT_HASH_UNBOUND"]


def test_runner_writes_only_readiness_for_an_unbound_legacy_style_ledger(tmp_path: Path) -> None:
    labels = _labels()
    label_path = tmp_path / "semantic_labels.parquet"
    labels.to_parquet(label_path, index=False)
    contract_path = tmp_path / "semantic_contract.json"
    contract_path.write_text("{}\n", encoding="utf-8")
    # This resembles the existing legacy support ledger: target prediction
    # columns are present, but row-level model/fold/fit lineage and the
    # semantic contract binding are absent.  It must not produce metrics.
    legacy = _predictions(labels).loc[:, [
        "candidate_id", "__ts__", "__decision_ts__", "__label_available_at__",
        "semantic_oof__meaningful_mfe_reach",
    ]]
    prediction_path = tmp_path / "legacy.parquet"
    legacy.to_parquet(prediction_path, index=False)
    manifest_path = tmp_path / "legacy_manifest.json"
    manifest_path.write_text(json.dumps({"schema": "legacy"}), encoding="utf-8")
    output = tmp_path / "readiness"
    manifest = run(
        semantic_labels=label_path,
        semantic_contract=contract_path,
        oof_predictions=prediction_path,
        oof_manifest=manifest_path,
        output=output,
    )
    assert manifest["status"] == "BLOCKED_FAIL_CLOSED_NO_SEMANTIC_METRICS"
    assert (output / "semantic_support_readiness.parquet").is_file()
    assert not (output / "semantic_support_metrics.parquet").exists()
    readiness = pd.read_parquet(output / "semantic_support_readiness.parquet")
    assert "BLOCKED_PREDICTION_LINEAGE_COLUMNS_MISSING" in readiness.status.tolist()
    assert "BLOCKED_SEMANTIC_CONTRACT_HASH_UNBOUND" in readiness.status.tolist()
    assert manifest["inputs"]["semantic_contract"]["sha256"] == hashlib.sha256(contract_path.read_bytes()).hexdigest()
