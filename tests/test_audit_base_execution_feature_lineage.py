"""Focused contracts for base/execution feature eligibility and score lineage."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_base_execution_feature_lineage import (
    FeatureEligibilityError,
    PredictionLineageError,
    assert_layer_eligibility,
    audit_base_to_execution_handoff,
    audit_oof_prediction_lineage,
    build_feature_eligibility_manifest,
    load_feature_contract,
    run,
)


def _contract(path: Path, payload: dict) -> pd.DataFrame:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_feature_contract(path)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["c1", "c2"],
        "decision_ts": pd.to_datetime(["2026-01-01T11:00:00Z", "2026-01-01T12:00:00Z"], utc=True),
        "feature_cutoff_ts": pd.to_datetime(["2026-01-01T10:45:00Z", "2026-01-01T11:45:00Z"], utc=True),
    })


def _base_predictions() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["c1", "c2"],
        "prediction_ts": pd.to_datetime(["2026-01-01T10:45:00Z", "2026-01-01T11:45:00Z"], utc=True),
        "fit_end_ts": pd.to_datetime(["2026-01-01T10:30:00Z", "2026-01-01T11:30:00Z"], utc=True),
        "is_oof": [True, True],
        "score_base_alpha": [.1, .2],
    })


def test_feature_manifest_excludes_realised_paths_and_keeps_oof_outputs_conditional(tmp_path: Path) -> None:
    base = _contract(tmp_path / "base.json", {"raw_feature_columns": ["ret_1h", "peak_mfe_12h_atr", "score_base_alpha"]})
    execution = _contract(tmp_path / "execution.json", {"feature_contract": {
        "long": ["ret_1h", "score_base_alpha", "predicted_peak_mfe_12h_atr", "predicted_mae_before_mfe_atr"],
        "short": ["ret_1h"],
    }})
    manifest = build_feature_eligibility_manifest(
        base_contract=base,
        execution_contract=execution,
        declared_oof_prediction_features=("score_base_alpha", "predicted_peak_mfe_12h_atr"),
    )
    lookup = manifest.set_index(["model_layer", "model_side", "feature_name"])
    assert lookup.loc[("base", "all", "ret_1h"), "eligibility_status"] == "ELIGIBLE_RESEARCH_CAUSAL"
    assert lookup.loc[("base", "all", "peak_mfe_12h_atr"), "eligibility_status"] == "REJECT_REALIZED_PATH"
    assert lookup.loc[("base", "all", "score_base_alpha"), "eligibility_status"] == "REJECT_BASE_MODEL_DERIVED_INPUT"
    assert lookup.loc[("execution", "long", "score_base_alpha"), "eligibility_status"] == "CONDITIONAL_OOF_LINEAGE_REQUIRED"
    assert lookup.loc[("execution", "long", "predicted_peak_mfe_12h_atr"), "eligibility_status"] == "CONDITIONAL_OOF_LINEAGE_REQUIRED"
    assert lookup.loc[("execution", "long", "predicted_mae_before_mfe_atr"), "eligibility_status"] == "REJECT_ACTION_LAYER_ONLY"


def test_layer_eligibility_requires_a_completed_oof_audit(tmp_path: Path) -> None:
    base = _contract(tmp_path / "base.json", {"raw_feature_columns": ["ret_1h"]})
    execution = _contract(tmp_path / "execution.json", {"feature_columns": ["score_base_alpha"]})
    manifest = build_feature_eligibility_manifest(
        base_contract=base, execution_contract=execution, declared_oof_prediction_features=("score_base_alpha",),
    )
    conditional = manifest.loc[manifest.model_layer.eq("execution")]
    with pytest.raises(FeatureEligibilityError, match="ineligible"):
        assert_layer_eligibility(conditional)
    assert_layer_eligibility(conditional, permit_conditional_oof=True)


def test_base_prediction_must_be_available_at_feature_cutoff_not_merely_decision_time() -> None:
    bad = _base_predictions()
    bad.loc[0, "prediction_ts"] = pd.Timestamp("2026-01-01T10:50:00Z")
    with pytest.raises(PredictionLineageError, match="strict OOF lineage failed"):
        audit_base_to_execution_handoff(_candidates(), bad, prediction_columns=("score_base_alpha",))


def test_oof_fit_end_must_precede_prediction_timestamp() -> None:
    bad = _base_predictions()
    bad.loc[1, "fit_end_ts"] = bad.loc[1, "prediction_ts"]
    with pytest.raises(PredictionLineageError, match="strict OOF lineage failed"):
        audit_oof_prediction_lineage(_candidates(), bad, model_layer="base", prediction_columns=("score_base_alpha",))


def test_prediction_join_is_exactly_one_to_one() -> None:
    missing = _base_predictions().iloc[:1].copy()
    with pytest.raises(PredictionLineageError, match="join is incomplete"):
        audit_oof_prediction_lineage(_candidates(), missing, model_layer="base", prediction_columns=("score_base_alpha",))
    duplicate = pd.concat([_base_predictions(), _base_predictions().iloc[[0]]], ignore_index=True)
    with pytest.raises(PredictionLineageError, match="one-to-one"):
        audit_oof_prediction_lineage(_candidates(), duplicate, model_layer="base", prediction_columns=("score_base_alpha",))


def test_execution_predictions_may_be_scored_at_decision_time_and_run_writes_machine_readable_outputs(tmp_path: Path) -> None:
    candidates = _candidates()
    execution = _base_predictions().rename(columns={"score_base_alpha": "execution_ev_score"})
    execution["prediction_ts"] = candidates["decision_ts"]
    audit = audit_oof_prediction_lineage(
        candidates, execution, model_layer="execution", prediction_columns=("execution_ev_score",),
    )
    assert audit.summary["status"] == "STRICT_OOF_LINEAGE_VERIFIED"
    assert audit.rows.lineage_pass.all()

    base_contract_path = tmp_path / "base_contract.json"
    execution_contract_path = tmp_path / "execution_contract.json"
    _contract(base_contract_path, {"raw_feature_columns": ["ret_1h"]})
    _contract(execution_contract_path, {"feature_columns": ["ret_1h", "score_base_alpha"]})
    candidates.to_parquet(tmp_path / "candidates.parquet", index=False)
    _base_predictions().to_parquet(tmp_path / "base_predictions.parquet", index=False)
    manifest = run(
        base_contract_path=base_contract_path,
        execution_contract_path=execution_contract_path,
        candidates_path=tmp_path / "candidates.parquet",
        base_predictions_path=tmp_path / "base_predictions.parquet",
        base_prediction_columns=("score_base_alpha",),
        declared_oof_prediction_features=("score_base_alpha",),
        output=tmp_path / "audit",
    )
    assert manifest["status"] == "COMPLETE_READ_ONLY_CONTRACT_AUDIT"
    assert (tmp_path / "audit/feature_eligibility_manifest.parquet").is_file()
    assert (tmp_path / "audit/prediction_lineage_audit.parquet").is_file()
    assert (tmp_path / "audit/prediction_lineage_rows.parquet").is_file()
    feature_only = run(
        base_contract_path=base_contract_path,
        execution_contract_path=execution_contract_path,
        declared_oof_prediction_features=("score_base_alpha",),
        output=tmp_path / "feature-only-audit",
    )
    assert feature_only["prediction_audits"] == []
    assert (tmp_path / "feature-only-audit/prediction_lineage_audit.parquet").is_file()
