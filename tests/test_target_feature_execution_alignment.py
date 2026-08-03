from __future__ import annotations

import pandas as pd

from extreme_price_movements.target_feature_execution_alignment import (
    build_execution_oof_lineage_manifest,
    materialize_supportive_metadata,
    validate_feature_manifest,
    validate_fold_and_oof,
    validate_global_tail,
    validate_primary_labels,
    validate_supportive_labels,
)


def _primary() -> pd.DataFrame:
    decision = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    return pd.DataFrame(
        {
            "candidate_id": ["a"], "symbol": ["X"], "side": ["long"],
            "decision_ts": [decision], "feature_cutoff_ts": [decision],
            "entry_ts": [decision], "label_end_ts": [decision + pd.Timedelta(hours=12)],
            "label_available_ts": [decision + pd.Timedelta(hours=12)],
            "execution_policy_id": ["p"], "cost_model_id": ["c"],
            "execution_geometry_id": ["g"], "execution_exact_h12_gross_bps": [150.0],
            "execution_exact_h12_cost_bps": [100.0], "execution_exact_h12_net_bps": [50.0],
        }
    )


def test_primary_contract_checks_exact_h12_and_cost_once() -> None:
    checks = validate_primary_labels(_primary(), {"horizon_minutes": 720})
    assert all(row["passed"] for row in checks)


def test_supportive_contract_flags_missing_explicit_support_metadata() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["a"],
        "__peak_mfe_atr_12h__": [1.0],
        "__time_to_first_meaningful_mfe_hours_12h__": [2.0],
        "__mae_before_meaningful_mfe_atr_12h__": [0.2],
        "__bars_before_price_stops_decreasing_12h__": [10.0],
        "__future_slope_atr_per_hour_12h__": [0.1],
    })
    result = validate_supportive_labels(frame)
    assert not next(row for row in result if row["check"] == "supportive_explicit_metadata")["passed"]


def test_oof_manifest_requires_candidate_level_lineage() -> None:
    folds = pd.DataFrame({
        "oof_fold": ["train", "test"], "fold_order": [0, 1],
        "start_utc": ["2024-01-01", "2024-02-01"],
        "end_exclusive_utc": ["2024-02-01", "2024-03-01"],
        "protocol_role": ["train", "test"],
    })
    predictions = pd.DataFrame({
        "prediction_fit_end_ts": ["2024-01-31T23:00:00Z"],
        "candidate_decision_min": ["2024-02-01T00:00:00Z"],
        "candidate_decision_max": ["2024-02-28T00:00:00Z"],
        "is_oof": [True], "rows": [1],
    })
    result = validate_fold_and_oof(folds, predictions)
    assert not next(row for row in result if row["check"] == "aggregate_oof_only_warning")["passed"]


def test_global_tail_is_pooled_and_requires_positive_economics() -> None:
    summary = pd.DataFrame({
        "target_arm": ["T1"], "support_stage": ["S0"],
        "selection_basis": ["pooled_global_post_score_top_k"],
        "top_k_fraction": [0.1], "population_rows": [100], "selected_rows": [10],
        "global_topk_net_bps": [-5.0], "latest_month_topk_net_bps": [-5.0],
        "months_selected": [2],
    })
    metrics = pd.DataFrame({"scope": ["pooled_global_top"], "fraction": [0.1], "net_bps": [-5.0]})
    result = validate_global_tail(summary, metrics)
    assert all(next(row for row in result if row["check"] == name)["passed"] for name in ("global_tail_is_pooled", "global_tail_row_count_exact"))
    assert not next(row for row in result if row["check"] == "supportive_global_tail_positive")["passed"]


def test_vectorized_support_metadata_has_all_head_suffixes(tmp_path) -> None:
    frame = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__path_auxiliary_target_valid__": [1, 1],
        "__time_to_first_meaningful_mfe_target_valid__": [1, 0],
        "__meaningful_mfe_reached_12h__": [1, 0],
    })
    out = tmp_path / "support.parquet"
    metadata = materialize_supportive_metadata(frame, out)
    assert metadata["rows"] == 2
    written = pd.read_parquet(out)
    assert written.filter(regex="__valid$").shape[1] == 5
    assert written.filter(regex="__condition_met$").shape[1] == 5
    assert written.filter(regex="__censored$").shape[1] == 5
    assert written.filter(regex="__support_count$").shape[1] == 5


def test_candidate_oof_scores_bind_to_execution_feature_lineage(tmp_path) -> None:
    source = tmp_path / "oof.parquet"
    predictions = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__decision_ts__": pd.to_datetime(["2024-02-01T00:00:00Z", "2024-02-01T01:00:00Z"]),
        "prediction_fit_end_ts": pd.to_datetime(["2024-01-31T23:00:00Z"] * 2),
        "prediction_generated_ts": pd.to_datetime(["2024-02-01T00:00:00Z", "2024-02-01T01:00:00Z"]),
        "oof_fold": ["test", "test"], "target_arm": ["T1", "T1"],
        "support_stage": ["S1", "S1"], "score": [0.2, 0.3],
        "prediction_model_id": ["m", "m"], "prediction_fold_id": ["test", "test"],
    })
    predictions.to_parquet(source, index=False)
    lineage = build_execution_oof_lineage_manifest(predictions, source)
    assert len(lineage) == 1
    assert lineage.iloc[0]["feature_name"] == "support_oof__T1__S1__score"
    feature_manifest = pd.DataFrame({
        "model_layer": ["execution"], "model_side": ["all"], "feature_name": [lineage.iloc[0]["feature_name"]],
        "semantic_class": ["MODEL_DERIVED"], "eligibility_status": ["ELIGIBLE_RESEARCH_OOF"],
        "eligible_now": [True], "requires_prediction_lineage_audit": [True],
    })
    feature_manifest["eligible_if_prediction_lineage_audited"] = True
    checks = validate_feature_manifest(feature_manifest, lineage)
    assert next(row for row in checks if row["check"] == "execution_derived_features_have_lineage")["passed"]
