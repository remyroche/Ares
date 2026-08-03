from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import json

from scripts.run_execution_ev_competing_risk_context_add_drop import (
    ARMS,
    ARM_CANDIDATE_CONTEXT,
    ARM_INCLUDE_ALPHA,
    ALL_CANDIDATE_CONTEXT,
    ALLOWED_SIDECAR_MODEL_FIELDS,
    CANDIDATE_CONTEXT_FIELDS,
    DAE_BLOCK_FIELDS,
    DIAGNOSTIC_ONLY_CANDIDATE_FIELDS,
    FORBIDDEN_REPRESENTATION_FIELDS,
    GMM_GEOMETRY_BLOCK_FIELDS,
    RAW_TRANSITION_BLOCK_FIELDS,
    begin_atomic_output,
    build_final_context_matrix,
    CONTEXT_CHANNELS,
    evaluate_global_topk,
    forbid_action_features,
    safe_oof_join,
    train_only_empirical_cdf,
    validate_primary100_contract,
    validate_context_sidecar_source,
    validate_sidecar_representation_missingness,
    resolve_runtime_controls,
)
from scripts.run_execution_ev_competing_risk_simplex_ablation import CLASS_NAMES


def _identity(rows: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-06-01", periods=rows, freq="h", tz="UTC"),
            "__symbol__": ["A", "B", "A", "B", "A", "B"][:rows],
            "side_name": ["long", "short", "long", "short", "long", "short"][:rows],
            "candidate_id": [f"candidate-{value}" for value in range(rows)],
        }
    )


def _panel(rows: int = 6) -> pd.DataFrame:
    result = _identity(rows)
    result["execution_decision_utc"] = result["__ts__"]
    result["label_resolution_utc"] = result["__ts__"] + pd.Timedelta(hours=12)
    result["execution_gross_ev_12h"] = np.linspace(-.03, .04, rows)
    result["execution_cost_return"] = .01
    result["execution_net_ev_12h"] = result["execution_gross_ev_12h"] - .01
    result["base_oof_score"] = np.linspace(-.5, .5, rows)
    classes = np.arange(rows) % 3
    result["competing_risk_class"] = classes
    for number, name in enumerate(CLASS_NAMES):
        result[name] = (classes == number).astype(int)
    return result


def test_context_arm_lattice_is_prediction_only_and_incremental() -> None:
    assert tuple(ARMS) == (
        "base_only",
        "direct_meta_only",
        "direct_meta_plus_alpha",
        "plus_clean_probability",
        "plus_competing_risk",
        "plus_clean_payoff",
        "plus_clean_value",
        "plus_clean_value_rank",
        "plus_cutoff_context",
        "plus_timestamp_relative_context",
        "plus_archetype_relative_context",
        "plus_rank_group_context",
        "plus_candidate_context_all",
        "plus_candidate_context_and_clean_value_rank",
        "plus_dae",
        "plus_gmm_geometry",
        "plus_raw_transition",
        "plus_candidate_context_clean_value_rank_dae",
        "plus_candidate_context_clean_value_rank_gmm_geometry",
        "plus_candidate_context_clean_value_rank_raw_transition",
    )
    assert set().union(*map(set, ARMS.values())).issubset(CONTEXT_CHANNELS)
    assert ARMS["plus_clean_value_rank"][-1] == "pred_clean_rank"
    assert ARM_CANDIDATE_CONTEXT["plus_candidate_context_all"] == ALL_CANDIDATE_CONTEXT
    assert CANDIDATE_CONTEXT_FIELDS["cutoff_context"] == (
        "base_margin_to_cutoff", "base_margin_to_cutoff_z",
    )
    assert CANDIDATE_CONTEXT_FIELDS["timestamp_relative_context"] == (
        "base_candidate_rank_pct_timestamp_side", "base_score_z_timestamp_side",
    )
    assert DIAGNOSTIC_ONLY_CANDIDATE_FIELDS == ("base_candidate_rank_timestamp_side",)
    assert DIAGNOSTIC_ONLY_CANDIDATE_FIELDS[0] not in ALL_CANDIDATE_CONTEXT
    assert ARM_INCLUDE_ALPHA["direct_meta_only"] is False
    assert ARM_INCLUDE_ALPHA["direct_meta_plus_alpha"] is True
    assert set(ARMS) == set(ARM_INCLUDE_ALPHA) == set(ARM_CANDIDATE_CONTEXT)
    assert ARM_CANDIDATE_CONTEXT["plus_dae"] == DAE_BLOCK_FIELDS
    assert ARM_CANDIDATE_CONTEXT["plus_gmm_geometry"] == GMM_GEOMETRY_BLOCK_FIELDS
    assert ARM_CANDIDATE_CONTEXT["plus_raw_transition"] == RAW_TRANSITION_BLOCK_FIELDS
    assert set(FORBIDDEN_REPRESENTATION_FIELDS).isdisjoint(ALLOWED_SIDECAR_MODEL_FIELDS)
    assert not any("mae" in channel or "wait" in channel for channel in CONTEXT_CHANNELS)


def test_action_layer_features_are_rejected_not_silently_dropped() -> None:
    assert forbid_action_features(["atr", "regime_transition_probability"]) == [
        "atr",
        "regime_transition_probability",
    ]
    with pytest.raises(ValueError, match="timing/MAE/wait/target-price"):
        forbid_action_features(["atr", "entry_timing_probability"])
    with pytest.raises(ValueError, match="timing/MAE/wait/target-price"):
        forbid_action_features(["mae_before_meaningful_mfe_atr"])


def test_primary100_contract_proves_12h_simplex_and_cost_identity() -> None:
    panel = _panel()
    validate_primary100_contract(panel, expected_rows=len(panel))
    bad = panel.copy()
    bad.loc[0, "execution_net_ev_12h"] += .001
    with pytest.raises(ValueError, match="gross-cost=net"):
        validate_primary100_contract(bad, expected_rows=len(bad))
    bad = panel.copy()
    bad.loc[0, "label_resolution_utc"] += pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="exactly 12h"):
        validate_primary100_contract(bad, expected_rows=len(bad))


def test_safe_oof_join_requires_outer_split_and_preserves_frozen_order() -> None:
    identity = _identity(4)
    anchor = identity.iloc[[2, 0]].copy().reset_index(drop=True)
    anchor.insert(0, "outer_split", "may_to_june")
    predicted = identity.iloc[[0, 2]].copy().reset_index(drop=True)
    predicted.insert(0, "outer_split", "may_to_june")
    predicted["p_clean"] = [.7, .2]
    joined = safe_oof_join(anchor, predicted, value_columns=["p_clean"])
    assert joined["candidate_id"].tolist() == ["candidate-2", "candidate-0"]
    assert joined["p_clean"].tolist() == [.2, .7]
    duplicate = pd.concat([predicted, predicted.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate split/identity"):
        safe_oof_join(anchor, duplicate, value_columns=["p_clean"])
    wrong_split = predicted.copy()
    wrong_split["outer_split"] = "june_to_july"
    with pytest.raises(ValueError, match="incomplete"):
        safe_oof_join(anchor, wrong_split, value_columns=["p_clean"])


def test_within_clean_rank_is_fixed_from_train_predictions_only() -> None:
    reference = np.array([.10, .20, .40, .90])
    evaluation = np.array([.05, .20, .50, 1.10])
    rank = train_only_empirical_cdf(reference, evaluation)
    assert np.allclose(rank, [.0, .5, .75, 1.0])
    # Moving an evaluation value must not alter another row's rank: this is
    # precisely what would fail if the evaluation distribution were used.
    changed = train_only_empirical_cdf(reference, np.array([.05, .20, 5.0, 1.10]))
    assert np.allclose(changed[:2], rank[:2])


def test_final_head_inputs_are_predicted_or_frozen_context_never_observed_labels() -> None:
    meta = pd.DataFrame({"meta_feature": [.1, .2]})
    predicted = pd.DataFrame({name: np.full(2, index + .1) for index, name in enumerate(CONTEXT_CHANNELS)})
    candidate = pd.DataFrame({name: np.full(2, index + .2) for index, name in enumerate(ALL_CANDIDATE_CONTEXT)})
    result = build_final_context_matrix(
        meta, np.array([.3, .4]), predicted,
        channels=("p_clean", "pred_clean_rank"),
        candidate_context=candidate,
        candidate_fields=("base_margin_to_cutoff",),
    )
    assert set(result.columns) == {
        "meta_feature", "__frozen_base_alpha_oof__", "__ctx_p_clean__",
        "__ctx_pred_clean_rank__", "__candidate_base_margin_to_cutoff__",
    }
    without_alpha = build_final_context_matrix(
        meta, np.array([.3, .4]), predicted, channels=(), include_alpha=False,
    )
    assert "__frozen_base_alpha_oof__" not in without_alpha.columns
    bad_meta = meta.assign(execution_net_ev_12h=[0.0, 0.0])
    with pytest.raises(ValueError, match="observed labels/payoffs"):
        build_final_context_matrix(bad_meta, np.array([.3, .4]), predicted, channels=())


def test_representation_nan_contract_allows_native_missing_only_when_unavailable() -> None:
    # A small representative block is sufficient to prove the availability
    # gate; the production loader validates all materialized fields.
    available = "gmm_representation_available"
    frame = pd.DataFrame({available: [1, 0], "dae_b16_00": [.1, np.nan]})
    validate_sidecar_representation_missingness(frame, ("dae_b16_00",))
    frame.loc[0, "dae_b16_00"] = np.nan
    with pytest.raises(ValueError, match="permitted only where availability=0"):
        validate_sidecar_representation_missingness(frame, ("dae_b16_00",))
    predicted = pd.DataFrame({name: np.full(2, .1) for name in CONTEXT_CHANNELS})
    sidecar = pd.DataFrame({available: [1, 0], "dae_b16_00": [.1, np.nan]})
    result = build_final_context_matrix(
        pd.DataFrame({"meta": [.1, .2]}), np.array([.2, .3]), predicted,
        channels=(), candidate_context=sidecar,
        candidate_fields=("dae_b16_00", available),
    )
    assert result["__candidate_dae_b16_00__"].isna().tolist() == [False, True]


def test_runtime_controls_reject_unknown_and_partial_grouped_july() -> None:
    available = ("may_to_june", "july_fold_1", "july_fold_2", "july_grouped_oof")
    arms, evaluations = resolve_runtime_controls(
        ("direct_meta_only",), ("may_to_june",),
        available_evaluations=available, grouped_july_folds=("july_fold_1", "july_fold_2"),
    )
    assert arms == ("direct_meta_only",) and evaluations == ("may_to_june",)
    with pytest.raises(ValueError, match="unknown context arms"):
        resolve_runtime_controls(("not_an_arm",), None, available_evaluations=available, grouped_july_folds=("july_fold_1", "july_fold_2"))
    with pytest.raises(ValueError, match="requires every"):
        resolve_runtime_controls(None, ("july_grouped_oof", "july_fold_1"), available_evaluations=available, grouped_july_folds=("july_fold_1", "july_fold_2"))


def test_global_topk_is_pooled_deterministic_and_reports_month_coverage() -> None:
    frame = _panel(6)
    frame["score"] = 1.0
    first = evaluate_global_topk(frame.sample(frac=1.0, random_state=9), "score", evaluation="toy")
    second = evaluate_global_topk(frame.sample(frac=1.0, random_state=19), "score", evaluation="toy")
    assert [row["selected_rows"] for row in first] == [1, 1, 1, 2]
    assert [row["net_ev_bps"] for row in first] == [row["net_ev_bps"] for row in second]
    assert all(row["population_rows"] == len(frame) for row in first)
    assert all("month_2026-06_rows" in row for row in first)
    assert all("timestamp" not in row for row in first)


def test_atomic_output_refuses_existing_artifacts(tmp_path) -> None:
    target = tmp_path / "context_result"
    partial = begin_atomic_output(target)
    assert partial.name.startswith(".context_result.partial-")
    assert partial.is_dir() and not target.exists()
    target.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        begin_atomic_output(target)


def test_context_sidecar_source_requires_manifest_hash_and_report_binding(tmp_path) -> None:
    source = tmp_path / "context.parquet"
    source.write_bytes(b"frozen-context")
    from scripts.run_execution_ev_competing_risk_context_add_drop import _sha256
    report = tmp_path / "report.json"
    report.write_text("{}")
    manifest = {
        "schema": "primary100_exact_outcome_free_context_sidecar_v1",
        "status": "MATERIALIZED_EXACT_OUTCOME_FREE_PRIMARY100_CONTEXT",
        "output": {"path": str(source), "sha256": _sha256(source), "rows": 134889},
        "report": {"path": str(report), "sha256": _sha256(report)},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    provenance = validate_context_sidecar_source(source)
    assert provenance["sha256"] == _sha256(source)
    manifest["output"]["sha256"] = "wrong"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="hash/path"):
        validate_context_sidecar_source(source)
