from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_causal_admission import Causal21dAdmissionSpec
from extreme_price_movements.stage_i_meta_target_execution import (
    StageIMetaTargetExecutionError,
    file_sha256,
    run_pooled_global_meta_target_evaluation,
    run_side_meta_target_funnel,
)
from extreme_price_movements.stage_i_meta_target_funnel import (
    default_meta_target_specs,
    focused_quantile_meta_target_specs,
)
from extreme_price_movements.stage_i_feature_selection import (
    STAGE_I_META_BASE_OOF_HANDOFF_FEATURES,
)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _inputs(
    tmp_path: Path, rows_per_side: int = 180, invalid_placeholder: float | None = None,
) -> tuple[Path, Path, Path]:
    selector, base_root, meta_root = tmp_path / "selector", tmp_path / "base", tmp_path / "meta"
    selector.mkdir()
    _write_json(selector / "manifest.json", {"status": "complete"})
    _write_json(selector / "selector_feature_contract.json", {"schema": "test"})
    pieces = []
    for side in ("long", "short"):
        signal = pd.date_range("2023-01-01", periods=rows_per_side, freq="12h", tz="UTC")
        x = np.linspace(-1.0, 1.0, rows_per_side)
        piece = pd.DataFrame({
            "candidate_id": [f"{side}-{index}" for index in range(rows_per_side)],
            "__ts__": signal, "__symbol__": ["BTC" if side == "long" else "ETH"] * rows_per_side,
            "side_name": side, "label_available_ts": signal + pd.Timedelta(hours=13),
            "exact_net_bps": 100.0 * np.sin(np.arange(rows_per_side) / 11.0) + 35.0 * x,
            # Synthetic input explicitly declares that every fixture path is
            # valid; production paths fail closed if this provenance is absent.
            "target_invalid": False,
            "label_valid": True,
            "path_complete": True,
        })
        if invalid_placeholder is not None and side == "long":
            # This is deliberately finite and extreme: it catches a path that
            # is quietly treated as an ordinary economic outcome.
            index = min(100, rows_per_side - 1)
            piece.loc[index, "target_invalid"] = True
            piece.loc[index, "label_valid"] = False
            piece.loc[index, "path_complete"] = False
            piece.loc[index, "exact_net_bps"] = float(invalid_placeholder)
        pieces.append(piece)
    ledger = pd.concat(pieces, ignore_index=True)
    ledger.to_parquet(selector / "selector_ledger.parquet", index=False)
    ledger.loc[:, ["candidate_id", "__ts__", "__symbol__"]].to_parquet(selector / "selector_features.parquet", index=False)
    for side in ("long", "short"):
        local = ledger.loc[ledger.side_name.eq(side)].reset_index(drop=True)
        # Oscillating score keeps the canonical global top-30 candidate stream
        # time-spread enough for the tiny chronological fixture.
        x = np.sin(np.arange(len(local)) / 5.0)
        clear = 1.0 / (1.0 + np.exp(-x))
        adverse = 1.0 - clear
        weak = np.full(len(local), 0.2)
        scale = adverse + weak + clear
        probability = np.column_stack([adverse / scale, weak / scale, clear / scale])
        base = local.copy()
        base["r3_p_adverse"], base["r3_p_weak"], base["r3_p_clear"] = probability[:, 0], probability[:, 1], probability[:, 2]
        base["r3_opportunity_score"] = probability[:, 2] - probability[:, 0]
        base_dir = base_root / side
        base_dir.mkdir(parents=True)
        base_path = base_dir / "selector_base_oof.parquet"
        base.to_parquet(base_path, index=False)
        base_manifest = {
            "schema": "stage_i_base_feature_selection_v1", "status": "complete", "side": side,
            "selector_base_oof_sha256": file_sha256(base_path),
            "selector_sample_manifest_sha256": file_sha256(selector / "manifest.json"),
            "selector_feature_contract_sha256": file_sha256(selector / "selector_feature_contract.json"),
        }
        _write_json(base_dir / "manifest.json", base_manifest)
        meta_dir = meta_root / side
        meta_dir.mkdir(parents=True)
        candidate = local.loc[:, ["candidate_id", "__ts__", "__symbol__", "side_name"]].copy()
        candidate["decision_ts"] = pd.to_datetime(candidate["__ts__"], utc=True) + pd.Timedelta(hours=1)
        candidate["r3_opportunity_score"] = base.r3_opportunity_score.to_numpy(float)
        selected = np.zeros(len(candidate), dtype=bool)
        count = int(np.ceil(0.30 * len(candidate)))
        selected[np.argsort(candidate.r3_opportunity_score.to_numpy(), kind="stable")[-count:]] = True
        candidate["selected_base_candidate"] = selected
        candidate["base_candidate_fraction"] = 0.30
        candidate["ranking_scope"] = "side_local_global_over_strict_oof_development_rows; never_per_timestamp"
        candidate_path = meta_dir / "base_candidate_handoff_audit.parquet"
        candidate.to_parquet(candidate_path, index=False)
        _write_json(meta_dir / "manifest.json", {
            "schema": "stage_i_meta_feature_selection_v1", "status": "complete", "side": side,
            "selected_features": list(STAGE_I_META_BASE_OOF_HANDOFF_FEATURES),
            "selected_feature_contract": list(STAGE_I_META_BASE_OOF_HANDOFF_FEATURES),
            "required_same_side_base_oof_handoff_features": list(
                STAGE_I_META_BASE_OOF_HANDOFF_FEATURES
            ),
            "best_params": {"n_estimators": 5, "random_state": 7},
            "selector_sample_manifest_sha256": file_sha256(selector / "manifest.json"),
            "selector_feature_contract_sha256": file_sha256(selector / "selector_feature_contract.json"),
            "base_selector_oof_sha256": file_sha256(base_path),
            "base_selector_manifest_sha256": file_sha256(base_dir / "manifest.json"),
            "base_candidate_handoff_audit_sha256": file_sha256(candidate_path),
            "base_candidate_fraction": 0.30,
            "base_candidate_ranking_scope": "side_local_global_over_strict_oof_development_rows; never_per_timestamp",
        })
    return selector, base_root, meta_root


def _predictor(train_x, target, weight, valid_x, spec):
    if spec.family in {"reliability", "overestimate_risk"}:
        probability = np.full(len(valid_x), np.average(target, weights=weight))
        return np.column_stack([1.0 - probability, probability])
    if spec.family == "ordinal_residual":
        counts = np.bincount(target.astype(int), weights=weight, minlength=4).astype(float)
        probability = counts / counts.sum()
        return np.tile(probability, (len(valid_x), 1))
    if spec.family == "quantile_ordinal_residual":
        counts = np.bincount(target.astype(int), weights=weight, minlength=3).astype(float)
        probability = counts / counts.sum()
        return np.tile(probability, (len(valid_x), 1))
    return np.zeros(len(valid_x), dtype=float)


def test_checkpointed_side_funnel_runs_all_predeclared_arms_and_resumes(tmp_path: Path) -> None:
    selector, base, meta = _inputs(tmp_path)
    output = tmp_path / "out" / "long"
    manifest = run_side_meta_target_funnel(
        selector_dir=selector, base_selection_dir=base, meta_selection_dir=meta,
        output_dir=output, side="long", n_validation_folds=3, min_train_rows=20,
        predictor=_predictor,
    )
    assert manifest["scope"] == "side_local_diagnostic_until_two_side_common_bps_mapping"
    assert manifest["arm_order"] == [spec.arm_id for spec in default_meta_target_specs()]
    for arm_id in manifest["arm_order"]:
        assert (output / "arms" / arm_id / "oof_predictions.parquet").is_file()
        assert (output / "arms" / arm_id / "metrics.parquet").is_file()
    assert manifest["base_candidate_fraction"] == 0.30
    assert "fixed_feature_target_isolation_diagnostic" in manifest[
        "feature_contract_disposition"
    ]
    handoff = pd.read_parquet(output / "base_candidate_handoff_audit.parquet")
    assert handoff.enforced_by_meta_target_funnel.all()
    assert handoff.selected_base_candidate.sum() == int(np.ceil(0.30 * len(handoff)))
    tercile = "T3Q_fold_quantile_ordinal_residual"
    prediction = pd.read_parquet(output / "arms" / tercile / "oof_predictions.parquet")
    assert {
        "prediction_class_0", "prediction_class_1", "prediction_class_2",
        "prior_prediction_class_0", "prior_prediction_class_1",
        "prior_prediction_class_2", "probability_lower_residual_tercile",
        "probability_middle_residual_tercile",
        "probability_upper_residual_tercile",
        "prior_probability_lower_residual_tercile",
        "prior_probability_middle_residual_tercile",
        "prior_probability_upper_residual_tercile",
    }.issubset(prediction.columns)
    provenance = pd.read_parquet(output / "arms" / tercile / "fold_provenance.parquet")
    assert {
        "residual_q33_bps", "residual_q67_bps", "class_0_support",
        "class_1_support", "class_2_support", "zero_in_middle_tercile",
        "fold_semantic_valid",
        "class_0_residual_location_bps", "class_1_residual_location_bps",
        "class_2_residual_location_bps", "class_location_method",
        "class_0_residual_median_bps", "class_1_residual_median_bps",
        "class_2_residual_median_bps",
        "class_0_location_uncertainty_bps", "class_1_location_uncertainty_bps",
        "class_2_location_uncertainty_bps",
        "class_0_training_prior", "class_1_training_prior",
        "class_2_training_prior",
    }.issubset(provenance.columns)
    metrics = pd.read_parquet(output / "arms" / tercile / "metrics.parquet")
    assert {
        "target_log_loss", "target_multiclass_brier", "target_rps",
        "target_confusion_json", "target_calibration_ece_10",
        "target_prior_accuracy", "target_majority_accuracy",
        "target_prior_log_loss", "target_prior_multiclass_brier",
        "target_prior_rps", "target_log_loss_skill", "target_brier_skill",
        "target_rps_skill", "target_balanced_accuracy",
        "target_ordinal_expected_class_spearman",
    }.issubset(metrics.columns)
    overall_metrics = pd.read_parquet(output / "metrics.parquet")
    assert "C4_T3Q_fold_prior_conversion" in set(overall_metrics.arm_id)
    map_control = overall_metrics.loc[
        overall_metrics.arm_id.eq("C1_causal_map_only")
    ].sort_values("top_fraction")
    prior_control = overall_metrics.loc[
        overall_metrics.arm_id.eq("C4_T3Q_fold_prior_conversion")
    ].sort_values("top_fraction")
    np.testing.assert_allclose(
        prior_control.net_bps_per_trade,
        map_control.net_bps_per_trade,
        equal_nan=True,
    )
    assert {
        "original_population_rows", "requested_topk_rows_original_population",
        "topk_saturated_due_candidate_or_admission_support", "ranking_tie_policy",
    }.issubset(overall_metrics.columns)
    assert overall_metrics.original_population_rows.eq(180).all()
    assert manifest["T3Q_promotion_semantic_gate"]["required"] == (
        "residual_q33_bps < 0 <= residual_q67_bps in every OOF fold"
    )
    resumed = run_side_meta_target_funnel(
        selector_dir=selector, base_selection_dir=base, meta_selection_dir=meta,
        output_dir=output, side="long", n_validation_folds=3, min_train_rows=20,
        predictor=_predictor, resume=True,
    )
    assert resumed["restart_status"] == "reused_verified_complete"


def test_focused_quantile_arm_set_runs_without_the_broad_grid(tmp_path: Path) -> None:
    selector, base, meta = _inputs(tmp_path)
    output = tmp_path / "focused" / "long"
    seen_tercile_columns = []

    def guarded_predictor(train_x, target, weight, valid_x, spec):
        if spec.family == "quantile_ordinal_residual":
            seen_tercile_columns.append(tuple(train_x.columns))
            assert not any(
                "prequential" in column.lower()
                or "expected_net_bps" in column.lower()
                or "converted" in column.lower()
                for column in train_x.columns
            )
            assert {"r3_p_adverse", "r3_p_weak", "r3_p_clear", "r3_opportunity_score"}.issubset(train_x.columns)
        return _predictor(train_x, target, weight, valid_x, spec)

    manifest = run_side_meta_target_funnel(
        selector_dir=selector,
        base_selection_dir=base,
        meta_selection_dir=meta,
        output_dir=output,
        side="long",
        n_validation_folds=3,
        min_train_rows=12,
        predictor=guarded_predictor,
        specs=focused_quantile_meta_target_specs(),
    )
    assert manifest["arm_order"] == [
        "T3Q_fold_quantile_ordinal_residual", "C3_current_map_huber"
    ]
    assert set(path.name for path in (output / "arms").iterdir()) == set(
        manifest["arm_order"]
    )
    assert seen_tercile_columns
    assert "prequential_base_expected_net_bps" not in manifest[
        "classifier_features_by_arm"
    ]["T3Q_fold_quantile_ordinal_residual"]
    assert "conversion_head" in manifest["T3Q_anchor_semantics"]


def test_finite_invalid_placeholder_never_enters_anchor_or_meta_evaluation(
    tmp_path: Path,
) -> None:
    """Changing an invalid finite net must not move any valid OOF row."""
    roots: list[Path] = []
    ledgers: list[pd.DataFrame] = []
    for name, placeholder in (("left", -1_000_000.0), ("right", 1_000_000.0)):
        work = tmp_path / name
        work.mkdir()
        selector, base, meta = _inputs(work, invalid_placeholder=placeholder)
        output = work / "out" / "long"
        run_side_meta_target_funnel(
            selector_dir=selector, base_selection_dir=base, meta_selection_dir=meta,
            output_dir=output, side="long", n_validation_folds=3,
            min_train_rows=20, predictor=_predictor,
            specs=focused_quantile_meta_target_specs(),
        )
        roots.append(output)
        ledgers.append(pd.read_parquet(output / "full_oof_reference_ledger.parquet"))
    for ledger in ledgers:
        assert "long-100" not in set(ledger.candidate_id)
        assert ledger.valid_resolved_target.all()
        assert ledger.mapping_reference_eligible.all()
    columns = [
        "candidate_key", "prequential_base_expected_net_bps", "exact_net_bps",
    ]
    pd.testing.assert_frame_equal(
        ledgers[0].loc[:, columns].reset_index(drop=True),
        ledgers[1].loc[:, columns].reset_index(drop=True),
        check_exact=True,
    )


def test_meta_execution_fails_closed_without_target_validity_provenance(
    tmp_path: Path,
) -> None:
    selector, base, meta = _inputs(tmp_path)
    ledger_path = selector / "selector_ledger.parquet"
    ledger = pd.read_parquet(ledger_path).drop(
        columns=["target_invalid", "label_valid", "path_complete"]
    )
    ledger.to_parquet(ledger_path, index=False)
    with pytest.raises(StageIMetaTargetExecutionError, match="target validity provenance"):
        run_side_meta_target_funnel(
            selector_dir=selector, base_selection_dir=base, meta_selection_dir=meta,
            output_dir=tmp_path / "out" / "long", side="long", n_validation_folds=3,
            min_train_rows=20, predictor=_predictor,
            specs=focused_quantile_meta_target_specs(),
        )


def test_two_side_finalizer_maps_to_common_bps_before_one_global_rank(tmp_path: Path) -> None:
    selector, base, meta = _inputs(tmp_path, rows_per_side=220)
    roots = {}
    for side in ("long", "short"):
        roots[side] = tmp_path / "out" / side
        run_side_meta_target_funnel(
            selector_dir=selector, base_selection_dir=base, meta_selection_dir=meta,
            output_dir=roots[side], side=side, n_validation_folds=3,
            min_train_rows=20, predictor=_predictor,
        )
    pooled = run_pooled_global_meta_target_evaluation(
        long_dir=roots["long"], short_dir=roots["short"],
        output_dir=tmp_path / "out" / "pooled_global",
        admission_spec=Causal21dAdmissionSpec(min_reference_rows=4, bins=4),
    )
    assert pooled["scope"] == "pooled_global_after_side_local_causal_common_bps_mapping"
    assert "raw side scores are diagnostic only" in pooled["comparability_boundary"]
    metrics = pd.read_parquet(tmp_path / "out" / "pooled_global" / "pooled_global_metrics.parquet")
    assert metrics.scope.eq("pooled_global_common_bps_after_21d_admission").all()
    assert set(metrics.top_fraction) == {0.01, 0.05, 0.10, 0.20}
    assert {"selected_long_rows", "selected_short_rows"}.issubset(metrics.columns)
    assert (
        metrics.selected_long_rows + metrics.selected_short_rows
        == metrics.selected_rows
    ).all()
    assert (tmp_path / "out" / "pooled_global" / "paired_week_arm_vs_raw.parquet").is_file()
    assert {
        "paired_week_delta_q025_bps",
        "worst_month_selected_rows",
        "worst_fold_selected_rows",
        "requested_topk_rows",
        "topk_saturated_due_admission",
        "unique_symbols",
        "max_symbol_share",
        "symbol_hhi",
        "max_day_share",
        "max_week_share",
        "trades_per_day",
        "positive_weeks",
        "negative_weeks",
        "positive_months",
        "negative_months",
    }.issubset(metrics.columns)
    assert (metrics.selected_rows <= metrics.requested_topk_rows).all()
    assert (
        metrics.topk_saturated_due_admission
        == (metrics.admitted_rows < metrics.requested_topk_rows)
    ).all()
    assert metrics.loc[metrics.selected_rows.gt(0), "unique_symbols"].ge(1).all()
    mapped = pd.read_parquet(
        tmp_path / "out" / "pooled_global" / "arms"
        / "C0_raw_base_exact_noop" / "mapped_predictions.parquet"
    )
    assert "__symbol__" in mapped.columns


def test_pooled_finalizer_handles_zero_admissions_with_empty_paired_week_schema(
    tmp_path: Path,
) -> None:
    selector, base, meta = _inputs(tmp_path, rows_per_side=180)
    roots = {}
    for side in ("long", "short"):
        roots[side] = tmp_path / "side" / side
        run_side_meta_target_funnel(
            selector_dir=selector,
            base_selection_dir=base,
            meta_selection_dir=meta,
            output_dir=roots[side],
            side=side,
            n_validation_folds=3,
            min_train_rows=12,
            predictor=_predictor,
            specs=focused_quantile_meta_target_specs(),
        )
    output = tmp_path / "pooled_empty"
    result = run_pooled_global_meta_target_evaluation(
        long_dir=roots["long"],
        short_dir=roots["short"],
        output_dir=output,
        # No fold can accumulate this much causal 21-day reference support.
        admission_spec=Causal21dAdmissionSpec(
            min_reference_rows=100_000, bins=4
        ),
        bootstrap_draws=100,
    )
    assert result["status"] == "complete"
    assert result["decision"]["winner_arm_id"] == "C0_raw_base_exact_noop"
    paired = pd.read_parquet(output / "paired_week_arm_vs_raw.parquet")
    assert paired.empty
    assert list(paired.columns) == [
        "arm_id", "top_fraction", "week", "arm_net_sum_bps",
        "arm_selected_rows", "raw_net_sum_bps", "raw_selected_rows",
    ]
    metrics = pd.read_parquet(output / "pooled_global_metrics.parquet")
    assert metrics.selected_rows.eq(0).all()
    assert metrics.paired_week_blocks.eq(0).all()
    assert metrics.paired_week_bootstrap_draws.eq(0).all()
    assert metrics.paired_week_delta_q025_bps.isna().all()


def test_resume_rejects_mutated_arm_checkpoint(tmp_path: Path) -> None:
    selector, base, meta = _inputs(tmp_path)
    output = tmp_path / "out" / "long"
    run_side_meta_target_funnel(
        selector_dir=selector, base_selection_dir=base, meta_selection_dir=meta,
        output_dir=output, side="long", n_validation_folds=3, min_train_rows=20,
        predictor=_predictor,
    )
    path = output / "arms" / default_meta_target_specs()[0].arm_id / "oof_predictions.parquet"
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(StageIMetaTargetExecutionError, match="checkpoint drift"):
        run_side_meta_target_funnel(
            selector_dir=selector, base_selection_dir=base, meta_selection_dir=meta,
            output_dir=output, side="long", n_validation_folds=3,
            min_train_rows=20, predictor=_predictor, resume=True,
        )


def test_full_population_oof_reference_is_candidate_trained_and_hash_bound(
    tmp_path: Path,
) -> None:
    selector, base, meta = _inputs(tmp_path, rows_per_side=220)
    output = tmp_path / "full_reference" / "long"
    calls: list[tuple[int, int]] = []

    def observing_predictor(train_x, target, weight, valid_x, spec):
        calls.append((len(train_x), len(valid_x)))
        return _predictor(train_x, target, weight, valid_x, spec)

    manifest = run_side_meta_target_funnel(
        selector_dir=selector, base_selection_dir=base, meta_selection_dir=meta,
        output_dir=output, side="long", n_validation_folds=3, min_train_rows=20,
        predictor=observing_predictor,
        specs=focused_quantile_meta_target_specs(),
    )
    full = pd.read_parquet(output / "full_oof_reference_ledger.parquet")
    action = pd.read_parquet(output / "evaluation_ledger.parquet")
    assert len(full) > len(action) > 0
    assert full.action_candidate.any() and full.mapping_reference_only.any()
    assert np.array_equal(
        full.action_candidate.to_numpy(bool),
        ~full.mapping_reference_only.to_numpy(bool),
    )
    assert full.loc[full.action_candidate, ["candidate_key", "meta_fold_id"]].reset_index(
        drop=True
    ).equals(action.loc[:, ["candidate_key", "meta_fold_id"]].reset_index(drop=True))
    assert manifest["full_oof_reference_ledger_sha256"] == file_sha256(
        output / "full_oof_reference_ledger.parquet"
    )
    # The callback sees only frozen candidates for fitting, but every call
    # scores the full contemporaneous fold population.
    all_rows = pd.read_parquet(selector / "selector_ledger.parquet")
    handoff = pd.read_parquet(meta / "long" / "base_candidate_handoff_audit.parquet")
    all_rows = all_rows.loc[all_rows.side_name.eq("long")].merge(
        handoff.loc[:, ["candidate_id", "selected_base_candidate"]],
        on="candidate_id", validate="one_to_one",
    )
    expected_calls = []
    for fold_id in sorted(full.meta_fold_id.unique()):
        validation = full.loc[full.meta_fold_id.eq(fold_id)]
        start = pd.to_datetime(validation.decision_ts, utc=True).min()
        expected_train = int(
            (
                all_rows.selected_base_candidate.astype(bool)
                & pd.to_datetime(all_rows.label_available_ts, utc=True).lt(start)
            ).sum()
        )
        expected_calls.append((expected_train, len(validation)))
    assert calls == expected_calls * 2
    provenance = pd.read_parquet(
        output / "arms" / "T3Q_fold_quantile_ordinal_residual" / "fold_provenance.parquet"
    )
    assert provenance.full_population_scored.all()
    assert (provenance.validation_full_population_rows > provenance.validation_action_candidate_rows).any()
    assert (provenance.train_rows == provenance.train_action_candidate_rows).all()
    assert (
        pd.to_datetime(provenance.train_max_label_available_utc, utc=True)
        < pd.to_datetime(provenance.validation_start_utc, utc=True)
    ).all()
    arm_root = output / "arms" / "T3Q_fold_quantile_ordinal_residual"
    arm_manifest = json.loads((arm_root / "manifest.json").read_text())
    full_prediction = pd.read_parquet(arm_root / "full_oof_reference_predictions.parquet")
    action_prediction = pd.read_parquet(arm_root / "oof_predictions.parquet")
    assert arm_manifest["full_oof_reference_predictions_sha256"] == file_sha256(
        arm_root / "full_oof_reference_predictions.parquet"
    )
    assert full_prediction.loc[full_prediction.action_candidate, ["candidate_key", "fold_id"]].reset_index(
        drop=True
    ).equals(action_prediction.loc[:, ["candidate_key", "fold_id"]].reset_index(drop=True))


def test_pooled_mapping_never_selects_reference_only_rows_and_has_no_raw_fallback(
    tmp_path: Path,
) -> None:
    selector, base, meta = _inputs(tmp_path, rows_per_side=220)
    roots = {}
    for side in ("long", "short"):
        roots[side] = tmp_path / "full" / side
        run_side_meta_target_funnel(
            selector_dir=selector, base_selection_dir=base, meta_selection_dir=meta,
            output_dir=roots[side], side=side, n_validation_folds=3,
            min_train_rows=20, predictor=_predictor,
            specs=focused_quantile_meta_target_specs(),
        )
    pooled_root = tmp_path / "full" / "pooled"
    run_pooled_global_meta_target_evaluation(
        long_dir=roots["long"], short_dir=roots["short"], output_dir=pooled_root,
        admission_spec=Causal21dAdmissionSpec(min_reference_rows=4, bins=4),
    )
    mapped = pd.read_parquet(
        pooled_root / "arms" / "C0_raw_base_exact_noop" / "mapped_predictions.parquet"
    )
    assert {"action_candidate", "mapping_reference_only", "model_action_admitted"}.issubset(mapped.columns)
    assert not mapped.loc[mapped.mapping_reference_only, "final_action_admitted"].any()
    # Mutating the full reference invalidates pooled admission rather than
    # silently falling back to the selected action-only ledger.
    damaged = roots["long"] / "full_oof_reference_ledger.parquet"
    damaged.write_bytes(damaged.read_bytes() + b"drift")
    insufficient_root = tmp_path / "full" / "insufficient"
    result = run_pooled_global_meta_target_evaluation(
        long_dir=roots["long"], short_dir=roots["short"], output_dir=insufficient_root,
        admission_spec=Causal21dAdmissionSpec(min_reference_rows=4, bins=4),
    )
    assert result["status"] == "ADMISSION_REFERENCE_INSUFFICIENT"
    assert not (insufficient_root / "pooled_global_metrics.parquet").exists()
