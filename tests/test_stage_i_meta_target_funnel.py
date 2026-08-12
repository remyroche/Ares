from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_i_meta_target_funnel import (
    MetaOOFArm,
    MetaTargetSpec,
    StageIMetaTargetError,
    current_huber_control_arm,
    default_meta_target_specs,
    focused_quantile_meta_target_specs,
    evaluate_meta_oof_arms,
    fit_meta_target,
    mandatory_control_arms,
    reconstruct_meta_action,
    run_strict_meta_target_arm,
    select_meta_arm_with_noop_gate,
)


def _frame(rows: int = 200) -> pd.DataFrame:
    decision = pd.date_range("2023-01-01", periods=rows, freq="12h", tz="UTC")
    raw = np.linspace(-1.0, 1.0, rows)
    mapped = 30.0 * raw + np.sin(np.arange(rows) / 7.0) * 10.0
    net = mapped + np.linspace(-150.0, 150.0, rows)
    return pd.DataFrame({
        "candidate_key": [f"long::{i}" for i in range(rows)],
        "side_name": "long",
        "decision_ts": decision,
        "label_available_ts": decision + pd.Timedelta(hours=12),
        "r3_opportunity_score": raw,
        "prequential_base_expected_net_bps": mapped,
        "exact_net_bps": net,
    })


def _fit(spec: MetaTargetSpec) -> tuple[pd.DataFrame, object]:
    frame = _frame()
    return frame, fit_meta_target(
        frame, spec, side="long", fit_before_utc="2024-01-01"
    )


def _controls(frame: pd.DataFrame, folds: np.ndarray) -> list[MetaOOFArm]:
    return [
        *mandatory_control_arms(frame, folds),
        current_huber_control_arm(frame, folds, np.zeros(len(frame))),
    ]


def test_default_funnel_covers_all_requested_target_families_and_values() -> None:
    specs = default_meta_target_specs()
    families = {spec.family for spec in specs}
    assert families == {
        "reliability", "overestimate_risk", "ordinal_residual",
        "quantile_ordinal_residual",
        "clipped_residual", "huber_residual",
    }
    reliability = [spec for spec in specs if spec.family == "reliability"]
    assert {(spec.hurdle_bps, spec.base_tail_fraction) for spec in reliability} == {
        (hurdle, fraction) for hurdle in (0.0, 25.0, 50.0) for fraction in (0.20, 0.30)
    }
    assert {spec.residual_clip_bps for spec in specs if spec.family == "clipped_residual"} == {50.0, 100.0, 200.0}


def test_reliability_is_tail_conditioned_and_produces_bounded_raw_score_correction() -> None:
    spec = MetaTargetSpec("reliable", "reliability", hurdle_bps=25.0, base_tail_fraction=0.20)
    frame, fit = _fit(spec)
    assert fit.sample_weight.sum() == pytest.approx(40.0)
    prediction = np.linspace(0.0, 1.0, len(frame))
    score, admitted = reconstruct_meta_action(frame, fit, prediction)
    raw = frame.r3_opportunity_score.to_numpy()
    assert admitted.all()
    assert np.max(np.abs(score - raw)) <= spec.correction_cap_score_std * fit.raw_base_scale + 1e-12
    # The score remains in raw base score space, not a bps/raw-unit mixture.
    assert np.max(np.abs(score)) < 2.0


@pytest.mark.parametrize("delta", [0.0, 25.0, 50.0])
def test_overestimate_target_is_map_relative_and_prediction_is_only_a_veto(delta: float) -> None:
    spec = MetaTargetSpec("risk", "overestimate_risk", hurdle_bps=delta)
    frame, fit = _fit(spec)
    expected = (
        frame.exact_net_bps.to_numpy() - frame.prequential_base_expected_net_bps.to_numpy() < -delta
    )
    np.testing.assert_array_equal(fit.target, expected)
    probability = np.linspace(0.0, 1.0, len(frame))
    score, admitted = reconstruct_meta_action(frame, fit, probability)
    np.testing.assert_array_equal(score, frame.r3_opportunity_score.to_numpy())
    np.testing.assert_array_equal(admitted, probability < spec.veto_probability)


def test_ordinal_payoff_mapping_is_side_local_training_only_and_shrunk() -> None:
    spec = MetaTargetSpec("ordinal", "ordinal_residual", shrinkage_support=20.0)
    frame, fit = _fit(spec)
    assert len(fit.class_payoff_bps) == 4
    assert fit.fit_rows == len(frame)
    assert pd.Timestamp(fit.max_label_available_utc) < pd.Timestamp("2024-01-01", tz="UTC")
    probability = np.tile([0.0, 0.0, 0.0, 1.0], (len(frame), 1))
    score, _ = reconstruct_meta_action(frame, fit, probability)
    assert np.isfinite(score).all()
    assert np.max(np.abs(score - frame.r3_opportunity_score)) <= spec.correction_cap_score_std * fit.raw_base_scale + 1e-12


def test_fold_quantile_ordinal_residual_uses_training_terciles_and_robust_bps_reconstruction() -> None:
    spec = MetaTargetSpec(
        "tercile", "quantile_ordinal_residual", residual_clip_bps=120.0
    )
    frame, fit = _fit(spec)
    residual = (
        frame.exact_net_bps.to_numpy()
        - frame.prequential_base_expected_net_bps.to_numpy()
    )
    np.testing.assert_allclose(
        fit.residual_thresholds_bps,
        np.quantile(residual, (1 / 3, 2 / 3), method="linear"),
    )
    assert fit.class_support == tuple(int(np.sum(fit.target == value)) for value in range(3))
    assert fit.class_location_method.startswith("training_class_winsorized_mean")
    assert fit.quantile_method == "linear"
    for value in range(3):
        assert fit.class_median_bps[value] == pytest.approx(
            np.median(residual[fit.target == value])
        )
        assert np.isfinite(fit.class_location_uncertainty_bps[value])
    probability = np.tile([0.0, 0.0, 1.0], (len(frame), 1))
    score, admitted = reconstruct_meta_action(frame, fit, probability)
    prior = np.asarray(fit.class_support, dtype=float) / sum(fit.class_support)
    expected_correction = (np.asarray([0.0, 0.0, 1.0]) - prior) @ np.asarray(
        fit.class_payoff_bps
    )
    expected = frame.prequential_base_expected_net_bps.to_numpy() + np.clip(
        expected_correction, -120.0, 120.0
    )
    np.testing.assert_allclose(score, expected)
    assert admitted.all()
    no_skill = np.tile(prior, (len(frame), 1))
    no_skill_score, _ = reconstruct_meta_action(frame, fit, no_skill)
    np.testing.assert_allclose(
        no_skill_score, frame.prequential_base_expected_net_bps.to_numpy()
    )


def test_quantile_ordinal_heldout_targets_cannot_change_fold_thresholds_or_corrections() -> None:
    spec = MetaTargetSpec(
        "tercile", "quantile_ordinal_residual", residual_clip_bps=200.0
    )
    original = _frame(240)
    changed = original.copy()
    folds = np.full(len(original), -1, dtype=np.int32)
    folds[180:] = 0
    changed.loc[180:, "exact_net_bps"] += np.linspace(-10000, 10000, 60)

    def predictor(train_x, target, weight, validation_x, _spec):
        counts = np.bincount(target.astype(int), weights=weight, minlength=3)
        return np.tile(counts / counts.sum(), (len(validation_x), 1))

    left = run_strict_meta_target_arm(
        original, spec,
        feature_columns=("r3_opportunity_score",), fold_id=folds,
        predictor=predictor,
    )
    right = run_strict_meta_target_arm(
        changed, spec,
        feature_columns=("r3_opportunity_score",), fold_id=folds,
        predictor=predictor,
    )
    columns = [
        "residual_q33_bps", "residual_q67_bps",
        "class_0_support", "class_1_support", "class_2_support",
        "class_0_residual_location_bps", "class_1_residual_location_bps",
        "class_2_residual_location_bps", "zero_in_middle_tercile",
        "fold_semantic_valid", "class_0_residual_median_bps",
        "class_1_residual_median_bps", "class_2_residual_median_bps",
    ]
    pd.testing.assert_frame_equal(
        left.fold_provenance[columns], right.fold_provenance[columns]
    )
    np.testing.assert_allclose(left.arm.prediction, right.arm.prediction)
    np.testing.assert_allclose(left.arm.prior_prediction, right.arm.prior_prediction)
    np.testing.assert_allclose(left.arm.score, right.arm.score)
    assert not np.array_equal(left.arm.target, right.arm.target)


def test_quantile_prior_baseline_and_skill_metrics_are_fold_local() -> None:
    frame = _frame(240)
    folds = np.full(len(frame), -1, dtype=np.int32)
    folds[120:180] = 0
    folds[180:] = 1
    spec = MetaTargetSpec("tercile", "quantile_ordinal_residual")

    def predictor(train_x, target, weight, validation_x, _spec):
        # Deliberately use a non-prior probability so skill is measurable.
        return np.tile([0.2, 0.3, 0.5], (len(validation_x), 1))

    result = run_strict_meta_target_arm(
        frame, spec, feature_columns=("r3_opportunity_score",),
        fold_id=folds, predictor=predictor,
    )
    assert result.arm.prior_prediction.shape == (120, 3)
    for fold_value, provenance in result.fold_provenance.set_index("fold_id").iterrows():
        positions = result.arm.fold_id == fold_value
        expected = np.asarray([
            provenance.class_0_training_prior,
            provenance.class_1_training_prior,
            provenance.class_2_training_prior,
        ])
        np.testing.assert_allclose(
            result.arm.prior_prediction[positions],
            np.tile(expected, (int(positions.sum()), 1)),
        )
    controls = _controls(
        frame.iloc[result.evaluation_positions].reset_index(drop=True),
        folds[result.evaluation_positions],
    )
    metrics = evaluate_meta_oof_arms(
        frame.iloc[result.evaluation_positions].reset_index(drop=True),
        [*controls, result.arm],
    )
    row = metrics.loc[metrics.arm_id.eq("tercile")].iloc[0]
    required = {
        "target_prior_accuracy", "target_majority_accuracy",
        "target_prior_log_loss", "target_prior_multiclass_brier",
        "target_prior_rps", "target_balanced_accuracy",
        "target_ordinal_expected_class_spearman",
        "target_log_loss_skill", "target_brier_skill", "target_rps_skill",
        "target_accuracy_delta_vs_prior", "target_accuracy_ratio_to_prior",
    }
    assert required.issubset(metrics.columns)
    assert np.isfinite(pd.to_numeric(row.loc[list(required)], errors="coerce")).all()


def test_focused_quantile_funnel_is_target_isolation_plus_current_control() -> None:
    specs = focused_quantile_meta_target_specs()
    assert [(spec.arm_id, spec.family) for spec in specs] == [
        ("T3Q_fold_quantile_ordinal_residual", "quantile_ordinal_residual"),
        ("C3_current_map_huber", "huber_residual"),
    ]


def test_quantile_semantics_require_strict_negative_q33_and_gate_promotion() -> None:
    frame = _frame(90)
    mapped = frame.prequential_base_expected_net_bps.to_numpy()
    # Training q33 is exactly zero, so neutral tercile names remain valid but
    # the economic over/right/under interpretation must not be promoted.
    residual = np.r_[np.zeros(30), np.linspace(1.0, 60.0, 30), np.ones(30)]
    frame["exact_net_bps"] = mapped + residual
    folds = np.full(len(frame), -1, dtype=np.int32)
    folds[60:] = 0

    def predictor(train_x, target, weight, valid_x, spec):
        prior = np.bincount(target.astype(int), minlength=3).astype(float)
        return np.tile(prior / prior.sum(), (len(valid_x), 1))

    result = run_strict_meta_target_arm(
        frame, MetaTargetSpec("tercile", "quantile_ordinal_residual"),
        feature_columns=("r3_opportunity_score",), fold_id=folds,
        predictor=predictor,
    )
    assert result.fold_provenance.residual_q33_bps.iloc[0] == pytest.approx(0.0)
    assert not result.fold_provenance.fold_semantic_valid.iloc[0]
    assert result.arm.semantic_valid is False

    evaluation = frame.iloc[result.evaluation_positions].reset_index(drop=True)
    local_folds = folds[result.evaluation_positions]
    controls = _controls(evaluation, local_folds)
    # Give the invalid-semantic arm an oracle score; the semantic gate must
    # still force the exact raw-base no-op.
    invalid_oracle = MetaOOFArm(
        "T3Q_invalid_semantics", evaluation.exact_net_bps.to_numpy(float),
        np.ones(len(evaluation), dtype=bool), local_folds,
        "quantile_ordinal_residual", semantic_valid=False,
    )
    metrics = evaluate_meta_oof_arms(evaluation, [*controls, invalid_oracle])
    assert select_meta_arm_with_noop_gate(metrics)["winner_arm_id"] == "C0_raw_base_exact_noop"


@pytest.mark.parametrize("clip", [50.0, 100.0, 200.0])
def test_clipped_residual_target_and_causal_tail_weights(clip: float) -> None:
    spec = MetaTargetSpec("clip", "clipped_residual", residual_clip_bps=clip, base_tail_fraction=0.30)
    _, fit = _fit(spec)
    assert np.max(np.abs(fit.target)) <= clip
    assert set(np.unique(fit.sample_weight)) == {0.25, 1.0}


def test_current_huber_is_explicit_map_reconstruction_negative_control() -> None:
    spec = MetaTargetSpec("huber", "huber_residual")
    frame, fit = _fit(spec)
    prediction = np.full(len(frame), 7.0)
    score, admitted = reconstruct_meta_action(frame, fit, prediction)
    np.testing.assert_allclose(score, frame.prequential_base_expected_net_bps + 7.0)
    assert admitted.all()


def test_oof_evaluation_requires_exact_same_fold_support_and_reports_paired_gaps() -> None:
    frame = _frame()
    frame["exact_net_bps"] = (
        np.sin(np.arange(len(frame)) * 0.73) * 120.0
        + frame.r3_opportunity_score.to_numpy() * 5.0
    )
    folds = np.repeat(np.arange(4), 50)
    controls = _controls(frame, folds)
    better = MetaOOFArm(
        "T1_better", frame.exact_net_bps.to_numpy(), np.ones(len(frame), dtype=bool),
        folds, "reliability", target=(frame.exact_net_bps >= 0).to_numpy(np.int8),
        prediction=np.linspace(0.0, 1.0, len(frame)),
    )
    metrics = evaluate_meta_oof_arms(frame, [*controls, better])
    assert set(metrics.top_fraction) == {0.01, 0.05, 0.10, 0.20}
    assert metrics[metrics.arm_id.eq("T1_better")].delta_vs_raw_net_bps.gt(0).all()
    assert metrics[metrics.arm_id.eq("T1_better")].target_auc.notna().all()
    gate = select_meta_arm_with_noop_gate(metrics)
    assert gate["winner_arm_id"] == "T1_better"
    assert gate["learned_meta_promoted"] is True

    changed = MetaOOFArm(
        "bad", better.score, better.action_admitted, folds[::-1], "reliability"
    )
    with pytest.raises(StageIMetaTargetError, match="fold/support"):
        evaluate_meta_oof_arms(frame, [*controls, changed])


def test_noop_wins_when_learned_arm_fails_any_mandatory_gate() -> None:
    frame = _frame()
    folds = np.repeat(np.arange(4), 50)
    controls = _controls(frame, folds)
    harmful = MetaOOFArm(
        "T4_harmful", -frame.r3_opportunity_score.to_numpy(),
        np.ones(len(frame), dtype=bool), folds, "clipped_residual",
    )
    metrics = evaluate_meta_oof_arms(frame, [*controls, harmful])
    gate = select_meta_arm_with_noop_gate(metrics)
    assert gate == {
        "winner_arm_id": "C0_raw_base_exact_noop",
        "deployment_action": "no_op",
        "learned_meta_promoted": False,
        "reason": "no learned arm cleared pooled top10 and worst-month/fold raw-base gates",
    }


def test_target_fit_rejects_unresolved_rows_and_cross_side_data() -> None:
    frame = _frame()
    spec = MetaTargetSpec("reliable", "reliability")
    with pytest.raises(StageIMetaTargetError, match="unresolved"):
        fit_meta_target(frame, spec, side="long", fit_before_utc="2023-01-02")
    frame.loc[0, "side_name"] = "short"
    with pytest.raises(StageIMetaTargetError, match="side-local"):
        fit_meta_target(frame, spec, side="long", fit_before_utc="2024-01-01")


def test_evaluator_rejects_a_fake_bounded_noop() -> None:
    frame = _frame()
    folds = np.repeat(np.arange(4), 50)
    controls = _controls(frame, folds)
    controls[2] = MetaOOFArm(
        controls[2].arm_id, controls[2].score + 1e-6, controls[2].action_admitted,
        controls[2].fold_id, controls[2].target_family,
    )
    with pytest.raises(StageIMetaTargetError, match="not exactly"):
        evaluate_meta_oof_arms(frame, controls)


def test_strict_arm_runner_uses_frozen_folds_and_only_prior_resolved_training_rows() -> None:
    frame = _frame(240)
    folds = np.full(len(frame), -1, dtype=np.int32)
    folds[80:120] = 0
    folds[120:180] = 1
    folds[180:] = 2
    calls: list[tuple[int, int]] = []

    def predictor(train_x, target, weight, validation_x, spec):
        calls.append((len(train_x), len(validation_x)))
        assert len(target) == len(weight) == len(train_x)
        probability = np.full(len(validation_x), np.average(target, weights=weight))
        return np.column_stack([1.0 - probability, probability])

    result = run_strict_meta_target_arm(
        frame,
        MetaTargetSpec("T1", "reliability", hurdle_bps=25.0),
        feature_columns=("r3_opportunity_score", "prequential_base_expected_net_bps"),
        fold_id=folds,
        predictor=predictor,
    )
    np.testing.assert_array_equal(result.evaluation_positions, np.arange(80, 240))
    np.testing.assert_array_equal(result.arm.fold_id, folds[80:])
    assert calls == [(79, 40), (119, 60), (179, 60)]
    assert result.fold_provenance.strict_prior_resolved.all()
    assert (
        pd.to_datetime(result.fold_provenance.train_max_label_available_utc, utc=True)
        < pd.to_datetime(result.fold_provenance.validation_start_utc, utc=True)
    ).all()


def test_strict_arm_runner_rejects_fold_ids_out_of_chronological_order() -> None:
    frame = _frame(120)
    folds = np.full(len(frame), -1, dtype=np.int32)
    folds[40:80] = 1
    folds[80:] = 0
    with pytest.raises(StageIMetaTargetError, match="chronologically ordered"):
        run_strict_meta_target_arm(
            frame, MetaTargetSpec("T1", "reliability"),
            feature_columns=("r3_opportunity_score",), fold_id=folds,
            predictor=lambda *args: np.zeros(len(args[3])),
        )
