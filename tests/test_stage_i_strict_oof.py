from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.prequential_r3_value_map import PrequentialR3ValueMapConfig
from extreme_price_movements.stage_i_causal_admission import Causal21dAdmissionSpec
from extreme_price_movements.stage_i_feature_selection import (
    STAGE_I_META_BASE_OOF_HANDOFF_FEATURES,
)
from extreme_price_movements.stage_i_strict_oof import (
    StageIStrictOOFResult,
    StageIStrictOOFPlan,
    generate_stage_i_strict_oof,
    write_stage_i_strict_oof_artifact,
)


def _plan(side: str, *, future_net_shift: float = 0.0) -> StageIStrictOOFPlan:
    # The exact +13h availability gate deliberately creates a longer
    # base-to-residual burn-in.  Keep enough chronological support to exercise
    # a genuine same-side OOF residual fold after that gate.
    n = 144
    rng = np.random.default_rng(31 if side == "long" else 47)
    r3 = np.tile(np.array([0, 1, 2], dtype=np.int8), n // 3)
    feature = r3.astype(float) + rng.normal(scale=.15, size=n)
    frame = pd.DataFrame({
        "base_signal": feature.astype(np.float32),
        "base_noise": rng.normal(size=n).astype(np.float32),
        "meta_signal": (feature + rng.normal(scale=.1, size=n)).astype(np.float32),
        "meta_context": rng.normal(size=n).astype(np.float32),
    })
    decision = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    net = np.where(r3 == 2, 180.0, np.where(r3 == 1, 20.0, -180.0)).astype(np.float32)
    net[-1] += future_net_shift
    tiny = {
        "n_estimators": 10, "learning_rate": .15, "max_depth": 3,
        "num_leaves": 7, "min_child_samples": 1, "random_state": 11,
        "n_jobs": 1, "verbosity": -1,
    }
    return StageIStrictOOFPlan(
        side=side, candidate_ids=[f"{side}-{i}" for i in range(n)], frame=frame,
        r3_target=r3, exact_net_bps=net, decision_timestamps=decision,
        label_available_timestamps=decision + pd.Timedelta(hours=13),
        base_feature_names=["base_signal", "base_noise"],
        meta_feature_names=[
            "meta_signal",
            "meta_context",
            *STAGE_I_META_BASE_OOF_HANDOFF_FEATURES,
        ],
        base_params=tiny, residual_params={**tiny, "objective": "huber"},
        n_validation_folds=3, min_train_rows=18,
        value_map=PrequentialR3ValueMapConfig(side=side, min_global_rows=4, bin_shrink_rows=4),
    )


def test_strict_oof_is_expanding_prior_resolved_and_has_r3_to_bps_bridge() -> None:
    result = generate_stage_i_strict_oof(_plan("long"))
    prediction = result.predictions
    scored = prediction.strict_oof_available
    base_scored = prediction.base_strict_oof_available
    assert scored.any()
    assert base_scored.any()
    assert not scored.all()  # Explicit burn-in is never filled in-sample.
    assert not prediction.loc[~base_scored, [
        "r3_p_adverse", "r3_p_weak", "r3_p_clear", "r3_opportunity_score",
        "prequential_base_expected_net_bps",
    ]].notna().any(axis=None)
    np.testing.assert_allclose(
        prediction.loc[scored, ["r3_p_adverse", "r3_p_weak", "r3_p_clear"]].sum(axis=1),
        1.0,
    )
    np.testing.assert_allclose(
        prediction.loc[scored, "r3_opportunity_score"],
        prediction.loc[scored, "r3_p_clear"] - prediction.loc[scored, "r3_p_adverse"],
    )
    assert prediction.loc[scored, "prequential_base_expected_net_bps"].notna().all()
    assert prediction.loc[scored, "residual_oof_bps"].notna().all()
    assert prediction.loc[~scored, "reconstructed_expected_net_bps"].isna().all()
    assert result.fold_provenance.strict_prior_resolved.all()
    fitted = result.fold_provenance.loc[
        result.fold_provenance.train_max_label_available_ts.notna()
    ]
    assert (
        pd.to_datetime(fitted.train_max_label_available_ts, utc=True)
        < pd.to_datetime(fitted.validation_start_ts, utc=True)
    ).all()
    assert result.value_map_provenance["prior_resolution_rule"] == "label_available_ts < decision_ts"


def test_later_outcome_cannot_change_prior_r3_to_bps_oof_rows() -> None:
    left = generate_stage_i_strict_oof(_plan("long", future_net_shift=0.0)).predictions
    right = generate_stage_i_strict_oof(_plan("long", future_net_shift=50_000.0)).predictions
    earlier = left.decision_ts.lt(left.decision_ts.max()) & left.strict_oof_available
    columns = ["r3_opportunity_score", "prequential_base_expected_net_bps", "residual_oof_bps"]
    np.testing.assert_allclose(left.loc[earlier, columns], right.loc[earlier, columns])


def test_strict_oof_rejects_non_h12_label_availability_contract() -> None:
    plan = _plan("long")
    invalid = StageIStrictOOFPlan(
        **{
            **plan.__dict__,
            "label_available_timestamps": pd.to_datetime(
                plan.decision_timestamps, utc=True
            )
            + pd.Timedelta(hours=12),
        }
    )
    with np.testing.assert_raises_regex(ValueError, "exact signal-close-to-H12"):
        generate_stage_i_strict_oof(invalid)


def test_strict_oof_rejects_late_unselected_base_handoff_append() -> None:
    plan = _plan("long")
    invalid = StageIStrictOOFPlan(
        **{
            **plan.__dict__,
            "meta_feature_names": ["meta_signal", "meta_context"],
        }
    )
    with np.testing.assert_raises_regex(ValueError, "direct same-side base handoff"):
        generate_stage_i_strict_oof(invalid)


def test_immutable_writer_emits_global_metrics_and_causal_admission(tmp_path) -> None:
    long = generate_stage_i_strict_oof(_plan("long"))
    short = generate_stage_i_strict_oof(_plan("short"))
    output = tmp_path / "stage_i_oof"
    manifest = write_stage_i_strict_oof_artifact(
        [long, short], output,
        admission_spec=Causal21dAdmissionSpec(min_reference_rows=4, bins=4),
    )
    assert manifest["status"] == "complete"
    for name in (
        "raw_oof_predictions.parquet", "strict_oof_predictions.parquet",
        "fold_provenance.parquet", "pooled_global_metrics.parquet",
        "candidates_with_causal_21d_admission.parquet", "manifest.json",
    ):
        assert (output / name).exists()
    metrics = pd.read_parquet(output / "pooled_global_metrics.parquet")
    assert {"base", "meta_residual"}.issubset(set(metrics.layer))
    assert metrics.loc[metrics.scope.eq("pooled_global"), "side"].eq("__all__").all()
    admission = pd.read_parquet(output / "candidates_with_causal_21d_admission.parquet")
    assert admission.candidate_key.is_unique
    assert not admission.loc[admission.causal_21d_side_expected_net_bps.isna(), "causal_21d_side_admitted_ge_50bps"].any()
    try:
        write_stage_i_strict_oof_artifact([long, short], output)
    except FileExistsError:
        pass
    else:  # pragma: no cover
        raise AssertionError("immutable artifact writer unexpectedly overwrote output")


def test_writer_uses_prior_strict_history_for_evaluation_boundary_admission(tmp_path) -> None:
    full = [generate_stage_i_strict_oof(_plan(side)) for side in ("long", "short")]
    evaluation = []
    for result in full:
        cutoff = result.predictions.decision_ts.max() - pd.Timedelta(hours=24)
        subset = result.predictions.loc[result.predictions.decision_ts.ge(cutoff)].copy()
        evaluation.append(StageIStrictOOFResult(
            side=result.side, predictions=subset,
            fold_provenance=result.fold_provenance,
            value_map_provenance=result.value_map_provenance,
            plan_summary=result.plan_summary,
        ))
    output = tmp_path / "history_supported_admission"
    write_stage_i_strict_oof_artifact(
        evaluation, output,
        admission_spec=Causal21dAdmissionSpec(min_reference_rows=4, bins=4),
        admission_reference_results=full,
    )
    candidates = pd.read_parquet(output / "candidates_with_causal_21d_admission.parquet")
    audit = pd.read_parquet(output / "causal_21d_admission_audit.parquet")
    expected = sum(
        int(result.predictions.strict_oof_available.sum()) for result in evaluation
    )
    assert len(candidates) == expected
    assert audit.used_prior_history_outside_evaluation.all()
    assert audit.reference_rows.min() > 0


def test_immutable_writer_rejects_truthy_string_fold_provenance(tmp_path) -> None:
    long = generate_stage_i_strict_oof(_plan("long"))
    short = generate_stage_i_strict_oof(_plan("short"))
    long.fold_provenance["strict_prior_resolved"] = long.fold_provenance[
        "strict_prior_resolved"
    ].astype(object)
    long.fold_provenance.loc[0, "strict_prior_resolved"] = "true"
    with np.testing.assert_raises_regex(ValueError, "explicit boolean/0/1"):
        write_stage_i_strict_oof_artifact(
            [long, short], tmp_path / "invalid_provenance",
            admission_spec=Causal21dAdmissionSpec(min_reference_rows=4, bins=4),
        )
