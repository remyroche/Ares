from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.stage_iii_robust_target_models import (
    COMMON_BPS_RECONSTRUCTION,
    ONE_SHARED_BOTH_SIDE_MODEL,
    ORDINAL_FORMULATION,
    QUANTILE_FORMULATION,
    RobustTargetModelConfig,
    RobustTargetModelError,
    fit_ordinal_shared_robust_target,
    fit_quantile_shared_robust_target,
    reconstruct_quantile_shared_outputs,
    repair_cumulative_ordinal_probabilities,
)


def _frame() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    start = pd.Timestamp("2024-01-01 09:00", tz="UTC")
    row = 0
    for day in range(4):
        for side_index, side in enumerate(("long", "short")):
            for within in range(8):
                decision = start + pd.Timedelta(days=day, minutes=within * 5 + side_index)
                residual = -192.5 + 55.0 * within + 8.0 * side_index
                base = 30.0 + 1.5 * within - 2.0 * side_index
                prior = -10.0 + 2.0 * day + side_index
                records.append(
                    {
                        "candidate_id": f"candidate_{row:03d}",
                        "decision_ts": decision,
                        "label_available_ts": decision + pd.Timedelta(hours=12),
                        "side_name": side,
                        "exact_net_bps": base + prior + residual,
                        "prequential_base_expected_net_bps": base,
                        "prequential_soft_regime_prior_residual_bps": prior,
                        "base_map_is_prequential": True,
                        "base_map_source_side": side,
                        "base_map_max_label_available_ts": decision - pd.Timedelta(hours=1),
                        "soft_regime_is_causal_prequential": True,
                        "soft_regime_fit_end_ts": decision - pd.Timedelta(minutes=30),
                        "prior_resolved_max_label_available_ts": decision - pd.Timedelta(minutes=15),
                        "feature_signal": float(within) + 0.1 * side_index,
                        "feature_context": float(day) - 0.05 * within,
                        "p_regime_calm": 0.82 - 0.015 * within,
                        "p_regime_stress": 0.18 + 0.015 * within,
                    }
                )
                row += 1
    return pd.DataFrame(records)


FEATURES = ("feature_signal", "feature_context", "p_regime_calm", "p_regime_stress")
CUTOFF = "2024-01-08T00:00:00Z"
CONFIG = RobustTargetModelConfig(
    n_estimators=22,
    learning_rate=0.06,
    num_leaves=7,
    min_child_samples=2,
    l2_regularization=2.0,
    random_state=23,
)


def test_t3_uses_true_cumulative_ordinal_shared_heads_and_common_bps_reconstruction() -> None:
    frame = _frame()
    fit = fit_ordinal_shared_robust_target(
        frame,
        feature_names=FEATURES,
        fit_before_utc=CUTOFF,
        config=CONFIG,
        sample_weight=np.linspace(0.5, 1.5, len(frame)),
    )

    audit = fit.audit
    assert audit.arm == "T3_ordinal"
    assert audit.routing == ONE_SHARED_BOTH_SIDE_MODEL
    assert audit.formulation == ORDINAL_FORMULATION
    assert audit.reconstruction == COMMON_BPS_RECONSTRUCTION
    assert audit.training_row_count == len(frame)
    assert dict(audit.training_rows_by_side) == {"long": 32, "short": 32}
    assert audit.feature_names == FEATURES
    assert len(audit.feature_sha256) == 64
    assert len(audit.ordinal_heads) == 3
    assert [head.threshold_edge_bps for head in audit.ordinal_heads] == [-150.0, 0.0, 100.0]
    assert all("<=" in head.event_definition for head in audit.ordinal_heads)
    assert all(head.positive_support > 0 and head.negative_support > 0 for head in audit.ordinal_heads)
    assert audit.max_label_available_utc < pd.Timestamp(CUTOFF)

    output = fit.predict_outputs(frame)
    classes = output.loc[:, [f"ordinal_class_{i}_probability" for i in range(4)]].to_numpy(float)
    assert np.isfinite(classes).all()
    np.testing.assert_allclose(classes.sum(axis=1), 1.0, atol=1e-6)
    expected = fit.predict_expected_net_bps(frame)
    np.testing.assert_allclose(
        expected,
        frame["prequential_base_expected_net_bps"].to_numpy()
        + frame["prequential_soft_regime_prior_residual_bps"].to_numpy()
        + output["candidate_residual_bps"].to_numpy(),
    )
    assert fit.audit.to_dict()["ordinal_heads"][0]["event_definition"].startswith("candidate_residual")


def test_t3_cdf_repair_creates_a_probability_simplex_without_outcomes() -> None:
    repaired, crossing = repair_cumulative_ordinal_probabilities(
        [[0.80, 0.25, 0.70], [0.10, 0.45, 0.90]]
    )
    np.testing.assert_allclose(repaired[0], [0.80, 0.80, 0.80])
    assert crossing.tolist() == [True, False]
    probabilities = np.column_stack(
        [repaired[:, 0], repaired[:, 1] - repaired[:, 0], repaired[:, 2] - repaired[:, 1], 1.0 - repaired[:, 2]]
    )
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert (probabilities >= 0.0).all()


def test_t4_has_five_shared_quantile_heads_crossing_repair_and_common_bps() -> None:
    frame = _frame()
    fit = fit_quantile_shared_robust_target(
        frame,
        feature_names=FEATURES,
        fit_before_utc=CUTOFF,
        config=CONFIG,
        sample_weight=np.linspace(0.75, 1.25, len(frame)),
    )

    audit = fit.audit
    assert audit.arm == "T4_quantile"
    assert audit.routing == ONE_SHARED_BOTH_SIDE_MODEL
    assert audit.formulation == QUANTILE_FORMULATION
    assert audit.quantile_heads == ("q10", "q25", "q50", "q75", "q90")
    assert len(fit.models) == 5
    assert audit.max_label_available_utc < pd.Timestamp(CUTOFF)
    output = fit.predict_outputs(frame)
    assert {"q10", "q25", "q50", "q75", "q90", "candidate_residual_median_bps"}.issubset(output.columns)
    values = output.loc[:, ["q10", "q25", "q50", "q75", "q90"]].to_numpy(float)
    assert np.isfinite(values).all()
    assert (np.diff(values, axis=1) >= -1e-6).all()
    expected = fit.predict_expected_net_bps(frame)
    np.testing.assert_allclose(
        expected,
        frame["prequential_base_expected_net_bps"].to_numpy()
        + frame["prequential_soft_regime_prior_residual_bps"].to_numpy()
        + output["candidate_residual_median_bps"].to_numpy(),
    )

    crossed = reconstruct_quantile_shared_outputs(
        {"q10": [0.0], "q25": [-10.0], "q50": [20.0], "q75": [5.0], "q90": [50.0]},
        fit.target_fit,
    )
    np.testing.assert_allclose(
        crossed.loc[0, ["q10", "q25", "q50", "q75", "q90"]].to_numpy(dtype=float),
        [0, 0, 20, 20, 50],
    )
    assert bool(crossed.loc[0, "quantile_crossing_repaired"])


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda frame: frame.assign(label_available_ts=pd.Timestamp(CUTOFF)), "unresolved/current/future"),
        (lambda frame: frame.assign(base_map_is_prequential="True"), "explicit true booleans"),
        (lambda frame: frame.assign(base_map_source_side="long"), "same-side"),
        (lambda frame: frame.assign(side_name="long"), "both-side"),
        (lambda frame: frame.assign(candidate_id="duplicate"), "unique"),
    ],
)
def test_robust_target_models_reject_noncausal_or_nonshared_ledger(
    mutation, message: str
) -> None:
    with pytest.raises(RobustTargetModelError, match=message):
        fit_ordinal_shared_robust_target(
            mutation(_frame()), feature_names=FEATURES, fit_before_utc=CUTOFF, config=CONFIG
        )


def test_robust_target_models_reject_hard_regime_and_missing_frozen_inference_feature() -> None:
    frame = _frame()
    with pytest.raises(RobustTargetModelError, match="hard regime"):
        fit_quantile_shared_robust_target(
            frame.assign(hard_regime_id=1.0),
            feature_names=(*FEATURES, "hard_regime_id"),
            fit_before_utc=CUTOFF,
            config=CONFIG,
        )
    fit = fit_quantile_shared_robust_target(
        frame, feature_names=FEATURES, fit_before_utc=CUTOFF, config=CONFIG
    )
    with pytest.raises(RobustTargetModelError, match="missing frozen features"):
        fit.predict_outputs(frame.drop(columns=["feature_context"]))


def test_deterministic_quantile_fit_has_identical_frozen_feature_and_target_digests() -> None:
    frame = _frame()
    left = fit_quantile_shared_robust_target(
        frame, feature_names=FEATURES, fit_before_utc=CUTOFF, config=CONFIG
    )
    right = fit_quantile_shared_robust_target(
        frame, feature_names=FEATURES, fit_before_utc=CUTOFF, config=CONFIG
    )
    assert left.audit.feature_sha256 == right.audit.feature_sha256
    assert left.audit.target_label_sha256 == right.audit.target_label_sha256
    np.testing.assert_allclose(
        left.predict_candidate_residual_bps(frame),
        right.predict_candidate_residual_bps(frame),
        atol=1e-6,
        rtol=0.0,
    )
