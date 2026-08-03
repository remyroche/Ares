from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.shared_regime_residual_expert import (
    SharedResidualColumns,
    SharedResidualExpertError,
    SoftRegimeResidualConfig,
    build_prequential_regime_relative_features,
    build_restricted_soft_regime_interactions,
    classify_cross_era_feature_transport,
    fit_shared_regime_residual_expert,
    mild_environment_weights,
    prequential_soft_side_regime_residual_baseline,
    prepare_shared_regime_residual_frame,
    reconstruct_shared_regime_expected_net_bps,
    robust_cross_era_selection_score,
)


def _frame(rows: int = 20) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    p_calm = np.where(np.arange(rows) % 2 == 0, 0.8, 0.2)
    return pd.DataFrame(
        {
            "decision_ts": ts,
            "label_available_ts": ts + pd.Timedelta(minutes=30),
            "side_name": np.where(np.arange(rows) % 3 == 0, "short", "long"),
            "exact_net_bps": np.linspace(-150.0, 180.0, rows),
            "prequential_base_expected_net_bps": np.linspace(-10.0, 10.0, rows),
            "p_regime_calm": p_calm,
            "p_regime_stress": 1.0 - p_calm,
            "broad_regime": np.where(p_calm >= 0.5, "calm", "stress"),
            "market_confirmation": np.linspace(-1.0, 1.0, rows),
            "cost_to_atr": np.linspace(0.5, 2.0, rows),
            "era": np.where(np.arange(rows) < rows // 2, "era_a", "era_b"),
        }
    )


def _cfg() -> SoftRegimeResidualConfig:
    return SoftRegimeResidualConfig(
        min_global_rows=2,
        side_shrink_rows=2.0,
        regime_shrink_rows=2.0,
        regime_weight_cap=0.5,
        residual_scale_floor_bps=1.0,
    )


def test_soft_regime_prior_uses_only_labels_resolved_before_decision() -> None:
    frame = _frame()
    left = prequential_soft_side_regime_residual_baseline(
        frame, soft_regime_columns=["p_regime_calm", "p_regime_stress"], config=_cfg()
    )
    changed = frame.copy()
    changed.loc[5, "exact_net_bps"] += 50_000.0
    right = prequential_soft_side_regime_residual_baseline(
        changed, soft_regime_columns=["p_regime_calm", "p_regime_stress"], config=_cfg()
    )
    np.testing.assert_allclose(
        left.loc[4:5, "prequential_soft_regime_prior_residual_bps"],
        right.loc[4:5, "prequential_soft_regime_prior_residual_bps"],
        equal_nan=True,
    )
    changed = frame.copy()
    changed.loc[12:, "exact_net_bps"] += 100_000.0
    right = prequential_soft_side_regime_residual_baseline(
        changed, soft_regime_columns=["p_regime_calm", "p_regime_stress"], config=_cfg()
    )
    np.testing.assert_allclose(
        left.loc[:11, "prequential_soft_regime_prior_residual_bps"],
        right.loc[:11, "prequential_soft_regime_prior_residual_bps"],
        equal_nan=True,
    )
    fitted = left.loc[left.prior_resolved_max_label_available_ts.notna()]
    assert (
        pd.to_datetime(fitted.prior_resolved_max_label_available_ts, utc=True)
        < pd.to_datetime(frame.loc[fitted.index, "decision_ts"], utc=True)
    ).all()
    assert left.loc[:1, "candidate_residual_bps"].isna().all()


def test_equal_timestamp_rows_do_not_see_each_others_outcomes() -> None:
    frame = _frame(10)
    frame.loc[4:5, "decision_ts"] = frame.loc[4, "decision_ts"]
    frame.loc[4:5, "label_available_ts"] = frame.loc[4, "decision_ts"] + pd.Timedelta(minutes=30)
    left = prequential_soft_side_regime_residual_baseline(
        frame, soft_regime_columns=["p_regime_calm", "p_regime_stress"], config=_cfg()
    )


def test_a0_a3_baseline_funnel_has_distinct_causal_targets() -> None:
    frame = _frame(40)
    kwargs = {
        "soft_regime_columns": ["p_regime_calm", "p_regime_stress"],
        "config": _cfg(),
    }
    a0 = prequential_soft_side_regime_residual_baseline(
        frame, baseline_mode="A0_current", **kwargs
    )
    a1 = prequential_soft_side_regime_residual_baseline(
        frame, baseline_mode="A1_side_centered", **kwargs
    )
    a2 = prequential_soft_side_regime_residual_baseline(
        frame,
        baseline_mode="A2_side_hard_regime_centered",
        hard_regime_column="broad_regime",
        **kwargs,
    )
    a3 = prequential_soft_side_regime_residual_baseline(
        frame, baseline_mode="A3_soft_regime_centered", **kwargs
    )

    assert a0.candidate_residual_bps.notna().all()
    assert np.allclose(a0.prequential_soft_regime_prior_residual_bps, 0.0)
    supported = a3.prequential_soft_regime_prior_residual_bps.notna()
    assert supported.any()
    assert not np.allclose(
        a1.loc[supported, "prequential_soft_regime_prior_residual_bps"],
        a3.loc[supported, "prequential_soft_regime_prior_residual_bps"],
    )
    assert set(a2.loc[supported, "soft_regime_prior_fallback"]) == {
        "shrunk_side_hard_regime_prior_diagnostic"
    }

    changed = frame.copy()
    changed.loc[30:, "exact_net_bps"] += 100_000.0
    a2_changed = prequential_soft_side_regime_residual_baseline(
        changed,
        baseline_mode="A2_side_hard_regime_centered",
        hard_regime_column="broad_regime",
        **kwargs,
    )
    np.testing.assert_allclose(
        a2.loc[:29, "prequential_soft_regime_prior_residual_bps"],
        a2_changed.loc[:29, "prequential_soft_regime_prior_residual_bps"],
        equal_nan=True,
    )
def test_relative_features_are_predecision_and_interactions_are_restricted() -> None:
    frame = _frame()
    left, names = build_prequential_regime_relative_features(
        frame,
        feature_names=["market_confirmation", "cost_to_atr"],
        soft_regime_columns=["p_regime_calm", "p_regime_stress"],
        min_reference_rows=2,
        side_shrink_rows=2,
        regime_shrink_rows=2,
    )
    changed = frame.copy()
    changed.loc[12:, "market_confirmation"] += 100_000.0
    right, _ = build_prequential_regime_relative_features(
        changed,
        feature_names=["market_confirmation", "cost_to_atr"],
        soft_regime_columns=["p_regime_calm", "p_regime_stress"],
        min_reference_rows=2,
        side_shrink_rows=2,
        regime_shrink_rows=2,
    )
    np.testing.assert_allclose(left.iloc[:12], right.iloc[:12], equal_nan=True)
    assert len(names) == 4
    interactions, interaction_names = build_restricted_soft_regime_interactions(
        frame,
        soft_regime_columns=["p_regime_calm", "p_regime_stress"],
        base_feature_names=["market_confirmation"],
    )
    assert len(interaction_names) == 2
    assert interactions.shape[1] == 2
    assert not any("cost_to_atr" in name for name in interaction_names)


def test_regime_relative_default_scale_resists_one_large_outlier() -> None:
    frame = _frame(40)
    frame["market_confirmation"] = 0.0
    frame.loc[10, "market_confirmation"] = 10_000.0
    frame.loc[30, "market_confirmation"] = 1.0
    kwargs = {
        "feature_names": ["market_confirmation"],
        "soft_regime_columns": ["p_regime_calm", "p_regime_stress"],
        "min_reference_rows": 2,
        "side_shrink_rows": 2,
        "regime_shrink_rows": 2,
    }
    robust, _ = build_prequential_regime_relative_features(frame, **kwargs)
    standard, _ = build_prequential_regime_relative_features(
        frame, scale_estimator="standard_deviation", **kwargs
    )
    column = "__srre__market_confirmation__soft_regime_z"
    assert abs(float(robust.loc[30, column])) > abs(float(standard.loc[30, column]))


def test_weighting_and_cross_era_score_are_mild_and_explicit() -> None:
    frame = _frame()
    frame["certainty"] = np.linspace(0.0, 1.0, len(frame))
    weights = mild_environment_weights(
        frame, environment_column="era", balance="era", label_certainty_column="certainty"
    )
    assert weights.dtype == np.float32
    assert weights.min() >= 0.25
    assert weights.max() <= 4.0
    assert float(weights.mean()) == pytest.approx(1.0, abs=1e-7)
    assert weights[-1] > weights[0]
    score = robust_cross_era_selection_score({"a": 10.0, "b": 10.0, "c": -10.0})
    assert score["selection_score"] < score["mean_environment_score"]
    assert score["worst_environment_score"] == -10.0


def test_transport_classifier_keeps_conditionable_groups_shared() -> None:
    mda = pd.DataFrame(
        {
            "feature_group": ["core", "core", "conditional", "conditional", "noise", "noise"],
            "environment": ["early", "late"] * 3,
            "transport_mda": [0.20, 0.10, -0.10, 0.15, -0.03, -0.02],
            "conditioned_mda": [0.20, 0.10, 0.11, 0.16, -0.03, -0.02],
        }
    )
    result = classify_cross_era_feature_transport(
        mda, conditioned_importance_column="conditioned_mda", latest_environment="late"
    )
    classes = dict(zip(result.feature_group, result.classification, strict=False))
    assert classes == {"conditional": "REGIME_CONDITIONAL", "core": "INVARIANT_CORE", "noise": "REDUNDANT"}
    core = result.set_index("feature_group").loc["core"]
    assert core.transport_mda == pytest.approx(0.125)


def test_transport_classifier_requires_latest_conditioned_support() -> None:
    mda = pd.DataFrame(
        {
            "feature_group": ["conditional"] * 4,
            "environment": ["a", "b", "c", "latest"],
            "transport_mda": [-0.30, 0.02, 0.01, -0.05],
            "conditioned_mda": [0.20, 0.20, 0.20, -0.01],
        }
    )
    result = classify_cross_era_feature_transport(
        mda,
        conditioned_importance_column="conditioned_mda",
        latest_environment="latest",
    )
    assert result.loc[0, "classification"] != "REGIME_CONDITIONAL"
    assert result.loc[0, "conditioned_latest_environment_mda"] == pytest.approx(-0.01)


def test_preparation_and_single_shared_fit_reconstruct_common_bps() -> None:
    frame = _frame(60)
    prepared, generated = prepare_shared_regime_residual_frame(
        frame,
        soft_regime_columns=["p_regime_calm", "p_regime_stress"],
        regime_relative_feature_names=["market_confirmation", "cost_to_atr"],
        restricted_interaction_feature_names=["market_confirmation", "cost_to_atr"],
        baseline_config=_cfg(),
    )
    # The first few rows are an explicit prior-resolved burn-in, not targets
    # fitted with same-fold information.
    train = prepared.loc[prepared.candidate_residual_bps.notna()].copy()
    features = [
        "market_confirmation", "cost_to_atr", "p_regime_calm", "p_regime_stress",
        "soft_regime_entropy", "shared_residual_side_is_long", *generated,
    ]
    fit = fit_shared_regime_residual_expert(
        train,
        feature_names=features,
        fit_before_utc="2025-01-01T00:00:00Z",
        params={"n_estimators": 8, "learning_rate": 0.15, "num_leaves": 7, "min_child_samples": 1, "n_jobs": 1, "verbosity": -1, "random_state": 7},
    )
    correction = fit.predict_candidate_residual_bps(train)
    prediction = reconstruct_shared_regime_expected_net_bps(train, correction)
    assert len(prediction) == len(train)
    assert np.isfinite(prediction).all()
    assert not hasattr(fit, "experts")
    assert not hasattr(fit, "regime_models")


def test_fit_rejects_outcome_features_and_current_labels() -> None:
    frame = _frame(20)
    prepared, _ = prepare_shared_regime_residual_frame(
        frame,
        soft_regime_columns=["p_regime_calm", "p_regime_stress"],
        regime_relative_feature_names=[],
        restricted_interaction_feature_names=["market_confirmation"],
        baseline_config=_cfg(),
    )
    train = prepared.loc[prepared.candidate_residual_bps.notna()].copy()
    with pytest.raises(SharedResidualExpertError, match="outcome-derived"):
        fit_shared_regime_residual_expert(
            train, feature_names=["exact_net_bps"], fit_before_utc="2025-01-01"
        )
    with pytest.raises(SharedResidualExpertError, match="unresolved/current/future"):
        fit_shared_regime_residual_expert(
            train, feature_names=["market_confirmation"], fit_before_utc=train.label_available_ts.max()
        )
    train["regime_id"] = np.arange(len(train)) % 2
    with pytest.raises(SharedResidualExpertError, match="hard regime identifiers"):
        fit_shared_regime_residual_expert(
            train, feature_names=["market_confirmation", "regime_id"], fit_before_utc="2025-01-01"
        )
