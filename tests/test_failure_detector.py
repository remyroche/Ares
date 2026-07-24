import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.unsupervised_regime_learning.failure_detector import (
    ProspectiveFailureDetectorConfig,
    _calibrate_probability_scores,
    _future_window_availability,
    add_causal_state_dynamics,
    attach_failure_mode_targets,
    chronological_failure_detection,
    is_batch_layout_dependent_ae_gmm_feature,
    nonlinear_feature_screen,
    purged_before_boundary,
    target_horizon_days,
)


def _fixture(days: int = 230) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2025-01-01", periods=days, freq="D", tz="UTC")
    state = pd.DataFrame(
        {
            "day": index,
            "side_name": "long",
            "archetype_policy_key": "compression",
            "shock": np.sin(np.arange(days) / 4.0).astype(np.float32),
            "entropy": np.cos(np.arange(days) / 9.0).astype(np.float32),
            "noise": np.random.default_rng(7).normal(size=days).astype(np.float32),
            "base_score": np.linspace(0.2, 0.8, days, dtype=np.float32),
            "score_meta_base_soft_label": np.linspace(
                0.25, 0.75, days, dtype=np.float32
            ),
            "hit_probability": np.linspace(0.3, 0.7, days, dtype=np.float32),
            "historical_rank": np.linspace(0.1, 0.9, days, dtype=np.float32),
        }
    )
    event = state["shock"].gt(0.75) & (np.arange(days) % 3 == 0)
    blocks = np.where(event, "event_001", "normal")
    calendar = pd.DataFrame(
        {
            "day": index,
            "side_name": "long",
            "archetype_policy_key": "compression",
            "adverse_event": event,
            "event_block": blocks,
            "mean_ev_after_1pct": np.where(event, -0.02, 0.01),
        }
    )
    assignments = pd.DataFrame(
        {
            "side_name": ["long"],
            "archetype_policy_key": ["compression"],
            "event_block": ["event_001"],
            "method": ["pca_gmm"],
            "latent_dim": [2],
            "clusters": [2],
            "cluster_id": [1],
            "semantic_label": ["overconfident_false_positive__liquidation_pressure"],
            "event_start": [index[event].min()],
            "event_end": [index[event].max()],
        }
    )
    return state, calendar, assignments


def test_attach_failure_modes_keeps_observable_state_separate() -> None:
    state, calendar, assignments = _fixture()
    result = attach_failure_mode_targets(state, calendar, assignments)
    assert result["target__any_failure"].sum() > 0
    assert result["target__failure_onset"].sum() > 0
    assert result["target__next1d__failure_onset"].sum() > 0
    assert result["target__failure_severity"].max() == pytest.approx(0.02)
    assert result["target__next1d__failure_severity"].max() == pytest.approx(0.02)
    assert pd.isna(result["target__next3d__failure_onset"].iloc[-1])
    assert result.loc[result["target__any_failure"], "failure_mode"].notna().all()
    assert result.loc[result["target__any_failure"], "failure_mode"].str.startswith(
        "overconfident_false_positive__liquidation_pressure::"
    ).all()
    assert not any(name.startswith("expost__") for name in state.columns)


def test_causal_dynamics_do_not_use_future_rows() -> None:
    state, _, _ = _fixture(80)
    first = add_causal_state_dynamics(state)
    assert {
        "state_transition_l1",
        "state_transition_l2",
        "state_transition_p90",
        "state_positive_cusum",
        "state_negative_cusum",
        "state_energy_distance_3d_30d",
        "state_mmd_rbf_3d_30d",
        "state_wasserstein_proxy_3d_30d",
        "model_meta_minus_base_score",
        "model_base_meta_abs_disagreement",
        "model_meta_minus_hit_probability",
        "model_rank_minus_hit_probability",
        "model_layer_score_dispersion",
        "state_delta1__model_meta_minus_base_score",
    }.issubset(first.columns)
    assert first["state_positive_cusum"].max() > 0.0
    assert first["state_negative_cusum"].max() > 0.0
    assert first["state_energy_distance_3d_30d"].notna().sum() > 0
    changed = state.copy()
    changed.loc[changed.index[-1], "shock"] = 1000.0
    second = add_causal_state_dynamics(changed)
    columns = [
        name for name in first if name.startswith(("state_", "market_", "local_"))
    ]
    pd.testing.assert_frame_equal(
        first.loc[first.index[:-1], columns],
        second.loc[second.index[:-1], columns],
    )


def test_nonlinear_screen_rejects_expost_features() -> None:
    state, calendar, assignments = _fixture()
    result = attach_failure_mode_targets(state, calendar, assignments)
    result["expost__future_error"] = result["target__any_failure"].astype(float)
    with pytest.raises(ValueError, match="Ex-post"):
        nonlinear_feature_screen(
            result,
            ["shock", "expost__future_error"],
            "target__any_failure",
            maximum=2,
            bins=5,
        )
    with pytest.raises(ValueError, match="Ex-post failure features"):
        nonlinear_feature_screen(
            result,
            ["shock", "clean_exec"],
            "target__any_failure",
            maximum=2,
            bins=5,
        )
    with pytest.raises(ValueError, match="Ex-post failure features"):
        nonlinear_feature_screen(
            result,
            ["shock", "failure_mode_available_day"],
            "target__any_failure",
            maximum=2,
            bins=5,
        )


def test_chronological_detector_emits_only_forward_fold_predictions() -> None:
    state, calendar, assignments = _fixture()
    labelled = attach_failure_mode_targets(state, calendar, assignments, lead_days=(1,))
    predictions, metrics, selections = chronological_failure_detection(
        labelled,
        config=ProspectiveFailureDetectorConfig(
            min_train_days=100,
            eval_days=40,
            inner_validation_days=30,
            min_positive_days=3,
            max_features=3,
            alert_quantile=0.90,
            lead_days=(1,),
        ),
    )
    assert not predictions.empty
    assert not metrics.empty
    assert not selections.empty
    assert (predictions["day"] >= predictions["train_end"]).all()
    assert (predictions["day"] < predictions["eval_end"]).all()
    assert predictions["expected_failure_severity"].ge(0.0).all()
    assert predictions["expected_failure_severity"].max() > 0.0
    assert predictions["risk_aleatoric_uncertainty"].between(0.0, 1.0).all()
    assert predictions["risk_support_uncertainty"].gt(0.0).all()
    assert predictions["risk"].between(0.0, 1.0).all()
    assert predictions["risk_raw"].between(0.0, 1.0).all()
    assert metrics["probability_calibration"].isin(
        {
            "platt_logit_inner_validation",
            "identity_insufficient_support",
            "identity_nonmonotonic_platt",
        }
    ).all()
    assert not selections["feature"].str.startswith(("target__", "expost__")).any()
    assert "mean_ev_after_1pct" not in set(selections["feature"])
    assert "next1d_failure_onset" in set(metrics["failure_mode"])
    forward = metrics.loc[metrics["target_horizon_days"].gt(0)]
    assert not forward.empty
    assert (
        pd.to_datetime(forward["train_label_max_day"], utc=True)
        + pd.to_timedelta(forward["target_horizon_days"], unit="D")
        < pd.to_datetime(forward["train_end"], utc=True)
    ).all()


def test_platt_calibration_preserves_score_order_and_probability_bounds() -> None:
    validation_score = np.array([0.10, 0.20, 0.35, 0.65, 0.80, 0.90])
    validation_target = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.15, 0.50, 0.85])

    calibrated_validation, calibrated, method = _calibrate_probability_scores(
        validation_score,
        validation_target,
        score,
        method="platt",
        seed=7,
    )

    assert method == "platt_logit_inner_validation"
    assert np.all(np.diff(calibrated_validation) > 0.0)
    assert np.all(np.diff(calibrated) > 0.0)
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))


def test_legacy_batch_layout_temporal_aegmm_features_are_identified() -> None:
    assert is_batch_layout_dependent_ae_gmm_feature("cluster_speed")
    assert is_batch_layout_dependent_ae_gmm_feature(
        "state_delta1__gmm_posterior_accel_1"
    )
    assert not is_batch_layout_dependent_ae_gmm_feature("gmm_entropy")
    assert not is_batch_layout_dependent_ae_gmm_feature(
        "state_delta1__gmm_entropy"
    )


def test_platt_calibration_does_not_reverse_an_inverted_validation_slice() -> None:
    validation_score = np.array([0.10, 0.20, 0.35, 0.65, 0.80, 0.90])
    validation_target = np.array([1, 1, 1, 0, 0, 0])
    score = np.array([0.15, 0.50, 0.85])

    calibrated_validation, calibrated, method = _calibrate_probability_scores(
        validation_score,
        validation_target,
        score,
        method="platt",
        seed=7,
    )

    assert method == "identity_nonmonotonic_platt"
    np.testing.assert_allclose(calibrated_validation, validation_score)
    np.testing.assert_allclose(calibrated, score)


def test_forward_target_purge_excludes_labels_touching_boundary() -> None:
    days = pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC")
    frame = pd.DataFrame({"day": days, "target__next3d__failure_onset": False})
    boundary = pd.Timestamp("2025-01-10", tz="UTC")

    result = purged_before_boundary(
        frame,
        boundary=boundary,
        target="target__next3d__failure_onset",
    )

    assert target_horizon_days("target__next3d__failure_onset") == 3
    assert result["day"].max() == pd.Timestamp("2025-01-05", tz="UTC")


def test_mode_label_waits_for_full_episode_recovery_horizon() -> None:
    state, calendar, assignments = _fixture(40)
    assignments = assignments.copy()
    assignments["event_end"] = pd.Timestamp("2025-01-20", tz="UTC")
    labelled = attach_failure_mode_targets(
        state,
        calendar,
        assignments,
        lead_days=(1,),
    )
    mode = str(labelled["failure_mode"].dropna().iloc[0])
    onset_target = f"target__mode_onset__{mode}"

    # The descriptive mode uses recovery features up to 14 days after the
    # episode. It must not be admitted to a fold before that path resolves.
    mode_rows = labelled.loc[labelled["failure_mode"].eq(mode)]
    expected_available = pd.Timestamp("2025-02-04", tz="UTC")
    assert (
        pd.to_datetime(mode_rows["failure_mode_available_day"], utc=True)
        .eq(expected_available)
        .all()
    )

    labelled[onset_target] = labelled["failure_mode"].eq(mode)
    labelled[f"availability__{onset_target}"] = (
        labelled["failure_mode_available_day"]
    )
    purged = purged_before_boundary(
        labelled,
        boundary=pd.Timestamp("2025-02-01", tz="UTC"),
        target=onset_target,
    )
    assert not purged[onset_target].fillna(False).any()


def test_negative_mode_window_waits_for_other_mode_classification() -> None:
    days = pd.date_range("2025-01-01", periods=6, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "day": days,
            "side_name": "long",
            "archetype_policy_key": "compression",
        }
    )
    any_failure = pd.Series([False, False, True, False, False, False])
    mode_available = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    mode_available.iloc[2] = pd.Timestamp("2025-01-18", tz="UTC")

    availability = _future_window_availability(
        frame,
        any_failure,
        mode_available,
        horizon_days=3,
    )

    # A row on Jan 1 can only learn that its Jan 2-4 window contains no onset
    # of the target mode after the Jan 3 event is classified as another mode.
    assert availability.iloc[0] == pd.Timestamp("2025-01-18", tz="UTC")
    assert availability.iloc[1] == pd.Timestamp("2025-01-18", tz="UTC")
