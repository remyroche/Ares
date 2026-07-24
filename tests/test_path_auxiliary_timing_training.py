"""Tests for the strict shared-HPO side-local timing-CDF trainer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.path_auxiliary_timing_training import (
    TIMING_CDF_HORIZONS,
    TIMING_CDF_HPO_MAX_TRIALS,
    TIMING_CDF_HPO_STALE_STOP,
    TIMING_CDF_PURGE_HOURS,
    fit_side_local_timing_cdf_family,
    predict_side_local_timing_cdf_family,
)


def test_censored_12h_clock_is_not_mislabelled_as_a_12h_hit() -> None:
    import extreme_price_movements.path_auxiliary_timing_training as training

    targets = training._horizon_targets(
        np.asarray([2.0, 12.0], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([True, True]),
    )
    assert targets[2].tolist() == [1.0, 0.0]
    assert targets[12].tolist() == [1.0, 0.0]


def test_monotone_projection_never_moves_the_shared_12h_event_owner() -> None:
    import extreme_price_movements.path_auxiliary_timing_training as training

    raw = {
        2: np.asarray([0.90, 0.10], dtype=np.float32),
        4: np.asarray([0.10, 0.80], dtype=np.float32),
        8: np.asarray([0.80, 0.20], dtype=np.float32),
        12: np.asarray([0.40, 0.60], dtype=np.float32),
    }
    projected, audit = training._project_oof_cdf(raw, np.asarray([True, True]))
    np.testing.assert_array_equal(projected[12], raw[12])
    assert np.all(
        np.diff(
            np.column_stack([projected[hours] for hours in TIMING_CDF_HORIZONS]), axis=1
        )
        >= -1e-7
    )
    assert audit["fixed_horizon_hours"] == 12
    assert audit["fixed_final_horizon_probability"] is True


def _timing_data() -> tuple[
    pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex
]:
    """Return enough pre-May and May--July rows for both side-local streams."""

    rng = np.random.default_rng(92)
    timestamps = pd.date_range(
        "2026-03-01T00:00:00Z",
        "2026-08-01T00:00:00Z",
        freq="6h",
        inclusive="left",
    )
    rows = len(timestamps)
    sides = np.where(np.arange(rows) % 2 == 0, "long", "short")
    x1 = rng.normal(size=rows).astype(np.float32)
    x2 = rng.normal(size=rows).astype(np.float32)
    x3 = rng.normal(size=rows).astype(np.float32)
    side_effect = np.where(sides == "long", 0.20, -0.10)
    hit = (x1 + 0.4 * x2 + side_effect + rng.normal(0.0, 0.35, rows) > 0.0).astype(
        np.float32
    )
    hit_time = np.where(
        x1 > 0.7,
        1.0,
        np.where(x1 > 0.0, 3.0, np.where(x2 > 0.0, 7.0, 11.0)),
    ).astype(np.float32)
    timing = np.where(hit > 0.5, hit_time, 12.0).astype(np.float32)
    train_mask = np.ones(rows, dtype=bool)
    # This is a real incomplete canonical timing path: it must never affect a
    # fit or metric, yet its May--July decision row must still be predicted.
    train_mask[
        (timestamps >= pd.Timestamp("2026-05-01T00:00:00Z"))
        & (np.arange(rows) % 17 == 0)
    ] = False
    return (
        pd.DataFrame({"x1": x1, "x2": x2, "x3": x3}),
        timing,
        hit,
        train_mask,
        sides,
        timestamps,
    )


def _preset_params() -> dict[str, dict[str, float | int]]:
    params: dict[str, float | int] = {
        "n_estimators": 25,
        "learning_rate": 0.10,
        "num_leaves": 7,
        "min_child_samples": 10,
    }
    return {"long": params, "short": params}


def _horizon_feature_contract() -> dict[str, dict[int | str, list[str]]]:
    return {
        "long": {2: ["x1"], "4h": ["x2"], 8: ["x1", "x3"], 12: ["x2", "x3"]},
        "short": {2: ["x2"], 4: ["x1"], "8h": ["x2", "x3"], 12: ["x1", "x3"]},
    }


def _normalized_horizon_feature_contract() -> dict[str, dict[int, list[str]]]:
    return {
        "long": {2: ["x1"], 4: ["x2"], 8: ["x1", "x3"], 12: ["x2", "x3"]},
        "short": {2: ["x2"], 4: ["x1"], 8: ["x2", "x3"], 12: ["x1", "x3"]},
    }


def test_feature_contract_accepts_shared_and_legacy_side_sequences() -> None:
    import extreme_price_movements.path_auxiliary_timing_training as training

    X, _, _, _, _, _ = _timing_data()
    shared = training._features_by_side_and_horizon(
        ["x1"], sides=("long", "short"), columns=X.columns
    )
    assert all(
        features == ["x1"]
        for by_horizon in shared.values()
        for features in by_horizon.values()
    )
    legacy_side = training._features_by_side_and_horizon(
        {"long": ["x1", "x2"], "short": ["x3"]},
        sides=("long", "short"),
        columns=X.columns,
    )
    assert all(features == ["x1", "x2"] for features in legacy_side["long"].values())
    assert all(features == ["x3"] for features in legacy_side["short"].values())


def _fit_with_presets() -> tuple[dict[str, object], tuple[object, ...]]:
    X, timing, hit, train_mask, sides, timestamps = _timing_data()
    family = fit_side_local_timing_cdf_family(
        X,
        timing,
        hit,
        timing_train_mask=train_mask,
        sides=sides,
        selected_features=_horizon_feature_contract(),
        timestamps=timestamps,
        label_resolved_at=timestamps + pd.Timedelta(hours=13),
        selection_hpo_reference_end="2026-05-01T00:00:00Z",
        preset_params_by_side=_preset_params(),
        random_state=19,
        n_jobs=1,
    )
    return family, (X, timing, hit, train_mask, sides, timestamps)


def test_side_local_timing_cdf_oof_is_full_row_purged_and_monotone() -> None:
    family, data = _fit_with_presets()
    _, _, _, train_mask, _, timestamps = data
    expected_oof = (timestamps >= pd.Timestamp("2026-05-01T00:00:00Z")) & (
        timestamps < pd.Timestamp("2026-08-01T00:00:00Z")
    )

    np.testing.assert_array_equal(family["oof_prediction_mask"], expected_oof)
    assert np.all(family["oof_fold_ids"][expected_oof] >= 0)
    assert set(family["side_models"]) == {"long", "short"}
    assert family["selected_features_by_side"]["short"] == ["x2", "x1", "x3"]
    assert (
        family["selected_features_by_side_and_horizon"]
        == _normalized_horizon_feature_contract()
    )
    for side, contracts in _normalized_horizon_feature_contract().items():
        assert family["side_models"][side]["selected_features_by_horizon"] == contracts
        for hours, features in contracts.items():
            assert (
                family["side_models"][side]["final_models"][hours].feature_name_
                == features
            )
    for hours in TIMING_CDF_HORIZONS:
        assert np.isfinite(
            family["oof_predictions_by_horizon"][hours][expected_oof]
        ).all()
        assert np.isfinite(
            family["oof_predictions_by_horizon"][hours][expected_oof & ~train_mask]
        ).all()
    cdf = np.column_stack(
        [
            family["oof_predictions_by_horizon"][hours][expected_oof]
            for hours in TIMING_CDF_HORIZONS
        ]
    )
    assert np.all(np.diff(cdf, axis=1) >= -1e-7)
    assert family["monotone_projection"]["applied_after_outer_fold_prediction"] is True
    assert family["monotone_projection"]["fixed_final_horizon_probability"] is True
    np.testing.assert_array_equal(
        family["oof_predictions_by_horizon"][12][expected_oof],
        family["raw_oof_predictions_by_horizon"][12][expected_oof],
    )
    assert family["oof_contract"].startswith("fixed expanding May/June/July")

    assert len(family["fold_provenance"]) == 6
    for fold in family["fold_provenance"]:
        assert fold["predicted_validation_rows"] == fold["validation_rows"]
        assert fold["conditional_validation_rows"] <= fold["validation_rows"]
        assert fold["resolution_before_valid_start_assertion"] is True
        assert pd.Timestamp(fold["training_label_resolved_max"]) < pd.Timestamp(
            fold["valid_start"]
        )
    for state in family["side_models"].values():
        assert state["hpo"]["reused_preset_params"] is True
        assert state["hpo"]["stale_trial_patience"] == TIMING_CDF_HPO_STALE_STOP
        assert "excluded from OOF metrics" in state["final_refit_contract"]["row_rule"]
        assert set(state["final_models"]) == set(TIMING_CDF_HORIZONS)


def test_hpo_uses_one_joint_parameter_set_across_all_four_horizons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extreme_price_movements.path_auxiliary_timing_training as training

    X, timing, hit, train_mask, _, timestamps = _timing_data()
    # One side keeps this test small while still exercising the side-local HPO
    # path.  A short fixed tree budget makes the test independent of machine
    # speed without changing the joint-score contract.
    sides = np.repeat("long", len(X))

    def tiny_params(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "objective": "binary",
            "n_estimators": 12,
            "learning_rate": 0.10,
            "num_leaves": 7,
            "min_child_samples": 8,
            "min_split_gain": 0.001,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "max_bin": 63,
            "random_state": 7,
            "n_jobs": 1,
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
        }

    monkeypatch.setattr(training, "_suggest_params", tiny_params)
    events: list[tuple[str, dict[str, object]]] = []
    family = training.fit_side_local_timing_cdf_family(
        X,
        timing,
        hit,
        timing_train_mask=train_mask,
        sides=sides,
        selected_features={
            "long": {2: ["x1"], 4: ["x2"], 8: ["x1", "x3"], 12: ["x2", "x3"]}
        },
        timestamps=timestamps,
        label_resolved_at=timestamps + pd.Timedelta(hours=13),
        selection_hpo_reference_end="2026-05-01T00:00:00Z",
        n_trials=1,
        hpo_rows=120,
        random_state=7,
        n_jobs=1,
        progress_callback=lambda event, payload: events.append((event, dict(payload))),
    )

    state = family["side_models"]["long"]
    hpo = state["hpo"]
    assert hpo["trial_count"] == 1
    assert hpo["maximum_trials"] == TIMING_CDF_HPO_MAX_TRIALS
    assert hpo["stale_trial_patience"] == TIMING_CDF_HPO_STALE_STOP
    assert set(hpo["best_trial_horizon_scores"]) == {"2", "4", "8", "12"}
    assert "joint score across 2/4/8/12-hour" in hpo["contract"]
    assert hpo["purged_fold_provenance"]
    for fold in hpo["purged_fold_provenance"]:
        assert fold["resolution_before_valid_start_assertion"] is True
        assert pd.Timestamp(fold["training_label_resolved_max"]) < pd.Timestamp(
            fold["valid_start"]
        )
    # There is one shared parameter vector, reused for each horizon's model.
    assert state["best_params"]["n_estimators"] >= 25
    assert state["selected_features_by_horizon"] == {
        2: ["x1"],
        4: ["x2"],
        8: ["x1", "x3"],
        12: ["x2", "x3"],
    }
    assert {
        payload["horizon_hours"]
        for event, payload in events
        if event == "hpo_horizon_start"
    } == set(TIMING_CDF_HORIZONS)
    assert {
        payload["horizon_hours"]
        for event, payload in events
        if event == "oof_horizon_start"
    } == set(TIMING_CDF_HORIZONS)
    assert events[0][0] == "timing_cdf_training_start"
    assert events[-1][0] == "timing_cdf_training_complete"


def test_final_prediction_projects_side_local_cdf_and_hpo_cap_is_enforced() -> None:
    family, data = _fit_with_presets()
    X, timing, hit, train_mask, sides, timestamps = data
    scored = predict_side_local_timing_cdf_family(family, X, sides=sides)
    assert scored["prediction_mask"].all()
    cdf = np.column_stack(
        [scored["predictions_by_horizon"][hours] for hours in TIMING_CDF_HORIZONS]
    )
    assert np.all(np.diff(cdf, axis=1) >= -1e-7)
    np.testing.assert_array_equal(
        scored["predictions_by_horizon"][12],
        scored["raw_predictions_by_horizon"][12],
    )
    assert np.all(
        (scored["expected_censored_time_hours"] >= 0.0)
        & (scored["expected_censored_time_hours"] <= 12.0)
    )

    with pytest.raises(ValueError, match="production cap of 40"):
        fit_side_local_timing_cdf_family(
            X,
            timing,
            hit,
            timing_train_mask=train_mask,
            sides=sides,
            selected_features=["x1", "x2"],
            timestamps=timestamps,
            label_resolved_at=timestamps + pd.Timedelta(hours=TIMING_CDF_PURGE_HOURS),
            selection_hpo_reference_end="2026-05-01T00:00:00Z",
            n_trials=TIMING_CDF_HPO_MAX_TRIALS + 1,
        )
