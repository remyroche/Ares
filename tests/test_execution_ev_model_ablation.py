from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import extreme_price_movements.execution_ev_model_ablation as ablation
from extreme_price_movements.execution_ev_meta import FeatureProvenance
from extreme_price_movements.path_archetype_labels import PATH_SHAPE_TYPES


class _StubRegressor:
    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = coefficients

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        values = x.to_numpy(dtype=float)
        return np.c_[np.ones(len(values)), values] @ self.coefficients


@pytest.fixture(autouse=True)
def _stub_regressor_fitter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep focused tests independent of slow optional tree-model imports."""

    def fit(
        _algorithm: str, x: pd.DataFrame, y: np.ndarray, **_kwargs: object
    ) -> _StubRegressor:
        values = x.to_numpy(dtype=float)
        coefficients = np.linalg.lstsq(
            np.c_[np.ones(len(values)), values], np.asarray(y, dtype=float), rcond=None
        )[0]
        return _StubRegressor(coefficients)

    monkeypatch.setattr(ablation, "_fit_regressor", fit)


def _frame(rows: int = 192) -> tuple[pd.DataFrame, dict[str, FeatureProvenance]]:
    times = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    position = np.arange(rows)
    x = np.linspace(-1.0, 1.0, rows)
    side = np.where(position % 2 == 0, "long", "short")
    winners = position % len(PATH_SHAPE_TYPES)
    probabilities = np.full((rows, len(PATH_SHAPE_TYPES)), 0.03, dtype=float)
    probabilities[np.arange(rows), winners] = 1.0 - 0.03 * (len(PATH_SHAPE_TYPES) - 1)
    entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
    alpha = 0.004 * x
    net_ev = alpha + 0.006 * x + np.where(side == "long", 0.0015, -0.0015)
    frame = pd.DataFrame(
        {
            "__ts__": times,
            "execution_label_end_utc": times + pd.Timedelta(hours=12),
            "side_name": side,
            "catboost_archetype": [PATH_SHAPE_TYPES[index] for index in winners],
            "execution_net_ev_12h": net_ev,
            "execution_gross_ev_12h": net_ev + 0.002,
            "existing_alpha_ev": alpha,
            "pred_time_to_mfe_12h": 5.0 - 2.0 * x,
            "pred_peak_mfe_12h": 0.02 + 0.006 * x,
            "pred_mae_before_meaningful_mfe_atr": 0.8 - 0.3 * x,
            "pred_bars_before_price_stops_decreasing": 6.0 - x,
            "pred_favorable_path_slope_atr_per_hour": 0.3 + 0.2 * x,
            "catboost_entropy": entropy,
            "base_prediction_uncertainty": 0.15 + 0.05 * np.abs(x),
            "meta_leaf_support_log1p": 2.0 + 0.4 * x,
            "base_archetype_label__family__trend": (x > 0.0).astype(float),
            "available_at": times,
        }
    )
    for index in range(len(PATH_SHAPE_TYPES)):
        frame[f"catboost_p_{index}"] = probabilities[:, index]
    provenance: dict[str, FeatureProvenance] = {
        "existing_alpha_ev": FeatureProvenance(
            "alpha_score", "frozen alpha", available_at_col="available_at"
        ),
        "pred_time_to_mfe_12h": FeatureProvenance(
            "time_to_mfe", "frozen aux", available_at_col="available_at"
        ),
        "pred_peak_mfe_12h": FeatureProvenance(
            "peak_mfe", "frozen aux", available_at_col="available_at"
        ),
        "pred_mae_before_meaningful_mfe_atr": FeatureProvenance(
            "mae_before_meaningful_mfe", "frozen aux", available_at_col="available_at"
        ),
        "pred_bars_before_price_stops_decreasing": FeatureProvenance(
            "adverse_turn_timing", "frozen aux", available_at_col="available_at"
        ),
        "pred_favorable_path_slope_atr_per_hour": FeatureProvenance(
            "favorable_path_slope", "frozen aux", available_at_col="available_at"
        ),
        "catboost_entropy": FeatureProvenance(
            "catboost_entropy",
            "frozen CatBoost entropy",
            available_at_col="available_at",
        ),
        "base_prediction_uncertainty": FeatureProvenance(
            "prediction_uncertainty",
            "frozen alpha uncertainty",
            available_at_col="available_at",
        ),
        "meta_leaf_support_log1p": FeatureProvenance(
            "leaf_support", "frozen alpha leaf support", available_at_col="available_at"
        ),
        "base_archetype_label__family__trend": FeatureProvenance(
            "base_archetype_labels",
            "frozen base archetype label",
            available_at_col="available_at",
        ),
        "catboost_archetype": FeatureProvenance(
            "predicted_path_archetype",
            "frozen CatBoost assignment",
            available_at_col="available_at",
            model_input=False,
        ),
    }
    provenance.update(
        {
            f"catboost_p_{index}": FeatureProvenance(
                "catboost_probabilities",
                "frozen CatBoost probability",
                available_at_col="available_at",
            )
            for index in range(len(PATH_SHAPE_TYPES))
        }
    )
    return frame, provenance


def _config(
    *, algorithms: tuple[str, ...] = ("extra_trees",)
) -> ablation.ExecutionEVModelAblationConfig:
    return ablation.ExecutionEVModelAblationConfig(
        n_splits=2,
        min_train_rows=48,
        min_fit_rows=12,
        n_estimators=8,
        n_jobs=1,
        mda_min_features=14,
        mda_max_steps=1,
        mda_repeats=1,
        isotonic_min_rows=4,
        hpo_trials=0,
        recent_ev_correction_enabled=False,
        algorithms=algorithms,
    )


def test_contract_rejects_late_and_outcome_like_inputs() -> None:
    frame, provenance = _frame()
    late = frame.copy()
    late.loc[0, "available_at"] = late.loc[0, "__ts__"] + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="available after entry"):
        ablation.validate_execution_ev_model_ablation_contract(late, provenance)

    leaked = frame.assign(actual_execution_ev_proxy=0.0)
    leaked_provenance = {
        **provenance,
        "actual_execution_ev_proxy": FeatureProvenance(
            "alpha_score", "incorrect outcome proxy", available_at_col="available_at"
        ),
    }
    with pytest.raises(ValueError, match="outcome-derived"):
        ablation.validate_execution_ev_model_ablation_contract(
            leaked, leaked_provenance
        )


def test_contract_allows_frozen_oof_future_slope_but_not_raw_future_target() -> None:
    frame, provenance = _frame()
    frame["future_slope"] = frame["pred_favorable_path_slope_atr_per_hour"]
    frozen = {
        **provenance,
        "future_slope": FeatureProvenance(
            "favorable_path_slope",
            "frozen OOF future-slope prediction",
            available_at_col="available_at",
        ),
    }
    raw_columns, _ = ablation.validate_execution_ev_model_ablation_contract(
        frame, frozen
    )
    assert "future_slope" in raw_columns

    raw = {
        **frozen,
        "future_slope": FeatureProvenance(
            "favorable_path_slope",
            "raw future target",
            oof_or_frozen=False,
            available_at_col="available_at",
        ),
    }
    with pytest.raises(ValueError, match="outcome-derived"):
        ablation.validate_execution_ev_model_ablation_contract(frame, raw)


def test_contract_includes_only_explicit_additional_families() -> None:
    frame, provenance = _frame()
    frame["oof_literal_reach_probability"] = np.linspace(0.1, 0.9, len(frame))
    augmented = {
        **provenance,
        "oof_literal_reach_probability": FeatureProvenance(
            "literal_reach_probability",
            "strict outer-OOF literal classifier",
            available_at_col="available_at",
        ),
    }
    baseline, _ = ablation.validate_execution_ev_model_ablation_contract(
        frame, augmented
    )
    repaired, _ = ablation.validate_execution_ev_model_ablation_contract(
        frame,
        augmented,
        additional_input_families=("literal_reach_probability",),
    )
    assert "oof_literal_reach_probability" not in baseline
    assert "oof_literal_reach_probability" in repaired


def test_contract_uses_signed_catboost_class_order() -> None:
    frame, provenance = _frame()
    signed_order = tuple(PATH_SHAPE_TYPES[:-1])
    keep = frame["catboost_archetype"].isin(signed_order)
    frame = frame.loc[keep].drop(columns=f"catboost_p_{len(PATH_SHAPE_TYPES) - 1}")
    frame = frame.reset_index(drop=True)
    probability_columns = [f"catboost_p_{index}" for index in range(len(signed_order))]
    probabilities = frame[probability_columns].to_numpy(dtype=float)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    frame.loc[:, probability_columns] = probabilities
    frame["catboost_entropy"] = -np.sum(
        probabilities * np.log(probabilities), axis=1
    )
    signed = {
        name: replace(spec, class_order=signed_order)
        for name, spec in provenance.items()
        if name != f"catboost_p_{len(PATH_SHAPE_TYPES) - 1}"
    }
    _, levels = ablation.validate_execution_ev_model_ablation_contract(frame, signed)
    assert levels == signed_order


def test_side_local_oof_has_purge_cutoff_and_train_only_isotonic() -> None:
    frame, provenance = _frame()
    bundle = ablation.train_execution_ev_model_ablation(
        frame, provenance, config=_config()
    )
    oof = bundle.oof_predictions["extra_trees__direct__without_hpo__all_features"]
    provenance_frame = bundle.oof_provenance
    scored = provenance_frame["execution_ev_model_ablation_oof_fold"].notna()
    assert scored.any()
    assert oof.loc[scored].notna().all()
    assert (
        provenance_frame.loc[
            scored, "execution_ev_model_ablation_oof_train_decision_cutoff_utc"
        ]
        < frame.loc[scored, "__ts__"]
    ).all()
    audits = bundle.report["oof_audit"]["extra_trees"]["direct"]["without_hpo"][
        "mda_1se"
    ]
    successful = [row for row in audits if row["status"] == "ok"]
    assert successful
    for row in successful:
        assert row["train_sides"] == [row["side"]]
        assert row["validation_sides"] == [row["side"]]
        assert row["train_decision_cutoff_utc"] < row["validation_start_utc"]
        assert row["isotonic"]["train_rows"] <= row["train_rows"]


def test_oof_and_final_prediction_do_not_require_outcomes_at_inference() -> None:
    frame, provenance = _frame()
    bundle = ablation.train_execution_ev_model_ablation(
        frame, provenance, config=_config()
    )
    inference = frame.drop(
        columns=[
            "execution_net_ev_12h",
            "execution_gross_ev_12h",
            "execution_label_end_utc",
        ]
    )
    scored = ablation.predict_execution_ev_model_ablation_bundle(
        bundle, inference, algorithms=("extra_trees",)
    )
    assert set(scored) == {
        "extra_trees__direct__without_hpo__all_features",
        "extra_trees__direct__without_hpo__mda_1se",
        "extra_trees__residual__without_hpo__all_features",
        "extra_trees__residual__without_hpo__mda_1se",
    }
    assert np.isfinite(scored.to_numpy(dtype=float)).all()


def test_residual_only_post_screening_run_skips_direct_target() -> None:
    frame, provenance = _frame()
    bundle = ablation.train_execution_ev_model_ablation(
        frame,
        provenance,
        config=replace(_config(), target_modes=("residual",)),
    )
    assert bundle.oof_predictions.columns.tolist() == [
        "baseline__frozen_alpha",
        "extra_trees__residual__without_hpo__all_features",
        "extra_trees__residual__without_hpo__mda_1se",
    ]


def test_leaderboard_keeps_fixed_and_hpo_arms_distinct() -> None:
    frame, provenance = _frame()
    bundle = ablation.train_execution_ev_model_ablation(
        frame,
        provenance,
        config=replace(_config(algorithms=ablation.ALGORITHM_NAMES), hpo_trials=1),
    )
    leaderboard = pd.DataFrame(bundle.report["leaderboard"])
    model_rows = leaderboard.loc[
        leaderboard["algorithm"].isin(ablation.ALGORITHM_NAMES)
    ]
    assert set(model_rows["algorithm"]) == set(ablation.ALGORITHM_NAMES)
    assert set(model_rows["target_mode"]) == {"direct", "residual"}
    assert set(model_rows["hpo_arm"]) == {"without_hpo", "with_hpo"}
    assert set(model_rows["feature_arm"]) == {"all_features", "mda_1se"}


def test_promotion_top10_is_global_and_after_21d_admission_calibrator() -> None:
    net = np.array([0.01, -0.02, 0.03, 0.04, -0.01])
    gross = net + 0.001
    predictions = pd.DataFrame(
        {
            "extra_trees__direct__with_hpo__mda_1se": np.arange(5, dtype=float),
            (
                "extra_trees__direct__with_hpo__mda_1se"
                "__recent_ev_catboost_predicted_archetype"
            ): np.arange(5, dtype=float)[::-1],
        }
    )
    corrected = predictions.columns[-1]
    leaderboard = ablation._leaderboard(
        net,
        gross,
        predictions,
        top_k_fraction=0.20,
        evaluation_mask=np.ones(5, dtype=bool),
        promotion_eligible_columns=frozenset({corrected}),
    )

    promoted = leaderboard.loc[leaderboard["promotion_eligible"]]
    assert promoted["prediction"].tolist() == [corrected]
    assert promoted["ranking_scope"].tolist() == ["global_shared_outer_oof"]
    assert promoted["ranking_stage"].tolist() == [
        "after_causal_21d_admission_calibrator"
    ]
    assert promoted["top_k_rows"].tolist() == [1]
    diagnostic = leaderboard.loc[~leaderboard["promotion_eligible"]]
    assert (
        diagnostic["ranking_stage"]
        .eq("before_admission_calibrator_diagnostic_only")
        .all()
    )


def test_direct_and_residual_arms_share_oof_rows_and_rank_absolute_net_ev() -> None:
    frame, provenance = _frame()
    bundle = ablation.train_execution_ev_model_ablation(
        frame, provenance, config=_config(algorithms=("extra_trees",))
    )
    columns = [
        column
        for column in bundle.oof_predictions
        if column.startswith("extra_trees__") and column.count("__") == 3
    ]
    masks = [bundle.oof_predictions[column].notna().to_numpy() for column in columns]
    assert len(columns) == 4
    assert all(np.array_equal(mask, masks[0]) for mask in masks[1:])
    leaderboard = pd.DataFrame(bundle.report["leaderboard"])
    model_rows = leaderboard.loc[leaderboard["algorithm"].eq("extra_trees")]
    assert set(model_rows["oof_rows"]) == {int(masks[0].sum())}
    assert set(model_rows["target_mode"]) == {"direct", "residual"}


@pytest.mark.parametrize(
    "family", ["prediction_uncertainty", "leaf_support", "base_archetype_labels"]
)
def test_contract_requires_repaired_input_families(family: str) -> None:
    frame, provenance = _frame()
    reduced = {name: spec for name, spec in provenance.items() if spec.family != family}
    with pytest.raises(ValueError, match=family):
        ablation.validate_execution_ev_model_ablation_contract(frame, reduced)


def test_mda_one_se_prefers_smallest_feature_set_within_one_standard_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = pd.DataFrame({"a": [0.0], "b": [1.0], "c": [2.0]})
    scores = {
        3: {"status": "ok", "objective_mean": 1.00, "objective_se": 0.10},
        2: {"status": "ok", "objective_mean": 0.93, "objective_se": 0.01},
        1: {"status": "ok", "objective_mean": 0.70, "objective_se": 0.01},
    }

    monkeypatch.setattr(
        ablation,
        "_evaluate_inner_feature_set",
        lambda _algorithm, _x, _net, _gross, features, _folds, **_kwargs: scores[
            len(features)
        ],
    )
    monkeypatch.setattr(
        ablation,
        "_permutation_mda",
        lambda _algorithm, _x, _net, _gross, features, _folds, **_kwargs: {
            name: float(-index) for index, name in enumerate(features)
        },
    )
    selected, report = ablation.select_features_by_mda_one_se(
        "extra_trees",
        x,
        np.array([0.0]),
        np.array([0.0]),
        [object()],
        params={},
        config=ablation.ExecutionEVModelAblationConfig(
            mda_min_features=1, mda_max_steps=2
        ),
    )
    assert selected == ["a", "b"]
    assert report["selected_step"] == 1
    assert report["one_se_threshold"] == pytest.approx(0.90)


def test_train_only_isotonic_mapping_is_monotone_and_has_identity_fallback() -> None:
    mapping = ablation.fit_train_only_isotonic_ev_mapping(
        np.array([-1.0, 0.0, 1.0, 2.0]),
        np.array([-0.02, -0.01, 0.01, 0.03]),
        min_rows=4,
    )
    prediction = mapping.predict(np.array([-0.5, 0.5, 1.5]))
    assert mapping.status == "isotonic_train_oof"
    assert np.all(np.diff(prediction) >= 0.0)
    fallback = ablation.fit_train_only_isotonic_ev_mapping(
        np.array([0.0]), np.array([0.1]), min_rows=4
    )
    np.testing.assert_allclose(fallback.predict(np.array([0.25])), [0.25])


def test_recent_ev_correction_is_daily_causal_and_reports_gmm_route_unavailability() -> (
    None
):
    frame, provenance = _frame()
    mapped = frame["existing_alpha_ev"].to_numpy(dtype=float)
    config = _config()
    corrected, report = ablation.apply_execution_ev_causal_recent_ev_correction(
        frame,
        mapped,
        frame["execution_net_ev_12h"].to_numpy(dtype=float),
        provenance,
        route="catboost_predicted_archetype",
        config=config,
    )
    assert report["status"] == "available"
    assert report["days"] > 1
    # The first UTC snapshot has no outcome resolved before it.
    np.testing.assert_allclose(corrected[:24], mapped[:24])

    gmm_corrected, gmm_report = ablation.apply_execution_ev_causal_recent_ev_correction(
        frame,
        mapped,
        frame["execution_net_ev_12h"].to_numpy(dtype=float),
        provenance,
        route="gmm_archetype",
        config=config,
    )
    assert gmm_report["status"] == "unavailable_missing_column"
    np.testing.assert_allclose(gmm_corrected, mapped)


@pytest.mark.parametrize("algorithm", ablation.ALGORITHM_NAMES)
def test_all_required_regressor_families_fit_via_injected_stub(algorithm: str) -> None:
    x = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 16), "z": np.arange(16, dtype=float)})
    y = 0.01 * x["x"].to_numpy() + 0.001 * x["z"].to_numpy()
    config = ablation.ExecutionEVModelAblationConfig(n_estimators=5, n_jobs=1)
    model = ablation._fit_regressor(
        algorithm, x, y, params=ablation._fixed_params(algorithm, config)
    )
    assert np.isfinite(np.asarray(model.predict(x), dtype=float)).all()


def test_hpo_is_bounded_deterministic_and_not_recycled_fixed_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 8)})
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC"),
            "execution_label_end_utc": pd.date_range(
                "2026-01-01 12:00", periods=8, freq="h", tz="UTC"
            ),
        }
    )
    monkeypatch.setattr(
        ablation,
        "_evaluate_inner_feature_set",
        lambda _algorithm, _x, _net, _gross, _features, _folds, **kwargs: {
            "status": "ok",
            "objective_mean": float(kwargs["params"]["learning_rate"]),
            "objective_se": 0.0,
        },
    )
    config = ablation.ExecutionEVModelAblationConfig(hpo_trials=5, n_estimators=10)
    params, report = ablation._tune_params(
        "lgbm",
        x,
        np.zeros(len(x)),
        np.zeros(len(x)),
        [object()],
        features=["x"],
        frame=frame,
        config=config,
    )
    assert report["status"] == "deterministic_randomized_purged_oof_hpo"
    assert len(report["trials"]) == 5
    assert len({trial["params"]["learning_rate"] for trial in report["trials"]}) == 5
    assert params["learning_rate"] == max(
        trial["params"]["learning_rate"] for trial in report["trials"]
    )
