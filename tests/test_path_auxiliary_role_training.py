import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.path_auxiliary_role_training import (
    FIXED_MAY_JULY_OOF_MONTHS,
    fit_auxiliary_role_model,
    select_auxiliary_role_features,
)


def _role_training_data() -> tuple[
    pd.DataFrame, np.ndarray, np.ndarray, pd.DatetimeIndex
]:
    rng = np.random.default_rng(71)
    timestamps = pd.date_range(
        "2026-03-01T00:00:00Z",
        "2026-08-01T00:00:00Z",
        freq="h",
        inclusive="left",
    )
    x1 = rng.normal(size=len(timestamps)).astype(np.float32)
    x2 = rng.normal(size=len(timestamps)).astype(np.float32)
    target = (0.8 * x1 - 0.3 * x2 + rng.normal(0.0, 0.2, len(timestamps))).astype(
        np.float32
    )
    role_mask = np.arange(len(timestamps)) % 3 != 0
    target[~role_mask] = np.nan
    return pd.DataFrame({"x1": x1, "x2": x2}), target, role_mask, timestamps


def _preset_params() -> dict[str, float | int]:
    return {
        "n_estimators": 25,
        "learning_rate": 0.10,
        "num_leaves": 7,
        "min_child_samples": 10,
    }


def test_role_hpo_is_reference_only_and_oof_predicts_nonconditional_rows() -> None:
    X, target, role_mask, timestamps = _role_training_data()
    resolution = timestamps + pd.Timedelta(hours=13)
    result = fit_auxiliary_role_model(
        X,
        target,
        role_train_mask=role_mask,
        task_kind="regression",
        selected_features=["x1", "x2"],
        timestamps=timestamps,
        label_resolved_at=resolution,
        selection_hpo_reference_end="2026-05-01T00:00:00Z",
        n_trials=1,
        hpo_rows=600,
        random_state=17,
        role_name="conditional_peak_magnitude",
    )

    cutoff = pd.Timestamp("2026-05-01T00:00:00Z")
    reference = result["reference_split_contract"]
    assert pd.Timestamp(reference["decision_bounds"]["max_utc"]) < cutoff
    assert pd.Timestamp(reference["label_resolved_bounds"]["max_utc"]) < cutoff
    assert result["hpo"]["trial_count"] == 1
    assert result["hpo"]["purged_fold_provenance"]
    for fold in result["hpo"]["purged_fold_provenance"]:
        assert fold["resolution_before_valid_start_assertion"] is True
        assert pd.Timestamp(fold["training_label_resolved_max"]) < pd.Timestamp(
            fold["valid_start"]
        )

    expected_oof = (timestamps >= cutoff) & (
        timestamps < pd.Timestamp("2026-08-01T00:00:00Z")
    )
    np.testing.assert_array_equal(result["oof_prediction_mask"], expected_oof)
    assert np.isfinite(result["oof_predictions"][expected_oof & ~role_mask]).all()
    assert np.all(result["oof_fold_ids"][expected_oof] >= 0)
    assert tuple(fold["fold_month"] for fold in result["fold_provenance"]) == (
        FIXED_MAY_JULY_OOF_MONTHS
    )
    for fold in result["fold_provenance"]:
        assert fold["predicted_validation_rows"] == fold["validation_rows"]
        assert fold["conditional_validation_rows"] < fold["validation_rows"]
        assert fold["resolution_before_valid_start_assertion"] is True
        assert pd.Timestamp(fold["training_label_resolved_max"]) < pd.Timestamp(
            fold["valid_start"]
        )
    assert "excluded from OOF metrics" in result["final_refit_contract"]["row_rule"]


def test_binary_calibration_and_quantile_alpha_are_role_specific() -> None:
    X, continuous_target, role_mask, timestamps = _role_training_data()
    resolution = timestamps + pd.Timedelta(hours=13)
    binary_target = np.where(
        np.isfinite(continuous_target), continuous_target > 0.0, np.nan
    )
    binary = fit_auxiliary_role_model(
        X,
        binary_target,
        role_train_mask=role_mask,
        task_kind="binary",
        selected_features=["x1", "x2"],
        timestamps=timestamps,
        label_resolved_at=resolution,
        selection_hpo_reference_end="2026-05-01T00:00:00Z",
        preset_params=_preset_params(),
        random_state=23,
        role_name="meaningful_mfe_hit",
    )
    assert binary["best_params"]["objective"] == "binary"
    assert binary["oof_metrics"]["calibration"]["available"] is True
    assert {"binary_logloss", "brier", "roc_auc", "ece_10bin"}.issubset(
        binary["oof_metrics"]
    )
    assert np.isfinite(binary["oof_predictions"][binary["oof_prediction_mask"]]).all()

    quantile = fit_auxiliary_role_model(
        X,
        continuous_target,
        role_train_mask=role_mask,
        task_kind="quantile",
        selected_features=["x1", "x2"],
        timestamps=timestamps,
        label_resolved_at=resolution,
        selection_hpo_reference_end="2026-05-01T00:00:00Z",
        preset_params=_preset_params(),
        random_state=23,
        role_name="conditional_adverse_q80",
    )
    assert quantile["quantile_alpha"] == 0.8
    assert quantile["best_params"]["objective"] == "quantile"
    assert quantile["best_params"]["alpha"] == 0.8
    assert np.isfinite(quantile["oof_metrics"]["pinball_loss_alpha_0_8"])
    assert np.isfinite(quantile["oof_metrics"]["empirical_coverage_alpha_0_8"])


def test_role_hpo_rejects_runs_longer_than_the_production_cap() -> None:
    X, target, role_mask, timestamps = _role_training_data()
    with pytest.raises(ValueError, match="production cap of 40"):
        fit_auxiliary_role_model(
            X,
            target,
            role_train_mask=role_mask,
            task_kind="regression",
            selected_features=["x1", "x2"],
            timestamps=timestamps,
            label_resolved_at=timestamps + pd.Timedelta(hours=13),
            selection_hpo_reference_end="2026-05-01T00:00:00Z",
            n_trials=41,
        )


def test_binary_role_feature_selection_is_independent_per_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = 800
    sides = np.repeat(["long", "short"], rows // 2)
    calls: list[tuple[str, str]] = []

    def fake_selector(
        X: pd.DataFrame,
        y: np.ndarray,
        **kwargs: object,
    ) -> dict[str, object]:
        local_side = str(np.asarray(kwargs["label_context"]["side"])[0])  # type: ignore[index]
        calls.append((local_side, str(kwargs["mode"])))
        assert len(X) == rows // 2
        assert np.isin(y, (0.0, 1.0)).all()
        return {
            "selected_feature_names": [
                "efficiency_ratio_20" if local_side == "long" else "prog_eff_24"
            ],
            "metrics": {},
        }

    monkeypatch.setattr(
        "extreme_price_movements.lgbm_pipeline.train_lgbm_stability_candidate",
        fake_selector,
    )
    result = select_auxiliary_role_features(
        pd.DataFrame(
            {
                "efficiency_ratio_20": np.linspace(0.0, 1.0, rows),
                "prog_eff_24": np.linspace(1.0, 0.0, rows),
            }
        ),
        np.arange(rows) % 2,
        task_kind="binary",
        timestamps=pd.date_range("2026-04-01", periods=rows, freq="h", tz="UTC"),
        assets=np.repeat(["AAA", "BBB"], rows // 2),
        sides=sides,
        archetypes=np.repeat("base", rows),
        role_name="meaningful_mfe_event",
    )

    assert calls == [("long", "classifier"), ("short", "classifier")]
    assert result["selected_features_by_side"] == {
        "long": ["efficiency_ratio_20"],
        "short": ["prog_eff_24"],
    }
