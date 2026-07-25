from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.execution_ev_meta import (
    ChronologicalPurgedSplit,
    ExecutionEVTargetSpec,
    ExecutionEVTrainerConfig,
    FeatureProvenance,
    _fit_arm_mode,
    build_execution_ev_target,
    catboost_class_order_sha256,
    chronological_purged_splits,
    compare_direct_and_residual,
    execution_ev_ablation_metrics,
    execution_ev_ablation_plan,
    execution_ev_feature_columns,
    execution_ev_metrics,
    load_execution_ev_bundle,
    predict_execution_ev_bundle,
    save_execution_ev_bundle,
    timing_slope_ablation_comparison,
    train_execution_ev_meta,
    validate_execution_ev_feature_provenance,
    validate_execution_ev_training_contract,
    write_execution_ev_report,
)
from extreme_price_movements.path_archetype_labels import PATH_SHAPE_TYPES


def _frame() -> pd.DataFrame:
    times = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "__ts__": times,
            "label_end": times + pd.Timedelta(hours=12),
            "execution_net_ev_12h": np.linspace(-0.02, 0.03, len(times)),
            "existing_alpha_ev": np.linspace(-0.01, 0.01, len(times)),
            "pred_time_to_mfe_12h": np.linspace(5.0, 1.0, len(times)),
            "pred_peak_mfe_12h": np.linspace(0.01, 0.04, len(times)),
            "pred_mae_before_meaningful_mfe_atr": np.linspace(1.2, 0.2, len(times)),
            "pred_bars_before_price_stops_decreasing": np.linspace(
                8.0, 1.0, len(times)
            ),
            "pred_favorable_path_slope_atr_per_hour": np.linspace(0.1, 1.0, len(times)),
            "catboost_p_clean": np.linspace(0.2, 0.8, len(times)),
            "catboost_entropy": np.linspace(0.8, 0.2, len(times)),
            "base_prediction_uncertainty": np.linspace(0.3, 0.1, len(times)),
            "meta_leaf_support_log1p": np.linspace(2.0, 4.0, len(times)),
            "score_existing_alpha": np.linspace(0.1, 0.9, len(times)),
            "base_archetype_label__family__trend": np.tile([0.0, 1.0], len(times) // 2),
            "available_at": times,
            "direct_pred": np.linspace(-0.018, 0.028, len(times)),
            "residual_pred": np.linspace(-0.008, 0.018, len(times)),
        }
    )


def _provenance() -> dict[str, FeatureProvenance]:
    return {
        "catboost_archetype": FeatureProvenance(
            "predicted_path_archetype",
            "frozen CatBoost path classifier",
            available_at_col="available_at",
            model_input=False,
        ),
        "pred_time_to_mfe_12h": FeatureProvenance(
            "time_to_mfe", "frozen path head", available_at_col="available_at"
        ),
        "pred_peak_mfe_12h": FeatureProvenance(
            "peak_mfe", "frozen path head", available_at_col="available_at"
        ),
        "pred_mae_before_meaningful_mfe_atr": FeatureProvenance(
            "mae_before_meaningful_mfe",
            "frozen path head",
            available_at_col="available_at",
        ),
        "pred_bars_before_price_stops_decreasing": FeatureProvenance(
            "adverse_turn_timing", "frozen path head", available_at_col="available_at"
        ),
        "pred_favorable_path_slope_atr_per_hour": FeatureProvenance(
            "favorable_path_slope", "frozen path head", available_at_col="available_at"
        ),
        "catboost_p_clean": FeatureProvenance(
            "catboost_probabilities",
            "CatBoost OOF full probability",
            available_at_col="available_at",
        ),
        "catboost_entropy": FeatureProvenance(
            "catboost_entropy", "CatBoost OOF entropy", available_at_col="available_at"
        ),
        "base_prediction_uncertainty": FeatureProvenance(
            "prediction_uncertainty",
            "base OOF uncertainty",
            available_at_col="available_at",
        ),
        "meta_leaf_support_log1p": FeatureProvenance(
            "leaf_support", "frozen leaf support", available_at_col="available_at"
        ),
        "score_existing_alpha": FeatureProvenance(
            "alpha_score",
            "existing frozen alpha EV score",
            available_at_col="available_at",
        ),
        "base_archetype_label__family__trend": FeatureProvenance(
            "base_archetype_labels",
            "frozen existing base archetype label",
            available_at_col="available_at",
        ),
    }


def test_direct_and_residual_targets_are_in_net_ev_units() -> None:
    frame = _frame()
    direct = build_execution_ev_target(frame)
    residual = build_execution_ev_target(
        frame, ExecutionEVTargetSpec(mode="residual", target_col="residual_target")
    )
    np.testing.assert_allclose(direct, frame["execution_net_ev_12h"])
    np.testing.assert_allclose(
        residual, frame["execution_net_ev_12h"] - frame["existing_alpha_ev"]
    )
    assert residual.name == "residual_target"


def test_feature_contract_accepts_only_declared_pre_entry_prediction_outputs() -> None:
    frame = _frame()
    names = execution_ev_feature_columns(frame, _provenance())
    assert set(names) == set(_provenance()) - {"catboost_archetype"}
    late = frame.copy()
    late.loc[3, "available_at"] = late.loc[3, "__ts__"] + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="available after entry"):
        validate_execution_ev_feature_provenance(late, names, _provenance())
    leaked = frame.assign(actual_time_to_mfe_12h=2.0)
    bad = {
        **_provenance(),
        "actual_time_to_mfe_12h": FeatureProvenance(
            "time_to_mfe", "incorrectly declared"
        ),
    }
    with pytest.raises(ValueError, match="outcome-derived"):
        validate_execution_ev_feature_provenance(
            leaked, ["actual_time_to_mfe_12h"], bad
        )


def test_chronological_splits_purge_overlapping_12h_outcomes_and_keep_timestamp_groups() -> (
    None
):
    frame = _frame()
    splits = chronological_purged_splits(
        frame,
        n_splits=2,
        min_train_size=6,
        label_end_time_col="label_end",
        embargo_hours=2.0,
    )
    assert len(splits) == 2
    assert all(isinstance(split, ChronologicalPurgedSplit) for split in splits)
    for split in splits:
        train = frame.iloc[split.train_indices]
        valid = frame.iloc[split.validation_indices]
        assert train["label_end"].max() <= valid["__ts__"].min()
        assert train["__ts__"].max() <= valid["__ts__"].min() - pd.Timedelta(hours=2)
        assert set(split.train_indices).isdisjoint(split.validation_indices)


def test_chronological_splits_enforce_minimum_training_rows_per_side() -> None:
    timestamps = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "__ts__": timestamps,
            "label_end": timestamps,
            "side_name": ["long"] * 16 + ["short"] * 8,
        }
    )
    splits = chronological_purged_splits(
        frame,
        n_splits=1,
        min_train_size=4,
        min_train_group_col="side_name",
        required_train_groups=("long", "short"),
        label_end_time_col="label_end",
        embargo_hours=0.0,
    )
    assert len(splits) == 1
    train = frame.iloc[splits[0].train_indices]
    assert train["side_name"].value_counts().to_dict() == {"long": 16, "short": 4}


def test_metrics_report_regression_tail_ev_and_direct_vs_residual_comparison() -> None:
    frame = _frame()
    metrics = execution_ev_metrics(
        frame["execution_net_ev_12h"], frame["direct_pred"], top_k_fraction=0.2
    )
    assert metrics["rows"] == len(frame)
    assert metrics["top_k_rows"] == 10
    assert metrics["top_k_mean_net_ev"] > 0.0
    assert metrics["positive_ev_auc"] > 0.5
    assert metrics["top_k_positive_ev_rate"] > metrics["positive_ev_rate"]
    report = compare_direct_and_residual(
        frame,
        direct_prediction_col="direct_pred",
        residual_prediction_col="residual_pred",
        top_k_fraction=0.2,
    )
    assert "direct__mae" in report
    assert "residual__spearman" in report
    assert "residual_minus_direct__top_k_sum_net_ev" in report


def test_ablation_plan_and_oos_metric_table_show_family_contributions() -> None:
    provenance = _provenance()
    plan = execution_ev_ablation_plan(provenance)
    assert plan["alpha_only"] == ("score_existing_alpha",)
    assert "alpha_context" in plan
    assert "alpha_context_plus_aux" in plan
    assert "alpha_context_plus_catboost" in plan
    assert "without_catboost_entropy" in plan
    assert "catboost_entropy" not in plan["without_catboost_entropy"]
    assert (
        "pred_favorable_path_slope_atr_per_hour"
        not in plan["without_favorable_path_slope"]
    )
    assert "pred_time_to_mfe_12h" not in plan["without_time_to_mfe"]
    actual = np.array([-0.01, 0.0, 0.01, 0.02])
    report = execution_ev_ablation_metrics(
        actual,
        {
            "alpha_only": np.array([-0.01, 0.02, 0.0, 0.01]),
            "all_features": actual,
        },
        top_k_fraction=0.5,
    )
    indexed = report.set_index("arm")
    assert indexed.loc["all_features", "mae"] == 0.0
    assert "top_k_sum_net_ev" in report.columns
    assert indexed.loc["alpha_only", "input_group"] == "all_non_alpha_features"
    assert indexed.loc["alpha_only", "all_features_advantage__mae"] > 0.0
    assert indexed.loc["alpha_only", "all_features_contribution"] == "helps"


def test_ablation_plan_excludes_research_only_features() -> None:
    provenance = _provenance()
    provenance["pred_mae_before_meaningful_mfe_atr"] = FeatureProvenance(
        "mae_before_meaningful_mfe",
        "research-only adverse-depth head",
        available_at_col="available_at",
        model_input=False,
    )
    plan = execution_ev_ablation_plan(provenance)
    assert "pred_mae_before_meaningful_mfe_atr" not in plan["all_features"]

    comparison = timing_slope_ablation_comparison(
        [
            {
                "arm": "without_favorable_path_slope",
                "top_k_mean_net_ev": 0.01,
                "mae": 0.02,
            },
            {"arm": "without_time_to_mfe", "top_k_mean_net_ev": 0.02, "mae": 0.01},
        ]
    )
    assert comparison["preferred_by_top10_net_ev"] == "slope"
    assert comparison["preferred_by_mae"] == "slope"


def _trainer_frame(
    class_order: tuple[str, ...] = PATH_SHAPE_TYPES,
    *,
    declare_class_contract: bool = False,
) -> tuple[pd.DataFrame, dict[str, FeatureProvenance]]:
    rows = 192
    times = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    x = np.linspace(-1.0, 1.0, rows)
    side = np.where(np.arange(rows) % 2 == 0, "long", "short")
    alpha = 0.01 * x
    path_peak = 0.02 + 0.01 * (x + 1.0) / 2.0
    net_ev = alpha + 0.004 * x + np.where(side == "long", 0.001, -0.001)
    probabilities = np.full((rows, len(class_order)), 0.28 / (len(class_order) - 1))
    winners = np.where(x > 0.0, 1, 0)
    probabilities[np.arange(rows), winners] = 0.72
    catboost_entropy = -np.sum(probabilities * np.log(probabilities), axis=1)
    frame = pd.DataFrame(
        {
            "__ts__": times,
            "label_end": times + pd.Timedelta(hours=12),
            "side_name": side,
            "catboost_archetype": [class_order[index] for index in winners],
            "execution_net_ev_12h": net_ev,
            "existing_alpha_ev": alpha,
            "pred_time_to_mfe_12h": 6.0 - 2.0 * x,
            "pred_peak_mfe_12h": path_peak,
            "pred_mae_before_meaningful_mfe_atr": 0.8 - 0.2 * x,
            "pred_bars_before_price_stops_decreasing": 5.0 - x,
            "pred_favorable_path_slope_atr_per_hour": 0.4 + 0.2 * x,
            "catboost_entropy": catboost_entropy,
            "base_prediction_uncertainty": 0.1 + 0.1 * np.abs(x),
            "meta_leaf_support_log1p": 2.0 + x,
            "score_existing_alpha": alpha,
            "base_archetype_label__family__trend": (side == "long").astype(float),
            "available_at": times,
        }
    )
    for index in range(len(class_order)):
        frame[f"catboost_p_{index}"] = probabilities[:, index]
    contract = (
        {
            "class_order": class_order,
            "class_order_sha256": catboost_class_order_sha256(class_order),
        }
        if declare_class_contract
        else {}
    )
    provenance = {
        "catboost_archetype": FeatureProvenance(
            "predicted_path_archetype",
            "frozen CatBoost path classifier",
            available_at_col="available_at",
            model_input=False,
            **contract,
        ),
        "pred_time_to_mfe_12h": FeatureProvenance(
            "time_to_mfe", "frozen time path LGBM", available_at_col="available_at"
        ),
        "pred_peak_mfe_12h": FeatureProvenance(
            "peak_mfe", "frozen peak path LGBM", available_at_col="available_at"
        ),
        "pred_mae_before_meaningful_mfe_atr": FeatureProvenance(
            "mae_before_meaningful_mfe",
            "frozen adverse-depth LGBM",
            available_at_col="available_at",
        ),
        "pred_bars_before_price_stops_decreasing": FeatureProvenance(
            "adverse_turn_timing",
            "frozen adverse-turn LGBM",
            available_at_col="available_at",
        ),
        "pred_favorable_path_slope_atr_per_hour": FeatureProvenance(
            "favorable_path_slope",
            "frozen path-slope LGBM",
            available_at_col="available_at",
        ),
        "catboost_entropy": FeatureProvenance(
            "catboost_entropy",
            "CatBoost OOF entropy",
            available_at_col="available_at",
            **contract,
        ),
        "base_prediction_uncertainty": FeatureProvenance(
            "prediction_uncertainty",
            "alpha stack OOF uncertainty",
            available_at_col="available_at",
        ),
        "meta_leaf_support_log1p": FeatureProvenance(
            "leaf_support", "frozen alpha leaf support", available_at_col="available_at"
        ),
        "score_existing_alpha": FeatureProvenance(
            "alpha_score", "frozen alpha EV", available_at_col="available_at"
        ),
        "base_archetype_label__family__trend": FeatureProvenance(
            "base_archetype_labels",
            "frozen existing base archetype label",
            available_at_col="available_at",
        ),
    }
    provenance.update(
        {
            f"catboost_p_{index}": FeatureProvenance(
                "catboost_probabilities",
                "CatBoost OOF probability vector",
                available_at_col="available_at",
                **contract,
            )
            for index in range(len(class_order))
        }
    )
    return frame, provenance


def test_trainer_contract_requires_full_catboost_probability_vector() -> None:
    frame, provenance = _trainer_frame()
    del provenance["catboost_p_1"]
    with pytest.raises(ValueError, match="full CatBoost probability vector"):
        validate_execution_ev_training_contract(frame, provenance)


def test_trainer_contract_accepts_signed_merged_seven_class_taxonomy() -> None:
    merged = (
        "fast_clean_winner",
        "fast_winner_early_drawdown",
        "slow_grinder",
        "late_breakout",
        "early_mfe_full_reversal",
        "immediate_adverse_path",
        "timeout_or_dead_path",
    )
    frame, provenance = _trainer_frame(merged, declare_class_contract=True)

    names = validate_execution_ev_training_contract(frame, provenance)

    assert len([name for name in names if name.startswith("catboost_p_")]) == 7


def test_trainer_contract_rejects_mismatched_signed_catboost_class_hash() -> None:
    frame, provenance = _trainer_frame(declare_class_contract=True)
    provenance["catboost_p_0"] = FeatureProvenance(
        "catboost_probabilities",
        "CatBoost OOF probability vector",
        available_at_col="available_at",
        class_order=PATH_SHAPE_TYPES,
        class_order_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="class-order hash"):
        validate_execution_ev_training_contract(frame, provenance)


def test_trainer_contract_rejects_invalid_catboost_probability_semantics() -> None:
    frame, provenance = _trainer_frame()
    not_normalized = frame.copy()
    not_normalized["catboost_p_1"] += 0.05
    with pytest.raises(ValueError, match="not normalized"):
        validate_execution_ev_training_contract(not_normalized, provenance)
    bad_entropy = frame.copy()
    bad_entropy["catboost_entropy"] += 0.05
    with pytest.raises(ValueError, match="entropy does not match"):
        validate_execution_ev_training_contract(bad_entropy, provenance)


def test_trainer_contract_requires_declared_predicted_path_archetype() -> None:
    frame, provenance = _trainer_frame()
    del provenance["catboost_archetype"]
    with pytest.raises(ValueError, match="predicted pre-entry path-archetype"):
        validate_execution_ev_training_contract(frame, provenance)


def test_side_aware_direct_residual_trainer_emits_oof_diagnostics_and_bundle(
    tmp_path: pytest.TempPathFactory,
) -> None:
    pytest.importorskip("lightgbm")
    frame, provenance = _trainer_frame()
    bundle = train_execution_ev_meta(
        frame,
        provenance,
        config=ExecutionEVTrainerConfig(
            n_splits=2,
            min_train_rows=20,
            hpo_trials=0,
            n_estimators=20,
            early_stopping_rounds=5,
            run_ablations=False,
            n_jobs=1,
        ),
    )
    assert set(bundle.models["direct__all_features"]) == {"long", "short"}
    assert set(bundle.calibration["direct__all_features"]) == {"long", "short"}
    assert "__global__" not in bundle.calibration["direct__all_features"]
    assert {"direct__all_features", "residual__all_features"}.issubset(
        bundle.oof_predictions
    )
    diagnostics = bundle.report["diagnostics"]
    assert {"week", "month", "side", "archetype"}.issubset(diagnostics["scope"])
    assert {
        "top_1pct_mean_net_ev",
        "top_5pct_mean_net_ev",
        "top_10pct_mean_net_ev",
        "top_20pct_mean_net_ev",
        "top_10pct_positive_ev_rate",
        "positive_ev_auc",
        "prediction_bias",
        "mae",
        "huber",
        "rmse",
        "ic",
    }.issubset(diagnostics.columns)
    assert bundle.report["oof_contract"].startswith("outer expanding purged")
    assert {"direct", "residual"} == set(bundle.report["ablations"])
    assert "all_features_contribution" in bundle.report["ablations"]["direct"][0]
    assert {
        "execution_ev_oof_fold",
        "execution_ev_oof_validation_start_utc",
        "execution_ev_oof_train_decision_cutoff_utc",
    }.issubset(bundle.oof_provenance.columns)
    selection = bundle.report["audits"]["direct__all_features"]["feature_selection"]
    assert set(selection["final"]) == {"long", "short"}
    for side, state in bundle.models["direct__all_features"].items():
        assert state["features"] == selection["final"][side]["selected_features"]
        assert "score_existing_alpha" in state["features"]
    for fold_selection in selection["outer"].values():
        assert set(fold_selection) == {"long", "short"}
        assert all(
            item["method"]
            in {"inner_oof_permutation_mda", "train_only_abs_spearman_fallback"}
            for item in fold_selection.values()
        )

    scored = predict_execution_ev_bundle(bundle, frame)
    assert {"execution_ev_direct", "execution_ev_residual"}.issubset(scored)
    assert np.isfinite(scored.to_numpy(dtype=float)).all()
    bundle_path = save_execution_ev_bundle(bundle, tmp_path / "execution_ev.joblib")
    loaded = load_execution_ev_bundle(bundle_path)
    assert loaded.schema == bundle.schema
    paths = write_execution_ev_report(loaded, tmp_path / "report")
    assert all(path.exists() for path in paths.values())


def test_outer_validation_targets_cannot_change_fold_predictions() -> None:
    pytest.importorskip("lightgbm")
    frame, provenance = _trainer_frame()
    train = np.arange(0, 144, dtype=int)
    valid = np.arange(144, len(frame), dtype=int)
    fold = ChronologicalPurgedSplit(
        fold=0,
        train_indices=train,
        validation_indices=valid,
        validation_start=frame.loc[valid[0], "__ts__"],
        validation_end=frame.loc[valid[-1], "__ts__"],
        purge_hours=12.0,
        embargo_hours=12.0,
    )
    config = ExecutionEVTrainerConfig(
        n_splits=2,
        min_train_rows=20,
        hpo_trials=0,
        n_estimators=20,
        early_stopping_rounds=5,
        calibration_min_rows=20,
        calibration_min_local_rows=20,
        run_ablations=False,
        n_jobs=1,
    )
    features = execution_ev_ablation_plan(provenance)["all_features"]
    original, _, _ = _fit_arm_mode(
        frame,
        features,
        ExecutionEVTargetSpec(mode="direct"),
        [fold],
        config=config,
        tune=False,
    )
    perturbed = frame.copy()
    perturbed.loc[valid, "execution_net_ev_12h"] += np.linspace(1.0, 2.0, len(valid))
    changed, _, _ = _fit_arm_mode(
        perturbed,
        features,
        ExecutionEVTargetSpec(mode="direct"),
        [fold],
        config=config,
        tune=False,
    )
    np.testing.assert_allclose(original[valid], changed[valid], rtol=0.0, atol=0.0)


def test_short_outcomes_cannot_change_long_selector_model_map_or_predictions() -> None:
    pytest.importorskip("lightgbm")
    frame, provenance = _trainer_frame()
    # Enough post-warm-up long OOF rows to fit a real final same-side map,
    # while retaining the small deterministic LightGBM settings used elsewhere
    # in this focused suite.
    extended = pd.concat(
        [
            frame.assign(__ts__=frame["__ts__"] + pd.Timedelta(hours=192 * part))
            for part in range(3)
        ],
        ignore_index=True,
    )
    extended["label_end"] = extended["__ts__"] + pd.Timedelta(hours=12)
    config = ExecutionEVTrainerConfig(
        n_splits=2,
        min_train_rows=20,
        hpo_trials=1,
        n_estimators=12,
        feature_selection_n_estimators=12,
        early_stopping_rounds=4,
        calibration_min_rows=20,
        calibration_min_local_rows=20,
        run_ablations=False,
        n_jobs=1,
    )
    original = train_execution_ev_meta(extended, provenance, config=config)
    changed_frame = extended.copy()
    short = changed_frame["side_name"].eq("short")
    changed_frame.loc[short, "execution_net_ev_12h"] += np.linspace(
        0.5, 1.5, int(short.sum())
    )
    changed = train_execution_ev_meta(changed_frame, provenance, config=config)

    long = extended["side_name"].eq("long").to_numpy()
    for mode in ("direct", "residual"):
        key = f"{mode}__all_features"
        original_audit = original.report["audits"][key]
        changed_audit = changed.report["audits"][key]
        assert (
            original_audit["feature_selection"]["final"]["long"]
            == changed_audit["feature_selection"]["final"]["long"]
        )
        for fold in original_audit["feature_selection"]["outer"]:
            assert (
                original_audit["feature_selection"]["outer"][fold]["long"]
                == changed_audit["feature_selection"]["outer"][fold]["long"]
            )
        original_hpo = [
            row["hpo"]
            for row in original_audit["folds"]
            if row["side"] == "long" and row["status"] == "ok"
        ]
        changed_hpo = [
            row["hpo"]
            for row in changed_audit["folds"]
            if row["side"] == "long" and row["status"] == "ok"
        ]
        assert original_hpo == changed_hpo
        assert original.calibration[key]["long"] is not None
        assert changed.calibration[key]["long"] is not None
    original_scores = predict_execution_ev_bundle(original, extended)
    changed_scores = predict_execution_ev_bundle(changed, extended)
    np.testing.assert_allclose(
        original_scores.loc[long, ["execution_ev_direct", "execution_ev_residual"]],
        changed_scores.loc[long, ["execution_ev_direct", "execution_ev_residual"]],
        rtol=0.0,
        atol=0.0,
    )
