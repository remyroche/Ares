from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import extreme_price_movements.path_archetype_geometry_search as geometry_search
from extreme_price_movements.base_candidate_population import candidate_identity_sha256
from extreme_price_movements.path_archetype_geometry_search import (
    DEFAULT_MAX_TRAIN_ROWS_PER_FOLD,
    GEOMETRY_EVALUATION_MODE_LEGACY,
    GEOMETRY_EVALUATION_MODE_SHORT_HISTORY,
    PATH_GEOMETRY_CLASSES,
    PathGeometryConfig,
    bounded_chronological_training_positions,
    economic_separation,
    ensure_risk_fraction,
    evaluate_geometry_config,
    fixed_four_month_ablation_fold,
    four_month_walk_forward_folds,
    label_path_geometry,
    multiclass_scores,
    reduced_joint_best_two_values,
    reduced_joint_design,
    short_history_purged_chronological_folds,
    stable_plateau_select,
    staged_geometry_search,
)


def _raw(values: list[float], hour: int) -> float:
    return values[min(hour - 1, len(values) - 1)]


def _frame() -> pd.DataFrame:
    # Each MFE sequence is cumulative R; ATR is exactly 2x R because risk is
    # 2% and ATR is 1%. The dynamic default meaningful threshold is 1.5 ATR.
    mfe = [
        [0.10],
        [0.80, 1.50],
        [0.80, 1.00],
        [0.80, 1.00],
        [0.10, 0.20, 0.30, 0.30, 0.50, 0.80, 1.20],
        [0.20, 0.50, 0.80, 1.00, 1.10],
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.90],
        [0.10, 0.20],
    ]
    mae = [[-0.1], [-0.1], [-0.1], [-0.6], [-0.1], [-0.2], [-0.3], [-0.1]]
    frame = pd.DataFrame(
        {
            "__ts__": pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC"),
            "__label_end_ts__": pd.date_range(
                "2024-01-02", periods=8, freq="D", tz="UTC"
            ),
            "__symbol__": ["A", "A", "A", "B", "B", "B", "C", "C"],
            "side": [
                "long",
                "long",
                "short",
                "short",
                "long",
                "long",
                "short",
                "short",
            ],
            "path_arch_close_return_r_12h": [
                -0.3,
                -0.2,
                1.0,
                1.0,
                1.0,
                0.8,
                -0.1,
                -0.1,
            ],
            "path_arch_time_to_stop_h": [
                1.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "path_arch_time_to_trailing_h": [
                np.nan,
                2.0,
                2.0,
                2.0,
                8.0,
                8.0,
                10.0,
                np.nan,
            ],
            "risk_distance": [2.0] * 8,
            "entry_price": [100.0] * 8,
            "path_arch_atr_fraction": [0.01] * 8,
            "path_arch_risk_fraction": [0.02] * 8,
        }
    )
    for hour in range(1, 13):
        frame[f"path_arch_raw_mfe_r_{hour}h"] = [_raw(row, hour) for row in mfe]
        frame[f"path_arch_raw_mfe_atr_{hour}h"] = [2.0 * _raw(row, hour) for row in mfe]
        frame[f"path_arch_raw_mae_r_{hour}h"] = [_raw(row, hour) for row in mae]
        frame[f"path_arch_cumulative_variation_r_{hour}h"] = [
            max(1.0, 2.0 * _raw(row, hour)) for row in mfe
        ]
    return frame


def _long_frame() -> pd.DataFrame:
    frame = pd.concat([_frame()] * 42, ignore_index=True)
    frame["__ts__"] = pd.date_range(
        "2024-01-01", periods=len(frame), freq="D", tz="UTC"
    )
    frame["__label_end_ts__"] = frame["__ts__"] + pd.Timedelta(days=1)
    frame["candidate_id"] = [f"candidate_{index}" for index in range(len(frame))]
    frame["frozen_feature"] = np.arange(len(frame), dtype=np.float32)
    return frame


def _short_history_frame() -> pd.DataFrame:
    """April-only synthetic path rows with resolved labels before the May cut-off."""
    frame = _long_frame().iloc[:30].copy().reset_index(drop=True)
    frame["__ts__"] = pd.date_range(
        "2026-04-01", periods=len(frame), freq="D", tz="UTC"
    )
    frame["__label_end_ts__"] = frame["__ts__"] + pd.Timedelta(hours=12)
    return frame


def _uniform_predictor(
    train: pd.DataFrame,
    target: pd.Series,
    test: pd.DataFrame,
    params: object,
    context: geometry_search.GeometryPredictorContext,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, object]]:
    del train, target, params
    class_count = len(PATH_GEOMETRY_CLASSES)
    return (
        np.full((len(test), class_count), 1.0 / class_count),
        PATH_GEOMETRY_CLASSES,
        {**context.audit(), "effective_tree_count": None, "refit_rows": None},
    )


def _geometry_runner() -> object:
    script_path = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_catboost_path_archetype_geometry_search.py"
    )
    spec = importlib.util.spec_from_file_location(
        "geometry_runner_contract", script_path
    )
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def _write_geometry_contract_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path, list[str], dict[str, object]]:
    selected = ["frozen_feature"]
    effective = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "iterations": 1_000,
        "od_wait": 100,
        "learning_rate": 0.03,
        "depth": 6,
        "l2_leaf_reg": 30.0,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
        "rsm": 0.8,
        "border_count": 64,
        "auto_class_weights": None,
        "bootstrap_type": "Bayesian",
        "grow_policy": "SymmetricTree",
        "random_seed": 20260722,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": 1,
    }
    features_path = tmp_path / "features.json"
    params_path = tmp_path / "params.json"
    contract_path = tmp_path / "feature_selection_hpo_contract.json"
    features_path.write_text(json.dumps(selected), encoding="utf-8")
    params_path.write_text(json.dumps(effective), encoding="utf-8")
    contract_path.write_text(
        json.dumps(
            {
                "schema": "catboost_path_archetype_feature_selection_hpo_contract_v1",
                "status": "feature_selection_hpo_complete",
                "fingerprint": "classifier-fingerprint",
                "selected_features": selected,
                "effective_model_params": effective,
                "hpo": {"best_params": {"depth": 6}},
            }
        ),
        encoding="utf-8",
    )
    return contract_path, features_path, params_path, selected, effective


def _write_side_geometry_contract_inputs(
    tmp_path: Path,
    frame: pd.DataFrame,
    *,
    side: str = "long",
) -> tuple[Path, Path, Path, Path, list[str], dict[str, object]]:
    """Write a frozen selection-only prerequisite plus its context evidence."""

    contract_path, features_path, params_path, selected, effective = (
        _write_geometry_contract_inputs(tmp_path)
    )
    side_frame = frame.loc[frame["side"].eq(side)].copy()
    side_frame.loc[:, "side"] = side
    candidate_sha = candidate_identity_sha256(
        side_frame, columns=("__ts__", "__symbol__", "side")
    )
    context_sha = "a" * 64
    ae_sha = "b" * 64 if side == "long" else "c" * 64
    selection_checkpoint = tmp_path / "feature_selection_checkpoint.json"
    selection_checkpoint.write_text("{}", encoding="utf-8")
    prerequisite_path = tmp_path / "geometry_prerequisite.json"
    prerequisite_path.write_text(
        json.dumps(
            {
                "schema": "catboost_path_archetype_geometry_prerequisite_v1",
                "status": "selection_complete_pending_geometry",
                "side": side,
                "model_side_scope": "per_side",
                "candidate_identity_sha256": candidate_sha,
                "selection_fingerprint": f"{side}-selection-fingerprint",
                "selected_features": selected,
                "geometry_search_model_params": effective,
                "geometry_search_model_params_sha256": hashlib.sha256(
                    json.dumps(effective, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest(),
                "canonical_context_sha256": context_sha,
                "side_ae_state_sha256": ae_sha,
                "feature_selection_checkpoint": str(selection_checkpoint),
                "feature_selection_checkpoint_sha256": hashlib.sha256(
                    selection_checkpoint.read_bytes()
                ).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    context_manifest = tmp_path / "canonical_context_manifest.json"
    context_manifest.write_text(
        json.dumps(
            {
                "context": {"sha256": context_sha},
                "ae_gmm": {
                    "loader_evidence_by_side": {
                        "long": {"ae_state_sha256": "b" * 64},
                        "short": {"ae_state_sha256": "c" * 64},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return (
        prerequisite_path,
        context_manifest,
        features_path,
        params_path,
        selected,
        effective,
    )


def test_canonical_context_provenance_prefers_materialized_output_hash(
    tmp_path: Path,
) -> None:
    runner = _geometry_runner()
    manifest = tmp_path / "canonical_context_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "context": {"sha256": "1" * 64},
                "output": {"sha256": "2" * 64},
                "ae_gmm": {
                    "loader_evidence_by_side": {"long": {"ae_state_sha256": "3" * 64}}
                },
            }
        ),
        encoding="utf-8",
    )

    provenance = runner._canonical_context_provenance(manifest, side="long")

    assert provenance["canonical_context_sha256"] == "2" * 64
    assert provenance["side_ae_state_sha256"] == "3" * 64


def test_dynamic_precedence_uses_raw_paths_not_fixed_summary_thresholds() -> None:
    labels = label_path_geometry(_frame())
    assert labels["path_geometry_label"].tolist() == [
        "immediate_adverse_path",
        "early_mfe_full_reversal",
        "fast_realization_winner",
        "fast_realization_winner",
        "late_breakout",
        "slow_grinder",
        "noisy_timeout_usable_mfe",
        "dead_timeout",
    ]
    assert labels.loc[2, "first_dynamic_meaningful_h"] == 1.0
    assert labels.loc[2, "pre_dynamic_meaningful_mae_ratio"] < 0.25
    assert labels.loc[3, "pre_dynamic_meaningful_mae_ratio"] > 0.5
    assert labels.loc[4, "boundary_late_expansion_r"] > 0.0
    assert {
        "margin_fast_net_margin",
        "number_of_matching_archetypes",
        "precedence_override_flag",
        "minimum_archetype_boundary_distance",
    }.issubset(labels.columns)
    assert labels.loc[2, "minimum_archetype_boundary_distance"] >= 0.0


def test_fast_realization_margin_can_relax_without_changing_broad_margin() -> None:
    config = PathGeometryConfig(net_margin_atr=0.75, fast_net_margin_atr=0.25)
    assert config.effective_fast_net_margin_atr == 0.25
    assert PathGeometryConfig(net_margin_atr=0.75).effective_fast_net_margin_atr == 0.75


def test_favorable_exemption_must_be_achieved_before_the_early_stop() -> None:
    frame = _frame().iloc[[0]].copy()
    frame.loc[:, "path_arch_raw_mfe_r_1h"] = 0.80
    frame.loc[:, "path_arch_raw_mfe_atr_1h"] = 1.60
    labels = label_path_geometry(frame)
    assert labels.loc[0, "path_geometry_label"] != "immediate_adverse_path"


def test_late_breakout_requires_incremental_expansion() -> None:
    frame = _frame().iloc[[4]].copy()
    frame.loc[:, "path_arch_raw_mfe_r_12h"] = 0.95
    frame.loc[:, "path_arch_raw_mfe_atr_12h"] = 1.90
    labels = label_path_geometry(frame)
    assert labels.loc[4, "path_geometry_label"] != "late_breakout"


def test_reversal_modes_are_alternatives_and_retention_is_net_of_cost() -> None:
    frame = _frame().iloc[[1]].copy()
    frame.loc[:, "path_arch_close_return_r_12h"] = 0.60
    cap = label_path_geometry(frame, PathGeometryConfig(reversal_mode="retention_cap"))
    final = label_path_geometry(
        frame, PathGeometryConfig(reversal_mode="final_net_nonpositive")
    )
    assert cap.loc[1, "path_geometry_label"] == "early_mfe_full_reversal"
    assert final.loc[1, "path_geometry_label"] != "early_mfe_full_reversal"
    assert cap.loc[1, "net_retention_after_1pct"] < 0.2


def test_risk_fraction_derives_from_entry_and_risk_distance() -> None:
    prepared = ensure_risk_fraction(_frame().drop(columns=["path_arch_risk_fraction"]))
    assert prepared["path_arch_risk_fraction"].eq(0.02).all()


def test_economic_primary_grouping_is_true_classes_with_optional_predicted_view() -> (
    None
):
    frame = pd.concat([_frame().iloc[:7], _frame().iloc[:7]], ignore_index=True)
    classes = list(PATH_GEOMETRY_CLASSES) * 2
    economics = economic_separation(frame, classes, list(reversed(classes)))
    assert (
        economics["economic_separation_score"]
        == economics["true_economic_separation_score"]
    )
    assert "predicted_net_ev_after_1pct_return_pairwise_effect_size" in economics


def test_multiclass_and_stable_selection_balance_metrics() -> None:
    classes = list(PATH_GEOMETRY_CLASSES) * 2
    class_count = len(PATH_GEOMETRY_CLASSES)
    probabilities = np.vstack([np.eye(class_count) * 0.9 + 0.1 / class_count] * 2)
    scores = multiclass_scores(classes, probabilities, PATH_GEOMETRY_CLASSES)
    assert scores["macro_f1"] == 1.0
    lower = {
        "config": {"x": 1},
        "summary": {"selection_score": 0.80, "fold_stability": 0.8, "oos_logloss": 0.2},
    }
    balanced = {
        "config": {"x": 2},
        "summary": {"selection_score": 0.81, "fold_stability": 0.7, "oos_logloss": 0.3},
    }
    assert stable_plateau_select([lower, balanced])["config"]["x"] == 1


def test_confidence_metrics_match_entropy_and_top_two_margin() -> None:
    probabilities = np.array([[0.70, 0.30, 0.0, 0.0, 0.0, 0.0, 0.0]])
    metrics = geometry_search.confidence_metrics(probabilities, "raw")
    entropy = -(0.70 * np.log(0.70) + 0.30 * np.log(0.30))
    assert metrics["raw_mean_max_probability"] == pytest.approx(0.70)
    assert metrics["raw_entropy"] == pytest.approx(entropy)
    assert metrics["raw_normalized_entropy"] == pytest.approx(entropy / np.log(7.0))
    assert metrics["raw_top1_top2_probability_margin"] == pytest.approx(0.40)


def test_economic_confusion_uses_exact_train_only_class_ev_penalties() -> None:
    train = _frame().iloc[: len(PATH_GEOMETRY_CLASSES)]
    truth = list(PATH_GEOMETRY_CLASSES)
    predicted = list(reversed(PATH_GEOMETRY_CLASSES))
    diagnostics = geometry_search.economic_confusion_diagnostics(
        train, truth, truth, predicted
    )
    expected_priors = (
        train["path_arch_close_return_r_12h"].to_numpy(dtype=float) * 0.02 - 0.01
    )
    expected_cost = float(np.mean(np.abs(expected_priors - expected_priors[::-1])))
    assert diagnostics["metrics"]["economic_confusion_cost"] == pytest.approx(
        expected_cost
    )
    priors = diagnostics["class_ev_priors"]
    assert priors["reference_geometry_net_ev_prior"].to_numpy() == pytest.approx(
        expected_priors
    )
    count = (
        diagnostics["matrix"]
        .loc[diagnostics["matrix"]["matrix_type"].eq("count")]
        .reset_index(drop=True)
    )
    penalty = (
        diagnostics["matrix"]
        .loc[diagnostics["matrix"]["matrix_type"].eq("penalty")]
        .reset_index(drop=True)
    )
    weighted = (
        diagnostics["matrix"]
        .loc[diagnostics["matrix"]["matrix_type"].eq("weighted_cost_contribution")]
        .reset_index(drop=True)
    )
    assert count.loc[0, f"predicted_{PATH_GEOMETRY_CLASSES[-1]}"] == 1.0
    assert weighted.loc[0, f"predicted_{PATH_GEOMETRY_CLASSES[-1]}"] == pytest.approx(
        penalty.loc[0, f"predicted_{PATH_GEOMETRY_CLASSES[-1]}"]
    )


def test_reduced_joint_design_uses_exact_best_two_values_per_parameter() -> None:
    def result(parameter: str, value: object, score: float) -> dict[str, object]:
        config = PathGeometryConfig().__dict__.copy()
        config[parameter] = value
        return {
            "config": config,
            "summary": {
                "selection_score": score,
                "fold_stability": 0.8,
                "oos_logloss": 0.3,
            },
        }

    values = reduced_joint_best_two_values(
        {
            "atr_floor": [
                result("atr_floor", 1.25, 0.4),
                result("atr_floor", 1.5, 0.7),
                result("atr_floor", 1.75, 0.6),
            ],
            "net_margin_atr": [
                result("net_margin_atr", 0.25, 0.6),
                result("net_margin_atr", 0.5, 0.8),
                result("net_margin_atr", 0.75, 0.5),
            ],
        }
    )
    assert values == {"atr_floor": (1.5, 1.75), "net_margin_atr": (0.5, 0.25)}
    design = reduced_joint_design(PathGeometryConfig(), values, max_joint_trials=3)
    assert len(design) == 3
    assert all(item.atr_floor in values["atr_floor"] for item in design)
    assert all(item.net_margin_atr in values["net_margin_atr"] for item in design)


def test_fixed_oos_reports_calendar_month_stability_and_predicted_economics() -> None:
    frame = _long_frame()
    fold = fixed_four_month_ablation_fold(
        frame["__ts__"], "2024-01-01", label_end=frame["__label_end_ts__"]
    )
    report = evaluate_geometry_config(
        frame,
        ["frozen_feature"],
        {},
        PathGeometryConfig(),
        folds=(fold,),
        predictor=_uniform_predictor,
    )
    assert report["summary"]["evaluated_oos_calendar_months"] == 4
    assert report["summary"]["temporal_month_stability"] > 0.0
    assert "predicted_economic_separation_score" in report["folds"].columns
    assert not report["temporal_month_stability"].empty


def test_bounded_training_sampling_is_deterministic_time_spread_and_keeps_full_oos() -> (
    None
):
    frame = _long_frame()
    fold = fixed_four_month_ablation_fold(
        frame["__ts__"], "2024-01-01", label_end=frame["__label_end_ts__"]
    )
    labels = label_path_geometry(frame)["path_geometry_label"]
    train_raw = labels.iloc[fold.train_indices].dropna().astype(str)
    train_positions = fold.train_indices[
        labels.iloc[fold.train_indices].notna().to_numpy()
    ]
    first = bounded_chronological_training_positions(
        frame, train_positions, train_raw, max_rows=16
    )
    second = bounded_chronological_training_positions(
        frame, train_positions, train_raw, max_rows=16
    )
    unbounded = bounded_chronological_training_positions(
        frame, train_positions, train_raw, max_rows=0
    )
    assert np.array_equal(first, second)
    assert len(first) == 16
    assert np.array_equal(unbounded, train_positions)
    assert first.max() > train_positions[len(train_positions) // 2]
    support = frame.iloc[first].assign(
        path_geometry_label=labels.iloc[first].to_numpy()
    )
    assert set(support["path_geometry_label"]) == set(PATH_GEOMETRY_CLASSES)
    assert support.groupby(["side", "path_geometry_label"]).size().shape[0] == 7

    seen: list[tuple[int, int]] = []

    def predictor(
        train: pd.DataFrame,
        target: pd.Series,
        test: pd.DataFrame,
        params: object,
        context: geometry_search.GeometryPredictorContext,
    ) -> tuple[np.ndarray, tuple[str, ...], dict[str, object]]:
        del target, params
        seen.append((len(train), len(test)))
        class_count = len(PATH_GEOMETRY_CLASSES)
        return (
            np.full((len(test), class_count), 1.0 / class_count),
            PATH_GEOMETRY_CLASSES,
            context.audit(),
        )

    report = evaluate_geometry_config(
        frame,
        ["frozen_feature"],
        {},
        PathGeometryConfig(),
        folds=(fold,),
        predictor=predictor,
        max_train_rows_per_fold=16,
    )
    assert seen == [(16, len(fold.oos_indices))]
    fold_report = report["folds"].iloc[0]
    assert fold_report["requested_train_rows"] == 16
    assert fold_report["effective_train_rows"] == 16
    assert fold_report["full_validation_rows"] == len(fold.oos_indices)
    assert report["summary"]["full_validation_rows"] == len(fold.oos_indices)


def test_internal_early_stop_tail_is_purged_embargoed_and_keeps_class_support() -> None:
    frame = _long_frame()
    fold = fixed_four_month_ablation_fold(
        frame["__ts__"], "2024-01-01", label_end=frame["__label_end_ts__"]
    )
    labels = label_path_geometry(frame)["path_geometry_label"]
    positions = fold.train_indices[labels.iloc[fold.train_indices].notna().to_numpy()]
    target = labels.iloc[positions].astype(str)
    context = geometry_search._early_stop_context(
        frame,
        positions,
        target,
        fold_id=fold.fold_id,
        columns=geometry_search.PathGeometryColumns(),
    )
    timestamps = pd.to_datetime(frame["__ts__"], utc=True)
    label_end = pd.to_datetime(frame["__label_end_ts__"], utc=True)
    assert set(context.early_stop_fit_positions).isdisjoint(fold.oos_indices)
    assert set(context.early_stop_validation_positions).isdisjoint(fold.oos_indices)
    assert (
        label_end.iloc[context.early_stop_fit_positions] < context.validation_start
    ).all()
    assert (
        timestamps.iloc[context.early_stop_fit_positions]
        < context.validation_start - context.embargo
    ).all()
    assert set(target.iloc[context.early_stop_validation_indices]).issubset(
        set(target.iloc[context.early_stop_fit_indices])
    )
    assert context.audit()["early_stop_embargo_hours"] == 24.0


def test_internal_early_stop_fails_when_tail_class_has_no_fit_support() -> None:
    frame = _frame()
    with pytest.raises(
        ValueError, match="no valid internal chronological early-stop split"
    ):
        geometry_search._early_stop_context(
            frame,
            np.arange(len(frame)),
            pd.Series(["a", "b", "a", "b", "a", "b", "a", "tail_only"]),
            fold_id=0,
            columns=geometry_search.PathGeometryColumns(),
        )


def test_catboost_predictor_early_stops_in_train_then_refits_all_sampled_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _long_frame()
    fold = fixed_four_month_ablation_fold(
        frame["__ts__"], "2024-01-01", label_end=frame["__label_end_ts__"]
    )
    labels = label_path_geometry(frame)["path_geometry_label"]
    positions = fold.train_indices[labels.iloc[fold.train_indices].notna().to_numpy()]
    target = labels.iloc[positions].astype(str)
    context = geometry_search._early_stop_context(
        frame,
        positions,
        target,
        fold_id=fold.fold_id,
        columns=geometry_search.PathGeometryColumns(),
    )
    fit_calls: list[tuple[int, dict[str, object]]] = []
    init_calls: list[dict[str, object]] = []

    class FakeCatBoost:
        def __init__(self, **kwargs: object) -> None:
            init_calls.append(kwargs)
            self.classes_ = np.array([], dtype=int)
            self.tree_count_ = 5

        def fit(
            self, x: pd.DataFrame, y: np.ndarray, **kwargs: object
        ) -> "FakeCatBoost":
            self.classes_ = np.unique(y)
            fit_calls.append((len(x), kwargs))
            return self

        def get_best_iteration(self) -> int:
            return 4

        def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
            return np.full((len(x), len(self.classes_)), 1.0 / len(self.classes_))

    monkeypatch.setitem(
        sys.modules, "catboost", types.SimpleNamespace(CatBoostClassifier=FakeCatBoost)
    )
    train_x = frame.loc[:, ["frozen_feature"]].iloc[positions]
    test_x = frame.loc[:, ["frozen_feature"]].iloc[fold.oos_indices]
    probabilities, classes, report = geometry_search.catboost_predictor(
        train_x,
        target,
        test_x,
        {
            "iterations": 100,
            "od_wait": 17,
            "loss_function": "MultiClass",
            "verbose": False,
            "random_seed": 20260722,
            "allow_writing_files": False,
        },
        context,
    )
    assert probabilities.shape == (len(test_x), len(PATH_GEOMETRY_CLASSES))
    assert set(classes) == set(PATH_GEOMETRY_CLASSES)
    assert len(fit_calls) == 2
    assert fit_calls[0][0] == len(context.early_stop_fit_indices)
    assert "eval_set" in fit_calls[0][1]
    assert fit_calls[0][1]["early_stopping_rounds"] == 17
    assert fit_calls[0][1]["use_best_model"] is True
    assert fit_calls[1][0] == len(target)
    assert "eval_set" not in fit_calls[1][1]
    assert init_calls[1]["iterations"] == 5
    assert init_calls[0]["allow_writing_files"] is False
    assert init_calls[1]["allow_writing_files"] is False
    assert report["effective_tree_count"] == 5
    assert report["refit_rows"] == len(target)


def test_staged_and_nested_search_share_the_train_row_sampling_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.concat([_long_frame()] * 3, ignore_index=True)
    frame["__ts__"] = pd.date_range(
        "2024-01-01", periods=len(frame), freq="D", tz="UTC"
    )
    frame["__label_end_ts__"] = frame["__ts__"] + pd.Timedelta(days=1)
    monkeypatch.setattr(
        geometry_search,
        "GEOMETRY_GRID",
        {"atr_floor": (1.25, 1.5, 1.75), "net_margin_atr": (0.25, 0.5, 0.75)},
    )
    seen_train_rows: list[int] = []

    def predictor(
        train: pd.DataFrame,
        target: pd.Series,
        test: pd.DataFrame,
        params: object,
        context: geometry_search.GeometryPredictorContext,
    ) -> tuple[np.ndarray, tuple[str, ...], dict[str, object]]:
        del target, params
        seen_train_rows.append(len(train))
        class_count = len(PATH_GEOMETRY_CLASSES)
        return (
            np.full((len(test), class_count), 1.0 / class_count),
            PATH_GEOMETRY_CLASSES,
            context.audit(),
        )

    progress_events: list[str] = []
    report = staged_geometry_search(
        frame,
        ["frozen_feature"],
        {},
        predictor=predictor,
        max_joint_trials=4,
        ablation_start_date="2024-01-01",
        nested_oof=True,
        capture_predictions=True,
        max_train_rows_per_fold=24,
        progress_reporter=lambda event, details: progress_events.append(event),
    )
    assert seen_train_rows and max(seen_train_rows) <= 24
    assert (
        report["search_contract"]["train_row_sampling"]["requested_train_rows_per_fold"]
        == 24
    )
    assert report["fold_reports"]["effective_train_rows"].le(24).all()
    assert report["fold_reports"]["full_validation_rows"].eq(123).all()
    assert "config_id" in report["fold_reports"]
    assert len(report["finalist_oos_predictions"]) == 5
    assert report["nested_oof"]
    split_contract = report["search_contract"]["evaluation_split"]
    assert split_contract == {
        "name": "4_month_train_4_month_oos",
        "train_months": 4,
        "oos_months": 4,
        "walk_forward_cadence_months": 4,
        "nested_minimum_months": 12,
        "nested_outer_oos_months": 4,
        "oos_row_contract": "all_labelled_oos_rows",
        "default_max_train_rows_per_fold": 70_000,
        "evaluation_mode": GEOMETRY_EVALUATION_MODE_LEGACY,
        "short_history_development_end": None,
        "short_history_subfold_count": 2,
    }
    first_nested = report["nested_oof"][0]
    assert first_nested["inner_train_start"] == "2024-01-01T00:00:00+00:00"
    assert first_nested["inner_oos_start"] == "2024-05-01T00:00:00+00:00"
    assert first_nested["inner_oos_end"] == "2024-09-01T00:00:00+00:00"
    assert first_nested["outer_oos_start"] == "2024-09-01T00:00:00+00:00"
    assert first_nested["outer_oos_end"] == "2025-01-01T00:00:00+00:00"
    assert all(
        entry["outer_fold_reports"][0]["effective_train_rows"] <= 24
        and entry["inner_fold_reports"][0]["effective_train_rows"] <= 24
        for entry in report["nested_oof"]
    )
    assert {
        "feature_prep_start",
        "feature_prep_complete",
        "fold_definitions",
        "baseline_candidate_complete",
        "one_dimensional_candidate_complete",
        "joint_candidate_complete",
        "finalist_capture_complete",
        "nested_fold_complete",
    }.issubset(progress_events)


def test_geometry_search_checkpoint_resumes_completed_configs_and_rejects_contract_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        geometry_search,
        "GEOMETRY_GRID",
        {"atr_floor": (1.25, 1.5), "net_margin_atr": (0.25, 0.5)},
    )
    checkpoint_path = tmp_path / "geometry_search_checkpoint.json"
    original_evaluate = geometry_search.evaluate_geometry_config
    evaluated: list[str] = []

    def interrupt_after_baseline(*args: object, **kwargs: object) -> dict[str, object]:
        config = args[3]
        assert isinstance(config, PathGeometryConfig)
        evaluated.append(geometry_search.geometry_config_id(config))
        if len(evaluated) == 2:
            raise RuntimeError("intentional interruption")
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(
        geometry_search, "evaluate_geometry_config", interrupt_after_baseline
    )
    with pytest.raises(RuntimeError, match="intentional interruption"):
        staged_geometry_search(
            _long_frame(),
            ["frozen_feature"],
            {},
            predictor=_uniform_predictor,
            max_joint_trials=2,
            ablation_start_date="2024-01-01",
            checkpoint_path=checkpoint_path,
            checkpoint_input_identity={
                "input": "test-input",
                "model_contract": "frozen-v1",
            },
        )

    interrupted = json.loads(checkpoint_path.read_text())
    baseline_id = geometry_search.geometry_config_id(PathGeometryConfig())
    assert interrupted["status"] == "running"
    assert set(interrupted["completed_configs"]) == {baseline_id}
    assert interrupted["contract"]["max_train_rows_per_fold"] == 70_000
    assert interrupted["contract"]["geometry_grid"] == {
        "atr_floor": [1.25, 1.5],
        "net_margin_atr": [0.25, 0.5],
    }
    assert interrupted["contract"]["selection_folds"]

    resumed: list[str] = []

    def record_resumed(*args: object, **kwargs: object) -> dict[str, object]:
        config = args[3]
        assert isinstance(config, PathGeometryConfig)
        resumed.append(geometry_search.geometry_config_id(config))
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(geometry_search, "evaluate_geometry_config", record_resumed)
    report = staged_geometry_search(
        _long_frame(),
        ["frozen_feature"],
        {},
        predictor=_uniform_predictor,
        max_joint_trials=2,
        ablation_start_date="2024-01-01",
        checkpoint_path=checkpoint_path,
        checkpoint_input_identity={
            "input": "test-input",
            "model_contract": "frozen-v1",
        },
    )
    assert baseline_id not in resumed
    assert report["search_contract"]["checkpoint"]["completed_config_count"] >= 1
    assert json.loads(checkpoint_path.read_text())["status"] == "complete"

    with pytest.raises(ValueError, match="checkpoint fingerprint"):
        staged_geometry_search(
            _long_frame(),
            ["frozen_feature"],
            {},
            predictor=_uniform_predictor,
            max_joint_trials=2,
            ablation_start_date="2024-01-01",
            checkpoint_path=checkpoint_path,
            checkpoint_input_identity={
                "input": "different-input",
                "model_contract": "frozen-v1",
            },
        )


def test_checkpoint_can_only_migrate_nested_true_to_disabled_post_search_refits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        geometry_search,
        "GEOMETRY_GRID",
        {"atr_floor": (1.25,), "net_margin_atr": (0.25,)},
    )
    checkpoint_path = tmp_path / "geometry_search_checkpoint.json"
    common = {
        "frame": _long_frame(),
        "feature_columns": ["frozen_feature"],
        "model_params": {},
        "predictor": _uniform_predictor,
        "max_joint_trials": 1,
        "ablation_start_date": "2024-01-01",
        "checkpoint_path": checkpoint_path,
        "checkpoint_input_identity": {
            "input": "test-input",
            "model_contract": "frozen-v1",
        },
        "run_post_search_refits": False,
    }
    staged_geometry_search(nested_oof=True, **common)
    original = json.loads(checkpoint_path.read_text())
    assert original["contract"]["nested_oof"] is True
    assert not original["nested_outer_folds"]

    staged_geometry_search(nested_oof=False, **common)
    migrated = json.loads(checkpoint_path.read_text())
    assert migrated["contract"]["nested_oof"] is False
    assert not migrated["nested_outer_folds"]
    assert not migrated["finalist_captures"]


def test_finalist_checkpoint_sidecars_resume_and_reject_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        geometry_search,
        "GEOMETRY_GRID",
        {
            "atr_floor": (1.25, 1.5, 1.75),
            "net_margin_atr": (0.25, 0.5, 0.75),
        },
    )
    checkpoint_path = tmp_path / "geometry_search_checkpoint.json"
    kwargs = {
        "predictor": _uniform_predictor,
        "max_joint_trials": 4,
        "ablation_start_date": "2024-01-01",
        "capture_predictions": True,
        "checkpoint_path": checkpoint_path,
        "checkpoint_input_identity": {
            "input": "test-input",
            "model_contract": "frozen-v1",
        },
    }
    first = staged_geometry_search(_long_frame(), ["frozen_feature"], {}, **kwargs)
    assert len(first["finalist_oos_predictions"]) == 5

    state = json.loads(checkpoint_path.read_text())
    captures = state["finalist_captures"]
    assert len(captures) == 5
    sidecar_dir = checkpoint_path.with_name("geometry_search_checkpoint_sidecars")
    for metadata in captures.values():
        assert "predictions" not in metadata
        assert (
            metadata["schema"]
            == "path_archetype_geometry_checkpoint_finalist_predictions_v1"
        )
        assert Path(metadata["sidecar_path"]).parent == sidecar_dir
        assert Path(metadata["sidecar_path"]).is_file()
        assert metadata["sidecar_sha256"] == geometry_search._file_sha256(
            Path(metadata["sidecar_path"])
        )
        assert metadata["rows"] > 0
        assert len(metadata["identity_sha256"]) == 64

    def no_refits(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("complete checkpoint should resume without refitting")

    monkeypatch.setattr(geometry_search, "evaluate_geometry_config", no_refits)
    resumed = staged_geometry_search(_long_frame(), ["frozen_feature"], {}, **kwargs)
    assert len(resumed["finalist_oos_predictions"]) == 5

    damaged = Path(next(iter(captures.values()))["sidecar_path"])
    damaged.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="sidecar checksum"):
        staged_geometry_search(_long_frame(), ["frozen_feature"], {}, **kwargs)


def test_prediction_capture_is_strict_oos_and_metrics_mode_retains_no_rows() -> None:
    frame = _long_frame()
    fold = fixed_four_month_ablation_fold(
        frame["__ts__"], "2024-01-01", label_end=frame["__label_end_ts__"]
    )
    metrics_only = evaluate_geometry_config(
        frame,
        ["frozen_feature"],
        {},
        PathGeometryConfig(),
        folds=(fold,),
        predictor=_uniform_predictor,
    )
    assert "oos_predictions" not in metrics_only
    captured = evaluate_geometry_config(
        frame,
        ["frozen_feature"],
        {},
        PathGeometryConfig(),
        folds=(fold,),
        predictor=_uniform_predictor,
        capture_predictions=True,
    )["oos_predictions"]
    required = {
        "candidate_id",
        "__ts__",
        "__symbol__",
        "side",
        "true_dynamic_label",
        "predicted_class",
        "probability_vector",
        "probability_entropy",
        "fold_id",
        "train_cutoff_utc",
        "available_at",
        "validation_start",
        "latest_train_decision_ts",
        "label_resolution_available_at",
        "train_decision_cutoff",
        "config_id",
        "config_atr_floor",
    }
    assert required.issubset(captured.columns)
    assert captured["__ts__"].dt.tz is not None
    assert np.allclose(
        captured[[f"probability_{name}" for name in PATH_GEOMETRY_CLASSES]].sum(axis=1),
        1.0,
    )
    assert captured["candidate_id"].is_unique
    assert (
        captured["label_resolution_available_at"] <= captured["train_decision_cutoff"]
    ).all()
    assert (captured["train_decision_cutoff"] < captured["validation_start"]).all()


def test_target_path_columns_are_rejected_as_model_inputs() -> None:
    frame = _long_frame()
    fold = fixed_four_month_ablation_fold(
        frame["__ts__"], "2024-01-01", label_end=frame["__label_end_ts__"]
    )
    with pytest.raises(ValueError, match="non-pre-entry features"):
        evaluate_geometry_config(
            frame,
            ["path_arch_raw_mfe_r_1h"],
            {},
            PathGeometryConfig(),
            folds=(fold,),
            predictor=_uniform_predictor,
        )


def test_geometry_runner_strictly_binds_features_and_params_to_completed_classifier_contract(
    tmp_path: Path,
) -> None:
    runner = _geometry_runner()
    contract_path, features_path, params_path, selected, effective = (
        _write_geometry_contract_inputs(tmp_path)
    )

    provenance = runner._verify_classifier_selection_hpo_contract(
        contract_path.parent,
        feature_columns=runner._feature_columns(features_path),
        model_params=runner._read_json(params_path),
        features_json_path=features_path,
        catboost_params_json_path=params_path,
    )

    assert provenance["verification"] == (
        "strict_completed_classifier_selection_hpo_contract"
    )
    assert provenance["contract_fingerprint"] == "classifier-fingerprint"
    assert provenance["features_sha256"] == runner._json_sha256(selected)
    assert provenance["effective_model_params_sha256"] == runner._json_sha256(effective)
    assert len(provenance["contract_sha256"]) == 64
    assert len(provenance["features_json_sha256"]) == 64
    assert len(provenance["catboost_params_json_sha256"]) == 64


def test_geometry_runner_rejects_mismatched_features_and_raw_hpo_params(
    tmp_path: Path,
) -> None:
    runner = _geometry_runner()
    contract_path, features_path, params_path, _, _ = _write_geometry_contract_inputs(
        tmp_path
    )

    with pytest.raises(ValueError, match="selected_features"):
        runner._verify_classifier_selection_hpo_contract(
            contract_path,
            feature_columns=["another_feature"],
            model_params=runner._read_json(params_path),
            features_json_path=features_path,
            catboost_params_json_path=params_path,
        )
    with pytest.raises(ValueError, match="raw HPO params"):
        runner._verify_classifier_selection_hpo_contract(
            contract_path,
            feature_columns=runner._feature_columns(features_path),
            model_params={"depth": 6},
            features_json_path=features_path,
            catboost_params_json_path=params_path,
        )
    input_path = tmp_path / "labels.parquet"
    _long_frame().to_parquet(input_path, index=False)
    with pytest.raises(ValueError, match="requires a verified frozen"):
        runner.run(
            input_path,
            tmp_path / "output",
            ["frozen_feature"],
            {"depth": 6},
            side="long",
        )


def test_geometry_runner_rejects_mismatched_optional_compatibility_jsons(
    tmp_path: Path,
) -> None:
    runner = _geometry_runner()
    contract_path, _, _, _, _ = _write_geometry_contract_inputs(tmp_path)
    mismatched_features = tmp_path / "mismatched-features.json"
    mismatched_params = tmp_path / "mismatched-params.json"
    mismatched_features.write_text(json.dumps(["another_feature"]), encoding="utf-8")
    mismatched_params.write_text(json.dumps({"depth": 6}), encoding="utf-8")

    with pytest.raises(ValueError, match="selected_features"):
        runner._verify_classifier_selection_hpo_contract(
            contract_path,
            features_json_path=mismatched_features,
        )
    with pytest.raises(ValueError, match="raw HPO params"):
        runner._verify_classifier_selection_hpo_contract(
            contract_path,
            catboost_params_json_path=mismatched_params,
        )


def test_geometry_runner_marks_explicit_unsafe_input_provenance(tmp_path: Path) -> None:
    runner = _geometry_runner()
    _, features_path, params_path, _, _ = _write_geometry_contract_inputs(tmp_path)

    provenance = runner._unsafe_input_provenance(
        features_json_path=features_path,
        catboost_params_json_path=params_path,
    )

    assert provenance["verification"] == "unsafe_unverified_inputs"
    assert provenance["contract_sha256"] is None
    assert len(provenance["features_json_sha256"]) == 64
    assert len(provenance["catboost_params_json_sha256"]) == 64


def test_geometry_runner_persists_strict_classifier_input_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _geometry_runner()
    input_path = tmp_path / "labels.parquet"
    labels = _long_frame()
    labels.to_parquet(input_path, index=False)
    (
        contract_path,
        context_manifest,
        features_path,
        params_path,
        selected,
        effective,
    ) = _write_side_geometry_contract_inputs(tmp_path, labels, side="long")
    empty = pd.DataFrame()
    fold_usage = pd.DataFrame(
        [
            {
                "fold_id": 0,
                "requested_train_rows": 70_000,
                "effective_train_rows": 70_000,
                "full_validation_rows": 123,
            }
        ]
    )

    staged_inputs: dict[str, object] = {}

    def fake_staged_geometry_search(
        frame: pd.DataFrame,
        feature_columns: list[str],
        model_params: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        del frame
        staged_inputs["feature_columns"] = feature_columns
        staged_inputs["model_params"] = model_params
        staged_inputs["checkpoint_path"] = kwargs["checkpoint_path"]
        staged_inputs["checkpoint_input_identity"] = kwargs["checkpoint_input_identity"]
        staged_inputs["progress_reporter"] = kwargs["progress_reporter"]
        return {
            "selected": {},
            "sweep_results": empty,
            "fold_reports": fold_usage,
            "selected_fold_reports": fold_usage,
            "boundary": empty,
            "temporal_month_stability": empty,
            "side_stability": empty,
            "symbol_stability": empty,
            "side_support": empty,
            "symbol_support": empty,
            "finalist_oos_predictions": [],
        }

    monkeypatch.setattr(
        runner,
        "staged_geometry_search",
        fake_staged_geometry_search,
    )
    finalist_path = tmp_path / "finalists.json"
    monkeypatch.setattr(
        runner,
        "_write_finalist_predictions",
        lambda *args, **kwargs: (finalist_path, {"finalists": []}),
    )

    output_dir = tmp_path / "geometry"
    runner.run(
        input_path,
        output_dir,
        geometry_prerequisite=contract_path.parent,
        canonical_context_manifest=context_manifest,
        side="long",
    )

    side_output_dir = output_dir / "side=long"
    manifest = json.loads(
        (side_output_dir / "geometry_search_manifest.json").read_text()
    )
    provenance = manifest["geometry_prerequisite_provenance"]
    assert provenance["verification"] == "strict_side_selection_geometry_prerequisite"
    assert provenance["geometry_prerequisite_sha256"] == runner._file_sha256(
        contract_path
    )
    assert provenance["side"] == "long"
    assert provenance["model_side_scope"] == "per_side"
    assert provenance["selected_features_sha256"] == runner._json_sha256(selected)
    assert provenance["geometry_search_model_params_sha256"] == runner._json_sha256(
        effective
    )
    assert provenance["features_json_sha256"] is None
    assert provenance["catboost_params_json_sha256"] is None
    assert staged_inputs["feature_columns"] == selected
    assert staged_inputs["model_params"] == effective
    assert (
        staged_inputs["checkpoint_path"]
        == side_output_dir / "geometry_search_checkpoint_long.json"
    )
    assert callable(staged_inputs["progress_reporter"])
    checkpoint_identity = staged_inputs["checkpoint_input_identity"]
    assert isinstance(checkpoint_identity, dict)
    assert checkpoint_identity["input_sha256"] == runner._file_sha256(input_path)
    assert checkpoint_identity["side"] == "long"
    assert (
        checkpoint_identity["candidate_identity_sha256"]
        == provenance["candidate_identity_sha256"]
    )
    assert checkpoint_identity["canonical_context_sha256"] == "a" * 64
    assert checkpoint_identity["side_ae_state_sha256"] == "b" * 64
    assert checkpoint_identity["selection_fingerprint"] == "long-selection-fingerprint"
    assert checkpoint_identity["geometry_prerequisite_sha256"] == runner._file_sha256(
        contract_path
    )
    assert len(checkpoint_identity["prepared_frame_sha256"]) == 64
    assert checkpoint_identity["geometry_prerequisite_provenance"] == provenance
    assert manifest["selected_fold_row_usage"] == fold_usage.to_dict(orient="records")
    assert Path(manifest["fold_reports_path"]).parent == side_output_dir
    assert manifest["side"] == "long"
    assert manifest["geometry_evaluation_contract"] == {
        "name": "4_month_train_4_month_oos",
        "train_months": 4,
        "oos_months": 4,
        "walk_forward_cadence_months": 4,
        "nested_minimum_months": 12,
        "max_train_rows_per_fold": 70_000,
        "default_max_train_rows_per_fold": 70_000,
        "oos_row_contract": "all_labelled_oos_rows",
        "evaluation_mode": GEOMETRY_EVALUATION_MODE_LEGACY,
        "short_history_development_end": None,
        "short_history_subfold_count": None,
    }
    geometry_contract = json.loads(
        (side_output_dir / "geometry_contract.json").read_text()
    )
    assert geometry_contract["status"] == "geometry_complete"
    assert geometry_contract["selection_fingerprint"] == "long-selection-fingerprint"
    assert geometry_contract["selected_features"] == selected
    assert geometry_contract["geometry_search_model_params"] == effective
    assert geometry_contract["geometry_search_model_params_sha256"] == (
        runner._json_sha256(effective)
    )
    assert geometry_contract["geometry_search_training_weight_contract"] == (
        "uniform_weights_v1"
    )
    assert geometry_contract["final_classifier_class_balance_contract"] == (
        "downstream_side_local_oof_selected_v1"
    )
    assert "sample_weight_contract" not in geometry_contract
    classifier_path = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_catboost_path_archetype_classifier.py"
    )
    classifier_spec = importlib.util.spec_from_file_location(
        "classifier_geometry_contract_consumer", classifier_path
    )
    assert classifier_spec and classifier_spec.loader
    classifier_runner = importlib.util.module_from_spec(classifier_spec)
    classifier_spec.loader.exec_module(classifier_runner)
    accepted = classifier_runner._read_side_geometry_contract(
        side_output_dir / "geometry_contract.json",
        side="long",
        candidate_identity=geometry_contract["candidate_identity_sha256"],
        selected_features=selected,
        selection_fingerprint="long-selection-fingerprint",
        geometry_prerequisite_path=contract_path,
        canonical_input_contract={
            "context": {"sha256": "a" * 64},
            "ae_gmm": {"state_sha256": "b" * 64},
        },
    )
    assert accepted is not None
    assert accepted["status"] == "geometry_complete"


def test_geometry_runner_short_history_excludes_may_to_july_before_feature_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _geometry_runner()
    april = _short_history_frame()
    future = _long_frame().iloc[:70].copy().reset_index(drop=True)
    future["__ts__"] = pd.date_range(
        "2026-05-01", periods=len(future), freq="D", tz="UTC"
    )
    future["__label_end_ts__"] = future["__ts__"] + pd.Timedelta(hours=12)
    labels = pd.concat([april, future], ignore_index=True)
    labels["side"] = "long"
    input_path = tmp_path / "labels.parquet"
    labels.to_parquet(input_path, index=False)
    (
        prerequisite,
        context_manifest,
        _,
        _,
        _,
        _,
    ) = _write_side_geometry_contract_inputs(tmp_path, labels, side="long")
    empty = pd.DataFrame()
    fold_usage = pd.DataFrame(
        [{"fold_id": 0, "requested_train_rows": 10, "effective_train_rows": 8}]
    )
    observed: dict[str, object] = {}

    def fake_staged(
        frame: pd.DataFrame,
        feature_columns: list[str],
        model_params: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        del feature_columns, model_params
        observed["frame"] = frame.copy()
        observed["kwargs"] = kwargs
        return {
            "selected": {},
            "sweep_results": empty,
            "fold_reports": fold_usage,
            "selected_fold_reports": fold_usage,
            "boundary": empty,
            "temporal_month_stability": empty,
            "side_stability": empty,
            "symbol_stability": empty,
            "side_support": empty,
            "symbol_support": empty,
            "finalist_oos_predictions": [],
        }

    monkeypatch.setattr(runner, "staged_geometry_search", fake_staged)
    finalist_path = tmp_path / "finalists.json"
    monkeypatch.setattr(
        runner,
        "_write_finalist_predictions",
        lambda *args, **kwargs: (finalist_path, {"finalists": []}),
    )

    output_dir = tmp_path / "geometry"
    runner.run(
        input_path,
        output_dir,
        geometry_prerequisite=prerequisite,
        canonical_context_manifest=context_manifest,
        side="long",
        evaluation_mode=GEOMETRY_EVALUATION_MODE_SHORT_HISTORY,
        resource_min_free_ram_gib=0.0,
        resource_max_process_rss_gib=1000.0,
        resource_min_free_disk_gib=0.0,
    )

    staged_frame = observed["frame"]
    assert isinstance(staged_frame, pd.DataFrame)
    assert (
        staged_frame["__label_end_ts__"] < pd.Timestamp("2026-05-01T00:00:00Z")
    ).all()
    staged_kwargs = observed["kwargs"]
    assert isinstance(staged_kwargs, dict)
    assert staged_kwargs["evaluation_mode"] == GEOMETRY_EVALUATION_MODE_SHORT_HISTORY
    assert staged_kwargs["short_history_development_end"] == pd.Timestamp(
        "2026-05-01T00:00:00Z"
    )

    side_dir = output_dir / "side=long"
    contract = json.loads((side_dir / "geometry_contract.json").read_text())
    holdout = contract["short_history_holdout"]
    assert contract["evaluation_mode"] == GEOMETRY_EVALUATION_MODE_SHORT_HISTORY
    assert holdout["may_and_later_used_for_geometry_selection"] is False
    assert holdout["untouched_rows_after_path_validity"] == len(future)
    assert len(holdout["development_input_sha256"]) == 64
    assert len(holdout["untouched_input_sha256"]) == 64


def test_geometry_runner_short_history_rejects_unverified_inputs(
    tmp_path: Path,
) -> None:
    runner = _geometry_runner()
    with pytest.raises(
        ValueError, match="requires a verified frozen geometry prerequisite"
    ):
        runner.run(
            tmp_path / "unused.parquet",
            tmp_path / "geometry",
            ["frozen_feature"],
            {},
            side="long",
            evaluation_mode=GEOMETRY_EVALUATION_MODE_SHORT_HISTORY,
            unsafe_allow_unverified_inputs=True,
        )


def test_geometry_runner_rejects_pooled_or_cross_side_selection_contracts(
    tmp_path: Path,
) -> None:
    runner = _geometry_runner()
    labels = _long_frame()
    input_path = tmp_path / "labels.parquet"
    labels.to_parquet(input_path, index=False)
    contract_path, context_manifest, *_ = _write_side_geometry_contract_inputs(
        tmp_path, labels, side="long"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["candidate_identity_sha256"] = "d" * 64
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate_identity_sha256"):
        runner.run(
            input_path,
            tmp_path / "geometry",
            geometry_prerequisite=contract_path,
            canonical_context_manifest=context_manifest,
            side="long",
        )

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["candidate_identity_sha256"] = candidate_identity_sha256(
        labels.loc[labels["side"].eq("long")],
        columns=("__ts__", "__symbol__", "side"),
    )
    contract["model_side_scope"] = "pooled"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="pooled or cross-side"):
        runner.run(
            input_path,
            tmp_path / "geometry",
            geometry_prerequisite=contract_path,
            canonical_context_manifest=context_manifest,
            side="long",
        )

    contract["model_side_scope"] = "per_side"
    contract["side"] = "short"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="pooled or cross-side"):
        runner.run(
            input_path,
            tmp_path / "geometry",
            geometry_prerequisite=contract_path,
            canonical_context_manifest=context_manifest,
            side="long",
        )


def test_geometry_runner_requires_selection_only_geometry_prerequisite(
    tmp_path: Path,
) -> None:
    runner = _geometry_runner()
    labels = _long_frame()
    input_path = tmp_path / "labels.parquet"
    labels.to_parquet(input_path, index=False)
    legacy_contract, _, _, _, _ = _write_geometry_contract_inputs(tmp_path)
    context_manifest = tmp_path / "canonical_context_manifest.json"
    context_manifest.write_text(
        json.dumps(
            {
                "context": {"sha256": "a" * 64},
                "ae_gmm": {
                    "loader_evidence_by_side": {"long": {"ae_state_sha256": "b" * 64}}
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported schema"):
        runner.run(
            input_path,
            tmp_path / "geometry",
            geometry_prerequisite=legacy_contract,
            canonical_context_manifest=context_manifest,
            side="long",
        )

    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    prerequisite, context_manifest, *_ = _write_side_geometry_contract_inputs(
        valid_dir, labels, side="long"
    )
    payload = json.loads(prerequisite.read_text(encoding="utf-8"))
    payload.pop("geometry_search_model_params")
    prerequisite.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="geometry_search_model_params"):
        runner.run(
            input_path,
            tmp_path / "geometry",
            geometry_prerequisite=prerequisite,
            canonical_context_manifest=context_manifest,
            side="long",
        )


def test_geometry_runner_filters_to_side_before_search_and_owns_side_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _geometry_runner()
    labels = _long_frame()
    input_path = tmp_path / "labels.parquet"
    labels.to_parquet(input_path, index=False)
    empty = pd.DataFrame()
    fold_usage = pd.DataFrame(
        [
            {
                "fold_id": 0,
                "requested_train_rows": 1,
                "effective_train_rows": 1,
                "full_validation_rows": 1,
            }
        ]
    )
    observed: dict[str, dict[str, object]] = {}

    def fake_staged(
        frame: pd.DataFrame,
        feature_columns: list[str],
        model_params: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        del feature_columns, model_params
        side = str(frame["side"].iloc[0])
        assert set(frame["side"].astype(str)) == {side}
        observed[side] = dict(kwargs["checkpoint_input_identity"])
        return {
            "selected": {},
            "sweep_results": empty,
            "fold_reports": fold_usage,
            "selected_fold_reports": fold_usage,
            "boundary": empty,
            "temporal_month_stability": empty,
            "side_stability": empty,
            "symbol_stability": empty,
            "side_support": empty,
            "symbol_support": empty,
            "finalist_oos_predictions": [],
        }

    monkeypatch.setattr(runner, "staged_geometry_search", fake_staged)
    parent = tmp_path / "geometry"
    for side in ("long", "short"):
        contract_dir = tmp_path / f"contract-{side}"
        contract_dir.mkdir()
        contract_path, context_manifest, *_ = _write_side_geometry_contract_inputs(
            contract_dir, labels, side=side
        )
        runner.run(
            input_path,
            parent,
            geometry_prerequisite=contract_path,
            canonical_context_manifest=context_manifest,
            side=side,
        )
        side_dir = parent / f"side={side}"
        assert (side_dir / "geometry_search_manifest.json").is_file()
        assert not (parent / "geometry_search_manifest.json").exists()
    assert observed["long"]["side"] == "long"
    assert observed["short"]["side"] == "short"
    filtered = runner._filter_side_before_search(
        labels,
        side="short",
        side_column="side",
    )
    assert str(filtered["side"].dtype) == "string"
    assert (
        observed["long"]["candidate_identity_sha256"]
        != observed["short"]["candidate_identity_sha256"]
    )


def test_geometry_runner_wires_default_train_row_cap_from_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _geometry_runner()
    _, features_path, params_path, _, _ = _write_geometry_contract_inputs(tmp_path)
    seen: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> dict[str, object]:
        del args
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_catboost_path_archetype_geometry_search.py",
            "--input",
            str(tmp_path / "labels.parquet"),
            "--output-dir",
            str(tmp_path / "output"),
            "--side",
            "long",
            "--features-json",
            str(features_path),
            "--catboost-params-json",
            str(params_path),
            "--unsafe-allow-unverified-inputs",
            "--evaluation-mode",
            GEOMETRY_EVALUATION_MODE_SHORT_HISTORY,
            "--resource-min-free-ram-gib",
            "1.5",
        ],
    )
    runner.main()
    assert seen["max_train_rows_per_fold"] == 70_000
    assert seen["evaluation_mode"] == GEOMETRY_EVALUATION_MODE_SHORT_HISTORY
    assert seen["resource_min_free_ram_gib"] == 1.5
    assert DEFAULT_MAX_TRAIN_ROWS_PER_FOLD == 70_000


def test_geometry_runner_cli_requires_explicit_side(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _geometry_runner()
    _, features_path, params_path, _, _ = _write_geometry_contract_inputs(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_catboost_path_archetype_geometry_search.py",
            "--input",
            str(tmp_path / "labels.parquet"),
            "--output-dir",
            str(tmp_path / "output"),
            "--features-json",
            str(features_path),
            "--catboost-params-json",
            str(params_path),
            "--unsafe-allow-unverified-inputs",
        ],
    )
    with pytest.raises(SystemExit):
        runner.main()


def test_geometry_runner_cli_allows_contract_only_canonical_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _geometry_runner()
    (
        contract_path,
        context_manifest,
        _,
        _,
        _,
        _,
    ) = _write_side_geometry_contract_inputs(tmp_path, _long_frame(), side="long")
    seen: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> dict[str, object]:
        seen["args"] = args
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_catboost_path_archetype_geometry_search.py",
            "--input",
            str(tmp_path / "labels.parquet"),
            "--output-dir",
            str(tmp_path / "output"),
            "--side",
            "long",
            "--geometry-prerequisite",
            str(contract_path.parent),
            "--canonical-context-manifest",
            str(context_manifest),
        ],
    )

    runner.main()

    assert seen["args"][2:4] == (None, None)
    assert seen["geometry_prerequisite"] == contract_path.parent
    assert seen["features_json_path"] is None
    assert seen["catboost_params_json_path"] is None
    assert seen["side"] == "long"
    assert seen["canonical_context_manifest"] == context_manifest


def test_staged_search_captures_only_top_five_and_runner_persists_each(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        geometry_search,
        "GEOMETRY_GRID",
        {
            "atr_floor": (1.25, 1.5, 1.75),
            "net_margin_atr": (0.25, 0.5, 0.75),
        },
    )
    report = staged_geometry_search(
        _long_frame(),
        ["frozen_feature"],
        {},
        predictor=_uniform_predictor,
        max_joint_trials=4,
        ablation_start_date="2024-01-01",
        capture_predictions=True,
    )
    assert len(report["finalist_oos_predictions"]) == 5
    assert all("oos_predictions" not in finalist for finalist in report["finalists"])
    script_path = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_catboost_path_archetype_geometry_search.py"
    )
    spec = importlib.util.spec_from_file_location(
        "geometry_prediction_runner", script_path
    )
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    manifest_path, manifest = runner._write_finalist_predictions(
        tmp_path,
        report["finalist_oos_predictions"],
        ["frozen_feature"],
    )
    assert manifest_path.exists()
    assert manifest["finalist_count"] == 5
    assert manifest["target_or_path_feature_columns"] == []
    assert all(
        Path(item["path"]).exists() and len(item["identity_sha256"]) == 64
        for item in manifest["finalists"]
    )
    assert all(
        Path(item["prediction_role_manifest"]).exists()
        and len(item["prediction_role_manifest_sha256"]) == 64
        for item in manifest["finalists"]
    )
    assert all(
        set(item["diagnostic_paths"])
        == {
            "folds",
            "probability_reliability_bins",
            "economic_confusion",
            "economic_confusion_priors",
            "side_diagnostics",
            "month_diagnostics",
        }
        and all(Path(path).exists() for path in item["diagnostic_paths"].values())
        for item in manifest["finalists"]
    )


def test_staged_search_materializes_invariant_feature_matrix_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = geometry_search._feature_matrix
    calls = 0

    def counted(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return original(frame, columns)

    monkeypatch.setattr(geometry_search, "_feature_matrix", counted)
    monkeypatch.setattr(
        geometry_search,
        "GEOMETRY_GRID",
        {"atr_floor": (1.25, 1.5), "net_margin_atr": (0.25, 0.5)},
    )
    report = staged_geometry_search(
        _long_frame(),
        ["frozen_feature"],
        {},
        predictor=_uniform_predictor,
        max_joint_trials=2,
        ablation_start_date="2024-01-01",
    )

    assert calls == 1
    assert report["search_contract"]["feature_matrix_materialization"] == (
        "once_per_staged_search"
    )


def test_geometry_finalist_fits_receive_the_same_ram_aware_catboost_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as classifier

    monkeypatch.setattr(classifier, "_physical_ram_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(
        geometry_search,
        "GEOMETRY_GRID",
        {"atr_floor": (1.25, 1.5, 1.75), "net_margin_atr": (0.25, 0.5, 0.75)},
    )
    seen: list[dict[str, object]] = []

    def predictor(
        train: pd.DataFrame,
        target: pd.Series,
        test: pd.DataFrame,
        params: dict[str, object],
        context: geometry_search.GeometryPredictorContext,
    ) -> tuple[np.ndarray, tuple[str, ...], dict[str, object]]:
        del train, target
        seen.append(dict(params))
        class_count = len(PATH_GEOMETRY_CLASSES)
        return (
            np.full((len(test), class_count), 1.0 / class_count),
            PATH_GEOMETRY_CLASSES,
            context.audit(),
        )

    report = staged_geometry_search(
        _long_frame(),
        ["frozen_feature"],
        {"thread_count": 64},
        predictor=predictor,
        max_joint_trials=4,
        ablation_start_date="2024-01-01",
        capture_predictions=True,
    )
    assert seen and all(params["thread_count"] == 2 for params in seen)
    assert all(params["used_ram_limit"] == "12288MB" for params in seen)
    assert (
        report["search_contract"]["catboost_resource_contract"][
            "effective_thread_count"
        ]
        == 2
    )


def test_exact_checkpoint_geometry_export_refits_only_requested_geometry_and_persists_raw_oos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        geometry_search,
        "GEOMETRY_GRID",
        {"atr_floor": (1.25, 1.5), "net_margin_atr": (0.25, 0.5)},
    )
    checkpoint_path = tmp_path / "geometry_search_checkpoint.json"
    staged_geometry_search(
        _long_frame(),
        ["frozen_feature"],
        {},
        predictor=_uniform_predictor,
        max_joint_trials=1,
        ablation_start_date="2024-01-01",
        capture_predictions=False,
        run_post_search_refits=False,
        checkpoint_path=checkpoint_path,
    )
    checkpoint = json.loads(checkpoint_path.read_text())
    config_id = next(iter(checkpoint["completed_configs"]))
    calls: list[geometry_search.GeometryPredictorContext] = []

    def capture_predictor(
        train: pd.DataFrame,
        target: pd.Series,
        test: pd.DataFrame,
        params: dict[str, object],
        context: geometry_search.GeometryPredictorContext,
    ) -> tuple[np.ndarray, tuple[str, ...], dict[str, object]]:
        del train, target, params
        calls.append(context)
        return (
            np.full(
                (len(test), len(geometry_search.EXACT_GEOMETRY_EXPORT_CLASSES)),
                1.0 / len(geometry_search.EXACT_GEOMETRY_EXPORT_CLASSES),
            ),
            geometry_search.EXACT_GEOMETRY_EXPORT_CLASSES,
            context.audit(),
        )

    export = geometry_search.export_checkpoint_geometry(
        _long_frame(),
        ["frozen_feature"],
        {},
        checkpoint_path=checkpoint_path,
        config_id=config_id,
        predictor=capture_predictor,
        persist_final_model=False,
    )

    assert len(calls) == 1
    assert export["config_id"] == config_id
    assert export["hard_label_target"] == "seven_class_path_geometry"
    assert export["class_order"] == list(geometry_search.EXACT_GEOMETRY_EXPORT_CLASSES)
    assert export["class_merge"] == {
        "merged_class": "fast_realization_winner",
        "source_classes": ["fast_clean_winner", "fast_winner_early_drawdown"],
    }
    assert export["sample_weight_contract"] == "uniform_weights_v1"
    assert export["probability_output"] == "raw_catboost_predict_proba"
    assert export["model_persistence"]["status"] == "not_persisted_by_caller"
    probabilities = export["predictions"][
        [
            f"probability_{name}"
            for name in geometry_search.EXACT_GEOMETRY_EXPORT_CLASSES
        ]
    ]
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    raw = export["predictions"]
    assert np.allclose(raw["max_probability"], probabilities.max(axis=1))
    assert np.allclose(raw["normalized_entropy"], np.log(7.0) / np.log(7.0))
    assert np.allclose(raw["top2_probability_margin"], 0.0)
    assert np.allclose(raw["adverse_probability_mass"], 3.0 / 7.0)
    assert np.allclose(raw["favorable_probability_mass"], 3.0 / 7.0)
    assert np.allclose(raw["raw_max_probability"], raw["max_probability"])
    assert np.allclose(raw["raw_normalized_entropy"], raw["normalized_entropy"])
    persisted = json.loads(checkpoint_path.read_text())
    assert set(persisted["exact_geometry_exports"]) == {config_id}
    assert len(persisted["completed_configs"]) == len(checkpoint["completed_configs"])

    runner = _geometry_runner()
    manifest_path, manifest = runner._write_exact_geometry_export(
        tmp_path / "output", export, ["frozen_feature"]
    )
    assert manifest_path.exists()
    assert manifest["config_id"] == config_id
    assert manifest["raw_scoring_contract"]["normalized_entropy"] == (
        "-sum(p_i * log(p_i)) / log(7)"
    )
    assert Path(manifest["prediction_path"]).is_file()

    def refit_must_not_run(*args: object, **kwargs: object) -> object:
        raise AssertionError("existing exact checkpoint capture should be reused")

    reused = geometry_search.export_checkpoint_geometry(
        _long_frame(),
        ["frozen_feature"],
        {},
        checkpoint_path=checkpoint_path,
        config_id=config_id,
        predictor=refit_must_not_run,
        persist_final_model=False,
    )
    assert reused["reused_checkpoint_capture"] is True

    class FakeCatBoost:
        def __init__(self, **kwargs: object) -> None:
            self.class_count = int(kwargs.get("classes_count", 7))
            self.tree_count_ = 3

        def fit(
            self, x: pd.DataFrame, y: np.ndarray, **kwargs: object
        ) -> "FakeCatBoost":
            del x, y, kwargs
            return self

        def get_best_iteration(self) -> int:
            return 2

        def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
            return np.full((len(x), self.class_count), 1.0 / self.class_count)

        def save_model(self, path: str) -> None:
            Path(path).write_text(
                json.dumps({"class_count": self.class_count}), encoding="utf-8"
            )

        def load_model(self, path: str) -> "FakeCatBoost":
            self.class_count = int(json.loads(Path(path).read_text())["class_count"])
            return self

    monkeypatch.setitem(
        sys.modules,
        "catboost",
        types.SimpleNamespace(CatBoostClassifier=FakeCatBoost),
    )
    persisted_model = geometry_search.export_checkpoint_geometry(
        _long_frame(),
        ["frozen_feature"],
        {},
        checkpoint_path=checkpoint_path,
        config_id=config_id,
        predictor=refit_must_not_run,
        persist_final_model=True,
    )
    final_model = persisted_model["model_persistence"]
    assert Path(final_model["model_path"]).is_file()
    assert Path(final_model["model_manifest_path"]).is_file()
    assert final_model["load_predict_verification"]["status"] == "passed"
    assert final_model["final_refit"]["oos_metrics_inclusion"] == (
        "excluded_final_refit_not_used_in_causal_4m_4m_oos_metrics"
    )


def test_exact_checkpoint_geometry_export_rejects_bad_fingerprint_and_nonuniform_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(geometry_search, "GEOMETRY_GRID", {"atr_floor": (1.25,)})
    checkpoint_path = tmp_path / "geometry_search_checkpoint.json"
    staged_geometry_search(
        _long_frame(),
        ["frozen_feature"],
        {},
        predictor=_uniform_predictor,
        ablation_start_date="2024-01-01",
        capture_predictions=False,
        run_post_search_refits=False,
        checkpoint_path=checkpoint_path,
    )
    checkpoint = json.loads(checkpoint_path.read_text())
    config_id = next(iter(checkpoint["completed_configs"]))
    with pytest.raises(ValueError, match="uniform weights"):
        geometry_search.export_checkpoint_geometry(
            _long_frame(),
            ["frozen_feature"],
            {"auto_class_weights": "Balanced"},
            checkpoint_path=checkpoint_path,
            config_id=config_id,
            predictor=_uniform_predictor,
            persist_final_model=False,
        )

    checkpoint["fingerprint"] = "not-the-contract-fingerprint"
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint fingerprint"):
        geometry_search.export_checkpoint_geometry(
            _long_frame(),
            ["frozen_feature"],
            {},
            checkpoint_path=checkpoint_path,
            config_id=config_id,
            predictor=_uniform_predictor,
            persist_final_model=False,
        )


def test_four_month_fold_purges_open_labels() -> None:
    timestamps = pd.date_range("2024-01-01", "2025-01-15", freq="D", tz="UTC")
    ends = timestamps + pd.Timedelta(days=10)
    folds = four_month_walk_forward_folds(timestamps, label_end=ends)
    assert np.all(ends[folds[0].train_indices] < folds[0].oos_start)
    assert folds[0].train_end == pd.Timestamp("2024-05-01", tz="UTC")
    assert folds[0].oos_end == pd.Timestamp("2024-09-01", tz="UTC")
    assert folds[1].train_end == pd.Timestamp("2024-09-01", tz="UTC")
    assert folds[1].oos_start == pd.Timestamp("2024-09-01", tz="UTC")
    assert folds[1].oos_end == pd.Timestamp("2025-01-01", tz="UTC")
    fixed = fixed_four_month_ablation_fold(timestamps, "2024-01-01", label_end=ends)
    assert fixed.oos_start == folds[0].oos_start


def test_short_history_folds_are_purged_and_never_cross_development_cutoff() -> None:
    frame = _short_history_frame()
    cutoff = pd.Timestamp("2026-05-01T00:00:00Z")
    folds = short_history_purged_chronological_folds(
        frame["__ts__"],
        label_end=frame["__label_end_ts__"],
        development_end=cutoff,
        subfold_count=2,
    )

    assert len(folds) == 2
    for fold in folds:
        assert (
            frame.loc[fold.train_indices, "__label_end_ts__"] < fold.oos_start
        ).all()
        assert (
            frame.loc[fold.train_indices, "__ts__"]
            < fold.oos_start - pd.Timedelta(hours=24)
        ).all()
        assert (frame.loc[fold.oos_indices, "__ts__"] < cutoff).all()
        assert fold.oos_end <= cutoff


def test_short_history_staged_search_is_april_only_and_checkpoint_mode_locked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _short_history_frame()
    monkeypatch.setattr(geometry_search, "GEOMETRY_GRID", {"atr_floor": (1.5,)})
    checkpoint = tmp_path / "short_history_checkpoint.json"
    report = staged_geometry_search(
        frame,
        ["frozen_feature"],
        {},
        predictor=_uniform_predictor,
        max_joint_trials=0,
        run_post_search_refits=False,
        checkpoint_path=checkpoint,
        evaluation_mode=GEOMETRY_EVALUATION_MODE_SHORT_HISTORY,
        short_history_development_end="2026-05-01T00:00:00Z",
        short_history_subfold_count=2,
    )

    split = report["search_contract"]["evaluation_split"]
    assert split["name"] == "purged_chronological_development_only"
    assert split["evaluation_mode"] == GEOMETRY_EVALUATION_MODE_SHORT_HISTORY
    assert split["short_history_development_end"] == "2026-05-01T00:00:00Z"
    checkpoint_contract = json.loads(checkpoint.read_text())["contract"]
    assert all(
        pd.Timestamp(fold["oos_end"]) <= pd.Timestamp("2026-05-01T00:00:00Z")
        for fold in checkpoint_contract["selection_folds"]
    )

    with pytest.raises(ValueError, match="checkpoint fingerprint"):
        staged_geometry_search(
            frame,
            ["frozen_feature"],
            {},
            predictor=_uniform_predictor,
            max_joint_trials=0,
            run_post_search_refits=False,
            checkpoint_path=checkpoint,
            evaluation_mode=GEOMETRY_EVALUATION_MODE_SHORT_HISTORY,
            short_history_development_end="2026-05-02T00:00:00Z",
            short_history_subfold_count=2,
        )


def test_short_history_folds_reject_may_label_rows() -> None:
    frame = _short_history_frame()
    frame.loc[len(frame) - 1, "__label_end_ts__"] = pd.Timestamp("2026-05-01T00:00:00Z")
    with pytest.raises(ValueError, match="outside its frozen development boundary"):
        short_history_purged_chronological_folds(
            frame["__ts__"],
            label_end=frame["__label_end_ts__"],
            development_end="2026-05-01T00:00:00Z",
            subfold_count=2,
        )


def test_runner_joins_frozen_features_from_separate_parquet(tmp_path: Path) -> None:
    script_path = (
        Path(__file__).parents[1]
        / "scripts"
        / "run_catboost_path_archetype_geometry_search.py"
    )
    spec = importlib.util.spec_from_file_location("geometry_runner", script_path)
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    labels = _frame().iloc[:2]
    features = labels.loc[:, ["__ts__", "__symbol__", "side"]].copy()
    features["frozen_feature"] = [1.0, 2.0]
    path = tmp_path / "features.parquet"
    features.to_parquet(path, index=False)
    joined = runner._join_frozen_features(
        labels,
        path,
        ["frozen_feature"],
        ["__ts__", "__symbol__", "side"],
        runner.PathGeometryColumns(),
    )
    assert joined["frozen_feature"].tolist() == [1.0, 2.0]


def test_runner_sidecar_loader_normalizes_string_side(tmp_path: Path) -> None:
    runner = _geometry_runner()
    labels = _frame().iloc[:2].copy()
    numeric_side = runner._canonical_side(labels["side"])
    sidecar = labels.loc[:, ["__ts__", "__symbol__"]].copy()
    sidecar["side"] = numeric_side.map({1: "long", -1: "short"})
    sidecar["frozen_feature"] = [1.0, 2.0]
    path = tmp_path / "sidecar.parquet"
    sidecar.to_parquet(path, index=False)

    loaded = runner._load_sidecar_matrix(
        labels,
        ["frozen_feature"],
        path,
        runner.PathGeometryColumns(),
    )
    assert loaded["frozen_feature"].tolist() == [1.0, 2.0]
