from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements import catboost_archetype_classifier as archetype
from extreme_price_movements.catboost_archetype_classifier import (
    CatBoostUnavailableError,
    OOFPathArchetypeResult,
    PathArchetypeClassifier,
    PathArchetypeConfig,
    build_staged_permutation_matrix_cache,
    catboost_available,
    catboost_hpo_objective_components,
    catboost_resource_contract,
    configured_base_meta_preselection_universe,
    default_catboost_hpo_space,
    fast_select_preentry_features,
    fit_purged_chronological_oof_catboost,
    multiclass_classification_diagnostics,
    path_archetype_probability_contract,
    path_summary_columns,
    purged_chronological_folds,
    staged_permutation_selection,
    suggest_catboost_hpo_params,
    summarize_future_path,
    validate_preentry_features,
)
from extreme_price_movements.features_gmm_ae import AE_GMM_FEATURE_COLUMNS
from extreme_price_movements.path_archetype_support import MERGED_PATH_ARCHETYPE_CLASSES


def test_catboost_matrix_preserves_nan_and_rejects_infinity() -> None:
    frame = pd.DataFrame({"x": [1.0, np.nan], "y": [2.0, 3.0]})
    values = archetype._catboost_matrix(frame, ["x", "y"])
    assert np.isnan(values[1, 0])
    assert values[0, 0] == 1.0

    frame.loc[0, "x"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        archetype._catboost_matrix(frame, ["x", "y"])


def test_future_path_summary_has_required_path_signatures() -> None:
    result = summarize_future_path(
        [0.2, 0.6, -0.3, 1.2, 0.8], take_profit_r=1.0, stop_r=0.25
    )
    assert set(path_summary_columns()).issubset(result)
    assert result["path_arch_mfe_4h_r"] == pytest.approx(1.2)
    assert result["path_arch_mae_4h_r"] == pytest.approx(-0.3)
    assert result["path_arch_time_to_05r_h"] == pytest.approx(2.0)
    assert result["path_arch_time_to_tp_h"] == pytest.approx(4.0)
    assert result["path_arch_time_to_stop_h"] == pytest.approx(3.0)
    assert result["path_arch_mfe_before_mae"] == 0.0
    assert result["path_arch_mae_before_mfe"] == 1.0
    assert result["path_arch_reversal_count"] == 3.0


def test_selector_excludes_warmup_and_prunes_spearman_duplicates() -> None:
    n = 80
    signal = np.arange(n, dtype=float)
    features = pd.DataFrame(
        {
            "pre_signal": signal,
            "pre_duplicate": signal * 2.0,
            "pre_sparse": [np.nan] * 10 + list(signal[10:]),
            "pre_noise": np.tile([0.0, 1.0], n // 2),
        }
    )
    target = np.where(signal % 3 == 0, "a", "b")
    result = fast_select_preentry_features(
        features,
        target,
        warmup_mask=np.r_[np.ones(10, dtype=bool), np.zeros(n - 10, dtype=bool)],
        mandatory_features=["pre_signal"],
        config=PathArchetypeConfig(max_feature_candidates=10, selector_sample_rows=80),
    )
    assert "pre_signal" in result.selected_features
    assert not ({"pre_signal", "pre_duplicate"} <= set(result.selected_features))
    assert result.availability["pre_sparse"] == pytest.approx(1.0)
    assert result.proxy_backend in {
        "binned_multiclass_proxy",
        "catboost_univariate_chronological_oos_logloss_gain",
    }


def test_univariate_catboost_skips_constant_columns_and_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    class FakeCatBoost:
        def __init__(self, **_: object) -> None:
            self.classes_ = np.array([0, 1])

        def fit(self, x: np.ndarray, y: np.ndarray) -> "FakeCatBoost":
            assert np.ptp(x[:, 0]) > 0.0
            self.classes_ = np.unique(y)
            return self

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            return np.full((len(x), len(self.classes_)), 1.0 / len(self.classes_))

    monkeypatch.setattr(module, "_require_catboost", lambda: FakeCatBoost)
    rows = 80
    sample = np.column_stack(
        [np.ones(rows), np.r_[np.zeros(60), np.arange(20, dtype=float)]]
    )
    scores = module._univariate_catboost_oos_logloss_gain(
        sample,
        np.arange(rows) % 2,
        ("constant", "late_varying"),
        random_state=7,
    )

    assert scores["constant"] == 0.0
    assert np.isfinite(scores["late_varying"])


def test_configured_universe_uses_base_meta_only_and_excludes_performance_outcomes() -> (
    None
):
    config = {
        "base_shared_feature_keys": ["base_x", "BASE_NESTED"],
        "base_long_feature_keys": [],
        "base_short_feature_keys": [],
        "meta_shared_feature_keys": [
            "meta_x",
            "base_model_score",
            "path_arch_peak_mfe_r",
        ],
        "BASE_NESTED": ["base_nested"],
        "META_BASE_PERFORMANCE_FEATURE_KEYS": ["base_model_score"],
        "MODEL_DERIVED_META_PERFORMANCE_FEATURE_KEYS": [],
    }
    selected = configured_base_meta_preselection_universe(
        [
            "base_x",
            "base_nested",
            "meta_x",
            "base_model_score",
            "path_arch_peak_mfe_r",
            "rogue",
        ],
        config_mapping=config,
    )
    assert selected == ("base_x", "base_nested", "meta_x")


def test_configured_universe_excludes_actual_model_derived_fields_but_allows_exact_frozen_aegmm() -> (
    None
):
    config = {
        "base_shared_feature_keys": ["base_x"],
        "base_long_feature_keys": [],
        "base_short_feature_keys": [],
        "meta_shared_feature_keys": ["meta_x", "base_oof_context"],
        "META_BASE_PERFORMANCE_FEATURE_KEYS": [],
        "MODEL_DERIVED_META_PERFORMANCE_FEATURE_KEYS": ["base_oof_context"],
    }
    available = ["base_x", "meta_x", "base_oof_context", *AE_GMM_FEATURE_COLUMNS]
    selected = configured_base_meta_preselection_universe(
        available,
        config_mapping=config,
        frozen_representation_features=AE_GMM_FEATURE_COLUMNS,
    )
    assert selected == ("base_x", "meta_x", *AE_GMM_FEATURE_COLUMNS)


def test_selector_requires_strictly_more_than_95_percent_availability() -> None:
    values = np.arange(100, dtype=float)
    values[:5] = np.nan
    frame = pd.DataFrame(
        {"pre_sparse": values, "pre_complete": np.arange(100, dtype=float)}
    )
    with pytest.raises(ValueError, match="strict >95%"):
        fast_select_preentry_features(
            frame,
            [0, 1] * 50,
            mandatory_features=["pre_sparse"],
            config=PathArchetypeConfig(selector_sample_rows=100),
        )


def test_multiclass_diagnostics_include_reliability_confusion_and_temporal_stability() -> (
    None
):
    diagnostics = multiclass_classification_diagnostics(
        [0, 1, 0, 1],
        np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]]),
        fold_ids=[0, 0, 1, 1],
        class_names=["failed", "clean"],
    )
    assert diagnostics["logloss"] > 0.0
    assert diagnostics["brier_macro"] == pytest.approx(diagnostics["brier_weighted"])
    assert diagnostics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert set(diagnostics["classwise"]["failed"]) >= {"ece", "f1", "recall"}
    assert (
        diagnostics["temporal_stability"]["logloss_worst"]
        >= diagnostics["temporal_stability"]["logloss_mean"]
    )
    json.dumps(diagnostics, allow_nan=False)


class _SuggestingTrial:
    def __init__(self) -> None:
        self.calls: dict[str, tuple[object, ...]] = {}

    def suggest_int(self, name: str, low: int, high: int) -> int:
        self.calls[name] = (low, high)
        return low

    def suggest_float(
        self, name: str, low: float, high: float, **kwargs: object
    ) -> float:
        self.calls[name] = (low, high, kwargs.get("log", False))
        return low

    def suggest_categorical(self, name: str, choices: list[object]) -> object:
        self.calls[name] = tuple(choices)
        return choices[0]


def test_catboost_hpo_space_matches_fixed_contract() -> None:
    space = default_catboost_hpo_space()
    assert space["depth"] == (5, 7)
    assert space["iterations"] == 3000
    assert space["od_wait"] == 150
    assert space["learning_rate"] == (0.015, 0.06)
    assert space["auto_class_weights"] is None
    assert space["bootstrap_type"] == "Bayesian"
    assert space["grow_policy"] == "SymmetricTree"
    trial = _SuggestingTrial()
    params = suggest_catboost_hpo_params(trial)
    assert params["iterations"] == 3000
    assert params["od_wait"] == 150
    assert params["border_count"] == 64
    assert trial.calls["rsm"] == (0.65, 1.0, False)
    assert trial.calls["learning_rate"] == (0.015, 0.06, True)
    assert trial.calls["l2_leaf_reg"] == (8.0, 80.0, True)
    assert trial.calls["random_strength"] == (0.1, 3.0, True)
    assert PathArchetypeConfig().permutation_stages == (150, 125, 100, 75)
    assert PathArchetypeConfig().catboost_thread_count == 4
    assert PathArchetypeConfig().relief_sample_rows == 2_500
    assert PathArchetypeConfig().selector_parallel_jobs == 4


def test_future_probability_contract_emits_fixed_raw_and_derived_fields() -> None:
    probabilities = np.array(
        [
            [0.10, 0.20, 0.30, 0.15, 0.10, 0.05, 0.10],
            [0.40, 0.05, 0.10, 0.05, 0.10, 0.20, 0.10],
        ]
    )
    result = path_archetype_probability_contract(
        probabilities, MERGED_PATH_ARCHETYPE_CLASSES
    )
    assert list(result.columns) == [
        *MERGED_PATH_ARCHETYPE_CLASSES,
        "max_probability",
        "probability_entropy",
        "normalized_entropy",
        "top2_probability_margin",
        "adverse_probability_mass",
        "favorable_probability_mass",
    ]
    assert result.loc[0, "max_probability"] == pytest.approx(0.30)
    assert result.loc[0, "normalized_entropy"] == pytest.approx(
        result.loc[0, "probability_entropy"] / np.log(7.0)
    )
    assert result.loc[0, "top2_probability_margin"] == pytest.approx(0.10)
    assert result.loc[0, "adverse_probability_mass"] == pytest.approx(0.40)
    assert result.loc[0, "favorable_probability_mass"] == pytest.approx(0.55)


def test_future_training_rejects_nonuniform_class_weights() -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    with pytest.raises(ValueError, match="uniform sample weights"):
        module._catboost_params(
            PathArchetypeConfig(), {"auto_class_weights": "Balanced"}
        )
    with pytest.raises(ValueError, match="uniform sample weights"):
        module._catboost_params(PathArchetypeConfig(), {"class_weights": [1.0, 2.0]})
    with pytest.raises(ValueError, match="uniform sample weights"):
        module.capped_catboost_params({"class_weights": [1.0, 2.0]})


def test_predeclared_balance_arms_are_bounded_ordered_and_require_oof_payload() -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    config = PathArchetypeConfig(class_order=("majority", "minority", "rare"))
    arms = module.predeclared_catboost_class_balance_arms(config)
    assert [arm["name"] for arm in arms] == [
        "uniform",
        "frequency_power_0.25",
        "frequency_power_0.50",
        "frequency_power_0.75",
    ]
    assert [arm["frequency_exponent"] for arm in arms] == [0.0, 0.25, 0.50, 0.75]
    weights, provenance = module._predeclared_class_balance_weights(
        [0] * 80 + [1] * 8 + [2] * 2,
        classes=config.class_order,
        arm_name="frequency_power_0.75",
        config=config,
        provenance_scope="final_refit_train_labels_after_oof_arm_selection",
    )
    assert provenance["class_order"] == ["majority", "minority", "rare"]
    assert provenance["class_support"] == [80, 8, 2]
    assert (
        provenance["selection_evidence"] == "purged_chronological_oof_validation_only"
    )
    assert provenance["final_refit_used_for_selection"] is False
    assert np.max(weights) / np.min(weights) <= 4.0

    with pytest.raises(ValueError, match="OOF-selected final weights"):
        module._catboost_params(config, {"class_balance_arm": "frequency_power_0.50"})

    final = module._catboost_params(
        config,
        {
            "class_balance_arm": "frequency_power_0.75",
            "class_balance_final_weights": weights.tolist(),
            "class_balance_provenance": provenance,
            "class_balance_selection_provenance": {
                "schema": module.CATBOOST_CLASS_BALANCE_SELECTION_SCHEMA,
                "arm": "frequency_power_0.75",
                "class_order": list(config.class_order),
                "selection_evidence": "purged_chronological_oof_validation_only",
                "final_refit_used_for_selection": False,
                "mandatory_initial_coverage_complete": True,
                "promotion_eligible": True,
            },
        },
    )
    assert final["class_weights"] == pytest.approx(weights.tolist())
    assert "class_balance_arm" not in final

    uniform_weights, uniform_provenance = module._predeclared_class_balance_weights(
        [0] * 80 + [1] * 8 + [2] * 2,
        classes=config.class_order,
        arm_name="uniform",
        config=config,
        provenance_scope="final_refit_train_labels_after_oof_arm_selection",
    )
    uniform_final = module._catboost_params(
        config,
        {
            "class_balance_arm": "uniform",
            "class_balance_final_weights": uniform_weights.tolist(),
            "class_balance_provenance": uniform_provenance,
            "class_balance_selection_provenance": {
                "schema": module.CATBOOST_CLASS_BALANCE_SELECTION_SCHEMA,
                "arm": "uniform",
                "class_order": list(config.class_order),
                "selection_evidence": "purged_chronological_oof_validation_only",
                "final_refit_used_for_selection": False,
                "mandatory_initial_coverage_complete": True,
                "promotion_eligible": True,
            },
        },
    )
    assert uniform_final["class_weights"] == pytest.approx([1.0, 1.0, 1.0])


def test_class_balance_oof_guard_rejects_support_predicted_share_and_entropy_collapse() -> (
    None
):
    import extreme_price_movements.catboost_archetype_classifier as module

    config = PathArchetypeConfig(class_order=("a", "b"))
    valid = module.class_balance_oof_guard(
        [0, 1, 0, 1],
        np.array([[0.8, 0.2], [0.2, 0.8], [0.6, 0.4], [0.4, 0.6]]),
        classes=config.class_order,
        config=config,
    )
    assert valid["passed"] is True
    assert valid["evaluation_scope"] == (
        "frozen_purged_oof_validation_rows_only_aggregate"
    )

    with pytest.raises(
        module.CatBoostClassBalanceError, match="insufficient validation support"
    ):
        module.class_balance_oof_guard(
            [0, 0, 0, 1],
            np.tile(np.array([0.5, 0.5]), (4, 1)),
            classes=config.class_order,
            config=config,
        )
    with pytest.raises(
        module.CatBoostClassBalanceError, match="predicted-share collapse"
    ):
        module.class_balance_oof_guard(
            [0, 1, 0, 1],
            np.tile(np.array([1.0, 0.0]), (4, 1)),
            classes=config.class_order,
            config=config,
        )
    with pytest.raises(module.CatBoostClassBalanceError, match="entropy collapse"):
        module.class_balance_oof_guard(
            [0, 1, 0, 1],
            np.array([[0.999, 0.001], [0.001, 0.999]] * 2),
            classes=config.class_order,
            config=config,
        )


def test_class_balance_oof_guard_rejects_fold_local_collapse_hidden_in_aggregate() -> (
    None
):
    import extreme_price_movements.catboost_archetype_classifier as module

    config = PathArchetypeConfig(
        class_order=("a", "b"),
        class_balance_min_class_support=1,
        class_balance_min_predicted_share=0.02,
    )
    # Aggregate predicted mass is balanced, but every individual chronological
    # fold erases one class.  Aggregate-only validation would accept this.
    with pytest.raises(
        module.CatBoostClassBalanceError, match=r"predicted-share collapse \(fold=0\)"
    ):
        module.class_balance_oof_guard(
            [0, 1, 0, 1],
            np.array(
                [
                    [0.99, 0.01],
                    [0.99, 0.01],
                    [0.01, 0.99],
                    [0.01, 0.99],
                ]
            ),
            classes=config.class_order,
            fold_ids=[0, 0, 1, 1],
            config=config,
        )


def test_frozen_future_taxonomy_uses_declared_class_order() -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    config = PathArchetypeConfig(class_order=MERGED_PATH_ARCHETYPE_CLASSES)
    target = module._categorical_target(
        ["dead_timeout", "fast_realization_winner"],
        pd.RangeIndex(2),
        config=config,
    )
    assert tuple(map(str, target.cat.categories)) == MERGED_PATH_ARCHETYPE_CLASSES
    assert target.cat.codes.tolist() == [6, 2]


def test_classifier_scoring_emits_merged_probability_contract() -> None:
    class FixedModel:
        classes_ = np.arange(7)

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            return np.tile(
                np.array([[0.10, 0.20, 0.30, 0.15, 0.10, 0.05, 0.10]]),
                (len(values), 1),
            )

    classifier = PathArchetypeClassifier(
        ("base_x",), MERGED_PATH_ARCHETYPE_CLASSES, FixedModel()
    )
    result = classifier.predict_proba(pd.DataFrame({"base_x": [1.0, 2.0]}))
    assert set(MERGED_PATH_ARCHETYPE_CLASSES).issubset(result.columns)
    assert result["probability_entropy"].notna().all()
    assert result["normalized_entropy"].between(0.0, 1.0).all()
    assert np.allclose(result["adverse_probability_mass"], 0.40)
    assert np.allclose(result["favorable_probability_mass"], 0.55)


def test_ram_aware_catboost_cap_reserves_os_memory_and_requires_unsafe_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    monkeypatch.setattr(module, "_physical_ram_bytes", lambda: 16 * 1024**3)
    safe = catboost_resource_contract(PathArchetypeConfig(catboost_thread_count=64))
    assert safe["physical_ram_bytes"] == 16 * 1024**3
    assert safe["os_reserve_bytes"] == 4 * 1024**3
    assert safe["used_ram_limit_bytes"] == 12 * 1024**3
    assert safe["effective_thread_count"] == 2
    assert safe["effective_selector_parallel_jobs"] == 2
    assert safe["used_ram_limit"] == "12288MB"

    unsafe = catboost_resource_contract(
        PathArchetypeConfig(
            catboost_thread_count=64, unsafe_allow_catboost_threads=True
        )
    )
    assert unsafe["effective_thread_count"] == 64
    assert unsafe["unsafe_allow_catboost_threads"] is True


def test_catboost_model_params_cannot_bypass_ram_aware_thread_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    monkeypatch.setattr(module, "_physical_ram_bytes", lambda: 16 * 1024**3)
    params = module._catboost_params(
        PathArchetypeConfig(catboost_thread_count=64),
        {"thread_count": 99, "used_ram_limit": "999GB"},
    )
    assert params["thread_count"] == 2
    assert params["used_ram_limit"] == "12288MB"


def test_catboost_hpo_objective_penalizes_each_error_component() -> None:
    components = catboost_hpo_objective_components(
        [0, 1, 0, 1],
        np.array([[0.8, 0.2], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]]),
        [0, 0, 1, 1],
    )
    expected = (
        components["mean_logloss"]
        + 0.25 * components["macro_brier"]
        + 0.15 * components["classwise_ece"]
        + 0.20 * components["fold_logloss_std"]
    )
    assert components["objective"] == pytest.approx(expected)
    assert all(value >= 0.0 for value in components.values())


def test_optuna_hpo_reports_minimizing_purged_oof_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    def fake_oof(*args: object, **kwargs: object) -> object:
        return module.OOFPathArchetypeResult(
            probabilities=np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]]),
            fold_ids=np.array([0, 0, 1, 1]),
            folds=[],
            models=[],
            classes=np.array(["a", "b"]),
            diagnostics={"logloss": 0.2},
        )

    monkeypatch.setattr(module, "fit_purged_chronological_oof_catboost", fake_oof)
    result = module.optimize_purged_catboost_hpo(
        pd.DataFrame({"pre_x": [0.0, 1.0, 2.0, 3.0]}),
        ["a", "b", "a", "b"],
        pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
        n_trials=4,
        config=PathArchetypeConfig(class_balance_min_class_support=1),
    )
    report = result.report()
    assert report["direction"] == "minimize"
    assert report["best_objective"] > 0.0
    assert report["best_params"]["iterations"] == 1500
    assert report["trials"][0]["state"] == "COMPLETE"
    assert report["trials"][0]["objective_components"]["objective"] == pytest.approx(
        report["best_objective"]
    )
    assert report["trials"][0]["effective_search_iterations"] == 1500
    assert report["trials"][0]["pruner"] == (
        "MedianPruner(startup_trials=3,warmup_steps=0,interval_steps=1,min_trials=2)"
    )
    assert report["trials"][0]["study_no_improvement_patience_trials"] == 30
    assert [trial["class_balance_arm"] for trial in report["trials"]] == [
        "uniform",
        "frequency_power_0.25",
        "frequency_power_0.50",
        "frequency_power_0.75",
    ]
    assert all(
        trial["class_balance_search_phase"] == "mandatory_matched_baseline"
        for trial in report["trials"]
    )
    balance = report["class_balance_search"]
    assert balance["candidate_arm_count"] == 4
    assert balance["mandatory_initial_coverage_complete"] is True
    assert balance["promotion_eligible"] is True
    assert balance["selected_arm"] in {
        "uniform",
        "frequency_power_0.25",
        "frequency_power_0.50",
        "frequency_power_0.75",
    }
    assert balance["selection_evidence"] == "purged_chronological_oof_validation_only"
    assert balance["final_refit_used_for_selection"] is False
    assert (
        report["best_params"]["class_balance_selection_provenance"][
            "final_refit_used_for_selection"
        ]
        is False
    )
    assert "class_balance_final_weights" not in report["best_params"]


def test_resumed_hpo_does_not_add_trial_after_persisted_stagnation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    def fake_oof(*args: object, **kwargs: object) -> object:
        return module.OOFPathArchetypeResult(
            probabilities=np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]]),
            fold_ids=np.array([0, 0, 1, 1]),
            folds=[],
            models=[],
            classes=np.array(["a", "b"]),
            diagnostics={"logloss": 0.2},
        )

    monkeypatch.setattr(module, "fit_purged_chronological_oof_catboost", fake_oof)
    kwargs = {
        "n_trials": 5,
        "no_improvement_trials": 2,
        "study_name": "resume_stagnation",
        "storage": f"sqlite:///{tmp_path / 'study.sqlite3'}",
        "structural_only_hpo": True,
        "config": PathArchetypeConfig(class_balance_min_class_support=1),
    }
    features = pd.DataFrame({"pre_x": [0.0, 1.0, 2.0, 3.0]})
    target = ["a", "b", "a", "b"]
    timestamps = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")

    first = module.optimize_purged_catboost_hpo(
        features,
        target,
        timestamps,
        **kwargs,
    )
    second = module.optimize_purged_catboost_hpo(
        features,
        target,
        timestamps,
        **kwargs,
    )

    assert len(first.trials) == 3
    assert len(second.trials) == 3


def test_class_balance_contract_is_versioned_and_production_requires_all_arms() -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    contract = module.catboost_class_balance_search_contract()
    assert contract["schema"] == module.CATBOOST_CLASS_BALANCE_SEARCH_SCHEMA
    assert contract["mandatory_initial_evaluation"]["scheduled_arm_order"] == [
        "uniform",
        "frequency_power_0.25",
        "frequency_power_0.50",
        "frequency_power_0.75",
    ]
    assert contract["coverage_requirement"]["production_minimum_total_trials"] == 4
    with pytest.raises(
        ValueError, match="requires at least one mandatory OOF evaluation"
    ):
        module.optimize_purged_catboost_hpo(
            pd.DataFrame({"pre_x": [0.0, 1.0, 2.0, 3.0]}),
            ["a", "b", "a", "b"],
            pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            n_trials=3,
        )


def test_structural_only_hpo_fixes_uniform_balance_and_requires_post_hpo_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    observed_arms: list[str] = []

    def fake_oof(*_args: object, **kwargs: object) -> object:
        observed_arms.append(str(kwargs["params"]["class_balance_arm"]))
        return module.OOFPathArchetypeResult(
            probabilities=np.array(
                [
                    [0.8, 0.2],
                    [0.2, 0.8],
                    [0.7, 0.3],
                    [0.1, 0.9],
                ]
            ),
            fold_ids=np.array([0, 0, 1, 1]),
            folds=[],
            models=[],
            classes=np.array(["a", "b"]),
            diagnostics={"logloss": 0.2},
        )

    monkeypatch.setattr(module, "fit_purged_chronological_oof_catboost", fake_oof)
    config = PathArchetypeConfig(class_balance_min_class_support=1)
    contract = module.catboost_structural_hpo_contract(config)
    assert contract["schema"] == module.CATBOOST_STRUCTURAL_HPO_SCHEMA
    assert contract["fixed_class_balance_arm"] == "uniform"
    assert contract["class_balance_arm_is_hpo_dimension"] is False
    assert contract["post_hpo_balance_mini_sweep_contract"]["schema"] == (
        module.CATBOOST_CLASS_BALANCE_MINI_SWEEP_SCHEMA
    )

    result = module.optimize_purged_catboost_hpo(
        pd.DataFrame({"pre_x": [0.0, 1.0, 2.0, 3.0]}),
        ["a", "b", "a", "b"],
        pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
        n_trials=2,
        structural_only_hpo=True,
        config=config,
    )

    report = result.report()
    assert observed_arms == ["uniform", "uniform"]
    assert result.best_params["class_balance_arm"] == "uniform"
    assert "class_balance_selection_provenance" not in result.best_params
    assert all("class_balance_arm" not in trial["params"] for trial in report["trials"])
    assert [trial["class_balance_arm"] for trial in report["trials"]] == [
        "uniform",
        "uniform",
    ]
    assert all(
        trial["class_balance_search_phase"] == "structural_only_uniform_hpo"
        for trial in report["trials"]
    )
    balance = report["class_balance_search"]
    assert balance["schema"] == module.CATBOOST_STRUCTURAL_HPO_SCHEMA
    assert balance["fixed_training_arm"] == "uniform"
    assert balance["arm_trial_counts"] == {
        "uniform": 2,
        "frequency_power_0.25": 0,
        "frequency_power_0.50": 0,
        "frequency_power_0.75": 0,
    }
    assert balance["balance_arm_selection_complete"] is False
    assert balance["promotion_eligible"] is False
    assert balance["selected_arm"] is None
    assert balance["post_hpo_mini_sweep_required"] is True


def test_class_balance_mandatory_schedule_resumes_without_duplicate_arms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    seen_arms: list[str] = []

    def fake_oof(*_args: object, **kwargs: object) -> object:
        seen_arms.append(str(kwargs["params"]["class_balance_arm"]))
        return module.OOFPathArchetypeResult(
            probabilities=np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]]),
            fold_ids=np.array([0, 0, 1, 1]),
            folds=[],
            models=[],
            classes=np.array(["a", "b"]),
            diagnostics={"logloss": 0.2},
        )

    monkeypatch.setattr(module, "fit_purged_chronological_oof_catboost", fake_oof)
    storage = f"sqlite:///{(tmp_path / 'hpo.sqlite3').resolve()}"
    common = {
        "features": pd.DataFrame({"pre_x": [0.0, 1.0, 2.0, 3.0]}),
        "target": ["a", "b", "a", "b"],
        "timestamps": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
        "study_name": "resume_class_balance_schedule",
        "storage": storage,
        "config": PathArchetypeConfig(class_balance_min_class_support=1),
    }
    first = module.optimize_purged_catboost_hpo(
        n_trials=2, allow_incomplete_class_balance_coverage=True, **common
    )
    assert first.class_balance_search["mandatory_initial_coverage_complete"] is False
    second = module.optimize_purged_catboost_hpo(n_trials=4, **common)
    assert second.class_balance_search["mandatory_initial_coverage_complete"] is True
    assert seen_arms[:4] == [
        "uniform",
        "frequency_power_0.25",
        "frequency_power_0.50",
        "frequency_power_0.75",
    ]
    assert [
        record["trial_number"]
        for record in second.class_balance_search["mandatory_initial_coverage"][
            "scheduled_records"
        ]
    ] == [0, 1, 2, 3]


def test_final_class_balance_weights_are_rematerialized_from_actual_final_labels() -> (
    None
):
    import extreme_price_movements.catboost_archetype_classifier as module

    config = PathArchetypeConfig(class_order=("a", "b"))
    hpo_weight, hpo_provenance = module._predeclared_class_balance_weights(
        [0, 1, 0, 1],
        classes=config.class_order,
        arm_name="frequency_power_0.75",
        config=config,
        provenance_scope="hpo_authorised_final_training_labels",
    )
    params = {
        "class_balance_arm": "frequency_power_0.75",
        "class_balance_final_weights": hpo_weight.tolist(),
        "class_balance_provenance": hpo_provenance,
        "class_balance_selection_provenance": {
            "schema": module.CATBOOST_CLASS_BALANCE_SELECTION_SCHEMA,
            "arm": "frequency_power_0.75",
            "class_order": ["a", "b"],
            "selection_evidence": "purged_chronological_oof_validation_only",
            "final_refit_used_for_selection": False,
            "mandatory_initial_coverage_complete": True,
            "promotion_eligible": True,
        },
    }
    with pytest.raises(ValueError, match="final-refit provenance"):
        module._catboost_params(config, params)
    materialized = module.rematerialize_final_class_balance_params(
        params, ["a"] * 20 + ["b"] * 2, config=config
    )
    assert materialized["class_balance_provenance"]["weight_estimation_scope"] == (
        "final_refit_train_labels_after_oof_arm_selection"
    )
    assert materialized["class_balance_provenance"]["class_support"] == [20, 2]
    assert (
        materialized["class_balance_final_weights"][1]
        > materialized["class_balance_final_weights"][0]
    )
    final = module._catboost_params(config, materialized)
    assert final["class_weights"] == pytest.approx(
        materialized["class_balance_final_weights"]
    )


def test_final_class_balance_uses_filtered_series_labels_positionally() -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    config = PathArchetypeConfig(class_order=("a", "b"))
    labels = pd.Series(["a"] * 20 + ["b"] * 2, index=np.arange(100, 122))
    params = {
        "class_balance_arm": "frequency_power_0.75",
        "class_balance_selection_provenance": {
            "schema": module.CATBOOST_CLASS_BALANCE_SELECTION_SCHEMA,
            "arm": "frequency_power_0.75",
            "class_order": ["a", "b"],
            "selection_evidence": "purged_chronological_oof_validation_only",
            "final_refit_used_for_selection": False,
            "mandatory_initial_coverage_complete": True,
            "promotion_eligible": True,
        },
    }

    materialized = module.rematerialize_final_class_balance_params(
        params,
        labels,
        config=config,
    )

    assert materialized["class_balance_provenance"]["class_support"] == [20, 2]


def test_final_balance_provenance_binds_compact_oof_selection_links() -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    config = PathArchetypeConfig(class_order=("a", "b"))
    selection = {
        "schema": module.CATBOOST_CLASS_BALANCE_SELECTION_SCHEMA,
        "arm": "frequency_power_0.75",
        "class_order": ["a", "b"],
        "selection_evidence": "purged_chronological_oof_validation_only",
        "final_refit_used_for_selection": False,
        "mandatory_initial_coverage_complete": True,
        "promotion_eligible": True,
        "selection_status": "economic_oof_promoted",
        "promotion_reason": "passed_strict_ml_and_economic_lexicographic_gates",
        "economic_oof_schema": "catboost_path_archetype_balance_economic_oof_v1",
        "economic_oof_report_sha256": "a" * 64,
        "economic_selector_config_sha256": "b" * 64,
        "mini_sweep_report_sha256": "c" * 64,
        "structural_fingerprint": "structural-hpo-v1",
        "feature_fingerprint": "features-v1",
        "geometry_fingerprint": "geometry-v1",
        # This compact diagnostic remains in the selection artifact but is
        # intentionally not copied into final model-weight provenance.
        "candidate_diagnostics": {"frequency_power_0.75": {"passed": True}},
    }
    materialized = module.rematerialize_final_class_balance_params(
        {
            "class_balance_arm": "frequency_power_0.75",
            "class_balance_selection_provenance": selection,
        },
        ["a"] * 20 + ["b"] * 2,
        config=config,
    )
    final_provenance = materialized["class_balance_provenance"]
    assert final_provenance["selected_arm_selection_provenance_sha256"] == (
        module._canonical_json_sha256(selection)
    )
    assert final_provenance["selected_arm_selection_status"] == (
        "economic_oof_promoted"
    )
    assert final_provenance["selected_arm_promotion_reason"] == (
        "passed_strict_ml_and_economic_lexicographic_gates"
    )
    assert final_provenance["selected_arm_economic_oof_report_sha256"] == "a" * 64
    assert final_provenance["selected_arm_economic_selector_config_sha256"] == (
        "b" * 64
    )
    assert final_provenance["selected_arm_mini_sweep_report_sha256"] == "c" * 64
    assert final_provenance["selected_arm_structural_fingerprint"] == (
        "structural-hpo-v1"
    )
    assert final_provenance["selected_arm_feature_fingerprint"] == "features-v1"
    assert final_provenance["selected_arm_geometry_fingerprint"] == "geometry-v1"
    assert "candidate_diagnostics" not in final_provenance
    assert module._catboost_params(config, materialized)[
        "class_weights"
    ] == pytest.approx(materialized["class_balance_final_weights"])

    mutated = dict(materialized)
    mutated["class_balance_selection_provenance"] = {
        **selection,
        "selection_status": "mutated_after_selection",
    }
    with pytest.raises(ValueError, match="selection provenance digest"):
        module._catboost_params(config, mutated)

    with pytest.raises(ValueError, match="must link to, not embed, raw OOF reports"):
        module.rematerialize_final_class_balance_params(
            {
                "class_balance_arm": "frequency_power_0.75",
                "class_balance_selection_provenance": {
                    **selection,
                    "economic_oof_report": {"per_arm": {"uniform": {}}},
                },
            },
            ["a"] * 20 + ["b"] * 2,
            config=config,
        )


def test_optuna_hpo_progress_is_atomic_and_reuses_fresh_best_oof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    calls = {"oof": 0}

    def fake_oof(*args: object, **kwargs: object) -> object:
        calls["oof"] += 1
        return module.OOFPathArchetypeResult(
            probabilities=np.array(
                [
                    [0.8, 0.2],
                    [0.2, 0.8],
                    [0.7, 0.3],
                    [0.1, 0.9],
                ]
            ),
            fold_ids=np.array([0, 0, 1, 1]),
            folds=[],
            models=[],
            classes=np.array(["a", "b"]),
            diagnostics={"logloss": 0.2},
        )

    monkeypatch.setattr(module, "fit_purged_chronological_oof_catboost", fake_oof)
    progress_path = tmp_path / "progress.json"
    common = {
        "features": pd.DataFrame({"pre_x": [0.0, 1.0, 2.0, 3.0]}),
        "target": ["a", "b", "a", "b"],
        "timestamps": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
        "progress_path": progress_path,
        "config": PathArchetypeConfig(class_balance_min_class_support=1),
    }
    first = module.optimize_purged_catboost_hpo(
        n_trials=1, allow_incomplete_class_balance_coverage=True, **common
    )
    assert first.best_oof_reused_from_current_process is True
    assert calls["oof"] == 1
    first_progress = json.loads(progress_path.read_text())
    assert first_progress["status"] == "coverage_incomplete"
    assert first_progress["target_trials"] == 1
    assert first_progress["completed_trial_count"] == 1
    assert first_progress["current_trial"]["number"] == 0
    assert first_progress["current_trial"]["state"] == "COMPLETE"
    assert first_progress["best_params"]
    assert first_progress["no_wall_clock_timeout"] is True
    assert first_progress["class_balance_search_contract"]["candidate_arm_count"] == 4
    assert (
        first_progress["class_balance_search_contract"][
            "final_refit_used_for_selection"
        ]
        is False
    )
    coverage = first_progress["class_balance_search_contract"][
        "mandatory_initial_coverage"
    ]
    assert coverage["mandatory_initial_coverage_complete"] is False
    assert coverage["scheduled_arm_order"] == [
        "uniform",
        "frequency_power_0.25",
        "frequency_power_0.50",
        "frequency_power_0.75",
    ]
    assert first.class_balance_search is not None
    assert first.class_balance_search["selected_arm"] is None
    assert first.class_balance_search["provisional_arm"] == "uniform"


def test_purged_oof_uses_validation_eval_set_and_early_stopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    fit_calls: list[dict[str, object]] = []
    init_calls: list[dict[str, object]] = []

    class FakeCatBoost:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            init_calls.append(kwargs)
            self.classes_ = np.array([0, 1])
            self.tree_count_ = 17

        def fit(self, x: np.ndarray, y: np.ndarray, **kwargs: object) -> "FakeCatBoost":
            self.classes_ = np.unique(y)
            fit_calls.append(kwargs)
            return self

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            return np.full((len(x), len(self.classes_)), 1.0 / len(self.classes_))

        def get_best_iteration(self) -> int:
            return 12

    monkeypatch.setattr(module, "_require_catboost", lambda: FakeCatBoost)
    rows = 36
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    result = fit_purged_chronological_oof_catboost(
        pd.DataFrame({"pre_x": np.arange(rows, dtype=float)}),
        [0, 1] * (rows // 2),
        timestamps,
        config=PathArchetypeConfig(oof_folds=3, embargo=pd.Timedelta(0)),
    )
    assert result.diagnostics is not None
    assert fit_calls
    assert all(
        "eval_set" in call and call["early_stopping_rounds"] == 150
        for call in fit_calls
    )
    assert init_calls and all(call["classes_count"] == 2 for call in init_calls)
    reports = result.diagnostics["fold_fit_reports"]
    assert len(reports) == len(fit_calls)
    assert all(report["use_best_model"] is True for report in reports)
    assert all(report["eval_set_used"] is True for report in reports)
    assert all(report["early_stopping_rounds"] == 150 for report in reports)
    assert all(report["best_iteration"] == 12 for report in reports)
    assert all(report["tree_count"] == 17 for report in reports)


def test_purged_oof_derives_nonuniform_weights_from_each_train_fold_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    init_calls: list[dict[str, object]] = []

    class FakeCatBoost:
        def __init__(self, **kwargs: object) -> None:
            init_calls.append(kwargs)
            self.classes_ = np.array([0, 1])

        def fit(
            self, _x: np.ndarray, y: np.ndarray, **_kwargs: object
        ) -> "FakeCatBoost":
            self.classes_ = np.unique(y)
            return self

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            return np.tile(np.array([[0.65, 0.35]]), (len(values), 1))

    monkeypatch.setattr(module, "_require_catboost", lambda: FakeCatBoost)
    rows = 72
    target = np.where(np.arange(rows) % 6 == 0, 1, 0)
    result = fit_purged_chronological_oof_catboost(
        pd.DataFrame({"pre_x": np.arange(rows, dtype=float)}),
        target,
        pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
        config=PathArchetypeConfig(oof_folds=3, embargo=pd.Timedelta(0)),
        params={"class_balance_arm": "frequency_power_0.75"},
    )
    assert result.diagnostics is not None
    assert init_calls
    for call in init_calls:
        weights = np.asarray(call["class_weights"], dtype=float)
        assert len(weights) == 2
        assert weights[1] > weights[0]
        assert np.max(weights) / np.min(weights) <= 4.0
        assert "class_balance_arm" not in call


def test_class_balance_hpo_fails_closed_when_oof_outputs_collapse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    def fake_oof(*args: object, **kwargs: object) -> object:
        return module.OOFPathArchetypeResult(
            probabilities=np.tile(np.array([[1.0, 0.0]]), (4, 1)),
            fold_ids=np.array([0, 0, 1, 1]),
            folds=[],
            models=[],
            classes=np.array(["a", "b"]),
            diagnostics={"logloss": 1.0},
        )

    monkeypatch.setattr(module, "fit_purged_chronological_oof_catboost", fake_oof)
    with pytest.raises(RuntimeError, match="without a successful trial"):
        module.optimize_purged_catboost_hpo(
            pd.DataFrame({"pre_x": [0.0, 1.0, 2.0, 3.0]}),
            ["a", "b", "a", "b"],
            pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            n_trials=1,
            allow_incomplete_class_balance_coverage=True,
        )


def test_multiclass_log_loss_vectorized_lookup_preserves_class_mapping_semantics() -> (
    None
):
    import extreme_price_movements.catboost_archetype_classifier as module

    target = np.array([4, 2, 9, 7, 2], dtype=int)
    classes = np.array([2, 4, 2], dtype=int)
    probabilities = np.array(
        [
            [0.20, 0.70, 0.10],
            [0.30, 0.20, 0.50],
            [0.15, 0.55, 0.30],
            [0.80, 0.10, 0.10],
            [0.25, 0.25, 0.50],
        ]
    )
    positions = {int(label): index for index, label in enumerate(classes)}
    expected = float(
        -np.mean(
            np.log(
                np.clip(
                    np.array(
                        [
                            probabilities[row, positions.get(int(label), 0)]
                            for row, label in enumerate(target)
                        ]
                    ),
                    1e-12,
                    1.0,
                )
            )
        )
    )

    assert module.multiclass_log_loss(target, probabilities, classes) == pytest.approx(
        expected
    )


def _staged_mda_fixture() -> tuple[
    pd.DataFrame, np.ndarray, list[object], OOFPathArchetypeResult
]:
    rows = 48
    columns = [f"pre_{index}" for index in range(6)]
    base = np.arange(rows, dtype=np.float32)
    features = pd.DataFrame(
        {
            column: (base * (index + 1) + (index % 2) * (base % 3)).astype(np.float32)
            for index, column in enumerate(columns)
        }
    )
    target = (base.astype(int) % 2).astype(int)
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    folds = purged_chronological_folds(timestamps, n_splits=2, embargo=pd.Timedelta(0))
    ordered = ("pre_5", *columns[:-1])
    ordered_features = features.loc[:, ordered]
    cache = build_staged_permutation_matrix_cache(
        ordered_features,
        timestamps,
        config=PathArchetypeConfig(oof_folds=2, embargo=pd.Timedelta(0)),
    )

    class ReusableModel:
        classes_ = np.array([0, 1])

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            score = np.asarray(values, dtype=float).sum(axis=1)
            probability = 1.0 / (1.0 + np.exp(-((score % 19.0) - 9.0)))
            return np.column_stack([1.0 - probability, probability])

    models = [ReusableModel() for _ in folds]
    probabilities = np.full((rows, 2), np.nan)
    fold_ids = np.full(rows, -1, dtype=int)
    for fold, model in zip(folds, models):
        values = cache.matrix(fold, ordered, training=False)
        probabilities[fold.validation_indices] = model.predict_proba(values)
        fold_ids[fold.validation_indices] = fold.fold_id
    oof = OOFPathArchetypeResult(
        probabilities=probabilities,
        fold_ids=fold_ids,
        folds=folds,
        models=models,
        classes=np.array([0, 1]),
        feature_columns=ordered,
        staged_matrix_cache=cache,
    )
    return ordered_features, target, folds, oof


def test_staged_mda_reuses_first_stage_models_and_batches_exact_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    class CountingCatBoost:
        fit_calls = 0
        predict_calls = 0

        def __init__(self, **_: object) -> None:
            self.classes_ = np.array([0, 1])

        def fit(
            self, values: np.ndarray, target: np.ndarray, **_: object
        ) -> "CountingCatBoost":
            type(self).fit_calls += 1
            self.classes_ = np.unique(target)
            return self

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            type(self).predict_calls += 1
            score = np.asarray(values, dtype=float).sum(axis=1)
            probability = 1.0 / (1.0 + np.exp(-((score % 19.0) - 9.0)))
            return np.column_stack([1.0 - probability, probability])

    monkeypatch.setattr(module, "_require_catboost", lambda: CountingCatBoost)
    features, target, folds, oof = _staged_mda_fixture()
    batched_config = PathArchetypeConfig(
        oof_folds=2,
        embargo=pd.Timedelta(0),
        permutation_screening_enabled=False,
        permutation_batch_max_bytes=1024**3,
    )
    batched_selected, batched = staged_permutation_selection(
        features,
        target,
        oof,
        mandatory_features=("pre_5",),
        stages=(6, 4),
        config=batched_config,
    )
    batched_calls = CountingCatBoost.predict_calls
    assert CountingCatBoost.fit_calls == len(folds)
    assert CountingCatBoost.fit_calls < len(folds) * 2
    first_stage = batched.loc[batched["stage"] == 6]
    assert first_stage["stage_reused_oof_models"].all()
    assert (first_stage["stage_fit_calls"] == 0).all()
    assert (first_stage["stage_baseline_predict_calls"] == 0).all()
    assert first_stage["stage_validation_matrix_cache_used"].all()
    assert (first_stage["stage_validation_matrix_cache_bytes"] > 0).all()
    assert all(
        np.isfinite(matrix).all()
        for matrix in oof.staged_matrix_cache.validation_matrices.values()
    )
    assert (batched["stage_selection_semantics"] == "exact_full_mda").all()

    CountingCatBoost.fit_calls = 0
    CountingCatBoost.predict_calls = 0
    _, _, _, unbatched_oof = _staged_mda_fixture()
    unbatched_selected, unbatched = staged_permutation_selection(
        features,
        target,
        unbatched_oof,
        mandatory_features=("pre_5",),
        stages=(6, 4),
        config=PathArchetypeConfig(
            oof_folds=2,
            embargo=pd.Timedelta(0),
            permutation_screening_enabled=False,
            permutation_batch_max_bytes=1,
        ),
    )
    assert batched_selected == unbatched_selected
    score_columns = [
        "stage",
        "feature",
        "loss_increase",
        "stability",
        "drift_instability",
        "score",
        "selected",
    ]
    pd.testing.assert_frame_equal(
        batched.loc[:, score_columns]
        .sort_values(["stage", "feature"])
        .reset_index(drop=True),
        unbatched.loc[:, score_columns]
        .sort_values(["stage", "feature"])
        .reset_index(drop=True),
        check_exact=True,
    )
    assert batched_calls < CountingCatBoost.predict_calls


def test_staged_mda_screening_is_deterministic_and_protects_mandatory_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    class FakeCatBoost:
        def __init__(self, **_: object) -> None:
            self.classes_ = np.array([0, 1])

        def fit(
            self, values: np.ndarray, target: np.ndarray, **_: object
        ) -> "FakeCatBoost":
            self.classes_ = np.unique(target)
            return self

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            score = np.asarray(values, dtype=float).sum(axis=1)
            probability = 1.0 / (1.0 + np.exp(-((score % 17.0) - 8.0)))
            return np.column_stack([1.0 - probability, probability])

    monkeypatch.setattr(module, "_require_catboost", lambda: FakeCatBoost)
    features, target, _folds, oof = _staged_mda_fixture()
    selected, report = staged_permutation_selection(
        features,
        target,
        oof,
        mandatory_features=("pre_5",),
        stages=(3,),
        config=PathArchetypeConfig(
            oof_folds=2,
            embargo=pd.Timedelta(0),
            permutation_screening_enabled=True,
            permutation_screen_margin=1,
            permutation_batch_max_bytes=1024**3,
        ),
    )
    repeated_selected, repeat = staged_permutation_selection(
        features,
        target,
        _staged_mda_fixture()[3],
        mandatory_features=("pre_5",),
        stages=(3,),
        config=PathArchetypeConfig(
            oof_folds=2,
            embargo=pd.Timedelta(0),
            permutation_screening_enabled=True,
            permutation_screen_margin=1,
            permutation_batch_max_bytes=1024**3,
        ),
    )
    assert "pre_5" in selected
    assert selected == repeated_selected
    assert report["stage_screening_used"].all()
    assert (report["stage_full_mda_candidate_count"] == 4).all()
    assert (report["stage_screened_out_count"] == 2).all()
    assert report["stage_screen_fold_ids"].iloc[0] == [0, 1]
    assert (report["stage_screen_fold_count"] == 2).all()
    assert (
        report["stage_screen_aggregation"]
        == "max_first_last_loss_conservative_retention"
    ).all()
    assert report.loc[report["feature"] == "pre_5", "full_mda_evaluated"].all()
    assert (
        report["stage_selection_semantics"]
        == "deterministic_screened_mda_approximation"
    ).all()
    pd.testing.assert_frame_equal(
        report.drop(
            columns=[
                "stage_fit_seconds",
                "stage_baseline_predict_seconds",
                "stage_screen_seconds",
                "stage_permutation_predict_seconds",
                "stage_total_seconds",
            ]
        )
        .sort_values("feature")
        .reset_index(drop=True),
        repeat.drop(
            columns=[
                "stage_fit_seconds",
                "stage_baseline_predict_seconds",
                "stage_screen_seconds",
                "stage_permutation_predict_seconds",
                "stage_total_seconds",
            ]
        )
        .sort_values("feature")
        .reset_index(drop=True),
        check_exact=True,
    )


def test_classifier_fit_passes_shared_cache_to_stage_one_mda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    captured: dict[str, object] = {}
    rows = 24
    features = pd.DataFrame(
        {
            "base_x": np.arange(rows, dtype=float),
            "meta_x": np.arange(rows, dtype=float) % 3,
        }
    )
    target = np.where(features["meta_x"] == 0.0, "a", "b")
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    config = PathArchetypeConfig(oof_folds=2, embargo=pd.Timedelta(0))

    selector = module.FastSelectorResult(
        selected_features=("meta_x", "base_x"),
        candidate_features=("meta_x", "base_x"),
        mandatory_features=("base_x",),
        availability={"base_x": 1.0, "meta_x": 1.0},
        scores=pd.DataFrame(index=["base_x", "meta_x"]),
        correlation_clusters=(("base_x",), ("meta_x",)),
        proxy_backend="test",
    )

    def fake_oof(*args: object, **kwargs: object) -> OOFPathArchetypeResult:
        cache = kwargs.get("staged_matrix_cache")
        captured.setdefault("first_cache", cache)
        captured.setdefault(
            "first_force_classes_count", kwargs.get("force_classes_count")
        )
        feature_frame = args[0]
        return OOFPathArchetypeResult(
            probabilities=np.full((len(feature_frame), 2), 0.5),
            fold_ids=np.zeros(len(feature_frame), dtype=int),
            folds=[],
            models=[],
            classes=np.array(["a", "b"]),
            feature_columns=tuple(feature_frame.columns),
            staged_matrix_cache=cache,
        )

    def fake_staged(
        feature_frame: pd.DataFrame,
        _target: object,
        oof: OOFPathArchetypeResult,
        **_: object,
    ) -> tuple[list[str], pd.DataFrame]:
        captured["stage_features"] = tuple(feature_frame.columns)
        captured["stage_cache"] = oof.staged_matrix_cache
        return ["base_x"], pd.DataFrame()

    class FakeFinalCatBoost:
        def __init__(self, **_: object) -> None:
            self.classes_ = np.array([0, 1])

        def fit(self, *_args: object, **_kwargs: object) -> "FakeFinalCatBoost":
            return self

    monkeypatch.setattr(
        module, "fast_select_preentry_features", lambda *_args, **_kwargs: selector
    )
    monkeypatch.setattr(module, "fit_purged_chronological_oof_catboost", fake_oof)
    monkeypatch.setattr(module, "staged_permutation_selection", fake_staged)
    monkeypatch.setattr(module, "_require_catboost", lambda: FakeFinalCatBoost)
    classifier = PathArchetypeClassifier.fit(
        features,
        target,
        timestamps,
        mandatory_features=("base_x",),
        config=config,
        config_mapping={
            "base_shared_feature_keys": ["base_x"],
            "base_long_feature_keys": [],
            "base_short_feature_keys": [],
            "meta_shared_feature_keys": ["meta_x"],
            "META_BASE_PERFORMANCE_FEATURE_KEYS": [],
            "MODEL_DERIVED_META_PERFORMANCE_FEATURE_KEYS": [],
        },
        run_permutation_selection=True,
    )
    assert classifier.feature_columns == ("base_x",)
    assert captured["stage_features"] == ("base_x", "meta_x")
    assert captured["first_cache"] is captured["stage_cache"]
    assert captured["first_cache"] is not None
    assert captured["first_force_classes_count"] is False


def test_classifier_fit_uses_configured_universe_and_records_oof_report() -> None:
    rows = 36
    features = pd.DataFrame(
        {
            "base_x": np.arange(rows, dtype=float),
            "meta_x": np.tile([0.0, 1.0], rows // 2),
            "base_model_score": np.arange(rows, dtype=float),
        }
    )
    config_mapping = {
        "base_shared_feature_keys": ["base_x"],
        "base_long_feature_keys": [],
        "base_short_feature_keys": [],
        "meta_shared_feature_keys": ["meta_x", "base_model_score"],
        "META_BASE_PERFORMANCE_FEATURE_KEYS": ["base_model_score"],
        "MODEL_DERIVED_META_PERFORMANCE_FEATURE_KEYS": [],
    }
    classifier = PathArchetypeClassifier.fit(
        features,
        np.where(features["meta_x"] > 0, "a", "b"),
        pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
        config_mapping=config_mapping,
        config=PathArchetypeConfig(
            oof_folds=2, embargo=pd.Timedelta(0), selector_sample_rows=rows
        ),
        params={"iterations": 20},
    )
    assert set(classifier.feature_columns) == {"base_x", "meta_x"}
    assert classifier.training_report is not None
    assert classifier.training_report["configured_universe"] == ["base_x", "meta_x"]
    assert classifier.training_report["oof_diagnostics"]["classwise"]


def test_post_hpo_class_balance_mini_sweep_uses_fixed_params_and_exposes_all_oof_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    observed_params: list[dict[str, object]] = []
    callback_arms: list[tuple[str, str]] = []
    callback_order: list[tuple[object, ...]] = []

    def fake_oof(*_args: object, **kwargs: object) -> object:
        params = dict(kwargs["params"])
        observed_params.append(params)
        fold_callback = kwargs["fold_callback"]
        assert callable(fold_callback)
        fold_callback(
            0,
            np.array([[0.8, 0.2], [0.2, 0.8]]),
            np.array([0, 0, -1, -1]),
        )
        fold_callback(
            1,
            np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]]),
            np.array([0, 0, 1, 1]),
        )
        return module.OOFPathArchetypeResult(
            probabilities=np.array(
                [
                    [0.8, 0.2],
                    [0.2, 0.8],
                    [0.7, 0.3],
                    [0.1, 0.9],
                ]
            ),
            fold_ids=np.array([0, 0, 1, 1]),
            folds=[],
            models=[],
            classes=np.array(["a", "b"]),
            diagnostics={"logloss": 0.2},
        )

    monkeypatch.setattr(module, "fit_purged_chronological_oof_catboost", fake_oof)
    config = PathArchetypeConfig(class_balance_min_class_support=1)
    result = module.sweep_purged_catboost_class_balance_arms(
        pd.DataFrame({"pre_x": [0.0, 1.0, 2.0, 3.0]}),
        ["a", "b", "a", "b"],
        pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
        structural_params={
            "depth": 7,
            "learning_rate": 0.04,
            # Legacy HPO balance metadata is intentionally stripped: this
            # post-HPO API evaluates every arm rather than inheriting a winner.
            "class_balance_arm": "frequency_power_0.25",
            "class_balance_selection_provenance": {"legacy": "oof"},
        },
        config=config,
        arm_callback=lambda arm: (
            callback_arms.append((arm.arm, arm.status)),
            callback_order.append(("arm", arm.arm, arm.status)),
        ),
        arm_fold_callback=lambda arm, fold_index, _probabilities, fold_ids: (
            callback_order.append(("fold", arm, fold_index, int(np.sum(fold_ids >= 0))))
        ),
    )
    expected_arms = [
        "uniform",
        "frequency_power_0.25",
        "frequency_power_0.50",
        "frequency_power_0.75",
    ]
    assert [arm.arm for arm in result.arms] == expected_arms
    assert [arm.status for arm in result.arms] == ["eligible"] * 4
    assert callback_arms == [(arm, "eligible") for arm in expected_arms]
    assert callback_order == [
        event
        for arm in expected_arms
        for event in (
            ("fold", arm, 0, 2),
            ("fold", arm, 1, 4),
            ("arm", arm, "eligible"),
        )
    ]
    assert list(result.oof_by_arm) == expected_arms
    assert all(arm.oof is not None for arm in result.arms)
    assert all(
        arm.guard["evaluation_scope"]
        == "frozen_purged_oof_validation_rows_only_aggregate_and_per_fold"
        for arm in result.arms
    )
    assert [params["class_balance_arm"] for params in observed_params] == expected_arms
    assert all(
        {key: value for key, value in params.items() if key != "class_balance_arm"}
        == {"depth": 7, "learning_rate": 0.04}
        for params in observed_params
    )
    report = result.report()
    assert report["winner_selected"] is False
    assert "probabilities" not in report["arms"][0]
    assert report["contract"]["intended_sequence"][-1] == "final_refit"
    assert report["contract"]["final_refit_used_for_selection"] is False


def test_post_hpo_class_balance_mini_sweep_reports_guard_rejections_without_stopping_other_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    callback_statuses: list[tuple[str, str]] = []

    def fake_oof(*_args: object, **kwargs: object) -> object:
        arm = kwargs["params"]["class_balance_arm"]
        probabilities = (
            np.array([[0.99, 0.01], [0.99, 0.01], [0.99, 0.01], [0.99, 0.01]])
            if arm == "frequency_power_0.50"
            else np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
        )
        return module.OOFPathArchetypeResult(
            probabilities=probabilities,
            fold_ids=np.array([0, 0, 1, 1]),
            folds=[],
            models=[],
            classes=np.array(["a", "b"]),
            diagnostics={"logloss": 0.2},
        )

    monkeypatch.setattr(module, "fit_purged_chronological_oof_catboost", fake_oof)
    result = module.sweep_purged_catboost_class_balance_arms(
        pd.DataFrame({"pre_x": [0.0, 1.0, 2.0, 3.0]}),
        ["a", "b", "a", "b"],
        pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
        structural_params={"depth": 6},
        config=PathArchetypeConfig(
            class_balance_min_class_support=1,
            class_balance_min_predicted_share=0.02,
        ),
        arm_callback=lambda arm: callback_statuses.append((arm.arm, arm.status)),
    )
    by_arm = {arm.arm: arm for arm in result.arms}
    assert by_arm["frequency_power_0.50"].status == "rejected_by_oof_guard"
    assert "predicted-share collapse" in by_arm["frequency_power_0.50"].rejection_reason
    assert len(callback_statuses) == 4
    assert len(result.eligible_arms) == 3
    with pytest.raises(ValueError, match="final-refit balance payloads are forbidden"):
        module.sweep_purged_catboost_class_balance_arms(
            pd.DataFrame({"pre_x": [0.0, 1.0, 2.0, 3.0]}),
            ["a", "b", "a", "b"],
            pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
            structural_params={"class_balance_final_weights": [1.0, 1.0]},
        )


def test_target_fields_are_rejected_before_classifier_fit() -> None:
    for realized_column in (
        "path_arch_efficiency",
        "path_archetype",
        "path_shape_archetype",
        "path_realization_strength",
        "discovery_cluster_id",
    ):
        with pytest.raises(ValueError, match="non-pre-entry"):
            validate_preentry_features(["pre_rsi", realized_column])


def test_chronological_folds_purge_open_labels_and_apply_embargo() -> None:
    ts = pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC")
    ends = ts + pd.Timedelta(hours=4)
    folds = purged_chronological_folds(
        ts, label_end=ends, n_splits=3, embargo=pd.Timedelta(hours=3)
    )
    for fold in folds:
        valid_start = ts[fold.validation_indices[0]]
        assert (ends[fold.train_indices] < valid_start).all()
        assert (ts[fold.train_indices] < valid_start - pd.Timedelta(hours=3)).all()


def test_fixed_monthly_outer_oof_uses_exact_may_june_july_boundaries_and_train_only_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    timestamps = pd.date_range(
        "2026-04-01",
        "2026-07-07",
        freq="D",
        tz="UTC",
        inclusive="left",
    )
    label_end = timestamps + pd.Timedelta(hours=12)
    # This row is before the May decision boundary but is unresolved exactly
    # at it, so it must never reach the May fold's training loss or weights.
    label_end = pd.Series(label_end)
    label_end.loc[timestamps == pd.Timestamp("2026-04-29", tz="UTC")] = pd.Timestamp(
        "2026-05-01", tz="UTC"
    )
    labels = np.where(np.arange(len(timestamps)) % 5 == 0, "b", "a")
    features = pd.DataFrame({"pre_x": np.arange(len(timestamps), dtype=float)})
    row_ids = [f"candidate-{index}" for index in range(len(features))]
    plan = module.build_fixed_monthly_outer_oof_plan(
        timestamps,
        label_end=label_end,
        row_ids=row_ids,
    )
    expected_starts = pd.DatetimeIndex(
        [
            "2026-05-01 00:00:00+00:00",
            "2026-06-01 00:00:00+00:00",
            "2026-07-01 00:00:00+00:00",
        ]
    )
    assert [window.validation_start for window in plan.windows] == list(expected_starts)
    assert plan.report()["final_validation_month_may_be_partial"] is True
    expected_validation = np.flatnonzero(
        (timestamps >= expected_starts[0])
        & (timestamps < pd.Timestamp("2026-08-01", tz="UTC"))
    )
    assert np.array_equal(
        np.sort(np.concatenate([window.validation_indices for window in plan.windows])),
        expected_validation,
    )
    for window in plan.windows:
        assert np.all(timestamps[window.validation_indices] >= window.validation_start)
        assert np.all(
            timestamps[window.validation_indices] < window.validation_end_exclusive
        )
        assert np.all(label_end.iloc[window.train_indices] < window.validation_start)
        assert np.all(
            timestamps[window.train_indices]
            < window.validation_start - pd.Timedelta(hours=24)
        )
    may_window = plan.windows[0]
    assert not np.any(
        timestamps[may_window.train_indices] == pd.Timestamp("2026-04-29", tz="UTC")
    )
    assert not np.any(
        timestamps[may_window.train_indices] == pd.Timestamp("2026-04-30", tz="UTC")
    )
    assert (
        len(plan.windows[-1].validation_indices) < 31
    )  # July is intentionally partial.

    init_calls: list[dict[str, object]] = []

    class FakeCatBoost:
        classes_ = np.array([0, 1])
        tree_count_ = 7

        def __init__(self, **kwargs: object) -> None:
            init_calls.append(kwargs)

        def fit(
            self, _x: np.ndarray, _y: np.ndarray, **_kwargs: object
        ) -> "FakeCatBoost":
            return self

        def predict_proba(self, values: np.ndarray) -> np.ndarray:
            return np.tile(np.array([[0.7, 0.3]]), (len(values), 1))

        def get_best_iteration(self) -> int:
            return 5

    monkeypatch.setattr(module, "_require_catboost", lambda: FakeCatBoost)
    params = {
        "iterations": 20,
        "od_wait": 4,
        "class_balance_arm": "frequency_power_0.75",
    }
    frozen = {
        "schema": module.FIXED_MONTHLY_OUTER_OOF_FROZEN_PARAMS_SCHEMA,
        "selection_scope": "pre_first_outer_validation_month_development_only",
        "development_label_end_exclusive": "2026-05-01T00:00:00Z",
        "params_sha256": module._canonical_json_sha256(params),
        "final_refit_used_for_selection": False,
        "class_balance_selection_scope": (
            "pre_first_outer_validation_month_development_only"
        ),
    }
    callback_events: list[tuple[int, int]] = []
    result = module.fit_fixed_monthly_outer_oof_catboost(
        features,
        labels,
        timestamps,
        label_end=label_end,
        params=params,
        frozen_params_provenance=frozen,
        row_ids=row_ids,
        plan=plan,
        config=PathArchetypeConfig(class_order=("a", "b")),
        fold_callback=lambda window, _probabilities, fold_ids: callback_events.append(
            (window.fold_id, int(np.sum(fold_ids >= 0)))
        ),
    )
    assert callback_events == [
        (0, len(plan.windows[0].validation_indices)),
        (
            1,
            len(plan.windows[0].validation_indices)
            + len(plan.windows[1].validation_indices),
        ),
        (2, len(expected_validation)),
    ]
    assert np.array_equal(np.flatnonzero(result.oof.fold_ids >= 0), expected_validation)
    reports = result.oof.diagnostics["fold_fit_reports"]
    assert [report["fold_id"] for report in reports] == [0, 1, 2]
    assert all(
        report["class_balance"]["weight_estimation_scope"]
        == "fixed_monthly_outer_oof_fold_train_only"
        for report in reports
    )
    assert all(
        report["fixed_monthly_outer_oof_window"]["latest_train_label_end_ts"]
        < report["fixed_monthly_outer_oof_window"]["validation_start_utc"]
        for report in reports
    )
    assert len(init_calls) == 3
    assert all(
        np.max(call["class_weights"]) / np.min(call["class_weights"]) <= 4.0
        for call in init_calls
    )
    assert (
        result.report()["outer_oof_plan"]["windows"][2]["validation_end_exclusive_utc"]
        == "2026-08-01 00:00:00+00:00"
    )

    original = plan.windows[0]
    tampered_plan = module.FixedMonthlyOuterOOFPlan(
        windows=(
            module.FixedMonthlyOuterOOFWindow(
                fold_id=original.fold_id,
                validation_start=original.validation_start,
                validation_end_exclusive=original.validation_end_exclusive,
                train_indices=original.train_indices[1:],
                validation_indices=original.validation_indices,
                latest_train_decision_ts=original.latest_train_decision_ts,
                latest_train_label_end_ts=original.latest_train_label_end_ts,
            ),
            *plan.windows[1:],
        ),
        embargo=plan.embargo,
        input_rows=plan.input_rows,
        input_row_identity_sha256=plan.input_row_identity_sha256,
        input_min_timestamp=plan.input_min_timestamp,
        input_max_timestamp=plan.input_max_timestamp,
    )
    with pytest.raises(
        ValueError, match="does not match the supplied ordered temporal rows"
    ):
        module.fit_fixed_monthly_outer_oof_catboost(
            features,
            labels,
            timestamps,
            label_end=label_end,
            params=params,
            frozen_params_provenance=frozen,
            row_ids=row_ids,
            plan=tampered_plan,
            config=PathArchetypeConfig(class_order=("a", "b")),
        )

    invalid_frozen = {
        **frozen,
        "development_label_end_exclusive": "2026-05-02T00:00:00Z",
    }
    with pytest.raises(ValueError, match="development cutoff exceeds"):
        module.fit_fixed_monthly_outer_oof_catboost(
            features,
            labels,
            timestamps,
            label_end=label_end,
            params=params,
            frozen_params_provenance=invalid_frozen,
            row_ids=row_ids,
            plan=plan,
            config=PathArchetypeConfig(class_order=("a", "b")),
        )


def test_catboost_requirement_is_clean_when_dependency_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    monkeypatch.setattr(module, "catboost_available", lambda: False)
    # Fast selection remains available without a CatBoost installation.
    frame = pd.DataFrame({"pre_x": range(12)})
    result = module.fast_select_preentry_features(
        frame, [0, 1] * 6, config=PathArchetypeConfig(selector_sample_rows=12)
    )
    assert result.proxy_backend == "binned_multiclass_proxy"
    if not catboost_available():
        with pytest.raises(CatBoostUnavailableError):
            module._require_catboost()
