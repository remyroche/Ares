from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def test_future_path_summary_has_required_path_signatures() -> None:
    result = summarize_future_path([0.2, 0.6, -0.3, 1.2, 0.8], take_profit_r=1.0, stop_r=0.25)
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
    features = pd.DataFrame({
        "pre_signal": signal,
        "pre_duplicate": signal * 2.0,
        "pre_sparse": [np.nan] * 10 + list(signal[10:]),
        "pre_noise": np.tile([0.0, 1.0], n // 2),
    })
    target = np.where(signal % 3 == 0, "a", "b")
    result = fast_select_preentry_features(
        features, target, warmup_mask=np.r_[np.ones(10, dtype=bool), np.zeros(n - 10, dtype=bool)],
        mandatory_features=["pre_signal"], config=PathArchetypeConfig(max_feature_candidates=10, selector_sample_rows=80),
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


def test_configured_universe_uses_base_meta_only_and_excludes_performance_outcomes() -> None:
    config = {
        "base_shared_feature_keys": ["base_x", "BASE_NESTED"],
        "base_long_feature_keys": [],
        "base_short_feature_keys": [],
        "meta_shared_feature_keys": ["meta_x", "base_model_score", "path_arch_peak_mfe_r"],
        "BASE_NESTED": ["base_nested"],
        "META_BASE_PERFORMANCE_FEATURE_KEYS": ["base_model_score"],
        "MODEL_DERIVED_META_PERFORMANCE_FEATURE_KEYS": [],
    }
    selected = configured_base_meta_preselection_universe(
        ["base_x", "base_nested", "meta_x", "base_model_score", "path_arch_peak_mfe_r", "rogue"],
        config_mapping=config,
    )
    assert selected == ("base_x", "base_nested", "meta_x")


def test_configured_universe_excludes_actual_model_derived_fields_but_allows_exact_frozen_aegmm() -> None:
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
    frame = pd.DataFrame({"pre_sparse": values, "pre_complete": np.arange(100, dtype=float)})
    with pytest.raises(ValueError, match="strict >95%"):
        fast_select_preentry_features(
            frame, [0, 1] * 50, mandatory_features=["pre_sparse"],
            config=PathArchetypeConfig(selector_sample_rows=100),
        )


def test_multiclass_diagnostics_include_reliability_confusion_and_temporal_stability() -> None:
    diagnostics = multiclass_classification_diagnostics(
        [0, 1, 0, 1],
        np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.7, 0.3]]),
        fold_ids=[0, 0, 1, 1], class_names=["failed", "clean"],
    )
    assert diagnostics["logloss"] > 0.0
    assert diagnostics["brier_macro"] == pytest.approx(diagnostics["brier_weighted"])
    assert diagnostics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert set(diagnostics["classwise"]["failed"]) >= {"ece", "f1", "recall"}
    assert diagnostics["temporal_stability"]["logloss_worst"] >= diagnostics["temporal_stability"]["logloss_mean"]
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
        module._catboost_params(
            PathArchetypeConfig(), {"class_weights": [1.0, 2.0]}
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

    monkeypatch.setattr(module, "_physical_ram_bytes", lambda: 16 * 1024 ** 3)
    safe = catboost_resource_contract(PathArchetypeConfig(catboost_thread_count=64))
    assert safe["physical_ram_bytes"] == 16 * 1024 ** 3
    assert safe["os_reserve_bytes"] == 4 * 1024 ** 3
    assert safe["used_ram_limit_bytes"] == 12 * 1024 ** 3
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

    monkeypatch.setattr(module, "_physical_ram_bytes", lambda: 16 * 1024 ** 3)
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


def test_optuna_hpo_reports_minimizing_purged_oof_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    def fake_oof(*args: object, **kwargs: object) -> object:
        return module.OOFPathArchetypeResult(
            probabilities=np.array([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]]),
            fold_ids=np.array([0, 0, 1, 1]), folds=[], models=[], classes=np.array(["a", "b"]),
            diagnostics={"logloss": 0.2},
        )

    monkeypatch.setattr(module, "fit_purged_chronological_oof_catboost", fake_oof)
    result = module.optimize_purged_catboost_hpo(
        pd.DataFrame({"pre_x": [0.0, 1.0, 2.0, 3.0]}), ["a", "b", "a", "b"],
        pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"), n_trials=1,
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
        "MedianPruner(startup_trials=3,warmup_steps=0,"
        "interval_steps=1,min_trials=2)"
    )
    assert report["trials"][0]["study_no_improvement_patience_trials"] == 30


def test_optuna_hpo_progress_is_atomic_and_reuses_fresh_best_oof(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    calls = {"oof": 0}

    def fake_oof(*args: object, **kwargs: object) -> object:
        calls["oof"] += 1
        return module.OOFPathArchetypeResult(
            probabilities=np.array([
                [0.8, 0.2], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9],
            ]),
            fold_ids=np.array([0, 0, 1, 1]), folds=[], models=[],
            classes=np.array(["a", "b"]), diagnostics={"logloss": 0.2},
        )

    monkeypatch.setattr(module, "fit_purged_chronological_oof_catboost", fake_oof)
    progress_path = tmp_path / "progress.json"
    common = {
        "features": pd.DataFrame({"pre_x": [0.0, 1.0, 2.0, 3.0]}),
        "target": ["a", "b", "a", "b"],
        "timestamps": pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"),
        "progress_path": progress_path,
    }
    first = module.optimize_purged_catboost_hpo(n_trials=1, **common)
    assert first.best_oof_reused_from_current_process is True
    assert calls["oof"] == 1
    first_progress = json.loads(progress_path.read_text())
    assert first_progress["status"] == "complete"
    assert first_progress["target_trials"] == 1
    assert first_progress["completed_trial_count"] == 1
    assert first_progress["current_trial"]["number"] == 0
    assert first_progress["current_trial"]["state"] == "COMPLETE"
    assert first_progress["best_params"]
    assert first_progress["no_wall_clock_timeout"] is True


def test_purged_oof_uses_validation_eval_set_and_early_stopping(monkeypatch: pytest.MonkeyPatch) -> None:
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
        pd.DataFrame({"pre_x": np.arange(rows, dtype=float)}), [0, 1] * (rows // 2), timestamps,
        config=PathArchetypeConfig(oof_folds=3, embargo=pd.Timedelta(0)),
    )
    assert result.diagnostics is not None
    assert fit_calls
    assert all("eval_set" in call and call["early_stopping_rounds"] == 150 for call in fit_calls)
    assert init_calls and all(call["classes_count"] == 2 for call in init_calls)
    reports = result.diagnostics["fold_fit_reports"]
    assert len(reports) == len(fit_calls)
    assert all(report["use_best_model"] is True for report in reports)
    assert all(report["eval_set_used"] is True for report in reports)
    assert all(report["early_stopping_rounds"] == 150 for report in reports)
    assert all(report["best_iteration"] == 12 for report in reports)
    assert all(report["tree_count"] == 17 for report in reports)


def test_multiclass_log_loss_vectorized_lookup_preserves_class_mapping_semantics() -> None:
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
    expected = float(-np.mean(np.log(np.clip(
        np.array([probabilities[row, positions.get(int(label), 0)] for row, label in enumerate(target)]),
        1e-12,
        1.0,
    ))))

    assert module.multiclass_log_loss(target, probabilities, classes) == pytest.approx(expected)


def _staged_mda_fixture() -> tuple[pd.DataFrame, np.ndarray, list[object], OOFPathArchetypeResult]:
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
    folds = purged_chronological_folds(
        timestamps, n_splits=2, embargo=pd.Timedelta(0)
    )
    ordered = ("pre_5", *columns[:-1])
    ordered_features = features.loc[:, ordered]
    cache = build_staged_permutation_matrix_cache(
        ordered_features, timestamps,
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

        def fit(self, values: np.ndarray, target: np.ndarray, **_: object) -> "CountingCatBoost":
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
        permutation_batch_max_bytes=1024 ** 3,
    )
    batched_selected, batched = staged_permutation_selection(
        features, target, oof, mandatory_features=("pre_5",), stages=(6, 4),
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
        features, target, unbatched_oof, mandatory_features=("pre_5",), stages=(6, 4),
        config=PathArchetypeConfig(
            oof_folds=2,
            embargo=pd.Timedelta(0),
            permutation_screening_enabled=False,
            permutation_batch_max_bytes=1,
        ),
    )
    assert batched_selected == unbatched_selected
    score_columns = [
        "stage", "feature", "loss_increase", "stability",
        "drift_instability", "score", "selected",
    ]
    pd.testing.assert_frame_equal(
        batched.loc[:, score_columns].sort_values(["stage", "feature"]).reset_index(drop=True),
        unbatched.loc[:, score_columns].sort_values(["stage", "feature"]).reset_index(drop=True),
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

        def fit(self, values: np.ndarray, target: np.ndarray, **_: object) -> "FakeCatBoost":
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
            permutation_batch_max_bytes=1024 ** 3,
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
            permutation_batch_max_bytes=1024 ** 3,
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
        report.drop(columns=[
            "stage_fit_seconds", "stage_baseline_predict_seconds",
            "stage_screen_seconds", "stage_permutation_predict_seconds",
            "stage_total_seconds",
        ]).sort_values("feature").reset_index(drop=True),
        repeat.drop(columns=[
            "stage_fit_seconds", "stage_baseline_predict_seconds",
            "stage_screen_seconds", "stage_permutation_predict_seconds",
            "stage_total_seconds",
        ]).sort_values("feature").reset_index(drop=True),
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
        captured.setdefault("first_force_classes_count", kwargs.get("force_classes_count"))
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
        feature_frame: pd.DataFrame, _target: object, oof: OOFPathArchetypeResult,
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

    monkeypatch.setattr(module, "fast_select_preentry_features", lambda *_args, **_kwargs: selector)
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
    features = pd.DataFrame({
        "base_x": np.arange(rows, dtype=float),
        "meta_x": np.tile([0.0, 1.0], rows // 2),
        "base_model_score": np.arange(rows, dtype=float),
    })
    config_mapping = {
        "base_shared_feature_keys": ["base_x"],
        "base_long_feature_keys": [],
        "base_short_feature_keys": [],
        "meta_shared_feature_keys": ["meta_x", "base_model_score"],
        "META_BASE_PERFORMANCE_FEATURE_KEYS": ["base_model_score"],
        "MODEL_DERIVED_META_PERFORMANCE_FEATURE_KEYS": [],
    }
    classifier = PathArchetypeClassifier.fit(
        features, np.where(features["meta_x"] > 0, "a", "b"),
        pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC"),
        config_mapping=config_mapping,
        config=PathArchetypeConfig(oof_folds=2, embargo=pd.Timedelta(0), selector_sample_rows=rows),
        params={"iterations": 20},
    )
    assert set(classifier.feature_columns) == {"base_x", "meta_x"}
    assert classifier.training_report is not None
    assert classifier.training_report["configured_universe"] == ["base_x", "meta_x"]
    assert classifier.training_report["oof_diagnostics"]["classwise"]


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
    folds = purged_chronological_folds(ts, label_end=ends, n_splits=3, embargo=pd.Timedelta(hours=3))
    for fold in folds:
        valid_start = ts[fold.validation_indices[0]]
        assert (ends[fold.train_indices] < valid_start).all()
        assert (ts[fold.train_indices] < valid_start - pd.Timedelta(hours=3)).all()


def test_catboost_requirement_is_clean_when_dependency_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    import extreme_price_movements.catboost_archetype_classifier as module

    monkeypatch.setattr(module, "catboost_available", lambda: False)
    # Fast selection remains available without a CatBoost installation.
    frame = pd.DataFrame({"pre_x": range(12)})
    result = module.fast_select_preentry_features(frame, [0, 1] * 6, config=PathArchetypeConfig(selector_sample_rows=12))
    assert result.proxy_backend == "binned_multiclass_proxy"
    if not catboost_available():
        with pytest.raises(CatBoostUnavailableError):
            module._require_catboost()
