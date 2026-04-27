import numpy as np
import pandas as pd

import extreme_price_movements.ebm_on_lgbm as lor
from extreme_price_movements.ebm_on_lgbm import (
    SplinePostProcessor,
    _feature_shape_scores,
    _fit_final_model,
    _hpo_objective_from_aggregate,
    _metric_pack,
    _post_hpo_manage_features,
    _prescreen_features,
    _select_smallest_within_one_se,
    _stage_partition_indices,
    train_ebm_on_lgbm_candidate,
)


class FakeEBMClassifier:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.coef_ = None
        self.term_scores_ = []
        self.term_features_ = []

    def fit(self, X, y, sample_weight=None):
        x = np.asarray(X, dtype=np.float32)
        yy = np.asarray(y, dtype=np.float32)
        yy = yy - float(np.mean(yy))
        denom = np.std(x, axis=0) * max(float(np.std(yy)), 1e-6)
        numer = np.mean((x - np.mean(x, axis=0)) * yy[:, None], axis=0)
        coef = np.divide(
            numer,
            denom,
            out=np.zeros_like(numer, dtype=np.float32),
            where=denom > 1e-9,
        )
        self.coef_ = np.nan_to_num(coef, nan=0.0).astype(np.float32)
        self.term_features_ = [(i,) for i in range(x.shape[1])]
        grid = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
        self.term_scores_ = [grid * float(c) for c in self.coef_]
        return self

    def predict_proba(self, X):
        x = np.asarray(X, dtype=np.float32)
        z = x @ self.coef_
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
        return np.column_stack([1.0 - p, p]).astype(np.float32)


class FakeEBMRegressor(FakeEBMClassifier):
    def predict(self, X):
        return np.asarray(X, dtype=np.float32) @ self.coef_


def test_spline_postprocessor_identity_fallback_is_finite():
    pp = SplinePostProcessor(mode="classifier").fit(
        np.ones(12, dtype=np.float32),
        np.array([0, 1] * 6, dtype=np.float32),
    )

    out = pp.predict(np.array([0.2, 0.5, 0.8], dtype=np.float32))

    assert pp.identity is True
    assert np.all(np.isfinite(out))
    assert np.all((out > 0.0) & (out < 1.0))


def test_feature_shape_scores_are_finite_non_negative():
    m1 = FakeEBMClassifier().fit(
        pd.DataFrame(np.random.default_rng(1).normal(size=(40, 3))),
        np.array([0, 1] * 20),
    )
    m2 = FakeEBMClassifier().fit(
        pd.DataFrame(np.random.default_rng(2).normal(size=(40, 3))),
        np.array([0, 1] * 20),
    )

    scores = _feature_shape_scores([m1, m2], ["a", "b", "c"])

    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0)


def test_hpo_objective_uses_lift_and_stability_weights():
    score = _hpo_objective_from_aggregate({"lift30": 2.0, "stability30": 0.5})

    assert score == 0.65 * 2.0 + 0.35 * 0.5


def test_metric_pack_uses_grouped_stability_when_groups_available():
    y = np.array(([0, 1] * 30) + ([0, 0, 1, 1] * 15) + ([1, 1, 0, 0] * 15))
    pred = np.linspace(0.0, 1.0, len(y), dtype=np.float32)
    groups = np.array(["w1"] * 60 + ["w2"] * 60 + ["w3"] * 60)

    metrics = _metric_pack(y, pred, classifier=True, groups=groups)

    assert metrics["stability30_n_groups"] == 3.0
    assert metrics["stability30"] != metrics["stability30_proxy"]
    assert "stability30_group_mean" in metrics
    assert "stability30_group_std" in metrics


def test_stage_partition_uses_interwoven_ratio_without_overlap():
    n = 400
    y = np.array([0, 1] * (n // 2), dtype=np.int8)
    timestamps = pd.date_range("2025-01-01", periods=n, freq="12h")
    assets = np.array([f"a{i % 5}" for i in range(n)])

    parts = _stage_partition_indices(
        y,
        timestamps=timestamps,
        assets=assets,
        random_state=42,
    )

    assert sorted(parts) == ["fit_oof", "hpo", "lgbm_prune"]
    combined = np.concatenate(list(parts.values()))
    assert len(np.unique(combined)) == n
    assert abs(len(parts["lgbm_prune"]) / n - 0.35) <= 0.03
    assert abs(len(parts["hpo"]) / n - 0.10) <= 0.03
    assert abs(len(parts["fit_oof"]) / n - 0.55) <= 0.03


def test_spline_postprocessor_uses_isotonic_for_classifier_oof():
    x = np.linspace(0.05, 0.95, 120, dtype=np.float32)
    y = (x > 0.45).astype(np.int8)

    pp = SplinePostProcessor(mode="classifier").fit(
        x,
        y,
        use_dynamic_smoothing=True,
    )

    pred = pp.predict(np.array([0.2, 0.8], dtype=np.float32))
    assert pp.calibration_method == "spline_isotonic"
    assert pp.isotonic is not None
    assert pred[0] < pred[1]
    assert np.all((pred > 0.0) & (pred < 1.0))


def test_feature_shape_scores_zero_out_negative_shape_correlation():
    class OppositeShapeModel:
        term_features_ = [(0,)]

        def __init__(self, sign):
            self.term_scores_ = [sign * np.linspace(-1.0, 1.0, 16)]

    scores = _feature_shape_scores(
        [OppositeShapeModel(1.0), OppositeShapeModel(-1.0)], ["x"]
    )

    assert scores.shape == (1,)
    assert scores[0] == 0.0


def test_select_smallest_within_one_se_prefers_smaller_model():
    history = [
        {"round": 1, "J_final": 1.00, "J_se": 0.05, "n_features_end": 90},
        {"round": 2, "J_final": 0.97, "J_se": 0.04, "n_features_end": 55},
        {"round": 3, "J_final": 0.90, "J_se": 0.03, "n_features_end": 30},
    ]

    chosen = _select_smallest_within_one_se(history)

    assert chosen["round"] == 2


def test_prescreen_features_reduces_to_configured_cap(monkeypatch):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(180, 24)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.2, size=180) > 0).astype(np.int8)
    names = [f"f{i}" for i in range(x.shape[1])]
    monkeypatch.setattr(lor, "EBM_PRESCREEN_MAX_FEATURES", 10)
    monkeypatch.setattr(lor, "EBM_MIN_FEATURES", 4)

    active = _prescreen_features(x, y, names, classifier=True, random_state=42)

    assert 1 <= len(active) <= 10
    assert active.dtype == np.int32


def test_post_hpo_shape_management_keeps_selected_contract(monkeypatch):
    rng = np.random.default_rng(33)
    x = rng.normal(size=(120, 8)).astype(np.float32)
    y = (x[:, 0] - x[:, 1] + rng.normal(scale=0.4, size=120) > 0).astype(np.int8)
    X = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    features = list(X.columns)
    monkeypatch.setattr(lor, "EBM_MIN_FEATURES", 2)

    managed, _smooth_policy, metrics = _post_hpo_manage_features(
        FakeEBMClassifier,
        X,
        y,
        np.ones(len(y), dtype=np.float32),
        features,
        {"outer_bags": 1, "n_jobs": 1, "min_samples_leaf": 2},
        "classifier",
    )

    assert managed == features
    assert metrics["post_hpo_shape_dropped"] == 0.0
    assert metrics["post_hpo_shape_features"] == float(len(features))


def test_final_fit_keeps_missing_selected_tree_feature_contract(monkeypatch):
    import builtins

    rng = np.random.default_rng(44)
    x = rng.normal(size=(220, 4)).astype(np.float32)
    y = (x[:, 0] + rng.normal(scale=0.2, size=220) > 0).astype(np.int8)
    X = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    selected = ["f0", "lgbm_depth3_minpct0200_tree999_leaf0_soft"]

    def fake_augment(X_train_raw, y_train, X_eval_raw, random_state):
        del y_train, random_state
        return (
            X_train_raw.copy(),
            X_eval_raw.copy(),
            {
                "models": [],
                "tree_feature_config": {},
                "tree_feature_names": [],
                "tree_feature_scales": None,
            },
        )

    monkeypatch.setattr(lor, "_augment_with_tree_features", fake_augment)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "optuna" or name.startswith("optuna."):
            raise ImportError("optuna disabled for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    model = _fit_final_model(
        FakeEBMClassifier,
        X,
        y,
        np.ones(len(y), dtype=np.float32),
        ["f0"],
        selected,
        "classifier",
        42,
        [],
        np.full(len(y), 0.5, dtype=np.float32),
        {},
    )

    assert model.selected_features == selected
    assert model.metrics["feature_count"] == len(selected)
    assert model.metrics["n_leaf_features_kept"] == 1


def test_train_ebm_on_lgbm_candidate_with_fake_ebm(monkeypatch):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(240, 12)).astype(np.float32)
    y = (x[:, 0] - x[:, 1] + rng.normal(scale=0.3, size=240) > 0).astype(np.int8)
    X = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    monkeypatch.setattr(
        lor, "_load_ebm_classes", lambda: (FakeEBMClassifier, FakeEBMRegressor)
    )
    monkeypatch.setattr(lor, "EBM_MAX_ROUNDS", 2)
    monkeypatch.setattr(lor, "EBM_PRESCREEN_MAX_FEATURES", 10)
    monkeypatch.setattr(lor, "EBM_MIN_FEATURES", 4)
    monkeypatch.setattr(lor, "EBM_FOLD_SUBSAMPLE_ROWS", 80)

    result = train_ebm_on_lgbm_candidate(X, y, random_state=42, mode="classifier")

    assert result is not None
    assert result["full_fit_needed"] is True
    assert np.isfinite(result["oof_probs"]).sum() > 20
    assert result["selected_features_from_cv"].dtype == np.int32
    assert result["metrics"]["feature_count"] > 0
