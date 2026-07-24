import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements import lgbm_pipeline
from extreme_price_movements.lgbm_pipeline import (
    _direction_score_for_feature,
    _metric_pack,
    _objective_value,
    _topk_mda_score,
)
from extreme_price_movements.path_auxiliary_lgbm import (
    auxiliary_hpo_objective,
    auxiliary_hpo_sample_indices,
    auxiliary_reference_split,
    build_auxiliary_sample_weights,
    configured_auxiliary_feature_universe,
    expanding_monthly_oos_folds,
    expanding_purged_folds,
    fit_base_archetype_label_feature_contract,
    fit_hpo_oof_model,
    select_features_with_current_pipeline,
    transform_base_archetype_label_features,
)
from extreme_price_movements.path_auxiliary_targets import (
    ALL_SUPPORTIVE_LABEL_COLUMNS,
)


def _supportive_weight_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__log1p_time_to_first_meaningful_mfe_hours_12h__": np.log1p(
                [12.0, 6.0, 1.0]
            ),
            "__mfe_ge_1_5atr__": [0, 1, 1],
            "__mfe_ge_2_0atr__": [0, 0, 1],
            "__mfe_ge_3_0atr__": [0, 0, 1],
            "__mfe_ge_4_0atr__": [0, 0, 1],
            "__fraction_bars_above_50pct_peak__": [0.0, 0.5, 1.0],
            "__fraction_bars_above_80pct_peak__": [0.0, 0.25, 1.0],
            "__reaches_1_5atr_within_12h__": [0, 1, 1],
            "__pre_1_5_mfe_mae_ge_0_25atr__": [0, 1, 1],
            "__pre_1_5_mfe_mae_ge_0_50atr__": [0, 0, 1],
            "__pre_1_5_mfe_mae_ge_0_75atr__": [0, 0, 1],
            "__pre_1_5_mfe_mae_ge_1_00atr__": [0, 0, 1],
            "__pre_1_5_mfe_mae_ge_1_50atr__": [0, 0, 1],
            "__bars_to_confirmed_adverse_trough__": [np.nan, 2.0, 1.0],
            "__adverse_trough_within_60m__": [0, 0, 1],
            "__adverse_trough_within_120m__": [0, 1, 1],
            "__trough_before_1_5atr_mfe__": [0, 1, 1],
            "__reaches_1_5atr_before_trough_confirmation__": [0, 0, 1],
            "__bars_to_1_5atr__": [np.nan, 6.0, 1.0],
            "__path_efficiency_12h__": [0.0, 0.4, 0.9],
            "__path_efficiency_to_first_meaningful_mfe__": [0.0, 0.6, 1.0],
        }
    )


def test_auxiliary_sample_weights_are_head_specific_and_bounded():
    frame = _supportive_weight_frame()
    weights = {
        target: build_auxiliary_sample_weights(frame, target)
        for target in (
            "peak_mfe_12h_atr",
            "time_to_first_meaningful_mfe",
            "mae_before_meaningful_mfe_atr",
            "bars_before_price_stops_decreasing",
            "future_slope_atr_per_hour",
        )
    }
    for values in weights.values():
        assert values.dtype == np.float32
        assert np.all((values >= 0.5) & (values <= 2.0))
        assert values[2] > values[0]
    assert weights["time_to_first_meaningful_mfe"][0] == 0.5
    assert weights["mae_before_meaningful_mfe_atr"][0] == 0.5
    assert weights["future_slope_atr_per_hour"][0] == 0.5


def test_auxiliary_hpo_sample_is_15k_per_temporal_third():
    timestamps = pd.date_range("2025-01-01", periods=90_000, freq="min", tz="UTC")
    selected = auxiliary_hpo_sample_indices(timestamps, max_rows=45_000, random_state=7)
    assert len(selected) == 45_000
    assert len(np.unique(selected)) == 45_000
    assert [
        int(((selected >= lower) & (selected < upper)).sum())
        for lower, upper in ((0, 30_000), (30_000, 60_000), (60_000, 90_000))
    ] == [15_000, 15_000, 15_000]


def test_auxiliary_reference_split_is_strict_and_persists_bounds() -> None:
    ts = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T02:00:00Z",
            "2026-01-01T03:00:00Z",
        ]
    )
    resolved = pd.to_datetime(
        [
            "2026-01-01T00:30:00Z",
            "2026-01-01T02:00:00Z",
            "2026-01-01T02:30:00Z",
            "2026-01-01T03:30:00Z",
        ]
    )
    reference, oof, contract = auxiliary_reference_split(
        ts,
        resolved,
        selection_hpo_reference_end="2026-01-01T02:00:00Z",
    )
    np.testing.assert_array_equal(reference, [True, False, False, False])
    np.testing.assert_array_equal(oof, [False, False, True, True])
    assert (
        contract["reference_decision_bounds"]["max_utc"] == "2026-01-01T00:00:00+00:00"
    )
    assert contract["oof_decision_bounds"]["min_utc"] == "2026-01-01T02:00:00+00:00"
    assert contract["contract_sha256"]
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        auxiliary_reference_split(
            ts, resolved, selection_hpo_reference_end="2026-01-01 02:00:00"
        )


def test_expanding_folds_purge_target_horizon():
    ts = pd.date_range("2025-01-01", periods=4000, freq="h", tz="UTC")
    folds = expanding_purged_folds(
        ts, n_splits=3, purge_hours=12, min_train_rows=100, min_valid_rows=100
    )
    assert folds
    for fold in folds:
        assert ts[fold.train_idx].max() < fold.valid_start - pd.Timedelta(hours=12)


def test_monthly_oos_folds_train_only_on_labels_resolved_before_fold_start():
    timestamps = pd.date_range("2026-01-15", periods=2_000, freq="h", tz="UTC")
    resolved = timestamps + pd.Timedelta(hours=12)
    cutoff = pd.Timestamp("2026-02-01T00:00:00Z")
    folds = expanding_monthly_oos_folds(timestamps, resolved, oos_start=cutoff)

    assert len(folds) >= 2
    for fold in folds:
        assert (timestamps[fold.valid_idx] >= cutoff).all()
        assert (timestamps[fold.train_idx] < fold.valid_start).all()
        assert (resolved[fold.train_idx] < fold.valid_start).all()
        assert fold.valid_start <= timestamps[fold.valid_idx].min()


def test_auxiliary_objectives_are_better_for_exact_predictions():
    y_time = np.log1p(np.array([1.0, 2.0, 4.0, 8.0, 12.0]))
    exact, _ = auxiliary_hpo_objective("time_to_first_meaningful_mfe", y_time, y_time)
    poor, _ = auxiliary_hpo_objective(
        "time_to_first_meaningful_mfe", y_time, y_time[::-1]
    )
    assert exact > poor
    for target_name in (
        "mae_before_meaningful_mfe_atr",
        "bars_before_price_stops_decreasing",
        "future_slope_atr_per_hour",
    ):
        target = np.log1p(np.array([0.0, 0.5, 1.0, 2.0, 4.0]))
        exact, _ = auxiliary_hpo_objective(target_name, target, target)
        reversed_score, _ = auxiliary_hpo_objective(target_name, target, target[::-1])
        assert exact > reversed_score


def test_timing_objective_rewards_meaningful_mfe_horizon_accuracy():
    target = np.log1p(np.array([1.0, 2.0, 4.0, 8.0, 12.0]))
    exact, metrics = auxiliary_hpo_objective(
        "time_to_first_meaningful_mfe", target, target
    )
    delayed = np.log1p(np.full(len(target), 12.0))
    delayed_score, _ = auxiliary_hpo_objective(
        "time_to_first_meaningful_mfe", target, delayed
    )
    assert metrics["accuracy_meaningful_mfe_by_2h"] == 1.0
    assert exact > delayed_score
    y_mfe = np.log1p(np.array([0.1, 0.5, 1.0, 2.0, 4.0]))
    exact, _ = auxiliary_hpo_objective("peak_mfe_12h_atr", y_mfe, y_mfe)
    poor, _ = auxiliary_hpo_objective("peak_mfe_12h_atr", y_mfe, y_mfe[::-1])
    assert exact > poor


def test_configured_universe_contains_base_meta_and_generated_state_features():
    columns = [
        "ret1h",
        "mkt_rv_ratio",
        "gmm_entropy",
        "dae_b16_00",
        "expected_mahalanobis",
        "score",
        "base_margin_to_cutoff",
        "base_margin_to_cutoff_z",
        "base_signal_zscore_within_archetype",
        "__log1p_peak_mfe_atr_12h__",
        *ALL_SUPPORTIVE_LABEL_COLUMNS,
    ]
    selected, report = configured_auxiliary_feature_universe(columns)
    assert "ret1h" in selected
    assert "mkt_rv_ratio" in selected
    assert "gmm_entropy" in selected
    assert "dae_b16_00" in selected
    assert "expected_mahalanobis" in selected
    assert "score" in selected
    assert "base_margin_to_cutoff" in selected
    assert "base_margin_to_cutoff_z" in selected
    assert "base_signal_zscore_within_archetype" in selected
    assert "__log1p_peak_mfe_atr_12h__" not in selected
    assert not set(ALL_SUPPORTIVE_LABEL_COLUMNS).intersection(selected)
    assert report["contract"] == (
        "config_base_plus_meta_plus_frozen_ae_gmm_candidate_context_v2"
    )
    assert report["candidate_model_context_missing"] == []
    assert set(report["base_requested_by_side"]) == {"long", "short"}
    assert set(report["base_available_by_side"]) == {"long", "short"}
    assert set(report["base_missing_by_side"]) == {"long", "short"}
    assert set(report["meta_requested_by_head"]) == {
        "reg",
        "clf",
        "mfe",
        "mae",
        "asym",
    }
    assert set(report["meta_available_by_head"]) == set(
        report["meta_requested_by_head"]
    )
    assert set(report["meta_missing_by_head"]) == set(report["meta_requested_by_head"])
    assert report["configured_requested_count"] > 0


def test_base_archetype_labels_have_a_frozen_non_catboost_onehot_contract():
    frame = pd.DataFrame(
        {
            "canonical": ["trend", "reversal", "unseen"],
            "policy_archetype": ["fast", "slow", "fast"],
        }
    )
    contract = fit_base_archetype_label_feature_contract(
        frame.iloc[:2],
        source_columns=["canonical", "policy_archetype"],
        canonical_source="canonical",
    )
    encoded = transform_base_archetype_label_features(frame, contract)
    assert contract["canonical_features"]
    assert all(name.startswith("base_archetype_label__") for name in encoded)
    assert "CatBoost" in contract["inference_contract"]
    assert encoded.iloc[2][contract["canonical_features"]].sum() == 0.0
    selected, report = configured_auxiliary_feature_universe(encoded.columns)
    assert set(encoded.columns).issubset(selected)
    assert set(encoded.columns) == set(
        report["base_archetype_label_features_available"]
    )


def test_shared_pipeline_auxiliary_objective_uses_regression_error_and_ic():
    target = np.linspace(0.0, 2.0, 50)
    exact = _objective_value(
        _metric_pack(target, target, classifier=False), "auxiliary_regression"
    )
    reversed_score = _objective_value(
        _metric_pack(target, target[::-1], classifier=False),
        "auxiliary_regression",
    )
    assert exact > reversed_score


def test_auxiliary_univariate_direction_uses_continuous_target_not_positive_class():
    target = np.linspace(0.05, 2.0, 200, dtype=np.float32)
    feature = target**2
    pos, neg, direction, margin = _direction_score_for_feature(
        feature,
        target,
        classifier=False,
        returns=target,
        objective_mode="auxiliary_regression",
    )
    assert direction == 1
    assert pos > neg
    assert margin > 0.0


def test_auxiliary_univariate_skips_unused_topk_metric_pack(monkeypatch):
    target = np.linspace(0.05, 2.0, 200, dtype=np.float32)
    features = pd.DataFrame(
        {"aligned": target, "reversed": target[::-1]}, dtype=np.float32
    )

    def fail_metric_pack(*_args, **_kwargs):
        raise AssertionError(
            "auxiliary regression must not compute unused top-k metrics"
        )

    monkeypatch.setattr(lgbm_pipeline, "_metric_pack", fail_metric_pack)
    selected, stats = lgbm_pipeline._univariate_directional_filter(
        features,
        target,
        classifier=False,
        returns=target,
        random_state=7,
        objective_mode="auxiliary_regression",
    )
    assert selected
    assert set(stats["feature"]) == {"aligned", "reversed"}


def test_auxiliary_mda_uses_regression_error_and_rank_not_topk_precision():
    target = np.linspace(0.05, 2.0, 200, dtype=np.float32)
    cfg = {"objective": "auxiliary_regression", "use_sample_weight": True}
    exact = _topk_mda_score(target, target, sample_weight=np.ones(len(target)), cfg=cfg)
    reversed_score = _topk_mda_score(
        target, target[::-1], sample_weight=np.ones(len(target)), cfg=cfg
    )
    assert exact["score"] > reversed_score["score"]
    assert exact["regression_mae"] == 0.0


def test_auxiliary_selector_runs_full_pipeline_independently_per_side(
    monkeypatch,
):
    rows = 240
    seen = []

    def fake_train(X, y, **kwargs):
        side = str(kwargs["label_context"]["side_name"][0])
        seen.append(
            {
                "side": side,
                "rows": len(X),
                "cfg": kwargs["cfg"],
                "weights": kwargs["sample_weight"],
                "label_context": kwargs["label_context"],
            }
        )
        return {
            "metrics": {
                "per_side_feature_selection_selected_features": {
                    side: ["ret1h" if side == "long" else "score"],
                }
            }
        }

    monkeypatch.setattr(lgbm_pipeline, "LGBM_PER_SIDE_FEATURE_SELECTION", True)
    monkeypatch.setattr(lgbm_pipeline, "train_lgbm_stability_candidate", fake_train)
    X = pd.DataFrame(
        {
            "ret1h": np.linspace(-1.0, 1.0, rows, dtype=np.float32),
            "score": np.linspace(0.0, 1.0, rows, dtype=np.float32),
        }
    )
    target = np.linspace(0.0, 1.0, rows, dtype=np.float32)
    weights = np.linspace(0.5, 2.0, rows, dtype=np.float32)
    result = select_features_with_current_pipeline(
        X,
        target,
        timestamps=pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC"),
        assets=np.array(["BTC"] * rows),
        sides=np.where(np.arange(rows) % 2, "short", "long"),
        archetypes=np.array(["trend"] * rows),
        target_name="peak_mfe_12h_atr",
        sample_weight=weights,
    )
    assert [record["side"] for record in seen] == ["long", "short"]
    assert [record["rows"] for record in seen] == [120, 120]
    mda = seen[0]["cfg"]["mda_config"]
    assert seen[0]["cfg"]["archetype_univariate_prescreen_enabled"] is False
    assert seen[0]["cfg"]["archetype_relief_prescreen_enabled"] is False
    assert mda["archetype_univariate_prescreen_enabled"] is False
    assert mda["archetype_relief_prescreen_enabled"] is False
    assert mda["correlation_pruning_threshold"] == 0.88
    assert mda["correlation_threshold"] == 0.88
    assert mda["use_sample_weight"] is False
    np.testing.assert_array_equal(seen[0]["weights"], weights[::2])
    np.testing.assert_array_equal(seen[1]["weights"], weights[1::2])
    for record in seen:
        np.testing.assert_array_equal(
            record["label_context"]["side_mda_sample_weight"],
            np.ones(120, dtype=np.float32),
        )
    assert "training_loss_only" in result["sample_weight_contract"]
    assert result["prescreen_contract"].startswith("strict_side_local")


def test_hpo_keeps_weights_in_training_but_not_early_stopping_or_oof(monkeypatch):
    fit_calls: list[dict[str, object]] = []

    class FakeRegressor:
        def __init__(self, **_params):
            self.best_iteration_ = 17

        def fit(self, *_args, **kwargs):
            fit_calls.append(kwargs)
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=np.float32)

    class FakeTrial:
        number = 0

        def suggest_categorical(self, _name, values):
            return values[0]

        def suggest_float(self, _name, low, _high, **_kwargs):
            return low

        def suggest_int(self, _name, low, _high, **_kwargs):
            return low

        def report(self, *_args, **_kwargs):
            return None

        def should_prune(self):
            return False

    class FakeStudy:
        best_params = {
            "objective": "regression",
            "learning_rate": 0.01,
            "max_depth": 3,
            "num_leaves": 8,
            "min_child_samples": 100,
            "min_split_gain": 1e-4,
            "reg_alpha": 1e-3,
            "reg_lambda": 0.5,
            "subsample": 0.6,
            "colsample_bytree": 0.5,
            "max_bin": 63,
        }
        best_trial = FakeTrial()

        def optimize(self, objective, **_kwargs):
            self.best_value = objective(self.best_trial)
            self.trials = [self.best_trial]

    fake_optuna = SimpleNamespace(
        create_study=lambda **_kwargs: FakeStudy(),
        samplers=SimpleNamespace(TPESampler=lambda **_kwargs: object()),
        pruners=SimpleNamespace(MedianPruner=lambda **_kwargs: object()),
        TrialPruned=RuntimeError,
    )
    fake_lgb = SimpleNamespace(
        LGBMRegressor=FakeRegressor,
        early_stopping=lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setitem(sys.modules, "lightgbm", fake_lgb)
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna)

    rows = 1_400
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    cutoff = timestamps[1_000]
    target = np.linspace(0.1, 1.0, rows, dtype=np.float32)
    weights = np.linspace(0.5, 2.0, rows, dtype=np.float32)
    result = fit_hpo_oof_model(
        pd.DataFrame({"x": target}),
        target,
        selected_features=["x"],
        timestamps=timestamps,
        label_resolved_at=timestamps + pd.Timedelta(hours=1),
        selection_hpo_reference_end=cutoff,
        target_name="peak_mfe_12h_atr",
        sample_weight=weights,
        n_trials=1,
        hpo_rows=1_000,
    )

    assert fit_calls
    assert all("eval_sample_weight" not in call for call in fit_calls)
    assert all("sample_weight" in call for call in fit_calls)
    available = np.isfinite(result["oof_predictions"])
    assert (timestamps[available] >= cutoff).all()
    assert np.isnan(result["oof_predictions"][timestamps < cutoff]).all()
    assert "unweighted" in result["sample_weight_contract"]
    assert result["final_inference_fit_contract"]["rows"] == rows
    assert result["final_inference_fit_contract"]["model_sha256"]
    assert result["fold_metrics"]
    assert all(metric["oos_model_sha256"] for metric in result["fold_metrics"])
    assert all(
        metric["validation_weighted"] is False for metric in result["fold_metrics"]
    )
