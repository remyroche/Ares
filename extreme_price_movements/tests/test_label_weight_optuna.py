import json

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.label_weight_optuna import (
    DISABLED_RECIPE_KEY,
    LabelWeightRecipe,
    ObjectiveParams,
    _enqueue_previous_best,
    _make_optuna_pruner,
    apply_distillation_recipe,
    apply_label_recipe,
    apply_weight_recipe,
    load_recipe_from_env_or_cfg,
    objective_score,
    recipe_path_from_env_or_cfg,
    suggest_optuna_params,
)


def _write_recipe(path, recipe=None):
    payload = (recipe or LabelWeightRecipe()).to_dict()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _path_df(n=4):
    return pd.DataFrame(
        {
            "__mfe_ret__": np.linspace(0.001, 0.004, n),
            "__mae_ret__": -np.linspace(0.0002, 0.001, n),
            "__barrier_pct__": np.full(n, 0.01),
            "__bars_to_mfe__": np.arange(1, n + 1),
            "__bars_to_mae__": np.arange(n, 0, -1),
            "__symbol__": ["A", "A", "B", "C"][:n],
        }
    )


def test_recipe_hooks_reject_length_mismatch(tmp_path, monkeypatch):
    recipe_path = _write_recipe(tmp_path / "recipe.json")
    monkeypatch.setenv("EPM_LABEL_WEIGHT_RECIPE", recipe_path)

    with pytest.raises(ValueError, match="apply_label_recipe length mismatch"):
        apply_label_recipe(
            _path_df(3),
            np.array([0, 1]),
            np.array([0.1, 0.9, 0.2]),
            stage="train_base",
            label="x",
        )

    with pytest.raises(ValueError, match="apply_weight_recipe length mismatch"):
        apply_weight_recipe(
            _path_df(3),
            np.array([0, 1, 0]),
            np.array([0.1, 0.9]),
            np.ones(3),
            stage="train_base",
            label="x",
        )


def test_weight_recipe_normalizes_against_fit_mask_only(tmp_path, monkeypatch):
    recipe = LabelWeightRecipe()
    recipe.weight.class_rebalance_strength = 0.0
    recipe.weight.recency_half_life_days = 0.0
    recipe.weight.concurrency_penalty = 0.0
    recipe.weight.mfe_weight_power = 1.0
    recipe.weight.mae_weight_power = 0.0
    recipe.weight.net_ev_weight_power = 0.0
    recipe_path = _write_recipe(tmp_path / "recipe.json", recipe)
    monkeypatch.setenv("EPM_LABEL_WEIGHT_RECIPE", recipe_path)

    fit_mask = np.array([True, True, False, False])
    out, stats = apply_weight_recipe(
        _path_df(4),
        np.array([0, 1, 0, 1]),
        np.array([0.2, 0.8, 0.3, 0.7]),
        np.ones(4),
        stage="train_base",
        label="x",
        fit_mask=fit_mask,
    )

    assert stats["fit_rows"] == 2
    assert np.isclose(float(np.mean(out[fit_mask])), 1.0, atol=1e-6)
    assert not np.isclose(float(np.mean(out)), 1.0, atol=1e-3)


def test_distillation_recipe_can_load_from_cfg(tmp_path, monkeypatch):
    monkeypatch.delenv("EPM_LABEL_WEIGHT_RECIPE", raising=False)
    recipe = LabelWeightRecipe()
    recipe.distillation.distill_error_power = 2.0
    recipe.distillation.false_positive_focus = 0.0
    recipe.distillation.false_negative_focus = 0.0
    recipe.distillation.distill_age_impact = 0.0
    recipe_path = _write_recipe(tmp_path / "recipe.json", recipe)

    dist, fp = apply_distillation_recipe(
        np.ones(4),
        np.ones(4),
        y_metric=np.array([0.0, 1.0, 0.0, 1.0]),
        pred=np.array([0.9, 0.8, 0.2, 0.1]),
        objective_mode="train_base",
        cfg={"label_weight_recipe": recipe_path},
    )

    assert np.isfinite(dist).all()
    assert np.isfinite(fp).all()
    assert not np.allclose(dist, np.ones(4))


def test_economic_distillation_focuses_high_rank_losses_and_missed_winners(tmp_path, monkeypatch):
    monkeypatch.delenv("EPM_LABEL_WEIGHT_RECIPE", raising=False)
    recipe = LabelWeightRecipe()
    recipe.distillation.distill_error_power = 0.0
    recipe.distillation.false_positive_focus = 1.0
    recipe.distillation.false_negative_focus = 1.0
    recipe.distillation.distill_age_impact = 0.0
    recipe.distillation.economic_error_mix = 1.0
    recipe.distillation.distill_net_loss_power = 1.0
    recipe.distillation.distill_stop_hit_focus = 1.0
    recipe.distillation.distill_missed_net_power = 1.0
    recipe.distillation.distill_rank_focus_threshold = 0.70
    recipe.distillation.distill_rank_focus_temperature = 0.05
    recipe.objective.impact_scale_bps = 50.0
    recipe.geometry.min_executable_net_bps = 10.0
    recipe.execution_costs = {"execution_cost_bps": 0.0}
    recipe_path = _write_recipe(tmp_path / "recipe.json", recipe)

    dist, fp = apply_distillation_recipe(
        np.ones(4),
        np.ones(4),
        y_metric=np.array([0.9, 0.1, 0.9, 0.1]),
        pred=np.array([0.9, 0.85, 0.05, 0.2]),
        returns=np.array([0.004, -0.004, 0.006, -0.0001]),
        objective_mode="train_base",
        cfg={"label_weight_recipe": recipe_path},
    )

    assert np.isfinite(dist).all()
    assert np.isfinite(fp).all()
    assert fp[1] > fp[3]
    assert fp[2] > fp[3]


def test_objective_params_affect_score():
    metrics = {
        "net_hit_at_10": 0.6,
        "net_hit_at_20": 0.55,
        "net_hit_at_40": 0.5,
        "avg_win_net_bps_at_10": 150.0,
        "avg_win_net_bps_at_20": 150.0,
        "mean_net_bps_at_10": 90.0,
        "mean_net_bps_at_20": 50.0,
        "mean_net_bps_at_30": 25.0,
        "median_net_bps_at_20": 25.0,
    }
    default = objective_score(metrics, objective=ObjectiveParams())
    stricter = objective_score(
        {**metrics, "mean_net_bps_at_20": -1.0},
        objective=ObjectiveParams(min_mean_net_bps_at_20=0.0),
    )
    assert stricter < default


def test_objective_prefers_broad_top30_net_ev():
    narrow = {
        "mean_net_bps_at_10": 120.0,
        "mean_net_bps_at_20": 20.0,
        "mean_net_bps_at_30": -10.0,
        "stop_hit_rate_at_20": 0.10,
        "avg_stop_loss_bps_at_20": 40.0,
    }
    broad = {
        "mean_net_bps_at_10": 60.0,
        "mean_net_bps_at_20": 50.0,
        "mean_net_bps_at_30": 45.0,
        "stop_hit_rate_at_20": 0.10,
        "avg_stop_loss_bps_at_20": 40.0,
    }

    assert objective_score(broad, objective=ObjectiveParams(min_mean_net_bps_at_20=-100.0)) > objective_score(
        narrow,
        objective=ObjectiveParams(min_mean_net_bps_at_20=-100.0),
    )


def test_objective_prefers_broad_top50_net_ev():
    narrow = {
        "mean_net_bps_at_10": 120.0,
        "mean_net_bps_at_20": 70.0,
        "mean_net_bps_at_30": 20.0,
        "mean_net_bps_at_50": -20.0,
        "stop_hit_rate_at_20": 0.10,
        "avg_stop_loss_bps_at_20": 40.0,
    }
    broad = {
        "mean_net_bps_at_10": 50.0,
        "mean_net_bps_at_20": 45.0,
        "mean_net_bps_at_30": 42.0,
        "mean_net_bps_at_50": 38.0,
        "stop_hit_rate_at_20": 0.10,
        "avg_stop_loss_bps_at_20": 40.0,
    }

    objective = ObjectiveParams(min_mean_net_bps_at_20=-100.0)

    assert objective_score(broad, objective=objective) > objective_score(narrow, objective=objective)


def test_objective_rejects_stop_heavy_trials():
    metrics = {
        "net_hit_at_10": 0.8,
        "net_hit_at_20": 0.7,
        "avg_win_net_bps_at_10": 250.0,
        "avg_win_net_bps_at_20": 250.0,
        "mean_net_bps_at_20": 100.0,
        "median_net_bps_at_20": 80.0,
        "stop_hit_rate_at_20": 0.70,
        "avg_stop_loss_bps_at_20": 80.0,
    }
    score = objective_score(metrics, objective=ObjectiveParams(max_stop_hit_rate_at_20=0.35))

    assert score < -10.0


def test_objective_uses_lower_confidence_bound_edge():
    base_metrics = {
        "mean_net_bps_at_10": 50.0,
        "mean_net_bps_at_20": 50.0,
        "mean_net_bps_at_30": 50.0,
        "stop_hit_rate_at_20": 0.10,
        "avg_stop_loss_bps_at_20": 40.0,
    }
    stable = {
        **base_metrics,
        "per_window": {
            "8w": {"20": {"mean_net_bps": 50.0}},
            "4w": {"20": {"mean_net_bps": 50.0}},
            "2w": {"20": {"mean_net_bps": 50.0}},
        },
    }
    noisy = {
        **base_metrics,
        "per_window": {
            "8w": {"20": {"mean_net_bps": -50.0}},
            "4w": {"20": {"mean_net_bps": 50.0}},
            "2w": {"20": {"mean_net_bps": 150.0}},
        },
    }
    objective = ObjectiveParams(
        min_window_mean_net_bps_at_20=-100.0,
        max_stop_hit_rate_at_20=0.90,
        max_window_stop_hit_rate_at_20=0.90,
    )

    assert objective_score(noisy, objective=objective) < objective_score(stable, objective=objective)


def test_objective_penalizes_bad_temporal_window():
    good = {
        "mean_net_bps_at_10": 60.0,
        "mean_net_bps_at_20": 50.0,
        "mean_net_bps_at_30": 40.0,
        "stop_hit_rate_at_20": 0.10,
        "avg_stop_loss_bps_at_20": 40.0,
        "per_window": {
            "8w": {"20": {"mean_net_bps": 30.0, "stop_hit_rate": 0.10}},
            "4w": {"20": {"mean_net_bps": 50.0, "stop_hit_rate": 0.10}},
            "2w": {"20": {"mean_net_bps": 70.0, "stop_hit_rate": 0.10}},
        },
    }
    bad_window = {
        **good,
        "per_window": {
            "8w": {"20": {"mean_net_bps": -30.0, "stop_hit_rate": 0.10}},
            "4w": {"20": {"mean_net_bps": 80.0, "stop_hit_rate": 0.10}},
            "2w": {"20": {"mean_net_bps": 100.0, "stop_hit_rate": 0.10}},
        },
    }

    assert objective_score(bad_window, objective=ObjectiveParams()) < objective_score(
        good,
        objective=ObjectiveParams(),
    )


def test_objective_penalizes_topk_concentration():
    diversified = {
        "mean_net_bps_at_10": 40.0,
        "mean_net_bps_at_20": 40.0,
        "mean_net_bps_at_30": 40.0,
        "mean_net_bps_at_50": 40.0,
        "stop_hit_rate_at_20": 0.10,
        "avg_stop_loss_bps_at_20": 40.0,
        "symbol_concentration_hhi_at_20": 0.08,
        "week_concentration_hhi_at_20": 0.12,
        "unique_symbols_at_20": 12.0,
    }
    concentrated = {
        **diversified,
        "symbol_concentration_hhi_at_20": 0.50,
        "week_concentration_hhi_at_20": 0.55,
        "unique_symbols_at_20": 2.0,
    }

    objective = ObjectiveParams(min_mean_net_bps_at_20=-100.0)

    assert objective_score(concentrated, objective=objective) < objective_score(diversified, objective=objective)


def test_previous_best_recipe_is_enqueued(tmp_path):
    recipe = LabelWeightRecipe()
    recipe.weight.positive_mass_target = 0.37
    best_path = tmp_path / "best_recipe.json"
    best_path.write_text(json.dumps(recipe.to_dict()), encoding="utf-8")

    class FakeStudy:
        def __init__(self):
            self.enqueued = []

        def enqueue_trial(self, params, skip_if_exists=False):
            del skip_if_exists
            self.enqueued.append(params)

    study = FakeStudy()
    assert _enqueue_previous_best(study, best_path, phase="weights")
    assert study.enqueued[0]["positive_mass_target"] == 0.37


def test_pruner_factory_defaults_to_successive_halving():
    optuna = pytest.importorskip("optuna")
    pruner = _make_optuna_pruner(optuna, "successive_halving")
    assert pruner.__class__.__name__ == "SuccessiveHalvingPruner"


def test_lgbm_estimator_cap_is_enforced(monkeypatch):
    import importlib
    import extreme_price_movements.lgbm_pipeline as lgbm_pipeline

    monkeypatch.setenv("EPM_LGBM_N_ESTIMATORS_CAP", "25")
    reloaded = importlib.reload(lgbm_pipeline)
    params = reloaded._effective_lgbm_params({"n_estimators": 100}, classifier=True)
    assert params["n_estimators"] == 25


def test_lgbm_noop_recipe_uses_default_recency_half_life(tmp_path, monkeypatch):
    import importlib
    import extreme_price_movements.lgbm_pipeline as lgbm_pipeline

    recipe = LabelWeightRecipe()
    recipe.stage = "__label_weight_noop__"
    recipe.weight.recency_half_life_days = 123.0
    recipe_path = _write_recipe(tmp_path / "noop_recipe.json", recipe)
    monkeypatch.setenv("EPM_LABEL_WEIGHT_RECIPE", recipe_path)
    monkeypatch.setenv("EPM_LGBM_BASE_RECENCY_HALF_LIFE_DAYS", "365")
    monkeypatch.setenv("EPM_LGBM_META_RECENCY_HALF_LIFE_DAYS", "182.5")

    reloaded = importlib.reload(lgbm_pipeline)

    assert reloaded._recency_half_life_days("train_base") == 365.0
    assert reloaded._recency_half_life_days("train_meta") == 182.5


def test_suggest_optuna_params_preserves_base_recipe_fields():
    class FakeTrial:
        number = 7

        def suggest_float(self, name, low, high, log=False):
            del low, high, log
            values = {
                "economic_error_mix": 0.5,
                "distill_net_loss_power": 1.1,
                "distill_stop_hit_focus": 0.7,
                "distill_missed_net_power": 1.2,
                "distill_rank_focus_threshold": 0.8,
                "distill_rank_focus_temperature": 0.05,
                "positive_mass_target": 0.41,
                "class_rebalance_strength": 0.3,
                "mfe_weight_power": 0.4,
                "mae_weight_power": 0.5,
                "net_ev_weight_power": 0.6,
                "hard_negative_weight": 1.4,
                "ambiguous_weight": 0.7,
                "concurrency_penalty": 0.2,
                "robustness_strength": 0.8,
                "path_quality_strength": 0.6,
                "portfolio_alignment_strength": 0.4,
            }
            return values[name]

        def suggest_categorical(self, name, choices):
            del choices
            if name == "recipe_enabled":
                return True
            values = {
                "recency_half_life_days": 150.0,
                "concurrency_window_hours": 1.0,
            }
            return values[name]

    base = LabelWeightRecipe()
    base.label.net_return_center_bps = 37.0
    recipe = suggest_optuna_params(FakeTrial(), phase="weights", base_recipe=base)

    assert recipe.label.net_return_center_bps == 37.0
    assert recipe.weight.positive_mass_target == 0.41
    assert recipe.weight.robustness_strength == 0.8
    assert recipe.weight.path_quality_strength == 0.6
    assert recipe.objective.portfolio_alignment_strength == 0.4


def test_suggest_optuna_params_can_enable_label_geometry():
    class FakeTrial:
        number = 8

        def suggest_float(self, name, low, high, log=False):
            del low, high, log
            values = {
                "tp_vol_mult": 1.1,
                "sl_as_tp_pct": 0.8,
                "timeout_value": 0.3,
                "trailing_activation_vol_mult": 1.2,
                "trailing_giveback_pct": 0.5,
                "min_executable_net_bps": 25.0,
                "mae_failure_vol_mult": 1.0,
                "geometry_anchor_mix": 0.6,
            }
            return values[name]

        def suggest_categorical(self, name, choices):
            del choices
            if name == "recipe_enabled":
                return True
            assert name == "label_horizon_bars"
            return 5.0

    recipe = suggest_optuna_params(FakeTrial(), phase="label_geometry")

    assert recipe.geometry.enabled is True
    assert recipe.geometry.tp_vol_mult == 1.1
    assert recipe.geometry.label_horizon_bars == 5.0


def test_suggest_optuna_params_noop_preserves_base_recipe():
    class FakeTrial:
        number = 9

        def suggest_categorical(self, name, choices):
            del choices
            assert name == "recipe_enabled"
            return False

    base = LabelWeightRecipe()
    base.geometry.enabled = True
    base.geometry.tp_vol_mult = 1.7
    base.label.net_return_center_bps = 55.0

    recipe = suggest_optuna_params(FakeTrial(), phase="labels", base_recipe=base)

    assert recipe.geometry.enabled is True
    assert recipe.geometry.tp_vol_mult == 1.7
    assert recipe.label.net_return_center_bps == 55.0
    assert recipe.provenance["noop_meaning"] == "fixed_base_recipe_unchanged"


def test_suggest_optuna_params_noop_without_base_disables_recipe_hooks():
    class FakeTrial:
        number = 10

        def suggest_categorical(self, name, choices):
            del choices
            assert name == "recipe_enabled"
            return False

    recipe = suggest_optuna_params(FakeTrial(), phase="label_geometry")

    assert recipe.stage == "__label_weight_noop__"
    assert recipe.provenance["noop_meaning"] == "pre_hpo_neutral_baseline_no_recipe_transforms"


def test_robustness_and_path_quality_downweight_rows(tmp_path, monkeypatch):
    recipe = LabelWeightRecipe()
    recipe.weight.class_rebalance_strength = 0.0
    recipe.weight.recency_half_life_days = 0.0
    recipe.weight.concurrency_penalty = 0.0
    recipe.weight.mfe_weight_power = 0.0
    recipe.weight.mae_weight_power = 0.0
    recipe.weight.net_ev_weight_power = 0.0
    recipe.weight.robustness_strength = 1.0
    recipe.weight.path_quality_strength = 1.0
    recipe_path = _write_recipe(tmp_path / "recipe.json", recipe)
    monkeypatch.setenv("EPM_LABEL_WEIGHT_RECIPE", recipe_path)

    df = _path_df(4)
    out, stats = apply_weight_recipe(
        df,
        np.array([0, 1, 0, 1]),
        np.array([0.2, 0.8, 0.3, 0.7]),
        np.ones(4),
        stage="train_base",
        label="x",
    )

    assert np.isfinite(out).all()
    assert stats["robustness_strength"] == 1.0
    assert stats["path_quality_strength"] == 1.0
    assert stats["uncertainty_mean"] > 0.0
    assert stats["path_timing_penalty_mean"] > 0.0
    assert stats["quick_mfe_profit_mean"] > 0.0


def test_path_quality_favours_fast_large_mfe(tmp_path, monkeypatch):
    recipe = LabelWeightRecipe()
    recipe.weight.class_rebalance_strength = 0.0
    recipe.weight.recency_half_life_days = 0.0
    recipe.weight.concurrency_penalty = 0.0
    recipe.weight.mfe_weight_power = 0.0
    recipe.weight.mae_weight_power = 0.0
    recipe.weight.net_ev_weight_power = 0.0
    recipe.weight.hard_negative_weight = 1.0
    recipe.weight.ambiguous_weight = 1.0
    recipe.weight.path_quality_strength = 1.0
    recipe_path = _write_recipe(tmp_path / "recipe.json", recipe)
    monkeypatch.setenv("EPM_LABEL_WEIGHT_RECIPE", recipe_path)

    df = pd.DataFrame(
        {
            "__mfe_ret__": [0.006, 0.001],
            "__mae_ret__": [-0.0001, -0.0001],
            "__barrier_pct__": [0.002, 0.002],
            "__bars_to_mfe__": [1, 20],
            "__bars_to_mae__": [20, 1],
        }
    )
    out, stats = apply_weight_recipe(
        df,
        np.array([1, 1]),
        np.array([0.8, 0.8]),
        np.ones(2),
        stage="train_base",
        label="x",
    )

    assert out[0] > out[1]
    assert stats["quick_mfe_profit_mean"] > 0.0


def test_label_geometry_changes_soft_anchor_and_weight_hard_negatives(tmp_path, monkeypatch):
    recipe = LabelWeightRecipe()
    recipe.geometry.enabled = True
    recipe.geometry.tp_vol_mult = 1.0
    recipe.geometry.sl_as_tp_pct = 0.75
    recipe.geometry.label_horizon_bars = 5.0
    recipe.geometry.min_executable_net_bps = 10.0
    recipe.geometry.geometry_anchor_mix = 0.75
    recipe.weight.class_rebalance_strength = 0.0
    recipe.weight.recency_half_life_days = 0.0
    recipe.weight.concurrency_penalty = 0.0
    recipe.weight.mfe_weight_power = 0.0
    recipe.weight.mae_weight_power = 0.0
    recipe.weight.net_ev_weight_power = 0.0
    recipe.weight.ambiguous_weight = 1.0
    recipe.weight.hard_negative_weight = 2.0
    recipe_path = _write_recipe(tmp_path / "recipe.json", recipe)
    monkeypatch.setenv("EPM_LABEL_WEIGHT_RECIPE", recipe_path)
    monkeypatch.setenv("EPM_EXECUTION_AWARE_COST_BPS", "0")

    df = pd.DataFrame(
        {
            "__mfe_ret__": [0.004, 0.0005],
            "__mae_ret__": [-0.0001, -0.004],
            "__barrier_pct__": [0.002, 0.002],
            "__bars_to_mfe__": [1, 10],
            "__bars_to_mae__": [10, 1],
        }
    )
    soft, label_stats = apply_label_recipe(
        df,
        np.array([1, 1]),
        np.array([0.5, 0.5]),
        stage="train_base",
        label="x",
    )
    weights, weight_stats = apply_weight_recipe(
        df,
        np.array([1, 1]),
        soft,
        np.ones(2),
        stage="train_base",
        label="x",
    )

    assert label_stats["geometry_enabled"] is True
    assert weight_stats["geometry_enabled"] is True
    assert soft[0] > soft[1]
    assert weights[1] > weights[0]


def test_stop_path_rows_are_soft_label_capped(tmp_path, monkeypatch):
    recipe = LabelWeightRecipe()
    recipe.geometry.enabled = False
    recipe.label.mae_penalty_scale = 1.0
    recipe.label.net_return_center_bps = 0.0
    recipe.label.net_return_temperature_bps = 20.0
    recipe.label.stop_penalty = 0.75
    recipe.label.max_stop_soft_label = 0.20
    recipe.label.max_bad_path_soft_label = 0.35
    recipe.execution_costs = {"execution_cost_bps": 0.0}
    recipe_path = _write_recipe(tmp_path / "recipe.json", recipe)
    monkeypatch.setenv("EPM_LABEL_WEIGHT_RECIPE", recipe_path)

    df = pd.DataFrame(
        {
            "__mfe_ret__": [0.05, 0.05],
            "__mae_ret__": [-0.001, -0.001],
            "__barrier_pct__": [0.002, 0.002],
            "__bars_to_mfe__": [1, 1],
            "__bars_to_mae__": [10, 10],
            "__y_outcome__": [0, 2],
        }
    )
    soft, stats = apply_label_recipe(
        df,
        np.array([1, 1]),
        np.array([0.9, 0.9]),
        stage="train_base",
        label="x",
    )

    assert soft[0] <= 0.20
    assert soft[1] > soft[0]
    assert stats["actual_stop_rate"] == 0.5


def test_portfolio_alignment_affects_objective_score():
    metrics = {
        "net_hit_at_10": 0.5,
        "net_hit_at_20": 0.5,
        "net_hit_at_40": 0.5,
        "mean_net_bps_at_20": 20.0,
        "median_net_bps_at_20": 10.0,
        "portfolio_sharpe": 2.0,
        "portfolio_sortino": 2.5,
        "max_drawdown_bps": 50.0,
        "symbol_concentration_hhi": 0.1,
    }
    base = objective_score(metrics, objective=ObjectiveParams(portfolio_alignment_strength=0.0))
    aligned = objective_score(metrics, objective=ObjectiveParams(portfolio_alignment_strength=0.5))

    assert aligned > base


def test_best_recipe_default_can_be_bypassed_to_hardcoded_values(tmp_path, monkeypatch):
    monkeypatch.delenv("EPM_LABEL_WEIGHT_RECIPE", raising=False)
    monkeypatch.setenv("EPM_LABEL_WEIGHT_BYPASS_BEST_DEFAULT", "1")
    best_recipe = LabelWeightRecipe()
    best_recipe.weight.positive_mass_target = 0.37
    best_path = tmp_path / "best_recipe.json"
    best_path.write_text(json.dumps(best_recipe.to_dict()), encoding="utf-8")

    recipe = load_recipe_from_env_or_cfg({"label_weight_best_recipe_path": str(best_path)})

    assert recipe is not None
    assert recipe.name == "hardcoded_default"
    assert recipe.weight.positive_mass_target == LabelWeightRecipe().weight.positive_mass_target


def test_best_recipe_default_is_used_when_not_bypassed(tmp_path, monkeypatch):
    monkeypatch.delenv("EPM_LABEL_WEIGHT_RECIPE", raising=False)
    monkeypatch.delenv("EPM_LABEL_WEIGHT_BYPASS_BEST_DEFAULT", raising=False)
    monkeypatch.delenv("EPM_LABEL_WEIGHT_USE_BEST_DEFAULT", raising=False)
    monkeypatch.delenv("EPM_LABEL_WEIGHT_DISABLE", raising=False)
    best_recipe = LabelWeightRecipe()
    best_recipe.weight.positive_mass_target = 0.37
    best_path = tmp_path / "best_recipe.json"
    best_path.write_text(json.dumps(best_recipe.to_dict()), encoding="utf-8")

    assert recipe_path_from_env_or_cfg({"label_weight_best_recipe_path": str(best_path)}) == str(best_path)
    recipe = load_recipe_from_env_or_cfg({"label_weight_best_recipe_path": str(best_path)})

    assert recipe is not None
    assert recipe.weight.positive_mass_target == 0.37


def test_scoped_recipe_overrides_global_recipe(tmp_path, monkeypatch):
    monkeypatch.delenv("EPM_LABEL_WEIGHT_DISABLE", raising=False)
    monkeypatch.setenv("EPM_LABEL_WEIGHT_USE_BEST_DEFAULT", "0")
    global_recipe = LabelWeightRecipe(name="global")
    global_recipe.weight.positive_mass_target = 0.37
    base_recipe = LabelWeightRecipe(name="base")
    base_recipe.weight.positive_mass_target = 0.41
    meta_recipe = LabelWeightRecipe(name="meta")
    meta_recipe.weight.positive_mass_target = 0.52
    global_path = _write_recipe(tmp_path / "global.json", global_recipe)
    base_path = _write_recipe(tmp_path / "base.json", base_recipe)
    meta_path = _write_recipe(tmp_path / "meta.json", meta_recipe)
    monkeypatch.setenv("EPM_LABEL_WEIGHT_RECIPE", global_path)
    monkeypatch.setenv("EPM_LABEL_WEIGHT_BASE_RECIPE", base_path)
    monkeypatch.setenv("EPM_LABEL_WEIGHT_META_RECIPE", meta_path)

    assert load_recipe_from_env_or_cfg(scope="base").name == "base"
    assert load_recipe_from_env_or_cfg(scope="meta").name == "meta"
    assert load_recipe_from_env_or_cfg().name == "global"


def test_scoped_disable_only_disables_that_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("EPM_LABEL_WEIGHT_USE_BEST_DEFAULT", "0")
    recipe = LabelWeightRecipe(name="active")
    recipe_path = _write_recipe(tmp_path / "recipe.json", recipe)
    monkeypatch.setenv("EPM_LABEL_WEIGHT_RECIPE", recipe_path)
    monkeypatch.setenv("EPM_LABEL_WEIGHT_META_DISABLE", "1")

    assert load_recipe_from_env_or_cfg(scope="meta") is None
    assert load_recipe_from_env_or_cfg(scope="base").name == "active"


def test_label_weight_disable_ignores_persisted_best(tmp_path, monkeypatch):
    monkeypatch.delenv("EPM_LABEL_WEIGHT_RECIPE", raising=False)
    monkeypatch.setenv("EPM_LABEL_WEIGHT_DISABLE", "1")
    best_recipe = LabelWeightRecipe()
    best_recipe.weight.positive_mass_target = 0.37
    best_path = tmp_path / "best_recipe.json"
    best_path.write_text(json.dumps(best_recipe.to_dict()), encoding="utf-8")

    assert recipe_path_from_env_or_cfg({"label_weight_best_recipe_path": str(best_path)}) == DISABLED_RECIPE_KEY
    assert load_recipe_from_env_or_cfg({"label_weight_best_recipe_path": str(best_path)}) is None


def test_label_weight_disable_is_exact_passthrough(tmp_path, monkeypatch):
    recipe = LabelWeightRecipe()
    recipe.weight.path_quality_strength = 1.0
    recipe.weight.robustness_strength = 1.0
    recipe.distillation.distill_error_power = 2.0
    recipe_path = _write_recipe(tmp_path / "recipe.json", recipe)
    monkeypatch.setenv("EPM_LABEL_WEIGHT_RECIPE", recipe_path)
    monkeypatch.setenv("EPM_LABEL_WEIGHT_DISABLE", "1")

    df = _path_df(4)
    y_hard = np.array([0, 1, 0, 1], dtype=np.float32)
    y_soft = np.array([0.2, 0.8, 0.3, 0.7], dtype=np.float32)
    weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    soft_out, soft_stats = apply_label_recipe(
        df,
        y_hard,
        y_soft,
        stage="train_base",
        label="x",
    )
    weight_out, weight_stats = apply_weight_recipe(
        df,
        y_hard,
        y_soft,
        weights,
        stage="train_base",
        label="x",
    )
    dist_out, fp_out = apply_distillation_recipe(
        np.ones(4, dtype=np.float32),
        np.ones(4, dtype=np.float32) * 2.0,
        y_metric=y_soft,
        pred=np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32),
        objective_mode="train_base",
    )

    assert soft_stats == {"enabled": False, "reason": "no_recipe"}
    assert weight_stats == {"enabled": False, "reason": "no_recipe"}
    np.testing.assert_array_equal(soft_out, y_soft)
    np.testing.assert_array_equal(weight_out, weights)
    np.testing.assert_array_equal(dist_out, np.ones(4, dtype=np.float32))
    np.testing.assert_array_equal(fp_out, np.ones(4, dtype=np.float32) * 2.0)
