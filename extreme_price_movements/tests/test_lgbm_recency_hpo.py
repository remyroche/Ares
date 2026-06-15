import json

import numpy as np
import pandas as pd

from extreme_price_movements.lgbm_recency_hpo import (
    active_recency_hpo_config,
    composite_decay_from_timestamps,
    final_selection_score,
    precision_score_top_fracs,
    recency_hpo_grid,
    recency_hpo_train_oos_masks,
)
from extreme_price_movements.label_weight_optuna import LabelWeightRecipe, apply_weight_recipe
from extreme_price_movements import lgbm_pipeline as lp


def test_composite_decay_formula_uses_uniform_floor():
    ts = pd.to_datetime(["2026-01-01", "2026-01-31"], utc=True)
    decay = composite_decay_from_timestamps(
        ts,
        2,
        half_life_days=30.0,
        composite_weight=0.3,
    )

    assert np.allclose(decay, np.array([0.65, 1.0], dtype=np.float32), atol=1e-6)


def test_recency_hpo_grid_matches_base_and_meta_contracts():
    base = recency_hpo_grid("base")
    meta = recency_hpo_grid("meta")

    assert len(base) == 9
    assert len(meta) == 9
    assert sorted({row["half_life_months"] for row in base}) == [6.0, 9.0, 12.0]
    assert sorted({row["half_life_months"] for row in meta}) == [3.0, 4.5, 6.0]
    assert sorted({row["composite_weight"] for row in base}) == [0.3, 0.4, 0.5]


def test_recency_hpo_grid_can_use_explicit_pairs(monkeypatch):
    monkeypatch.setenv("EPM_RECENCY_HPO_BASE_GRID_PAIRS", "9:0.4,12:0.3,9:0.3")

    grid = recency_hpo_grid("base")

    assert [
        (row["half_life_months"], row["composite_weight"])
        for row in grid
    ] == [(9.0, 0.4), (12.0, 0.3), (9.0, 0.3)]


def test_train_oos_masks_use_three_years_and_last_two_months():
    ts = pd.date_range("2022-01-01", "2026-06-01", freq="MS", tz="UTC")
    train_mask, oos_mask, meta = recency_hpo_train_oos_masks(
        ts,
        train_years=3,
        holdout_months=2,
    )

    selected_train = ts[train_mask]
    selected_oos = ts[oos_mask]
    assert selected_train.min() == pd.Timestamp("2023-04-01", tz="UTC")
    assert selected_train.max() == pd.Timestamp("2026-03-01", tz="UTC")
    assert selected_oos.min() == pd.Timestamp("2026-04-01", tz="UTC")
    assert selected_oos.max() == pd.Timestamp("2026-06-01", tz="UTC")
    assert meta["train_rows"] == len(selected_train)
    assert meta["oos_rows"] == len(selected_oos)


def test_precision_selection_score_uses_top_10_20_30_and_last_windows():
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1], dtype=np.float32)
    score = np.arange(10, dtype=np.float32)
    precision = precision_score_top_fracs(y, score)

    assert precision["p_at_10"] == 1.0
    assert precision["p_at_20"] == 1.0
    assert np.isclose(precision["p_at_30"], 2.0 / 3.0)
    assert np.isclose(precision["precision_score"], 0.25 + 0.50 + 2.0 / 3.0)

    ts = pd.date_range("2026-01-01", periods=10, freq="7D", tz="UTC")
    final = final_selection_score(y, score, ts)
    assert "precision_last_4w" in final
    assert "precision_last_8w" in final
    assert np.isfinite(final["final_selection_score"])


def test_active_recency_hpo_config_loads_saved_winner(tmp_path, monkeypatch):
    winner_path = tmp_path / "base_winner.json"
    winner_path.write_text(
        json.dumps(
            {
                "winner": {
                    "half_life_months": 9.0,
                    "half_life_days": 273.75,
                    "composite_weight": 0.4,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EPM_RECENCY_HPO_BASE_WINNER_PATH", str(winner_path))

    active = active_recency_hpo_config({}, "train_base")

    assert active is not None
    assert active["scope"] == "base"
    assert active["half_life_days"] == 273.75
    assert active["composite_weight"] == 0.4


def test_label_weight_recipe_skips_legacy_recency_when_recency_hpo_active(tmp_path):
    recipe = LabelWeightRecipe()
    recipe.weight.weight_modifier_strength = 1.0
    recipe.weight.class_rebalance_strength = 0.0
    recipe.weight.concurrency_penalty = 0.0
    recipe.weight.mfe_weight_power = 0.0
    recipe.weight.mae_weight_power = 0.0
    recipe.weight.net_ev_weight_power = 0.0
    recipe.weight.hard_negative_weight = 1.0
    recipe.weight.ambiguous_weight = 1.0
    recipe.weight.robustness_strength = 0.0
    recipe.weight.path_quality_strength = 0.0
    recipe.weight.recency_half_life_days = 1.0
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe.to_dict()), encoding="utf-8")
    df = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03"],
                utc=True,
            ),
            "__mfe_ret__": [0.0, 0.0, 0.0],
            "__mae_ret__": [0.0, 0.0, 0.0],
            "__barrier_pct__": [0.01, 0.01, 0.01],
        }
    )

    out, stats = apply_weight_recipe(
        df,
        np.array([0, 1, 0], dtype=np.float32),
        np.array([0.2, 0.8, 0.2], dtype=np.float32),
        np.ones(3, dtype=np.float32),
        cfg={
            "label_weight_recipe": str(recipe_path),
            "recency_hpo_half_life_days": 30.0,
            "recency_hpo_composite_weight": 0.4,
        },
        stage="train_base",
        label="x",
    )

    assert stats["legacy_recency_disabled_by_recency_hpo"] is True
    assert np.allclose(out, np.ones(3, dtype=np.float32))


def test_recency_hpo_confirms_top_candidates_with_self_distillation(monkeypatch):
    fit_calls = []
    distill_calls = []

    def fake_fit_lgbm_model(X, y, sample_weight, *, classifier, params, **_kwargs):
        fit_calls.append({"rows": len(X), "params": dict(params)})
        return {
            "mean": float(np.average(np.asarray(y, dtype=np.float32), weights=sample_weight)),
            "seed": int(params.get("random_state", 0)),
        }

    def fake_predict_lgbm_raw(model, X, mode):
        base = float(model["mean"])
        ramp = np.linspace(0.0, 0.05, len(X), dtype=np.float32)
        return np.clip(base + ramp, 0.0, 1.0).astype(np.float32)

    def fake_oof_distilled_weights(
        X,
        y,
        base_weight,
        features,
        *,
        cfg,
        objective_mode,
        timestamps,
        **_kwargs,
    ):
        decay, active = lp.recency_hpo_decay_from_config(
            timestamps,
            len(y),
            cfg=cfg,
            objective_mode=objective_mode,
        )
        assert decay is not None
        assert active is not None
        distill_calls.append(
            {
                "features": list(features),
                "half_life_days": float(active["half_life_days"]),
                "composite_weight": float(active["composite_weight"]),
            }
        )
        return np.asarray(base_weight, dtype=np.float32), np.full(len(y), 0.5, dtype=np.float32)

    monkeypatch.setattr(lp, "_fit_lgbm_model", fake_fit_lgbm_model)
    monkeypatch.setattr(lp, "_predict_lgbm_raw", fake_predict_lgbm_raw)
    monkeypatch.setattr(lp, "_oof_distilled_sample_weights_lgbm", fake_oof_distilled_weights)
    monkeypatch.setattr(lp, "LGBM_FINAL_MODEL_COUNT", 2)

    timestamps = pd.date_range("2022-01-01", periods=60, freq="MS", tz="UTC")
    X = pd.DataFrame({"f0": np.linspace(0.0, 1.0, len(timestamps), dtype=np.float32)})
    y = (np.arange(len(timestamps)) % 3 == 0).astype(np.float32)

    payload = lp.run_lgbm_recency_hpo_fixed_contract(
        X,
        y,
        sample_weight=None,
        selected_features=["f0"],
        best_params={"n_estimators": 8, "learning_rate": 0.05},
        timestamps=timestamps,
        hard_labels=y,
        cfg={
            "recency_hpo_base_half_life_months": [6.0, 9.0],
            "recency_hpo_composite_weights": [0.3],
            "recency_hpo_min_train_rows": 1,
            "recency_hpo_min_oos_rows": 1,
            "recency_hpo_confirmation_top_n": 2,
            "recency_hpo_require_distillation_confirmation": True,
        },
        scope_key="long_demo_H5",
        persist_winner=False,
    )

    confirmation = payload["distillation_confirmation"]
    assert confirmation["enabled"] is True
    assert confirmation["status"] == "confirmed"
    assert confirmation["winner_confirmed_best_among_confirmed"] is True
    assert len(confirmation["candidates"]) == 2
    assert len(distill_calls) == 2
    assert all(row["final_ensemble_count"] == 2 for row in confirmation["candidates"])
    assert all(row["final_ensemble_sequential_distillation"] is True for row in confirmation["candidates"])
    assert payload["winner"]["distillation_confirmation_status"] == "confirmed"
    assert np.isfinite(payload["winner"]["distillation_confirmation_score"])
    assert len(fit_calls) == 6


def test_recency_hpo_confirmation_can_select_best_confirmed_candidate(monkeypatch):
    def fake_fit_lgbm_model(X, y, sample_weight, *, classifier, params, **_kwargs):
        return {"mean": float(np.mean(y))}

    def fake_predict_lgbm_raw(model, X, mode):
        return np.linspace(0.0, 1.0, len(X), dtype=np.float32)

    def fake_confirmation(candidate, **_kwargs):
        trial = int(candidate["trial"])
        score = {1: 0.25, 2: 0.50}[trial]
        return {
            "enabled": True,
            "status": "confirmed",
            "trial": trial,
            "scope": candidate["scope"],
            "scope_key": candidate["scope_key"],
            "half_life_months": float(candidate["half_life_months"]),
            "half_life_days": float(candidate["half_life_days"]),
            "composite_weight": float(candidate["composite_weight"]),
            "no_distillation_final_selection_score": float(candidate["final_selection_score"]),
            "final_selection_score": score,
            "score_delta_vs_no_distillation": score - float(candidate["final_selection_score"]),
        }

    monkeypatch.setattr(lp, "_fit_lgbm_model", fake_fit_lgbm_model)
    monkeypatch.setattr(lp, "_predict_lgbm_raw", fake_predict_lgbm_raw)
    monkeypatch.setattr(lp, "_recency_hpo_distillation_confirmation", fake_confirmation)

    timestamps = pd.date_range("2022-01-01", periods=60, freq="MS", tz="UTC")
    X = pd.DataFrame({"f0": np.linspace(0.0, 1.0, len(timestamps), dtype=np.float32)})
    y = (np.arange(len(timestamps)) % 3 == 0).astype(np.float32)

    payload = lp.run_lgbm_recency_hpo_fixed_contract(
        X,
        y,
        sample_weight=None,
        selected_features=["f0"],
        best_params={"n_estimators": 8, "learning_rate": 0.05},
        timestamps=timestamps,
        hard_labels=y,
        cfg={
            "recency_hpo_base_grid_pairs": "6:0.3,9:0.3",
            "recency_hpo_min_train_rows": 1,
            "recency_hpo_min_oos_rows": 1,
            "recency_hpo_confirmation_top_n": 2,
            "recency_hpo_require_distillation_confirmation": True,
        },
        scope_key="long_demo_H5",
        persist_winner=False,
    )

    confirmation = payload["distillation_confirmation"]
    assert confirmation["status"] == "confirmed_reranked_winner"
    assert confirmation["raw_winner_trial"] == 1
    assert confirmation["selected_trial"] == 2
    assert confirmation["winner_trial"] == 2
    assert confirmation["winner_distillation_score"] == 0.50
    assert confirmation["raw_winner_distillation_score"] == 0.25
    assert payload["winner"]["trial"] == 2
    assert payload["winner"]["final_selection_score"] == 0.50
    assert payload["winner"]["distillation_confirmation_score"] == 0.50
