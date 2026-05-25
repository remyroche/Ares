import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import extreme_price_movements.lgbm_pipeline as lp


def test_distillation_passes_apply_to_base_and_meta(monkeypatch):
    monkeypatch.setattr(lp, "LGBM_OOF_DISTILLATION_PASSES", 1)
    monkeypatch.setattr(lp, "LGBM_MIN_OOF_DISTILLATION_PASSES", 2)
    monkeypatch.setattr(lp, "LGBM_META_MIN_OOF_DISTILLATION_PASSES", 3)

    assert lp._distillation_passes_for_objective("train_base") == 2
    assert lp._distillation_passes_for_objective("train_meta") == 3

    monkeypatch.setattr(lp, "LGBM_OOF_DISTILLATION_PASSES", 4)
    assert lp._distillation_passes_for_objective("train_base") == 4
    assert lp._distillation_passes_for_objective("train_meta") == 4


def test_lgbm_hpo_does_not_search_extra_trees():
    source = Path(lp.__file__).read_text(encoding="utf-8")

    assert 'suggest_categorical("extra_trees"' not in source
    assert "suggest_categorical('extra_trees'" not in source


def test_lgbm_hpo_path_smooth_cap_is_10():
    assert 0.0 <= lp.LGBM_HPO_PATH_SMOOTH_MAX <= 10.0


def test_apply_overfit_gap_penalty_basic_no_gap_and_cap():
    out = lp._apply_overfit_gap_penalty(
        {"J_base": 0.80},
        {"J_base": 0.60},
        objective_mode="train_base",
        penalty=0.15,
        deadband=0.02,
        gap_cap=0.50,
    )
    assert out["J_overfit_gap"] == pytest.approx(0.18)
    assert out["J_overfit_penalty"] == pytest.approx(0.027)
    assert out["J_final"] == pytest.approx(0.573)
    assert out["J_base"] == pytest.approx(0.573)
    assert out["J_train"] == pytest.approx(0.80)
    assert out["J_valid_raw"] == pytest.approx(0.60)

    no_gap = lp._apply_overfit_gap_penalty(
        {"J_base": 0.50},
        {"J_base": 0.60},
        objective_mode="train_base",
        penalty=0.15,
        deadband=0.02,
        gap_cap=0.50,
    )
    assert no_gap["J_overfit_penalty"] == pytest.approx(0.0)
    assert no_gap["J_final"] == pytest.approx(0.60)

    capped = lp._apply_overfit_gap_penalty(
        {"J_base": 2.00},
        {"J_base": 0.00},
        objective_mode="train_base",
        penalty=0.15,
        deadband=0.02,
        gap_cap=0.50,
    )
    assert capped["J_overfit_gap"] == pytest.approx(0.50)
    assert capped["J_overfit_penalty"] == pytest.approx(0.075)
    assert capped["J_final"] == pytest.approx(-0.075)


def test_topk_oof_focus_weights_are_rank_aligned_and_constant_safe():
    pred = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    w = lp._topk_oof_focus_weights(pred, top_frac=0.4)
    assert float(w.sum()) == pytest.approx(1.0)
    assert np.all(np.diff(w) >= -1e-7)
    assert np.isfinite(1.0 / np.sum(w * w))

    constant = lp._topk_oof_focus_weights(np.ones(6, dtype=np.float32), top_frac=0.3)
    assert float(constant.sum()) == pytest.approx(1.0)
    assert np.all(np.isfinite(constant))


def test_normalize_importance_vector_handles_bad_and_zero_values():
    v = lp._normalize_importance_vector(
        np.asarray([-1.0, np.nan, np.inf, 3.0], dtype=np.float32),
        prior_strength=0.0,
    )
    assert float(v.sum()) == pytest.approx(1.0)
    assert np.all(v >= 0.0)
    assert np.argmax(v) == 3

    zero = lp._normalize_importance_vector(np.zeros(4, dtype=np.float32))
    assert float(zero.sum()) == pytest.approx(1.0)
    assert np.allclose(zero, np.full(4, 0.25))


def test_stratified_spread_subsample_spans_ordered_sample():
    y = np.asarray([0] * 60 + [1] * 60, dtype=np.int8)
    idx = lp._stratified_spread_subsample_indices(
        y,
        max_n=12,
        random_state=123,
        classifier=True,
    )
    assert len(idx) == 12
    assert idx[0] == 0
    assert idx[-1] == 119
    assert np.any((idx >= 40) & (idx <= 80))
    assert np.sum(y[idx] == 0) == 6
    assert np.sum(y[idx] == 1) == 6


def test_feature_selection_oi_mask_prefers_explicit_availability():
    X = pd.DataFrame(
        {
            "f0": [1.0, 2.0, 3.0, 4.0],
            "__open_interest_available__": [True, False, True, False],
            "open_interest": [np.nan, 99.0, np.nan, 42.0],
        }
    )
    mask, diagnostics, drop_cols = lp._feature_selection_oi_present_mask(X, len(X))
    assert mask.tolist() == [True, False, True, False]
    assert diagnostics["feature_selection_oi_filter_enforced"] is True
    assert diagnostics["feature_selection_oi_present_rows_total"] == 2
    assert "__open_interest_available__" in drop_cols


def test_train_meta_recent_coverage_exempts_model_derived_features(monkeypatch):
    n = 240
    timestamps = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    X = pd.DataFrame(
        {
            "raw_good": np.ones(n, dtype=np.float32),
            "raw_bad": np.where(np.arange(n) < 30, 1.0, np.nan).astype(np.float32),
            "feature_drift_psi_core": np.full(n, np.nan, dtype=np.float32),
            "variance_proxy": np.full(n, np.nan, dtype=np.float32),
            "leaf_count_p10": np.full(n, np.nan, dtype=np.float32),
            "contrib_entropy": np.full(n, np.nan, dtype=np.float32),
            "score_path_std": np.full(n, np.nan, dtype=np.float32),
            "base_prob_x_vol_regime": np.full(n, np.nan, dtype=np.float32),
            "base_model_score_pct": np.full(n, np.nan, dtype=np.float32),
            "oof_ebm_unc_demo": np.full(n, np.nan, dtype=np.float32),
            "pred_demo_H10_vote_entropy": np.full(n, np.nan, dtype=np.float32),
        }
    )
    monkeypatch.setattr(lp, "LGBM_FEATURE_RECENT_MIN_COVERAGE", 0.90)

    survivors, diagnostics = lp._recent_feature_coverage_survivors(
        X,
        timestamps,
        exempt_features={
            c for c in X.columns if lp._is_lgbm_model_derived_meta_feature(c)
        },
    )

    assert "raw_good" in survivors
    assert "raw_bad" not in survivors
    assert "feature_drift_psi_core" in survivors
    assert "variance_proxy" in survivors
    assert "leaf_count_p10" in survivors
    assert "contrib_entropy" in survivors
    assert "score_path_std" in survivors
    assert "base_prob_x_vol_regime" not in survivors
    assert "base_model_score_pct" not in survivors
    assert "oof_ebm_unc_demo" not in survivors
    assert "pred_demo_H10_vote_entropy" in survivors
    assert diagnostics["feature_recent_exempt_model_derived_count"] == 6
    assert diagnostics["feature_recent_joint_coverage"] == pytest.approx(1.0)


def test_bounded_importance_instability_properties():
    stable = np.tile(np.asarray([0.7, 0.2, 0.1], dtype=np.float32), (4, 1))
    stable_info = lp._bounded_importance_instability_from_matrix(stable)
    assert stable_info["instability"] == pytest.approx(0.0)

    alternating = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    unstable_info = lp._bounded_importance_instability_from_matrix(alternating)
    assert 0.0 <= unstable_info["instability"] <= 1.0
    assert unstable_info["instability"] > stable_info["instability"]

    low_noise = np.asarray(
        [
            [0.90, 0.09, 0.01, 0.00],
            [0.90, 0.09, 0.00, 0.01],
            [0.90, 0.09, 0.01, 0.00],
            [0.90, 0.09, 0.00, 0.01],
        ],
        dtype=np.float32,
    )
    low_noise_info = lp._bounded_importance_instability_from_matrix(
        low_noise,
        material_top_frac=0.5,
    )
    assert 0.0 <= low_noise_info["instability"] <= 1.0
    assert low_noise_info["instability"] < unstable_info["instability"]


def test_combined_gain_split_instability_weighting_and_disable(monkeypatch):
    gain_runs = [
        np.asarray([1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0], dtype=np.float32),
    ]
    split_runs = [
        np.asarray([0.6, 0.4], dtype=np.float32),
        np.asarray([0.6, 0.4], dtype=np.float32),
    ]
    monkeypatch.setattr(lp, "LGBM_IMPORTANCE_INSTABILITY_ENABLE", True)
    monkeypatch.setattr(lp, "LGBM_IMPORTANCE_INSTABILITY_GAIN_WEIGHT", 2.0)
    monkeypatch.setattr(lp, "LGBM_IMPORTANCE_INSTABILITY_SPLIT_WEIGHT", 1.0)
    info = lp._combined_gain_split_instability(gain_runs, split_runs)
    expected = (2.0 / 3.0) * info["gain_instability"] + (1.0 / 3.0) * info[
        "split_instability"
    ]
    assert info["importance_instability"] == pytest.approx(expected)

    monkeypatch.setattr(lp, "LGBM_IMPORTANCE_INSTABILITY_ENABLE", False)
    disabled = lp._combined_gain_split_instability(gain_runs, split_runs)
    assert disabled["importance_instability"] == 0.0
    assert disabled["gain_instability"] == 0.0
    assert disabled["split_instability"] == 0.0


def test_lgbm_candidate_smoke_includes_importance_penalty_metrics(monkeypatch):
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(42)
    n = 240
    X = pd.DataFrame(
        {
            f"f{i}": rng.normal(size=n).astype(np.float32)
            for i in range(8)
        }
    )
    logits = 1.2 * X["f0"].to_numpy() - 0.8 * X["f1"].to_numpy()
    y = (logits + 0.2 * rng.normal(size=n) > 0.0).astype(np.int8)
    returns = (0.01 * (2 * y - 1) + 0.002 * rng.normal(size=n)).astype(np.float32)

    monkeypatch.setattr(lp, "LGBM_RACE_MAX_ROWS", 220)
    monkeypatch.setattr(lp, "LGBM_MAX_ROUNDS", 1)
    monkeypatch.setattr(lp, "LGBM_MIN_FEATURES", 2)
    monkeypatch.setattr(lp, "LGBM_SELECTED_FEATURES_MIN", 4)
    monkeypatch.setattr(lp, "LGBM_SELECTED_FEATURES_MAX", 8)
    monkeypatch.setattr(lp, "LGBM_UNIVARIATE_MAX_ROWS", 120)
    monkeypatch.setattr(lp, "LGBM_RELIEF_REPEATS", 1)
    monkeypatch.setattr(lp, "LGBM_RELIEF_RESCUE_MIN", 2)
    monkeypatch.setattr(lp, "LGBM_RELIEF_RESCUE_MAX", 4)
    monkeypatch.setattr(lp, "LGBM_PERMUTATION_MAX_FEATURES", 4)
    monkeypatch.setattr(lp, "LGBM_FINAL_MODEL_COUNT", 1)
    monkeypatch.setattr(lp, "LGBM_OOF_DISTILLATION_PASSES", 0)
    monkeypatch.setattr(lp, "LGBM_MIN_OOF_DISTILLATION_PASSES", 0)
    monkeypatch.setattr(lp, "LGBM_META_MIN_OOF_DISTILLATION_PASSES", 0)
    monkeypatch.setattr(lp, "LGBM_HPO_TRIALS", 1)
    monkeypatch.setattr(lp, "LGBM_HPO_EARLY_STOP_PATIENCE", 1)

    candidate = lp.train_lgbm_stability_candidate(
        X,
        y,
        np.ones(n, dtype=np.float32),
        random_state=123,
        mode="classifier",
        returns=returns,
    )
    assert candidate is not None
    metrics = candidate["metrics"]
    for key in (
        "importance_instability",
        "gain_instability",
        "split_instability",
        "importance_instability_penalty",
        "J_final_pre_importance_instability_penalty",
        "topk_contrib_available_rate",
        "topk_focus_effective_rows_mean",
    ):
        assert key in metrics

    _, hpo_metrics = lp._run_lgbm_hpo(
        X,
        y,
        np.ones(n, dtype=np.float32),
        list(X.columns[:4]),
        classifier=True,
        returns=returns,
        metric_y=y,
        random_state=321,
        max_trials=1,
        patience=1,
        objective_mode="train_base",
    )
    for key in (
        "importance_instability",
        "gain_instability",
        "split_instability",
        "importance_instability_penalty",
        "J_final_pre_importance_instability_penalty",
        "topk_contrib_available_rate",
        "topk_focus_effective_rows_mean",
    ):
        assert key in hpo_metrics
