from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.regime_adaptor import (
    apply_regime_adaptor,
    build_regime_feature_frame,
)
from extreme_price_movements.simple_policy_optimiser import (
    _attach_regime_model_context_columns,
    _path_extrema_from_policy_paths,
)


def test_simple_policy_path_extrema_produces_trust_positive_candidates():
    rows = pd.DataFrame(
        {
            "return": [0.018, 0.015, -0.006],
            "side": [1, -1, 1],
        }
    )
    opens = np.asarray(
        [[100.0, 101.0, 103.0], [100.0, 99.0, 97.5], [100.0, 99.0, 98.0]],
        dtype=np.float32,
    )
    highs = np.asarray(
        [[101.0, 103.0, 102.0], [100.5, 101.0, 100.2], [100.2, 100.1, 99.8]],
        dtype=np.float32,
    )
    lows = np.asarray(
        [[99.5, 100.5, 101.0], [99.0, 97.5, 98.0], [99.0, 98.5, 98.0]],
        dtype=np.float32,
    )
    closes = opens.copy()

    mfe, mae, t_mfe, t_mae = _path_extrema_from_policy_paths(
        rows, (opens, highs, lows, closes)
    )
    ratio = mfe / (mae + 1e-9)

    assert np.isfinite(mfe).all()
    assert np.isfinite(mae).all()
    assert np.isfinite(t_mfe).all()
    assert np.isfinite(t_mae).all()
    assert ratio[0] > 1.2
    assert ratio[1] > 1.2
    assert int(np.sum((rows["return"].to_numpy() > 0.01) & (ratio > 1.2))) == 2


def test_retired_regime_adaptor_artifacts_are_not_applied():
    pred = np.array([0.2, 0.6, 0.9], dtype=float)
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    for artifact in (
        {"schema_version": "v1", "enable_regime_adaptor": True},
        {"schema_version": "rolling_bad_regime_v2", "enable_regime_adaptor": True},
    ):
        applied = apply_regime_adaptor(frame, pred, artifact)
        assert np.allclose(applied["deployment_score_pre_rank"], pred)
        assert not np.any(applied["regime_adjustment_enabled"])
        assert applied["regime_disabled_reason"][0] == "unsupported_retired_regime_adaptor"


def test_regime_adaptor_spread_baseline_features_use_50_bins(monkeypatch, tmp_path):
    from extreme_price_movements import regime_adaptor as ra

    baseline_path = tmp_path / "per_asset_spread_baseline_latest.csv"
    baseline = pd.DataFrame(
        {
            "symbol": [f"S{i:02d}/USD:USD" for i in range(50)],
            "rows": np.ones(50, dtype=int),
            "average_spread_bps": np.arange(1, 51, dtype=float),
            "p75_spread_bps": np.arange(2, 102, 2, dtype=float),
        }
    )
    baseline.to_csv(baseline_path, index=False)
    monkeypatch.setenv("EPM_SIMPLE_POLICY_SPREAD_BASELINE_PATH", str(baseline_path))
    ra._REGIME_SPREAD_BASELINE_CACHE.clear()

    symbols = np.asarray(["S00/USD:USD", "S25/USD:USD", "S49/USD:USD"], dtype=object)
    frame = pd.DataFrame(
        {
            "symbol": symbols,
            "barrier_pct": [0.01, 0.01, 0.01],
        }
    )
    ts = pd.date_range("2026-01-01", periods=len(frame), freq="h", tz="UTC")

    regime, mapping = build_regime_feature_frame(frame, ts, symbols)

    assert regime["asset_spread_decile"].tolist() == pytest.approx([0.0, 25.0, 49.0])
    assert regime["asset_p75_spread_bps"].tolist() == pytest.approx([2.0, 52.0, 100.0])
    assert regime["asset_p75_spread_decile"].tolist() == pytest.approx(
        [0.0, 25.0, 49.0]
    )
    assert regime["spread_to_expected_move"].tolist() == pytest.approx(
        [0.01, 0.26, 0.50]
    )
    assert "50_bins" in str(mapping["asset_spread_decile"])
    assert "50_bins" in str(mapping["asset_p75_spread_decile"])


def test_error_archetype_weak_clusters_are_report_only():
    from extreme_price_movements import regime_adaptor as ra

    contrib = np.zeros((200, 4), dtype=np.float32)
    y_bad = np.tile([0, 1], 100).astype(np.int8)

    spec = ra._fit_error_archetype_spec(
        contrib,
        y_bad,
        contribution_features=[f"f{i}" for i in range(contrib.shape[1])],
    )

    assert spec["enabled"] is False
    assert spec["reason"] == "weak_cluster_separation"
    assert spec["bad_cluster_ids"] == []
    assert spec["good_cluster_ids"] == []
    assert spec["role_diagnostics"]["role_signal_ok"] is False


def test_error_risk_family_access_preserves_drift_uncertainty_and_archetype():
    from extreme_price_movements import regime_adaptor as ra

    ranked_rows = [
        {"feature": f"plain_signal_{i}", "feature_selection_score": 1.0 - i * 0.01}
        for i in range(ra.ERROR_RISK_MAX_FEATURES)
    ] + [
        {"feature": "feature_drift_psi_core", "feature_selection_score": 0.10},
        {"feature": "pred_std_norm", "feature_selection_score": 0.09},
        {"feature": "base_error_archetype_is_bad", "feature_selection_score": 0.08},
    ]
    selected = [str(row["feature"]) for row in ranked_rows[: ra.ERROR_RISK_MAX_FEATURES]]

    out = ra._ensure_error_risk_family_access(selected, ranked_rows)

    assert len(out) <= ra.ERROR_RISK_MAX_FEATURES
    assert "feature_drift_psi_core" in out
    assert "pred_std_norm" in out
    assert "base_error_archetype_is_bad" in out


def test_meta_lgbm_predictive_atlas_fit_and_apply_features_are_finite():
    from extreme_price_movements import regime_adaptor as ra

    n = 80
    timestamps = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    scores = np.r_[np.linspace(0.20, 0.45, n // 2), np.linspace(0.65, 0.95, n // 2)]
    pnl = np.r_[
        np.where(np.arange(n // 2) % 4 == 0, 0.01, -0.01),
        np.where(np.arange(n // 2) % 4 == 0, -0.01, 0.01),
    ]
    candidate_frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": np.where(np.arange(n) % 2 == 0, "AAA/USD:USD", "BBB/USD:USD"),
            "normalized_rank_score": pd.Series(scores).rank(pct=True).to_numpy(),
            "calibrated_score": scores,
            "net_return": pnl,
        }
    )
    base = pd.DataFrame({"meta_score_for_correctness": scores})

    out, state, report = ra._append_meta_lgbm_predictive_atlas_features(
        base,
        candidate_frame,
        scores,
        pnl=pnl,
        timestamps=timestamps,
        fit=True,
        min_support=5,
    )

    assert state["enabled"] is True
    assert report["enabled"] is True
    assert set(ra.META_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS) <= set(out.columns)
    assert np.isfinite(out[list(ra.META_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS)].to_numpy()).all()
    assert out["meta_lgbm_predictive_atlas_hit_rate_surprise_z"].abs().max() <= 8.0

    live_frame = pd.DataFrame(
        {
            "normalized_rank_score": [0.95, 0.05],
            "calibrated_score": [0.90, 0.20],
        }
    )
    live_base = pd.DataFrame(index=live_frame.index)
    live, _, live_report = ra._append_meta_lgbm_predictive_atlas_features(
        live_base,
        live_frame,
        [0.90, 0.20],
        fit=False,
        state=state,
        min_support=5,
    )

    assert live_report["enabled"] is True
    assert set(ra.META_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS) <= set(live.columns)
    assert np.isfinite(live[list(ra.META_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS)].to_numpy()).all()


def test_pooled_walk_forward_split_is_by_time_and_keeps_assets_together():
    from extreme_price_movements.regime_adaptor import _walk_forward_splits

    anchors = np.repeat(pd.date_range("2025-01-01", periods=30, freq="D", tz="UTC"), 2)
    symbols = np.tile(["AAA", "BBB"], 30)
    splits = _walk_forward_splits(anchors, len(anchors), n_splits=5)
    assert splits
    for train_idx, valid_idx in splits:
        assert pd.Series(anchors[train_idx]).max() < pd.Series(anchors[valid_idx]).min()
        valid_frame = pd.DataFrame(
            {"anchor": anchors[valid_idx], "symbol": symbols[valid_idx]}
        )
        for _, group in valid_frame.groupby("anchor"):
            assert set(group["symbol"]) == {"AAA", "BBB"}

def test_positive_boost_uplift_diagnostics_drive_churn_penalty():
    from extreme_price_movements import regime_adaptor as ra

    baseline = np.array(
        [0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72]
    )
    candidate = np.array(
        [0.90, 0.88, 0.85, 0.87, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72]
    )
    returns = np.array(
        [1.0, 0.9, -1.0, 2.0, 0.1, 0.0, -0.2, 0.1, 0.0, 0.0]
    )
    combined = {
        "good_regime_offset": np.array(
            [0.0, 0.0, 0.0, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        "score_delta_from_regime_adjustment": candidate - baseline,
    }

    diagnostics = ra._regime_positive_boost_uplift_diagnostics(
        baseline, candidate, returns, combined, top_frac=0.30
    )
    penalty = ra._positive_boost_churn_penalty(diagnostics)

    assert diagnostics["rows_boosted"] == 1
    assert diagnostics["mean_boost"] == pytest.approx(0.03)
    assert diagnostics["boosted_pnl_contribution"] == pytest.approx(2.0)
    assert diagnostics["top30_turnover_delta"] == pytest.approx(1.0 / 3.0)
    assert diagnostics["top30_promoted_by_boost_rows"] == 1
    assert diagnostics["top30_positive_boost_promoted_share"] == pytest.approx(
        1.0 / 3.0
    )
    assert diagnostics["top30_promoted_by_boost_pnl_contribution"] == pytest.approx(
        2.0
    )
    assert penalty["positive_boost_churn_penalty"] > 0.0
    assert penalty["positive_boost_churn_penalty_multiplier"] < 1.0


def test_meta_correctness_features_exclude_bad_regime_probabilities():
    from extreme_price_movements import regime_adaptor as ra

    frame = pd.DataFrame(
        {
            "meta_pred_calibrated": [0.2, 0.8],
            "base_model_pred": [0.3, 0.7],
            "feature_drift_psi_core": [0.1, 0.2],
            "raw_state_svd_00": [0.4, 0.5],
            "shap_archetype_is_bad": [0.0, 1.0],
            "asset_rv_mean_24h": [1.2, 1.3],
            "recent_meta_ece_5d": [0.04, 0.06],
            "recent_meta_brier_10d": [0.12, 0.14],
            "recent_meta_global_top15_hit_rate_5d": [0.6, 0.4],
            "prior_3d_expected_hit_rate": [0.55, 0.45],
            "prior_3d_hit_rate_surprise_z": [1.0, -1.0],
            "raw_state_reconstruction_error": [0.02, 0.03],
            "p_bad_regime_asset_3d": [0.9, 0.1],
            "p_bad_regime_asset_5d": [0.8, 0.2],
            "ebm_unc_logodds_var": [0.4, 0.5],
            "oof_ebm_unc_entropy_mean": [0.2, 0.3],
            "global_ebm_unc_dispersion_mean_7d": [0.7, 0.8],
            "future_horizon_wallet_pnl": [1.0, -1.0],
            "wallet_return": [0.01, -0.02],
        }
    )
    x, _ = ra._build_meta_correctness_feature_frame(
        frame,
        frame["meta_pred_calibrated"],
        pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
        ["AAA", "BBB"],
    )

    assert "p_bad_regime_asset_3d" not in x.columns
    assert "p_bad_regime_asset_5d" not in x.columns
    assert "ebm_unc_logodds_var" not in x.columns
    assert "oof_ebm_unc_entropy_mean" not in x.columns
    assert "global_ebm_unc_dispersion_mean_7d" not in x.columns
    assert "future_horizon_wallet_pnl" not in x.columns
    assert "wallet_return" not in x.columns
    for col in (
        "meta_score_for_correctness",
        "base_model_pred",
        "feature_drift_psi_core",
        "raw_state_svd_00",
        "shap_archetype_is_bad",
        "asset_rv_mean_24h",
        "recent_meta_ece_5d",
        "recent_meta_brier_10d",
        "recent_meta_global_top15_hit_rate_5d",
        "prior_3d_expected_hit_rate",
        "prior_3d_hit_rate_surprise_z",
        "raw_state_reconstruction_error",
        "regime_hour_sin",
    ):
        assert col in x.columns


def test_meta_correctness_soft_labels_are_pnl_and_conviction_aware():
    from extreme_price_movements import regime_adaptor as ra

    meta = np.array([0.95, 0.95, 0.52, 0.52], dtype=float)
    pnl = np.array([0.05, -0.05, 0.05, -0.05], dtype=float)

    y_soft, y_hard, weights, diag = ra._meta_correctness_soft_labels_and_weights(
        meta, pnl
    )

    assert y_hard.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert y_soft[0] > y_soft[2] > 0.5
    assert y_soft[1] < y_soft[3] < 0.5
    assert weights[1] > 0.0
    assert diag["hard_positive_rate"] == pytest.approx(0.5)
    assert diag["soft_label_center"] == pytest.approx(0.5)
    assert diag["conviction_floor"] == pytest.approx(0.25)
    assert diag["sample_weight_clip_min"] == pytest.approx(0.25)
    assert diag["sample_weight_clip_max"] == pytest.approx(5.0)


def test_fit_meta_correctness_uses_soft_labels_for_lgbm(monkeypatch):
    from extreme_price_movements import lgbm_pipeline as lgbm_pipe
    from extreme_price_movements import regime_adaptor as ra

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC"),
            "symbol": ["AAA", "BBB"] * 4,
            "calibrated_score": [0.92, 0.88, 0.76, 0.66, 0.55, 0.44, 0.33, 0.22],
            "net_return": [0.05, -0.04, 0.03, -0.02, 0.01, -0.01, 0.02, -0.03],
            "feature_drift_psi_core": np.linspace(0.1, 0.8, 8),
            "asset_rv_mean_24h": np.linspace(1.0, 1.7, 8),
        }
    )
    expected_soft, expected_hard, _, _ = ra._meta_correctness_soft_labels_and_weights(
        frame["calibrated_score"].to_numpy(dtype=float),
        frame["net_return"].to_numpy(dtype=float),
    )
    captured = {}

    class FakeRawModel:
        def __init__(self, pred):
            self.pred = np.asarray(pred, dtype=float)

        def predict(self, x, **_kwargs):
            return np.resize(self.pred, len(x))

    def fake_candidate(X, y, **kwargs):
        captured["candidate_y"] = np.asarray(y, dtype=float)
        captured["candidate_hard_labels"] = np.asarray(
            kwargs["hard_labels"], dtype=float
        )
        captured["candidate_cfg"] = kwargs.get("cfg", {})
        return {
            "selected_features_from_cv": list(X.columns[:2]),
            "selected_feature_names": list(X.columns[:2]),
            "oof_probs": np.clip(expected_soft.astype(float), 1e-4, 1.0 - 1e-4),
            "metrics": {},
            "pruning_history": {},
            "stage_indices": {
                "fit_oof": np.asarray([0, 1, 2, 3], dtype=np.int32),
                "lgbm_select": np.asarray([4, 5], dtype=np.int32),
                "hpo": np.asarray([6, 7], dtype=np.int32),
            },
        }

    def fake_fit_lgbm_model(X, y, sample_weight, **kwargs):
        captured.setdefault("fit_calls", []).append(
            {
                "y": np.asarray(y, dtype=float),
                "sample_weight": np.asarray(sample_weight, dtype=float),
                "params": dict(kwargs.get("params", {})),
            }
        )
        return FakeRawModel(np.clip(np.asarray(y, dtype=float), 1e-4, 1.0 - 1e-4))

    def fail_full_model(*_args, **_kwargs):
        raise AssertionError("old full-fit API should not be used")

    def fake_feature_refine(
        _lgbm_pipe,
        _X_fit,
        _y_soft_fit,
        _y_hard_fit,
        _sw_model,
        _pnl_fit,
        _meta_fit,
        _ts_fit,
        selected_features,
        _feature_stats,
        selection_idx,
        **_kwargs,
    ):
        selected = list(selected_features)
        captured["feature_refine_called"] = True
        captured["feature_refine_input_features"] = selected
        captured["feature_refine_selection_rows"] = int(len(selection_idx))
        return {
            "enabled": True,
            "reason": "unit_refine",
            "selected_features": selected[:1],
            "input_feature_count": len(selected),
            "selected_feature_count": 1,
            "best_value": 1.23,
            "objective": "unit_regime_pnl_refine",
        }

    monkeypatch.setattr(ra, "META_CORRECTNESS_MIN_ROWS", 4)
    monkeypatch.setattr(lgbm_pipe, "train_lgbm_stability_candidate", fake_candidate)
    monkeypatch.setattr(lgbm_pipe, "_fit_lgbm_model", fake_fit_lgbm_model)
    monkeypatch.setattr(
        ra,
        "_refine_meta_correctness_features_for_regime_pnl",
        fake_feature_refine,
    )
    monkeypatch.setattr(
        lgbm_pipe,
        "_feature_importances",
        lambda _model, n: (np.ones(n, dtype=float), np.ones(n, dtype=float)),
    )
    monkeypatch.setattr(lgbm_pipe, "fit_lgbm_stability_full_model", fail_full_model)

    artifact = ra.fit_meta_correctness_regime_adaptor(
        frame,
        strategy_id="unit",
        optuna_trials=0,
        cfg={"regime_adaptor_weight_hpo_enable": False},
    )

    assert artifact["return_column"] == "net_return"
    assert np.allclose(captured["candidate_y"], expected_soft)
    assert np.allclose(captured["fit_calls"][0]["y"], expected_soft[:4])
    assert np.allclose(captured["fit_calls"][1]["y"], expected_soft)
    assert np.allclose(captured["candidate_hard_labels"], expected_hard)
    assert not np.all(np.isin(np.unique(captured["fit_calls"][1]["y"]), [0.0, 1.0]))
    assert captured["candidate_cfg"]["lgbm_hpo_overrides"]["max_depth_max"] == 4
    assert captured["candidate_cfg"]["lgbm_hpo_overrides"][
        "min_child_samples_pct_min"
    ] == pytest.approx(0.03)
    assert captured["feature_refine_called"] is True
    assert captured["feature_refine_selection_rows"] == 2
    assert artifact["feature_refine"]["enabled"] is True
    assert artifact["feature_refine"]["selected_feature_count"] == 1
    assert "J_regime_pnl feature refinement" in artifact["training_scheme"]
    assert "top 5/10/20%" in artifact["feature_selection_policy"]
    assert artifact["self_distillation_policy"] == "disabled_for_regime_adaptor"
    assert artifact["stage_row_counts"]["fit_oof_train"] == 4
    assert artifact["stage_row_counts"]["validation_lgbm_select_plus_hpo"] == 4


def test_meta_correctness_weight_formula_params_change_weights():
    from extreme_price_movements import regime_adaptor as ra

    meta = np.array([0.95, 0.80, 0.55, 0.40, 0.20], dtype=float)
    pnl = np.array([0.05, -0.03, 0.01, -0.02, 0.04], dtype=float)
    y_soft, _, default_weights, _ = ra._meta_correctness_soft_labels_and_weights(
        meta,
        pnl,
    )
    tuned_weights, diag = ra._meta_correctness_weight_formula(
        meta,
        pnl,
        y_soft,
        weight_params={
            "rank_power": 0.75,
            "conviction_multiplier": 0.10,
            "pnl_mag_multiplier": 2.50,
            "label_distance_multiplier": 0.20,
        },
    )

    assert diag["sample_weight_rank_power"] == pytest.approx(0.75)
    assert diag["sample_weight_pnl_mag_multiplier"] == pytest.approx(2.50)
    assert not np.allclose(default_weights, tuned_weights)
    assert np.isclose(float(np.mean(tuned_weights)), 1.0, atol=0.25)


def test_meta_correctness_recent_scope_keeps_last_two_calendar_months():
    from extreme_price_movements import regime_adaptor as ra

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-02-14T23:00:00Z",
                    "2026-02-15T00:00:00Z",
                    "2026-03-15T00:00:00Z",
                    "2026-04-15T00:00:00Z",
                ]
            ),
            "value": [1, 2, 3, 4, 5],
        }
    )

    scoped, metadata = ra._limit_meta_correctness_to_recent_scope(frame, months=2)

    assert scoped["value"].tolist() == [3, 4, 5]
    assert metadata["enabled"] is True
    assert metadata["timestamp_column"] == "timestamp"
    assert metadata["input_rows"] == 5
    assert metadata["output_rows"] == 3
    assert metadata["dropped_rows"] == 2
    assert metadata["cutoff_ts"].startswith("2026-02-15T00:00:00")


def test_meta_correctness_recent_scope_requires_timestamp():
    from extreme_price_movements import regime_adaptor as ra

    scoped, metadata = ra._limit_meta_correctness_to_recent_scope(
        pd.DataFrame({"value": [1, 2, 3]}),
        months=2,
    )

    assert scoped.empty
    assert metadata["enabled"] is False
    assert metadata["reason"] == "missing_timestamp_column"


def test_meta_correctness_integration_hpo_uses_bounded_search_space():
    from extreme_price_movements import regime_adaptor as ra

    assert ra.META_CORRECTNESS_INTEGRATION_HPO_TRIALS == 15
    assert ra.META_CORRECTNESS_INTEGRATION_SEARCH_SPACE["lambda_correctness"] == (
        0.0,
        1.0,
    )
    assert ra.META_CORRECTNESS_INTEGRATION_SEARCH_SPACE[
        "correctness_offset_cap"
    ] == (0.20, 0.60)


def test_regime_pnl_objective_uses_top_5_10_20_and_ranking_economics():
    from extreme_price_movements import regime_adaptor as ra

    scores = np.linspace(0.0, 1.0, 240)
    pnl = np.linspace(-0.03, 0.04, 240)

    good = ra._regime_pnl_topk_objective(scores, pnl)
    bad = ra._regime_pnl_topk_objective(1.0 - scores, pnl)

    assert good["top_fracs"] == pytest.approx([0.05, 0.10, 0.20])
    assert len(good["per_top_frac"]) == 3
    assert sorted(row["top_frac"] for row in good["per_top_frac"]) == pytest.approx(
        [0.05, 0.10, 0.20]
    )
    assert "lcb_mean_net_pnl_weighted" in good
    assert "bps_weighted_profit_rate_weighted" in good
    assert "sortino_component_weighted" in good
    assert "spread_component_weighted" in good
    assert good["objective"] > bad["objective"]


def test_meta_correctness_integration_objective_promotes_regime_pnl_uplift(monkeypatch):
    from extreme_price_movements import regime_adaptor as ra

    n = 240
    meta = np.linspace(0.55, 0.45, n)
    p_correct = np.linspace(0.05, 0.95, n)
    pnl = np.linspace(-0.03, 0.04, n)
    monkeypatch.setattr(ra, "optuna", None)

    result = ra._optimise_meta_correctness_integration_for_policy_topk(
        meta,
        p_correct,
        pnl,
        top_rank_threshold=0.80,
        size_power=1.0,
        trials=0,
    )

    assert result["objective"] > 1.0
    assert result["regime_pnl_metrics"]["top_fracs"] == pytest.approx(
        [0.05, 0.10, 0.20]
    )
    assert result["candidate_objective"] > result["baseline_objective"]
    assert result["integration_search"]["method"] == "fallback_grid"


def test_meta_correctness_probability_calibrator_maps_raw_probability():
    from extreme_price_movements import regime_adaptor as ra

    raw = np.r_[np.linspace(0.05, 0.45, 30), np.linspace(0.55, 0.95, 30)]
    y = np.r_[np.zeros(30), np.ones(30)]

    spec = ra._fit_meta_correctness_probability_calibrator(raw, y)
    calibrated = ra._apply_meta_correctness_probability_calibrator(
        np.asarray([0.10, 0.90], dtype=float),
        spec,
    )

    assert spec["enabled"] is True
    assert calibrated[0] < 0.25
    assert calibrated[1] > 0.75


def test_meta_correctness_oof_metrics_include_quality_and_combination_delta():
    from extreme_price_movements import regime_adaptor as ra

    meta = np.array([0.95, 0.90, 0.85, 0.60, 0.55, 0.50], dtype=float)
    pnl = np.array([0.10, -0.20, 0.05, 0.30, 0.20, -0.10], dtype=float)
    p_correct = np.array([0.95, 0.05, 0.85, 0.90, 0.80, 0.20], dtype=float)
    y_hard = (pnl > 0.0).astype(float)
    y_soft = np.where(y_hard > 0.0, 0.8, 0.2)

    metrics = ra._meta_correctness_oof_quality_metrics(
        meta,
        p_correct,
        y_hard,
        y_soft,
        pnl,
        integration_params={"lambda_correctness": 1.0, "correctness_offset_cap": 0.75},
    )

    assert metrics["rows"] == 6
    assert metrics["p_meta_correct_auc"] > 0.5
    assert metrics["integration_policy"] == "bounded_logit_adjustment"
    assert "combination_improvement" in metrics
    assert "top30_precision_delta" in metrics["combination_improvement"]
    assert metrics["combination_improvement"]["top30_net_pnl_delta"] >= 0.0


def test_apply_meta_correctness_regime_adaptor_uses_serialized_model_branch():
    from extreme_price_movements.regime_adaptor import apply_regime_adaptor

    class FakeCorrectnessModel:
        selected_features = ["meta_score_for_correctness", "feature_drift_psi_core"]

        def predict(self, x):
            assert "p_bad_regime_asset_3d" not in x.columns
            assert "feature_drift_psi_core" in x.columns
            return np.array([0.2, 0.5, 0.9], dtype=np.float32)

        def inference_schema_diagnostics(self, x):
            return {"missing_selected_features_preview": []}

    artifact = {
        "schema_version": "rolling_meta_correctness_v2",
        "model_type": "meta_correctness_lgbm",
        "enable_regime_adaptor": True,
        "selected_correctness_integration_params": {
            "lambda_correctness": 0.5,
            "correctness_offset_cap": 0.5,
        },
        "meta_correctness_probability_calibrator": {
            "enabled": True,
            "method": "isotonic",
            "x_thresholds": [0.2, 0.9],
            "y_thresholds": [0.1, 0.8],
        },
        "deployment_score_rank_reference": [0.45, 0.50, 0.55],
        "_meta_correctness_model_object": FakeCorrectnessModel(),
    }
    frame = pd.DataFrame(
        {
            "feature_drift_psi_core": [0.1, 0.2, 0.3],
            "p_bad_regime_asset_3d": [0.9, 0.9, 0.9],
        }
    )
    pred = np.array([0.5, 0.5, 0.5], dtype=float)

    applied = apply_regime_adaptor(
        frame,
        pred,
        artifact,
        timestamps=pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"),
        symbols=["AAA", "BBB", "CCC"],
    )

    assert np.all(applied["regime_adjustment_enabled"])
    assert applied["deployment_score_pre_rank"][0] < pred[0]
    assert applied["deployment_score_pre_rank"][-1] > pred[-1]
    assert "meta_correctness_probability" in applied
    assert np.allclose(applied["meta_correctness_probability_raw"], [0.2, 0.5, 0.9])
    assert np.allclose(
        applied["meta_correctness_probability_calibrated"],
        [0.1, 0.4, 0.8],
    )
    assert np.all(applied["meta_correctness_probability_calibrator_enabled"])
    assert "deployment_score_reference_rank" in applied
    assert np.all(applied["deployment_score_rank_reference_n"] == 3)
    assert applied["rank_scope"][0] == "training_validation_reference"
    assert "selected_correctness_integration_params" in applied

def test_direct_asset_level_columns_are_not_re_rolled():
    from extreme_price_movements.regime_adaptor import build_regime_feature_frame

    frame = pd.DataFrame(
        {
            "asset_volume_30d": [10.0, 20.0, 30.0],
            "asset_atr_30d": [0.1, 0.2, 0.3],
        }
    )
    features, mapping = build_regime_feature_frame(
        frame,
        pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC"),
        ["AAA", "AAA", "AAA"],
    )
    assert np.allclose(features["asset_volume_30d"], [10.0, 20.0, 30.0])
    assert np.allclose(features["asset_atr_30d"], [0.1, 0.2, 0.3])
    assert mapping["asset_volume_30d"] == "asset_volume_30d"


def test_prediction_reliability_adds_base_meta_model_disagreement():
    from extreme_price_movements import regime_adaptor as ra

    frame = pd.DataFrame(
        {
            "base_pred": [0.20, 0.55, 0.90],
            "calibrated_score": [0.30, 0.45, 0.60],
        }
    )
    base, _ = ra.build_regime_feature_frame(
        frame,
        pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"),
        ["AAA", "AAA", "AAA"],
    )
    out, _ = ra._append_prediction_reliability_features(
        base,
        frame,
        np.asarray([0.30, 0.45, 0.60], dtype=np.float32),
        None,
        pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"),
        np.asarray(["AAA", "AAA", "AAA"]),
    )

    expected = np.asarray([0.10, 0.10, 0.30], dtype=np.float32)
    assert "base_meta_model_disagreement" in ra.REGIME_FEATURE_ORDER
    assert np.allclose(out["base_meta_model_disagreement"], expected)
    assert np.allclose(out["base_meta_model_disagreement"], out["abs_base_meta_diff"])


def test_daily_anchor_asset_level_fallback_uses_daily_30_period_window():
    from extreme_price_movements.regime_adaptor import build_regime_feature_frame

    values = np.arange(40, dtype=float)
    frame = pd.DataFrame({"volume": values, "atr_pct": values / 100.0})
    features, _ = build_regime_feature_frame(
        frame,
        pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC"),
        ["AAA"] * 40,
    )
    assert np.isclose(features["asset_volume_30d"].iloc[-1], np.mean(values[9:39]))
    assert np.isclose(
        features["asset_atr_30d"].iloc[-1], np.mean((values / 100.0)[9:39])
    )


def test_funding_side_alignment_uses_trade_side_not_trend_alias():
    from extreme_price_movements.regime_adaptor import build_regime_feature_frame

    frame = pd.DataFrame(
        {
            "asset_funding_z": [2.0, -3.0],
            "trade_side": [1.0, -1.0],
            "asset_funding_trend_alignment": [-99.0, -99.0],
        }
    )
    features, _ = build_regime_feature_frame(
        frame,
        pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC"),
        ["AAA", "AAA"],
    )
    assert np.allclose(features["asset_funding_side_alignment"], [2.0, 3.0])
    assert np.allclose(features["asset_funding_trend_alignment"], [-99.0, -99.0])


def test_regime_feature_frame_keeps_model_state_and_drift_features():
    from extreme_price_movements.inference.feature_generator import (
        is_model_derived_feature_key,
    )
    from extreme_price_movements.regime_adaptor import build_regime_feature_frame

    frame = pd.DataFrame(
        {
            "feature_drift_psi_core": [0.1, 0.2],
            "feature_drift_ks_core": [0.3, 0.4],
            "feature_drift_psi_bin_mean": [0.5, 0.6],
            "feature_drift_ks_bin_mean": [0.7, 0.8],
            "contrib_abs_sum": [1.0, 2.0],
            "top_3_contrib_abs_sum": [1.5, 2.5],
            "archetype_contrib_svd_00": [0.01, 0.02],
            "raw_state_svd_00": [0.03, 0.04],
            "raw_state_svd_mean": [0.035, 0.045],
            "raw_state_svd_std": [0.005, 0.006],
            "raw_state_mahalanobis": [3.0, 4.0],
            "raw_state_psi_mean": [0.05, 0.06],
            "raw_state_ks_max": [0.07, 0.08],
            "raw_state_svd_psi_mean": [0.09, 0.10],
            "raw_state_svd_ks_max": [0.11, 0.12],
            "state_log_likelihood": [-2.0, -1.5],
            "state_tod_mahalanobis": [0.9, 1.1],
        }
    )
    features, mapping = build_regime_feature_frame(
        frame,
        pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
        ["AAA", "AAA"],
    )
    expected = [
        "feature_drift_psi_core",
        "feature_drift_ks_core",
        "feature_drift_psi_bin_mean",
        "feature_drift_ks_bin_mean",
        "contrib_abs_sum",
        "top_3_contrib_abs_sum",
        "archetype_contrib_svd_00",
        "raw_state_svd_00",
        "raw_state_svd_mean",
        "raw_state_svd_std",
        "raw_state_mahalanobis",
        "raw_state_psi_mean",
        "raw_state_ks_max",
        "raw_state_svd_psi_mean",
        "raw_state_svd_ks_max",
        "state_log_likelihood",
        "state_tod_mahalanobis",
    ]
    assert expected == [c for c in expected if c in features.columns]
    for col in expected:
        assert mapping[col] == col
        assert np.allclose(features[col], frame[col])
        assert is_model_derived_feature_key(col)


def test_simple_policy_generated_rows_carry_regime_model_context():
    events = pd.DataFrame(
        {"timestamp": pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")}
    )
    meta_base = pd.DataFrame(
        {
            "raw_state_svd_00": [0.1, 0.2, 0.3],
            "state_tod_mahalanobis": [1.0, 1.1, 1.2],
            "archetype_contrib_svd_00": [0.4, 0.5, 0.6],
            "feature_drift_ks_core": [0.7, 0.8, 0.9],
            "oof_feature_drift_psi_core": [0.2, 0.3, 0.4],
            "oof_regime_centroid_similarity_train": [0.9, 0.8, 0.7],
            "oof_raw_state_svd_mean": [0.01, 0.02, 0.03],
            "oof_leaf_hit_rate_avg": [0.6, 0.5, 0.4],
            "oof_support_gap": [0.1, -0.1, 0.0],
            "regular_raw_feature": [10.0, 11.0, 12.0],
        }
    )
    out = _attach_regime_model_context_columns(events, meta_base)
    assert "regular_raw_feature" not in out.columns
    for col in (
        "raw_state_svd_00",
        "state_tod_mahalanobis",
        "archetype_contrib_svd_00",
        "feature_drift_ks_core",
        "oof_feature_drift_psi_core",
        "oof_regime_centroid_similarity_train",
        "oof_raw_state_svd_mean",
        "oof_leaf_hit_rate_avg",
        "oof_support_gap",
    ):
        assert np.allclose(out[col], meta_base[col])
    assert sorted(out.attrs["regime_model_context_columns"]) == [
        "archetype_contrib_svd_00",
        "feature_drift_ks_core",
        "oof_feature_drift_psi_core",
        "oof_leaf_hit_rate_avg",
        "oof_raw_state_svd_mean",
        "oof_regime_centroid_similarity_train",
        "oof_support_gap",
        "raw_state_svd_00",
        "state_tod_mahalanobis",
    ]
