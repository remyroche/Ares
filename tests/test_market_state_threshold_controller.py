import numpy as np
import pandas as pd
import pytest

from scripts import run_market_state_threshold_controller as mstc
from scripts import materialize_market_state_controller_bundle as mat
from scripts import run_market_state_threshold_controller_walkforward as wf
from scripts import score_market_state_controller_bundle as score_bundle


def test_market_state_bundle_defaults_are_maturity_safe_noop_paths() -> None:
    assert "20260626_t1_lgbm_maturity_contract_v1" in str(mat.DEFAULT_SELECTED_CONTROLLER)
    assert "market_state_controller_bundle_t1_lgbm_maturity_noop_20260626" in str(mat.DEFAULT_OUTPUT_DIR)
    assert "market_state_controller_bundle_t1_lgbm_maturity_noop_20260626" in str(score_bundle.DEFAULT_BUNDLE)
    assert "market_state_controller_bundle_score_t1_lgbm_maturity_noop_20260626" in str(
        score_bundle.DEFAULT_OUTPUT_DIR
    )


def test_materializer_default_null_selection_is_noop() -> None:
    selected, payload = mat._load_selected_arm(
        mat.DEFAULT_SELECTED_CONTROLLER,
        "S1_observed_axes_shared_response",
        allow_default=False,
        allow_null_noop=True,
    )

    assert selected == mat.NOOP_CONTROLLER_ARM
    assert payload["selected_arm_noop_used"] is True
    assert payload["selected_arm_default_used"] is False


def _response_training_frame(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    ts = pd.date_range("2026-05-01", periods=rows, freq="h", tz="UTC")
    rank = np.linspace(0.52, 0.98, rows)
    state_shock = np.sin(np.linspace(0, 8, rows))
    strategy = np.where(np.arange(rows) % 2 == 0, "strategy_a", "strategy_b")
    full_sl = ((state_shock > 0.35) & (rank < 0.82)).astype(int)
    timeout = ((state_shock < -0.65) & (rank < 0.78)).astype(int)
    net_return = 0.02 * (rank - 0.70) - 0.035 * full_sl - 0.008 * timeout
    net_return += rng.normal(0.0, 0.002, rows)
    exit_reason = np.where(full_sl == 1, "sl", np.where(timeout == 1, "timeout", "tp"))
    return pd.DataFrame(
        {
            "timestamp": ts,
            "strategy_id": strategy,
            "head": np.where(strategy == "strategy_a", "short_asset", "short_boll"),
            "side": "short",
            "normalized_rank_score": rank,
            "base_strategy_threshold": 0.70,
            "deployment_rank_threshold": 0.70,
            "calibrated_score": rank,
            "state_shock": state_shock,
            "state_liquidity_stress_proxy": np.cos(np.linspace(0, 5, rows)),
            "net_return": net_return,
            "simple_policy_exit_reason": exit_reason,
        }
    )


def _tiny_observed_axis_encoder(*, minimum_input_coverage: float = 0.80) -> dict:
    ts = pd.date_range("2026-05-01", periods=8, freq="h", tz="UTC")
    return mstc.fit_observed_axis_encoder(
        pd.DataFrame(
            {
                "timestamp": ts,
                "fs__mkt_ret_eq_1h__mean": np.linspace(-0.012, 0.015, len(ts)),
                "fs__rv_24h__mean": np.linspace(0.018, 0.036, len(ts)),
                "fs__mkt_oi_chg_z_24h__mean": np.linspace(-0.7, 0.6, len(ts)),
            }
        ),
        minimum_input_coverage=minimum_input_coverage,
    )


class _ZeroModel:
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=float)


class _ConstantModel:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value, dtype=float)


class _FlatCurves:
    def predict(self, strategy: pd.Series, rank: pd.Series, target: str) -> np.ndarray:
        if target == "mu":
            return np.full(len(rank), -0.01, dtype=float)
        if target == "psl":
            return np.full(len(rank), 0.20, dtype=float)
        if target == "pto":
            return np.full(len(rank), 0.10, dtype=float)
        return np.zeros(len(rank), dtype=float)

    def strategy_cap(self, strategy_id: str, target: str, base_threshold: float, pad: float) -> float:
        return 0.30 if target == "psl" else 0.50


def test_apply_rank_contract_uses_frozen_global_policy_reference(monkeypatch, tmp_path) -> None:
    from scripts import reliability_blend_rank_reference as rank_reference

    calls = {}

    def fake_apply(frame, *, data_root, run_id, score_col, allow_window_rank_debug):
        calls.update(
            {
                "rows": len(frame),
                "data_root": data_root,
                "run_id": run_id,
                "score_col": score_col,
                "allow_window_rank_debug": allow_window_rank_debug,
            }
        )
        out = frame.copy()
        ranks = np.array([0.82, 0.41], dtype=float)
        out["policy_rank_pct"] = ranks
        out["strategy_rank_pct"] = ranks
        out["normalized_rank_score"] = ranks
        out["rank_pct"] = ranks
        out["auction_rank_score"] = np.array([0.77, 0.35], dtype=float)
        out["threshold_rank_score_source"] = "policy_rank_reference_percentile"
        return out, {"rank_source": "policy_rank_reference_percentile"}

    monkeypatch.setattr(rank_reference, "apply_frozen_policy_rank_reference", fake_apply)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-01", periods=2, freq="h", tz="UTC"),
            "symbol": ["AAA", "BBB"],
            "side": ["short", "short"],
            "strategy_id": ["short_asset_x", "short_boll_y"],
            "head": ["short_asset", "short_boll"],
            "anchor_score": [0.7, 0.6],
            "calibrated_score": [0.65, 0.55],
            "base_strategy_threshold": [0.70, 0.70],
            "entry_price": [100.0, 200.0],
            "exit_price": [101.0, 198.0],
            "exit_timestamp": pd.date_range("2026-06-01 01:00", periods=2, freq="h", tz="UTC"),
            "gross_return": [0.01, -0.01],
            "net_return": [0.009, -0.011],
            "holding_bars": [1, 1],
            "simple_policy_exit_reason": ["tp", "sl"],
        }
    )

    out = mstc._apply_rank_contract(
        frame,
        "anchor_global_policy_rank_reference",
        data_root=tmp_path,
        rank_reference_run_id="rank_ref_test",
    )

    assert calls == {
        "rows": 2,
        "data_root": tmp_path,
        "run_id": "rank_ref_test",
        "score_col": "anchor_score",
        "allow_window_rank_debug": False,
    }
    assert out["rank_contract_source"].eq("anchor_global_policy_rank_reference").all()
    np.testing.assert_allclose(out["normalized_rank_score"].to_numpy(), [0.82, 0.41])
    np.testing.assert_allclose(out["auction_rank_score"].to_numpy(), [0.77, 0.35])


def test_response_risk_heads_model_excess_risk_residuals() -> None:
    frame = mstc.build_response_frame(_response_training_frame(), pd.DataFrame({"timestamp": []}))
    state_cols = ["state_shock", "state_liquidity_stress_proxy"]

    models, feature_cols, report = mstc.fit_response_models(
        frame,
        state_cols,
        per_strategy_residual=False,
        max_rows=0,
        max_keyword_cols=8,
    )
    preds = mstc.predict_response(models, frame.iloc[:20].copy(), feature_cols, state_cols)

    assert report["response_model_kind"] == "additive_ebm"
    assert report["risk_model"] == "rank_curve_plus_additive_ebm_response"
    assert report["response_model_family"] == "pooled_additive_binned_response_model"
    assert report["additive_bins"] >= 4
    assert report["response_weighting"]["timestamp_balanced"] is True
    assert report["response_weighting"]["strategy_balanced"] is True
    assert report["response_weighting"]["frontier_gamma"] == 3.0
    assert report["response_weighting"]["effective_sample_size"] > 0
    assert report["response_weighting"]["near_frontier_005_weight_share"] > 0
    assert "state_feature_coverage" in preds.columns
    assert "response_feature_coverage" in preds.columns
    assert "state_ood_score" in preds.columns
    assert "state_ood_cutoff" in preds.columns
    assert "state_ood_flag" in preds.columns
    assert preds["state_feature_coverage"].between(0.0, 1.0).all()
    assert preds["response_feature_coverage"].between(0.0, 1.0).all()
    assert "full_sl_residual_mean" in report["risk_baseline"]
    assert "timeout_residual_mean" in report["risk_baseline"]
    assert models["response_model_kind"] == "additive_ebm"
    assert not hasattr(models["risk"]["full_sl"], "predict_proba")
    assert not hasattr(models["risk"]["timeout"], "predict_proba")
    assert "raw_pred_full_sl" not in preds.columns
    assert "raw_pred_timeout" not in preds.columns
    np.testing.assert_allclose(
        preds["pred_full_sl"].to_numpy(),
        (preds["base_psl"] + preds["pred_excess_full_sl"]).clip(0.0, 1.0).to_numpy(),
    )
    np.testing.assert_allclose(
        preds["pred_timeout"].to_numpy(),
        (preds["base_pto"] + preds["pred_excess_timeout"]).clip(0.0, 1.0).to_numpy(),
    )
    assert preds["pred_full_sl"].between(0.0, 1.0).all()
    assert preds["pred_timeout"].between(0.0, 1.0).all()

    missing_state = frame.iloc[:20].drop(columns=["state_liquidity_stress_proxy"]).copy()
    missing_preds = mstc.predict_response(models, missing_state, feature_cols, state_cols)
    assert missing_preds["state_feature_coverage"].lt(1.0).all()


def test_strategy_response_artifact_builders_emit_residual_predictions_and_effects() -> None:
    frame = _response_training_frame(140)
    response_frame = mstc.build_response_frame(
        frame,
        pd.DataFrame({"timestamp": frame["timestamp"].drop_duplicates().to_numpy()}),
    )
    state_cols = ["state_shock", "state_liquidity_stress_proxy"]
    models, feature_cols, report = mstc.fit_response_models(
        response_frame,
        state_cols,
        per_strategy_residual=False,
        max_rows=0,
        max_keyword_cols=8,
    )
    pred = mstc.predict_response(models, response_frame, feature_cols, state_cols)

    residual = wf._strategy_residual_target_ledger(
        response_frame,
        models["curves"],
        fold=1,
        arm="S1_observed_axes_shared_response",
    )
    prediction = wf._strategy_response_prediction_ledger(
        response_frame,
        pred,
        fold=1,
        arm="S1_observed_axes_shared_response",
    )
    effects = wf._strategy_state_effect_matrix(
        response_frame,
        pred,
        state_cols,
        fold=1,
        arm="S1_observed_axes_shared_response",
    )

    assert report["risk_model"] == "rank_curve_plus_additive_ebm_response"
    assert {"base_mu", "base_psl", "base_pto", "resid_utility", "resid_full_sl", "resid_timeout"}.issubset(
        residual.columns
    )
    assert residual["fold"].eq(1).all()
    assert prediction["pred_resid_utility"].notna().all()
    assert prediction["pred_resid_full_sl"].notna().all()
    assert prediction["state_prediction_contract"].eq("outer_fold_validation_state_scores").all()
    assert not effects.empty
    assert {"state_feature", "target", "spearman", "target_q90_minus_q10"}.issubset(effects.columns)

    contract = wf._response_state_training_contract(
        arm="S2_observed_forecast_shared_response",
        state_cols=["state_shock", "forecast_h6_shock_up"],
        state_report={
            "forecast_report": {
                "targets": {
                    "forecast_h6_shock_up": {
                        "train_prediction_mode": "chronological_expanding_oof_or_fallback"
                    }
                }
            }
        },
    )
    assert contract["response_training_uses_oof_state_scores"] is True
    assert contract["learned_state_non_oof_columns"] == []
    assert set(effects["state_feature"]).issubset(set(state_cols))


def test_walkforward_adds_pruned_state_pack_without_mutating_base_arms() -> None:
    ts = pd.date_range("2026-05-01", periods=3, freq="h", tz="UTC")
    train_observed = pd.DataFrame({"timestamp": ts, "state_shock": [0.1, 0.2, 0.3]})
    valid_observed = pd.DataFrame({"timestamp": ts, "state_shock": [0.4, 0.5, 0.6]})
    train_forecast = pd.DataFrame(
        {
            "timestamp": ts,
            "state_shock": [0.1, 0.2, 0.3],
            "forecast_h6_shock_up": [0.7, 0.8, 0.9],
            "forecast_h6_rv_ratio": [0.2, 0.2, 0.2],
        }
    )
    valid_forecast = pd.DataFrame(
        {
            "timestamp": ts,
            "state_shock": [0.4, 0.5, 0.6],
            "forecast_h6_shock_up": [0.3, 0.4, 0.5],
            "forecast_h6_rv_ratio": [0.1, 0.1, 0.1],
        }
    )
    states = {
        "S1_observed_axes_shared_response": (
            train_observed,
            valid_observed,
            ["state_shock"],
        ),
        "S2_observed_forecast_shared_response": (
            train_forecast,
            valid_forecast,
            ["state_shock", "forecast_h6_shock_up", "forecast_h6_rv_ratio"],
        ),
    }

    out = wf._add_pruned_state_pack(
        states,
        state_head_allowlist=["forecast_h6_shock_up"],
    )

    assert wf.PRUNED_STATE_ARM in out
    assert out["S2_observed_forecast_shared_response"][2] == [
        "state_shock",
        "forecast_h6_shock_up",
        "forecast_h6_rv_ratio",
    ]
    assert out[wf.PRUNED_STATE_ARM][2] == ["forecast_h6_shock_up"]
    assert list(out[wf.PRUNED_STATE_ARM][0].columns) == ["timestamp", "forecast_h6_shock_up"]


def test_pruned_state_pack_has_known_controller_complexity() -> None:
    assert wf._controller_arm_complexity(wf.PRUNED_STATE_ARM) == 2


def test_hist_gradient_response_challenger_remains_available() -> None:
    frame = mstc.build_response_frame(_response_training_frame(), pd.DataFrame({"timestamp": []}))
    state_cols = ["state_shock", "state_liquidity_stress_proxy"]

    models, feature_cols, report = mstc.fit_response_models(
        frame,
        state_cols,
        per_strategy_residual=False,
        max_rows=120,
        max_keyword_cols=4,
        response_model_kind="hist_gradient_boosting",
    )
    preds = mstc.predict_response(models, frame.iloc[:12].copy(), feature_cols, state_cols)

    assert report["response_model_kind"] == "hist_gradient_boosting"
    assert report["risk_model"] == "rank_curve_plus_excess_risk_regressors"
    assert models["response_model_kind"] == "hist_gradient_boosting"
    assert preds["pred_mean_utility"].notna().all()


def test_xgboost_response_challenger_remains_available() -> None:
    frame = mstc.build_response_frame(_response_training_frame(), pd.DataFrame({"timestamp": []}))
    state_cols = ["state_shock", "state_liquidity_stress_proxy"]

    models, feature_cols, report = mstc.fit_response_models(
        frame,
        state_cols,
        per_strategy_residual=False,
        max_rows=120,
        max_keyword_cols=4,
        response_model_kind="xgboost",
    )
    preds = mstc.predict_response(models, frame.iloc[:12].copy(), feature_cols, state_cols)

    assert report["response_model_kind"] == "xgboost"
    assert report["risk_model"] == "rank_curve_plus_xgboost_response"
    assert report["response_model_family"] == "rank_curve_plus_xgboost_response"
    assert models["response_model_kind"] == "xgboost"
    assert preds["pred_mean_utility"].notna().all()


def test_response_frame_rejects_duplicate_state_timestamps() -> None:
    candidates = _response_training_frame(6)
    state = pd.DataFrame(
        {
            "timestamp": [candidates["timestamp"].iloc[0], candidates["timestamp"].iloc[0]],
            "state_shock": [0.1, 0.2],
        }
    )

    with pytest.raises(ValueError, match="one row per timestamp"):
        mstc.build_response_frame(candidates, state)


def test_response_frame_rejects_state_candidate_column_overlap() -> None:
    candidates = _response_training_frame(6)
    candidates["state_shock"] = 0.0
    state = pd.DataFrame(
        {
            "timestamp": candidates["timestamp"].drop_duplicates().iloc[:3].to_numpy(),
            "state_shock": [0.1, 0.2, 0.3],
        }
    )

    with pytest.raises(ValueError, match="overlap candidate columns"):
        mstc.build_response_frame(candidates, state)


def test_joined_state_invariance_rejects_within_timestamp_variance() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-06-15T00:00:00Z"),
                pd.Timestamp("2026-06-15T00:00:00Z"),
                pd.Timestamp("2026-06-15T01:00:00Z"),
            ],
            "strategy_id": ["a", "b", "a"],
            "state_shock": [0.1, 0.2, 0.3],
            "state_realized_vol": [0.5, 0.5, 0.6],
        }
    )

    with pytest.raises(ValueError, match="joined state columns vary within timestamp: state_shock"):
        mstc._validate_joined_state_invariance(frame, ["state_shock", "state_realized_vol"])


def test_response_frame_preserves_one_state_value_per_timestamp_after_join() -> None:
    candidates = _response_training_frame(8)
    state = pd.DataFrame(
        {
            "timestamp": candidates["timestamp"].drop_duplicates().to_numpy(),
            "state_market_shock": np.linspace(0.0, 1.0, candidates["timestamp"].nunique()),
        }
    )

    joined = mstc.build_response_frame(candidates, state)

    nunique = joined.groupby("timestamp")["state_market_shock"].nunique(dropna=False)
    assert nunique.eq(1).all()


def test_state_frame_contract_report_measures_one_row_per_timestamp() -> None:
    state = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-15", periods=3, freq="h", tz="UTC"),
            "state_shock": [0.1, 0.2, np.nan],
            "state_realized_vol": [0.5, 0.6, 0.7],
        }
    )

    report = mstc.state_frame_contract_report(state, context="unit")

    assert report["one_row_per_timestamp"] is True
    assert report["duplicate_timestamp_count"] == 0
    assert report["state_feature_count"] == 2
    assert report["nonfinite_state_value_count"] == 1


def test_joined_state_invariance_report_measures_timestamp_constant_join() -> None:
    candidates = _response_training_frame(8)
    state = pd.DataFrame(
        {
            "timestamp": candidates["timestamp"].drop_duplicates().to_numpy(),
            "state_market_shock": np.linspace(0.0, 1.0, candidates["timestamp"].nunique()),
        }
    )
    joined = mstc.build_response_frame(candidates, state)

    report = mstc.joined_state_invariance_report(joined, ["state_market_shock"], context="unit")

    assert report["state_join_timestamp_constant"] is True
    assert report["max_state_values_per_timestamp"] == 1
    assert report["row_count"] == len(joined)


def test_market_state_timestamp_panel_and_feature_coverage_schema() -> None:
    train_state = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-15", periods=2, freq="h", tz="UTC"),
            "state_shock": [0.1, np.nan],
            "state_realized_vol": [0.5, 0.6],
        }
    )
    eval_state = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-16", periods=2, freq="h", tz="UTC"),
            "state_shock": [0.2, 0.3],
            "state_realized_vol": [0.7, 0.8],
        }
    )

    panel = mstc.market_state_timestamp_panel(
        [
            ("train", "observed", train_state),
            ("eval", "observed", eval_state),
        ]
    )
    coverage = mstc.market_state_feature_coverage(panel)

    assert panel.columns[:3].tolist() == ["split", "state_level", "timestamp"]
    assert set(panel["split"]) == {"train", "eval"}
    train_shock = coverage.loc[
        coverage["split"].eq("train") & coverage["feature"].eq("state_shock")
    ].iloc[0]
    assert int(train_shock["row_count"]) == 2
    assert int(train_shock["finite_count"]) == 1
    np.testing.assert_allclose(float(train_shock["finite_share"]), 0.5)


def test_accepted_decision_key_guard_rejects_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-06-15T00:00:00Z"),
                pd.Timestamp("2026-06-15T00:00:00Z"),
            ],
            "symbol": ["BTC", "BTC"],
            "side": ["short", "short"],
            "strategy_id": ["strategy_a", "strategy_a"],
        }
    )

    with pytest.raises(ValueError, match="duplicate decision keys"):
        mstc._assert_unique_decision_keys(frame, context="unit-test accepted trades")


def test_accepted_decision_key_guard_allows_unique_keys() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-06-15T00:00:00Z"),
                pd.Timestamp("2026-06-15T00:00:00Z"),
            ],
            "symbol": ["BTC", "ETH"],
            "side": ["short", "short"],
            "strategy_id": ["strategy_a", "strategy_a"],
        }
    )

    mstc._assert_unique_decision_keys(frame, context="unit-test accepted trades")


def test_controller_enabled_head_manifest_resolves_active_scope() -> None:
    disabled = {"long_bars", "long_dist"}

    all_active = mstc._controller_enabled_heads_manifest(None, disabled)
    assert all_active["controller_enabled_scope"] == "all_active_heads"
    assert all_active["controller_enabled_heads"] == ["short_asset", "short_boll"]

    explicit = mstc._controller_enabled_heads_manifest({"short_boll", "long_bars"}, disabled)
    assert explicit["controller_enabled_scope"] == "explicit"
    assert explicit["controller_enabled_heads"] == ["short_boll"]
    assert explicit["controller_enabled_heads_ignored_inactive"] == ["long_bars"]


def test_score_bundle_enabled_head_parser_handles_legacy_and_explicit_values() -> None:
    assert score_bundle._bundle_enabled_heads("all_active_heads") is None
    assert score_bundle._bundle_enabled_heads(["short_asset", "short_boll"]) == {"short_asset", "short_boll"}
    assert score_bundle._bundle_enabled_heads("short_boll,long_dist") == {"short_boll", "long_dist"}
    assert score_bundle._bundle_enabled_heads(None) is None


def test_frontier_weights_balance_timestamps_and_emphasize_threshold_band() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-06-15T00:00:00Z"),
                pd.Timestamp("2026-06-15T00:00:00Z"),
                pd.Timestamp("2026-06-15T00:00:00Z"),
                pd.Timestamp("2026-06-15T00:00:00Z"),
                pd.Timestamp("2026-06-15T01:00:00Z"),
                pd.Timestamp("2026-06-15T01:00:00Z"),
            ],
            "strategy_id": ["a", "a", "b", "b", "a", "b"],
            "_rank": [0.70, 0.90, 0.71, 0.95, 0.70, 0.95],
            "_threshold": 0.70,
        }
    )

    timestamp_only = mstc._frontier_weights(
        frame,
        frontier_gamma=0.0,
        balance_timestamps=True,
        balance_strategies=False,
    )
    mass = pd.Series(timestamp_only).groupby(frame["timestamp"].to_numpy()).sum()
    np.testing.assert_allclose(mass.iloc[0], mass.iloc[1])

    frontier = mstc._frontier_weights(
        frame,
        frontier_gamma=3.0,
        frontier_bandwidth=0.03,
        balance_timestamps=False,
        balance_strategies=False,
    )
    near = frontier[np.isclose(frame["_rank"].to_numpy(dtype=float), 0.70)]
    far = frontier[frame["_rank"].to_numpy(dtype=float) >= 0.90]
    assert float(near.mean()) > float(far.mean())


def test_forecast_state_heads_report_chronological_holdout_metrics() -> None:
    n_train = 140
    n_eval = 24
    ts_train = pd.date_range("2026-04-01", periods=n_train, freq="h", tz="UTC")
    ts_eval = pd.date_range("2026-04-07", periods=n_eval, freq="h", tz="UTC")
    x_train = np.linspace(0.0, 14.0, n_train)
    x_eval = np.linspace(14.0, 16.4, n_eval)

    def agg_frame(ts: pd.DatetimeIndex, x: np.ndarray) -> pd.DataFrame:
        ret = 0.006 * np.sin(x) + 0.003 * np.cos(0.35 * x)
        return pd.DataFrame(
            {
                "timestamp": ts,
                "fs__mkt_ret_eq_1h__mean": ret,
                "fs__rv_24h__mean": 0.02 + 0.01 * np.abs(np.sin(0.5 * x)),
                "fs__mkt_oi_chg_z_24h__mean": -0.4 * np.sin(0.8 * x),
                "fs__amihud_z__mean": 0.2 + np.abs(np.cos(0.7 * x)),
            }
        )

    train_agg = agg_frame(ts_train, x_train)
    eval_agg = agg_frame(ts_eval, x_eval)
    train_state = pd.DataFrame(
        {
            "timestamp": ts_train,
            "state_shock": np.sin(x_train),
            "state_realized_vol": np.abs(np.sin(0.5 * x_train)),
            "state_trend": np.cos(0.25 * x_train),
        }
    )
    eval_state = pd.DataFrame(
        {
            "timestamp": ts_eval,
            "state_shock": np.sin(x_eval),
            "state_realized_vol": np.abs(np.sin(0.5 * x_eval)),
            "state_trend": np.cos(0.25 * x_eval),
        }
    )

    train_forecast, eval_forecast, report = mstc.add_forecast_state_heads(
        train_state,
        eval_state,
        horizon_steps=[4],
        train_agg=train_agg,
        eval_agg=eval_agg,
    )

    forecast_cols = [c for c in train_forecast.columns if c.startswith("forecast_h4_")]
    assert forecast_cols
    assert train_forecast[forecast_cols].apply(lambda s: s.between(0.0, 1.0).all()).all()
    assert eval_forecast[forecast_cols].apply(lambda s: s.between(0.0, 1.0).all()).all()
    assert train_forecast["state_forecast_disagreement"].between(0.0, 1.0).all()
    assert eval_forecast["state_forecast_disagreement"].between(0.0, 1.0).all()
    assert train_forecast["state_uncertainty"].between(0.0, 1.0).all()
    assert report["forecast_model_kind"] == "lightgbm"
    assert report["model_backend"] in {"lightgbm_lgbm_regressor", "sklearn_gradient_boosting_regressor_fallback"}
    assert report["reliability_channels"]["state_forecast_disagreement"] == "rowwise_cross_forecast_std_scaled_to_0_1"
    trained_reports = [
        target_report
        for target_report in report["targets"].values()
        if target_report.get("mode") == "gbm_soft_empirical_cdf_target"
    ]
    assert trained_reports
    assert any(int(target_report.get("validation_rows", 0)) > 0 for target_report in trained_reports)
    assert any("validation_top_decile_lift" in target_report for target_report in trained_reports)
    assert any("validation_tail_brier_p90" in target_report for target_report in trained_reports)
    assert any("validation_tail_ece_5bin" in target_report for target_report in trained_reports)
    assert any("validation_collapsed" in target_report for target_report in trained_reports)
    assert all(
        target_report.get("train_prediction_mode") == "chronological_expanding_oof_or_fallback"
        for target_report in trained_reports
    )
    assert any(int(target_report.get("oof_rows", 0)) > 0 for target_report in trained_reports)


def test_forecast_artifact_replays_eval_predictions_without_refit() -> None:
    n_train = 130
    n_eval = 16
    ts_train = pd.date_range("2026-04-01", periods=n_train, freq="h", tz="UTC")
    ts_eval = pd.date_range("2026-04-07", periods=n_eval, freq="h", tz="UTC")
    x_train = np.linspace(0.0, 12.0, n_train)
    x_eval = np.linspace(12.0, 13.5, n_eval)

    def agg_frame(ts: pd.DatetimeIndex, x: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": ts,
                "fs__mkt_ret_eq_1h__mean": 0.004 * np.sin(x),
                "fs__rv_24h__mean": 0.02 + 0.01 * np.abs(np.cos(x)),
                "fs__mkt_oi_chg_z_24h__mean": -0.2 * np.sin(0.5 * x),
                "fs__amihud_z__mean": 0.5 + np.abs(np.sin(0.2 * x)),
            }
        )

    train_state = pd.DataFrame(
        {
            "timestamp": ts_train,
            "state_shock": np.sin(x_train),
            "state_realized_vol": np.abs(np.cos(x_train)),
            "state_trend": np.cos(0.25 * x_train),
        }
    )
    eval_state = pd.DataFrame(
        {
            "timestamp": ts_eval,
            "state_shock": np.sin(x_eval),
            "state_realized_vol": np.abs(np.cos(x_eval)),
            "state_trend": np.cos(0.25 * x_eval),
        }
    )
    train_agg = agg_frame(ts_train, x_train)
    eval_agg = agg_frame(ts_eval, x_eval)

    train_fit, artifact, report = mstc.fit_forecast_state_heads(
        train_state,
        [4],
        train_agg=train_agg,
    )
    eval_replay = mstc.transform_forecast_state_heads(eval_state, artifact, agg=eval_agg)
    train_legacy, eval_legacy, legacy_report = mstc.add_forecast_state_heads(
        train_state,
        eval_state,
        [4],
        train_agg=train_agg,
        eval_agg=eval_agg,
    )

    cols = [c for c in eval_replay.columns if c.startswith("forecast_h4_")]
    assert cols
    assert artifact["mode"] == "primitive_future_soft_severity_regressors_v1"
    assert artifact["forecast_model_kind"] == "lightgbm"
    trained_target_specs = [
        spec
        for spec in dict(artifact.get("targets", {})).values()
        if spec.get("mode") == "gbm_soft_empirical_cdf_target"
    ]
    assert trained_target_specs
    assert all("target_cdf_reference" in spec for spec in trained_target_specs)
    assert all(int(spec["target_cdf_reference"]["n"]) > 0 for spec in trained_target_specs)
    assert report["targets"].keys() == legacy_report["targets"].keys()
    replay_cols = [*cols, "state_forecast_disagreement", "state_uncertainty"]
    pd.testing.assert_frame_equal(train_fit[["timestamp", *replay_cols]], train_legacy[["timestamp", *replay_cols]])
    pd.testing.assert_frame_equal(eval_replay[["timestamp", *replay_cols]], eval_legacy[["timestamp", *replay_cols]])


def test_xgboost_forecast_state_heads_are_available() -> None:
    n_train = 110
    ts_train = pd.date_range("2026-04-01", periods=n_train, freq="h", tz="UTC")
    x = np.linspace(0.0, 11.0, n_train)
    train_agg = pd.DataFrame(
        {
            "timestamp": ts_train,
            "fs__mkt_ret_eq_1h__mean": 0.004 * np.sin(x),
            "fs__rv_24h__mean": 0.02 + 0.01 * np.abs(np.cos(x)),
            "fs__mkt_oi_chg_z_24h__mean": -0.2 * np.sin(0.5 * x),
            "fs__amihud_z__mean": 0.5 + np.abs(np.sin(0.2 * x)),
        }
    )
    train_state = pd.DataFrame(
        {
            "timestamp": ts_train,
            "state_shock": np.sin(x),
            "state_realized_vol": np.abs(np.cos(x)),
            "state_trend": np.cos(0.25 * x),
        }
    )

    train_fit, artifact, report = mstc.fit_forecast_state_heads(
        train_state,
        [4],
        train_agg=train_agg,
        forecast_model_kind="xgboost",
    )

    forecast_cols = [c for c in train_fit.columns if c.startswith("forecast_h4_")]
    assert forecast_cols
    assert artifact["forecast_model_kind"] == "xgboost"
    assert report["forecast_model_kind"] == "xgboost"
    assert report["model_backend"] == "xgboost_xgb_regressor"
    assert train_fit[forecast_cols].apply(lambda s: s.between(0.0, 1.0).all()).all()


def test_walkforward_state_head_registry_summarizes_active_fallback_and_shadow_heads() -> None:
    fold_reports = [
        {
            "fold": 1,
            "split_maturity_contract": {
                "training_entry_end": pd.Timestamp("2026-05-01", tz="UTC"),
                "training_outcome_available_before": pd.Timestamp("2026-05-05", tz="UTC"),
                "uses_matured_training_outcomes_only": True,
                "train_broad": {
                    "entry_filtered_rows": 100,
                    "matured_rows": 100,
                    "dropped_immature_outcome_rows": 0,
                    "missing_outcome_available_rows": 0,
                    "max_outcome_available_timestamp": pd.Timestamp("2026-05-04", tz="UTC"),
                },
                "train_deployable": {
                    "entry_filtered_rows": 40,
                    "matured_rows": 40,
                    "dropped_immature_outcome_rows": 0,
                    "missing_outcome_available_rows": 0,
                    "max_outcome_available_timestamp": pd.Timestamp("2026-05-04", tz="UTC"),
                },
            },
            "state_report": {
                "axis_sources": {
                    "state_shock": ["fs__mkt_ret_eq_1h__mean"],
                    "state_input_coverage": ["fs__mkt_ret_eq_1h__mean", "fs__rv_24h__mean"],
                },
                "forecast_report": {
                    "features_used": 12,
                    "targets": {
                        "forecast_h6_shock_up": {
                            "mode": "gbm_soft_empirical_cdf_target",
                            "rows": 100,
                            "target_std": 0.2,
                            "validation_rows": 20,
                            "validation_top_decile_lift": 0.12,
                            "validation_tail_average_precision": 0.40,
                            "validation_tail_ap_lift_p90": 0.25,
                            "validation_tail_brier_p90": 0.18,
                            "validation_tail_ece_5bin": 0.07,
                            "validation_tail_false_alarm_rate_p90": 0.10,
                            "validation_tail_recall_p90": 0.50,
                            "validation_collapsed": False,
                            "oof_coverage": 0.70,
                        },
                        "forecast_h6_liquidity_stress_proxy": {
                            "mode": "current_axis_fallback",
                            "rows": 15,
                        },
                    },
                },
                "latent_report": {
                    "mode": "shadow_disabled_by_default",
                    "reason": "latent_gmm_outputs_removed_from_active_controller_architecture",
                },
            },
        },
        {
            "fold": 2,
            "state_report": {
                "axis_sources": {
                    "state_shock": ["fs__mkt_ret_eq_1h__mean"],
                    "state_input_coverage": ["fs__mkt_ret_eq_1h__mean", "fs__rv_24h__mean"],
                },
                "forecast_report": {
                    "features_used": 12,
                    "targets": {
                        "forecast_h6_shock_up": {
                            "mode": "gbm_soft_empirical_cdf_target",
                            "rows": 90,
                            "target_std": 0.3,
                            "validation_rows": 18,
                            "validation_top_decile_lift": -0.02,
                            "validation_tail_average_precision": 0.20,
                            "validation_tail_ap_lift_p90": 0.05,
                            "validation_tail_brier_p90": 0.22,
                            "validation_tail_ece_5bin": 0.09,
                            "validation_tail_false_alarm_rate_p90": 0.20,
                            "validation_tail_recall_p90": 0.25,
                            "validation_collapsed": True,
                            "oof_coverage": 0.50,
                        },
                        "forecast_h6_liquidity_stress_proxy": {
                            "mode": "current_axis_fallback",
                            "rows": 10,
                            "oof_reason": "insufficient_valid_rows",
                        },
                    },
                },
                "latent_report": {
                    "mode": "shadow_disabled_by_default",
                    "reason": "latent_gmm_outputs_removed_from_active_controller_architecture",
                },
            },
        },
    ]

    registry = wf._state_head_registry(fold_reports)

    shock = registry.loc[registry["state_head"].eq("forecast_h6_shock_up")].iloc[0]
    assert shock["aggregate_status"] == "active"
    assert shock["trained_folds"] == 2
    assert shock["component_group"] == "return_shock"
    assert shock["positive_validation_lift_share"] == 0.5
    np.testing.assert_allclose(shock["mean_oof_coverage"], 0.60)
    np.testing.assert_allclose(shock["mean_tail_average_precision"], 0.30)
    np.testing.assert_allclose(shock["mean_tail_brier_p90"], 0.20)
    assert shock["collapsed_folds"] == 1

    liquidity = registry.loc[registry["state_head"].eq("forecast_h6_liquidity_stress_proxy")].iloc[0]
    assert liquidity["aggregate_status"] == "fallback"
    assert liquidity["fallback_folds"] == 2
    assert liquidity["component_group"] == "liquidity_proxy"
    assert "insufficient_valid_rows" in liquidity["disable_reasons"]

    latent = registry.loc[registry["state_head"].eq("latent_gmm_probabilities")].iloc[0]
    assert latent["aggregate_status"] == "shadow_disabled"
    assert latent["shadow_disabled_folds"] == 2
    assert "latent_gmm_outputs_removed" in latent["disable_reasons"]

    observed = registry.loc[registry["state_head"].eq("state_input_coverage")].iloc[0]
    assert observed["state_level"] == "observed_axis"
    assert observed["aggregate_status"] == "active"
    assert observed["mean_source_count"] == 2.0

    flattened = " ".join(registry.astype(str).to_numpy().ravel().tolist())
    assert "short_asset" not in flattened
    assert "short_boll" not in flattened


def test_market_state_activation_registry_uses_generic_state_evidence_only() -> None:
    state_head_registry = pd.DataFrame(
        {
            "state_level": ["observed_axis", "observed_axis", "forecast"],
            "state_head": ["state_shock", "state_redundant", "forecast_h6_shock_up"],
            "component_group": ["return_shock", "return_shock", "return_shock"],
            "aggregate_status": ["active", "active", "active"],
            "folds_seen": [2, 2, 2],
            "trained_folds": [2, 2, 2],
            "fallback_folds": [0, 0, 0],
            "shadow_disabled_folds": [0, 0, 0],
            "active_fold_share": [1.0, 1.0, 1.0],
            "fallback_fold_share": [0.0, 0.0, 0.0],
            "mean_source_count": [2.0, 2.0, 4.0],
            "mean_validation_rows": [np.nan, np.nan, 100.0],
            "mean_validation_top_decile_lift": [np.nan, np.nan, 0.02],
            "mean_tail_average_precision": [np.nan, np.nan, 0.35],
            "mean_tail_ap_lift_p90": [np.nan, np.nan, 0.05],
            "mean_tail_brier_p90": [np.nan, np.nan, 0.20],
            "mean_tail_ece_5bin": [np.nan, np.nan, 0.05],
            "mean_tail_false_alarm_rate_p90": [np.nan, np.nan, 0.10],
            "mean_tail_recall_p90": [np.nan, np.nan, 0.30],
            "collapsed_folds": [0, 0, 0],
            "positive_validation_lift_share": [np.nan, np.nan, 1.0],
            "mean_oof_coverage": [np.nan, np.nan, 0.80],
            "min_oof_coverage": [np.nan, np.nan, 0.75],
            "mean_target_rows": [np.nan, np.nan, 200.0],
            "mean_target_std": [np.nan, np.nan, 0.25],
            "status_counts": ["{}", "{}", "{}"],
            "disable_reasons": ["", "", ""],
        }
    )
    ts = pd.date_range("2026-06-01", periods=8, freq="h", tz="UTC")
    state_panel = pd.DataFrame(
        {
            "fold": [1] * len(ts),
            "split": ["valid"] * len(ts),
            "timestamp": ts,
            "arm": ["S1_observed_axes_shared_response"] * len(ts),
            "state_shock": np.linspace(0.0, 1.0, len(ts)),
            "state_redundant": np.linspace(0.0, 1.0, len(ts)),
            "forecast_h6_shock_up": np.linspace(1.0, 0.0, len(ts)),
        }
    )
    state_effect_matrix = pd.DataFrame(
        {
            "state_feature": ["state_shock", "forecast_h6_shock_up"],
            "target": ["pred_resid_utility", "pred_resid_utility"],
            "target_q90_minus_q10": [0.025, 0.020],
            "spearman": [0.08, 0.05],
        }
    )
    schedules = pd.DataFrame(
        {
            "arm": [
                "S1_observed_axes_shared_response",
                "S2_observed_forecast_shared_response",
            ],
            "head": ["short_asset", "short_boll"],
            "base_threshold": [0.70, 0.70],
            "state_threshold": [0.73, 0.74],
            "raw_state_threshold": [0.73, 0.74],
            "suppressed_candidate_count": [3, 2],
            "state_ood_share": [0.0, 0.0],
        }
    )
    loo_aggregate = pd.DataFrame(
        {
            "state_head": ["state_shock", "state_redundant", "forecast_h6_shock_up"],
            "action_arm_hint": [
                "S1_observed_axes_shared_response",
                "S1_observed_axes_shared_response",
                "S2_observed_forecast_shared_response",
            ],
            "loo_replay_folds": [2, 2, 2],
            "loo_mode": ["neutralized_valid_state_no_refit"] * 3,
            "loo_median_increment_net_pnl": [10.0, -2.0, 4.0],
            "loo_mean_increment_net_pnl": [9.0, -1.5, 4.5],
            "loo_q25_increment_net_pnl": [5.0, -3.0, 2.0],
            "loo_positive_increment_share": [1.0, 0.0, 0.5],
            "loo_mean_accepted_jaccard": [0.80, 0.99, 0.85],
            "loo_mean_delta_trade_count": [1.0, 0.0, 0.5],
            "loo_mean_threshold_raise_delta": [2.0, 0.0, 1.0],
            "loo_state_head_defensive_success": [3.0, -1.0, 2.0],
            "loo_state_head_median_defensive_success": [1.5, -1.0, 1.0],
            "loo_state_head_positive_defensive_share": [1.0, 0.0, 1.0],
            "loo_state_head_loss_avoided": [5.0, 0.5, 3.0],
            "loo_state_head_winner_pnl_sacrificed": [2.0, 1.5, 1.0],
            "loo_state_head_net_action_pnl_delta": [3.0, -1.0, 2.0],
        }
    )

    registry = wf._market_state_activation_registry(
        state_head_registry,
        state_panel,
        state_effect_matrix,
        schedules,
        loo_aggregate,
    )

    shock = registry.loc[registry["state_head"].eq("state_shock")].iloc[0]
    assert shock["recommended_status"] == "active_candidate"
    assert shock["leave_one_head_out_status"] == "required_before_promotion"

    redundant = registry.loc[registry["state_head"].eq("state_redundant")].iloc[0]
    assert redundant["recommended_status"] == "disabled_candidate"
    assert "redundant_without_response_effect" in redundant["activation_disable_reason"]

    forecast = registry.loc[registry["state_head"].eq("forecast_h6_shock_up")].iloc[0]
    assert forecast["recommended_status"] == "active_candidate"

    flattened = " ".join(registry.astype(str).to_numpy().ravel().tolist())
    assert "short_asset" not in flattened
    assert "short_boll" not in flattened


def test_market_state_activation_registry_disables_winner_sacrificing_state_head() -> None:
    state_head_registry = pd.DataFrame(
        {
            "state_level": ["observed_axis"],
            "state_head": ["state_shock"],
            "component_group": ["return_shock"],
            "aggregate_status": ["active"],
            "folds_seen": [2],
            "trained_folds": [2],
            "fallback_folds": [0],
            "shadow_disabled_folds": [0],
            "active_fold_share": [1.0],
            "fallback_fold_share": [0.0],
            "mean_source_count": [2.0],
            "mean_validation_rows": [np.nan],
            "mean_validation_top_decile_lift": [np.nan],
            "mean_tail_average_precision": [np.nan],
            "mean_tail_ap_lift_p90": [np.nan],
            "mean_tail_brier_p90": [np.nan],
            "mean_tail_ece_5bin": [np.nan],
            "mean_tail_false_alarm_rate_p90": [np.nan],
            "mean_tail_recall_p90": [np.nan],
            "collapsed_folds": [0],
            "positive_validation_lift_share": [np.nan],
            "mean_oof_coverage": [np.nan],
            "min_oof_coverage": [np.nan],
            "mean_target_rows": [np.nan],
            "mean_target_std": [np.nan],
            "status_counts": ["{}"],
            "disable_reasons": [""],
        }
    )
    state_panel = pd.DataFrame(
        {
            "split": ["valid", "valid", "valid"],
            "timestamp": pd.date_range("2026-06-01", periods=3, freq="h", tz="UTC"),
            "state_shock": [0.1, 0.2, 0.3],
        }
    )
    state_effect_matrix = pd.DataFrame(
        {
            "state_feature": ["state_shock"],
            "target": ["pred_resid_utility"],
            "target_q90_minus_q10": [0.03],
            "spearman": [0.1],
        }
    )
    schedules = pd.DataFrame(
        {
            "arm": ["S1_observed_axes_shared_response"],
            "base_threshold": [0.70],
            "state_threshold": [0.75],
            "raw_state_threshold": [0.75],
            "suppressed_candidate_count": [4],
        }
    )
    loo_aggregate = pd.DataFrame(
        {
            "state_head": ["state_shock"],
            "action_arm_hint": ["S1_observed_axes_shared_response"],
            "loo_replay_folds": [2],
            "loo_median_increment_net_pnl": [5.0],
            "loo_positive_increment_share": [1.0],
            "loo_state_head_defensive_success": [-2.0],
            "loo_state_head_loss_avoided": [1.0],
            "loo_state_head_winner_pnl_sacrificed": [3.0],
        }
    )

    registry = wf._market_state_activation_registry(
        state_head_registry,
        state_panel,
        state_effect_matrix,
        schedules,
        loo_aggregate,
    )

    row = registry.iloc[0]
    assert row["recommended_status"] == "disabled_candidate"
    assert "state_action_sacrifices_winners" in row["activation_disable_reason"]


def test_state_leave_one_out_aggregate_reports_incremental_value_by_state_and_arm() -> None:
    replay = pd.DataFrame(
        {
            "fold": [1, 2, 1],
            "arm": [
                "S1_observed_axes_shared_response",
                "S1_observed_axes_shared_response",
                "S2_observed_forecast_shared_response",
            ],
            "state_head": ["state_shock", "state_shock", "forecast_h6_shock_up"],
            "leave_one_out_mode": ["neutralized_valid_state_no_refit"] * 3,
            "increment_net_pnl": [5.0, -1.0, 3.0],
            "accepted_jaccard_vs_full": [0.80, 0.90, 0.70],
            "delta_trade_count": [1, 0, 2],
            "full_threshold_raise_count": [4, 5, 3],
            "loo_threshold_raise_count": [2, 4, 1],
            "state_head_removed_loss_avoided": [4.0, 1.0, 3.0],
            "state_head_removed_winner_pnl_sacrificed": [1.0, 2.0, 1.0],
            "state_head_defensive_success": [3.0, -1.0, 2.0],
            "state_head_net_action_pnl_delta": [3.5, -0.5, 2.5],
        }
    )

    out = wf._aggregate_state_leave_one_out(replay)

    shock = out.loc[out["state_head"].eq("state_shock")].iloc[0]
    assert shock["action_arm_hint"] == "S1_observed_axes_shared_response"
    assert shock["loo_replay_folds"] == 2
    np.testing.assert_allclose(shock["loo_median_increment_net_pnl"], 2.0)
    np.testing.assert_allclose(shock["loo_positive_increment_share"], 0.5)
    np.testing.assert_allclose(shock["loo_mean_threshold_raise_delta"], 1.5)
    np.testing.assert_allclose(shock["loo_state_head_defensive_success"], 2.0)
    np.testing.assert_allclose(shock["loo_state_head_loss_avoided"], 5.0)
    np.testing.assert_allclose(shock["loo_state_head_winner_pnl_sacrificed"], 3.0)


def test_materialized_bundle_activation_registry_filters_disabled_state_heads() -> None:
    activation_report = {
        "available": True,
        "active_state_heads": ["state_shock"],
        "disabled_state_heads": ["state_redundant"],
        "shadow_state_heads": [],
    }

    active_cols, report = mat._filter_state_columns_by_activation_registry(
        ["state_shock", "state_redundant"],
        activation_report,
    )

    assert active_cols == ["state_shock"]
    assert report["enforced"] is True
    assert report["active_state_feature_count"] == 1
    assert report["dropped_state_feature_columns"] == ["state_redundant"]


def test_materialized_bundle_missing_activation_registry_fails_closed() -> None:
    active_cols, report = mat._filter_state_columns_by_activation_registry(
        ["state_shock", "forecast_h6_shock_up"],
        {"available": False, "reason": "activation_registry_not_provided_or_not_found"},
    )

    assert active_cols == []
    assert report["enforced"] is True
    assert report["reason"] == "activation_registry_unavailable_fail_closed"
    assert report["active_state_feature_count"] == 0
    assert report["dropped_state_feature_columns"] == ["state_shock", "forecast_h6_shock_up"]


def test_materialized_bundle_missing_activation_registry_debug_override_uses_all_heads() -> None:
    active_cols, report = mat._filter_state_columns_by_activation_registry(
        ["state_shock", "forecast_h6_shock_up"],
        {"available": False, "reason": "activation_registry_not_provided_or_not_found"},
        fail_closed_when_unavailable=False,
    )

    assert active_cols == ["state_shock", "forecast_h6_shock_up"]
    assert report["enforced"] is False
    assert report["reason"] == "activation_registry_unavailable"


def test_materialized_bundle_empty_activation_registry_disables_controller_scope() -> None:
    class Args:
        disable_heads = "long_bars,long_dist"
        controller_enabled_heads = ""
        rank_contract = "short_boll_timestamp_rank"
        allow_candidate_state_fallback = False

    state_artifacts = {
        "candidate_feature_cols": ["feature_a"],
        "feature_store_cols": ["fs_feature_a"],
        "states": {
            "observed": (
                pd.DataFrame(),
                pd.DataFrame(),
                ["state_shock", "state_redundant"],
            )
        },
        "reports": {
            "feature_store": {"train": {}, "eval": {}},
            "market_state_source": {},
            "observed_axis_encoder": {},
            "axis_sources": {},
            "forecast_report": {},
            "latent_report": {},
        },
    }
    activation_filter = {
        "active_state_feature_columns": [],
        "dropped_state_feature_columns": ["state_shock", "state_redundant"],
    }
    contract = mat._make_market_state_feature_contract(
        args=Args(),
        selected_arm="S1_observed_axes_shared_response",
        state_spec={"state_level": "observed", "per_strategy_residual": False},
        state_artifacts=state_artifacts,
        response_feature_cols=["rank"],
        activation_report={"available": True, "active_state_heads": []},
        state_activation_filter=activation_filter,
        controller_execution_enabled=False,
        walkforward_config={"available": True, "controller": {"forecast_model_kind": "xgboost"}},
        forecast_model_kind_report={"value": "xgboost", "source": "walkforward_manifest"},
        response_model_kind_report={"value": "hist_gradient_boosting", "source": "walkforward_manifest"},
    )

    assert contract["controller_execution_enabled"] is False
    assert contract["controller_enabled_scope"] == "disabled_by_activation_registry"
    assert contract["controller_enabled_heads"] == []
    assert contract["source_schema"]["state_feature_columns"] == []


def test_bundle_enabled_heads_preserves_empty_set_as_disabled_scope() -> None:
    assert score_bundle._bundle_enabled_heads([]) == set()
    assert score_bundle._bundle_enabled_heads(None) is None
    assert score_bundle._bundle_enabled_heads("short_asset,short_boll") == {"short_asset", "short_boll"}


def test_executable_materialized_bundle_rejects_candidate_population_fallback() -> None:
    with pytest.raises(RuntimeError, match="Refusing to materialize an executable or shadow"):
        mat._validate_candidate_state_fallback_execution_contract(
            allow_candidate_state_fallback=True,
            controller_execution_enabled=True,
        )

    with pytest.raises(RuntimeError, match="Refusing to materialize an executable or shadow"):
        mat._validate_candidate_state_fallback_execution_contract(
            allow_candidate_state_fallback=True,
            controller_execution_enabled=False,
            shadow_controller_only=True,
        )

    mat._validate_candidate_state_fallback_execution_contract(
        allow_candidate_state_fallback=True,
        controller_execution_enabled=False,
    )


def test_materialized_shadow_bundle_rejects_collapsed_eval_feature_store_references() -> None:
    state_artifacts = {
        "reports": {
            "observed_axis_encoder": {"source_column_count": 0},
            "market_state_source": {
                "train": {"feature_count": 128},
                "eval": {"feature_count": 0},
            },
            "feature_store": {"eval": {"timestamp_coverage": 0.0}},
        }
    }

    with pytest.raises(RuntimeError, match="no common train/eval observed-axis references"):
        mat._validate_state_reference_materialization_contract(
            state_cols=["state_shock"],
            state_artifacts=state_artifacts,
            controller_execution_enabled=False,
            shadow_controller_only=True,
        )


def test_shadow_no_backfill_accepted_delta_report_counts_removed_winner_cost() -> None:
    base = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-24", periods=3, freq="h", tz="UTC"),
            "symbol": ["AAA", "BBB", "CCC"],
            "strategy_id": ["s", "s", "s"],
            "head": ["short_asset", "short_asset", "short_boll"],
            "side": ["short", "short", "short"],
            "net_pnl": [-5.0, 3.0, 7.0],
        }
    )
    shadow = base.iloc[[0, 2]].copy()

    delta, summary = score_bundle._accepted_trade_delta_report(base, shadow)

    assert summary["available"] is True
    assert summary["baseline_trade_count"] == 3
    assert summary["shadow_trade_count"] == 2
    assert summary["removed_trade_count"] == 1
    assert summary["added_trade_count"] == 0
    assert summary["shadow_subset_of_baseline"] is True
    assert summary["removed_net_pnl"] == 3.0
    assert summary["baseline_net_pnl"] == 5.0
    assert summary["shadow_net_pnl"] == 2.0
    assert summary["total_net_pnl_delta"] == -3.0
    assert summary["full_path_replay_net_pnl_delta"] == -3.0
    assert summary["common_net_pnl_delta"] == 0.0
    assert summary["path_dependent_common_trade_net_pnl_delta"] == 0.0
    assert summary["action_only_fixed_common_size_net_pnl_delta"] == -3.0
    assert summary["removed_loss_avoided"] == 0.0
    assert summary["removed_winner_pnl_sacrificed"] == 3.0
    assert summary["accepted_delta_defensive_success"] == -3.0
    assert set(delta["delta_action"]) == {"common_accepted", "removed_by_shadow_no_backfill"}


def test_locked_accepted_overlay_from_direct_marks_no_replay_invariants() -> None:
    direct_accepted = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-24", periods=2, freq="h", tz="UTC"),
            "symbol": ["AAA", "BBB"],
            "strategy_id": ["s", "s"],
            "head": ["short_asset", "short_boll"],
            "side": ["short", "short"],
            "net_pnl": [1.0, -2.0],
            "arm": ["old", "old"],
        }
    )
    direct_delta = direct_accepted.assign(delta_action="common_accepted")
    direct_summary = {
        "direct_threshold_only": True,
        "no_path_or_capacity_replay": True,
        "added_trade_count": 0,
        "common_net_pnl_delta": 0.0,
    }

    arm, locked_accepted, locked_delta, locked_summary = (
        score_bundle._locked_accepted_overlay_from_direct(
            arm="S1_observed_axes_shared_response",
            direct_accepted=direct_accepted,
            direct_delta=direct_delta,
            direct_summary=direct_summary,
        )
    )

    assert arm == "S1_observed_axes_shared_response__shadow_locked_accepted_overlay"
    assert set(locked_accepted["arm"]) == {arm}
    assert set(direct_accepted["arm"]) == {"old"}
    assert locked_delta.equals(direct_delta)
    assert locked_summary["locked_accepted_overlay"] is True
    assert locked_summary["direct_threshold_only"] is True
    assert locked_summary["no_path_or_capacity_replay"] is True
    assert locked_summary["no_replacement_candidates"] is True
    assert locked_summary["common_trade_sizing_locked"] is True
    assert locked_summary["auction_ordering_locked"] is True
    assert direct_summary.get("locked_accepted_overlay") is None


def test_materialized_bundle_backend_auto_resolution_uses_walkforward_manifest() -> None:
    walkforward = {
        "path": "wf/manifest.json",
        "controller": {
            "forecast_model_kind": "xgboost",
            "response_model_kind": "hist_gradient_boosting",
        },
    }

    forecast_kind, forecast_report = mat._resolve_backend_kind(
        "auto",
        walkforward,
        "forecast_model_kind",
        "lightgbm",
        {"lightgbm", "xgboost"},
    )
    response_kind, response_report = mat._resolve_backend_kind(
        "auto",
        walkforward,
        "response_model_kind",
        "additive_ebm",
        {"additive_ebm", "hist_gradient_boosting", "xgboost"},
    )

    assert forecast_kind == "xgboost"
    assert forecast_report["source"] == "walkforward_manifest"
    assert response_kind == "hist_gradient_boosting"
    assert response_report["source"] == "walkforward_manifest"


def test_materialized_bundle_backend_cli_override_beats_walkforward_manifest() -> None:
    kind, report = mat._resolve_backend_kind(
        "lightgbm",
        {"controller": {"forecast_model_kind": "xgboost"}},
        "forecast_model_kind",
        "lightgbm",
        {"lightgbm", "xgboost"},
    )

    assert kind == "lightgbm"
    assert report["source"] == "cli"


def test_materialized_bundle_runtime_params_inherit_walkforward_manifest_defaults() -> None:
    class Args:
        inherit_walkforward_controller_config = True
        response_frontier_weight_gamma = 3.0
        response_frontier_weight_bandwidth = 0.06
        response_balance_timestamps = True
        response_balance_strategies = True
        threshold_delta_max = 0.10
        max_threshold_up_step = 0.03
        threshold_relax_alpha = 0.25
        controller_mode = "rank_grid"
        controller_min_lcb_utility = 0.0
        controller_min_prediction_coverage = 0.80
        controller_min_usable_candidates = 1
        controller_max_state_ood_score = None
        controller_min_action_edge = 0.0
        controller_winner_sacrifice_multiplier = 1.0
        use_timeout_cap = False

    args = Args()
    report = mat._apply_walkforward_runtime_defaults(
        args,
        {
            "path": "wf/manifest.json",
            "controller": {
                "response_weighting": {
                    "frontier_gamma": 9.0,
                    "frontier_bandwidth": 0.11,
                    "timestamp_balanced": False,
                    "strategy_balanced": False,
                },
                "threshold_delta_max": 0.04,
                "max_threshold_up_step": 0.02,
                "threshold_relax_alpha": 0.10,
                "controller_mode": "frontier_rank_grid",
                "controller_min_lcb_utility": -0.002,
                "controller_min_prediction_coverage": 0.91,
                "controller_min_usable_candidates": 4,
                "controller_max_state_ood_score": 0.80,
                "controller_min_action_edge": 0.003,
                "controller_winner_sacrifice_multiplier": 1.4,
                "use_timeout_cap": True,
            },
        },
    )

    assert args.response_frontier_weight_gamma == 9.0
    assert args.response_balance_timestamps is False
    assert args.threshold_delta_max == 0.04
    assert args.controller_mode == "frontier_rank_grid"
    assert args.controller_min_usable_candidates == 4
    assert args.use_timeout_cap is True
    assert report["params"]["threshold_delta_max"]["source"] == "walkforward_manifest"
    assert report["params"]["response_frontier_weight_gamma"]["source"] == "walkforward_manifest"
    assert report["walkforward_manifest"] == "wf/manifest.json"


def test_materialized_bundle_runtime_params_preserve_cli_overrides() -> None:
    class Args:
        inherit_walkforward_controller_config = True
        response_frontier_weight_gamma = 5.0
        response_frontier_weight_bandwidth = 0.06
        response_balance_timestamps = True
        response_balance_strategies = True
        threshold_delta_max = 0.07
        max_threshold_up_step = 0.03
        threshold_relax_alpha = 0.25
        controller_mode = "severity"
        controller_min_lcb_utility = 0.0
        controller_min_prediction_coverage = 0.80
        controller_min_usable_candidates = 1
        controller_max_state_ood_score = None
        controller_min_action_edge = 0.0
        controller_winner_sacrifice_multiplier = 1.0
        use_timeout_cap = False

    args = Args()
    report = mat._apply_walkforward_runtime_defaults(
        args,
        {
            "path": "wf/manifest.json",
            "controller": {
                "response_weighting": {"frontier_gamma": 9.0},
                "threshold_delta_max": 0.04,
                "controller_mode": "frontier_rank_grid",
            },
        },
    )

    assert args.response_frontier_weight_gamma == 5.0
    assert args.threshold_delta_max == 0.07
    assert args.controller_mode == "severity"
    assert report["params"]["threshold_delta_max"]["source"] == "cli_or_materializer_default"
    assert report["params"]["controller_mode"]["source"] == "cli_or_materializer_default"


def test_materialized_bundle_runtime_params_can_disable_walkforward_inheritance() -> None:
    class Args:
        inherit_walkforward_controller_config = False
        response_frontier_weight_gamma = 3.0
        response_frontier_weight_bandwidth = 0.06
        response_balance_timestamps = True
        response_balance_strategies = True
        threshold_delta_max = 0.10
        max_threshold_up_step = 0.03
        threshold_relax_alpha = 0.25
        controller_mode = "rank_grid"
        controller_min_lcb_utility = 0.0
        controller_min_prediction_coverage = 0.80
        controller_min_usable_candidates = 1
        controller_max_state_ood_score = None
        controller_min_action_edge = 0.0
        controller_winner_sacrifice_multiplier = 1.0
        use_timeout_cap = False

    args = Args()
    report = mat._apply_walkforward_runtime_defaults(
        args,
        {
            "path": "wf/manifest.json",
            "controller": {
                "response_weighting": {"frontier_gamma": 9.0},
                "threshold_delta_max": 0.04,
            },
        },
    )

    assert args.response_frontier_weight_gamma == 3.0
    assert args.threshold_delta_max == 0.10
    assert report["params"]["threshold_delta_max"]["source"] == "cli_or_materializer_default_no_inherit"


def test_walkforward_market_state_contract_and_target_artifacts_summarize_fold_reports() -> None:
    class Args:
        rank_contract = "short_boll_timestamp_rank"
        policy_variant = "refit_bar4_strategy_bar2"
        allow_candidate_state_fallback = False
        include_latent_shadow_arms = False
        n_folds = 2
        min_train_days = 21
        valid_days = 7
        embargo_hours = 96
        min_valid_rows = 25
        min_valid_timestamps = 4

    fold_reports = [
        {
            "fold": 1,
            "split_maturity_contract": {
                "training_entry_end": pd.Timestamp("2026-05-01", tz="UTC"),
                "training_outcome_available_before": pd.Timestamp("2026-05-05", tz="UTC"),
                "uses_matured_training_outcomes_only": True,
                "train_broad": {
                    "entry_filtered_rows": 100,
                    "matured_rows": 100,
                    "dropped_immature_outcome_rows": 0,
                    "missing_outcome_available_rows": 0,
                    "max_outcome_available_timestamp": pd.Timestamp("2026-05-04", tz="UTC"),
                },
                "train_deployable": {
                    "entry_filtered_rows": 40,
                    "matured_rows": 40,
                    "dropped_immature_outcome_rows": 0,
                    "missing_outcome_available_rows": 0,
                    "max_outcome_available_timestamp": pd.Timestamp("2026-05-04", tz="UTC"),
                },
            },
            "state_report": {
                "candidate_feature_count": 12,
                "feature_store": {
                    "selected_column_count": 2,
                    "train": {
                        "feature_dir": "features/train",
                        "enabled": True,
                        "columns": ["mkt_ret_eq_1h", "rv_24h"],
                        "timestamp_coverage": 1.0,
                        "symbols_read": 2,
                        "universe_contract": {
                            "universe_definition_version": "feature_store_timestamp_market_state_v1",
                            "available_symbol_count": 3,
                            "eligible_symbol_count": 2,
                            "eligible_symbols": ["AAA", "BBB"],
                        },
                    },
                    "valid": {
                        "feature_dir": "features/valid",
                        "enabled": True,
                        "timestamp_coverage": 0.95,
                        "symbols_read": 2,
                        "universe_contract": {
                            "universe_definition_version": "feature_store_timestamp_market_state_v1",
                            "available_symbol_count": 3,
                            "eligible_symbol_count": 2,
                        },
                    },
                },
                "market_state_source": {
                    "train": {
                        "source": "feature_store_market_aggregates",
                        "feature_count": 10,
                        "production_safe": True,
                        "candidate_aggregate_feature_count": 6,
                        "feature_store_aggregate_feature_count": 10,
                        "validation": {
                            "row_count": 100,
                            "feature_count": 10,
                            "forbidden_column_count": 0,
                            "timestamp_unique": True,
                            "market_wide_one_row_per_timestamp": True,
                        },
                    },
                    "valid": {
                        "source": "feature_store_market_aggregates",
                        "feature_count": 10,
                        "production_safe": True,
                        "candidate_aggregate_feature_count": 5,
                        "feature_store_aggregate_feature_count": 10,
                        "validation": {
                            "row_count": 25,
                            "feature_count": 10,
                            "forbidden_column_count": 0,
                            "timestamp_unique": True,
                            "market_wide_one_row_per_timestamp": True,
                        },
                    },
                },
                "axis_sources": {"state_shock": ["fs__mkt_ret_eq_1h__mean"]},
                "forecast_report": {
                    "horizon_steps": [6],
                    "targets": {
                        "forecast_h6_shock_up": {
                            "mode": "gbm_soft_empirical_cdf_target",
                            "horizon_steps": 6,
                            "raw_target": "target_h6_shock_up",
                            "fallback_axis": "state_shock",
                            "rows": 100,
                            "target_std": 0.2,
                            "soft_target_mean": 0.5,
                            "hard_tail_rate_p90": 0.1,
                            "train_prediction_mode": "chronological_expanding_oof_or_fallback",
                            "oof_rows": 70,
                            "oof_coverage": 0.70,
                        }
                    },
                    "target_source_reports": {
                        "h6": {
                            "horizon_steps": 6,
                            "source_columns": {"return": "mkt_ret_eq_1h"},
                            "valid_rows": 100,
                        }
                    },
                },
                "latent_report": {"mode": "shadow_disabled_by_default", "reason": "off"},
            },
        }
    ]
    registry = wf._state_head_registry(fold_reports)
    controller_manifest = mstc._controller_enabled_heads_manifest(None, {"long_bars", "long_dist"})

    contract = wf._market_state_feature_contract(
        args=Args(),
        folds=[{"fold": 1, "train_end": "2026-05-01", "valid_start": "2026-05-05"}],
        disabled_heads={"long_bars", "long_dist"},
        controller_enabled_manifest=controller_manifest,
        fold_reports=fold_reports,
        state_head_registry=registry,
    )
    target_defs = wf._market_state_target_definitions(fold_reports)
    target_cdfs = wf._market_state_target_cdfs(
        {
            "fold_1": {
                "forecast_artifact": {
                    "forecast_model_kind": "lightgbm",
                    "model_backend": "lightgbm_lgbm_regressor",
                    "horizon_steps": [6],
                    "targets": {
                        "forecast_h6_shock_up": {
                            "mode": "gbm_soft_empirical_cdf_target",
                            "horizon_steps": 6,
                            "raw_target": "target_h6_shock_up",
                            "target_cdf_reference": {
                                "reference_version": "empirical_cdf_reference_v1",
                                "n": 3,
                                "sorted_values": [0.1, 0.2, 0.3],
                                "quantiles": {"q50": 0.2},
                            },
                        }
                    },
                }
            }
        }
    )
    oof_state = wf._market_state_oof_predictions(
        pd.DataFrame(
            {
                "fold": [1, 1],
                "split": ["train", "valid"],
                "state_arm": ["S1", "S1"],
                "timestamp": pd.date_range("2026-06-01", periods=2, freq="h", tz="UTC"),
                "state_shock": [0.1, 0.9],
            }
        )
    )
    coverage = wf._market_state_feature_coverage(fold_reports)

    assert contract["contract_version"] == "market_state_walkforward_feature_contract_v1"
    assert contract["rank_contract"] == "short_boll_timestamp_rank"
    assert contract["active_heads"] == ["short_asset", "short_boll"]
    assert contract["validation"]["passed"] is True
    assert contract["validation"]["training_outcome_maturity_contract_passed"] is True
    assert contract["validation"]["training_immature_outcome_rows_dropped"] == 0
    assert contract["source_contract_audit"]["overall_passed"] is True
    assert contract["source_contract_audit"]["actual_order_book_features_allowed"] is False
    assert contract["source_contract_audit"]["candidate_population_fallback_allowed_for_production"] is False
    assert contract["source_contract_audit"]["splits"]["fold_1_train"]["production_safe"] is True
    assert contract["source_contract_audit"]["splits"]["fold_1_train"]["candidate_fallback_enabled"] is False
    assert contract["source_contract_audit"]["splits"]["fold_1_valid"]["validation_forbidden_column_count"] == 0
    assert contract["invariants"]["market_state_uses_candidate_counts"] is False
    assert contract["invariants"]["actual_order_book_features_allowed"] is False
    assert contract["invariants"]["controller_changes_scores_or_ranks"] is False
    assert contract["feature_store"]["train_universe_contract"]["eligible_symbols"] == ["AAA", "BBB"]
    assert "forecast_h6_shock_up" in contract["state_head_summary"]["active_state_heads"]

    target = target_defs["forecast_targets"]["forecast_h6_shock_up"]
    assert target["horizon_steps"] == [6]
    assert target["raw_targets"] == ["target_h6_shock_up"]
    assert target["mean_oof_coverage"] == 0.70
    assert target_defs["target_source_reports"]["h6"]["source_columns"]["return"] == "mkt_ret_eq_1h"
    assert target_cdfs["artifact_version"] == "market_state_target_cdfs_v1"
    assert target_cdfs["target_count"] == 1
    assert target_cdfs["folds"]["fold_1"]["targets"]["forecast_h6_shock_up"]["target_cdf_reference"]["n"] == 3
    assert oof_state["split"].tolist() == ["valid"]
    assert oof_state["prediction_contract"].tolist() == ["outer_fold_validation_state_scores"]

    assert len(coverage) == 2
    assert coverage["validation_forbidden_column_count"].eq(0).all()
    assert coverage["validation_market_wide_one_row_per_timestamp"].all()


def test_walkforward_state_timestamp_panel_and_artifact_hashes(tmp_path) -> None:
    ts = pd.date_range("2026-06-01", periods=2, freq="h", tz="UTC")
    states = {
        "S1_observed_axes_shared_response": (
            pd.DataFrame({"timestamp": ts, "state_shock": [0.1, 0.2]}),
            pd.DataFrame({"timestamp": ts + pd.Timedelta(days=1), "state_shock": [0.3, 0.4]}),
            ["state_shock"],
        ),
        "S2_observed_forecast_shared_response": (
            pd.DataFrame({"timestamp": ts, "state_shock": [0.1, 0.2], "forecast_h6_shock_up": [0.6, 0.7]}),
            pd.DataFrame(
                {
                    "timestamp": ts + pd.Timedelta(days=1),
                    "state_shock": [0.3, 0.4],
                    "forecast_h6_shock_up": [0.8, 0.9],
                }
            ),
            ["state_shock", "forecast_h6_shock_up"],
        ),
    }

    panel = wf._state_timestamp_panel(states, fold=3)

    assert len(panel) == 8
    assert set(panel["split"]) == {"train", "valid"}
    assert set(panel["state_arm"]) == set(states)
    assert panel["fold"].eq(3).all()
    assert panel.loc[panel["state_arm"].eq("S2_observed_forecast_shared_response"), "forecast_h6_shock_up"].notna().all()

    artifact = tmp_path / "artifact.txt"
    artifact.write_text("abc", encoding="utf-8")
    hashes = wf._artifact_hashes({"artifact": str(artifact), "missing": str(tmp_path / "missing.txt")})

    assert hashes["hash_version"] == "sha256_artifact_hashes_v1"
    assert hashes["artifacts"]["artifact"]["exists"] is True
    assert hashes["artifacts"]["artifact"]["sha256"]
    assert hashes["artifacts"]["missing"]["exists"] is False
    assert hashes["artifacts"]["missing"]["sha256"] is None

    score_file = tmp_path / "score.txt"
    score_file.write_text("score", encoding="utf-8")
    output_hashes = score_bundle._output_sha256(
        {
            "score": str(score_file),
            "missing": str(tmp_path / "missing_score.txt"),
        }
    )

    assert set(output_hashes) == {"score"}
    assert isinstance(output_hashes["score"], str)
    assert len(output_hashes["score"]) == 64

    materializer_hashes = mat._artifact_hashes(
        {"score": str(score_file), "missing": str(tmp_path / "missing_score.txt")}
    )
    assert materializer_hashes["hash_version"] == "sha256_artifact_hashes_v1"
    assert materializer_hashes["artifacts"]["score"]["exists"] is True
    assert len(materializer_hashes["artifacts"]["score"]["sha256"]) == 64
    assert materializer_hashes["artifacts"]["missing"]["exists"] is False
    assert materializer_hashes["artifacts"]["missing"]["sha256"] is None


def test_observed_axis_encoder_freezes_train_only_scaling() -> None:
    train = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-01", periods=5, freq="h", tz="UTC"),
            "fs__mkt_ret_eq_1h__mean": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "fs__rv_24h__mean": [0.1, 0.2, 0.3, 0.4, 0.5],
            "fs__ema50_slope__mean": [-0.5, -0.2, 0.0, 0.2, 0.5],
        }
    )
    # Extreme eval values would look ordinary if the transform refit on eval.
    eval_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-02", periods=2, freq="h", tz="UTC"),
            "fs__mkt_ret_eq_1h__mean": [100.0, 120.0],
            "fs__rv_24h__mean": [9.0, 10.0],
            "fs__ema50_slope__mean": [4.0, 5.0],
        }
    )

    encoder = mstc.fit_observed_axis_encoder(train, eval_frame)
    transformed = mstc.transform_observed_axes(eval_frame, encoder)
    _, built_eval, axis_sources = mstc.build_observed_axes(train, eval_frame)

    assert encoder["mode"] == "observed_axis_robust_z_v1"
    assert encoder["column_refs"]["fs__mkt_ret_eq_1h__mean"]["median"] == 0.0
    assert axis_sources["state_transition_pressure"] == ["mean_abs_state_axis_diff"]
    np.testing.assert_allclose(
        transformed["state_shock_up"].to_numpy(dtype=float),
        built_eval["state_shock_up"].to_numpy(dtype=float),
    )
    assert transformed["state_shock_up"].iloc[0] > 5.9
    assert transformed["state_realized_vol"].iloc[0] > 5.9
    assert "state_liquidity_stress" not in transformed.columns
    for col in (
        "state_input_coverage",
        "state_extreme_value_share",
        "state_novelty",
        "state_drift_score",
        "state_uncertainty",
        "state_low_input_coverage",
    ):
        assert col in transformed.columns
        assert transformed[col].between(0.0, 1.0).all()
        assert col in axis_sources
    assert transformed["state_input_coverage"].eq(1.0).all()
    assert transformed["state_low_input_coverage"].eq(0.0).all()
    assert transformed["state_extreme_value_share"].iloc[0] > 0.0
    assert transformed["state_novelty"].iloc[0] > 0.90


def test_observed_axis_encoder_emits_causal_spectral_position_features() -> None:
    ts = pd.date_range("2026-05-01", periods=36, freq="h", tz="UTC")
    x = np.linspace(-1.0, 1.0, len(ts))
    train = pd.DataFrame(
        {
            "timestamp": ts,
            "fs__mkt_ret_eq_1h__mean": x,
            "fs__market_breadth_1h__median": 0.5 * x,
            "fs__rv_24h__median": np.sin(np.arange(len(ts)) / 5.0),
            "fs__mkt_oi_chg_z_24h__median": np.cos(np.arange(len(ts)) / 6.0),
        }
    )
    eval_frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-05-03", tz="UTC")],
            "fs__mkt_ret_eq_1h__mean": [4.0],
            "fs__market_breadth_1h__median": [2.0],
            "fs__rv_24h__median": [0.2],
            "fs__mkt_oi_chg_z_24h__median": [0.1],
        }
    )

    encoder = mstc.fit_observed_axis_encoder(train, eval_frame)
    transformed = mstc.transform_observed_axes(eval_frame, encoder)

    assert "spectral_position" in encoder
    assert "state_spectral_eig_lambda1_share" in transformed.columns
    assert "state_spectral_pc1_z" in transformed.columns
    assert "state_spectral_top3_reconstruction_ratio" in transformed.columns
    assert encoder["axis_sources"]["state_spectral_pc1_z"]
    assert np.isfinite(
        transformed[
            [
                "state_spectral_eig_lambda1_share",
                "state_spectral_pc1_z",
                "state_spectral_top3_mahalanobis",
            ]
        ].to_numpy()
    ).all()


def test_observed_axis_encoder_neutralizes_low_input_coverage_without_refit() -> None:
    ts = pd.date_range("2026-05-01", periods=8, freq="h", tz="UTC")
    train = pd.DataFrame(
        {
            "timestamp": ts,
            "fs__mkt_ret_eq_1h__mean": np.linspace(-0.04, 0.04, len(ts)),
            "fs__rv_24h__mean": np.linspace(0.10, 0.30, len(ts)),
            "fs__volume_zscore_48h__mean": np.linspace(-1.0, 1.0, len(ts)),
            "fs__oi_value_1d_log_chg__mean": np.linspace(-0.02, 0.02, len(ts)),
        }
    )
    live_like = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-02", periods=2, freq="h", tz="UTC"),
            # One available reference column out of four should be treated as
            # insufficient market coverage even though state values are finite.
            "fs__mkt_ret_eq_1h__mean": [0.50, -0.50],
        }
    )

    encoder = mstc.fit_observed_axis_encoder(train, minimum_input_coverage=0.80)
    transformed = mstc.transform_observed_axes(live_like, encoder)

    assert encoder["minimum_input_coverage"] == 0.80
    assert transformed["state_input_coverage"].eq(0.25).all()
    assert transformed["state_low_input_coverage"].eq(1.0).all()
    assert transformed["state_uncertainty"].eq(1.0).all()
    mechanism_cols = [
        c
        for c in transformed.columns
        if c != "timestamp"
        and c.startswith("state_")
        and c not in mstc.OBSERVED_RELIABILITY_STATE_COLUMNS
    ]
    assert mechanism_cols
    assert transformed[mechanism_cols].eq(0.0).all().all()


def test_market_state_source_uses_feature_store_and_rejects_candidate_population_fields() -> None:
    ts = pd.date_range("2026-06-15", periods=3, freq="h", tz="UTC")
    candidate_agg = pd.DataFrame(
        {
            "timestamp": ts,
            "candidate_count": [10, 11, 12],
            "strategy_count": [2, 2, 2],
            "symbol_count": [8, 8, 9],
            "short_asset__rank_mean": [0.72, 0.74, 0.76],
            "short_asset__score_mean": [0.61, 0.62, 0.63],
            "fs_like_mkt_ret_eq_1h__mean": [0.01, -0.02, 0.03],
        }
    )
    feature_store_agg = pd.DataFrame(
        {
            "timestamp": ts,
            "fs__mkt_ret_eq_1h__mean": [0.01, -0.02, 0.03],
            "fs__rv_24h__median": [0.2, 0.3, 0.4],
        }
    )

    source, report = mstc._state_source_aggregate_frame(
        candidate_agg,
        feature_store_agg,
        allow_candidate_fallback=False,
    )

    assert report["source"] == "feature_store_market_aggregates"
    assert report["production_safe"] is True
    assert report["validation"]["market_wide_one_row_per_timestamp"] is True
    assert "candidate_count" not in source.columns
    assert "short_asset__rank_mean" not in source.columns
    assert "short_asset__score_mean" not in source.columns
    assert {"fs__mkt_ret_eq_1h__mean", "fs__rv_24h__median"}.issubset(source.columns)


def test_market_state_candidate_fallback_is_sanitized_debug_only() -> None:
    ts = pd.date_range("2026-06-15", periods=2, freq="h", tz="UTC")
    candidate_agg = pd.DataFrame(
        {
            "timestamp": ts,
            "candidate_count": [10, 12],
            "strategy_count": [2, 2],
            "symbol_count": [8, 9],
            "short_boll__rank_max": [0.9, 0.8],
            "short_boll__score_mean": [0.7, 0.6],
            "clean_mkt_volatility_proxy__mean": [0.3, 0.4],
        }
    )
    feature_store_agg = pd.DataFrame({"timestamp": ts})

    source, report = mstc._state_source_aggregate_frame(
        candidate_agg,
        feature_store_agg,
        allow_candidate_fallback=True,
    )

    assert report["source"] == "debug_candidate_population_fallback_sanitized"
    assert report["production_safe"] is False
    assert source.columns.tolist() == ["timestamp", "clean_mkt_volatility_proxy__mean"]


def test_market_state_column_contract_excludes_actual_order_book_but_keeps_proxies() -> None:
    cols = [
        "mkt_ret_eq_1h",
        "xasset_ob_liquidity_divergence_z_24h",
        "ob_top_liquidity_to_qv_24h",
        "median_spread_bps",
        "pct_assets_wide_spread",
        "spread_proxy_hl_range_bps_robust_z",
        "ema50_ema200_spread_atr",
        "cs_rank_ret_24h",
    ]

    filtered = mstc._filter_market_state_source_columns(cols)

    assert "mkt_ret_eq_1h" in filtered
    assert "spread_proxy_hl_range_bps_robust_z" in filtered
    assert "ema50_ema200_spread_atr" in filtered
    assert "cs_rank_ret_24h" in filtered
    assert "xasset_ob_liquidity_divergence_z_24h" not in filtered
    assert "ob_top_liquidity_to_qv_24h" not in filtered
    assert "median_spread_bps" not in filtered
    assert "pct_assets_wide_spread" not in filtered


def test_feature_store_aggregates_persist_universe_contract(tmp_path) -> None:
    ts = pd.date_range("2026-06-15", periods=3, freq="h", tz="UTC")
    for i, symbol in enumerate(["AAA", "BBB", "CCC"], start=1):
        pd.DataFrame(
            {
                "mkt_ret_eq_1h": np.linspace(0.01 * i, 0.03 * i, len(ts)),
                "rv_24h": np.linspace(0.1 * i, 0.3 * i, len(ts)),
            },
            index=ts,
        ).to_parquet(tmp_path / f"symbol={symbol}.parquet")

    agg, report = mstc._feature_store_timestamp_aggregates(
        tmp_path,
        pd.Series(ts),
        ["mkt_ret_eq_1h", "rv_24h"],
        symbol_cap=2,
    )

    contract = report["universe_contract"]
    assert contract["universe_definition_version"] == "feature_store_timestamp_market_state_v1"
    assert contract["available_symbol_count"] == 3
    assert contract["eligible_symbol_count"] == 2
    assert contract["eligible_symbols"] == ["AAA", "CCC"]
    assert contract["excluded_symbols"] == ["BBB"]
    assert contract["excluded_reasons"] == {"BBB": "symbol_cap_subsample"}
    assert report["symbols_read"] == 2
    assert "fs__mkt_ret_eq_1h__mean" in agg.columns
    assert "fs__mkt_ret_eq_1h__iqr" in agg.columns
    assert "fs__mkt_ret_eq_1h__robust_dispersion" in agg.columns
    assert "fs__mkt_ret_eq_1h__finite_share" in agg.columns
    assert "fs__mkt_ret_eq_1h__share_pos" in agg.columns
    assert "fs__mkt_ret_eq_1h__share_neg" in agg.columns
    assert "fs__mkt_ret_eq_1h__share_gt_train_q90" in agg.columns
    assert "fs__mkt_ret_eq_1h__share_lt_train_q10" in agg.columns
    assert report["aggregation_contract"] == "median,p10,p90,iqr,finite_coverage,breadth,basis_assets,train_reference_tail_shares"
    assert report["tail_reference_source"] == "self_window_reference"
    assert set(report["tail_reference_quantiles"]) == {"mkt_ret_eq_1h", "rv_24h"}


def test_feature_store_aggregates_vectorized_shares_ignore_nonfinite_values(tmp_path) -> None:
    ts = pd.date_range("2026-06-15", periods=2, freq="h", tz="UTC")
    values = {
        "BTC_USD:BTC": [1.0, 0.0],
        "ETH_USD:ETH": [-1.0, np.nan],
        "ZZZ": [np.inf, -2.0],
    }
    for symbol, series in values.items():
        pd.DataFrame({"mkt_ret_eq_1h": series}, index=ts).to_parquet(
            tmp_path / f"symbol={symbol}.parquet"
        )

    agg, _report = mstc._feature_store_timestamp_aggregates(
        tmp_path,
        pd.Series(ts),
        ["mkt_ret_eq_1h"],
        symbol_cap=0,
    )
    by_ts = agg.set_index("timestamp")

    assert float(by_ts.loc[ts[0], "fs__mkt_ret_eq_1h__finite_count"]) == 2.0
    assert float(by_ts.loc[ts[0], "fs__mkt_ret_eq_1h__finite_share"]) == pytest.approx(2 / 3)
    assert float(by_ts.loc[ts[0], "fs__mkt_ret_eq_1h__share_pos"]) == pytest.approx(0.5)
    assert float(by_ts.loc[ts[0], "fs__mkt_ret_eq_1h__share_neg"]) == pytest.approx(0.5)
    assert float(by_ts.loc[ts[0], "fs__mkt_ret_eq_1h__btc_value"]) == pytest.approx(1.0)
    assert float(by_ts.loc[ts[0], "fs__mkt_ret_eq_1h__eth_value"]) == pytest.approx(-1.0)
    assert float(by_ts.loc[ts[1], "fs__mkt_ret_eq_1h__share_pos"]) == pytest.approx(0.0)
    assert float(by_ts.loc[ts[1], "fs__mkt_ret_eq_1h__share_neg"]) == pytest.approx(0.5)


def test_feature_store_aggregate_pair_uses_training_tail_reference(tmp_path) -> None:
    train_ts = pd.date_range("2026-06-10", periods=3, freq="h", tz="UTC")
    eval_ts = pd.date_range("2026-06-15", periods=2, freq="h", tz="UTC")
    all_ts = train_ts.append(eval_ts)
    values = {
        "BTC_USD:BTC": [0.0, 0.1, 0.2, 10.0, 11.0],
        "ETH_USD:ETH": [0.0, 0.2, 0.4, 12.0, 13.0],
    }
    for symbol, series in values.items():
        pd.DataFrame(
            {
                "mkt_ret_eq_1h": series,
                "oi_value": [100.0, 100.0, 100.0, 200.0, 200.0],
            },
            index=all_ts,
        ).to_parquet(tmp_path / f"symbol={symbol}.parquet")

    train_agg, train_report, eval_agg, eval_report = mstc._feature_store_timestamp_aggregate_pair(
        tmp_path,
        tmp_path,
        pd.Series(train_ts),
        pd.Series(eval_ts),
        ["mkt_ret_eq_1h", "oi_value"],
        symbol_cap=0,
    )

    assert train_report["tail_reference_role"] == "fit_on_training_timestamps"
    assert eval_report["tail_reference_source"] == "provided_train_reference"
    assert eval_report["tail_reference_role"] == "transformed_with_training_timestamp_reference"
    assert "fs__mkt_ret_eq_1h__btc_value" in eval_agg.columns
    assert "fs__mkt_ret_eq_1h__eth_value" in eval_agg.columns
    assert "fs__mkt_ret_eq_1h__oi_weighted_mean" in eval_agg.columns
    assert float(eval_agg["fs__mkt_ret_eq_1h__share_gt_train_q90"].min()) == 1.0
    assert float(train_agg["fs__mkt_ret_eq_1h__share_gt_train_q90"].max()) <= 0.5


def test_feature_store_aggregate_pair_uses_common_frozen_symbol_universe(tmp_path) -> None:
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    train_dir.mkdir()
    eval_dir.mkdir()
    train_ts = pd.date_range("2026-06-10", periods=2, freq="h", tz="UTC")
    eval_ts = pd.date_range("2026-06-15", periods=2, freq="h", tz="UTC")

    for directory, symbols, ts in (
        (train_dir, ["AAA", "BBB", "CCC"], train_ts),
        (eval_dir, ["AAA", "CCC", "DDD"], eval_ts),
    ):
        for i, symbol in enumerate(symbols, start=1):
            pd.DataFrame(
                {
                    "mkt_ret_eq_1h": np.linspace(0.01 * i, 0.02 * i, len(ts)),
                    "rv_24h": np.linspace(0.10 * i, 0.20 * i, len(ts)),
                },
                index=ts,
            ).to_parquet(directory / f"symbol={symbol}.parquet")

    _train_agg, train_report, eval_agg, eval_report = mstc._feature_store_timestamp_aggregate_pair(
        train_dir,
        eval_dir,
        pd.Series(train_ts),
        pd.Series(eval_ts),
        ["mkt_ret_eq_1h", "rv_24h"],
        symbol_cap=0,
    )

    train_universe = train_report["universe_contract"]
    eval_universe = eval_report["universe_contract"]
    assert train_report["frozen_eligible_symbol_source"] == "common_train_eval_feature_store_symbols"
    assert eval_report["frozen_eligible_symbol_source"] == "common_train_eval_feature_store_symbols"
    assert train_universe["eligible_symbols"] == ["AAA", "CCC"]
    assert eval_universe["eligible_symbols"] == ["AAA", "CCC"]
    assert train_universe["eligible_symbol_coverage"] == 1.0
    assert eval_universe["eligible_symbol_coverage"] == 1.0
    assert train_universe["excluded_reasons"] == {"BBB": "outside_frozen_eligible_universe"}
    assert eval_universe["excluded_reasons"] == {"DDD": "outside_frozen_eligible_universe"}
    assert eval_agg["fs__mkt_ret_eq_1h__finite_share"].eq(1.0).all()


def test_materialized_state_artifacts_use_training_tail_reference_for_eval(tmp_path, monkeypatch) -> None:
    train_ts = pd.date_range("2026-06-10", periods=4, freq="h", tz="UTC")
    eval_ts = pd.date_range("2026-06-15", periods=2, freq="h", tz="UTC")
    all_ts = train_ts.append(eval_ts)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for symbol, offset in {"AAA": 0.0, "BBB": 0.1}.items():
        pd.DataFrame(
            {
                "mkt_ret_eq_1h": list(np.linspace(0.0 + offset, 0.3 + offset, len(train_ts)))
                + [10.0 + offset, 11.0 + offset],
                "rv_24h": list(np.linspace(0.1 + offset, 0.4 + offset, len(train_ts)))
                + [12.0 + offset, 13.0 + offset],
            },
            index=all_ts,
        ).to_parquet(feature_dir / f"symbol={symbol}.parquet")

    def _candidates(ts: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": ts,
                "strategy_id": "strategy_a",
                "head": "short_asset",
                "symbol": "AAA",
                "side": "short",
                "normalized_rank_score": 0.75,
                "base_strategy_threshold": 0.70,
                "deployment_rank_threshold": 0.70,
                "calibrated_score": 0.65,
                "entry_price": 100.0,
                "exit_timestamp": ts + pd.Timedelta(hours=1),
                "exit_price": 100.0,
                "net_return": 0.0,
                "gross_return": 0.0,
                "holding_bars": 1,
                "simple_policy_exit_reason": "timeout",
            }
        )

    monkeypatch.setattr(
        mstc,
        "fit_forecast_state_heads",
        lambda train_observed, **kwargs: (
            train_observed.copy(),
            {"mode": "test_forecast_noop"},
            {"mode": "test_forecast_noop"},
        ),
    )
    monkeypatch.setattr(
        mstc,
        "transform_forecast_state_heads",
        lambda observed, artifact, agg=None: observed.copy(),
    )
    monkeypatch.setattr(
        mstc,
        "fit_latent_state_probs",
        lambda train_forecast, n_states: (
            train_forecast.copy(),
            {"mode": "test_latent_noop"},
            {"mode": "test_latent_noop"},
        ),
    )
    monkeypatch.setattr(
        mstc,
        "transform_latent_state_probs",
        lambda forecast, artifact: forecast.copy(),
    )

    artifacts = mat._build_state_artifacts(
        _candidates(train_ts),
        _candidates(eval_ts),
        train_feature_store_dir=feature_dir,
        eval_feature_store_dir=feature_dir,
        max_feature_cols=8,
        max_feature_store_cols=2,
        feature_store_symbol_cap=0,
        allow_candidate_state_fallback=False,
        forecast_horizons_steps=(6,),
        forecast_model_kind="lightgbm",
        latent_states=2,
    )

    feature_store_report = artifacts["reports"]["feature_store"]
    assert feature_store_report["train"]["tail_reference_role"] == "fit_on_training_timestamps"
    assert feature_store_report["eval"]["tail_reference_source"] == "provided_train_reference"
    assert feature_store_report["eval"]["tail_reference_role"] == "transformed_with_training_timestamp_reference"
    assert set(artifacts["feature_store_tail_reference_quantiles"]) == {"mkt_ret_eq_1h", "rv_24h"}
    eval_source = artifacts["eval_state_source"]
    assert float(eval_source["fs__mkt_ret_eq_1h__share_gt_train_q90"].min()) == 1.0

    eval_observed = artifacts["states"]["observed"][1].sort_values("timestamp").reset_index(drop=True)
    bundle = {
        "selected_arm": "S1_observed_axes_shared_response",
        "selected_controller": {"selected_arm": "S1_observed_axes_shared_response"},
        "controller_execution_enabled": True,
        "state_spec": {"state_level": "observed"},
        "models": {
            "curves": _FlatCurves(),
            "dummy_columns": [],
            "shared": {"eu_mean": _ZeroModel(), "eu_q10": _ZeroModel()},
            "risk": {},
            "residual": {},
            "risk_baseline": {"residual_scale": 1.0},
            "state_ood_reference": {},
        },
        "response_feature_columns": ["normalized_rank_score"],
        "state_feature_columns": [c for c in eval_observed.columns if c != "timestamp"],
        "candidate_feature_columns": artifacts["candidate_feature_cols"],
        "feature_store_columns": artifacts["feature_store_cols"],
        "feature_store_tail_reference_quantiles": artifacts["feature_store_tail_reference_quantiles"],
        "observed_axis_encoder": artifacts["observed_axis_encoder"],
        "controller_params": {
            "threshold_delta_max": 0.08,
            "max_threshold_up_step": 0.08,
            "threshold_relax_alpha": 0.10,
            "controller_mode": "rank_grid",
            "controller_min_prediction_coverage": 0.80,
            "controller_min_usable_candidates": 1,
        },
    }
    _scored, _predictions, _schedule, scored_state, score_report, _proposed_schedule = score_bundle.score_candidates(
        bundle=bundle,
        candidates=_candidates(eval_ts),
        feature_store_dir=feature_dir,
        feature_store_symbol_cap=0,
        allow_candidate_state_fallback=False,
    )
    scored_state = scored_state.sort_values("timestamp").reset_index(drop=True)

    assert score_report["feature_store"]["tail_reference_source"] == "provided_train_reference"
    assert score_report["feature_store"]["tail_reference_role"] == "transformed_with_bundle_training_reference"
    pd.testing.assert_series_equal(scored_state["timestamp"], eval_observed["timestamp"])
    for col in [c for c in eval_observed.columns if c != "timestamp"]:
        np.testing.assert_allclose(
            scored_state[col].to_numpy(dtype=float),
            eval_observed[col].to_numpy(dtype=float),
            atol=1e-12,
            rtol=1e-12,
        )


def test_observed_axis_encoder_rejects_model_rank_state_sources() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-15", periods=2, freq="h", tz="UTC"),
            "fs__mkt_ret_eq_1h__mean": [0.01, -0.01],
            "normalized_rank_score": [0.8, 0.9],
        }
    )

    with pytest.raises(ValueError, match="forbidden columns"):
        mstc.fit_observed_axis_encoder(frame)


def test_latent_state_probs_include_duration_and_hazard_diagnostics_without_raw_ids() -> None:
    n_train = 90
    n_eval = 18
    ts_train = pd.date_range("2026-04-01", periods=n_train, freq="h", tz="UTC")
    ts_eval = pd.date_range("2026-04-05", periods=n_eval, freq="h", tz="UTC")
    x_train = np.linspace(0.0, 9.0, n_train)
    x_eval = np.linspace(9.0, 11.0, n_eval)
    train_state = pd.DataFrame(
        {
            "timestamp": ts_train,
            "state_shock": np.sin(x_train),
            "state_trend": np.cos(x_train * 0.7),
            "forecast_h6_shock_up": np.clip((np.sin(x_train + 0.5) + 1.0) / 2.0, 0.0, 1.0),
        }
    )
    eval_state = pd.DataFrame(
        {
            "timestamp": ts_eval,
            "state_shock": np.sin(x_eval),
            "state_trend": np.cos(x_eval * 0.7),
            "forecast_h6_shock_up": np.clip((np.sin(x_eval + 0.5) + 1.0) / 2.0, 0.0, 1.0),
        }
    )

    train_latent, eval_latent, report = mstc.add_latent_state_probs(train_state, eval_state, n_states=3)

    prob_cols = [f"latent_gmm_p{i}" for i in range(3)]
    for frame in (train_latent, eval_latent):
        assert set(prob_cols).issubset(frame.columns)
        np.testing.assert_allclose(frame[prob_cols].sum(axis=1).to_numpy(dtype=float), 1.0, atol=1e-6)
        for col in (
            "latent_entropy",
            "latent_max_prob",
            "latent_transition_pressure",
            "latent_transition_hazard",
            "latent_time_since_state_change",
            "latent_time_since_state_change_log_norm",
            "latent_expected_duration",
            "latent_regime_maturity",
        ):
            assert col in frame.columns
            assert np.isfinite(frame[col].to_numpy(dtype=float)).all()
        assert "latent_hard_state" not in frame.columns
    assert report["latent_feature_contract"] == "probabilities_entropy_transition_hazard_duration_no_raw_cluster_ids"
    assert report["hard_state_ids_not_semantic"] is True
    assert "state_mean_duration" in report
    assert "state_transition_hazard" in report


def test_latent_artifact_replays_probabilities_without_refit(tmp_path) -> None:
    import joblib
    from scripts import score_market_state_controller_bundle as score_bundle

    n_train = 90
    n_eval = 15
    ts_train = pd.date_range("2026-04-01", periods=n_train, freq="h", tz="UTC")
    ts_eval = pd.date_range("2026-04-05", periods=n_eval, freq="h", tz="UTC")
    x_train = np.linspace(0.0, 9.0, n_train)
    x_eval = np.linspace(9.0, 10.5, n_eval)
    train_state = pd.DataFrame(
        {
            "timestamp": ts_train,
            "state_shock": np.sin(x_train),
            "state_trend": np.cos(0.4 * x_train),
            "forecast_h6_shock_up": np.clip((np.sin(x_train + 0.2) + 1.0) / 2.0, 0.0, 1.0),
        }
    )
    eval_state = pd.DataFrame(
        {
            "timestamp": ts_eval,
            "state_shock": np.sin(x_eval),
            "state_trend": np.cos(0.4 * x_eval),
            "forecast_h6_shock_up": np.clip((np.sin(x_eval + 0.2) + 1.0) / 2.0, 0.0, 1.0),
        }
    )

    train_fit, artifact, report = mstc.fit_latent_state_probs(train_state, 3)
    eval_replay = mstc.transform_latent_state_probs(eval_state, artifact)
    train_legacy, eval_legacy, legacy_report = mstc.add_latent_state_probs(train_state, eval_state, 3)
    prob_cols = [f"latent_gmm_p{i}" for i in range(3)]

    assert report["latent_feature_contract"] == legacy_report["latent_feature_contract"]
    np.testing.assert_allclose(train_fit[prob_cols].to_numpy(float), train_legacy[prob_cols].to_numpy(float))
    np.testing.assert_allclose(eval_replay[prob_cols].to_numpy(float), eval_legacy[prob_cols].to_numpy(float))

    bundle_path = tmp_path / "latent_bundle.joblib"
    joblib.dump(
        {
            "state_spec": {"state_level": "latent"},
            "models": {},
            "response_feature_columns": [],
            "state_feature_columns": [c for c in train_fit.columns if c != "timestamp"],
            "feature_store_columns": [],
            "observed_axis_encoder": _tiny_observed_axis_encoder(),
            "forecast_artifact": {"mode": "primitive_future_soft_severity_regressors_v1", "targets": {}},
            "latent_artifact": artifact,
        },
        bundle_path,
    )
    loaded = score_bundle._load_bundle(bundle_path)
    assert loaded["state_spec"]["state_level"] == "latent"


def test_scoring_rejects_default_selected_arm_bundle_without_override(tmp_path) -> None:
    import joblib
    from scripts import score_market_state_controller_bundle as score_bundle

    bundle_path = tmp_path / "debug_default_bundle.joblib"
    joblib.dump(
        {
            "selected_arm": "S1_observed_axes_shared_response",
            "selected_controller": {
                "selected_arm": "S1_observed_axes_shared_response",
                "selected_arm_default_used": True,
                "reason": "no_arm_passed_selection_gates",
            },
            "state_spec": {"state_level": "observed"},
            "market_state_feature_contract": {"contract_version": "market_state_feature_contract_v1"},
            "models": {},
            "response_feature_columns": [],
            "state_feature_columns": [],
            "feature_store_columns": [],
            "observed_axis_encoder": _tiny_observed_axis_encoder(),
        },
        bundle_path,
    )

    with pytest.raises(RuntimeError, match="selected_arm_default_used=true"):
        score_bundle._load_bundle(bundle_path)

    loaded = score_bundle._load_bundle(bundle_path, allow_selected_arm_default_bundle=True)
    assert loaded["selected_arm"] == "S1_observed_axes_shared_response"
    assert loaded["market_state_feature_contract"]["contract_version"] == "market_state_feature_contract_v1"


def test_scoring_rejects_bundle_with_dropped_activation_state_feature(tmp_path) -> None:
    import joblib
    from scripts import score_market_state_controller_bundle as score_bundle

    activation_filter = {
        "enforced": True,
        "reason": "activation_registry_active_candidate_filter",
        "active_state_feature_columns": ["state_shock"],
        "dropped_state_feature_columns": ["forecast_h6_shock_down"],
    }
    bundle_path = tmp_path / "stale_activation_bundle.joblib"
    joblib.dump(
        {
            "selected_arm": "S1_observed_axes_shared_response",
            "selected_controller": {"selected_arm": "S1_observed_axes_shared_response"},
            "controller_execution_enabled": False,
            "state_spec": {"state_level": "observed"},
            "market_state_feature_contract": {
                "contract_version": "market_state_feature_contract_v1",
                "state_activation_filter": activation_filter,
                "source_schema": {
                    "state_feature_columns": ["state_shock", "forecast_h6_shock_down"],
                    "response_feature_columns": ["rank", "forecast_h6_shock_down"],
                },
            },
            "state_activation_filter": activation_filter,
            "models": {},
            "response_feature_columns": ["rank", "forecast_h6_shock_down"],
            "state_feature_columns": ["state_shock", "forecast_h6_shock_down"],
            "feature_store_columns": [],
            "observed_axis_encoder": {"mode": "observed_axis_robust_z_v1"},
        },
        bundle_path,
    )

    with pytest.raises(ValueError, match="dropped activation-registry state features"):
        score_bundle._load_bundle(bundle_path)


def test_scoring_accepts_activation_filtered_bundle(tmp_path) -> None:
    import joblib
    from scripts import score_market_state_controller_bundle as score_bundle

    activation_filter = {
        "enforced": True,
        "reason": "activation_registry_active_candidate_filter",
        "active_state_feature_columns": ["state_shock"],
        "dropped_state_feature_columns": ["forecast_h6_shock_down"],
    }
    bundle_path = tmp_path / "filtered_activation_bundle.joblib"
    joblib.dump(
        {
            "selected_arm": "S1_observed_axes_shared_response",
            "selected_controller": {"selected_arm": "S1_observed_axes_shared_response"},
            "controller_execution_enabled": False,
            "state_spec": {"state_level": "observed"},
            "market_state_feature_contract": {
                "contract_version": "market_state_feature_contract_v1",
                "state_activation_filter": activation_filter,
                "source_schema": {
                    "state_feature_columns": ["state_shock"],
                    "response_feature_columns": ["rank", "state_shock"],
                },
            },
            "state_activation_filter": activation_filter,
            "models": {},
            "response_feature_columns": ["rank", "state_shock"],
            "state_feature_columns": ["state_shock"],
            "feature_store_columns": [],
            "observed_axis_encoder": {"mode": "observed_axis_robust_z_v1"},
        },
        bundle_path,
    )

    loaded = score_bundle._load_bundle(bundle_path)

    report = loaded["state_activation_filter_validation"]
    assert report["available"] is True
    assert report["enforced"] is True
    assert report["active_state_feature_count"] == 1
    assert report["dropped_state_feature_count"] == 1


def test_scoring_rejects_executable_bundle_without_observed_train_references(tmp_path) -> None:
    import joblib
    from scripts import score_market_state_controller_bundle as score_bundle

    bundle_path = tmp_path / "legacy_executable_bundle.joblib"
    joblib.dump(
        {
            "selected_arm": "S1_observed_axes_shared_response",
            "selected_controller": {"selected_arm": "S1_observed_axes_shared_response"},
            "controller_execution_enabled": True,
            "state_spec": {"state_level": "observed"},
            "market_state_feature_contract": {"contract_version": "market_state_feature_contract_v1"},
            "models": {},
            "response_feature_columns": [],
            "state_feature_columns": ["state_shock"],
            "feature_store_columns": [],
            "observed_axis_encoder": {"mode": "observed_axis_robust_z_v1"},
        },
        bundle_path,
    )

    with pytest.raises(ValueError, match="minimum_input_coverage|missing train column_refs"):
        score_bundle._load_bundle(bundle_path)


def test_scoring_rejects_executable_bundle_without_feature_store_tail_reference(tmp_path) -> None:
    import joblib
    from scripts import score_market_state_controller_bundle as score_bundle

    bundle_path = tmp_path / "missing_tail_reference_bundle.joblib"
    joblib.dump(
        {
            "selected_arm": "S1_observed_axes_shared_response",
            "selected_controller": {"selected_arm": "S1_observed_axes_shared_response"},
            "controller_execution_enabled": True,
            "state_spec": {"state_level": "observed"},
            "market_state_feature_contract": {"contract_version": "market_state_feature_contract_v1"},
            "models": {},
            "response_feature_columns": [],
            "state_feature_columns": ["state_shock"],
            "feature_store_columns": ["mkt_ret_eq_1h"],
            "observed_axis_encoder": _tiny_observed_axis_encoder(),
        },
        bundle_path,
    )

    with pytest.raises(ValueError, match="feature_store_tail_reference_quantiles"):
        score_bundle._load_bundle(bundle_path)


def test_bundle_scoring_replays_low_input_coverage_fallback(tmp_path) -> None:
    eval_ts = pd.date_range("2026-06-15", periods=2, freq="h", tz="UTC")
    candidates = pd.DataFrame(
        {
            "timestamp": np.repeat(eval_ts, 2),
            "strategy_id": "strategy_a",
            "head": "short_asset",
            "symbol": ["AAA", "BBB"] * len(eval_ts),
            "side": "short",
            "normalized_rank_score": [0.72, 0.84, 0.73, 0.86],
            "base_strategy_threshold": 0.70,
            "deployment_rank_threshold": 0.70,
            "calibrated_score": [0.60, 0.70, 0.61, 0.71],
            "entry_price": 100.0,
            "exit_timestamp": np.repeat(eval_ts + pd.Timedelta(hours=1), 2),
            "exit_price": 100.0,
            "net_return": [0.0, 0.0, 0.0, 0.0],
            "gross_return": [0.0, 0.0, 0.0, 0.0],
            "holding_bars": 1,
            "simple_policy_exit_reason": "timeout",
        }
    )
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    pd.DataFrame(
        {
            "mkt_ret_eq_1h": [0.01, -0.02],
            # Deliberately omit rv_24h and mkt_oi_chg_z_24h. The frozen
            # encoder expects them, so raw input coverage must fail closed.
        },
        index=eval_ts,
    ).to_parquet(feature_dir / "symbol=AAA.parquet")

    bundle = {
        "selected_arm": "S1_observed_axes_shared_response",
        "selected_controller": {"selected_arm": "S1_observed_axes_shared_response"},
        "controller_execution_enabled": True,
        "state_spec": {"state_level": "observed"},
        "models": {
            "curves": _FlatCurves(),
            "dummy_columns": [],
            "shared": {"eu_mean": _ZeroModel(), "eu_q10": _ZeroModel()},
            "risk": {},
            "residual": {},
            "risk_baseline": {"residual_scale": 1.0},
            "state_ood_reference": {},
        },
        "response_feature_columns": ["normalized_rank_score", "state_shock", "state_uncertainty"],
        "state_feature_columns": ["state_shock", "state_uncertainty", "state_low_input_coverage"],
        "candidate_feature_columns": [],
        "feature_store_columns": ["mkt_ret_eq_1h", "rv_24h", "mkt_oi_chg_z_24h"],
        "feature_store_tail_reference_quantiles": {
            "mkt_ret_eq_1h": {"q10": -0.010, "q90": 0.010},
            "rv_24h": {"q10": 0.015, "q90": 0.040},
            "mkt_oi_chg_z_24h": {"q10": -0.8, "q90": 0.8},
        },
        "observed_axis_encoder": _tiny_observed_axis_encoder(minimum_input_coverage=0.80),
        "controller_params": {
            "threshold_delta_max": 0.08,
            "max_threshold_up_step": 0.08,
            "threshold_relax_alpha": 0.10,
            "controller_mode": "rank_grid",
            "controller_min_prediction_coverage": 0.80,
            "controller_min_usable_candidates": 1,
        },
    }

    with pytest.raises(RuntimeError, match="Refusing to score an executable or shadow"):
        score_bundle.score_candidates(
            bundle=bundle,
            candidates=candidates,
            feature_store_dir=feature_dir,
            feature_store_symbol_cap=10,
            allow_candidate_state_fallback=True,
        )

    scored, predictions, schedule, state, report, proposed_schedule = score_bundle.score_candidates(
        bundle=bundle,
        candidates=candidates,
        feature_store_dir=feature_dir,
        feature_store_symbol_cap=10,
        allow_candidate_state_fallback=False,
    )

    assert report["observed_axis_encoder_validation"]["minimum_input_coverage"] == 0.80
    assert proposed_schedule.empty
    assert report["feature_store"]["tail_reference_source"] == "provided_train_reference"
    assert report["feature_store"]["tail_reference_role"] == "transformed_with_bundle_training_reference"
    assert report["feature_store_tail_reference_validation"]["tail_reference_quantile_count"] == 3
    assert state["state_input_coverage"].lt(0.80).all()
    assert state["state_low_input_coverage"].eq(1.0).all()
    assert state["state_uncertainty"].eq(1.0).all()
    mechanism_cols = [
        c
        for c in state.columns
        if c.startswith("state_")
        and c
        not in {
            "state_input_coverage",
            "state_extreme_value_share",
            "state_novelty",
            "state_drift_score",
            "state_uncertainty",
            "state_low_input_coverage",
        }
    ]
    assert mechanism_cols
    assert state[mechanism_cols].eq(0.0).all().all()
    assert predictions["state_low_input_coverage"].eq(1.0).all()
    assert schedule["controller_reason"].eq("low_input_coverage_fallback").all()
    assert schedule["state_threshold"].eq(schedule["base_threshold"]).all()
    assert scored["base_strategy_threshold"].eq(0.70).all()


def test_shadow_controller_scoring_persists_proposals_without_execution(tmp_path) -> None:
    eval_ts = pd.date_range("2026-06-15", periods=3, freq="h", tz="UTC")
    candidates = pd.DataFrame(
        {
            "timestamp": np.repeat(eval_ts, 2),
            "strategy_id": ["strategy_asset", "strategy_boll"] * len(eval_ts),
            "head": ["short_asset", "short_boll"] * len(eval_ts),
            "symbol": ["AAA", "BBB"] * len(eval_ts),
            "side": "short",
            "normalized_rank_score": [0.72, 0.83] * len(eval_ts),
            "base_strategy_threshold": 0.70,
            "deployment_rank_threshold": 0.70,
            "calibrated_score": [0.62, 0.71] * len(eval_ts),
            "entry_price": 100.0,
            "exit_timestamp": np.repeat(eval_ts + pd.Timedelta(hours=1), 2),
            "exit_price": 100.0,
            "net_return": [-0.02, 0.03] * len(eval_ts),
            "gross_return": [-0.015, 0.035] * len(eval_ts),
            "holding_bars": 1,
            "simple_policy_exit_reason": ["sl", "tp"] * len(eval_ts),
        }
    )
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    pd.DataFrame(
        {
            "mkt_ret_eq_1h": [0.01, -0.02, 0.00],
            "rv_24h": [0.020, 0.025, 0.030],
            "mkt_oi_chg_z_24h": [0.1, -0.2, 0.0],
        },
        index=eval_ts,
    ).to_parquet(feature_dir / "symbol=AAA.parquet")

    bundle = {
        "selected_arm": "S1_observed_axes_shared_response",
        "selected_controller": {
            "selected_arm": "S1_observed_axes_shared_response",
            "selected_arm_shadow_used": True,
            "shadow_controller_only": True,
        },
        "controller_execution_enabled": False,
        "shadow_controller_only": True,
        "shadow_controller_enabled_heads": ["short_asset", "short_boll"],
        "state_spec": {"state_level": "observed"},
        "models": {
            "curves": _FlatCurves(),
            "dummy_columns": [],
            "shared": {
                "eu_mean": _ConstantModel(-0.02),
                "eu_q10": _ConstantModel(-0.04),
            },
            "risk": {},
            "residual": {},
            "risk_baseline": {"residual_scale": 1.0},
            "state_ood_reference": {},
        },
        "response_feature_columns": ["normalized_rank_score", "state_uncertainty"],
        "state_feature_columns": ["state_uncertainty", "state_input_coverage"],
        "candidate_feature_columns": [],
        "feature_store_columns": ["mkt_ret_eq_1h", "rv_24h", "mkt_oi_chg_z_24h"],
        "feature_store_tail_reference_quantiles": {
            "mkt_ret_eq_1h": {"q10": -0.010, "q90": 0.010},
            "rv_24h": {"q10": 0.015, "q90": 0.040},
            "mkt_oi_chg_z_24h": {"q10": -0.8, "q90": 0.8},
        },
        "observed_axis_encoder": _tiny_observed_axis_encoder(minimum_input_coverage=0.80),
        "controller_params": {
            "execution_enabled": False,
            "shadow_controller_only": True,
            "threshold_delta_max": 0.08,
            "max_threshold_up_step": 0.08,
            "threshold_relax_alpha": 0.10,
            "controller_mode": "rank_grid",
            "controller_min_prediction_coverage": 0.80,
            "controller_min_usable_candidates": 1,
        },
    }

    scored, _predictions, schedule, _state, report, proposed_schedule = score_bundle.score_candidates(
        bundle=bundle,
        candidates=candidates,
        feature_store_dir=feature_dir,
        feature_store_symbol_cap=10,
        allow_candidate_state_fallback=False,
    )

    assert report["controller_execution_enabled"] is False
    assert report["shadow_controller_only"] is True
    assert report["shadow_proposed_schedule_rows"] == len(proposed_schedule)
    assert report["shadow_threshold_raised_count"] > 0
    assert schedule["state_threshold"].eq(schedule["base_threshold"]).all()
    assert schedule["controller_reason"].eq("head_not_enabled_for_threshold_action").all()
    assert proposed_schedule["state_threshold"].gt(proposed_schedule["base_threshold"]).any()
    assert scored["base_strategy_threshold"].eq(0.70).all()


def test_scoring_accepts_explicit_rejected_noop_bundle(tmp_path) -> None:
    import joblib
    from scripts import score_market_state_controller_bundle as score_bundle

    bundle_path = tmp_path / "noop_bundle.joblib"
    joblib.dump(
        {
            "selected_arm": mat.NOOP_CONTROLLER_ARM,
            "selected_controller": {
                "selected_arm": None,
                "selected_arm_noop_used": True,
                "selected_arm_default_used": False,
                "reason": "no_arm_passed_selection_gates",
            },
            "controller_execution_enabled": False,
            "state_spec": {"state_level": "observed", "controller_noop": True},
            "market_state_feature_contract": {"contract_version": "market_state_feature_contract_v1"},
            "models": {},
            "response_feature_columns": [],
            "state_feature_columns": [],
            "feature_store_columns": [],
            "observed_axis_encoder": {"mode": "observed_axis_robust_z_v1"},
        },
        bundle_path,
    )

    loaded = score_bundle._load_bundle(bundle_path)
    assert loaded["selected_arm"] == mat.NOOP_CONTROLLER_ARM
    assert loaded["controller_execution_enabled"] is False


def test_materialized_bundle_arm_spec_uses_selected_s1_contract(tmp_path) -> None:
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(
        '{"selected_arm": "S1_observed_axes_shared_response", "selected_metrics": {"median_delta_net_pnl": 1.5}}',
        encoding="utf-8",
    )

    selected, payload = mat._load_selected_arm(selected_path, "S2_observed_forecast_shared_response")
    spec = mat._arm_state_spec(selected)

    assert selected == "S1_observed_axes_shared_response"
    assert payload["exists"] is True
    assert payload["selected_arm_default_used"] is False
    assert spec == {
        "state_level": "observed",
        "per_strategy_residual": False,
        "controller_noop": False,
        "controller_no_backfill_overlay": False,
        "base_arm": "S1_observed_axes_shared_response",
    }


def test_materialized_bundle_arm_spec_accepts_no_backfill_overlay() -> None:
    spec = mat._arm_state_spec("S1_observed_axes_shared_response__post_selection_overlay")

    assert spec == {
        "state_level": "observed",
        "per_strategy_residual": False,
        "controller_noop": False,
        "controller_no_backfill_overlay": True,
        "base_arm": "S1_observed_axes_shared_response",
    }


def test_materialized_bundle_rejects_null_selected_arm_without_explicit_override(tmp_path) -> None:
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(
        '{"selected_arm": null, "reason": "no_arm_passed_selection_gates"}',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="no_arm_passed_selection_gates"):
        mat._load_selected_arm(
            selected_path,
            "S1_observed_axes_shared_response",
            allow_default=False,
        )

    selected, payload = mat._load_selected_arm(
        selected_path,
        "S1_observed_axes_shared_response",
        allow_default=True,
    )
    assert selected == "S1_observed_axes_shared_response"
    assert payload["selected_arm_default_used"] is True

    selected, payload = mat._load_selected_arm(
        selected_path,
        "S1_observed_axes_shared_response",
        allow_null_noop=True,
    )
    assert selected == mat.NOOP_CONTROLLER_ARM
    assert payload["selected_arm"] is None
    assert payload["selected_arm_default_used"] is False
    assert payload["selected_arm_noop_used"] is True
    assert payload["noop_reason"] == "no_arm_passed_selection_gates"
    assert mat._arm_state_spec(selected) == {
        "state_level": "observed",
        "per_strategy_residual": False,
        "controller_noop": True,
        "controller_no_backfill_overlay": False,
        "base_arm": mat.NOOP_CONTROLLER_ARM,
    }

    shadow_selected, shadow_payload = mat._apply_shadow_selected_arm_override(
        selected_arm=selected,
        selected_payload=payload,
        shadow_selected_arm="S2_observed_forecast_shared_response",
    )
    assert shadow_selected == "S2_observed_forecast_shared_response"
    assert shadow_payload["selected_arm"] == "S2_observed_forecast_shared_response"
    assert shadow_payload["selected_arm_shadow_used"] is True
    assert shadow_payload["shadow_controller_only"] is True
    assert shadow_payload["shadow_source_noop_arm"] == mat.NOOP_CONTROLLER_ARM


def test_shadow_selected_arm_override_rejects_different_promoted_arm() -> None:
    with pytest.raises(RuntimeError, match="already promoted"):
        mat._apply_shadow_selected_arm_override(
            selected_arm="S1_observed_axes_shared_response",
            selected_payload={"selected_arm": "S1_observed_axes_shared_response"},
            shadow_selected_arm="S2_observed_forecast_shared_response",
        )


def test_materialized_bundle_rejects_missing_selection_without_explicit_override(tmp_path) -> None:
    selected_path = tmp_path / "missing.json"

    with pytest.raises(RuntimeError, match="missing"):
        mat._load_selected_arm(
            selected_path,
            "S1_observed_axes_shared_response",
            allow_default=False,
        )


def test_materialized_bundle_manifest_records_live_contract(tmp_path) -> None:
    class Args:
        train_broad_candidates = tmp_path / "train_broad.parquet"
        train_deployable_candidates = tmp_path / "train_deployable.parquet"
        eval_candidates = tmp_path / "eval.parquet"
        policy_manifest = tmp_path / "policy.json"
        policy_variant = "refit_bar4_strategy_bar2"
        rank_contract = "short_boll_timestamp_rank"
        disable_heads = "long_bars,long_dist"
        controller_enabled_heads = ""
        threshold_delta_max = 0.10
        max_threshold_up_step = 0.03
        threshold_relax_alpha = 0.25
        controller_mode = "rank_grid"
        controller_min_lcb_utility = 0.0
        controller_min_prediction_coverage = 0.80
        controller_min_usable_candidates = 1
        controller_max_state_ood_score = None
        controller_min_action_edge = 0.0
        controller_winner_sacrifice_multiplier = 1.0
        use_timeout_cap = False
        allow_candidate_state_fallback = False

    observed_encoder = _tiny_observed_axis_encoder()
    state_artifacts = {
        "candidate_feature_cols": ["mkt_ret_eq_1h"],
        "feature_store_cols": ["mkt_ret_eq_1h", "rv_24h"],
        "feature_store_tail_reference_quantiles": {
            "mkt_ret_eq_1h": {"q10": -0.01, "q90": 0.01},
            "rv_24h": {"q10": 0.10, "q90": 0.40},
        },
        "states": {
            "observed": (
                pd.DataFrame({"timestamp": pd.date_range("2026-06-01", periods=1, tz="UTC")}),
                pd.DataFrame({"timestamp": pd.date_range("2026-06-02", periods=1, tz="UTC")}),
                ["state_shock", "state_transition"],
            )
        },
        "reports": {
            "feature_store": {
                "selected_column_count": 2,
                "selected_columns": ["mkt_ret_eq_1h", "rv_24h"],
                "train": {
                    "universe_contract": {
                        "universe_definition_version": "feature_store_timestamp_market_state_v1",
                        "available_symbol_count": 2,
                        "eligible_symbol_count": 2,
                        "eligible_symbols": ["AAA", "BBB"],
                        "excluded_symbols": [],
                        "excluded_reasons": {},
                    },
                    "timestamp_coverage": 1.0,
                    "symbols_read": 2,
                },
                "eval": {
                    "universe_contract": {
                        "universe_definition_version": "feature_store_timestamp_market_state_v1",
                        "available_symbol_count": 2,
                        "eligible_symbol_count": 2,
                        "eligible_symbols": ["AAA", "BBB"],
                        "excluded_symbols": [],
                        "excluded_reasons": {},
                    },
                    "timestamp_coverage": 1.0,
                    "symbols_read": 2,
                },
            },
            "market_state_source": {
                "train": {
                    "source": "feature_store_market_aggregates",
                    "feature_count": 2,
                    "production_safe": True,
                    "allow_candidate_fallback": False,
                    "candidate_aggregate_feature_count": 1,
                    "feature_store_aggregate_feature_count": 2,
                    "forbidden_candidate_aggregate_columns_removed": ["short_asset__rank_mean"],
                    "validation": {
                        "row_count": 1,
                        "feature_count": 2,
                        "forbidden_column_count": 0,
                        "timestamp_unique": True,
                        "market_wide_one_row_per_timestamp": True,
                    },
                },
                "eval": {
                    "source": "feature_store_market_aggregates",
                    "feature_count": 2,
                    "production_safe": True,
                    "allow_candidate_fallback": False,
                    "candidate_aggregate_feature_count": 1,
                    "feature_store_aggregate_feature_count": 2,
                    "forbidden_candidate_aggregate_columns_removed": ["short_asset__rank_mean"],
                    "validation": {
                        "row_count": 1,
                        "feature_count": 2,
                        "forbidden_column_count": 0,
                        "timestamp_unique": True,
                        "market_wide_one_row_per_timestamp": True,
                    },
                },
            },
            "axis_sources": {"state_shock": ["a"]},
            "observed_axis_encoder": {
                "mode": observed_encoder.get("mode"),
                "contract": observed_encoder.get("contract"),
                "fit_rows": observed_encoder.get("fit_rows"),
                "fit_timestamp_min": observed_encoder.get("fit_timestamp_min"),
                "fit_timestamp_max": observed_encoder.get("fit_timestamp_max"),
                "minimum_input_coverage": observed_encoder.get("minimum_input_coverage"),
                "axis_count": int(len(observed_encoder.get("axes", {}))),
                "source_column_count": int(len(observed_encoder.get("column_refs", {}))),
                "ret_col": observed_encoder.get("ret_col"),
                "transition_column_count": int(
                    len((observed_encoder.get("transition", {}) or {}).get("columns", []) or [])
                ),
                "reliability_column_count": int(
                    len((observed_encoder.get("reliability", {}) or {}).get("columns", []) or [])
                ),
                "low_input_coverage_fail_closed": bool(
                    "state_low_input_coverage" in dict(observed_encoder.get("axis_sources", {}) or {})
                ),
                "source_validation_train_present": bool(
                    dict(observed_encoder.get("source_validation", {}) or {}).get("train")
                ),
            },
            "forecast_report": {"mode": "not_selected"},
            "latent_report": {"mode": "not_selected"},
        },
    }

    manifest = mat._make_manifest(
        args=Args(),
        selected_arm="S1_observed_axes_shared_response",
        selected_payload={"selected_arm": "S1_observed_axes_shared_response"},
        state_spec={"state_level": "observed", "per_strategy_residual": False},
        state_artifacts=state_artifacts,
        response_feature_cols=["normalized_rank_score", "state_shock"],
        response_report={"risk_model": "rank_curve_plus_excess_risk_regressors"},
        activation_report={"available": True, "active_state_heads": ["state_shock", "state_transition"]},
        state_activation_filter={
            "active_state_feature_columns": ["state_shock", "state_transition"],
            "dropped_state_feature_columns": [],
        },
        controller_execution_enabled=True,
        walkforward_config={"available": True, "controller": {"forecast_model_kind": "lightgbm"}},
        forecast_model_kind_report={"value": "lightgbm", "source": "walkforward_manifest"},
        response_model_kind_report={"value": "additive_ebm", "source": "walkforward_manifest"},
        runtime_param_resolution={
            "inherit_walkforward_controller_config": True,
            "params": {"threshold_delta_max": {"source": "walkforward_manifest", "value": 0.10}},
        },
        state_join_validation={
            "train": {
                "state_join_timestamp_constant": True,
                "max_state_values_per_timestamp": 1,
            }
        },
        bundle_path=tmp_path / "bundle.joblib",
        outputs={"bundle": str(tmp_path / "bundle.joblib")},
    )

    assert manifest["selected_arm"] == "S1_observed_axes_shared_response"
    assert manifest["state_level"] == "observed"
    assert manifest["rank_contract"] == "short_boll_timestamp_rank"
    assert manifest["disabled_heads"] == ["long_bars", "long_dist"]
    assert manifest["active_heads"] == ["short_asset", "short_boll"]
    assert manifest["controller_enabled_heads"] == ["short_asset", "short_boll"]
    assert manifest["controller_enabled_scope"] == "all_active_heads"
    assert manifest["controller_execution_enabled"] is True
    assert manifest["controller"]["penalty_only"] is True
    assert manifest["controller"]["execution_enabled"] is True
    assert manifest["controller"]["forecast_model_kind"] == "lightgbm"
    assert manifest["controller"]["response_model_kind"] == "additive_ebm"
    assert manifest["forecast_model_kind_resolution"]["source"] == "walkforward_manifest"
    assert manifest["response_model_kind_resolution"]["source"] == "walkforward_manifest"
    assert manifest["runtime_param_resolution"]["inherit_walkforward_controller_config"] is True
    assert manifest["state_join_validation"]["train"]["state_join_timestamp_constant"] is True
    assert manifest["source_contract_audit"]["overall_passed"] is True
    assert manifest["source_contract_audit"]["splits"]["train"]["production_safe"] is True
    assert manifest["source_contract_audit"]["splits"]["train"]["validation_forbidden_column_count"] == 0
    assert manifest["controller"]["changes_scores_or_ranks"] is False
    assert manifest["feature_store_tail_reference"]["quantile_count"] == 2
    assert manifest["observed_axis_encoder"]["mode"] == "observed_axis_robust_z_v1"
    assert manifest["observed_axis_encoder"]["minimum_input_coverage"] == 0.80
    assert manifest["observed_axis_encoder"]["low_input_coverage_fail_closed"] is True
    assert manifest["observed_axis_encoder"]["source_validation_train_present"] is True
    assert manifest["state_feature_columns"] == ["state_shock", "state_transition"]
    assert manifest["response_feature_count"] == 2

    contract = mat._make_market_state_feature_contract(
        args=Args(),
        selected_arm="S1_observed_axes_shared_response",
        state_spec={"state_level": "observed", "per_strategy_residual": False},
        state_artifacts=state_artifacts,
        response_feature_cols=["normalized_rank_score", "state_shock"],
        activation_report={"available": True, "active_state_heads": ["state_shock", "state_transition"]},
        state_activation_filter={
            "active_state_feature_columns": ["state_shock", "state_transition"],
            "dropped_state_feature_columns": [],
        },
        controller_execution_enabled=True,
        walkforward_config={"available": True, "controller": {"forecast_model_kind": "lightgbm"}},
        forecast_model_kind_report={"value": "lightgbm", "source": "walkforward_manifest"},
        response_model_kind_report={"value": "additive_ebm", "source": "walkforward_manifest"},
        runtime_param_resolution={
            "inherit_walkforward_controller_config": True,
            "params": {"threshold_delta_max": {"source": "walkforward_manifest", "value": 0.10}},
        },
    )
    assert contract["contract_version"] == "market_state_feature_contract_v1"
    assert contract["active_heads"] == ["short_asset", "short_boll"]
    assert contract["controller_enabled_heads"] == ["short_asset", "short_boll"]
    assert contract["controller_enabled_scope"] == "all_active_heads"
    assert contract["controller_execution_enabled"] is True
    assert contract["forecast_model_kind"] == "lightgbm"
    assert contract["response_model_kind"] == "additive_ebm"
    assert contract["runtime_param_resolution"]["inherit_walkforward_controller_config"] is True
    assert contract["feature_store"]["train_universe_contract"]["eligible_symbols"] == ["AAA", "BBB"]
    assert contract["feature_store_tail_reference"]["quantile_count"] == 2
    assert set(contract["feature_store_tail_reference"]["quantiles"]) == {"mkt_ret_eq_1h", "rv_24h"}
    assert contract["source_schema"]["state_feature_columns"] == ["state_shock", "state_transition"]
    assert contract["source_contract_audit"]["overall_passed"] is True
    assert contract["source_contract_audit"]["splits"]["eval"]["market_wide_one_row_per_timestamp"] is True
    assert contract["invariants"]["market_state_uses_model_predictions"] is False
    assert contract["invariants"]["actual_order_book_features_allowed"] is False
    assert contract["invariants"]["controller_can_lower_thresholds"] is False

    universe_contract = mat._make_market_state_universe_contract(state_artifacts)
    assert universe_contract["contract_version"] == "market_state_universe_contract_v1"
    assert universe_contract["generated_by"] == "materialize_market_state_controller_bundle"
    assert universe_contract["validation"]["passed"] is True
    assert universe_contract["candidate_independent"] is True
    assert universe_contract["actual_order_book_features_allowed"] is False
    assert universe_contract["eligible_symbols"] == ["AAA", "BBB"]


def test_threshold_schedule_is_penalty_only_and_bounded() -> None:
    ts = pd.date_range("2026-06-15", periods=3, freq="h", tz="UTC")
    eval_frame = pd.DataFrame(
        {
            "timestamp": np.repeat(ts, 3),
            "strategy_id": "strategy_a",
            "head": "short_asset",
            "_rank": [0.70, 0.74, 0.93] * 3,
            "_threshold": 0.70,
        }
    )
    predictions = pd.DataFrame(
        {
            "pred_mean_utility": [-0.01, -0.02, 0.01] * 3,
            "pred_lcb_utility": [-0.04, -0.03, -0.01] * 3,
            "pred_full_sl": [0.80, 0.70, 0.55] * 3,
            "pred_timeout": [0.20, 0.20, 0.20] * 3,
        }
    )

    class Curves:
        def strategy_cap(self, strategy_id: str, target: str, base_threshold: float, pad: float) -> float:
            return 0.30 if target == "psl" else 0.50

    schedule = mstc.threshold_schedule(
        eval_frame,
        predictions,
        Curves(),
        delta_max=0.08,
        max_down_step=0.03,
        relax_alpha=0.10,
        controller_mode="rank_grid",
    )

    assert schedule["state_threshold"].ge(schedule["base_threshold"]).all()
    assert (schedule["state_threshold"] <= schedule["base_threshold"] + 0.08 + 1e-12).all()
    assert schedule["state_threshold"].gt(schedule["base_threshold"]).any()


def test_threshold_action_audit_summarizes_schedule_without_outcomes() -> None:
    schedule = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-06-15", periods=3, freq="h", tz="UTC"),
            "strategy_id": ["s1", "s1", "s2"],
            "head": ["short_asset", "short_asset", "short_boll"],
            "base_threshold": [0.70, 0.70, 0.72],
            "raw_state_threshold": [0.70, 0.76, 0.72],
            "state_threshold": [0.70, 0.74, 0.72],
            "force_base_threshold": [True, False, True],
            "controller_reason": [
                "head_not_enabled_for_threshold_action",
                "rank_grid_penalty",
                "head_not_enabled_for_threshold_action",
            ],
            "risk_severity": [0.0, 0.8, 0.0],
            "prediction_coverage": [1.0, 0.9, 1.0],
            "min_prediction_coverage": [0.8, 0.8, 0.8],
            "state_ood_share": [0.0, 0.1, 0.0],
            "base_candidate_count": [10, 12, 5],
            "frontier_candidate_count": [0, 3, 0],
            "tail_candidate_count": [10, 8, 5],
            "suppressed_candidate_count": [0, 4, 0],
            "predicted_removed_loss_avoided": [0.0, 0.2, 0.0],
            "predicted_removed_winner_sacrificed": [0.0, 0.05, 0.0],
            "predicted_action_edge": [0.0, 0.15, 0.0],
        }
    )

    audit = mstc.threshold_action_audit(schedule)

    all_row = audit.loc[audit["scope"].eq("all")].iloc[0]
    assert int(all_row["schedule_rows"]) == 3
    assert int(all_row["threshold_raised_count"]) == 1
    np.testing.assert_allclose(float(all_row["threshold_raised_share"]), 1 / 3)
    assert int(all_row["total_suppressed_candidate_count"]) == 4

    reason_rows = audit.loc[audit["scope"].eq("controller_reason")]
    assert set(reason_rows["scope_value"]) == {
        "head_not_enabled_for_threshold_action",
        "rank_grid_penalty",
    }


def test_threshold_schedule_can_limit_action_to_enabled_heads() -> None:
    ts = pd.date_range("2026-06-15", periods=3, freq="h", tz="UTC")
    eval_frame = pd.DataFrame(
        {
            "timestamp": np.tile(ts, 6),
            "strategy_id": ["strategy_asset"] * 9 + ["strategy_boll"] * 9,
            "head": ["short_asset"] * 9 + ["short_boll"] * 9,
            "_rank": [0.70, 0.74, 0.93] * 6,
            "_threshold": 0.70,
        }
    )
    predictions = pd.DataFrame(
        {
            "pred_mean_utility": [-0.01, -0.02, 0.01] * 6,
            "pred_lcb_utility": [-0.04, -0.03, -0.01] * 6,
            "pred_full_sl": [0.80, 0.70, 0.55] * 6,
            "pred_timeout": [0.20, 0.20, 0.20] * 6,
        }
    )

    class Curves:
        def strategy_cap(self, strategy_id: str, target: str, base_threshold: float, pad: float) -> float:
            return 0.30 if target == "psl" else 0.50

    schedule = mstc.threshold_schedule(
        eval_frame,
        predictions,
        Curves(),
        delta_max=0.08,
        max_down_step=0.03,
        relax_alpha=0.10,
        controller_mode="rank_grid",
        enabled_heads={"short_boll"},
    )

    disabled = schedule.loc[schedule["head"].eq("short_asset")]
    enabled = schedule.loc[schedule["head"].eq("short_boll")]

    assert disabled["state_threshold"].eq(disabled["base_threshold"]).all()
    assert disabled["raw_state_threshold"].eq(disabled["base_threshold"]).all()
    assert not disabled["threshold_action_enabled"].any()
    assert disabled["controller_reason"].eq("head_not_enabled_for_threshold_action").all()
    assert enabled["threshold_action_enabled"].all()
    assert enabled["state_threshold"].gt(enabled["base_threshold"]).any()


def test_threshold_schedule_fails_closed_on_low_prediction_coverage() -> None:
    ts = pd.date_range("2026-06-15", periods=2, freq="h", tz="UTC")
    eval_frame = pd.DataFrame(
        {
            "timestamp": np.repeat(ts, 3),
            "strategy_id": "strategy_a",
            "head": "short_asset",
            "_rank": [0.70, 0.74, 0.93] * 2,
            "_threshold": 0.70,
        }
    )
    predictions = pd.DataFrame(
        {
            "pred_mean_utility": [-0.01, -0.02, 0.01, -0.01, -0.02, 0.01],
            "pred_lcb_utility": [-0.04, -0.03, -0.01, -0.04, -0.03, -0.01],
            "pred_full_sl": [0.80, 0.70, 0.55, 0.80, 0.70, 0.55],
            "pred_timeout": [0.20] * 6,
            "state_feature_coverage": [1.0, 1.0, 1.0, 0.20, 0.20, 0.20],
            "response_feature_coverage": [1.0] * 6,
        }
    )

    class Curves:
        def strategy_cap(self, strategy_id: str, target: str, base_threshold: float, pad: float) -> float:
            return 0.30 if target == "psl" else 0.50

    schedule = mstc.threshold_schedule(
        eval_frame,
        predictions,
        Curves(),
        delta_max=0.08,
        max_down_step=0.08,
        relax_alpha=0.10,
        controller_mode="rank_grid",
        min_prediction_coverage=0.80,
    ).sort_values("timestamp")

    first = schedule.iloc[0]
    second = schedule.iloc[1]
    assert first["state_threshold"] > first["base_threshold"]
    assert not bool(first["force_base_threshold"])
    assert second["controller_reason"] == "insufficient_prediction_coverage"
    assert bool(second["force_base_threshold"])
    assert second["state_threshold"] == second["base_threshold"]


def test_threshold_schedule_fails_closed_on_state_ood() -> None:
    ts = pd.date_range("2026-06-15", periods=2, freq="h", tz="UTC")
    eval_frame = pd.DataFrame(
        {
            "timestamp": np.repeat(ts, 3),
            "strategy_id": "strategy_a",
            "head": "short_asset",
            "_rank": [0.70, 0.74, 0.93] * 2,
            "_threshold": 0.70,
        }
    )
    predictions = pd.DataFrame(
        {
            "pred_mean_utility": [-0.01, -0.02, 0.01, -0.01, -0.02, 0.01],
            "pred_lcb_utility": [-0.04, -0.03, -0.01, -0.04, -0.03, -0.01],
            "pred_full_sl": [0.80, 0.70, 0.55, 0.80, 0.70, 0.55],
            "pred_timeout": [0.20] * 6,
            "state_feature_coverage": [1.0] * 6,
            "response_feature_coverage": [1.0] * 6,
            "state_ood_score": [0.5, 0.5, 0.5, 5.0, 5.0, 5.0],
            "state_ood_cutoff": [2.0] * 6,
        }
    )

    class Curves:
        def strategy_cap(self, strategy_id: str, target: str, base_threshold: float, pad: float) -> float:
            return 0.30 if target == "psl" else 0.50

    schedule = mstc.threshold_schedule(
        eval_frame,
        predictions,
        Curves(),
        delta_max=0.08,
        max_down_step=0.08,
        relax_alpha=0.10,
        controller_mode="rank_grid",
    ).sort_values("timestamp")

    first = schedule.iloc[0]
    second = schedule.iloc[1]
    assert first["state_threshold"] > first["base_threshold"]
    assert first["state_ood_share"] == 0.0
    assert second["controller_reason"] == "state_ood_fallback"
    assert second["state_ood_share"] == 1.0
    assert bool(second["force_base_threshold"])
    assert second["state_threshold"] == second["base_threshold"]


def test_threshold_schedule_fails_closed_without_frontier_support() -> None:
    ts = pd.Timestamp("2026-06-15T00:00:00Z")
    eval_frame = pd.DataFrame(
        {
            "timestamp": [ts] * 4,
            "strategy_id": "strategy_a",
            "head": "short_boll",
            "_rank": [0.88, 0.91, 0.94, 0.98],
            "_threshold": 0.70,
        }
    )
    predictions = pd.DataFrame(
        {
            "pred_mean_utility": [-0.03, -0.02, -0.01, -0.01],
            "pred_lcb_utility": [-0.08, -0.06, -0.05, -0.04],
            "pred_full_sl": [0.90, 0.85, 0.80, 0.75],
            "pred_timeout": [0.20, 0.20, 0.20, 0.20],
        }
    )

    class Curves:
        def strategy_cap(self, strategy_id: str, target: str, base_threshold: float, pad: float) -> float:
            return 0.30 if target == "psl" else 0.50

    schedule = mstc.threshold_schedule(
        eval_frame,
        predictions,
        Curves(),
        delta_max=0.08,
        max_down_step=0.08,
        relax_alpha=0.25,
        controller_mode="rank_grid",
        min_frontier_candidates=1,
    )

    row = schedule.iloc[0]
    assert int(row["base_candidate_count"]) == 4
    assert int(row["frontier_candidate_count"]) == 0
    assert int(row["min_frontier_candidate_count"]) == 1
    assert row["controller_reason"] == "insufficient_frontier_candidate_support"
    assert bool(row["force_base_threshold"])
    assert row["raw_state_threshold"] == row["base_threshold"]
    assert row["state_threshold"] == row["base_threshold"]
    assert int(row["suppressed_candidate_count"]) == 0


def test_frontier_rank_grid_focuses_on_marginal_threshold_band() -> None:
    ts = pd.Timestamp("2026-06-15T00:00:00Z")
    ranks = [0.70, 0.74, 0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.97, 0.99, 1.00, 1.01]
    eval_frame = pd.DataFrame(
        {
            "timestamp": [ts] * len(ranks),
            "strategy_id": "strategy_a",
            "head": "short_asset",
            "_rank": ranks,
            "_threshold": 0.70,
        }
    )
    predictions = pd.DataFrame(
        {
            "pred_mean_utility": [-0.03, -0.03] + [0.03] * 10,
            "pred_lcb_utility": [-0.05, -0.05] + [0.02] * 10,
            "pred_full_sl": [0.90, 0.90] + [0.10] * 10,
            "pred_timeout": [0.10] * 12,
        }
    )

    class Curves:
        def strategy_cap(self, strategy_id: str, target: str, base_threshold: float, pad: float) -> float:
            return 0.30 if target == "psl" else 0.50

    broad = mstc.threshold_schedule(
        eval_frame,
        predictions,
        Curves(),
        delta_max=0.08,
        max_down_step=0.08,
        relax_alpha=0.25,
        controller_mode="rank_grid",
    )
    frontier = mstc.threshold_schedule(
        eval_frame,
        predictions,
        Curves(),
        delta_max=0.08,
        max_down_step=0.08,
        relax_alpha=0.25,
        controller_mode="frontier_rank_grid",
    )

    assert broad.iloc[0]["state_threshold"] == broad.iloc[0]["base_threshold"]
    assert frontier.iloc[0]["state_threshold"] > frontier.iloc[0]["base_threshold"]
    assert int(frontier.iloc[0]["frontier_candidate_count"]) == 2
    assert int(frontier.iloc[0]["suppressed_candidate_count"]) == 2


def test_action_aware_threshold_vetoes_timeout_heavy_removed_rows() -> None:
    ts = pd.Timestamp("2026-06-15T00:00:00Z")
    eval_frame = pd.DataFrame(
        {
            "timestamp": [ts, ts, ts],
            "strategy_id": "strategy_a",
            "head": "short_boll",
            "_rank": [0.70, 0.74, 0.92],
            "_threshold": 0.70,
        }
    )
    predictions = pd.DataFrame(
        {
            "pred_mean_utility": [0.0, 0.01, 0.01],
            "pred_lcb_utility": [-0.20, 0.03, 0.03],
            "pred_full_sl": [0.10, 0.10, 0.10],
            "pred_timeout": [0.95, 0.10, 0.10],
        }
    )

    class Curves:
        def strategy_cap(self, strategy_id: str, target: str, base_threshold: float, pad: float) -> float:
            return 0.30 if target == "psl" else 0.50

    permissive = mstc.threshold_schedule(
        eval_frame,
        predictions,
        Curves(),
        delta_max=0.08,
        max_down_step=0.08,
        relax_alpha=0.25,
        controller_mode="frontier_action_rank_grid",
        min_lcb_utility=0.0,
        min_action_edge=0.0,
        min_frontier_candidates=1,
    )
    vetoed = mstc.threshold_schedule(
        eval_frame,
        predictions,
        Curves(),
        delta_max=0.08,
        max_down_step=0.08,
        relax_alpha=0.25,
        controller_mode="frontier_action_rank_grid",
        min_lcb_utility=0.0,
        min_action_edge=0.0,
        min_frontier_candidates=1,
        min_removed_full_sl=0.50,
        max_removed_timeout=0.50,
    )

    assert permissive.iloc[0]["state_threshold"] > permissive.iloc[0]["base_threshold"]
    assert permissive.iloc[0]["predicted_action_edge"] > 0.0
    assert vetoed.iloc[0]["state_threshold"] == vetoed.iloc[0]["base_threshold"]
    assert vetoed.iloc[0]["controller_reason"] == "frontier_action_rank_grid_no_positive_edge"
    assert int(vetoed.iloc[0]["suppressed_candidate_count"]) == 0


def test_accepted_frontier_action_grid_requires_direct_positive_suppression() -> None:
    ts = pd.Timestamp("2026-06-15T00:00:00Z")
    eval_frame = pd.DataFrame(
        {
            "timestamp": [ts, ts],
            "strategy_id": "strategy_a",
            "head": "short_asset",
            "_rank": [0.74, 0.96],
            "_threshold": 0.70,
        }
    )
    predictions = pd.DataFrame(
        {
            "pred_mean_utility": [-0.02, 0.03],
            "pred_lcb_utility": [-0.08, 0.02],
            "pred_full_sl": [0.90, 0.10],
            "pred_timeout": [0.10, 0.10],
        }
    )

    class Curves:
        def strategy_cap(self, strategy_id: str, target: str, base_threshold: float, pad: float) -> float:
            return 0.30 if target == "psl" else 0.50

    schedule = mstc.threshold_schedule(
        eval_frame,
        predictions,
        Curves(),
        delta_max=0.08,
        max_down_step=0.08,
        relax_alpha=0.25,
        controller_mode="accepted_frontier_action_rank_grid",
        min_lcb_utility=0.0,
        min_action_edge=0.0,
        min_frontier_candidates=1,
    )

    row = schedule.iloc[0]
    assert bool(row["accepted_frontier_direct_required"])
    assert row["state_threshold"] > 0.74
    assert int(row["suppressed_candidate_count"]) == 1
    assert row["predicted_action_edge"] > 0.0
    assert row["controller_reason"] == "accepted_frontier_action_rank_grid_positive_edge_penalty"


def test_accepted_frontier_action_grid_uses_baseline_accepted_keys() -> None:
    ts = pd.Timestamp("2026-06-15T00:00:00Z")
    eval_frame = pd.DataFrame(
        {
            "timestamp": [ts, ts],
            "symbol": ["BAD_NEAR_FRONTIER", "GOOD_HIGH_RANK"],
            "side": ["short", "short"],
            "strategy_id": ["strategy_a", "strategy_a"],
            "head": ["short_asset", "short_asset"],
            "_rank": [0.74, 0.96],
            "_threshold": [0.70, 0.70],
        }
    )
    predictions = pd.DataFrame(
        {
            "pred_mean_utility": [-0.02, 0.03],
            "pred_lcb_utility": [-0.08, 0.02],
            "pred_full_sl": [0.90, 0.10],
            "pred_timeout": [0.10, 0.10],
        }
    )

    class Curves:
        def strategy_cap(self, strategy_id: str, target: str, base_threshold: float, pad: float) -> float:
            return 0.30 if target == "psl" else 0.50

    accepted_high_only = mstc._accepted_key_set(
        pd.DataFrame(
            {
                "timestamp": [ts],
                "symbol": ["GOOD_HIGH_RANK"],
                "side": ["short"],
                "strategy_id": ["strategy_a"],
            }
        )
    )
    high_only_schedule = mstc.threshold_schedule(
        eval_frame,
        predictions,
        Curves(),
        delta_max=0.08,
        max_down_step=0.08,
        relax_alpha=0.25,
        controller_mode="accepted_frontier_action_rank_grid",
        min_lcb_utility=0.0,
        min_action_edge=0.0,
        min_frontier_candidates=1,
        accepted_decision_keys=accepted_high_only,
    )
    high_only = high_only_schedule.iloc[0]
    assert bool(high_only["accepted_frontier_key_filter_active"])
    assert int(high_only["accepted_frontier_candidate_count"]) == 0
    assert int(high_only["accepted_frontier_suppressed_count"]) == 0
    assert high_only["state_threshold"] == high_only["base_threshold"]
    assert high_only["controller_reason"] == "no_baseline_accepted_candidate_in_frontier"

    accepted_near_frontier = mstc._accepted_key_set(
        pd.DataFrame(
            {
                "timestamp": [ts],
                "symbol": ["BAD_NEAR_FRONTIER"],
                "side": ["short"],
                "strategy_id": ["strategy_a"],
            }
        )
    )
    near_schedule = mstc.threshold_schedule(
        eval_frame,
        predictions,
        Curves(),
        delta_max=0.08,
        max_down_step=0.08,
        relax_alpha=0.25,
        controller_mode="accepted_frontier_action_rank_grid",
        min_lcb_utility=0.0,
        min_action_edge=0.0,
        min_frontier_candidates=1,
        accepted_decision_keys=accepted_near_frontier,
    )
    near = near_schedule.iloc[0]
    assert bool(near["accepted_frontier_key_filter_active"])
    assert int(near["accepted_frontier_candidate_count"]) == 1
    assert int(near["accepted_frontier_suppressed_count"]) == 1
    assert near["state_threshold"] > 0.74
    assert near["controller_reason"] == "accepted_frontier_action_rank_grid_positive_edge_penalty"


def test_accepted_frontier_action_grid_jumps_to_direct_floor_when_risk_rises() -> None:
    ts = pd.date_range("2026-06-15", periods=2, freq="h", tz="UTC")
    eval_frame = pd.DataFrame(
        {
            "timestamp": [ts[0], ts[0], ts[1], ts[1]],
            "strategy_id": "strategy_a",
            "head": "short_asset",
            "_rank": [0.74, 0.96, 0.79, 0.96],
            "_threshold": 0.70,
        }
    )
    predictions = pd.DataFrame(
        {
            "pred_mean_utility": [0.02, 0.03, -0.02, 0.03],
            "pred_lcb_utility": [0.01, 0.02, -0.08, 0.02],
            "pred_full_sl": [0.10, 0.10, 0.90, 0.10],
            "pred_timeout": [0.10, 0.10, 0.10, 0.10],
        }
    )

    class Curves:
        def strategy_cap(self, strategy_id: str, target: str, base_threshold: float, pad: float) -> float:
            return 0.30 if target == "psl" else 0.50

    schedule = mstc.threshold_schedule(
        eval_frame,
        predictions,
        Curves(),
        delta_max=0.10,
        max_down_step=0.03,
        relax_alpha=0.25,
        controller_mode="accepted_frontier_action_rank_grid",
        min_lcb_utility=0.0,
        min_action_edge=0.0,
        min_frontier_candidates=1,
    )

    schedule = schedule.sort_values("timestamp")
    first = schedule.iloc[0]
    row = schedule.iloc[1]
    assert first["state_threshold"] == first["base_threshold"]
    assert bool(row["accepted_frontier_direct_required"])
    assert row["raw_state_threshold"] > 0.79
    assert row["state_threshold"] >= row["direct_suppression_threshold_floor"]
    assert row["state_threshold"] > 0.79


def test_accepted_frontier_action_grid_fails_closed_on_ood_without_direct_floor() -> None:
    ts = pd.Timestamp("2026-06-15T00:00:00Z")
    eval_frame = pd.DataFrame(
        {
            "timestamp": [ts],
            "strategy_id": ["strategy_a"],
            "head": ["short_boll"],
            "_rank": [0.74],
            "_threshold": [0.70],
        }
    )
    predictions = pd.DataFrame(
        {
            "pred_mean_utility": [-0.02],
            "pred_lcb_utility": [-0.08],
            "pred_full_sl": [0.90],
            "pred_timeout": [0.10],
            "state_feature_coverage": [1.0],
            "response_feature_coverage": [1.0],
            "state_ood_score": [10.0],
            "state_ood_cutoff": [1.0],
            "state_ood_flag": [True],
        }
    )

    schedule = mstc.threshold_schedule(
        eval_frame,
        predictions,
        _FlatCurves(),
        delta_max=0.08,
        max_down_step=0.08,
        relax_alpha=0.25,
        controller_mode="accepted_frontier_action_rank_grid",
        min_lcb_utility=0.0,
        min_action_edge=0.0,
        min_frontier_candidates=1,
    )

    row = schedule.iloc[0]
    assert row["controller_reason"] == "state_ood_fallback"
    assert row["state_threshold"] == row["base_threshold"]
    assert int(row["accepted_frontier_suppressed_count"]) == 0
    assert pd.isna(row["direct_suppression_threshold_floor"])


def test_candidate_suppression_utility_measures_realized_defensive_success() -> None:
    candidates = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-06-15T00:00:00Z"] * 4),
            "strategy_id": "strategy_a",
            "head": "short_asset",
            "symbol": ["A", "B", "C", "D"],
            "side": "short",
            "normalized_rank_score": [0.69, 0.72, 0.75, 0.82],
            "base_strategy_threshold": 0.70,
            "deployment_rank_threshold": 0.70,
            "net_return": [0.02, -0.03, 0.01, 0.04],
            "simple_policy_exit_reason": ["tp", "sl", "tp", "tp"],
        }
    )
    schedules = pd.DataFrame(
        {
            "arm": "S1",
            "timestamp": pd.to_datetime(["2026-06-15T00:00:00Z"]),
            "strategy_id": ["strategy_a"],
            "base_threshold": [0.70],
            "state_threshold": [0.80],
            "risk_severity": [0.5],
            "predicted_action_edge": [0.02],
        }
    )

    out = mstc._threshold_candidate_suppression_utility(candidates, schedules)
    row = out.loc[out["scope"].eq("all")].iloc[0]

    assert int(row["suppressed_candidates"]) == 2
    assert np.isclose(row["suppressed_loss_avoided"], 0.03)
    assert np.isclose(row["suppressed_winner_pnl_sacrificed"], 0.01)
    assert np.isclose(row["realized_defensive_success"], 0.02)
    assert row["suppressed_full_sl_rate"] == 0.5


def test_walkforward_empty_suppression_aggregate_keeps_schema() -> None:
    empty = mstc._threshold_candidate_suppression_utility(pd.DataFrame(), pd.DataFrame())
    empty_with_fold = empty.copy()
    empty_with_fold["fold"] = pd.Series(dtype="int64")
    aggregate = wf._aggregate_suppression_utility(empty)

    assert empty.empty
    assert empty_with_fold.empty
    assert aggregate.empty
    assert "suppressed_candidates" in empty.columns
    assert "fold" in empty_with_fold.columns
    assert "suppressed_candidates" in aggregate.columns
    assert "realized_defensive_success" in aggregate.columns


def test_threshold_action_edge_validation_links_predicted_edge_to_realized_action() -> None:
    ts = pd.Timestamp("2026-06-15T00:00:00Z")
    accepted = pd.DataFrame(
        {
            "arm": ["S0_baseline_static_thresholds", "S0_baseline_static_thresholds", "S1", "S1"],
            "timestamp": [ts, ts, ts, ts],
            "strategy_id": ["strategy_a"] * 4,
            "head": ["short_asset"] * 4,
            "symbol": ["A", "B", "B", "C"],
            "side": ["short"] * 4,
            "net_pnl": [-10.0, 5.0, 4.0, 2.0],
            "gross_pnl": [-9.0, 6.0, 5.0, 3.0],
            "net_return": [-0.10, 0.05, 0.04, 0.02],
            "simple_policy_exit_reason": ["sl", "tp", "tp", "tp"],
        }
    )
    schedules = pd.DataFrame(
        {
            "arm": ["S1"],
            "timestamp": [ts],
            "strategy_id": ["strategy_a"],
            "base_threshold": [0.70],
            "state_threshold": [0.78],
            "predicted_action_edge": [8.0],
            "predicted_removed_loss_avoided": [10.0],
            "predicted_removed_winner_sacrificed": [2.0],
        }
    )

    detail = mstc._threshold_action_edge_validation(
        accepted,
        schedules,
        "S0_baseline_static_thresholds",
    )
    row = detail.iloc[0]

    assert int(row["baseline_accepted"]) == 2
    assert int(row["current_accepted"]) == 2
    assert int(row["entrants"]) == 1
    assert int(row["removed"]) == 1
    assert np.isclose(row["net_replacement_pnl"], 12.0)
    assert np.isclose(row["same_key_net_pnl_delta"], -1.0)
    assert np.isclose(row["net_action_pnl_delta"], 11.0)
    assert np.isclose(row["removed_loss_avoided"], 10.0)
    assert np.isclose(row["removed_winner_pnl_sacrificed"], 0.0)
    assert np.isclose(row["defensive_success"], 10.0)

    bucket = mstc._threshold_action_edge_bucket_performance(detail)
    assert np.isclose(bucket.iloc[0]["net_action_pnl_delta"], 11.0)
    assert np.isclose(bucket.iloc[0]["realized_minus_predicted_action_edge"], 3.0)


def test_action_edge_bucket_uses_edge_values_not_row_order() -> None:
    detail = pd.DataFrame(
        {
            "arm": ["S1"] * 6,
            "predicted_action_edge": [0.0, 0.0, 0.0, 1.0, 2.0, 3.0],
            "threshold_delta": [0.01] * 6,
            "baseline_accepted": [1] * 6,
            "current_accepted": [1] * 6,
            "entrants": [0] * 6,
            "removed": [0] * 6,
            "net_replacement_pnl": [0.0] * 6,
            "same_key_net_pnl_delta": [0.0] * 6,
            "net_action_pnl_delta": [0.0] * 6,
            "removed_loss_avoided": [0.0] * 6,
            "removed_winner_pnl_sacrificed": [0.0] * 6,
            "defensive_success": [0.0] * 6,
        }
    )

    bucket = mstc._threshold_action_edge_bucket_performance(detail, buckets=4)
    zero_rows = bucket.loc[np.isclose(bucket["mean_predicted_action_edge"], 0.0)]

    assert len(zero_rows) == 1
    assert int(zero_rows.iloc[0]["schedule_rows"]) == 3


def test_controller_candidate_selection_prefers_simpler_tied_passing_arm() -> None:
    aggregate = pd.DataFrame(
        {
            "arm": [
                "S0_baseline_static_thresholds",
                "S1_observed_axes_shared_response",
                "S2_observed_forecast_shared_response",
                "S1_observed_axes_shared_response__post_selection_overlay",
                "S2_observed_forecast_shared_response__post_selection_overlay",
            ],
            "folds": [3, 3, 3, 3, 3],
            "median_delta_net_pnl": [0.0, 55.0, 56.0, 12.0, 12.0],
            "mean_delta_net_pnl": [0.0, 100.0, 100.0, 20.0, 20.0],
            "q25_delta_net_pnl": [0.0, 20.0, 20.0, 5.0, 5.0],
            "positive_delta_share": [0.0, 2 / 3, 2 / 3, 2 / 3, 2 / 3],
            "median_delta_max_drawdown": [0.0, 0.01, 0.01, 0.01, 0.01],
            "median_delta_worst_24h": [0.0, 1.0, 1.0, 1.0, 1.0],
            "median_trade_retention_share": [1.0, 0.95, 0.95, 0.95, 0.95],
            "median_delta_full_sl_rate": [0.0, -0.01, -0.01, -0.01, -0.01],
        }
    )
    suppression = pd.DataFrame(
        {
            "arm": [
                "S1_observed_axes_shared_response",
                "S2_observed_forecast_shared_response",
                "S1_observed_axes_shared_response__post_selection_overlay",
                "S2_observed_forecast_shared_response__post_selection_overlay",
            ],
            "scope": ["all", "all", "all", "all"],
            "scope_value": ["all", "all", "all", "all"],
            "realized_defensive_success": [0.2, 0.2, 0.1, 0.1],
            "positive_suppression_fold_share": [1.0, 1.0, 1.0, 1.0],
            "suppressed_candidates": [10, 10, 5, 5],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "arm": [
                "S1_observed_axes_shared_response",
                "S2_observed_forecast_shared_response",
            ],
            "mean_prediction_coverage": [1.0, 1.0],
            "mean_state_ood_share": [0.0, 0.0],
            "force_base_share": [0.0, 0.0],
        }
    )

    table, payload = wf._select_controller_candidate(
        aggregate,
        suppression,
        diagnostics,
        median_delta_tie_abs_tol=2.0,
        require_post_selection_confirmation=False,
        select_no_backfill_overlay_only=False,
    )

    assert payload["selected_arm"] == "S1_observed_axes_shared_response"
    assert not bool(table.loc[table["arm"].eq("S0_baseline_static_thresholds"), "passed_selection_gates"].iloc[0])


def test_controller_candidate_selection_rejects_high_ood_latent_arm() -> None:
    aggregate = pd.DataFrame(
        {
            "arm": [
                "S1_observed_axes_shared_response",
                "S3_observed_forecast_latent_shared_response",
                "S1_observed_axes_shared_response__post_selection_overlay",
                "S3_observed_forecast_latent_shared_response__post_selection_overlay",
            ],
            "folds": [3, 3, 3, 3],
            "median_delta_net_pnl": [40.0, 80.0, 5.0, 10.0],
            "mean_delta_net_pnl": [60.0, 120.0, 10.0, 20.0],
            "q25_delta_net_pnl": [10.0, 30.0, 1.0, 2.0],
            "positive_delta_share": [2 / 3, 2 / 3, 2 / 3, 2 / 3],
            "median_delta_max_drawdown": [0.01, 0.02, 0.01, 0.02],
            "median_delta_worst_24h": [1.0, 2.0, 1.0, 2.0],
            "median_trade_retention_share": [0.95, 0.95, 0.95, 0.95],
            "median_delta_full_sl_rate": [-0.01, -0.02, -0.01, -0.02],
        }
    )
    suppression = pd.DataFrame(
        {
            "arm": [
                "S1_observed_axes_shared_response",
                "S3_observed_forecast_latent_shared_response",
                "S1_observed_axes_shared_response__post_selection_overlay",
                "S3_observed_forecast_latent_shared_response__post_selection_overlay",
            ],
            "scope": ["all", "all", "all", "all"],
            "scope_value": ["all", "all", "all", "all"],
            "realized_defensive_success": [0.1, 0.2, 0.05, 0.05],
            "positive_suppression_fold_share": [1.0, 1.0, 1.0, 1.0],
            "suppressed_candidates": [10, 10, 5, 5],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "arm": [
                "S1_observed_axes_shared_response",
                "S3_observed_forecast_latent_shared_response",
            ],
            "mean_prediction_coverage": [1.0, 0.7],
            "mean_state_ood_share": [0.0, 0.4],
            "force_base_share": [0.0, 0.2],
        }
    )

    table, payload = wf._select_controller_candidate(
        aggregate,
        suppression,
        diagnostics,
        max_mean_state_ood_share=0.10,
        require_post_selection_confirmation=False,
        select_no_backfill_overlay_only=False,
    )

    assert payload["selected_arm"] == "S1_observed_axes_shared_response"
    latent = table.loc[table["arm"].eq("S3_observed_forecast_latent_shared_response")].iloc[0]
    assert not bool(latent["passed_selection_gates"])
    assert "mean_state_ood_share_too_high" in str(latent["selection_fail_reasons"])


def test_controller_candidate_selection_uses_baseline_accepted_suppression_gate() -> None:
    aggregate = pd.DataFrame(
        {
            "arm": ["S1_observed_axes_shared_response"],
            "folds": [3],
            "median_delta_net_pnl": [40.0],
            "mean_delta_net_pnl": [60.0],
            "q25_delta_net_pnl": [10.0],
            "positive_delta_share": [2 / 3],
            "median_delta_max_drawdown": [0.01],
            "median_delta_worst_24h": [1.0],
            "median_trade_retention_share": [0.95],
            "median_delta_full_sl_rate": [-0.01],
        }
    )
    broad_suppression = pd.DataFrame(
        {
            "arm": ["S1_observed_axes_shared_response"],
            "scope": ["all"],
            "scope_value": ["all"],
            "realized_defensive_success": [10.0],
            "positive_suppression_fold_share": [1.0],
            "suppressed_candidates": [100],
            "suppressed_loss_avoided": [12.0],
            "suppressed_winner_pnl_sacrificed": [2.0],
        }
    )
    accepted_suppression = pd.DataFrame(
        {
            "arm": ["S1_observed_axes_shared_response"],
            "scope": ["all"],
            "scope_value": ["all"],
            "realized_defensive_success": [-4.0],
            "positive_suppression_fold_share": [1 / 3],
            "suppressed_candidates": [3],
            "suppressed_loss_avoided": [1.0],
            "suppressed_winner_pnl_sacrificed": [5.0],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "arm": ["S1_observed_axes_shared_response"],
            "mean_prediction_coverage": [1.0],
            "mean_state_ood_share": [0.0],
            "force_base_share": [0.0],
        }
    )

    table, payload = wf._select_controller_candidate(
        aggregate,
        broad_suppression,
        diagnostics,
        None,
        accepted_suppression,
        require_post_selection_confirmation=False,
    )

    assert payload["selected_arm"] is None
    row = table.iloc[0]
    assert row["suppression_gate_source"] == "baseline_accepted_suppression"
    assert float(row["candidate_realized_defensive_success"]) == 10.0
    assert float(row["realized_defensive_success"]) == -4.0
    assert "defensive_success_not_positive" in str(row["selection_fail_reasons"])
    assert "suppression_not_recurrent" in str(row["selection_fail_reasons"])


def test_controller_candidate_selection_rejects_full_replay_without_post_selection_confirmation() -> None:
    aggregate = pd.DataFrame(
        {
            "arm": [
                "S1_observed_axes_shared_response",
                "S1_observed_axes_shared_response__post_selection_overlay",
            ],
            "folds": [3, 3],
            "median_delta_net_pnl": [100.0, -5.0],
            "mean_delta_net_pnl": [120.0, -10.0],
            "q25_delta_net_pnl": [50.0, -20.0],
            "positive_delta_share": [2 / 3, 1 / 3],
            "median_delta_max_drawdown": [0.01, 0.0],
            "median_delta_worst_24h": [1.0, 0.0],
            "median_trade_retention_share": [0.95, 0.95],
            "median_delta_full_sl_rate": [-0.01, -0.01],
        }
    )
    suppression = pd.DataFrame(
        {
            "arm": [
                "S1_observed_axes_shared_response",
                "S1_observed_axes_shared_response__post_selection_overlay",
            ],
            "scope": ["all", "all"],
            "scope_value": ["all", "all"],
            "realized_defensive_success": [0.5, -0.2],
            "positive_suppression_fold_share": [1.0, 1 / 3],
            "suppressed_candidates": [12, 6],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "arm": ["S1_observed_axes_shared_response"],
            "mean_prediction_coverage": [1.0],
            "mean_state_ood_share": [0.0],
            "force_base_share": [0.0],
        }
    )

    table, payload = wf._select_controller_candidate(aggregate, suppression, diagnostics)

    assert payload["selected_arm"] is None
    row = table.loc[table["arm"].eq("S1_observed_axes_shared_response")].iloc[0]
    assert not bool(row["passed_selection_gates"])
    reasons = str(row["selection_fail_reasons"])
    assert "post_selection_median_delta_not_positive" in reasons
    assert "post_selection_defensive_success_not_positive" in reasons


def test_controller_candidate_selection_can_select_no_backfill_overlay() -> None:
    aggregate = pd.DataFrame(
        {
            "arm": [
                "S1_observed_axes_shared_response",
                "S1_observed_axes_shared_response__post_selection_overlay",
            ],
            "folds": [3, 3],
            "median_delta_net_pnl": [20.0, 30.0],
            "mean_delta_net_pnl": [40.0, 35.0],
            "q25_delta_net_pnl": [-5.0, 10.0],
            "positive_delta_share": [1 / 3, 2 / 3],
            "median_delta_max_drawdown": [0.01, 0.01],
            "median_delta_worst_24h": [1.0, 1.0],
            "median_trade_retention_share": [0.95, 0.92],
            "median_delta_full_sl_rate": [-0.01, -0.02],
        }
    )
    suppression = pd.DataFrame(
        {
            "arm": [
                "S1_observed_axes_shared_response",
                "S1_observed_axes_shared_response__post_selection_overlay",
            ],
            "scope": ["all", "all"],
            "scope_value": ["all", "all"],
            "realized_defensive_success": [0.2, 0.0],
            "positive_suppression_fold_share": [1.0, 0.0],
            "suppressed_candidates": [10, 0],
        }
    )
    action = pd.DataFrame(
        {
            "arm": ["S1_observed_axes_shared_response__post_selection_overlay"],
            "scope": ["all"],
            "scope_value": ["all"],
            "action_entrants": [0],
            "action_removed": [6],
            "action_removed_loss_avoided": [4.0],
            "action_removed_winner_pnl_sacrificed": [1.0],
            "action_defensive_success": [3.0],
            "positive_action_fold_share": [2 / 3],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "arm": [
                "S1_observed_axes_shared_response",
                "S1_observed_axes_shared_response__post_selection_overlay",
            ],
            "mean_prediction_coverage": [1.0, 1.0],
            "mean_state_ood_share": [0.0, 0.0],
            "force_base_share": [0.0, 0.0],
        }
    )

    table, payload = wf._select_controller_candidate(
        aggregate,
        suppression,
        diagnostics,
        action,
        select_no_backfill_overlay_only=True,
    )

    assert payload["selected_arm"] == "S1_observed_axes_shared_response__post_selection_overlay"
    full = table.loc[table["arm"].eq("S1_observed_axes_shared_response")].iloc[0]
    overlay = table.loc[
        table["arm"].eq("S1_observed_axes_shared_response__post_selection_overlay")
    ].iloc[0]
    assert "full_replay_can_promote_replacements" in str(full["selection_fail_reasons"])
    assert bool(overlay["passed_selection_gates"])
    assert float(overlay["realized_defensive_success"]) == 3.0
    assert float(overlay["suppressed_candidates"]) == 6.0


def test_controller_candidate_selection_does_not_use_overlay_action_as_direct_gate() -> None:
    aggregate = pd.DataFrame(
        {
            "arm": [
                "S1_observed_axes_shared_response",
                "S1_observed_axes_shared_response__post_selection_overlay",
            ],
            "folds": [3, 3],
            "median_delta_net_pnl": [10.0, 30.0],
            "mean_delta_net_pnl": [20.0, 35.0],
            "q25_delta_net_pnl": [5.0, 10.0],
            "positive_delta_share": [2 / 3, 2 / 3],
            "median_delta_max_drawdown": [0.01, 0.01],
            "median_delta_worst_24h": [1.0, 1.0],
            "median_trade_retention_share": [0.95, 0.95],
            "median_delta_full_sl_rate": [-0.01, -0.02],
        }
    )
    broad_suppression = pd.DataFrame(
        {
            "arm": [
                "S1_observed_axes_shared_response",
                "S1_observed_axes_shared_response__post_selection_overlay",
            ],
            "scope": ["all", "all"],
            "scope_value": ["all", "all"],
            "realized_defensive_success": [2.0, 2.0],
            "positive_suppression_fold_share": [1.0, 1.0],
            "suppressed_candidates": [50, 6],
        }
    )
    accepted_suppression = pd.DataFrame(
        {
            "arm": [
                "S1_observed_axes_shared_response",
                "S1_observed_axes_shared_response__post_selection_overlay",
            ],
            "scope": ["all", "all"],
            "scope_value": ["all", "all"],
            "realized_defensive_success": [1.0, 0.0],
            "positive_suppression_fold_share": [1.0, 0.0],
            "suppressed_candidates": [2, 0],
            "suppressed_loss_avoided": [1.0, 0.0],
            "suppressed_winner_pnl_sacrificed": [0.0, 0.0],
        }
    )
    action = pd.DataFrame(
        {
            "arm": ["S1_observed_axes_shared_response__post_selection_overlay"],
            "scope": ["all"],
            "scope_value": ["all"],
            "action_entrants": [0],
            "action_removed": [6],
            "action_removed_loss_avoided": [12.0],
            "action_removed_winner_pnl_sacrificed": [0.0],
            "action_defensive_success": [12.0],
            "positive_action_fold_share": [1.0],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "arm": [
                "S1_observed_axes_shared_response",
                "S1_observed_axes_shared_response__post_selection_overlay",
            ],
            "mean_prediction_coverage": [1.0, 1.0],
            "mean_state_ood_share": [0.0, 0.0],
            "force_base_share": [0.0, 0.0],
        }
    )

    table, payload = wf._select_controller_candidate(
        aggregate,
        broad_suppression,
        diagnostics,
        action,
        accepted_suppression,
        select_no_backfill_overlay_only=True,
    )

    assert payload["selected_arm"] is None
    overlay = table.loc[
        table["arm"].eq("S1_observed_axes_shared_response__post_selection_overlay")
    ].iloc[0]
    assert overlay["suppression_gate_source"] == "baseline_accepted_suppression"
    assert float(overlay["action_defensive_success"]) == 12.0
    assert float(overlay["realized_defensive_success"]) == 0.0
    assert "defensive_success_not_positive" in str(overlay["selection_fail_reasons"])


def test_controller_candidate_selection_treats_empty_accepted_suppression_as_no_direct_action() -> None:
    aggregate = pd.DataFrame(
        {
            "arm": ["S1_observed_axes_shared_response__post_selection_overlay"],
            "folds": [1],
            "median_delta_net_pnl": [10.0],
            "mean_delta_net_pnl": [10.0],
            "q25_delta_net_pnl": [10.0],
            "positive_delta_share": [1.0],
            "median_delta_max_drawdown": [0.0],
            "median_delta_worst_24h": [0.0],
            "median_trade_retention_share": [1.0],
            "median_delta_full_sl_rate": [0.0],
        }
    )
    broad_suppression = pd.DataFrame(
        {
            "arm": ["S1_observed_axes_shared_response__post_selection_overlay"],
            "scope": ["all"],
            "scope_value": ["all"],
            "realized_defensive_success": [10.0],
            "positive_suppression_fold_share": [1.0],
            "suppressed_candidates": [5],
            "suppressed_loss_avoided": [10.0],
            "suppressed_winner_pnl_sacrificed": [0.0],
        }
    )
    empty_accepted = broad_suppression.iloc[0:0].copy()
    action = pd.DataFrame(
        {
            "arm": ["S1_observed_axes_shared_response__post_selection_overlay"],
            "scope": ["all"],
            "scope_value": ["all"],
            "action_entrants": [0],
            "action_removed": [5],
            "action_removed_loss_avoided": [10.0],
            "action_removed_winner_pnl_sacrificed": [0.0],
            "action_defensive_success": [10.0],
            "positive_action_fold_share": [1.0],
        }
    )

    table, payload = wf._select_controller_candidate(
        aggregate,
        broad_suppression,
        pd.DataFrame(),
        action,
        empty_accepted,
        select_no_backfill_overlay_only=True,
    )

    row = table.iloc[0]
    assert payload["selected_arm"] is None
    assert row["suppression_gate_source"] == "baseline_accepted_suppression"
    assert float(row["candidate_realized_defensive_success"]) == 10.0
    assert float(row["action_defensive_success"]) == 10.0
    assert float(row["realized_defensive_success"]) == 0.0
    assert "defensive_success_not_positive" in str(row["selection_fail_reasons"])


def test_walkforward_fold_builder_filters_sparse_validation_windows() -> None:
    dense_a = pd.date_range("2026-05-01", periods=30, freq="6h", tz="UTC")
    dense_b = pd.date_range("2026-05-20", periods=30, freq="6h", tz="UTC")
    sparse_tail = pd.to_datetime(["2026-06-08T00:00:00Z", "2026-06-09T00:00:00Z"])
    timestamps = pd.Series(list(dense_a.repeat(3)) + list(dense_b.repeat(3)) + list(sparse_tail))

    folds = wf._build_time_folds(
        timestamps,
        n_folds=3,
        min_train_days=5,
        valid_days=3,
        embargo_hours=24,
        min_valid_rows=10,
        min_valid_timestamps=4,
    )

    assert folds
    assert all(int(fold["valid_rows_available"]) >= 10 for fold in folds)
    assert all(int(fold["valid_timestamps_available"]) >= 4 for fold in folds)
    assert all(pd.Timestamp(fold["valid_start"]) < pd.Timestamp("2026-06-08T00:00:00Z") for fold in folds)


def test_walkforward_training_filter_drops_unmatured_outcomes_before_validation() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-05-01T00:00:00Z",
                    "2026-05-01T01:00:00Z",
                    "2026-05-01T02:00:00Z",
                    "2026-05-04T00:00:00Z",
                ],
                utc=True,
            ),
            "exit_timestamp": pd.to_datetime(
                [
                    "2026-05-02T00:00:00Z",
                    "2026-05-03T00:00:00Z",
                    None,
                    "2026-05-04T01:00:00Z",
                ],
                utc=True,
                errors="coerce",
            ),
        }
    )

    filtered, diag = wf._filter_matured_training_time(
        frame,
        None,
        pd.Timestamp("2026-05-02T00:00:00Z"),
        pd.Timestamp("2026-05-03T00:00:00Z"),
    )

    assert len(filtered) == 1
    assert filtered["timestamp"].iloc[0] == pd.Timestamp("2026-05-01T00:00:00Z", tz="UTC")
    assert diag["entry_filtered_rows"] == 3
    assert diag["matured_rows"] == 1
    assert diag["dropped_immature_outcome_rows"] == 2
    assert diag["missing_outcome_available_rows"] == 1
    assert diag["uses_outcome_available_timestamp"] is True
