from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.regime_adaptor import (
    apply_regime_adaptor,
    audit_rolling_regime_readiness,
    fit_regime_adaptor,
    load_regime_adaptor,
    save_regime_adaptor_outputs,
)


def test_regime_adaptor_training_live_parity(tmp_path):
    n = 240
    rng = np.random.RandomState(7)
    ts = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    symbols = np.where(np.arange(n) % 2 == 0, "AAA_USDT", "BBB_USDT")
    frame = pd.DataFrame(
        {
            "rv_24h": rng.lognormal(-4.0, 0.3, n).astype(np.float32),
            "ret1h": rng.normal(0.0, 0.01, n).astype(np.float32),
            "rv_6h": rng.lognormal(-4.2, 0.25, n).astype(np.float32),
            "adx_14": rng.uniform(5, 35, n).astype(np.float32),
            "trend_regime": rng.normal(0.0, 1.0, n).astype(np.float32),
            "dist_ema_fast": rng.normal(0.0, 1.0, n).astype(np.float32),
            "dist_ema_slow": rng.normal(0.0, 1.0, n).astype(np.float32),
            "loc_vwap_dev_z_24": rng.normal(0.0, 1.0, n).astype(np.float32),
            "dist_prior_day_low": rng.normal(0.0, 1.0, n).astype(np.float32),
            "dist_prior_day_high": rng.normal(0.0, 1.0, n).astype(np.float32),
            "rvol_z": rng.normal(0.0, 1.0, n).astype(np.float32),
            "spectral_entropy_ret_24": rng.uniform(0, 1, n).astype(np.float32),
            "volume": rng.lognormal(10.0, 0.4, n).astype(np.float32),
            "atr_pct": rng.lognormal(-4.0, 0.2, n).astype(np.float32),
        }
    )
    pred = pd.Series(rng.normal(0.0, 1.0, n)).rank(pct=True).to_numpy()
    returns = (0.002 * (pred - 0.5) + rng.normal(0.0, 0.01, n)).astype(np.float32)

    fit = fit_regime_adaptor(
        frame,
        pred,
        returns,
        ts,
        symbols,
        strategy_id="long_test_strategy",
        model_name="unit",
    )
    artifact_path = save_regime_adaptor_outputs(
        str(tmp_path), "run", "long_test_strategy", fit
    )
    artifact = load_regime_adaptor(artifact_path)

    training_apply = apply_regime_adaptor(frame, pred, artifact, ts, symbols)
    live_apply = apply_regime_adaptor(frame.copy(), pred.copy(), artifact, ts, symbols)

    assert np.allclose(
        training_apply["regime_weight"],
        live_apply["regime_weight"],
        atol=1e-10,
        rtol=0.0,
    )
    assert np.array_equal(training_apply["eligible"], live_apply["eligible"])


def test_bad_regime_panel_is_pooled_and_strict_point_in_time():
    from extreme_price_movements.regime_adaptor import (
        build_rolling_bad_regime_panel,
        compute_hit_rate_surprise,
    )

    ts = pd.date_range("2025-01-01", periods=90, freq="D", tz="UTC")
    trades = pd.DataFrame(
        {
            "timestamp": np.tile(ts, 2),
            "strategy_id": "s",
            "symbol": np.repeat(["AAA", "BBB"], len(ts)),
            "net_pnl": np.r_[np.ones(45), np.full(45, -2.0), np.ones(90)],
            "wallet_return": np.r_[
                np.full(45, 0.01), np.full(45, -0.03), np.full(90, 0.01)
            ],
            "meta_pred_calibrated": 0.5,
        }
    )
    feature_frame = pd.DataFrame(
        {
            "timestamp": np.tile(ts, 2),
            "symbol": np.repeat(["AAA", "BBB"], len(ts)),
            "asset_rv_mean_7d": np.r_[np.arange(len(ts)), np.arange(len(ts)) + 1000],
            "market_breadth_7d": np.repeat(np.linspace(0.1, 0.9, len(ts)), 2),
        }
    )
    panel, artifact = build_rolling_bad_regime_panel(
        trades, feature_frame, strategy_id="s", timestamp_col="timestamp"
    )
    assert set(panel["horizon_days"]) == {3, 5}
    assert {"strategy_id", "symbol", "anchor_date", "horizon_days"}.issubset(
        panel.columns
    )
    assert artifact["schema_version"] == "rolling_bad_regime_v2"
    anchor = pd.Timestamp(panel.iloc[0]["anchor_date"])
    assert panel.iloc[0]["prior_30d_strategy_asset_trade_count"] == len(
        trades[
            (trades["symbol"] == panel.iloc[0]["symbol"])
            & (trades["timestamp"] < anchor)
            & (trades["timestamp"] >= anchor - pd.Timedelta(days=30))
        ]
    )

    surprise = compute_hit_rate_surprise([1.0, -1.0, 2.0], [0.25, 0.5, 0.75])
    assert surprise["wins"] == 2
    assert np.isclose(surprise["expected_wins"], 1.5)
    assert np.isclose(surprise["variance"], 0.25 * 0.75 + 0.5 * 0.5 + 0.75 * 0.25)


def test_bad_regime_label_conditions_and_global_asset_features():
    from extreme_price_movements.regime_adaptor import (
        add_consolidated_ebm_regime_features,
        compute_bad_regime_label,
    )

    assert compute_bad_regime_label(
        future_horizon_wallet_pnl=-0.01,
        future_horizon_maxDD=0.0,
        future_horizon_hit_rate_surprise_z=0.0,
        horizon_days=3,
    )
    assert compute_bad_regime_label(
        future_horizon_wallet_pnl=0.01,
        future_horizon_maxDD=0.10,
        future_horizon_hit_rate_surprise_z=0.0,
        horizon_days=3,
    )
    assert compute_bad_regime_label(
        future_horizon_wallet_pnl=0.01,
        future_horizon_maxDD=0.0,
        future_horizon_hit_rate_surprise_z=-1.5,
        horizon_days=5,
    )

    ts = pd.date_range("2025-01-01", periods=20, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "symbol": np.where(np.arange(20) % 2 == 0, "AAA", "BBB"),
            "ebm_unc_logodds_var": np.linspace(0.1, 2.0, 20),
            "ebm_unc_pi_width": np.linspace(0.1, 2.0, 20),
            "ebm_unc_conflict_norm": np.linspace(0.0, 1.0, 20),
            "ebm_unc_friction_weight": np.linspace(1.0, 0.0, 20),
            "ebm_unc_support_min": np.r_[np.full(10, 100.0), np.full(10, 1.0)],
            "ebm_unc_concentration": np.linspace(0.0, 1.0, 20),
            "ebm_unc_interaction_share": np.linspace(0.0, 1.0, 20),
        },
        index=ts,
    )
    out, mapping = add_consolidated_ebm_regime_features(
        frame, ts, symbols=frame["symbol"]
    )
    assert "asset_ebm_unc_dispersion_mean_7d" in out.columns
    assert "global_ebm_unc_dispersion_mean_7d" in out.columns
    assert "1.0-ebm_unc_friction_weight" in mapping["ebm_conflict"]
    same_time = out.index == ts[-1]
    assert not out.loc[same_time, "global_ebm_unc_dispersion_mean_7d"].empty


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


def test_score_sign_objective_and_global_rank_are_correct():
    from extreme_price_movements.regime_adaptor import (
        combine_meta_bad_regime_scores,
        regime_acceptance_objective,
        select_by_final_rank,
    )

    obj = regime_acceptance_objective(
        {
            "net_pnl": 1.0,
            "sortino": 0.2,
            "maxDD": 0.0,
            "period_std": 0.0,
            "worst_loss": 0.0,
        },
        {
            "net_pnl": 0.0,
            "sortino": -1.0,
            "maxDD": 0.0,
            "period_std": 0.0,
            "worst_loss": 0.0,
        },
    )
    assert obj["valid"] is False
    assert obj["fallback_reason"] == "baseline_net_pnl_non_positive"
    assert np.isfinite(obj["objective"])

    meta = np.array([0.5, 0.5, 0.5, 0.5])
    low_bad = np.array([0.1, 0.2, 0.3, 0.4])
    high_bad = np.array([0.9, 0.8, 0.7, 0.6])
    combined = combine_meta_bad_regime_scores(
        meta,
        high_bad,
        high_bad,
        params={"global_weight": 0.5, "asset_weight": 0.5, "lambda_regime": 1.0},
    )
    combined_low = combine_meta_bad_regime_scores(
        meta,
        low_bad,
        low_bad,
        params={"global_weight": 0.5, "asset_weight": 0.5, "lambda_regime": 1.0},
    )
    assert combined["final_score_raw"][0] < combined_low["final_score_raw"][0]
    selected = select_by_final_rank(combined["final_score"], top_frac=0.25)
    assert selected.sum() >= 1
    assert selected[np.argmax(combined["final_global_rank"])]


def test_bad_regime_combination_uses_positive_logit_zscores_and_gammas():
    from extreme_price_movements.regime_adaptor import combine_meta_bad_regime_scores

    meta = np.array([0.25, 0.50, 0.75], dtype=float)
    p_global = np.array([0.2, 0.5, 0.8], dtype=float)
    p_asset = np.array([0.7, 0.5, 0.3], dtype=float)
    params = {
        "global_weight": 0.75,
        "asset_weight": 0.25,
        "lambda_regime": 0.66,
        "gamma_global": 1.5,
        "gamma_asset": 0.75,
        "interaction_weight": 0.1,
    }
    combined = combine_meta_bad_regime_scores(meta, p_global, p_asset, params=params)

    def logit(x):
        x = np.clip(np.asarray(x, dtype=float), 1e-6, 1.0 - 1e-6)
        return np.log(x / (1.0 - x))

    def zscore(x):
        x = np.asarray(x, dtype=float)
        return (x - np.mean(x)) / np.std(x)

    g_raw = zscore(logit(p_global))
    a_raw = zscore(logit(p_asset))
    g = np.maximum(g_raw, 0.0)
    a = np.maximum(a_raw, 0.0)
    w_global = params["global_weight"] / (
        params["global_weight"] + params["asset_weight"]
    )
    w_asset = params["asset_weight"] / (
        params["global_weight"] + params["asset_weight"]
    )
    expected_offset = (
        w_global * (g ** params["gamma_global"])
        + w_asset * (a ** params["gamma_asset"])
        + params["interaction_weight"] * g * a
    )
    expected_raw = logit(meta) - params["lambda_regime"] * expected_offset
    expected_score = 1.0 / (1.0 + np.exp(-expected_raw))

    assert np.allclose(combined["global_bad_regime_zscore_raw"], g_raw)
    assert np.allclose(combined["asset_bad_regime_zscore_raw"], a_raw)
    assert np.all(combined["bad_regime_offset"] >= 0.0)
    assert np.allclose(combined["bad_regime_offset"], expected_offset)
    assert np.allclose(combined["final_score_raw"], expected_raw)
    assert np.allclose(combined["final_score"], expected_score)

def test_rolling_bad_regime_fallback_and_live_precomputed_application():
    artifact = {
        "schema_version": "rolling_bad_regime_v2",
        "enable_regime_adaptor": True,
        "selected_3d_5d_blend": {"3d": 0.6, "5d": 0.4},
        "selected_combination_params": {
            "global_weight": 0.5,
            "asset_weight": 0.5,
            "lambda_regime": 1.0,
        },
    }
    pred = np.array([0.2, 0.4, 0.8])
    missing = apply_regime_adaptor(
        pd.DataFrame({"x": [1, 2, 3]}),
        pred,
        artifact,
        pd.date_range("2025-01-01", periods=3, tz="UTC"),
        ["a", "a", "a"],
    )
    assert np.allclose(missing["deployment_score_pre_rank"], pred)
    assert not np.any(missing["regime_adjustment_enabled"])
    assert missing["regime_disabled_reason"][0] == "missing_live_p_bad_regime_columns"
    assert not np.any(missing["live_required_columns_available"])
    assert "p_bad_regime_global_3d" in missing["missing_live_p_bad_regime_columns"][0]

    live = pd.DataFrame(
        {
            "p_bad_regime_global_3d": [0.9, 0.2, 0.1],
            "p_bad_regime_global_5d": [0.9, 0.2, 0.1],
            "p_bad_regime_asset_3d": [0.9, 0.2, 0.1],
            "p_bad_regime_asset_5d": [0.9, 0.2, 0.1],
        }
    )
    applied = apply_regime_adaptor(
        live,
        pred,
        artifact,
        pd.date_range("2025-01-01", periods=3, tz="UTC"),
        ["a", "b", "c"],
    )
    assert np.all(applied["regime_adjustment_enabled"])
    assert applied["deployment_score_pre_rank"][0] < pred[0]
    assert "local_batch_rank" in applied
    assert "final_rank" not in applied
    assert "final_global_rank" not in applied
    assert applied["rank_scope"][0] == "local_batch"


def test_no_trade_future_windows_are_unlabelled_and_fit_outputs_wide_oof():
    from extreme_price_movements.regime_adaptor import (
        build_rolling_bad_regime_panel,
        fit_rolling_regime_adaptor,
    )

    ts = pd.date_range("2025-01-01", periods=70, freq="D", tz="UTC")
    trades = pd.DataFrame(
        {
            "timestamp": ts[:35],
            "strategy_id": "s",
            "symbol": "AAA",
            "net_pnl": 1.0,
            "wallet_return": 0.01,
            "meta_pred_calibrated": 0.5,
        }
    )
    panel, artifact = build_rolling_bad_regime_panel(trades, strategy_id="s")
    assert "future_horizon_trade_count" in panel.columns
    assert (
        panel.loc[panel["future_horizon_trade_count"] < 1, "bad_regime_label"]
        .isna()
        .all()
    )
    assert artifact["no_trade_label_policy"].startswith("bad_regime_label is NaN")

    anchors = pd.date_range("2025-01-01", periods=48, freq="D", tz="UTC")
    rows = []
    for anchor_i, anchor in enumerate(anchors):
        for sym_i, sym in enumerate(["AAA", "BBB"]):
            for horizon in (3, 5):
                bad = int((anchor_i + sym_i + horizon) % 3 == 0)
                rows.append(
                    {
                        "strategy_id": "s",
                        "symbol": sym,
                        "anchor_date": anchor,
                        "horizon_days": horizon,
                        "market_breadth_24h": anchor_i / 50.0,
                        "asset_rv_mean_24h": sym_i + anchor_i / 100.0,
                        "prior_3d_strategy_asset_pnl": 0.01 * (1 - bad),
                        "bad_regime_label": bad,
                        "future_horizon_wallet_pnl": -0.01 if bad else 0.02,
                        "future_horizon_trade_count": 2,
                        "meta_pred_calibrated": 0.4 + 0.01 * anchor_i,
                    }
                )
    fit = fit_rolling_regime_adaptor(
        pd.DataFrame(rows),
        global_feature_columns=["market_breadth_24h"],
        asset_feature_columns=["asset_rv_mean_24h", "prior_3d_strategy_asset_pnl"],
        optuna_trials=0,
    )
    assert fit["schema_version"] == "rolling_bad_regime_v2"
    assert fit["global_classifier_label_definition"].startswith("mean(strategy-symbol")
    assert fit["oof_p_bad_regime_predictions"]
    first = fit["oof_p_bad_regime_predictions"][0]
    assert "p_bad_regime_global_3d_oof" in first
    assert "p_bad_regime_global_5d_oof" in first
    assert "combined_global_bad_regime_oof" in first


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


def test_fit_requires_trade_candidate_evaluation_to_enable():
    from extreme_price_movements.regime_adaptor import fit_rolling_regime_adaptor

    anchors = pd.date_range("2025-01-01", periods=48, freq="D", tz="UTC")
    rows = []
    for anchor_i, anchor in enumerate(anchors):
        for sym in ["AAA", "BBB"]:
            for horizon in (3, 5):
                bad = int(anchor_i % 2 == 0)
                rows.append(
                    {
                        "strategy_id": "s",
                        "symbol": sym,
                        "anchor_date": anchor,
                        "horizon_days": horizon,
                        "market_breadth_24h": float(bad),
                        "asset_rv_mean_24h": float(bad),
                        "bad_regime_label": bad,
                        "future_horizon_wallet_pnl": -0.01 if bad else 0.02,
                        "future_horizon_trade_count": 2,
                        "meta_pred_calibrated": 0.5,
                    }
                )
    fit = fit_rolling_regime_adaptor(
        pd.DataFrame(rows),
        global_feature_columns=["market_breadth_24h"],
        asset_feature_columns=["asset_rv_mean_24h"],
        optuna_trials=0,
    )
    assert fit["enabled"] is False
    assert fit["reason"] == "missing_trade_candidate_oof_evaluation"
    assert fit["trade_candidate_eval_available"] is False
    assert fit["rank_scope"] == "local_batch"
    assert fit["global_bad_rate_threshold"] == 0.5
    assert fit["asset_bad_rate_threshold"] == 0.5
    assert "optuna_best_trial_diagnostics" in fit


def test_audit_rolling_regime_readiness_reports_integration_gates():
    artifact = {
        "schema_version": "rolling_bad_regime_v2",
        "enable_regime_adaptor": True,
        "reason": "accepted",
        "trade_candidate_eval_available": True,
        "evaluation_universe": "trade_candidates",
        "rank_scope": "local_batch",
        "rank_requirement": (
            "portfolio_global_or_per_side_rank_required_downstream_before_"
            "thresholding_or_sizing"
        ),
        "feature_key_lists": {"global": ["market_breadth_24h"]},
    }
    live = pd.DataFrame(
        {
            "p_bad_regime_global_3d": [0.1],
            "p_bad_regime_global_5d": [0.1],
            "p_bad_regime_asset_3d": [0.2],
            "p_bad_regime_asset_5d": [0.2],
            "market_breadth_24h": [0.5],
        }
    )
    downstream = pd.DataFrame(
        {"deployment_score_pre_rank": [0.8], "final_global_rank": [1.0]}
    )
    audit = audit_rolling_regime_readiness(
        artifact, live_feature_frame=live, downstream_candidate_frame=downstream
    )
    assert audit["enablement_uses_trade_candidates"] is True
    assert audit["live_required_columns_available"] is True
    assert audit["downstream_rank_available"] is True
    assert (
        audit["live_feature_missingness_by_scope"]["global"]["market_breadth_24h"]
        == 0.0
    )

    missing_live = audit_rolling_regime_readiness(
        artifact, live_feature_frame=pd.DataFrame({"p_bad_regime_global_3d": [0.1]})
    )
    assert missing_live["live_required_columns_available"] is False
    assert (
        "p_bad_regime_asset_5d"
        in missing_live["missing_live_p_bad_regime_columns"]
    )
