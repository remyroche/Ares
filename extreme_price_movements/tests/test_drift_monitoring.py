from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.drift_monitoring import (
    ALL_DRIFT_METRICS,
    DRIFT_METRIC_REGISTRY,
    DRIFT_SCHEMA_VERSION,
    EXPANDED_DRIFT_METRICS,
    TIER1_DRIFT_METRICS,
    attach_recent_bar_forward_returns,
    build_metric_panel,
    build_recent_bar_metrics,
    drift_regime_feature_names,
    load_latest_drift_regime_features,
    write_live_drift_recap,
    write_lgbm_reference_drift_benchmarks,
    write_policy_drift_benchmarks,
)
from extreme_price_movements.regime_adaptor import build_regime_feature_frame


def _synthetic_policy_rows(days: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    timestamps = pd.date_range("2026-01-01", periods=days * 6, freq="4h", tz="UTC")
    symbols = np.where(np.arange(len(timestamps)) % 2 == 0, "AAA/USD:USD", "BBB/USD:USD")
    rank = pd.Series(np.sin(np.arange(len(timestamps)) / 13.0)).rank(pct=True).to_numpy()
    base = np.clip(0.2 + 0.65 * rank + rng.normal(0.0, 0.03, len(timestamps)), 0.0, 1.0)
    meta = np.clip(0.25 + 0.60 * rank + rng.normal(0.0, 0.02, len(timestamps)), 0.0, 1.0)
    gross = 0.0015 * (rank - 0.5) + rng.normal(0.0, 0.002, len(timestamps))
    net = gross - 0.00025
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "signal_bar_ts": timestamps,
            "symbol": symbols,
            "asset_class": "crypto_perp",
            "strategy_id": np.where(np.arange(len(timestamps)) % 3 == 0, "s0", "s1"),
            "base_pred": base.astype(np.float32),
            "meta_pred": meta.astype(np.float32),
            "auction_rank_pct": rank.astype(np.float32),
            "calibrated_score": meta.astype(np.float32),
            "net_return": net.astype(np.float32),
            "gross_return": gross.astype(np.float32),
            "was_traded": rank >= 0.75,
            "feature_drift_psi_core": (0.1 + 0.03 * rank).astype(np.float32),
            "feature_drift_ks_core": (0.05 + 0.02 * (1.0 - rank)).astype(np.float32),
            "raw_state_mahalanobis": (0.6 + 0.2 * rank).astype(np.float32),
            "raw_state_knn_distance": (0.7 + 0.1 * rank).astype(np.float32),
            "raw_state_min_cluster_distance": (0.4 + 0.08 * rank).astype(np.float32),
            "raw_state_reconstruction_error": (0.2 + 0.05 * rank).astype(np.float32),
            "raw_state_transition_norm": (0.12 + 0.02 * rank).astype(np.float32),
            "raw_state_svd_00": (0.1 * np.sin(rank * np.pi)).astype(np.float32),
            "raw_state_svd_01": (0.1 * np.cos(rank * np.pi)).astype(np.float32),
            "soft_label": np.clip(meta + rng.normal(0.0, 0.03, len(timestamps)), 0.0, 1.0).astype(np.float32),
            "near_barrier_ambiguity": (0.2 + 0.2 * (1.0 - np.abs(rank - 0.5) * 2.0)).astype(np.float32),
            "unstable_label_score": (0.05 + 0.1 * (1.0 - rank)).astype(np.float32),
            "mfe": (0.003 + 0.002 * rank).astype(np.float32),
            "mae": (-0.002 - 0.001 * (1.0 - rank)).astype(np.float32),
            "mfe_time_frac": (0.2 + 0.3 * (1.0 - rank)).astype(np.float32),
            "mae_time_frac": (0.4 + 0.2 * rank).astype(np.float32),
            "mfe_before_mae": (rank > 0.55).astype(np.float32),
            "rare_leaf_fraction": (0.02 + 0.03 * (1.0 - rank)).astype(np.float32),
            "leaf_surprisal_mean": (1.0 + 0.2 * rank).astype(np.float32),
            "leaf_surprisal_p90": (1.2 + 0.3 * rank).astype(np.float32),
            "leaf_train_freq_mean": (0.8 - 0.2 * rank).astype(np.float32),
            "leaf_train_freq_p10": (0.5 - 0.1 * rank).astype(np.float32),
            "leaf_target_std_mean": (0.1 + 0.05 * rank).astype(np.float32),
            "leaf_target_iqr_mean": (0.08 + 0.03 * rank).astype(np.float32),
            "leaf_centroid_dist_norm_mean": (0.3 + 0.1 * rank).astype(np.float32),
            "leaf_centroid_dist_norm_p90": (0.4 + 0.1 * rank).astype(np.float32),
            "contrib_abs_sum": (1.5 + 0.1 * rank).astype(np.float32),
            "contrib_l2_norm": (0.8 + 0.1 * rank).astype(np.float32),
            "contrib_entropy": (0.7 + 0.05 * (1.0 - rank)).astype(np.float32),
            "contrib_balance": (-0.2 + 0.4 * rank).astype(np.float32),
            "archetype_contrib_svd_00": (0.15 * np.sin(rank)).astype(np.float32),
            "archetype_contrib_svd_01": (0.12 * np.cos(rank)).astype(np.float32),
            "score_path_std": (0.02 + 0.01 * rank).astype(np.float32),
            "rank_path_std": (0.01 + 0.01 * (1.0 - rank)).astype(np.float32),
            "rank_bin_lift_oof": (0.9 + 0.2 * rank).astype(np.float32),
            "rank_bin_net_ret_oof": (0.001 * (rank - 0.5)).astype(np.float32),
            "uncertainty_score": (0.3 + 0.2 * (1.0 - np.abs(rank - 0.5) * 2.0)).astype(np.float32),
            "prob_uncertainty": (0.25 - np.abs(meta - 0.5) * 0.2).astype(np.float32),
            "entropy": (0.6 + 0.1 * (1.0 - rank)).astype(np.float32),
            "disagreement": (0.03 + 0.02 * rank).astype(np.float32),
            "estimated_hit_rate": meta.astype(np.float32),
            "estimated_ev_net_return": (0.001 * (rank - 0.5)).astype(np.float32),
            "expected_edge_bps": (15.0 + 10.0 * rank).astype(np.float32),
            "spread_bps": (12.0 + 2.0 * (1.0 - rank)).astype(np.float32),
            "entry_gap_bps": (8.0 + 1.5 * rank).astype(np.float32),
            "slippage_bps": (2.0 + rank).astype(np.float32),
            "fill_slippage_bps": (1.0 + rank).astype(np.float32),
            "shadow_exit_vs_live_stop_bps": (3.0 + rank).astype(np.float32),
            "shadow_live_entry_gap_bps": (1.0 + 0.2 * rank).astype(np.float32),
            "shadow_live_exit_gap_bps": (1.5 + 0.3 * rank).astype(np.float32),
            "stop_update_failed": (rank < 0.02).astype(np.float32),
            "rv_24h": (0.02 + 0.01 * rank).astype(np.float32),
        }
    )


def test_drift_registry_requires_severity_direction():
    assert DRIFT_METRIC_REGISTRY
    assert {spec.severity_direction for spec in DRIFT_METRIC_REGISTRY.values()} <= {
        "high",
        "low",
        "two_sided",
    }
    assert "target_return_mean" in DRIFT_METRIC_REGISTRY
    assert "uncertainty_score_mean" in DRIFT_METRIC_REGISTRY
    assert "raw_state_knn_distance_mean" in DRIFT_METRIC_REGISTRY
    assert "leaf_target_dispersion" in DRIFT_METRIC_REGISTRY
    assert "adwin_net_return_shift" in DRIFT_METRIC_REGISTRY
    assert not any(spec.family == "execution_drift" for spec in DRIFT_METRIC_REGISTRY.values())
    assert len(ALL_DRIFT_METRICS) == len(TIER1_DRIFT_METRICS) + len(EXPANDED_DRIFT_METRICS)
    assert all(spec.tier == 1 for spec in TIER1_DRIFT_METRICS)
    assert all(spec.tier > 1 for spec in EXPANDED_DRIFT_METRICS)


def test_policy_benchmarks_and_live_recap_emit_regime_features(tmp_path):
    rows = _synthetic_policy_rows()
    benchmark_dir = tmp_path / "artifacts" / "run" / "drift_benchmarks"
    summary = write_policy_drift_benchmarks(
        rows,
        output_dir=benchmark_dir,
        provenance={
            "asof_ts": pd.Timestamp("2026-05-15T00:00:00Z"),
            "label_maturity_cutoff_ts": pd.Timestamp("2026-05-15T00:00:00Z"),
            "model_run_id": "unit_model",
            "policy_run_id": "unit_policy",
        },
    )

    assert summary["schema_version"] == DRIFT_SCHEMA_VERSION
    assert summary["baseline_rows"] > 0
    assert summary["all_metric_count"] > summary["tier1_metric_count"]
    assert (benchmark_dir / "metric_registry.json").exists()
    assert (benchmark_dir / "metric_baselines.parquet").exists()
    assert (benchmark_dir / "daily_cross_section.parquet").exists()
    daily_metrics = pd.read_parquet(benchmark_dir / "daily_metric_observations.parquet")
    assert "raw_state_knn_distance_mean" in set(daily_metrics["metric_name"])
    assert "leaf_target_dispersion" in set(daily_metrics["metric_name"])

    ledger_path = tmp_path / "prediction_ledger.parquet"
    rows.tail(90).to_parquet(ledger_path, index=False)
    recap = write_live_drift_recap(
        ledger_path=ledger_path,
        output_root=tmp_path / "live_state" / "drift_monitoring",
        benchmark_dir=benchmark_dir,
        asof_ts=pd.Timestamp("2026-04-30T00:00:00Z"),
        model_run_id="unit_model",
        policy_run_id="unit_policy",
    )

    assert recap["scored_metric_rows"] > 0
    assert recap["regime_feature_rows"] == 2
    assert recap["recent_bar_metric_rows"] > 0
    recent = pd.read_parquet(tmp_path / "live_state" / "drift_monitoring" / "latest" / "recent_bar_metrics.parquet")
    assert {"1h", "3h", "6h", "12h", "24h"} <= set(recent["window"])
    assert "hit_rate" in recent.columns
    assert "rank_ic" in recent.columns
    assert "top05_hit_rate" in recent.columns
    assert "top20_hit_rate" in recent.columns
    latest = load_latest_drift_regime_features(live_data_root=tmp_path)
    assert set(latest["symbol"]) == {"AAA/USD:USD", "BBB/USD:USD"}
    assert "drift_feature_regime_drift_score_1d" in latest.columns
    assert "drift_feature_regime_drift_all_score_1d" in latest.columns
    assert "drift_feature_regime_drift_feature_psi_mean_14d" in latest.columns
    assert "drift_feature_regime_drift_raw_state_reconstruction_error_14d" not in latest.columns
    assert "drift_feature_regime_drift_raw_state_knn_distance_mean_7d" in latest.columns
    assert "drift_feature_regime_drift_raw_state_knn_distance_mean_14d" not in latest.columns
    assert "drift_feature_regime_drift_coverage_ratio_1d" not in latest.columns
    assert "drift_performance_drift_realized_return_top_decile_1d" in latest.columns
    assert "drift_prediction_drift_score_7d" in latest.columns
    assert not any(str(col).startswith("drift_execution_drift_") for col in latest.columns)
    assert latest.filter(like="drift_").notna().any().any()


def test_weekly_metric_panel_keeps_utc_periods():
    panel = build_metric_panel(
        _synthetic_policy_rows(days=21),
        freq="W",
        asof_ts="2026-02-01T00:00:00Z",
        label_maturity_cutoff_ts="2026-02-01T00:00:00Z",
    )

    assert not panel.empty
    assert str(panel["period_start_ts"].dt.tz) == "UTC"
    assert str(panel["period_end_ts"].dt.tz) == "UTC"


def test_label_dependent_drift_metrics_respect_maturity_cutoff():
    ts = pd.date_range("2026-06-01", periods=4, freq="h", tz="UTC")
    rows = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "AAA/USD:USD",
            "base_pred": [0.10, 0.20, 0.30, 0.40],
            "meta_pred": [0.55, 0.60, 0.65, 0.70],
            "auction_rank_pct": [0.10, 0.40, 0.70, 0.95],
            "net_return": [0.01, 0.02, 0.03, 1.00],
            "gross_return": [0.011, 0.021, 0.031, 1.001],
            "estimated_hit_rate": [0.55, 0.60, 0.65, 0.70],
            "estimated_ev_net_return": [0.01, 0.02, 0.03, 1.00],
            "was_traded": [True, True, True, True],
            "label_available_ts": [
                "2026-06-01T01:00:00Z",
                "2026-06-01T02:00:00Z",
                "2026-06-01T03:00:00Z",
                "2026-06-02T00:00:00Z",
            ],
        }
    )

    panel = build_metric_panel(
        rows,
        freq="D",
        asof_ts="2026-06-01T12:00:00Z",
        label_maturity_cutoff_ts="2026-06-01T12:00:00Z",
    )

    metrics = panel.set_index("metric_name")
    assert metrics.loc["base_prediction_mean", "metric_count"] == 4
    assert metrics.loc["target_return_mean", "metric_count"] == 3
    assert metrics.loc["net_return_per_accepted_trade", "metric_count"] == 3
    assert np.isclose(float(metrics.loc["target_return_mean", "metric_value"]), 0.02)
    assert np.isclose(float(metrics.loc["net_return_per_accepted_trade", "metric_value"]), 0.02)


def test_recent_bar_metrics_report_ic_hr_and_unmatured_status():
    ts = pd.date_range("2026-06-14T00:00:00Z", periods=8, freq="h")
    rows = pd.DataFrame(
        {
            "timestamp": ts,
            "signal_bar_ts": ts,
            "symbol": ["AAA/USD:USD", "BBB/USD:USD"] * 4,
            "strategy_id": ["s0", "s1"] * 4,
            "side": ["short"] * 8,
            "auction_rank_pct": np.linspace(0.1, 0.9, 8),
            "meta_pred": np.linspace(0.55, 0.90, 8),
            "net_return": np.linspace(-0.003, 0.004, 8),
            "label_available_ts": ts + pd.Timedelta(hours=1),
        }
    )

    metrics = build_recent_bar_metrics(
        rows,
        asof_ts="2026-06-14T08:30:00Z",
        label_maturity_cutoff_ts="2026-06-14T08:30:00Z",
        model_run_id="unit_model",
        policy_run_id="unit_policy",
        min_count=2,
    )

    global_6h = metrics[(metrics["window"] == "6h") & (metrics["scope"] == "global")].iloc[0]
    assert global_6h["candidate_rows"] == 5
    assert global_6h["matured_rows"] == 5
    assert global_6h["status"] == "ok"
    assert np.isfinite(global_6h["rank_ic"])
    assert np.isfinite(global_6h["hit_rate"])
    assert np.isfinite(global_6h["top30_hit_rate"])
    assert "strategy_id" in set(metrics["scope"])
    assert "side" in set(metrics["scope"])

    immature = rows.drop(columns=["net_return", "label_available_ts"]).copy()
    immature_metrics = build_recent_bar_metrics(
        immature,
        asof_ts="2026-06-14T08:30:00Z",
        label_maturity_cutoff_ts="2026-06-14T08:30:00Z",
        attach_market_outcomes=False,
    )
    immature_global = immature_metrics[
        (immature_metrics["window"] == "6h") & (immature_metrics["scope"] == "global")
    ].iloc[0]
    assert immature_global["candidate_rows"] == 5
    assert immature_global["matured_rows"] == 0
    assert immature_global["status"] == "missing_return_column"


def test_recent_bar_metrics_attach_hourly_market_outcomes(tmp_path):
    market_root = tmp_path / "orderbook_hourly"
    market_root.mkdir()
    price_ts = pd.date_range("2026-06-14T00:00:00Z", periods=13, freq="h")
    pd.DataFrame({"mid": np.linspace(100.0, 112.0, len(price_ts))}, index=price_ts).to_parquet(
        market_root / "AAA_USD_USD.parquet"
    )
    rows = pd.DataFrame(
        {
            "signal_bar_ts": [pd.Timestamp("2026-06-14T00:00:00Z")] * 4,
            "symbol": ["AAA/USD:USD"] * 4,
            "strategy_id": ["long_head"] * 4,
            "side": ["long"] * 4,
            "auction_rank_pct": [0.1, 0.4, 0.8, 0.95],
        }
    )

    enriched = attach_recent_bar_forward_returns(
        rows,
        asof_ts="2026-06-14T12:00:00Z",
        market_data_root=market_root,
        horizon_hours=10,
    )
    assert set(enriched["recent_bar_outcome_status"]) == {"ok"}
    assert np.allclose(enriched["recent_forward_return"], 0.10)

    metrics = build_recent_bar_metrics(
        rows,
        asof_ts="2026-06-14T12:00:00Z",
        label_maturity_cutoff_ts="2026-06-14T12:00:00Z",
        market_data_root=market_root,
        outcome_horizon_hours=10,
        windows_hours={"12h": 12},
        min_count=2,
    )
    global_row = metrics[(metrics["window"] == "12h") & (metrics["scope"] == "global")].iloc[0]
    assert global_row["status"] == "ok"
    assert global_row["matured_rows"] == 4
    assert np.isclose(global_row["hit_rate"], 1.0)

    pending = build_recent_bar_metrics(
        rows,
        asof_ts="2026-06-14T05:00:00Z",
        label_maturity_cutoff_ts="2026-06-14T05:00:00Z",
        market_data_root=market_root,
        outcome_horizon_hours=10,
        windows_hours={"6h": 6},
        min_count=2,
    )
    pending_row = pending[(pending["window"] == "6h") & (pending["scope"] == "global")].iloc[0]
    assert pending_row["status"] == "awaiting_maturity"
    assert pending_row["pending_label_rows"] == 4


def test_live_recap_auto_builds_missing_lgbm_reference_benchmarks(tmp_path):
    artifact_root = tmp_path / "artifacts" / "run_a"
    ref_dir = artifact_root / "lgbm_reference" / "base" / "head_a"
    ref_dir.mkdir(parents=True)
    ts = pd.date_range("2026-01-01", periods=60, freq="D", tz="UTC")
    pd.DataFrame(
        {
            "timestamp": ts,
            "asset": ["AAA/USD:USD"] * len(ts),
            "score": np.linspace(0.2, 0.9, len(ts)),
            "rank_pct": np.linspace(0.1, 0.95, len(ts)),
            "target": np.where(np.arange(len(ts)) % 2 == 0, 1, 0),
            "return": np.linspace(-0.002, 0.004, len(ts)),
            "regime_centroid_similarity_train": np.linspace(0.95, 1.0, len(ts)),
            "feature_drift_psi_core": np.linspace(0.01, 0.02, len(ts)),
            "feature_drift_cov_shift": np.linspace(0.001, 0.005, len(ts)),
        }
    ).to_parquet(ref_dir / "lgbm_reference_sample.parquet", index=False)
    (ref_dir / "manifest.json").write_text('{"strategy_id":"head_a","reference_rows":60,"oof_rows":60}')
    ledger = tmp_path / "prediction_ledger.parquet"
    pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-03-01T00:00:00Z")],
            "signal_bar_ts": [pd.Timestamp("2026-03-01T00:00:00Z")],
            "symbol": ["AAA/USD:USD"],
            "strategy_id": ["head_a"],
            "side": ["long"],
            "auction_rank_pct": [0.9],
            "meta_pred": [0.8],
            "base_lgbm_regime_centroid_similarity_train": [0.96],
            "base_lgbm_feature_drift_cov_shift": [0.004],
        }
    ).to_parquet(ledger, index=False)

    recap = write_live_drift_recap(
        ledger_path=ledger,
        output_root=tmp_path / "live_state" / "drift_monitoring",
        benchmark_dir=artifact_root / "drift_benchmarks",
        asof_ts="2026-03-02T00:00:00Z",
        model_run_id="run_a",
        policy_run_id="run_a",
    )

    assert recap["benchmark_available"] is True
    assert recap["benchmark_rows"] > 0
    assert (artifact_root / "drift_benchmarks" / "metric_baselines.parquet").exists()


def test_regime_adaptor_accepts_drift_monitoring_features():
    drift_cols = [
        "drift_feature_regime_drift_raw_state_knn_distance_mean_7d",
        "drift_model_internal_drift_leaf_target_dispersion_7d",
        "drift_residual_drift_adwin_net_return_shift_7d",
        "drift_prediction_drift_base_meta_disagreement_mean_7d",
        "drift_feature_regime_drift_score_7d",
    ]
    available = set(drift_regime_feature_names())
    assert set(drift_cols) <= available
    frame = pd.DataFrame({name: np.linspace(0.1, 0.5, 4, dtype=np.float32) for name in drift_cols})
    regime_frame, mapping = build_regime_feature_frame(frame)

    for name in drift_cols:
        assert name in regime_frame.columns
        assert mapping[name] == name
        assert np.allclose(regime_frame[name].to_numpy(dtype=np.float32), frame[name].to_numpy(dtype=np.float32))


def test_regime_adaptor_contract_prunes_lgbm_and_ebm_consolidated_features():
    from extreme_price_movements import config as cfg
    from extreme_price_movements import regime_adaptor as ra
    from extreme_price_movements.model_drift_features import MODEL_DRIFT_FEATURE_KEYS

    assert len(cfg.REGIME_ADAPTOR_LGBM_INTERNAL_TOP25_METRIC_KEYS) == 30
    assert len(cfg.REGIME_ADAPTOR_LGBM_INTERNAL_FEATURE_KEYS) == 60
    assert "BASE_LGBM_META_UNCERTAINTY_FEATURE_KEYS" in cfg.CFG["meta_shared_feature_keys"]
    assert "base_lgbm_uncertainty_score" in cfg.CFG[
        "BASE_LGBM_META_UNCERTAINTY_FEATURE_KEYS"
    ]
    assert "base_lgbm_inference_drift_score" in cfg.CFG[
        "BASE_LGBM_META_UNCERTAINTY_FEATURE_KEYS"
    ]
    assert "BASE_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS" in cfg.CFG["meta_shared_feature_keys"]
    assert "base_lgbm_predictive_atlas_ic" in cfg.CFG[
        "BASE_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS"
    ]
    assert "base_lgbm_predictive_atlas_hit_rate_surprise" in cfg.CFG[
        "BASE_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS"
    ]
    assert "base_lgbm_predictive_atlas_hit_rate_surprise_z" in cfg.CFG[
        "BASE_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS"
    ]
    assert "meta_lgbm_predictive_atlas_ic" in cfg.CFG[
        "META_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS"
    ]
    assert "meta_lgbm_predictive_atlas_hit_rate_surprise" in cfg.CFG[
        "META_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS"
    ]
    for key in cfg.CFG["META_LGBM_PREDICTIVE_ATLAS_FEATURE_KEYS"]:
        assert key in cfg.REGIME_ADAPTOR_BASE_FEATURE_KEYS
        assert key in ra.REGIME_FEATURE_ORDER
    for key in MODEL_DRIFT_FEATURE_KEYS:
        assert f"meta_lgbm_{key}" in cfg.REGIME_ADAPTOR_BASE_FEATURE_KEYS
        assert f"meta_lgbm_{key}" in ra.REGIME_FEATURE_ORDER
    assert "meta_lgbm_feature_drift_psi_core" in ra.REGIME_FEATURE_ORDER
    assert "meta_lgbm_feature_drift_ks_core" in ra.REGIME_FEATURE_ORDER
    assert cfg.REGIME_ADAPTOR_EBM_CONSOLIDATED_FEATURE_KEYS == []
    assert 25 <= len(cfg.REGIME_ADAPTOR_GLOBAL_FEATURE_KEYS) <= 35
    assert 45 <= len(cfg.REGIME_ADAPTOR_ASSET_FEATURE_KEYS) <= 60
    assert not any("global_ebm_" in name for name in ra.REGIME_FEATURE_ORDER)
    assert not any("asset_ebm_" in name for name in ra.REGIME_FEATURE_ORDER)
    assert not any("cross_sectional_return_dispersion" in name for name in cfg.REGIME_ADAPTOR_GLOBAL_FEATURE_KEYS)
    assert not any("_10d" in name or "_15d" in name for name in cfg.REGIME_ADAPTOR_ASSET_FEATURE_KEYS)
    assert len(ra.REGIME_FEATURE_ORDER) < 550
