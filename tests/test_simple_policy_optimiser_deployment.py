import json

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.simple_policy_optimiser import (
    _allow_feature_only_policy_replay,
    _allow_final_fit_policy_generation,
    _allow_meta_oof_policy_source,
    _assert_policy_path_coverage,
    _apply_delayed_entry_execution_model,
    _apply_deployment_strategy_contract,
    _assert_deployment_has_selected_strategies,
    _apply_local_candidate_hit_rate_guard,
    _apply_simple_rank_net_ev_prefilter,
    _apply_simple_policy_calibrated_drift_risk,
    _build_deployment_payload,
    _deployment_selected_strategy_ids,
    _filter_candidates_to_deployment_strategies,
    _filter_policy_quote_rows,
    _fetch_policy_paths,
    _fit_simple_policy_calibrated_drift_risk,
    _finalise_simple_policy_candidates,
    _load_feature_rows_for_events,
    _load_policy_1m_klines_cached,
    _policy_market_data_root,
    _policy_path_finite_mask,
    _policy_prediction_source_label,
    _policy_prediction_source_uses_policy_oos,
    _policy_prediction_source_uses_precomputed_meta_oof,
    _strategy_id_matches_allowlist,
    _sync_deployment_threshold_metrics_with_active_policy,
    _summarize_policy_prediction_sources,
    _validate_delayed_entry_execution_coverage,
    _validate_policy_rows_in_trained_universe,
    _validate_policy_prediction_oos_contract,
    _write_policy_export_failure_diagnostics,
    _write_rank_threshold_band_reports,
    _write_simple_policy_candidate_metadata,
    simulate_and_score,
)


def test_simple_rank_net_ev_prefilter_is_diagnostic_only_by_default():
    rows = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
            "symbol": ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD"],
            "side": [1, 1, -1],
            "rank_pct": [0.72, 0.81, 0.96],
        }
    )
    fit = {
        "enabled": True,
        "status": "fit",
        "selected_sl_mult": 1.2,
        "selected_tp_mult": 2.5,
        "bucket_count": 2,
        "min_net_ev_bps": 0.0,
        "global_gross_ev_bps": -25.0,
        "global_net_ev_bps": -35.0,
        "global_execution_friction_bps": 10.0,
        "bucket_table": [
            {
                "bucket": 0,
                "gross_ev_bps": -25.0,
                "net_ev_bps": -35.0,
                "execution_friction_bps": 10.0,
            },
            {
                "bucket": 1,
                "gross_ev_bps": -25.0,
                "net_ev_bps": -35.0,
                "execution_friction_bps": 10.0,
            },
        ],
    }

    out, keep, summary = _apply_simple_rank_net_ev_prefilter(
        rows,
        fit,
        cost_pct=0.001,
        market_mode="spot",
        context="unit_test",
    )

    assert len(out) == len(rows)
    assert keep.tolist() == [True, True, True]
    assert summary["status"] == "diagnostic_only"
    assert summary["binding"] is False
    assert summary["rows_after"] == len(rows)
    assert summary["diagnostic_rows_after"] == 0
    assert "simple_grid_net_ev_bps" in out.columns


def test_deployment_threshold_metadata_uses_active_sl_mult():
    out = _sync_deployment_threshold_metrics_with_active_policy(
        {
            "deployment_rank_threshold": 0.7,
            "simple_sl_mult": 1.5,
            "simple_tp_mult": 2.0,
        },
        {"sl_mult": 1.2},
    )

    assert out["simple_sl_mult"] == pytest.approx(1.2)
    assert out["active_policy_sl_mult"] == pytest.approx(1.2)
    assert out["simple_sl_mult_source"] == "active_best_params.sl_mult"
    assert out["diagnostic_simple_sl_mult"] == pytest.approx(1.5)
    assert out["simple_tp_mult"] == pytest.approx(2.0)


def _result(avg_pnl: float, *, holding: dict | None = None) -> dict:
    holding = holding or {}
    metrics = {
        "top_5": {
            "avg_pnl_bankroll": avg_pnl,
            "n_trades": 10,
            "start_date": "2026-01-01",
            "end_date": "2026-01-10",
            **holding,
        },
        "top_1": {
            "n_trades": 1,
            "start_date": "2026-01-01",
            "end_date": "2026-01-10",
        },
    }
    return {
        "validation_metrics": metrics,
        "deployment_threshold_metrics": {"deployment_rank_threshold": 0.7},
        "best_params": {"sl_mult": 1.2},
        "best_size_power": 1.1,
    }


def test_simple_policy_calibrated_drift_risk_adds_policy_only_columns():
    rng = np.random.default_rng(123)
    n = 180
    risk_axis = np.linspace(0.0, 1.0, n)
    rows = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
            "symbol": np.where(np.arange(n) % 2 == 0, "BTC/USD:USD", "ETH/USD:USD"),
            "rank_pct": np.linspace(0.70, 0.99, n),
            "ood_risk_score": risk_axis,
            "similarity_support_score": 1.0 - risk_axis,
            "knn_dist_pct_k50": risk_axis + rng.normal(0.0, 0.02, n),
            "atlas_support_quality": 1.0 - risk_axis,
            "net_gain": np.where(risk_axis > 0.55, -0.015, 0.012)
            + rng.normal(0.0, 0.001, n),
        }
    )

    fit, summary = _fit_simple_policy_calibrated_drift_risk(rows)
    assert summary["enabled"] is True
    assert summary["oof_rows"] > 0
    out, apply_summary = _apply_simple_policy_calibrated_drift_risk(
        rows.drop(columns=["net_gain"]),
        fit,
        context="unit_test",
    )

    assert apply_summary["enabled"] is True
    assert "simple_policy_calibrated_bad_trade_prob" in out.columns
    assert "simple_policy_calibrated_expected_net_gain" in out.columns
    high = out.loc[risk_axis > 0.75, "simple_policy_calibrated_bad_trade_prob"].mean()
    low = out.loc[risk_axis < 0.25, "simple_policy_calibrated_bad_trade_prob"].mean()
    assert high > low


def test_strategy_allowlist_matches_side_prefixed_and_core_ids():
    core = "asset_vol_level_pct_0_20587213_compression_score_-0_99787366"

    assert _strategy_id_matches_allowlist(f"long_{core}", {core})
    assert _strategy_id_matches_allowlist(f"long_{core}", {f"long_{core}"})
    assert not _strategy_id_matches_allowlist(f"short_{core}", {f"long_{core}"})


def test_live_portfolio_contract_uses_selected_deployment_strategies_only():
    deployment_payload = {
        "strategies": [
            {
                "strategy_id": "long_alpha",
                "strategy_for_inference": "long_alpha",
                "selected": True,
            },
            {
                "strategy_id": "short_beta",
                "strategy_for_inference": "short_beta",
                "selected": True,
            },
        ],
        "rejected_strategies": [
            {
                "strategy_id": "long_rejected",
                "strategy_for_inference": "long_rejected",
                "selected": False,
                "reject_reasons": ["outside_top_2_per_side"],
            }
        ],
    }
    payload = {
        "portfolio_policy_version": "global_auction_v1",
        "strategy_contract": {
            "strategy_ids": ["long_alpha", "short_beta", "long_rejected"],
            "strategy_cores": ["alpha", "beta", "rejected"],
        },
    }

    selected = _deployment_selected_strategy_ids(deployment_payload)
    narrowed = _apply_deployment_strategy_contract(payload, selected)

    assert selected == ["long_alpha", "short_beta"]
    assert narrowed["strategy_contract"]["strategy_ids"] == [
        "long_alpha",
        "short_beta",
    ]
    assert narrowed["strategy_contract"]["strategy_cores"] == ["alpha", "beta"]


def test_portfolio_replay_refuses_empty_deployment_selection(tmp_path):
    deployment_payload = {
        "strategies": [],
        "rejected_strategies": [
            {
                "strategy_id": "long_missing_mask",
                "regime_mask_source": "missing_lgbm_mask_contract",
            }
        ],
    }

    with pytest.raises(RuntimeError, match="No deployable strategies selected"):
        _assert_deployment_has_selected_strategies(
            deployment_payload,
            candidate_path=tmp_path / "simple_policy_candidates.parquet",
        )


def test_portfolio_replay_candidates_filter_to_selected_strategies():
    candidates = pd.DataFrame(
        {
            "strategy_id": ["long_a", "short_b", "long_rejected"],
            "symbol": ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD"],
        }
    )

    filtered = _filter_candidates_to_deployment_strategies(
        candidates,
        ["long_a", "short_b"],
    )

    assert filtered["strategy_id"].tolist() == ["long_a", "short_b"]


def test_policy_quote_filter_infers_homogeneous_kraken_perp_quote(monkeypatch):
    monkeypatch.delenv("EPM_EXCHANGE", raising=False)
    monkeypatch.delenv("EXCHANGE_NAME", raising=False)
    monkeypatch.delenv("PRIMARY_EXCHANGE", raising=False)
    monkeypatch.delenv("EPM_POLICY_OOS_QUOTE_FILTER", raising=False)
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-22T10:00:00Z", "2026-01-22T10:00:00Z"],
                utc=True,
            ),
            "symbol": ["BTC/USD:USD", "SOL/USD:USD"],
            "clf": [0.7, 0.8],
        }
    )

    filtered = _filter_policy_quote_rows(rows, "perps")

    assert filtered["symbol"].tolist() == ["BTC/USD:USD", "SOL/USD:USD"]


def test_policy_market_data_root_prefers_exchange_for_perps(tmp_path, monkeypatch):
    (tmp_path / "ohlcv").mkdir()
    (tmp_path / "exchanges" / "krakenfutures" / "ohlcv").mkdir(parents=True)
    monkeypatch.setenv("EPM_EXCHANGE", "krakenfutures")

    root = _policy_market_data_root(tmp_path, "perps")

    assert root == str(tmp_path / "exchanges" / "krakenfutures")


def test_policy_path_coverage_guard_rejects_sparse_paths():
    finite = np.ones((2, 3), dtype=np.float32)
    sparse = finite.copy()
    sparse[1, :] = np.nan

    with pytest.raises(RuntimeError, match="insufficient executable policy path coverage"):
        _assert_policy_path_coverage(
            strategy_id="long_demo",
            paths=(sparse, finite, finite, finite),
        )


def test_deployment_payload_requires_current_trained_meta_model(monkeypatch):
    monkeypatch.delenv("EPM_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("EPM_REQUIRE_LGBM_REGIME_MASK_CONTRACTS", raising=False)
    payload = _build_deployment_payload(
        run_id="test-run",
        oos_results_json={
            "strategies": {
                "long_available": _result(0.02),
                "short_available": _result(0.03),
                "short_missing": _result(0.99),
            }
        },
        available_strategy_ids={"long_available", "short_available"},
    )

    selected = {row["strategy_id"] for row in payload["strategies"]}
    assert selected == {"long_available", "short_available"}

    rejected = {
        row["strategy_id"]: row.get("reject_reasons", [])
        for row in payload["rejected_strategies"]
    }
    assert "missing_trained_meta_model" in rejected["short_missing"]
    assert payload["selection_rules"]["requires_current_trained_meta_model"] is True


def test_deployment_payload_rejects_missing_lgbm_masks_for_lgbm_backend(monkeypatch):
    monkeypatch.setenv("EPM_MODEL_BACKEND", "lgbm_pipeline")

    from extreme_price_movements.offline_optimisers import params_store

    monkeypatch.setattr(
        params_store,
        "load_inference_candidate_mask_params_per_bucket",
        lambda **_: [],
    )

    payload = _build_deployment_payload(
        run_id="test-run",
        oos_results_json={
            "strategies": {
                "long_available": _result(0.02),
            }
        },
        available_strategy_ids={"long_available"},
        market_mode="perps",
    )

    assert payload["strategies"] == []
    assert payload["selection_rules"]["requires_lgbm_regime_mask_contract"] is True
    rejected = {
        row["strategy_id"]: row.get("reject_reasons", [])
        for row in payload["rejected_strategies"]
    }
    assert "missing_lgbm_mask_contract" in rejected["long_available"]


def test_deployment_payload_embeds_market_specific_lgbm_mask_contract(monkeypatch):
    calls = []
    monkeypatch.setenv("EPM_MODEL_BACKEND", "lgbm_pipeline")

    from extreme_price_movements.offline_optimisers import params_store

    def fake_loader(**kwargs):
        calls.append(dict(kwargs))
        return [
            {
                "strategy_id": "long_available",
                "trade_side": "long",
                "base_event_trigger": "(*)|(ret1h>0)|(*)",
                "mask_params": {"canonical_key": "(*)|(ret1h>0)|(*)"},
                "source_target": "manual_new_score_meta_metrics",
                "source_horizon": 10,
            }
        ]

    monkeypatch.setattr(
        params_store,
        "load_inference_candidate_mask_params_per_bucket",
        fake_loader,
    )

    payload = _build_deployment_payload(
        run_id="test-run",
        oos_results_json={
            "strategies": {
                "long_available": _result(0.02),
            }
        },
        available_strategy_ids={"long_available"},
        market_mode="perps",
    )

    assert calls and calls[0]["market_mode"] == "perps"
    assert [row["strategy_id"] for row in payload["strategies"]] == ["long_available"]
    strategy = payload["strategies"][0]
    assert strategy["regime_mask_source"] == "embedded_lgbm_final_rule_registry"
    assert strategy["lgbm_regime_mask"]["base_event_trigger"] == "(*)|(ret1h>0)|(*)"


def test_deployment_payload_recovers_source_run_lgbm_mask_contract(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("EPM_MODEL_BACKEND", "lgbm_pipeline")
    monkeypatch.setenv("EPM_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("EPM_ARTIFACT_SOURCE_RUN_ID", "source-run")

    from extreme_price_movements.offline_optimisers import params_store

    monkeypatch.setattr(
        params_store,
        "load_inference_candidate_mask_params_per_bucket",
        lambda **_: [],
    )

    source_dir = tmp_path / "artifacts" / "source-run"
    source_dir.mkdir(parents=True)
    parseable_rule = "(*)|(ret1h>0.1&volume_z<=2)|(*)"
    pd.DataFrame(
        [
            {
                "strategy_id": "long_safe_model_id",
                "canonical_key": "safe_model_id",
                "base_event_trigger": parseable_rule,
                "side": "long",
                "trade_side": "long",
                "source_horizon": 10,
                "source_target": "manual_new_score_meta_metrics",
                "market_mode": "perps",
            }
        ]
    ).to_csv(source_dir / "policy_oos_retrain_strategy_source_perps.csv", index=False)

    payload = _build_deployment_payload(
        run_id="test-run",
        oos_results_json={
            "strategies": {
                "long_safe_model_id": _result(0.02),
            }
        },
        available_strategy_ids={"long_safe_model_id"},
        market_mode="perps",
    )

    assert [row["strategy_id"] for row in payload["strategies"]] == [
        "long_safe_model_id"
    ]
    strategy = payload["strategies"][0]
    assert strategy["lgbm_regime_mask"]["base_event_trigger"] == parseable_rule
    assert strategy["lgbm_regime_mask"]["mask_params"]["canonical_key"] == parseable_rule
    assert payload["selection_rules"]["ranking_space"] == (
        "rank-normalized score per-strategy rank percentiles"
    )


def test_mask_source_loader_preserves_explicit_id_and_parseable_rule(tmp_path, monkeypatch):
    from extreme_price_movements.offline_optimisers.params_store import (
        load_inference_candidate_mask_params_per_bucket,
    )

    source_csv = tmp_path / "policy_oos_retrain_strategy_source_perps.csv"
    parseable_rule = "(*)|(ret1h>0.1&volume_z<=2)|(*)"
    pd.DataFrame(
        [
            {
                "strategy_id": "long_safe_model_id",
                "canonical_key": "long_safe_model_id",
                "base_event_trigger": parseable_rule,
                "side": "long",
                "source_horizon": 10,
                "source_target": "manual_new_score_meta_metrics",
                "move_bucket": "up",
                "stage_e_rank_score": 1.0,
                "market_mode": "perps",
            }
        ]
    ).to_csv(source_csv, index=False)
    monkeypatch.setenv("EPM_MASK_STRATEGY_SOURCE_CSV", str(source_csv))

    rows = load_inference_candidate_mask_params_per_bucket(
        top_n=1,
        ranking_metric="stage_e_rank_score",
        market_mode="perps",
    )

    assert len(rows) == 1
    assert rows[0]["strategy_id"] == "long_safe_model_id"
    assert rows[0]["base_event_trigger"] == parseable_rule
    assert rows[0]["canonical_key"] == parseable_rule
    assert rows[0]["mask_params"]["canonical_key"] == parseable_rule


def test_deployment_payload_persists_realized_holding_time_metrics(monkeypatch):
    monkeypatch.delenv("EPM_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("EPM_REQUIRE_LGBM_REGIME_MASK_CONTRACTS", raising=False)
    payload = _build_deployment_payload(
        run_id="test-run",
        oos_results_json={
            "strategies": {
                "long_available": _result(
                    0.02,
                    holding={
                        "avg_holding_bars": 18.0,
                        "median_holding_bars": 12.0,
                        "p90_holding_bars": 48.0,
                        "max_holding_bars": 96.0,
                        "avg_holding_time_hours": 4.5,
                        "median_holding_time_hours": 3.0,
                        "p90_holding_time_hours": 12.0,
                        "max_holding_time_hours": 24.0,
                    },
                )
            }
        },
        available_strategy_ids={"long_available"},
    )

    strategy = payload["strategies"][0]
    assert strategy["configured_max_holding_time_hours"] == 24.0
    assert strategy["avg_holding_time_hours"] == 4.5
    assert strategy["median_holding_time_hours"] == 3.0
    assert strategy["p90_holding_time_hours"] == 12.0
    assert strategy["max_holding_time_hours"] == 24.0


def test_candidate_finalise_splits_strategy_rank_from_cross_strategy_score():
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"]
            ),
            "symbol": ["LOW_CAL", "HIGH_CAL"],
            "strategy_rank_pct": [0.91, 0.55],
            "normalized_rank_score": [0.91, 0.55],
            "calibrated_score": [0.10, 0.99],
        }
    )

    out = _finalise_simple_policy_candidates([rows], rank_floor=0.0)

    by_symbol = out.set_index("symbol")
    assert by_symbol.loc["LOW_CAL", "strategy_rank_pct"] == 0.91
    assert by_symbol.loc["HIGH_CAL", "strategy_rank_pct"] == 0.55
    assert by_symbol.loc["LOW_CAL", "normalized_rank_score"] == 0.5
    assert by_symbol.loc["HIGH_CAL", "normalized_rank_score"] == 1.0


def test_candidate_finalise_default_export_floor_starts_at_70pct_rank():
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"] * 20),
            "symbol": [f"SYM_{i}" for i in range(20)],
            "strategy_rank_pct": np.linspace(0.05, 1.0, 20),
            "normalized_rank_score": np.linspace(0.05, 1.0, 20),
            "calibrated_score": np.arange(20, dtype=float),
        }
    )

    out = _finalise_simple_policy_candidates([rows])

    assert out["normalized_rank_score"].min() == pytest.approx(0.70)
    assert len(out) == 7
    assert out["base_strategy_threshold"].unique().tolist() == [0.70]


def test_policy_prediction_source_summary_uses_actual_sources():
    summary = _summarize_policy_prediction_sources(
        ["long_dist", "short_loc"],
        {
            "long_dist": "data_perp/artifacts/run/policy_oos_predictions/policy_oos_long_dist_clf.parquet",
            "short_loc": "data_perp/artifacts/run/meta_oof/meta_oof_short_loc_clf.parquet",
        },
    )

    assert summary["source"] == "mixed prediction sources"
    assert summary["uses_policy_oos"] is True
    assert summary["uses_policy_oos_by_strategy"] == {
        "long_dist": True,
        "short_loc": False,
    }
    assert summary["uses_precomputed_meta_oof"] is True
    assert summary["uses_precomputed_meta_oof_by_strategy"] == {
        "long_dist": False,
        "short_loc": True,
    }
    assert summary["actual_source_labels"]["long_dist"] == (
        "verified policy-OOS predictions"
    )
    assert summary["actual_source_labels"]["short_loc"] == (
        "meta_oof training-window OOF predictions, not policy-OOS evidence"
    )


def test_policy_prediction_source_label_distinguishes_requested_from_actual():
    generated = "generated_from_inference_models:feature_events_no_labels"
    parquet = "data_perp/artifacts/run/meta_oof/meta_oof_long_dist_clf.parquet"

    assert _policy_prediction_source_label(generated) == (
        "diagnostic final-fit predictions on feature-only rows, not executable policy evidence"
    )
    assert _policy_prediction_source_uses_precomputed_meta_oof(generated) is False
    assert _policy_prediction_source_label(parquet) == (
        "meta_oof training-window OOF predictions, not policy-OOS evidence"
    )
    assert _policy_prediction_source_uses_precomputed_meta_oof(parquet) is True
    policy_oos = (
        "data_perp/artifacts/run/policy_oos_predictions/"
        "policy_oos_long_dist_clf.parquet"
    )
    assert _policy_prediction_source_label(policy_oos) == "verified policy-OOS predictions"
    assert _policy_prediction_source_uses_policy_oos(policy_oos) is True


def test_policy_rows_must_be_inside_trained_universe(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"] * 2),
            "symbol": ["AAA/USD:USD", "ZZZ/USD:USD"],
            "clf": [0.9, 0.8],
        }
    )

    def fake_universe(data_root, run_id):
        return {"AAA/USD:USD"}

    monkeypatch.setattr(
        "extreme_price_movements.simple_policy_optimiser.load_trained_symbol_universe",
        fake_universe,
    )
    ok, report = _validate_policy_rows_in_trained_universe(
        df,
        data_root=tmp_path,
        run_id="run_a",
        source_path=tmp_path / "policy_oos_demo.parquet",
    )

    assert ok is False
    assert report["reason"] == "policy_oos_symbols_outside_trained_universe"
    assert report["outside_sample"] == ["ZZZ/USD:USD"]


def test_diagnostic_policy_sources_are_opt_in(monkeypatch):
    monkeypatch.delenv("EPM_SIMPLE_POLICY_ALLOW_FEATURE_ONLY_REPLAY", raising=False)
    monkeypatch.delenv("EPM_SIMPLE_POLICY_ALLOW_META_OOF_POLICY_SOURCE", raising=False)
    monkeypatch.delenv("EPM_SIMPLE_POLICY_ALLOW_FINAL_FIT_POLICY_GENERATION", raising=False)
    assert _allow_feature_only_policy_replay() is False
    assert _allow_meta_oof_policy_source() is False
    assert _allow_final_fit_policy_generation() is False

    monkeypatch.setenv("EPM_SIMPLE_POLICY_ALLOW_FEATURE_ONLY_REPLAY", "1")
    monkeypatch.setenv("EPM_SIMPLE_POLICY_ALLOW_META_OOF_POLICY_SOURCE", "1")
    monkeypatch.setenv("EPM_SIMPLE_POLICY_ALLOW_FINAL_FIT_POLICY_GENERATION", "1")
    assert _allow_feature_only_policy_replay() is True
    assert _allow_meta_oof_policy_source() is True
    assert _allow_final_fit_policy_generation() is True


def test_policy_oos_contract_requires_predictions_after_train_end(tmp_path):
    source_validation = {
        "policy_optimiser_fit_end": "2026-01-19T06:00:00+00:00",
        "policy_optimiser_predict_start": "2026-01-22T10:00:00+00:00",
        "policy_optimiser_predict_end": "2026-05-22T00:00:00+00:00",
    }
    valid = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-22T10:00:00Z", "2026-05-21T23:00:00Z"]
            )
        }
    )
    source_path = tmp_path / "policy_oos_long_dist_clf.parquet"
    source_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "source_model_fit_end": "2026-01-19T06:00:00+00:00",
                "generated_from_final_fit_bundle": False,
                "model_provenance": "train_meta_frozen_model_state",
                "prediction_source": "generated_from_train_meta_state:labels",
                "candidate_rows_source": "policy_slice_feature_events",
                "executable_path_source": "simple_policy_optimiser_recomputes_from_ohlcv_and_execution_1m",
                "rank_normalization": "simple_policy_optimiser recalculates rank_pct from clf",
                "source_model_state_sha256": "abc123",
                "source_artifact_preflight": {"valid": True, "errors": []},
            }
        )
    )
    ok, diag = _validate_policy_prediction_oos_contract(
        valid,
        source_validation=source_validation,
        source_path=source_path,
    )
    assert ok is True
    assert diag["temporal_disjoint_from_train"] is True
    assert diag["model_temporal_disjoint_from_policy_oos"] is True

    leaked = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-01-19T06:00:00Z"])}
    )
    ok, diag = _validate_policy_prediction_oos_contract(
        leaked,
        source_validation=source_validation,
        source_path=tmp_path / "meta_oof_long_dist_clf.parquet",
    )
    assert ok is False
    assert diag["reason"] == "prediction_timestamps_not_strict_policy_oos"


def test_policy_oos_contract_requires_manifest_provenance(tmp_path):
    source_validation = {
        "policy_optimiser_fit_end": "2026-01-19T06:00:00+00:00",
        "policy_optimiser_predict_start": "2026-01-22T10:00:00+00:00",
        "policy_optimiser_predict_end": "2026-05-22T00:00:00+00:00",
    }
    rows = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-01-22T10:00:00Z"])}
    )
    source_path = tmp_path / "policy_oos_long_dist_clf.parquet"
    ok, diag = _validate_policy_prediction_oos_contract(
        rows,
        source_validation=source_validation,
        source_path=source_path,
    )
    assert ok is False
    assert diag["reason"] == "missing_policy_oos_manifest"

    source_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "source_model_fit_end": "2026-05-22T00:00:00+00:00",
                "generated_from_final_fit_bundle": True,
            }
        )
    )
    ok, diag = _validate_policy_prediction_oos_contract(
        rows,
        source_validation=source_validation,
        source_path=source_path,
    )
    assert ok is False
    assert diag["reason"] == "policy_oos_model_manifest_not_oos_safe"


def test_policy_oos_contract_requires_scoring_contract(tmp_path):
    source_validation = {
        "policy_optimiser_fit_end": "2026-01-19T06:00:00+00:00",
        "policy_optimiser_predict_start": "2026-01-22T10:00:00+00:00",
        "policy_optimiser_predict_end": "2026-05-22T00:00:00+00:00",
    }
    rows = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2026-01-22T10:00:00Z"])}
    )
    source_path = tmp_path / "policy_oos_long_dist_clf.parquet"
    source_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "source_model_fit_end": "2026-01-19T06:00:00+00:00",
                "generated_from_final_fit_bundle": False,
                "model_provenance": "train_meta_frozen_model_state",
                "prediction_source": "generated_from_inference_models:labels",
                "candidate_rows_source": "policy_slice_feature_events",
                "executable_path_source": "simple_policy_optimiser_recomputes_from_ohlcv_and_execution_1m",
                "rank_normalization": "simple_policy_optimiser recalculates rank_pct from clf",
                "source_model_state_sha256": "abc123",
                "source_artifact_preflight": {"valid": True, "errors": []},
            }
        )
    )

    ok, diag = _validate_policy_prediction_oos_contract(
        rows,
        source_validation=source_validation,
        source_path=source_path,
    )

    assert ok is False
    assert diag["reason"] == "policy_oos_scoring_contract_mismatch"
    assert diag["scoring_contract_ok"] is False


def test_candidate_metadata_records_delay_contract(tmp_path, monkeypatch):
    import json

    import extreme_price_movements.simple_policy_optimiser as spo

    monkeypatch.setattr(spo, "POLICY_DELAYED_ENTRY_MINUTES", 10)
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"]
            ),
            "delayed_entry_ts": pd.to_datetime(
                ["2026-01-01T00:10:00Z", "2026-01-01T01:07:00Z"]
            ),
            "entry_execution_source": [
                "delayed_1m_intraminute_proxy",
                "theoretical_15m_open",
            ],
            "delay_window_candle_count": [11.0, 8.0],
        }
    )
    out_path = tmp_path / "simple_policy_candidates_metadata.json"

    _write_simple_policy_candidate_metadata(rows, output_path=out_path)

    payload = json.loads(out_path.read_text())
    assert payload["configured_delayed_entry_minutes"] == 10
    assert payload["artifact_entry_delay_matches_config"] is False
    assert payload["artifact_entry_delay_mismatch_rows"] == 1
    assert payload["artifact_entry_delay_minutes"]["value_counts"] == {
        "7.0": 1,
        "10.0": 1,
    }
    assert payload["delay_window_complete_rows"] == 1


def test_delayed_entry_uses_nearby_1m_fallback(monkeypatch, tmp_path):
    import extreme_price_movements.simple_policy_optimiser as spo

    monkeypatch.setattr(spo, "POLICY_DELAYED_ENTRY_MINUTES", 10)
    monkeypatch.setattr(spo, "POLICY_DELAYED_ENTRY_FALLBACK_MINUTES", 3)
    monkeypatch.setattr(spo, "POLICY_DELAYED_ENTRY_ALPHA", 0.5)
    monkeypatch.setattr(spo, "POLICY_DELAYED_ENTRY_MIN_RANK", 0.5)

    def fake_load_klines(symbol, store, *, needed_ts, market_mode):
        idx = pd.to_datetime(
            [
                "2026-01-01T01:00:00Z",
                "2026-01-01T01:01:00Z",
                "2026-01-01T01:02:00Z",
                "2026-01-01T01:03:00Z",
                "2026-01-01T01:04:00Z",
                "2026-01-01T01:05:00Z",
                "2026-01-01T01:06:00Z",
                "2026-01-01T01:07:00Z",
                "2026-01-01T01:08:00Z",
                "2026-01-01T01:09:00Z",
                "2026-01-01T01:12:00Z",
                "2026-01-01T01:13:00Z",
            ],
            utc=True,
        )
        return pd.DataFrame(
            {
                "open": [100.0] * 10 + [101.0, 104.0],
                "high": [100.0] * 10 + [103.0, 105.0],
                "low": [100.0] * 10 + [100.0, 103.0],
                "close": [100.0] * 10 + [102.0, 104.0],
            },
            index=idx,
        )

    monkeypatch.setattr(spo, "_load_policy_1m_klines_cached", fake_load_klines)
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"]),
            "symbol": ["BTC/USD:USD"],
            "rank_pct": [0.9],
            "side": [1.0],
        }
    )
    f_opens = np.array([[100.0, 101.0]], dtype=np.float32)
    f_highs = np.array([[104.0, 105.0]], dtype=np.float32)
    f_lows = np.array([[99.0, 98.0]], dtype=np.float32)
    f_closes = np.array([[101.0, 102.0]], dtype=np.float32)

    out_rows, out_paths = _apply_delayed_entry_execution_model(
        rows,
        (f_opens, f_highs, f_lows, f_closes),
        data_root=str(tmp_path),
        market_mode="perps",
    )

    assert out_rows.loc[0, "entry_execution_source"] == "delayed_1m_intraminute_proxy"
    assert out_rows.loc[0, "entry_delay_fallback_minutes"] == 2.0
    assert out_rows.loc[0, "entry_delay_actual_minutes"] == 12.0
    assert out_rows.loc[0, "delayed_entry_effective_ts"] == pd.Timestamp(
        "2026-01-01T01:12:00Z"
    )
    # The path is rebuilt from the delayed observable reference. The adverse
    # intraminute proxy remains separate and is applied exactly once by the
    # executable-entry helper.
    assert out_paths[0][0, 0] == pytest.approx(101.0)
    assert out_paths[0][0, 1] == pytest.approx(104.0)
    assert out_rows.loc[0, "entry_gap_bps"] == pytest.approx(200.0)
    assert out_rows.loc[0, "entry_slippage_proxy_bps"] == pytest.approx(
        (102.0 / 101.0 - 1.0) * 10000.0
    )
    executable, half_spread, slippage, reanchor = spo._policy_executable_entry_prices(
        out_rows,
        out_paths[0][:, 0],
        out_rows["side"].to_numpy(dtype=np.float32),
        market_mode="perps",
    )
    assert executable[0] == pytest.approx(
        101.0 * (1.0 + float(reanchor[0]) / 10000.0), rel=1e-6
    )
    assert reanchor[0] == pytest.approx(half_spread[0] + slippage[0])


def test_delayed_entry_rebuilds_partial_15m_bar_without_pre_entry_extremes(
    monkeypatch, tmp_path
):
    import extreme_price_movements.simple_policy_optimiser as spo

    monkeypatch.setattr(spo, "POLICY_DELAYED_ENTRY_MINUTES", 10)
    monkeypatch.setattr(spo, "POLICY_DELAYED_ENTRY_FALLBACK_MINUTES", 0)
    monkeypatch.setattr(spo, "POLICY_DELAYED_ENTRY_ALPHA", 0.5)
    monkeypatch.setattr(spo, "POLICY_DELAYED_ENTRY_MIN_RANK", 0.5)

    def fake_load_klines(symbol, store, *, needed_ts, market_mode):
        idx = pd.date_range("2026-01-01T01:00:00Z", periods=31, freq="1min")
        frame = pd.DataFrame(
            {
                "open": np.full(len(idx), 100.0),
                "high": np.full(len(idx), 102.0),
                "low": np.full(len(idx), 99.0),
                "close": np.full(len(idx), 101.0),
            },
            index=idx,
        )
        # These extremes occur before the delayed 01:10 entry and must not be
        # present in the rebuilt first 15-minute path bar.
        frame.loc[pd.Timestamp("2026-01-01T01:05:00Z"), "high"] = 120.0
        frame.loc[pd.Timestamp("2026-01-01T01:05:00Z"), "low"] = 80.0
        return frame

    monkeypatch.setattr(spo, "_load_policy_1m_klines_cached", fake_load_klines)
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"]),
            "symbol": ["BTC/USD:USD"],
            "rank_pct": [0.9],
            "side": [1.0],
        }
    )
    paths = (
        np.array([[100.0, 101.0]], dtype=np.float32),
        np.array([[120.0, 105.0]], dtype=np.float32),
        np.array([[80.0, 98.0]], dtype=np.float32),
        np.array([[101.0, 102.0]], dtype=np.float32),
    )

    out_rows, out_paths = _apply_delayed_entry_execution_model(
        rows,
        paths,
        data_root=str(tmp_path),
        market_mode="perps",
        path_timeframe="15m",
    )

    assert out_rows.loc[0, "first_path_timestamp"] == pd.Timestamp(
        "2026-01-01T01:10:00Z"
    )
    assert out_paths[1][0, 0] == pytest.approx(102.0)
    assert out_paths[2][0, 0] == pytest.approx(99.0)
    assert out_paths[1][0, 1] == pytest.approx(105.0)
    assert out_paths[2][0, 1] == pytest.approx(98.0)


def test_1m_execution_loader_refetches_sparse_cached_range(monkeypatch, tmp_path):
    import extreme_price_movements.simple_policy_optimiser as spo

    class DummyStore:
        def __init__(self):
            self.root_dir = str(tmp_path)
            self.df = pd.DataFrame(
                {
                    "open": [100.0],
                    "high": [100.0],
                    "low": [100.0],
                    "close": [100.0],
                },
                index=pd.to_datetime(["2026-01-01T00:07:00Z"], utc=True),
            )

        def load(self, symbol, columns=None, start_ts=None, end_ts=None):
            frame = self.df.copy()
            if start_ts is not None:
                frame = frame.loc[frame.index >= pd.Timestamp(start_ts)]
            if end_ts is not None:
                frame = frame.loc[frame.index <= pd.Timestamp(end_ts)]
            return frame

        def save_partitioned(self, symbol, df, defer_compact=False):
            self.df = pd.concat([self.df, df]).sort_index()
            self.df = self.df.loc[~self.df.index.duplicated(keep="last")]

        def compact_partition(self, symbol, year):
            return None

        def _downcast(self, df):
            return df

    spo._POLICY_1M_KLINES_CACHE.clear()
    monkeypatch.setenv("EPM_SIMPLE_POLICY_1M_DOWNLOAD", "1")
    monkeypatch.setenv("EPM_SIMPLE_POLICY_1M_FETCH_BUCKET_MINUTES", "60")
    monkeypatch.setattr(spo, "_policy_execution_exchange", lambda market_mode: object())
    fetch_calls = []

    def fake_fetch(exchange, symbol, since_ms, until_ms, timeframe="1m", limit=1000):
        fetch_calls.append((since_ms, until_ms))
        idx = pd.date_range(
            "2026-01-01T00:00:00Z",
            periods=60,
            freq="1min",
            tz="UTC",
        )
        return pd.DataFrame(
            {
                "open": np.arange(60, dtype=np.float64) + 100.0,
                "high": np.arange(60, dtype=np.float64) + 101.0,
                "low": np.arange(60, dtype=np.float64) + 99.0,
                "close": np.arange(60, dtype=np.float64) + 100.5,
            },
            index=idx,
        )

    monkeypatch.setattr(spo, "_fetch_ohlcv_paged", fake_fetch)
    needed = pd.date_range(
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:13:00Z",
        freq="1min",
        tz="UTC",
    )
    sparse = pd.DataFrame(
        {"open": [100.0], "high": [100.0], "low": [100.0], "close": [100.0]},
        index=pd.to_datetime(["2026-01-01T00:07:00Z"], utc=True),
    )
    spo._POLICY_1M_KLINES_CACHE[(str(tmp_path), "BTC/USD:USD")] = (
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T00:14:00Z"),
        sparse,
    )

    out = _load_policy_1m_klines_cached(
        "BTC/USD:USD",
        DummyStore(),
        needed_ts=needed,
        market_mode="perps",
    )

    assert fetch_calls == [
        (
            int(pd.Timestamp("2026-01-01T00:00:00Z").value // 10**6),
            int(pd.Timestamp("2026-01-01T01:00:00Z").value // 10**6),
        )
    ]
    assert pd.Timestamp("2026-01-01T00:10:00Z") in out.index
    assert out.loc[pd.Timestamp("2026-01-01T00:10:00Z"), "open"] == pytest.approx(110.0)


def test_1m_execution_loader_is_read_only_by_default(monkeypatch, tmp_path):
    import extreme_price_movements.simple_policy_optimiser as spo

    class DummyStore:
        root_dir = str(tmp_path)

        def load(self, symbol, columns=None, start_ts=None, end_ts=None):
            return pd.DataFrame()

    monkeypatch.delenv("EPM_SIMPLE_POLICY_1M_DOWNLOAD", raising=False)
    monkeypatch.setattr(
        spo,
        "_policy_execution_exchange",
        lambda market_mode: (_ for _ in ()).throw(
            AssertionError("default replay must not create an exchange")
        ),
    )
    spo._POLICY_1M_KLINES_CACHE.clear()

    out = _load_policy_1m_klines_cached(
        "BTC/USD:USD",
        DummyStore(),
        needed_ts=pd.date_range(
            "2026-01-01T00:00:00Z", periods=2, freq="1min", tz="UTC"
        ),
        market_mode="perps",
    )

    assert out.empty


def test_strict_delayed_entry_guard_rejects_theoretical_fallback(monkeypatch):
    import extreme_price_movements.simple_policy_optimiser as spo

    monkeypatch.setattr(spo, "POLICY_REQUIRE_1M_EXECUTION", True)
    monkeypatch.setattr(spo, "POLICY_MIN_1M_EXECUTION_COVERAGE", 0.95)
    rows = pd.DataFrame(
        {
            "entry_execution_source": [
                "delayed_1m_intraminute_proxy",
                "theoretical_15m_open",
                "theoretical_15m_open",
            ]
        }
    )

    with pytest.raises(RuntimeError, match="insufficient delayed 1m execution coverage"):
        _validate_delayed_entry_execution_coverage(rows, artifact_label="candidates")


def test_strict_export_failure_diagnostics_persist_fallback_rows(tmp_path):
    rows = pd.DataFrame(
        {
            "strategy_id": ["long_a", "long_a", "short_b"],
            "symbol": ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD"],
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-02T00:00:00Z",
                ]
            ),
            "entry_execution_source": [
                "delayed_1m_intraminute_proxy",
                "theoretical_15m_open",
                "theoretical_15m_open",
            ],
            "delayed_entry_ts": pd.to_datetime(
                [
                    "2026-01-01T00:10:00Z",
                    "2026-01-01T01:10:00Z",
                    "2026-01-02T00:10:00Z",
                ]
            ),
            "delay_window_candle_count": [11.0, np.nan, 4.0],
            "net_return": [0.01, -0.02, 0.03],
        }
    )

    _write_policy_export_failure_diagnostics(
        rows,
        tmp_path,
        run_id="test-run",
        market_mode="perps",
    )

    summary = json.loads((tmp_path / "policy_export_invalid_summary.json").read_text())
    fallback = pd.read_parquet(tmp_path / "policy_export_invalid_fallback_rows.parquet")
    assert summary["delayed_1m_execution_rows"] == 1
    assert summary["fallback_rows"] == 2
    assert summary["fallback_by_strategy"] == {"long_a": 1, "short_b": 1}
    assert set(fallback["symbol"]) == {"ETH/USD:USD", "SOL/USD:USD"}
    assert "delay_window_candle_count" in fallback.columns


def test_rank_threshold_band_report_separates_local_from_cumulative(tmp_path):
    rows = pd.DataFrame(
        {
            "strategy_id": ["long_a", "long_a", "short_b", "short_b"],
            "auction_rank_score": [0.82, 0.87, 0.92, 0.97],
            "normalized_rank_score": [0.82, 0.87, 0.92, 0.97],
            "strategy_rank_pct": [0.75, 0.90, 0.88, 0.99],
            "net_return": [-0.01, 0.02, 0.03, 0.04],
            "gross_return": [-0.005, 0.025, 0.035, 0.045],
        }
    )

    _write_rank_threshold_band_reports(rows, output_dir=tmp_path, band_width=0.05)

    report = pd.read_csv(tmp_path / "rank_threshold_band_report.csv")
    payload = json.loads((tmp_path / "rank_threshold_band_report.json").read_text())
    assert payload
    global_rows = report[
        (report["group"] == "global")
        & (report["rank_col"] == "auction_rank_score")
    ]
    local_80 = global_rows[
        (global_rows["selection_type"] == "local_band")
        & np.isclose(global_rows["band_lo"], 0.80)
    ].iloc[0]
    cumulative_80 = global_rows[
        (global_rows["selection_type"] == "cumulative_at_or_above")
        & np.isclose(global_rows["band_lo"], 0.80)
    ].iloc[0]

    assert int(local_80["row_count"]) == 1
    assert local_80["mean_net_return"] < 0.0
    assert int(cumulative_80["row_count"]) == 4
    assert cumulative_80["mean_net_return"] > 0.0


def test_local_candidate_guard_requires_lower_band_gross_hit_and_ev(tmp_path, monkeypatch):
    import extreme_price_movements.simple_policy_optimiser as spo

    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_ROWS", 2)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_NET_HIT_RATE", 0.50)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_GROSS_HIT_RATE", 0.60)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_MEAN_NET_RETURN", 0.002)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_BAND_WIDTH", 0.05)

    deployment_payload = {
        "strategies": [
            {
                "strategy_id": "long_a",
                "strategy_for_inference": "long_a",
                "selected": True,
                "deployment_rank_threshold": 0.80,
            }
        ],
        "rejected_strategies": [],
    }
    candidates = pd.DataFrame(
        {
            "strategy_id": ["long_a"] * 6,
            "auction_rank_score": [0.81, 0.82, 0.86, 0.87, 0.91, 0.92],
            "net_return": [-0.001, 0.001, 0.003, 0.004, 0.002, 0.003],
            "gross_return": [-0.0005, 0.0015, 0.004, 0.005, 0.003, 0.004],
        }
    )

    summary = _apply_local_candidate_hit_rate_guard(
        deployment_payload,
        candidates,
        candidate_path=tmp_path / "simple_policy_candidates.parquet",
    )

    guard = summary["strategies"]["long_a"]
    assert guard["passed"] is True
    assert np.isclose(guard["selected_threshold"], 0.82)
    assert np.isclose(guard["selected_local_band_lo"], 0.82)
    assert np.isclose(guard["selected_local_band_hi"], 0.87)
    assert guard["selected_gross_hit_rate"] == 1.0
    assert guard["selected_mean_net_return"] >= 0.002
    assert deployment_payload["strategies"][0]["deployment_rank_threshold"] == 0.82
    assert deployment_payload["rejected_strategies"] == []


def test_local_candidate_guard_uses_lowest_ev_positive_band(
    tmp_path, monkeypatch
):
    import extreme_price_movements.simple_policy_optimiser as spo

    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_ROWS", 4)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_NET_HIT_RATE", 0.50)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_GROSS_HIT_RATE", 0.55)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_MEAN_NET_RETURN", 0.002)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_BAND_WIDTH", 0.05)
    monkeypatch.setattr(spo, "DEPLOYMENT_THRESHOLD_PRECISION", 0.01)

    deployment_payload = {
        "strategies": [
            {
                "strategy_id": "long_dist",
                "strategy_for_inference": "long_dist",
                "selected": True,
                "deployment_rank_threshold": 0.90,
            }
        ],
        "rejected_strategies": [],
    }
    candidates = pd.DataFrame(
        {
            "strategy_id": ["long_dist"] * 8,
            "auction_rank_score": [
                0.901,
                0.912,
                0.921,
                0.934,
                0.951,
                0.962,
                0.973,
                0.984,
            ],
            # The 0.90-0.95 band has positive EV but only 50% gross hit.
            # The 0.95-1.00 band has 75% gross hit and clears all floors.
            "net_return": [
                -0.010,
                0.030,
                -0.006,
                0.025,
                -0.004,
                0.018,
                0.022,
                0.026,
            ],
            "gross_return": [
                -0.008,
                0.032,
                -0.004,
                0.027,
                -0.002,
                0.020,
                0.024,
                0.028,
            ],
        }
    )

    summary = _apply_local_candidate_hit_rate_guard(
        deployment_payload,
        candidates,
        candidate_path=tmp_path / "simple_policy_candidates.parquet",
    )

    guard = summary["strategies"]["long_dist"]
    assert guard["passed"] is True
    assert np.isclose(guard["selected_threshold"], 0.89)
    assert np.isclose(guard["selected_local_band_lo"], 0.89)
    assert np.isclose(guard["selected_local_band_hi"], 0.94)
    assert guard["selected_gross_hit_rate"] < 0.55
    assert guard["hit_rate_floors_are_diagnostic_only"] is True
    assert guard["next_band_positive_count"] == 1
    assert guard["selected_mean_net_return"] >= 0.002
    assert deployment_payload["strategies"][0]["deployment_rank_threshold"] == 0.89
    assert deployment_payload["rejected_strategies"] == []


def test_local_candidate_guard_rejects_strategy_when_no_lower_band_meets_floor(
    tmp_path, monkeypatch
):
    import extreme_price_movements.simple_policy_optimiser as spo

    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_ROWS", 2)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_NET_HIT_RATE", 0.50)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_GROSS_HIT_RATE", 0.60)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_MEAN_NET_RETURN", 0.002)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_BAND_WIDTH", 0.05)

    deployment_payload = {
        "strategies": [
            {
                "strategy_id": "long_a",
                "strategy_for_inference": "long_a",
                "selected": True,
                "deployment_rank_threshold": 0.80,
            }
        ],
        "rejected_strategies": [],
    }
    candidates = pd.DataFrame(
        {
            "strategy_id": ["long_a"] * 4,
            "auction_rank_score": [0.81, 0.82, 0.86, 0.87],
            "net_return": [-0.001, 0.001, 0.001, 0.001],
            "gross_return": [-0.0005, 0.0015, 0.0015, 0.0015],
        }
    )

    summary = _apply_local_candidate_hit_rate_guard(
        deployment_payload,
        candidates,
        candidate_path=tmp_path / "simple_policy_candidates.parquet",
    )

    guard = summary["strategies"]["long_a"]
    assert guard["passed"] is False
    assert guard["threshold_applied"] is False
    assert guard["applied_threshold"] == 0.80
    assert guard["deployment_threshold_after_guard"] == 0.80
    assert guard["fallback_candidate_threshold"] != 0.80
    assert deployment_payload["strategies"] == []
    assert deployment_payload["rejected_strategies"][0]["deployment_rank_threshold"] == 0.80
    assert deployment_payload["rejected_strategies"][0]["reject_reasons"] == [
        "local_lower_band_hit_or_ev_floor_not_met"
    ]


def test_local_candidate_guard_requires_recent_window_positive_ev(
    tmp_path, monkeypatch
):
    import extreme_price_movements.simple_policy_optimiser as spo

    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_ROWS", 2)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_MEAN_NET_RETURN", 0.002)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_BAND_WIDTH", 0.05)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_CONFIRMATION_BANDS", 1)
    monkeypatch.setattr(
        spo,
        "DEPLOYMENT_LOCAL_CANDIDATE_CONFIRMATION_MIN_POSITIVE",
        1,
    )
    monkeypatch.setattr(
        spo,
        "DEPLOYMENT_REPLAY_THRESHOLD_HARD_FLOOR_WINDOWS_DAYS_RAW",
        "7,14,28",
    )

    deployment_payload = {
        "strategies": [
            {
                "strategy_id": "long_recent",
                "strategy_for_inference": "long_recent",
                "selected": True,
                "deployment_rank_threshold": 0.70,
            }
        ],
        "rejected_strategies": [],
    }
    candidates = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-05-11T00:00:00Z",
                    "2026-05-12T00:00:00Z",
                    "2026-05-13T00:00:00Z",
                    "2026-05-14T00:00:00Z",
                    "2026-06-06T00:00:00Z",
                ],
                utc=True,
            ),
            "strategy_id": ["long_recent"] * 5,
            "auction_rank_score": [0.701, 0.712, 0.755, 0.762, 0.905],
            "net_return": [0.010, 0.012, 0.008, 0.009, -0.020],
            "gross_return": [0.011, 0.013, 0.009, 0.010, -0.019],
        }
    )

    summary = _apply_local_candidate_hit_rate_guard(
        deployment_payload,
        candidates,
        candidate_path=tmp_path / "simple_policy_candidates.parquet",
    )

    guard = summary["strategies"]["long_recent"]
    assert guard["passed"] is False
    assert guard["recent_window_hard_floor_pass"] is False
    assert guard["recent_window_hard_floor_available"] is True
    assert guard["recent_window_metrics"][0]["window_days"] == 7
    assert guard["recent_window_metrics"][0]["net_pnl"] < 0
    assert deployment_payload["strategies"] == []
    assert deployment_payload["rejected_strategies"][0]["strategy_id"] == "long_recent"


def test_local_candidate_guard_rescues_performance_rejected_strategy_at_higher_threshold(
    tmp_path, monkeypatch
):
    import extreme_price_movements.simple_policy_optimiser as spo

    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_ROWS", 2)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_MEAN_NET_RETURN", 0.002)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_BAND_WIDTH", 0.05)
    monkeypatch.setattr(spo, "DEPLOYMENT_THRESHOLD_PRECISION", 0.01)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_CONFIRMATION_BANDS", 1)
    monkeypatch.setattr(
        spo,
        "DEPLOYMENT_LOCAL_CANDIDATE_CONFIRMATION_MIN_POSITIVE",
        1,
    )
    monkeypatch.setattr(
        spo,
        "DEPLOYMENT_REPLAY_THRESHOLD_HARD_FLOOR_WINDOWS_DAYS_RAW",
        "",
    )

    deployment_payload = {
        "strategies": [],
        "rejected_strategies": [
            {
                "strategy_id": "short_rejected",
                "strategy_for_inference": "short_rejected",
                "side": "short",
                "selected": False,
                "deployment_rank_threshold": 0.70,
                "reject_reasons": ["top_5_net_pnl_not_positive"],
            }
        ],
    }
    candidates = pd.DataFrame(
        {
            "strategy_id": ["short_rejected"] * 4,
            "strategy_rank_pct": [0.901, 0.912, 0.951, 0.962],
            "net_return": [0.010, 0.012, 0.008, 0.009],
            "gross_return": [0.011, 0.013, 0.009, 0.010],
        }
    )

    summary = _apply_local_candidate_hit_rate_guard(
        deployment_payload,
        candidates,
        candidate_path=tmp_path / "simple_policy_candidates.parquet",
    )

    assert len(deployment_payload["strategies"]) == 1
    rescued = deployment_payload["strategies"][0]
    assert rescued["strategy_id"] == "short_rejected"
    assert rescued["selected"] is True
    assert "reject_reasons" not in rescued
    assert rescued["deployment_rank_threshold"] == pytest.approx(0.87)
    assert rescued["threshold_rank_score_source"] == "policy_rank_pct"
    assert rescued["deployment_selection_rescue"]["previous_reject_reasons"] == [
        "top_5_net_pnl_not_positive"
    ]
    assert deployment_payload["rejected_strategies"] == []
    assert summary["rescued_strategies"]["short_rejected"]["applied_threshold"] == pytest.approx(
        0.87
    )


def test_local_candidate_guard_can_rescue_rejected_strategy_above_099(
    tmp_path, monkeypatch
):
    import extreme_price_movements.simple_policy_optimiser as spo

    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_ROWS", 1)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_MIN_MEAN_NET_RETURN", 0.002)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_BAND_WIDTH", 0.002)
    monkeypatch.setattr(spo, "DEPLOYMENT_THRESHOLD_PRECISION", 0.002)
    monkeypatch.setattr(spo, "DEPLOYMENT_LOCAL_CANDIDATE_CONFIRMATION_BANDS", 1)
    monkeypatch.setattr(
        spo,
        "DEPLOYMENT_LOCAL_CANDIDATE_CONFIRMATION_MIN_POSITIVE",
        1,
    )
    monkeypatch.setattr(
        spo,
        "DEPLOYMENT_REPLAY_THRESHOLD_HARD_FLOOR_WINDOWS_DAYS_RAW",
        "7,14,28",
    )

    deployment_payload = {
        "strategies": [],
        "rejected_strategies": [
            {
                "strategy_id": "long_tail",
                "strategy_for_inference": "long_tail",
                "side": "long",
                "selected": False,
                "deployment_rank_threshold": 0.70,
                "reject_reasons": ["top_5_net_pnl_not_positive"],
            }
        ],
    }
    candidates = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-06-09T00:00:00Z",
                    "2026-06-09T01:00:00Z",
                    "2026-06-09T02:00:00Z",
                    "2026-06-09T03:00:00Z",
                ],
                utc=True,
            ),
            "strategy_id": ["long_tail"] * 4,
            "strategy_rank_pct": [0.9905, 0.9925, 0.9930, 0.9945],
            "net_return": [-0.020, 0.030, 0.025, 0.010],
            "gross_return": [-0.018, 0.032, 0.027, 0.012],
        }
    )

    summary = _apply_local_candidate_hit_rate_guard(
        deployment_payload,
        candidates,
        candidate_path=tmp_path / "simple_policy_candidates_broad.parquet",
    )

    assert len(deployment_payload["strategies"]) == 1
    rescued = deployment_payload["strategies"][0]
    assert rescued["strategy_id"] == "long_tail"
    assert rescued["deployment_rank_threshold"] == pytest.approx(0.992)
    guard = rescued["local_candidate_hit_rate_guard"]
    assert guard["passed"] is True
    assert guard["recent_window_hard_floor_pass"] is True
    assert guard["threshold_rows"][0]["deployment_rank_threshold"] == pytest.approx(
        0.99
    )
    assert guard["threshold_rows"][0]["mean_net_return"] < 0.0
    assert summary["rescued_strategies"]["long_tail"]["applied_threshold"] == pytest.approx(
        0.992
    )


def test_policy_feature_loader_falls_back_to_source_run(tmp_path, monkeypatch):
    data_root = tmp_path / "data_perp"
    source_run_id = "source_run"
    current_run_id = "current_run"
    feature_dir = data_root / "features" / source_run_id
    feature_dir.mkdir(parents=True)
    ts = pd.Timestamp("2026-01-01T00:00:00Z")
    pd.DataFrame(
        {"feature_a": [1.25], "feature_b": [2.5]},
        index=pd.DatetimeIndex([ts], name="timestamp"),
    ).to_parquet(feature_dir / "symbol=BTC_USD_USD.parquet")
    events = pd.DataFrame(
        {
            "timestamp": [ts],
            "symbol": ["BTC/USD:USD"],
        }
    )
    monkeypatch.setenv("EPM_ARTIFACT_SOURCE_RUN_ID", source_run_id)

    out = _load_feature_rows_for_events(
        events,
        data_root=str(data_root),
        run_id=current_run_id,
    )

    assert out.shape == (1, 2)
    assert out.loc[0, "feature_a"] == 1.25


def test_simulate_and_score_selected_mask_stays_in_input_coordinates():
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T02:00:00Z",
                    "2026-01-01T03:00:00Z",
                ]
            ),
            "symbol": ["BTC/USD:USD", "ETH/USD:USD", "SOL/USD:USD", "XRP/USD:USD"],
            "rank_pct": [0.9, 0.8, 0.7, 0.6],
            "side": [1.0, 1.0, 1.0, 1.0],
            "barrier_pct": [0.02, 0.02, 0.02, 0.02],
        }
    )
    f_opens = np.array(
        [
            [100.0, 101.0, 102.0],
            [np.nan, np.nan, np.nan],
            [200.0, 201.0, 202.0],
            [300.0, 301.0, 302.0],
        ],
        dtype=np.float32,
    )
    f_highs = f_opens + 1.0
    f_lows = f_opens - 1.0
    f_closes = f_opens.copy()
    f_closes[3, :] = np.nan

    metrics = simulate_and_score(
        rows,
        f_opens,
        f_highs,
        f_lows,
        f_closes,
        max_concurrent_trades=1_000_000,
        max_concurrent_per_asset=1_000_000,
    )

    selected_mask = np.asarray(metrics["selected_mask"], dtype=bool)
    assert selected_mask.tolist() == [True, False, True, False]
    assert metrics["input_candidate_count"] == 4
    assert metrics["valid_entry_count"] == 2
    assert len(metrics["raw_gains"]) == int(selected_mask.sum())
    assert np.isfinite(metrics["raw_gains"]).all()


def test_simulate_and_score_timeout_marks_to_last_executable_close():
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"]),
            "symbol": ["BTC/USD:USD"],
            "rank_pct": [0.9],
            "side": [1.0],
            "barrier_pct": [0.02],
            "expected_spread_bps": [0.0],
        }
    )
    opens = np.array([[100.0, 101.0, 104.0]], dtype=np.float32)
    highs = np.array([[100.1, 101.1, 105.1]], dtype=np.float32)
    lows = np.array([[99.9, 100.9, 103.9]], dtype=np.float32)
    closes = np.array([[100.0, np.nan, 105.0]], dtype=np.float32)

    metrics = simulate_and_score(
        rows,
        opens,
        highs,
        lows,
        closes,
        cost_pct=0.0,
        sl_mult=10.0,
        trailing_activation_mult=10.0,
        max_concurrent_trades=10,
        max_concurrent_per_asset=10,
    )

    assert list(metrics["exit_reason"]) == ["timeout"]
    assert metrics["gross_gains"][0] > 0.0
    assert metrics["raw_gains"][0] > 0.0
    assert metrics["exit_bars"][0] == 2


def test_capital_protection_floor_uses_asset_spread_multiple():
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"]),
            "symbol": ["BTC/USD:USD"],
            "rank_pct": [0.9],
            "side": [1.0],
            "barrier_pct": [0.02],
            "expected_spread_bps": [100.0],
        }
    )
    opens = np.array([[100.0, 100.5, 102.0, 102.0]], dtype=np.float32)
    highs = np.array([[100.1, 103.0, 102.5, 102.2]], dtype=np.float32)
    lows = np.array([[99.9, 100.4, 100.0, 100.0]], dtype=np.float32)
    closes = np.array([[100.0, 102.5, 102.0, 102.0]], dtype=np.float32)

    common = dict(
        cost_pct=0.0,
        sl_mult=3.0,
        trailing_activation_mult=10.0,
        capital_protect_mfe_mult=0.5,
        capital_protect_lock_frac=0.0,
        capital_protect_min_lock_bps=0.0,
        max_concurrent_trades=10,
        max_concurrent_per_asset=10,
    )
    without_spread_floor = simulate_and_score(
        rows,
        opens,
        highs,
        lows,
        closes,
        capital_protect_spread_lock_mult=0.0,
        **common,
    )
    metrics = simulate_and_score(
        rows,
        opens,
        highs,
        lows,
        closes,
        capital_protect_spread_lock_mult=1.5,
        **common,
    )

    assert list(metrics["exit_reason"]) == ["capital_protect"]
    assert list(without_spread_floor["exit_reason"]) == ["capital_protect"]
    assert metrics["gross_gains"][0] > without_spread_floor["gross_gains"][0]


def test_capital_protection_waits_until_spread_lock_is_earned():
    rows = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01T00:00:00Z"]),
            "symbol": ["BTC/USD:USD"],
            "rank_pct": [0.9],
            "side": [1.0],
            "barrier_pct": [0.02],
            "expected_spread_bps": [100.0],
        }
    )
    opens = np.array([[100.0, 100.5, 100.8, 100.8]], dtype=np.float32)
    highs = np.array([[100.1, 101.0, 100.9, 100.9]], dtype=np.float32)
    lows = np.array([[99.9, 100.4, 100.6, 100.6]], dtype=np.float32)
    closes = np.array([[100.0, 100.8, 100.8, 100.8]], dtype=np.float32)

    metrics = simulate_and_score(
        rows,
        opens,
        highs,
        lows,
        closes,
        cost_pct=0.0,
        sl_mult=3.0,
        trailing_activation_mult=10.0,
        capital_protect_mfe_mult=0.1,
        capital_protect_lock_frac=0.0,
        capital_protect_min_lock_bps=0.0,
        capital_protect_spread_lock_mult=1.5,
        max_concurrent_trades=10,
        max_concurrent_per_asset=10,
    )

    assert list(metrics["exit_reason"]) == ["timeout"]
def test_fetch_policy_paths_does_not_pad_unobserved_future_bars():
    class PartialStore:
        timeframe = "15m"

        def load(self, symbol, columns=None, start_ts=None, end_ts=None):
            index = pd.date_range("2026-07-13 00:00", periods=2, freq="15min", tz="UTC")
            frame = pd.DataFrame(
                {
                    "open": [100.0, 101.0],
                    "high": [101.0, 102.0],
                    "low": [99.0, 100.0],
                    "close": [100.5, 101.5],
                    "ts": index,
                },
                index=index,
            )
            return frame

    rows = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-07-12 23:00", tz="UTC")],
            "symbol": ["BTC/USD:USD"],
        }
    )
    paths = _fetch_policy_paths(rows, PartialStore(), path_len=4)

    np.testing.assert_allclose(paths[0][0, :2], [100.0, 101.0])
    assert np.isnan(paths[0][0, 2:]).all()
    assert not _policy_path_finite_mask(paths)[0]
