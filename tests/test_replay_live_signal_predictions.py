import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.replay_live_signal_predictions as replay_module

from extreme_price_movements.data_store import PartitionedOHLCVStore
from extreme_price_movements.inference.canonical_meta_postprocessor import (
    _normalize_model_archetype_keys,
)
from scripts.historical_inference_parity import (
    _add_required_context_symbols,
    _build_runtime_cfg,
)
from scripts.replay_live_signal_predictions import (
    _alpha_contract_columns_for_replay,
    _batched_replay_feature_rows,
    _hydrate_replay_residual_context_features,
    _live_synthesized_feature_delta_summary,
    _live_feature_cache_symbols_for_end,
    _source_parity_context_symbols_for_end,
    _load_persisted_live_candidate_feature_matrix,
    _load_panel,
    _load_recent_decisions,
    _model_runtime_cfg,
    _parity_failures,
    _predict_exact_logged_meta_input,
    _slice_panel,
    _summary,
)


def test_replay_overlay_contract_includes_generated_alpha_features():
    state = {
        "bundle": {
            "alpha_models": {
                "long_model": {
                    "feat_cols": ["ret1h", "dae_b16_11", "gmm_mahal_0"]
                },
                "short_model": {
                    "feat_cols": ["ret1h", "AE_reconstruction_error"]
                },
            }
        }
    }

    assert _alpha_contract_columns_for_replay(state) == {
        "ret1h",
        "dae_b16_11",
        "gmm_mahal_0",
        "AE_reconstruction_error",
    }


def test_feature_contract_summary_compares_categorical_values_explicitly():
    frame = pd.DataFrame({"numeric": [1.0]}, index=["BTC/USD:USD"])
    logged = json.dumps(
        {
            "numeric": 1.0,
            "side_name": "long",
            "archetype_policy_key": "long__compression_release",
        }
    )

    summary = replay_module._feature_value_contract_summary(
        logged_values_raw=logged,
        feature_row=frame,
        symbol="BTC/USD:USD",
        categorical_values={
            "side_name": "long",
            "archetype_policy_key": "long__compression_release",
        },
    )
    assert summary["missing_count"] == 0
    assert summary["common_finite_count"] == 1

    mismatch = replay_module._feature_value_contract_summary(
        logged_values_raw=logged,
        feature_row=frame,
        symbol="BTC/USD:USD",
        categorical_values={
            "side_name": "long",
            "archetype_policy_key": "long__different",
        },
    )
    assert mismatch["missing_features"] == ["archetype_policy_key"]


def test_batched_replay_repairs_nonfinite_synthetic_overlay(monkeypatch):
    monkeypatch.setattr(
        replay_module,
        "get_features_for_candidates",
        lambda feats, symbols, ts: pd.DataFrame(
            {"__regime_source_demo_score__": [np.nan]},
            index=["BTC/USD:USD"],
        ),
    )

    def _repair_source(frame, **kwargs):
        out = frame.copy()
        assert "__regime_source_demo_score__" in kwargs["required_columns"]
        out["__regime_source_demo_score__"] = np.float32(0.75)
        return out

    monkeypatch.setattr(
        replay_module, "materialize_live_source_regime_features", _repair_source
    )
    monkeypatch.setattr(
        replay_module,
        "materialize_live_ae_gmm_features",
        lambda frame, **kwargs: frame,
    )
    group = pd.DataFrame(
        {"symbol": ["BTC/USD:USD"], "side": ["long"]}, index=[7]
    )

    rows = _batched_replay_feature_rows(
        feats={},
        group=group,
        signal_bar_ts=pd.Timestamp("2026-07-15T12:00:00Z"),
        overlay_required_columns={"__regime_source_demo_score__"},
        live_ae_gmm_state_payload={},
    )

    assert rows[7].loc["BTC/USD:USD", "__regime_source_demo_score__"] == pytest.approx(
        0.75
    )


def test_residual_hydration_preserves_finite_persisted_ae_gmm(monkeypatch):
    """Post-meta hydration must not redefine frozen full-universe AE/GMM state."""
    symbol = "BTC/USD:USD"
    persisted = pd.DataFrame(
        {
            "expected_mahalanobis": [4.25],
            "gmm_cluster_posterior_0": [0.72],
            "cluster_speed": [0.08],
            "missing_context": [np.nan],
        },
        index=[symbol],
    )
    supplemental = pd.DataFrame(
        {
            "expected_mahalanobis": [8.50],
            "gmm_cluster_posterior_0": [0.11],
            "cluster_speed": [0.91],
            "missing_context": [0.33],
        },
        index=[symbol],
    )
    monkeypatch.setattr(
        replay_module,
        "residual_event_state_input_feature_columns",
        lambda payload: set(),
    )
    monkeypatch.setattr(
        replay_module,
        "_live_regime_calibration_raw_feature_columns",
        lambda *args, **kwargs: set(),
    )
    monkeypatch.setattr(
        replay_module,
        "_build_residual_event_feature_runtime_cfg",
        lambda runtime_cfg, **kwargs: dict(runtime_cfg),
    )
    recompute_call = {}

    def _load_recomputed(**kwargs):
        recompute_call.update(kwargs)
        return {"unused": pd.DataFrame()}

    monkeypatch.setattr(
        replay_module,
        "load_or_compute_features",
        _load_recomputed,
    )
    monkeypatch.setattr(
        replay_module,
        "get_features_for_candidates",
        lambda feats, symbols, ts: supplemental.copy(),
    )
    monkeypatch.setattr(
        replay_module,
        "_hydrate_optional_frozen_features",
        lambda frame, **kwargs: (frame, [], []),
    )
    postprocessor = SimpleNamespace(
        regime_ev_artifact={},
        required_input_features=lambda: {
            "expected_mahalanobis",
            "gmm_cluster_posterior_0",
            "cluster_speed",
            "missing_context",
        },
    )

    hydrated = _hydrate_replay_residual_context_features(
        feature_rows={7: persisted},
        group=pd.DataFrame(
            {"symbol": [symbol], "side": ["long"]}, index=[7]
        ),
        panel_slice={"close": pd.DataFrame([[1.0]], columns=[symbol])},
        signal_bar_ts=pd.Timestamp("2026-07-15T12:00:00Z"),
        feature_cfg={},
        runtime_cfg={},
        run_id="run",
        data_root="data_perp",
        residual_event_payload={},
        canonical_postprocessor=postprocessor,
    )

    row = hydrated[7].loc[symbol]
    assert row["expected_mahalanobis"] == pytest.approx(4.25)
    assert row["gmm_cluster_posterior_0"] == pytest.approx(0.72)
    assert row["cluster_speed"] == pytest.approx(0.08)
    assert row["missing_context"] == pytest.approx(0.33)
    runtime_cfg = recompute_call["cfg"]["runtime_cfg"]
    assert runtime_cfg["live_model_feature_tail_recompute_enabled"] is False
    assert runtime_cfg["live_feature_prefer_offline_cache"] is True
    assert runtime_cfg["live_feature_offline_cache_enabled"] is True
    assert runtime_cfg["live_feature_offline_cache_authoritative"] is True


def test_residual_hydration_collapses_duplicate_cached_rows_per_decision(monkeypatch):
    symbol = "BTC/USD:USD"
    duplicate_cached = pd.DataFrame(
        {
            "persisted_context": [0.4, 0.4, 0.4],
            "missing_context": [np.nan, np.nan, np.nan],
        },
        index=[symbol, symbol, symbol],
    )
    supplemental = pd.DataFrame(
        {"persisted_context": [0.9], "missing_context": [0.3]},
        index=[symbol],
    )
    monkeypatch.setattr(
        replay_module,
        "residual_event_state_input_feature_columns",
        lambda payload: set(),
    )
    monkeypatch.setattr(
        replay_module,
        "_live_regime_calibration_raw_feature_columns",
        lambda *args, **kwargs: set(),
    )
    monkeypatch.setattr(
        replay_module,
        "_build_residual_event_feature_runtime_cfg",
        lambda runtime_cfg, **kwargs: dict(runtime_cfg),
    )
    monkeypatch.setattr(
        replay_module,
        "load_or_compute_features",
        lambda **kwargs: {"unused": pd.DataFrame()},
    )
    monkeypatch.setattr(
        replay_module,
        "get_features_for_candidates",
        lambda feats, symbols, ts: supplemental.copy(),
    )
    monkeypatch.setattr(
        replay_module,
        "_hydrate_optional_frozen_features",
        lambda frame, **kwargs: (frame, [], []),
    )
    postprocessor = SimpleNamespace(
        regime_ev_artifact={},
        required_input_features=lambda: {
            "persisted_context",
            "missing_context",
        },
    )

    hydrated = _hydrate_replay_residual_context_features(
        feature_rows={7: duplicate_cached},
        group=pd.DataFrame({"symbol": [symbol], "side": ["long"]}, index=[7]),
        panel_slice={"close": pd.DataFrame([[1.0]], columns=[symbol])},
        signal_bar_ts=pd.Timestamp("2026-07-15T12:00:00Z"),
        feature_cfg={},
        runtime_cfg={},
        run_id="run",
        data_root="data_perp",
        residual_event_payload={},
        canonical_postprocessor=postprocessor,
    )

    assert len(hydrated[7]) == 1
    assert hydrated[7].iloc[0]["persisted_context"] == pytest.approx(0.4)
    assert hydrated[7].iloc[0]["missing_context"] == pytest.approx(0.3)


class _AlignedMetaModel:
    selected_features = ["feature_a"]
    s52_meta_score_alignment_ = {
        "enabled": True,
        "sides": {
            "long": {
                "source_knots": [0.0, 1.0],
                "target_knots": [0.0, 0.5],
            }
        },
    }

    def predict(self, frame):
        return np.asarray(frame["feature_a"], dtype=float)


def test_canonical_postprocessor_uses_frozen_unprefixed_archetype_keys():
    frame = pd.DataFrame(
        {
            "side_name": ["long", "short", "long"],
            "archetype_policy_key": [
                "long__long_breakout_diagnostic_candidate",
                "short__short_mixed_clean_path",
                "long_mixed_wideslow_tentative",
            ],
        }
    )

    normalized = _normalize_model_archetype_keys(frame)

    assert normalized["archetype_policy_key"].tolist() == [
        "long_breakout_diagnostic_candidate",
        "short_mixed_clean_path",
        "long_mixed_wideslow_tentative",
    ]
    assert frame.iloc[0]["archetype_policy_key"].startswith("long__")


def test_logged_canonical_input_uses_base_score_alias_and_persisted_expert_rank():
    captured = {}

    class _Postprocessor:
        def transform(self, frame, *, copy=False):
            captured.update(frame.iloc[0].to_dict())
            out = frame.copy()
            out["historical_rank"] = out["score_meta_base_soft_label"]
            out["score_regime_calibrated"] = out["score_meta_base_soft_label"]
            out["expected_net_ev_after_1pct"] = 0.01
            out["expected_ev_rank_score"] = 0.9
            out["meta_postprocessor_policy_id"] = "test"
            return out

    # This narrow regression is exercised through the row scorer elsewhere;
    # assert the production contract explicitly here so future replay adapters
    # cannot substitute the obsolete direct-meta diagnostic for expert rank.
    canonical_input = pd.DataFrame(index=["BTC/USD:USD"])
    logged_base = 0.41
    persisted_expert = 0.83
    persisted_raw = 0.82
    canonical_input["score_base"] = logged_base
    canonical_input["score"] = logged_base
    canonical_input["score_meta_base_soft_label"] = persisted_expert
    canonical_input["score_meta_base_soft_label_raw_refit"] = persisted_raw
    _Postprocessor().transform(canonical_input, copy=False)

    assert captured["score"] == pytest.approx(logged_base)
    assert captured["score_meta_base_soft_label"] == pytest.approx(
        persisted_expert
    )
    assert captured["score_meta_base_soft_label_raw_refit"] == pytest.approx(
        persisted_raw
    )


def test_loads_persisted_live_candidate_feature_matrix(tmp_path):
    root = (
        tmp_path
        / "artifacts"
        / "run"
        / "live_candidate_feature_matrix"
        / "long"
        / "batch"
    )
    root.mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps({"signal_bar_ts": "2026-07-14T11:00:00Z"})
    )
    pd.DataFrame({"feature_a": [1.25]}, index=["BTC/USD:USD"]).to_parquet(
        root / "data.parquet"
    )

    loaded = _load_persisted_live_candidate_feature_matrix(
        artifact_data_root=tmp_path,
        run_id="run",
        signal_bar_ts=pd.Timestamp("2026-07-14T11:00:00Z"),
        side="long",
    )

    assert loaded.loc["BTC/USD:USD", "feature_a"] == pytest.approx(1.25)


def test_logged_meta_replay_applies_frozen_score_alignment():
    orchestrator = SimpleNamespace(meta_models={"long_demo": _AlignedMetaModel()})

    pred = _predict_exact_logged_meta_input(
        orchestrator=orchestrator,
        side="long",
        strategy_id="long_demo",
        logged_meta_frame=pd.DataFrame({"feature_a": [0.8]}),
    )

    assert pred == pytest.approx(0.4)

    raw_pred = _predict_exact_logged_meta_input(
        orchestrator=orchestrator,
        side="long",
        strategy_id="long_demo",
        logged_meta_frame=pd.DataFrame({"feature_a": [0.8]}),
        apply_score_alignment=False,
    )
    assert raw_pred == pytest.approx(0.8)


def test_load_recent_decisions_filters_rank_source_and_start(tmp_path):
    ledger_path = tmp_path / "prediction_ledger.parquet"
    trades_path = tmp_path / "missing_trades.csv"
    ledger = pd.DataFrame(
        {
            "decision_ts": pd.to_datetime(
                [
                    "2026-05-15T10:00:00Z",
                    "2026-05-15T11:00:00Z",
                    "2026-05-15T12:00:00Z",
                ],
                utc=True,
            ),
            "signal_bar_ts": pd.to_datetime(
                [
                    "2026-05-15T09:00:00Z",
                    "2026-05-15T10:00:00Z",
                    "2026-05-15T11:00:00Z",
                ],
                utc=True,
            ),
            "symbol": ["AAAUSDC", "BBB/USDC", "CCC/USDC"],
            "side": ["long", "long", "short"],
            "strategy_id": ["long_demo", "long_demo", "short_demo"],
            "rank_score_source": [
                "historical_meta_oof_percentile",
                "policy_rank_reference_percentile",
                "policy_rank_reference_percentile",
            ],
        }
    )
    ledger.to_parquet(ledger_path, index=False)

    decisions = _load_recent_decisions(
        ledger_path=ledger_path,
        trades_path=trades_path,
        max_rows=10,
        decision_start="2026-05-15T10:30:00Z",
        require_rank_source="policy_rank_reference_percentile",
    )

    assert decisions["symbol"].tolist() == ["BBB/USDC", "CCC/USDC"]
    assert set(decisions["rank_score_source"]) == {"policy_rank_reference_percentile"}


def test_load_recent_decisions_does_not_overwrite_ledger_live_values(tmp_path):
    ledger_path = tmp_path / "prediction_ledger.parquet"
    trades_path = tmp_path / "inference_trades.csv"
    ledger = pd.DataFrame(
        {
            "decision_ts": pd.to_datetime(["2026-05-15T10:00:00Z"], utc=True),
            "signal_bar_ts": pd.to_datetime(["2026-05-15T09:00:00Z"], utc=True),
            "symbol": ["AAA/USDC"],
            "side": ["long"],
            "strategy_id": ["long_demo"],
            "base_pred": [0.2],
            "meta_pred": [0.3],
            "calibrated_score": [0.3],
            "policy_rank_pct": [0.7],
            "rank_score_source": ["policy_rank_reference_percentile"],
        }
    )
    ledger.to_parquet(ledger_path, index=False)
    pd.DataFrame(
        {
            "timestamp": ["2026-05-15T10:01:00Z"],
            "lifecycle_event": ["entry_placed"],
            "symbol": ["AAA/USDC"],
            "side": ["long"],
            "strategy_id": ["long_demo"],
            "base_pred": [0.9],
            "meta_pred": [0.8],
            "calibrated_score": [0.8],
            "policy_rank_pct": [0.1],
            "rank_score_source": ["trade_log_copy"],
        }
    ).to_csv(trades_path, index=False)

    decisions = _load_recent_decisions(
        ledger_path=ledger_path,
        trades_path=trades_path,
        max_rows=10,
    )

    row = decisions.iloc[0]
    assert row["live_base_pred"] == 0.2
    assert row["live_meta_pred"] == 0.3
    assert row["live_calibrated_score"] == 0.3
    assert row["live_policy_rank_pct"] == 0.7
    assert row["live_rank_score_source"] == "policy_rank_reference_percentile"


def test_parity_failures_require_live_values_and_tolerance():
    frame = pd.DataFrame(
        {
            "live_base_pred": [0.1],
            "live_meta_pred": [0.2],
            "live_calibrated_score": [0.2],
            "live_policy_rank_pct": [0.6],
            "replay_base_pred": [0.1],
            "replay_meta_pred": [0.22],
            "replay_calibrated_score": [0.2],
            "replay_policy_rank_pct": [0.6],
            "replay_policy_rank_reference_n": [10],
            "base_pred_delta": [0.0],
            "meta_pred_delta": [0.02],
            "calibrated_score_delta": [0.0],
            "rank_percentile_delta": [0.0],
        }
    )

    failures = _parity_failures(
        frame,
        tolerance=0.01,
        require_policy_rank_reference=True,
        require_live_values=True,
    )

    assert failures == ["meta_pred_delta_max_abs=0.02>tol=0.01"]


def test_parity_failures_can_gate_on_logged_model_input():
    frame = pd.DataFrame(
        {
            "live_base_pred": [0.1],
            "live_meta_pred": [0.2],
            "live_calibrated_score": [0.2],
            "live_policy_rank_pct": [0.6],
            "replay_base_pred": [0.5],
            "replay_meta_pred": [0.2],
            "replay_calibrated_score": [0.2],
            "replay_policy_rank_pct": [0.6],
            "logged_base_input_pred": [0.1],
            "logged_meta_input_pred": [0.2],
            "logged_meta_input_calibrated_score": [0.2],
            "logged_meta_input_policy_rank_pct": [0.6],
            "base_pred_delta": [0.4],
            "logged_base_input_pred_delta": [0.0],
            "logged_meta_input_pred_delta": [0.0],
            "logged_meta_input_calibrated_score_delta": [0.0],
            "logged_meta_input_rank_percentile_delta": [0.0],
        }
    )

    assert _parity_failures(
        frame,
        tolerance=0.01,
        parity_source="replay",
    ) == ["base_pred_delta_max_abs=0.4>tol=0.01"]
    assert _parity_failures(
        frame,
        tolerance=0.01,
        parity_source="logged-input",
    ) == []


def test_replay_parity_gates_on_active_side_residual_expert_not_direct_meta():
    frame = pd.DataFrame(
        {
            "live_base_pred": [0.1],
            "live_meta_pred": [0.8],
            "live_calibrated_score": [0.8],
            "live_policy_rank_pct": [0.9],
            "replay_base_pred": [0.1],
            "replay_meta_pred": [0.2],
            "replay_v9_input_meta_score_raw_refit": [0.8],
            "replay_calibrated_score": [0.8],
            "replay_policy_rank_pct": [0.9],
            "base_pred_delta": [0.0],
            "meta_pred_delta": [-0.6],
            "v9_input_meta_raw_refit_delta": [0.0],
            "calibrated_score_delta": [0.0],
            "rank_percentile_delta": [0.0],
            "side_residual_expert_active": [True],
            "residual_expert_feature_missing_count": [0],
            "residual_expert_feature_max_abs_delta": [0.0],
            "residual_expert_aegmm_feature_common_count": [7],
        }
    )

    assert _parity_failures(
        frame,
        tolerance=1e-7,
        prediction_tolerance=1e-7,
        parity_source="replay",
    ) == []

    frame.loc[0, "v9_input_meta_raw_refit_delta"] = 0.01
    assert _parity_failures(
        frame,
        tolerance=1e-7,
        prediction_tolerance=1e-7,
        parity_source="replay",
    ) == ["v9_input_meta_raw_refit_delta_max_abs=0.01>tol=1e-07"]

    frame.loc[0, "v9_input_meta_raw_refit_delta"] = 0.0
    frame.loc[0, "residual_expert_aegmm_feature_common_count"] = 0
    assert _parity_failures(
        frame,
        tolerance=1e-7,
        prediction_tolerance=1e-7,
        parity_source="replay",
    ) == ["residual_expert_aegmm_parity_unavailable_rows=1"]


def test_threshold_basis_parity_does_not_require_unmaterialized_dynamic_threshold():
    frame = pd.DataFrame(
        {
            "live_base_pred": [0.1],
            "live_meta_pred": [0.2],
            "live_calibrated_score": [0.2],
            "live_policy_rank_pct": [0.6],
            "live_threshold_basis_rank_score": [0.6],
            "live_rank_score_source": ["threshold_basis:ev_target_v1"],
            "logged_base_input_pred": [0.1],
            "logged_meta_input_pred": [0.2],
            "replay_policy_rank_reference_n": [10],
            "logged_base_input_pred_delta": [0.0],
            "logged_meta_input_pred_delta": [0.0],
            "threshold_basis_rank_internal_delta": [0.0],
            "threshold_basis_policy_rank_internal_delta": [0.0],
            "v9_parent_rank_delta": [0.0],
            "mlp_hier_ev_score_delta": [0.0],
            "expected_net_ev_after_1pct_delta": [0.0],
            "expected_ev_rank_score_delta": [0.0],
            "canonical_final_rank_delta": [0.0],
        }
    )

    assert _parity_failures(
        frame,
        tolerance=1e-7,
        prediction_tolerance=1e-7,
        parity_source="logged-input",
        require_policy_rank_reference=True,
        require_live_values=True,
    ) == []


def test_live_synthesized_feature_drift_summary_flags_gated_keys():
    feature_row = pd.DataFrame(
        [{"ret1h_G_VOL_1": 0.059209734, "ret1h": 0.059209734}],
        index=["PORTAL/USD:USD"],
    )
    logged = json.dumps({"ret1h_G_VOL_1": -0.0088211894, "ret1h": 0.059209734})

    drift = _live_synthesized_feature_delta_summary(
        logged_values_raw=logged,
        feature_row=feature_row,
        symbol="PORTAL/USD:USD",
    )

    assert drift["count"] == 1
    assert drift["worst_feature"] == "ret1h_G_VOL_1"
    assert drift["max_abs"] > 0.068


def test_summary_reports_live_synthesized_reconstruction_drift():
    frame = pd.DataFrame(
        {
            "replay_missing_features": [False],
            "base_live_synth_feature_value_max_abs_delta": [0.068],
            "base_live_synth_feature_value_worst_feature": ["ret1h_G_VOL_1"],
            "meta_live_synth_feature_value_max_abs_delta": [0.0],
            "meta_live_synth_feature_value_worst_feature": [""],
        }
    )

    summary = _summary(frame)

    drift = summary["live_synthesized_feature_reconstruction_drift"]
    assert drift["base"]["rows_gt_1e-7"] == 1
    assert drift["base"]["top_worst_features"] == {"ret1h_G_VOL_1": 1}


def test_parity_failures_detect_missing_policy_reference_and_live_fields():
    frame = pd.DataFrame(
        {
            "live_meta_pred": [0.2],
            "replay_policy_rank_reference_n": [0],
            "meta_pred_delta": [0.0],
        }
    )

    failures = _parity_failures(
        frame,
        tolerance=0.01,
        require_policy_rank_reference=True,
        require_live_values=True,
    )

    assert "missing_policy_rank_reference_rows=1" in failures
    assert "missing_live_base_pred_rows=1" in failures
    assert "missing_live_calibrated_score_rows=1" in failures
    assert "missing_live_policy_rank_pct_rows=1" in failures


def test_model_runtime_cfg_preserves_feature_cfg_and_can_disable_diagnostics():
    model_bundle = {"models": {"demo": object()}}
    cfg = _model_runtime_cfg(
        model_bundle=model_bundle,
        feature_runtime_cfg={"market_mode": "perps", "live_feature_source_run_id": "run_a"},
        disable_model_diagnostics=True,
        disable_model_timing=True,
    )

    assert cfg["model_bundle"] is model_bundle
    assert cfg["market_mode"] == "perps"
    assert cfg["live_feature_source_run_id"] == "run_a"
    assert cfg["inference_lgbm_internal_diagnostics_enabled"] is False
    assert cfg["inference_model_timing_enabled"] is False


def test_slice_panel_limits_datetime_frames_to_replay_window():
    idx = pd.date_range("2026-05-15T00:00:00Z", periods=4, freq="h")
    panel = {
        "close": pd.DataFrame({"AAA/USD:USD": [1, 2, 3, 4]}, index=idx),
        "metadata": pd.DataFrame({"value": [1]}),
    }

    sliced = _slice_panel(
        panel,
        start_ts=pd.Timestamp("2026-05-15T01:00:00Z"),
        end_ts=pd.Timestamp("2026-05-15T02:00:00Z"),
    )

    assert sliced["close"].index.tolist() == list(idx[1:3])
    assert sliced["metadata"].equals(panel["metadata"])


def test_live_feature_cache_symbols_prefers_smallest_matching_universe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "cache" / "inference_live_features" / "run_a"
    small = root / "small"
    large = root / "large"
    small.mkdir(parents=True)
    large.mkdir(parents=True)
    ts = "2026-05-15T23:45:00+00:00"
    small_symbols = [f"S{i}/USDC" for i in range(30)]
    large_symbols = [f"S{i}/USDC" for i in range(45)]
    (small / "meta.json").write_text(json.dumps({"end_ts": ts, "symbols": small_symbols}))
    (large / "meta.json").write_text(json.dumps({"end_ts": ts, "symbols": large_symbols}))

    symbols = _live_feature_cache_symbols_for_end(
        tmp_path,
        run_id="run_a",
        end_ts=pd.Timestamp(ts),
    )

    assert symbols == sorted(small_symbols)


def test_source_parity_context_symbols_include_source_rejected_compute_symbols(tmp_path):
    report_dir = (
        tmp_path
        / "exchanges"
        / "krakenfutures"
        / "artifacts"
        / "run_a"
        / "live_source_parity"
    )
    report_dir.mkdir(parents=True)
    ts = pd.Timestamp("2026-07-15T07:00:00Z")
    (report_dir / "20260715T070000Z_model_sources.json").write_text(
        json.dumps(
            {
                "end_ts": ts.isoformat(),
                "accepted_symbols": ["AAA/USD:USD", "BBB/USD:USD"],
                "rejected_symbols": ["GBP/USD:USD"],
            }
        )
    )

    symbols = _source_parity_context_symbols_for_end(
        tmp_path,
        run_id="run_a",
        end_ts=ts,
        live_quote_currency="USD",
        market_mode="perps",
        exchange_id="krakenfutures",
    )

    assert symbols == ["AAA/USD:USD", "BBB/USD:USD", "GBP/USD:USD"]


def test_historical_parity_adds_benchmark_context_for_residual_features():
    symbols = _add_required_context_symbols(
        ["AAA/USD:USD"],
        {"ret4h_bench_resid"},
        market_mode="perps",
    )

    assert "AAA/USD:USD" in symbols
    assert "BTC/USD:USD" in symbols


def test_historical_parity_keeps_sample_basket_without_residual_features():
    symbols = _add_required_context_symbols(
        ["AAA/USD:USD"],
        {"ret24h"},
        market_mode="perps",
    )

    assert symbols == ["AAA/USD:USD"]


def test_historical_parity_uses_rolling_cache_not_latest_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.historical_inference_parity.load_inference_config",
        lambda **kwargs: {"runtime_cfg": {}},
    )
    cfg = _build_runtime_cfg(
        data_root=tmp_path / "data",
        artifact_data_root=tmp_path / "artifacts",
        run_id="run_a",
        market_mode="perps",
        state={"bundle": {}},
    )
    runtime_cfg = cfg["runtime_cfg"]

    assert runtime_cfg["live_feature_snapshot_cache_enabled"] is False
    assert runtime_cfg["live_feature_rolling_cache_enabled"] is True
    assert runtime_cfg["live_feature_return_latest_only"] is False


def test_load_panel_preserves_perp_ohlcv_extras_and_overlays_microdata(tmp_path):
    symbol = "AAA/USD:USD"
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2026-05-15 10:00", tz="UTC")],
        name="ts",
    )
    store = PartitionedOHLCVStore(str(tmp_path), timeframe="1h")
    store.save_partitioned(
        symbol,
        pd.DataFrame(
            {
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.0],
                "volume": [100.0],
                "mark_open": [1.01],
                "mark_price": [1.02],
                "index_price": [1.00],
            },
            index=idx,
        ),
    )
    funding_dir = tmp_path / "funding_hourly"
    funding_dir.mkdir()
    pd.DataFrame(
        {
            "mark_price": [1.03],
            "index_price": [1.00],
            "funding_rate": [0.0001],
            "open_interest": [np.nan],
        },
        index=idx,
    ).to_parquet(funding_dir / "AAA_USD_USD.parquet")
    open_interest_dir = tmp_path / "open_interest_hourly"
    open_interest_dir.mkdir()
    pd.DataFrame(
        {"openInterestValue": [1234.0]},
        index=idx,
    ).to_parquet(open_interest_dir / "AAA_USD_USD.parquet")

    panel = _load_panel(
        data_root=tmp_path,
        symbols=[symbol],
        start_ts=idx[0],
        end_ts=idx[0],
    )

    assert panel["mark_open"].loc[idx[0], symbol] == 1.01
    assert panel["mark_price"].loc[idx[0], symbol] == 1.03
    assert panel["funding_rate"].loc[idx[0], symbol] == 0.0001
    assert panel["open_interest"].loc[idx[0], symbol] == 1234.0
