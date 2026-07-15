import json
import math

import pandas as pd

from extreme_price_movements.data_store import PartitionedOHLCVStore
from scripts.historical_inference_parity import (
    _add_required_context_symbols,
    _build_runtime_cfg,
)
from scripts.replay_live_signal_predictions import (
    _live_synthesized_feature_delta_summary,
    _live_feature_cache_symbols_for_end,
    _load_panel,
    _load_recent_decisions,
    _model_runtime_cfg,
    _parity_failures,
    _slice_panel,
    _summary,
)


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

    assert failures == ["meta_pred_delta_max_abs=0.02"]


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
    ) == ["base_pred_delta_max_abs=0.4"]
    assert _parity_failures(
        frame,
        tolerance=0.01,
        parity_source="logged-input",
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
        },
        index=idx,
    ).to_parquet(funding_dir / "AAA_USD_USD.parquet")

    panel = _load_panel(
        data_root=tmp_path,
        symbols=[symbol],
        start_ts=idx[0],
        end_ts=idx[0],
    )

    assert math.isclose(float(panel["mark_open"].loc[idx[0], symbol]), 1.01, rel_tol=1e-5)
    assert math.isclose(float(panel["mark_price"].loc[idx[0], symbol]), 1.03, rel_tol=1e-5)
    assert math.isclose(float(panel["funding_rate"].loc[idx[0], symbol]), 0.0001, rel_tol=1e-5)
