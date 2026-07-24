import pandas as pd
import numpy as np

from extreme_price_movements.inference import feature_generator as fg


def test_authoritative_offline_load_skips_only_deterministic_materializations():
    required = {
        "selected_training_feature",
        "barrier_pct",
        "ret1h_G_VOL_0",
        "ret1h_G_VOL_1",
        "side",
        "score",
    }

    assert fg._offline_cache_feature_keys(
        required,
        authoritative_model_offline_cache=True,
    ) == {"selected_training_feature"}
    assert fg._offline_cache_feature_keys(
        required,
        authoritative_model_offline_cache=False,
    ) == required


def test_static_feature_persistence_is_deferred_and_latest_row_only(monkeypatch):
    calls = []
    monkeypatch.setattr(
        fg,
        "append_static_features",
        lambda feats, **kwargs: calls.append((feats, kwargs)),
    )
    index = pd.date_range("2026-07-10T14:00:00Z", periods=3, freq="1h")
    frame = pd.DataFrame(
        np.arange(6, dtype=np.float32).reshape(3, 2),
        index=index,
        columns=["A", "B"],
    )

    queued = fg.defer_static_feature_append_after_scoring(
        {"feature_a": frame},
        feature_store_ts=index[-1],
        data_root="data_perp",
        feature_store_id="store",
        index=index,
        columns=["A", "B"],
        min_timestamp_by_symbol={"A": index[-2], "B": index[-2]},
        save_workers=2,
        source="test",
        block_max_timestamps=1,
    )

    assert queued == 1
    assert calls == []
    result = fg.flush_deferred_static_feature_appends(wait=True)
    assert result["launched"] == 1
    assert result["failed"] == 0
    assert len(calls) == 1
    persisted = calls[0][0]["feature_a"]
    assert list(persisted.index) == [index[-1]]


def test_wait_for_live_feature_sync_returns_when_sidecar_ready(monkeypatch):
    monkeypatch.setattr(
        fg,
        "_live_feature_syncs_for_target",
        lambda **kwargs: [{"pid": 123, "alive": True, "process_status": "running"}],
    )
    monkeypatch.setattr(
        fg,
        "_live_latest_feature_matrix_presence",
        lambda **kwargs: (
            True,
            {
                "rows": 10,
                "features": 3,
                "full_feature_rows": 10,
                "missing_features": 0,
            },
        ),
    )

    result = fg._wait_for_live_feature_syncs_for_target(
        data_root="data_perp",
        run_id="run",
        end_ts=pd.Timestamp("2026-07-10T16:00:00Z"),
        timeout_s=60.0,
        heartbeat_s=5.0,
        reason="test",
        symbols=["BTC/USD:USD"],
        feature_keys=["a", "b"],
    )

    assert result["status"] == "existing_sync_sidecar_ready"
    assert result["sidecar"]["full_feature_rows"] == 10


def test_prewarm_uses_ready_sidecar_before_existing_sync_wait(monkeypatch):
    monkeypatch.setattr(fg, "_live_model_feature_prewarm_enabled", lambda cfg: True)
    monkeypatch.setattr(fg, "_live_model_feature_auto_sync_enabled", lambda cfg: True)
    monkeypatch.setattr(
        fg,
        "_live_training_path_sync_feature_keys",
        lambda keys, cfg: (set(keys), []),
    )
    monkeypatch.setattr(
        fg, "_offline_feature_lookup_run_ids", lambda cfg, run_id: [run_id]
    )
    monkeypatch.setattr(
        fg, "_offline_feature_lookup_data_root", lambda cfg, data_root: data_root
    )
    monkeypatch.setattr(
        fg,
        "_live_latest_feature_matrix_presence",
        lambda **kwargs: (
            True,
            {
                "rows": 10,
                "features": 2,
                "full_feature_rows": 10,
                "missing_features": 0,
            },
        ),
    )
    monkeypatch.setattr(
        fg,
        "_prewarm_selected_latest_matrix_memory",
        lambda **kwargs: {
            "matrix_loaded": True,
            "matrix_complete": True,
            "matrix_missing_features": 0,
        },
    )

    def _unexpected_existing_sync_probe(**kwargs):
        raise AssertionError("existing sync wait should not run when sidecar is ready")

    monkeypatch.setattr(
        fg, "_live_feature_syncs_for_target", _unexpected_existing_sync_probe
    )

    result = fg.prewarm_selected_model_feature_cache_for_live(
        run_id="run",
        data_root="data_perp",
        symbols=["BTC/USD:USD"],
        end_ts=pd.Timestamp("2026-07-10T16:00:00Z"),
        cfg={},
        required_feature_keys=["a", "b"],
    )

    assert result["status"] == "cache_hit"
    assert result["matrix_loaded"] is True


def test_prewarm_uses_incremental_sidecar_before_training_sync(monkeypatch):
    monkeypatch.setattr(fg, "_live_model_feature_prewarm_enabled", lambda cfg: True)
    monkeypatch.setattr(fg, "_live_model_feature_auto_sync_enabled", lambda cfg: True)
    monkeypatch.setattr(
        fg,
        "_live_training_path_sync_feature_keys",
        lambda keys, cfg: (set(keys), []),
    )
    monkeypatch.setattr(
        fg, "_offline_feature_lookup_run_ids", lambda cfg, run_id: [run_id]
    )
    monkeypatch.setattr(
        fg, "_offline_feature_lookup_data_root", lambda cfg, data_root: data_root
    )
    monkeypatch.setattr(
        fg,
        "_live_latest_feature_matrix_presence",
        lambda **kwargs: (
            False,
            {"last_error": "missing_sidecar", "missing_features": 2},
        ),
    )
    monkeypatch.setattr(fg, "_live_feature_syncs_for_target", lambda **kwargs: [])
    monkeypatch.setattr(
        fg,
        "_prewarm_selected_latest_matrix_memory",
        lambda **kwargs: {"matrix_loaded": False, "matrix_reason": "missing"},
    )

    calls = {"incremental": 0}

    def _fake_incremental(**kwargs):
        calls["incremental"] += 1
        assert kwargs["panel"] is panel
        assert kwargs["lookback_hours"] == 720
        assert kwargs["force_full_repair"] is False
        return {
            "status": "incremental_selected_matrix_ready",
            "ok": True,
            "source_run_id": "run",
            "matrix_loaded": True,
            "matrix_complete": True,
            "matrix_missing_features": 0,
        }

    monkeypatch.setattr(
        fg, "_build_incremental_selected_feature_sidecar_for_live", _fake_incremental
    )

    def _unexpected_training_sync(**kwargs):
        raise AssertionError(
            "training sync should not run after incremental sidecar succeeds"
        )

    monkeypatch.setattr(
        fg, "_run_training_path_feature_sync_for_live", _unexpected_training_sync
    )

    panel = {
        "close": pd.DataFrame(
            {"BTC/USD:USD": [1.0]},
            index=[pd.Timestamp("2026-07-10T16:00:00Z")],
        )
    }
    result = fg.prewarm_selected_model_feature_cache_for_live(
        run_id="run",
        data_root="data_perp",
        symbols=["BTC/USD:USD"],
        end_ts=pd.Timestamp("2026-07-10T16:00:00Z"),
        cfg={},
        required_feature_keys=["a", "b"],
        panel=panel,
        lookback_hours=720,
    )

    assert calls["incremental"] == 1
    assert result["status"] == "incremental_selected_matrix_ready"
    assert result["matrix_loaded"] is True


def test_incremental_sidecar_reuses_only_validated_rolling_cache(monkeypatch):
    end_ts = pd.Timestamp("2026-07-10T16:00:00Z")
    panel = {
        "close": pd.DataFrame(
            {"BTC/USD:USD": [100.0, 101.0]},
            index=[end_ts - pd.Timedelta(hours=1), end_ts],
        )
    }
    captured_cfg = {}

    monkeypatch.setattr(fg, "_sidecar_backed_feature_keys", lambda keys: set(keys))

    def _fake_compute(**kwargs):
        captured_cfg.update(kwargs["cfg"])
        return {
            "lr_1h": pd.DataFrame(
                {"BTC/USD:USD": [0.01]},
                index=[end_ts],
            )
        }

    monkeypatch.setattr(fg, "load_or_compute_features", _fake_compute)
    monkeypatch.setattr(
        fg,
        "_resolve_feature_store_ts",
        lambda run_id, data_root, end_ts=None: pd.Timestamp("2026-07-01T07:00:00Z"),
    )
    monkeypatch.setattr(fg, "write_live_latest_feature_matrix", lambda *a, **k: None)
    monkeypatch.setattr(
        fg, "_write_selected_feature_latest_matrix_cache", lambda *a, **k: None
    )
    monkeypatch.setattr(
        fg,
        "_live_latest_feature_matrix_presence",
        lambda **kwargs: (True, {"rows": 1, "features": 1}),
    )
    monkeypatch.setattr(
        fg,
        "_prewarm_selected_latest_matrix_memory",
        lambda **kwargs: {
            "matrix_loaded": True,
            "matrix_complete": True,
            "matrix_missing_features": 0,
        },
    )

    result = fg._build_incremental_selected_feature_sidecar_for_live(
        panel=panel,
        lookback_hours=720,
        run_id="live-run",
        data_root="data_perp",
        source_run_id="feature-run",
        source_data_root="data_perp",
        symbols=["BTC/USD:USD"],
        end_ts=end_ts,
        cfg={},
        feature_keys={"lr_1h"},
        min_symbol_coverage=0.8,
    )

    assert result["status"] == "incremental_selected_matrix_ready"
    assert captured_cfg["live_feature_offline_cache_enabled"] is False
    assert captured_cfg["live_feature_prefer_offline_cache"] is False
    assert captured_cfg["live_feature_cycle_cache_bypass"] is True
    assert captured_cfg["live_feature_memory_cache_enabled"] is False
    assert captured_cfg["live_feature_snapshot_cache_enabled"] is False
    assert captured_cfg["live_feature_rolling_cache_enabled"] is True


def test_incremental_sidecar_disables_rolling_cache_for_full_repair(monkeypatch):
    end_ts = pd.Timestamp("2026-07-10T16:00:00Z")
    captured_cfg = {}
    monkeypatch.setattr(fg, "_sidecar_backed_feature_keys", lambda keys: set(keys))

    def _fake_compute(**kwargs):
        captured_cfg.update(kwargs["cfg"])
        return {
            "lr_1h": pd.DataFrame(
                {"BTC/USD:USD": [0.01]},
                index=[end_ts],
            )
        }

    monkeypatch.setattr(fg, "load_or_compute_features", _fake_compute)
    monkeypatch.setattr(
        fg,
        "_resolve_feature_store_ts",
        lambda run_id, data_root, end_ts=None: pd.Timestamp("2026-07-01T07:00:00Z"),
    )
    monkeypatch.setattr(fg, "write_live_latest_feature_matrix", lambda *a, **k: None)
    monkeypatch.setattr(
        fg, "_write_selected_feature_latest_matrix_cache", lambda *a, **k: None
    )
    monkeypatch.setattr(
        fg,
        "_live_latest_feature_matrix_presence",
        lambda **kwargs: (True, {"rows": 1, "features": 1}),
    )
    monkeypatch.setattr(
        fg,
        "_prewarm_selected_latest_matrix_memory",
        lambda **kwargs: {
            "matrix_loaded": True,
            "matrix_complete": True,
            "matrix_missing_features": 0,
        },
    )

    result = fg._build_incremental_selected_feature_sidecar_for_live(
        panel={
            "close": pd.DataFrame(
                {"BTC/USD:USD": [100.0, 101.0]},
                index=[end_ts - pd.Timedelta(hours=1), end_ts],
            )
        },
        lookback_hours=720,
        run_id="live-run",
        data_root="data_perp",
        source_run_id="feature-run",
        source_data_root="data_perp",
        symbols=["BTC/USD:USD"],
        end_ts=end_ts,
        cfg={},
        feature_keys={"lr_1h"},
        min_symbol_coverage=0.8,
        force_full_repair=True,
    )

    assert result["status"] == "incremental_selected_matrix_ready"
    assert captured_cfg["live_feature_rolling_cache_enabled"] is False


def test_prewarm_does_not_reload_known_incomplete_matrix_before_repair(monkeypatch):
    monkeypatch.setattr(fg, "_live_model_feature_prewarm_enabled", lambda cfg: True)
    monkeypatch.setattr(fg, "_live_model_feature_auto_sync_enabled", lambda cfg: True)
    monkeypatch.setattr(
        fg,
        "_live_training_path_sync_feature_keys",
        lambda keys, cfg: (set(keys), []),
    )
    monkeypatch.setattr(
        fg, "_offline_feature_lookup_run_ids", lambda cfg, run_id: [run_id]
    )
    monkeypatch.setattr(
        fg, "_offline_feature_lookup_data_root", lambda cfg, data_root: data_root
    )
    monkeypatch.setattr(
        fg,
        "_live_latest_feature_matrix_presence",
        lambda **kwargs: (True, {"rows": 1, "features": 1}),
    )
    monkeypatch.setattr(fg, "_live_feature_syncs_for_target", lambda **kwargs: [])
    prewarm_calls = {"count": 0}

    def _incomplete_matrix(**kwargs):
        prewarm_calls["count"] += 1
        return {
            "matrix_loaded": True,
            "matrix_complete": False,
            "matrix_missing_features": 0,
            "matrix_low_finite_features": 1,
            "matrix_full_feature_rows": 0,
        }

    monkeypatch.setattr(fg, "_prewarm_selected_latest_matrix_memory", _incomplete_matrix)
    incremental_calls = []

    def _complete_repair(**kwargs):
        incremental_calls.append(kwargs)
        return {
            "status": "incremental_selected_matrix_ready",
            "ok": True,
            "matrix_loaded": True,
            "matrix_complete": True,
            "matrix_missing_features": 0,
        }

    monkeypatch.setattr(
        fg,
        "_build_incremental_selected_feature_sidecar_for_live",
        _complete_repair,
    )

    result = fg.prewarm_selected_model_feature_cache_for_live(
        run_id="run",
        data_root="data_perp",
        symbols=["BTC/USD:USD"],
        end_ts=pd.Timestamp("2026-07-10T16:00:00Z"),
        cfg={"live_model_feature_prewarm_accept_low_finite_with_row_strict": False},
        required_feature_keys=["lr_1h"],
        panel={
            "close": pd.DataFrame(
                {"BTC/USD:USD": [1.0]},
                index=[pd.Timestamp("2026-07-10T16:00:00Z")],
            )
        },
        lookback_hours=720,
    )

    assert result["status"] == "incremental_selected_matrix_ready"
    assert prewarm_calls["count"] == 1
    assert len(incremental_calls) == 1
    assert incremental_calls[0]["force_full_repair"] is True


def test_selected_feature_source_aliases_are_deduplicated_after_resolution(
    monkeypatch, tmp_path
):
    resolved_ts = pd.Timestamp("2026-07-11T07:00:00Z")
    calls = []
    monkeypatch.setattr(
        fg, "_offline_feature_lookup_data_roots", lambda data_root: [str(tmp_path)]
    )
    monkeypatch.setattr(
        fg,
        "_resolve_feature_store_ts",
        lambda run_id, root, end_ts=None: resolved_ts,
    )
    monkeypatch.setattr(
        fg, "_feature_store_dir_has_materialized_data", lambda path: True
    )

    def _fake_load(**kwargs):
        calls.append(kwargs["run_id"])
        return {
            "lr_1h": pd.DataFrame(
                {"BTC/USD:USD": [0.01]},
                index=[pd.Timestamp("2026-07-15T20:00:00Z")],
            )
        }

    monkeypatch.setattr(fg, "load_cached_features_for_inference", _fake_load)

    result = fg.load_cached_features_for_inference_sources(
        ["20260711_070000", "descriptive-model-run"],
        str(tmp_path),
        ["BTC/USD:USD"],
        feature_keys={"lr_1h"},
        end_ts=pd.Timestamp("2026-07-15T20:00:00Z"),
    )

    assert calls == ["20260711_070000"]
    assert list(result) == ["lr_1h"]
