import pandas as pd

from extreme_price_movements.inference import feature_generator as fg


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
