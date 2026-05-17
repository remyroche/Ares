import pandas as pd

from extreme_price_movements.inference import feature_generator
from extreme_price_movements.inference import run_inference


def test_model_features_are_computed_on_full_tradable_universe(monkeypatch):
    symbols = ["AAA/USDC", "BBB/USDC", "CCC/USDC"]
    idx = pd.date_range("2026-05-15", periods=3, freq="1h", tz="UTC")
    panel = {
        "close": pd.DataFrame(
            {
                "AAA/USDC": [1.0, 1.1, 1.2],
                "BBB/USDC": [2.0, 2.1, 2.2],
                "CCC/USDC": [3.0, 3.1, 3.2],
            },
            index=idx,
        )
    }
    captured = {}

    monkeypatch.setattr(run_inference, "compute_selector_features", lambda panel, symbols: {})
    monkeypatch.setattr(run_inference, "get_candidate_thresholds", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_inference, "raw_required_feature_keys", lambda keys: set())
    monkeypatch.setattr(
        run_inference,
        "build_strategy_candidate_masks",
        lambda panel, selector_feats, rows: {"long_demo": ["AAA/USDC"]},
    )

    def fake_load_or_compute_features(**kwargs):
        captured["basket_syms"] = list(kwargs["basket_syms"])
        captured["panel_symbols"] = list(kwargs["panel"]["close"].columns)
        return {}

    monkeypatch.setattr(run_inference, "load_or_compute_features", fake_load_or_compute_features)
    monkeypatch.setattr(run_inference, "validate_required_feature_frames", lambda *args, **kwargs: None)

    run_inference._select_candidates_and_load_features(
        panel=panel,
        symbols=symbols,
        run_id="run_a",
        data_root="data",
        cfg={},
        lookback_hours=1440,
        required_feature_keys=set(),
        lgbm_strategy_mask_rows={"long_demo": {"trade_side": "long"}},
    )

    assert captured["basket_syms"] == symbols
    assert captured["panel_symbols"] == symbols


def test_stale_orderbook_features_are_not_forward_filled_into_live_snapshot():
    old_ts = pd.Timestamp("2026-05-08 18:00", tz="UTC")
    end_ts = pd.Timestamp("2026-05-15 22:00", tz="UTC")
    feats = {
        "ob_spread_bps": pd.DataFrame(
            {"AAA/USDC": [15.0]},
            index=pd.DatetimeIndex([old_ts]),
        ),
        "ret24h": pd.DataFrame(
            {"AAA/USDC": [0.03]},
            index=pd.DatetimeIndex([old_ts]),
        ),
    }

    matrix = feature_generator._latest_feature_matrix(
        feats,
        symbols=["AAA/USDC"],
        end_ts=end_ts,
        required_feature_keys={"ob_spread_bps", "ret24h"},
    )

    assert "ret24h" in matrix.columns
    assert matrix.loc["AAA/USDC", "ret24h"] == 0.03
    assert "ob_spread_bps" not in matrix.columns


def test_stale_orderbook_features_are_dropped_before_live_safe_materialization():
    old_ts = pd.Timestamp("2026-05-08 18:00", tz="UTC")
    end_ts = pd.Timestamp("2026-05-15 22:00", tz="UTC")
    idx = pd.DatetimeIndex([end_ts])
    panel = {
        "close": pd.DataFrame({"AAA/USDC": [1.0]}, index=idx),
        "volume": pd.DataFrame({"AAA/USDC": [100.0]}, index=idx),
    }
    feats = {
        "ob_spread_bps": pd.DataFrame(
            {"AAA/USDC": [15.0]},
            index=pd.DatetimeIndex([old_ts]),
        )
    }

    dropped = feature_generator._drop_stale_live_sensitive_features(
        feats,
        end_ts=end_ts,
        required_feature_keys={"ob_spread_bps"},
    )
    materialized = feature_generator._synthesize_live_safe_feature_keys(
        dropped,
        panel,
        ["AAA/USDC"],
        {"ob_spread_bps"},
    )

    assert "ob_spread_bps" in materialized
    assert list(materialized["ob_spread_bps"].index) == [end_ts]
    assert materialized["ob_spread_bps"].loc[end_ts, "AAA/USDC"] == 0.0


def test_live_orderbook_model_features_stay_neutral_without_training_support(tmp_path):
    run_id = "run_a"
    health_dir = tmp_path / "artifacts" / run_id / "features"
    health_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "symbol": ["AAA/USDC"],
            "feature": ["ob_spread_bps"],
            "rows": [10],
            "nan_count": [0],
            "nan_pct": [0.0],
            "leading_nan_count": [0],
            "interior_nan_count": [0],
            "trailing_nan_count": [0],
            "is_all_nan": [False],
            "is_constant_non_nan": [True],
        }
    ).to_csv(health_dir / "feature_health_feature_detail.csv", index=False)
    end_ts = pd.Timestamp("2026-05-15 22:00", tz="UTC")
    idx = pd.DatetimeIndex([end_ts - pd.Timedelta(hours=1), end_ts])
    panel = {
        "close": pd.DataFrame({"AAA/USDC": [1.0, 1.0]}, index=idx),
        "volume": pd.DataFrame({"AAA/USDC": [100.0, 100.0]}, index=idx),
        "orderbook_best_bid": pd.DataFrame({"AAA/USDC": [0.99, 0.99]}, index=idx),
        "orderbook_best_ask": pd.DataFrame({"AAA/USDC": [1.01, 1.01]}, index=idx),
        "orderbook_mid": pd.DataFrame({"AAA/USDC": [1.0, 1.0]}, index=idx),
    }

    materialized = feature_generator._synthesize_live_safe_feature_keys(
        {},
        panel,
        ["AAA/USDC"],
        {"ob_spread_bps"},
        data_root=str(tmp_path),
        run_id=run_id,
        cfg={},
    )

    assert "ob_spread_bps" in materialized
    assert materialized["ob_spread_bps"].loc[end_ts, "AAA/USDC"] == 0.0


def test_candidate_extraction_skips_stale_orderbook_values_at_requested_ts():
    old_ts = pd.Timestamp("2026-05-08 18:00", tz="UTC")
    end_ts = pd.Timestamp("2026-05-15 22:00", tz="UTC")
    feats = {
        "ob_spread_bps": pd.DataFrame(
            {"AAA/USDC": [15.0]},
            index=pd.DatetimeIndex([old_ts]),
        ),
        "ret24h": pd.DataFrame(
            {"AAA/USDC": [0.03]},
            index=pd.DatetimeIndex([old_ts]),
        ),
    }

    matrix = feature_generator.get_features_for_candidates(
        feats,
        ["AAA/USDC"],
        ts=end_ts,
    )

    assert matrix.loc["AAA/USDC", "ret24h"] == 0.03
    assert "ob_spread_bps" not in matrix.columns


def test_tail_warmup_covers_causal_transform_window():
    warmup = feature_generator._required_tail_warmup_hours(
        lookback_hours=24 * 60,
        trend_sma_hours=24 * 14,
        gate_vol_lookback_hours=24 * 7,
    )

    assert warmup >= (
        feature_generator.DEFAULT_CAUSAL_TRANSFORM_ROLL_WINDOW_HOURS
        + feature_generator.DEFAULT_TAIL_WARMUP_BUFFER_HOURS
    )
    assert warmup >= (
        feature_generator.DEFAULT_IDENTITY_EWMA_WARMUP_HOURS
        + feature_generator.DEFAULT_TAIL_WARMUP_BUFFER_HOURS
    )


def test_live_feature_cache_key_includes_runtime_feature_config():
    base = {
        "run_id": "run_a",
        "symbols": ["AAA/USDC"],
        "required_feature_keys": {"ret24h"},
        "lookback_hours": 24 * 60,
    }

    key_a = feature_generator._live_feature_cache_key(
        **base,
        cfg={"train_min_range_pct": 0.06},
    )
    key_b = feature_generator._live_feature_cache_key(
        **base,
        cfg={"train_min_range_pct": 0.07},
    )

    assert key_a != key_b


def test_cached_feature_coverage_uses_stalest_required_frame():
    idx = pd.date_range("2026-05-15", periods=4, freq="1h", tz="UTC")
    feats = {
        "fresh": pd.DataFrame({"AAA/USDC": [1, 2, 3, 4]}, index=idx),
        "stale": pd.DataFrame({"AAA/USDC": [1, 2]}, index=idx[:2]),
        "irrelevant_older": pd.DataFrame({"AAA/USDC": [1]}, index=idx[:1]),
    }

    coverage = feature_generator._cached_feature_coverage_end_ts(
        feats,
        required_feature_keys={"fresh", "stale"},
    )

    assert coverage == idx[1]


def test_merge_feature_dicts_is_deterministic_sorted_order():
    idx = pd.date_range("2026-05-15", periods=1, freq="1h", tz="UTC")
    merged = feature_generator._merge_feature_dicts(
        {
            "z_feature": pd.DataFrame({"AAA/USDC": [1.0]}, index=idx),
            "a_feature": pd.DataFrame({"AAA/USDC": [2.0]}, index=idx),
        },
        {"m_feature": pd.DataFrame({"AAA/USDC": [3.0]}, index=idx)},
    )

    assert list(merged) == ["a_feature", "m_feature", "z_feature"]
