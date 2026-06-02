import json

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements import features
from extreme_price_movements.data_store import LazyFeatureDict
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


def test_candidate_loader_can_stop_after_mask_features(monkeypatch):
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
    selector_feats = {
        "ret1h": pd.DataFrame(
            {"AAA/USDC": [0.1], "BBB/USDC": [-0.1], "CCC/USDC": [0.0]},
            index=pd.DatetimeIndex([idx[-1]]),
        )
    }
    load_calls = []

    monkeypatch.setattr(
        run_inference,
        "compute_selector_features",
        lambda panel, symbols: selector_feats,
    )
    monkeypatch.setattr(run_inference, "get_candidate_thresholds", lambda *args, **kwargs: {})
    monkeypatch.setattr(run_inference, "raw_required_feature_keys", lambda keys: {"model_feat"})
    monkeypatch.setattr(
        run_inference,
        "build_strategy_candidate_masks",
        lambda panel, selector_feats, rows: {"long_demo": ["AAA/USDC"]},
    )

    def fail_load_or_compute_features(**kwargs):
        load_calls.append(kwargs)
        raise AssertionError("model features should not be materialized")

    monkeypatch.setattr(
        run_inference,
        "load_or_compute_features",
        fail_load_or_compute_features,
    )

    _, long_cands, short_cands, feats, masks = (
        run_inference._select_candidates_and_load_features(
            panel=panel,
            symbols=symbols,
            run_id="run_a",
            data_root="data",
            cfg={},
            lookback_hours=1440,
            required_feature_keys={"model_feat"},
            lgbm_strategy_mask_rows={
                "long_demo": {
                    "trade_side": "long",
                    "base_event_trigger": "(*)|(ret1h>0.0)|(*)",
                }
            },
            model_features_required=False,
        )
    )

    assert load_calls == []
    assert long_cands == ["AAA/USDC"]
    assert short_cands == []
    assert masks == {"long_demo": ["AAA/USDC"]}
    assert sorted(feats) == ["ret1h"]


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


def test_stale_orderbook_features_require_live_source_or_training_neutrality():
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
    with pytest.raises(ValueError, match="cannot be materialized"):
        feature_generator._synthesize_live_safe_feature_keys(
            dropped,
            panel,
            ["AAA/USDC"],
            {"ob_spread_bps"},
        )


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


def test_live_orderbook_residual_features_materialize_from_summary_primitives():
    idx = pd.date_range("2026-05-01", periods=220, freq="1h", tz="UTC")
    symbols = ["BTC/USD:USD", "ETH/USD:USD", "AAVE/USD:USD", "DOGE/USD:USD"]
    t = np.arange(len(idx), dtype=np.float32)

    close = pd.DataFrame(
        {
            sym: 100.0 + i * 10.0 + 0.1 * t + np.sin(t / (7.0 + i))
            for i, sym in enumerate(symbols)
        },
        index=idx,
        dtype=np.float32,
    )
    volume = pd.DataFrame(
        {
            sym: 1000.0 + 25.0 * i + 10.0 * np.cos(t / (9.0 + i))
            for i, sym in enumerate(symbols)
        },
        index=idx,
        dtype=np.float32,
    )
    mid = close * (1.0 + 0.0001 * np.sin(t[:, None] / 11.0))
    spread = pd.DataFrame(
        {
            sym: 0.02 + 0.002 * i + 0.001 * np.sin(t / (5.0 + i))
            for i, sym in enumerate(symbols)
        },
        index=idx,
        dtype=np.float32,
    )
    bid_qty_1 = pd.DataFrame(
        {
            sym: 10.0 + i + np.sin(t / (6.0 + i))
            for i, sym in enumerate(symbols)
        },
        index=idx,
        dtype=np.float32,
    )
    ask_qty_1 = pd.DataFrame(
        {
            sym: 9.0 + i + np.cos(t / (8.0 + i))
            for i, sym in enumerate(symbols)
        },
        index=idx,
        dtype=np.float32,
    )

    panel = {
        "close": close,
        "volume": volume,
        "orderbook_best_bid": (mid - spread).astype(np.float32),
        "orderbook_best_ask": (mid + spread).astype(np.float32),
        "orderbook_mid": mid.astype(np.float32),
        "orderbook_bid_qty_1": bid_qty_1,
        "orderbook_ask_qty_1": ask_qty_1,
        "orderbook_cum_bid_qty_l10": bid_qty_1 * 8.0,
        "orderbook_cum_ask_qty_l10": ask_qty_1 * 8.5,
        "orderbook_cum_bid_qty_l20": bid_qty_1 * 14.0,
        "orderbook_cum_ask_qty_l20": ask_qty_1 * 15.0,
        "orderbook_buy_notional_1h": (close * volume * 0.55).astype(np.float32),
        "orderbook_sell_notional_1h": (close * volume * 0.45).astype(np.float32),
    }
    required = {"ob_imbalance_mkt_resid", "xasset_ob_liquidity_peer_resid"}

    materialized = feature_generator._materialize_live_orderbook_summary_features(
        {},
        panel,
        symbols,
        required,
        cfg={"market_basket": ["BTC/USDT", "ETH/USDT"]},
    )

    assert required.issubset(materialized)
    for key in sorted(required):
        tail = materialized[key].iloc[-1].to_numpy(dtype=np.float32)
        assert np.isfinite(tail).any(), key


def test_market_regime_scores_broadcast_to_required_live_symbols():
    idx = pd.date_range("2026-05-15", periods=2, freq="1h", tz="UTC")
    panel = {
        "close": pd.DataFrame(
            {
                "AAA/USDC": [1.0, 1.1],
                "BBB/USDC": [2.0, 2.1],
            },
            index=idx,
        )
    }
    feats = {
        "regime_trend_score": pd.DataFrame({"market": [0.2, 0.3]}, index=idx),
        "ret24h": pd.DataFrame({"AAA/USDC": [0.01, 0.02]}, index=idx),
    }

    materialized = feature_generator._ensure_required_symbol_columns(
        feats,
        panel,
        ["AAA/USDC", "BBB/USDC"],
        {"regime_trend_score", "ret24h"},
    )

    assert list(materialized["regime_trend_score"].columns) == ["AAA/USDC", "BBB/USDC"]
    assert materialized["regime_trend_score"].loc[idx[-1], "AAA/USDC"] == 0.3
    assert materialized["regime_trend_score"].loc[idx[-1], "BBB/USDC"] == 0.3
    assert pd.isna(materialized["ret24h"].loc[idx[-1], "BBB/USDC"])


def test_market_wide_frames_are_materialized_from_existing_regime_gates():
    idx = pd.date_range("2026-05-15", periods=2, freq="1h", tz="UTC")
    mkt_gates = pd.DataFrame(
        {
            "regime_trend_score": [0.2, 0.3],
            "regime_vol_score": [0.7, 0.8],
            "mkt_rv_ratio": [1.1, 1.2],
        },
        index=idx,
    )

    materialized = feature_generator._market_wide_feature_frames(
        mkt_gates,
        ["AAA/USDC", "BBB/USDC"],
        {"regime_trend_score", "regime_vol_score", "mkt_rv_ratio", "ret24h"},
    )

    assert sorted(materialized) == [
        "mkt_rv_ratio",
        "regime_trend_score",
        "regime_vol_score",
    ]
    assert materialized["regime_trend_score"].loc[idx[-1], "AAA/USDC"] == 0.3
    assert materialized["regime_trend_score"].loc[idx[-1], "BBB/USDC"] == 0.3
    assert materialized["regime_vol_score"].loc[idx[-1], "AAA/USDC"] == 0.8
    assert materialized["regime_vol_score"].loc[idx[-1], "BBB/USDC"] == 0.8
    assert materialized["mkt_rv_ratio"].loc[idx[-1], "AAA/USDC"] == 1.2
    assert materialized["mkt_rv_ratio"].loc[idx[-1], "BBB/USDC"] == 1.2


def test_live_lgbm_mask_market_basket_resolves_perp_symbols_by_base(monkeypatch):
    idx = pd.date_range("2026-05-15", periods=30, freq="1h", tz="UTC")
    columns = ["BTC/USD:USD", "ETH/USD:USD", "DOGE/USD:USD"]
    close = pd.DataFrame(
        {
            "BTC/USD:USD": np.linspace(100.0, 130.0, len(idx)),
            "ETH/USD:USD": np.linspace(50.0, 65.0, len(idx)),
            "DOGE/USD:USD": np.linspace(10.0, 40.0, len(idx)),
        },
        index=idx,
    )
    panel = {
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": pd.DataFrame(100.0, index=idx, columns=columns),
    }

    monkeypatch.setattr(
        features,
        "_apply_causal_transform_live_state_or_batch",
        lambda feats, cfg, **kwargs: feats,
    )

    out = features._compute_live_lgbm_mask_features_fast(
        panel,
        {
            "market_basket": ["BTC/USDT", "ETH/USDT"],
            "live_lgbm_mask_feature_fast_path_enabled": True,
        },
        {"mkt_ret_eq_24h"},
    )

    assert out is not None
    feats, _, _ = out
    c_log = features._safe_log_df(close.astype(np.float32), eps=1e-9)
    ffd04 = features._transform_close_fixed_ffd(
        c_log,
        d=0.4,
        _label="test_live_mask_close_d04",
        already_logged=True,
        thres=1e-5,
    )
    ret24 = features.ff.numba_rolling_sum(ffd04, 24).astype(np.float32)
    expected = ret24[["BTC/USD:USD", "ETH/USD:USD"]].mean(axis=1).iloc[-1]
    all_symbols = ret24.mean(axis=1).iloc[-1]

    actual = feats["mkt_ret_eq_24h"].loc[idx[-1], "DOGE/USD:USD"]
    assert actual == pytest.approx(expected)
    assert actual != pytest.approx(all_symbols)


def test_shared_market_basket_resolver_handles_perp_and_spot_symbol_styles():
    columns = pd.Index(["BTC/USD:USD", "ETH/USD:USD", "SOL/USDC", "DOGE/USD:USD"])

    resolved = features._resolve_basket_symbols_by_base(
        ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BTC/USDT", "MISSING/USDT"],
        columns,
    )

    assert resolved == ["BTC/USD:USD", "ETH/USD:USD", "SOL/USDC"]
    assert features._resolve_symbol_by_base("DOGE/USDT", columns) == "DOGE/USD:USD"
    assert features._resolve_symbol_by_base("BTC/USD:USD", columns) == "BTC/USD:USD"
    assert features._resolve_symbol_by_base("MISSING/USDT", columns) is None


def test_tail_append_keeps_history_for_newly_materialized_feature_keys():
    idx = pd.date_range("2026-05-15", periods=4, freq="1h", tz="UTC")
    cached_last_ts = idx[-1]
    cached = {
        "ret24h": pd.DataFrame({"AAA/USDC": [1.0]}, index=[idx[-1]]),
    }
    tail = {
        "ret24h": pd.DataFrame({"AAA/USDC": [0.8, 0.9]}, index=idx[-2:]),
        "regime_trend_score": pd.DataFrame(
            {"AAA/USDC": [0.1, 0.2, 0.3, 0.4]},
            index=idx,
        ),
    }

    sliced = feature_generator._slice_tail_features_for_cache_append(
        tail,
        cached,
        cached_last_ts,
    )

    assert "ret24h" not in sliced
    assert list(sliced["regime_trend_score"].index) == list(idx)
    assert sliced["regime_trend_score"].loc[idx[-1], "AAA/USDC"] == 0.4


def test_missing_feature_merge_replaces_non_dataframe_placeholders():
    idx = pd.date_range("2026-05-15", periods=2, freq="1h", tz="UTC")
    cached = {
        "regime_trend_score": pd.Series([0.1, 0.2], index=idx),
        "ret24h": pd.DataFrame({"AAA/USDC": [0.01, 0.02]}, index=idx),
    }
    replacement = {
        "regime_trend_score": pd.DataFrame({"AAA/USDC": [0.1, 0.2]}, index=idx),
    }

    merged = feature_generator._merge_missing_feature_dicts(cached, replacement)

    assert isinstance(merged["regime_trend_score"], pd.DataFrame)
    assert merged["regime_trend_score"].loc[idx[-1], "AAA/USDC"] == 0.2
    assert merged["ret24h"].loc[idx[-1], "AAA/USDC"] == 0.02


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


def test_live_feature_cache_key_ignores_hourly_refresh_token():
    base = {
        "run_id": "run_a",
        "symbols": ["AAA/USDC"],
        "required_feature_keys": {"ret24h"},
        "lookback_hours": 24 * 60,
    }

    key_a = feature_generator._live_feature_cache_key(
        **base,
        cfg={"live_feature_cache_raw_refresh_token": "2026-01-01T00:00:00Z:10"},
    )
    key_b = feature_generator._live_feature_cache_key(
        **base,
        cfg={"live_feature_cache_raw_refresh_token": "2026-01-01T01:00:00Z:10"},
    )

    assert key_a == key_b


def test_live_feature_cache_key_ignores_cycle_cache_controls():
    base = {
        "run_id": "run_a",
        "symbols": ["AAA/USDC"],
        "required_feature_keys": {"ret24h"},
        "lookback_hours": 24 * 60,
    }

    key_a = feature_generator._live_feature_cache_key(
        **base,
        cfg={
            "train_min_range_pct": 0.06,
            "live_feature_cycle_cache_bypass": False,
            "live_feature_return_latest_only": True,
            "live_feature_rolling_cache_seed_hours": 24 * 14,
        },
    )
    key_b = feature_generator._live_feature_cache_key(
        **base,
        cfg={
            "train_min_range_pct": 0.06,
            "live_feature_cycle_cache_bypass": True,
            "live_feature_return_latest_only": False,
            "live_feature_rolling_cache_seed_hours": 24 * 7,
        },
    )

    assert key_a == key_b


def test_live_feature_cache_key_splits_mask_and_model_namespaces():
    base = {
        "run_id": "run_a",
        "symbols": ["AAA/USDC"],
        "required_feature_keys": {"ret24h"},
        "lookback_hours": 24 * 60,
    }

    key_mask = feature_generator._live_feature_cache_key(
        **base,
        cfg={"live_feature_cache_namespace": "mask"},
    )
    key_model = feature_generator._live_feature_cache_key(
        **base,
        cfg={"live_feature_cache_namespace": "model"},
    )

    assert key_mask != key_model


def test_prediction_ledger_row_persists_live_execution_fee_diagnostics():
    row = run_inference._prediction_ledger_row(
        {
            "symbol": "BTC/USD:USD",
            "strategy_id": "long_demo",
            "raw_score": 0.72,
            "rank_threshold": 0.65,
            "effective_threshold": 0.65,
            "chain_results": {
                "normalized_rank_score": 0.88,
                "meta_head_hash": "head123",
            },
        },
        timestamp="2026-05-30T12:00:00Z",
        side="long",
        portfolio_decision="traded",
        execution_snapshot={"expected_fill_price": 100.0},
        was_traded=True,
        trade_result={
            "realized_entry_price": 101.0,
            "entry_notional_quote": 250.0,
            "base_amount": 2.475,
            "entry_fee_quote": 0.5,
            "entry_fee_cost": 0.5,
            "entry_fee_currency": "USD",
            "entry_fee_source": "order_fee",
            "entry_delay_adverse_bps": 100.0,
            "decision_to_entry_seconds": 12.5,
            "signal_to_entry_seconds": 612.5,
        },
    )

    assert row["was_traded"] is True
    assert row["realized_entry_price"] == 101.0
    assert row["entry_fee_quote"] == 0.5
    assert row["entry_fee_source"] == "order_fee"
    assert row["entry_fee_bps"] == pytest.approx(20.0)
    assert row["realized_fee_bps"] == pytest.approx(20.0)
    assert row["decision_to_entry_seconds"] == 12.5
    assert row["signal_to_entry_seconds"] == 612.5


def test_prediction_ledger_row_persists_selected_model_feature_snapshot():
    row = run_inference._prediction_ledger_row(
        {
            "symbol": "BTC/USD:USD",
            "strategy_id": "long_demo",
            "raw_score": 0.72,
            "rank_threshold": 0.65,
            "effective_threshold": 0.65,
            "chain_results": {
                "normalized_rank_score": 0.88,
                "meta_head_hash": "head123",
                "model_feature_audit_schema": "selected_model_features_v1",
                "model_feature_snapshot_hash": "abc123",
                "base_model_key": "long_demo_base",
                "meta_model_feature_key": "long_demo_meta",
                "base_model_feature_count": 2,
                "meta_model_feature_count": 2,
                "base_model_features_json": json.dumps(["ret24h", "volume_z"]),
                "meta_model_features_json": json.dumps(["long_demo", "ret24h"]),
                "base_model_feature_values_json": json.dumps(
                    {"ret24h": 0.04, "volume_z": 1.5}
                ),
                "meta_model_feature_values_json": json.dumps(
                    {"long_demo": 0.61, "ret24h": 0.04}
                ),
                "model_feature_value_sources_json": json.dumps(
                    {"ret24h": "candidate_features", "long_demo": "base_pred"}
                ),
                "model_feature_missing_json": json.dumps({"base": [], "meta": []}),
            },
        },
        timestamp="2026-05-30T12:00:00Z",
        side="long",
        portfolio_decision="accepted",
    )

    assert row["model_feature_audit_schema"] == "selected_model_features_v1"
    assert row["model_feature_snapshot_hash"] == "abc123"
    assert row["base_model_feature_count"] == 2
    assert row["meta_model_feature_count"] == 2
    assert json.loads(row["base_model_features_json"]) == ["ret24h", "volume_z"]
    assert json.loads(row["meta_model_feature_values_json"]) == {
        "long_demo": 0.61,
        "ret24h": 0.04,
    }


def test_model_feature_ledger_snapshot_uses_selected_base_and_meta_contracts_only():
    class DummyModel:
        def __init__(self, selected_features):
            self.selected_features = selected_features

    class DummyOrchestrator:
        alpha_by_strategy = {
            "long_demo": {
                "model": DummyModel(["ret24h", "volume_z"]),
                "feat_cols": ["ret24h", "volume_z", "unused_feature"],
            }
        }
        meta_models = {
            "long_demo": DummyModel(["long_demo", "ret24h"]),
        }

    snapshot = run_inference._model_feature_ledger_snapshot_for_decision(
        orchestrator=DummyOrchestrator(),
        side="long",
        strategy_id="long_demo",
        symbol="BTC/USD:USD",
        candidate_features=pd.DataFrame(
            {
                "ret24h": [0.04],
                "volume_z": [1.5],
                "unused_feature": [999.0],
            },
            index=["BTC/USD:USD"],
        ),
        feats={},
        chain_results={"base_pred": 0.61},
        signal_bar_ts="2026-05-30T12:00:00Z",
    )

    assert snapshot["base_model_feature_count"] == 2
    assert snapshot["meta_model_feature_count"] == 2
    assert json.loads(snapshot["base_model_features_json"]) == ["ret24h", "volume_z"]
    assert json.loads(snapshot["meta_model_features_json"]) == ["long_demo", "ret24h"]
    assert "unused_feature" not in snapshot["base_model_feature_values_json"]
    assert json.loads(snapshot["meta_model_feature_values_json"])["long_demo"] == 0.61


def test_prediction_ledger_path_supports_run_scoped_override(monkeypatch, tmp_path):
    monkeypatch.delenv("EPM_PREDICTION_LEDGER_PATH", raising=False)
    monkeypatch.delenv("EPM_RUN_SCOPED_PREDICTION_LEDGER", raising=False)

    default_path = run_inference._resolve_prediction_ledger_path(
        live_data_root=tmp_path,
        run_id="run_a",
    )
    assert default_path == tmp_path / "live_state" / "prediction_ledger.parquet"

    scoped_path = run_inference._resolve_prediction_ledger_path(
        live_data_root=tmp_path,
        run_id="run_a",
        run_scoped=True,
    )
    assert scoped_path == (
        tmp_path / "live_state" / "prediction_ledgers" / "run_a" / "prediction_ledger.parquet"
    )

    monkeypatch.setenv("EPM_RUN_SCOPED_PREDICTION_LEDGER", "1")
    env_scoped_path = run_inference._resolve_prediction_ledger_path(
        live_data_root=tmp_path,
        run_id="run_b",
    )
    assert env_scoped_path == (
        tmp_path / "live_state" / "prediction_ledgers" / "run_b" / "prediction_ledger.parquet"
    )

    explicit = tmp_path / "custom" / "ledger.parquet"
    monkeypatch.setenv("EPM_PREDICTION_LEDGER_PATH", str(explicit))
    assert (
        run_inference._resolve_prediction_ledger_path(
            live_data_root=tmp_path,
            run_id="run_c",
            run_scoped=True,
        )
        == explicit
    )


def test_offline_feature_lookup_accepts_artifact_source_run_id(monkeypatch):
    monkeypatch.setenv("EPM_ARTIFACT_SOURCE_RUN_ID", "20260523_015947")

    assert (
        feature_generator._offline_feature_lookup_run_id({}, "20260525_010004_nopenalty")
        == "20260523_015947"
    )


def test_offline_feature_lookup_roots_fall_back_from_exchange_scope(monkeypatch):
    monkeypatch.setenv("EPM_DATA_ROOT", "data_perp")

    assert feature_generator._offline_feature_lookup_data_roots(
        "data_perp/exchanges/krakenfutures"
    ) == [
        "data_perp/exchanges/krakenfutures",
        "data_perp",
    ]


def test_slice_feature_window_preserves_lazy_feature_dict():
    class LazyLike:
        def __init__(self):
            self._raw = {"ret24h": object()}
            self._symbol_indices = {"AAA/USD:USD": pd.date_range("2026-01-01", periods=2)}

        def __bool__(self):
            return True

    lazy = LazyLike()

    assert feature_generator._slice_feature_window(lazy) is lazy
    assert (
        feature_generator._cached_feature_coverage_end_ts(lazy)
        == pd.Timestamp("2026-01-02", tz="UTC")
    )


def test_cached_feature_merge_preserves_lazy_payloads():
    idx = pd.date_range("2026-05-01", periods=2, freq="1h", tz="UTC")
    lazy = LazyFeatureDict(
        {"ret4h": {"AAA/USD:USD": np.array([1.0, 2.0], dtype=np.float32)}},
        symbol_indices={"AAA/USD:USD": idx},
    )
    selector = {
        "range_12h_pct": pd.DataFrame(
            {"AAA/USD:USD": [0.1, 0.2]},
            index=idx,
        )
    }

    merged = feature_generator._merge_missing_feature_dicts(lazy, selector)

    assert isinstance(merged, LazyFeatureDict)
    assert "ret4h" in merged._raw
    assert "ret4h" not in merged._assembled
    assert "range_12h_pct" in merged._assembled


def test_get_features_for_candidates_reads_lazy_values_without_wide_assembly():
    idx = pd.date_range("2026-05-01", periods=2, freq="1h", tz="UTC")
    lazy = LazyFeatureDict(
        {
            "ret4h": {
                "AAA/USD:USD": np.array([1.0, 2.0], dtype=np.float32),
                "BBB/USD:USD": np.array([3.0, 4.0], dtype=np.float32),
            }
        },
        symbol_indices={"AAA/USD:USD": idx, "BBB/USD:USD": idx},
    )

    matrix = feature_generator.get_features_for_candidates(
        lazy,
        ["AAA/USD:USD", "BBB/USD:USD"],
        ts=idx[-1],
    )

    assert matrix.loc["AAA/USD:USD", "ret4h"] == pytest.approx(2.0)
    assert matrix.loc["BBB/USD:USD", "ret4h"] == pytest.approx(4.0)
    assert "ret4h" in lazy._raw
    assert "ret4h" not in lazy._assembled


def test_latest_only_features_reads_lazy_values_without_wide_assembly():
    idx = pd.date_range("2026-05-01", periods=2, freq="1h", tz="UTC")
    lazy = LazyFeatureDict(
        {
            "ret4h": {
                "AAA/USD:USD": np.array([1.0, 2.0], dtype=np.float32),
                "BBB/USD:USD": np.array([3.0, 4.0], dtype=np.float32),
            },
            "ret12h": {
                "AAA/USD:USD": np.array([5.0, 6.0], dtype=np.float32),
            },
        },
        symbol_indices={"AAA/USD:USD": idx, "BBB/USD:USD": idx},
    )

    latest = run_inference._latest_only_features(
        lazy,
        latest_ts=idx[-1],
        symbols=["AAA/USD:USD", "BBB/USD:USD"],
    )

    assert latest["ret4h"].loc[idx[-1], "AAA/USD:USD"] == pytest.approx(2.0)
    assert latest["ret4h"].loc[idx[-1], "BBB/USD:USD"] == pytest.approx(4.0)
    assert latest["ret12h"].loc[idx[-1], "AAA/USD:USD"] == pytest.approx(6.0)
    assert "BBB/USD:USD" not in latest["ret12h"].columns
    assert set(lazy._raw) == {"ret4h", "ret12h"}
    assert lazy._assembled == {}


def test_required_feature_validation_accepts_lazy_feature_dict_without_assembly():
    idx = pd.date_range("2026-05-01", periods=2, freq="1h", tz="UTC")
    lazy = LazyFeatureDict(
        {
            "ret4h": {
                "AAA/USD:USD": np.array([1.0, 2.0], dtype=np.float32),
                "BBB/USD:USD": np.array([3.0, 4.0], dtype=np.float32),
            }
        },
        symbol_indices={"AAA/USD:USD": idx, "BBB/USD:USD": idx},
    )

    assert run_inference.validate_required_feature_frames(
        lazy,
        {"ret4h"},
        symbols=["AAA/USD:USD", "BBB/USD:USD"],
        strict=True,
    )
    assert "ret4h" in lazy._raw
    assert lazy._assembled == {}


def test_symbols_with_required_feature_coverage_filters_sparse_lazy_candidates():
    idx = pd.date_range("2026-05-01", periods=2, freq="1h", tz="UTC")
    lazy = LazyFeatureDict(
        {
            "ret4h": {
                "AAA/USD:USD": np.array([1.0, 2.0], dtype=np.float32),
                "BBB/USD:USD": np.array([3.0, 4.0], dtype=np.float32),
            },
            "unwind_score": {
                "AAA/USD:USD": np.array([0.1, 0.2], dtype=np.float32),
            },
        },
        symbol_indices={"AAA/USD:USD": idx, "BBB/USD:USD": idx},
    )

    allowed, missing = run_inference._symbols_with_required_feature_coverage(
        lazy,
        {"ret4h", "unwind_score"},
        ["AAA/USD:USD", "BBB/USD:USD"],
    )

    assert allowed == ["AAA/USD:USD"]
    assert missing == {"BBB/USD:USD": ["unwind_score"]}
    assert lazy._assembled == {}


def test_live_state_current_does_not_mark_raw_features_transformed(
    monkeypatch,
    tmp_path,
):
    idx = pd.date_range("2026-01-01", periods=16, freq="h", tz="UTC")
    raw = pd.DataFrame(
        {"AAA/USD:USD": np.linspace(0.0, 100.0, len(idx), dtype=np.float32)},
        index=idx,
    )

    class CurrentState:
        last_timestamp = idx[-1].isoformat()

    from extreme_price_movements.inference import live_zscore_state

    monkeypatch.setattr(
        live_zscore_state.RollingZScoreState,
        "load",
        staticmethod(lambda *args, **kwargs: CurrentState()),
    )

    out = features._apply_causal_transform_live_state_or_batch(
        {"raw_magnitude_feature": raw.copy()},
        {
            "live_causal_transform_state_enabled": True,
            "live_causal_transform_state_path": str(tmp_path / "state.npz"),
            "feature_transform_cache_enabled": False,
            "transform_chunk_size": 8,
        },
        feature_index=idx,
        feature_columns=["AAA/USD:USD"],
        skip_transform_set=set(),
    )

    transformed = out["raw_magnitude_feature"]
    assert transformed.shape == raw.shape
    assert not transformed.equals(raw)
    assert np.nanmax(np.abs(transformed.to_numpy(dtype=np.float64))) < 100.0


def test_live_feature_source_run_id_override_is_used(monkeypatch):
    monkeypatch.setenv("EPM_LIVE_FEATURE_SOURCE_RUN_ID", "20260321_140000")

    run_id = feature_generator._offline_feature_lookup_run_id({}, "current_run")

    assert run_id == "20260321_140000"


def test_run_inference_feature_source_prefers_artifact_source(monkeypatch):
    monkeypatch.delenv("EPM_LIVE_FEATURE_SOURCE_RUN_ID", raising=False)
    monkeypatch.delenv("EPM_FEATURE_SOURCE_RUN_ID", raising=False)
    monkeypatch.setenv("EPM_ARTIFACT_SOURCE_RUN_ID", "20260523_015947")

    run_id = run_inference._resolve_live_feature_source_run_id(
        {"run_id": "20260525_010004_nopenalty"}
    )

    assert run_id == "20260523_015947"


def test_live_feature_restart_uses_model_superset_rolling_cache_before_offline(
    monkeypatch,
    tmp_path,
):
    symbols = ["AAA/USD:USD", "BBB/USD:USD"]
    idx = pd.date_range("2026-06-01 00:00", periods=30, freq="1h", tz="UTC")
    close = pd.DataFrame(
        {
            "AAA/USD:USD": np.linspace(100.0, 110.0, len(idx)),
            "BBB/USD:USD": np.linspace(50.0, 55.0, len(idx)),
        },
        index=idx,
    )
    panel = {
        "close": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": pd.DataFrame(1000.0, index=idx, columns=symbols),
    }
    required = {"ret24h"}
    model_required = {"ret24h", "range_24h_pct"}
    cfg = {
        "live_feature_snapshot_cache_enabled": False,
        "live_feature_rolling_cache_enabled": True,
        "live_feature_snapshot_cache_dir": str(tmp_path / "live_feature_cache"),
        "live_feature_cache_namespace": "model",
        "live_feature_offline_cache_enabled": True,
        "live_feature_return_latest_only": True,
    }
    cache_key = feature_generator._live_feature_cache_key(
        run_id="run_a",
        symbols=symbols,
        required_feature_keys=model_required,
        lookback_hours=24 * 30,
        cfg=cfg,
        data_root=str(tmp_path / "exchange_root"),
    )
    ret24h = close / close.shift(24) - 1.0
    feature_generator._write_live_feature_rolling_cache(
        cfg=cfg,
        run_id="run_a",
        cache_key=cache_key,
        feats={
            "ret24h": ret24h.astype(np.float32),
            "range_24h_pct": (ret24h.abs() + 0.01).astype(np.float32),
        },
        symbols=symbols,
        end_ts=idx[-1],
        required_feature_keys=model_required,
        append_after_ts=idx[-3],
        keep_start_ts=idx[0],
    )

    def fail_offline(*args, **kwargs):
        raise AssertionError("offline selected-feature cache should not be touched")

    def fail_tail(*args, **kwargs):
        raise AssertionError("tail feature recompute should not run")

    monkeypatch.setattr(feature_generator, "load_cached_features_for_inference", fail_offline)
    monkeypatch.setattr(feature_generator, "compute_features_hourly", fail_tail)

    feats = feature_generator.load_or_compute_features(
        panel=panel,
        basket_syms=symbols,
        run_id="run_a",
        data_root=str(tmp_path / "exchange_root"),
        cfg={**cfg, "live_feature_cache_namespace": "mask"},
        lookback_hours=24 * 30,
        required_feature_keys=required,
    )

    assert set(feats) >= {"ret24h"}
    latest = feats["ret24h"]
    assert list(latest.columns) == symbols
    assert latest.index[-1] == idx[-1]


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
