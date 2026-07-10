import json
import os

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements import features
from extreme_price_movements.data_store import LazyFeatureDict, write_live_latest_feature_matrix
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
        captured["required_feature_keys"] = set(kwargs["required_feature_keys"])
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
    assert "barrier_pct" in captured["required_feature_keys"]


def test_strategy_mask_candidates_require_finite_deployed_contract():
    idx = pd.date_range("2026-06-03 09:00", periods=1, freq="1h", tz="UTC")
    feats = {
        "good_feature": pd.DataFrame(
            {"AAA/USD:USD": [1.0], "BBB/USD:USD": [2.0]},
            index=idx,
        ),
        "sparse_feature": pd.DataFrame(
            {"AAA/USD:USD": [np.nan], "BBB/USD:USD": [3.0]},
            index=idx,
        ),
    }

    filtered, diagnostics = run_inference._filter_strategy_masks_by_finite_model_contract(
        feats,
        {
            "long_sparse": ["AAA/USD:USD", "BBB/USD:USD"],
            "short_good": ["AAA/USD:USD"],
        },
        {
            "long_sparse": ["good_feature", "sparse_feature"],
            "short_good": ["good_feature"],
        },
        latest_ts=idx[-1],
    )

    assert filtered["long_sparse"] == ["BBB/USD:USD"]
    assert filtered["short_good"] == ["AAA/USD:USD"]
    assert diagnostics["long_sparse"]["rejected"] == 1
    assert diagnostics["long_sparse"]["top_nonfinite_features"][0]["feature"] == "sparse_feature"


def test_stale_model_feature_detail_does_not_materialize_lazy_cache():
    idx = pd.date_range("2026-06-03 08:00", periods=2, freq="1h", tz="UTC")
    lazy = LazyFeatureDict(
        {
            "model_feature": {
                "AAA/USD:USD": (idx, np.array([1.0, 2.0], dtype=np.float32)),
            }
        }
    )

    detail = feature_generator._cached_feature_stale_detail(
        lazy,
        {"model_feature"},
        pd.Timestamp("2026-06-03 10:00", tz="UTC"),
        coverage_symbols=["AAA/USD:USD"],
    )

    assert detail == ["model_feature=2026-06-03 09:00:00+00:00"]
    assert lazy._assembled == {}
    assert "model_feature" in lazy._raw


def test_latest_matrix_support_ignores_live_synthesized_sidecar_keys():
    matrix = pd.DataFrame(
        {
            "core_feature": [1.0, 2.0, 3.0],
        },
        index=["AAA/USD:USD", "BBB/USD:USD", "CCC/USD:USD"],
    )

    sidecar_keys = feature_generator._sidecar_backed_feature_keys(
        {
            "core_feature",
            "barrier_pct",
            "ret1h_G_VOL_0",
            "ret1h_G_VOL_1",
        }
    )

    assert sidecar_keys == {"core_feature"}
    assert (
        feature_generator._latest_matrix_low_finite_support(
            matrix,
            required_feature_keys=sidecar_keys,
            min_fraction=0.8,
        )
        == []
    )
    assert feature_generator._latest_matrix_low_finite_support(
        matrix,
        required_feature_keys={"core_feature", "missing_core"},
        min_fraction=0.8,
    )[0]["feature"] == "missing_core"


def test_prewarm_compacts_full_selected_matrix_when_global_sidecar_partial(
    tmp_path, monkeypatch
):
    run_id = "20260101_000000"
    end_ts = pd.Timestamp("2026-01-01 03:00:00", tz="UTC")
    feature_dir = tmp_path / "features" / run_id
    feature_dir.mkdir(parents=True)
    idx = pd.date_range("2026-01-01 00:00:00", periods=4, freq="1h", tz="UTC")
    for symbol, offset in [("AAA/USD:USD", 0.0), ("BBB/USD:USD", 10.0)]:
        frame = pd.DataFrame(
            {
                "core_a": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
                + offset,
                "core_b": np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
                + offset,
                "__symbol__": symbol,
            },
            index=idx,
        )
        frame.to_parquet(feature_dir / f"symbol={symbol.replace('/', '_')}.parquet")

    write_live_latest_feature_matrix(
        {
            "core_a": pd.DataFrame(
                {
                    "AAA/USD:USD": [4.0],
                    "BBB/USD:USD": [14.0],
                },
                index=pd.DatetimeIndex([end_ts]),
                dtype=np.float32,
            )
        },
        pd.to_datetime(run_id, format="%Y%m%d_%H%M%S", utc=True),
        str(tmp_path),
        end_ts=end_ts,
        symbols=["AAA/USD:USD", "BBB/USD:USD"],
        merge_existing=False,
    )

    def fail_sync(**kwargs):
        raise AssertionError("feature sync should not run when selected files are complete")

    monkeypatch.setattr(
        feature_generator,
        "_run_training_path_feature_sync_for_live",
        fail_sync,
    )

    result = feature_generator.prewarm_selected_model_feature_cache_for_live(
        run_id=run_id,
        data_root=str(tmp_path),
        symbols=["AAA/USD:USD", "BBB/USD:USD"],
        end_ts=end_ts,
        cfg={},
        required_feature_keys={"core_a", "core_b"},
        source_run_ids=[run_id],
    )

    assert result["status"] == "selected_matrix_cache_ready"
    assert result["matrix_complete"] is True
    loaded = feature_generator._load_selected_feature_latest_matrix_cache(
        cache_root=str(tmp_path),
        source_run_id=run_id,
        source_root=str(tmp_path),
        symbols=["AAA/USD:USD", "BBB/USD:USD"],
        feature_keys={"core_a", "core_b"},
        end_ts=end_ts,
    )
    assert set(loaded) == {"core_a", "core_b"}
    assert loaded["core_b"].loc[end_ts, "BBB/USD:USD"] == np.float32(18.0)


def test_selected_latest_matrix_rejects_core_low_finite_for_source_fallback(
    tmp_path, monkeypatch
):
    run_id = "20260101_000000"
    end_ts = pd.Timestamp("2026-01-01 03:00:00", tz="UTC")
    symbols = ["AAA/USD:USD", "BBB/USD:USD", "CCC/USD:USD", "DDD/USD:USD"]
    monkeypatch.setenv("EPM_SELECTED_FEATURE_LATEST_MATRIX_MIN_FINITE_FRACTION", "0.80")

    feats = {
        "core_a": pd.DataFrame(
            {symbol: [1.0] for symbol in symbols},
            index=pd.DatetimeIndex([end_ts]),
            dtype=np.float32,
        ),
        "core_b": pd.DataFrame(
            {
                "AAA/USD:USD": [2.0],
                "BBB/USD:USD": [3.0],
                "CCC/USD:USD": [np.nan],
                "DDD/USD:USD": [np.nan],
            },
            index=pd.DatetimeIndex([end_ts]),
            dtype=np.float32,
        ),
    }
    feature_generator._write_selected_feature_latest_matrix_cache(
        cache_root=str(tmp_path),
        source_run_id=run_id,
        source_root=str(tmp_path),
        symbols=symbols,
        feature_keys={"core_a", "core_b"},
        end_ts=end_ts,
        feats=feats,
    )

    loaded = feature_generator._load_selected_feature_latest_matrix_cache(
        cache_root=str(tmp_path),
        source_run_id=run_id,
        source_root=str(tmp_path),
        symbols=symbols,
        feature_keys={"core_a", "core_b"},
        end_ts=end_ts,
    )

    assert loaded == {}


def test_selected_latest_matrix_loads_history_low_finite_for_row_strict_scoring(
    tmp_path, monkeypatch
):
    run_id = "20260101_000000"
    end_ts = pd.Timestamp("2026-01-01 03:00:00", tz="UTC")
    symbols = ["AAA/USD:USD", "BBB/USD:USD", "CCC/USD:USD", "DDD/USD:USD"]
    monkeypatch.setenv("EPM_SELECTED_FEATURE_LATEST_MATRIX_MIN_FINITE_FRACTION", "0.80")

    feats = {
        "lr_24h": pd.DataFrame(
            {
                "AAA/USD:USD": [2.0],
                "BBB/USD:USD": [3.0],
                "CCC/USD:USD": [np.nan],
                "DDD/USD:USD": [np.nan],
            },
            index=pd.DatetimeIndex([end_ts]),
            dtype=np.float32,
        ),
    }
    feature_generator._write_selected_feature_latest_matrix_cache(
        cache_root=str(tmp_path),
        source_run_id=run_id,
        source_root=str(tmp_path),
        symbols=symbols,
        feature_keys={"lr_24h"},
        end_ts=end_ts,
        feats=feats,
    )

    loaded = feature_generator._load_selected_feature_latest_matrix_cache(
        cache_root=str(tmp_path),
        source_run_id=run_id,
        source_root=str(tmp_path),
        symbols=symbols,
        feature_keys={"lr_24h"},
        end_ts=end_ts,
    )

    assert set(loaded) == {"lr_24h"}
    assert np.isfinite(loaded["lr_24h"].loc[end_ts, "AAA/USD:USD"])
    assert np.isnan(float(loaded["lr_24h"].loc[end_ts, "DDD/USD:USD"]))


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
    import math
    assert math.isclose(float(matrix.loc["AAA/USDC", "ret24h"]), 0.03, rel_tol=1e-5)
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
        cfg={"live_materialize_orderbook_model_features": True},
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
        cfg={
            "live_materialize_orderbook_model_features": True,
            "market_basket": ["BTC/USDT", "ETH/USDT"],
        },
    )

    assert required.issubset(materialized)
    for key in sorted(required):
        tail = materialized[key].iloc[-1].to_numpy(dtype=np.float32)
        assert np.isfinite(tail).any(), key


def test_live_orderbook_residual_preserves_populated_cached_feature_in_parity_mode():
    idx = pd.date_range("2026-05-01", periods=32, freq="1h", tz="UTC")
    symbols = ["BTC/USD:USD", "ETH/USD:USD", "AAA/USD:USD"]
    close = pd.DataFrame(
        {
            sym: 100.0 + i + np.arange(len(idx), dtype=np.float32) * 0.1
            for i, sym in enumerate(symbols)
        },
        index=idx,
        dtype=np.float32,
    )
    volume = pd.DataFrame(
        {sym: 1000.0 + i for i, sym in enumerate(symbols)},
        index=idx,
        dtype=np.float32,
    )
    cached = pd.DataFrame(
        {"AAA/USD:USD": np.linspace(-0.2, 0.3, len(idx), dtype=np.float32)},
        index=idx,
        dtype=np.float32,
    )
    panel = {
        "close": close,
        "volume": volume,
        "orderbook_best_bid": (close * 0.999).astype(np.float32),
        "orderbook_best_ask": (close * 1.001).astype(np.float32),
        "orderbook_mid": close.astype(np.float32),
        "orderbook_bid_qty_1": pd.DataFrame(10.0, index=idx, columns=symbols),
        "orderbook_ask_qty_1": pd.DataFrame(9.0, index=idx, columns=symbols),
        "orderbook_cum_bid_qty_l10": pd.DataFrame(80.0, index=idx, columns=symbols),
        "orderbook_cum_ask_qty_l10": pd.DataFrame(75.0, index=idx, columns=symbols),
        "orderbook_cum_bid_qty_l20": pd.DataFrame(140.0, index=idx, columns=symbols),
        "orderbook_cum_ask_qty_l20": pd.DataFrame(130.0, index=idx, columns=symbols),
        "orderbook_buy_notional_1h": (close * volume * 0.55).astype(np.float32),
        "orderbook_sell_notional_1h": (close * volume * 0.45).astype(np.float32),
    }

    materialized = feature_generator._materialize_live_orderbook_summary_features(
        {"ob_imbalance_mkt_resid": cached},
        panel,
        symbols,
        {"ob_imbalance_mkt_resid"},
        cfg={
            "live_materialize_orderbook_model_features": True,
            "historical_inference_parity_preserve_cached_features": True,
            "market_basket": ["BTC/USD:USD", "ETH/USD:USD"],
        },
    )

    pd.testing.assert_series_equal(
        materialized["ob_imbalance_mkt_resid"]["AAA/USD:USD"],
        cached["AAA/USD:USD"],
        check_names=False,
    )


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
    import math
    assert math.isclose(float(materialized["regime_trend_score"].loc[idx[-1], "AAA/USDC"]), 0.3, rel_tol=1e-5)
    assert math.isclose(float(materialized["regime_trend_score"].loc[idx[-1], "BBB/USDC"]), 0.3, rel_tol=1e-5)
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
    import math
    assert math.isclose(float(materialized["regime_trend_score"].loc[idx[-1], "AAA/USDC"]), 0.3, rel_tol=1e-5)
    assert math.isclose(float(materialized["regime_trend_score"].loc[idx[-1], "BBB/USDC"]), 0.3, rel_tol=1e-5)
    assert math.isclose(float(materialized["regime_vol_score"].loc[idx[-1], "AAA/USDC"]), 0.8, rel_tol=1e-5)
    assert math.isclose(float(materialized["regime_vol_score"].loc[idx[-1], "BBB/USDC"]), 0.8, rel_tol=1e-5)
    assert math.isclose(float(materialized["mkt_rv_ratio"].loc[idx[-1], "AAA/USDC"]), 1.2, rel_tol=1e-5)
    assert math.isclose(float(materialized["mkt_rv_ratio"].loc[idx[-1], "BBB/USDC"]), 1.2, rel_tol=1e-5)


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


def test_live_feature_cache_key_ignores_selected_sync_controls():
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
            "live_feature_cache_namespace": "model",
            "live_model_feature_auto_sync_selected_cache": True,
            "live_model_feature_auto_sync_blocking": False,
            "live_model_feature_auto_sync_timeout_seconds": 1200,
        },
    )
    key_b = feature_generator._live_feature_cache_key(
        **base,
        cfg={
            "train_min_range_pct": 0.06,
            "live_feature_cache_namespace": "model",
            "live_model_feature_auto_sync_selected_cache": False,
            "live_model_feature_auto_sync_blocking": True,
            "live_model_feature_auto_sync_timeout_seconds": 5,
        },
    )

    assert key_a == key_b


def test_live_feature_cache_key_ignores_kraken_perp_ohlcv_gap_repair_controls():
    base = {
        "run_id": "run_a",
        "symbols": ["AAA/USD:USD"],
        "required_feature_keys": {"price_trend_10d_vol_norm"},
        "lookback_hours": 24 * 60,
    }

    key_a = feature_generator._live_feature_cache_key(
        **base,
        cfg={
            "market_mode": "perps",
            "exchange": "kraken",
            "live_model_feature_kraken_perp_ohlcv_gap_backfill": True,
            "live_model_feature_kraken_perp_ohlcv_gap_backfill_lookback_days": 21,
            "live_model_feature_kraken_perp_ohlcv_gap_backfill_max_gap_hours": 720,
        },
    )
    key_b = feature_generator._live_feature_cache_key(
        **base,
        cfg={
            "market_mode": "perps",
            "exchange": "kraken",
            "live_model_feature_kraken_perp_ohlcv_gap_backfill": False,
            "live_model_feature_kraken_perp_ohlcv_gap_backfill_lookback_days": 7,
            "live_model_feature_kraken_perp_ohlcv_gap_backfill_max_gap_hours": 48,
        },
    )

    assert key_a == key_b


def test_selected_feature_sync_can_launch_out_of_band(tmp_path, monkeypatch):
    launched = {}

    class FakeProc:
        pid = 12345

    def fake_popen(cmd, **kwargs):
        launched["cmd"] = list(cmd)
        launched["kwargs"] = dict(kwargs)
        return FakeProc()

    monkeypatch.setattr(feature_generator.subprocess, "Popen", fake_popen)

    ok = feature_generator._run_training_path_feature_sync_for_live(
        run_id="feature_run",
        data_root=str(tmp_path),
        end_ts=pd.Timestamp("2026-06-10T09:00:00Z"),
        cfg={"exchange": "kraken", "market_mode": "perps"},
        required_feature_keys={"ret24h"},
        symbols=["BBB/USD:USD", "AAA/USD:USD"],
        blocking=False,
        sync_label="selected_missing_contract",
    )

    assert ok is True
    assert "--perps" in launched["cmd"]
    assert launched["cmd"][-4:] == ["--exchange", "kraken", "--run-id", "feature_run"]
    assert launched["kwargs"]["env"]["EPM_FEATURE_SYMBOLS"] == (
        "AAA/USD:USD,BBB/USD:USD"
    )
    assert launched["kwargs"]["env"]["EPM_FEATURE_LIVE_DECISION_TAIL_ONLY"] == "1"
    assert launched["kwargs"]["start_new_session"] is True
    state_dir = tmp_path / "artifacts" / "feature_run" / "live_state"
    assert (state_dir / "feature_selected_missing_contract_sync.pid").read_text() == "12345"
    meta = json.loads(
        (state_dir / "feature_selected_missing_contract_sync.json").read_text()
    )
    assert meta["requested_keys"] == 1
    assert meta["requested_symbols"] == 2
    assert meta["end_ts"] == "2026-06-10T09:00:00+00:00"


def test_selected_feature_sync_large_repair_uses_bounded_symbol_chunk(tmp_path, monkeypatch):
    launched = {}

    class FakeProc:
        pid = 12345

    def fake_popen(cmd, **kwargs):
        launched["cmd"] = list(cmd)
        launched["kwargs"] = dict(kwargs)
        return FakeProc()

    monkeypatch.setattr(feature_generator.subprocess, "Popen", fake_popen)

    ok = feature_generator._run_training_path_feature_sync_for_live(
        run_id="feature_run",
        data_root=str(tmp_path),
        end_ts=pd.Timestamp("2026-06-10T09:00:00Z"),
        cfg={"exchange": "kraken", "market_mode": "perps"},
        required_feature_keys={f"feat_{i}" for i in range(150)},
        symbols=[f"S{i}/USD:USD" for i in range(247)],
        blocking=False,
        sync_label="selected_large_contract",
    )

    assert ok is True
    env = launched["kwargs"]["env"]
    assert env["EPM_FEATURE_BACKFILL_ALL_INCOMPLETE_KEYS"] == "0"
    assert env["EPM_FEATURE_LIVE_DECISION_TAIL_ONLY"] == "1"
    assert "EPM_FEATURE_BACKFILL_KEY_BATCH_SIZE" not in env
    assert env["EPM_FEATURE_BACKFILL_SYMBOL_CHUNK_SIZE"] == "64"
    assert len(env["EPM_FEATURE_SYMBOLS"].split(",")) == 247


def test_kraken_perp_selected_feature_sync_repairs_ohlcv_gaps_first(tmp_path, monkeypatch):
    data_root = tmp_path / "data_perp"
    perp_root = data_root / "exchanges" / "krakenfutures"
    (perp_root / "ohlcv").mkdir(parents=True)
    manifest = perp_root / "manifests" / "kraken_dual_market_verified_universe_latest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"symbols": ["AAA/USD:USD"]}))
    launched = {}

    class FakeProc:
        pid = 12345

    def fake_popen(cmd, **kwargs):
        launched["cmd"] = list(cmd)
        launched["kwargs"] = dict(kwargs)
        return FakeProc()

    monkeypatch.setattr(feature_generator.subprocess, "Popen", fake_popen)

    ok = feature_generator._run_training_path_feature_sync_for_live(
        run_id="feature_run",
        data_root=str(data_root),
        end_ts=pd.Timestamp("2026-06-19T18:00:00Z"),
        cfg={
            "exchange": "kraken",
            "market_mode": "perps",
            "live_model_feature_kraken_perp_ohlcv_gap_backfill": True,
            "live_model_feature_kraken_perp_ohlcv_gap_backfill_lookback_days": 21,
        },
        required_feature_keys={"price_trend_10d_vol_norm"},
        blocking=False,
        sync_label="selected_missing_contract",
    )

    assert ok is True
    assert launched["cmd"][:3] == [feature_generator.sys.executable, "-u", "-c"]
    chain = json.loads(
        launched["kwargs"]["env"]["EPM_LIVE_FEATURE_SYNC_COMMANDS_JSON"]
    )
    assert "scripts/backfill_kraken_missing_ohlcv_gaps.py" in chain[0]
    assert "--lookback-days" in chain[0]
    assert chain[0][chain[0].index("--lookback-days") + 1] == "21"
    assert chain[1][-4:] == ["--exchange", "kraken", "--run-id", "feature_run"]
    state_dir = data_root / "artifacts" / "feature_run" / "live_state"
    meta = json.loads(
        (state_dir / "feature_selected_missing_contract_sync.json").read_text()
    )
    assert meta["ohlcv_gap_backfill"] is True


def test_kraken_perp_selected_feature_sync_does_not_repair_ohlcv_by_default(tmp_path, monkeypatch):
    data_root = tmp_path / "data_perp"
    perp_root = data_root / "exchanges" / "krakenfutures"
    (perp_root / "ohlcv").mkdir(parents=True)
    manifest = perp_root / "manifests" / "kraken_dual_market_verified_universe_latest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"symbols": ["AAA/USD:USD"]}))
    launched = {}

    class FakeProc:
        pid = 12345

    def fake_popen(cmd, **kwargs):
        launched["cmd"] = list(cmd)
        launched["kwargs"] = dict(kwargs)
        return FakeProc()

    monkeypatch.setattr(feature_generator.subprocess, "Popen", fake_popen)

    ok = feature_generator._run_training_path_feature_sync_for_live(
        run_id="feature_run",
        data_root=str(data_root),
        end_ts=pd.Timestamp("2026-06-19T18:00:00Z"),
        cfg={"exchange": "kraken", "market_mode": "perps"},
        required_feature_keys={"price_trend_10d_vol_norm"},
        blocking=False,
        sync_label="selected_missing_contract",
    )

    assert ok is True
    assert "scripts/backfill_kraken_missing_ohlcv_gaps.py" not in launched["cmd"]
    assert launched["cmd"][-4:] == ["--exchange", "kraken", "--run-id", "feature_run"]
    state_dir = data_root / "artifacts" / "feature_run" / "live_state"
    meta = json.loads(
        (state_dir / "feature_selected_missing_contract_sync.json").read_text()
    )
    assert meta["ohlcv_gap_backfill"] is False


def test_selected_model_feature_store_gap_report_excludes_live_unavailable_labels():
    idx = pd.date_range("2026-06-19 18:00", periods=1, freq="1h", tz="UTC")
    symbols = [f"S{i}/USD:USD" for i in range(10)]
    feats = {
        "price_trend_10d_vol_norm": pd.DataFrame(
            [[1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]],
            columns=symbols,
            index=idx,
        ),
        "signed_prediction_error": pd.DataFrame(
            [[np.nan] * len(symbols)],
            columns=symbols,
            index=idx,
        ),
    }

    report = run_inference._selected_model_feature_store_gap_report(
        feats=feats,
        symbols=symbols,
        required_feature_keys={
            "price_trend_10d_vol_norm",
            "signed_prediction_error",
        },
        signal_bar_ts=idx[-1],
        min_finite_fraction=0.8,
    )

    assert report["ok"] is False
    assert report["reason"] == "feature_store_gap"
    assert report["min_finite"] == 8
    assert [row["feature"] for row in report["low_finite_features"]] == [
        "price_trend_10d_vol_norm"
    ]

    low_finite = feature_generator._latest_required_feature_low_finite_support(
        feats,
        symbols=symbols,
        required_feature_keys={
            "price_trend_10d_vol_norm",
            "signed_prediction_error",
        },
        end_ts=idx[-1],
        min_fraction=0.8,
    )
    assert [row["feature"] for row in low_finite] == [
        "price_trend_10d_vol_norm"
    ]


def test_selected_model_feature_store_gap_report_attributes_stale_cache():
    signal_ts = pd.Timestamp("2026-06-19 18:00", tz="UTC")
    stale_ts = signal_ts - pd.Timedelta(hours=1)
    symbols = [f"S{i}/USD:USD" for i in range(10)]
    feats = {
        "price_trend_10d_vol_norm": pd.DataFrame(
            [[1.0] + [np.nan] * 9],
            columns=symbols,
            index=pd.DatetimeIndex([stale_ts]),
        )
    }
    panel = {
        key: pd.DataFrame(
            [np.arange(10, dtype=np.float32) + 1.0],
            columns=symbols,
            index=pd.DatetimeIndex([signal_ts]),
        )
        for key in ("close", "high", "low")
    }

    report = run_inference._selected_model_feature_store_gap_report(
        feats=feats,
        panel=panel,
        symbols=symbols,
        required_feature_keys={"price_trend_10d_vol_norm"},
        signal_bar_ts=signal_ts,
        min_finite_fraction=0.8,
    )

    issue = report["low_finite_features"][0]
    assert issue["feature"] == "price_trend_10d_vol_norm"
    assert issue["source_attribution"] == "feature_cache_stale"
    assert issue["source_groups"] == ["ohlcv"]
    assert issue["feature_stale_hours"] == pytest.approx(1.0)
    assert {item["reason"] for item in issue["source_coverage"]} == {"ok"}


def test_selected_model_feature_store_gap_report_attributes_exchange_source_gap():
    signal_ts = pd.Timestamp("2026-06-19 18:00", tz="UTC")
    symbols = [f"S{i}/USD:USD" for i in range(10)]
    feats = {
        "oi_7d_x_funding_1d_chg": pd.DataFrame(
            [[1.0] + [np.nan] * 9],
            columns=symbols,
            index=pd.DatetimeIndex([signal_ts]),
        )
    }
    panel = {
        "open_interest": pd.DataFrame(
            [[np.nan] * len(symbols)],
            columns=symbols,
            index=pd.DatetimeIndex([signal_ts]),
        ),
        "funding_rate": pd.DataFrame(
            [[np.nan] * len(symbols)],
            columns=symbols,
            index=pd.DatetimeIndex([signal_ts]),
        ),
    }

    report = run_inference._selected_model_feature_store_gap_report(
        feats=feats,
        panel=panel,
        symbols=symbols,
        required_feature_keys={"oi_7d_x_funding_1d_chg"},
        signal_bar_ts=signal_ts,
        min_finite_fraction=0.8,
    )

    issue = report["low_finite_features"][0]
    assert issue["feature"] == "oi_7d_x_funding_1d_chg"
    assert issue["source_attribution"] == "exchange_source_data_lacking"
    assert set(issue["source_groups"]) == {"open_interest", "funding"}
    assert {item["reason"] for item in issue["source_coverage"]} == {
        "low_source_coverage"
    }


def test_selected_latest_cache_invalidation_resolves_descriptive_feature_run_id(
    tmp_path,
):
    data_root = tmp_path / "data_perp"
    source_run_id = "20260619_011500_no_mkt4_evband002_shadow"
    feature_dir = data_root / "features" / "20260619_011500"
    feature_dir.mkdir(parents=True)
    manifest = feature_dir / "_feature_cache_scan_manifest.json"
    manifest.write_text("{}")
    end_ts = pd.Timestamp("2026-06-19T19:00:00Z")
    symbols = ["AAA/USD:USD"]
    feature_keys = {"price_trend_10d_vol_norm"}
    feats = {
        "price_trend_10d_vol_norm": pd.DataFrame(
            [[1.0]],
            index=pd.DatetimeIndex([end_ts]),
            columns=symbols,
        )
    }

    feature_generator._write_selected_feature_latest_matrix_cache(
        cache_root=str(data_root),
        source_run_id=source_run_id,
        source_root=str(data_root),
        symbols=symbols,
        feature_keys=feature_keys,
        end_ts=end_ts,
        feats=feats,
    )
    cache_dir = feature_generator._selected_feature_latest_cache_dir(
        cache_root=str(data_root),
        source_run_id=source_run_id,
        source_root=str(data_root),
        symbols=symbols,
        feature_keys=feature_keys,
        end_ts=end_ts,
    )
    cache_mtime = (cache_dir / "latest.parquet").stat().st_mtime
    os.utime(manifest, (cache_mtime + 10.0, cache_mtime + 10.0))

    loaded = feature_generator._load_selected_feature_latest_matrix_cache(
        cache_root=str(data_root),
        source_run_id=source_run_id,
        source_root=str(data_root),
        symbols=symbols,
        feature_keys=feature_keys,
        end_ts=end_ts,
    )

    assert loaded == {}


def test_selected_latest_cache_low_finite_falls_back_to_live_sidecar(
    tmp_path,
    monkeypatch,
):
    data_root = tmp_path / "data_perp"
    source_run_id = "20260619_011500_no_mkt4_evband002_shadow"
    end_ts = pd.Timestamp("2026-06-19T19:00:00Z")
    source_ts = pd.Timestamp("2026-06-19T01:15:00Z")
    symbols = [f"S{i}/USD:USD" for i in range(10)]
    feature_keys = {"price_trend_10d_vol_norm"}
    sparse = [1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    stale_feats = {
        "price_trend_10d_vol_norm": pd.DataFrame(
            [sparse],
            index=pd.DatetimeIndex([end_ts]),
            columns=symbols,
        )
    }
    repaired_feats = {
        "price_trend_10d_vol_norm": pd.DataFrame(
            [np.arange(10, dtype=np.float32)],
            index=pd.DatetimeIndex([end_ts]),
            columns=symbols,
        )
    }
    monkeypatch.setenv(
        "EPM_SELECTED_FEATURE_LATEST_MATRIX_MIN_FINITE_FRACTION",
        "0.8",
    )
    feature_generator._write_selected_feature_latest_matrix_cache(
        cache_root=str(data_root),
        source_run_id=source_run_id,
        source_root=str(data_root),
        symbols=symbols,
        feature_keys=feature_keys,
        end_ts=end_ts,
        feats=stale_feats,
    )
    write_live_latest_feature_matrix(
        repaired_feats,
        source_ts,
        str(data_root),
        end_ts=end_ts,
        symbols=symbols,
        feature_keys=feature_keys,
    )

    loaded = feature_generator.load_cached_features_for_inference(
        source_run_id,
        str(data_root),
        symbols=symbols,
        feature_keys=feature_keys,
        start_ts=end_ts,
        end_ts=end_ts,
    )

    values = loaded["price_trend_10d_vol_norm"].loc[end_ts, symbols]
    assert int(np.isfinite(values.to_numpy(dtype=float)).sum()) == len(symbols)


def test_live_feature_cache_key_ignores_coverage_symbols():
    base = {
        "run_id": "run_a",
        "symbols": ["AAA/USDC", "BBB/USDC"],
        "required_feature_keys": {"ret24h"},
        "lookback_hours": 24 * 60,
    }

    key_a = feature_generator._live_feature_cache_key(
        **base,
        cfg={
            "train_min_range_pct": 0.06,
            "live_feature_coverage_symbols": ["AAA/USDC"],
        },
    )
    key_b = feature_generator._live_feature_cache_key(
        **base,
        cfg={
            "train_min_range_pct": 0.06,
            "live_feature_coverage_symbols": ["BBB/USDC"],
        },
    )

    assert key_a == key_b


def test_lazy_feature_coverage_can_ignore_source_rejected_symbols():
    fresh_ts = pd.Timestamp("2026-06-03 06:00", tz="UTC")
    stale_ts = pd.Timestamp("2026-06-02 21:00", tz="UTC")
    feats = LazyFeatureDict(
        raw_data_buffers={
            "ret24h": {
                "AAA/USDC": np.array([1.0], dtype=np.float32),
                "BBB/USDC": np.array([2.0], dtype=np.float32),
            }
        },
        symbol_indices={
            "AAA/USDC": np.array([fresh_ts.to_datetime64()]),
            "BBB/USDC": np.array([stale_ts.to_datetime64()]),
        },
    )

    assert feature_generator._cached_feature_coverage_end_ts(
        feats,
        required_feature_keys={"ret24h"},
    ) == stale_ts
    assert feature_generator._cached_feature_coverage_end_ts(
        feats,
        required_feature_keys={"ret24h"},
        coverage_symbols=["AAA/USDC"],
    ) == fresh_ts


def test_lazy_feature_coverage_ignores_stale_symbol_index_without_payload():
    fresh_ts = pd.Timestamp("2026-06-03 06:00", tz="UTC")
    stale_ts = pd.Timestamp("2026-06-02 21:00", tz="UTC")
    feats = LazyFeatureDict(
        raw_data_buffers={
            "ret24h": {
                "AAA/USDC": np.array([1.0], dtype=np.float32),
            }
        },
        symbol_indices={
            "AAA/USDC": np.array([fresh_ts.to_datetime64()]),
            "BBB/USDC": np.array([stale_ts.to_datetime64()]),
        },
    )

    assert feature_generator._cached_feature_coverage_end_ts(
        feats,
        required_feature_keys={"ret24h"},
        coverage_symbols=["AAA/USDC", "BBB/USDC"],
    ) == fresh_ts


def test_lazy_feature_coverage_keeps_stale_symbol_when_payload_exists():
    fresh_ts = pd.Timestamp("2026-06-03 06:00", tz="UTC")
    stale_ts = pd.Timestamp("2026-06-02 21:00", tz="UTC")
    feats = LazyFeatureDict(
        raw_data_buffers={
            "ret24h": {
                "AAA/USDC": np.array([1.0], dtype=np.float32),
                "BBB/USDC": np.array([2.0], dtype=np.float32),
            }
        },
        symbol_indices={
            "AAA/USDC": np.array([fresh_ts.to_datetime64()]),
            "BBB/USDC": np.array([stale_ts.to_datetime64()]),
        },
    )

    assert feature_generator._cached_feature_coverage_end_ts(
        feats,
        required_feature_keys={"ret24h"},
        coverage_symbols=["AAA/USDC", "BBB/USDC"],
    ) == stale_ts


def test_lazy_feature_coverage_uses_materialized_frame_after_assembly():
    fresh_ts = pd.Timestamp("2026-06-03 06:00", tz="UTC")
    stale_ts = pd.Timestamp("2026-06-02 21:00", tz="UTC")
    feats = LazyFeatureDict(
        raw_data_buffers={
            "ret24h": {
                "AAA/USDC": np.array([1.0], dtype=np.float32),
                "BBB/USDC": np.array([2.0], dtype=np.float32),
            }
        },
        symbol_indices={
            "AAA/USDC": np.array([fresh_ts.to_datetime64()]),
            "BBB/USDC": np.array([stale_ts.to_datetime64()]),
        },
    )

    assert feature_generator._cached_feature_coverage_end_ts(
        feats,
        required_feature_keys={"ret24h"},
        coverage_symbols=["AAA/USDC", "BBB/USDC"],
    ) == stale_ts
    _ = feats["ret24h"]
    assert feature_generator._cached_feature_coverage_end_ts(
        feats,
        required_feature_keys={"ret24h"},
        coverage_symbols=["AAA/USDC", "BBB/USDC"],
    ) == fresh_ts


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


def test_live_feature_cache_transform_contract_applies_to_model_not_mask_by_default():
    cfg_model = {
        "live_feature_cache_namespace": "model",
        "feature_transform_contract_hash": "transform_hash",
    }
    cfg_mask = {
        "live_feature_cache_namespace": "mask",
        "feature_transform_contract_hash": "transform_hash",
    }

    assert feature_generator._live_feature_cache_applies_feature_transform(cfg_model)
    assert feature_generator._live_feature_cache_contract_hash_from_cfg(cfg_model) == "transform_hash"
    assert not feature_generator._live_feature_cache_applies_feature_transform(cfg_mask)
    assert feature_generator._live_feature_cache_contract_hash_from_cfg(cfg_mask) is None


def test_live_feature_cache_mask_transform_contract_requires_explicit_opt_in():
    cfg = {
        "live_feature_cache_namespace": "mask",
        "feature_transform_contract_hash": "transform_hash",
        "live_feature_transform_non_model_namespaces": True,
    }

    assert feature_generator._live_feature_cache_applies_feature_transform(cfg)
    assert feature_generator._live_feature_cache_contract_hash_from_cfg(cfg) == "transform_hash"


def test_live_feature_cache_key_splits_raw_mask_from_transformed_mask():
    base = {
        "run_id": "run_a",
        "symbols": ["AAA/USDC"],
        "required_feature_keys": {"compression_ratio"},
        "lookback_hours": 24 * 60,
    }

    key_raw_mask = feature_generator._live_feature_cache_key(
        **base,
        cfg={
            "live_feature_cache_namespace": "mask",
            "feature_transform_contract_hash": "transform_hash",
        },
    )
    key_transformed_mask = feature_generator._live_feature_cache_key(
        **base,
        cfg={
            "live_feature_cache_namespace": "mask",
            "feature_transform_contract_hash": "transform_hash",
            "live_feature_transform_non_model_namespaces": True,
        },
    )

    assert key_raw_mask != key_transformed_mask


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


def test_prediction_ledger_row_falls_back_to_snapshot_fee_and_order_id_for_replay():
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
        execution_snapshot={"expected_fill_price": 100.0, "fee_bps": 7.0},
        was_traded=True,
        trade_result={
            "order": {"id": "order-123"},
            "realized_entry_price": 101.0,
            "entry_notional_quote": 250.0,
            "base_amount": 2.475,
            "entry_fee_source": "missing",
            "decision_to_entry_seconds": 12.5,
            "signal_to_entry_seconds": 300.0,
        },
    )

    assert row["entry_fee_bps"] == pytest.approx(7.0)
    assert row["fee_bps"] == pytest.approx(7.0)
    assert row["realized_fee_bps"] == pytest.approx(7.0)
    assert row["order_id"] == "order-123"
    assert row["position_id"] == "order-123"


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


def test_prediction_ledger_row_persists_dynamic_hr_surprise_diagnostics():
    row = run_inference._prediction_ledger_row(
        {
            "symbol": "BTC/USD:USD",
            "strategy_id": "short_asset_demo",
            "raw_score": 0.74,
            "rank_threshold": 0.70,
            "effective_threshold": 0.736165,
            "chain_results": {
                "normalized_rank_score": 0.72,
                "meta_head_hash": "head123",
                "dynamic_hr_surprise_threshold": 0.736165,
                "dynamic_hr_surprise_applied": True,
                "dynamic_hr_surprise_reason": "applied",
                "dynamic_hr_surprise_head": "short_asset",
                "dynamic_hr_surprise_z_eff": -0.402695,
                "dynamic_hr_surprise_guarded_y": 0.67,
                "dynamic_hr_surprise_w_lower": 0.1083,
                "dynamic_hr_surprise_w_raise": 0.1643,
                "dynamic_hr_surprise_state_age_days": 2.96,
            },
        },
        timestamp="2026-06-27T22:00:00Z",
        side="short",
        portfolio_decision="rejected",
        portfolio_reject_reason="rank_below_dynamic_threshold",
    )

    assert row["final_threshold"] == pytest.approx(0.736165)
    assert row["dynamic_hr_surprise_threshold"] == pytest.approx(0.736165)
    assert row["dynamic_hr_surprise_applied"] is True
    assert row["dynamic_hr_surprise_reason"] == "applied"
    assert row["dynamic_hr_surprise_head"] == "short_asset"
    assert row["dynamic_hr_surprise_z_eff"] == pytest.approx(-0.402695)
    assert row["dynamic_hr_surprise_guarded_y"] == pytest.approx(0.67)
    assert row["dynamic_hr_surprise_w_lower"] == pytest.approx(0.1083)
    assert row["dynamic_hr_surprise_w_raise"] == pytest.approx(0.1643)
    assert row["dynamic_hr_surprise_state_age_days"] == pytest.approx(2.96)


def test_prediction_ledger_uses_threshold_rank_for_portfolio_gate():
    row = run_inference._prediction_ledger_row(
        {
            "symbol": "IMX/USD:USD",
            "strategy_id": "long_dist_demo",
            "raw_score": 0.88,
            "rank_threshold": 0.91,
            "effective_threshold": 0.9244368837,
            "threshold_rank_score": 0.9877,
            "threshold_rank_score_source": "policy_rank_reference_percentile",
            "normalized_rank_score": 0.6313,
            "auction_rank_pct": 0.6313,
            "chain_results": {
                "meta_head_hash": "head123",
                "policy_rank_pct": 0.9877,
                "threshold_rank_score": 0.9877,
                "threshold_rank_score_source": "policy_rank_reference_percentile",
                "normalized_rank_score": 0.6313,
                "auction_rank_pct": 0.6313,
                "portfolio_gate": {
                    "rank_score": 0.9877,
                    "rank_score_source": "policy_rank_reference_percentile",
                    "ordering_rank_score": 0.6313,
                    "allocation_rank_score": 0.6313,
                    "initial_threshold": 0.9244368837,
                    "final_threshold": 0.9244368837,
                },
            },
        },
        timestamp="2026-06-28T06:00:00Z",
        side="long",
        portfolio_decision="accepted",
    )

    assert row["passed_rank_gate"] is True
    assert row["final_gate_rank_score"] == pytest.approx(0.9877)
    assert row["final_gate_rank_score_source"] == "policy_rank_reference_percentile"
    assert row["portfolio_gate_rank_score"] == pytest.approx(0.9877)
    assert row["portfolio_gate_rank_score_source"] == "policy_rank_reference_percentile"
    assert row["portfolio_ordering_rank_score"] == pytest.approx(0.6313)
    assert row["auction_rank_pct"] == pytest.approx(0.6313)


def test_live_spread_ev_haircut_lowers_policy_rank_score():
    class FakePolicyRankStore:
        def lookup(self, *, strategy_id, calibrated_score, side):
            del strategy_id, side
            return type(
                "Lookup",
                (),
                {
                    "policy_rank_pct": float(calibrated_score),
                    "n_rows": 100,
                    "source": "unit_test_policy_rank_reference",
                },
            )()

    out = run_inference._ev_adjusted_prediction_after_entry_friction(
        calibrated_score=0.80,
        strategy_id="short_asset_demo",
        side="short",
        calibration={
            "short_asset_demo": [
                {"mean_score": 0.60, "mean_net_return": 0.010, "count": 100},
                {"mean_score": 0.80, "mean_net_return": 0.030, "count": 100},
            ]
        },
        live_entry_friction_bps=120.0,
        observed_spread_bps=120.0,
        orderbook_slippage_bps=0.0,
        adverse_signal_gap_bps=0.0,
        spread_baseline_bps=20.0,
        spread_baseline_source="unit_test_symbol_average_spread",
        delay_slippage_baseline_bps=0.0,
        policy_rank_reference_store=FakePolicyRankStore(),
    )

    assert out["ev_haircut_observed_spread_bps"] == pytest.approx(120.0)
    assert out["ev_haircut_spread_baseline_bps"] == pytest.approx(20.0)
    assert out["ev_haircut_spread_excess_bps"] == pytest.approx(50.0)
    assert out["ev_adjusted_net_return_after_friction"] < out[
        "ev_adjusted_net_return_before_friction"
    ]
    assert out["ev_adjusted_calibrated_score"] < 0.80
    assert out["ev_adjusted_rank_score"] < 0.80
    assert out["ev_adjusted_source"] == (
        "hierarchical_side_archetype_ev_curve_inverse_after_excess_live_entry_friction"
    )


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


def test_model_feature_ledger_snapshot_resolves_side_prefixed_meta_clf_key():
    class DummyModel:
        def __init__(self, selected_features):
            self.selected_features = selected_features

    class DummyOrchestrator:
        alpha_by_strategy = {
            "short_demo": {
                "model": DummyModel(["ret24h"]),
                "feat_cols": ["ret24h"],
            }
        }
        meta_models = {
            "short_demo_clf": DummyModel(["demo", "pred_demo_H5", "pred_logit_H5", "ret24h"]),
        }

    snapshot = run_inference._model_feature_ledger_snapshot_for_decision(
        orchestrator=DummyOrchestrator(),
        side="short",
        strategy_id="demo",
        symbol="BTC/USD:USD",
        candidate_features=pd.DataFrame({"ret24h": [0.04]}, index=["BTC/USD:USD"]),
        feats={},
        chain_results={"base_pred": 0.61},
        signal_bar_ts="2026-05-30T12:00:00Z",
    )

    assert snapshot["meta_model_feature_key"] == "short_demo_clf"
    assert snapshot["meta_model_feature_count"] == 4
    values = json.loads(snapshot["meta_model_feature_values_json"])
    assert values["demo"] == 0.61
    assert values["pred_demo_H5"] == 0.61
    assert values["ret24h"] == 0.04
    assert values["pred_logit_H5"] > 0.0


def test_model_feature_ledger_snapshot_prefers_exact_meta_model_input_matrix():
    class DummyModel:
        def __init__(self, selected_features):
            self.selected_features = selected_features

    class DummyOrchestrator:
        alpha_by_strategy = {
            "long_demo": {
                "model": DummyModel(["ret24h"]),
                "feat_cols": ["ret24h"],
            }
        }
        meta_models = {
            "long_demo": DummyModel(
                ["long_demo", "feature_drift_cov_shift", "ret24h"]
            ),
        }

    snapshot = run_inference._model_feature_ledger_snapshot_for_decision(
        orchestrator=DummyOrchestrator(),
        side="long",
        strategy_id="long_demo",
        symbol="BTC/USD:USD",
        candidate_features=pd.DataFrame(
            {
                "ret24h": [0.04],
                "feature_drift_cov_shift": [0.10],
            },
            index=["BTC/USD:USD"],
        ),
        meta_model_input_features=pd.DataFrame(
            {
                "long_demo": [0.61],
                "feature_drift_cov_shift": [0.90],
                "ret24h": [0.04],
            },
            index=["BTC/USD:USD"],
        ),
        feats={},
        chain_results={"base_pred": 0.61},
        signal_bar_ts="2026-05-30T12:00:00Z",
    )

    values = json.loads(snapshot["meta_model_feature_values_json"])
    assert values["feature_drift_cov_shift"] == 0.90
    assert values["long_demo"] == 0.61
    assert values["ret24h"] == 0.04


def test_feature_store_gap_guard_allows_enough_full_rows():
    ts = pd.Timestamp("2026-06-19 20:00", tz="UTC")
    symbols = [f"S{i}/USD:USD" for i in range(10)]
    valid = symbols[:6]
    frame_a = pd.DataFrame(
        {sym: [1.0 if sym in valid else np.nan] for sym in symbols},
        index=[ts],
    )
    frame_b = pd.DataFrame(
        {sym: [2.0 if sym in valid else np.nan] for sym in symbols},
        index=[ts],
    )

    report = run_inference._selected_model_feature_store_gap_report(
        feats={"ret24h": frame_a, "adx_10": frame_b},
        symbols=symbols,
        required_feature_keys={"ret24h", "adx_10"},
        signal_bar_ts=ts,
        min_finite_fraction=0.80,
        min_full_rows=5,
    )

    assert report["ok"]
    assert report["reason"] == "ok_min_full_rows"
    assert report["full_feature_rows"] == 6
    assert report["low_finite_features"]


def test_feature_store_gap_guard_blocks_when_no_full_rows():
    ts = pd.Timestamp("2026-06-19 20:00", tz="UTC")
    symbols = [f"S{i}/USD:USD" for i in range(10)]
    frame_a = pd.DataFrame(
        {sym: [1.0 if i < 6 else np.nan] for i, sym in enumerate(symbols)},
        index=[ts],
    )
    frame_b = pd.DataFrame(
        {sym: [2.0 if i >= 6 else np.nan] for i, sym in enumerate(symbols)},
        index=[ts],
    )

    report = run_inference._selected_model_feature_store_gap_report(
        feats={"ret24h": frame_a, "adx_10": frame_b},
        symbols=symbols,
        required_feature_keys={"ret24h", "adx_10"},
        signal_bar_ts=ts,
        min_finite_fraction=0.80,
        min_full_rows=5,
    )

    assert not report["ok"]
    assert report["reason"] == "feature_store_gap"
    assert report["full_feature_rows"] == 0


def test_prediction_ledger_path_supports_run_scoped_override(monkeypatch, tmp_path):
    monkeypatch.delenv("EPM_PREDICTION_LEDGER_PATH", raising=False)
    monkeypatch.delenv("EPM_RUN_SCOPED_PREDICTION_LEDGER", raising=False)

    default_path = run_inference._resolve_prediction_ledger_path(
        live_data_root=tmp_path,
        run_id="run_a",
    )
    assert default_path == (
        tmp_path / "live_state" / "prediction_ledgers" / "run_a" / "prediction_ledger.parquet"
    )

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


def test_scored_prediction_candidate_logged_for_audit_below_legacy_top_floor(monkeypatch):
    monkeypatch.delenv("EPM_LOG_ALL_PREDICTION_CANDIDATES", raising=False)
    monkeypatch.delenv("EPM_LOG_ALL_SCORED_PREDICTION_CANDIDATES", raising=False)
    policy = run_inference.PortfolioPolicyConfig(top_prediction_ledger_pct=0.15)

    assert run_inference._should_log_prediction_candidate(
        {"sizer_rank_percentile": 0.589},
        policy=policy,
    )

    monkeypatch.setenv("EPM_LOG_ALL_SCORED_PREDICTION_CANDIDATES", "0")
    assert not run_inference._should_log_prediction_candidate(
        {"sizer_rank_percentile": 0.589},
        policy=policy,
    )


def test_offline_feature_lookup_accepts_artifact_source_run_id(monkeypatch):
    monkeypatch.setenv("EPM_ARTIFACT_SOURCE_RUN_ID", "20260523_015947")

    assert (
        feature_generator._offline_feature_lookup_run_id({}, "20260525_010004_nopenalty")
        == "20260523_015947"
    )


def test_offline_feature_lookup_accepts_parity_contract_feature_source(monkeypatch):
    monkeypatch.delenv("EPM_LIVE_FEATURE_SOURCE_RUN_ID", raising=False)
    monkeypatch.delenv("EPM_FEATURE_SOURCE_RUN_ID", raising=False)
    monkeypatch.delenv("EPM_ARTIFACT_SOURCE_RUN_ID", raising=False)

    assert (
        feature_generator._offline_feature_lookup_run_id(
            {
                "training_live_parity_contract": {
                    "feature_source": {"run_id": "20260523_015947"}
                }
            },
            "20260525_010004_nopenalty",
        )
        == "20260523_015947"
    )


def test_model_feature_source_override_prefers_offline_cache(monkeypatch):
    idx = pd.date_range("2026-06-04 10:00", periods=2, freq="1h", tz="UTC")
    symbols = ["AAA/USD:USD"]
    panel = {
        "close": pd.DataFrame({"AAA/USD:USD": [100.0, 101.0]}, index=idx),
    }
    rolling = {
        "feat_a": pd.DataFrame({"AAA/USD:USD": [1.0, 1.0]}, index=idx),
    }
    snapshot = {
        "feat_a": pd.DataFrame({"AAA/USD:USD": [0.5]}, index=idx[-1:]),
    }
    offline = {
        "feat_a": pd.DataFrame({"AAA/USD:USD": [2.0, 2.0]}, index=idx),
    }
    captured_offline_request = {}

    monkeypatch.setattr(
        feature_generator,
        "_load_live_feature_snapshot",
        lambda **kwargs: snapshot,
    )
    monkeypatch.setattr(
        feature_generator,
        "_load_live_feature_rolling_cache",
        lambda **kwargs: rolling,
    )
    monkeypatch.setattr(
        feature_generator,
        "load_cached_features_for_inference",
        lambda **kwargs: captured_offline_request.update(kwargs) or offline,
    )
    monkeypatch.setattr(feature_generator, "_compute_per_symbol_features", lambda *a, **k: {})
    monkeypatch.setattr(
        feature_generator,
        "_compute_policy_barrier_pct",
        lambda *a, **k: pd.DataFrame(),
    )
    monkeypatch.setattr(feature_generator, "_write_live_feature_snapshot", lambda **kwargs: None)
    monkeypatch.setattr(feature_generator, "_write_live_feature_rolling_cache", lambda **kwargs: None)

    feats = feature_generator.load_or_compute_features(
        panel=panel,
        basket_syms=symbols,
        run_id="deploy_run",
        data_root="data_perp/exchanges/krakenfutures",
        cfg={
            "live_feature_cache_namespace": "model",
            "live_feature_offline_cache_enabled": True,
            "training_live_parity_contract": {
                "feature_source": {"run_id": "feature_run"}
            },
        },
        lookback_hours=2,
        required_feature_keys={"feat_a"},
    )

    assert float(feats["feat_a"].loc[idx[-1], "AAA/USD:USD"]) == 2.0
    assert pd.Timestamp(captured_offline_request["start_ts"]) == idx[-1]
    assert pd.Timestamp(captured_offline_request["end_ts"]) == idx[-1]


def test_model_feature_source_override_does_not_backfill_missing_offline_keys(monkeypatch):
    idx = pd.date_range("2026-06-04 10:00", periods=2, freq="1h", tz="UTC")
    symbols = ["AAA/USD:USD"]
    panel = {
        "close": pd.DataFrame({"AAA/USD:USD": [100.0, 101.0]}, index=idx),
    }
    rolling_called = {"value": False}

    def fail_rolling(**kwargs):
        rolling_called["value"] = True
        return {
            "feat_missing": pd.DataFrame(
                {"AAA/USD:USD": [9.0, 9.0]},
                index=idx,
            )
        }

    monkeypatch.setattr(feature_generator, "_load_live_feature_rolling_cache", fail_rolling)
    monkeypatch.setattr(
        feature_generator,
        "load_cached_features_for_inference",
        lambda **kwargs: {
            "feat_a": pd.DataFrame({"AAA/USD:USD": [2.0]}, index=idx[-1:]),
        },
    )
    monkeypatch.setattr(feature_generator, "_compute_per_symbol_features", lambda *a, **k: {})
    monkeypatch.setattr(
        feature_generator,
        "_compute_policy_barrier_pct",
        lambda *a, **k: pd.DataFrame(),
    )
    monkeypatch.setattr(feature_generator, "_write_live_feature_snapshot", lambda **kwargs: None)
    monkeypatch.setattr(feature_generator, "_write_live_feature_rolling_cache", lambda **kwargs: None)

    feats = feature_generator.load_or_compute_features(
        panel=panel,
        basket_syms=symbols,
        run_id="deploy_run",
        data_root="data_perp/exchanges/krakenfutures",
        cfg={
            "live_feature_cache_namespace": "model",
            "live_feature_offline_cache_enabled": True,
            "live_model_feature_store_strict": True,
            "live_model_feature_auto_sync_selected_cache": False,
            "training_live_parity_contract": {
                "feature_source": {"run_id": "feature_run"}
            },
        },
        lookback_hours=2,
        required_feature_keys={"feat_a", "feat_missing"},
    )

    assert rolling_called["value"] is False
    assert float(feats["feat_a"].loc[idx[-1], "AAA/USD:USD"]) == 2.0
    assert "feat_missing" in feats
    assert np.isnan(feats["feat_missing"].loc[idx[-1], "AAA/USD:USD"])


def test_strict_model_source_override_repairs_source_derived_before_strict_nan(
    monkeypatch,
):
    idx = pd.date_range("2026-06-04 10:00", periods=2, freq="1h", tz="UTC")
    symbols = ["AAA/USD:USD"]
    panel = {
        "close": pd.DataFrame({"AAA/USD:USD": [100.0, 101.0]}, index=idx),
    }

    sync_calls = []

    def record_sync(*args, **kwargs):
        sync_calls.append(kwargs)
        return False

    monkeypatch.setattr(
        feature_generator, "_run_training_path_feature_sync_for_live", record_sync
    )
    monkeypatch.setattr(
        feature_generator,
        "load_cached_features_for_inference",
        lambda **kwargs: {
            "feat_a": pd.DataFrame({"AAA/USD:USD": [2.0]}, index=idx[-1:]),
        },
    )
    monkeypatch.setattr(
        feature_generator, "_write_live_feature_snapshot", lambda **kwargs: None
    )
    monkeypatch.setattr(
        feature_generator, "_write_live_feature_rolling_cache", lambda **kwargs: None
    )

    feats = feature_generator.load_or_compute_features(
        panel=panel,
        basket_syms=symbols,
        run_id="deploy_run",
        data_root="data_perp/exchanges/krakenfutures",
        cfg={
            "live_feature_cache_namespace": "model",
            "live_feature_offline_cache_enabled": True,
            "live_feature_prefer_offline_cache": True,
            "live_model_feature_store_strict": True,
            "live_model_feature_auto_sync_selected_cache": True,
            "training_live_parity_contract": {
                "feature_source": {"run_id": "feature_run"}
            },
        },
        lookback_hours=2,
        required_feature_keys={"feat_a", "oi_3d_chg_z"},
    )

    assert float(feats["feat_a"].loc[idx[-1], "AAA/USD:USD"]) == 2.0
    assert sync_calls
    assert sync_calls[0]["required_feature_keys"] == ["oi_3d_chg_z"]
    assert sync_calls[0]["symbols"] == symbols
    assert "oi_3d_chg_z" in feats
    assert np.isnan(feats["oi_3d_chg_z"].loc[idx[-1], "AAA/USD:USD"])


def test_low_finite_selected_cache_does_not_sync_when_full_rows_available(monkeypatch):
    idx = pd.date_range("2026-06-04 10:00", periods=2, freq="1h", tz="UTC")
    symbols = [f"S{i}/USD:USD" for i in range(4)]
    panel = {
        "close": pd.DataFrame(
            {symbol: [100.0, 101.0] for symbol in symbols},
            index=idx,
        ),
    }
    offline = {
        "feat_a": pd.DataFrame(
            {symbol: [1.0] for symbol in symbols},
            index=idx[-1:],
        ),
        "feat_b": pd.DataFrame(
            {
                symbols[0]: [2.0],
                symbols[1]: [3.0],
                symbols[2]: [np.nan],
                symbols[3]: [np.nan],
            },
            index=idx[-1:],
        ),
    }

    def fail_sync(*args, **kwargs):
        raise AssertionError(
            "low finite support should not trigger selected-cache sync when "
            "enough full-parity rows remain"
        )

    monkeypatch.setattr(
        feature_generator, "_run_training_path_feature_sync_for_live", fail_sync
    )
    monkeypatch.setattr(
        feature_generator,
        "load_cached_features_for_inference",
        lambda **kwargs: offline,
    )
    monkeypatch.setattr(
        feature_generator, "_write_live_feature_snapshot", lambda **kwargs: None
    )
    monkeypatch.setattr(
        feature_generator, "_write_live_feature_rolling_cache", lambda **kwargs: None
    )

    feats = feature_generator.load_or_compute_features(
        panel=panel,
        basket_syms=symbols,
        run_id="deploy_run",
        data_root="data_perp/exchanges/krakenfutures",
        cfg={
            "live_feature_cache_namespace": "model",
            "live_feature_offline_cache_enabled": True,
            "live_feature_prefer_offline_cache": True,
            "live_model_feature_store_strict": True,
            "live_model_feature_auto_sync_selected_cache": True,
            "live_model_feature_auto_sync_on_low_finite": True,
            "live_model_feature_store_gap_min_full_rows": 2,
            "live_model_feature_selected_cache_min_latest_finite_fraction": 0.80,
            "training_live_parity_contract": {
                "feature_source": {"run_id": "feature_run"}
            },
        },
        lookback_hours=2,
        required_feature_keys={"feat_a", "feat_b"},
    )

    assert int(
        np.isfinite(feats["feat_b"].loc[idx[-1], symbols].to_numpy(dtype=float)).sum()
    ) == 2


def test_model_feature_source_override_materializes_execution_barrier(monkeypatch):
    idx = pd.date_range("2026-06-04 10:00", periods=4, freq="1h", tz="UTC")
    symbols = ["AAA/USD:USD"]
    close = pd.DataFrame({"AAA/USD:USD": [100.0, 101.0, 102.0, 103.0]}, index=idx)
    panel = {
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
    }
    offline = {
        "feat_a": pd.DataFrame({"AAA/USD:USD": [2.0]}, index=idx[-1:]),
    }

    monkeypatch.setattr(
        feature_generator,
        "load_cached_features_for_inference",
        lambda **kwargs: offline,
    )
    monkeypatch.setattr(feature_generator, "_write_live_feature_snapshot", lambda **kwargs: None)
    monkeypatch.setattr(feature_generator, "_write_live_feature_rolling_cache", lambda **kwargs: None)

    feats = feature_generator.load_or_compute_features(
        panel=panel,
        basket_syms=symbols,
        run_id="deploy_run",
        data_root="data_perp/exchanges/krakenfutures",
        cfg={
            "live_feature_cache_namespace": "model",
            "live_feature_offline_cache_enabled": True,
            "live_feature_prefer_offline_cache": True,
            "live_model_feature_store_strict": True,
            "live_model_feature_auto_sync_selected_cache": False,
            "training_live_parity_contract": {
                "feature_source": {"run_id": "feature_run"}
            },
        },
        lookback_hours=4,
        required_feature_keys={"feat_a", "barrier_pct"},
    )

    assert "barrier_pct" in feats
    barrier = feats["barrier_pct"].loc[idx[-1], "AAA/USD:USD"]
    assert np.isfinite(barrier)
    assert barrier > 0.0
    assert float(feats["feat_a"].loc[idx[-1], "AAA/USD:USD"]) == 2.0


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


def test_model_feature_offline_cache_defaults_to_feature_source(monkeypatch):
    monkeypatch.delenv("EPM_LIVE_FEATURE_SOURCE_RUN_ID", raising=False)
    monkeypatch.delenv("EPM_FEATURE_SOURCE_RUN_ID", raising=False)
    monkeypatch.delenv("EPM_ARTIFACT_SOURCE_RUN_ID", raising=False)

    cfg = {
        "training_live_parity_contract": {
            "feature_source": {"run_id": "20260523_015947"}
        }
    }

    assert run_inference._model_feature_offline_cache_enabled(cfg) is True
    assert (
        run_inference._model_feature_offline_cache_enabled(
            {**cfg, "live_model_feature_offline_cache_enabled": False}
        )
        is False
    )


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


def test_mask_feature_source_override_does_not_load_offline_selected_cache(monkeypatch):
    symbols = ["AAA/USD:USD"]
    idx = pd.date_range("2026-06-04 07:00", periods=6, freq="1h", tz="UTC")
    close = pd.DataFrame(
        {"AAA/USD:USD": [97.0, 98.0, 99.0, 100.0, 101.0, 102.0]},
        index=idx,
    )
    panel = {
        "close": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": pd.DataFrame(1000.0, index=idx, columns=symbols),
    }
    stale_idx = idx[:-1]
    stale_cache = {
        "ret1h": pd.DataFrame(
            {"AAA/USD:USD": [0.0, 0.01, 0.01, 0.01, 0.01]},
            index=stale_idx,
        )
    }
    computed = {"ret1h": close.pct_change().astype(np.float32)}
    captured_tail = {}

    monkeypatch.setattr(feature_generator, "_load_live_feature_snapshot", lambda **kwargs: {})
    monkeypatch.setattr(
        feature_generator,
        "_load_live_feature_rolling_cache",
        lambda **kwargs: stale_cache,
    )

    def fail_offline(**kwargs):
        raise AssertionError("mask namespace should not load selected offline features")

    monkeypatch.setattr(feature_generator, "load_cached_features_for_inference", fail_offline)
    monkeypatch.setattr(feature_generator, "_compute_per_symbol_features", lambda *a, **k: {})
    monkeypatch.setattr(
        feature_generator,
        "_compute_policy_barrier_pct",
        lambda *a, **k: pd.DataFrame(),
    )
    def fake_compute_features_hourly(panel_tail, *args, **kwargs):
        captured_tail["first_ts"] = panel_tail["close"].index.min()
        return (
            {"ret1h": panel_tail["close"].pct_change().astype(np.float32)},
            {},
            {},
        )

    monkeypatch.setattr(
        feature_generator,
        "compute_features_hourly",
        fake_compute_features_hourly,
    )
    monkeypatch.setattr(feature_generator, "_write_live_feature_snapshot", lambda **kwargs: None)
    monkeypatch.setattr(feature_generator, "_write_live_feature_rolling_cache", lambda **kwargs: None)

    feats = feature_generator.load_or_compute_features(
        panel=panel,
        basket_syms=symbols,
        run_id="deploy_run",
        data_root="data_perp/exchanges/krakenfutures",
        cfg={
            "live_feature_cache_namespace": "mask",
            "live_feature_offline_cache_enabled": True,
            "training_live_parity_contract": {
                "feature_source": {"run_id": "feature_run"}
            },
            "live_mask_feature_tail_warmup_hours": 1,
        },
        lookback_hours=3,
        required_feature_keys={"ret1h"},
    )

    assert captured_tail["first_ts"] == idx[-3]
    assert "ret1h" in feats
    assert feats["ret1h"].index[-1] == idx[-1]
    assert float(feats["ret1h"].loc[idx[-1], "AAA/USD:USD"]) == pytest.approx(
        102.0 / 101.0 - 1.0
    )


def test_live_feature_cache_prune_does_not_delete_other_namespace(tmp_path):
    root = tmp_path / "live_feature_cache"
    model_dir = root / "model_cache"
    mask_dir = root / "mask_cache"
    model_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    (model_dir / "rolling_meta.json").write_text(
        json.dumps(
            {
                "version": feature_generator.LIVE_FEATURE_CACHE_VERSION,
                "cache_namespace": "model",
                "contract_hash": None,
                "symbols_hash": "symbols_a",
                "required_hash": "model_required",
            }
        )
    )
    active_meta = {
        "version": feature_generator.LIVE_FEATURE_CACHE_VERSION,
        "cache_namespace": "mask",
        "contract_hash": None,
        "symbols_hash": "symbols_a",
        "required_hash": "mask_required",
    }
    (mask_dir / "rolling_meta.json").write_text(json.dumps(active_meta))

    feature_generator._prune_stale_live_feature_cache_dirs(
        cfg={"live_feature_snapshot_cache_dir": str(root)},
        run_id="run_a",
        active_cache_dir=mask_dir,
        active_meta=active_meta,
    )

    assert model_dir.exists()
    assert mask_dir.exists()


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


def test_live_causal_transform_fast_path_uses_hybrid_state_for_overlap(tmp_path):
    idx = pd.date_range("2026-06-01", periods=10, freq="1h", tz="UTC")
    columns = ["AAA/USD:USD", "BBB/USD:USD"]
    raw = pd.DataFrame(
        {
            "AAA/USD:USD": np.linspace(-0.04, 0.05, len(idx)),
            "BBB/USD:USD": np.linspace(0.03, -0.02, len(idx)),
        },
        index=idx,
    ).astype(np.float32)
    state_path = tmp_path / "causal_zscore_state.npz"
    cfg = {
        "live_causal_transform_state_enabled": True,
        "live_causal_transform_state_path": str(state_path),
        "feature_causal_transform_state_bootstrap_max_rows": 100,
        "feature_causal_transform_min_required_ts": idx[3].isoformat(),
    }

    first = features._apply_causal_transform_live_state_or_batch(
        {"ret24h": raw.iloc[:6].copy()},
        cfg,
        feature_index=idx[:6],
        feature_columns=columns,
        skip_transform_set=set(),
    )
    assert np.isfinite(first["ret24h"].to_numpy()).all()

    second = features._apply_causal_transform_live_state_or_batch(
        {"ret24h": raw.copy()},
        cfg,
        feature_index=idx,
        feature_columns=columns,
        skip_transform_set=set(),
    )

    from extreme_price_movements.inference.live_zscore_state import RollingZScoreState

    expected_state = RollingZScoreState(
        ["ret24h"],
        columns,
        window=24 * 30,
        sigma_k=2.0537489106318225,
        winsor_qt=0.02,
    )
    expected_rows = []
    for ts, row in raw.iterrows():
        expected_rows.append(
            expected_state.update({"ret24h": row.to_numpy()}, timestamp=ts.isoformat())[
                "ret24h"
            ]
        )
    expected = np.vstack(expected_rows)

    # Rows after the saved state timestamp should be produced by the persisted
    # state update, while the overlapping prefix can still be filled by the
    # vectorized batch path for cache writeback.
    np.testing.assert_allclose(
        second["ret24h"].iloc[6:].to_numpy(),
        expected[6:],
        rtol=1e-6,
        atol=1e-6,
    )
