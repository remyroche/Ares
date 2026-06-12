from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements import kraken_spread_model as ksm
from extreme_price_movements import universe as universe_mod


def test_parse_kraken_futures_tickers_payload_computes_bps_and_ticks():
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    item = ksm.SpreadUniverseItem(
        base="BTC",
        perp_symbol="BTC/USD:USD",
        perp_market_id="PF_XBTUSD",
        tick_size=0.5,
    )
    payload = {
        "result": "success",
        "serverTime": "2026-01-01T00:00:30.000Z",
        "tickers": [
            {
                "symbol": "PF_XBTUSD",
                "pair": "BTC:USD",
                "bid": "100.0",
                "ask": "100.5",
                "bidSize": "12",
                "askSize": "13",
            }
        ],
    }

    out = ksm.parse_kraken_futures_tickers_payload(
        payload,
        universe=[item],
    )

    assert list(out.index) == [start]
    assert out.iloc[0]["symbol"] == "BTC/USD:USD"
    assert out.iloc[0]["perp_market_id"] == "PF_XBTUSD"
    assert out.iloc[0]["bid_size"] == pytest.approx(12.0)
    assert out.iloc[0]["ask_size"] == pytest.approx(13.0)
    assert out["spread_ticks"].tolist() == pytest.approx([1.0])
    assert out.iloc[0]["spread_bps"] == pytest.approx(10000.0 * 0.5 / 100.25)
    assert out.iloc[0]["min_tick_spread_bps"] == pytest.approx(
        10000.0 * 0.5 / 100.25
    )


def test_seconds_until_next_hour_respects_top_of_hour_grace():
    assert ksm._seconds_until_next_hour(
        pd.Timestamp("2026-01-01T00:00:03Z"),
        grace_seconds=5.0,
    ) == pytest.approx(0.0)
    assert ksm._seconds_until_next_hour(
        pd.Timestamp("2026-01-01T00:00:06Z"),
        grace_seconds=5.0,
    ) == pytest.approx(3594.0)
    assert ksm._seconds_until_next_hour(
        pd.Timestamp("2026-01-01T00:59:30Z"),
        grace_seconds=5.0,
    ) == pytest.approx(30.0)


def test_collect_spread_snapshots_hourly_mode_forces_single_minute(monkeypatch, tmp_path):
    item = ksm.SpreadUniverseItem(
        base="BTC",
        perp_symbol="BTC/USD:USD",
        perp_market_id="PF_XBTUSD",
        tick_size=0.5,
    )
    ts = pd.Timestamp("2026-01-01T00:00:00Z")
    calls = {}

    monkeypatch.setattr(ksm, "make_perp_exchange", lambda: object())
    monkeypatch.setattr(
        ksm,
        "resolve_spread_universe",
        lambda _exchange, symbols=None: (
            [item],
            [
                {
                    "base": item.base,
                    "perp_symbol": item.perp_symbol,
                    "perp_market_id": item.perp_market_id,
                    "tick_size": item.tick_size,
                    "status": "eligible",
                }
            ],
        ),
    )
    monkeypatch.setattr(ksm, "_seconds_until_next_hour", lambda *args, **kwargs: 0.0)

    def fake_collect_spreads(*, universe, snapshot_count, snapshot_interval_seconds):
        calls["snapshot_count"] = snapshot_count
        calls["snapshot_interval_seconds"] = snapshot_interval_seconds
        return pd.DataFrame(
            {
                "symbol": [item.perp_symbol],
                "perp_market_id": [item.perp_market_id],
                "base": [item.base],
                "bid": [100.0],
                "ask": [100.5],
                "spread_bps": [49.875],
                "spread_ticks": [1.0],
                "min_tick_spread_bps": [49.875],
                "tick_size": [0.5],
            },
            index=[ts],
        )

    def fake_collect_candles(**_kwargs):
        candles = pd.DataFrame(
            {"symbol": [item.perp_symbol], "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5]},
            index=[ts],
        )
        training = candles.assign(spread_bps=49.875, spread_ticks=1.0)
        return candles, training, [{"symbol": item.perp_symbol, "status": "ok"}]

    monkeypatch.setattr(ksm, "collect_kraken_futures_ticker_spreads", fake_collect_spreads)
    monkeypatch.setattr(ksm, "collect_associated_candles_and_training", fake_collect_candles)
    monkeypatch.setattr(
        ksm,
        "save_spread_snapshot_collection",
        lambda *args, **kwargs: (tmp_path / "snapshot.parquet", tmp_path / "summary.json"),
    )

    rc = ksm.collect_spread_snapshots_main(
        [
            "--hourly-top-of-hour",
            "--cycles",
            "1",
            "--snapshot-count",
            "5",
            "--snapshot-interval-seconds",
            "60",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert calls["snapshot_count"] == 1
    assert calls["snapshot_interval_seconds"] == pytest.approx(0.0)


def test_spread_cost_universe_exclusions_use_average_spread_baseline(monkeypatch, tmp_path):
    baseline_path = tmp_path / "per_asset_spread_baseline_latest.csv"
    baseline_path.write_text(
        "symbol,rows,average_spread_bps,median_spread_bps,p75_spread_bps,average_spread_ticks\n"
        "GOOD/USD:USD,10,124.9,100.0,120.0,1.0\n"
        "BAD/USD:USD,10,125.1,100.0,130.0,2.0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EPM_SPREAD_BLACKLIST_BASELINE_PATH", str(baseline_path))
    monkeypatch.setenv("EPM_SPREAD_BLACKLIST_THRESHOLD_BPS", "125")
    monkeypatch.delenv("EPM_DISABLE_SPREAD_BLACKLIST", raising=False)
    universe_mod._SPREAD_COST_EXCLUSION_CACHE.clear()

    out = universe_mod.apply_hardcoded_universe_exclusions(
        ["GOOD/USD:USD", "BAD/USD:USD"]
    )

    assert out == ["GOOD/USD:USD"]


def test_resolve_spread_universe_uses_futures_markets_only():
    class FakePerpExchange:
        markets = {
            "BTC/USD:USD": {
                "active": True,
                "swap": True,
                "base": "BTC",
                "quote": "USD",
                "settle": "USD",
                "symbol": "BTC/USD:USD",
                "id": "PF_XBTUSD",
                "info": {"symbol": "PF_XBTUSD", "status": "online"},
                "tickSize": "0.5",
            },
            "ETH/USD:USD": {
                "active": False,
                "swap": True,
                "base": "ETH",
                "quote": "USD",
                "settle": "USD",
                "symbol": "ETH/USD:USD",
                "id": "PF_ETHUSD",
            },
        }

    universe, audit = ksm.resolve_spread_universe(FakePerpExchange())

    assert universe == [
        ksm.SpreadUniverseItem(
            base="BTC",
            perp_symbol="BTC/USD:USD",
            perp_market_id="PF_XBTUSD",
            tick_size=0.5,
        )
    ]
    assert audit[0]["status"] == "eligible"
    assert set(audit[0]) == {
        "base",
        "perp_symbol",
        "perp_market_id",
        "tick_size",
        "status",
    }


def test_compute_spread_relevant_candle_features_matches_contract():
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="1min")
    candles = pd.DataFrame(
        {
            "open": [100.0, 102.0],
            "high": [104.0, 105.0],
            "low": [99.0, 101.0],
            "close": [102.0, 104.0],
            "volume": [10.0, 11.0],
        },
        index=idx,
    )

    feats = ksm.compute_spread_relevant_candle_features(candles)

    assert list(feats.columns) == ksm.SPREAD_CANDLE_FEATURES
    assert feats.loc[idx[0], "hl_range_bps"] == pytest.approx(
        10000.0 * (104.0 - 99.0) / 102.0
    )
    assert feats.loc[idx[0], "abs_return_bps"] == pytest.approx(
        10000.0 * abs(102.0 / 100.0 - 1.0)
    )
    assert feats.loc[idx[0], "upper_wick_bps"] == pytest.approx(
        10000.0 * (104.0 - 102.0) / 102.0
    )
    assert feats.loc[idx[0], "lower_wick_bps"] == pytest.approx(
        10000.0 * (100.0 - 99.0) / 102.0
    )
    assert feats.loc[idx[0], "close_location"] == pytest.approx((102.0 - 99.0) / 5.0)
    assert feats.loc[idx[1], "gap_bps"] == pytest.approx(
        10000.0 * abs(102.0 - 102.0) / 102.0
    )
    assert feats.loc[idx[0], "log_candle_volume"] == pytest.approx(np.log1p(10.0))
    assert feats.loc[idx[0], "log_candle_quote_volume"] == pytest.approx(
        np.log1p(10.0 * 102.0)
    )


def test_add_spread_model_derived_features_adds_only_ohlcv_time_features():
    idx = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:01:00Z",
            "2026-01-01T00:01:00Z",
        ]
    )
    frame = pd.DataFrame(
        {
            "symbol": ["BTC/USD:USD", "ETH/USD:USD", "BTC/USD:USD", "ETH/USD:USD"],
            "spread_bps": [40.0, 100.0, 45.0, 110.0],
            "spread_ticks": [0.8, 5.0, 0.9, 5.5],
            "hl_range_bps": [10.0, 20.0, 11.0, 21.0],
            "abs_return_bps": [2.0, 4.0, 3.0, 5.0],
            "log_candle_volume": [1.0, 2.0, 1.1, 2.1],
            "log_candle_quote_volume": [5.0, 6.0, 5.1, 6.1],
        },
        index=idx,
    )

    out = ksm.add_spread_model_derived_features(frame)

    assert out["candle_quote_volume_rank"].tolist() == pytest.approx([0.5, 1.0, 0.5, 1.0])
    assert out["hl_range_bps_rank"].tolist() == pytest.approx([0.5, 1.0, 0.5, 1.0])
    assert out["minute_of_day_sin"].notna().all()
    assert out["day_of_week_cos"].notna().all()
    assert out.iloc[0]["asset_log_candle_quote_volume_lag1"] != out.iloc[0][
        "asset_log_candle_quote_volume_lag1"
    ]
    assert out.iloc[2]["asset_log_candle_quote_volume_lag1"] == pytest.approx(5.0)
    assert out.iloc[3]["asset_log_candle_quote_volume_lag1"] == pytest.approx(6.0)
    forbidden = {
        "l1_top_depth_quote_rank",
        "low_l1_top_depth_rank",
        "asset_spread_bps_lag1",
        "asset_spread_ticks_lag1",
        "tick_size_bps",
    }
    assert forbidden.isdisjoint(out.columns)


def test_fit_ridge_spread_model_selects_top5_and_records_diagnostics(tmp_path):
    n = 120
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=n, freq="1min")
    trend = np.linspace(1.0, 30.0, n)
    frame = pd.DataFrame(
        {
            "symbol": np.where(np.arange(n) < n // 2, "BTC/USD:USD", "ETH/USD:USD"),
            "spread_bps": 2.0 + 0.35 * trend,
            "spread_ticks": 1.0 + 0.50 * trend,
            "min_tick_spread_bps": 0.25,
            "hl_range_bps": trend,
            "abs_return_bps": trend * 0.7,
            "body_bps": trend * 0.5,
            "upper_wick_bps": trend * 0.2,
            "lower_wick_bps": trend * 0.1,
            "wick_to_range": np.linspace(0.1, 0.9, n),
            "close_location": np.linspace(0.2, 0.8, n),
            "gap_bps": np.r_[0.0, np.diff(trend)],
        },
        index=idx,
    )

    artifact, scored = ksm.fit_ridge_spread_model(frame, top_k=5, alpha=0.5)
    path = ksm.save_spread_model_outputs(
        artifact,
        scored,
        pd.DataFrame({"symbol": ["BTC/USD:USD"], "rows": [60]}),
        output_dir=tmp_path,
    )

    assert path.exists()
    assert len(artifact["selected_features"]) == 5
    assert artifact["model_type"] == "kraken_spread_deviation_ridge_v3"
    assert "candidate_features" in artifact
    assert artifact["target"] == "log1p(spread_bps)-log1p(asset_average_spread_bps_baseline)"
    assert artifact["baseline_type"] == "per_asset_average_spread_bps"
    assert artifact["predicted_cost_bps"] == pytest.approx(
        artifact["predicted_spread_75th_percentile"]
    )
    assert artifact["metrics"]["mae_spread_bps"] < 1.2
    assert "baseline_mae_spread_bps" in artifact["metrics"]
    assert "asset_spread_baseline_bps" in scored.columns
    assert "predicted_spread_deviation_from_baseline_bps" in scored.columns
    assert "wide_spread_classification" in artifact["metrics"]
    assert len(artifact["metrics"]["error_by_pair"]) == 2
    assert len(artifact["per_asset_average_spread"]) == 2
    assert len(artifact["per_asset_spread_baseline"]) == 2
    assert (tmp_path / "per_asset_spread_baseline_latest.csv").exists()
    assert ksm.load_spread_cost_bps(path) == pytest.approx(artifact["predicted_cost_bps"])


def test_save_spread_snapshot_collection_writes_candles_and_training(tmp_path):
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=2, freq="1min")
    spreads = pd.DataFrame(
        {
            "symbol": ["BTC/USD:USD", "BTC/USD:USD"],
            "perp_market_id": ["PF_XBTUSD", "PF_XBTUSD"],
            "base": ["BTC", "BTC"],
            "bid": [100.0, 101.0],
            "ask": [100.5, 101.5],
            "spread_bps": [49.875, 49.383],
            "spread_ticks": [1.0, 1.0],
            "min_tick_spread_bps": [49.875, 49.383],
            "tick_size": [0.5, 0.5],
        },
        index=idx,
    )
    candles = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.5, 100.5],
            "close": [100.5, 101.5],
            "volume": [10.0, 11.0],
            "symbol": ["BTC/USD:USD", "BTC/USD:USD"],
            "base": ["BTC", "BTC"],
            "perp_market_id": ["PF_XBTUSD", "PF_XBTUSD"],
        },
        index=idx,
    )
    training = spreads.join(
        ksm.compute_spread_relevant_candle_features(candles),
        how="inner",
    )

    parquet_path, summary_path = ksm.save_spread_snapshot_collection(
        spreads,
        universe_audit=[
            {
                "base": "BTC",
                "perp_symbol": "BTC/USD:USD",
                "perp_market_id": "PF_XBTUSD",
                "tick_size": 0.5,
                "status": "eligible",
            }
        ],
        candles=candles,
        training=training,
        candle_audit=[{"symbol": "BTC/USD:USD", "status": "ok"}],
        output_dir=tmp_path,
        run_id="test_run",
    )

    summary = json.loads(summary_path.read_text())
    assert parquet_path.exists()
    assert (tmp_path / "latest.parquet").exists()
    assert (tmp_path / "latest_candles.parquet").exists()
    assert (tmp_path / "latest_training.parquet").exists()
    assert summary["rows"] == 2
    assert summary["candle_rows"] == 2
    assert summary["training_rows"] == 2
    assert pd.read_parquet(tmp_path / "latest_training.parquet").shape[0] == 2
