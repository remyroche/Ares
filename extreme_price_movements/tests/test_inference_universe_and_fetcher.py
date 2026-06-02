import numpy as np
import pandas as pd

from extreme_price_movements.inference import config as inference_config
from extreme_price_movements.inference.data_fetcher import DataFetcher


class _FakeExchange:
    def load_markets(self):
        return {
            "BTC/USDT": {"quote": "USDT", "active": True, "margin": True},
            "BTC/USDC": {"quote": "USDC", "active": True, "margin": True},
            "ETH/USDT": {
                "quote": "USDT",
                "active": True,
                "info": {"isMarginTradingAllowed": True},
            },
            "ETH/USDC": {
                "quote": "USDC",
                "active": True,
                "info": {"isMarginTradingAllowed": True},
            },
            "XRP/BTC": {"quote": "BTC", "active": True, "margin": True},
            "DOGE/USDC": {"quote": "USDC", "active": False, "margin": True},
            "SOL/USDC": {"quote": "USDC", "active": True, "margin": False},
        }


def test_get_margin_universe_uses_exchange_market_metadata(monkeypatch):
    monkeypatch.setattr(inference_config, "_MARGIN_UNIVERSE_CACHE", None)
    monkeypatch.setattr(inference_config, "_MARGIN_UNIVERSE_CACHE_DAY", None)
    monkeypatch.setattr(inference_config, "_MARGIN_UNIVERSE_CACHE_QUOTE", None)
    symbols = inference_config.get_margin_universe(exchange=_FakeExchange())
    assert symbols == ["BTC/USDC", "ETH/USDC"]


def test_get_margin_universe_filters_young_symbols(monkeypatch):
    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    old_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=365)).timestamp() * 1000)

    class _Exchange:
        def load_markets(self):
            return {
                "OLD/USDT": {
                    "quote": "USDT",
                    "active": True,
                    "margin": True,
                    "info": {"onboardDate": old_ms},
                },
                "OLD/USDC": {
                    "quote": "USDC",
                    "active": True,
                    "margin": True,
                    "info": {"onboardDate": old_ms},
                },
                "NEW/USDC": {
                    "quote": "USDC",
                    "active": True,
                    "margin": True,
                    "info": {"onboardDate": now_ms},
                },
            }

    monkeypatch.setattr(inference_config, "_MARGIN_UNIVERSE_CACHE", None)
    monkeypatch.setattr(inference_config, "_MARGIN_UNIVERSE_CACHE_DAY", None)
    monkeypatch.setattr(inference_config, "_MARGIN_UNIVERSE_CACHE_QUOTE", None)
    symbols = inference_config.get_margin_universe(exchange=_Exchange())
    assert symbols == ["OLD/USDC"]


def test_resolve_inference_universes_restricts_tradable_to_training_symbols(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "data"
    feature_dir = data_root / "artifacts" / "run1" / "features"
    feature_dir.mkdir(parents=True)
    (feature_dir / "feature_health_symbol_summary.csv").write_text(
        "symbol,rows\nETH/USDT,100\n"
    )

    class _Exchange:
        def load_markets(self):
            return {
                "BTC/USDC": {"quote": "USDC", "active": True, "margin": True},
                "ETH/USDC": {"quote": "USDC", "active": True, "margin": True},
            }

    monkeypatch.setattr(inference_config, "_MARGIN_UNIVERSE_CACHE", None)
    monkeypatch.setattr(inference_config, "_MARGIN_UNIVERSE_CACHE_DAY", None)
    monkeypatch.setattr(inference_config, "_MARGIN_UNIVERSE_CACHE_QUOTE", None)
    out = inference_config.resolve_inference_universes(
        _Exchange(), data_root=str(data_root), run_id="run1"
    )

    assert out["download_symbols"] == ["BTC/USDC", "ETH/USDC"]
    assert out["tradable_symbols"] == ["ETH/USDC"]
    assert out["trained_symbols"] == ["ETH/USDT"]
    assert out["live_quote_currency"] == "USDC"


def test_trained_universe_prefers_label_manifest_over_tiny_health_summary(tmp_path):
    data_root = tmp_path / "data"
    run_dir = data_root / "artifacts" / "run1"
    feature_dir = run_dir / "features"
    labels_dir = run_dir / "labels"
    feature_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    (feature_dir / "feature_health_symbol_summary.csv").write_text(
        "symbol,rows\nBTC/USDT,100\n"
    )
    label_path = labels_dir / "train_example.parquet"
    pd.DataFrame({"__symbol__": ["BTC/USDT", "ETH/USDT", "SOL/USDT"]}).to_parquet(
        label_path
    )
    (labels_dir / "labels_manifest.json").write_text(
        (
            '{"datasets": {"train_example": {"file": "train_example.parquet", '
            '"rows": 3, "columns": ["__symbol__"]}}}'
        )
    )

    symbols = inference_config.load_trained_symbol_universe(
        str(data_root), run_id="run1"
    )

    assert symbols == {"BTC/USDT", "ETH/USDT", "SOL/USDT"}


def test_resolve_inference_universes_maps_explicit_usdt_symbols_to_usdc(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "data"
    feature_dir = data_root / "artifacts" / "run1" / "features"
    feature_dir.mkdir(parents=True)
    (feature_dir / "feature_health_symbol_summary.csv").write_text(
        "symbol,rows\nBTC/USDT,100\nETH/USDT,100\n"
    )

    out = inference_config.resolve_inference_universes(
        object(),
        data_root=str(data_root),
        run_id="run1",
        explicit_symbols=["BTC/USDT", "SOL/USDT"],
    )

    assert out["download_symbols"] == ["BTC/USDC", "SOL/USDC"]
    assert out["tradable_symbols"] == ["BTC/USDC"]


def test_fetch_incremental_universe_triggers_backfill_on_recent_gaps(monkeypatch):
    fetcher = DataFetcher(exchange=object(), data_root="data")

    def _fake_fetch_incremental(symbol):
        idx = pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC")
        return pd.DataFrame({"close": [1.0, 1.1]}, index=idx)

    gap_by_symbol = {"A/USDT": False, "B/USDT": True}
    backfill_calls = []

    monkeypatch.setattr(fetcher, "fetch_incremental", _fake_fetch_incremental)
    monkeypatch.setattr(fetcher, "has_recent_gap", lambda s, days=7: gap_by_symbol[s])
    monkeypatch.setattr(
        fetcher,
        "trigger_gap_backfill",
        lambda symbol, **kwargs: backfill_calls.append(symbol),
    )

    out = fetcher.fetch_incremental_universe(["A/USDT", "B/USDT"], max_workers=2)
    assert set(out.keys()) == {"A/USDT", "B/USDT"}
    assert backfill_calls == ["B/USDT"]


def test_data_fetcher_does_not_rescope_exchange_scoped_live_root(tmp_path):
    class _Exchange:
        id = "krakenfutures"

    live_root = tmp_path / "data_perp" / "exchanges" / "krakenfutures"
    fetcher = DataFetcher(
        exchange=_Exchange(),
        data_root=str(live_root),
        market_mode="perps",
    )

    assert fetcher.ohlcv_store.root_dir == str(live_root)
    assert fetcher.orderbook_dir == live_root / "orderbook_hourly"
    assert fetcher.funding_dir == live_root / "funding_hourly"


def test_data_fetcher_scopes_artifact_root_by_exchange(tmp_path):
    class _Exchange:
        id = "krakenfutures"

    artifact_root = tmp_path / "data_perp"
    expected_root = artifact_root / "exchanges" / "krakenfutures"
    fetcher = DataFetcher(
        exchange=_Exchange(),
        data_root=str(artifact_root),
        market_mode="perps",
    )

    assert fetcher.ohlcv_store.root_dir == str(expected_root)
    assert fetcher.orderbook_dir == expected_root / "orderbook_hourly"
    assert fetcher.funding_dir == expected_root / "funding_hourly"


def test_fetch_incremental_universe_respects_lightweight_probe(monkeypatch):
    fetcher = DataFetcher(exchange=object(), data_root="data")
    calls = []

    monkeypatch.setattr(
        fetcher,
        "needs_incremental_update",
        lambda symbol: symbol == "A/USDT",
    )
    monkeypatch.setattr(
        fetcher,
        "fetch_incremental",
        lambda symbol: calls.append(symbol) or pd.DataFrame(),
    )
    monkeypatch.setattr(fetcher, "has_recent_gap", lambda symbol, days=7: False)

    fetcher.fetch_incremental_universe(
        ["A/USDT", "B/USDT"],
        max_workers=2,
        use_lightweight_probe=True,
    )
    assert calls == ["A/USDT"]


def test_fetch_incremental_universe_skips_microdata_refresh_by_default(monkeypatch):
    fetcher = DataFetcher(exchange=object(), data_root="data")
    micro_calls = []

    monkeypatch.setattr(
        fetcher,
        "fetch_incremental",
        lambda symbol: pd.DataFrame({"close": [1.0]}),
    )
    monkeypatch.setattr(fetcher, "has_recent_gap", lambda symbol, days=7: False)
    monkeypatch.setattr(
        fetcher,
        "update_microdata_symbol",
        lambda symbol: micro_calls.append(symbol),
    )

    fetcher.fetch_incremental_universe(["A/USDT"], refresh_microdata=False)

    assert micro_calls == []


def test_fetch_incremental_universe_microdata_failure_is_not_dead_letter(monkeypatch):
    fetcher = DataFetcher(exchange=object(), data_root="data")

    monkeypatch.setattr(
        fetcher,
        "fetch_incremental",
        lambda symbol: pd.DataFrame({"close": [1.0]}),
    )
    monkeypatch.setattr(fetcher, "has_recent_gap", lambda symbol, days=7: False)
    monkeypatch.setattr(
        fetcher,
        "update_microdata_symbol",
        lambda symbol: (_ for _ in ()).throw(TimeoutError("microdata timeout")),
    )

    out = fetcher.fetch_incremental_universe(["A/USDT"], refresh_microdata=True)

    assert set(out.keys()) == {"A/USDT"}
    assert fetcher.dead_letter_symbols == {}


def test_fetch_incremental_universe_records_dead_letters(monkeypatch):
    fetcher = DataFetcher(exchange=object(), data_root="data")

    def _fake_fetch_incremental(symbol):
        if symbol == "B/USDT":
            raise TimeoutError("network timeout")
        idx = pd.date_range("2026-01-01", periods=1, freq="1h", tz="UTC")
        return pd.DataFrame({"close": [1.0]}, index=idx)

    monkeypatch.setattr(fetcher, "fetch_incremental", _fake_fetch_incremental)
    monkeypatch.setattr(fetcher, "has_recent_gap", lambda symbol, days=7: False)

    out = fetcher.fetch_incremental_universe(
        ["A/USDT", "B/USDT"],
        max_workers=2,
        use_lightweight_probe=False,
    )

    assert set(out.keys()) == {"A/USDT"}
    assert "B/USDT" in fetcher.dead_letter_symbols
    assert fetcher.api_error_counts["timeout"] == 1


def test_fetch_hourly_universe_once_saves_latest_closed_candle(tmp_path, monkeypatch):
    target_hour = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")

    class _Exchange:
        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):
            assert timeframe == "1h"
            assert limit == 1
            if symbol == "B/USDT":
                raise TimeoutError("network timeout")
            return [[int(target_hour.timestamp() * 1000), 1.0, 2.0, 0.5, 1.5, 10.0]]

    fetcher = DataFetcher(exchange=_Exchange(), data_root=str(tmp_path))
    monkeypatch.setattr(fetcher, "has_recent_gap", lambda symbol, days=7: False)

    out = fetcher.fetch_hourly_universe_once(
        ["A/USDT", "B/USDT"],
        max_workers=2,
        target_hour=target_hour,
        check_recent_gaps_days=7,
    )

    assert set(out.keys()) == {"A/USDT"}
    stored = fetcher.ohlcv_store.load("A/USDT")
    assert not stored.empty
    assert pd.Timestamp(stored.index.max()) == target_hour
    assert "B/USDT" in fetcher.dead_letter_symbols
    assert fetcher.api_error_counts["timeout"] == 1


def test_fetch_hourly_universe_once_rejects_stale_closed_candle(tmp_path, monkeypatch):
    target_hour = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    stale_hour = target_hour - pd.Timedelta(hours=1)

    class _Exchange:
        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):
            assert timeframe == "1h"
            assert limit == 1
            return [[int(stale_hour.timestamp() * 1000), 1.0, 2.0, 0.5, 1.5, 10.0]]

    fetcher = DataFetcher(exchange=_Exchange(), data_root=str(tmp_path))
    monkeypatch.setattr(fetcher, "has_recent_gap", lambda symbol, days=7: False)

    out = fetcher.fetch_hourly_universe_once(
        ["A/USDT"],
        max_workers=1,
        target_hour=target_hour,
        check_recent_gaps_days=7,
    )

    assert out == {}
    stored = fetcher.ohlcv_store.load("A/USDT")
    assert stored.empty


def test_fetch_hourly_universe_once_can_skip_recent_gap_backfill(tmp_path, monkeypatch):
    target_hour = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")

    class _Exchange:
        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):
            return [[int(target_hour.timestamp() * 1000), 1.0, 2.0, 0.5, 1.5, 10.0]]

    fetcher = DataFetcher(exchange=_Exchange(), data_root=str(tmp_path))
    gap_checks = []
    backfills = []
    monkeypatch.setattr(
        fetcher, "has_recent_gap", lambda symbol, days=7: gap_checks.append(symbol) or True
    )
    monkeypatch.setattr(
        fetcher, "trigger_gap_backfill", lambda symbol, days=7, backfill_fn=None: backfills.append(symbol)
    )

    fetcher.fetch_hourly_universe_once(
        ["A/USDT"],
        max_workers=1,
        target_hour=target_hour,
        check_recent_gaps_days=0,
    )

    assert gap_checks == []
    assert backfills == []


def test_load_microdata_panel_preserves_saved_orderbook_fields(tmp_path):
    fetcher = DataFetcher(exchange=object(), data_root=str(tmp_path))
    idx = pd.date_range("2026-01-01", periods=2, freq="1h", tz="UTC")
    pd.DataFrame(
        {
            "mid": [1.0, 1.1],
            "best_bid": [0.9, 1.0],
            "best_ask": [1.1, 1.2],
            "bid_qty_1": [10.0, 11.0],
            "ask_qty_1": [12.0, 13.0],
            "cum_bid_qty_l10": [20.0, 21.0],
            "cum_ask_qty_l10": [22.0, 23.0],
            "cum_bid_qty_l20": [30.0, 31.0],
            "cum_ask_qty_l20": [32.0, 33.0],
        },
        index=idx,
    ).to_parquet(fetcher.orderbook_dir / "A_USDT.parquet")

    panel = fetcher._load_microdata_panel(["A/USDT"])

    assert "orderbook_hourly" in panel
    assert "orderbook_best_bid" in panel
    assert "orderbook_cum_bid_qty_l20" in panel
    np.testing.assert_allclose(panel["orderbook_hourly"]["A/USDT"], [1.0, 1.1])
    np.testing.assert_allclose(panel["orderbook_best_bid"]["A/USDT"], [0.9, 1.0])


def test_update_microdata_symbol_uses_perp_exchange_for_spot_funding(
    tmp_path, monkeypatch
):
    class _SpotExchange:
        pass

    class _PerpExchange:
        def fetch_funding_rate_history(self, symbol, since=None, limit=None):
            assert symbol == "A/USDT:USDT"
            return [
                {
                    "timestamp": int(
                        pd.Timestamp("2026-01-01 00:00:00", tz="UTC").timestamp() * 1000
                    ),
                    "fundingRate": 0.0015,
                }
            ]

    monkeypatch.setattr(
        "extreme_price_movements.inference.data_fetcher.make_perp_exchange",
        lambda: _PerpExchange(),
    )
    monkeypatch.setattr(
        "extreme_price_movements.inference.data_fetcher._resolve_perp_symbol",
        lambda exchange, symbol: "A/USDT:USDT",
    )
    monkeypatch.setattr(
        "extreme_price_movements.inference.data_fetcher._compute_missing_hourly_ranges",
        lambda existing_idx, start_ts, end_ts: [
            (
                pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
                pd.Timestamp("2026-01-01 01:00:00", tz="UTC"),
            )
            for _ in range(10)
        ],
    )
    monkeypatch.setattr(
        "extreme_price_movements.inference.data_fetcher._compute_missing_funding_ranges",
        lambda existing_idx, start_ts, end_ts: [
            (
                pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
                pd.Timestamp("2026-01-01 01:00:00", tz="UTC"),
            )
            for _ in range(10)
        ],
    )
    monkeypatch.setattr(
        "extreme_price_movements.inference.data_fetcher.fetch_hourly_orderbook_proxy",
        lambda exchange, symbol, since_ms, until_ms: pd.DataFrame(
            {
                "best_bid": [1.0],
                "best_ask": [1.2],
                "mid": [1.1],
                "bid_qty_1": [10.0],
                "ask_qty_1": [12.0],
                "cum_bid_qty_l10": [20.0],
                "cum_ask_qty_l10": [22.0],
                "cum_bid_qty_l20": [30.0],
                "cum_ask_qty_l20": [32.0],
                "snapshot_ts": [pd.Timestamp("2026-01-01 00:59:00", tz="UTC")],
            },
            index=pd.DatetimeIndex(
                [pd.Timestamp("2026-01-01 00:00:00", tz="UTC")], name="ts"
            ),
        ),
    )

    fetcher = DataFetcher(exchange=_SpotExchange(), data_root=str(tmp_path))
    out = fetcher.update_microdata_symbol("A/USDT", backfill_days=30)

    assert out["orderbook"] is True
    assert out["funding"] is True

    orderbook = pd.read_parquet(fetcher.orderbook_dir / "A_USDT.parquet")
    assert "mid" in orderbook.columns
    np.testing.assert_allclose(orderbook["mid"].to_numpy(), [1.1], rtol=1e-6)

    funding = pd.read_parquet(fetcher.funding_dir / "A_USDT.parquet")
    assert "funding_rate" in funding.columns
    assert len(funding) >= 1
    np.testing.assert_allclose(
        funding["funding_rate"].to_numpy(),
        np.full(len(funding), 0.0015),
        rtol=1e-6,
    )


def test_update_microdata_symbol_skips_missing_perp_funding_without_error(
    tmp_path, monkeypatch
):
    class _SpotExchange:
        pass

    class _PerpExchange:
        def fetch_funding_rate_history(self, symbol, since=None, limit=None):
            raise AssertionError("funding endpoint should not be called")

    logs = []
    monkeypatch.setattr(
        "extreme_price_movements.inference.data_fetcher.make_perp_exchange",
        lambda: _PerpExchange(),
    )
    monkeypatch.setattr(
        "extreme_price_movements.inference.data_fetcher._resolve_perp_symbol",
        lambda exchange, symbol: None,
    )
    monkeypatch.setattr(
        "extreme_price_movements.inference.data_fetcher._compute_missing_hourly_ranges",
        lambda existing_idx, start_ts, end_ts: [],
    )
    monkeypatch.setattr(
        "extreme_price_movements.inference.data_fetcher.tprint",
        lambda msg: logs.append(str(msg)),
    )

    fetcher = DataFetcher(exchange=_SpotExchange(), data_root=str(tmp_path))
    out = fetcher.update_microdata_symbol("A/USDC", backfill_days=30)

    assert out == {"orderbook": False, "funding": False}
    assert "A/USDC" in fetcher._symbols_without_perp_funding
    assert not any("microdata_funding failed" in msg for msg in logs)
