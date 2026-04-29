import pandas as pd

from extreme_price_movements.inference import config as inference_config
from extreme_price_movements.inference.data_fetcher import DataFetcher


class _FakeExchange:
    def load_markets(self):
        return {
            "BTC/USDT": {"quote": "USDT", "active": True, "margin": True},
            "ETH/USDT": {
                "quote": "USDT",
                "active": True,
                "info": {"isMarginTradingAllowed": True},
            },
            "XRP/BTC": {"quote": "BTC", "active": True, "margin": True},
            "DOGE/USDT": {"quote": "USDT", "active": False, "margin": True},
            "SOL/USDT": {"quote": "USDT", "active": True, "margin": False},
        }


def test_get_margin_universe_uses_exchange_market_metadata(monkeypatch):
    monkeypatch.setattr(inference_config, "_MARGIN_UNIVERSE_CACHE", None)
    monkeypatch.setattr(inference_config, "_MARGIN_UNIVERSE_CACHE_DAY", None)
    symbols = inference_config.get_margin_universe(exchange=_FakeExchange())
    assert symbols == ["BTC/USDT", "ETH/USDT"]


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
                "NEW/USDT": {
                    "quote": "USDT",
                    "active": True,
                    "margin": True,
                    "info": {"onboardDate": now_ms},
                },
            }

    monkeypatch.setattr(inference_config, "_MARGIN_UNIVERSE_CACHE", None)
    monkeypatch.setattr(inference_config, "_MARGIN_UNIVERSE_CACHE_DAY", None)
    symbols = inference_config.get_margin_universe(exchange=_Exchange())
    assert symbols == ["OLD/USDT"]


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
                "BTC/USDT": {"quote": "USDT", "active": True, "margin": True},
                "ETH/USDT": {"quote": "USDT", "active": True, "margin": True},
            }

    monkeypatch.setattr(inference_config, "_MARGIN_UNIVERSE_CACHE", None)
    monkeypatch.setattr(inference_config, "_MARGIN_UNIVERSE_CACHE_DAY", None)
    out = inference_config.resolve_inference_universes(
        _Exchange(), data_root=str(data_root), run_id="run1"
    )

    assert out["download_symbols"] == ["BTC/USDT", "ETH/USDT"]
    assert out["tradable_symbols"] == ["ETH/USDT"]
    assert out["trained_symbols"] == ["ETH/USDT"]


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
        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
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
