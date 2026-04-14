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
    symbols = inference_config.get_margin_universe(exchange=_FakeExchange())
    assert symbols == ["BTC/USDT", "ETH/USDT"]


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
