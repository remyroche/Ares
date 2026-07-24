from __future__ import annotations

import inspect

import pandas as pd

from extreme_price_movements.inference import run_inference


def _bars(start: str, periods: int) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
        },
        index=index,
    )


def test_live_policy_1m_fetches_only_missing_tail_and_rereads_store(monkeypatch):
    cached_before = _bars("2026-07-18 10:00", 2)
    fetched = _bars("2026-07-18 10:02", 2)
    cached_after = pd.concat([cached_before, fetched])
    reads = iter([cached_before, cached_after])
    observed: dict[str, object] = {}

    def fake_read(data_root, symbol, *, start, end):
        observed.setdefault("reads", []).append((data_root, symbol, start, end))
        return next(reads)

    def fake_fetch(exchange, symbol, timeframe, start, end, *, use_cache):
        observed["fetch"] = (exchange, symbol, timeframe, start, end, use_cache)
        return fetched

    def fake_append(data_root, symbol, frame):
        observed["append"] = (data_root, symbol, frame.copy())
        return {"appended_rows": len(frame)}

    monkeypatch.setattr(run_inference, "read_kraken_execution_1m", fake_read)
    monkeypatch.setattr(
        run_inference.hf_data_loader, "fetch_specific_period", fake_fetch
    )
    monkeypatch.setattr(
        run_inference, "append_missing_kraken_execution_1m", fake_append
    )

    result = run_inference._load_live_policy_bars(
        cfg={"data_root": "data_perp"},
        exchange=object(),
        symbol="BTC/USD:USD",
        timeframe="1m",
        start=pd.Timestamp("2026-07-18 10:00", tz="UTC"),
        end=pd.Timestamp("2026-07-18 10:03", tz="UTC"),
    )

    _, symbol, timeframe, fetch_start, fetch_end, use_cache = observed["fetch"]
    assert symbol == "BTC/USD:USD"
    assert timeframe == "1m"
    assert fetch_start == pd.Timestamp("2026-07-18 10:02", tz="UTC")
    assert fetch_end == pd.Timestamp("2026-07-18 10:03", tz="UTC")
    assert use_cache is False
    assert observed["append"][0:2] == ("data_perp", "BTC/USD:USD")
    pd.testing.assert_frame_equal(result, cached_after)
    assert len(observed["reads"]) == 2


def test_live_policy_1m_uses_complete_cached_window_without_fetch(monkeypatch):
    cached = _bars("2026-07-18 10:00", 4)
    monkeypatch.setattr(
        run_inference,
        "read_kraken_execution_1m",
        lambda *args, **kwargs: cached,
    )

    def fail_fetch(*args, **kwargs):
        raise AssertionError("complete canonical store window must not be fetched again")

    monkeypatch.setattr(
        run_inference.hf_data_loader, "fetch_specific_period", fail_fetch
    )

    result = run_inference._load_live_policy_bars(
        cfg={"data_root": "data_perp"},
        exchange=object(),
        symbol="BTC/USD:USD",
        timeframe="1m",
        start=pd.Timestamp("2026-07-18 10:00", tz="UTC"),
        end=pd.Timestamp("2026-07-18 10:03", tz="UTC"),
    )

    pd.testing.assert_frame_equal(result, cached)


def test_policy_monitor_defaults_to_ohlcv_without_ticker_sentinel():
    parameter = inspect.signature(
        run_inference._monitor_active_position_price_action
    ).parameters["include_executable_sentinel"]

    assert parameter.default is False


def test_challenger_monitor_does_not_call_executable_ticker_sentinel(monkeypatch):
    class EndMonitor(Exception):
        pass

    observed: dict[str, object] = {}

    def fail_sentinel(*args, **kwargs):
        raise AssertionError("ticker sentinel must not run in the policy monitor")

    def fake_policy_monitor(*args, **kwargs):
        observed["include_executable_sentinel"] = kwargs.get(
            "include_executable_sentinel"
        )
        return {}

    def stop_after_first_iteration(*args, **kwargs):
        raise EndMonitor

    class Executor:
        exchange = object()

    monkeypatch.setattr(
        run_inference, "_monitor_executable_stop_sentinel_only", fail_sentinel
    )
    monkeypatch.setattr(
        run_inference, "_monitor_active_position_price_action", fake_policy_monitor
    )
    monkeypatch.setattr(run_inference.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(run_inference.time, "sleep", stop_after_first_iteration)

    try:
        run_inference.run_challenger_monitor(
            [], None, None, Executor(), None, {}, 60, 0
        )
    except EndMonitor:
        pass

    assert observed["include_executable_sentinel"] is False


def test_oco_policy_does_not_fetch_ticker_for_policy_decisions(monkeypatch):
    class Executor:
        mode = "live"

        def get_simple_policy_stop_params(self, bucket_key):
            return {}

        def update_position_policy_state(self, symbol, **kwargs):
            self.update = kwargs
            return {}

    executor = Executor()
    position = {
        "side": "long",
        "entry_price": 100.0,
        "policy_bar_minutes": 1,
        "stop_price": 90.0,
        "bucket_key": "long__default",
    }

    monkeypatch.setattr(
        run_inference,
        "_fetch_live_closeable_price",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("policy evaluation must not fetch a ticker")
        ),
    )
    monkeypatch.setattr(
        run_inference,
        "compute_simple_policy_stop_decision",
        lambda **kwargs: type(
            "Decision",
            (),
            {
                "should_exit": False,
                "peak_price": 101.0,
                "mfe": 0.01,
                "mae": 0.0,
                "stop_price": 90.0,
                "reason": "original_stop_loss",
                "reason_detail": "unchanged",
                "should_replace": False,
            },
        )(),
    )

    run_inference._evaluate_oco_policy(
        "BTC/USD:USD",
        position,
        _bars("2026-07-18 10:00", 1),
        executor,
    )

    assert executor.update["current_price"] == 100.5
    assert executor.update["current_price_source"] == "trade_1m_close"
