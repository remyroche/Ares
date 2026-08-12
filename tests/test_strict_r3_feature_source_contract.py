"""Regression coverage for the strict-R3 coarse-bar source contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_tp6_sl4_exact170_canonical_consensus.py"
SPEC = importlib.util.spec_from_file_location("strict_r3_exact170_source", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_default_panel_build_never_uses_minute_fallback(monkeypatch) -> None:
    """A missing coarse source stays missing under the canonical contract."""
    start = pd.Timestamp("2026-06-01T00:00:00Z")
    end = pd.Timestamp("2026-06-01T03:00:00Z")
    symbol = "TEST/USD:USD"
    calls = {"minute": 0}

    monkeypatch.setattr(MODULE, "_read_canonical_input_cache", lambda *args: None)
    monkeypatch.setattr(MODULE, "_read_downloaded_15m_hourly", lambda *args: None)
    monkeypatch.setattr(MODULE, "_read_hourly_source", lambda *args: None)

    def _minute(*args, **kwargs):
        calls["minute"] += 1
        raise AssertionError("minute fallback must not be called")

    monkeypatch.setattr(MODULE, "_read_minute_fallback", _minute)
    panel, _ = MODULE._make_panel([symbol], start, end)

    assert calls["minute"] == 0
    assert panel["close"][symbol].isna().all()


def test_legacy_minute_fallback_requires_explicit_opt_in(monkeypatch) -> None:
    """The only path to minute data is an explicit non-canonical request."""
    start = pd.Timestamp("2026-06-01T00:00:00Z")
    end = pd.Timestamp("2026-06-01T03:00:00Z")
    symbol = "TEST/USD:USD"
    index = pd.date_range(start, periods=3, freq="1h", tz="UTC")
    sample = pd.DataFrame(
        {
            "open": [1.0, 1.0, 1.0],
            "high": [1.1, 1.1, 1.1],
            "low": [0.9, 0.9, 0.9],
            "close": [1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0],
        },
        index=index,
    )
    monkeypatch.setattr(MODULE, "_read_canonical_input_cache", lambda *args: None)
    monkeypatch.setattr(MODULE, "_read_downloaded_15m_hourly", lambda *args: None)
    monkeypatch.setattr(MODULE, "_read_hourly_source", lambda *args: None)
    monkeypatch.setattr(MODULE, "_read_minute_fallback", lambda *args, **kwargs: sample)

    panel, _ = MODULE._make_panel(
        [symbol],
        start,
        end,
        allow_minute_fallback=True,
    )

    assert panel["close"][symbol].notna().all()


def test_default_panel_build_does_not_extend_a_stale_coarse_source_with_minutes(
    monkeypatch,
) -> None:
    """A coarse source may be incomplete; canonical strict-R3 does not bridge it."""
    start = pd.Timestamp("2026-06-01T00:00:00Z")
    end = pd.Timestamp("2026-06-01T03:00:00Z")
    symbol = "TEST/USD:USD"
    stale = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
            "volume": [1.0],
        },
        index=pd.DatetimeIndex([start]),
    )
    calls = {"minute": 0}
    monkeypatch.setattr(MODULE, "_read_canonical_input_cache", lambda *args: stale)
    monkeypatch.setattr(MODULE, "_read_downloaded_15m_hourly", lambda *args: None)
    monkeypatch.setattr(MODULE, "_read_hourly_source", lambda *args: None)

    def _minute(*args, **kwargs):
        calls["minute"] += 1
        raise AssertionError("minute tail extension must not be called")

    monkeypatch.setattr(MODULE, "_read_minute_fallback", _minute)
    panel, _ = MODULE._make_panel([symbol], start, end)

    assert calls["minute"] == 0
    assert panel["close"].loc[start, symbol] == 1.0
    assert pd.isna(panel["close"].loc[start + pd.Timedelta(hours=1), symbol])


def test_raw_15m_download_beats_synthetic_shared_mirror(tmp_path, monkeypatch) -> None:
    """The raw downloaded archive repairs a flat shared-cache interval."""
    start = pd.Timestamp("2026-06-01T00:00:00Z")
    end = pd.Timestamp("2026-06-01T01:00:00Z")
    index = pd.date_range(start, periods=4, freq="15min")
    shared = pd.DataFrame(
        {
            "open": [100.0] * 4,
            "high": [100.0] * 4,
            "low": [100.0] * 4,
            "close": [100.0] * 4,
            "volume": [0.0] * 4,
        },
        index=index,
    )
    downloaded = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [101.0, 102.0, 103.0, 104.0],
            "volume": [1.0, 2.0, 3.0, 4.0],
        },
        index=index,
    )
    hf = tmp_path / "hf"
    raw = tmp_path / "raw"
    hf.mkdir()
    raw.mkdir()
    name = "btcusd:usd_15m.parquet"
    shared.to_parquet(hf / name)
    downloaded.to_parquet(raw / name)
    monkeypatch.setattr(MODULE, "HF_15M_ROOT", hf)
    monkeypatch.setattr(MODULE, "RAW_15M_ROOT", raw)

    hourly = MODULE._read_downloaded_15m_hourly("BTC/USD:USD", start, end)

    assert hourly is not None
    assert hourly.loc[start, "close"] == 104.0
    assert hourly.loc[start, "volume"] == 10.0


def test_synthetic_shared_15m_padding_does_not_make_an_hour(monkeypatch, tmp_path) -> None:
    """With no raw replacement, flat zero-volume padding remains unavailable."""
    start = pd.Timestamp("2026-06-01T00:00:00Z")
    end = pd.Timestamp("2026-06-01T01:00:00Z")
    index = pd.date_range(start, periods=4, freq="15min")
    flat = pd.DataFrame(
        {
            "open": [100.0] * 4,
            "high": [100.0] * 4,
            "low": [100.0] * 4,
            "close": [100.0] * 4,
            "volume": [0.0] * 4,
        },
        index=index,
    )
    hf = tmp_path / "hf"
    raw = tmp_path / "raw"
    hf.mkdir()
    raw.mkdir()
    flat.to_parquet(hf / "btcusd:usd_15m.parquet")
    monkeypatch.setattr(MODULE, "HF_15M_ROOT", hf)
    monkeypatch.setattr(MODULE, "RAW_15M_ROOT", raw)

    assert MODULE._read_downloaded_15m_hourly("BTC/USD:USD", start, end) is None


def test_official_hourly_fills_missing_coarse_tail_without_overwriting_15m(
    monkeypatch, tmp_path,
) -> None:
    """Official one-hour candles are a missing-cell fallback, not a rewrite."""
    start = pd.Timestamp("2026-08-12T08:00:00Z")
    end = pd.Timestamp("2026-08-12T10:00:00Z")
    symbol = "BTC/USD:USD"
    index = pd.date_range(start, periods=2, freq="1h")
    official = pd.DataFrame(
        {
            "open": [100.0, 101.0], "high": [102.0, 103.0],
            "low": [99.0, 100.0], "close": [101.0, 102.0],
            "volume": [10.0, 11.0],
        },
        index=index,
    )
    root = tmp_path / "official"
    root.mkdir()
    official.to_parquet(root / "BTC_USD_USD.parquet")
    monkeypatch.setattr(MODULE, "FROZEN_INPUT_BACKFILL_ROOT", root)
    monkeypatch.setattr(MODULE, "_read_canonical_input_cache", lambda *args: None)
    monkeypatch.setattr(MODULE, "_read_downloaded_15m_hourly", lambda *args: None)
    monkeypatch.setattr(MODULE, "_read_hourly_source", lambda *args: None)

    panel, _ = MODULE._make_panel([symbol], start, end)

    assert panel["close"].loc[start, symbol] == 101.0
    assert panel["close"].loc[index[1], symbol] == 102.0


def test_complete_15m_value_precedes_official_hourly(monkeypatch, tmp_path) -> None:
    """The approved 15-minute contract wins when both sources have a bar."""
    start = pd.Timestamp("2026-08-12T08:00:00Z")
    end = pd.Timestamp("2026-08-12T10:00:00Z")
    symbol = "BTC/USD:USD"
    index = pd.date_range(start, periods=2, freq="1h")
    official = pd.DataFrame(
        {
            "open": [100.0, 101.0], "high": [102.0, 103.0],
            "low": [99.0, 100.0], "close": [101.0, 102.0],
            "volume": [10.0, 11.0],
        }, index=index,
    )
    coarse = official.copy()
    coarse.loc[start, "close"] = 111.0
    root = tmp_path / "official"
    root.mkdir()
    official.to_parquet(root / "BTC_USD_USD.parquet")
    monkeypatch.setattr(MODULE, "FROZEN_INPUT_BACKFILL_ROOT", root)
    monkeypatch.setattr(MODULE, "_read_canonical_input_cache", lambda *args: None)
    monkeypatch.setattr(MODULE, "_read_downloaded_15m_hourly", lambda *args: coarse)
    monkeypatch.setattr(MODULE, "_read_hourly_source", lambda *args: None)

    panel, _ = MODULE._make_panel([symbol], start, end)

    assert panel["close"].loc[start, symbol] == 111.0
    assert panel["close"].loc[index[1], symbol] == 102.0


def test_rebound_score_causal_parents_are_generation_dependencies() -> None:
    """Long-only materialisation must not emit an uncomputable selected field."""
    required = {
        "mkt_recovery_from_24h_low_atr",
        "mkt_price_up_oi_down_1h",
        "funding_mean_reversion_after_oi_flush",
    }
    assert required.issubset(set(MODULE.FROZEN_GENERATION_DEPENDENCIES))
