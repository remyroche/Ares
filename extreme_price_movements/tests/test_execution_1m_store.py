from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from extreme_price_movements.data_store import (
    append_missing_kraken_execution_1m,
    canonical_kraken_execution_1m_root,
)
from extreme_price_movements.scripts.live_closed_trade_exit_replay import (
    _candidate_execution_1m_dirs,
    _summarise,
)
from scripts.download_policy_execution_1m import main as download_execution_1m_main


def _candles(*, close: float = 101.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.date_range("2026-07-10T10:00:00Z", periods=2, freq="1min"),
            "open": [100.0, 100.0],
            "high": [102.0, 102.0],
            "low": [99.0, 99.0],
            "close": [close, close],
            "volume": [10.0, 11.0],
        }
    )


def test_execution_1m_root_is_canonical_and_idempotent(tmp_path):
    root = tmp_path / "data_perp"
    expected = root / "exchanges" / "krakenfutures" / "execution_1m"

    assert canonical_kraken_execution_1m_root(root) == expected
    assert canonical_kraken_execution_1m_root(expected.parent) == expected
    assert canonical_kraken_execution_1m_root(expected) == expected
    assert _candidate_execution_1m_dirs(root, "BTC/USD:USD") == [
        expected / "ohlcv" / "symbol=BTC_USD:USD"
    ]


def test_execution_1m_append_is_immutable_and_ignores_exact_duplicates(tmp_path):
    root = tmp_path / "data_perp"
    first = append_missing_kraken_execution_1m(root, "BTC/USD:USD", _candles())
    parts = sorted(
        canonical_kraken_execution_1m_root(root).glob("ohlcv/symbol=BTC_USD:USD/year=*/*.parquet")
    )

    second = append_missing_kraken_execution_1m(root, "BTC/USD:USD", _candles())

    assert first["appended_rows"] == 2
    assert second["appended_rows"] == 0
    assert second["duplicate_rows"] == 2
    assert parts == sorted(
        canonical_kraken_execution_1m_root(root).glob("ohlcv/symbol=BTC_USD:USD/year=*/*.parquet")
    )
    assert all(path.name.startswith("part-") for path in parts)
    assert not list(canonical_kraken_execution_1m_root(root).glob("**/compact-*.parquet"))


def test_execution_1m_append_rejects_conflicts_and_off_grid_rows(tmp_path):
    root = tmp_path / "data_perp"
    append_missing_kraken_execution_1m(root, "BTC/USD:USD", _candles())

    with pytest.raises(ValueError, match="conflicting execution_1m append"):
        append_missing_kraken_execution_1m(root, "BTC/USD:USD", _candles(close=101.5))

    invalid = _candles().iloc[:1].copy()
    invalid.loc[:, "ts"] = pd.Timestamp("2026-07-10T10:00:01Z")
    with pytest.raises(ValueError, match="UTC minute grid"):
        append_missing_kraken_execution_1m(root, "BTC/USD:USD", invalid)


def test_strict_exit_parity_rejects_logged_exit_only_evidence():
    results = pd.DataFrame(
        [
            {
                "symbol": "BTC/USD:USD",
                "entry_time": "2026-07-10T10:00:00Z",
                "exit_time": "2026-07-10T10:01:00Z",
                "bar_source": "execution_1m_cache",
                "replay_hit": True,
                "live_exit_reason_detail": "original_stop_loss",
                "replay_exit_reason": "original_stop_loss",
                "replay_exit_price_vs_live_bps": 0.0,
                "replay_vs_live_exit_status": "logged_live_exchange_stop_fill",
            }
        ]
    )

    assert _summarise(results, strict_execution_1m=True)["exit_parity_status"] == "fail"


def test_execution_1m_downloader_resolves_only_the_canonical_root(tmp_path, monkeypatch):
    data_root = tmp_path / "data_perp"
    candidates = tmp_path / "candidates.parquet"
    manifest = tmp_path / "manifest.json"
    pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-07-10T10:00:00Z")],
            "symbol": ["BTC/USD:USD"],
        }
    ).to_parquet(candidates)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "download_policy_execution_1m.py",
            "--candidates",
            str(candidates),
            "--manifest",
            str(manifest),
            "--data-root",
            str(data_root),
            "--verify-only",
        ],
    )

    assert download_execution_1m_main() == 2  # Empty cache is reported incomplete.
    payload = json.loads(manifest.read_text())
    assert payload["store_root"] == str(canonical_kraken_execution_1m_root(data_root))
