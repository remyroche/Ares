from __future__ import annotations

from pathlib import Path

import pandas as pd

import scripts.run_tp6_sl4_exact170_canonical_consensus as subject


def test_hourly_source_reads_only_requested_year_partitions(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "ohlcv"
    symbol_root = source_root / "symbol=AAA_USDT"
    old = symbol_root / "year=2023" / "old.parquet"
    current = symbol_root / "year=2026" / "current.parquet"
    old.parent.mkdir(parents=True)
    current.parent.mkdir(parents=True)
    old.touch()
    current.touch()
    reads: list[Path] = []

    def fake_read(path: Path) -> pd.DataFrame:
        reads.append(Path(path))
        return pd.DataFrame(
            {"ts": [pd.Timestamp("2026-08-15T00:00:00Z")], "close": [1.0]}
        )

    monkeypatch.setattr(subject, "OHLCV_ROOT", source_root)
    monkeypatch.setattr(subject.pd, "read_parquet", fake_read)
    result = subject._read_hourly_source(
        "AAA_USDT",
        pd.Timestamp("2026-08-14T00:00:00Z"),
        pd.Timestamp("2026-09-01T00:00:00Z"),
    )
    assert result is not None
    assert reads == [current]


def test_hourly_source_does_not_scan_nonoverlapping_partitioned_history(monkeypatch, tmp_path: Path) -> None:
    source_root = tmp_path / "ohlcv"
    old = source_root / "symbol=AAA_USDT" / "year=2023" / "old.parquet"
    old.parent.mkdir(parents=True)
    old.touch()
    reads: list[Path] = []

    def fake_read(path: Path) -> pd.DataFrame:
        reads.append(Path(path))
        raise AssertionError("non-overlapping partition must not be opened")

    monkeypatch.setattr(subject, "OHLCV_ROOT", source_root)
    monkeypatch.setattr(subject.pd, "read_parquet", fake_read)
    result = subject._read_hourly_source(
        "AAA_USDT",
        pd.Timestamp("2026-08-14T00:00:00Z"),
        pd.Timestamp("2026-09-01T00:00:00Z"),
    )
    assert result is None
    assert reads == []


def test_oi_funding_sidecars_use_causal_arrow_time_filter(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "ares"
    oi_root = root / "data_perp/exchanges/krakenfutures/open_interest_hourly"
    funding_root = root / "data_perp/exchanges/krakenfutures/funding_hourly"
    oi_root.mkdir(parents=True)
    funding_root.mkdir(parents=True)
    (oi_root / "AAA_USD_AAA.parquet").touch()
    (funding_root / "AAA_USD_AAA.parquet").touch()
    calls: list[dict[str, object]] = []

    def fake_read(path: Path, **kwargs: object) -> pd.DataFrame:
        calls.append({"path": Path(path), **kwargs})
        field = kwargs["columns"][0]  # type: ignore[index]
        return pd.DataFrame(
            {field: [1.0]}, index=pd.DatetimeIndex([pd.Timestamp("2026-08-14T00:00:00Z")])
        )

    monkeypatch.setattr(subject, "ROOT", root)
    monkeypatch.setattr(subject, "_panel_sidecar_is_quarantined", lambda *_: False)
    monkeypatch.setattr(subject.pd, "read_parquet", fake_read)
    panel: dict[str, pd.DataFrame] = {}
    idx = pd.DatetimeIndex([pd.Timestamp("2026-08-14T00:00:00Z")])
    subject._add_oi_funding_panels(
        panel,
        ["AAA/USD:USD"],
        idx,
        pd.Timestamp("2026-08-14T00:00:00Z"),
        pd.Timestamp("2026-08-14T01:00:00Z"),
    )
    assert len(calls) == 2
    assert all("filters" in call for call in calls)
    assert all(call["filters"][0][0] == "__index_level_0__" for call in calls)  # type: ignore[index]
