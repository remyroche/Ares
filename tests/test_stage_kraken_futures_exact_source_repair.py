from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from extreme_price_movements.data_store import PartitionedOHLCVStore
from scripts.stage_kraken_futures_exact_source_repair import (
    ExactSourceRepairError,
    _parse_response_series,
    derive_scope,
    revalidate_staged_exact_source_patch,
    stage_exact_source_patch,
)


def _write_raw(
    store: PartitionedOHLCVStore, symbol: str, timestamps: list[str]
) -> None:
    index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    store.save_partitioned(
        symbol,
        pd.DataFrame(
            {
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.0,
                "volume": 2.0,
            },
            index=index,
        ),
        defer_compact=True,
    )


def _context(path: Path) -> None:
    pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                [
                    "2026-06-01T00:00:00Z",
                    "2026-06-01T01:00:00Z",
                    "2026-06-01T00:00:00Z",
                ],
                utc=True,
            ),
            "__symbol__": ["AAA/USD:USD", "AAA/USD:USD", "BBB/USD:USD"],
            "side_name": ["short", "short", "short"],
            "candidate_id": ["a0", "a1", "b0"],
            "gmm_representation_available": [0.0, 0.0, 0.0],
        }
    ).to_parquet(path, index=False)


def _scope(tmp_path: Path) -> dict[str, object]:
    raw_root = tmp_path / "raw"
    store = PartitionedOHLCVStore(str(raw_root), "1h")
    _write_raw(store, "AAA/USD:USD", ["2026-06-01T00:00:00Z"])
    _write_raw(
        store,
        "BBB/USD:USD",
        ["2026-06-01T00:00:00Z", "2026-06-01T01:00:00Z"],
    )
    context = tmp_path / "context.parquet"
    _context(context)
    return derive_scope(
        context_path=context,
        raw_root=raw_root,
        start_ts="2026-06-01T00:00:00Z",
        end_ts="2026-06-01T02:00:00Z",
        candidate_start_ts="2026-06-01T00:00:00Z",
        top_n=2,
        expected_unavailable_candidates=3,
        expected_missing_hours=1,
    )


def _scope_with_two_requested_symbols(tmp_path: Path) -> dict[str, object]:
    raw_root = tmp_path / "raw"
    store = PartitionedOHLCVStore(str(raw_root), "1h")
    _write_raw(store, "AAA/USD:USD", ["2026-06-01T00:00:00Z"])
    _write_raw(store, "BBB/USD:USD", ["2026-06-01T00:00:00Z"])
    context = tmp_path / "context.parquet"
    _context(context)
    return derive_scope(
        context_path=context,
        raw_root=raw_root,
        start_ts="2026-06-01T00:00:00Z",
        end_ts="2026-06-01T02:00:00Z",
        candidate_start_ts="2026-06-01T00:00:00Z",
        top_n=2,
        expected_unavailable_candidates=3,
        expected_missing_hours=2,
    )


def test_derive_scope_is_candidate_driven_and_fails_closed_on_bound(
    tmp_path: Path,
) -> None:
    scope = _scope(tmp_path)

    assert scope["counts"] == {
        "context_short_candidates": 3,
        "context_short_unavailable_candidates": 3,
        "scoped_unavailable_candidates": 3,
        "scoped_missing_source_hours": 1,
    }
    assert [item["symbol"] for item in scope["symbols"]] == [
        "AAA/USD:USD",
        "BBB/USD:USD",
    ]
    assert scope["symbols"][0]["missing_source_hours"] == ["2026-06-01T01:00:00+00:00"]

    with pytest.raises(ExactSourceRepairError, match="approved bound"):
        derive_scope(
            context_path=tmp_path / "context.parquet",
            raw_root=tmp_path / "raw",
            start_ts="2026-06-01T00:00:00Z",
            end_ts="2026-06-01T02:00:00Z",
            candidate_start_ts="2026-06-01T00:00:00Z",
            top_n=2,
            expected_unavailable_candidates=4,
            expected_missing_hours=1,
        )


def test_derive_scope_hash_is_deterministic_for_identical_locked_inputs(
    tmp_path: Path,
) -> None:
    first = _scope(tmp_path)
    second = _scope(tmp_path)

    assert first["scope_sha256"] == second["scope_sha256"]


class _Response:
    headers = {
        "content-type": "application/json",
        "date": "Fri, 25 Jul 2026 00:00:00 GMT",
    }

    def __init__(
        self,
        payload: dict[str, object],
        *,
        status_code: int = 200,
        error: Exception | None = None,
    ) -> None:
        self.content = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.status_code = status_code
        self.error = error

    def raise_for_status(self) -> None:
        if self.error is not None:
            raise self.error


class _Session:
    def __init__(
        self, payloads: list[dict[str, object] | _Response | Exception]
    ) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> _Response:
        self.calls.append({"url": url, **kwargs})
        response = self.payloads.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, _Response):
            return response
        return _Response(response)


def test_stage_persists_exact_responses_and_patch_only(tmp_path: Path) -> None:
    scope = _scope(tmp_path)
    original = {
        path.relative_to(tmp_path / "raw"): path.read_bytes()
        for path in (tmp_path / "raw").rglob("*.parquet")
    }
    timestamp = int(pd.Timestamp("2026-06-01T01:00:00Z").value // 10**6)
    session = _Session(
        [
            {
                "candles": [
                    {
                        "time": timestamp,
                        "open": "10",
                        "high": "11",
                        "low": "9",
                        "close": "10.5",
                        "volume": "2",
                    },
                    # An endpoint candle outside the missing set must never be
                    # staged as a replacement or a synthetic insertion.
                    {
                        "time": timestamp - 3_600_000,
                        "open": "10",
                        "high": "11",
                        "low": "9",
                        "close": "10",
                        "volume": "2",
                    },
                ]
            },
        ]
    )

    result = stage_exact_source_patch(
        scope=scope,
        output_dir=tmp_path / "patch",
        session=session,  # type: ignore[arg-type]
    )

    ledger = pd.read_parquet(tmp_path / "patch/accepted_candle_ledger.parquet")
    assert result["baseline_raw_store_mutated"] is False
    assert result["network_retries"] == 0
    assert len(session.calls) == 1
    assert len(ledger) == 1
    assert ledger.loc[0, "symbol"] == "AAA/USD:USD"
    assert ledger.loc[0, "ts"] == pd.Timestamp("2026-06-01T01:00:00Z")
    assert len(list((tmp_path / "patch/endpoint_responses").glob("*.json"))) == 1
    assert original == {
        path.relative_to(tmp_path / "raw"): path.read_bytes()
        for path in (tmp_path / "raw").rglob("*.parquet")
    }


def test_stage_rejects_overwrite_and_never_accepts_invalid_candle(
    tmp_path: Path,
) -> None:
    scope = _scope(tmp_path)
    timestamp = int(pd.Timestamp("2026-06-01T01:00:00Z").value // 10**6)
    session = _Session(
        [
            {
                "candles": [
                    {
                        "time": timestamp,
                        "open": "10",
                        "high": "9",
                        "low": "11",
                        "close": "10",
                        "volume": "1",
                    }
                ]
            },
        ]
    )
    output = tmp_path / "patch"
    stage_exact_source_patch(scope=scope, output_dir=output, session=session)  # type: ignore[arg-type]
    assert pd.read_parquet(output / "accepted_candle_ledger.parquet").empty
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        stage_exact_source_patch(scope=scope, output_dir=output, session=_Session([]))  # type: ignore[arg-type]


def test_stage_records_failed_request_and_continues_without_retry(
    tmp_path: Path,
) -> None:
    scope = _scope_with_two_requested_symbols(tmp_path)
    timestamp = int(pd.Timestamp("2026-06-01T01:00:00Z").value // 10**6)
    session = _Session(
        [
            _Response(
                {"error": "rate limited"},
                status_code=429,
                error=RuntimeError("429 Too Many Requests"),
            ),
            {
                "candles": [
                    {
                        "time": timestamp,
                        "open": "10",
                        "high": "11",
                        "low": "9",
                        "close": "10.5",
                        "volume": "2",
                    }
                ]
            },
        ]
    )

    result = stage_exact_source_patch(
        scope=scope,
        output_dir=tmp_path / "partial-patch",
        session=session,  # type: ignore[arg-type]
    )

    manifest = json.loads(
        (tmp_path / "partial-patch/endpoint_response_manifest.json").read_text()
    )
    records = manifest["responses"]
    assert result["endpoint_responses"]["records"] == 2
    assert len(session.calls) == 2
    assert len(session.calls) <= sum(
        bool(item["missing_source_hours"]) for item in scope["symbols"]
    )
    assert [record["status"] for record in records] == [
        "http_error",
        "inspected_response",
    ]
    assert [record["request_attempts"] for record in records] == [1, 1]
    assert all(record["status"] for record in records)
    assert (
        len(pd.read_parquet(tmp_path / "partial-patch/accepted_candle_ledger.parquet"))
        == 1
    )


def test_response_series_rejects_linked_zero_carry_but_keeps_isolated_no_trade() -> (
    None
):
    start = pd.Timestamp("2026-06-01T00:00:00Z")
    milliseconds = lambda hours: int((start + pd.Timedelta(hours=hours)).value // 10**6)
    candles, invalid, duplicates, rejected_carry_timestamps = _parse_response_series(
        [
            {
                "time": milliseconds(0),
                "open": 10,
                "high": 10,
                "low": 10,
                "close": 10,
                "volume": 0,
            },
            {
                "time": milliseconds(1),
                "open": 10,
                "high": 10,
                "low": 10,
                "close": 10,
                "volume": 0,
            },
            # This is an isolated genuine no-trade candle: its open differs
            # from the preceding carry close, so the repository filter keeps it.
            {
                "time": milliseconds(2),
                "open": 12,
                "high": 12,
                "low": 12,
                "close": 12,
                "volume": 0,
            },
        ],
        start=start,
        end=start + pd.Timedelta(hours=3),
    )

    assert invalid == 0
    assert duplicates == 0
    assert rejected_carry_timestamps == {
        start,
        start + pd.Timedelta(hours=1),
    }
    assert list(candles) == [start + pd.Timedelta(hours=2)]


def test_offline_revalidation_is_bound_to_v1_and_does_not_change_it(
    tmp_path: Path,
) -> None:
    scope = _scope(tmp_path)
    timestamp = int(pd.Timestamp("2026-06-01T01:00:00Z").value // 10**6)
    source_dir = tmp_path / "v1"
    stage_exact_source_patch(
        scope=scope,
        output_dir=source_dir,
        session=_Session(
            [
                {
                    "candles": [
                        {
                            "time": timestamp,
                            "open": "10",
                            "high": "11",
                            "low": "9",
                            "close": "10.5",
                            "volume": "2",
                        }
                    ]
                }
            ]
        ),  # type: ignore[arg-type]
    )
    source_bytes = {
        path.relative_to(source_dir): path.read_bytes()
        for path in source_dir.rglob("*")
        if path.is_file()
    }

    result = revalidate_staged_exact_source_patch(
        source_dir=source_dir,
        output_dir=tmp_path / "revalidated",
    )

    assert result["network_calls"] == 0
    assert result["source_patch"]["scope_sha256"] == scope["scope_sha256"]
    assert result["accepted_candle_ledger"]["rows"] == 1
    assert result["rejected_requested_zero_volume_carry_candles"] == 0
    assert source_bytes == {
        path.relative_to(source_dir): path.read_bytes()
        for path in source_dir.rglob("*")
        if path.is_file()
    }
