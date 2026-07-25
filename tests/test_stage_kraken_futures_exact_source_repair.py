from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from extreme_price_movements.data_store import PartitionedOHLCVStore
from scripts.stage_kraken_futures_exact_source_repair import (
    ExactSourceRepairError,
    derive_scope,
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


class _Response:
    status_code = 200
    headers = {
        "content-type": "application/json",
        "date": "Fri, 25 Jul 2026 00:00:00 GMT",
    }

    def __init__(self, payload: dict[str, object]) -> None:
        self.content = json.dumps(payload, sort_keys=True).encode("utf-8")

    def raise_for_status(self) -> None:
        return None


class _Session:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> _Response:
        self.calls.append({"url": url, **kwargs})
        return _Response(self.payloads.pop(0))


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
