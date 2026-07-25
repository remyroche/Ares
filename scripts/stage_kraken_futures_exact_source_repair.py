#!/usr/bin/env python3
"""Audit and stage one bounded Kraken Futures hourly source-repair patch.

This is deliberately *not* a general backfill runner.  Its default mode is
read-only: it derives the exact candidate-driven scope and prints it.  The
only write mode (``--stage``) makes one HTTP request per scoped symbol, writes
the unmodified endpoint response and an accepted-candle ledger to a new patch
artifact, and never touches the baseline raw store.

The caller must apply the staged patch to a separately cloned raw challenger
and rematerialize a separately cloned feature/context challenger.  No fill,
interpolation, retry, or broad-universe scan is implemented here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import PartitionedOHLCVStore  # noqa: E402
from extreme_price_movements.training_resource_guard import (  # noqa: E402
    TrainingResourceGuard,
    TrainingResourceLimits,
)

SCHEMA = "kraken_futures_exact_source_repair_patch_v1"
SCOPE_SCHEMA = "kraken_futures_exact_source_repair_scope_v1"
DEFAULT_CONTEXT = ROOT / (
    "data_perp/artifacts/"
    "packb_downstream_context_20260725_v2_31_8_frozen_ae_gmm/context.parquet"
)
DEFAULT_RAW_ROOT = ROOT / "data_perp/exchanges/krakenfutures"
DEFAULT_START = "2026-05-01T00:00:00Z"
DEFAULT_END = "2026-07-01T00:00:00Z"
DEFAULT_CANDIDATE_START = "2026-06-01T00:00:00Z"
DEFAULT_TOP_N = 30
DEFAULT_EXPECTED_UNAVAILABLE = 4_227
DEFAULT_EXPECTED_MISSING_HOURS = 6_917
RESPONSE_TIMEOUT_SECONDS = 30.0
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


class ExactSourceRepairError(RuntimeError):
    """Raised when the strictly bounded repair contract cannot be proven."""


@dataclass(frozen=True)
class ScopeSymbol:
    symbol: str
    product_id: str
    unavailable_candidate_count: int
    missing_source_hours: tuple[str, ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_hour(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.floor("h")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fallback_product_id(symbol: str) -> str:
    """Return Kraken Futures' documented perpetual-style id without CCXT I/O."""

    text = str(symbol).strip()
    if "/" not in text:
        raise ExactSourceRepairError(f"invalid canonical perp symbol: {symbol!r}")
    base, quote = text.split("/", 1)
    quote = quote.split(":", 1)[0]
    base = "XBT" if base.upper() == "BTC" else base.upper()
    if not base or not quote:
        raise ExactSourceRepairError(f"invalid canonical perp symbol: {symbol!r}")
    return f"PF_{base}{quote.upper()}"


def _missing_hours(
    store: PartitionedOHLCVStore,
    symbol: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DatetimeIndex:
    existing = store.load(symbol, start_ts=start, end_ts=end - pd.Timedelta(hours=1))
    if existing.empty:
        observed = pd.DatetimeIndex([], tz="UTC")
    else:
        observed = pd.DatetimeIndex(pd.to_datetime(existing.index, utc=True)).floor("h")
        observed = observed[(observed >= start) & (observed < end)].unique()
    expected = pd.date_range(start, end - pd.Timedelta(hours=1), freq="1h", tz="UTC")
    return expected.difference(observed)


def derive_scope(
    *,
    context_path: Path,
    raw_root: Path,
    start_ts: Any = DEFAULT_START,
    end_ts: Any = DEFAULT_END,
    candidate_start_ts: Any = DEFAULT_CANDIDATE_START,
    top_n: int = DEFAULT_TOP_N,
    expected_unavailable_candidates: int | None = DEFAULT_EXPECTED_UNAVAILABLE,
    expected_missing_hours: int | None = DEFAULT_EXPECTED_MISSING_HOURS,
) -> dict[str, Any]:
    """Derive the one permitted scope from frozen short-side candidate context."""

    context_path = Path(context_path)
    if not context_path.is_file():
        raise ExactSourceRepairError(f"context parquet is missing: {context_path}")
    start = _utc_hour(start_ts)
    end = _utc_hour(end_ts)
    candidate_start = _utc_hour(candidate_start_ts)
    if end <= start or candidate_start < start or candidate_start >= end:
        raise ExactSourceRepairError("end timestamp must be after start timestamp")
    if int(top_n) < 1:
        raise ExactSourceRepairError("top_n must be positive")

    columns = [
        "__ts__",
        "__symbol__",
        "side_name",
        "candidate_id",
        "gmm_representation_available",
    ]
    context = pd.read_parquet(context_path, columns=columns)
    context["__ts__"] = pd.to_datetime(context["__ts__"], utc=True, errors="raise")
    short = context.loc[
        context["side_name"].astype(str).str.lower().eq("short")
        & context["__ts__"].ge(candidate_start)
        & context["__ts__"].lt(end)
    ].copy()
    if short.empty:
        raise ExactSourceRepairError("short-side context scope is empty")
    available = pd.to_numeric(
        short["gmm_representation_available"], errors="coerce"
    ).eq(1.0)
    unavailable = short.loc[~available]
    grouped = (
        unavailable.groupby("__symbol__", sort=True)["candidate_id"]
        .size()
        .rename("unavailable_candidate_count")
        .reset_index()
        .sort_values(
            ["unavailable_candidate_count", "__symbol__"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(int(top_n))
    )
    store = PartitionedOHLCVStore(str(raw_root), "1h")
    symbols: list[ScopeSymbol] = []
    for _, row in grouped.iterrows():
        symbol = str(row["__symbol__"])
        missing = _missing_hours(store, symbol, start=start, end=end)
        symbols.append(
            ScopeSymbol(
                symbol=symbol,
                product_id=_fallback_product_id(symbol),
                unavailable_candidate_count=int(row["unavailable_candidate_count"]),
                missing_source_hours=tuple(
                    timestamp.isoformat() for timestamp in missing
                ),
            )
        )
    missing_hours = sum(len(item.missing_source_hours) for item in symbols)
    unavailable_candidates = sum(item.unavailable_candidate_count for item in symbols)
    if expected_unavailable_candidates is not None and unavailable_candidates != int(
        expected_unavailable_candidates
    ):
        raise ExactSourceRepairError(
            "candidate-driven scope differs from the approved bound: "
            f"unavailable_candidates={unavailable_candidates}, "
            f"expected={expected_unavailable_candidates}"
        )
    if expected_missing_hours is not None and missing_hours != int(
        expected_missing_hours
    ):
        raise ExactSourceRepairError(
            "raw missing-hour scope differs from the approved bound: "
            f"missing_hours={missing_hours}, expected={expected_missing_hours}"
        )
    payload: dict[str, Any] = {
        "schema": SCOPE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "context": {"path": str(context_path), "sha256": _sha256_file(context_path)},
        "raw_root": str(Path(raw_root)),
        "window": {"start_ts": start.isoformat(), "end_ts_exclusive": end.isoformat()},
        "candidate_window": {
            "start_ts": candidate_start.isoformat(),
            "end_ts_exclusive": end.isoformat(),
        },
        "selection": {
            "side": "short",
            "availability_feature": "gmm_representation_available",
            "selection": "largest_unavailable_candidate_count_desc_symbol_asc",
            "top_n": int(top_n),
        },
        "counts": {
            "context_short_candidates": int(len(short)),
            "context_short_unavailable_candidates": int(len(unavailable)),
            "scoped_unavailable_candidates": int(unavailable_candidates),
            "scoped_missing_source_hours": int(missing_hours),
        },
        "symbols": [
            {
                **asdict(item),
                "missing_source_hours": list(item.missing_source_hours),
            }
            for item in symbols
        ],
        "no_synthetic_fill": True,
        "one_pass_only": True,
    }
    payload["scope_sha256"] = _sha256_bytes(_canonical_json(payload).encode("utf-8"))
    return payload


def _parse_exact_candle(
    candle: Mapping[str, Any], *, start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, Any] | None:
    try:
        raw_time = float(candle.get("time"))
        if not math.isfinite(raw_time):
            return None
        # Kraken's chart response uses Unix milliseconds. Reject a non-hour-aligned
        # value rather than silently moving it into an adjacent decision hour.
        timestamp = pd.to_datetime(int(raw_time), unit="ms", utc=True)
        if timestamp != timestamp.floor("h") or timestamp < start or timestamp >= end:
            return None
        values = {column: float(candle.get(column)) for column in OHLCV_COLUMNS}
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in values.values()):
        return None
    if (
        values["open"] <= 0.0
        or values["high"] <= 0.0
        or values["low"] <= 0.0
        or values["close"] <= 0.0
        or values["volume"] < 0.0
        or values["low"] > min(values["open"], values["close"])
        or values["high"] < max(values["open"], values["close"])
        or values["low"] > values["high"]
        or (values["volume"] == 0.0 and values["open"] != values["close"])
    ):
        return None
    return {"ts": timestamp, **values}


def _request_exact_chart(
    session: requests.Session,
    *,
    product_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[bytes, Mapping[str, Any], dict[str, Any]]:
    url = f"https://futures.kraken.com/api/charts/v1/trade/{product_id}/1h"
    params = {
        "from": int(start.value // 10**9),
        "to": int(end.value // 10**9),
    }
    # Do not call the repository's retry wrapper: exactly one request is the
    # repair contract, even when an exchange response is transiently bad.
    response = session.get(
        url,
        params=params,
        timeout=RESPONSE_TIMEOUT_SECONDS,
        headers={"User-Agent": "Ares-exact-source-repair/1"},
    )
    response.raise_for_status()
    raw = bytes(response.content)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExactSourceRepairError("Kraken chart response is not valid JSON") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("candles"), list):
        raise ExactSourceRepairError("Kraken chart response has no candle list")
    metadata = {
        "url": url,
        "params": params,
        "status_code": int(response.status_code),
        "response_sha256": _sha256_bytes(raw),
        "response_bytes": len(raw),
        "response_headers": {
            key: value
            for key, value in dict(response.headers).items()
            if str(key).lower() in {"date", "content-type", "etag", "last-modified"}
        },
    }
    return raw, payload, metadata


def stage_exact_source_patch(
    *,
    scope: Mapping[str, Any],
    output_dir: Path,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Persist exact endpoint responses and a patch ledger without raw-store writes."""

    if scope.get("schema") != SCOPE_SCHEMA or not bool(scope.get("one_pass_only")):
        raise ExactSourceRepairError("a canonical one-pass repair scope is required")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite repair patch artifact: {output_dir}"
        )
    symbols = scope.get("symbols")
    if not isinstance(symbols, list) or not symbols:
        raise ExactSourceRepairError("repair scope has no symbols")
    window = scope.get("window", {})
    start = _utc_hour(window.get("start_ts"))
    end = _utc_hour(window.get("end_ts_exclusive"))
    if end <= start:
        raise ExactSourceRepairError("repair scope has an invalid window")
    stage = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    responses_dir = stage / "endpoint_responses"
    responses_dir.mkdir()
    guard = TrainingResourceGuard(
        limits=TrainingResourceLimits(
            min_free_ram_bytes=2 * 1024**3,
            max_process_rss_bytes=12 * 1024**3,
            min_free_disk_bytes=10 * 1024**3,
            check_interval_seconds=1.0,
        ),
        disk_path=output_dir.parent,
        telemetry_path=stage / "training_resource_telemetry.jsonl",
    )
    http = session or requests.Session()
    accepted_rows: list[dict[str, Any]] = []
    response_records: list[dict[str, Any]] = []
    try:
        guard.preflight("exact_source_repair:preflight")
        _write_json(stage / "scope.json", dict(scope))
        for ordinal, item in enumerate(symbols, start=1):
            if not isinstance(item, Mapping):
                raise ExactSourceRepairError(
                    "repair scope has a malformed symbol entry"
                )
            symbol = str(item.get("symbol") or "")
            product_id = str(item.get("product_id") or "")
            requested = {
                _utc_hour(value) for value in item.get("missing_source_hours", [])
            }
            if not symbol or not product_id:
                raise ExactSourceRepairError(
                    "repair scope has an empty symbol/product entry"
                )
            # A deterministic top-N candidate scope can include a complete
            # control symbol after tie-breaking.  It receives no endpoint
            # request; the zero is preserved in the response manifest.
            if not requested:
                response_records.append(
                    {
                        "ordinal": ordinal,
                        "symbol": symbol,
                        "product_id": product_id,
                        "requested_missing_hours": 0,
                        "accepted_candles": 0,
                        "status": "skipped_complete_local_window",
                    }
                )
                continue
            guard.checkpoint(f"exact_source_repair:{ordinal}:before_request")
            raw, payload, record = _request_exact_chart(
                http, product_id=product_id, start=start, end=end
            )
            filename = (
                f"{ordinal:03d}_{product_id}_{record['response_sha256'][:16]}.json"
            )
            response_path = responses_dir / filename
            response_path.write_bytes(raw)
            candles: dict[pd.Timestamp, dict[str, Any]] = {}
            rejected_duplicate_timestamps = 0
            rejected_invalid_candles = 0
            for raw_candle in payload["candles"]:
                if not isinstance(raw_candle, Mapping):
                    rejected_invalid_candles += 1
                    continue
                parsed = _parse_exact_candle(raw_candle, start=start, end=end)
                if parsed is None:
                    rejected_invalid_candles += 1
                    continue
                timestamp = parsed["ts"]
                if timestamp in candles:
                    rejected_duplicate_timestamps += 1
                    continue
                candles[timestamp] = parsed
            accepted_for_symbol = 0
            for timestamp in sorted(requested.intersection(candles)):
                candle = candles[timestamp]
                accepted_rows.append(
                    {
                        "symbol": symbol,
                        "product_id": product_id,
                        "ts": timestamp,
                        **{
                            column: np.float32(candle[column])
                            for column in OHLCV_COLUMNS
                        },
                        "endpoint_response_sha256": record["response_sha256"],
                        "endpoint_response_file": str(
                            Path("endpoint_responses") / filename
                        ),
                    }
                )
                accepted_for_symbol += 1
            response_records.append(
                {
                    "ordinal": ordinal,
                    "symbol": symbol,
                    "product_id": product_id,
                    "requested_missing_hours": len(requested),
                    "accepted_candles": accepted_for_symbol,
                    "returned_candles": len(payload["candles"]),
                    "valid_unique_candles": len(candles),
                    "rejected_invalid_candles": rejected_invalid_candles,
                    "rejected_duplicate_timestamps": rejected_duplicate_timestamps,
                    "response_file": str(Path("endpoint_responses") / filename),
                    **record,
                }
            )
            guard.checkpoint(f"exact_source_repair:{ordinal}:response_persisted")
        ledger_columns = [
            "symbol",
            "product_id",
            "ts",
            *OHLCV_COLUMNS,
            "endpoint_response_sha256",
            "endpoint_response_file",
        ]
        ledger = pd.DataFrame(accepted_rows, columns=ledger_columns)
        if not ledger.empty:
            ledger["ts"] = pd.to_datetime(ledger["ts"], utc=True, errors="raise")
            if ledger.duplicated(["symbol", "ts"]).any():
                raise ExactSourceRepairError(
                    "accepted ledger contains duplicate source candles"
                )
            ledger = ledger.sort_values(["symbol", "ts"], kind="mergesort")
        ledger_path = stage / "accepted_candle_ledger.parquet"
        ledger.to_parquet(
            ledger_path, index=False, compression="zstd", compression_level=5
        )
        responses_path = stage / "endpoint_response_manifest.json"
        _write_json(responses_path, {"responses": response_records})
        result = {
            "schema": SCHEMA,
            "status": "STAGED_EXACT_SOURCE_PATCH_NOT_APPLIED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": {
                "sha256": str(scope["scope_sha256"]),
                "path": str(output_dir / "scope.json"),
            },
            "baseline_raw_store_mutated": False,
            "one_pass_only": True,
            "network_retries": 0,
            "synthetic_fill": False,
            "accepted_candle_ledger": {
                "path": str(output_dir / ledger_path.name),
                "sha256": _sha256_file(ledger_path),
                "rows": int(len(ledger)),
            },
            "endpoint_responses": {
                "manifest_path": str(output_dir / responses_path.name),
                "manifest_sha256": _sha256_file(responses_path),
                "records": len(response_records),
            },
            "resource_guard": {
                "telemetry": str(output_dir / "training_resource_telemetry.jsonl"),
                "limits": asdict(guard.limits),
            },
            "next_step": (
                "apply only this accepted ledger to a separately cloned raw challenger; "
                "do not mutate the baseline raw store"
            ),
        }
        _write_json(stage / "manifest.json", result)
        guard.checkpoint("exact_source_repair:complete")
        os.replace(stage, output_dir)
        return result
    except BaseException:
        # Preserve the baseline regardless of an endpoint failure.  The hidden
        # stage is intentionally retained so already received raw responses are
        # available for diagnosis, but it is never a canonical patch artifact.
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--start-ts", default=DEFAULT_START)
    parser.add_argument("--end-ts", default=DEFAULT_END)
    parser.add_argument("--candidate-start-ts", default=DEFAULT_CANDIDATE_START)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument(
        "--expected-unavailable-candidates",
        type=int,
        default=DEFAULT_EXPECTED_UNAVAILABLE,
    )
    parser.add_argument(
        "--expected-missing-hours", type=int, default=DEFAULT_EXPECTED_MISSING_HOURS
    )
    parser.add_argument(
        "--stage",
        action="store_true",
        help="perform the one-pass endpoint audit and write a patch artifact",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    scope = derive_scope(
        context_path=args.context,
        raw_root=args.raw_root,
        start_ts=args.start_ts,
        end_ts=args.end_ts,
        candidate_start_ts=args.candidate_start_ts,
        top_n=args.top_n,
        expected_unavailable_candidates=args.expected_unavailable_candidates,
        expected_missing_hours=args.expected_missing_hours,
    )
    if not args.stage:
        print(json.dumps(scope, indent=2, sort_keys=True))
        return 0
    if args.output_dir is None:
        raise ExactSourceRepairError("--stage requires an explicit --output-dir")
    print(
        json.dumps(
            stage_exact_source_patch(scope=scope, output_dir=args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
