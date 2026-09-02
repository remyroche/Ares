#!/usr/bin/env python3
"""Resumably download native Kraken Futures L2 data for a stop replay.

The Kraken Futures historical ``book_snapshot`` channel anchors a book at the
start of a UTC day; it is *not* a periodic fresh snapshot feed.  A valid
completed-minute VWAP replay therefore needs the day-start snapshot plus every
subsequent native ``book`` delta through each relevant position's exit.  This
utility downloads precisely that sequence, batch-requested by timestamp across
the selected markets.  It is not an OHLCV-derived proxy and does not query
outcomes or change any live runtime state.

The durable SQLite task ledger makes retries idempotent.  Raw provider
responses are stored by UTC date as gzip-compressed JSONL together with the
requested timestamp and market IDs, so coverage can be audited independently
of the replay implementation.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import gzip
import hashlib
import json
import sqlite3
import ssl
import sys
import threading
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import certifi


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / (
    "data_perp/artifacts/strict_r3_exact_1m_rich_matched_attribution_2025_2026_"
    "20260817_v5/exact_1m_rich_v1_decision_portfolio_decisions.parquet"
)
DEFAULT_OUT = ROOT / "data_perp/exchanges/krakenfutures/tardis_l2_stop_replay_2026_v2"
TARDIS_URL = "https://api.tardis.dev/v1/data-feeds/cryptofacilities"
TARDIS_COVERAGE_URL = "https://api.tardis.dev/v1/exchanges/cryptofacilities"
BTC_ALIAS = {"BTC": "XBT"}
TASK_SCHEMA = "kraken_futures_l2_daily_anchor_plus_deltas_stop_replay_v2"
TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_path(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _iso(value: pd.Timestamp) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _market_id(symbol: str) -> str:
    base = str(symbol).split("/")[0].strip().upper()
    return f"PF_{BTC_ALIAS.get(base, base)}USD"


def _coverage_markets(timeout: float) -> set[str]:
    request = Request(TARDIS_COVERAGE_URL, headers={"Accept": "application/json"})
    with urlopen(request, timeout=timeout, context=TLS_CONTEXT) as response:  # nosec B310: fixed HTTPS
        payload = json.loads(response.read().decode("utf-8"))
    return {str(row["id"]).upper() for row in payload.get("availableSymbols", [])}


def _load_tasks(
    ledger: Path,
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    available_markets: set[str],
    exit_reasons: set[str] | None,
) -> tuple[list[tuple[str, tuple[str, ...]]], dict[str, int]]:
    columns = [
        "timestamp", "symbol", "accepted", "position_exit_timestamp",
        "candidate_id", "position_exit_reason",
    ]
    rows = pd.read_parquet(ledger, columns=columns)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows["position_exit_timestamp"] = pd.to_datetime(
        rows["position_exit_timestamp"], utc=True, errors="coerce",
    )
    rows = rows.loc[
        rows["accepted"].fillna(False).astype(bool)
        & rows["timestamp"].dt.year.eq(2026)
        & rows["position_exit_timestamp"].notna()
    ].copy()
    if start is not None:
        rows = rows.loc[rows["timestamp"].ge(start)]
    if end is not None:
        rows = rows.loc[rows["timestamp"].lt(end)]
    if exit_reasons is not None:
        rows = rows.loc[rows["position_exit_reason"].astype(str).isin(exit_reasons)]
    required_by_day: dict[pd.Timestamp, dict[str, pd.Timestamp]] = defaultdict(dict)
    missing_market_rows = 0
    for row in rows.itertuples(index=False):
        market = _market_id(row.symbol)
        if market not in available_markets:
            missing_market_rows += 1
            continue
        first_completed = _utc(row.timestamp).floor("min") + pd.Timedelta(minutes=1)
        last_completed = _utc(row.position_exit_timestamp).floor("min")
        if first_completed > last_completed:
            continue
        day = first_completed.normalize()
        while day <= last_completed.normalize():
            day_end = day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
            cutoff = min(last_completed, day_end)
            prior = required_by_day[day].get(market)
            required_by_day[day][market] = cutoff if prior is None else max(prior, cutoff)
            day += pd.Timedelta(days=1)
    tasks: list[tuple[str, tuple[str, ...]]] = []
    total_market_minutes = 0
    for day, market_cutoffs in sorted(required_by_day.items()):
        final_minute = max(market_cutoffs.values())
        current = day
        while current <= final_minute:
            markets = tuple(sorted(
                market for market, cutoff in market_cutoffs.items() if current <= cutoff
            ))
            if markets:
                tasks.append((_iso(current), markets))
                total_market_minutes += len(markets)
            current += pd.Timedelta(minutes=1)
    audit = {
        "accepted_position_rows": int(len(rows)),
        "accepted_symbols": int(rows["symbol"].nunique()),
        "missing_tardis_market_rows": int(missing_market_rows),
        "l2_delta_tasks": int(len(tasks)),
        "market_minute_deltas": int(total_market_minutes),
        "utc_day_anchors": int(len(required_by_day)),
    }
    return tasks, audit


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS task (
            request_ts TEXT NOT NULL,
            markets_json TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            response_sha256 TEXT,
            response_bytes INTEGER,
            received_markets_json TEXT,
            response_path TEXT,
            error TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (request_ts, markets_json)
        )
        """
    )
    connection.commit()
    return connection


def _seed_tasks(connection: sqlite3.Connection, tasks: list[tuple[str, tuple[str, ...]]]) -> None:
    now = datetime.now(UTC).isoformat()
    connection.executemany(
        """
        INSERT OR IGNORE INTO task (request_ts, markets_json, status, updated_at)
        VALUES (?, ?, 'pending', ?)
        """,
        [(timestamp, json.dumps(markets, separators=(",", ":")), now) for timestamp, markets in tasks],
    )
    connection.commit()


def _pending_tasks(connection: sqlite3.Connection, limit: int | None) -> list[tuple[str, tuple[str, ...]]]:
    sql = "SELECT request_ts, markets_json FROM task WHERE status != 'complete' ORDER BY request_ts"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return [(row[0], tuple(json.loads(row[1]))) for row in connection.execute(sql)]


def _request_book_delta(
    request_ts: str,
    markets: tuple[str, ...],
    *,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    timestamp = _utc(request_ts)
    next_minute = timestamp + pd.Timedelta(minutes=1)
    filters = [{"channel": "book", "symbols": list(markets)}]
    needs_anchor = timestamp == timestamp.normalize()
    if needs_anchor:
        filters.append({"channel": "book_snapshot", "symbols": list(markets)})
    query = urlencode({
        "from": _iso(timestamp), "to": _iso(next_minute),
        "filters": json.dumps(filters, separators=(",", ":")), "offset": 0,
    })
    request = Request(
        f"{TARDIS_URL}?{query}",
        headers={"Accept": "application/json", "User-Agent": "Ares-L2-Replay/1.0"},
    )
    failure = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=timeout, context=TLS_CONTEXT) as response:  # nosec B310: fixed HTTPS
                body = response.read()
                headers = {key.lower(): value for key, value in response.headers.items()}
            received = []
            for line in body.decode("utf-8", errors="replace").splitlines():
                try:
                    message = json.loads(line.split(" ", 1)[1])
                except (IndexError, json.JSONDecodeError):
                    continue
                if message.get("feed") == "book_snapshot":
                    received.append(str(message.get("product_id") or "").upper())
            return {
                "ok": True, "request_ts": request_ts, "markets": markets,
                "attempts": attempt, "body": body,
                "received_anchor_markets": sorted(set(received)),
                "needs_anchor": needs_anchor,
                "headers": {key: headers.get(key) for key in ("x-name", "x-slice-size")},
            }
        except (HTTPError, URLError, TimeoutError) as error:
            failure = f"{type(error).__name__}: {error}"
            if isinstance(error, HTTPError) and error.code not in {429, 500, 502, 503, 504}:
                break
            time.sleep(min(15.0, 0.5 * (2 ** (attempt - 1))))
    return {
        "ok": False, "request_ts": request_ts, "markets": markets,
        "attempts": retries, "error": failure or "unknown request failure",
    }


def _append_response(out: Path, result: dict[str, Any], lock: threading.Lock) -> str:
    timestamp = _utc(result["request_ts"])
    relative = Path(f"date={timestamp:%Y-%m-%d}") / "book_delta_requests.jsonl.gz"
    target = out / relative
    payload = {
        "schema": TASK_SCHEMA,
        "request_ts": result["request_ts"],
        "markets": list(result["markets"]),
        "received_anchor_markets": result["received_anchor_markets"],
        "needs_daily_anchor": result["needs_anchor"],
        "provider_headers": result["headers"],
        "response_sha256": _sha_bytes(result["body"]),
        "response_utf8": result["body"].decode("utf-8", errors="replace"),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with gzip.open(target, "at", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return str(relative)


def _run_download(
    connection: sqlite3.Connection,
    *,
    out: Path,
    workers: int,
    timeout: float,
    retries: int,
    limit: int | None,
) -> dict[str, int]:
    pending = _pending_tasks(connection, limit)
    stats = {"scheduled": len(pending), "complete": 0, "partial": 0, "failed": 0}
    lock = threading.Lock()
    if not pending:
        return stats
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        calls = [pool.submit(_request_book_delta, ts, markets, timeout=timeout, retries=retries) for ts, markets in pending]
        for number, call in enumerate(futures.as_completed(calls), start=1):
            result = call.result()
            now = datetime.now(UTC).isoformat()
            markets_json = json.dumps(result["markets"], separators=(",", ":"))
            if result["ok"]:
                relative = _append_response(out, result, lock)
                received = set(result["received_anchor_markets"])
                requested = set(result["markets"])
                status = "complete" if (not result["needs_anchor"] or received == requested) else "partial"
                stats[status] += 1
                connection.execute(
                    """
                    UPDATE task SET status=?, attempts=attempts+?, response_sha256=?,
                    response_bytes=?, received_markets_json=?, response_path=?, error=NULL,
                    updated_at=? WHERE request_ts=? AND markets_json=?
                    """,
                    (status, result["attempts"], _sha_bytes(result["body"]), len(result["body"]),
                     json.dumps(result["received_anchor_markets"], separators=(",", ":")), relative, now,
                     result["request_ts"], markets_json),
                )
            else:
                stats["failed"] += 1
                connection.execute(
                    """
                    UPDATE task SET status='failed', attempts=attempts+?, error=?, updated_at=?
                    WHERE request_ts=? AND markets_json=?
                    """,
                    (result["attempts"], result["error"], now, result["request_ts"], markets_json),
                )
            if number % 50 == 0 or number == len(calls):
                connection.commit()
                print(json.dumps({"event": "progress", "completed": number, **stats}, sort_keys=True), flush=True)
    connection.commit()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start", default=None, help="optional UTC entry timestamp, inclusive")
    parser.add_argument("--end", default=None, help="optional UTC entry timestamp, exclusive")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-requests", type=int, default=None)
    parser.add_argument(
        "--exit-reason", action="append",
        default=["stop_loss", "fast_adverse", "capital_protect"],
        help="repeatable replay-scope exit reason; defaults to the hard-stop-sensitive population",
    )
    parser.add_argument("--all-exit-reasons", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not args.ledger.is_file():
        raise FileNotFoundError(args.ledger)
    if args.workers < 1 or args.workers > 32:
        raise ValueError("workers must be between 1 and 32")
    start = _utc(args.start) if args.start else None
    end = _utc(args.end) if args.end else None
    if start is not None and end is not None and start >= end:
        raise ValueError("start must precede end")
    available = _coverage_markets(args.timeout_seconds)
    exit_reasons = None if args.all_exit_reasons else set(map(str, args.exit_reason))
    tasks, audit = _load_tasks(
        args.ledger, start=start, end=end, available_markets=available,
        exit_reasons=exit_reasons,
    )
    manifest = {
        "schema": TASK_SCHEMA,
        "provider": "Tardis historical Crypto Facilities/Kraken Futures book plus day-start book_snapshot",
        "ledger": str(args.ledger), "ledger_sha256": _sha_path(args.ledger),
        "start": _iso(start) if start is not None else None,
        "end_exclusive": _iso(end) if end is not None else None,
        "coverage_markets": int(len(available)), "task_audit": audit,
        "request_contract": "daily native L2 snapshot anchor plus every following minute's book deltas through selected post-entry exits; no OHLCV order-book proxy",
        "exit_reasons": sorted(exit_reasons) if exit_reasons is not None else "all",
        "created_at": datetime.now(UTC).isoformat(),
    }
    if args.plan_only:
        print(json.dumps({"event": "plan", **manifest}, sort_keys=True))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "download_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("ledger_sha256") != manifest["ledger_sha256"]:
            raise ValueError("output path belongs to a different accepted-position ledger")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    connection = _connect(args.out / "download_state.sqlite")
    try:
        _seed_tasks(connection, tasks)
        stats = _run_download(
            connection, out=args.out, workers=args.workers, timeout=args.timeout_seconds,
            retries=args.retries, limit=args.max_requests,
        )
        status_rows = dict(connection.execute("SELECT status, COUNT(*) FROM task GROUP BY status"))
    finally:
        connection.close()
    print(json.dumps({"event": "complete", "out": str(args.out), **audit, **stats, "task_status": status_rows}, sort_keys=True))


if __name__ == "__main__":
    main()
