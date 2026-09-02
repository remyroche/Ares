#!/usr/bin/env python3
"""Create an immutable, fee-confirmed outcome sidecar for strict-R3 trades.

The close-notification ledger is deliberately immutable. Early live records
contain observed fill prices and gross PnL but not Kraken's final entry/exit
fees or funding. This read-only utility joins those records to the Kraken
Futures account log and writes a separate receipt. It never changes the
ledger, local position state, exchange positions, or orders.

A result is confirmed only when both the recorded entry and exit have a
contract-matched Kraken futures event inside the configured timestamp window.
The net result is exchange-booked realised PnL, less booked fees, plus booked
funding while the recorded position was open. Missing history, overlapping
local positions, or a timing mismatch fail closed rather than estimating PnL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.data_fetcher import make_exchange  # noqa: E402


SCHEMA = "strict_r3_fee_confirmed_execution_sidecar_v1"
FUTURES_EVENT_MARKERS = (
    "futures trade",
    "futures partial liquidation",
    "futures liquidation",
)
FUNDING_EVENT = "funding rate change"


def _utc(value: object) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _number(value: object) -> float:
    number = pd.to_numeric(value, errors="coerce")
    return float(number) if np.isfinite(number) else 0.0


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _contract_for_symbol(symbol: object) -> str:
    token = str(symbol).split("/", 1)[0].strip().lower()
    if not token:
        raise ValueError(f"cannot derive Kraken contract from symbol={symbol!r}")
    return f"pf_{token}usd"


def _trade_records(ledger: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    records = ledger.get("records")
    if not isinstance(records, Mapping):
        raise ValueError("close-notification ledger lacks mapping records")
    return [(str(key), value) for key, value in records.items() if isinstance(value, Mapping)]


def _telemetry_times(record: Mapping[str, Any]) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    telemetry = record.get("trade_telemetry")
    if not isinstance(telemetry, Mapping):
        return None, None
    entry = telemetry.get("entry")
    exit_ = telemetry.get("exit")
    entry_ts = _utc(entry.get("entry_fill_time")) if isinstance(entry, Mapping) else None
    exit_ts = _utc(exit_.get("exit_time")) if isinstance(exit_, Mapping) else None
    return entry_ts, exit_ts


def _event_kind(row: Mapping[str, Any]) -> str:
    return str(row.get("info") or "").strip().lower()


def _is_trade_event(row: Mapping[str, Any]) -> bool:
    return any(marker in _event_kind(row) for marker in FUTURES_EVENT_MARKERS)


def _sanitise_event(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "id", "booking_uid", "date", "info", "contract", "asset", "execution",
            "fee", "realized_pnl", "realized_funding", "trade_price", "funding_rate",
        )
    }


def reconcile(
    *,
    ledger: Mapping[str, Any],
    account_logs: Sequence[Mapping[str, Any]],
    tolerance_seconds: int = 300,
) -> dict[str, Any]:
    """Reconcile booked fee/funding outcomes from a single exchange-log page."""
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be non-negative")
    prepared: list[tuple[pd.Timestamp, Mapping[str, Any]]] = []
    for event in account_logs:
        if not isinstance(event, Mapping):
            continue
        stamp = _utc(event.get("date"))
        if stamp is not None:
            prepared.append((stamp, event))
    prepared.sort(key=lambda item: item[0])
    log_min = prepared[0][0] if prepared else None
    log_max = prepared[-1][0] if prepared else None
    tolerance = pd.Timedelta(seconds=int(tolerance_seconds))

    windows: dict[str, list[tuple[str, pd.Timestamp, pd.Timestamp]]] = {}
    for key, record in _trade_records(ledger):
        entry_ts, exit_ts = _telemetry_times(record)
        if entry_ts is not None and exit_ts is not None and exit_ts >= entry_ts:
            windows.setdefault(_contract_for_symbol(record.get("symbol")), []).append(
                (key, entry_ts, exit_ts)
            )
    overlapping_keys: set[str] = set()
    for entries in windows.values():
        entries = sorted(entries, key=lambda item: item[1])
        for previous, current in zip(entries, entries[1:]):
            if current[1] <= previous[2] + tolerance:
                overlapping_keys.update((previous[0], current[0]))

    rows: list[dict[str, Any]] = []
    for key, record in _trade_records(ledger):
        telemetry = record.get("trade_telemetry")
        entry_ts, exit_ts = _telemetry_times(record)
        contract = _contract_for_symbol(record.get("symbol"))
        row: dict[str, Any] = {
            "record_key": key,
            "candidate_id": record.get("candidate_id"),
            "symbol": record.get("symbol"),
            "contract": contract,
            "entry_fill_time": entry_ts.isoformat() if entry_ts is not None else None,
            "exit_time": exit_ts.isoformat() if exit_ts is not None else None,
            "status": "unconfirmed",
            "reason": None,
        }
        if not isinstance(telemetry, Mapping) or entry_ts is None or exit_ts is None:
            row["reason"] = "missing_structured_entry_or_exit_time"
            rows.append(row)
            continue
        if exit_ts < entry_ts:
            row["reason"] = "exit_precedes_entry"
            rows.append(row)
            continue
        if key in overlapping_keys:
            row["reason"] = "overlapping_local_contract_position"
            rows.append(row)
            continue
        if log_min is None or log_max is None or log_min > entry_ts + tolerance or log_max < exit_ts - tolerance:
            row["reason"] = "account_log_history_does_not_cover_trade_window"
            rows.append(row)
            continue

        events = [
            (stamp, event)
            for stamp, event in prepared
            if str(event.get("asset") or "").lower() == "usd"
            and str(event.get("contract") or "").lower() == contract
            and entry_ts - tolerance <= stamp <= exit_ts + tolerance
            and (_is_trade_event(event) or _event_kind(event) == FUNDING_EVENT)
        ]
        entry_events = [
            event for stamp, event in events
            if _is_trade_event(event) and abs(stamp - entry_ts) <= tolerance
        ]
        exit_events = [
            event for stamp, event in events
            if _is_trade_event(event) and abs(stamp - exit_ts) <= tolerance
        ]
        if not entry_events:
            row["reason"] = "missing_contract_matched_entry_event"
            rows.append(row)
            continue
        if not exit_events:
            row["reason"] = "missing_contract_matched_exit_event"
            rows.append(row)
            continue

        pnl_quote = sum(_number(event.get("realized_pnl")) for _, event in events)
        fees_quote = sum(_number(event.get("fee")) for _, event in events)
        funding_quote = sum(_number(event.get("realized_funding")) for _, event in events)
        entry = telemetry.get("entry") if isinstance(telemetry.get("entry"), Mapping) else {}
        entry_notional = _number(entry.get("notional_quote"))
        net_quote = pnl_quote - fees_quote + funding_quote
        net_bps = 10_000.0 * net_quote / entry_notional if entry_notional > 0.0 else math.nan
        row.update({
            "status": "confirmed",
            "reason": "contract_and_time_matched_kraken_account_log",
            "entry_notional_quote": entry_notional if entry_notional > 0.0 else None,
            "realized_pnl_quote": pnl_quote,
            "fees_quote": fees_quote,
            "funding_quote": funding_quote,
            "net_quote": net_quote,
            "net_bps": net_bps if np.isfinite(net_bps) else None,
            "entry_event_count": len(entry_events),
            "exit_event_count": len(exit_events),
            "event_count": len(events),
            "account_log_events": [_sanitise_event(event) for _, event in events],
        })
        rows.append(row)

    return {
        "schema": SCHEMA,
        "source": {
            "exchange": "krakenfutures",
            "exchange_read_only": True,
            "account_log_row_count": len(prepared),
            "account_log_sha256": _sha256_json([dict(event) for _, event in prepared]),
            "account_log_min_timestamp": log_min.isoformat() if log_min is not None else None,
            "account_log_max_timestamp": log_max.isoformat() if log_max is not None else None,
            "tolerance_seconds": int(tolerance_seconds),
        },
        "ledger_records": len(_trade_records(ledger)),
        "ledger_sha256": None,
        "confirmed_trade_count": sum(row["status"] == "confirmed" for row in rows),
        "unconfirmed_trade_count": sum(row["status"] != "confirmed" for row in rows),
        "rows": rows,
    }


def _load_logs(path: Path | None) -> list[Mapping[str, Any]]:
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        logs = payload.get("logs") if isinstance(payload, Mapping) else payload
        if not isinstance(logs, list):
            raise ValueError("account-log JSON must be a list or mapping with logs")
        return [row for row in logs if isinstance(row, Mapping)]
    exchange = make_exchange("perps")
    if str(getattr(exchange, "id", "")) != "krakenfutures":
        raise ValueError("fee reconciliation requires Kraken Futures")
    payload = exchange.historyGetAccountLog({})
    logs = payload.get("logs") if isinstance(payload, Mapping) else payload
    if not isinstance(logs, list):
        raise ValueError("Kraken account log response lacks logs")
    return [row for row in logs if isinstance(row, Mapping)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--account-log-json", type=Path, default=None)
    parser.add_argument("--tolerance-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("sidecar output path must be immutable")
    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    result = reconcile(
        ledger=ledger,
        account_logs=_load_logs(args.account_log_json),
        tolerance_seconds=args.tolerance_seconds,
    )
    result["ledger_sha256"] = _file_sha256(args.ledger)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "schema": result["schema"],
        "confirmed_trade_count": result["confirmed_trade_count"],
        "unconfirmed_trade_count": result["unconfirmed_trade_count"],
        "out": str(args.out),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
