#!/usr/bin/env python3
"""Stream Tardis Kraken L2 files into compact causal minute execution states.

The raw incremental book remains immutable.  This producer keeps one mutable
book and one current-minute row per symbol/day in memory; it does not retain a
full book per tick.  Future deterioration labels are appended only after the
causal surface has been formed and are explicitly marked as offline-only.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.execution.surface import (  # noqa: E402
    add_causal_transition_features,
    add_future_deterioration_targets,
    execution_state_row,
)
from src.execution.tardis_book import IncrementalL2Book, to_utc_timestamp  # noqa: E402


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"invalid timestamp {value!r}")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _parse_notionals(args: argparse.Namespace) -> tuple[float, ...]:
    direct = tuple(float(value) for value in (args.notionals or ()))
    if direct and args.notional_source:
        raise ValueError("provide --notionals or --notional-source, not both")
    if direct:
        return tuple(sorted(set(direct)))
    if not args.notional_source:
        raise ValueError(
            "execution cost needs an explicit historical sizing grid: provide "
            "--notionals or --notional-source/--notional-column"
        )
    frame = pd.read_parquet(args.notional_source, columns=[args.notional_column])
    values = pd.to_numeric(frame[args.notional_column], errors="coerce").dropna()
    values = values.loc[values.gt(0.0)]
    if values.empty:
        raise ValueError("notional source contains no finite positive values")
    return tuple(float(value) for value in np.unique(np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])))


def _iter_complete_records(path: Path) -> Iterator[list[dict[str, str]]]:
    """Stream source-ordered atomic messages without Pandas grouping.

    Tardis emits all rows belonging to one WebSocket message contiguously and
    gives them the same ``local_timestamp``.  Retaining this grouping while
    reading gzip CSV directly eliminates the dominant per-message DataFrame
    allocation/groupby overhead of the first pilot implementation.
    """
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        missing = {"timestamp", "local_timestamp", "is_snapshot", "side", "price", "amount"}.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} has no required L2 columns: {sorted(missing)}")
        current_local: str | None = None
        records: list[dict[str, str]] = []
        for record in reader:
            local = record["local_timestamp"]
            if current_local is None:
                current_local = local
            elif local != current_local:
                yield records
                records = []
                current_local = local
            records.append(record)
        if records:
            yield records


def _minute_key(local_timestamp: str) -> int:
    """Return a UTC minute key without constructing a Timestamp per message."""
    try:
        raw = abs(int(float(local_timestamp)))
    except (TypeError, ValueError):
        return int(to_utc_timestamp(local_timestamp).value // 60_000_000_000)
    if raw >= 100_000_000_000_000:
        return int(raw // 60_000_000)  # Tardis raw microseconds.
    if raw >= 100_000_000_000:
        return int(raw // 60_000)  # milliseconds.
    return int(raw // 60)  # seconds.


def _message_exchange_timestamp(records: list[dict[str, str]]) -> str:
    """Select the latest raw exchange timestamp without Pandas conversion."""
    return max((record["timestamp"] for record in records), key=lambda value: float(value))


_FLOW_COLUMNS = (
    "bid_cancel_notional",
    "ask_cancel_notional",
    "bid_replenish_notional",
    "ask_replenish_notional",
)


def _empty_book_flow() -> dict[str, float]:
    return {column: 0.0 for column in _FLOW_COLUMNS}


def _message_book_flow(book: IncrementalL2Book, records: list[dict[str, str]]) -> dict[str, float]:
    """Quote-notional cancelled/replenished by one atomic non-snapshot update.

    Tardis L2 levels are replacements rather than additive deltas.  We compare
    the message to the *pre-message* reconstructed level, never to a later
    state.  Snapshot resets are intentionally excluded: a reconnect snapshot
    is a feed event, not a cancellation/replenishment observation.
    """
    flow = _empty_book_flow()
    if not book._has_snapshot or any(str(row["is_snapshot"]).lower() in {"true", "1", "t", "yes"} for row in records):
        return flow
    for row in records:
        side = str(row["side"]).strip().lower()
        if side not in {"bid", "ask"}:
            raise ValueError(f"unexpected L2 side {row['side']!r}")
        price, amount = float(row["price"]), float(row["amount"])
        if not (np.isfinite(price) and price > 0.0 and np.isfinite(amount) and amount >= 0.0):
            raise ValueError("L2 flow record contains invalid price or amount")
        prior = (book.bids if side == "bid" else book.asks).get(price, 0.0)
        difference = (amount - prior) * price
        if difference < 0.0:
            flow[f"{side}_cancel_notional"] += -difference
        elif difference > 0.0:
            flow[f"{side}_replenish_notional"] += difference
    return flow


def _iter_trade_flow(path: Path) -> pd.DataFrame:
    """Aggregate immutable Tardis prints into fully observed one-minute flow."""
    if not path.exists():
        return pd.DataFrame(columns=["state_minute", "trade_quote_volume", "trade_intensity", "sell_order_flow_imbalance"])
    accumulators: dict[pd.Timestamp, dict[str, float]] = {}
    with gzip.open(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"local_timestamp", "side", "price", "amount"}
        if reader.fieldnames is None or required.difference(reader.fieldnames):
            raise ValueError(f"{path} lacks required trade fields {sorted(required)}")
        for row in reader:
            minute = to_utc_timestamp(row["local_timestamp"]).floor("min")
            side = str(row["side"]).strip().lower()
            price, amount = float(row["price"]), float(row["amount"])
            if side not in {"buy", "sell"} or not (np.isfinite(price) and price > 0.0 and np.isfinite(amount) and amount >= 0.0):
                continue
            totals = accumulators.setdefault(minute, {"buy": 0.0, "sell": 0.0, "count": 0.0})
            totals[side] += price * amount
            totals["count"] += 1.0
    rows: list[dict[str, float | pd.Timestamp]] = []
    for minute, totals in accumulators.items():
        total = totals["buy"] + totals["sell"]
        rows.append({
            "state_minute": minute,
            "buy_trade_quote_volume": totals["buy"],
            "sell_trade_quote_volume": totals["sell"],
            "trade_quote_volume": total,
            "trade_intensity": totals["count"],
            # Positive means aggressive selling dominates the completed minute.
            "sell_order_flow_imbalance": (totals["sell"] - totals["buy"]) / total if total > 0.0 else 0.0,
        })
    return pd.DataFrame.from_records(rows)


class _ParquetBatchWriter:
    def __init__(self, path: Path, *, batch_size: int = 20_000) -> None:
        self.path = path
        self.batch_size = batch_size
        self._writer: pq.ParquetWriter | None = None
        self._rows: list[dict[str, Any]] = []

    def append(self, row: dict[str, Any]) -> None:
        self._rows.append(row)
        if len(self._rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        table = pa.Table.from_pandas(pd.DataFrame.from_records(self._rows), preserve_index=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self._writer.write_table(table)
        self._rows.clear()

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()


def build_symbol_day_surface(
    path: Path,
    *,
    symbol: str,
    notionals: tuple[float, ...],
    chunksize: int,
    trade_path: Path | None = None,
    message_writer: _ParquetBatchWriter | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Reconstruct one raw symbol-day and keep only final state per minute."""
    book = IncrementalL2Book()
    last_minute: int | None = None
    last_update = None
    last_local_raw: str | None = None
    last_exchange_raw: str | None = None
    current_minute_flow = _empty_book_flow()
    minute_rows: list[dict[str, Any]] = []
    stats = {"messages": 0, "pre_snapshot_messages": 0, "invalid_messages": 0, "valid_messages": 0}
    del chunksize  # Kept as a compatible CLI argument; raw streaming is chunk-free.
    for records in _iter_complete_records(path):
        # Materialize the preceding minute *before* applying this message.
        # Thus a compact row is the final full book known in that minute,
        # without sorting all levels for every tick.
        message_minute = _minute_key(records[0]["local_timestamp"])
        if last_minute is not None and message_minute != last_minute and last_update is not None:
            completed = execution_state_row(
                book.materialize(
                    last_update,
                    local_timestamp=to_utc_timestamp(last_local_raw),
                    exchange_timestamp=to_utc_timestamp(last_exchange_raw),
                ),
                symbol=symbol,
                notional_buckets=notionals,
            )
            completed.update(current_minute_flow)
            completed["book_cancel_notional"] = completed["bid_cancel_notional"] + completed["ask_cancel_notional"]
            completed["book_replenish_notional"] = completed["bid_replenish_notional"] + completed["ask_replenish_notional"]
            minute_rows.append(completed)
            current_minute_flow = _empty_book_flow()
        preserve_timestamps = message_writer is not None
        flow = _message_book_flow(book, records)
        update = book.apply_records(records, materialize=False, preserve_timestamps=preserve_timestamps)
        stats["messages"] += 1
        if update is None:
            stats["pre_snapshot_messages"] += 1
            continue
        # Detailed state files are for exact executable joins.  Invalid/crossed
        # books are logged in the audit but deliberately never become a usable
        # message-state record.
        if message_writer is not None and update.valid:
            message_writer.append(execution_state_row(book.materialize(update), symbol=symbol, notional_buckets=notionals))
        if update.valid:
            stats["valid_messages"] += 1
        else:
            stats["invalid_messages"] += 1
        last_update = update
        last_minute = message_minute
        last_local_raw = records[0]["local_timestamp"]
        last_exchange_raw = _message_exchange_timestamp(records)
        for column, value in flow.items():
            current_minute_flow[column] += value
    if last_update is not None:
        completed = execution_state_row(
            book.materialize(
                last_update,
                local_timestamp=to_utc_timestamp(last_local_raw),
                exchange_timestamp=to_utc_timestamp(last_exchange_raw),
            ),
            symbol=symbol,
            notional_buckets=notionals,
        )
        completed.update(current_minute_flow)
        completed["book_cancel_notional"] = completed["bid_cancel_notional"] + completed["ask_cancel_notional"]
        completed["book_replenish_notional"] = completed["bid_replenish_notional"] + completed["ask_replenish_notional"]
        minute_rows.append(completed)
    surface = pd.DataFrame.from_records(minute_rows)
    if surface.empty:
        return surface, stats
    if trade_path is not None and trade_path.exists():
        trade_flow = _iter_trade_flow(trade_path)
        surface = surface.merge(trade_flow, on="state_minute", how="left", sort=False)
        for column in ("buy_trade_quote_volume", "sell_trade_quote_volume", "trade_quote_volume", "trade_intensity", "sell_order_flow_imbalance"):
            surface[column] = pd.to_numeric(surface[column], errors="coerce").fillna(0.0)
        stats["trade_flow_joined"] = int(len(trade_flow))
    else:
        stats["trade_flow_joined"] = 0
    return surface, stats


def _surface_path(root: Path, *, symbol: str, sample_date: pd.Timestamp) -> Path:
    return root / "features" / "kraken_execution_surface" / f"year={sample_date.year}" / f"date={sample_date.date()}" / f"symbol={symbol.replace('/', '__')}" / "surface.parquet"


def _message_path(root: Path, *, symbol: str, sample_date: pd.Timestamp) -> Path:
    return root / "processed" / "kraken_execution_message_states" / f"year={sample_date.year}" / f"date={sample_date.date()}" / f"symbol={symbol.replace('/', '__')}" / "states.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Download-status manifest containing raw paths")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/execution/tardis")
    parser.add_argument("--notionals", type=float, nargs="+", help="Explicit spot quote-notional grid")
    parser.add_argument("--notional-source", type=Path, help="Historical portfolio table for p10/p25/p50/p75/p90 notionals")
    parser.add_argument("--notional-column", default="position_notional")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--write-message-states", action="store_true", help="Persist detailed states for exact 0/250ms/1s/5s oracle joins")
    parser.add_argument("--features-only", action="store_true", help="Do not attach future-only deterioration labels")
    parser.add_argument("--limit", type=int, help="Process only this many downloaded files (pilot convenience)")
    parser.add_argument("--dataset-symbols", nargs="+", help="Exact Tardis dataset symbols to process")
    parser.add_argument("--sample-dates", nargs="+", help="Exact UTC YYYY-MM-DD partitions to process")
    parser.add_argument("--allow-existing-raw", action="store_true", help="Accept an atomically completed raw path before a resumable downloader writes its final status manifest")
    parser.add_argument("--audit-path", type=Path, help="Date-scoped audit output; required for parallel partition builds")
    parser.add_argument("--skip-existing", action="store_true", help="Preserve an existing immutable compact surface")
    args = parser.parse_args()

    notionals = _parse_notionals(args)
    manifest = pd.read_parquet(args.manifest)
    manifest["sample_date"] = pd.to_datetime(manifest["sample_date"], utc=True, errors="coerce")
    has_path = manifest["download_target"].notna() & manifest["download_target"].astype(str).str.strip().ne("")
    downloaded = manifest["status"].eq("downloaded")
    # ``Path(\"\")`` resolves to the working directory; use ``is_file`` and
    # an explicit non-empty raw path so an unavailable manifest row can never
    # become a fake raw input under resumable acquisition.
    existing = manifest["download_target"].map(lambda value: Path(str(value)).is_file() if pd.notna(value) and str(value).strip() else False)
    selected = manifest.loc[
        manifest["data_type"].eq("incremental_book_L2")
        & has_path
        & (downloaded | (existing if args.allow_existing_raw else False))
    ].copy()
    selected = selected.sort_values(["sample_date", "dataset_symbol"], kind="stable")
    if args.dataset_symbols:
        wanted = set(args.dataset_symbols)
        selected = selected.loc[selected["dataset_symbol"].isin(wanted)].copy()
    if args.sample_dates:
        wanted_dates = set(args.sample_dates)
        selected = selected.loc[selected["sample_date"].dt.strftime("%Y-%m-%d").isin(wanted_dates)].copy()
    if args.limit:
        selected = selected.head(int(args.limit))
    if selected.empty:
        raise RuntimeError("no downloaded incremental_book_L2 rows in manifest")
    trade_targets = manifest.loc[
        manifest["data_type"].eq("trades")
        & manifest["status"].eq("downloaded")
        & manifest["download_target"].notna(),
        ["sample_date", "dataset_symbol", "download_target"],
    ].copy()
    trade_targets["sample_date"] = pd.to_datetime(trade_targets["sample_date"], utc=True, errors="coerce")
    trade_paths = {
        (pd.Timestamp(row.sample_date), str(row.dataset_symbol)): Path(str(row.download_target))
        for row in trade_targets.itertuples(index=False)
    }
    diagnostics: list[dict[str, Any]] = []
    for source in selected.to_dict("records"):
        raw = Path(str(source["download_target"]))
        if not raw.exists():
            diagnostics.append({**source, "status": "missing_raw_file"})
            continue
        sample_date = _utc(source["sample_date"])
        symbol = str(source["dataset_symbol"])
        output = _surface_path(args.data_root, symbol=symbol, sample_date=sample_date)
        if args.skip_existing and output.exists():
            diagnostics.append({**source, "status": "existing_surface", "surface_path": str(output)})
            continue
        message_writer = None
        if args.write_message_states:
            message_writer = _ParquetBatchWriter(_message_path(args.data_root, symbol=symbol, sample_date=sample_date))
        try:
            surface, stats = build_symbol_day_surface(
                raw, symbol=symbol, notionals=notionals, chunksize=int(args.chunksize),
                trade_path=trade_paths.get((sample_date, symbol)), message_writer=message_writer,
            )
        finally:
            if message_writer is not None:
                message_writer.close()
        if surface.empty:
            diagnostics.append({**source, **stats, "status": "no_complete_book_state"})
            continue
        surface = add_causal_transition_features(surface)
        if not args.features_only:
            surface = add_future_deterioration_targets(surface)
        output.parent.mkdir(parents=True, exist_ok=True)
        staged = output.with_name(f".{output.name}.partial")
        surface.to_parquet(staged, index=False)
        os.replace(staged, output)
        diagnostics.append({
            **source, **stats, "status": "processed", "surface_path": str(output),
            "surface_rows": int(len(surface)), "valid_minutes": int(surface["book_valid"].fillna(False).sum()),
        })
    audit_path = args.audit_path or (args.data_root / "reports" / "kraken_execution_state_build_audit.parquet")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame.from_records(diagnostics)
    # A resumed multi-symbol build must retain prior symbol receipts.  Replace
    # only an identically keyed source row; raw files and completed surfaces
    # themselves are never overwritten by ``--skip-existing``.
    key_columns = ["sample_date", "data_type", "dataset_symbol"]
    if audit_path.exists():
        prior = pd.read_parquet(audit_path)
        if not audit.empty and all(column in prior.columns for column in key_columns):
            fresh_keys = set(map(tuple, audit.loc[:, key_columns].astype(str).itertuples(index=False, name=None)))
            prior_keys = prior.loc[:, key_columns].astype(str).apply(tuple, axis=1)
            prior = prior.loc[~prior_keys.isin(fresh_keys)].copy()
        audit = pd.concat([prior, audit], ignore_index=True, sort=False)
    staged_audit = audit_path.with_name(f".{audit_path.name}.partial")
    audit.to_parquet(staged_audit, index=False)
    os.replace(staged_audit, audit_path)
    receipt = {
        "schema": "ares.kraken_execution_surface.v1",
        "manifest": str(args.manifest),
        "notional_grid_quote": list(notionals),
        "features_only": bool(args.features_only),
        "message_states_written": bool(args.write_message_states),
        "processed": int(audit["status"].eq("processed").sum()),
        "audit": str(audit_path),
        "causality": "minute features retain only source states observed by the completed minute; future deterioration fields are offline labels only and are never live inputs",
    }
    audit_path.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
