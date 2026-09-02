#!/usr/bin/env python3
"""Materialise compact Kraken order-book recaps from Hugging Face snapshots.

``Abraxasccs/kraken-market-data`` publishes Kraken *spot* book snapshots in
small Parquet files.  The source has no futures book tree, so this producer
records each selected symbol as an explicit ``spot_fallback`` rather than
silently treating it as a futures market.

The raw Parquet payload is kept only in process memory.  Each source file is
parsed into a flat L2 recap immediately and discarded; only the aggregate
order-book rows, a source hash, and retention audit are persisted.  It never
downloads or retains the source's individual ``trade`` data.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import re
import ssl
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.execution.surface import (  # noqa: E402
    add_causal_transition_features,
    add_future_deterioration_targets,
    execution_state_row,
)
from src.execution.tardis_book import BookState  # noqa: E402


DEFAULT_REPO = "Abraxasccs/kraken-market-data"
API_ROOT = "https://huggingface.co/api/datasets"
RESOLVE_ROOT = "https://huggingface.co/datasets"
BOOK_FILE = re.compile(r"/(\d{4})\.parquet$")
REQUIRED_COLUMNS = {"ts", "pair", "bids_json", "asks_json"}


@dataclass(frozen=True)
class SourceFile:
    source_date: str
    path: str
    source_oid: str
    source_size: int
    minute_of_day: int
    decision_ts: str = ""


def _request_bytes(url: str, *, retries: int = 4) -> bytes:
    """Retrieve a public source object without persisting its raw payload."""
    request = Request(url, headers={"User-Agent": "Ares-orderbook-recap/1.0"})
    # The managed Python runtime does not always expose the system trust
    # store.  certifi provides the standard Mozilla roots without weakening
    # certificate verification or accepting an unverified TLS connection.
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ModuleNotFoundError:  # pragma: no cover - regular Python installs
        context = ssl.create_default_context()
    last_error: Exception | None = None
    for attempt in range(int(retries)):
        try:
            with urlopen(request, timeout=90, context=context) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = error
            if attempt + 1 < int(retries):
                time.sleep(float(1 << attempt))
    raise RuntimeError(f"unable to retrieve {url}: {last_error}")


def _api_files(repo: str, date: str) -> list[SourceFile]:
    path = f"data/crypto/book/{date}"
    url = f"{API_ROOT}/{repo}/tree/main/{path}?recursive=false&expand=false&limit=1000"
    payload = json.loads(_request_bytes(url).decode("utf-8"))
    output: list[SourceFile] = []
    for item in payload:
        if item.get("type") != "file":
            continue
        relative = str(item.get("path", ""))
        match = BOOK_FILE.search(relative)
        if not match:
            continue
        encoded_time = match.group(1)
        hour, minute = int(encoded_time[:2]), int(encoded_time[2:])
        if hour > 23 or minute > 59:
            continue
        output.append(SourceFile(
            source_date=str(date), path=relative, source_oid=str(item.get("oid", "")),
            source_size=int(item.get("size", 0)), minute_of_day=hour * 60 + minute,
        ))
    if not output:
        raise ValueError(f"no book files found for {date} in {repo}")
    return sorted(output, key=lambda item: (item.minute_of_day, item.path))


def _select_files(files: list[SourceFile], *, retained_cadence_minutes: int) -> list[SourceFile]:
    """Select the last observed snapshot before each fixed decision boundary.

    The source's collection minute drifts a little (for example 00:01, 00:06,
    00:12).  Selecting by minute modulo would incorrectly retain only one
    snapshot.  Bucketing each observed filename into its *next* 15-minute
    decision boundary gives the full causal state available at that boundary,
    without inventing a bar between observations.
    """
    cadence = int(retained_cadence_minutes)
    if cadence <= 0 or 60 % cadence:
        raise ValueError("retained_cadence_minutes must divide 60")
    by_boundary: dict[int, SourceFile] = {}
    for item in files:
        boundary = (item.minute_of_day // cadence + 1) * cadence
        previous = by_boundary.get(boundary)
        if previous is None or item.minute_of_day > previous.minute_of_day:
            by_boundary[boundary] = item
    selected: list[SourceFile] = []
    date_start = pd.Timestamp(files[0].source_date, tz="UTC")
    for boundary, item in sorted(by_boundary.items()):
        selected.append(SourceFile(
            **{**asdict(item), "decision_ts": (date_start + pd.Timedelta(minutes=boundary)).isoformat()},
        ))
    if not selected:
        raise ValueError("no source book files match selected recap cadence")
    return selected


def _parse_levels(value: object, *, descending: bool) -> tuple[tuple[float, float], ...]:
    raw = json.loads(str(value))
    if not isinstance(raw, list):
        raise ValueError("book levels are not a JSON list")
    levels: list[tuple[float, float]] = []
    for level in raw:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            continue
        price, amount = float(level[0]), float(level[1])
        if np.isfinite(price) and price > 0.0 and np.isfinite(amount) and amount >= 0.0:
            levels.append((price, amount))
    return tuple(sorted(levels, key=lambda level: level[0], reverse=descending))


def _snapshot_rows(frame: pd.DataFrame, *, source: SourceFile, notionals: tuple[float, ...]) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{source.path} lacks required source fields: {sorted(missing)}")
    # A source file may contain several snapshots for the same pair.  The
    # decision contract is the final *available* book in that file's window;
    # choosing it per pair avoids a duplicate candidate identity while never
    # using a snapshot after the declared decision boundary.
    selected = frame.loc[:, ["ts", "pair", "bids_json", "asks_json"]].copy()
    selected["ts"] = pd.to_numeric(selected["ts"], errors="coerce")
    selected = selected.dropna(subset=["ts", "pair"]).sort_values(["pair", "ts"], kind="stable")
    selected = selected.drop_duplicates("pair", keep="last")
    rows: list[dict[str, Any]] = []
    for record in selected.itertuples(index=False):
        timestamp = pd.to_datetime(record.ts, unit="ms", utc=True, errors="coerce")
        symbol = str(record.pair)
        if pd.isna(timestamp) or not symbol:
            continue
        bids, asks = _parse_levels(record.bids_json, descending=True), _parse_levels(record.asks_json, descending=False)
        state = BookState(
            local_timestamp=pd.Timestamp(timestamp), exchange_timestamp=pd.Timestamp(timestamp),
            bids=bids, asks=asks, has_snapshot=True,
            crossed_or_empty=not bids or not asks or bids[0][0] >= asks[0][0], source_rows=1,
        )
        row = execution_state_row(state, symbol=symbol, notional_buckets=notionals)
        decision_ts = pd.Timestamp(source.decision_ts)
        # A complete source book is used at the next fixed decision boundary.
        # Its raw arrival is retained for provenance and must precede that
        # boundary; a late source is explicitly invalid instead of backfilled.
        source_available = bool(pd.Timestamp(timestamp) <= decision_ts)
        row["state_minute"] = decision_ts
        row["decision_ts"] = decision_ts
        row["available_ts"] = pd.Timestamp(timestamp)
        row["source_available_by_decision"] = source_available
        row["book_valid"] = bool(row["book_valid"] and source_available)
        row["hf_source_path"] = source.path
        row["hf_source_oid"] = source.source_oid
        row["source_market"] = "spot"
        row["market_selection"] = "spot_fallback_no_futures_in_abraxasccs_dataset"
        row["raw_trade_data_retained"] = False
        rows.append(row)
    result = pd.DataFrame.from_records(rows)
    if result.empty:
        raise ValueError(f"{source.path} produced no valid snapshot rows")
    if result.duplicated(["symbol", "state_minute"]).any():
        raise ValueError(f"{source.path} produced duplicate symbol/minute keys")
    return result


def _download_and_recap(source: SourceFile, *, repo: str, notionals: tuple[float, ...]) -> tuple[pd.DataFrame, dict[str, Any]]:
    url = f"{RESOLVE_ROOT}/{repo}/resolve/main/{source.path}"
    raw = _request_bytes(url)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    frame = pd.read_parquet(io.BytesIO(raw))
    recap = _snapshot_rows(frame, source=source, notionals=notionals)
    # Explicitly drop references before this worker returns: raw order-book
    # files must never enter the research store.
    del raw, frame
    return recap, {
        **asdict(source), "raw_sha256": raw_sha256,
        "raw_payload_retained": False, "raw_payload_discarded_after_recap": True,
        "recap_rows": int(len(recap)), "symbols": int(recap["symbol"].nunique()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f".{path.name}.partial")
    frame.to_parquet(staged, index=False)
    os.replace(staged, path)


def _aggregate_existing_partitions(out_root: Path) -> list[dict[str, Any]]:
    """Rebuild the root manifest from immutable per-date receipts.

    The materializer is deliberately resumable in small network batches.  A
    root receipt must therefore describe every already materialized partition,
    not merely the last CLI invocation.
    """
    output: list[dict[str, Any]] = []
    for audit_path in sorted(out_root.rglob("source_recap_audit.json")):
        receipt = json.loads(audit_path.read_text())
        surface_path = Path(str(receipt["output"]))
        if not surface_path.exists():
            raise FileNotFoundError(f"receipt references missing compact surface: {surface_path}")
        frame = pd.read_parquet(surface_path, columns=["symbol"])
        sources = receipt.get("sources", [])
        output.append({
            "source_date": str(receipt["source_date"]), "status": "materialized",
            "output": str(surface_path), "sha256": str(receipt["output_sha256"]),
            "rows": int(len(frame)), "symbols": int(frame["symbol"].nunique()),
            "source_files": int(len(sources)),
            "raw_payload_bytes_discarded": int(sum(int(source.get("source_size", 0)) for source in sources)),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", required=True, help="UTC source-date directories, YYYY-MM-DD")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--retained-cadence-minutes", type=int, choices=(5, 10, 15, 20, 30, 60), default=15)
    parser.add_argument("--notionals", type=float, nargs="+", default=(1_000.0, 5_000.0, 10_000.0))
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--allow-missing-source-dates", action="store_true",
        help="Record catalogue dates with no book files and continue the resumable batch; never fabricate a partition.",
    )
    args = parser.parse_args()

    notionals = tuple(sorted({float(value) for value in args.notionals if float(value) > 0.0}))
    if not notionals:
        raise ValueError("--notionals must contain at least one positive value")
    audit_rows: list[dict[str, Any]] = []
    for date in sorted(set(args.dates)):
        output = args.out_root / "features" / "kraken_execution_surface" / f"source_date={date}" / "surface.parquet"
        if args.skip_existing and output.exists():
            audit_rows.append({"source_date": date, "status": "existing", "output": str(output), "sha256": _sha256(output)})
            continue
        try:
            source_files = _api_files(args.repo, date)
        except (ValueError, RuntimeError) as error:
            is_catalogue_gap = isinstance(error, ValueError) or "HTTP Error 404" in str(error)
            if not args.allow_missing_source_dates or not is_catalogue_gap:
                raise
            audit_rows.append({"source_date": date, "status": "source_unavailable", "reason": str(error)})
            continue
        selected = _select_files(source_files, retained_cadence_minutes=int(args.retained_cadence_minutes))
        frames: list[pd.DataFrame] = []
        source_audit: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.max_workers)) as executor:
            futures = [executor.submit(_download_and_recap, item, repo=args.repo, notionals=notionals) for item in selected]
            for future in concurrent.futures.as_completed(futures):
                recap, source_row = future.result()
                frames.append(recap)
                source_audit.append(source_row)
        raw_surface = pd.concat(frames, ignore_index=True, copy=False).sort_values(["symbol", "state_minute", "available_ts"], kind="stable")
        if raw_surface.duplicated(["symbol", "state_minute"]).any():
            raise ValueError(f"{date} has duplicate retained snapshot identities")
        # The selected source is observed every declared cadence.  All
        # transition and future-label support requires that exact sequence.
        surface = add_causal_transition_features(raw_surface, cadence_minutes=int(args.retained_cadence_minutes))
        surface = add_future_deterioration_targets(surface, cadence_minutes=int(args.retained_cadence_minutes))
        _write_atomic(surface, output)
        source_audit.sort(key=lambda row: (int(row["minute_of_day"]), str(row["path"])))
        audit_path = output.parent / "source_recap_audit.json"
        audit_path.write_text(json.dumps({
            "schema": "ares.hf_kraken_orderbook_recap.v1",
            "repo": args.repo,
            "source_date": date,
            "source_market": "spot",
            "market_selection": "spot_fallback_no_futures_in_abraxasccs_dataset",
            "retained_cadence_minutes": int(args.retained_cadence_minutes),
            "retention": "order-book recap only; individual trade files were never requested; raw book payloads were streamed and discarded",
            "sources": source_audit,
            "output": str(output), "output_sha256": _sha256(output),
        }, indent=2) + "\n")
        audit_rows.append({
            "source_date": date, "status": "materialized", "output": str(output), "sha256": _sha256(output),
            "rows": int(len(surface)), "symbols": int(surface["symbol"].nunique()), "source_files": int(len(source_audit)),
            "raw_payload_bytes_discarded": int(sum(int(row["source_size"]) for row in source_audit)),
        })
    args.out_root.mkdir(parents=True, exist_ok=True)
    all_partitions = _aggregate_existing_partitions(args.out_root)
    pd.DataFrame.from_records(all_partitions).to_parquet(args.out_root / "materialization_audit.parquet", index=False)
    manifest = {
        "schema": "ares.hf_kraken_orderbook_recap.v1",
        "repo": args.repo, "source_market": "spot",
        "market_selection": "spot_fallback_no_futures_in_abraxasccs_dataset",
        "retained_cadence_minutes": int(args.retained_cadence_minutes), "notionals": list(notionals),
        "retention": "compact order-book recaps only; raw book payloads and all trade prints absent",
        "partitions": all_partitions,
    }
    (args.out_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
