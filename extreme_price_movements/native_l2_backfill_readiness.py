"""Fail-closed readiness checks for historical native-L2 backfill.

The repository contains many parquet files whose order-book-shaped rows are
OHLCV-derived proxies.  Those rows must not be presented as native depth or
flow history.  This module scans only parquet metadata plus the source,
identity, and timestamp columns needed to classify the available substrate.
It never reads labels, scores, costs, or portfolio fields.

The output is intentionally a *readiness* result rather than a training
dataset.  A native source is considered sufficient for the current roadmap
only if its timestamp window contains the candidate windows that the strict
overlap audit declares.  Sparse native rows are still reported, but they do
not pass that gate.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow.parquet as pq


NATIVE_L2_SOURCE = "kraken_futures_l2_snapshot"
PROXY_SOURCES = frozenset({"local_ohlcv_summary"})
DEFAULT_SCAN_ROOTS = (
    Path("data_perp/orderbook_hourly"),
    Path("data_perp/exchanges/krakenfutures/orderbook_hourly"),
)
DEFAULT_EXCLUDED_PARTS = frozenset({".git", "artifacts", ".staging"})


def discover_parquet_files(
    roots: Iterable[Path],
    *,
    excluded_parts: Iterable[str] = DEFAULT_EXCLUDED_PARTS,
) -> list[Path]:
    """Return deterministic parquet paths, excluding generated artifacts."""
    excluded = frozenset(str(part) for part in excluded_parts)
    found: set[Path] = set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*.parquet"):
            parts = set(path.parts)
            if parts.intersection(excluded):
                continue
            found.add(path)
    return sorted(found)


def _first_column(names: set[str], candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def _utc_bounds(series: pd.Series) -> tuple[str | None, str | None]:
    if series.empty:
        return None, None
    parsed = pd.to_datetime(series, utc=True, errors="coerce").dropna()
    if parsed.empty:
        return None, None
    return parsed.min().isoformat(), parsed.max().isoformat()


def _utc_day_counts(series: pd.Series) -> dict[str, int]:
    """Count valid observations by UTC calendar day without filling gaps."""
    parsed = pd.to_datetime(series, utc=True, errors="coerce").dropna()
    if parsed.empty:
        return {}
    counts = parsed.dt.floor("D").dt.strftime("%Y-%m-%d").value_counts()
    return {str(day): int(count) for day, count in counts.sort_index().items()}


def inventory_parquet_file(path: Path) -> dict[str, Any]:
    """Inventory one file without loading unrelated model/outcome columns."""
    parquet = pq.ParquetFile(path)
    metadata = parquet.metadata
    row_count = int(metadata.num_rows) if metadata is not None else 0
    names = set(parquet.schema.names)
    source_column = _first_column(names, ("source", "source_name", "data_source"))
    symbol_column = _first_column(names, ("symbol", "__symbol__", "product", "instrument"))
    timestamp_column = _first_column(
        names,
        ("snapshot_ts", "timestamp", "ts", "datetime", "__ts__", "available_at"),
    )
    columns = list(
        dict.fromkeys(
            column
            for column in (source_column, symbol_column, timestamp_column)
            if column is not None
        )
    )
    if columns:
        frame = parquet.read(columns=columns).to_pandas()
        # Parquet pandas metadata may declare a timestamp as the index.  The
        # inventory still needs the field as a normal column, but resetting is
        # safe only when it is not already present (some files store both).
        if frame.index.name in columns and frame.index.name not in frame.columns:
            frame = frame.reset_index()
    else:
        frame = pd.DataFrame(index=range(row_count))

    if source_column is None:
        source_counts: Counter[str] = Counter({"<missing_source_column>": row_count})
    else:
        source_values = frame[source_column].astype("string").fillna("<null>")
        source_counts = Counter(str(value) for value in source_values.tolist())

    native_rows = int(source_counts.get(NATIVE_L2_SOURCE, 0))
    proxy_rows = int(sum(source_counts.get(value, 0) for value in PROXY_SOURCES))
    tagged_rows = int(sum(source_counts.values()))
    if native_rows and proxy_rows:
        classification = "mixed_native_and_proxy"
    elif native_rows:
        classification = "native_exact"
    elif proxy_rows and proxy_rows == tagged_rows:
        classification = "proxy_only"
    elif source_column is None:
        classification = "source_unclassified"
    else:
        classification = "other_source_tags"

    overall_min, overall_max = (None, None)
    native_min, native_max = (None, None)
    proxy_min, proxy_max = (None, None)
    native_day_counts: dict[str, int] = {}
    if timestamp_column is not None and not frame.empty:
        parsed = pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce")
        overall_min, overall_max = _utc_bounds(parsed)
        if source_column is not None:
            native_mask = frame[source_column].astype("string").eq(NATIVE_L2_SOURCE)
            proxy_mask = frame[source_column].astype("string").isin(PROXY_SOURCES)
            native_min, native_max = _utc_bounds(parsed.loc[native_mask])
            proxy_min, proxy_max = _utc_bounds(parsed.loc[proxy_mask])
            native_day_counts = _utc_day_counts(parsed.loc[native_mask])

    symbol_count = 0
    native_symbol_count = 0
    if symbol_column is not None and not frame.empty:
        symbols = frame[symbol_column].astype("string").replace("<NA>", pd.NA).dropna()
        symbol_count = int(symbols.nunique())
        if source_column is not None:
            native_symbol_count = int(
                frame.loc[
                    frame[source_column].astype("string").eq(NATIVE_L2_SOURCE),
                    symbol_column,
                ]
                .astype("string")
                .replace("<NA>", pd.NA)
                .dropna()
                .nunique()
            )
    elif row_count > 0:
        # Kraken's historical per-product files encode the exact product in
        # the filename rather than repeating a symbol column.  Keep that
        # identity explicit instead of reporting zero products or silently
        # collapsing collateral variants.
        symbol_count = 1
        native_symbol_count = int(native_rows > 0)

    return {
        "path": str(path),
        "rows": row_count,
        "source_column": source_column,
        "symbol_column": symbol_column,
        "timestamp_column": timestamp_column,
        "identity_source": symbol_column or "file_stem",
        "classification": classification,
        "native_rows": native_rows,
        "proxy_rows": proxy_rows,
        "source_tag_rows": tagged_rows,
        "source_counts": dict(sorted(source_counts.items())),
        "symbols": symbol_count,
        "native_symbols": native_symbol_count,
        "min_ts": overall_min,
        "max_ts": overall_max,
        "native_min_ts": native_min,
        "native_max_ts": native_max,
        "native_day_counts": native_day_counts,
        "proxy_min_ts": proxy_min,
        "proxy_max_ts": proxy_max,
    }


def aggregate_inventory(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate file-level records without inventing unique-row counts."""
    rows = list(records)
    def _sum(key: str) -> int:
        return int(sum(int(record.get(key, 0) or 0) for record in rows))

    native_bounds = [
        value
        for record in rows
        for value in (record.get("native_min_ts"), record.get("native_max_ts"))
        if value
    ]
    native_day_counts: Counter[str] = Counter()
    for record in rows:
        for day, count in dict(record.get("native_day_counts") or {}).items():
            native_day_counts[str(day)] += int(count)
    native_days = sorted(native_day_counts)
    missing_native_days: list[str] = []
    if native_days:
        expected = pd.date_range(native_days[0], native_days[-1], freq="D")
        observed = set(native_days)
        missing_native_days = [
            day.strftime("%Y-%m-%d")
            for day in expected
            if day.strftime("%Y-%m-%d") not in observed
        ]
    return {
        "files": len(rows),
        "rows": _sum("rows"),
        "native_rows": _sum("native_rows"),
        "proxy_rows": _sum("proxy_rows"),
        "native_files": sum(record.get("native_rows", 0) > 0 for record in rows),
        # The current Kraken surface is one product per file.  This is a
        # file-key identity count, not a claim that arbitrary multi-product
        # files would have been safely de-duplicated.
        "native_product_file_identities": sum(
            record.get("native_rows", 0) > 0 for record in rows
        ),
        "proxy_only_files": sum(record.get("classification") == "proxy_only" for record in rows),
        "source_unclassified_files": sum(
            record.get("classification") == "source_unclassified" for record in rows
        ),
        "native_min_ts": min(native_bounds) if native_bounds else None,
        "native_max_ts": max(native_bounds) if native_bounds else None,
        "native_day_counts": dict(sorted(native_day_counts.items())),
        "native_coverage_days": len(native_days),
        "native_missing_calendar_days": missing_native_days,
    }


def assess_candidate_window(
    inventory: Mapping[str, Any],
    panels: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the full-window gate to the declared candidate panels."""
    panel_rows = list(panels)
    starts = [str(panel["min_candidate_ts"]) for panel in panel_rows if panel.get("min_candidate_ts")]
    ends = [str(panel["max_candidate_ts"]) for panel in panel_rows if panel.get("max_candidate_ts")]
    required_min = min(starts) if starts else None
    required_max = max(ends) if ends else None
    native_min = inventory.get("native_min_ts")
    native_max = inventory.get("native_max_ts")
    contains_window = bool(
        required_min
        and required_max
        and native_min
        and native_max
        and native_min <= required_min
        and native_max >= required_max
    )
    native_starts_after_requirement = bool(native_min and required_min and native_min > required_min)
    return {
        "required_candidate_min_ts": required_min,
        "required_candidate_max_ts": required_max,
        "native_min_ts": native_min,
        "native_max_ts": native_max,
        "native_window_contains_declared_candidate_window": contains_window,
        "native_starts_after_required_candidate_window": native_starts_after_requirement,
        "historical_native_backfill_required": not contains_window,
        "panel_count": len(panel_rows),
    }
