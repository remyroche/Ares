#!/usr/bin/env python3
"""Create an explicit, free-day-only Tardis Kraken download manifest.

The manifest is intentionally generated from official exchange metadata plus a
user-supplied exact mapping.  It never infers an internal symbol from a Tardis
symbol, and it only emits first-UTC-day-of-month requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METADATA_URL = "https://api.tardis.dev/v1/exchanges/{exchange}"
DEFAULT_DATA_TYPES = ("incremental_book_L2", "trades")
REQUIRED_MAPPING_COLUMNS = {"internal_symbol", "dataset_symbol", "valid_from"}


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError(f"invalid UTC timestamp {value!r}")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_exchange_metadata(exchange: str, *, metadata_url: str = METADATA_URL) -> dict[str, Any]:
    url = metadata_url.format(exchange=exchange)
    request = urllib.request.Request(url, headers={"User-Agent": "Ares-execution-research/1"})
    with urllib.request.urlopen(request, timeout=30) as response:  # nosec B310: official fixed endpoint/CLI override
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or not payload.get("availableSymbols"):
        raise ValueError(f"Tardis metadata at {url} does not expose availableSymbols")
    return payload


def month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Return UTC first-of-month dates in ``[start, end)``."""
    start, end = _utc(start), _utc(end)
    if end <= start:
        raise ValueError("end must be after start")
    first = start.normalize().replace(day=1)
    if first < start:
        first += pd.offsets.MonthBegin(1)
    return list(pd.date_range(first, end.normalize(), freq="MS", tz="UTC", inclusive="left"))


def load_exact_mapping(path: Path) -> pd.DataFrame:
    mapping = pd.read_csv(path)
    missing = REQUIRED_MAPPING_COLUMNS.difference(mapping.columns)
    if missing:
        raise ValueError(f"mapping lacks columns: {sorted(missing)}")
    if mapping[["internal_symbol", "dataset_symbol"]].isna().any().any():
        raise ValueError("mapping symbols may not be null")
    if mapping.duplicated(["internal_symbol", "dataset_symbol", "valid_from"], keep=False).any():
        raise ValueError("mapping has duplicate internal/dataset/valid_from rows")
    mapping = mapping.copy()
    mapping["valid_from"] = pd.to_datetime(mapping["valid_from"], utc=True, errors="coerce")
    # ``DataFrame.get`` would return ``None`` when the optional column is
    # absent, which is easy to accidentally turn into a scalar column.  Keep
    # an index-aligned all-NaT series instead: open-ended mappings are an
    # explicit, stable part of the manifest contract.
    valid_to = mapping["valid_to"] if "valid_to" in mapping.columns else pd.Series(pd.NaT, index=mapping.index)
    mapping["valid_to"] = pd.to_datetime(valid_to, utc=True, errors="coerce")
    if mapping["valid_from"].isna().any():
        raise ValueError("mapping contains invalid valid_from")
    if (mapping["valid_to"].notna() & mapping["valid_to"].le(mapping["valid_from"])).any():
        raise ValueError("mapping valid_to must be later than valid_from")
    return mapping.sort_values(["internal_symbol", "valid_from"], kind="stable").reset_index(drop=True)


def _available_symbols(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values = metadata.get("availableSymbols", [])
    output: dict[str, dict[str, Any]] = {}
    for item in values:
        if isinstance(item, str):
            output[item] = {"id": item}
        elif isinstance(item, dict) and item.get("id"):
            output[str(item["id"])] = item
    return output


def _available_dataset_types(metadata: dict[str, Any]) -> dict[str, set[str]]:
    """Return official per-Spot-symbol dataset types in download notation."""
    output: dict[str, set[str]] = {}
    for item in metadata.get("datasets", {}).get("symbols", []):
        if not isinstance(item, dict) or item.get("type") != "spot" or not item.get("id"):
            continue
        symbol = str(item["id"]).replace("-", "/")
        output[symbol] = {str(value) for value in item.get("dataTypes", [])}
    return output


def build_manifest(
    *,
    mapping: pd.DataFrame,
    metadata: dict[str, Any],
    exchange: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_types: tuple[str, ...] = DEFAULT_DATA_TYPES,
) -> pd.DataFrame:
    available = _available_symbols(metadata)
    dataset_types = _available_dataset_types(metadata)
    records: list[dict[str, Any]] = []
    for sample_date in month_starts(start, end):
        for source in mapping.to_dict("records"):
            dataset_symbol = str(source["dataset_symbol"])
            valid_from = _utc(source["valid_from"])
            valid_to = source.get("valid_to")
            valid_to = _utc(valid_to) if pd.notna(valid_to) else pd.NaT
            eligible = valid_from <= sample_date and (pd.isna(valid_to) or sample_date < valid_to)
            metadata_item = available.get(dataset_symbol)
            metadata_since = None
            if metadata_item and metadata_item.get("availableSince"):
                metadata_since = _utc(metadata_item["availableSince"])
            for data_type in data_types:
                status = "pending"
                error = ""
                if not eligible:
                    status, error = "outside_explicit_mapping_window", "sample date outside mapping validity"
                elif metadata_item is None:
                    status, error = "unavailable_symbol", "dataset symbol absent from official metadata"
                elif metadata_since is not None and sample_date < metadata_since.normalize():
                    status, error = "unavailable_before_metadata_start", "sample date precedes official metadata availability"
                elif dataset_symbol in dataset_types and data_type not in dataset_types[dataset_symbol]:
                    status, error = "unavailable_data_type", "datatype absent from official per-symbol metadata"
                records.append({
                    "exchange": exchange,
                    "dataset_symbol": dataset_symbol,
                    "internal_symbol": str(source["internal_symbol"]),
                    "valid_from": valid_from,
                    "valid_to": valid_to,
                    "available_since": metadata_since,
                    "sample_date": sample_date,
                    "data_type": data_type,
                    "url": "",  # filled by the downloader/Tardis client receipt
                    "download_target": "",
                    "status": status,
                    "file_size": pd.NA,
                    "checksum_sha256": "",
                    "error": error,
                })
    return pd.DataFrame.from_records(records).sort_values(
        ["sample_date", "dataset_symbol", "data_type"], kind="stable"
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True, help="CSV with exact internal_symbol/dataset_symbol mapping")
    parser.add_argument("--out", type=Path, required=True, help="Manifest parquet output")
    parser.add_argument("--start", required=True, help="UTC inclusive start")
    parser.add_argument("--end", required=True, help="UTC exclusive end")
    parser.add_argument("--exchange", default="kraken", choices=("kraken", "cryptofacilities"))
    parser.add_argument("--metadata-url", default=METADATA_URL)
    parser.add_argument(
        "--metadata-json", type=Path,
        help="Optional immutable official metadata response, instead of fetching at manifest time.",
    )
    parser.add_argument("--data-types", nargs="+", default=list(DEFAULT_DATA_TYPES))
    args = parser.parse_args()

    mapping = load_exact_mapping(args.mapping)
    metadata = json.loads(args.metadata_json.read_text()) if args.metadata_json else fetch_exchange_metadata(args.exchange, metadata_url=args.metadata_url)
    if not isinstance(metadata, dict) or not metadata.get("availableSymbols"):
        raise ValueError("metadata JSON does not expose availableSymbols")
    manifest = build_manifest(
        mapping=mapping, metadata=metadata, exchange=args.exchange,
        start=_utc(args.start), end=_utc(args.end), data_types=tuple(args.data_types),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.out, index=False)
    receipt = {
        "schema": "ares.tardis_free_manifest.v1",
        "target_free": True,
        "exchange": args.exchange,
        "metadata_url": args.metadata_url.format(exchange=args.exchange),
        "metadata_json": str(args.metadata_json) if args.metadata_json else None,
        "mapping": str(args.mapping),
        "mapping_sha256": _sha256(args.mapping),
        "manifest": str(args.out),
        "manifest_sha256": _sha256(args.out),
        "sample_rule": "first UTC day of each month only",
        "rows": int(len(manifest)),
        "pending": int(manifest["status"].eq("pending").sum()),
        "unavailable": int((~manifest["status"].eq("pending")).sum()),
    }
    args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(json.dumps(receipt, indent=2, default=str))


if __name__ == "__main__":
    main()
