#!/usr/bin/env python3
"""Build an auditable exact Kraken-perp to Tardis Kraken-Spot mapping.

The frozen Strict-R3 universe is Kraken perpetual notation, for example
``SOL/USD:USD``.  This script permits only the one explicit conversion to
Kraken Spot ``SOL/USD`` and then requires that exact value to occur in the
official Tardis Kraken *spot* metadata.  It never applies fuzzy aliases,
symbol search, or quote substitutions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.execution.build_tardis_free_manifest import fetch_exchange_metadata  # noqa: E402


PERP_USD_PATTERN = re.compile(r"^(?P<base>[^/:]+)/USD:USD$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_spot_symbols(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for item in metadata.get("availableSymbols", []):
        if not isinstance(item, dict) or item.get("type") != "spot" or not item.get("id"):
            continue
        output[str(item["id"])] = item
    return output


def build_exact_mapping(
    symbols: list[str],
    *,
    metadata: dict[str, Any],
    valid_from: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return accepted exact mappings and a complete unmapped/rejected audit."""
    spot = _metadata_spot_symbols(metadata)
    mapping_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for internal_symbol in sorted(set(str(value) for value in symbols)):
        match = PERP_USD_PATTERN.fullmatch(internal_symbol)
        if not match:
            audit_rows.append({
                "internal_symbol": internal_symbol, "dataset_symbol": "", "status": "unsupported_perp_symbol_contract",
                "reason": "only exact BASE/USD:USD perpetual notation can map to exact BASE/USD spot",
            })
            continue
        dataset_symbol = f"{match.group('base')}/USD"
        item = spot.get(dataset_symbol)
        if item is None:
            audit_rows.append({
                "internal_symbol": internal_symbol, "dataset_symbol": dataset_symbol, "status": "absent_from_official_tardis_spot_metadata",
                "reason": "no exact Kraken Spot Tardis id; no alias or quote conversion allowed",
            })
            continue
        available_since = pd.to_datetime(item.get("availableSince"), utc=True, errors="coerce")
        mapping_rows.append({
            "internal_symbol": internal_symbol,
            "dataset_symbol": dataset_symbol,
            "valid_from": valid_from,
            "valid_to": "",
            "metadata_available_since": available_since,
            "mapping_rule": "exact_BASE/USD:USD_to_BASE/USD_verified_in_official_tardis_spot_metadata",
        })
        audit_rows.append({
            "internal_symbol": internal_symbol, "dataset_symbol": dataset_symbol, "status": "mapped",
            "reason": "exact spot identifier present in official metadata",
        })
    mapping = pd.DataFrame.from_records(mapping_rows)
    if mapping.empty:
        mapping = pd.DataFrame(columns=["internal_symbol", "dataset_symbol", "valid_from", "valid_to", "metadata_available_since", "mapping_rule"])
    audit = pd.DataFrame.from_records(audit_rows)
    return mapping, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True, help="Frozen target-free candidate population parquet")
    parser.add_argument("--out-mapping", type=Path, required=True)
    parser.add_argument("--out-audit", type=Path, required=True)
    parser.add_argument("--symbol-column", default="__symbol__")
    parser.add_argument("--valid-from", default="2024-01-01T00:00:00Z")
    parser.add_argument(
        "--metadata-json", type=Path,
        help="Optional immutable official Tardis metadata response; avoids a local TLS-store dependency.",
    )
    args = parser.parse_args()

    candidates = pd.read_parquet(args.candidates, columns=[args.symbol_column])
    symbols = candidates[args.symbol_column].dropna().astype(str).unique().tolist()
    metadata = json.loads(args.metadata_json.read_text()) if args.metadata_json else fetch_exchange_metadata("kraken")
    if not isinstance(metadata, dict) or not metadata.get("availableSymbols"):
        raise ValueError("metadata JSON does not expose availableSymbols")
    valid_from = pd.Timestamp(args.valid_from)
    valid_from = valid_from.tz_localize("UTC") if valid_from.tzinfo is None else valid_from.tz_convert("UTC")
    mapping, audit = build_exact_mapping(symbols, metadata=metadata, valid_from=valid_from)
    args.out_mapping.parent.mkdir(parents=True, exist_ok=True)
    args.out_audit.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.out_mapping, index=False)
    audit.to_parquet(args.out_audit, index=False)
    metadata_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    receipt = {
        "schema": "ares.kraken_perp_to_tardis_spot_exact_mapping.v1",
        "candidate_population": str(args.candidates),
        "candidate_sha256": _sha256(args.candidates),
        "official_metadata": "https://api.tardis.dev/v1/exchanges/kraken",
        "official_metadata_sha256": metadata_hash,
        "mapping_rule": "exact BASE/USD:USD -> BASE/USD, then exact official Tardis Spot metadata membership",
        "symbols_in": int(len(symbols)),
        "mapped": int(len(mapping)),
        "unmapped": int(len(audit) - audit["status"].eq("mapped").sum()),
        "mapping": str(args.out_mapping),
        "audit": str(args.out_audit),
    }
    args.out_mapping.with_suffix(".json").write_text(json.dumps(receipt, indent=2, default=str) + "\n")
    print(json.dumps(receipt, indent=2, default=str))


if __name__ == "__main__":
    main()
