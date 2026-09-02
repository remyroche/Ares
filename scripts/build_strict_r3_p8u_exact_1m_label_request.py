#!/usr/bin/env python3
"""Create an immutable target-free exact-1m label request from P8U candidates.

The request contains only identities, decision timestamps, a fixed +5-minute
entry timestamp, and a neutral priority placeholder.  It deliberately does
not read an outcome, score, policy, portfolio, or exchange account.  The
placeholder cannot influence the policy label materialiser; it exists only
because the shared exact-path request schema is also used by portfolio tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


REQUIRED = ("candidate_id", "__decision_ts__", "__symbol__", "side_name")
KRAKEN_CHART_BASE_ALIAS = {"BTC": "XBT"}


def _kraken_charts_product_id(symbol: str) -> str:
    """Map the retained USD-linear symbol identity to Charts API syntax.

    The source map holds an upstream OHLCV product key such as ``AAVE_USDT``;
    the Kraken Futures Charts API instead requires ``PF_AAVEUSD``.  This is a
    deterministic transport translation, not a lookup against the mutable
    current instrument catalogue.
    """
    base = str(symbol).split("/", 1)[0].strip().upper()
    if not base:
        raise ValueError(f"cannot derive Kraken Charts product from {symbol!r}")
    return f"PF_{KRAKEN_CHART_BASE_ALIAS.get(base, base)}USD"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(raw: str) -> pd.Timestamp:
    stamp = pd.Timestamp(raw)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _write_once(path: Path, payload: dict[str, object]) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--start", required=True, help="inclusive decision timestamp")
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--entry-delay-minutes", type=int, default=5)
    parser.add_argument(
        "--source-manifest", type=Path,
        help="optional frozen source-map manifest; binds each request row to its historical Kraken product_id",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.entry_delay_minutes < 0:
        raise ValueError("entry delay must be non-negative")
    if args.out_dir.exists():
        raise FileExistsError(f"immutable request already exists: {args.out_dir}")
    start, end = _utc(args.start), _utc(args.end_exclusive)
    if end <= start:
        raise ValueError("end-exclusive must be after start")
    source = pd.read_parquet(args.candidates, columns=list(REQUIRED)).copy()
    source["__decision_ts__"] = pd.to_datetime(source["__decision_ts__"], utc=True, errors="raise")
    source["candidate_id"] = source["candidate_id"].astype(str)
    source["__symbol__"] = source["__symbol__"].astype(str)
    source["side_name"] = source["side_name"].astype(str).str.lower()
    if not source["side_name"].eq("long").all() or source["candidate_id"].duplicated().any():
        raise AssertionError("source candidate universe is not a unique long-only target-free ledger")
    selected = source.loc[
        source["__decision_ts__"].ge(start) & source["__decision_ts__"].lt(end)
    ].sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    if selected.empty:
        raise ValueError("selected target-free label request is empty")
    product_map: dict[str, str] = {}
    source_manifest_audit: dict[str, object] | None = None
    if args.source_manifest is not None:
        source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
        raw_map = source_manifest.get("source_map")
        if not isinstance(raw_map, dict):
            raise ValueError("source manifest lacks a source_map")
        source_product_map = {
            str(symbol): str(product).strip()
            for symbol, product in raw_map.items()
            if product is not None and str(product).strip().lower() not in {"", "none", "nan", "null"}
        }
        product_map = {
            symbol: _kraken_charts_product_id(symbol)
            for symbol in source_product_map
        }
        selected_symbols = set(selected["__symbol__"].astype(str))
        source_manifest_audit = {
            "path": str(args.source_manifest.resolve()),
            "sha256": _sha256(args.source_manifest),
        "source_map_symbols": len(raw_map),
            "frozen_product_id_symbols": len(product_map),
            "symbols_without_frozen_product_id": len(selected_symbols.difference(product_map)),
            "product_id_transport": "deterministic_source_symbol_to_kraken_charts_PF_USD_linear",
        }
    output = pd.DataFrame({
        "candidate_id": selected["candidate_id"],
        "timestamp": selected["__decision_ts__"],
        "symbol": selected["__symbol__"],
        "side_name": selected["side_name"],
        "entry_ts": selected["__decision_ts__"] + pd.Timedelta(minutes=int(args.entry_delay_minutes)),
        "priority_bps": 0.0,
    })
    if args.source_manifest is not None:
        output["product_id"] = output["symbol"].map(product_map).astype("string")
    if output["candidate_id"].duplicated().any():
        raise AssertionError("request identity collision")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    path = args.out_dir / "candidates.parquet"
    output.to_parquet(path, index=False, compression="zstd")
    _write_once(args.out_dir / "candidate_manifest.json", {
        "schema": "strict_r3_p8u_exact_1m_target_free_label_request_v1",
        "target_free": True,
        "purpose": "exact one-minute policy label materialisation only; priority_bps is neutral and has no selection authority",
        "source_candidates": str(args.candidates.resolve()),
        "source_candidates_sha256": _sha256(args.candidates),
        "candidate_sha256": _sha256(path),
        "candidate_rows": int(len(output)),
        "symbols": int(output["symbol"].nunique()),
        "decision_start": start.isoformat(),
        "decision_end_exclusive": end.isoformat(),
        "entry_delay_minutes": int(args.entry_delay_minutes),
        "source_manifest": source_manifest_audit,
        "future_path_or_outcome_filter_applied": False,
        "outcome_columns_consumed": [],
    })
    print(json.dumps({"rows": int(len(output)), "symbols": int(output['symbol'].nunique()), "out": str(args.out_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
