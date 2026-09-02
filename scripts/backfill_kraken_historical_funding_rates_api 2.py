#!/usr/bin/env python3
"""Backfill Kraken Futures hourly funding rates from the public history API."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import _load_local_env_if_present, make_perp_exchange
from extreme_price_movements.utils import tprint


ENDPOINT = "https://futures.kraken.com/derivatives/api/v3/historical-funding-rates"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _frozen_pf_product_id(symbol: str) -> str:
    """Derive the Kraken Futures Charts/Funding product from frozen identity.

    The frozen source map determines whether a historical product identity is
    available; this deterministic transport spelling deliberately does *not*
    consult today's exchange catalogue.
    """
    text = str(symbol or "").strip()
    if "/" not in text:
        raise ValueError(f"expected canonical perp symbol, got {symbol!r}")
    base = text.split("/", 1)[0].upper()
    return f"PF_{'XBT' if base == 'BTC' else base}USD"


def _load_symbols(manifest_path: Path) -> list[tuple[str, str | None]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("source_map"), dict):
        # A non-null value proves the historical source had a fixed product
        # identity.  The value itself is an upstream OHLCV ID (for example
        # ``AAVE_USDT``), not a Kraken Charts/Funding endpoint ID, so convert
        # only its *availability* into the deterministic PF spelling.
        return [
            (str(symbol), _frozen_pf_product_id(str(symbol)) if source_id else None)
            for symbol, source_id in sorted(payload["source_map"].items())
        ]
    rows = payload.get("symbols") if isinstance(payload, dict) else payload
    out: list[tuple[str, str | None]] = []
    for row in rows or []:
        sym = row.get("perp_symbol") or row.get("symbol") if isinstance(row, dict) else row
        if sym:
            out.append((str(sym), None))
    return list(dict.fromkeys(out))


def _symbol_to_filename(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace(":", "_")


def _fallback_product_id(symbol: str) -> str:
    text = str(symbol or "").strip()
    if "/" not in text:
        return text.replace("/", "").replace(":", "").upper()
    base, raw_quote = text.split("/", 1)
    quote = raw_quote.split(":", 1)[0]
    base = "XBT" if base.upper() == "BTC" else base.upper()
    return f"PF_{base}{quote.upper()}"


def _product_id(exchange: Any, symbol: str) -> str:
    try:
        market = exchange.market(symbol)
        market_id = str(market.get("id") or "").strip()
        if market_id:
            return market_id
    except Exception:
        pass
    return _fallback_product_id(symbol)


def _fetch_funding(product_id: str, session: requests.Session, timeout: float) -> pd.DataFrame:
    response = session.get(ENDPOINT, params={"symbol": product_id}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if str(payload.get("result", "")).lower() != "success":
        raise RuntimeError(f"Kraken funding API error for {product_id}: {payload}")
    rates = payload.get("rates") or []
    if not rates:
        return pd.DataFrame(columns=["funding_rate"])
    df = pd.DataFrame(rates)
    if "timestamp" not in df.columns:
        return pd.DataFrame(columns=["funding_rate"])
    rate_col = "relativeFundingRate" if "relativeFundingRate" in df.columns else "fundingRate"
    if rate_col not in df.columns:
        return pd.DataFrame(columns=["funding_rate"])
    out = pd.DataFrame(index=pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.floor("h"))
    out = out[~out.index.isna()]
    out["funding_rate"] = pd.to_numeric(df[rate_col], errors="coerce").to_numpy()
    out = out.dropna(subset=["funding_rate"]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.astype({"funding_rate": "float32"})


def _merge_funding(path: Path, funding: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    if path.exists():
        old = pd.read_parquet(path)
        old.index = pd.to_datetime(old.index, utc=True, errors="coerce").floor("h")
        old = old[~old.index.isna()].sort_index()
        old = old[~old.index.duplicated(keep="last")]
    else:
        old = pd.DataFrame()

    before = 0
    if not old.empty and "funding_rate" in old.columns:
        before = int(pd.to_numeric(old["funding_rate"], errors="coerce").notna().sum())

    if old.empty:
        merged = funding.copy()
    else:
        all_index = old.index.union(funding.index).sort_values()
        merged = old.reindex(all_index)
        prior = (
            pd.to_numeric(merged.get("funding_rate"), errors="coerce")
            if "funding_rate" in merged.columns
            else pd.Series(index=all_index, dtype="float32")
        )
        # The live feature ledger is append-only: a later public-history pull
        # may revise a previously observed funding value, but it must never
        # rewrite a timestamp already used by a sealed decision.  Keep the
        # stored value and use the API only for genuinely missing/new rows.
        merged["funding_rate"] = prior.combine_first(
            funding["funding_rate"].reindex(all_index)
        )

    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    after = int(pd.to_numeric(merged["funding_rate"], errors="coerce").notna().sum())
    return merged, before, after


def _backfill_one(
    *,
    ordinal: int,
    total: int,
    symbol: str,
    product_id: str,
    funding_dir: Path,
    timeout: float,
    sleep_seconds: float,
    dry_run: bool,
) -> dict[str, Any]:
    """Fetch one symbol with an isolated HTTP session and symbol-local write.

    The funding ledger is append-only: each worker owns a distinct symbol file,
    while ``_merge_funding`` preserves values already present in that file.
    """
    try:
        with requests.Session() as session:
            funding = _fetch_funding(product_id, session, timeout)
        if funding.empty:
            return {
                "ordinal": ordinal,
                "symbol": symbol,
                "product_id": product_id,
                "status": "empty",
                "added_non_null": 0,
                "detail": "empty",
            }
        path = funding_dir / f"{_symbol_to_filename(symbol)}.parquet"
        merged, before, after = _merge_funding(path, funding)
        added = max(0, after - before)
        if not dry_run:
            merged.to_parquet(path, compression="zstd")
        return {
            "ordinal": ordinal,
            "symbol": symbol,
            "product_id": product_id,
            "status": "ok",
            "added_non_null": int(added),
            "detail": (
                f"api_rows={len(funding)} funding_non_null={before}->{after} "
                f"span={funding.index.min()}->{funding.index.max()}"
            ),
        }
    except Exception as exc:
        return {
            "ordinal": ordinal,
            "symbol": symbol,
            "product_id": product_id,
            "status": "failed",
            "added_non_null": 0,
            "detail": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data_perp/exchanges/krakenfutures/manifests/kraken_dual_market_universe_latest.json",
        help="Symbol list, exchange manifest, or sealed target-free source_map manifest.",
    )
    parser.add_argument("--funding-dir", default="data_perp/exchanges/krakenfutures/funding_hourly")
    parser.add_argument("--symbols", default="", help="Comma-separated ccxt symbols to backfill.")
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent symbol-local fetches; sealed funding rows remain append-only.",
    )
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="optional immutable JSON receipt for this append-only refresh",
    )
    parser.add_argument(
        "--require-frozen-product-id",
        action="store_true",
        help=(
            "with a frozen source_map manifest, fetch only symbols whose historical "
            "product identity is present; never consult the current exchange catalog"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _load_local_env_if_present()
    exchange = make_perp_exchange()
    if args.symbols.strip():
        symbol_products = [(s.strip(), None) for s in args.symbols.split(",") if s.strip()]
    else:
        symbol_products = _load_symbols(Path(args.manifest))
    if args.max_symbols and args.max_symbols > 0:
        symbol_products = symbol_products[: int(args.max_symbols)]

    funding_dir = Path(args.funding_dir)
    funding_dir.mkdir(parents=True, exist_ok=True)
    stats = {"ok": 0, "empty": 0, "failed": 0, "skipped_missing_frozen_product_id": 0, "added_non_null": 0}

    workers = max(1, int(args.workers))
    tprint(
        f"Kraken historical funding API backfill start: "
        f"symbols={len(symbol_products)} workers={workers}"
    )
    jobs: list[tuple[int, str, str]] = []
    skipped: list[dict[str, Any]] = []
    for i, (symbol, frozen_product_id) in enumerate(symbol_products, start=1):
        if args.require_frozen_product_id:
            if not frozen_product_id:
                skipped.append({
                    "ordinal": i,
                    "symbol": symbol,
                    "product_id": None,
                    "status": "skipped_missing_frozen_product_id",
                    "added_non_null": 0,
                    "detail": "frozen source map has no historical product identity",
                })
                continue
            product_id = frozen_product_id
        else:
            product_id = frozen_product_id or _product_id(exchange, symbol)
        jobs.append((i, symbol, product_id))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _backfill_one,
                ordinal=i,
                total=len(jobs),
                symbol=symbol,
                product_id=product_id,
                funding_dir=funding_dir,
                timeout=float(args.timeout),
                sleep_seconds=max(0.0, float(args.sleep_seconds)),
                dry_run=bool(args.dry_run),
            )
            for i, symbol, product_id in jobs
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    all_results = sorted([*results, *skipped], key=lambda item: int(item["ordinal"]))
    for row in all_results:
        status = str(row["status"])
        stats[status] += 1
        stats["added_non_null"] += int(row["added_non_null"])
        tprint(
            f"[{int(row['ordinal']):04d}/{len(symbol_products):04d}] "
            f"{row['symbol']} ({row['product_id']}) {row['detail']}"
        )

    receipt_payload = {
        "schema": "kraken_historical_funding_append_only_v2",
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": _sha256(Path(args.manifest)),
        "funding_dir": str(funding_dir.resolve()),
        "require_frozen_product_id": bool(args.require_frozen_product_id),
        "dry_run": bool(args.dry_run),
        "workers": workers,
        "timeout": float(args.timeout),
        "source_contract": (
            "frozen_source_map_deterministic_pf_transport_or_fail_closed"
            if args.require_frozen_product_id
            else "legacy_catalog_fallback_permitted"
        ),
        "stats": stats,
        "results": all_results,
    }
    if args.receipt is not None:
        receipt = Path(args.receipt)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        if receipt.exists():
            raise FileExistsError(f"refusing to overwrite immutable receipt: {receipt}")
        temporary = receipt.with_name(f".{receipt.name}.tmp.{os.getpid()}")
        try:
            temporary.write_text(json.dumps(receipt_payload, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary, receipt)
        finally:
            temporary.unlink(missing_ok=True)
    tprint(f"Kraken historical funding API backfill complete: {stats}")
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0 if stats["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
