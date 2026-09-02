#!/usr/bin/env python3
"""Backfill Kraken Futures hourly funding rates from the public history API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
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


def _load_symbols(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("source_map"), dict):
        rows = list(payload["source_map"])
    else:
        rows = payload.get("symbols") if isinstance(payload, dict) else payload
    out: list[str] = []
    for row in rows or []:
        sym = (
            row.get("perp_symbol") or row.get("symbol")
            if isinstance(row, dict) else row
        )
        if sym:
            out.append(str(sym))
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _load_local_env_if_present()
    exchange = make_perp_exchange()
    symbols = (
        [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.symbols.strip()
        else _load_symbols(Path(args.manifest))
    )
    if args.max_symbols and args.max_symbols > 0:
        symbols = symbols[: int(args.max_symbols)]

    funding_dir = Path(args.funding_dir)
    funding_dir.mkdir(parents=True, exist_ok=True)
    stats = {"ok": 0, "empty": 0, "failed": 0, "added_non_null": 0}

    workers = max(1, int(args.workers))
    tprint(
        f"Kraken historical funding API backfill start: "
        f"symbols={len(symbols)} workers={workers}"
    )
    jobs = [
        (i, symbol, _product_id(exchange, symbol))
        for i, symbol in enumerate(symbols, start=1)
    ]
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

    for row in sorted(results, key=lambda item: int(item["ordinal"])):
        status = str(row["status"])
        stats[status] += 1
        stats["added_non_null"] += int(row["added_non_null"])
        tprint(
            f"[{int(row['ordinal']):04d}/{len(jobs):04d}] "
            f"{row['symbol']} ({row['product_id']}) {row['detail']}"
        )

    tprint(f"Kraken historical funding API backfill complete: {stats}")
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0 if stats["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
