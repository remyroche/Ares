#!/usr/bin/env python3
"""Seal an immutable Kraken Futures ``PF_*`` product-identity ledger.

This is deliberately an identity-only bridge.  It reads a pre-existing local
market-definition snapshot and a frozen P8U universe manifest; it makes no
network request and it never reads score, policy, outcome, or portfolio data.
The resulting ledger lets exact-one-minute replay distinguish a missing chart
identity from a missing price path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, payload: dict) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--market-definitions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    source_path = args.source_manifest.resolve()
    market_path = args.market_definitions.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    symbols = [str(value) for value in source.get("symbols", [])]
    if len(symbols) != 160 or len(set(symbols)) != 160:
        raise AssertionError("source manifest must declare exactly 160 unique symbols")
    market_payload = json.loads(market_path.read_text(encoding="utf-8"))
    markets = market_payload.get("markets")
    if not isinstance(markets, dict):
        raise ValueError("market definition snapshot lacks markets mapping")

    rows: list[dict[str, object]] = []
    for symbol in symbols:
        market = markets.get(symbol)
        product_id = None
        if isinstance(market, dict):
            candidate = str(market.get("id") or "")
            native = str((market.get("info") or {}).get("symbol") or "")
            if candidate.startswith("PF_") and candidate == native:
                product_id = candidate
        rows.append({"symbol": symbol, "product_id": product_id})
    ledger = pd.DataFrame(rows).sort_values("symbol", kind="stable").reset_index(drop=True)
    if ledger["product_id"].dropna().duplicated().any():
        raise AssertionError("one Kraken product identity maps to more than one P8U symbol")

    out.mkdir(parents=True, exist_ok=False)
    ledger_path = out / "kraken_product_ledger.parquet"
    ledger.to_parquet(ledger_path, index=False, compression="zstd")
    manifest = {
        "schema": "p8u_kraken_product_identity_ledger_v1",
        "scope": "identity-only local snapshot seal; no network, score, policy, outcome, or portfolio input",
        "source_manifest": {"path": str(source_path), "sha256": _sha256(source_path)},
        "market_definitions": {"path": str(market_path), "sha256": _sha256(market_path)},
        "symbols": int(len(ledger)),
        "mapped": int(ledger["product_id"].notna().sum()),
        "unmapped_symbols": ledger.loc[ledger["product_id"].isna(), "symbol"].astype(str).tolist(),
        "output": {"kraken_product_ledger.parquet": _sha256(ledger_path)},
    }
    _write_once(out / "manifest.json", manifest)
    print(json.dumps({"out": str(out), "mapped": manifest["mapped"], "unmapped": manifest["unmapped_symbols"]}, sort_keys=True))


if __name__ == "__main__":
    main()
