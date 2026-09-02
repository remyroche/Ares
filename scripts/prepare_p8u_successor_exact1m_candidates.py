#!/usr/bin/env python3
"""Seal target-free exact-one-minute requests from successor C0/C1 admissions.

This bridge intentionally knows only the already sealed, target-free mapper
selection plus the frozen historical source map.  It is suitable for forward
path recovery: entries and frozen Kraken product identities are fixed before
the incremental downloader opens a one-minute source or any policy outcome is
materialised.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapper", type=Path, required=True)
    parser.add_argument("--frozen-source-manifest", type=Path, required=True)
    parser.add_argument(
        "--frozen-kraken-product-ledger", type=Path, required=True,
        help="prior immutable exact-1m candidate Parquet containing the frozen Kraken chart product_id map",
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--entry-delay-minutes", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start, end = _utc(args.start), _utc(args.end)
    if end <= start:
        raise ValueError("require start < end")
    if int(args.entry_delay_minutes) < 0:
        raise ValueError("entry delay must be non-negative")
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    mapper_root = args.mapper.resolve()
    route_path = mapper_root / "agreement_tier_target_free_predictions.parquet"
    mapper_manifest_path = mapper_root / "run_manifest.json"
    if not route_path.is_file() or not mapper_manifest_path.is_file():
        raise FileNotFoundError("mapper target-free admission receipt is incomplete")
    mapper_manifest = json.loads(mapper_manifest_path.read_text(encoding="utf-8"))
    if mapper_manifest.get("status") != "complete":
        raise AssertionError("mapper receipt is incomplete")
    route = pd.read_parquet(route_path).copy()
    forbidden = sorted(
        column for column in route.columns
        if column.startswith("policy_") or "outcome" in column.lower() or "label" in column.lower()
    )
    if forbidden:
        raise AssertionError(f"target-free mapper route has forbidden columns: {forbidden}")
    required = {"candidate_id", "__decision_ts__", "__symbol__", "side_name", "portfolio_order_priority_bps"}
    missing = sorted(required.difference(route.columns))
    if missing:
        raise KeyError(f"mapper route lacks {missing}")
    route["candidate_id"] = route["candidate_id"].astype(str)
    route["__decision_ts__"] = pd.to_datetime(route["__decision_ts__"], utc=True, errors="raise")
    route["__symbol__"] = route["__symbol__"].astype(str)
    route["side_name"] = route["side_name"].astype(str).str.lower()
    route["portfolio_order_priority_bps"] = pd.to_numeric(route["portfolio_order_priority_bps"], errors="raise")
    scoped = route.loc[
        route["__decision_ts__"].ge(start) & route["__decision_ts__"].lt(end)
    ].copy()
    if scoped.empty:
        raise RuntimeError("no target-free mapper admissions in requested interval")
    if scoped.duplicated("candidate_id").any() or not scoped["side_name"].eq("long").all():
        raise AssertionError("invalid long-only candidate identities")
    if not np.isfinite(scoped["portfolio_order_priority_bps"].to_numpy(float)).all():
        raise AssertionError("candidate priority must be finite")
    source_manifest_path = args.frozen_source_manifest.resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_map = source_manifest.get("source_map")
    if not isinstance(source_map, dict) or len(source_map) != 160:
        raise AssertionError("frozen source manifest does not declare the 160-symbol source map")
    # The upstream source-map aliases (for example ``ONT_USDT``) are not
    # Kraken Futures chart identifiers.  Product IDs must come only from a
    # prior immutable Kraken request ledger (for example ``PF_ONTUSD``), not
    # from the current catalog and not from a similarly named source alias.
    product_ledger = args.frozen_kraken_product_ledger.resolve()
    product_frame = pd.read_parquet(product_ledger, columns=["symbol", "product_id"]).copy()
    product_frame["symbol"] = product_frame["symbol"].astype(str)
    product_frame["product_id"] = product_frame["product_id"].astype("string")
    product_frame = product_frame.loc[
        product_frame["product_id"].notna()
        & product_frame["product_id"].astype(str).str.strip().ne("")
        & product_frame["product_id"].astype(str).str.lower().ne("none")
    ].copy()
    if product_frame.groupby("symbol")["product_id"].nunique().gt(1).any():
        raise AssertionError("frozen Kraken product ledger assigns multiple IDs to one symbol")
    product_map = product_frame.drop_duplicates("symbol").set_index("symbol")["product_id"].astype(str).to_dict()
    candidates = pd.DataFrame({
        "candidate_id": scoped["candidate_id"],
        "timestamp": scoped["__decision_ts__"],
        "symbol": scoped["__symbol__"],
        "side_name": scoped["side_name"],
        "entry_ts": scoped["__decision_ts__"] + pd.Timedelta(minutes=int(args.entry_delay_minutes)),
        "priority_bps": scoped["portfolio_order_priority_bps"],
    }).sort_values(["timestamp", "priority_bps", "candidate_id"], ascending=[True, False, True], kind="stable").reset_index(drop=True)
    candidates["product_id"] = candidates["symbol"].map(product_map).astype("string")
    if set(candidates["symbol"]).difference(source_map):
        raise AssertionError("mapper candidate escaped frozen source universe")
    out.mkdir(parents=True, exist_ok=False)
    candidate_path = out / "candidates.parquet"
    candidates.to_parquet(candidate_path, index=False, compression="zstd")
    minute_path = out / "minute_download_candidates.parquet"
    candidates.loc[:, ["candidate_id", "timestamp", "symbol", "entry_ts", "product_id"]].to_parquet(
        minute_path, index=False, compression="zstd"
    )
    manifest = {
        "schema": "p8u_successor_c0_c1_exact1m_targetfree_candidates_v1",
        "target_free": True,
        "scope": "offline forward exact-one-minute candidate request; no policy/outcome/exchange authority",
        "period": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "entry_delay_minutes": int(args.entry_delay_minutes),
        "selection": "sealed C0/C1 agreement-tier mapped-EV admissions only; no outcome field read",
        "candidate_rows": int(len(candidates)),
        # Kept at the historical top-level location for compatibility with the
        # existing exact-one-minute policy materialiser, which independently
        # verifies this request before opening any path source.
        "candidate_sha256": _sha256(candidate_path),
        "known_frozen_product_rows": int(candidates["product_id"].notna().sum()),
        "missing_frozen_product_rows": int(candidates["product_id"].isna().sum()),
        "mapper": {"path": str(mapper_root), "manifest_sha256": _sha256(mapper_manifest_path), "route_sha256": _sha256(route_path)},
        "frozen_source_manifest": {"path": str(source_manifest_path), "sha256": _sha256(source_manifest_path)},
        "frozen_kraken_product_ledger": {"path": str(product_ledger), "sha256": _sha256(product_ledger)},
        "outputs": {"candidates.parquet": _sha256(candidate_path), "minute_download_candidates.parquet": _sha256(minute_path)},
        "causality": {
            "candidate_selection": "target-free mapper route only",
            "product_identity": "frozen historical Kraken chart product ledger only; upstream source aliases are never sent to Kraken",
            "forbidden_inputs": ["future path", "one-minute outcome", "policy outcome", "label validity", "portfolio result"],
        },
    }
    _write_once(out / "candidate_manifest.json", manifest)
    print(json.dumps({"out": str(out), "rows": len(candidates), "known_product_rows": int(candidates.product_id.notna().sum())}, sort_keys=True))


if __name__ == "__main__":
    main()
