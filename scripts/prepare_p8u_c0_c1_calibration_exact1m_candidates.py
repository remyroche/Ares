#!/usr/bin/env python3
"""Seal full-Router50 target-free exact-one-minute calibration requests.

The C0/C1 daily EV mapper is calibrated on all causally routed opportunities,
not just trades the current mapper happens to admit.  This utility freezes the
entire Router50 score population and its historical Kraken product identity
*before* opening any one-minute path.  It has no outcome, model, portfolio,
account, or order authority.
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


IDENTITY = ("candidate_id", "__decision_ts__", "__symbol__", "side_name")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _router50(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    required = {*IDENTITY, "base_rank_ts"}
    if missing := required.difference(frame.columns):
        raise KeyError(f"upstream target-free score lacks {sorted(missing)}")
    forbidden = [
        column for column in frame.columns
        if any(token in column.lower() for token in ("outcome", "policy_", "label", "net_bps", "gross_bps"))
    ]
    if forbidden:
        raise ValueError(f"upstream score carries forbidden outcome fields: {forbidden[:4]}")
    work = frame.copy()
    for field in ("candidate_id", "__symbol__"):
        work[field] = work[field].astype(str)
    work["side_name"] = work["side_name"].astype(str).str.lower()
    work["__decision_ts__"] = pd.to_datetime(work["__decision_ts__"], utc=True, errors="raise")
    if work.duplicated(list(IDENTITY)).any() or not work["side_name"].eq("long").all():
        raise ValueError("upstream score violates long-only target-free identity")
    explicit_router50 = "router50_eligible" in work.columns
    if explicit_router50:
        routed = work["router50_eligible"].fillna(False).astype(bool)
    else:
        # Historical v7 OOF rows persist only Base scores inside Router50.
        routed = np.isfinite(pd.to_numeric(work["base_rank_ts"], errors="coerce"))
    result = work.loc[routed].copy()
    if result.empty:
        raise ValueError("no Router50 rows in target-free upstream score")
    if not np.isfinite(pd.to_numeric(result["base_rank_ts"], errors="coerce")).all():
        raise AssertionError("Router50 contains a non-finite Base rank")
    if explicit_router50:
        by_time = result.groupby("__decision_ts__", sort=False)["candidate_id"].size()
        full = work.groupby("__decision_ts__", sort=False)["candidate_id"].size()
        expected = np.ceil(full.reindex(by_time.index).to_numpy(float) * .50).astype(int)
        if not np.array_equal(by_time.to_numpy(int), expected):
            raise AssertionError("Router50 request does not satisfy exact timestamp-local ceil(50%)")
        provenance = "explicit_full_universe_router50"
    else:
        # The sealed historical source stores only the already-routed Base
        # rows.  It cannot prove a fresh 160-row denominator, so keep that
        # distinction explicit rather than inventing a false 50% check.
        provenance = "sealed_prerouted_router50_source"
    return result.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True), provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-scores", type=Path, required=True)
    parser.add_argument("--frozen-source-manifest", type=Path, required=True)
    parser.add_argument("--frozen-kraken-product-ledger", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--entry-delay-minutes", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start, end = _utc(args.start), _utc(args.end)
    if end <= start or int(args.entry_delay_minutes) < 0:
        raise ValueError("invalid request window or entry delay")
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError("calibration candidate request must be immutable")
    source_manifest_path = args.frozen_source_manifest.resolve()
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_map = source_manifest.get("source_map")
    if not isinstance(source_map, dict) or len(source_map) != 160:
        raise AssertionError("frozen source manifest does not bind the 160-symbol universe")
    product_path = args.frozen_kraken_product_ledger.resolve()
    product = pd.read_parquet(product_path, columns=["symbol", "product_id"]).copy()
    product["symbol"] = product["symbol"].astype(str)
    product["product_id"] = product["product_id"].astype("string")
    valid_product = product["product_id"].notna() & product["product_id"].astype(str).str.strip().ne("")
    product = product.loc[valid_product].copy()
    if product.groupby("symbol")["product_id"].nunique().gt(1).any():
        raise AssertionError("frozen product ledger maps one symbol to multiple products")
    product_map = product.drop_duplicates("symbol").set_index("symbol")["product_id"].astype(str).to_dict()

    raw = pd.read_parquet(args.upstream_scores.resolve())
    routed, router50_provenance = _router50(raw)
    routed = routed.loc[routed["__decision_ts__"].ge(start) & routed["__decision_ts__"].lt(end)].copy()
    if routed.empty:
        raise ValueError("no Router50 candidates inside requested window")
    if set(routed["__symbol__"]).difference(source_map):
        raise AssertionError("Router50 candidate escaped the frozen source universe")
    candidates = pd.DataFrame({
        "candidate_id": routed["candidate_id"].astype(str),
        "timestamp": routed["__decision_ts__"],
        "symbol": routed["__symbol__"].astype(str),
        "side_name": "long",
        "entry_ts": routed["__decision_ts__"] + pd.Timedelta(minutes=int(args.entry_delay_minutes)),
        "priority_bps": pd.to_numeric(routed["base_rank_ts"], errors="raise"),
    })
    candidates["product_id"] = candidates["symbol"].map(product_map).astype("string")
    candidates = candidates.sort_values(["timestamp", "priority_bps", "candidate_id"], ascending=[True, False, True], kind="stable").reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=False)
    candidates.to_parquet(output / "candidates.parquet", index=False, compression="zstd")
    candidates.loc[:, ["candidate_id", "timestamp", "symbol", "entry_ts", "product_id"]].to_parquet(
        output / "minute_download_candidates.parquet", index=False, compression="zstd"
    )
    _write_once(output / "candidate_manifest.json", {
        "schema": "p8u_c0_c1_calibration_exact1m_candidates_v1",
        "status": "complete_target_free_router50_request",
        "scope": "full Router50 exact-one-minute calibration request; no outcome/model/portfolio/exchange/order authority",
        "target_free": True,
        "selection": "all exact timestamp-local Router50 rows; not mapper admissions",
        "router50_provenance": router50_provenance,
        "period": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "entry_delay_minutes": int(args.entry_delay_minutes),
        "candidate_rows": int(len(candidates)),
        "candidate_sha256": _sha256(output / "candidates.parquet"),
        "known_frozen_product_rows": int(candidates["product_id"].notna().sum()),
        "missing_frozen_product_rows": int(candidates["product_id"].isna().sum()),
        "upstream_scores": {"path": str(args.upstream_scores.resolve()), "sha256": _sha256(args.upstream_scores.resolve())},
        "frozen_source_manifest": {"path": str(source_manifest_path), "sha256": _sha256(source_manifest_path)},
        "frozen_kraken_product_ledger": {"path": str(product_path), "sha256": _sha256(product_path)},
        "causality": {
            "candidate_selection": "target-free Router50 scores only",
            "product_identity": "frozen historical Kraken product ledger only",
            "forbidden_inputs": ["future path", "policy outcome", "label validity", "mapper admission", "portfolio result"],
        },
    })
    print(output)


if __name__ == "__main__":
    main()
