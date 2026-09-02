#!/usr/bin/env python3
"""Seal target-free P8U candidates for an exact one-minute exit audit.

This producer deliberately consumes only current/BCF MC1 predictions and
identity fields.  Exact minute paths, policy outcomes, and any feature target
are outside this stage and can be attached only after this immutable request
has been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_policy_contract import Exact1mExecutionContract


DEFAULT_DUAL = ROOT / (
    "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_nov25_aug27_"
    "20260828_v1/dual_predictions.parquet"
)
DEFAULT_SOURCE_STATE = ROOT / (
    "data_perp/artifacts/strict_r3_p8u_canonical_source_append_20260829_t17_v1/"
    "source_panel_state.joblib"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def materialize(args: argparse.Namespace) -> Path:
    out = Path(args.out_dir).resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    start, end = _utc(args.start), _utc(args.end)
    if end <= start:
        raise ValueError("end must be after start")
    threshold = float(args.threshold_bps)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    contract = Exact1mExecutionContract(entry_delay_minutes=int(args.entry_delay_minutes))
    contract.validate()

    dual_path = Path(args.dual_predictions).resolve()
    source_path = Path(args.source_state).resolve()
    frame = pd.read_parquet(
        dual_path,
        columns=[
            "candidate_id", "__decision_ts__", "__symbol__", "side_name",
            "bcf_mc1_expected_bps", "current_mc1_expected_bps",
        ],
    ).rename(columns={"__decision_ts__": "timestamp", "__symbol__": "symbol"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    for field in ("bcf_mc1_expected_bps", "current_mc1_expected_bps"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("dual MC1 input has duplicate candidate identities")
    if not frame["side_name"].eq("long").all():
        raise AssertionError("P8U exact-exit request must remain long-only")
    selected = frame.loc[
        frame["timestamp"].ge(start)
        & frame["timestamp"].lt(end)
        & frame["bcf_mc1_expected_bps"].ge(threshold)
        & frame["current_mc1_expected_bps"].ge(threshold)
    ].copy()
    if selected.empty:
        raise RuntimeError("no target-free dual-MC1 candidates pass the frozen threshold")
    if not np.isfinite(selected.loc[:, ["bcf_mc1_expected_bps", "current_mc1_expected_bps"]].to_numpy(float)).all():
        raise AssertionError("selected target-free candidates contain non-finite MC1 values")

    source = joblib.load(source_path)
    if not isinstance(source, Mapping):
        raise ValueError("source state is not a mapping")
    product_map = source.get("source_map")
    frozen_symbols = tuple(map(str, source.get("symbols") or ()))
    if not isinstance(product_map, Mapping) or len(frozen_symbols) != 160:
        raise ValueError("source state lacks frozen 160-symbol source mapping")
    unexpected = sorted(set(selected["symbol"]).difference(frozen_symbols))
    if unexpected:
        raise AssertionError(f"selected symbols escape frozen P8U source universe: {unexpected[:5]}")

    candidates = pd.DataFrame({
        "candidate_id": selected["candidate_id"],
        "timestamp": selected["timestamp"],
        "symbol": selected["symbol"],
        "side_name": selected["side_name"],
        "entry_ts": selected["timestamp"] + pd.Timedelta(minutes=int(contract.entry_delay_minutes)),
        "priority_bps": selected["bcf_mc1_expected_bps"],
    }).sort_values(["timestamp", "priority_bps", "candidate_id"], ascending=[True, False, True], kind="stable").reset_index(drop=True)
    if candidates["candidate_id"].duplicated().any():
        raise AssertionError("candidate request has duplicate identities")
    candidates["source_product_id"] = candidates["symbol"].map(product_map)
    known_products = int(candidates["source_product_id"].notna().sum())

    out.mkdir(parents=True, exist_ok=False)
    candidate_path = out / "candidates.parquet"
    candidates.to_parquet(candidate_path, index=False, compression="zstd")
    # The frozen P8U ``source_map`` belongs to the upstream source provider,
    # not Kraken Futures.  Its values must never be passed through as chart
    # API product IDs.  The execution downloader resolves the canonical
    # Kraken product from the frozen symbol identity itself.
    download_candidates = candidates.loc[:, ["candidate_id", "timestamp", "symbol"]].copy()
    download_path = out / "minute_download_candidates.parquet"
    download_candidates.to_parquet(download_path, index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_p8u_exact_1m_rich_targetfree_candidates_v1",
        "target_free": True,
        "side": "long",
        "period": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "entry": "decision timestamp plus declared execution delay; no future path queried",
        "entry_delay_minutes": int(contract.entry_delay_minutes),
        "contract_hash": contract.hash,
        "selection_inputs": [
            "bcf_mc1_expected_bps", "current_mc1_expected_bps", "candidate identity", "decision timestamp",
        ],
        "forbidden_selection_inputs": [
            "future path", "one-minute OHLCV after decision", "policy outcome", "exit", "MFE", "MAE", "label validity",
        ],
        "selection": {
            "bcf_mc1_expected_bps_gte": threshold,
            "current_mc1_expected_bps_gte": threshold,
            "priority": "bcf_mc1_expected_bps",
        },
        "candidate_sha256": _sha256(candidate_path),
        "rows": int(len(candidates)),
        "dual_predictions": str(dual_path),
        "dual_predictions_sha256": _sha256(dual_path),
        "source_state": str(source_path),
        "source_state_sha256": _sha256(source_path),
        "source_product_map": "upstream-source provenance only; never used as a Kraken Futures product_id",
        "known_product_rows": known_products,
        "unknown_product_rows": int(len(candidates) - known_products),
        "outcome_columns_consumed": [],
    }
    _write_json_once(out / "candidate_manifest.json", manifest)
    _write_json_once(out / "minute_download_manifest.json", {
        "schema": "strict_r3_p8u_exact_1m_kraken_symbol_download_request_v1",
        "target_free": True,
        "parent_candidate_sha256": _sha256(candidate_path),
        "candidate_sha256": _sha256(download_path),
        "rows": int(len(download_candidates)),
        "product_mapping": "canonical Kraken Futures client resolution from frozen symbol identity",
        "source_state_sha256": _sha256(source_path),
        "entry_delay_minutes": int(contract.entry_delay_minutes),
        "horizon_minutes": int(contract.horizon_minutes),
        "outcome_columns_consumed": [],
    })
    print(json.dumps({"out": str(out), "rows": len(candidates), "known_product_rows": known_products}, sort_keys=True))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-predictions", type=Path, default=DEFAULT_DUAL)
    parser.add_argument("--source-state", type=Path, default=DEFAULT_SOURCE_STATE)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--threshold-bps", type=float, default=50.0)
    parser.add_argument("--entry-delay-minutes", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    materialize(args)


if __name__ == "__main__":
    main()
