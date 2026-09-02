#!/usr/bin/env python3
"""Create an atomic target-free bootstrap commit for the P8U direct executor.

The resulting root can advance exactly one new hourly snapshot without
replaying raw history. It remains intentionally unscorable until all 175
sealed feature nodes have independent parity receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from extreme_price_movements.inference.p8u_single_timestamp_graph import (
    P8UOneTimestampExecutor,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-state", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--feature-plan",
        default=(
            "data_perp/artifacts/strict_r3_p8u_preproduction_bundle_20260828_v8/"
            "audit/required_feature_plan.json"
        ),
        help="sealed P8U feature plan used for coverage accounting",
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    source_path = Path(args.source_state).resolve()
    out_root = Path(args.out_root).resolve()
    feature_plan_path = Path(args.feature_plan).resolve()
    if ROOT not in source_path.parents or ROOT not in out_root.parents:
        raise ValueError("P8U paths must remain below repository root")
    if ROOT not in feature_plan_path.parents:
        raise ValueError("P8U feature plan must remain below repository root")
    if out_root.exists():
        raise FileExistsError(f"immutable direct executor root exists: {out_root}")
    payload = joblib.load(source_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("panel"), dict):
        raise ValueError("source state lacks target-free panel")
    feature_plan = json.loads(feature_plan_path.read_text())
    sealed_features = {
        str(value) for value in feature_plan.get("full_union", ()) if str(value)
    }
    if len(sealed_features) != 175:
        raise ValueError("sealed P8U feature plan must contain exactly 175 fields")
    symbols = tuple(map(str, payload.get("symbols") or payload.get("universe_symbols") or ()))
    close = payload["panel"].get("close")
    if not isinstance(close, pd.DataFrame) or len(symbols) != 160:
        raise ValueError("source state lacks sealed 160-symbol panel")
    started = time.perf_counter()
    executor = P8UOneTimestampExecutor(root=out_root, symbols=symbols, market_basket=symbols)
    output = executor.bootstrap(payload["panel"])
    runtime = time.perf_counter() - started
    ledger = json.loads(executor.ledger_path.read_text())
    receipt = {
        "schema": "strict_r3_p8u_one_timestamp_executor_bootstrap_v1",
        "status": "pass_target_free_partial_direct_executor",
        "source_state": str(source_path),
        "source_state_sha256": _sha256(source_path),
        "source_rows": int(len(close)),
        "source_start": pd.Timestamp(close.index[0]).isoformat(),
        "source_end": pd.Timestamp(close.index[-1]).isoformat(),
        "symbols": len(symbols),
        "executor_contract_hash": executor.contract_hash,
        "ledger": ledger,
        "available_direct_features": sorted(output),
        "feature_plan": str(feature_plan_path),
        "sealed_feature_count": len(sealed_features),
        "available_sealed_features": sorted(set(output).intersection(sealed_features)),
        "missing_sealed_feature_count": len(sealed_features.difference(output)),
        "outcome_columns_consumed": [],
        "model_features_emitted": False,
        "runtime_seconds": runtime,
    }
    (out_root / "bootstrap_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
