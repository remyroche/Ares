#!/usr/bin/env python3
"""Produce one immutable, target-free successor decision and auction proposal.

The runner reads already-materialised features and C1 snapshots only.  It has
no downloader, exchange, account, policy-label, or order-submission import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_successor_inference_contract import P8USuccessorInferenceContract, sha256_file
from extreme_price_movements.inference.p8u_successor_portfolio_adapter import P8USuccessorPortfolioState, auction


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _write_json_once(path: Path, payload: dict[str, object]) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _verify_dynamic_inputs(
    *,
    contract: P8USuccessorInferenceContract,
    features: Path,
    feature_commit: Path | None,
    c1_snapshots: Path,
    c1_snapshot_receipt: Path | None,
    decision: pd.Timestamp,
) -> None:
    """Verify the mutable per-decision inputs of the schema-v2 contract.

    The contract pins immutable code, model, source-map, and calibration-state
    manifests.  These receipts bind one generated feature vector and one C1
    snapshot vector to the exact decision without pretending either can be a
    static monthly input.
    """
    if contract.payload.get("schema") != "p8u_c0_c1_successor_no_order_inference_contract_v2":
        return
    if feature_commit is None or c1_snapshot_receipt is None:
        raise ValueError("schema-v2 no-order decision requires feature and C1 snapshot receipts")
    if not feature_commit.is_file() or not c1_snapshot_receipt.is_file():
        raise FileNotFoundError("schema-v2 feature or C1 snapshot receipt is missing")
    feature_payload = json.loads(feature_commit.read_text(encoding="utf-8"))
    if str(feature_payload.get("features")) != str(features.resolve()):
        raise ValueError("feature receipt does not bind the supplied feature panel")
    if str(feature_payload.get("features_sha256")) != sha256_file(features):
        raise ValueError("feature receipt hash mismatch")
    parity = dict(feature_payload.get("parity") or {})
    if parity.get("status") != "pass" or int(parity.get("rows_outside_tolerance", -1)) != 0:
        raise ValueError("feature receipt is not an exact-parity commit")

    c1_payload = json.loads(c1_snapshot_receipt.read_text(encoding="utf-8"))
    if c1_payload.get("status") != "PASS_TARGET_FREE_C1_RUNTIME_SNAPSHOT":
        raise ValueError("C1 snapshot receipt is incomplete")
    if _utc(c1_payload.get("decision_ts")) != decision:
        raise ValueError("C1 snapshot receipt decision differs from current decision")
    output = dict(c1_payload.get("output") or {})
    if str(output.get("entry_sr_oof_features.parquet")) != sha256_file(c1_snapshots):
        raise ValueError("C1 snapshot receipt hash mismatch")
    if int(c1_payload.get("candidate_rows", -1)) <= 0:
        raise ValueError("C1 snapshot receipt has no target-free candidates")
    artifacts = contract.payload["artifacts"]
    if str(c1_payload.get("c1_bundle_manifest_sha256")) != str(artifacts["c1_lva_bundle_manifest"]["sha256"]):
        raise ValueError("C1 snapshot receipt uses another frozen C1-LVA bundle")
    if str(c1_payload.get("source_map_sha256")) != str(artifacts["c1_source_map_manifest"]["sha256"]):
        raise ValueError("C1 snapshot receipt uses another frozen C1 source map")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--feature-commit", type=Path)
    parser.add_argument("--c1-snapshots", type=Path, required=True)
    parser.add_argument("--c1-snapshot-receipt", type=Path)
    parser.add_argument("--decision-ts", required=True)
    parser.add_argument("--wallet-equity-quote", type=float, required=True)
    parser.add_argument("--open-symbol", action="append", default=[])
    parser.add_argument("--pending-symbol", action="append", default=[])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.out_dir.resolve()
    if output.exists():
        raise FileExistsError(f"immutable no-order decision exists: {output}")
    decision = _utc(args.decision_ts)
    contract = P8USuccessorInferenceContract.load(args.contract, root=ROOT)
    _verify_dynamic_inputs(
        contract=contract,
        features=args.features.resolve(), feature_commit=args.feature_commit.resolve() if args.feature_commit else None,
        c1_snapshots=args.c1_snapshots.resolve(),
        c1_snapshot_receipt=args.c1_snapshot_receipt.resolve() if args.c1_snapshot_receipt else None,
        decision=decision,
    )
    features = pd.read_parquet(args.features)
    features["__decision_ts__"] = pd.to_datetime(features["__decision_ts__"], utc=True, errors="raise")
    features = features.loc[features["__decision_ts__"].eq(decision)].copy()
    if features.empty:
        raise ValueError("no target-free features for requested decision")
    c1_snapshots = pd.read_parquet(args.c1_snapshots)
    snapshot_time = "__decision_ts__" if "__decision_ts__" in c1_snapshots.columns else "snapshot_ts"
    c1_snapshots[snapshot_time] = pd.to_datetime(c1_snapshots[snapshot_time], utc=True, errors="raise")
    c1_snapshots = c1_snapshots.loc[c1_snapshots[snapshot_time].eq(decision)].copy()
    if len(c1_snapshots) != len(features):
        raise ValueError("C1 snapshot rows do not cover the complete target-free feature universe")
    if set(c1_snapshots["candidate_id"].astype(str)) != set(features["candidate_id"].astype(str)):
        raise ValueError("C1 snapshot identities differ from the target-free feature universe")
    scores = contract.build_stack().score(
        full_population=features, c1_snapshots=c1_snapshots
    )
    selected = scores.mapper.selected.loc[scores.mapper.selected["__decision_ts__"].eq(decision)].copy()
    portfolio_state = P8USuccessorPortfolioState(
        wallet_equity_quote=float(args.wallet_equity_quote),
        open_symbols=tuple(map(str, args.open_symbol)),
        pending_symbols=tuple(map(str, args.pending_symbol)),
    )
    proposed = auction(selected_scores=selected, state=portfolio_state) if not selected.empty else selected.copy()
    output.mkdir(parents=True, exist_ok=False)
    scores.router.to_parquet(output / "router_scores.parquet", index=False, compression="zstd")
    scores.coordinates.to_parquet(output / "routed_coordinates.parquet", index=False, compression="zstd")
    scores.mapper.c0.to_parquet(output / "c0_scores.parquet", index=False, compression="zstd")
    scores.mapper.c1.to_parquet(output / "c1_scores.parquet", index=False, compression="zstd")
    selected.to_parquet(output / "selected_scores.parquet", index=False, compression="zstd")
    proposed.to_parquet(output / "auction.parquet", index=False, compression="zstd")
    receipt = {
        "schema": "p8u_c0_c1_successor_no_order_decision_v1",
        "status": "complete_target_free_no_order",
        "order_submission": False,
        "network_or_exchange_calls": 0,
        "decision_timestamp": decision.isoformat(),
        "contract": {"path": str(args.contract.resolve()), "sha256": contract.sha256},
        "features": {"path": str(args.features.resolve()), "sha256": sha256_file(args.features)},
        "feature_commit": (
            {"path": str(args.feature_commit.resolve()), "sha256": sha256_file(args.feature_commit)}
            if args.feature_commit else None
        ),
        "c1_snapshots": {"path": str(args.c1_snapshots.resolve()), "sha256": sha256_file(args.c1_snapshots)},
        "c1_snapshot_receipt": (
            {"path": str(args.c1_snapshot_receipt.resolve()), "sha256": sha256_file(args.c1_snapshot_receipt)}
            if args.c1_snapshot_receipt else None
        ),
        "full_universe_rows": int(len(features)),
        "router50_rows": int(len(scores.coordinates)),
        "c0_rows": int(len(scores.mapper.c0)),
        "c1_rows": int(len(scores.mapper.c1)),
        "selected_rows": int(len(selected)),
        "auction_proposals": int((proposed.get("execution_action", pd.Series(dtype=str)) == "propose_no_order").sum()),
        "outputs": {
            name: sha256_file(output / name)
            for name in ("router_scores.parquet", "routed_coordinates.parquet", "c0_scores.parquet", "c1_scores.parquet", "selected_scores.parquet", "auction.parquet")
        },
    }
    _write_json_once(output / "receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
