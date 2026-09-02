#!/usr/bin/env python3
"""Run the offline P8U Router-first staged scorer on completed source hours.

Inputs are an append-only primitive source state and a separate target-free
complete-universe candidate file.  The command writes no exchange, policy,
portfolio, or order state.  Each source hour is committed atomically only
after direct state, Router, Base, Under, and dual MC1 have all succeeded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_sealed_inference_stack import (  # noqa: E402
    P8USealedInferenceStack,
)
from extreme_price_movements.inference.p8u_staged_timestamp_executor import (  # noqa: E402
    P8UVectorStateSpec,
    P8UStagedTimestampExecutor,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--direct-bootstrap-root", type=Path, required=True)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--regular-vector-features",
        type=Path,
        action="append",
        default=[],
        help=(
            "One append-only, target-free regular feature-state snapshot. May "
            "be repeated once per source timestamp; Router/Base/Under consume "
            "those exact states instead of the 1,536-hour canonical vector graph."
        ),
    )
    parser.add_argument(
        "--regular-vector-receipt",
        type=Path,
        action="append",
        default=[],
        help=(
            "Hash-bound receipt for the corresponding --regular-vector-features "
            "argument. Required for every snapshot."
        ),
    )
    parser.add_argument(
        "--vector-bootstrap-state",
        type=Path,
        help=(
            "Sealed canonical vector transform state directory.  When set, "
            "the regular (non-direct) feature graph advances transactionally "
            "from this state using the exact bounded warm tail."
        ),
    )
    parser.add_argument("--vector-state-scope")
    parser.add_argument(
        "--vector-state-components",
        nargs="+",
        help="Declared canonical state components for the regular vector graph.",
    )
    parser.add_argument("--vector-tail-hours", type=int, default=1536)
    parser.add_argument(
        "--allow-unsealed-vector-bootstrap",
        action="store_true",
        help=(
            "Research parity only: accept a target-free bounded bootstrap receipt "
            "before it is sealed by a zero-mismatch score audit. Never use this "
            "flag for a promotion or execution bundle."
        ),
    )
    parser.add_argument(
        "--timestamps", nargs="+",
        help="Optional exact source timestamps to score from the candidate file, in chronological order.",
    )
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    if args.out_root.exists():
        raise FileExistsError(f"immutable staged score root already exists: {args.out_root}")
    for path in (args.bundle, args.source_state, args.candidates):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.direct_bootstrap_root.is_dir():
        raise NotADirectoryError(args.direct_bootstrap_root)
    vector_options = (
        args.vector_bootstrap_state,
        args.vector_state_scope,
        args.vector_state_components,
    )
    if any(value is not None for value in vector_options) and not all(
        value is not None for value in vector_options
    ):
        raise ValueError(
            "--vector-bootstrap-state, --vector-state-scope, and "
            "--vector-state-components must be supplied together"
        )
    regular_snapshots: dict[pd.Timestamp, tuple[pd.DataFrame, str, Path, Path, str]] = {}
    if bool(args.regular_vector_features) != bool(args.regular_vector_receipt):
        raise ValueError(
            "--regular-vector-features and --regular-vector-receipt must be supplied together"
        )
    if len(args.regular_vector_features) != len(args.regular_vector_receipt):
        raise ValueError("every regular feature-state snapshot needs exactly one matching receipt")
    if args.regular_vector_features:
        if any(value is not None for value in vector_options):
            raise ValueError(
                "regular feature-state snapshot mode cannot be combined with "
                "the bounded canonical vector-state graph"
            )
        for feature_path, receipt_path in zip(
            args.regular_vector_features, args.regular_vector_receipt, strict=True
        ):
            if not feature_path.is_file() or not receipt_path.is_file():
                raise FileNotFoundError("regular vector feature state or receipt")
            feature_sha256 = _sha256(feature_path)
            receipt_payload = json.loads(receipt_path.read_text())
            if receipt_payload.get("outcome_columns_consumed") not in (None, []):
                raise ValueError("regular vector feature receipt consumed outcomes")
            receipt_feature_hash = str(
                receipt_payload.get("features_sha256")
                or receipt_payload.get("feature_sha256")
                or ""
            )
            if receipt_feature_hash != feature_sha256:
                raise ValueError("regular vector feature receipt hash mismatch")
            snapshot = pd.read_parquet(feature_path)
            if "__ts__" not in snapshot.columns:
                raise KeyError("regular vector feature snapshot lacks __ts__")
            snapshot_ts = pd.to_datetime(snapshot["__ts__"], utc=True, errors="raise")
            if snapshot_ts.nunique() != 1:
                raise ValueError("each regular vector feature snapshot must contain exactly one source hour")
            signal = snapshot_ts.iloc[0]
            receipt_signal = receipt_payload.get("signal_ts")
            if receipt_signal is not None:
                parsed_receipt_signal = pd.Timestamp(receipt_signal)
                parsed_receipt_signal = (
                    parsed_receipt_signal.tz_localize("UTC")
                    if parsed_receipt_signal.tzinfo is None
                    else parsed_receipt_signal.tz_convert("UTC")
                )
                if parsed_receipt_signal != signal:
                    raise ValueError("regular vector receipt signal timestamp mismatch")
            if signal in regular_snapshots:
                raise ValueError("duplicate regular vector feature-state timestamp")
            regular_snapshots[signal] = (
                snapshot,
                feature_sha256,
                feature_path,
                receipt_path,
                _sha256(receipt_path),
            )
    vector_state = None
    if args.vector_bootstrap_state is not None:
        if not args.vector_bootstrap_state.is_dir():
            raise NotADirectoryError(args.vector_bootstrap_state)
        vector_state = P8UVectorStateSpec(
            bootstrap_state_root=args.vector_bootstrap_state,
            state_scope=str(args.vector_state_scope),
            state_components=tuple(map(str, args.vector_state_components)),
            tail_hours=int(args.vector_tail_hours),
            allow_unsealed_bootstrap=bool(args.allow_unsealed_vector_bootstrap),
        )
    source = joblib.load(args.source_state)
    panel = source.get("panel") if isinstance(source, Mapping) else None
    symbols = tuple(map(str, source.get("symbols") or ())) if isinstance(source, Mapping) else ()
    if not isinstance(panel, Mapping) or len(symbols) != 160:
        raise ValueError("staged score requires a frozen 160-symbol source state")
    candidates = pd.read_parquet(args.candidates)
    required = {"candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"}
    if missing := sorted(required.difference(candidates.columns)):
        raise ValueError(f"target-free candidate file misses {missing}")
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True, errors="raise")
    candidates["__decision_ts__"] = pd.to_datetime(candidates["__decision_ts__"], utc=True, errors="raise")
    timestamps = tuple(sorted(pd.DatetimeIndex(candidates["__ts__"].unique())))
    if args.timestamps:
        selected = tuple(pd.Timestamp(value) for value in args.timestamps)
        selected = tuple(
            stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
            for stamp in selected
        )
        if len(selected) != len(set(selected)):
            raise ValueError("--timestamps contains a duplicate source timestamp")
        missing = sorted(set(selected).difference(timestamps))
        if missing:
            raise ValueError(f"candidate file has no requested source timestamp(s): {missing[:3]}")
        timestamps = tuple(sorted(selected))
    if not timestamps:
        raise ValueError("candidate file has no source timestamps")
    stack = P8USealedInferenceStack.load(args.bundle, root=ROOT)
    runner = P8UStagedTimestampExecutor(
        root=args.out_root,
        direct_bootstrap_root=args.direct_bootstrap_root,
        symbols=symbols,
        stack=stack,
        market_basket=symbols,
        vector_state=vector_state,
    )
    started = time.monotonic()
    checkpoints: list[dict[str, object]] = []
    for stamp in timestamps:
        block = candidates.loc[candidates["__ts__"].eq(stamp)].copy()
        snapshot_block = None
        snapshot_sha256 = None
        if regular_snapshots:
            item = regular_snapshots.get(stamp)
            if item is None:
                raise ValueError(f"regular feature-state snapshot missing for {stamp.isoformat()}")
            snapshot_block, snapshot_sha256, _feature_path, _receipt_path, _receipt_sha256 = item
        result = runner.advance(
            source_timestamp=stamp,
            candidates=block,
            panel=panel,
            regular_vector_snapshot=snapshot_block,
            regular_vector_snapshot_sha256=snapshot_sha256,
        )
        checkpoints.append({
            "source_timestamp": stamp.isoformat(),
            "decision_timestamp": result.decision_timestamp.isoformat(),
            "candidate_rows": len(block),
            "router50_rows": len(result.routed_features),
            "admitted_rows": len(result.scores.admitted),
            "commit": str(result.commit.relative_to(args.out_root)),
        })
    receipt = {
        "schema": "strict_r3_p8u_staged_timestamp_score_run_v1",
        "status": "pass_target_free_staged_scoring",
        "bundle": str(args.bundle.resolve()),
        "bundle_sha256": _sha256(args.bundle),
        "source_state": str(args.source_state.resolve()),
        "source_state_sha256": _sha256(args.source_state),
        "candidate_file": str(args.candidates.resolve()),
        "candidate_file_sha256": _sha256(args.candidates),
        "direct_bootstrap_root": str(args.direct_bootstrap_root.resolve()),
        "regular_vector_state": (
            None
            if vector_state is None
            else {
                "bootstrap_state": str(vector_state.bootstrap_state_root.resolve()),
                "state_scope": vector_state.state_scope,
                "state_components": list(vector_state.state_components),
                "tail_hours": vector_state.tail_hours,
                "allow_unsealed_bootstrap": vector_state.allow_unsealed_bootstrap,
            }
        ),
        "regular_vector_execution": (
            None
            if not regular_snapshots
            else {
                "mode": "append_only_feature_state_snapshot",
                "snapshots": [
                    {
                        "signal_ts": stamp.isoformat(),
                        "features": str(feature_path.resolve()),
                        "features_sha256": feature_sha256,
                        "receipt": str(receipt_path.resolve()),
                        "receipt_sha256": receipt_sha256,
                    }
                    for stamp, (_snapshot, feature_sha256, feature_path, receipt_path, receipt_sha256)
                    in sorted(regular_snapshots.items())
                ],
                "batch_feature_graph_called": False,
            }
        ),
        "timestamps": checkpoints,
        "runtime_seconds": time.monotonic() - started,
        "outcome_columns_consumed": [],
        "policy_or_portfolio_called": False,
        "exchange_or_order_submission_called": False,
    }
    _atomic_json(args.out_root / "run_manifest.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
