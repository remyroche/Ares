#!/usr/bin/env python3
"""Prepare one immutable, target-free full-vs-warm P8U parity probe.

The source state available for the August P8U causal panel has 170 symbols,
while the historical full-causal reference was produced for its contemporaneous
160-symbol frozen universe.  This script makes a *new, reduced offline copy*
of the primitive source state for exactly that reference universe.  It never
mutates the source checkpoint and deliberately includes no outcome columns.

The result is suitable for bootstrap_strict_r3_p8u_warm_feature_state.py:
the same feature graph can bootstrap its transform states and then be compared
against the full causal panel on one held timestamp.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_warm_feature_state import (  # noqa: E402
    P8UWarmFeatureConfig,
    atomic_json,
    sha256_file,
)


def _utc(value: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--full-reference", type=Path, required=True)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--signal-ts", required=True)
    parser.add_argument(
        "--max-symbols",
        type=int,
        help=(
            "Optional deterministic bounded symbol subset for a resource-limited "
            "all-field parity probe. It does not change the feature plan."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.full_reference = args.full_reference.resolve()
    args.source_panel = args.source_panel.resolve()
    args.out_dir = args.out_dir.resolve()
    config = P8UWarmFeatureConfig.load(args.config, root=ROOT)
    signal_ts = _utc(args.signal_ts)
    if args.out_dir.exists():
        raise FileExistsError(f"immutable P8U parity probe exists: {args.out_dir}")
    if not args.full_reference.is_file() or not args.source_panel.is_file():
        raise FileNotFoundError("parity probe source is unavailable")

    reference_columns = [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
        *config.feature_plan,
    ]
    # Keep the timestamp comparison in pandas: Arrow's equality kernel refuses
    # to compare a UTC nanosecond column with the platform's timestamp-second
    # scalar.  Projection still limits this read to identities plus the sealed
    # 175-field union, never outcomes or the wider 1,412-column panel.
    reference = pd.read_parquet(args.full_reference, columns=reference_columns)
    reference["__ts__"] = pd.to_datetime(reference["__ts__"], utc=True)
    reference["__decision_ts__"] = pd.to_datetime(reference["__decision_ts__"], utc=True)
    reference = reference.loc[reference["__ts__"].eq(signal_ts)].copy()
    if reference.empty:
        raise ValueError(f"full reference has no candidates at {signal_ts.isoformat()}")
    if reference["side_name"].astype(str).str.lower().ne("long").any():
        raise ValueError("full reference is not long-only")
    if reference["candidate_id"].duplicated().any():
        raise ValueError("full reference has duplicate candidate identities")
    if args.max_symbols is not None:
        if args.max_symbols < 2:
            raise ValueError("--max-symbols must be at least two")
        selected = sorted(reference["__symbol__"].astype(str).unique())[:args.max_symbols]
        reference = reference.loc[reference["__symbol__"].astype(str).isin(selected)].copy()
        if len(reference) != len(selected):
            raise AssertionError("bounded parity selection lost a symbol")
    symbols = tuple(sorted(reference["__symbol__"].astype(str).unique()))

    state = joblib.load(args.source_panel)
    panel = state.get("panel")
    source_symbols = tuple(map(str, state.get("symbols", [])))
    if not isinstance(panel, dict) or not symbols or not set(symbols).issubset(source_symbols):
        raise ValueError("source panel cannot supply the full reference universe")
    copied_panel: dict[str, object] = {}
    for key, value in panel.items():
        if isinstance(value, pd.DataFrame):
            missing = sorted(set(symbols).difference(value.columns.astype(str)))
            if missing:
                raise ValueError(f"source panel {key} misses reference symbols: {missing[:3]}")
            copied_panel[key] = value.loc[:, list(symbols)].copy()
        else:
            copied_panel[key] = value
    reduced = dict(state)
    reduced["symbols"] = list(symbols)
    reduced["panel"] = copied_panel

    args.out_dir.mkdir(parents=True)
    candidates = reference.loc[:, [
        "candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name",
    ]].sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    candidates.to_parquet(args.out_dir / "candidates.parquet", index=False, compression="zstd")
    reference.to_parquet(args.out_dir / "full_causal_reference_features.parquet", index=False, compression="zstd")
    joblib.dump(reduced, args.out_dir / "reduced_feature_panel_state.joblib", compress=3)
    request = {
        "schema": "strict_r3_p8u_warm_feature_request_v1",
        "signal_ts": signal_ts.isoformat(),
        "candidates": str((args.out_dir / "candidates.parquet").relative_to(ROOT)),
        "candidates_sha256": sha256_file(args.out_dir / "candidates.parquet"),
        "panel_state": str((args.out_dir / "reduced_feature_panel_state.joblib").relative_to(ROOT)),
        "panel_state_sha256": sha256_file(args.out_dir / "reduced_feature_panel_state.joblib"),
        "reference_features": str((args.out_dir / "full_causal_reference_features.parquet").relative_to(ROOT)),
        "reference_features_sha256": sha256_file(args.out_dir / "full_causal_reference_features.parquet"),
        "outcome_columns_consumed": [],
    }
    atomic_json(args.out_dir / "request.json", request)
    receipt = {
        "schema": "strict_r3_p8u_warm_feature_parity_probe_input_v1",
        "status": "prepared",
        "config": str(config.path),
        "config_sha256": sha256_file(config.path),
        "feature_union_sha256": config.feature_union_sha256,
        "signal_ts": signal_ts.isoformat(),
        "candidate_rows": int(len(candidates)),
        "symbols": int(len(symbols)),
        "bounded_symbol_probe": args.max_symbols is not None,
        "full_reference": str(args.full_reference),
        "full_reference_sha256": sha256_file(args.full_reference),
        "source_panel": str(args.source_panel),
        "source_panel_sha256": sha256_file(args.source_panel),
        "outcome_columns_consumed": [],
    }
    atomic_json(args.out_dir / "probe_input_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
