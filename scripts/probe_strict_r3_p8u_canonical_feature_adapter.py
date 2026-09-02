#!/usr/bin/env python3
"""Run an offline exact-canonical P8U feature parity probe.

The raw append-only source panel and full historical feature reference are
read-only.  The feature graph always receives the full reference universe;
``--max-symbols`` only limits the final comparison rows, never the
cross-sectional construction universe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_canonical_feature_adapter import canonical_features_from_saved_panel
from extreme_price_movements.inference.p8u_warm_feature_state import atomic_json, sha256_file


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _project(
    generated: dict[str, pd.DataFrame], reference: pd.DataFrame, fields: list[str], stamp: pd.Timestamp,
) -> pd.DataFrame:
    index = pd.MultiIndex.from_frame(reference[["__ts__", "__symbol__"]])
    unique_ts = pd.DatetimeIndex([stamp])
    values: dict[str, np.ndarray] = {}
    for field in fields:
        frame = generated[field].reindex(index=unique_ts, columns=reference["__symbol__"].unique())
        values[field] = frame.stack(dropna=False).reindex(index).to_numpy(dtype=np.float32)
    return pd.concat([reference.reset_index(drop=True), pd.DataFrame(values)], axis=1, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-plan", type=Path, required=True)
    parser.add_argument(
        "--canonical-manifest", type=Path, required=True,
        help="Canonical full-universe feature manifest used to build the reference.",
    )
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--full-reference", type=Path, required=True)
    parser.add_argument(
        "--source-anchored-reference-manifest",
        type=Path,
        help=(
            "Immutable source-anchor manifest produced by "
            "seal_strict_r3_p8u_source_anchored_reference.py. When supplied, "
            "the named reference checkpoint, source panel, plan and universe "
            "must all match before an exact parity claim is made."
        ),
    )
    parser.add_argument(
        "--require-source-anchored-reference",
        action="store_true",
        help=(
            "Fail closed unless --source-anchored-reference-manifest proves "
            "the supplied historical reference was produced from this exact "
            "immutable source panel."
        ),
    )
    parser.add_argument("--signal-ts", required=True)
    parser.add_argument("--max-symbols", type=int, default=20)
    parser.add_argument(
        "--selected-feature-only", action="store_true",
        help="Use the sealed feature/dependency projection instead of the canonical full-config union.",
    )
    parser.add_argument(
        "--history-tail-hours", type=int,
        help="Use only an append-only retained raw-history tail for an exactness probe.",
    )
    parser.add_argument(
        "--state-dir", type=Path,
        help=(
            "Private state namespace for an exact canonical-state continuity probe. "
            "Never point this at an active inference state."
        ),
    )
    parser.add_argument(
        "--state-scope",
        help="Required semantic identity when --state-dir is supplied.",
    )
    parser.add_argument(
        "--state-bootstrap-end-exclusive",
        help=(
            "Seed the private state with raw rows strictly before this timestamp, "
            "then score the requested timestamp with the same state."
        ),
    )
    parser.add_argument(
        "--exact-causal-transform-state-seed",
        action="store_true",
        help=(
            "Build the per-feature causal-transform state from the complete "
            "causal bootstrap history. Required when sealing an exact "
            "appendable state checkpoint; never enables outcomes."
        ),
    )
    parser.add_argument(
        "--reuse-state", action="store_true",
        help="Use an existing private state only for the scored append step.",
    )
    parser.add_argument(
        "--state-components",
        help=(
            "Comma-separated private-state components for an offline diagnostic. "
            "Default is the complete canonical state contract; this flag is never "
            "accepted by a sealed inference bundle."
        ),
    )
    parser.add_argument(
        "--debug-fields",
        help=(
            "Comma-separated features to snapshot at canonical transform "
            "boundaries. Diagnostic only; never accepted by a sealed bundle."
        ),
    )
    parser.add_argument(
        "--debug-full-history",
        action="store_true",
        help="Persist full history for --debug-fields; diagnostic only.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable probe already exists: {args.out_dir}")
    if args.max_symbols < 1:
        raise ValueError("--max-symbols must be positive")
    plan = json.loads(args.feature_plan.read_text())
    fields = [str(value) for value in plan["full_union"]]
    canonical_manifest = json.loads(args.canonical_manifest.read_text())
    raw_universe = canonical_manifest.get("symbols")
    if not isinstance(raw_universe, list) or not raw_universe:
        raise ValueError("canonical manifest lacks non-empty symbols universe")
    universe = tuple(sorted(dict.fromkeys(map(str, raw_universe))))
    stamp = _utc(args.signal_ts)
    source_anchor_manifest: dict[str, object] | None = None
    source_anchor_sha: str | None = None
    if args.source_anchored_reference_manifest is not None:
        anchor_path = args.source_anchored_reference_manifest
        if not anchor_path.is_file():
            raise FileNotFoundError(anchor_path)
        source_anchor_manifest = json.loads(anchor_path.read_text())
        if source_anchor_manifest.get("schema") != "strict_r3_p8u_source_anchored_reference_v1":
            raise ValueError("unrecognised source-anchored reference manifest")
        if source_anchor_manifest.get("legacy_historical_reference_reconciled") is not False:
            raise ValueError("source anchor must explicitly preserve legacy reconciliation status")
        source_anchor_sha = sha256_file(anchor_path)
        expected_hashes = {
            "source_panel": sha256_file(args.source_panel),
            "feature_plan": sha256_file(args.feature_plan),
            "canonical_manifest": sha256_file(args.canonical_manifest),
        }
        for key, expected_hash in expected_hashes.items():
            observed = (source_anchor_manifest.get(key) or {}).get("sha256")
            if observed != expected_hash:
                raise ValueError(
                    f"source anchor {key} hash mismatch: {observed!r} != {expected_hash!r}"
                )
        matching = [
            row for row in source_anchor_manifest.get("checkpoints", [])
            if str(row.get("signal_ts")) == stamp.isoformat()
        ]
        if len(matching) != 1:
            raise ValueError("source anchor lacks exactly one checkpoint for signal timestamp")
        anchor_reference = (anchor_path.parent / str(matching[0].get("path"))).resolve()
        if anchor_reference != args.full_reference.resolve():
            raise ValueError("full reference is not the source-anchor checkpoint for this timestamp")
        if str(matching[0].get("sha256")) != sha256_file(args.full_reference):
            raise ValueError("source-anchor checkpoint hash mismatch")
    elif args.require_source_anchored_reference:
        raise ValueError("exact parity requires --source-anchored-reference-manifest")
    identity = ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"]
    reference = pd.read_parquet(args.full_reference, columns=[*identity, *fields])
    reference["__ts__"] = pd.to_datetime(reference["__ts__"], utc=True)
    reference["__decision_ts__"] = pd.to_datetime(reference["__decision_ts__"], utc=True)
    full = reference.loc[reference["__ts__"].eq(stamp)].copy()
    if full.empty or full["candidate_id"].duplicated().any():
        raise ValueError("reference identity panel is empty or duplicated")
    candidate_symbols = set(full["__symbol__"].astype(str))
    if not candidate_symbols.issubset(set(universe)):
        raise ValueError("reference candidate symbols are not contained in canonical universe")
    comparison_symbols = sorted(candidate_symbols)[:args.max_symbols]
    comparison = full.loc[full["__symbol__"].astype(str).isin(comparison_symbols)].copy()
    state = joblib.load(args.source_panel)
    panel = state.get("panel")
    if not isinstance(panel, dict):
        raise ValueError("source panel has no primitive panel")
    if (args.state_dir is None) != (args.state_scope is None):
        raise ValueError("--state-dir and --state-scope must be supplied together")
    bootstrap_end = (
        _utc(args.state_bootstrap_end_exclusive)
        if args.state_bootstrap_end_exclusive is not None
        else None
    )
    if args.reuse_state and bootstrap_end is not None:
        raise ValueError("--reuse-state cannot be combined with a state bootstrap")
    if args.reuse_state and args.state_dir is None:
        raise ValueError("--reuse-state requires --state-dir")
    state_components = (
        tuple(part.strip() for part in str(args.state_components).split(",") if part.strip())
        if args.state_components
        else None
    )
    debug_fields = (
        tuple(part.strip() for part in str(args.debug_fields).split(",") if part.strip())
        if args.debug_fields
        else None
    )
    debug_dir = args.out_dir / "state_debug" if debug_fields else None
    if bootstrap_end is not None:
        if args.state_dir is None:
            raise ValueError("--state-bootstrap-end-exclusive requires --state-dir")
        if args.state_dir.exists():
            raise FileExistsError("stateful parity probe requires a new private state directory")
        seed_panel = {
            key: value.loc[value.index < bootstrap_end].copy(deep=False)
            if isinstance(value, pd.DataFrame) else value
            for key, value in panel.items()
        }
        canonical_features_from_saved_panel(
            seed_panel,
            universe_symbols=universe,
            requested_features=fields,
            full_config_causal_universe=not args.selected_feature_only,
            state_dir=args.state_dir,
            state_scope=args.state_scope,
            state_components=state_components,
            exact_causal_transform_state_seed=bool(args.exact_causal_transform_state_seed),
            debug_snapshot_dir=debug_dir,
            debug_snapshot_fields=debug_fields,
            debug_snapshot_full_history=bool(args.debug_full_history),
        )
    elif args.state_dir is not None and args.state_dir.exists() and not args.reuse_state:
        raise FileExistsError("stateful parity probe state exists; pass --reuse-state explicitly")
    elif args.reuse_state and not args.state_dir.exists():
        raise FileNotFoundError("--reuse-state needs an existing private state directory")
    if args.history_tail_hours is not None:
        if args.history_tail_hours < 2:
            raise ValueError("--history-tail-hours must be at least two")
        panel = {
            key: value.tail(args.history_tail_hours).copy(deep=False)
            if isinstance(value, pd.DataFrame) else value
            for key, value in panel.items()
        }
    # A private stateful probe is allowed to generate only through the
    # asserted checkpoint.  Passing a source panel which happens to contain
    # later rows would otherwise update persistent operator state beyond the
    # named state timestamp even though projection compares only ``stamp``.
    # That would make the next append step read future state.  Keep the full
    # universe but cut every time-indexed primitive at the checkpoint first.
    if args.state_dir is not None:
        panel = {
            key: value.loc[value.index <= stamp].copy(deep=False)
            if isinstance(value, pd.DataFrame) else value
            for key, value in panel.items()
        }
    started = time.monotonic()
    generated = canonical_features_from_saved_panel(
        panel,
        universe_symbols=universe,
        requested_features=fields,
        full_config_causal_universe=not args.selected_feature_only,
        state_dir=args.state_dir,
        state_scope=args.state_scope,
        state_components=state_components,
        debug_snapshot_dir=debug_dir,
        debug_snapshot_fields=debug_fields,
        debug_snapshot_full_history=bool(args.debug_full_history),
    )
    elapsed = time.monotonic() - started
    actual = _project(generated, comparison[identity], fields, stamp)
    actual = actual.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    expected = comparison[[*identity, *fields]].sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if not actual[identity].equals(expected[identity]):
        raise AssertionError("canonical projection changed target-free identity")
    rows = []
    for field in fields:
        a = pd.to_numeric(actual[field], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(expected[field], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(a) & np.isfinite(b)
        equal_nan = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=1e-6, atol=1e-6, equal_nan=True)
        rows.append({
            "feature": field,
            "rows": len(a),
            "matching_rows": int(close.sum()),
            "mismatch_rows": int((~close).sum()),
            "finite_mismatch_rows": int((finite & ~close).sum()),
            "missing_mismatch_rows": int((~finite & ~equal_nan).sum()),
            "max_abs_delta": float(np.nanmax(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0,
        })
    audit = pd.DataFrame(rows)
    # A diagnostics-only snapshot may have already created ``out_dir`` during
    # graph evaluation.  The immutable-run check above still rejects a
    # pre-existing caller target; accepting this directory here only lets the
    # probe finish its own receipt.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    actual.to_parquet(args.out_dir / "canonical_adapter_features.parquet", index=False, compression="zstd")
    audit.to_parquet(args.out_dir / "feature_parity_by_field.parquet", index=False, compression="zstd")
    status = "pass" if int(audit.mismatch_rows.sum()) == 0 else "fail"
    receipt = {
        "schema": "strict_r3_p8u_canonical_adapter_parity_v1",
        "status": status,
        "signal_ts": stamp.isoformat(),
        "full_cross_section_universe_symbols": len(universe),
        "comparison_symbols": len(comparison_symbols),
        "comparison_rows": len(comparison),
        "fields": len(fields),
        "feature_graph_mode": "selected_projection" if args.selected_feature_only else "full_config_causal_universe",
        "history_tail_hours": args.history_tail_hours,
        "mismatch_cells": int(audit.mismatch_rows.sum()),
        "failing_fields": audit.loc[audit.mismatch_rows.gt(0), "feature"].tolist(),
        "max_abs_delta": float(audit.max_abs_delta.max()),
        "runtime_seconds": elapsed,
        "source_panel_sha256": sha256_file(args.source_panel),
        "reference_sha256": sha256_file(args.full_reference),
        "feature_plan_sha256": sha256_file(args.feature_plan),
        "canonical_manifest_sha256": sha256_file(args.canonical_manifest),
        "reference_contract": (
            "source_anchored_current_code_checkpoint_v1"
            if source_anchor_manifest is not None
            else "legacy_unanchored_reference"
        ),
        "source_anchored_reference_manifest_sha256": source_anchor_sha,
        "exact_reference_provenance": bool(source_anchor_manifest is not None),
        "outcome_columns_consumed": [],
        "state_bundle_published": False,
        "stateful_canonical_graph": args.state_dir is not None,
        "state_scope": args.state_scope,
        "state_components": list(state_components) if state_components else None,
        "state_bootstrap_end_exclusive": (
            bootstrap_end.isoformat() if bootstrap_end is not None else None
        ),
        "exact_causal_transform_state_seed": bool(args.exact_causal_transform_state_seed),
    }
    atomic_json(args.out_dir / "parity_summary.json", receipt)
    if args.state_dir is not None:
        # The state sealer consumes this independent checkpoint assertion.
        # Together with the source-panel and feature-plan hashes in the parity
        # receipt it prevents a state directory whose operators were advanced
        # past its advertised timestamp from being promoted as a bootstrap.
        atomic_json(args.state_dir / "canonical_state_checkpoint.json", {
            "schema": "strict_r3_p8u_canonical_state_checkpoint_v1",
            "state_scope": args.state_scope,
            "as_of_timestamp": stamp.isoformat(),
            "source_panel_sha256": sha256_file(args.source_panel),
            "feature_plan_sha256": sha256_file(args.feature_plan),
            "stateful_tail_hours": args.history_tail_hours,
            "outcome_columns_consumed": [],
        })
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Stateful parity is deliberately fail-closed.  Leave a small
        # immutable diagnostic next to the otherwise-empty probe directory so
        # a failed bootstrap can be corrected without relying on terminal
        # scrollback (which is intentionally not a runtime dependency).
        raw_args = sys.argv[1:]
        if "--out-dir" in raw_args:
            out_dir = Path(raw_args[raw_args.index("--out-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(out_dir / "failure.json", {
                "schema": "strict_r3_p8u_canonical_adapter_parity_failure_v1",
                "status": "fail",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "outcome_columns_consumed": [],
                "state_bundle_published": False,
            })
        raise
