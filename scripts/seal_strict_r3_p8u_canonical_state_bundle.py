#!/usr/bin/env python3
"""Seal a private canonical P8U transform-state checkpoint.

The P8U canonical feature graph stores several different state mechanisms
(NPZ operators and two SQLite containers).  A state directory is not itself a
safe inference artifact: a live WAL can omit committed pages and it carries no
semantic identity.  This utility copies an already parity-proven private state
directory, checkpoints its SQLite WALs *in the copy*, inventories every state
file, and writes an immutable, hash-bound manifest.

It has no scoring, policy, portfolio, exchange, or order-submission code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_production_contract import artifact_hash  # noqa: E402
from extreme_price_movements.inference.p8u_warm_feature_state import (  # noqa: E402
    P8U_REQUIRED_STATE_KINDS,
    feature_union_sha256,
    sha256_file,
)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _utc(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _features(path: Path) -> tuple[str, ...]:
    payload = _json(path)
    fields = payload.get("full_union")
    if not isinstance(fields, list) or not fields:
        raise ValueError("feature plan lacks full_union")
    result = tuple(map(str, fields))
    if len(result) != len(set(result)):
        raise ValueError("feature plan has duplicate features")
    return result


def _checkpoint_sqlite(path: Path) -> None:
    """Fold a copied SQLite WAL into its database; source is never opened RW."""
    connection = sqlite3.connect(str(path))
    try:
        # A state snapshot must be self-contained.  ``TRUNCATE`` is stronger
        # than a passive checkpoint and makes it safe to remove sidecar files
        # after the connection closes.
        result = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if result is not None and int(result[0]) != 0:
            raise RuntimeError(f"SQLite checkpoint remained busy for {path}: {result}")
    finally:
        connection.close()
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{path}{suffix}")
        if sidecar.exists():
            if sidecar.stat().st_size:
                raise RuntimeError(f"SQLite checkpoint left non-empty sidecar: {sidecar}")
            sidecar.unlink()


def _state_kinds(root: Path) -> dict[str, list[Path]]:
    checks: dict[str, list[Path]] = {
        "raw_rolling": sorted(root.glob("raw_rolling_state.*.npz")),
        "causal_transform": [root / "causal_transform_state.container.sqlite"],
        "derived_history": sorted((root / "derived_history").glob("*.npz")),
        "nested_derived": [root / "nested_derived_state.sqlite"],
        "oi_long_iqr": sorted((root / "oi_long_iqr").glob("*.npz")),
        "fixed_ffd": sorted((root / "fixed_ffd").glob("*.npz")),
        "market_spectral_history": [root / "market_spectral_state.history.npz", root / "market_spectral_state.npz"],
        "grouped_rolling": [root / "grouped_rolling_state.npz"],
        "ewma": sorted((root / "ewma_state").glob("*.npz")),
        "regime_transition": [root / "regime_transition_state.npz"],
    }
    required_alias = {
        "raw_rolling",
        "causal_transform",
        "derived_history",
        "nested_derived",
        "oi_long_iqr",
        "fixed_ffd",
        "market_spectral_history",
    }
    if required_alias != set(P8U_REQUIRED_STATE_KINDS):
        raise AssertionError("P8U state-kind contract changed; update state sealer")
    missing = {
        name: [path for path in paths if not path.is_file()]
        for name, paths in checks.items()
        if not paths or any(not path.is_file() for path in paths)
    }
    if missing:
        preview = {name: [str(path) for path in paths[:3]] for name, paths in missing.items()}
        raise FileNotFoundError(f"canonical transform state is incomplete: {preview}")
    return checks


def _inventory(root: Path, kinds: dict[str, list[Path]]) -> pd.DataFrame:
    names: dict[Path, str] = {}
    for kind, files in kinds.items():
        for file in files:
            names[file.resolve()] = kind
    rows: list[dict[str, object]] = []
    for file in sorted((candidate for candidate in root.rglob("*") if candidate.is_file()), key=lambda item: item.as_posix()):
        if file.name.endswith(("-wal", "-shm")):
            raise RuntimeError(f"uncheckpointed SQLite sidecar in sealed state: {file}")
        role = names.get(file.resolve(), "auxiliary")
        rows.append({
            "relative_path": file.relative_to(root).as_posix(),
            "kind": role,
            "bytes": int(file.stat().st_size),
            "sha256": sha256_file(file),
        })
    return pd.DataFrame(rows)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state-dir", type=Path, required=True)
    parser.add_argument("--feature-plan", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--source-panel", type=Path, required=True)
    parity_group = parser.add_mutually_exclusive_group(required=True)
    parity_group.add_argument(
        "--parity-receipt",
        type=Path,
        help="Zero-mismatch canonical feature-probe receipt (legacy seal path).",
    )
    parity_group.add_argument(
        "--staged-score-parity-receipt",
        type=Path,
        help=(
            "Zero-mismatch staged-score receipt which is itself bound to an "
            "independent canonical full-vector audit."
        ),
    )
    parser.add_argument("--state-scope", required=True)
    parser.add_argument("--latest-state-timestamp", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_state_dir.resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    parity_receipt_path = args.parity_receipt or args.staged_score_parity_receipt
    for path in (args.feature_plan, args.canonical_manifest, args.source_panel, parity_receipt_path):
        if not path.resolve().is_file():
            raise FileNotFoundError(path)
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f"immutable P8U state bundle already exists: {out}")
    if ROOT not in source.parents or ROOT not in out.parents:
        raise ValueError("P8U state paths must remain below repository root")

    parity = _json(parity_receipt_path)
    features = _features(args.feature_plan)
    union_hash = feature_union_sha256(features)
    latest = _utc(args.latest_state_timestamp)
    staged_mode = args.staged_score_parity_receipt is not None
    if parity.get("status") != "pass" or int(parity.get("mismatch_cells", -1)) != 0:
        raise ValueError("P8U canonical state may only seal from a zero-mismatch parity receipt")
    if staged_mode:
        # The staged score receipt is deliberately a two-leg proof: it first
        # verifies the cached-state chain against persisted control outputs;
        # the named control receipt must independently verify those outputs
        # against the broad canonical vector graph.
        if int(parity.get("control_full_vector_feature_count", -1)) != len(features):
            raise ValueError("staged score parity feature count differs from sealed plan")
        if int(parity.get("direct_feature_count", -1)) + int(
            parity.get("regular_vector_feature_count", -1)
        ) != len(features):
            raise ValueError("staged score parity does not cover the full frozen feature contract")
        if parity.get("outcome_columns_consumed") or parity.get("policy_or_portfolio_called") or parity.get("exchange_or_order_submission_called"):
            raise ValueError("staged score parity must remain target-free and non-executable")
        control_path = Path(str(parity.get("control_correctness_receipt", "")))
        if not control_path.is_file() or sha256_file(control_path) != str(
            parity.get("control_correctness_receipt_sha256", "")
        ):
            raise ValueError("staged score parity control receipt is absent or hash-mismatched")
        control = _json(control_path)
        if control.get("status") != "pass" or int(control.get("mismatch_cells", -1)) != 0:
            raise ValueError("staged score parity control is not an independent zero-mismatch audit")
        if int(control.get("feature_count", -1)) != len(features):
            raise ValueError("staged score parity control feature count differs from sealed plan")
        timestamps = parity.get("timestamps")
        if not isinstance(timestamps, list) or not timestamps:
            raise ValueError("staged score parity receipt has no checkpoint timeline")
        last_source_timestamp = _utc(str(timestamps[-1].get("source_timestamp")))
        if latest != last_source_timestamp:
            raise ValueError("staged state latest timestamp must equal final score-parity source timestamp")
        candidate_root = Path(str(parity.get("candidate_root", ""))).resolve()
        if candidate_root not in source.parents:
            raise ValueError("staged score parity receipt does not own the source state directory")
        checkpoint = {
            "schema": "strict_r3_p8u_canonical_state_checkpoint_v1",
            "state_scope": str(args.state_scope),
            "as_of_timestamp": latest.isoformat(),
            "source_panel_sha256": sha256_file(args.source_panel),
            "feature_plan_sha256": sha256_file(args.feature_plan),
            "feature_contract_sha256": union_hash,
            "proof_kind": "staged_score_parity_bound_to_full_vector_control",
            "staged_score_parity_receipt_sha256": sha256_file(parity_receipt_path),
        }
        parity_summary = {
            "fields": len(features),
            "mismatch_cells": int(parity["mismatch_cells"]),
            "max_abs_delta": 0.0,
            "feature_graph_mode": "direct4_plus_regular_vector_state",
            "full_cross_section_universe_symbols": 160,
            "history_tail_hours": 1536,
        }
    else:
        checkpoint_path = source / "canonical_state_checkpoint.json"
        if not checkpoint_path.is_file():
            raise FileNotFoundError("canonical state seal requires a checkpoint assertion from the stateful parity probe")
        checkpoint = _json(checkpoint_path)
        if int(parity.get("fields", -1)) != len(features):
            raise ValueError("P8U parity receipt feature count differs from sealed plan")
        if str(parity.get("feature_plan_sha256")) != sha256_file(args.feature_plan):
            raise ValueError("P8U parity receipt belongs to another feature plan")
        if str(parity.get("canonical_manifest_sha256")) != sha256_file(args.canonical_manifest):
            raise ValueError("P8U parity receipt belongs to another canonical universe")
        if str(parity.get("source_panel_sha256")) != sha256_file(args.source_panel):
            raise ValueError("P8U parity receipt belongs to another primitive source")
        if int(parity.get("history_tail_hours", 0)) < 1536:
            raise ValueError("P8U state parity receipt lacks the required 1,536-hour tail")
        if str(parity.get("state_scope")) != str(args.state_scope):
            raise ValueError("P8U parity receipt state scope differs from requested seal")
        if checkpoint.get("schema") != "strict_r3_p8u_canonical_state_checkpoint_v1":
            raise ValueError("P8U state checkpoint schema mismatch")
        if str(checkpoint.get("state_scope")) != str(args.state_scope):
            raise ValueError("P8U state checkpoint scope differs from requested seal")
        if _utc(checkpoint.get("as_of_timestamp")) != _utc(parity.get("signal_ts")):
            raise ValueError("P8U state checkpoint timestamp differs from parity receipt")
        if str(checkpoint.get("source_panel_sha256")) != sha256_file(args.source_panel):
            raise ValueError("P8U state checkpoint source differs from parity receipt")
        if str(checkpoint.get("feature_plan_sha256")) != sha256_file(args.feature_plan):
            raise ValueError("P8U state checkpoint feature plan differs from parity receipt")
        if latest != _utc(parity["signal_ts"]):
            raise ValueError("state latest timestamp must equal the parity-proven signal timestamp")
        parity_summary = {
            "fields": int(parity["fields"]),
            "mismatch_cells": int(parity["mismatch_cells"]),
            "max_abs_delta": float(parity["max_abs_delta"]),
            "feature_graph_mode": str(parity["feature_graph_mode"]),
            "full_cross_section_universe_symbols": int(parity["full_cross_section_universe_symbols"]),
            "history_tail_hours": int(parity["history_tail_hours"]),
        }

    out.mkdir(parents=True)
    copied = out / "state"
    shutil.copytree(source, copied, copy_function=shutil.copy2)
    for database in (copied / "causal_transform_state.container.sqlite", copied / "nested_derived_state.sqlite"):
        _checkpoint_sqlite(database)
    copied_checkpoint = copied / "canonical_state_checkpoint.json"
    if staged_mode:
        _atomic_json(copied_checkpoint, checkpoint)
    kinds = _state_kinds(copied)
    table = _inventory(copied, kinds)
    table["last_timestamp"] = latest
    table.to_parquet(out / "operator_state_inventory.parquet", index=False, compression="zstd")
    state_hash = artifact_hash(copied, "tree")
    manifest = {
        "schema": "strict_r3_p8u_canonical_transform_state_bundle_v1",
        "status": "sealed_unactivated",
        "state_scope": str(args.state_scope),
        "feature_contract_sha256": union_hash,
        "feature_plan_path": str(args.feature_plan.resolve().relative_to(ROOT)),
        "feature_plan_sha256": sha256_file(args.feature_plan),
        "canonical_manifest_path": str(args.canonical_manifest.resolve().relative_to(ROOT)),
        "canonical_manifest_sha256": sha256_file(args.canonical_manifest),
        "source_panel_path": str(args.source_panel.resolve().relative_to(ROOT)),
        "source_panel_sha256": sha256_file(args.source_panel),
        "parity_receipt_path": str(parity_receipt_path.resolve().relative_to(ROOT)),
        "parity_receipt_sha256": sha256_file(parity_receipt_path),
        "parity_mode": "staged_score_bound" if staged_mode else "canonical_feature_probe",
        "state_checkpoint_sha256": sha256_file(copied_checkpoint),
        "parity": {
            **parity_summary,
        },
        "latest_state_timestamp": latest.isoformat(),
        "state_tree_sha256": state_hash,
        "operator_file_count": int(len(table)),
        "required_state_kinds": sorted(P8U_REQUIRED_STATE_KINDS),
        "additional_state_kinds": sorted(set(kinds).difference(P8U_REQUIRED_STATE_KINDS)),
        "sqlite_wal_checkpointed": True,
        "outcome_columns_consumed": [],
        "state_bundle_published": False,
    }
    _atomic_json(out / "state_bundle_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
