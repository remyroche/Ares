#!/usr/bin/env python3
"""Seal target-free P8U feature checkpoints to one immutable source panel.

Historical feature parquet files can be numerically unreproducible when the
raw public derivatives sidecars used to create them were not retained.  This
utility deliberately does not infer, repair, or backfill those inputs.  It
creates a new *source-anchored* reference from already-generated canonical
feature checkpoints only after proving that their receipts name the same
source-panel, feature-plan, and universe manifest hashes.

The resulting manifest is accepted by
``probe_strict_r3_p8u_canonical_feature_adapter.py`` for an exact stateful
parity claim.  A legacy reference without this manifest remains useful for
diagnosis, but is explicitly not an exact-contract reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_warm_feature_state import atomic_json, sha256_file


IDENTITY = ("candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name")
CODE_PATHS = (
    ROOT / "extreme_price_movements/features.py",
    ROOT / "extreme_price_movements/feature_transforms.py",
    ROOT / "extreme_price_movements/fast_funcs.py",
    ROOT / "extreme_price_movements/inference/p8u_canonical_feature_adapter.py",
)


def _sha(path: Path) -> str:
    return sha256_file(path)


def _parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=/absolute/or/relative/path")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in name):
        raise argparse.ArgumentTypeError("checkpoint name must be alphanumeric, '_' or '-'")
    result = Path(path).expanduser()
    if not result.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint parquet does not exist: {result}")
    return name, result


def _read_receipt(checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    receipt = checkpoint.parent / "parity_summary.json"
    if not receipt.is_file():
        raise ValueError(f"checkpoint requires sibling parity_summary.json: {checkpoint}")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if payload.get("schema") != "strict_r3_p8u_canonical_adapter_parity_v1":
        raise ValueError(f"unrecognised canonical checkpoint receipt: {receipt}")
    if payload.get("outcome_columns_consumed") not in ([], None):
        raise ValueError(f"checkpoint receipt declares outcome consumption: {receipt}")
    return receipt, payload


def seal_source_anchored_reference(
    *,
    source_panel: Path,
    feature_plan: Path,
    canonical_manifest: Path,
    checkpoints: dict[str, Path],
    out_dir: Path,
) -> dict[str, Any]:
    """Copy verified target-free checkpoints into a hash-bound reference root."""
    if out_dir.exists():
        raise FileExistsError(f"immutable source-anchored output already exists: {out_dir}")
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    for path in (source_panel, feature_plan, canonical_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    source = joblib.load(source_panel)
    if source.get("schema") != "strict_r3_p8u_canonical_source_panel_state_v1":
        raise ValueError("source panel does not carry the P8U canonical source schema")
    if not isinstance(source.get("panel"), dict):
        raise ValueError("source panel has no primitive panel")

    source_sha = _sha(source_panel)
    plan_sha = _sha(feature_plan)
    manifest_sha = _sha(canonical_manifest)
    plan = json.loads(feature_plan.read_text(encoding="utf-8"))
    fields = [str(value) for value in plan.get("full_union", [])]
    if not fields:
        raise ValueError("feature plan lacks full_union")

    out_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    seen_timestamps: set[str] = set()
    for name, checkpoint in sorted(checkpoints.items()):
        receipt_path, receipt = _read_receipt(checkpoint)
        mismatched = {
            "source_panel_sha256": (receipt.get("source_panel_sha256"), source_sha),
            "feature_plan_sha256": (receipt.get("feature_plan_sha256"), plan_sha),
            "canonical_manifest_sha256": (receipt.get("canonical_manifest_sha256"), manifest_sha),
        }
        failed = {key: value for key, value in mismatched.items() if value[0] != value[1]}
        if failed:
            raise ValueError(f"checkpoint {name} is not bound to this source contract: {failed}")
        frame = pd.read_parquet(checkpoint)
        missing = [column for column in (*IDENTITY, *fields) if column not in frame.columns]
        if missing:
            raise ValueError(f"checkpoint {name} misses reference columns: {missing[:8]}")
        frame = frame[[*IDENTITY, *fields]].copy()
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
        timestamps = frame["__ts__"].drop_duplicates()
        if len(timestamps) != 1:
            raise ValueError(f"checkpoint {name} must contain exactly one signal timestamp")
        stamp = timestamps.iloc[0]
        declared = pd.Timestamp(receipt.get("signal_ts"))
        declared = declared.tz_localize("UTC") if declared.tzinfo is None else declared.tz_convert("UTC")
        if stamp != declared:
            raise ValueError(f"checkpoint {name} timestamp disagrees with receipt")
        if stamp.isoformat() in seen_timestamps:
            raise ValueError(f"duplicate source-anchored timestamp: {stamp.isoformat()}")
        seen_timestamps.add(stamp.isoformat())
        if frame["candidate_id"].duplicated().any():
            raise ValueError(f"checkpoint {name} contains duplicate candidate identity")
        target = out_dir / f"checkpoint_{name}.parquet"
        frame.to_parquet(target, index=False, compression="zstd")
        records.append({
            "name": name,
            "signal_ts": stamp.isoformat(),
            "rows": int(len(frame)),
            "fields": int(len(fields)),
            "path": target.name,
            "sha256": _sha(target),
            "producer_checkpoint": str(checkpoint),
            "producer_checkpoint_sha256": _sha(checkpoint),
            "producer_receipt": str(receipt_path),
            "producer_receipt_sha256": _sha(receipt_path),
            # A legacy comparison may fail while its generated values remain
            # valid source-anchored candidates.  Preserve that fact rather
            # than laundering it into a historical-parity assertion.
            "producer_legacy_comparison_status": receipt.get("status"),
            "producer_stateful_graph": bool(receipt.get("stateful_canonical_graph")),
        })

    source_index = source["panel"].get("close")
    payload = {
        "schema": "strict_r3_p8u_source_anchored_reference_v1",
        "reference_kind": "immutable_source_panel_current_code_checkpoint",
        "exact_contract": "only this named immutable source panel and code hash set",
        "legacy_historical_reference_reconciled": False,
        "source_panel": {"path": str(source_panel), "sha256": source_sha},
        "feature_plan": {"path": str(feature_plan), "sha256": plan_sha, "fields": len(fields)},
        "canonical_manifest": {"path": str(canonical_manifest), "sha256": manifest_sha},
        "source_history": {
            "start": str(getattr(source_index, "index", source_index).min()) if source_index is not None else None,
            "end": str(getattr(source_index, "index", source_index).max()) if source_index is not None else None,
            "symbols": len(source.get("symbols", [])),
        },
        "code_sha256": {str(path.relative_to(ROOT)): _sha(path) for path in CODE_PATHS},
        "checkpoints": records,
        "outcome_columns_consumed": [],
    }
    manifest_path = out_dir / "source_anchor_manifest.json"
    atomic_json(manifest_path, payload)
    receipt = {
        "schema": "strict_r3_p8u_source_anchored_reference_receipt_v1",
        "status": "sealed",
        "source_anchor_manifest_sha256": _sha(manifest_path),
        "source_panel_sha256": source_sha,
        "checkpoints": len(records),
        "outcome_columns_consumed": [],
    }
    atomic_json(out_dir / "receipt.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--feature-plan", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, type=_parse_named_path)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    checkpoints = dict(args.checkpoint)
    if len(checkpoints) != len(args.checkpoint):
        raise ValueError("checkpoint names must be unique")
    result = seal_source_anchored_reference(
        source_panel=args.source_panel,
        feature_plan=args.feature_plan,
        canonical_manifest=args.canonical_manifest,
        checkpoints=checkpoints,
        out_dir=args.out_dir,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
