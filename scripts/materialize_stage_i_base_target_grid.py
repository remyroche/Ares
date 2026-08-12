#!/usr/bin/env python3
"""Materialise the frozen 60-arm Stage-I base-target label surface.

The input path pack is deliberately explicit and hash-bound:

``candidate_paths.parquet``
    identity, side/timestamps, entry price, ATR, path-validity, exact gross/net
    reconciliation fields, and a causal regime label used only for slicing.
``h12_paths.npz``
    aligned ``high``, ``low`` and ``close`` float matrices with 720 columns.
``manifest.json``
    declares exact next-minute entry and the aligned artifact hashes.

This command creates labels only.  It never loads model features or fits a
model, and path-derived columns are never inference inputs.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.stage_i_base_target_ablation import (  # noqa: E402
    BaseTargetAblationError,
    file_sha256,
    geometry_grid,
    materialize_h12_geometry_traversal,
    materialize_h12_path_primitives,
    materialize_geometry_labels_from_traversal,
    ordinal_o_target,
    scalar_s_target,
    validate_entry_timing,
)


SCHEMA = "stage_i_base_target_label_grid_v2"
REQUIRED = {
    "candidate_id", "__ts__", "__symbol__", "side_name", "decision_ts", "entry_ts",
    "entry_price", "atr_1h", "path_complete", "label_available_ts", "causal_regime",
    "path_start_ts", "path_end_exclusive",
}


def _canonical_sha(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _materializer_source_fingerprint() -> dict[str, Any]:
    paths = (Path(__file__).resolve(), ROOT / "extreme_price_movements" / "stage_i_base_target_ablation.py")
    payload = {
        "schema": "stage_i_target_grid_materializer_source_v1",
        "files": {str(path.resolve()): file_sha256(path) for path in paths},
    }
    payload["contract_sha256"] = _canonical_sha(payload)
    return payload


def _completed_resume(path: Path, request_sha256: str) -> dict[str, Any] | None:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("request_sha256") != request_sha256:
        raise BaseTargetAblationError("completed target-grid request/source lineage drift")
    inventory = manifest.get("artifact_sha256")
    if not isinstance(inventory, dict) or not inventory:
        raise BaseTargetAblationError("completed target-grid lacks immutable artifact inventory")
    for relative, expected in inventory.items():
        artifact = path / relative
        if not artifact.is_file() or file_sha256(artifact) != expected:
            raise BaseTargetAblationError(f"completed target-grid artifact drift: {relative}")
    return manifest


def _require_path_pack(root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    manifest_path = root / "manifest.json"
    ledger_path = root / "candidate_paths.parquet"
    paths_path = root / "h12_paths.npz"
    if not all(path.is_file() for path in (manifest_path, ledger_path, paths_path)):
        raise BaseTargetAblationError("exact H12 path pack is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("entry_convention") != "signal_timestamp_plus_1h_exact_minute_open"
        or int(manifest.get("horizon_minutes", -1)) != 720
    ):
        raise BaseTargetAblationError("path pack does not declare the frozen signal+1h exact-minute-open H12 semantics")
    source_fingerprint = manifest.get("materializer_source_fingerprint")
    minute_inventory = manifest.get("minute_source_inventory")
    if (
        not isinstance(source_fingerprint, dict)
        or not isinstance(source_fingerprint.get("contract_sha256"), str)
        or not isinstance(minute_inventory, dict)
        or not isinstance(minute_inventory.get("inventory_sha256"), str)
        or not isinstance(minute_inventory.get("path"), str)
    ):
        raise BaseTargetAblationError("path pack lacks source-code/minute-source lineage required by the target grid")
    minute_inventory_path = root / str(minute_inventory["path"])
    if not minute_inventory_path.is_file() or file_sha256(minute_inventory_path) != manifest.get("artifact_sha256", {}).get(str(minute_inventory["path"])):
        raise BaseTargetAblationError("path pack minute-source inventory artifact drift")
    stored_inventory = json.loads(minute_inventory_path.read_text(encoding="utf-8"))
    if stored_inventory.get("inventory_sha256") != minute_inventory["inventory_sha256"]:
        raise BaseTargetAblationError("path pack minute-source inventory contract drift")
    regime = manifest.get("causal_regime_contract")
    if (
        not isinstance(regime, dict)
        or regime.get("column") != "causal_regime"
        or regime.get("causal_at_decision_time") is not True
        or regime.get("diagnostic_noncausal") is not False
        or not isinstance(regime.get("source_manifest_sha256"), str)
        or len(regime["source_manifest_sha256"]) != 64
    ):
        raise BaseTargetAblationError("path pack lacks a hash-bound causal decision-time regime contract")
    artifacts = manifest.get("artifact_sha256")
    if not isinstance(artifacts, dict):
        raise BaseTargetAblationError("path pack lacks artifact hashes")
    if artifacts.get("candidate_paths.parquet") != file_sha256(ledger_path) or artifacts.get("h12_paths.npz") != file_sha256(paths_path):
        raise BaseTargetAblationError("path pack bytes drifted")
    ledger = pd.read_parquet(ledger_path)
    if missing := sorted(REQUIRED.difference(ledger.columns)):
        raise BaseTargetAblationError(f"path-pack ledger lacks {missing}")
    if ledger.candidate_id.isna().any() or ledger.candidate_id.duplicated().any():
        raise BaseTargetAblationError("path-pack candidate identities are invalid")
    validate_entry_timing(ledger["__ts__"], ledger.decision_ts, ledger.entry_ts)
    archive = np.load(paths_path, mmap_mode="r", allow_pickle=False)
    if set(archive.files) != {"high", "low", "close", "entry_open", "path_start_ns", "identity_sha256"}:
        raise BaseTargetAblationError(
            "path NPZ must contain high/low/close, entry_open, path_start_ns and identity_sha256 row lineage"
        )
    high, low, close = (archive[name] for name in ("high", "low", "close"))
    if any(array.shape != (len(ledger), 720) for array in (high, low, close)):
        raise BaseTargetAblationError("path arrays are not aligned rows x 720")
    start = pd.to_datetime(ledger.path_start_ts, utc=True, errors="raise")
    end = pd.to_datetime(ledger.path_end_exclusive, utc=True, errors="raise")
    entry = pd.to_datetime(ledger.entry_ts, utc=True, errors="raise")
    if not start.eq(entry).all() or not end.eq(start + pd.Timedelta(minutes=720)).all():
        raise BaseTargetAblationError("path ledger must bind [entry_ts, entry_ts+H12) exactly")
    available = pd.to_datetime(ledger.label_available_ts, utc=True, errors="raise")
    if not available.eq(end).all():
        raise BaseTargetAblationError(
            "label_available_ts must equal the exclusive H12 path endpoint; premature availability leaks"
        )
    np_start = np.asarray(archive["path_start_ns"], dtype=np.int64)
    if np_start.shape != (len(ledger),) or not np.array_equal(np_start, start.astype("int64").to_numpy(np.int64)):
        raise BaseTargetAblationError("path-array start timestamps are not identity-row aligned")
    expected_identity = np.vstack([
        np.frombuffer(sha256(
            (str(row.candidate_id) + "\x1f" + pd.Timestamp(row.entry_ts).isoformat()).encode("utf-8")
        ).digest(), dtype=np.uint8)
        for row in ledger[["candidate_id", "entry_ts"]].itertuples(index=False)
    ])
    observed_identity = np.asarray(archive["identity_sha256"], dtype=np.uint8)
    if observed_identity.shape != expected_identity.shape or not np.array_equal(observed_identity, expected_identity):
        raise BaseTargetAblationError("path arrays do not bind candidate_id + exact entry timestamp row order")
    observed_entry = np.asarray(archive["entry_open"], dtype=np.float64)
    declared_entry = pd.to_numeric(ledger.entry_price, errors="coerce").to_numpy(np.float64)
    if observed_entry.shape != (len(ledger),) or not np.allclose(
        observed_entry, declared_entry, rtol=0.0, atol=1e-7, equal_nan=True,
    ):
        raise BaseTargetAblationError("declared entry_price differs from the hash-bound exact minute open")
    return ledger, high, low, close, manifest


def materialize(path_pack: Path, output_dir: Path, *, resume: bool = False) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"refusing to overwrite target-grid output: {output_dir}")
    ledger, high, low, close, source = _require_path_pack(path_pack)
    source_fingerprint = _materializer_source_fingerprint()
    request = {
        "schema": "stage_i_base_target_label_grid_request_v2",
        "path_pack_manifest_sha256": file_sha256(path_pack / "manifest.json"),
        "path_pack_contract_sha256": source.get("contract_sha256"),
        "path_pack_source_code_sha256": source["materializer_source_fingerprint"]["contract_sha256"],
        "minute_source_inventory_sha256": source["minute_source_inventory"]["inventory_sha256"],
        "grid_materializer_source_fingerprint": source_fingerprint,
        "geometry_grid": [item.to_dict() for item in geometry_grid()],
    }
    request_sha256 = _canonical_sha(request)
    if resume:
        prior = _completed_resume(output_dir, request_sha256)
        if prior is not None:
            return prior
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("partial/unmatched target-grid requires a fresh output directory")
    sign = np.where(ledger.side_name.astype(str).str.lower().eq("short"), -1, 1).astype(np.int8)
    base_columns = [
        "candidate_id", "__ts__", "__symbol__", "side_name", "decision_ts", "entry_ts",
        "label_available_ts", "causal_regime",
    ]
    frames: list[pd.DataFrame] = []
    event_matrix: list[np.ndarray] = []
    primitives = materialize_h12_path_primitives(
        entry_price=ledger.entry_price.to_numpy(), atr=ledger.atr_1h.to_numpy(),
        side_sign=sign, high=high, low=low, close=close,
        path_complete=ledger.path_complete.to_numpy(),
    )
    traversal = materialize_h12_geometry_traversal(primitives)
    for geometry in geometry_grid():
        label = materialize_geometry_labels_from_traversal(traversal, geometry)
        event_matrix.append(label.event.copy())
        item = ledger.loc[:, base_columns].copy()
        item["geometry"] = geometry.key
        item["target_valid"] = label.valid
        item["event"] = label.event
        item["event_minute"] = label.event_minute
        item["favorable_progress"] = label.favorable_progress
        item["adverse_progress"] = label.adverse_progress
        item["dominance"] = label.dominance
        item["gross_bps"] = label.gross_bps
        item["net_bps"] = label.net_bps
        item["upper_fraction"] = label.upper_fraction
        item["lower_fraction"] = label.lower_fraction
        item["upper_floor_bound"] = label.upper_floor_bound
        item["upper_cap_bound"] = label.upper_cap_bound
        item["S_target"] = scalar_s_target(label.event, label.dominance)
        for alpha in (0.25, 0.33, 0.50):
            item[f"O_a{str(alpha).replace('.', 'p')}_target"] = ordinal_o_target(
                label.event, label.dominance, alpha
            )
        item["promotion_eligible_geometry"] = geometry.promotion_eligible_geometry
        item["geometry_disposition"] = geometry.disposition
        frames.append(item)
    # Contract certainty is the fraction of *all 15 preregistered contracts*
    # agreeing with the local event.  It is persisted as training-only weight
    # metadata and explicitly barred from inference features.
    events = np.column_stack(event_matrix)
    for index, item in enumerate(frames):
        valid = item.target_valid.to_numpy(bool)
        agreement = np.mean(events == events[:, [index]], axis=1)
        agreement[~valid] = np.nan
        item["contract_certainty"] = agreement.astype(np.float32)
    surface = pd.concat(frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_path = output_dir / "target_repair_labels.parquet"
    surface.to_parquet(label_path, index=False, compression="zstd")
    manifest = {
        "schema": SCHEMA, "status": "complete", "rows": int(len(surface)),
        "candidate_rows": int(len(ledger)), "geometries": 15, "target_arms": 60,
        "promotable_geometries": 12, "promotable_target_arms": 48,
        "entry": "decision_ts = signal __ts__ +1h; exact minute-bar open indexed at decision_ts (no additional minute)",
        "horizon_minutes": 720, "same_minute_conflict": "adverse precedence",
        "label_availability": "entry_ts + H12, equal to path_end_exclusive",
        "geometry": "lower=k_sl*ATR; upper=clip(k_tp*ATR,1.5%,4%)",
        "progress": {
            "favorable": "clip(H12 side-normalized peak favorable excursion / upper barrier,0,1)",
            "adverse": "clip(H12 side-normalized peak adverse excursion / lower barrier,0,1)",
            "dominance": "favorable_progress - adverse_progress",
        },
        "target_S": "lower=0; upper=1; timeout=.35+.30*sigmoid(dominance/.20)",
        "target_O": "classes 0=lower,4=upper; timeout 1/2/3 by dominance below -alpha, within +/-alpha, above +alpha; alpha=.25/.33/.50",
        "cost": "100 bps applied once to economic net diagnostic; never changes physical barriers",
        "invalid": "incomplete/nonfinite paths retain identity but every target is invalid/excluded",
        "contract_certainty": "training-only nearby-contract event agreement; forbidden inference feature",
        "causal_regime_contract": source["causal_regime_contract"],
        "target_storage": "compact four columns per geometry (S plus three O alphas); 60 arm identities are geometry x target contract, not 60 sparse columns",
        "path_primitive_reuse": {
            "schema": "stage_i_h12_path_primitive_reuse_v1",
            "raw_ohlc_normalisations": 1,
            "distinct_upper_first_touch_traversals": 5,
            "distinct_lower_first_touch_traversals": 3,
            "geometry_contracts_derived": 15,
            "target_neutral": True,
        },
        "source_manifest_path": str((path_pack / "manifest.json").resolve()),
        "source_manifest_sha256": file_sha256(path_pack / "manifest.json"),
        "source_manifest_contract_sha256": _canonical_sha(source),
        "source_materializer_code_contract_sha256": source["materializer_source_fingerprint"]["contract_sha256"],
        "minute_source_inventory_sha256": source["minute_source_inventory"]["inventory_sha256"],
        "materializer_source_fingerprint": source_fingerprint,
        "request": request,
        "request_sha256": request_sha256,
        "artifact_sha256": {"target_repair_labels.parquet": file_sha256(label_path)},
    }
    manifest["contract_sha256"] = _canonical_sha({
        key: value for key, value in manifest.items() if key not in {"artifact_sha256", "contract_sha256"}
    })
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-pack", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(materialize(args.path_pack, args.output_dir, resume=args.resume), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
