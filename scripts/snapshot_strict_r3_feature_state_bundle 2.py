#!/usr/bin/env python3
"""Freeze and inventory a reusable strict-R3 causal feature-state bundle.

The snapshot is content addressed and contains every persisted operator state
needed to continue either live hourly generation or chronological training
materialisation.  Source caches remain mutable; the snapshot never is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


SCHEMA = "strict_r3_causal_feature_state_bundle_v2"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
IMPLEMENTATION_FILES = (
    "extreme_price_movements/features.py",
    "extreme_price_movements/features_oi.py",
    "extreme_price_movements/inference/live_zscore_state.py",
    "extreme_price_movements/inference/derived_feature_state.py",
    "extreme_price_movements/inference/price_memory_state.py",
    "extreme_price_movements/inference/primitive_price_state.py",
    "extreme_price_movements/inference/price_memory_pipeline_state.py",
    "extreme_price_movements/inference/cross_sectional_composite_state.py",
    "extreme_price_movements/inference/orderbook_feature_state.py",
    "extreme_price_movements/inference/residual_surprise_state.py",
    "extreme_price_movements/inference/simple_context_state.py",
    "extreme_price_movements/inference/spectral_oi_geometry_state.py",
    "extreme_price_movements/inference/strict_r3_final14_state.py",
    "scripts/materialize_strict_r3_forward_features_incremental_v13.py",
    "scripts/snapshot_strict_r3_feature_state_bundle.py",
    "scripts/restore_strict_r3_feature_state_bundle.py",
    "scripts/bootstrap_strict_r3_fixed_ffd_state.py",
    "scripts/bootstrap_strict_r3_price_memory_pipeline_state.py",
    "scripts/bootstrap_strict_r3_orderbook_feature_state.py",
    "scripts/update_strict_r3_feature_panel_state.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npz_metadata(path: Path) -> dict:
    try:
        with np.load(path, allow_pickle=False) as data:
            if "metadata" not in data.files:
                return {}
            return json.loads(str(data["metadata"].item()))
    except Exception as exc:
        return {"metadata_error": f"{type(exc).__name__}: {exc}"}


def _sqlite_metadata(path: Path) -> dict:
    """Read semantic identity and time bounds from a closed state database."""
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5.0)
        try:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            metadata = (
                dict(connection.execute("SELECT key, value FROM metadata"))
                if "metadata" in tables else {}
            )
            if "feature_rows" in tables:
                bounds = connection.execute(
                    "SELECT MIN(timestamp_ns), MAX(timestamp_ns), "
                    "COUNT(DISTINCT feature_key), COUNT(*) FROM feature_rows"
                ).fetchone()
                metadata.update(
                    {
                        "first_timestamp": (
                            pd.Timestamp(int(bounds[0]), tz="UTC").isoformat()
                            if bounds and bounds[0] is not None else None
                        ),
                        "last_timestamp": (
                            pd.Timestamp(int(bounds[1]), tz="UTC").isoformat()
                            if bounds and bounds[1] is not None else None
                        ),
                        "feature_count": int(bounds[2] or 0) if bounds else 0,
                        "row_count": int(bounds[3] or 0) if bounds else 0,
                    }
                )
            return metadata
        finally:
            connection.close()
    except Exception as exc:
        return {"metadata_error": f"{type(exc).__name__}: {exc}"}


def _json_metadata(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception as exc:
        return {"metadata_error": f"{type(exc).__name__}: {exc}"}


def _final14_metadata(path: Path) -> dict:
    """Read only the frozen final-14 envelope metadata.

    Feature-state bundles are trusted local artifacts.  The state itself is
    intentionally not returned or mutated here; snapshotting only needs its
    semantic identity and append watermark for inventory validation.
    """
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("final-14 state envelope is not a mapping")
        return {
            "schema": payload.get("schema"),
            "contract_hash": payload.get("contract_hash"),
            "last_timestamp": payload.get("last_timestamp"),
        }
    except Exception as exc:
        return {"metadata_error": f"{type(exc).__name__}: {exc}"}


def _link_or_copy(source: Path, target: Path) -> str:
    # A state snapshot is immutable even if its source cache remains active.
    # Hardlinks violate that guarantee for SQLite/in-place writers, so pay the
    # small one-time copy cost at bundle publication.
    shutil.copy2(source, target)
    return "copy"


def _freeze_panel_state(
    source: Path,
    target: Path,
    *,
    tail_hours: int | None,
) -> dict[str, object]:
    """Copy or causally trim the source-panel checkpoint into the bundle.

    Long-memory feature operators live in the separately persisted operator
    states.  A resumed producer therefore needs only a bounded primitive tail
    for direct transforms and continuity checks, not the complete historical
    panel.  Keeping the tail inside the immutable bundle makes a training or
    inference resume self-contained and avoids loading years of source arrays.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    original_sha256 = _sha256(source)
    if tail_hours is None:
        shutil.copy2(source, target)
        return {
            "source_panel_original_sha256": original_sha256,
            "panel_state_sha256": _sha256(target),
            "panel_tail_hours": None,
            "panel_rows": None,
            "panel_start": None,
            "panel_end": None,
        }
    if tail_hours < 72:
        raise ValueError("bundled source-panel tail must retain at least 72 hours")
    state = joblib.load(source)
    panel = state.get("panel")
    if not isinstance(panel, dict) or not isinstance(panel.get("close"), pd.DataFrame):
        raise ValueError("panel-state checkpoint lacks a canonical close panel")
    close = panel["close"]
    if close.empty:
        raise ValueError("panel-state checkpoint is empty")
    latest = pd.Timestamp(close.index.max())
    latest = latest.tz_localize("UTC") if latest.tzinfo is None else latest.tz_convert("UTC")
    tail_start = latest - pd.Timedelta(hours=tail_hours - 1)
    bounded: dict[str, object] = {}
    for name, value in panel.items():
        if isinstance(value, pd.DataFrame):
            index = pd.to_datetime(value.index, utc=True)
            tail = value.loc[index >= tail_start].copy()
            # Order-book feature definitions explicitly forward-fill their
            # causal source before applying a one-bar shift. Seed the retained
            # tail with the exact pre-tail carry so restoring the bounded panel
            # is numerically identical to retaining the discarded prefix.
            if str(name).startswith("orderbook_") and not tail.empty:
                prior = value.loc[index < tail_start]
                if not prior.empty:
                    carry = prior.ffill().iloc[-1]
                    first = tail.iloc[0].copy()
                    missing = first.isna() & carry.notna()
                    if bool(missing.any()):
                        first.loc[missing] = carry.loc[missing]
                        tail.iloc[0] = first
            bounded[name] = tail
        else:
            bounded[name] = value
    frozen = dict(state)
    frozen["source_history_start"] = state.get("source_history_start", state.get("history_start"))
    frozen["history_start"] = tail_start
    frozen["panel_tail_hours"] = int(tail_hours)
    frozen["panel"] = bounded
    joblib.dump(frozen, target, compress=3)
    bounded_close = bounded["close"]
    return {
        "source_panel_original_sha256": original_sha256,
        "panel_state_sha256": _sha256(target),
        "panel_tail_hours": int(tail_hours),
        "panel_rows": int(len(bounded_close)),
        "panel_start": str(pd.Timestamp(bounded_close.index.min())),
        "panel_end": str(pd.Timestamp(bounded_close.index.max())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--panel-state", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--contract-hash", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument(
        "--panel-tail-hours",
        type=int,
        help=(
            "Freeze only this causal source-panel tail inside the bundle. "
            "Operator states retain all longer memory. Omit to copy the "
            "panel checkpoint byte-for-byte."
        ),
    )
    parser.add_argument(
        "--required-state-kind",
        action="append",
        default=[],
        help=(
            "Require this operator-state kind in the snapshot. Repeat for the "
            "complete schema-v3 DAG."
        ),
    )
    parser.add_argument(
        "--expected-state-timestamp",
        help=(
            "Require every timestamped row of each --required-state-kind to "
            "end at this UTC timestamp."
        ),
    )
    args = parser.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(f"immutable state bundle exists: {args.out_dir}")
    if not args.cache_dir.is_dir():
        raise FileNotFoundError(args.cache_dir)
    if not args.panel_state.is_file():
        raise FileNotFoundError(args.panel_state)
    active_sqlite = sorted(args.cache_dir.rglob("*.sqlite-wal"))
    if active_sqlite:
        raise RuntimeError(
            "feature-state cache has an active SQLite WAL; close the producer "
            f"before snapshotting: {active_sqlite[:3]}"
        )

    temporary = args.out_dir.with_name(args.out_dir.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary state bundle exists: {temporary}")
    states_dir = temporary / "states"
    states_dir.mkdir(parents=True)
    bundled_panel = temporary / "source_panel" / "feature_panel_state.joblib"
    panel_receipt = _freeze_panel_state(
        args.panel_state,
        bundled_panel,
        tail_hours=args.panel_tail_hours,
    )

    source_files = sorted(
        path
        for path in args.cache_dir.rglob("*")
        if path.is_file()
        and path.name != "state_restore_receipt.json"
        and not path.name.endswith(("-wal", "-shm"))
    )
    if not source_files:
        raise RuntimeError("feature-state cache is empty")

    inventory: list[dict] = []
    transfer_modes: set[str] = set()
    for source in source_files:
        relative = source.relative_to(args.cache_dir)
        target = states_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        transfer = _link_or_copy(source, target)
        transfer_modes.add(transfer)
        metadata = (
            _final14_metadata(source)
            if source.name == "strict_r3_final14.state"
            else
            _npz_metadata(source)
            if source.suffix == ".npz"
            else _sqlite_metadata(source)
            if source.suffix == ".sqlite"
            else _json_metadata(source)
            if source.suffix == ".json"
            else {}
        )
        kind = (
            "strict_r3_final14"
            if source.name == "strict_r3_final14.state"
            else
            "primitive_price"
            if source.name == "primitive_price_state.npz"
            else "price_memory"
            if source.name == "price_memory_feature_state.npz"
            else "cross_sectional_composite"
            if source.name == "cross_sectional_composite_state.json"
            else "orderbook_precomposite"
            if source.name == "orderbook_feature_state.npz"
            else
            "raw_rolling"
            if source.name.startswith("raw_rolling_state")
            else "causal_transform"
            if source.name.startswith("causal_transform_state")
            else "nested_derived"
            if source.name == "nested_derived_feature_state.sqlite"
            else "derived_history"
            if "derived_history" in relative.parts
            else "oi_long_iqr"
            if "oi_long_iqr" in relative.parts
            else "fixed_ffd"
            if "fixed_ffd" in relative.parts
            else "market_spectral_history"
            if source.name == "market_spectral_source_state.history.npz"
            else "market_spectral_contract"
            if source.name == "market_spectral_source_state.json"
            else "other"
        )
        inventory.append(
            {
                "relative_path": relative.as_posix(),
                "kind": kind,
                "op": metadata.get("op"),
                "name": metadata.get("name", metadata.get("schema")),
                "window": metadata.get("window"),
                "first_timestamp": metadata.get("first_timestamp"),
                "last_timestamp": metadata.get("last_timestamp"),
                "version": metadata.get("version"),
                "symbol_count": len(
                    metadata.get("symbol_order", metadata.get("symbols", []))
                ),
                "feature_count": metadata.get("feature_count"),
                "row_count": metadata.get("row_count"),
                "state_contract_hash": metadata.get("contract_hash"),
                "bytes": int(source.stat().st_size),
                "sha256": _sha256(source),
                "transfer": transfer,
                "metadata_error": metadata.get("metadata_error"),
            }
        )

    table = pd.DataFrame(inventory).sort_values(
        ["kind", "op", "name", "window", "relative_path"],
        na_position="last",
        kind="stable",
    )
    # Re-read the composite envelope after inventory construction.  Mixed
    # metadata rows can otherwise leave its scalar watermark represented as
    # missing in the DataFrame even though the hash-bound state restores
    # correctly.  The explicit second read is cheap (one file) and keeps the
    # publication gate fail-closed.
    for row_index in table.index[table["kind"].eq("strict_r3_final14")]:
        metadata = _final14_metadata(
            args.cache_dir / str(table.at[row_index, "relative_path"])
        )
        if metadata.get("metadata_error"):
            raise ValueError(
                "strict-R3 final-14 metadata cannot be verified: "
                f"{metadata['metadata_error']}"
            )
        table.at[row_index, "last_timestamp"] = metadata.get("last_timestamp")
        table.at[row_index, "state_contract_hash"] = metadata.get(
            "contract_hash"
        )
    required_kinds = tuple(dict.fromkeys(map(str, args.required_state_kind)))
    missing_kinds = sorted(set(required_kinds).difference(set(table["kind"])))
    if missing_kinds:
        raise ValueError(f"required operator-state kinds are absent: {missing_kinds}")
    expected_state_timestamp = None
    if args.expected_state_timestamp:
        expected_state_timestamp = pd.Timestamp(args.expected_state_timestamp)
        if expected_state_timestamp.tzinfo is None:
            expected_state_timestamp = expected_state_timestamp.tz_localize("UTC")
        else:
            expected_state_timestamp = expected_state_timestamp.tz_convert("UTC")
        required_rows = table.loc[table["kind"].isin(required_kinds)]
        missing_watermark = required_rows["last_timestamp"].isna()
        if bool(missing_watermark.any()):
            paths = required_rows.loc[missing_watermark, "relative_path"].tolist()
            raise ValueError(f"required operator states lack watermarks: {paths}")
        actual = pd.to_datetime(required_rows["last_timestamp"], utc=True)
        mismatched = required_rows.loc[
            ~actual.eq(expected_state_timestamp).to_numpy(), "relative_path"
        ].tolist()
        if mismatched:
            raise ValueError(
                "required operator states do not share the expected timestamp: "
                f"{mismatched}"
            )
    table.to_parquet(temporary / "operator_state_inventory.parquet", index=False)
    raw = table.loc[table["kind"].eq("raw_rolling")]
    manifest = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": args.scope,
        "feature_contract_sha256": args.contract_hash,
        "panel_state": "source_panel/feature_panel_state.joblib",
        **panel_receipt,
        "source_cache_dir": str(args.cache_dir),
        "state_files": int(len(table)),
        "state_bytes": int(table["bytes"].sum()),
        "raw_rolling_states": int(len(raw)),
        "causal_transform_states": int(
            table["kind"].eq("causal_transform").sum()
        ),
        "derived_history_states": int(table["kind"].eq("derived_history").sum()),
        "nested_derived_states": int(table["kind"].eq("nested_derived").sum()),
        "price_memory_states": int(table["kind"].eq("price_memory").sum()),
        "primitive_price_states": int(table["kind"].eq("primitive_price").sum()),
        "cross_sectional_composite_states": int(
            table["kind"].eq("cross_sectional_composite").sum()
        ),
        "orderbook_precomposite_states": int(
            table["kind"].eq("orderbook_precomposite").sum()
        ),
        "strict_r3_final14_states": int(
            table["kind"].eq("strict_r3_final14").sum()
        ),
        "oi_long_iqr_states": int(table["kind"].eq("oi_long_iqr").sum()),
        "fixed_ffd_states": int(table["kind"].eq("fixed_ffd").sum()),
        "market_spectral_history_states": int(
            table["kind"].eq("market_spectral_history").sum()
        ),
        "raw_operator_counts": {
            str(key): int(value)
            for key, value in raw["op"].value_counts(dropna=False).items()
        },
        "earliest_state_timestamp": (
            str(pd.to_datetime(table["last_timestamp"], utc=True).min())
            if table["last_timestamp"].notna().any()
            else None
        ),
        "latest_state_timestamp": (
            str(pd.to_datetime(table["last_timestamp"], utc=True).max())
            if table["last_timestamp"].notna().any()
            else None
        ),
        "transfer_modes": sorted(transfer_modes),
        "required_state_kinds": list(required_kinds),
        "expected_state_timestamp": (
            expected_state_timestamp.isoformat()
            if expected_state_timestamp is not None else None
        ),
        "inventory_sha256": _sha256(
            temporary / "operator_state_inventory.parquet"
        ),
        "implementation_sha256": {
            relative: _sha256(ROOT / relative)
            for relative in IMPLEMENTATION_FILES
        },
    }
    (temporary / "state_bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    temporary.rename(args.out_dir)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
