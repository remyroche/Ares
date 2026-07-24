#!/usr/bin/env python3
"""Add immutable close-email diagnostics to a frozen admission reference.

The threshold-basis admission contract deliberately keeps a narrow reference
schema.  That is useful for live latency, but it omits path-quality outcomes
that make an archetype-specific close email actionable.  This utility copies
the exact reference rows to a new parquet, appends only historical diagnostics,
and repoints the policy to the enriched copy.  The row count and every column
used by admission remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


_SOURCE_COLUMNS = (
    "__ts__",
    "__symbol__",
    "side_name",
    "archetype_policy_key",
    "clean_exec",
    "dirty_positive",
    "first_touch_bad_mae_1r",
    "full_path_bad_mae_1r",
    "bad_mae_1r",
    "timeout",
    "timed_out",
    "full_stop_loss",
    "stop_loss",
    "stop_hit",
    "gmm_cluster_id",
    "aegmm_cluster",
    "side_aegmm_cluster",
    "gmm_posterior_max",
)
_DIAGNOSTIC_COLUMNS = tuple(column for column in _SOURCE_COLUMNS[4:])
_KEY_COLUMNS = ("timestamp", "symbol", "side_name", "policy_archetype")
_LABEL_KEY_COLUMNS = ("timestamp", "symbol", "side_name")
_LABEL_MAE_COLUMN = "first_touch_mae_to_sl"
_LABEL_PATH_COLUMNS = (
    "first_touch_tp_hit",
    "first_touch_stop",
    "first_touch_timeout",
)
_LABEL_SOURCE_COLUMNS = (
    "__ts__",
    "__symbol__",
    "side_name",
    "__first_touch_mae_to_sl__",
    "__first_touch_hit__",
    "__first_touch_stop__",
    "__first_touch_timeout__",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backup_once(path: Path) -> Path | None:
    if not path.is_file():
        return None
    backup = path.with_suffix(path.suffix + ".pre_email_baseline_28d")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def _canonical_archetype(side: object, value: object) -> str:
    side_name = str(side or "").strip().lower()
    label = str(value or "").strip()
    prefix = f"{side_name}__" if side_name else ""
    return label[len(prefix) :] if prefix and label.startswith(prefix) else label


def _resolve_policy_path(policy_path: Path, value: object) -> Path:
    path = Path(str(value or "").strip())
    if not path:
        raise ValueError("policy has no reference_candidates_path")
    return path if path.is_absolute() else policy_path.parent / path


def _normalise_reference(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True, errors="coerce")
    output["symbol"] = output["symbol"].astype(str)
    output["side_name"] = output["side_name"].astype(str).str.lower()
    output["policy_archetype"] = [
        _canonical_archetype(side, label)
        for side, label in zip(
            output["side_name"].to_numpy(copy=False),
            output["policy_archetype"].to_numpy(copy=False),
        )
    ]
    if output[list(_KEY_COLUMNS)].isna().any().any():
        raise ValueError("reference contains invalid diagnostic join keys")
    return output


def _source_diagnostics(source_paths: list[Path]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    sources: list[pd.DataFrame] = []
    required = set(_SOURCE_COLUMNS[:4])
    for source_path in source_paths:
        available = set(pq.ParquetFile(source_path).schema.names)
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                f"source is missing diagnostic join keys ({source_path}): {missing}"
            )
        columns = [column for column in _SOURCE_COLUMNS if column in available]
        source = pd.read_parquet(source_path, columns=columns).rename(
            columns={
                "__ts__": "timestamp",
                "__symbol__": "symbol",
                "archetype_policy_key": "policy_archetype",
            }
        )
        sources.append(_normalise_reference(source))
    if not sources:
        raise ValueError("at least one matrix source is required")
    source = pd.concat(sources, ignore_index=True, copy=False)
    source = source.drop_duplicates(list(_KEY_COLUMNS), keep="last")
    return source


def _label_path_diagnostics(
    label_root: Path,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, list[Path]]:
    """Load exact first-touch path labels for the reference months only.

    ``__first_touch_mae_to_sl__`` is the adverse excursion actually experienced
    before the first-touch exit, in units of the entry stop. It is therefore
    the correct path diagnostic for successful executable trades; full-path
    MAE could include movement that happened after the exit.
    """

    import pyarrow.parquet as pq

    label_root = label_root.resolve()
    if not label_root.is_dir():
        raise FileNotFoundError(label_root)
    timestamps = pd.to_datetime(reference["timestamp"], utc=True, errors="coerce")
    months = sorted({stamp.strftime("%Y_%m") for stamp in timestamps.dropna()})
    sides = sorted(
        {
            str(side).strip().lower()
            for side in reference["side_name"].dropna().to_numpy(copy=False)
            if str(side).strip().lower() in {"long", "short"}
        }
    )
    paths: list[Path] = []
    frames: list[pd.DataFrame] = []
    required = set(_LABEL_SOURCE_COLUMNS)
    for month in months:
        for side in sides:
            path = label_root / f"train_global_{side}_5_{month}.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
            available = set(pq.ParquetFile(path).schema.names)
            missing = sorted(required - available)
            if missing:
                raise ValueError(
                    f"label source is missing path-diagnostic columns ({path}): {missing}"
                )
            labels = pd.read_parquet(path, columns=list(_LABEL_SOURCE_COLUMNS)).rename(
                columns={
                    "__ts__": "timestamp",
                    "__symbol__": "symbol",
                    "__first_touch_mae_to_sl__": _LABEL_MAE_COLUMN,
                    "__first_touch_hit__": "first_touch_tp_hit",
                    "__first_touch_stop__": "first_touch_stop",
                    "__first_touch_timeout__": "first_touch_timeout",
                }
            )
            labels["timestamp"] = pd.to_datetime(
                labels["timestamp"], utc=True, errors="coerce"
            )
            labels["symbol"] = labels["symbol"].astype(str)
            labels["side_name"] = labels["side_name"].astype(str).str.lower()
            frames.append(labels)
            paths.append(path)
    if not frames:
        raise ValueError("could not derive label sources for the admission reference")
    labels = pd.concat(frames, ignore_index=True, copy=False)
    if labels[list(_LABEL_KEY_COLUMNS)].isna().any().any():
        raise ValueError("label source contains invalid path-diagnostic join keys")
    duplicates = labels.duplicated(list(_LABEL_KEY_COLUMNS), keep=False)
    if duplicates.any():
        raise ValueError("label source has duplicate timestamp/symbol/side path rows")
    labels["__label_path_row_present__"] = True
    return labels, paths


def _refresh_hash_contracts(
    artifact_root: Path,
    policy_path: Path,
    *,
    policy_aliases: tuple[Path, ...] = (),
) -> list[Path]:
    """Refresh only the policy hash records that live validation consumes."""

    updated: list[Path] = []
    policy_hash = _sha256(policy_path)
    candidates = (
        artifact_root / "policy_params" / "training_live_parity_contract.json",
        artifact_root / "simple_policy_optimiser" / "training_live_parity_contract.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        payload = _read_json(path)
        hashes = payload.get("artifact_hashes")
        record = hashes.get("threshold_basis_policy") if isinstance(hashes, dict) else None
        if not isinstance(record, dict):
            continue
        _backup_once(path)
        record.update(
            {
                "artifact_type": "file",
                "exists": True,
                "path": str(policy_path),
                "sha256": policy_hash,
            }
        )
        _write_json_atomic(path, payload)
        updated.append(path)

    side_manifest = artifact_root / "policy_params" / "side_archetype_expected_ev_policy_manifest.json"
    if side_manifest.is_file():
        payload = _read_json(side_manifest)
        _backup_once(side_manifest)
        payload["policy_sha256"] = policy_hash
        payload["email_archetype_baseline_enriched_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        _write_json_atomic(side_manifest, payload)
        updated.append(side_manifest)

    promoted_manifest = artifact_root / "policy_params" / "promoted_policy_manifest.json"
    if promoted_manifest.is_file():
        payload = _read_json(promoted_manifest)
        hashes = payload.get("file_sha256")
        if isinstance(hashes, dict):
            _backup_once(promoted_manifest)
            hashes[policy_path.relative_to(artifact_root).as_posix()] = policy_hash
            for alias in policy_aliases:
                if alias.is_file():
                    hashes[alias.relative_to(artifact_root).as_posix()] = _sha256(alias)
            _write_json_atomic(promoted_manifest, payload)
            updated.append(promoted_manifest)
    return updated


def materialize(
    *,
    policy_path: Path,
    matrix_sources: list[Path],
    output_name: str,
    label_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    policy_path = policy_path.resolve()
    matrix_sources = [path.resolve() for path in matrix_sources]
    if not policy_path.is_file() or not matrix_sources or not all(
        path.is_file() for path in matrix_sources
    ):
        raise FileNotFoundError((policy_path, matrix_sources))
    policy = _read_json(policy_path)
    reference_path = _resolve_policy_path(policy_path, policy.get("reference_candidates_path"))
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)

    reference = _normalise_reference(pd.read_parquet(reference_path))
    source = _source_diagnostics(matrix_sources)
    existing_columns = list(reference.columns)
    diagnostics = [column for column in _DIAGNOSTIC_COLUMNS if column in source.columns]
    if not diagnostics:
        raise ValueError("source has no requested email diagnostic columns")
    merged = reference.merge(
        source[list(_KEY_COLUMNS) + diagnostics],
        on=list(_KEY_COLUMNS),
        how="left",
        validate="one_to_one",
        sort=False,
        suffixes=("", "__source"),
    )
    for column in diagnostics:
        source_column = f"{column}__source"
        if source_column not in merged:
            continue
        merged[column] = merged[source_column]
        merged.drop(columns=[source_column], inplace=True)
    label_paths: list[Path] = []
    label_match_rate: float | None = None
    if label_root is not None:
        labels, label_paths = _label_path_diagnostics(label_root, reference)
        merged = merged.merge(
            labels[
                list(_LABEL_KEY_COLUMNS)
                + [
                    _LABEL_MAE_COLUMN,
                    *_LABEL_PATH_COLUMNS,
                    "__label_path_row_present__",
                ]
            ],
            on=list(_LABEL_KEY_COLUMNS),
            how="left",
            validate="many_to_one",
            sort=False,
            suffixes=("", "__label"),
        )
        for column in (_LABEL_MAE_COLUMN, *_LABEL_PATH_COLUMNS):
            label_column = f"{column}__label"
            if label_column in merged:
                merged[column] = merged[label_column]
                merged.drop(columns=[label_column], inplace=True)
        label_match_rate = float(merged["__label_path_row_present__"].fillna(False).mean())
        if label_match_rate < 0.995:
            raise ValueError(
                "path-label join coverage is too low: "
                f"{label_match_rate:.2%} matched"
            )
        label_mae_value_availability = float(merged[_LABEL_MAE_COLUMN].notna().mean())
        merged.drop(columns=["__label_path_row_present__"], inplace=True)
    else:
        label_mae_value_availability = None
    missing_fraction = float(merged[diagnostics].isna().all(axis=1).mean())
    if missing_fraction > 0.005:
        raise ValueError(
            "diagnostic join coverage is too low: "
            f"{(1.0 - missing_fraction):.2%} matched"
        )
    if len(merged) != len(reference) or not merged[existing_columns].equals(reference):
        raise AssertionError("email enrichment modified an admission reference column")

    output_path = reference_path.with_name(output_name)
    next_policy = dict(policy)
    next_policy.update(
        {
            "reference_candidates_path": output_path.name,
            "reference_columns": list(merged.columns),
            "email_archetype_baseline_window_days": 28,
            "email_archetype_baseline_min_rows": 40,
            "email_archetype_baseline_contract": (
                "fixed_28d_resolved_outcomes; side_x_archetype with side/global "
                "fallback; same median/IQR daily residual trim as admission"
            ),
            "email_archetype_baseline_reference_path": output_path.name,
            "email_archetype_baseline_reference_sources": [
                str(path) for path in matrix_sources
            ],
            "email_archetype_baseline_diagnostics": [
                *diagnostics,
                *(
                    [_LABEL_MAE_COLUMN, *_LABEL_PATH_COLUMNS]
                    if label_root is not None
                    else []
                ),
            ],
            "email_archetype_baseline_successful_trade_mae_contract": (
                "first_touch_mae_to_sl on ev_after_1pct > 0; fraction of initial stop"
            )
            if label_root is not None
            else policy.get("email_archetype_baseline_successful_trade_mae_contract"),
            "email_archetype_baseline_label_sources": [str(path) for path in label_paths],
        }
    )
    result = {
        "policy_path": str(policy_path),
        "reference_path": str(reference_path),
        "matrix_sources": [str(path) for path in matrix_sources],
        "output_path": str(output_path),
        "rows": int(len(merged)),
        "diagnostics": diagnostics,
        "join_match_rate": 1.0 - missing_fraction,
        "label_mae_join_match_rate": label_match_rate,
        "label_mae_value_availability": label_mae_value_availability,
        "label_sources": [str(path) for path in label_paths],
        "selection_columns_preserved": True,
        "dry_run": bool(dry_run),
    }
    if dry_run:
        return result

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    merged.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(output_path)
    _backup_once(policy_path)
    _write_json_atomic(policy_path, next_policy)
    policy_aliases: tuple[Path, ...] = ()
    canonical_alias = policy_path.parent / "threshold_basis_policy.json"
    if policy_path != canonical_alias:
        _backup_once(canonical_alias)
        _write_json_atomic(canonical_alias, next_policy)
        policy_aliases = (canonical_alias,)
    updated = _refresh_hash_contracts(
        policy_path.parents[1],
        policy_path,
        policy_aliases=policy_aliases,
    )
    result.update(
        {
            "policy_sha256": _sha256(policy_path),
            "reference_sha256": _sha256(output_path),
            "updated_hash_contracts": [str(path) for path in updated],
            "synchronized_policy_aliases": [str(path) for path in policy_aliases],
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument(
        "--matrix-source",
        type=Path,
        action="append",
        required=True,
        help="Exact scored source parquet. Repeat this flag for split OOS lineage.",
    )
    parser.add_argument(
        "--output-name",
        default="threshold_basis_reference_sidearch_ev21d_email_diagnostics_v1.parquet",
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        help=(
            "Exact label artifact directory. Adds first-touch MAE relative to the "
            "initial stop for successful-trade email diagnostics."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                policy_path=args.policy_path,
                matrix_sources=list(args.matrix_source),
                output_name=str(args.output_name),
                label_root=args.label_root,
                dry_run=bool(args.dry_run),
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
