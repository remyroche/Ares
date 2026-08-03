#!/usr/bin/env python3
"""Materialize a provenance-bound causal multi-view regime feature panel.

The input may be an hourly parquet or an artifact root containing
``hourly_state_calendar.parquet`` (preferred) or ``hourly_transition_dataset``.
Only existing, observable numeric fields are used.  Missing liquidity inputs
remain missing from the output family; this runner never creates synthetic
spread, depth or volume fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.regime_multiview import (  # noqa: E402
    FORBIDDEN_INPUT_TOKENS,
    MultiViewRegimeConfig,
    build_causal_multiview_regime_features,
)


DEFAULT_INPUT = ROOT / "data_perp/artifacts/regime_episode_ledger_2022_2026_20260730_v1"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/regime_multiview_panel_2022_2026_20260730_v1"
OUTPUT_FEATURES = "multiview_regime_features.parquet"
OUTPUT_MANIFEST = "manifest.json"
OUTPUT_SIGNATURE = "manifest.sha256"
INPUT_FILENAMES = ("hourly_state_calendar.parquet", "hourly_transition_dataset.parquet")
IDENTITY_EXCLUDED_PREFIXES = ("target__", "expost__", "state_context__", "source_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (Path, pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def resolve_input(path: Path) -> tuple[Path, Path | None]:
    """Resolve a direct parquet or a supported artifact-root parquet."""

    path = Path(path)
    if path.is_file():
        if path.suffix != ".parquet":
            raise ValueError(f"input must be parquet: {path}")
        return path, path.parent if (path.parent / "manifest.json").exists() else None
    if not path.is_dir():
        raise FileNotFoundError(path)
    for name in INPUT_FILENAMES:
        candidate = path / name
        if candidate.exists():
            return candidate, path
    raise FileNotFoundError(
        f"artifact root lacks one of {list(INPUT_FILENAMES)}: {path}"
    )


def _verify_source_manifest(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {"manifest_present": False, "detached_checksum_verified": False}
    manifest = root / "manifest.json"
    signature = root / "manifest.sha256"
    if not manifest.exists():
        return {"manifest_present": False, "detached_checksum_verified": False}
    verified = False
    if signature.exists():
        fields = signature.read_text(encoding="utf-8").strip().split()
        if not fields or fields[0] != sha256(manifest):
            raise ValueError(f"source manifest detached checksum fails: {manifest}")
        verified = True
    return {
        "manifest_present": True,
        "manifest_sha256": sha256(manifest),
        "detached_checksum_verified": verified,
    }


def _is_forbidden(name: str) -> bool:
    lower = str(name).lower()
    return any(token in lower for token in FORBIDDEN_INPUT_TOKENS)


def observable_numeric_columns(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    group_columns: Iterable[str],
    requested: Sequence[str] | None = None,
) -> list[str]:
    """Select only source-observable numeric input fields.

    Explicit requests fail rather than silently dropping a target or post-entry
    field.  Automatic selection excludes identities, targets and state labels.
    """

    groups = set(group_columns)
    if requested:
        columns = list(dict.fromkeys(str(column) for column in requested))
        missing = [column for column in columns if column not in frame]
        if missing:
            raise KeyError(f"requested feature columns missing: {missing[:8]}")
        forbidden = [column for column in columns if _is_forbidden(column)]
        if forbidden:
            raise ValueError(f"requested feature columns are forbidden: {forbidden[:8]}")
    else:
        columns = [
            str(column)
            for column in frame.columns
            if column != timestamp_col
            and column not in groups
            and pd.api.types.is_numeric_dtype(frame[column])
            and not str(column).startswith(IDENTITY_EXCLUDED_PREFIXES)
            and not _is_forbidden(str(column))
        ]
    non_numeric = [column for column in columns if not pd.api.types.is_numeric_dtype(frame[column])]
    if non_numeric:
        raise TypeError(f"requested feature columns must be numeric: {non_numeric[:8]}")
    if not columns:
        raise ValueError("no observable numeric source fields remain after causal denylist")
    return columns


def default_group_columns(frame: pd.DataFrame) -> list[str]:
    for name in ("calendar_segment_id", "segment_id", "source_segment_id"):
        if name in frame:
            return [name]
    return []


def _feature_families(columns: Sequence[str]) -> dict[str, int]:
    return {
        "distribution_dynamics": int(
            sum(name.startswith("mv__") and not name.startswith(("mv__dependence__", "mv__liquidity__")) for name in columns)
        ),
        "liquidity_proxy": int(sum(name.startswith("mv__liquidity__") for name in columns)),
        "dependence_covariance": int(sum(name.startswith("mv__dependence__") for name in columns)),
        "volatility": int(sum("__realized_vol_" in name or "__vol_of_vol_" in name for name in columns)),
    }


def materialize_multiview_regime_panel(
    *,
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
    timestamp_col: str = "source_utc",
    group_columns: Sequence[str] | None = None,
    feature_columns: Sequence[str] | None = None,
    dependence_columns: Sequence[str] | None = None,
    max_dependence_columns: int = 12,
    enrichment_path: Path | None = None,
) -> dict[str, Any]:
    """Materialize immutable feature parquet plus a detached signed manifest."""

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"immutable output exists: {output_dir}")
    parquet, root = resolve_input(Path(input_path))
    source_manifest = _verify_source_manifest(root)
    frame = pd.read_parquet(parquet)
    if timestamp_col not in frame:
        raise KeyError(f"source panel lacks timestamp column: {timestamp_col}")
    enrichment_contract: dict[str, Any] | None = None
    if enrichment_path is not None:
        enrichment_file, enrichment_root = resolve_input(Path(enrichment_path))
        enrichment = pd.read_parquet(enrichment_file)
        if timestamp_col not in enrichment:
            raise KeyError(f"enrichment lacks timestamp column: {timestamp_col}")
        if enrichment[timestamp_col].duplicated().any():
            raise ValueError("enrichment timestamp identity is not unique")
        original_rows = len(frame)
        overlap = [
            column
            for column in enrichment.columns
            if column in frame.columns and column != timestamp_col
        ]
        enrichment = enrichment.drop(columns=overlap)
        frame = frame.merge(
            enrichment,
            on=timestamp_col,
            how="left",
            validate="one_to_one",
            sort=False,
        )
        if len(frame) != original_rows:
            raise AssertionError("enrichment changed source row identity")
        enrichment_contract = {
            "source": str(enrichment_file),
            "source_sha256": sha256(enrichment_file),
            "source_manifest": _verify_source_manifest(enrichment_root),
            "join": "exact one-to-one source_utc; no asof or fill",
            "overlapping_nonidentity_fields_dropped": overlap,
            "added_fields": [
                column
                for column in enrichment.columns
                if column != timestamp_col
            ],
        }
    groups = list(group_columns) if group_columns is not None else default_group_columns(frame)
    missing_groups = [column for column in groups if column not in frame]
    if missing_groups:
        raise KeyError(f"source panel lacks requested group columns: {missing_groups}")
    fields = observable_numeric_columns(
        frame,
        timestamp_col=timestamp_col,
        group_columns=groups,
        requested=feature_columns,
    )
    dependency = list(dependence_columns) if dependence_columns is not None else fields
    if any(column not in fields for column in dependency):
        raise ValueError("dependence columns must be selected observable feature columns")
    work = frame.loc[:, [*groups, timestamp_col, *fields]].copy()
    features, metadata = build_causal_multiview_regime_features(
        work,
        config=MultiViewRegimeConfig(
            timestamp_col=timestamp_col,
            group_columns=tuple(groups),
            feature_columns=tuple(fields),
            dependence_columns=tuple(dependency),
            max_dependence_columns=int(max_dependence_columns),
        ),
    )
    identity = frame.loc[:, [timestamp_col, *groups]].copy()
    identity[timestamp_col] = pd.to_datetime(identity[timestamp_col], utc=True, errors="raise")
    materialized = pd.concat([identity, features], axis=1)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    try:
        feature_path = temporary / OUTPUT_FEATURES
        materialized.to_parquet(feature_path, index=False, compression="zstd")
        output_columns = list(features.columns)
        report = {
            "schema": "materialized_multiview_regime_panel_v1",
            "research_only": True,
            "promotion_evidence": False,
            "input_contract": {
                "source": str(parquet),
                "source_sha256": sha256(parquet),
                "timestamp_col": timestamp_col,
                "group_columns": groups,
                "causal_denylist": list(FORBIDDEN_INPUT_TOKENS),
                "source_manifest": source_manifest,
                "enrichment": enrichment_contract,
                "gap_contract": "exact-cadence segments are built within group columns; missing or misaligned bars begin a new segment and no transform bridges it",
            },
            "multiview_contract": metadata,
            "families": _feature_families(output_columns),
            "counts": {
                "input_rows": int(len(frame)),
                "source_observable_numeric_fields": int(len(fields)),
                "output_feature_fields": int(len(output_columns)),
                "output_rows": int(len(materialized)),
            },
            "output": {
                "features": {
                    "path": OUTPUT_FEATURES,
                    "sha256": sha256(feature_path),
                    "identity_columns": [timestamp_col, *groups],
                }
            },
        }
        manifest_path = temporary / OUTPUT_MANIFEST
        manifest_path.write_text(
            json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / OUTPUT_SIGNATURE).write_text(
            f"{sha256(manifest_path)}  {OUTPUT_MANIFEST}\n", encoding="utf-8"
        )
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="hourly parquet or ledger/transition artifact root")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timestamp-col", default="source_utc")
    parser.add_argument("--group-column", action="append", dest="group_columns")
    parser.add_argument("--feature-column", action="append", dest="feature_columns")
    parser.add_argument("--dependence-column", action="append", dest="dependence_columns")
    parser.add_argument("--max-dependence-columns", type=int, default=12)
    parser.add_argument(
        "--enrichment",
        type=Path,
        help="optional signed hourly parquet/artifact joined by exact timestamp",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = materialize_multiview_regime_panel(
        input_path=args.input,
        output_dir=args.output_dir,
        timestamp_col=args.timestamp_col,
        group_columns=args.group_columns,
        feature_columns=args.feature_columns,
        dependence_columns=args.dependence_columns,
        max_dependence_columns=args.max_dependence_columns,
        enrichment_path=args.enrichment,
    )
    print(json.dumps(_json_safe(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
