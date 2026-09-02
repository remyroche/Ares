#!/usr/bin/env python3
"""Materialize a no-training strict-OOF calendar for canonical 31/8 base models.

The only input rows read from the archived label ledgers are identity, side,
signal time, decision time and the selected 31/8 pre-entry features.  Outcome
fields are intentionally not read, so this cannot use evaluation outcomes to
choose historical event windows or training samples.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.canonical_318_historical_calendar import (
    HistoricalOOFCalendarSpec,
    IDENTITY,
    build_historical_base_oof_calendar,
    canonical_json_sha256,
)


DEFAULT_LABELS_ROOT = Path(
    "data_perp/artifacts/20260720_s59_h5_signalclose_causal_trailing_cost100bps_labels_v2/labels"
)
DEFAULT_LONG_MANIFEST = Path(
    "data_perp/artifacts/packb_side_local_outer_oof_july20_20260726_v1_31_8/long/manifest.json"
)
DEFAULT_SHORT_MANIFEST = Path(
    "data_perp/artifacts/packb_side_local_outer_oof_july20_20260726_v1_31_8/short/manifest.json"
)
DEFAULT_OUTPUT = Path(
    "data_perp/artifacts/canonical_31_8_historical_reconstruction_calendar_20260727_v1"
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    return value.item() if hasattr(value, "item") else value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _month_range(start: str, end_exclusive: str) -> list[pd.Period]:
    first = pd.Period(start, freq="M")
    stop = pd.Period(end_exclusive, freq="M")
    if first >= stop:
        raise ValueError("source history start must precede score end")
    return list(pd.period_range(first, stop - 1, freq="M"))


def _side_features(path: Path, *, expected: int) -> list[str]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    features = [str(value) for value in manifest.get("features", [])]
    if len(features) != expected or len(set(features)) != expected:
        raise ValueError(f"{path} is not a canonical {expected}-feature side contract")
    return features


def _source_paths(labels_root: Path, *, start: str, end_exclusive: str) -> list[tuple[str, Path]]:
    paths: list[tuple[str, Path]] = []
    for month in _month_range(start, end_exclusive):
        for side in ("long", "short"):
            path = labels_root / f"train_global_{side}_5_{month.year}_{month.month:02d}.parquet"
            if not path.exists():
                raise FileNotFoundError(path)
            paths.append((side, path))
    return paths


def load_feature_complete_identities(
    labels_root: Path,
    *,
    source_start: str,
    score_end_exclusive: str,
    long_features: list[str],
    short_features: list[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Load no outcome fields and retain only observable 31/8-complete rows."""

    parts: list[pd.DataFrame] = []
    source_report: list[dict[str, Any]] = []
    for side, path in _source_paths(labels_root, start=source_start, end_exclusive=score_end_exclusive):
        features = long_features if side == "long" else short_features
        schema = pq.ParquetFile(path).schema_arrow.names
        required = [*IDENTITY, "__decision_ts__", *features]
        missing = sorted(set(required).difference(schema))
        if missing:
            raise ValueError(f"{path} is missing canonical {side} features: {missing}")
        frame = pd.read_parquet(path, columns=required)
        if not frame["side_name"].astype(str).str.lower().eq(side).all():
            raise ValueError(f"{path} has rows outside its declared side")
        feature_complete = frame.loc[:, features].notna().all(axis=1)
        retained = frame.loc[feature_complete, list(IDENTITY) + ["__decision_ts__"]].copy()
        retained["__label_resolution_ts__"] = pd.to_datetime(
            retained["__decision_ts__"], utc=True, errors="raise"
        ) + pd.Timedelta(hours=24)
        parts.append(retained)
        source_report.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "side": side,
                "source_rows": int(len(frame)),
                "feature_complete_rows": int(len(retained)),
                "feature_complete_fraction": float(feature_complete.mean()),
                "features": features,
            }
        )
    result = pd.concat(parts, ignore_index=True)
    if result.duplicated(list(IDENTITY)).any():
        raise ValueError("source ledgers produced duplicate candidate identities")
    return result, source_report


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    long_features = _side_features(args.long_manifest, expected=31)
    short_features = _side_features(args.short_manifest, expected=8)
    identities, sources = load_feature_complete_identities(
        args.labels_root,
        source_start=args.source_start,
        score_end_exclusive=args.score_end,
        long_features=long_features,
        short_features=short_features,
    )
    spec = HistoricalOOFCalendarSpec(
        score_start=pd.Timestamp(args.score_start, tz="UTC"),
        score_end=pd.Timestamp(args.score_end, tz="UTC"),
        minimum_train_rows_per_side=int(args.minimum_train_rows_per_side),
        maximum_fit_rows_per_side=int(args.maximum_fit_rows_per_side),
    )
    frozen, train, contract = build_historical_base_oof_calendar(identities, spec=spec)
    args.output_dir.mkdir(parents=True)
    frozen_path = args.output_dir / "frozen_score_identities.parquet"
    train_path = args.output_dir / "deterministic_train_sample_identities.parquet"
    frozen.to_parquet(frozen_path, index=False)
    train.to_parquet(train_path, index=False)
    manifest = {
        "schema": "canonical_31_8_historical_reconstruction_calendar_artifact_v1",
        "research_status": "CONTRACT_ONLY_NO_MODEL_TRAINING",
        "features": {
            "long": long_features,
            "short": short_features,
            "long_manifest_sha256": _sha256(args.long_manifest),
            "short_manifest_sha256": _sha256(args.short_manifest),
        },
        "source_contract": {
            "read_columns": [*IDENTITY, "__decision_ts__", "canonical_side_features_only"],
            "not_read": ["targets", "returns", "transition labels", "economic failures", "event IDs"],
            "sources": sources,
        },
        "calendar_contract": contract,
        "outputs": {
            "frozen_score_identities": {"path": str(frozen_path), "sha256": _sha256(frozen_path)},
            "deterministic_train_sample_identities": {"path": str(train_path), "sha256": _sha256(train_path)},
        },
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    _write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", type=Path, default=DEFAULT_LABELS_ROOT)
    parser.add_argument("--long-manifest", type=Path, default=DEFAULT_LONG_MANIFEST)
    parser.add_argument("--short-manifest", type=Path, default=DEFAULT_SHORT_MANIFEST)
    parser.add_argument("--source-start", default="2025-01")
    parser.add_argument("--score-start", default="2025-02-01")
    parser.add_argument("--score-end", default="2025-05-01")
    parser.add_argument("--minimum-train-rows-per-side", type=int, default=50_000)
    parser.add_argument("--maximum-fit-rows-per-side", type=int, default=100_000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    manifest = run(_parser().parse_args())
    print(json.dumps(_safe(manifest["calendar_contract"]["calendar"]["folds"]), indent=2))


if __name__ == "__main__":
    main()
