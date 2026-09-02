#!/usr/bin/env python3
"""Append deterministic session/calendar fields to a target-free feature panel.

The source panel is never overwritten. The materialised result and a compact
receipt are written to a fresh output directory so it can be used as an input
to a later feature-selection/retraining run without changing a sealed bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features import (
    SESSION_CALENDAR_FEATURE_KEYS,
    session_calendar_features,
)


TIMESTAMP_CANDIDATES = ("__decision_ts__", "__ts__", "decision_ts", "timestamp")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in frame.columns:
            raise KeyError(f"timestamp column absent: {requested}")
        return requested
    for name in TIMESTAMP_CANDIDATES:
        if name in frame.columns:
            return name
    raise KeyError(f"no timestamp column found; tried {TIMESTAMP_CANDIDATES}")


def materialize(
    source: Path,
    out_dir: Path,
    *,
    timestamp_column: str | None = None,
) -> dict[str, object]:
    source = Path(source)
    out_dir = Path(out_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    if out_dir.exists():
        raise FileExistsError(f"immutable output directory already exists: {out_dir}")

    frame = pd.read_parquet(source)
    column = _timestamp_column(frame, timestamp_column)
    timestamps = pd.to_datetime(frame[column], utc=True, errors="raise")
    generated = session_calendar_features(pd.DatetimeIndex(timestamps))
    for name, values in generated.items():
        if name in frame.columns:
            raise ValueError(f"source already contains {name}; refusing ambiguous overwrite")
        frame[name] = values

    out_dir.mkdir(parents=True, exist_ok=False)
    output = out_dir / "features_with_session_calendar.parquet"
    frame.to_parquet(output, index=False)
    coverage = {
        name: float(frame[name].notna().mean()) for name in SESSION_CALENDAR_FEATURE_KEYS
    }
    receipt: dict[str, object] = {
        "schema": "strict_r3_session_calendar_feature_materialization_v1",
        "source": str(source),
        "source_sha256": _sha256(source),
        "output": str(output),
        "output_sha256": _sha256(output),
        "rows": int(len(frame)),
        "timestamp_column": column,
        "timestamp_min": timestamps.min().isoformat(),
        "timestamp_max": timestamps.max().isoformat(),
        "feature_keys": list(SESSION_CALENDAR_FEATURE_KEYS),
        "coverage": coverage,
        "causality": "timestamp-only; no labels, outcomes, future paths, or candidate-universe inputs",
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--timestamp-column", default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                args.source,
                args.out_dir,
                timestamp_column=args.timestamp_column,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
