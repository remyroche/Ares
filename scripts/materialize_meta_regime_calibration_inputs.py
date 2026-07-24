#!/usr/bin/env python3
"""Attach frozen regime-calibration inputs to a compact meta comparison frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from extreme_price_movements.regime_ev_calibration import (
    default_regime_ev_calibration_artifact,
    default_regime_ev_feature_handoff,
    load_regime_ev_calibration,
    required_feature_columns,
)
from scripts.run_s52_train_meta_regime_handoff_smoke import (
    SUPPORT_DRIFT_COLUMNS,
    _add_fold_support_drift_features,
)

KEYS = ("__ts__", "__symbol__", "side_name", "archetype_policy_key")


def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def materialize(
    *,
    compact_path: Path,
    feature_handoff_path: Path,
    calibration_path: Path,
    output_path: Path,
) -> dict[str, object]:
    artifact = load_regime_ev_calibration(calibration_path)
    required = required_feature_columns(artifact)
    available = set(pq.ParquetFile(feature_handoff_path).schema_arrow.names)
    missing_source = [
        name for name in required if name not in available and name != "support_mean_frequency"
    ]
    if missing_source:
        raise ValueError(f"Calibration source is missing required columns: {missing_source}")

    support_sources = [name for name in SUPPORT_DRIFT_COLUMNS if name in available]
    read_columns = list(
        dict.fromkeys(
            [
                *KEYS,
                *[name for name in required if name in available],
                *support_sources,
            ]
        )
    )
    source = pd.read_parquet(feature_handoff_path, columns=read_columns)
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True, errors="coerce")
    source["side_name"] = source["side_name"].astype(str).str.lower()
    source["archetype_policy_key"] = source["archetype_policy_key"].astype(str)
    source = source.sort_values(list(KEYS), kind="stable").drop_duplicates(
        list(KEYS), keep="last"
    )

    windows = {
        (
            str(effect.get("train_start")),
            str(effect.get("train_end")),
            str(effect.get("valid_from")),
            str(effect.get("valid_to")),
        )
        for effect in artifact.get("effects", [])
        if effect.get("valid_from") and effect.get("valid_to")
    }
    support_parts: list[pd.DataFrame] = []
    for train_start, train_end, valid_start, valid_end in sorted(windows):
        train_mask = source["__ts__"].ge(pd.Timestamp(train_start)) & source[
            "__ts__"
        ].lt(pd.Timestamp(train_end))
        valid_mask = source["__ts__"].ge(pd.Timestamp(valid_start)) & source[
            "__ts__"
        ].lt(pd.Timestamp(valid_end))
        train = source.loc[train_mask, [*KEYS, *support_sources]]
        valid = source.loc[valid_mask, [*KEYS, *support_sources]]
        if train.empty or valid.empty:
            continue
        _, enriched = _add_fold_support_drift_features(train, valid)
        support_parts.append(enriched[[*KEYS, "support_mean_frequency"]])
    support = (
        pd.concat(support_parts, ignore_index=True)
        .sort_values(list(KEYS), kind="stable")
        .drop_duplicates(list(KEYS), keep="last")
        if support_parts
        else pd.DataFrame(columns=[*KEYS, "support_mean_frequency"])
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    support_path = output_path.with_suffix(".support.parquet")
    source_path = output_path.with_suffix(".calibration_source.parquet")
    support.to_parquet(support_path, index=False, compression="zstd")
    source[[*KEYS, *[name for name in required if name in source.columns]]].to_parquet(
        source_path, index=False, compression="zstd"
    )

    projected = [name for name in required if name != "support_mean_frequency"]
    projections = ",\n".join(f"f.{_quote(name)}" for name in projected)
    if projections:
        projections = ",\n" + projections
    sql = f"""
    COPY (
      SELECT c.*{projections}, s.support_mean_frequency
      FROM read_parquet('{compact_path.as_posix()}') c
      LEFT JOIN read_parquet('{source_path.as_posix()}') f
        ON CAST(c.__ts__ AS TIMESTAMPTZ) = CAST(f.__ts__ AS TIMESTAMPTZ)
       AND c.__symbol__ = f.__symbol__
       AND lower(c.side_name) = lower(f.side_name)
       AND cast(c.archetype_policy_key AS VARCHAR) = cast(f.archetype_policy_key AS VARCHAR)
      LEFT JOIN read_parquet('{support_path.as_posix()}') s
        ON CAST(c.__ts__ AS TIMESTAMPTZ) = CAST(s.__ts__ AS TIMESTAMPTZ)
       AND c.__symbol__ = s.__symbol__
       AND lower(c.side_name) = lower(s.side_name)
       AND cast(c.archetype_policy_key AS VARCHAR) = cast(s.archetype_policy_key AS VARCHAR)
    ) TO '{output_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    connection = duckdb.connect()
    connection.execute(sql)
    coverage_expr = ", ".join(
        f"avg(CASE WHEN {_quote(name)} IS NOT NULL THEN 1.0 ELSE 0.0 END) AS {_quote(name)}"
        for name in required
    )
    coverage = connection.execute(
        f"SELECT count(*) AS rows, {coverage_expr} FROM read_parquet('{output_path.as_posix()}')"
    ).fetchdf()
    connection.close()
    support_path.unlink(missing_ok=True)
    source_path.unlink(missing_ok=True)
    return {
        "schema": "meta_regime_calibration_input_materialization_v1",
        "compact_path": str(compact_path),
        "feature_handoff_path": str(feature_handoff_path),
        "calibration_path": str(calibration_path),
        "output_path": str(output_path),
        "required_features": required,
        "support_windows": len(windows),
        "support_rows": int(len(support)),
        "coverage": coverage.iloc[0].to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-handoff",
        type=Path,
        default=default_regime_ev_feature_handoff(),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=default_regime_ev_calibration_artifact(),
    )
    args = parser.parse_args()
    if args.feature_handoff is None or args.calibration is None:
        raise FileNotFoundError("Default calibration artifacts are unavailable")
    manifest = materialize(
        compact_path=args.compact,
        feature_handoff_path=args.feature_handoff,
        calibration_path=args.calibration,
        output_path=args.output,
    )
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, default=str), flush=True)


if __name__ == "__main__":
    main()
