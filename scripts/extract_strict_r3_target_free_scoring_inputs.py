#!/usr/bin/env python3
"""Create immutable target-free scoring inputs from an existing feature source.

Candidate identities always come from the schema-v2 target-free grid.  A
prequential ledger may be used as a *feature* source, but no label, outcome,
or prequential prediction column is allowed to leave this utility.  This is
useful when a newly opened forward period needs the preceding 42-day reference
without rematerialising an otherwise validated feature surface.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_fields(contract_path: Path, side: str) -> list[str]:
    payload = json.loads(contract_path.read_text())
    fields = payload.get("base_fields_by_side", {}).get(side, payload.get("base_fields", []))
    fields = [str(field) for field in fields]
    if len(fields) != 120 or len(set(fields)) != 120:
        raise ValueError("strict-R3 target-free scoring requires 120 unique base fields")
    return fields


def _read_parquet_window(
    path: Path,
    *,
    columns: list[str],
    timestamp_column: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    schema_names = set(dataset.schema.names)
    missing = sorted(set(columns).difference(schema_names))
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    timestamp_type = dataset.schema.field(timestamp_column).type
    if pa.types.is_timestamp(timestamp_type) and timestamp_type.tz:
        lower = pa.scalar(start.to_pydatetime(), type=timestamp_type)
        upper = pa.scalar(end.to_pydatetime(), type=timestamp_type)
    else:
        # Arrow cannot compare a UTC-aware Python timestamp against a naïve
        # parquet timestamp.  Stored naïve times in this legacy surface are
        # UTC by contract, so remove only the representation timezone for the
        # predicate and restore UTC immediately after reading.
        lower = pa.scalar(start.tz_localize(None).to_datetime64(), type=timestamp_type)
        upper = pa.scalar(end.tz_localize(None).to_datetime64(), type=timestamp_type)
    table = dataset.to_table(
        columns=columns,
        filter=(
            (ds.field(timestamp_column) >= lower)
            & (ds.field(timestamp_column) < upper)
        ),
    )
    return table.to_pandas()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--feature-source", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end-exclusive", required=True)
    parser.add_argument("--side", choices=("long", "short"), required=True)
    parser.add_argument(
        "--feature-identity", choices=("candidate_id", "signal_keys"), required=True,
        help="Whether the feature source is keyed by candidate ID or __ts__/symbol.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    start = pd.to_datetime(args.start, utc=True)
    end = pd.to_datetime(args.end_exclusive, utc=True)
    if end <= start:
        raise ValueError("end-exclusive must be after start")
    fields = _base_fields(args.feature_contract, args.side)

    candidate_columns = ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"]
    candidates = _read_parquet_window(
        args.candidate_source,
        columns=candidate_columns,
        timestamp_column="__decision_ts__",
        start=start,
        end=end,
    )
    candidates["__decision_ts__"] = pd.to_datetime(candidates["__decision_ts__"], utc=True)
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True)
    candidates = candidates.loc[candidates["side_name"].astype(str).str.lower().eq(args.side)].copy()
    if candidates.empty or candidates["candidate_id"].duplicated().any():
        raise ValueError("target-free candidate grid is empty or has duplicate identities")

    if args.feature_identity == "candidate_id":
        source_columns = ["candidate_id", "__decision_ts__", "side_name", *fields]
        source = _read_parquet_window(
            args.feature_source,
            columns=source_columns,
            timestamp_column="__decision_ts__",
            start=start,
            end=end,
        )
        source["__decision_ts__"] = pd.to_datetime(source["__decision_ts__"], utc=True)
        source = source.loc[source["side_name"].astype(str).str.lower().eq(args.side)].copy()
        if source["candidate_id"].duplicated().any():
            raise ValueError("feature source has duplicate candidate identities")
        features = candidates.loc[:, ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"]].merge(
            source.loc[:, ["candidate_id", *fields]], on="candidate_id", how="left", validate="one_to_one",
        )
    else:
        source_columns = ["__ts__", "__symbol__", *fields]
        source = _read_parquet_window(
            args.feature_source,
            columns=source_columns,
            timestamp_column="__ts__",
            start=start - pd.Timedelta(hours=1),
            end=end - pd.Timedelta(hours=1),
        )
        source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True)
        if source.duplicated(["__ts__", "__symbol__"]).any():
            raise ValueError("signal-key feature source has duplicate identities")
        features = candidates.loc[:, ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"]].merge(
            source, on=["__ts__", "__symbol__"], how="left", validate="one_to_one",
        )

    if len(features) != len(candidates) or features["candidate_id"].duplicated().any():
        raise ValueError("feature extraction changed target-free candidate identities")
    prohibited = [
        column for column in features.columns
        if any(token in column.lower() for token in ("label", "outcome", "policy_net", "gross_bps", "h12_", "target_invalid", "path_valid"))
    ]
    if prohibited:
        raise ValueError(f"target-free output contains prohibited columns: {prohibited}")
    args.out_dir.mkdir(parents=True)
    candidates.to_parquet(args.out_dir / "candidates.parquet", index=False, compression="zstd")
    features.to_parquet(args.out_dir / "features.parquet", index=False, compression="zstd")
    complete = features.loc[:, fields].notna().all(axis=1)
    coverage = pd.DataFrame({
        "feature": fields,
        "finite_fraction": [float(features[column].notna().mean()) for column in fields],
        "n_unique": [int(features[column].nunique(dropna=True)) for column in fields],
    })
    coverage.to_parquet(args.out_dir / "feature_coverage.parquet", index=False)
    manifest = {
        "schema": "strict_r3_target_free_scoring_input_v1",
        "candidate_source": str(args.candidate_source),
        "candidate_source_sha256": _sha(args.candidate_source),
        "feature_source": str(args.feature_source),
        "feature_source_sha256": _sha(args.feature_source),
        "feature_identity": args.feature_identity,
        "feature_contract": str(args.feature_contract),
        "feature_contract_sha256": _sha(args.feature_contract),
        "start": start.isoformat(), "end_exclusive": end.isoformat(),
        "side": args.side, "candidate_rows": int(len(candidates)),
        "complete_contract_rows": int(complete.sum()),
        "complete_contract_fraction": float(complete.mean()),
        "outcome_columns_consumed": [],
        "candidate_identity_source": "point_in_time_target_free_grid",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **manifest}))


if __name__ == "__main__":
    main()
