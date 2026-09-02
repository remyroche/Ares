#!/usr/bin/env python3
"""Project historical target-free candidates and features onto short identities.

Candidate availability, spread, and decision-time entry executability are
side-neutral facts.  The strict-R3 scorer nevertheless keys every candidate by
side, so an existing long-only target-free grid cannot be fed to a short model
without a deliberate identity projection.  This utility performs only that
projection: it copies target-free columns unchanged, replaces ``side_name``
with ``short``, and recreates the canonical `symbol|short|UTC-signal` ID.

It is intentionally unsuitable for labels: it neither reads nor writes
outcomes, future paths, label validity, or target fields.  An optional feature
source must itself be target-free.  It is streamed record-batch by
record-batch so the full historical panel does not need to fit in memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = ("candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name")
FEATURE_IDENTITIES = ("__ts__", "__symbol__")
FORBIDDEN_FEATURE_COLUMNS = frozenset({
    "candidate_id",
    "side_name",
    "label_valid",
    "target_invalid",
    "policy_net_bps",
    "gross_bps",
    "net_bps",
    "exact_net_bps",
    "future_path_complete",
    "h12_complete",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _short_id(frame: pd.DataFrame) -> pd.Series:
    timestamp = _utc(frame["__ts__"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return frame["__symbol__"].astype(str) + "|short|" + timestamp


def _validate_feature_schema(schema: pa.Schema, *, source: Path) -> None:
    missing = sorted(set(FEATURE_IDENTITIES).difference(schema.names))
    if missing:
        raise ValueError(f"{source} lacks target-free feature identities: {missing}")
    forbidden = sorted(FORBIDDEN_FEATURE_COLUMNS.intersection(schema.names))
    if forbidden:
        raise ValueError(f"{source} is not target-free; forbidden columns: {forbidden}")
    if not pa.types.is_timestamp(schema.field("__ts__").type):
        raise ValueError(f"{source} __ts__ must be a timestamp")


def _project_feature_batch(batch: pa.RecordBatch) -> pa.Table:
    """Add side-specific identities without altering source feature values."""
    table = pa.Table.from_batches([batch])
    timestamps = table["__ts__"]
    timestamps_utc = pc.assume_timezone(
        timestamps,
        timezone="UTC",
        ambiguous="raise",
        nonexistent="raise",
    ) if timestamps.type.tz is None else timestamps
    hourly = pc.floor_temporal(timestamps_utc, multiple=1, unit="hour")
    if bool(pc.any(pc.invert(pc.equal(timestamps_utc, hourly))).as_py()):
        raise ValueError("feature timestamps must be exact completed hourly observations")
    timestamps_for_id = pc.cast(timestamps_utc, pa.timestamp("s", tz="UTC"))
    decision_ts = pc.add(timestamps_utc, pa.scalar(pd.Timedelta(hours=1).to_pytimedelta()))
    signal_text = pc.strftime(timestamps_for_id, format="%Y-%m-%dT%H:%M:%SZ")
    candidate_id = pc.binary_join_element_wise(
        pc.cast(table["__symbol__"], pa.string()),
        signal_text,
        pa.scalar("|short|"),
    )
    return table.append_column("__decision_ts__", decision_ts).append_column(
        "side_name", pa.array(["short"] * len(table), type=pa.string())
    ).append_column("candidate_id", candidate_id)


def _project_features(
    source: Path,
    *,
    destination: Path,
    candidate_symbols: frozenset[str],
    candidate_start: pd.Timestamp,
    candidate_end: pd.Timestamp,
    batch_size: int = 50_000,
) -> dict[str, object]:
    """Project target-free feature rows within the declared candidate universe.

    The feature source can contain instruments or times outside the frozen
    candidate universe.  Keeping them would break one-to-one identity joins
    and permit a later loader to score a row that was not a target-free
    candidate.  The restriction is solely on decision-time symbol/timestamp
    identity; it never examines labels or outcomes.
    """
    parquet = pq.ParquetFile(source)
    _validate_feature_schema(parquet.schema_arrow, source=source)
    if destination.exists():
        raise FileExistsError(f"immutable feature output exists: {destination}")

    writer: pq.ParquetWriter | None = None
    rows = 0
    min_ts: pd.Timestamp | None = None
    max_ts: pd.Timestamp | None = None
    try:
        allowed_symbols = pa.array(sorted(candidate_symbols), type=pa.string())
        for batch in parquet.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            mask = pc.and_(
                pc.is_in(pc.cast(table["__symbol__"], pa.string()), value_set=allowed_symbols),
                pc.and_(
                    pc.greater_equal(table["__ts__"], pa.scalar(candidate_start)),
                    pc.less(table["__ts__"], pa.scalar(candidate_end)),
                ),
            )
            if not bool(pc.any(mask).as_py()):
                continue
            projected = _project_feature_batch(table.filter(mask).to_batches()[0])
            ts_values = pd.to_datetime(projected["__ts__"].to_pandas(), utc=True, errors="raise")
            batch_min, batch_max = ts_values.min(), ts_values.max()
            min_ts = batch_min if min_ts is None else min(min_ts, batch_min)
            max_ts = batch_max if max_ts is None else max(max_ts, batch_max)
            if writer is None:
                writer = pq.ParquetWriter(destination, projected.schema, compression="zstd")
            writer.write_table(projected)
            rows += len(projected)
    finally:
        if writer is not None:
            writer.close()
    if rows == 0 or min_ts is None or max_ts is None:
        raise ValueError(f"{source} has no feature rows")
    return {
        "path": str(destination.resolve()),
        "sha256": _sha256(destination),
        "rows": rows,
        "start": min_ts.isoformat(),
        "end_exclusive": (max_ts + pd.Timedelta(hours=1)).isoformat(),
        "projection": "target-free feature values copied unchanged; only short identities appended",
        "candidate_universe_filter": {
            "symbols": len(candidate_symbols),
            "start": candidate_start.isoformat(),
            "end_exclusive": candidate_end.isoformat(),
        },
    }


def _project(source: Path, *, allow_short: bool) -> pd.DataFrame:
    frame = pd.read_parquet(source)
    missing = sorted(set(IDENTITY).difference(frame.columns))
    if missing:
        raise ValueError(f"{source} lacks target-free identities: {missing}")
    frame["__ts__"] = _utc(frame["__ts__"])
    frame["__decision_ts__"] = _utc(frame["__decision_ts__"])
    expected_decision = frame["__ts__"] + pd.Timedelta(hours=1)
    if not frame["__decision_ts__"].eq(expected_decision).all():
        raise ValueError(f"{source} does not use signal close + one-hour decision timestamps")
    observed_side = set(frame["side_name"].astype(str).str.lower())
    if observed_side == {"short"}:
        if not allow_short:
            raise ValueError("a short source requires --allow-short-source")
    elif observed_side != {"long"}:
        raise ValueError(f"{source} is not a one-side target-free grid: {sorted(observed_side)}")
    expected = _short_id(frame)
    frame["side_name"] = "short"
    frame["candidate_id"] = expected
    if frame["candidate_id"].duplicated().any() or frame.duplicated(["__ts__", "__symbol__"]).any():
        raise ValueError(f"{source} cannot form unique short candidate identities")
    return frame


def run(*, sources: list[Path], out: Path, features_source: Path | None = None) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    pieces = [_project(path, allow_short=True) for path in sources]
    frame = pd.concat(pieces, ignore_index=True).sort_values(["__ts__", "__symbol__"], kind="stable").reset_index(drop=True)
    if frame["candidate_id"].duplicated().any() or frame.duplicated(["__ts__", "__symbol__"]).any():
        raise ValueError("sources overlap after short-side projection")
    if not frame["side_name"].eq("short").all():
        raise AssertionError("short projection leaked another side")
    if not frame["candidate_id"].eq(_short_id(frame)).all():
        raise AssertionError("short candidate ID contract mismatch")
    out.mkdir(parents=True)
    path = out / "short_target_free_candidate_population.parquet"
    frame.to_parquet(path, index=False, compression="zstd")
    feature_output = None
    if features_source is not None:
        feature_output = _project_features(
            features_source,
            destination=out / "canonical120_features.parquet",
            candidate_symbols=frozenset(frame["__symbol__"].astype(str).unique()),
            candidate_start=pd.Timestamp(frame["__ts__"].min()),
            candidate_end=pd.Timestamp(frame["__ts__"].max()) + pd.Timedelta(hours=1),
        )
    manifest = {
        "schema": "strict_r3_short_target_free_history_projection_v2",
        "status": "complete",
        "side": "short",
        "sources": [{"path": str(source.resolve()), "sha256": _sha256(source)} for source in sources],
        "output": {"path": str(path.resolve()), "sha256": _sha256(path)},
        "rows": int(len(frame)),
        "start": frame["__ts__"].min().isoformat(),
        "end_exclusive": (frame["__ts__"].max() + pd.Timedelta(hours=1)).isoformat(),
        "identity": "symbol|short|UTC signal timestamp",
        "projection": (
            "target-free, side-neutral decision-time candidate facts copied unchanged; "
            "only side_name/candidate_id are projected"
        ),
        "feature_source": (
            {"path": str(features_source.resolve()), "sha256": _sha256(features_source)}
            if features_source is not None
            else None
        ),
        "feature_output": feature_output,
        "prohibited_inputs": ["outcome", "future_path", "label", "target", "label_valid", "target_invalid"],
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument(
        "--features-source",
        type=Path,
        help="target-free causal feature panel to project with the same short identity contract",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(sources=args.source, out=args.out, features_source=args.features_source)


if __name__ == "__main__":
    main()
