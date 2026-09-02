#!/usr/bin/env python3
"""Assemble source-aligned strict-R3 policy supervision without scoring rows.

The selected policy is immutable: first 15-minute open one hour after the
signal, H12, SL 3 ATR, trailing activation 0.5 ATR, giveback 0.25 ATR and a
100-bps round-trip cost applied once.  This utility joins precomputed labels
to the exact target-free source identities and makes missing paths explicit;
it neither creates candidates nor falls back to a different policy geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


IDENTITY = ["candidate_id", "__ts__", "__decision_ts__", "__symbol__", "side_name"]
POLICY = [
    "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price",
    "policy_exit_price", "policy_label_available_ts", "policy_outcome_source",
    "policy_cost_bps",
]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def _read(
    path: Path, *, start: pd.Timestamp, end: pd.Timestamp, side: str,
) -> pd.DataFrame:
    available = set(pq.ParquetFile(path).schema.names)
    columns = [column for column in [*IDENTITY, *POLICY] if column in available]
    frame = pd.read_parquet(
        path,
        columns=columns,
        filters=[
            ("__decision_ts__", ">=", start),
            ("__decision_ts__", "<", end),
            ("side_name", "==", side),
        ],
    )
    for column in ("__ts__", "__decision_ts__", "policy_label_available_ts"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"duplicate candidate IDs in {path}")
    observed = frame["side_name"].astype(str).str.strip().str.lower()
    if frame.empty or not observed.eq(side).all():
        raise ValueError(f"{path} is not an explicit {side}-side label source")
    frame["side_name"] = side
    return frame


def _normalise_labels(frame: pd.DataFrame, *, label_source: str) -> pd.DataFrame:
    result = frame.copy()
    for column in POLICY:
        if column not in result:
            if column == "policy_path_valid":
                result[column] = False
            elif column == "policy_label_available_ts":
                result[column] = result["__decision_ts__"] + pd.Timedelta(hours=12)
            elif column == "policy_outcome_source":
                result[column] = label_source
            else:
                result[column] = np.nan
    valid = result["policy_path_valid"].fillna(False).astype(bool)
    result["policy_path_valid"] = valid
    result["policy_outcome_source"] = np.where(
        valid,
        result["policy_outcome_source"].fillna(label_source).astype(str),
        "unavailable",
    )
    label_available = pd.to_datetime(result["policy_label_available_ts"], utc=True, errors="coerce")
    result["policy_label_available_ts"] = label_available.where(
        label_available.notna(), result["__decision_ts__"] + pd.Timedelta(hours=12),
    )
    finite = np.isfinite(pd.to_numeric(result["policy_net_bps"], errors="coerce"))
    if (valid & ~finite).any():
        raise ValueError(f"{label_source} labels mark non-finite net outcomes valid")
    with_gross = valid & np.isfinite(pd.to_numeric(result["policy_gross_bps"], errors="coerce"))
    if with_gross.any() and not np.allclose(
        result.loc[with_gross, "policy_net_bps"].to_numpy(float),
        result.loc[with_gross, "policy_gross_bps"].to_numpy(float) - 100.0,
        atol=1e-9,
        rtol=0.0,
    ):
        raise ValueError(f"{label_source} does not apply the 100-bps cost exactly once")
    return result.loc[:, [*IDENTITY, *POLICY]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument("--early-policy-labels", type=Path, required=True)
    parser.add_argument("--later-policy-outcomes", type=Path, required=True)
    parser.add_argument("--early-end", default="2025-01-01")
    parser.add_argument("--side", choices=("long", "short"), default="long")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    start, early_end, end = _utc(args.start), _utc(args.early_end), _utc(args.end)
    if not start < early_end < end:
        raise ValueError("require start < early-end < end")

    source_columns = IDENTITY
    source = pd.read_parquet(
        args.source_panel,
        columns=source_columns,
        filters=[
            ("__decision_ts__", ">=", start),
            ("__decision_ts__", "<", end),
            ("side_name", "==", args.side),
        ],
    )
    for column in ("__ts__", "__decision_ts__"):
        source[column] = pd.to_datetime(source[column], utc=True, errors="raise")
    if source.empty or source["candidate_id"].duplicated().any():
        raise ValueError("source panel is empty or has duplicate candidate IDs")
    source_side = source["side_name"].astype(str).str.strip().str.lower()
    if not source_side.eq(args.side).all():
        raise ValueError("source panel is not side-local after filtering")
    source["side_name"] = args.side

    early = _normalise_labels(
        _read(args.early_policy_labels, start=start, end=early_end, side=args.side),
        label_source="frozen_15m_2024",
    )
    later = _normalise_labels(
        _read(args.later_policy_outcomes, start=early_end, end=end, side=args.side),
        label_source="existing_policy_outcomes",
    )
    labels = pd.concat([early, later], ignore_index=True)
    if labels["candidate_id"].duplicated().any():
        raise ValueError("early/later policy labels overlap by candidate identity")

    label_index = labels.set_index("candidate_id", drop=False)
    present = source["candidate_id"].isin(label_index.index)
    joined = source.copy()
    if present.any():
        values = label_index.reindex(source.loc[present, "candidate_id"])
        for column in POLICY:
            joined.loc[present, column] = values[column].to_numpy()
        label_ts = values["__decision_ts__"].to_numpy()
        if not np.array_equal(label_ts, source.loc[present, "__decision_ts__"].to_numpy()):
            raise ValueError("policy label identity disagrees with source decision timestamp")
    absent = ~present
    joined.loc[absent, "policy_path_valid"] = False
    joined.loc[absent, "policy_exit_reason"] = "unavailable_policy_path"
    joined.loc[absent, "policy_outcome_source"] = "unavailable"
    joined.loc[absent, "policy_label_available_ts"] = (
        joined.loc[absent, "__decision_ts__"] + pd.Timedelta(hours=12)
    )
    for column in ("policy_gross_bps", "policy_net_bps", "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price", "policy_cost_bps"):
        joined.loc[absent, column] = np.nan
    joined["policy_path_valid"] = joined["policy_path_valid"].fillna(False).astype(bool)
    joined["policy_label_available_ts"] = pd.to_datetime(
        joined["policy_label_available_ts"], utc=True, errors="raise",
    )
    valid = joined["policy_path_valid"] & np.isfinite(pd.to_numeric(joined["policy_net_bps"], errors="coerce"))
    if (joined.loc[valid, "policy_label_available_ts"] < joined.loc[valid, "__decision_ts__"]).any():
        raise ValueError("a policy outcome is available before its decision")
    if joined["candidate_id"].duplicated().any() or len(joined) != len(source):
        raise AssertionError("policy ledger changed exact source identities")

    joined["month"] = joined["__decision_ts__"].dt.strftime("%Y-%m")
    coverage = joined.assign(_valid=valid).groupby(["month", "policy_outcome_source"], as_index=False).agg(
        rows=("candidate_id", "size"), valid_rows=("_valid", "sum"),
        mean_net_bps=("policy_net_bps", "mean"),
    )
    args.out_dir.mkdir(parents=True)
    joined.drop(columns="month").sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    ).to_parquet(args.out_dir / "candidate_policy_outcomes.parquet", index=False, compression="zstd")
    coverage.to_parquet(args.out_dir / "policy_coverage_by_month_source.parquet", index=False)
    manifest = {
        "schema": "strict_r3_source_aligned_policy_outcome_ledger_v1",
        "side": args.side,
        "source_panel": str(args.source_panel), "source_panel_sha256": _sha(args.source_panel),
        "early_policy_labels": str(args.early_policy_labels), "early_policy_labels_sha256": _sha(args.early_policy_labels),
        "later_policy_outcomes": str(args.later_policy_outcomes), "later_policy_outcomes_sha256": _sha(args.later_policy_outcomes),
        "period_start": start.isoformat(), "early_end_exclusive": early_end.isoformat(), "period_end_exclusive": end.isoformat(),
        "policy": "15m entry signal+1h; H12; SL3; activation0.5; giveback0.25; 100bps once",
        "source_rows": int(len(source)), "label_identity_matches": int(present.sum()),
        "unavailable_source_rows": int(absent.sum()), "valid_rows": int(valid.sum()),
        "candidate_contract": "all target-free source identities retained; unavailable paths explicit",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), **manifest}, default=str))


if __name__ == "__main__":
    main()
