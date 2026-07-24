#!/usr/bin/env python3
"""Replace a sparse frozen candidate tail with a full-universe rescore."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEYS = ("__ts__", "__symbol__", "side_name")


def _read(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--tail", type=Path, required=True)
    parser.add_argument("--replace-from", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    replace_from = pd.Timestamp(args.replace_from, tz="UTC")
    head = _read(args.head)
    tail = _read(args.tail)
    head = head.loc[head["__ts__"].lt(replace_from)].copy()
    tail = tail.loc[tail["__ts__"].ge(replace_from)].copy()
    if set(head.columns) != set(tail.columns):
        missing_head = sorted(set(tail.columns) - set(head.columns))
        missing_tail = sorted(set(head.columns) - set(tail.columns))
        raise ValueError(f"Candidate schemas differ; head_missing={missing_head}; tail_missing={missing_tail}")
    merged = pd.concat([head, tail.loc[:, head.columns]], ignore_index=True, copy=False)
    if merged.duplicated(list(KEYS)).any():
        raise ValueError("Merged candidate stream has duplicate timestamp/symbol/side keys")
    merged = merged.sort_values(list(KEYS), kind="stable").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False, compression="zstd")
    audit = {
        "head": str(args.head),
        "tail": str(args.tail),
        "replace_from": replace_from.isoformat(),
        "rows": int(len(merged)),
        "head_rows_retained": int(len(head)),
        "tail_rows_replaced": int(len(tail)),
        "min_ts": merged["__ts__"].min().isoformat(),
        "max_ts": merged["__ts__"].max().isoformat(),
        "symbols": int(merged["__symbol__"].nunique()),
        "daily_side_rows": (
            merged.assign(day=merged["__ts__"].dt.strftime("%Y-%m-%d"))
            .groupby(["day", "side_name"], observed=True)
            .size()
            .rename("rows")
            .reset_index()
            .to_dict(orient="records")
        ),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
