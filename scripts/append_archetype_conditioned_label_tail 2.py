#!/usr/bin/env python3
"""Safely append a newly materialized monthly label tail to a canonical shard.

This is intentionally narrow: it preserves every existing row, appends only
strictly newer signal timestamps, and records enough evidence to make a later
training/OOS comparison reproducible.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _assert_causal_tail(frame: pd.DataFrame, *, side: str) -> None:
    required = {"__ts__", "__decision_ts__", "__first_path_ts__"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{side}: tail lacks causal provenance columns: {missing}")
    signal = _utc(frame["__ts__"])
    decision = _utc(frame["__decision_ts__"])
    first = _utc(frame["__first_path_ts__"])
    invalid = signal.isna() | decision.isna() | first.isna()
    invalid |= decision.lt(signal + pd.Timedelta(hours=1))
    invalid |= first.lt(decision)
    if invalid.any():
        raise ValueError(
            f"{side}: causal invariant failed for {int(invalid.sum()):,} tail rows; "
            "first path must begin at or after one full signal bar."
        )


def _reject_degenerate_archetypes(frame: pd.DataFrame, *, side: str) -> dict[str, int]:
    column = "__archetype_label_family__"
    if column not in frame.columns:
        raise ValueError(f"{side}: tail lacks {column}")
    counts = frame[column].fillna("mixed").astype(str).value_counts().sort_index()
    result = {str(key): int(value) for key, value in counts.items()}
    if not result or set(result) == {"mixed"}:
        raise ValueError(
            f"{side}: refusing archetype append because all rows are mixed; "
            "feature/archetype materialization is incomplete."
        )
    return result


def _dedupe_sort(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["__ts__"] = _utc(ordered["__ts__"])
    keys = [key for key in ("candidate_id", "__ts__", "__symbol__", "side_name") if key in ordered]
    if "candidate_id" in keys:
        ordered = ordered.drop_duplicates(subset=["candidate_id"], keep="last")
    else:
        ordered = ordered.drop_duplicates(subset=keys, keep="last")
    sort_keys = [key for key in ("__ts__", "__symbol__", "side_name", "candidate_id") if key in ordered]
    return ordered.sort_values(sort_keys, kind="stable").reset_index(drop=True)


def _update_manifest(labels_dir: Path, *, side: str, columns: list[str]) -> None:
    path = labels_dir / "labels_manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    dataset = f"train_global_{side}_5"
    shards = sorted(labels_dir.glob(f"{dataset}_????_??.parquet"))
    if not shards:
        return
    rows = int(sum(len(pd.read_parquet(shard, columns=["__ts__"])) for shard in shards))
    spec = dict((manifest.get("datasets") or {}).get(dataset) or {})
    spec.update({"rows": rows, "columns": columns, "file": shards[-1].name})
    manifest.setdefault("datasets", {})[dataset] = spec
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_side(target: Path, tail: Path, *, side: str, backup_dir: Path) -> dict[str, object]:
    existing = _read(target)
    materialized = _read(tail)
    _assert_causal_tail(materialized, side=side)
    archetypes = _reject_degenerate_archetypes(materialized, side=side)
    existing_ts = _utc(existing["__ts__"])
    tail_ts = _utc(materialized["__ts__"])
    last_existing = existing_ts.max()
    new = materialized.loc[tail_ts.gt(last_existing)].copy()
    if new.empty:
        raise ValueError(f"{side}: no strictly newer rows to append after {last_existing}")
    # A schema union preserves the old contract exactly and permits new
    # observable source columns without dropping historical values.
    columns = list(existing.columns) + [c for c in materialized.columns if c not in existing.columns]
    merged = pd.concat(
        [existing.reindex(columns=columns), new.reindex(columns=columns)],
        ignore_index=True,
        copy=False,
    )
    merged = _dedupe_sort(merged)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / target.name
    if not backup.exists():
        shutil.copy2(target, backup)
    tmp = target.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False, compression="snappy")
    tmp.replace(target)
    _update_manifest(target.parent, side=side, columns=columns)
    return {
        "side": side,
        "target": str(target),
        "tail": str(tail),
        "backup": str(backup),
        "existing_rows": int(len(existing)),
        "appended_rows": int(len(new)),
        "final_rows": int(len(merged)),
        "last_existing_before_append": last_existing.isoformat(),
        "final_max_ts": _utc(merged["__ts__"]).max().isoformat(),
        "tail_archetype_counts": archetypes,
        "nonfinite_y_lbl": int(
            (~np.isfinite(pd.to_numeric(merged.get("__y_lbl__"), errors="coerce"))).sum()
        ) if "__y_lbl__" in merged else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-labels-dir", type=Path, required=True)
    parser.add_argument("--tail-labels-dir", type=Path, required=True)
    parser.add_argument("--month", required=True, help="YYYY_MM, for example 2026_07")
    parser.add_argument("--backup-dir", type=Path, required=True)
    args = parser.parse_args()

    reports = []
    for side in ("long", "short"):
        name = f"train_global_{side}_5_{args.month}.parquet"
        reports.append(
            _append_side(
                args.canonical_labels_dir / name,
                args.tail_labels_dir / name,
                side=side,
                backup_dir=args.backup_dir,
            )
        )
    report_path = args.canonical_labels_dir / f"label_tail_append_{args.month}.json"
    report_path.write_text(json.dumps({"sides": reports}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "sides": reports}, indent=2))


if __name__ == "__main__":
    main()
