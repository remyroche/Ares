#!/usr/bin/env python3
"""Index atomic exact-1m path-head shards and aggregate label-only support.

The index intentionally reads only the compact target shards.  It never opens
the 720-minute JSON paths, so it can be rerun cheaply after each bounded shard
without re-materialising market data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "febapr2025_exact1m_path_head_shard_index_v1"
TARGETS = {
    "peak_mfe_12h_atr": "__log1p_peak_mfe_atr_12h__",
    "time_to_first_meaningful_mfe": "__log1p_time_to_first_meaningful_mfe_hours_12h__",
    "mae_before_meaningful_mfe_atr": "__log1p_mae_before_meaningful_mfe_atr_12h__",
    "bars_before_price_stops_decreasing": "__log1p_bars_before_price_stops_decreasing_12h__",
    "future_slope_atr_per_hour": "__log1p_future_slope_atr_per_hour_12h__",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        Path(temporary).replace(path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def _shards(root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for manifest_path in sorted((root / "shards").glob("part-*/manifest.json")):
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema") != "febapr2025_exact1m_execution_path_head_labels_v1":
            raise ValueError(f"unexpected shard schema: {manifest_path}")
        labels = Path(payload["labels"]["path"])
        if not labels.is_file() or labels.parent != manifest_path.parent:
            raise ValueError(f"shard labels are missing or outside shard directory: {manifest_path}")
        if _sha256(labels) != payload["labels"]["sha256"]:
            raise ValueError(f"shard hash mismatch: {labels}")
        count = pq.ParquetFile(labels).metadata.num_rows
        if count != int(payload["labels"]["rows"]):
            raise ValueError(f"shard parquet row count mismatch: {labels}")
        output.append({"manifest": manifest_path, "_labels_path": labels, **payload})
    return output


def _coverage(shards: list[dict[str, Any]], expected_rows: int) -> dict[str, Any]:
    intervals: list[tuple[int, int, dict[str, Any]]] = []
    for shard in shards:
        source = shard["source_slice"]
        start = int(source["offset"])
        rows = int(source["expected_rows"])
        if rows != int(shard["labels"]["rows"]):
            raise ValueError("source slice and label row count disagree")
        intervals.append((start, start + rows, shard))
    intervals.sort(key=lambda item: item[0])
    cursor = 0
    gaps: list[list[int]] = []
    overlap: list[list[int]] = []
    for start, end, _ in intervals:
        if start > cursor:
            gaps.append([cursor, start])
        if start < cursor:
            overlap.append([start, min(end, cursor)])
        cursor = max(cursor, end)
    if cursor < expected_rows:
        gaps.append([cursor, expected_rows])
    if cursor > expected_rows:
        overlap.append([expected_rows, cursor])
    return {
        "expected_rows": expected_rows,
        "materialized_rows": int(sum(end - start for start, end, _ in intervals)),
        "complete": not gaps and not overlap and cursor == expected_rows,
        "gaps": gaps,
        "overlap": overlap,
        "completed_offsets": [start for start, _, _ in intervals],
        "missing_ranges": gaps,
    }


def _support(shards: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    columns = [
        "__ts__", "side_name", "path_arch_complete_12h", "path_archetype",
        "__path_auxiliary_target_valid__", "__meaningful_mfe_reached_12h__",
        "__soft_tb_first_event__", "__soft_tb_order_ambiguous__", *TARGETS.values(),
    ]
    frames: list[pd.DataFrame] = []
    for shard in shards:
        file = pq.ParquetFile(shard["_labels_path"])
        for batch in file.iter_batches(batch_size=25_000, columns=columns):
            frames.append(batch.to_pandas())
    if not frames:
        return {"rows": 0, "coverage_support": [], "archetype_class_support_rows": 0}
    frame = pd.concat(frames, ignore_index=True)
    frame["month"] = pd.to_datetime(frame["__ts__"], utc=True).dt.strftime("%Y-%m")
    records: list[dict[str, Any]] = []
    for (month, side), group in frame.groupby(["month", "side_name"], sort=True):
        valid = group["__path_auxiliary_target_valid__"].astype(bool)
        item: dict[str, Any] = {
            "month": month, "side_name": side, "rows": int(len(group)),
            "complete_1m_12h_rows": int(group["path_arch_complete_12h"].sum()),
            "auxiliary_valid_rows": int(valid.sum()),
            "meaningful_mfe_positive_rows": int(group.loc[valid, "__meaningful_mfe_reached_12h__"].sum()),
            "meaningful_mfe_positive_fraction": float(group.loc[valid, "__meaningful_mfe_reached_12h__"].mean()),
            "soft_tb_favorable_first_rows": int((group["__soft_tb_first_event__"] == "favorable_first").sum()),
            "soft_tb_adverse_first_or_conflict_rows": int((group["__soft_tb_first_event__"] == "adverse_first_or_conflict").sum()),
            "soft_tb_timeout_rows": int((group["__soft_tb_first_event__"] == "timeout").sum()),
            "soft_tb_order_ambiguous_rows": int(group["__soft_tb_order_ambiguous__"].sum()),
        }
        for name, column in TARGETS.items():
            values = pd.to_numeric(group.loc[valid, column], errors="coerce")
            item[f"{name}_finite_rows"] = int(values.notna().sum())
            item[f"{name}_mean"] = float(values.mean())
            item[f"{name}_std"] = float(values.std(ddof=0))
        records.append(item)
    pd.DataFrame(records).to_csv(root / "coverage_support_by_month_side.csv", index=False)
    classes = frame.groupby(["month", "side_name", "path_archetype"], dropna=False, sort=True).size().rename("rows").reset_index()
    classes.to_csv(root / "archetype_class_support_by_month_side.csv", index=False)
    return {"rows": int(len(frame)), "coverage_support": records, "archetype_class_support_rows": int(len(classes))}


def index(root: Path, source_paths: Path) -> dict[str, Any]:
    shards = _shards(root)
    expected = pq.ParquetFile(source_paths).metadata.num_rows
    coverage = _coverage(shards, expected)
    support = _support(shards, root)
    result = {
        "schema": SCHEMA, "source_paths": {"path": str(source_paths), "sha256": _sha256(source_paths), "rows": int(expected)},
        "coverage": coverage, "shards": [
            {"offset": item["source_slice"]["offset"], "rows": item["labels"]["rows"], "labels": str(item["_labels_path"]), "sha256": item["labels"]["sha256"]}
            for item in shards
        ],
        "support": support,
        "catboost_12h_vs_v6_24h_limitation": "Preserved: 12h execution-compatible v6-rule derivative is not bitwise comparable with frozen 24h v6 labels.",
        "models_trained": False,
    }
    _atomic_text(root / "index.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-paths", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(index(args.root, args.source_paths), sort_keys=True))


if __name__ == "__main__":
    main()
