#!/usr/bin/env python3
"""Validate materialized trailing-profit labels before base/meta retraining."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "__ts__",
    "__symbol__",
    "__y_lbl__",
    "__y_bin__",
    "__y_ret__",
    "__y_outcome__",
    "__u_policy_net__",
    "__r_policy_net__",
    "__first_touch_target_soft__",
    "__first_touch_policy_soft__",
    "__first_touch_capture_net__",
    "__first_touch_round_trip_cost__",
    "__first_touch_hit__",
    "__first_touch_stop__",
    "__first_touch_timeout__",
    "__first_touch_valid_path__",
    "__first_touch_effective_tp_abs__",
    "__first_touch_effective_sl_abs__",
    "__first_touch_effective_trail_abs__",
    "__trailing_profit_activated__",
    "__archetype_label_family__",
    "__archetype_label_source__",
    "__archetype_policy_key__",
    "__archetype_policy_role__",
    "__archetype_policy_confidence__",
    "__archetype_policy_tp_r__",
    "__archetype_policy_sl_r__",
    "__archetype_policy_trail_r__",
)

TOP30_TARGET_COLUMNS = (
    "__first_touch_policy_soft__",
    "__first_touch_target_soft__",
    "__first_touch_capture_net__",
    "__first_touch_hit__",
    "__first_touch_stop__",
    "__first_touch_timeout__",
    "__first_touch_valid_path__",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return val if math.isfinite(val) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(path).schema.names)
    except Exception:
        return list(pd.read_parquet(path, nrows=1).columns)  # type: ignore[call-arg]


def _side_values(frame: pd.DataFrame) -> pd.Series:
    if "__side__" in frame.columns:
        raw = frame["__side__"]
    elif "side" in frame.columns:
        raw = frame["side"]
    elif "side_name" in frame.columns:
        raw = frame["side_name"].astype(str).str.lower().map({"long": 1.0, "short": -1.0})
    else:
        return pd.Series(np.nan, index=frame.index)
    if pd.api.types.is_numeric_dtype(raw):
        return pd.to_numeric(raw, errors="coerce")
    return raw.astype(str).str.lower().map({"long": 1.0, "short": -1.0})


def validate_labels(
    labels_dir: Path,
    *,
    expected_cost: float,
    min_start: str | None,
    min_end: str | None,
    out_dir: Path,
) -> dict[str, Any]:
    files = sorted(labels_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {labels_dir}")

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    total_rows = 0
    side_counts = {"long": 0, "short": 0, "unknown": 0}
    global_min_ts: pd.Timestamp | None = None
    global_max_ts: pd.Timestamp | None = None
    all_months: set[str] = set()
    cost_min = float("nan")
    cost_max = float("nan")

    for file in files:
        cols = _parquet_columns(file)
        missing = sorted(set(REQUIRED_COLUMNS).difference(cols))
        if missing:
            failures.append(f"{file.name}: missing columns {missing}")
        read_cols = list(dict.fromkeys([col for col in [*REQUIRED_COLUMNS, "side", "side_name", "__side__"] if col in cols]))
        frame = pd.read_parquet(file, columns=read_cols)
        total_rows += int(len(frame))
        ts = pd.to_datetime(frame["__ts__"], utc=True, errors="coerce")
        if ts.notna().any():
            file_min = ts.min()
            file_max = ts.max()
            global_min_ts = file_min if global_min_ts is None else min(global_min_ts, file_min)
            global_max_ts = file_max if global_max_ts is None else max(global_max_ts, file_max)
            all_months.update(ts.dt.to_period("M").dropna().astype(str).unique().tolist())
        else:
            file_min = pd.NaT
            file_max = pd.NaT
            failures.append(f"{file.name}: no valid timestamps")

        side = _side_values(frame)
        long_rows = int(side.gt(0).sum())
        short_rows = int(side.lt(0).sum())
        unknown_rows = int(len(frame) - long_rows - short_rows)
        side_counts["long"] += long_rows
        side_counts["short"] += short_rows
        side_counts["unknown"] += unknown_rows

        cost = pd.to_numeric(frame.get("__first_touch_round_trip_cost__"), errors="coerce")
        finite_cost = cost[np.isfinite(cost)]
        if len(finite_cost):
            local_min = float(finite_cost.min())
            local_max = float(finite_cost.max())
            cost_min = local_min if not math.isfinite(cost_min) else min(cost_min, local_min)
            cost_max = local_max if not math.isfinite(cost_max) else max(cost_max, local_max)
            if not np.allclose(finite_cost.to_numpy(dtype=np.float64), float(expected_cost), atol=1e-8, rtol=1e-6):
                failures.append(f"{file.name}: round-trip cost differs from {expected_cost}")
        else:
            failures.append(f"{file.name}: no finite round-trip cost")

        null_policy_key = int(frame.get("__archetype_policy_key__", pd.Series(index=frame.index, dtype=object)).isna().sum())
        target_finite = {
            col: float(pd.to_numeric(frame[col], errors="coerce").notna().mean())
            for col in TOP30_TARGET_COLUMNS
            if col in frame.columns
        }
        rows.append(
            {
                "file": file.name,
                "rows": int(len(frame)),
                "min_ts": file_min,
                "max_ts": file_max,
                "long_rows": long_rows,
                "short_rows": short_rows,
                "unknown_side_rows": unknown_rows,
                "policy_key_null_rows": null_policy_key,
                "cost_min": float(finite_cost.min()) if len(finite_cost) else float("nan"),
                "cost_max": float(finite_cost.max()) if len(finite_cost) else float("nan"),
                **{f"{col}_finite_rate": val for col, val in target_finite.items()},
            }
        )

    if side_counts["long"] <= 0:
        failures.append("no long rows found")
    if side_counts["short"] <= 0:
        failures.append("no short rows found")
    if side_counts["unknown"] > 0:
        failures.append(f"unknown side rows found: {side_counts['unknown']}")
    if min_start is not None and global_min_ts is not None:
        required_start = pd.Timestamp(min_start, tz="UTC")
        if global_min_ts > required_start:
            failures.append(f"period starts too late: {global_min_ts} > {required_start}")
    if min_end is not None and global_max_ts is not None:
        required_end = pd.Timestamp(min_end, tz="UTC")
        if global_max_ts < required_end:
            failures.append(f"period ends too early: {global_max_ts} < {required_end}")

    out_dir.mkdir(parents=True, exist_ok=True)
    per_file = pd.DataFrame(rows)
    per_file_path = out_dir / "label_validation_by_file.csv"
    per_file.to_csv(per_file_path, index=False)
    result = {
        "labels_dir": str(labels_dir),
        "files": int(len(files)),
        "rows": int(total_rows),
        "timestamp_min": global_min_ts,
        "timestamp_max": global_max_ts,
        "months": sorted(all_months),
        "month_count": int(len(all_months)),
        "side_counts": side_counts,
        "expected_cost": float(expected_cost),
        "cost_min": cost_min,
        "cost_max": cost_max,
        "required_columns": list(REQUIRED_COLUMNS),
        "top30_target_columns": list(TOP30_TARGET_COLUMNS),
        "failures": failures,
        "status": "pass" if not failures else "fail",
        "per_file": str(per_file_path),
    }
    result_path = out_dir / "label_validation_summary.json"
    result_path.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--expected-cost", type=float, default=0.01)
    parser.add_argument("--min-start", default="2025-01-01")
    parser.add_argument("--min-end", default="2026-06-30")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_labels(
        args.labels_dir,
        expected_cost=float(args.expected_cost),
        min_start=args.min_start,
        min_end=args.min_end,
        out_dir=args.out_dir,
    )
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
