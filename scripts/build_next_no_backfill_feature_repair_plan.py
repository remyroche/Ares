#!/usr/bin/env python3
"""Build a bounded feature-history repair plan for no-backfill shadow windows.

The readiness audit identifies whether a no-backfill window is blocked by
missing feature-store timestamps. This helper turns that evidence into a
reproducible, explicit repair plan without executing feature generation.
"""

from __future__ import annotations

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_READINESS = Path(
    "data_perp/reports/market_state_next_no_backfill_shadow_window_readiness_globalrank_activefs_20260628_v1/"
    "next_no_backfill_shadow_window_readiness.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/market_state_next_no_backfill_feature_repair_plan"
)
DEFAULT_MODEL_RUN_ID = "20260617_090000_no_mkt4_labelhpo_final_fit"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _as_utc(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _symbols_from_feature_store(feature_store_dir: Path) -> list[str]:
    symbols: list[str] = []
    for path in sorted(feature_store_dir.glob("symbol=*.parquet")):
        name = path.name
        raw = name[len("symbol=") : -len(".parquet")]
        symbols.append(raw.replace("_", "/", 1))
    return symbols


def _latest_matrix_root(data_root: Path, model_run_id: str) -> Path:
    return (
        data_root
        / "artifacts"
        / model_run_id
        / "live_selected_feature_latest_matrix"
    )


def _feature_keys_from_latest_matrices(matrix_root: Path) -> list[str]:
    keys: set[str] = set()
    if not matrix_root.exists():
        return []
    for path in sorted(matrix_root.glob("*/*/latest.parquet")):
        try:
            names = list(pq.ParquetFile(path).schema_arrow.names)
        except Exception:
            continue
        keys.update(str(name) for name in names)
    blocked = {"", "__index_level_0__", "ts", "timestamp", "symbol"}
    return sorted(key for key in keys if key not in blocked)


def _coverage_gap_rows(coverage: dict[str, Any]) -> list[dict[str, Any]]:
    gap_types = dict(coverage.get("low_coverage_gap_type_by_timestamp") or {})
    blocks = dict(coverage.get("low_coverage_blocks_threshold_by_timestamp") or {})
    ratios = dict(coverage.get("low_coverage_feature_file_coverage_by_timestamp") or {})
    present = dict(coverage.get("low_coverage_present_file_count_by_timestamp") or {})
    missing = dict(coverage.get("low_coverage_missing_file_count_by_timestamp") or {})
    rows: list[dict[str, Any]] = []
    for timestamp in sorted(gap_types):
        rows.append(
            {
                "timestamp": timestamp,
                "gap_type": gap_types.get(timestamp),
                "blocks_threshold": bool(blocks.get(timestamp)),
                "coverage": ratios.get(timestamp),
                "present_feature_file_count": present.get(timestamp),
                "missing_feature_file_count": missing.get(timestamp),
            }
        )
    return rows


def _contiguous_ranges(timestamps: list[pd.Timestamp]) -> list[dict[str, str]]:
    if not timestamps:
        return []
    ordered = sorted(pd.Timestamp(ts).floor("h") for ts in timestamps)
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = ordered[0]
    prev = ordered[0]
    for ts in ordered[1:]:
        if ts == prev + pd.Timedelta(hours=1):
            prev = ts
            continue
        ranges.append((start, prev))
        start = prev = ts
    ranges.append((start, prev))
    return [
        {"start": start.isoformat(), "end": end.isoformat()}
        for start, end in ranges
    ]


def _repair_command(
    *,
    data_root: Path,
    model_run_id: str,
    exchange: str,
    symbols_file: Path,
    keys_file: Path,
    feature_end_ts: str,
    symbol_chunk_size: int,
) -> list[str]:
    return [
        "env",
        "PYTHONUNBUFFERED=1",
        "PYTHONPATH=.",
        f"EPM_DATA_ROOT={data_root}",
        f"EPM_FEATURE_END_TS={feature_end_ts}",
        "EPM_FEATURE_SAVE_WORKERS=1",
        "EPM_FEATURE_BACKFILL_COMPUTE_WORKERS=1",
        f"EPM_FEATURE_BACKFILL_SYMBOL_CHUNK_SIZE={int(symbol_chunk_size)}",
        "EPM_FEATURE_BACKFILL_KEY_BATCH_SIZE=0",
        "EPM_FEATURE_BACKFILL_ALL_INCOMPLETE_KEYS=0",
        "EPM_FEATURE_LIVE_DECISION_TAIL_ONLY=0",
        f"EPM_FEATURE_SYMBOLS_FILE={symbols_file}",
        f"EPM_FEATURE_BACKFILL_KEYS_FILE={keys_file}",
        "python3",
        "-u",
        "extreme_price_movements/run_pipeline.py",
        "features",
        "--perps",
        "--exchange",
        exchange,
        "--run-id",
        model_run_id,
        "--skip-feature-postsave-checks",
    ]


def build_plan(
    *,
    readiness: dict[str, Any],
    readiness_path: Path,
    output_dir: Path,
    data_root: Path,
    model_run_id: str,
    exchange: str,
    symbol_chunk_size: int,
) -> dict[str, Any]:
    feature_store_dir = Path(str(readiness.get("feature_store_dir") or ""))
    if not feature_store_dir.exists():
        raise FileNotFoundError(f"Feature store not found: {feature_store_dir}")
    symbols = _symbols_from_feature_store(feature_store_dir)
    matrix_root = _latest_matrix_root(data_root, model_run_id)
    feature_keys = _feature_keys_from_latest_matrices(matrix_root)
    min_coverage = dict(readiness.get("minimum_window_feature_coverage") or {})
    full_coverage = dict(readiness.get("full_window_feature_coverage") or {})
    min_rows = _coverage_gap_rows(min_coverage)
    full_rows = _coverage_gap_rows(full_coverage)
    internal_rows = [
        row
        for row in full_rows
        if bool(row.get("blocks_threshold"))
        and str(row.get("gap_type", "")).startswith("internal_")
    ]
    tail_rows = [
        row
        for row in full_rows
        if bool(row.get("blocks_threshold"))
        and str(row.get("gap_type")) == "tail_not_generated_yet"
    ]
    internal_ts = [
        ts for ts in (_as_utc(row.get("timestamp")) for row in internal_rows) if ts is not None
    ]
    tail_ts = [
        ts for ts in (_as_utc(row.get("timestamp")) for row in tail_rows) if ts is not None
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    symbols_file = output_dir / "feature_repair_symbols.txt"
    keys_file = output_dir / "feature_repair_keys.txt"
    symbols_file.write_text("\n".join(symbols) + ("\n" if symbols else ""), encoding="utf-8")
    keys_file.write_text("\n".join(feature_keys) + ("\n" if feature_keys else ""), encoding="utf-8")

    min_end = readiness.get("needed_feature_timestamp_max_for_min_window")
    full_end = readiness.get("needed_feature_timestamp_max_for_full_window")
    commands = {
        "repair_minimum_scoreable_window": _repair_command(
            data_root=data_root,
            model_run_id=model_run_id,
            exchange=exchange,
            symbols_file=symbols_file,
            keys_file=keys_file,
            feature_end_ts=str(min_end),
            symbol_chunk_size=symbol_chunk_size,
        )
        if min_end
        else [],
        "repair_full_target_window": _repair_command(
            data_root=data_root,
            model_run_id=model_run_id,
            exchange=exchange,
            symbols_file=symbols_file,
            keys_file=keys_file,
            feature_end_ts=str(full_end),
            symbol_chunk_size=symbol_chunk_size,
        )
        if full_end
        else [],
    }
    return {
        "generated_by": "build_next_no_backfill_feature_repair_plan",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "readiness_json": str(readiness_path),
        "output_dir": str(output_dir),
        "data_root": str(data_root),
        "feature_store_dir": str(feature_store_dir),
        "model_run_id": model_run_id,
        "exchange": exchange,
        "status_before_repair": readiness.get("status"),
        "next_action_before_repair": readiness.get("next_action"),
        "scoreable_min_window_now": bool(readiness.get("scoreable_min_window_now")),
        "scoreable_full_window_now": bool(readiness.get("scoreable_full_window_now")),
        "next_window_start": readiness.get("next_window_start"),
        "minimum_window_end": readiness.get("minimum_window_end"),
        "target_window_end": readiness.get("target_window_end"),
        "needed_feature_timestamp_max_for_min_window": min_end,
        "needed_feature_timestamp_max_for_full_window": full_end,
        "feature_timestamp_max_before_repair": readiness.get("feature_timestamp_max"),
        "internal_blocking_gap_count": len(internal_rows),
        "tail_blocking_gap_count": len(tail_rows),
        "internal_blocking_gap_ranges": _contiguous_ranges(internal_ts),
        "tail_blocking_gap_ranges": _contiguous_ranges(tail_ts),
        "minimum_window_blocking_gap_counts": min_coverage.get(
            "blocking_low_coverage_gap_type_counts"
        ),
        "full_window_blocking_gap_counts": full_coverage.get(
            "blocking_low_coverage_gap_type_counts"
        ),
        "feature_symbol_count": len(symbols),
        "feature_symbols_file": str(symbols_file),
        "repair_feature_key_count": len(feature_keys),
        "repair_feature_keys_file": str(keys_file),
        "selected_latest_matrix_root": str(matrix_root),
        "memory_safety": {
            "execution_recommendation": (
                "do_not_run_while_live_feature_generation_or_high_rss_inference_is_active"
            ),
            "suggested_symbol_chunk_size": int(symbol_chunk_size),
            "save_workers": 1,
            "compute_workers": 1,
            "key_batch_size": 0,
            "tail_only": False,
        },
        "commands": commands,
        "command_strings": {
            name: shlex.join(command) for name, command in commands.items() if command
        },
    }


def write_plan(plan: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_repair_plan.json").write_text(
        json.dumps(_json_safe(plan), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    command_rows = [
        {"command_name": name, "command": command}
        for name, command in plan.get("command_strings", {}).items()
    ]
    pd.DataFrame(command_rows).to_csv(output_dir / "feature_repair_commands.csv", index=False)
    lines = [
        "# Next No-Backfill Feature Repair Plan",
        "",
        f"- Status before repair: `{plan['status_before_repair']}`",
        f"- Next action before repair: `{plan['next_action_before_repair']}`",
        f"- Feature store: `{plan['feature_store_dir']}`",
        f"- Feature max before repair: `{plan['feature_timestamp_max_before_repair']}`",
        f"- Next window start: `{plan['next_window_start']}`",
        f"- Minimum window end: `{plan['minimum_window_end']}`",
        f"- Target window end: `{plan['target_window_end']}`",
        f"- Needed feature max for minimum window: `{plan['needed_feature_timestamp_max_for_min_window']}`",
        f"- Needed feature max for full window: `{plan['needed_feature_timestamp_max_for_full_window']}`",
        f"- Internal blocking gaps: `{plan['internal_blocking_gap_count']}`",
        f"- Tail blocking gaps: `{plan['tail_blocking_gap_count']}`",
        f"- Feature symbols: `{plan['feature_symbol_count']}`",
        f"- Repair feature keys: `{plan['repair_feature_key_count']}`",
        f"- Symbols file: `{plan['feature_symbols_file']}`",
        f"- Keys file: `{plan['repair_feature_keys_file']}`",
        "",
        "## Internal Gap Ranges",
        "",
        pd.DataFrame(plan["internal_blocking_gap_ranges"]).to_markdown(index=False)
        if plan["internal_blocking_gap_ranges"]
        else "_None._",
        "",
        "## Tail Gap Ranges",
        "",
        pd.DataFrame(plan["tail_blocking_gap_ranges"]).to_markdown(index=False)
        if plan["tail_blocking_gap_ranges"]
        else "_None._",
        "",
        "## Commands",
        "",
    ]
    for name, command in plan.get("command_strings", {}).items():
        lines.extend([f"### {name}", "", f"```bash\n{command}\n```", ""])
    (output_dir / "feature_repair_plan.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-json", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-run-id", default=DEFAULT_MODEL_RUN_ID)
    parser.add_argument("--exchange", default="kraken")
    parser.add_argument("--symbol-chunk-size", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    readiness = _load_json(args.readiness_json)
    plan = build_plan(
        readiness=readiness,
        readiness_path=args.readiness_json,
        output_dir=args.output_dir,
        data_root=args.data_root,
        model_run_id=str(args.model_run_id),
        exchange=str(args.exchange),
        symbol_chunk_size=int(args.symbol_chunk_size),
    )
    write_plan(plan, args.output_dir)
    print(json.dumps(_json_safe(plan), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
