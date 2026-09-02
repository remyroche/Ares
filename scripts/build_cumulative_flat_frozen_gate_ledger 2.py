#!/usr/bin/env python3
"""Build a cumulative flat candidate ledger for frozen A/B gate evaluation.

The frozen contextual TP/SL gate needs enough post-freeze rows to make accepted
trade and tail metrics meaningful.  Prospective rows may arrive in several
small ledgers and may not all carry generated reliability diagnostics.  This
script merges compatible flat ledgers, de-duplicates decision rows, and
optionally rematerializes the reliability diagnostics over the cumulative
history.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.materialize_flat_contextual_tp_sl_diagnostics import (  # noqa: E402
    CONTRACT_COLUMNS,
    GENERATED_COLUMNS,
    _add_recent_performance_features,
    _add_score_diagnostics,
    _coverage,
    _downcast_float_columns,
    _normalize_side,
)


DEFAULT_CANDIDATES = (
    ROOT
    / "data_perp/reports/contextual_tp_sl_flat_diagnostics_jun28_with_history_20260701/"
    "combo_candidates_history_jun28_with_diagnostics.parquet",
    ROOT / "data_perp/reports/contextual_tp_sl_latest_jun26_28_static_20260701/combo_candidates.parquet",
    ROOT
    / "data_perp/exchanges/krakenfutures/live_state/prediction_ledgers/20260629_050000_lgbm_mda/"
    "prediction_ledger.parquet",
)


def _json_safe(value: Any) -> Any:
    if not isinstance(value, (dict, list, tuple)):
        try:
            missing = pd.isna(value)
        except Exception:
            missing = False
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return val if np.isfinite(val) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _candidate_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(item) for item in (args.candidate or DEFAULT_CANDIDATES)]
    for root in args.root or []:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for pattern in ("**/combo_candidates*.parquet", "**/*with_diagnostics*.parquet"):
            paths.extend(root_path.glob(pattern))
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _read_candidate(path: Path, source_order: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        return pd.DataFrame(), {"path": str(path), "status": "missing"}
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        return pd.DataFrame(), {"path": str(path), "status": f"read_error:{type(exc).__name__}:{exc}"}
    if frame.empty:
        return frame, {"path": str(path), "status": "empty", "rows": 0}
    missing = sorted({"timestamp", "strategy_id", "symbol"} - set(frame.columns))
    if missing:
        return pd.DataFrame(), {
            "path": str(path),
            "status": "missing_required_columns",
            "missing_required_columns": missing,
            "rows": int(len(frame)),
        }
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame[frame["timestamp"].notna()].copy()
    if "side" not in frame.columns:
        frame["side"] = ""
    else:
        frame["side"] = frame["side"].map(_normalize_side)
    frame["_source_path"] = str(path)
    frame["_source_order"] = int(source_order)
    generated_present = [col for col in GENERATED_COLUMNS if col in frame.columns]
    if generated_present:
        diag_finite = frame[generated_present].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1)
    else:
        diag_finite = pd.Series(0, index=frame.index)
    frame["_diagnostic_finite_count"] = diag_finite.astype("int16")
    return frame, {
        "path": str(path),
        "status": "loaded",
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "timestamp_min": frame["timestamp"].min().isoformat() if len(frame) else "",
        "timestamp_max": frame["timestamp"].max().isoformat() if len(frame) else "",
        "generated_columns_present": int(len(generated_present)),
    }


def _dedupe(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["timestamp", "strategy_id", "symbol", "side"]
    before = int(len(frame))
    duplicate_rows = int(frame.duplicated(keys).sum())
    if duplicate_rows:
        frame = frame.sort_values(keys + ["_diagnostic_finite_count", "_source_order"]).drop_duplicates(
            keys,
            keep="last",
        )
    frame = frame.sort_values(["timestamp", "strategy_id", "symbol", "side"]).reset_index(drop=True)
    return frame, {
        "dedupe_keys": keys,
        "rows_before_dedupe": before,
        "duplicate_rows": duplicate_rows,
        "rows_after_dedupe": int(len(frame)),
    }


def _post_cutoff_summary(frame: pd.DataFrame, cutoff: str) -> dict[str, Any]:
    cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
    mask = frame["timestamp"].ge(cutoff_ts)
    post = frame.loc[mask]
    heads = post["strategy_id"].astype(str).str.extract(
        r"^(short_bollinger|short_boll|long_bars|long_dist|short_asset)",
        expand=False,
    ).replace({"short_boll": "short_bollinger"})
    return {
        "cutoff": cutoff_ts.isoformat(),
        "post_cutoff_rows": int(len(post)),
        "post_cutoff_timestamps": int(post["timestamp"].nunique()),
        "post_cutoff_active_heads": int(heads.dropna().nunique()),
        "post_cutoff_timestamp_min": post["timestamp"].min().isoformat() if len(post) else "",
        "post_cutoff_timestamp_max": post["timestamp"].max().isoformat() if len(post) else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=None, help="Flat candidate parquet. Repeatable.")
    parser.add_argument("--root", action="append", default=None, help="Optional root to scan for flat candidate ledgers.")
    parser.add_argument("--cutoff", default="2026-06-27T00:00:00+00:00")
    parser.add_argument(
        "--rematerialize-diagnostics",
        action="store_true",
        help="Recompute generated reliability diagnostics over the merged cumulative ledger.",
    )
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    source_rows: list[dict[str, Any]] = []
    for idx, path in enumerate(_candidate_paths(args)):
        frame, row = _read_candidate(path, idx)
        source_rows.append(row)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("No usable candidate ledgers were loaded")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined, dedupe_meta = _dedupe(combined)
    remat_meta: dict[str, Any] = {"enabled": bool(args.rematerialize_diagnostics)}
    if args.rematerialize_diagnostics:
        combined = _add_score_diagnostics(combined, overwrite=True)
        combined, perf_meta = _add_recent_performance_features(combined, overwrite=True)
        _downcast_float_columns(combined, GENERATED_COLUMNS)
        remat_meta["performance_feature_status"] = perf_meta

    source_columns = ["_source_path", "_source_order", "_diagnostic_finite_count"]
    output = combined.drop(columns=[col for col in source_columns if col in combined.columns], errors="ignore")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)

    source_table = pd.DataFrame(source_rows)
    source_table.to_csv(args.report_dir / "cumulative_flat_ledger_sources.csv", index=False)
    coverage = _coverage(output)
    coverage.to_csv(args.report_dir / "cumulative_flat_ledger_diagnostic_coverage.csv", index=False)
    post = _post_cutoff_summary(output, str(args.cutoff))
    manifest = {
        "generated_by": Path(__file__).name,
        "output": str(args.output),
        "rows": int(len(output)),
        "columns": int(len(output.columns)),
        "timestamp_min": output["timestamp"].min().isoformat() if len(output) else "",
        "timestamp_max": output["timestamp"].max().isoformat() if len(output) else "",
        "sources": source_rows,
        "dedupe": dedupe_meta,
        "diagnostic_rematerialization": remat_meta,
        "post_cutoff": post,
        "missing_contract_columns": [col for col in CONTRACT_COLUMNS if col not in output.columns],
    }
    (args.report_dir / "cumulative_flat_ledger_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Cumulative Flat Frozen-Gate Ledger",
        "",
        f"Output: `{args.output}`",
        f"Rows: `{len(output)}`",
        f"Period: `{manifest['timestamp_min']}` to `{manifest['timestamp_max']}`",
        f"Rematerialized diagnostics: `{bool(args.rematerialize_diagnostics)}`",
        "",
        "## Post-Cutoff Summary",
        "",
        pd.DataFrame([post]).to_markdown(index=False),
        "",
        "## Dedupe",
        "",
        pd.DataFrame([dedupe_meta]).to_markdown(index=False),
        "",
        "## Sources",
        "",
        source_table.to_markdown(index=False) if not source_table.empty else "_No sources._",
        "",
        "## Diagnostic Coverage",
        "",
        coverage.to_markdown(index=False),
    ]
    (args.report_dir / "cumulative_flat_ledger_report.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
