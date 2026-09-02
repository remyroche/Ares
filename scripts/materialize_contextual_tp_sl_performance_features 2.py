#!/usr/bin/env python3
"""Backfill causal recent-performance diagnostics into contextual TP/SL tables.

Some contextual TP/SL candidate sources contain drift/OOD/uncertainty columns
but not the recent hit-rate surprise columns used by the `performance`
diagnostic family. This script copies a source directory into a new source
directory and adds the missing performance columns to each available candidate
arm under `portfolio_replay/`.

The generated columns are causal within the table: a decision row at time t only
receives performance events whose `exit_timestamp <= t`, grouped by strategy.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


DEFAULT_ARMS = (
    "static",
    "rank_only",
    "performance_only",
    "joint_all",
    "independent_all",
    "best_by_head",
)
RECENT_PERFORMANCE_FEATURES = (
    "generated_hr_surprise_24",
    "generated_hr_surprise_96",
    "generated_weighted_hr_surprise_24",
    "generated_weighted_hr_surprise_96",
    "generated_loss_rate_24",
    "generated_loss_rate_96",
    "generated_matured_count_24",
    "generated_matured_count_96",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _score_series(rows: pd.DataFrame) -> pd.Series:
    for col in (
        "reliability_blend_score",
        "calibrated_score",
        "normalized_rank_score",
        "rank_pct",
        "strategy_rank_pct",
    ):
        if col in rows.columns:
            score = pd.to_numeric(rows[col], errors="coerce")
            if score.notna().any():
                return score.clip(0.0, 1.0)
    return pd.Series(0.5, index=rows.index, dtype="float64")


def _add_recent_performance_features(rows: pd.DataFrame, *, overwrite: bool) -> tuple[pd.DataFrame, Dict[str, Any]]:
    out = rows.copy()
    existing = [col for col in RECENT_PERFORMANCE_FEATURES if col in out.columns]
    if existing and not overwrite:
        return out, {
            "status": "skipped_existing_columns",
            "existing_feature_count": int(len(existing)),
            "added_feature_count": 0,
            "event_count": None,
        }
    missing_base = sorted({"timestamp", "strategy_id", "exit_timestamp", "net_return"} - set(out.columns))
    if missing_base:
        for col in RECENT_PERFORMANCE_FEATURES:
            if overwrite or col not in out.columns:
                out[col] = np.nan
        return out, {
            "status": "missing_base_columns",
            "missing_base_columns": missing_base,
            "existing_feature_count": int(len(existing)),
            "added_feature_count": int(len(RECENT_PERFORMANCE_FEATURES) - len(existing)),
            "event_count": 0,
        }

    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["exit_timestamp"] = pd.to_datetime(out["exit_timestamp"], utc=True, errors="coerce")
    score = _score_series(out).astype("float64")
    events = pd.DataFrame(
        {
            "strategy_id": out["strategy_id"].astype(str).to_numpy(),
            "event_ts": out["exit_timestamp"].to_numpy(),
            "win": (pd.to_numeric(out["net_return"], errors="coerce") > 0.0).astype(float).to_numpy(),
            "expected": score.to_numpy(dtype=float),
        }
    )
    events = events.dropna(subset=["event_ts"]).sort_values(["strategy_id", "event_ts"]).reset_index(drop=True)
    if events.empty:
        for col in RECENT_PERFORMANCE_FEATURES:
            if overwrite or col not in out.columns:
                out[col] = np.nan
        return out, {
            "status": "no_matured_events",
            "existing_feature_count": int(len(existing)),
            "added_feature_count": int(len(RECENT_PERFORMANCE_FEATURES) - len(existing)),
            "event_count": 0,
        }

    events["error"] = events["win"] - events["expected"]
    events["weighted_error"] = events["error"] * events["expected"].clip(0.0, 1.0)
    feature_frames: List[pd.DataFrame] = []
    for strategy_id, group in events.groupby("strategy_id", sort=False):
        g = group.sort_values("event_ts").copy()
        for window in (24, 96):
            min_periods = max(4, min(window // 4, 12))
            g[f"generated_hr_surprise_{window}"] = g["error"].rolling(window, min_periods=min_periods).mean()
            g[f"generated_weighted_hr_surprise_{window}"] = (
                g["weighted_error"].rolling(window, min_periods=min_periods).mean()
            )
            g[f"generated_loss_rate_{window}"] = (1.0 - g["win"]).rolling(window, min_periods=min_periods).mean()
            g[f"generated_matured_count_{window}"] = (
                g["win"].rolling(window, min_periods=1).count().clip(upper=window) / float(window)
            )
        feature_frames.append(g[["strategy_id", "event_ts", *RECENT_PERFORMANCE_FEATURES]])

    perf = pd.concat(feature_frames, ignore_index=True).sort_values(["strategy_id", "event_ts"])
    pieces: List[pd.DataFrame] = []
    original = out.reset_index().rename(columns={"index": "_row_id"})
    for strategy_id, group in original.groupby("strategy_id", sort=False):
        left = group.sort_values("timestamp")
        right = perf.loc[perf["strategy_id"].eq(strategy_id)].sort_values("event_ts")
        if right.empty:
            for col in RECENT_PERFORMANCE_FEATURES:
                left[col] = np.nan
            pieces.append(left)
            continue
        merged = pd.merge_asof(
            left,
            right.drop(columns=["strategy_id"]),
            left_on="timestamp",
            right_on="event_ts",
            direction="backward",
            allow_exact_matches=True,
        ).drop(columns=["event_ts"], errors="ignore")
        pieces.append(merged)

    out = pd.concat(pieces, ignore_index=True).sort_values("_row_id").drop(columns=["_row_id"])
    for col in RECENT_PERFORMANCE_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    finite_rates = {
        col: float(pd.to_numeric(out[col], errors="coerce").notna().mean())
        for col in RECENT_PERFORMANCE_FEATURES
    }
    return out.reset_index(drop=True), {
        "status": "materialized",
        "existing_feature_count": int(len(existing)),
        "added_feature_count": int(len([col for col in RECENT_PERFORMANCE_FEATURES if col not in existing])),
        "event_count": int(len(events)),
        "finite_rates": finite_rates,
    }


def _copy_non_portfolio_files(source_dir: Path, out_dir: Path) -> None:
    for item in source_dir.iterdir():
        if item.name == "portfolio_replay":
            continue
        dest = out_dir / item.name
        if item.is_file():
            shutil.copy2(item, dest)
        elif item.is_dir() and not dest.exists():
            shutil.copytree(item, dest, ignore=shutil.ignore_patterns("*.parquet"))


def _parse_arms(values: Sequence[str]) -> List[str]:
    arms: List[str] = []
    for raw in values:
        for part in str(raw).split(","):
            arm = part.strip()
            if arm:
                arms.append(arm)
    return arms or list(DEFAULT_ARMS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arm", action="append", default=[], help="Arm to materialize. Repeatable or comma-separated.")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--copy-sidecars", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    out_dir = args.out_dir.resolve()
    if source_dir == out_dir:
        raise ValueError("--out-dir must differ from --source-dir")
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    out_portfolio = out_dir / "portfolio_replay"
    out_portfolio.mkdir(parents=True, exist_ok=True)
    if args.copy_sidecars:
        _copy_non_portfolio_files(source_dir, out_dir)

    rows: List[Dict[str, Any]] = []
    for arm in _parse_arms(args.arm):
        in_path = source_dir / "portfolio_replay" / f"{arm}_contextual_tp_sl_candidates.parquet"
        out_path = out_portfolio / f"{arm}_contextual_tp_sl_candidates.parquet"
        if not in_path.exists():
            rows.append({"arm": arm, "status": "missing_input", "input_path": str(in_path)})
            continue
        frame = pd.read_parquet(in_path)
        materialized, meta = _add_recent_performance_features(frame, overwrite=bool(args.overwrite_existing))
        materialized.to_parquet(out_path, index=False)
        rows.append(
            {
                "arm": arm,
                "status": meta.get("status"),
                "input_path": str(in_path),
                "output_path": str(out_path),
                "rows": int(len(materialized)),
                "columns": int(len(materialized.columns)),
                **{k: v for k, v in meta.items() if k != "status"},
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "performance_feature_materialization_summary.csv", index=False)
    manifest = {
        "generated_by": "materialize_contextual_tp_sl_performance_features",
        "source_dir": str(source_dir),
        "out_dir": str(out_dir),
        "overwrite_existing": bool(args.overwrite_existing),
        "copy_sidecars": bool(args.copy_sidecars),
        "performance_features": list(RECENT_PERFORMANCE_FEATURES),
        "arms": rows,
    }
    (out_dir / "performance_feature_materialization_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Contextual TP/SL Performance Feature Materialization",
        "",
        f"Source: `{source_dir}`",
        f"Output: `{out_dir}`",
        "",
        summary.to_markdown(index=False) if not summary.empty else "_No arms processed._",
        "",
    ]
    (out_dir / "performance_feature_materialization_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(out_dir / "performance_feature_materialization_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
