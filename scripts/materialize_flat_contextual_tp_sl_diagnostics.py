#!/usr/bin/env python3
"""Materialize causal contextual TP/SL diagnostics for flat candidate ledgers.

The frozen wf_recent smooth-penalty challenger expects generated diagnostics
for uncertainty, score drift, strategy-level OOD, recent hit-rate surprise, and
execution friction.  Some prospective candidate ledgers are flat parquet files
that contain the raw replay columns but not these generated diagnostics.  This
script adds them without running a replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


UNCERTAINTY_COLUMNS = (
    "generated_score_uncertainty_p1mp",
    "generated_score_entropy",
    "generated_score_abs_distance_from_half",
)
DRIFT_COLUMNS = (
    "generated_score_abs_diff_1",
    "generated_score_abs_diff_4",
    "generated_score_abs_diff_24",
    "generated_score_abs_minus_prev24_mean",
    "generated_score_prev24_std",
    "generated_strategy_score_shift_abs_z",
)
OOD_COLUMNS = (
    "generated_strategy_score_ood_abs_z",
    "generated_strategy_barrier_ood_abs_z",
    "generated_strategy_friction_ood_abs_z",
)
RECENT_PERFORMANCE_COLUMNS = (
    "generated_hr_surprise_24",
    "generated_hr_surprise_96",
    "generated_weighted_hr_surprise_24",
    "generated_weighted_hr_surprise_96",
    "generated_loss_rate_24",
    "generated_loss_rate_96",
    "generated_matured_count_24",
    "generated_matured_count_96",
)
FRICTION_COLUMNS = (
    "expected_friction_bps",
    "price_gap_bps",
    "entry_gap_bps",
    "entry_slippage_proxy_bps",
    "orderbook_slippage_bps",
    "delay_max_adverse_bps",
    "liquidity_capacity_weight",
)
GENERATED_COLUMNS = (
    *UNCERTAINTY_COLUMNS,
    *DRIFT_COLUMNS,
    *OOD_COLUMNS,
    *RECENT_PERFORMANCE_COLUMNS,
)
CONTRACT_COLUMNS = (
    *UNCERTAINTY_COLUMNS,
    *DRIFT_COLUMNS,
    *OOD_COLUMNS,
    "generated_hr_surprise_24",
    "generated_hr_surprise_96",
    "generated_weighted_hr_surprise_24",
    "generated_weighted_hr_surprise_96",
    "generated_loss_rate_24",
    "generated_loss_rate_96",
    *FRICTION_COLUMNS,
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
        "contextual_tp_sl_score",
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


def _normalize_side(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"short", "sell", "-1", "-1.0"} or text.startswith("short"):
        return "short"
    if text in {"long", "buy", "1", "1.0"} or text.startswith("long"):
        return "long"
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric) and float(numeric) < 0.0:
        return "short"
    return "long"


def _rolling_shifted_z(values: pd.Series, group_key: pd.Series, *, window: int) -> pd.Series:
    grouped = values.groupby(group_key, sort=False)
    mean = grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=8).mean())
    std = grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=8).std())
    return ((values - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _add_score_diagnostics(rows: pd.DataFrame, *, overwrite: bool) -> pd.DataFrame:
    out = rows.copy()
    score = _score_series(out).astype("float64")
    eps = 1e-6
    clipped = score.clip(eps, 1.0 - eps)
    if overwrite or "generated_score_uncertainty_p1mp" not in out.columns:
        out["generated_score_uncertainty_p1mp"] = (clipped * (1.0 - clipped)).astype("float32")
    if overwrite or "generated_score_entropy" not in out.columns:
        out["generated_score_entropy"] = (
            -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
        ).astype("float32")
    if overwrite or "generated_score_abs_distance_from_half" not in out.columns:
        out["generated_score_abs_distance_from_half"] = (score - 0.5).abs().astype("float32")

    original_order = out.reset_index().rename(columns={"index": "_row_id"})
    out = original_order.sort_values(["strategy_id", "symbol", "timestamp"]).copy()
    score = _score_series(out).astype("float64")
    by_symbol = out.groupby(["strategy_id", "symbol"], sort=False)
    for lag in (1, 4, 24):
        col = f"generated_score_abs_diff_{lag}"
        if overwrite or col not in out.columns:
            diff = score - by_symbol[score.name].shift(lag)
            out[col] = pd.to_numeric(diff, errors="coerce").abs().astype("float32")

    if overwrite or "generated_score_abs_minus_prev24_mean" not in out.columns:
        prev24_mean = by_symbol[score.name].transform(lambda s: s.shift(1).rolling(24, min_periods=6).mean())
        out["generated_score_abs_minus_prev24_mean"] = (score - prev24_mean).abs().astype("float32")
    if overwrite or "generated_score_prev24_std" not in out.columns:
        prev24_std = by_symbol[score.name].transform(lambda s: s.shift(1).rolling(24, min_periods=6).std())
        out["generated_score_prev24_std"] = prev24_std.astype("float32")

    strategy_key = out["strategy_id"].astype(str)
    if overwrite or "generated_strategy_score_shift_abs_z" not in out.columns:
        out["generated_strategy_score_shift_abs_z"] = _rolling_shifted_z(
            score, strategy_key, window=96
        ).abs().astype("float32")
    if overwrite or "generated_strategy_score_ood_abs_z" not in out.columns:
        out["generated_strategy_score_ood_abs_z"] = _rolling_shifted_z(
            score, strategy_key, window=384
        ).abs().astype("float32")
    if overwrite or "generated_strategy_barrier_ood_abs_z" not in out.columns:
        barrier = pd.to_numeric(
            out.get("policy_effective_barrier_pct", out.get("barrier_pct", np.nan)),
            errors="coerce",
        )
        out["generated_strategy_barrier_ood_abs_z"] = _rolling_shifted_z(
            barrier, strategy_key, window=384
        ).abs().astype("float32")
    if overwrite or "generated_strategy_friction_ood_abs_z" not in out.columns:
        friction = pd.to_numeric(out.get("expected_friction_bps", np.nan), errors="coerce")
        out["generated_strategy_friction_ood_abs_z"] = _rolling_shifted_z(
            friction, strategy_key, window=384
        ).abs().astype("float32")

    return out.sort_values("_row_id").drop(columns=["_row_id"]).reset_index(drop=True)


def _add_recent_performance_features(rows: pd.DataFrame, *, overwrite: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = rows.copy()
    if overwrite:
        out = out.drop(columns=[col for col in RECENT_PERFORMANCE_COLUMNS if col in out.columns], errors="ignore")
    missing = sorted({"timestamp", "strategy_id", "exit_timestamp", "net_return"} - set(out.columns))
    if missing:
        for col in RECENT_PERFORMANCE_COLUMNS:
            if overwrite or col not in out.columns:
                out[col] = np.nan
        return out, {"status": "missing_base_columns", "missing_base_columns": missing, "event_count": 0}

    out["exit_timestamp"] = pd.to_datetime(out["exit_timestamp"], utc=True, errors="coerce")
    score = _score_series(out).astype("float64")
    events = pd.DataFrame(
        {
            "strategy_id": out["strategy_id"].astype(str).to_numpy(),
            "event_ts": out["exit_timestamp"].to_numpy(),
            "win": (pd.to_numeric(out["net_return"], errors="coerce") > 0.0).astype(float).to_numpy(),
            "expected": score.to_numpy(dtype=float),
        }
    ).dropna(subset=["event_ts"])
    events = events.sort_values(["strategy_id", "event_ts"]).reset_index(drop=True)
    if events.empty:
        for col in RECENT_PERFORMANCE_COLUMNS:
            if overwrite or col not in out.columns:
                out[col] = np.nan
        return out, {"status": "no_matured_events", "event_count": 0}

    events["error"] = events["win"] - events["expected"]
    events["weighted_error"] = events["error"] * events["expected"].clip(0.0, 1.0)
    frames: list[pd.DataFrame] = []
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
        frames.append(g[["strategy_id", "event_ts", *RECENT_PERFORMANCE_COLUMNS]])

    perf = pd.concat(frames, ignore_index=True).sort_values(["strategy_id", "event_ts"])
    original = out.reset_index().rename(columns={"index": "_row_id"})
    pieces: list[pd.DataFrame] = []
    for strategy_id, group in original.groupby("strategy_id", sort=False):
        left = group.sort_values("timestamp")
        right = perf.loc[perf["strategy_id"].eq(strategy_id)].sort_values("event_ts")
        if right.empty:
            for col in RECENT_PERFORMANCE_COLUMNS:
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
    for col in RECENT_PERFORMANCE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out.reset_index(drop=True), {"status": "materialized", "event_count": int(len(events))}


def _coverage(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = {
        "uncertainty": UNCERTAINTY_COLUMNS,
        "drift": DRIFT_COLUMNS,
        "ood": OOD_COLUMNS,
        "recent_hit_rate_surprise": RECENT_PERFORMANCE_COLUMNS,
        "friction": FRICTION_COLUMNS,
    }
    for group, columns in groups.items():
        for col in columns:
            if col in frame.columns:
                values = pd.to_numeric(frame[col], errors="coerce")
                present = True
                finite_rate = float(np.isfinite(values.to_numpy(dtype=float, copy=False)).mean()) if len(values) else 0.0
                nonzero_rate = float((values.fillna(0.0).to_numpy(dtype=float, copy=False) != 0.0).mean()) if len(values) else 0.0
            else:
                present = False
                finite_rate = 0.0
                nonzero_rate = 0.0
            rows.append(
                {
                    "group": group,
                    "column": col,
                    "present": bool(present),
                    "finite_rate": finite_rate,
                    "nonzero_rate": nonzero_rate,
                }
            )
    return pd.DataFrame(rows)


def _downcast_float_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--history-candidates",
        type=Path,
        default=None,
        help="Optional earlier candidate ledger prepended before materializing shifted diagnostics.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.candidates)
    input_rows = int(len(frame))
    history_rows = 0
    if args.history_candidates is not None:
        history = pd.read_parquet(args.history_candidates)
        history_rows = int(len(history))
        frame = pd.concat([history, frame], ignore_index=True, sort=False)
    missing_required = sorted({"timestamp", "strategy_id", "symbol"} - set(frame.columns))
    if missing_required:
        raise ValueError(f"{args.candidates} missing required columns: {missing_required}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame[frame["timestamp"].notna()].sort_values(["timestamp", "strategy_id", "symbol"]).reset_index(drop=True)
    if "side" in frame.columns:
        frame["side"] = frame["side"].map(_normalize_side)

    frame = _add_score_diagnostics(frame, overwrite=bool(args.overwrite_existing))
    frame, perf_meta = _add_recent_performance_features(frame, overwrite=bool(args.overwrite_existing))
    _downcast_float_columns(frame, GENERATED_COLUMNS)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)

    coverage = _coverage(frame)
    coverage.to_csv(args.report_dir / "diagnostic_feature_coverage.csv", index=False)
    manifest = {
        "generated_by": "materialize_flat_contextual_tp_sl_diagnostics",
        "input": str(args.candidates),
        "history_input": str(args.history_candidates) if args.history_candidates is not None else "",
        "input_rows": input_rows,
        "history_rows": history_rows,
        "output": str(args.output),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "timestamp_min": frame["timestamp"].min().isoformat() if len(frame) else "",
        "timestamp_max": frame["timestamp"].max().isoformat() if len(frame) else "",
        "performance_feature_status": perf_meta,
        "missing_contract_columns": [col for col in CONTRACT_COLUMNS if col not in frame.columns],
    }
    (args.report_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")
    lines = [
        "# Flat Contextual TP/SL Diagnostic Materialization",
        "",
        f"Input: `{args.candidates}`",
        f"History input: `{args.history_candidates}`",
        f"Output: `{args.output}`",
        f"Rows: `{len(frame)}` (`{history_rows}` history + `{input_rows}` input before timestamp cleanup)",
        f"Period: `{manifest['timestamp_min']}` to `{manifest['timestamp_max']}`",
        "",
        "## Coverage",
        "",
        coverage.to_markdown(index=False),
    ]
    (args.report_dir / "diagnostic_materialization_report.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
