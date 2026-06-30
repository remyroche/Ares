#!/usr/bin/env python3
"""Build a direct accepted-trade suppression training ledger.

The threshold controller failed because replay/action effects looked useful
while direct suppression of baseline-accepted losers was absent or not
recurrent.  This script materializes the row-level target surface needed to
train the next controller on the right problem:

    baseline-accepted frontier row + state/response schedule context
    -> should raising the strategy threshold suppress this row?

It is read-only with respect to model artifacts.  It does not select,
promote, score, or execute a controller.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")

from scripts import run_market_state_threshold_controller as mstc  # noqa: E402


DEFAULT_SOURCE_DIR = Path(
    "data_perp/reports/market_state_threshold_controller_walkforward_globalrank_no_backfill_20260627_v1"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_direct_suppression_ledger")
BASELINE_ARM = "S0_baseline_static_thresholds"
ACCEPTED_ARM_MODES = {"filter_baseline_arm", "all_accepted_as_baseline"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
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


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _safe_num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _rank_score(base: pd.DataFrame) -> tuple[pd.Series, str]:
    effective = _safe_num(base, "effective_rank_score")
    if effective.notna().any():
        fallback = _safe_num(base, "normalized_rank_score")
        return effective.fillna(fallback).clip(0.0, 1.01), "effective_rank_score"
    return _safe_num(base, "normalized_rank_score").clip(0.0, 1.01), "normalized_rank_score"


def _prepare_baseline_accepted(
    accepted: pd.DataFrame,
    baseline_arm: str,
    *,
    accepted_arm_mode: str = "filter_baseline_arm",
) -> pd.DataFrame:
    if accepted_arm_mode not in ACCEPTED_ARM_MODES:
        raise ValueError(
            f"accepted_arm_mode must be one of {sorted(ACCEPTED_ARM_MODES)}, "
            f"got {accepted_arm_mode!r}"
        )
    if accepted_arm_mode == "filter_baseline_arm":
        if "arm" not in accepted.columns:
            raise KeyError("accepted_trades is missing arm")
        base = accepted.loc[accepted["arm"].astype(str).eq(str(baseline_arm))].copy()
    else:
        base = accepted.copy()
    if base.empty:
        return base
    base["source_accepted_arm"] = (
        base["arm"].astype(str) if "arm" in base.columns else "missing_arm"
    )
    base["accepted_arm_mode"] = accepted_arm_mode
    base = mstc._trade_outcome_flags(base)
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True, errors="coerce")
    rank, source = _rank_score(base)
    base["rank_score"] = rank
    base["rank_score_source"] = source
    base["decision_key"] = (
        base["timestamp"].astype(str)
        + "|"
        + base["symbol"].astype(str)
        + "|"
        + base["side"].astype(str)
        + "|"
        + base["strategy_id"].astype(str)
    )
    return base


def _schedule_context(
    schedules: pd.DataFrame,
    baseline_arm: str,
    *,
    controller_arm_fallback: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if schedules.empty:
        return schedules.copy(), {"controller_arm_source": "empty_schedule"}
    required = {"timestamp", "strategy_id", "base_threshold", "state_threshold"}
    if "arm" not in schedules.columns and not controller_arm_fallback:
        required.add("arm")
    missing = sorted(required - set(schedules.columns))
    if missing:
        raise KeyError(f"strategy_threshold_schedule is missing columns: {missing}")
    if "arm" in schedules.columns:
        sched = schedules.loc[~schedules["arm"].astype(str).eq(str(baseline_arm))].copy()
        controller_arm_source = "schedule_arm"
    else:
        sched = schedules.copy()
        sched["arm"] = str(controller_arm_fallback)
        controller_arm_source = "fallback_controller_arm"
    sched["timestamp"] = pd.to_datetime(sched["timestamp"], utc=True, errors="coerce")
    sched = sched.rename(
        columns={
            "arm": "controller_arm",
            "head": "schedule_head",
            "base_threshold": "schedule_base_threshold",
            "state_threshold": "schedule_state_threshold",
        }
    )
    return sched, {
        "controller_arm_source": controller_arm_source,
        "controller_arm_fallback": controller_arm_fallback,
    }


def build_direct_suppression_ledger(
    source_dir: Path,
    *,
    baseline_arm: str = BASELINE_ARM,
    accepted_arm_mode: str = "filter_baseline_arm",
    controller_arm_fallback: str | None = None,
    source_kind: str | None = None,
    source_window_id: str | None = None,
    frontier_bandwidth: float = 0.06,
    default_frontier_width: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    accepted = _read_frame(source_dir / "accepted_trades.parquet")
    schedules = _read_frame(source_dir / "strategy_threshold_schedule.parquet")
    base = _prepare_baseline_accepted(
        accepted,
        baseline_arm,
        accepted_arm_mode=accepted_arm_mode,
    )
    sched, schedule_meta = _schedule_context(
        schedules,
        baseline_arm,
        controller_arm_fallback=controller_arm_fallback,
    )
    if base.empty or sched.empty:
        empty = pd.DataFrame()
        summary = {
            "source_dir": str(source_dir),
            "baseline_arm": str(baseline_arm),
            "accepted_arm_mode": accepted_arm_mode,
            "controller_arm_fallback": controller_arm_fallback,
            "source_kind": source_kind,
            "source_window_id": source_window_id,
            "row_count": 0,
            "reason": "missing_baseline_accepted_or_schedules",
        }
        summary.update(schedule_meta)
        return empty, empty, summary

    sched_cols = [
        "controller_arm",
        "timestamp",
        "strategy_id",
        "schedule_head",
        "schedule_base_threshold",
        "schedule_state_threshold",
        "raw_state_threshold",
        "controller_mode",
        "threshold_action_enabled",
        "force_base_threshold",
        "risk_severity",
        "controller_reason",
        "prediction_coverage",
        "min_prediction_coverage",
        "state_ood_score_mean",
        "state_ood_score_max",
        "state_ood_cutoff",
        "state_ood_share",
        "state_low_input_coverage_share",
        "mean_pred_utility",
        "mean_pred_lcb",
        "mean_pred_full_sl",
        "mean_pred_timeout",
        "base_candidate_count",
        "frontier_candidate_count",
        "min_frontier_candidate_count",
        "frontier_upper_rank",
        "tail_candidate_count",
        "suppressed_candidate_count",
        "accepted_frontier_key_filter_active",
        "accepted_frontier_candidate_count",
        "accepted_frontier_suppressed_count",
        "predicted_removed_loss_avoided",
        "predicted_removed_winner_sacrificed",
        "predicted_action_edge",
        "action_edge_per_suppressed",
        "fold",
    ]
    sched = sched[[col for col in sched_cols if col in sched.columns]].copy()
    base_cols = [
        "timestamp",
        "symbol",
        "side",
        "strategy_id",
        "head",
        "rank_score",
        "rank_score_source",
        "_threshold",
        "_net_return",
        "_is_full_sl",
        "_is_timeout",
        "net_return",
        "gross_return",
        "net_pnl",
        "gross_pnl",
        "cost_pnl",
        "simple_policy_exit_reason",
        "position_size",
        "decision_key",
        "source_accepted_arm",
        "accepted_arm_mode",
    ]
    base = base[[col for col in base_cols if col in base.columns]].copy()
    work = base.merge(
        sched,
        on=["timestamp", "strategy_id"],
        how="inner",
        validate="many_to_many",
    )
    if work.empty:
        empty = pd.DataFrame()
        summary = {
            "source_dir": str(source_dir),
            "baseline_arm": str(baseline_arm),
            "accepted_arm_mode": accepted_arm_mode,
            "controller_arm_fallback": controller_arm_fallback,
            "source_kind": source_kind,
            "source_window_id": source_window_id,
            "row_count": 0,
            "reason": "no_matching_schedule_rows",
        }
        summary.update(schedule_meta)
        return empty, empty, summary
    if "head" not in work.columns and "schedule_head" in work.columns:
        work["head"] = work["schedule_head"]
    elif "head" in work.columns and "schedule_head" in work.columns:
        work["head"] = work["head"].where(work["head"].notna(), work["schedule_head"])
    elif "head" not in work.columns:
        work["head"] = work["strategy_id"].astype(str)

    rank = _safe_num(work, "rank_score")
    base_threshold = _safe_num(work, "schedule_base_threshold").fillna(_safe_num(work, "_threshold"))
    state_threshold = _safe_num(work, "schedule_state_threshold")
    frontier_upper = _safe_num(work, "frontier_upper_rank")
    fallback_upper = (base_threshold + float(default_frontier_width)).clip(upper=1.01)
    frontier_upper = frontier_upper.where(frontier_upper.notna(), fallback_upper).clip(
        lower=base_threshold,
        upper=1.01,
    )
    work["base_threshold"] = base_threshold
    work["state_threshold"] = state_threshold
    work["frontier_upper_rank"] = frontier_upper
    work["rank_minus_base_threshold"] = rank - base_threshold
    work["frontier_distance"] = (rank - base_threshold).clip(lower=0.0)
    work["required_threshold_raise_to_suppress"] = (rank - base_threshold).clip(lower=0.0)
    work["in_direct_frontier"] = (rank >= base_threshold) & (rank <= frontier_upper)
    work["would_suppress_at_state_threshold"] = (rank >= base_threshold) & (rank < state_threshold)
    work = work.loc[work["in_direct_frontier"]].copy()
    if work.empty:
        empty = pd.DataFrame()
        summary = {
            "source_dir": str(source_dir),
            "baseline_arm": str(baseline_arm),
            "accepted_arm_mode": accepted_arm_mode,
            "controller_arm_fallback": controller_arm_fallback,
            "source_kind": source_kind,
            "source_window_id": source_window_id,
            "row_count": 0,
            "reason": "no_baseline_accepted_rows_in_direct_frontier",
        }
        summary.update(schedule_meta)
        return empty, empty, summary

    net_return = _safe_num(work, "_net_return").fillna(0.0)
    work["loss_avoided_if_suppressed"] = (-np.minimum(net_return.to_numpy(dtype=float), 0.0)).astype(float)
    work["winner_pnl_sacrificed_if_suppressed"] = np.maximum(net_return.to_numpy(dtype=float), 0.0).astype(float)
    work["direct_defensive_utility"] = (
        work["loss_avoided_if_suppressed"] - work["winner_pnl_sacrificed_if_suppressed"]
    )
    work["direct_suppression_profitable"] = work["direct_defensive_utility"] > 0.0
    work["direct_suppression_full_sl"] = _safe_num(work, "_is_full_sl").fillna(0.0) > 0.5
    work["direct_suppression_timeout"] = _safe_num(work, "_is_timeout").fillna(0.0) > 0.5
    bandwidth = max(float(frontier_bandwidth), 1e-6)
    work["frontier_sample_weight"] = 1.0 + np.exp(
        -work["frontier_distance"].to_numpy(dtype=float) / bandwidth
    )
    work["suppressed_defensive_utility_under_current_schedule"] = np.where(
        work["would_suppress_at_state_threshold"],
        work["direct_defensive_utility"],
        0.0,
    )
    work["artifact_contract"] = "direct_accepted_frontier_training_ledger_v1"
    work["source_walkforward_dir"] = str(source_dir)
    work["source_dir"] = str(source_dir)
    work["source_kind"] = source_kind or "single_source"
    work["source_window_id"] = source_window_id or source_dir.name
    work["controller_arm_source"] = str(schedule_meta.get("controller_arm_source") or "")

    key_cols = ["controller_arm", "decision_key"]
    duplicate_rows = int(work.duplicated(key_cols).sum()) if set(key_cols).issubset(work.columns) else 0
    by_cols = ["controller_arm"]
    if "head" in work.columns:
        by_cols.append("head")
    grouped = work.groupby(by_cols, dropna=False, sort=True)
    by_group = grouped.agg(
        frontier_rows=("decision_key", "count"),
        unique_decision_keys=("decision_key", "nunique"),
        direct_profitable_rate=("direct_suppression_profitable", "mean"),
        full_sl_rate=("direct_suppression_full_sl", "mean"),
        timeout_rate=("direct_suppression_timeout", "mean"),
        mean_direct_defensive_utility=("direct_defensive_utility", "mean"),
        total_direct_defensive_utility=("direct_defensive_utility", "sum"),
        current_schedule_suppressed_rows=("would_suppress_at_state_threshold", "sum"),
        current_schedule_defensive_utility=(
            "suppressed_defensive_utility_under_current_schedule",
            "sum",
        ),
        mean_required_threshold_raise=("required_threshold_raise_to_suppress", "mean"),
        mean_frontier_sample_weight=("frontier_sample_weight", "mean"),
    ).reset_index()
    summary = {
        "generated_by": "build_market_state_direct_suppression_ledger",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(source_dir),
        "baseline_arm": str(baseline_arm),
        "accepted_arm_mode": accepted_arm_mode,
        "controller_arm_fallback": controller_arm_fallback,
        "controller_arm_source": schedule_meta.get("controller_arm_source"),
        "source_kind": source_kind,
        "source_window_id": source_window_id,
        "artifact_contract": "direct_accepted_frontier_training_ledger_v1",
        "row_count": int(len(work)),
        "baseline_accepted_rows": int(len(base)),
        "controller_arm_count": int(work["controller_arm"].nunique()),
        "unique_decision_key_count": int(work["decision_key"].nunique()),
        "duplicate_controller_decision_key_rows": duplicate_rows,
        "direct_profitable_rate": float(work["direct_suppression_profitable"].mean()),
        "full_sl_rate": float(work["direct_suppression_full_sl"].mean()),
        "timeout_rate": float(work["direct_suppression_timeout"].mean()),
        "mean_direct_defensive_utility": float(work["direct_defensive_utility"].mean()),
        "total_direct_defensive_utility": float(work["direct_defensive_utility"].sum()),
        "current_schedule_suppressed_rows": int(work["would_suppress_at_state_threshold"].sum()),
        "current_schedule_defensive_utility": float(
            work["suppressed_defensive_utility_under_current_schedule"].sum()
        ),
        "frontier_bandwidth": float(frontier_bandwidth),
        "default_frontier_width": float(default_frontier_width),
        "rank_score_sources": sorted(set(work["rank_score_source"].astype(str))),
    }
    return work.reset_index(drop=True), by_group, summary


def _render_report(summary: dict[str, Any], by_group: pd.DataFrame) -> str:
    lines = [
        "# Direct Accepted-Frontier Suppression Ledger",
        "",
        f"Source: `{summary.get('source_dir')}`",
        "",
        "This artifact turns the controller problem into a direct training target:",
        "baseline-accepted frontier rows should be suppressed only when their realized direct defensive utility is positive.",
        "",
        "## Summary",
        "",
        f"- Rows: `{summary.get('row_count')}`",
        f"- Unique decision keys: `{summary.get('unique_decision_key_count')}`",
        f"- Controller arms: `{summary.get('controller_arm_count')}`",
        f"- Direct profitable suppression rate: `{summary.get('direct_profitable_rate')}`",
        f"- Full-SL rate: `{summary.get('full_sl_rate')}`",
        f"- Mean direct defensive utility: `{summary.get('mean_direct_defensive_utility')}`",
        f"- Current schedule suppressed rows: `{summary.get('current_schedule_suppressed_rows')}`",
        f"- Current schedule defensive utility: `{summary.get('current_schedule_defensive_utility')}`",
        "",
        "## By Arm / Head",
        "",
        by_group.to_markdown(index=False) if not by_group.empty else "_No grouped metrics._",
        "",
        "## Contract",
        "",
        "- Uses baseline-accepted rows only.",
        "- Supports explicit later-window mode where the scored accepted-trades file is the baseline accepted set.",
        "- Joins only fold-produced threshold schedules and realized trade outcomes.",
        "- Does not change scores, ranks, thresholds, auction order, sizing, or selected controller arm.",
        "- Intended for the next response/controller training step, not for promotion by itself.",
    ]
    return "\n".join(lines) + "\n"


def write_direct_suppression_ledger(
    ledger: pd.DataFrame,
    by_group: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ledger_parquet": output_dir / "direct_accepted_frontier_training_ledger.parquet",
        "ledger_csv": output_dir / "direct_accepted_frontier_training_ledger.csv",
        "by_group_csv": output_dir / "direct_accepted_frontier_training_by_arm_head.csv",
        "summary_json": output_dir / "direct_accepted_frontier_training_summary.json",
        "report_md": output_dir / "direct_accepted_frontier_training_report.md",
    }
    ledger.to_parquet(paths["ledger_parquet"], index=False)
    ledger.to_csv(paths["ledger_csv"], index=False)
    by_group.to_csv(paths["by_group_csv"], index=False)
    summary = {**summary, "outputs": {key: str(path) for key, path in paths.items()}}
    paths["summary_json"].write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    paths["report_md"].write_text(_render_report(summary, by_group), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-arm", default=BASELINE_ARM)
    parser.add_argument(
        "--accepted-arm-mode",
        choices=sorted(ACCEPTED_ARM_MODES),
        default="filter_baseline_arm",
    )
    parser.add_argument("--controller-arm-fallback", default=None)
    parser.add_argument("--source-kind", default=None)
    parser.add_argument("--source-window-id", default=None)
    parser.add_argument("--frontier-bandwidth", type=float, default=0.06)
    parser.add_argument("--default-frontier-width", type=float, default=0.10)
    args = parser.parse_args()

    ledger, by_group, summary = build_direct_suppression_ledger(
        args.source_dir,
        baseline_arm=args.baseline_arm,
        accepted_arm_mode=args.accepted_arm_mode,
        controller_arm_fallback=args.controller_arm_fallback,
        source_kind=args.source_kind,
        source_window_id=args.source_window_id,
        frontier_bandwidth=float(args.frontier_bandwidth),
        default_frontier_width=float(args.default_frontier_width),
    )
    outputs = write_direct_suppression_ledger(ledger, by_group, summary, args.output_dir)
    print(json.dumps(_json_safe({**summary, "outputs": outputs}), indent=2))


if __name__ == "__main__":
    main()
