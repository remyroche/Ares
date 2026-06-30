#!/usr/bin/env python3
"""Diagnose why a no-backfill market-state threshold overlay failed.

The no-backfill overlay is intentionally conservative: it can only remove
baseline-accepted trades and cannot add replacements.  A negative replay can
still happen when threshold raises change the path/capacity state of common
trades or when an accepted winner is removed indirectly.  This report separates
direct threshold suppressions from indirect path/capacity suppressions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_no_backfill_failure_diagnostics")
REMOVED_ACTION = "removed_by_shadow_no_backfill"
COMMON_ACTION = "common_accepted"
STATE_PREFIX = "state_"
EPS = 1e-12


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _num(series: pd.Series | float | int | None, default: float = 0.0) -> float:
    if series is None:
        return default
    try:
        value = float(series)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def _sum_positive(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return float(numeric.clip(lower=0.0).sum())


def _sum_negative_as_positive(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return float((-numeric.clip(upper=0.0)).sum())


def _bool_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(False).astype(bool)


def _period_from_manifest(manifest: dict[str, Any], score_dir: Path) -> tuple[str | None, str | None]:
    candidates_path = manifest.get("eval_candidates")
    if candidates_path:
        path = Path(str(candidates_path))
        if path.exists():
            timestamps = _to_utc(pd.read_parquet(path, columns=["timestamp"])["timestamp"])
            if not timestamps.dropna().empty:
                return timestamps.min().isoformat(), timestamps.max().isoformat()
    delta = _read_frame(score_dir / "shadow_no_backfill_accepted_trade_delta.csv")
    if "timestamp" in delta.columns and not delta.empty:
        timestamps = _to_utc(delta["timestamp"])
        if not timestamps.dropna().empty:
            return timestamps.min().isoformat(), timestamps.max().isoformat()
    return None, None


def _schedule_columns(schedule: pd.DataFrame) -> list[str]:
    wanted = [
        "timestamp",
        "strategy_id",
        "head",
        "base_threshold",
        "state_threshold",
        "raw_state_threshold",
        "risk_severity",
        "controller_reason",
        "prediction_coverage",
        "state_ood_score_mean",
        "state_ood_score_max",
        "state_ood_share",
        "state_low_input_coverage_share",
        "mean_pred_utility",
        "mean_pred_lcb",
        "mean_pred_full_sl",
        "mean_pred_timeout",
        "base_candidate_count",
        "frontier_candidate_count",
        "tail_candidate_count",
        "suppressed_candidate_count",
        "predicted_removed_loss_avoided",
        "predicted_removed_winner_sacrificed",
        "predicted_action_edge",
        "action_edge_per_suppressed",
    ]
    return [column for column in wanted if column in schedule.columns]


def _prepare_removed_rows(score_dir: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    delta = _read_frame(score_dir / "shadow_no_backfill_accepted_trade_delta.csv")
    if delta.empty:
        return pd.DataFrame()
    if "delta_action" not in delta.columns:
        raise ValueError(f"{score_dir} accepted-trade delta is missing delta_action")
    delta["timestamp"] = _to_utc(delta["timestamp"])
    removed = delta[delta["delta_action"].eq(REMOVED_ACTION)].copy()
    if removed.empty:
        return removed

    schedule = _read_frame(score_dir / "shadow_controller_proposed_schedule.csv")
    if not schedule.empty:
        schedule["timestamp"] = _to_utc(schedule["timestamp"])
        schedule = schedule[_schedule_columns(schedule)].copy()
        schedule = schedule.rename(
            columns={
                "base_threshold": "schedule_base_threshold",
                "state_threshold": "schedule_state_threshold",
                "raw_state_threshold": "schedule_raw_state_threshold",
            }
        )
        removed = removed.merge(
            schedule,
            on=["timestamp", "strategy_id", "head"],
            how="left",
        )

    state_panel = _read_frame(score_dir / "market_state_timestamp_panel.parquet")
    state_columns: list[str] = []
    if not state_panel.empty and "timestamp" in state_panel.columns:
        state_panel["timestamp"] = _to_utc(state_panel["timestamp"])
        state_columns = [
            column
            for column in state_panel.columns
            if column.startswith(STATE_PREFIX) and column != "state_level"
        ]
        removed = removed.merge(
            state_panel[["timestamp", *state_columns]].drop_duplicates("timestamp"),
            on="timestamp",
            how="left",
        )

    rank = pd.to_numeric(
        removed.get("normalized_rank_score", removed.get("effective_rank_score")),
        errors="coerce",
    )
    base_threshold = pd.to_numeric(
        removed.get("schedule_base_threshold", removed.get("base_threshold")),
        errors="coerce",
    )
    state_threshold = pd.to_numeric(
        removed.get("schedule_state_threshold", removed.get("dynamic_threshold")),
        errors="coerce",
    )
    removed["rank_score_for_threshold"] = rank
    removed["base_threshold_for_threshold"] = base_threshold
    removed["state_threshold_for_threshold"] = state_threshold
    removed["threshold_raise"] = state_threshold - base_threshold
    removed["rank_margin_above_base"] = rank - base_threshold
    removed["rank_margin_to_state_threshold"] = rank - state_threshold
    removed["direct_threshold_suppression"] = (
        rank.notna()
        & state_threshold.notna()
        & base_threshold.notna()
        & (rank + EPS >= base_threshold)
        & (rank + EPS < state_threshold)
    )
    removed["indirect_path_or_capacity_suppression"] = ~removed[
        "direct_threshold_suppression"
    ].fillna(False)
    net_pnl = pd.to_numeric(removed.get("net_pnl"), errors="coerce").fillna(0.0)
    removed["loss_avoided"] = (-net_pnl.clip(upper=0.0)).astype(float)
    removed["winner_pnl_sacrificed"] = net_pnl.clip(lower=0.0).astype(float)
    removed["defensive_success"] = removed["loss_avoided"] - removed[
        "winner_pnl_sacrificed"
    ]
    removed["removed_trade_was_winner"] = removed["winner_pnl_sacrificed"] > 0
    removed["score_dir"] = str(score_dir)
    period_start, period_end = _period_from_manifest(manifest, score_dir)
    removed["period_start"] = period_start
    removed["period_end"] = period_end
    return removed


def _summarize_removed(rows: pd.DataFrame, prefix: str) -> dict[str, Any]:
    if rows.empty:
        return {
            f"{prefix}_removed_count": 0,
            f"{prefix}_loss_avoided": 0.0,
            f"{prefix}_winner_pnl_sacrificed": 0.0,
            f"{prefix}_defensive_success": 0.0,
            f"{prefix}_winner_removed_count": 0,
            f"{prefix}_full_sl_count": 0,
            f"{prefix}_timeout_count": 0,
        }
    exit_reason = rows.get("simple_policy_exit_reason", pd.Series(index=rows.index, dtype=object))
    return {
        f"{prefix}_removed_count": int(len(rows)),
        f"{prefix}_loss_avoided": float(rows["loss_avoided"].sum()),
        f"{prefix}_winner_pnl_sacrificed": float(rows["winner_pnl_sacrificed"].sum()),
        f"{prefix}_defensive_success": float(rows["defensive_success"].sum()),
        f"{prefix}_winner_removed_count": int(rows["removed_trade_was_winner"].sum()),
        f"{prefix}_full_sl_count": int(exit_reason.eq("full_sl").sum()),
        f"{prefix}_timeout_count": int(exit_reason.eq("timeout").sum()),
    }


def _window_summary(score_dir: Path, manifest: dict[str, Any], removed: pd.DataFrame) -> dict[str, Any]:
    delta = dict(manifest.get("shadow_no_backfill_accepted_delta_summary") or {})
    replay = dict(manifest.get("shadow_no_backfill_replay_summary") or {})
    period_start, period_end = _period_from_manifest(manifest, score_dir)
    direct = removed[_bool_mask(removed, "direct_threshold_suppression")]
    indirect = removed[
        _bool_mask(removed, "indirect_path_or_capacity_suppression")
    ]
    summary: dict[str, Any] = {
        "score_dir": str(score_dir),
        "period_start": period_start,
        "period_end": period_end,
        "selected_arm": manifest.get("selected_arm"),
        "rank_contract": manifest.get("rank_contract"),
        "source_contract_passed": bool(
            ((manifest.get("source_contract_audit") or {}).get("overall_passed"))
        ),
        "baseline_trade_count": int(delta.get("baseline_trade_count", 0)),
        "shadow_trade_count": int(delta.get("shadow_trade_count", 0)),
        "removed_trade_count_manifest": int(delta.get("removed_trade_count", 0)),
        "added_trade_count": int(delta.get("added_trade_count", 0)),
        "baseline_net_pnl": float(delta.get("baseline_net_pnl", np.nan)),
        "shadow_net_pnl": float(delta.get("shadow_net_pnl", np.nan)),
        "total_net_pnl_delta": float(delta.get("total_net_pnl_delta", np.nan)),
        "common_net_pnl_delta": float(delta.get("common_net_pnl_delta", np.nan)),
        "manifest_removed_loss_avoided": float(delta.get("removed_loss_avoided", 0.0)),
        "manifest_removed_winner_pnl_sacrificed": float(
            delta.get("removed_winner_pnl_sacrificed", 0.0)
        ),
        "manifest_defensive_success": float(
            delta.get("accepted_delta_defensive_success", np.nan)
        ),
        "shadow_full_sl_rate": replay.get("full_sl_rate"),
        "shadow_timeout_rate": replay.get("timeout_rate"),
    }
    summary.update(_summarize_removed(removed, "all"))
    summary.update(_summarize_removed(direct, "direct"))
    summary.update(_summarize_removed(indirect, "indirect"))
    summary["direct_share_of_removed"] = (
        float(len(direct) / len(removed)) if len(removed) else 0.0
    )
    summary["indirect_share_of_removed"] = (
        float(len(indirect) / len(removed)) if len(removed) else 0.0
    )
    summary["direct_threshold_counterfactual_positive"] = (
        summary["direct_defensive_success"] > 0.0
    )
    summary["indirect_path_harmed"] = summary["indirect_defensive_success"] < 0.0
    return summary


def _group_removed(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    present = [column for column in group_cols if column in rows.columns]
    if not present:
        return pd.DataFrame()
    grouped = (
        rows.groupby(present, dropna=False)
        .agg(
            removed_count=("net_pnl", "size"),
            direct_removed_count=("direct_threshold_suppression", "sum"),
            indirect_removed_count=("indirect_path_or_capacity_suppression", "sum"),
            winner_removed_count=("removed_trade_was_winner", "sum"),
            loss_avoided=("loss_avoided", "sum"),
            winner_pnl_sacrificed=("winner_pnl_sacrificed", "sum"),
            defensive_success=("defensive_success", "sum"),
            mean_rank=("rank_score_for_threshold", "mean"),
            mean_threshold_raise=("threshold_raise", "mean"),
            mean_risk_severity=("risk_severity", "mean"),
            mean_predicted_action_edge=("predicted_action_edge", "mean"),
        )
        .reset_index()
    )
    grouped["winner_removed_share"] = grouped["winner_removed_count"] / grouped[
        "removed_count"
    ].replace(0, np.nan)
    return grouped


def _state_axis_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    state_cols = [column for column in rows.columns if column.startswith(STATE_PREFIX)]
    records: list[dict[str, Any]] = []
    for column in state_cols:
        values = pd.to_numeric(rows[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        harmful = rows["winner_pnl_sacrificed"] > 0
        direct = _bool_mask(rows, "direct_threshold_suppression")
        records.append(
            {
                "state_axis": column,
                "removed_count_with_value": int(values.notna().sum()),
                "mean_all_removed": float(values.mean()),
                "mean_direct_removed": float(values[direct].mean())
                if values[direct].notna().any()
                else np.nan,
                "mean_indirect_removed": float(values[~direct].mean())
                if values[~direct].notna().any()
                else np.nan,
                "mean_winner_removed": float(values[harmful].mean())
                if values[harmful].notna().any()
                else np.nan,
                "mean_loss_removed": float(values[~harmful].mean())
                if values[~harmful].notna().any()
                else np.nan,
                "winner_minus_loss_mean": (
                    float(values[harmful].mean() - values[~harmful].mean())
                    if values[harmful].notna().any() and values[~harmful].notna().any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(records).sort_values(
        "winner_minus_loss_mean", key=lambda s: s.abs(), ascending=False
    )


def _aggregate_summary(
    score_dirs: list[Path],
    window_rows: pd.DataFrame,
    removed_rows: pd.DataFrame,
) -> dict[str, Any]:
    direct = removed_rows[_bool_mask(removed_rows, "direct_threshold_suppression")]
    indirect = removed_rows[
        _bool_mask(removed_rows, "indirect_path_or_capacity_suppression")
    ]
    total_delta = float(window_rows["total_net_pnl_delta"].sum()) if not window_rows.empty else 0.0
    direct_success = float(direct["defensive_success"].sum()) if not direct.empty else 0.0
    indirect_success = float(indirect["defensive_success"].sum()) if not indirect.empty else 0.0
    failures: list[str] = []
    if total_delta <= 0:
        failures.append("full_replay_total_delta_not_positive")
    if direct_success > 0 and total_delta < 0:
        failures.append("direct_threshold_counterfactual_positive_but_full_replay_negative")
    if indirect_success < 0:
        failures.append("indirect_path_or_capacity_suppression_harmed")
    if _sum_positive(indirect.get("net_pnl", pd.Series(dtype=float))) > 0:
        failures.append("indirect_suppression_removed_winners")
    if _sum_positive(removed_rows.get("net_pnl", pd.Series(dtype=float))) >= _sum_negative_as_positive(
        removed_rows.get("net_pnl", pd.Series(dtype=float))
    ):
        failures.append("winner_sacrifice_not_below_loss_avoided")
    if not window_rows.empty and (window_rows["total_net_pnl_delta"] > 0).mean() <= 0.5:
        failures.append("positive_window_share_not_above_chance")

    summary = {
        "generated_by": "diagnose_market_state_no_backfill_failures",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "score_dirs": [str(path) for path in score_dirs],
        "window_count": int(len(window_rows)),
        "removed_trade_count": int(len(removed_rows)),
        "direct_removed_count": int(len(direct)),
        "indirect_removed_count": int(len(indirect)),
        "positive_window_share": float((window_rows["total_net_pnl_delta"] > 0).mean())
        if not window_rows.empty
        else 0.0,
        "total_net_pnl_delta": total_delta,
        "median_window_net_pnl_delta": float(window_rows["total_net_pnl_delta"].median())
        if not window_rows.empty
        else 0.0,
        "q25_window_net_pnl_delta": float(window_rows["total_net_pnl_delta"].quantile(0.25))
        if not window_rows.empty
        else 0.0,
        "removed_loss_avoided": float(removed_rows["loss_avoided"].sum())
        if not removed_rows.empty
        else 0.0,
        "removed_winner_pnl_sacrificed": float(
            removed_rows["winner_pnl_sacrificed"].sum()
        )
        if not removed_rows.empty
        else 0.0,
        "removed_defensive_success": float(removed_rows["defensive_success"].sum())
        if not removed_rows.empty
        else 0.0,
        "direct_loss_avoided": float(direct["loss_avoided"].sum()) if not direct.empty else 0.0,
        "direct_winner_pnl_sacrificed": float(direct["winner_pnl_sacrificed"].sum())
        if not direct.empty
        else 0.0,
        "direct_defensive_success": direct_success,
        "indirect_loss_avoided": float(indirect["loss_avoided"].sum())
        if not indirect.empty
        else 0.0,
        "indirect_winner_pnl_sacrificed": float(
            indirect["winner_pnl_sacrificed"].sum()
        )
        if not indirect.empty
        else 0.0,
        "indirect_defensive_success": indirect_success,
        "failure_modes": failures,
        "promotion_safe_subset_found": False,
        "interpretation": (
            "No-backfill threshold overlay remains unsafe: accepted-trade removals "
            "do not produce positive full replay economics, and indirect path/capacity "
            "effects can remove profitable winners."
        ),
    }
    by_head = _group_removed(removed_rows, ["head"])
    if not by_head.empty:
        safe_heads = by_head[
            (by_head["defensive_success"] > 0)
            & (by_head["indirect_removed_count"] == 0)
            & (by_head["winner_pnl_sacrificed"] <= by_head["loss_avoided"])
        ]
        summary["promotion_safe_subset_found"] = bool(not safe_heads.empty)
        summary["safe_head_candidates"] = safe_heads["head"].astype(str).tolist()
    else:
        summary["safe_head_candidates"] = []
    return summary


def build_diagnostics(score_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_removed: list[pd.DataFrame] = []
    window_summaries: list[dict[str, Any]] = []
    for score_dir in score_dirs:
        manifest = _read_json(score_dir / "manifest.json")
        removed = _prepare_removed_rows(score_dir, manifest)
        all_removed.append(removed)
        window_summaries.append(_window_summary(score_dir, manifest, removed))

    removed_rows = pd.concat(all_removed, ignore_index=True) if all_removed else pd.DataFrame()
    window_rows = pd.DataFrame(window_summaries)
    by_head = _group_removed(removed_rows, ["head"])
    by_strategy = _group_removed(removed_rows, ["head", "strategy_id"])
    by_reason = _group_removed(removed_rows, ["controller_reason"])
    state_axes = _state_axis_summary(removed_rows)
    summary = _aggregate_summary(score_dirs, window_rows, removed_rows)

    outputs = {
        "window_diagnostics_csv": output_dir / "no_backfill_failure_window_diagnostics.csv",
        "removed_trade_diagnostics_csv": output_dir / "no_backfill_removed_trade_diagnostics.csv",
        "by_head_csv": output_dir / "no_backfill_failure_by_head.csv",
        "by_strategy_csv": output_dir / "no_backfill_failure_by_strategy.csv",
        "by_controller_reason_csv": output_dir / "no_backfill_failure_by_controller_reason.csv",
        "state_axis_csv": output_dir / "no_backfill_failure_state_axis_diagnostics.csv",
        "summary_json": output_dir / "no_backfill_failure_diagnostics_summary.json",
        "report_md": output_dir / "no_backfill_failure_diagnostics_report.md",
    }
    window_rows.to_csv(outputs["window_diagnostics_csv"], index=False)
    removed_rows.to_csv(outputs["removed_trade_diagnostics_csv"], index=False)
    by_head.to_csv(outputs["by_head_csv"], index=False)
    by_strategy.to_csv(outputs["by_strategy_csv"], index=False)
    by_reason.to_csv(outputs["by_controller_reason_csv"], index=False)
    state_axes.to_csv(outputs["state_axis_csv"], index=False)
    summary["outputs"] = {key: str(path) for key, path in outputs.items()}
    outputs["summary_json"].write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    outputs["report_md"].write_text(_render_report(summary, window_rows, by_head), encoding="utf-8")
    return summary


def _render_report(summary: dict[str, Any], window_rows: pd.DataFrame, by_head: pd.DataFrame) -> str:
    lines = [
        "# Market-State No-Backfill Failure Diagnostics",
        "",
        f"Generated at: `{summary['generated_at_utc']}`",
        "",
        "## Aggregate",
        "",
        f"- Windows: `{summary['window_count']}`",
        f"- Total net PnL delta: `{summary['total_net_pnl_delta']}`",
        f"- Median window delta: `{summary['median_window_net_pnl_delta']}`",
        f"- Q25 window delta: `{summary['q25_window_net_pnl_delta']}`",
        f"- Positive window share: `{summary['positive_window_share']}`",
        f"- Removed trades: `{summary['removed_trade_count']}`",
        f"- Direct threshold removals: `{summary['direct_removed_count']}`",
        f"- Indirect path/capacity removals: `{summary['indirect_removed_count']}`",
        f"- Loss avoided: `{summary['removed_loss_avoided']}`",
        f"- Winner PnL sacrificed: `{summary['removed_winner_pnl_sacrificed']}`",
        f"- Defensive success: `{summary['removed_defensive_success']}`",
        f"- Direct defensive success: `{summary['direct_defensive_success']}`",
        f"- Indirect defensive success: `{summary['indirect_defensive_success']}`",
        "",
        "## Failure Modes",
        "",
    ]
    for failure in summary.get("failure_modes", []):
        lines.append(f"- `{failure}`")
    lines.extend(
        [
            "",
            "## Window Deltas",
            "",
        ]
    )
    if window_rows.empty:
        lines.append("No windows available.")
    else:
        for row in window_rows.to_dict("records"):
            lines.append(
                "- "
                f"`{row.get('period_start')}` -> `{row.get('period_end')}`: "
                f"delta `{row.get('total_net_pnl_delta')}`, "
                f"removed `{row.get('all_removed_count')}`, "
                f"direct `{row.get('direct_removed_count')}`, "
                f"indirect `{row.get('indirect_removed_count')}`, "
                f"defensive `{row.get('all_defensive_success')}`"
            )
    lines.extend(["", "## By Head", ""])
    if by_head.empty:
        lines.append("No removed trades by head.")
    else:
        for row in by_head.to_dict("records"):
            lines.append(
                "- "
                f"`{row.get('head')}`: removed `{row.get('removed_count')}`, "
                f"direct `{row.get('direct_removed_count')}`, "
                f"indirect `{row.get('indirect_removed_count')}`, "
                f"loss avoided `{row.get('loss_avoided')}`, "
                f"winner sacrificed `{row.get('winner_pnl_sacrificed')}`, "
                f"defensive `{row.get('defensive_success')}`"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary["interpretation"],
            "",
        ]
    )
    if summary.get("promotion_safe_subset_found"):
        lines.append(
            "A narrow safe subset was found in removed-trade accounting, but it still needs full path replay before promotion."
        )
    else:
        lines.append(
            "No safe promotion subset was found under the current accepted-trade removal evidence."
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose direct and indirect failures in no-backfill threshold overlay scores."
    )
    parser.add_argument("--score-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_diagnostics(args.score_dir, args.output_dir)
    print(json.dumps(_json_safe(summary), indent=2))


if __name__ == "__main__":
    main()
