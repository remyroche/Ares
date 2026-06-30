#!/usr/bin/env python3
"""Audit whether market-state priority schedules can actually alter the auction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_market_state_priority_shadow_promotion import resolve_arm_selector  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_priority_actionability_audit_20260626")


ORDERING_CAPACITY_REASONS = {
    "max_new_entries_per_bar_reached",
    "max_concurrent_positions_reached",
    "max_concurrent_per_side_reached",
    "max_capital_allocation_reached",
}
STRATEGY_CAPACITY_REASONS = {
    "max_new_entries_per_strategy_per_bar_reached",
    "max_concurrent_per_strategy_reached",
}
SYMBOL_PATH_REASONS = {
    "symbol_already_open",
    "symbol_in_cooldown",
}
EXECUTION_FILTER_REASONS = {
    "insufficient_liquidity_capacity",
    "price_gap_too_large",
    "position_size_too_small",
    "insufficient_wallet_capacity",
}


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


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or path.is_dir():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_arm(metrics: pd.DataFrame, arm_contains: str) -> str:
    if metrics.empty or "arm" not in metrics.columns:
        raise ValueError("cap-sweep metrics are missing arm rows")
    mask = metrics["arm"].astype(str).str.contains(str(arm_contains), regex=False, na=False)
    selected = metrics.loc[mask, "arm"].dropna().astype(str)
    if selected.empty:
        raise ValueError(f"no cap-sweep arm contains {arm_contains!r}")
    return str(selected.iloc[0])


def _normalise_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _candidate_enrichment(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    if "candidate_index" not in out.columns:
        out.insert(0, "candidate_index", np.arange(len(out), dtype=np.int64))
    keep = [
        col
        for col in [
            "candidate_index",
            "head",
            "net_return",
            "gross_return",
            "fees_bps",
            "simple_policy_exit_reason",
            "portfolio_priority_adjustment",
        ]
        if col in out.columns
    ]
    return out[keep].copy()


def _prepare_decisions(decisions: pd.DataFrame, candidates: pd.DataFrame, *, suffix: str) -> pd.DataFrame:
    required = {
        "candidate_index",
        "timestamp",
        "portfolio_priority",
        "accepted",
        "rejection_reason",
    }
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"decisions missing required columns: {missing}")
    out = decisions.copy()
    out["timestamp"] = _normalise_timestamp(out["timestamp"])
    out["candidate_index"] = pd.to_numeric(out["candidate_index"], errors="raise").astype(np.int64)
    out["accepted"] = out["accepted"].astype(bool)
    enrich = _candidate_enrichment(candidates)
    out = out.merge(enrich, on="candidate_index", how="left", suffixes=("", "_candidate"))
    rename = {
        "portfolio_priority": f"portfolio_priority_{suffix}",
        "accepted": f"accepted_{suffix}",
        "rejection_reason": f"rejection_reason_{suffix}",
    }
    for col in ["position_size", "open_positions_before", "open_positions_after"]:
        if col in out.columns:
            rename[col] = f"{col}_{suffix}"
    return out.rename(columns=rename)


def _accepted_index_set(frame: pd.DataFrame, column: str) -> set[int]:
    if frame.empty:
        return set()
    mask = frame[column].fillna(False).astype(bool)
    return set(
        pd.to_numeric(frame.loc[mask, "candidate_index"], errors="coerce")
        .dropna()
        .astype(np.int64)
        .tolist()
    )


def _frontier_stats(merged: pd.DataFrame) -> dict[str, Any]:
    if merged.empty:
        return {
            "frontier_gap_static": np.nan,
            "frontier_gap_after_adjustment_on_static_set": np.nan,
            "frontier_crossed_on_static_set": False,
            "best_rejected_reason_static": None,
        }
    accepted = merged["accepted_static"].fillna(False).astype(bool)
    rejected = ~accepted
    if not bool(accepted.any()) or not bool(rejected.any()):
        return {
            "frontier_gap_static": np.nan,
            "frontier_gap_after_adjustment_on_static_set": np.nan,
            "frontier_crossed_on_static_set": False,
            "best_rejected_reason_static": None,
        }
    static_acc = pd.to_numeric(merged.loc[accepted, "portfolio_priority_static"], errors="coerce")
    static_rej = pd.to_numeric(merged.loc[rejected, "portfolio_priority_static"], errors="coerce")
    shadow_acc = pd.to_numeric(merged.loc[accepted, "portfolio_priority_shadow"], errors="coerce")
    shadow_rej = pd.to_numeric(merged.loc[rejected, "portfolio_priority_shadow"], errors="coerce")
    if not static_acc.notna().any() or not static_rej.notna().any():
        return {
            "frontier_gap_static": np.nan,
            "frontier_gap_after_adjustment_on_static_set": np.nan,
            "frontier_crossed_on_static_set": False,
            "best_rejected_reason_static": None,
        }
    best_rej_idx = static_rej.idxmax()
    static_gap = float(static_acc.min() - static_rej.max())
    adjusted_gap = np.nan
    crossed = False
    if shadow_acc.notna().any() and shadow_rej.notna().any():
        adjusted_gap = float(shadow_acc.min() - shadow_rej.max())
        crossed = bool(adjusted_gap < 0.0)
    return {
        "frontier_gap_static": static_gap,
        "frontier_gap_after_adjustment_on_static_set": adjusted_gap,
        "frontier_crossed_on_static_set": crossed,
        "best_rejected_reason_static": str(merged.loc[best_rej_idx, "rejection_reason_static"]),
    }


def _blocker_category(reason: Any) -> str:
    text = str(reason)
    if text == "accepted":
        return "accepted"
    if text == "below_dynamic_threshold":
        return "threshold_blocked"
    if text in ORDERING_CAPACITY_REASONS:
        return "ordering_capacity_blocked"
    if text in STRATEGY_CAPACITY_REASONS:
        return "strategy_capacity_blocked"
    if text in SYMBOL_PATH_REASONS:
        return "symbol_path_blocked"
    if text in EXECUTION_FILTER_REASONS:
        return "execution_filter_blocked"
    if not text or text == "nan":
        return "unknown"
    return "other_blocked"


def _active_schedule_by_timestamp(schedule: pd.DataFrame) -> dict[pd.Timestamp, set[str]]:
    if schedule.empty:
        return {}
    sched = schedule.copy()
    sched["timestamp"] = _normalise_timestamp(sched["timestamp"])
    sched["portfolio_priority_adjustment"] = pd.to_numeric(
        sched.get("portfolio_priority_adjustment", 0.0),
        errors="coerce",
    ).fillna(0.0)
    active = sched.loc[sched["portfolio_priority_adjustment"].abs() > 1e-12]
    out: dict[pd.Timestamp, set[str]] = {}
    for ts, part in active.groupby("timestamp", dropna=False):
        if pd.isna(ts):
            continue
        out[pd.Timestamp(ts)] = set(part.get("head", pd.Series(dtype=str)).dropna().astype(str).tolist())
    return out


def frontier_blockers(
    *,
    static_decisions: pd.DataFrame,
    shadow_decisions: pd.DataFrame,
    shadow_candidates: pd.DataFrame,
    schedule: pd.DataFrame,
    window_label: str,
    top_n_rejected_per_timestamp: int = 10,
) -> pd.DataFrame:
    """Return the highest-priority rejected rows and their blocking reasons."""
    static_prepared = _prepare_decisions(static_decisions, shadow_candidates, suffix="static")
    shadow_prepared = _prepare_decisions(shadow_decisions, shadow_candidates, suffix="shadow")
    shadow_cols = [
        col
        for col in [
            "candidate_index",
            "portfolio_priority_shadow",
            "accepted_shadow",
            "rejection_reason_shadow",
        ]
        if col in shadow_prepared.columns
    ]
    merged = static_prepared.merge(
        shadow_prepared[shadow_cols],
        on="candidate_index",
        how="left",
        validate="one_to_one",
    )
    active_heads_by_ts = _active_schedule_by_timestamp(schedule)
    rows: list[dict[str, Any]] = []
    n = max(int(top_n_rejected_per_timestamp), 1)
    for ts, part in merged.groupby("timestamp", dropna=False):
        if pd.isna(ts):
            continue
        accepted = part["accepted_static"].fillna(False).astype(bool)
        static_priority = pd.to_numeric(part["portfolio_priority_static"], errors="coerce")
        cutoff = (
            float(static_priority.loc[accepted].min())
            if bool(accepted.any()) and static_priority.loc[accepted].notna().any()
            else np.nan
        )
        rejected = part.loc[~accepted].copy()
        if rejected.empty:
            continue
        rejected["_static_priority"] = pd.to_numeric(rejected["portfolio_priority_static"], errors="coerce")
        rejected = rejected.sort_values("_static_priority", ascending=False).head(n)
        active_heads = active_heads_by_ts.get(pd.Timestamp(ts), set())
        for rank, (_, row) in enumerate(rejected.iterrows(), start=1):
            reason = row.get("rejection_reason_static")
            priority = float(row.get("_static_priority")) if np.isfinite(row.get("_static_priority", np.nan)) else np.nan
            gap = priority - cutoff if np.isfinite(priority) and np.isfinite(cutoff) else np.nan
            head = str(row.get("head", "unknown"))
            shadow_priority = pd.to_numeric(pd.Series([row.get("portfolio_priority_shadow")]), errors="coerce").iloc[0]
            state_adjustment = pd.to_numeric(
                pd.Series([row.get("portfolio_priority_adjustment")]),
                errors="coerce",
            ).fillna(0.0).iloc[0]
            rows.append(
                {
                    "window_label": window_label,
                    "timestamp": pd.Timestamp(ts),
                    "candidate_index": int(row.get("candidate_index")),
                    "head": head,
                    "symbol": row.get("symbol"),
                    "strategy_id": row.get("strategy_id"),
                    "frontier_rejected_rank": int(rank),
                    "static_priority": priority,
                    "shadow_priority": float(shadow_priority) if np.isfinite(shadow_priority) else np.nan,
                    "state_priority_adjustment": float(state_adjustment),
                    "priority_gap_to_static_frontier": float(gap) if np.isfinite(gap) else np.nan,
                    "above_static_frontier": bool(np.isfinite(gap) and gap > 0.0),
                    "static_rejection_reason": str(reason),
                    "shadow_rejection_reason": str(row.get("rejection_reason_shadow")),
                    "blocker_category": _blocker_category(reason),
                    "active_schedule_heads": ",".join(sorted(active_heads)),
                    "row_head_had_active_schedule": bool(head in active_heads),
                    "active_schedule_timestamp": bool(active_heads),
                    "net_return": row.get("net_return"),
                    "simple_policy_exit_reason": row.get("simple_policy_exit_reason"),
                }
            )
    return pd.DataFrame(rows)


def frontier_blocker_summary(blockers: pd.DataFrame) -> pd.DataFrame:
    if blockers.empty:
        return pd.DataFrame()
    work = blockers.copy()
    work["above_static_frontier"] = work["above_static_frontier"].astype(bool)
    work["row_head_had_active_schedule"] = work["row_head_had_active_schedule"].astype(bool)
    work["frontier_top1"] = pd.to_numeric(work["frontier_rejected_rank"], errors="coerce").eq(1)
    group_cols = ["window_label", "head", "blocker_category", "static_rejection_reason"]
    grouped = work.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        rejected_rows=("candidate_index", "size"),
        timestamps=("timestamp", "nunique"),
        top1_rows=("frontier_top1", "sum"),
        above_static_frontier_rows=("above_static_frontier", "sum"),
        active_schedule_rows=("row_head_had_active_schedule", "sum"),
        median_priority_gap=("priority_gap_to_static_frontier", "median"),
        mean_state_adjustment=("state_priority_adjustment", "mean"),
    ).reset_index()
    return summary.sort_values(
        ["window_label", "above_static_frontier_rows", "top1_rows", "rejected_rows"],
        ascending=[True, False, False, False],
    )


def timestamp_actionability(
    *,
    static_decisions: pd.DataFrame,
    shadow_decisions: pd.DataFrame,
    shadow_candidates: pd.DataFrame,
    schedule: pd.DataFrame,
    window_label: str,
) -> pd.DataFrame:
    """Return one row per timestamp describing schedule activity and auction changes."""
    static_prepared = _prepare_decisions(static_decisions, shadow_candidates, suffix="static")
    shadow_prepared = _prepare_decisions(shadow_decisions, shadow_candidates, suffix="shadow")
    merged = static_prepared.merge(
        shadow_prepared[
            [
                col
                for col in shadow_prepared.columns
                if col
                in {
                    "candidate_index",
                    "portfolio_priority_shadow",
                    "accepted_shadow",
                    "rejection_reason_shadow",
                    "position_size_shadow",
                    "open_positions_before_shadow",
                    "open_positions_after_shadow",
                }
            ]
        ],
        on="candidate_index",
        how="outer",
        validate="one_to_one",
    )
    if "timestamp" not in merged.columns:
        raise ValueError("merged decisions lost timestamp column")
    merged["timestamp"] = _normalise_timestamp(merged["timestamp"])
    sched = schedule.copy()
    if sched.empty:
        sched = pd.DataFrame(columns=["timestamp", "head", "portfolio_priority_adjustment"])
    sched["timestamp"] = _normalise_timestamp(sched["timestamp"])
    sched["portfolio_priority_adjustment"] = pd.to_numeric(
        sched.get("portfolio_priority_adjustment", 0.0),
        errors="coerce",
    ).fillna(0.0)
    all_timestamps = pd.Index(sorted(set(merged["timestamp"].dropna()) | set(sched["timestamp"].dropna())))
    rows: list[dict[str, Any]] = []
    for ts in all_timestamps:
        part = merged.loc[merged["timestamp"].eq(ts)].copy()
        sched_part = sched.loc[sched["timestamp"].eq(ts)].copy()
        active_sched = sched_part.loc[sched_part["portfolio_priority_adjustment"].abs() > 1e-12]
        active_heads = set(active_sched.get("head", pd.Series(dtype=str)).dropna().astype(str).tolist())
        static_set = _accepted_index_set(part, "accepted_static")
        shadow_set = _accepted_index_set(part, "accepted_shadow")
        entrants = shadow_set - static_set
        removed = static_set - shadow_set
        frontier = _frontier_stats(part)
        candidate_heads = part.get("head", pd.Series(index=part.index, dtype=object)).astype(str)
        active_candidate_rows = int(candidate_heads.isin(active_heads).sum()) if active_heads else 0
        active_no_action = bool(len(active_sched) > 0 and not entrants and not removed)
        rows.append(
            {
                "window_label": window_label,
                "timestamp": ts,
                "candidate_rows": int(len(part)),
                "active_schedule_rows": int(len(active_sched)),
                "active_schedule_heads": ",".join(sorted(active_heads)),
                "active_candidate_rows": active_candidate_rows,
                "max_abs_priority_adjustment": (
                    float(active_sched["portfolio_priority_adjustment"].abs().max())
                    if not active_sched.empty
                    else 0.0
                ),
                "static_accepted_count": int(len(static_set)),
                "shadow_accepted_count": int(len(shadow_set)),
                "entrants": int(len(entrants)),
                "removed": int(len(removed)),
                "accepted_set_changed": bool(entrants or removed),
                "active_no_action": active_no_action,
                "action_without_active_schedule": bool((entrants or removed) and active_sched.empty),
                **frontier,
            }
        )
    return pd.DataFrame(rows)


def _head_actionability(
    timestamp_rows: pd.DataFrame,
    static_decisions: pd.DataFrame,
    shadow_decisions: pd.DataFrame,
    shadow_candidates: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    window_label: str,
) -> pd.DataFrame:
    static_prepared = _prepare_decisions(static_decisions, shadow_candidates, suffix="static")
    shadow_prepared = _prepare_decisions(shadow_decisions, shadow_candidates, suffix="shadow")
    merged = static_prepared.merge(
        shadow_prepared[["candidate_index", "accepted_shadow"]],
        on="candidate_index",
        how="left",
        validate="one_to_one",
    )
    if "head" not in merged.columns:
        merged["head"] = "unknown"
    merged["head"] = merged["head"].fillna("unknown").astype(str)
    sched = schedule.copy()
    if not sched.empty:
        sched["portfolio_priority_adjustment"] = pd.to_numeric(
            sched.get("portfolio_priority_adjustment", 0.0),
            errors="coerce",
        ).fillna(0.0)
    rows: list[dict[str, Any]] = []
    for head, part in merged.groupby("head", dropna=False):
        static_set = _accepted_index_set(part, "accepted_static")
        shadow_set = _accepted_index_set(part, "accepted_shadow")
        sched_head = sched.loc[sched.get("head", pd.Series(dtype=str)).astype(str).eq(str(head))] if not sched.empty else pd.DataFrame()
        active_sched = sched_head.loc[sched_head["portfolio_priority_adjustment"].abs() > 1e-12] if not sched_head.empty else pd.DataFrame()
        rows.append(
            {
                "window_label": window_label,
                "head": str(head),
                "candidate_rows": int(len(part)),
                "static_accepted_count": int(len(static_set)),
                "shadow_accepted_count": int(len(shadow_set)),
                "entrants": int(len(shadow_set - static_set)),
                "removed": int(len(static_set - shadow_set)),
                "schedule_rows": int(len(sched_head)),
                "active_schedule_rows": int(len(active_sched)),
                "active_schedule_share": (
                    float(len(active_sched) / len(sched_head)) if len(sched_head) else 0.0
                ),
                "mean_priority_adjustment": (
                    float(sched_head["portfolio_priority_adjustment"].mean())
                    if not sched_head.empty
                    else 0.0
                ),
                "max_abs_priority_adjustment": (
                    float(sched_head["portfolio_priority_adjustment"].abs().max())
                    if not sched_head.empty
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _summarise_window(
    *,
    cap_dir: Path,
    window_label: str,
    arm: str,
    timestamp_rows: pd.DataFrame,
    metrics_row: pd.Series | None,
) -> dict[str, Any]:
    active = timestamp_rows["active_schedule_rows"].gt(0)
    changed = timestamp_rows["accepted_set_changed"].astype(bool)
    crossed = timestamp_rows["frontier_crossed_on_static_set"].astype(bool)
    row: dict[str, Any] = {
        "source_dir": str(cap_dir),
        "window_label": window_label,
        "arm": arm,
        "timestamp_count": int(len(timestamp_rows)),
        "candidate_rows": int(timestamp_rows["candidate_rows"].sum()),
        "active_schedule_timestamps": int(active.sum()),
        "action_timestamps": int(changed.sum()),
        "active_action_timestamps": int((active & changed).sum()),
        "active_no_action_timestamps": int((active & ~changed).sum()),
        "action_without_active_schedule_timestamps": int((~active & changed).sum()),
        "active_action_rate": float((active & changed).sum() / max(active.sum(), 1)),
        "frontier_cross_timestamps": int(crossed.sum()),
        "active_frontier_cross_timestamps": int((active & crossed).sum()),
        "active_frontier_cross_no_action_timestamps": int((active & crossed & ~changed).sum()),
        "entrants": int(timestamp_rows["entrants"].sum()),
        "removed": int(timestamp_rows["removed"].sum()),
        "median_frontier_gap_static": float(
            pd.to_numeric(timestamp_rows["frontier_gap_static"], errors="coerce").median()
        ),
        "median_frontier_gap_after_adjustment_on_static_set": float(
            pd.to_numeric(timestamp_rows["frontier_gap_after_adjustment_on_static_set"], errors="coerce").median()
        ),
        "max_abs_priority_adjustment": float(
            pd.to_numeric(timestamp_rows["max_abs_priority_adjustment"], errors="coerce").max()
        ),
    }
    if metrics_row is not None:
        for col in [
            "delta_net_pnl",
            "net_pnl",
            "accepted_jaccard",
            "full_sl_rate",
            "delta_full_sl_rate",
            "timeout_rate",
            "delta_timeout_rate",
            "defensive_success",
            "coverage",
        ]:
            if col in metrics_row.index:
                row[col] = metrics_row[col]
    return row


def audit_cap_sweep_dir(
    cap_dir: Path,
    *,
    arm_contains: str,
    window_label: str | None = None,
    top_n_rejected_per_timestamp: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_path = cap_dir / "head_priority_cap_sweep_metrics.csv"
    schedule_path = cap_dir / "head_priority_cap_sweep_schedules.parquet"
    static_path = cap_dir / "P0_static_priority_decisions.parquet"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not schedule_path.exists():
        raise FileNotFoundError(schedule_path)
    if not static_path.exists():
        raise FileNotFoundError(static_path)
    metrics = pd.read_csv(metrics_path)
    arm = _select_arm(metrics, arm_contains)
    shadow_path = cap_dir / f"{arm}_decisions.parquet"
    candidates_path = cap_dir / f"{arm}_candidates.parquet"
    if not shadow_path.exists():
        raise FileNotFoundError(shadow_path)
    if not candidates_path.exists():
        raise FileNotFoundError(candidates_path)
    label = str(window_label or cap_dir.parent.name or cap_dir.name)
    static_decisions = pd.read_parquet(static_path)
    shadow_decisions = pd.read_parquet(shadow_path)
    candidates = pd.read_parquet(candidates_path)
    schedule = pd.read_parquet(schedule_path)
    if "priority_arm" in schedule.columns:
        mask = schedule["priority_arm"].astype(str).eq(arm)
        if bool(mask.any()):
            schedule = schedule.loc[mask].copy()
    timestamp_rows = timestamp_actionability(
        static_decisions=static_decisions,
        shadow_decisions=shadow_decisions,
        shadow_candidates=candidates,
        schedule=schedule,
        window_label=label,
    )
    head_rows = _head_actionability(
        timestamp_rows,
        static_decisions,
        shadow_decisions,
        candidates,
        schedule,
        window_label=label,
    )
    blocker_rows = frontier_blockers(
        static_decisions=static_decisions,
        shadow_decisions=shadow_decisions,
        shadow_candidates=candidates,
        schedule=schedule,
        window_label=label,
        top_n_rejected_per_timestamp=top_n_rejected_per_timestamp,
    )
    selected_metric = metrics.loc[metrics["arm"].astype(str).eq(arm)]
    metrics_row = selected_metric.iloc[0] if not selected_metric.empty else None
    summary = pd.DataFrame(
        [
            _summarise_window(
                cap_dir=cap_dir,
                window_label=label,
                arm=arm,
                timestamp_rows=timestamp_rows,
                metrics_row=metrics_row,
            )
        ]
    )
    return summary, timestamp_rows, head_rows, blocker_rows


def _render_report(
    summary: pd.DataFrame,
    by_timestamp: pd.DataFrame,
    by_head: pd.DataFrame,
    blocker_summary: pd.DataFrame,
    blockers: pd.DataFrame,
    manifest: dict[str, Any],
) -> str:
    lines = [
        "# Market-State Priority Actionability Audit",
        "",
        "This audit explains whether the frozen market-state priority schedule actually reaches the global auction frontier.",
        "",
        "## Contract",
        "",
        f"- Arm selector: `{manifest['params']['resolved_arm_contains']}`",
        f"- Arm selector source: `{manifest['params']['arm_selector_source']}`",
        "- Active production remains static T1; this is a shadow-only audit.",
        "- Scores, thresholds, q-fail, HeadHealth and the threshold controller remain unchanged.",
        "",
        "## Window Summary",
        "",
    ]
    summary_cols = [
        "window_label",
        "timestamp_count",
        "active_schedule_timestamps",
        "action_timestamps",
        "active_no_action_timestamps",
        "active_action_rate",
        "entrants",
        "removed",
        "delta_net_pnl",
        "accepted_jaccard",
        "defensive_success",
        "median_frontier_gap_static",
        "median_frontier_gap_after_adjustment_on_static_set",
        "max_abs_priority_adjustment",
    ]
    lines.append(summary[[c for c in summary_cols if c in summary.columns]].to_markdown(index=False))
    lines.extend(["", "## By Head", ""])
    head_cols = [
        "window_label",
        "head",
        "candidate_rows",
        "static_accepted_count",
        "shadow_accepted_count",
        "entrants",
        "removed",
        "active_schedule_rows",
        "active_schedule_share",
        "mean_priority_adjustment",
        "max_abs_priority_adjustment",
    ]
    lines.append(by_head[[c for c in head_cols if c in by_head.columns]].to_markdown(index=False) if not by_head.empty else "_No by-head rows._")
    lines.extend(["", "## Interpretation Fields", ""])
    lines.extend(
        [
            "- `active_no_action_timestamps`: the state schedule was nonzero, but accepted trades did not change.",
            "- `frontier_gap_static`: static accepted frontier priority minus the best static rejected priority; positive values mean rejected rows were below the accepted frontier.",
            "- `frontier_gap_after_adjustment_on_static_set`: the same frontier check after applying the shadow adjustment to the static accepted/rejected sets.",
            "- `frontier_crossed_on_static_set`: the shadow adjustment was large enough to move at least one static rejected row above the weakest static accepted row, before replay state constraints.",
            "",
            "## Most Active Timestamps",
            "",
        ]
    )
    active_view = by_timestamp.loc[by_timestamp["active_schedule_rows"].gt(0)].copy()
    if not active_view.empty:
        active_view = active_view.sort_values(
            ["accepted_set_changed", "max_abs_priority_adjustment", "candidate_rows"],
            ascending=[False, False, False],
        ).head(20)
        ts_cols = [
            "window_label",
            "timestamp",
            "active_schedule_heads",
            "candidate_rows",
            "static_accepted_count",
            "shadow_accepted_count",
            "entrants",
            "removed",
            "frontier_gap_static",
            "frontier_gap_after_adjustment_on_static_set",
            "frontier_crossed_on_static_set",
        ]
        lines.append(active_view[[c for c in ts_cols if c in active_view.columns]].to_markdown(index=False))
    else:
        lines.append("_No active timestamps._")
    lines.extend(["", "## Frontier Blockers", ""])
    if not blocker_summary.empty:
        block_cols = [
            "window_label",
            "head",
            "blocker_category",
            "static_rejection_reason",
            "rejected_rows",
            "timestamps",
            "top1_rows",
            "above_static_frontier_rows",
            "active_schedule_rows",
            "median_priority_gap",
        ]
        lines.append(
            blocker_summary[[c for c in block_cols if c in blocker_summary.columns]]
            .head(40)
            .to_markdown(index=False)
        )
    else:
        lines.append("_No frontier blockers._")
    lines.extend(["", "## Top Rejected Rows Above Static Frontier", ""])
    if not blockers.empty:
        top = blockers.loc[blockers["above_static_frontier"].astype(bool)].copy()
        top = top.sort_values(
            ["row_head_had_active_schedule", "priority_gap_to_static_frontier"],
            ascending=[False, False],
        ).head(30)
        top_cols = [
            "window_label",
            "timestamp",
            "head",
            "frontier_rejected_rank",
            "priority_gap_to_static_frontier",
            "static_rejection_reason",
            "blocker_category",
            "row_head_had_active_schedule",
            "state_priority_adjustment",
            "net_return",
            "simple_policy_exit_reason",
        ]
        lines.append(top[[c for c in top_cols if c in top.columns]].to_markdown(index=False) if not top.empty else "_No rejected rows above the static accepted frontier._")
    else:
        lines.append("_No rejected rows above the static accepted frontier._")
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-sweep-dir", action="append", type=Path, required=True)
    parser.add_argument("--window-label", action="append", default=[])
    parser.add_argument("--arm-contains", default="cap_0p15_zge_0p5")
    parser.add_argument("--use-selected-challenger", action="store_true")
    parser.add_argument("--top-n-rejected-per-timestamp", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cap_dirs = list(args.cap_sweep_dir or [])
    labels = list(args.window_label or [])
    if labels and len(labels) != len(cap_dirs):
        raise ValueError("--window-label must be supplied once per --cap-sweep-dir, or omitted")
    if not labels:
        labels = [path.parent.name for path in cap_dirs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_arm_contains, arm_selector_source = resolve_arm_selector(
        cap_dirs,
        arm_contains=str(args.arm_contains),
        use_selected_challenger=bool(args.use_selected_challenger),
    )

    summary_frames: list[pd.DataFrame] = []
    timestamp_frames: list[pd.DataFrame] = []
    head_frames: list[pd.DataFrame] = []
    blocker_frames: list[pd.DataFrame] = []
    inputs: list[dict[str, Any]] = []
    for cap_dir, label in zip(cap_dirs, labels, strict=True):
        summary, timestamp_rows, head_rows, blocker_rows = audit_cap_sweep_dir(
            cap_dir,
            arm_contains=resolved_arm_contains,
            window_label=str(label),
            top_n_rejected_per_timestamp=int(args.top_n_rejected_per_timestamp),
        )
        summary_frames.append(summary)
        timestamp_frames.append(timestamp_rows)
        head_frames.append(head_rows)
        blocker_frames.append(blocker_rows)
        inputs.append(
            {
                "cap_sweep_dir": str(cap_dir),
                "manifest_sha256": _sha256(cap_dir / "manifest.json"),
                "metrics_sha256": _sha256(cap_dir / "head_priority_cap_sweep_metrics.csv"),
                "schedule_sha256": _sha256(cap_dir / "head_priority_cap_sweep_schedules.parquet"),
            }
        )

    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    by_timestamp = pd.concat(timestamp_frames, ignore_index=True) if timestamp_frames else pd.DataFrame()
    by_head = pd.concat(head_frames, ignore_index=True) if head_frames else pd.DataFrame()
    blockers = pd.concat(blocker_frames, ignore_index=True) if blocker_frames else pd.DataFrame()
    blocker_summary = frontier_blocker_summary(blockers)

    summary.to_csv(args.output_dir / "market_state_priority_actionability_by_window.csv", index=False)
    by_timestamp.to_csv(args.output_dir / "market_state_priority_actionability_by_timestamp.csv", index=False)
    by_head.to_csv(args.output_dir / "market_state_priority_actionability_by_head.csv", index=False)
    blockers.to_csv(args.output_dir / "market_state_priority_frontier_blockers.csv", index=False)
    blocker_summary.to_csv(args.output_dir / "market_state_priority_frontier_blocker_summary.csv", index=False)
    manifest = {
        "generated_by": "audit_market_state_priority_actionability",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "shadow_priority_frontier_actionability_audit",
        "params": {
            "arm_contains": str(args.arm_contains),
            "use_selected_challenger": bool(args.use_selected_challenger),
            "resolved_arm_contains": resolved_arm_contains,
            "arm_selector_source": arm_selector_source,
            "top_n_rejected_per_timestamp": int(args.top_n_rejected_per_timestamp),
        },
        "inputs": inputs,
        "outputs": {
            "summary": str(args.output_dir / "market_state_priority_actionability_by_window.csv"),
            "by_timestamp": str(args.output_dir / "market_state_priority_actionability_by_timestamp.csv"),
            "by_head": str(args.output_dir / "market_state_priority_actionability_by_head.csv"),
            "frontier_blockers": str(args.output_dir / "market_state_priority_frontier_blockers.csv"),
            "frontier_blocker_summary": str(args.output_dir / "market_state_priority_frontier_blocker_summary.csv"),
            "report": str(args.output_dir / "market_state_priority_actionability_report.md"),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "market_state_priority_actionability_report.md").write_text(
        _render_report(summary, by_timestamp, by_head, blocker_summary, blockers, manifest),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe({"output_dir": str(args.output_dir), "summary": summary.to_dict("records")}), indent=2))


if __name__ == "__main__":
    main()
