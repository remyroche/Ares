#!/usr/bin/env python3
"""Audit per-head exact-state size-action label support for C3el.

The head-native C3el learner should only be tuned for heads with enough
recurrent positive action labels. This script summarizes exact-state action
panels by head and week so we can separate:

* heads with enough support for production-like sparse gates;
* heads worth keeping as research candidates;
* heads where further replay tuning is likely just overfitting sparse labels.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HEADS = ("long_bars", "long_dist", "short_asset", "short_boll")
HEAD_ALIASES = {"short_bollinger": "short_boll"}
AUDIT_COLUMNS = {
    "timestamp",
    "strategy_id",
    "multiplier",
    "delta_full_J",
    "delta_immediate_J",
    "affected_notional",
    "action_binds",
    "projected_removed_trade_count",
    "projected_removed_trade_share_strategy",
    "projected_removed_trade_share_timestamp",
    "y_intervene",
}


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq

            available = set(pq.ParquetFile(path).schema.names)
            return pd.read_parquet(path, columns=sorted(AUDIT_COLUMNS.intersection(available)))
        except Exception:
            return pd.read_parquet(path)
    return pd.read_csv(path, usecols=lambda col: col in AUDIT_COLUMNS)


def _head_from_strategy(strategy_id: Any) -> str:
    text = str(strategy_id)
    for alias, head in HEAD_ALIASES.items():
        if text == alias or text.startswith(f"{alias}_"):
            return head
    for head in HEADS:
        if text == head or text.startswith(f"{head}_"):
            return head
    return text.split("_", 1)[0] if text else "unknown"


def _week_start(ts: pd.Series) -> pd.Series:
    stamped = pd.to_datetime(ts, utc=True, errors="coerce")
    day = stamped.dt.normalize()
    return day - pd.to_timedelta(stamped.dt.weekday, unit="D")


def _panel_name(path: Path) -> str:
    if path.name.startswith("size_action_exact_panel"):
        return path.parent.name
    return path.stem


def _resolve_panels(paths: list[str], patterns: list[str], *, max_panels: int) -> list[Path]:
    resolved: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.exists():
            resolved.append(path)
    for pattern in patterns:
        for match in glob.glob(pattern, recursive=True):
            path = Path(match)
            if path.exists():
                resolved.append(path)
    unique = sorted(dict.fromkeys(path.resolve() for path in resolved))
    if max_panels > 0:
        unique = unique[: int(max_panels)]
    if not unique:
        raise ValueError("No exact-state action panels were found.")
    return unique


def _normalise_panel(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "strategy_id", "multiplier", "delta_full_J"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"panel missing required columns: {missing}")
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.loc[out["timestamp"].notna()].copy()
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["head"] = out["strategy_id"].map(_head_from_strategy)
    out["multiplier"] = pd.to_numeric(out["multiplier"], errors="coerce")
    out["delta_full_J"] = pd.to_numeric(out["delta_full_J"], errors="coerce").fillna(0.0)
    out["delta_immediate_J"] = (
        pd.to_numeric(out["delta_immediate_J"], errors="coerce").fillna(0.0)
        if "delta_immediate_J" in out.columns
        else 0.0
    )
    out["affected_notional"] = (
        pd.to_numeric(out["affected_notional"], errors="coerce").fillna(0.0)
        if "affected_notional" in out.columns
        else 0.0
    )
    if "action_binds" in out.columns:
        out["action_binds"] = pd.to_numeric(out["action_binds"], errors="coerce").fillna(0.0)
    else:
        removed = (
            pd.to_numeric(out["projected_removed_trade_count"], errors="coerce").fillna(0.0)
            if "projected_removed_trade_count" in out.columns
            else pd.Series(0.0, index=out.index)
        )
        out["action_binds"] = np.where(out["multiplier"].lt(1.0) & removed.gt(0.0), 1.0, 0.0)
    return out.dropna(subset=["multiplier"])


def _group_panel(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (timestamp, strategy_id), group in panel.groupby(["timestamp", "strategy_id"], sort=True):
        nonbase = group.loc[group["multiplier"].lt(1.0)].copy()
        bind = nonbase.loc[pd.to_numeric(nonbase.get("action_binds"), errors="coerce").fillna(0.0).gt(0.0)].copy()
        if bind.empty:
            best_nonbase_delta = 0.0
            best_nonbase_multiplier = 1.0
            worst_nonbase_delta = 0.0
            best_removed_share_strategy = 0.0
            best_removed_share_timestamp = 0.0
            can_bind = False
        else:
            best_idx = bind["delta_full_J"].idxmax()
            best = bind.loc[best_idx]
            best_nonbase_delta = float(best["delta_full_J"])
            best_nonbase_multiplier = float(best["multiplier"])
            worst_nonbase_delta = float(bind["delta_full_J"].min())
            best_removed_share_strategy = float(
                pd.to_numeric(best.get("projected_removed_trade_share_strategy"), errors="coerce")
            )
            best_removed_share_timestamp = float(
                pd.to_numeric(best.get("projected_removed_trade_share_timestamp"), errors="coerce")
            )
            if not np.isfinite(best_removed_share_strategy):
                best_removed_share_strategy = 0.0
            if not np.isfinite(best_removed_share_timestamp):
                best_removed_share_timestamp = 0.0
            can_bind = True
        best_any = group.loc[group["delta_full_J"].idxmax()]
        group_notional = float(pd.to_numeric(group.get("affected_notional"), errors="coerce").fillna(0.0).max())
        if "y_intervene" in group.columns:
            strict_y = float(pd.to_numeric(group["y_intervene"], errors="coerce").fillna(0.0).max())
        else:
            strict_y = 0.0
        rows.append(
            {
                "timestamp": timestamp,
                "strategy_id": strategy_id,
                "head": str(group["head"].iloc[0]),
                "group_rows": int(len(group)),
                "can_bind": bool(can_bind),
                "best_any_delta": float(best_any["delta_full_J"]),
                "best_any_multiplier": float(best_any["multiplier"]),
                "best_nonbase_delta": best_nonbase_delta,
                "best_nonbase_multiplier": best_nonbase_multiplier,
                "worst_nonbase_delta": worst_nonbase_delta,
                "affected_notional": group_notional,
                "best_nonbase_delta_per_notional": best_nonbase_delta / max(group_notional, 1.0),
                "best_removed_share_strategy": best_removed_share_strategy,
                "best_removed_share_timestamp": best_removed_share_timestamp,
                "strict_y_intervene": strict_y,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["week_start"] = _week_start(out["timestamp"])
    for threshold in (25.0, 50.0, 100.0, 150.0, 250.0):
        out[f"positive_nonbase_e{int(threshold)}"] = out["can_bind"] & out["best_nonbase_delta"].gt(threshold)
    out["harmful_nonbase"] = out["can_bind"] & out["worst_nonbase_delta"].lt(0.0)
    return out


def _support_status(pos_groups: int, pos_weeks: int, *, production_groups: int, production_weeks: int) -> str:
    if pos_groups >= production_groups and pos_weeks >= production_weeks:
        return "production_candidate"
    if pos_groups >= max(20, production_groups // 3) and pos_weeks >= max(2, production_weeks - 1):
        return "research_candidate"
    if pos_groups >= 5 and pos_weeks >= 1:
        return "sparse_diagnostic_only"
    return "insufficient_support"


def _summarise_groups(
    groups: pd.DataFrame,
    *,
    panel_name: str,
    panel_path: Path,
    recent_start: pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_head_rows: list[dict[str, Any]] = []
    weekly_rows: list[dict[str, Any]] = []
    for head, head_group in groups.groupby("head", sort=True):
        recent_group = (
            head_group.loc[pd.to_datetime(head_group["timestamp"], utc=True, errors="coerce").ge(recent_start)]
            if recent_start is not None
            else head_group.iloc[0:0]
        )
        row: dict[str, Any] = {
            "panel": panel_name,
            "panel_path": str(panel_path),
            "head": head,
            "timestamp_min": head_group["timestamp"].min(),
            "timestamp_max": head_group["timestamp"].max(),
            "recent_start": recent_start.isoformat() if recent_start is not None else "",
            "groups": int(len(head_group)),
            "weeks": int(head_group["week_start"].nunique()),
            "recent_groups": int(len(recent_group)),
            "recent_weeks": int(recent_group["week_start"].nunique()) if not recent_group.empty else 0,
            "can_bind_groups": int(head_group["can_bind"].sum()),
            "harmful_nonbase_groups": int(head_group["harmful_nonbase"].sum()),
            "strict_y_groups": int(pd.to_numeric(head_group["strict_y_intervene"], errors="coerce").fillna(0.0).gt(0.0).sum()),
            "best_nonbase_delta_mean": float(head_group["best_nonbase_delta"].mean()),
            "best_nonbase_delta_p90": float(head_group["best_nonbase_delta"].quantile(0.90)),
            "best_nonbase_delta_p99": float(head_group["best_nonbase_delta"].quantile(0.99)),
            "worst_nonbase_delta_p05": float(head_group["worst_nonbase_delta"].quantile(0.05)),
        }
        for threshold in (25, 50, 100, 150, 250):
            col = f"positive_nonbase_e{threshold}"
            row[f"{col}_groups"] = int(head_group[col].sum())
            row[f"{col}_weeks"] = int(head_group.loc[head_group[col], "week_start"].nunique())
            row[f"recent_{col}_groups"] = int(recent_group[col].sum()) if not recent_group.empty else 0
            row[f"recent_{col}_weeks"] = int(recent_group.loc[recent_group[col], "week_start"].nunique()) if not recent_group.empty else 0
        by_head_rows.append(row)
        for week, week_group in head_group.groupby("week_start", sort=True):
            weekly_row: dict[str, Any] = {
                "panel": panel_name,
                "head": head,
                "week_start": week,
                "groups": int(len(week_group)),
                "can_bind_groups": int(week_group["can_bind"].sum()),
                "strict_y_groups": int(pd.to_numeric(week_group["strict_y_intervene"], errors="coerce").fillna(0.0).gt(0.0).sum()),
                "best_nonbase_delta_sum_positive": float(week_group["best_nonbase_delta"].clip(lower=0.0).sum()),
                "worst_nonbase_delta_sum_negative": float(week_group["worst_nonbase_delta"].clip(upper=0.0).sum()),
            }
            for threshold in (25, 50, 100, 150, 250):
                weekly_row[f"positive_nonbase_e{threshold}_groups"] = int(week_group[f"positive_nonbase_e{threshold}"].sum())
            weekly_rows.append(weekly_row)
    return pd.DataFrame(by_head_rows), pd.DataFrame(weekly_rows)


def _decision_table(by_panel: pd.DataFrame, *, production_groups: int, production_weeks: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if by_panel.empty:
        return pd.DataFrame()
    for head, group in by_panel.groupby("head", sort=True):
        best = group.sort_values(
            ["positive_nonbase_e50_weeks", "positive_nonbase_e50_groups", "best_nonbase_delta_p90"],
            ascending=[False, False, False],
        ).iloc[0]
        pos_groups = int(best["positive_nonbase_e50_groups"])
        pos_weeks = int(best["positive_nonbase_e50_weeks"])
        group_gap = max(0, int(production_groups) - pos_groups)
        week_gap = max(0, int(production_weeks) - pos_weeks)
        recent_pos_groups = int(best.get("recent_positive_nonbase_e50_groups", 0))
        recent_pos_weeks = int(best.get("recent_positive_nonbase_e50_weeks", 0))
        blockers: list[str] = []
        if group_gap > 0:
            blockers.append(f"need_{group_gap}_more_e50_groups")
        if week_gap > 0:
            blockers.append(f"need_{week_gap}_more_positive_weeks")
        if recent_pos_groups <= 0:
            blockers.append("no_recent_e50_positive_groups")
        support_blocker = "none" if not blockers else ";".join(blockers)
        rows.append(
            {
                "head": head,
                "status": _support_status(
                    pos_groups,
                    pos_weeks,
                    production_groups=production_groups,
                    production_weeks=production_weeks,
                ),
                "best_panel": best["panel"],
                "best_panel_groups": int(best["groups"]),
                "best_panel_weeks": int(best["weeks"]),
                "positive_e50_groups": pos_groups,
                "positive_e50_weeks": pos_weeks,
                "recent_positive_e50_groups": recent_pos_groups,
                "recent_positive_e50_weeks": recent_pos_weeks,
                "production_group_gap": group_gap,
                "production_week_gap": week_gap,
                "support_blocker": support_blocker,
                "positive_e100_groups": int(best["positive_nonbase_e100_groups"]),
                "positive_e150_groups": int(best["positive_nonbase_e150_groups"]),
                "strict_y_groups": int(best["strict_y_groups"]),
                "can_bind_groups": int(best["can_bind_groups"]),
                "harmful_nonbase_groups": int(best["harmful_nonbase_groups"]),
                "best_nonbase_delta_p90": float(best["best_nonbase_delta_p90"]),
                "best_nonbase_delta_p99": float(best["best_nonbase_delta_p99"]),
                "worst_nonbase_delta_p05": float(best["worst_nonbase_delta_p05"]),
            }
        )
    return pd.DataFrame(rows)


def _decision_table_from_groups(
    groups: pd.DataFrame,
    *,
    production_groups: int,
    production_weeks: int,
    recent_start: pd.Timestamp | None,
) -> pd.DataFrame:
    """Build support decisions from de-duplicated exact-state action groups.

    Per-panel summaries are useful diagnostics, but a support decision should
    not ignore newer panels simply because an older larger panel has more rows.
    We therefore de-duplicate by the deployable action group key and take the
    strongest observed non-baseline exact-state delta for that group.
    """
    if groups.empty:
        return pd.DataFrame()
    required = {"head", "timestamp", "strategy_id", "week_start", "best_nonbase_delta", "worst_nonbase_delta"}
    missing = sorted(required.difference(groups.columns))
    if missing:
        raise ValueError(f"group support table missing required columns: {missing}")
    work = groups.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.loc[work["timestamp"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    work["_abs_best"] = pd.to_numeric(work["best_nonbase_delta"], errors="coerce").fillna(0.0)
    work = work.sort_values(["head", "timestamp", "strategy_id", "_abs_best"], ascending=[True, True, True, False])
    work = work.drop_duplicates(["head", "timestamp", "strategy_id"], keep="first").copy()
    if recent_start is not None:
        recent_mask = work["timestamp"].ge(recent_start)
    else:
        recent_mask = pd.Series(False, index=work.index)
    rows: list[dict[str, Any]] = []
    for head, head_group in work.groupby("head", sort=True):
        recent_group = head_group.loc[recent_mask.loc[head_group.index]]
        pos_groups = int(pd.to_numeric(head_group["best_nonbase_delta"], errors="coerce").fillna(0.0).gt(50.0).sum())
        pos_weeks = int(
            head_group.loc[pd.to_numeric(head_group["best_nonbase_delta"], errors="coerce").fillna(0.0).gt(50.0), "week_start"].nunique()
        )
        recent_pos_groups = int(
            pd.to_numeric(recent_group["best_nonbase_delta"], errors="coerce").fillna(0.0).gt(50.0).sum()
            if not recent_group.empty
            else 0
        )
        recent_pos_weeks = int(
            recent_group.loc[
                pd.to_numeric(recent_group["best_nonbase_delta"], errors="coerce").fillna(0.0).gt(50.0), "week_start"
            ].nunique()
            if not recent_group.empty
            else 0
        )
        group_gap = max(0, int(production_groups) - pos_groups)
        week_gap = max(0, int(production_weeks) - pos_weeks)
        blockers: list[str] = []
        if group_gap > 0:
            blockers.append(f"need_{group_gap}_more_e50_groups")
        if week_gap > 0:
            blockers.append(f"need_{week_gap}_more_positive_weeks")
        if recent_pos_groups <= 0:
            blockers.append("no_recent_e50_positive_groups")
        rows.append(
            {
                "head": head,
                "status": _support_status(
                    pos_groups,
                    pos_weeks,
                    production_groups=production_groups,
                    production_weeks=production_weeks,
                ),
                "best_panel": "aggregate_unique_groups",
                "best_panel_groups": int(len(head_group)),
                "best_panel_weeks": int(head_group["week_start"].nunique()),
                "positive_e50_groups": pos_groups,
                "positive_e50_weeks": pos_weeks,
                "recent_positive_e50_groups": recent_pos_groups,
                "recent_positive_e50_weeks": recent_pos_weeks,
                "production_group_gap": group_gap,
                "production_week_gap": week_gap,
                "support_blocker": "none" if not blockers else ";".join(blockers),
                "positive_e100_groups": int(pd.to_numeric(head_group["best_nonbase_delta"], errors="coerce").fillna(0.0).gt(100.0).sum()),
                "positive_e150_groups": int(pd.to_numeric(head_group["best_nonbase_delta"], errors="coerce").fillna(0.0).gt(150.0).sum()),
                "strict_y_groups": int(pd.to_numeric(head_group["strict_y_intervene"], errors="coerce").fillna(0.0).gt(0.0).sum())
                if "strict_y_intervene" in head_group.columns
                else 0,
                "can_bind_groups": int(head_group["can_bind"].sum()) if "can_bind" in head_group.columns else 0,
                "harmful_nonbase_groups": int(head_group["harmful_nonbase"].sum()) if "harmful_nonbase" in head_group.columns else 0,
                "best_nonbase_delta_p90": float(pd.to_numeric(head_group["best_nonbase_delta"], errors="coerce").fillna(0.0).quantile(0.90)),
                "best_nonbase_delta_p99": float(pd.to_numeric(head_group["best_nonbase_delta"], errors="coerce").fillna(0.0).quantile(0.99)),
                "worst_nonbase_delta_p05": float(pd.to_numeric(head_group["worst_nonbase_delta"], errors="coerce").fillna(0.0).quantile(0.05)),
            }
        )
    return pd.DataFrame(rows)


def _write_markdown(out_dir: Path, decisions: pd.DataFrame, by_panel: pd.DataFrame, weekly: pd.DataFrame) -> None:
    lines = [
        "# C3el head action-support audit",
        "",
        "This audit measures exact-state size-action label support by head. It does not prove a policy is profitable; it tells us whether a head has enough recurrent positive action labels to justify more head-specific C3el tuning.",
        "",
        "## Decision table",
        "",
    ]
    if decisions.empty:
        lines.append("No decisions available.")
    else:
        lines.append(decisions.to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Best panel by head", ""])
    if not by_panel.empty:
        cols = [
            "panel",
            "head",
            "groups",
            "weeks",
            "recent_groups",
            "recent_weeks",
            "can_bind_groups",
            "positive_nonbase_e50_groups",
            "positive_nonbase_e50_weeks",
            "recent_positive_nonbase_e50_groups",
            "recent_positive_nonbase_e50_weeks",
            "positive_nonbase_e100_groups",
            "strict_y_groups",
            "best_nonbase_delta_p90",
            "worst_nonbase_delta_p05",
        ]
        compact = by_panel[cols].sort_values(["head", "positive_nonbase_e50_weeks", "positive_nonbase_e50_groups"], ascending=[True, False, False])
        lines.append(compact.groupby("head", as_index=False).head(3).to_markdown(index=False, floatfmt=".4f"))
    lines.extend(["", "## Weekly positive e50 groups by head", ""])
    if not weekly.empty:
        pivot = (
            weekly.pivot_table(
                index=["head", "week_start"],
                values="positive_nonbase_e50_groups",
                aggfunc="max",
                fill_value=0,
            )
            .reset_index()
            .sort_values(["head", "week_start"])
        )
        lines.append(pivot.to_markdown(index=False))
    out_dir.joinpath("summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="*", default=[])
    parser.add_argument("--panel-glob", action="append", default=[])
    parser.add_argument("--max-panels", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--production-groups", type=int, default=60)
    parser.add_argument("--production-weeks", type=int, default=3)
    parser.add_argument("--recent-start", default="2026-06-01T00:00:00+00:00")
    parser.add_argument(
        "--write-group-source",
        action="store_true",
        help="Write the de-duplicated group-source table. This can be large on archive-wide scans.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recent_start = pd.Timestamp(args.recent_start) if str(args.recent_start).strip() else None
    if recent_start is not None:
        recent_start = recent_start.tz_localize("UTC") if recent_start.tzinfo is None else recent_start.tz_convert("UTC")
    panels = _resolve_panels(args.panels, args.panel_glob, max_panels=int(args.max_panels))
    by_panel_parts: list[pd.DataFrame] = []
    weekly_parts: list[pd.DataFrame] = []
    group_parts: list[pd.DataFrame] = []
    manifest_panels: list[dict[str, Any]] = []
    for path in panels:
        panel_name = _panel_name(path)
        frame = _normalise_panel(_read_frame(path))
        groups = _group_panel(frame)
        if not groups.empty:
            groups = groups.copy()
            groups["panel"] = panel_name
            group_parts.append(groups)
        by_panel, weekly = _summarise_groups(groups, panel_name=panel_name, panel_path=path, recent_start=recent_start)
        by_panel_parts.append(by_panel)
        weekly_parts.append(weekly)
        manifest_panels.append(
            {
                "panel": panel_name,
                "path": str(path),
                "rows": int(len(frame)),
                "groups": int(len(groups)),
                "timestamp_min": str(frame["timestamp"].min()) if not frame.empty else "",
                "timestamp_max": str(frame["timestamp"].max()) if not frame.empty else "",
            }
        )
    by_panel_all = pd.concat(by_panel_parts, ignore_index=True) if by_panel_parts else pd.DataFrame()
    weekly_all = pd.concat(weekly_parts, ignore_index=True) if weekly_parts else pd.DataFrame()
    groups_all = pd.concat(group_parts, ignore_index=True) if group_parts else pd.DataFrame()
    decisions = _decision_table_from_groups(
        groups_all,
        production_groups=int(args.production_groups),
        production_weeks=int(args.production_weeks),
        recent_start=recent_start,
    )
    by_panel_all.to_csv(args.out_dir / "head_support_by_panel.csv", index=False)
    weekly_all.to_csv(args.out_dir / "head_support_weekly.csv", index=False)
    if bool(args.write_group_source):
        groups_all.to_csv(args.out_dir / "head_support_groups_dedup_source.csv", index=False)
    decisions.to_csv(args.out_dir / "head_support_decision.csv", index=False)
    _write_markdown(args.out_dir, decisions, by_panel_all, weekly_all)
    args.out_dir.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "generated_by": "audit_c3el_head_action_support",
                "panels": manifest_panels,
                "production_groups": int(args.production_groups),
                "production_weeks": int(args.production_weeks),
                "recent_start": recent_start.isoformat() if recent_start is not None else "",
                "decision_basis": "aggregate_unique_head_timestamp_strategy_groups",
                "wrote_group_source": bool(args.write_group_source),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    print(decisions.to_string(index=False))


if __name__ == "__main__":
    main()
