#!/usr/bin/env python3
"""Summarize a size-action candidate across exact-state replay runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_ARM = "C3el_bagged_safety_c3ed_or_high_value_zero_classifier_broad_union_gate"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _panel_name(run_dir: Path) -> str:
    text = run_dir.name
    if "train720_eval120" in text:
        return "train720_eval120"
    if "train960_eval168" in text:
        return "train960_eval168"
    return text


def _arm_rows(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    if frame.empty or "arm" not in frame.columns:
        return pd.DataFrame()
    return frame.loc[frame["arm"].astype(str).eq(str(arm))].copy()


def _promotion_rows(run_dir: Path, arm: str) -> pd.DataFrame:
    rows = _arm_rows(_read_csv(run_dir / "size_action_promotion_summary.csv"), arm)
    if rows.empty:
        return rows
    rows.insert(0, "panel", _panel_name(run_dir))
    return rows


def _fold_replay_rows(run_dir: Path, arm: str) -> pd.DataFrame:
    rows = _arm_rows(_read_csv(run_dir / "size_action_replay_vs_label_audit.csv"), arm)
    if rows.empty:
        return rows
    rows.insert(0, "panel", _panel_name(run_dir))
    return rows


def _action_quality_rows(run_dir: Path, arm: str) -> pd.DataFrame:
    rows = _arm_rows(_read_csv(run_dir / "size_action_action_quality.csv"), arm)
    if rows.empty:
        return rows
    rows.insert(0, "panel", _panel_name(run_dir))
    return rows


def _schedule_summary(run_dir: Path, arm: str) -> pd.DataFrame:
    schedules = _arm_rows(_read_csv(run_dir / "size_action_schedules.csv"), arm)
    if schedules.empty:
        return pd.DataFrame()
    schedules["multiplier"] = pd.to_numeric(schedules.get("multiplier"), errors="coerce").fillna(1.0)
    active = schedules.loc[schedules["multiplier"] < 1.0].copy()
    if active.empty:
        return pd.DataFrame(
            [
                {
                    "panel": _panel_name(run_dir),
                    "active_schedule_rows": 0,
                    "active_folds": 0,
                    "secondary_rows": 0,
                    "primary_rows": 0,
                    "min_multiplier": 1.0,
                    "median_multiplier": 1.0,
                }
            ]
        )
    source = active.get("union_preferred_source", pd.Series("", index=active.index)).astype(str)
    return pd.DataFrame(
        [
            {
                "panel": _panel_name(run_dir),
                "active_schedule_rows": int(len(active)),
                "active_folds": int(active["fold_id"].nunique()) if "fold_id" in active.columns else 0,
                "secondary_rows": int(source.eq("secondary").sum()),
                "primary_rows": int(source.eq("primary").sum()),
                "min_multiplier": float(active["multiplier"].min()),
                "median_multiplier": float(active["multiplier"].median()),
            }
        ]
    )


def _compact_promotion(promotion: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "panel",
        "folds",
        "median_delta_net_pnl",
        "q25_delta_net_pnl",
        "mean_delta_net_pnl",
        "positive_delta_net_pnl_share",
        "median_delta_cost_pnl",
        "median_exposure_ratio",
        "median_multiplier",
    ]
    return promotion[[c for c in cols if c in promotion.columns]].copy()


def _compact_replay(replay: pd.DataFrame) -> pd.DataFrame:
    if replay.empty:
        return replay
    cols = [
        "panel",
        "fold_id",
        "delta_net_pnl",
        "exposure_ratio",
        "intervention_count",
        "positive_action_count",
        "positive_action_rate",
        "realized_delta_full_J_sum",
        "realized_delta_full_net_pnl_sum",
        "oracle_gain_capture_ratio",
        "sequential_replay_positive",
        "independent_label_positive",
        "sequential_replay_disagrees_with_label",
    ]
    return replay[[c for c in cols if c in replay.columns]].copy()


def _write_markdown(
    out_path: Path,
    arm: str,
    promotion: pd.DataFrame,
    replay: pd.DataFrame,
    quality: pd.DataFrame,
    schedule_summary: pd.DataFrame,
) -> None:
    lines: list[str] = ["# Size-Action Champion Report", "", f"Arm: `{arm}`", ""]
    if not promotion.empty:
        lines.extend(["## Promotion Summary", "", _compact_promotion(promotion).to_markdown(index=False), ""])
    if not schedule_summary.empty:
        lines.extend(["## Schedule Summary", "", schedule_summary.to_markdown(index=False), ""])
    if not replay.empty:
        compact = _compact_replay(replay)
        lines.extend(["## Sequential Replay By Fold", "", compact.to_markdown(index=False), ""])
        agg = (
            compact.groupby("panel", dropna=False)
            .agg(
                folds=("fold_id", "nunique"),
                replay_positive_folds=("sequential_replay_positive", "sum"),
                replay_disagreements=("sequential_replay_disagrees_with_label", "sum"),
                delta_net_pnl_sum=("delta_net_pnl", "sum"),
                median_delta_net_pnl=("delta_net_pnl", "median"),
                min_delta_net_pnl=("delta_net_pnl", "min"),
                intervention_count=("intervention_count", "sum"),
                positive_action_count=("positive_action_count", "sum"),
            )
            .reset_index()
        )
        lines.extend(["## Sequential Replay Aggregate", "", agg.to_markdown(index=False), ""])
    if not quality.empty:
        cols = [
            "panel",
            "fold_id",
            "scheduled_groups",
            "intervention_count",
            "positive_action_count",
            "positive_action_rate",
            "realized_delta_full_J_sum",
            "realized_delta_full_net_pnl_sum",
            "oracle_positive_group_capture_rate",
            "oracle_gain_capture_ratio",
        ]
        lines.extend(["## Exact-State Action Quality", "", quality[[c for c in cols if c in quality.columns]].to_markdown(index=False), ""])
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    promotion = pd.concat([_promotion_rows(run_dir, args.arm) for run_dir in args.run_dirs], ignore_index=True)
    replay = pd.concat([_fold_replay_rows(run_dir, args.arm) for run_dir in args.run_dirs], ignore_index=True)
    quality = pd.concat([_action_quality_rows(run_dir, args.arm) for run_dir in args.run_dirs], ignore_index=True)
    schedule_summary = pd.concat([_schedule_summary(run_dir, args.arm) for run_dir in args.run_dirs], ignore_index=True)

    promotion.to_csv(args.out_dir / "champion_promotion_summary.csv", index=False)
    replay.to_csv(args.out_dir / "champion_fold_replay.csv", index=False)
    quality.to_csv(args.out_dir / "champion_action_quality.csv", index=False)
    schedule_summary.to_csv(args.out_dir / "champion_schedule_summary.csv", index=False)
    _write_markdown(args.out_dir / "size_action_champion_report.md", args.arm, promotion, replay, quality, schedule_summary)
    print(
        {
            "out_dir": str(args.out_dir),
            "arm": str(args.arm),
            "runs": int(len(args.run_dirs)),
            "promotion_rows": int(len(promotion)),
            "replay_rows": int(len(replay)),
            "quality_rows": int(len(quality)),
        }
    )


if __name__ == "__main__":
    main()
