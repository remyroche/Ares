#!/usr/bin/env python3
"""Test a causal changepoint guard as side x archetype V9 overlay context.

The guard is deliberately non-parametric.  It cannot create a trade or refit
the V9 parent.  For each chronological fold it uses only prior top-10 OOF rows
to match adverse side x archetype timestamps to benign timestamps with similar
parent-rank and candidate-count context.  A local, frozen BOCPD score threshold
is admitted only when it separates those train-only matched controls.  The
evaluation then blocks matching OOS rows without backfilling the selection.

This is a research ablation, not a live policy.  Calendar/outcome labels are
used only in the train-side reliability estimate and OOS reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data_perp/reports/meta_residual_interpretable_rule_overlay_20260714_v18_episode_intervention"
DEFAULT_CHANGEPOINT = ROOT / "data_perp/reports/bayesian_market_state_changepoints_20260714_v5_strict/hourly_changepoint_scores.csv.gz"
DEFAULT_OUTPUT = ROOT / "data_perp/reports/bayesian_changepoint_matched_control_overlay_20260714_v1"
KEYS = ["__ts__", "side_name", "archetype_policy_key"]
FOLDS = (
    ("2025-10-01", "2026-01-01"),
    ("2026-01-01", "2026-04-01"),
    ("2026-04-01", "2026-07-01"),
)
QUANTILES = (0.70, 0.80, 0.85, 0.90, 0.95)


def _utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def _load_predictions(source: Path) -> pd.DataFrame:
    columns = [
        "__ts__", "side_name", "archetype_policy_key", "parent_rank_v9",
        "ev_after_1pct", "clean_exec", "dirty_positive", "full_path_bad_mae_1r",
        "timeout", "top10_adverse_period_target",
    ]
    frames: list[pd.DataFrame] = []
    for name in ("train_oof_predictions.parquet", "oos_predictions.parquet"):
        path = source / name
        available = set(pq.read_schema(path).names)
        frame = pd.read_parquet(path, columns=[column for column in columns if column in available])
        for column in columns:
            if column not in frame:
                frame[column] = np.nan
        frame = frame.reindex(columns=columns)
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
        frames.append(frame.loc[pd.to_numeric(frame["parent_rank_v9"], errors="coerce").ge(0.90)])
    result = pd.concat(frames, ignore_index=True, copy=False)
    return result.sort_values("__ts__", kind="stable").reset_index(drop=True)


def _load_changepoints(path: Path, method: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=["__ts__", "method", "synchronized_break_score", "simultaneous_break_alert"],
        parse_dates=["__ts__"],
    )
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    frame = frame.loc[frame["method"].eq(method)].drop(columns="method")
    return frame.drop_duplicates("__ts__", keep="last")


def _timestamp_panel(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate candidate context without using realized outcomes as inputs."""

    panel = (
        rows.groupby(KEYS, observed=True, sort=True)
        .agg(
            candidate_rows=("parent_rank_v9", "size"),
            parent_rank_mean=("parent_rank_v9", "mean"),
            target=("top10_adverse_period_target", "max"),
        )
        .reset_index()
    )
    panel["hour"] = panel["__ts__"].dt.hour.astype(np.int8)
    panel["day"] = panel["__ts__"].dt.floor("D")
    return panel


def _event_block_count(panel: pd.DataFrame) -> int:
    daily = (
        panel.groupby("day", observed=True, sort=True)["target"].max()
        .astype(bool)
    )
    if daily.empty:
        return 0
    starts = daily & ~daily.shift(1, fill_value=False)
    return int(starts.sum())


def _matched_controls(
    panel: pd.DataFrame,
    *,
    controls_per_positive: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match adverse timestamps to benign context using train-only observables."""

    positives = panel.loc[panel["target"].eq(1)].copy()
    benign = panel.loc[panel["target"].eq(0)].copy()
    if positives.empty or benign.empty:
        return positives, benign.iloc[:0].copy()
    fields = ["parent_rank_mean", "candidate_rows"]
    train = pd.concat([positives[fields], benign[fields]], ignore_index=True, copy=False)
    median = train.median(numeric_only=True).to_numpy(np.float64)
    iqr = (train.quantile(0.75, numeric_only=True) - train.quantile(0.25, numeric_only=True)).to_numpy(np.float64)
    scale = np.maximum(np.nan_to_num(iqr, nan=1.0), 1e-4)
    pos_x = (positives[fields].to_numpy(np.float64) - median) / scale
    benign_x = (benign[fields].to_numpy(np.float64) - median) / scale
    selected: list[pd.DataFrame] = []
    for index, (_, row) in enumerate(positives.iterrows()):
        same_hour = benign["hour"].to_numpy(np.int8) == int(row["hour"])
        candidates = np.flatnonzero(same_hour)
        if len(candidates) < controls_per_positive:
            candidates = np.arange(len(benign), dtype=np.int64)
        distance = np.abs(benign_x[candidates] - pos_x[index]).sum(axis=1)
        take = min(controls_per_positive, len(candidates))
        nearest = candidates[np.argpartition(distance, take - 1)[:take]]
        controls = benign.iloc[nearest].copy()
        controls["matched_event_ts"] = row["__ts__"]
        controls["match_distance"] = distance[np.argpartition(distance, take - 1)[:take]]
        selected.append(controls)
    return positives, pd.concat(selected, ignore_index=True, copy=False) if selected else benign.iloc[:0].copy()


def _select_threshold(positives: pd.DataFrame, controls: pd.DataFrame) -> dict[str, float | int | str]:
    score_pos = pd.to_numeric(positives["synchronized_break_score"], errors="coerce").to_numpy(float)
    score_ctl = pd.to_numeric(controls["synchronized_break_score"], errors="coerce").to_numpy(float)
    score_pos = score_pos[np.isfinite(score_pos)]
    score_ctl = score_ctl[np.isfinite(score_ctl)]
    if len(score_pos) < 12 or len(score_ctl) < 24:
        return {"status": "insufficient_matched_support"}
    candidates: list[dict[str, float]] = []
    for quantile in QUANTILES:
        threshold = float(np.quantile(score_ctl, quantile))
        recall = float(np.mean(score_pos >= threshold))
        fpr = float(np.mean(score_ctl >= threshold))
        lift = recall / max(fpr, 1e-6)
        candidates.append({
            "quantile": quantile,
            "threshold": threshold,
            "train_recall": recall,
            "train_matched_fpr": fpr,
            "train_lift": lift,
            "objective": recall - 0.75 * fpr,
        })
    eligible = [
        item for item in candidates
        if item["train_recall"] >= 0.20
        and item["train_matched_fpr"] <= 0.15
        and item["train_lift"] >= 1.5
    ]
    best = max(candidates, key=lambda item: (item["objective"], item["train_lift"]))
    if not eligible:
        return {"status": "no_train_only_discriminative_threshold", **best}
    return {"status": "accepted", **max(eligible, key=lambda item: (item["objective"], item["train_lift"]))}


def _metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {"selected_rows": 0}
    numeric = lambda name: pd.to_numeric(frame.get(name), errors="coerce")
    event = numeric("top10_adverse_period_target").fillna(0).astype(bool)
    return {
        "selected_rows": int(len(frame)),
        "mean_ev_after_1pct": float(numeric("ev_after_1pct").mean()),
        "sum_ev_after_1pct": float(numeric("ev_after_1pct").sum()),
        "positive_ev_rate": float(numeric("ev_after_1pct").gt(0).mean()),
        "clean_exec_precision": float(numeric("clean_exec").mean()),
        "dirty_positive_rate": float(numeric("dirty_positive").mean()),
        "full_path_bad_mae_rate": float(numeric("full_path_bad_mae_1r").mean()),
        "timeout_rate": float(numeric("timeout").mean()),
        "event_row_share": float(event.mean()),
        "event_mean_ev_after_1pct": float(numeric("ev_after_1pct")[event].mean()) if event.any() else np.nan,
        "normal_mean_ev_after_1pct": float(numeric("ev_after_1pct")[~event].mean()) if (~event).any() else np.nan,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    rows = _load_predictions(args.source)
    changepoints = _load_changepoints(args.changepoints, args.method)
    rows = rows.merge(changepoints, on="__ts__", how="inner", validate="many_to_one")
    panel = _timestamp_panel(rows).merge(changepoints, on="__ts__", how="left", validate="many_to_one")
    fold_reports: list[dict[str, Any]] = []
    group_reports: list[dict[str, Any]] = []
    controls_outputs: list[pd.DataFrame] = []
    actions: list[pd.DataFrame] = []

    for fold, (start_raw, end_raw) in enumerate(FOLDS):
        start, end = _utc(start_raw), _utc(end_raw)
        # The period target can incorporate a two-day adverse sequence, so the
        # latest prior observations are embargoed before each score interval.
        train_cutoff = start - pd.Timedelta(hours=args.embargo_hours)
        train_panel = panel.loc[panel["__ts__"].lt(train_cutoff)]
        score_rows = rows.loc[rows["__ts__"].ge(start) & rows["__ts__"].lt(end)].copy()
        score_panel = panel.loc[panel["__ts__"].ge(start) & panel["__ts__"].lt(end)]
        for (side, archetype), local_train in train_panel.groupby(["side_name", "archetype_policy_key"], observed=True):
            local_score = score_rows.loc[
                score_rows["side_name"].eq(side) & score_rows["archetype_policy_key"].eq(archetype)
            ].copy()
            if local_score.empty:
                continue
            positives, controls = _matched_controls(
                local_train,
                controls_per_positive=args.controls_per_positive,
            )
            decision = _select_threshold(positives, controls)
            blocks = _event_block_count(local_train)
            accepted = decision.get("status") == "accepted" and blocks >= args.min_event_blocks
            if not controls.empty:
                controls_outputs.append(controls.assign(
                    fold=fold, fold_start=start, fold_end=end, side_name=side,
                    archetype_policy_key=archetype,
                ))
            base = _metrics(local_score)
            if accepted:
                blocked = local_score["synchronized_break_score"].ge(float(decision["threshold"]))
            else:
                blocked = np.zeros(len(local_score), dtype=bool)
            guarded = local_score.loc[~blocked].copy()
            guard = _metrics(guarded)
            report = {
                "fold": fold,
                "fold_start": start,
                "fold_end": end,
                "side_name": side,
                "archetype_policy_key": archetype,
                "train_event_blocks": blocks,
                "train_positive_timestamps": int(len(positives)),
                "train_matched_controls": int(len(controls)),
                "overlay_status": "accepted" if accepted else str(decision.get("status")),
                **{f"decision_{key}": value for key, value in decision.items() if key != "status"},
                "baseline_selected_rows": base.get("selected_rows", 0),
                "guarded_selected_rows": guard.get("selected_rows", 0),
                "removed_rows": int(blocked.sum()),
                "activity_retained": float((~blocked).mean()),
                **{f"baseline_{key}": value for key, value in base.items() if key != "selected_rows"},
                **{f"guarded_{key}": value for key, value in guard.items() if key != "selected_rows"},
            }
            report["delta_mean_ev_after_1pct"] = (
                report.get("guarded_mean_ev_after_1pct", np.nan)
                - report.get("baseline_mean_ev_after_1pct", np.nan)
            )
            group_reports.append(report)
            actions.append(local_score.loc[:, ["__ts__", "side_name", "archetype_policy_key", "ev_after_1pct", "clean_exec", "top10_adverse_period_target", "synchronized_break_score"]].assign(
                fold=fold,
                changepoint_guard_accepted=int(accepted),
                changepoint_guard_blocked=blocked.astype(np.int8),
                changepoint_guard_threshold=float(decision.get("threshold", np.nan)),
            ))
        fold_frame = pd.DataFrame([row for row in group_reports if row["fold"] == fold])
        if not fold_frame.empty:
            fold_reports.append({
                "fold": fold,
                "fold_start": start,
                "fold_end": end,
                "groups_scored": int(len(fold_frame)),
                "groups_accepted": int(fold_frame["overlay_status"].eq("accepted").sum()),
                "baseline_rows": int(fold_frame["baseline_selected_rows"].sum()),
                "guarded_rows": int(fold_frame["guarded_selected_rows"].sum()),
                "removed_rows": int(fold_frame["removed_rows"].sum()),
            })

    group_frame = pd.DataFrame(group_reports)
    action_frame = pd.concat(actions, ignore_index=True, copy=False) if actions else pd.DataFrame()
    control_frame = pd.concat(controls_outputs, ignore_index=True, copy=False) if controls_outputs else pd.DataFrame()
    pd.DataFrame(fold_reports).to_csv(args.output / "fold_summary.csv", index=False)
    group_frame.to_csv(args.output / "side_archetype_metrics.csv", index=False)
    control_frame.to_csv(args.output / "matched_benign_controls.csv", index=False)
    action_frame.to_csv(args.output / "oos_guard_actions.csv.gz", index=False, compression="gzip")
    manifest = {
        "purpose": "research-only side x archetype BOCPD guard ablation on frozen V9 candidates",
        "parent": "meta_residual_extreme_local_champion_overlay_ooftrain_tieaware_downonly_20260712_v9",
        "method": args.method,
        "candidate_scope": "identical V9 parent top-10 rows; no backfill after a guard removal",
        "training_contract": f"prior chronological OOF rows only with {args.embargo_hours}h label embargo; matched benign timestamps use parent-rank and candidate-count observables",
        "threshold_contract": "local threshold selected only from prior adverse timestamps and matched benign controls",
        "outcome_inputs_at_inference": False,
        "policy_wiring": False,
        "folds": [f"{start}::{end}" for start, end in FOLDS],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--changepoints", type=Path, default=DEFAULT_CHANGEPOINT)
    parser.add_argument("--method", default="bocpd_h48_sync4")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--controls-per-positive", type=int, default=3)
    parser.add_argument("--min-event-blocks", type=int, default=3)
    parser.add_argument("--embargo-hours", type=int, default=48)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, default=str))
