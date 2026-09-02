#!/usr/bin/env python3
"""Causal score-combination ablations for saved temporal residual predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _top_mask(frame: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    result = np.zeros(len(frame), dtype=bool)
    months = frame["month"].astype(str).to_numpy()
    budget_source = pd.to_numeric(frame["policy_parent_rank"], errors="coerce").fillna(0.0).to_numpy()
    for month in sorted(set(months)):
        positions = np.flatnonzero(months == month)
        budget = int(np.sum(budget_source[positions] >= 0.90))
        if budget <= 0:
            continue
        local = np.nan_to_num(score[positions], nan=-np.inf)
        chosen = np.argpartition(local, -budget)[-budget:]
        result[positions[chosen]] = True
    return result


def _stats(frame: pd.DataFrame, selected: np.ndarray) -> dict[str, float]:
    work = frame.loc[selected].copy()
    work["week"] = (
        work["__ts__"].dt.floor("D")
        - pd.to_timedelta(work["__ts__"].dt.weekday, unit="D")
    ).dt.strftime("%Y-%m-%d")
    return {
        "selected_rows": int(len(work)),
        "mean_ev": float(work["ev_after_1pct"].mean()),
        "worst_week": float(work.groupby("week", observed=True)["ev_after_1pct"].mean().min()),
        "worst_month": float(work.groupby("month", observed=True)["ev_after_1pct"].mean().min()),
    }


def _causal_residual_z(history: pd.DataFrame, score: pd.DataFrame) -> np.ndarray:
    result = np.zeros(len(score), dtype=np.float32)
    for side in ("long", "short"):
        side_hist = history.loc[history["side_name"].eq(side)]
        side_score = score["side_name"].eq(side).to_numpy()
        if not side_score.any() or side_hist.empty:
            continue
        global_values = side_hist["raw_temporal_signal"].to_numpy(dtype=np.float32)
        global_med = float(np.nanmedian(global_values))
        global_scale = float(np.nanquantile(global_values, 0.75) - np.nanquantile(global_values, 0.25))
        global_scale = max(global_scale, 1e-4)
        for archetype, positions in score.loc[side_score].groupby("archetype_policy_key", observed=True).groups.items():
            destination = np.asarray(list(positions), dtype=np.int64)
            local = side_hist.loc[side_hist["archetype_policy_key"].eq(archetype), "raw_temporal_signal"].to_numpy(dtype=np.float32)
            if len(local) >= 500:
                med = float(np.nanmedian(local))
                scale = float(np.nanquantile(local, 0.75) - np.nanquantile(local, 0.25))
                scale = max(scale, global_scale * 0.25, 1e-4)
            else:
                med, scale = global_med, global_scale
            result[destination] = np.clip((score.loc[destination, "raw_temporal_signal"].to_numpy(dtype=np.float32) - med) / scale, -4.0, 4.0)
    return result


def _timestamp_rank(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    work = pd.Series(values, index=frame.index)
    ranked = work.groupby(frame["__ts__"], observed=True).rank(method="average", pct=True)
    return (ranked.fillna(0.5).to_numpy(dtype=np.float32) - 0.5) * 2.0


def _timestamp_side_rank(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    work = pd.Series(values, index=frame.index)
    ranked = work.groupby(
        [frame["__ts__"], frame["side_name"]], observed=True
    ).rank(method="average", pct=True)
    return (ranked.fillna(0.5).to_numpy(dtype=np.float32) - 0.5) * 2.0


def _score(frame: pd.DataFrame, arm: str, alpha: float, z: np.ndarray) -> np.ndarray:
    parent = frame["sparse_parent_rank_score"].to_numpy(dtype=np.float32)
    quality = frame["temporal_quality"].to_numpy(dtype=np.float32)
    complete = frame["temporal_sequence_complete"].astype(bool).to_numpy(dtype=np.float32)
    weighted = z * quality * complete
    if arm == "rank_overlay":
        return parent + alpha * weighted
    if arm == "cutoff_tiebreak":
        active = (parent >= 0.80) & (parent <= 0.995)
        return parent + alpha * weighted * active
    if arm == "negative_veto":
        return parent - alpha * np.maximum(-weighted, 0.0)
    if arm == "timestamp_rank_overlay":
        return parent + alpha * _timestamp_rank(frame, weighted) * quality * complete
    if arm.startswith("side_rank_overlay"):
        local = _timestamp_side_rank(frame, weighted) * quality * complete
        if arm.endswith("_long_only"):
            local *= frame["side_name"].eq("long").to_numpy(dtype=np.float32)
        elif arm.endswith("_short_only"):
            local *= frame["side_name"].eq("short").to_numpy(dtype=np.float32)
        return parent + alpha * local
    raise ValueError(arm)


def _choose_alpha(history: pd.DataFrame, arm: str) -> tuple[float, dict[str, float]]:
    parent = history["sparse_parent_rank_score"].to_numpy(dtype=np.float32)
    baseline = _stats(history, _top_mask(history, parent))
    z = _causal_residual_z(history, history)
    best_alpha, best = 0.0, baseline
    best_objective = 0.0
    grid = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08)
    for alpha in grid:
        candidate = _stats(history, _top_mask(history, _score(history, arm, alpha, z)))
        gain = candidate["mean_ev"] - baseline["mean_ev"]
        stable = (
            candidate["worst_week"] >= baseline["worst_week"] - 1e-9
            and candidate["worst_month"] >= baseline["worst_month"] - 1e-9
        )
        objective = gain if stable else -np.inf
        if objective > best_objective:
            best_alpha, best, best_objective = alpha, candidate, objective
    return float(best_alpha), {**best, "objective_gain": float(best_objective)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "__ts__", "__symbol__", "side_name", "archetype_policy_key",
        "policy_parent_rank", "ev_after_1pct", "clean_exec",
        "full_path_bad_mae_1r", "timeout", "sparse_parent_rank_score",
        "temporal_correction", "temporal_quality", "temporal_sequence_complete",
    ]
    frame = pd.read_parquet(args.predictions, columns=columns)
    frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
    frame["month"] = frame["__ts__"].dt.strftime("%Y-%m")
    frame["raw_temporal_signal"] = (
        pd.to_numeric(frame["temporal_correction"], errors="coerce").fillna(0.0)
        * pd.to_numeric(frame["temporal_quality"], errors="coerce").fillna(0.0)
    ).astype(np.float32)
    arms = (
        "rank_overlay", "cutoff_tiebreak", "negative_veto", "timestamp_rank_overlay",
        "side_rank_overlay", "side_rank_overlay_long_only", "side_rank_overlay_short_only",
    )
    summary: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    parent_selected = _top_mask(frame, frame["sparse_parent_rank_score"].to_numpy(dtype=np.float32))
    parent_stats = _stats(frame, parent_selected)
    for arm in arms:
        selected = np.zeros(len(frame), dtype=bool)
        for month in sorted(frame["month"].unique()):
            current_index = np.flatnonzero(frame["month"].eq(month).to_numpy())
            current = frame.iloc[current_index].copy().reset_index(drop=True)
            history = frame.loc[frame["month"].lt(month)].copy().reset_index(drop=True)
            if history.empty:
                alpha, policy = 0.0, {"objective_gain": 0.0}
                z = np.zeros(len(current), dtype=np.float32)
            else:
                alpha, policy = _choose_alpha(history, arm)
                z = _causal_residual_z(history, current)
            current_score = _score(current, arm, alpha, z)
            selected[current_index] = _top_mask(current, current_score)
            folds.append({"arm": arm, "month": month, "alpha": alpha, **policy})
        arm_stats = _stats(frame, selected)
        summary.append({
            "arm": arm, **arm_stats,
            "delta_mean_ev": arm_stats["mean_ev"] - parent_stats["mean_ev"],
            "delta_worst_week": arm_stats["worst_week"] - parent_stats["worst_week"],
            "delta_worst_month": arm_stats["worst_month"] - parent_stats["worst_month"],
            "swapped_rows": int(np.sum(selected ^ parent_selected)),
            "promotable": bool(
                arm_stats["mean_ev"] > parent_stats["mean_ev"]
                and arm_stats["worst_week"] >= parent_stats["worst_week"]
                and arm_stats["worst_month"] >= parent_stats["worst_month"]
            ),
        })
        frame[f"selected_{arm}"] = selected
    pd.DataFrame(summary).to_csv(args.output_dir / "summary.csv", index=False)
    pd.DataFrame(folds).to_csv(args.output_dir / "fold_policy.csv", index=False)
    frame.to_parquet(args.output_dir / "scored_ablation.parquet", index=False)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "source": str(args.predictions),
        "parent": "immutable sparse_parent_rank_score",
        "activity": "same monthly policy_parent_rank >= 0.90 count",
        "parameter_selection": "completed prior OOS months only; alpha=0 included",
        "arms": list(arms),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
